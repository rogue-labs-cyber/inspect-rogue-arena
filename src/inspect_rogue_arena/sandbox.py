from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeVar, overload

import anyio
import httpx
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    SandboxUnavailableError,
    sandboxenv,
)

from inspect_rogue_arena.client import (
    EvalActivityEntry,
    EvalCatalogEntry,
    EvalChatMessage,
    EvalClaimDenied,
    EvalTrial,
    EvalTrialOutcome,
    EvalTrialTerminalReason,
    RogueArenaApiError,
    RogueArenaClient,
    TrialClosedBeforeReady,
    TrialReadyTimeout,
)
from inspect_rogue_arena.config import EVAL_MAX_EXEC_SECONDS, RogueArenaSandboxConfig

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# How long to wait for a close to score every hop server-side before treating
# a slow close as a client timeout: a base allowance plus a per-hop allowance.
_CLOSE_BASE_SECONDS = 5 * 60
_CLOSE_PER_HOP_SECONDS = 2 * 30 + 60

# Bound on a shielded step of a close (the heartbeat stop): a close that has
# started always finishes, but never holds a cancellation open forever.
_SHIELD_SECONDS = 60

# The heartbeat loop's sleep; a module attribute so tests can drive the
# cadence without real waits.
_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


@dataclass(slots=True)
class _TaskRuntime:
    """One eval run's Rogue Arena job for one sandbox config, and its open trials.

    Inspect calls ``task_init("startup")`` once per distinct sandbox config and
    ``task_cleanup("shutdown")`` at the end of the run; only ``sample_init``
    sees the real task name. So every task sharing a config shares this job,
    and each claim carries its own ``inspectTaskId`` for attribution.
    """

    config: RogueArenaSandboxConfig
    client: RogueArenaClient
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry: EvalCatalogEntry | None = None
    job_id: str | None = None
    attempt_number: int = 0
    environments: set[RogueArenaSandboxEnvironment] = field(default_factory=set)
    prepare_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    attempt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    trial_slots: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.trial_slots = asyncio.Semaphore(self.config.concurrency)

    async def next_attempt_key(self, task_name: str, sample_id: str) -> str:
        async with self.attempt_lock:
            self.attempt_number += 1
            raw = (
                f"{self.run_id}:{self.attempt_number}:{task_name}:{sample_id}"
            ).encode()
            return f"inspect_{hashlib.sha256(raw).hexdigest()}"


def _as_config(
    config: SandboxEnvironmentConfigType | None,
) -> RogueArenaSandboxConfig:
    if config is None or isinstance(config, RogueArenaSandboxConfig):
        return RogueArenaSandboxConfig.from_sources(config)
    if isinstance(config, Mapping):
        return RogueArenaSandboxConfig.from_sources(config)
    raise TypeError(
        "rogue-arena sandbox config must be a mapping or RogueArenaSandboxConfig"
    )


def _runtime_key(config: RogueArenaSandboxConfig, client: RogueArenaClient) -> str:
    # No task name: Inspect's task hooks receive "startup"/"shutdown", not it.
    public_config = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(
        (
            f"{client.base_url}\0{client.credential_fingerprint}\0{public_config}"
        ).encode()
    ).hexdigest()


def _reserved_str(metadata: Mapping[str, object], name: str) -> str | None:
    value = metadata.get(name)
    return str(value) if value is not None else None


def _reserved_int(metadata: Mapping[str, object], name: str) -> int | None:
    value = metadata.get(name)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _require_sample_id(metadata: Mapping[str, object]) -> str:
    sample_id = _reserved_str(metadata, "__sample_id__")
    if sample_id is None:
        raise RuntimeError(
            "Inspect did not provide reserved metadata key '__sample_id__'; "
            "upgrade inspect_ai rather than supplying a user metadata sample_id"
        )
    return sample_id


def select_eval(
    entries: tuple[EvalCatalogEntry, ...], config: RogueArenaSandboxConfig
) -> EvalCatalogEntry:
    """Pick the eval ``config.eval`` names from ``list_evals()``.

    An environment id matches first, then a unique name. An unset selector, an
    unknown one, or a name shared by several evals raises ``RuntimeError``.
    """
    selector = config.eval
    if selector is None:
        raise RuntimeError(
            "No Rogue Arena eval selected: set the sandbox config `eval` (or "
            "ROGUE_ARENA_EVAL) to an environment id or name from "
            "`rogue-arena-inspect evals`"
        )
    for entry in entries:
        if entry.environment_id == selector:
            return entry
    named = [entry for entry in entries if entry.name == selector]
    if len(named) == 1:
        return named[0]
    if named:
        ids = ", ".join(sorted(entry.environment_id for entry in named))
        raise RuntimeError(
            f"Rogue Arena eval name '{selector}' is ambiguous (environment ids: "
            f"{ids}); set `eval` to one environment id"
        )
    available = (
        ", ".join(sorted(f"{entry.name} ({entry.environment_id})" for entry in entries))
        or "none"
    )
    raise RuntimeError(
        f"Rogue Arena eval '{selector}' was not found among this token's evals: "
        f"{available}"
    )


async def _prepare(runtime: _TaskRuntime) -> None:
    async with runtime.prepare_lock:
        if runtime.job_id is not None:
            return
        config = runtime.config
        entry = select_eval(await runtime.client.list_evals(), config)
        client_job_key = f"inspect_{runtime.run_id}"

        async def create() -> str:
            job = await runtime.client.create_job(
                environment_id=entry.environment_id,
                n=config.max_trials,
                concurrency=config.concurrency,
                wall_clock_limit_seconds=config.max_runtime_seconds,
                client_job_key=client_job_key,
                visibility=config.team_visibility,
            )
            return job.id

        try:
            job_id = await create()
        except httpx.TransportError:
            # clientJobKey makes the create idempotent: after an unknown
            # transport outcome the retry returns the same job, never a second.
            job_id = await create()
        runtime.entry = entry
        runtime.job_id = job_id
        logger.info(
            "Rogue Arena job %s for eval %s (%s): n=%s concurrency=%s "
            "wall_clock_limit_seconds=%s",
            job_id,
            entry.name,
            entry.environment_id,
            config.max_trials,
            config.concurrency,
            config.max_runtime_seconds,
        )


def _log_claim_failure(sample_id: str, error: BaseException) -> None:
    if isinstance(error, EvalClaimDenied):
        kind = "funding" if error.is_funding else "slot/state"
        logger.error(
            "Rogue Arena denied the trial claim for sample %s (HTTP %s, %s: "
            "verdict=%s reason=%s): %s",
            sample_id,
            error.status_code,
            kind,
            error.denial.verdict,
            error.denial.reason_code,
            error.denial.error_message,
        )
    elif isinstance(error, TrialClosedBeforeReady):
        logger.error(
            "Rogue Arena trial %s for sample %s closed before it was ready "
            "(outcome=%s terminal_reason=%s)",
            error.trial.id,
            sample_id,
            error.trial.outcome.value if error.trial.outcome else None,
            error.trial.terminal_reason.value if error.trial.terminal_reason else None,
        )
    elif isinstance(error, TrialReadyTimeout):
        logger.error(
            "Rogue Arena trial %s for sample %s was not ready within %ss",
            error.trial_id,
            sample_id,
            error.max_wait_seconds,
        )


def _log_closed(trial: EvalTrial) -> None:
    outcome = trial.outcome.value if trial.outcome else None
    reason = trial.terminal_reason.value if trial.terminal_reason else None
    # Mirrors the scorer: any server score is a model
    # result, a timeout included, except an infra_error or a cancellation.
    scored_agent_run = (
        trial.score is not None
        and trial.outcome is not EvalTrialOutcome.CANCELLED
        and trial.terminal_reason
        not in (EvalTrialTerminalReason.INFRA_ERROR, EvalTrialTerminalReason.CANCELLED)
    )
    if scored_agent_run:
        logger.info(
            "Rogue Arena trial %s closed: outcome=%s terminal_reason=%s hops=%s/%s "
            "timed_out=%s",
            trial.id,
            outcome,
            reason,
            trial.hops_passed,
            trial.total_steps,
            trial.terminal_reason is EvalTrialTerminalReason.TIMEOUT,
        )
    else:
        # Not a model result: an infra failure, reap, expiry or cancellation.
        logger.warning(
            "Rogue Arena trial %s ended without a scored agent run: outcome=%s "
            "terminal_reason=%s",
            trial.id,
            outcome,
            reason,
        )


async def _cancel_job(client: RogueArenaClient, job_id: str) -> None:
    try:
        await client.cancel_job(job_id)
    except RogueArenaApiError as error:
        if error.status_code == 409:
            logger.info("Rogue Arena job %s is already complete", job_id)
        else:
            logger.warning(
                "Failed to cancel Rogue Arena job %s during task cleanup: %s",
                job_id,
                error,
            )
    except httpx.TransportError as error:
        logger.warning(
            "Failed to cancel Rogue Arena job %s during task cleanup: %s",
            job_id,
            type(error).__name__,
        )


def _api_message(error: RogueArenaApiError) -> str:
    body = error.body
    if isinstance(body, Mapping) and isinstance(body.get("errorMessage"), str):
        return str(body["errorMessage"])
    return str(body)


class _CommandTimeout(TimeoutError):
    """A guest command hit its own timeout; Inspect surfaces the partial output."""

    def __init__(self, seconds: int, truncated_output: str | None) -> None:
        super().__init__(f"Command timed out after {seconds} seconds")
        self.truncated_output = truncated_output


def _agent_path(file: str) -> str:
    # The hub accepts absolute agent paths only; a relative Inspect path is
    # resolved against the agent's filesystem root.
    return file if file.startswith("/") else f"/{file}"


@sandboxenv(name="rogue-arena")
class RogueArenaSandboxEnvironment(SandboxEnvironment):
    """An Inspect sandbox backed by one ready Rogue Arena eval trial.

    Concurrency: Inspect's sandbox limit (``default_concurrency()``, or
    ``--max-sandboxes``) queues samples before ``sample_init``; the provider's
    own semaphore, sized to the config's ``concurrency`` (the hub job's), is
    the backstop. A task config that sets ``concurrency`` explicitly should
    pass the same value as ``--max-sandboxes`` (or ``ROGUE_ARENA_CONCURRENCY``),
    since ``default_concurrency()`` cannot see the task config.

    ``--no-sandbox-cleanup``: Inspect skips ``sample_cleanup``, so a trial
    stays open and keeps its concurrency slot until a scorer closes it
    (``close_trial()``). ``task_cleanup`` then only stops the heartbeats and
    leaves the open trials and the job for the hub to reap. Without a closing
    scorer, a run with more samples than ``concurrency`` waits indefinitely
    for a slot.
    """

    _runtimes: ClassVar[dict[str, _TaskRuntime]] = {}
    _runtimes_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        client: RogueArenaClient,
        trial: EvalTrial,
        *,
        runtime: _TaskRuntime,
        deadline: float,
    ) -> None:
        super().__init__()
        self._client = client
        self._trial = trial
        self._runtime = runtime
        self._deadline = deadline
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._close_lock = asyncio.Lock()
        self._close_sent = False
        self._closed_trial: EvalTrial | None = None
        self._slot_released = False

    @property
    def trial_id(self) -> str:
        return self._trial.id

    @property
    def job_id(self) -> str:
        """The run (job) this trial belongs to; the team chatroom is keyed by it."""
        return self._trial.job_id

    async def post_team_message(self, author: str, content: str) -> EvalChatMessage:
        """Post to this run's shared team chatroom through the hub (tagged by member)."""
        return await self._client.post_chat_message(
            self._trial.job_id, author=author, content=content
        )

    async def read_team_messages(self, since: int = 0) -> tuple[EvalChatMessage, ...]:
        """Read this run's shared team chatroom through the hub, in time order."""
        return await self._client.list_chat_messages(self._trial.job_id, since=since)

    async def post_activity_entry(
        self,
        member: str,
        command: str,
        summary: str,
        *,
        client_entry_key: str | None = None,
    ) -> EvalActivityEntry:
        """Append one summarized action to this run's team activity feed (tagged by member).

        The feed stores only a summary: the command and a short client-side
        summary of its result (exit code and a capped head of the output). The
        command's full output still passed through Rogue Arena when it ran.
        """
        return await self._client.post_activity_entry(
            self._trial.job_id,
            member=member,
            command=command,
            summary=summary,
            client_entry_key=client_entry_key,
        )

    async def read_activity_entries(
        self, since: int = 0, *, limit: int | None = None
    ) -> tuple[EvalActivityEntry, ...]:
        """Read this run's shared team activity feed through the hub, in time order."""
        return await self._client.list_activity_entries(
            self._trial.job_id, since=since, limit=limit
        )

    @property
    def closed_trial(self) -> EvalTrial | None:
        """The trial as the hub closed it (outcome, terminal reason, score)."""
        return self._closed_trial

    @property
    def team_visibility(self) -> str:
        """The team visibility this run's hub job was created with."""
        return self._runtime.config.team_visibility

    @classmethod
    def default_concurrency(cls) -> int | None:
        # Inspect's max_sandboxes default: samples queue in Inspect, before
        # sample_init, at the hub job's concurrency. Inspect calls this before
        # task_init and passes no config, so it resolves the sources a config
        # without an explicit `concurrency` uses (ROGUE_ARENA_CONCURRENCY, else
        # the default). --max-sandboxes overrides it.
        return RogueArenaSandboxConfig.from_sources(None).concurrency

    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> RogueArenaSandboxConfig:
        return RogueArenaSandboxConfig.model_validate(config)

    @classmethod
    def _get_or_create_runtime(
        cls, config: RogueArenaSandboxConfig
    ) -> tuple[str, _TaskRuntime]:
        client = RogueArenaClient.from_env()
        key = _runtime_key(config, client)
        with cls._runtimes_lock:
            runtime = cls._runtimes.get(key)
            if runtime is None:
                runtime = _TaskRuntime(config=config, client=client)
                cls._runtimes[key] = runtime
        return key, runtime

    @classmethod
    async def task_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None
    ) -> None:
        del task_name  # "startup": the job is per config, not per task
        resolved = _as_config(config)
        key, runtime = cls._get_or_create_runtime(resolved)
        try:
            await _prepare(runtime)
        except BaseException:
            with cls._runtimes_lock:
                if cls._runtimes.get(key) is runtime:
                    cls._runtimes.pop(key, None)
            raise

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        resolved = _as_config(config)
        _, runtime = cls._get_or_create_runtime(resolved)
        await _prepare(runtime)
        entry, job_id = runtime.entry, runtime.job_id
        if entry is None or job_id is None:
            raise RuntimeError("Rogue Arena eval job was not created")

        sample_id = _require_sample_id(metadata)
        attempt_key = await runtime.next_attempt_key(task_name, sample_id)
        # Waiting for a slot is ordinary queueing, so it is unbounded
        # (max_wait_seconds bounds provisioning, not the queue).
        await runtime.trial_slots.acquire()
        # The hub's wall clock runs from the claim, provisioning included.
        claimed_at = time.monotonic()
        try:
            trial = await runtime.client.claim_trial(
                job_id,
                client_attempt_key=attempt_key,
                inspect_sample_id=sample_id,
                # Inspect sets __sample_id__; these four are forwarded when
                # Inspect sets them, and sent as null otherwise.
                inspect_epoch=_reserved_int(metadata, "__sample_epoch__"),
                inspect_sample_uuid=_reserved_str(metadata, "__sample_uuid__"),
                inspect_eval_id=_reserved_str(metadata, "__eval_id__"),
                inspect_run_id=runtime.run_id,
                inspect_eval_set_id=_reserved_str(metadata, "__eval_set_id__"),
                inspect_task_id=task_name,
                sandbox_name=entry.name or entry.environment_id,
                max_wait_seconds=resolved.max_wait_seconds,
                poll_interval_seconds=resolved.poll_interval_seconds,
            )
        except BaseException as error:
            runtime.trial_slots.release()
            _log_claim_failure(sample_id, error)
            raise
        sandbox = cls(
            runtime.client,
            trial,
            runtime=runtime,
            deadline=claimed_at + resolved.max_runtime_seconds,
        )
        runtime.environments.add(sandbox)
        sandbox._start_heartbeat()
        logger.info(
            "Rogue Arena sample_id=%s trial_id=%s job_id=%s ready (warm=%s)",
            sample_id,
            trial.id,
            job_id,
            trial.warm,
        )
        return {"default": sandbox}

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        del task_name, config
        outcome = EvalTrialOutcome.CANCELLED if interrupted else EvalTrialOutcome.CLOSED
        for environment in environments.values():
            if isinstance(environment, RogueArenaSandboxEnvironment):
                await environment._close(outcome)

    @classmethod
    async def task_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        cleanup: bool,
    ) -> None:
        del task_name  # "shutdown": the runtime is keyed by config
        resolved = _as_config(config)
        client = RogueArenaClient.from_env()
        key = _runtime_key(resolved, client)
        with cls._runtimes_lock:
            runtime = cls._runtimes.pop(key, None)
        if runtime is None:
            return
        leftovers = tuple(runtime.environments)
        if not cleanup:
            for environment in leftovers:
                await environment._stop_heartbeat()
            logger.info(
                "Rogue Arena job %s and %d open trial(s) left for inspection; "
                "Rogue Arena reaps them when their lease lapses",
                runtime.job_id,
                len(leftovers),
            )
            return
        try:
            for environment in leftovers:
                try:
                    await environment._close(EvalTrialOutcome.CANCELLED)
                except Exception as error:  # noqa: BLE001 - best-effort cleanup
                    logger.warning(
                        "Failed to close Rogue Arena trial %s during task cleanup: "
                        "%s: %s",
                        environment.trial_id,
                        type(error).__name__,
                        error,
                    )
        finally:
            if runtime.job_id is not None:
                await _cancel_job(runtime.client, runtime.job_id)

    # --- lease --------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name=f"rogue-arena-heartbeat-{self.trial_id}"
        )

    async def _heartbeat_loop(self) -> None:
        interval = self._runtime.config.heartbeat_interval_seconds
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                return  # the hub's wall clock has ended the lease
            await _sleep(min(interval, remaining))
            if time.monotonic() >= self._deadline:
                return
            if await self._beat() is not None:
                return

    async def _beat(self) -> str | None:
        """Extend the lease once.

        Returns the hub's reason when it says the trial is no longer live
        (409); None when the lease moved or a transient failure will retry.
        """
        try:
            await self._client.heartbeat_trial(self.trial_id)
        except RogueArenaApiError as error:
            if error.status_code == 409:
                reason = _api_message(error)
                logger.warning(
                    "Rogue Arena trial %s is no longer live: %s", self.trial_id, reason
                )
                return reason
            logger.warning(
                "Rogue Arena heartbeat for trial %s failed (HTTP %s); will retry",
                self.trial_id,
                error.status_code,
            )
        except httpx.TransportError as error:
            logger.warning(
                "Rogue Arena heartbeat for trial %s failed (%s); will retry",
                self.trial_id,
                type(error).__name__,
            )
        except (KeyError, TypeError, ValueError) as error:
            # The hub answered 200, so the lease moved; only the parse of the
            # returned trial failed (e.g. a status this client does not know).
            logger.warning(
                "Rogue Arena heartbeat answer for trial %s could not be parsed "
                "(%s); will retry",
                self.trial_id,
                type(error).__name__,
            )
        return None

    async def _stop_heartbeat(self) -> None:
        task, self._heartbeat_task = self._heartbeat_task, None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.wait({task})
        if not task.cancelled() and task.exception() is not None:
            logger.warning(
                "Rogue Arena heartbeat loop for trial %s failed: %r",
                self.trial_id,
                task.exception(),
            )

    # --- close --------------------------------------------------------------

    async def close_trial(self) -> EvalTrial:
        """Close the trial ``closed`` so the hub scores it; return the scored trial.

        Inspect runs scorers before ``sample_cleanup``, so a scorer that needs
        the server score calls this. Idempotent: a later call (or the cleanup)
        returns the same trial and never resends the close.
        """
        trial = await self._close(EvalTrialOutcome.CLOSED)
        if trial is None:
            raise RuntimeError(
                f"Rogue Arena trial {self.trial_id} close already failed; a close "
                "is never resent"
            )
        return trial

    async def _close(self, outcome: EvalTrialOutcome) -> EvalTrial | None:
        async with self._close_lock:
            if self._close_sent:
                return self._closed_trial
            # Sent at most once: the client already reads a gateway-timed-out
            # close back from the server, so no caller retries it.
            # From here to the (client-shielded) close POST nothing may be
            # cancelled, or a cancelled scorer close would leave the trial open.
            self._close_sent = True
            with anyio.move_on_after(_SHIELD_SECONDS, shield=True):
                await self._stop_heartbeat()
            total_steps = max(
                self._trial.total_steps,
                self._runtime.entry.hop_count if self._runtime.entry else 0,
                1,
            )
            try:
                trial = await self._client.close_trial(
                    self.trial_id,
                    outcome=outcome,
                    max_wait_seconds=_CLOSE_BASE_SECONDS
                    + total_steps * _CLOSE_PER_HOP_SECONDS,
                    poll_interval_seconds=self._runtime.config.poll_interval_seconds,
                )
            finally:
                self._runtime.environments.discard(self)
                self._release_slot()
            self._closed_trial = trial
            _log_closed(trial)
            return trial

    def _release_slot(self) -> None:
        if not self._slot_released:
            self._slot_released = True
            self._runtime.trial_slots.release()

    # --- guest operations ---------------------------------------------------

    def _require_live(self) -> float:
        if self._close_sent:
            raise SandboxUnavailableError(
                f"Rogue Arena trial {self.trial_id} is closed"
            )
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxUnavailableError(
                f"Rogue Arena trial {self.trial_id} has used its "
                f"max_runtime_seconds ({self._runtime.config.max_runtime_seconds}s); "
                "Rogue Arena no longer runs commands on it"
            )
        return remaining

    async def _bounded(self, start: Callable[[], Awaitable[_T]]) -> _T:
        """Run one trial operation inside the trial's remaining wall clock."""
        remaining = self._require_live()
        task = asyncio.ensure_future(start())
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
        except BaseException:
            task.cancel()
            raise
        if task not in done:
            task.cancel()
            await asyncio.wait({task})
            raise SandboxUnavailableError(
                f"Rogue Arena trial {self.trial_id} reached its max_runtime_seconds "
                f"({self._runtime.config.max_runtime_seconds}s) mid-operation"
            )
        try:
            return task.result()
        except RogueArenaApiError as error:
            if error.status_code == 409:
                raise SandboxUnavailableError(
                    f"Rogue Arena trial {self.trial_id} is not running: "
                    f"{_api_message(error)}"
                ) from error
            raise

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        """Run ``cmd`` on the trial's agent machine.

        Failure mapping (the asymmetry is deliberate):

        - The hub reports the exec ``lost``, or the trial is no longer live (409,
          closed, or past ``max_runtime_seconds``) → ``SandboxUnavailableError``:
          a tool error, and the sample continues.
        - ``ExecOutcomeUnknown`` (a gateway timeout or transport failure on a
          sync exec or background launch) propagates and errors the sample:
          the command may still be running, so a retry could run it twice.
        - The command's own timeout → ``TimeoutError`` with the partial output.
        """
        del user, timeout_retry, concurrency
        config = self._runtime.config
        stdin = (
            input.decode("utf-8", errors="replace")
            if isinstance(input, bytes)
            else input
        )
        # No timeout means a sync exec of at most the ceiling: Inspect's own
        # calls pass none and stay one request.
        command_timeout = min(
            timeout or config.sync_exec_ceiling_seconds, config.max_runtime_seconds
        )
        not_live = await self._bounded(self._beat)
        if not_live is not None:
            raise SandboxUnavailableError(
                f"Rogue Arena trial {self.trial_id} is not running: {not_live}"
            )
        result = await self._bounded(
            lambda: self._client.run_command(
                self.trial_id,
                cmd=cmd,
                cwd=cwd,
                env=env or None,
                stdin=stdin,
                timeout_seconds=command_timeout,
                sync_ceiling_seconds=config.sync_exec_ceiling_seconds,
                poll_interval_seconds=config.poll_interval_seconds,
                max_total_seconds=EVAL_MAX_EXEC_SECONDS,
            )
        )
        if result.lost:
            # The hub's own failure, not the command's output (Inspect's
            # SandboxUnavailableError contract): the command may or may not
            # have run.
            raise SandboxUnavailableError(
                f"Rogue Arena lost exec {result.exec_id} on trial {self.trial_id}: "
                "no exit status was recorded"
            )
        if result.timed_out:
            raise _CommandTimeout(
                command_timeout, (result.stdout + result.stderr) or None
            )
        exit_code = result.exit_code if result.exit_code is not None else 1
        return ExecResult(
            success=exit_code == 0,
            returncode=exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def write_file(self, file: str, contents: str | bytes) -> None:
        payload = contents.encode("utf-8") if isinstance(contents, str) else contents
        await self._bounded(
            lambda: self._client.upload_file(self.trial_id, _agent_path(file), payload)
        )

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        raw = await self._bounded(
            lambda: self._client.download_file(self.trial_id, _agent_path(file))
        )
        if len(raw) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )
        return raw.decode("utf-8") if text else raw
