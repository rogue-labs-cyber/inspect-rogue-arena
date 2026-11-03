from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import anyio
import httpx

_LOGGER = logging.getLogger(__name__)

# A started close (or unready cancel) is shielded from an outer cancellation so
# it always reaches Rogue Arena (it is never resent), bounded by the
# close request's own timeout so it cannot hold a cancellation open forever.
_CLOSE_REQUEST_SECONDS = 60


class EvalJobStatus(str, Enum):
    OPEN = "open"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETE = "complete"


class EvalTrialStatus(str, Enum):
    PROVISIONING = "provisioning"
    READY = "ready"
    RUNNING = "running"
    CLOSING = "closing"
    CLOSED = "closed"


class EvalTrialOutcome(str, Enum):
    CLOSED = "closed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REAPED = "reaped"


class EvalTrialTerminalReason(str, Enum):
    AGENT_SUCCESS = "agent_success"
    AGENT_FAIL = "agent_fail"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"
    HALTED = "halted"
    # A server terminal reason; accepted so parsing never fails on it.
    TEAM_MEMBER = "team_member"


class EvalClientKind(str, Enum):
    INSPECT = "inspect"
    HARBOR = "harbor"
    HUMAN = "human"


class EvalScoringMode(str, Enum):
    SEQUENTIAL = "sequential"
    INDEPENDENT = "independent"


class EvalTrialEventKind(str, Enum):
    EXEC = "exec"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    HOP = "hop"
    LIFECYCLE = "lifecycle"


# The error message carries at most this much of a response body; the full
# body stays on ``RogueArenaApiError.body``.
_ERROR_BODY_CHARS = 500
_HTML_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# Any personal eval token (or the start of one) a response body echoes.
_TOKEN_SHAPE = re.compile(r"ra_eval_[A-Za-z0-9_-]*")
_REDACTED = "[redacted]"


def _redact_secrets(text: str, secret: str | None) -> str:
    """Replace the configured token and any ``ra_eval_`` run with a marker.

    Runs on the whole text BEFORE any cut, so a token echoed across the cut
    never leaves a prefix behind.
    """
    if secret:
        text = text.replace(secret, _REDACTED)
    return _TOKEN_SHAPE.sub(_REDACTED, text)


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:512].lower()
    return head.startswith(("<!doctype html", "<html")) or "<html" in head


def _describe_error_body(status_code: int, body: Any, secret: str | None) -> str:
    """One line for an error body: an HTML error page is named, not dumped.

    Tokens are redacted before any truncation (see ``_redact_secrets``).
    """
    if 300 <= status_code < 400:
        # request_result never follows redirects: a 3xx means the base URL is
        # not answering the API directly.
        return (
            "the server redirected the request instead of answering it "
            "(check the base URL)"
        )
    if isinstance(body, str) and _looks_like_html(body):
        match = _HTML_TITLE.search(body)
        title = (
            _redact_secrets(" ".join(match.group(1).split()), secret) if match else ""
        )
        named = f" titled {title[:120]!r}" if title else ""
        return (
            f"an HTML page{named} ({len(body)} characters), not a Rogue Arena "
            "API response"
        )
    text = _redact_secrets(str(body), secret)
    if len(text) > _ERROR_BODY_CHARS:
        return (
            f"{text[:_ERROR_BODY_CHARS]}... "
            f"[{len(text) - _ERROR_BODY_CHARS} more characters]"
        )
    return text


class RogueArenaApiError(RuntimeError):
    """A hub response outside the expected statuses.

    The message is one line with tokens redacted (``secret``, the client's own
    token, and any ``ra_eval_`` run); ``body`` keeps the raw response.
    """

    def __init__(
        self,
        status_code: int,
        method: str,
        path: str,
        body: Any,
        *,
        secret: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.path = path
        self.body = body
        super().__init__(
            f"Rogue Arena API returned {status_code} for {method} {path}: "
            f"{_describe_error_body(status_code, body, secret)}"
        )


class EvalClaimDenied(RogueArenaApiError):
    """A 402 (funding) or 409 (slot/state) claim denial carrying the deploy decision."""

    def __init__(
        self,
        status_code: int,
        path: str,
        denial: TrialClaimDenied,
        body: Any,
        *,
        secret: str | None = None,
    ) -> None:
        self.denial = denial
        super().__init__(status_code, "POST", path, body, secret=secret)

    @property
    def is_funding(self) -> bool:
        return self.status_code == 402


class TrialReadyTimeout(TimeoutError):
    def __init__(self, trial_id: str, max_wait_seconds: float) -> None:
        self.trial_id = trial_id
        self.max_wait_seconds = max_wait_seconds
        super().__init__(
            f"Rogue Arena trial {trial_id} was not ready within {max_wait_seconds:g} seconds"
        )


class TrialClosedBeforeReady(RuntimeError):
    def __init__(self, trial: EvalTrial) -> None:
        self.trial = trial
        outcome = trial.outcome.value if trial.outcome else None
        reason = trial.terminal_reason.value if trial.terminal_reason else None
        super().__init__(
            f"Rogue Arena trial {trial.id} became {trial.status.value} before it "
            f"was ready (outcome={outcome}, terminalReason={reason})"
        )


def _str(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Rogue Arena returned a non-string {field}")
    return value


def _opt_str(value: Any, field: str) -> str | None:
    # The contract says string or null: a non-string is a server defect.
    return None if value is None else _str(value, field)


def _opt_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Rogue Arena returned an invalid {field}")
    return value


@dataclass(frozen=True, slots=True)
class HttpResult:
    status_code: int
    body: Any
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EvalWhoami:
    user_main_id: str
    organization_id: str | None
    token_label: str
    token_last4: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalWhoami:
        return cls(
            user_main_id=_str(value["userMainId"], "userMainId"),
            organization_id=_opt_str(value.get("organizationId"), "organizationId"),
            token_label=_str(value.get("tokenLabel", ""), "tokenLabel"),
            token_last4=_str(value["tokenLast4"], "tokenLast4"),
        )


@dataclass(frozen=True, slots=True)
class EvalReadiness:
    """Token-scoped launch readiness for one eval, when the hub reports it.

    ``warm_count`` is the number of ready/warm copies this token can claim
    right now; ``on_demand_eta_seconds`` is the hub's estimate of how long a
    cold claim takes to provision (``None`` when the hub does not estimate it).
    The whole ``readiness`` object is optional on the listing: when it is
    absent ``EvalCatalogEntry.readiness`` is ``None``, which the CLI shows as
    "unknown" rather than failing.
    """

    warm_count: int
    on_demand_eta_seconds: int | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalReadiness:
        return cls(
            warm_count=int(value["warmCount"]),
            on_demand_eta_seconds=_opt_int(value.get("onDemandEtaSeconds")),
        )

    def label(self) -> str:
        """A one-line launch-visibility label for ``evals``."""
        if self.warm_count > 0:
            return f"ready now ({self.warm_count} warm)"
        if self.on_demand_eta_seconds is not None:
            minutes = max(1, round(self.on_demand_eta_seconds / 60))
            return f"on demand, ~{minutes} min"
        return "on demand"


@dataclass(frozen=True, slots=True)
class EvalCatalogEntry:
    environment_id: str
    name: str
    environment_build_id: str
    canvas_version_number: int
    hop_count: int
    agent_machine_nickname: str | None
    scoring_mode: EvalScoringMode
    readiness: EvalReadiness | None
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalCatalogEntry:
        readiness = value.get("readiness")
        return cls(
            environment_id=_str(value["environmentId"], "environmentId"),
            name=_str(value.get("name", ""), "name"),
            environment_build_id=_str(
                value["environmentBuildId"], "environmentBuildId"
            ),
            canvas_version_number=int(value["canvasVersionNumber"]),
            hop_count=int(value["hopCount"]),
            agent_machine_nickname=_opt_str(
                value.get("agentMachineNickname"), "agentMachineNickname"
            ),
            scoring_mode=EvalScoringMode(_str(value["scoringMode"], "scoringMode")),
            readiness=(
                EvalReadiness.from_dict(_mapping(readiness, "readiness"))
                if readiness is not None
                else None
            ),
            raw=MappingProxyType(dict(value)),
        )

    def readiness_label(self) -> str:
        """The launch-visibility label, or "unknown" when the hub omits it."""
        return self.readiness.label() if self.readiness is not None else "unknown"


@dataclass(frozen=True, slots=True)
class EvalJob:
    id: str
    environment_id: str
    environment_build_id: str
    canvas_version_id: str
    owner_user_main_id: str
    organization_id: str | None
    n: int
    concurrency: int
    wall_clock_limit_seconds: int
    client_job_key: str | None
    status: EvalJobStatus
    trial_count: int
    closed_trial_count: int
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalJob:
        return cls(
            id=_str(value["id"], "job id"),
            environment_id=_str(value["environmentId"], "environmentId"),
            environment_build_id=_str(
                value["environmentBuildId"], "environmentBuildId"
            ),
            canvas_version_id=_str(value["canvasVersionId"], "canvasVersionId"),
            owner_user_main_id=_str(value["ownerUserMainId"], "ownerUserMainId"),
            organization_id=_opt_str(value.get("organizationId"), "organizationId"),
            n=int(value["n"]),
            concurrency=int(value["concurrency"]),
            wall_clock_limit_seconds=int(value["wallClockLimitSeconds"]),
            client_job_key=_opt_str(value.get("clientJobKey"), "clientJobKey"),
            status=EvalJobStatus(_str(value["status"], "job status")),
            trial_count=int(value.get("trialCount", 0)),
            closed_trial_count=int(value.get("closedTrialCount", 0)),
            raw=MappingProxyType(dict(value)),
        )


@dataclass(frozen=True, slots=True)
class EvalScoreStep:
    step_number: int
    name: str
    passed: bool
    passed_at: str | None
    detail: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalScoreStep:
        return cls(
            step_number=int(value["stepNumber"]),
            name=_str(value.get("name", ""), "name"),
            passed=bool(value["passed"]),
            passed_at=_opt_str(value.get("passedAt"), "passedAt"),
            detail=_str(value.get("detail", ""), "detail"),
        )


@dataclass(frozen=True, slots=True)
class EvalScore:
    reached_step: int
    hops_passed: int
    total_steps: int
    scoring_mode: EvalScoringMode
    steps: tuple[EvalScoreStep, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalScore:
        steps = value.get("steps", [])
        if not isinstance(steps, list):
            raise TypeError("Rogue Arena returned an invalid score steps list")
        return cls(
            reached_step=int(value["reachedStep"]),
            hops_passed=int(value["hopsPassed"]),
            total_steps=int(value["totalSteps"]),
            scoring_mode=EvalScoringMode(_str(value["scoringMode"], "scoringMode")),
            steps=tuple(
                EvalScoreStep.from_dict(_mapping(row, "score step")) for row in steps
            ),
        )


@dataclass(frozen=True, slots=True)
class EvalHopResult:
    step_number: int
    name: str
    target_found: bool
    check_ran: bool
    passed: bool
    detail: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalHopResult:
        return cls(
            step_number=int(value["stepNumber"]),
            name=_str(value.get("name", ""), "name"),
            target_found=bool(value["targetFound"]),
            check_ran=bool(value["checkRan"]),
            passed=bool(value["passed"]),
            detail=_str(value.get("detail", ""), "detail"),
        )


@dataclass(frozen=True, slots=True)
class EvalTrial:
    id: str
    job_id: str
    deployment_record_id: str | None
    agent_deployed_vm_id: str | None
    agent_machine_nickname: str | None
    status: EvalTrialStatus
    outcome: EvalTrialOutcome | None
    terminal_reason: EvalTrialTerminalReason | None
    warm: bool
    reached_step: int | None
    hops_passed: int | None
    total_steps: int
    claimed_at: str | None
    ready_at: str | None
    closed_at: str | None
    last_heartbeat_at: str | None
    lease_expires_at: str | None
    client_kind: EvalClientKind | None
    client_attempt_key: str | None
    score: EvalScore | None
    hop_results: tuple[EvalHopResult, ...] | None
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalTrial:
        outcome = value.get("outcome")
        reason = value.get("terminalReason")
        kind = value.get("clientKind")
        score = value.get("score")
        hops = value.get("hopResults")
        if hops is not None and not isinstance(hops, list):
            raise TypeError("Rogue Arena returned an invalid hopResults list")
        return cls(
            id=_str(value["id"], "trial id"),
            job_id=_str(value["jobId"], "jobId"),
            deployment_record_id=_opt_str(
                value.get("deploymentRecordId"), "deploymentRecordId"
            ),
            agent_deployed_vm_id=_opt_str(
                value.get("agentDeployedVmId"), "agentDeployedVmId"
            ),
            agent_machine_nickname=_opt_str(
                value.get("agentMachineNickname"), "agentMachineNickname"
            ),
            status=EvalTrialStatus(_str(value["status"], "trial status")),
            outcome=EvalTrialOutcome(outcome) if outcome is not None else None,
            terminal_reason=(
                EvalTrialTerminalReason(reason) if reason is not None else None
            ),
            warm=bool(value.get("warm", False)),
            reached_step=_opt_int(value.get("reachedStep")),
            hops_passed=_opt_int(value.get("hopsPassed")),
            total_steps=int(value["totalSteps"]),
            claimed_at=_opt_str(value.get("claimedAt"), "claimedAt"),
            ready_at=_opt_str(value.get("readyAt"), "readyAt"),
            closed_at=_opt_str(value.get("closedAt"), "closedAt"),
            last_heartbeat_at=_opt_str(value.get("lastHeartbeatAt"), "lastHeartbeatAt"),
            lease_expires_at=_opt_str(value.get("leaseExpiresAt"), "leaseExpiresAt"),
            client_kind=EvalClientKind(kind) if kind is not None else None,
            client_attempt_key=_opt_str(
                value.get("clientAttemptKey"), "clientAttemptKey"
            ),
            score=(
                EvalScore.from_dict(_mapping(score, "trial score"))
                if score is not None
                else None
            ),
            hop_results=(
                tuple(
                    EvalHopResult.from_dict(_mapping(row, "hop result")) for row in hops
                )
                if hops is not None
                else None
            ),
            raw=MappingProxyType(dict(value)),
        )


@dataclass(frozen=True, slots=True)
class EvalTrialEvent:
    id: str
    at: str
    kind: EvalTrialEventKind
    summary: str
    detail: str | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalTrialEvent:
        return cls(
            id=_str(value["id"], "event id"),
            at=_str(value["at"], "event at"),
            kind=EvalTrialEventKind(_str(value["kind"], "event kind")),
            summary=_str(value.get("summary", ""), "summary"),
            detail=_opt_str(value.get("detail"), "detail"),
        )


@dataclass(frozen=True, slots=True)
class EvalChatMessage:
    """One message in a shared run's team chatroom, tagged by its author.

    The room is per job (the shared run), so every member of a team run reads
    and writes the same time-ordered log. ``seq`` is a server-assigned,
    strictly increasing order key. It is drawn from a global sequence, not a
    per-job 1-based counter, so a job's seqs are monotonic but need not start at
    1 or be contiguous; order and the ``since`` cursor (``seq`` greater than
    ``since``, within the job) are unaffected.
    """

    id: str
    job_id: str
    seq: int
    at: str
    author: str
    content: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalChatMessage:
        return cls(
            id=_str(value["id"], "chat id"),
            job_id=_str(value["jobId"], "jobId"),
            seq=int(value["seq"]),
            at=_str(value["at"], "chat at"),
            author=_str(value["author"], "author"),
            content=_str(value["content"], "content"),
        )


@dataclass(frozen=True, slots=True)
class EvalActivityEntry:
    """One entry in a shared run's team activity feed, tagged by its member.

    The feed is per job (the shared run), so every member of a team run reads
    and writes the same time-ordered feed. ``seq`` is a server-assigned,
    strictly increasing order key (a global sequence, so a job's seqs are
    monotonic but need not start at 1 or be contiguous); order and the ``since``
    cursor (``seq`` greater than ``since``, within the job) are unaffected. The
    feed stores only the ``command`` and a condensed ``summary`` of its result
    (exit code and a capped head of the output); the command's full output still
    passes through Rogue Arena when the command runs.
    """

    id: str
    job_id: str
    seq: int
    at: str
    member: str
    command: str
    summary: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalActivityEntry:
        return cls(
            id=_str(value["id"], "activity id"),
            job_id=_str(value["jobId"], "jobId"),
            seq=int(value["seq"]),
            at=_str(value["at"], "activity at"),
            member=_str(value["member"], "member"),
            command=_str(value["command"], "command"),
            summary=_str(value["summary"], "summary"),
        )


@dataclass(frozen=True, slots=True)
class TrialClaimDenied:
    error_message: str
    verdict: str
    reason_code: str
    decision: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrialClaimDenied:
        decision = _mapping(value["decision"], "claim decision")
        return cls(
            error_message=_str(value.get("errorMessage", ""), "errorMessage"),
            verdict=_str(decision["verdict"], "verdict"),
            reason_code=_str(decision["reasonCode"], "reasonCode"),
            decision=MappingProxyType(dict(decision)),
        )


_LOST_EXIT_CODE = 255
_EXEC_LOST_DETAIL = "exec lost: no exit status was recorded within the exec max age"
_TRANSIENT_STATUSES = (502, 503, 504)
# Ceiling of the background-poll backoff after consecutive transient failures.
_POLL_BACKOFF_CAP_SECONDS = 30.0
# A background exec's first poll comes this soon, and the wait doubles while it
# runs up to poll_interval_seconds, so a quick command answers in well under a
# second while a long one is polled at the configured cadence.
_FIRST_POLL_SECONDS = 0.25


class ExecOutcomeUnknown(RuntimeError):
    """A sync exec or background launch got no answer; it may still be running."""

    def __init__(
        self,
        trial_id: str,
        cmd: Sequence[str],
        cause: str,
        exec_id: str | None = None,
    ) -> None:
        self.trial_id = trial_id
        self.cmd = tuple(cmd)
        self.exec_id = exec_id
        super().__init__(
            f"exec outcome unknown ({cause}); the command {list(self.cmd)!r} may "
            f"still be running on the VM of trial {trial_id}; not retried"
        )


def _is_transient(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return (
        isinstance(error, RogueArenaApiError)
        and error.status_code in _TRANSIENT_STATUSES
    )


@dataclass(frozen=True, slots=True)
class EvalExecResult:
    running: bool
    exec_id: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def lost(self) -> bool:
        """True only for the hub's lost-exec report (exit 255 plus its detail text)."""
        return self.exit_code == _LOST_EXIT_CODE and _EXEC_LOST_DETAIL in self.stderr

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvalExecResult:
        return cls(
            running=bool(value["running"]),
            exec_id=_opt_str(value.get("execId"), "execId"),
            exit_code=_opt_int(value.get("exitCode")),
            stdout=_str(value.get("stdout", ""), "stdout"),
            stderr=_str(value.get("stderr", ""), "stderr"),
            timed_out=bool(value.get("timedOut", False)),
        )


def _claim_body(
    job_id: str,
    *,
    client_attempt_key: str,
    inspect_sample_id: str | None,
    inspect_epoch: int | None,
    inspect_sample_uuid: str | None,
    inspect_eval_id: str | None,
    inspect_run_id: str | None,
    inspect_eval_set_id: str | None,
    inspect_task_id: str | None,
    sandbox_name: str,
) -> dict[str, str | int | bool | None]:
    body: dict[str, str | int | bool | None] = {
        "jobId": job_id,
        "clientAttemptKey": client_attempt_key,
        "clientKind": EvalClientKind.INSPECT.value,
        "inspectSampleId": inspect_sample_id or None,
        "inspectEpoch": inspect_epoch,
        "inspectSampleUuid": inspect_sample_uuid or None,
        "inspectEvalId": inspect_eval_id or None,
        "inspectRunId": inspect_run_id or None,
        "inspectEvalSetId": inspect_eval_set_id or None,
        "inspectTaskId": inspect_task_id or None,
        "harborTrialUuid": None,
        "harborTrialName": None,
        "harborJobId": None,
        "harborAttempt": None,
        "sandboxName": sandbox_name,
    }
    return body


def _looks_like_claim_denial(body: Any) -> bool:
    return isinstance(body, Mapping) and isinstance(body.get("decision"), Mapping)


class RogueArenaClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        request_timeout_seconds: float = 3600.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._token = token or None
        self._transport = transport
        self._request_timeout_seconds = request_timeout_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def base_url(self) -> str:
        return self._base

    @property
    def credential_fingerprint(self) -> str:
        if self._token is None:
            raise RuntimeError("an authenticated Rogue Arena token is required")
        return hashlib.sha256(self._token.encode()).hexdigest()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def request_result(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
        expected_statuses: Sequence[int] | None = None,
        timeout_seconds: float | None = None,
    ) -> HttpResult:
        if not path.startswith("/"):
            raise ValueError("Rogue Arena API paths must begin with '/'")
        effective = (
            min(self._request_timeout_seconds, timeout_seconds)
            if timeout_seconds is not None
            else self._request_timeout_seconds
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(effective), transport=self._transport
        ) as client:
            response = await client.request(
                method,
                f"{self._base}{path}",
                headers=self._headers(),
                json=json,
                params=params,
            )
        body: Any
        if response.status_code == 204 or not response.content:
            body = None
        else:
            try:
                body = response.json()
            except ValueError:
                body = response.text
        accepted = (
            response.status_code in expected_statuses
            if expected_statuses is not None
            else 200 <= response.status_code < 300
        )
        if not accepted:
            raise RogueArenaApiError(
                response.status_code, method, path, body, secret=self._token
            )
        return HttpResult(
            response.status_code, body, MappingProxyType(dict(response.headers))
        )

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        return (await self.request_result(method, path, **kwargs)).body

    def with_token(self, token: str) -> RogueArenaClient:
        return RogueArenaClient(
            self._base,
            token,
            transport=self._transport,
            request_timeout_seconds=self._request_timeout_seconds,
            sleep=self._sleep,
            monotonic=self._monotonic,
        )

    async def whoami(self) -> EvalWhoami:
        return EvalWhoami.from_dict(
            _mapping(await self._request("GET", "/hub/eval/whoami"), "whoami")
        )

    async def list_evals(self) -> tuple[EvalCatalogEntry, ...]:
        body = await self._request("GET", "/hub/eval/evals")
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid evals list")
        return tuple(
            EvalCatalogEntry.from_dict(_mapping(row, "eval entry")) for row in body
        )

    async def list_runs(self) -> tuple[EvalJob, ...]:
        """List this token's recent eval runs (jobs).

        Calls ``GET /hub/eval/jobs``, a token-scoped list endpoint. If it is
        not available a 404/405 surfaces as a ``RogueArenaApiError`` the CLI
        turns into a clean "not available" note rather than a crash.
        """
        body = await self._request("GET", "/hub/eval/jobs")
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid runs list")
        return tuple(EvalJob.from_dict(_mapping(row, "run")) for row in body)

    async def post_chat_message(
        self,
        job_id: str,
        *,
        author: str,
        content: str,
        client_message_key: str | None = None,
    ) -> EvalChatMessage:
        """Post one message to a shared run's team chatroom.

        ``client_message_key`` makes the post idempotent: a retry with the same
        key replays the stored message, never a second one.
        """
        payload: dict[str, str] = {"author": author, "content": content}
        if client_message_key is not None:
            payload["clientMessageKey"] = client_message_key
        body = await self._request(
            "POST",
            f"/hub/eval/jobs/{job_id}/chat",
            json=payload,
            expected_statuses=(200,),
        )
        return EvalChatMessage.from_dict(_mapping(body, "chat message"))

    async def list_chat_messages(
        self, job_id: str, *, since: int = 0
    ) -> tuple[EvalChatMessage, ...]:
        """Read a shared run's team chatroom in time order; ``since`` skips seqs at or below it."""
        body = await self._request(
            "GET",
            f"/hub/eval/jobs/{job_id}/chat",
            params={"since": since} if since else None,
        )
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid chat list")
        return tuple(
            EvalChatMessage.from_dict(_mapping(row, "chat message")) for row in body
        )

    async def post_activity_entry(
        self,
        job_id: str,
        *,
        member: str,
        command: str,
        summary: str,
        client_entry_key: str | None = None,
    ) -> EvalActivityEntry:
        """Append one entry to a shared run's team activity feed.

        ``client_entry_key`` makes the append idempotent: a retry with the same
        key replays the stored entry, never a second one. The server assigns
        ordering (``seq``).
        """
        payload: dict[str, str] = {
            "member": member,
            "command": command,
            "summary": summary,
        }
        if client_entry_key is not None:
            payload["clientEntryKey"] = client_entry_key
        body = await self._request(
            "POST",
            f"/hub/eval/jobs/{job_id}/activity",
            json=payload,
            expected_statuses=(200,),
        )
        return EvalActivityEntry.from_dict(_mapping(body, "activity entry"))

    async def list_activity_entries(
        self, job_id: str, *, since: int = 0, limit: int | None = None
    ) -> tuple[EvalActivityEntry, ...]:
        """Read a shared run's team activity feed in time order.

        ``since`` skips seqs at or below it; ``limit`` bounds the page so a busy
        feed does not overflow the reader's context.
        """
        params: dict[str, int] = {}
        if since:
            params["since"] = since
        if limit is not None:
            params["limit"] = limit
        body = await self._request(
            "GET",
            f"/hub/eval/jobs/{job_id}/activity",
            params=params or None,
        )
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid activity list")
        return tuple(
            EvalActivityEntry.from_dict(_mapping(row, "activity entry")) for row in body
        )

    async def create_job(
        self,
        *,
        environment_id: str,
        n: int,
        concurrency: int,
        wall_clock_limit_seconds: int,
        client_job_key: str | None = None,
        visibility: str | None = None,
    ) -> EvalJob:
        body: dict[str, Any] = {
            "environmentId": environment_id,
            "n": n,
            "concurrency": concurrency,
            "wallClockLimitSeconds": wall_clock_limit_seconds,
        }
        if client_job_key is not None:
            body["clientJobKey"] = client_job_key
        # Visibility is sent only for communicate_only; shared_sight (the default,
        # and every non-team run) is omitted, and Rogue Arena stores shared_sight.
        if visibility is not None and visibility != "shared_sight":
            body["visibility"] = visibility
        posted = await self._request(
            "POST", "/hub/eval/jobs", json=body, expected_statuses=(200,)
        )
        return EvalJob.from_dict(_mapping(posted, "job"))

    async def get_job(self, job_id: str) -> EvalJob:
        return EvalJob.from_dict(
            _mapping(await self._request("GET", f"/hub/eval/jobs/{job_id}"), "job")
        )

    async def cancel_job(self, job_id: str) -> EvalJob:
        body = await self._request(
            "POST", f"/hub/eval/jobs/{job_id}/cancel", json={}, expected_statuses=(200,)
        )
        return EvalJob.from_dict(_mapping(body, "job"))

    # --- trials -------------------------------------------------------------

    async def claim_trial(
        self,
        job_id: str,
        *,
        client_attempt_key: str,
        inspect_sample_id: str | None,
        inspect_epoch: int | None,
        inspect_sample_uuid: str | None,
        inspect_eval_id: str | None,
        inspect_run_id: str | None,
        inspect_eval_set_id: str | None,
        inspect_task_id: str | None,
        sandbox_name: str,
        max_wait_seconds: float,
        poll_interval_seconds: float,
    ) -> EvalTrial:
        body = _claim_body(
            job_id,
            client_attempt_key=client_attempt_key,
            inspect_sample_id=inspect_sample_id,
            inspect_epoch=inspect_epoch,
            inspect_sample_uuid=inspect_sample_uuid,
            inspect_eval_id=inspect_eval_id,
            inspect_run_id=inspect_run_id,
            inspect_eval_set_id=inspect_eval_set_id,
            inspect_task_id=inspect_task_id,
            sandbox_name=sandbox_name,
        )
        started = self._monotonic()
        try:
            claimed = await self._post_claim(body, max_wait_seconds)
        except (httpx.TransportError, RogueArenaApiError) as error:
            if not _is_transient(error):
                raise
            # clientAttemptKey makes the claim idempotent: after an unknown
            # outcome (transport failure or 502/503/504) the same body is sent
            # once more, and the hub answers the trial it already opened for
            # this key (created: false), never a second one.
            remaining = max(1.0, max_wait_seconds - (self._monotonic() - started))
            try:
                claimed = await self._post_claim(body, remaining)
            except BaseException as retry_error:
                await self._cancel_unanswered_claim(job_id, client_attempt_key)
                if isinstance(retry_error, Exception) and _is_transient(retry_error):
                    raise error  # still unknown: report the first failure
                raise  # the hub's definite answer, or a cancellation
        try:
            trial = EvalTrial.from_dict(_mapping(claimed.body, "trial claim"))
        except BaseException:
            # The claim landed but its answer does not parse: cancel it by id.
            raw_id = (
                claimed.body.get("id") if isinstance(claimed.body, Mapping) else None
            )
            if isinstance(raw_id, str):
                await self._cancel_unready(raw_id)
            raise
        try:
            return await self._poll_ready(
                trial, started, max_wait_seconds, poll_interval_seconds
            )
        except TrialClosedBeforeReady:
            raise  # the hub already closed it; nothing to cancel
        except BaseException:
            # Any failure after the claim landed (timeout, cancellation, a 404,
            # an unparseable poll answer) cancels the unready trial, so it
            # never holds a billed copy and a concurrency slot unobserved.
            await self._cancel_unready(trial.id)
            raise

    async def _post_claim(self, body: Mapping[str, Any], timeout: float) -> HttpResult:
        try:
            return await self.request_result(
                "POST",
                "/hub/eval/trials",
                json=body,
                expected_statuses=(200,),
                timeout_seconds=timeout,
            )
        except RogueArenaApiError as error:
            if error.status_code in (402, 409) and _looks_like_claim_denial(error.body):
                raise EvalClaimDenied(
                    error.status_code,
                    "/hub/eval/trials",
                    TrialClaimDenied.from_dict(error.body),
                    error.body,
                    secret=self._token,
                ) from error
            raise

    async def _cancel_unanswered_claim(
        self, job_id: str, client_attempt_key: str
    ) -> None:
        """Best effort after a claim whose outcome stayed unknown twice.

        The POST may have opened a trial (a billed copy holding one of the
        job's concurrency slots) whose answer never arrived: find it by its
        attempt key and cancel it. Shielded like the unready cancel.
        """
        trials: tuple[EvalTrial, ...] = ()
        with anyio.move_on_after(_CLOSE_REQUEST_SECONDS, shield=True):
            try:
                trials = await self.list_job_trials(job_id)
            except Exception as error:  # noqa: BLE001 - best-effort cleanup
                _LOGGER.warning(
                    "could not list Rogue Arena job %s trials to recover an "
                    "unanswered claim: %s",
                    job_id,
                    type(error).__name__,
                )
        for trial in trials:
            if (
                trial.client_attempt_key == client_attempt_key
                and trial.status is not EvalTrialStatus.CLOSED
            ):
                _LOGGER.warning(
                    "cancelling Rogue Arena trial %s: its claim was never answered",
                    trial.id,
                )
                await self._cancel_unready(trial.id)

    async def _cancel_unready(self, trial_id: str) -> None:
        # Runs while the claim is being cancelled: shielded, or anyio's
        # level-triggered cancellation would cancel this request at once.
        with anyio.move_on_after(_CLOSE_REQUEST_SECONDS, shield=True):
            try:
                await self.close_trial(trial_id, outcome=EvalTrialOutcome.CANCELLED)
            except Exception as error:  # noqa: BLE001 - best-effort cleanup
                _LOGGER.warning(
                    "best-effort cancel of unready Rogue Arena trial %s failed: %s",
                    trial_id,
                    type(error).__name__,
                )

    async def _poll_ready(
        self,
        trial: EvalTrial,
        started: float,
        max_wait_seconds: float,
        poll_interval_seconds: float,
    ) -> EvalTrial:
        while trial.status is EvalTrialStatus.PROVISIONING:
            elapsed = self._monotonic() - started
            if elapsed >= max_wait_seconds:
                raise TrialReadyTimeout(trial.id, max_wait_seconds)
            await self._sleep(min(poll_interval_seconds, max_wait_seconds - elapsed))
            remaining = max(1.0, max_wait_seconds - (self._monotonic() - started))
            try:
                trial = await self.get_trial(trial.id, timeout_seconds=remaining)
            except (httpx.TransportError, RogueArenaApiError) as error:
                # server-owned trial: a transient 5xx / transport blip retries;
                # a definite state error (409) or foreign id (404) is fatal.
                if (
                    isinstance(error, RogueArenaApiError)
                    and not 500 <= error.status_code < 600
                ):
                    raise
                continue
        if trial.status in (EvalTrialStatus.CLOSED, EvalTrialStatus.CLOSING):
            raise TrialClosedBeforeReady(trial)
        return trial

    async def get_trial(
        self, trial_id: str, *, timeout_seconds: float | None = None
    ) -> EvalTrial:
        body = await self._request(
            "GET", f"/hub/eval/trials/{trial_id}", timeout_seconds=timeout_seconds
        )
        return EvalTrial.from_dict(_mapping(body, "trial"))

    async def list_job_trials(self, job_id: str) -> tuple[EvalTrial, ...]:
        body = await self._request("GET", f"/hub/eval/jobs/{job_id}/trials")
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid trials list")
        return tuple(EvalTrial.from_dict(_mapping(row, "trial")) for row in body)

    async def heartbeat_trial(self, trial_id: str) -> EvalTrial:
        body = await self._request(
            "POST",
            f"/hub/eval/trials/{trial_id}/heartbeat",
            json={},
            expected_statuses=(200,),
        )
        return EvalTrial.from_dict(_mapping(body, "trial"))

    async def close_trial(
        self,
        trial_id: str,
        *,
        outcome: EvalTrialOutcome | None = None,
        max_wait_seconds: float = 120.0,
        poll_interval_seconds: float = 1.0,
    ) -> EvalTrial:
        payload: dict[str, str] = {}
        if outcome is not None:
            if outcome not in (EvalTrialOutcome.CLOSED, EvalTrialOutcome.CANCELLED):
                raise ValueError("close outcome must be closed or cancelled")
            payload["outcome"] = outcome.value
        started = self._monotonic()
        result: HttpResult | None = None
        try:
            with anyio.move_on_after(_CLOSE_REQUEST_SECONDS, shield=True):
                result = await self.request_result(
                    "POST",
                    f"/hub/eval/trials/{trial_id}/close",
                    json=payload,
                    expected_statuses=(200,),
                    timeout_seconds=_CLOSE_REQUEST_SECONDS,
                )
        except (httpx.TransportError, RogueArenaApiError) as error:
            if isinstance(error, RogueArenaApiError) and error.status_code not in (
                502,
                504,
            ):
                raise
        if result is None:
            # The server close is idempotent and finishes server-side: after a
            # gateway timeout never resend it, read the trial until closed.
            return await self._await_closed(
                trial_id, started, max_wait_seconds, poll_interval_seconds
            )
        trial = EvalTrial.from_dict(_mapping(result.body, "trial"))
        if trial.status is EvalTrialStatus.CLOSING:
            return await self._await_closed(
                trial_id, started, max_wait_seconds, poll_interval_seconds
            )
        return trial

    async def _await_closed(
        self,
        trial_id: str,
        started: float,
        max_wait_seconds: float,
        poll_interval_seconds: float,
    ) -> EvalTrial:
        while True:
            elapsed = self._monotonic() - started
            if elapsed >= max_wait_seconds:
                raise TimeoutError(
                    f"Rogue Arena trial {trial_id} did not finish closing within "
                    f"{max_wait_seconds:g} seconds"
                )
            await self._sleep(min(poll_interval_seconds, max_wait_seconds - elapsed))
            remaining = max(1.0, max_wait_seconds - (self._monotonic() - started))
            try:
                trial = await self.get_trial(trial_id, timeout_seconds=remaining)
            except (httpx.TransportError, RogueArenaApiError) as error:
                if (
                    isinstance(error, RogueArenaApiError)
                    and not 500 <= error.status_code < 600
                ):
                    raise
                continue
            if trial.status is EvalTrialStatus.CLOSED:
                return trial

    async def list_trial_events(self, trial_id: str) -> tuple[EvalTrialEvent, ...]:
        body = await self._request("GET", f"/hub/eval/trials/{trial_id}/events")
        if not isinstance(body, list):
            raise TypeError("Rogue Arena returned an invalid trial event list")
        return tuple(
            EvalTrialEvent.from_dict(_mapping(row, "trial event")) for row in body
        )

    async def exec_trial(
        self,
        trial_id: str,
        *,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        stdin: str | None,
        timeout_seconds: int,
        background: bool = False,
        poll_exec_id: str | None = None,
    ) -> EvalExecResult:
        """Run one exec request against the trial's agent VM.

        With ``background=True`` the server starts the command and returns
        ``running=True`` plus an ``execId``; poll it by passing ``poll_exec_id``
        (with a placeholder ``cmd``). A lost exec is reported as finished with
        exit code 255 and a lost-exec detail.
        """
        body: dict[str, Any] = {
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "stdin": stdin,
            "timeoutSeconds": timeout_seconds,
            "background": background,
        }
        if poll_exec_id is not None:
            body["pollExecId"] = poll_exec_id
        try:
            result = await self.request_result(
                "POST",
                f"/hub/eval/trials/{trial_id}/exec",
                json=body,
                expected_statuses=(200,),
                timeout_seconds=min(timeout_seconds + 5, 60),
            )
        except (httpx.TransportError, RogueArenaApiError) as error:
            if poll_exec_id is None and _is_transient(error):
                raise ExecOutcomeUnknown(trial_id, cmd, type(error).__name__) from error
            raise
        parsed = EvalExecResult.from_dict(_mapping(result.body, "exec result"))
        if not background and poll_exec_id is None and parsed.running:
            raise RuntimeError(
                "Rogue Arena sync exec returned running=true; outcome unknown for "
                f"trial {trial_id}"
            )
        return parsed

    async def run_command(
        self,
        trial_id: str,
        *,
        cmd: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        stdin: str | None,
        timeout_seconds: int,
        sync_ceiling_seconds: int,
        poll_interval_seconds: float,
        max_total_seconds: int,
    ) -> EvalExecResult:
        """Run synchronously up to the ceiling, else background and poll.

        Timeouts above ``sync_ceiling_seconds`` start a background exec and
        poll it until finished or ``max_total_seconds`` (the 4 h server
        ceiling).
        """
        if timeout_seconds <= sync_ceiling_seconds:
            return await self.exec_trial(
                trial_id,
                cmd=cmd,
                cwd=cwd,
                env=env,
                stdin=stdin,
                timeout_seconds=timeout_seconds,
                background=False,
            )
        started = self._monotonic()
        launched = await self.exec_trial(
            trial_id,
            cmd=cmd,
            cwd=cwd,
            env=env,
            stdin=stdin,
            timeout_seconds=min(timeout_seconds, max_total_seconds),
            background=True,
        )
        if launched.exec_id is None:
            raise RuntimeError("Rogue Arena background exec returned no execId")
        ceiling = min(max_total_seconds, timeout_seconds + 300)
        # A running exec is polled after _FIRST_POLL_SECONDS, doubling up to
        # poll_interval_seconds. Consecutive transient poll failures back off
        # exponentially from poll_interval_seconds (capped); a successful poll
        # resets that backoff.
        backoff_cap = max(_POLL_BACKOFF_CAP_SECONDS, poll_interval_seconds)
        cadence = min(_FIRST_POLL_SECONDS, poll_interval_seconds)
        failures = 0
        last_error: Exception | None = None
        while True:
            if self._monotonic() - started >= ceiling:
                raise TimeoutError(
                    f"Rogue Arena background exec {launched.exec_id} exceeded {ceiling}s"
                ) from last_error
            await self._sleep(
                min(poll_interval_seconds * 2**failures, backoff_cap)
                if failures
                else cadence
            )
            try:
                polled = await self.exec_trial(
                    trial_id,
                    cmd=["true"],  # a poll ignores cmd; send a placeholder
                    cwd=None,
                    env=None,
                    stdin=None,
                    timeout_seconds=30,
                    background=False,
                    poll_exec_id=launched.exec_id,
                )
            except (httpx.TransportError, RogueArenaApiError) as error:
                if _is_transient(error):
                    # polls are read-only and idempotent; retry to the deadline
                    failures += 1
                    last_error = error
                    continue
                raise
            failures, last_error = 0, None
            if not polled.running:
                return polled
            cadence = min(cadence * 2, poll_interval_seconds)

    async def upload_file(self, trial_id: str, path: str, content: bytes) -> None:
        await self._request(
            "POST",
            f"/hub/eval/trials/{trial_id}/upload",
            json={
                "path": path,
                "contentBase64": base64.b64encode(content).decode("ascii"),
            },
            expected_statuses=(200,),
        )

    async def download_file(self, trial_id: str, path: str) -> bytes:
        body = _mapping(
            await self._request(
                "GET", f"/hub/eval/trials/{trial_id}/download", params={"path": path}
            ),
            "download",
        )
        try:
            return base64.b64decode(
                _str(body["contentBase64"], "contentBase64"), validate=True
            )
        except binascii.Error as error:
            raise RuntimeError(
                f"Rogue Arena returned invalid base64 for download of {path}"
            ) from error

    @classmethod
    def from_env(
        cls, token: str | None = None, *, require_token: bool = True
    ) -> RogueArenaClient:
        base = os.environ.get("ROGUE_ARENA_BASE_URL", "").rstrip("/")
        auth = token or os.environ.get("ROGUE_ARENA_TOKEN", "")
        if not base:
            raise RuntimeError("ROGUE_ARENA_BASE_URL is required")
        if require_token and not auth:
            raise RuntimeError(
                "ROGUE_ARENA_TOKEN is required for the Rogue Arena eval API"
            )
        return cls(base, auth or None)
