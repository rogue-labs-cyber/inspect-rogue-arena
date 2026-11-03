from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time
import unittest
import uuid
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import httpx
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import (
    OutputLimitExceededError,
    SandboxEnvironmentSpec,
    SandboxUnavailableError,
    override_sandbox_output_limit,
    sandbox,
)

from inspect_rogue_arena.client import (
    EvalCatalogEntry,
    EvalClaimDenied,
    EvalTrialOutcome,
    EvalTrialTerminalReason,
    ExecOutcomeUnknown,
    RogueArenaApiError,
    RogueArenaClient,
    TrialClosedBeforeReady,
    TrialReadyTimeout,
)
from inspect_rogue_arena.config import RogueArenaSandboxConfig
from inspect_rogue_arena.sandbox import (
    RogueArenaSandboxEnvironment,
    _require_sample_id,
    _TaskRuntime,
    select_eval,
)
from inspect_rogue_arena.scorer import rogue_arena_hops

BASE_URL = "https://hub.example.test"
TOKEN = "ra_eval_TOKEN"
ENV_ID = "11111111-1111-4111-8111-111111111111"
BUILD_ID = "22222222-2222-4222-8222-222222222222"
EVAL_NAME = "Example AD Chain"
LIVE = ("ready", "running")
LOST_STDERR = (
    "exec lost: no exit status was recorded within the exec max age"  # hub wording
)


def _entry(environment_id: str, name: str) -> dict[str, Any]:
    return {
        "environmentId": environment_id,
        "name": name,
        "environmentBuildId": BUILD_ID,
        "canvasVersionNumber": 3,
        "hopCount": 2,
        "agentMachineNickname": "agent-vm",
        "scoringMode": "sequential",
    }


def _score() -> dict[str, Any]:
    return {
        "reachedStep": 1,
        "hopsPassed": 1,
        "totalSteps": 2,
        "scoringMode": "sequential",
        "steps": [
            {
                "stepNumber": 1,
                "name": "foothold",
                "passed": True,
                "passedAt": "2026-09-30T12:00:00Z",
                "detail": "canary seen",
            },
            {
                "stepNumber": 2,
                "name": "domain admin",
                "passed": False,
                "passedAt": None,
                "detail": "not reached",
            },
        ],
    }


async def _no_wait(_: float) -> None:
    await asyncio.sleep(0)


async def _until(predicate: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


class FakeHub:
    """An in-memory eval API (the shipped wire shapes) behind MockTransport.

    Trials follow the hub's state machine: a claim is `provisioning` until the
    first GET moves it to `becomes`; heartbeat/exec/upload/download require a
    live (`ready`/`running`) trial and answer 409 otherwise, exactly like the
    hub's live-trial check. A claim is idempotent on `clientAttemptKey`: a
    repeated key answers the trial already opened for it (the hub's
    `created: false`), never a second one.
    """

    def __init__(self, test: unittest.TestCase) -> None:
        self._test = test
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self.entries: list[dict[str, Any]] = [_entry(ENV_ID, EVAL_NAME)]
        self.jobs: dict[str, dict[str, Any]] = {}
        # The per-run team chatroom the hub owns, keyed by job id.
        self.chat: dict[str, list[dict[str, Any]]] = {}
        # The per-run team activity feed the hub owns, keyed by job id.
        self.activity: dict[str, list[dict[str, Any]]] = {}
        # A hub without the feed answers the activity route with 404.
        self.activity_supported = True
        self.status: dict[str, str] = {}
        self.outcome: dict[str, tuple[str | None, str | None]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.claim_status = "provisioning"
        self.becomes = "ready"
        self.becomes_outcome: tuple[str | None, str | None] = (None, None)
        self.claim_denial: tuple[int, dict[str, Any]] | None = None
        # Each entry loses one claim answer after the hub opened the trial:
        # None as a transport error, an int as that gateway status.
        self.claim_lost: list[int | None] = []
        self.trial_for_key: dict[str, str] = {}
        self.key_for_trial: dict[str, str] = {}
        self.heartbeat_status: str | None = None
        # The hub closes (and scores) the trial with this outcome and reason.
        self.close_as: tuple[str, str] | None = None
        self.scored_on_close: set[str] = set()
        self.job_post_failures = 0
        self.exec_responses: list[tuple[int, dict[str, Any]]] = []
        # Transient ReadTimeouts to raise on the request that carries a command's
        # result (a sync exec, or a background exec's poll) before answering it.
        # A background launch is never failed: it returns immediately, so only a
        # sync exec or a poll waits on the command and can time out.
        self.exec_result_transport_failures = 0
        self.heartbeat_conflict = False
        self.close_gateway_timeout = False
        self.close_delay = 0.0
        self.close_malformed: set[str] = set()
        self.open_trials = 0
        self.peak_open_trials = 0

    # --- helpers ------------------------------------------------------------

    def client(self) -> RogueArenaClient:
        return RogueArenaClient(
            BASE_URL, TOKEN, transport=httpx.MockTransport(self.handler), sleep=_no_wait
        )

    def calls(self, method: str, path: str) -> list[dict[str, Any]]:
        return [body for m, p, body in self.requests if m == method and p == path]

    def order(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _ in self.requests]

    def count_suffix(self, method: str, suffix: str) -> int:
        return sum(1 for m, p, _ in self.requests if m == method and p.endswith(suffix))

    def trial(self, trial_id: str) -> dict[str, Any]:
        status = self.status[trial_id]
        outcome, reason = self.outcome.get(trial_id, (None, None))
        scored = status == "closed" and (
            outcome == "closed" or trial_id in self.scored_on_close
        )
        return {
            "id": trial_id,
            "jobId": "job-1",
            "deploymentRecordId": None if status == "provisioning" else "dep-1",
            "agentDeployedVmId": None if status == "provisioning" else "vm-1",
            "agentMachineNickname": "agent-vm",
            "status": status,
            "outcome": outcome,
            "terminalReason": reason,
            "warm": False,
            "reachedStep": 1 if scored else None,
            "hopsPassed": 1 if scored else None,
            "totalSteps": 2,
            "clientKind": "inspect",
            "clientAttemptKey": self.key_for_trial.get(trial_id),
            "score": _score() if scored else None,
            "hopResults": None,
        }

    @staticmethod
    def _job(job_id: str, body: dict[str, Any], status: str) -> dict[str, Any]:
        return {
            "id": job_id,
            "environmentId": body["environmentId"],
            "environmentBuildId": BUILD_ID,
            "canvasVersionId": "cv-1",
            "ownerUserMainId": "user-1",
            "organizationId": None,
            "n": body["n"],
            "concurrency": body["concurrency"],
            "wallClockLimitSeconds": body["wallClockLimitSeconds"],
            "clientJobKey": body.get("clientJobKey"),
            "status": status,
            "trialCount": 0,
            "closedTrialCount": 0,
        }

    @staticmethod
    def _conflict(message: str) -> httpx.Response:
        return httpx.Response(409, json={"errorMessage": message, "errorSeverity": 2})

    # --- transport ------------------------------------------------------------

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self._test.assertEqual(request.headers["Authorization"], f"Bearer {TOKEN}")
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        method, path = request.method, request.url.path
        self.requests.append((method, path, body))
        if self.close_delay and path.endswith("/close"):
            # The close lands only once the request completes: a client that
            # abandons it mid-flight never closes the trial.
            await asyncio.sleep(self.close_delay)
        return self._route(request, method, path, body)

    def _route(
        self, request: httpx.Request, method: str, path: str, body: dict[str, Any]
    ) -> httpx.Response:
        parts = path.split("/")

        if (method, path) == ("GET", "/hub/eval/evals"):
            return httpx.Response(200, json=self.entries)
        if (method, path) == ("POST", "/hub/eval/jobs"):
            if self.job_post_failures > 0:
                self.job_post_failures -= 1
                raise httpx.ConnectError("connection reset", request=request)
            for job in self.jobs.values():
                if job["clientJobKey"] == body.get("clientJobKey"):
                    return httpx.Response(200, json=job)
            job_id = f"job-{len(self.jobs) + 1}"
            self.jobs[job_id] = self._job(job_id, body, "open")
            return httpx.Response(200, json=self.jobs[job_id])
        if (
            method == "POST"
            and path.startswith("/hub/eval/jobs/")
            and parts[-1] == ("cancel")
        ):
            job = {**self.jobs[parts[4]], "status": "cancelled"}
            self.jobs[parts[4]] = job
            return httpx.Response(200, json=job)
        if path.startswith("/hub/eval/jobs/") and parts[-1] == "chat":
            job_id = parts[4]
            room = self.chat.setdefault(job_id, [])
            if method == "POST":
                seq = len(room) + 1
                message = {
                    "id": f"chat-{job_id}-{seq}",
                    "jobId": job_id,
                    "seq": seq,
                    "at": f"2026-09-30T00:00:{seq:02d}Z",
                    "author": body["author"],
                    "content": body["content"],
                }
                room.append(message)
                return httpx.Response(200, json=message)
            since = int(request.url.params.get("since", 0) or 0)
            return httpx.Response(200, json=[row for row in room if row["seq"] > since])
        if path.startswith("/hub/eval/jobs/") and parts[-1] == "activity":
            if not self.activity_supported:
                return httpx.Response(
                    404, json={"errorMessage": "not found", "errorSeverity": 2}
                )
            job_id = parts[4]
            feed = self.activity.setdefault(job_id, [])
            if method == "POST":
                seq = len(feed) + 1
                entry = {
                    "id": f"activity-{job_id}-{seq}",
                    "jobId": job_id,
                    "seq": seq,
                    "at": f"2026-09-30T01:00:{seq:02d}Z",
                    "member": body["member"],
                    "command": body["command"],
                    "summary": body["summary"],
                }
                feed.append(entry)
                return httpx.Response(200, json=entry)
            since = int(request.url.params.get("since", 0) or 0)
            return httpx.Response(200, json=[row for row in feed if row["seq"] > since])
        if (method, path) == ("POST", "/hub/eval/trials"):
            if self.claim_denial is not None:
                return httpx.Response(self.claim_denial[0], json=self.claim_denial[1])
            key = body["clientAttemptKey"]
            trial_id = self.trial_for_key.get(key)
            if trial_id is None:
                trial_id = f"trial-{len(self.trial_for_key) + 1}"
                self.trial_for_key[key] = trial_id
                self.key_for_trial[trial_id] = key
                self.status[trial_id] = self.claim_status
                self.open_trials += 1
                self.peak_open_trials = max(self.peak_open_trials, self.open_trials)
            if self.claim_lost:
                lost = self.claim_lost.pop(0)
                if lost is None:
                    raise httpx.ReadError("connection reset", request=request)
                return httpx.Response(lost, text="Gateway Time-out")
            return httpx.Response(200, json=self.trial(trial_id))
        if (
            method == "GET"
            and path.startswith("/hub/eval/jobs/")
            and parts[-1] == "trials"
        ):
            return httpx.Response(
                200, json=[self.trial(trial_id) for trial_id in self.status]
            )

        trial_id = parts[4]
        action = parts[5] if len(parts) > 5 else None
        if method == "GET" and action is None:
            if self.status[trial_id] == "provisioning":
                self.status[trial_id] = self.becomes
                self.outcome[trial_id] = self.becomes_outcome
            return httpx.Response(200, json=self.trial(trial_id))
        if action == "close":
            if trial_id in self.close_malformed:
                return httpx.Response(200, json={"id": trial_id})
            if self.status[trial_id] == "closed":
                return httpx.Response(200, json=self.trial(trial_id))
            self.status[trial_id] = "closed"
            self.open_trials -= 1
            if self.close_gateway_timeout:
                # the server finishes the close; the gateway gives up first
                self.outcome[trial_id] = ("reaped", "timeout")
                return httpx.Response(504, text="Gateway Time-out")
            outcome = body.get("outcome", "closed")
            reason = "cancelled" if outcome == "cancelled" else "agent_fail"
            if self.close_as is not None:
                outcome, reason = self.close_as
                self.scored_on_close.add(trial_id)
            self.outcome[trial_id] = (outcome, reason)
            return httpx.Response(200, json=self.trial(trial_id))
        if self.status[trial_id] not in LIVE:
            return self._conflict("Trial is not running")
        if action == "heartbeat":
            if self.heartbeat_conflict:
                return self._conflict("Trial lease has expired")
            self.status[trial_id] = "running"
            if self.heartbeat_status is not None:
                # a 200 whose trial carries a status this client does not know
                return httpx.Response(
                    200,
                    json={**self.trial(trial_id), "status": self.heartbeat_status},
                )
            return httpx.Response(200, json=self.trial(trial_id))
        if action == "exec":
            carries_command_result = not body.get("background")
            if carries_command_result and self.exec_result_transport_failures > 0:
                self.exec_result_transport_failures -= 1
                raise httpx.ReadTimeout("exec result slow", request=request)
            if self.exec_responses:
                status_code, payload = self.exec_responses.pop(0)
                return httpx.Response(status_code, json=payload)
            finished = {
                "running": False,
                "execId": None,
                "exitCode": 0,
                "stdout": "uid=0(root) gid=0(root)\n",
                "stderr": "",
                "timedOut": False,
            }
            # Like the hub: a background launch answers running with an execId,
            # and a poll of that id answers its finished result.
            if body.get("pollExecId"):
                return httpx.Response(
                    200, json={**finished, "execId": body["pollExecId"]}
                )
            if body.get("background"):
                exec_id = str(uuid.uuid4())
                return httpx.Response(
                    200,
                    json={
                        **finished,
                        "running": True,
                        "execId": exec_id,
                        "exitCode": None,
                        "stdout": "",
                    },
                )
            return httpx.Response(200, json=finished)
        if action == "upload":
            content = base64.b64decode(body["contentBase64"], validate=True)
            self.files[(trial_id, body["path"])] = content
            return httpx.Response(200, json={"ok": True})
        if action == "download":
            file_path = request.url.params["path"]
            content = self.files[(trial_id, file_path)]
            return httpx.Response(
                200,
                json={
                    "path": file_path,
                    "contentBase64": base64.b64encode(content).decode("ascii"),
                },
            )
        return httpx.Response(500, json={"errorMessage": f"unexpected {path}"})


def _config(**overrides: Any) -> RogueArenaSandboxConfig:
    values: dict[str, Any] = {
        "eval": EVAL_NAME,
        "max_trials": 2,
        "concurrency": 2,
        "max_runtime_seconds": 600,
        "max_wait_seconds": 30,
        "poll_interval_seconds": 1,
    }
    values.update(overrides)
    return RogueArenaSandboxConfig(**values)


def _only_runtime() -> _TaskRuntime:
    runtimes = list(RogueArenaSandboxEnvironment._runtimes.values())
    assert len(runtimes) == 1
    return runtimes[0]


def _env(
    environments: dict[str, Any],
) -> RogueArenaSandboxEnvironment:
    environment = environments["default"]
    assert isinstance(environment, RogueArenaSandboxEnvironment)
    return environment


class RogueArenaSandboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        RogueArenaSandboxEnvironment._runtimes.clear()
        env_patch = patch.dict(os.environ, {}, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.hub = FakeHub(self)
        from_env = patch.object(
            RogueArenaClient, "from_env", return_value=self.hub.client()
        )
        from_env.start()
        self.addCleanup(from_env.stop)

    def tearDown(self) -> None:
        RogueArenaSandboxEnvironment._runtimes.clear()

    async def _sample(
        self, config: RogueArenaSandboxConfig, sample_id: str, task: str = "task"
    ) -> dict[str, Any]:
        return await RogueArenaSandboxEnvironment.sample_init(
            task, config, {"__sample_id__": sample_id}
        )

    # --- the whole lifecycle through one MockTransport -------------------------

    async def test_one_job_two_trials_exec_files_close_and_job_cancel(self) -> None:
        config = _config()
        with self.assertLogs("inspect_rogue_arena", level="INFO") as logs:
            await RogueArenaSandboxEnvironment.task_init("startup", config)
            first = await RogueArenaSandboxEnvironment.sample_init(
                "model-task", config, {"__sample_id__": "inspect-1", "sample_id": "u-1"}
            )
            second = await RogueArenaSandboxEnvironment.sample_init(
                "model-task", config, {"__sample_id__": "inspect-2", "sample_id": "u-2"}
            )
            env1, env2 = _env(first), _env(second)

            result = await env1.exec(["id"])
            await env1.write_file("/tmp/blob.bin", b"\x00\xffbinary\n")
            blob = await env1.read_file("/tmp/blob.bin", text=False)
            await env1.write_file("notes.txt", "héllo\r\n")
            notes = await env1.read_file("notes.txt")

            await RogueArenaSandboxEnvironment.sample_cleanup(
                "model-task", config, first, False
            )
            await RogueArenaSandboxEnvironment.sample_cleanup(
                "model-task", config, second, True
            )
            await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)

        # exactly one job carrying exactly the hub's strict job-create body keys
        jobs = self.hub.calls("POST", "/hub/eval/jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            set(jobs[0]),
            {
                "environmentId",
                "n",
                "concurrency",
                "wallClockLimitSeconds",
                "clientJobKey",
            },
        )
        # shared_sight (the default) is never sent.
        self.assertNotIn("visibility", jobs[0])
        self.assertEqual(jobs[0]["environmentId"], ENV_ID)
        self.assertEqual(
            (jobs[0]["n"], jobs[0]["concurrency"], jobs[0]["wallClockLimitSeconds"]),
            (2, 2, 600),
        )
        self.assertTrue(str(jobs[0]["clientJobKey"]).startswith("inspect_"))

        # two claims keyed by Inspect's reserved sample id, never user metadata
        claims = self.hub.calls("POST", "/hub/eval/trials")
        self.assertEqual(
            [claim["inspectSampleId"] for claim in claims], ["inspect-1", "inspect-2"]
        )
        for claim in claims:
            self.assertEqual(claim["jobId"], "job-1")
            self.assertEqual(claim["clientKind"], "inspect")
            self.assertEqual(claim["sandboxName"], EVAL_NAME)
            self.assertEqual(claim["inspectTaskId"], "model-task")
            self.assertNotIn("u-", claim["clientAttemptKey"])
        self.assertNotEqual(
            claims[0]["clientAttemptKey"], claims[1]["clientAttemptKey"]
        )
        self.assertEqual(claims[0]["inspectRunId"], claims[1]["inspectRunId"])

        # the provisioning claim was polled to ready before any command ran
        order = self.hub.order()
        claimed = order.index(("POST", "/hub/eval/trials"))
        ready = order.index(("GET", "/hub/eval/trials/trial-1"))
        executed = order.index(("POST", "/hub/eval/trials/trial-1/exec"))
        self.assertLess(claimed, ready)
        self.assertLess(ready, executed)
        self.assertLess(
            order.index(("POST", "/hub/eval/trials/trial-1/heartbeat")), executed
        )

        # exec is one sync run_command call; success maps from exitCode == 0
        execs = self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertEqual(
            execs,
            [
                {
                    "cmd": ["id"],
                    "cwd": None,
                    "env": None,
                    "stdin": None,
                    "timeoutSeconds": config.sync_exec_ceiling_seconds,
                    "background": False,
                }
            ],
        )
        self.assertTrue(result.success)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "uid=0(root) gid=0(root)\n")

        # files round-trip raw bytes as base64; relative paths become absolute
        self.assertEqual(blob, b"\x00\xffbinary\n")
        self.assertEqual(notes, "héllo\r\n")
        uploads = self.hub.calls("POST", "/hub/eval/trials/trial-1/upload")
        self.assertEqual(
            uploads[0]["contentBase64"],
            base64.b64encode(b"\x00\xffbinary\n").decode("ascii"),
        )
        self.assertEqual(uploads[1]["path"], "/notes.txt")

        # normal cleanup closes `closed`, interrupted closes `cancelled`
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "closed"}],
        )
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-2/close"),
            [{"outcome": "cancelled"}],
        )
        closed1, closed2 = env1.closed_trial, env2.closed_trial
        assert closed1 is not None and closed2 is not None
        self.assertEqual(closed1.outcome, EvalTrialOutcome.CLOSED)
        self.assertEqual(closed1.terminal_reason, EvalTrialTerminalReason.AGENT_FAIL)
        assert closed1.score is not None
        self.assertEqual((closed1.score.hops_passed, closed1.score.total_steps), (1, 2))
        self.assertEqual(closed2.outcome, EvalTrialOutcome.CANCELLED)
        self.assertEqual(closed2.terminal_reason, EvalTrialTerminalReason.CANCELLED)

        # task cleanup cancels the job once, after the trials closed
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/jobs/job-1/cancel")), 1)
        self.assertGreater(
            order.index(("POST", "/hub/eval/jobs/job-1/cancel")),
            order.index(("POST", "/hub/eval/trials/trial-2/close")),
        )
        self.assertEqual(self.hub.count_suffix("POST", "/close"), 2)
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})
        self.assertNotIn(TOKEN, "\n".join(logs.output))

    # --- concurrency -------------------------------------------------------

    async def test_concurrency_semaphore_holds_a_claim_until_a_trial_closes(
        self,
    ) -> None:
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        first = await self._sample(config, "s-1")
        second_task = asyncio.create_task(self._sample(config, "s-2"))
        for _ in range(50):
            await asyncio.sleep(0)
        self.assertFalse(second_task.done())
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/trials")), 1)

        await RogueArenaSandboxEnvironment.sample_cleanup("task", config, first, False)
        second = await asyncio.wait_for(second_task, timeout=2)
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/trials")), 2)
        await RogueArenaSandboxEnvironment.sample_cleanup("task", config, second, False)
        self.assertFalse(_only_runtime().trial_slots.locked())

    # --- heartbeat -----------------------------------------------------------

    async def test_heartbeat_runs_at_the_configured_interval_and_stops_before_close(
        self,
    ) -> None:
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            if len(sleeps) > 3:
                await asyncio.Event().wait()  # park the loop until it is cancelled
            await asyncio.sleep(0)

        config = _config(heartbeat_interval_seconds=120)
        with patch("inspect_rogue_arena.sandbox._sleep", fake_sleep):
            await RogueArenaSandboxEnvironment.task_init("startup", config)
            environments = await self._sample(config, "s-1")
            loop = _env(environments)._heartbeat_task
            assert loop is not None
            await _until(
                lambda: (
                    self.hub.count_suffix("POST", "/heartbeat") == 3
                    and len(sleeps) == 4
                )
            )
            await RogueArenaSandboxEnvironment.sample_cleanup(
                "task", config, environments, False
            )

        self.assertEqual(sleeps, [120.0] * 4)
        self.assertTrue(loop.done())
        order = self.hub.order()
        close = order.index(("POST", "/hub/eval/trials/trial-1/close"))
        beats = [
            index
            for index, (_, path) in enumerate(order)
            if path.endswith("/heartbeat")
        ]
        self.assertEqual(len(beats), 3)
        self.assertLess(max(beats), close)

    # --- exec mapping --------------------------------------------------------

    async def _live_env(
        self, config: RogueArenaSandboxConfig | None = None
    ) -> tuple[RogueArenaSandboxConfig, RogueArenaSandboxEnvironment]:
        resolved = config or _config()
        await RogueArenaSandboxEnvironment.task_init("startup", resolved)
        environment = _env(await self._sample(resolved, "s-1"))
        self.addAsyncCleanup(environment._close, EvalTrialOutcome.CANCELLED)
        return resolved, environment

    async def test_exec_maps_exit_code_stdin_and_command_timeout(self) -> None:
        self.hub.exec_responses = [
            (
                200,
                {
                    "running": False,
                    "execId": None,
                    "exitCode": 3,
                    "stdout": "",
                    "stderr": "boom\n",
                    "timedOut": False,
                },
            ),
            (
                200,
                {
                    "running": False,
                    "execId": None,
                    "exitCode": None,
                    "stdout": "partial",
                    "stderr": "",
                    "timedOut": True,
                },
            ),
        ]
        _, environment = await self._live_env()

        failed = await environment.exec(
            ["sh", "-c", "exit 3"], input=b"data\n", cwd="/root", env={"A": "1"}
        )
        self.assertFalse(failed.success)
        self.assertEqual((failed.returncode, failed.stderr), (3, "boom\n"))
        sent = self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")[0]
        self.assertEqual(
            (sent["stdin"], sent["cwd"], sent["env"]), ("data\n", "/root", {"A": "1"})
        )

        with self.assertRaisesRegex(TimeoutError, "timed out after 20 seconds") as ctx:
            await environment.exec(["sleep", "99"], timeout=20)
        self.assertEqual(getattr(ctx.exception, "truncated_output", None), "partial")
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")[1][
                "timeoutSeconds"
            ],
            20,
        )

    async def test_exec_above_the_sync_ceiling_runs_background_and_polls(self) -> None:
        self.hub.exec_responses = [
            (
                200,
                {
                    "running": True,
                    "execId": "33333333-3333-4333-8333-333333333333",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            ),
            (
                200,
                {
                    "running": False,
                    "execId": "33333333-3333-4333-8333-333333333333",
                    "exitCode": 0,
                    "stdout": "built\n",
                    "stderr": "",
                    "timedOut": False,
                },
            ),
        ]
        config, environment = await self._live_env()

        result = await environment.exec(["make"], timeout=600)

        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "built\n")
        launch, poll = self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertTrue(launch["background"])
        self.assertEqual(launch["timeoutSeconds"], 600)
        self.assertGreater(600, config.sync_exec_ceiling_seconds)
        self.assertEqual(poll["pollExecId"], "33333333-3333-4333-8333-333333333333")
        self.assertEqual(poll["cmd"], ["true"])

    async def test_a_trial_the_server_no_longer_runs_is_sandbox_unavailable(
        self,
    ) -> None:
        self.hub.exec_responses = [
            (409, {"errorMessage": "Trial wall clock has run out", "errorSeverity": 2}),
            (
                200,
                {
                    "running": False,
                    "execId": "33333333-3333-4333-8333-333333333333",
                    "exitCode": 255,
                    "stdout": "",
                    "stderr": f"\n{LOST_STDERR}",
                    "timedOut": False,
                },
            ),
        ]
        _, environment = await self._live_env()

        with self.assertRaisesRegex(SandboxUnavailableError, "wall clock has run out"):
            await environment.exec(["id"])
        with self.assertRaisesRegex(SandboxUnavailableError, "lost"):
            await environment.exec(["id"])

        self.hub.heartbeat_conflict = True
        before = len(self.hub.calls("POST", "/hub/eval/trials/trial-1/exec"))
        with self.assertRaisesRegex(SandboxUnavailableError, "lease has expired"):
            await environment.exec(["id"])
        self.assertEqual(
            len(self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")), before
        )

    async def test_closed_or_expired_sandbox_sends_nothing(self) -> None:
        config, environment = await self._live_env()
        environment._deadline = time.monotonic() - 1
        sent = len(self.hub.requests)
        with self.assertRaisesRegex(SandboxUnavailableError, "max_runtime_seconds"):
            await environment.exec(["id"])
        with self.assertRaisesRegex(SandboxUnavailableError, "max_runtime_seconds"):
            await environment.write_file("/tmp/x", b"x")
        with self.assertRaisesRegex(SandboxUnavailableError, "max_runtime_seconds"):
            await environment.read_file("/tmp/x")
        self.assertEqual(len(self.hub.requests), sent)

        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, {"default": environment}, False
        )
        sent = len(self.hub.requests)
        with self.assertRaisesRegex(SandboxUnavailableError, "closed"):
            await environment.exec(["id"])
        self.assertEqual(len(self.hub.requests), sent)

    async def test_read_file_over_the_inspect_limit_raises_output_limit(self) -> None:
        _, environment = await self._live_env()
        await environment.write_file("/tmp/big", b"12345")
        with (
            override_sandbox_output_limit(4, "read_file"),
            self.assertRaises(OutputLimitExceededError),
        ):
            await environment.read_file("/tmp/big", text=False)
        self.assertEqual(await environment.read_file("/tmp/big", text=False), b"12345")

    # --- claim failures surface clearly ---------------------------------------

    async def test_claim_402_denial_propagates_from_sample_init(self) -> None:
        self.hub.claim_denial = (
            402,
            {
                "errorMessage": "Your balance cannot fund this eval",
                "decision": {
                    "verdict": "deny",
                    "reasonCode": "insufficient_funds",
                },
            },
        )
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with (
            self.assertLogs("inspect_rogue_arena.sandbox", level="ERROR") as logs,
            self.assertRaises(EvalClaimDenied) as ctx,
        ):
            await self._sample(config, "s-1")

        self.assertTrue(ctx.exception.is_funding)
        self.assertEqual(ctx.exception.denial.reason_code, "insufficient_funds")
        self.assertIn("insufficient_funds", "\n".join(logs.output))
        runtime = _only_runtime()
        self.assertEqual(runtime.environments, set())
        self.assertFalse(runtime.trial_slots.locked())

        # the released slot lets the next sample claim at once
        self.hub.claim_denial = None
        environments = await asyncio.wait_for(self._sample(config, "s-2"), timeout=2)
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, environments, False
        )

    async def test_trial_closed_before_ready_propagates_from_sample_init(self) -> None:
        self.hub.becomes = "closed"
        self.hub.becomes_outcome = ("expired", "infra_error")
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with self.assertRaisesRegex(
            TrialClosedBeforeReady, "before it was ready"
        ) as ctx:
            await self._sample(config, "s-1")

        self.assertEqual(ctx.exception.trial.outcome, EvalTrialOutcome.EXPIRED)
        self.assertEqual(
            ctx.exception.trial.terminal_reason, EvalTrialTerminalReason.INFRA_ERROR
        )
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/trials")), 1)
        self.assertEqual(self.hub.count_suffix("POST", "/exec"), 0)
        runtime = _only_runtime()
        self.assertEqual(runtime.environments, set())
        self.assertFalse(runtime.trial_slots.locked())

    # --- close semantics ---------------------------------------------------------

    async def test_scorer_close_before_cleanup_is_kept_and_not_resent(self) -> None:
        config, environment = await self._live_env()

        scored = await environment.close_trial()
        again = await environment.close_trial()
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, {"default": environment}, True
        )

        self.assertIs(again, scored)
        self.assertIs(environment.closed_trial, scored)
        self.assertEqual(scored.outcome, EvalTrialOutcome.CLOSED)
        self.assertEqual(scored.terminal_reason, EvalTrialTerminalReason.AGENT_FAIL)
        assert scored.score is not None
        self.assertEqual(scored.score.hops_passed, 1)
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "closed"}],
        )
        self.assertFalse(_only_runtime().trial_slots.locked())

    async def test_close_after_gateway_timeout_reads_back_and_is_never_resent(
        self,
    ) -> None:
        self.hub.close_gateway_timeout = True
        config, environment = await self._live_env()

        with self.assertLogs("inspect_rogue_arena.sandbox", level="WARNING") as logs:
            await RogueArenaSandboxEnvironment.sample_cleanup(
                "task", config, {"default": environment}, False
            )
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, {"default": environment}, False
        )
        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)

        closed = environment.closed_trial
        assert closed is not None
        self.assertEqual(closed.outcome, EvalTrialOutcome.REAPED)
        self.assertEqual(closed.terminal_reason, EvalTrialTerminalReason.TIMEOUT)
        self.assertIn("reaped", "\n".join(logs.output))
        self.assertEqual(self.hub.count_suffix("POST", "/close"), 1)
        self.assertEqual(
            self.hub.order()[-2:],
            [
                ("GET", "/hub/eval/trials/trial-1"),
                ("POST", "/hub/eval/jobs/job-1/cancel"),
            ],
        )

    async def test_a_cancelled_scorer_close_still_lands_once(self) -> None:
        # The eval is cancelled while the scorer's close is in flight: the
        # started close completes (shielded) and is never resent.
        self.hub.close_delay = 0.05
        config, environment = await self._live_env()

        with anyio.move_on_after(0.01):
            await environment.close_trial()
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, {"default": environment}, True
        )

        self.assertEqual(self.hub.status["trial-1"], "closed")
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "closed"}],
        )
        self.assertFalse(_only_runtime().trial_slots.locked())

    async def test_a_claim_cancelled_while_provisioning_still_cancels_its_trial(
        self,
    ) -> None:
        self.hub.becomes = "provisioning"  # never ready
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with anyio.move_on_after(0.05):
            await self._sample(config, "s-1")

        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "cancelled"}],
        )
        self.assertEqual(self.hub.status["trial-1"], "closed")
        runtime = _only_runtime()
        self.assertEqual(runtime.environments, set())
        self.assertFalse(runtime.trial_slots.locked())

    async def test_a_cancelled_claims_cancel_reads_back_after_a_gateway_timeout(
        self,
    ) -> None:
        # The whole unready cancel is shielded: after a 504 on its close, the
        # read-back still runs although the claim is being cancelled.
        self.hub.becomes = "provisioning"  # never ready
        self.hub.close_gateway_timeout = True
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with anyio.move_on_after(0.05):
            await self._sample(config, "s-1")

        order = self.hub.order()
        close = order.index(("POST", "/hub/eval/trials/trial-1/close"))
        self.assertIn(("GET", "/hub/eval/trials/trial-1"), order[close + 1 :])
        self.assertEqual(self.hub.count_suffix("POST", "/close"), 1)

    async def test_ready_timeout_releases_the_slot_and_propagates_typed(self) -> None:
        self.hub.becomes = "provisioning"  # never ready
        config = _config(concurrency=1, max_wait_seconds=0.05)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with self.assertRaises(TrialReadyTimeout) as ctx:
            await self._sample(config, "s-1")

        self.assertEqual(ctx.exception.trial_id, "trial-1")
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "cancelled"}],
        )
        runtime = _only_runtime()
        self.assertEqual(runtime.environments, set())
        self.assertFalse(runtime.trial_slots.locked())

    async def test_a_lost_claim_answer_is_resent_with_the_same_key_once(
        self,
    ) -> None:
        # The hub opened the trial but its answer was lost (a transport error,
        # then a gateway 503): the resend with the same clientAttemptKey gets
        # the same trial back, so each sample holds exactly one trial.
        config = _config()
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        self.hub.claim_lost = [None]
        first = _env(await self._sample(config, "s-1"))
        self.hub.claim_lost = [503]
        second = _env(await self._sample(config, "s-2"))

        claims = self.hub.calls("POST", "/hub/eval/trials")
        self.assertEqual(len(claims), 4)
        self.assertEqual(claims[0], claims[1])
        self.assertEqual(claims[2], claims[3])
        self.assertNotEqual(
            claims[0]["clientAttemptKey"], claims[2]["clientAttemptKey"]
        )
        self.assertEqual((first.trial_id, second.trial_id), ("trial-1", "trial-2"))
        self.assertEqual(
            sorted(self.hub.trial_for_key.values()), ["trial-1", "trial-2"]
        )
        self.assertEqual(self.hub.open_trials, 2)
        self.assertEqual(self.hub.count_suffix("POST", "/close"), 0)

        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "cancelled"}],
        )
        self.assertEqual(self.hub.open_trials, 0)

    async def test_a_claim_unanswered_twice_cancels_its_orphan_and_raises(
        self,
    ) -> None:
        # Both answers are lost after the hub opened the trial: the provider
        # finds it by its attempt key, cancels it, and raises the first error.
        self.hub.claim_lost = [None, 504]
        config = _config(concurrency=1)
        await RogueArenaSandboxEnvironment.task_init("startup", config)

        with (
            self.assertLogs("inspect_rogue_arena.client", level="WARNING") as logs,
            self.assertRaises(httpx.ReadError) as ctx,
        ):
            await self._sample(config, "s-1")

        retry_error = ctx.exception.__context__
        assert isinstance(retry_error, RogueArenaApiError)
        self.assertEqual(retry_error.status_code, 504)
        claims = self.hub.calls("POST", "/hub/eval/trials")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[0], claims[1])
        order = self.hub.order()
        listed = order.index(("GET", "/hub/eval/jobs/job-1/trials"))
        self.assertGreater(listed, 1)
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "cancelled"}],
        )
        self.assertGreater(
            order.index(("POST", "/hub/eval/trials/trial-1/close")), listed
        )
        self.assertEqual(self.hub.status["trial-1"], "closed")
        self.assertEqual(self.hub.open_trials, 0)
        self.assertIn("trial-1", "\n".join(logs.output))
        runtime = _only_runtime()
        self.assertEqual(runtime.environments, set())
        self.assertFalse(runtime.trial_slots.locked())

        # the job's only concurrency slot is free again: the next claim lands
        environments = await asyncio.wait_for(self._sample(config, "s-2"), timeout=2)
        self.assertEqual(_env(environments).trial_id, "trial-2")
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "task", config, environments, False
        )

    async def test_an_unparseable_heartbeat_answer_keeps_the_lease_loop_alive(
        self,
    ) -> None:
        # A 200 heartbeat whose trial has a status this client does not know
        # still moved the lease: the loop logs, keeps beating, and exec runs.
        self.hub.heartbeat_status = "draining"
        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)
            if len(sleeps) > 2:
                await asyncio.Event().wait()  # park the loop until it is cancelled
            await asyncio.sleep(0)

        with (
            patch("inspect_rogue_arena.sandbox._sleep", fake_sleep),
            self.assertLogs("inspect_rogue_arena.sandbox", level="WARNING") as logs,
        ):
            _, environment = await self._live_env()
            loop = environment._heartbeat_task
            assert loop is not None
            await _until(
                lambda: (
                    self.hub.count_suffix("POST", "/heartbeat") == 2
                    and len(sleeps) == 3
                )
            )
            self.assertFalse(loop.done())
            result = await environment.exec(["id"])

        self.assertTrue(result.success)
        self.assertEqual(self.hub.count_suffix("POST", "/heartbeat"), 3)
        self.assertEqual(self.hub.count_suffix("POST", "/exec"), 1)
        self.assertIn("could not be parsed (ValueError)", "\n".join(logs.output))

    async def test_a_scored_timeout_expiry_is_logged_as_a_scored_run(self) -> None:
        # An expired trial the hub scored before finalizing
        # it is a model result, so the close logs it as scored, timed out.
        self.hub.close_as = ("expired", "timeout")
        _, environment = await self._live_env()

        with self.assertLogs("inspect_rogue_arena.sandbox", level="INFO") as logs:
            trial = await environment.close_trial()

        assert trial.score is not None
        output = "\n".join(logs.output)
        self.assertIn(
            "INFO:inspect_rogue_arena.sandbox:Rogue Arena trial trial-1 closed: "
            "outcome=expired terminal_reason=timeout hops=1/2 timed_out=True",
            output,
        )
        self.assertNotIn("without a scored agent run", output)

    async def test_exec_outcome_unknown_propagates_as_a_sample_error(self) -> None:
        self.hub.exec_responses = [(504, {"errorMessage": "Gateway Time-out"})]
        _, environment = await self._live_env()

        with self.assertRaises(ExecOutcomeUnknown) as ctx:
            await environment.exec(["make", "install"])

        self.assertNotIsInstance(ctx.exception, SandboxUnavailableError)
        self.assertEqual(ctx.exception.cmd, ("make", "install"))
        self.assertEqual(self.hub.count_suffix("POST", "/exec"), 1)  # never re-run

    async def test_a_queued_sample_waits_for_a_slot_past_max_wait_seconds(
        self,
    ) -> None:
        # Waiting for a slot is ordinary queueing, never an error: the
        # provisioning bound (max_wait_seconds) does not apply to it.
        self.hub.claim_status = "ready"
        config = _config(concurrency=1, max_wait_seconds=0.05)
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        held = await self._sample(config, "s-1")

        queued = asyncio.create_task(self._sample(config, "s-2"))
        await asyncio.sleep(0.2)
        self.assertFalse(queued.done())
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/trials")), 1)

        await RogueArenaSandboxEnvironment.sample_cleanup("task", config, held, False)
        second = await asyncio.wait_for(queued, timeout=2)
        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/trials")), 2)
        await RogueArenaSandboxEnvironment.sample_cleanup("task", config, second, False)

    def test_inspect_sandbox_limit_is_the_configured_concurrency(self) -> None:
        # Inspect queues samples before sample_init at this limit; the hook
        # sees no task config, so it resolves the environment/default config.
        self.assertEqual(
            RogueArenaSandboxEnvironment.default_concurrency(),
            RogueArenaSandboxConfig().concurrency,
        )
        with patch.dict(os.environ, {"ROGUE_ARENA_CONCURRENCY": "5"}):
            self.assertEqual(RogueArenaSandboxEnvironment.default_concurrency(), 5)

    # --- task cleanup ------------------------------------------------------------

    async def test_task_cleanup_cancels_leftover_trials_then_the_job(self) -> None:
        config = _config()
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        environment = _env(await self._sample(config, "s-1"))

        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)

        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-1/close"),
            [{"outcome": "cancelled"}],
        )
        self.assertEqual(self.hub.order()[-1], ("POST", "/hub/eval/jobs/job-1/cancel"))
        closed = environment.closed_trial
        assert closed is not None
        self.assertEqual(closed.outcome, EvalTrialOutcome.CANCELLED)
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    async def test_task_cleanup_survives_a_failed_leftover_close(self) -> None:
        self.hub.close_malformed = {"trial-1"}
        config = _config()
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        await self._sample(config, "s-1")
        await self._sample(config, "s-2")

        with self.assertLogs("inspect_rogue_arena.sandbox", level="WARNING") as logs:
            await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)

        self.assertIn("trial-1", "\n".join(logs.output))
        self.assertEqual(
            self.hub.calls("POST", "/hub/eval/trials/trial-2/close"),
            [{"outcome": "cancelled"}],
        )
        self.assertEqual(self.hub.order()[-1], ("POST", "/hub/eval/jobs/job-1/cancel"))
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    async def test_task_cleanup_without_cleanup_leaves_the_job_and_stops_heartbeats(
        self,
    ) -> None:
        config = _config()
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        environment = _env(await self._sample(config, "s-1"))
        loop = environment._heartbeat_task
        assert loop is not None

        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, False)

        self.assertTrue(loop.done())
        self.assertEqual(self.hub.count_suffix("POST", "/close"), 0)
        self.assertEqual(self.hub.count_suffix("POST", "/cancel"), 0)
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    async def test_inspect_hook_names_mint_one_job_and_cancel_it_at_shutdown(
        self,
    ) -> None:
        # Inspect 0.3.260 calls task_init("startup") and task_cleanup("shutdown");
        # only sample_init carries the real task name.
        config = _config()
        await RogueArenaSandboxEnvironment.task_init("startup", config)
        environments = await self._sample(config, "s-1", task="my_task")
        await RogueArenaSandboxEnvironment.sample_cleanup(
            "my_task", config, environments, False
        )
        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", config, True)

        self.assertEqual(len(self.hub.calls("POST", "/hub/eval/jobs")), 1)
        self.assertEqual(self.hub.count_suffix("POST", "/cancel"), 1)
        self.assertEqual(self.hub.jobs["job-1"]["status"], "cancelled")
        claim = self.hub.calls("POST", "/hub/eval/trials")[0]
        self.assertEqual((claim["jobId"], claim["inspectTaskId"]), ("job-1", "my_task"))
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    async def test_tasks_sharing_a_config_share_one_job_other_configs_get_their_own(
        self,
    ) -> None:
        shared, other = _config(), _config(max_trials=5)
        await asyncio.gather(
            RogueArenaSandboxEnvironment.task_init("startup", shared),
            RogueArenaSandboxEnvironment.task_init("startup", other),
        )
        first = await self._sample(shared, "s-1", task="task-a")
        second = await self._sample(shared, "s-1", task="task-b")
        third = await self._sample(other, "s-1", task="task-c")
        self.assertEqual(len(RogueArenaSandboxEnvironment._runtimes), 2)
        jobs = self.hub.calls("POST", "/hub/eval/jobs")
        self.assertEqual(sorted(job["n"] for job in jobs), [2, 5])
        shared_job = next(
            job_id for job_id, job in self.hub.jobs.items() if job["n"] == 2
        )
        other_job = next(
            job_id for job_id, job in self.hub.jobs.items() if job["n"] == 5
        )
        claims = self.hub.calls("POST", "/hub/eval/trials")
        self.assertEqual(
            [(claim["jobId"], claim["inspectTaskId"]) for claim in claims],
            [(shared_job, "task-a"), (shared_job, "task-b"), (other_job, "task-c")],
        )

        for environments in (first, second, third):
            await RogueArenaSandboxEnvironment.sample_cleanup(
                "task", shared, environments, False
            )
        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", shared, True)
        self.assertEqual(self.hub.jobs[shared_job]["status"], "cancelled")
        self.assertEqual(self.hub.jobs[other_job]["status"], "open")
        self.assertEqual(len(RogueArenaSandboxEnvironment._runtimes), 1)
        await RogueArenaSandboxEnvironment.task_cleanup("shutdown", other, True)
        self.assertEqual(self.hub.jobs[other_job]["status"], "cancelled")
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    async def test_job_create_retries_a_transport_error_with_the_same_key(
        self,
    ) -> None:
        self.hub.job_post_failures = 1
        config = _config()

        await RogueArenaSandboxEnvironment.task_init("startup", config)

        posts = self.hub.calls("POST", "/hub/eval/jobs")
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]["clientJobKey"], posts[1]["clientJobKey"])
        self.assertEqual(len(self.hub.jobs), 1)
        self.assertEqual(_only_runtime().job_id, "job-1")


def _without_rogue_arena_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ROGUE_ARENA_")
    }


class InspectEvalSmokeTests(unittest.TestCase):
    """A real (offline) `inspect_ai.eval()` over the FakeHub: mockllm, no network."""

    def test_eval_queues_extra_samples_on_one_job_and_cancels_it_at_shutdown(
        self,
    ) -> None:
        hub = FakeHub(self)
        open_at_scoring: list[bool] = []

        @solver
        def run_id() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                result = await sandbox().exec(["id"])
                state.metadata["uid"] = result.stdout
                await anyio.sleep(0.4)  # a trial outlives max_wait_seconds
                return state

            return solve

        @scorer(metrics=[mean()])
        def open_probe() -> Scorer:
            # runs before rogue_arena_hops (Inspect scores in list order)
            async def score(state: TaskState, target: Target) -> Score:
                environment = sandbox().as_type(RogueArenaSandboxEnvironment)
                open_at_scoring.append(environment.closed_trial is None)
                return Score(value=0)

            return score

        # more samples than concurrency: the extras queue, none may error
        config = _config(max_trials=4, concurrency=2, max_wait_seconds=0.3)
        task = Task(
            dataset=[Sample(input=f"q{n}", id=f"s-{n}") for n in range(1, 5)],
            solver=run_id(),
            scorer=[open_probe(), rogue_arena_hops()],
            sandbox=SandboxEnvironmentSpec("rogue-arena", config),
            name="smoke_task",
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            log = inspect_eval(
                task, model="mockllm/model", log_dir=log_dir, display="none"
            )[0]
            self.assertEqual(log.status, "success", log.error)
            samples = log.samples or []
            scores = [sample.scores["rogue_arena_hops"] for sample in samples]
            assert log.results is not None
            hops_metrics = next(
                score.metrics
                for score in log.results.scores
                if score.name == "rogue_arena_hops"
            )

        self.assertEqual([score.value for score in scores], [0.5] * 4)
        self.assertEqual(
            {
                (
                    score.metadata["hops_passed"],
                    score.metadata["hops_total"],
                    score.metadata["reached_step"],
                    score.metadata["terminal_reason"],
                    len(score.metadata["per_hop"]),
                )
                for score in scores
                if score.metadata is not None
            },
            {(1, 2, 1, "agent_fail", 2)},
        )
        self.assertEqual(hops_metrics["mean"].value, 0.5)
        self.assertIn("stderr", hops_metrics)
        # scorers run before sample_cleanup, so the scorer's close is the one
        self.assertEqual(open_at_scoring, [True] * 4)
        self.assertEqual(len(hub.calls("POST", "/hub/eval/jobs")), 1)
        claims = hub.calls("POST", "/hub/eval/trials")
        self.assertEqual(len(claims), 4)
        self.assertEqual({claim["inspectTaskId"] for claim in claims}, {"smoke_task"})
        self.assertLessEqual(hub.peak_open_trials, 2)
        for trial_id in ("trial-1", "trial-2", "trial-3", "trial-4"):
            self.assertEqual(
                hub.calls("POST", f"/hub/eval/trials/{trial_id}/close"),
                [{"outcome": "closed"}],
            )
        self.assertEqual(hub.order()[-1], ("POST", "/hub/eval/jobs/job-1/cancel"))
        self.assertEqual(hub.jobs["job-1"]["status"], "cancelled")
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    def test_an_infrastructure_outcome_errors_the_sample_instead_of_scoring(
        self,
    ) -> None:
        # The close hits a gateway timeout and reads back a reaped trial: the
        # scorer raises, so the sample errors and never enters the mean.
        hub = FakeHub(self)
        hub.close_gateway_timeout = True
        task = Task(
            dataset=[Sample(input="q1", id="s-1")],
            scorer=rogue_arena_hops(),
            sandbox=SandboxEnvironmentSpec("rogue-arena", _config(max_trials=1)),
            name="infra_task",
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            log = inspect_eval(
                task,
                model="mockllm/model",
                log_dir=log_dir,
                display="none",
                fail_on_error=False,
            )[0]
            samples = log.samples or []

        self.assertEqual(len(samples), 1)
        error = samples[0].error
        assert error is not None
        self.assertIn("TrialNotScored", error.traceback)
        self.assertIn("reaped", error.message)
        self.assertIn("not a model result", error.message)
        self.assertFalse(samples[0].scores)
        self.assertEqual(
            hub.calls("POST", "/hub/eval/trials/trial-1/close"), [{"outcome": "closed"}]
        )
        self.assertEqual(hub.jobs["job-1"]["status"], "cancelled")


# The examples ship inside the package; resolve them the way `launch` does.
EXAMPLES = Path(str(resources.files("inspect_rogue_arena").joinpath("examples")))
EXAMPLE_TASK = EXAMPLES / "eval_example.py"
EXAMPLE_RUN_CONFIG = EXAMPLES / "eval-example.eval.yaml"


class ExampleTaskTests(unittest.TestCase):
    """CI guard: the shipped example task and run config work end to end, offline."""

    def test_example_task_runs_one_job_and_scores_every_sample(self) -> None:
        run_config = yaml.safe_load(EXAMPLE_RUN_CONFIG.read_text())
        # the run config's task is the example file, beside the config
        self.assertEqual(EXAMPLES / run_config["task"], EXAMPLE_TASK)
        # parsed the way `inspect eval --run-config` parses its `sandbox:` block
        spec = SandboxEnvironmentSpec.model_validate(run_config["sandbox"])
        self.assertEqual(spec.type, "rogue-arena")
        config = RogueArenaSandboxConfig.model_validate(run_config["sandbox"]["config"])
        assert config.eval is not None
        hub = FakeHub(self)
        hub.entries = [_entry(ENV_ID, config.eval)]  # the eval the config selects
        model = get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call("mockllm/model", "bash", {"command": "id"}),
                ModelOutput.for_tool_call(
                    "mockllm/model", "submit", {"answer": "done"}
                ),
            ],
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        cwd = os.getcwd()
        with (
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            # like `inspect eval --run-config ...` from the repo root: Inspect
            # loads the task file itself from the run config's relative path
            os.chdir(EXAMPLES)
            try:
                log = inspect_eval(
                    run_config["task"],
                    model=model,
                    sandbox=spec,
                    log_dir=log_dir,
                    display="none",
                )[0]
            finally:
                os.chdir(cwd)
            self.assertEqual(log.status, "success", log.error)
            samples = log.samples or []

        self.assertEqual(log.eval.task, "rogue_arena_eval")
        # one job for the run, created from the run config's sizing
        jobs = hub.calls("POST", "/hub/eval/jobs")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            (
                jobs[0]["environmentId"],
                jobs[0]["n"],
                jobs[0]["concurrency"],
                jobs[0]["wallClockLimitSeconds"],
            ),
            (ENV_ID, config.max_trials, config.concurrency, config.max_runtime_seconds),
        )
        # one trial per sample, and every sample scored from the server score
        claims = hub.calls("POST", "/hub/eval/trials")
        self.assertTrue(samples)
        self.assertEqual(len(claims), len(samples))
        self.assertEqual(
            {claim["inspectTaskId"] for claim in claims}, {"rogue_arena_eval"}
        )
        for sample in samples:
            self.assertIsNone(sample.error)
            self.assertEqual(sample.scores["rogue_arena_hops"].value, 0.5)
        # The example's bash timeout is above the sync ceiling, so the react
        # agent's command runs as a background exec (launch + poll) on the
        # retry-safe path, not a single synchronous exec.
        launch, *polls = hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertEqual(launch["cmd"], ["bash", "--login", "-c", "id"])
        self.assertTrue(launch["background"])
        self.assertGreater(launch["timeoutSeconds"], config.sync_exec_ceiling_seconds)
        self.assertTrue(polls)
        self.assertTrue(all(poll["pollExecId"] for poll in polls))
        bash_results = [
            m for m in samples[0].messages if m.role == "tool" and m.function == "bash"
        ]
        self.assertEqual([m.error for m in bash_results], [None])
        self.assertIn("uid=0(root)", bash_results[0].text)
        self.assertEqual(
            hub.calls("POST", "/hub/eval/trials/trial-1/close"), [{"outcome": "closed"}]
        )
        self.assertEqual(hub.order()[-1], ("POST", "/hub/eval/jobs/job-1/cancel"))
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    def test_solo_command_survives_a_transient_readtimeout(self) -> None:
        """A long agent command on the solo example path survives a transient
        gateway/transport failure.

        The example's ``rogue_bash`` runs above the sync ceiling, so a command
        is a background exec whose poll is retried through the failure. (A
        single synchronous exec cannot be retried: its unanswered request
        raises ``ExecOutcomeUnknown`` and errors the sample.)
        """
        run_config = yaml.safe_load(EXAMPLE_RUN_CONFIG.read_text())
        spec = SandboxEnvironmentSpec.model_validate(run_config["sandbox"])
        config = RogueArenaSandboxConfig.model_validate(run_config["sandbox"]["config"])
        assert config.eval is not None
        hub = FakeHub(self)
        hub.entries = [_entry(ENV_ID, config.eval)]
        # One transient ReadTimeout on the request that waits on the command's
        # result. A synchronous exec is one such request and cannot be retried
        # (ExecOutcomeUnknown errors the sample); the example's above-ceiling
        # timeout runs the command in the background, where the poll is retried.
        hub.exec_result_transport_failures = 1
        model = get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    "mockllm/model", "bash", {"command": "sleep 120; id"}
                ),
                ModelOutput.for_tool_call(
                    "mockllm/model", "submit", {"answer": "done"}
                ),
            ],
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        cwd = os.getcwd()
        with (
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            os.chdir(EXAMPLES)
            try:
                log = inspect_eval(
                    run_config["task"],
                    model=model,
                    sandbox=spec,
                    log_dir=log_dir,
                    display="none",
                )[0]
            finally:
                os.chdir(cwd)
            self.assertEqual(log.status, "success", log.error)
            samples = log.samples or []

        self.assertTrue(samples)
        sample = samples[0]
        # The transient failure did not error the sample.
        self.assertIsNone(sample.error)
        bash_results = [
            m for m in sample.messages if m.role == "tool" and m.function == "bash"
        ]
        self.assertEqual([m.error for m in bash_results], [None])
        self.assertIn("uid=0(root)", bash_results[0].text)
        # The command went background (launch), the transient poll was retried,
        # and a later poll returned the result. The hub consumed its one failure.
        launch, *polls = hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertTrue(launch["background"])
        self.assertGreaterEqual(len(polls), 2)
        self.assertEqual(hub.exec_result_transport_failures, 0)


class RogueArenaSelectionTests(unittest.TestCase):
    ENTRIES = (
        EvalCatalogEntry.from_dict(_entry(ENV_ID, EVAL_NAME)),
        EvalCatalogEntry.from_dict(
            _entry("44444444-4444-4444-8444-444444444444", "Twin")
        ),
        EvalCatalogEntry.from_dict(
            _entry("55555555-5555-4555-8555-555555555555", "Twin")
        ),
    )

    def test_selects_by_environment_id_or_unique_name(self) -> None:
        by_id = select_eval(self.ENTRIES, RogueArenaSandboxConfig(eval=ENV_ID))
        by_name = select_eval(self.ENTRIES, RogueArenaSandboxConfig(eval=EVAL_NAME))
        twin = select_eval(
            self.ENTRIES,
            RogueArenaSandboxConfig(eval="55555555-5555-4555-8555-555555555555"),
        )
        self.assertEqual(by_id.environment_id, ENV_ID)
        self.assertEqual(by_name.environment_id, ENV_ID)
        self.assertEqual(twin.name, "Twin")

    def test_missing_ambiguous_and_unset_selectors_are_errors(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not found.*Example AD Chain"):
            select_eval(self.ENTRIES, RogueArenaSandboxConfig(eval="Nope"))
        with self.assertRaisesRegex(RuntimeError, "ambiguous.*44444444"):
            select_eval(self.ENTRIES, RogueArenaSandboxConfig(eval="Twin"))
        with self.assertRaisesRegex(RuntimeError, "ROGUE_ARENA_EVAL"):
            select_eval(self.ENTRIES, RogueArenaSandboxConfig())

    def test_missing_reserved_sample_id_is_rejected(self) -> None:
        # a user metadata key must never replace Inspect's sample identity
        with self.assertRaisesRegex(RuntimeError, "__sample_id__"):
            _require_sample_id({"sample_id": "not-inspect-reserved"})


if __name__ == "__main__":
    unittest.main()
