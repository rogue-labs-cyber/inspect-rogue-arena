from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from importlib import resources
from pathlib import Path
from unittest.mock import patch

import httpx

from inspect_rogue_arena.cli import _PARALLEL_FLAGS, build_parser, main
from inspect_rogue_arena.client import RogueArenaClient
from inspect_rogue_arena.config import RogueArenaSandboxConfig

BASE_URL = "https://hub.example.test"
TOKEN = "ra_eval_S3cretTokenValueNeverPrinted9f2c"
FLAG_BASE_URL = "https://flag.example.test"
FLAG_TOKEN = "ra_eval_FlagTokenValueNeverPrinted77aa"
ENV = {"ROGUE_ARENA_BASE_URL": BASE_URL, "ROGUE_ARENA_TOKEN": TOKEN}

WHOAMI = {
    "userMainId": "user-1",
    "organizationId": "org-1",
    "tokenLabel": "ci laptop",
    "tokenLast4": "9f2c",
}
EVALS = [
    {
        "environmentId": "11111111-1111-4111-8111-111111111111",
        "name": "example-ad-chain",
        "environmentBuildId": "build-1",
        "canvasVersionNumber": 3,
        "hopCount": 4,
        "agentMachineNickname": "agent",
        "scoringMode": "sequential",
        "readiness": {"warmCount": 2, "onDemandEtaSeconds": None},
    },
    {
        "environmentId": "22222222-2222-4222-8222-222222222222",
        "name": "example-web-app",
        "environmentBuildId": "build-2",
        "canvasVersionNumber": 1,
        "hopCount": 2,
        "agentMachineNickname": None,
        "scoringMode": "independent",
        "readiness": {"warmCount": 0, "onDemandEtaSeconds": 420},
    },
]
# An eval listing with no readiness field at all.
EVALS_NO_READINESS = [
    {key: value for key, value in entry.items() if key != "readiness"}
    for entry in EVALS
]
JOBS = [
    {
        "id": "job-1",
        "environmentId": "11111111-1111-4111-8111-111111111111",
        "environmentBuildId": "build-1",
        "canvasVersionId": "cv-1",
        "ownerUserMainId": "user-1",
        "organizationId": None,
        "n": 4,
        "concurrency": 2,
        "wallClockLimitSeconds": 3600,
        "clientJobKey": "inspect_abc",
        "status": "open",
        "trialCount": 2,
        "closedTrialCount": 1,
    }
]
TRIALS = [
    {
        "id": "trial-1",
        "jobId": "job-1",
        "deploymentRecordId": "dep-1",
        "agentDeployedVmId": "vm-1",
        "agentMachineNickname": "agent",
        "status": "closed",
        "outcome": "closed",
        "terminalReason": "agent_fail",
        "warm": False,
        "reachedStep": 1,
        "hopsPassed": 1,
        "totalSteps": 4,
        "clientKind": "inspect",
        "clientAttemptKey": "inspect_k1",
        "score": {
            "reachedStep": 1,
            "hopsPassed": 1,
            "totalSteps": 4,
            "scoringMode": "sequential",
            "steps": [
                {
                    "stepNumber": 1,
                    "name": "foothold",
                    "passed": True,
                    "passedAt": "2026-09-30T12:00:00Z",
                    "detail": "",
                },
                {
                    "stepNumber": 2,
                    "name": "pivot",
                    "passed": False,
                    "passedAt": None,
                    "detail": "",
                },
            ],
        },
        "hopResults": None,
    }
]

Handler = Callable[[httpx.Request], httpx.Response]


def _straddling_body(token: str) -> dict[str, str]:
    """An error body whose text puts ``token`` across the client's 500-char cut."""
    lead = "{'errorMessage': '"
    pad = 500 - len(lead) - len(" Bearer ") - len(token) // 2
    body = {"errorMessage": "x" * pad + " Bearer " + token}
    start = str(body).index(token)
    assert start < 500 < start + len(token)
    return body


def _token_fragments(text: str, token: str, size: int = 6) -> list[str]:
    """Every ``size``-character piece of ``token`` that appears in ``text``."""
    pieces = {token[i : i + size] for i in range(len(token) - size + 1)}
    return sorted(piece for piece in pieces if piece in text)


@dataclass(frozen=True)
class CliRun:
    code: int | str | None
    stdout: str
    stderr: str
    requests: tuple[httpx.Request, ...]

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def _hub(
    whoami: Mapping[str, str | None] = WHOAMI,
    evals: list[dict[str, str | int | None]] | None = None,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if (request.method, request.url.path) == ("GET", "/hub/eval/whoami"):
            return httpx.Response(200, json=dict(whoami))
        if (request.method, request.url.path) == ("GET", "/hub/eval/evals"):
            return httpx.Response(200, json=EVALS if evals is None else evals)
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


def _respond(status_code: int, body: Mapping[str, str | int]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=dict(body))

    return handler


def _run(
    argv: list[str],
    handler: Handler | None = None,
    env: Mapping[str, str] = ENV,
) -> CliRun:
    """Run ``main`` with a clean environment and every request on a MockTransport."""
    requests: list[httpx.Request] = []
    respond = handler or _hub()

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    client_factory = partial(RogueArenaClient, transport=httpx.MockTransport(record))
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        patch.dict(os.environ, dict(env), clear=True),
        patch("inspect_rogue_arena.cli.RogueArenaClient", client_factory),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            code: int | str | None = main(argv)
        except SystemExit as exited:
            code = exited.code
    return CliRun(code, stdout.getvalue(), stderr.getvalue(), tuple(requests))


class CliParserTests(unittest.TestCase):
    def test_subcommands_are_whoami_evals_runs_run_and_launch(self) -> None:
        subcommands = [
            action
            for action in build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        ]
        self.assertEqual(len(subcommands), 1)
        self.assertEqual(
            set(subcommands[0].choices),
            {"whoami", "evals", "runs", "run", "launch"},
        )

    def test_unknown_subcommands_exit_2_without_a_request(self) -> None:
        for command in ("deploy", "status", "bogus"):
            with self.subTest(command=command):
                run = _run([command])
                self.assertEqual(run.code, 2)
                self.assertIn("invalid choice", run.stderr)
                self.assertEqual(run.stdout, "")
                self.assertEqual(run.requests, ())

    def test_a_subcommand_is_required(self) -> None:
        run = _run([])
        self.assertEqual(run.code, 2)
        self.assertEqual(run.requests, ())

    def test_unknown_flags_exit_2(self) -> None:
        for flag in ("--budget", "--mode", "--max-runtime-seconds"):
            with self.subTest(flag=flag):
                run = _run(["evals", flag, "1"])
                self.assertEqual(run.code, 2)
                self.assertIn("unrecognized arguments", run.stderr)
                self.assertEqual(run.requests, ())


class CliWhoamiTests(unittest.TestCase):
    def test_whoami_prints_the_token_owner_as_sorted_json(self) -> None:
        run = _run(["whoami"])

        self.assertEqual(run.code, 0)
        self.assertEqual(run.stderr, "")
        self.assertEqual(json.loads(run.stdout), WHOAMI)
        self.assertEqual(
            run.stdout, json.dumps(WHOAMI, indent=2, sort_keys=True) + "\n"
        )
        self.assertEqual(len(run.requests), 1)
        request = run.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(str(request.url), f"{BASE_URL}/hub/eval/whoami")
        self.assertEqual(request.headers["Authorization"], f"Bearer {TOKEN}")
        self.assertNotIn(TOKEN, run.output)

    def test_whoami_prints_a_null_organization_for_a_personal_token(self) -> None:
        personal = {**WHOAMI, "organizationId": None}
        run = _run(["whoami"], _hub(whoami=personal))

        self.assertEqual(run.code, 0)
        self.assertEqual(json.loads(run.stdout), personal)

    def test_whoami_requires_a_token(self) -> None:
        for token in (None, "", "   "):
            with self.subTest(token=token):
                env = {"ROGUE_ARENA_BASE_URL": BASE_URL}
                if token is not None:
                    env["ROGUE_ARENA_TOKEN"] = token
                run = _run(["whoami"], env=env)

                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertTrue(run.stderr.startswith("rogue-arena-inspect: "))
                self.assertIn("ROGUE_ARENA_TOKEN", run.stderr)
                self.assertIn("--token", run.stderr)
                self.assertEqual(run.requests, ())

    def test_whoami_requires_a_base_url(self) -> None:
        run = _run(["whoami"], env={"ROGUE_ARENA_TOKEN": TOKEN})

        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("ROGUE_ARENA_BASE_URL", run.stderr)
        self.assertIn("--base-url", run.stderr)
        self.assertEqual(run.requests, ())
        self.assertNotIn(TOKEN, run.output)

    def test_flags_override_the_environment(self) -> None:
        run = _run(["--base-url", FLAG_BASE_URL, "--token", FLAG_TOKEN, "whoami"])

        self.assertEqual(run.code, 0)
        self.assertEqual(len(run.requests), 1)
        request = run.requests[0]
        self.assertEqual(str(request.url), f"{FLAG_BASE_URL}/hub/eval/whoami")
        self.assertEqual(request.headers["Authorization"], f"Bearer {FLAG_TOKEN}")
        self.assertNotIn(FLAG_TOKEN, run.output)

    def test_flags_work_with_an_empty_environment(self) -> None:
        run = _run(
            ["--base-url", FLAG_BASE_URL, "--token", FLAG_TOKEN, "whoami"], env={}
        )

        self.assertEqual(run.code, 0)
        self.assertEqual(json.loads(run.stdout), WHOAMI)
        self.assertEqual(
            run.requests[0].headers["Authorization"], f"Bearer {FLAG_TOKEN}"
        )


class CliEvalsTests(unittest.TestCase):
    def test_evals_default_shows_launch_visibility_per_eval(self) -> None:
        run = _run(["evals"])

        self.assertEqual(run.code, 0)
        self.assertEqual(run.stderr, "")
        out = run.stdout
        # name, version, stage count, scoring and the launch-visibility label
        self.assertIn(
            "example-ad-chain  |  v3 · 4 stages · sequential · ready now (2 warm)", out
        )
        self.assertIn("    id: 11111111-1111-4111-8111-111111111111", out)
        self.assertIn(
            "example-web-app  |  v1 · 2 stages · independent · on demand, ~7 min", out
        )
        self.assertIn("    id: 22222222-2222-4222-8222-222222222222", out)
        self.assertEqual(len(run.requests), 1)
        self.assertEqual(str(run.requests[0].url), f"{BASE_URL}/hub/eval/evals")
        self.assertNotIn(TOKEN, run.output)

    def test_evals_json_prints_the_raw_listing(self) -> None:
        run = _run(["evals", "--json"])

        self.assertEqual(run.code, 0)
        self.assertEqual(json.loads(run.stdout), {"items": EVALS})

    def test_evals_shows_unknown_when_the_hub_omits_readiness(self) -> None:
        run = _run(["evals"], _hub(evals=EVALS_NO_READINESS))

        self.assertEqual(run.code, 0)
        self.assertIn(
            "example-ad-chain  |  v3 · 4 stages · sequential · unknown", run.stdout
        )
        self.assertIn(
            "example-web-app  |  v1 · 2 stages · independent · unknown", run.stdout
        )

    def test_evals_prints_an_empty_list(self) -> None:
        run = _run(["evals"], _hub(evals=[]))

        self.assertEqual(run.code, 0)
        self.assertEqual(run.stdout.strip(), "no evals are available for this token")

    def test_evals_requires_a_token(self) -> None:
        run = _run(["evals"], env={"ROGUE_ARENA_BASE_URL": BASE_URL})

        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("ROGUE_ARENA_TOKEN", run.stderr)
        self.assertEqual(run.requests, ())


def _runs_hub(
    jobs: list[dict[str, object]] | None = None,
    job: dict[str, object] | None = None,
    trials: list[dict[str, object]] | None = None,
    jobs_status: int = 200,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        if (method, path) == ("GET", "/hub/eval/jobs"):
            if jobs_status != 200:
                return httpx.Response(jobs_status, json={"errorMessage": "nope"})
            return httpx.Response(200, json=JOBS if jobs is None else jobs)
        if method == "GET" and path == "/hub/eval/jobs/job-1":
            return httpx.Response(200, json=JOBS[0] if job is None else job)
        if method == "GET" and path == "/hub/eval/jobs/job-1/trials":
            return httpx.Response(200, json=TRIALS if trials is None else trials)
        if method == "GET" and path.startswith("/hub/eval/jobs/"):
            return httpx.Response(404, json={"errorMessage": "no such job"})
        if method == "GET" and path.startswith("/hub/eval/trials/"):
            return httpx.Response(200, json=TRIALS[0])
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


class CliRunsTests(unittest.TestCase):
    def test_runs_lists_the_tokens_runs(self) -> None:
        run = _run(["runs"], _runs_hub())

        self.assertEqual(run.code, 0)
        self.assertIn(
            "job-1  open  env 11111111-1111-4111-8111-111111111111", run.stdout
        )
        self.assertIn("trials 2 (closed 1)", run.stdout)
        self.assertNotIn(TOKEN, run.output)

    def test_runs_json(self) -> None:
        run = _run(["runs", "--json"], _runs_hub())
        self.assertEqual(run.code, 0)
        self.assertEqual(json.loads(run.stdout), {"items": JOBS})

    def test_runs_degrades_gracefully_when_the_endpoint_is_missing(self) -> None:
        for status in (404, 405, 501):
            with self.subTest(status=status):
                run = _run(["runs"], _runs_hub(jobs_status=status))
                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertIn("no token-scoped run-list endpoint", run.stderr)
                self.assertIn(str(status), run.stderr)
                self.assertNotIn("Traceback", run.stderr)

    def test_runs_empty(self) -> None:
        run = _run(["runs"], _runs_hub(jobs=[]))
        self.assertEqual(run.code, 0)
        self.assertEqual(run.stdout.strip(), "this token has no runs yet")


class CliRunTests(unittest.TestCase):
    def test_run_shows_job_status_and_per_stage_score(self) -> None:
        run = _run(["run", "job-1"], _runs_hub())

        self.assertEqual(run.code, 0)
        out = run.stdout
        self.assertIn("run job-1  open  env 11111111-1111-4111-8111-111111111111", out)
        self.assertIn("trial trial-1  closed  outcome=closed reason=agent_fail", out)
        self.assertIn("score: 1/4 hops", out)
        self.assertIn("#1 foothold: pass", out)
        self.assertIn("#2 pivot: fail", out)
        self.assertNotIn(TOKEN, run.output)

    def test_run_falls_back_to_a_trial_id(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/hub/eval/jobs/trial-1":
                return httpx.Response(404, json={"errorMessage": "no job"})
            if path == "/hub/eval/trials/trial-1":
                return httpx.Response(200, json=TRIALS[0])
            return httpx.Response(404, json={"errorMessage": "nope"})

        run = _run(["run", "trial-1"], handler)
        self.assertEqual(run.code, 0)
        self.assertIn("trial trial-1  closed", run.stdout)
        self.assertIn("score: 1/4 hops", run.stdout)

    def test_run_unknown_id_is_a_clean_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"errorMessage": "nope"})

        run = _run(["run", "missing"], handler)
        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("no run or trial found with id 'missing'", run.stderr)
        self.assertNotIn("Traceback", run.stderr)

    def test_run_json(self) -> None:
        run = _run(["run", "job-1", "--json"], _runs_hub())
        self.assertEqual(run.code, 0)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["run"], JOBS[0])
        self.assertEqual(payload["trials"], TRIALS)

    def test_run_glosses_the_terminal_reason(self) -> None:
        run = _run(["run", "job-1"], _runs_hub())
        self.assertEqual(run.code, 0)
        # the bare enum stays (scannable), with a human gloss appended
        self.assertIn(
            "reason=agent_fail (the agent stopped without finishing)", run.stdout
        )


CHAT_MESSAGES = [
    {
        "id": "chat-1",
        "jobId": "job-1",
        "seq": 1,
        "at": "2026-10-27T01:00:00Z",
        "author": "recon",
        "content": "mapping the reachable hosts",
    },
    {
        "id": "chat-2",
        "jobId": "job-1",
        "seq": 2,
        "at": "2026-10-27T01:00:05Z",
        "author": "exploit",
        "content": "taking the first foothold",
    },
]
EVENTS = [
    {
        "id": "ev-1",
        "at": "2026-10-27T01:00:01Z",
        "kind": "exec",
        "summary": "ran a command",
        "detail": None,
    },
    {
        "id": "ev-2",
        "at": "2026-10-27T01:00:02Z",
        "kind": "hop",
        "summary": "reached step 1",
        "detail": "foothold",
    },
]


def _chat_hub(
    messages: list[dict[str, object]] | None = None,
    job: dict[str, object] | None = None,
) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        if method == "GET" and path == "/hub/eval/jobs/job-1":
            return httpx.Response(200, json=JOBS[0] if job is None else job)
        if method == "GET" and path == "/hub/eval/jobs/job-1/chat":
            return httpx.Response(
                200, json=CHAT_MESSAGES if messages is None else messages
            )
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


class CliRunChatTests(unittest.TestCase):
    def test_chat_prints_the_transcript(self) -> None:
        run = _run(["run", "job-1", "--chat"], _chat_hub())
        self.assertEqual(run.code, 0)
        self.assertIn("run job-1 team chatroom (2 message(s)):", run.stdout)
        self.assertIn(
            "#1 [2026-10-27T01:00:00Z] recon: mapping the reachable hosts", run.stdout
        )
        self.assertIn(
            "#2 [2026-10-27T01:00:05Z] exploit: taking the first foothold", run.stdout
        )
        self.assertNotIn(TOKEN, run.output)

    def test_chat_empty_room(self) -> None:
        run = _run(["run", "job-1", "--chat"], _chat_hub(messages=[]))
        self.assertEqual(run.code, 0)
        self.assertEqual(run.stdout.strip(), "run job-1 has no team chatroom messages")

    def test_chat_json(self) -> None:
        run = _run(["run", "job-1", "--chat", "--json"], _chat_hub())
        self.assertEqual(run.code, 0)
        payload = json.loads(run.stdout)
        self.assertEqual(payload["jobId"], "job-1")
        self.assertEqual([m["seq"] for m in payload["messages"]], [1, 2])

    def test_chat_resolves_a_trial_id_to_its_job(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/hub/eval/jobs/trial-1":
                return httpx.Response(404, json={"errorMessage": "no job"})
            if path == "/hub/eval/trials/trial-1":
                return httpx.Response(200, json=TRIALS[0])
            if path == "/hub/eval/jobs/job-1":
                return httpx.Response(200, json=JOBS[0])
            if path == "/hub/eval/jobs/job-1/chat":
                return httpx.Response(200, json=CHAT_MESSAGES)
            return httpx.Response(404, json={"errorMessage": "nope"})

        run = _run(["run", "trial-1", "--chat"], handler)
        self.assertEqual(run.code, 0)
        self.assertIn("run job-1 team chatroom", run.stdout)

    def test_watch_and_chat_are_mutually_exclusive(self) -> None:
        run = _run(["run", "job-1", "--watch", "--chat"])
        self.assertEqual(run.code, 2)
        self.assertIn("not allowed with argument", run.stderr)
        self.assertEqual(run.requests, ())


class CliRunWatchTests(unittest.TestCase):
    def test_watch_tails_events_then_renders_the_settled_run(self) -> None:
        # job-1 is open on the first poll (with two events), complete on the
        # second; _watch_sleep is a no-op so the loop turns immediately.
        open_job = {**JOBS[0], "status": "open"}
        done_job = {**JOBS[0], "status": "complete"}
        state = {"job_calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            method, path = request.method, request.url.path
            if method == "GET" and path == "/hub/eval/jobs/job-1":
                state["job_calls"] += 1
                return httpx.Response(
                    200, json=open_job if state["job_calls"] == 1 else done_job
                )
            if method == "GET" and path == "/hub/eval/jobs/job-1/trials":
                return httpx.Response(200, json=TRIALS)
            if method == "GET" and path == "/hub/eval/trials/trial-1/events":
                return httpx.Response(200, json=EVENTS)
            return httpx.Response(404, json={"errorMessage": "nope"})

        async def _no_sleep(_seconds: float) -> None:
            return None

        with patch("inspect_rogue_arena.cli._watch_sleep", _no_sleep):
            run = _run(["run", "job-1", "--watch"], handler)

        self.assertEqual(run.code, 0)
        self.assertIn("exec: ran a command", run.stdout)
        self.assertIn("hop: reached step 1: foothold", run.stdout)
        # each event prints once across the two polls
        self.assertEqual(run.stdout.count("ran a command"), 1)
        # the final render is the settled run
        self.assertIn("run job-1  complete", run.stdout)
        self.assertNotIn(TOKEN, run.output)

    def test_watch_tolerates_a_hub_without_a_per_trial_events_endpoint(self) -> None:
        done_job = {**JOBS[0], "status": "complete"}

        def handler(request: httpx.Request) -> httpx.Response:
            method, path = request.method, request.url.path
            if method == "GET" and path == "/hub/eval/jobs/job-1":
                return httpx.Response(200, json=done_job)
            if method == "GET" and path == "/hub/eval/jobs/job-1/trials":
                return httpx.Response(200, json=TRIALS)
            if path == "/hub/eval/trials/trial-1/events":
                return httpx.Response(404, json={"errorMessage": "no events"})
            return httpx.Response(404, json={"errorMessage": "nope"})

        run = _run(["run", "job-1", "--watch"], handler)
        self.assertEqual(run.code, 0)
        self.assertIn("run job-1  complete", run.stdout)

    def test_watch_floors_the_interval(self) -> None:
        done_job = {**JOBS[0], "status": "complete"}
        slept: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            method, path = request.method, request.url.path
            if method == "GET" and path == "/hub/eval/jobs/job-1":
                return httpx.Response(200, json=done_job)
            if method == "GET" and path == "/hub/eval/jobs/job-1/trials":
                return httpx.Response(200, json=TRIALS)
            if path == "/hub/eval/trials/trial-1/events":
                return httpx.Response(200, json=EVENTS)
            return httpx.Response(404, json={"errorMessage": "nope"})

        async def _record(seconds: float) -> None:
            slept.append(seconds)

        with patch("inspect_rogue_arena.cli._watch_sleep", _record):
            run = _run(["run", "job-1", "--watch", "--interval", "0.001"], handler)
        self.assertEqual(run.code, 0)
        # the job was complete on the first poll, so it never slept
        self.assertEqual(slept, [])


EXAMPLES_DIR = Path(
    str(resources.files("inspect_rogue_arena").joinpath("examples"))
).resolve()


def _launch(
    argv: list[str],
    env: Mapping[str, str] = ENV,
    *,
    returncode: int = 0,
    child_stderr: Sequence[str] = (),
) -> tuple[int | str | None, list[str] | None, dict[str, str] | None, str]:
    """Run ``launch`` with ``subprocess.Popen`` stubbed; capture its cmd and env.

    ``returncode``/``child_stderr`` simulate the ``inspect eval`` child, so a
    failing run with provider-policy stderr can be exercised.
    """
    captured: dict[str, object] = {}

    class _FakePopen:
        def __init__(
            self,
            cmd: list[str],
            env: dict[str, str],
            stderr: object = None,
            text: bool = False,
            bufsize: int = -1,
        ) -> None:
            captured["cmd"] = cmd
            captured["env"] = env
            self.stderr = iter(list(child_stderr))

        def wait(self) -> int:
            return returncode

    stderr = io.StringIO()
    with (
        patch.dict(os.environ, dict(env), clear=True),
        patch("inspect_rogue_arena.cli.subprocess.Popen", _FakePopen),
        redirect_stdout(io.StringIO()),
        redirect_stderr(stderr),
    ):
        try:
            code: int | str | None = main(argv)
        except SystemExit as exited:
            code = exited.code
    return (
        code,
        captured.get("cmd"),  # type: ignore[return-value]
        captured.get("env"),  # type: ignore[return-value]
        stderr.getvalue(),
    )


class CliLaunchTests(unittest.TestCase):
    def test_single_agent_launch_passes_a_relative_task_and_keeps_the_token_in_env(
        self,
    ) -> None:
        code, cmd, env, _ = _launch(
            ["launch", "--model", "mockllm/model", "--eval", "example-ad-chain"]
        )

        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertEqual(cmd[:4], [sys.executable, "-m", "inspect_ai", "eval"])
        task_path = cmd[4]
        self.assertFalse(os.path.isabs(task_path))
        self.assertEqual(
            (Path.cwd() / task_path).resolve(), EXAMPLES_DIR / "eval_example.py"
        )
        self.assertEqual(cmd[cmd.index("--model") + 1], "mockllm/model")
        # the token travels in the child env, never on its argv
        self.assertEqual(env["ROGUE_ARENA_TOKEN"], TOKEN)
        self.assertEqual(env["ROGUE_ARENA_EVAL"], "example-ad-chain")
        self.assertNotIn(TOKEN, " ".join(cmd))

    def test_team_launch_sets_team_env_and_derives_the_model_from_the_config(
        self,
    ) -> None:
        team_yaml = EXAMPLES_DIR / "team.yaml"
        code, cmd, env, _ = _launch(["launch", "--team", str(team_yaml), "--eval", "x"])

        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertEqual(
            (Path.cwd() / cmd[4]).resolve(), EXAMPLES_DIR / "team_example.py"
        )
        self.assertEqual(env["ROGUE_ARENA_TEAM"], str(team_yaml.resolve()))
        # no --model given: taken from the team config's first member
        self.assertEqual(cmd[cmd.index("--model") + 1], "mockllm/model")

    def test_launch_task_outside_the_working_directory_is_globbable(self) -> None:
        # inspect eval globs the task spec from the working directory; an
        # absolute spec raises "Non-relative patterns are unsupported".
        with tempfile.TemporaryDirectory() as root:
            task_file = Path(root) / "tasks" / "my_task.py"
            task_file.parent.mkdir()
            task_file.write_text("")
            workdir = Path(root) / "work" / "here"
            workdir.mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(workdir)
            try:
                code, cmd, _, _ = _launch(
                    ["launch", "--model", "m", "--eval", "x", "--task", str(task_file)]
                )
                assert cmd is not None
                self.assertEqual(code, 0)
                matches = list(Path.cwd().glob(cmd[4]))
            finally:
                os.chdir(previous)
        self.assertFalse(os.path.isabs(cmd[4]))
        self.assertEqual([path.resolve() for path in matches], [task_file.resolve()])

    def test_launch_max_runtime_sets_the_env_var(self) -> None:
        code, cmd, env, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--max-runtime", "1800"]
        )
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_MAX_RUNTIME_SECONDS"], "1800")
        # it stays in the child env, never on the argv
        self.assertNotIn("--max-runtime", cmd)
        self.assertNotIn("1800", cmd)

    def test_launch_without_max_runtime_leaves_the_env_var_unset(self) -> None:
        code, _, env, _ = _launch(["launch", "--model", "m", "--eval", "x"])
        assert env is not None
        self.assertEqual(code, 0)
        self.assertNotIn("ROGUE_ARENA_MAX_RUNTIME_SECONDS", env)

    def test_launch_report_both_sets_the_env_var_the_tasks_read(self) -> None:
        code, cmd, env, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--report", "both"]
        )
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_REPORT"], "both")
        self.assertNotIn("--report", cmd)

    def test_launch_without_report_leaves_the_env_var_alone(self) -> None:
        code, _, env, _ = _launch(["launch", "--model", "m", "--eval", "x"])
        assert env is not None
        self.assertEqual(code, 0)
        self.assertNotIn("ROGUE_ARENA_REPORT", env)

    def test_launch_rejects_an_unknown_report(self) -> None:
        code, cmd, _, stderr = _launch(
            ["launch", "--model", "m", "--eval", "x", "--report", "sequential"]
        )
        self.assertEqual(code, 2)
        self.assertIsNone(cmd)
        self.assertIn("invalid choice", stderr)

    def test_launch_parallel_sets_epochs_sandboxes_samples_and_job_concurrency(
        self,
    ) -> None:
        code, cmd, env, stderr = _launch(
            ["launch", "--model", "m", "--eval", "x", "--parallel", "4"],
            env={**ENV, "ROGUE_ARENA_CONCURRENCY": "2"},
        )
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        for flag in ("--epochs", "--max-sandboxes", "--max-samples"):
            with self.subTest(flag):
                self.assertEqual(cmd[cmd.index(flag) + 1], "4")
        # the flag wins over an inherited concurrency, so all three agree
        self.assertEqual(env["ROGUE_ARENA_CONCURRENCY"], "4")
        self.assertNotIn("--parallel", cmd)
        self.assertIn("admits 4 trials at once", stderr)
        # the child's sandbox config resolves the same limit, and its default
        # max_trials admits them
        config = RogueArenaSandboxConfig.from_sources(None, environ=env)
        self.assertEqual(config.concurrency, 4)
        self.assertGreaterEqual(config.max_trials, 4)

    def test_launch_parallel_keeps_extra_args_after_its_own(self) -> None:
        code, cmd, _, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--parallel", "2", "--"]
            + ["--limit", "1"]
        )
        assert cmd is not None
        self.assertEqual(code, 0)
        self.assertEqual(cmd[-2:], ["--limit", "1"])

    def test_launch_parallel_refuses_flags_it_sets_itself(self) -> None:
        for extra in (["--epochs", "2"], ["--max-sandboxes=8"], ["--max-samples", "3"]):
            with self.subTest(extra=extra):
                code, cmd, _, stderr = _launch(
                    ["launch", "--model", "m", "--eval", "x", "--parallel", "4", "--"]
                    + extra
                )
                self.assertEqual(code, 1)
                self.assertIsNone(cmd)
                self.assertIn("--parallel sets", stderr)
                self.assertIn(extra[0].split("=")[0], stderr)

    def test_launch_parallel_refuses_a_max_trials_cap_below_n(self) -> None:
        code, cmd, _, stderr = _launch(
            ["launch", "--model", "m", "--eval", "x", "--parallel", "3"],
            env={**ENV, "ROGUE_ARENA_MAX_TRIALS": "2"},
        )
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("ROGUE_ARENA_MAX_TRIALS=2", stderr)
        # a cap that covers N is the user's choice and stays as set
        code, _, env, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--parallel", "3"],
            env={**ENV, "ROGUE_ARENA_MAX_TRIALS": "6"},
        )
        assert env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_MAX_TRIALS"], "6")

    def test_launch_parallel_needs_a_whole_number_of_at_least_one(self) -> None:
        for value in ("0", "-2", "two"):
            with self.subTest(value=value):
                code, cmd, _, stderr = _launch(
                    ["launch", "--model", "m", "--eval", "x", "--parallel", value]
                )
                self.assertEqual(code, 2)
                self.assertIsNone(cmd)
                self.assertIn("--parallel", stderr)

    def test_launch_parallel_refuses_a_team_run(self) -> None:
        team_yaml = EXAMPLES_DIR / "team.yaml"
        code, cmd, _, stderr = _launch(
            ["launch", "--team", str(team_yaml), "--eval", "x", "--parallel", "2"]
        )
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("does not run team tasks", stderr)

    def test_launch_without_parallel_sets_none_of_its_flags(self) -> None:
        code, cmd, env, _ = _launch(["launch", "--model", "m", "--eval", "x"])
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        for flag in ("--epochs", "--max-sandboxes", "--max-samples"):
            self.assertNotIn(flag, cmd)
        self.assertNotIn("ROGUE_ARENA_CONCURRENCY", env)

    def test_launch_passes_extra_args_through_to_inspect(self) -> None:
        code, cmd, _, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--", "--limit", "1"]
        )
        assert cmd is not None
        self.assertEqual(code, 0)
        self.assertEqual(cmd[-2:], ["--limit", "1"])
        self.assertNotIn("--", cmd)

    def test_launch_requires_a_token(self) -> None:
        code, cmd, _, stderr = _launch(
            ["launch", "--model", "m"], env={"ROGUE_ARENA_BASE_URL": BASE_URL}
        )
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("ROGUE_ARENA_TOKEN", stderr)

    def test_single_agent_launch_without_a_model_is_a_clean_error(self) -> None:
        code, cmd, _, stderr = _launch(["launch", "--eval", "x"])
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("set --model", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_a_provider_policy_failure_gets_a_clean_provider_hint(
        self,
    ) -> None:
        child = [
            (
                "openai.PermissionDeniedError: Error code: 403 - this model is not "
                "permitted under your organization's content policy\n"
            ),
        ]
        code, _, _, stderr = _launch(
            ["launch", "--model", "openai/some-blocked", "--eval", "x"],
            returncode=1,
            child_stderr=child,
        )
        self.assertEqual(code, 1)
        # the child's own stderr is still streamed through
        self.assertIn("PermissionDeniedError", stderr)
        # plus a clean, actionable hint pointing at Inspect's provider docs
        self.assertIn("rejected this run under its policy or auth", stderr)
        self.assertIn("https://inspect.aisi.org.uk/providers.html", stderr)
        self.assertNotIn(TOKEN, stderr)

    def test_an_ordinary_run_failure_gets_no_provider_hint(self) -> None:
        child = ["Task failed: the agent made no progress\n"]
        code, _, _, stderr = _launch(
            ["launch", "--model", "mockllm/model", "--eval", "x"],
            returncode=1,
            child_stderr=child,
        )
        self.assertEqual(code, 1)
        self.assertNotIn("rejected this run under its policy", stderr)

    def test_a_successful_launch_prints_no_hint(self) -> None:
        code, _, _, stderr = _launch(
            ["launch", "--model", "mockllm/model", "--eval", "x"],
            returncode=0,
            child_stderr=["403 forbidden mentioned but the run still passed\n"],
        )
        self.assertEqual(code, 0)
        self.assertNotIn("rejected this run under its policy", stderr)


class CliFailureTests(unittest.TestCase):
    def test_a_rejected_token_exits_1_with_a_clear_message(self) -> None:
        rejected = _respond(
            401, {"errorMessage": "Eval service token expired", "errorSeverity": 2}
        )
        for command in ("whoami", "evals"):
            with self.subTest(command=command):
                run = _run([command], rejected)

                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertTrue(run.stderr.startswith("rogue-arena-inspect: "))
                self.assertIn("rejected the token", run.stderr)
                self.assertIn("401", run.stderr)
                self.assertIn("Eval service token expired", run.stderr)
                self.assertIn("ROGUE_ARENA_TOKEN", run.stderr)
                self.assertNotIn(TOKEN, run.output)

    def test_a_transport_failure_exits_1_naming_the_hub(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        for command in ("whoami", "evals"):
            with self.subTest(command=command):
                run = _run([command], refuse)

                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertIn(f"could not reach Rogue Arena at {BASE_URL}", run.stderr)
                self.assertIn("ConnectError", run.stderr)
                self.assertIn("connection refused", run.stderr)
                self.assertNotIn(TOKEN, run.output)

    def test_an_api_error_body_never_echoes_the_token(self) -> None:
        echoed = _respond(500, {"errorMessage": f"upstream saw Bearer {TOKEN}"})
        run = _run(["whoami"], echoed)

        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("500", run.stderr)
        self.assertIn("upstream saw Bearer [redacted]", run.stderr)
        self.assertNotIn(TOKEN, run.output)

    def test_an_api_error_body_never_echoes_a_flag_token(self) -> None:
        echoed = _respond(500, {"errorMessage": f"upstream saw Bearer {FLAG_TOKEN}"})
        run = _run(["--token", FLAG_TOKEN, "whoami"], echoed)

        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("upstream saw Bearer [redacted]", run.stderr)
        self.assertNotIn(FLAG_TOKEN, run.output)
        self.assertNotIn(TOKEN, run.output)

    def test_a_redirect_exits_1_and_points_at_the_base_url(self) -> None:
        def redirect(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"Location": "https://sso.example/auth?client_id=x"},
                text="<html><head><title>Found</title></head><body>x</body></html>",
            )

        for command in ("whoami", "evals"):
            with self.subTest(command=command):
                run = _run([command], redirect)

                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertEqual(
                    run.stderr,
                    "rogue-arena-inspect: the server redirected the request "
                    "(HTTP 302) instead of answering it: check that "
                    "ROGUE_ARENA_BASE_URL (or --base-url) points at Rogue Arena\n",
                )

    def test_a_token_echoed_across_the_error_cut_leaves_no_fragment(self) -> None:
        # The client redacts before its 500-character cut; the CLI's own
        # full-token redaction alone would miss a prefix left by the cut.
        for argv, token in (
            (["whoami"], TOKEN),
            (["--token", FLAG_TOKEN, "whoami"], FLAG_TOKEN),
        ):
            with self.subTest(argv=argv):
                run = _run(argv, _respond(500, _straddling_body(token)))

                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertIn("Bearer [redacted]", run.stderr)
                self.assertEqual(_token_fragments(run.output, token), [])

    def test_a_malformed_response_exits_1(self) -> None:
        def malformed(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["not", "an", "object"])

        run = _run(["whoami"], malformed)

        self.assertEqual(run.code, 1)
        self.assertEqual(run.stdout, "")
        self.assertIn("invalid whoami", run.stderr)
        self.assertNotIn(TOKEN, run.output)


class LaunchParallelInspectRunTests(unittest.TestCase):
    """``launch --parallel N``'s settings, applied to a real offline Inspect run."""

    def test_n_trials_run_at_once_each_on_its_own_copy(self) -> None:
        import anyio
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import Sample
        from inspect_ai.solver import Generate, Solver, TaskState, solver
        from inspect_ai.util import sandbox
        from test_sandbox import EVAL_NAME, FakeHub

        from inspect_rogue_arena.sandbox import RogueArenaSandboxEnvironment
        from inspect_rogue_arena.scorer import rogue_arena_hops

        parallel = 3
        code, cmd, env, _ = _launch(
            ["launch", "--model", "mockllm/model", "--eval", EVAL_NAME]
            + ["--parallel", str(parallel)]
        )
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        flags = {flag: int(cmd[cmd.index(flag) + 1]) for flag in _PARALLEL_FLAGS}

        # Each trial holds its copy until all N are ready at once; a limit
        # below N anywhere (Inspect's or the job's) would time this out.
        arrived = 0
        all_ready = anyio.Event()

        @solver
        def hold_the_copy() -> Solver:
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                nonlocal arrived
                await sandbox().exec(["id"])
                arrived += 1
                if arrived == parallel:
                    all_ready.set()
                with anyio.fail_after(10):
                    await all_ready.wait()
                return state

            return solve

        hub = FakeHub(self)
        task = Task(
            dataset=[Sample(input="q", id="s-1")],
            solver=hold_the_copy(),
            scorer=rogue_arena_hops(),
            sandbox="rogue-arena",  # configured from the launch's child env
            name="parallel_task",
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, env, clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            log = inspect_eval(
                task,
                model="mockllm/model",
                epochs=flags["--epochs"],
                max_sandboxes=flags["--max-sandboxes"],
                max_samples=flags["--max-samples"],
                log_dir=log_dir,
                display="none",
            )[0]
            self.assertEqual(log.status, "success", log.error)
            samples = log.samples or []

        self.assertEqual(len(samples), parallel)
        self.assertTrue(all(sample.scores for sample in samples))
        (job,) = hub.calls("POST", "/hub/eval/jobs")
        self.assertEqual(job["concurrency"], parallel)
        self.assertGreaterEqual(job["n"], parallel)
        self.assertEqual(len(hub.calls("POST", "/hub/eval/trials")), parallel)
        # all N were open at once, not queued behind a smaller limit
        self.assertEqual(hub.peak_open_trials, parallel)


if __name__ == "__main__":
    unittest.main()
