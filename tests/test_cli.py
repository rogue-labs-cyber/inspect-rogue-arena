from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import httpx

from inspect_rogue_arena.cli import (
    _PARALLEL_FLAGS,
    _log_facts,
    _LogFacts,
    build_parser,
    main,
)
from inspect_rogue_arena.client import RogueArenaClient, transcript_upload_key
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
            {
                "whoami",
                "evals",
                "runs",
                "run",
                "eval",
                "launch",
                "share-transcript",
                "close",
            },
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


class CliBaseUrlTests(unittest.TestCase):
    def test_a_plain_http_base_url_is_refused_before_any_request(self) -> None:
        run = _run(["whoami"], env={**ENV, "ROGUE_ARENA_BASE_URL": "http://hub.test"})

        self.assertEqual(run.code, 1)
        self.assertIn("https://", run.stderr)
        self.assertEqual(run.requests, ())
        self.assertNotIn(TOKEN, run.stderr + run.stdout)

    def test_launch_refuses_a_plain_http_base_url_before_starting_inspect(
        self,
    ) -> None:
        code, cmd, _, stderr = _launch(
            ["launch", "--model", "m", "--eval", "x"],
            env={**ENV, "ROGUE_ARENA_BASE_URL": "http://hub.test"},
        )

        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("https://", stderr)
        self.assertNotIn(TOKEN, stderr)


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


GUIDE_ENV_ID = "11111111-1111-4111-8111-111111111111"
GUIDE = {
    "environmentId": GUIDE_ENV_ID,
    "name": "Example AD Chain",
    "environmentBuildId": "22222222-2222-4222-8222-222222222222",
    "canvasVersionNumber": 3,
    "agentMachineNickname": "agent-vm",
    "scoringMode": "sequential",
    "hops": [
        {
            "stepNumber": 1,
            "name": "foothold",
            "description": "pop the web box",
            "machineNickname": "web",
            "proofType": "keyFile",
            "keyFilepath": "/root/flag1.txt",
            "command": None,
            "regex": None,
            "checkScript": None,
        },
        {
            "stepNumber": 2,
            "name": "domain admin",
            "description": "own the DC",
            "machineNickname": "dc",
            "proofType": "commandCheck",
            "keyFilepath": None,
            "command": "whoami",
            "regex": "admin",
            "checkScript": None,
        },
    ],
    "solutionWriteup": "web -> app VLAN -> dc",
    "recommendedPrompts": {"easy": "do the easy thing", "medium": "", "hard": "win"},
}


def _guide_hub() -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/hub/eval/evals/{GUIDE_ENV_ID}/guide":
            return httpx.Response(200, json=GUIDE)
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


class CliEvalGuideTests(unittest.TestCase):
    def test_guide_renders_graders_solution_and_prompts(self) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--guide"], _guide_hub())

        self.assertEqual(run.code, 0)
        out = run.stdout
        self.assertIn("Example AD Chain", out)
        self.assertIn("#1 foothold", out)
        self.assertIn("key file at /root/flag1.txt", out)
        self.assertIn("`whoami`", out)
        self.assertIn("/admin/", out)
        self.assertIn("web -> app VLAN -> dc", out)
        # every level appears; the blank one says so
        self.assertIn("easy:", out)
        self.assertIn("do the easy thing", out)
        self.assertIn("(none written)", out)
        self.assertEqual(
            run.requests[0].url.path, f"/hub/eval/evals/{GUIDE_ENV_ID}/guide"
        )

    def test_guide_is_the_default_view_without_the_flag(self) -> None:
        # the read is the same whether or not --guide is passed
        run = _run(["eval", GUIDE_ENV_ID], _guide_hub())
        self.assertEqual(run.code, 0)
        self.assertIn("Graders", run.stdout)

    def test_prompt_level_prints_only_that_prompt(self) -> None:
        for level, expected in (("easy", "do the easy thing"), ("hard", "win")):
            with self.subTest(level=level):
                run = _run(
                    ["eval", GUIDE_ENV_ID, "--prompt-level", level], _guide_hub()
                )
                self.assertEqual(run.code, 0)
                self.assertEqual(run.stdout.strip(), expected)
                # only the prompt, not the graders or solution
                self.assertNotIn("Graders", run.stdout)
                self.assertNotIn("web -> app", run.stdout)

    def test_a_blank_recommended_prompt_is_an_error_not_empty_output(self) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--prompt-level", "medium"], _guide_hub())
        self.assertEqual(run.code, 1)
        self.assertIn("no medium recommended prompt", run.stderr)
        self.assertEqual(run.stdout, "")

    def test_json_prints_the_raw_guide(self) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--json"], _guide_hub())
        self.assertEqual(run.code, 0)
        self.assertEqual(json.loads(run.stdout)["environmentId"], GUIDE_ENV_ID)

    def test_an_eval_the_token_cannot_run_is_a_clear_error(self) -> None:
        run = _run(["eval", "99999999-9999-4999-8999-999999999999", "--guide"])
        self.assertEqual(run.code, 1)
        self.assertNotIn(TOKEN, run.stderr)

    def test_an_eval_name_resolves_through_the_listing(self) -> None:
        run = _run(["eval", "ad-chain", "--guide"], _guide_catalog_hub())
        self.assertEqual(run.code, 0)
        self.assertIn("Example AD Chain", run.stdout)

    def test_an_unknown_eval_name_is_a_clear_error(self) -> None:
        run = _run(["eval", "no-such-eval", "--guide"], _guide_catalog_hub())
        self.assertEqual(run.code, 1)
        self.assertIn("no-such-eval", run.stderr)

    def test_an_unknown_prompt_level_exits_2(self) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--prompt-level", "trivial"], _guide_hub())
        self.assertEqual(run.code, 2)
        self.assertIn("invalid choice", run.stderr)

    def test_a_served_guide_takes_one_request(self) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--guide"], _guide_hub())
        self.assertEqual(run.code, 0)
        self.assertEqual(
            [request.url.path for request in run.requests],
            [f"/hub/eval/evals/{GUIDE_ENV_ID}/guide"],
        )


def _hub_refusing_the_guide(
    guide_status: int, *, whoami_status: int = 200, evals: bool = False
) -> Handler:
    """A server whose guide route answers ``guide_status`` (a gateway that does
    not let tokens through to it); whoami answers ``whoami_status``."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/guide"):
            if 300 <= guide_status < 400:
                return httpx.Response(
                    guide_status, headers={"Location": "https://sso.example/login"}
                )
            return httpx.Response(guide_status, json={"message": "Unauthorized"})
        if path == "/hub/eval/whoami":
            if whoami_status == 200:
                return httpx.Response(200, json=WHOAMI)
            return httpx.Response(
                whoami_status, json={"errorMessage": "Eval service token expired"}
            )
        if evals and path == "/hub/eval/evals":
            return httpx.Response(
                200,
                json=[{**EVALS[0], "environmentId": GUIDE_ENV_ID, "name": "ad-chain"}],
            )
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


class CliEvalGuideNotOfferedTests(unittest.TestCase):
    """A server that does not serve the guide route to tokens is named as such,
    not blamed on a token that works."""

    def test_a_401_on_the_guide_with_a_working_token_says_guides_are_not_offered(
        self,
    ) -> None:
        for argv in (
            ["eval", GUIDE_ENV_ID, "--guide"],
            ["eval", GUIDE_ENV_ID, "--prompt-level", "hard"],
            ["eval", "ad-chain", "--json"],
        ):
            with self.subTest(argv=argv):
                run = _run(argv, _hub_refusing_the_guide(401, evals=True))
                self.assertEqual(run.code, 1)
                self.assertEqual(run.stdout, "")
                self.assertIn("does not offer eval guides", run.stderr)
                self.assertIn("HTTP 401", run.stderr)
                self.assertIn("your token works", run.stderr)
                self.assertNotIn("rejected the token", run.stderr)
                self.assertNotIn(TOKEN, run.output)
                self.assertEqual(run.requests[-1].url.path, "/hub/eval/whoami")

    def test_a_redirect_on_the_guide_with_a_working_token_says_the_same(
        self,
    ) -> None:
        run = _run(["eval", GUIDE_ENV_ID, "--guide"], _hub_refusing_the_guide(302))
        self.assertEqual(run.code, 1)
        self.assertIn("does not offer eval guides", run.stderr)
        self.assertIn("HTTP 302", run.stderr)
        self.assertNotIn("redirected the request", run.stderr)

    def test_a_token_that_whoami_also_rejects_is_still_blamed_on_the_token(
        self,
    ) -> None:
        run = _run(
            ["eval", GUIDE_ENV_ID, "--guide"],
            _hub_refusing_the_guide(401, whoami_status=401),
        )
        self.assertEqual(run.code, 1)
        self.assertIn("rejected the token (HTTP 401", run.stderr)
        self.assertNotIn("does not offer eval guides", run.stderr)

    def test_launch_prompt_level_stops_before_the_child_with_the_same_message(
        self,
    ) -> None:
        code, cmd, _, stderr = _launch(
            [
                "launch",
                "--model",
                "m",
                "--eval",
                GUIDE_ENV_ID,
                "--prompt-level",
                "hard",
            ],
            client_handler=_hub_refusing_the_guide(401),
        )
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("does not offer eval guides", stderr)
        self.assertNotIn("rejected the token", stderr)
        self.assertNotIn(TOKEN, stderr)


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


KEPT_JOB = {
    **JOBS[0],
    "status": "cancelled",
    "keepRange": True,
    "rangeClosedAt": None,
}
CLOSED_AT = "2026-11-17T21:00:00.000Z"


def _close_hub(job: Mapping[str, object] = KEPT_JOB) -> Handler:
    """A hub holding one run; ``close-range`` closes its kept range once."""
    state: dict[str, object] = dict(job)

    def handler(request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        if (method, path) == ("GET", "/hub/eval/jobs/job-1"):
            return httpx.Response(200, json=state)
        if (method, path) == ("POST", "/hub/eval/jobs/job-1/close-range"):
            if state.get("keepRange") and state.get("rangeClosedAt") is None:
                state["rangeClosedAt"] = CLOSED_AT
            return httpx.Response(200, json=state)
        if method == "GET" and path == "/hub/eval/jobs/trial-1":
            return httpx.Response(404, json={"errorMessage": "no job"})
        if method == "GET" and path == "/hub/eval/trials/trial-1":
            return httpx.Response(200, json=TRIALS[0])
        return httpx.Response(404, json={"errorMessage": "not found"})

    return handler


def _posts(run: CliRun) -> list[str]:
    return [request.url.path for request in run.requests if request.method == "POST"]


class CliCloseTests(unittest.TestCase):
    def test_close_closes_a_kept_range_once(self) -> None:
        run = _run(["close", "job-1"], _close_hub())
        self.assertEqual(run.code, 0, run.output)
        self.assertEqual(_posts(run), ["/hub/eval/jobs/job-1/close-range"])
        self.assertIn(f"run job-1: range closed at {CLOSED_AT}", run.stdout)
        self.assertIn("removing every range the run kept up", run.stdout)
        self.assertNotIn(TOKEN, run.output)

    def test_closing_twice_is_fine(self) -> None:
        run = _run(
            ["close", "job-1"], _close_hub({**KEPT_JOB, "rangeClosedAt": CLOSED_AT})
        )
        self.assertEqual(run.code, 0, run.output)
        self.assertIn(f"its range was already closed at {CLOSED_AT}", run.stdout)

    def test_a_run_without_keep_range_has_nothing_to_close(self) -> None:
        run = _run(["close", "job-1"], _close_hub(JOBS[0]))
        self.assertEqual(run.code, 0, run.output)
        self.assertEqual(_posts(run), [])
        self.assertIn("was not launched with --keep-range", run.stdout)
        as_json = _run(["close", "job-1", "--json"], _close_hub(JOBS[0]))
        self.assertEqual(json.loads(as_json.stdout)["run"]["id"], "job-1")
        self.assertEqual(_posts(as_json), [])

    def test_close_takes_a_trial_id_for_its_run(self) -> None:
        run = _run(["close", "trial-1"], _close_hub())
        self.assertEqual(run.code, 0, run.output)
        self.assertEqual(_posts(run), ["/hub/eval/jobs/job-1/close-range"])

    def test_another_users_run_reads_as_not_found(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"errorMessage": "Not found"})

        run = _run(["close", "job-9"], handler)
        self.assertEqual(run.code, 1)
        self.assertEqual(_posts(run), [])
        self.assertIn("no run or trial found with id 'job-9'", run.stderr)

    def test_close_json_prints_the_closed_run(self) -> None:
        run = _run(["close", "job-1", "--json"], _close_hub())
        self.assertEqual(run.code, 0, run.output)
        self.assertEqual(json.loads(run.stdout)["run"]["rangeClosedAt"], CLOSED_AT)

    def test_run_never_says_kept_open_once_every_kept_range_was_removed(self) -> None:
        # The owner removed both kept copies on the web: the range is still open, no range is up.
        removed = {**KEPT_JOB, "keptRangeCount": 0}
        run = _run(["run", "job-1"], _runs_hub(job=removed))
        self.assertEqual(run.code, 0, run.output)
        self.assertNotIn("range kept open", run.stdout)
        self.assertNotIn("rogue-arena-inspect close", run.stdout)
        self.assertIn("no kept range up", run.stdout)
        kept = _run(["run", "job-1"], _runs_hub(job={**KEPT_JOB, "keptRangeCount": 2}))
        self.assertIn("range kept open (2 up)", kept.stdout)

    def test_run_shows_a_kept_range_and_how_to_close_it(self) -> None:
        run = _run(["run", "job-1"], _runs_hub(job=KEPT_JOB))
        self.assertEqual(run.code, 0, run.output)
        self.assertIn("range kept open", run.stdout)
        self.assertIn("rogue-arena-inspect close job-1", run.stdout)
        closed = _run(
            ["run", "job-1"], _runs_hub(job={**KEPT_JOB, "rangeClosedAt": CLOSED_AT})
        )
        self.assertIn("range closed", closed.stdout)
        self.assertNotIn("rogue-arena-inspect close", closed.stdout)
        plain = _run(["run", "job-1"], _runs_hub())
        self.assertNotIn("range", plain.stdout.split("\n")[0])


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
    on_spawn: Callable[[list[str], dict[str, str]], None] | None = None,
    client_handler: Handler | None = None,
) -> tuple[int | str | None, list[str] | None, dict[str, str] | None, str]:
    """Run ``launch`` with ``subprocess.Popen`` stubbed; capture its cmd and env.

    ``returncode``/``child_stderr`` simulate the ``inspect eval`` child, so a
    failing run with provider-policy stderr can be exercised. ``on_spawn`` runs
    with the child's cmd and env (e.g. to write a ``.eval`` into its log dir),
    and ``client_handler`` puts every ``RogueArenaClient`` on a MockTransport,
    so the post-run transcript upload hits the fake hub, not the network.
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
            if on_spawn is not None:
                on_spawn(cmd, env)
            self.stderr = iter(list(child_stderr))

        def wait(self) -> int:
            return returncode

    stderr = io.StringIO()
    patches = [
        patch.dict(os.environ, dict(env), clear=True),
        patch("inspect_rogue_arena.cli.subprocess.Popen", _FakePopen),
        redirect_stdout(io.StringIO()),
        redirect_stderr(stderr),
    ]
    if client_handler is not None:
        patches.append(
            patch(
                "inspect_rogue_arena.cli.RogueArenaClient",
                partial(
                    RogueArenaClient, transport=httpx.MockTransport(client_handler)
                ),
            )
        )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
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

    def test_launch_without_share_transcript_sets_no_run_id_and_no_log_dir(
        self,
    ) -> None:
        code, cmd, env, _ = _launch(["launch", "--model", "m", "--eval", "x"])
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertNotIn("ROGUE_ARENA_RUN_ID", env)
        self.assertNotIn("--log-dir", cmd)

    def test_launch_keep_range_sets_the_env_and_names_each_kept_run_after(
        self,
    ) -> None:
        run_ids: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if (request.method, request.url.path) == ("GET", "/hub/eval/jobs"):
                run_id = run_ids[0]
                return httpx.Response(
                    200,
                    json=[
                        {
                            **KEPT_JOB,
                            "id": "job-kept",
                            "clientJobKey": f"inspect_{run_id}-a",
                        },
                        # Another launch's kept run is not this launch's to report.
                        {
                            **KEPT_JOB,
                            "id": "job-other",
                            "clientJobKey": "inspect_other-a",
                        },
                        # A run with no trials kept nothing.
                        {
                            **KEPT_JOB,
                            "id": "job-empty",
                            "clientJobKey": f"inspect_{run_id}-b",
                            "trialCount": 0,
                        },
                    ],
                )
            return httpx.Response(404, json={"errorMessage": "not found"})

        code, _, env, stderr = _launch(
            ["launch", "--model", "mockllm/model", "--keep-range"],
            on_spawn=lambda _cmd, child_env: run_ids.append(
                child_env["ROGUE_ARENA_RUN_ID"]
            ),
            client_handler=handler,
        )
        assert env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_KEEP_RANGE"], "true")
        self.assertIn("rogue-arena-inspect close job-kept", stderr)
        self.assertNotIn("job-other", stderr)
        self.assertNotIn("job-empty", stderr)
        self.assertNotIn(TOKEN, stderr)

    def test_launch_without_keep_range_leaves_the_env_var_unset(self) -> None:
        _, _, env, stderr = _launch(["launch", "--model", "mockllm/model"])
        assert env is not None
        self.assertNotIn("ROGUE_ARENA_KEEP_RANGE", env)
        self.assertNotIn("--keep-range", stderr)

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


def _guide_catalog_hub() -> Handler:
    """A hub with the guide route and a listing that names the eval."""
    guide = _guide_hub()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/hub/eval/evals":
            return httpx.Response(
                200,
                json=[{**EVALS[0], "environmentId": GUIDE_ENV_ID, "name": "ad-chain"}],
            )
        return guide(request)

    return handler


class CliLaunchPromptLevelTests(unittest.TestCase):
    """`launch --prompt-level` gives the model the eval author's prompt, nothing else."""

    def _launch_at(self, level: str, selector: str = GUIDE_ENV_ID) -> tuple[Any, ...]:
        return _launch(
            ["launch", "--model", "m", "--eval", selector, "--prompt-level", level],
            client_handler=_guide_catalog_hub(),
        )

    def test_the_prompt_reaches_the_child_and_the_graders_never_do(self) -> None:
        code, cmd, env, _ = self._launch_at("hard")
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_PROMPT"], "win")
        child_view = " ".join(cmd) + " ".join(env.values())
        for secret in ("/root/flag1.txt", "web -> app VLAN -> dc", "whoami"):
            self.assertNotIn(secret, child_view)

    def test_an_eval_name_resolves_to_its_guide(self) -> None:
        code, _, env, _ = self._launch_at("easy", selector="ad-chain")
        assert env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_PROMPT"], "do the easy thing")

    def test_a_blank_prompt_stops_before_the_child_starts(self) -> None:
        code, cmd, _, stderr = self._launch_at("medium")
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("no medium recommended prompt", stderr)

    def test_it_needs_an_eval_to_read_the_prompt_from(self) -> None:
        code, cmd, _, stderr = _launch(
            ["launch", "--model", "m", "--prompt-level", "hard"],
            client_handler=_guide_catalog_hub(),
        )
        self.assertEqual(code, 1)
        self.assertIsNone(cmd)
        self.assertIn("--prompt-level needs an eval", stderr)

    def test_without_the_flag_no_prompt_is_set(self) -> None:
        _, _, env, _ = _launch(["launch", "--model", "m", "--eval", "x"])
        assert env is not None
        self.assertNotIn("ROGUE_ARENA_PROMPT", env)


class ExampleTaskPromptTests(unittest.TestCase):
    """The built-in tasks use the prompt `launch` chose, and their own without one."""

    def _load(self, name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            f"_example_{name}", EXAMPLES_DIR / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_the_solo_task_uses_the_chosen_prompt(self) -> None:
        task = self._load("eval_example").rogue_arena_eval
        with patch.dict(os.environ, {"ROGUE_ARENA_PROMPT": "the author's"}):
            self.assertEqual(task().dataset[0].input, "the author's")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("WIN CONDITION", task().dataset[0].input)

    def test_the_team_task_uses_the_chosen_prompt(self) -> None:
        task = self._load("team_example").rogue_arena_team
        team = {"ROGUE_ARENA_TEAM": str(EXAMPLES_DIR / "team.yaml")}
        with patch.dict(os.environ, {**team, "ROGUE_ARENA_PROMPT": "the author's"}):
            self.assertEqual(task().dataset[0].input, "the author's")
        with patch.dict(os.environ, team, clear=True):
            self.assertIn("WIN CONDITION", task().dataset[0].input)


class CliShareTranscriptTests(unittest.TestCase):
    """`launch --share-transcript` uploads the finished run's .eval."""

    JOB_ID = "33333333-3333-4333-8333-333333333333"
    LOG_NAME = "2026-11-05T00-00-00_task_abc.eval"

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.log_dir = Path(directory.name)
        self.run_id: str | None = None
        self.uploads: list[dict[str, object]] = []
        # Only a log the test tagged belongs to this launch; any other .eval
        # in the folder stands for another run's.
        self.own_logs = {self.LOG_NAME}
        self.log_models: dict[str, str] = {}
        self.log_statuses: dict[str, str] = {}
        patcher = patch(
            "inspect_rogue_arena.cli._log_facts",
            lambda path: _LogFacts(
                self.run_id if path.name in self.own_logs else "someone-else",
                self.log_models.get(path.name),
                self.log_statuses.get(path.name, "success"),
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _hub(
        self, jobs: int = 1, models: Sequence[str | None] = (), empty_jobs: int = 0
    ) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            if (request.method, request.url.path) == ("GET", "/hub/eval/jobs"):
                other = {**JOBS[0], "id": "other", "clientJobKey": "inspect_other"}
                own = [
                    {
                        **JOBS[0],
                        "id": self.JOB_ID if index == 0 else f"job-{index}",
                        "status": "cancelled",
                        "clientJobKey": f"inspect_{self.run_id}-{index}",
                        "model": models[index] if index < len(models) else None,
                    }
                    for index in range(jobs)
                ]
                empty = [
                    {
                        **JOBS[0],
                        "id": f"empty-{index}",
                        "status": "cancelled",
                        "trialCount": 0,
                        "closedTrialCount": 0,
                        "clientJobKey": f"inspect_{self.run_id}-empty{index}",
                    }
                    for index in range(empty_jobs)
                ]
                return httpx.Response(200, json=[other, *own, *empty])
            if request.method == "PUT" and request.url.path.endswith("/transcript"):
                data = request.content
                self.uploads.append(
                    {
                        "path": request.url.path,
                        "headers": dict(request.headers),
                        "size": len(data),
                    }
                )
                return httpx.Response(
                    200,
                    json={
                        "transcript": {
                            "id": "t-9",
                            "jobId": self.JOB_ID,
                            "status": "ready",
                            "byteLength": len(data),
                            "sha256": request.headers["x-eval-transcript-sha256"],
                            "fileName": request.headers.get(
                                "x-eval-transcript-file-name"
                            ),
                            "uploadedAt": "2026-11-05T00:00:00.000Z",
                            "readyAt": "2026-11-05T00:00:01.000Z",
                            "expiresAt": "2026-12-05T00:00:00.000Z",
                        },
                        "replayed": False,
                    },
                )
            return httpx.Response(404, json={"errorMessage": "nope"})

        return handler

    def _on_spawn(
        self, *names: str, contents: bytes = b"EVAL-BYTES"
    ) -> Callable[[list[str], dict[str, str]], None]:
        def on_spawn(cmd: list[str], env: dict[str, str]) -> None:
            self.run_id = env["ROGUE_ARENA_RUN_ID"]
            for name in names:
                (self.log_dir / name).write_bytes(contents)

        return on_spawn

    def _argv(self) -> list[str]:
        return [
            "launch",
            "--model",
            "m",
            "--eval",
            "x",
            "--share-transcript",
            "--",
            "--log-dir",
            str(self.log_dir),
        ]

    def test_launch_uploads_the_runs_eval_and_keeps_the_local_log(self) -> None:
        code, cmd, env, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=self._hub(),
        )
        assert cmd is not None and env is not None
        self.assertEqual(code, 0)
        # the child is tagged with the run id, in its env and in the log's metadata
        self.assertEqual(env["ROGUE_ARENA_RUN_ID"], self.run_id)
        self.assertEqual(
            cmd[cmd.index("--metadata") + 1], f"rogue_arena_run_id={self.run_id}"
        )
        self.assertEqual(len(self.uploads), 1)
        upload = self.uploads[0]
        self.assertEqual(upload["path"], f"/hub/eval/jobs/{self.JOB_ID}/transcript")
        self.assertEqual(upload["size"], len(b"EVAL-BYTES"))
        headers = upload["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["content-type"], "application/octet-stream")
        self.assertEqual(
            headers["x-eval-transcript-sha256"],
            hashlib.sha256(b"EVAL-BYTES").hexdigest(),
        )
        self.assertEqual(uuid.UUID(headers["idempotency-key"]).version, 4)
        self.assertEqual(headers["x-eval-transcript-file-name"], self.LOG_NAME)
        self.assertIn("shared transcript t-9", stderr)
        # the user's own log is still where Inspect wrote it
        self.assertEqual((self.log_dir / self.LOG_NAME).read_bytes(), b"EVAL-BYTES")

    def test_launch_leaves_the_log_dir_to_inspect(self) -> None:
        argv = ["launch", "--model", "m", "--eval", "x", "--share-transcript"]
        with tempfile.TemporaryDirectory() as cwd:
            previous = Path.cwd()
            os.chdir(cwd)
            try:
                code, cmd, _, stderr = _launch(argv)
            finally:
                os.chdir(previous)
        assert cmd is not None
        self.assertEqual(code, 0)
        self.assertNotIn("--log-dir", cmd)
        self.assertIn("found 0", stderr)

    def test_inspect_log_dir_env_is_where_it_looks(self) -> None:
        code, _, _, _ = _launch(
            ["launch", "--model", "m", "--eval", "x", "--share-transcript"],
            env={**ENV, "INSPECT_LOG_DIR": str(self.log_dir)},
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=self._hub(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(self.uploads), 1)

    def test_another_runs_log_in_the_same_folder_is_never_uploaded(self) -> None:
        code, _, _, _ = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME, "2026-11-05T00-00-01_other.eval"),
            client_handler=self._hub(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(self.uploads), 1)
        headers = self.uploads[0]["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["x-eval-transcript-file-name"], self.LOG_NAME)

    def test_an_older_log_is_not_read_as_this_runs(self) -> None:
        stale = self.log_dir / self.LOG_NAME
        stale.write_bytes(b"OLD")
        os.utime(stale, (1_000_000_000, 1_000_000_000))
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(),
            client_handler=self._hub(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.uploads, [])
        self.assertIn("found 0", stderr)

    def test_a_failed_run_shares_nothing(self) -> None:
        code, _, _, stderr = _launch(
            self._argv(),
            returncode=2,
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=self._hub(),
        )
        self.assertEqual(code, 2)
        self.assertEqual(self.uploads, [])
        self.assertIn("no transcript was shared", stderr)

    def test_no_eval_file_skips_the_upload_without_failing(self) -> None:
        code, _, _, stderr = _launch(
            self._argv(), on_spawn=self._on_spawn(), client_handler=self._hub()
        )
        self.assertEqual(code, 0)  # the eval still succeeded
        self.assertEqual(self.uploads, [])
        self.assertIn("found 0", stderr)

    def test_a_remote_log_dir_is_reported_not_read(self) -> None:
        code, _, _, stderr = _launch(
            [*self._argv()[:-1], "s3://bucket/logs"],
            on_spawn=lambda cmd, env: None,
            client_handler=self._hub(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.uploads, [])
        self.assertIn("local log folder", stderr)

    def test_a_multi_model_run_says_how_to_upload_each_log_to_its_run(self) -> None:
        other_log = "2026-11-05T00-00-01_task_def.eval"
        self.own_logs.add(other_log)
        self.log_models = {self.LOG_NAME: "mockllm/a", other_log: "mockllm/b"}
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME, other_log),
            client_handler=self._hub(jobs=2, models=["mockllm/a", "mockllm/b"]),
        )
        self.assertEqual(code, 0)  # the eval still succeeded
        self.assertEqual(self.uploads, [])
        self.assertIn("2 jobs", stderr)
        # each log is paired with the run of its own model
        for name, job in ((self.LOG_NAME, self.JOB_ID), (other_log, "job-1")):
            self.assertIn(
                f"share-transcript {self.log_dir / name} --run {job}",
                stderr,
            )

    def test_a_task_that_errored_shares_nothing_and_says_so(self) -> None:
        # Inspect can exit 0 when one task of a multi-model run errored, so the exit
        # code doesn't say which logs finished: each log's own status does.
        other_log = "2026-11-05T00-00-01_task_def.eval"
        self.own_logs.add(other_log)
        self.log_models = {self.LOG_NAME: "mockllm/a", other_log: "mockllm/b"}
        self.log_statuses = {other_log: "error"}
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME, other_log),
            client_handler=self._hub(jobs=1, models=["mockllm/a"]),
        )
        self.assertEqual(code, 0)
        # the finished log still goes up, to its own run
        self.assertEqual(len(self.uploads), 1)
        headers = self.uploads[0]["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["x-eval-transcript-file-name"], self.LOG_NAME)
        self.assertIn(f"not sharing {other_log}", stderr)
        self.assertIn("error", stderr)

    def test_only_unfinished_logs_upload_nothing(self) -> None:
        self.log_statuses = {self.LOG_NAME: "cancelled"}
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=self._hub(),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.uploads, [])
        self.assertIn(f"not sharing {self.LOG_NAME}", stderr)
        self.assertNotIn("found 0", stderr)

    def test_the_report_never_pairs_a_log_with_another_models_run(self) -> None:
        # One job left, but it is another model's: it must not be offered.
        other_log = "2026-11-05T00-00-01_task_def.eval"
        self.own_logs.add(other_log)
        self.log_models = {self.LOG_NAME: "mockllm/a", other_log: "mockllm/b"}
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME, other_log),
            client_handler=self._hub(jobs=1, models=["mockllm/a"]),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.uploads, [])
        self.assertIn(
            f"share-transcript {self.log_dir / self.LOG_NAME} --run {self.JOB_ID}",
            stderr,
        )
        self.assertIn(
            f"share-transcript {self.log_dir / other_log} --run <run>", stderr
        )

    def test_a_log_that_matches_no_job_names_the_runs_to_choose_from(self) -> None:
        other_log = "2026-11-05T00-00-01_task_def.eval"
        self.own_logs.add(other_log)
        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME, other_log),
            client_handler=self._hub(jobs=2, models=["mockllm/a", "mockllm/b"]),
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.uploads, [])
        self.assertIn(
            f"share-transcript {self.log_dir / self.LOG_NAME} --run <run>", stderr
        )
        self.assertIn(f"{self.JOB_ID} (mockllm/a)", stderr)
        self.assertIn("job-1 (mockllm/b)", stderr)

    def test_an_empty_job_is_not_one_of_the_runs_jobs(self) -> None:
        # e.g. a task with its own model: Inspect's startup ran under the --model.
        code, _, _, _ = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=self._hub(jobs=1, empty_jobs=1),
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(
            self.uploads[0]["path"], f"/hub/eval/jobs/{self.JOB_ID}/transcript"
        )

    def test_a_missing_run_list_endpoint_is_reported_not_fatal(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"errorMessage": "nope"})

        code, _, _, stderr = _launch(
            self._argv(),
            on_spawn=self._on_spawn(self.LOG_NAME),
            client_handler=handler,
        )
        self.assertEqual(code, 0)
        self.assertIn("could not share the transcript", stderr)
        self.assertNotIn(TOKEN, stderr)


class CliShareTranscriptCommandTests(unittest.TestCase):
    """`share-transcript FILE` uploads an existing .eval, e.g. after a failed upload."""

    JOB_ID = "33333333-3333-4333-8333-333333333333"
    RUN_ID = "44444444-4444-4444-8444-444444444444"

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "2026-11-05T00-00-00_task_abc.eval"
        self.path.write_bytes(b"EVAL-BYTES")
        self.tag: str | None = self.RUN_ID
        self.log_model: str | None = None
        self.uploads: list[httpx.Request] = []
        self.replayed = False
        patcher = patch(
            "inspect_rogue_arena.cli._log_facts",
            lambda path: _LogFacts(self.tag, self.log_model, "success"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _job(self, key: str | None = None) -> dict[str, object]:
        return {
            **JOBS[0],
            "id": self.JOB_ID,
            "status": "cancelled",
            "clientJobKey": f"inspect_{self.RUN_ID}-abc" if key is None else key,
        }

    def _hub(self, *, job_key: str | None = None) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if (request.method, path) == ("GET", "/hub/eval/jobs"):
                return httpx.Response(200, json=[self._job(job_key)])
            if (request.method, path) == ("GET", f"/hub/eval/jobs/{self.JOB_ID}"):
                return httpx.Response(200, json=self._job(job_key))
            if request.method == "PUT" and path.endswith("/transcript"):
                self.uploads.append(request)
                return httpx.Response(
                    200,
                    json={
                        "transcript": {
                            "id": "t-9",
                            "jobId": self.JOB_ID,
                            "status": "ready",
                            "byteLength": len(request.content),
                            "sha256": request.headers["x-eval-transcript-sha256"],
                            "fileName": None,
                            "uploadedAt": "2026-11-05T00:00:00.000Z",
                            "readyAt": "2026-11-05T00:00:01.000Z",
                            "expiresAt": "2026-12-05T00:00:00.000Z",
                        },
                        "replayed": self.replayed,
                    },
                )
            return httpx.Response(404, json={"errorMessage": "nope"})

        return handler

    def test_a_tagged_log_finds_its_own_run(self) -> None:
        run = _run(["share-transcript", str(self.path)], self._hub())
        self.assertEqual(run.code, 0, run.stderr)
        self.assertEqual(len(self.uploads), 1)
        self.assertEqual(
            self.uploads[0].url.path, f"/hub/eval/jobs/{self.JOB_ID}/transcript"
        )
        self.assertIn("shared transcript t-9", run.stdout)
        self.assertNotIn(TOKEN, run.stdout + run.stderr)

    def test_a_retry_uses_the_same_key_as_the_failed_upload(self) -> None:
        run = _run(["share-transcript", str(self.path)], self._hub())
        self.assertEqual(run.code, 0, run.stderr)
        self.assertEqual(
            self.uploads[0].headers["idempotency-key"],
            transcript_upload_key(
                self.JOB_ID, hashlib.sha256(b"EVAL-BYTES").hexdigest()
            ),
        )

    def test_a_replay_says_it_was_already_shared(self) -> None:
        self.replayed = True
        run = _run(["share-transcript", str(self.path)], self._hub())
        self.assertEqual(run.code, 0)
        self.assertIn("already shared", run.stdout)

    def test_new_upload_sends_a_fresh_key(self) -> None:
        run = _run(["share-transcript", str(self.path), "--new-upload"], self._hub())
        self.assertEqual(run.code, 0, run.stderr)
        key = self.uploads[0].headers["idempotency-key"]
        self.assertEqual(uuid.UUID(key).version, 4)
        self.assertNotEqual(
            key,
            transcript_upload_key(
                self.JOB_ID, hashlib.sha256(b"EVAL-BYTES").hexdigest()
            ),
        )

    def test_an_untagged_log_needs_the_run_named(self) -> None:
        self.tag = None
        run = _run(["share-transcript", str(self.path)], self._hub())
        self.assertEqual(run.code, 1)
        self.assertIn("--run", run.stderr)
        self.assertEqual(self.uploads, [])

    def test_an_untagged_log_goes_to_the_named_run(self) -> None:
        self.tag = None
        run = _run(
            ["share-transcript", str(self.path), "--run", self.JOB_ID], self._hub()
        )
        self.assertEqual(run.code, 0, run.stderr)
        self.assertEqual(len(self.uploads), 1)

    def test_a_tagged_log_is_refused_for_another_run(self) -> None:
        run = _run(
            ["share-transcript", str(self.path), "--run", self.JOB_ID],
            self._hub(job_key="inspect_someone-else-abc"),
        )
        self.assertEqual(run.code, 1)
        self.assertIn("belongs to a different run", run.stderr)
        self.assertEqual(self.uploads, [])

    def _two_jobs_hub(self) -> Handler:
        jobs = [
            {**self._job(), "model": "mockllm/a"},
            {
                **self._job(f"inspect_{self.RUN_ID}-def"),
                "id": "job-b",
                "model": "mockllm/b",
            },
        ]
        base = self._hub()

        def handler(request: httpx.Request) -> httpx.Response:
            if (request.method, request.url.path) == ("GET", "/hub/eval/jobs"):
                return httpx.Response(200, json=jobs)
            if (request.method, request.url.path) == ("GET", "/hub/eval/jobs/job-b"):
                return httpx.Response(200, json=jobs[1])
            return base(request)

        return handler

    def test_a_multi_model_log_finds_the_run_of_its_own_model(self) -> None:
        self.log_model = "mockllm/b"
        run = _run(["share-transcript", str(self.path)], self._two_jobs_hub())
        self.assertEqual(run.code, 0, run.stderr)
        self.assertEqual(self.uploads[0].url.path, "/hub/eval/jobs/job-b/transcript")

    def test_a_log_is_refused_for_another_models_run(self) -> None:
        self.log_model = "mockllm/a"
        run = _run(
            ["share-transcript", str(self.path), "--run", "job-b"],
            self._two_jobs_hub(),
        )
        self.assertEqual(run.code, 1)
        self.assertIn("belongs to a different run", run.stderr)
        self.assertEqual(self.uploads, [])

    def test_a_tagged_log_is_not_sent_to_a_lone_job_of_another_model(self) -> None:
        self.log_model = "mockllm/b"
        job = {**self._job(), "model": "mockllm/a"}

        def handler(request: httpx.Request) -> httpx.Response:
            if (request.method, request.url.path) == ("GET", "/hub/eval/jobs"):
                return httpx.Response(200, json=[job])
            return self._hub()(request)

        run = _run(["share-transcript", str(self.path)], handler)
        self.assertEqual(run.code, 1)
        self.assertIn("--run", run.stderr)
        self.assertEqual(self.uploads, [])

    def test_a_log_that_cannot_choose_between_runs_names_them(self) -> None:
        run = _run(["share-transcript", str(self.path)], self._two_jobs_hub())
        self.assertEqual(run.code, 1)
        self.assertIn("--run", run.stderr)
        self.assertIn("job-b (mockllm/b)", run.stderr)
        self.assertEqual(self.uploads, [])

    def test_a_missing_file_is_a_clear_error(self) -> None:
        run = _run(["share-transcript", str(self.path) + ".missing"], self._hub())
        self.assertEqual(run.code, 1)
        self.assertIn("not found", run.stderr)
        self.assertEqual(run.requests, ())

    def test_a_refused_upload_exits_nonzero_with_the_hubs_reason(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "PUT":
                return httpx.Response(
                    409,
                    json={
                        "errorMessage": "already has a transcript",
                        "errorSeverity": 5,
                    },
                )
            return self._hub()(request)

        run = _run(["share-transcript", str(self.path)], handler)
        self.assertEqual(run.code, 1)
        self.assertIn("already has a transcript", run.stderr)

    def test_the_launch_failure_message_points_at_the_command(self) -> None:
        patcher = patch(
            "inspect_rogue_arena.cli._logs_of_run",
            lambda *args: [(self.path, _LogFacts(self.RUN_ID, None, "success"))],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"errorMessage": "boom"})

        code, _, _, stderr = _launch(
            ["launch", "--model", "m", "--eval", "x", "--share-transcript"],
            client_handler=handler,
        )
        self.assertEqual(code, 0)
        self.assertIn(f"share-transcript {self.path}", stderr)


class LogRunIdTests(unittest.TestCase):
    """The run id `launch` tags a run with is the one it reads back from the log."""

    def test_a_real_eval_log_carries_the_run_metadata(self) -> None:
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import Sample

        with tempfile.TemporaryDirectory() as log_dir:
            inspect_eval(
                Task(dataset=[Sample(input="q")]),
                model="mockllm/model",
                metadata={"rogue_arena_run_id": "run-42"},
                log_dir=log_dir,
                display="none",
            )
            inspect_eval(
                Task(dataset=[Sample(input="q")]),
                model="mockllm/model",
                log_dir=log_dir,
                display="none",
            )
            found = sorted(
                (str(facts.run_id), str(facts.model), str(facts.status))
                for facts in map(_log_facts, Path(log_dir).glob("*.eval"))
            )

        self.assertEqual(
            found,
            [
                ("None", "mockllm/model", "success"),
                ("run-42", "mockllm/model", "success"),
            ],
        )

    def test_an_unreadable_log_has_no_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            path = Path(log_dir) / "broken.eval"
            path.write_bytes(b"not a log")
            self.assertEqual(_log_facts(path), _LogFacts(None, None, None))


class CliRunModelLabelTests(unittest.TestCase):
    """A run shows the model it evaluated, when it has one."""

    def _jobs_hub(self, model: str | None) -> Handler:
        job = {**JOBS[0], "model": model}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/hub/eval/jobs":
                return httpx.Response(200, json=[job])
            if request.url.path == "/hub/eval/jobs/job-1":
                return httpx.Response(200, json=job)
            if request.url.path == "/hub/eval/jobs/job-1/trials":
                return httpx.Response(200, json=[])
            return httpx.Response(404, json={"errorMessage": "nope"})

        return handler

    def test_runs_and_run_show_the_model(self) -> None:
        for argv in (["runs"], ["run", "job-1"]):
            with self.subTest(argv=argv):
                run = _run(argv, self._jobs_hub("openai/gpt-4o"))
                self.assertEqual(run.code, 0, run.stderr)
                self.assertIn("model openai/gpt-4o", run.stdout)

    def test_a_run_with_no_model_omits_it(self) -> None:
        run = _run(["runs"], self._jobs_hub(None))
        self.assertEqual(run.code, 0)
        self.assertNotIn("model", run.stdout)


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
