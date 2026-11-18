from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

from inspect_rogue_arena.cli import main
from inspect_rogue_arena.client import (
    EvalMachine,
    EvalTrial,
    EvalTrialEvent,
    EvalTrialTerminalReason,
    RogueArenaClient,
    TrialClosedBeforeReady,
    _claim_body,
    preseed_failure_output,
)
from inspect_rogue_arena.config import RogueArenaSandboxConfig
from inspect_rogue_arena.preseed import (
    PRESEED_DEFAULT_TIMEOUT_SECONDS,
    PRESEED_MAX_STEPS,
    PRESEED_MAX_TIMEOUT_SECONDS,
    PRESEED_SCRIPT_MAX_CHARS,
    PRESEED_STEP_ALLOWANCE_SECONDS,
    Preseed,
    PreseedError,
    PreseedShell,
    check_preseed_targets,
    load_preseed,
    parse_preseed,
)

BASE_URL = "https://hub.example.test"
TOKEN = "ra_eval_S3cretTokenValueNeverPrinted9f2c"
ENV = {"ROGUE_ARENA_BASE_URL": BASE_URL, "ROGUE_ARENA_TOKEN": TOKEN}
EVAL_ID = "11111111-1111-4111-8111-111111111111"
EVALS = [
    {
        "environmentId": EVAL_ID,
        "name": "example-ad-chain",
        "environmentBuildId": "build-1",
        "canvasVersionNumber": 3,
        "hopCount": 4,
        "agentMachineNickname": "KALI-01",
        "scoringMode": "sequential",
        "readiness": None,
    }
]
MACHINES = [
    {
        "machineId": "aaaaaaaa-0000-4000-8000-000000000001",
        "name": "KALI-01",
        "operatingSystem": "Kali",
        "agent": True,
    },
    {
        "machineId": "aaaaaaaa-0000-4000-8000-000000000002",
        "name": "WEB-01",
        "operatingSystem": "Ubuntu",
        "agent": False,
    },
    {
        "machineId": "aaaaaaaa-0000-4000-8000-000000000003",
        "name": "DC-01",
        "operatingSystem": "WindowsServer2022",
        "agent": False,
    },
]

YAML = """\
steps:
  - target: WEB-01
    shell: bash
    script: |
      echo seeded > /opt/notes.txt
  - target: aaaaaaaa-0000-4000-8000-000000000003
    shell: powershell
    timeout: 30
    script: Set-Content C:\\notes.txt seeded
"""


def _write(text: str, suffix: str = ".yaml") -> Path:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
        handle.write(text)
    return Path(handle.name)


class PreseedFileTests(unittest.TestCase):
    def test_a_yaml_file_parses_into_ordered_steps_with_the_default_timeout(
        self,
    ) -> None:
        path = _write(YAML)
        self.addCleanup(path.unlink)

        preseed = load_preseed(path)

        self.assertEqual(
            [step.target for step in preseed.steps],
            ["WEB-01", MACHINES[2]["machineId"]],
        )
        self.assertEqual(preseed.steps[0].shell, PreseedShell.BASH)
        self.assertEqual(
            preseed.steps[0].timeout_seconds, PRESEED_DEFAULT_TIMEOUT_SECONDS
        )
        self.assertEqual(preseed.steps[1].timeout_seconds, 30)
        self.assertEqual(preseed.steps[0].script, "echo seeded > /opt/notes.txt\n")
        self.assertEqual(
            preseed.to_wire()[1],
            {
                "target": MACHINES[2]["machineId"],
                "shell": "powershell",
                "timeoutSeconds": 30,
                "script": "Set-Content C:\\notes.txt seeded",
            },
        )
        self.assertEqual(
            preseed.wait_seconds,
            PRESEED_DEFAULT_TIMEOUT_SECONDS + 30 + 2 * PRESEED_STEP_ALLOWANCE_SECONDS,
        )

    def test_a_json_file_and_a_bare_list_parse_too(self) -> None:
        path = _write(
            json.dumps([{"target": "WEB-01", "shell": "bash", "script": "true"}]),
            ".json",
        )
        self.addCleanup(path.unlink)
        self.assertEqual(len(load_preseed(path).steps), 1)

    def test_invalid_documents_name_the_bad_step_and_field(self) -> None:
        step = {"target": "WEB-01", "shell": "bash", "script": "true"}
        cases = [
            ([], "at least one step"),
            ({"steps": [step], "extra": 1}, "unknown top-level"),
            ("just text", "list of steps"),
            ([{**step, "user": "root"}], "step 1: unknown field(s) user"),
            ([{**step, "target": " "}], "step 1: target"),
            ([{**step, "target": "x" * 201}], "longer than 200"),
            ([{**step, "shell": "zsh"}], "step 1: shell must be bash"),
            ([step, {**step, "timeout": 0}], "step 2: timeout must be 1 to"),
            (
                [{**step, "timeout": PRESEED_MAX_TIMEOUT_SECONDS + 1}],
                "timeout must be 1 to",
            ),
            ([{**step, "timeout": True}], "whole number of seconds"),
            ([{**step, "timeout": "600"}], "whole number of seconds"),
            ([{**step, "script": ""}], "script must be a non-empty string"),
            ([{**step, "script": 5}], "script must be a non-empty string"),
            ([step] * (PRESEED_MAX_STEPS + 1), f"at most {PRESEED_MAX_STEPS} steps"),
            (
                [{**step, "script": "x" * (PRESEED_SCRIPT_MAX_CHARS // 2 + 1)}] * 2,
                "scripts add up to",
            ),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(PreseedError) as raised:
                    parse_preseed(raw)
                self.assertIn(message, str(raised.exception))

    def test_a_missing_or_unparseable_file_is_a_clear_error(self) -> None:
        with self.assertRaises(PreseedError) as missing:
            load_preseed("/nonexistent/preseed.yaml")
        self.assertIn("cannot read preseed file", str(missing.exception))
        path = _write("steps: [unclosed")
        self.addCleanup(path.unlink)
        with self.assertRaises(PreseedError) as broken:
            load_preseed(path)
        self.assertIn("does not parse", str(broken.exception))

    def test_the_sandbox_config_reads_the_preseed_path_from_the_environment(
        self,
    ) -> None:
        config = RogueArenaSandboxConfig.from_sources(
            None, environ={"ROGUE_ARENA_PRESEED": " /tmp/p.yaml "}
        )
        self.assertEqual(config.preseed, "/tmp/p.yaml")
        self.assertIsNone(
            RogueArenaSandboxConfig.from_sources(None, environ={}).preseed
        )


class PreseedClaimTests(unittest.TestCase):
    def _body(self, preseed: Preseed | None = None) -> dict[str, Any]:
        return _claim_body(
            "job-1",
            client_attempt_key="k",
            inspect_sample_id="s",
            inspect_epoch=1,
            inspect_sample_uuid=None,
            inspect_eval_id=None,
            inspect_run_id=None,
            inspect_eval_set_id=None,
            inspect_task_id=None,
            sandbox_name="x",
            preseed=preseed,
        )

    def test_the_claim_carries_the_preseed_only_when_one_is_set(self) -> None:
        self.assertNotIn("preseed", self._body())
        preseed = parse_preseed(
            [{"target": "WEB-01", "shell": "bash", "script": "true"}]
        )
        self.assertEqual(
            self._body(preseed)["preseed"],
            [
                {
                    "target": "WEB-01",
                    "shell": "bash",
                    "timeoutSeconds": 600,
                    "script": "true",
                }
            ],
        )

    def test_a_preseed_failed_trial_parses_and_its_close_shows_the_steps_output(
        self,
    ) -> None:
        trial = EvalTrial.from_dict(
            {
                "id": "trial-1",
                "jobId": "job-1",
                "status": "closed",
                "outcome": "expired",
                "terminalReason": "preseed_failed",
                "totalSteps": 4,
                "preseeded": True,
                "closedAt": "2026-11-17T12:00:05Z",
            }
        )
        self.assertIs(trial.terminal_reason, EvalTrialTerminalReason.PRESEED_FAILED)
        self.assertTrue(trial.preseeded)
        events = [
            EvalTrialEvent.from_dict(row)
            for row in [
                {
                    "id": "e1",
                    "at": "2026-11-17T12:00:00Z",
                    "kind": "lifecycle",
                    "summary": "preseed step 1/1 on WEB-01 (bash): running (timeout 600 s)",
                    "detail": None,
                },
                {
                    "id": "e2",
                    "at": "2026-11-17T12:00:04Z",
                    "kind": "lifecycle",
                    "summary": "preseed step 1/1 on WEB-01 (bash): exit 3 after 1 s",
                    "detail": "permission denied\n",
                },
                {
                    "id": "e3",
                    "at": "2026-11-17T12:00:05Z",
                    "kind": "lifecycle",
                    "summary": "preseed failed: preseed step 1/1 on WEB-01 (bash): exit 3 after 1 s",
                    "detail": None,
                },
                {
                    "id": "e4",
                    "at": "2026-11-17T12:00:05Z",
                    "kind": "lifecycle",
                    "summary": "closed: expired (preseed_failed), 0/4 hops",
                    "detail": None,
                },
            ]
        ]
        self.assertEqual(preseed_failure_output(events), "permission denied\n")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "id": event.id,
                        "at": event.at,
                        "kind": event.kind.value,
                        "summary": event.summary,
                        "detail": event.detail,
                    }
                    for event in events
                ],
            )

        async def closed_reason() -> str | None:
            client = RogueArenaClient(
                BASE_URL, TOKEN, transport=httpx.MockTransport(handler)
            )
            return await client._closed_reason(trial)

        reason = asyncio.run(closed_reason())
        assert reason is not None
        self.assertIn(
            "preseed failed: preseed step 1/1 on WEB-01 (bash): exit 3", reason
        )
        self.assertIn("permission denied", reason)
        self.assertIn("preseed_failed", str(TrialClosedBeforeReady(trial, reason)))
        self.assertNotIn(TOKEN, reason)


class PreseedTargetTests(unittest.TestCase):
    machines = tuple(EvalMachine.from_dict(row) for row in MACHINES)

    def test_targets_resolve_by_id_or_name_in_any_case(self) -> None:
        check_preseed_targets(load_preseed_text(YAML), self.machines)
        check_preseed_targets(
            parse_preseed([{"target": "kali-01", "shell": "bash", "script": "true"}]),
            self.machines,
        )

    def test_an_unknown_target_or_the_wrong_shell_fails_before_any_job(self) -> None:
        with self.assertRaisesRegex(PreseedError, "names no machine of this eval"):
            check_preseed_targets(
                parse_preseed(
                    [{"target": "CoreRouter", "shell": "bash", "script": "x"}]
                ),
                self.machines,
            )
        with self.assertRaisesRegex(PreseedError, "must use powershell, not bash"):
            check_preseed_targets(
                parse_preseed([{"target": "DC-01", "shell": "bash", "script": "x"}]),
                self.machines,
            )

    def test_a_name_two_machines_carry_needs_the_id(self) -> None:
        twins = (
            *self.machines,
            EvalMachine.from_dict({**MACHINES[1], "machineId": "other"}),
        )
        with self.assertRaisesRegex(PreseedError, "names 2 machines"):
            check_preseed_targets(
                parse_preseed([{"target": "WEB-01", "shell": "bash", "script": "x"}]),
                twins,
            )

    def test_the_failure_output_is_the_last_steps_never_an_earlier_one(self) -> None:
        events = [
            EvalTrialEvent.from_dict(
                {
                    "id": "e1",
                    "at": "2026-11-17T12:00:00Z",
                    "kind": "lifecycle",
                    "summary": "preseed step 1/2 on WEB-01 (bash): exit 0 after 1 s",
                    "detail": "step one ok",
                }
            ),
            EvalTrialEvent.from_dict(
                {
                    "id": "e2",
                    "at": "2026-11-17T12:00:01Z",
                    "kind": "lifecycle",
                    "summary": "preseed step 2/2 on WEB-01 (bash): exit 1 after 0 s",
                    "detail": None,
                }
            ),
        ]
        self.assertIsNone(preseed_failure_output(events))


def load_preseed_text(text: str) -> Preseed:
    path = _write(text)
    try:
        return load_preseed(path)
    finally:
        path.unlink()


def _run(
    argv: list[str], routes: dict[tuple[str, str], object]
) -> tuple[int | str | None, str, str, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = routes.get((request.method, request.url.path))
        if body is None:
            return httpx.Response(404, json={"errorMessage": "Not found"})
        return httpx.Response(200, json=body)

    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        patch.dict(os.environ, ENV, clear=True),
        patch(
            "inspect_rogue_arena.cli.RogueArenaClient",
            partial(RogueArenaClient, transport=httpx.MockTransport(handler)),
        ),
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        try:
            code: int | str | None = main(argv)
        except SystemExit as exited:
            code = exited.code
    return code, stdout.getvalue(), stderr.getvalue(), requests


class DumpVmsTests(unittest.TestCase):
    machines_route = ("GET", f"/hub/eval/evals/{EVAL_ID}/machines")

    def test_dump_vms_lists_name_os_and_stable_id_with_marks(self) -> None:
        code, out, _, requests = _run(
            ["evals", "--dump-vms", EVAL_ID], {self.machines_route: MACHINES}
        )

        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(
            lines[0].split(),
            [
                "KALI-01",
                "Kali",
                "id",
                MACHINES[0]["machineId"],
                "(agent:",
                "the",
                "model's",
                "machine)",
            ],
        )
        self.assertIn("WEB-01", lines[1])
        self.assertIn(f"id {MACHINES[1]['machineId']}", lines[1])
        self.assertIn("DC-01", lines[2])
        self.assertNotIn("agent", lines[2])
        self.assertIn("base machines every range carries", out)
        self.assertEqual(
            [request.url.path for request in requests], [self.machines_route[1]]
        )

    def test_dump_vms_resolves_an_eval_name_and_prints_json(self) -> None:
        code, out, _, requests = _run(
            ["evals", "--dump-vms", "example-ad-chain", "--json"],
            {("GET", "/hub/eval/evals"): EVALS, self.machines_route: MACHINES},
        )

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), {"items": MACHINES})
        self.assertEqual(
            [request.url.path for request in requests],
            ["/hub/eval/evals", self.machines_route[1]],
        )

    def test_dump_vms_for_an_eval_the_token_cannot_run_is_a_clean_error(self) -> None:
        code, out, err, _ = _run(["evals", "--dump-vms", EVAL_ID], {})
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertIn("404", err)
        self.assertNotIn(TOKEN, err)


class LaunchPreseedTests(unittest.TestCase):
    def _launch(
        self, argv: list[str]
    ) -> tuple[int | str | None, dict[str, str] | None, str]:
        captured: dict[str, dict[str, str]] = {}

        class _FakePopen:
            def __init__(
                self,
                cmd: list[str],
                env: dict[str, str],
                stderr: object = None,
                text: bool = False,
                bufsize: int = -1,
            ) -> None:
                captured["env"] = env
                self.stderr = iter([])

            def wait(self) -> int:
                return 0

        stderr = io.StringIO()
        with (
            patch.dict(os.environ, ENV, clear=True),
            patch("inspect_rogue_arena.cli.subprocess.Popen", _FakePopen),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            try:
                code: int | str | None = main(argv)
            except SystemExit as exited:
                code = exited.code
        return code, captured.get("env"), stderr.getvalue()

    def test_launch_preseed_checks_the_file_and_hands_its_path_to_the_sandbox(
        self,
    ) -> None:
        path = _write(YAML)
        self.addCleanup(path.unlink)

        code, env, stderr = self._launch(
            ["launch", "--model", "mockllm/model", "--preseed", str(path)]
        )

        assert env is not None
        self.assertEqual(code, 0)
        self.assertEqual(env["ROGUE_ARENA_PRESEED"], str(path.resolve()))
        self.assertIn("--preseed: 2 step(s)", stderr)

    def test_launch_refuses_a_bad_preseed_before_starting_inspect(self) -> None:
        path = _write("steps:\n  - target: WEB-01\n    shell: zsh\n    script: true\n")
        self.addCleanup(path.unlink)

        code, env, stderr = self._launch(
            ["launch", "--model", "mockllm/model", "--preseed", str(path)]
        )

        self.assertEqual(code, 1)
        self.assertIsNone(env)
        self.assertIn("shell must be bash", stderr)

    def test_launch_without_preseed_leaves_the_env_var_unset(self) -> None:
        _, env, stderr = self._launch(["launch", "--model", "mockllm/model"])
        assert env is not None
        self.assertNotIn("ROGUE_ARENA_PRESEED", env)
        self.assertNotIn("--preseed", stderr)


if __name__ == "__main__":
    unittest.main()
