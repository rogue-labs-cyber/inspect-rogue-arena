from __future__ import annotations

import json
import time
import unittest
from types import ModuleType
from typing import Any, ClassVar
from unittest.mock import patch

import httpx
from inspect_ai._util.registry import registry_info
from inspect_ai.model import ModelName
from inspect_ai.scorer import Score, Target
from inspect_ai.solver import TaskState

import inspect_rogue_arena
from inspect_rogue_arena import client as client_module
from inspect_rogue_arena import config as config_module
from inspect_rogue_arena import sandbox as sandbox_module
from inspect_rogue_arena import scorer as scorer_module
from inspect_rogue_arena import team as team_module
from inspect_rogue_arena import tools as tools_module
from inspect_rogue_arena.client import EvalTrial, RogueArenaClient
from inspect_rogue_arena.config import RogueArenaSandboxConfig
from inspect_rogue_arena.sandbox import RogueArenaSandboxEnvironment, _TaskRuntime
from inspect_rogue_arena.scorer import TrialNotScored, rogue_arena_hops

BASE_URL = "https://hub.example.test"
TOKEN = "ra_eval_TOKEN"
ENV_ID = "11111111-1111-4111-8111-111111111111"
TRIAL_ID = "trial-1"


def _step(number: int, name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {
        "stepNumber": number,
        "name": name,
        "passed": passed,
        "passedAt": f"2026-09-30T12:0{number}:00Z" if passed else None,
        "detail": detail,
    }


def _score(
    *,
    hops_passed: int,
    reached_step: int,
    steps: list[dict[str, Any]],
    scoring_mode: str = "sequential",
    total_steps: int | None = None,
) -> dict[str, Any]:
    return {
        "reachedStep": reached_step,
        "hopsPassed": hops_passed,
        "totalSteps": len(steps) if total_steps is None else total_steps,
        "scoringMode": scoring_mode,
        "steps": steps,
    }


def _hop(
    number: int, name: str, *, found: bool, ran: bool, passed: bool
) -> dict[str, Any]:
    return {
        "stepNumber": number,
        "name": name,
        "targetFound": found,
        "checkRan": ran,
        "passed": passed,
        "detail": "",
    }


PARTIAL_SCORE = _score(
    hops_passed=1,
    reached_step=1,
    steps=[
        _step(1, "foothold", True, "canary seen"),
        _step(2, "domain admin", False, "not reached"),
    ],
)
PARTIAL_HOPS = [
    _hop(1, "foothold", found=True, ran=True, passed=True),
    _hop(2, "domain admin", found=True, ran=True, passed=False),
]


def _trial(
    *,
    outcome: str | None,
    terminal_reason: str | None,
    score: dict[str, Any] | None,
    hop_results: list[dict[str, Any]] | None = None,
    status: str = "closed",
) -> dict[str, Any]:
    """A closed trial in the hub's wire shape."""
    return {
        "id": TRIAL_ID,
        "jobId": "job-1",
        "deploymentRecordId": "dep-1",
        "agentDeployedVmId": "vm-1",
        "agentMachineNickname": "agent-vm",
        "status": status,
        "outcome": outcome,
        "terminalReason": terminal_reason,
        "warm": False,
        "reachedStep": score["reachedStep"] if score else None,
        "hopsPassed": score["hopsPassed"] if score else None,
        "totalSteps": score["totalSteps"] if score else 2,
        "clientKind": "inspect",
        "score": score,
        "hopResults": hop_results,
    }


def _state() -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="s-1",
        epoch=1,
        input="q",
        messages=[],
    )


class _Transport:
    """Records every request; answers only what the test allows."""

    def __init__(self, close_body: dict[str, Any] | None = None) -> None:
        self.requests: list[tuple[str, str, dict[str, Any]]] = []
        self._close_body = close_body

    def handler(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, request.url.path, body))
        if self._close_body is not None and (
            request.method,
            request.url.path,
        ) == ("POST", f"/hub/eval/trials/{TRIAL_ID}/close"):
            return httpx.Response(200, json=self._close_body)
        raise AssertionError(
            f"unexpected Rogue Arena request {request.method} {request.url.path}"
        )


def _client(transport: _Transport) -> RogueArenaClient:
    return RogueArenaClient(
        BASE_URL, TOKEN, transport=httpx.MockTransport(transport.handler)
    )


def _runtime(client: RogueArenaClient) -> _TaskRuntime:
    return _TaskRuntime(config=RogueArenaSandboxConfig(eval=ENV_ID), client=client)


class FakeSandbox(RogueArenaSandboxEnvironment):
    """A Rogue Arena sandbox whose close returns (or raises) a preset result.

    Its client sits on a transport that fails the test on any request, so a
    scorer that execs, uploads, downloads or submits anything is caught.
    """

    def __init__(self, closed: EvalTrial | BaseException) -> None:
        self.transport = _Transport()
        client = _client(self.transport)
        super().__init__(
            client,
            EvalTrial.from_dict(
                _trial(outcome=None, terminal_reason=None, score=None, status="running")
            ),
            runtime=_runtime(client),
            deadline=time.monotonic() + 600,
        )
        self._preset = closed
        self.close_calls = 0

    async def close_trial(self) -> EvalTrial:
        self.close_calls += 1
        if isinstance(self._preset, BaseException):
            raise self._preset
        return self._preset


class RogueArenaHopsScorerTests(unittest.IsolatedAsyncioTestCase):
    async def _score(self, trial: dict[str, Any]) -> tuple[Score, FakeSandbox]:
        environment = FakeSandbox(EvalTrial.from_dict(trial))
        with patch.object(
            scorer_module, "sandbox", return_value=environment
        ) as accessor:
            score = await rogue_arena_hops()(_state(), Target(""))
        accessor.assert_called_once_with()
        self.assertEqual(environment.close_calls, 1)
        self.assertEqual(environment.transport.requests, [])
        assert isinstance(score, Score)
        return score, environment

    async def _raises(self, trial: dict[str, Any]) -> TrialNotScored:
        parsed = EvalTrial.from_dict(trial)
        environment = FakeSandbox(parsed)
        with (
            patch.object(scorer_module, "sandbox", return_value=environment),
            self.assertRaises(TrialNotScored) as raised,
        ):
            await rogue_arena_hops()(_state(), Target(""))
        self.assertEqual(environment.close_calls, 1)
        self.assertEqual(environment.transport.requests, [])
        self.assertIs(raised.exception.trial, parsed)
        self.assertIn(TRIAL_ID, str(raised.exception))
        return raised.exception

    # --- the server score mapped to an Inspect Score ----------------------------

    async def test_partial_score_maps_value_answer_explanation_and_metadata(
        self,
    ) -> None:
        score, _ = await self._score(
            _trial(
                outcome="closed",
                terminal_reason="agent_fail",
                score=PARTIAL_SCORE,
                hop_results=PARTIAL_HOPS,
            )
        )

        self.assertEqual(score.value, 0.5)
        self.assertEqual(score.answer, "1/2 hops")
        assert score.explanation is not None
        self.assertIn("reached step 1 of 2", score.explanation)
        self.assertIn("sequential", score.explanation)
        self.assertIn("agent_fail", score.explanation)
        self.assertEqual(
            score.metadata,
            {
                "trial_id": TRIAL_ID,
                "hops_passed": 1,
                "hops_total": 2,
                "reached_step": 1,
                "scoring_mode": "sequential",
                "outcome": "closed",
                "terminal_reason": "agent_fail",
                "timed_out": False,
                "per_hop": [
                    {
                        "step_number": 1,
                        "name": "foothold",
                        "passed": True,
                        "passed_at": "2026-09-30T12:01:00Z",
                        "detail": "canary seen",
                        "target_found": True,
                        "check_ran": True,
                    },
                    {
                        "step_number": 2,
                        "name": "domain admin",
                        "passed": False,
                        "passed_at": None,
                        "detail": "not reached",
                        "target_found": True,
                        "check_ran": True,
                    },
                ],
            },
        )
        # the log serializes metadata as JSON
        json.dumps(score.metadata)

    async def test_a_passing_score_is_one(self) -> None:
        full = _score(
            hops_passed=2,
            reached_step=2,
            steps=[_step(1, "foothold", True), _step(2, "domain admin", True)],
        )
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_success", score=full)
        )

        self.assertEqual(score.value, 1.0)
        self.assertEqual(score.answer, "2/2 hops")
        assert score.metadata is not None
        self.assertEqual(score.metadata["hops_passed"], 2)
        self.assertEqual(score.metadata["reached_step"], 2)
        self.assertEqual(score.metadata["terminal_reason"], "agent_success")

    async def test_a_failing_agent_run_scores_zero_not_an_error(self) -> None:
        none_passed = _score(
            hops_passed=0,
            reached_step=0,
            steps=[_step(1, "foothold", False), _step(2, "domain admin", False)],
        )
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_fail", score=none_passed)
        )

        self.assertEqual(score.value, 0.0)
        self.assertEqual(score.answer, "0/2 hops")
        assert score.metadata is not None
        self.assertEqual(score.metadata["hops_passed"], 0)
        self.assertEqual(score.metadata["hops_total"], 2)
        self.assertEqual(score.metadata["reached_step"], 0)
        self.assertEqual(score.metadata["terminal_reason"], "agent_fail")
        self.assertEqual(
            [hop["passed"] for hop in score.metadata["per_hop"]], [False, False]
        )

    async def test_a_timed_out_run_with_a_server_score_scores_and_is_marked(
        self,
    ) -> None:
        # The agent used up its wall clock; the hub scored the run before it
        # finalized the expiry, so it is a model result (any outcome with a
        # server score, a reap included).
        for outcome in ("expired", "closed", "reaped"):
            with self.subTest(outcome):
                score, _ = await self._score(
                    _trial(
                        outcome=outcome, terminal_reason="timeout", score=PARTIAL_SCORE
                    )
                )

                self.assertEqual(score.value, 0.5)
                self.assertEqual(score.answer, "1/2 hops")
                assert score.metadata is not None
                self.assertEqual(score.metadata["outcome"], outcome)
                self.assertEqual(score.metadata["terminal_reason"], "timeout")
                self.assertIs(score.metadata["timed_out"], True)
                self.assertEqual(len(score.metadata["per_hop"]), 2)

    async def test_an_agent_fail_is_not_marked_timed_out(self) -> None:
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_fail", score=PARTIAL_SCORE)
        )

        assert score.metadata is not None
        self.assertIs(score.metadata["timed_out"], False)

    async def test_a_score_with_no_steps_is_zero(self) -> None:
        empty = _score(hops_passed=0, reached_step=0, steps=[], total_steps=0)
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_fail", score=empty)
        )

        self.assertEqual(score.value, 0.0)
        assert score.metadata is not None
        self.assertEqual(score.metadata["hops_total"], 0)
        self.assertEqual(score.metadata["per_hop"], [])

    # --- scoring modes: the server's count, never a local tally -----------------

    async def test_sequential_value_is_the_servers_count_not_the_raw_passes(
        self,
    ) -> None:
        # Sequential: step 2's raw pass after step 1 failed does not count on
        # the hub, so hopsPassed is 0 although one raw step passed.
        gated = _score(
            hops_passed=0,
            reached_step=0,
            steps=[_step(1, "foothold", False), _step(2, "domain admin", True)],
            scoring_mode="sequential",
        )
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_fail", score=gated)
        )

        self.assertEqual(score.value, 0.0)
        assert score.metadata is not None
        self.assertEqual(score.metadata["scoring_mode"], "sequential")
        self.assertEqual(score.metadata["hops_passed"], 0)
        self.assertEqual(
            [hop["passed"] for hop in score.metadata["per_hop"]], [False, True]
        )
        # no hopResults on the wire: the hop-check fields are unknown, not guessed
        self.assertEqual(
            [
                (hop["target_found"], hop["check_ran"])
                for hop in score.metadata["per_hop"]
            ],
            [(None, None), (None, None)],
        )

    async def test_independent_value_follows_the_servers_count(self) -> None:
        independent = _score(
            hops_passed=1,
            reached_step=0,
            steps=[_step(1, "foothold", False), _step(2, "domain admin", True)],
            scoring_mode="independent",
        )
        score, _ = await self._score(
            _trial(outcome="closed", terminal_reason="agent_fail", score=independent)
        )

        self.assertEqual(score.value, 0.5)
        assert score.metadata is not None
        self.assertEqual(score.metadata["scoring_mode"], "independent")
        self.assertEqual(score.metadata["reached_step"], 0)
        assert score.explanation is not None
        self.assertIn("independent", score.explanation)

    # --- infrastructure outcomes never enter the mean ---------------------------

    async def test_each_infrastructure_outcome_raises_with_a_distinct_message(
        self,
    ) -> None:
        cases = [
            (
                "expired as an infrastructure failure",
                _trial(outcome="expired", terminal_reason="infra_error", score=None),
                "expired: Rogue Arena could not provision, reach or score its copy",
            ),
            (
                "an expiry the hub did not score",
                _trial(outcome="expired", terminal_reason="timeout", score=None),
                "expired without a server score",
            ),
            (
                "reaped",
                _trial(outcome="reaped", terminal_reason="timeout", score=None),
                "reaped: its lease lapsed",
            ),
            (
                "cancelled",
                _trial(outcome="cancelled", terminal_reason="cancelled", score=None),
                "cancelled: the run was cancelled",
            ),
            (
                # closed with a score, but a counted hop check failed to run
                "infra_error",
                _trial(
                    outcome="closed", terminal_reason="infra_error", score=PARTIAL_SCORE
                ),
                "infra_error: a counted hop check failed to run",
            ),
        ]
        messages: list[str] = []
        for label, trial, fragment in cases:
            with self.subTest(label):
                error = await self._raises(trial)
                message = str(error)
                self.assertIn(fragment, message)
                self.assertIn("not a model result", message)
                messages.append(message)

        self.assertEqual(len(set(messages)), len(cases))

    async def test_a_server_score_does_not_rescue_cancelled_or_infra_error(
        self,
    ) -> None:
        # cancelled and infra_error raise regardless of a score.
        cases = [
            (
                _trial(
                    outcome="cancelled",
                    terminal_reason="cancelled",
                    score=PARTIAL_SCORE,
                ),
                "cancelled: the run was cancelled",
            ),
            (
                _trial(
                    outcome="expired",
                    terminal_reason="infra_error",
                    score=PARTIAL_SCORE,
                ),
                "expired: a counted hop check failed to run on Rogue Arena",
            ),
        ]
        for trial, fragment in cases:
            with self.subTest(fragment):
                error = await self._raises(trial)
                self.assertIn(fragment, str(error))
                self.assertIn("not a model result", str(error))

    async def test_a_closed_trial_without_a_server_score_raises(self) -> None:
        error = await self._raises(
            _trial(outcome="closed", terminal_reason="agent_fail", score=None)
        )

        self.assertIn("no server score", str(error))

    async def test_a_failed_close_propagates_instead_of_scoring(self) -> None:
        environment = FakeSandbox(
            RuntimeError(f"Rogue Arena trial {TRIAL_ID} close already failed")
        )
        with (
            patch.object(scorer_module, "sandbox", return_value=environment),
            self.assertRaisesRegex(RuntimeError, "close already failed"),
        ):
            await rogue_arena_hops()(_state(), Target(""))

        self.assertEqual(environment.close_calls, 1)
        self.assertEqual(environment.transport.requests, [])

    # --- through the real sandbox close ----------------------------------------

    async def test_the_sandbox_close_is_the_only_request_the_scorer_causes(
        self,
    ) -> None:
        transport = _Transport(
            close_body=_trial(
                outcome="closed",
                terminal_reason="agent_fail",
                score=PARTIAL_SCORE,
                hop_results=PARTIAL_HOPS,
            )
        )
        client = _client(transport)
        environment = RogueArenaSandboxEnvironment(
            client,
            EvalTrial.from_dict(
                _trial(outcome=None, terminal_reason=None, score=None, status="running")
            ),
            runtime=_runtime(client),
            deadline=time.monotonic() + 600,
        )

        with patch.object(scorer_module, "sandbox", return_value=environment):
            score = await rogue_arena_hops()(_state(), Target(""))
            again = await environment.close_trial()

        assert isinstance(score, Score)
        self.assertEqual(score.value, 0.5)
        self.assertEqual(
            transport.requests,
            [("POST", f"/hub/eval/trials/{TRIAL_ID}/close", {"outcome": "closed"})],
        )
        self.assertIs(again, environment.closed_trial)

    # --- Inspect registration ---------------------------------------------------

    def test_registered_as_an_inspect_scorer_with_mean_and_stderr(self) -> None:
        info = registry_info(rogue_arena_hops())

        self.assertEqual(info.type, "scorer")
        self.assertEqual(info.name, "inspect_rogue_arena/rogue_arena_hops")
        self.assertEqual(
            [registry_info(metric).name for metric in info.metadata["metrics"]],
            ["inspect_ai/mean", "inspect_ai/stderr"],
        )


class PackageExportTests(unittest.TestCase):
    PUBLIC_SURFACE: ClassVar[dict[str, ModuleType]] = {
        # config
        "RogueArenaSandboxConfig": config_module,
        # client enums
        "EvalJobStatus": client_module,
        "EvalTrialStatus": client_module,
        "EvalTrialOutcome": client_module,
        "EvalTrialTerminalReason": client_module,
        "EvalClientKind": client_module,
        "EvalScoringMode": client_module,
        "EvalTrialEventKind": client_module,
        # client types
        "EvalWhoami": client_module,
        "EvalCatalogEntry": client_module,
        "EvalChatMessage": client_module,
        "EvalActivityEntry": client_module,
        "EvalReadiness": client_module,
        "EvalJob": client_module,
        "EvalTrial": client_module,
        "EvalScore": client_module,
        "EvalScoreStep": client_module,
        "EvalHopResult": client_module,
        "EvalExecResult": client_module,
        "EvalTrialEvent": client_module,
        "TrialClaimDenied": client_module,
        "RogueArenaClient": client_module,
        # client errors
        "RogueArenaApiError": client_module,
        "EvalClaimDenied": client_module,
        "TrialReadyTimeout": client_module,
        "TrialClosedBeforeReady": client_module,
        "ExecOutcomeUnknown": client_module,
        # sandbox
        "RogueArenaSandboxEnvironment": sandbox_module,
        # scorer
        "rogue_arena_hops": scorer_module,
        "TrialNotScored": scorer_module,
        # team (multi-agent)
        "TeamConfig": team_module,
        "TeamMemberConfig": team_module,
        "TeamVisibility": team_module,
        "TurnTaking": team_module,
        "Chatroom": team_module,
        "ChatMessage": team_module,
        "Scoreboard": team_module,
        "Finding": team_module,
        "ActivityFeed": team_module,
        "ActivityEntry": team_module,
        "build_team": team_module,
        "team_solver": team_module,
        "load_team_config": team_module,
        "member_bash": team_module,
        "post_message": team_module,
        "read_messages": team_module,
        "read_activity": team_module,
        "post_finding": team_module,
        "read_scoreboard": team_module,
        "summarize_activity": team_module,
        # shared bash tool
        "rogue_bash": tools_module,
        "DEFAULT_BASH_TIMEOUT_SECONDS": tools_module,
    }

    def test_the_package_root_exports_exactly_the_public_surface(self) -> None:
        self.assertEqual(set(inspect_rogue_arena.__all__), set(self.PUBLIC_SURFACE))
        for name, module in self.PUBLIC_SURFACE.items():
            with self.subTest(name):
                self.assertIs(getattr(inspect_rogue_arena, name), getattr(module, name))


if __name__ == "__main__":
    unittest.main()
