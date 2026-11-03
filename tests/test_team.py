from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib import resources
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.agent import handoff, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.tool import ToolError
from inspect_ai.tool._tool_def import ToolDef
from inspect_ai.util import ExecResult, SandboxEnvironmentSpec, Store, store_as
from inspect_ai.util._sandbox.context import (
    sandbox_default_context_var,
    sandbox_environments_context_var,
)
from inspect_ai.util._sandbox.environment import SandboxEnvironment
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy
from inspect_ai.util._store import init_subtask_store

# Reuse the offline fake hub and helpers from the sandbox tests (the tests
# directory is on sys.path under `unittest discover`).
from test_sandbox import FakeHub, _config, _without_rogue_arena_env

import inspect_rogue_arena.team as team_mod
from inspect_rogue_arena.client import EvalTrial, RogueArenaClient
from inspect_rogue_arena.sandbox import RogueArenaSandboxEnvironment, _TaskRuntime
from inspect_rogue_arena.scorer import rogue_arena_hops
from inspect_rogue_arena.team import (
    ActivityFeed,
    Chatroom,
    Scoreboard,
    TeamConfig,
    TeamMemberConfig,
    TeamVisibility,
    TurnTaking,
    build_team,
    load_team_config,
    member_bash,
    post_finding,
    post_message,
    read_activity,
    read_messages,
    read_scoreboard,
    summarize_activity,
)

_TWO_MEMBERS = {
    "members": [
        {
            "name": "recon",
            "model": "mockllm/model",
            "prompt": "do recon",
            "role": "scout",
        },
        {"name": "exploit", "model": "mockllm/model", "prompt": "exploit it"},
    ]
}


def _reset_context_var(var: Any, token: Any) -> None:
    """Undo a test's ContextVar.set, tolerating Python 3.10's async cleanups.

    On Python 3.10, IsolatedAsyncioTestCase runs a cleanup in a different
    context than the test body, so the body's token cannot be reset there. That
    body context is discarded with the test, so there is nothing to restore.
    """
    try:
        var.reset(token)
    except ValueError:
        pass


class TeamConfigTests(unittest.TestCase):
    def test_loads_yaml_and_json_equivalently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            yaml_path = Path(directory) / "team.yaml"
            yaml_path.write_text(
                "turn_taking: round_robin\n"
                "rounds: 2\n"
                "members:\n"
                "  - name: recon\n    model: mockllm/model\n    prompt: do recon\n"
                "  - name: exploit\n    model: mockllm/model\n    prompt: exploit it\n"
            )
            json_path = Path(directory) / "team.json"
            json_path.write_text(
                json.dumps(
                    {
                        "turn_taking": "round_robin",
                        "rounds": 2,
                        "members": [
                            {
                                "name": "recon",
                                "model": "mockllm/model",
                                "prompt": "do recon",
                            },
                            {
                                "name": "exploit",
                                "model": "mockllm/model",
                                "prompt": "exploit it",
                            },
                        ],
                    }
                )
            )
            from_yaml = load_team_config(yaml_path)
            from_json = load_team_config(json_path)

        self.assertEqual(from_yaml.turn_taking, TurnTaking.ROUND_ROBIN)
        self.assertEqual(from_yaml.rounds, 2)
        self.assertEqual([m.name for m in from_yaml.members], ["recon", "exploit"])
        self.assertEqual(from_yaml.members[0].role, None)
        self.assertEqual(from_json, from_yaml)

    def test_member_model_defaults_to_first_member_then_coordinator(self) -> None:
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        self.assertEqual(config.turn_taking, TurnTaking.COORDINATOR)
        self.assertEqual(config.member_model(), "mockllm/model")
        with_coord = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "coordinator_model": "openrouter/x/y"}
        )
        self.assertEqual(with_coord.member_model(), "openrouter/x/y")

    def test_rejects_unknown_keys_and_bad_shapes(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):  # unknown key (typo)
            TeamConfig.model_validate({**_TWO_MEMBERS, "turntaking": "coordinator"})
        with self.assertRaisesRegex(ValidationError, "at least two members"):
            TeamConfig.model_validate(
                {"members": [{"name": "solo", "model": "m", "prompt": "p"}]}
            )
        with self.assertRaisesRegex(ValidationError, "names must be unique"):
            TeamConfig.model_validate(
                {
                    "members": [
                        {"name": "dup", "model": "m", "prompt": "p"},
                        {"name": "dup", "model": "m", "prompt": "p"},
                    ]
                }
            )
        with self.assertRaisesRegex(ValidationError, "identifier"):
            TeamMemberConfig(name="2bad", model="m", prompt="p")
        with self.assertRaises(ValidationError):  # empty prompt
            TeamMemberConfig(name="ok", model="m", prompt="  ")

    def test_shipped_example_team_yaml_parses(self) -> None:
        example = Path(
            str(
                resources.files("inspect_rogue_arena").joinpath("examples", "team.yaml")
            )
        )
        config = load_team_config(example)
        self.assertGreaterEqual(len(config.members), 2)
        self.assertEqual(config.turn_taking, TurnTaking.COORDINATOR)

    def test_visibility_defaults_to_shared_sight_and_shows_the_activity_feed(
        self,
    ) -> None:
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        self.assertEqual(config.visibility, TeamVisibility.SHARED_SIGHT)
        self.assertTrue(config.shows_activity_feed())

    def test_communicate_only_hides_the_activity_feed(self) -> None:
        config = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        self.assertEqual(config.visibility, TeamVisibility.COMMUNICATE_ONLY)
        self.assertFalse(config.shows_activity_feed())

    def test_visibility_rejects_an_unknown_value(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            TeamConfig.model_validate({**_TWO_MEMBERS, "visibility": "see_all"})

    def test_session_channel_defaults_to_the_member_name(self) -> None:
        member = TeamMemberConfig(name="recon", model="m", prompt="p")
        self.assertEqual(member.session_channel(), "recon")

    def test_session_channel_uses_an_explicit_channel_when_set(self) -> None:
        member = TeamMemberConfig(name="recon", model="m", prompt="p", channel="alpha")
        self.assertEqual(member.session_channel(), "alpha")


class ChatroomAndScoreboardToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        init_subtask_store(Store())

    async def test_chatroom_is_tagged_and_time_ordered(self) -> None:
        await post_message("recon")(message="ssh open on host A")
        await post_message("exploit")(message="popped host A")
        await post_message("recon")(message="next hop is host B")

        everything = await read_messages()(since=0)
        lines = everything.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(
            lines[0].startswith("#1 ") and "recon: ssh open on host A" in lines[0]
        )
        self.assertTrue(
            lines[1].startswith("#2 ") and "exploit: popped host A" in lines[1]
        )
        self.assertTrue(
            lines[2].startswith("#3 ") and "recon: next hop is host B" in lines[2]
        )
        # a single shared, ordered transcript
        stored = store_as(Chatroom).messages
        self.assertEqual([m.seq for m in stored], [1, 2, 3])
        self.assertEqual([m.author for m in stored], ["recon", "exploit", "recon"])
        # `since` returns only newer messages
        self.assertEqual(
            await read_messages()(since=3), "no new messages in the team chatroom"
        )
        tail = await read_messages()(since=1)
        self.assertNotIn("#1 ", tail)
        self.assertIn("#2 ", tail)

    async def test_chatroom_writes_through_the_hub_on_a_live_range(self) -> None:
        from inspect_rogue_arena.client import EvalChatMessage

        posted: list[tuple[str, str]] = []

        class _FakeRange:
            async def post_team_message(
                self, author: str, content: str
            ) -> EvalChatMessage:
                posted.append((author, content))
                return EvalChatMessage(
                    id="m1",
                    job_id="job-1",
                    seq=len(posted),
                    at="2026-09-30T00:00:00Z",
                    author=author,
                    content=content,
                )

            async def read_team_messages(
                self, since: int = 0
            ) -> tuple[EvalChatMessage, ...]:
                rows = [
                    EvalChatMessage(
                        id=f"m{index}",
                        job_id="job-1",
                        seq=index,
                        at="2026-09-30T00:00:00Z",
                        author=author,
                        content=content,
                    )
                    for index, (author, content) in enumerate(posted, start=1)
                ]
                return tuple(row for row in rows if row.seq > since)

        with patch.object(team_mod, "_hub_chatroom", return_value=_FakeRange()):
            self.assertEqual(
                await post_message("recon")(message="ssh open"),
                "posted message #1 to the team chatroom",
            )
            await post_message("exploit")(message="popped it")
            everything = await read_messages()(since=0)
            tail = await read_messages()(since=1)
        self.assertEqual(posted, [("recon", "ssh open"), ("exploit", "popped it")])
        self.assertIn("#1 [2026-09-30T00:00:00Z] recon: ssh open", everything)
        self.assertIn("#2 [2026-09-30T00:00:00Z] exploit: popped it", everything)
        self.assertNotIn("#1 ", tail)
        self.assertIn("#2 ", tail)
        # the per-sample Store MIRRORS the hub room (so the local .eval log has
        # the room too), carrying the hub's assigned seq/author/content
        mirror = store_as(Chatroom).messages
        self.assertEqual(
            [(m.seq, m.author, m.content) for m in mirror],
            [(1, "recon", "ssh open"), (2, "exploit", "popped it")],
        )

    async def test_activity_feed_offline_store_append_and_read(self) -> None:
        # No sandbox configured -> the feed falls back to the per-sample Store.
        self.assertIsNone(team_mod._hub_activity_feed())
        await team_mod._append_activity(
            "recon", "nmap -sV 203.0.113.5", "exit 0; 3 ports"
        )
        await team_mod._append_activity(
            "exploit", "curl http://203.0.113.5", "exit 0; 200"
        )
        feed = await read_activity()(since=0)
        lines = feed.splitlines()
        self.assertTrue(
            lines[0].startswith("#1 ")
            and "recon: nmap -sV 203.0.113.5 -> exit 0; 3 ports" in lines[0]
        )
        self.assertTrue(
            lines[1].startswith("#2 ")
            and "exploit: curl http://203.0.113.5 -> exit 0; 200" in lines[1]
        )
        self.assertIn("(cursor: 2", lines[-1])
        stored = store_as(ActivityFeed).entries
        self.assertEqual([e.seq for e in stored], [1, 2])
        self.assertEqual([e.member for e in stored], ["recon", "exploit"])
        # `since` returns only newer entries
        self.assertEqual(await read_activity()(since=2), "no new team activity")
        self.assertIn("#2 ", await read_activity()(since=1))

    async def test_read_activity_pages_a_busy_feed_with_a_more_marker(self) -> None:
        self.assertIsNone(team_mod._hub_activity_feed())
        for index in range(team_mod._ACTIVITY_READ_PAGE + 10):
            await team_mod._record_activity("recon", f"cmd {index}", "exit 0")
        page = await read_activity()(since=0)
        lines = page.splitlines()
        # bounded to one page, with a "more" marker that carries the cursor forward
        self.assertEqual(
            len([ln for ln in lines if ln.startswith("#")]),
            team_mod._ACTIVITY_READ_PAGE,
        )
        self.assertIn(
            f"more activity may remain; call read_activity with since={team_mod._ACTIVITY_READ_PAGE}",
            page,
        )

    async def test_member_bash_execs_and_auto_appends_through_the_hub(self) -> None:
        from inspect_rogue_arena.client import EvalActivityEntry

        posted: list[tuple[str, str, str]] = []

        class _FakeFeed:
            async def post_activity_entry(
                self, member, command, summary, *, client_entry_key=None
            ):
                posted.append((member, command, summary))
                return EvalActivityEntry(
                    id="a1",
                    job_id="job-1",
                    seq=len(posted),
                    at="2026-09-30T01:00:00Z",
                    member=member,
                    command=command,
                    summary=summary,
                )

            async def read_activity_entries(self, since=0, *, limit=None):
                rows = [
                    EvalActivityEntry(
                        id=f"a{i}",
                        job_id="job-1",
                        seq=i,
                        at="2026-09-30T01:00:00Z",
                        member=m,
                        command=c,
                        summary=s,
                    )
                    for i, (m, c, s) in enumerate(posted, start=1)
                    if i > since
                ]
                return tuple(rows[:limit] if limit is not None else rows)

        class _FakeSandbox:
            async def exec(self, cmd, timeout=None, user=None):
                self.cmd = cmd
                return ExecResult(
                    success=True, returncode=0, stdout="root\n", stderr=""
                )

        sandbox = _FakeSandbox()
        resolved: list[str | None] = []

        def fake_active_sandbox(channel=None):
            resolved.append(channel)
            return sandbox

        config = TeamConfig.model_validate(_TWO_MEMBERS)  # shared_sight default
        tool = member_bash(config.members[0], config)  # recon
        with (
            patch("inspect_ai.util.sandbox", fake_active_sandbox),
            patch.object(team_mod, "_hub_activity_feed", return_value=_FakeFeed()),
        ):
            output = await tool(command="id")
        self.assertEqual(output, "root\n")  # the member still sees the real output
        self.assertEqual(sandbox.cmd, ["bash", "--login", "-c", "id"])
        # bash runs on the member's own session channel (defaults to the name)
        self.assertEqual(resolved, ["recon"])
        # one summarized entry was auto-recorded (summary, never full output)
        self.assertEqual(posted, [("recon", "id", "exit 0; root")])
        with patch.object(team_mod, "_hub_activity_feed", return_value=_FakeFeed()):
            # a different member reads the first member's command back
            feed = await read_activity()(since=0)
        self.assertIn("recon: id -> exit 0; root", feed)

    async def test_communicate_only_member_bash_still_records_but_no_read_tool(
        self,
    ) -> None:
        # communicate_only RECORDS the feed (the owner's run page
        # shows it); members just never read it during the run.
        class _FakeSandbox:
            async def exec(self, cmd, timeout=None, user=None):
                return ExecResult(success=True, returncode=0, stdout="data", stderr="")

        appended: list[Any] = []
        config = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        tool = member_bash(config.members[0], config)  # records defaults True
        with (
            patch("inspect_ai.util.sandbox", return_value=_FakeSandbox()),
            patch.object(
                team_mod,
                "_append_activity",
                side_effect=lambda *a, **k: appended.append(a),
            ),
        ):
            output = await tool(command="ls")
        self.assertEqual(output, "data")
        self.assertEqual(
            appended, [("recon", "ls", "exit 0; data")]
        )  # recorded in communicate_only
        # but members get no read_activity tool in communicate_only
        names = {
            ToolDef(t).name for t in team_mod._member_tools(config.members[0], config)
        }
        self.assertNotIn("read_activity", names)

    async def test_member_bash_does_not_record_when_records_false(self) -> None:
        # The team solver passes records=False only when the hub has no feed route.
        class _FakeSandbox:
            async def exec(self, cmd, timeout=None, user=None):
                return ExecResult(success=True, returncode=0, stdout="data", stderr="")

        appended: list[Any] = []
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        tool = member_bash(config.members[0], config, records=False)
        with (
            patch("inspect_ai.util.sandbox", return_value=_FakeSandbox()),
            patch.object(
                team_mod,
                "_append_activity",
                side_effect=lambda *a, **k: appended.append(a),
            ),
        ):
            await tool(command="ls")
        self.assertEqual(appended, [])

    def test_summarize_activity_is_bounded_and_deterministic(self) -> None:
        first = summarize_activity(0, "a.txt\nb.txt", "")
        self.assertEqual(first, "exit 0; a.txt b.txt")
        self.assertEqual(first, summarize_activity(0, "a.txt\nb.txt", ""))
        # a non-zero exit and stderr are condensed into the one line
        self.assertEqual(summarize_activity(1, "", "boom"), "exit 1; boom")
        # the summary is hard-capped, so huge output never blows the feed
        huge = summarize_activity(0, "x" * 100_000, "")
        self.assertLessEqual(len(huge), team_mod._ACTIVITY_SUMMARY_CAP)
        self.assertTrue(huge.endswith("..."))

    async def test_scoreboard_records_findings(self) -> None:
        self.assertEqual(await read_scoreboard()(), "the team scoreboard is empty")
        await post_finding("exploit")(label="hop 1 key", detail="proof key captured")
        await post_finding("recon")(label="hop 2 reachable", detail="")
        board = await read_scoreboard()()
        self.assertIn("exploit: hop 1 key: proof key captured", board)
        self.assertIn("recon: hop 2 reachable", board)
        self.assertEqual(
            [f.author for f in store_as(Scoreboard).findings], ["exploit", "recon"]
        )


class ToolSchemaProviderSafetyTests(unittest.TestCase):
    """The team tools' schemas must be valid for strict-schema providers.

    OpenAI's strict function schema requires every property to appear in
    ``required``; an optional (defaulted) parameter is rejected with
    ``invalid_function_parameters``. So the team tools carry no optional
    parameters: ``read_messages.since`` and ``post_finding.detail`` are required.
    """

    def test_read_messages_since_is_required(self) -> None:
        params = ToolDef(read_messages()).parameters
        self.assertIn("since", params.properties)
        self.assertIn("since", params.required)

    def test_post_finding_detail_is_required(self) -> None:
        params = ToolDef(post_finding("recon")).parameters
        self.assertEqual(set(params.properties), {"label", "detail"})
        self.assertEqual(set(params.required), {"label", "detail"})

    def test_post_message_param_is_required(self) -> None:
        params = ToolDef(post_message("recon")).parameters
        self.assertEqual(set(params.required), set(params.properties))


class HubChatroomResolutionTests(unittest.IsolatedAsyncioTestCase):
    """The team chatroom resolves the live range through Inspect's sandbox proxy.

    ``sandbox()`` returns a ``SandboxEnvironmentProxy`` wrapping the real
    environment. These tests pin the behavior: resolve via ``as_type``, mirror
    to the Store, and never swallow a configured-but-unresolvable sandbox.
    """

    async def asyncSetUp(self) -> None:
        init_subtask_store(Store())

    def _real_range(self) -> RogueArenaSandboxEnvironment:
        hub = FakeHub(self)
        self.hub = hub
        hub.status["trial-1"] = "ready"
        trial = EvalTrial.from_dict(hub.trial("trial-1"))
        runtime = _TaskRuntime(config=_config(), client=hub.client())
        return RogueArenaSandboxEnvironment(
            hub.client(), trial, runtime=runtime, deadline=1e18
        )

    def _enter(self, environments: dict[str, Any]) -> None:
        token = sandbox_environments_context_var.set(environments)
        self.addCleanup(_reset_context_var, sandbox_environments_context_var, token)
        if environments:
            name = next(iter(environments))
            default_token = sandbox_default_context_var.set(name)
            self.addCleanup(
                _reset_context_var, sandbox_default_context_var, default_token
            )

    async def test_resolves_the_range_through_inspects_proxy(self) -> None:
        environment = self._real_range()
        proxy = SandboxEnvironmentProxy(environment)
        self._enter({"default": proxy})
        # the proxy is NOT the environment, but the chatroom must still find it
        self.assertNotIsInstance(proxy, RogueArenaSandboxEnvironment)
        self.assertIs(team_mod._hub_chatroom(), environment)

    async def test_post_routes_to_the_hub_and_mirrors_the_store(self) -> None:
        environment = self._real_range()
        self._enter({"default": SandboxEnvironmentProxy(environment)})

        self.assertEqual(
            await post_message("recon")(message="ssh open on host A"),
            "posted message #1 to the team chatroom",
        )
        await post_message("exploit")(message="popped host A")

        # the hub's per-run room is the source of truth
        room = self.hub.chat["job-1"]
        self.assertEqual(
            [(m["seq"], m["author"]) for m in room], [(1, "recon"), (2, "exploit")]
        )
        # read_messages reads back from the hub
        everything = await read_messages()(since=0)
        self.assertIn("#1 ", everything)
        self.assertIn("recon: ssh open on host A", everything)
        self.assertIn("exploit: popped host A", everything)
        # and the per-sample Store mirrors it, so the local .eval log has the room
        mirror = store_as(Chatroom).messages
        self.assertEqual(
            [(m.seq, m.author, m.content) for m in mirror],
            [(1, "recon", "ssh open on host A"), (2, "exploit", "popped host A")],
        )

    async def test_no_sandbox_configured_falls_back_to_the_store(self) -> None:
        self._enter({})  # no sandbox: sandbox() raises -> Store
        self.assertIsNone(team_mod._hub_chatroom())
        await post_message("recon")(message="offline note")
        self.assertEqual([m.author for m in store_as(Chatroom).messages], ["recon"])

    async def test_a_non_rogue_arena_sandbox_fails_loudly(self) -> None:
        other = MagicMock(spec=SandboxEnvironment)
        self._enter({"default": SandboxEnvironmentProxy(other)})
        with self.assertRaises(TypeError):
            team_mod._hub_chatroom()
        # the misrouted post is NOT silently dropped into the Store
        with self.assertRaises(TypeError):
            await post_message("recon")(message="should not reach the store")
        self.assertEqual(store_as(Chatroom).messages, [])

    async def test_activity_probe_true_when_the_hub_serves_the_route(self) -> None:
        environment = self._real_range()  # FakeHub, activity_supported defaults True
        self._enter({"default": SandboxEnvironmentProxy(environment)})
        self.assertTrue(await team_mod._activity_feed_supported())

    async def test_activity_probe_false_when_the_hub_lacks_the_route(self) -> None:
        environment = self._real_range()
        self.hub.activity_supported = False  # a hub without the feed -> 404
        self._enter({"default": SandboxEnvironmentProxy(environment)})
        self.assertFalse(await team_mod._activity_feed_supported())

    async def test_activity_probe_offline_is_supported_via_the_store(self) -> None:
        self._enter({})  # no sandbox -> the Store fallback always works
        self.assertTrue(await team_mod._activity_feed_supported())

    async def test_activity_probe_reraises_a_non_404_error(self) -> None:
        from inspect_rogue_arena.client import RogueArenaApiError

        class _Boom:
            async def read_activity_entries(self, since=0, *, limit=None):
                raise RogueArenaApiError(500, "GET", "/activity", {"errorMessage": "x"})

        with (
            patch.object(team_mod, "_hub_activity_feed", return_value=_Boom()),
            self.assertRaises(RogueArenaApiError),
        ):
            await team_mod._activity_feed_supported()

    async def test_team_visibility_must_match_the_hub_job(self) -> None:
        environment = self._real_range()  # job created as shared_sight (config default)
        self._enter({"default": SandboxEnvironmentProxy(environment)})
        team_mod._require_job_visibility(TeamConfig.model_validate(_TWO_MEMBERS))
        quiet = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        # a communicate_only team on a shared_sight job: the hub would not hide the feed
        with self.assertRaisesRegex(RuntimeError, "communicate_only.*shared_sight"):
            team_mod._require_job_visibility(quiet)

    async def test_team_visibility_matches_a_communicate_only_job(self) -> None:
        environment = self._real_range()
        environment._runtime.config = _config(team_visibility="communicate_only")
        self._enter({"default": SandboxEnvironmentProxy(environment)})
        quiet = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        team_mod._require_job_visibility(quiet)
        with self.assertRaisesRegex(RuntimeError, "shared_sight.*communicate_only"):
            team_mod._require_job_visibility(TeamConfig.model_validate(_TWO_MEMBERS))

    async def test_team_visibility_offline_has_no_job_to_match(self) -> None:
        self._enter({})
        quiet = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        team_mod._require_job_visibility(quiet)


_EXEC_ID = "77777777-7777-4777-8777-777777777777"


def _exec_answer(
    *,
    running: bool,
    exec_id: str | None,
    stdout: str = "",
    exit_code: int | None = None,
) -> tuple[int, dict[str, Any]]:
    return (
        200,
        {
            "running": running,
            "execId": exec_id,
            "exitCode": exit_code,
            "stdout": stdout,
            "stderr": "",
            "timedOut": False,
        },
    )


class MemberBashTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """A member's bash states its default timeout in the tool description and
    takes a longer one on request; a long command runs to completion."""

    async def asyncSetUp(self) -> None:
        init_subtask_store(Store())

    def _real_range(self) -> RogueArenaSandboxEnvironment:
        hub = FakeHub(self)
        self.hub = hub
        hub.status["trial-1"] = "ready"
        trial = EvalTrial.from_dict(hub.trial("trial-1"))
        runtime = _TaskRuntime(config=_config(), client=hub.client())
        return RogueArenaSandboxEnvironment(
            hub.client(), trial, runtime=runtime, deadline=1e18
        )

    def _enter(self, environments: dict[str, Any]) -> None:
        token = sandbox_environments_context_var.set(environments)
        self.addCleanup(_reset_context_var, sandbox_environments_context_var, token)
        default_token = sandbox_default_context_var.set(next(iter(environments)))
        self.addCleanup(_reset_context_var, sandbox_default_context_var, default_token)

    async def test_a_long_command_runs_to_completion_and_returns_its_output(
        self,
    ) -> None:
        environment = self._real_range()
        self._enter({"recon": SandboxEnvironmentProxy(environment)})
        self.hub.exec_responses = [
            _exec_answer(running=True, exec_id=_EXEC_ID),  # the background launch
            _exec_answer(
                running=True, exec_id=_EXEC_ID
            ),  # still scanning at the first poll
            _exec_answer(
                running=False, exec_id=_EXEC_ID, stdout="scan done\n", exit_code=0
            ),
        ]
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        tool = member_bash(config.members[0], config, records=False)

        output = await tool(command="sleep 150; echo scan done", timeout=300)

        self.assertEqual(output, "scan done\n")
        launch, *polls = self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertEqual(
            launch["cmd"], ["bash", "--login", "-c", "sleep 150; echo scan done"]
        )
        self.assertTrue(launch["background"])
        self.assertEqual(launch["timeoutSeconds"], 300)
        self.assertEqual([poll["pollExecId"] for poll in polls], [_EXEC_ID, _EXEC_ID])

    async def test_a_command_without_a_timeout_completes_under_the_team_default(
        self,
    ) -> None:
        environment = self._real_range()
        self._enter({"recon": SandboxEnvironmentProxy(environment)})
        self.hub.exec_responses = [
            _exec_answer(running=True, exec_id=_EXEC_ID),  # the background launch
            _exec_answer(running=True, exec_id=_EXEC_ID),  # still building at 1st poll
            _exec_answer(
                running=False, exec_id=_EXEC_ID, stdout="build ok\n", exit_code=0
            ),
        ]
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        self.assertEqual(config.tool_timeout_seconds, 600)
        tool = member_bash(config.members[0], config, records=False)

        output = await tool(command="make all")

        self.assertEqual(output, "build ok\n")
        launch, *polls = self.hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        self.assertTrue(launch["background"])
        self.assertEqual(launch["timeoutSeconds"], 600)
        self.assertEqual(len(polls), 2)

    async def test_a_timed_out_command_is_recorded_with_the_timeout_it_ran_under(
        self,
    ) -> None:
        class _TimingOutSandbox:
            async def exec(self, cmd, timeout=None, user=None):
                raise TimeoutError(f"Command timed out after {timeout} seconds")

        appended: list[Any] = []
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        tool = member_bash(config.members[0], config)
        with (
            patch("inspect_ai.util.sandbox", return_value=_TimingOutSandbox()),
            patch.object(
                team_mod,
                "_append_activity",
                side_effect=lambda *a, **k: appended.append(a),
            ),
            self.assertRaisesRegex(TimeoutError, "after 600 seconds"),
        ):
            await tool(command="make all", timeout=600)
        self.assertEqual(appended, [("recon", "make all", "timed out after 600s")])

    async def test_a_non_positive_timeout_is_a_tool_error(self) -> None:
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        tool = member_bash(config.members[0], config, records=False)
        with self.assertRaisesRegex(ToolError, "positive number of seconds"):
            await tool(command="id", timeout=0)

    def test_the_tool_states_its_timeout_to_the_model(self) -> None:
        config = TeamConfig.model_validate({**_TWO_MEMBERS, "tool_timeout_seconds": 30})
        definition = ToolDef(member_bash(config.members[0], config))
        self.assertEqual(definition.name, "bash")
        self.assertIn(
            "runs to completion and returns its full output", definition.description
        )
        self.assertIn("(default 30)", definition.description)
        self.assertIn("Pass a larger `timeout`", definition.description)
        self.assertEqual(set(definition.parameters.properties), {"command", "timeout"})
        self.assertEqual(definition.parameters.required, ["command"])


class TeamStructureTests(unittest.TestCase):
    def test_coordinator_exposes_one_handoff_per_member(self) -> None:
        config = TeamConfig.model_validate(_TWO_MEMBERS)
        members = [
            team_mod._member_agent(member, config, submit=True)
            for member in config.members
        ]
        names = {ToolDef(handoff(member)).name for member in members}
        self.assertEqual(names, {"transfer_to_recon", "transfer_to_exploit"})

    def test_build_team_dispatches_on_turn_taking(self) -> None:
        coordinator = build_team(TeamConfig.model_validate(_TWO_MEMBERS))
        round_robin = build_team(
            TeamConfig.model_validate({**_TWO_MEMBERS, "turn_taking": "round_robin"})
        )
        self.assertTrue(callable(coordinator))
        self.assertTrue(callable(round_robin))

    def test_member_tools_expose_bash_and_gate_read_activity_on_shared_sight(
        self,
    ) -> None:
        shared = TeamConfig.model_validate(_TWO_MEMBERS)
        names = {
            ToolDef(tool).name
            for tool in team_mod._member_tools(shared.members[0], shared)
        }
        # the member's shell is presented to the model as the ordinary "bash" tool
        self.assertIn("bash", names)
        # shared_sight (the default) attaches read_activity
        self.assertIn("read_activity", names)

        quiet = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "visibility": "communicate_only"}
        )
        quiet_names = {
            ToolDef(tool).name
            for tool in team_mod._member_tools(quiet.members[0], quiet)
        }
        # communicate_only withholds read_activity (the feed is still recorded)
        self.assertNotIn("read_activity", quiet_names)
        self.assertIn("bash", quiet_names)


def _member_outputs(
    messages: list[Any], tools: list[Any], tool_choice: Any, config: Any
) -> ModelOutput:
    """Each member posts to the chatroom once, then runs one bash command."""
    names = {tool.name for tool in tools}
    turns = sum(1 for message in messages if message.role == "assistant")
    if "post_message" in names and turns == 0:
        return ModelOutput.for_tool_call(
            "mockllm/model", "post_message", {"message": "checking in"}
        )
    if "bash" in names:
        return ModelOutput.for_tool_call("mockllm/model", "bash", {"command": "id"})
    return ModelOutput.from_content("mockllm/model", "done")


def _coordinator_outputs(
    messages: list[Any], tools: list[Any], tool_choice: Any, config: Any
) -> ModelOutput:
    """Drive a coordinator run: the lead hands off to each member once; each
    member posts to the chatroom once, then submits back to the lead."""
    names = {tool.name for tool in tools}
    transfer_names = sorted(name for name in names if name.startswith("transfer_to_"))
    if transfer_names:  # the coordinator (it has the handoff tools)
        already = {
            call.function
            for message in messages
            if message.role == "assistant" and message.tool_calls
            for call in message.tool_calls
            if call.function.startswith("transfer_to_")
        }
        pending = [name for name in transfer_names if name not in already]
        if pending:
            return ModelOutput.for_tool_call("mockllm/model", pending[0], {})
        return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})
    # a member: post once (first action), then submit to return control
    last = messages[-1] if messages else None
    if last is not None and last.role == "tool" and "posted message" in last.text:
        return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})
    if "post_message" in names:
        return ModelOutput.for_tool_call(
            "mockllm/model", "post_message", {"message": "checking in"}
        )
    return ModelOutput.for_tool_call("mockllm/model", "submit", {"answer": "done"})


class TeamEvalTests(unittest.TestCase):
    """A real (offline) team run: two members, one shared range, a chatroom."""

    def test_round_robin_two_members_share_a_range_and_a_chatroom(self) -> None:
        hub = FakeHub(self)
        mock = get_model("mockllm/model", custom_outputs=_member_outputs)

        def mock_member(
            member: TeamMemberConfig,
            config: TeamConfig,
            *,
            submit: bool,
            records=None,
            read_tool=None,
        ):
            return react(
                name=member.name,
                description=member.role or member.name,
                prompt=member.prompt + team_mod._MEMBER_ADDENDUM,
                tools=team_mod._member_tools(
                    member, config, records=records, read_tool=read_tool
                ),
                model=mock,
                submit=submit,
            )

        config = TeamConfig.model_validate(
            {
                **_TWO_MEMBERS,
                "turn_taking": "round_robin",
                "rounds": 1,
                "turns_per_member": 2,
            }
        )
        task = Task(
            dataset=[Sample(input="team objective", id="s-1")],
            solver=team_mod.team_solver(config),
            scorer=rogue_arena_hops(),
            sandbox=SandboxEnvironmentSpec(
                "rogue-arena", _config(max_trials=1, concurrency=1)
            ),
            name="team_task",
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            patch.object(team_mod, "_member_agent", mock_member),
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            log = inspect_eval(
                task, model="mockllm/model", log_dir=log_dir, display="none"
            )[0]
            self.assertEqual(log.status, "success", log.error)
            sample = (log.samples or [None])[0]

        assert sample is not None
        # one shared range (one trial), scored by the hub on close
        self.assertEqual(len(hub.calls("POST", "/hub/eval/trials")), 1)
        self.assertEqual(
            hub.calls("POST", "/hub/eval/trials/trial-1/close"), [{"outcome": "closed"}]
        )
        self.assertEqual(sample.scores["rogue_arena_hops"].value, 0.5)
        # two independent tool streams: each member ran `id` once on the shared
        # range, as a background exec (the member default) polled once to its end
        execs = hub.calls("POST", "/hub/eval/trials/trial-1/exec")
        launches = [call for call in execs if not call.get("pollExecId")]
        self.assertEqual(len(launches), 2)
        self.assertTrue(
            all(
                call["cmd"] == ["bash", "--login", "-c", "id"] and call["background"]
                for call in launches
            )
        )
        self.assertEqual(len(execs) - len(launches), 2)
        # each member posted to the hub's per-run room (the source of truth,
        # what the hub /chat API and the web app's panel show live)
        chat_posts = hub.calls("POST", "/hub/eval/jobs/job-1/chat")
        self.assertEqual([p["author"] for p in chat_posts], ["recon", "exploit"])
        room = hub.chat["job-1"]
        self.assertEqual(
            [(m["seq"], m["author"]) for m in room], [(1, "recon"), (2, "exploit")]
        )
        # and the per-sample Store MIRRORS the hub room, so the local .eval log
        # carries one shared, time-ordered transcript tagged per member
        transcript = sample.store["Chatroom:messages"]
        self.assertEqual([m["seq"] for m in transcript], [1, 2])
        self.assertEqual([m["author"] for m in transcript], ["recon", "exploit"])
        # the orchestrator summarized the run into the main transcript
        self.assertTrue(
            any(
                m.role == "assistant" and "Round-robin team run complete" in m.text
                for m in sample.messages
            )
        )
        self.assertEqual(hub.jobs["job-1"]["status"], "cancelled")
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    def test_coordinator_members_post_through_the_hub_room(self) -> None:
        """Coordinator mode: members run via handoff and still reach the hub room.

        The members run inside ``handoff()`` sub-agents, not the top-level
        sample; the ``as_type`` resolution means their chat still routes
        to the hub's per-run room (and mirrors the Store) from there.
        """
        hub = FakeHub(self)
        mock = get_model("mockllm/model", custom_outputs=_coordinator_outputs)

        def mock_member(
            member: TeamMemberConfig,
            config: TeamConfig,
            *,
            submit: bool,
            records=None,
            read_tool=None,
        ):
            return react(
                name=member.name,
                description=member.role or member.name,
                prompt=member.prompt + team_mod._MEMBER_ADDENDUM,
                tools=team_mod._member_tools(
                    member, config, records=records, read_tool=read_tool
                ),
                model=mock,
                submit=submit,
            )

        config = TeamConfig.model_validate(
            {**_TWO_MEMBERS, "turn_taking": "coordinator"}
        )
        # the coordinator react agent is built inside team_mod via team_mod.react;
        # force it onto the scripted mock (members are patched via _member_agent).
        real_react = team_mod.react

        def coord_react(*args: Any, **kwargs: Any):
            kwargs["model"] = mock
            return real_react(*args, **kwargs)

        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            patch.object(team_mod, "_member_agent", mock_member),
            patch.object(team_mod, "react", coord_react),
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            # the coordinator builds its members eagerly, so construct the task
            # (and solver) under the patches, not before them.
            task = Task(
                dataset=[Sample(input="team objective", id="s-1")],
                solver=team_mod.team_solver(config),
                scorer=rogue_arena_hops(),
                sandbox=SandboxEnvironmentSpec(
                    "rogue-arena", _config(max_trials=1, concurrency=1)
                ),
                name="team_coord_task",
            )
            log = inspect_eval(task, model=mock, log_dir=log_dir, display="none")[0]
            self.assertEqual(log.status, "success", log.error)
            sample = (log.samples or [None])[0]

        assert sample is not None
        # one shared range for the whole team
        self.assertEqual(len(hub.calls("POST", "/hub/eval/trials")), 1)
        # both members posted to the hub room from inside their handoff sub-agents
        room = hub.chat["job-1"]
        self.assertEqual(sorted(m["author"] for m in room), ["exploit", "recon"])
        self.assertEqual([m["seq"] for m in room], list(range(1, len(room) + 1)))
        # the Store mirrors the hub room for the local .eval log
        transcript = sample.store["Chatroom:messages"]
        self.assertEqual(
            [(m["seq"], m["author"]) for m in transcript],
            [(m["seq"], m["author"]) for m in room],
        )
        self.assertEqual(RogueArenaSandboxEnvironment._runtimes, {})

    def _run_round_robin(self, hub: FakeHub):
        """Run a two-member round-robin team against ``hub``; return the sample.

        The mock member honours ``records``/``read_tool`` so the team's runtime probe is
        reflected in the member tools, exactly as the real ``_member_agent`` does.
        """
        mock = get_model("mockllm/model", custom_outputs=_member_outputs)

        def mock_member(member, config, *, submit, records=None, read_tool=None):
            return react(
                name=member.name,
                description=member.role or member.name,
                prompt=member.prompt + team_mod._MEMBER_ADDENDUM,
                tools=team_mod._member_tools(
                    member, config, records=records, read_tool=read_tool
                ),
                model=mock,
                submit=submit,
            )

        config = TeamConfig.model_validate(
            {
                **_TWO_MEMBERS,
                "turn_taking": "round_robin",
                "rounds": 1,
                "turns_per_member": 2,
            }
        )
        task = Task(
            dataset=[Sample(input="team objective", id="s-1")],
            solver=team_mod.team_solver(config),
            scorer=rogue_arena_hops(),
            sandbox=SandboxEnvironmentSpec(
                "rogue-arena", _config(max_trials=1, concurrency=1)
            ),
            name="team_feed_task",
        )
        RogueArenaSandboxEnvironment._runtimes.clear()
        with (
            patch.object(team_mod, "_member_agent", mock_member),
            tempfile.TemporaryDirectory() as log_dir,
            patch.dict(os.environ, _without_rogue_arena_env(), clear=True),
            patch.object(RogueArenaClient, "from_env", return_value=hub.client()),
        ):
            log = inspect_eval(
                task, model="mockllm/model", log_dir=log_dir, display="none"
            )[0]
            self.assertEqual(log.status, "success", log.error)
            sample = (log.samples or [None])[0]
        assert sample is not None
        return sample

    def test_round_robin_records_activity_when_the_hub_supports_the_feed(self) -> None:
        hub = FakeHub(self)  # activity_supported defaults True
        sample = self._run_round_robin(hub)
        # each member's one bash command auto-recorded one summarized entry
        feed = hub.activity["job-1"]
        self.assertEqual([e["member"] for e in feed], ["recon", "exploit"])
        self.assertTrue(all(e["command"] == "id" for e in feed))
        self.assertTrue(all(e["summary"].startswith("exit ") for e in feed))
        # no chat-only notice when the feed works
        self.assertFalse(
            any("chat-only" in getattr(m, "text", "") for m in sample.messages)
        )

    def test_round_robin_runs_chat_only_when_the_hub_lacks_the_feed(self) -> None:
        hub = FakeHub(self)
        hub.activity_supported = False  # a hub without the feed
        sample = self._run_round_robin(hub)
        # the run still succeeds and records NO activity, instead of every bash
        # call hitting a 404 and killing the sample
        self.assertNotIn("job-1", hub.activity)
        # exactly one activity probe was made, and it was answered 404
        self.assertEqual(len(hub.calls("GET", "/hub/eval/jobs/job-1/activity")), 1)
        # one clear line says the hub is chat-only
        self.assertTrue(
            any("chat-only" in getattr(m, "text", "") for m in sample.messages)
        )
        # chat still flows on the same run
        self.assertEqual(
            [p["author"] for p in hub.calls("POST", "/hub/eval/jobs/job-1/chat")],
            ["recon", "exploit"],
        )


if __name__ == "__main__":
    unittest.main()
