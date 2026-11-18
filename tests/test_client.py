from __future__ import annotations

import asyncio
import json
import os
import unittest
import uuid
from typing import Any
from unittest.mock import patch

import httpx

from inspect_rogue_arena.client import (
    EvalActivityEntry,
    EvalChatMessage,
    EvalClaimDenied,
    EvalGuideNotOffered,
    EvalJob,
    EvalJobStatus,
    EvalPromptLevel,
    EvalProofType,
    EvalReadiness,
    EvalScoringMode,
    EvalTrial,
    EvalTrialEventKind,
    EvalTrialOutcome,
    EvalTrialStatus,
    EvalTrialTerminalReason,
    ExecOutcomeUnknown,
    HttpResult,
    RogueArenaApiError,
    RogueArenaClient,
    TrialClosedBeforeReady,
    TrialReadyTimeout,
    sanitize_transcript_file_name,
    transcript_sha256,
    transcript_upload_key,
)
from inspect_rogue_arena.config import RogueArenaSandboxConfig


def _client(handler) -> RogueArenaClient:
    return RogueArenaClient(
        "https://hub.example.test",
        "ra_eval_TOKEN",
        transport=httpx.MockTransport(handler),
    )


class RogueArenaClientSyncTests(unittest.TestCase):
    def test_a_base_url_must_be_https_so_the_token_never_crosses_in_clear(
        self,
    ) -> None:
        for base_url in (
            "http://hub.example.test",
            "http://203.0.113.40:8080",
            "hub.example.test",
            "ftp://hub.example.test",
            "https://",
            "http://user:secret@hub.example.test",
        ):
            with self.subTest(base_url=base_url):
                with self.assertRaisesRegex(ValueError, "https://") as caught:
                    RogueArenaClient(base_url, "ra_eval_TOKEN")
                # never echoes credentials from the URL
                self.assertNotIn("secret", str(caught.exception))

    def test_plain_http_is_allowed_only_to_this_machine(self) -> None:
        for base_url in (
            "https://hub.example.test",
            "https://hub.example.test/",
            "HTTPS://Hub.Example.Test",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://[::1]:3000",
        ):
            with self.subTest(base_url=base_url):
                client = RogueArenaClient(base_url, "ra_eval_TOKEN")
                self.assertFalse(client.base_url.endswith("/"))

    def test_from_env_refuses_a_plain_http_base_url(self) -> None:
        env = {
            "ROGUE_ARENA_BASE_URL": "http://hub.example.test",
            "ROGUE_ARENA_TOKEN": "ra_eval_TOKEN",
        }
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ValueError):
            RogueArenaClient.from_env()

    def test_from_env_requires_base_and_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            RogueArenaClient.from_env()

    def test_bearer_header_is_the_personal_token(self) -> None:
        self.assertEqual(
            _client(lambda r: httpx.Response(200, json={}))._headers()["Authorization"],
            "Bearer ra_eval_TOKEN",
        )


class RogueArenaClientAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_whoami_and_evals_are_typed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["Authorization"], "Bearer ra_eval_TOKEN")
            if request.url.path == "/hub/eval/whoami":
                return httpx.Response(
                    200,
                    json={
                        "userMainId": "user-1",
                        "organizationId": "org-1",
                        "tokenLabel": "ci",
                        "tokenLast4": "cd34",
                    },
                )
            if request.url.path == "/hub/eval/evals":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "environmentId": "11111111-1111-4111-8111-111111111111",
                            "name": "Example AD Chain",
                            "environmentBuildId": "22222222-2222-4222-8222-222222222222",
                            "canvasVersionNumber": 3,
                            "hopCount": 2,
                            "agentMachineNickname": "agent-vm",
                            "scoringMode": "sequential",
                        }
                    ],
                )
            return httpx.Response(500, json={"errorMessage": "x", "errorSeverity": 2})

        client = _client(handler)
        who = await client.whoami()
        self.assertEqual(
            (who.user_main_id, who.organization_id, who.token_last4),
            ("user-1", "org-1", "cd34"),
        )
        evals = await client.list_evals()
        self.assertEqual(evals[0].name, "Example AD Chain")
        self.assertEqual(evals[0].hop_count, 2)
        self.assertIs(evals[0].scoring_mode, EvalScoringMode.SEQUENTIAL)

    async def test_evals_parse_optional_token_scoped_readiness(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[
                    {
                        "environmentId": "11111111-1111-4111-8111-111111111111",
                        "name": "warm-eval",
                        "environmentBuildId": "b1",
                        "canvasVersionNumber": 2,
                        "hopCount": 3,
                        "agentMachineNickname": "agent",
                        "scoringMode": "sequential",
                        "readiness": {"warmCount": 2, "onDemandEtaSeconds": None},
                    },
                    {
                        "environmentId": "22222222-2222-4222-8222-222222222222",
                        "name": "cold-eval",
                        "environmentBuildId": "b2",
                        "canvasVersionNumber": 1,
                        "hopCount": 1,
                        "agentMachineNickname": None,
                        "scoringMode": "independent",
                        "readiness": {"warmCount": 0, "onDemandEtaSeconds": 300},
                    },
                    {
                        "environmentId": "33333333-3333-4333-8333-333333333333",
                        "name": "no-readiness-eval",
                        "environmentBuildId": "b3",
                        "canvasVersionNumber": 1,
                        "hopCount": 1,
                        "agentMachineNickname": None,
                        "scoringMode": "sequential",
                    },
                ],
            )

        evals = await _client(handler).list_evals()
        warm, cold, no_readiness = evals
        assert warm.readiness is not None
        self.assertEqual(warm.readiness.warm_count, 2)
        self.assertEqual(warm.readiness_label(), "ready now (2 warm)")
        self.assertEqual(cold.readiness_label(), "on demand, ~5 min")
        self.assertIsNone(no_readiness.readiness)
        self.assertEqual(no_readiness.readiness_label(), "unknown")
        # label logic: warm beats eta; eta-less cold is plain "on demand"
        self.assertEqual(
            EvalReadiness(warm_count=0, on_demand_eta_seconds=None).label(), "on demand"
        )

    async def test_get_eval_guide_is_typed(self) -> None:
        env_id = "11111111-1111-4111-8111-111111111111"
        guide = {
            "environmentId": env_id,
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
            "recommendedPrompts": {"easy": "do X", "medium": "", "hard": "win"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["Authorization"], "Bearer ra_eval_TOKEN")
            self.assertEqual(
                (request.method, request.url.path),
                ("GET", f"/hub/eval/evals/{env_id}/guide"),
            )
            return httpx.Response(200, json=guide)

        result = await _client(handler).get_eval_guide(env_id)
        self.assertEqual(result.name, "Example AD Chain")
        self.assertIs(result.scoring_mode, EvalScoringMode.SEQUENTIAL)
        self.assertEqual([hop.step_number for hop in result.hops], [1, 2])
        self.assertIs(result.hops[0].proof_type, EvalProofType.KEY_FILE)
        self.assertEqual(result.hops[0].key_filepath, "/root/flag1.txt")
        self.assertIs(result.hops[1].proof_type, EvalProofType.COMMAND_CHECK)
        self.assertEqual(result.hops[1].regex, "admin")
        self.assertEqual(result.solution_writeup, "web -> app VLAN -> dc")
        self.assertEqual(result.prompt(EvalPromptLevel.EASY), "do X")
        self.assertEqual(result.prompt(EvalPromptLevel.HARD), "win")
        # a prompt the author left blank is not silently returned
        with self.assertRaisesRegex(RuntimeError, "no medium recommended prompt"):
            result.prompt(EvalPromptLevel.MEDIUM)

    async def test_get_eval_guide_escapes_the_id_in_the_path(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            # raw_path keeps the percent-encoding; .path would decode it
            seen.append(request.url.raw_path.decode())
            return httpx.Response(404, json={"errorMessage": "no", "errorSeverity": 2})

        with self.assertRaises(RogueArenaApiError) as ctx:
            await _client(handler).get_eval_guide("a b/c")
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(seen, ["/hub/eval/evals/a%20b%2Fc/guide"])

    async def test_a_refused_guide_with_a_working_token_is_guides_not_offered(
        self,
    ) -> None:
        def refusing_the_guide(status: int, seen: list[str]):
            def handler(request: httpx.Request) -> httpx.Response:
                seen.append(request.url.path)
                if request.url.path == "/hub/eval/whoami":
                    return httpx.Response(
                        200,
                        json={
                            "userMainId": "u",
                            "organizationId": None,
                            "tokenLabel": "t",
                            "tokenLast4": "OKEN",
                        },
                    )
                return httpx.Response(status, json={"message": "Unauthorized"})

            return handler

        for status in (401, 302):
            seen: list[str] = []
            handler = refusing_the_guide(status, seen)
            with self.subTest(status=status):
                with self.assertRaises(EvalGuideNotOffered) as ctx:
                    await _client(handler).get_eval_guide("env-1")
                # still a RogueArenaApiError carrying the guide's own answer
                self.assertIsInstance(ctx.exception, RogueArenaApiError)
                self.assertEqual(ctx.exception.status_code, status)
                self.assertEqual(ctx.exception.path, "/hub/eval/evals/env-1/guide")
                self.assertIn("does not offer eval guides", str(ctx.exception))
                self.assertEqual(
                    seen, ["/hub/eval/evals/env-1/guide", "/hub/eval/whoami"]
                )

    async def test_a_refused_guide_with_a_refused_token_stays_the_guides_error(
        self,
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"errorMessage": "token expired"})

        with self.assertRaises(RogueArenaApiError) as ctx:
            await _client(handler).get_eval_guide("env-1")
        self.assertNotIsInstance(ctx.exception, EvalGuideNotOffered)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.path, "/hub/eval/evals/env-1/guide")

    async def test_list_runs_parses_the_jobs(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                (request.method, request.url.path), ("GET", "/hub/eval/jobs")
            )
            return httpx.Response(200, json=[_job_json("open"), _job_json("complete")])

        runs = await _client(handler).list_runs()
        self.assertEqual(
            [run.status for run in runs], [EvalJobStatus.OPEN, EvalJobStatus.COMPLETE]
        )

    async def test_list_runs_surfaces_a_missing_endpoint_as_an_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"errorMessage": "not found"})

        with self.assertRaises(RogueArenaApiError) as ctx:
            await _client(handler).list_runs()
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_chatroom_post_is_idempotent_and_reads_in_order(self) -> None:
        posts: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if (
                request.method == "POST"
                and request.url.path == "/hub/eval/jobs/job-1/chat"
            ):
                posts.append(json.loads(request.content))
                return httpx.Response(
                    200,
                    json={
                        "id": "m1",
                        "jobId": "job-1",
                        "seq": 3,
                        "at": "2026-09-30T00:00:00.000Z",
                        "author": "planner",
                        "content": "found it",
                    },
                )
            if (
                request.method == "GET"
                and request.url.path == "/hub/eval/jobs/job-1/chat"
            ):
                self.assertEqual(request.url.params.get("since"), "2")
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "m1",
                            "jobId": "job-1",
                            "seq": 3,
                            "at": "2026-09-30T00:00:00.000Z",
                            "author": "planner",
                            "content": "found it",
                        }
                    ],
                )
            return httpx.Response(500, json={"errorMessage": "x", "errorSeverity": 2})

        client = _client(handler)
        posted = await client.post_chat_message(
            "job-1", author="planner", content="found it", client_message_key="k1"
        )
        self.assertIsInstance(posted, EvalChatMessage)
        self.assertEqual((posted.seq, posted.author), (3, "planner"))
        self.assertEqual(posts[0]["clientMessageKey"], "k1")
        read = await client.list_chat_messages("job-1", since=2)
        self.assertEqual([message.content for message in read], ["found it"])

    async def test_activity_post_is_idempotent_and_reads_in_order(self) -> None:
        posts: list[dict[str, object]] = []
        entry = {
            "id": "a1",
            "jobId": "job-1",
            "seq": 5,
            "at": "2026-09-30T01:00:00.000Z",
            "member": "recon",
            "command": "nmap -sV 203.0.113.5",
            "summary": "exit 0; 3 ports open",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if (
                request.method == "POST"
                and request.url.path == "/hub/eval/jobs/job-1/activity"
            ):
                posts.append(json.loads(request.content))
                return httpx.Response(200, json=entry)
            if (
                request.method == "GET"
                and request.url.path == "/hub/eval/jobs/job-1/activity"
            ):
                self.assertEqual(request.url.params.get("since"), "4")
                return httpx.Response(200, json=[entry])
            return httpx.Response(500, json={"errorMessage": "x", "errorSeverity": 2})

        client = _client(handler)
        posted = await client.post_activity_entry(
            "job-1",
            member="recon",
            command="nmap -sV 203.0.113.5",
            summary="exit 0; 3 ports open",
            client_entry_key="k1",
        )
        self.assertIsInstance(posted, EvalActivityEntry)
        self.assertEqual(
            (posted.seq, posted.member, posted.summary),
            (5, "recon", "exit 0; 3 ports open"),
        )
        self.assertEqual(posts[0]["clientEntryKey"], "k1")
        read = await client.list_activity_entries("job-1", since=4)
        self.assertEqual([e.command for e in read], ["nmap -sV 203.0.113.5"])

    async def test_create_and_cancel_job(self) -> None:
        created: list[dict[str, object]] = []
        cancels: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content) if request.content else {}
            if request.method == "POST" and request.url.path == "/hub/eval/jobs":
                created.append(body)
                return httpx.Response(200, json=_job_json("open"))
            if request.url.path == "/hub/eval/jobs/job-1/cancel":
                cancels.append(request.content)
                return httpx.Response(200, json=_job_json("cancelling"))
            return httpx.Response(500, json={"errorMessage": "x", "errorSeverity": 2})

        client = _client(handler)
        job = await client.create_job(
            environment_id="11111111-1111-4111-8111-111111111111",
            n=4,
            concurrency=2,
            wall_clock_limit_seconds=3600,
            client_job_key="inspect_run",
        )
        self.assertIs(job.status, EvalJobStatus.OPEN)
        self.assertEqual(created[0]["clientJobKey"], "inspect_run")
        self.assertNotIn(
            "visibility", created[0]
        )  # omitted when not passed (Rogue Arena defaults it)
        cancelled = await client.cancel_job("job-1")
        self.assertIs(cancelled.status, EvalJobStatus.CANCELLING)
        self.assertEqual([json.loads(body) for body in cancels], [{}])

    async def test_create_job_sends_visibility_only_for_communicate_only(self) -> None:
        created: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            created.append(json.loads(request.content))
            return httpx.Response(200, json=_job_json("open"))

        client = _client(handler)
        kwargs = {
            "environment_id": "11111111-1111-4111-8111-111111111111",
            "n": 1,
            "concurrency": 1,
            "wall_clock_limit_seconds": 600,
        }
        await client.create_job(**kwargs, visibility="communicate_only")
        self.assertEqual(created[0]["visibility"], "communicate_only")
        # shared_sight (the default) and no visibility are never sent.
        await client.create_job(**kwargs, visibility="shared_sight")
        self.assertNotIn("visibility", created[1])
        await client.create_job(**kwargs)
        self.assertNotIn("visibility", created[2])

    async def test_keep_range_is_sent_only_when_asked_and_read_back(self) -> None:
        created: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            created.append(body)
            return httpx.Response(
                200,
                json={
                    **_job_json("open"),
                    "keepRange": bool(body.get("keepRange")),
                    "rangeClosedAt": None,
                },
            )

        client = _client(handler)
        kwargs = {
            "environment_id": "11111111-1111-4111-8111-111111111111",
            "n": 1,
            "concurrency": 1,
            "wall_clock_limit_seconds": 600,
        }
        kept = await client.create_job(**kwargs, keep_range=True)
        self.assertIs(created[0]["keepRange"], True)
        self.assertTrue(kept.keep_range and kept.range_kept_open)
        plain = await client.create_job(**kwargs)
        self.assertNotIn("keepRange", created[1])
        self.assertFalse(plain.keep_range or plain.range_kept_open)
        # A hub that predates keep-range answers no field: read as not kept.
        self.assertFalse(EvalJob.from_dict(_job_json("open")).keep_range)
        # Every kept range removed (keptRangeCount 0) is no longer kept open, though the range is not closed.
        emptied = EvalJob.from_dict(
            {
                **_job_json("open"),
                "keepRange": True,
                "rangeClosedAt": None,
                "keptRangeCount": 0,
            }
        )
        self.assertEqual(emptied.kept_range_count, 0)
        self.assertFalse(emptied.range_kept_open)
        # A hub without the count: kept open until closed, as before.
        self.assertTrue(
            EvalJob.from_dict(
                {**_job_json("open"), "keepRange": True, "rangeClosedAt": None}
            ).range_kept_open
        )

    async def test_close_range_posts_an_empty_body_to_the_runs_close_range(
        self,
    ) -> None:
        posts: list[tuple[str, bytes]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append((request.url.path, request.content))
            return httpx.Response(
                200,
                json={
                    **_job_json("cancelled"),
                    "keepRange": True,
                    "rangeClosedAt": "2026-11-17T21:00:00.000Z",
                },
            )

        closed = await _client(handler).close_range("job-1")
        self.assertEqual(
            [(path, json.loads(body)) for path, body in posts],
            [("/hub/eval/jobs/job-1/close-range", {})],
        )
        self.assertEqual(closed.range_closed_at, "2026-11-17T21:00:00.000Z")
        self.assertFalse(closed.range_kept_open)

    async def test_create_job_raises_a_visibility_rejection_without_retrying(
        self,
    ) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                400,
                json={
                    "errorMessage": "Unrecognized key: visibility",
                    "errorSeverity": 2,
                },
            )

        client = _client(handler)
        with self.assertRaises(RogueArenaApiError):
            await client.create_job(
                environment_id="11111111-1111-4111-8111-111111111111",
                n=1,
                concurrency=1,
                wall_clock_limit_seconds=600,
                visibility="communicate_only",
            )
        # Sent exactly once, with the field: a 400 is raised, not retried.
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0].get("visibility"), "communicate_only")

    async def test_create_job_raises_an_error_without_retrying(self) -> None:
        for status, body in (
            (409, {"errorMessage": "clientJobKey conflict"}),
            (500, {"errorMessage": "boom"}),
            (400, {"errorMessage": "n must be positive"}),
        ):
            attempts: list[dict[str, object]] = []

            def handler(
                request: httpx.Request,
                _body: object = body,
                _status: int = status,
                _attempts: list[dict[str, object]] = attempts,
            ) -> httpx.Response:
                _attempts.append(json.loads(request.content))
                return httpx.Response(_status, json={**_body, "errorSeverity": 2})

            client = _client(handler)
            with self.assertRaises(RogueArenaApiError):
                await client.create_job(
                    environment_id="11111111-1111-4111-8111-111111111111",
                    n=1,
                    concurrency=1,
                    wall_clock_limit_seconds=600,
                    visibility="communicate_only",
                )
            # a 409/5xx, or a 400 that is not about visibility, is raised without a retry
            self.assertEqual(len(attempts), 1, f"status {status} should not retry")

    async def test_api_error_carries_status_and_body(self) -> None:
        client = _client(
            lambda r: httpx.Response(
                409, json={"errorMessage": "unpinned", "errorSeverity": 2}
            )
        )
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.get_job("job-x")
        self.assertEqual(caught.exception.status_code, 409)

    async def test_a_non_json_error_body_is_kept_as_text(self) -> None:
        client = _client(lambda r: httpx.Response(503, text="upstream unavailable"))
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.get_job("job-x")
        self.assertEqual(caught.exception.body, "upstream unavailable")
        self.assertIn("upstream unavailable", str(caught.exception))

    async def test_an_empty_or_204_body_is_none(self) -> None:
        for response in (httpx.Response(204), httpx.Response(200, content=b"")):
            with self.subTest(status=response.status_code):
                client = _client(lambda r, response=response: response)
                result = await client.request_result("GET", "/hub/eval/whoami")
                self.assertIsNone(result.body)

    async def test_with_token_changes_the_bearer_and_the_fingerprint(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["Authorization"])
            return httpx.Response(200, json={})

        client = _client(handler)
        other = client.with_token("ra_eval_OTHER")
        await client.request_result("GET", "/hub/eval/whoami")
        await other.request_result("GET", "/hub/eval/whoami")
        self.assertEqual(seen, ["Bearer ra_eval_TOKEN", "Bearer ra_eval_OTHER"])
        self.assertNotEqual(client.credential_fingerprint, other.credential_fingerprint)
        self.assertEqual(other.base_url, client.base_url)

    async def test_a_long_error_body_is_truncated_in_the_message_only(self) -> None:
        long_body = {"errorMessage": "x" * 5000}
        client = _client(lambda r: httpx.Response(500, json=long_body))
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.get_job("job-x")
        message = str(caught.exception)
        self.assertLess(len(message), 700)
        self.assertIn("more characters]", message)
        self.assertEqual(caught.exception.body, long_body)

    async def test_a_token_echoed_across_the_cut_leaves_no_fragment(self) -> None:
        # The body is redacted BEFORE the 500-character cut: the configured
        # token (whatever its shape) and any other ra_eval_ run.
        cases = (
            ("ra_eval_S3cretTokenValueNeverPrinted9f2c", None),
            ("unprefixedS3cretValueWithoutThePrefix77", None),
            ("ra_eval_TOKEN", "ra_eval_SomeoneElsesTokenSeenUpstream42"),
        )
        for configured, echoed in cases:
            token = echoed or configured
            with self.subTest(token=token):
                body = _straddling_body(token)
                client = RogueArenaClient(
                    "https://hub.example.test",
                    configured,
                    transport=httpx.MockTransport(
                        lambda r, body=body: httpx.Response(500, json=body)
                    ),
                )
                with self.assertRaises(RogueArenaApiError) as caught:
                    await client.whoami()
                message = str(caught.exception)
                self.assertIn("Bearer [redacted]", message)
                self.assertEqual(_token_fragments(message, token), [])
                self.assertEqual(caught.exception.body, body)  # raw body kept

        titled = f"<html><head><title>{'y' * 100} {TOKEN}</title></head></html>"
        client = RogueArenaClient(
            "https://hub.example.test",
            TOKEN,
            transport=httpx.MockTransport(lambda r: httpx.Response(502, text=titled)),
        )
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.whoami()
        self.assertEqual(_token_fragments(str(caught.exception), TOKEN), [])

    async def test_a_gateway_redirect_or_html_page_is_one_clear_line(self) -> None:
        login = "<!DOCTYPE html><html><head><title>Sign in</title></head>" + (
            "<body>" + "x" * 4000 + "</body></html>"
        )
        redirect = _client(
            lambda r: httpx.Response(
                302, headers={"Location": "https://sso.example/auth"}, text=login
            )
        )
        with self.assertRaises(RogueArenaApiError) as caught:
            await redirect.whoami()
        message = str(caught.exception)
        self.assertEqual(caught.exception.status_code, 302)
        self.assertIn("redirected the request", message)
        self.assertIn("check the base URL", message)
        self.assertNotIn("<html", message)
        self.assertNotIn("\n", message)
        self.assertEqual(caught.exception.body, login)

        page = (
            "<html>\n<head><title>502 Bad Gateway</title></head>\n<body></body></html>"
        )
        gateway = _client(lambda r: httpx.Response(502, text=page))
        with self.assertRaises(RogueArenaApiError) as caught:
            await gateway.whoami()
        message = str(caught.exception)
        self.assertIn("an HTML page titled '502 Bad Gateway'", message)
        self.assertNotIn("<html", message)
        self.assertNotIn("\n", message)

    async def test_a_non_string_where_the_contract_says_string_is_a_type_error(
        self,
    ) -> None:
        for field, value in (("deploymentRecordId", 5), ("agentMachineNickname", [])):
            with self.subTest(field=field):
                body = _trial_json("ready", **{field: value})
                client = _client(lambda r, body=body: httpx.Response(200, json=body))
                with self.assertRaisesRegex(TypeError, field):
                    await client.get_trial("trial-1")


TOKEN = "ra_eval_S3cretTokenValueNeverPrinted9f2c"


def _straddling_body(token: str) -> dict[str, str]:
    """An error body whose text puts ``token`` across the 500-character cut."""
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


def _job_json(status: str) -> dict[str, object]:
    return {
        "id": "job-1",
        "environmentId": "11111111-1111-4111-8111-111111111111",
        "environmentBuildId": "22222222-2222-4222-8222-222222222222",
        "canvasVersionId": "33333333-3333-4333-8333-333333333333",
        "ownerUserMainId": "user-1",
        "organizationId": "org-1",
        "n": 4,
        "concurrency": 2,
        "wallClockLimitSeconds": 3600,
        "clientJobKey": "inspect_run",
        "status": status,
        "trialCount": 0,
        "closedTrialCount": 0,
        "createdAt": "2026-09-30T00:00:00.000Z",
        "updatedAt": "2026-09-30T00:00:00.000Z",
    }


def _trial_json(status: str, **overrides: object) -> dict[str, object]:
    trial: dict[str, object] = {
        "id": "trial-1",
        "jobId": "job-1",
        "deploymentRecordId": None,
        "agentDeployedVmId": None,
        "agentMachineNickname": None,
        "status": status,
        "outcome": None,
        "terminalReason": None,
        "warm": False,
        "reachedStep": None,
        "hopsPassed": None,
        "totalSteps": 2,
        "claimedAt": "2026-09-30T00:00:00.000Z",
        "readyAt": None,
        "closedAt": None,
        "lastHeartbeatAt": "2026-09-30T00:00:00.000Z",
        "leaseExpiresAt": "2026-09-30T00:15:00.000Z",
        "clientKind": "inspect",
        "clientAttemptKey": "attempt-1",
        "score": None,
        "hopResults": None,
    }
    trial.update(overrides)
    return trial


class RogueArenaClaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_polls_a_provisioning_trial_to_ready(self) -> None:
        polls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST" and request.url.path == "/hub/eval/trials":
                return httpx.Response(200, json=_trial_json("provisioning"))
            if request.url.path == "/hub/eval/trials/trial-1":
                polls += 1
                if polls == 1:
                    raise httpx.ConnectError("blip", request=request)
                if polls == 2:
                    return httpx.Response(200, json=_trial_json("provisioning"))
                return httpx.Response(
                    200,
                    json=_trial_json(
                        "ready",
                        agentDeployedVmId="vm-1",
                        readyAt="2026-09-30T00:01:00.000Z",
                    ),
                )
            return httpx.Response(500, json={"errorMessage": "x", "errorSeverity": 2})

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        trial = await client.claim_trial(
            "job-1",
            client_attempt_key="attempt-1",
            inspect_sample_id="s1",
            inspect_epoch=1,
            inspect_sample_uuid=None,
            inspect_eval_id=None,
            inspect_run_id="run-1",
            inspect_eval_set_id=None,
            inspect_task_id="task",
            sandbox_name="gold",
            max_wait_seconds=30,
            poll_interval_seconds=1,
        )
        self.assertIs(trial.status, EvalTrialStatus.READY)
        self.assertEqual(trial.agent_deployed_vm_id, "vm-1")

    async def test_claim_402_funding_denial_carries_the_decision(self) -> None:
        denied = {
            "errorMessage": "wallet empty",
            "decision": _decision("deny", "wallet_funds_insufficient"),
        }
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(lambda r: httpx.Response(402, json=denied)),
        )
        with self.assertRaises(EvalClaimDenied) as caught:
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertTrue(caught.exception.is_funding)
        self.assertEqual(
            caught.exception.denial.decision["reasonCode"], "wallet_funds_insufficient"
        )
        self.assertEqual(
            caught.exception.denial.reason_code, "wallet_funds_insufficient"
        )
        self.assertEqual(caught.exception.denial.verdict, "deny")

    async def test_claim_409_slot_denial_carries_the_decision(self) -> None:
        denied = {
            "errorMessage": "slots full",
            "decision": _decision("deny", "organization_slots_full"),
        }
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(lambda r: httpx.Response(409, json=denied)),
        )
        with self.assertRaises(EvalClaimDenied) as caught:
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertFalse(caught.exception.is_funding)

    async def test_claim_409_plain_state_error_is_not_a_denial(self) -> None:
        posts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(request.url.path)
            return httpx.Response(
                409,
                json={
                    "errorMessage": "Job is at its concurrency limit",
                    "errorSeverity": 2,
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertNotIsInstance(caught.exception, EvalClaimDenied)
        self.assertEqual(caught.exception.status_code, 409)
        # a definite refusal is the hub's answer: never resent
        self.assertEqual(posts, ["/hub/eval/trials"])

    async def test_claim_resent_after_a_502_returns_the_same_trial(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            if len(bodies) == 1:
                return httpx.Response(502, text="Bad Gateway")
            return httpx.Response(200, json=_trial_json("ready"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        trial = await client.claim_trial("job-1", **_claim_kwargs())
        self.assertEqual(trial.id, "trial-1")
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0], bodies[1])

    async def test_close_reads_back_the_server_score(self) -> None:
        score = {
            "reachedStep": 1,
            "hopsPassed": 1,
            "totalSteps": 2,
            "scoringMode": "sequential",
            "steps": [
                {
                    "stepNumber": 1,
                    "name": "foothold",
                    "passed": True,
                    "passedAt": "2026-09-30T00:05:00.000Z",
                    "detail": "hit",
                },
                {
                    "stepNumber": 2,
                    "name": "root",
                    "passed": False,
                    "passedAt": None,
                    "detail": "miss",
                },
            ],
        }
        hop_results = [
            {
                "stepNumber": 1,
                "name": "foothold",
                "targetFound": True,
                "checkRan": True,
                "passed": True,
                "detail": "hit",
            }
        ]
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json=_trial_json(
                        "closed",
                        outcome="closed",
                        terminalReason="agent_fail",
                        reachedStep=1,
                        hopsPassed=1,
                        score=score,
                        hopResults=hop_results,
                    ),
                )
            ),
        )
        trial = await client.close_trial("trial-1", outcome=EvalTrialOutcome.CLOSED)
        assert trial.score is not None
        self.assertEqual(trial.score.hops_passed, 1)
        self.assertIs(trial.terminal_reason, EvalTrialTerminalReason.AGENT_FAIL)
        self.assertEqual(trial.hop_results[0].passed, True)


async def _no_sleep(_seconds: float) -> None:
    return None


def _decision(kind: str, reason: str) -> dict[str, object]:
    # A refusal body carries the admission decision with exactly these keys.
    return {
        "verdict": kind,
        "lane": "deployment",
        "fundedBy": None,
        "reasonCode": reason,
        "validUntil": None,
        "validityBoundary": {"kind": "none", "at": None},
        "remediation": None,
        "retryAfterSeconds": None,
        "evidence": None,
        "trace": [],
        "checks": [],
        "decisionId": "44444444-4444-4444-8444-444444444444",
        "policyVersion": "1",
        "factVersions": {},
    }


def _claim_kwargs() -> dict[str, object]:
    return {
        "client_attempt_key": "attempt-1",
        "inspect_sample_id": "s1",
        "inspect_epoch": None,
        "inspect_sample_uuid": None,
        "inspect_eval_id": None,
        "inspect_run_id": "run-1",
        "inspect_eval_set_id": None,
        "inspect_task_id": "task",
        "sandbox_name": "gold",
        "max_wait_seconds": 5,
        "poll_interval_seconds": 1,
    }


class RogueArenaTrialLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_never_reclaims_and_times_out_while_provisioning(self) -> None:
        clock = 0.0
        claims = 0
        closes: list[object] = []
        sleeps: list[float] = []

        def monotonic() -> float:
            return clock

        async def sleep(seconds: float) -> None:
            nonlocal clock
            sleeps.append(seconds)
            clock += seconds

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal claims
            if request.method == "POST" and request.url.path == "/hub/eval/trials":
                claims += 1
            if request.url.path.endswith("/close"):
                closes.append(json.loads(request.content))
                return httpx.Response(
                    200, json=_trial_json("closed", outcome="cancelled")
                )
            return httpx.Response(200, json=_trial_json("provisioning"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=sleep,
            monotonic=monotonic,
        )
        with self.assertRaises(TrialReadyTimeout):
            await client.claim_trial(
                "job-1", **{**_claim_kwargs(), "max_wait_seconds": 3}
            )
        self.assertEqual(claims, 1)
        self.assertEqual(sleeps, [1.0, 1.0, 1.0])
        self.assertEqual(closes, [{"outcome": "cancelled"}])

    async def test_claim_ready_immediately_does_not_poll(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return httpx.Response(200, json=_trial_json("ready"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        trial = await client.claim_trial("job-1", **_claim_kwargs())
        self.assertIs(trial.status, EvalTrialStatus.READY)
        self.assertEqual(paths, ["/hub/eval/trials"])

    async def test_terminal_closed_during_poll_is_a_distinct_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json=_trial_json("provisioning"))
            return httpx.Response(
                200,
                json=_trial_json(
                    "closed", outcome="expired", terminalReason="infra_error"
                ),
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        with self.assertRaises(TrialClosedBeforeReady) as caught:
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertIs(
            caught.exception.trial.terminal_reason,
            EvalTrialTerminalReason.INFRA_ERROR,
        )
        # this hub answers the event read with a trial, not an event list: the
        # close reason stays unknown rather than guessed
        self.assertIsNone(caught.exception.reason)

    async def test_poll_404_is_fatal(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, json=_trial_json("provisioning"))
            return httpx.Response(404, json={"errorMessage": "nope"})

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        with self.assertRaises(RogueArenaApiError) as caught:
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertEqual(caught.exception.status_code, 404)

    async def test_claim_body_is_the_strict_fifteen_field_shape(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=_trial_json("ready"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        await client.claim_trial("job-1", **_claim_kwargs())
        self.assertEqual(
            set(seen),
            {
                "jobId",
                "clientAttemptKey",
                "clientKind",
                "inspectSampleId",
                "inspectEpoch",
                "inspectSampleUuid",
                "inspectEvalId",
                "inspectRunId",
                "inspectEvalSetId",
                "inspectTaskId",
                "harborTrialUuid",
                "harborTrialName",
                "harborJobId",
                "harborAttempt",
                "sandboxName",
            },
        )
        self.assertEqual(seen["clientKind"], "inspect")
        self.assertEqual(seen["jobId"], "job-1")
        self.assertIsNone(seen["harborTrialUuid"])
        self.assertEqual(seen["sandboxName"], "gold")

    async def test_close_sends_only_outcome_and_heartbeat_sends_empty_body(
        self,
    ) -> None:
        bodies: dict[str, list[object]] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.setdefault(request.url.path, []).append(json.loads(request.content))
            return httpx.Response(200, json=_trial_json("closed", outcome="cancelled"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        trial = await client.close_trial("trial-1", outcome=EvalTrialOutcome.CANCELLED)
        await client.close_trial("trial-1")
        await client.heartbeat_trial("trial-1")
        self.assertIs(trial.outcome, EvalTrialOutcome.CANCELLED)
        self.assertEqual(bodies["/hub/eval/trials/trial-1/heartbeat"], [{}])
        self.assertEqual(
            bodies["/hub/eval/trials/trial-1/close"], [{"outcome": "cancelled"}, {}]
        )

    async def test_close_rejects_server_only_outcomes(self) -> None:
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(lambda r: httpx.Response(500)),
        )
        with self.assertRaises(ValueError):
            await client.close_trial("trial-1", outcome=EvalTrialOutcome.EXPIRED)

    async def test_close_after_lease_expiry_reads_back_expired(self) -> None:
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200,
                    json=_trial_json(
                        "closed", outcome="expired", terminalReason="timeout"
                    ),
                )
            ),
        )
        trial = await client.close_trial("trial-1", outcome=EvalTrialOutcome.CLOSED)
        self.assertIs(trial.outcome, EvalTrialOutcome.EXPIRED)
        self.assertIs(trial.terminal_reason, EvalTrialTerminalReason.TIMEOUT)

    async def test_heartbeat_409_and_get_404_surface_as_api_errors(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(409, json={"errorMessage": "not running"})
            return httpx.Response(404, json={"errorMessage": "missing"})

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(RogueArenaApiError) as hb:
            await client.heartbeat_trial("trial-1")
        self.assertEqual(hb.exception.status_code, 409)
        with self.assertRaises(RogueArenaApiError) as got:
            await client.get_trial("trial-1")
        self.assertEqual(got.exception.status_code, 404)

    async def test_poll_retries_a_503_then_reaches_ready(self) -> None:
        gets = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal gets
            if request.method == "POST":
                return httpx.Response(200, json=_trial_json("provisioning"))
            gets += 1
            if gets == 1:
                return httpx.Response(503, json={"errorMessage": "busy"})
            return httpx.Response(200, json=_trial_json("ready"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        trial = await client.claim_trial("job-1", **_claim_kwargs())
        self.assertIs(trial.status, EvalTrialStatus.READY)
        self.assertEqual(gets, 2)

    async def test_claim_that_returns_closed_raises_closed_before_ready(self) -> None:
        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, json=_trial_json("closed", outcome="cancelled")
                )
            ),
        )
        with self.assertRaises(TrialClosedBeforeReady):
            await client.claim_trial("job-1", **_claim_kwargs())

    async def test_close_polls_a_closing_readback_until_closed(self) -> None:
        reads = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal reads
            if request.method == "POST":
                return httpx.Response(200, json=_trial_json("closing"))
            reads += 1
            status = "closed" if reads >= 2 else "closing"
            return httpx.Response(200, json=_trial_json(status, outcome="closed"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        trial = await client.close_trial("trial-1")
        self.assertIs(trial.status, EvalTrialStatus.CLOSED)
        self.assertEqual(reads, 2)

    async def test_close_504_polls_and_never_resends_close(self) -> None:
        posts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal posts
            if request.method == "POST":
                posts += 1
                return httpx.Response(504, text="gateway timeout")
            return httpx.Response(200, json=_trial_json("closed", outcome="closed"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        trial = await client.close_trial("trial-1")
        self.assertIs(trial.status, EvalTrialStatus.CLOSED)
        self.assertEqual(posts, 1)

    async def test_close_timeout_transport_error_polls_instead_of_resending(
        self,
    ) -> None:
        posts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal posts
            if request.method == "POST":
                posts += 1
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json=_trial_json("closed", outcome="closed"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        trial = await client.close_trial("trial-1")
        self.assertIs(trial.status, EvalTrialStatus.CLOSED)
        self.assertEqual(posts, 1)

    async def test_cancelled_poll_best_effort_cancels_the_claimed_trial(self) -> None:
        closes: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/close"):
                closes.append(json.loads(request.content))
                return httpx.Response(200, json=_trial_json("closed"))
            return httpx.Response(200, json=_trial_json("provisioning"))

        async def cancelling_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=cancelling_sleep,
        )
        with self.assertRaises(asyncio.CancelledError):
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertEqual(closes, [{"outcome": "cancelled"}])

    async def test_an_unparseable_poll_answer_cancels_the_claimed_trial(
        self,
    ) -> None:
        closes: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/close"):
                closes.append(json.loads(request.content))
                return httpx.Response(
                    200, json=_trial_json("closed", outcome="cancelled")
                )
            if request.method == "POST":
                return httpx.Response(200, json=_trial_json("provisioning"))
            return httpx.Response(200, json=_trial_json("draining"))

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        with self.assertRaisesRegex(ValueError, "draining"):
            await client.claim_trial("job-1", **_claim_kwargs())
        self.assertEqual(closes, [{"outcome": "cancelled"}])

    async def test_lists_trials_and_events(self) -> None:
        event = {
            "id": "55555555-5555-4555-8555-555555555555",
            "at": "2026-09-30T00:02:00.000Z",
            "kind": "hop",
            "summary": "foothold passed",
            "detail": None,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/hub/eval/jobs/job-1/trials":
                return httpx.Response(200, json=[_trial_json("running")])
            return httpx.Response(200, json=[event])

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
        )
        trials = await client.list_job_trials("job-1")
        self.assertIs(trials[0].status, EvalTrialStatus.RUNNING)
        events = await client.list_trial_events("trial-1")
        self.assertIs(events[0].kind, EvalTrialEventKind.HOP)
        self.assertEqual(events[0].summary, "foothold passed")
        self.assertIsNone(events[0].detail)


_EXEC_ARGS = {
    "cmd": ["whoami"],
    "cwd": None,
    "env": None,
    "stdin": None,
    "sync_ceiling_seconds": 50,
    "poll_interval_seconds": 1,
    "max_total_seconds": 14400,
}


def _exec_client(handler) -> RogueArenaClient:
    return RogueArenaClient(
        "https://hub.example.test",
        "ra_eval_T",
        transport=httpx.MockTransport(handler),
        sleep=_no_sleep,
    )


class _ProvisioningHub:
    """A hub whose claimed trial stays provisioning until a set (fake) time.

    ``ends_at`` is when the hub ends the provisioning wait: the trial turns
    ``ends_as`` (a trial body), or stays provisioning forever when ``ends_at``
    is None. Time only moves when the client sleeps.
    """

    def __init__(
        self,
        *,
        ends_at: float | None,
        ends_as: dict[str, object] | None = None,
        events: list[dict[str, object]] | None = None,
    ) -> None:
        self.clock = 0.0
        self.ends_at = ends_at
        self.ends_as = ends_as
        self.events = events or []
        self.polls = 0
        self.closes: list[object] = []

    def monotonic(self) -> float:
        return self.clock

    async def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path == "/hub/eval/trials":
            return httpx.Response(200, json=_trial_json("provisioning"))
        if path.endswith("/close"):
            self.closes.append(json.loads(request.content))
            return httpx.Response(200, json=_trial_json("closed", outcome="cancelled"))
        if path.endswith("/events"):
            return httpx.Response(200, json=self.events)
        self.polls += 1
        if self.ends_at is not None and self.clock >= self.ends_at:
            assert self.ends_as is not None
            return httpx.Response(200, json=self.ends_as)
        return httpx.Response(200, json=_trial_json("provisioning"))

    def client(self) -> RogueArenaClient:
        return RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(self.handler),
            sleep=self.sleep,
            monotonic=self.monotonic,
        )


def _lifecycle(event_id: str, at: str, summary: str) -> dict[str, object]:
    return {
        "id": event_id,
        "at": at,
        "kind": "lifecycle",
        "summary": summary,
        "detail": None,
    }


CLAIMED_AT = "2026-11-04T00:00:00.000Z"
# A provisioning trial's lease is its claim plus the wall clock (one hour here).
LEASE_AT = "2026-11-04T01:00:00.000Z"
STALL_REASON = (
    "expired while provisioning: the copy's build made no progress for 30 min "
    "or ran past 3 h"
)


def _expired(closed_at: str) -> dict[str, object]:
    return _trial_json(
        "closed",
        outcome="expired",
        terminalReason="infra_error",
        infraReason="provisioning_failed",
        claimedAt=CLAIMED_AT,
        leaseExpiresAt=LEASE_AT,
        closedAt=closed_at,
    )


def _close_events(closed_at: str, *reasons: str) -> list[dict[str, object]]:
    """The hub's lifecycle rows: the claim, then the close's reason rows and
    its closing summary (always written last, in the close's transaction)."""
    return [
        _lifecycle("e0", CLAIMED_AT, "claimed: cold copy dep-1 requested"),
        *(
            _lifecycle(f"e{index}", closed_at, reason)
            for index, reason in enumerate(reasons, start=1)
        ),
        _lifecycle(
            "e9",
            closed_at,
            "closed: expired (infra_error: provisioning_failed), 0/2 hops",
        ),
    ]


class RogueArenaReadyWaitTests(unittest.IsolatedAsyncioTestCase):
    """The ready wait follows the hub's provisioning state, not a fixed age."""

    # the sandbox's default wait, so these pin the shipped behaviour
    DEFAULT_WAIT = RogueArenaSandboxConfig().max_wait_seconds

    def _kwargs(self) -> dict[str, object]:
        return {
            **_claim_kwargs(),
            "max_wait_seconds": self.DEFAULT_WAIT,
            "poll_interval_seconds": 30,
        }

    def test_the_default_backstop_outlasts_the_hubs_three_hour_build_limit(
        self,
    ) -> None:
        # The hub ends any build 3 hours after its claim; its answer lands first.
        self.assertEqual(self.DEFAULT_WAIT, 3 * 60 * 60 + 5 * 60)

    def test_the_wall_clock_does_not_change_the_backstop(self) -> None:
        # Build time no longer spends max_runtime_seconds, so the two are unrelated.
        longer = RogueArenaSandboxConfig(max_runtime_seconds=6 * 60 * 60)
        self.assertEqual(longer.max_wait_seconds, self.DEFAULT_WAIT)

    def test_an_explicit_max_wait_is_the_backstop(self) -> None:
        config = RogueArenaSandboxConfig.from_sources(
            None, environ={"ROGUE_ARENA_MAX_WAIT_SECONDS": "900"}
        )
        self.assertEqual(config.max_wait_seconds, 900)

    async def test_a_slow_build_the_hub_still_reports_provisioning_is_waited_for(
        self,
    ) -> None:
        # A large scenario's cold build takes 40 minutes, well past the old
        # fixed 15-minute wait; the hub reports it provisioning all along.
        hub = _ProvisioningHub(ends_at=40 * 60, ends_as=_trial_json("ready"))

        trial = await hub.client().claim_trial("job-1", **self._kwargs())

        self.assertIs(trial.status, EvalTrialStatus.READY)
        self.assertGreaterEqual(hub.clock, 40 * 60)
        self.assertEqual(hub.closes, [])

    async def test_a_stalled_build_ends_when_the_hub_expires_it_with_its_reason(
        self,
    ) -> None:
        # The hub judges progress: after its no-progress window (before the
        # wall clock) it expires the trial as an infrastructure failure and
        # records why, then its closing summary.
        closed_at = "2026-11-04T00:45:00.000Z"
        hub = _ProvisioningHub(
            ends_at=45 * 60,
            ends_as=_expired(closed_at),
            events=_close_events(closed_at, STALL_REASON),
        )

        with self.assertRaises(TrialClosedBeforeReady) as caught:
            await hub.client().claim_trial("job-1", **self._kwargs())

        error = caught.exception
        # the reason row, not the closing summary or the claim row
        self.assertEqual(error.reason, STALL_REASON)
        self.assertEqual(error.trial.infra_reason, "provisioning_failed")
        message = str(error)
        self.assertIn("infraReason=provisioning_failed", message)
        self.assertIn(STALL_REASON, message)
        # no wall clock runs before ready, so a build never blames it
        self.assertNotIn("max_runtime_seconds", message)
        # the hub already closed it: the client never sends a cancel
        self.assertEqual(hub.closes, [])
        self.assertLess(hub.clock, self.DEFAULT_WAIT)

    async def test_a_close_the_hub_gave_no_reason_for_has_none(self) -> None:
        # Only the claim row and the closing summary: no reason to report, and
        # neither of those is passed off as one.
        closed_at = "2026-11-04T00:20:00.000Z"
        hub = _ProvisioningHub(
            ends_at=20 * 60,
            ends_as=_expired(closed_at),
            events=_close_events(closed_at),
        )

        with self.assertRaises(TrialClosedBeforeReady) as caught:
            await hub.client().claim_trial("job-1", **self._kwargs())

        self.assertIsNone(caught.exception.reason)

    async def test_the_backstop_cancels_a_build_the_hub_never_ends(self) -> None:
        hub = _ProvisioningHub(ends_at=None)

        with self.assertRaises(TrialReadyTimeout) as caught:
            await hub.client().claim_trial("job-1", **self._kwargs())

        self.assertEqual(caught.exception.max_wait_seconds, self.DEFAULT_WAIT)
        self.assertIn("still provisioning", str(caught.exception))
        self.assertEqual(hub.clock, self.DEFAULT_WAIT)
        self.assertEqual(hub.closes, [{"outcome": "cancelled"}])

    async def test_an_explicit_max_wait_still_overrides_the_backstop(self) -> None:
        explicit = RogueArenaSandboxConfig.from_sources(
            None, environ={"ROGUE_ARENA_MAX_WAIT_SECONDS": "900"}
        ).max_wait_seconds
        hub = _ProvisioningHub(ends_at=40 * 60, ends_as=_trial_json("ready"))

        with self.assertRaises(TrialReadyTimeout):
            await hub.client().claim_trial(
                "job-1", **{**self._kwargs(), "max_wait_seconds": explicit}
            )

        self.assertEqual(hub.clock, 900)
        self.assertEqual(hub.closes, [{"outcome": "cancelled"}])

    async def test_each_request_of_a_long_wait_has_its_own_short_timeout(
        self,
    ) -> None:
        # A two-hour wait never lets one hung request block for an hour.
        timeouts: list[float | None] = []
        original = RogueArenaClient.request_result

        async def recording(
            self_: RogueArenaClient, method: str, path: str, **kwargs: Any
        ) -> HttpResult:
            timeouts.append(kwargs.get("timeout_seconds"))
            return await original(self_, method, path, **kwargs)

        hub = _ProvisioningHub(ends_at=5 * 60, ends_as=_trial_json("ready"))
        with patch.object(RogueArenaClient, "request_result", recording):
            await hub.client().claim_trial("job-1", **self._kwargs())

        claim, *polls = timeouts
        self.assertEqual(claim, 900)
        self.assertTrue(polls)
        self.assertTrue(all(timeout == 60 for timeout in polls))


class RogueArenaJobModelTests(unittest.IsolatedAsyncioTestCase):
    """A job carries the model it evaluates, so runs can be compared by model."""

    async def _created_body(self, model: str | None) -> dict[str, object]:
        created: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            created.append(json.loads(request.content))
            return httpx.Response(200, json=_job_json("open"))

        await _client(handler).create_job(
            environment_id="env-1",
            n=1,
            concurrency=1,
            wall_clock_limit_seconds=600,
            client_job_key="inspect_run",
            model=model,
        )
        return created[0]

    async def test_create_job_sends_the_model_trimmed(self) -> None:
        body = await self._created_body("  openai/gpt-4o ")
        self.assertEqual(body["model"], "openai/gpt-4o")

    async def test_create_job_omits_an_unknown_or_blank_model(self) -> None:
        for model in (None, "", "   "):
            with self.subTest(model=model):
                self.assertNotIn("model", await self._created_body(model))

    async def test_create_job_cuts_the_model_at_the_hubs_200_characters(self) -> None:
        body = await self._created_body("p/" + "m" * 300)
        self.assertEqual(body["model"], "p/" + "m" * 198)
        body = await self._created_body("x" * 200)
        self.assertEqual(body["model"], "x" * 200)
        # a cut that ends on whitespace is trimmed, as the hub trims it
        body = await self._created_body("y" * 199 + " " + "z" * 10)
        self.assertEqual(body["model"], "y" * 199)

    async def test_a_job_carries_the_model_or_none(self) -> None:
        for model in ("openai/gpt-4o", None):
            with self.subTest(model=model):
                job_body = {**_job_json("open"), "model": model}
                job = await _client(
                    lambda r, job_body=job_body: httpx.Response(200, json=job_body)
                ).get_job("job-1")
                self.assertEqual(job.model, model)
        # a hub that does not send the field
        job = await _client(
            lambda r: httpx.Response(200, json=_job_json("open"))
        ).get_job("job-1")
        self.assertIsNone(job.model)

    async def test_a_non_string_job_model_is_refused(self) -> None:
        job_body = {**_job_json("open"), "model": 7}
        with self.assertRaisesRegex(TypeError, "model"):
            await _client(lambda r: httpx.Response(200, json=job_body)).get_job("j")


class RogueArenaTranscriptTests(unittest.IsolatedAsyncioTestCase):
    JOB = "job-1"
    DATA = b"inspect .eval bytes" * 10

    def _result(self, replayed: bool = False) -> dict[str, object]:
        return {
            "transcript": {
                "id": "t-1",
                "jobId": self.JOB,
                "status": "ready",
                "byteLength": len(self.DATA),
                "sha256": transcript_sha256(self.DATA),
                "fileName": "run.eval",
                "uploadedAt": "2026-11-05T00:00:00.000Z",
                "readyAt": "2026-11-05T00:00:01.000Z",
                "expiresAt": "2026-12-05T00:00:00.000Z",
            },
            "replayed": replayed,
        }

    def _clock_client(self, handler: object) -> tuple[RogueArenaClient, list[float]]:
        slept: list[float] = []
        clock = {"t": 0.0}

        async def sleep(seconds: float) -> None:
            slept.append(seconds)
            clock["t"] += seconds

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_TOKEN",
            transport=httpx.MockTransport(handler),
            sleep=sleep,
            monotonic=lambda: clock["t"],
        )
        return client, slept

    async def test_upload_sends_the_required_headers_and_the_raw_bytes(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["method"] = request.method
            seen["path"] = request.url.path
            seen["headers"] = dict(request.headers)
            seen["content"] = request.content
            return httpx.Response(200, json=self._result())

        result = await _client(handler).upload_transcript(
            self.JOB,
            data=self.DATA,
            idempotency_key="11111111-1111-4111-8111-111111111111",
            sha256=transcript_sha256(self.DATA),
            file_name="run.eval",
        )
        self.assertEqual(seen["method"], "PUT")
        self.assertEqual(seen["path"], f"/hub/eval/jobs/{self.JOB}/transcript")
        headers = seen["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["content-type"], "application/octet-stream")
        self.assertEqual(headers["content-length"], str(len(self.DATA)))
        self.assertEqual(
            headers["idempotency-key"], "11111111-1111-4111-8111-111111111111"
        )
        self.assertEqual(
            headers["x-eval-transcript-sha256"], transcript_sha256(self.DATA)
        )
        self.assertEqual(headers["x-eval-transcript-file-name"], "run.eval")
        self.assertEqual(seen["content"], self.DATA)
        self.assertEqual(result.transcript.id, "t-1")
        self.assertIs(result.replayed, False)

    def test_the_upload_key_is_a_v4_uuid_stable_for_a_job_and_file(self) -> None:
        key = transcript_upload_key(self.JOB, "a" * 64)
        self.assertEqual(uuid.UUID(key).version, 4)
        # the hub's v4 check also pins the variant
        self.assertEqual(uuid.UUID(key).variant, uuid.RFC_4122)
        self.assertEqual(key, transcript_upload_key(self.JOB, "a" * 64))
        self.assertNotEqual(key, transcript_upload_key(self.JOB, "b" * 64))
        self.assertNotEqual(key, transcript_upload_key("job-2", "a" * 64))

    async def test_two_separate_shares_of_one_file_use_one_key(self) -> None:
        # A retry from a new process must resume or replay: the hub refuses a new
        # key while the run still has a live upload.
        keys: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["idempotency-key"])
            return httpx.Response(200, json=self._result(replayed=len(keys) > 1))

        client, _ = self._clock_client(handler)
        await client.share_transcript(self.JOB, data=self.DATA)
        second = await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(keys[0], keys[1])
        self.assertIs(second.replayed, True)

    async def test_a_new_upload_can_ask_for_its_own_key(self) -> None:
        keys: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["idempotency-key"])
            return httpx.Response(200, json=self._result())

        client, _ = self._clock_client(handler)
        chosen = str(uuid.uuid4())
        await client.share_transcript(self.JOB, data=self.DATA, idempotency_key=chosen)
        self.assertEqual(keys, [chosen])

    async def test_an_upload_gets_32_minutes_whatever_the_client_wide_timeout(
        self,
    ) -> None:
        # A 100 MB upload on a slow link still ends with Rogue Arena's answer.
        # A lower client-wide timeout never cuts it.
        seen: list[dict[str, float]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"])
            return httpx.Response(200, json=self._result())

        for ceiling in (3600.0, 60.0):
            with self.subTest(request_timeout_seconds=ceiling):
                seen.clear()
                client = RogueArenaClient(
                    "https://hub.example.test",
                    "ra_eval_TOKEN",
                    transport=httpx.MockTransport(handler),
                    request_timeout_seconds=ceiling,
                )
                await client.share_transcript(self.JOB, data=self.DATA)
                self.assertEqual(seen[0]["read"], 32 * 60)
                self.assertEqual(seen[0]["write"], 32 * 60)

    async def test_share_generates_one_key_and_reuses_it_on_a_retry(self) -> None:
        keys: list[str] = []
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["idempotency-key"])
            attempts["n"] += 1
            if attempts["n"] == 1:
                # still streaming an earlier send of this key
                return httpx.Response(
                    409,
                    headers={"Retry-After": "7"},
                    json={"errorMessage": "still uploading", "errorSeverity": 5},
                )
            return httpx.Response(200, json=self._result(replayed=True))

        client, slept = self._clock_client(handler)
        result = await client.share_transcript(
            self.JOB, data=self.DATA, file_name="run.eval"
        )
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])  # same key reused
        self.assertEqual(slept, [7.0])  # honoured Retry-After
        self.assertIs(result.replayed, True)

    async def test_share_retries_a_503_on_backoff_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] <= 2:
                return httpx.Response(
                    503, json={"errorMessage": "storage down", "errorSeverity": 5}
                )
            return httpx.Response(200, json=self._result())

        client, slept = self._clock_client(handler)
        result = await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(attempts["n"], 3)
        self.assertEqual(slept, [2.0, 4.0])  # exponential backoff
        self.assertEqual(result.transcript.job_id, self.JOB)

    async def test_share_honours_retry_after_on_a_503(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(
                    503,
                    headers={"Retry-After": "11"},
                    json={"errorMessage": "storage down", "errorSeverity": 5},
                )
            return httpx.Response(200, json=self._result())

        client, slept = self._clock_client(handler)
        await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(slept, [11.0])

    async def test_share_retries_a_rate_limit_and_a_gateway_timeout(self) -> None:
        answers = [
            httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"errorMessage": "slow down", "errorSeverity": 5},
            ),
            httpx.Response(504, text="Gateway Time-out"),
            httpx.Response(200, json=self._result()),
        ]
        keys: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["idempotency-key"])
            return answers.pop(0)

        client, slept = self._clock_client(handler)
        await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(slept, [3.0, 2.0])
        self.assertEqual(len(set(keys)), 1)

    async def test_share_retries_a_lost_connection_with_the_same_key(self) -> None:
        keys: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            keys.append(request.headers["idempotency-key"])
            if len(keys) == 1:
                raise httpx.ReadError("connection reset", request=request)
            return httpx.Response(200, json=self._result(replayed=True))

        client, slept = self._clock_client(handler)
        result = await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(slept, [2.0])
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])
        self.assertIs(result.replayed, True)

    async def test_share_gives_up_on_a_connection_that_stays_lost(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route", request=request)

        client, slept = self._clock_client(handler)
        with self.assertRaises(httpx.ConnectError):
            await client.share_transcript(self.JOB, data=self.DATA)
        self.assertLessEqual(sum(slept), 5 * 60 + 30)

    async def test_share_does_not_retry_a_400(self) -> None:
        # The hub asks for a new key after an integrity mismatch and its body has
        # no code to tell that from a bad request, so a 400 is final.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(
                400, json={"errorMessage": "not the file", "errorSeverity": 5}
            )

        client, slept = self._clock_client(handler)
        with self.assertRaises(RogueArenaApiError) as ctx:
            await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(attempts["n"], 1)
        self.assertEqual(slept, [])

    async def test_share_does_not_retry_a_409_without_retry_after(self) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(
                409,
                json={"errorMessage": "a different transcript", "errorSeverity": 5},
            )

        client, slept = self._clock_client(handler)
        with self.assertRaises(RogueArenaApiError) as ctx:
            await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(attempts["n"], 1)
        self.assertEqual(slept, [])

    async def test_share_does_not_retry_a_413_or_404(self) -> None:
        for status in (413, 404, 401):
            with self.subTest(status=status):
                attempts = {"n": 0}

                def handler(
                    request: httpx.Request,
                    status: int = status,
                    attempts: dict[str, int] = attempts,
                ) -> httpx.Response:
                    attempts["n"] += 1
                    return httpx.Response(
                        status, json={"errorMessage": "no", "errorSeverity": 5}
                    )

                client, _ = self._clock_client(handler)
                with self.assertRaises(RogueArenaApiError) as ctx:
                    await client.share_transcript(self.JOB, data=self.DATA)
                self.assertEqual(ctx.exception.status_code, status)
                self.assertEqual(attempts["n"], 1)

    async def test_share_refuses_empty_or_oversize_before_any_request(self) -> None:
        sent = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            sent["n"] += 1
            return httpx.Response(200, json=self._result())

        client, _ = self._clock_client(handler)
        with self.assertRaisesRegex(ValueError, "empty"):
            await client.share_transcript(self.JOB, data=b"")
        from inspect_rogue_arena.client import EVAL_TRANSCRIPT_MAX_BYTES

        with self.assertRaisesRegex(ValueError, "cap"):
            await client.share_transcript(
                self.JOB, data=b"x" * (EVAL_TRANSCRIPT_MAX_BYTES + 1)
            )
        self.assertEqual(sent["n"], 0)

    async def test_the_cap_is_the_100_mb_rogue_arena_accepts(self) -> None:
        # Rogue Arena accepts a body of at most 100 MB (decimal), so a file between
        # that and 100 MiB would pass a higher check and then get a 413.
        cap = 100_000_000
        sent: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(len(request.content))
            return httpx.Response(200, json=self._result())

        client, _ = self._clock_client(handler)
        with self.assertRaisesRegex(ValueError, "cap"):
            await client.share_transcript(self.JOB, data=b"x" * (cap + 1))
        self.assertEqual(sent, [])
        await client.share_transcript(self.JOB, data=b"x" * cap)
        self.assertEqual(sent, [cap])

    async def test_share_gives_up_at_the_deadline(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                503, json={"errorMessage": "storage down", "errorSeverity": 5}
            )

        client, slept = self._clock_client(handler)
        with self.assertRaises(RogueArenaApiError) as ctx:
            await client.share_transcript(self.JOB, data=self.DATA)
        self.assertEqual(ctx.exception.status_code, 503)
        # it stopped within the 5-minute retry deadline
        self.assertLessEqual(sum(slept), 5 * 60 + 30)

    def test_sanitize_file_name_matches_the_hubs_rule(self) -> None:
        self.assertEqual(sanitize_transcript_file_name("my run!.eval"), "my-run-.eval")
        self.assertEqual(sanitize_transcript_file_name("/a/b/c.eval"), "c.eval")
        self.assertEqual(sanitize_transcript_file_name("report"), "report.eval")
        # the hub requires a leading letter or digit
        self.assertEqual(sanitize_transcript_file_name("_run.eval"), "run.eval")
        self.assertEqual(sanitize_transcript_file_name("+.-run.eval"), "run.eval")
        # Inspect's own names pass through unchanged
        inspect_name = "2000-01-02T03-04-05+00-00_my-task_AbCdEf12.eval"
        self.assertEqual(sanitize_transcript_file_name(inspect_name), inspect_name)
        self.assertIsNone(sanitize_transcript_file_name(None))
        self.assertIsNone(sanitize_transcript_file_name("///"))
        long_name = "x" * 300 + ".eval"
        sanitized = sanitize_transcript_file_name(long_name)
        assert sanitized is not None
        self.assertEqual(len(sanitized), 255)
        self.assertTrue(sanitized.endswith(".eval"))
        # a cut that leaves a bad first character is trimmed to a good one
        trimmed = sanitize_transcript_file_name(
            "x" * 300 + "_" * 5 + "y" * 245 + ".eval"
        )
        self.assertEqual(trimmed, "y" * 245 + ".eval")
        # nothing usable left: the optional header is simply omitted
        self.assertIsNone(
            sanitize_transcript_file_name("x" * 300 + "_" * 251 + ".eval")
        )


class RogueArenaTerminalReasonTests(unittest.TestCase):
    def test_a_team_member_terminal_reason_parses(self) -> None:
        # Rogue Arena can report this reason; parsing must accept it.
        self.assertIs(
            EvalTrial.from_dict(
                _trial_json("closed", outcome="closed", terminalReason="team_member")
            ).terminal_reason,
            EvalTrialTerminalReason.TEAM_MEMBER,
        )


class RogueArenaExecTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_exec_below_the_ceiling_sends_no_background(self) -> None:
        seen: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "running": False,
                    "execId": None,
                    "exitCode": 0,
                    "stdout": "root\n",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
        )
        result = await client.run_command(
            "trial-1",
            cmd=["whoami"],
            cwd=None,
            env=None,
            stdin=None,
            timeout_seconds=30,
            sync_ceiling_seconds=50,
            poll_interval_seconds=1,
            max_total_seconds=14400,
        )
        self.assertEqual(result.stdout, "root\n")
        self.assertFalse(seen[0]["background"])
        self.assertNotIn("pollExecId", seen[0])

    async def test_threshold_is_inclusive_sync_at_ceiling_background_above(
        self,
    ) -> None:
        backgrounds: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            backgrounds.append(body["background"])
            return httpx.Response(
                200,
                json={
                    "running": False,
                    "execId": "e1",
                    "exitCode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        for timeout in (50, 51):
            await client.run_command(
                "trial-1",
                cmd=["x"],
                cwd=None,
                env=None,
                stdin=None,
                timeout_seconds=timeout,
                sync_ceiling_seconds=50,
                poll_interval_seconds=1,
                max_total_seconds=14400,
            )
        # 50 -> sync; 51 -> background launch (returns finished) + one poll
        self.assertEqual(backgrounds, [False, True, False])

    async def test_long_command_goes_background_and_polls_pollexecid(self) -> None:
        calls: list[dict[str, object]] = []
        polls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            body = json.loads(request.content)
            calls.append(body)
            if body.get("pollExecId"):
                polls += 1
                if polls == 1:
                    return httpx.Response(
                        200,
                        json={
                            "running": True,
                            "execId": body["pollExecId"],
                            "exitCode": None,
                            "stdout": "",
                            "stderr": "",
                            "timedOut": False,
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "running": False,
                        "execId": body["pollExecId"],
                        "exitCode": 0,
                        "stdout": "done\n",
                        "stderr": "",
                        "timedOut": False,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "55555555-5555-4555-8555-555555555555",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        result = await client.run_command(
            "trial-1",
            cmd=["bash", "-lc", "sleep 600"],
            cwd=None,
            env=None,
            stdin=None,
            timeout_seconds=600,
            sync_ceiling_seconds=50,
            poll_interval_seconds=1,
            max_total_seconds=14400,
        )
        self.assertEqual(result.stdout, "done\n")
        self.assertEqual(polls, 2)
        self.assertTrue(calls[0]["background"])
        self.assertNotIn("pollExecId", calls[0])
        self.assertTrue(all(len(c["cmd"]) >= 1 for c in calls if c.get("pollExecId")))
        self.assertTrue(
            all(
                c["pollExecId"] == "55555555-5555-4555-8555-555555555555"
                for c in calls[1:]
            )
        )

    async def test_background_poll_stops_at_the_four_hour_ceiling(self) -> None:
        now = 0.0
        launched_timeouts: list[object] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if not body.get("pollExecId"):
                launched_timeouts.append(body["timeoutSeconds"])
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "e1",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        async def advance(seconds: float) -> None:
            nonlocal now
            now += seconds

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=advance,
            monotonic=lambda: now,
        )
        with self.assertRaises(TimeoutError):
            await client.run_command(
                "trial-1",
                cmd=["x"],
                cwd=None,
                env=None,
                stdin=None,
                timeout_seconds=99999,
                sync_ceiling_seconds=50,
                poll_interval_seconds=60,
                max_total_seconds=14400,
            )
        self.assertEqual(launched_timeouts, [14400])
        self.assertGreaterEqual(now, 14400)
        self.assertLess(now, 14400 + 60)

    async def test_lost_exec_is_returned_as_finished_exit_255(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body.get("pollExecId"):
                return httpx.Response(
                    200,
                    json={
                        "running": False,
                        "execId": body["pollExecId"],
                        "exitCode": 255,
                        "stdout": "",
                        "stderr": "exec lost: no exit status was recorded within the exec max age",
                        "timedOut": False,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "e1",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=_no_sleep,
        )
        result = await client.run_command(
            "trial-1",
            cmd=["x"],
            cwd=None,
            env=None,
            stdin=None,
            timeout_seconds=600,
            sync_ceiling_seconds=50,
            poll_interval_seconds=1,
            max_total_seconds=14400,
        )
        self.assertFalse(result.running)
        self.assertEqual(result.exit_code, 255)
        self.assertTrue(result.lost)

    async def test_upload_and_download_round_trip_bytes(self) -> None:
        stored: dict[str, str] = {}
        payload = b"\x00\x01\x02rogue\xff\xfe\x80"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/upload"):
                body = json.loads(request.content)
                stored[body["path"]] = body["contentBase64"]
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(
                200,
                json={
                    "path": request.url.params["path"],
                    "contentBase64": stored[request.url.params["path"]],
                },
            )

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(UnicodeDecodeError):
            payload.decode("utf-8")
        await client.upload_file("trial-1", "/tmp/x", payload)
        self.assertEqual(await client.download_file("trial-1", "/tmp/x"), payload)

    async def test_real_exit_255_is_not_lost(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "running": False,
                    "execId": None,
                    "exitCode": 255,
                    "stdout": "",
                    "stderr": "boom",
                    "timedOut": False,
                },
            )

        result = await _exec_client(handler).run_command(
            "trial-1", **_EXEC_ARGS, timeout_seconds=10
        )
        self.assertEqual(result.exit_code, 255)
        self.assertFalse(result.lost)

    async def test_sync_504_502_and_transport_error_raise_outcome_unknown(
        self,
    ) -> None:
        for failure in (504, 502, "transport"):
            count = 0

            def handler(
                request: httpx.Request, failure: int | str = failure
            ) -> httpx.Response:
                nonlocal count
                count += 1
                if failure == "transport":
                    raise httpx.ReadTimeout("slow", request=request)
                return httpx.Response(failure, text="gateway")

            with self.assertRaises(ExecOutcomeUnknown) as ctx:
                await _exec_client(handler).run_command(
                    "trial-1", **_EXEC_ARGS, timeout_seconds=10
                )
            self.assertEqual(count, 1)
            self.assertEqual(ctx.exception.cmd, ("whoami",))
            self.assertIn("may still be running", str(ctx.exception))

    async def test_background_launch_502_raises_outcome_unknown_once(self) -> None:
        count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal count
            count += 1
            return httpx.Response(502, text="bad gateway")

        with self.assertRaises(ExecOutcomeUnknown) as ctx:
            await _exec_client(handler).run_command(
                "trial-1", **_EXEC_ARGS, timeout_seconds=600
            )
        self.assertEqual(count, 1)
        self.assertIsNone(ctx.exception.exec_id)

    async def test_poll_retries_a_transient_failure_then_finishes(self) -> None:
        polls = 0
        sleeps: list[float] = []
        running = {
            "running": True,
            "execId": "e1",
            "exitCode": None,
            "stdout": "",
            "stderr": "",
            "timedOut": False,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            body = json.loads(request.content)
            if not body.get("pollExecId"):
                return httpx.Response(200, json=running)
            polls += 1
            if polls == 1:
                return httpx.Response(504, text="timeout")
            if polls == 2:
                raise httpx.ConnectError("reset", request=request)
            if polls == 3:
                return httpx.Response(200, json=running)  # resets the backoff
            if polls == 4:
                return httpx.Response(503, text="busy")
            return httpx.Response(
                200,
                json={
                    "running": False,
                    "execId": "e1",
                    "exitCode": 0,
                    "stdout": "ok",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        async def record(seconds: float) -> None:
            sleeps.append(seconds)

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=record,
        )
        result = await client.run_command("trial-1", **_EXEC_ARGS, timeout_seconds=600)
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(polls, 5)
        # the first poll comes quickly (0.25 s), the running cadence doubles toward
        # poll_interval_seconds; consecutive failures double the wait from
        # poll_interval_seconds, and a successful poll resets that backoff
        self.assertEqual(sleeps, [0.25, 2, 4, 0.5, 2])

    async def test_poll_backoff_is_capped(self) -> None:
        sleeps: list[float] = []
        now = 0.0

        def handler(request: httpx.Request) -> httpx.Response:
            if json.loads(request.content).get("pollExecId"):
                return httpx.Response(504, text="timeout")
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "e1",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        async def advance(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        client = RogueArenaClient(
            "https://hub.example.test",
            "ra_eval_T",
            transport=httpx.MockTransport(handler),
            sleep=advance,
            monotonic=lambda: now,
        )
        with self.assertRaises(TimeoutError) as ctx:
            await client.run_command("trial-1", **_EXEC_ARGS, timeout_seconds=600)
        # the ceiling keeps the last transient cause
        cause = ctx.exception.__cause__
        assert isinstance(cause, RogueArenaApiError)
        self.assertEqual(cause.status_code, 504)
        self.assertEqual(sleeps[:6], [0.25, 2, 4, 8, 16, 30])
        self.assertEqual(max(sleeps), 30)
        self.assertGreaterEqual(now, 900)

    async def test_poll_non_transient_error_aborts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body.get("pollExecId"):
                return httpx.Response(404, json={"error": "gone"})
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "e1",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        with self.assertRaises(RogueArenaApiError):
            await _exec_client(handler).run_command(
                "trial-1", **_EXEC_ARGS, timeout_seconds=600
            )

    async def test_sync_running_true_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "running": True,
                    "execId": "e1",
                    "exitCode": None,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        with self.assertRaises(RuntimeError):
            await _exec_client(handler).run_command(
                "trial-1", **_EXEC_ARGS, timeout_seconds=10
            )

    async def test_sync_http_timeout_is_bounded_by_the_gateway_limit(self) -> None:
        seen: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.extensions["timeout"]["read"])
            return httpx.Response(
                200,
                json={
                    "running": False,
                    "execId": None,
                    "exitCode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timedOut": False,
                },
            )

        client = _exec_client(handler)
        await client.run_command("trial-1", **_EXEC_ARGS, timeout_seconds=10)
        await client.run_command("trial-1", **_EXEC_ARGS, timeout_seconds=50)
        self.assertEqual(seen, [15, 55])

    async def test_upload_body_is_base64_of_the_raw_bytes(self) -> None:
        bodies: list[dict[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True})

        await _exec_client(handler).upload_file("trial-1", "/tmp/b", b"\xff\xfe\x00")
        self.assertEqual(bodies[0], {"path": "/tmp/b", "contentBase64": "//4A"})

    async def test_download_bad_base64_raises_clear_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"path": "/x", "contentBase64": "@@@"})

        with self.assertRaisesRegex(RuntimeError, "invalid base64"):
            await _exec_client(handler).download_file("trial-1", "/x")


if __name__ == "__main__":
    unittest.main()
