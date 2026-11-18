"""``rogue-arena-inspect``: identity, launch visibility, run status, and launch.

Commands:

- ``whoami``: the token's owner and organization.
- ``evals``: the evals this token can run, with per-eval launch visibility
  (``ready now (N warm)`` / ``on demand, ~X min`` / ``unknown``), name,
  version and stage count. ``--json`` prints the raw hub listing.
- ``runs``: this token's recent runs (jobs). Degrades to a clear note when
  the token-scoped list endpoint is not available.
- ``run ID``: one run's status, score and per-stage result (``ID`` is a job
  id, or a trial id as a fallback), with a human gloss on the hub terminal
  reason. ``--watch`` live-tails the run's trial events until it settles;
  ``--chat`` prints the run's team chatroom transcript.
- ``eval ID``: the eval's graders, solution writeup and recommended prompts
  (``--guide``); ``--prompt-level easy|medium|hard`` prints only that prompt.
  The graders and solution are for the person running the eval, never the
  model under test.
- ``share-transcript FILE``: upload an existing ``.eval`` to its run, e.g. after
  ``launch --share-transcript`` could not.
- ``close ID``: close a ``--keep-range`` run's kept range (``ID`` is a run id,
  or a trial id): Rogue Arena removes every range the run kept up. Closing twice
  is fine.
- ``launch``: a thin wrapper around ``inspect eval`` that passes the task
  path relative to the working directory (``inspect eval`` globs it from there,
  and Python 3.14's pathlib rejects absolute glob patterns) and, with
  ``--team FILE``, runs a multi-agent team task. ``--max-runtime SECONDS`` sets
  the per-trial wall clock (``ROGUE_ARENA_MAX_RUNTIME_SECONDS``); ``--report
  both`` also reports the sequential and independent hop counts
  (``ROGUE_ARENA_REPORT``); ``--parallel N`` runs N trials at once;
  ``--share-transcript`` uploads the finished run's .eval to Rogue Arena;
  ``--prompt-level`` gives the model the eval author's recommended prompt;
  ``--keep-range`` keeps each scored trial's range up until ``close``.

The API commands use a personal ``ra_eval_`` token from ``ROGUE_ARENA_TOKEN``
(or ``--token``) against ``ROGUE_ARENA_BASE_URL`` (or ``--base-url``). The
token is never printed and never passed on a child process's argv.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import httpx
from inspect_ai.log import read_eval_log

from inspect_rogue_arena.client import (
    EvalCatalogEntry,
    EvalChatMessage,
    EvalGuide,
    EvalGuideHop,
    EvalGuideNotOffered,
    EvalJob,
    EvalJobStatus,
    EvalMachine,
    EvalPromptLevel,
    EvalProofType,
    EvalScore,
    EvalTranscriptUpload,
    EvalTrial,
    EvalTrialEvent,
    EvalTrialTerminalReason,
    RogueArenaApiError,
    RogueArenaClient,
    checked_base_url,
    sanitize_transcript_file_name,
)
from inspect_rogue_arena.config import (
    KEEP_RANGE_ENV,
    PROMPT_ENV,
    RogueArenaSandboxConfig,
)
from inspect_rogue_arena.preseed import PRESEED_ENV, load_preseed
from inspect_rogue_arena.sandbox import select_eval
from inspect_rogue_arena.scorer import REPORT_ENV, HopReport

_PROG = "rogue-arena-inspect"
_REDACTED = "[redacted]"
# Statuses that mean a token-scoped endpoint is not available on this server.
_ABSENT_STATUSES = (404, 405, 501)
# Job statuses that mean the run has settled, so `run --watch` stops polling.
_SETTLED_JOB_STATUSES = (EvalJobStatus.CANCELLED, EvalJobStatus.COMPLETE)
# `run --watch` poll cadence. The floor keeps a tail from hammering the hub.
_WATCH_INTERVAL_SECONDS = 3.0
_WATCH_MIN_INTERVAL_SECONDS = 1.0
# Patchable so tests drive the cadence without real waits.
_watch_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def _base_url(args: argparse.Namespace) -> str:
    return (args.base_url or os.environ.get("ROGUE_ARENA_BASE_URL", "")).strip()


def _token(args: argparse.Namespace) -> str:
    return (args.token or os.environ.get("ROGUE_ARENA_TOKEN", "")).strip()


def _require_env(args: argparse.Namespace) -> tuple[str, str]:
    base_url, token = _base_url(args), _token(args)
    if not base_url:
        raise RuntimeError("set ROGUE_ARENA_BASE_URL (or pass --base-url)")
    # Refused here too, so `launch` never hands an http URL to its child.
    base_url = checked_base_url(base_url)
    if not token:
        raise RuntimeError(
            "set ROGUE_ARENA_TOKEN (or pass --token) to a personal ra_eval_ token"
        )
    return base_url, token


def _client(args: argparse.Namespace) -> RogueArenaClient:
    base_url, token = _require_env(args)
    return RogueArenaClient(base_url, token)


# --- rendering --------------------------------------------------------------

# A short human gloss for each hub terminal reason, so `run <id>` reads clearer
# than a bare enum. Rogue Arena only knows wall-clock `timeout`; an Inspect
# message- or token-limit stop is not a Rogue Arena reason and shows only in
# the local `.eval` log.
_TERMINAL_REASON_GLOSS: Mapping[EvalTrialTerminalReason, str] = {
    EvalTrialTerminalReason.AGENT_SUCCESS: "the agent signaled success",
    EvalTrialTerminalReason.AGENT_FAIL: "the agent stopped without finishing",
    EvalTrialTerminalReason.TIMEOUT: "ran out of wall clock",
    EvalTrialTerminalReason.INFRA_ERROR: "an infrastructure failure, not scored",
    EvalTrialTerminalReason.CANCELLED: "the run was cancelled",
    EvalTrialTerminalReason.HALTED: "Rogue Arena halted the run",
    EvalTrialTerminalReason.TEAM_MEMBER: "ended as part of a team run",
    EvalTrialTerminalReason.PRESEED_FAILED: "a preseed step failed before the model started, not scored",
}


def _reason_cell(reason: EvalTrialTerminalReason | None) -> str:
    """``<reason> (<gloss>)`` for a trial/member line, or a dash placeholder when unset."""
    if reason is None:
        return "-"
    return f"{reason.value} ({_TERMINAL_REASON_GLOSS[reason]})"


def _render_evals(entries: Sequence[EvalCatalogEntry]) -> str:
    if not entries:
        return "no evals are available for this token"
    lines: list[str] = []
    for entry in entries:
        stages = f"{entry.hop_count} stage" + ("" if entry.hop_count == 1 else "s")
        lines.append(
            f"{entry.name or '(unnamed)'}  |  v{entry.canvas_version_number} · "
            f"{stages} · {entry.scoring_mode.value} · {entry.readiness_label()}"
        )
        lines.append(f"    id: {entry.environment_id}")
    return "\n".join(lines)


def _render_machines(machines: Sequence[EvalMachine]) -> str:
    """One line per machine a preseed may target: name, OS and stable id, with marks."""
    if not machines:
        return "this eval has no machines a preseed can target"
    width = max(len(machine.name) for machine in machines)
    os_width = max(len(machine.operating_system) for machine in machines)
    lines = []
    for machine in machines:
        note = "  (agent: the model's machine)" if machine.agent else ""
        lines.append(
            f"{machine.name.ljust(width)}  {machine.operating_system.ljust(os_width)}  "
            f"id {machine.machine_id}{note}"
        )
    lines.append(
        "(base machines every range carries, such as the router, VPN gateway and "
        "proxy, and machines hidden from you are left out; target a machine by "
        "its name or id)"
    )
    return "\n".join(lines)


def _render_score(score: EvalScore) -> list[str]:
    rows = [
        (
            f"    score: {score.hops_passed}/{score.total_steps} hops "
            f"(reached step {score.reached_step}, {score.scoring_mode.value})"
        )
    ]
    for step in score.steps:
        mark = "pass" if step.passed else "fail"
        at = f" at {step.passed_at}" if step.passed_at else ""
        rows.append(f"      #{step.step_number} {step.name or '(unnamed)'}: {mark}{at}")
    return rows


def _render_guide_hop(hop: EvalGuideHop) -> list[str]:
    rows = [
        f"  #{hop.step_number} {hop.name or '(unnamed)'}  on {hop.machine_nickname}"
    ]
    if hop.description:
        rows.append(f"      {hop.description}")
    if hop.proof_type is EvalProofType.KEY_FILE:
        rows.append(f"      proof: key file at {hop.key_filepath}")
    elif hop.check_script is not None:
        rows.append(f"      proof: check script, output must match /{hop.regex}/")
    else:
        rows.append(f"      proof: `{hop.command}`, output must match /{hop.regex}/")
    return rows


def _render_guide(guide: EvalGuide) -> str:
    header = (
        f"{guide.name or '(unnamed)'}  |  v{guide.canvas_version_number} · "
        f"{len(guide.hops)} stage(s) · {guide.scoring_mode.value}"
    )
    graders_note = (
        "Graders (for you, the person running the eval; never give these to the "
        "model under test):"
    )
    rows = [header, f"    id: {guide.environment_id}", "", graders_note]
    for hop in guide.hops:
        rows += _render_guide_hop(hop)
    rows += ["", "Solution writeup:"]
    rows.append(
        _indent(guide.solution_writeup)
        if guide.solution_writeup
        else "  (none written)"
    )
    rows += ["", "Recommended prompts (written for the model under test):"]
    for level in EvalPromptLevel:
        text = guide.recommended_prompts[level].strip()
        rows.append(f"  {level.value}:")
        rows.append(_indent(text) if text else "    (none written)")
    return "\n".join(rows)


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _render_trial(trial: EvalTrial) -> list[str]:
    outcome = trial.outcome.value if trial.outcome else "-"
    preseeded = "  preseeded" if trial.preseeded else ""
    rows = [
        (
            f"  trial {trial.id}  {trial.status.value}  outcome={outcome} "
            f"reason={_reason_cell(trial.terminal_reason)}{preseeded}"
        )
    ]
    if trial.score is not None:
        rows += _render_score(trial.score)
    elif trial.hops_passed is not None:
        rows.append(
            f"    progress: {trial.hops_passed}/{trial.total_steps} hops "
            "(final score on close)"
        )
    else:
        rows.append("    no score yet (Rogue Arena scores a trial when it closes)")
    return rows


def _range_cell(job: EvalJob) -> str:
    """``  range kept open`` / ``  range closed`` for a keep-range run; nothing otherwise."""
    if not job.keep_range:
        return ""
    if job.range_closed_at is not None:
        return "  range closed"
    if job.kept_range_count == 0:
        return "  no kept range up"
    if job.kept_range_count is None:
        return "  range kept open"
    return f"  range kept open ({job.kept_range_count} up)"


def _preseed_cell(job: EvalJob) -> str:
    """``  preseeded`` for a run whose trials were claimed with a preseed; nothing otherwise."""
    return "  preseeded" if job.preseeded else ""


def _render_run(job: EvalJob, trials: Sequence[EvalTrial]) -> str:
    header = (
        f"run {job.id}  {job.status.value}  env {job.environment_id}{_model_cell(job)}  "
        f"n={job.n} concurrency={job.concurrency}  "
        f"trials {job.trial_count} (closed {job.closed_trial_count}){_range_cell(job)}"
        f"{_preseed_cell(job)}"
    )
    rows = [header]
    if job.range_kept_open:
        rows.append(
            f"  its scored trials' ranges stay up (billed like any lab) until you "
            f"run `{_PROG} close {job.id}`"
        )
    for trial in trials:
        rows += _render_trial(trial)
    if not trials:
        rows.append("  no trials claimed yet")
    return "\n".join(rows)


def _model_cell(job: EvalJob) -> str:
    """``  model <name>`` for a run line, or nothing when the run has no label."""
    return f"  model {job.model}" if job.model else ""


def _render_runs(jobs: Sequence[EvalJob]) -> str:
    if not jobs:
        return "this token has no runs yet"
    return "\n".join(
        f"{job.id}  {job.status.value}  env {job.environment_id}{_model_cell(job)}  "
        f"n={job.n} concurrency={job.concurrency}  "
        f"trials {job.trial_count} (closed {job.closed_trial_count}){_range_cell(job)}"
        f"{_preseed_cell(job)}"
        for job in jobs
    )


# --- async API commands -----------------------------------------------------


async def _cmd_whoami(args: argparse.Namespace) -> str:
    who = await _client(args).whoami()
    return json.dumps(
        {
            "userMainId": who.user_main_id,
            "organizationId": who.organization_id,
            "tokenLabel": who.token_label,
            "tokenLast4": who.token_last4,
        },
        indent=2,
        sort_keys=True,
    )


async def _cmd_evals(args: argparse.Namespace) -> str:
    if args.dump_vms is not None:
        client = _client(args)
        machines = await client.list_eval_machines(
            await _environment_id(client, args.dump_vms)
        )
        if args.json:
            return json.dumps(
                {
                    "items": [
                        {
                            "machineId": machine.machine_id,
                            "name": machine.name,
                            "operatingSystem": machine.operating_system,
                            "agent": machine.agent,
                        }
                        for machine in machines
                    ]
                },
                indent=2,
                sort_keys=True,
            )
        return _render_machines(machines)
    entries = await _client(args).list_evals()
    if args.json:
        return json.dumps(
            {"items": [dict(entry.raw) for entry in entries]}, indent=2, sort_keys=True
        )
    return _render_evals(entries)


async def _cmd_runs(args: argparse.Namespace) -> str:
    client = _client(args)
    try:
        jobs = await client.list_runs()
    except RogueArenaApiError as error:
        if error.status_code in _ABSENT_STATUSES:
            raise RuntimeError(
                "Rogue Arena has no token-scoped run-list endpoint "
                f"(GET /hub/eval/jobs returned {error.status_code}); look a run up "
                "directly with `run <id>` using a job id from your eval run"
            ) from error
        raise
    if args.json:
        return json.dumps(
            {"items": [dict(job.raw) for job in jobs]}, indent=2, sort_keys=True
        )
    return _render_runs(jobs)


def _format_chat_line(message: EvalChatMessage) -> str:
    return f"#{message.seq} [{message.at}] {message.author}: {message.content}"


def _format_event_line(trial_id: str, event: EvalTrialEvent) -> str:
    detail = f": {event.detail}" if event.detail else ""
    return f"  [{event.at}] {trial_id} {event.kind.value}: {event.summary}{detail}"


async def _job_for_id(client: RogueArenaClient, run_id: str) -> EvalJob:
    """The run (job) for a job id, or for a trial id via its ``jobId``."""
    try:
        return await client.get_job(run_id)
    except RogueArenaApiError as error:
        if error.status_code != 404:
            raise
    try:
        trial = await client.get_trial(run_id)
    except RogueArenaApiError as trial_error:
        if trial_error.status_code == 404:
            raise RuntimeError(
                f"no run or trial found with id {run_id!r} for this token"
            ) from trial_error
        raise
    return await client.get_job(trial.job_id)


async def _cmd_run_chat(args: argparse.Namespace) -> str:
    """Print the run's team chatroom transcript (the hub's per-job room)."""
    client = _client(args)
    job = await _job_for_id(client, args.id)
    messages = await client.list_chat_messages(job.id)
    if args.json:
        return json.dumps(
            {
                "jobId": job.id,
                "messages": [
                    {
                        "id": m.id,
                        "jobId": m.job_id,
                        "seq": m.seq,
                        "at": m.at,
                        "author": m.author,
                        "content": m.content,
                    }
                    for m in messages
                ],
            },
            indent=2,
            sort_keys=True,
        )
    if not messages:
        return f"run {job.id} has no team chatroom messages"
    header = f"run {job.id} team chatroom ({len(messages)} message(s)):"
    return "\n".join([header, *(_format_chat_line(m) for m in messages)])


async def _watch_events(
    client: RogueArenaClient, trials: Sequence[EvalTrial], seen: set[str]
) -> list[str]:
    """New per-trial event lines since the last poll; tolerate a hub without events."""
    lines: list[str] = []
    for trial in trials:
        try:
            events = await client.list_trial_events(trial.id)
        except RogueArenaApiError as error:
            if error.status_code in _ABSENT_STATUSES:
                return lines  # this hub serves no per-trial events; status-only tail
            raise
        for event in events:
            if event.id not in seen:
                seen.add(event.id)
                lines.append(_format_event_line(trial.id, event))
    return lines


async def _cmd_run_watch(args: argparse.Namespace) -> str:
    """Live-tail a run's trial events and status until the job settles.

    Polls ``GET /hub/eval/jobs/:id`` and each trial's events on an interval,
    printing new events as they arrive, and returns the final run render once
    the job is cancelled or complete. A team run is one trial, so this
    follows that trial like any other.
    """
    client = _client(args)
    job = await _job_for_id(client, args.id)
    interval = max(args.interval, _WATCH_MIN_INTERVAL_SECONDS)
    seen: set[str] = set()
    print(f"watching run {job.id} (polling every {interval:g}s)…", file=sys.stderr)
    while True:
        trials = await client.list_job_trials(job.id)
        for line in await _watch_events(client, trials, seen):
            print(line, flush=True)
        if job.status in _SETTLED_JOB_STATUSES:
            return _render_run(job, trials)
        await _watch_sleep(interval)
        job = await client.get_job(job.id)


async def _environment_id(client: RogueArenaClient, selector: str) -> str:
    """The environment id for an eval's id or unique name (the guide route takes an id)."""
    try:
        uuid.UUID(selector)
    except ValueError:
        config = RogueArenaSandboxConfig(eval=selector)
        return select_eval(await client.list_evals(), config).environment_id
    return selector


async def _guide(args: argparse.Namespace, selector: str) -> EvalGuide:
    client = _client(args)
    return await client.get_eval_guide(await _environment_id(client, selector))


async def _cmd_eval(args: argparse.Namespace) -> str:
    guide = await _guide(args, args.id)
    if args.prompt_level is not None:
        # Just the prompt, so it pipes straight into a task.
        return guide.prompt(EvalPromptLevel(args.prompt_level))
    if args.json:
        return json.dumps(dict(guide.raw), indent=2, sort_keys=True)
    return _render_guide(guide)


async def _cmd_share_transcript(args: argparse.Namespace) -> str:
    """Upload an existing ``.eval``, e.g. after ``launch``'s own upload failed."""
    path = Path(args.file).expanduser()
    if not path.is_file():
        raise RuntimeError(f"transcript file not found: {path}")
    client = _client(args)
    # In a thread: Inspect's log reader runs its own event loop.
    facts = await asyncio.to_thread(_log_facts, path)
    if args.run is not None:
        job = await client.get_job(args.run)
        if not _job_accepts_log(job, facts):
            raise RuntimeError(
                f"{path.name} belongs to a different run than {job.id}; drop --run to "
                "use its own"
            )
    elif facts.run_id is not None:
        jobs = await _jobs_of_run(client, facts.run_id)
        found = _job_for_log(facts, jobs)
        if found is None:
            raise RuntimeError(
                f"can't tell which of this run's jobs {path.name} belongs to: name it "
                f"with --run <id> (from `{_PROG} runs`). Its jobs with trials: "
                f"{_describe_jobs(jobs)}"
            )
        job = found
    else:
        raise RuntimeError(
            f"{path.name} does not say which run it belongs to (it was not launched "
            "with --share-transcript): name it with --run <id> from "
            f"`{_PROG} runs`"
        )
    key = str(uuid.uuid4()) if args.new_upload else None
    return _render_transcript_upload(await _share_file(client, job.id, path, key))


async def _cmd_run(args: argparse.Namespace) -> str:
    if args.chat:
        return await _cmd_run_chat(args)
    if args.watch:
        return await _cmd_run_watch(args)
    client = _client(args)
    try:
        job = await client.get_job(args.id)
    except RogueArenaApiError as error:
        if error.status_code != 404:
            raise
        # Not a job id: a trial id is the other thing a user has in hand.
        try:
            trial = await client.get_trial(args.id)
        except RogueArenaApiError as trial_error:
            if trial_error.status_code == 404:
                raise RuntimeError(
                    f"no run or trial found with id {args.id!r} for this token"
                ) from trial_error
            raise
        if args.json:
            return json.dumps({"trial": dict(trial.raw)}, indent=2, sort_keys=True)
        return "\n".join(_render_trial(trial))
    trials = await client.list_job_trials(job.id)
    if args.json:
        return json.dumps(
            {"run": dict(job.raw), "trials": [dict(trial.raw) for trial in trials]},
            indent=2,
            sort_keys=True,
        )
    return _render_run(job, trials)


async def _cmd_close(args: argparse.Namespace) -> str:
    """Close a keep-range run's kept range; Rogue Arena removes every range it kept."""
    client = _client(args)
    job = await _job_for_id(client, args.id)
    if not job.keep_range:
        if args.json:
            return json.dumps({"run": dict(job.raw)}, indent=2, sort_keys=True)
        return (
            f"run {job.id} was not launched with --keep-range: Rogue Arena removes "
            "each of its ranges when its trial ends, so there is nothing to close"
        )
    already_closed = job.range_closed_at is not None
    closed = await client.close_range(job.id)
    if args.json:
        return json.dumps({"run": dict(closed.raw)}, indent=2, sort_keys=True)
    if already_closed:
        return (
            f"run {closed.id}: its range was already closed at {closed.range_closed_at}"
        )
    if job.kept_range_count == 0:
        return (
            f"run {closed.id}: range closed at {closed.range_closed_at}; it had no kept "
            "range up, and a trial of it that ends later is removed as usual"
        )
    return (
        f"run {closed.id}: range closed at {closed.range_closed_at}; Rogue Arena is "
        "removing every range the run kept up"
    )


_ASYNC_COMMANDS = {
    "whoami": _cmd_whoami,
    "evals": _cmd_evals,
    "runs": _cmd_runs,
    "run": _cmd_run,
    "eval": _cmd_eval,
    "share-transcript": _cmd_share_transcript,
    "close": _cmd_close,
}


# --- launch (a thin `inspect eval` wrapper) ---------------------------------

# Inspect's model-provider docs: which models and keys each provider takes.
_INSPECT_PROVIDERS_URL = "https://inspect.aisi.org.uk/providers.html"

# Signatures of a provider refusal (policy or auth), matched in the child's
# stderr. These are between the runner and the model provider, not the hub, so
# a raw 403/provider stack trace reads as a platform bug when it is not. Kept
# narrow (auth/policy wording, not any failure) to avoid false positives on an
# ordinary agent failure.
_PROVIDER_POLICY_SIGNATURE = re.compile(
    r"(?i)("
    r"\b(401|403)\b.*\b(provider|model|openai|anthropic|google|gemini|openrouter|azure)\b"
    r"|authenticationerror|permissiondeniederror|invalid[_ ]?api[_ ]?key"
    r"|unauthorized|forbidden|permission denied"
    r"|content[_ ]?(policy|management)|moderation|safety policy"
    r"|not (permitted|allowed|available)\b|your region|data policy"
    r"|no (endpoints|allowed providers) found"
    r")"
)


def _provider_policy_hint(child_stderr: str) -> str | None:
    """A clean, actionable message when the child failed on a provider refusal.

    Returns None when nothing in the child's stderr looks like a provider
    policy/auth refusal, so an ordinary run failure is left alone. The message
    never contains a token (it is fixed text).
    """
    if not _PROVIDER_POLICY_SIGNATURE.search(child_stderr):
        return None
    return (
        "the model provider rejected this run under its policy or auth. This is "
        "between your runner and the model provider, not a Rogue Arena error.\n"
        "check your provider API key, and that the model is allowed for your "
        "account and region. Pick a model by its Inspect provider id; see "
        f"{_INSPECT_PROVIDERS_URL}"
    )


def _run_inspect(cmd: list[str], env: Mapping[str, str]) -> tuple[int, str]:
    """Run ``inspect eval``, streaming its stderr live while capturing it.

    stdout is inherited (Inspect's live display is untouched); only stderr is
    teed, so a provider refusal in the child's error output can be classified
    after the run without hiding anything.
    """
    process = subprocess.Popen(
        cmd, env=dict(env), stderr=subprocess.PIPE, text=True, bufsize=1
    )
    captured: list[str] = []
    stream = process.stderr
    assert stream is not None
    for line in stream:
        sys.stderr.write(line)
        sys.stderr.flush()
        captured.append(line)
    return process.wait(), "".join(captured)


def _default_task(team: bool, stack: contextlib.ExitStack) -> Path:
    """The built-in example task file, shipped inside the package.

    Resolved with ``importlib.resources`` so it works from a pip install, not
    only a source checkout. ``stack`` keeps the file on disk for the caller
    (``resources.as_file`` extracts it to a temporary path on a zipped install).
    """
    name = "team_example.py" if team else "eval_example.py"
    resource = resources.files("inspect_rogue_arena").joinpath("examples", name)
    return stack.enter_context(resources.as_file(resource))


def _resolve_task(args: argparse.Namespace, stack: contextlib.ExitStack) -> Path:
    if args.task:
        task = Path(args.task).expanduser().resolve()
        if not task.exists():
            raise RuntimeError(f"task path does not exist: {task}")
        return task
    return _default_task(team=bool(args.team), stack=stack)


def _task_argument(task: Path) -> str:
    """The task path for ``inspect eval``, relative to the working directory.

    ``inspect eval`` globs its task spec from the working directory, and
    Python 3.14's pathlib refuses an absolute glob pattern ("Non-relative
    patterns are unsupported").
    """
    try:
        return os.path.relpath(task)
    except ValueError as error:  # Windows: the task is on another drive
        raise RuntimeError(
            f"run launch from the drive that holds the task file: {task}"
        ) from error


# Inspect's --log-dir default, and the metadata key that tags a run's .eval.
_INSPECT_DEFAULT_LOG_DIR = "./logs"
_RUN_METADATA_KEY = "rogue_arena_run_id"
# An eval log's status when its task ran to the end.
_LOG_FINISHED = "success"
# Filesystems round a file's mtime down; don't miss a log written right at the start.
_LOG_MTIME_SLACK_SECONDS = 2.0
# Enough of the token's newest runs to hold this one, even with others under way.
_JOB_LOOKUP_LIMIT = 200

# The `inspect eval` flags `launch --parallel` sets itself.
_PARALLEL_FLAGS = ("--epochs", "--max-sandboxes", "--max-samples")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {number}")
    return number


def _parallel_args(
    parallel: int, passthrough: Sequence[str], env: dict[str, str]
) -> list[str]:
    """``inspect eval`` args that run ``parallel`` trials at once.

    Each epoch of a sample is its own trial on its own copy of the range, so
    ``--epochs N`` makes N trials of every sample; ``--max-sandboxes`` and
    ``--max-samples`` let Inspect run all N at once, and the job's concurrency
    (``ROGUE_ARENA_CONCURRENCY``, set in ``env``) lets the hub admit them. The
    job's ``max_trials`` must cover them: its default admits 10000.
    """
    clashes = sorted(
        {arg.split("=", 1)[0] for arg in passthrough} & set(_PARALLEL_FLAGS)
    )
    if clashes:
        raise RuntimeError(
            f"--parallel sets {', '.join(_PARALLEL_FLAGS)} itself; drop "
            f"{', '.join(clashes)} from the args after --"
        )
    cap = env.get("ROGUE_ARENA_MAX_TRIALS", "").strip()
    if cap.isdigit() and int(cap) < parallel:
        raise RuntimeError(
            f"--parallel {parallel} runs at least {parallel} trials, but "
            f"ROGUE_ARENA_MAX_TRIALS={cap} lets the job admit only {cap}; raise "
            "it or unset it (the default admits "
            f"{RogueArenaSandboxConfig().max_trials})"
        )
    env["ROGUE_ARENA_CONCURRENCY"] = str(parallel)
    count = str(parallel)
    return [flag_arg for flag in _PARALLEL_FLAGS for flag_arg in (flag, count)]


def _log_dir_of_run(passthrough: Sequence[str], env: Mapping[str, str]) -> str:
    """Where the child writes its logs: as Inspect resolves ``--log-dir``."""
    for index, arg in enumerate(passthrough):
        if arg == "--log-dir" and index + 1 < len(passthrough):
            return passthrough[index + 1]
        if arg.startswith("--log-dir="):
            return arg.split("=", 1)[1]
    return env.get("INSPECT_LOG_DIR") or _INSPECT_DEFAULT_LOG_DIR


@dataclass(frozen=True, slots=True)
class _LogFacts:
    """What a ``.eval`` says about its run; None for what it doesn't say."""

    run_id: str | None
    model: str | None
    status: str | None


def _log_facts(path: Path) -> _LogFacts:
    """A ``.eval``'s run tag, the model it evaluated, and how its task ended."""
    try:
        header = read_eval_log(str(path), header_only=True)
    except Exception:  # noqa: BLE001 - another run's log may still be being written
        return _LogFacts(None, None, None)
    value = (header.eval.metadata or {}).get(_RUN_METADATA_KEY)
    return _LogFacts(
        None if value is None else str(value), header.eval.model, header.status
    )


def _logs_of_run(
    log_dir: Path, run_id: str, started_at: float
) -> list[tuple[Path, _LogFacts]]:
    """This run's ``.eval`` files: written since it started and tagged with its id."""
    since = started_at - _LOG_MTIME_SLACK_SECONDS
    recent = (
        path for path in sorted(log_dir.glob("*.eval")) if path.stat().st_mtime >= since
    )
    logs = ((path, _log_facts(path)) for path in recent)
    return [(path, facts) for path, facts in logs if facts.run_id == run_id]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _describe_jobs(jobs: Sequence[EvalJob]) -> str:
    return ", ".join(f"{job.id} ({job.model or 'no model'})" for job in jobs)


def _several_logs_report(
    logs: Sequence[tuple[Path, _LogFacts]], jobs: Sequence[EvalJob]
) -> str:
    """How to upload each log to its own run, when one upload can't cover the run."""
    summary = (
        f"{_PROG}: not sharing a transcript automatically: this run has "
        f"{_count(len(logs), 'log')} and {_count(len(jobs), 'job')}, and a job holds "
        "one transcript. To upload each log to its own run:"
    )
    lines = [summary]
    unmatched = False
    for path, facts in logs:
        job = _job_for_log(facts, jobs)
        unmatched = unmatched or job is None
        run = job.id if job is not None else "<run>"
        lines.append(f"  {_PROG} share-transcript {shlex.quote(str(path))} --run {run}")
    if unmatched:
        lines.append(f"  where <run> is one of: {_describe_jobs(jobs)}")
    return "\n".join(lines)


def _share_transcript_after_run(
    args: argparse.Namespace, run_id: str, log_dir: str, started_at: float
) -> None:
    """Upload the finished run's ``.eval``; never changes the eval's exit code.

    Finds the ``.eval`` by the id the run was tagged with and the job by its
    ``clientJobKey``, then shares it once. A run with several jobs (one per model)
    gets a report of how to upload each log instead. Any problem prints a clear
    line and returns: the eval already succeeded, and its logs stay where Inspect
    wrote them.
    """
    if "://" in log_dir:
        print(
            f"{_PROG}: not sharing a transcript: it can only be read from a local "
            f"log folder, not {log_dir}",
            file=sys.stderr,
        )
        return
    all_logs = _logs_of_run(Path(log_dir), run_id, started_at)
    if not all_logs:
        print(
            f"{_PROG}: not sharing a transcript: expected a .eval for this run in "
            f"{log_dir}, found 0 (pass --log-dir after -- if Inspect logs somewhere "
            "else)",
            file=sys.stderr,
        )
        return
    # Each log's own status: Inspect can exit 0 when one task of a multi-model run
    # errored, so the exit code doesn't say which logs finished.
    logs = [log for log in all_logs if log[1].status == _LOG_FINISHED]
    for path, facts in all_logs:
        if facts.status != _LOG_FINISHED:
            print(
                f"{_PROG}: not sharing {path.name}: its task did not finish "
                f"({facts.model or 'no model'}, status {facts.status})",
                file=sys.stderr,
            )
    if not logs:
        return
    try:
        jobs, result = asyncio.run(_upload_transcript(args, run_id, logs))
    except Exception as error:  # noqa: BLE001 - sharing never fails a finished eval
        message = _redact(_describe(error, args), _token(args))
        retries = "\n".join(
            f"{_PROG}: to retry: {_PROG} share-transcript {shlex.quote(str(path))}"
            for path, _ in logs
        )
        print(
            f"{_PROG}: could not share the transcript: {message}\n{retries}",
            file=sys.stderr,
        )
        return
    if result is None:
        print(_several_logs_report(logs, jobs), file=sys.stderr)
        return
    print(f"{_PROG}: {_render_transcript_upload(result)}", file=sys.stderr)


def _render_transcript_upload(result: EvalTranscriptUpload) -> str:
    transcript = result.transcript
    expiry = f", expires {transcript.expires_at}" if transcript.expires_at else ""
    state = "already shared" if result.replayed else "shared"
    return (
        f"{state} transcript {transcript.id} on run {transcript.job_id} "
        f"({transcript.byte_length} bytes{expiry})"
    )


async def _upload_transcript(
    args: argparse.Namespace, run_id: str, logs: Sequence[tuple[Path, _LogFacts]]
) -> tuple[Sequence[EvalJob], EvalTranscriptUpload | None]:
    """The run's jobs, and the upload when one log and one job make it unambiguous."""
    client = _client(args)
    jobs = await _jobs_of_run(client, run_id)
    if len(logs) == 1 and len(jobs) == 1:
        return jobs, await _share_file(client, jobs[0].id, logs[0][0])
    return jobs, None


async def _share_file(
    client: RogueArenaClient,
    job_id: str,
    log_path: Path,
    idempotency_key: str | None = None,
) -> EvalTranscriptUpload:
    return await client.share_transcript(
        job_id,
        data=log_path.read_bytes(),
        file_name=sanitize_transcript_file_name(log_path.name),
        idempotency_key=idempotency_key,
    )


def _job_belongs_to_run(job: EvalJob, run_id: str) -> bool:
    """Whether a run tagged ``run_id`` created this job (``inspect_<run_id>-…``)."""
    return (job.client_job_key or "").startswith(f"inspect_{run_id}-")


def _job_accepts_log(job: EvalJob, facts: _LogFacts) -> bool:
    """Whether what the log says about its run and model fits this job."""
    if facts.run_id is not None and not _job_belongs_to_run(job, facts.run_id):
        return False
    return job.model is None or facts.model is None or job.model == facts.model


def _job_for_log(facts: _LogFacts, jobs: Sequence[EvalJob]) -> EvalJob | None:
    """The one job this log fits; None when none does, or several could."""
    matches = [job for job in jobs if _job_accepts_log(job, facts)]
    return matches[0] if len(matches) == 1 else None


async def _jobs_of_run(client: RogueArenaClient, run_id: str) -> tuple[EvalJob, ...]:
    """The run's jobs with trials: their ``clientJobKey`` starts ``inspect_<run_id>-``.

    One per model. A job with no trials never ran anything, e.g. the one Inspect's
    startup made under ``--model`` for a task that sets its own model.
    """
    try:
        jobs = await client.list_runs(limit=_JOB_LOOKUP_LIMIT)
    except RogueArenaApiError as error:
        if error.status_code in _ABSENT_STATUSES:
            raise RuntimeError(
                "this hub has no token-scoped run list (GET /hub/eval/jobs), so "
                "the run's job cannot be found"
            ) from error
        raise
    matches = tuple(
        job for job in jobs if _job_belongs_to_run(job, run_id) and job.trial_count > 0
    )
    if not matches:
        raise RuntimeError("the run's job was not found")
    return matches


def _launch(args: argparse.Namespace) -> int:
    with contextlib.ExitStack() as stack:
        return _launch_within(args, stack)


def _launch_within(args: argparse.Namespace, stack: contextlib.ExitStack) -> int:
    base_url, token = _require_env(args)
    task = _resolve_task(args, stack)
    env = dict(os.environ)
    env["ROGUE_ARENA_BASE_URL"] = base_url
    env["ROGUE_ARENA_TOKEN"] = token  # in env only, never on the child argv
    if args.eval:
        env["ROGUE_ARENA_EVAL"] = args.eval
    if args.max_runtime is not None:
        # The sandbox config owns the 60..86400 bound; it validates this value.
        env["ROGUE_ARENA_MAX_RUNTIME_SECONDS"] = str(args.max_runtime)
    if args.report is not None:
        # The built-in tasks read it through rogue_arena_scorers().
        env[REPORT_ENV] = args.report
    if args.keep_range:
        # The sandbox config reads it; the hub stores it on the job at create.
        env[KEEP_RANGE_ENV] = "true"
    if args.preseed:
        preseed_path = Path(args.preseed).expanduser().resolve()
        # Checked here so a bad file fails before inspect starts; the sandbox
        # reads it again (ROGUE_ARENA_PRESEED) and the hub checks the targets.
        preseed = load_preseed(preseed_path)
        env[PRESEED_ENV] = str(preseed_path)
        print(
            f"{_PROG}: --preseed: {len(preseed.steps)} step(s) run on each trial's "
            "range before the model starts; the wall clock starts after them",
            file=sys.stderr,
        )
    if args.prompt_level is not None:
        selector = env.get("ROGUE_ARENA_EVAL", "").strip()
        if not selector:
            raise RuntimeError(
                "--prompt-level needs an eval to read the prompt from: pass --eval "
                "(or set ROGUE_ARENA_EVAL)"
            )
        # Only the prompt crosses to the child: the guide's graders and solution
        # are for the person running the eval, never the model under test.
        guide = asyncio.run(_guide(args, selector))
        env[PROMPT_ENV] = guide.prompt(EvalPromptLevel(args.prompt_level))
    if args.parallel is not None and args.team:
        raise RuntimeError(
            "--parallel does not run team tasks: a team's chatroom and activity "
            "feed belong to its run, so parallel team trials in one run would "
            "share them; launch separate team runs instead"
        )
    model = args.model
    if args.team:
        team_path = Path(args.team).expanduser().resolve()
        if not team_path.is_file():
            raise RuntimeError(f"team config not found: {team_path}")
        env["ROGUE_ARENA_TEAM"] = str(team_path)
        from inspect_rogue_arena.team import load_team_config

        team_config = load_team_config(team_path)
        # The sandbox config reads the run's visibility from this same team file
        # (ROGUE_ARENA_TEAM).
        if not model:
            model = team_config.member_model()
    if not model:
        raise RuntimeError(
            "set --model (a single-agent run needs a top-level model; a team run "
            "takes it from the team config if you omit --model)"
        )
    passthrough = [arg for arg in args.inspect_args if arg != "--"]
    cmd = [
        sys.executable,
        "-m",
        "inspect_ai",
        "eval",
        _task_argument(task),
        "--model",
        model,
    ]
    if args.parallel is not None:
        cmd += _parallel_args(args.parallel, passthrough, env)
        print(
            f"{_PROG}: --parallel {args.parallel}: the job admits "
            f"{args.parallel} trials at once (ROGUE_ARENA_CONCURRENCY)",
            file=sys.stderr,
        )
    run_id = str(uuid.uuid4())
    if args.share_transcript or args.keep_range:
        # Ties the child's jobs to this launch, so the upload and the keep-range
        # notice find them after the child exits.
        env["ROGUE_ARENA_RUN_ID"] = run_id
    if args.share_transcript:
        # ...and its .eval, whatever else shares the log folder.
        cmd += ["--metadata", f"{_RUN_METADATA_KEY}={run_id}"]
    cmd += passthrough
    # The token is in env, not in cmd; this line is safe to show.
    print(f"{_PROG}: {shlex.join(cmd)}", file=sys.stderr)
    started_at = time.time()
    code, child_stderr = _run_inspect(cmd, env)
    if args.keep_range:
        # Whatever the exit code: a scored trial's range may be up and billing.
        _report_kept_ranges(args, run_id)
    if code != 0:
        hint = _provider_policy_hint(child_stderr)
        if hint is not None:
            print(f"{_PROG}: {_redact(hint, token)}", file=sys.stderr)
        if args.share_transcript:
            print(
                f"{_PROG}: the eval did not finish, so no transcript was shared",
                file=sys.stderr,
            )
        return code
    if args.share_transcript:
        _share_transcript_after_run(
            args, run_id, _log_dir_of_run(passthrough, env), started_at
        )
    return code


async def _kept_runs_of_launch(
    client: RogueArenaClient, run_id: str
) -> tuple[EvalJob, ...]:
    """This launch's runs that ran trials and still keep their range open."""
    jobs = await client.list_runs(limit=_JOB_LOOKUP_LIMIT)
    return tuple(
        job
        for job in jobs
        if _job_belongs_to_run(job, run_id)
        and job.trial_count > 0
        and job.range_kept_open
    )


def _report_kept_ranges(args: argparse.Namespace, run_id: str) -> None:
    """Tell the user which runs kept ranges up, and how to close them (stderr)."""
    try:
        jobs = asyncio.run(_kept_runs_of_launch(_client(args), run_id))
    except Exception as error:  # noqa: BLE001 - the notice must not fail the launch
        print(
            f"{_PROG}: --keep-range: could not list this launch's runs "
            f"({_redact(_describe(error, args), _token(args))}); find them with "
            f"`{_PROG} runs` and close each kept range with `{_PROG} close <run-id>`",
            file=sys.stderr,
        )
        return
    for job in jobs:
        print(
            f"{_PROG}: --keep-range: run {job.id} keeps its scored trials' ranges "
            f"up, billed like any lab, until you run `{_PROG} close {job.id}`",
            file=sys.stderr,
        )


# --- errors, redaction, parsing ---------------------------------------------


def _error_text(error: Exception) -> str:
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _describe(error: Exception, args: argparse.Namespace) -> str:
    if isinstance(error, EvalGuideNotOffered):
        # Its own status is a 401 or a redirect, but whoami cleared the token.
        return str(error)
    if isinstance(error, RogueArenaApiError) and error.status_code == 401:
        body = error.body
        detail = body.get("errorMessage") if isinstance(body, Mapping) else None
        reason = f": {detail}" if isinstance(detail, str) and detail else ""
        return (
            f"Rogue Arena rejected the token (HTTP 401{reason}); set "
            "ROGUE_ARENA_TOKEN (or --token) to a current personal ra_eval_ token"
        )
    if isinstance(error, RogueArenaApiError) and 300 <= error.status_code < 400:
        # The client never follows redirects: a 3xx means the base URL is not
        # answering the API directly.
        return (
            f"the server redirected the request (HTTP {error.status_code}) instead "
            "of answering it: check that ROGUE_ARENA_BASE_URL (or --base-url) "
            "points at Rogue Arena"
        )
    if isinstance(error, httpx.TransportError):
        return (
            f"could not reach Rogue Arena at {_base_url(args)} ({_error_text(error)})"
        )
    if isinstance(error, RuntimeError):
        return str(error)
    return _error_text(error)


def _redact(message: str, token: str) -> str:
    return message.replace(token, _REDACTED) if token else message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Rogue Arena model-eval identity, launch visibility and runs",
    )
    parser.add_argument("--base-url", help="defaults to ROGUE_ARENA_BASE_URL")
    parser.add_argument(
        "--token",
        help=(
            "defaults to ROGUE_ARENA_TOKEN (prefer the environment variable: a "
            "flag value is visible in the process list)"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("whoami", help="show the token owner and organization")
    evals = commands.add_parser(
        "evals", help="list the deployable evals for this token"
    )
    evals.add_argument(
        "--json", action="store_true", help="print the raw listing as JSON"
    )
    evals.add_argument(
        "--dump-vms",
        metavar="EVAL",
        help=(
            "list one eval's machines (name, stable id, OS) a --preseed step can "
            "target; an eval id or its unique name. Base machines are left out"
        ),
    )
    runs = commands.add_parser("runs", help="list this token's recent runs")
    runs.add_argument("--json", action="store_true", help="print the raw jobs as JSON")
    eval_guide = commands.add_parser(
        "eval",
        help="show one eval's graders, solution and recommended prompts (--guide)",
    )
    eval_guide.add_argument("id", help="an eval environment id, or its unique name")
    eval_guide.add_argument(
        "--guide",
        action="store_true",
        help=(
            "show the eval's graders, solution writeup and recommended prompts "
            "(these are for you, not the model under test)"
        ),
    )
    eval_guide.add_argument(
        "--prompt-level",
        choices=[level.value for level in EvalPromptLevel],
        help=(
            "print only the author's recommended prompt at this level "
            "(easy gives the most direction, hard the least)"
        ),
    )
    eval_guide.add_argument(
        "--json", action="store_true", help="print the raw guide as JSON"
    )
    share = commands.add_parser(
        "share-transcript",
        help="upload an existing .eval transcript to its run (opt-in)",
    )
    share.add_argument("file", help="the run's .eval log file")
    share.add_argument(
        "--run",
        metavar="ID",
        help=(
            "the run (job) id; needed only for a log not launched with "
            "--share-transcript"
        ),
    )
    share.add_argument(
        "--new-upload",
        action="store_true",
        help=(
            "use a new upload key, when Rogue Arena asks for one: after an integrity "
            "failure, or when the run's earlier transcript was deleted"
        ),
    )
    close = commands.add_parser(
        "close",
        help="close a --keep-range run's kept range (Rogue Arena removes its ranges)",
    )
    close.add_argument("id", help="a run (job) id, or a trial id")
    close.add_argument(
        "--json", action="store_true", help="print the closed run as JSON"
    )
    run = commands.add_parser("run", help="show one run's status, score and stages")
    run.add_argument("id", help="a run (job) id, or a trial id")
    run.add_argument(
        "--json", action="store_true", help="print the raw objects as JSON"
    )
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--watch",
        action="store_true",
        help="live-tail the run's trial events and status until it settles",
    )
    run_mode.add_argument(
        "--chat",
        action="store_true",
        help="print the run's team chatroom transcript",
    )
    run.add_argument(
        "--interval",
        type=float,
        default=_WATCH_INTERVAL_SECONDS,
        help=(
            "--watch poll cadence in seconds "
            f"(default {_WATCH_INTERVAL_SECONDS:g}, floor {_WATCH_MIN_INTERVAL_SECONDS:g})"
        ),
    )
    launch = commands.add_parser(
        "launch", help="run `inspect eval` on the built-in (or your) task"
    )
    launch.add_argument("--team", help="a team config file (YAML/JSON) for a team run")
    launch.add_argument("--eval", help="eval selector (sets ROGUE_ARENA_EVAL)")
    launch.add_argument("--model", help="the model to run (team: defaults from config)")
    launch.add_argument("--task", help="an Inspect task file (default: the built-in)")
    launch.add_argument(
        "--max-runtime",
        type=int,
        metavar="SECONDS",
        help=(
            "per-trial wall clock in seconds (sets ROGUE_ARENA_MAX_RUNTIME_SECONDS; "
            "60 to 86400)"
        ),
    )
    launch.add_argument(
        "--parallel",
        type=_positive_int,
        metavar="N",
        help=(
            "run N trials at once: sets inspect's --epochs, --max-sandboxes and "
            "--max-samples to N, and the job's concurrency "
            "(ROGUE_ARENA_CONCURRENCY) to N"
        ),
    )
    launch.add_argument(
        "--prompt-level",
        choices=[level.value for level in EvalPromptLevel],
        help=(
            "give the model the eval author's recommended prompt at this level "
            "(the built-in tasks use it; your own task reads it with task_prompt())"
        ),
    )
    launch.add_argument(
        "--share-transcript",
        action="store_true",
        help=(
            "after the eval finishes, upload the run's .eval transcript to Rogue "
            "Arena (opt-in; one upload per run)"
        ),
    )
    launch.add_argument(
        "--keep-range",
        action="store_true",
        help=(
            "advanced: keep each trial's range up after it is scored, billed like "
            f"any lab, until you run `{_PROG} close <run-id>` (sets "
            f"{KEEP_RANGE_ENV}); off by default"
        ),
    )
    launch.add_argument(
        "--preseed",
        metavar="FILE",
        help=(
            "a preseed file (YAML/JSON) of bash or PowerShell steps Rogue Arena runs "
            "on each trial's range before the model starts, each on a machine named "
            f"by its name or id (see `{_PROG} evals --dump-vms <eval>`); sets "
            f"{PRESEED_ENV}"
        ),
    )
    launch.add_argument(
        "--report",
        choices=[report.value for report in HopReport],
        help=(
            "official: the eval's own scoring mode (default); both: also report "
            "the sequential and independent hop counts (sets ROGUE_ARENA_REPORT)"
        ),
    )
    launch.add_argument(
        "inspect_args",
        nargs=argparse.REMAINDER,
        help="extra args passed through to `inspect eval` (use -- to separate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "launch":
        try:
            return _launch(args)
        except Exception as error:  # noqa: BLE001 - the CLI turns failures into exit 1
            message = _redact(_describe(error, args), _token(args))
            parser.exit(1, f"{_PROG}: {message}\n")
    try:
        result = asyncio.run(_ASYNC_COMMANDS[args.command](args))
    except Exception as error:  # noqa: BLE001 - the CLI turns failures into exit 1
        message = _redact(_describe(error, args), _token(args))
        parser.exit(1, f"{_PROG}: {message}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
