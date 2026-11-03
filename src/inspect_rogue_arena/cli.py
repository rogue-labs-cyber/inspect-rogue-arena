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
- ``launch``: a thin wrapper around ``inspect eval`` that passes the task
  path relative to the working directory (``inspect eval`` globs it from there,
  and Python 3.14's pathlib rejects absolute glob patterns) and, with
  ``--team FILE``, runs a multi-agent team task. ``--max-runtime SECONDS`` sets
  the per-trial wall clock (``ROGUE_ARENA_MAX_RUNTIME_SECONDS``).

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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from importlib import resources
from pathlib import Path

import httpx

from inspect_rogue_arena.client import (
    EvalCatalogEntry,
    EvalChatMessage,
    EvalJob,
    EvalJobStatus,
    EvalScore,
    EvalTrial,
    EvalTrialEvent,
    EvalTrialTerminalReason,
    RogueArenaApiError,
    RogueArenaClient,
)

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


def _render_trial(trial: EvalTrial) -> list[str]:
    outcome = trial.outcome.value if trial.outcome else "-"
    rows = [
        (
            f"  trial {trial.id}  {trial.status.value}  outcome={outcome} "
            f"reason={_reason_cell(trial.terminal_reason)}"
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


def _render_run(job: EvalJob, trials: Sequence[EvalTrial]) -> str:
    header = (
        f"run {job.id}  {job.status.value}  env {job.environment_id}  "
        f"n={job.n} concurrency={job.concurrency}  "
        f"trials {job.trial_count} (closed {job.closed_trial_count})"
    )
    rows = [header]
    for trial in trials:
        rows += _render_trial(trial)
    if not trials:
        rows.append("  no trials claimed yet")
    return "\n".join(rows)


def _render_runs(jobs: Sequence[EvalJob]) -> str:
    if not jobs:
        return "this token has no runs yet"
    return "\n".join(
        f"{job.id}  {job.status.value}  env {job.environment_id}  "
        f"n={job.n} concurrency={job.concurrency}  "
        f"trials {job.trial_count} (closed {job.closed_trial_count})"
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


_ASYNC_COMMANDS = {
    "whoami": _cmd_whoami,
    "evals": _cmd_evals,
    "runs": _cmd_runs,
    "run": _cmd_run,
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
    cmd += passthrough
    # The token is in env, not in cmd; this line is safe to show.
    print(f"{_PROG}: {shlex.join(cmd)}", file=sys.stderr)
    code, child_stderr = _run_inspect(cmd, env)
    if code != 0:
        hint = _provider_policy_hint(child_stderr)
        if hint is not None:
            print(f"{_PROG}: {_redact(hint, token)}", file=sys.stderr)
    return code


# --- errors, redaction, parsing ---------------------------------------------


def _error_text(error: Exception) -> str:
    detail = str(error)
    return f"{type(error).__name__}: {detail}" if detail else type(error).__name__


def _describe(error: Exception, args: argparse.Namespace) -> str:
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
    runs = commands.add_parser("runs", help="list this token's recent runs")
    runs.add_argument("--json", action="store_true", help="print the raw jobs as JSON")
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
