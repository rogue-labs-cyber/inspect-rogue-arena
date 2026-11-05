# inspect-rogue-arena

[Inspect](https://inspect.aisi.org.uk/) sandbox provider for running AI model
evals on [Rogue Arena](https://arena.roguelabs.io) cyber ranges. You write an
ordinary Inspect task; each sample claims a model-eval copy, runs on its own
range, and Rogue Arena scores the eval's hops when the trial closes.

- The model runs **on your machine**, through your own model-provider key (for
  example Anthropic, Google, OpenAI or OpenRouter).
- The target range runs **in Rogue Arena**, reached with a personal token.
- Scoring is done **by Rogue Arena** (it checks each hop); the package never
  computes a score or submits a key.

Support for running Rogue Arena evals from [Harbor](https://harborframework.com)
is in development.

## Quickstart

1. **Get access.** Running evals needs a Rogue Arena account on a team with
   model-eval access. Access is not self-service: to get access to our library
   of ready-built cyber-range eval scenarios and model-eval runs, email
   <info@roguelabs.io>. Once you're added, create your token below. You will
   also need an API key for a model provider (for example Anthropic, Google,
   OpenAI or OpenRouter).

2. **Install** (Python 3.10+), with the extra for your provider:

   ```bash
   pip install "inspect-rogue-arena[<provider>]"
   # <provider> is one of: anthropic, google, openai, openrouter
   # or [all-providers] for all of them
   ```

   The extra pulls in that provider's client library.

3. **Create your token.** Sign in at <https://arena.roguelabs.io> and go to
   **Team Hub → Model Testing → Tokens** (`/team-hub/model-testing/tokens`).
   Click **Create token**, give it a **Token label** (e.g. `laptop` or
   `ci-runner`), choose **Expires in** (7, 30, 90 or 365 days), and submit. The
   token (`ra_eval_…`) is shown **once**: "Copy it now. You won't see it
   again." It runs any published eval you can deploy, as you.

4. **Set your environment:**

   ```bash
   export ROGUE_ARENA_BASE_URL=https://arena.roguelabs.io
   export ROGUE_ARENA_TOKEN=ra_eval_…     # the token you just copied
   # plus your model-provider key, e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY,
   # GOOGLE_API_KEY or OPENROUTER_API_KEY
   ```

   Keep the token in the environment: it is never a config field, so it never
   lands in an Inspect log, and the CLI redacts it from errors.

5. **Check the token and list evals:**

   ```bash
   rogue-arena-inspect whoami   # your user/org, token label and last 4
   rogue-arena-inspect evals    # the evals this token can run
   ```

6. **Run** the built-in example eval against a model of your choice (replace the
   id with one from `evals`):

   ```bash
   rogue-arena-inspect launch --eval <environmentId> --model <provider>/<model>
   ```

   Pick a model by its Inspect provider id; see Inspect's
   [model providers](https://inspect.aisi.org.uk/providers.html) for the ids and
   the keys each needs. For a dry run with no model spend, use Inspect's
   built-in offline mock, `mockllm/model`.

   The agent's shell tool runs long commands to completion (a background exec
   the sandbox polls), and the model can ask for a longer timeout per command,
   so a slow scan, build or download is not cut off.

7. **See results.** The run prints a log location; open it with Inspect's
   viewer:

   ```bash
   inspect view
   ```

   Your runs also appear in the web app under **Team Hub → Model Testing →
   Runs** (`/team-hub/model-testing/runs`), with the per-hop score. From the
   CLI, `rogue-arena-inspect runs` lists your runs and
   `rogue-arena-inspect run <id>` shows one run's status, score and per-stage
   result.

8. **Revoke** a token any time from the same **Tokens** page: click **Revoke**
   on its row. If you lose a token, revoke it and create a new one.

## Troubleshooting

- **`401` / "set a current token".** The token is wrong, expired or revoked.
  Create a fresh one on the Tokens page and re-export `ROGUE_ARENA_TOKEN`.
- **`evals` lists nothing.** Your team has no model-eval access or no eval
  shared with it yet. Access is not self-service: email <info@roguelabs.io> to
  get access to our library of ready-built cyber-range eval scenarios and
  model-eval runs, and to have an eval shared with your team.
- **A provider error (`401`/`403`, a model/region policy block).** The model
  runs through *your* provider credentials, so this comes from the provider,
  not from Rogue Arena. Check the provider API key and that the model is
  permitted for your account and region. `launch` prints a clean line pointing
  here instead of a raw provider stack trace.
- **A trial won't start (`TrialClosedBeforeReady`, `TrialReadyTimeout`, or a
  claim refused with `409`).** `TrialClosedBeforeReady` means Rogue Arena ended
  the trial while building its range; the message carries Rogue Arena's reason.
  If the message says the trial's wall clock ran out, raise `--max-runtime`
  (see [Large scenarios](#large-scenarios)). A `409` means no free
  deployment/concurrency slot right now: lower `concurrency`, or retry when a
  slot frees. A `402` funding denial means the eval could not be funded for
  your organization.

## What the client sends to Rogue Arena

The model's conversation (prompts, reasoning, replies) stays on your machine;
what it does on the range (commands and their output, file transfers, team chat
posts) passes through Rogue Arena. In detail:

- **Run and task identity, on each claim.** The eval you chose (its environment
  id, and the eval's name), your Inspect task name, a run id the client
  generates, and Inspect's sample id; plus Inspect's epoch, sample uuid, eval id
  and eval-set id when Inspect sets them.
- **Commands and their output.** Every shell command the agent runs on the
  range goes to Rogue Arena to run on the VM, and its stdout/stderr come back
  through Rogue Arena (which returns at most the last 10 MiB of a command's
  output). Files you upload or download pass through Rogue Arena as their raw
  bytes.
- **Team chat and activity (team runs only).** Chat messages members post, and
  the activity feed. The feed stores only a summary of each command (the
  command, its exit code and about 400 characters of output); the command's
  full output still passes through Rogue Arena when it runs.
- **Not sent.** Your model-provider key never goes to Rogue Arena, and neither
  does the model's conversation. Your Rogue Arena token is sent only as the
  bearer credential.

---

The rest of this document is reference material.

## Authenticate (detail)

```bash
export ROGUE_ARENA_BASE_URL=https://arena.roguelabs.io
export ROGUE_ARENA_TOKEN=ra_eval_…
```

`ROGUE_ARENA_TOKEN` is your personal model-eval token (`ra_eval_…`), sent as the
bearer on every call. Keep it in the environment: it is not a sandbox config
field, so it is never serialized into an Inspect log, and the CLI never prints
it (it is redacted from error messages).

Supported Inspect versions: `>=0.3.260,<0.3.274` (pip resolves one
automatically). 0.3.260 is the first with the sandbox errors the provider
raises.

## Check the token and pick an eval

```bash
rogue-arena-inspect whoami   # userMainId, organizationId, tokenLabel, tokenLast4
rogue-arena-inspect evals    # the evals this token can run, with launch visibility
```

`evals` prints one block per eval with its name, version, stage (hop) count,
scoring mode, and **launch visibility**:

```
example-ad-chain  |  v3 · 4 stages · sequential · ready now (2 warm)
    id: 11111111-1111-4111-8111-111111111111
example-web-app  |  v1 · 2 stages · independent · on demand, ~7 min
    id: 22222222-2222-4222-8222-222222222222
```

`ready now (N warm)` means warm copies are claimable now; `on demand, ~X min`
means a claim builds a fresh copy; `unknown` means readiness was not reported
(the command still succeeds). `evals --json` prints the raw listing; `whoami`
prints JSON. `--base-url` and `--token` override the environment, but prefer the
environment variable for the token (a flag is visible in the process list).

## Runs

```bash
rogue-arena-inspect runs             # this token's recent runs (jobs)
rogue-arena-inspect run <id>         # one run's status, score and per-stage result
rogue-arena-inspect run <id> --watch # live-tail the run's events until it settles
rogue-arena-inspect run <id> --chat  # print the run's team chatroom transcript
```

`run <id>` takes a run (job) id (or a trial id as a fallback) and prints the
run's status and, per trial, the server score and each stage (`#<step> <name>:
pass/fail`). Each trial's terminal reason reads as the Rogue Arena enum plus a
short gloss, e.g. `reason=timeout (ran out of wall clock)`.

- `--watch` polls the run and its trial events on an interval (`--interval`,
  default 3 s), printing new exec/hop/lifecycle events as they arrive until the
  job settles, then prints the final render. A team run is one trial, so it
  follows that trial like any other.
- `--chat` prints the run's team chatroom, one line per message.

`runs`, `run`, and `run --chat` take `--json` for the raw objects.

## Launch

```bash
rogue-arena-inspect launch --eval <id> --model <provider>/<model>  # built-in example task
rogue-arena-inspect launch --eval <id> --task my_task.py --model … # your own task
rogue-arena-inspect launch --team team.yaml --eval <id>            # a multi-agent team
rogue-arena-inspect launch --eval <id> --model … --max-runtime 1800 # per-trial wall clock (s)
rogue-arena-inspect launch --eval <id> --model … --report both      # sequential and independent counts
rogue-arena-inspect launch --eval <id> --model … --parallel 4       # 4 trials at once
```

`launch` is a thin wrapper around `inspect eval`. It resolves the built-in
example task (shipped inside the package), sets `ROGUE_ARENA_EVAL`, keeps the
token in the environment (never on the child argv), and runs `inspect eval`.
Anything after `--` is passed straight through to `inspect eval` (e.g.
`-- --limit 1`). `--max-runtime SECONDS` sets the per-trial wall clock (60 to
86400). `--report both` also reports the sequential and independent hop counts;
see [Both scoring modes](#both-scoring-modes-reportboth). `--parallel N` runs N
trials at once; see [Parallel runs](#parallel-runs-launch---parallel-n).

### Provider errors

The model runs on your machine through your own provider credentials, so a
provider refusal (a model/region policy block, a bad or missing API key) comes
from the provider, not from Rogue Arena. When `launch`'s `inspect eval` child
exits on one of these, `launch` prints a clean, actionable line instead of a raw
provider stack trace; the child's own output is still shown. Check your provider
API key, that the model is allowed for your account and region, and that the
matching provider extra is installed. Inspect's
[model providers](https://inspect.aisi.org.uk/providers.html) page lists each
provider's model ids and keys.

## Write your own task

The example task files ship inside the package, under
`inspect_rogue_arena/examples/` (`eval_example.py`, `team_example.py`,
`team.yaml`, and the `eval-example.eval.yaml` run config).
`rogue-arena-inspect launch` runs them without a checkout. To customize one,
find the folder, copy the file out, and pass it with `--task`:

```bash
python -c "import importlib.resources as r; print(r.files('inspect_rogue_arena') / 'examples')"
```

Or write your own task. Use the scorer and give the sandbox a config:

```python
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.util import SandboxEnvironmentSpec

from inspect_rogue_arena import RogueArenaSandboxConfig, rogue_arena_hops, rogue_bash


@task
def my_eval() -> Task:
    return Task(
        dataset=[Sample(input="…your objective…", target="scored by Rogue Arena")],
        solver=react(tools=[rogue_bash()]),
        scorer=rogue_arena_hops(),
        sandbox=SandboxEnvironmentSpec(
            "rogue-arena", RogueArenaSandboxConfig(eval="<environmentId>")
        ),
    )
```

`rogue_bash()` is the shell tool the examples use: it states its default timeout
to the model and runs long commands as a background exec, so a slow command runs
to completion and a transient network failure on a poll is retried rather than
ending the run.

## Teams (multi-agent)

A **team** run puts several agents on **one range**. Each member has its own
model, system prompt and shell channel into the one VM; they share a chatroom
(`post_message` / `read_messages`), a scoreboard (`post_finding` /
`read_scoreboard`) and, in `shared_sight`, an activity feed. A team is one
Inspect sample, so it claims exactly one trial and drives one VM.

Describe the team in a YAML/JSON file (the packaged `team.yaml` is a template;
copy it out to edit) and launch it:

```bash
rogue-arena-inspect launch --team team.yaml --eval <environmentId>
```

The team file sets `turn_taking` (`coordinator`, where a lead agent delegates
via Inspect's `handoff`, or `round_robin`), `visibility`, optional
`tool_timeout_seconds`, `rounds` / `turns_per_member`, and the `members` (each
with `model`, `name`, `role`, `prompt`).

**Command timeouts.** A member's `bash` runs every command in background mode:
it runs to completion and returns its full output, for at most
`tool_timeout_seconds` (a top-level team field, default 600), and the tool
description tells the model so. The model may pass its own `timeout` (seconds)
per command. A command that outlives its timeout is stopped, with everything it
started, and ends with Inspect's timeout error.

### Visibility: `shared_sight` vs `communicate_only`

`visibility` decides whether members can read each other's actions. In **both**
modes the activity feed is recorded and sent to Rogue Arena (the run owner can
see it on the run page). The feed stores only a summary of each command (the
command, its exit code and about 400 characters of output); the command's full
output still passes through Rogue Arena when it runs.

- `shared_sight` (the default) gives each member a `read_activity` tool to read
  the feed, so teammates can avoid colliding.
- `communicate_only` withholds `read_activity`, and Rogue Arena also refuses a
  member's read of the feed during the run, so members coordinate through the
  chatroom alone. The feed is still recorded for the run owner.

Either way, members share the one VM, chatroom and scoreboard. If Rogue Arena
does not serve the activity routes, a `shared_sight` team prints one line and
runs chat-only rather than letting a `read_activity` call end the run.

**Team scoring.** A team run is one trial; its score is the team's. Rogue Arena
scores it when the trial closes, like any other trial, and a key-file hop counts
a key any member surfaced.

## Sandbox config

| Key | Environment variable | Default | Meaning |
|---|---|---|---|
| `eval` | `ROGUE_ARENA_EVAL` | none (required) | The eval to run: an `environmentId`, or a `name` that matches exactly one eval. An id match wins; an unknown or ambiguous selector is an error. |
| `max_trials` | `ROGUE_ARENA_MAX_TRIALS` | `10000` | The job's `n`: the most trials the job admits. Set it to at least samples × epochs plus retries. |
| `concurrency` | `ROGUE_ARENA_CONCURRENCY` | `2` | The job's `concurrency`: the most trials open at once. |
| `max_runtime_seconds` | `ROGUE_ARENA_MAX_RUNTIME_SECONDS` | `3600` (60 to 86400) | Each trial's wall clock, counted from its claim, provisioning included. Raise it for a large scenario; see [Large scenarios](#large-scenarios). |
| `max_wait_seconds` | `ROGUE_ARENA_MAX_WAIT_SECONDS` | 2 hours, or `max_runtime_seconds` plus 5 minutes if longer | The backstop on a trial's range build. The provider waits while Rogue Arena reports the build in progress, and Rogue Arena ends a stalled build itself; the default only bounds a wait that never ends. |
| `heartbeat_interval_seconds` | `ROGUE_ARENA_HEARTBEAT_INTERVAL_SECONDS` | `300` | Lease heartbeat cadence (must be under 900). |
| `sync_exec_ceiling_seconds` | `ROGUE_ARENA_SYNC_EXEC_CEILING_SECONDS` | `50` (at most 55) | The longest command timeout that runs as one synchronous request. |
| `poll_interval_seconds` | `ROGUE_ARENA_POLL_INTERVAL_SECONDS` | `2` (at most 60) | Cadence of the ready, close and background-exec polls. |
| `team_visibility` | (from the team file) | `shared_sight` | A team run's activity-feed visibility: `shared_sight` or `communicate_only` (see Teams). |

Explicit Inspect config wins over environment variables, which win over the
defaults. The config is typed, validated and frozen, and rejects unknown keys.

## Lifecycle

There is **one job per sandbox config per eval run**, and **one trial per
sample**:

1. **`task_init`** resolves `eval` and creates the job with `n`, `concurrency`,
   the wall clock and an idempotency key (a transport failure retries once with
   the same key, so it never creates a second job). Tasks in the same run that
   share a sandbox config share that job.
2. **`sample_init`** waits for a free concurrency slot, then claims a trial with
   its own idempotency key: a transport failure or a 502/503/504 resends the
   claim once and Rogue Arena answers the trial it already opened, never a
   second one. A claim can return a trial still `provisioning`; the provider
   polls it for as long as Rogue Arena reports it `provisioning`, up to the
   `max_wait_seconds` backstop, and cancels it if anything fails before it is
   `ready`. See [Large scenarios](#large-scenarios).
3. **Heartbeat.** A background task heartbeats the trial every
   `heartbeat_interval_seconds` until the wall clock ends; every exec also
   heartbeats first. Only a 409 (the trial is no longer live) stops the loop.
4. **Tool calls** exec, upload and download on the trial, each bounded by the
   trial's remaining wall clock.
5. **Scoring.** `rogue_arena_hops()` closes the trial; Rogue Arena scores every
   hop during the close and returns the scored trial.
6. **`sample_cleanup`** closes a trial that is still open (`cancelled` when the
   sample was interrupted). A close is sent at most once; if it times out, the
   provider reads the trial back until it is `closed` (Rogue Arena finishes the
   close server-side).
7. **`task_cleanup`** closes any leftover trials as `cancelled`, then cancels
   the job.

### Claim errors

Each of these fails the sample and releases its concurrency slot:

| Error | When |
|---|---|
| `EvalClaimDenied` with HTTP 402 (`is_funding` is true) | Funding denial, with the decision (`verdict`, `reason_code`, `error_message`, raw `decision`). |
| `EvalClaimDenied` with HTTP 409 | A deploy denial, with the same decision fields. |
| `RogueArenaApiError` with HTTP 409 | A state refusal: the job is not open, out of trials, already at `concurrency`, or Rogue Arena is not accepting new trials for your organization right now. |
| `TrialReadyTimeout` | The trial was still `provisioning` when the `max_wait_seconds` backstop ran out. The provider cancels it. |
| `TrialClosedBeforeReady` | Rogue Arena closed the trial while it was provisioning: a build that stopped progressing, one that outran the wall clock, or a failed hand-over check. The message (and `reason`) carries Rogue Arena's account of why. |
| `httpx.TransportError`, or `RogueArenaApiError` with HTTP 502/503/504 | The claim and its one resend both went unanswered. The provider cancels any trial that was opened, then raises. |

### Large scenarios

A trial may build a fresh copy of the eval's range, and a large scenario's
build regularly takes more than 15 minutes. The provider judges the wait by
Rogue Arena's state, not by age: it keeps waiting while Rogue Arena reports the
trial `provisioning`. Rogue Arena watches the build's progress and ends a build
that stops progressing (currently after 30 minutes without progress); the
sample then fails with `TrialClosedBeforeReady` and Rogue Arena's reason.
Rogue Arena also ends any build after 3 hours, and a build still running when
the trial's wall clock ends. `max_wait_seconds` is only a backstop for a wait
that never ends: by default 2 hours, or the wall clock plus 5 minutes if that is
longer, so Rogue Arena's own answer always lands first. Set it (or
`ROGUE_ARENA_MAX_WAIT_SECONDS`) to cap the wait yourself. When the trial
reached its wall clock, the message says so and suggests a higher
`--max-runtime`.

**Raise `--max-runtime` for a large scenario.** Each trial's wall clock
(`max_runtime_seconds`, default 3600) runs from the claim, so build time comes
out of the agent's time, and a build still running when the wall clock ends
ends the trial. With a 40-minute build and the default hour, the agent gets
about 20 minutes. Add the expected build time to the time the agent needs, for
example `rogue-arena-inspect launch --eval <id> --model … --max-runtime 7200`.

### Concurrency

A semaphore per sandbox config, sized to `concurrency`, holds each claim until
an earlier trial closes, so the provider never sends a claim that would be
refused (409). Waiting for a slot is ordinary queueing with no time limit.

The provider also implements Inspect's `default_concurrency()` hook, which sets
Inspect's default `--max-sandboxes` to `ROGUE_ARENA_CONCURRENCY` (default 2).
When a task config sets `concurrency`, pass the same value as `--max-sandboxes`
so Inspect's queue matches the job's limit.

### Parallel runs: `launch --parallel N`

```bash
rogue-arena-inspect launch --eval <id> --model … --parallel 4
```

`--parallel N` runs N trials at once from one command. It passes `--epochs N`,
`--max-sandboxes N` and `--max-samples N` to `inspect eval`, and sets the job's
concurrency (`ROGUE_ARENA_CONCURRENCY=N`), so Inspect and Rogue Arena agree on the
limit. Each epoch of a sample is its own trial on its own copy of the range: the
built-in task has one sample, so it runs N trials, and a task with S samples
runs S × N. The job's `max_trials` defaults to 10000, enough for any run, and
`launch` refuses a `ROGUE_ARENA_MAX_TRIALS` below N. `launch` also refuses
`--epochs`, `--max-sandboxes` or `--max-samples` after `--`, and `--team` (a
team's chatroom and activity feed belong to its run, so parallel team trials
would share them; launch separate team runs instead). With more than one epoch,
Inspect reduces each sample's epochs (by default to their mean) before it
computes the metrics, so on the one-sample built-in task the mean is the mean
over the N trials but `stderr` is 0 (Inspect needs two samples for it); each
trial's own score stays in the log's samples.

N is what you ask for. Two things cap how many trials actually run at once:

- **The eval's concurrency limit.** Rogue Arena admits at most the job's
  `concurrency` trials at once, and each trial's copy still needs your
  organization's deploy admission and funding. A claim that is refused
  (`EvalClaimDenied`, HTTP 402 or 409) errors its sample; it does not wait.
  Under Inspect's default `fail_on_error`, that one error stops the whole run
  and cancels the other trials: pick an N your organization can deploy at once,
  or let the run go on past such errors with `-- --no-fail-on-error` (or a
  threshold such as `-- --fail-on-error 0.25`).
- **The warm pool.** Only warm copies start at once (`evals` shows how many,
  e.g. `ready now (2 warm)`). Each trial beyond them builds a fresh copy
  (`on demand, ~X min`), and a large scenario's build can take well over 15
  minutes, so those trials start as their builds finish (see
  [Large scenarios](#large-scenarios)).

`--parallel` sets the environment, so a task whose sandbox config sets
`concurrency` or `max_trials` explicitly keeps its own values (explicit config
wins), and only Inspect's limits change. Remove them from the task's config, or
set them to match.

## Commands

`exec` is **synchronous by default**. A command whose timeout is at most
`sync_exec_ceiling_seconds` (default 50, at most 55) runs as one request; a
command with no timeout gets the ceiling as its timeout.

A longer timeout runs in **background mode**: the provider launches the command,
then polls it (first after 0.25 s, the wait doubling up to
`poll_interval_seconds`) until it finishes, for up to the command's timeout plus
5 minutes (the server ceiling is 4 hours). Rogue Arena stops a background
command and everything it started at its timeout and when its trial ends.

| Result | What the provider does |
|---|---|
| The command exits | `ExecResult`, with `success` when the exit code is 0. |
| The command hits its own timeout | `TimeoutError` with the partial output (Inspect's convention). |
| The trial is not live (409), is closed, or is past `max_runtime_seconds` | `SandboxUnavailableError`: a tool error, and the sample continues. |
| A background exec was lost (no exit status recorded) | `SandboxUnavailableError`. |
| A gateway timeout or transport failure during a sync exec or a background launch | `ExecOutcomeUnknown`: the command may still be running, so it is not retried, and the sample errors. |

A background poll that hits a transient failure retries with exponential backoff
(capped at 30 s) until the deadline. Rogue Arena returns at most the last 10 MiB of a
command's output. `cwd`, `env` and `input` (stdin) pass through; `user`,
`timeout_retry` and `concurrency` are ignored.

## Files

`write_file` and `read_file` move raw bytes, base64-encoded in JSON. Text is
UTF-8. Absolute agent paths only, so a relative path resolves against `/`. A
`read_file` larger than Inspect's read limit raises `OutputLimitExceededError`.
An upload is capped at 10 MiB.

## Scoring: `rogue_arena_hops()`

The scorer closes the trial and reads the **server score** the close returns. It
never computes a hop locally, never runs a command, and never submits a key. To
also report the other scoring mode, see
[Both scoring modes](#both-scoring-modes-reportboth).

Key-file hops are proven server-side: Rogue Arena counts a key-file hop only
when the planted key appears in the agent's exec output (including background
results) or in a file the agent downloads. The agent proves a hop by reading the
key through its tools; there is nothing to submit.

- **Value:** `hops_passed / hops_total`, as counted under the eval's scoring
  mode (`sequential` or `independent`). The answer reads
  `"<hops_passed>/<hops_total> hops"`.
- **Metadata:** `trial_id`, `hops_passed`, `hops_total`, `reached_step`,
  `scoring_mode`, `outcome`, `terminal_reason`, `timed_out`, and `per_hop`.
- **`per_hop.passed` is the raw check result**, not whether the hop counted. In
  `sequential` mode a hop after the first failed step does not count even when
  its check passed; use `hops_passed` and `reached_step` for the count.

| Trial as closed | Result |
|---|---|
| A server score with terminal reason `agent_success` or `agent_fail` | Scored. |
| A server score with terminal reason `timeout` | Scored normally, with `timed_out: true`. |
| Terminal reason `infra_error` (even with a score), or a `cancelled` outcome | Raises `TrialNotScored`: the sample errors and never enters the mean. |
| No server score (reaped/expired/unscored) | Raises `TrialNotScored`. |

With Inspect's default `fail_on_error`, one such infrastructure outcome aborts
the whole run. To tolerate them, pass `--no-fail-on-error`, or a threshold such
as `--fail-on-error 0.1`. From Python, pass `fail_on_error=False` to
`inspect_ai.eval()`. To retry, `--retry-on-error N` claims a fresh trial for
each retry, so leave headroom in `max_trials`.

### Both scoring modes: `report="both"`

An eval's author picks its scoring mode, and `rogue_arena_hops()` keeps that
mode's count as the official value. To also see the other mode, report both:

```bash
rogue-arena-inspect launch --eval <id> --model … --report both   # sets ROGUE_ARENA_REPORT
inspect eval eval_example.py -T report=both --model …             # the task arg
```

In your own task, use `scorer=rogue_arena_scorers(report)`: `official` (the
default; `None` reads `ROGUE_ARENA_REPORT`) is `[rogue_arena_hops()]`, and
`both` adds `rogue_arena_hops_by_mode()`.

- **`sequential`** counts the hops before the first failed step.
  **`independent`** counts every passed hop. They differ when a later hop passed
  after an earlier one failed: hops `[fail, pass, pass]` count 0/3 sequential
  and 2/3 independent.
- The results show a `sequential` and an `independent` metric, each with `mean`
  and `stderr` and each reduced over epochs. The eval's own mode is Rogue
  Arena's count (it always matches `rogue_arena_hops`); the other is recounted
  from Rogue Arena's per-hop check results.
- If a hop that counts under the recounted mode reached no verdict (its check
  did not run, `per_hop[].check_ran: false`), Rogue Arena would not score the
  trial under that mode, so that metric leaves the trial out (its value is NaN
  and its metadata counts are null).
- **Metadata:** `trial_id`, `scoring_mode` (the eval's own), `hops_total`, and
  `hops_passed_<mode>` and `reached_step_<mode>` for each mode.

## Python client

`RogueArenaClient.from_env()` reads `ROGUE_ARENA_BASE_URL` and
`ROGUE_ARENA_TOKEN`. It covers the whole eval API: `whoami`, `list_evals`,
`list_runs`, `create_job`, `get_job`, `cancel_job`, `list_job_trials`,
`claim_trial`, `get_trial`, `heartbeat_trial`, `run_command` (and the
single-request `exec_trial`), `upload_file`, `download_file`, `close_trial` and
`list_trial_events`.

```python
import asyncio

from inspect_rogue_arena import RogueArenaClient


async def show_evals() -> None:
    client = RogueArenaClient.from_env()
    for entry in await client.list_evals():
        print(
            entry.environment_id, entry.name, entry.hop_count, entry.scoring_mode.value
        )


asyncio.run(show_evals())
```

## Development

```bash
pip install -e '.[all-providers]'
python -m unittest discover -s tests
```

The tests run offline: HTTP goes to a mock transport and an in-memory fake
server, and the Inspect runs use `mockllm`, including a run of the shipped
example task.

## License

Apache License 2.0. See
[LICENSE](https://github.com/rogue-labs-cyber/inspect-rogue-arena/blob/main/LICENSE)
and
[NOTICE](https://github.com/rogue-labs-cyber/inspect-rogue-arena/blob/main/NOTICE).
The license grants no rights to the "Rogue Arena" or "Rogue Labs" names or
marks.

## Changelog

### 0.5.0

- **Fix: waiting for a range build.** A trial is now waited for as long as
  Rogue Arena reports its range `provisioning`; Rogue Arena ends a build that
  stops progressing itself. Before, a large scenario whose build took over 15
  minutes failed with `TrialReadyTimeout` while it was still building.
  `max_wait_seconds` (`ROGUE_ARENA_MAX_WAIT_SECONDS`) stays as the override; its
  default is now a backstop of 2 hours, or the wall clock plus 5 minutes if
  that is longer. `TrialClosedBeforeReady` now carries Rogue Arena's reason.
  Build time counts against the trial's wall clock, so raise `--max-runtime`
  for a large scenario; see [Large scenarios](#large-scenarios).
- **Both scoring modes.** `report="both"` (`-T report=both`,
  `ROGUE_ARENA_REPORT=both`, `launch --report both`, or
  `rogue_arena_scorers("both")` in your own task) also reports the sequential
  and independent hop counts as two metrics. `rogue_arena_hops()` and its
  official value are unchanged. See
  [Both scoring modes](#both-scoring-modes-reportboth).
- **Parallel runs.** `launch --parallel N` runs N trials at once. See
  [Parallel runs](#parallel-runs-launch---parallel-n).

### 0.4.0

- First public release.
