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
   built-in offline mock, `mockllm/model`, and add `-- --message-limit 6` to the
   `launch` command: the mock never finishes on its own. Only the model is
   offline, so the run still builds and uses a real range.

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
  See [Large scenarios](#large-scenarios). A `409` means no free
  deployment/concurrency slot right now: lower `concurrency`, or retry when a
  slot frees. A `402` funding denial means the eval could not be funded for
  your organization.
- **A trial ends `preseed_failed`.** One of your [preseed](#preseed-the-range-launch---preseed)
  steps exited non-zero, timed out or could not run, so the model never started
  and the trial is not scored. The error names the step and shows the last part
  of its output; fix the script and run again. With `--keep-range` the range
  stays up, so you can look at what the step did, until you `close` it.

## What the client sends to Rogue Arena

The client talks only to the base URL you set (`ROGUE_ARENA_BASE_URL`), over
HTTPS. It has no telemetry, analytics or update checks. Every request carries
your Rogue Arena token, only as its bearer credential.

The commands that only read (`whoami`, `evals`, `runs`, `run`, `eval`) send the
token and any id you give them. `close <id>` sends the run's id and changes its
state: Rogue Arena removes the ranges the run kept up. A run sends the following.

**On every run, without being asked:**

- **The model's name.** Each job is labelled with the model Inspect is
  evaluating, as Inspect names it (`<provider>/<model>`, for example
  `openai/gpt-4o`), cut to 200 characters. A run over several models creates
  one job per model, each labelled with its own. Rogue Arena keeps it with the
  run; `runs` and `run` show it.
- **Run settings, when the run starts.** The eval you chose (its id), the trial
  count, concurrency and wall clock limit, a run id the client generates,
  whether to keep the range up after scoring (`keepRange`, from `--keep-range`)
  and, for a team run, its visibility mode. A range you keep up is an ordinary
  lab you own: it keeps billing to the run's funding source until you `close`
  the run (or remove it in Rogue Arena).
- **Run and task identity, on each claim.** The eval's name, your Inspect task
  name and Inspect's sample id; plus Inspect's epoch, sample uuid, eval id and
  eval-set id when Inspect sets them.
- **Commands and their output.** Every command the agent runs on the range goes
  to Rogue Arena to run on the VM, with its working directory, environment
  variables and standard input. Its stdout and stderr come back through Rogue
  Arena (at most the last 10 MiB of a command's output). Files the task uploads
  to or downloads from the range pass through Rogue Arena as their raw bytes.
- **Team chat and activity (team runs only).** Every chat message a member
  posts, with the member's name. The activity feed: for each command, the
  member's name, the command (up to 500 characters), its exit code and about 400
  characters of its output. The command's full output still passes through
  Rogue Arena when it runs.

**Only when you ask: a preseed.** With `launch --preseed FILE` (or the
`preseed` sandbox setting), every trial's claim carries the whole preseed: each
step's target machine, shell, timeout and **full script text**. Rogue Arena runs
the scripts as root (Linux) or SYSTEM (Windows) on the lab machines of your
trial's range, before the model starts, and keeps the last 8 KiB of each step's
output on the trial's log. Before the run starts, the client also reads the
eval's machine list (`evals --dump-vms`) to check the targets. Without a
preseed, no script is sent.

**Only when you ask: a shared transcript.** `launch --share-transcript` and
`share-transcript FILE` upload the run's whole `.eval` log file (up to 100 MB),
with its file name and a checksum. Without one of those, no log is uploaded.
The file is Inspect's complete record of the run, so it holds:

- the model's whole conversation: every prompt, reply and reasoning step
  Inspect recorded, and every tool call with its output;
- the scores, the task's arguments and settings, and the model's name and
  generation settings;
- what Inspect records about your setup, such as package versions and, if you
  ran from a git repository, its remote URL and commit.

Inspect leaves an `api_key` model argument out of the log, but anything your
task put in a prompt, in metadata or in tool output is in the file. Open it with
`inspect view` before sharing if you are not sure what it holds. A shared
transcript can be read by you, by anyone who can edit the eval in Rogue Arena,
and by Rogue Arena staff. It is kept for 90 days unless a team admin picks
another period (7, 30, 90, 180 or 365 days); you or a team admin can delete it
sooner.

**Never sent:**

- Your model-provider key. The model runs on your machine through your own
  provider credentials.
- The model's conversation (prompts, reasoning, replies), unless you share a
  transcript. What the model does on the range (its commands, their output and
  its team chat posts) is sent, as listed above.
- Files from your machine, other than the ones a task uploads to the range, a
  transcript you share, and the scripts in a preseed file you pass (the file is
  read on your machine and its scripts are sent as text).

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

`ROGUE_ARENA_BASE_URL` must be an `https://` URL, so the token is never
sent in clear; plain `http://` is accepted only for `localhost` (a local
server or tunnel). Any other URL is refused before a request is made.

Supported Inspect versions: `>=0.3.260,<0.3.274` (pip resolves one
automatically). 0.3.260 is the first with the sandbox errors the provider
raises.

## Check the token and pick an eval

```bash
rogue-arena-inspect whoami   # userMainId, organizationId, tokenLabel, tokenLast4
rogue-arena-inspect evals    # the evals this token can run, with launch visibility
rogue-arena-inspect evals --dump-vms <id>   # one eval's machines a preseed can target
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

### An eval's machines: `evals --dump-vms <id>`

`evals --dump-vms <id>` (an eval id, or its unique name) lists the machines of
the eval's current version that a [preseed](#preseed-the-range-launch---preseed)
step can target: each machine's name, operating system and **stable machine
id**. The id is the machine's id in the eval's scenario, the same in every copy
of that version (never one run's VM id), so a preseed can name it. The base
machines every range carries (the core router, student share, VPN gateway,
internet proxy, package cache and the like) are left out, and so are machines
hidden from you (a protected plugin is on them): those can't be preseeded. The
agent machine (the one the model drives) is marked. `--json` prints the list as
JSON.

```
KALI-01  Kali               id 1f0c…  (agent: the model's machine)
WEB-01   Ubuntu             id 6b2e…
WS-01    Windows11          id 90ad…
```

## Eval guide: graders, solution and prompts

```bash
rogue-arena-inspect eval <id> --guide                      # graders, solution, prompts
rogue-arena-inspect eval <id> --prompt-level easy|medium|hard  # one prompt, to stdout
rogue-arena-inspect launch --eval <id> --model … --prompt-level hard  # run with that prompt
```

Every eval carries an author's guide: the full grading config (each hop's
machine, proof type, key path or command and regex), a solution writeup, and
three recommended task prompts. `<id>` is an environment id or a unique name.
`--guide` prints all of it; `--json` prints the raw object.

The graders and the solution are **for you, the person running the eval** --
they say exactly how it is scored and how to solve it. Never put them in front
of the model under test. The recommended prompts, by contrast, are written to be
given to the model: `easy` gives the most direction, `hard` the least (but still
the win condition). On `eval`, `--prompt-level` prints just that one. On `launch`,
it gives the model that prompt in place of the built-in task's own; only the
prompt reaches the run, never the graders or solution. Your own task picks it up
with `task_prompt(default)` from `inspect_rogue_arena`. A level the author left
blank is an error, not empty output, so a run never starts without its prompt.

## Runs

```bash
rogue-arena-inspect runs             # this token's recent runs (jobs)
rogue-arena-inspect run <id>         # one run's status, score and per-stage result
rogue-arena-inspect run <id> --watch # live-tail the run's events until it settles
rogue-arena-inspect run <id> --chat  # print the run's team chatroom transcript
rogue-arena-inspect close <id>       # remove the ranges a --keep-range run kept up
```

`run <id>` takes a run (job) id (or a trial id as a fallback) and prints the
run's status and, per trial, the server score and each stage (`#<step> <name>:
pass/fail`). Each trial's terminal reason reads as the Rogue Arena enum plus a
short gloss, e.g. `reason=timeout (ran out of wall clock)`. A run that kept its
ranges up shows `range kept open (N up)` until you close it (or `no kept range
up` once every kept range is gone), and a run and its trials with a preseed show
`preseeded`.

- `--watch` polls the run and its trial events on an interval (`--interval`,
  default 3 s), printing new exec/hop/lifecycle events as they arrive until the
  job settles, then prints the final render. A team run is one trial, so it
  follows that trial like any other.
- `--chat` prints the run's team chatroom, one line per message.
- `close <id>` removes every range a [`--keep-range`](#keep-the-range-up-launch---keep-range)
  run kept up (`<id>` is a run id from `runs`, or one of its trial ids).
  Closing twice is fine, and a trial of that run that ends later is removed as
  usual. Only the run's owner can close it.

`runs`, `run`, `run --chat` and `close` take `--json` for the raw objects.

## Launch

```bash
rogue-arena-inspect launch --eval <id> --model <provider>/<model>  # built-in example task
rogue-arena-inspect launch --eval <id> --task my_task.py --model … # your own task
rogue-arena-inspect launch --team team.yaml --eval <id>            # a multi-agent team
rogue-arena-inspect launch --eval <id> --model … --max-runtime 1800 # per-trial wall clock (s)
rogue-arena-inspect launch --eval <id> --model … --report both      # sequential and independent counts
rogue-arena-inspect launch --eval <id> --model … --parallel 4       # 4 trials at once
rogue-arena-inspect launch --eval <id> --model … --share-transcript # upload the run's .eval
rogue-arena-inspect launch --eval <id> --model … --keep-range       # keep scored ranges up
rogue-arena-inspect launch --eval <id> --model … --preseed seed.yaml # run your scripts first
```

`launch` is a thin wrapper around `inspect eval`. It resolves the built-in
example task (shipped inside the package), sets `ROGUE_ARENA_EVAL`, keeps the
token in the environment (never on the child argv), and runs `inspect eval`.
Anything after `--` is passed straight through to `inspect eval` (e.g.
`-- --limit 1`). `--max-runtime SECONDS` sets the per-trial wall clock (60 to
86400). `--report both` also reports the sequential and independent hop counts;
see [Both scoring modes](#both-scoring-modes-reportboth). `--parallel N` runs N
trials at once; see [Parallel runs](#parallel-runs-launch---parallel-n).
`--keep-range` keeps each scored range up until you close it; see
[Keep the range up](#keep-the-range-up-launch---keep-range). `--preseed FILE`
runs your own scripts on each range before the model starts; see
[Preseed the range](#preseed-the-range-launch---preseed).

### Share a transcript: `launch --share-transcript`

```bash
rogue-arena-inspect launch --eval <id> --model … --share-transcript
```

With `--share-transcript`, once the eval finishes, `launch` uploads the run's
`.eval` transcript to Rogue Arena, where it opens on the run's page. See
[What the client sends](#what-the-client-sends-to-rogue-arena) for what the file
holds and who can read it. It is **opt-in** and happens only on a clean finish: a log whose task errored or
was cancelled is left out, and `launch` says so. One upload per run.

Your local log stays where Inspect wrote it; `launch` tags the run with
`--metadata`, so it uploads the right `.eval` even if other runs share the
folder. It needs a local log folder, and looks where Inspect does: `--log-dir`
after `--`, then `INSPECT_LOG_DIR`, then `./logs`. The transcript is capped at
100 MB. The upload's idempotency key comes from the run and the file, so any retry
resumes or replays rather than duplicates. A problem sharing the transcript is
reported but never fails the eval, which has already run.

To upload an existing `.eval`, for example after that upload failed or for a run
you launched without the flag:

```bash
rogue-arena-inspect share-transcript <file.eval>             # launched with --share-transcript
rogue-arena-inspect share-transcript <file.eval> --run <id>  # any other run, from `runs`
```

A run over several models has one job per model, and each job holds one transcript,
so `launch` doesn't upload them for you: it prints the exact `share-transcript FILE
--run <id>` command for each log. Running one again for the same file resumes or
replays, never a second copy. A run
holds one transcript, so a different file is refused until the first is deleted on
the run page. Pass `--new-upload` when Rogue Arena asks for a new key: after an
integrity failure, or after the run's earlier transcript was deleted.

### Keep the range up: `launch --keep-range`

```bash
rogue-arena-inspect launch --eval <id> --model … --keep-range
rogue-arena-inspect close <run-id>   # when you're done with the range
```

By default Rogue Arena removes each trial's range as soon as the trial ends.
With `--keep-range`, a trial that ends **scored** (the scorer's close, or the
wall clock running out) leaves its range up after scoring, so you can look
around the machines the model worked on. The score is final when the range is
kept: nothing you do on the range afterwards changes it. A trial whose
[preseed](#preseed-the-range-launch---preseed) failed keeps its range too, so
you can see what your script did. A cancelled, failed or halted trial is still
removed at once.

A kept range is an ordinary lab you own. It **keeps billing** to the same
funding source the run used, and it follows the normal lab idle rules. Close it
when you're done:

- `rogue-arena-inspect close <run-id>` removes every range the run kept up
  (`<run-id>` is a run id from `runs`, or a trial id). Closing twice is fine.
  After a close, any trial of that run that ends later is removed as usual.
- Or remove it in Rogue Arena, as you would any lab.

`launch --keep-range` prints the `close` command for each run that kept its
range, and `runs` and `run <id>` show `range kept open (N up)` until you close
it. Only the run's owner can close it. The choice is made when the run starts
and can't be changed later. Without `launch`, set `keep_range` in the sandbox
config, or `ROGUE_ARENA_KEEP_RANGE=true` (see [Sandbox config](#sandbox-config)).
A Rogue Arena that does not support keep-range refuses the run before any trial
starts, so a range you asked to keep is never removed without warning.

### Preseed the range: `launch --preseed`

```bash
rogue-arena-inspect evals --dump-vms <eval>                 # the machines you can target
rogue-arena-inspect launch --eval <eval> --model … --preseed seed.yaml
```

A preseed runs your own bash (Linux) or PowerShell (Windows) scripts on a
trial's range **before the model starts**: plant files, change a
configuration, add users. Each step names one machine by its name or its stable
id from `evals --dump-vms`:

```yaml
steps:
  - target: WEB-01            # a machine name or machine id
    shell: bash               # bash (Linux) or powershell (Windows)
    timeout: 600              # seconds, optional (default 600, at most 3600)
    script: |
      echo "Quarterly numbers: see /srv/finance" > /opt/notes.txt
  - target: WS-01
    shell: powershell
    script: |
      New-Item -ItemType Directory -Force C:\Finance | Out-Null
      Set-Content C:\Finance\q3.txt "draft"
```

Rogue Arena runs the steps in order, as SYSTEM on Windows and root on Linux, on
every trial's range once the range is ready and before the model's first turn.
The trial's wall clock (`--max-runtime`) starts only after the last step, so
preseeding never spends the model's time. With `--parallel N`, each trial's
range runs the preseed once; a team run's one range runs it once.

A step that exits non-zero, times out or can't run ends that trial as
`preseed_failed`: it is **not scored**, the model never starts, and the step's
output (its last 8 KiB) is on the trial's log. `launch` prints it with the
failure. A step still running when its trial is cancelled runs until its
timeout or until the range is removed.

`launch` checks the file before `inspect eval` starts, and the run checks every
target against the eval's machines before it creates anything: a target the
eval doesn't list (a base machine, a hidden one, or a typo) or the wrong shell
for the machine's operating system fails the run before any range is built.
Rogue Arena checks the same again when each trial is claimed. Every trial of a
run carries the same preseed. Limits: at most 20 steps, and all scripts
together at most 1,000,000 characters. See
[What the client sends](#what-the-client-sends-to-rogue-arena) for what a
preseed sends.

Runs and trials with a preseed are labelled `preseeded` in `runs`, `run <id>`
and on the Model Testing page. Without `launch`, set `preseed` (a file path) in
the sandbox config, or `ROGUE_ARENA_PRESEED` (see [Sandbox config](#sandbox-config)).
A JSON file (`.json`) works too, and so does a bare list of steps.

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
| `max_runtime_seconds` | `ROGUE_ARENA_MAX_RUNTIME_SECONDS` | `3600` (60 to 86400) | Each trial's wall clock, counted from when its range is ready, so build time doesn't count. |
| `max_wait_seconds` | `ROGUE_ARENA_MAX_WAIT_SECONDS` | `11100` (3 h 5 min) | The backstop on a trial's range build. The provider waits while Rogue Arena reports the build in progress, and Rogue Arena ends a stalled build itself; the default only bounds a wait that never ends. |
| `heartbeat_interval_seconds` | `ROGUE_ARENA_HEARTBEAT_INTERVAL_SECONDS` | `300` | Lease heartbeat cadence (must be under 900). |
| `sync_exec_ceiling_seconds` | `ROGUE_ARENA_SYNC_EXEC_CEILING_SECONDS` | `50` (at most 55) | The longest command timeout that runs as one synchronous request. |
| `poll_interval_seconds` | `ROGUE_ARENA_POLL_INTERVAL_SECONDS` | `2` (at most 60) | Cadence of the ready, close and background-exec polls. |
| `team_visibility` | (from the team file) | `shared_sight` | A team run's activity-feed visibility: `shared_sight` or `communicate_only` (see Teams). |
| `keep_range` | `ROGUE_ARENA_KEEP_RANGE` | `false` | Keep each scored trial's range up, billed like any lab, until you close it with `rogue-arena-inspect close <run-id>`. See [Keep the range up](#keep-the-range-up-launch---keep-range). |
| `preseed` | `ROGUE_ARENA_PRESEED` | none | A preseed file (YAML or JSON) whose steps run on each trial's range before the model starts. The wait for a ready range grows by each step's timeout. See [Preseed the range](#preseed-the-range-launch---preseed). |

Explicit Inspect config wins over environment variables, which win over the
defaults. The config is typed, validated and frozen, and rejects unknown keys.

## Lifecycle

There is **one job per sandbox config and model per eval run**, and **one trial per
sample**:

1. **`task_init`** resolves `eval` and creates the job with `n`, `concurrency`,
   the wall clock, the model Inspect is evaluating, and an idempotency key (a transport failure retries once with
   the same key, so it never creates a second job). Tasks in the same run that
   share a sandbox config and model share that job.
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
| `TrialClosedBeforeReady` | Rogue Arena closed the trial while it was provisioning: a build that stopped progressing or ran too long, a failed hand-over check, or a failed preseed step (terminal reason `preseed_failed`, with the step's output in the message). The message (and `reason`) carries Rogue Arena's account of why. |
| `RogueArenaApiError` with HTTP 400 or 409 on a preseeded claim | Rogue Arena refused the preseed: a target that names no machine of the eval (or a base or hidden one), the wrong shell for its operating system (400), or a preseed that differs from the one the run's first trial was claimed with (409). |
| `httpx.TransportError`, or `RogueArenaApiError` with HTTP 502/503/504 | The claim and its one resend both went unanswered. The provider cancels any trial that was opened, then raises. |

### Large scenarios

A trial may build a fresh copy of the eval's range, and a large scenario's
build regularly takes more than 15 minutes. The provider judges the wait by
Rogue Arena's state, not by age: it keeps waiting while Rogue Arena reports the
trial `provisioning`. Rogue Arena watches the build's progress and ends a build
that stops progressing (currently after 30 minutes without progress) or runs too
long (currently 3 hours); the sample then fails with `TrialClosedBeforeReady` and
Rogue Arena's reason. `max_wait_seconds` (default 3 h 5 min) is only a backstop
for a wait that never ends; set it (or `ROGUE_ARENA_MAX_WAIT_SECONDS`) to cap the
wait yourself.

**Build time doesn't spend the agent's time.** Each trial's wall clock
(`max_runtime_seconds`, default 3600) starts when its range is ready, so size
`--max-runtime` for the agent's work alone.

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
| Terminal reason `preseed_failed` | Never reaches the scorer: the claim fails with `TrialClosedBeforeReady`, so the sample errors and never enters the mean. |

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
`get_eval_guide`, `list_eval_machines`, `list_runs`, `create_job` (with
`keep_range=`), `get_job`, `cancel_job`, `close_range`, `list_job_trials`,
`claim_trial` (with `preseed=`), `get_trial`, `heartbeat_trial`, `run_command`
(and the single-request `exec_trial`), `upload_file`, `download_file`,
`close_trial`, `list_trial_events` and `share_transcript`.

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

### 0.6.0 (2026-11-18)

- A trial's wall clock now starts when its range is ready, so a slow build no
  longer spends the agent's time. The `max_wait_seconds` backstop's default is
  now 3 h 5 min.
- Each job records the model it evaluates (shown by `runs` and `run <id>`), and a
  run over several models gets one job per model, instead of one job showing no
  model.
- CLI `launch --share-transcript` uploads a finished run's `.eval` transcript to
  Rogue Arena (opt-in, once per run). `share-transcript FILE` uploads an existing `.eval`. Client: `RogueArenaClient.share_transcript()`.
- CLI `eval <id> --guide` shows an eval's graders, solution writeup and recommended
  prompts; `--prompt-level easy|medium|hard` prints one prompt, and
  `launch --prompt-level` runs with it.
- The client refuses a base URL that is not `https://` (plain `http://` only
  to `localhost`), so the token is never sent in clear.
- Against a server that does not offer eval guides, `eval --guide`,
  `eval --prompt-level` and `launch --prompt-level` now say so, after checking
  that the token works, instead of reporting a rejected token.
- **Preseed.** `launch --preseed seed.yaml` (or `preseed` /
  `ROGUE_ARENA_PRESEED`) runs your own bash or PowerShell steps on each trial's
  range before the model starts; the wall clock starts after them. A failed
  step ends the trial `preseed_failed`, unscored, with the step's output shown.
  Runs and trials with a preseed are labelled `preseeded`. Client:
  `claim_trial(preseed=…)`. See [Preseed the range](#preseed-the-range-launch---preseed).
- **An eval's machines.** `evals --dump-vms <id>` lists the machines a preseed
  step can target: name, stable machine id and operating system, with the base
  machines every range carries left out. Client: `list_eval_machines()`.
- **Keep the range up.** `launch --keep-range` (or `keep_range` /
  `ROGUE_ARENA_KEEP_RANGE`) keeps each scored trial's range up after scoring,
  billed and idled like any lab, until you close it; a failed preseed's range is
  kept too. `runs` and `run <id>` show `range kept open (N up)`. Client:
  `create_job(keep_range=True)`. See [Keep the range up](#keep-the-range-up-launch---keep-range).
- **`close <run-id>`** removes the ranges a keep-range run kept up. Client:
  `close_range()`.
- `pyyaml` is now a declared dependency (team configs and preseed files are
  YAML).
- [What the client sends](#what-the-client-sends-to-rogue-arena) now lists the
  model's name, describes what a shared transcript holds and who can read it,
  and covers preseed scripts and kept ranges.

### 0.5.1

- **Fix: a fresh install could not run an eval.** `agent-client-protocol`
  1.0.0 removed a module that every supported Inspect version imports, so a
  new install failed with `No module named 'acp.helpers'`. The package now
  requires `agent-client-protocol>=0.12.0,<1`.

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
