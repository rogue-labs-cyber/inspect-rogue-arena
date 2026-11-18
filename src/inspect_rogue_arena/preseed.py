"""Preseed: your own scripts, run on a trial's range before the model starts.

A preseed file (YAML or JSON) lists steps. Each step runs one bash (Linux) or
PowerShell (Windows) script on one machine of the eval, named by its name or
its stable machine id (``rogue-arena-inspect evals --dump-vms <eval>`` lists
them). Rogue Arena runs the steps in order on every trial's range once it is
ready and before the model's first turn; the trial's wall clock starts after
the last step passes. A step that exits non-zero or times out ends the trial
unscored (``preseed_failed``) with the step's output on the trial's log.

    steps:
      - target: WEB-01            # a machine name or machine id
        shell: bash               # bash (Linux) or powershell (Windows)
        timeout: 600              # seconds; optional, default 600
        script: |
          echo "seeded" > /opt/notes.txt

The limits mirror the hub's (it checks them again): at most 20 steps, each
timeout 1 to 3600 seconds, and all scripts together at most 1,000,000
characters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

# `launch --preseed` sets it; any `inspect eval` run can set it too (a file path).
PRESEED_ENV = "ROGUE_ARENA_PRESEED"

# Mirror Rogue Arena's limits (it checks them again).
PRESEED_MAX_STEPS = 20
PRESEED_SCRIPT_MAX_CHARS = 1_000_000
PRESEED_DEFAULT_TIMEOUT_SECONDS = 600
PRESEED_MAX_TIMEOUT_SECONDS = 3_600
PRESEED_TARGET_MAX = 200
# Past a step's timeout, how long the hub may still take to answer it (the
# agent's readiness probe, copying the script over, the transport's own wait).
PRESEED_STEP_ALLOWANCE_SECONDS = 120

_STEP_KEYS = frozenset({"target", "shell", "timeout", "script"})


class PreseedShell(str, Enum):
    BASH = "bash"
    POWERSHELL = "powershell"


class PreseedError(ValueError):
    """The preseed file is not a valid preseed; the message says where."""


@dataclass(frozen=True, slots=True)
class PreseedStep:
    target: str
    shell: PreseedShell
    timeout_seconds: int
    script: str

    def to_wire(self) -> dict[str, str | int]:
        return {
            "target": self.target,
            "shell": self.shell.value,
            "timeoutSeconds": self.timeout_seconds,
            "script": self.script,
        }


@dataclass(frozen=True, slots=True)
class Preseed:
    steps: tuple[PreseedStep, ...]

    def to_wire(self) -> list[dict[str, str | int]]:
        """The claim body's ``preseed`` field."""
        return [step.to_wire() for step in self.steps]

    @property
    def wait_seconds(self) -> int:
        """The longest the hub may take to run every step, on top of provisioning."""
        return sum(
            step.timeout_seconds + PRESEED_STEP_ALLOWANCE_SECONDS for step in self.steps
        )


def _step(raw: Any, number: int) -> PreseedStep:
    where = f"step {number}"
    if not isinstance(raw, Mapping):
        raise PreseedError(f"{where}: must be a mapping with target, shell and script")
    unknown = sorted(str(key) for key in raw if key not in _STEP_KEYS)
    if unknown:
        raise PreseedError(
            f"{where}: unknown field(s) {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(_STEP_KEYS))})"
        )
    target = raw.get("target")
    if not isinstance(target, str) or not target.strip():
        raise PreseedError(f"{where}: target must be a machine name or machine id")
    if len(target.strip()) > PRESEED_TARGET_MAX:
        raise PreseedError(
            f"{where}: target is longer than {PRESEED_TARGET_MAX} characters"
        )
    shell = raw.get("shell")
    try:
        parsed_shell = PreseedShell(shell)
    except ValueError:
        raise PreseedError(
            f"{where}: shell must be bash (Linux) or powershell (Windows), not {shell!r}"
        ) from None
    timeout = raw.get("timeout", PRESEED_DEFAULT_TIMEOUT_SECONDS)
    # bool is an int in Python: `timeout: true` is not a number of seconds.
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise PreseedError(f"{where}: timeout must be a whole number of seconds")
    if not 1 <= timeout <= PRESEED_MAX_TIMEOUT_SECONDS:
        raise PreseedError(
            f"{where}: timeout must be 1 to {PRESEED_MAX_TIMEOUT_SECONDS} seconds"
        )
    script = raw.get("script")
    if not isinstance(script, str) or not script.strip():
        raise PreseedError(f"{where}: script must be a non-empty string")
    if len(script) > PRESEED_SCRIPT_MAX_CHARS:
        raise PreseedError(
            f"{where}: script is longer than {PRESEED_SCRIPT_MAX_CHARS} characters"
        )
    return PreseedStep(
        target=target.strip(),
        shell=parsed_shell,
        timeout_seconds=timeout,
        script=script,
    )


def parse_preseed(raw: Any) -> Preseed:
    """Validate a parsed preseed document: ``{steps: [...]}`` or a bare list of steps."""
    steps_raw = raw.get("steps") if isinstance(raw, Mapping) else raw
    if isinstance(raw, Mapping):
        unknown = sorted(str(key) for key in raw if key != "steps")
        if unknown:
            raise PreseedError(
                f"unknown top-level field(s) {', '.join(unknown)} (allowed: steps)"
            )
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise PreseedError("a preseed is a list of steps (or a mapping with `steps:`)")
    if not steps_raw:
        raise PreseedError("a preseed needs at least one step")
    if len(steps_raw) > PRESEED_MAX_STEPS:
        raise PreseedError(
            f"a preseed has at most {PRESEED_MAX_STEPS} steps, not {len(steps_raw)}"
        )
    steps = tuple(_step(step, number) for number, step in enumerate(steps_raw, start=1))
    total = sum(len(step.script) for step in steps)
    if total > PRESEED_SCRIPT_MAX_CHARS:
        raise PreseedError(
            f"the scripts add up to {total} characters; at most "
            f"{PRESEED_SCRIPT_MAX_CHARS} are allowed"
        )
    return Preseed(steps=steps)


class _TargetMachine(Protocol):
    """What a target check reads of a listed machine (``client.EvalMachine``)."""

    @property
    def machine_id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def operating_system(self) -> str: ...


def check_preseed_targets(preseed: Preseed, machines: Sequence[_TargetMachine]) -> None:
    """Fail fast on a target that names no machine ``evals --dump-vms`` lists.

    ``machines`` are the eval's targetable machines (``EvalMachine``: name,
    machine id, operating system). A target matches a machine's id, or its name
    (any case) when exactly one machine has it; the step's shell must be the
    machine's (PowerShell on Windows, bash otherwise). Rogue Arena checks the
    same again when each trial is claimed.
    """
    for number, step in enumerate(preseed.steps, start=1):
        where = f"preseed step {number}: target {step.target!r}"
        by_id = [m for m in machines if m.machine_id == step.target]
        named = by_id or [m for m in machines if m.name.lower() == step.target.lower()]
        if not named:
            raise PreseedError(
                f"{where} names no machine of this eval a preseed can target; "
                "list them with `rogue-arena-inspect evals --dump-vms <eval>`"
            )
        if len(named) > 1:
            raise PreseedError(
                f"{where} names {len(named)} machines; use its machine id"
            )
        windows = named[0].operating_system.lower().startswith("windows")
        shell = PreseedShell.POWERSHELL if windows else PreseedShell.BASH
        if step.shell is not shell:
            raise PreseedError(
                f"{where} runs {named[0].operating_system}, so its step must use "
                f"{shell.value}, not {step.shell.value}"
            )


def load_preseed(path: str | Path) -> Preseed:
    """Read and validate a preseed file (YAML, or JSON by its .json suffix)."""
    file = Path(path).expanduser()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise PreseedError(
            f"cannot read preseed file {file}: {error.strerror or error}"
        ) from None
    try:
        if file.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            import yaml  # a declared dependency (pyyaml>=6)

            raw = yaml.safe_load(text)
    except Exception as error:  # noqa: BLE001 - any parse failure is the file's
        raise PreseedError(f"preseed file {file} does not parse: {error}") from None
    try:
        return parse_preseed(raw)
    except PreseedError as error:
        raise PreseedError(f"preseed file {file}: {error}") from None
