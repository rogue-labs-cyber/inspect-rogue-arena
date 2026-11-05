from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Mirrors the hub's heartbeat grace period. A heartbeat must
# land inside this window or the server lets the lease lapse and reaps the trial.
EVAL_HEARTBEAT_GRACE_SECONDS = 900
# Mirrors the hub's hard exec ceiling.
EVAL_MAX_EXEC_SECONDS = 4 * 60 * 60
# The provisioning backstop's floor: a cold build of a large scenario regularly
# takes over 15 minutes.
DEFAULT_MAX_WAIT_SECONDS = 2 * 60 * 60
# How far the default backstop reaches past the trial's wall clock, so the hub's
# own wall-clock expiry (on its next maintenance pass) always lands first.
READY_WAIT_WALL_CLOCK_MARGIN_SECONDS = 5 * 60


class RogueArenaSandboxConfig(BaseModel):
    """Frozen, validated configuration for the Rogue Arena sandbox provider.

    Explicit Inspect config wins over environment variables, which win over the
    built-in defaults. The token is intentionally environment-only
    (ROGUE_ARENA_TOKEN); it is never a config field so it is never serialized
    into an Inspect log.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # One selector: an eval environment id (uuid) or a unique name matched
    # against `list_evals()`.
    eval: str | None = None
    # The backstop on a claimed trial's provisioning (see ready_wait_seconds).
    # None (the default) derives it from the wall clock. It does not bound the
    # wait for a concurrency slot.
    max_wait_seconds: float | None = Field(default=None, gt=0)
    max_runtime_seconds: int = Field(default=3600, ge=60, le=86_400)
    poll_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    heartbeat_interval_seconds: float = Field(
        default=300.0, gt=0, lt=EVAL_HEARTBEAT_GRACE_SECONDS
    )
    # A command whose timeout is at most this runs as one synchronous request;
    # a longer timeout runs in the background and is polled.
    sync_exec_ceiling_seconds: int = Field(default=50, gt=0, le=55)
    max_trials: int = Field(default=10_000, gt=0)
    concurrency: int = Field(default=2, gt=0)
    # The team run's activity-feed visibility, stored on the hub job at create:
    # "shared_sight" (members read the feed) or "communicate_only" (the feed is
    # recorded, but the hub refuses a member's read while the run is live). The
    # default keeps a non-team run's feed readable.
    team_visibility: Literal["shared_sight", "communicate_only"] = "shared_sight"

    @property
    def ready_wait_seconds(self) -> float:
        """How long the client waits for a claimed trial to become ready.

        The client keeps waiting while the hub reports the trial
        provisioning: the hub judges build progress itself and expires a copy
        that stops progressing or outruns the trial's wall clock, so this only
        bounds a wait the hub never ends. ``max_wait_seconds`` when set; else
        the longer of 2 hours and the wall clock plus 5 minutes, so the client
        never gives up on a build the hub would still allow.
        """
        if self.max_wait_seconds is not None:
            return self.max_wait_seconds
        return float(
            max(
                DEFAULT_MAX_WAIT_SECONDS,
                self.max_runtime_seconds + READY_WAIT_WALL_CLOCK_MARGIN_SECONDS,
            )
        )

    @field_validator("eval", mode="before")
    @classmethod
    def _strip_optional_string(cls, value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        return value.strip() or None

    @classmethod
    def from_sources(
        cls,
        config: RogueArenaSandboxConfig | Mapping[str, Any] | None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> RogueArenaSandboxConfig:
        source_env = os.environ if environ is None else environ
        env_map = {
            "eval": "ROGUE_ARENA_EVAL",
            "max_wait_seconds": "ROGUE_ARENA_MAX_WAIT_SECONDS",
            "max_runtime_seconds": "ROGUE_ARENA_MAX_RUNTIME_SECONDS",
            "poll_interval_seconds": "ROGUE_ARENA_POLL_INTERVAL_SECONDS",
            "heartbeat_interval_seconds": "ROGUE_ARENA_HEARTBEAT_INTERVAL_SECONDS",
            "sync_exec_ceiling_seconds": "ROGUE_ARENA_SYNC_EXEC_CEILING_SECONDS",
            "max_trials": "ROGUE_ARENA_MAX_TRIALS",
            "concurrency": "ROGUE_ARENA_CONCURRENCY",
        }
        env_values: dict[str, Any] = {}
        for field_name, env_name in env_map.items():
            value = source_env.get(env_name)
            if value is not None and value.strip() != "":
                env_values[field_name] = value

        if isinstance(config, cls):
            explicit = {
                field: getattr(config, field) for field in config.model_fields_set
            }
        elif config is None:
            explicit = {}
        else:
            explicit = dict(config)

        # A team run's visibility comes from the team config file the run loads
        # (ROGUE_ARENA_TEAM, set by `launch --team` and read by the team
        # example). An explicit sandbox config value still wins.
        team_path = source_env.get("ROGUE_ARENA_TEAM")
        if team_path and team_path.strip() and "team_visibility" not in explicit:
            from inspect_rogue_arena.team import load_team_config

            env_values["team_visibility"] = load_team_config(team_path).visibility.value

        return cls.model_validate({**env_values, **explicit})
