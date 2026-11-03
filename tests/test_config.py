from __future__ import annotations

import unittest

from pydantic import ValidationError

from inspect_rogue_arena.config import (
    EVAL_HEARTBEAT_GRACE_SECONDS,
    RogueArenaSandboxConfig,
)


class RogueArenaConfigTests(unittest.TestCase):
    def test_config_is_frozen_and_rejects_unknown_keys(self) -> None:
        config = RogueArenaSandboxConfig(eval="env-1")
        with self.assertRaises(ValidationError):
            config.eval = "env-2"  # type: ignore[misc]
        with self.assertRaises(ValidationError):
            RogueArenaSandboxConfig(not_a_field="x")  # type: ignore[call-arg]

    def test_explicit_config_wins_over_environment(self) -> None:
        resolved = RogueArenaSandboxConfig.from_sources(
            {"eval": "explicit-env"},
            environ={"ROGUE_ARENA_EVAL": "env-var-env", "ROGUE_ARENA_MAX_TRIALS": "9"},
        )
        self.assertEqual(resolved.eval, "explicit-env")
        self.assertEqual(resolved.max_trials, 9)

    def test_env_supplies_the_eval_and_timing(self) -> None:
        resolved = RogueArenaSandboxConfig.from_sources(
            None,
            environ={
                "ROGUE_ARENA_EVAL": "  env-7  ",
                "ROGUE_ARENA_MAX_RUNTIME_SECONDS": "1800",
                "ROGUE_ARENA_CONCURRENCY": "3",
            },
        )
        self.assertEqual(resolved.eval, "env-7")
        self.assertEqual(resolved.max_runtime_seconds, 1800)
        self.assertEqual(resolved.concurrency, 3)

    def test_team_visibility_defaults_and_comes_from_the_team_file(self) -> None:
        import tempfile
        from pathlib import Path

        self.assertEqual(
            RogueArenaSandboxConfig.from_sources(None, environ={}).team_visibility,
            "shared_sight",
        )
        with tempfile.TemporaryDirectory() as directory:
            team = Path(directory) / "team.yaml"
            team.write_text(
                "turn_taking: round_robin\nvisibility: communicate_only\nmembers:\n"
                "  - {name: a, model: m, prompt: p}\n  - {name: b, model: m, prompt: p}\n",
                encoding="utf-8",
            )
            # The run's visibility comes from the one team file the run loads.
            resolved = RogueArenaSandboxConfig.from_sources(
                None, environ={"ROGUE_ARENA_TEAM": str(team)}
            )
            self.assertEqual(resolved.team_visibility, "communicate_only")
            # An explicit sandbox config value still wins over the file.
            explicit = RogueArenaSandboxConfig.from_sources(
                {"team_visibility": "shared_sight"},
                environ={"ROGUE_ARENA_TEAM": str(team)},
            )
            self.assertEqual(explicit.team_visibility, "shared_sight")

    def test_heartbeat_interval_must_be_below_the_lease_grace(self) -> None:
        self.assertLess(
            RogueArenaSandboxConfig().heartbeat_interval_seconds,
            EVAL_HEARTBEAT_GRACE_SECONDS,
        )
        with self.assertRaises(ValidationError):
            RogueArenaSandboxConfig(
                heartbeat_interval_seconds=EVAL_HEARTBEAT_GRACE_SECONDS + 1
            )

    def test_sync_exec_ceiling_stays_under_the_gateway_timeout(self) -> None:
        self.assertLessEqual(RogueArenaSandboxConfig().sync_exec_ceiling_seconds, 55)
        with self.assertRaises(ValidationError):
            RogueArenaSandboxConfig(sync_exec_ceiling_seconds=90)


if __name__ == "__main__":
    unittest.main()
