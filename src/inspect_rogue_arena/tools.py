"""The shared Rogue Arena ``bash`` tool.

Both the solo task and a team member drive the agent VM through the same shell
tool: it states its default timeout to the model, takes an optional ``timeout``
argument, and runs any command whose timeout exceeds the sandbox's synchronous
ceiling as a background exec that the sandbox polls until it finishes. A long
scan, build or download therefore runs to completion and returns its full
output, and a transient gateway/transport failure on a poll is retried rather
than ending the run.

The solo task uses ``rogue_bash``; a team member wraps the same
``build_bash_tool_def`` with its own session ``channel`` and its activity-feed
hooks (see ``team.member_bash``).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from inspect_ai.tool import Tool, ToolDef, ToolError, tool
from inspect_ai.util import ExecResult

logger = logging.getLogger(__name__)

# The default command timeout. It sits above the sandbox's sync-exec ceiling
# (RogueArenaSandboxConfig.sync_exec_ceiling_seconds, default 50), so by default
# every command runs as a background exec the sandbox polls until it finishes.
DEFAULT_BASH_TIMEOUT_SECONDS = 600

# Hooks a caller (a team member) may attach. They run for their side effects
# only; the command result the model sees is unchanged.
OnSuccess = Callable[[str, ExecResult[str]], Awaitable[None]]
OnTimeout = Callable[[str, int], Awaitable[None]]


def rogue_bash_description(default_timeout: int) -> str:
    """The bash tool's description: its timeout behavior, stated to the model."""
    return (
        "Use this function to execute bash commands. A command runs to completion "
        "and returns its full output, for at most `timeout` seconds (default "
        f"{default_timeout}); one that runs longer is stopped, with everything it "
        "started, and reported as timed out. Pass a larger `timeout` for a longer "
        "scan, build or download, or a smaller one for a command that may hang."
    )


def build_bash_tool_def(
    *,
    channel: str | None = None,
    default_timeout: int = DEFAULT_BASH_TIMEOUT_SECONDS,
    on_success: OnSuccess | None = None,
    on_timeout: OnTimeout | None = None,
) -> ToolDef:
    """Build the Rogue Arena bash tool.

    ``channel`` picks a sandbox session (a team member's own shell into the one
    shared VM); ``None`` uses the default sandbox. ``on_success`` runs after a
    command returns, ``on_timeout`` after it times out; both are optional and used
    only by the team path to record its activity feed. ``on_timeout`` is
    best-effort: a failure in it is logged, never raised, so it can never mask
    the timeout the model needs to see.
    """

    async def execute(command: str, timeout: int | None = None) -> str:
        """Use this function to execute bash commands.

        Args:
          command: The bash command to execute.
          timeout: Seconds the command may run (default: the tool's timeout).

        Returns:
          The output of the command.
        """
        from inspect_ai.util import sandbox as active_sandbox

        if timeout is not None and timeout <= 0:
            raise ToolError("timeout must be a positive number of seconds")
        seconds = default_timeout if timeout is None else timeout
        try:
            # A timeout above the sandbox's sync ceiling (the default is) runs as
            # a background exec the sandbox polls until it finishes
            # (RogueArenaClient.run_command), so a long scan, build or download
            # returns its full output and a transient poll failure is retried.
            result = await active_sandbox(channel).exec(
                cmd=["bash", "--login", "-c", command], timeout=seconds
            )
        except TimeoutError:
            if on_timeout is not None and command.strip():
                try:
                    await on_timeout(command, seconds)
                except Exception as error:  # noqa: BLE001 - never mask the timeout
                    logger.warning("bash timeout hook failed: %s", error)
            raise
        if on_success is not None:
            await on_success(command, result)
        output = f"{result.stderr}\n" if result.stderr else ""
        return f"{output}{result.stdout}"

    # Presented to the model as the ordinary ``bash`` tool, so model habits (and
    # a member prompt that says "your own shell via the bash tool") still apply.
    # The description is where the model learns its command timeout.
    return ToolDef(
        execute,
        name="bash",
        description=rogue_bash_description(default_timeout),
        parameters={
            "command": "The bash command to execute.",
            "timeout": (
                f"Seconds the command may run (default {default_timeout}). "
                "Pass a larger value for a longer scan, build or download."
            ),
        },
    )


@tool(parallel=True)
def rogue_bash(default_timeout: int = DEFAULT_BASH_TIMEOUT_SECONDS) -> Tool:
    """The Rogue Arena bash tool for a solo task (one agent, the default VM).

    A drop-in replacement for Inspect's ``bash`` whose default timeout runs
    commands through the sandbox's background-and-poll path, so a long command
    runs to completion and a transient gateway/transport failure does not end the
    run. The model may pass its own ``timeout`` per command.
    """
    return build_bash_tool_def(default_timeout=default_timeout).as_tool()
