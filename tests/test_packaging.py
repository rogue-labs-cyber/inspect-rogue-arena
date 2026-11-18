from __future__ import annotations

import re
import unittest
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


class PackagingTests(unittest.TestCase):
    def test_agent_client_protocol_stays_below_1(self) -> None:
        # agent-client-protocol 1.0.0 removed acp.helpers, which every supported
        # Inspect imports when `inspect eval` starts: without the upper bound a
        # fresh install crashes with "No module named 'acp.helpers'" (0.5.1).
        text = PYPROJECT.read_text(encoding="utf-8")
        match = re.search(r'"agent-client-protocol([^"]*)"', text)
        assert match is not None, "pyproject.toml must require agent-client-protocol"
        self.assertIn("<1", match.group(1).replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
