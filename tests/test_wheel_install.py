"""Install the built wheel into a fresh venv and resolve the shipped example.

This proves ``launch`` works from a pip install (not only a source checkout):
the example task files must ship inside the package and resolve through
``importlib.resources``. It builds a wheel, installs it into a throwaway venv
with no source on the path, and resolves the example there.

It is slow and needs ``uv``, so it runs only when
``INSPECT_ROGUE_ARENA_RUN_WHEEL_TEST=1`` is set and ``uv`` is available; the
default offline suite skips it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_ENABLED = os.environ.get("INSPECT_ROGUE_ARENA_RUN_WHEEL_TEST") == "1"

# In the installed wheel, with no source tree on sys.path, resolve the example
# the way the CLI does and confirm it is a real file.
_RESOLVE_SNIPPET = (
    "from importlib import resources;"
    "import inspect_rogue_arena;"
    "p = resources.files('inspect_rogue_arena').joinpath('examples', 'eval_example.py');"
    "assert p.is_file(), p;"
    "print('EXAMPLE_OK')"
)


@unittest.skipUnless(_ENABLED, "set INSPECT_ROGUE_ARENA_RUN_WHEEL_TEST=1 to run")
@unittest.skipUnless(
    shutil.which("uv"), "uv is required to build and install the wheel"
)
class WheelInstallTests(unittest.TestCase):
    def test_launch_example_resolves_from_an_installed_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            subprocess.run(
                ["uv", "build", "--wheel", "--out-dir", str(out), str(REPO_ROOT)],
                check=True,
                capture_output=True,
            )
            wheels = list(out.glob("*.whl"))
            self.assertEqual(len(wheels), 1, wheels)

            venv = out / "venv"
            subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True)
            python = venv / "bin" / "python"
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), str(wheels[0])],
                check=True,
                capture_output=True,
            )

            # Run from a directory with no package source, isolated from env.
            resolved = subprocess.run(
                [str(python), "-I", "-c", _RESOLVE_SNIPPET],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("EXAMPLE_OK", resolved.stdout)

            cli = subprocess.run(
                [str(venv / "bin" / "rogue-arena-inspect"), "--help"],
                cwd=tmp,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("launch", cli.stdout)


if __name__ == "__main__":
    unittest.main()
