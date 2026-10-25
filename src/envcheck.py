"""Check that the project runs in its own virtual environment with the pinned package versions.

Other projects on the same machine may need different versions of torch, Pillow,
numpy and so on. This project avoids conflicts by living in ``.venv``. Run
``python -m src.envcheck`` to see whether the current interpreter matches
``requirements.txt``; the pipeline and the API print the same warning at start.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"
_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)")


def pinned_versions(path: Path = REQUIREMENTS) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def mismatches(pins: dict[str, str] | None = None) -> list[tuple[str, str, str | None]]:
    """Return (package, required version, installed version or None) for every difference."""
    problems = []
    for name, required in sorted((pins or pinned_versions()).items()):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        # Local build tags such as "+cpu" do not matter for compatibility.
        if installed is None or installed.split("+")[0] != required:
            problems.append((name, required, installed))
    return problems


def report() -> list[str]:
    lines = []
    if not in_virtualenv():
        lines.append(f"Not running inside a virtual environment ({sys.executable}). "
                     "Activate the project .venv first (see README.md, 'Install').")
    for name, required, installed in mismatches():
        lines.append(f"{name}: requirements.txt pins {required}, installed {installed or 'nothing'}")
    return lines


def warn_if_needed() -> None:
    """Print a short warning to stderr; never stops the program."""
    lines = report()
    if lines:
        print("Environment warning (run 'python -m src.envcheck' for details):\n  " + "\n  ".join(lines[:6]),
              file=sys.stderr)


def main() -> int:
    lines = report()
    print(f"Python: {sys.executable}")
    if not lines:
        print("OK: running in a virtual environment and every pinned package matches requirements.txt.")
        return 0
    print("\n".join(lines))
    print("\nFix: activate .venv and run 'python -m pip install -r requirements.txt'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
