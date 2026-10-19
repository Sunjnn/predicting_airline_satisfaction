"""Local smoke check: execute the notebook source on a 50k-row sample.

Run: .venv\\Scripts\\python.exe src\\smoke_check.py

The real run happens on Kaggle (see kaggle/kernel-metadata.json). This only
proves the pipeline executes end to end: feature engineering, both CV loops,
the blend, and submission.csv formatting.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "airline_satisfaction.py"
PATCHED = ROOT / ".cache" / "smoke_run.py"


def main() -> int:
    code = SRC.read_text(encoding="utf-8")
    if 'RUN_MODE = "full"' not in code:
        raise SystemExit("RUN_MODE assignment not found; update this patch when the config cell changes")
    PATCHED.parent.mkdir(parents=True, exist_ok=True)
    PATCHED.write_text(code.replace('RUN_MODE = "full"', 'RUN_MODE = "smoke"'), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(PATCHED)], cwd=ROOT)
    if proc.returncode != 0:
        return proc.returncode

    sub = ROOT / "outputs" / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"expected submission at {sub}")
    print(f"\nsmoke check OK -> {sub.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
