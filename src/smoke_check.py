"""Local smoke check: run a notebook source on a 50k-row sample.

Usage:
  .venv\\Scripts\\python.exe src\\smoke_check.py                       # main notebook
  .venv\\Scripts\\python.exe src\\smoke_check.py --src src\\airline_fe_lgbm.py

The real runs happen on Kaggle. This only proves the pipeline executes end to end:
feature engineering, CV loop, and submission formatting.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "src" / "airline_satisfaction.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--name", default=None, help="name of the patched copy under .cache/")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    name = args.name or src.stem
    patched = ROOT / ".cache" / f"smoke_{name}.py"

    code = src.read_text(encoding="utf-8")
    if 'RUN_MODE = "full"' not in code:
        raise SystemExit(f"RUN_MODE assignment not found in {src}; update this patch when the config cell changes")
    patched.parent.mkdir(parents=True, exist_ok=True)
    patched.write_text(code.replace('RUN_MODE = "full"', 'RUN_MODE = "smoke"'), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(patched)], cwd=ROOT)
    if proc.returncode != 0:
        return proc.returncode

    sub = ROOT / "outputs" / "submission.csv"
    if not sub.exists():
        raise SystemExit(f"expected submission at {sub}")
    print(f"\nsmoke check OK ({src.name}) -> {sub.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
