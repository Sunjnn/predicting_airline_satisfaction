"""Convert a jupytext percent-format source into a Kaggle-ready .ipynb.

Run:
  .venv\\Scripts\\python.exe src\\make_notebook.py                  # default notebook
  .venv\\Scripts\\python.exe src\\make_notebook.py --src src\\foo.py --dst kaggle\\foo.ipynb

Implemented directly instead of shelling out to jupytext: jupytext/nbformat
writes signature state into the user profile, which the project sandbox blocks.
The percent format is small enough to parse here, and this also syntax-checks
every code cell while converting.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "src" / "airline_satisfaction.py"
DEFAULT_DST = ROOT / "kaggle" / "airline_satisfaction.ipynb"

CELL_MARKER = re.compile(r"^#\s*%%(?:\s*\[(?P<kind>\w+)\])?\s*$")


def _strip_comment(line: str) -> str:
    """Markdown cells live in the .py source as commented lines: '# text'."""
    if line.strip() == "#":
        return ""
    return re.sub(r"^#\s?", "", line, count=1)


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    kind = "code"
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        while buf and not buf[0].strip():
            buf.pop(0)
        while buf and not buf[-1].strip():
            buf.pop()
        if not buf:
            buf = []
            return
        if kind == "markdown":
            source = [_strip_comment(line) + "\n" for line in buf]
        else:
            source = [line + "\n" for line in buf]
        cells.append({"kind": kind, "source": source})
        buf = []

    for line in text.splitlines():
        m = CELL_MARKER.match(line)
        if m:
            flush()
            kind = m.group("kind") or "code"
            continue
        buf.append(line)
    flush()
    return cells


def build_notebook(cells: list[dict]) -> dict:
    nb_cells = []
    for i, cell in enumerate(cells):
        source = cell["source"]
        if source and source[-1].endswith("\n"):
            source[-1] = source[-1][:-1]  # nbformat: no trailing newline on last line
        if cell["kind"] == "markdown":
            nb_cells.append(
                {
                    "cell_type": "markdown",
                    "id": f"md-{i:02d}",
                    "metadata": {},
                    "source": source,
                }
            )
        else:
            compile("".join(source), f"<cell {i}>", "exec")  # syntax check
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": f"code-{i:02d}",
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    text = src.read_text(encoding="utf-8")
    nb = build_notebook(parse_cells(text))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    try:
        shown = dst.relative_to(ROOT)
    except ValueError:
        shown = dst
    print(f"wrote {shown}: {len(nb['cells'])} cells ({code_cells} code, {md_cells} markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
