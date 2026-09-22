#!/usr/bin/env python
"""Structural checks on the manuscript.

Not a substitute for running LaTeX -- it catches the failures that are cheap to
catch: unmatched environments, \\input targets that do not exist, citation keys
with no bib entry, labels referenced but never defined, and leftover \\todo
markers.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"


def main() -> int:
    tex = (PAPER / "main.tex").read_text()
    problems: list[str] = []

    # environments
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            problems.append(
                f"environment {env!r}: {begins.count(env)} begin vs {ends.count(env)} end")

    # braces
    depth = 0
    for i, ch in enumerate(tex):
        if ch == "{" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1
        elif ch == "}" and (i == 0 or tex[i - 1] != "\\"):
            depth -= 1
            if depth < 0:
                problems.append(f"unbalanced closing brace at offset {i}")
                break
    if depth > 0:
        problems.append(f"{depth} unclosed brace(s)")

    # \input targets
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    for rel in inputs:
        f = PAPER / (rel if rel.endswith(".tex") else rel + ".tex")
        if not f.exists():
            problems.append(f"\\input target missing: {rel}")

    # citations
    bib = (PAPER / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for m in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
        cited |= {c.strip() for c in m.split(",")}
    for c in sorted(cited - keys):
        problems.append(f"cited but not in refs.bib: {c}")
    unused = sorted(keys - cited)

    # labels / refs -- labels also live in the generated \input files
    all_tex = tex
    for rel in inputs:
        f = PAPER / (rel if rel.endswith(".tex") else rel + ".tex")
        if f.exists():
            all_tex += "\n" + f.read_text()
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_tex))
    refs = set(re.findall(r"\\(?:ref|autoref)\{([^}]+)\}", tex))
    for r in sorted(refs - labels):
        problems.append(f"\\ref to an undefined label: {r}")

    todos = re.findall(r"\\todo\{([^}]*)\}", tex)
    pending = [f for f in inputs
               if (PAPER / (f if f.endswith('.tex') else f + '.tex')).exists()
               and "pending" in (PAPER / (f if f.endswith('.tex') else f + '.tex')).read_text()]

    print(f"main.tex: {len(tex.splitlines())} lines, {len(inputs)} \\input, "
          f"{len(cited)} citations, {len(labels)} labels")
    if unused:
        print(f"note: {len(unused)} bib entries not cited: {', '.join(unused)}")
    if todos:
        print(f"note: {len(todos)} \\todo marker(s): {todos}")
    if pending:
        print(f"note: {len(pending)} table(s) still awaiting results: "
              f"{', '.join(pending)}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nstructure OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
