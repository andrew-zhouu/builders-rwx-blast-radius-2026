"""Outline of the pinned tree's Python files: the line span of every module-level and
class-level statement (functions and methods included, nested classes followed).

The selector uses it to parse just the statement a patch touches instead of the whole file.

Usage: python3 build_outline.py <sqlglot checkout> <out.json.z>
"""
from __future__ import annotations

import ast
import json
import sys
import zlib
from pathlib import Path


def first_line(node) -> int:
    line = node.lineno
    for d in getattr(node, "decorator_list", ()) or ():
        line = min(line, d.lineno)
    return line


def outline(tree) -> dict:
    classes: list[str] = [""]
    nodes: list[list[int]] = []

    def scan(body, cls_idx):
        for stmt in body:
            # [first line, last line, indentation, enclosing class, is a class]
            nodes.append([first_line(stmt), stmt.end_lineno, stmt.col_offset, cls_idx,
                          int(isinstance(stmt, ast.ClassDef))])
            if isinstance(stmt, ast.ClassDef):
                name = f"{classes[cls_idx]}.{stmt.name}" if classes[cls_idx] else stmt.name
                classes.append(name)
                scan(stmt.body, len(classes) - 1)

    scan(tree.body, 0)
    return {"classes": classes, "nodes": nodes}


def main():
    root, out_path = Path(sys.argv[1]), sys.argv[2]
    out = {}
    for p in sorted(list((root / "sqlglot").rglob("*.py")) + list((root / "tests").rglob("*.py"))):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError:
            continue
        out[p.relative_to(root).as_posix()] = outline(tree)
    raw = json.dumps(out, separators=(",", ":")).encode()
    comp = zlib.compress(raw, 9)
    Path(out_path).write_bytes(comp)
    print(f"files={len(out)} statements={sum(len(v['nodes']) for v in out.values())} "
          f"raw={len(raw)/1e6:.2f}MB z={len(comp)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
