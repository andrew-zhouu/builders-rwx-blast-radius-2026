"""Static index of where module-level and class-level names of the pinned tree are used.

For every name assigned at module or class level (constants, tables, flags) it
lists each place the name appears, and whether that place runs when a function
is called or when a module is imported. The selector uses it to find the code
that can read a changed definition.

Usage: python3 build_idents.py <sqlglot checkout> <out.json.z>
"""
from __future__ import annotations

import ast
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

MAX_OCCURRENCES = 400
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def targets_of(stmt) -> list[str]:
    out = []
    if isinstance(stmt, ast.Assign):
        nodes = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        nodes = [stmt.target]
    else:
        return out
    for t in nodes:
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                out.append(n.id)
    return out


def defined_names(tree) -> set[str]:
    names: set[str] = set()

    def scan(body):
        for stmt in body:
            names.update(targets_of(stmt))
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # `import x as y` / `from m import x as y`: y is a second name for a definition.
                names.update(a.asname for a in stmt.names if a.asname)
            if isinstance(stmt, ast.ClassDef):
                scan(stmt.body)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
                for field in ("body", "orelse", "finalbody"):
                    scan(getattr(stmt, field, []) or [])
                for h in getattr(stmt, "handlers", []) or []:
                    scan(h.body)

    scan(tree.body)
    return {n for n in names if len(n) > 1 and not (n.startswith("__") and n.endswith("__"))}


def occurrences(tree, wanted: set[str]):
    """Yield (name, line, kind, cls, target)."""

    def walk(node, in_func, cls, target):
        for child in ast.iter_child_nodes(node):
            c_in_func, c_cls, c_target = in_func, cls, target
            if isinstance(node, FUNC) and child in (node.body if isinstance(node.body, list) else [node.body]):
                c_in_func = True
            if isinstance(node, ast.ClassDef) and child in node.body and not in_func:
                c_cls = f"{cls}.{node.name}" if cls else node.name
                c_target = ""
            if isinstance(child, ast.stmt) and not c_in_func:
                ts = targets_of(child)
                c_target = ts[0] if ts else (child.name if isinstance(child, (ast.FunctionDef, ast.ClassDef)) else "")
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    for a in child.names:
                        # The imported name is used here to define its alias.
                        if a.asname and a.name in wanted:
                            yield a.name, child.lineno, "i", c_cls, a.asname, "n"
            name = how = None
            if isinstance(child, ast.Name):
                name, how = child.id, "n"
                if isinstance(child.ctx, ast.Store) and not c_in_func and name == c_target:
                    name = None  # the definition itself
            elif isinstance(child, ast.Attribute):
                name, how = child.attr, "a"
            elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                name, how = child.value, "s"
            if name in wanted:
                yield name, child.lineno, ("c" if c_in_func else "i"), c_cls, c_target, how
            yield from walk(child, c_in_func, c_cls, c_target)

    yield from walk(tree, False, "", "")


def main():
    root, out_path = Path(sys.argv[1]), sys.argv[2]
    paths = sorted(list((root / "sqlglot").rglob("*.py")) + list((root / "tests").rglob("*.py")))
    trees = {}
    for p in paths:
        try:
            trees[p.relative_to(root).as_posix()] = ast.parse(p.read_text())
        except SyntaxError:
            continue
    wanted: set[str] = set()
    for rel, tree in trees.items():
        if rel.startswith("sqlglot/") or rel in ("tests/helpers.py", "tests/dialects/test_dialect.py"):
            wanted |= defined_names(tree)
    files = sorted(trees)
    fidx = {f: i for i, f in enumerate(files)}
    occ = defaultdict(set)
    for rel, tree in trees.items():
        for name, line, kind, cls, target, how in occurrences(tree, wanted):
            occ[name].add((fidx[rel], line, kind, cls, target, how))
    names = {}
    generic = 0
    for name in sorted(wanted):
        o = sorted(occ.get(name, ()))
        if len(o) > MAX_OCCURRENCES:
            names[name] = None
            generic += 1
        else:
            names[name] = [list(x) for x in o]
    raw = json.dumps({"files": files, "names": names}, separators=(",", ":")).encode()
    comp = zlib.compress(raw, 9)
    Path(out_path).write_bytes(comp)
    print(f"names={len(names)} generic={generic} occurrences={sum(len(v) for v in names.values() if v)} "
          f"raw={len(raw)/1e6:.2f}MB z={len(comp)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
