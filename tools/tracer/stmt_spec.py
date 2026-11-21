"""Which test methods can be re-run a part at a time?

The statements of a plain `def test_x(self)` body are grouped into units: two statements
belong together when they use the same local variable between two fresh assignments of
it (so one may read what the other wrote). Statements of different units exchange
nothing through locals, so a unit that does nothing but compute and assert
("independent") can be left out without changing what the others do. Units that may
change state outside the method (assignments to attributes of `self` or of globals,
class definitions, helper methods of the test class, ...) are always kept.

Output {test id: {"number": n, "units": [[independent 0/1, first line, last line, first
line, last line, ...], ...]}} (one line range per statement of the unit) is the job list
for run_trace.py --units and ends up in the probe map.

Usage: python3 stmt_spec.py <sqlglot checkout> <tests.json> <out.json> [--min-units 3]
                            [--skip <json list of test ids>] [--cpu <map.json.z> --min-cpu 0.008]
"""
from __future__ import annotations

import argparse
import ast
import json
import zlib

FORBIDDEN_NAMES = {"super", "__class__", "locals", "vars", "globals", "eval", "exec", "dir"}
# Modules / helpers whose use suggests state outside the test method is being changed.
STATEFUL_NAMES = {"os", "sys", "logging", "warnings", "mock", "patch", "importlib", "gc", "tracemalloc",
                  "threading", "subprocess", "setattr", "delattr", "time_machine", "freeze_time", "random",
                  "sqlglot", "tempfile", "shutil", "open"}
MUTATORS = {"update", "pop", "append", "extend", "add", "remove", "discard", "clear", "insert", "setdefault",
            "popitem", "sort", "reverse", "register", "start", "stop", "set", "__setitem__", "__delitem__"}
SELF_CALLS = ("assert", "validate", "subTest", "fail", "skipTest")


def _bound_names(fn: ast.FunctionDef) -> set[str]:
    """Names the method body binds in its own scope (not those of nested scopes)."""
    names = set()
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue   # its body is another scope
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # Only the first iterable is evaluated in the enclosing scope.
            stack.append(node.generators[0].iter)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.add(node.rest)
        stack.extend(ast.iter_child_nodes(node))
    names.discard("self")
    return names


def _self_attr(node) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return "self." + node.attr
    return None


def _root(node):
    """The expression an attribute / subscript chain starts from."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
        node = node.value
    return node


def _reads_writes(stmt: ast.stmt, local: set[str], self_written: set[str]) -> tuple[set[str], set[str]]:
    """(names the statement may read before it has surely written them, names it surely writes).

    Names are the method's locals plus "self.x" for attributes of self the method assigns."""
    reads: set[str] = set()
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name) and node.id in local and not isinstance(node.ctx, ast.Store):
            reads.add(node.id)
        else:
            attr = _self_attr(node)
            if attr in self_written and not isinstance(node.ctx, ast.Store):
                reads.add(attr)
    sure: set[str] = set()
    targets = []
    value = None
    if isinstance(stmt, ast.Assign):
        targets, value = stmt.targets, stmt.value
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        targets, value = [stmt.target], stmt.value
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        sure = {(a.asname or a.name).split(".")[0] for a in stmt.names} & local
    elif isinstance(stmt, ast.FunctionDef):
        sure = {stmt.name}
    for t in targets:
        elts = t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]
        for e in elts:
            if isinstance(e, ast.Name):
                sure.add(e.id)
            elif _self_attr(e) in self_written:
                sure.add(_self_attr(e))
    if value is not None:
        # A plain assignment reads only what its right-hand side (and subscripted targets) read.
        reads = set()
        nodes = [value] + [t for t in targets if not isinstance(t, ast.Name)]
        for top in nodes:
            for node in ast.walk(top):
                if isinstance(node, ast.Name) and node.id in local and not isinstance(node.ctx, ast.Store):
                    reads.add(node.id)
                else:
                    attr = _self_attr(node)
                    if attr in self_written and not isinstance(node.ctx, ast.Store):
                        reads.add(attr)
    return reads, sure


def _stateful(stmts: list[ast.stmt], local: set[str]) -> bool:
    """May these statements change anything other than the method's own locals?"""
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.NamedExpr)):
                return True
            if isinstance(node, ast.Name) and node.id in STATEFUL_NAMES and node.id not in local:
                return True
            if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root = _root(node)
                if not (isinstance(root, ast.Name) and root.id in local):
                    return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                root = _root(node.func.value)
                if isinstance(root, ast.Name) and root.id == "self" and isinstance(node.func.value, ast.Name):
                    if not node.func.attr.startswith(SELF_CALLS):
                        return True    # a helper of the test class: unknown effects
                elif node.func.attr in MUTATORS and isinstance(root, ast.Name) and root.id not in local:
                    return True
    return False


def method_spec(fn: ast.FunctionDef, lines: list[str]) -> list[list[int]] | None:
    a = fn.args
    if (not isinstance(fn, ast.FunctionDef) or fn.decorator_list or a.vararg or a.kwarg or a.kwonlyargs
            or a.defaults or a.posonlyargs or [x.arg for x in a.args] != ["self"]):
        return None
    for node in ast.walk(fn):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)):
            return None
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return None
    indent = fn.body[0].col_offset
    if indent < 8 or indent % 4 or indent != fn.col_offset + 4:
        return None
    local = _bound_names(fn)
    self_written = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)) and _self_attr(node):
            self_written.add(_self_attr(node))
    spans = []
    prev_end = fn.body[0].lineno - 1
    for stmt in fn.body:
        lo, hi = stmt.lineno, stmt.end_lineno
        for d in getattr(stmt, "decorator_list", None) or ():
            lo = min(lo, d.lineno)
        # One statement per line run, each starting its own line at the body's indentation.
        if lo <= prev_end or stmt.col_offset != indent or lines[lo - 1][:indent].strip():
            return None
        spans.append((lo, hi))
        prev_end = hi
    n = len(fn.body)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # A name's uses between two fresh assignments form one generation; statements that
    # touch the same generation may pass a value to each other.
    generation: dict[str, int] = {}
    owner: dict[tuple, int] = {}
    for i, stmt in enumerate(fn.body):
        reads, sure = _reads_writes(stmt, local, self_written)
        mentions = set()
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id in local:
                mentions.add(node.id)
            elif _self_attr(node) in self_written:
                mentions.add(_self_attr(node))
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in local:
                mentions.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mentions.update({(a.asname or a.name).split(".")[0] for a in node.names} & local)
            elif isinstance(node, ast.ExceptHandler) and node.name in local:
                mentions.add(node.name)
        for v in sorted(mentions):
            if v in sure and v not in reads:
                generation[v] = generation.get(v, 0) + 1
            key = (v, generation.get(v, 0))
            if key in owner:
                parent[find(i)] = find(owner[key])
            else:
                owner[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    units = []
    for members in sorted(groups.values()):
        stmts = [fn.body[i] for i in members]
        independent = not _stateful(stmts, local)
        # A docstring or a bare constant is not worth a unit of its own: keep it.
        if all(isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) for x in stmts):
            independent = False
        if any(isinstance(node, (ast.Return, ast.Raise)) for x in stmts for node in [x]):
            independent = False
        unit = [int(independent)]
        for i in members:
            unit.extend(spans[i])
        units.append(unit)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("tests")
    ap.add_argument("out")
    ap.add_argument("--min-units", type=int, default=3)
    ap.add_argument("--skip", default="", help="JSON list of test ids to leave out")
    ap.add_argument("--cpu", default="", help="map.json.z with per-test CPU")
    ap.add_argument("--min-cpu", type=float, default=0.0)
    a = ap.parse_args()
    tests = json.load(open(a.tests))["tests"]
    skip = set(json.load(open(a.skip))) if a.skip else set()
    cpu = {}
    if a.cpu:
        m = json.loads(zlib.decompress(open(a.cpu, "rb").read()))
        cpu = dict(zip(m["tests"], m.get("cpu") or m["trace_cpu"]))
    trees: dict[str, tuple] = {}
    out = {}
    n_units = n_indep = 0
    for t in tests:
        tid = t["id"]
        if tid in skip or cpu.get(tid, 1.0) < a.min_cpu:
            continue
        f, cls, meth = tid.split("::")
        if f not in trees:
            src = open(f"{a.root}/{f}", encoding="utf-8").read()
            trees[f] = (ast.parse(src), src.split("\n"))
        tree, lines = trees[f]
        fn = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls:
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef) and sub.name == meth:
                        fn = sub
        if fn is None:
            continue
        units = method_spec(fn, lines)
        if units is None:
            continue
        k = sum(u[0] for u in units)
        if k < a.min_units:
            continue
        out[tid] = {"number": t["number"], "units": units}
        n_units += len(units)
        n_indep += k
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} test methods, {n_units} units, {n_indep} independent "
          f"({sum(cpu.get(t, 0) for t in out):.1f}s CPU of {sum(cpu.values()):.1f}s)")


if __name__ == "__main__":
    main()
