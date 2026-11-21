"""Find, for expensive tests, the top-level statements of the test method that take most
of its time and can be left out without breaking the rest of the method.

The selector uses this to re-run a "light" version of such a test: everything except
the dominant statement(s).

Usage (inside the build image):
  python stmt_profile.py --root <sqlglot> --harness <selector dir> --out light.json <test id>...
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import sys
import textwrap
import time

MIN_HEAVY_S = 0.5
MIN_SHARE = 0.25


def droppable(stmt) -> bool:
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        return True
    if isinstance(stmt, ast.With):
        return all(isinstance(s, (ast.Expr, ast.Pass)) or droppable(s) for s in stmt.body)
    return False


def checks(stmt) -> bool:
    """Does the statement assert something (self.assert*, assertRaises block, assert)?"""
    for n in ast.walk(stmt):
        if isinstance(n, ast.Assert):
            return True
        if isinstance(n, ast.Attribute) and (n.attr.startswith("assert") or n.attr.startswith("validate")):
            return True
    return False


def bound_names(stmt) -> set[str]:
    out = set()
    if isinstance(stmt, ast.With):
        for item in stmt.items:
            if item.optional_vars is not None:
                out |= {n.id for n in ast.walk(item.optional_vars) if isinstance(n, ast.Name)}
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For)):
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        for t in targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return out


def read_names(stmt) -> set[str]:
    return {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def plan_drop(body, cost):
    """body: top-level statements; cost: {index: seconds}. Returns indices to drop or None."""
    total = sum(cost.values()) or 1e-9
    heavy = {i for i, c in cost.items() if c >= MIN_HEAVY_S and c / total >= MIN_SHARE}
    if not heavy or not all(droppable(body[i]) for i in heavy):
        return None
    drop = set(heavy)
    dead: set[str] = set()
    for i, stmt in enumerate(body):
        if i in drop:
            dead |= bound_names(stmt)
            continue
        if read_names(stmt) & dead:
            if not droppable(stmt):
                return None
            drop.add(i)
            dead |= bound_names(stmt)
        else:
            dead -= bound_names(stmt)
    return sorted(drop)


def profile(test_id: str):
    import brh

    case = brh.load_case(test_id)
    meth = test_id.rsplit("::", 1)[1]
    func = inspect.unwrap(getattr(type(case), meth))
    if func is not getattr(type(case), meth):
        return None  # decorated: the source does not match the callable
    code = func.__code__
    fdef = brh.find_function(code.co_filename, meth, code.co_firstlineno)
    if fdef is None or fdef.decorator_list:
        return None
    offset = 0
    body = fdef.body
    mon = sys.monitoring
    events = []

    def on_line(_code, line):
        events.append((line, time.process_time()))

    mon.use_tool_id(4, "stmtprof")
    mon.register_callback(4, mon.events.LINE, on_line)
    mon.set_local_events(4, code, mon.events.LINE)
    t0 = time.process_time()
    failed = brh.run_tests([test_id])
    t1 = time.process_time()
    mon.set_local_events(4, code, 0)
    mon.free_tool_id(4)
    if failed or not events:
        return None
    spans = [(s.lineno + offset, s.end_lineno + offset) for s in body]

    def index_of(line):
        for i, (lo, hi) in enumerate(spans):
            if lo <= line <= hi:
                return i
        return None

    cost: dict[int, float] = {}
    for (line, t), nxt in zip(events, events[1:] + [(None, t1)]):
        i = index_of(line)
        if i is not None:
            cost[i] = cost.get(i, 0.0) + (nxt[1] - t)
    drop = plan_drop(body, cost)
    if not drop:
        return None
    # What is left must still check something, or a passing light run would mean nothing.
    kept = [s for i, s in enumerate(body) if i not in drop]
    if not any(checks(s) for s in kept):
        return None
    dropped = sum(cost.get(i, 0.0) for i in drop)
    return {
        "drop": [list(spans[i]) for i in drop],
        "first_line": code.co_firstlineno,
        "total_cpu": round(t1 - t0, 3),
        "light_cpu": round((t1 - t0) - dropped, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("tests", nargs="+")
    a = ap.parse_args()
    sys.path.insert(0, a.root)
    sys.path.append(a.harness)
    os.chdir(a.root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    import brh

    brh.install()
    out = {}
    for tid in a.tests:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                res = profile(tid)
                os.write(w, json.dumps(res).encode())
            except BaseException as e:  # noqa: BLE001
                os.write(w, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            finally:
                os._exit(0)
        os.close(w)
        buf = b""
        while chunk := os.read(r, 65536):
            buf += chunk
        os.close(r)
        os.waitpid(pid, 0)
        res = json.loads(buf) if buf else None
        print(tid, "->", res, flush=True)
        if res and not res.get("error"):
            out[tid] = res
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
