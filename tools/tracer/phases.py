"""Map for settling the executor's TPC tests in two phases (see selector/brh.py, PHASE_RECIPES).

For every query of TestExecutor.test_execute_tpcds / test_execute_tpch, on the pinned commit:

  fp       digests of the parsed expression (as a pool worker receives it) and of the
           optimized expression that execute() hands to the planner
  opt_cpu  CPU seconds of getting that far (parse + optimize)
  other    the lines of the tree the query runs OUTSIDE parsing and optimizing: class
           fixtures, pickling to and from the worker, planning, the Python executor, the
           comparison with DuckDB. They are measured in a fresh process per query that is given
           the pinned parsed and optimized expressions from disk and never runs the parser or
           the optimizer itself, so nothing those two would have left in caches hides a line.

A patch that touches none of a query's `other` lines and leaves both digests as they are
cannot change what that query does after the optimizer.

Usage (inside the image):
  python phases.py --root /app/src --out maps/phases.json.z [--jobs 2]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
SELECTOR = os.path.normpath(os.path.join(HERE, "..", "..", "selector"))


def setup(root: str):
    sys.path.insert(0, SELECTOR)
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    import unittest  # noqa: F401 - loaded before the package, as under pytest

    import brh

    return brh


def cold(a):
    """One query, started from the pinned expressions on disk; prints the lines it runs."""
    brh = setup(a.root)
    root = os.path.join(os.path.abspath(a.root), "")
    lines: set = set()
    mon = sys.monitoring
    tool = 3
    mon.use_tool_id(tool, "phases")

    def on_line(code, line):
        if code.co_filename.startswith(root):
            lines.add((code.co_filename[len(root):], line))
        return mon.DISABLE

    def on_start(code, offset):
        return mon.DISABLE

    # LINE events alone are not delivered for functions that consist of one line (the
    # lambdas of the executor's operator table, say): CPython 3.13 reports those lines only
    # while PY_START is being monitored as well.
    mon.register_callback(tool, mon.events.LINE, on_line)
    mon.register_callback(tool, mon.events.PY_START, on_start)
    import tests.test_executor as te

    recipe = brh.PHASE_RECIPES[a.test]
    cls = te.TestExecutor
    with open(a.cold, "rb") as f:
        blob_parsed, blob_optimized = pickle.load(f)
    mon.set_events(tool, mon.events.LINE | mon.events.PY_START)
    ok = True
    try:
        cls.setUpClass()
        te.initializer(getattr(te, recipe["schema"]), getattr(cls, recipe["tables"]))
        meta, sql, _ = getattr(cls, recipe["sqls"])[a.entry]
        parsed = pickle.loads(blob_parsed)
        pickle.dumps(parsed)                       # the parent's side of sending it
        optimized = pickle.loads(blob_optimized) if blob_optimized is not None else None
        import sqlglot.executor as ex

        ex.optimize = lambda *args, **kw: optimized
        table = te.mp_execute(parsed, meta)
        table = pickle.loads(pickle.dumps(table))
        if table is not None:
            case = cls(a.test.rsplit("::", 1)[1])
            before = len(getattr(case, "_outcome", None) or ())
            case.subtestHelper(a.entry, table, tpch=recipe["sqls"].startswith("tpch"))
            del before
    except BaseException as e:  # noqa: BLE001
        ok = False
        sys.stderr.write(f"entry {a.entry}: {type(e).__name__}: {str(e)[:200]}\n")
    mon.set_events(tool, 0)
    json.dump({"ok": ok, "lines": sorted(lines)}, sys.stdout)


def digests(a):
    """The digests of every query in this process (whose hash seed the caller chose)."""
    brh = setup(a.root)
    import sqlglot.executor as ex
    import tests.test_executor as te

    cls = te.TestExecutor
    cls.setUpClass()
    ex.Plan = brh.plan_stub
    out = {}
    for tid, recipe in brh.PHASE_RECIPES.items():
        te.initializer(getattr(te, recipe["schema"]), getattr(cls, recipe["tables"]))
        out[tid] = {str(k): brh.executor_phase_one(te, sql, meta)
                    for k, (meta, sql, _) in enumerate(getattr(cls, recipe["sqls"]))}
    json.dump(out, sys.stdout)


SEEDS = ("1", "2", "3", "4", "5", "6")
ONLY_PARSED = "*"       # in place of a digest: the query only has to get as far as it does


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cold")
    ap.add_argument("--test")
    ap.add_argument("--entry", type=int)
    ap.add_argument("--digests", action="store_true", help="print the digests of every query and stop")
    a = ap.parse_args()
    if a.cold:
        cold(a)
        return
    if a.digests:
        digests(a)
        return
    brh = setup(a.root)
    import sqlglot.executor as ex
    import tests.test_executor as te

    cls = te.TestExecutor
    cls.setUpClass()
    real_plan = ex.Plan
    ex.Plan = brh.plan_stub
    tmp = tempfile.mkdtemp(prefix="phases_")
    out = {"tests": {}}
    jobs = []
    for tid, recipe in brh.PHASE_RECIPES.items():
        te.initializer(getattr(te, recipe["schema"]), getattr(cls, recipe["tables"]))
        sqls = getattr(cls, recipe["sqls"])
        rec = {"file": recipe["file"], "fp": {}, "opt_cpu": {}, "lines": [], "other": {}}
        # twice: the second pass has warm caches, like a query in the middle of the test
        for attempt in (0, 1):
            for k, (meta, sql, _) in enumerate(sqls):
                kept: list = []
                t0 = time.process_time()
                fp = brh.executor_phase_one(te, sql, meta, kept)
                cpu = time.process_time() - t0
                if attempt == 0:
                    rec["fp"][str(k)] = fp
                    parsed = pickle.dumps(pickle.loads(pickle.dumps(te.parse_one(sql))))
                    path = os.path.join(tmp, f"{recipe['sqls']}_{k}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump((parsed, pickle.dumps(kept[0]) if kept else None), f)
                    jobs.append((tid, k, path))
                else:
                    if fp != rec["fp"][str(k)]:
                        rec["fp"][str(k)] = None       # not reproducible: never trusted
                    rec["opt_cpu"][str(k)] = round(cpu, 4)
        out["tests"][tid] = rec
    ex.Plan = real_plan

    # the lines each query runs outside parsing and optimizing, one fresh process per query
    def run(job):
        tid, k, path = job
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--root", a.root, "--cold", path,
                              "--test", tid, "--entry", str(k)], capture_output=True, text=True)
        try:
            return tid, k, json.loads(res.stdout), res.stderr[-300:]
        except ValueError:
            return tid, k, {"ok": False, "lines": []}, res.stderr[-300:]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(run, jobs))
    # Digests must not depend on the hash seed of the process (the order in which sets are
    # walked): they are compared with the ones of other processes, each with its own seed.
    others = []
    for seed in SEEDS:
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--root", a.root, "--digests"],
                             capture_output=True, text=True, env=dict(os.environ, PYTHONHASHSEED=seed))
        others.append(json.loads(res.stdout))
    for tid in out["tests"]:
        rec = out["tests"][tid]
        unstable = 0
        for k, fp in list(rec["fp"].items()):
            if fp is None:
                continue
            if fp[1] is None:
                # A query the test does not execute: it is parsed and sent to a worker, and
                # that is all. What its expression looks like does not matter.
                if all(o[tid][k] is not None and o[tid][k][1] is None for o in others):
                    rec["fp"][k] = [ONLY_PARSED, None]
                else:
                    rec["fp"][k] = None
                continue
            if any(o[tid][k] != fp for o in others):
                rec["fp"][k] = None                    # depends on the hash seed: never trusted
                unstable += 1
        print(f"{tid}: {unstable} queries whose digests depend on the hash seed")
    for tid in out["tests"]:
        rec = out["tests"][tid]
        mine = [(k, r) for t, k, r, _ in results if t == tid]
        union = sorted({tuple(x) for _, r in mine for x in r["lines"]})
        pos = {x: i for i, x in enumerate(union)}
        rec["lines"] = [list(x) for x in union]
        bad = 0
        for k, r in mine:
            if not r["ok"]:
                rec["fp"][str(k)] = None               # cannot be reproduced from its expressions
                bad += 1
                continue
            bits = 0
            for x in r["lines"]:
                bits |= 1 << pos[tuple(x)]
            rec["other"][str(k)] = format(bits, "x")
        n = len(rec["fp"])
        usable = sum(1 for v in rec["fp"].values() if v is not None)
        print(f"{tid}: {n} queries, {usable} usable, {bad} not reproducible, {len(union)} lines outside "
              f"parse/optimize, parse+optimize {sum(rec['opt_cpu'].values()):.1f} s")
    for t, k, r, err in results:
        if not r["ok"]:
            print("   not reproducible:", t.rsplit("::", 1)[1], k, err.strip()[-160:])
    with open(a.out, "wb") as f:
        f.write(zlib.compress(json.dumps(out, separators=(",", ":")).encode(), 9))
    print("wrote", a.out, os.path.getsize(a.out), "bytes")


if __name__ == "__main__":
    main()
