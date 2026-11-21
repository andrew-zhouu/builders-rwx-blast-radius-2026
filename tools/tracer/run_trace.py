"""Trace every test of the pinned sqlglot commit in isolation.

Each test runs in its own forked child that starts with only third-party
libraries loaded: it imports its test module and runs the single test under
brtrace, so a test's record holds everything that test needs on its own
(including import-time code and lazily computed caches).

Usage (inside br-dev):
  python run_trace.py --root /src/sqlglot --tests tests.json --out out.pkl --shard 0/8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import shutil
import sys
import time


def _mods():
    return sorted(m for m in sys.modules if m == "sqlglot" or m.startswith(("sqlglot.", "tests")))


def run_one(test: dict, root: str, tmp_base: str) -> dict:
    """Runs in the forked child."""
    import unittest

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    # The harness the selector itself uses at run time, so tests are traced the way they are re-run.
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "selector")))
    sys.path.insert(0, root)
    os.chdir(root)
    sub_dir = os.path.join(tmp_base, f"sub_{test['number']}")
    os.makedirs(sub_dir, exist_ok=True)
    os.environ["BR_TRACE_SUBDIR"] = sub_dir
    os.environ["BR_TRACE_ROOT"] = root
    os.environ["PYTHONPATH"] = f"{root}:{here}"
    os.environ.setdefault("SKIP_INTEGRATION", "0")

    import brtrace

    tracer = brtrace.Tracer(root)
    tracer.start()

    import brh

    rec: dict = {"n": test["number"], "id": test["id"]}
    if test.get("filter") is not None:
        rec["filter"] = test["filter"]
    c0 = os.times()
    w0 = time.perf_counter()
    try:
        brh.install()
        if test.get("filter") is not None:
            brh.FIXTURE_FILTER.update({k: set(v) for k, v in test["filter"].items()})
        brh.load_case(test["id"])
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"import: {type(e).__name__}: {e}"
        tracer.stop()
        return rec
    c1 = os.times()
    w1 = time.perf_counter()
    rec["modules_at_start"] = _mods()
    failed = brh.run_tests([test["id"]])
    c2 = os.times()
    w2 = time.perf_counter()
    tracer.active = False
    snap = tracer.snapshot()
    tracer.stop()
    rec["fixture_counts"] = dict(brh.FIXTURE_COUNTS)

    def cpu(a, b):
        return (b.user - a.user) + (b.system - a.system) + (
            b.children_user - a.children_user) + (b.children_system - a.children_system)

    rec.update(snap)
    rec["ok"] = not failed
    rec["problems"] = list(failed.values())[:3]
    rec["t_import_cpu"] = cpu(c0, c1)
    rec["t_run_cpu"] = cpu(c1, c2)
    rec["t_import_wall"] = w1 - w0
    rec["t_run_wall"] = w2 - w1
    rec["modules_at_end"] = _mods()
    snaps = []
    for p in glob.glob(os.path.join(sub_dir, "*.pkl")):
        try:
            with open(p, "rb") as f:
                snaps.append(pickle.load(f))
        except Exception:
            continue
    rec["n_sub"] = len(snaps)
    rec["sub"] = brtrace.merge_snapshots(snaps) if snaps else None
    shutil.rmtree(sub_dir, ignore_errors=True)
    return rec


def _setup_reads(root: str):
    here = os.path.dirname(os.path.abspath(__file__))
    for extra in (here, os.path.normpath(os.path.join(here, "..", "..", "selector")),
                  os.path.normpath(os.path.join(here, "..", ".."))):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{here}"
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ.pop("BR_TRACE_SUBDIR", None)


def run_reads(test: dict, root: str) -> dict:
    """Runs in the forked child: which entries of sqlglot's tables does this test read?"""
    _setup_reads(root)
    import brtrace
    import readtrace

    readtrace.install(root)
    tracer = brtrace.Tracer(root, reads=True)
    tracer.start()
    import brh

    rec: dict = {"n": test["number"], "id": test["id"]}
    try:
        brh.install()
        brh.load_case(test["id"])
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"import: {type(e).__name__}: {e}"
        return rec
    c0 = os.times()
    failed = brh.run_tests([test["id"]])
    c1 = os.times()
    rec["ok"] = not failed
    rec["problems"] = list(failed.values())[:2]
    rec["reads"] = readtrace.snapshot()
    rec["t_run_cpu"] = (c1.user - c0.user) + (c1.system - c0.system)
    return rec


def all_tables(root: str) -> list[str]:
    """Runs in a forked child: every table path the package defines."""
    _setup_reads(root)
    import brtrace
    import readtrace

    readtrace.install(root)
    tracer = brtrace.Tracer(root, reads=True)
    tracer.start()
    from selector import state as st

    st.import_all(st.module_names())
    return sorted({p for paths in readtrace.REG.values() for p in paths})


def main_reads(args):
    tests = json.load(open(args.tests))["tests"]
    i, n = (int(x) for x in args.shard.split("/"))
    order = sorted(tests, key=lambda t: -t["seconds"])
    if args.only:
        want = set(args.only.split(","))
        order = [t for t in order if t["id"] in want or str(t["number"]) in want]
    tests = [t for k, t in enumerate(order) if k % n == i]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import readtrace

    t0 = time.time()
    count = readtrace.precompile(args.root)
    sys.stderr.write(f"[{args.shard}] rewrote {count} files in {time.time()-t0:.0f}s\n")
    out = []
    if i == 0:
        out.append({"tables": _fork_collect(lambda: all_tables(args.root))})
    for k, test in enumerate(tests):
        rec = _fork_collect(lambda: run_reads(test, args.root))
        out.append(rec if rec is not None else {"n": test["number"], "id": test["id"], "error": "no result"})
        if k % 50 == 0:
            sys.stderr.write(f"[{args.shard}] {k}/{len(tests)} {time.time()-t0:.0f}s\n")
            sys.stderr.flush()
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    bad = [r for r in out if "id" in r and (r.get("error") or not r.get("ok"))]
    sys.stderr.write(f"[{args.shard}] done {len(out)} records, {len(bad)} not ok, {time.time()-t0:.0f}s\n")
    for r in bad[:15]:
        sys.stderr.write(f"   {r['id']}: {r.get('error') or r.get('problems')}\n")


def _fork_collect(fn):
    """Run fn() in a forked child and return what it returns (pickled through a file)."""
    path = f"/tmp/brtrace_res_{os.getpid()}_{time.monotonic_ns()}.pkl"
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            res = fn()
            with open(path, "wb") as f:
                pickle.dump(res, f)
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    os.waitpid(pid, 0)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        res = pickle.load(f)
    os.unlink(path)
    return res


def run_entries(test: dict, root: str) -> list[dict]:
    """Trace a fixture-driven test one fixture entry at a time ("probes").

    Runs in a child that has imported the test module; every probe is a further
    fork that restricts the fixture loaders to a single entry, so its record
    holds what that entry alone executes. Probe `base` keeps no entry at all:
    it is the part of the test that does not depend on fixture entries."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "selector")))
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{here}"
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ.pop("BR_TRACE_SUBDIR", None)

    import brtrace

    tracer = brtrace.Tracer(root)
    tracer.start()
    import brh

    brh.install()
    brh.load_case(test["id"])
    files = {k.split(":", 1)[1]: n for k, n in test["files"].items()}

    def probe(keep: dict, label: str):
        def body():
            tracer.reset()
            brh.FIXTURE_FILTER.clear()
            for f in files:
                brh.FIXTURE_FILTER[f] = set(keep.get(f, ()))
            c0 = os.times()
            failed = brh.run_tests([test["id"]])
            c1 = os.times()
            tracer.active = False
            rec = {"n": test["number"], "id": test["id"], "probe": label,
                   "filter": {f: sorted(v) for f, v in keep.items()}}
            rec.update(tracer.snapshot())
            rec["ok"] = not failed
            rec["problems"] = list(failed.values())[:2]
            rec["t_run_cpu"] = (c1.user - c0.user) + (c1.system - c0.system)
            rec["modules_at_end"] = _mods()
            rec["sub"] = None
            return rec

        return _fork_collect(body)

    out = []
    base = probe({}, "base")
    if base is None:
        return out
    out.append(base)
    base_lines = set(map(tuple, base["run"]["lines"])) | set(map(tuple, base["imp"]["lines"]))
    for f, n in sorted(files.items()):
        whole = probe({f: range(n)}, f"{f}#all")
        if whole is None:
            continue
        whole_lines = set(map(tuple, whole["run"]["lines"])) | set(map(tuple, whole["imp"]["lines"]))
        if whole_lines <= base_lines:
            continue  # the test loads this fixture but nothing it runs depends on the entries
        for k in range(n):
            rec = probe({f: [k]}, f"{f}#{k}")
            if rec is not None:
                out.append(rec)
    return out


def run_units(test: dict, root: str) -> list[dict]:
    """Trace a test method one unit at a time ("probes"; units come from stmt_spec.py).

    Runs in a child that has imported the test module; every probe is a further fork that
    reduces the method to the units that are always kept plus one independent unit, so
    its record holds what that unit needs on its own. Probe `base` runs the kept units only."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "selector")))
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["PYTHONPATH"] = f"{root}:{here}"
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ.pop("BR_TRACE_SUBDIR", None)

    import brtrace

    tracer = brtrace.Tracer(root)
    tracer.start()
    import brh

    brh.install()
    brh.load_case(test["id"])
    units = test["units"]

    def spans_of(unit):
        return [[unit[k], unit[k + 1]] for k in range(1, len(unit), 2)]

    base = sorted(s for u in units if not u[0] for s in spans_of(u))

    def probe(extra, label: str):
        def body():
            tracer.reset()
            brh.KEEP_STATEMENTS[test["id"]] = sorted(base + extra)
            c0 = os.times()
            failed = brh.run_tests([test["id"]])
            c1 = os.times()
            tracer.active = False
            rec = {"n": test["number"], "id": test["id"], "probe": label, "filter": {}}
            rec.update(tracer.snapshot())
            rec["ok"] = not failed and not brh.UNVERIFIED
            rec["problems"] = list(failed.values())[:2] + [f"unverified {u}" for u in brh.UNVERIFIED[:1]]
            rec["t_run_cpu"] = (c1.user - c0.user) + (c1.system - c0.system)
            rec["modules_at_end"] = _mods()
            rec["sub"] = None
            return rec

        return _fork_collect(body)

    out = []
    if base:
        rec = probe([], "base")
        if rec is None:
            return out
        out.append(rec)
    for k, unit in enumerate(units):
        if unit[0]:
            rec = probe(spans_of(unit), f"@unit#{k}")
            if rec is not None:
                out.append(rec)
    return out


def main_units(args):
    spec = json.load(open(args.units))
    jobs = [dict(v, id=k) for k, v in sorted(spec.items(), key=lambda kv: -len(kv[1]["units"]))]
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    out = []
    t0 = time.time()
    for job in jobs:
        recs = _fork_collect(lambda: run_units(job, args.root)) or []
        out.extend(recs)
        bad = sum(1 for r in recs if not r["ok"])
        sys.stderr.write(f"[{args.shard}] {job['id']}: {len(recs)} probes, {bad} not ok, {time.time()-t0:.0f}s\n")
        sys.stderr.flush()
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    sys.stderr.write(f"[{args.shard}] done {len(out)} probes {time.time()-t0:.0f}s\n")


def main_entries(args):
    jobs = json.load(open(args.entries))
    i, n = (int(x) for x in args.shard.split("/"))
    jobs = [j for k, j in enumerate(jobs) if k % n == i]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    out = []
    t0 = time.time()
    for job in jobs:
        recs = _fork_collect(lambda: run_entries(job, args.root)) or []
        out.extend(recs)
        bad = sum(1 for r in recs if not r["ok"])
        sys.stderr.write(f"[{args.shard}] {job['id']}: {len(recs)} probes, {bad} not ok, {time.time()-t0:.0f}s\n")
        sys.stderr.flush()
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    sys.stderr.write(f"[{args.shard}] done {len(out)} probes {time.time()-t0:.0f}s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", default="", help="JSON list of fixture-driven tests to trace per entry")
    ap.add_argument("--units", default="", help="output of stmt_spec.py: trace these tests one unit at a time")
    ap.add_argument("--reads", action="store_true", help="record table reads (readtrace.py) instead of coverage")
    ap.add_argument("--root", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="comma separated test numbers")
    ap.add_argument("--jobs", default="", help="JSON job list (overrides the default per-test jobs)")
    ap.add_argument("--tmp", default="/tmp/brtrace")
    args = ap.parse_args()
    if args.entries:
        return main_entries(args)
    if args.units:
        return main_units(args)
    if args.reads:
        return main_reads(args)

    tests = json.load(open(args.tests))["tests"]
    if args.jobs:
        # Explicit job list: [{"number":..,"id":..,"filter":{file:[idx]}}], sharded round-robin.
        jobs = json.load(open(args.jobs))
        i, n = (int(x) for x in args.shard.split("/"))
        tests = [j for k, j in enumerate(jobs) if k % n == i]
    elif args.only:
        want = {int(x) for x in args.only.split(",")}
        tests = [t for t in tests if t["number"] in want]
    else:
        i, n = (int(x) for x in args.shard.split("/"))
        order = sorted(tests, key=lambda t: -t["seconds"])
        tests = [t for k, t in enumerate(order) if k % n == i]

    # Heavy third-party imports are shared by every child.
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    os.makedirs(args.tmp, exist_ok=True)
    out = []
    t0 = time.time()
    for k, test in enumerate(tests):
        res_path = os.path.join(args.tmp, f"res_{os.getpid()}_{k}.pkl")
        pid = os.fork()
        if pid == 0:
            code = 0
            try:
                rec = run_one(test, args.root, args.tmp)
                with open(res_path, "wb") as f:
                    pickle.dump(rec, f)
            except BaseException as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                sys.stderr.write(f"child {test['id']}: {type(e).__name__}: {e}\n")
                code = 1
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        if os.path.exists(res_path):
            with open(res_path, "rb") as f:
                out.append(pickle.load(f))
            os.unlink(res_path)
        else:
            out.append({"n": test["number"], "id": test["id"], "error": f"no result, status {status}"})
        if k % 50 == 0:
            sys.stderr.write(f"[{args.shard}] {k}/{len(tests)} {time.time()-t0:.0f}s\n")
            sys.stderr.flush()
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    bad = [r for r in out if r.get("error") or not r.get("ok")]
    sys.stderr.write(f"[{args.shard}] done {len(out)} tests, {len(bad)} not ok, {time.time()-t0:.0f}s\n")
    for r in bad[:10]:
        sys.stderr.write(f"   {r['id']}: {r.get('error') or r.get('problems')}\n")


if __name__ == "__main__":
    main()
