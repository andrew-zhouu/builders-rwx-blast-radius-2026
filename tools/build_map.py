"""Compile per-test trace shards into the compact dependency map the selector loads.

Everything here is derived from running the pinned sqlglot commit's own tests
(coverage, imports, files read, run times): there is no patch-to-answer
information in it.

Usage: python3 build_map.py --tests tests.json --out map.json.z [--timing dir] <shard.pkl>...
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import zlib
from collections import defaultdict

# Fixture files read through tests.helpers.load_sql_fixtures (one statement per entry);
# only used for traces that predate the loader kind being recorded.
SINGLES = {"identity.sql", "partial.sql", "optimizer/qualify_columns__invalid.sql"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", default="")
    ap.add_argument("--classinfo", default="")
    ap.add_argument("--light", default="", help="output of tracer/stmt_profile.py")
    ap.add_argument("--no-class-setup", default="", help="output of tracer/class_setup_check.py")
    ap.add_argument("--never-alone", default="", help="JSON: tests measured never to fail on their own")
    ap.add_argument("--reads", default="", help="glob of run_trace.py --reads shards")
    ap.add_argument("--reads-out", default="", help="where to write the table-read map (reads.json.z)")
    ap.add_argument("--list-expensive", default="", help="write ids of tests costing > 1 s CPU here")
    ap.add_argument("--probes", action="store_true", help="build the per-fixture-entry map")
    ap.add_argument("--test-shards", default="", help="glob of whole-test shards (with --probes)")
    ap.add_argument("--units", default="", help="output of tracer/stmt_spec.py (with --probes)")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()

    recs = {}
    if a.probes:
        # One record per fixture entry of a fixture-driven test; ids are "<test id>##<file>#<entry>".
        # A probe is forked after its test module was imported, so what that import executed
        # and instantiated is taken from the test's own (whole-test) record.
        at_import = {}
        for s in glob.glob(a.test_shards):
            for r in pickle.load(open(s, "rb")):
                if r.get("filter") is None and not r.get("error"):
                    at_import[r["id"]] = (r["imp"]["lines"], r["imp"]["classes"])
        # Tests traced one unit of their method at a time (ids "<test id>##@unit#<k>"): kept
        # only when every probe of the test was traced and passed.
        unit_spec = json.load(open(a.units)) if a.units else {}
        unit_recs = {}
        for s in a.shards:
            for r in pickle.load(open(s, "rb")):
                if r.get("probe") and not r["probe"].endswith("#all"):
                    lines, classes = at_import.get(r["id"], ((), ()))
                    r["imp"]["lines"] = list(r["imp"]["lines"]) + list(lines)
                    r["imp"]["classes"] = list(r["imp"]["classes"]) + list(classes)
                    tid = r["id"]
                    r["id"] = f"{tid}##{r['probe']}"
                    if r["probe"].startswith("@unit#") or tid in unit_spec:
                        unit_recs.setdefault(tid, []).append(r)
                    else:
                        recs[r["id"]] = r
        units_kept = {}
        for tid, rs in sorted(unit_recs.items()):
            units = unit_spec[tid]["units"]
            want = sum(u[0] for u in units) + (1 if any(not u[0] for u in units) else 0)
            bad = [r for r in rs if r.get("error") or not r.get("ok")]
            if len(rs) != want or bad:
                print(f"   units dropped for {tid}: {len(rs)}/{want} probes, {len(bad)} not ok "
                      f"{bad[0].get('problems') if bad else ''}")
                continue
            units_kept[tid] = units
            for r in rs:
                recs[r["id"]] = r
        print(f"unit probes: {len(units_kept)} tests, {sum(sum(u[0] for u in v) for v in units_kept.values())} units")
        tests = [{"id": i, "seconds": 0.0} for i in sorted(recs)]
    else:
        tests = json.load(open(a.tests))["tests"]
        for s in a.shards:
            for r in pickle.load(open(s, "rb")):
                if r.get("filter") is None:
                    recs[r["id"]] = r
    idx = {t["id"]: i for i, t in enumerate(tests)}
    missing = [t["id"] for t in tests if t["id"] not in recs]
    errors = [r["id"] for r in recs.values() if r.get("error") or not r.get("ok")]
    print(f"{len(recs)} records, missing={len(missing)}, not ok={len(errors)}")
    for e in errors[:20]:
        print("   NOT OK", e, recs[e].get("error") or recs[e].get("problems"))

    sets: dict[int, int] = {0: 0}

    def sid(bits: int) -> int:
        i = sets.get(bits)
        if i is None:
            i = sets[bits] = len(sets)
        return i

    run = defaultdict(int)   # (file, line) -> bitset of tests
    imp = defaultdict(int)
    top = defaultdict(int)
    modules = defaultdict(int)
    opened = defaultdict(int)
    listed = defaultdict(int)
    classes = defaultdict(int)       # instantiated at any time (including while importing)
    run_classes = defaultdict(int)   # instantiated while the test itself runs
    words_in = defaultdict(int)
    words_out = defaultdict(int)
    toktypes = defaultdict(int)
    fixtures: dict[str, dict[int, int]] = defaultdict(dict)
    trace_cpu = [0.0] * len(tests)

    def absorb(bit, snap, as_run=False):
        """as_run: an interpreter the test spawned; all of its activity is part of the test's run."""
        for ph in ("run", "imp"):
            target = run if (as_run or ph == "run") else imp
            for key in snap[ph]["lines"]:
                target[tuple(key)] |= bit
            for c in snap[ph]["classes"]:
                classes[c] |= bit
                if as_run or ph == "run":
                    run_classes[c] |= bit
            if as_run or ph == "run":
                for w in snap[ph]["words_in"]:
                    words_in[w] |= bit
                for w in snap[ph]["words_out"]:
                    words_out[w] |= bit
                for w in snap[ph]["toktypes"]:
                    toktypes[w] |= bit
        for key in snap["ilines"]:
            top[tuple(key)] |= bit
        for f in snap["files"]:
            opened[f] |= bit
        for d in snap["dirs"]:
            listed[d] |= bit

    for tid, r in recs.items():
        if r.get("error"):
            continue
        i = idx[tid]
        bit = 1 << i
        trace_cpu[i] = round(r["t_run_cpu"], 4)
        absorb(bit, r)
        if r.get("sub"):
            absorb(bit, r["sub"], as_run=True)
        for m in r.get("modules_at_end", []):
            modules[m] |= bit
        for k, n in r.get("fixture_counts", {}).items():
            if ":" not in k:
                k = ("singles:" if k in SINGLES else "pairs:") + k
            fixtures[k][i] = n

    files: dict[str, dict] = {}
    by_file = defaultdict(set)
    for f, line in set(run) | set(imp) | set(top):
        by_file[f].add(line)
    n_lines = 0
    for f, lines in sorted(by_file.items()):
        ls = sorted(lines)
        n_lines += len(ls)
        files[f] = {
            "lines": ls,
            "run": [sid(run.get((f, l), 0)) for l in ls],
            "imp": [sid(imp.get((f, l), 0)) for l in ls],
            "top": [sid(top.get((f, l), 0)) for l in ls],
        }

    out = {
        "tests": [t["id"] for t in tests],
        "files": files,
        "modules": {k: sid(v) for k, v in sorted(modules.items())},
        "opened": {k: sid(v) for k, v in sorted(opened.items())},
        "classes": {k: sid(v) for k, v in sorted(classes.items())},
        "run_classes": {k: sid(v) for k, v in sorted(run_classes.items())},
        "words_in": {k: sid(v) for k, v in sorted(words_in.items())},
        "words_out": {k: sid(v) for k, v in sorted(words_out.items())},
        "toktypes": {k: sid(v) for k, v in sorted(toktypes.items())},
        "fixtures": {k: {str(i): n for i, n in sorted(v.items())} for k, v in sorted(fixtures.items())},
        "trace_cpu": trace_cpu,
        "extra": {"dir_listers": {k: sid(v) for k, v in sorted(listed.items())}},
    }

    if a.timing:
        # Several timing runs may be given (comma separated): the quietest measurement wins.
        timing_dirs = a.timing.split(",")
        timing = {}
        for d in timing_dirs:
            for p in glob.glob(d + "/time_*.json"):
                for r in json.load(open(p)):
                    old = timing.get(r["id"])
                    if old is None or r.get("error"):
                        if old is None and not r.get("error"):
                            timing[r["id"]] = r
                        continue
                    for k in ("import", "run", "rerun"):
                        old[k] = min(old[k], r[k])
        rec_s = {t["id"]: t["seconds"] for t in tests}
        # What a class costs before its first test can run (setUpClass), estimated as the
        # smallest gap between a warm re-run of one test alone and its recorded duration.
        gaps = defaultdict(list)
        for tid, r in timing.items():
            cls = tid.rsplit("::", 1)[0]
            gaps[cls].append(r["rerun"] - rec_s[tid])
        setup = {}
        for cls, g in gaps.items():
            g.sort()
            s = max(0.0, g[len(g) // 4])  # lower quartile: robust to tests that run faster alone
            setup[cls] = round(s, 3) if s >= 0.05 else 0.0
        # Second pass, when present: tests of a class run together, timed one by one, which is
        # how the selector re-runs them. It gives the class fixture cost directly.
        in_suite = {}
        suite_setup = {}
        for d in timing_dirs:
            for p in glob.glob(d + "/suite_*.json"):
                for cls, r in json.load(open(p)).items():
                    if r.get("error"):
                        continue
                    suite_setup[cls] = min(suite_setup.get(cls, 1e9), r["setup"])
                    for tid, c in r["tests"].items():
                        in_suite[tid] = min(in_suite.get(tid, 1e9), c)
        for cls, s in suite_setup.items():
            setup[cls] = round(s, 3) if s >= 0.03 else 0.0
        cpu, iso, imp_cost, cold = [], [], [], []
        for t in tests:
            r = timing.get(t["id"])
            cls = t["id"].rsplit("::", 1)[0]
            if r is None:
                cpu.append(round(max(t["seconds"], 0.002), 4))
                iso.append(round(max(t["seconds"], 0.002), 4))
                imp_cost.append(0.15)
                cold.append(0.0)
                continue
            warm = in_suite.get(t["id"])
            if warm is None:
                warm = r["rerun"] - setup[cls]
            warm = max(0.001, warm)
            cpu.append(round(warm, 4))
            iso.append(round(max(0.002, r["run"]), 4))
            imp_cost.append(round(r["import"], 3))
            # What the test costs on top when it is the first to run in a fresh process
            # (lazy imports, cold caches).
            cold.append(round(max(0.0, r["run"] - setup[cls] - warm), 4))
        out["cpu"] = cpu
        out["cpu_iso"] = iso
        out["cold"] = cold
        out["import_cost"] = imp_cost
        out["class_setup"] = {k: v for k, v in setup.items() if v}
        print("class setup costs:", out["class_setup"])

    if a.probes:
        out["extra"]["units"] = units_kept
    if a.classinfo:
        info = json.load(open(a.classinfo))
        out["extra"]["trackable"] = info["trackable"]
    if a.light:
        out["extra"]["light"] = json.load(open(a.light))
    if a.no_class_setup:
        out["extra"]["no_class_setup"] = json.load(open(a.no_class_setup))
    if a.never_alone:
        out["extra"]["never_alone"] = json.load(open(a.never_alone))
    if a.list_expensive and "cpu" in out:
        with open(a.list_expensive, "w") as f:
            for t, c in zip(tests, out["cpu"]):
                if c > 1.0:
                    f.write(t["id"] + "\n")

    if a.reads and a.reads_out:
        # Which entries of the package's dict / set valued tables each test reads
        # (tracer/readtrace.py). Kept in a file of its own: it is only needed for changes
        # to such tables. Its test sets index into this map's "sets".
        r_all, r_size = defaultdict(int), defaultdict(int)
        r_key: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        ok_bits = 0
        tables: list[str] = []
        n_bad = 0
        for s in glob.glob(a.reads):
            for r in pickle.load(open(s, "rb")):
                if "tables" in r:
                    tables = r["tables"] or tables
                    continue
                i = idx.get(r["id"])
                if i is None:
                    continue
                # A test that failed while being observed, or that starts other interpreters
                # (which are not observed), counts as reading everything.
                if r.get("error") or not r.get("ok") or not r.get("reads") or (recs.get(r["id"]) or {}).get("n_sub"):
                    n_bad += 1
                    continue
                bit = 1 << i
                ok_bits |= bit
                for ph in ("run", "imp"):
                    d = r["reads"][ph]
                    for pth in d["all"]:
                        r_all[pth] |= bit
                    for pth in d["size"]:
                        r_size[pth] |= bit
                    for pth, ks in d["key"].items():
                        tab = r_key[pth]
                        for k in ks:
                            tab[k] |= bit
        reads = {
            "ok": sid(ok_bits),
            "tables": sorted(tables),
            "all": {k: sid(v) for k, v in sorted(r_all.items())},
            "size": {k: sid(v) for k, v in sorted(r_size.items())},
            "keys": {k: {kk: sid(vv) for kk, vv in sorted(v.items())} for k, v in sorted(r_key.items())},
        }
        raw_r = json.dumps(reads, separators=(",", ":")).encode()
        with open(a.reads_out, "wb") as f:
            f.write(zlib.compress(raw_r, 9))
        out["extra"]["reads"] = True
        print(f"table reads: {ok_bits.bit_count()} tests observed, {n_bad} not, {len(tables)} tables, "
              f"{sum(len(v) for v in r_key.values())} (table, key) pairs, raw={len(raw_r)/1e6:.1f}MB")

    ordered = sorted(sets.items(), key=lambda kv: kv[1])
    out["sets"] = [format(bits, "x") for bits, _ in ordered]
    raw = json.dumps(out, separators=(",", ":")).encode()
    comp = zlib.compress(raw, 9)
    with open(a.out, "wb") as f:
        f.write(comp)
    print(f"files={len(files)} lines={n_lines} unique sets={len(sets)} classes={len(classes)} "
          f"words_in={len(words_in)} words_out={len(words_out)} opened={len(opened)} listed={len(listed)} "
          f"raw={len(raw)/1e6:.1f}MB z={len(comp)/1e6:.2f}MB")


if __name__ == "__main__":
    main()
