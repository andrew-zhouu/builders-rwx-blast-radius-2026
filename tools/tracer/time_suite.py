"""Measure, per test class, what its fixtures cost and what each test costs once the
class is set up (CPU seconds, tests of the class run together under the shared harness).

Usage (inside the build image):
  python time_suite.py --root <sqlglot> --harness <selector dir> --tests tests.json --out out.json --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict


def cpu_now():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def run_class(ids, root, harness):
    import unittest

    sys.path.insert(0, root)
    sys.path.append(harness)
    os.chdir(root)
    os.environ["PYTHONPATH"] = root
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    import brh

    c0 = cpu_now()
    brh.install()
    cases = []
    for tid in ids:
        case = brh.load_case(tid)
        case._br_id = tid
        cases.append(case)
    c1 = cpu_now()
    per_test: dict[str, float] = {}
    marks = {"first": None, "start": 0.0, "failed": []}

    class R(unittest.TestResult):
        def startTest(self, test):
            now = cpu_now()
            if marks["first"] is None:
                marks["first"] = now
            marks["start"] = now

        def stopTest(self, test):
            per_test[test._br_id] = cpu_now() - marks["start"]

        def addError(self, test, err):
            marks["failed"].append(str(test))

        def addFailure(self, test, err):
            marks["failed"].append(str(test))

        def addSubTest(self, test, subtest, err):
            if err is not None:
                marks["failed"].append(str(test))

    unittest.TestSuite(cases).run(R())
    return {
        "import": c1 - c0,
        "setup": (marks["first"] or c1) - c1,
        "tests": per_test,
        "failed": marks["failed"][:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    tests = json.load(open(a.tests))["tests"]
    by_class = defaultdict(list)
    for t in tests:
        by_class[t["id"].rsplit("::", 1)[0]].append(t)
    # Heaviest classes first, dealt round-robin over shards.
    classes = sorted(by_class, key=lambda c: -sum(t["seconds"] for t in by_class[c]))
    i, n = (int(x) for x in a.shard.split("/"))
    classes = [c for k, c in enumerate(classes) if k % n == i]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    out = {}
    t0 = time.time()
    for cls in classes:
        ids = [t["id"] for t in by_class[cls]]
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                payload = json.dumps(run_class(ids, a.root, a.harness)).encode()
            except BaseException as e:  # noqa: BLE001
                payload = json.dumps({"error": f"{type(e).__name__}: {e}"[:300]}).encode()
            while payload:
                payload = payload[os.write(w, payload):]
            os._exit(0)
        os.close(w)
        buf = b""
        while chunk := os.read(r, 65536):
            buf += chunk
        os.close(r)
        os.waitpid(pid, 0)
        out[cls] = json.loads(buf) if buf else {"error": "no output"}
    json.dump(out, open(a.out, "w"))
    bad = {c: v.get("error") or v.get("failed") for c, v in out.items() if v.get("error") or v.get("failed")}
    total = sum(sum(v.get("tests", {}).values()) for v in out.values())
    print(f"[{a.shard}] {len(out)} classes, tests cpu {total:.1f}s, {time.time()-t0:.0f}s wall, problems: {bad}", flush=True)


if __name__ == "__main__":
    main()
