"""Measure what re-running each test costs the selector: CPU seconds of one test
run alone in a forked child under the shared harness (no tracing).

Usage (inside br-dev): python time_tests.py --root /src/sqlglot --tests tests.json --out out.json --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def cpu_now():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def run_one(test, root, harness):
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
    brh.load_case(test["id"])
    c1 = cpu_now()
    failed = brh.run_tests([test["id"]])
    c2 = cpu_now()
    # A second run in the same process shows the cost once class fixtures and caches are warm.
    failed2 = brh.run_tests([test["id"]])
    c3 = cpu_now()
    return {"n": test["number"], "id": test["id"], "import": c1 - c0, "run": c2 - c1, "rerun": c3 - c2,
            "ok": not failed and not failed2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    tests = json.load(open(a.tests))["tests"]
    i, n = (int(x) for x in a.shard.split("/"))
    order = sorted(tests, key=lambda t: -t["seconds"])
    tests = [t for k, t in enumerate(order) if k % n == i]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    out = []
    t0 = time.time()
    for test in tests:
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                rec = run_one(test, a.root, a.harness)
                os.write(w, json.dumps(rec).encode())
            except BaseException as e:  # noqa: BLE001
                os.write(w, json.dumps({"n": test["number"], "id": test["id"], "error": str(e)[:200]}).encode())
            finally:
                os._exit(0)
        os.close(w)
        buf = b""
        while True:
            d = os.read(r, 65536)
            if not d:
                break
            buf += d
        os.close(r)
        os.waitpid(pid, 0)
        out.append(json.loads(buf) if buf else {"n": test["number"], "id": test["id"], "error": "no output"})
    json.dump(out, open(a.out, "w"))
    bad = [r for r in out if r.get("error") or not r.get("ok")]
    print(f"[{a.shard}] {len(out)} tests, {len(bad)} bad, {time.time()-t0:.0f}s, "
          f"sum import={sum(r.get('import', 0) for r in out):.1f}s run={sum(r.get('run', 0) for r in out):.1f}s "
          f"rerun={sum(r.get('rerun', 0) for r in out):.1f}s", flush=True)
    for r in bad[:8]:
        print("   BAD", r)


if __name__ == "__main__":
    main()
