"""Sanity check of the shared test harness: every test of the pinned tree must pass when
run the way the selector re-runs tests (selector/brh.py: in-process pools, capped DuckDB
threads), class by class in forked children.

Usage (inside the image): python harness_check.py <sqlglot root> <tests.json> [shard i/n]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict


def main():
    root, tests_path = sys.argv[1], sys.argv[2]
    i, n = (int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0/1").split("/"))
    here = os.path.dirname(os.path.abspath(__file__))
    harness = os.path.normpath(os.path.join(here, "..", "selector"))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    tests = json.load(open(tests_path))["tests"]
    by_class = defaultdict(list)
    for t in tests:
        by_class[t["id"].rsplit("::", 1)[0]].append(t["id"])
    classes = [c for k, c in enumerate(sorted(by_class)) if k % n == i]
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    bad: dict[str, str] = {}
    total = 0
    t0 = time.time()
    for cls in classes:
        ids = by_class[cls]
        total += len(ids)
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                devnull = os.open(os.devnull, os.O_RDWR)
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                sys.path.insert(0, root)
                sys.path.append(harness)
                os.chdir(root)
                os.environ["PYTHONPATH"] = root
                import brh

                brh.install()
                payload = json.dumps(brh.run_tests(ids)).encode()
            except BaseException as e:  # noqa: BLE001
                payload = json.dumps({ids[0]: f"harness crashed: {type(e).__name__}: {e}"}).encode()
            while payload:
                payload = payload[os.write(w, payload):]
            os._exit(0)
        os.close(w)
        buf = b""
        while chunk := os.read(r, 1 << 16):
            buf += chunk
        os.close(r)
        os.waitpid(pid, 0)
        bad.update(json.loads(buf) if buf else {ids[0]: "no output from child"})
    print(f"[{i}/{n}] {total} tests in {len(classes)} classes, {len(bad)} failing, {time.time() - t0:.0f}s")
    for tid, why in list(bad.items())[:20]:
        print("   FAIL", tid, "|", why[:200].replace("\n", " "))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
