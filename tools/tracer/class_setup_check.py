"""For test classes with expensive class fixtures, find the tests that pass without them.

Such a test does not use what setUpClass builds, so the selector can re-run it without
paying for the fixtures (as long as the patch leaves the test file alone).

Usage (inside the build image):
  python class_setup_check.py --root <sqlglot> --harness <selector dir> --tests tests.json \
      --timing <dir with suite_*.json> --out out.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

MIN_SETUP_S = 0.3


def check(ids, root, harness):
    sys.path.insert(0, root)
    sys.path.append(harness)
    os.chdir(root)
    os.environ["PYTHONPATH"] = root
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    import brh

    brh.install()
    brh.SKIP_CLASS_SETUP.add(ids[0].rsplit("::", 1)[0])
    failed = brh.run_tests(ids)
    return [t for t in ids if t not in failed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    setup = {}
    for p in glob.glob(a.timing + "/suite_*.json"):
        for cls, r in json.load(open(p)).items():
            if not r.get("error"):
                setup[cls] = r["setup"]
    tests = json.load(open(a.tests))["tests"]
    import unittest  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import duckdb  # noqa: F401

    ok: list[str] = []
    for cls, cost in sorted(setup.items()):
        if cost < MIN_SETUP_S:
            continue
        ids = [t["id"] for t in tests if t["id"].rsplit("::", 1)[0] == cls]
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(r)
                payload = json.dumps(check(ids, a.root, a.harness)).encode()
            except BaseException:  # noqa: BLE001
                payload = b"[]"
            while payload:
                payload = payload[os.write(w, payload):]
            os._exit(0)
        os.close(w)
        buf = b""
        while chunk := os.read(r, 65536):
            buf += chunk
        os.close(r)
        os.waitpid(pid, 0)
        passing = json.loads(buf) if buf else []
        print(f"{cls}: setup {cost:.2f}s, {len(passing)}/{len(ids)} tests pass without it", flush=True)
        ok.extend(passing)
    json.dump(sorted(ok), open(a.out, "w"))


if __name__ == "__main__":
    main()
