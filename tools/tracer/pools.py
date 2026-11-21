"""Which tests share a process pool, and which code runs while a pool pickles?

Some test classes make one ProcessPoolExecutor in a class fixture and use it in many
tests. When a patch makes one of those tests kill a worker (a call that cannot be
unpickled in the worker does that, and so does a result the parent cannot unpickle), the
pool is broken for every test of the class that runs after it, whatever those tests test.
The selector has to know which tests those are, and which code a patch has to touch to be
able to do that.

This runs the whole suite on the pinned commit through the harness (selector/brh.py), whose
stand-in pool notes every call submitted to a pool that was made outside the running test,
and writes
  {"classes": {"tests/x.py::Class": [test ids that use the class's pool]},
   "sites":   {"tests/x.py::Class": ["file:line of the submit call"]},
   "pickle":  {"sqlglot/x.py": [[first line, last line] of each function that ran while
               any pool pickled or unpickled a call, a result or an exception]}}

Usage (inside br-dev):  python pools.py --root /src/sqlglot --tests tests.json --out maps/pools.json.z
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import zlib


def function_spans(source: str) -> list[tuple[int, int]]:
    """(first line incl. decorators, last line) of every function and lambda."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            out.append((first, node.end_lineno))
        elif isinstance(node, ast.Lambda):
            out.append((node.lineno, node.end_lineno))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "selector")))
    sys.path.insert(0, a.root)
    os.chdir(a.root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    import unittest  # noqa: F401 - loaded before the package, as under pytest

    import brh

    root = os.path.join(os.path.abspath(a.root), "")
    codes: set = set()

    def note(frame, event, arg):
        if event == "call":
            code = frame.f_code
            if code.co_filename.startswith(root):
                codes.add((code.co_filename, code.co_firstlineno))

    brh.PICKLE_TRACE = note
    brh.install()
    ids = [t["id"] for t in json.load(open(a.tests))["tests"]]
    failed = brh.run_tests(ids)
    brh.PICKLE_TRACE = None

    classes: dict[str, set] = {}
    sites: dict[str, set] = {}
    for tid, filename, line in brh.SHARED_SUBMITS:
        key = tid.rsplit("::", 1)[0]
        classes.setdefault(key, set()).add(tid)
        sites.setdefault(key, set()).add(f"{os.path.relpath(filename, a.root)}:{line}")

    by_file: dict[str, set] = {}
    for filename, first in codes:
        by_file.setdefault(os.path.relpath(filename, a.root).replace(os.sep, "/"), set()).add(first)
    pickle_time: dict[str, list] = {}
    for rel, firsts in sorted(by_file.items()):
        spans = function_spans(open(os.path.join(a.root, rel), encoding="utf-8").read())
        keep = set()
        for first in firsts:
            # the function that starts there, else the innermost one around the line
            # (a generator expression), else the line itself (module or class level)
            own = [s for s in spans if s[0] == first]
            around = sorted((s for s in spans if s[0] <= first <= s[1]), key=lambda s: s[1] - s[0])
            keep.add(own[0] if own else around[0] if around else (first, first))
        pickle_time[rel] = sorted(list(s) for s in keep)

    out = {"classes": {k: sorted(v) for k, v in sorted(classes.items())},
           "sites": {k: sorted(v) for k, v in sorted(sites.items())},
           "pickle": pickle_time}
    with open(a.out, "wb") as f:
        f.write(zlib.compress(json.dumps(out, separators=(",", ":")).encode(), 9))
    print(f"{len(ids)} tests run ({len(failed)} failed), pool breaks at the pinned commit: {brh.POOL_BREAKS}")
    for k, v in out["classes"].items():
        print(f"{k}: {len(v)} tests share a pool; calls submitted at {out['sites'][k]}")
        print("   ", [t.rsplit('::', 1)[1] for t in v])
    print("code that runs while pools pickle:")
    for rel, spans in pickle_time.items():
        print(f"   {rel}: {len(spans)} functions {spans[:14]}")


if __name__ == "__main__":
    main()
