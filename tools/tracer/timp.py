"""Which functions of the package do the *test modules* run while they are being imported?

The coverage map says for each test which lines of the package ran "at import", i.e. while
some module was being imported. For the package's own modules, what such code computed is
in their namespaces afterwards, where selector/state.py can compare it with the pinned
tree. Module-level code of a test file is different: what it computes (a constant built by
calling into the package, a class attribute ...) stays in the test module, which that
comparison does not look at. This tool lists the package functions reached that way, so
that changes to them keep the plain treatment (every test whose imports run them).

It imports every module under tests/ in one process and records each function of the
package that starts while the innermost module being imported is a test module (functions
run by a package module that the test module imports are not recorded: that is the package
importing itself).

Usage (inside br-dev):  python timp.py --root /src/sqlglot --out maps/timp.json.z
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import zlib
from importlib.machinery import PathFinder

STACK: list[str] = []            # modules of the repository being imported, innermost last
SEEN: set = set()                # code objects of the package started from test-module code
ROOT = ""
PKG = ""


class _Loader:
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        STACK.append(self._name)
        try:
            self._inner.exec_module(module)
        finally:
            STACK.pop()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, name, path=None, target=None):
        if not (name in ("sqlglot", "tests") or name.startswith(("sqlglot.", "tests."))):
            return None
        spec = PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _Loader(spec.loader, name)
        return spec


def _on_start(code, offset):
    if STACK and STACK[-1].startswith("tests") and code.co_filename.startswith(PKG) and code.co_name != "<module>":
        SEEN.add(code)


def main():
    global ROOT, PKG
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ROOT = os.path.join(os.path.abspath(a.root), "")
    PKG = os.path.join(ROOT, "sqlglot", "")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(here, "..", "..", "selector")))
    sys.path.insert(0, a.root)
    os.chdir(a.root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    import unittest  # noqa: F401 - loaded before the package, as under pytest

    import brh

    brh.install()
    sys.meta_path.insert(0, _Finder())
    mon = sys.monitoring
    tool = 2
    mon.use_tool_id(tool, "br-timp")
    mon.register_callback(tool, mon.events.PY_START, _on_start)
    mon.set_events(tool, mon.events.PY_START)
    names = []
    for dirpath, _dirs, files in os.walk(os.path.join(a.root, "tests")):
        for fn in sorted(files):
            if fn.endswith(".py"):
                rel = os.path.relpath(os.path.join(dirpath, fn), a.root)[:-3].replace(os.sep, ".")
                names.append(rel[: -len(".__init__")] if rel.endswith(".__init__") else rel)
    failed = {}
    for name in sorted(names):
        try:
            importlib.import_module(name)
        except BaseException as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"[:200]
    mon.set_events(tool, 0)
    spans: dict[str, list] = {}
    for code in SEEN:
        rel = code.co_filename[len(ROOT):]
        last = max((ln for _, _, ln in code.co_lines() if ln), default=code.co_firstlineno)
        spans.setdefault(rel, []).append([code.co_firstlineno, last, code.co_qualname])
    out = {"files": {}, "modules": len(names), "failed": failed}
    for rel, items in sorted(spans.items()):
        items.sort()
        merged: list[list[int]] = []
        for lo, hi, _ in items:
            if merged and lo <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        out["files"][rel] = merged
    with open(a.out, "wb") as f:
        f.write(zlib.compress(json.dumps(out, separators=(",", ":")).encode(), 9))
    n = sum(len(v) for v in spans.values())
    print(f"{len(names)} test modules imported ({len(failed)} failed), {n} package functions run by their "
          f"module-level code, in {len(spans)} files")
    for name, err in failed.items():
        print("   failed:", name, err)
    for rel, items in sorted(spans.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"   {rel}: {len(items)} e.g. {[q for _, _, q in items[:4]]}")


if __name__ == "__main__":
    main()
