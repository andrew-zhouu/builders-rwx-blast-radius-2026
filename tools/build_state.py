"""Static facts about the pinned sqlglot tree that the import-state comparison needs
(see selector/state.py): the list of its modules, which module imports which, and the
definitions whose fingerprint is not reproducible from one process to the next (values
built from the iteration order of sets), which the comparison must ignore.

The fingerprint itself is taken at run time, in a child process forked like the ones
that import patched trees, so the two are always comparable.

Usage: python build_state.py <sqlglot root> <out.json.z> [--check]
  --check  also verify, module by module, that importing only a module's dependents
           yields the same fingerprint as importing everything (no false differences).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib


def setup(root: str):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(here, ".."))
    import unittest  # noqa: F401 - the tests run under unittest; some flags look for it

    from selector import state

    return state


def main():
    if sys.argv[1] == "--take":
        state = setup(sys.argv[2])
        state.import_all(state.module_names())
        json.dump(state.snapshot(), sys.stdout)
        return
    root, out_path = sys.argv[1], sys.argv[2]
    state = setup(root)
    names = state.module_names()
    imports = state.static_imports(root, names)
    shots = []
    for seed in ("1", "7919", "123456"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, os.path.abspath(__file__), "--take", root],
                             env=env, capture_output=True, text=True, check=True)
        shots.append(json.loads(res.stdout))
    unstable = sorted(p for p in set().union(*shots) if len({s.get(p) for s in shots}) > 1)
    if "--check" in sys.argv:
        check(state, names, imports, set(unstable))
    as_object = state.modules_used_as_objects(
        [os.path.join(root, "sqlglot"), os.path.join(root, "tests")], names)
    out = {"modules": names, "imports": imports, "unstable": unstable, "as_object": as_object}
    raw = json.dumps(out, separators=(",", ":")).encode()
    with open(out_path, "wb") as f:
        f.write(zlib.compress(raw, 9))
    private = [n for n in names if n not in set(as_object)]
    print(f"modules={len(names)} import edges={sum(len(v) for v in imports.values())} "
          f"definitions={len(shots[0])} unstable={unstable} "
          f"never used as objects={len(private)} e.g. {private[:6]} raw={len(raw)/1e3:.0f}kB")


def _in_child(fn):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            payload = json.dumps(fn()).encode()
            while payload:
                payload = payload[os.write(w, payload):]
        finally:
            os._exit(0)
    os.close(w)
    buf = b""
    while chunk := os.read(r, 1 << 16):
        buf += chunk
    os.close(r)
    os.waitpid(pid, 0)
    return json.loads(buf)


def check(state, names, imports, unstable):
    def full():
        failed = state.import_all(names)
        return {"failed": failed, "state": state.snapshot()}

    base = _in_child(full)
    print(f"full import: {len(base['state'])} paths, failures {base['failed']}")
    bad = 0
    for name in names:
        scope = state.dependents(imports, {name})

        def scoped():
            failed = state.import_all([n for n in names if n in scope])
            cur = state.snapshot(only=scope)
            return {"failed": failed, "diff": state.diff(base["state"], cur, unstable, only=scope)}

        res = _in_child(scoped)
        if res["diff"] or res["failed"]:
            bad += 1
            print(f"  {name}: scope {len(scope)} modules, {len(res['diff'])} false differences, "
                  f"failures {res['failed']}")
            for p in res["diff"][:6]:
                print("       ", p)
    print(f"scope check: {bad} of {len(names)} modules give false differences")


if __name__ == "__main__":
    main()
