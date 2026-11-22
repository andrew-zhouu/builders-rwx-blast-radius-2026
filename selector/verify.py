"""Re-run chosen tests against a patched copy of the tree, in a forked child.

The patched tree is a farm of symlinks into the read-only pinned checkout in
which only the directories leading to changed files are real, so building it
costs a few system calls.
"""
from __future__ import annotations

import gc
import json
import os
import select
import shutil
import signal
import sys
import time

GC_FREEZE = int(os.environ.get("BR_GC_FREEZE", "0"))
GC_GEN0 = int(os.environ.get("BR_GC_GEN0", "0"))   # allocations between young collections in children (0: default)


def build_overlay(src_root: str, dst_root: str, changed: dict[str, str | None]) -> None:
    """changed: repo-relative path -> new text, or None when the file is deleted."""
    tree: dict = {}
    for rel, content in changed.items():
        node = tree
        parts = rel.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"path conflict at {rel}")
        node[parts[-1]] = ("file", content)

    def build(src_dir: str | None, dst_dir: str, node: dict):
        os.mkdir(dst_dir)
        existing = sorted(os.listdir(src_dir)) if src_dir and os.path.isdir(src_dir) else []
        for name in existing:
            if name in node:
                continue
            os.symlink(os.path.join(src_dir, name), os.path.join(dst_dir, name))
        for name, sub in node.items():
            dst = os.path.join(dst_dir, name)
            src = os.path.join(src_dir, name) if src_dir else None
            if isinstance(sub, dict):
                build(src if src and os.path.isdir(src) else None, dst, sub)
            else:
                content = sub[1]
                if content is not None:
                    with open(dst, "w", encoding="utf-8", newline="") as f:
                        f.write(content)

    build(src_root, dst_root, tree)


def remove_overlay(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


# -- parent side: pinned modules kept warm ------------------------------------------
_MON_TOOL = 4
# Code objects of the pinned tree that have started running in this (parent) process:
# whatever the imports called.
_STARTED: set = set()
# (file, qualified name) -> function objects defined by the pinned modules
_FUNCS: dict = {}
_ROOT = ""


def _on_start(code, offset):
    _STARTED.add(code)
    return sys.monitoring.DISABLE


def warm_up(src_root: str, harness_dir: str, modules=()) -> bool:
    """Import the pinned sqlglot (every module in `modules`) and the test harness in this
    (parent) process, so that children handling patches which leave what the imports
    compute as it is can skip the imports. Records which functions the imports ran."""
    global _ROOT
    try:
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        if harness_dir not in sys.path:
            sys.path.append(harness_dir)
        _ROOT = os.path.join(os.path.abspath(src_root), "")
        # sqlglot looks at import time whether a test runner is loaded (expressions.core.UNITTEST
        # turns on a stricter argument check): the tests must see it as they do under pytest.
        import unittest  # noqa: F401

        watching = False
        if modules:
            try:
                mon = sys.monitoring
                mon.use_tool_id(_MON_TOOL, "br-warm")
                mon.register_callback(_MON_TOOL, mon.events.PY_START, _on_start)
                mon.set_events(_MON_TOOL, mon.events.PY_START)
                watching = True
            except Exception:  # noqa: BLE001
                watching = False
        import brh

        brh.install()
        import sqlglot  # noqa: F401
        import importlib

        complete = watching
        for name in sorted(modules):
            try:
                importlib.import_module(name)
            except BaseException:  # noqa: BLE001
                complete = False
        if complete:
            import gc
            import types

            for obj in gc.get_objects():
                if type(obj) is types.FunctionType:
                    code = obj.__code__
                    if code.co_filename.startswith(_ROOT):
                        _FUNCS.setdefault((code.co_filename[len(_ROOT):], code.co_qualname), []).append(obj)
        return True
    except BaseException:  # noqa: BLE001
        return False


def can_swap() -> bool:
    return bool(_FUNCS)


class Unswappable(Exception):
    pass


def _find_code(code, qualname: str):
    stack = [code]
    while stack:
        c = stack.pop()
        for const in c.co_consts:
            if hasattr(const, "co_code"):
                if const.co_qualname == qualname:
                    return const
                stack.append(const)
    return None


_ARG_FLAGS = 0x04 | 0x08 | 0x20 | 0x80 | 0x100 | 0x200   # *args, **kwargs, generator, coroutine kinds
_FUTURE_ANNOTATIONS = 0x1000000


def apply_swaps(swaps) -> None:
    """Give the pinned function objects of this process the code of the patched tree.

    Only for functions that have not run yet in this process (nor any function nested in
    them): then nothing computed so far depends on their old code. Raises Unswappable
    when a function cannot be replaced this way."""
    ran: dict[str, set] = {}
    for code in _STARTED:
        fn = code.co_filename
        if fn.startswith(_ROOT):
            ran.setdefault(fn[len(_ROOT):], set()).add(code.co_qualname)
    for sw in swaps:
        rel, chain = sw["rel"], sw["cls"]
        qual = f"{chain}.{sw['name']}" if chain else sw["name"]
        funcs = [f for f in _FUNCS.get((rel, qual), ()) if f.__code__.co_firstlineno == sw["old"][0]]
        if not funcs:
            raise Unswappable(f"{rel}:{qual} is not a loaded function")
        inner = qual + ".<locals>."
        if any(q == qual or q.startswith(inner) for q in ran.get(rel, ())):
            raise Unswappable(f"{rel}:{qual} already ran")
        parts = chain.split(".") if chain else []
        first = sw["new"][0]
        if sw["col"] != 4 * len(parts) or first - 1 < len(parts):
            raise Unswappable(f"{rel}:{qual} layout")
        # The def inside stand-ins for its classes (same names, so private names are mangled
        # alike), padded so that line numbers are those of the patched file.
        head = "".join(f"{'    ' * d}class {c}:\n" for d, c in enumerate(parts))
        src = "\n" * (first - 1 - len(parts)) + head + sw["text"] + "\n"
        old = funcs[0].__code__
        try:
            module_code = compile(src, old.co_filename, "exec", flags=old.co_flags & _FUTURE_ANNOTATIONS,
                                  dont_inherit=True)
        except (SyntaxError, ValueError) as e:
            raise Unswappable(f"{rel}:{qual} does not compile: {e}") from None
        new = _find_code(module_code, qual)
        if new is None:
            raise Unswappable(f"{rel}:{qual} not found in the new code")
        for f in funcs:
            old = f.__code__
            n_args = old.co_argcount + old.co_kwonlyargcount
            if (old.co_freevars != new.co_freevars or old.co_argcount != new.co_argcount
                    or old.co_posonlyargcount != new.co_posonlyargcount
                    or old.co_kwonlyargcount != new.co_kwonlyargcount
                    or (old.co_flags ^ new.co_flags) & _ARG_FLAGS
                    or old.co_varnames[:n_args] != new.co_varnames[:n_args]):
                raise Unswappable(f"{rel}:{qual} signature")
        for f in funcs:
            f.__code__ = new


def warm_tests(test_ids: list[str]) -> None:
    """Import the test modules of these tests in the parent (pinned code)."""
    import brh

    seen = set()
    for tid in test_ids:
        f = tid.split("::", 1)[0]
        if f not in seen:
            seen.add(f)
            try:
                brh.load_case(tid)
            except BaseException:  # noqa: BLE001
                pass


# -- child side ---------------------------------------------------------------------
def _limit_memory() -> None:
    """If patched code allocates without bound and the container runs out of memory, this
    child is the process to kill, not the selector. (An address-space limit would be
    tidier but breaks DuckDB, which reserves large ranges.)"""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:  # noqa: BLE001
        pass


def enter_patched_world(overlay: str, harness_dir: str, warm_root: str | None = None,
                        parent_root: str | None = None) -> None:
    """Make this (child) process see the patched tree.

    warm_root: the parent has the pinned code imported and the patch does not change
    Python code, so those modules are reused and only fixtures are redirected.
    parent_root: the parent has the pinned code imported; it is dropped here so that
    everything is imported again from the patched tree."""
    import unittest  # noqa: F401 - loaded before sqlglot, as under pytest (see warm_up)

    sys.dont_write_bytecode = True
    os.environ["PYTHONPATH"] = overlay
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["SKIP_INTEGRATION"] = "0"
    os.chdir(overlay)
    if harness_dir not in sys.path:
        sys.path.append(harness_dir)
    import brh

    if warm_root is not None:
        import tests.helpers as helpers

        helpers.FIXTURES_DIR = os.path.join(overlay, "tests", "fixtures")
        return
    if parent_root is not None:
        for name in [m for m in sys.modules
                     if m in ("sqlglot", "tests") or m.startswith(("sqlglot.", "tests."))]:
            del sys.modules[name]
        sys.path[:] = [p for p in sys.path if p != parent_root]
    sys.path.insert(0, overlay)
    try:
        brh.install()
    except BaseException:  # tests/helpers.py itself may be what the patch broke
        try:
            brh.install(fixture_hooks=False)
        except BaseException:
            pass


def run_groups(groups: list[dict], drops: dict | None = None, stop_after: int = 0,
               keep: dict | None = None, scale: dict | None = None, phased: dict | None = None) -> dict:
    """Run groups of tests in order in this process. A group is
      {"ids": [test id], "filter": {fixture file: [entries]}, "skip_setup": [class key],
       "if_failed": bool}
    where if_failed marks a group that is only worth running once something has failed.

    drops: {test id: statement line spans to leave out of the test method}.
    keep: {test id: statement line spans the test method is reduced to}.
    Returns {"failed": {test id: reason}, "unverified": [...], "not_run": [...],
             "deferred": [test ids of if_failed groups that were not run],
             "pool_breaks": [tests during which a shared process pool would have broken],
             "bad_files": [test files that could not be imported]}."""
    import brh

    brh.DROP_STATEMENTS.update(drops or {})
    brh.KEEP_STATEMENTS.update(keep or {})
    brh.SCALE_TESTS.update(scale or {})
    brh.PHASED.update(phased or {})        # tests settled in two phases (brh.PHASE_RECIPES)
    report_times = bool(os.environ.get("BR_REPORT"))
    if scale and brh.TIMES is None:
        brh.TIMES = {}      # how long a stand-in takes is part of what it shows
    if os.environ.get("BR_REPORT"):
        brh.TIMES = {}
    t_start = time.process_time()
    failed: dict[str, str] = {}
    deferred: list[str] = []
    stopped_early = False
    for n, group in enumerate(groups):
        test_ids = group["ids"]
        if group.get("if_failed"):
            if not failed:
                deferred.extend(test_ids)
                continue
            if stopped_early:
                # The first tests all failed: the change breaks things wholesale, and what is
                # left is assumed to fail as well rather than paid for.
                brh.NOT_RUN.extend(test_ids)
                continue
        elif stop_after < 0 and failed:
            # One failure was all the caller needed to know.
            brh.NOT_RUN.extend(test_ids)
            continue
        brh.FIXTURE_FILTER.clear()
        for k, v in group.get("filter", {}).items():
            brh.FIXTURE_FILTER[k] = set(v)
        brh.SKIP_CLASS_SETUP.clear()
        brh.SKIP_CLASS_SETUP.update(group.get("skip_setup", ()))
        # Early stop only applies to the first, unfiltered group (whole tests, cheapest first).
        failed.update(brh.run_tests(test_ids, stop_after=stop_after if n == 0 and not group.get("filter") else 0))
        if n == 0 and stop_after > 0 and brh.NOT_RUN:
            stopped_early = True
    out = {"failed": failed, "unverified": list(brh.UNVERIFIED), "not_run": list(brh.NOT_RUN),
           "deferred": deferred, "pool_breaks": list(brh.POOL_BREAKS), "bad_files": sorted(brh.BAD_FILES)}
    if scale:
        out["scaled_s"] = {t: round(brh.TIMES.get(t, -1.0), 3) for t in scale}
    if brh.PHASE_LOG:
        out["phase"] = dict(brh.PHASE_LOG)     # test: [queries looked at, left, executed]
    if brh.TIMES is not None and report_times:
        top = sorted(brh.TIMES.items(), key=lambda kv: -kv[1])[:12]
        out["times"] = {"total": round(time.process_time() - t_start, 3),
                        "tests": round(sum(v for v in brh.TIMES.values() if v > 0), 3),
                        "top": [[t.split("::", 1)[1], round(v, 3)] for t, v in top]}
    return out


def fork_call(fn, timeout: float = 30.0):
    """Run fn() in a forked child and return its JSON-serialisable result, or None when
    the child hung, crashed or produced nothing."""
    rfd, wfd = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    if GC_FREEZE:
        # What this process holds is shared with the child copy-on-write; keeping the
        # collector away from it keeps it shared.
        gc.freeze()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            if GC_GEN0:
                gc.set_threshold(GC_GEN0, 20, 20)
            os.close(rfd)
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.setsid()
            _limit_memory()
            if _STARTED:
                try:   # the parent's bookkeeping of what its imports ran is not needed here
                    sys.monitoring.set_events(_MON_TOOL, 0)
                except Exception:  # noqa: BLE001
                    pass
            payload = json.dumps(fn()).encode()
            while payload:
                payload = payload[os.write(wfd, payload):]
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    os.close(wfd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    alive = True
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                alive = False
                break
            ready, _, _ = select.select([rfd], [], [], left)
            if not ready:
                alive = False
                break
            data = os.read(rfd, 1 << 16)
            if not data:
                break
            chunks.append(data)
    finally:
        os.close(rfd)
    if not alive:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    # A test may leave helpers behind (process pools, spawned interpreters).
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if not alive or not chunks:
        return None
    try:
        return json.loads(b"".join(chunks))
    except ValueError:
        return None


def failing_imports(names) -> dict:
    """{module: error} for those of the given modules that cannot be imported here."""
    from . import state as st

    return st.import_all(list(names))


def run_tests(overlay: str, harness_dir: str, groups, timeout: float = 30.0, drops=None, stop_after: int = 0,
              warm_root: str | None = None, parent_root: str | None = None, swaps=None, keep=None,
              check=None, scale=None, phased=None):
    """Re-run the groups against the patched tree in one child; None when it hung or died.

    swaps: with warm_root, functions whose code is replaced in the child (apply_swaps);
    the result is {"unswappable": reason} when that is not possible.
    check: modules to import afterwards; the failing ones come back under "import_failed"."""

    def job():
        enter_patched_world(overlay, harness_dir, warm_root, parent_root)
        if swaps:
            try:
                apply_swaps(swaps)
            except Unswappable as e:
                return {"unswappable": str(e)}
        out = run_groups(groups, drops, stop_after, keep, scale, phased)
        if check:
            out["import_failed"] = failing_imports(check)
        return out

    return fork_call(job, timeout)


def probe_imports(overlay: str, harness_dir: str, names, timeout: float = 30.0,
                  parent_root: str | None = None):
    """Import the given modules from the patched tree in a child. {module: error} for the
    failing ones; None when the child hung or died."""

    def job():
        enter_patched_world(overlay, harness_dir, None, parent_root)
        return {"failed": failing_imports(names)}

    res = fork_call(job, timeout)
    return None if res is None else res["failed"]
