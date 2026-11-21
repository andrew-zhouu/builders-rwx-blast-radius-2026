"""Re-run chosen tests against a patched copy of the tree, in a forked child.

The patched tree is a farm of symlinks into the read-only pinned checkout in
which only the directories leading to changed files are real, so building it
costs a few system calls.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import signal
import sys
import time

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
def warm_up(src_root: str, harness_dir: str) -> bool:
    """Import the pinned sqlglot and the test harness in this (parent) process, so that
    children handling patches which leave the Python code as it is can skip the imports."""
    try:
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        if harness_dir not in sys.path:
            sys.path.append(harness_dir)
        import brh

        brh.install()
        import sqlglot  # noqa: F401
        return True
    except BaseException:  # noqa: BLE001
        return False


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


def run_groups(groups: list[dict], drops: dict | None = None, stop_after: int = 0) -> dict:
    """Run groups of tests in order in this process. A group is
      {"ids": [test id], "filter": {fixture file: [entries]}, "skip_setup": [class key],
       "if_failed": bool}
    where if_failed marks a group that is only worth running once something has failed.

    drops: {test id: statement line spans to leave out of the test method}.
    Returns {"failed": {test id: reason}, "unverified": [...], "not_run": [...],
             "deferred": [test ids of if_failed groups that were not run]}."""
    import brh

    brh.DROP_STATEMENTS.update(drops or {})
    failed: dict[str, str] = {}
    deferred: list[str] = []
    for n, group in enumerate(groups):
        test_ids = group["ids"]
        if group.get("if_failed"):
            if not failed:
                deferred.extend(test_ids)
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
    return {"failed": failed, "unverified": list(brh.UNVERIFIED), "not_run": list(brh.NOT_RUN),
            "deferred": deferred}


def fork_call(fn, timeout: float = 30.0):
    """Run fn() in a forked child and return its JSON-serialisable result, or None when
    the child hung, crashed or produced nothing."""
    rfd, wfd = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(rfd)
            devnull = os.open(os.devnull, os.O_RDWR)
            os.dup2(devnull, 0)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.setsid()
            _limit_memory()
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


def run_tests(overlay: str, harness_dir: str, groups, timeout: float = 30.0, drops=None, stop_after: int = 0,
              warm_root: str | None = None, parent_root: str | None = None):
    """Re-run the groups against the patched tree in one child; None when it hung or died."""

    def job():
        enter_patched_world(overlay, harness_dir, warm_root, parent_root)
        return run_groups(groups, drops, stop_after)

    return fork_call(job, timeout)
