"""What a change to a fixture file means for the tests that load it.

Fixture files are read by two loaders in sqlglot's tests/helpers.py. The pinned copy of
that module is loaded here, so a file is split into entries exactly the way the tests
will split it, and a file the loader chokes on is known to break every test loading it.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

_HELPERS: dict[str, object] = {}


def _pinned_helpers(src_root: str):
    mod = _HELPERS.get(src_root)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            "br_pinned_test_helpers", os.path.join(src_root, "tests", "helpers.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HELPERS[src_root] = mod
    return mod


def load_entries(src_root: str, kind: str, text: str) -> list[str]:
    """Entries the loader yields for a fixture with this content ("pairs" or "singles").
    Raises whatever the loader raises on it."""
    helpers = _pinned_helpers(src_root)
    fd, path = tempfile.mkstemp(prefix="br_fixture_", suffix=".sql")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        loader = helpers.load_sql_fixture_pairs if kind == "pairs" else helpers.load_sql_fixtures
        # An absolute path makes the loader's os.path.join(FIXTURES_DIR, name) return it unchanged.
        return [repr(item) for item in loader(path)]
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def plan_fixture(dm, plan, rel: str, old: str | None, new: str | None, src_root: str,
                 entry_users=None, helpers_changed: bool = False) -> None:
    """entry_users(name) -> bitset of tests whose behaviour depends on the entries of the
    fixture (None when unknown)."""
    name = rel[len("tests/fixtures/"):]
    by_kind = {}
    loader_users = 0
    for kind in ("pairs", "singles"):
        bits = 0
        for i in dm.fixtures.get(f"{kind}:{name}", {}):
            bits |= 1 << int(i)
        by_kind[kind] = bits
        loader_users |= bits
    readers = dm.openers(rel)
    direct = readers & ~loader_users   # tests reading the file some other way
    if old is None or new is None or helpers_changed or direct or not loader_users:
        plan.warm_ok = plan.hot_ok = False
        plan.add(readers | loader_users)
        plan.whole_only |= readers | loader_users
        plan.note(f"{rel}: re-run {(readers | loader_users).bit_count()} readers in full")
        return
    changed: set[int] = set()
    broken = 0
    for kind, bits in by_kind.items():
        if not bits:
            continue
        try:
            before = load_entries(src_root, kind, old)
        except Exception:  # noqa: BLE001 - cannot reason about entries: re-run in full
            plan.warm_ok = plan.hot_ok = False
            plan.add(bits)
            plan.whole_only |= bits
            continue
        try:
            after = load_entries(src_root, kind, new)
        except Exception as e:  # noqa: BLE001
            broken |= bits
            plan.note(f"{rel}: loader fails on the new file ({type(e).__name__}): "
                      f"{bits.bit_count()} tests load it")
            continue
        changed.update(i for i, e in enumerate(after) if i >= len(before) or before[i] != e)
    if broken:
        plan.blind |= broken    # the loader raises for every test that loads the file
    remaining = loader_users & ~broken
    if changed and remaining:
        dependent = entry_users(name) if entry_users is not None else None
        if dependent is not None:
            remaining &= dependent
        plan.add_sliced(remaining)
        plan.fixture_filter[name] = sorted(changed)
        plan.note(f"{rel}: {len(changed)} fixture entries changed, {remaining.bit_count()} tests use the entries")
    elif not broken:
        plan.note(f"{rel}: no fixture entry changed")
