"""From the definitions an import-time change really altered (state.diff paths) to the
tests that can observe it."""
from __future__ import annotations

import re
from collections import defaultdict

from .depmap import DepMap
from .plan import _class_keys, readers_of, untracked_class_readers

WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
MAX_PATHS = 600


def _class_from_ref(dm: DepMap, ref: str) -> list[str]:
    """'pkg.mod.Outer.Inner' -> map keys ('pkg.mod:Outer.Inner') that exist for it."""
    out = []
    parts = ref.split(".")
    for i in range(len(parts) - 1, 0, -1):
        key = ".".join(parts[:i]) + ":" + ".".join(parts[i:])
        if key in dm.classes:
            out.append(key)
    return out


def _narrow(dm: DepMap, owner: str, keys: set[str], cands: int) -> tuple[int, str]:
    """Restrict readers of a table to tests that can reach the changed entries."""
    low = owner.lower()
    parses = "parser" in low or "tokenizer" in low or "tokens" in low
    if all(k.startswith("c:sqlglot.expressions.") for k in keys):
        # A table keyed by expression classes is consulted for instances of those classes.
        users = 0
        for k in keys:
            for ck in _class_from_ref(dm, k[2:]):
                users |= dm.class_users(ck)
        return cands & users, "instances of the changed keys"
    if parses and all(k.startswith("e:TokenType.") for k in keys):
        users = 0
        for k in keys:
            sid = dm.toktypes.get(k.split(".", 1)[1])
            users |= dm.set(sid) if sid else 0
        return cands & users, "token types of the changed keys"
    if parses and all(k.startswith("s:") for k in keys):
        users = 0
        for k in keys:
            text = k[2:]
            words = WORDS.findall(text)
            if not (words and text == text.upper() and " ".join(words) == " ".join(text.split())
                    and all(len(w) >= 3 for w in words)):
                return cands, ""
            users |= dm.word_users(words)
        return cands & users, "sql containing the changed keys"
    return cands, ""


def _module_file(dm: DepMap, mod: str) -> str | None:
    base = mod.replace(".", "/")
    for cand in (base + ".py", base + "/__init__.py"):
        if cand in dm.files:
            return cand
    return None


def _by_reads(dm: DepMap, reads: dict, path: str, keys: set[str], added: set[str], removed: set[str]) -> int:
    """Tests that read the changed entries of a table (or were not observed)."""
    def tests(table: str) -> int:
        sid = reads[table].get(path)
        return dm.set(sid) if sid else 0

    key_tab = reads["keys"].get(path, {})
    readers = tests("all")
    # Without entry-level information, or with keys whose spelling is not reproducible,
    # every lookup counts; an entry that appears or disappears also changes the size.
    any_key = not keys or any(not k.startswith(("s:", "c:", "e:", "v:")) for k in keys)
    resized = any_key or any(f"{path}|{k}" in added or f"{path}|{k}" in removed for k in keys)
    if any_key:
        for sid in key_tab.values():
            readers |= dm.set(sid)
    else:
        for k in keys:
            sid = key_tab.get(k)
            if sid:
                readers |= dm.set(sid)
    if resized:
        readers |= tests("size")
    return readers | (dm.all & ~reads["ok"])


def from_paths(dm: DepMap, paths: list[str], added: set[str], failed: dict, facts: dict, cheap,
               notes: list[str], removed: set[str] = frozenset(), use_reads: bool = True) -> int | None:
    """Tests that can observe the changed definitions; None when the change is too wide
    to reason about (the caller keeps its conservative candidates).

    added: the paths that do not exist in the pinned tree.
    facts: output of tools/build_state.py (import graph, modules used as objects)."""
    import builtins

    from .state import dependents

    imports = facts.get("imports", {})
    as_object = set(facts.get("as_object") or imports)   # unknown: assume every module is

    if len(paths) > MAX_PATHS:
        notes.append(f"state: {len(paths)} definitions changed, too many to follow")
        return None
    groups: dict[tuple, set] = defaultdict(set)
    new_names = set()
    for p in paths:
        parts = p.split("|")
        if parts[1] == "g" and len(parts) == 3 and p in added and not hasattr(builtins, parts[2]):
            # A module-level name that did not exist: code reading it would have raised
            # before, so nothing that passes at the pinned commit reads it.
            new_names.add((parts[0], parts[2]))
            notes.append(f"state: new name {parts[0]}.{parts[2]} (unobservable)")
    for p in paths:
        parts = p.split("|")
        if parts[1] == "g":
            if (parts[0], parts[2]) in new_names:
                continue
            groups[(parts[0], None, parts[2])].add(parts[3] if len(parts) > 3 else None)
        else:
            groups[(parts[0], parts[2], parts[3])].add("|".join(parts[4:]) if len(parts) > 4 else None)
    bits = 0
    for (mod, cls, attr), keys in sorted(groups.items(), key=lambda kv: str(kv[0])):
        keys.discard(None)
        users = dm.module_users(mod)
        if attr == "<bases>":
            readers = None
        elif cls is None and mod not in as_object:
            # A global of a module nobody holds as an object: only that module's own code
            # reads it by name (importers get their own binding, reported separately).
            readers = readers_of(dm, attr, own_file=_module_file(dm, mod) or "?")
        else:
            readers = readers_of(dm, attr)
        if cls is None:
            cands = (readers[0] & users) | readers[1] if readers is not None else users
            how = "readers of the name" if readers is not None else "module users"
        else:
            ckeys = _class_keys(mod, cls)
            if ckeys[0] not in dm.trackable:
                readers = untracked_class_readers(dm, cls, readers)
                cands = (readers[0] & users) | readers[1] if readers is not None else users
                how = "untracked class, readers of the name or of the class" if readers is not None else "untracked class"
            else:
                # Expression nodes built while importing (templates) are copied before use,
                # so only tests creating such nodes themselves depend on the class body.
                run_only = mod.startswith("sqlglot.expressions")
                wide = dm.class_users(ckeys[0], run_only=run_only)
                for outer in ckeys[1:]:
                    wide |= dm.class_users(outer)
                cands = (wide & readers[0]) | readers[1] if readers is not None else wide
                how = "class users reading the name" if readers is not None else "class users"
        if keys and not cheap(cands):
            narrowed, why = _narrow(dm, f"{mod}.{cls or ''}", keys, cands)
            if why:
                cands, how = narrowed, f"{how}, {why}"
        reads = dm.reads() if use_reads else None
        path = f"{mod}|g|{attr}" if cls is None else f"{mod}|c|{cls}|{attr}"
        if reads is not None and path in reads["tables"] and not cheap(cands):
            # A dict / set valued table whose reads were recorded test by test.
            before = cands.bit_count()
            cands &= _by_reads(dm, reads, path, keys, added, removed)
            how = f"{how}, {cands.bit_count()} of {before} read the changed entries"
        notes.append(f"state: {mod} {cls or ''}.{attr} {sorted(keys)[:3]} -> {cands.bit_count()} ({how})")
        bits |= cands
    if failed:
        # Modules that no longer import: everything that loads them or a module importing them.
        for mod in dependents(imports, set(failed)):
            bits |= dm.module_users(mod)
        notes.append(f"state: import fails for {sorted(failed)[:4]}")
    return bits
