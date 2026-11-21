"""Work out which tests a patch could affect, from maps of the pinned commit.

The result is a superset ("candidates"); whether a candidate really fails is
decided afterwards by running it against the patched tree.
"""
from __future__ import annotations

import os
import re

from . import astdiff
from .depmap import DepMap
from .fixtures import plan_fixture
from .patch import FilePatch, PatchError, apply_hunks


class Plan:
    def __init__(self):
        self.full = 0                            # tests to re-run completely (bitset over map tests)
        self.sliced = 0                          # tests where only changed fixture entries need a re-run
        self.changed: dict[str, str | None] = {}  # path -> new content (None = deleted)
        self.fixture_filter: dict[str, list[int]] = {}
        self.select_all = False                  # could not analyse: run everything
        self.blind = 0                           # tests selected without a re-run (cannot be re-run)
        # Import-time changes whose effect can be read off the import-time state:
        self.import_static = 0                   # ... their statically derived candidates
        self.import_modules: set[str] = set()    # ... the modules they are in
        self.import_words = 0                    # ... tests whose SQL contains a changed string constant
        # Some of them stand for function bodies that the imports run (see plan_python).
        self.import_runs = False
        # True while every candidate comes from statement coverage of changed function
        # bodies: then each candidate really executes the changed code.
        self.all_call_time = True
        # True while the patch leaves the behaviour of all Python code as it is and only
        # changes fixture entries read through the loaders (then the pinned modules can be reused).
        self.warm_ok = True
        # True while every change is inside the body of a module- or class-level function of
        # the package that no import ever runs: then a process that has the pinned modules
        # loaded behaves like the patched tree once those functions' code is replaced.
        self.hot_ok = True
        self.swaps: dict[tuple, dict] = {}       # (file, first old line) -> what to replace
        # Tests that are candidates because of something other than Python code (a data file
        # they read, ...): the maps cannot tell which part of such a test is affected.
        self.whole_only = 0
        # Modules with a changed module- or class-level statement that tests execute. Such a
        # statement may now raise, and then every test that loads the module fails, not only
        # the ones that use what the statement defines: whether the patched modules still
        # import is always looked at (main.py).
        self.import_check: set[str] = set()
        # True when the patch changes code that may run while a process pool pickles or
        # unpickles (DepMap.pickle_time), or something that could not be analysed: then a
        # test that uses a shared pool may break it for the tests after it (main.py).
        self.pool_risk = False
        # Words from the changed files / functions; tests named after them are re-run first.
        self.tokens: set[str] = set()
        self.notes: list[str] = []

    @property
    def candidates(self) -> int:
        return self.full | self.sliced | self.import_static

    def settle_imports(self, tests: int | None = None):
        """Replace the stand-in candidates of import-time changes by the given tests
        (None: keep the stand-ins)."""
        self.full |= self.import_static if tests is None else tests
        if self.import_runs and (tests is None or tests):
            # The imports do compute something else (or that was not measured): the tests
            # that only load the modules are candidates without executing the changed code.
            self.all_call_time = False
        self.import_static = 0
        self.import_modules = set()

    def add(self, tests: int):
        self.full |= tests

    def add_sliced(self, tests: int):
        self.sliced |= tests

    def note(self, msg: str):
        self.notes.append(msg)


def _is_test_config(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name in ("conftest.py", "pytest.ini", "tox.ini", ".pytest.ini", "sitecustomize.py", "usercustomize.py")


def module_name(path: str) -> str:
    mod = path[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def _read(src_root: str, rel: str) -> str | None:
    p = os.path.join(src_root, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


# -- python files ------------------------------------------------------------
CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)")
# Measure what import-time runs of a changed function body compute instead of selecting
# every test whose imports run it.
USE_IMPORT_RUNS = os.environ.get("BR_IMPORT_RUNS", "1") != "0"
FUTURE_ANNOTATIONS = re.compile(r"^from __future__ import [^\n]*\bannotations\b", re.M)


def _coding(text: str) -> str | None:
    """The source encoding a file declares in its first two lines, if any."""
    for line in text.split("\n", 2)[:2]:
        m = CODING.match(line)
        if m:
            return m.group(1)
    return None


def plan_python(dm: DepMap, plan: Plan, rel: str, old: str | None, new: str | None, cheap,
                edits: list | None = None):
    mod = module_name(rel)
    users = dm.module_users(mod)
    # Tests that read this source file as data (not by importing it) see any edit, even a comment.
    readers = dm.openers(rel)
    if not rel.startswith("sqlglot/"):
        plan.pool_risk = True       # test code: not looked at any closer
    if readers:
        plan.warm_ok = plan.hot_ok = False
        plan.all_call_time = False
        plan.pool_risk = True
        plan.add(readers)
        plan.note(f"{rel}: read as a file by {readers.bit_count()} tests")
    if old is None:
        plan.warm_ok = plan.hot_ok = False
        plan.pool_risk = True
        # A new module: nothing imports it at the pinned commit, but code that
        # enumerates its package directory may now see it.
        listers = dm.extra_set("dir_listers", os.path.dirname(rel))
        plan.add(listers)
        plan.note(f"{rel}: new python file, {listers.bit_count()} tests list its directory")
        return
    if new is None:
        plan.warm_ok = plan.hot_ok = False
        plan.all_call_time = False
        plan.pool_risk = True
        plan.add(users)
        plan.note(f"{rel}: deleted, {users.bit_count()} users")
        return
    if _coding(old) != _coding(new):
        plan.warm_ok = plan.hot_ok = False
        plan.all_call_time = False
        plan.pool_risk = True
        plan.add(users)
        plan.note(f"{rel}: source encoding declaration changed")
        return
    try:
        changes = None
        syntax_error = None
        outline = dm.outline(rel) if edits else None
        if outline is not None:
            try:
                changes = astdiff.diff_region(old, new, edits, outline)
            except Exception:  # noqa: BLE001 - the whole-file diff below is the reference
                changes = None
        if changes is None:
            changes, syntax_error = astdiff.diff_sources(old, new)
    except (SyntaxError, RecursionError, ValueError) as e:
        plan.warm_ok = plan.hot_ok = False
        plan.all_call_time = False
        plan.pool_risk = True
        plan.add(users)
        plan.note(f"{rel}: cannot diff ({e})")
        return
    if syntax_error:
        plan.warm_ok = plan.hot_ok = False
        plan.all_call_time = False
        plan.pool_risk = True
        plan.add(users)
        plan.note(f"{rel}: new version does not parse; {users.bit_count()} users")
        return
    if not changes:
        plan.note(f"{rel}: no semantic change")
        return
    if any(ch.annotation for ch in changes) and FUTURE_ANNOTATIONS.search(old) and FUTURE_ANNOTATIONS.search(new):
        # Annotations of parameters, return values and locals are never evaluated in this
        # module, and nothing in the tree inspects them at run time.
        kept = [ch for ch in changes if not ch.annotation]
        plan.note(f"{rel}: {len(changes) - len(kept)} annotation-only change(s) ignored")
        changes = kept
        if not changes:
            return
    plan.warm_ok = False
    plan.tokens.add(rel.rsplit("/", 1)[-1][:-3].lower())
    new_lines = None
    for ch in changes:
        if ch.func is not None:
            plan.tokens.add(ch.func[0].strip("_").lower())
        if ch.kind in ("call", "func"):
            lo, hi = ch.lines
            hit = dm.lines(rel, lo, hi, kinds=("run",))
            # (the whole top-level function counts: an inner function's behaviour depends
            # on the variables of the ones around it)
            region = ch.top["old"] if ch.top is not None else (
                (ch.func[1], ch.func[2]) if ch.func is not None else (lo, hi))
            if dm.pickle_time(rel, min(lo, region[0]), max(hi, region[1])):
                plan.pool_risk = True
            # The changed code may also run while modules are imported, so what those
            # imports compute may change for every test that triggers them.
            at_import = dm.lines(rel, lo, hi, kinds=("imp",))
            span = (ch.func[1], ch.func[2]) if ch.func is not None else (lo, hi)
            if (at_import and USE_IMPORT_RUNS and rel.startswith("sqlglot/")
                    and not dm.run_by_test_imports(rel, span[0], span[1])):
                # The code also runs while modules of the package are imported (some build
                # their templates by parsing SQL at import time). What those runs computed
                # is in the modules' namespaces afterwards, so the tests that merely import
                # such a module are affected only if that state differs: they stand in as
                # candidates until it has been compared (state.py). Not so for code that
                # the top level of a test module runs: what that computes stays there.
                plan.add(hit)
                plan.import_static |= at_import & ~hit
                plan.import_modules.add(mod)
                plan.import_runs = True
            else:
                if at_import:
                    plan.all_call_time = False
                plan.add(hit | at_import)
            top = ch.top
            if (top is None or not rel.startswith("sqlglot/")
                    or dm.lines(rel, top["old"][0], top["old"][1], kinds=("imp",))):
                plan.hot_ok = False
            elif (rel, top["old"][0]) not in plan.swaps:
                if new_lines is None:
                    new_lines = new.split("\n")
                plan.swaps[(rel, top["old"][0])] = dict(
                    top, rel=rel, text="\n".join(new_lines[top["new"][0] - 1:top["new"][1]]))
            plan.note(f"{rel}:{lo}-{hi} {ch.kind} run={hit.bit_count()} import={at_import.bit_count()}")
        else:
            executed = dm.lines(rel, ch.lines[0], ch.lines[1], kinds=("top",))
            if not executed:
                # Module / class level code that no test's imports ever run (e.g. under
                # `if TYPE_CHECKING:` or in a module no test loads).
                plan.note(f"{rel}:{ch.lines[0]} import-time code never executed by any test")
                continue
            cands, why = import_candidates(dm, rel, mod, users & executed, ch, cheap)
            plan.all_call_time = False
            plan.hot_ok = False
            plan.import_check.add(mod)
            if dm.pickle_time(rel):
                plan.pool_risk = True
            # What importing the patched module actually changes can be measured (see
            # state.py); until then these statically derived candidates stand in. They are
            # kept as they are when the statement's effect is not visible in a namespace.
            keep = ch.func is not None or ch.side_effects or not rel.startswith("sqlglot/")
            if keep:
                plan.add(cands)
            else:
                plan.import_static |= cands
                plan.import_modules.add(mod)
                for n in ch.names:
                    if n.isidentifier() and len(n) >= 3:
                        w = dm.word_users([n.upper()])
                        if w.bit_count() <= 150:
                            plan.import_words |= w
            plan.note(f"{rel}:{ch.lines[0]} import-time cls={ch.cls} target={ch.target} key={ch.key} "
                      f"-> {cands.bit_count()} ({why})")


# -- import-time (module / class level) changes ---------------------------------
WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")


def readers_of(dm: DepMap, name: str, depth: int = 0, seen: set | None = None, own_file: str | None = None):
    """Tests that execute code mentioning `name`.

    own_file: `name` is a global of the module in that file and the module is never
    handed around as an object, so only bare uses of the name inside that file (and
    string constants, for getattr) can read it.

    Returns (all_readers, readers_in_test_code) or None when the uses cannot be
    enumerated (name not indexed, too common, or used by import-time code we do
    not follow)."""
    idents = dm.idents()
    occ = idents.get(name)
    if occ is None:
        return None
    seen = seen if seen is not None else set()
    seen.add(name)
    everything = in_tests = 0
    for file, line, kind, cls, target, how in occ:
        outside = not file.startswith("sqlglot/")
        if own_file is not None and not outside and not (how == "s" or (how == "n" and file == own_file)):
            continue
        if kind == "c":
            hit = dm.lines(file, line, line, kinds=("run", "imp"))
            everything |= hit
            if file.startswith("tests/"):
                in_tests |= hit
        elif target == name:
            continue  # a redefinition under the same name (e.g. a subclass extending a table)
        elif outside and target and idents.get(target) is None:
            # A module of the tests keeps the value under a name of its own (from package
            # import NAME as alias). That binding is not part of the compared import state
            # and its uses are not indexed: whatever loads the module may use it.
            hit = dm.module_users(file[:-3].replace("/", ".")) if file.endswith(".py") else 0
            if not hit:
                return None
            everything |= hit
            in_tests |= hit
        elif target and depth < 3:
            if target in seen:
                continue
            sub = readers_of(dm, target, depth + 1, seen)
            if sub is None:
                return None
            everything |= sub[0]
            in_tests |= sub[1]
        else:
            return None
    return everything, in_tests


def untracked_class_readers(dm: DepMap, cls: str, readers):
    """Readers of an attribute of a class whose users the maps cannot follow by its instances
    (an enumeration, a holder of constants ...). Such an attribute is also reached without
    being named: iterating an enumeration, looking a member up by its value. So the tests
    that execute code naming the class count as readers too. readers: readers_of(the
    attribute's name); returns the same kind of pair, or None when either is unknown."""
    by_class = readers_of(dm, cls.split(".")[-1])
    if readers is None or by_class is None:
        return None
    return readers[0] | by_class[0], readers[1] | by_class[1]


def _class_keys(mod: str, cls: str) -> list[str]:
    parts = cls.split(".")
    return [f"{mod}:{'.'.join(parts[:i])}" for i in range(len(parts), 0, -1)]


def _narrow_by_key(dm: DepMap, ch, cls: str, mod: str, cands: int) -> tuple[int, str]:
    """Restrict users of a table to tests that can reach the changed entry."""
    key = ch.key or ""
    owner = f"{mod}.{cls}".lower()
    parses = "parser" in owner or "tokenizer" in owner or "tokens" in owner
    generates = "generator" in owner
    if len(key) >= 2 and key[0] in "\"'" and key[-1] == key[0]:
        text = key[1:-1]
        words = WORDS.findall(text)
        # Only keys that read like SQL keywords / function names ("REGEXP_SPLIT", "GROUPS FOR USER"):
        # those reach the table through the words of the SQL text being parsed.
        if (parses and words and text == text.upper() and " ".join(words) == " ".join(text.split())
                and all(len(w) >= 3 for w in words)):
            return cands & dm.word_users(words), f"sql contains {words}"
    else:
        last = key.rsplit(".", 1)[-1]
        if key.startswith("TokenType.") and parses:
            sid = dm.toktypes.get(last)
            return cands & (dm.set(sid) if sid else 0), f"token {last}"
        if generates and last in dm.class_by_name and key.split(".")[0] in ("exp", "expressions", last):
            users = 0
            for k in dm.class_by_name[last]:
                users |= dm.class_users(k)
            return cands & users, f"instances of {last}"
    return cands, ""


def import_candidates(dm: DepMap, rel: str, mod: str, users: int, ch, cheap) -> tuple[int, str]:
    if ch.func is not None:
        # Signature or decorators of a function: tests that execute the function.
        _, lo, hi = ch.func
        return dm.lines(rel, lo, hi, kinds=("run", "imp")), "function definition"
    readers = readers_of(dm, ch.target) if ch.target else None
    if not ch.cls:
        if readers is None:
            return users, "module level, uses unknown"
        return readers[0] & users | readers[1], "module level, readers of name"
    keys = _class_keys(mod, ch.cls)
    if keys[0] not in dm.trackable:
        readers = untracked_class_readers(dm, ch.cls, readers)
        if readers is None:
            return users, "class not trackable"
        return readers[0] & users | readers[1], "class not trackable, readers of the name or of the class"
    wide = 0
    for k in keys:
        wide |= dm.class_users(k)
    # String constants in the change may be SQL names: tests whose SQL has them can be affected
    # even without touching the class at the pinned commit (e.g. a function renamed to one they use).
    by_word = 0
    for n in ch.names:
        if n.isidentifier() and len(n) >= 3:
            w = dm.word_users([n.upper()])
            if w.bit_count() <= 150:
                by_word |= w
    cands, why = wide, "class users"
    if readers is not None:
        cands = (wide & readers[0]) | readers[1]
        why = "class users that read the name"
    if ch.key and not cheap(cands | by_word):
        narrowed, how = _narrow_by_key(dm, ch, ch.cls, mod, cands)
        if how:
            cands, why = narrowed, f"{why}; {how}"
    return cands | (by_word & users), why


def make_plan(dm: DepMap, src_root: str, files: list[FilePatch], cheap=lambda bits: False,
              python_only: bool = False, entry_users=None) -> Plan:
    """cheap(bits) says whether re-running that set of tests is inexpensive; narrowing
    heuristics are only applied when the plain candidate set is not.

    python_only: look at Python changes only (used on the per-fixture-entry map, where
    changed fixture entries are handled by the caller).
    entry_users(fixture name) -> tests whose behaviour depends on that fixture's entries."""
    plan = Plan()
    helpers_changed = any("tests/helpers.py" in (fp.old_path, fp.new_path) for fp in files)
    for fp in files:
        if fp.old_path and fp.new_path and fp.old_path != fp.new_path:
            rels = [(fp.old_path, True, False), (fp.new_path, False, True)]  # rename = delete + add
        else:
            rels = [(fp.path, fp.is_deleted, fp.is_new)]
        if any(_is_test_config(rel) for rel, _, _ in rels):
            # Changes how pytest itself collects or judges tests; re-running tests our own
            # way cannot tell what that does, so run everything.
            plan.select_all = True
            plan.note(f"{fp.path}: test runner configuration")
            continue
        if fp.is_binary:
            # Cannot rebuild the file, so cannot re-run anything against it: whoever reads it is selected.
            for rel, _, _ in rels:
                users = dm.openers(rel)
                if rel.endswith(".py"):
                    users |= dm.module_users(module_name(rel))
                plan.blind |= users
                plan.pool_risk = True
                plan.note(f"{rel}: binary change, {users.bit_count()} readers")
            continue
        edits: list = []
        try:
            base = _read(src_root, fp.old_path) if fp.old_path else None
            new_text = None if fp.is_deleted else apply_hunks(base, fp.hunks, edits)
        except (PatchError, IndexError, UnicodeDecodeError) as e:
            plan.select_all = True
            plan.note(f"{fp.path}: {e}")
            continue
        for rel, deleted, added in rels:
            old = None if (added and rel != fp.old_path) else _read(src_root, rel)
            new = None if deleted else new_text
            if old == new:
                continue
            plan.changed[rel] = new
            if rel.endswith(".py"):
                if rel.startswith(("sqlglot/", "tests/")):
                    plain = len(rels) == 1 and not deleted and not added
                    plan_python(dm, plan, rel, old, new, cheap, edits if plain else None)
                else:
                    users = dm.openers(rel) | dm.module_users(module_name(rel))
                    plan.all_call_time = False
                    plan.warm_ok = plan.hot_ok = False
                    plan.pool_risk = True
                    plan.add(users)
                    plan.whole_only |= users
                    plan.note(f"{rel}: python outside the package, {users.bit_count()} users")
            elif python_only:
                continue
            elif rel.startswith("tests/fixtures/"):
                plan.all_call_time = False
                plan_fixture(dm, plan, rel, old, new, src_root, entry_users, helpers_changed)
            else:
                if rel in ("setup.cfg", "pyproject.toml") and _pytest_section(old) != _pytest_section(new):
                    plan.select_all = True
                    plan.note(f"{rel}: pytest configuration changed")
                    continue
                readers = dm.openers(rel)
                plan.all_call_time = False
                plan.warm_ok = plan.hot_ok = False
                plan.pool_risk = True
                plan.add(readers)
                plan.whole_only |= readers
                plan.note(f"{rel}: read by {readers.bit_count()} tests")
    return plan


def _pytest_section(text: str | None) -> list[str]:
    """Lines of the pytest sections of a setup.cfg / pyproject.toml."""
    out, inside = [], False
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("["):
            inside = "pytest" in s
        if inside and s:
            out.append(s)
    return out
