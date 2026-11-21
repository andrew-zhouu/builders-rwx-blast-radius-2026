"""Which kinds of objects meet in the package's identity comparisons?

`a is b` and `a == b` give the same answer for many pairs of objects: always when one side
is None and the other side's class does not claim to equal None, when both are members of
the same enumeration, when either class compares by identity ... Whether a given `is` in
the source only ever sees such pairs is a property of the pinned program that running its
tests can show: this module records, for every single-operator `is` / `is not` comparison
in the package, the pairs of operand classes the test suite produces there, together with
how each of those classes compares for equality (see `classify`).

The selector uses the record for one purpose: a patch that turns such an `is` into `==`
(or `is not` into `!=`) cannot change what any test does when every pair of classes seen
at that comparison is one for which the two operators agree, because the patched program
then runs exactly like the pinned one up to and including each evaluation of it.

Only the classes are recorded and no comparison other than the original `is` is evaluated
here; `--check` (a development aid, not shipped data) additionally evaluates `==` on the
same operands and reports where the two operators really differ, to test the rule in
selector/ident.py against.

Usage (inside br-dev):
  python cmptrace.py --root /src/sqlglot --tests tests.json --shard 0/4 --out cmp_0.pkl [--check]
"""
from __future__ import annotations

import argparse
import ast
import builtins
import datetime
import decimal
import enum
import hashlib
import inspect
import json
import os
import pickle
import sys
import time
from importlib.machinery import PathFinder

# ---------------------------------------------------------------- rewriting side
SITES: list[tuple[str, int, int]] = []      # site id -> (relative file, line, column)
CODE: dict[str, object] = {}                # source path -> rewritten code


# Comparisons inside these methods are not recorded: `==` on the operands may run the very
# method the comparison is in, so replacing its `is` by `==` changes what `==` itself does.
COMPARING_METHODS = frozenset(("__eq__", "__ne__", "__hash__", "__lt__", "__le__", "__gt__", "__ge__", "__bool__"))


class Rewriter(ast.NodeTransformer):
    def __init__(self, rel: str):
        self.rel = rel
        self.inside = 0          # depth of enclosing comparison methods

    def visit_FunctionDef(self, node):
        special = node.name in COMPARING_METHODS
        self.inside += special
        try:
            return self.generic_visit(node)
        finally:
            self.inside -= special

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Compare(self, node):
        self.generic_visit(node)
        if self.inside or len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            return node
        sid = len(SITES)
        SITES.append((self.rel, node.lineno, node.col_offset))
        call = ast.Call(func=ast.Name(id="_br_is", ctx=ast.Load()),
                        args=[node.left, node.comparators[0], ast.Constant(sid)], keywords=[])
        if isinstance(node.ops[0], ast.IsNot):
            call = ast.UnaryOp(op=ast.Not(), operand=call)
        return ast.copy_location(call, node)


def precompile(root: str) -> int:
    n = 0
    for dirpath, _dirs, files in os.walk(os.path.join(root, "sqlglot")):
        for fn in sorted(files):
            if fn.endswith(".py"):
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), path)
                tree = Rewriter(os.path.relpath(path, root).replace(os.sep, "/")).visit(tree)
                ast.fix_missing_locations(tree)
                CODE[path] = compile(tree, path, "exec", dont_inherit=True)
                n += 1
    return n


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        code = CODE.get(self._inner.get_filename(module.__name__))
        if code is None:
            self._inner.exec_module(module)
        else:
            exec(code, module.__dict__)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    def find_spec(self, name, path=None, target=None):
        if not (name == "sqlglot" or name.startswith("sqlglot.")):
            return None
        spec = PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "get_filename"):
            spec.loader = _Loader(spec.loader)
        return spec


# ---------------------------------------------------------------- run-time side
SEEN: set = set()            # (site id, class of the left operand, class of the right operand)
DIFFER: set = set()          # --check only: (site id, left class, right class) where == disagreed
CHECK = False


def _br_is(left, right, sid):
    res = left is right
    key = (sid, type(left), type(right))
    if key not in SEEN:
        SEEN.add(key)
    if CHECK:
        try:
            other = bool(left == right)
        except BaseException:  # noqa: BLE001
            other = None
        if other is not res:
            DIFFER.add(key)
    return res


BUILTIN_FAMILIES = ((int, "num"), (float, "num"), (complex, "num"), (decimal.Decimal, "num"),
                    (str, "str"), (bytes, "bytes"), (bytearray, "bytes"), (tuple, "tuple"),
                    (list, "list"), (dict, "dict"), (set, "set"), (frozenset, "set"),
                    (datetime.datetime, "date"), (datetime.date, "date"), (datetime.time, "time"),
                    (datetime.timedelta, "timedelta"), (type(len), "bmeth"))
# __eq__ methods of the pinned commit that were read by hand (the dialect class and its
# metaclass): they answer True only for the dialect itself, for a string naming it and for
# a dialect object of the same class, so never for None or for an object that compares by
# identity, and they do nothing else on the way to that answer. The fingerprint of the
# method's syntax tree guards against applying this to different code.
NEVER_EQUAL_NONE = {
    "sqlglot.dialects.dialect._Dialect": "8dd1615398f5b384cfb89c02f561735800bc71a5",
    "sqlglot.dialects.dialect.Dialect": "4abad8f4a919c5c64e3343e449c57a6afee61812",
}
_EXACT_MEMO: dict[type, bool] = {}


def _is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _type_of(node, name):
    return (isinstance(node, ast.Call) and _is_name(node.func, "type") and len(node.args) == 1
            and not node.keywords and _is_name(node.args[0], name))


def _strict_by_exact_type(owner: type) -> bool:
    """Is owner.__eq__ written as `return self is other or (type(self) is type(other) and ...)`
    (and __ne__, if it has one, as `return not self.__eq__(other)`)? Such a class only
    ever equals objects of exactly its own class, and looks at nothing else of the others."""
    got = _EXACT_MEMO.get(owner)
    if got is not None:
        return got
    ok = False
    try:
        import textwrap

        fn = ast.parse(textwrap.dedent(inspect.getsource(owner.__dict__["__eq__"]))).body[0]
        args = [a.arg for a in fn.args.args]
        body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
        if len(args) == 2 and len(body) == 1 and isinstance(body[0], ast.Return):
            me, other = args
            v = body[0].value
            if (isinstance(v, ast.BoolOp) and isinstance(v.op, ast.Or) and len(v.values) == 2):
                first, second = v.values
                same_obj = (isinstance(first, ast.Compare) and len(first.ops) == 1 and isinstance(first.ops[0], ast.Is)
                            and _is_name(first.left, me) and _is_name(first.comparators[0], other))
                guarded = (isinstance(second, ast.BoolOp) and isinstance(second.op, ast.And) and second.values
                           and isinstance(second.values[0], ast.Compare) and len(second.values[0].ops) == 1
                           and isinstance(second.values[0].ops[0], ast.Is)
                           and _type_of(second.values[0].left, me)
                           and _type_of(second.values[0].comparators[0], other))
                ok = bool(same_obj and guarded)
        ne = owner.__dict__.get("__ne__")
        if ok and ne is not None:
            fn = ast.parse(textwrap.dedent(inspect.getsource(ne))).body[0]
            args = [a.arg for a in fn.args.args]
            body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            v = body[0].value if len(body) == 1 and isinstance(body[0], ast.Return) else None
            ok = (len(args) == 2 and isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not)
                  and isinstance(v.operand, ast.Call) and isinstance(v.operand.func, ast.Attribute)
                  and v.operand.func.attr == "__eq__" and _is_name(v.operand.func.value, args[0])
                  and len(v.operand.args) == 1 and _is_name(v.operand.args[0], args[1])
                  and not v.operand.keywords)
    except Exception:  # noqa: BLE001
        ok = False
    _EXACT_MEMO[owner] = ok
    return ok


def classify(t: type) -> str:
    """How objects of exactly this class compare for equality:
      none     the class of None
      bool     True / False
      id       by identity (object.__eq__): equal to nothing but itself
      num str bytes tuple list dict set
               by value, the way that built-in family does, and unequal to everything
               outside the family; "+uniq" marks enumerations with such a mixin, whose
               members are equal only to themselves among the members
      exact    a class of the package whose __eq__ only accepts its own exact class
      notnone  a class of the package whose hand-read __eq__ equals neither None nor
               an object that compares by identity (NEVER_EQUAL_NONE)
      unknown  anything else"""
    if t is type(None):
        return "none"
    if t is bool:
        return "bool"
    name = f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', '?')}"
    # A class made inside a function has no name of its own: several classes may carry it.
    # What its __eq__ does still holds for each of them; only "members equal themselves
    # alone" needs the two operands to be of one class, which the name cannot show.
    local = "<locals>" in name
    try:
        eq, ne = t.__eq__, t.__ne__
    except Exception:  # noqa: BLE001
        return "unknown"
    if eq is object.__eq__ and ne is object.__ne__:
        return "id"
    for base, fam in BUILTIN_FAMILIES:
        if issubclass(t, base) and eq is base.__eq__ and ne is base.__ne__:
            if issubclass(t, enum.Enum) and not issubclass(t, enum.Flag) and not local:
                return fam + "+uniq"
            return fam
    for klass in t.__mro__:
        if "__eq__" in klass.__dict__ or "__ne__" in klass.__dict__:
            mod = getattr(klass, "__module__", "") or ""
            fn = klass.__dict__.get("__eq__")
            if not (mod == "sqlglot" or mod.startswith("sqlglot.")) or not inspect.isfunction(fn):
                return "unknown"
            if _strict_by_exact_type(klass):
                return "exact"
            want = NEVER_EQUAL_NONE.get(f"{mod}.{klass.__qualname__}")
            if want and "__ne__" not in klass.__dict__ and _tree_hash(fn) == want:
                return "notnone"
            return "unknown"
    return "unknown"


def _tree_hash(fn) -> str:
    try:
        import textwrap

        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        return hashlib.sha1(ast.dump(node).encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return ""


def _tname(t: type) -> str:
    return f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', '?')}"


def snapshot() -> dict:
    types: dict[str, str] = {}
    owner: dict[str, type] = {}
    pairs: dict[int, set] = {}
    for sid, tl, tr in SEEN:
        for t in (tl, tr):
            n = _tname(t)
            if owner.setdefault(n, t) is not t:
                types[n] = "unknown"       # two classes under one name
            elif n not in types:
                types[n] = classify(t)
        pairs.setdefault(sid, set()).add((_tname(tl), _tname(tr)))
    out = {"types": types, "pairs": {sid: sorted(ps) for sid, ps in pairs.items()}}
    if CHECK:
        differ: dict[int, set] = {}
        for sid, tl, tr in DIFFER:
            differ.setdefault(sid, set()).add((_tname(tl), _tname(tr)))
        out["differ"] = {sid: sorted(ps) for sid, ps in differ.items()}
    return out


def run_test(test: dict, root: str) -> dict:
    """Runs in the forked child."""
    here = os.path.dirname(os.path.abspath(__file__))
    for extra in (os.path.normpath(os.path.join(here, "..", "..", "selector")),):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    builtins._br_is = _br_is
    sys.meta_path.insert(0, _Finder())
    import brh

    rec: dict = {"n": test["number"], "id": test["id"]}
    try:
        brh.install()
        brh.load_case(test["id"])
    except BaseException as e:  # noqa: BLE001
        rec["error"] = f"import: {type(e).__name__}: {e}"
        return rec
    failed = brh.run_tests([test["id"]])
    rec["ok"] = not failed
    rec["problems"] = list(failed.values())[:2]
    rec.update(snapshot())
    return rec


def run_together(tests: list[dict], root: str) -> dict:
    """Runs in the forked child: all the tests in one process, in the order given, so that
    whatever one test leaves behind in the package's caches is there for the next ones
    (as when the whole suite runs in one pytest process)."""
    here = os.path.dirname(os.path.abspath(__file__))
    extra = os.path.normpath(os.path.join(here, "..", "..", "selector"))
    if extra not in sys.path:
        sys.path.insert(0, extra)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    builtins._br_is = _br_is
    sys.meta_path.insert(0, _Finder())
    import brh

    brh.install()
    failed = brh.run_tests([t["id"] for t in tests])
    rec: dict = {"n": -1, "id": "<whole suite in one process>", "ok": True, "together": len(tests),
                 "failed_together": sorted(failed)[:50]}
    rec.update(snapshot())
    return rec


def _fork_collect(fn):
    path = f"/tmp/cmptrace_{os.getpid()}_{time.monotonic_ns()}.pkl"
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            with open(path, "wb") as f:
                pickle.dump(fn(), f)
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    os.waitpid(pid, 0)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        res = pickle.load(f)
    os.unlink(path)
    return res


def main():
    global CHECK
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--together", action="store_true",
                    help="one process for the whole suite, in the listed order, instead of one per test")
    a = ap.parse_args()
    CHECK = a.check
    tests = json.load(open(a.tests))["tests"]
    i, n = (int(x) for x in a.shard.split("/"))
    order = sorted(tests, key=lambda t: -t["seconds"])
    if a.only:
        want = set(a.only.split(","))
        order = [t for t in order if t["id"] in want or str(t["number"]) in want]
    listed = tests
    tests = [t for k, t in enumerate(order) if k % n == i]
    import unittest  # noqa: F401 - loaded before the package, as under pytest

    import duckdb  # noqa: F401
    import numpy  # noqa: F401
    import pandas  # noqa: F401

    t0 = time.time()
    count = precompile(a.root)
    sys.stderr.write(f"[{a.shard}] rewrote {count} files, {len(SITES)} identity comparisons, {time.time() - t0:.0f}s\n")
    out: list = [{"sites": list(SITES)}]
    if a.together:
        rec = _fork_collect(lambda: run_together(listed, a.root))
        out.append(rec if rec is not None else {"n": -1, "id": "<whole suite in one process>", "error": "no result"})
        tests = []
        if rec is not None:
            sys.stderr.write(f"[together] {rec['together']} tests, failing in this order: "
                             f"{len(rec['failed_together'])} {rec['failed_together'][:5]}\n")
    for k, test in enumerate(tests):
        rec = _fork_collect(lambda: run_test(test, a.root))
        out.append(rec if rec is not None else {"n": test["number"], "id": test["id"], "error": "no result"})
        if k % 100 == 0:
            sys.stderr.write(f"[{a.shard}] {k}/{len(tests)} {time.time() - t0:.0f}s\n")
            sys.stderr.flush()
    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    bad = [r for r in out if "id" in r and (r.get("error") or not r.get("ok"))]
    sys.stderr.write(f"[{a.shard}] done {len(out) - 1} tests, {len(bad)} not ok, {time.time() - t0:.0f}s\n")
    for r in bad[:15]:
        sys.stderr.write(f"   {r['id']}: {r.get('error') or r.get('problems')}\n")


if __name__ == "__main__":
    main()
