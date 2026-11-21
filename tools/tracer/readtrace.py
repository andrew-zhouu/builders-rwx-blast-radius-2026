"""Which entries of sqlglot's tables does a test read?

Class-level and module-level dicts and sets (FUNCTIONS, TRANSFORMS, KEYWORDS, NUMERIC_ESCAPES,
token sets ...) are consulted by key far more often than they are walked. When one entry
of such a table changes, only the tests that looked that entry up (or walked the table)
can notice. This module records, for one test run:

  key   (table, key)  a lookup:  table[key], key in table, table.get(key), ...
  size  table         a truth / length test
  all   table         anything else that looks inside: iteration, copying, comparison,
                      or handing the table to code that is not instrumented

The tables are tracked by object identity, so a table reached through another name (a
parameter, an attribute it was stored in) is still the same table. Nothing in the objects
is changed: instead, the source of the package and of its tests is rewritten when it is
imported (see `instrument`), so that every expression whose value is consumed by something
other than a plain move passes through one of the `_br_*` hooks below. The rewriting is
conservative by construction: a consumer it does not know is treated as "all".

Used only while building maps (run_trace.py --reads); nothing here ships in the selector's
run-time path.
"""
from __future__ import annotations

import ast
import builtins
import functools
import types

# ---------------------------------------------------------------- run-time side
REG: dict[int, set] = {}          # id(table) -> paths ("module|g|NAME" / "module|c|Class|ATTR")
_KEEP: list = []                  # the registered tables (so ids stay theirs)
ROOT = ""                         # source root: functions defined under it are instrumented
PHASE = "run"                     # "run" | "imp" (a module of the package is being imported)
LOG = {"run": {"key": set(), "size": set(), "all": set()}, "imp": {"key": set(), "size": set(), "all": set()}}
# Reads of not (yet) registered containers made while importing: a table is only known once
# its module has finished importing, but import-time code may consult it before that.
PENDING: dict[int, list] = {}
_PENDING_KEEP: list = []
TABLE_TYPES = (dict, set, frozenset)
KEYED_METHODS = frozenset(("get", "__getitem__", "__contains__", "setdefault", "pop"))
NO_READ_CALLEES = frozenset((isinstance, type, id, callable, hasattr, getattr, issubclass))
SIZE_CALLEES = frozenset((len, bool))
_key = None                       # selector.state._key, set by install()


def _note(obj, kind, key=None):
    i = id(obj)
    paths = REG.get(i)
    if paths is not None:
        log = LOG[PHASE]
        if kind == "key":
            try:
                k = _key(key)
            except Exception:  # noqa: BLE001
                log["all"].add(i)
                return
            log["key"].add((i, k))
        else:
            log[kind].add(i)
    elif PHASE == "imp" and type(obj) in TABLE_TYPES:
        got = PENDING.get(i)
        if got is None:
            got = PENDING[i] = []
            _PENDING_KEEP.append(obj)
        if kind == "key":
            try:
                got.append(("key", _key(key)))
            except Exception:  # noqa: BLE001
                got.append(("all", None))
        else:
            got.append((kind, None))


def _tracked(obj) -> bool:
    return id(obj) in REG or (PHASE == "imp" and type(obj) in TABLE_TYPES)


def _br_gi(obj, key):
    """obj[key]"""
    if _tracked(obj):
        _note(obj, "key", key)
    if _tracked(key):
        _note(key, "all")
    return obj[key]


def _br_in(key, obj):
    """key in obj"""
    if _tracked(obj):
        _note(obj, "key", key)
    if _tracked(key):
        _note(key, "all")
    return key in obj


def _br_all(obj):
    """obj is consumed in a way that may look at all of it."""
    if _tracked(obj):
        _note(obj, "all")
    return obj


def _br_size(obj):
    """obj is tested for truth."""
    if _tracked(obj):
        _note(obj, "size")
    return obj


def _br_r(obj, name, has_args):
    """obj is the receiver of the method call obj.name(...)."""
    if _tracked(obj):
        if name in KEYED_METHODS and has_args:
            pass          # the key is logged when the first argument is evaluated (_br_a)
        elif name == "__len__":
            _note(obj, "size")
        else:
            _note(obj, "all")
    return obj


def _br_copy(obj):
    """dict(obj). A shallow copy of a known table made while a test runs (the generator's
    dispatch table is one: dict(cls.TRANSFORMS), then looked up by key) stands for that
    table from then on: what is read from the copy is recorded as read from the table,
    instead of counting the copy itself as a look at every entry. Entries the copy gets
    later are keys the table did not have, so lookups of those are lookups of missing keys.
    While modules are still being imported the copy counts as before."""
    res = dict(obj)
    if type(obj) is dict and PHASE == "run":
        paths = REG.get(id(obj))
        if paths is not None:
            REG[id(res)] = paths
            _KEEP.append(res)
            return res
    if _tracked(obj):
        _note(obj, "all")
    return res


def _instrumented(f) -> bool:
    """Is f code of ours (rewritten), so that what it does with an argument is observed?"""
    t = type(f)
    if t is types.MethodType:
        f = f.__func__
        t = type(f)
    if t is functools.partial:
        return _instrumented(f.func)
    if t is types.FunctionType:
        return f.__code__.co_filename.startswith(ROOT)
    if isinstance(f, type):
        for name in ("__init__", "__new__"):
            m = getattr(f, name, None)
            if type(m) is types.FunctionType and m.__code__.co_filename.startswith(ROOT):
                return True
        return False
    return False


def _br_a(callee, arg, pos):
    """arg is passed to callee (positional index pos, or -1 for a keyword)."""
    if _tracked(arg):
        if callee in NO_READ_CALLEES:
            pass
        elif callee in SIZE_CALLEES:
            _note(arg, "size")
        elif not _instrumented(callee):
            _note(arg, "all")
    if pos == 0 and type(callee) is types.BuiltinFunctionType:
        owner = getattr(callee, "__self__", None)
        if owner is not None and _tracked(owner) and callee.__name__ in KEYED_METHODS:
            _note(owner, "key", arg)
    return arg


def register_module(module):
    """Called when a module of the package has been imported: its dict / set valued globals
    and class attributes become known tables (paths as in selector/state.py)."""
    mname = getattr(module, "__name__", "")

    def add(obj, path):
        if type(obj) not in TABLE_TYPES:
            return
        i = id(obj)
        paths = REG.get(i)
        if paths is None:
            paths = REG[i] = set()
            _KEEP.append(obj)
            for kind, key in PENDING.pop(i, ()):
                if kind == "key":
                    LOG["imp"]["key"].add((i, key))
                else:
                    LOG["imp"][kind].add(i)
        paths.add(path)

    def walk_class(cls):
        q = cls.__qualname__
        for attr, val in list(vars(cls).items()):
            if isinstance(val, type) and val.__module__ == mname and val.__qualname__.startswith(q + "."):
                walk_class(val)
            add(val, f"{mname}|c|{q}|{attr}")

    for name, val in list(vars(module).items()):
        if name.startswith("__") and name.endswith("__"):
            continue
        if isinstance(val, type):
            if val.__module__ == mname and "." not in val.__qualname__:
                walk_class(val)
            continue
        add(val, f"{mname}|g|{name}")


def snapshot() -> dict:
    out = {}
    for phase, log in LOG.items():
        keys: dict[str, set] = {}
        for i, k in log["key"]:
            for p in REG.get(i, ()):
                keys.setdefault(p, set()).add(k)
        out[phase] = {
            "key": {p: sorted(ks) for p, ks in keys.items()},
            "size": sorted({p for i in log["size"] for p in REG.get(i, ())}),
            "all": sorted({p for i in log["all"] for p in REG.get(i, ())}),
        }
    out["tables"] = len(REG)
    return out


def reset():
    for log in LOG.values():
        for s in log.values():
            s.clear()


def install(root: str):
    """Make the hooks available to rewritten code (they are looked up as builtins)."""
    global ROOT, _key
    import os

    from selector.state import _key as state_key

    ROOT = os.path.join(os.path.abspath(root), "")
    _key = state_key
    for f in (_br_gi, _br_in, _br_all, _br_size, _br_r, _br_a, _br_copy):
        setattr(builtins, f.__name__, f)


# ---------------------------------------------------------------- rewriting side
MAY_BE_TABLE = (ast.Name, ast.Attribute, ast.Subscript, ast.Call, ast.BoolOp, ast.IfExp, ast.NamedExpr)
PLAIN_CALLS = frozenset(("locals", "globals", "vars", "super", "eval", "exec", "dir", "breakpoint", "__import__"))
SELF_NAMES = frozenset(("self", "cls"))


def _call(name, *args):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


class Rewriter(ast.NodeTransformer):
    """Wraps consumed expressions in the hooks. Children are rewritten first (generic
    visit), then the node's own consumers are wrapped."""

    def __init__(self):
        self.counter = 0
        self.temps_ok = [False]      # may a call site bind its callee to a temporary here?

    # -- helpers -----------------------------------------------------------
    def wrap(self, node, hook="_br_all"):
        if isinstance(node, MAY_BE_TABLE) and not (isinstance(node, ast.Name) and node.id in SELF_NAMES):
            return ast.copy_location(_call(hook, node), node)
        return node

    def scoped(self, node, ok):
        self.temps_ok.append(ok)
        try:
            return self.generic_visit(node)
        finally:
            self.temps_ok.pop()

    # -- scopes ------------------------------------------------------------
    def visit_FunctionDef(self, node):
        # Annotations are not code that runs; decorators and defaults belong to the
        # enclosing scope.
        outer = self.temps_ok[-1]
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        a = node.args
        a.defaults = [self.visit(d) for d in a.defaults]
        a.kw_defaults = [self.visit(d) if d is not None else None for d in a.kw_defaults]
        self.temps_ok.append(True)
        try:
            node.body = [self.visit(s) for s in node.body]
        finally:
            self.temps_ok.pop()
        self.temps_ok[-1] = outer
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        a = node.args
        a.defaults = [self.visit(d) for d in a.defaults]
        a.kw_defaults = [self.visit(d) if d is not None else None for d in a.kw_defaults]
        self.temps_ok.append(False)
        try:
            node.body = self.visit(node.body)
        finally:
            self.temps_ok.pop()
        return node

    def visit_ClassDef(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        self.temps_ok.append(False)
        try:
            node.body = [self.visit(s) for s in node.body]
        finally:
            self.temps_ok.pop()
        return node

    def visit_AnnAssign(self, node):
        node.target = self.visit(node.target)
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_arg(self, node):
        return node

    def visit_TypeAlias(self, node):
        return node

    # -- comprehensions ----------------------------------------------------
    def _comp(self, node):
        for gen in node.generators:
            # No temporaries in iterables (assignment expressions are not allowed there).
            self.temps_ok.append(False)
            try:
                gen.iter = self.wrap(self.visit(gen.iter))
            finally:
                self.temps_ok.pop()
            gen.target = self.visit(gen.target)
            gen.ifs = [self.wrap(self.visit(c), "_br_size") for c in gen.ifs]
        if isinstance(node, ast.DictComp):
            node.key = self.wrap(self.visit(node.key))
            node.value = self.visit(node.value)
        else:
            node.elt = self.visit(node.elt)
            if isinstance(node, ast.SetComp):
                node.elt = self.wrap(node.elt)
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    # -- consumers ---------------------------------------------------------
    def visit_Subscript(self, node):
        self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load):
            return node
        if isinstance(node.slice, ast.Slice) or (isinstance(node.slice, ast.Tuple) and any(
                isinstance(e, (ast.Slice, ast.Starred)) for e in node.slice.elts)):
            node.value = self.wrap(node.value)
            return node
        return ast.copy_location(_call("_br_gi", node.value, node.slice), node)

    def visit_Slice(self, node):
        self.generic_visit(node)
        for f in ("lower", "upper", "step"):
            v = getattr(node, f)
            if v is not None:
                setattr(node, f, self.wrap(v))
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            test = ast.copy_location(_call("_br_in", node.left, node.comparators[0]), node)
            if isinstance(node.ops[0], ast.NotIn):
                test = ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=test), node)
            return test
        if all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
            return node         # identity looks at nothing inside
        node.left = self.wrap(node.left)
        node.comparators = [self.wrap(c) for c in node.comparators]
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        node.values = [self.wrap(v, "_br_size") for v in node.values]
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        node.operand = self.wrap(node.operand, "_br_size" if isinstance(node.op, ast.Not) else "_br_all")
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        node.left = self.wrap(node.left)
        node.right = self.wrap(node.right)
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        node.value = self.wrap(node.value)
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        node.test = self.wrap(node.test, "_br_size")
        return node

    def _tested(self, node):
        self.generic_visit(node)
        node.test = self.wrap(node.test, "_br_size")
        return node

    visit_If = visit_While = visit_Assert = _tested

    def _iterated(self, node):
        self.generic_visit(node)
        node.iter = self.wrap(node.iter)
        return node

    visit_For = visit_AsyncFor = _iterated

    def visit_Starred(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            node.value = self.wrap(node.value)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        for i, k in enumerate(node.keys):
            if k is None:
                node.values[i] = self.wrap(node.values[i])     # {**table}
            else:
                node.keys[i] = self.wrap(k)                    # hashing a key looks inside it
        return node

    def visit_Set(self, node):
        self.generic_visit(node)
        node.elts = [e if isinstance(e, ast.Starred) else self.wrap(e) for e in node.elts]
        return node

    def visit_FormattedValue(self, node):
        self.generic_visit(node)
        node.value = self.wrap(node.value)
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Load):
            node.value = self.wrap(node.value)      # table.items etc. taken without a call
        return node

    def visit_YieldFrom(self, node):
        self.generic_visit(node)
        node.value = self.wrap(node.value)
        return node

    def visit_Await(self, node):
        self.generic_visit(node)
        node.value = self.wrap(node.value)
        return node

    def visit_withitem(self, node):
        self.generic_visit(node)
        node.context_expr = self.wrap(node.context_expr)
        return node

    def visit_Raise(self, node):
        self.generic_visit(node)
        return node

    def visit_Match(self, node):
        self.generic_visit(node)
        node.subject = self.wrap(node.subject)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if any(isinstance(t, (ast.Tuple, ast.List)) for t in node.targets):
            node.value = self.wrap(node.value)      # unpacking walks the value
        return node

    def visit_Call(self, node):
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "dict" and len(node.args) == 1 and not node.keywords
                and not isinstance(node.args[0], ast.Starred)):
            return ast.copy_location(_call("_br_copy", self.visit(node.args[0])), node)
        plain = isinstance(func, ast.Name) and func.id in PLAIN_CALLS
        use_temp = self.temps_ok[-1] and bool(node.args or node.keywords) and not plain
        # Receiver of a method call: its own hook (so that table.get(k) is a lookup).
        if isinstance(func, ast.Attribute) and not (isinstance(func.value, ast.Call) and isinstance(
                func.value.func, ast.Name) and func.value.func.id == "super"):
            method = func.attr
            recv = self.visit(func.value)
            if isinstance(recv, MAY_BE_TABLE) and not (isinstance(recv, ast.Name) and recv.id in SELF_NAMES)                     and not (method.startswith("__") and not method.endswith("__")):
                # The key of a lookup is only seen when the first argument goes through _br_a.
                keyed = use_temp and bool(node.args) and not isinstance(node.args[0], ast.Starred)
                recv = ast.copy_location(_call("_br_r", recv, ast.Constant(method), ast.Constant(keyed)), recv)
            func.value = recv
        else:
            node.func = func = self.visit(func)
        args = [self.visit(a) for a in node.args]
        keywords = [self.visit(k) for k in node.keywords]
        if plain:
            node.args, node.keywords = args, keywords
            return node
        if use_temp:
            self.counter += 1
            tmp = f"_br_c{self.counter}"
            node.func = ast.copy_location(
                ast.NamedExpr(target=ast.Name(id=tmp, ctx=ast.Store()), value=func), func)

            def passed(a, pos):
                # The first argument always goes through the hook (it may be the key of a
                # lookup made through a bound method); the others when they may be a table.
                if pos != 0 and (not isinstance(a, MAY_BE_TABLE) or (isinstance(a, ast.Name) and a.id in SELF_NAMES)):
                    return a
                return ast.copy_location(_call("_br_a", ast.Name(id=tmp, ctx=ast.Load()), a, ast.Constant(pos)), a)
        else:
            def passed(a, pos):
                return self.wrap(a)
        new_args = []
        for pos, a in enumerate(args):
            if isinstance(a, ast.Starred):
                new_args.append(a)          # *seq was wrapped by visit_Starred
            else:
                new_args.append(passed(a, pos))
        for k in keywords:
            if k.arg is None:
                k.value = self.wrap(k.value)        # **table
            else:
                k.value = passed(k.value, -1)
        node.args, node.keywords = new_args, keywords
        return node


def instrument(source: str, filename: str):
    """Code object of the rewritten module."""
    tree = ast.parse(source, filename)
    tree = Rewriter().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, filename, "exec", dont_inherit=True)


CODE: dict[str, object] = {}      # source path -> rewritten code (filled once, shared by forks)


def precompile(root: str) -> int:
    """Rewrite every module of the package and of its tests ahead of time."""
    import os

    n = 0
    for top in ("sqlglot", "tests"):
        for dirpath, _dirs, files in os.walk(os.path.join(root, top)):
            for fn in files:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    with open(path, encoding="utf-8") as f:
                        CODE[path] = instrument(f.read(), path)
                    n += 1
    return n
