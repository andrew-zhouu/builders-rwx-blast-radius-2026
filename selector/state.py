"""Fingerprint of what importing sqlglot computes: every module-level name and every class
attribute, hashed by value.

Comparing the fingerprint of a patched tree with that of the pinned tree tells exactly
which definitions an import-time change (class bodies, module constants, tables) ends up
altering, including the ones derived from it by metaclasses and other import-time code.

Functions are fingerprinted by identity (module, name, defaults, closure), not by body:
a changed function body is a call-time change and is handled through coverage.
Hashes are independent of line numbers, memory addresses and hash randomisation.
"""
from __future__ import annotations

import enum
import functools
import hashlib
import importlib
import pkgutil
import re
import sys
import types

PREFIX = "sqlglot"
SKIP_CLASS_ATTRS = frozenset((
    "__dict__", "__weakref__", "__module__", "__qualname__", "__doc__", "__firstlineno__",
    "__static_attributes__", "__annotations__", "__annotate__", "__orig_bases__",
    "__parameters__", "__abstractmethods__", "_abc_impl", "__slots__", "__match_args__",
    "__dataclass_fields__", "__dataclass_params__", "__type_params__",
))
SCALARS = (bool, int, float, complex, str, bytes)


def _h(*parts) -> str:
    m = hashlib.blake2b(digest_size=6)
    for p in parts:
        m.update(p if isinstance(p, bytes) else str(p).encode("utf-8", "backslashreplace"))
        m.update(b"\0")
    return m.hexdigest()


def _ref(cls) -> str:
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', '?')}"


def _in_repo(module_name) -> bool:
    return isinstance(module_name, str) and (
        module_name == PREFIX or module_name.startswith(PREFIX + ".") or module_name.startswith("tests")
    )


class Fingerprinter:
    def __init__(self):
        self.memo: dict[int, str] = {}
        self.active: set[int] = set()

    def fp(self, v, depth: int = 0) -> str:
        t = type(v)
        if v is None or t in SCALARS:
            return _h(t.__name__, repr(v))
        key = id(v)
        got = self.memo.get(key)
        if got is not None:
            return got
        if key in self.active or depth > 80:
            return "cycle"
        self.active.add(key)
        try:
            out = self._fp(v, t, depth + 1)
        except Exception:  # noqa: BLE001 - never let one odd object break the snapshot
            out = _h("opaque", _ref(t))
        self.active.discard(key)
        self.memo[key] = out
        return out

    def code(self, c) -> str:
        consts = []
        for k in c.co_consts:
            if isinstance(k, types.CodeType):
                consts.append(self.code(k))
            elif isinstance(k, frozenset):
                consts.append(_h("fs", *sorted(repr(x) for x in k)))
            else:
                consts.append(repr(k))
        return _h("code", c.co_code, c.co_names, c.co_varnames, c.co_freevars, c.co_cellvars,
                  c.co_argcount, c.co_kwonlyargcount, c.co_posonlyargcount, c.co_flags, *consts)

    def _fp(self, v, t, d) -> str:
        fp = self.fp
        if t in (tuple, list):
            if v and all(isinstance(x, type) for x in v):
                # A tuple of classes (isinstance / except targets), often built from a set:
                # its order carries no meaning and is not reproducible.
                return _h("classes", *sorted(_ref(x) for x in v))
            return _h(t.__name__, *[fp(x, d) for x in v])
        if t in (set, frozenset):
            return _h("set", *sorted(fp(x, d) for x in v))
        if isinstance(v, dict) or t is types.MappingProxyType:
            # Order-insensitive: tables built from sets must not depend on hash randomisation.
            return _h("map", *sorted(fp(k, d) + fp(x, d) for k, x in v.items()))
        if t is types.FunctionType:
            mod = getattr(v, "__module__", None)
            if not _in_repo(mod):
                return _h("fn", mod, v.__qualname__)
            cells = []
            for cell in v.__closure__ or ():
                try:
                    cells.append(fp(cell.cell_contents, d))
                except ValueError:
                    cells.append("empty")
            body = self.code(v.__code__) if v.__name__ == "<lambda>" else ""
            return _h("fn", mod, v.__qualname__, body, fp(v.__defaults__, d), fp(v.__kwdefaults__, d), *cells)
        if isinstance(v, type):
            return _h("class", _ref(v))
        if t is types.ModuleType:
            return _h("module", v.__name__)
        if t is types.MethodType:
            return _h("method", fp(v.__func__, d), fp(v.__self__, d))
        if t in (staticmethod, classmethod):
            return _h(t.__name__, fp(v.__func__, d))
        if t is property:
            return _h("property", fp(v.fget, d), fp(v.fset, d), fp(v.fdel, d))
        if t is functools.partial:
            return _h("partial", fp(v.func, d), fp(v.args, d), fp(v.keywords, d))
        if t is functools.cached_property:
            return _h("cached_property", fp(v.func, d))
        if isinstance(v, enum.Enum):
            return _h("enum", _ref(t), v.name)
        if t is re.Pattern:
            return _h("re", v.pattern, v.flags)
        if t in (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType,
                 types.MethodWrapperType, types.ClassMethodDescriptorType, types.GetSetDescriptorType,
                 types.MemberDescriptorType):
            return _h("builtin", getattr(v, "__qualname__", ""), _ref(getattr(v, "__objclass__", t)))
        mod = getattr(t, "__module__", "")
        if _in_repo(mod):
            parts = ["obj", _ref(t)]
            state = getattr(v, "__dict__", None)
            if isinstance(state, dict):
                parts.append(fp(state, d))
            for klass in t.__mro__:
                slots = klass.__dict__.get("__slots__", ())
                for name in ((slots,) if isinstance(slots, str) else slots):
                    if name in ("__dict__", "__weakref__"):
                        continue
                    try:
                        parts.append(name + "=" + fp(getattr(v, name), d))
                    except AttributeError:
                        parts.append(name + "=unset")
            return _h(*parts)
        if mod in ("typing", "types", "collections.abc", "datetime", "decimal", "fractions", "pathlib"):
            return _h("repr", _ref(t), repr(v))
        return _h("opaque", _ref(t))


def _key(k) -> str:
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, type):
        return "c:" + _ref(k)
    if isinstance(k, enum.Enum):
        return "e:" + type(k).__name__ + "." + k.name
    if k is None or isinstance(k, (bool, int, float)):
        return "v:" + repr(k)
    return "o:" + repr(k)[:80]


def module_names(root_package=PREFIX) -> list[str]:
    """Every module of the package, importable or not."""
    pkg = importlib.import_module(root_package)
    names = [root_package]
    for m in pkgutil.walk_packages(pkg.__path__, root_package + "."):
        if not m.name.endswith(".__main__"):
            names.append(m.name)
    return sorted(names)


def import_all(names: list[str]) -> dict[str, str]:
    """Import the given modules; returns {module: error} for those that fail."""
    failed = {}
    for name in names:
        try:
            importlib.import_module(name)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            failed[name] = f"{type(e).__name__}: {e}"[:200]
    return failed


def static_imports(root: str, names: list[str]) -> dict[str, list[str]]:
    """{module: modules of the package its top-level code imports}, read from the source."""
    import ast
    import os

    known = set(names)
    out: dict[str, list[str]] = {}

    def resolve(base: str) -> str | None:
        while base:
            if base in known:
                return base
            base = base.rpartition(".")[0]
        return None

    def scan(body, deps, pkg):
        for node in body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    r = resolve(a.name)
                    if r:
                        deps.add(r)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")
                    base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
                for a in node.names:
                    r = resolve(f"{base}.{a.name}") or resolve(base)
                    if r:
                        deps.add(r)
                r = resolve(base)
                if r:
                    deps.add(r)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.ClassDef)):
                for field in ("body", "orelse", "finalbody"):
                    scan(getattr(node, field, None) or [], deps, pkg)
                for h in getattr(node, "handlers", None) or []:
                    scan(h.body, deps, pkg)

    for name in names:
        rel = name.replace(".", "/")
        path = os.path.join(root, rel + ".py")
        pkg = name.rpartition(".")[0]
        if not os.path.isfile(path):
            path = os.path.join(root, rel, "__init__.py")
            pkg = name
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError):
            continue
        deps: set[str] = set()
        scan(tree.body, deps, pkg)
        deps.discard(name)
        out[name] = sorted(deps)
    return out


def modules_used_as_objects(roots: list[str], names: list[str]) -> list[str]:
    """Modules of the package that some source file imports as a module object
    (`import a.b`, `from a import b`) — their globals can then be read as attributes
    (`b.NAME`) from anywhere. Submodules of such a package count too."""
    import ast
    import os

    known = set(names)
    used: set[str] = set()
    pairs: set[tuple[str, str]] = set()   # (x, y) for every attribute access `….x.y`
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                try:
                    with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                except (OSError, SyntaxError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            if a.asname:
                                used.add(a.name)        # `import a.b as m`: m is the module
                            else:
                                used.add(a.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                        for a in node.names:
                            used.add(f"{node.module}.{a.name}")
                    elif isinstance(node, ast.Attribute):
                        v = node.value
                        if isinstance(v, ast.Name):
                            pairs.add((v.id, node.attr))
                        elif isinstance(v, ast.Attribute):
                            pairs.add((v.attr, node.attr))
    used &= known
    # A submodule is also reachable through its package object when some code spells the
    # chain (`pkg.sub.NAME`); modules loaded by name at run time are handed around too.
    for n in known - used:
        parent, _, tail = n.rpartition(".")
        if parent and (parent.rpartition(".")[2], tail) in pairs:
            used.add(n)
    return sorted(used)


def dependents(imports: dict[str, list[str]], changed: set[str]) -> set[str]:
    """The changed modules plus every module that (transitively) imports one of them."""
    rev: dict[str, set[str]] = {}
    for mod, deps in imports.items():
        for d in deps:
            rev.setdefault(d, set()).add(mod)
    seen = set(changed)
    todo = list(changed)
    while todo:
        for m in rev.get(todo.pop(), ()):
            if m not in seen:
                seen.add(m)
                todo.append(m)
    return seen


def snapshot(only: set[str] | None = None) -> dict[str, str]:
    """{path: hash}. Paths: "module|g|name[|key]" for module globals and
    "module|c|Class.Name|attr[|key]" for class attributes; a key suffix addresses one
    entry of a dict. `only` restricts the snapshot to those modules."""
    out: dict[str, str] = {}
    f = Fingerprinter()

    def entries(base: str, val):
        if isinstance(val, dict) and len(val) <= 5000:
            for k, x in val.items():
                out[f"{base}|{_key(k)}"] = f.fp(x)

    def walk_class(cls, mname: str):
        q = cls.__qualname__
        out[f"{mname}|c|{q}|<bases>"] = _h(*[_ref(b) for b in cls.__bases__])
        for attr, val in list(vars(cls).items()):
            if attr in SKIP_CLASS_ATTRS:
                continue
            if isinstance(val, type) and val.__module__ == mname and val.__qualname__.startswith(q + "."):
                walk_class(val, mname)
            base = f"{mname}|c|{q}|{attr}"
            out[base] = f.fp(val)
            entries(base, val)

    for mname in sorted(sys.modules):
        if not (mname == PREFIX or mname.startswith(PREFIX + ".")):
            continue
        if only is not None and mname not in only:
            continue
        mod = sys.modules[mname]
        if mod is None:
            continue
        for name, val in list(vars(mod).items()):
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(val, types.ModuleType):
                # Submodules appear in a package as they get imported, by whoever imports
                # them first: not a property of this module's own code.
                continue
            if isinstance(val, type):
                if val.__module__ != mname:
                    # A class from elsewhere (an import, or a lazily resolved attribute):
                    # its content is fingerprinted where it is defined.
                    continue
                if "." not in val.__qualname__:
                    walk_class(val, mname)
            base = f"{mname}|g|{name}"
            out[base] = f.fp(val)
            entries(base, val)
    return out


def diff(baseline: dict[str, str], current: dict[str, str], unstable=(), only: set[str] | None = None) -> list[str]:
    """Paths whose hash differs; with `only`, just the paths of those modules."""
    skip = set(unstable)
    changed = []
    for path in baseline.keys() | current.keys():
        if path in skip or baseline.get(path) == current.get(path):
            continue
        if only is not None and path.split("|", 1)[0] not in only:
            continue
        changed.append(path)
    return sorted(changed)
