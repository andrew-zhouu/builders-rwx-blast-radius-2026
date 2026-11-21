"""List the classes of the pinned sqlglot tree whose instantiation the tracer can see
(some class in their ancestry defines __init__ in the repository), and each class's
ancestry. Usage (inside the build image): python class_info.py <sqlglot root> <out.json>
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
import types


def main():
    root, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    import sqlglot

    failed = []
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        try:
            importlib.import_module(m.name)
        except Exception as e:  # noqa: BLE001
            failed.append((m.name, str(e)[:80]))
    prefix = root.rstrip("/") + "/"
    trackable, ancestry = [], {}
    seen = set()
    for name, module in sorted(sys.modules.items()):
        if not (name == "sqlglot" or name.startswith("sqlglot.")) or module is None:
            continue
        stack = [v for v in vars(module).values() if inspect.isclass(v)]
        while stack:
            cls = stack.pop()
            if cls in seen or not str(getattr(cls, "__module__", "")).startswith("sqlglot"):
                continue
            seen.add(cls)
            stack.extend(v for v in vars(cls).values() if inspect.isclass(v))
            key = f"{cls.__module__}:{cls.__qualname__}"
            ok = False
            for base in cls.__mro__:
                init = base.__dict__.get("__init__")
                if isinstance(init, types.FunctionType) and init.__code__.co_filename.startswith(prefix):
                    ok = True
                    break
            if ok:
                trackable.append(key)
            ancestry[key] = [
                f"{b.__module__}:{b.__qualname__}" for b in cls.__mro__[1:]
                if str(getattr(b, "__module__", "")).startswith("sqlglot")
            ]
    json.dump({"trackable": sorted(set(trackable)), "ancestry": ancestry, "failed": failed}, open(out, "w"))
    print(f"classes={len(ancestry)} trackable={len(set(trackable))} import failures={failed[:5]}")


if __name__ == "__main__":
    main()
