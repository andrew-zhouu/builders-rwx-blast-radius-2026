"""Low-overhead dependency tracer for one process, built on sys.monitoring.

For code under a repository root it records, split by phase (module import
vs. everything else):
  * lines executed inside function bodies               -> run.lines / imp.lines
  * lines executed by module and class bodies           -> ilines
  * classes instantiated through a repo-defined __init__-> run.classes / imp.classes
  * words of every SQL string tokenized or generated     -> run.words_in / run.words_out
  * token types produced by the tokenizer                -> run.toktypes
  * non-Python files opened, directories listed          -> files / dirs

Every (code, line) location fires once per phase switch and is then disabled,
so the cost is proportional to the number of distinct lines, not run time.
"""
from __future__ import annotations

import os
import re
import sys
from importlib.machinery import PathFinder

mon = sys.monitoring
TOOL = 3
CO_NEWLOCALS = 0x2
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
PREFIXES = ("sqlglot.", "tests.", "tests")


class Phase:
    __slots__ = ("lines", "classes", "words_in", "words_out", "toktypes")

    def __init__(self):
        self.lines: set[tuple[str, int]] = set()
        self.classes: set[type] = set()
        self.words_in: set[str] = set()
        self.words_out: set[str] = set()
        self.toktypes: set[str] = set()

    def snapshot(self) -> dict:
        # A class counts as used when it or any subclass is instantiated, so record the
        # whole ancestry (classes made on the fly share names; their named bases do not).
        names: set[str] = set()
        for c in self.classes:
            for base in c.__mro__:
                mod = getattr(base, "__module__", "") or ""
                if mod == "sqlglot" or mod.startswith(PREFIXES):
                    names.add(f"{mod}:{base.__qualname__}")
        return {
            "lines": sorted(self.lines),
            "classes": sorted(names),
            "words_in": sorted(self.words_in),
            "words_out": sorted(self.words_out),
            "toktypes": sorted(self.toktypes),
        }


class _Loader:
    """Wraps a module loader so the tracer knows when repo module bodies run."""

    def __init__(self, inner, tracer):
        self._inner = inner
        self._tracer = tracer

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        tr = self._tracer
        tr._enter_import()
        try:
            self._inner.exec_module(module)
        finally:
            tr._exit_import(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ReadLoader(_Loader):
    """Like _Loader, but runs the module's rewritten code (readtrace.instrument)."""

    def exec_module(self, module):
        import readtrace

        tr = self._tracer
        tr._enter_import()
        try:
            path = self._inner.get_filename(module.__name__)
            code = readtrace.CODE.get(path)
            if code is None:
                code = readtrace.CODE[path] = readtrace.instrument(self._inner.get_source(module.__name__), path)
            exec(code, module.__dict__)
        finally:
            tr._exit_import(module)


class _Finder:
    def __init__(self, tracer):
        self._tracer = tracer

    def find_spec(self, name, path=None, target=None):
        if not (name == "sqlglot" or name.startswith(PREFIXES)):
            return None
        spec = PathFinder.find_spec(name, path, target)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            if self._tracer.reads and hasattr(spec.loader, "get_source"):
                spec.loader = _ReadLoader(spec.loader, self._tracer)
            else:
                spec.loader = _Loader(spec.loader, self._tracer)
        return spec


class Tracer:
    def __init__(self, root: str, reads: bool = False):
        # reads: record table reads (readtrace.py) instead of line coverage
        self.reads = reads
        self.root = os.path.join(os.path.abspath(root), "")
        self._rel: dict[str, str | None] = {}
        self.run = Phase()
        self.imp = Phase()
        self.cur = self.run
        self.depth = 0
        self.ilines: set[tuple[str, int]] = set()
        self.files: set[str] = set()
        self.dirs: set[str] = set()
        self.active = False
        self._finder = _Finder(self)

    # -- phase handling ----------------------------------------------------
    def _enter_import(self):
        self.depth += 1
        if self.reads:
            import readtrace

            readtrace.PHASE = "imp"
            return
        if self.depth == 1:
            self.cur = self.imp
            mon.restart_events()

    def _exit_import(self, module):
        self.depth -= 1
        if self.reads:
            import readtrace

            readtrace.register_module(module)
            if self.depth == 0:
                readtrace.PHASE = "run"
            return
        if self.depth == 0:
            self.cur = self.run
            mon.restart_events()
        name = getattr(module, "__name__", "")
        try:
            if name == "sqlglot.tokens":
                self._wrap_tokenize(module)
            elif name == "sqlglot.generator":
                self._wrap_generate(module)
        except Exception as e:  # pragma: no cover - diagnostics only
            sys.stderr.write(f"brtrace: wrap failed for {name}: {e}\n")

    def _wrap_tokenize(self, module):
        tracer = self
        cls = module.Tokenizer
        orig = cls.tokenize

        def tokenize(self, sql, *a, **k):
            tokens = orig(self, sql, *a, **k)
            cur = tracer.cur
            if isinstance(sql, str):
                cur.words_in.update(w.upper() for w in set(WORD.findall(sql)))
            try:
                cur.toktypes.update({t.token_type.name for t in tokens})
            except Exception:
                pass
            return tokens

        cls.tokenize = tokenize

    def _wrap_generate(self, module):
        tracer = self
        cls = module.Generator
        orig = cls.generate

        def generate(self, *a, **k):
            sql = orig(self, *a, **k)
            if isinstance(sql, str):
                tracer.cur.words_out.update(w.upper() for w in set(WORD.findall(sql)))
            return sql

        cls.generate = generate

    # -- helpers -----------------------------------------------------------
    def _relpath(self, filename: str) -> str | None:
        rel = self._rel.get(filename, 0)
        if rel == 0:
            rel = filename[len(self.root):] if filename.startswith(self.root) else None
            self._rel[filename] = rel
        return rel  # type: ignore[return-value]

    # -- callbacks ---------------------------------------------------------
    def _on_line(self, code, line):
        rel = self._relpath(code.co_filename)
        if rel is not None:
            if code.co_flags & CO_NEWLOCALS:
                self.cur.lines.add((rel, line))
            else:
                self.ilines.add((rel, line))
        return mon.DISABLE

    def _on_start(self, code, offset):
        if code.co_name != "__init__" or self._relpath(code.co_filename) is None:
            return mon.DISABLE
        try:
            obj = sys._getframe(1).f_locals.get("self")
        except Exception:
            return None
        if obj is not None:
            self.cur.classes.add(type(obj))
        return None

    @staticmethod
    def _from_import_system() -> bool:
        """True when the audited call was made by the import machinery, not by the program."""
        f = sys._getframe(2)
        for _ in range(6):
            if f is None:
                return False
            name = f.f_code.co_filename
            if name.startswith("<frozen importlib") or name.endswith(("/linecache.py", "/tokenize.py")):
                return True
            f = f.f_back
        return False

    def _audit(self, event, args):
        if not self.active:
            return
        if event == "open":
            path = args[0]
            if isinstance(path, str) and not path.endswith(".pyc"):
                path = os.path.abspath(path)
                if path.startswith(self.root):
                    if path.endswith(".py") and self._from_import_system():
                        return
                    self.files.add(path[len(self.root):])
        elif event in ("os.listdir", "os.scandir"):
            path = args[0]
            if isinstance(path, str):
                path = os.path.abspath(path)
                if (path.startswith(self.root) or path + "/" == self.root) and not self._from_import_system():
                    self.dirs.add(path[len(self.root):])

    # -- control -----------------------------------------------------------
    def start(self):
        if self.reads:
            sys.meta_path.insert(0, self._finder)
            self.active = True
            return
        mon.use_tool_id(TOOL, "brtrace")
        mon.register_callback(TOOL, mon.events.LINE, self._on_line)
        mon.register_callback(TOOL, mon.events.PY_START, self._on_start)
        mon.set_events(TOOL, mon.events.LINE | mon.events.PY_START)
        sys.meta_path.insert(0, self._finder)
        self.active = True
        sys.addaudithook(self._audit)

    def reset(self):
        """Forget everything recorded so far (used after fork to trace one slice of a test)."""
        self.run = Phase()
        self.imp = Phase()
        self.cur = self.run if self.depth == 0 else self.imp
        self.ilines = set()
        self.files = set()
        self.dirs = set()
        mon.restart_events()

    def stop(self):
        self.active = False
        if self.reads:
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass
            return
        mon.set_events(TOOL, 0)
        mon.free_tool_id(TOOL)
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            pass

    def snapshot(self) -> dict:
        return {
            "run": self.run.snapshot(),
            "imp": self.imp.snapshot(),
            "ilines": sorted(self.ilines),
            "files": sorted(self.files),
            "dirs": sorted(self.dirs),
        }


def merge_snapshots(snaps: list[dict]) -> dict:
    """Union of several snapshots (used for interpreters a test spawned)."""
    out = {
        "run": {k: set() for k in ("lines", "classes", "words_in", "words_out", "toktypes")},
        "imp": {k: set() for k in ("lines", "classes", "words_in", "words_out", "toktypes")},
        "ilines": set(), "files": set(), "dirs": set(),
    }
    for s in snaps:
        for ph in ("run", "imp"):
            for k, v in s[ph].items():
                out[ph][k].update(tuple(x) if isinstance(x, list) else x for x in v)
        for k in ("ilines", "files", "dirs"):
            out[k].update(tuple(x) if isinstance(x, list) else x for x in s[k])
    return {
        "run": {k: sorted(v) for k, v in out["run"].items()},
        "imp": {k: sorted(v) for k, v in out["imp"].items()},
        "ilines": sorted(out["ilines"]), "files": sorted(out["files"]), "dirs": sorted(out["dirs"]),
    }


_SUB: Tracer | None = None
_DUMPED = False


def _dump_subprocess(out_dir: str):
    global _DUMPED
    if _DUMPED or _SUB is None:
        return
    _DUMPED = True
    import pickle

    try:
        snap = _SUB.snapshot()
        tmp = os.path.join(out_dir, f".{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(snap, f)
        os.replace(tmp, os.path.join(out_dir, f"{os.getpid()}.pkl"))
    except Exception:
        pass


def start_subprocess(out_dir: str, root: str):
    """Called from sitecustomize in child interpreters of a traced test."""
    global _SUB
    import atexit

    _SUB = Tracer(root)
    _SUB.start()
    atexit.register(_dump_subprocess, out_dir)
    try:
        from multiprocessing import util

        util.Finalize(None, _dump_subprocess, args=(out_dir,), exitpriority=1000)
    except Exception:
        pass
