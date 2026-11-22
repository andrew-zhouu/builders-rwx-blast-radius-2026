"""Loads the dependency map built from the pinned commit and answers queries on it.

Test sets are Python ints used as bitsets over the map's own test order.
"""
from __future__ import annotations

import json
import os
import pickle
import zlib
from bisect import bisect_left, bisect_right


def load(path: str):
    """A map file: the unpacked copy (<name>.pkl, written by tools/pack_maps.py when the
    image is built) when there is one, else the compressed JSON it was made from."""
    if path.endswith(".json.z"):
        fast = path[: -len(".json.z")] + ".pkl"
        if os.path.exists(fast):
            try:
                with open(fast, "rb") as f:
                    return pickle.load(f)
            except Exception:  # noqa: BLE001 - fall back to the JSON
                pass
    with open(path, "rb") as f:
        return json.loads(zlib.decompress(f.read()))


class DepMap:
    def __init__(self, path: str):
        data = load(path)
        self.tests: list[str] = data["tests"]
        self.index = {t: i for i, t in enumerate(self.tests)}
        self.n = len(self.tests)
        self.all = (1 << self.n) - 1
        self._sets_hex: list = data["sets"]      # hexadecimal strings, or integers when unpacked
        self._sets: dict[int, int] = {}
        self.files: dict[str, dict] = data["files"]
        self.modules: dict[str, int] = data["modules"]
        self.opened: dict[str, int] = data["opened"]
        self.classes: dict[str, int] = data["classes"]
        self.run_classes: dict[str, int] = data.get("run_classes") or data["classes"]
        self.words_in: dict[str, int] = data["words_in"]
        self.words_out: dict[str, int] = data["words_out"]
        self.toktypes: dict[str, int] = data["toktypes"]
        self.fixtures: dict[str, dict[str, int]] = data["fixtures"]
        self.cpu: list[float] = data.get("cpu") or data["trace_cpu"]
        self.cpu_iso: list[float] = data.get("cpu_iso") or self.cpu
        self.cold: list[float] = data.get("cold") or [0.0] * len(self.tests)
        self.import_cost: list[float] = data.get("import_cost") or [0.15] * len(self.tests)
        self.class_setup: dict[str, float] = data.get("class_setup", {})
        self.extra: dict = data.get("extra", {})
        self.trackable: set[str] = set(self.extra.get("trackable", ()))
        self._idents_path = os.path.join(os.path.dirname(path), "idents.json.z")
        self._idents: dict | None = None
        self._outline: dict | None = None
        self._reads_path = os.path.join(os.path.dirname(path), "reads.json.z") if self.extra.get("reads") else None
        self._reads: dict | None = None
        self._cmp_path: str | None = os.path.join(os.path.dirname(path), "cmp.json.z")
        self._cmp: dict | None = None
        self._timp_path: str | None = os.path.join(os.path.dirname(path), "timp.json.z")
        self._timp: dict | None = None
        self._pools_path = os.path.join(os.path.dirname(path), "pools.json.z")
        self._pools: list | None = None
        self._pickle: dict | None = None
        self._phases: dict | None = None
        self._duck: dict | None = None
        # simple class name -> full keys ("module:Qual.Name"), for classes some test instantiates
        self.class_by_name: dict[str, list[str]] = {}
        for key in self.classes:
            self.class_by_name.setdefault(key.rsplit(":", 1)[1].rsplit(".", 1)[-1], []).append(key)
        # tests grouped by test file / class, as bitsets
        self.by_file: dict[str, int] = {}
        self.by_class: dict[str, int] = {}
        for i, t in enumerate(self.tests):
            f, c, _ = t.split("::")
            self.by_file[f] = self.by_file.get(f, 0) | (1 << i)
            key = f"{f}::{c}"
            self.by_class[key] = self.by_class.get(key, 0) | (1 << i)

    def set(self, sid: int) -> int:
        v = self._sets.get(sid)
        if v is None:
            raw = self._sets_hex[sid]
            v = self._sets[sid] = raw if isinstance(raw, int) else int(raw, 16)
        return v

    def lines(self, path: str, lo: int, hi: int, kinds=("run", "imp")) -> int:
        """Tests that executed any line of path in [lo, hi] (old file coordinates)."""
        f = self.files.get(path)
        if f is None:
            return 0
        ls = f["lines"]
        a, b = bisect_left(ls, lo), bisect_right(ls, hi)
        out = 0
        for kind in kinds:
            col = f[kind]
            for sid in set(col[a:b]):
                if sid:
                    out |= self.set(sid)
        return out

    def module_users(self, module: str) -> int:
        sid = self.modules.get(module)
        return self.set(sid) if sid else 0

    def openers(self, path: str) -> int:
        sid = self.opened.get(path)
        return self.set(sid) if sid else 0

    def idents(self) -> dict:
        """name -> [[file, line, 'c'|'i', class, target], ...] or None when too common."""
        if self._idents is None:
            try:
                raw = load(self._idents_path)
                files = raw["files"]
                # (file, line, 'c' call-time | 'i' import-time, class, target,
                #  'n' bare name | 'a' attribute | 's' string constant)
                self._idents = {
                    name: None if occ is None else [
                        (files[o[0]], o[1], o[2], o[3], o[4], o[5] if len(o) > 5 else "a") for o in occ
                    ]
                    for name, occ in raw["names"].items()
                }
            except (OSError, ValueError, KeyError):
                self._idents = {}
        return self._idents

    def reads(self) -> dict | None:
        """Which entries of the package's dict / set valued tables each test reads
        (tools/tracer/readtrace.py), or None when this map has no such record:
          "tables" set of table paths, "ok" tests that were observed,
          "all" / "size" {table: set id}, "keys" {table: {key: set id}}."""
        if self._reads is None and self._reads_path is not None:
            try:
                data = load(self._reads_path)
                data["tables"] = set(data["tables"])
                data["ok"] = self.set(data["ok"])
                self._reads = data
            except (OSError, ValueError, KeyError):
                self._reads_path = None
        return self._reads

    def cmp(self) -> dict | None:
        """Classes of the operands of the package's identity comparisons (tools/tracer/
        cmptrace.py), or None when the map has no such record. "bad_bits": the tests
        that could not be profiled, as a set of this map's tests."""
        if self._cmp is None and self._cmp_path is not None:
            try:
                data = load(self._cmp_path)
                bits = 0
                for tid in data.get("bad", ()):
                    i = self.index.get(tid)
                    if i is not None:
                        bits |= 1 << i
                data["bad_bits"] = bits
                self._cmp = data
            except (OSError, ValueError, KeyError, zlib.error):
                self._cmp_path = None
        return self._cmp

    def run_by_test_imports(self, rel: str, lo: int, hi: int) -> bool:
        """Is code in rel:lo-hi run by the module-level code of a test module while it is
        imported (tools/tracer/timp.py)? True also when that is not known."""
        if self._timp is None and self._timp_path is not None:
            try:
                self._timp = load(self._timp_path)["files"]
            except (OSError, ValueError, KeyError, zlib.error):
                self._timp_path = None
        if self._timp is None:
            return True
        return any(a <= hi and lo <= b for a, b in self._timp.get(rel, ()))

    def phases(self) -> dict:
        """Per test that can be settled in two phases (tools/tracer/phases.py):
        {"file": fixture file, "fp": {entry: [parsed, optimized] or None}, "opt_cpu": {entry: s},
         "index": {(file, line): bit}, "other": {entry: bits of the lines the entry runs
         outside parsing and optimizing}}."""
        if self._phases is None:
            self._phases = {}
            try:
                data = load(os.path.join(os.path.dirname(self._pools_path), "phases.json.z"))
            except (OSError, ValueError, KeyError, zlib.error):
                data = {}
            for tid, rec in data.get("tests", {}).items():
                if tid not in self.index:
                    continue
                by_file: dict[str, list] = {}
                for bit, (rel, line) in enumerate(rec["lines"]):
                    by_file.setdefault(rel, []).append((line, bit))
                self._phases[tid] = {"file": rec["file"], "fp": rec["fp"], "opt_cpu": rec["opt_cpu"],
                                     "by_file": by_file,
                                     "other": {k: int(v, 16) for k, v in rec["other"].items()}}
        return self._phases

    def duck_costs(self) -> dict | None:
        """{query index: CPU seconds of one query of the executor's TPC-DS test once DuckDB's
        answer is at hand} (tools/build_duck.py), or None without the record."""
        if self._duck is None:
            import pickle

            self._duck = {}
            try:
                with open(os.path.join(os.path.dirname(self._pools_path), "duck.bin"), "rb") as f:
                    data = pickle.load(f)
                if data.get("answers"):
                    self._duck = {int(k): float(v) for k, v in (data.get("exec_cpu") or {}).items()}
            except Exception:  # noqa: BLE001 - no record
                self._duck = {}
        return self._duck or None

    def shared_pools(self) -> list[int]:
        """Sets of tests that submit calls to one process pool made in a class fixture
        (tools/tracer/pools.py): one set per test class that has such a pool."""
        if self._pools is None:
            self._pools = []
            try:
                data = load(self._pools_path)
            except (OSError, ValueError, KeyError, zlib.error):
                data = {}
            self._pickle = data.get("pickle")
            for ids in data.get("classes", {}).values():
                bits = 0
                for tid in ids:
                    i = self.index.get(tid)
                    if i is not None:
                        bits |= 1 << i
                if bits:
                    self._pools.append(bits)
        return self._pools

    # Files all of whose code counts as running while a pool pickles: the serialiser, the
    # exception classes (an exception raised in a worker is pickled back) and the modules of
    # the expression classes (unpickling builds the nodes through their constructors).
    PICKLE_FILES = ("sqlglot/serde.py", "sqlglot/errors.py", "sqlglot/expressions.py")
    PICKLE_DIRS = ("sqlglot/expressions/",)

    def pickle_time(self, rel: str, lo: int | None = None, hi: int | None = None) -> bool:
        """Could a change to these lines of the file (None: to its module-level code) change
        what happens while a process pool pickles or unpickles calls, results or exceptions?
        True for the files above, for the functions seen running at such a moment on the
        pinned commit (tools/tracer/pools.py) and for the module-level code of their files;
        also when that was not measured."""
        self.shared_pools()
        if self._pickle is None or rel in self.PICKLE_FILES or rel.startswith(self.PICKLE_DIRS):
            return True
        spans = self._pickle.get(rel)
        if not spans:
            return False
        if lo is None:
            return True
        return any(a <= hi and lo <= b for a, b in spans)

    def outline(self, rel: str) -> dict | None:
        """Statement outline of a pinned Python file (see tools/build_outline.py)."""
        if self._outline is None:
            try:
                self._outline = load(os.path.join(os.path.dirname(self._idents_path), "outline.json.z"))
            except (OSError, ValueError):
                self._outline = {}
        return self._outline.get(rel)

    def word_users(self, words: list[str]) -> int:
        """Tests whose tokenized or generated SQL contains every given word."""
        out = self.all
        for w in words:
            a = self.words_in.get(w)
            b = self.words_out.get(w)
            out &= (self.set(a) if a else 0) | (self.set(b) if b else 0)
            if not out:
                break
        return out

    def extra_set(self, table: str, key: str) -> int:
        sid = self.extra.get(table, {}).get(key)
        return self.set(sid) if sid else 0

    def class_users(self, name: str, run_only: bool = False) -> int:
        """Tests that instantiate the class or a subclass (run_only: not counting imports)."""
        sid = (self.run_classes if run_only else self.classes).get(name)
        return self.set(sid) if sid else 0

    def ids(self, bits: int) -> list[int]:
        out = []
        i = 0
        while bits:
            low = bits & -bits
            i = low.bit_length() - 1
            out.append(i)
            bits ^= low
        return out


def popcount(bits: int) -> int:
    return bits.bit_count()
