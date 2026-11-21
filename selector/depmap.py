"""Loads the dependency map built from the pinned commit and answers queries on it.

Test sets are Python ints used as bitsets over the map's own test order.
"""
from __future__ import annotations

import json
import os
import zlib
from bisect import bisect_left, bisect_right


class DepMap:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = json.loads(zlib.decompress(f.read()))
        self.tests: list[str] = data["tests"]
        self.index = {t: i for i, t in enumerate(self.tests)}
        self.n = len(self.tests)
        self.all = (1 << self.n) - 1
        self._sets_hex: list[str] = data["sets"]
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
            v = self._sets[sid] = int(self._sets_hex[sid], 16)
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
                with open(self._idents_path, "rb") as f:
                    raw = json.loads(zlib.decompress(f.read()))
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

    def outline(self, rel: str) -> dict | None:
        """Statement outline of a pinned Python file (see tools/build_outline.py)."""
        if self._outline is None:
            try:
                with open(os.path.join(os.path.dirname(self._idents_path), "outline.json.z"), "rb") as f:
                    self._outline = json.loads(zlib.decompress(f.read()))
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
