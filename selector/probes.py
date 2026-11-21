"""Per-fixture-entry map: what each entry of a fixture file makes a fixture-driven test execute.

A probe is one test restricted to one fixture entry ("<test id>##<file>#<k>") or to
no entry at all ("<test id>##base", the part of the test that does not depend on entries).
"""
from __future__ import annotations

from .depmap import DepMap


class Probes:
    def __init__(self, path: str):
        self.dm = DepMap(path)
        # test id -> {"base": probe index or None, "files": {file: {entry: probe index}}, "bits": bitset}
        self.by_test: dict[str, dict] = {}
        for i, pid in enumerate(self.dm.tests):
            tid, rest = pid.split("##", 1)
            entry = self.by_test.setdefault(tid, {"base": None, "files": {}, "bits": 0})
            entry["bits"] |= 1 << i
            if rest == "base":
                entry["base"] = i
            else:
                f, k = rest.rsplit("#", 1)
                entry["files"].setdefault(f, {})[int(k)] = i

    def entry_files(self, test_id: str) -> dict:
        info = self.by_test.get(test_id)
        return info["files"] if info else {}
