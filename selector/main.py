"""Blast radius selector: for every patch, print the tests to run and how to split them.

For each commit:
  1. apply the patch in memory and diff it structurally against the pinned tree;
  2. look the changed code / files up in maps built from the pinned commit to get
     the tests that could be affected. For module- and class-level changes, import the
     patched tree in a child, fingerprint what the import computes and compare it with
     the pinned tree's fingerprint: the definitions that really changed give the tests;
  3. re-run the affordable candidates against the patched tree and keep those that fail.
     Fixture-driven tests are re-run on the affected fixture entries only, long test
     methods on the affected units only. A candidate that was not settled that way is
     selected (but see MIN_EVIDENCE for the three measured exceptions).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

# Numeric libraries size their thread pools from the host's core count; the run has two
# cores and a cap on processes/threads, so keep them single-threaded (set before they load).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from . import verify  # noqa: E402
from .depmap import DepMap  # noqa: E402
from .patch import parse_patch
from .plan import make_plan
from .probes import Probes
from .split import best_split, split_cost

DATA = Path(os.environ.get("BR_DATA", "/data"))
SRC = os.environ.get("BR_SRC", "/app/src")
MAP = os.environ.get("BR_MAP", "/app/maps/map.json.z")
PMAP = os.environ.get("BR_PMAP", os.path.join(os.path.dirname(MAP), "pmap.json.z"))
STATE = os.environ.get("BR_STATE", os.path.join(os.path.dirname(MAP), "state.json.z"))
TMP = os.environ.get("BR_TMP", "/tmp")
HARNESS = os.environ.get("BR_HARNESS", str(Path(__file__).resolve().parent))
MODE = os.environ.get("BR_MODE", "verify")          # verify | static
REPORT = os.environ.get("BR_REPORT")                # optional per-commit JSON lines for analysis


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# Most CPU spent re-running the candidates of one commit. On labelled trial edits re-runs
# estimated at 12-15 s paid for themselves and those above 15 s about broke even; 12 s
# and 21 s were both worse in total.
VERIFY_CAP_S = _f("BR_VERIFY_CAP", 18.0)
# Re-run only when that is expected to save at least this much: about half of the changes
# break something, and one selected test already costs a worker (2 s) ...
GAIN_MIN_S = _f("BR_GAIN_MIN", 1.0)
LEAN = int(_f("BR_LEAN", 0))                 # also consider not re-running any part of the tests left out on evidence
GAIN_RATIO = _f("BR_GAIN_RATIO", 0.05)       # ... and this share of what selecting every candidate costs
HARD_FAIL_SHARE = _f("BR_HARD_FAIL_SHARE", 0.3)  # how often tests that may be left out on evidence end up selected
EXPENSIVE_S = _f("BR_EXPENSIVE", 1.0)        # tests costing more CPU than this are not re-run whole ...
WHOLE_RATIO = _f("BR_WHOLE_RATIO", 0.8)      # ... unless that is under this share of selecting them blindly
SLICE_MIN_S = _f("BR_SLICE_MIN", 0.25)       # fixture-driven tests cheaper than this are just re-run whole
UNIT_MIN_S = _f("BR_UNIT_MIN", 0.02)         # tests cheaper than this are not re-run a unit at a time
UNIT_RATIO = _f("BR_UNIT_RATIO", 0.7)        # ... nor when the affected units are more than this share of the test
SLICE_BUDGET_S = _f("BR_SLICE_BUDGET", 6.0)  # most CPU spent re-running affected entries of one test
SLICE_RATIO = _f("BR_SLICE_RATIO", 0.8)      # ... and at most this share of what selecting the test costs
SAMPLE_BUDGET_S = _f("BR_SAMPLE_BUDGET", 0.5)  # ... and when that is exceeded, CPU spent on a sample of them
BIG_SLICE_S = _f("BR_BIG_SLICE", 12.0)       # CPU cap for settling an expensive test that would otherwise be selected
BIG_SLICE_RATIO = _f("BR_BIG_SLICE_RATIO", 0.5)  # ... and at most this share of what selecting it costs
# A test that was not settled is selected. The exception: tests the maps list as never
# failing on their own (tools/never_alone.json; measured, see README) are left out when the
# change is call-time only and every other candidate was settled and passed ...
MIN_EVIDENCE = int(_f("BR_MIN_EVIDENCE", 8))  # ... at least this many of them
SKIP_HARD = int(_f("BR_SKIP_HARD", 1))        # 0: never leave a test out that way
EARLY_STOP = int(_f("BR_EARLY_STOP", 5))     # stop re-running once the first k tests have all failed
FIRST_FAIL_MARGIN_MS = _f("BR_FIRST_FAIL_MARGIN_MS", 2500)
WARM = int(_f("BR_WARM", 1))                 # reuse pinned modules for patches that change no Python code
HOT = int(_f("BR_HOT", 1))                   # replace function code in place for pure function-body changes
# TestGenerator::test_generator_memory does the same thing for every dialect; it is re-run on
# the dialects a patch can reach only (Selector.loop_slice, brh.LOOP_RECIPES).
PIN = int(_f("BR_PIN", 0))                   # keep the process and its children on one CPU
USE_LOOPS = int(_f("BR_LOOPS", 1))
LOOP_TEST = "tests/test_generator.py::TestGenerator::test_generator_memory"
# The executor's TPC-DS test, when it would be selected without having been seen to fail, is
# settled in two phases instead (brh.PHASE_RECIPES; Selector.phase_spec): a look at what the
# parser and the optimizer make of each affected query, and a run of the queries where that
# differs from the pinned commit.
USE_PHASES = int(_f("BR_PHASES", 1))
PHASE_PROBE_S = _f("BR_PHASE_PROBE", 9.0)    # ... when that is expected to cost at most this much CPU
PHASE_EXEC_S = _f("BR_PHASE_EXEC", 0.0)      # ... plus this much for running queries that differ (0: only looks)
PHASE_SLICES = int(_f("BR_PHASE_SLICES", 1))  # the same looks when the test is re-run on its affected queries
PHASE_TEST = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
PHASE_AFTER = ("sqlglot/executor/", "sqlglot/planner.py",
               "sqlglot/serde.py")            # (the digests of the expressions are made with it)
PHASE_DATA = ("tests/fixtures/optimizer/tpc-ds/", "tests/fixtures/optimizer/tpc-h/")   # what the test reads
PHASE_MORE = int(_f("BR_PHASE_MORE", 1))      # also beside edits of fixture files the test does not read
# Leaving an expensive test out on the evidence of its neighbours (off by default; see
# Selector.evidence_probe).
RISK = int(_f("BR_RISK", 0))
RISK_EXEC_S = _f("BR_RISK_EXEC_S", 1.0)      # evidence: the executor tests cheaper than this,
RISK_OPT_S = _f("BR_RISK_OPT_S", 0.0)        # ... and the optimizer tests cheaper than this (0: none)
RISK_ANY = int(_f("BR_RISK_ANY", 0))         # (for experiments: also for patches outside the package)
RISK_IMPORT = int(_f("BR_RISK_IMPORT", 1))   # 0: no bet at all on a change to import-time code
RISK_IMPORT_T = int(_f("BR_RISK_IMPORT_T", 0))  # (for experiments: the neighbour bet on import-time changes too)
RISK_SAMPLE = int(_f("BR_RISK_SAMPLE", 30))  # this many of the cheapest other selected tests are evidence too
RISK_VU = int(_f("BR_RISK_VU", 0))           # (for experiments: the neighbour bet after a re-run too)
RISK_PROBE_S = _f("BR_RISK_PROBE", 12.0)     # getting the evidence may cost at most this much CPU
RISK_CLASS = int(_f("BR_RISK_CLASS", 0))     # evidence: every candidate of the test's own class (see evidence_probe)
RISK_MANY = int(_f("BR_RISK_MANY", 8))       # ... and with this many tests known to fail nothing is left out
RISK_PARSER = "tests/test_parser.py::TestParser::"
RISK_MIN_EVIDENCE = int(_f("BR_RISK_MIN_EVIDENCE", 3))   # ... and it takes at least this many passing neighbours
RISK_TESTS = ("tests/test_executor.py::TestExecutor::test_execute_tpcds",
              "tests/test_parser.py::TestParser::test_max_nodes")
RISK_EXEC = "tests/test_executor.py::TestExecutor::"
# The expensive parser test builds one statement of a million nodes and expects the parser to
# refuse it at limits of 1,000,000 and of 50,000 nodes. The stand-in is the same test method
# with a statement a hundredth as wide (some 12,000 nodes) and limits of 4,000 and 5,000.
RISK_SCALE = int(_f("BR_RISK_SCALE", 1))
RISK_SCALE_S = _f("BR_RISK_SCALE_S", 0.8)     # the stand-in takes 0.3 s of CPU on the pinned commit ...
RISK_SCALE_X = _f("BR_RISK_SCALE_X", 2.5)     # ... and may take this many times what it takes there on this machine
RISK_SCALED = {
    "tests/test_parser.py::TestParser::test_max_nodes": [
        ["        width = 10000" + chr(10), "        width = 100" + chr(10)],
        ["            parse_one(sql, max_nodes=1_000_000)" + chr(10), "            parse_one(sql, max_nodes=4_000)" + chr(10)],
        ["            parse_one(sql, max_nodes=50_000)" + chr(10), "            parse_one(sql, max_nodes=5_000)" + chr(10)],
        ['self.assertIn("(50000)", str(ctx.exception))', 'self.assertIn("(5000)", str(ctx.exception))'],
    ],
}
RISK_OPT = "tests/test_optimizer.py::TestOptimizer::"
USE_STATE = int(_f("BR_USE_STATE", 1))       # measure what import-time changes alter (state.py)
STATE_MIN_S = _f("BR_STATE_MIN", 1.2)        # ... when re-running the static candidates costs more than this
STATE_IMPORT_LIMIT_S = int(_f("BR_STATE_IMPORT_LIMIT", 30))   # ... and the most that importing and comparing may take
USE_READS = int(_f("BR_READS", 1))           # narrow changes to tables by the recorded table reads
CHEAP_SET_S = _f("BR_CHEAP_SET", 1.5)
HEAVY_RE = re.compile(r"^\s*(?:import|from)\s+(?:duckdb|pandas|numpy)\b", re.M)


def log(msg: str):
    sys.stderr.write(msg + "\n")


class Job:
    """How one commit's candidates are going to be re-run."""

    def __init__(self):
        self.final: list[list[int]] | None = None   # set when no re-run is needed
        self.blind_ms = 0
        self.blind: dict[int, int] = {}
        self.unknown_outcome: dict[int, int] = {}   # what to select when the re-run dies or hangs
        self.loops: dict = {}                       # test id -> members its loop is cut down to (brh.LOOP_SLICE)
        self.phased: dict = {}                      # test id -> spec for brh.PHASED
        self.whole = 0
        self.slices: dict = {}
        self.lights: dict = {}
        self.light_bits = 0
        self.hard = 0
        self.hard_dominates = False
        self.est = 0.0
        self.est_max = 0.0
        self.groups: list = []
        self.drops: dict = {}
        self.keep: dict = {}
        self.skip_setup: list = []
        self.stop = 0
        self.group_ids: list[str] = []


class Selector:
    def __init__(self):
        self.t0 = time.monotonic()
        tests = json.loads((DATA / "tests.json").read_text())["tests"]
        self.ms = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
        num_by_id = {t["id"]: t["number"] for t in tests}
        self.all_numbers = sorted(self.ms)
        self.dm = DepMap(MAP)
        self.num = [num_by_id.get(tid) for tid in self.dm.tests]   # map index -> test number
        known = {n for n in self.num if n is not None}
        self.unknown = [n for n in self.all_numbers if n not in known]
        self.commits = sorted((DATA / "commits").glob("*.patch"))
        n = max(1, len(self.commits))
        self.wall_budget = float(os.environ.get("BR_WALL_BUDGET", min(150.0, 3.6 * n)))
        self._heavy_loaded = False
        self._heavy_cache: dict[str, bool] = {}
        self.report = open(REPORT, "w") if REPORT else None
        self._all_split = None
        self._cls = [t.rsplit("::", 1)[0] for t in self.dm.tests]
        self._probes: Probes | None = None
        self._probes_failed = False
        self._warm: bool | None = None
        self._entry_users: dict[str, int] | None = None
        self._state: dict | None = None
        self._state_failed = False
        self._facts: dict | None = None
        self._base: dict = {}              # pinned import state taken in this process (prepare)
        self._base_scope: set = set()      # ... the modules it covers
        self._plans: dict = {}             # commit path -> (files, plan) or the exception planning raised
        self._meth: list | None = None     # method name of each test of the map
        self._plan = None                  # the plan of the commit being handled (evidence_probe)
        self._loop_uses: dict | None = None    # verify.dialect_uses, once
        self._scaled_base: dict = {}           # stand-in -> CPU seconds on the pinned tree, measured in this run
        self._loop_failed = False
        self._phase_costs: list | None = None  # the queries of PHASE_TEST that it reaches
        self._class_bits: dict | None = None
        # tests known not to use their class-level fixtures (map index)
        self.no_setup = {self.dm.index[t] for t in self.dm.extra.get("no_class_setup", ())
                         if t in self.dm.index}
        # tests that read fixture files through the loaders: test index -> [file]
        self.fixture_files: dict[int, list[str]] = defaultdict(list)
        for key, users in self.dm.fixtures.items():
            name = key.split(":", 1)[1]
            for i in users:
                self.fixture_files[int(i)].append(name)
        # tests measured never to fail on their own (tools/never_alone.json)
        self.never_alone = 0
        for tid in self.dm.extra.get("never_alone", {}):
            if tid in self.dm.index:
                self.never_alone |= 1 << self.dm.index[tid]
        self.t_startup = time.process_time()
        self.t_plan = 0.0
        self.t_imports = 0.0

    # -- lazily loaded maps ------------------------------------------------
    @property
    def probes(self) -> Probes | None:
        if self._probes is None and not self._probes_failed:
            try:
                self._probes = Probes(PMAP)
            except (OSError, ValueError, KeyError) as e:
                log(f"probe map unavailable: {e}")
                self._probes_failed = True
        return self._probes

    @property
    def facts(self) -> dict | None:
        """The pinned tree's modules and import graph (tools/build_state.py)."""
        if self._facts is None and not self._state_failed:
            try:
                with open(STATE, "rb") as f:
                    self._facts = json.loads(zlib.decompress(f.read()))
            except (OSError, ValueError, KeyError) as e:
                log(f"state facts unavailable: {e}")
                self._state_failed = True
        return self._facts

    def wants_state(self, plan) -> bool:
        """Module / class level changes whose statically derived candidates are expensive:
        worth measuring what the imports really compute differently."""
        if not (USE_STATE and MODE == "verify" and plan.import_static
                and self.estimate(plan.candidates & ~plan.blind) > STATE_MIN_S):
            return False
        facts = self.facts
        if facts is None or not plan.import_modules <= set(facts["modules"]):
            return False    # a module the comparison does not import: nothing would be measured
        if plan.import_runs:
            ordinary = (plan.full | plan.sliced) & ~plan.blind & ~self.never_alone
            if self.estimate(ordinary) > 1.5 * VERIFY_CAP_S:
                return False    # too many tests execute the changed code to re-run them: no point
        return True

    def prepare(self):
        """Plan every commit up front. For the import-time changes among them, the pinned
        tree's import state (the fingerprint patched imports are compared with) is taken
        here, once, from this process's own freshly imported modules, before anything else
        has run in it, and only for the modules those commits can affect."""
        from . import state as st

        need: set[str] = set()
        for path in self.commits:
            p0 = time.process_time()
            try:
                files = parse_patch(path.read_text(encoding="utf-8", errors="replace"))
                plan = make_plan(self.dm, SRC, files, self.cheap, entry_users=self.entry_users)
                self._plans[path] = (files, plan)
                if self.wants_state(plan) and self.facts is not None:
                    need |= st.dependents(self.facts["imports"], set(plan.import_modules))
            except Exception as e:  # noqa: BLE001 - reported when the commit is handled
                self._plans[path] = e
            self.t_plan += time.process_time() - p0
        if need and HOT:
            p0 = time.process_time()
            self.warm_up()
            if verify.can_swap():      # every module of the package is imported here
                import unittest  # noqa: F401

                self._base = st.snapshot(only=need)
                self._base_scope = need
                log(f"state baseline: {len(self._base)} definitions of {len(need)} modules, "
                    f"{time.process_time() - p0:.2f}s with imports")
            self.t_imports += time.process_time() - p0

    def state_for(self, scope: set) -> dict | None:
        """Facts plus, under "state", the pinned fingerprint of (at least) the given modules."""
        facts = self.facts
        if facts is None:
            return None
        if scope <= self._base_scope:
            return dict(facts, state=self._base)
        if self._state is None and not self._state_failed:
            # Not prepared (or this process could not import everything): take the whole
            # fingerprint in a child forked like the ones that import patched trees.
            try:
                import unittest  # noqa: F401 - loaded in every child, before sqlglot is

                from . import state as st

                def in_child():
                    for name in [m for m in sys.modules
                                 if m in ("sqlglot", "tests") or m.startswith(("sqlglot.", "tests."))]:
                        del sys.modules[name]
                    sys.path[:] = [SRC] + [p for p in sys.path if p != SRC]
                    failed = st.import_all(facts["modules"])
                    return {"failed": failed, "state": st.snapshot()}

                c0 = os.times()
                res = verify.fork_call(in_child, 60.0)
                c1 = os.times()
                if not res or res["failed"]:
                    raise ValueError(f"pinned tree did not import cleanly: {res and res['failed']}")
                self._state = dict(facts, state=res["state"])
                log(f"state baseline: {len(res['state'])} definitions, "
                    f"{c1.children_user - c0.children_user + c1.children_system - c0.children_system:.2f}s")
            except (OSError, ValueError, KeyError) as e:
                log(f"state fingerprint unavailable: {e}")
                self._state_failed = True
        return self._state

    def entry_users(self, name: str) -> int | None:
        """Tests whose behaviour depends on the entries of this fixture file."""
        if self.probes is None:
            return None
        if self._entry_users is None:
            table: dict[str, int] = defaultdict(int)
            for tid, info in self.probes.by_test.items():
                i = self.dm.index.get(tid)
                if i is not None:
                    for f in info["files"]:
                        table[f] |= 1 << i
            self._entry_users = table
        return self._entry_users.get(name, 0)

    # -- helpers -----------------------------------------------------------
    def all_tests(self) -> list[list[int]]:
        if self._all_split is None:
            self._all_split = best_split(self.ms)[1]
        return self._all_split

    def durations(self, bits: int) -> dict[int, int]:
        out = {}
        for i in self.dm.ids(bits):
            n = self.num[i]
            if n is not None:
                out[n] = self.ms[n]
        return out

    def estimate(self, bits: int, extra_cpu: float = 0.0, extra_classes=()) -> float:
        """CPU seconds one child needs to re-run these tests whole (plus extra sliced work)."""
        idx = self.dm.ids(bits)
        if not idx and not extra_cpu:
            return 0.0
        dm = self.dm
        est = 0.05 + extra_cpu
        imp = 0.15 if extra_cpu else 0.0
        cold_max = cold_sum = 0.0
        need_setup = set(extra_classes)
        for i in idx:
            est += dm.cpu[i]
            if i not in self.no_setup:
                need_setup.add(self._cls[i])
            if dm.import_cost[i] > imp:
                imp = dm.import_cost[i]
            c = min(dm.cold[i], 0.6)
            cold_sum += c
            if c > cold_max:
                cold_max = c
        est += imp
        for c in need_setup:
            est += dm.class_setup.get(c, 0.0)
        # Lazy imports and cold caches are paid once per process, mostly by the first test.
        est += min(1.0, cold_max + 0.1 * (cold_sum - cold_max))
        return est

    def cheap(self, bits: int) -> bool:
        return self.estimate(bits) <= CHEAP_SET_S

    def needs_heavy(self, test_ids) -> bool:
        for tid in test_ids:
            f = tid.split("::", 1)[0]
            v = self._heavy_cache.get(f)
            if v is None:
                try:
                    v = bool(HEAVY_RE.search(Path(SRC, f).read_text()))
                except OSError:
                    v = True
                self._heavy_cache[f] = v
            if v:
                return True
        return False

    def load_heavy(self):
        if self._heavy_loaded:
            return
        self._heavy_loaded = True
        try:
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import duckdb  # noqa: F401
        except Exception as e:  # noqa: BLE001
            log(f"heavy imports failed: {e}")

    def warm_up(self):
        """Once: load the pinned tree into this process (children fork from it)."""
        if WARM and self._warm is None:
            mods = [m for m in self.dm.modules if m == "sqlglot" or m.startswith("sqlglot.")] if HOT else ()
            self._warm = verify.warm_up(SRC, HARNESS, mods)

    def import_names(self, plan) -> list[str]:
        """The modules to import to see whether a patch that changed import-time code left
        the tree importable: the changed modules of the package and every module importing
        one of them, plus the changed test modules."""
        own = {m for m in plan.import_check if m == "sqlglot" or m.startswith("sqlglot.")}
        names: list[str] = []
        if own:
            if self.facts is not None:
                from . import state as st

                scope = st.dependents(self.facts["imports"], own)
                names = [m for m in self.facts["modules"] if m in scope]
            names += sorted(own - set(names))
        return names + sorted(plan.import_check - own)

    def import_victims(self, failed) -> int:
        """Tests that cannot pass when these modules do not import: whatever loads one of
        them or a module importing one of them, and the tests of a test file among them."""
        dm = self.dm
        mods = set(failed)
        if self.facts is not None:
            from . import state as st

            mods = st.dependents(self.facts["imports"], mods)
        bits = 0
        for mod in mods:
            bits |= dm.module_users(mod)
        files = tuple(m.replace(".", "/") + ".py::" for m in failed if m.startswith("tests."))
        if files:
            for i, tid in enumerate(dm.tests):
                if tid.startswith(files):
                    bits |= 1 << i
        return bits

    def with_import_check(self, final, plan, overlay: str, names, info: dict):
        """A selection made without re-running anything has not seen whether the patched
        modules still import: look, and add the tests that cannot pass if one does not."""
        verify.remove_overlay(overlay)
        c0 = os.times()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            limit = min(float(STATE_IMPORT_LIMIT_S), max(5.0, self.wall_budget - (time.monotonic() - self.t0)))
            failed = verify.probe_imports(overlay, HARNESS, names, limit, SRC if self._warm else None)
        finally:
            verify.remove_overlay(overlay)
        c1 = os.times()
        info["verify_cpu"] = round((info.get("verify_cpu") or 0.0) + (c1.children_user - c0.children_user)
                                   + (c1.children_system - c0.children_system), 3)
        if failed is None:
            # Importing the patched tree does not come back: nothing can run.
            info["decision"] = "all(import hang)"
            return self.all_tests()
        if not failed:
            return final
        info["import_failed"] = sorted(failed)[:4]
        chosen = {n: self.ms[n] for w in final for n in w}
        chosen.update(self.durations(self.import_victims(failed)))
        if not chosen:
            return final
        for n in self.unknown:
            chosen[n] = self.ms[n]
        return best_split(chosen)[1]

    # -- a test that loops over the dialects -------------------------------------------------
    def loop_ready(self, plan) -> bool:
        """Could the loop test be cut down for this patch? (Measures, once, what each round of
        the loop uses on the pinned tree: that has to happen in this process's lineage before
        a child imports a patched tree.)"""
        if not USE_LOOPS or self._loop_failed or self.facts is None or "imports_all" not in self.facts:
            return False
        i = self.dm.index.get(LOOP_TEST)
        if i is None or not (plan.candidates >> i) & 1:
            return False
        if not plan.changed or any(not (f.startswith("sqlglot/") and f.endswith(".py")) or text is None
                                   for f, text in plan.changed.items()):
            return False
        if not self.same_imports(plan):
            return False
        if self._loop_uses is None:
            c0 = os.times()
            uses = verify.dialect_uses(SRC)
            c1 = os.times()
            self.t_imports += (c1.children_user - c0.children_user) + (c1.children_system - c0.children_system)
            if not uses:
                self._loop_failed = True
                return False
            self._loop_uses = {k: set(v) for k, v in uses.items()}
        return True

    def same_imports(self, plan) -> bool:
        """Does every changed file import what it imports on the pinned commit? (The import
        graph of the pinned tree is only good for such patches.)"""
        import ast

        def imports(text: str):
            return sorted(ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.Import, ast.ImportFrom)))

        try:
            for rel, text in plan.changed.items():
                if imports(Path(SRC, rel).read_text(encoding="utf-8")) != imports(text):
                    return False
        except (OSError, SyntaxError, ValueError, RecursionError):
            return False
        return True

    def loop_slice(self, plan, extra_modules=()):
        """The members of the dialect enumeration whose round of the loop test the patch can
        reach, or None when that is all of them (or cannot be told). A round is reached when
        one of the modules it uses on the pinned tree is a changed module, a module in which
        importing the patched tree computed something different (extra_modules), or a module
        that imports one of those, directly or not. Any other round runs the same code on the
        same objects as on the pinned commit."""
        if self._loop_uses is None:
            return None
        from . import state as st

        mods = set(extra_modules)
        for rel in plan.changed:
            m = rel[:-3].replace("/", ".")
            mods.add(m[:-9] if m.endswith(".__init__") else m)
        reach = st.dependents(self.facts["imports_all"], mods)
        hit = sorted(d for d, used in self._loop_uses.items() if used & reach)
        if not hit or len(hit) >= len(self._loop_uses):
            return None
        return hit

    def with_loop_slice(self, plan, extra_modules=()):
        """(members to keep or None, the CPU of the loop test as the map has it): while the
        commit is decided and re-run, the test counts for what the kept rounds cost."""
        i = self.dm.index.get(LOOP_TEST)
        hit = self.loop_slice(plan, extra_modules)
        if i is None or hit is None:
            return None, None
        before = self.dm.cpu[i]
        self.dm.cpu[i] = before * len(hit) / len(self._loop_uses) + 0.02
        return hit, before

    def phase_spec(self, tid: str, plan, costs: list):
        """(expected CPU of the looks, CPU of the queries that have to be run in any case, spec
        for brh.PHASED) for settling this test in two phases on the given affected fixture
        entries [(cpu, file, entry)], or None when that is not possible. The entries whose
        lines outside parsing and optimizing the patch leaves alone only need a look at their
        parsed and optimized expressions; the others have to be executed."""
        if not USE_PHASES or not plan.phase_ok or not plan.regions or not costs:
            return None
        if plan.phase_fixtures and not PHASE_MORE:
            return None
        if plan.phase_fixtures:
            i = self.dm.index.get(tid)
            mine = self.fixture_files.get(i, ()) if i is not None else ()
            for rel in plan.phase_fixtures:
                if (i is None or rel.startswith(PHASE_DATA) or (self.dm.openers(rel) >> i) & 1
                        or any(rel.endswith(name) for name in mine)):
                    return None     # the test reads the edited file
        if any(rel.startswith(PHASE_AFTER) for rel in plan.changed):
            return None     # the executor and the planner are what runs after the optimizer
        ph = self.dm.phases().get(tid)
        if ph is None:
            return None
        mask = 0
        for rel, lo, hi in plan.regions:
            for line, bit in ph["by_file"].get(rel, ()):
                if lo <= line <= hi:
                    mask |= 1 << bit
        eligible = []
        look = owed = 0.0
        for c, f, k in costs:
            if f != ph["file"]:
                return None
            other = ph["other"].get(str(k))
            if other is None or ph["fp"].get(str(k)) is None or other & mask:
                owed += c
            else:
                eligible.append(k)
                look += ph["opt_cpu"].get(str(k), 0.1)
        if not eligible:
            return None
        return look, owed, {"fp": {str(k): ph["fp"][str(k)] for k in eligible}, "eligible": eligible,
                            "cost": {str(k): round(c, 3) for c, _, k in costs}, "max_exec": PHASE_EXEC_S,
                            "look_cost": {str(k): ph["opt_cpu"].get(str(k), 0.1) for k in eligible}}

    def after_pool_break(self, risky: int, broke: int = 0) -> int:
        """Tests that fail when one of `risky` breaks a process pool.

        A test class may make one process pool in a class fixture and use it in many tests
        (maps/pools.json.z lists them). If a test kills a worker of that pool, every test of
        the class that uses the pool and runs after it fails with BrokenProcessPool, whether
        or not it executes the changed code. A test that was re-run here to the end is known
        to break the pool or not (brh.InlinePool); one that is selected without that is not,
        so the tests after it are selected with it (`risky`). The suite runs the methods of
        a unittest class in the order of their names. When a re-run did break a pool
        (`broke`), every test using that pool is selected: just when a real worker dies
        need not be when the stand-in noticed."""
        extra = 0
        dm = self.dm
        for users in dm.shared_pools():
            if broke & users:
                extra |= users
                continue
            hit = risky & users
            if not hit:
                continue
            if self._meth is None:
                self._meth = [t.rsplit("::", 1)[1] for t in dm.tests]
            first = min(self._meth[i] for i in dm.ids(hit))
            for i in dm.ids(users):
                if self._meth[i] > first:
                    extra |= 1 << i
        return extra

    def entry_dependent(self, i: int) -> bool:
        """Does what this test runs depend on the entries of a fixture file it loads?"""
        if i not in self.fixture_files:
            return False
        if self.probes is None:
            return True
        info = self.probes.by_test.get(self.dm.tests[i])
        return info is None or bool(info["files"])

    # -- slicing -----------------------------------------------------------
    def slice_plan(self, plan, files, cand: int, settle, settle_all: bool = False, lean: bool = False):
        """Decide how each candidate is re-run.

        lean: do not re-run any part of the tests that may be left out unsettled.

        settle_all: re-run every candidate completely, whatever it costs compared with
        selecting it, except the tests that may be left out unsettled (never_alone).

        Returns (whole, slices, hard, lights):
          whole  bitset of tests re-run completely
          slices {test index: ({fixture file: entries}, est cpu, complete?)} tests re-run on some
                 fixture entries only; complete=False means only a sample of the affected entries
          hard   bitset of tests that are not re-run at all
          lights {test index: spec} tests re-run without their dominant statement(s)
        """
        dm = self.dm
        sliced_only = plan.sliced & ~plan.full
        whole = hard = 0
        slices: dict[int, tuple[dict, float, bool]] = {}
        lights: dict[int, dict] = {}

        def affordable(i: int) -> bool:
            # Worth re-running whole: cheap, or clearly cheaper than selecting it blindly.
            n = self.num[i]
            rec = self.ms[n] / 1000 if n is not None else 0.0
            if settle_all and not (self.never_alone >> i) & 1:
                return True
            return dm.cpu[i] <= max(EXPENSIVE_S, WHOLE_RATIO * (rec + 2.0))

        def sliceable(i: int) -> bool:
            # A fixture-driven test costly enough for a partial re-run to be worth planning.
            files_of = self.probes.entry_files(dm.tests[i]) if self.probes is not None else {}
            return dm.cpu[i] > SLICE_MIN_S and bool(files_of) and "@unit" not in files_of

        todo = [i for i in dm.ids(cand) if not (sliced_only >> i) & 1]
        pcand = None
        unit_spec = self.probes.dm.extra.get("units", {}) if (self.probes is not None and UNIT_MIN_S >= 0) else {}

        def unit_test(i: int) -> bool:
            # A test whose method can be re-run a unit at a time (see tools/tracer/stmt_spec.py).
            tid = dm.tests[i]
            return (tid in unit_spec and dm.cpu[i] >= UNIT_MIN_S and not (plan.whole_only >> i) & 1
                    and tid.split("::", 1)[0] not in plan.changed)

        if self.probes is not None and any(not affordable(i) or sliceable(i) or unit_test(i) for i in todo):
            pplan = make_plan(self.probes.dm, SRC, files, python_only=True)
            settle(pplan, self.probes.dm)
            pcand = pplan.candidates
        light_specs = dm.extra.get("light", {})

        for i in dm.ids(cand):
            tid = dm.tests[i]
            my_files = self.fixture_files.get(i, [])
            if (sliced_only >> i) & 1:
                # Only fixture entries changed: re-run those (other fixtures of the test are untouched).
                flt = {f: set(plan.fixture_filter.get(f, ())) for f in my_files}
                slices[i] = (flt, min(dm.cpu[i], 0.3), True, False)
                continue
            if lean and (self.never_alone >> i) & 1:
                hard |= 1 << i
                continue
            if pcand is not None and unit_test(i):
                info = self.probes.by_test.get(tid)
                units = info["files"].get("@unit") if info else None
                if units:
                    pcpu = self.probes.dm.cpu
                    hit = pcand & info["bits"]
                    base_cost = pcpu[info["base"]] if info["base"] is not None else 0.0
                    extra = {k: max(0.0, pcpu[pi] - base_cost) for k, pi in units.items()}
                    chosen = [k for k, pi in units.items() if (hit >> pi) & 1]
                    base_hit = info["base"] is not None and (hit >> info["base"]) & 1
                    total = base_cost + sum(extra.values())
                    part = base_cost + sum(extra[k] for k in chosen)
                    # Worth it when the affected units are clearly less than the whole test;
                    # when the kept units themselves are affected, or no unit is (the maps
                    # disagree), the test is run whole as before.
                    if chosen and not base_hit and total > 0 and part <= UNIT_RATIO * total:
                        slices[i] = ({"@unit": set(chosen)}, 0.002 + dm.cpu[i] * part / total, True, False)
                        continue
            if (plan.whole_only >> i) & 1 or tid in unit_spec:
                # A candidate for more than its Python dependencies, or a test without
                # fixture entries to choose from: only a whole run settles it.
                spec = light_specs.get(tid)
                if affordable(i):
                    whole |= 1 << i
                elif (spec and tid.split("::", 1)[0] not in plan.changed and spec["light_cpu"] <= SLICE_BUDGET_S
                      and not (plan.whole_only >> i) & 1 and (self.never_alone >> i) & 1):
                    # A test that may be left out on evidence: its light run (the test without
                    # its dominant statements) is cheap extra evidence. It settles nothing.
                    lights[i] = spec
                else:
                    hard |= 1 << i
                continue
            can_slice = pcand is not None and sliceable(i)
            if affordable(i) and not can_slice:
                whole |= 1 << i
                continue
            info = self.probes.by_test.get(tid) if (self.probes is not None and pcand is not None) else None
            if not info or not info["files"]:
                spec = light_specs.get(tid)
                if (spec and tid.split("::", 1)[0] not in plan.changed and spec["light_cpu"] <= SLICE_BUDGET_S
                        and (self.never_alone >> i) & 1):
                    lights[i] = spec   # extra evidence only (see above)
                else:
                    hard |= 1 << i
                continue
            pcpu = self.probes.dm.cpu
            hit = pcand & info["bits"]
            flt = {f: set() for f in set(my_files) | set(info["files"])}
            # Per-entry timings come from traced one-entry runs and carry a fixed overhead
            # each; scale them so that all entries together cost what the test costs.
            scale = info.get("scale")
            if scale is None:
                total = sum(pcpu[pi] for entries in info["files"].values() for pi in entries.values())
                scale = info["scale"] = dm.cpu[i] / total if total > 0 else 1.0
            costs = []
            for f, entries in info["files"].items():
                for k, pi in entries.items():
                    if (hit >> pi) & 1:
                        costs.append((pcpu[pi] * scale, f, k))
            if (plan.sliced >> i) & 1:
                for f in my_files:
                    for k in plan.fixture_filter.get(f, ()):
                        costs.append((0.05, f, k))
            base_hit = info["base"] is not None and (hit >> info["base"]) & 1
            if not costs and not base_hit:
                if affordable(i):
                    whole |= 1 << i
                else:
                    hard |= 1 << i   # the maps disagree about this test: do not trust a slice
                continue
            costs.sort()
            total = sum(c for c, _, _ in costs)
            if tid == PHASE_TEST:
                self._phase_costs = list(costs)
            # Settling the test by re-running every affected entry must be cheaper than
            # simply selecting it.
            n = self.num[i]
            rec = self.ms[n] / 1000 if n is not None else 0.0
            complete = (total <= min(SLICE_BUDGET_S, max(0.5, SLICE_RATIO * (rec + 2.0)))
                        or (settle_all and not (self.never_alone >> i) & 1))
            if affordable(i) and (not complete or total >= 0.8 * dm.cpu[i]):
                whole |= 1 << i   # nearly everything is affected: just run it whole
                continue
            setup = dm.class_setup.get(self._cls[i], 0.0)
            if not complete and total + setup <= min(BIG_SLICE_S, BIG_SLICE_RATIO * (rec + 2.0)):
                # Too much to re-run on spec, but far cheaper than selecting the test. Worth
                # it exactly when the test would otherwise be selected: always for import-time
                # changes, and for call-time ones once something else has failed (until then
                # the passing re-runs speak for it).
                for _, f, k in costs:
                    flt[f].add(k)
                slices[i] = (flt, total + 0.05, True, plan.all_call_time and bool((self.never_alone >> i) & 1))
                continue
            if not complete and (not plan.all_call_time or setup >= 1.0 or not (self.never_alone >> i) & 1):
                # A sample of entries is only useful as evidence for leaving the test out,
                # and is not worth building expensive class fixtures for.
                hard |= 1 << i
                continue
            budget = SLICE_BUDGET_S if complete else SAMPLE_BUDGET_S
            spent = 0.0
            for c, f, k in costs:
                if spent + c > budget and spent > 0:
                    break
                flt[f].add(k)
                spent += c
            slices[i] = (flt, spent + 0.05, complete, False)
        return whole, slices, hard, lights

    def groups(self, whole: int, slices: dict, tokens=(), changed=()) -> list[dict]:
        """Tests of a class that has sliced tests run together under the union of the
        slices, so class-level fixtures are built once; everything else runs unfiltered.
        The result is what verify.run_groups takes."""
        dm = self.dm
        tokens = [t for t in tokens if len(t) >= 4]

        def affinity(i: int) -> int:
            # Tests named after the changed file or function are the likeliest to notice.
            name = dm.tests[i].lower()
            return sum(len(t) for t in tokens if t in name)

        sliced_classes = {self._cls[i] for i, s in slices.items() if not s[3]}
        g0: list[int] = []
        per_class: dict[str, tuple[list[int], dict]] = {}
        later: dict[str, tuple[list[int], dict]] = {}   # slices only run once something failed
        for i in dm.ids(whole):
            c = self._cls[i]
            if c in sliced_classes and not self.entry_dependent(i):
                per_class.setdefault(c, ([], defaultdict(set)))[0].append(i)
            else:
                g0.append(i)
        for i, (flt, _, _, conditional) in slices.items():
            target = later if conditional else per_class
            ids, union = target.setdefault(self._cls[i], ([], defaultdict(set)))
            ids.append(i)
            for f, ks in flt.items():
                union[f] |= ks

        def group(ids: list[int], flt: dict, if_failed: bool = False) -> dict:
            # Class fixtures are not built for classes whose tests in the group do not use them.
            uses: dict[str, bool] = {}
            for i in ids:
                uses[self._cls[i]] = uses.get(self._cls[i], False) or i not in self.no_setup
            skip = [c for c, used in uses.items()
                    if not used and c.split("::", 1)[0] not in changed and dm.class_setup.get(c)]
            return {"ids": [dm.tests[i] for i in ids], "filter": {f: sorted(ks) for f, ks in flt.items()},
                    "skip_setup": skip, "if_failed": if_failed}

        out = []
        # Likeliest and cheapest first, a class at a time (class fixtures are built once),
        # so an early stop after a run of failures has cost little.
        by_cls: dict[str, list[int]] = defaultdict(list)
        for i in g0:
            by_cls[self._cls[i]].append(i)
        ordered: list[int] = []
        for c in sorted(by_cls, key=lambda c: (-max(affinity(i) for i in by_cls[c]),
                                               dm.class_setup.get(c, 0.0),
                                               min(dm.cpu[i] for i in by_cls[c]), c)):
            ordered.extend(sorted(by_cls[c], key=lambda i: (-affinity(i), dm.cpu[i], i)))
        out.append(group(ordered, {}))   # always first, even when empty
        for c in sorted(per_class):
            ids, union = per_class[c]
            out.append(group(sorted(ids), union))
        for c in sorted(later):
            ids, union = later[c]
            out.append(group(sorted(ids), union, if_failed=True))
        return out

    # -- one commit --------------------------------------------------------
    def decide(self, plan, files, info: dict, settle) -> Job:
        """Everything up to the re-run: what to re-run, how, and whether it is worth it."""
        dm = self.dm
        job = Job()
        cand = plan.candidates & ~plan.blind
        if not cand:
            if plan.blind:
                info["decision"] = "blind(unverifiable)"
                follow = self.after_pool_break(plan.blind) if plan.pool_risk else 0
                job.final = best_split(self.durations(plan.blind | follow))[1]
            else:
                info["decision"] = "none"
                job.final = []
            return job
        # Selecting the candidates without re-running them includes the tests that fail if one
        # of them breaks a shared process pool: possible when the patch touches code that
        # runs while pools pickle, and never excluded when the re-run dies or hangs.
        follow = self.after_pool_break(cand | plan.blind)
        job.unknown_outcome = self.durations(cand | plan.blind | follow)
        blind = self.durations(cand | plan.blind | (follow if plan.pool_risk else 0))
        for n in self.unknown:
            blind[n] = self.ms[n]
            job.unknown_outcome[n] = self.ms[n]
        job.blind = blind
        job.blind_ms = blind_ms = split_cost(blind)

        budget = VERIFY_CAP_S
        may_skip_hard = bool(SKIP_HARD) and plan.all_call_time

        def assess(settle_all: bool, lean: bool = False):
            """One way of re-running the candidates and what it is expected to cost."""
            whole, slices, hard, lights = self.slice_plan(plan, files, cand, settle, settle_all, lean)

            def cost_of():
                # Slices that only run once something has failed are not part of the plain cost.
                lb = 0
                for i in lights:
                    lb |= 1 << i
                firm = {i: s for i, s in slices.items() if not s[3]}
                extra = sum(s[1] for s in firm.values()) + sum(s["light_cpu"] for s in lights.values())
                return lb, self.estimate(whole, extra, {self._cls[i] for i in list(firm) + list(lights)})

            light_bits, est = cost_of()
            if est > budget:
                # Too much to re-run. Give up first on the tests that are expensive to re-run
                # but cheap to select: leaving those unverified costs the least.
                def worth(cost: float, i: int) -> float:
                    n = self.num[i]
                    return cost / ((self.ms[n] / 1000 if n is not None else 0.0) + 0.2)

                items = [(worth(dm.cpu[i], i), dm.cpu[i], "w", i) for i in dm.ids(whole)]
                items += [(worth(s[1], i), s[1], "s", i) for i, s in slices.items() if not s[3]]
                items += [(worth(s["light_cpu"], i), s["light_cpu"], "l", i) for i, s in lights.items()]
                items.sort(reverse=True)
                for _, c, kind, i in items:
                    if est <= budget:
                        break
                    if kind == "w":
                        whole &= ~(1 << i)
                    elif kind == "s":
                        del slices[i]
                    else:
                        del lights[i]
                    hard |= 1 << i
                    est -= c
                light_bits, est = cost_of()
            # The tests whose outcome the re-run cannot settle (not re-run, sampled, or run
            # without their dominant statement) ...
            open_bits = hard | light_bits
            for i, s in slices.items():
                if not s[2]:
                    open_bits |= 1 << i
            open_ms = split_cost(self.durations(open_bits | plan.blind)) if (open_bits or plan.blind) else 0
            # ... stay selected whatever it shows, except those that may be left out on
            # evidence, and only when no other candidate is left unsettled.
            firm = open_bits & ~self.never_alone
            firm_ms = split_cost(self.durations(firm | plan.blind)) if (firm or plan.blind) else 0
            if may_skip_hard and not firm:
                after = est + (firm_ms + HARD_FAIL_SHARE * (open_ms - firm_ms)) / 1000
            else:
                after = est + open_ms / 1000
            return after, whole, slices, hard, lights, light_bits, est, open_bits, open_ms

        best = assess(False)
        if may_skip_hard and best[7] & self.never_alone and best[7] & ~self.never_alone:
            # A giant test could be left out on evidence if only the ordinary candidates
            # were all settled: see what settling every one of them would cost.
            other = assess(True)
            if other[0] < best[0]:
                best = other
            if LEAN:
                other = assess(True, lean=True)
                if other[0] < best[0]:
                    best = other
        after, whole, slices, hard, lights, light_bits, est, open_bits, open_ms = best
        info.update(n_cand=cand.bit_count(), n_whole=whole.bit_count(), n_sliced=len(slices),
                    n_light=len(lights), n_hard=hard.bit_count(), est=round(est, 3), blind=blind_ms / 1000)
        if REPORT:
            info["hard"] = [dm.tests[i].split("::")[-1] for i in dm.ids(hard)][:8]
            info["slices"] = {dm.tests[i].split("::")[-1]: [{f: len(ks) for f, ks in flt.items() if ks},
                                                            round(c, 2), ok, "if-failed" if cond else ""]
                              for i, (flt, c, ok, cond) in list(slices.items())[:8]}
            costs = [(dm.cpu[i], "whole", i) for i in dm.ids(whole)]
            costs += [(c, "slice" if ok else "sample", i) for i, (_, c, ok, _) in slices.items()]
            costs += [(dm.cpu[i], "open", i) for i in dm.ids(hard)]
            info["top"] = [[dm.tests[i].split("::")[-1], kind, round(c, 2)]
                           for c, kind, i in sorted(costs, reverse=True)[:14]]
        elapsed = time.monotonic() - self.t0
        # Do the unsettled tests already cost about as much as selecting everything?
        hard_dominates = bool(open_bits) and open_ms >= blind_ms - FIRST_FAIL_MARGIN_MS
        # What the re-run costs plus what stays selected whatever it shows, against selecting
        # every candidate: the re-run has to be expected to pay for itself.
        gain = blind_ms / 1000 - after
        info["gain"] = round(gain, 2)
        can_verify = (
            MODE == "verify"
            and (whole or slices or lights)
            and elapsed + est * 2 + 1 < self.wall_budget
            and gain >= max(GAIN_MIN_S, GAIN_RATIO * blind_ms / 1000)
        )
        if not can_verify:
            info["decision"] = "blind"
            job.final = best_split(blind)[1]
            return job
        job.whole, job.slices, job.lights, job.light_bits, job.hard = whole, slices, lights, light_bits, hard
        if PHASE_SLICES and USE_PHASES and self._phase_costs:
            # The executor's TPC-DS test, re-run on every query the patch reaches: a query
            # whose parsed and optimized expressions are those of the pinned commit, and
            # that runs no changed line afterwards, is not executed (brh.PHASED). The
            # others are, one at a time; nothing is given up, so the test is settled as
            # it would be by executing them all.
            it = dm.index.get(PHASE_TEST)
            sl = slices.get(it) if it is not None else None
            if sl is not None and sl[2] and "@unit" not in sl[0]:
                kept = {(f, q) for f, ks in sl[0].items() for q in ks}
                mine = [c for c in self._phase_costs if (c[1], c[2]) in kept]
                spec = self.phase_spec(PHASE_TEST, plan, mine) if len(mine) == len(kept) else None
                if spec is not None:
                    what = spec[2]
                    what.update(max_exec=1e9, max_differ=10 ** 9, bail=False)
                    job.phased = {PHASE_TEST: what}
                    info["phase_slice"] = [len(what["eligible"]), len(mine)]
        job.hard_dominates, job.est = hard_dominates, est
        # ... and what it may cost at most, with the slices that only run after a failure
        later = [i for i, sl in slices.items() if sl[3]]
        job.est_max = est + sum(slices[i][1] for i in later) + sum(
            dm.class_setup.get(c, 0.0) for c in {self._cls[i] for i in later})
        # Tests reduced to some of their units run with the whole tests; the rest of the
        # slices are fixture entries and run class by class.
        unit_bits = 0
        entry_slices = {}
        unit_spec = self.probes.dm.extra.get("units", {}) if self.probes is not None else {}
        for i, s in slices.items():
            chosen = s[0].get("@unit")
            if chosen is None:
                entry_slices[i] = s
                continue
            unit_bits |= 1 << i
            spans = []
            for k, unit in enumerate(unit_spec[dm.tests[i]]):
                if not unit[0] or k in chosen:
                    spans.extend([unit[j], unit[j + 1]] for j in range(1, len(unit), 2))
            job.keep[dm.tests[i]] = sorted(spans)
        job.groups = self.groups(whole | light_bits | unit_bits, entry_slices, plan.tokens, plan.changed)
        job.drops = {dm.tests[i]: spec["drop"] for i, spec in lights.items()}
        job.group_ids = [t for g in job.groups for t in g["ids"]]
        # If the tests we cannot re-run already cost about as much as selecting every
        # candidate, one failure decides the outcome: stop at the first.
        job.stop = -1 if (hard_dominates and EARLY_STOP) else EARLY_STOP
        return job

    def conclude(self, job: Job, res: dict, plan, info: dict) -> list[list[int]]:
        """From the outcome of the re-run to the selection."""
        dm = self.dm
        ran = job.whole | job.light_bits
        for i in job.slices:
            ran |= 1 << i
        index_by_id = {dm.tests[i]: i for i in dm.ids(ran)}
        failed_bits = 0
        for tid in res["failed"]:
            i = index_by_id.get(tid)
            if i is not None:
                failed_bits |= 1 << i
        info["n_fail"] = failed_bits.bit_count()
        if res.get("phase"):
            info["phase_run"] = res["phase"].get(PHASE_TEST)    # [queries looked at, different, executed]
        if "times" in res:
            info["times"] = res["times"]
        info["fail_reasons"] = {t: r[:100] for t, r in list(res["failed"].items())[:3]}
        # Tests whose outcome is not settled: never re-run, or re-run on a sample that passed.
        unsettled = job.hard
        for tid in res.get("unverified", ()):
            i = index_by_id.get(tid)
            if i is not None:
                unsettled |= 1 << i
                ran &= ~(1 << i)
        stopped = 0
        for tid in res.get("not_run", ()):
            i = index_by_id.get(tid)
            if i is not None:
                stopped |= 1 << i
        if stopped:
            # The run stopped after a streak of failures: the rest is assumed to fail too.
            failed_bits |= stopped
            info["early_stop"] = stopped.bit_count()
        for tid in res.get("deferred", ()):
            # Slices that were only to run after a failure, and nothing failed.
            i = index_by_id.get(tid)
            if i is not None:
                unsettled |= 1 << i
                ran &= ~(1 << i)
        for i, s in job.slices.items():
            if not s[2] and not (failed_bits >> i) & 1:
                unsettled |= 1 << i
        unsettled |= job.light_bits & ~failed_bits     # a light run settles nothing
        selected = failed_bits | plan.blind
        if res.get("import_failed"):
            # Modules that no longer import: every test loading one of them fails.
            info["import_failed"] = sorted(res["import_failed"])[:4]
            selected |= self.import_victims(res["import_failed"])
        if res.get("bad_files"):
            # A test file that cannot be imported: all of its tests fail, asked for or not.
            info["bad_files"] = res["bad_files"][:4]
            prefixes = tuple(f + "::" for f in res["bad_files"])
            for i, tid in enumerate(dm.tests):
                if tid.startswith(prefixes):
                    selected |= 1 << i
        decision = "verified"
        if unsettled:
            passed = (ran & ~failed_bits).bit_count()
            # A test may only be left out unsettled when it is one of those shown never to
            # fail on their own, and every other candidate was settled and passed.
            skippable = unsettled & self.never_alone
            if unsettled & ~self.never_alone:
                skippable = 0
            harmless = (SKIP_HARD and plan.all_call_time and not failed_bits and passed >= MIN_EVIDENCE
                        and bool(skippable))
            costly = skippable if harmless else 0
            selected |= unsettled & ~costly
            if costly:
                decision = "verified,skip-unsettled"
                info["skipped"] = [dm.tests[i].split("::")[-1] for i in dm.ids(costly)][:8]
            if unsettled & ~costly:
                decision += "+unsettled"
        # A test seen breaking a shared process pool fails, and so does every test using
        # that pool. Selected tests that were not re-run to the end may break one as well if
        # the patch touches code that runs while pools pickle: the tests that would then
        # fail after them come along.
        seen = ran & ~stopped & ~job.light_bits
        for i, s in job.slices.items():
            if not s[2]:
                seen &= ~(1 << i)
        breakers = 0
        for tid in res.get("pool_breaks", ()):
            i = dm.index.get(tid)
            if i is not None:
                breakers |= 1 << i
        selected |= breakers
        extra = self.after_pool_break((selected & ~seen) if plan.pool_risk else 0, breakers) & ~selected
        if extra or breakers:
            selected |= extra
            info["pool"] = [breakers.bit_count(), extra.bit_count()]
        if RISK or USE_PHASES:
            settled = ran & ~failed_bits & ~unsettled & ~stopped
            # (seen failing: not the ones only assumed to fail after an early stop)
            info["_failed"] = [self.num[i] for i in dm.ids(failed_bits & ~stopped) if self.num[i] is not None]
            info["_passed"] = [self.num[i] for i in dm.ids(settled) if self.num[i] is not None]
        out = self.durations(selected)
        if out:
            for n in self.unknown:
                out[n] = self.ms[n]
        info["decision"] = decision
        return best_split(out)[1]

    # -- leaving an expensive test out on evidence (BR_RISK) ----------------------------------
    def evidence_probe(self, plan, workers, info: dict, k: int):
        """Two tests cost 21 s and 29 s to select and more than that to re-run. When one of them
        is in the selection without having been seen to fail, something cheaper is tried:

        * the executor's TPC-DS test is settled in two phases when the patch allows it
          (phase_spec): this is exact, not a bet;
        * (BR_RISK) the parser test (a statement of a million nodes has to be refused) is run
          in a scaled-down form, RISK_SCALED, and has to pass in about the time that takes
          on the pinned commit;
        * (BR_RISK) for the executor's test, where two phases are not possible, only when the
          commit was decided without a re-run and only call-time code changed: the selected
          cheap tests of its class and the cheapest of the other selected tests are re-run,
          unless an executor or optimizer test is already known to fail.

        A test that is settled as passing, or whose evidence passes, is taken out of the
        selection; everything else stays. The last two are bets, not proofs: see the README."""
        selected = {n for w in workers for n in w}
        if not selected:
            return workers
        decision = str(info.get("decision"))
        if decision.startswith("all") or "hang" in decision or info.get("import_failed") or info.get("bad_files"):
            return workers
        # Only for changes to the package's own code: the evidence says nothing about a test
        # whose own code or data (tests/, fixtures, other files) was changed.
        risk = RISK
        if not RISK_ANY and (not plan.changed or any(not (f.startswith("sqlglot/") and f.endswith(".py")) or text is None
                                                     for f, text in plan.changed.items())):
            # ... except that the two-phase check, which is exact, holds beside edits of
            # fixture files that the executor's test does not read (phase_spec looks at that).
            others = [f for f, text in plan.changed.items()
                      if not (f.startswith("sqlglot/") and f.endswith(".py") and text is not None)]
            if not (PHASE_MORE and USE_PHASES and others and len(others) < len(plan.changed)
                    and all(f in plan.phase_fixtures for f in others)):
                return workers
            risk = 0
        dm = self.dm
        failed = set(info.get("_failed") or ())
        passed = set(info.get("_passed") or ())
        index_of = {n: i for i, n in enumerate(self.num) if n is not None}
        name = {n: dm.tests[i] for n, i in index_of.items()}
        big = {self.num[dm.index[t]] for t in RISK_TESTS if t in dm.index}
        unsure = (big & selected) - failed
        if not unsure:
            return workers
        report: dict = {"unsure": len(unsure)}
        info["risk"] = report
        both = {n for n in index_of if name[n].startswith((RISK_EXEC, RISK_OPT))}
        # -- two phases for the executor test: exact
        phased: dict = {}
        phase_group = None
        phase_est = phase_wall = 0.0
        phase_all = False
        phase_n = next((n for n in unsure if name[n] == PHASE_TEST), None)
        if phase_n is not None and USE_PHASES and not self._phase_costs:
            # Which queries the patch reaches was not worked out for this commit: all of them, then.
            ph = dm.phases().get(PHASE_TEST)
            if ph is not None:
                self._phase_costs = [(0.65, ph["file"], int(q)) for q in sorted(ph["fp"], key=int)]
                phase_all = True
        if phase_n is not None and USE_PHASES:
            # (why two phases were not tried, for the report)
            report["phase_why"] = ("no affected queries known" if not self._phase_costs
                                   else "an executor test fails" if failed & {n for n in both if name[n].startswith(RISK_EXEC)}
                                   else "not only call-time code" if not plan.phase_ok or not plan.regions
                                   else "fixture files edited" if plan.phase_fixtures
                                   else "queries run changed code after the optimizer")
        if phase_n is not None and self._phase_costs and not (failed & {n for n in both if name[n].startswith(RISK_EXEC)}):
            spec = self.phase_spec(PHASE_TEST, plan, self._phase_costs)
            if spec is not None:
                report.pop("phase_why", None)
                look, owed, what = spec
                cls_key = self._cls[index_of[phase_n]]
                setup = dm.class_setup.get(cls_key, 0.0) if owed else 0.0
                report["phase_est"] = [round(look, 2), round(owed, 2), len(what["eligible"]), len(self._phase_costs)]
                saves = (best_split({n: self.ms[n] for n in selected})[0]
                         - best_split({n: self.ms[n] for n in selected if n != phase_n})[0]) / 1000.0
                report["phase_saves"] = round(saves, 1)
                if PHASE_MORE and saves < 1.5 * (look + owed + setup) + 1.0:
                    report["phase_why"] = "saves too little"
                elif owed <= PHASE_EXEC_S and look + owed + setup <= PHASE_PROBE_S:
                    flt: dict = {}
                    for _, f, q in (() if phase_all else self._phase_costs):
                        flt.setdefault(f, set()).add(q)
                    # A ceiling; the looks watch their own pace. Both leave room for a machine
                    # several times slower than the one the map's seconds were measured on.
                    what["max_cpu"] = round(10.0 * look + 4.0, 2)
                    if not PHASE_EXEC_S:
                        # Looks only: the first query that differs leaves the test unsettled, no
                        # fixture is built and nothing is executed, so what a try costs is
                        # bounded by the looks whatever the machine.
                        what["max_differ"] = 0
                    phased[PHASE_TEST] = what
                    phase_group = {"ids": [PHASE_TEST], "filter": {f: sorted(ks) for f, ks in flt.items()},
                                   "skip_setup": [cls_key], "if_failed": False, "own_setup": True}
                    phase_est = look + owed + setup
                    phase_wall = what["max_cpu"] + PHASE_EXEC_S * 2 + 4.0
        if not risk and phase_group is None:
            report["outcome"] = "not tried"
            return workers
        # -- which evidence to gather for the bets
        scaled = sorted(n for n in unsure if risk and RISK_SCALE and name[n] in RISK_SCALED)
        if RISK_IMPORT or (plan.all_call_time and not info.get("state")):
            pass
        else:
            scaled = []
        by_neighbours = (unsure - set(scaled)) if risk else set()
        if phase_group is not None:
            by_neighbours.discard(phase_n)      # settled exactly instead
        todo: list = []
        todo_m: list = []
        if RISK_CLASS and (scaled or by_neighbours) and len(failed) >= RISK_MANY:
            # A commit that breaks this much can break any test.
            scaled, by_neighbours = [], set()
            report["neighbours"] = "many failures"
        if RISK_CLASS and scaled:
            # The parser's test: every candidate of its class has to pass as well.
            mates = {n for n in index_of if name[n].startswith(RISK_PARSER)} - big
            if mates & failed:
                scaled = []
                report["stand_in"] = "a parser test fails"
            else:
                todo_m = sorted((mates & selected) - passed)
        if by_neighbours and not (RISK_VU or RISK_CLASS) and decision.split("(")[0] != "blind":
            # After a re-run of the other candidates the expensive test is one of the few
            # tests left that could notice the change: there the neighbours were wrong too often.
            by_neighbours = set()
            report["neighbours"] = "not asked"
        if by_neighbours and (not plan.all_call_time or info.get("state")):
            # A change to what importing a module does: when it could not be worked out which
            # tests use the changed definitions, tests can depend on what ran before them in
            # the suite, and a run of a few tests on their own does not show that.
            notes = info.get("notes") or ()
            if not RISK_IMPORT_T or any("module users" in n or "uses unknown" in n for n in notes):
                by_neighbours = set()
                report["neighbours"] = "not asked (import time)"
        if by_neighbours:
            never = {self.num[i] for i in dm.ids(self.never_alone)}
            evidence = set()
            for n in both - never:
                limit = RISK_EXEC_S if name[n].startswith(RISK_EXEC) else RISK_OPT_S
                if self.ms[n] < limit * 1000:
                    evidence.add(n)
            if RISK_CLASS:
                # every test of the executor's and the optimizer's class, whatever it costs:
                # the ones still selected without a verdict are run now
                evidence = both - big
            todo = sorted((evidence & selected) - passed)
            # ... and the cheapest of the other selected tests: when they fail, the change
            # breaks things wholesale (the generator, say, which the expensive test also
            # runs and its cheap neighbours do not).
            others = sorted((selected - big - evidence - passed - failed - never), key=lambda n: (self.ms[n], n))
            todo += [n for n in others if self.ms[n] < 500][:RISK_SAMPLE]
            report["to_run"] = len(todo)
            report["seen_passing"] = len(evidence & passed)
            if both & failed:
                by_neighbours = set()           # a neighbour is known to fail: the test stays
                todo = []
                report["neighbours"] = "seen failing"
            # (the executor tests are the ones that count: the optimizer tests only add to them)
            elif sum(1 for n in evidence & (passed | set(todo)) if name[n].startswith(RISK_EXEC)) < RISK_MIN_EVIDENCE:
                by_neighbours = set()
                todo = []
                report["neighbours"] = "too few"
        if not scaled and not by_neighbours and phase_group is None:
            report["outcome"] = "too little evidence"
            return workers
        out = set()
        if scaled or todo or phase_group is not None:
            bits_m = 0
            for n in todo_m:
                bits_m |= 1 << index_of[n]
            est_m = self.estimate(bits_m) if bits_m else 0.0
            if est_m > RISK_PROBE_S / 2:
                scaled, todo_m, bits_m, est_m = [], [], 0, 0.0      # too much to pay for
                report["stand_in"] = "over budget"
            bits = 0
            for n in todo:
                bits |= 1 << index_of[n]
            est = (self.estimate(bits | bits_m) if (bits | bits_m) else 0.0) + 0.4 * len(scaled)
            if est > RISK_PROBE_S:
                # too much evidence to pay for: the neighbour bet is dropped, the rest stays
                by_neighbours, todo, bits = set(), [], 0
                est = est_m + 0.4 * len(scaled)
                report["neighbours"] = "over budget"
            bits |= bits_m
            if RISK_CLASS and (scaled or by_neighbours):
                # ... and not when it costs more than leaving the tests out saves
                gone = set(scaled) | by_neighbours
                saves = (best_split({n: self.ms[n] for n in selected})[0]
                         - best_split({n: self.ms[n] for n in selected if n not in gone})[0]) / 1000.0
                report["saves"] = round(saves, 1)
                if saves < 1.2 * est + 0.5:
                    scaled, todo_m, by_neighbours, todo, bits, est = [], [], set(), [], 0, 0.0
                    report["neighbours"] = "saves too little"
            est += phase_est
            report["est"] = round(est, 2)
            if time.monotonic() - self.t0 + est * 2 + 2 > self.wall_budget or not (scaled or bits or phase_group):
                report["outcome"] = "over budget"
                return workers
            res, cpu = self.probe_run(plan, k, bits, [name[n] for n in scaled], phase_group, phased,
                                      extra_s=0.4 * len(scaled), extra_wall=phase_wall)
            info["verify_cpu"] = round((info.get("verify_cpu") or 0.0) + cpu, 3)
            report["cpu"] = round(cpu, 2)
            if (res is None or res.get("deferred") or res.get("pool_breaks")
                    or res.get("bad_files") or res.get("import_failed") or "unswappable" in res):
                report["outcome"] = "kept"
                return workers
            bad = set(res.get("failed") or ())
            skipped = set(res.get("not_run") or ()) | set(res.get("unverified") or ())
            took = res.get("scaled_s") or {}
            if took:
                report["scaled_s"] = took
            if res.get("phase"):
                report["phase"] = res["phase"].get(PHASE_TEST)
            mates_ok = not any(name[x] in bad or name[x] in skipped for x in todo_m)
            for n in scaled:
                # A stand-in that passes but takes much longer than it does on the pinned
                # commit says the real test may not finish in the time it is given.
                if (mates_ok and name[n] not in bad and name[n] not in skipped
                        and 0 <= took.get(name[n], -1.0) <= self.scaled_limit(name[n], k)):
                    out.add(n)
            if phase_group is not None and PHASE_TEST not in bad and PHASE_TEST not in skipped:
                out.add(phase_n)                # every affected query does what it does on the pinned commit
                report["settled"] = "two phases"
            special = {name[n] for n in scaled} | {PHASE_TEST}
            if by_neighbours and not (bad - special) and not (skipped - special):
                out |= by_neighbours
        else:
            out |= by_neighbours                # every neighbour was seen passing already
        if not out:
            report["outcome"] = "kept"
            return workers
        report["outcome"] = "left out"
        report["tests"] = sorted(name[n].rsplit("::", 1)[1] for n in out)
        left = {n: self.ms[n] for n in selected - out}
        return best_split(left)[1] if left else []

    def scaled_limit(self, tid: str, k: int) -> float:
        """How long the stand-in of a test may take: a few times what it takes on the pinned
        tree on this machine (measured once, in a child that runs it on the pinned modules);
        the fixed limit when that cannot be measured."""
        base = self._scaled_base.get(tid)
        if base is None:
            base = -1.0
            overlay = os.path.join(TMP, f"br_sb_{os.getpid()}_{k}")
            try:
                self.warm_up()
                if self._warm:
                    verify.warm_tests([tid])
                    verify.remove_overlay(overlay)
                    verify.build_overlay(SRC, overlay, {})
                    res = verify.run_tests(overlay, HARNESS, [{"ids": [tid], "filter": {}, "skip_setup": [], "if_failed": False}],
                                           timeout=20.0, warm_root=SRC, scale={tid: RISK_SCALED[tid]})
                    if res is not None and not res.get("failed") and not res.get("unverified"):
                        base = float((res.get("scaled_s") or {}).get(tid, -1.0))
            except Exception:  # noqa: BLE001 - no baseline: the fixed limit
                base = -1.0
            finally:
                verify.remove_overlay(overlay)
            self._scaled_base[tid] = base
        if base <= 0:
            return RISK_SCALE_S
        return max(RISK_SCALE_X * base, base + 0.3)

    def class_bits(self, cls: str) -> int:
        """The tests of a class, as a bitset over the map's tests."""
        if self._class_bits is None:
            self._class_bits = {}
            for i, c in enumerate(self._cls):
                self._class_bits[c] = self._class_bits.get(c, 0) | (1 << i)
        return self._class_bits.get(cls, 0)

    def probe_run(self, plan, k: int, bits: int, scaled=(), phase_group=None, phased=None, extra_s: float = 0.0,
                  extra_wall: float = 0.0):
        """Run the scaled-down stand-ins, the two-phase test, then re-run the given tests
        whole, against the patched tree. Each of the three is run whatever the others show.
        (result of verify.run_tests or None, CPU seconds it took)"""
        dm = self.dm
        groups = self.groups(bits, {}, plan.tokens, plan.changed) if bits else []
        if phase_group is not None:
            groups.insert(0, phase_group)
        if scaled:
            # first, and on their own: what they show does not depend on the other tests
            groups.insert(0, {"ids": list(scaled), "filter": {}, "skip_setup": [], "if_failed": False})
        # When a class's fixtures cannot be built, every test of the class fails, the expensive
        # one included. What setUpClass runs is in the coverage of every test of its class, so
        # the fixtures can only be affected when all of them are candidates: then they are
        # built here even though the cheap tests do not use them.
        candidates = plan.full | plan.sliced
        for g in groups:
            if not g.get("own_setup"):      # (the two-phase runner builds its fixture when it needs it)
                g["skip_setup"] = [c for c in g.get("skip_setup", ()) if self.class_bits(c) & ~candidates]
        ids = [t for g in groups for t in g["ids"]]
        overlay = os.path.join(TMP, f"br_ev_{os.getpid()}_{k}")
        if self.needs_heavy(ids):
            self.load_heavy()
        self.warm_up()
        use_warm = bool(self._warm) and plan.warm_ok
        use_hot = (bool(self._warm) and not use_warm and bool(HOT) and plan.hot_ok and bool(plan.swaps)
                   and verify.can_swap())
        if use_warm or use_hot:
            verify.warm_tests(ids)
        est = (self.estimate(bits) if bits else 0.0) + extra_s
        verify.remove_overlay(overlay)
        c0 = os.times()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            base = max(12.0, 6 * est + 8) if (bits or scaled or not extra_wall) else 0.0
            timeout = min(base + extra_wall, max(5.0, self.wall_budget - (time.monotonic() - self.t0)))
            scale = {t: RISK_SCALED[t] for t in scaled}
            res = verify.run_tests(overlay, HARNESS, groups, timeout=timeout, stop_after=0,
                                   warm_root=SRC if (use_warm or use_hot) else None,
                                   parent_root=SRC if self._warm else None,
                                   swaps=list(plan.swaps.values()) if use_hot else None, scale=scale,
                                   phased=phased)
            if use_hot and res is not None and "unswappable" in res:
                res = verify.run_tests(overlay, HARNESS, groups, timeout=timeout, stop_after=0, parent_root=SRC,
                                       scale=scale, phased=phased)
        finally:
            verify.remove_overlay(overlay)
        c1 = os.times()
        return res, (c1.children_user - c0.children_user) + (c1.children_system - c0.children_system)

    def handle(self, path: Path, k: int) -> tuple[list[list[int]], dict]:
        self._plan = None
        self._phase_costs = None
        workers, info = self.handle_safely(path, k)
        if (RISK or USE_PHASES) and self._plan is not None:
            try:
                workers = self.evidence_probe(self._plan, workers, info, k)
            except Exception as e:  # noqa: BLE001 - the safe selection stands
                info["risk"] = f"error {type(e).__name__}: {e}"
        info.pop("_failed", None)
        info.pop("_passed", None)
        return workers, info

    def handle_safely(self, path: Path, k: int) -> tuple[list[list[int]], dict]:
        info: dict = {"commit": path.stem}
        planned = self._plans.pop(path, None)
        if planned is None:
            p0 = time.process_time()
            files = parse_patch(path.read_text(encoding="utf-8", errors="replace"))
            plan = make_plan(self.dm, SRC, files, self.cheap, entry_users=self.entry_users)
            self.t_plan += time.process_time() - p0
        elif isinstance(planned, Exception):
            raise planned
        else:
            files, plan = planned
        info["notes"] = plan.notes
        if plan.select_all:
            info["decision"] = "all"
            return self.all_tests(), info
        self._plan = plan
        dm = self.dm
        overlay = os.path.join(TMP, f"br_ov_{os.getpid()}_{k}")

        # Module / class level changes: measure what they alter instead of guessing,
        # unless the guess is cheap to re-run anyway.
        if self.wants_state(plan) and self.facts is not None:
            from . import state as st

            base = self.state_for(st.dependents(self.facts["imports"], set(plan.import_modules)))
            if base is not None:
                return self.handle_with_state(plan, files, overlay, info, base)

        def settle(pl, _dm):
            pl.settle_imports(None)

        settle(plan, dm)
        loops, loop_cpu = self.with_loop_slice(plan) if self.loop_ready(plan) else (None, None)
        try:
            job = self.decide(plan, files, info, settle)
        finally:
            if loop_cpu is not None:
                dm.cpu[dm.index[LOOP_TEST]] = loop_cpu
        job.loops = {LOOP_TEST: loops} if loops is not None else {}
        if loops is not None:
            info["loop"] = len(loops)
        # Import-time code changed: the patched modules may not import any more.
        check = self.import_names(plan) if plan.import_check else []
        if job.final is not None:
            if check:
                job.final = self.with_import_check(job.final, plan, overlay, check, info)
            return job.final, info
        p0 = time.process_time()
        if self.needs_heavy(job.group_ids):
            self.load_heavy()
        # Patches that leave the Python code alone are re-run on the modules this process
        # has already imported from the pinned tree.
        self.warm_up()
        use_warm = bool(self._warm) and plan.warm_ok
        # Function bodies only: the pinned modules stay, the functions get their new code.
        use_hot = (bool(self._warm) and not use_warm and bool(HOT) and plan.hot_ok and bool(plan.swaps)
                   and verify.can_swap())
        if use_warm or use_hot:
            verify.warm_tests(job.group_ids)
        info["warm"] = "hot" if use_hot else use_warm
        self.t_imports += time.process_time() - p0
        verify.remove_overlay(overlay)
        c0 = os.times()
        w0 = time.monotonic()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            elapsed = time.monotonic() - self.t0
            timeout = min(max(12.0, 6 * job.est_max + 8), max(5.0, self.wall_budget - elapsed))
            res = verify.run_tests(overlay, HARNESS, job.groups, timeout=timeout, drops=job.drops,
                                   stop_after=job.stop, warm_root=SRC if (use_warm or use_hot) else None,
                                   parent_root=SRC if self._warm else None,
                                   swaps=list(plan.swaps.values()) if use_hot else None, keep=job.keep,
                                   check=check, loops=job.loops, phased=job.phased)
            if use_hot and res is not None and "unswappable" in res:
                # Not replaceable in place after all: import the patched tree afresh.
                info["unswappable"] = res["unswappable"]
                info["warm"] = False
                res = verify.run_tests(overlay, HARNESS, job.groups, timeout=timeout, drops=job.drops,
                                       stop_after=job.stop, parent_root=SRC, keep=job.keep, check=check,
                                       loops=job.loops, phased=job.phased)
        finally:
            verify.remove_overlay(overlay)
        c1 = os.times()
        info["verify_cpu"] = round((c1.children_user - c0.children_user)
                                   + (c1.children_system - c0.children_system), 3)
        info["verify_wall"] = round(time.monotonic() - w0, 3)
        if res is None:
            # The patched code hung or killed the runner: nothing was settled, so every
            # candidate is selected (the test that hangs is one of them).
            info["decision"] = "blind(hang)"
            return best_split(job.unknown_outcome)[1], info
        return self.conclude(job, res, plan, info), info

    def handle_with_state(self, plan, files, overlay: str, info: dict, base: dict):
        """One child imports the patched tree, compares what the import computed with the
        pinned tree's fingerprint, derives the candidates from the difference and re-runs
        them right there."""
        from . import state as st
        from . import statecands

        dm = self.dm
        static_ids = [dm.tests[i] for i in dm.ids(plan.candidates)]
        p0 = time.process_time()
        if self.needs_heavy(static_ids):
            self.load_heavy()
        self.t_imports += time.process_time() - p0
        modules = set(plan.import_modules)
        info["state"] = True
        loop_ok = self.loop_ready(plan)
        if self.probes is not None:
            self.probes.dm.idents()   # load before forking: the child may need it

        # What the child leaves behind before it starts re-running tests: the selection to
        # use if that re-run hangs or dies (every candidate, as on the other path).
        word = overlay + ".hang"

        def in_child():
            import signal

            # Each stage has its own limit; past it the child is killed (the default action
            # of the alarm signal) and the parent finds no result. Importing a patched
            # tree that hangs means no test can run: nothing is left behind, and every
            # test is selected.
            signal.alarm(STATE_IMPORT_LIMIT_S)
            verify.enter_patched_world(overlay, HARNESS, None, SRC if self._warm else None)
            scope = st.dependents(base["imports"], modules)
            names = [m for m in base["modules"] if m in scope]
            failed = st.import_all(names)
            current = st.snapshot(only=scope)
            paths = st.diff(base["state"], current, base.get("unstable", ()), only=scope)
            added = {p for p in paths if p not in base["state"]}
            removed = {p for p in paths if p not in current}
            notes: list[str] = [f"state: {len(scope)} modules compared, {len(paths)} definitions differ"]
            touches_expressions = any(p.startswith("sqlglot.expressions") for p in paths)
            memo: dict[int, int | None] = {}

            def settle(pl, some_dm):
                key = id(some_dm)
                if key not in memo:
                    is_main = some_dm is dm
                    bits = statecands.from_paths(some_dm, paths, added, failed, base,
                                                 self.cheap if is_main else (lambda b: False),
                                                 notes if is_main else [], removed=removed,
                                                 use_reads=bool(USE_READS))
                    if bits is not None and is_main and touches_expressions:
                        bits |= plan.import_words
                    memo[key] = bits
                pl.settle_imports(memo[key])

            settle(plan, dm)
            info["notes"] = plan.notes + notes
            loops = None
            if loop_ok and not failed and not added and not removed:
                # (the modules in which the import computed something else count as changed)
                loops, _ = self.with_loop_slice(plan, {p.split("|", 1)[0] for p in paths})
            elif REPORT and USE_LOOPS:
                info["loop_why"] = [bool(loop_ok), len(failed), len(added), len(removed), self._loop_failed]
            job = self.decide(plan, files, info, settle)
            if loops is not None:
                info["loop"] = len(loops)
            if job.final is not None:
                signal.alarm(0)
                return {"workers": job.final, "info": info}
            try:
                with open(word, "w") as f:
                    json.dump({"workers": best_split(job.unknown_outcome)[1],
                               "info": dict(info, decision="blind(hang)")}, f)
                signal.alarm(int(max(12.0, 6 * job.est_max + 8)) + 1)
            except OSError:
                signal.alarm(0)     # no word left: the parent's limit applies, and it selects everything
            res = verify.run_groups(job.groups, job.drops, job.stop, job.keep,
                                    loops={LOOP_TEST: loops} if loops is not None else None, phased=job.phased)
            signal.alarm(0)
            return {"workers": self.conclude(job, res, plan, info), "info": info}

        verify.remove_overlay(overlay)
        c0 = os.times()
        w0 = time.monotonic()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            elapsed = time.monotonic() - self.t0
            timeout = min(60.0, max(5.0, self.wall_budget - elapsed))
            out = verify.fork_call(in_child, timeout)
            if out is None:
                try:
                    with open(word) as f:
                        out = json.load(f)
                except (OSError, ValueError):
                    out = None
        finally:
            verify.remove_overlay(overlay)
            try:
                os.unlink(word)
            except OSError:
                pass
        c1 = os.times()
        cpu = round((c1.children_user - c0.children_user) + (c1.children_system - c0.children_system), 3)
        if out is None:
            info["decision"] = "all(hang)"
            info["verify_cpu"] = cpu
            return self.all_tests(), info
        child_info = out["info"]
        child_info["verify_cpu"] = cpu
        child_info["verify_wall"] = round(time.monotonic() - w0, 3)
        return out["workers"], child_info

    def run(self):
        out = sys.stdout
        try:
            self.prepare()
        except Exception as e:  # noqa: BLE001 - planning is repeated commit by commit below
            log(f"prepare failed: {type(e).__name__}: {e}")
            self._plans = {}
        for k, path in enumerate(self.commits):
            w0 = time.monotonic()
            try:
                workers, info = self.handle(path, k)
            except Exception as e:  # noqa: BLE001 - never lose the whole batch to one commit
                workers, info = self.all_tests(), {"commit": path.stem,
                                                   "decision": f"all(error {type(e).__name__}: {e})"}
            out.write(json.dumps({"commit": path.stem, "workers": workers}, separators=(",", ":")) + "\n")
            out.flush()
            info["wall"] = round(time.monotonic() - w0, 3)
            info["selected"] = sum(len(w) for w in workers)
            release_pin_if_slow(self.t0)
            log(json.dumps(info))
            if self.report:
                self.report.write(json.dumps(info) + "\n")
                self.report.flush()
        t = os.times()
        log(f"done {len(self.commits)} commits wall={time.monotonic()-self.t0:.1f}s "
            f"cpu={t.user+t.system+t.children_user+t.children_system:.2f}s "
            f"(own {t.user+t.system:.2f}s, children {t.children_user+t.children_system:.2f}s; "
            f"startup {self.t_startup:.2f}s, planning {self.t_plan:.2f}s, imports {self.t_imports:.2f}s)")
        if self.report:
            self.report.close()


_ALL_CPUS: set | None = None       # the CPUs this process may use, while it keeps to one of them


def pin_to_one_cpu():
    """Everything here runs one process after the other, so a second CPU buys nothing; staying
    on one spares migrations and the helper threads that libraries start per CPU."""
    global _ALL_CPUS
    try:
        cpus = os.sched_getaffinity(0)
        if len(cpus) > 1:
            os.sched_setaffinity(0, {max(cpus)})
            _ALL_CPUS = set(cpus)
    except Exception:  # noqa: BLE001 - not available: run as before
        _ALL_CPUS = None


def release_pin_if_slow(t0: float):
    """Give the other CPUs back when the clock runs well ahead of the CPU time: the one CPU
    is then shared with something else, and waiting for it could cost the time limit."""
    global _ALL_CPUS
    if _ALL_CPUS is None:
        return
    t = os.times()
    cpu = t.user + t.system + t.children_user + t.children_system
    wall = time.monotonic() - t0
    if wall > 1.5 * cpu + 3.0:
        try:
            os.sched_setaffinity(0, _ALL_CPUS)
        except Exception:  # noqa: BLE001
            pass
        _ALL_CPUS = None
        log(f"one CPU given up: wall {wall:.1f}s, cpu {cpu:.1f}s")


def main():
    if PIN:
        pin_to_one_cpu()
    Selector().run()


if __name__ == "__main__":
    main()
