"""Blast radius selector: for every patch, print the tests to run and how to split them.

For each commit:
  1. apply the patch in memory and diff it structurally against the pinned tree;
  2. look the changed code / files up in maps built from the pinned commit to get
     the tests that could be affected. For module- and class-level changes, import the
     patched tree in a child, fingerprint what the import computes and compare it with
     the pinned tree's fingerprint: the definitions that really changed give the tests;
  3. re-run the affordable candidates against the patched tree and keep those that fail.
     Fixture-driven tests are re-run on the affected fixture entries only. Tests that
     stay too expensive are selected unless the evidence says the change is harmless.
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


VERIFY_RATIO = _f("BR_VERIFY_RATIO", 0.6)    # re-run when estimated CPU <= ratio * cost of selecting blindly
VERIFY_FLOOR_S = _f("BR_VERIFY_FLOOR", 1.0)  # ... or when it is this cheap anyway
VERIFY_CAP_S = _f("BR_VERIFY_CAP", 12.0)     # ... and never more than this per commit
EXPENSIVE_S = _f("BR_EXPENSIVE", 1.0)        # tests costing more CPU than this are not re-run whole ...
WHOLE_RATIO = _f("BR_WHOLE_RATIO", 0.6)      # ... unless that is under this share of selecting them blindly
SLICE_MIN_S = _f("BR_SLICE_MIN", 0.25)       # fixture-driven tests cheaper than this are just re-run whole
SLICE_BUDGET_S = _f("BR_SLICE_BUDGET", 6.0)  # most CPU spent re-running affected entries of one test
SLICE_RATIO = _f("BR_SLICE_RATIO", 0.8)      # ... and at most this share of what selecting the test costs
SAMPLE_BUDGET_S = _f("BR_SAMPLE_BUDGET", 0.5)  # ... and when that is exceeded, CPU spent on a sample of them
BIG_SLICE_S = _f("BR_BIG_SLICE", 12.0)       # CPU cap for settling an expensive test that would otherwise be selected
BIG_SLICE_RATIO = _f("BR_BIG_SLICE_RATIO", 0.5)  # ... and at most this share of what selecting it costs
MIN_EVIDENCE = int(_f("BR_MIN_EVIDENCE", 8))  # passing re-runs needed before an unverified test is skipped
SKIP_HARD = int(_f("BR_SKIP_HARD", 1))        # 0: always select tests that could not be re-run
SKIP_MIN_MS = _f("BR_SKIP_MIN_MS", 0)         # ... and never leave out one cheaper than this to select
LIGHT_SETTLES = int(_f("BR_LIGHT_SETTLES", 0))  # 1: a passing light run settles the test as passing
EARLY_STOP = int(_f("BR_EARLY_STOP", 5))     # stop re-running once the first k tests have all failed
FIRST_FAIL_MARGIN_MS = _f("BR_FIRST_FAIL_MARGIN_MS", 2500)
WARM = int(_f("BR_WARM", 1))                 # reuse pinned modules for patches that change no Python code
USE_STATE = int(_f("BR_USE_STATE", 1))       # measure what import-time changes alter (state.py)
STATE_MIN_S = _f("BR_STATE_MIN", 1.2)        # ... when re-running the static candidates costs more than this
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
        self.whole = 0
        self.slices: dict = {}
        self.lights: dict = {}
        self.light_bits = 0
        self.hard = 0
        self.hard_dominates = False
        self.est = 0.0
        self.groups: list = []
        self.drops: dict = {}
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
        # tests known not to use their class-level fixtures (map index)
        self.no_setup = {self.dm.index[t] for t in self.dm.extra.get("no_class_setup", ())
                         if t in self.dm.index}
        # tests that read fixture files through the loaders: test index -> [file]
        self.fixture_files: dict[int, list[str]] = defaultdict(list)
        for key, users in self.dm.fixtures.items():
            name = key.split(":", 1)[1]
            for i in users:
                self.fixture_files[int(i)].append(name)
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
    def state(self) -> dict | None:
        """The pinned tree's modules and import graph (tools/build_state.py) plus, under
        "state", the fingerprint of what importing them computes. The fingerprint is taken
        here, once, in a child forked like the ones that later import patched trees, so
        both see the same interpreter state (hash seed, loaded libraries)."""
        if self._state is None and not self._state_failed:
            try:
                with open(STATE, "rb") as f:
                    data = json.loads(zlib.decompress(f.read()))
                import unittest  # noqa: F401 - loaded in every child, before sqlglot is

                from . import state as st

                def in_child():
                    for name in [m for m in sys.modules
                                 if m in ("sqlglot", "tests") or m.startswith(("sqlglot.", "tests."))]:
                        del sys.modules[name]
                    sys.path[:] = [SRC] + [p for p in sys.path if p != SRC]
                    failed = st.import_all(data["modules"])
                    return {"failed": failed, "state": st.snapshot()}

                c0 = os.times()
                res = verify.fork_call(in_child, 60.0)
                c1 = os.times()
                if not res or res["failed"]:
                    raise ValueError(f"pinned tree did not import cleanly: {res and res['failed']}")
                data["state"] = res["state"]
                self._state = data
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

    def entry_dependent(self, i: int) -> bool:
        """Does what this test runs depend on the entries of a fixture file it loads?"""
        if i not in self.fixture_files:
            return False
        if self.probes is None:
            return True
        info = self.probes.by_test.get(self.dm.tests[i])
        return info is None or bool(info["files"])

    # -- slicing -----------------------------------------------------------
    def slice_plan(self, plan, files, cand: int, settle):
        """Decide how each candidate is re-run.

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
            return dm.cpu[i] <= max(EXPENSIVE_S, WHOLE_RATIO * (rec + 2.0))

        def sliceable(i: int) -> bool:
            # A fixture-driven test costly enough for a partial re-run to be worth planning.
            return (self.probes is not None and dm.cpu[i] > SLICE_MIN_S
                    and bool(self.probes.entry_files(dm.tests[i])))

        todo = [i for i in dm.ids(cand) if not (sliced_only >> i) & 1]
        pcand = None
        if self.probes is not None and any(not affordable(i) or sliceable(i) for i in todo):
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
            can_slice = pcand is not None and sliceable(i)
            if affordable(i) and not can_slice:
                whole |= 1 << i
                continue
            info = self.probes.by_test.get(tid) if (self.probes is not None and pcand is not None) else None
            if not info or not info["files"]:
                spec = light_specs.get(tid)
                if spec and tid.split("::", 1)[0] not in plan.changed and spec["light_cpu"] <= SLICE_BUDGET_S:
                    lights[i] = spec   # re-run without its dominant statement(s)
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
            # Settling the test by re-running every affected entry must be cheaper than
            # simply selecting it.
            n = self.num[i]
            rec = self.ms[n] / 1000 if n is not None else 0.0
            complete = total <= min(SLICE_BUDGET_S, max(0.5, SLICE_RATIO * (rec + 2.0)))
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
                slices[i] = (flt, total + 0.05, True, plan.all_call_time)
                continue
            if not complete and (not plan.all_call_time or setup >= 1.0):
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
                job.final = best_split(self.durations(plan.blind))[1]
            else:
                info["decision"] = "none"
                job.final = []
            return job
        blind = self.durations(cand | plan.blind)
        for n in self.unknown:
            blind[n] = self.ms[n]
        job.blind = blind
        job.blind_ms = blind_ms = split_cost(blind)

        whole, slices, hard, lights = self.slice_plan(plan, files, cand, settle)
        budget = min(VERIFY_CAP_S, max(VERIFY_FLOOR_S, VERIFY_RATIO * blind_ms / 1000))

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
        info.update(n_cand=cand.bit_count(), n_whole=whole.bit_count(), n_sliced=len(slices),
                    n_light=len(lights), n_hard=hard.bit_count(), est=round(est, 3), blind=blind_ms / 1000)
        if REPORT:
            info["hard"] = [dm.tests[i].split("::")[-1] for i in dm.ids(hard)][:8]
            info["slices"] = {dm.tests[i].split("::")[-1]: [{f: len(ks) for f, ks in flt.items() if ks},
                                                            round(c, 2), ok, "if-failed" if cond else ""]
                              for i, (flt, c, ok, cond) in list(slices.items())[:8]}
        elapsed = time.monotonic() - self.t0
        # Do the tests whose outcome the re-run cannot settle (not re-run, sampled, or run
        # without their dominant statement) already cost about as much as selecting everything?
        open_bits = hard
        for i, s in slices.items():
            if not s[2]:
                open_bits |= 1 << i
        if not LIGHT_SETTLES:
            open_bits |= light_bits
        hard_dominates = bool(open_bits) and (
            split_cost(self.durations(open_bits)) >= blind_ms - FIRST_FAIL_MARGIN_MS)
        may_skip_hard = bool(SKIP_HARD) and plan.all_call_time
        can_verify = (
            MODE == "verify"
            and (whole or slices or lights)
            and est <= budget * 1.2
            and elapsed + est * 2 + 1 < self.wall_budget
            # ... if so, re-running the rest only pays off when it can clear them.
            and not (hard_dominates and not may_skip_hard)
        )
        if not can_verify:
            info["decision"] = "blind"
            job.final = best_split(blind)[1]
            return job
        job.whole, job.slices, job.lights, job.light_bits, job.hard = whole, slices, lights, light_bits, hard
        job.hard_dominates, job.est = hard_dominates, est
        job.groups = self.groups(whole | light_bits, slices, plan.tokens, plan.changed)
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
        if not LIGHT_SETTLES:
            unsettled |= job.light_bits & ~failed_bits
        selected = failed_bits | plan.blind
        decision = "verified"
        if unsettled:
            passed = (ran & ~failed_bits).bit_count()
            harmless = (SKIP_HARD and plan.all_call_time and not failed_bits and passed >= MIN_EVIDENCE)
            # Leaving a test out on evidence alone risks a miss; SKIP_MIN_MS can restrict
            # that to tests that are expensive to select.
            costly = 0
            if harmless:
                for i in dm.ids(unsettled):
                    n = self.num[i]
                    if n is not None and self.ms[n] >= SKIP_MIN_MS:
                        costly |= 1 << i
            selected |= unsettled & ~costly
            if costly:
                decision = "verified,skip-unsettled"
                info["skipped"] = [dm.tests[i].split("::")[-1] for i in dm.ids(costly)][:8]
            if unsettled & ~costly:
                decision += "+unsettled"
        out = self.durations(selected)
        if out:
            for n in self.unknown:
                out[n] = self.ms[n]
        info["decision"] = decision
        return best_split(out)[1]

    def handle(self, path: Path, k: int) -> tuple[list[list[int]], dict]:
        info: dict = {"commit": path.stem}
        p0 = time.process_time()
        text = path.read_text(encoding="utf-8", errors="replace")
        files = parse_patch(text)
        plan = make_plan(self.dm, SRC, files, self.cheap, entry_users=self.entry_users)
        self.t_plan += time.process_time() - p0
        info["notes"] = plan.notes
        if plan.select_all:
            info["decision"] = "all"
            return self.all_tests(), info
        dm = self.dm
        overlay = os.path.join(TMP, f"br_ov_{os.getpid()}_{k}")

        # Module / class level changes: measure what they alter instead of guessing,
        # unless the guess is cheap to re-run anyway.
        if (USE_STATE and MODE == "verify" and plan.import_static
                and self.estimate(plan.candidates & ~plan.blind) > STATE_MIN_S
                and self.state is not None):
            return self.handle_with_state(plan, files, overlay, info)

        def settle(pl, _dm):
            pl.settle_imports(None)

        settle(plan, dm)
        job = self.decide(plan, files, info, settle)
        if job.final is not None:
            return job.final, info
        p0 = time.process_time()
        if self.needs_heavy(job.group_ids):
            self.load_heavy()
        # Patches that leave the Python code alone are re-run on the modules this process
        # has already imported from the pinned tree.
        if WARM and self._warm is None:
            self._warm = verify.warm_up(SRC, HARNESS)
        use_warm = bool(self._warm) and plan.warm_ok
        if use_warm:
            verify.warm_tests(job.group_ids)
        info["warm"] = use_warm
        self.t_imports += time.process_time() - p0
        verify.remove_overlay(overlay)
        c0 = os.times()
        w0 = time.monotonic()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            elapsed = time.monotonic() - self.t0
            timeout = min(max(12.0, 6 * job.est + 8), max(5.0, self.wall_budget - elapsed))
            res = verify.run_tests(overlay, HARNESS, job.groups, timeout=timeout, drops=job.drops,
                                   stop_after=job.stop, warm_root=SRC if use_warm else None,
                                   parent_root=SRC if self._warm else None)
        finally:
            verify.remove_overlay(overlay)
        c1 = os.times()
        info["verify_cpu"] = round((c1.children_user - c0.children_user)
                                   + (c1.children_system - c0.children_system), 3)
        info["verify_wall"] = round(time.monotonic() - w0, 3)
        if res is None:
            # The patched code hung or killed the runner: assume the worst.
            info["decision"] = "all(hang)"
            return self.all_tests(), info
        return self.conclude(job, res, plan, info), info

    def handle_with_state(self, plan, files, overlay: str, info: dict):
        """One child imports the patched tree, compares what the import computed with the
        pinned tree's fingerprint, derives the candidates from the difference and re-runs
        them right there."""
        from . import state as st
        from . import statecands

        base = self.state
        dm = self.dm
        static_ids = [dm.tests[i] for i in dm.ids(plan.candidates)]
        p0 = time.process_time()
        if self.needs_heavy(static_ids):
            self.load_heavy()
        self.t_imports += time.process_time() - p0
        modules = set(plan.import_modules)
        info["state"] = True
        if self.probes is not None:
            self.probes.dm.idents()   # load before forking: the child may need it

        def in_child():
            verify.enter_patched_world(overlay, HARNESS, None, SRC if self._warm else None)
            scope = st.dependents(base["imports"], modules)
            names = [m for m in base["modules"] if m in scope]
            failed = st.import_all(names)
            current = st.snapshot(only=scope)
            paths = st.diff(base["state"], current, base.get("unstable", ()), only=scope)
            added = {p for p in paths if p not in base["state"]}
            notes: list[str] = [f"state: {len(scope)} modules compared, {len(paths)} definitions differ"]
            touches_expressions = any(p.startswith("sqlglot.expressions") for p in paths)
            memo: dict[int, int | None] = {}

            def settle(pl, some_dm):
                key = id(some_dm)
                if key not in memo:
                    is_main = some_dm is dm
                    bits = statecands.from_paths(some_dm, paths, added, failed, base,
                                                 self.cheap if is_main else (lambda b: False),
                                                 notes if is_main else [])
                    if bits is not None and is_main and touches_expressions:
                        bits |= plan.import_words
                    memo[key] = bits
                pl.settle_imports(memo[key])

            settle(plan, dm)
            info["notes"] = plan.notes + notes
            job = self.decide(plan, files, info, settle)
            if job.final is not None:
                return {"workers": job.final, "info": info}
            res = verify.run_groups(job.groups, job.drops, job.stop)
            return {"workers": self.conclude(job, res, plan, info), "info": info}

        verify.remove_overlay(overlay)
        c0 = os.times()
        w0 = time.monotonic()
        try:
            verify.build_overlay(SRC, overlay, plan.changed)
            elapsed = time.monotonic() - self.t0
            timeout = min(60.0, max(5.0, self.wall_budget - elapsed))
            out = verify.fork_call(in_child, timeout)
        finally:
            verify.remove_overlay(overlay)
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


def main():
    Selector().run()


if __name__ == "__main__":
    main()
