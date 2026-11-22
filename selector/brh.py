"""Shared harness for running sqlglot's unittest tests in-process.

Used both when maps are built from the pinned commit and when the selector
re-runs tests against a patched tree, so the two always execute tests the
same way:

  * process pools are replaced by an inline executor that still pickles the
    call and its result, as crossing a process boundary would;
  * fixture loaders can be restricted to chosen entries ("fixture slicing").
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import Future


# The test that is running (None between tests, e.g. in class fixtures).
CURRENT: str | None = None
# Calls submitted to a pool that outlives the test making them (one made in a class
# fixture): {(test id, file of the submit call, line)}. Filled on every run; the map
# builder (tools/tracer/pools.py) turns it into "which tests share a pool".
SHARED_SUBMITS: set = set()
# Tests during which a shared pool would have broken (see InlinePool.submit).
POOL_BREAKS: list = []
# Test files that could not be imported when one of their tests was asked for: {file: reason}.
BAD_FILES: dict = {}
# A profile function that is active while a pool pickles or unpickles (map building only:
# tools/tracer/pools.py records with it which code runs then).
PICKLE_TRACE = None


def _pickling(step, data):
    if PICKLE_TRACE is None:
        return step(data)
    sys.setprofile(PICKLE_TRACE)
    try:
        return step(data)
    finally:
        sys.setprofile(None)


class InlinePool:
    """Drop-in for ProcessPoolExecutor that runs submitted calls immediately.

    A real pool breaks for good when a worker dies, and a worker dies when it cannot
    unpickle the call it was sent; the pool also breaks when the parent cannot unpickle a
    result or the exception a call raised. Every later test using the same pool then fails
    with BrokenProcessPool, whatever it tests. Here the call fails the same way, and the
    event is recorded (POOL_BREAKS) when the pool is shared between tests, so that the
    selector can add the other tests using it. The pool itself keeps working: each test
    re-run here is judged on its own, in whatever order."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_):
        self._initializer = initializer
        self._initargs = initargs
        self._ready = False
        self._owner = CURRENT       # None: made outside any test, so shared between tests

    def _site(self):
        f = sys._getframe(2)
        while f is not None and f.f_code.co_filename == __file__:
            f = f.f_back
        return (f.f_code.co_filename, f.f_lineno) if f is not None else ("?", 0)

    def _broken(self, fut, shared):
        from concurrent.futures.process import BrokenProcessPool

        if shared and CURRENT is not None and CURRENT not in POOL_BREAKS:
            POOL_BREAKS.append(CURRENT)
        fut.set_exception(BrokenProcessPool(
            "A process in the process pool was terminated abruptly while the future was running or pending."))
        return fut

    def submit(self, fn, /, *args, **kwargs):
        fut: Future = Future()
        shared = self._owner is None or self._owner != CURRENT
        if shared and CURRENT is not None:
            SHARED_SUBMITS.add((CURRENT,) + self._site())
        try:
            if not self._ready:
                self._ready = True
                if self._initializer is not None:
                    self._initializer(*self._initargs)
            call = _pickling(pickle.dumps, (fn, args, kwargs))
        except Exception as e:  # noqa: BLE001 - the call never leaves: only this future fails
            fut.set_exception(e)
            return fut
        try:
            fn, args, kwargs = _pickling(pickle.loads, call)
        except Exception:  # noqa: BLE001 - the worker dies reading the call
            return self._broken(fut, shared)
        try:
            result = fn(*args, **kwargs)
            result = _pickling(pickle.dumps, result)
        except Exception as e:  # noqa: BLE001 - a worker sends it back as the call's outcome
            # ... pickled like a result: what cannot be pickled is replaced by the error
            # saying so, and what the parent cannot unpickle breaks the pool.
            try:
                sent = _pickling(pickle.dumps, e)
            except Exception as e2:  # noqa: BLE001
                fut.set_exception(e2)
                return fut
            try:
                fut.set_exception(_pickling(pickle.loads, sent))
            except Exception:  # noqa: BLE001
                return self._broken(fut, shared)
            return fut
        try:
            fut.set_result(_pickling(pickle.loads, result))
        except Exception:  # noqa: BLE001 - the parent cannot read the result
            return self._broken(fut, shared)
        return fut

    def map(self, fn, *iterables, **_):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, wait=True, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# filename (as passed to the loaders) -> set of 0-based entry indices to keep.
# A missing key keeps every entry.
FIXTURE_FILTER: dict[str, set[int]] = {}
# filename -> number of entries seen on the last full read (filled when tracing).
FIXTURE_COUNTS: dict[str, int] = {}


def _wrap_loader(orig, kind):
    def loader(filename):
        keep = FIXTURE_FILTER.get(filename)
        n = 0
        for i, item in enumerate(orig(filename)):
            n = i + 1
            if keep is None or i in keep:
                yield item
        FIXTURE_COUNTS[f"{kind}:{filename}"] = n

    loader.__name__ = orig.__name__
    loader.__wrapped__ = orig
    return loader


def install(inline_pools: bool = True, fixture_hooks: bool = True):
    """Call after sys.path points at the tree and before test modules are imported."""
    if inline_pools:
        import concurrent.futures as cf

        cf.ProcessPoolExecutor = InlinePool  # type: ignore[misc]
        import concurrent.futures.process as cfp

        cfp.ProcessPoolExecutor = InlinePool  # type: ignore[misc]
    if fixture_hooks:
        import tests.helpers as helpers

        if not hasattr(helpers.load_sql_fixtures, "__wrapped__"):
            helpers.load_sql_fixtures = _wrap_loader(helpers.load_sql_fixtures, "singles")
            helpers.load_sql_fixture_pairs = _wrap_loader(helpers.load_sql_fixture_pairs, "pairs")


# test id -> [(first line, last line)] of top-level statements of the test method to leave
# out ("light" run of an expensive test: everything but its dominant statement).
DROP_STATEMENTS: dict[str, list] = {}
# tests that were asked for but could not be run the way they were asked
UNVERIFIED: list[str] = []


def _light_method(case, meth: str, spans):
    """The test method recompiled without the statements at the given file line spans."""
    import ast
    import inspect

    func = getattr(type(case), meth)
    if inspect.unwrap(func) is not func or not inspect.isfunction(func):
        return None
    code = func.__code__
    fdef = find_function(code.co_filename, meth, code.co_firstlineno)
    if fdef is None or fdef.decorator_list:
        return None
    wanted = {tuple(s) for s in spans}
    kept = [s for s in fdef.body if (s.lineno, s.end_lineno) not in wanted]
    if len(fdef.body) - len(kept) != len(wanted):
        return None  # the method no longer looks like the one the spans were computed for
    fdef.body = kept or [ast.Pass()]
    module = ast.Module(body=[fdef], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {}
    exec(compile(module, code.co_filename, "exec"), func.__globals__, namespace)
    return namespace[fdef.name]


# test id -> [[first line, last line], ...]: the statements of the test method to run, in
# file order (the others are left out). Used to re-run only the parts of a long test that
# a change can reach; see tools/tracer/stmt_spec.py for which parts may be left out.
KEEP_STATEMENTS: dict[str, list] = {}
_SOURCE_LINES: dict[str, list[str]] = {}


def _kept_method(case, meth: str, spans):
    """The test method rebuilt from the given statement line spans of its source file."""
    import inspect

    func = getattr(type(case), meth)
    if inspect.unwrap(func) is not func or not inspect.isfunction(func):
        return None
    code = func.__code__
    lines = _SOURCE_LINES.get(code.co_filename)
    if lines is None:
        with open(code.co_filename, encoding="utf-8") as f:
            lines = _SOURCE_LINES[code.co_filename] = f.read().split("\n")
    body: list[str] = []
    for lo, hi in spans:
        if lo <= code.co_firstlineno or hi > len(lines):
            return None
        body.extend(lines[lo - 1:hi])
    if not body:
        return None
    indent = len(body[0]) - len(body[0].lstrip(" "))
    if indent < 8 or indent % 4:
        return None
    # The owning class is named in the stand-in so that private names are mangled alike.
    owner = func.__qualname__.split(".")[-2] if "." in func.__qualname__ else type(case).__name__
    pad = indent - 8
    head = "".join(" " * (4 * d) + "if 1:\n" for d in range(pad // 4))
    head += " " * pad + f"class {owner}:\n" + " " * (pad + 4) + f"def {meth}(self):\n"
    namespace: dict = {}
    exec(compile(head + "\n".join(body) + "\n", code.co_filename, "exec"), func.__globals__, namespace)
    return namespace[owner].__dict__[meth]


# test id -> [(old text, new text)]: a scaled-down stand-in for an expensive test, made by
# replacing these pieces of the test method's source (each has to occur exactly once). What
# such a run shows is evidence about the real test, not its verdict (see main.evidence_probe).
SCALE_TESTS: dict[str, list] = {}


def _scaled_method(case, meth: str, replacements):
    """The test method recompiled from its source with the given replacements."""
    import inspect
    import textwrap

    func = getattr(type(case), meth)
    if inspect.unwrap(func) is not func or not inspect.isfunction(func):
        return None
    code = func.__code__
    fdef = find_function(code.co_filename, meth, code.co_firstlineno)
    if fdef is None or fdef.decorator_list:
        return None
    lines = _SOURCE_LINES.get(code.co_filename)
    if lines is None:
        with open(code.co_filename, encoding="utf-8") as f:
            lines = _SOURCE_LINES[code.co_filename] = f.read().split("\n")
    src = "\n".join(lines[fdef.lineno - 1:fdef.end_lineno]) + "\n"
    for old, new in replacements:
        if src.count(old) != 1:
            return None     # the method no longer looks like the one the stand-in was written for
        src = src.replace(old, new)
    owner = func.__qualname__.split(".")[-2] if "." in func.__qualname__ else type(case).__name__
    body = textwrap.indent(textwrap.dedent(src), "    ")
    namespace: dict = {}
    exec(compile(f"class {owner}:\n{body}", code.co_filename, "exec"), func.__globals__, namespace)
    return namespace[owner].__dict__[meth]


def find_function(filename: str, name: str, first_line: int):
    """The ast.FunctionDef called `name` that starts at `first_line` of the file."""
    import ast

    with open(filename, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name and node.lineno == first_line:
            return node
    return None


def load_case(test_id: str):
    """'tests/x.py::Class::method' -> unittest.TestCase instance."""
    import types

    file_part, cls_name, meth = test_id.split("::")
    mod_name = file_part[:-3].replace("/", ".")
    module = __import__(mod_name, fromlist=[cls_name])
    case = getattr(module, cls_name)(meth)
    scale = SCALE_TESTS.get(test_id)
    if scale:
        scaled = None
        try:
            scaled = _scaled_method(case, meth, scale)
        except Exception:  # noqa: BLE001
            scaled = None
        if scaled is None:
            raise LightUnavailable(test_id)
        setattr(case, meth, types.MethodType(scaled, case))
        return case
    keep = KEEP_STATEMENTS.get(test_id)
    if keep is not None:
        kept = None
        try:
            kept = _kept_method(case, meth, keep)
        except Exception:  # noqa: BLE001
            kept = None
        if kept is None:
            raise LightUnavailable(test_id)
        setattr(case, meth, types.MethodType(kept, case))
        return case
    spans = DROP_STATEMENTS.get(test_id)
    if spans:
        light = None
        try:
            light = _light_method(case, meth, spans)
        except Exception:  # noqa: BLE001
            light = None
        if light is None:
            raise LightUnavailable(test_id)
        setattr(case, meth, types.MethodType(light, case))
    only = LOOP_SLICE.get(test_id)
    if only is not None and test_id in LOOP_RECIPES:
        setattr(case, meth, types.MethodType(
            lambda self, _run=_sliced_loop(getattr(case, meth), LOOP_RECIPES[test_id], only): _run(self), case))
    spec = PHASED.get(test_id)
    if spec is not None and test_id in PHASE_RECIPES:
        setattr(case, meth, types.MethodType(
            lambda self, _run=_phased_method(getattr(case, meth), spec, PHASE_RECIPES[test_id], test_id): _run(self),
            case))
    return case


class LightUnavailable(Exception):
    pass


# -- a test that does the same thing once per dialect ---------------------------------------
# TestGenerator.test_generator_memory builds a thousand generators of every dialect and
# watches the memory. One round of its loop touches the classes and modules of one dialect
# family and the code they share; a patch outside what a round touches cannot change that
# round (main.Selector.loop_slice works out which rounds a patch can reach). The test is then
# run with the name its loop walks bound to those members only.
LOOP_RECIPES = {
    "tests/test_generator.py::TestGenerator::test_generator_memory": {"name": "Dialects"},
}
# test id -> values of the enumeration's members to keep
LOOP_SLICE: dict[str, list] = {}


def _sliced_loop(orig, recipe: dict, only):
    def run(self):
        module = sys.modules[type(self).__module__]
        whole = getattr(module, recipe["name"])
        keep = set(only)
        setattr(module, recipe["name"], [m for m in whole if m.value in keep])
        try:
            return orig()
        finally:
            setattr(module, recipe["name"], whole)

    return run


# -- settling the executor's TPC tests in two phases -------------------------------------------
# TestExecutor.test_execute_tpcds / test_execute_tpch do, for every query of a fixture file:
# parse it, send the expression to a pool worker, and there execute() it: optimize, plan, run
# the plan in the Python executor; back in the test the rows are compared with DuckDB's.
# Running the plans is most of the cost. For one query, everything after the optimizer sees
# the patched program through two values only, the parsed and the optimized expression. If both
# are what they are on the pinned commit, and the patch touches no line that the query runs
# outside parsing and optimizing (measured with nothing but the pinned optimized expression
# to start from: tools/tracer/phases.py), the rest is unchanged code on unchanged input and
# the query passes as it does on the pinned commit. Such queries are taken out; the test
# runs on the others, or not at all when none is left.
PHASE_RECIPES = {
    "tests/test_executor.py::TestExecutor::test_execute_tpcds":
        {"sqls": "tpcds_sqls", "tables": "tpcds_tables", "schema": "TPCDS_SCHEMA",
         "file": "optimizer/tpc-ds/tpc-ds.sql"},
    "tests/test_executor.py::TestExecutor::test_execute_tpch":
        {"sqls": "tpch_sqls", "tables": "tpch_tables", "schema": "TPCH_SCHEMA",
         "file": "optimizer/tpc-h/tpc-h.sql"},
}
# test id -> {"fp": {entry: [parsed, optimized]}, "eligible": [entries], "cost": {entry: cpu s},
#             "max_exec": cpu s that may be spent executing the entries that are left}
PHASED: dict[str, dict] = {}
# what the two-phase runs did: test id -> [entries looked at, entries left, entries executed]
PHASE_LOG: dict[str, list] = {}


class PlanReached(BaseException):
    """Raised in place of building the plan: the optimized expression has been seen."""


def expr_fingerprint(expression) -> str:
    """A digest of everything an expression tree holds (classes, arguments, types, comments, meta)."""
    import hashlib
    import json

    from sqlglot import serde

    return hashlib.sha1(json.dumps(serde.dump(expression), default=repr, sort_keys=True).encode()).hexdigest()[:20]


def executor_phase_one(te, sql: str, meta: dict, keep_optimized: list | None = None) -> list:
    """What one query of an executor TPC test looks like up to the plan: [digest of the parsed
    expression as the worker receives it, digest of the optimized expression (None for a query
    the test does not execute)]. sqlglot.executor.Plan must be the stub of `plan_stub`."""
    parsed = pickle.loads(pickle.dumps(te.parse_one(sql)))
    out = [expr_fingerprint(parsed), None]
    _PHASE_BOX.clear()
    try:
        te.mp_execute(parsed, meta)
    except PlanReached:
        out[1] = _PHASE_BOX.get("fp")
        if keep_optimized is not None:
            keep_optimized.append(_PHASE_BOX.get("expression"))
    return out


_PHASE_BOX: dict = {}


def plan_stub(expression):
    _PHASE_BOX["fp"] = expr_fingerprint(expression)
    _PHASE_BOX["expression"] = expression
    raise PlanReached()


class _NoRows(dict):
    """Stands in for the tables of the class fixture while queries are only parsed and
    optimized: every name is a table without rows (the optimizer is given the schema, not
    the tables)."""

    def __missing__(self, name):
        from sqlglot.executor.table import Table

        return Table()


def _phased_method(orig, spec: dict, recipe: dict, test_id: str):
    def run(self):
        import sqlglot.executor as ex

        cls = type(self)
        te = sys.modules[cls.__module__]
        built = recipe["sqls"] in cls.__dict__          # was the class fixture built?
        if built:
            sqls = getattr(cls, recipe["sqls"])
        else:
            # the queries as the fixture would load them (the loader honours FIXTURE_FILTER)
            sqls = list(sys.modules["tests.helpers"].load_sql_fixture_pairs(recipe["file"]))
        keep = FIXTURE_FILTER.get(recipe["file"])
        index = sorted(keep) if keep is not None else list(range(len(sqls)))
        if len(index) != len(sqls):
            UNVERIFIED.append(test_id)      # the fixture was not loaded the way it was asked for
            return
        eligible = set(spec["eligible"])
        cost = spec.get("cost", {})
        limit = spec.get("max_exec", 0.0)
        max_cpu = spec.get("max_cpu")
        max_differ = spec.get("max_differ", 6)
        t0 = _cpu_now()
        log = PHASE_LOG[test_id] = [0, 0, 0]    # looked at, different (or to be run anyway), run

        def must_run(pos) -> bool:
            return index[pos] not in eligible or spec["fp"].get(str(index[pos])) is None

        # -- the looks: no table is needed to parse and optimize a query
        todo = [pos for pos in range(len(sqls)) if must_run(pos)]
        owed = sum(cost.get(str(index[pos]), 0.5) for pos in todo)
        log[1] = len(todo)
        if owed > limit or len(todo) > max_differ:
            UNVERIFIED.append(test_id)
            return
        te.initializer(getattr(te, recipe["schema"]), getattr(cls, recipe["tables"]) if built else _NoRows())
        real_plan = ex.Plan
        ex.Plan = plan_stub
        try:
            for pos, (meta, sql, _) in enumerate(sqls):
                if must_run(pos):
                    continue
                if max_cpu is not None and _cpu_now() - t0 > max_cpu:
                    UNVERIFIED.append(test_id)  # far slower than on the pinned commit: not settled
                    return
                log[0] += 1
                try:
                    got = executor_phase_one(te, sql, meta)
                except BaseException as e:  # noqa: BLE001 - whatever it is, the real run will show it
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    got = None
                want = list(spec["fp"][str(index[pos])])
                if got == want or (want[0] == "*" and got is not None and got[1] is None):
                    continue                # does what it does on the pinned commit
                todo.append(pos)
                log[1] += 1
                owed += cost.get(str(index[pos]), 0.5)
                # When the optimizer gives the first queries, or many of them, another shape,
                # looking is no shortcut: the test stays unsettled (and selected).
                many = (log[1] >= 2 and 4 * log[1] > log[0]) or log[1] > max_differ
                if owed > limit or many:
                    UNVERIFIED.append(test_id)
                    return
        finally:
            ex.Plan = real_plan
        if not todo:
            return                          # every affected query does what it does on the pinned commit
        # -- the few queries that differ are run the way the test runs them, one at a time
        if not built:
            # (run_tests put a stand-in in place of the fixture this group asked to skip)
            real = cls.__dict__.get("_br_fixture")
            if real is not None and real[0] is not None:
                real[0].__get__(None, cls)()
            else:
                cls.setUpClass()
            full = getattr(cls, recipe["sqls"], None)
            if full is None:
                UNVERIFIED.append(test_id)
                return
            if len(full) != len(sqls):
                UNVERIFIED.append(test_id)
                return
            sqls = full
        te.initializer(getattr(te, recipe["schema"]), getattr(cls, recipe["tables"]))
        for pos in sorted(todo, key=lambda q: (cost.get(str(index[q]), 0.5), q)):
            if max_cpu is not None and _cpu_now() - t0 > max_cpu + limit:
                UNVERIFIED.append(test_id)
                return
            log[2] += 1
            setattr(cls, recipe["sqls"], [sqls[pos]])
            try:
                orig()
            finally:
                setattr(cls, recipe["sqls"], sqls)
            outcome = getattr(self, "_outcome", None)
            if outcome is not None and not getattr(outcome, "success", True):
                return                      # a query fails: so does the test

    return run


class _Result:
    """Minimal unittest result: a test fails when it or any of its subtests fails or errors."""

    def __init__(self, stop_after: int = 0):
        self.failed: dict[str, str] = {}
        self.started: set[str] = set()
        self.stop_after = stop_after

    def run(self, suite, ids):
        import unittest

        outer = self

        class R(unittest.TestResult):
            def startTest(self, test):
                global CURRENT
                tid = getattr(test, "_br_id", None)
                CURRENT = tid
                if tid is not None:
                    outer.started.add(tid)
                    if TIMES is not None:
                        TIMES[tid] = -_cpu_now()

            def stopTest(self, test):
                global CURRENT
                CURRENT = None
                if TIMES is not None:
                    tid = getattr(test, "_br_id", None)
                    if tid in TIMES and TIMES[tid] < 0:
                        TIMES[tid] += _cpu_now()
                # When the first few tests all fail the change is clearly live: stop and let
                # the caller treat the rest as failing rather than pay to confirm it.
                k = outer.stop_after
                if k < 0:
                    # Caller only needs to know whether anything fails.
                    if outer.failed:
                        self.shouldStop = True
                elif k and len(outer.started) == k and all(t in outer.failed for t in outer.started):
                    self.shouldStop = True

            def _mark(self, test, err):
                # Class / module level fixtures report an _ErrorHolder: blame every test asked for.
                tid = getattr(test, "_br_id", None)
                msg = "".join(str(x) for x in err[1:2])[-300:] if err else ""
                if tid is None:
                    desc = str(test)
                    for i in ids:
                        cls = i.split("::")[1]
                        if cls in desc:
                            outer.failed.setdefault(i, f"fixture: {msg}")
                else:
                    outer.failed.setdefault(tid, msg)

            def addError(self, test, err):
                self._mark(test, err)

            def addFailure(self, test, err):
                self._mark(test, err)

            def addSubTest(self, test, subtest, err):
                if err is not None:
                    self._mark(test, err)

            def addUnexpectedSuccess(self, test):
                self._mark(test, None)

        suite.run(R())
        return self.failed


# test id -> CPU seconds of its run, filled only when a dict is put here (analysis runs).
TIMES: dict | None = None


def _cpu_now() -> float:
    import time

    return time.process_time()


DUCKDB_THREADS = 2


def _cap_duckdb_threads() -> None:
    """DuckDB starts one thread per core of the host for every connection, whatever the
    container's CPU quota is. Under a process/thread cap that can fail on a large host,
    so connections the tests open without a config get a fixed, small thread count."""
    import sys

    duckdb = sys.modules.get("duckdb")
    if duckdb is None or getattr(duckdb.connect, "_br_capped", False):
        return
    orig = duckdb.connect

    def connect(*args, **kwargs):
        if len(args) < 3 and "config" not in kwargs:
            kwargs["config"] = {"threads": DUCKDB_THREADS}
        return orig(*args, **kwargs)

    connect._br_capped = True
    duckdb.connect = connect


NOT_RUN: list[str] = []
# "tests/x.py::Class" keys whose class-level fixtures are not built: every test asked for
# in such a class is known not to use them.
SKIP_CLASS_SETUP: set[str] = set()


def _no_fixture(cls):
    return None


def run_tests(test_ids: list[str], stop_after: int = 0) -> dict[str, str]:
    """Run the tests in the given order in this process; return {failing id: reason}.

    stop_after=k: stop when the first k tests have all failed (k < 0: at the first
    failure); the tests that were not run are appended to NOT_RUN."""
    import unittest

    failed: dict[str, str] = {}
    cases = []
    bad_files: dict[str, str] = {}
    for pos, tid in enumerate(test_ids):
        file_part = tid.split("::", 1)[0]
        if file_part in bad_files:
            failed[tid] = bad_files[file_part]
            continue
        if stop_after and pos >= abs(stop_after) and len(failed) == pos and not cases:
            # Nothing can even be loaded (e.g. a module no longer imports): stop trying.
            NOT_RUN.extend(test_ids[pos:])
            return failed
        try:
            case = load_case(tid)
        except LightUnavailable:
            UNVERIFIED.append(tid)
            continue
        except BaseException as e:  # noqa: BLE001 - un-importable file or missing test
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            reason = f"load: {type(e).__name__}: {e}"[:300]
            failed[tid] = reason
            if not isinstance(e, AttributeError):   # the file itself cannot be imported
                bad_files[file_part] = reason
                if file_part[:-3].replace("/", ".") not in sys.modules:
                    # ... for certain (a failed import leaves no module behind): then every
                    # test of the file fails, not only the ones asked for.
                    BAD_FILES[file_part] = reason
            continue
        case._br_id = tid
        klass = type(case)
        skip = tid.rsplit("::", 1)[0] in SKIP_CLASS_SETUP
        if skip and "_br_fixture" not in klass.__dict__:
            # Remember the real fixtures so a later group in this process can use them.
            klass._br_fixture = (klass.__dict__.get("setUpClass"), klass.__dict__.get("tearDownClass"))
            klass.setUpClass = classmethod(_no_fixture)
            klass.tearDownClass = classmethod(_no_fixture)
        elif not skip and "_br_fixture" in klass.__dict__:
            for name, orig in zip(("setUpClass", "tearDownClass"), klass._br_fixture):
                if orig is None:
                    delattr(klass, name)
                else:
                    setattr(klass, name, orig)
            del klass._br_fixture
        cases.append(case)
    _cap_duckdb_threads()
    suite = unittest.TestSuite(cases)
    result = _Result(stop_after)
    failed.update(result.run(suite, test_ids))
    if stop_after:
        for case in cases:
            if case._br_id not in result.started and case._br_id not in failed:
                NOT_RUN.append(case._br_id)
    return failed
