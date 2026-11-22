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
         "file": "optimizer/tpc-ds/tpc-ds.sql", "conn": "tpcds_conn"},
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
    """A digest of everything an expression tree holds (classes, arguments in their order,
    types, comments, meta). It is made without any code of the package: a patch must not be
    able to change what the digest sees."""
    import hashlib

    out: list = []
    try:
        _fp_walk(expression, out)
    except RecursionError:
        return "!" + hex(id(out))           # too deep to walk: equal to nothing
    return hashlib.sha1("\x00".join(out).encode("utf-8", "surrogatepass")).hexdigest()[:20]


def _fp_walk(node, out: list) -> None:
    import enum

    cls = type(node)
    slots = _fp_slots(cls)
    if slots is not None:
        out.append("(" + cls.__module__ + "." + cls.__qualname__)
        comments = _fp_get(node, "comments", slots)
        if comments:
            out.append("#" + repr(comments))
        meta = _fp_get(node, "_meta", slots)
        if meta is not None:
            out.append("@" + (repr(sorted(meta.items(), key=repr)) if type(meta) is dict else repr(meta)))
        typ = _fp_get(node, "_type", slots)
        if typ is not None and typ is not node:
            out.append("^")
            _fp_walk(typ, out)
        args = _fp_get(node, "args", slots)
        for k, v in (args.items() if type(args) is dict else ()):
            if v is None:
                continue
            out.append("." + str(k))
            if type(v) is list:
                out.append("[")
                for x in v:
                    _fp_walk(x, out)
                out.append("]")
            else:
                _fp_walk(v, out)
        out.append(")")
    elif isinstance(node, enum.Enum):
        out.append("E:" + cls.__qualname__ + "." + node.name)
    elif cls in (str, int, float, bool, bytes, type(None)):
        out.append(cls.__name__ + ":" + repr(node))
    else:
        out.append("?" + cls.__module__ + "." + cls.__qualname__ + ":" + repr(node))


_FP_SLOTS: dict = {}


def _fp_slots(cls):
    """The slot descriptors of an expression class (args, comments, _type, _meta), found in
    its MRO without calling anything the package defines; None for any other class."""
    got = _FP_SLOTS.get(cls, 0)
    if got == 0:
        found = {}
        for name in ("args", "comments", "_type", "_meta"):
            for k in cls.__mro__:
                d = k.__dict__.get(name)
                if d is not None and type(d).__name__ == "member_descriptor":
                    found[name] = d
                    break
        got = found if "args" in found and any(k.__dict__.get("parent") is not None for k in cls.__mro__) else None
        _FP_SLOTS[cls] = got
    return got


def _fp_get(node, name, slots):
    d = slots.get(name)
    if d is None:
        return None
    try:
        return d.__get__(node, type(node))
    except AttributeError:
        return None


def executor_phase_one(te, sql: str, meta: dict, keep_optimized: list | None = None) -> list:
    """What one query of an executor TPC test looks like up to the plan: [digest of the parsed
    expression as the worker receives it, digest of the optimized expression (None for a query
    the test does not execute)]. sqlglot.executor.Plan must be the stub of `plan_stub`.
    The parsed expression is pickled and unpickled first, as on its way to a worker of the
    test's pool: what the serialiser makes of it under the patch is part of what is compared.
    (The SQL text the test sends to DuckDB is not looked at: making it asks the dialect
    registry for a dialect by name, and what that does depends on which dialect modules the
    process has loaded, which is not the same here as in the test's own process.)"""
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


# -- DuckDB's reference answers, replayed ----------------------------------------------------
# The executor's TPC-DS test compares each query's rows with what DuckDB returns for the
# query's DuckDB text. That answer depends on the text and on the fixture tables, not on the
# patched package, and computing it is nearly half of the test's CPU. maps/duck.bin holds
# the frames DuckDB returned on the pinned commit (tools/build_duck.py), keyed by a digest of
# the text. The stand-in below answers a text it knows from the record and hands anything
# else to DuckDB, so a patch that changes the text a query is transpiled to is still
# answered by DuckDB itself. It is only put in place where the fixture is the pinned one
# (the two-phase runner: package code changed, no test code or data the test reads).
_DUCK: list = []


def duck_answers() -> dict:
    if not _DUCK:
        import os
        import pickle

        answers: dict = {}
        try:
            path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps", "duck.bin")
            with open(os.environ.get("BR_DUCK_MAP", path), "rb") as f:
                answers = pickle.load(f).get("answers") or {}
        except Exception:  # noqa: BLE001 - no record: DuckDB answers everything
            answers = {}
        _DUCK.append(answers)
    return _DUCK[0]


class DuckReplay:
    def __init__(self, real, answers: dict):
        self._real = real
        self._answers = answers

    def execute(self, query, *args, **kwargs):
        if not args and not kwargs and type(query) is str:
            import hashlib

            blob = self._answers.get(hashlib.sha1(query.encode("utf-8", "surrogatepass")).hexdigest())
            if blob is not None:
                return DuckAnswer(self._real, query, blob)
        return self._real.execute(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class DuckAnswer:
    """What execute() returned for a recorded text: fetchdf() gives a fresh copy of the
    recorded frame; anything else is asked of DuckDB after all."""

    def __init__(self, real, query: str, blob: bytes):
        self._real, self._query, self._blob = real, query, blob

    def fetchdf(self):
        import pickle
        import zlib

        return pickle.loads(zlib.decompress(self._blob))

    def __getattr__(self, name):
        return getattr(self._real.execute(self._query), name)


class PhaseTimeout(BaseException):
    """A query of a two-phase run used up its CPU allowance."""


def _cpu_timer():
    """timer(seconds) arms a CPU-time alarm that raises PhaseTimeout in this process,
    timer(0) disarms it, timer(None) puts the previous handler back. Does nothing where the
    interval timer is not available."""
    import signal

    state = {"old": None, "ok": True}

    def on_alarm(signum, frame):
        raise PhaseTimeout()

    try:
        state["old"] = signal.signal(signal.SIGVTALRM, on_alarm)
    except (AttributeError, ValueError, OSError):
        state["ok"] = False

    def timer(seconds):
        if not state["ok"]:
            return
        try:
            if seconds is None:
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
                signal.signal(signal.SIGVTALRM, state["old"] if state["old"] is not None else signal.SIG_DFL)
            else:
                signal.setitimer(signal.ITIMER_VIRTUAL, float(seconds))
        except (AttributeError, ValueError, OSError):
            state["ok"] = False

    return timer


def _replay_duck(cls, recipe: dict) -> None:
    name = recipe.get("conn")
    answers = duck_answers() if name else None
    if not answers:
        return
    conn = cls.__dict__.get(name)
    if conn is not None and not isinstance(conn, DuckReplay):
        setattr(cls, name, DuckReplay(conn, answers))


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
        look_cost = spec.get("look_cost", {})       # CPU seconds of each look on the pinned commit (the map's)
        slow = spec.get("max_slow", 8.0)            # (the map's seconds are those of a fast machine)
        looked_s = 0.0
        t0 = _cpu_now()
        log = PHASE_LOG[test_id] = [0, 0, 0]    # looked at, different (or to be run anyway), run

        def must_run(pos) -> bool:
            return index[pos] not in eligible or spec["fp"].get(str(index[pos])) is None

        # -- the looks: no table is needed to parse and optimize a query
        todo = [pos for pos in range(len(sqls)) if must_run(pos)]
        owed = sum(cost.get(str(index[pos]), 0.5) for pos in todo)
        log[1] = len(todo)
        owed_first = len(todo)
        max_rate = spec.get("max_rate", 0.0)
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
                if max_cpu is not None:
                    # Far slower than on the pinned commit (several times what the looks so
                    # far take there, once a few are done), or past the ceiling: not settled.
                    spent = _cpu_now() - t0
                    if spent > max_cpu or (log[0] >= 4 and spent > slow * looked_s + 1.0):
                        UNVERIFIED.append(test_id)
                        return
                log[0] += 1
                looked_s += look_cost.get(str(index[pos]), 0.05)
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
                many = spec.get("bail", True) and ((log[1] >= 2 and 4 * log[1] > log[0]) or log[1] > max_differ)
                # (when the differing queries are to be run: a patch that gives most of the
                # first queries another shape changes too much for that to pay)
                wide = max_rate and log[0] >= 4 and (log[1] - owed_first) > max_rate * log[0]
                if owed > limit or many or wide:
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
        if spec.get("replay"):
            _replay_duck(cls, recipe)
        # A patched optimizer or executor can make a query run for minutes (a join without
        # its condition, say). Each query gets a CPU allowance of several times what it costs
        # on the pinned commit; past it the test is left unsettled, as it was before the try.
        timer = _cpu_timer()
        try:
            for pos in sorted(todo, key=lambda q: (cost.get(str(index[q]), 0.5), q)):
                if max_cpu is not None and _cpu_now() - t0 > max_cpu + limit:
                    UNVERIFIED.append(test_id)
                    return
                log[2] += 1
                setattr(cls, recipe["sqls"], [sqls[pos]])
                timer(max(2.0, 6.0 * cost.get(str(index[pos]), 0.5) + 1.0))
                try:
                    orig()
                except PhaseTimeout:
                    UNVERIFIED.append(test_id)
                    return
                finally:
                    timer(0)
                    setattr(cls, recipe["sqls"], sqls)
                outcome = getattr(self, "_outcome", None)
                if outcome is not None and not getattr(outcome, "success", True):
                    return                  # a query fails (or ran out of its allowance inside
                    #                         a subtest, which counts the same): so does the test
        finally:
            timer(None)

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
        seed = repr((args, sorted(kwargs.items(), key=repr)))
        if len(args) < 3 and "config" not in kwargs:
            kwargs["config"] = {"threads": DUCKDB_THREADS}
        real = orig(*args, **kwargs)
        if DUCK_ALL or DUCK_RECORD is not None:
            return DuckConn(real, _duck_digest("connect\0" + seed))
        return real

    connect._br_capped = True
    duckdb.connect = connect


# -- every DuckDB connection the tests open ---------------------------------------------------
# What DuckDB returns for a read-only statement depends on the statement and on what was done
# to the database before, not on the patched package. DuckConn keeps a digest of everything
# that could have changed the database of its connection: the connect arguments and, in
# order, every statement that is not plainly a query. A query whose text, in that state, is
# in the record of the pinned suite (maps/duck.bin, tools/build_duck.py) gets the recorded
# rows without DuckDB running it; every other statement goes to DuckDB as before, and
# anything the stand-in does not follow (parameters, other methods) ends the replay for that
# connection. DUCK_ALL is set per run, only for patches that touch nothing but the package's
# Python code: the tables the fixtures load from files are then the pinned ones.
DUCK_ALL = False
DUCK_RECORD: dict | None = None      # tools/build_duck.py: the record being made
_DUCK_RE: list = []


def _duck_digest(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def _duck_norm(text: str) -> str:
    """The statement without the place the tree happens to be at."""
    if not _DUCK_RE:
        import re

        _DUCK_RE.append(re.compile(r"'[^']*?/tests/fixtures/"))
        _DUCK_RE.append(re.compile(
            r"\b(insert|update|delete|create|drop|alter|copy|attach|detach|pragma|install|load|call|truncate|"
            r"merge|set|export|import|vacuum|checkpoint|nextval|use|begin|commit|rollback)\b"))
    return _DUCK_RE[0].sub("'<fixtures>/", text)


def _duck_query(norm: str) -> bool:
    """Plainly a query: starts like one and names nothing that writes."""
    low = norm.lstrip().lower()
    return low.startswith(("select", "with", "from", "values", "(")) and not _DUCK_RE[1].search(low)


class DuckConn:
    def __init__(self, real, state: str):
        self._real = real
        self._state = state          # None: no longer known

    def execute(self, query, *args, **kwargs):
        if args or kwargs or type(query) is not str or self._state is None:
            self._state = None
            return self._real.execute(query, *args, **kwargs)
        norm = _duck_norm(query)
        if not _duck_query(norm):
            self._state = _duck_digest(self._state + "\0" + norm)
            self._real.execute(query)
            return self
        base = self._state + "\0" + norm
        if DUCK_RECORD is not None:
            self._real.execute(query)
            return _DuckRecording(self._real, query, base)
        answers = duck_answers()
        rows, frame = answers.get(_duck_digest("rows\0" + base)), answers.get(_duck_digest("frame\0" + base))
        if rows is None and frame is None:
            self._real.execute(query)        # not in the record: DuckDB answers (and raises, if it must)
            return _DuckLive(self._real)
        return _DuckKnown(self._real, query, rows, frame)

    def close(self):
        return self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._real.close()
        return False

    def __getattr__(self, name):
        self._state = None
        return getattr(self._real, name)


class _DuckLive:
    """The result of a query DuckDB has just run."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)


class _DuckKnown:
    """The result of a query the pinned suite sent in the same state: it succeeded there."""

    def __init__(self, real, query, rows, frame):
        self._real, self._query, self._rows, self._frame = real, query, rows, frame

    @staticmethod
    def _load(blob):
        import pickle
        import zlib

        return pickle.loads(zlib.decompress(blob))

    def fetchall(self):
        if self._rows is not None:
            return self._load(self._rows)
        return self._real.execute(self._query).fetchall()

    def df(self, *args, **kwargs):
        if self._frame is not None and not args and not kwargs:
            return self._load(self._frame)
        return self._real.execute(self._query).df(*args, **kwargs)

    def fetchdf(self, *args, **kwargs):
        return self.df(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real.execute(self._query), name)


class _DuckRecording:
    """Recording: the query has been run; what is fetched is kept when DuckDB gives the same
    answer a second time."""

    def __init__(self, real, query, base):
        self._real, self._query, self._base = real, query, base

    def _keep(self, kind, value, again, same):
        import pickle
        import zlib

        try:
            if same(value, again):
                DUCK_RECORD[_duck_digest(kind + "\0" + self._base)] = zlib.compress(pickle.dumps(value, protocol=4), 6)
        except Exception:  # noqa: BLE001 - not comparable or not picklable: never replayed
            pass
        return value

    def fetchall(self):
        value = self._real.fetchall()
        again = self._real.execute(self._query).fetchall()
        return self._keep("rows", value, again, lambda a, b: a == b and repr(a) == repr(b))

    def df(self, *args, **kwargs):
        value = self._real.df(*args, **kwargs)
        if args or kwargs:
            return value
        again = self._real.execute(self._query).df()
        return self._keep("frame", value, again, lambda a, b: a.equals(b) and list(a.dtypes) == list(b.dtypes))

    def fetchdf(self, *args, **kwargs):
        return self.df(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


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
