"""DuckDB's answers to the reference queries of TestExecutor.test_execute_tpcds, recorded on
the pinned commit (see selector/brh.py, DuckAnswers).

For each query it executes, the test asks DuckDB for the rows of the query's DuckDB text
(transpile(sql, write="duckdb")) and compares them with what sqlglot's Python executor
returns. Nearly half of the test's CPU is DuckDB computing those reference rows, and they
depend on nothing but the text and the fixture tables. This tool runs the pinned test's
own fixture and records, for every such text, the frame DuckDB returns, and how long the
rest of a query takes (parse, optimize, plan, execute, compare).

maps/duck.bin: pickle of {"answers": {sha1 of the text: zlib(pickle(frame))},
                          "exec_cpu": {query index: seconds without DuckDB}}

Usage (inside the image): python build_duck.py /app/src maps/duck.bin
"""
from __future__ import annotations

import hashlib
import os
import pickle
import sys
import time
import zlib


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def main():
    root, out = sys.argv[1], sys.argv[2]
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ["SKIP_INTEGRATION"] = "0"
    import unittest  # noqa: F401 - loaded before the package, as under pytest

    import tests.test_executor as te

    cls = te.TestExecutor
    cls.setUpClass()
    te.initializer(te.TPCDS_SCHEMA, cls.tpcds_tables)
    case = cls("test_execute_tpcds")
    answers: dict = {}
    exec_cpu: dict = {}
    conn = cls.tpcds_conn
    for attempt in (0, 1):                      # the second pass is timed (warm caches)
        for i, (meta, sql, _) in enumerate(cls.tpcds_sqls):
            t0 = time.process_time()
            table = te.mp_execute(te.parse_one(sql), meta)
            if table is None:
                continue
            text = te.transpile(sql, write="duckdb")[0]
            spent = time.process_time() - t0
            if attempt == 0:
                frame = conn.execute(text).fetchdf()
                again = conn.execute(text).fetchdf()
                if not frame.equals(again):
                    continue                    # not the same answer twice: never replayed
                answers[key(text)] = zlib.compress(pickle.dumps(frame, protocol=4), 6)
                cls.cache[sql] = frame
            else:
                t1 = time.process_time()
                case.subtestHelper(i, table, tpch=False)
                exec_cpu[i] = round(spent + time.process_time() - t1, 4)
    cls.tearDownClass()
    n_tpcds = len(answers)
    # Every other read-only statement the DuckDB-using test files send, in the state of the
    # database they send it in (selector/brh.py, DuckConn): the tests are run once through
    # the selector's own harness with the recording stand-in on their connections.
    selector = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selector"))
    sys.path.append(selector)
    import brh
    import verify

    verify.warm_up(root, selector, ())
    brh.DUCK_RECORD = record = {}
    groups = []
    for rel in ("tests/test_optimizer.py", "tests/test_executor.py"):
        mod = sys.modules.get(rel[:-3].replace("/", ".")) or __import__(rel[:-3].replace("/", "."), fromlist=["x"])
        ids = []
        for name in sorted(vars(mod)):
            klass = getattr(mod, name)
            if isinstance(klass, type) and issubclass(klass, unittest.TestCase) and klass.__module__ == mod.__name__:
                ids += [f"{rel}::{name}::{m}" for m in sorted(dir(klass))
                        if m.startswith("test") and m != "test_execute_tpcds"]
        groups.append({"ids": ids, "filter": {}, "skip_setup": [], "if_failed": False})
    res = verify.run_groups(groups)
    brh.DUCK_RECORD = None
    answers.update(record)
    with open(out, "wb") as f:
        pickle.dump({"answers": answers, "exec_cpu": exec_cpu}, f, protocol=4)
    print(f"{n_tpcds} answers for the executor's TPC-DS test, {len(record)} for the other statements of "
          f"{sum(len(g['ids']) for g in groups)} tests ({len(res.get('failed') or {})} of them failed here), "
          f"{os.path.getsize(out)} bytes; the queries without DuckDB: {sum(exec_cpu.values()):.1f} s")


if __name__ == "__main__":
    main()
