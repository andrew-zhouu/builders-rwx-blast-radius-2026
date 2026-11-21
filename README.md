# Blast Radius selector

Entry for the RWX Blast Radius track of the Builders Buildathon: given a patch to sqlglot
at commit `eb11aec4be8cb2122afe713a4816662c3f0fb2fa`, print the tests that have to run and
how to split them over workers.

Written during the event (Nov 20-22, 2026). Released under the MIT License (`LICENSE`).

## Idea

Selecting from a static map is cheap but coarse; running tests is exact but is charged as
CPU time. This entry does both, in that order:

1. **Diff the patch structurally.** The patch is applied in memory and the old and new
   versions of each Python file are compared as syntax trees (`selector/astdiff.py`). Every
   change is classified as *call-time* (inside a function body: which statement of which
   function) or *import-time* (module or class level). Comment-only, whitespace-only and
   annotation-only edits compare equal and cost nothing.
2. **Look the change up in maps of the pinned commit** (`selector/plan.py`) to get the tests
   it could affect: statement-level coverage for call-time changes; for fixture files the
   tests that load them and the entries that changed (`selector/fixtures.py`).
3. **Measure import-time changes instead of guessing** (`selector/state.py`,
   `selector/statecands.py`). A child process imports the patched tree and fingerprints
   every module-level name and class attribute; comparing that with the same fingerprint
   of the pinned tree gives the definitions the change really altered, including derived
   ones (subclass tables, metaclass products). The tests that instantiate those classes
   or execute code reading those names are the candidates. When the altered definition is
   a dict or a set (sqlglot keeps most of its dialect knowledge in such tables), the
   candidates are narrowed to the tests that read the altered entries: a map records, per
   test, which entries of which table it looks up, tests for size, or walks.
   The same comparison settles function bodies that also run while modules are imported
   (the DuckDB generator, for one, parses its SQL templates at import time, so every test
   that loads it "runs" much of the parser): the tests that execute the changed lines
   themselves are candidates as usual, the ones that only import such a module are
   candidates only if the import state differs.
   A changed module- or class-level statement may also simply raise, and then every test
   that loads the module fails, whatever the statement defines. So whenever such code
   changed, the selector looks whether the patched modules still import (in the child
   that compares the import state or re-runs the candidates, or in a child of its own when
   nothing is re-run) and selects every test that loads a module that does not, and every
   test of a test file that cannot be imported.
4. **Re-run the candidates against the patched tree** (`selector/verify.py`) and keep the
   ones that fail. The patched tree is a symlink farm over the read-only checkout, the run
   happens in a forked child, and it is kept cheap by:
   * re-running fixture-driven tests (`check_file`, TPC-H/TPC-DS, `identity.sql` ...) on
     the affected fixture entries only, using a second, per-entry coverage map;
   * re-running long test methods on the affected *units* only: the statements of a test
     method are grouped by the local variables they share, and a group that only computes
     and asserts can be left out without changing what the others do
     (`tools/tracer/stmt_spec.py`; each unit has its own coverage record);
   * not importing anything twice: the selector imports the pinned sqlglot once; for a
     patch that only changes function bodies no import ever runs, a child forks from that
     process and the affected function objects get the patched code (`verify.apply_swaps`);
     for patches that change no Python code the pinned modules are used as they are;
   * stopping early once the outcome can no longer change the selection.
5. **A candidate that was not settled is selected.** Settled means re-run whole, or on
   every affected fixture entry / unit. The one exception is the short list in
   `tools/never_alone.json`: three tests that cost more to re-run than to select, and that
   were measured never to fail on their own (see "How this was checked"). They are left out
   when the change is call-time only and every other candidate was settled and passed.
6. **Add the tests that fail because of another test.** `TestOptimizer` makes one process
   pool for the whole class and 22 of its tests use it. A pool breaks for good when a worker
   cannot unpickle a call, or the parent cannot unpickle a result or a raised exception, and
   from then on every test using it fails with `BrokenProcessPool`, whether or not it runs
   the changed code. The harness's stand-in pool reports when that would have happened in
   a re-run, and then all tests using that pool are selected. A pool-using test that is
   selected without having been re-run to the end could do the same unseen: if the patch
   touches code that runs while pools pickle (the serialiser, the expression and exception
   classes, and whatever else was seen running at such a moment on the pinned commit), the
   pool-using tests that come after it in the suite's order are selected as well.
7. **Split** the selected tests over workers to minimise `largest load + 2 s x workers`
   (`selector/split.py`).

Anything the planner cannot analyse (binary patches, hunks that do not apply, changes to
test-runner configuration) falls back to selecting every test; a re-run that hangs or
dies settles nothing, so every candidate of that commit is selected (every test, when it
is the import of the patched tree that does not come back).

## What is in the maps

All maps are derived from the pinned commit only, by running its own test suite under a
tracer or by reading its source. They describe that commit (what each test executes, reads
and costs) and contain no patches, no outcomes of edits and no expected failures. Which
tests fail on a patch is decided at run time by running them on the patched code; the one
decision made without running anything is "this change cannot alter any run", and it is
made only for comments, whitespace and annotations.

| file | content | built by |
| --- | --- | --- |
| `maps/map.json.z` | per test: lines executed (split into import time / test time), modules loaded, classes instantiated, words and token types of the SQL it tokenizes or generates, files read, CPU cost | `tools/tracer/run_trace.py`, `time_tests.py`, `time_suite.py`, `tools/build_map.py` |
| `maps/pmap.json.z` | the same per fixture entry of every fixture-driven test, and per unit of every long test method | `tools/tracer/run_trace.py --entries` / `--units`, `tools/build_map.py --probes` |
| `maps/reads.json.z` | per test: which entries of sqlglot's class-level and module-level dicts and sets it looks up, tests for size, or walks | `tools/tracer/run_trace.py --reads` (`tools/tracer/readtrace.py`), `tools/build_map.py --reads` |
| `maps/timp.json.z` | functions of the package that the module-level code of test files runs (what that computes stays in the test module, outside the import-state comparison) | `tools/tracer/timp.py` |
| `maps/pools.json.z` | which tests submit calls to a process pool made in a class fixture, and which functions of the package run while a pool pickles or unpickles calls and results | `tools/tracer/pools.py` |
| `maps/idents.json.z` | where each module-level / class-level name is used in the source | `tools/build_idents.py` |
| `maps/outline.json.z` | line spans of module-level and class-level statements | `tools/build_outline.py` |
| `maps/state.json.z` (made by the `Dockerfile`) | module list, static import graph | `tools/build_state.py` |

`tools/tracer/brtrace.py` is the tracer (`sys.monitoring`); `selector/brh.py` is the test
harness shared by the tracer and the run-time re-runs, so tests are traced exactly the way
they are later re-executed (in-process, with process pools replaced by an inline executor
that still pickles what crosses the process boundary).

The table-read map comes from a second kind of trace (`tools/tracer/readtrace.py`): the
source of sqlglot and of its tests is rewritten as it is imported, so that every expression
whose value is consumed (subscripted, tested for membership, iterated, compared, passed to
code that was not rewritten ...) goes through a hook. The hooks know the tables by object
identity, so a table reached through another name is still the same table. A consumer the
rewriter does not know counts as reading the whole table. A copy made while a test runs
(`dict(table)`: the generator's dispatch table is built that way from `TRANSFORMS`) stands
for the table it was copied from, so lookups in the copy are recorded as lookups in the table.

The traced maps are committed because tracing 1620 tests in isolation, plus 13,000 fixture
entries and units, takes longer than the build allows. To rebuild them, inside the image:

```sh
T=/app/tools/tracer
bash $T/trace_all.sh /app/src tests.json /tmp/trace 8
python $T/entry_jobs.py /tmp/entry_jobs.json /tmp/trace/shard_*.pkl
bash $T/trace_entries.sh /app/src tests.json /tmp/entry_jobs.json /tmp/probes 8
bash $T/time_all.sh /app/src tests.json /tmp/timing 4
python /app/tools/class_info.py /app/src /tmp/classinfo.json
python /app/tools/build_map.py --tests tests.json --out maps/map.json.z --timing /tmp/timing \
    --classinfo /tmp/classinfo.json --never-alone tools/never_alone.json /tmp/trace/shard_*.pkl
python $T/stmt_spec.py /app/src tests.json /tmp/units.json --cpu maps/map.json.z --min-cpu 0.008 \
    --skip /tmp/entry_tests.json          # tests that already have per-entry records
bash $T/trace_units.sh /app/src tests.json /tmp/units.json /tmp/units 8
python /app/tools/build_map.py --tests tests.json --probes --test-shards '/tmp/trace/shard_*.pkl' \
    --classinfo /tmp/classinfo.json --units /tmp/units.json --out maps/pmap.json.z \
    /tmp/probes/probes_*.pkl /tmp/units/units_*.pkl
python $T/run_trace.py --root /app/src --tests tests.json --reads --out /tmp/reads_0.pkl
python /app/tools/build_map.py --tests tests.json --out maps/map.json.z --timing /tmp/timing \
    --classinfo /tmp/classinfo.json --never-alone tools/never_alone.json \
    --reads '/tmp/reads_*.pkl' --reads-out maps/reads.json.z /tmp/trace/shard_*.pkl
python $T/timp.py --root /app/src --out maps/timp.json.z
python $T/pools.py --root /app/src --tests tests.json --out maps/pools.json.z
python /app/tools/build_idents.py /app/src maps/idents.json.z
python /app/tools/build_outline.py /app/src maps/outline.json.z
```

## How this was checked

A missed failing test costs 300 s, so every shortcut above was checked against the plain
way of doing the same thing, on trial edits made for the purpose (single-token changes of
the kinds seen in the sample set, plus deletions, renames and test-file edits). The trial
edits and their results are not part of the entry: nothing derived from them is looked up
at run time, except the three test names in `tools/never_alone.json`.

* **Selections against labels.** 30 sample commits (official answers), about 730 synthetic
  commits labelled by full `pytest` runs (mixed kinds, unusual kinds such as deletions and
  renames, class-level flags and constants, the sample's operator mix drawn over lines and
  over files, function bodies that imports run), 107 + 32 + 85 commits labelled by
  re-running every candidate: no failing test left out. Four such labels did catch misses
  on the way (a renamed logger, a changed member of an enumeration, the process pool that
  the optimizer tests share, a class-level statement that raises on import); the causes are
  fixed and described in the code.
* **Edits that stop test files from importing.** 30 such edits were labelled with
  `pytest --continue-on-collection-errors`, so that the tests of the files that still import
  have labels too: no failing test left out.
* **Shared process pool.** 54 trial edits in the code that runs while pools pickle were
  run under `pytest` with real pools on the optimizer tests; 12 of them break the pool
  there. No failing test left out.
* **Unit and fixture-entry re-runs, in-place code replacement.** For trial edits, the tests
  are run both the short way and whole against a fresh import; the verdicts have to agree.
* **Table-read map.** For 969 import-time trial edits (391 of them in generator tables),
  every test the read map removed from the candidates was re-run whole (12,462 runs) and
  passed.
* **Leaving a test out unsettled.** An earlier version left out any expensive test once
  enough cheaper candidates had passed. Trial edits showed that this is wrong for
  fixture-driven optimizer tests (they are often the only test to notice a change), so it
  was removed. What remains is the list of three tests, each of which failed dozens of
  times on trial edits and never as the only failing test.

## Build and run

The `Dockerfile` starts from the image the commits were labeled in, installs
`requirements.lock` from PyPI, downloads sqlglot at the pinned commit from its GitHub
repository, precompiles bytecode and unpacks the maps for fast loading. At run time nothing
is written outside `/tmp`.

```sh
make sample   # needs the starter kit's sample/ and score.py next to the Makefile
```

## Third-party software

No code written before the event is included. The entry uses:

* [sqlglot](https://github.com/tobymao/sqlglot) (MIT) — the project under test, fetched at
  the pinned commit during the build; its own test suite is what gets re-run.
* The packages in `requirements.lock` (the organisers' labeling environment), installed
  from PyPI; of these the run uses duckdb, pandas, numpy and their dependencies, because
  sqlglot's tests import them.
* The Python 3.13 standard library (`python:3.13-slim` base image).
