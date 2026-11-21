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
   or execute code reading those names are the candidates.
4. **Re-run the candidates against the patched tree** (`selector/verify.py`) and keep the
   ones that fail. The patched tree is a symlink farm over the read-only checkout, the run
   happens in a forked child, and it is kept cheap by:
   * re-running fixture-driven tests (`check_file`, TPC-H/TPC-DS, `identity.sql` ...) on
     the affected fixture entries only, using a second, per-entry coverage map;
   * reusing the already imported modules for patches that change no Python code;
   * stopping early once the outcome can no longer change the selection.
5. **Tests too expensive to re-run** are selected, unless the change is purely call-time
   and every candidate that was re-run passed (all of them execute the changed statement,
   so the change is then taken to be behaviour-preserving).
6. **Split** the selected tests over workers to minimise `largest load + 2 s x workers`
   (`selector/split.py`).

Anything the planner cannot analyse (binary patches, hunks that do not apply, changes to
test-runner configuration, a hung re-run) falls back to selecting every test.

## What is in the maps

All maps are derived from the pinned commit only, by running its own test suite under a
tracer or by reading its source. They contain no patches, no kinds of edits and no
expected failures; which tests fail on a patch is always decided at run time, by running
them on the patched code.

| file | content | built by |
| --- | --- | --- |
| `maps/map.json.z` | per test: lines executed (split into import time / test time), modules loaded, classes instantiated, words and token types of the SQL it tokenizes or generates, files read, CPU cost | `tools/tracer/run_trace.py`, `time_tests.py`, `time_suite.py`, `tools/build_map.py` |
| `maps/pmap.json.z` | the same, per fixture entry of every fixture-driven test | `tools/tracer/run_trace.py --entries`, `tools/build_map.py --probes` |
| `maps/idents.json.z` | where each module-level / class-level name is used in the source | `tools/build_idents.py` |
| `maps/outline.json.z` | line spans of module-level and class-level statements | `tools/build_outline.py` |
| `maps/state.json.z` (made by the `Dockerfile`) | module list, static import graph | `tools/build_state.py` |

`tools/tracer/brtrace.py` is the tracer (`sys.monitoring`); `selector/brh.py` is the test
harness shared by the tracer and the run-time re-runs, so tests are traced exactly the way
they are later re-executed (in-process, with process pools replaced by an inline executor
that still pickles what crosses the process boundary).

The traced maps are committed because tracing 1620 tests in isolation, plus 6400 fixture
entries, takes longer than the build allows. To rebuild them, inside the image:

```sh
T=/app/tools/tracer
bash $T/trace_all.sh /app/src tests.json /tmp/trace 8
python $T/entry_jobs.py /tmp/entry_jobs.json /tmp/trace/shard_*.pkl
bash $T/trace_entries.sh /app/src tests.json /tmp/entry_jobs.json /tmp/probes 8
bash $T/time_all.sh /app/src tests.json /tmp/timing 4
python /app/tools/class_info.py /app/src /tmp/classinfo.json
python /app/tools/build_map.py --tests tests.json --out maps/map.json.z --timing /tmp/timing \
    --classinfo /tmp/classinfo.json /tmp/trace/shard_*.pkl
python /app/tools/build_map.py --tests tests.json --probes --test-shards '/tmp/trace/shard_*.pkl' \
    --classinfo /tmp/classinfo.json --out maps/pmap.json.z /tmp/probes/probes_*.pkl
python /app/tools/build_idents.py /app/src maps/idents.json.z
python /app/tools/build_outline.py /app/src maps/outline.json.z
```

## Build and run

The `Dockerfile` starts from the image the commits were labeled in, installs
`requirements.lock` from PyPI, downloads sqlglot at the pinned commit from its GitHub
repository and precompiles bytecode. At run time nothing is written outside `/tmp`.

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
