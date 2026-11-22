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
2. **Drop changes that cannot alter any run** (`selector/ident.py`). Turning `a is b` into
   `a == b` (or `is not` into `!=`) changes nothing when the two operators agree for every
   pair of objects that reaches the comparison: one side is None and the other side's class
   never equals None, both are members of one enumeration, either class compares by
   identity, and so on. A map lists, for each identity comparison of the package, the pairs
   of operand classes the pinned test suite produces there and how each class compares for
   equality; if all pairs are of that kind the patched program does exactly what the pinned
   one does, test by test, and nothing is selected or re-run for that change.
3. **Look the change up in maps of the pinned commit** (`selector/plan.py`) to get the tests
   it could affect: statement-level coverage for call-time changes; for fixture files the
   tests that load them and the entries that changed (`selector/fixtures.py`).
4. **Measure import-time changes instead of guessing** (`selector/state.py`,
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
5. **Re-run the candidates against the patched tree** (`selector/verify.py`) and keep the
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
   * re-running `TestGenerator::test_generator_memory` on the dialects a patch can reach
     only (`brh.LOOP_RECIPES`, `Selector.loop_slice`). The test builds 1,100 generators of
     each of 33 dialects and costs 4.5 s of CPU whenever a change to any dialect, generator
     or parser module makes it a candidate. What one round of its loop uses is measured on
     the pinned tree, in a forked child, before anything else of the commit runs: the modules
     whose code runs, and the modules of the classes that are initialised, handed out by the
     dialect registry, or set as the dialect's tokenizer, parser and generator. A round is
     reached when one of those modules is a changed module, a module in which importing the
     patched tree computed something different, or a module that imports one of them,
     directly or not, at top level or inside a function. Any other round runs the same code
     on the same objects as on the pinned commit. Not used when a changed file imports
     something else than before, or when the import of the patched tree added, removed or
     failed anything;
   * stopping early once the outcome can no longer change the selection.
6. **A candidate that was not settled is selected.** Settled means re-run whole, or on
   every affected fixture entry / unit. The one exception is the short list in
   `tools/never_alone.json`: three tests that cost more to re-run than to select, and that
   were measured never to fail on their own (see "How this was checked"). They are left out
   when the change is call-time only and every other candidate was settled and passed.
   For the executor's TPC-DS test there is one more way to settle it, tried when it would
   otherwise be selected without having been seen to fail (`brh.PHASE_RECIPES`,
   `Selector.phase_spec`). The test sends its queries through parse, optimize, plan and the
   Python executor and compares the rows with DuckDB's; it costs 37 s of CPU to re-run and
   21 s to select. For one query, everything after the optimizer sees the patched program
   through two values only: the parsed and the optimized expression. If both are what they
   are on the pinned commit, and the patch only changes function bodies none of whose lines
   the query runs outside parsing and optimizing, the rest is unchanged code on unchanged
   input, and the query does what it does on the pinned commit. So the affected queries are
   parsed and optimized one by one (3 s for all of them) and compared with a digest of the
   pinned expressions. When all of them are the same the test is left out; at the first
   query that differs, or when the patch touches what queries run after the optimizer, the
   test stays selected as before. Nothing is executed and no fixture is built for this, so
   a try costs at most the looks, and it is only made when leaving the test out saves more
   than that. The check is also made beside edits of fixture files that this test does not
   read. It is never made for patches to the executor, the planner or `serde.py` (the
   digests are made with it). The same looks are used when the test is re-run on the
   queries a patch reaches (a slice, step 5): there a query that differs, or that runs a
   changed line after the optimizer, is executed, one at a time, so the outcome is that of
   executing every reached query, for less CPU.
   (`Selector.evidence_probe` also holds two bets that leave this test and the parser's
   29 s test out on indirect evidence. They are switched on in this entry, `BR_RISK=1`: see
   "Two bets" below.)
7. **Add the tests that fail because of another test.** `TestOptimizer` makes one process
   pool for the whole class and 22 of its tests use it. A pool breaks for good when a worker
   cannot unpickle a call, or the parent cannot unpickle a result or a raised exception, and
   from then on every test using it fails with `BrokenProcessPool`, whether or not it runs
   the changed code. The harness's stand-in pool reports when that would have happened in
   a re-run, and then all tests using that pool are selected. A pool-using test that is
   selected without having been re-run to the end could do the same unseen: if the patch
   touches code that runs while pools pickle (the serialiser, the expression and exception
   classes, and whatever else was seen running at such a moment on the pinned commit), the
   pool-using tests that come after it in the suite's order are selected as well.
8. **Split** the selected tests over workers to minimise `largest load + 2 s x workers`
   (`selector/split.py`).

(`pin_to_one_cpu`, `BR_PIN`, is switched off. The selector and its children run one after
the other, and keeping to one CPU took a few percent off the CPU time in most interleaved
runs, but no more than those runs differ among themselves, so it was not adopted.)

**Two bets on the two most expensive tests** (`Selector.evidence_probe` in
`selector/main.py`). Up to here a test is selected unless it was shown to pass. For two tests
that rule is expensive: `TestParser::test_max_nodes` (29 s recorded) and
`TestExecutor::test_execute_tpcds` (21 s) cost more CPU to re-run than they cost to select, so
whenever a change can reach them and they were not seen to fail they are selected, mostly for
nothing. This entry gathers cheaper evidence against the patched tree instead, and leaves the
test out when the evidence passes. Everything else stays selected.

* **The parser test** builds one statement of a million nodes and expects the parser to
  refuse it at limits of 1,000,000 and of 50,000 nodes. A stand-in is run in its place: the
  same test method, recompiled from its source with a statement one hundredth as wide and
  limits of 4,000 and 5,000 (`RISK_SCALED`; 0.3 s of CPU instead of 30). It has to pass, and
  in no more than two and a half times the CPU the same stand-in takes on the pinned tree,
  which is measured once in the same run (`Selector.scaled_limit`). An earlier submission
  compared with a fixed 0.8 s instead; with every clock the selector reads scaled three
  times, as on a machine that much slower, that limit ended one of three tries on trial
  commits, and the limit used now none.
* **The executor's TPC-DS test**, where the two-phase check of step 6 could not be tried,
  only when the commit was decided without any re-run and only call-time code changed: the
  selected cheap tests of its class (under a second each, at least three of them) and the 30
  cheapest of the other selected tests are re-run, stopping at the first failure. The test
  is left out when all of them pass and no executor or optimizer test is known to fail.

Neither bet is made when the patch touches anything but the package's own Python code (the
evidence says nothing about a test whose own code or data changed), when a patched module
does not import, or when a re-run hung. Class fixtures are built for the evidence run
whenever every test of the class is a candidate, because then the change may be in what the
fixture runs. These are bets, not proofs; "How this was checked" says how they did.

Anything the planner cannot analyse (binary patches, hunks that do not apply, changes to
test-runner configuration) falls back to selecting every test; a re-run that hangs or
dies settles nothing, so every candidate of that commit is selected (every test, when it
is the import of the patched tree that does not come back).

## What is in the maps

All maps are derived from the pinned commit only, by running its own test suite under a
tracer or by reading its source. They describe that commit (what each test executes, reads
and costs; what classes of objects meet where) and contain no patches, no outcomes of
edits and no expected failures. Which tests fail on a patch is decided at run time by
running them on the patched code; the one decision made without running anything is "this
change cannot alter any run" (comments, annotations, and the identity comparisons of
step 2), and that is reasoned from the patch and the maps when the patch arrives.

| file | content | built by |
| --- | --- | --- |
| `maps/map.json.z` | per test: lines executed (split into import time / test time), modules loaded, classes instantiated, words and token types of the SQL it tokenizes or generates, files read, CPU cost | `tools/tracer/run_trace.py`, `time_tests.py`, `time_suite.py`, `tools/build_map.py` |
| `maps/pmap.json.z` | the same per fixture entry of every fixture-driven test, and per unit of every long test method | `tools/tracer/run_trace.py --entries` / `--units`, `tools/build_map.py --probes` |
| `maps/reads.json.z` | per test: which entries of sqlglot's class-level and module-level dicts and sets it looks up, tests for size, or walks | `tools/tracer/run_trace.py --reads` (`tools/tracer/readtrace.py`), `tools/build_map.py --reads` |
| `maps/cmp.json.z` | per `is` / `is not` comparison in the package: the pairs of operand classes the suite produces there, and per class how it compares for equality (by identity, as a built-in value type, only with its own exact class, never with None, unknown) | `tools/tracer/cmptrace.py`, `tools/build_cmp.py` |
| `maps/timp.json.z` | functions of the package that the module-level code of test files runs (what that computes stays in the test module, outside the import-state comparison) | `tools/tracer/timp.py` |
| `maps/phases.json.z` | per query of the executor's TPC tests: digests of its parsed and its optimized expression, the CPU of getting that far, and the lines it runs outside parsing and optimizing (measured in a fresh process per query that starts from the pinned expressions and never runs the parser or optimizer itself) | `tools/tracer/phases.py` |
| `maps/pools.json.z` | which tests submit calls to a process pool made in a class fixture, and which functions of the package run while a pool pickles or unpickles calls and results | `tools/tracer/pools.py` |
| `maps/idents.json.z` | where each module-level / class-level name is used in the source | `tools/build_idents.py` |
| `maps/outline.json.z` | line spans of module-level and class-level statements | `tools/build_outline.py` |
| `maps/state.json.z` (made by the `Dockerfile`) | module list, static import graph (top-level imports, and separately every import statement) | `tools/build_state.py` |

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

The identity-comparison map is a third trace of the same kind (`tools/tracer/cmptrace.py`):
each `a is b` in the package is rewritten to record the classes of `a` and `b`, in one
process per test and once more with the whole suite in one process. Nothing but the
original `is` is evaluated. Comparisons inside `__eq__`, `__hash__` and the like are left
out, because there `==` could re-enter the method a patch changes. How a class compares
is read off the class: `object.__eq__`, the `__eq__` of a built-in type, or an `__eq__`
of the package whose source has the shape `self is other or (type(self) is type(other)
and ...)`. Two more `__eq__` methods of the package (the dialect class and its metaclass)
were read by hand and are listed in the tracer with a fingerprint of their source: they
only equal strings and dialects. Every other class counts as unknown, and a comparison that sees one is
handled like any other change.

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
python $T/cmptrace.py --root /app/src --tests tests.json --out /tmp/cmp_0.pkl
python $T/cmptrace.py --root /app/src --tests tests.json --together --out /tmp/cmp_together.pkl
python /app/tools/build_cmp.py --out maps/cmp.json.z '/tmp/cmp_*.pkl'
python $T/timp.py --root /app/src --out maps/timp.json.z
python $T/pools.py --root /app/src --tests tests.json --out maps/pools.json.z
python $T/phases.py --root /app/src --out maps/phases.json.z
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
* **Identity comparisons.** A development pass of the tracer evaluated `==` next to each of
  the 665 recorded `is` comparisons, in every test: of the 646 that tests execute, the
  operators really disagree at 13, the rule calls 551 safe, and none of the 13 is among
  those. All 675 patches that swap one such operator were planned: 545 are dropped, none
  of them one where `==` differs. Twelve of the dropped ones (the six executed by most
  tests, six at random) were also run under full `pytest`: nothing fails.
* **Leaving a test out unsettled.** An earlier version left out any expensive test once
  enough cheaper candidates had passed. Trial edits showed that this is wrong for
  fixture-driven optimizer tests (they are often the only test to notice a change), so it
  was removed. What remains is the list of three tests, each of which failed dozens of
  times on trial edits and never as the only failing test.

* **Two phases for the executor's TPC-DS test.** On the labelled sets (1,482 commits: the
  30 of the sample, 1,175 labelled by full `pytest` runs, 53 by `pytest` on the optimizer
  tests with real pools, 224 by re-running every candidate) the check settles the test as
  passing in 33 commits, and no failing test is left out. 2,303 fresh single-token edits
  (the sample's operator mix, one random file each) were run through the selector as well,
  and the test was then run for real under `pytest` wherever the check had settled it: it
  passes in all of them (25 commits at the time, 55 with the later forms described
  below). A first build of the map got one of these wrong, an edit to a lambda in the
  executor's operator table: the tracer listened for line events only,
  and CPython 3.13 delivers those for a function that consists of a single line only while
  function starts are monitored as well, so the lines of such functions were missing from
  what queries run after the optimizer. The tracer now monitors both, the map was rebuilt
  (48 more lines), digests are compared across six hash seeds, and changes to the executor
  and the planner are excluded outright. A form that also executed the few queries that
  differ was tried and given up: what it cost depended too much on the machine. Where the
  test is re-run on the queries a patch reaches, the looks only save executions (used in
  31 labelled and 73 fresh commits); the selections there are the ones executing every
  reached query gives. A wider form was tried last and taken back: it also looked at the
  SQL text the test sends to DuckDB, so that edits to the parser and the DuckDB generator
  would no longer rule the check out. 2,480 edits were made on purpose on the lines that
  form newly covered, and the test was run for real under `pytest` wherever it had been
  left out: it fails in 3 of 90. All three break the lazy loading of dialects in the
  registry; the selector's process has every dialect module loaded, so the look saw
  nothing, while the test's own process asks for "duckdb" first there. The look at the
  DuckDB text is gone, and with it the map goes back to the one before. What stays from
  that round is the check beside fixture edits the test does not read.

* **Generator memory test on the reached dialects only.** On the labelled sets (1,482
  commits) the shortened re-run is used in 34 commits and no failing test is left out. On
  the 2,303 fresh edits it is used in 86 commits; the whole test was then run for real
  under `pytest` for each of them and passes in all 86, as the shortened one did. Among
  the labelled commits there are two where it is used and the whole test fails: the
  shortened one fails there too. Nine labelled commits that used to be decided without a
  re-run (300 tests selected, none or a handful failing) are now re-run and settled,
  because the test no longer makes the re-run cost more than the selection.

* **The two bets.** They were first written wider and narrowed by what went wrong. 2,303
  fresh single-token edits (the sample's operator mix, one random file each) were run through
  the selector, and for every commit where a bet left a test out that test was then run for
  real under `pytest`. Asking the executor test's cheap neighbours alone was wrong in 4 of 86
  such commits, and in 3 more among the labelled commits: after a re-run of the other
  candidates the expensive test is often one of the few tests left that could notice the
  change; a change that breaks the generator shows in the dialect tests and not in the
  executor's cheap ones; and one edit to a module's import-time code made tests fail only
  after others had run. Hence the restrictions above. In the form described here the same
  2,303 edits give 41 commits where a bet leaves a test out (the executor test in 10, the
  parser test in 31), and the real run of that test passes in all 41. On the labelled sets
  (1,482 commits: the 30 of the sample, 1,175 labelled by full `pytest` runs, 53 by `pytest`
  on the optimizer tests with real pools, 224 by re-running every candidate) no failing test
  is left out; the bets leave the parser test out in 57 of those commits and the executor
  test in 10, and take 585 s (4.6%) of selection out of them. The stand-in of the parser
  test agreed with the real test on all 37 labelled commits tried one by one, 12 of them
  commits where the real test fails. A wider form of evidence for the executor test, every
  candidate of its own class and of the optimizer's (on the labelled sets the test never
  fails without one of those failing, 205 failures), was implemented and left off
  (`BR_RISK_CLASS`): gathering it costs about what leaving the test out saves. The
  restrictions and thresholds of the bets come from these trial edits; no outcome of any
  edit is stored or looked up.

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
