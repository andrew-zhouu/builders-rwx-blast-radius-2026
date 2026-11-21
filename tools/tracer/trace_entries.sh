#!/usr/bin/env bash
# Trace fixture-driven tests one fixture entry at a time, across N shards.
# Usage: trace_entries.sh <sqlglot_root> <tests.json> <entry_jobs.json> <out_dir> <shards>
set -uo pipefail
root="${1:?sqlglot root}"; tests="${2:?tests.json}"; jobs="${3:?entry jobs}"; out="${4:?out dir}"; n="${5:-8}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
for i in $(seq 0 $((n-1))); do
  python "$here/run_trace.py" --root "$root" --tests "$tests" --entries "$jobs" \
    --out "$out/probes_$i.pkl" --shard "$i/$n" \
    2> >(grep -v 'Unable to set' > "$out/probes_$i.log") &
done
wait
tail -q -n 2 "$out"/probes_*.log | grep -E 'done' | head -20
grep -h 'not ok' "$out"/probes_*.log | grep -v ' 0 not ok' | head -20
