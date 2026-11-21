#!/usr/bin/env bash
# Trace test methods one unit at a time (units from stmt_spec.py), across N shards.
# Usage: trace_units.sh <sqlglot_root> <tests.json> <unit spec json> <out_dir> <shards>
set -uo pipefail
root="${1:?sqlglot root}"; tests="${2:?tests.json}"; spec="${3:?unit spec}"; out="${4:?out dir}"; n="${5:-8}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
for i in $(seq 0 $((n-1))); do
  python "$here/run_trace.py" --root "$root" --tests "$tests" --units "$spec" \
    --out "$out/units_$i.pkl" --shard "$i/$n" \
    2> >(grep -v 'Unable to set' > "$out/units_$i.log") &
done
wait
tail -q -n 1 "$out"/units_*.log | grep -E 'done' | head -20
grep -h 'not ok' "$out"/units_*.log | grep -v ' 0 not ok' | head -40
