#!/usr/bin/env bash
# Trace all tests (or a job list) of the pinned checkout across N shards in parallel.
# Usage: trace_all.sh <sqlglot_root> <tests.json> <out_dir> <shards> [jobs.json]
set -uo pipefail
root="${1:?sqlglot root}"; tests="${2:?tests.json}"; out="${3:?out dir}"; n="${4:-8}"; jobs="${5:-}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
for i in $(seq 0 $((n-1))); do
  python "$here/run_trace.py" --root "$root" --tests "$tests" \
    --out "$out/shard_$i.pkl" --shard "$i/$n" --tmp "/tmp/brtrace_$i" ${jobs:+--jobs "$jobs"} \
    2> >(grep -v 'Unable to set' > "$out/shard_$i.log") &
done
wait
tail -q -n 6 "$out"/shard_*.log | grep -E 'done|   tests' | head -60
