#!/usr/bin/env bash
# Time the tests of the pinned checkout across N shards: alone (time_tests.py, pass 1) and
# class by class (time_suite.py, pass 2).
# Usage: time_all.sh <sqlglot_root> <tests.json> <out_dir> <shards> [suite|alone|both]
set -uo pipefail
root="${1:?sqlglot root}"; tests="${2:?tests.json}"; out="${3:?out dir}"; n="${4:-4}"; what="${5:-both}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
if [ "$what" != "suite" ]; then
  for i in $(seq 0 $((n-1))); do
    python "$here/time_tests.py" --root "$root" --harness "$here/../../selector" \
      --tests "$tests" --out "$out/time_$i.json" --shard "$i/$n" &
  done
  wait
fi
if [ "$what" != "alone" ]; then
  for i in $(seq 0 $((n-1))); do
    python "$here/time_suite.py" --root "$root" --harness "$here/../../selector" \
      --tests "$tests" --out "$out/suite_$i.json" --shard "$i/$n" &
  done
  wait
fi
