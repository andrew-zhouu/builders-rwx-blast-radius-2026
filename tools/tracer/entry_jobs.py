"""From per-test trace shards, list the tests that read fixtures through the loaders
(the input of run_trace.py --entries). Usage: entry_jobs.py <out.json> <shard.pkl>..."""
import json
import pickle
import sys

out, *shards = sys.argv[1:]
jobs = []
for s in shards:
    for r in pickle.load(open(s, "rb")):
        fc = r.get("fixture_counts") or {}
        if fc and r.get("filter") is None:
            jobs.append({"number": r["n"], "id": r["id"], "files": fc, "cpu": r["t_run_cpu"]})
# Interleave heavy and light tests across shards.
jobs.sort(key=lambda j: -j["cpu"])
json.dump(jobs, open(out, "w"), indent=1)
total = sum(sum(j["files"].values()) for j in jobs)
print(f"{len(jobs)} fixture-driven tests, {total} (test, entry) pairs at most")
for j in jobs[:60]:
    print(f"  {j['number']:5d} cpu={j['cpu']:6.2f} {j['id']} {j['files']}")
