"""Compile the identity-comparison profile (tools/tracer/cmptrace.py) into maps/cmp.json.z.

  {"types": [[class name, how it compares], ...],
   "sites": {file: {"line:col": [[left class index, right class index], ...]}},
   "bad":   [tests that could not be profiled]}

Usage: python3 build_cmp.py --out maps/cmp.json.z cmp_*.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import zlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    sites = None
    kinds: dict[str, str] = {}
    pairs: dict[int, set] = {}
    bad: list[str] = []
    n_tests = 0
    for pattern in a.shards:
        for path in sorted(glob.glob(pattern)):
            with open(path, "rb") as f:
                recs = pickle.load(f)
            if sites is None:
                sites = recs[0]["sites"]
            elif [tuple(s) for s in sites] != [tuple(s) for s in recs[0]["sites"]]:
                raise SystemExit(f"{path}: site table differs between shards")
            for r in recs[1:]:
                n_tests += 1
                if r.get("error") or not r.get("ok") or "pairs" not in r:
                    bad.append(r["id"])
                    if "pairs" not in r:
                        continue
                for name, kind in r["types"].items():
                    if kinds.setdefault(name, kind) != kind:
                        kinds[name] = "unknown"
                for sid, ps in r["pairs"].items():
                    pairs.setdefault(sid, set()).update(tuple(p) for p in ps)
    names = sorted(kinds)
    index = {n: i for i, n in enumerate(names)}
    out_sites: dict[str, dict[str, list]] = {}
    for sid, ps in pairs.items():
        rel, line, col = sites[sid]
        out_sites.setdefault(rel, {})[f"{line}:{col}"] = sorted([index[x], index[y]] for x, y in ps)
    data = {"types": [[n, kinds[n]] for n in names], "sites": out_sites, "bad": sorted(bad)}
    raw = json.dumps(data, separators=(",", ":")).encode()
    with open(a.out, "wb") as f:
        f.write(zlib.compress(raw, 9))
    n_pairs = sum(len(v) for v in pairs.values())
    print(f"identity comparisons: {n_tests} tests ({len(bad)} not profiled), {len(sites)} sites, "
          f"{len(pairs)} executed, {n_pairs} class pairs, {len(names)} classes, "
          f"raw={len(raw) / 1e6:.2f}MB z={len(zlib.compress(raw, 9)) / 1e6:.2f}MB")
    by_kind: dict[str, int] = {}
    for k in kinds.values():
        by_kind[k] = by_kind.get(k, 0) + 1
    print("classes by kind:", dict(sorted(by_kind.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
