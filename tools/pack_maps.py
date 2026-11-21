"""Unpack the maps for fast loading: <name>.json.z -> <name>.pkl (same content, test sets as
integers). Run once when the image is built; the selector prefers the .pkl files.

Usage: python pack_maps.py <maps dir> [<out dir>]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import zlib


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json.z"):
            continue
        with open(os.path.join(src, fn), "rb") as f:
            data = json.loads(zlib.decompress(f.read()))
        if isinstance(data, dict) and isinstance(data.get("sets"), list):
            data["sets"] = [int(h, 16) for h in data["sets"]]
        out = os.path.join(dst, fn[: -len(".json.z")] + ".pkl")
        with open(out, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{fn} -> {os.path.basename(out)} {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
