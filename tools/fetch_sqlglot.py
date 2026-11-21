"""Download sqlglot's source at one commit from its GitHub repository and unpack it.

Usage: python fetch_sqlglot.py <commit sha> <destination dir>
"""
import io
import os
import sys
import tarfile
import time
import urllib.request


def main():
    sha, dest = sys.argv[1], sys.argv[2]
    url = f"https://github.com/tobymao/sqlglot/archive/{sha}.tar.gz"
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    else:
        raise SystemExit(f"could not download {url}: {last}")
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        prefix = f"sqlglot-{sha}/"
        members = []
        for m in tar.getmembers():
            if not m.name.startswith(prefix):
                continue
            m.name = m.name[len(prefix):]
            if not m.name or m.name.startswith(("/", "..")) or "/../" in m.name:
                continue
            members.append(m)
        tar.extractall(dest, members=members, filter="data")
    print(f"sqlglot {sha} unpacked into {dest}: {len(members)} entries")


if __name__ == "__main__":
    main()
