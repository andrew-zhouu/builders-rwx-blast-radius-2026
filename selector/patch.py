"""Parse unified diffs and apply them to file contents held in memory."""
from __future__ import annotations

import re

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


class Hunk:
    __slots__ = ("old_start", "old_len", "new_start", "new_len", "lines", "old_no_eol", "new_no_eol")

    def __init__(self, old_start, old_len, new_start, new_len):
        self.old_start = old_start
        self.old_len = old_len
        self.new_start = new_start
        self.new_len = new_len
        self.lines: list[tuple[str, str]] = []  # (tag, text without newline); tag in " -+"
        self.old_no_eol = False
        self.new_no_eol = False


class FilePatch:
    __slots__ = ("old_path", "new_path", "hunks", "is_new", "is_deleted", "is_binary", "mode_only")

    def __init__(self):
        self.old_path: str | None = None
        self.new_path: str | None = None
        self.hunks: list[Hunk] = []
        self.is_new = False
        self.is_deleted = False
        self.is_binary = False
        self.mode_only = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


def _strip_prefix(p: str) -> str | None:
    p = p.strip()
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    p = p.split("\t")[0]
    if p == "/dev/null":
        return None
    if p.startswith(("a/", "b/")):
        return p[2:]
    return p


def parse_patch(text: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    hunk: Hunk | None = None
    remaining_old = remaining_new = 0
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if hunk is not None and (remaining_old > 0 or remaining_new > 0):
            tag = line[:1]
            if tag == "\\":
                # "\ No newline at end of file" applies to the previous line.
                if hunk.lines:
                    prev = hunk.lines[-1][0]
                    if prev in " -":
                        hunk.old_no_eol = True
                    if prev in " +":
                        hunk.new_no_eol = True
                continue
            if tag == "" and line == "":
                tag = " "  # some tools strip the trailing space of empty context lines
            if tag == " ":
                remaining_old -= 1
                remaining_new -= 1
            elif tag == "-":
                remaining_old -= 1
            elif tag == "+":
                remaining_new -= 1
            else:
                raise PatchError(f"unexpected hunk line: {line[:40]!r}")
            hunk.lines.append((tag, line[1:]))
            continue
        if line.startswith("\\") and hunk is not None and hunk.lines:
            prev = hunk.lines[-1][0]
            if prev in " -":
                hunk.old_no_eol = True
            if prev in " +":
                hunk.new_no_eol = True
            continue
        if line.startswith("diff --git "):
            cur = FilePatch()
            files.append(cur)
            hunk = None
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                cur.old_path, cur.new_path = m.group(1), m.group(2)
            cur.mode_only = True
            continue
        if cur is None:
            if line.startswith("--- "):
                cur = FilePatch()
                files.append(cur)
            else:
                continue
        if line.startswith("new file mode"):
            cur.is_new = True
        elif line.startswith("deleted file mode"):
            cur.is_deleted = True
        elif line.startswith("rename from "):
            cur.old_path = line[len("rename from "):].strip()
            cur.mode_only = False
        elif line.startswith("rename to "):
            cur.new_path = line[len("rename to "):].strip()
            cur.mode_only = False
        elif line.startswith(("Binary files ", "GIT binary patch")):
            cur.is_binary = True
            cur.mode_only = False
        elif line.startswith("--- "):
            if cur.hunks:  # a headerless second file
                cur = FilePatch()
                files.append(cur)
            p = _strip_prefix(line[4:])
            if p is None:
                cur.is_new = True
            else:
                cur.old_path = p
            cur.mode_only = False
            hunk = None
        elif line.startswith("+++ "):
            p = _strip_prefix(line[4:])
            if p is None:
                cur.is_deleted = True
            else:
                cur.new_path = p
        else:
            m = HUNK_RE.match(line)
            if m:
                hunk = Hunk(
                    int(m.group(1)),
                    int(m.group(2)) if m.group(2) is not None else 1,
                    int(m.group(3)),
                    int(m.group(4)) if m.group(4) is not None else 1,
                )
                cur.hunks.append(hunk)
                cur.mode_only = False
                remaining_old, remaining_new = hunk.old_len, hunk.new_len
    return files


def apply_hunks(old: str | None, hunks: list[Hunk], edits: list | None = None) -> str:
    """Apply hunks to text. Raises PatchError when the context does not match.

    edits, when given, receives one (old_lo, old_hi, new_lo, new_hi) per changed run of
    lines, 1-based and inclusive; an empty range has hi == lo - 1 (pure insertion/deletion)."""
    old_lines = [] if not old else old.split("\n")
    had_eol = bool(old) and old.endswith("\n")
    if had_eol:
        old_lines.pop()
    out: list[str] = []
    pos = 0
    new_no_eol = not had_eol and bool(old)
    for h in hunks:
        expect = [t for tag, t in h.lines if tag in " -"]
        start = h.old_start - 1 if h.old_len else h.old_start
        if not expect:
            start = h.old_start  # pure insertion after line old_start
        start = _locate(old_lines, expect, start, pos)
        out.extend(old_lines[pos:start])
        old_i = start
        run = None
        for tag, t in h.lines:
            if tag == " ":
                if run is not None and edits is not None:
                    edits.append((run[0] + 1, old_i, run[1] + 1, len(out)))
                run = None
                old_i += 1
                out.append(t)
            else:
                if run is None:
                    run = (old_i, len(out))
                if tag == "-":
                    old_i += 1
                else:
                    out.append(t)
        if run is not None and edits is not None:
            edits.append((run[0] + 1, old_i, run[1] + 1, len(out)))
        pos = start + len(expect)
        if pos >= len(old_lines):
            new_no_eol = h.new_no_eol
    out.extend(old_lines[pos:])
    text = "\n".join(out)
    if out and not new_no_eol:
        text += "\n"
    return text


def _locate(old_lines: list[str], expect: list[str], start: int, floor: int) -> int:
    n = len(expect)
    if old_lines[start:start + n] == expect and start >= floor:
        return start
    for delta in range(1, 400):  # tolerate an offset, like patch(1) does
        for s in (start - delta, start + delta):
            if s >= floor and s + n <= len(old_lines) and old_lines[s:s + n] == expect:
                return s
    raise PatchError("hunk does not apply")
