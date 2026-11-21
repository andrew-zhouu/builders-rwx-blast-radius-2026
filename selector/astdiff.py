"""Structural diff of two versions of a Python file.

Turns "these lines changed" into "this statement of this function changed" or
"this class-level definition changed", in the line numbers of the OLD file so
the result can be looked up in maps built from the pinned commit.
"""
from __future__ import annotations

import ast
from difflib import SequenceMatcher

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
IDENTITY_SWAPS = ((ast.Is, ast.Eq), (ast.IsNot, ast.NotEq))
BODY_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")


class Change:
    """One changed spot.

    kind:
      "call"    code inside a function body; `lines` is the old span of the
                smallest enclosing statement (header only for compound ones)
      "func"    statements were inserted into or removed from a function body;
                `lines` is the old span of the whole function
      "import"  code that runs when the module is imported (module or class level,
                decorators, default arguments)
    """

    __slots__ = ("kind", "lines", "cls", "target", "key", "func", "names", "note", "annotation",
                 "side_effects", "top", "ident")

    def __init__(self, kind, lines, cls=None, target=None, key=None, func=None, names=(), note="",
                 annotation=False, side_effects=False):
        self.kind = kind
        self.lines = lines
        self.cls = cls          # qualified name of the enclosing class, if any
        self.target = target    # name assigned by the enclosing module/class level statement
        self.key = key          # source of the dict key whose value changed, if known
        self.func = func        # (name, lo, hi) of the enclosing function, old coordinates
        self.names = tuple(names)  # identifiers / string constants in the old and new code
        self.note = note
        # The change is confined to the annotation of a function parameter / return value
        # or of a local variable: never evaluated under `from __future__ import annotations`.
        self.annotation = annotation
        # Import-time code that may do more than bind names (bare calls, loops, ...): its
        # effect cannot be read off the module's namespace.
        self.side_effects = side_effects
        # For a change inside the body of a module- or class-level `def` (diff_region only):
        # {"name", "cls", "col", "old": (first, last line), "new": (first, last line)} of that def.
        self.top = None
        # (line, column) in the old file of an `a is b` / `a is not b` comparison whose
        # operator became `==` / `!=` with the operands left as they were, when that is
        # the whole change (it is reported as one change at the comparison).
        self.ident = None

    def __repr__(self):
        return (f"Change({self.kind}, lines={self.lines}, cls={self.cls}, target={self.target}, "
                f"key={self.key}, func={self.func and self.func[0]}, note={self.note})")


class _Ctx:
    __slots__ = ("scope", "func", "cls", "stmt", "in_header", "target", "key", "annotation")

    def __init__(self):
        self.scope = "import"   # "import" | "call"
        self.func = None        # innermost function/lambda node (old tree)
        self.cls = ()           # tuple of enclosing class names
        self.stmt = None        # innermost statement node (old tree)
        self.in_header = True   # change is in the statement's own expression, not a nested body
        self.target = None
        self.key = None
        self.annotation = False

    def copy(self):
        c = _Ctx()
        c.scope, c.func, c.cls, c.stmt = self.scope, self.func, self.cls, self.stmt
        c.in_header, c.target, c.key = self.in_header, self.target, self.key
        c.annotation = self.annotation
        return c


def _dump(node) -> str:
    return ast.dump(node) if isinstance(node, ast.AST) else repr(node)


def _first_line(node) -> int:
    line = node.lineno
    for d in getattr(node, "decorator_list", ()) or ():
        line = min(line, d.lineno)
    return line


def _header_span(stmt) -> tuple[int, int]:
    """Old line span of a statement's own code (excluding nested statement bodies)."""
    lo = _first_line(stmt)
    hi = getattr(stmt, "end_lineno", lo) or lo
    first_inner = None
    for field in BODY_FIELDS:
        inner = getattr(stmt, field, None)
        if isinstance(inner, list) and inner and isinstance(inner[0], ast.AST):
            cand = _first_line(inner[0]) if hasattr(inner[0], "lineno") else None
            if cand is not None and (first_inner is None or cand < first_inner):
                first_inner = cand
    if first_inner is not None:
        hi = max(lo, first_inner - 1) if first_inner > lo else lo
    return lo, hi


def _func_span(fn) -> tuple[str, int, int]:
    name = getattr(fn, "name", "<lambda>")
    return name, _first_line(fn), fn.end_lineno


def _names_of(*nodes) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        if not isinstance(node, ast.AST):
            continue
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, ast.Attribute):
                out.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and 0 < len(n.value) <= 60:
                out.add(n.value)
    return out


def _assign_target(stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and stmt.targets:
        t = stmt.targets[0]
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        t = stmt.target
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return stmt.name
    else:
        return None
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Tuple) and t.elts and isinstance(t.elts[0], ast.Name):
        return t.elts[0].id
    return None


_BINDING = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.FunctionDef, ast.AsyncFunctionDef,
            ast.Import, ast.ImportFrom, ast.Pass)


def _side_effects(stmt) -> bool:
    """Does import-time statement `stmt` do anything besides binding names?

    Assignments, defs, classes and imports only bind (what they compute shows up in the
    namespace); a bare call, a loop, a `with`, a `del` ... may change state elsewhere."""
    if stmt is None or not isinstance(stmt, (ast.stmt, ast.excepthandler)):
        return False
    if isinstance(stmt, _BINDING):
        return False
    if isinstance(stmt, ast.Expr):
        return not isinstance(stmt.value, ast.Constant)
    if isinstance(stmt, (ast.ClassDef, ast.If, ast.Try, ast.excepthandler)):
        for field in ("body", "orelse", "finalbody", "handlers"):
            for child in getattr(stmt, field, None) or ():
                if _side_effects(child):
                    return True
        return False
    return True


def _is_docstring(stmt, index) -> bool:
    return (index == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


class Differ:
    def __init__(self, old_lines: list[str] | None = None, new_lines: list[str] | None = None):
        self.changes: list[Change] = []
        self._seen: set = set()
        self.old = old_lines
        self.new = new_lines

    def same(self, x, y) -> bool:
        """Do two nodes describe the same code? Identical source text at the same columns
        settles it without comparing the trees."""
        if (self.old is not None and isinstance(x, ast.AST) and type(x) is type(y)
                and hasattr(x, "end_lineno") and hasattr(y, "end_lineno")):
            xa, ya = _first_line(x), _first_line(y)
            if (x.end_lineno - xa == y.end_lineno - ya
                    and x.col_offset == y.col_offset and x.end_col_offset == y.end_col_offset
                    and self.old[xa - 1:x.end_lineno] == self.new[ya - 1:y.end_lineno]):
                return True
        return _dump(x) == _dump(y)

    def key(self, x, lines) -> object:
        """Alignment key of a list element: its source lines for statements, else its tree."""
        if lines is not None and isinstance(x, ast.stmt):
            return (type(x).__name__, tuple(lines[_first_line(x) - 1:x.end_lineno]))
        return _dump(x)

    # -- reporting ---------------------------------------------------------
    def report(self, ctx: _Ctx, old_node=None, new_node=None, note="", whole_func=False, span=None,
               ident=None):
        names = _names_of(old_node, new_node)
        cls = ".".join(ctx.cls) if ctx.cls else None
        if ctx.scope == "call":
            fn = ctx.func
            if whole_func or ctx.stmt is None:
                kind, lines = "func", (_first_line(fn), fn.end_lineno)
            else:
                kind = "call"
                if span is not None:
                    lines = span
                elif isinstance(fn, ast.Lambda) and _inside(fn, ctx.stmt):
                    # The statement holds the lambda; the lambda's own body is what runs later.
                    lines = (fn.body.lineno, fn.body.end_lineno)
                elif ctx.in_header:
                    lines = _header_span(ctx.stmt)
                else:
                    lines = (_first_line(ctx.stmt), ctx.stmt.end_lineno)
            ch = Change(kind, lines, cls=cls, func=_func_span(fn), names=names, note=note,
                        annotation=ctx.annotation)
        else:
            stmt = ctx.stmt
            lines = span or ((_first_line(stmt), stmt.end_lineno) if stmt is not None else (1, 1))
            # A change in a def's signature or decorators: the function object itself changes.
            fn = _func_span(stmt) if isinstance(stmt, FUNC) else None
            ch = Change("import", lines, cls=cls, target=ctx.target, key=ctx.key, func=fn,
                        names=names, note=note, annotation=ctx.annotation,
                        side_effects=_side_effects(stmt) or _side_effects(new_node))
        ch.ident = ident
        sig = (ch.kind, ch.lines, ch.cls, ch.target, ch.key, ch.annotation, ident)
        if sig not in self._seen:
            self._seen.add(sig)
            self.changes.append(ch)

    # -- traversal ---------------------------------------------------------
    def nodes(self, a, b, ctx: _Ctx):
        if type(a) is not type(b):
            self.report(ctx, a, b, note="node type")
            return
        if (isinstance(a, ast.Compare) and len(a.ops) == 1 and len(b.ops) == 1
                and (type(a.ops[0]), type(b.ops[0])) in IDENTITY_SWAPS
                and _dump(a.left) == _dump(b.left) and _dump(a.comparators[0]) == _dump(b.comparators[0])):
            self.report(ctx, a, b, note="identity to equality", ident=(a.lineno, a.col_offset))
            return
        if isinstance(a, ast.Dict) and len(a.keys) == len(b.keys) and all(
            self.same(x, y) for x, y in zip(a.keys, b.keys)
        ):
            for k, va, vb in zip(a.keys, a.values, b.values):
                if not self.same(va, vb):
                    c = ctx.copy()
                    if c.key is None and k is not None:
                        try:
                            c.key = ast.unparse(k)
                        except Exception:
                            c.key = None
                    self.nodes(va, vb, c)
            return
        for field in a._fields:
            va, vb = getattr(a, field, None), getattr(b, field, None)
            if isinstance(va, list) or isinstance(vb, list):
                self.lists(va or [], vb or [], a, field, ctx)
            elif isinstance(va, ast.AST) or isinstance(vb, ast.AST):
                if not (isinstance(va, ast.AST) and isinstance(vb, ast.AST)):
                    self.report(self.child_ctx(ctx, a, field, va), va, vb, note=f"{field} added/removed")
                elif not self.same(va, vb):
                    self.nodes(va, vb, self.child_ctx(ctx, a, field, va))
            elif (type(va), va) != (type(vb), vb):
                self.report(ctx, a, b, note=f"{type(a).__name__}.{field}")

    def child_ctx(self, ctx: _Ctx, parent, field, child) -> _Ctx:
        c = ctx.copy()
        if (isinstance(parent, ast.arg) and field == "annotation") or (
            isinstance(parent, FUNC) and field == "returns"
        ) or (isinstance(parent, ast.AnnAssign) and field == "annotation" and ctx.scope == "call"):
            c.annotation = True
        if isinstance(parent, FUNC) and field == "body":
            c.scope, c.func, c.stmt, c.target, c.key = "call", parent, None, None, None
        elif isinstance(parent, ast.Lambda) and field == "body":
            c.scope, c.func, c.target, c.key = "call", parent, None, None
        elif isinstance(parent, ast.ClassDef) and field == "body":
            c.cls = ctx.cls + (parent.name,)
            if c.scope == "import":
                c.stmt, c.target, c.key = None, None, None
        if isinstance(child, (ast.stmt, ast.excepthandler)):
            c.stmt = child
            c.in_header = True
            if c.scope == "import":
                c.target = _assign_target(child)
                c.key = None
        return c

    def lists(self, la, lb, parent, field, ctx: _Ctx):
        is_body = field in BODY_FIELDS and all(isinstance(x, (ast.stmt, ast.excepthandler, ast.match_case))
                                               for x in la + lb)
        if len(la) == len(lb):
            for i, (x, y) in enumerate(zip(la, lb)):
                if self.same(x, y):
                    continue
                if is_body and _is_docstring(x, i) and _is_docstring(y, i):
                    continue
                if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                    c = self.child_ctx(ctx, parent, field, x)
                    self.nodes(x, y, c)
                else:
                    self.report(ctx, parent, None, note=f"{field}[{i}]")
            return
        da, db = [self.key(x, self.old) for x in la], [self.key(x, self.new) for x in lb]
        sm = SequenceMatcher(None, da, db, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            if not is_body:
                self.report(ctx, parent, None, note=f"{field} list changed")
                continue
            if tag == "replace" and (i2 - i1) == (j2 - j1):
                for x, y in zip(la[i1:i2], lb[j1:j2]):
                    if isinstance(x, ast.AST) and isinstance(y, ast.AST):
                        self.nodes(x, y, self.child_ctx(ctx, parent, field, x))
                continue
            # Statements appeared or disappeared in this block.
            block_ctx = self.child_ctx(ctx, parent, field, la[i1] if i1 < len(la) else (la[-1] if la else None))
            if block_ctx.scope == "call":
                self.report(block_ctx, parent, None, note=f"{field} statements {tag}", whole_func=True)
            else:
                olds = la[i1:i2]
                news = lb[j1:j2]
                if olds:
                    for x in olds:
                        c = self.child_ctx(ctx, parent, field, x)
                        self.report(c, x, None, note=f"statement {tag}")
                for y in news:
                    c = ctx.copy()
                    if isinstance(parent, ast.ClassDef):
                        c.cls = ctx.cls + (parent.name,)
                    c.stmt = None
                    c.target = _assign_target(y)
                    anchor = la[i1 - 1] if i1 > 0 else (la[i1] if i1 < len(la) else parent)
                    lo = getattr(anchor, "lineno", 1)
                    self.report(c, None, y, note="statement inserted", span=(lo, getattr(anchor, "end_lineno", lo)))


def _inside(inner, outer) -> bool:
    return (getattr(outer, "lineno", 0) <= inner.lineno
            and inner.end_lineno <= getattr(outer, "end_lineno", 0))


def _blank(line: str) -> bool:
    s = line.strip()
    return not s or s.startswith("#")


def diff_region(old_src: str, new_src: str, edits: list, outline: dict) -> list[Change] | None:
    """Fast path of diff_sources: parse only the statements the edits fall in.

    outline is the pinned file's statement outline (tools/build_outline.py); edits are
    the changed line runs from patch.apply_hunks. Returns None when the edits do not sit
    inside single statements, and the caller must then diff the whole file."""
    old_lines = old_src.split("\n")
    new_lines = new_src.split("\n")
    nodes = outline["nodes"]
    classes = outline["classes"]
    touched: dict[int, list] = {}
    for old_lo, old_hi, new_lo, new_hi in edits:
        best = None
        for idx, (lo, hi, _col, _cls, _is_class) in enumerate(nodes):
            if old_hi >= old_lo:
                inside = lo <= old_lo and old_hi <= hi
            else:  # insertion before old line old_lo
                inside = lo < old_lo <= hi
            if inside and (best is None or lo >= nodes[best][0]):
                best = idx
        if best is None:
            # Between statements: harmless only if nothing but blank lines / comments moved.
            if all(_blank(x) for x in old_lines[old_lo - 1:old_hi]) and all(
                _blank(x) for x in new_lines[new_lo - 1:new_hi]
            ):
                continue
            return None
        if nodes[best][4]:
            return None   # in a class header or between its members
        touched.setdefault(best, []).append((old_lo, old_hi, new_lo, new_hi))
    changes: list[Change] = []
    for idx, node_edits in touched.items():
        lo, hi, col, cls_idx, _ = nodes[idx]
        # Position of the statement in the new file: shifted by the edits before it,
        # stretched by the edits inside it.
        shift = sum((e[3] - e[2] + 1) - (e[1] - e[0] + 1) for e in edits if e[1] < lo and e[0] <= lo)
        grow = sum((e[3] - e[2] + 1) - (e[1] - e[0] + 1) for e in node_edits)
        old_snip = old_lines[lo - 1:hi]
        new_snip = new_lines[lo - 1 + shift:hi + shift + grow]
        pad = 1 if col > 0 else 0
        if pad:
            old_snip = ["if 1:"] + old_snip
            new_snip = ["if 1:"] + new_snip
        try:
            old_tree = ast.parse("\n".join(old_snip))
            new_tree = ast.parse("\n".join(new_snip))
        except (SyntaxError, ValueError, RecursionError):
            return None
        if pad:
            if not (len(old_tree.body) == 1 and len(new_tree.body) == 1
                    and isinstance(old_tree.body[0], ast.If) and isinstance(new_tree.body[0], ast.If)
                    and not old_tree.body[0].orelse and not new_tree.body[0].orelse):
                return None
            old_body, new_body = old_tree.body[0].body, new_tree.body[0].body
        else:
            old_body, new_body = old_tree.body, new_tree.body
        if len(old_body) != 1 or len(new_body) != 1 or type(old_body[0]) is not type(new_body[0]):
            return None
        s_old, s_new = old_body[0], new_body[0]
        if s_old.col_offset != col or s_new.col_offset != col:
            return None
        if all(isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str)
               for x in (s_old, s_new)):
            continue    # a bare string (a docstring, or a statement without effect) was edited
        d = Differ(old_snip, new_snip)
        ctx = _Ctx()
        chain = classes[cls_idx]
        ctx.cls = tuple(chain.split(".")) if chain else ()
        ctx.stmt = s_old
        ctx.target = _assign_target(s_old)
        if not d.same(s_old, s_new):
            d.nodes(s_old, s_new, ctx)
        off = lo - 1 - pad
        top = None
        if type(s_old) is ast.FunctionDef and s_old.name == s_new.name:
            top = {"name": s_old.name, "cls": chain, "col": col, "old": (lo, hi),
                   "new": (lo + shift, hi + shift + grow)}
        for ch in d.changes:
            ch.lines = (ch.lines[0] + off, ch.lines[1] + off)
            if ch.ident is not None:
                ch.ident = (ch.ident[0] + off, ch.ident[1])
            if ch.func is not None:
                ch.func = (ch.func[0], ch.func[1] + off, ch.func[2] + off)
            if ch.kind in ("call", "func"):
                ch.top = top
            changes.append(ch)
    return changes


def diff_sources(old_src: str, new_src: str) -> tuple[list[Change], str | None]:
    """Returns (changes, syntax_error). An empty change list means the two
    versions compile to the same program (comments, whitespace, docstrings)."""
    old_tree = ast.parse(old_src)
    try:
        new_tree = ast.parse(new_src)
    except (SyntaxError, ValueError, RecursionError) as e:
        return [], f"{type(e).__name__}: {e}"
    if old_src == new_src:
        return [], None
    d = Differ(old_src.split("\n"), new_src.split("\n"))
    d.nodes(old_tree, new_tree, _Ctx())
    return d.changes, None
