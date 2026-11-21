"""When does replacing `is` by `==` (or `is not` by `!=`) change nothing?

The map maps/cmp.json.z (tools/tracer/cmptrace.py) lists, for every identity comparison in
the package, the pairs of operand classes the test suite produces there on the pinned
commit, and how each class compares for equality. For many pairs the two operators cannot
disagree: one side is None and the other side's class never equals None, both sides are
members of one enumeration, a class compares by identity, the classes belong to different
built-in families ... (`agree` below).

If every pair seen at a comparison is of that kind, a patch that only swaps the operator
there leaves every test exactly as it was: until the comparison is evaluated the patched
program has done what the pinned one does, so the operands are the ones recorded, the new
operator gives the old answer, and so on to the end of the test. No test needs re-running.

Comparisons inside __eq__ / __hash__ and the like are not in the map (there `==` could
re-enter the method being changed), nor are chained comparisons.
"""
from __future__ import annotations

VALUE_FAMILIES = ("num", "str", "bytes", "tuple", "list", "dict", "set", "date", "time", "timedelta", "bmeth")


def agree(name_a: str, kind_a: str, name_b: str, kind_b: str) -> bool:
    """Do `x is y` and `x == y` always give the same answer for x of class a and y of class b?

    kinds (see cmptrace.classify): none, bool, id (compares by identity), a built-in value
    family (optionally "+uniq": an enumeration with that mixin), exact (a class of the
    package whose __eq__ accepts only its own exact class), notnone (the dialect classes
    and their instances: their hand-read __eq__ only accepts strings and dialects, so they
    equal neither None nor an object that compares by identity), unknown."""
    if kind_a == "unknown" or kind_b == "unknown":
        return False
    if kind_a == "notnone" or kind_b == "notnone":
        other = kind_b if kind_a == "notnone" else kind_a
        return other in ("none", "id")
    if name_a == name_b:
        # Same class on both sides: equal objects must be the same object.
        return kind_a in ("none", "bool", "id") or kind_a.endswith("+uniq")
    # Different classes: never the same object, so `==` must never hold either.
    if kind_a in ("none", "id") or kind_b in ("none", "id"):
        return True     # None / an identity-comparing object equals nothing else, and the
        #                 other side (a built-in family or an exact class) rejects strangers
    fam_a = "num" if kind_a == "bool" else kind_a.split("+")[0]
    fam_b = "num" if kind_b == "bool" else kind_b.split("+")[0]
    if fam_a == "exact" and fam_b == "exact":
        return True     # two different exact classes of the package
    if fam_a not in VALUE_FAMILIES + ("exact",) or fam_b not in VALUE_FAMILIES + ("exact",):
        return False
    return fam_a != fam_b


def same_outcome(cmp_map: dict | None, rel: str, line: int, col: int) -> str | None:
    """A sentence saying why swapping the operator of the identity comparison at
    rel:line:col changes nothing, or None when that cannot be shown."""
    if not cmp_map:
        return None
    pairs = cmp_map.get("sites", {}).get(rel, {}).get(f"{line}:{col}")
    if not pairs:
        return None
    types = cmp_map["types"]
    for a, b in pairs:
        (name_a, kind_a), (name_b, kind_b) = types[a], types[b]
        if not agree(name_a, kind_a, name_b, kind_b):
            return None
    return (f"identity comparison: `==` cannot differ from `is` for the {len(pairs)} pair(s) of "
            f"operand classes the suite produces here")
