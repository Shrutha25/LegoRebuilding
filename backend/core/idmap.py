"""Map Brickognize (BrickLink-style) part IDs to Rebrickable part numbers.

Order: exact -> strip trailing letter variant (3001a -> 3001) -> strip BrickLink print
suffix (26603pb509 -> 26603) -> append "a" (BrickLink 41879 is Rebrickable 41879a)
-> Rebrickable API bricklink_id lookup -> unmapped.
"""

import re

_PRINT_SUFFIX = re.compile(r"(pb|pr)\d+$")
_TRAILING_LETTERS = re.compile(r"[a-z]+$")


def _candidates(part):
    """Local rewrites of a part id, most specific first (the id itself excluded)."""
    out = []
    bases = [part]
    stripped = _PRINT_SUFFIX.sub("", part)
    if stripped != part:
        bases.append(stripped)
    for b in bases:
        letterless = _TRAILING_LETTERS.sub("", b)
        if letterless and letterless != b:
            out.append(letterless)
        if b is not part:
            out.append(b)
        if not b.endswith("a") and not b[-1:].isalpha():
            out.append(b + "a")
    return [c for c in out if c != part]


def resolve_local(part, known):
    """Rebrickable id for `part` using only the parts table, or None.
    `known` maps lowercase part number -> actual Rebrickable part number."""
    part = part.strip().lower()
    if part in known:
        return known[part]
    for c in _candidates(part):
        if c in known:
            return known[c]
    return None


async def resolve(part, known, api_lookup=None):
    """Full chain. Returns (rebrickable_id, unmapped). `api_lookup` is an async callable
    bricklink_id -> rebrickable id or None."""
    hit = resolve_local(part, known)
    if hit:
        return hit, False
    if api_lookup is not None:
        try:
            hit = await api_lookup(part.strip().lower())
        except Exception:
            hit = None
        if hit:
            return hit, False
    return part, True
