"""Part number -> role in a model (brick, plate, tile, slope, round, cone, arch, wheel, wheelbase ...).

The table is backend/data/part_roles.json, generated from Rebrickable's parts.csv and part_categories.csv by
`python -m backend.core.fetch_part_roles` (committed). Parts that play no building role (minifigs, stickers, Technic, Duplo ...)
are not in the file and get "other".

    role_of("3001")                -> "brick"
    role_of("3040b")               -> "slope"
    role_of("9999", "Plate 2 x 4") -> "plate"        (the name decides for parts Rebrickable does not list)
"""

import json
import re
from functools import lru_cache
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "data" / "part_roles.json"
_VARIANT = re.compile(r"(pr\d+|pat\d+|c\d+[a-z]*\d*|px\d+|pb\d+|[a-z]\d*)$", re.I)

# groups used by the recipes: which roles can stand in for a "plate-like", "brick-like" ... need
PLATE_LIKE = ("plate",)
BRICK_LIKE = ("brick",)
SMOOTH = ("tile",)
SLOPES = ("slope",)
ROUND = ("round", "round_plate", "cone")
WHEEL_UNIT = ("wheelbase",)


@lru_cache(maxsize=1)
def _table():
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))["parts"]
    except (OSError, ValueError, KeyError):
        return {}


def role_of(part_num, name=None):
    table = _table()
    key = str(part_num or "").strip()
    role = table.get(key)
    if role is None:                                   # printed / patterned / mould variants: try the base number
        base = key
        while base and role is None:
            m = _VARIANT.search(base)
            if not m or m.start() == 0:
                break
            base = base[:m.start()]
            role = table.get(base)
    if role is None and name:
        from backend.core.partdims import dims_from_name
        d = dims_from_name(name)
        role = d.kind if d else None
    return role or "other"
