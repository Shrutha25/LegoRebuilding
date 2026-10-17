"""Realistic inventories for tests, the gallery and the offline QA: no LDraw library, no Rebrickable needed.

Part numbers are the real ones of the plain bricks and plates the 3D grid supports (partdims.KNOWN); colours carry their LDraw code and
a display hex (the same values LDConfig uses for these colours).
"""

import random

from backend.core.buildmodel import Inventory, InvItem
from backend.core.partdims import KNOWN

COLOURS = {                       # name -> (LDraw code, hex)
    "Red": (4, "#C91A09"), "Blue": (1, "#0055BF"), "Yellow": (14, "#F2CD37"), "Green": (2, "#237841"), "White": (15, "#FFFFFF"),
    "Black": (0, "#05131D"), "Light Bluish Gray": (71, "#A0A5A9"), "Dark Bluish Gray": (72, "#6C6E68"), "Orange": (25, "#FE8A18"),
    "Tan": (19, "#E4CD9E"), "Dark Red": (320, "#720E0F"), "Lime": (27, "#BBE90B"), "Medium Azure": (322, "#36AEBF"),
    "Reddish Brown": (70, "#582A12"),
}

# (part number, name, how common): the usual contents of a mixed bucket
COMMON = [
    ("3005", "Brick 1 x 1", 5), ("3004", "Brick 1 x 2", 6), ("3622", "Brick 1 x 3", 3), ("3010", "Brick 1 x 4", 4),
    ("3009", "Brick 1 x 6", 2), ("3003", "Brick 2 x 2", 6), ("3002", "Brick 2 x 3", 3), ("3001", "Brick 2 x 4", 5),
    ("2456", "Brick 2 x 6", 2), ("3024", "Plate 1 x 1", 4), ("3023", "Plate 1 x 2", 5), ("3623", "Plate 1 x 3", 2),
    ("3710", "Plate 1 x 4", 3), ("3666", "Plate 1 x 6", 2), ("3022", "Plate 2 x 2", 4), ("3020", "Plate 2 x 4", 4),
    ("3795", "Plate 2 x 6", 2), ("3034", "Plate 2 x 8", 1), ("3460", "Plate 1 x 8", 1),
]


def make_item(part_id, name, colour, qty):
    from backend.core.partdims import dims_for
    code = COLOURS[colour][0]
    return InvItem(part_id, name, colour, qty, dims_for(part_id, name), part_id, code)


def hex_of(code):
    for _, (c, h) in COLOURS.items():
        if c == code:
            return h
    return "#9BA19D"


def random_inventory(rng, size=None):
    """A random but realistic inventory: 2-5 colours, 20-120 plain bricks and plates, common sizes plentiful."""
    if not isinstance(rng, random.Random):
        rng = random.Random(rng)
    colours = rng.sample(list(COLOURS), rng.randint(2, 5))
    weights = [3, 2, 2, 1, 1][:len(colours)]
    target = size or rng.randint(30, 130)
    names, qty = {}, {}
    pool = [(pn, nm) for pn, nm, w in COMMON for _ in range(w)]
    for _ in range(target):
        pn, nm = rng.choice(pool)
        key = (pn, rng.choices(colours, weights)[0])
        names[key] = nm
        qty[key] = qty.get(key, 0) + 1
    return Inventory([make_item(pn, names[(pn, col)], col, q) for (pn, col), q in sorted(qty.items())])


def inventory_from(spec):
    """[(part number, colour, qty)] -> Inventory; names come from partdims' table (Brick/Plate sizes)."""
    out = []
    for pn, colour, qty in spec:
        d = KNOWN[pn]
        name = f"{d.kind.title()} {d.long} x {d.short}"
        out.append(make_item(pn, name, colour, qty))
    return Inventory(out)
