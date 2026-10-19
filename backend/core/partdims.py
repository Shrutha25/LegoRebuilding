"""Dimensions of the parts an AI build may use: the ONE place that knows footprint and height.

Only plain rectangular bricks, plates and tiles are supported: their footprint (studs) and height (plates: brick 3, plate 1,
tile 1) are known exactly and their LDraw origin is the same everywhere: the middle of the footprint, at the TOP of the body,
the long side along X (checked against the real LDraw geometry in tests/test_buildmodel.py).

    long   studs along X at rotation 0   (>= short)
    short  studs along Z at rotation 0
    height in plates (1 plate = 8 LDU, 1 brick = 3 plates = 24 LDU)
    studs_top  False for tiles: nothing can be placed on top of a tile

`dims_for(part_num, name)` uses the explicit table first (LDraw / Rebrickable part numbers), then the part's name
("Brick 2 x 4", "Plate 1 x 6", "Tile 2 x 2 with Groove"), so any plain brick, plate or tile in the catalogue works.
"""

import re
from dataclasses import dataclass

STUD_LDU = 20            # horizontal size of one stud cell
PLATE_LDU = 8            # height of one plate
BRICK_PLATES = 3


@dataclass(frozen=True)
class Dims:
    long: int
    short: int
    height: int
    studs_top: bool = True
    kind: str = "brick"
    shape: str = "rect"            # rect | round | cone | slope (what the part looks like; the grid only needs its footprint)
    box: tuple | None = None       # (x0, x1, z0, z1) in LDU at rotation 0 for parts whose LDraw origin is NOT the middle of the footprint
    descent: str | None = None     # slopes: the direction ("-z") the surface falls towards at rotation 0
    oy: int | None = None          # wheels: LDU from the bottom of the part up to its LDraw origin (the origin is the wheel's centre, not the top)

    def footprint(self, rotation):
        """(studs along X, studs along Z) for a rotation of 0/90/180/270 degrees."""
        if self.box is not None:
            x0, x1, z0, z1 = _turn_box(self.box, rotation)
            return (round((x1 - x0) / STUD_LDU), round((z1 - z0) / STUD_LDU))
        return (self.long, self.short) if rotation % 180 == 0 else (self.short, self.long)

    def origin(self, rotation):
        """LDU from the footprint's minimum corner (low x, low z) to the part's LDraw origin, for this rotation."""
        if self.box is not None:
            x0, _, z0, _ = _turn_box(self.box, rotation)
            return (-x0, -z0)
        fx, fz = self.footprint(rotation)
        return (fx * STUD_LDU / 2, fz * STUD_LDU / 2)

    def descent_at(self, rotation):
        """Slopes: the world direction ("+x", "-z" ...) the surface falls towards when the part is turned by `rotation`."""
        if not self.descent:
            return None
        vx, vz = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}[self.descent]
        c, s = _cos_sin(rotation)
        wx, wz = c * vx - s * vz, s * vx + c * vz
        return {(1, 0): "+x", (-1, 0): "-x", (0, 1): "+z", (0, -1): "-z"}[(wx, wz)]


def _cos_sin(rotation):
    return {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[rotation % 360]


def _turn_box(box, rotation):
    """The LDU box turned about the vertical axis like ldrconv turns a part (x' = c*x - s*z, z' = s*x + c*z)."""
    x0, x1, z0, z1 = box
    c, s = _cos_sin(rotation)
    xs = [c * x - s * z for x in (x0, x1) for z in (z0, z1)]
    zs = [s * x + c * z for x in (x0, x1) for z in (z0, z1)]
    return (min(xs), max(xs), min(zs), max(zs))


def _brick(long, short):
    return Dims(long, short, BRICK_PLATES, True, "brick")


def _plate(long, short):
    return Dims(long, short, 1, True, "plate")


def _tile(long, short):
    return Dims(long, short, 1, False, "tile")


def _round(side, plate=False):
    return Dims(side, side, 1 if plate else BRICK_PLATES, True, "plate" if plate else "brick", "round")


def _round_tile(side):
    return Dims(side, side, 1, False, "tile", "round")


def _cone(side):
    return Dims(side, side, BRICK_PLATES, True, "brick", "cone")


def _slope(fx, fz):
    """A 45/33 degree slope brick (one brick tall) with its LDraw footprint fx x fz studs at rotation 0 and the origin on the stud
    row at the HIGH end (+z), so the surface falls towards -z. Nothing can be placed on top of it."""
    return Dims(max(fx, fz), min(fx, fz), BRICK_PLATES, False, "brick", "slope",
                (-fx * 10, fx * 10, -fz * 20 + 10, 10), "-z")


# LDraw / Rebrickable part number -> dims. Every entry is verified against the LDraw geometry in the tests.
KNOWN = {
    # bricks
    "3005": _brick(1, 1), "3004": _brick(2, 1), "3622": _brick(3, 1), "3010": _brick(4, 1), "3009": _brick(6, 1),
    "3008": _brick(8, 1), "3003": _brick(2, 2), "3002": _brick(3, 2), "3001": _brick(4, 2), "2456": _brick(6, 2),
    "3007": _brick(8, 2), "3006": _brick(10, 2),
    # plates
    "3024": _plate(1, 1), "3023": _plate(2, 1), "3623": _plate(3, 1), "3710": _plate(4, 1), "3666": _plate(6, 1),
    "3460": _plate(8, 1), "4477": _plate(10, 1), "3022": _plate(2, 2), "3021": _plate(3, 2), "3020": _plate(4, 2),
    "3795": _plate(6, 2), "3034": _plate(8, 2), "3832": _plate(10, 2), "3031": _plate(4, 4), "3032": _plate(6, 4),
    "3035": _plate(8, 4), "3030": _plate(10, 4), "3958": _plate(6, 6), "3036": _plate(8, 6), "3033": _plate(10, 6),
    # tiles
    "3070b": _tile(1, 1), "3069b": _tile(2, 1), "3068b": _tile(2, 2), "63864": _tile(3, 1), "2431": _tile(4, 1),
    "6636": _tile(6, 1),
    # round bricks and plates, a cone and round tiles (footprint n x n, origin in the middle like the rectangular parts)
    "3062b": _round(1), "3941": _round(2), "6143": _round(2), "87081": _round(4),
    "4032a": _round(2, True), "60474": _round(4, True),
    "4589": _cone(1), "98138": _round_tile(1), "14769": _round_tile(2), "4150": _round_tile(2),
}

# Slopes: origin on the high stud row, so they carry their LDraw box. 3040b 2x1 is 1 stud wide in LDraw's x, 2 deep in z.
KNOWN.update({"3040b": _slope(1, 2), "3039": _slope(2, 2), "3037": _slope(4, 2), "3038": _slope(3, 2), "3298": _slope(2, 3),
              "4286": _slope(1, 3)})

# Wheels. A wheel set is one 4600-style holder + two rims + two tyres standing on the ground, 4 studs wide and 2 deep (geometry from OMR 1478-1,
# verified against LDraw in tests/test_wheels.py). The holder's plate sits 3 plates up with a wheel on each side of it, so in the grid:
#   holder  2 x 2 studs, 3 plates tall (the empty space under its plate is reserved), studs on top, origin at the top of its plate
#   tyre    1 stud wide, 2 deep, 5 plates tall (36 LDU in reality), origin = its centre, 18 LDU above the ground; its axle runs along X at rotation 0
#   rim     the same cells as its tyre and no height of its own (it sits inside the tyre)
WHEEL_HOLDER_PLATES, WHEEL_PLATES = 3, 5
_WHEEL_BOX = (-20, 20, -10, 10)
KNOWN.update({"4600": Dims(2, 2, WHEEL_HOLDER_PLATES, True, "wheelbase"),
              "3641": Dims(2, 1, WHEEL_PLATES, False, "tyre", "wheel", _WHEEL_BOX, None, 18),
              "4624": Dims(2, 1, 0, False, "rim", "wheel", _WHEEL_BOX, None, 18)})
WHEEL_KINDS = ("wheelbase", "tyre", "rim")

_NAME = re.compile(r"^(Brick|Plate|Tile)\s+(\d+)\s*x\s*(\d+)(\s+with Groove)?$", re.I)
MAX_SIDE = 16


def dims_from_name(name):
    """Dims for 'Brick 2 x 4' / 'Plate 1 x 6' / 'Tile 2 x 2 with Groove'; None for anything else (modified, round, slope...)."""
    m = _NAME.match((name or "").strip())
    if not m:
        return None
    kind, a, b = m.group(1).lower(), int(m.group(2)), int(m.group(3))
    if not 1 <= a <= MAX_SIDE or not 1 <= b <= MAX_SIDE:
        return None
    long, short = max(a, b), min(a, b)
    return {"brick": _brick, "plate": _plate, "tile": _tile}[kind](long, short)


def dims_for(part_num, name=None):
    """Dims of a part, or None when the part is not a plain brick, plate or tile."""
    key = (part_num or "").strip().lower()
    return KNOWN.get(key) or dims_from_name(name)


# ---------------------------------------------------------------- footprints seen on real parts (detection)
# Footprints (short, long) in studs of real bricks, plates, tiles and baseplates. The detector uses them to ask "is this
# blob a real piece, or two pieces merged / one piece in parts?" (detect rule 3e). Anything up to 48 x 48 studs may be a
# baseplate, so size alone is never a reason to reject a piece, but the SHAPE of the footprint must be one that exists.
BASEPLATE_SIDES = (6, 8, 10, 12, 16, 24, 32, 48)


def _real_footprints():
    out = {(d.short, d.long) for d in KNOWN.values()}
    out.update((1, n) for n in range(1, 17))                       # 1 x n beams, bricks, plates, tiles
    out.update((2, n) for n in range(2, 13))
    out.update((3, n) for n in (3, 4, 5, 6, 8, 10))                # 3 x 3 .. 3 x 10 plates / bricks
    out.update((4, n) for n in (4, 5, 6, 8, 10, 12))
    out.update((6, n) for n in (6, 8, 10, 12, 14, 16))
    out.update((8, n) for n in (8, 11, 16))                        # 8 x 8 plate, 8 x 11 plate, 8 x 16 plate
    out.update((n, n) for n in BASEPLATE_SIDES)
    out.update((a, b) for a in BASEPLATE_SIDES for b in BASEPLATE_SIDES if a <= b)
    return frozenset(out)


REAL_FOOTPRINTS = _real_footprints()
MAX_STUDS = 48


def is_real_footprint(a, b):
    """Is an a x b stud footprint (whole studs, either order) a footprint real parts have?"""
    a, b = sorted((int(a), int(b)))
    return (a, b) in REAL_FOOTPRINTS


def nearest_footprint(a, b, tol=0.35):
    """Measured footprint (a x b studs, fractional, either order) -> the real footprint within `tol` studs on both sides, else None."""
    a, b = sorted((float(a), float(b)))
    best = None
    for s, l in REAL_FOOTPRINTS:
        if abs(s - a) <= tol and abs(l - b) <= tol:
            d = abs(s - a) + abs(l - b)
            if best is None or d < best[0]:
                best = (d, (s, l))
    return best[1] if best else None


def partition_footprint(cols, rows):
    """Cut a cols x rows stud rectangle into real footprints with straight (guillotine) cuts along the stud grid.
    -> (fewest pieces, [every optimal partition]); a partition is a list of (col0, row0, cols, rows). A rectangle that is
    itself real is its own (single) partition. Returns (None, []) when it cannot be partitioned."""
    cols, rows = int(cols), int(rows)
    memo = {}

    def best(c, r):
        key = (c, r)
        if key in memo:
            return memo[key]
        if is_real_footprint(c, r):
            memo[key] = (1, [[(0, 0, c, r)]])
            return memo[key]
        n_best, sols = None, []
        for i in range(1, c // 2 + 1):                             # vertical cut after column i (mirror cuts are the same split)
            for n, parts in _combine(best(i, r), best(c - i, r), (i, 0)):
                n_best, sols = _keep(n_best, sols, n, parts)
        for j in range(1, r // 2 + 1):
            for n, parts in _combine(best(c, j), best(c, r - j), (0, j)):
                n_best, sols = _keep(n_best, sols, n, parts)
        memo[key] = (n_best, sols) if n_best else (None, [])
        return memo[key]

    def _combine(left, right, shift):
        if not left[0] or not right[0]:
            return []
        out = []
        for a in left[1][:4]:
            for b in right[1][:4]:
                out.append((left[0] + right[0], a + [(x + shift[0], y + shift[1], w, h) for x, y, w, h in b]))
        return out

    def _keep(n_best, sols, n, parts):
        if n_best is None or n < n_best:
            return n, [parts]
        if n == n_best and len(sols) < 8:
            sols.append(parts)
        return n_best, sols

    return best(cols, rows)
