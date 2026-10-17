"""Recipe building blocks: regions (what to fill), roles (colour slots) and the Recipe record.

Coordinates are the ones of buildmodel.py: x, z in studs (x right, z towards the viewer), y in plates (a brick is 3 plates, a plate 1).
A recipe never places a part. It describes REGIONS: sets of stud cells, layer by layer, with a role (a colour slot). The filler
(filler.py) turns the regions into parts from the user's inventory.

Region kinds (all end up as a list of cell sets, one per layer, so the filler has one job):
    solid    every cell of a w x d rectangle, for each layer
    hollow   only the walls (thickness 1 or 2), with openings (a door, a window) cut out of them
    stepped  each layer shrinks by `shrink` studs per side (roofs, noses); `axes` says which sides ("xz", "x" or "z")
    cells    any explicit set of cells (wheels, crenellations, fins)
`unit` is the height of one layer: "brick" = 3 plates, "plate" = 1 plate.
"""

from dataclasses import dataclass, field
from typing import Callable

UNIT_HEIGHT = {"brick": 3, "plate": 1}


@dataclass(frozen=True)
class Opening:
    """A gap in a hollow region: on `side` ("front" = highest z, "back", "left" = lowest x, "right"), `offset` studs from the wall's
    start, `width` studs wide, for layers layer_from .. layer_to (inclusive)."""
    side: str
    offset: int
    width: int
    layer_from: int
    layer_to: int


@dataclass
class Region:
    name: str
    role: str
    y: int                                   # bottom of the first layer, in plates
    layers: list                             # one frozenset of (x, z) per layer
    unit: str = "brick"
    required: bool = True
    min_coverage: float = 0.9                # share of the cells that must be filled for a required region
    kind: str = "solid"
    prefer: str | None = None                # part shape wanted: "slope" and "tile" ONLY (slopes also need `descent`); "round"/"cone" first, rectangles if short
    descent: str | None = None               # slope regions: the direction ("+z") the roof surface falls towards

    @property
    def layer_height(self):
        return UNIT_HEIGHT[self.unit]

    @property
    def cell_count(self):
        return sum(len(c) for c in self.layers)

    @property
    def top(self):
        return self.y + len(self.layers) * self.layer_height


def rect(x, z, w, d):
    return frozenset((x + i, z + j) for i in range(w) for j in range(d))


def solid(name, role, x, z, w, d, y, layers=1, unit="brick", required=True, min_coverage=0.9):
    return Region(name, role, y, [rect(x, z, w, d)] * layers, unit, required, min_coverage, "solid")


def hollow(name, role, x, z, w, d, y, layers=1, thickness=1, openings=(), unit="brick", required=True, min_coverage=0.9):
    """Walls only. Thickness 2 on a narrow footprint simply becomes solid."""
    ring = rect(x, z, w, d) - rect(x + thickness, z + thickness, max(0, w - 2 * thickness), max(0, d - 2 * thickness))
    out = []
    for k in range(layers):
        cells = set(ring)
        for o in openings:
            if o.layer_from <= k <= o.layer_to:
                cells -= _opening_cells(o, x, z, w, d, thickness)
        out.append(frozenset(cells))
    return Region(name, role, y, out, unit, required, min_coverage, "hollow")


def _opening_cells(o, x, z, w, d, t):
    if o.side in ("front", "back"):
        rows = range(z + d - t, z + d) if o.side == "front" else range(z, z + t)
        return {(cx, cz) for cx in range(x + o.offset, x + o.offset + o.width) for cz in rows}
    cols = range(x, x + t) if o.side == "left" else range(x + w - t, x + w)
    return {(cx, cz) for cz in range(z + o.offset, z + o.offset + o.width) for cx in cols}


def stepped(name, role, x, z, w, d, y, layers=None, shrink=1, axes="xz", unit="plate", required=True, min_coverage=0.9):
    """Layer k is the rectangle shrunk by k*shrink studs on each side (only the sides named in `axes`). Stops when nothing is left
    (or after `layers` layers)."""
    out, k = [], 0
    while layers is None or k < layers:
        sx = k * shrink if "x" in axes else 0
        sz = k * shrink if "z" in axes else 0
        cw, cd = w - 2 * sx, d - 2 * sz
        if cw <= 0 or cd <= 0:
            break
        out.append(rect(x + sx, z + sz, cw, cd))
        k += 1
    return Region(name, role, y, out, unit, required, min_coverage, "stepped")


def cells_region(name, role, y, layers_cells, unit="brick", required=True, min_coverage=0.9):
    return Region(name, role, y, [frozenset(c) for c in layers_cells], unit, required, min_coverage, "cells")


@dataclass(frozen=True)
class Role:
    """A colour slot. `want` steers the automatic assignment: major (the colour you own most of), second, dark, neutral."""
    name: str
    want: str = "major"


@dataclass
class Recipe:
    id: str
    category: str
    nouns: tuple                              # words that make a name fit this recipe ("car", "racer", ...)
    roles: tuple                              # of Role
    params: dict                              # name -> (low, high), integers
    generate: Callable                        # (params: dict, facts: InvFacts) -> list[Region]
    tags: tuple = ()
    min_parts: int = 8                        # the inventory needs at least this many bricks/plates ...
    min_long: int = 2                         # ... and one with a side at least this long
    min_area: int = 0                         # ... and at least this many studs of bricks/plates in total
    min_build_parts: int | None = None        # the finished build must have at least this many parts (default: min_parts)
    variants: Callable | None = None          # (facts) -> list of param dicts to try (defaults to a spread over the ranges)
    notes: str = ""
    title: str = ""                           # what the recipe makes, for hints ("Bridge")
    names: tuple = ()                         # evocative names used when no concept names the build
    silhouettes: dict = field(default_factory=dict)   # silhouette -> {param: "lo" | "mid" | "hi" | int}
    binds: dict = field(default_factory=dict)         # param -> "w" | "d" | "h": which side of a concept's bounding box it follows
    needs: tuple = ()                         # capabilities the inventory must offer; "wheels" is not supported yet, so such recipes are never offered
    features: tuple = ()                      # features every build of this recipe has (checked on the finished build)
    symmetry: str = ""                        # "x", "z", "xz" or "": the build mirrors itself about that axis
    spec: dict | None = None                  # the declarative JSON this recipe was compiled from
    allowed: Callable | None = None           # (params, facts) -> bool: is this parameter setting allowed (the spec's constraint)
    exact_variants: bool = False              # the spec lists its variants explicitly: only those settings are used

    def role_names(self):
        return [r.name for r in self.roles]

    def default_params(self):
        return {k: (lo + hi) // 2 for k, (lo, hi) in self.params.items()}

    def check_params(self, params):
        for k, (lo, hi) in self.params.items():
            v = params.get(k)
            if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
                raise ValueError(f"{self.id}: parameter {k}={v!r} is outside {lo}..{hi}")
