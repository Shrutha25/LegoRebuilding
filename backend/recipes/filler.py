"""The filler: turns regions into a brick layout using ONLY the user's inventory.

For every layer of every region (required regions first, bottom-up; then the optional ones, each all-or-nothing) it tries several
tilings of the layer's cells and keeps the best:

  * largest part first, in the role's colour first, then in a fallback colour (with a penalty);
  * scanned from either end, with the long side along x or z, and with the first part of each row capped to 1, 2 or 3 studs, so a layer
    can be tiled with its seams offset from the layer below (running bond: no seam of this layer lies on a seam of the one below);
  * a part above the ground must touch studs below (never hang in the air), so the result passes the validator;
  * substitutions fall out of the tiling: a longer part by shorter ones, a 2x4 by two 2x2, and in a brick layer a brick by
    3 stacked plates of the same size and colour.

Stock is tracked per (part, colour); no part is ever used twice. The result is deterministic: same regions, same inventory, same build.
"""

from dataclasses import dataclass, field

from backend.core.buildmodel import Placement
from backend.recipes.facts import usable

LEAD_CAPS = (None, 2, 3, 1)
SCORE_UNCOVERED = 50
SCORE_FALLBACK = 4
SCORE_ALIGNED = 6
SCORE_PART = 1
SCORE_LOOSE = 9                  # a part that nothing above rests on (it would be a loose plate hanging off the model)


@dataclass(frozen=True)
class Piece:
    item: object
    fx: int
    fz: int
    rot: int
    stack: int                      # 1 = the part itself, 3 = three stacked plates replacing a brick

    def __post_init__(self):                       # cached: the tiler asks for these millions of times
        object.__setattr__(self, "area", self.fx * self.fz)
        object.__setattr__(self, "key", (self.item.part_id.lower(), self.item.colour.lower()))
        object.__setattr__(self, "offsets", tuple((i, j) for i in range(self.fx) for j in range(self.fz)))


@dataclass
class RegionResult:
    name: str
    role: str
    required: bool
    covered: int
    total: int
    min_cov: float = 0.9
    kept: bool = True

    @property
    def coverage(self):
        return 1.0 if self.total == 0 else self.covered / self.total


@dataclass
class FillResult:
    placements: list = field(default_factory=list)
    regions: list = field(default_factory=list)          # RegionResult
    owner: list = field(default_factory=list)            # region index of each placement (parallel to placements)
    fallback_parts: int = 0                              # parts not in their role's colour
    role_parts: dict = field(default_factory=dict)       # role -> [parts, parts in the role colour]

    @property
    def required_ok(self):
        return all(r.coverage >= r.min_cov for r in self.regions if r.required)


class _State:
    def __init__(self, stock):
        self.stock = dict(stock)
        self.placements = []
        self.owner = []           # region index of each placement (parallel to placements)
        self.studs_at = {}        # plate height -> cells that have studs sticking up to that height
        self.seams_at = {}        # plate height -> seam edges of the layer whose top is there
        self.fallback = 0
        self.role_parts = {}

    def snapshot(self):
        return (dict(self.stock), list(self.placements), list(self.owner), {k: set(v) for k, v in self.studs_at.items()},
                {k: set(v) for k, v in self.seams_at.items()}, self.fallback, {k: list(v) for k, v in self.role_parts.items()})

    def restore(self, snap):
        self.stock, self.placements, self.owner = dict(snap[0]), list(snap[1]), list(snap[2])
        self.studs_at = {k: set(v) for k, v in snap[3].items()}
        self.seams_at = {k: set(v) for k, v in snap[4].items()}
        self.fallback = snap[5]
        self.role_parts = {k: list(v) for k, v in snap[6].items()}


def _pieces(items, unit, prefer=None, descent=None):
    """Candidate pieces for a layer of `unit` (unsorted).

    Normal regions use plain bricks and plates (a brick may be replaced by 3 stacked plates). A "slope" region uses only slope bricks turned so
    the surface falls towards `descent`; a "tile" region only tiles; "round" and "cone" regions also take round bricks / cones (those first)."""
    out = []
    for it in items:
        d = it.dims
        if prefer == "slope":
            if d.shape != "slope" or unit != "brick":
                continue
            rots = [r for r in (0, 90, 180, 270) if d.descent_at(r) == descent]
            stack = 1
        elif prefer == "tile":
            if d.kind != "tile" or unit != "plate":
                continue
            rots = (0,) if d.long == d.short else (0, 90)
            stack = 1
        else:
            if not d.studs_top or d.shape == "slope" or (d.shape in ("round", "cone") and prefer not in ("round", "cone")):
                continue
            if unit == "brick" and d.kind == "brick":
                stack = 1
            elif unit == "brick" and d.kind == "plate":
                stack = 3
            elif unit == "plate" and d.kind == "plate":
                stack = 1
            else:
                continue
            rots = (0,) if d.long == d.short else (0, 90)
        for r in rots:
            fx, fz = d.footprint(r)
            out.append(Piece(it, fx, fz, r, stack))
    return out


def _sorted(pieces, colour, orient, prefer=None):
    """Best first: the wanted shape (round / cone), role colour before fallback colours, then big before small, long side along `orient`."""
    def key(p):
        shape = 0 if (prefer is None or p.item.dims.shape == prefer) else 1
        tier = 0 if (colour is None or p.item.colour.lower() == colour.lower()) else 1
        along = (p.fx - p.fz) if orient == "x" else (p.fz - p.fx)
        return (shape, tier, -p.area / p.stack, p.stack, -along, p.item.part_id, p.item.colour.lower())      # a plate stack counts for a third
    return sorted(pieces, key=key)


def _seams(tiles):
    """Seam edges of a tiling: ('v', x, z) between cells (x-1, z) and (x, z); ('h', x, z) between (x, z-1) and (x, z)."""
    owner = {}
    for n, (p, px, pz) in enumerate(tiles):
        for i in range(p.fx):
            for j in range(p.fz):
                owner[(px + i, pz + j)] = n
    out = set()
    for (x, z), n in owner.items():
        if owner.get((x - 1, z), n) != n:
            out.add(("v", x, z))
        if owner.get((x, z - 1), n) != n:
            out.add(("h", x, z))
    return out


def _tile_once(cells, y, pieces_sorted, stock, studs_below, reverse, lead_cap):
    free = set(cells)
    order = sorted(cells, key=lambda c: (c[1], c[0]), reverse=reverse)
    local = dict(stock)
    tiles, uncovered = [], 0
    for ax, az in order:
        if (ax, az) not in free:
            continue
        row_start = ((ax + 1, az) if reverse else (ax - 1, az)) not in cells
        chosen = None
        for p in pieces_sorted:
            if lead_cap is not None and row_start and p.fx > lead_cap:
                continue
            px, pz = (ax - p.fx + 1, az - p.fz + 1) if reverse else (ax, az)
            if local.get(p.key, 0) < p.stack:
                continue
            foot = [(px + i, pz + j) for i, j in p.offsets]
            if any(c not in free for c in foot):
                continue
            if y > 0 and not any(c in studs_below for c in foot):
                continue
            chosen = (p, px, pz, foot)
            break
        free.discard((ax, az))
        if chosen is None:
            uncovered += 1
            continue
        p, px, pz, foot = chosen
        local[p.key] -= p.stack
        for c in foot:
            free.discard(c)
        tiles.append((p, px, pz))
    return tiles, uncovered


def _score(tiles, uncovered, colour, below_seams, anchor=None):
    fallback = sum(1 for p, _, _ in tiles if colour is not None and p.item.colour.lower() != colour.lower())
    aligned = len(_seams(tiles) & below_seams) if below_seams else 0
    loose = 0
    if anchor:
        loose = sum(1 for p, px, pz in tiles if not any((px + i, pz + j) in anchor for i, j in p.offsets))
    return uncovered * SCORE_UNCOVERED + fallback * SCORE_FALLBACK + aligned * SCORE_ALIGNED + len(tiles) * SCORE_PART + loose * SCORE_LOOSE


def tile_layer(cells, y, unit, colour, items, state, anchor=None, prefer=None, descent=None):
    """The best tiling of one layer. Returns (tiles, uncovered_count)."""
    base = _pieces(items, unit, prefer, descent)
    if not base or not cells:
        return [], len(cells)
    below_seams = state.seams_at.get(y, set())
    studs_below = state.studs_at.get(y, set())
    best = None
    for orient in ("x", "z"):
        pieces = _sorted(base, colour, orient, prefer if prefer in ("round", "cone") else None)
        for reverse in (False, True):
            for cap in LEAD_CAPS:
                tiles, unc = _tile_once(cells, y, pieces, state.stock, studs_below, reverse, cap)
                score = _score(tiles, unc, colour, below_seams, anchor)
                if best is None or score < best[0]:
                    best = (score, tiles, unc)
    return best[1], best[2]


def _commit(tiles, y, unit, role, colour, state, region_index=0):
    h = 3 if unit == "brick" else 1
    for p, px, pz in tiles:
        state.stock[p.key] -= p.stack
        for k in range(p.stack):
            state.placements.append(Placement(p.item.part_id, p.item.colour, px, pz, y + k, p.rot, 1))
            state.owner.append(region_index)
        rp = state.role_parts.setdefault(role, [0, 0])
        rp[0] += p.stack
        if colour is None or p.item.colour.lower() == colour.lower():
            rp[1] += p.stack
        else:
            state.fallback += p.stack
    top = y + h
    state.studs_at.setdefault(top, set()).update((px + i, pz + j) for p, px, pz in tiles if p.item.dims.studs_top
                                                  for i in range(p.fx) for j in range(p.fz))
    state.seams_at[top] = _seams(tiles)


def fill(regions, inv, colours):
    """Fill the regions. `colours` is {role: colour name}. Returns a FillResult (it reports coverage rather than raising when parts run out)."""
    items = list(inv.parts())
    stock = {(i.part_id.lower(), i.colour.lower()): i.qty for i in items}
    state = _State(stock)
    starts = {}                                                   # plate height -> cells of every layer that starts there
    for r in regions:
        for n, cells in enumerate(r.layers):
            starts.setdefault(r.y + n * r.layer_height, set()).update(cells)
    result = FillResult()
    results = [None] * len(regions)                    # in region order (the fill order differs)
    order = sorted(range(len(regions)), key=lambda k: (not regions[k].required, regions[k].y, k))
    for k in order:
        r = regions[k]
        colour = colours.get(r.role)
        snap = state.snapshot()
        covered = 0
        for n, cells in enumerate(r.layers):
            y = r.y + n * r.layer_height
            tiles, unc = tile_layer(cells, y, r.unit, colour, items, state, starts.get(y + r.layer_height), r.prefer, r.descent)
            _commit(tiles, y, r.unit, r.role, colour, state, k)
            covered += len(cells) - unc
        res = RegionResult(r.name, r.role, r.required, covered, r.cell_count, r.min_coverage)
        if not r.required and covered < r.cell_count:
            state.restore(snap)                                   # an optional detail is all or nothing
            res.kept, res.covered = False, 0
        results[k] = res
    result.regions = results
    result.placements, owner = _prune_loose(state.placements, state.owner, regions, results, inv)
    result.owner = owner
    result.fallback_parts = state.fallback
    result.role_parts = state.role_parts
    return result


def _recount(regions, results, placements, owner, inv):
    """Coverage per region from the parts that are really there (after pruning)."""
    by_region = {}
    for p, k in zip(placements, owner):
        by_region.setdefault(k, []).append(p)
    for k, r in enumerate(regions):
        d = r.layer_height
        got = 0
        for n, cells in enumerate(r.layers):
            y0 = r.y + n * d
            covered = set()
            for p in by_region.get(k, ()):
                if y0 <= p.y < y0 + d:
                    fx, fz = inv.get(p.part_id, p.colour).dims.footprint(p.rotation)
                    covered.update((p.x + i, p.z + j) for i in range(fx) for j in range(fz))
            got += len(covered & cells)
        results[k].covered = got


def _prune_loose(placements, owner, regions, results, inv):
    """Drop parts that are not part of the main connected piece (a plate next to the tower that nothing sits on would just fall off),
    then drop optional regions that are no longer complete; the coverage of every region is counted again from what remains."""
    from backend.core import buildmodel as bm
    placements, owner = list(placements), list(owner)
    for _ in range(3):
        build_parts = placements
        occ, _c = bm.occupancy(build_parts, inv)
        comps = bm.components(build_parts, inv, occ)
        if len(comps) > 1:
            keep = set(max(comps, key=lambda c: (len(c), -min(c))))
            placements = [p for i, p in enumerate(placements) if i in keep]
            owner = [o for i, o in enumerate(owner) if i in keep]
        _recount(regions, results, placements, owner, inv)
        changed = False
        for k, (r, res) in enumerate(zip(regions, results)):
            if res.kept and not r.required and res.covered < r.cell_count:
                res.kept, res.covered = False, 0
                placements = [p for p, o in zip(placements, owner) if o != k]
                owner = [o for o in owner if o != k]
                changed = True
        if not changed and len(comps) <= 1:
            break
    _recount(regions, results, placements, owner, inv)
    for k, res in enumerate(results):
        if not res.kept:
            res.covered = 0
    return placements, owner
