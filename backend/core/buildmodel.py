"""AI builds on a stud grid: the model, a local validator, local repair, and step generation. No Gemini calls in here.

Coordinates (the contract the Gemini prompt states):

    x, z   integer stud coordinates of the part's minimum-x / minimum-z corner stud, AFTER rotation (x to the right, z towards
           the viewer; one cell = one stud = 20 LDU)
    y      integer height of the part's BOTTOM in plate units (ground = 0; a brick is 3 plates tall, a plate or tile 1)
    rotation  0 | 90 | 180 | 270 degrees about the vertical axis. At 0 the long side runs along x; at 90 along z.

A part occupies the cells x .. x+fx-1, z .. z+fz-1 for layers y .. y+height-1. Two parts must not share a cell in a layer.
A part above the ground must rest on studs: some part below it whose top (y_below + height_below == y) has studs (not a tile)
under at least one of its cells. Parts on the ground (y == 0) stand on an imaginary baseplate.
"""

import math
from dataclasses import dataclass, field, replace

from backend.core.partdims import Dims

ROTATIONS = (0, 90, 180, 270)


@dataclass
class Placement:
    part_id: str                  # as in the user's inventory
    colour: str                   # inventory colour name
    x: int
    z: int
    y: int
    rotation: int
    step: int


@dataclass
class Build:
    title: str
    description: str
    parts: list


@dataclass(frozen=True)
class InvItem:
    """One kind of piece the user owns that an AI build may use (known footprint and height, LDraw file and colour)."""
    part_id: str
    name: str
    colour: str
    qty: int
    dims: Dims
    ld_part: str                  # LDraw part number
    ld_colour: int                # LDraw colour code


class Inventory:
    def __init__(self, items):
        self.items = {(i.part_id.lower(), i.colour.lower()): i for i in items}

    def get(self, part_id, colour):
        return self.items.get((str(part_id).lower(), str(colour).lower()))

    def parts(self):
        return list(self.items.values())

    def equivalents(self, item):
        """Other inventory items the same size and kind (same footprint, height, studs): swap candidates, same colour first."""
        d = item.dims
        out = [i for i in self.items.values()
               if i is not item and (i.dims.long, i.dims.short, i.dims.height, i.dims.studs_top, i.dims.shape) == (d.long, d.short, d.height, d.studs_top, d.shape)]
        return sorted(out, key=lambda i: (i.colour.lower() != item.colour.lower(), i.part_id != item.part_id))


@dataclass
class Issue:
    kind: str                     # unknown_part | colour | overuse | overlap | floating | unsupported | step_order | bad_value
    message: str
    index: int | None = None      # index into build.parts


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    components: int = 0

    @property
    def ok(self):
        return not self.errors


# ---------------------------------------------------------------- geometry
def _dims(inv, p):
    item = inv.get(p.part_id, p.colour)
    return item.dims if item else None


def cells(p, dims):
    fx, fz = dims.footprint(p.rotation)
    return [(p.x + i, p.z + j) for i in range(fx) for j in range(fz)]


def layers(p, dims):
    return range(p.y, p.y + dims.height)


def occupancy(parts, inv):
    """{(x, z, layer): index} for the parts that have dimensions; also the list of colliding (index, index) pairs."""
    occ, clashes = {}, []
    for i, p in enumerate(parts):
        d = _dims(inv, p)
        if d is None:
            continue
        for (cx, cz) in cells(p, d):
            for ly in layers(p, d):
                other = occ.get((cx, cz, ly))
                if other is not None and other != i:
                    clashes.append((other, i))
                else:
                    occ[(cx, cz, ly)] = i
    return occ, sorted(set(clashes))


def supporters(i, parts, inv, occ):
    """Indices of parts directly below part i whose studs it sits on."""
    p = parts[i]
    d = _dims(inv, p)
    if d is None or p.y == 0:
        return []
    found = set()
    for (cx, cz) in cells(p, d):
        j = occ.get((cx, cz, p.y - 1))
        if j is not None and j != i:
            dj = _dims(inv, parts[j])
            if dj is not None and dj.studs_top and parts[j].y + dj.height == p.y:
                found.add(j)
    return sorted(found)


def components(parts, inv, occ):
    """Connected groups of parts joined through studs (a part resting on another). Returns a list of index lists."""
    n = len(parts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i in range(n):
        for j in supporters(i, parts, inv, occ):
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        if _dims(inv, parts[i]) is not None:
            groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------- validation
def validate(build, inv, max_parts=None):
    """Check a build against the user's inventory and the physical rules. No repair, no Gemini."""
    rep = Report()
    parts = build.parts
    used = {}
    for i, p in enumerate(parts):
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (p.x, p.z, p.y, p.rotation, p.step)):
            rep.errors.append(Issue("bad_value", f"part {i}: x, z, y, rotation and step must be integers", i))
            continue
        if p.rotation not in ROTATIONS:
            rep.errors.append(Issue("bad_value", f"part {i}: rotation {p.rotation} is not 0, 90, 180 or 270", i))
        if p.y < 0 or p.step < 1:
            rep.errors.append(Issue("bad_value", f"part {i}: y must be >= 0 and step >= 1", i))
        item = inv.get(p.part_id, p.colour)
        if item is None:
            known = any(k[0] == str(p.part_id).lower() for k in inv.items)
            kind = "colour" if known else "unknown_part"
            what = "has no such colour in your inventory" if known else "is not in your inventory (or has unknown size)"
            rep.errors.append(Issue(kind, f"part {i}: {p.part_id} / {p.colour} {what}", i))
            continue
        used[item] = used.get(item, 0) + 1
    for item, count in used.items():
        if count > item.qty:
            rep.errors.append(Issue("overuse", f"{item.part_id} ({item.colour}): uses {count}, you own {item.qty}"))
    if max_parts is not None and len(parts) > max_parts:
        rep.warnings.append(Issue("bad_value", f"{len(parts)} parts, more than the limit of {max_parts}"))

    occ, clashes = occupancy(parts, inv)
    for a, b in clashes:
        rep.errors.append(Issue("overlap", f"parts {a} and {b} occupy the same space", b))

    for i, p in enumerate(parts):
        d = _dims(inv, p)
        if d is None or p.y == 0:
            continue
        if not supporters(i, parts, inv, occ):
            rep.errors.append(Issue("unsupported", f"part {i} ({p.part_id} at y={p.y}) rests on nothing", i))

    comps = components(parts, inv, occ)
    rep.components = len(comps)
    for comp in comps:
        if all(parts[i].y > 0 for i in comp):
            rep.errors.append(Issue("floating", f"parts {comp} are not connected to the ground", comp[0]))
    if len(comps) > 1:
        rep.warnings.append(Issue("floating", f"the build is {len(comps)} separate sub-assemblies"))

    step_errors = check_steps(parts, inv, occ)
    rep.errors.extend(step_errors)
    return rep


def check_steps(parts, inv, occ=None):
    """Each part must rest on parts of an earlier step or of its own step."""
    if occ is None:
        occ, _ = occupancy(parts, inv)
    errors = []
    for i, p in enumerate(parts):
        if _dims(inv, p) is None or p.y == 0:
            continue
        sup = supporters(i, parts, inv, occ)
        if sup and all(parts[j].step > p.step for j in sup):
            errors.append(Issue("step_order", f"part {i} (step {p.step}) rests only on parts of later steps", i))
    return errors


# ---------------------------------------------------------------- repair
@dataclass
class RepairLog:
    actions: list = field(default_factory=list)       # human-readable repairs
    dropped: int = 0
    swapped: int = 0
    nudged: int = 0

    def add(self, text):
        self.actions.append(text)


def _normalise(build, rlog):
    """Snap to the grid, normalise rotations and steps."""
    fixed = []
    for i, p in enumerate(build.parts):
        try:
            x, z, y = round(float(p.x)), round(float(p.z)), round(float(p.y))
            rot = round(float(p.rotation) / 90) * 90 % 360
            step = max(1, round(float(p.step))) if p.step is not None else 1
        except (TypeError, ValueError):
            rlog.add(f"part {i}: unreadable numbers, dropped")
            rlog.dropped += 1
            continue
        y = max(0, y)
        if (x, z, y, rot, step) != (p.x, p.z, p.y, p.rotation, p.step):
            rlog.add(f"part {i}: snapped to the grid ({p.x},{p.z},{p.y}, rot {p.rotation}) -> ({x},{z},{y}, rot {rot})")
        fixed.append(replace(p, x=x, z=z, y=y, rotation=rot, step=step))
    return fixed


def _fix_inventory(parts, inv, rlog):
    """Swap parts the user does not have to an equivalent inventory part/colour (same size and kind); drop what cannot be
    swapped; trim overuse. A swap only ever uses SPARE stock, never a piece another part asks for exactly."""
    demand = {}
    for p in parts:
        k = (str(p.part_id).lower(), str(p.colour).lower())
        if k in inv.items:
            demand[k] = demand.get(k, 0) + 1
    spare = {k: max(0, it.qty - demand.get(k, 0)) for k, it in inv.items.items()}
    used = {}
    out = []
    for i, p in enumerate(parts):
        k = (str(p.part_id).lower(), str(p.colour).lower())
        item = inv.items.get(k)
        if item is not None and used.get(k, 0) < item.qty:
            used[k] = used.get(k, 0) + 1
            out.append(p)
            continue
        base = item
        if base is None:                                    # unknown part or colour: same part number in another colour?
            base = next((it for kk, it in inv.items.items() if kk[0] == str(p.part_id).lower()), None)
        swap = None
        if base is not None:
            same_part = [it for it in inv.items.values() if it is not base and it.part_id == base.part_id]
            for cand in inv.equivalents(base) + same_part:
                ck = (cand.part_id.lower(), cand.colour.lower())
                if spare.get(ck, 0) > 0 and cand.dims.footprint(p.rotation) == base.dims.footprint(p.rotation) \
                        and cand.dims.height == base.dims.height and cand.dims.shape == base.dims.shape:
                    swap = cand
                    break
        if swap is not None:
            ck = (swap.part_id.lower(), swap.colour.lower())
            spare[ck] -= 1
            used[ck] = used.get(ck, 0) + 1
            rlog.swapped += 1
            rlog.add(f"part {i}: {p.part_id}/{p.colour} not available, used {swap.part_id}/{swap.colour}")
            out.append(replace(p, part_id=swap.part_id, colour=swap.colour))
            continue
        rlog.add(f"part {i}: {p.part_id}/{p.colour} is not in your inventory (or used up), dropped")
    return out


def _nudge_overlaps(parts, inv, rlog, radius=3):
    """Parts that collide with an earlier/lower part move to the nearest free position at the same height; else they are dropped."""
    order = sorted(range(len(parts)), key=lambda i: (parts[i].step, parts[i].y, i))
    placed, occ = [], {}
    result = [None] * len(parts)

    def free(p, d):
        return all((cx, cz, ly) not in occ for (cx, cz) in cells(p, d) for ly in layers(p, d))

    def claim(idx, p, d):
        for (cx, cz) in cells(p, d):
            for ly in layers(p, d):
                occ[(cx, cz, ly)] = idx

    for i in order:
        p = parts[i]
        d = _dims(inv, p)
        if d is None:
            continue
        if free(p, d):
            claim(i, p, d)
            result[i] = p
            continue
        best = None
        for r in range(1, radius + 1):
            ring = [(dx, dz) for dx in range(-r, r + 1) for dz in range(-r, r + 1) if max(abs(dx), abs(dz)) == r]
            ring.sort(key=lambda t: abs(t[0]) + abs(t[1]))
            for dx, dz in ring:
                q = replace(p, x=p.x + dx, z=p.z + dz)
                if free(q, d):
                    best = q
                    break
            if best:
                break
        if best is None:
            rlog.dropped += 1
            rlog.add(f"part {i}: overlapped another part and no free spot was nearby, dropped")
            continue
        rlog.nudged += 1
        rlog.add(f"part {i}: overlapped another part, moved by ({best.x - p.x},{best.z - p.z}) studs")
        claim(i, best, d)
        result[i] = best
    return [p for p in result if p is not None]


def _settle(parts, inv, rlog):
    """Parts hanging in the air with a gap below them are lowered onto what is under them (an off-by-a-few-plates mistake);
    what still rests on nothing is dropped (and a part above a dropped part may lose its support, so this repeats)."""
    parts = list(parts)
    while True:
        occ, _ = occupancy(parts, inv)
        loose = [i for i in sorted(range(len(parts)), key=lambda k: (parts[k].y, k))
                 if parts[i].y > 0 and not supporters(i, parts, inv, occ)]
        if not loose:
            break
        i = loose[0]
        p, d = parts[i], _dims(inv, parts[i])
        landed = None
        for y2 in range(p.y - 1, -1, -1):
            q = replace(p, y=y2)
            if any(occ.get((cx, cz, ly), i) != i for (cx, cz) in cells(q, d) for ly in layers(q, d)):
                break                                                   # something is in the way: cannot fall further
            trial = parts[:i] + [q] + parts[i + 1:]
            if y2 == 0 or supporters(i, trial, inv, occupancy(trial, inv)[0]):
                landed = q
                break
        if landed is not None:
            rlog.add(f"part {i}: hung {p.y - landed.y} plates in the air, lowered onto its support")
            rlog.nudged += 1
            parts[i] = landed
        else:
            rlog.add(f"part {i}: rested on nothing, dropped")
            del parts[i]
    occ, _ = occupancy(parts, inv)                                      # sub-assemblies that never touch the ground
    keep = set()
    for comp in components(parts, inv, occ):
        if any(parts[i].y == 0 for i in comp):
            keep.update(comp)
        else:
            rlog.add(f"{len(comp)} parts were not connected to the ground, dropped")
    return [p for i, p in enumerate(parts) if i in keep]


def repair(build, inv):
    """Local repair, in order: snap/normalise, swap to inventory equivalents, nudge overlaps, settle/drop unsupported.
    Returns (repaired Build with steps regenerated if they were invalid, RepairLog)."""
    rlog = RepairLog()
    original = len(build.parts)
    parts = _normalise(build, rlog)
    parts = _fix_inventory(parts, inv, rlog)
    parts = _nudge_overlaps(parts, inv, rlog)
    parts = _settle(parts, inv, rlog)
    fixed = Build(build.title, build.description, parts)
    if check_steps(fixed.parts, inv) or any(p.step < 1 for p in fixed.parts):
        rlog.add("the step order was invalid: steps regenerated")
        fixed = Build(fixed.title, fixed.description, generate_steps(fixed.parts, inv))
    rlog.dropped = original - len(fixed.parts)
    return fixed, rlog


def drop_fraction(original_count, repaired_count):
    return 0.0 if original_count == 0 else 1.0 - repaired_count / original_count


# ---------------------------------------------------------------- steps
def generate_steps(parts, inv, size=3):
    """Steps built locally: bottom-up by y, 1-`size` parts per step, the parts of a step close together, and never a part before
    its support. Returns a new list of Placements with `step` set (in build order)."""
    size = max(1, min(3, size))
    remaining = sorted(range(len(parts)), key=lambda i: (parts[i].y, parts[i].z, parts[i].x))
    steps, current = [], 0
    occ, _ = occupancy(parts, inv)
    placed = set()
    out_order = []
    while remaining:
        ready = [i for i in remaining if parts[i].y == 0 or any(j in placed for j in supporters(i, parts, inv, occ))]
        if not ready:                                                # cannot happen for a settled build; avoid an endless loop
            ready = [remaining[0]]
        low = min(parts[i].y for i in ready)
        level = [i for i in ready if parts[i].y == low]
        first = level[0]
        centre = (parts[first].x, parts[first].z)
        group = [first]
        for i in level[1:]:
            if len(group) >= size:
                break
            if math.dist(centre, (parts[i].x, parts[i].z)) <= 6:       # physically close to the step's first part
                group.append(i)
        current += 1
        for i in group:
            placed.add(i)
            out_order.append((current, i))
            remaining.remove(i)
    result = []
    for step, i in out_order:
        result.append(replace(parts[i], step=step))
    return result


def steps_of(build):
    """{step: [placements]} in step order."""
    out = {}
    for p in build.parts:
        out.setdefault(p.step, []).append(p)
    return dict(sorted(out.items()))
