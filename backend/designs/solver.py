"""The local solver: fit the bricks of the pile into a concept's voxel sketch with OR-Tools CP-SAT.

    res = solve(sketch, parts, time_limit=2.0)          # parts: [(part_id, colour, Dims, qty)], one row per part and colour of the pile
    res.placements                                      # [(part_id, colour, x, z, y, rotation)] in studs / plates, y = bottom plate

One boolean per way of placing a part (kind, colour, rotation, x, z, y) wholly inside the sketch (at plate resolution: a layer is 3 plates). Constraints:

    no overlap      every plate cell is covered at most once
    exact counts    no part in a colour is used more often than the pile has it, so the build passes can_build in both colour modes (the colours are the
                    pile's own; with colour off any of them may be used where a colour was wished)
    held            a part above the ground needs a part below it with studs on top that overlaps its footprint (so nothing floats; slopes and tiles have
                    no studs, nothing sits on them)
    one piece       by lazy cuts: after a solve, every group of parts that is not tied to the rest gets a constraint "if all of this group is chosen, some
                    part touching it is chosen too" and the model is solved again from the previous answer (exact connectivity inside the model is too
                    slow for a 2 s limit)
    objective       cover as much of the sketch as the pile allows (a few holes are allowed: that is how it "adjusts voxels"), a little for fewer, bigger parts,
                    a bonus for a part in the colour a sketch cell wished and a small cost for another colour there, and (second phase, when time is left)
                    a penalty for a joint that lies directly over another joint, so the layers are staggered

A greedy answer (biggest ground part first, then the biggest part that touches what is there) is both the search's hint and the result if nothing better
is found in time. Parts still not connected to the biggest group at the end are dropped. Model building takes a few tens of milliseconds for piles of up
to about 20 pieces.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from backend.core.partdims import WHEEL_HOLDER_PLATES, WHEEL_KINDS, WHEEL_PLATES

W_VOLUME, W_PART, W_SEAM = 100, 2, 8
W_WISH, W_UNWISHED = 40, 25                            # per plate cell: a part in the wished colour, a part in another colour than wished
PHASE_ONE = 0.45                                       # share of the time for the first phase (volume, colour and connectivity)
MAX_CUT_ROUNDS = 8


@dataclass
class SolveResult:
    placements: list = field(default_factory=list)        # (part_id, colour, x, z, y, rotation)
    wish_cells: int = 0                                   # plate cells of the placed parts that sit where a colour was wished
    wish_met: int = 0                                     # ... and have that colour
    status: str = "NONE"
    covered: int = 0                                      # plate cells filled
    total: int = 0                                        # plate cells in the sketch
    options: int = 0
    seconds: float = 0.0
    build_seconds: float = 0.0
    dropped: int = 0                                      # parts removed because they were not connected to the main group
    cut_rounds: int = 0                                   # how many times connectivity cuts were added
    cells: frozenset = frozenset()                        # the (x, z, plate) cells the kept parts fill

    @property
    def coverage(self):
        return self.covered / self.total if self.total else 0.0


def _rotations(d):
    if d.shape == "slope":
        return (0, 90, 180, 270)
    return (0,) if d.long == d.short else (0, 90)


def _windows(occ, fx, fz, h):
    """Boolean array: True where an fx x fz x h block starting at (x, z, y) lies entirely in `occ` (a 3D boolean array)."""
    X, Z, Y = occ.shape
    if fx > X or fz > Z or h > Y:
        return np.zeros((0, 0, 0), bool)
    s = np.zeros((X + 1, Z + 1, Y + 1), dtype=np.int32)
    s[1:, 1:, 1:] = occ.astype(np.int32).cumsum(0).cumsum(1).cumsum(2)
    a, b, c = X - fx + 1, Z - fz + 1, Y - h + 1
    total = (s[fx:fx + a, fz:fz + b, h:h + c] - s[0:a, fz:fz + b, h:h + c] - s[fx:fx + a, 0:b, h:h + c] + s[0:a, 0:b, h:h + c]
             - s[fx:fx + a, fz:fz + b, 0:c] + s[0:a, fz:fz + b, 0:c] + s[fx:fx + a, 0:b, 0:c] - s[0:a, 0:b, 0:c])
    return total == fx * fz * h


def _edges(x, z, fx, fz):
    """The vertical joint segments around a footprint: ('v', x, z) lies between cells (x-1, z) and (x, z); ('h', x, z) between (x, z-1) and (x, z)."""
    out = [("v", x, z + j) for j in range(fz)] + [("v", x + fx, z + j) for j in range(fz)]
    out += [("h", x + i, z) for i in range(fx)] + [("h", x + i, z + fz) for i in range(fx)]
    return out


def _cells(o):
    """The (x, z, plate) cells an option occupies."""
    if "cells" in o:
        return o["cells"]
    return [(o["x"] + a, o["z"] + b, o["y"] + q) for a in range(o["fx"]) for b in range(o["fz"]) for q in range(o["h"])]


def _uses(o):
    """{row index: how many of that row the option uses}: one of its own part, or a holder, two rims and two tyres for a wheel set."""
    return o.get("uses") or {o["t"]: 1}


def _tops(o):
    """The (x, z, plate) positions where something can be set on this option's studs: the plate level of the top face."""
    if "tops" in o:
        return o["tops"]
    if not o["studs"]:
        return []
    return [(o["x"] + a, o["z"] + b, o["y"] + o["h"]) for a in range(o["fx"]) for b in range(o["fz"])]


def _bases(o):
    """The (x, z, plate) cells right under the option that must hold it (none on the ground, and none for a wheel set, which stands on its tyres)."""
    if o["y"] == 0 or "cells" in o:
        return []
    return [(o["x"] + a, o["z"] + b, o["y"]) for a in range(o["fx"]) for b in range(o["fz"])]


def _wheel_sets(parts):
    """[(holder row, rim row, tyre row)] that the pile can make at least one wheel set of: a holder, two rims and two tyres. For each holder the rim and
    tyre rows with the most pieces (a pile rarely has more than one colour of each)."""
    kinds = {k: [t for t, p in enumerate(parts) if p[2].kind == k and p[3] > 0] for k in ("wheelbase", "rim", "tyre")}
    rims = sorted((t for t in kinds["rim"] if parts[t][3] >= 2), key=lambda t: -parts[t][3])[:2]
    tyres = sorted((t for t in kinds["tyre"] if parts[t][3] >= 2), key=lambda t: -parts[t][3])[:2]
    return [(h, r, w) for h in kinds["wheelbase"] for r in rims for w in tyres]


def _wheel_options(occ, parts):
    """The ways to stand a wheel set on the ground (plate 0) inside the sketch. Axle along X (rotation 0, 4 x 2 studs): tyre columns at both ends, the
    holder between them. Axle along Z (rotation 90, 2 x 4). Each is one option that uses a holder, two rims and two tyres. Tyre cells go 5 plates up, the
    holder's 3 plates under its plate are reserved, and its studs face up at plate 3."""
    out = []
    X, Z, Y = occ.shape
    for h, r, w in _wheel_sets(parts):
        for rot in (0, 90):
            fx, fz = (4, 2) if rot == 0 else (2, 4)
            for x in range(X - fx + 1):
                for z in range(Z - fz + 1):
                    cells, tops, tyres, holder = [], [], [], []
                    for a in range(fx):
                        for b in range(fz):
                            at_end = a in (0, fx - 1) if rot == 0 else b in (0, fz - 1)
                            (tyres if at_end else holder).append((x + a, z + b))
                    cells = [(cx, cz, q) for cx, cz in tyres for q in range(WHEEL_PLATES)] + [(cx, cz, q) for cx, cz in holder for q in range(WHEEL_HOLDER_PLATES)]
                    if Y < WHEEL_PLATES or not all(occ[cx, cz, q] for cx, cz, q in cells):
                        continue
                    tops = [(cx, cz, WHEEL_HOLDER_PLATES) for cx, cz in holder]
                    if rot == 0:
                        place = [(parts[h][0], parts[h][1], x + 1, z, 0, 0), (parts[r][0], parts[r][1], x, z, 0, 270), (parts[r][0], parts[r][1], x + 3, z, 0, 90),
                                 (parts[w][0], parts[w][1], x, z, 0, 270), (parts[w][0], parts[w][1], x + 3, z, 0, 90)]
                    else:
                        place = [(parts[h][0], parts[h][1], x, z + 1, 0, 90), (parts[r][0], parts[r][1], x, z, 0, 0), (parts[r][0], parts[r][1], x, z + 3, 0, 180),
                                 (parts[w][0], parts[w][1], x, z, 0, 0), (parts[w][0], parts[w][1], x, z + 3, 0, 180)]
                    out.append({"t": h, "x": x, "z": z, "y": 0, "fx": fx, "fz": fz, "h": WHEEL_PLATES, "rot": rot, "studs": True, "vol": len(cells), "wished": 0, "match": 0,
                                "cells": cells, "tops": tops, "uses": {h: 1, r: 2, w: 2}, "wheel": True, "place": place})
    return out


def _options(sketch, parts):
    cells = sketch.plate_cells()
    X, Z, Y = sketch.width, sketch.depth, 3 * sketch.layers
    occ = np.zeros((X, Z, Y), bool)
    for x, z, p in cells:
        occ[x, z, p] = True
    wish = {(x, z, 3 * k + q): c.strip().lower() for (x, z, k), c in sketch.colours.items() for q in range(3)}
    opts = []
    for t, (pid, colour, d, qty) in enumerate(parts):
        if qty <= 0 or d.kind in WHEEL_KINDS:                                   # wheel parts only come in sets (below)
            continue
        for rot in _rotations(d):
            fx, fz = d.footprint(rot)
            ok = _windows(occ, fx, fz, d.height)
            for x, z, y in zip(*np.nonzero(ok)):
                o = {"t": t, "x": int(x), "z": int(z), "y": int(y), "fx": fx, "fz": fz, "h": d.height, "rot": rot, "studs": bool(d.studs_top),
                     "vol": fx * fz * d.height, "wished": 0, "match": 0}
                if wish:
                    want = [wish.get((o["x"] + i, o["z"] + j, o["y"] + q)) for i in range(fx) for j in range(fz) for q in range(d.height)]
                    o["wished"] = sum(1 for w in want if w)
                    o["match"] = sum(1 for w in want if w and w == colour.strip().lower())
                opts.append(o)
    opts += _wheel_options(occ, parts)
    return opts, len(cells)


def _neighbours(opts):
    """[set of option indexes each option touches through studs, above or below]."""
    below = {}
    for i, o in enumerate(opts):
        for c in _tops(o):
            below.setdefault(c, []).append(i)
    nbrs = [set() for _ in opts]
    for i, o in enumerate(opts):
        for c in _bases(o):
            for j in below.get(c, ()):                                              # j holds i
                nbrs[i].add(j)
                nbrs[j].add(i)
    return nbrs


def _greedy_one(opts, parts, nbrs, mode):
    """The option indexes of a valid first answer (nothing overlaps, the stock lasts, everything is held and one piece): start with the biggest ground part
    and keep adding the biggest part that touches what is there."""
    stock = {t: p[3] for t, p in enumerate(parts)}
    used, tops, picked = set(), set(), set()

    def fits(o):
        return all(stock[t] >= n for t, n in _uses(o).items())

    def take(i):
        o = opts[i]
        used.update(_cells(o))
        for t, n in _uses(o).items():
            stock[t] -= n
        picked.add(i)
        tops.update(_tops(o))

    ground = [i for i, o in enumerate(opts) if o["y"] == 0 and fits(o)]
    if not ground:
        return set()
    sign = -1 if mode == 2 else 1
    take(max(ground, key=lambda i: (sign * opts[i]["vol"], -opts[i]["z"], -opts[i]["x"])))
    frontier = set(nbrs[next(iter(picked))])
    while frontier:
        best = None
        for i in frontier:
            o = opts[i]
            if i in picked or not fits(o) or any(c in used for c in _cells(o)):
                continue
            if _bases(o) and not any(c in tops for c in _bases(o)):
                continue
            key = (sign * o["vol"], -o["y"], -o["z"], -o["x"]) if mode != 1 else (-o["y"], o["vol"], -o["z"], -o["x"])
            if best is None or key > best[0]:
                best = (key, i)
        if best is None:
            break
        take(best[1])
        frontier |= nbrs[best[1]]
        frontier.discard(best[1])
    return picked


def _greedy(opts, parts, nbrs):
    """The best of three greedy orders: biggest first, lowest layer first, smallest first."""
    tries = [_greedy_one(opts, parts, nbrs, m) for m in (0, 1, 2)]
    return max(tries, key=lambda g: _volume(opts, g))


def _groups(idx, nbrs):
    """The connected groups (lists of option indexes) among the chosen options `idx`, joined through studs."""
    chosen, seen, out = set(idx), set(), []
    for i in sorted(chosen):
        if i in seen:
            continue
        group, todo = [], [i]
        while todo:
            k = todo.pop()
            if k in seen:
                continue
            seen.add(k)
            group.append(k)
            todo += [j for j in nbrs[k] if j in chosen and j not in seen]
        out.append(group)
    return out


def _volume(opts, idx):
    return sum(opts[i]["vol"] for i in idx)


def _main_group(opts, idx, nbrs):
    """The connected group of chosen options with the most volume: that is the build, the rest is dropped."""
    groups = _groups(idx, nbrs)
    return max(groups, key=lambda g: _volume(opts, g)) if groups else []


def solve(sketch, parts, time_limit=2.0, workers=2, seed=0, seams=True):
    t0 = time.perf_counter()
    opts, total = _options(sketch, parts)
    res = SolveResult(total=total, options=len(opts))
    if not opts:
        res.status, res.seconds = ("NO_OPTIONS" if total else "NONE"), time.perf_counter() - t0
        return res
    model = cp_model.CpModel()
    var = [model.NewBoolVar(f"o{i}") for i in range(len(opts))]
    cover, by_type = {}, {}
    for i, o in enumerate(opts):
        for t, n in _uses(o).items():
            by_type.setdefault(t, []).append(n * var[i])
        for c in _cells(o):
            cover.setdefault(c, []).append(var[i])
    for vs in cover.values():
        if len(vs) > 1:
            model.AddAtMostOne(vs)
    for t, vs in by_type.items():
        model.Add(sum(vs) <= parts[t][3])
    top = {}                                               # (x, z, plate) -> options whose studs are on top there, the top face being at that plate
    for i, o in enumerate(opts):
        for c in _tops(o):
            top.setdefault(c, []).append(i)
    for i, o in enumerate(opts):
        if not _bases(o):
            continue
        below = {j for c in _bases(o) for j in top.get(c, [])}
        if below:
            model.Add(sum(var[j] for j in below) >= var[i])
        else:
            model.Add(var[i] == 0)
    nbrs = _neighbours(opts)
    start = _greedy(opts, parts, nbrs)
    base = [(W_VOLUME * o["vol"] - W_PART + W_WISH * o["match"] - W_UNWISHED * (o["wished"] - o["match"])) * var[i] for i, o in enumerate(opts)]
    model.Maximize(sum(base))
    res.build_seconds = time.perf_counter() - t0
    end = t0 + time_limit

    def run(limit, hint):
        model.ClearHints()
        for i in range(len(opts)):
            model.AddHint(var[i], 1 if i in hint else 0)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.05, limit)
        solver.parameters.num_workers = workers
        solver.parameters.random_seed = seed
        status = solver.Solve(model)
        res.status = solver.StatusName(status)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {i for i in range(len(opts)) if solver.Value(var[i])}
        return None

    def cut(group):
        """If all of this group is chosen, some part touching it must be chosen too, unless nothing outside it is chosen."""
        g = set(group)
        around = set().union(*(nbrs[i] for i in g)) - g
        outside = [var[k] for k in range(len(opts)) if k not in g]
        none_outside = model.NewBoolVar("")
        model.Add(none_outside + sum(outside) >= 1)
        model.Add(sum(var[i] for i in g) <= len(g) - 1 + sum(var[j] for j in around) + none_outside)

    # phase 1: volume and colour, with connectivity cuts
    chosen = run(PHASE_ONE * time_limit - res.build_seconds, start)
    for _ in range(MAX_CUT_ROUNDS):
        if chosen is None:
            break
        groups = _groups(chosen, nbrs)
        if len(groups) <= 1 or end - time.perf_counter() < 0.25:
            break
        res.cut_rounds += 1
        for g in groups:
            cut(g)
        nxt = run(min(0.5 * time_limit, end - time.perf_counter() - 0.1), set(max(groups, key=lambda g: _volume(opts, g))))
        if nxt is None:
            break
        chosen = nxt
    best = chosen
    # phase 2: staggered joints, starting from the phase-1 answer
    if seams and best is not None and end - time.perf_counter() > 0.3:
        tb, bb = {}, {}
        for i, o in enumerate(opts):
            if o.get("wheel"):
                continue
            for e in _edges(o["x"], o["z"], o["fx"], o["fz"]):
                tb.setdefault((e, o["y"] + o["h"]), []).append(var[i])
                bb.setdefault((e, o["y"]), []).append(var[i])
        pens = []
        for key in set(tb) & set(bb):
            a, b, pen = model.NewBoolVar(""), model.NewBoolVar(""), model.NewBoolVar("")
            for v in tb[key]:
                model.AddImplication(v, a)
            for v in bb[key]:
                model.AddImplication(v, b)
            model.Add(pen >= a + b - 1)
            pens.append(pen)
        model.Maximize(sum(base) - W_SEAM * sum(pens))
        second = run(end - time.perf_counter(), best)
        if second is not None and _volume(opts, _main_group(opts, second, nbrs)) >= _volume(opts, _main_group(opts, best, nbrs)):
            best = second
    if best is None or _volume(opts, _main_group(opts, best, nbrs)) < _volume(opts, _main_group(opts, start, nbrs)):
        if not start:
            res.seconds = time.perf_counter() - t0
            return res
        best, res.status = start, "GREEDY"
    kept = _main_group(opts, best, nbrs)
    res.dropped = len(best) - len(kept)
    res.placements = []
    for i in sorted(kept, key=lambda i: (opts[i]["y"], opts[i]["z"], opts[i]["x"])):
        o = opts[i]
        res.placements += o["place"] if o.get("wheel") else [(parts[o["t"]][0], parts[o["t"]][1], o["x"], o["z"], o["y"], o["rot"])]
    res.covered = _volume(opts, kept)
    res.cells = frozenset(c for i in kept for c in _cells(opts[i]))
    res.wish_cells = sum(opts[i]["wished"] for i in kept)
    res.wish_met = sum(opts[i]["match"] for i in kept)
    res.seconds = time.perf_counter() - t0
    return res
