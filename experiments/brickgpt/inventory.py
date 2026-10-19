"""Inventory, colour and step logic for the BrickGPT experiment. Pure Python (no torch), so it can be tested on its own.

An inventory is {(part, colour_name): qty} with part one of the 8 BrickGPT types (LDraw part numbers, no ".DAT").
Bricks are brickgpt.data.Brick objects (h, w, x, y, z); z is the layer, 0 = ground.
"""
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# BrickGPT's 8 types: LDraw part -> (short side, long side) in studs. Checked against brick_library.json in the repo.
PART_DIMS = {"3005": (1, 1), "3004": (1, 2), "3010": (1, 4), "3009": (1, 6), "3008": (1, 8),
             "3003": (2, 2), "3001": (2, 4), "2456": (2, 6)}
DIMS_PART = {v: k for k, v in PART_DIMS.items()}
DEFAULT_COLOUR = (115, "Medium Lime")   # what BrickGPT's own to_ldr() uses; for runs without an inventory


def part_of(brick) -> str:
    return brick.part_id.split(".")[0]


def dims_part(h: int, w: int):
    """LDraw part number for a h x w brick (either orientation), or None if BrickGPT has no such type."""
    return DIMS_PART.get((min(h, w), max(h, w)))


# ------------------------------------------------------------------ inventories
def inv_total(inv: dict) -> int:
    return sum(inv.values())


def type_totals(inv: dict) -> Counter:
    t = Counter()
    for (part, _), n in inv.items():
        t[part] += n
    return t


def make_fixtures(builds_glob=str(ROOT / "build_cache" / "builds" / "*.json"), seed=42, sizes=(12, 40, 120), n_colours=5):
    """small/medium/large inventories (nested: small is inside medium is inside large), sampled with a fixed seed from the
    (type, colour) mix of the 8 BrickGPT types seen in our cached builds, restricted to the n most common colours so it looks
    like a pile. The saved scan (parts.json) has no brick of these 8 types, so this is the closest real distribution we have."""
    mix = Counter()
    for f in sorted(glob.glob(builds_glob)):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in d.get("steps", []):
            for p in s["parts"]:
                if p["ld_part"] in PART_DIMS:
                    mix[(p["ld_part"], p["colour_name"])] += p["qty"]
    by_colour = Counter()
    for (_, c), n in mix.items():
        by_colour[c] += n
    keep = {c for c, _ in by_colour.most_common(n_colours)}
    pairs = sorted((k, n) for k, n in mix.items() if k[1] in keep)
    rng = random.Random(seed)
    draws = rng.choices([k for k, _ in pairs], weights=[n for _, n in pairs], k=max(sizes))
    out = {}
    for name, size in zip(("small", "medium", "large"), sizes):
        c = Counter(draws[:size])
        out[name] = [{"part": p, "colour": col, "qty": n} for (p, col), n in sorted(c.items())]
    return out


def load_fixtures(path=HERE / "inventories.json", real_pile=HERE / "real_pile.json") -> dict:
    """The sampled small/medium/large inventories, plus "real_pile" (a real photo as the scanner read it) when present."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(real_pile).exists():
        raw["real_pile"] = json.loads(Path(real_pile).read_text(encoding="utf-8"))["rows"]
    return {name: {(r["part"], r["colour"]): r["qty"] for r in rows} for name, rows in raw.items()}


def missing_bricks(bricks, inv: dict) -> dict:
    """Shopping mode: {part: how many more you need} for a (free) design against an inventory. By type; colours are not
    considered because shopping mode generates without any colour."""
    need = Counter(part_of(b) for b in bricks)
    have = type_totals(inv)
    return {p: n - have[p] for p, n in sorted(need.items()) if n > have[p]}


# ------------------------------------------------------------------ colours
def read_ldconfig(path=ROOT / "frontend" / "public" / "demo" / "LDConfig.ldr") -> dict:
    """{colour name with spaces: (ldraw code, '#RRGGBB')}"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.search(r"!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+(#[0-9A-Fa-f]{6})", line)
        if m:
            out[m.group(1).replace("_", " ")] = (int(m.group(2)), m.group(3).upper())
    out.update({n.replace("Grey", "Gray"): v for n, v in list(out.items()) if "Grey" in n})   # the scanner says "Gray", LDraw "Grey"
    return out


def _neighbours(bricks):
    """Index adjacency: same layer with edges touching, or neighbouring layers with overlapping footprints."""
    adj = defaultdict(set)
    for i, a in enumerate(bricks):
        for j in range(i + 1, len(bricks)):
            b = bricks[j]
            ox = min(a.x + a.h, b.x + b.h) - max(a.x, b.x)   # overlap along x (>0 overlapping, 0 touching)
            oy = min(a.y + a.w, b.y + b.w) - max(a.y, b.y)
            if ox < 0 or oy < 0:
                continue
            if (abs(a.z - b.z) == 1 and ox > 0 and oy > 0) or (a.z == b.z and (ox > 0) != (oy > 0)):
                adj[i].add(j)
                adj[j].add(i)
    return adj


def assign_colours(bricks, inv: dict):
    """One colour name per brick, never using more of a (type, colour) than the inventory holds.

    Colours are handed out by region: the most plentiful colour first; it floods outwards (through touching bricks) from the
    lowest brick that can take it until none of it is left, so it gets the largest region; then the next colour, and so on.
    Bricks left over (their type's stock of that colour ran out) take the colour of their neighbours, else any colour of their
    type that is left. Needs inv to hold at least as many bricks of each type as the structure uses."""
    avail = Counter(inv)
    colours = [c for c, _ in sorted(Counter({c: sum(n for (_, cc), n in inv.items() if cc == c) for (_, c) in inv}).items(),
                                    key=lambda kv: (-kv[1], kv[0]))]
    adj = _neighbours(bricks)
    out = [None] * len(bricks)
    for c in colours:
        while True:
            seeds = [i for i, b in enumerate(bricks) if out[i] is None and avail[(part_of(b), c)] > 0]
            if not seeds:
                break
            queue, seen = [min(seeds, key=lambda i: (bricks[i].z, i))], set()
            while queue:
                i = queue.pop(0)
                if i in seen:
                    continue
                seen.add(i)
                p = part_of(bricks[i])
                if out[i] is None and avail[(p, c)] > 0:
                    out[i] = c
                    avail[(p, c)] -= 1
                    queue.extend(sorted(adj[i] - seen))
    for i, b in enumerate(bricks):
        if out[i] is not None:
            continue
        p = part_of(b)
        options = [c for c in colours if avail[(p, c)] > 0]
        if not options:
            raise ValueError(f"inventory has no {p} left for brick {i}")
        near = Counter(out[j] for j in adj[i] if out[j] in options)
        c = near.most_common(1)[0][0] if near else max(options, key=lambda c: avail[(p, c)])
        out[i] = c
        avail[(p, c)] -= 1
    return out


# ------------------------------------------------------------------ steps and LDraw
def make_steps(bricks, per_step=3):
    """Lists of brick indices: bottom-up by layer, then placement order, 1..per_step bricks per step, never across layers."""
    order = sorted(range(len(bricks)), key=lambda i: (bricks[i].z, i))
    steps, cur = [], []
    for i in order:
        if cur and (bricks[cur[0]].z != bricks[i].z or len(cur) == per_step):
            steps.append(cur)
            cur = []
        cur.append(i)
    if cur:
        steps.append(cur)
    return steps


def brick_line(b, code: int) -> str:
    """Same geometry as BrickGPT's Brick.to_ldr(), with our colour code and no STEP line."""
    x = (b.x + b.h * 0.5) * 20
    z = (b.y + b.w * 0.5) * 20
    y = b.z * -24
    matrix = "0 0 1 0 1 0 -1 0 0" if b.h <= b.w else "-1 0 0 0 1 0 0 0 -1"
    return f"1 {code} {x:g} {y:g} {z:g} {matrix} {part_of(b)}.dat\n"


def to_ldr(bricks, colour_names, steps, ldconfig: dict) -> str:
    out = []
    for step in steps:
        out.extend(brick_line(bricks[i], ldconfig.get(colour_names[i], (DEFAULT_COLOUR[0], ""))[0]) for i in step)
        out.append("0 STEP\n")
    return "".join(out)


def step_meta(bricks, colour_names, steps, ldconfig: dict) -> list:
    """The viewer's `steps` list: per step the parts added, aggregated by (part, colour)."""
    names = {"3005": "Brick 1 x 1", "3004": "Brick 1 x 2", "3010": "Brick 1 x 4", "3009": "Brick 1 x 6",
             "3008": "Brick 1 x 8", "3003": "Brick 2 x 2", "3001": "Brick 2 x 4", "2456": "Brick 2 x 6"}
    out = []
    for n, step in enumerate(steps):
        agg = Counter((part_of(bricks[i]), colour_names[i]) for i in step)
        out.append({"index": n, "parts": [
            {"ld_part": p, "name": names[p], "colour_code": str(ldconfig.get(c, (DEFAULT_COLOUR[0], "#C0F500"))[0]),
             "colour_name": c, "hex": ldconfig.get(c, (0, "#C0F500"))[1], "qty": q} for (p, c), q in sorted(agg.items())]})
    return out


# ------------------------------------------------------------------ one-piece check
def single_piece(bricks) -> dict:
    """Is the design ONE piece you could pick up? Bricks are linked through stud overlaps: two bricks in neighbouring layers whose
    footprints overlap by at least one stud. (BrickGPT's own is_connected() only asks that every brick reaches the ground, so two
    separate towers pass it.) -> {"single_piece": bool, "pieces": number of linked groups, "sizes": brick count of each, biggest first}"""
    n = len(bricks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(bricks):
        for j in range(i + 1, n):
            b = bricks[j]
            if abs(a.z - b.z) == 1 and min(a.x + a.h, b.x + b.h) > max(a.x, b.x) and min(a.y + a.w, b.y + b.w) > max(a.y, b.y):
                parent[find(i)] = find(j)
    sizes = sorted(Counter(find(i) for i in range(n)).values(), reverse=True)
    return {"single_piece": len(sizes) == 1, "pieces": len(sizes), "sizes": sizes}
