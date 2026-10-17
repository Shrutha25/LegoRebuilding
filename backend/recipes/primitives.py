"""Primitives: parameterised shape builders that turn a few numbers into Regions (layers of stud cells) for the filler to tile.

A declarative recipe (schema.py) names a primitive per part and gives its arguments as expressions; the primitive knows how to make
cells. Adding a shape is one decorated function:

    @primitive("dome", "A half sphere of bricks.", args={"diameter": "studs across", "layers": "bricks tall"}, size=lambda a: (a["diameter"],) * 2)
    def dome(ctx, a):
        return [cells_region(ctx.name, ctx.role, ctx.y, [...], ctx.unit, ctx.required, ctx.min_coverage)]

`ctx` carries the part's name, role, unit, position (x, z, y in studs / plates), required, min_coverage and the inventory facts;
`a` the evaluated arguments (ints, bools, or the enum strings the primitive declares). `size(a)` returns the (width, depth) in studs
used by attachments ("on_top_of", "beside" ...).

Primitives: box, wall_with_window, tapered_box, slope_roof, arch, pillar, leg, cylinder, cone, wing, stairs, head, posts, checker, cells, rails, wheel_axle.
"""

from dataclasses import dataclass, field

from backend.recipes.base import Opening, Region, cells_region, hollow, rect, solid, stepped

PRIMITIVES = {}


@dataclass
class Ctx:
    name: str
    role: str
    unit: str
    x: int
    z: int
    y: int
    required: bool
    min_coverage: float
    facts: object = None


@dataclass
class Primitive:
    name: str
    summary: str
    build: object
    args: dict = field(default_factory=dict)         # argument -> description (shown to the model that writes recipes)
    defaults: dict = field(default_factory=dict)
    enums: dict = field(default_factory=dict)        # argument -> allowed strings (not expressions)
    size: object = None                              # (args) -> (width, depth) in studs
    needs: tuple = ()

    def arg_names(self):
        return set(self.args)


def primitive(name, summary, args, defaults=None, enums=None, size=None, needs=()):
    def deco(fn):
        PRIMITIVES[name] = Primitive(name, summary, fn, args, defaults or {}, enums or {}, size, needs)
        return fn
    return deco


def describe_all():
    """One line per primitive for prompts: name(args) - summary."""
    out = []
    for p in PRIMITIVES.values():
        a = ", ".join(f"{k}: {v}" for k, v in p.args.items())
        out.append(f"- {p.name}({a}) - {p.summary}")
    return "\n".join(out)


def _wd(a, w="w", d="d"):
    return (max(1, a[w]), max(1, a[d]))


def _circle(diameter):
    """Cells of a blocky disc `diameter` studs across (the corners of the square are cut)."""
    r = diameter / 2
    return frozenset((x, z) for x in range(diameter) for z in range(diameter)
                     if (x + 0.5 - r) ** 2 + (z + 0.5 - r) ** 2 <= (r + 0.45) ** 2 - (0.9 if diameter > 2 else 0))


# ---------------------------------------------------------------- walls and blocks
@primitive("box", "A solid block, or hollow walls when thickness > 0.",
           {"w": "studs along x", "d": "studs along z", "layers": "number of layers (bricks, or plates when unit is plate)",
            "thickness": "0 = solid, else wall thickness in studs"},
           {"layers": 1, "thickness": 0}, size=lambda a: _wd(a))
def box(ctx, a):
    if a["thickness"] > 0:
        return [hollow(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], a["thickness"], (), ctx.unit, ctx.required, ctx.min_coverage)]
    return [solid(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("wall_with_window", "Hollow walls with a door gap on one side and window gaps on the sides.",
           {"w": "studs along x", "d": "studs along z", "layers": "wall height in bricks", "thickness": "1 or 2",
            "door_w": "door width (0 = no door)", "door_layers": "door height in bricks", "door_off": "studs from the wall start to the door",
            "windows": "1 = one window in each side wall, 0 = none", "window_layer": "which brick layer the windows are in (0 = lowest)"},
           {"thickness": 1, "door_w": 1, "door_layers": 2, "door_off": 0, "windows": 0, "window_layer": 1},
           enums={"door_side": ["front", "back", "left", "right"]}, size=lambda a: _wd(a))
def wall_with_window(ctx, a):
    side = a.get("door_side", "front")
    ops = []
    if a["door_w"] > 0:
        ops.append(Opening(side, a["door_off"], a["door_w"], 0, max(0, min(a["door_layers"], a["layers"]) - 1)))
    if a["windows"] and a["layers"] > a["window_layer"]:
        for s in ("left", "right"):
            ops.append(Opening(s, max(0, (a["d"] - 1) // 2), 1, a["window_layer"], a["window_layer"]))
    return [hollow(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], a["thickness"], ops, ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("tapered_box", "A block whose layers shrink step by step (a stepped roof, a nose, a pyramid).",
           {"w": "studs along x at the bottom", "d": "studs along z at the bottom", "layers": "max layers (0 = until nothing is left)",
            "shrink": "studs lost per side per layer"},
           {"layers": 0, "shrink": 1}, enums={"axes": ["xz", "x", "z"]}, size=lambda a: _wd(a))
def tapered_box(ctx, a):
    return [stepped(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"] or None, a["shrink"], a.get("axes", "xz"),
                    ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("slope_roof", "A gable roof. Uses real slope bricks when the inventory has enough of them, else stepped plates.",
           {"w": "studs along x", "d": "studs along z"}, {}, enums={"ridge": ["x", "z"]}, size=lambda a: _wd(a))
def slope_roof(ctx, a):
    w, d, ridge = a["w"], a["d"], a.get("ridge", "x")
    run = d if ridge == "x" else w                          # the side the roof falls along
    facts = ctx.facts
    cells = w * d
    if facts is not None and 2 in facts.slope_depths and facts.slope_area >= 0.5 * cells and run >= 4:
        out, k = [], 0
        while run - 4 * k >= 4:
            lo, hi = 2 * k, run - 2 * k

            def band(a0, a1):
                return rect(ctx.x, ctx.z + a0, w, a1 - a0) if ridge == "x" else rect(ctx.x + a0, ctx.z, a1 - a0, d)

            y = ctx.y + 3 * k
            low, high = ("-z", "+z") if ridge == "x" else ("-x", "+x")
            out.append(Region(f"{ctx.name}-{k}a", ctx.role, y, [band(lo, lo + 2)], "brick", ctx.required and k == 0, ctx.min_coverage, "slope", "slope", low))
            out.append(Region(f"{ctx.name}-{k}b", ctx.role, y, [band(hi - 2, hi)], "brick", ctx.required and k == 0, ctx.min_coverage, "slope", "slope", high))
            k += 1
        rest = run - 4 * k
        if rest > 0:
            c = rect(ctx.x, ctx.z + 2 * k, w, rest) if ridge == "x" else rect(ctx.x + 2 * k, ctx.z, rest, d)
            out.append(Region(f"{ctx.name}-ridge", ctx.role, ctx.y + 3 * k, [c], "plate", False, ctx.min_coverage, "solid"))
        return out
    return [stepped(ctx.name, ctx.role, ctx.x, ctx.z, w, d, ctx.y, None, 1, "z" if ridge == "x" else "x", "plate", ctx.required, ctx.min_coverage)]


@primitive("arch", "An opening with a leg on each side and a stepped (corbelled) top. Gives bridges, gates and windows.",
           {"span": "studs between the legs", "layers": "total height in bricks", "leg_w": "width of each leg", "depth": "studs along z"},
           {"leg_w": 1, "depth": 2}, size=lambda a: (a["span"] + 2 * a["leg_w"], a["depth"]))
def arch(ctx, a):
    span, n, lw, dep = a["span"], a["layers"], a["leg_w"], a["depth"]
    total = span + 2 * lw
    straight = n - max(1, span // 2)                       # layers with a straight-sided opening
    layers = []
    for k in range(n):
        narrow = max(0, k - straight + 1)
        opening = span - 2 * narrow
        cells = set(rect(ctx.x, ctx.z, total, dep))
        if opening > 0 and k < n - 1:
            cells -= set(rect(ctx.x + lw + narrow, ctx.z, opening, dep))
        layers.append(frozenset(cells))
    return [cells_region(ctx.name, ctx.role, ctx.y, layers, ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("pillar", "A solid column.", {"w": "studs along x", "d": "studs along z", "layers": "height in layers"},
           {"w": 2, "d": 2}, size=lambda a: _wd(a))
def pillar(ctx, a):
    return [solid(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("leg", "A leg or support: a column, optionally with a wider foot plate.",
           {"w": "studs along x", "d": "studs along z", "layers": "height in bricks", "foot": "1 = add a one-stud-wider foot plate"},
           {"w": 1, "d": 1, "foot": 0}, size=lambda a: _wd(a))
def leg(ctx, a):
    out = [solid(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]
    if a["foot"]:
        out.append(solid(ctx.name + "-foot", ctx.role, ctx.x, ctx.z, a["w"], a["d"] + 1, ctx.y, 1, "plate", False, ctx.min_coverage))
    return out


@primitive("cylinder", "A round column (a blocky disc, round bricks where the inventory has them).",
           {"diameter": "studs across", "layers": "height in layers"}, size=lambda a: (a["diameter"], a["diameter"]))
def cylinder(ctx, a):
    c = frozenset((ctx.x + x, ctx.z + z) for x, z in _circle(a["diameter"]))
    r = cells_region(ctx.name, ctx.role, ctx.y, [c] * a["layers"], ctx.unit, ctx.required, ctx.min_coverage)
    r.prefer = "round"
    return [r]


@primitive("cone", "A round tapering shape: discs that get smaller, ending in one stud (a cone brick where available).",
           {"base": "studs across at the bottom", "layers": "max layers (0 = until one stud)"}, {"layers": 0},
           size=lambda a: (a["base"], a["base"]))
def cone(ctx, a):
    layers, d = [], a["base"]
    while d >= 1 and (not a["layers"] or len(layers) < a["layers"]):
        off = (a["base"] - d) // 2
        layers.append(frozenset((ctx.x + off + x, ctx.z + off + z) for x, z in _circle(d)))
        if d <= 2:
            break
        d -= 2
    r = cells_region(ctx.name, ctx.role, ctx.y, layers, ctx.unit, ctx.required, ctx.min_coverage)
    r.prefer = "cone"
    return [r]


@primitive("wing", "A flat wing, fin or bow of plates that narrows towards its tip(s). Symmetric about its middle when sym is on.",
           {"span": "studs along the long direction (z, or x when axis is x)", "chord": "studs across at the root",
            "taper": "studs lost across by the tip", "sym": "1 = symmetric about the middle (wings), 0 = narrows to one end"}, {"taper": 0, "sym": 0},
           enums={"axis": ["z", "x"]}, size=lambda a: (a["chord"], a["span"]) if a.get("axis", "z") == "z" else (a["span"], a["chord"]))
def wing(ctx, a):
    span, chord, taper, sym = a["span"], a["chord"], a["taper"], a["sym"]
    cells = set()
    for j in range(span):
        if sym:
            frac = abs(j - (span - 1) / 2) / max(0.5, (span - 1) / 2)
            c = max(1, round(chord - taper * frac))
        else:
            c = max(1, chord - (taper * j) // max(1, span - 1))
        for i in range(c):
            cells.add((ctx.x + i, ctx.z + j) if a.get("axis", "z") == "z" else (ctx.x + j, ctx.z + i + (chord - c) // 2))
    return [cells_region(ctx.name, ctx.role, ctx.y, [cells], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("stairs", "Steps rising along x: each layer is one step shorter than the one below.",
           {"width": "studs along z", "steps": "number of steps", "run": "studs per step"}, {"run": 1},
           size=lambda a: (a["steps"] * a["run"], a["width"]))
def stairs(ctx, a):
    n, run, width = a["steps"], a["run"], a["width"]
    layers = [frozenset((ctx.x + x, ctx.z + z) for x in range(k * run, n * run) for z in range(width)) for k in range(n)]
    return [cells_region(ctx.name, ctx.role, ctx.y, layers, ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("head", "A head: a block with two ears on top and an optional snout in front (+z).",
           {"w": "studs along x", "d": "studs along z", "layers": "height in layers", "ears": "1 = two ears (needs w >= 3)",
            "snout": "studs the snout sticks out in front (0 = none)"},
           {"ears": 0, "snout": 0}, size=lambda a: (a["w"], a["d"] + a["snout"]))
def head(ctx, a):
    out = [solid(ctx.name, ctx.role, ctx.x, ctx.z, a["w"], a["d"], ctx.y, a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]
    top = ctx.y + a["layers"] * (3 if ctx.unit == "brick" else 1)
    if a["ears"] and a["w"] >= 3:
        out.append(cells_region(ctx.name + "-ears", ctx.role, top, [{(ctx.x, ctx.z), (ctx.x + a["w"] - 1, ctx.z)}], ctx.unit, False, ctx.min_coverage))
    if a["snout"]:
        sw = max(1, a["w"] - 2)
        out.append(solid(ctx.name + "-snout", ctx.role, ctx.x + (a["w"] - sw) // 2, ctx.z + a["d"], sw, a["snout"], ctx.y, max(1, a["layers"] - 1), ctx.unit, False, ctx.min_coverage))
    return out


@primitive("posts", "Single-stud columns at the crossings of some x and z positions (corner posts, lamp posts, feet).",
           {"x0": "first x", "x1": "second x (same as x0 for one column)", "x2": "third x", "x3": "fourth x", "z0": "first z", "z1": "second z",
            "layers": "height in layers"}, {"x1": None, "x2": None, "x3": None, "z1": None},
           size=lambda a: (max(v for k, v in a.items() if k.startswith("x") and v is not None) + 1, max(a["z0"], a["z1"] if a["z1"] is not None else 0) + 1))
def posts(ctx, a):
    xs = {a[k] for k in ("x0", "x1", "x2", "x3") if a.get(k) is not None}
    zs = {a[k] for k in ("z0", "z1") if a.get(k) is not None}
    cells = {(ctx.x + x, ctx.z + z) for x in xs for z in zs}
    return [cells_region(ctx.name, ctx.role, ctx.y, [cells] * a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("checker", "Every second stud of a rectangle (crenellations, a grille).",
           {"w": "studs along x", "d": "studs along z", "layers": "layers"}, {"layers": 1}, enums={},
           size=lambda a: _wd(a))
def checker(ctx, a):
    cells = {(x, z) for (x, z) in rect(ctx.x, ctx.z, a["w"], a["d"]) if (x + z) % 2 == 0}
    return [cells_region(ctx.name, ctx.role, ctx.y, [cells] * a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("cells", "Explicit stud cells, for details no other primitive makes (fins, teeth, a nose).",
           {"cells": "a list of [x, z] pairs, relative to the part's position", "layers": "layers"}, {"layers": 1},
           size=lambda a: (max(c[0] for c in a["cells"]) + 1, max(c[1] for c in a["cells"]) + 1))
def cells(ctx, a):
    c = {(ctx.x + x, ctx.z + z) for x, z in a["cells"]}
    return [cells_region(ctx.name, ctx.role, ctx.y, [c] * a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("rails", "Two lines of studs along x, one on each long side (railings, kerbs, wheel arches).",
           {"length": "studs along x", "width": "studs between the two lines along z", "layers": "layers"}, {"layers": 1},
           size=lambda a: (a["length"], a["width"]))
def rails(ctx, a):
    c = {(ctx.x + x, ctx.z + z) for x in range(a["length"]) for z in (0, a["width"] - 1)}
    return [cells_region(ctx.name, ctx.role, ctx.y, [c] * a["layers"], ctx.unit, ctx.required, ctx.min_coverage)]


@primitive("wheel_axle", "Wheels on an axle at the two sides of a body. NOT SUPPORTED YET: the grid cannot place real wheel sets, so recipes using it are never offered.",
           {"length": "studs along x", "width": "studs along z"}, size=lambda a: (a["length"], a["width"]), needs=("wheels",))
def wheel_axle(ctx, a):
    cells = {(ctx.x, ctx.z), (ctx.x, ctx.z + a["width"] - 1), (ctx.x + a["length"] - 1, ctx.z), (ctx.x + a["length"] - 1, ctx.z + a["width"] - 1)}
    return [cells_region(ctx.name, ctx.role, ctx.y, [cells], "brick", ctx.required, ctx.min_coverage)]
