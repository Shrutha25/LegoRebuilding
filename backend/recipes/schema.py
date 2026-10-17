"""Declarative recipes: a JSON description of a model that compiles into a `Recipe` (and so into regions the filler tiles).

    {
      "id": "cottage", "title": "Cottage", "category": "building", "tags": ["building"], "nouns": ["cottage"], "names": ["Mossy Cottage"],
      "roles": [{"name": "wall", "want": "major"}, {"name": "roof", "want": "second"}],
      "params": {"width": {"range": [4, 8], "bind": "w"}, "depth": {"range": [4, 6], "bind": "d"}, "floors": {"range": [2, 4], "bind": "h"}},
      "silhouettes": {"wide-squat": {"width": "hi", "floors": "lo"}, "tall-narrow": {"width": "lo", "floors": "hi"}},
      "requires": {"min_parts": 14, "min_long": 3, "min_area": 50},
      "variants": {"grid": {"width": [4, 6, 8], "depth": [4, 6], "floors": [2, 3]}, "where": "width >= depth"},
      "derive": {"top": "1 + 3 * floors"},
      "parts": [
        {"name": "base", "primitive": "box", "role": "wall", "unit": "plate", "args": {"w": "width", "d": "depth"}, "at": {"x": 1, "z": 1, "y": 0}},
        {"name": "walls", "primitive": "wall_with_window", "role": "wall", "args": {"w": "width", "d": "depth", "layers": "floors"},
         "attach": {"to": "base", "rel": "on_top_of", "anchor": "center"}},
        {"name": "roof", "primitive": "slope_roof", "role": "roof", "unit": "plate", "args": {"w": "width + 2", "d": "depth + 2", "ridge": "x"},
         "attach": {"to": "walls", "rel": "on_top_of", "anchor": "center"}}
      ],
      "features": {"roof": ["roof"], "door": ["walls"]},
      "symmetry": ""
    }

Everything in "args", "at", "attach" offsets, "when", "derive", "repeat" and "variants" is an expression over the recipe's params, the derived
names and a few facts about the inventory (see FACT_NAMES). Numbers are whole numbers (studs; heights in layers). `validate_spec` returns a
list of plain-language problems (used to send errors back to the model that writes recipes); `compile_spec` builds the Recipe.
"""

import itertools
import json
import logging
import re
from pathlib import Path

from backend.recipes import primitives as prim
from backend.recipes.base import Recipe, Region, Role
from backend.recipes.expr import ExprError, evaluate

log = logging.getLogger("lego.recipes.schema")

SPECS_DIR = Path(__file__).resolve().parent / "specs"
FACT_NAMES = ("total_area", "total_parts", "longest", "thin", "has_slopes", "has_tiles", "has_rounds", "has_cones", "colours")
SILHOUETTES = ("long-low", "tall-narrow", "wide-squat", "arch", "compact", "spread", "tapering")
WANTS = ("major", "second", "dark", "neutral")
UNITS = ("brick", "plate")
RELS = ("on_top_of", "in_front_of", "behind", "beside", "at")
ANCHORS = ("center", "start", "end")
ATTACH_KEYS = {"to", "rel", "anchor", "anchor_x", "anchor_z", "dx", "dz", "dy", "side"}
MAX_PARTS, MAX_STUDS, MAX_LAYERS, MAX_PARAMS = 30, 24, 22, 6
TOP_KEYS = {"id", "title", "category", "tags", "nouns", "names", "roles", "params", "silhouettes", "requires", "variants", "derive", "parts",
            "features", "symmetry", "needs", "notes", "constraint"}
PART_KEYS = {"name", "primitive", "role", "unit", "args", "at", "attach", "required", "min_coverage", "when", "repeat", "mirror_x", "kind"}


class SpecError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


# ---------------------------------------------------------------- context
def fact_values(facts):
    if facts is None:
        return {"total_area": 999, "total_parts": 99, "longest": 8, "thin": True, "has_slopes": False, "has_tiles": False,
                "has_rounds": False, "has_cones": False, "colours": 3}
    return {"total_area": facts.total_area, "total_parts": facts.total_parts, "longest": facts.longest, "thin": facts.has_thin(),
            "has_slopes": facts.slope_area >= 4 and bool(facts.slope_depths), "has_tiles": facts.tile_area >= 4,
            "has_rounds": facts.round_parts > 0, "has_cones": facts.cones > 0, "colours": len(facts.colours)}


def _ev(x, names):
    """An argument: whole number, expression string, or a list of those."""
    if isinstance(x, list):
        return [_ev(v, names) for v in x]
    return evaluate(x, names)


# ---------------------------------------------------------------- generation
class _Bounds:
    def __init__(self):
        self.parts = {}                                     # part name -> [x0, x1, z0, z1, y0, y1] over its regions

    def add(self, name, regions):
        for r in regions:
            cells = set().union(*r.layers) if r.layers else set()
            if not cells:
                continue
            xs, zs = [c[0] for c in cells], [c[1] for c in cells]
            box = [min(xs), max(xs) + 1, min(zs), max(zs) + 1, r.y, r.top]
            old = self.parts.get(name)
            self.parts[name] = box if old is None else [min(old[0], box[0]), max(old[1], box[1]), min(old[2], box[2]),
                                                       max(old[3], box[3]), min(old[4], box[4]), max(old[5], box[5])]


def _place(attach, size, bounds, names):
    """(x, z, y) of a part of footprint `size` attached to an earlier part."""
    t = bounds.parts.get(attach["to"])
    if t is None:
        raise ExprError(f"attach target {attach['to']!r} has no cells (empty or not built yet)")
    w, d = size
    dx, dz, dy = (evaluate(attach.get(k, 0), names) for k in ("dx", "dz", "dy"))
    anchor, rel = attach.get("anchor", "center"), attach.get("rel", "on_top_of")
    ax, az = attach.get("anchor_x", anchor), attach.get("anchor_z", anchor)

    def along(lo, hi, extent, a):
        return {"center": lo + ((hi - lo) - extent) // 2, "start": lo, "end": hi - extent}[a]

    if rel == "on_top_of":
        return along(t[0], t[1], w, ax) + dx, along(t[2], t[3], d, az) + dz, t[5] + dy
    if rel == "in_front_of":
        return along(t[0], t[1], w, ax) + dx, t[3] + dz, t[4] + dy
    if rel == "behind":
        return along(t[0], t[1], w, ax) + dx, t[2] - d - dz, t[4] + dy
    side = attach.get("side", "right")
    x = t[1] + dx if side == "right" else t[0] - w - dx
    return x, along(t[2], t[3], d, az) + dz, t[4] + dy


def _mirror(regions, width):
    out = []
    for r in regions:
        layers = [frozenset((width - 1 - x, z) for x, z in c) for c in r.layers]
        out.append(Region(r.name + "-mirror", r.role, r.y, layers, r.unit, r.required, r.min_coverage, r.kind, r.prefer, r.descent))
    return out


def _translate(regions):
    cells = [c for r in regions for layer in r.layers for c in layer]
    if not cells:
        return regions
    sx, sz = max(0, -min(c[0] for c in cells)), max(0, -min(c[1] for c in cells))
    sy = max(0, -min(r.y for r in regions))
    if not (sx or sz or sy):
        return regions
    return [Region(r.name, r.role, r.y + sy, [frozenset((x + sx, z + sz) for x, z in c) for c in r.layers], r.unit, r.required,
                   r.min_coverage, r.kind, r.prefer, r.descent) for r in regions]


def generate_regions(spec, params, facts):
    names = {**params, **fact_values(facts)}
    for k, v in spec.get("derive", {}).items():
        names[k] = evaluate(v, names)
    bounds, out = _Bounds(), []
    for part in spec["parts"]:
        repeat = part.get("repeat")
        count = evaluate(repeat["count"], names) if repeat else 1
        if count > 12:
            raise ExprError(f"part {part['name']!r} repeats {count} times (max 12)")
        for k in range(count):
            local = dict(names, **({repeat.get("var", "k"): k} if repeat else {}))
            if "when" in part and not evaluate(part["when"], local):
                continue
            p = prim.PRIMITIVES[part["primitive"]]
            given = part.get("args", {})
            args = {**p.defaults, **{k2: _ev(v, local) for k2, v in given.items() if k2 not in p.enums}}
            args.update({k2: v for k2, v in given.items() if k2 in p.enums})      # enum arguments are plain strings, not expressions
            at = part.get("at", {})
            x, z, y = (evaluate(at.get(c, 0), local) for c in ("x", "z", "y"))
            if "attach" in part:
                x, z, y = _place(part["attach"], p.size(args) if p.size else (1, 1), bounds, local)
            role = part["role"]
            if isinstance(role, list):
                role = role[k % len(role)]
            name = part["name"].replace("{k}", str(k))
            ctx = prim.Ctx(name, role, part.get("unit", "brick"), x, z, y, part.get("required", True), part.get("min_coverage", 0.9), facts)
            regions = p.build(ctx, args)
            if "kind" in part:
                for r in regions:
                    r.kind = part["kind"]
            if "mirror_x" in part:
                regions = regions + _mirror(regions, evaluate(part["mirror_x"], local))
            bounds.add(part["name"], regions)
            out.extend(regions)
    return _translate(out)


# ---------------------------------------------------------------- validation
def _combos(spec):
    """Parameter settings to dry-run: low, middle and high of every parameter, and every silhouette preset."""
    params = spec.get("params", {})
    lo = {k: v["range"][0] for k, v in params.items()}
    hi = {k: v["range"][1] for k, v in params.items()}
    mid = {k: (v["range"][0] + v["range"][1]) // 2 for k, v in params.items()}
    out = [lo, mid, hi]
    for preset in spec.get("silhouettes", {}).values():
        out.append(resolve_preset(spec, preset))
    return out


def resolve_preset(spec, preset):
    out = {}
    for k, v in spec.get("params", {}).items():
        lo, hi = v["range"]
        want = preset.get(k, "mid")
        out[k] = {"lo": lo, "mid": (lo + hi) // 2, "hi": hi}.get(want, want) if isinstance(want, str) else want
    return out


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_spec(spec, dry_run=True):
    """Problems with a recipe spec, in plain language; an empty list means it is valid (and generates regions for sample parameters).
    Never raises: JSON of the wrong shape is reported."""
    try:
        return _validate_spec(spec, dry_run)
    except (AttributeError, TypeError, KeyError, IndexError, ValueError, RecursionError) as e:
        return [f"the recipe is malformed ({type(e).__name__}: {e}); check that every value has the type the format shows"]


def _validate_spec(spec, dry_run=True):
    errs = []
    if not isinstance(spec, dict):
        return ["the recipe must be a JSON object"]
    for k in spec:
        if k not in TOP_KEYS:
            errs.append(f"unknown top-level key {k!r}")
    for k in ("id", "category", "roles", "params", "parts"):
        if k not in spec:
            errs.append(f"missing {k!r} (the recipe has the keys {sorted(spec)})")
    if errs:
        return errs
    if not isinstance(spec["id"], str) or not spec["id"].replace("_", "").replace("-", "").isalnum():
        errs.append("id must be a short name made of letters, digits, - and _")
    roles = spec["roles"]
    if not isinstance(roles, list) or not roles or not all(isinstance(r, dict) and isinstance(r.get("name"), str) for r in roles):
        return errs + ["roles must be a list of {name, want}"]
    role_names = [r["name"] for r in roles]
    for r in roles:
        if r.get("want", "major") not in WANTS:
            errs.append(f"role {r['name']!r}: want must be one of {WANTS}")
    for key, kind in (("silhouettes", dict), ("requires", dict), ("derive", dict), ("features", dict), ("variants", dict), ("tags", list), ("nouns", list),
                      ("names", list), ("needs", list)):
        if key in spec and not isinstance(spec[key], kind):
            errs.append(f"{key!r} must be a {'JSON object' if kind is dict else 'list'}, got {type(spec[key]).__name__}")
    if errs:
        return errs
    params = spec["params"]
    if not isinstance(params, dict) or not params or len(params) > MAX_PARAMS:
        return errs + [f"params must have 1 to {MAX_PARAMS} entries"]
    for k, v in params.items():
        rng = v.get("range") if isinstance(v, dict) else None
        if not (isinstance(rng, list) and len(rng) == 2 and all(_is_int(n) for n in rng) and 0 <= rng[0] <= rng[1] <= MAX_STUDS):
            errs.append(f"param {k!r}: range must be [low, high] whole numbers between 0 and {MAX_STUDS}")
        elif v.get("bind") not in (None, "w", "d", "h"):
            errs.append(f"param {k!r}: bind must be w, d or h")
    for sil, preset in spec.get("silhouettes", {}).items():
        if sil not in SILHOUETTES:
            errs.append(f"silhouette {sil!r} is not one of {SILHOUETTES}")
        for k, v in (preset or {}).items():
            if k not in params:
                errs.append(f"silhouette {sil!r}: unknown param {k!r}")
            elif not (v in ("lo", "mid", "hi") or _is_int(v)):
                errs.append(f"silhouette {sil!r}: {k} must be lo, mid, hi or a whole number")
    parts = spec["parts"]
    if not isinstance(parts, list) or not parts or len(parts) > MAX_PARTS:
        return errs + [f"parts must be a list of 1 to {MAX_PARTS} parts"]
    seen, conditional = set(), set()
    known = set(params) | set(FACT_NAMES) | set(spec.get("derive", {}))
    for i, part in enumerate(parts):
        if not isinstance(part, dict):
            errs.append(f"part {i}: must be an object")
            continue
        label = f"part {part.get('name', i)!r}"
        for k in part:
            if k not in PART_KEYS:
                errs.append(f"{label}: unknown key {k!r}")
        if not isinstance(part.get("name"), str) or not part["name"]:
            errs.append(f"part {i}: missing name")
            continue
        base = part["name"].split("{")[0]
        if part["name"] in seen and not (part.get("when") and part["name"] in conditional):      # alternatives share a name when each has a "when"
            errs.append(f"{label}: duplicate name (give alternatives a \"when\" each)")
        if part.get("when"):
            conditional.add(part["name"])
        seen.add(part["name"])
        seen.add(base)
        p = prim.PRIMITIVES.get(part.get("primitive"))
        if p is None:
            errs.append(f"{label}: unknown primitive {part.get('primitive')!r}; use one of {sorted(prim.PRIMITIVES)}")
            continue
        roles_ok = [part.get("role")] if isinstance(part.get("role"), str) else part.get("role")
        if not roles_ok or not all(r in role_names for r in roles_ok):
            errs.append(f"{label}: role must be one of {role_names}")
        if part.get("unit", "brick") not in UNITS:
            errs.append(f"{label}: unit must be brick or plate")
        args = part.get("args", {})
        for k in args:
            if k not in p.args and k not in p.enums:
                errs.append(f"{label}: {p.name} has no argument {k!r}; it takes {sorted(p.args)}")
        for k in p.args:
            if k not in args and k not in p.defaults:
                errs.append(f"{label}: {p.name} needs argument {k!r}")
        for k, allowed in p.enums.items():
            if k in args and args[k] not in allowed:
                errs.append(f"{label}: {k} must be one of {allowed}")
        if "attach" in part:
            at = part["attach"]
            if at.get("to") not in seen - {part["name"], base}:
                errs.append(f"{label}: attach.to must name an EARLIER part, got {at.get('to')!r}")
            if at.get("rel", "on_top_of") not in RELS:
                errs.append(f"{label}: attach.rel must be one of {RELS}, got {at.get('rel')!r}")
            for ak in ("anchor", "anchor_x", "anchor_z"):
                if at.get(ak, "center") not in ANCHORS:
                    errs.append(f"{label}: attach.{ak} must be one of {ANCHORS}, got {at.get(ak)!r}")
            for ak in at:
                if ak not in ATTACH_KEYS:
                    errs.append(f"{label}: unknown attach key {ak!r}; attach takes only {sorted(ATTACH_KEYS)} (offsets are dx, dz, dy)")
            if p.size is None:
                errs.append(f"{label}: {p.name} cannot be attached")
        if part.get("repeat"):
            known = known | {part["repeat"].get("var", "k")}
        if not (0 <= float(part.get("min_coverage", 0.9)) <= 1):
            errs.append(f"{label}: min_coverage must be between 0 and 1")
    for k, f in spec.get("features", {}).items():
        if not isinstance(f, list) or any(x not in seen for x in f):
            bad = [x for x in f if x not in seen] if isinstance(f, list) else f
            errs.append(f"feature {k!r} lists parts that do not exist: {bad}; the parts are {sorted(n for n in seen if '{' not in n)}")
    wheelish = [n for n in role_names + [str(pt.get("name", "")) for pt in parts if isinstance(pt, dict)] if re.search(r"wheel|tyre|tire", n, re.I)]
    if wheelish and "wheels" not in spec.get("needs", ()):
        errs.append(f"{wheelish[0]!r}: wheels cannot be made from plain bricks, and real wheel sets cannot be placed yet; leave wheels out")
    if spec.get("symmetry", "") not in ("", "x", "z", "xz"):
        errs.append("symmetry must be '', 'x', 'z' or 'xz'")
    if errs or not dry_run:
        return errs
    # dry run: every expression must evaluate, and the shapes must be sane, for low / middle / high parameters and each silhouette
    try:
        recipe = compile_spec(spec, check=False)
    except (ExprError, KeyError, TypeError, ValueError) as e:
        return [f"cannot compile: {e}"]
    for params_try in _combos(spec):
        try:
            regions = recipe.generate(params_try, None)
        except ExprError as e:
            errs.append(f"with {params_try}: {e}")
            continue
        except (KeyError, TypeError, ValueError, IndexError) as e:
            errs.append(f"with {params_try}: {type(e).__name__}: {e}")
            continue
        cells = {c for r in regions for layer in r.layers for c in layer}
        if not cells:
            errs.append(f"with {params_try}: the recipe has no cells at all")
            continue
        w = max(c[0] for c in cells) + 1
        d = max(c[1] for c in cells) + 1
        h = max(r.top for r in regions)
        if w > MAX_STUDS or d > MAX_STUDS or h > MAX_LAYERS * 3 + 6:
            errs.append(f"with {params_try}: the model is {w} x {d} studs and {h} plates tall; keep it within {MAX_STUDS} x {MAX_STUDS} x {MAX_LAYERS * 3 + 6}")
        if not any(r.required for r in regions):
            errs.append(f"with {params_try}: no part is required")
    return list(dict.fromkeys(errs))[:12]


# ---------------------------------------------------------------- compile
def _auto_variants(spec):
    """No explicit variants: the silhouette presets, the middle, and every low / middle / high combination, filtered by "constraint"."""
    params = spec["params"]
    keys = list(params)
    lohi = {k: (params[k]["range"][0], (params[k]["range"][0] + params[k]["range"][1]) // 2, params[k]["range"][1]) for k in keys}
    cons = spec.get("constraint")

    def variants(facts):
        base = fact_values(facts)
        out, seen = [], set()
        cands = [resolve_preset(spec, p) for p in spec.get("silhouettes", {}).values()]
        cands += [dict(zip(keys, combo)) for combo in itertools.product(*[dict.fromkeys(lohi[k]) for k in keys])]
        for p in cands:
            sig = tuple(sorted(p.items()))
            if sig in seen or (cons and not evaluate(cons, {**p, **base})):
                continue
            seen.add(sig)
            out.append(p)
        return out
    return variants


def _variants_fn(spec):
    v = spec.get("variants")
    if not v:
        return _auto_variants(spec)
    grid, where = v["grid"], v.get("where")

    def variants(facts):
        base = fact_values(facts)
        keys = list(grid)
        vals = []
        for k in keys:
            g = grid[k]
            if isinstance(g, dict):                                  # {"if": expr, "then": [...], "else": [...]}
                g = g["then"] if evaluate(g["if"], base) else g["else"]
            vals.append(g)
        out = []
        for combo in itertools.product(*vals):
            p = dict(zip(keys, combo))
            if where is None or evaluate(where, {**p, **base}):
                out.append(p)
        return out
    return variants


def compile_spec(spec, check=True):
    """The Recipe a valid spec describes. With check=True the spec is validated first (SpecError lists the problems)."""
    if check:
        errs = validate_spec(spec)
        if errs:
            raise SpecError(errs)
    req = spec.get("requires", {})
    needs = tuple(spec.get("needs", ())) + tuple(sorted({n for part in spec["parts"] for n in prim.PRIMITIVES[part["primitive"]].needs}))
    recipe = Recipe(
        id=spec["id"], category=spec["category"], nouns=tuple(spec.get("nouns", ())),
        roles=tuple(Role(r["name"], r.get("want", "major")) for r in spec["roles"]),
        params={k: tuple(v["range"]) for k, v in spec["params"].items()},
        generate=lambda params, facts, _s=spec: generate_regions(_s, params, facts),
        tags=tuple(spec.get("tags", ())), min_parts=req.get("min_parts", 8), min_long=req.get("min_long", 2), min_area=req.get("min_area", 0),
        min_build_parts=req.get("min_build_parts"), variants=_variants_fn(spec), notes=spec.get("notes", ""),
        title=spec.get("title", spec["id"].title()), names=tuple(spec.get("names", ())), silhouettes=spec.get("silhouettes", {}),
        binds={k: v["bind"] for k, v in spec["params"].items() if v.get("bind")}, needs=needs,
        features=tuple(spec.get("features", {})), symmetry=spec.get("symmetry", ""), spec=spec,
        exact_variants=bool(spec.get("variants")))
    cons = spec.get("constraint") or (spec.get("variants") or {}).get("where")
    recipe.allowed = (lambda params, facts, _c=cons: bool(evaluate(_c, {**params, **fact_values(facts)}))) if cons else (lambda params, facts: True)
    return recipe


def load_specs(directory=SPECS_DIR):
    """Every *.json in the specs folder as a Recipe, in file-name order. A broken file is logged and skipped."""
    out = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            out.append(compile_spec(json.loads(path.read_text(encoding="utf-8"))))
        except (SpecError, ValueError, KeyError) as e:
            log.error("recipe spec %s is invalid: %s", path.name, e)
    return out
