"""From an inventory to scored recipe builds: parameters x colour assignments -> filler -> validator -> score.

    facts = facts_of(inv)
    cands = plan_candidates(inv, seed=1)          # best first, at most MAX_CANDIDATES, at most PER_RECIPE of one recipe
    unlock_hints(inv)                             # recipes whose minimum is NOT met and what would unlock them

A candidate is only returned when its REQUIRED regions are covered, the recipe's minimum parts / long part are met, the build is one
connected piece and the validator (buildmodel.validate) finds nothing wrong. Local repair is NOT expected to do anything for a recipe
build: when it does, that is logged (a recipe bug), and the build is dropped if repair had to remove parts.
"""

import logging
import random
import re
from dataclasses import dataclass, field

from backend.core import buildmodel as bm
from backend.core.inventory_check import can_build
from backend.core.buildmodel import Build
from backend.recipes import all_recipes, get
from backend.recipes.facts import assign_colours, facts_of
from backend.recipes.filler import fill

log = logging.getLogger("lego.recipes")

MAX_CANDIDATES = 10
PER_RECIPE = 3
PARAM_TRIES = 14                  # parameter settings tried per recipe
COLOUR_TRIES = 2                  # colour assignments tried per parameter setting
MAX_PARTS = 80


@dataclass
class Candidate:
    recipe_id: str
    params: dict
    colours: dict                 # role -> colour name
    build: Build
    score: float
    coverage: float               # share of the REQUIRED cells that are filled
    colour_consistency: float
    optional_kept: int
    optional_total: int
    part_count: int
    repaired: int = 0
    regions: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)   # (x, z, y) of each part -> the colour role of its region (for recolouring)

    @property
    def recipe(self):
        return get(self.recipe_id)

    @property
    def signature(self):
        return (self.recipe_id, tuple(sorted(self.params.items())), tuple(sorted(self.colours.items())))

    def default_name(self):
        return name_for(self.recipe, self.colours, self.params)


def name_for(recipe, colours=None, params=None):
    """A name for a build nobody named: one of the recipe's own evocative names, picked by the shape so different shapes get different names."""
    if not recipe.names:
        return recipe.title or recipe.id.title()
    key = sum((i + 1) * v for i, v in enumerate(sorted((params or {}).values())))
    return recipe.names[key % len(recipe.names)]


# ---------------------------------------------------------------- minimums
SUPPORTED_NEEDS = {"wheels": lambda f: f.wheel_sets > 0}          # what the grid can place; a recipe that needs more is never offered


def minimum_met(recipe, facts):
    if any(not SUPPORTED_NEEDS.get(n, lambda f: False)(facts) for n in recipe.needs):
        return False
    return facts.total_parts >= recipe.min_parts and facts.longest >= recipe.min_long and facts.total_area >= recipe.min_area


def _size_ok(recipe, params, facts):
    """Cheap pre-filter: a model needs about one stud of part per stud of region, so skip settings far bigger than the inventory."""
    try:
        regions = recipe.generate(params, facts)
    except (ValueError, KeyError):
        return None
    cells = sum(r.cell_count * (3 if r.unit == "brick" else 1) / 3 for r in regions if r.required)
    return regions if cells <= facts.total_area else None


# ---------------------------------------------------------------- one build
def make_candidate(recipe, params, inv, colours, facts=None, reasons=None):
    """Fill, validate, score. Returns a Candidate or None when the build is not acceptable. `reasons`, when given, collects why not, as
    (stage, message) pairs with stage "fill", "inventory" or "structure"."""

    def no(stage, msg):
        log.debug("%s %s: %s", recipe.id, params, msg)
        if reasons is not None:
            reasons.append((stage, msg))
        return None

    facts = facts or facts_of(inv)
    recipe.check_params(params)
    regions = recipe.generate(params, facts)
    res = fill(regions, inv, colours)
    if not res.placements:
        return no("fill", "no part could be placed at all with these pieces")
    if not res.required_ok:
        short = [f"{r.name} ({r.covered}/{r.total} studs filled, needs {round(r.min_cov * 100)}%)" for r in res.regions if r.required and r.coverage < r.min_cov]
        none = [r.name for r in res.regions if r.required and r.covered == 0]
        hint = (f"; nothing at all of {', '.join(none)} could be placed: a part above the ground must overlap the top of an earlier part, so give it an "
                f"'attach' on_top_of that part or an 'at' whose y is where that part ends") if none else ""
        return no("fill", "required parts could not be filled from the inventory: " + ", ".join(short) + hint)
    parts = bm.generate_steps(res.placements, inv)
    build = Build(name_for(recipe, colours, params), "", parts)
    if len(parts) < (recipe.min_build_parts or recipe.min_parts) or len(parts) > MAX_PARTS:
        return no("fill", f"the build has {len(parts)} parts, it needs {recipe.min_build_parts or recipe.min_parts} to {MAX_PARTS}")
    if max(inv.get(p.part_id, p.colour).dims.long for p in parts) < recipe.min_long:
        return no("fill", f"no part of length {recipe.min_long} or more was used")
    rep = bm.validate(build, inv, MAX_PARTS)
    repaired = 0
    if not rep.ok:
        fixed, rlog = bm.repair(build, inv)
        log.warning("recipe %s %s needed repair (%d actions): %s", recipe.id, params, len(rlog.actions), rlog.actions[:3])
        repaired = len(rlog.actions)
        if rlog.dropped:
            return no("structure", "parts had to be removed to make the build valid: " + "; ".join(rlog.actions[:3]))
        build = fixed
        rep = bm.validate(build, inv, MAX_PARTS)
        if not rep.ok:
            return no("structure", "; ".join(i.message for i in rep.errors[:3]))
    if rep.components != 1:
        return no("structure", f"the build falls into {rep.components} separate pieces; every brick must connect to the rest through studs")
    if not can_build(build.parts, inv, True):                       # the rule every build must pass; a failing build is dropped, not fixed
        log.warning("recipe %s %s uses pieces that are not in the inventory: dropped", recipe.id, params)
        return no("inventory", "the build uses pieces that are not in the inventory")
    req = [r for r in res.regions if r.required]
    cov = sum(r.covered for r in req) / max(1, sum(r.total for r in req))
    role_total = sum(v[0] for v in res.role_parts.values()) or 1
    consistency = sum(v[1] for v in res.role_parts.values()) / role_total
    opt = [r for r in res.regions if not r.required]
    kept = sum(1 for r in opt if r.kept)
    detail = kept / len(opt) if opt else 1.0
    size = min(1.0, len(parts) / 24)
    score = round(0.35 * cov + 0.25 * consistency + 0.2 * detail + 0.2 * size, 4)
    return Candidate(recipe.id, dict(params), dict(colours), build, score, cov, consistency, kept, len(opt), len(parts), repaired,
                     [(r.name, r.role, r.kept, round(r.coverage, 2)) for r in res.regions],
                     {(pl.x, pl.z, pl.y): regions[o].role for pl, o in zip(res.placements, res.owner)})


def _param_sets(recipe, facts, rng):
    base = recipe.variants(facts) if recipe.variants else [recipe.default_params()]
    base = [p for p in base if all(lo <= p.get(k, lo) <= hi for k, (lo, hi) in recipe.params.items())]
    if len(base) > PARAM_TRIES:
        picked = rng.sample(base, PARAM_TRIES)
        picked.sort(key=lambda p: tuple(sorted(p.items())))          # stable order whatever the sample order
        base = picked
    return base


# ---------------------------------------------------------------- planning
def plan_candidates(inv, seed=0, limit=MAX_CANDIDATES, recipes=None):
    """The best recipe builds for the inventory, best first. Deterministic for a given (inventory, seed)."""
    facts = facts_of(inv)
    rng = random.Random(seed)
    out, seen = [], set()
    for recipe in (recipes or all_recipes()):
        if not minimum_met(recipe, facts):
            continue
        found = []
        for params in _param_sets(recipe, facts, rng):
            if _size_ok(recipe, params, facts) is None:
                continue
            for rot in range(COLOUR_TRIES):
                colours = assign_colours(recipe, facts, rot)
                if not colours:
                    continue
                try:
                    c = make_candidate(recipe, params, inv, colours, facts)
                except ValueError:
                    c = None
                if c and c.signature not in seen:
                    seen.add(c.signature)
                    found.append(c)
        found.sort(key=lambda c: (-c.score, c.part_count, c.signature))
        out.extend(_diverse(found, PER_RECIPE))
    out.sort(key=lambda c: (-c.score, c.recipe_id, c.signature))
    return out[:limit]


def _diverse(found, n):
    """Up to n candidates that really differ (different size), best first."""
    picked, shapes = [], set()
    for c in found:
        shape = (c.part_count // 4, tuple(sorted(c.colours.values())))
        if shape in shapes:
            continue
        shapes.add(shape)
        picked.append(c)
        if len(picked) == n:
            break
    return picked


# ---------------------------------------------------------------- names
_WORD = re.compile(r"[a-z]+")


def name_fits(name, recipe):
    """A name fits a recipe when it contains one of the recipe's nouns and no noun of another recipe's category."""
    words = set(_WORD.findall((name or "").lower()))
    text = " " + " ".join(_WORD.findall((name or "").lower())) + " "
    if not any((" " + n + " " in text) or (n in words) for n in recipe.nouns):
        return False
    for other in all_recipes():
        if other.category == recipe.category:
            continue
        if any((" " + n + " " in text) or (n in words) for n in other.nouns):
            return False
    return True


def clean_name(name, recipe, colours):
    """The model's name when it fits the recipe, else the recipe's own template name."""
    name = re.sub(r"\s+", " ", str(name or "")).strip()[:60]
    return name if name and name_fits(name, recipe) else name_for(recipe, colours)


# ---------------------------------------------------------------- unlock hints
def _hint_for(recipe, facts):
    """What would unlock `recipe`: alternatives, each a list of {part, qty}. Empty list when the minimum is already met."""
    need_parts = max(0, recipe.min_parts - facts.total_parts)
    need_area = max(0, recipe.min_area - facts.total_area)
    alts = []
    if facts.longest < recipe.min_long:
        L = recipe.min_long
        alts.append([{"part": f"Brick 2 x {L}" if L in (3, 4, 6, 8, 10) else f"Brick 1 x {L}", "qty": 1}])
        alts.append([{"part": f"Plate 2 x {L}" if L in (3, 4, 6, 8, 10) else f"Plate 1 x {L}", "qty": 1}])
        return alts
    if need_parts or need_area:
        n24 = max(need_parts, -(-need_area // 8))
        n22 = max(need_parts, -(-need_area // 4))
        alts.append([{"part": "Brick 2 x 4", "qty": n24}])
        if n22 != n24:
            alts.append([{"part": "Brick 2 x 2", "qty": n22}])
    return alts


def unlock_hints(inv, limit=3):
    """The `limit` recipes closest to being unlocked: [{recipe, name, missing_parts, options}] (a gentle hint, never a blocker)."""
    facts = facts_of(inv)
    rows = []
    for r in all_recipes():
        if minimum_met(r, facts) or any(not SUPPORTED_NEEDS.get(n, lambda f: False)(facts) for n in r.needs):
            continue                                       # already offered, or needs something the grid cannot place (wheels)
        opts = _hint_for(r, facts)
        cost = min((sum(p["qty"] for p in o) for o in opts), default=0)
        rows.append((cost, r.id, {"recipe": r.id, "name": r.title, "options": opts, "cost": cost}))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in rows[:limit]]
