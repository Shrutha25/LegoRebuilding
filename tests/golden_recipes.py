"""Signatures of what the five hand-written recipes produce, for the regression test of the declarative schema.

    python -m tests.golden_recipes        rewrites tests/data/recipe_golden.json (only do this on purpose)
"""

import hashlib
import json
from pathlib import Path

from backend.recipes import all_recipes
from backend.recipes.facts import assign_colours, facts_of
from backend.recipes.filler import fill
from backend.recipes.inventories import random_inventory

PATH = Path(__file__).parent / "data" / "recipe_golden.json"
INVENTORIES = (7, 11)


def sha(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=list).encode("utf-8")).hexdigest()[:16]


def region_sig(regions):
    return sha([[r.name, r.role, r.y, r.unit, r.required, r.min_coverage, r.kind, [sorted(c) for c in r.layers]] for r in regions])


def snapshot(recipes=None):
    out = {}
    for seed in INVENTORIES:
        inv = random_inventory(seed, 120)
        facts = facts_of(inv)
        for recipe in (recipes or all_recipes()):
            sets = recipe.variants(facts) if recipe.variants else [recipe.default_params()]
            for params in sets:
                key = f"{seed}/{recipe.id}/" + ",".join(f"{k}={v}" for k, v in sorted(params.items()))
                regions = recipe.generate(params, facts)
                colours = assign_colours(recipe, facts, 0)
                res = fill(regions, inv, colours)
                out[key] = {"regions": region_sig(regions),
                            "build": sha(sorted((p.part_id, p.colour, p.x, p.z, p.y, p.rotation) for p in res.placements))}
    return out


if __name__ == "__main__":
    PATH.write_text(json.dumps(snapshot(), indent=1, sort_keys=True), encoding="utf-8")
    print("wrote", PATH)
