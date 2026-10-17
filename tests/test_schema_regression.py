"""The five hand-written recipes were converted to declarative JSON (backend/recipes/specs). Same regions, same builds as before:
tests/data/recipe_golden.json holds signatures of what the old Python recipes produced (region cells and the filled build) for two
inventories and every parameter variant."""

import json

import pytest

from backend.recipes import REGISTRY, get
from tests.golden_recipes import PATH, snapshot

OLD = ("bridge", "car", "house", "rocket", "tower")


def test_the_five_recipes_are_compiled_from_json_specs():
    for rid in OLD:
        r = get(rid)
        assert r is not None and r.spec is not None and r.spec["id"] == rid


def test_converted_recipes_give_exactly_the_same_regions_and_builds_as_the_old_code():
    golden = json.loads(PATH.read_text(encoding="utf-8"))
    now = snapshot([get(r) for r in OLD])
    assert set(now) == set(golden), sorted(set(now) ^ set(golden))[:5]
    bad = [k for k in golden if now[k] != golden[k]]
    assert not bad, f"{len(bad)} of {len(golden)} differ, first: {bad[:3]} {golden[bad[0]]} vs {now[bad[0]]}" if bad else ""
