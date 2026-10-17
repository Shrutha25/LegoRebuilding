"""Recipe builds: filler, recipes, planner, naming, Gemini role, unlock hints, set matching, renderer, API."""

import asyncio
import json
import random

import pandas as pd
import pytest

from backend import config
from backend.core import buildmodel as bm
from backend.core.buildmodel import Build, Inventory
from backend.core.render_iso import render_png
from backend.core.setmatch import SetMatcher, excluded_parts
from backend.recipes import REGISTRY, all_recipes, get
from backend.recipes.base import Opening, cells_region, hollow, rect, solid, stepped
from backend.recipes.facts import assign_colours, facts_of
from backend.recipes.filler import fill
from backend.recipes.inventories import hex_of, inventory_from, random_inventory
from backend.recipes.planner import Candidate, make_candidate, minimum_met, name_for, plan_candidates, unlock_hints
from backend.recipes.suggest import suggest


def run(coro):
    return asyncio.run(coro)


def rich(seed=3, size=110):
    return random_inventory(random.Random(seed), size)


def valid(parts, inv):
    return bm.validate(Build("t", "", parts), inv, 200)


# ---------------------------------------------------------------- regions
def test_regions_are_cell_sets_per_layer():
    s = solid("a", "r", 1, 2, 3, 2, 0, 2)
    assert len(s.layers) == 2 and s.layers[0] == rect(1, 2, 3, 2) and s.cell_count == 12 and s.top == 6
    h = hollow("w", "r", 0, 0, 5, 4, 1, 3, 1, [Opening("front", 2, 1, 0, 1)])
    assert len(h.layers[0]) == 14 - 1 and (2, 3) not in h.layers[0] and (2, 3) not in h.layers[1] and (2, 3) in h.layers[2]
    assert (1, 1) not in h.layers[0]                                       # the inside is empty
    thick = hollow("w", "r", 0, 0, 4, 4, 0, 1, 2)
    assert thick.layers[0] == rect(0, 0, 4, 4)                             # thickness 2 on 4 x 4 is solid
    st = stepped("roof", "r", 0, 0, 6, 5, 0, None, 1, "z", "plate")
    assert [len(c) for c in st.layers] == [30, 18, 6]                      # only z shrinks: 5, 3, 1 rows of 6
    st2 = stepped("nose", "r", 0, 0, 4, 4, 0, None, 1, "xz")
    assert [len(c) for c in st2.layers] == [16, 4]
    c = cells_region("c", "r", 0, [{(0, 0)}, [(1, 1)]])
    assert c.layers == [frozenset({(0, 0)}), frozenset({(1, 1)})]


# ---------------------------------------------------------------- filler
def test_filler_tiles_exactly_without_overlap_and_largest_first():
    inv = inventory_from([("3001", "Red", 4), ("3003", "Red", 4), ("3005", "Red", 4)])
    res = fill([solid("wall", "body", 0, 0, 8, 2, 0, 2)], inv, {"body": "Red"})
    assert res.required_ok and not valid(res.placements, inv).errors
    ground = [p for p in res.placements if p.y == 0]
    assert len(ground) == 2 and {p.part_id for p in ground} == {"3001"}          # the biggest part first
    occ, clashes = bm.occupancy(res.placements, inv)
    assert not clashes and len(occ) == 16 * 3 * 2                                  # every cell of both layers filled, nothing twice


def test_a_single_layer_of_bricks_side_by_side_is_not_one_model():
    inv = inventory_from([("3001", "Red", 4)])
    res = fill([solid("wall", "body", 0, 0, 8, 2, 0, 1)], inv, {"body": "Red"})
    assert len(res.placements) == 1 and not res.required_ok                          # nothing joins the two bricks: the loose one is dropped


def test_filler_offsets_seams_between_layers_running_bond():
    inv = inventory_from([("3001", "Red", 6), ("3003", "Red", 6)])
    res = fill([solid("wall", "body", 0, 0, 8, 2, 0, 2)], inv, {"body": "Red"})
    seams = {}
    for p in res.placements:
        fx, _ = inv.get(p.part_id, p.colour).dims.footprint(p.rotation)
        seams.setdefault(p.y, set()).update({p.x, p.x + fx} - {0, 8})
    assert seams[0] and seams[3] and not (seams[0] & seams[3])               # no vertical seam lines up with the one below


def test_filler_substitutes_stacked_plates_for_a_brick_and_shorter_parts_for_longer():
    plates = inventory_from([("3020", "Red", 9)])                               # only 2 x 4 plates
    res = fill([solid("wall", "body", 0, 0, 4, 2, 0, 1)], plates, {"body": "Red"})
    assert res.required_ok and len(res.placements) == 3 and {p.y for p in res.placements} == {0, 1, 2}      # a brick = 3 stacked plates
    small = inventory_from([("3003", "Red", 2), ("3020", "Red", 1)])           # no 2 x 4 brick: two 2 x 2 instead, a plate joins them
    res2 = fill([solid("wall", "body", 0, 0, 4, 2, 0, 1), solid("cap", "body", 0, 0, 4, 2, 3, 1, "plate")], small, {"body": "Red"})
    assert res2.required_ok and sorted(p.part_id for p in res2.placements) == ["3003", "3003", "3020"]


def test_filler_never_uses_a_part_twice_and_reports_missing_coverage():
    inv = inventory_from([("3001", "Red", 2)])
    res = fill([solid("tall", "body", 0, 0, 4, 2, 0, 3)], inv, {"body": "Red"})   # three layers need three bricks
    assert len(res.placements) == 2 and not res.required_ok and abs(res.regions[0].coverage - 2 / 3) < 1e-9
    used = {}
    for p in res.placements:
        used[(p.part_id, p.colour)] = used.get((p.part_id, p.colour), 0) + 1
    assert all(n <= inv.get(*k).qty for k, n in used.items())


def test_filler_prefers_the_role_colour_then_falls_back():
    inv = inventory_from([("3001", "Red", 1), ("3001", "Blue", 5)])
    res = fill([solid("w", "body", 0, 0, 4, 2, 0, 3)], inv, {"body": "Red"})
    colours = [p.colour for p in res.placements]
    assert colours.count("Red") == 1 and colours.count("Blue") == 2 and res.fallback_parts == 2
    assert res.role_parts["body"] == [3, 1]


def test_filler_never_hangs_a_part_in_the_air():
    inv = inventory_from([("3001", "Red", 3), ("3020", "Red", 6)])
    pillars = [solid("l", "body", 0, 0, 2, 2, 0, 1), solid("r", "body", 10, 0, 2, 2, 0, 1)]
    deck = solid("deck", "body", 1, 0, 11, 2, 3, 1, "plate")                  # longer than any plate: cannot be spanned
    res = fill(pillars + [deck], inv, {"body": "Red"})
    assert not valid(res.placements, inv).errors


def test_filler_is_deterministic():
    inv = rich()
    regions = get("house").generate(get("house").default_params(), facts_of(inv))
    a = fill(regions, inv, {"wall": "Red", "roof": "Blue", "base": "White"})
    b = fill(regions, inv, {"wall": "Red", "roof": "Blue", "base": "White"})
    assert a.placements == b.placements


def test_parts_nothing_sits_on_are_dropped_not_left_loose():
    inv = inventory_from([("3023", "Red", 6), ("3001", "Red", 4)])
    regions = [solid("base", "base", 0, 0, 6, 4, 0, 1, "plate", True, 0.3), solid("col", "body", 2, 1, 2, 2, 1, 2)]
    res = fill(regions, inv, {"base": "Red", "body": "Red"})
    assert bm.validate(Build("t", "", res.placements), inv).components == 1


# ---------------------------------------------------------------- recipes
def test_the_five_recipes_are_registered_with_a_noun_vocabulary():
    assert {"car", "house", "tower", "bridge", "rocket"} <= set(REGISTRY)
    for r in all_recipes():
        assert r.nouns and r.roles and r.params


@pytest.mark.parametrize("rid", ["car", "house", "tower", "bridge", "rocket"])
def test_every_recipe_builds_valid_models_with_its_required_regions(rid):
    recipe = get(rid)
    made = 0
    for seed in range(1, 9):
        inv = rich(seed, 100)
        facts = facts_of(inv)
        for params in recipe.variants(facts)[:12]:
            c = make_candidate(recipe, params, inv, assign_colours(recipe, facts), facts)
            if c is None:
                continue
            made += 1
            assert c.coverage >= 0.7 and c.part_count >= recipe.min_parts
            rep = valid(c.build.parts, inv)
            assert not rep.errors and rep.components == 1 and c.repaired == 0         # recipe builds need no repair
            assert all(r[2] for r in c.regions if r[0] in ("walls", "chassis", "body", "deck"))
    assert made >= 8


@pytest.mark.parametrize("rid", ["car", "house", "tower", "bridge", "rocket"])
def test_every_recipe_has_at_least_two_variants_by_parameters(rid):
    recipe = get(rid)
    inv = rich(5, 120)
    facts = facts_of(inv)
    shapes = set()
    for params in recipe.variants(facts):
        regions = recipe.generate(params, facts)
        shapes.add(tuple(sorted((r.name, r.cell_count) for r in regions)))
    assert len(shapes) >= 2


def test_a_recipe_rejects_parameters_outside_its_ranges():
    with pytest.raises(ValueError):
        get("car").check_params({"length": 99, "width": 2, "cabin": 1})
    inv = rich()
    with pytest.raises(ValueError):
        make_candidate(get("car"), {"length": 99, "width": 2, "cabin": 1}, inv, {"body": "Red"})


def test_the_house_has_hollow_walls_with_a_door_gap_and_a_roof():
    recipe = get("house")
    regions = {r.name: r for r in recipe.generate({"width": 6, "depth": 5, "wall_layers": 3, "roof": 1}, facts_of(rich()))}
    walls = regions["walls"]
    assert walls.kind == "hollow" and (3, 5) not in walls.layers[0] and (4, 5) not in walls.layers[0] and (3, 5) in walls.layers[2]
    assert (3, 3) not in walls.layers[0]                                       # hollow inside
    assert regions["roof"].kind == "stepped" and len(regions["roof"].layers) == 4        # eaves 7 deep, rising to a 1-row ridge
    assert (0, 0) in regions["roof"].layers[0] and (0, 0) not in walls.layers[0]            # the roof overhangs the walls


def test_the_car_has_wheels_at_the_four_corners_in_the_darkest_colour():
    recipe = get("car")
    inv = inventory_from([("3001", "Red", 8), ("3003", "Red", 6), ("3003", "Black", 6), ("3020", "White", 4), ("3010", "Red", 4), ("3004", "Black", 4), ("3004", "Red", 6), ("3005", "Black", 4), ("3005", "Red", 4), ("3795", "White", 2), ("3666", "Red", 2)])
    facts = facts_of(inv)
    colours = assign_colours(recipe, facts)
    assert colours["wheel"] == "Black"
    c = make_candidate(recipe, {"length": 6, "width": 3, "cabin": 1}, inv, colours, facts)
    assert c is not None
    wheels = [p for p in c.build.parts if p.y == 0]
    assert wheels and all(p.colour == "Black" for p in wheels)
    xs = {p.x for p in wheels}
    assert min(xs) <= 1 and max(xs) >= 4                                        # front and rear axle


def test_the_bridge_deck_overlaps_both_pillars_and_the_tower_alternates_colours():
    inv = inventory_from([("3001", "Red", 8), ("3003", "Red", 8), ("3003", "Blue", 8), ("3795", "Blue", 3), ("3034", "Blue", 3), ("3020", "Blue", 4), ("3710", "Red", 4), ("3031", "Red", 2), ("3005", "Red", 4)])
    facts = facts_of(inv)
    bridge = get("bridge")
    c = make_candidate(bridge, {"gap": 4, "width": 2, "height": 3}, inv, {"pillar": "Red", "deck": "Blue", "rail": "Red"}, facts)
    assert c is not None
    deck = [p for p in c.build.parts if p.y == 9]
    assert any(inv.get(p.part_id, p.colour).dims.long >= 6 for p in deck)       # one part reaches from pillar to pillar
    tower = get("tower")
    t = make_candidate(tower, {"depth": 2, "layers": 6, "band": 1}, inv, {"body": "Red", "accent": "Blue", "base": "Blue"}, facts)
    assert t is not None
    bands = [next(p.colour for p in t.build.parts if p.y == y) for y in (1, 4, 7, 10, 13, 16)]
    assert bands == ["Red", "Blue", "Red", "Blue", "Red", "Blue"]


def test_the_rocket_has_a_base_fins_and_a_nose():
    inv = rich(14, 120)
    facts = facts_of(inv)
    c = make_candidate(get("rocket"), {"core_layers": 4, "fin_layers": 2}, inv, assign_colours(get("rocket"), facts), facts)
    assert c is not None and c.part_count >= 8
    names = {r[0]: r[2] for r in c.regions}
    assert names["base"] and names["core-lower"] and names["core-upper"]


# ---------------------------------------------------------------- planner
def test_minimum_parts_reject_a_tiny_inventory():
    tiny = inventory_from([("3001", "Red", 2)])
    assert plan_candidates(tiny) == []
    assert not any(minimum_met(r, facts_of(tiny)) for r in all_recipes())


def test_plan_candidates_is_deterministic_capped_and_best_first():
    inv = rich(9, 100)
    a = plan_candidates(inv, seed=4)
    b = plan_candidates(inv, seed=4)
    assert [c.signature for c in a] == [c.signature for c in b] and [c.build.parts for c in a] == [c.build.parts for c in b]
    assert 1 <= len(a) <= 10 and [c.score for c in a] == sorted((c.score for c in a), reverse=True)
    per = {}
    for c in a:
        per[c.recipe_id] = per.get(c.recipe_id, 0) + 1
    assert max(per.values()) <= 3
    for c in a:
        assert not valid(c.build.parts, inv).errors and c.repaired == 0


def test_most_realistic_inventories_get_a_recipe_build():
    got = 0
    for seed in range(30):
        rng = random.Random(seed)
        inv = random_inventory(rng, rng.randint(40, 120))
        got += bool(plan_candidates(inv, seed, 1))
    assert got >= 28


# ---------------------------------------------------------------- names
def test_build_names_are_evocative_not_colour_plus_category():
    from backend.recipes.categories import is_colour_category_name
    for r in all_recipes():
        assert r.names, r.id
        for n in r.names:
            assert not is_colour_category_name(n, r.category, ["Red", "Blue"]), n
    assert is_colour_category_name("Red House", "building", ["Red"]) and is_colour_category_name("blue tower", "tower", ["Blue"])
    assert not is_colour_category_name("Lantern Cottage", "building", ["Red"]) and not is_colour_category_name("Desert Rover", "vehicle", [])
    house = get("house")
    assert name_for(house, {"wall": "Red"}, {"width": 4}) in house.names and "Red" not in name_for(house, {"wall": "Red"}, {"width": 4})
    assert len({name_for(house, None, {"width": w}) for w in range(4, 9)}) > 1                    # different shapes, different names


# ---------------------------------------------------------------- Gemini's role: concepts only
class FakeLLM:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    async def complete(self, system, user, schema):
        self.prompts.append((system, user, schema))
        out = self.answers.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def concept_json(*items):
    return json.dumps({"concepts": [{"name": n, "category": c, "description": "d", "silhouette": s, "bbox": {"width": w, "depth": d, "height": h},
                                     "features": f} for n, c, s, (w, d, h), f in items]})


def test_one_gemini_call_with_concepts_and_no_coordinates_in_the_prompt():
    from backend.recipes.concepts import SCHEMA
    inv = rich(9, 100)
    names = {"Lantern Cottage", "Stonefoot Crossing", "Moon Skipper", "Old Oak", "Puddle Duck"}
    llm = FakeLLM(concept_json(("Lantern Cottage", "building", "wide-squat", (8, 5, 6), ["a sloped roof"]),
                               ("Stonefoot Crossing", "bridge", "long-low", (10, 3, 3), ["an arch span"]),
                               ("Moon Skipper", "spacecraft", "tall-narrow", (4, 4, 14), ["fins"]),
                               ("Old Oak", "plant", "wide-squat", (6, 6, 8), []), ("Puddle Duck", "animal", "compact", (3, 4, 5), ["a head"])))
    out = run(suggest(inv, llm, seed=0, show=5))
    assert len(llm.prompts) == 1 and llm.prompts[0][2] is SCHEMA
    user = llm.prompts[0][1]
    assert "pieces" in user and "x=" not in user and "z=" not in user and "rotation" not in user.lower()
    assert out.ai_used and out.shown and all(s.name in names or not s.from_ai for s in out.shown)


def test_concepts_from_gemini_are_checked_and_follow_the_offered_categories():
    from backend.recipes.concepts import clean_concept, feasible_categories
    inv = rich(9, 100)
    facts = facts_of(inv)
    feasible = feasible_categories(facts, all_recipes())
    assert "vehicle" not in feasible                              # no wheel parts the grid can place: vehicles are never offered
    good = {"name": "Mossy Footbridge", "category": "bridge", "description": "d", "silhouette": "long-low", "bbox": {"width": 8, "depth": 3, "height": 3}, "features": []}
    c, why = clean_concept(good, feasible, facts.colours, all_recipes())
    assert c is not None and c.source == "gemini" and c.bbox == (8, 3, 3)
    for change in ({"category": "vehicle"}, {"name": "Red Bridge"}, {"name": "x"}, {"silhouette": "round"},
                   {"bbox": {"width": 99, "depth": 3, "height": 3}}, {"bbox": {"width": 1}}):
        assert clean_concept({**good, **change}, feasible, facts.colours, all_recipes())[0] is None, change


@pytest.mark.parametrize("answer", [RuntimeError("budget"), "not json", json.dumps({"concepts": []}), json.dumps({"concepts": [{"name": "Red House"}]})])
def test_without_gemini_or_with_nonsense_the_fallback_concepts_still_give_builds(answer):
    inv = rich(9, 100)
    out = run(suggest(inv, FakeLLM(answer), seed=0, show=5))
    assert len(out.shown) >= 3 and not out.ai_used
    for s in out.shown:
        assert s.name in s.candidate.recipe.names and not s.from_ai


def test_no_llm_at_all_is_fine():
    out = run(suggest(rich(9, 100), None, seed=0))
    assert out.shown and not out.ai_used and out.unlock is not None


def test_a_concept_decides_the_shape_the_same_recipe_gives():
    """Parameterised recipes: different silhouettes and bounding boxes give clearly different builds from the same recipe."""
    from backend.recipes.checks import iou, voxels_of_build
    from backend.recipes.concepts import Concept, nearest_params
    from backend.recipes.suggest import build_concept
    inv = rich(9, 130)
    facts = facts_of(inv)
    recipes = [get("house")]
    shapes = {}
    for sil, bbox in (("wide-squat", (8, 5, 4)), ("tall-narrow", (4, 4, 11)), ("compact", (4, 5, 6))):
        c = Concept("x" + sil, "building", "", sil, bbox, (), "gemini")
        cand, why = build_concept(c, recipes, inv, facts)
        assert cand is not None, why
        shapes[sil] = (cand.params, voxels_of_build(cand.build, inv))
    assert len({tuple(sorted(p.items())) for p, _ in shapes.values()}) == 3
    assert iou(shapes["wide-squat"][1], shapes["tall-narrow"][1]) < 0.6
    wide = Concept("w", "building", "", "wide-squat", (8, 5, 4), (), "gemini")
    tall = Concept("t", "building", "", "tall-narrow", (4, 4, 11), (), "gemini")
    assert nearest_params(get("house"), wide, facts)[0]["width"] > nearest_params(get("house"), tall, facts)[0]["width"]
    assert nearest_params(get("house"), tall, facts)[0]["wall_layers"] >= nearest_params(get("house"), wide, facts)[0]["wall_layers"]


# ---------------------------------------------------------------- unlock hints
def test_unlock_hints_say_what_to_scan_for_the_closest_recipes():
    small = inventory_from([("3003", "Red", 5), ("3004", "Red", 2)])                  # 7 parts, longest 2: nothing is unlocked
    hints = unlock_hints(small)
    assert len(hints) == 3 and all(h["options"] for h in hints) and [h["cost"] for h in hints] == sorted(h["cost"] for h in hints)
    some = {tuple(o[0]["part"] for o in h["options"]) for h in hints}
    assert any("Brick 2 x 4" in t or "Brick 2 x 2" in t for t in some)
    long_missing = inventory_from([("3003", "Red", 20)])                               # plenty of parts, but no long part
    h = {x["recipe"]: x for x in unlock_hints(long_missing, 5)}
    assert "car" not in h                                                               # needs wheels the grid cannot place: never a hint
    assert h and all(o for x in h.values() for o in x["options"])
    assert unlock_hints(rich(9, 120)) == [] or all(x["recipe"] not in {c.recipe_id for c in plan_candidates(rich(9, 120))} for x in unlock_hints(rich(9, 120)))


# ---------------------------------------------------------------- renderer
def test_the_isometric_renderer_gives_a_png_per_build():
    inv = rich(9, 100)
    c = plan_candidates(inv, seed=0)[0]
    png = render_png(c.build, inv, hex_of, (300, 240))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(png))
    assert im.size == (300, 240) and len(im.convert("RGB").getcolors(maxcolors=1000000)) > 10
    assert render_png(Build("e", "", []), inv, hex_of, (50, 40))[:4] == b"\x89PNG"


# ---------------------------------------------------------------- set matching (small fixture)
class FakeIndex:
    def __init__(self):
        self.sets = pd.DataFrame({"set_num": ["1-1", "2-1", "3-1", "4-1"], "name": ["Tiny Car", "Small House", "Big Ship", "Sticker Set"],
                                  "year": ["2000", "2001", "2002", "2003"], "img_url": ["", "", "", ""], "theme_id": ["1", "1", "1", "1"]})
        self._theme = {s: "1" for s in self.sets.set_num}
        self.skip = set()
        base = {"3001": 2, "3003": 4, "3020": 2, "3004": 3}                     # 11 parts
        self.req = {
            False: {"1-1": dict(base), "2-1": {"3001": 4, "3003": 4, "3023": 4}, "3-1": {"3001": 30, "3010": 30}, "4-1": {"90000": 20, "3001": 1}},
            True: {"1-1": {(k, "red"): v for k, v in base.items()}, "2-1": {("3001", "red"): 4, ("3003", "blue"): 4, ("3023", "red"): 4},
                   "3-1": {("3001", "red"): 30, ("3010", "red"): 30}, "4-1": {("90000", "red"): 20, ("3001", "red"): 1}},
        }

    def is_buildable(self, set_num):
        return True

    def canon(self, p):
        return {"3001a": "3001"}.get(p, p)

    def image_for(self, part, colour):
        return None


@pytest.fixture
def matcher(monkeypatch):
    monkeypatch.setattr(config, "MIN_SET_PARTS", 8)           # the minimum these expectations were written for
    parts = pd.DataFrame({"part_num": ["3001", "3003", "3020", "3004", "3023", "3010", "90000"],
                          "name": ["Brick 2 x 4", "Brick 2 x 2", "Plate 2 x 4", "Brick 1 x 2", "Plate 1 x 2", "Brick 1 x 4", "Sticker Sheet for Set 4-1"],
                          "part_cat_id": ["11", "11", "14", "11", "14", "11", "58"]})
    return SetMatcher(FakeIndex(), parts)


def test_stickers_and_minifig_parts_are_excluded():
    parts = pd.DataFrame({"part_num": ["a", "b", "c", "d"], "name": ["Brick 1 x 1", "Sticker Sheet", "Torso with Print", "Head"],
                          "part_cat_id": ["11", "58", "60", "59"]})
    assert excluded_parts(parts) == {"b", "c", "d"}


def test_sets_you_can_build_and_almost_build_with_the_missing_parts(matcher):
    have = [{"part_num": "3001", "color": "Red", "qty": 2}, {"part_num": "3003", "color": "Red", "qty": 4}, {"part_num": "3020", "color": "Red", "qty": 2},
            {"part_num": "3004", "color": "Red", "qty": 3}]
    r = matcher.match(have, {"3003": "Brick 2 x 2"})
    ids = [s["set_num"] for s in r["can_build"]]
    assert ids == ["1-1"] and r["can_build"][0]["colour_match"] == "exact"
    assert r["can_build"][0]["exact_colour"] == {"have": 11, "need": 11} and r["can_build"][0]["any_colour"] == {"have": 11, "need": 11}
    near = {s["set_num"]: s for s in r["almost"]}
    assert "2-1" not in near                                                      # 2-1 is missing 6 parts (not <= 3, share 60%)
    have.append({"part_num": "3023", "color": "Red", "qty": 3})
    have[0]["qty"] = 4
    r2 = matcher.match(have, {"3023": "Plate 1 x 2"})
    near = {s["set_num"]: s for s in r2["almost"]}
    assert near["2-1"]["missing_count"] == 1 and near["2-1"]["missing"][0]["part"] == "3023" or near["2-1"]["missing"][0]["qty"] == 1
    assert "4-1" not in ids and "4-1" not in near                                   # the sticker sheet does not count towards the set


def test_exact_colour_and_any_colour_are_both_reported(matcher):
    have = [{"part_num": "3001", "color": "Blue", "qty": 2}, {"part_num": "3003", "color": "Blue", "qty": 4}, {"part_num": "3020", "color": "Blue", "qty": 2},
            {"part_num": "3004", "color": "Blue", "qty": 3}]
    r = matcher.match(have)
    row = r["can_build"][0]
    assert row["set_num"] == "1-1" and row["colour_match"] == "any" and row["exact_colour"]["have"] == 0 and row["any_colour"]["have"] == 11


def test_one_colour_rule_for_set_matching_exact_when_on_part_only_when_off(matcher):
    have = [{"part_num": "3001", "color": "Blue", "qty": 2}, {"part_num": "3003", "color": "Blue", "qty": 4}, {"part_num": "3020", "color": "Blue", "qty": 2},
            {"part_num": "3004", "color": "Blue", "qty": 3}]
    off = matcher.match(have, match_color=False)
    on = matcher.match(have, match_color=True)
    assert [r["set_num"] for r in off["can_build"]] == ["1-1"] and off["can_build"][0]["colour_match"] == "any"
    assert on["can_build"] == [] and all(r["set_num"] != "1-1" for r in on["can_build"])               # the set is red: not buildable with blue pieces
    red = [dict(p, color="Red") for p in have]
    both = matcher.match(red)
    assert matcher.match(red, match_color=True)["can_build"][0]["colour_match"] == "exact" and both["can_build"][0]["colour_match"] == "exact"


def test_set_matching_is_fast_on_the_real_data_when_present():
    from pathlib import Path
    data = Path(__file__).resolve().parents[1] / "rb_data"
    if not (data / "inventory_parts.csv").exists():
        pytest.skip("Rebrickable data not downloaded")
    import time
    from backend.core.index import MatchIndex
    idx = MatchIndex.load(str(data))
    m = SetMatcher(idx, pd.read_csv(data / "parts.csv", dtype=str, keep_default_na=False))
    inv = [{"part_num": p, "color": c, "qty": 3} for p in ("3001", "3002", "3003", "3004", "3005", "3010", "3020", "3022", "3023", "3024", "3710", "3039")
           for c in ("Red", "Blue", "Yellow", "White", "Black")]
    t = time.perf_counter()
    r = m.match(inv)
    assert (time.perf_counter() - t) < 0.3 and r["can_build"] is not None


# ---------------------------------------------------------------- the HTTP API
from tests.test_builds import INV, client, lib  # noqa: E402,F401  (fixtures: a fake LDraw library, the app with a small data set)


def _parts():
    return [{"part_num": i.part_id, "name": i.name, "color": i.colour, "qty": i.qty * 3} for i in INV.parts() if i.dims.studs_top]


def test_api_recipe_builds_without_gemini_use_templates_and_the_viewer_format(client):
    r = client.post("/api/builds/recipes", json={"parts": _parts(), "use_ai": False})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["builds"] and not data["ai_used"] and "timings_ms" in data and isinstance(data["unlock"], list)
    b = data["builds"][0]
    assert b["kind"] == "ai" and b["recipe"]["id"] in REGISTRY and b["step_count"] >= 3 and b["part_count"] >= 8
    assert b["title"] in get(b["recipe"]["id"]).names and b["recipe"]["concept"]["source"] == "fallback"
    assert client.get(b["model_url"]).text.startswith("0 FILE model.ldr")
    png = client.get(b["thumb_url"])
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG" and "immutable" in png.headers["cache-control"]
    again = client.post("/api/builds/recipes", json={"parts": _parts(), "use_ai": False}).json()
    assert [x["id"] for x in again["builds"]] == [x["id"] for x in data["builds"]]       # the same pieces give the same builds


def test_api_recipe_builds_use_one_gemini_call_when_a_key_exists(client):
    client.llm.answers[:] = [concept_json(("Lantern Cottage", "building", "wide-squat", (8, 5, 6), []), ("Mossy Span", "bridge", "long-low", (8, 3, 3), []),
                                          ("Quiet Spire", "tower", "tall-narrow", (2, 2, 12), []), ("Pond Paddler", "animal", "compact", (3, 4, 4), []),
                                          ("Round Hedge", "plant", "wide-squat", (5, 5, 4), []))]
    r = client.post("/api/builds/recipes", json={"parts": _parts()}).json()
    assert r["ai_used"] and len(client.llm.prompts) == 1 and r["builds"]
    mine = {"Lantern Cottage", "Mossy Span", "Quiet Spire", "Pond Paddler", "Round Hedge"}
    assert {b["title"] for b in r["builds"]} & mine                                      # Gemini's names are used for the concepts it proposed
    assert all(b["title"] in mine or b["recipe"]["concept"]["source"] == "fallback" for b in r["builds"])   # the rest are the recipes' own names


def test_api_recipe_builds_survive_a_failing_gemini(client):
    client.llm.answers[:] = [RuntimeError("boom")]
    r = client.post("/api/builds/recipes", json={"parts": _parts()})
    assert r.status_code == 200 and r.json()["builds"] and not r.json()["ai_used"]


def test_api_thumbnail_ids_are_validated(client):
    assert client.get("/api/builds/ai-000000000000/thumb.png").status_code == 404
    assert client.get("/api/builds/..%2Fx/thumb.png").status_code in (404, 422)


def test_api_nearby_sets_have_the_shape_the_set_card_uses(client):
    r = client.post("/api/sets/nearby", json={"parts": _parts()})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"can_build", "almost", "timings_ms"}
    for row in d["can_build"] + d["almost"]:
        assert {"set_num", "name", "parts_you_have", "coverage", "complete", "missing", "instructions_url", "rebrickable_url"} <= set(row)


def test_the_recipe_gallery_is_dev_only(client, monkeypatch):
    from backend import config
    assert client.get("/api/dev/recipe-gallery").status_code == 404
    monkeypatch.setattr(config, "RECIPE_GALLERY", True)
    r = client.get("/api/dev/recipe-gallery?seed=2&per=1")
    assert r.status_code == 200 and {x["recipe"] for x in r.json()["recipes"]} == set(REGISTRY)


# ---------------------------------------------------------------- "does it look like its name?"
class VisionLLM:
    def __init__(self, text_answers=(), image_answers=()):
        self.text, self.images, self.image_calls = list(text_answers), list(image_answers), []

    async def complete(self, system, user, schema):
        return self.text.pop(0)

    async def complete_with_images(self, system, user, schema, images):
        self.image_calls.append((user, len(images)))
        out = self.images.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_contact_sheets_and_grading_ignore_unknown_labels():
    from backend.recipes.vision import contact_sheet, grade
    inv = rich(9, 100)
    c = plan_candidates(inv, seed=0)[0]
    png = render_png(c.build, inv, hex_of, (300, 240))
    sheet = contact_sheet([png, png, png], ["a-1", "a-2", "a-3"])
    assert sheet[:4] == b"\x89PNG"
    llm = VisionLLM(image_answers=[json.dumps({"answers": [
        {"label": "a-1", "verdict": "yes", "note": "clear"}, {"label": "a-2", "verdict": "maybe", "note": ""},
        {"label": "zzz", "verdict": "no", "note": ""}, {"label": "a-3", "verdict": "no", "note": "a pile"}]})])
    out = run(grade(llm, [sheet], {"a-1": "car", "a-2": "car", "a-3": "car"}))
    assert out == {"a-1": ("yes", "clear"), "a-3": ("no", "a pile")} and llm.image_calls[0][1] == 1


def test_qa_renders_grids_of_four_and_writes_a_report_without_gemini(tmp_path):
    from backend.recipes import qa
    r = get("tower")
    rows = qa.generate(r, 6, 1)
    grids = qa.make_grids(r, rows, 4, tmp_path)
    assert sum(len(g[2]) for g in grids) == sum(1 for x in rows if x[3]) and all(len(g[2]) <= 4 for g in grids)
    assert all((tmp_path / f"{g[0]}.png").exists() for g in grids)
    verdicts = {rows[0][0]: ("yes", ""), rows[1][0]: ("no", "a plain stack")}

    class A:
        inventories, per_grid = 6, 4
    qa.write_report(tmp_path / "r.md", [r], {r.id: rows}, verdicts, A, 1, "", False)
    text = (tmp_path / "r.md").read_text(encoding="utf-8")
    assert "## tower" in text and "yes 1" in text and "no 1" in text and "a plain stack" in text


def test_freeform_fallback_is_only_used_when_the_flag_is_on_and_a_false_name_is_replaced(client, monkeypatch):
    from backend import builds_api, config
    from tests.test_builds import fixture_json
    tiny = [{"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 2}, {"part_num": "3003", "name": "Brick 2 x 2", "color": "Red", "qty": 3}]
    off = client.post("/api/builds/recipes", json={"parts": tiny, "use_ai": False}).json()
    assert off["builds"] == [] and not off["freeform"]                                  # flag off (default): no free-form build
    monkeypatch.setattr(config, "AI_FREEFORM_BUILDS", True)
    llm = VisionLLM([fixture_json()], [json.dumps({"answers": [{"label": "1", "verdict": "no", "note": "random blocks"}]})])
    monkeypatch.setitem(builds_api._deps, "get_llm", lambda: llm)
    from backend.recipes import suggest as suggest_mod
    monkeypatch.setattr(suggest_mod, "feasible_categories", lambda f, r: {})          # nothing fits: only then the free-form fallback is tried
    on = client.post("/api/builds/recipes", json={"parts": [dict(p, qty=p["qty"] * 3) for p in tiny]}).json()
    assert on["freeform"] and len(on["builds"]) == 1 and len(llm.image_calls) == 1       # one vision check for the free-form build
    assert on["builds"][0]["title"] == "Abstract build" and on["builds"][0]["look_check"] == "no"
    assert client.get(f"/api/builds/{on['builds'][0]['id']}").json()["title"] == "Abstract build"


def test_the_default_for_free_form_builds_is_off():
    import importlib
    import os
    from backend import config
    old = os.environ.pop("AI_FREEFORM_BUILDS", None)
    try:
        assert importlib.reload(config).AI_FREEFORM_BUILDS is False
        os.environ["AI_FREEFORM_BUILDS"] = "1"
        assert importlib.reload(config).AI_FREEFORM_BUILDS is True
    finally:
        os.environ.pop("AI_FREEFORM_BUILDS", None)
        if old is not None:
            os.environ["AI_FREEFORM_BUILDS"] = old
        importlib.reload(config)
