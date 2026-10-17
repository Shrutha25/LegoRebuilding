"""Declarative recipes: validation, primitives, on-demand generation with repair, recognizability, the recipe library, structure/feature/IoU checks."""

import asyncio
import copy
import json
import random

import pytest

from backend.core import buildmodel as bm
from backend.core.buildmodel import Build, Placement
from backend.recipes import REGISTRY, all_recipes, get
from backend.recipes.categories import feature_tags, match_category
from backend.recipes.checks import IOU_MAX, feature_problems, iou, normalise, structural_problems, voxels_of_build, voxels_of_regions
from backend.recipes.concepts import Concept, feasible_categories
from backend.recipes.facts import assign_colours, facts_of
from backend.recipes.generate import MAX_REPAIRS, generate_recipe, parse_answer, tally
from backend.recipes.inventories import hex_of, inventory_from, random_inventory
from backend.recipes.library import RecipeLibrary
from backend.recipes.planner import make_candidate
from backend.recipes.primitives import PRIMITIVES, Ctx, describe_all
from backend.recipes.schema import SpecError, compile_spec, generate_regions, validate_spec


def run(coro):
    return asyncio.run(coro)


def inv9():
    """A realistic bucket plus a dozen 1 x 1 bricks, which small furniture legs and posts need (buckets usually have plenty)."""
    from backend.core.buildmodel import Inventory
    from backend.recipes.inventories import make_item
    base = random_inventory(random.Random(3), 140)
    colour = facts_of(base).colours[0]
    return Inventory(base.parts() + [make_item("3005", "Brick 1 x 1", colour, 12), make_item("3024", "Plate 1 x 1", colour, 6)])


def spec(rid="chair"):
    return copy.deepcopy(get(rid).spec)


CONCEPT = Concept("Reading Nook", "furniture", "a small chair", "compact", (3, 3, 6), ("a backrest",), "gemini")


class LLM:
    """Scripted Gemini: `recipes` answer complete(), `looks` answer complete_with_images()."""

    def __init__(self, recipes, looks=("a chair",)):
        self.recipes, self.looks, self.prompts = list(recipes), list(looks), []

    async def complete(self, system, user, schema):
        self.prompts.append(user)
        out = self.recipes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps({"recipe": json.dumps(out)})

    async def complete_with_images(self, system, user, schema, images):
        self.prompts.append(user)
        assert images and images[0][:4] == b"\x89PNG" and "chair" not in user.lower() and "furniture" not in user.lower()      # name and category are NOT given
        return json.dumps({"answer": self.looks.pop(0)})


# ---------------------------------------------------------------- the library of primitives
def test_every_primitive_is_documented_and_builds_cells():
    assert {"box", "tapered_box", "slope_roof", "arch", "pillar", "cylinder", "cone", "wing", "wheel_axle", "wall_with_window", "stairs", "leg", "head"} <= set(PRIMITIVES)
    text = describe_all()
    for name in PRIMITIVES:
        assert f"- {name}(" in text
    samples = {"box": {"w": 3, "d": 2, "layers": 2}, "tapered_box": {"w": 4, "d": 4}, "slope_roof": {"w": 6, "d": 6}, "arch": {"span": 3, "layers": 4},
               "pillar": {"layers": 3}, "leg": {"layers": 2}, "cylinder": {"diameter": 4, "layers": 2}, "cone": {"base": 5},
               "wing": {"span": 6, "chord": 3, "taper": 2, "sym": 1}, "stairs": {"width": 2, "steps": 4}, "head": {"w": 3, "d": 2, "layers": 1, "ears": 1},
               "wall_with_window": {"w": 6, "d": 4, "layers": 3, "windows": 1}, "wheel_axle": {"length": 6, "width": 4}}
    for name, a in samples.items():
        p = PRIMITIVES[name]
        regions = p.build(Ctx("t", "r", "brick", 0, 0, 0, True, 0.9, None), {**p.defaults, **a})
        assert regions and any(c for r in regions for layer in r.layers for c in layer), name
        assert p.size is None or all(v >= 1 for v in p.size({**p.defaults, **a})), name


def test_the_arch_has_an_opening_that_closes_at_the_top_and_the_cone_narrows_to_one_stud():
    r = PRIMITIVES["arch"].build(Ctx("a", "r", "brick", 0, 0, 0, True, 0.9, None), {**PRIMITIVES["arch"].defaults, "span": 4, "layers": 4})[0]
    sizes = [len(layer) for layer in r.layers]
    assert sizes[0] == 4 and sizes[-1] == 12 and sizes == sorted(sizes)                       # two legs, an opening that narrows, a closed top (a 6 x 2 block)
    c = PRIMITIVES["cone"].build(Ctx("c", "r", "brick", 0, 0, 0, True, 0.9, None), {**PRIMITIVES["cone"].defaults, "base": 5})[0]
    assert [len(layer) for layer in c.layers][-1] == 1 and len(c.layers[0]) > len(c.layers[-1])


# ---------------------------------------------------------------- schema validation
def test_a_valid_spec_has_no_problems_and_compiles():
    for r in all_recipes():
        assert validate_spec(r.spec) == [], r.id


def test_validation_names_each_problem_in_plain_language():
    s = spec()
    s["parts"][1]["primitive"] = "teapot"
    assert any("unknown primitive" in e and "teapot" in e for e in validate_spec(s))
    s = spec()
    s["parts"][2]["attach"]["to"] = "roof-of-the-world"
    assert any("EARLIER part" in e for e in validate_spec(s))
    s = spec()
    s["parts"][1]["args"]["w"] = "width + banana"
    assert any("banana" in e for e in validate_spec(s))
    s = spec()
    s["parts"][1]["args"]["w"] = "__import__('os').system('x')"
    assert validate_spec(s)
    s = spec()
    s["parts"][1]["args"]["w"] = 99
    assert any("studs" in e for e in validate_spec(s))
    s = spec()
    s["roles"].append({"name": "wheel", "want": "dark"})
    assert any("wheels cannot be made from plain bricks" in e for e in validate_spec(s))
    s = spec()
    s["parts"][0]["role"] = "nobody"
    assert any("role must be one of" in e for e in validate_spec(s))
    s = spec()
    s["params"]["width"]["range"] = [9, 2]
    assert any("range" in e for e in validate_spec(s))
    s = spec()
    s["surprise"] = 1
    assert any("unknown top-level key" in e for e in validate_spec(s))
    with pytest.raises(SpecError):
        compile_spec({"id": "x"})
    assert validate_spec("not a dict") == ["the recipe must be a JSON object"]


def test_expressions_cannot_do_anything_but_arithmetic():
    from backend.recipes.expr import ExprError, evaluate
    assert evaluate("max(2, L // 2) + (1 if thin and w >= 4 else 2)", {"L": 9, "thin": True, "w": 5}) == 5
    for bad in ("open('x')", "L.real", "'a' + 'b'", "[1, 2]", "lambda: 1", "L ** 9", "1 / 0", "x"):
        with pytest.raises(ExprError):
            evaluate(bad, {"L": 1})


def test_attachments_place_a_part_on_top_of_beside_in_front_of_and_behind_an_earlier_one():
    s = {"id": "att", "category": "x", "roles": [{"name": "r"}], "params": {"n": {"range": [1, 1]}},
         "parts": [{"name": "a", "primitive": "box", "role": "r", "args": {"w": 4, "d": 2, "layers": 2}, "at": {"x": 2, "z": 2, "y": 0}},
                   {"name": "top", "primitive": "box", "role": "r", "args": {"w": 2, "d": 2, "layers": 1}, "attach": {"to": "a", "rel": "on_top_of", "anchor": "center"}},
                   {"name": "front", "primitive": "box", "role": "r", "args": {"w": 2, "d": 1, "layers": 1}, "attach": {"to": "a", "rel": "in_front_of", "anchor": "start"}},
                   {"name": "back", "primitive": "box", "role": "r", "args": {"w": 2, "d": 1, "layers": 1}, "attach": {"to": "a", "rel": "behind", "anchor": "end"}},
                   {"name": "side", "primitive": "box", "role": "r", "args": {"w": 1, "d": 2, "layers": 1}, "attach": {"to": "a", "rel": "beside", "side": "right"}}]}
    assert validate_spec(s) == []
    regions = {r.name: r for r in generate_regions(s, {"n": 1}, None)}
    cells = lambda n: sorted(regions[n].layers[0])
    assert regions["top"].y == 6 and cells("top")[0] == (3, 2)                              # centred on the 4 wide block, resting on its top (2 bricks = 6 plates)
    assert cells("front")[0] == (2, 4) and regions["front"].y == 0                          # in front (higher z), at the same height, aligned to the start
    assert cells("back")[-1] == (5, 1)                                                      # behind (lower z), aligned to the end
    assert cells("side")[0] == (6, 2)                                                       # to the right (higher x)


def test_a_mirrored_part_is_added_with_the_same_shape():
    r = get("robot")
    regions = r.generate({"width": 4, "legs": 1, "torso": 2, "antenna": 0}, None)
    legs = [x for x in regions if x.name.startswith("leg")]
    assert len(legs) == 2 and {c[0] for c in legs[1].layers[0]} == {5 - x for x in {c[0] for c in legs[0].layers[0]}}


# ---------------------------------------------------------------- slopes, tiles, rounds: roles in the filler
def test_slopes_are_used_for_a_gable_roof_when_the_inventory_has_enough_and_plates_when_not():
    base = [("3001", "Red", 14), ("3020", "Red", 12), ("3004", "Red", 6), ("3003", "Red", 8), ("3023", "Red", 6)]
    rich_inv = random_inventory(random.Random(9), 150)
    from backend.core.buildmodel import Inventory
    from backend.recipes.inventories import make_item
    sloped = Inventory(rich_inv.parts() + [make_item("3037", "slope", "Blue", 6), make_item("3039", "slope", "Blue", 6), make_item("3040b", "slope", "Blue", 4)])
    plain = rich_inv
    house = get("house")
    params = {"width": 4, "depth": 4, "wall_layers": 2, "roof": 1}
    regions_plain = house.generate(params, facts_of(plain))
    regions_sloped = house.generate(params, facts_of(sloped))
    assert [r.prefer for r in regions_plain if r.name.startswith("roof")] == [None]
    assert {r.prefer for r in regions_sloped if r.name.startswith("roof")} >= {"slope"}
    cols = {"wall": facts_of(sloped).colours[0], "roof": "Blue", "base": facts_of(sloped).colours[0]}
    reasons = []
    cand = make_candidate(house, params, sloped, cols, facts_of(sloped), reasons)
    assert cand is not None, reasons
    kinds = {sloped.get(p.part_id, p.colour).dims.shape for p in cand.build.parts}
    assert "slope" in kinds
    assert not bm.validate(cand.build, sloped, 200).errors
    slopes = [p for p in cand.build.parts if sloped.get(p.part_id, p.colour).dims.shape == "slope"]
    dirs = {sloped.get(p.part_id, p.colour).dims.descent_at(p.rotation) for p in slopes}
    assert dirs == {"-z", "+z"}                                                              # one slope falls to each side of the ridge


def test_round_parts_fill_a_cylinder_first_and_tiles_cap_a_flat_top():
    inv = inventory_from([("3941", "Green", 6), ("3003", "Green", 8), ("3001", "Green", 8), ("3020", "Green", 8), ("3023", "Green", 6), ("3068b", "White", 6)])
    from backend.recipes.filler import fill
    from backend.recipes.base import Region, rect
    cyl = PRIMITIVES["cylinder"].build(Ctx("c", "leaves", "brick", 0, 0, 0, True, 0.7, None), {"diameter": 4, "layers": 1})
    res = fill(cyl, inv, {"leaves": "Green"})
    assert any(inv.get(p.part_id, p.colour).dims.shape == "round" for p in res.placements)      # 2x2 round bricks where they fit
    cap = Region("cap", "top", 3, [rect(0, 0, 2, 2)], "plate", True, 0.9, "solid", "tile")
    base = Region("base", "top", 0, [rect(0, 0, 2, 2)] * 1, "brick", True, 0.9)
    res2 = fill([base, cap], inv, {"top": "Green"})
    assert any(inv.get(p.part_id, p.colour).dims.kind == "tile" for p in res2.placements)


def test_vehicles_are_never_offered_because_wheels_cannot_be_placed():
    inv = inv9()
    facts = facts_of(inv)
    assert "vehicle" not in feasible_categories(facts, all_recipes())
    assert get("car").needs == ("wheels",)
    assert not any(c.recipe_id == "car" for c in __import__("backend.recipes.planner", fromlist=["x"]).plan_candidates(inv, 1))
    s = spec()
    s["parts"].append({"name": "wheels", "primitive": "posts", "role": "leg", "args": {"x0": 0, "z0": 0, "layers": 1}})
    assert any("wheels cannot be made" in e for e in validate_spec(s))


# ---------------------------------------------------------------- structure, features, IoU
def P(part, colour, x, z, y, step=1):
    return Placement(part, colour, x, z, y, 0, step)


def test_structural_check_finds_floating_and_separate_parts():
    inv = inventory_from([("3001", "Red", 4), ("3003", "Red", 4)])
    ok = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 0, 0, 3, 2)])
    assert structural_problems(ok, inv) == []
    floating = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 8, 8, 6, 2)])
    assert any("rests on nothing" in m or "not connected to the ground" in m for m in structural_problems(floating, inv))
    apart = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 9, 9, 0, 2)])
    assert any("separate pieces" in m for m in structural_problems(apart, inv))
    overlap = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 1, 0, 0, 2)])
    assert structural_problems(overlap, inv)


def test_voxel_iou_ignores_position_and_tells_shapes_apart():
    inv = random_inventory(random.Random(9), 130)
    facts = facts_of(inv)
    house = get("house")
    a = make_candidate(house, {"width": 6, "depth": 4, "wall_layers": 2, "roof": 0}, inv, assign_colours(house, facts), facts)
    b = make_candidate(house, {"width": 6, "depth": 4, "wall_layers": 2, "roof": 0}, inv, assign_colours(house, facts, 1), facts)
    tall = make_candidate(get("tower"), {"depth": 2, "layers": 8, "band": 1}, inv, assign_colours(get("tower"), facts), facts)
    assert a is not None and b is not None and tall is not None
    va, vb, vt = (voxels_of_build(c.build, inv) for c in (a, b, tall))
    assert iou(va, vb) > 0.9                                                                # the same shape in other colours: nearly identical
    assert iou(va, va) == 1.0
    shifted = frozenset((x + 5, z - 3, y + 2) for x, z, y in va)
    assert iou(va, shifted) == 1.0 and normalise(shifted) == normalise(va)                 # position does not matter
    assert iou(va, vt) < IOU_MAX < iou(va, va)
    assert iou(frozenset(), frozenset()) == 1.0 and iou(va, frozenset()) == 0.0


def test_feature_check_finds_missing_features_and_unsupported_asks():
    inv = inv9()
    facts = facts_of(inv)
    house = get("house")
    cand = make_candidate(house, {"width": 6, "depth": 4, "wall_layers": 3, "roof": 1}, inv, assign_colours(house, facts), facts)
    assert cand is not None and feature_problems(house, cand.regions, ["a sloped roof", "a door"]) == []
    assert any("cannot build" in p for p in feature_problems(house, cand.regions, ["4 wheels"]))
    stripped = [r for r in cand.regions if r[0] != "roof"]
    assert any("roof" in p for p in feature_problems(house, stripped, ["a sloped roof"]))              # asked for, but gone: a problem
    assert feature_problems(house, stripped, []) == []                                                # a part that exists only for some settings is not mandatory
    assert feature_tags("two big wheels and a sloped hood") == {"wheels"} or "wheels" in feature_tags("two big wheels and a sloped hood")


def test_category_matching_for_the_recognisability_check():
    recipes = all_recipes()
    assert match_category("a small sailing boat", "boat", recipes)[0]
    assert match_category("a toy airplane", "aircraft", recipes)[0]
    assert not match_category("a tall tower", "boat", recipes)[0] and match_category("a tall tower", "boat", recipes)[1] == "tower"
    assert not match_category("a pile of blocks", "robot", recipes)[0]


# ---------------------------------------------------------------- generation and repair
def lib(tmp_path):
    return RecipeLibrary(tmp_path / "lib")


def gen(llm, tmp_path, concept=CONCEPT, **kw):
    inv = inv9()
    return run(generate_recipe(llm, concept, inv, facts_of(inv), lib(tmp_path), hex_of, log_dir=tmp_path / "log", **kw)), inv


def test_a_valid_recipe_passes_every_stage_and_is_stored(tmp_path):
    s = spec()
    s["title"], s["names"] = "Reading Nook", ["Reading Nook"]
    llm = LLM([s], ["a small chair"])
    res, inv = gen(llm, tmp_path)
    assert res.ok and len(res.attempts) == 1 and res.attempts[0].stage is None and res.attempts[0].answer == "a small chair"
    assert res.attempts[0].passed == ["schema", "fill", "inventory", "structure", "recognizability"]
    assert res.candidate is not None and not structural_problems(res.candidate.build, inv)
    stored = RecipeLibrary(tmp_path / "lib")
    assert res.stored is False and "too similar" in res.store_note or res.stored       # the chair recipe is the built-in chair: a duplicate shape is not stored
    prompt = llm.prompts[0]
    assert "Primitives (use only these)" in prompt and "Examples of valid recipes" in prompt and "Inventory budget" in prompt and "Reading Nook" in prompt
    lines = (tmp_path / "log" / "recognizability.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["answer"] == "a small chair" and json.loads(lines[0])["matched"] is True


def test_a_new_shape_is_stored_in_the_library_and_a_duplicate_is_rejected(tmp_path):
    s = spec("table")
    s["id"] = "side-table"
    s["category"] = "furniture"
    s["parts"][1]["args"]["d"] = "depth + 3"                 # a clearly different shape from the built-in table and chair
    s["params"]["depth"]["range"] = [2, 3]
    library = lib(tmp_path)
    ok, why = library.add(s, {"concept": "x"})
    inv = inv9()
    dup = spec("chair")
    dup["id"] = "my-chair"
    ok2, why2 = library.add(dup)
    assert not ok2 and "too similar" in why2 and "chair" in why2                           # same category, voxel IoU 1.0 with the built-in chair
    assert library.add(spec("chair") | {"id": "house"}) == (False, "'house' is the id of a built-in recipe")
    assert library.add({"id": "../x"}) == (False, "bad recipe id '../x'")
    if ok:
        assert [r.id for r in library.recipes()] == ["side-table"] and library.for_category("furniture") and library.for_category("boat") == []
        assert library.entries()[0]["meta"]["category"] == "furniture"


def test_a_failing_stage_sends_the_specific_errors_back_and_the_repair_passes(tmp_path):
    bad = spec()
    bad["parts"][1]["primitive"] = "teapot"
    good = spec()
    llm = LLM([bad, good], ["an armchair"])
    res, _ = gen(llm, tmp_path)
    assert res.ok and len(res.attempts) == 2 and res.attempts[0].stage == "schema" and res.attempts[1].stage is None
    repair = llm.prompts[1]
    assert "Failed at the 'schema' stage" in repair and "unknown primitive" in repair and "teapot" in repair and "complete corrected recipe" in repair


def test_a_fill_failure_names_the_part_that_could_not_be_filled(tmp_path):
    big = spec()
    big["parts"][1]["args"]["w"] = 24
    big["parts"][1]["args"]["d"] = 24
    big["parts"][1]["args"]["layers"] = 1
    big["parts"][0]["args"]["x1"] = 23
    big["params"]["width"]["range"] = [2, 6]
    llm = LLM([big, spec()], ["a chair"])
    res, _ = gen(llm, tmp_path, recognise=True)
    first = res.attempts[0]
    assert first.stage in ("fill", "schema") and first.errors
    assert res.ok is True or res.attempts[-1].stage is not None


def test_a_recipe_that_never_passes_is_dropped_after_three_repairs(tmp_path):
    bad = spec()
    bad["parts"][1]["primitive"] = "teapot"
    llm = LLM([bad, bad, bad, bad])
    res, _ = gen(llm, tmp_path)
    assert not res.ok and len(res.attempts) == 1 + MAX_REPAIRS and res.failed_stage == "schema" and res.candidate is None and not res.stored
    t = tally([res])
    assert t["schema"]["attempts_reached"] == 4 and t["schema"]["attempts_passed"] == 0 and t["final"]["ok"] == 0 and t["final"]["recipe_calls"] == 4


def test_recognizability_asks_without_the_name_and_a_wrong_answer_is_a_failure_that_is_repaired(tmp_path):
    llm = LLM([spec(), spec()], ["a tall tower", "a chair"])
    res, _ = gen(llm, tmp_path)
    assert res.ok and [a.stage for a in res.attempts] == ["recognizability", None]
    assert res.attempts[0].answer == "a tall tower"
    assert "looks like 'a tall tower'" in llm.prompts[2] or any("looks like 'a tall tower'" in p for p in llm.prompts)
    rows = [json.loads(x) for x in (tmp_path / "log" / "recognizability.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["answer"], r["matched"], r["looks_like"]) for r in rows][0][1:] == (False, "tower")


def test_gemini_failure_drops_the_concept_without_raising(tmp_path):
    res, _ = gen(LLM([RuntimeError("quota")]), tmp_path)
    assert not res.ok and res.attempts[0].stage == "gemini" and res.attempts[0].service_error and "Gemini failed" in res.attempts[0].errors[0]
    t = tally([res])
    assert t["schema"]["attempts_reached"] == 0 and t["final"]["gemini_service_errors"] == 1 and t["final"]["concepts_without_a_working_gemini"] == 1


def test_parse_answer_accepts_the_shapes_gemini_uses():
    s = {"id": "x"}
    assert parse_answer(json.dumps({"recipe": json.dumps(s)}))[0] == s and parse_answer(json.dumps({"recipe": s}))[0] == s and parse_answer(json.dumps(s))[0] == s
    assert parse_answer("nope")[1] and parse_answer(json.dumps({"recipe": "{broken"}))[1] and parse_answer("[1]")[1]


@pytest.mark.parametrize("change", [
    {"silhouettes": []}, {"silhouettes": "tall"}, {"requires": 5}, {"derive": []}, {"features": "roof"}, {"variants": []}, {"tags": {"a": 1}},
    {"params": []}, {"params": {"w": 5}}, {"params": {"w": {"range": "big"}}}, {"roles": "wall"}, {"roles": [1, 2]}, {"parts": {}}, {"parts": [5]},
    {"parts": [{"name": 3}]}, {"parts": [{"name": "a", "primitive": "box", "role": "seat", "args": []}]},
    {"parts": [{"name": "a", "primitive": "box", "role": "seat", "args": {"w": [1], "d": None, "layers": {}}}]},
    {"parts": [{"name": "a", "primitive": "box", "role": "seat", "args": {"w": 1, "d": 1}, "attach": []}]},
    {"parts": [{"name": "a", "primitive": "box", "role": "seat", "args": {"w": 1, "d": 1}, "repeat": 3}]},
])
def test_whatever_shape_the_json_has_validation_reports_instead_of_raising(change):
    s = spec()
    s.update(change)
    errors = validate_spec(s)
    assert errors and all(isinstance(e, str) for e in errors)


def test_a_malformed_recipe_from_gemini_is_a_failed_attempt_not_a_crash(tmp_path):
    weird = spec()
    weird["silhouettes"] = ["compact", "tall"]
    llm = LLM([weird, spec()], ["a chair"])
    res, _ = gen(llm, tmp_path)
    assert res.attempts[0].stage == "schema" and "silhouettes" in res.attempts[0].errors[0] and res.ok
