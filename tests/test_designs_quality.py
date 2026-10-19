"""Step 4b: teaching examples, the exact size budget, diversity at the source, the solver tolerance (coverage + outlines), the child question, small piles."""

import json

from backend.designs import concepts as cn
from backend.designs import pipeline, teaching
from backend.designs import sketch as sk
from backend.designs.affordance import role_class, summarize
from backend.designs.examples import ExampleBank
from backend.designs.solver import SolveResult
from tests.test_builds import INV, client, lib, parts_for, service  # noqa: F401
from tests.test_designs import (FakeLLM, STEP, TOWER, WALL, collect, concept, kinds, three_calls_allowed, unscaled_sketches)  # noqa: F401  (autouse: tiny sketches stay as written)


# ---------------------------------------------------------------- teaching examples
def test_the_fixed_teaching_examples_are_valid_simple_and_in_the_prompt():
    names = [e["name"] for e in teaching.EXAMPLES]
    assert names[0] == "Delivery Truck" and len(names) >= 4
    truck = teaching.EXAMPLES[0]
    assert truck["cells"] == 44 and truck["type"] == "vehicle" and "chassis" in truck["description"] and "cab" in truck["description"] and "load" in truck["description"]
    for e in teaching.EXAMPLES:
        s = sk.parse(e["sketch"], 12, 12, 8, palette=e["palette"])
        assert s.filled == e["cells"] and 12 <= s.filled <= 60
        assert sum(1 for x, z, k in s.cells if k > 0 and (x, z, k - 1) not in s.cells) <= 3, e["name"]          # nothing floats (a door gap or two aside)
    text = cn.build_prompt(summarize(INV), True, [])
    assert "Delivery Truck" in text and "Sail Boat" in text and "Pond Duck" in text and "(44 cells)" in text


def test_the_prompt_asks_for_simple_playful_builds_and_allows_vehicles_without_wheels():
    s = cn.SYSTEM_PROMPT
    for needle in ("Build like a child", "few bricks", "big bricks used fully", "1 or 2 telling", "Vehicles need no wheel parts", "chassis + cab", "hull + mast",
                   "body + wings", "different silhouette type", "different bounding box"):
        assert needle in s, needle


def test_examples_are_picked_by_size_and_the_mix_of_parts_and_logged():
    aff = summarize(INV)
    big = {"set": "1-1", "title": "Bricks", "parts": 8, "filled": round(0.75 * aff.cells), "sketch": {"layers": [["##"]]}, "mix": dict(aff.mix)}
    other = {"set": "2-1", "title": "Slopes", "parts": 8, "filled": round(0.75 * aff.cells), "sketch": {"layers": [["##"]]}, "mix": {"slope/small": 8}}
    small = {"set": "3-1", "title": "Tiny", "parts": 6, "filled": 6, "sketch": {"layers": [["##"]]}, "mix": dict(aff.mix)}
    assert [e["title"] for e in ExampleBank([small, other, big]).pick(aff.cells, aff.mix, 3)] == ["Bricks", "Tiny", "Slopes"]
    assert role_class(INV.parts()[0].dims).count("/") == 1


def test_the_examples_used_are_logged_in_the_run_stats(lib, tmp_path):
    aff = summarize(INV)
    bank = ExampleBank([{"set": "9-1", "title": "Rover", "parts": 9, "filled": 40, "sketch": {"layers": [["##"]]}, "mix": dict(aff.mix)}])
    import asyncio
    from backend.designs.pipeline import stream_designs

    async def go():
        out = []
        async for chunk in stream_designs(service(lib, tmp_path), parts_for(), FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"]), True, bank=bank):
            out.append(chunk)
        return out
    asyncio.run(go())
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"])

    async def go2():
        async for chunk in stream_designs(service(lib, tmp_path), parts_for(), llm, True, bank=bank):
            if chunk.startswith("event: done"):
                return json.loads(chunk.split("data: ")[1])
    done = asyncio.run(go2())
    assert done["stats"]["examples"] == ["9-1 Rover"] and "Rover" in llm.prompts[0]


# ---------------------------------------------------------------- exact size budget
def test_the_prompt_gives_the_exact_volume_and_per_layer_limits():
    aff = summarize(INV)
    whole, low, high = aff.budget
    text = aff.prompt(True)
    assert whole == int(aff.cells) and high == whole and low == -(-3 * whole // 4)
    assert f"{whole} cells" in text and f"must fill {low} to {high} cells" in text and "Bottom layer: at most" in text and "no row longer than" in text
    area, row = aff.ground
    assert area == aff.area and row == sum(aff.lengths[:2]) and row >= aff.longest


def test_an_oversized_sketch_is_rejected_and_an_undersized_one_is_flagged(monkeypatch):
    monkeypatch.setattr(cn, "SLACK_LOW", 0.93)                                                  # the real tolerance (the other tests switch it off)
    monkeypatch.setattr(cn, "SLACK_HIGH", 1.05)
    limits = {"width": 8, "depth": 8, "layers": 6}
    budget = (20, 15, 20)
    ok, _ = cn.clean(json.loads(WALL), limits, budget=budget)                                   # 20 cells
    assert ok is not None
    small, why = cn.clean(json.loads(concept("Tiny", [["##", "##"], ["##", ".."]])), limits, budget=budget)           # too small: kept, flagged, enlarged later
    assert small is not None and small.undersized and not ok.undersized and why is None
    big, why = cn.clean(json.loads(concept("Huge", [["####", "####"]] * 4)), limits, budget=budget)
    assert big is None and "only has 20" in why
    near, _ = cn.clean(json.loads(concept("Near", [["####", "####"], ["####", "####"], ["..#.", "..#."]])), limits, budget=budget)      # 18: inside
    assert near is not None


# ---------------------------------------------------------------- diversity
def test_concepts_carry_their_silhouette_type_and_a_turn_invariant_bounding_box():
    c, _ = cn.clean({"name": "Hay Truck", "type": "Vehicle!", "sketch": {"layers": [["####", "####"]]}}, {"width": 8, "depth": 8, "layers": 6})
    t, _ = cn.clean({"name": "Turned Truck", "sketch": {"layers": [["##", "##", "##", "##"]]}}, {"width": 8, "depth": 8, "layers": 6})
    assert c.kind == "vehicle" and c.bbox == (4, 2, 1) == t.bbox and t.kind == ""


def test_a_concept_with_the_bounding_box_of_an_earlier_one_is_dropped_and_the_repair_asks_for_a_difference(lib, tmp_path):
    same_box = concept("Other Gate", [["####", "####"], ["#...", "...."], ["#...", "...."]])
    llm = FakeLLM([f"{WALL}\n{same_box}\n", f"{TOWER}\n{STEP}\n"])
    ev = collect(service(lib, tmp_path), llm)
    done = ev[-1][1]
    assert any("bounding box" in f["why"] and "Red Gatepost" in f["why"] for f in done["failures"]) and done["stats"]["dup_dropped"] >= 1
    assert llm.stream_calls == 2 and "Already kept" in llm.prompts[1] and "Red Gatepost" in llm.prompts[1] and "4 x 2 x 3" in llm.prompts[1]


# ---------------------------------------------------------------- solver tolerance
def test_outline_iou_compares_top_front_and_side_views():
    s = sk.parse({"layers": [["####", "####"], ["##..", "##.."]]})
    assert sk.outline_iou(s, s.plate_cells()) == 1.0
    filled = {c for c in s.plate_cells() if c[0] < 3}                                           # the right-hand column is missing
    assert 0.5 < sk.outline_iou(s, filled) < 0.9
    line = {c for c in s.plate_cells() if c[1] == 0 and c[2] == 0}                              # a single row of plates: the outlines are gone
    assert sk.outline_iou(s, line) < 0.3


def _fake_solve(monkeypatch, coverage, keep):
    real = pipeline.solve_concept

    def fake(concept, parts, sketch):
        res = real(concept, parts, sketch)
        res.covered = int(coverage * res.total)
        res.cells = frozenset(c for c in res.cells if keep(c))
        return res
    monkeypatch.setattr(pipeline, "solve_concept", fake)


def test_a_fill_of_85_to_95_percent_is_rescued_when_the_outlines_hold(lib, tmp_path, monkeypatch):
    assert pipeline.COVERAGE_TOLERATED == 0.85 and pipeline.COVERAGE_CLEAN == 0.95 and 0 < pipeline.OUTLINE_IOU_MIN <= 1
    _fake_solve(monkeypatch, 0.90, lambda c: True)                                              # every outline intact
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"]))
    assert ev[-1][1]["stats"]["rescued"] == 3 and all(d["build"]["design"]["rescued"] for e, d in ev if e == "design")


def test_a_fill_of_85_to_95_percent_is_rejected_when_the_outline_is_lost(lib, tmp_path, monkeypatch):
    _fake_solve(monkeypatch, 0.90, lambda c: c[1] == 0 and c[2] == 0)
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n", f"{WALL}\n"]))
    done = ev[-1][1]
    assert done["stats"]["rescued"] == 0 and done["stats"]["outline_rejected"] >= 3 and not [1 for e, _ in ev if e == "design"]
    assert any("outline is lost" in f["why"] for f in done["failures"])


def test_a_fill_below_85_percent_is_rejected(lib, tmp_path, monkeypatch):
    _fake_solve(monkeypatch, 0.80, lambda c: True)
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n", f"{WALL}\n"]))
    assert not [1 for e, _ in ev if e == "design"] and ev[-1][1]["stats"]["rescued"] == 0


# ---------------------------------------------------------------- small piles
def test_a_small_pile_with_no_design_gets_a_friendly_message(lib, tmp_path):
    small = [p for p in parts_for() if p["qty"]][:1]
    small[0] = {**small[0], "qty": 2}
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n", f"{WALL}\n"]), parts=small)
    note = next(d for e, d in ev if e == "note")
    assert note["kind"] == "small_pile" and "only 2 pieces" in note["message"] and "Add a few more pieces" in note["message"]


def test_a_larger_pile_with_no_design_keeps_the_plain_message(lib, tmp_path):
    ev = collect(service(lib, tmp_path), FakeLLM([f"{concept('Huge', [['#' * 12] * 12] * 4)}\n"] * 2))
    assert next(d for e, d in ev if e == "note")["kind"] == "no_design"


# ---------------------------------------------------------------- the child question
def test_the_look_check_asks_whether_a_child_could_see_the_name_and_logs_both_answers(lib, tmp_path):
    seen = {}

    class Llm(FakeLLM):
        async def complete_with_images(self, system, user, schema, images):
            seen.update(system=system, user=user, props=schema["properties"]["answers"]["items"]["properties"])
            names = schema["properties"]["answers"]["items"]["properties"]["choice"]["enum"][:-1]
            # the picked name is wrong for every design, but a child could see two of them as what they are named
            return json.dumps({"answers": [{"number": i + 1, "choice": "none of these", "child_yes": n != "Little Lighthouse"} for i, n in enumerate(names)]})
    ev = collect(service(lib, tmp_path), Llm([f"{WALL}\n{TOWER}\n{STEP}\n"]))
    assert "could a child see this simple build" in seen["user"] and "child_yes" in seen["props"] and "choice" in seen["props"] and "children and beginners" in seen["system"]
    done = ev[-1][1]
    assert done["stats"]["look_ok_child"] == 2 and done["stats"]["look_ok_choice"] == 0 and done["stats"]["look_ok"] == 2
    verdicts = {d["title"]: (d["ok"], d["ok_choice"], d["ok_child"], d["choice"]) for e, d in ev if e == "verdict"}
    assert verdicts["Little Lighthouse"] == (False, False, False, "none of these") and verdicts["Red Gatepost"] == (True, False, True, "none of these")
    assert done["designs"] == 2


def test_the_old_question_can_still_decide(lib, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "LOOK_RULE", "choice")
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"], looks=lambda names: ["none of these" if n == "Little Lighthouse" else n for n in names])
    ev = collect(service(lib, tmp_path), llm)
    assert ev[-1][1]["designs"] == 2 and ev[-1][1]["stats"]["look_ok_choice"] == 2 and ev[-1][1]["stats"]["look_ok_child"] == 2
