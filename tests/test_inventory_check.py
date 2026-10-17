"""The "Consider color" rule (can_build), colour-blind planning (pool_inventory) and the recolouring pass."""

from collections import Counter
from dataclasses import replace

import pytest

from backend.core.buildmodel import Inventory, Placement
from backend.core.inventory_check import POOLED, can_build, pool_inventory, recolour, shortfall
from backend.recipes.inventories import make_item, random_inventory
from backend.recipes.planner import plan_candidates
from backend.recipes.service import finalize
from backend.recipes.suggest import Shown

B24, B23 = "3001", "3002"                 # Brick 2 x 4, Brick 2 x 3


def part(part_id, colour, x=0, y=0):
    return Placement(part_id, colour, x, 0, y, 0, 1)


@pytest.fixture
def inv():
    """3 x red 2x4 + 1 x blue 2x3"""
    return Inventory([make_item(B24, "Brick 2 x 4", "Red", 3), make_item(B23, "Brick 2 x 3", "Blue", 1)])


# ---------------------------------------------------------------- can_build
def test_colour_on_exact_inventory_passes(inv):
    assert can_build([part(B24, "Red"), part(B24, "Red", 4), part(B24, "Red", 8), part(B23, "Blue", 12)], inv, True)


def test_colour_on_four_red_2x4_fails(inv):
    build = [part(B24, "Red", 4 * i) for i in range(4)]
    assert not can_build(build, inv, True)
    assert shortfall(build, inv, True) == {(B24, "red"): 1}


def test_colour_on_a_red_2x3_fails_because_the_2x3_is_blue(inv):
    assert not can_build([part(B23, "Red")], inv, True)


def test_colour_off_two_blue_2x4_and_a_red_2x3_pass(inv):
    assert can_build([part(B24, "Blue"), part(B24, "Blue", 4), part(B23, "Red", 8)], inv, False)


def test_sizes_never_substitute_even_with_colour_off(inv):
    assert not can_build([part(B24, "Red", 4 * i) for i in range(4)], inv, False)           # a 4th 2x4 does not exist


def test_colour_off_one_2x4_and_two_2x3_fail(inv):
    assert not can_build([part(B24, "Red"), part(B23, "Blue", 4), part(B23, "Blue", 8)], inv, False)    # only one 2x3 exists


def test_an_empty_build_and_unknown_parts(inv):
    assert can_build([], inv, True)
    assert not can_build([part("9999", "Red")], inv, False)


def test_inventory_can_be_a_plain_list_and_names_are_case_insensitive():
    class Row:
        def __init__(self, part_id, colour, qty):
            self.part_id, self.colour, self.qty = part_id, colour, qty

    stock = [Row("3001", "Red", 2), Row("3001", "Blue", 1)]
    assert can_build([part("3001", "RED"), part("3001", "red", 4), part("3001", "blue", 8)], stock, True)
    assert not can_build([part("3001", "blue"), part("3001", "blue", 4)], stock, True)
    assert can_build([part("3001", "blue"), part("3001", "blue", 4)], stock, False)


# ---------------------------------------------------------------- pooling and recolouring
def test_pool_inventory_sums_colours_per_part(inv):
    both = Inventory(inv.parts() + [make_item(B24, "Brick 2 x 4", "Blue", 2)])
    pooled = pool_inventory(both)
    assert {(i.part_id, i.qty, i.colour) for i in pooled.parts()} == {(B24, 5, POOLED), (B23, 1, POOLED)}


def test_a_colour_blind_build_is_coloured_from_the_real_stock(inv):
    pooled_build = [part(B24, POOLED), part(B24, POOLED, 4), part(B23, POOLED, 8)]
    out = recolour(pooled_build, {}, inv)
    assert can_build(out, inv, True)                                          # real (part, colour) units, nothing invented
    assert Counter((p.part_id, p.colour) for p in out) == Counter({(B24, "Red"): 2, (B23, "Blue"): 1})
    assert [(p.x, p.step) for p in out] == [(p.x, p.step) for p in pooled_build]       # same places, same steps


def test_mirrored_parts_share_a_colour_and_the_biggest_section_gets_the_plentiful_colour():
    wheel, deck = "3005", "3020"                                               # Brick 1 x 1, Plate 2 x 4
    inv = Inventory([make_item(wheel, "Brick 1 x 1", "Red", 4), make_item(wheel, "Brick 1 x 1", "Blue", 4),
                     make_item(deck, "Plate 2 x 4", "Blue", 1)])
    wheels = [Placement(wheel, POOLED, x, z, 0, 0, 1) for x in (0, 3) for z in (0, 1)]
    body = Placement(deck, POOLED, 0, 0, 3, 0, 2)
    sections = {(p.x, p.z, p.y): "wheel" for p in wheels} | {(0, 0, 3): "body"}
    out = recolour(wheels + [body], sections, inv)
    assert can_build(out, inv, True)
    assert {p.colour for p in out if p.part_id == wheel} in ({"Red"}, {"Blue"})      # all four wheels match
    assert out[-1].colour == "Blue"                                                  # the largest section got the colour owned most
    assert {p.colour for p in out if p.part_id == wheel} == {"Red"}                  # and the wheels took the other one


def test_one_section_uses_one_colour_when_stock_allows_and_splits_when_not():
    inv = Inventory([make_item(B24, "Brick 2 x 4", "Red", 2), make_item(B24, "Brick 2 x 4", "Blue", 1)])
    build = [part(B24, POOLED, 4 * i) for i in range(2)]
    assert len({p.colour for p in recolour(build, {}, inv)}) == 1
    three = [part(B24, POOLED, 4 * i) for i in range(3)]
    out = recolour(three, {}, inv)
    assert can_build(out, inv, True) and Counter(p.colour for p in out) == Counter({"Red": 2, "Blue": 1})


# ---------------------------------------------------------------- the gate in the recipe service
def shown_for(inv, match_color):
    plan = inv if match_color else pool_inventory(inv)
    cand = plan_candidates(plan, seed=1)[0]
    return Shown(cand, cand.default_name(), "d")


@pytest.mark.parametrize("seed", [3, 4, 5])
def test_colour_off_builds_come_out_in_real_colours_and_pass_the_inventory_rule(seed):
    inv = random_inventory(seed, 110)
    for match_color in (True, False):
        out = finalize(shown_for(inv, match_color), inv, match_color)
        assert out is not None
        parts = out.candidate.build.parts
        assert can_build(parts, inv, match_color) and can_build(parts, inv, True)
        assert POOLED not in {p.colour for p in parts} and "Mixed" not in out.name


def test_a_build_that_needs_more_than_the_user_owns_is_dropped_not_fixed():
    inv = random_inventory(3, 110)
    shown = shown_for(inv, True)
    extra = replace(shown.candidate.build.parts[0])
    cand = replace(shown.candidate, build=replace(shown.candidate.build, parts=shown.candidate.build.parts + [extra] * 99))
    assert finalize(Shown(cand, "x", "y"), inv, True) is None
    assert finalize(Shown(cand, "x", "y"), inv, False) is None


# ---------------------------------------------------------------- the endpoint
from tests.test_builds import INV as API_INV, client, lib  # noqa: E402,F401  (fixtures: a fake LDraw library, the app with a small data set)


def test_api_consider_colour_off_marks_cards_as_mixed_and_shows_only_owned_pieces(client):
    parts = [{"part_num": i.part_id, "name": i.name, "color": i.colour, "qty": i.qty * 3} for i in API_INV.parts() if i.dims.studs_top]
    on = client.post("/api/builds/recipes", json={"parts": parts, "use_ai": False}).json()
    off = client.post("/api/builds/recipes", json={"parts": parts, "use_ai": False, "match_color": False}).json()
    assert on["match_colour"] is True and not any(b["mixed_colours"] for b in on["builds"])
    assert off["builds"] and off["match_colour"] is False and all(b["mixed_colours"] for b in off["builds"])
    owned = {(p["part_num"].lower(), p["color"].lower()): p["qty"] for p in parts}
    for b in off["builds"]:
        used = Counter()
        for step in b["steps"]:
            for p in step["parts"]:
                used[(p["ld_part"].lower(), p["colour_name"].lower())] += p["qty"]
        assert b["part_count"] == sum(used.values()) and "mixed" not in b["title"].lower()
        by_part = Counter()
        for (ld, _), n in used.items():
            by_part[ld] += n
        assert all(by_part[k] <= sum(q for (pn, _), q in owned.items() if pn == k) for k in by_part)
