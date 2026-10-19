"""Run with the experiment venv:  venv/Scripts/python.exe -m pytest tests -q   (from experiments/brickgpt)"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brickgpt.data import Brick, BrickStructure

from inventory import PART_DIMS, assign_colours, load_fixtures, make_steps, missing_bricks, part_of, to_ldr, read_ldconfig
from inventory_sampler import InventoryBrickGPT


def B(h, w, x, y, z):
    return Brick(h=h, w=w, x=x, y=y, z=z)


def wall():
    return [B(2, 4, 0, 0, 0), B(2, 4, 2, 0, 0), B(2, 4, 0, 0, 1), B(1, 2, 0, 0, 2), B(1, 1, 5, 5, 0)]


def test_colours_never_exceed_stock_and_biggest_colour_gets_biggest_region():
    inv = {("3001", "Red"): 2, ("3001", "Blue"): 1, ("3004", "Red"): 1, ("3005", "Blue"): 1}
    bricks = wall()
    cols = assign_colours(bricks, inv)
    used = Counter((part_of(b), c) for b, c in zip(bricks, cols))
    assert all(used[k] <= inv.get(k, 0) for k in used)
    assert Counter(cols)["Red"] >= Counter(cols)["Blue"]          # Red (3 in stock) is the most plentiful colour


def test_colours_fail_loudly_when_stock_too_small():
    try:
        assign_colours(wall(), {("3001", "Red"): 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_steps_bottom_up_in_groups_of_at_most_three_within_a_layer():
    bricks = [B(1, 1, i, 0, z) for i, z in enumerate([0, 1, 0, 0, 0, 2, 1])]
    steps = make_steps(bricks)
    assert [bricks[i].z for s in steps for i in s] == sorted(b.z for b in bricks)
    assert all(1 <= len(s) <= 3 and len({bricks[i].z for i in s}) == 1 for s in steps)
    assert sorted(i for s in steps for i in s) == list(range(len(bricks)))


def test_ldr_has_one_step_line_per_step():
    bricks = wall()
    steps = make_steps(bricks)
    text = to_ldr(bricks, ["Red"] * len(bricks), steps, read_ldconfig())
    assert text.count("0 STEP") == len(steps) and text.count("\n1 ") + text.startswith("1 ") == len(bricks)


def test_missing_bricks_by_type():
    assert missing_bricks(wall(), {("3001", "Red"): 1, ("3005", "Red"): 1}) == {"3001": 2, "3004": 1}


def _sampler(stock):
    s = InventoryBrickGPT.__new__(InventoryBrickGPT)           # no model: only the bookkeeping is under test
    s.set_inventory({(p, "x"): n for p, n in stock.items()})
    return s


def test_stock_rejects_used_up_type_and_rollback_restores_it():
    s = _sampler({"3001": 1, "3005": 2})
    bricks = BrickStructure([])
    assert s._try_adding_brick("2x4 (0,0,0)", bricks, set()) == "success"
    bricks.add_brick(Brick.from_txt("2x4 (0,0,0)"))
    assert s._try_adding_brick("4x2 (0,2,1)", bricks, set()) == "no_inventory"   # either orientation of the same type
    assert s._try_adding_brick("1x1 (0,0,1)", bricks, set()) == "success"
    bricks.undo_add_brick()                                                    # the physics rollback removes the 2x4
    assert s._try_adding_brick("2x4 (0,0,0)", bricks, set()) == "success"
    assert s.remaining(bricks)["3001"] == 1


def test_fixtures_are_nested_and_sized():
    fx = load_fixtures()
    assert [sum(fx[k].values()) for k in ("small", "medium", "large")] == [12, 40, 120]
    assert all(fx["large"].get(k, 0) >= n for k, n in fx["medium"].items())


def test_real_pile_fixture_is_built_from_the_txt_ground_truth():
    import json
    from build_real_pile import TXT, parse
    rows, other, total = parse(TXT.read_text(encoding="utf-8"))
    assert total == 16 and sum(r["qty"] for r in rows) == 11 and len(other) == 5      # 2x3, 1x3, two modified, technic are not BrickGPT types
    saved = json.loads((Path(__file__).resolve().parents[1] / "real_pile.json").read_text())
    assert saved["rows"] == rows                                                      # the file on disk is what the .txt gives
    fx = load_fixtures()["real_pile"]
    assert fx == {("2456", "Red"): 2, ("3001", "Red"): 2, ("2456", "Dark Bluish Gray"): 1, ("3001", "Blue"): 1, ("3010", "Medium Azure"): 1,
                  ("3003", "Bright Light Yellow"): 1, ("3003", "Reddish Brown"): 1, ("3004", "Yellowish Green"): 1, ("3009", "Yellow"): 1}
    ld = read_ldconfig()
    assert all(part in PART_DIMS and colour in ld for (part, colour) in fx)           # every colour name is an LDraw/Rebrickable one


def test_single_piece_needs_stud_links_not_just_ground_contact():
    from inventory import single_piece
    tower = [B(2, 4, 0, 0, 0), B(2, 4, 0, 0, 1), B(1, 2, 0, 0, 2)]
    assert single_piece(tower) == {"single_piece": True, "pieces": 1, "sizes": [3]}
    two_towers = tower + [B(2, 2, 10, 10, 0), B(2, 2, 10, 10, 1)]               # both stand on the ground, but are not joined
    assert single_piece(two_towers) == {"single_piece": False, "pieces": 2, "sizes": [3, 2]}
    side_by_side = [B(2, 4, 0, 0, 0), B(2, 4, 2, 0, 0)]                          # touching in one layer is not a stud link
    assert single_piece(side_by_side)["pieces"] == 2
    bridge = side_by_side + [B(6, 1, 0, 0, 1)]                                   # a brick across both joins them
    assert single_piece(bridge)["single_piece"]
    assert single_piece([B(1, 1, 0, 0, 0), B(1, 1, 1, 1, 1)])["pieces"] == 2     # diagonal neighbours share no stud
