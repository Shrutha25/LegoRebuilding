"""Sketches, the affordance summary and the CP-SAT solver."""

import pytest

from backend.core.partdims import KNOWN
from backend.designs import sketch as sk
from backend.designs.affordance import MIN_USE, summarize
from backend.designs.solver import solve
from backend.recipes.inventories import inventory_from


def part(pid, qty, colour="Any"):
    return (pid, colour, KNOWN[pid], qty)


def box(w, d, layers):
    return sk.parse({"layers": [["#" * w] * d] * layers})


def test_sketch_parse_normalises_checks_limits_and_round_trips():
    s = sk.parse({"layers": [["..##", "..##"], ["...#", "...."]]})
    assert (s.width, s.depth, s.layers, s.filled) == (2, 2, 2, 5) and (0, 0, 0) in s.cells and (1, 0, 1) in s.cells
    assert len(s.plate_cells()) == 15 and sk.parse(sk.render(s)).cells == s.cells
    assert sk.parse([["##", "##"]]).filled == 4
    for bad, why in (({"layers": []}, "no layers"), ({"layers": [["#"]]}, "at least"), ({"layers": [["##", "#?"]]}, "not allowed"),
                     ({"layers": [["#" * 13] * 3]}, "limits"), ({"layers": ["x"]}, "at least")):
        with pytest.raises(sk.SketchError, match=why):
            sk.parse(bad, min_cells=4)
    with pytest.raises(sk.SketchError):
        sk.parse({"layers": [["##", "##"]] * 9})


def test_affordance_counts_volume_roles_and_limits():
    inv = inventory_from([("3001", "Red", 2), ("3020", "Blue", 1), ("3039", "White", 1), ("3941", "Green", 1)])
    a = summarize(inv)
    assert a.pieces == 5 and a.longest == 4 and abs(a.cells - (2 * 8 + 8 / 3 + 4 * 1 + 4)) < 0.01
    text = a.prompt(match_color=True)
    assert "2 x 4x2 brick" in text and "in Red" in text and "slope" in text and "round brick" in text and "Volume budget" in text and "must fill" in text
    assert "only a wish" in a.prompt(match_color=False) and "used only as often as listed" in text and "colours: Red" in text
    assert 4 <= a.limits["width"] <= 12 and 3 <= a.limits["layers"] <= 8


def test_a_single_brick_fills_a_matching_sketch_and_stock_is_never_exceeded():
    res = solve(box(4, 2, 1), [part("3001", 1)])
    assert res.status == "OPTIMAL" and res.coverage == 1.0 and len(res.placements) == 1
    two = solve(box(4, 4, 1), [part("3001", 1)])                       # one 2x4 can cover only half of a 4x4 layer
    assert two.coverage == 0.5 and len(two.placements) == 1
    both = solve(box(4, 4, 1), [part("3001", 2)])                      # two bricks side by side on the ground are two pieces: only one is kept
    assert both.coverage == 0.5 and len(both.placements) == 1
    roofed = solve(box(4, 4, 2), [part("3001", 2), part("3020", 2)])   # ...unless something above ties them together
    assert roofed.dropped == 0 and len(roofed.placements) >= 3


def test_parts_rotate_and_many_sizes_combine():
    res = solve(box(2, 4, 1), [part("3001", 1)])                       # the long side must run along z
    assert res.coverage == 1.0 and res.placements[0][5] == 90
    mixed = solve(box(5, 3, 2), [part("3001", 1), part("3003", 1), part("3004", 1), part("3005", 1), part("3002", 1), part("3020", 1), part("3021", 1)])
    assert mixed.dropped == 0 and mixed.coverage >= 0.5 and len({p[0] for p in mixed.placements}) >= 4


def test_nothing_floats_a_part_above_the_ground_needs_studs_below():
    s = sk.parse({"layers": [["##", "##"], ["##", "##"]]})
    res = solve(s, [part("3003", 1), part("3003", 1)])                  # two 2x2 bricks: one on the other
    assert res.coverage == 1.0 and sorted(p[4] for p in res.placements) == [0, 3]
    floating = solve(sk.parse({"layers": [["#.", ".."], [".#", ".."]]}, min_cells=2), [part("3005", 2)])
    assert len(floating.placements) == 1 and floating.placements[0][4] == 0                      # the upper cell is not above anything: not built
    tile_below = solve(s, [part("3068b", 4), part("3003", 1)])                                   # a tile has no studs: a brick cannot sit on it
    assert all(not (p[0] == "3003" and p[4] > 0) for p in tile_below.placements)


def test_a_slope_never_carries_anything():
    s = sk.parse({"layers": [["##", "##"], ["##", "##"]]})
    res = solve(s, [part("3039", 1), part("3003", 1)])
    ys = {p[0]: p[4] for p in res.placements}
    assert res.coverage == 1.0 and ys["3003"] == 0 and ys["3039"] == 3                            # the slope goes on top


def test_the_result_is_one_connected_piece():
    s = sk.parse({"layers": [["#.#", "#.#"]]})                                                    # two separate columns on the ground
    res = solve(s, [part("3004", 2)])
    assert len(res.placements) == 1 and res.coverage == 0.5


def test_joints_are_staggered_when_there_is_a_choice():
    s = box(4, 1, 2)
    res = solve(s, [part("3004", 4), part("3010", 2)])                                           # 1x2 bricks and 1x4 bricks into a 4x1x2 wall
    assert res.coverage == 1.0
    layers = {y: sorted((p[2], p[2] + (4 if p[0] == "3010" else 2)) for p in res.placements if p[4] == y) for y in (0, 3)}
    joints = {y: {b for _, b in spans if b < 4} for y, spans in layers.items()}
    assert not (joints[0] & joints[3])                                                           # no joint directly over another joint


def test_a_realistic_mixed_pile_solves_fast():
    inv = inventory_from([("3001", "Red", 2), ("3003", "Blue", 1), ("3004", "Yellow", 2), ("3005", "Lime", 1), ("3010", "Red", 1), ("3002", "Tan", 1),
                          ("3020", "Green", 1), ("3022", "White", 1), ("3023", "Black", 2), ("3024", "Red", 1), ("3039", "White", 1), ("3622", "Yellow", 1)])
    parts = [(i.part_id, i.colour, i.dims, i.qty) for i in inv.parts()]
    s = sk.parse({"layers": [["######", "######", "######"], ["######", "######", "######"], ["..##..", "..##..", "..##.."]]})
    res = solve(s, parts, time_limit=2.0)
    assert res.status in ("OPTIMAL", "FEASIBLE") and res.coverage > 0.4 and res.seconds < 4.5
    used = {}
    for p in res.placements:
        used[p[0]] = used.get(p[0], 0) + 1
    stock = {}
    for pid, colour, _, q in parts:
        stock[(pid, colour)] = q
    used = {}
    for p in res.placements:
        used[(p[0], p[1])] = used.get((p[0], p[1]), 0) + 1
    assert all(used[k] <= stock[k] for k in used)


def test_a_small_sketch_is_enlarged_to_use_the_pile_but_never_beyond_it_or_the_limits():
    tree = sk.parse({"layers": [["...", ".#.", "..."], ["...", ".#.", "..."], ["###", "###", "###"], [".#.", "###", ".#."]]})        # 3x3x4, 16 cells
    assert tree.filled == 16
    big, s, t = sk.fit_scale(tree, 200, 12, 12, 8)                                                   # a pile of 200 cells: enlarged to fill most of it
    assert big.filled == tree.filled * s * s * t and 0.6 * 200 <= big.filled <= 200 and big.width == 3 * s and big.layers == 4 * t
    same, s, t = sk.fit_scale(tree, 20, 12, 12, 8)                                                   # a pile of 20 cells: left as drawn
    assert (s, t) == (1, 1) and same is tree
    over, s, t = sk.fit_scale(tree, 10, 12, 12, 8)                                                   # smaller than the sketch itself: left alone
    assert (s, t) == (1, 1)
    capped, s, t = sk.fit_scale(tree, 5000, 6, 6, 8)                                                 # the size limits win
    assert capped.width <= 6 and capped.depth <= 6 and capped.layers <= 8
    flat = sk.parse({"layers": [["######", "######"]]})
    assert 0.6 * 40 <= sk.fit_scale(flat, 40, 12, 12, 8)[0].filled <= 40
    scaled = sk.scale(sk.parse({"layers": [["##", "#."]]}, min_cells=3), 2, 2)
    assert scaled.filled == 3 * 8 and (scaled.width, scaled.depth, scaled.layers) == (4, 4, 2)


def test_colour_wishes_steer_the_colours_but_never_leave_a_hole():
    s = sk.parse({"layers": [["aabb", "aabb"], ["####", "####"]]}, palette={"a": "Red", "b": "Blue"})
    assert s.colours[(0, 0, 0)] == "Red" and s.colours[(3, 1, 0)] == "Blue" and s.filled == 16 and (0, 0, 1) not in s.colours
    parts = [("3003", "Red", KNOWN["3003"], 1), ("3003", "Blue", KNOWN["3003"], 1), ("3001", "Any", KNOWN["3001"], 1)]
    res = solve(s, parts)
    placed = {(p[0], p[1]): p[2] for p in res.placements}
    assert res.coverage == 1.0 and placed[("3003", "Red")] == 0 and placed[("3003", "Blue")] == 2          # red on the red side, blue on the blue side
    assert res.wish_cells == res.wish_met == 24
    only_red = solve(s, [("3003", "Red", KNOWN["3003"], 2), ("3001", "Any", KNOWN["3001"], 1)])
    assert only_red.coverage == 1.0 and only_red.wish_met < only_red.wish_cells                          # no blue in the pile: the cells are still filled
    assert sk.scale(s, 2, 1).colours[(7, 3, 0)] == "Blue" and len(sk.scale(s, 2, 1).colours) == 8 * 4
    assert sk.parse({"layers": [["a#"]]}, min_cells=2, palette={}).colours == {}                         # a letter without a palette colour: any colour


def test_the_search_starts_from_a_greedy_answer_so_a_short_time_limit_still_gives_a_good_design():
    inv = inventory_from([("3001", "Red", 2), ("3003", "Blue", 3), ("3004", "Yellow", 2), ("3020", "Green", 2), ("3023", "Black", 2), ("3039", "White", 1)])
    parts = [(i.part_id, i.colour, i.dims, i.qty) for i in inv.parts()]
    s = sk.scale(sk.parse({"layers": [["###", "###"], ["###", "###"], [".#.", ".#."]]}), 2, 1)
    quick = solve(s, parts, time_limit=1.5)
    assert quick.status in ("FEASIBLE", "OPTIMAL") and quick.coverage >= 0.35 and quick.seconds < 3.5
