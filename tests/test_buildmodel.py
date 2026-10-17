import numpy as np
import pytest

from backend import config
from backend.core import buildmodel as bm
from backend.core import ldrconv
from backend.core.buildmodel import Build
from backend.core.partdims import KNOWN, dims_for
from tests.build_fixtures import P, inventory, sample_ai_build, tower

INV = inventory()


def kinds(report):
    return [e.kind for e in report.errors]


# ---------------------------------------------------------------- dimensions
def test_dimensions_from_the_table_and_from_names():
    assert dims_for("3001").footprint(0) == (4, 2) and dims_for("3001").footprint(90) == (2, 4) and dims_for("3001").height == 3
    assert dims_for("3020").height == 1 and dims_for("3068b").studs_top is False
    assert dims_for("nope", "Brick 2 x 3").long == 3 and dims_for("nope", "Plate 1 x 8").short == 1
    assert dims_for("nope", "Tile 2 x 2 with Groove").studs_top is False
    for name in ("Brick, Round 1 x 1", "Brick, Modified 1 x 2 with Studs on 1 Side", "Slope 45 2 x 2", "Jumbo Plate 2 x 4", "", None):
        assert dims_for("nope", name) is None


# ---------------------------------------------------------------- validator
def test_a_good_build_validates():
    rep = bm.validate(tower(), INV)
    assert rep.ok and rep.components == 1 and not rep.warnings


def test_the_ai_fixture_is_a_valid_build():
    rep = bm.validate(sample_ai_build(), INV)
    assert rep.ok, [e.message for e in rep.errors]


def test_parts_and_colours_must_be_in_the_inventory_and_counts_respected():
    b = Build("t", "", [P("9999", "Red", 0, 0, 0), P("3001", "Green", 5, 0, 0)] + [P("3022", "Blue", 10 + 3 * i, 0, 0) for i in range(3)])
    rep = bm.validate(b, INV)
    assert set(kinds(rep)) >= {"unknown_part", "colour", "overuse"}


def test_overlap_is_found_including_partial_and_vertical_overlap():
    assert "overlap" in kinds(bm.validate(Build("t", "", [P("3001", "Red", 0, 0, 0), P("3001", "Red", 2, 0, 0)]), INV))   # side by side
    assert "overlap" in kinds(bm.validate(Build("t", "", [P("3001", "Red", 0, 0, 0), P("3020", "Red", 1, 0, 2)]), INV))   # plate inside the brick's top layer
    assert bm.validate(Build("t", "", [P("3001", "Red", 0, 0, 0), P("3020", "Red", 0, 0, 3)]), INV).ok                     # on top: fine
    rotated = Build("t", "", [P("3001", "Red", 0, 0, 0, 0), P("3001", "Red", 0, 1, 0, 90)])                                # rotated one reaches into the first
    assert "overlap" in kinds(bm.validate(rotated, INV))


def test_a_floating_part_is_found():
    b = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 0, 0, 5)])                  # a gap of two plates
    rep = bm.validate(b, INV)
    assert "unsupported" in kinds(rep) and "floating" in kinds(rep)
    side = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 5, 0, 3)])                # above nothing: next to the brick
    assert "unsupported" in kinds(bm.validate(side, INV))


def test_nothing_can_stand_on_a_tile():
    b = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3068b", "White", 0, 0, 3), P("3003", "Red", 0, 0, 4)])
    assert "unsupported" in kinds(bm.validate(b, INV))


def test_step_order_a_part_must_rest_on_earlier_or_same_step():
    ok = Build("t", "", [P("3001", "Red", 0, 0, 0, 0, 1), P("3003", "Red", 0, 0, 3, 0, 1)])                 # same step: fine
    assert bm.validate(ok, INV).ok
    bad = Build("t", "", [P("3001", "Red", 0, 0, 0, 0, 3), P("3003", "Red", 0, 0, 3, 0, 1)])                # placed before its support
    assert "step_order" in kinds(bm.validate(bad, INV))


def test_separate_sub_assemblies_are_reported_not_rejected():
    b = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3001", "Red", 10, 0, 0)])
    rep = bm.validate(b, INV)
    assert rep.ok and rep.components == 2 and any("sub-assemblies" in w.message for w in rep.warnings)


def test_non_integer_values_and_bad_rotation_are_errors():
    b = Build("t", "", [P("3001", "Red", 0.5, 0, 0), P("3003", "Red", 9, 0, 0, 45)])
    assert kinds(bm.validate(b, INV)).count("bad_value") == 2


# ---------------------------------------------------------------- repair
def test_repair_snaps_to_the_grid_and_normalises_rotation():
    b = Build("t", "", [P("3001", "Red", 0.4, 0.2, 0.0, 91, 1.0)])
    fixed, log = bm.repair(b, INV)
    p = fixed.parts[0]
    assert (p.x, p.z, p.y, p.rotation) == (0, 0, 0, 90) and any("snapped" in a for a in log.actions)


def test_repair_nudges_an_overlapping_part_to_a_free_spot():
    b = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3001", "Red", 2, 0, 0)])
    fixed, log = bm.repair(b, INV)
    assert len(fixed.parts) == 2 and log.nudged == 1 and bm.validate(fixed, INV).ok


def test_repair_swaps_to_an_equivalent_inventory_part_or_colour():
    b = Build("t", "", [P("3001", "Green", 0, 0, 0)])                 # no green brick: another colour of the same part
    fixed, log = bm.repair(b, INV)
    assert fixed.parts[0].colour in ("Red", "Blue") and log.swapped == 1
    b2 = Build("t", "", [P("3010", "Red", 0, 0, 0)])                  # no 1x4 at all: nothing equivalent -> dropped
    with_nothing, log2 = bm.repair(b2, INV)
    assert with_nothing.parts == [] and log2.dropped == 1


def test_a_swap_never_uses_a_piece_another_part_needs_exactly():
    reds = [P("3003", "Red", 3 * i, 0, 0) for i in range(3)]                       # all three red 2x2
    b = Build("t", "", reds + [P("3003", "Green", 12, 0, 0)])                     # a green one that does not exist
    fixed, log = bm.repair(b, INV)
    assert len(fixed.parts) == 3 and log.dropped == 1 and log.swapped == 0


def test_overuse_is_trimmed_by_swapping_or_dropping():
    b = Build("t", "", [P("3022", "Blue", 4 * i, 0, 0) for i in range(4)])        # own 2 blue 2x2 plates; equivalents: none
    fixed, log = bm.repair(b, INV)
    assert len(fixed.parts) == 2 and log.dropped == 2


def test_repair_lowers_a_hanging_part_onto_its_support_and_drops_what_rests_on_nothing():
    gap = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 0, 0, 5)])  # hangs 2 plates above the brick
    fixed, log = bm.repair(gap, INV)
    assert [p.y for p in fixed.parts] == [0, 3] and bm.validate(fixed, INV).ok
    nowhere = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3003", "Red", 9, 0, 6)])   # above empty ground... falls to the ground
    fixed2, _ = bm.repair(nowhere, INV)
    assert bm.validate(fixed2, INV).ok
    tile_on_top = Build("t", "", [P("3001", "Red", 0, 0, 0), P("3068b", "White", 0, 0, 3), P("3003", "Red", 0, 0, 4)])
    fixed3, log3 = bm.repair(tile_on_top, INV)
    assert bm.validate(fixed3, INV).ok and not any(p.part_id == "3003" and p.y == 4 for p in fixed3.parts)


def test_repair_regenerates_invalid_steps():
    b = Build("t", "", [P("3001", "Red", 0, 0, 0, 0, 3), P("3003", "Red", 0, 0, 3, 0, 1)])
    fixed, log = bm.repair(b, INV)
    assert not bm.check_steps(fixed.parts, INV) and any("steps regenerated" in a for a in log.actions)


def test_drop_fraction():
    assert bm.drop_fraction(10, 8) == pytest.approx(0.2) and bm.drop_fraction(0, 0) == 0.0


# ---------------------------------------------------------------- steps
def test_generated_steps_go_bottom_up_in_small_groups_and_never_before_their_support():
    parts = [P("3001", "Red", 0, 0, 0), P("3001", "Red", 4, 0, 0), P("3001", "Red", 0, 0, 3), P("3003", "Red", 4, 0, 3),
             P("3020", "Red", 0, 0, 6), P("3022", "Blue", 4, 0, 6), P("3005", "Yellow", 0, 0, 7)]
    stepped = bm.generate_steps(parts, INV, size=3)
    assert not bm.check_steps(stepped, INV)
    by = bm.steps_of(Build("t", "", stepped))
    assert sorted(by) == list(range(1, len(by) + 1)) and all(1 <= len(v) <= 3 for v in by.values())
    first = by[1]
    assert all(p.y == 0 for p in first)
    for step, ps in by.items():                                                  # each step's parts are close together
        xs = [(p.x, p.z) for p in ps]
        assert max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in xs for b in xs) <= 12
    assert bm.validate(Build("t", "", stepped), INV).ok


def test_generated_steps_respect_the_group_size_limit():
    parts = [P("3005", "Yellow", i, 0, 0) for i in range(4)]
    assert all(len(v) <= 2 for v in bm.steps_of(Build("t", "", bm.generate_steps(parts, INV, size=2))).values())


# ---------------------------------------------------------------- LDraw converter
def test_ldraw_coordinates_for_known_placements():
    b = Build("Known", "", [P("3001", "Red", 0, 0, 0, 0, 1), P("3001", "Blue", 4, 0, 0, 90, 2), P("3020", "Red", 0, 0, 3, 0, 2)])
    ldr = ldrconv.build_to_ldr(b, INV)
    lines = [ln for ln in ldr.splitlines() if ln.startswith("1 ")]
    # 4x2 brick at the origin: centre (2 studs, 1 stud) = (40, 20), top of a 3-plate brick on the ground = -24
    assert lines[0] == "1 4 40 -24 20 1 0 0 0 1 0 0 0 1 3001.dat"
    # the same brick turned 90 at x=4: footprint 2 x 4 studs, centre (4+1, 0+2) = (100, 40); matrix for 90 degrees
    assert lines[1] == "1 1 100 -24 40 0 0 -1 0 1 0 1 0 0 3001.dat"
    # a plate on top: bottom at plate 3, 1 plate tall -> top at -(3+1)*8 = -32
    assert lines[2] == "1 4 40 -32 20 1 0 0 0 1 0 0 0 1 3020.dat"
    assert ldr.count("0 STEP") == 2 and "0 !LDRAW_ORG Unofficial_Model" in ldr


def test_round_trip_build_to_ldr_and_back():
    build = sample_ai_build()
    back = ldrconv.ldr_to_build(ldrconv.build_to_ldr(build, INV), INV)
    key = lambda p: (p.step, p.part_id, p.colour, p.x, p.z, p.y, p.rotation)          # noqa: E731
    assert sorted(map(key, back.parts)) == sorted(map(key, build.parts))
    assert bm.validate(back, INV).ok


def test_ldr_steps_are_numbered_like_threejs_and_inherit_colour():
    mpd = "\n".join([
        "0 FILE main.ldr", "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat", "0 STEP", "1 16 0 -24 0 1 0 0 0 1 0 0 0 1 sub.ldr", "0 STEP",
        "1 14 0 -48 0 1 0 0 0 1 0 0 0 1 3005.dat", "0 FILE sub.ldr", "1 16 0 0 0 1 0 0 0 1 0 0 0 1 3003.dat", "0 STEP",
        "1 16 0 -24 0 1 0 0 0 1 0 0 0 1 3022.dat"])
    uses = ldrconv.ldr_steps(mpd)
    assert [(u.step, u.ld_part, u.colour) for u in uses] == [(0, "3001", "4"), (1, "3003", "16"), (2, "3022", "16"), (3, "3005", "14")]
    assert ldrconv.has_steps(mpd) and not ldrconv.has_steps("1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat")


def test_add_steps_gives_a_stepless_model_bottom_up_steps():
    flat = "\n".join(["0 Test", "0 Author: Someone", "0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt",
                      "1 4 0 -48 0 1 0 0 0 1 0 0 0 1 3001.dat", "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat", "1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat"])
    out = ldrconv.add_steps(flat)
    assert "0 Author: Someone" in out and "CCAL" in out
    lines = [ln for ln in out.splitlines() if ln.startswith("1 ")]
    ys = [float(ln.split()[3]) for ln in lines]
    assert ys == sorted(ys, reverse=True)                                       # LDraw Y points down: lowest part first
    assert ldrconv.has_steps(out)


# ---------------------------------------------------------------- against the real LDraw geometry (skipped without the library)
def _library():
    try:
        from backend.core.ldraw_lib import get_library
        return get_library()
    except Exception:                                                          # noqa: BLE001
        pytest.skip("LDraw library not downloaded (python -m backend.core.fetch_ldraw)")


def test_every_table_entry_matches_the_real_part_geometry():
    """Footprint, height, long side along X and the origin at the top-middle: the facts the converter relies on."""
    from backend.core.ldraw_lib import part_bounds
    lib = _library()
    for part, d in KNOWN.items():
        bb = part_bounds(lib, part)
        assert bb is not None, part
        lo, hi = bb
        size = hi - lo
        fx, fz = d.footprint(0)
        assert size[0] == pytest.approx(fx * 20, abs=0.5) and size[2] == pytest.approx(fz * 20, abs=0.5), part
        assert size[1] == pytest.approx(d.height * 8, abs=0.5), part
        x0, x1, z0, z1 = d.box or (-fx * 10, fx * 10, -fz * 10, fz * 10)     # origin in the middle, or where the table says it is
        assert np.allclose([lo[0], hi[0], lo[2], hi[2]], [x0, x1, z0, z1], atol=0.5), part
        assert lo[1] == pytest.approx(0, abs=0.5) and hi[1] == pytest.approx(d.height * 8, abs=0.5), part   # origin at the top, body below
        if d.box is None:
            assert (d.long, d.short) == (fx, fz), part                      # long side along X


def test_placed_parts_land_exactly_on_their_grid_cells_for_every_rotation():
    """Orientation check against the real geometry: take the LDraw line written for a placement, transform the part's own bounding
    box with its matrix and position, and compare with the cells the validator thinks it occupies (x right, z towards the viewer,
    y up in plates, LDraw's -Y up). A wrong matrix, centre or height would show here."""
    from backend.core.ldraw_lib import part_bounds
    lib = _library()
    inv = INV
    for part, colour, ptype in (("3001", "Red", "4x2 brick"), ("3020", "Red", "4x2 plate"), ("3005", "Yellow", "1x1 brick"),
                                ("3068b", "White", "2x2 tile"), ("3003", "Red", "2x2 brick")):
        item = inv.get(part, colour)
        lo, hi = part_bounds(lib, item.ld_part)
        for rot in (0, 90, 180, 270):
            for (x, z, y) in ((0, 0, 0), (3, 5, 6), (-2, 1, 3)):
                b = Build("t", "", [P(part, colour, x, z, y, rot, 1)])
                line = next(ln for ln in ldrconv.build_to_ldr(b, inv).splitlines() if ln.startswith("1 "))
                t = [float(v) for v in line.split()[2:14]]
                pos, m = np.array(t[:3]), np.array(t[3:]).reshape(3, 3)
                corners = np.array([[a, bb, c] for a in (lo[0], hi[0]) for bb in (lo[1], hi[1]) for c in (lo[2], hi[2])])
                world = corners @ m.T + pos
                fx, fz = item.dims.footprint(rot)
                assert np.allclose(world.min(axis=0), [x * 20, -(y + item.dims.height) * 8, z * 20], atol=0.6), (part, rot, x, z, y)
                assert np.allclose(world.max(axis=0), [(x + fx) * 20, -y * 8, (z + fz) * 20], atol=0.6), (part, rot, x, z, y)
