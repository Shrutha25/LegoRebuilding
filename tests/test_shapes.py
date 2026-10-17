"""Round parts, cones and slopes in the grid: their dimensions, LDraw origin, rotation and slope direction against the real geometry."""

import numpy as np
import pytest

from backend.core import ldrconv
from backend.core.buildmodel import Build, Inventory, Placement as P
from backend.core.partdims import KNOWN, dims_for
from backend.recipes.inventories import make_item

SLOPES = ("3040b", "3039", "3037", "3038", "3298", "4286")
ROUND = ("3062b", "3941", "6143", "87081", "4032a", "60474", "4589", "98138", "14769", "4150")


def _library():
    try:
        from backend.core.ldraw_lib import get_library
        return get_library()
    except Exception:                                                          # noqa: BLE001
        pytest.skip("LDraw library not downloaded (python -m backend.core.fetch_ldraw)")


def test_shape_tables():
    for part in SLOPES:
        d = KNOWN[part]
        assert d.shape == "slope" and not d.studs_top and d.height == 3 and d.descent == "-z" and d.box
    for part in ROUND:
        d = KNOWN[part]
        assert d.shape in ("round", "cone") and d.long == d.short and d.box is None
    assert dims_for("3040b").footprint(0) == (1, 2) and dims_for("3040b").footprint(90) == (2, 1)
    assert dims_for("3037").footprint(0) == (4, 2) and dims_for("3298").footprint(90) == (3, 2)


def test_slope_direction_turns_with_the_part():
    d = KNOWN["3039"]
    assert [d.descent_at(r) for r in (0, 90, 180, 270)] == ["-z", "+x", "+z", "-x"]


@pytest.mark.parametrize("part", SLOPES)
def test_a_slope_falls_towards_its_declared_direction_in_the_real_geometry(part):
    from backend.core.ldraw_lib import part_points
    pts = part_points(_library(), part)
    d = KNOWN[part]
    # LDraw: -Y is up, the origin is at the top. The top surface's highest points (y smallest) are at the high end (+z).
    top_z = pts[pts[:, 1] < 0.5][:, 2]
    assert top_z.min() > pts[:, 2].min() + 5, part                 # the full-height edge sits at the high end, not along the whole length
    assert top_z.max() == pytest.approx(d.box[3], abs=0.5), part
    low_end = pts[pts[:, 2] < d.box[2] + 1]
    assert low_end[:, 1].max() == pytest.approx(24, abs=0.5) and low_end[:, 1].min() > 8, part   # at the low end the surface is lower than the top


@pytest.mark.parametrize("part", SLOPES + ("3941", "4589", "60474"))
def test_placed_shaped_parts_land_exactly_on_their_cells_for_every_rotation(part):
    from backend.core.ldraw_lib import part_bounds
    lib = _library()
    item = make_item(part, part, "Red", 1)
    inv = Inventory([item])
    lo, hi = part_bounds(lib, item.ld_part)
    for rot in (0, 90, 180, 270):
        for (x, z, y) in ((0, 0, 0), (3, 5, 6), (-2, 1, 3)):
            b = Build("t", "", [P(part, "Red", x, z, y, rot, 1)])
            line = next(ln for ln in ldrconv.build_to_ldr(b, inv).splitlines() if ln.startswith("1 "))
            t = [float(v) for v in line.split()[2:14]]
            pos, m = np.array(t[:3]), np.array(t[3:]).reshape(3, 3)
            corners = np.array([[a, bb, c] for a in (lo[0], hi[0]) for bb in (lo[1], hi[1]) for c in (lo[2], hi[2])])
            world = corners @ m.T + pos
            fx, fz = item.dims.footprint(rot)
            assert np.allclose(world.min(axis=0), [x * 20, -(y + item.dims.height) * 8, z * 20], atol=0.6), (part, rot, x, z, y)
            assert np.allclose(world.max(axis=0), [(x + fx) * 20, -y * 8, (z + fz) * 20], atol=0.6), (part, rot, x, z, y)
            assert ldrconv.ldr_to_build(ldrconv.build_to_ldr(b, inv), inv).parts[0].x == x             # and back again
