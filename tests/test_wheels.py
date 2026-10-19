"""Wheel sets in the grid (4600-style holder + 2 rims + 2 tyres): the dimension table against the real LDraw geometry, the solver placing them only when the pile has them."""

import numpy as np
import pytest

from backend.core import ldrconv
from backend.core.buildmodel import Build, Placement as P, validate
from backend.core.partdims import KNOWN
from backend.designs import sketch as sk
from backend.designs.solver import solve
from backend.recipes.inventories import inventory_from


def _library():
    try:
        from backend.core.ldraw_lib import get_library
        return get_library()
    except Exception:                                                          # noqa: BLE001
        pytest.skip("LDraw library not downloaded (python -m backend.core.fetch_ldraw)")


def _parts(inv):
    return [(i.part_id, i.colour, i.dims, i.qty) for i in inv.parts()]


WHEELS = [("4600", "Red", 1), ("4624", "Light Bluish Gray", 2), ("3641", "Black", 2)]


def _world_bounds(lib, build, inv):
    from backend.core.ldraw_lib import part_bounds
    out = {}
    for ln, p in zip([ln for ln in ldrconv.build_to_ldr(build, inv).splitlines() if ln.startswith("1 ")], sorted(build.parts, key=lambda q: (q.y, q.z, q.x))):
        t = ln.split()
        pos, m = np.array([float(v) for v in t[2:5]]), np.array([float(v) for v in t[5:14]]).reshape(3, 3)
        lo, hi = part_bounds(lib, t[14].removesuffix(".dat"))
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        w = corners @ m.T + pos
        out.setdefault(p.part_id, []).append((w.min(axis=0), w.max(axis=0), p))
    return out


def test_wheel_tables():
    assert KNOWN["4600"].kind == "wheelbase" and KNOWN["4600"].footprint(0) == (2, 2) and KNOWN["4600"].height == 3
    assert KNOWN["3641"].footprint(90) == (1, 2) and KNOWN["3641"].footprint(0) == (2, 1) and KNOWN["3641"].height == 5
    assert KNOWN["4624"].height == 0 and KNOWN["4624"].footprint(270) == (1, 2)


@pytest.mark.parametrize("x,z", [(0, 0), (3, 5)])
def test_a_wheel_set_matches_the_ldraw_geometry_along_x(x, z):
    lib = _library()
    inv = inventory_from(WHEELS)
    b = Build("t", "", [P("4600", "Red", x + 1, z, 0, 0, 1), P("4624", "Light Bluish Gray", x, z, 0, 270, 1), P("4624", "Light Bluish Gray", x + 3, z, 0, 90, 1),
                        P("3641", "Black", x, z, 0, 270, 1), P("3641", "Black", x + 3, z, 0, 90, 1)])
    got = _world_bounds(lib, b, inv)
    for lo, hi, p in got["3641"]:                                      # a tyre: 16 wide (the axle), 36 across, inside its 1 x 2 stud column, standing on the ground
        cx = p.x * 20 + 10
        assert np.allclose([lo[0], hi[0]], [cx - 8, cx + 8], atol=0.6)
        assert np.allclose([lo[2], hi[2]], [z * 20 + 2, z * 20 + 38], atol=0.6)
        assert np.allclose([lo[1], hi[1]], [-36, 0], atol=0.6)
    for lo, hi, p in got["4624"]:                                      # the rim has the same centre as its tyre
        cx = p.x * 20 + 10
        assert np.allclose([lo[0], hi[0]], [cx - 8, cx + 8], atol=0.6) and abs((lo[1] + hi[1]) / 2 + 18) < 0.6 and abs((lo[2] + hi[2]) / 2 - (z * 20 + 20)) < 0.6
    (lo, hi, p), = got["4600"]                                         # the holder: its plate top 24 above the ground, between the tyres, pins reaching in
    assert abs(lo[1] + 24) < 0.6 and abs((lo[0] + hi[0]) / 2 - (x + 2) * 20) < 0.6 and np.allclose([lo[2], hi[2]], [z * 20, z * 20 + 40], atol=0.6)
    assert lo[0] > x * 20 and hi[0] < (x + 4) * 20                     # all inside the 4 x 2 stud footprint
    rep = validate(b, inv)
    assert rep.ok and rep.components == 1, rep.errors
    assert ldrconv.ldr_to_build(ldrconv.build_to_ldr(b, inv), inv).parts[0].y == 0


def test_a_wheel_set_matches_the_ldraw_geometry_along_z():
    lib = _library()
    inv = inventory_from(WHEELS)
    b = Build("t", "", [P("4600", "Red", 2, 1, 0, 90, 1), P("4624", "Light Bluish Gray", 2, 0, 0, 0, 1), P("4624", "Light Bluish Gray", 2, 3, 0, 180, 1),
                        P("3641", "Black", 2, 0, 0, 0, 1), P("3641", "Black", 2, 3, 0, 180, 1)])
    got = _world_bounds(lib, b, inv)
    for lo, hi, p in got["3641"]:
        cz = p.z * 20 + 10
        assert np.allclose([lo[2], hi[2]], [cz - 8, cz + 8], atol=0.6) and np.allclose([lo[0], hi[0]], [2 * 20 + 2, 2 * 20 + 38], atol=0.6)
    (lo, hi, _), = got["4600"]
    assert abs(lo[1] + 24) < 0.6 and abs((lo[2] + hi[2]) / 2 - 2 * 20) < 0.6 and np.allclose([lo[0], hi[0]], [40, 80], atol=0.6)
    assert validate(b, inv).ok


def _car():
    """A 4 x 2 chassis two bricks tall with a cabin on top: the wheel zone is the whole chassis."""
    return sk.parse({"layers": [["####", "####"], ["####", "####"], [".##.", ".##."]]})


def test_the_solver_stands_a_wheel_set_in_a_vehicle_sketch_when_the_pile_has_one():
    inv = inventory_from(WHEELS + [("3003", "Red", 1), ("3004", "Yellow", 2)])
    r = solve(_car(), _parts(inv), time_limit=2.0)
    kinds = sorted(p[0] for p in r.placements)
    assert kinds.count("4600") == 1 and kinds.count("4624") == 2 and kinds.count("3641") == 2, kinds
    b = Build("car", "", [P(pid, c, x, z, y, rot, 1) for pid, c, x, z, y, rot in r.placements])
    rep = validate(b, inv)
    assert rep.ok and rep.components == 1, (rep.errors, rep.warnings)
    assert any(p[0] == "3003" or p[0] == "3004" for p in r.placements)            # something sits on the holder's studs


def test_wheels_are_not_used_when_the_pile_has_no_complete_set():
    for pile in (WHEELS[:1] + [("3001", "Red", 2)], [("4600", "Red", 1), ("4624", "Light Bluish Gray", 1), ("3641", "Black", 2), ("3001", "Red", 2)],
                 [("3001", "Red", 2), ("3003", "Red", 1)]):
        inv = inventory_from(pile)
        r = solve(_car(), _parts(inv), time_limit=1.0)
        assert not any(p[0] in ("4600", "4624", "3641") for p in r.placements)


def test_wheels_need_a_sketch_that_has_room_for_them():
    inv = inventory_from(WHEELS)
    thin = sk.parse({"layers": [["####", "####"]]}, min_cells=2)                  # one brick tall: the tyres (5 plates) do not fit
    assert not solve(thin, _parts(inv), time_limit=1.0).placements


def test_gemini_is_told_about_wheel_sets_only_when_the_pile_has_one():
    from backend.designs.affordance import summarize
    assert "Wheel set" in summarize(inventory_from(WHEELS + [("3001", "Red", 1)])).prompt()
    assert "Wheel set" not in summarize(inventory_from([("3001", "Red", 1), ("3641", "Black", 2)])).prompt()
