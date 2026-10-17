"""Inventories and builds for the 3D-build tests (no LDraw library needed)."""

import json
from pathlib import Path

from backend.core.buildmodel import Build, Inventory, InvItem, Placement
from backend.core.partdims import dims_for

FIXTURES = Path(__file__).resolve().parent / "data"


def item(part_id, name, colour, qty, ld_colour):
    return InvItem(part_id, name, colour, qty, dims_for(part_id, name), part_id, ld_colour)


def inventory():
    return Inventory([
        item("3001", "Brick 2 x 4", "Red", 4, 4),
        item("3001", "Brick 2 x 4", "Blue", 2, 1),
        item("3003", "Brick 2 x 2", "Red", 3, 4),
        item("3005", "Brick 1 x 1", "Yellow", 4, 14),
        item("3020", "Plate 2 x 4", "Red", 2, 4),
        item("3022", "Plate 2 x 2", "Blue", 2, 1),
        item("3068b", "Tile 2 x 2 with Groove", "White", 2, 15),
    ])


def P(part_id, colour, x, z, y, rot=0, step=1):
    return Placement(part_id, colour, x, z, y, rot, step)


def tower():
    """Two bricks and a plate on top: a valid little build."""
    return Build("Tower", "test", [
        P("3001", "Red", 0, 0, 0, 0, 1),          # 4 x 2 on the ground
        P("3001", "Red", 0, 0, 3, 0, 2),          # same footprint, one brick up
        P("3020", "Red", 0, 0, 6, 0, 3),          # plate on top
    ])


def sample_ai_build():
    """The AI-build fixture (tests/data/ai_build.json) as a Build."""
    data = json.loads((FIXTURES / "ai_build.json").read_text(encoding="utf-8"))["build"]
    return Build(data["title"], data["description"],
                 [Placement(p["part_id"], p["colour"], p["x"], p["z"], p["y"], p["rotation"], p["step"]) for p in data["parts"]])


# ---------------------------------------------------------------- a tiny LDraw library in a zip
LDCONFIG = """0 LDraw.org Configuration File
0 !COLOUR Black CODE 0 VALUE #1B2A34 EDGE #808080
0 !COLOUR Blue CODE 1 VALUE #1E5AA8 EDGE #333333
0 !COLOUR Red CODE 4 VALUE #B40000 EDGE #333333
0 !COLOUR Yellow CODE 14 VALUE #FAC80A EDGE #333333
0 !COLOUR White CODE 15 VALUE #F4F4F4 EDGE #333333
0 !COLOUR Light_Bluish_Grey CODE 71 VALUE #969696 EDGE #333333
0 !COLOUR Medium_Azure CODE 322 VALUE #68C3E2 EDGE #333333
0 !COLOUR Trans_Clear CODE 47 VALUE #FCFCFC EDGE #C3C3C3 ALPHA 128
"""


def _box(w, h, d):
    return (f"4 16 {-w} 0 {-d} {-w} {h} {-d} {w} {h} {-d} {w} 0 {-d}\n4 16 {w} 0 {d} {w} {h} {d} {-w} {h} {d} {-w} 0 {d}\n")


def fake_library_zip(path, parts=None):
    """A zip laid out like complete.zip with a handful of simple parts (box + a stud primitive + one s\ subpart)."""
    import zipfile
    parts = parts or {"3001": ("Brick 2 x 4", 40, 24, 20), "3003": ("Brick 2 x 2", 20, 24, 20), "3005": ("Brick 1 x 1", 10, 24, 10),
                      "3020": ("Plate 2 x 4", 40, 8, 20), "3022": ("Plate 2 x 2", 20, 8, 20), "3068b": ("Tile 2 x 2 with Groove", 20, 8, 20)}
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ldraw/LDConfig.ldr", LDCONFIG)
        z.writestr("ldraw/p/stud.dat", "0 Stud\n4 16 -6 0 -6 -6 -4 -6 6 -4 -6 6 0 -6\n")
        z.writestr("ldraw/parts/s/3001s01.dat", "0 ~Brick 2 x 4 Sub\n" + _box(40, 24, 20))
        for pid, (name, w, h, d) in parts.items():
            body = f"0 {name}\n0 Name: {pid}.dat\n1 16 0 -4 0 1 0 0 0 1 0 0 0 1 stud.dat\n"
            body += "1 16 0 0 0 1 0 0 0 1 0 0 0 1 s" + chr(92) + "3001s01.dat\n" if pid == "3001" else _box(w, h, d)
            z.writestr(f"ldraw/parts/{pid}.dat", body)
    return path
