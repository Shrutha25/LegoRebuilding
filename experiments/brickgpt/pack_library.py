"""Pack the offline library (experiments/brickgpt/library/*.json) for the curation page: our own LDraw packer for the 3D viewer, a small
isometric thumbnail per design, and gallery/library-index.json. Uses backend/core/ldraw_lib, so run with the app venv:

    python experiments/brickgpt/pack_library.py
Colours are not tuned: every brick is the same lime."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from backend.core.ldraw_lib import get_library  # noqa: E402
from inventory import dims_part, make_steps, read_ldconfig, step_meta, to_ldr  # noqa: E402

LIB, GALLERY = HERE / "library", HERE / "gallery"
EXCLUDE_CATEGORIES = {"bowl", "mug"}   # never produced a usable design; the curation page only lists one-piece designs
COLOUR = "Medium Lime"
LIME = (170, 215, 20)


class B:
    """The few Brick fields inventory.py needs, without importing brickgpt (torch) into the app venv."""
    def __init__(self, d):
        self.h, self.w, self.x, self.y, self.z = d["h"], d["w"], d["x"], d["y"], d["z"]
        self.part_id = dims_part(self.h, self.w) + ".DAT"


def thumbnail(bricks, path, size=(260, 220)):
    """Isometric sketch: boxes, far-to-near, three shades. Only for scanning many designs quickly; the 3D button opens the real viewer."""
    def pt(x, y, z):
        return ((x - y) * 0.866, (x + y) * 0.5 - z * 1.2)
    boxes = []
    for b in sorted(bricks, key=lambda b: (b.z, b.x + b.h / 2 + b.y + b.w / 2)):
        x0, y0, z0, x1, y1, z1 = b.x, b.y, b.z, b.x + b.h, b.y + b.w, b.z + 1
        boxes.append({"top": [pt(x0, y0, z1), pt(x1, y0, z1), pt(x1, y1, z1), pt(x0, y1, z1)],
                      "left": [pt(x0, y1, z1), pt(x1, y1, z1), pt(x1, y1, z0), pt(x0, y1, z0)],
                      "right": [pt(x1, y0, z1), pt(x1, y1, z1), pt(x1, y1, z0), pt(x1, y0, z0)]})
    pts = [p for bx in boxes for f in bx.values() for p in f]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    s = min((size[0] - 16) / (max(xs) - min(xs) or 1), (size[1] - 16) / (max(ys) - min(ys) or 1))
    ox, oy = (size[0] - s * (max(xs) - min(xs))) / 2 - s * min(xs), (size[1] - s * (max(ys) - min(ys))) / 2 - s * min(ys)
    img = Image.new("RGB", size, (244, 244, 246))
    d = ImageDraw.Draw(img)
    for bx in boxes:
        for face, k in (("left", 0.72), ("right", 0.55), ("top", 1.0)):
            d.polygon([(ox + s * x, oy + s * y) for x, y in bx[face]], fill=tuple(int(c * k) for c in LIME), outline=(40, 50, 20))
    img.save(path)


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    lib, ld = get_library(), read_ldconfig()
    index = []
    for stale in GALLERY.glob("lib-*"):     # rebuilt from scratch, so designs that no longer qualify disappear
        stale.unlink()
    for f in sorted(LIB.glob("*.json")):
        rec = json.loads(f.read_text())
        bricks = [B(d) for d in rec["brick_list"]]
        if not bricks or not rec["single_piece"] or rec["category"] in EXCLUDE_CATEGORIES:
            continue
        gid = f"lib-{rec['id']}"
        steps = make_steps(bricks)
        text = to_ldr(bricks, [COLOUR] * len(bricks), steps, ld)
        (GALLERY / f"{gid}.mpd").write_text(lib.pack(text))
        (GALLERY / f"{gid}.lite.mpd").write_text(lib.pack(text, drop_conditional=True))
        run = {k: v for k, v in rec.items() if k != "brick_list"}
        meta = {"kind": "ai", "title": f"AI design: {rec['category']} (seed {rec['seed']})", "description": rec["prompt"], "attribution": None,
                "repairs": [], "warnings": [], "id": f"brickgpt-{gid}", "steps": step_meta(bricks, [COLOUR] * len(bricks), steps, ld),
                "step_count": len(steps), "part_count": len(bricks), "missing_files": [],
                "model_url": f"/brickgpt-files/{gid}.mpd", "lite_model_url": f"/brickgpt-files/{gid}.lite.mpd", "run": run}
        (GALLERY / f"{gid}.meta.json").write_text(json.dumps(meta, indent=1))
        thumbnail(bricks, GALLERY / f"{gid}.png")
        index.append({**run, "gid": gid, "thumb": f"/brickgpt-files/{gid}.png"})
    (GALLERY / "library-index.json").write_text(json.dumps(index, indent=1))
    print("packed", len(index), "one-piece designs")


if __name__ == "__main__":
    main()
