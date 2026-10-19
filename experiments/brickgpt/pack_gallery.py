"""Pack the BrickGPT outputs (Phase 0 + Phase 1) for the dev gallery, into experiments/brickgpt/gallery/ (gitignored).
Uses our own packer (backend/core/ldraw_lib), so run it with the app venv:   python experiments/brickgpt/pack_gallery.py
The gallery page (frontend/app/brickgpt-gallery) reads gallery/index.json through the dev-only /brickgpt-files route."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.core.ldraw_lib import get_library  # noqa: E402

GALLERY = ROOT / "experiments" / "brickgpt" / "gallery"
P0 = ROOT / "docs" / "brickgpt" / "phase0"
NAMES = {"3005": "Brick 1 x 1", "3004": "Brick 1 x 2", "3010": "Brick 1 x 4", "3009": "Brick 1 x 6", "3008": "Brick 1 x 8",
         "3003": "Brick 2 x 2", "3001": "Brick 2 x 4", "2456": "Brick 2 x 6"}


def phase0_entries():
    """Phase 0 outputs (one colour, a STEP per brick) as gallery entries."""
    for r in json.loads((P0 / "results.json").read_text())["results"]:
        name = r["prompt"].replace("a ", "").replace(" ", "-")
        text = (P0 / f"{name}.ldr").read_text()
        parts = [l.split()[-1].split(".")[0] for l in text.splitlines() if l.startswith("1 ")]
        rid = f"phase0-{name}"
        (GALLERY / f"{rid}.ldr").write_text(text)
        steps = [{"index": i, "parts": [{"ld_part": p, "name": NAMES[p], "colour_code": "115", "colour_name": "Medium Lime",
                                         "hex": "#C0F500", "qty": 1}]} for i, p in enumerate(parts)]
        run = {"id": rid, "prompt": r["prompt"], "mode": "unconstrained", "inventory": None, "inventory_bricks": None,
               "bricks": r["bricks"], "steps": len(steps), "end_reason": "finished", "connected": r["connected"],
               "seconds": r["seconds"], "rollbacks": r["rollbacks"]}
        meta = {"kind": "ai", "title": f"{r['prompt']} (phase 0)", "description": "BrickGPT, seed 42", "attribution": None,
                "repairs": [], "warnings": [], "id": f"brickgpt-{rid}", "steps": steps, "step_count": len(steps),
                "part_count": len(parts), "missing_files": [], "run": run}
        (GALLERY / f"{rid}.meta.json").write_text(json.dumps(meta, indent=1))


def main():
    GALLERY.mkdir(parents=True, exist_ok=True)
    lib = get_library()
    phase0_entries()
    index = []
    for ldr in sorted(GALLERY.glob("*.ldr")):
        meta = json.loads((GALLERY / f"{ldr.stem}.meta.json").read_text())
        text = ldr.read_text()
        (GALLERY / f"{ldr.stem}.mpd").write_text(lib.pack(text))
        (GALLERY / f"{ldr.stem}.lite.mpd").write_text(lib.pack(text, drop_conditional=True))
        meta["model_url"], meta["lite_model_url"] = f"/brickgpt-files/{ldr.stem}.mpd", f"/brickgpt-files/{ldr.stem}.lite.mpd"
        (GALLERY / f"{ldr.stem}.meta.json").write_text(json.dumps(meta, indent=1))
        index.append(meta["run"])
    (GALLERY / "index.json").write_text(json.dumps(index, indent=1))
    print("packed", len(index), "builds")


if __name__ == "__main__":
    main()
