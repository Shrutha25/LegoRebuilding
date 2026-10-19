"""Thumbnails and viewer files for the dataset structures in the index. Uses backend/core/ldraw_lib (read only), so run with the app venv:

    .venv/Scripts/python.exe experiments/brickgpt/pack_dataset.py
Run `ingest_dataset.py convert` first. Every dataset structure within max_bricks gets a thumbnail (gallery/ds-<id>.png); the shortlisted
ones also get the packed .mpd / .lite.mpd / .meta.json the 3D viewer opens and a card in gallery/library-index.json, next to the generated
designs pack_library.py put there. Existing files are kept, so a re-run only does what is missing (delete gallery/ds-* to redo)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from backend.core.ldraw_lib import get_library  # noqa: E402
from inventory import make_steps, read_ldconfig, step_meta, to_ldr  # noqa: E402
from library_index import DATASET_DIR, load_bricks, load_config  # noqa: E402
from pack_library import COLOUR, GALLERY, B, thumbnail  # noqa: E402


def card(rec, gid):
    return {"id": rec["id"], "category": rec["category"], "prompt": rec["display_name"], "seed": 0, "bricks": rec["total_bricks"], "seconds": 0, "rollbacks": 0,
            "connected": True, "single_piece": rec["single_piece"], "linked_pieces": rec["linked_pieces"], "linked_sizes": rec["linked_sizes"],
            "tag": "dataset", "gid": gid, "thumb": f"/brickgpt-files/{gid}.png", "stability_min": rec["stability_min"]}


def main():
    cfg = load_config()
    GALLERY.mkdir(parents=True, exist_ok=True)
    index = [json.loads(line) for line in (DATASET_DIR / "library_index.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = {r["id"]: r for r in index}
    short = [recs[i] for i in json.loads((DATASET_DIR / "shortlist.json").read_text(encoding="utf-8")) if recs[i]["source"] == "dataset"]
    lib, ld = get_library(), read_ldconfig()
    thumbs = 0
    for r in index:
        if r["source"] != "dataset" or r["total_bricks"] > cfg["max_bricks"] or (GALLERY / f"{r['id']}.png").exists():
            continue
        thumbnail([B(d) for d in load_bricks(r)], GALLERY / f"{r['id']}.png")
        thumbs += 1
    cards = []
    for r in short:
        gid = r["id"]
        bricks = [B(d) for d in load_bricks(r)]
        steps = make_steps(bricks)
        if not (GALLERY / f"{gid}.meta.json").exists():
            text = to_ldr(bricks, [COLOUR] * len(bricks), steps, ld)
            (GALLERY / f"{gid}.mpd").write_text(lib.pack(text))
            (GALLERY / f"{gid}.lite.mpd").write_text(lib.pack(text, drop_conditional=True))
            meta = {"kind": "ai", "title": f"Dataset: {r['display_name']}", "description": "; ".join(r["captions"]), "attribution": None, "repairs": [], "warnings": [],
                    "id": f"brickgpt-{gid}", "steps": step_meta(bricks, [COLOUR] * len(bricks), steps, ld), "step_count": len(steps), "part_count": len(bricks),
                    "missing_files": [], "model_url": f"/brickgpt-files/{gid}.mpd", "lite_model_url": f"/brickgpt-files/{gid}.lite.mpd", "run": card(r, gid)}
            (GALLERY / f"{gid}.meta.json").write_text(json.dumps(meta, indent=1))
        cards.append(card(r, gid))
    path = GALLERY / "library-index.json"
    existing = [c for c in (json.loads(path.read_text()) if path.exists() else []) if c.get("tag") != "dataset"]
    path.write_text(json.dumps(existing + cards, indent=1))
    print(f"{thumbs} new thumbnails, {len(cards)} shortlisted dataset cards added to {len(existing)} generated ones")


if __name__ == "__main__":
    main()
