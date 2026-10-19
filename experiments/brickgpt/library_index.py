"""Index of brick structures for reverse lookup (pile -> builds). Pure Python + numpy, no torch.

One index record per structure, from two sources: the StableText2Brick dataset ("dataset") and our own BrickGPT designs in library/
("brickgpt_generated"). A record has the library/ fields (id, category, prompt, bricks, types, single_piece, ...) plus source, captions,
display_name, category_id, count_vector (over the 8 types, in reverse_config.json order), total_bricks, bbox [dx, dy, dz] and, for the dataset,
stability_min / stability_mean. The brick list is not in the index: see load_bricks().
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from inventory import dims_part, single_piece

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
LIBRARY_DIR = HERE / "library"
BRICK_LINE = re.compile(r"^(\d+)x(\d+) \((-?\d+),(-?\d+),(-?\d+)\)$")
GRID = 20   # StableText2Brick world: 20 x 20 x 20 voxels


def load_config(path=HERE / "reverse_config.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class LBrick:
    """The Brick fields inventory.py needs (h along x, w along y, z = layer)."""
    def __init__(self, d):
        self.h, self.w, self.x, self.y, self.z = d["h"], d["w"], d["x"], d["y"], d["z"]
        self.part_id = (dims_part(self.h, self.w) or "0") + ".DAT"


def parse_bricks(text):
    """'2x4 (3,5,0)\\n...' -> [{h, w, x, y, z}]; raises ValueError on a malformed line."""
    out = []
    for line in text.strip().splitlines():
        m = BRICK_LINE.match(line.strip())
        if not m:
            raise ValueError(f"bad brick line: {line!r}")
        h, w, x, y, z = map(int, m.groups())
        out.append({"h": h, "w": w, "x": x, "y": y, "z": z})
    return out


def count_vector(brick_list, part_ids) -> list:
    """Bricks per type, in the order of part_ids (1xN and Nx1 are the same part). Raises ValueError for a size outside the 8 types."""
    n = Counter()
    for b in brick_list:
        part = dims_part(b["h"], b["w"])
        if part not in part_ids:
            raise ValueError(f"unsupported brick {b['h']}x{b['w']}")
        n[part] += 1
    return [n[p] for p in part_ids]


def bbox(brick_list) -> list:
    """[dx, dy, dz] in studs / layers."""
    return [max(b["x"] + b["h"] for b in brick_list) - min(b["x"] for b in brick_list),
            max(b["y"] + b["w"] for b in brick_list) - min(b["y"] for b in brick_list),
            max(b["z"] for b in brick_list) - min(b["z"] for b in brick_list) + 1]


def usable_captions(captions, min_chars=3) -> list:
    return [c.strip() for c in (captions or []) if isinstance(c, str) and len(c.strip()) >= min_chars]


def display_name(captions) -> str:
    """The shortest caption (ties: alphabetical, so the result is stable)."""
    return min(captions, key=lambda c: (len(c), c)) if captions else ""


def stability_summary(scores_flat, brick_list):
    """(min, mean) of the stability score over the voxels the bricks occupy. scores_flat: 20*20*20 floats, x-major (empty voxels hold 1.0)."""
    grid = np.asarray(scores_flat, dtype=float).reshape(GRID, GRID, GRID)
    occ = np.zeros((GRID, GRID, GRID), bool)
    for b in brick_list:
        occ[b["x"]:b["x"] + b["h"], b["y"]:b["y"] + b["w"], b["z"]] = True
    vals = grid[occ]
    return float(vals.min()), float(vals.mean())


def category_key(rec, cfg) -> str:
    """Grouping key for diversification: the ShapeNet synset; generated designs map to it by category name."""
    if rec.get("category_id"):
        return rec["category_id"]
    by_name = {v: k for k, v in cfg["synsets"].items()}
    return by_name.get(rec.get("category"), rec.get("category") or "unknown")


def dataset_entry(row, shard, row_no, cfg, stab=None, analyse=True) -> dict:
    """Index record for one dataset row. single_piece is only computed when analyse (it is quadratic in the brick count)."""
    bl = parse_bricks(row["bricks"])
    vec = count_vector(bl, cfg["part_ids"])
    caps = usable_captions(row["captions"], cfg["min_caption_chars"])
    sid = row["structure_id"]
    name = display_name(caps)
    rec = {"id": f"ds-{sid[:12]}", "source": "dataset", "structure_id": sid, "object_id": row.get("object_id"), "shard": shard, "row": row_no,
           "category_id": row.get("category_id"), "category": cfg["synsets"].get(row.get("category_id"), row.get("category_id")),
           "prompt": name, "display_name": name, "captions": caps, "bricks": len(bl), "total_bricks": len(bl),
           "types": {p: n for p, n in zip(cfg["part_ids"], vec) if n}, "count_vector": vec, "bbox": bbox(bl),
           "stability_min": stab[0] if stab else None, "stability_mean": stab[1] if stab else None,
           "single_piece": None, "linked_pieces": None, "linked_sizes": None}
    if analyse:
        sp = single_piece([LBrick(d) for d in bl])
        rec.update(single_piece=sp["single_piece"], linked_pieces=sp["pieces"], linked_sizes=sp["sizes"])
    return rec


def generated_entry(rec, cfg) -> dict:
    """Index record for a library/*.json design (its brick_list stays in that file)."""
    bl = rec["brick_list"]
    vec = count_vector(bl, cfg["part_ids"])
    caps = usable_captions([rec.get("prompt")], cfg["min_caption_chars"])
    out = {k: v for k, v in rec.items() if k != "brick_list"}
    out.update(source="brickgpt_generated", category_id=None, captions=caps, display_name=display_name(caps), total_bricks=len(bl),
               types={p: n for p, n in zip(cfg["part_ids"], vec) if n}, count_vector=vec, bbox=bbox(bl), stability_min=None, stability_mean=None)
    return out


def load_bricks(rec) -> list:
    """The brick list of an index record: library/<id>.json for generated designs, dataset/entries/<id>.json for dataset structures
    (written by `ingest_dataset.py convert` for structures within max_bricks)."""
    path = (LIBRARY_DIR if rec["source"] == "brickgpt_generated" else DATASET_DIR / "entries") / f"{rec['id']}.json"
    return json.loads(path.read_text(encoding="utf-8"))["brick_list"]


# ------------------------------------------------------------------ pre-filter
def passes_filters(rec, cfg) -> bool:
    """Within the cutoff and bbox limit, one piece, a usable caption, stability above the optional floor."""
    if rec["total_bricks"] > cfg["max_bricks"] or not rec["captions"] or rec["single_piece"] is not True:
        return False
    if any(d > m for d, m in zip(rec["bbox"], cfg["max_bbox"])):
        return False
    floor = cfg.get("min_stability")
    return floor is None or rec["stability_min"] is None or rec["stability_min"] >= floor


def rank_key(rec):
    """Preference order: our own designs, then more stable (the tiebreaker), then id."""
    return (rec["source"] != "brickgpt_generated", -(rec["stability_min"] if rec["stability_min"] is not None else 1.0), rec["id"])


def similar_bbox(a, b, tol) -> bool:
    """Footprints equal within tol (orientation ignored) and height within tol."""
    return all(abs(x - y) <= tol for x, y in zip(sorted(a[:2]), sorted(b[:2]))) and abs(a[2] - b[2]) <= tol


def dedupe(recs, tol) -> list:
    """Keep one structure per cluster of same count vector + similar bbox (best ranked first)."""
    kept, by_vec = [], defaultdict(list)
    for r in sorted(recs, key=rank_key):
        group = by_vec[tuple(r["count_vector"])]
        if any(similar_bbox(r["bbox"], k["bbox"], tol) for k in group):
            continue
        group.append(r)
        kept.append(r)
    return kept


def shortlist(index, cfg) -> list:
    """Records for the curation page: filtered, deduplicated, then capped at shortlist_max by round-robin over categories
    (best ranked first in each) so no category crowds out the others."""
    pool = dedupe([r for r in index if passes_filters(r, cfg)], cfg["dedupe"]["bbox_tolerance"])
    groups = defaultdict(list)
    for r in sorted(pool, key=rank_key):
        groups[category_key(r, cfg)].append(r)
    out, keys = [], sorted(groups)
    while len(out) < cfg["shortlist_max"] and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < cfg["shortlist_max"]:
                out.append(groups[k].pop(0))
    return out
