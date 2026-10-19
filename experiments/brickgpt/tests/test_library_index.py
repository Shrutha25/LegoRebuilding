"""Ingestion parsing, count vectors, bbox, captions, dedupe and shortlist. Run:  venv/Scripts/python.exe -m pytest tests/test_library_index.py -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library_index import (count_vector, bbox, dataset_entry, dedupe, display_name, generated_entry, load_config, parse_bricks, shortlist,
                           stability_summary, usable_captions)

CFG = load_config()
TEXT = "1x1 (0,0,0)\n2x1 (1,0,0)\n2x4 (0,2,0)\n1x2 (0,2,1)\n"


def row(text=TEXT, caps=("A tiny thing.", "tiny", "A much longer description of it."), sid="0123456789abcdef"):
    return {"structure_id": sid, "object_id": "o", "category_id": "03001627", "captions": list(caps), "bricks": text}


def test_parse_bricks_reads_dims_and_position():
    assert parse_bricks(TEXT)[1] == {"h": 2, "w": 1, "x": 1, "y": 0, "z": 0}
    assert len(parse_bricks(TEXT)) == 4


@pytest.mark.parametrize("bad", ["2x4 0,0,0", "2x4 (0,0)", "hello"])
def test_parse_bricks_rejects_malformed_lines(bad):
    with pytest.raises(ValueError):
        parse_bricks(bad)


def test_count_vector_follows_config_order_and_merges_orientations():
    vec = count_vector(parse_bricks(TEXT), CFG["part_ids"])
    order = list(CFG["part_ids"])   # 3005, 3004, 3010, 3009, 3008, 3003, 3001, 2456
    assert vec[order.index("3005")] == 1
    assert vec[order.index("3004")] == 2          # 2x1 and 1x2 are the same part
    assert vec[order.index("3001")] == 1
    assert sum(vec) == 4


def test_count_vector_rejects_unsupported_size():
    with pytest.raises(ValueError):
        count_vector([{"h": 1, "w": 3, "x": 0, "y": 0, "z": 0}], CFG["part_ids"])


def test_bbox_is_extent_in_studs_and_layers():
    assert bbox(parse_bricks(TEXT)) == [3, 6, 2]


def test_display_name_is_shortest_caption_and_unusable_captions_are_dropped():
    caps = usable_captions(["  ", "", None, "ab", "Long caption here", " short one "], CFG["min_caption_chars"])
    assert caps == ["Long caption here", "short one"]
    assert display_name(caps) == "short one"
    assert display_name([]) == ""


def test_dataset_entry_has_library_fields_and_flags_unconnected():
    rec = dataset_entry(row(), "train-0", 7, CFG)
    assert rec["id"] == "ds-0123456789ab" and rec["source"] == "dataset" and rec["total_bricks"] == rec["bricks"] == 4
    assert rec["types"] == {"3005": 1, "3004": 2, "3001": 1} and rec["category"] == "chair" and rec["display_name"] == "tiny"
    assert rec["shard"] == "train-0" and rec["row"] == 7 and len(rec["count_vector"]) == 8
    assert rec["single_piece"] is False and rec["linked_pieces"] == 3      # only the 2x4 and the 1x2 on it are linked
    assert dataset_entry(row(), "s", 0, CFG, analyse=False)["single_piece"] is None


def test_entry_without_usable_captions_is_kept_with_empty_captions_and_fails_the_filter():
    rec = dataset_entry(row(caps=["", " "]), "s", 0, CFG)
    assert rec["captions"] == []
    assert shortlist([rec], CFG) == []


def test_stability_summary_only_looks_at_occupied_voxels():
    import numpy as np
    grid = np.ones((20, 20, 20))
    grid[0, 0, 0] = 0.5
    grid[19, 19, 19] = 0.1      # empty voxel: ignored
    lo, mean = stability_summary(grid.ravel(), [{"h": 1, "w": 2, "x": 0, "y": 0, "z": 0}])
    assert lo == 0.5 and mean == 0.75


def test_generated_entry_keeps_library_fields_and_drops_brick_list():
    lib = {"id": "bed-s42", "category": "bed", "prompt": "a bed", "seed": 42, "bricks": 2, "single_piece": True,
           "brick_list": [{"h": 2, "w": 4, "x": 0, "y": 0, "z": 0}, {"h": 2, "w": 4, "x": 0, "y": 0, "z": 1}]}
    rec = generated_entry(lib, CFG)
    assert "brick_list" not in rec and rec["source"] == "brickgpt_generated" and rec["total_bricks"] == 2
    assert rec["types"] == {"3001": 2} and rec["captions"] == ["a bed"] and rec["seed"] == 42


def _rec(i, vec, bb, src="dataset", cat="03001627", stab=0.99, total=10):
    return {"id": i, "source": src, "count_vector": vec, "bbox": bb, "category_id": cat if src == "dataset" else None, "category": "chair",
            "captions": ["x"], "single_piece": True, "total_bricks": total, "stability_min": stab if src == "dataset" else None}


def test_dedupe_merges_same_vector_and_similar_bbox_preferring_generated_then_stable():
    v = [1, 0, 0, 0, 0, 0, 0, 0]
    a, b, c = _rec("a", v, [4, 6, 3], stab=0.9), _rec("b", v, [6, 4, 3], stab=0.99), _rec("c", v, [8, 8, 3])
    g = _rec("g", v, [5, 5, 3], src="brickgpt_generated")
    assert [r["id"] for r in dedupe([a, b, c], 1)] == ["b", "c"]
    assert [r["id"] for r in dedupe([a, b, c, g], 1)] == ["g", "c"]
    assert len(dedupe([_rec("d", [2, 0, 0, 0, 0, 0, 0, 0], [4, 6, 3]), a], 1)) == 2     # different count vector: both kept


def test_shortlist_filters_and_balances_categories():
    cfg = {**CFG, "shortlist_max": 4}
    recs = [_rec(f"t{i}", [i + 1] + [0] * 7, [4, 4, 3], cat="04379243") for i in range(6)] + [_rec("c0", [0, 9] + [0] * 6, [3, 3, 3])]
    recs.append(_rec("big", [0, 0, 5] + [0] * 5, [3, 3, 3], total=cfg["max_bricks"] + 1))
    ids = [r["id"] for r in shortlist(recs, cfg)]
    assert "big" not in ids and "c0" in ids and len(ids) == 4
    assert shortlist(recs, cfg) == shortlist(list(reversed(recs)), cfg)    # deterministic whatever the input order
