"""Section 6 and 7: the Brickognize tie-break, tap-to-merge / draw-to-split, and the stored corrections."""

import asyncio
import json

import cv2
import numpy as np
import pytest

from backend import corrections, jobs
from backend.core import detect, segment as seg
from tests.detection import scenes


def result(score, part="3001"):
    return {"items": [{"id": part, "name": f"Brick {part}", "score": score, "img_url": None}], "colors": [{"name": "Red"}]}


async def fake_recognizer(crops, scores=None):
    return [result(0.5) for _ in crops]


def make_job(tmp_path, img, recognizer=fake_recognizer):
    job = jobs.Job(id="t", dir=tmp_path)
    cv2.imwrite(str(tmp_path / "image.jpg"), img)
    analysis = detect.analyse_masks(img)
    items = [jobs.Item(tuple(p.box), p) for p in analysis.pieces]
    job.items = items
    job.pieces = [jobs.build_piece(i, it.box, f"/c/{i}", result(0.8), False) for i, it in enumerate(items)]
    job.status = "done"
    return job, analysis


# ---------------------------------------------------------------- tie-break
def item_for(region):
    p = detect.Piece(seg.padded_box(region, 0.08, (700, 1000, 3)), "test", region=region, check=region.check)
    return jobs.Item(p.box, p)


def ambiguous_pair(n=0):
    y = 40 + 130 * n
    whole = seg.Region(np.ones((100, 300), np.uint8), 100, y)
    a, b = (seg.Region(np.ones((100, 150), np.uint8), 100 + 150 * k, y) for k in (0, 1))
    amb = seg.Ambiguity(whole, [a, b], "size", 0.1)
    a.ambig = b.ambig = amb
    a.check = b.check = True
    return amb, [item_for(a), item_for(b)]


def test_the_split_wins_only_if_every_part_beats_the_whole_by_the_margin():
    amb, items = ambiguous_pair()
    out, res = jobs.tiebreak(items, [result(0.9), result(0.85)], [result(0.7)], [amb])
    assert len(out) == 2 and not any(i.piece.check for i in out)              # 0.85 >= 0.7 + 0.10: split kept and no longer in doubt


def test_one_weak_part_keeps_the_whole():
    amb, items = ambiguous_pair()
    out, res = jobs.tiebreak(items, [result(0.9), result(0.75)], [result(0.7)], [amb])
    assert len(out) == 1 and out[0].region is amb.whole and top(res[0]) == 0.7


def top(r):
    return r["items"][0]["score"]


def test_the_tiebreak_logs_both_scores(caplog):
    amb, items = ambiguous_pair()
    with caplog.at_level("INFO", logger="lego.jobs"):
        jobs.tiebreak(items, [result(0.9), result(0.85)], [result(0.7)], [amb])
    assert "whole 0.70" in caplog.text and "0.9" in caplog.text


def test_a_job_runs_at_most_max_tiebreaks_and_flags_the_rest_check(tmp_path, monkeypatch):
    img = np.full((700, 1000, 3), 235, np.uint8)
    calls = []

    async def rec(crops):
        calls.append(len(crops))
        return [result(0.9) for _ in crops]

    async def fake_detect(img_, mode, detector, reference=None):
        from backend.core.gemini_detect import Detection
        amb = [ambiguous_pair(n) for n in range(5)]
        items = [it for _, its in amb for it in its]
        return Detection([it.box for it in items], "opencv", "", 0, len(items), [], [it.piece for it in items], [a for a, _ in amb])

    monkeypatch.setattr(jobs, "detect_pieces", fake_detect)
    job = jobs.new_job()
    asyncio.run(jobs.run_photo_job(job, img, recognizer=rec))
    assert job.status == "done", job.error
    assert calls[0] == 10 + seg.SEG.max_tiebreaks                              # ten part crops + 3 whole crops
    assert sum(1 for p in job.pieces if p["needs_review"]) >= 4                # the two ambiguities beyond the limit stay "check"


# ---------------------------------------------------------------- merge / split
def test_merge_joins_two_masks_recognises_the_merged_crop_and_stores_the_correction(tmp_path, monkeypatch):
    from backend.core import studs
    monkeypatch.setattr(studs.CFG, "tilt_stretch", 1.0)
    monkeypatch.setattr(corrections.config, "CORRECTIONS_DIR", tmp_path / "corr")
    img, _ = scenes.two_same_colour_bricks()
    job, _ = make_job(tmp_path, img)
    n = len(job.items)
    assert n == 2
    asyncio.run(jobs.merge_pieces(job, img, 0, 1, fake_recognizer))
    assert len(job.items) == 1 and job.pieces[0]["index"] == 0 and job.pieces[0]["score"] == 0.5
    assert job.items[0].region.area > 0.9 * (sum(it.region.area for it in make_job(tmp_path, img)[0].items))
    assert (tmp_path / "crop_0.jpg").exists()
    assert job.pieces[0]["crop_url"].endswith("/crop/0?v=1")                     # a changed crop gets a new URL: no stale thumbnail
    rows = [json.loads(line) for line in (tmp_path / "corr" / "corrections.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "merge" and len(rows[0]["before"]) == 2 and len(rows[0]["after"]) == 1
    assert (tmp_path / "corr" / f"{rows[0]['photo']}.jpg").exists()           # the photo is kept with it


def test_split_cuts_the_mask_along_the_drawn_line_and_recognises_each_part(tmp_path, monkeypatch):
    from backend.core import studs
    monkeypatch.setattr(studs.CFG, "tilt_stretch", 1.0)
    monkeypatch.setattr(corrections.config, "CORRECTIONS_DIR", tmp_path / "corr")
    img, gt = scenes.rotated_plate()
    job, _ = make_job(tmp_path, img)
    assert len(job.items) == 1
    r = job.items[0].region
    cx, cy = r.centroid()
    asyncio.run(jobs.split_piece(job, img, 0, [cx, cy - 400, cx, cy + 400], fake_recognizer))
    assert len(job.items) == 2 and [p["index"] for p in job.pieces] == [0, 1]
    assert abs(job.items[0].region.area - job.items[1].region.area) < 0.3 * r.area
    rows = [json.loads(line) for line in (tmp_path / "corr" / "corrections.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "split" and len(rows[0]["before"]) == 1 and len(rows[0]["after"]) == 2


def test_split_by_a_line_that_misses_the_piece_is_refused(tmp_path, monkeypatch):
    from backend.core import studs
    monkeypatch.setattr(studs.CFG, "tilt_stretch", 1.0)
    img, _ = scenes.rotated_plate()
    job, _ = make_job(tmp_path, img)
    with pytest.raises(jobs.EditError):
        asyncio.run(jobs.split_piece(job, img, 0, [5, 5, 40, 5], fake_recognizer))
    with pytest.raises(jobs.EditError):
        asyncio.run(jobs.merge_pieces(job, img, 0, 0, fake_recognizer))


def test_boxes_without_a_mask_are_merged_and_split_as_rectangles(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections.config, "CORRECTIONS_DIR", tmp_path / "corr")
    img = np.full((200, 300, 3), 230, np.uint8)
    cv2.imwrite(str(tmp_path / "image.jpg"), img)
    job = jobs.Job(id="x", dir=tmp_path)
    job.items = [jobs.Item((10, 10, 100, 60)), jobs.Item((120, 10, 100, 60))]
    job.pieces = [{"index": 0}, {"index": 1}]
    asyncio.run(jobs.merge_pieces(job, img, 0, 1, fake_recognizer))
    assert len(job.items) == 1 and job.items[0].region.w >= 200
    asyncio.run(jobs.split_piece(job, img, 0, [115, 0, 115, 100], fake_recognizer))
    assert len(job.items) == 2


def test_endpoints_validate(tmp_path):
    from fastapi.testclient import TestClient
    from backend import main
    c = TestClient(main.app)
    assert c.post("/api/scan/nope/merge", json={"indices": [0, 1]}).status_code == 404
    assert c.post("/api/scan/nope/split", json={"index": 0, "line": [0, 0, 1, 1]}).status_code == 404
    job = jobs.new_job()
    job.status = "recognizing"
    assert c.post(f"/api/scan/{job.id}/merge", json={"indices": [0, 1]}).status_code == 409
    assert c.post(f"/api/scan/{job.id}/merge", json={"indices": [0]}).status_code == 422
    job.status = "done"
    cv2.imwrite(str(job.dir / "image.jpg"), np.full((50, 50, 3), 200, np.uint8))
    assert c.post(f"/api/scan/{job.id}/merge", json={"indices": [0, 1]}).status_code == 400
    jobs.JOBS.pop(job.id, None)


def test_the_polygon_stored_for_a_correction_is_in_photo_pixels():
    r = seg.Region(np.ones((40, 80), np.uint8), 100, 50)
    poly = corrections.polygon(r)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    assert (min(xs), min(ys), max(xs), max(ys)) == (100, 50, 179, 89)


# ---------------------------------------------------------------- crops hold one piece
def test_a_crop_has_its_neighbours_painted_out_but_never_the_piece_itself():
    """Brickognize judges the whole crop: a grey brick and a red one inside the margin of a lime plate made it answer 'grey modified plate'."""
    img = np.full((300, 600, 3), 70, np.uint8)
    cv2.rectangle(img, (100, 120), (400, 180), (40, 230, 190), -1)            # the plate
    cv2.rectangle(img, (110, 20), (190, 90), (180, 180, 180), -1)            # a grey piece just above it
    cv2.rectangle(img, (90, 185), (200, 260), (30, 30, 220), -1)             # a red brick just below it
    plate = seg.Region.from_full(rect(100, 120, 401, 181, img.shape))
    others = [seg.Region.from_full(rect(110, 20, 191, 91, img.shape)), seg.Region.from_full(rect(90, 185, 201, 261, img.shape))]
    out = jobs.isolate(img, plate, others + [plate])
    assert (out[130:170, 150:350] == img[130:170, 150:350]).all()              # the plate is untouched
    assert (np.abs(out[40:80, 130:170].astype(int) - 70).max() <= 3)           # the grey piece is gone (surface colour)
    assert (np.abs(out[200:250, 110:190].astype(int) - 70).max() <= 3)         # so is the red one
    assert (img[40:80, 130:170] == 180).all()                                  # the photo itself is not modified
    item = jobs.Item((100, 120, 300, 60), detect.Piece((100, 120, 300, 60), "t", region=plate, pitch=40.0))
    crop = jobs.crop_for(img, item, others)
    assert not ((crop[:, :, 2] > 200) & (crop[:, :, 1] < 80)).any()            # no red anywhere in the plate's crop


def rect(x0, y0, x1, y1, shape):
    m = np.zeros(shape[:2], np.uint8)
    m[y0:y1, x0:x1] = 1
    return m
