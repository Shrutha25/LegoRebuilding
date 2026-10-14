import asyncio

import cv2
import numpy as np

from backend import config, jobs


def image():
    img = np.full((300, 300, 3), 255, np.uint8)
    img[50:120, 50:120] = (0, 0, 200)
    img[180:250, 160:240] = (200, 0, 0)
    return img


def run(recognizer, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    job = jobs.new_job()
    asyncio.run(jobs.run_photo_job(job, image(), recognizer))
    return job


def brick(score):
    return {"items": [{"id": "3001", "name": "Brick 2 x 4", "score": score}],
            "colors": [{"name": "Red"}]}


def test_partial_recognition_failure_does_not_fail_job(tmp_path, monkeypatch):
    async def rec(crops):
        return [brick(0.9), {"error": "timeout"}][:len(crops)]

    job = run(rec, tmp_path, monkeypatch)
    assert job.status == "done" and len(job.pieces) == 2
    ok, bad = job.pieces
    assert ok["part_id"] == "3001" and not ok["needs_review"]
    assert bad["part_id"] is None and bad["needs_review"] and bad["error"] == "timeout"
    assert (job.dir / "crop_0.jpg").exists() and (job.dir / "image.jpg").exists()


def test_low_score_needs_review(tmp_path, monkeypatch):
    async def rec(crops):
        return [brick(0.59), brick(0.6)]

    pieces = run(rec, tmp_path, monkeypatch).pieces
    assert [p["needs_review"] for p in pieces] == [True, False]


def test_recognizer_crash_becomes_job_error(tmp_path, monkeypatch):
    async def rec(crops):
        raise RuntimeError("boom")

    job = run(rec, tmp_path, monkeypatch)
    assert job.status == "error" and "boom" in job.error


def test_prepare_image_downscales_and_rejects_garbage():
    big = np.full((3000, 4000, 3), 255, np.uint8)
    out = jobs.prepare_image(cv2.imencode(".jpg", big)[1].tobytes())
    assert max(out.shape[:2]) == config.MAX_IMAGE_SIDE
    assert jobs.prepare_image(b"not an image") is None


def test_cleanup_removes_expired_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    job = jobs.new_job()
    jobs.cleanup_expired(now=job.created + config.JOB_TTL_SECONDS + 1)
    assert job.id not in jobs.JOBS and not job.dir.exists()
