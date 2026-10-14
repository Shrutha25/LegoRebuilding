"""In-memory scan jobs: detect -> crop -> recognize. A failure on one piece never fails the job."""

import asyncio
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.core.detect import find_pieces
from backend.core.recognize import encode_crop, recognize_crops, summarize


@dataclass
class Job:
    id: str
    dir: Path
    status: str = "queued"          # queued | detecting | recognizing | done | error
    created: float = field(default_factory=time.time)
    total: int = 0
    done: int = 0
    width: int = 0
    height: int = 0
    pieces: list = field(default_factory=list)
    error: str | None = None
    task: object = None              # asyncio task, held so it is not garbage-collected

    def public(self):
        return {"job_id": self.id, "status": self.status, "total": self.total,
                "image_url": f"/api/scan/{self.id}/image", "width": self.width,
                "height": self.height, "pieces": self.pieces, "error": self.error}


JOBS: dict[str, Job] = {}


def new_job():
    jid = uuid.uuid4().hex[:12]
    d = config.UPLOAD_DIR / jid
    d.mkdir(parents=True, exist_ok=True)
    job = Job(id=jid, dir=d)
    JOBS[jid] = job
    return job


def prepare_image(data: bytes):
    """Decode and downscale very large photos. Returns a BGR image or None."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    side = max(img.shape[:2])
    if side > config.MAX_IMAGE_SIDE:
        s = config.MAX_IMAGE_SIDE / side
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def build_piece(index, box, crop_url, result):
    s = summarize(result) if result and "error" not in result else None
    if s:                       # attach Brickognize reference images (lets users compare visually)
        items = result["items"]
        s["image"] = items[0].get("img_url")
        for alt, item in zip(s["alternatives"], items[1:]):
            alt["image"] = item.get("img_url")
    return {
        "index": index, "box": list(map(int, box)), "crop_url": crop_url,
        "part_id": s["part_id"] if s else None,
        "name": s["name"] if s else None,
        "color": s["color"] if s else None,
        "score": s["score"] if s else None,
        "image": s["image"] if s else None,
        "alternatives": s["alternatives"] if s else [],
        "needs_review": s is None or s["score"] < config.MIN_SCORE,
        "error": (result or {}).get("error"),
    }


async def run_photo_job(job: Job, img, recognizer=recognize_crops):
    try:
        job.status = "detecting"
        job.height, job.width = img.shape[:2]
        cv2.imwrite(str(job.dir / "image.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        boxes = await asyncio.to_thread(find_pieces, img)
        job.total = len(boxes)
        crops = [img[y:y + h, x:x + w] for x, y, w, h in boxes]
        for i, crop in enumerate(crops):
            data = encode_crop(crop)
            if data:
                (job.dir / f"crop_{i}.jpg").write_bytes(data)

        job.status = "recognizing"
        results = await recognizer(crops)
        job.pieces = [build_piece(i, b, f"/api/scan/{job.id}/crop/{i}", r)
                      for i, (b, r) in enumerate(zip(boxes, results))]
        job.done = len(job.pieces)
        job.status = "done"
    except Exception as e:                      # surface as a job error, never crash the server
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"


def cleanup_expired(now=None):
    now = now or time.time()
    for jid, job in list(JOBS.items()):
        if now - job.created > config.JOB_TTL_SECONDS:
            JOBS.pop(jid, None)
            shutil.rmtree(job.dir, ignore_errors=True)
    # orphaned dirs, e.g. left behind by a previous process
    if config.UPLOAD_DIR.exists():
        for d in config.UPLOAD_DIR.iterdir():
            if d.is_dir() and d.name not in JOBS and now - d.stat().st_mtime > config.JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
