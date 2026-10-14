"""FastAPI app: /api/scan, /api/match, /api/parts/search, /api/health."""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend import config, jobs
from backend.core import idmap
from backend.core.index import DataStore
from backend.core.rebrickable_api import RebrickableAPI

store = DataStore(config.DATA_DIR)
rb_api = RebrickableAPI(config.REBRICKABLE_API_KEY)
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_TYPES = {"video/mp4", "video/quicktime"}


async def _cleanup_loop():
    while True:
        await asyncio.sleep(300)
        jobs.cleanup_expired()


@asynccontextmanager
async def lifespan(app):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    store.start()                                  # background: scans work while this loads
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="LEGO What Can I Build", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health():
    return {"status": "ok", "data_loaded": store.data_loaded, "parts_loaded": store.parts_loaded,
            "data_error": store.error, "load_seconds": store.load_seconds,
            "rebrickable_api": rb_api.enabled}


# ------------------------------------------------------------------ scan
@app.post("/api/scan")
async def scan(file: UploadFile = File(...)):
    ctype = (file.content_type or "").lower()
    if ctype in VIDEO_TYPES:
        raise HTTPException(501, "Video upload is not available yet; please upload a photo.")
    if ctype not in IMAGE_TYPES:
        raise HTTPException(415, "Upload a jpg, png or webp image.")
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 50 MB).")
    img = await run_in_threadpool(jobs.prepare_image, data)
    if img is None:
        raise HTTPException(400, "Could not read that image.")
    job = jobs.new_job()
    job.task = asyncio.create_task(jobs.run_photo_job(job, img))   # keep a reference
    return {"job_id": job.id}


def _job(job_id):
    job = jobs.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job.")
    return job


@app.get("/api/scan/{job_id}")
def scan_status(job_id: str):
    return _job(job_id).public()


@app.get("/api/scan/{job_id}/image")
def scan_image(job_id: str):
    path = _job(job_id).dir / "image.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/scan/{job_id}/crop/{index}")
def scan_crop(job_id: str, index: int):
    path = _job(job_id).dir / f"crop_{int(index)}.jpg"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/jpeg")


# ------------------------------------------------------------------ match
class PartIn(BaseModel):
    part_num: str
    color: str | None = None
    qty: int = Field(default=1, ge=1, le=10000)


class MatchIn(BaseModel):
    parts: list[PartIn] = Field(min_length=1, max_length=500)
    match_color: bool = False
    max_parts: int = Field(default=150, ge=1, le=20000)


async def map_parts(parts):
    """Brickognize ids -> Rebrickable ids. Returns (mapped parts, unmapped original ids)."""
    out, unmapped = [], []
    for p in parts:
        rid, bad = await idmap.resolve(p.part_num, store.known_ids, rb_api.bricklink_to_rebrickable)
        if bad:
            unmapped.append(p.part_num)
        out.append({"part_num": rid, "color": p.color, "qty": p.qty, "original": p.part_num})
    return out, unmapped


@app.post("/api/match")
async def match(body: MatchIn):
    if not store.data_loaded:
        raise HTTPException(503, store.error or "Set data is still loading; try again shortly.")
    mapped, unmapped = await map_parts(body.parts)
    sets = await run_in_threadpool(store.index.match, mapped, body.match_color, body.max_parts)
    return {"sets": sets, "unmapped": sorted(set(unmapped)),
            "mapped": {m["original"]: m["part_num"] for m in mapped if m["original"] != m["part_num"]}}


# ------------------------------------------------------------------ parts search
@app.get("/api/parts/search")
async def parts_search(q: str = ""):
    if not store.parts_loaded:
        raise HTTPException(503, store.error or "Parts data is still loading.")
    return {"results": await run_in_threadpool(store.search_parts, q)}
