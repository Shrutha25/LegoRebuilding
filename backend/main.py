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
from backend.core.ideas import AnthropicLLM, IdeasError, generate_ideas
from backend.core.index import DataStore
from backend.core.rebrickable_api import RebrickableAPI

store = DataStore(config.DATA_DIR)
rb_api = RebrickableAPI(config.REBRICKABLE_API_KEY)
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


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
# localhost and private-network origins are always allowed (phone on the same Wi-Fi);
# CORS_ORIGINS adds explicit extra origins (e.g. a deployed frontend).
LAN_ORIGIN = (r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+"
              r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")
app.add_middleware(CORSMiddleware,
                   allow_origins=[o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o],
                   allow_origin_regex=LAN_ORIGIN,
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
    if ctype not in IMAGE_TYPES:
        if ctype.startswith("video/"):
            raise HTTPException(415, "Videos aren't supported. Please upload a photo (jpg, png or webp).")
        raise HTTPException(415, "Unsupported file type. Please upload a photo (jpg, png or webp).")
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
    name: str | None = None
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
    sets = await run_in_threadpool(store.index.match, mapped, body.match_color, body.max_parts,
                                  min_coverage=config.MIN_COVERAGE)
    return {"sets": sets, "unmapped": sorted(set(unmapped)),
            "mapped": {m["original"]: m["part_num"] for m in mapped if m["original"] != m["part_num"]}}


# ------------------------------------------------------------------ parts search
@app.get("/api/parts/search")
async def parts_search(q: str = ""):
    if not store.parts_loaded:
        raise HTTPException(503, store.error or "Parts data is still loading.")
    return {"results": await run_in_threadpool(store.search_parts, q)}


class ResolveIn(BaseModel):
    parts: list[str] = Field(min_length=1, max_length=500)
    colors: dict[str, str | None] = {}      # optional part id -> colour, for a colour-specific image


@app.post("/api/parts/resolve")
async def parts_resolve(body: ResolveIn):
    """Map recognized (Brickognize) ids to Rebrickable ids so the review screen can show
    which parts will not be used for matching, plus Rebrickable names/images."""
    if not store.parts_loaded:
        raise HTTPException(503, store.error or "Parts data is still loading.")
    names = store.part_names()
    out = {}
    for pid in dict.fromkeys(body.parts):
        rid, bad = await idmap.resolve(pid, store.known_ids, rb_api.bricklink_to_rebrickable)
        color = (body.colors.get(pid) or "").strip().lower() or None
        # image from Rebrickable's inventory_parts.csv (per part + colour, no API key needed)
        image = store.index.image_for(rid, color) if store.index else None
        out[pid] = {"part_num": rid, "unmapped": bad, "name": names.get(rid), "image": image}
    return {"results": out}


# ------------------------------------------------------------------ image proxy
# Brickognize serves thumbnails as application/octet-stream, which browsers block when used
# in <img> (ORB). Re-serve them with a proper image type. Strict allowlist: not an open proxy.
IMG_HOST = "storage.googleapis.com"
IMG_PREFIX = "/brickognize-static/"
IMG_TYPES = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_img_cache: dict[str, bytes] = {}


def _allowed_image(url: str):
    from urllib.parse import urlparse
    u = urlparse(url)
    ext = os.path.splitext(u.path)[1].lower()
    ok = (u.scheme == "https" and u.hostname == IMG_HOST and u.port is None
          and u.path.startswith(IMG_PREFIX) and ".." not in u.path and ext in IMG_TYPES)
    return IMG_TYPES[ext] if ok else None


@app.get("/api/img")
async def image_proxy(url: str):
    from fastapi import Response
    ctype = _allowed_image(url)
    if not ctype:
        raise HTTPException(400, "Image host not allowed.")
    data = _img_cache.get(url)
    if data is None:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(url)
            r.raise_for_status()
        except Exception:
            raise HTTPException(502, "Could not fetch image.")
        data = r.content
        if len(data) > 2_000_000:
            raise HTTPException(502, "Image too large.")
        if len(_img_cache) < 2000:
            _img_cache[url] = data
    return Response(data, media_type=ctype, headers={"Cache-Control": "public, max-age=86400"})


# ------------------------------------------------------------------ AI ideas
class IdeasIn(BaseModel):
    parts: list[PartIn] = Field(min_length=1, max_length=500)


_llm = None
_llm_slots = asyncio.Semaphore(3)         # cap concurrent paid calls


def get_llm():
    """Created lazily so the server (and tests) run without an API key."""
    global _llm
    if _llm is None:
        if not config.ANTHROPIC_API_KEY:
            raise IdeasError("AI ideas need an ANTHROPIC_API_KEY in the server's .env.", 503)
        _llm = AnthropicLLM(config.ANTHROPIC_API_KEY, config.LLM_MODEL, config.LLM_EFFORT)
    return _llm


@app.post("/api/ideas")
async def ideas(body: IdeasIn):
    names = store.part_names()
    parts = []
    for p in body.parts:
        rid = idmap.resolve_local(p.part_num, store.known_ids) if store.known_ids else None
        parts.append({"part_num": p.part_num, "color": p.color, "qty": p.qty,
                      "name": p.name or (names.get(rid) if rid else None)})
    try:
        async with _llm_slots:
            result = await generate_ideas(parts, get_llm())
    except IdeasError as e:
        raise HTTPException(e.status, str(e))
    for idea in result["ideas"]:                      # add images where the catalogue has them
        for piece in idea["pieces_used"]:
            rid = idmap.resolve_local(piece["part_num"], store.known_ids) if store.known_ids else None
            piece["image"] = store.index.image_for(rid, None) if (rid and store.index) else None
    return result
