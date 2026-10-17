"""Brickognize client (legacy endpoint). Kept in one module so it is easy to swap."""

import asyncio
import hashlib

import cv2
import httpx
import requests

from backend import timing

API_URL = "https://api.brickognize.com/predict/parts/"


def recognize_crop(crop, timeout=15):
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"error": "encode failed"}
    try:
        r = requests.post(
            API_URL,
            params={"predict_color": "true", "top_k_items": 3, "top_k_colors": 1,
                    "min_similarity_items": 0.1},
            files={"query_image": ("piece.jpg", buf.tobytes(), "image/jpeg")},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def summarize(result):
    """Pull the top part + colour out of a Brickognize response."""
    items = result.get("items") or []
    if not items:
        return None
    top = items[0]
    colors = top.get("colors") or result.get("colors") or []
    return {
        "part_id": top["id"],
        "name": top["name"],
        "score": round(top["score"], 3),
        "color": colors[0]["name"] if colors else None,
        "alternatives": [{"part_id": i["id"], "name": i["name"], "score": round(i["score"], 3)}
                         for i in items[1:]],
    }


# ---------------------------------------------------------------- async client (server)
MAX_CONCURRENT = 8
_cache = {}                     # sha256(crop bytes) -> Brickognize JSON
_sem = None


def encode_crop(crop):
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


async def recognize_bytes(data, client, url=API_URL, timeout=15, retries=1, backoff=1.0):
    """POST one crop. Never raises: returns the JSON, or {"error": ...}. Successes are cached."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    key = hashlib.sha256(data).hexdigest()
    if key in _cache:
        timing.count("brickognize_cache_hits")
        return _cache[key]
    err = "unknown"
    for attempt in range(retries + 1):
        try:
            async with _sem:
                with timing.stage("brickognize", attempt=attempt):
                    r = await client.post(
                        url,
                        params={"predict_color": "true", "top_k_items": 3, "top_k_colors": 1,
                                "min_similarity_items": 0.1},
                        files={"query_image": ("piece.jpg", data, "image/jpeg")},
                        timeout=timeout,
                    )
            timing.count("brickognize_requests")
            r.raise_for_status()
            result = r.json()
            _cache[key] = result
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(backoff * (attempt + 1))
    return {"error": err}


async def recognize_crops(crops, url=API_URL):
    """Recognize many crops concurrently (max 8 in flight). Returns one result per crop."""
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[
            recognize_bytes(b, client, url) if b else _failed("encode failed")
            for b in map(encode_crop, crops)])


async def _failed(msg):
    return {"error": msg}
