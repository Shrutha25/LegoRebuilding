"""In-memory scan jobs: detect -> crop -> recognize. A failure on one piece never fails the job."""

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend import config, corrections, timing
from backend.core import segment as seg
from backend.core import studs as studmod
from backend.core.gemini_detect import detect_pieces
from backend.core.recognize import encode_crop, recognize_crops, summarize

log = logging.getLogger("lego.jobs")


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
    detector: str | None = None      # which detector produced the boxes (opencv | gemini | hybrid)
    detector_note: str = ""          # why it differs from the configured mode, e.g. a Gemini failure
    items: list = field(default_factory=list)   # per piece: Item (box, mask piece), aligned with `pieces`; not sent to the browser
    rev: int = 0                     # bumped by every merge / split: crop URLs carry it so the browser never shows a stale crop
    task: object = None              # asyncio task, held so it is not garbage-collected

    def public(self):
        return {"job_id": self.id, "status": self.status, "total": self.total,
                "image_url": f"/api/scan/{self.id}/image", "width": self.width,
                "height": self.height, "pieces": self.pieces, "error": self.error,
                "detector": self.detector, "detector_note": self.detector_note}


JOBS: dict[str, Job] = {}
REFERENCES: dict[str, tuple] = {}      # id -> (BGR photo of the empty surface, created)


def new_reference(img):
    """Keep a photo of the empty surface (in memory) for later scans. Returns its id."""
    rid = uuid.uuid4().hex[:12]
    REFERENCES[rid] = (img, time.time())
    return rid


def get_reference(rid):
    item = REFERENCES.get(rid)
    return item[0] if item else None


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


def build_piece(index, box, crop_url, result, check=False):
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
        "needs_review": check or s is None or s["score"] < config.MIN_SCORE,
        "error": (result or {}).get("error"),
    }


@dataclass
class Item:
    """One detected piece on its way to the browser: its box and, for OpenCV detections, the mask piece behind it."""
    box: tuple
    piece: object = None             # detect.Piece with a segment.Region, or None (boxes drawn by another detector)

    @property
    def region(self):
        return self.piece.region if self.piece is not None else None


def isolate(img, region, others, grow=3):
    """The photo with every OTHER detected piece painted out (in the colour of the surface around `region`), so a crop of `region` holds
    one piece, not its neighbours. Brickognize looks at the whole crop: a grey brick and a red one inside the margin of a lime plate made
    it answer "grey modified plate" (score 0.44) where the plate alone is "Plate 2 x 4, Lime" (0.78)."""
    others = [o for o in others if o is not None and o is not region]
    if not others:
        return img
    h_img, w_img = img.shape[:2]
    reach = int(max(region.w, region.h) * 0.7) + 20                     # only what the crop's margin can reach
    x0, y0 = max(0, region.x - reach), max(0, region.y - reach)
    x1, y1 = min(w_img, region.x + region.w + reach), min(h_img, region.y + region.h + reach)
    paint = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for o in others:
        ox0, oy0, ox1, oy1 = max(o.x, x0), max(o.y, y0), min(o.x + o.w, x1), min(o.y + o.h, y1)
        if ox1 > ox0 and oy1 > oy0:
            paint[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] |= o.mask[oy0 - o.y:oy1 - o.y, ox0 - o.x:ox1 - o.x]
    if not paint.any():
        return img
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    paint = cv2.dilate(paint, k)
    own = np.zeros_like(paint)
    ry0, rx0 = region.y - y0, region.x - x0
    own[max(0, ry0):ry0 + region.h, max(0, rx0):rx0 + region.w] = region.mask[max(0, -ry0):, max(0, -rx0):][:y1 - y0 - max(0, ry0), :x1 - x0 - max(0, rx0)]
    paint &= 1 - cv2.dilate(own, k)                                     # never paint over the piece itself
    window = img[y0:y1, x0:x1]
    free = window[(paint == 0) & (cv2.dilate(own, np.ones((41, 41), np.uint8)) == 0)]
    fill = np.median(free, axis=0) if len(free) > 200 else np.array([70, 70, 70])
    out = img.copy()
    out[y0:y1, x0:x1][paint > 0] = fill.astype(np.uint8)
    return out


def crop_for(img, item, others=()):
    """The crop sent to recognition: the deskewed rotated rectangle plus half a stud pitch of margin when the piece has a mask (with the other
    detected pieces painted out of it), else the plain box (Gemini boxes). `others`: the other pieces' masks (segment.Region)."""
    if item.region is not None:
        stretch = float(studmod.CFG.tilt_stretch or 1.0) if studmod.CFG.enabled else 1.0
        return seg.crop_region(isolate(img, item.region, others), item.region, item.piece.pitch, stretch)
    x, y, w, h = item.box
    return img[y:y + h, x:x + w]


def crops_for(img, items, targets=None):
    """crop_for each of `items` (or only `targets`, a list of items), each isolated from all the others."""
    regions = [it.region for it in items]
    return [crop_for(img, it, regions) for it in (targets if targets is not None else items)]


def top_score(result):
    s = summarize(result) if result and "error" not in result else None
    return s["score"] if s else 0.0


def tiebreak(items, results, whole_results, ambiguities, shape=(1 << 30, 1 << 30)):
    """Section 6: for each ambiguous split, the split wins only if EVERY part's top score beats the whole crop's top score by
    config SEG.tiebreak_delta; otherwise the whole is kept. -> (items, results) after the decisions. Both scores are logged."""
    items, results = list(items), list(results)
    for amb, whole_result in zip(ambiguities, whole_results):
        idx = [i for i, it in enumerate(items) if it.region is not None and any(it.region is r for r in amb.parts)]
        if len(idx) < 2:
            continue
        whole, parts = top_score(whole_result), [top_score(results[i]) for i in idx]
        split = all(p >= whole + seg.SEG.tiebreak_delta for p in parts)
        log.info("tiebreak (%s): whole %.2f vs parts %s -> %s", amb.rule, whole, [round(p, 2) for p in parts], "split" if split else "whole")
        if split:
            for i in idx:
                items[i].piece.check = False
            continue
        box = seg.padded_box(amb.whole, 0.08, shape)
        from backend.core.detect import Piece
        merged = Item(box, Piece(box, amb.whole.source, region=amb.whole, pitch=amb.whole.pitch))
        for i in reversed(idx):
            del items[i], results[i]
        items.insert(idx[0], merged)
        results.insert(idx[0], whole_result)
    return items, results


async def run_photo_job(job: Job, img, recognizer=recognize_crops, detector=None, mode=None, reference=None):
    trace, token = timing.begin("scan_job", job=job.id)
    try:
        job.status = "detecting"
        job.height, job.width = img.shape[:2]
        cv2.imwrite(str(job.dir / "image.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if reference is not None:                                # kept with the scan so a bad empty photo can be inspected
            cv2.imwrite(str(job.dir / "reference.jpg"), reference, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with timing.stage("detect", mode=mode or config.DETECTOR):
            detection = await detect_pieces(img, mode or config.DETECTOR, detector, reference=reference)
        boxes = detection.boxes
        job.detector, job.detector_note = detection.mode, detection.note
        items = [Item(tuple(b), detection.piece_for(b)) for b in boxes]
        job.total = len(items)
        with timing.stage("crops"):
            crops = crops_for(img, items)
        tie = [a for a in detection.ambiguities[:seg.SEG.max_tiebreaks]
               if all(any(it.region is r for it in items) for r in a.parts)]
        job.status = "recognizing"
        with timing.stage("identify", crops=len(crops) + len(tie)):
            results = await recognizer(crops + [seg.crop_region(img, a.whole, a.whole.pitch) for a in tie])
        results, whole_results = list(results[:len(items)]), list(results[len(items):])
        if tie:
            with timing.stage("tiebreak"):
                items, results = tiebreak(items, results, whole_results, tie, img.shape)
            job.total = len(items)
        with timing.stage("save_crops"):
            for i, crop in enumerate(crops_for(img, items)):
                data = encode_crop(crop)
                if data:
                    (job.dir / f"crop_{i}.jpg").write_bytes(data)
        job.items = items
        job.pieces = [build_piece(i, it.box, f"/api/scan/{job.id}/crop/{i}", r, it.piece is not None and it.piece.check)
                      for i, (it, r) in enumerate(zip(items, results))]
        job.done = len(job.pieces)
        job.status = "done"
    except Exception as e:                      # surface as a job error, never crash the server
        job.status = "error"
        job.error = f"{type(e).__name__}: {e}"
    finally:
        timing.end(trace, token, status=job.status, pieces=len(job.pieces) if job.pieces else 0)


class EditError(ValueError):
    """A merge / split the piece cannot take (no mask, parts that do not separate...)."""


def _region_of(item, shape):
    """The mask of a piece; one the box-based detector found has none, so its box stands in as a rectangle (merge and split still work)."""
    if item.region is not None:
        return item.region
    x, y, w, h = item.box
    x0, y0 = max(0, x), max(0, y)
    r = seg.Region(np.ones((max(1, min(shape[0], y + h) - y0), max(1, min(shape[1], x + w) - x0)), np.uint8), x0, y0, "box")
    return r


def _pitch_of(items):
    return next((it.piece.pitch for it in items if it.piece is not None and it.piece.pitch), None)


def _make_item(region, shape, rule="manual"):
    from backend.core.detect import Piece
    region.rules = list(region.rules) + [rule]
    box = seg.padded_box(region, 0.08, shape)
    return Item(box, Piece(box, "merged", region=region, rules=list(region.rules), pitch=region.pitch))


async def _install(job, img, items, pieces, fresh):
    """Make `items` the job's pieces. `pieces` holds the summary dicts already built (None = new, built from `fresh[index]`, a
    recognition result). Crops are rewritten and every piece re-indexed."""
    out = []
    job.rev += 1
    crops = crops_for(img, items)
    for i, (it, p) in enumerate(zip(items, pieces)):
        data = encode_crop(crops[i])
        if data:
            (job.dir / f"crop_{i}.jpg").write_bytes(data)
        url = f"/api/scan/{job.id}/crop/{i}?v={job.rev}"
        out.append(build_piece(i, it.box, url, fresh.get(i), False) if p is None
                   else dict(p, index=i, crop_url=url, box=list(map(int, it.box))))
    job.items, job.pieces = items, out
    job.total = job.done = len(items)


async def merge_pieces(job, img, i, j, recognizer=recognize_crops):
    """Merge two pieces: their masks are joined and recognition runs again on the merged crop."""
    if i == j or not (0 <= i < len(job.items) and 0 <= j < len(job.items)):
        raise EditError("choose two different pieces")
    a, b = job.items[i], job.items[j]
    ra, rb = _region_of(a, img.shape), _region_of(b, img.shape)
    gap = int(0.5 * (_pitch_of([a, b]) or 16))
    u = seg.union_region(ra, rb, "merged", bridge=gap)
    u.pitch = ra.pitch or rb.pitch
    item = _make_item(u, img.shape, "manual:merge")
    result = (await recognizer(crops_for(img, [item] + [it for k, it in enumerate(job.items) if k not in (i, j)], [item])))[0]
    keep = [k for k in range(len(job.items)) if k not in (i, j)]
    pos = min(i, j)
    items = [job.items[k] for k in keep]
    pieces = [job.pieces[k] for k in keep]
    items.insert(pos, item)
    pieces.insert(pos, None)
    corrections.record(job.dir, "merge", [ra, rb], [u])
    await _install(job, img, items, pieces, {pos: result})
    return job


async def split_piece(job, img, index, line, recognizer=recognize_crops):
    """Split one piece along a line drawn across it (x0, y0, x1, y1 in photo pixels): its mask is cut there and each part is recognised."""
    if not 0 <= index < len(job.items):
        raise EditError("unknown piece")
    r = _region_of(job.items[index], img.shape)
    x0, y0, x1, y1 = (float(v) for v in line)
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5
    if n < 2:
        raise EditError("draw a line across the piece")
    ux, uy = dx / n, dy / n
    far = 2.0 * (r.w + r.h)
    cut = np.zeros(r.mask.shape, np.uint8)                          # the drawn line, extended across the whole piece
    cv2.line(cut, (int(x0 - ux * far - r.x), int(y0 - uy * far - r.y)), (int(x1 + ux * far - r.x), int(y1 + uy * far - r.y)), 1, 3)
    parts = seg.parts_from_cut(r.mask, cut, max(30, int(0.02 * r.area)))
    if len(parts) < 2:
        raise EditError("the line does not cut the piece in two")
    regions = [p for p in (r.derive(m) for m in parts) if p is not None]
    for p in regions:
        p.pitch = r.pitch
    new = [_make_item(p, img.shape, "manual:split") for p in regions]
    results = await recognizer(crops_for(img, new + [it for k, it in enumerate(job.items) if k != index], new))
    items = job.items[:index] + new + job.items[index + 1:]
    pieces = job.pieces[:index] + [None] * len(new) + job.pieces[index + 1:]
    corrections.record(job.dir, "split", [r], regions)
    await _install(job, img, items, pieces, {index + k: res for k, res in enumerate(results)})
    return job


def cleanup_expired(now=None):
    now = now or time.time()
    for rid, (_, created) in list(REFERENCES.items()):
        if now - created > config.REFERENCE_TTL_SECONDS:
            REFERENCES.pop(rid, None)
    for jid, job in list(JOBS.items()):
        if now - job.created > config.JOB_TTL_SECONDS:
            JOBS.pop(jid, None)
            shutil.rmtree(job.dir, ignore_errors=True)
    # orphaned dirs, e.g. left behind by a previous process
    if config.UPLOAD_DIR.exists():
        for d in config.UPLOAD_DIR.iterdir():
            if d.is_dir() and d.name not in JOBS and now - d.stat().st_mtime > config.JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
