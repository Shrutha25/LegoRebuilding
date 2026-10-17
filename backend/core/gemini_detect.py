"""Find LEGO pieces with Gemini object detection, and merge that with the OpenCV detector.

Gemini returns one box per piece as `box_2d = [ymin, xmin, ymax, xmax]`, normalised to 0-1000 of the
image's height and width (note: y first). `detect_pieces` runs the configured mode:

    opencv  - the OpenCV detector only
    gemini  - Gemini only (falls back to OpenCV if the call fails)
    hybrid  - both, merged: every Gemini box is kept; an OpenCV box is added only if it is not
              already covered by a Gemini box; anything bigger than 25% of the photo is dropped.

A Gemini failure or timeout never fails the scan: it falls back to OpenCV and is logged.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import cv2

from backend.core import gemini_budget
from backend.core.detect import Analysis, _reading_order, analyse, draw_debug, find_pieces
from backend.core.studs import stud_info_for_box
from backend.core.gemini_budget import BudgetRefused

log = logging.getLogger("lego.detect")

MODES = ("opencv", "gemini", "hybrid")
BOX_SCALE = 1000
PAD_FRAC = 0.08              # padding added to Gemini boxes (OpenCV boxes are already padded the same)
MAX_BOX_FRAC = 0.25          # a "piece" larger than this share of the photo is wrong
IOU_THRESHOLD = 0.3          # an OpenCV box overlapping a Gemini box by at least this is a duplicate
CONTAINED_FRAC = 0.7         # ...as is one lying mostly inside a Gemini box (low IoU, same piece)
MIN_BOX_PX = 12              # ignore degenerate boxes
GEMINI_TIMEOUT = 20.0        # seconds for the whole Gemini detection, including a fallback model
SEND_MAX_SIDE = 1600

PROMPT = (
    "This photo shows individual LEGO pieces on a flat surface. Detect every individual LEGO piece and "
    "return one tight bounding box per piece. One box per piece: do not box shadows, creases or edges "
    "of the paper or table, the background, or groups of pieces; if two pieces touch, give each its own "
    "box. Include small, white, transparent and dark pieces. For each box give box_2d as "
    "[ymin, xmin, ymax, xmax] normalised to 0-1000, and a short label (colour and shape)."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "pieces": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                    "label": {"type": "string"},
                },
                "required": ["box_2d", "label"],
            },
        }
    },
    "required": ["pieces"],
}


class DetectError(Exception):
    """Gemini detection failed in a way the caller should log and work around."""


# ---------------------------------------------------------------- geometry
def boxes_from_gemini(pieces, width, height):
    """[{"box_2d": [ymin, xmin, ymax, xmax] (0-1000)}] -> pixel (x, y, w, h). Malformed or degenerate
    boxes are skipped; coordinates are clamped to the image; swapped min/max are fixed."""
    out = []
    for p in pieces or []:
        box = p.get("box_2d") if isinstance(p, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        ymin, ymax = sorted((ymin, ymax))
        xmin, xmax = sorted((xmin, xmax))
        x0 = round(max(0.0, min(xmin, BOX_SCALE)) / BOX_SCALE * width)
        x1 = round(max(0.0, min(xmax, BOX_SCALE)) / BOX_SCALE * width)
        y0 = round(max(0.0, min(ymin, BOX_SCALE)) / BOX_SCALE * height)
        y1 = round(max(0.0, min(ymax, BOX_SCALE)) / BOX_SCALE * height)
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            continue
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def pad_box(box, frac, shape):
    x, y, w, h = box
    p = int(max(w, h) * frac)
    h_img, w_img = shape[:2]
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(w_img, x + w + p), min(h_img, y + h + p)
    return (x0, y0, x1 - x0, y1 - y0)


def _intersection(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def iou(a, b):
    inter = _intersection(a, b)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def too_big(box, shape):
    return box[2] * box[3] > MAX_BOX_FRAC * shape[0] * shape[1]


def merge_boxes(gemini_boxes, opencv_boxes, shape, iou_threshold=IOU_THRESHOLD, contained=CONTAINED_FRAC,
                pad=PAD_FRAC):
    """Keep all Gemini boxes (padded), add OpenCV boxes only where Gemini found nothing, and drop
    anything larger than 25% of the photo. `contained=None` applies the IoU rule alone."""
    kept = [pad_box(b, pad, shape) for b in gemini_boxes]
    kept = [b for b in kept if not too_big(b, shape)]
    merged = list(kept)
    for ob in opencv_boxes:
        if too_big(ob, shape):
            continue
        dup = False
        for gb in kept:
            if iou(ob, gb) >= iou_threshold:
                dup = True
            elif contained is not None and _intersection(ob, gb) >= contained * ob[2] * ob[3]:
                dup = True
            if dup:
                break
        if not dup:
            merged.append(ob)
    return _reading_order(merged)


# ---------------------------------------------------------------- Gemini client
def _is_overload(exc):
    """503 / 504 style errors, where trying a lighter model is worth it."""
    code = getattr(exc, "code", None)
    status = (getattr(exc, "status", "") or "").upper()
    return code in (503, 504) or status in ("UNAVAILABLE", "DEADLINE_EXCEEDED")


class GeminiDetector:
    def __init__(self, api_key, model, fallback_model="gemini-3.1-flash-lite", timeout=GEMINI_TIMEOUT,
                 thinking_level="minimal", budget=None):
        from google import genai
        from google.genai import errors, types
        self._types, self._errors = types, errors
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        self.model, self.fallback_model, self.timeout, self.thinking_level = model, fallback_model, timeout, thinking_level
        self._level_ok = {}                                # model -> thinking level that worked
        self.budget = budget                               # every request is counted here (None: not counted)

    def _levels(self, model):
        """Thinking levels to try for `model`: the one that worked last time, else preferred -> low -> none.
        Not every model accepts every level (e.g. 3.8-flash rejects "minimal")."""
        if model in self._level_ok:
            return [self._level_ok[model]]
        levels = []
        for lv in (self.thinking_level, "low", None):
            if lv not in levels:
                levels.append(lv)
        return levels

    async def _call(self, model, jpeg):
        types = self._types
        err = None
        for level in self._levels(model):
            cfg = dict(response_mime_type="application/json", response_json_schema=SCHEMA, max_output_tokens=8192)
            if level:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            try:
                resp = await gemini_budget.generate(
                    self.client, self.budget, gemini_budget.DETECTION, model,
                    contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
                    config=types.GenerateContentConfig(**cfg))
            except self._errors.APIError as e:
                if getattr(e, "code", None) == 400 and "thinking" in (getattr(e, "message", "") or "").lower():
                    err = e                                     # this level isn't supported here: try the next
                    continue
                raise
            self._level_ok[model] = level
            return self._parse(resp)
        raise err

    @staticmethod
    def _parse(resp):
        if getattr(resp.prompt_feedback, "block_reason", None):
            raise DetectError("Gemini blocked the request")
        try:
            text = resp.text
        except Exception:
            text = None
        if not text:
            raise DetectError("Gemini returned no text")
        try:
            data = json.loads(text)
        except ValueError:
            raise DetectError("Gemini returned invalid JSON")
        return data.get("pieces") if isinstance(data, dict) else data

    async def _detect(self, jpeg):
        try:
            return await self._call(self.model, jpeg), self.model
        except self._errors.APIError as e:
            if _is_overload(e) and self.fallback_model and self.fallback_model != self.model:
                log.warning("gemini detection: %s overloaded (%s), retrying with %s", self.model, e.code, self.fallback_model)
                return await self._call(self.fallback_model, jpeg), self.fallback_model
            raise

    async def detect(self, img):
        """-> (pixel boxes, model used). Raises DetectError on any failure or after `timeout` seconds."""
        h, w = img.shape[:2]
        side = max(h, w)
        if side > SEND_MAX_SIDE:
            s = SEND_MAX_SIDE / side
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise DetectError("could not encode image")
        try:
            pieces, model = await asyncio.wait_for(self._detect(buf.tobytes()), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise DetectError(f"timed out after {self.timeout:.0f}s")
        except (DetectError, BudgetRefused):
            raise
        except Exception as e:
            raise DetectError(f"{type(e).__name__}: {str(e)[:120]}")
        return boxes_from_gemini(pieces, w, h), model


# ---------------------------------------------------------------- orchestration
@dataclass
class Detection:
    boxes: list
    mode: str                  # what actually produced the boxes: opencv | gemini | hybrid
    note: str = ""             # why it differs from the configured mode (logged and kept for debugging)
    gemini_count: int = 0
    opencv_count: int = 0
    stud_info: list = field(default_factory=list)   # one StudInfo per box (empty when the OpenCV stage did not run)
    pieces: list = field(default_factory=list)      # detect.Piece (mask, rules, check) of the OpenCV boxes among `boxes`
    ambiguities: list = field(default_factory=list) # segment.Ambiguity: splits for the Brickognize tie-break

    def piece_for(self, box):
        """The OpenCV piece (with its mask) behind a box, or None for a box another detector drew."""
        return next((p for p in self.pieces if tuple(p.box) == tuple(box)), None)


def _run_opencv(opencv, img):
    """`opencv` is called with the photo; the default is `analyse` (boxes plus stud details). A plain
    `find_pieces`-style function returning boxes works too. -> (boxes, Analysis or None)"""
    res = analyse(img) if opencv is None or opencv is find_pieces else opencv(img)
    return (res.boxes, res) if isinstance(res, Analysis) else (list(res), None)


def _mask_info(boxes, analysis):
    """-> (pieces, ambiguities) of the OpenCV analysis that survive as `boxes`."""
    if analysis is None:
        return [], []
    keep = {tuple(b) for b in boxes}
    pieces = [p for p in analysis.pieces if tuple(p.box) in keep]
    ids = {id(p.region) for p in pieces}
    amb = [a for a in analysis.ambiguities if all(id(r) in ids for r in a.parts)]
    return pieces, amb


def _stud_info(boxes, analysis):
    if analysis is None:
        return []
    try:
        return [stud_info_for_box(b, analysis.groups) for b in boxes]
    except Exception:                                               # noqa: BLE001  studs are a signal only
        return []


def _debug_overlay(img, analysis, boxes, labels, gemini_boxes):
    """Hybrid overlay with the Gemini boxes, when DETECT_DEBUG_DIR is set (OpenCV-only overlays come from analyse)."""
    import os
    import time
    from pathlib import Path
    target = os.environ.get("DETECT_DEBUG_DIR")
    if not target or analysis is None:
        return
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
        draw_debug(img, analysis, Path(target) / f"detect_{time.time_ns()}_hybrid.jpg", boxes=boxes, labels=labels)
    except Exception as e:                                          # noqa: BLE001
        log.warning("debug overlay failed: %s", e)


def analysis_source(analysis, box):
    """Source label of an OpenCV box in a hybrid result."""
    if analysis is not None:
        for p in analysis.pieces:
            if p.box == box:
                return p.source
    return "merged"


async def _opencv_only(img, shape, opencv, mode, note):
    log.info("%s", note)
    raw, analysis = await asyncio.to_thread(_run_opencv, opencv, img)
    boxes = [b for b in raw if not too_big(b, shape)]
    return Detection(boxes, "opencv", note, 0, len(boxes), _stud_info(boxes, analysis), *_mask_info(boxes, analysis))


async def _reference_warning(img, boxes, min_missing=2):
    """Cross-check a reference scan: pieces the ordinary detector sees but the comparison did not are pieces that were
    already in the "empty" photo (or the camera moved). Say so, because those pieces are silently missing from the scan."""
    try:
        others = (await asyncio.to_thread(analyse, img)).boxes
    except Exception:                                               # noqa: BLE001  a diagnostic must never fail a scan
        return ""
    missing = [o for o in others if not any(
        max(0, min(o[0] + o[2], b[0] + b[2]) - max(o[0], b[0])) * max(0, min(o[1] + o[3], b[1] + b[3]) - max(o[1], b[1]))
        >= 0.3 * o[2] * o[3] for b in boxes)]
    if len(missing) < min_missing:
        return ""
    log.warning("reference scan: %d pieces visible without the empty photo are missing from the result", len(missing))
    return (f"; WARNING: {len(missing)} more pieces are visible without the empty-surface photo, so it probably still had "
            "pieces on it or the camera moved. Retake it with the surface completely empty")


async def detect_pieces(img, mode="hybrid", detector=None, opencv=None, reference=None):
    """Run the configured detector(s). Never raises for a Gemini problem: falls back to OpenCV.

    With `reference` (a photo of the same empty surface) the pieces are what differs from it, OpenCV only: no Gemini
    request is made and no colour or stud steps run."""
    if mode not in MODES:
        mode = "hybrid"
    shape = img.shape
    if reference is not None:
        analysis = await asyncio.to_thread(analyse, img, reference=reference)
        if not analysis.matched:                                    # the empty photo is not of this scene: do not trust it
            note = (f"WARNING: the empty-surface photo does not match this photo (surface pattern correlation {analysis.match:.2f}): "
                    "the camera moved or zoomed, or the light changed. It was not used. Keep the camera fixed for both photos")
            log.warning("detection: %s", note)
            det = await _opencv_only(img, shape, opencv, mode, note)
            return det
        boxes = [b for b in analysis.boxes if not too_big(b, shape)]
        note = f"empty-surface photo used (aligned by {analysis.alignment}; {analysis.changed:.0%} of the photo changed)"
        note += await _reference_warning(img, boxes)
        _debug_overlay(img, analysis, boxes, ["reference"] * len(boxes), None)
        return Detection(boxes, "reference", note, 0, len(boxes))
    if mode == "opencv" or detector is None:
        note = "" if mode == "opencv" else "gemini not configured; used opencv"
        if note:
            log.warning("detection: %s", note)
        raw, analysis = await asyncio.to_thread(_run_opencv, opencv, img)
        boxes = [b for b in raw if not too_big(b, shape)]
        return Detection(boxes, "opencv", note, 0, len(boxes), _stud_info(boxes, analysis), *_mask_info(boxes, analysis))

    budget = getattr(detector, "budget", None)
    if budget is not None:                                           # cheap early gate; the real count happens per request
        verdict = await asyncio.to_thread(budget.check, gemini_budget.DETECTION, detector.model)
        if not verdict:
            return await _opencv_only(img, shape, opencv, mode, f"gemini skipped: {verdict.reason}")

    cv_task = asyncio.create_task(asyncio.to_thread(_run_opencv, opencv, img)) if mode == "hybrid" else None
    try:
        g_boxes, model = await detector.detect(img)
    except BudgetRefused as e:                                       # refused mid-chain (fallback model, retry)
        cv_boxes, analysis = await (cv_task if cv_task else asyncio.to_thread(_run_opencv, opencv, img))
        boxes = [b for b in cv_boxes if not too_big(b, shape)]
        log.info("gemini skipped: %s", e.reason)
        return Detection(boxes, "opencv", f"gemini skipped: {e.reason}", 0, len(boxes), _stud_info(boxes, analysis))
    except Exception as e:                                           # DetectError or anything unexpected
        log.warning("gemini detection failed (%s); falling back to opencv", e)
        cv_boxes, analysis = await (cv_task if cv_task else asyncio.to_thread(_run_opencv, opencv, img))
        boxes = [b for b in cv_boxes if not too_big(b, shape)]
        return Detection(boxes, "opencv", f"gemini failed: {e}", 0, len(boxes), _stud_info(boxes, analysis))

    if mode == "gemini":
        boxes = _reading_order([b for b in (pad_box(b, PAD_FRAC, shape) for b in g_boxes) if not too_big(b, shape)])
        return Detection(boxes, "gemini", "", len(g_boxes), 0)

    cv_boxes, analysis = await cv_task
    boxes = merge_boxes(g_boxes, cv_boxes, shape)
    gem = [pad_box(b, PAD_FRAC, shape) for b in g_boxes]
    labels = ["gemini" if b in gem else analysis_source(analysis, b) for b in boxes]
    _debug_overlay(img, analysis, boxes, labels, gem)
    return Detection(boxes, "hybrid", "", len(g_boxes), len(cv_boxes), _stud_info(boxes, analysis), *_mask_info(boxes, analysis))
