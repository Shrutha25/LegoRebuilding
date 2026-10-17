"""Brick-vs-plate second opinion from cast shadows.

A brick is 9.6 mm tall and a plate 3.2 mm, so under one light a brick casts a longer shadow. From
straight above, the picture of the piece barely shows its height, so an image-recognition model
confuses the two. This module measures how far each piece's shadow reaches and, converted to
millimetres with the piece's own width as the ruler, compares it with the recognised kind.

It is a *second opinion*, never an answer: it only produces a verdict when the photo has a clear
shadow direction, and the UI lets the user flip brick/plate either way.

Calibration: the thresholds below come from ONE photo taken in low light from the side (4 bricks
at 12.7-19.3 mm, 4 plates at 6.3-8.8 mm). Shadow length depends on how low the light is: in a second
photo under soft overhead light the same bricks cast only 4.0-6.6 mm and the plates 2.5-3.8 mm, so the
absolute thresholds would have called real bricks plates. The module therefore abstains unless the
photo contains at least one clearly long shadow (MIN_LONGEST_MM), which shows the light is low enough
for the thresholds to mean something. The band between the thresholds is deliberately "unclear".
"""

import re

import cv2
import numpy as np

BRICK_MM = 11.5          # shadow at least this long (measured in mm at the piece's own scale): brick
PLATE_MM = 10.0          # at most this long: plate. Between the two: unclear
STUD_PITCH_MM = 8.0
STUD_CLEARANCE_MM = 0.2
MIN_PIECES = 2           # light direction is estimated across pieces; one piece is not enough
MIN_LONGEST_MM = 12.0    # if even the longest shadow is shorter, the light is too diffuse: no opinion
DIRECTION_MARGIN = 1.3   # the best direction must beat the runner-up by this factor
MIN_MASK_FRAC = 0.15     # pale pieces (white, transparent) leave too little mask to measure

_NAME = re.compile(r"^(brick|plate)\s+(\d+)\s*x\s*(\d+)$", re.I)


def parse_kind_dims(name):
    """'Brick 2 x 4' -> ('brick', 2, 4). None for anything else (round, modified, sloped...)."""
    m = _NAME.match((name or "").strip())
    if not m:
        return None
    a, b = sorted((int(m.group(2)), int(m.group(3))))
    return m.group(1).lower(), a, b


def counterpart_name(name):
    """'Brick 2 x 4' <-> 'Plate 2 x 4' (same footprint), or None."""
    parsed = parse_kind_dims(name)
    if not parsed:
        return None
    kind, _, _ = parsed
    other = "Plate" if kind == "brick" else "Brick"
    return re.sub(r"^(brick|plate)", other, (name or "").strip(), flags=re.I)


def _piece_mask(hsv, box):
    """Mask of the piece itself inside its box: strongly coloured or very dark pixels."""
    h_img, w_img = hsv.shape[:2]
    x, y, w, h = box
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = np.zeros((h_img, w_img), np.uint8)
    mask[y:y + h, x:x + w] = (((s[y:y + h, x:x + w] > 110) | (v[y:y + h, x:x + w] < 60))).astype(np.uint8)
    k = max(5, int(min(h_img, w_img) * 0.0167)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    mask = (lab == i).astype(np.uint8)
    if mask.sum() < MIN_MASK_FRAC * w * h:
        return None
    return cv2.dilate(mask, np.ones((max(3, k // 3) | 1,) * 2, np.uint8))


def _runs_left(dark, piece, limit):
    """For each row crossing the piece: how far the dark region extends to the left of the piece's
    left edge, tolerating small gaps (paper texture). `dark` and `piece` are boolean 2-D arrays."""
    out = []
    kern = np.ones(5)
    for r in np.nonzero(piece.any(axis=1))[0]:
        c0 = int(np.argmax(piece[r]))
        seg = dark[r, max(0, c0 - limit):c0][::-1]
        if seg.size == 0:
            out.append(0)
            continue
        solid = np.convolve(seg.astype(float), kern, mode="same") >= 3
        out.append(int(np.argmin(solid)) if not solid.all() else len(solid))
    return np.array(out, float)


_ROT = {"left": 0, "up": 1, "right": 2, "down": 3}      # np.rot90 turns the scan direction into "left"


def _measure(hsv, v_smooth, plain, box, dims, pitch_px=None):
    """Shadow p75 length in mm for each of 4 directions, or None if the piece can't be measured.
    `pitch_px` (the stud pitch measured by stud detection) is the ruler when known; otherwise the piece's own width."""
    h_img, w_img = hsv.shape[:2]
    piece = _piece_mask(hsv, box)
    if piece is None:
        return None
    cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (rw, rh), _ = cv2.minAreaRect(max(cs, key=cv2.contourArea))
    short_px = min(rw, rh)
    width_mm = min(dims) * STUD_PITCH_MM - STUD_CLEARANCE_MM
    if short_px < 20 or width_mm <= 0:
        return None
    px_per_mm = pitch_px / STUD_PITCH_MM if pitch_px and pitch_px >= 5 else short_px / width_mm

    x, y, w, h = box
    m = int(0.3 * max(w, h))
    X0, Y0, X1, Y1 = max(0, x - m), max(0, y - m), min(w_img, x + w + m), min(h_img, y + h + m)
    ring = plain[Y0:Y1, X0:X1] & (piece[Y0:Y1, X0:X1] == 0)
    if ring.sum() < 200:
        return None
    local_v = float(np.median(hsv[Y0:Y1, X0:X1, 2][ring]))          # brightness of the paper here
    dark = (v_smooth[Y0:Y1, X0:X1] < 0.86 * local_v) & (piece[Y0:Y1, X0:X1] == 0)
    pc = piece[Y0:Y1, X0:X1].astype(bool)
    limit = int(0.6 * max(w, h))
    result = {}
    for name, k in _ROT.items():
        runs = _runs_left(np.rot90(dark, k), np.rot90(pc, k), limit)
        result[name] = float(np.percentile(runs, 75)) / px_per_mm if runs.size else 0.0
    return result


def verdict_for(mm):
    if mm >= BRICK_MM:
        return "brick"
    if mm <= PLATE_MM:
        return "plate"
    return "unclear"


def analyse(img, boxes, names, stud_info=None):
    """img: BGR photo; boxes: (x, y, w, h) per piece; names: recognised part name per piece (or None).

    stud_info (optional, one StudInfo per box, from detection) supplies each piece's measured stud pitch, which
    is then the ruler instead of re-measuring the piece's width.

    Returns one entry per piece: None when no opinion is possible, else
    {"kind", "shadow_mm", "verdict", "conflict", "reason"}. Never raises."""
    results = [None] * len(boxes)
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_smooth = cv2.GaussianBlur(hsv[:, :, 2].astype(np.float32), (0, 0), 2)
        plain = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 110)
        measured = {}
        for i, (box, name) in enumerate(zip(boxes, names)):
            parsed = parse_kind_dims(name)
            if not parsed:
                continue
            pitch = getattr(stud_info[i], "pitch_px", None) if stud_info and i < len(stud_info) else None
            m = _measure(hsv, v_smooth, plain, box, parsed[1:], pitch)
            if m:
                measured[i] = (parsed[0], m)
        if len(measured) < MIN_PIECES:
            return results
        # Light direction: the direction in which shadows reach furthest, for the photo as a whole
        med = {d: float(np.median([m[d] for _, m in measured.values()])) for d in _ROT}
        ranked = sorted(med, key=med.get, reverse=True)
        best, second = ranked[0], ranked[1]
        if med[best] <= 0 or med[best] < DIRECTION_MARGIN * max(med[second], 1e-6):
            return results                                           # no clear shadow direction
        if max(m[best] for _, m in measured.values()) < MIN_LONGEST_MM:
            return results                                           # diffuse light: shadows say nothing
        for i, (kind, m) in measured.items():
            mm = m[best]
            verdict = verdict_for(mm)
            conflict = verdict != "unclear" and verdict != kind
            results[i] = {"kind": kind, "shadow_mm": round(mm, 1), "verdict": verdict, "conflict": conflict,
                          "reason": f"The shadow looks like a {verdict}, not a {kind}." if conflict else None}
    except Exception:
        return [None] * len(boxes)
    return results
