# Snapshot of backend/core/detect.py before the mask rewrite (commit f45a18f), kept only so the before/after table can be reproduced.
# Do not edit. Not imported by the app.
"""Piece detection: find one bounding box per LEGO piece on a plain background.

`find_pieces` first runs the original method, which compares every pixel with the colour of the
photo's border (right when the plain sheet fills the photo). If that result is implausible, for
example one box covering most of the photo because wood or a table shows around the sheet, it
retries with a colour-based method that ignores brightness texture and uneven lighting.
"""

import cv2
import numpy as np

# A single "piece" bigger than this share of the photo means the background was misjudged.
GIANT_BOX_FRAC = 0.40


def _reading_order(boxes):
    """Sort in reading order (rows, then left-to-right)."""
    row_h = np.median([b[3] for b in boxes]) if boxes else 1
    boxes.sort(key=lambda b: (int((b[1] + b[3] / 2) // row_h), b[0]))
    return boxes


def find_pieces_on_plain_background(img, min_area_frac=0.0005, pad_frac=0.08):
    """Original method. Return a list of (x, y, w, h) boxes, one per piece.

    Assumes the border of the photo is plain surface. Finds transparent and white pieces too."""
    h_img, w_img = img.shape[:2]

    # Estimate background colour from the image border (works for white or any plain surface)
    border = np.concatenate([img[:10].reshape(-1, 3), img[-10:].reshape(-1, 3),
                             img[:, :10].reshape(-1, 3), img[:, -10:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    # Foreground = pixels that differ from background in colour OR are saturated
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((diff > 25) | (hsv[:, :, 1] > 40)).astype(np.uint8) * 255

    # Clean up: remove specks, then close gaps so a piece with highlights
    # (or a thin/transparent piece) stays one blob
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h_img * w_img

    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Skip very faint blobs (watermarks, shadows, smudges). Even transparent
        # or white pieces have strong edges/shading somewhere; these don't.
        if np.percentile(diff[y:y + h, x:x + w], 99) < 50:
            continue
        p = int(max(w, h) * pad_frac)
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(w_img, x + w + p), min(h_img, y + h + p)
        boxes.append((x0, y0, x1 - x0, y1 - y0))

    return _reading_order(boxes)


def find_pieces_by_colour(img, min_area_frac=0.0005, pad_frac=0.08, sat=110, dark=60, grow=0.02,
                          weak_sat=60, weak_dark=100):
    """Fallback for photos where the plain surface does not fill the frame or is unevenly lit.

    Each piece is found from a strong seed (strongly saturated or very dark pixels), then grown a
    bounded distance (`grow` = share of the photo's short side) to take in its paler edges. Seeds
    are used because the surface around a piece picks up a faint colour cast and long shadows,
    which would bridge neighbouring pieces if every weakly coloured pixel counted. This method also
    ignores brightness texture (an embossed paper towel). Anything touching the photo edge is
    ignored (the table or frame around the sheet; a piece cut off by the frame can't be identified
    anyway), and a candidate must be surrounded by light, plain surface. Pale pieces that are
    neither strongly coloured nor dark (white, transparent) are not found by this method; the
    original one handles those when the surface fills the photo."""
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1

    seeds = ((s_ch > sat) | (v_ch < dark)).astype(np.uint8) * 255
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((seeds > 0).astype(np.uint8), connectivity=8)
    for i in range(1, n):                                     # the table / frame around the sheet
        x, y, w, h, _ = stats[i]
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            seeds[labels == i] = 0

    r = max(3, int(min(h_img, w_img) * grow))
    near = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    weak = ((s_ch > weak_sat) | (v_ch < weak_dark)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(seeds, cv2.bitwise_and(weak, near))
    # Re-join fragments of one piece (e.g. the dark pockets between the studs of a grey brick) when
    # they are closer than the growth distance; pieces further apart than that stay separate.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))

    plain = (s_ch < 50) & (v_ch > 110)                         # light, unsaturated surface
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h_img * w_img
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            continue
        m = int(0.15 * max(w, h)) + 4                          # ring just outside the object
        X0, Y0, X1, Y1 = max(0, x - m), max(0, y - m), min(w_img, x + w + m), min(h_img, y + h + m)
        ring = np.ones((Y1 - Y0, X1 - X0), bool)
        ring[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
        if plain[Y0:Y1, X0:X1][ring].mean() < 0.5:
            continue
        p = int(max(w, h) * pad_frac)
        x0, y0 = max(0, x - p), max(0, y - p)
        x1, y1 = min(w_img, x + w + p), min(h_img, y + h + p)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    return _reading_order(boxes)


def _implausible(boxes, shape):
    """No boxes, or one 'piece' that is most of the photo: the background was misjudged."""
    if not boxes:
        return True
    h_img, w_img = shape[:2]
    return any(w * h > GIANT_BOX_FRAC * h_img * w_img for _, _, w, h in boxes)


def find_pieces(img, min_area_frac=0.0005, pad_frac=0.08):
    """Return a list of (x, y, w, h) boxes, one per piece."""
    boxes = find_pieces_on_plain_background(img, min_area_frac, pad_frac)
    if _implausible(boxes, img.shape):
        retry = find_pieces_by_colour(img, min_area_frac, pad_frac)
        if retry and not _implausible(retry, img.shape):
            return retry
    return boxes
