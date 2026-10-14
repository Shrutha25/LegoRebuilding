"""Piece detection: find one bounding box per LEGO piece on a plain background."""

import cv2
import numpy as np


def find_pieces(img, min_area_frac=0.0005, pad_frac=0.08):
    """Return a list of (x, y, w, h) boxes, one per piece."""
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

    # Sort in reading order (rows, then left-to-right)
    row_h = np.median([b[3] for b in boxes]) if boxes else 1
    boxes.sort(key=lambda b: (int((b[1] + b[3] / 2) // row_h), b[0]))
    return boxes
