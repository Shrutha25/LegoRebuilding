"""Synthetic scenes for testing detection on uniform backgrounds of any colour.

The same nine pieces (including white, light grey, dark grey and black ones) with soft cast shadows,
mild noise and a lighting gradient, on a plain background of a chosen colour. A piece whose colour is
too close to the background cannot be told apart by any detector, so scenes only expect the pieces that
differ from the background (Lab distance > MIN_DELTA_E).

    python -m tests.background_sweep            # prints the failure map over a grid of backgrounds
"""

import cv2
import numpy as np

MIN_DELTA_E = 40
W, H = 1000, 700
PALETTE = {                               # BGR
    "red": (40, 40, 200), "blue": (200, 110, 20), "green": (60, 170, 60), "yellow": (40, 210, 240),
    "orange": (30, 130, 240), "white": (235, 235, 235), "light grey": (170, 170, 170),
    "dark grey": (75, 72, 70), "black": (25, 25, 25),
}
# 3 x 3 layout with comfortable gaps; (x0, y0, x1, y1)
POSITIONS = [(60, 50, 220, 150), (380, 60, 500, 170), (720, 50, 900, 140),
             (70, 280, 190, 400), (400, 290, 600, 380), (740, 280, 880, 420),
             (60, 520, 240, 610), (400, 510, 520, 640), (730, 520, 920, 620)]


def lab_of(bgr):
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def delta_e(a, b):
    return float(np.linalg.norm(lab_of(a) - lab_of(b)))


def scene(bg_bgr, seed=0, shadows=True, gradient=0.12):
    """-> (image, {name: centre}, expected names). Background colour bg_bgr (BGR)."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = bg_bgr
    ramp = np.linspace(1 - gradient, 1 + gradient, W, dtype=np.float32)[None, :, None]   # lighting gradient
    img *= ramp
    img += rng.normal(0, 3.5, (H, W, 1))                                                  # mild texture
    img = np.clip(img, 0, 255).astype(np.uint8)
    if shadows:
        sh = np.zeros((H, W), np.float32)
        for x0, y0, x1, y1 in POSITIONS:
            cv2.rectangle(sh, (x0 + 14, y0 + 16), (x1 + 14, y1 + 16), 1.0, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 9) * 0.42
        img = (img.astype(np.float32) * (1 - sh[:, :, None])).astype(np.uint8)
    centres, expected = {}, []
    for (name, colour), (x0, y0, x1, y1) in zip(PALETTE.items(), POSITIONS):
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        for xx in range(x0 + 18, x1 - 8, 34):                                              # studs: a little shading
            cv2.circle(img, (xx, y0 + 16), 9, tuple(int(c * 0.82) for c in colour), -1)
        centres[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
        if delta_e(colour, bg_bgr) > MIN_DELTA_E:
            expected.append(name)
    return img, centres, expected


def hsv_bg(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def score(boxes, centres, expected):
    held = {n: [i for i, (x, y, w, h) in enumerate(boxes) if x <= cx <= x + w and y <= cy <= y + h]
            for n, (cx, cy) in centres.items()}
    missed = [n for n in expected if not held[n]]
    merged = sum(1 for i in range(len(boxes)) if sum(1 for v in held.values() if i in v) > 1)
    extra = sum(1 for i in range(len(boxes)) if not any(i in v for v in held.values()))
    return missed, merged, extra


def grid():
    """Backgrounds: neutral from black to white, plus coloured ones at several lightness/saturation levels."""
    out = []
    for v in (10, 35, 70, 110, 150, 190, 225, 250):
        out.append((f"grey V={v}", hsv_bg(0, 0, v)))
    for hue, hname in ((5, "red-brown"), (15, "orange/wood"), (30, "yellow"), (60, "green"), (100, "blue"), (140, "purple")):
        for s in (70, 140, 220):
            for v in (60, 130, 210):
                out.append((f"{hname} S={s} V={v}", hsv_bg(hue, s, v)))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from backend.core import detect
    rows, bad = [], []
    for name, bg in grid():
        img, centres, expected = scene(bg)
        missed, merged, extra = score(detect.find_pieces(img), centres, expected)
        ok = not missed and not merged and not extra
        rows.append(ok)
        if not ok:
            bad.append((name, len(expected), missed, merged, extra))
    print(f"{sum(rows)} of {len(rows)} backgrounds perfect")
    for name, n, missed, merged, extra in bad:
        print(f"  {name:22s} expected {n}: missed {missed or '-'} merged {merged} extra {extra}")
