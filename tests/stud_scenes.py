"""Synthetic photos with studded pieces on a plain or embossed sheet, for the stud tests."""

import cv2
import numpy as np

PITCH = 56          # px between stud centres
STUD_R = 17         # px radius (pitch / radius = 3.3, as on a real piece)


def sheet(h=700, w=1000, base=235, emboss=0, seed=1):
    """A pale sheet; `emboss` > 0 adds a fine quilted pattern like a paper towel."""
    img = np.full((h, w, 3), base, np.float32)
    if emboss:
        ys, xs = np.mgrid[0:h, 0:w]
        img += emboss * (np.sin(xs / 4.0) * np.sin(ys / 4.0))[..., None]
    img += np.random.default_rng(seed).normal(0, 1.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def piece(img, x, y, cols, rows, colour, angle=0.0, shade=0.22):
    """Draw a cols x rows piece with its top-left stud at (x, y), turned by `angle` degrees. Studs are discs with a lit
    upper-left rim and a shaded lower-right rim. Returns the stud centres and the piece's polygon."""
    th = np.radians(angle)
    ax, ay = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    origin = np.array([x, y], float)
    centres = [origin + ax * PITCH * c + ay * PITCH * r for r in range(rows) for c in range(cols)]
    half = PITCH / 2
    poly = np.array([origin - ax * half - ay * half, origin + ax * (PITCH * (cols - 1) + half) - ay * half,
                     origin + ax * (PITCH * (cols - 1) + half) + ay * (PITCH * (rows - 1) + half),
                     origin - ax * half + ay * (PITCH * (rows - 1) + half)])
    # soft shadow of the piece
    shadow = np.zeros(img.shape[:2], np.float32)
    cv2.fillPoly(shadow, [np.rint(poly + [10, 12]).astype(np.int32)], 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 7)
    img[:] = (img * (1 - 0.25 * shadow[..., None])).astype(np.uint8)
    cv2.fillPoly(img, [np.rint(poly).astype(np.int32)], colour)
    col = np.array(colour, float)
    for c in centres:
        cx, cy = int(round(c[0])), int(round(c[1]))
        cv2.circle(img, (cx + 2, cy + 3), STUD_R + 1, tuple(int(v) for v in col * (1 - shade * 1.6)), -1)      # shaded side
        cv2.circle(img, (cx, cy), STUD_R, tuple(int(v) for v in np.clip(col * (1 + shade * 0.4), 0, 255)), -1)  # stud top
        cv2.circle(img, (cx - 1, cy - 1), STUD_R, tuple(int(v) for v in col * (1 - shade * 0.5)), 2)           # rim
        cv2.circle(img, (cx - 5, cy - 5), 3, tuple(int(v) for v in np.clip(col * 1.25 + 8, 0, 255)), -1)       # highlight
    k = 3
    img[:] = cv2.GaussianBlur(img, (k, k), 0.8)
    return [tuple(c) for c in centres], poly
