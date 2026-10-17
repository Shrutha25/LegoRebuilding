"""The detection test set: synthetic photos with hand-placed pieces, each with its ground-truth masks.

The real photos behind the failures this work fixes (a large green plate cut into six boxes, a quilted towel with six pieces) are
not in the repository, so every case here is rebuilt from what went wrong in them. Drop real photos next to these scenes by adding a
`<name>.jpg` and `<name>.json` (a list of polygons, one per piece) to tests/detection/real/: `all_scenes()` picks them up.

Each scene function returns (image, [mask, ...]) where a mask is a full-size uint8 0/1 array of one real piece.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from tests import stud_scenes as S

H, W = 1000, 1400
REAL = Path(__file__).parent / "real"

RED, BLUE, YELLOW, GREEN, ORANGE = (40, 40, 200), (200, 110, 30), (30, 210, 240), (70, 170, 60), (30, 130, 240)
PURPLE, PINK, GREY, BLACK, WHITE = (160, 50, 120), (170, 130, 240), (120, 118, 116), (40, 38, 38), (232, 232, 236)


def _mask(shape, polys):
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.rint(np.asarray(p)).astype(np.int32)], 1)
    return m


def _studded(img, x, y, cols, rows, colour, angle=0.0):
    """A studded piece; -> (ground-truth mask)."""
    _, poly = S.piece(img, x, y, cols, rows, colour, angle)
    return _mask(img.shape, [poly])


def _glare(img, mask, bands, strength=0.8, width=34):
    """Glossy plastic: pale, hardly coloured streaks of reflected light across a piece (they break the piece's colour in parts)."""
    streaks = np.zeros(mask.shape, np.float32)
    for (x0, y0, x1, y1) in bands:
        cv2.line(streaks, (x0, y0), (x1, y1), 1.0, width)
    streaks = cv2.GaussianBlur(streaks, (0, 0), 4) * mask
    img[:] = (img.astype(np.float32) * (1 - strength * streaks[..., None]) + 250 * strength * streaks[..., None]).astype(np.uint8)


def green_plate():
    """The failure: ONE large green plate (8 x 6 studs) among nine small pieces. Its glossy surface throws pale glare streaks across
    it, which cut its saturated colour into several parts: a colour-seed detector boxes each part. Expected 10."""
    img = S.sheet(H, W, base=205, emboss=9)
    img[:70] = img[-70:] = (35, 70, 115)                    # the towel does not fill the photo: wood shows at the top and bottom,
    img[:, :60] = img[:, -60:] = (35, 70, 115)              # so the border is not the surface and the colour method leads
    gt = [_studded(img, 180, 170, 8, 6, GREEN)]
    _glare(img, gt[0], [(190, 150, 300, 500), (330, 150, 440, 500), (470, 150, 580, 500)], strength=0.65, width=70)
    slots = [(740, 150, 2, 2, RED), (960, 150, 3, 1, BLUE), (1190, 150, 2, 2, YELLOW), (740, 400, 2, 3, ORANGE),
             (990, 400, 2, 2, PURPLE), (1220, 400, 1, 3, PINK), (170, 700, 2, 2, GREY), (420, 700, 3, 2, BLACK), (700, 700, 2, 2, WHITE)]
    for x, y, c, r, col in slots:
        gt.append(_studded(img, x, y, c, r, col))
    return img, gt


def two_same_colour_bricks():
    """Two red 2 x 4 bricks touching along their long sides: a dark seam is the only evidence. Expected 2."""
    img = S.sheet(H, W)
    a = _studded(img, 500, 330, 4, 2, RED)
    b = _studded(img, 500, 442, 4, 2, RED)
    seam_y = 330 + S.PITCH + S.PITCH // 2
    cv2.line(img, (500 - S.PITCH // 2, seam_y), (500 + 3 * S.PITCH + S.PITCH // 2, seam_y), tuple(int(c * 0.45) for c in RED), 3)
    img[:] = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img, [a, b]


def rotated_plate():
    """A large plate on its own, turned about 30 degrees. Expected 1."""
    img = S.sheet(H, W)
    return img, [_studded(img, 420, 260, 8, 6, (190, 120, 40), angle=30)]


def beam_and_round_brick():
    """A Technic beam with eight pin holes and a round brick with a cross-shaped axle hole: the holes show the sheet. Expected 2."""
    img = S.sheet(H, W)
    beam = np.zeros((H, W), np.uint8)
    cv2.line(beam, (250, 300), (250 + 8 * 50, 300), 1, 46)
    cv2.circle(beam, (250, 300), 23, 1, -1)
    cv2.circle(beam, (250 + 8 * 50, 300), 23, 1, -1)
    img[beam > 0] = (60, 60, 60)
    for k in range(9):
        cv2.circle(img, (250 + k * 50, 300), 13, tuple(int(v) for v in img[100, 100]), -1)
    cv2.circle(img, (260, 296), 22, (95, 95, 95), 2)
    disc = np.zeros((H, W), np.uint8)
    cv2.circle(disc, (900, 560), 110, 1, -1)
    img[disc > 0] = (40, 60, 190)
    cv2.circle(img, (900, 560), 70, (60, 80, 215), -1)                  # the stud ring
    cv2.circle(img, (900, 560), 70, (30, 40, 150), 3)
    sheet_px = tuple(int(v) for v in img[100, 100])
    cv2.rectangle(img, (892, 536), (908, 584), sheet_px, -1)            # the cross-shaped axle hole
    cv2.rectangle(img, (876, 552), (924, 568), sheet_px, -1)
    shadow = cv2.GaussianBlur(np.maximum(beam, disc).astype(np.float32), (0, 0), 6)
    out = img.astype(np.float32)
    shifted = np.roll(np.roll(shadow, 9, 0), 9, 1)
    out *= (1 - 0.18 * shifted * (1 - np.maximum(beam, disc)))[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), [beam, disc]


def two_tone_and_printed():
    """A blue 2 x 4 brick with a white printed star (no hue) and a two-tone red piece (dark and light of ONE hue). Expected 2."""
    img = S.sheet(H, W)
    a = _studded(img, 300, 330, 4, 2, BLUE)
    star = [(300 + 84 + 40 * np.cos(np.radians(90 + 36 * k)) * (1 if k % 2 == 0 else 0.45),
             330 + 28 + 40 * np.sin(np.radians(90 + 36 * k)) * (1 if k % 2 == 0 else 0.45)) for k in range(10)]
    cv2.fillPoly(img, [np.rint(np.array(star)).astype(np.int32)], (245, 245, 245))
    b = _studded(img, 900, 330, 4, 2, (30, 30, 150))
    lower = _mask(img.shape, [[(900 - 28, 330 + 28), (900 + 3 * 56 + 28, 330 + 28), (900 + 3 * 56 + 28, 330 + 84), (900 - 28, 330 + 84)]])
    img[lower > 0] = (120, 120, 235)                                       # same hue, much lighter
    return img, [a, b]


def lit_and_shaded_faces():
    """A brick photographed from the side: bright studded top and a much darker front face, one hue. Expected 1."""
    img = S.sheet(H, W)
    top = _studded(img, 500, 380, 4, 2, (60, 200, 240))
    front = _mask(img.shape, [[(500 - 28, 380 + 84), (500 + 3 * 56 + 28, 380 + 84), (500 + 3 * 56 + 28, 380 + 190), (500 - 28, 380 + 190)]])
    img[front > 0] = (20, 110, 150)
    img[:] = cv2.GaussianBlur(img, (3, 3), 0.6)
    return img, [np.maximum(top, front)]


def quilted_towel():
    """Six pieces (two of them pale) on an embossed, quilted paper towel. Expected 6."""
    img = S.sheet(H, W, base=205, emboss=9)
    specs = [(200, 200, 4, 2, RED, 0), (700, 180, 2, 2, BLUE, 15), (1100, 220, 3, 2, YELLOW, 0), (250, 600, 2, 4, (230, 230, 225), 0),
             (700, 620, 4, 2, GREEN, 40), (1100, 640, 2, 2, (200, 200, 200), 0)]
    gt = [_studded(img, x, y, c, r, col, a) for x, y, c, r, col, a in specs]
    return img, gt


SCENES = {
    "green_plate": (green_plate, 10),
    "two_same_colour_bricks": (two_same_colour_bricks, 2),
    "rotated_plate": (rotated_plate, 1),
    "beam_and_round_brick": (beam_and_round_brick, 2),
    "two_tone_and_printed": (two_tone_and_printed, 2),
    "lit_and_shaded_faces": (lit_and_shaded_faces, 1),
    "quilted_towel": (quilted_towel, 6),
}


def all_scenes():
    """name -> (image, [ground-truth masks]): the synthetic cases and any real photos added under tests/detection/real/."""
    out = {name: fn() for name, (fn, _) in SCENES.items()}
    if REAL.exists():
        for jpg in sorted(REAL.glob("*.jpg")):
            labels = jpg.with_suffix(".json")
            img = cv2.imread(str(jpg))
            if img is not None and labels.exists():
                out[jpg.stem] = (img, [_mask(img.shape, [p]) for p in json.loads(labels.read_text())])
    return out
