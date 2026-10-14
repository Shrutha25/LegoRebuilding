from pathlib import Path

import cv2

from backend.core.detect import find_pieces

SAMPLE = Path(__file__).resolve().parent.parent / "lego_pieces.jpg"


def test_sample_image_has_42_pieces():
    img = cv2.imread(str(SAMPLE))
    assert img is not None
    boxes = find_pieces(img)
    assert len(boxes) == 42


# ------------------------------------------------------------------ photos where the surface does not fill the frame
import numpy as np  # noqa: E402

from backend.core import detect  # noqa: E402


def towel_on_table():
    """Bricks on an embossed paper towel with a lighting gradient, on a wooden table; the towel
    does not reach the photo edges (the failure that a user's real photo showed)."""
    h, w = 800, 450
    rng = np.random.default_rng(3)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (35, 70, 115)                                                    # wood (BGR)
    img = np.clip(img + rng.integers(-18, 18, (h, w, 1)), 0, 255).astype(np.uint8)
    towel = np.zeros((h, w), np.uint8)
    cv2.fillPoly(towel, [np.array([[40, 120], [450, 90], [450, 700], [0, 740], [0, 160]])], 255)
    xs = np.linspace(0, 1, w)[None, :, None]
    base = (150 + 45 * xs).repeat(h, 0)                                        # shaded left -> lit right
    ys, xx = np.mgrid[0:h, 0:w]
    emboss = 9 * (np.sin(xx / 3.0) * np.sin(ys / 3.0))                          # fine embossed pattern
    cloth = np.clip(base + emboss[..., None], 0, 255).repeat(3, 2) * np.array([1.0, 0.99, 0.97])
    img[towel > 0] = cloth[towel > 0].astype(np.uint8)
    bricks = {"blue": ((170, 100, 20), (100, 470, 240, 540)), "lime": ((40, 200, 140), (300, 190, 380, 290)),
              "yellow": ((110, 240, 235), (310, 380, 380, 450)), "grey": ((62, 58, 58), (70, 190, 150, 400))}
    centres = {}
    for name, (colour, (x0, y0, x1, y1)) in bricks.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        if name == "grey":                                                # studs: real dark bricks have dark shadows between them
            for yy in range(y0 + 22, y1, 45):
                cv2.circle(img, ((x0 + x1) // 2, yy), 12, (28, 26, 26), -1)
        centres[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
    return img, centres


def count_per_brick(boxes, centres):
    return {n: sum(1 for x, y, w, h in boxes if x <= cx <= x + w and y <= cy <= y + h) for n, (cx, cy) in centres.items()}


def test_original_method_is_fooled_when_the_surface_does_not_fill_the_photo():
    img, _ = towel_on_table()
    boxes = detect.find_pieces_on_plain_background(img)
    assert detect._implausible(boxes, img.shape)            # this is what the fallback exists for


def test_fallback_finds_each_brick_once_on_a_towel_with_uneven_light():
    img, centres = towel_on_table()
    boxes = detect.find_pieces(img)
    assert count_per_brick(boxes, centres) == {"blue": 1, "lime": 1, "yellow": 1, "grey": 1}
    assert len(boxes) == 4                                   # no patches of wood or towel texture
    assert max(w * h for _, _, w, h in boxes) < 0.1 * img.shape[0] * img.shape[1]


def test_sample_uses_the_original_method_unchanged():
    img = cv2.imread(str(SAMPLE))
    assert detect.find_pieces(img) == detect.find_pieces_on_plain_background(img)
    assert not detect._implausible(detect.find_pieces_on_plain_background(img), img.shape)


def test_fallback_is_not_run_when_the_first_result_is_plausible(monkeypatch):
    img = cv2.imread(str(SAMPLE))
    monkeypatch.setattr(detect, "find_pieces_by_colour", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert len(detect.find_pieces(img)) == 42


def test_fallback_ignores_things_touching_the_photo_edge():
    img = np.full((400, 400, 3), 225, np.uint8)
    cv2.rectangle(img, (150, 150), (230, 230), (170, 100, 20), -1)      # a piece on the sheet
    cv2.rectangle(img, (0, 0), (60, 90), (170, 100, 20), -1)            # something cut off by the frame
    boxes = detect.find_pieces_by_colour(img)
    assert len(boxes) == 1 and boxes[0][0] > 100


def test_blank_photo_gives_no_pieces():
    assert detect.find_pieces(np.full((300, 300, 3), 230, np.uint8)) == []


def test_implausible_helper():
    assert detect._implausible([], (100, 100, 3))
    assert detect._implausible([(0, 0, 90, 90)], (100, 100, 3))
    assert not detect._implausible([(10, 10, 20, 20)], (100, 100, 3))


def crowded_scene():
    """Photo whose sheet fills the frame, with strong side lighting: each coloured piece casts a long
    dark shadow and tints the paper around it, which used to bridge neighbouring pieces into one box.
    Returns (image, centres)."""
    h, w = 480, 800
    img = np.full((h, w, 3), 205, np.uint8)
    pieces = {"blue": ((170, 100, 20), (360, 120, 470, 200)), "brown": ((20, 40, 110), (230, 60, 300, 130)),
              "yellow": ((110, 240, 235), (80, 300, 190, 400)), "grey": ((70, 66, 66), (260, 300, 560, 390))}
    for colour, (x0, y0, x1, y1) in pieces.values():
        # a halo of the piece's colour on the paper and a long dark shadow towards the next piece
        halo = img.copy()
        cv2.rectangle(halo, (x0 - 40, y0 - 40), (x1 + 40, y1 + 40), colour, -1)
        halo = cv2.GaussianBlur(halo, (0, 0), 25)
        img = cv2.addWeighted(img, 0.82, halo, 0.18, 0)
        cv2.fillPoly(img, [np.array([[x1, y0 + 8], [x1 + 90, y0 + 40], [x1 + 90, y1 + 40], [x1, y1]])], (120, 120, 118))
        img = cv2.GaussianBlur(img, (0, 0), 3)
    centres = {}
    for name, (colour, (x0, y0, x1, y1)) in pieces.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        for xx in range(x0 + 18, x1, 40):
            cv2.circle(img, (xx, (y0 + y1) // 2), 9, tuple(int(c * 0.55) for c in colour), -1)
        centres[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
    return img, centres


def test_halos_and_shadows_do_not_merge_neighbouring_pieces():
    img, centres = crowded_scene()
    boxes = detect.find_pieces_by_colour(img)
    assert count_per_brick(boxes, centres) == {n: 1 for n in centres}
    for b in boxes:                                           # no box holds two pieces
        assert sum(1 for cx, cy in centres.values() if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]) == 1


def test_find_pieces_recovers_from_a_giant_first_result():
    img, centres = towel_on_table()
    assert detect._implausible(detect.find_pieces_on_plain_background(img), img.shape)
    assert len(detect.find_pieces(img)) == 4
