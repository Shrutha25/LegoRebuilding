import cv2
import numpy as np
import pytest

from backend.core import shadow
from backend.core.shadow import analyse, counterpart_name, parse_kind_dims, verdict_for

PAPER = 205


def scene(specs, shadow_sides=("left",)):
    """Light paper with coloured two-stud-wide pieces. Each spec: (colour, (x, y, w, h), shadow_px).
    A soft dark shadow of that length is drawn on the given sides of the piece. Returns (img, boxes)."""
    img = np.full((600, 900, 3), PAPER, np.uint8)
    boxes = []
    for colour, (x, y, w, h), length in specs:
        for side in shadow_sides:
            if side == "left":
                cv2.rectangle(img, (x - length, y + 4), (x, y + h), (118, 118, 116), -1)
            elif side == "right":
                cv2.rectangle(img, (x + w, y + 4), (x + w + length, y + h), (118, 118, 116), -1)
            elif side == "up":
                cv2.rectangle(img, (x + 4, y - length), (x + w, y), (118, 118, 116), -1)
            elif side == "down":
                cv2.rectangle(img, (x + 4, y + h), (x + w, y + h + length), (118, 118, 116), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    for colour, (x, y, w, h), _ in specs:
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
        boxes.append((x - 12, y - 12, w + 24, h + 24))
    return img, boxes


# 60 px wide = 2 studs (15.8 mm) -> 3.8 px/mm.  A 60 px shadow = 15.8 mm (brick), 20 px = 5.3 mm (plate)
BRICK_PX, PLATE_PX = 60, 20
SPECS = [((170, 100, 20), (200, 60, 150, 60), BRICK_PX),      # blue, long shadow
         ((40, 200, 140), (420, 60, 150, 60), PLATE_PX),      # lime, short shadow
         ((110, 240, 235), (200, 260, 150, 60), BRICK_PX),    # yellow, long
         ((45, 42, 42), (420, 260, 200, 60), PLATE_PX),       # dark grey, short
         ((20, 40, 110), (200, 460, 120, 60), PLATE_PX)]      # brown, short
NAMES = ["Brick 2 x 4", "Plate 2 x 4", "Brick 2 x 3", "Plate 2 x 6", "Plate 2 x 5"]


def test_parse_and_counterpart_names():
    assert parse_kind_dims("Brick 2 x 4") == ("brick", 2, 4)
    assert parse_kind_dims("Plate  4 x 2") == ("plate", 2, 4)         # order-insensitive, tolerant of spaces
    for odd in ("Brick, Round 2 x 2", "Slope 45 2 x 2", "Brick 2 x 4 with Studs", "Tile 2 x 2", "", None):
        assert parse_kind_dims(odd) is None
    assert counterpart_name("Brick 2 x 4") == "Plate 2 x 4"
    assert counterpart_name("Plate 1 x 2") == "Brick 1 x 2"
    assert counterpart_name("Tile 2 x 2") is None


def test_verdict_thresholds_leave_an_unclear_band():
    assert verdict_for(15) == "brick" and verdict_for(shadow.BRICK_MM) == "brick"
    assert verdict_for(6) == "plate" and verdict_for(shadow.PLATE_MM) == "plate"
    assert verdict_for((shadow.BRICK_MM + shadow.PLATE_MM) / 2) == "unclear"


def test_long_shadows_are_bricks_and_short_ones_plates():
    img, boxes = scene(SPECS)
    res = analyse(img, boxes, NAMES)
    assert [r["verdict"] for r in res] == ["brick", "plate", "brick", "plate", "plate"]
    assert not any(r["conflict"] for r in res)
    assert res[0]["shadow_mm"] > shadow.BRICK_MM > shadow.PLATE_MM > res[1]["shadow_mm"]


def test_disagreement_with_the_recognised_kind_is_a_conflict_with_a_reason():
    img, boxes = scene(SPECS)
    res = analyse(img, boxes, ["Plate 2 x 4", "Brick 2 x 4", "Brick 2 x 3", "Plate 2 x 6", "Plate 2 x 5"])
    assert res[0]["conflict"] and res[0]["verdict"] == "brick" and "not a plate" in res[0]["reason"]   # long shadow, called a plate
    assert res[1]["conflict"] and res[1]["verdict"] == "plate" and "not a brick" in res[1]["reason"]   # short shadow, called a brick
    assert not res[2]["conflict"] and res[2]["reason"] is None


@pytest.mark.parametrize("side", ["left", "right", "up", "down"])
def test_light_direction_is_found_automatically(side):
    img, boxes = scene(SPECS, shadow_sides=(side,))
    res = analyse(img, boxes, NAMES)
    assert [r["verdict"] for r in res] == ["brick", "plate", "brick", "plate", "plate"]


def test_no_opinion_without_a_clear_shadow_direction():
    img, boxes = scene(SPECS, shadow_sides=("left", "right", "up", "down"))       # shadows all around: flat light
    assert analyse(img, boxes, ["Brick 2 x 4"] * 5) == [None] * 5


def test_no_opinion_with_fewer_than_two_measurable_pieces():
    img, boxes = scene(SPECS[:1])
    assert analyse(img, boxes, ["Brick 2 x 4"]) == [None]
    img, boxes = scene(SPECS)
    res = analyse(img, boxes, ["Brick 2 x 4", None, None, "Brick, Round 2 x 2", None])
    assert res == [None] * 5


def test_only_plain_bricks_and_plates_get_an_opinion():
    img, boxes = scene(SPECS)
    res = analyse(img, boxes, ["Brick 2 x 4", "Plate 2 x 4", "Slope 45 2 x 2", "Tile 2 x 6", "Brick, Round 2 x 2"])
    assert res[0] and res[1] and res[2:] == [None, None, None]


def test_pale_pieces_are_skipped_and_garbage_never_raises():
    img, boxes = scene([((232, 232, 232), (200, 60, 150, 60), BRICK_PX), ((235, 235, 235), (420, 60, 150, 60), BRICK_PX)])
    assert analyse(img, boxes, ["Brick 2 x 4", "Brick 2 x 4"]) == [None, None]
    assert analyse(None, [(0, 0, 10, 10)], ["Brick 2 x 4"]) == [None]
    assert analyse(np.zeros((5, 5, 3), np.uint8), [(0, 0, 5, 5), (0, 0, 5, 5)], ["Brick 2 x 4"] * 2) == [None, None]


def test_no_opinion_when_the_light_is_too_diffuse_for_shadows_to_mean_anything():
    """Same pieces, but every shadow short (overhead/soft light): bricks measure like plates, so an
    absolute threshold would call real bricks plates. The check must abstain, not flag them."""
    short = [(c, box, 18) for c, box, _ in SPECS]                       # ~4.7 mm for every piece
    img, boxes = scene(short)
    assert analyse(img, boxes, NAMES) == [None] * 5
    img, boxes = scene([(c, box, 16) for c, box, _ in SPECS])
    assert analyse(img, boxes, ["Plate 2 x 4", "Plate 2 x 4", "Plate 2 x 3", "Plate 2 x 6", "Plate 2 x 5"]) == [None] * 5


def test_one_clearly_long_shadow_in_the_photo_is_enough_to_trust_the_scale():
    img, boxes = scene(SPECS)                                           # includes 60 px (15.8 mm) shadows
    assert any(r is not None for r in analyse(img, boxes, NAMES))
    assert shadow.MIN_LONGEST_MM > shadow.BRICK_MM
