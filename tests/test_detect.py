from pathlib import Path

import cv2

from backend.core.detect import find_pieces

SAMPLE = Path(__file__).resolve().parent.parent / "lego_pieces.jpg"


def test_sample_image_has_42_pieces():
    img = cv2.imread(str(SAMPLE))
    assert img is not None
    boxes = find_pieces(img)
    assert len(boxes) == 42
