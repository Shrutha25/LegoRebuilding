"""Detection must not depend on the colour of the (uniform) background: light, dark, neutral or coloured.

Synthetic scenes (tests/background_sweep.py): the same nine pieces with shadows, noise and a lighting
gradient on 62 uniform backgrounds. A representative subset is checked one by one, then the whole grid.
"""

import cv2
import numpy as np
import pytest

from backend.core import detect
from tests import background_sweep as B

GRID = dict(B.grid())

# Known limits (see README): pieces too close to the background's own shadows or colour to separate.
KNOWN_LIMITS = {
    "red-brown S=140 V=130",      # a black piece is about the colour of this surface's deep shadows
    "green S=220 V=210",          # a green piece on bright saturated green: low contrast
    "blue S=140 V=130",           # a dark grey piece splits in two against its own shadow
    "blue S=220 V=130",
}

# The scene does not expect the dark grey piece here (it is the colour of the surface), but its studs are visible, so
# the stud detector reports it: a box with no expected piece in it, which the scorer counts as "extra".
STUD_FINDS = {"yellow S=140 V=60"}

# The scene does not expect the white piece on a white sheet, but its studs and rim are visible, so it is found: one box with no
# expected piece in it, which the scorer counts as "extra". (The pre-mask detector hid it inside a neighbouring blob's box.)
PALE_PIECE_FOUND = {"grey V=250"}

FAST = ["grey V=10", "grey V=70", "grey V=150", "grey V=225", "grey V=250",
        "orange/wood S=140 V=130", "red-brown S=70 V=130", "blue S=70 V=210", "green S=220 V=60", "purple S=140 V=210",
        "yellow S=70 V=60", "green S=140 V=210"]


def run(name):
    img, centres, expected = B.scene(GRID[name])
    return B.score(detect.find_pieces(img), centres, expected)


@pytest.mark.parametrize("name", FAST)
def test_every_piece_found_once_on_representative_backgrounds(name):
    missed, merged, extra = run(name)
    if name in PALE_PIECE_FOUND:
        assert (missed, merged) == ([], 0) and extra <= 1, name
        return
    assert (missed, merged, extra) == ([], 0, 0), name


def test_full_background_grid_has_no_regressions_beyond_the_known_limits():
    bad = {name for name in GRID if run(name) != ([], 0, 0)}
    allowed = KNOWN_LIMITS | STUD_FINDS | PALE_PIECE_FOUND
    assert bad <= allowed, f"new failures: {sorted(bad - allowed)}"


def test_scene_generator_expects_only_pieces_that_differ_from_the_background():
    _, _, expected = B.scene(GRID["grey V=10"])
    assert "black" not in expected and "white" in expected
    _, _, expected = B.scene(GRID["grey V=250"])
    assert "white" not in expected and "black" in expected


# ---------------------------------------------------------------- helpers
def test_edge_strips_are_recognised_but_pieces_are_not():
    shape = (700, 1000, 3)
    assert detect._is_edge_strip((855, 0, 145, 700), shape)                    # full-height strip on the right edge
    assert detect._is_edge_strip((0, 0, 1000, 60), shape)
    assert not detect._is_edge_strip((400, 100, 200, 150), shape)              # an ordinary piece
    assert not detect._is_edge_strip((0, 100, 200, 150), shape)                # touches the edge but is not long


def test_the_whole_photo_box_still_signals_a_failed_background_method():
    """The edge-strip filter must not hide the failure signal: trust is judged on the raw result."""
    img, centres = np.full((600, 900, 3), 120, np.uint8), None
    raw = [(0, 0, 900, 600), (100, 100, 80, 60)]
    assert detect._implausible(raw, img.shape)
    assert detect._is_edge_strip(raw[0], img.shape)                            # it would be filtered afterwards


def test_merge_fragments_joins_overlapping_parts_but_not_neighbours():
    fragment, body = (399, 508, 107, 38), (415, 531, 121, 128)                 # one piece in two parts
    assert detect._merge_fragments([fragment, body]) == [(399, 508, 137, 151)]
    a, b = (100, 100, 200, 200), (290, 100, 200, 200)                          # neighbours: padded boxes just touch
    assert detect._merge_fragments([a, b]) == [a, b]


def test_add_missing_adds_only_where_nothing_was_found():
    shape = (700, 1000, 3)
    kept = [(100, 100, 150, 100)]
    assert detect.add_missing(kept, [(600, 400, 100, 80)], shape) == [(100, 100, 150, 100), (600, 400, 100, 80)]
    assert detect.add_missing(kept, [(105, 105, 140, 90)], shape) == kept                    # duplicate
    assert detect.add_missing(kept, [(50, 50, 400, 300)], shape) == kept                     # blob around a found piece
    assert detect.add_missing(kept, [(0, 0, 900, 600)], shape) == kept                       # giant
    joined = detect.add_missing(kept, [(180, 140, 120, 90)], shape)                          # partly overlaps: same piece
    assert joined == [(100, 100, 200, 130)]


def test_surface_chroma_distance_treats_shadows_as_surface_on_any_colour():
    for bg in (B.hsv_bg(15, 140, 130), B.hsv_bg(0, 0, 220), B.hsv_bg(100, 70, 210)):
        img, centres, _ = B.scene(bg)
        dist = detect._surface_chroma_distance(img, detect._surface_model(img))
        bare = dist[330, 380]                                    # empty surface between pieces
        shadow = dist[165, 235]                                  # inside a cast shadow of the first piece
        assert bare < 25 and shadow < 25
    img, centres, _ = B.scene(B.hsv_bg(15, 140, 130))
    dist = detect._surface_chroma_distance(img, detect._surface_model(img))
    bx, by = centres["blue"]
    assert dist[by, bx] > 25                                     # a blue piece is not surface
