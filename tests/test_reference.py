import dataclasses

import cv2
import numpy as np

from backend.core import detect, reference

H, W = 450, 800
SPLIT = dataclasses.replace(reference.CFG, split_touching=True)      # cut touching pieces by colour (off by default)


def mesh(seed=5):
    """A dark textured surface (mesh pattern + noise)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    v = 60 + 18 * (np.sin(xs / 2.2) * np.sin(ys / 2.2)) + rng.normal(0, 3, (H, W))
    return np.clip(np.dstack([v * 0.95, v, v]), 0, 255).astype(np.uint8)


def add_brick(img, x, y, w, h, colour, shadow=True):
    if shadow:
        sh = np.zeros((H, W), np.float32)
        cv2.rectangle(sh, (x + 6, y + 8), (x + w + 15, y + h + 13), 1, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 6)
        img[:] = (img * (1 - 0.35 * sh[..., None])).astype(np.uint8)
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
    for i in range(3):
        cv2.circle(img, (x + 13 + i * 23, y + 11), 6, tuple(int(c * 0.8) for c in colour), -1)


PIECES = [(100, 100, 100, 60, (40, 40, 210)), (350, 150, 110, 55, (50, 200, 150)), (550, 250, 60, 60, (230, 230, 235)),
          (200, 300, 90, 50, (20, 20, 25)), (625, 75, 45, 45, (200, 120, 30))]       # red, green, white, black, blue


def scene(shift=(0, 0), gain=1.0):
    img = mesh()
    for x, y, w, h, c in PIECES:
        add_brick(img, x, y, w, h, c)
    img = cv2.warpAffine(img, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    return (img * gain).astype(np.uint8)


def centres_found(boxes):
    return [sum(1 for x, y, w, h in boxes if x <= px <= x + w and y <= py <= y + h) for px, py in
            [(x + w // 2, y + h // 2) for x, y, w, h, _ in PIECES]]


def test_every_piece_is_found_once_including_black_and_white_on_a_dark_textured_surface():
    boxes = reference.find_pieces_by_reference(scene(), mesh())
    assert len(boxes) == 5 and centres_found(boxes) == [1] * 5


def test_a_few_pixels_of_camera_shift_and_a_change_of_exposure_are_compensated():
    info = {}
    boxes = reference.find_pieces_by_reference(scene(shift=(4, -3), gain=0.94), mesh(), info=info)
    assert len(boxes) == 5 and centres_found(boxes) == [1] * 5
    assert info["alignment"] in ("translation", "homography")


def test_identical_photos_have_no_pieces():
    assert reference.find_pieces_by_reference(mesh(), mesh()) == []


def test_a_soft_cast_shadow_is_not_a_piece():
    img = mesh()
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (400, 225), (120, 50), 20, 0, 360, 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 25)
    img = (img * (1 - 0.3 * sh[..., None])).astype(np.uint8)                   # 30% darker at most, same chroma
    assert reference.find_pieces_by_reference(img, mesh()) == []


def test_a_reference_of_another_size_is_resized_and_a_wrong_reference_does_not_crash():
    big = cv2.resize(mesh(), (W * 2, H * 2))
    assert len(reference.find_pieces_by_reference(scene(), big)) == 5
    other = np.full((H, W, 3), 200, np.uint8)
    boxes = reference.find_pieces_by_reference(scene(), other)                    # nothing like the scene: no crash
    assert isinstance(boxes, list)


def test_analyse_with_a_reference_uses_only_the_reference(tmp_path):
    a = detect.analyse(scene(), reference=mesh(), debug=tmp_path / "o.jpg")
    assert len(a.pieces) == 5 and {p.source for p in a.pieces} == {"reference"}
    assert a.studs == [] and a.groups == [] and set(a.timings) == {"reference"}   # no colour seeds, no studs, no surface
    assert (tmp_path / "o.jpg").exists()
    assert len(detect.analyse(scene()).pieces) >= 1                                # without a reference: the usual pipeline


def test_a_white_piece_a_lime_plate_and_a_red_brick_touching_each_other_are_three_pieces():
    """One difference blob holds all three (what a real scan did: one box over three pieces): it is cut along the
    colours, and the unsaturated white piece becomes a box of its own instead of vanishing into the others."""
    img = mesh()
    add_brick(img, 60, 150, 60, 60, (225, 228, 232))                  # white, with its shadow touching the lime plate
    add_brick(img, 110, 120, 170, 50, (40, 220, 170))                 # lime plate
    add_brick(img, 130, 165, 150, 70, (40, 40, 210))                  # red brick against the lime plate
    for x, y, w, h, c in [(450, 60, 70, 55, (200, 120, 30)), (600, 100, 80, 60, (50, 200, 150)),
                          (500, 300, 90, 60, (30, 30, 150)), (650, 300, 60, 60, (20, 120, 220))]:
        add_brick(img, x, y, w, h, c)
    boxes = reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)
    inside = lambda px, py: sum(1 for x, y, w, h in boxes if x <= px <= x + w and y <= py <= y + h)    # noqa: E731
    assert len(boxes) == 7
    assert inside(90, 180) == 1 and inside(195, 145) == 1 and inside(205, 200) == 1        # white, lime, red: one box each


def _big_reds(img):
    for x, y, w, h in [(60, 330, 160, 70), (300, 340, 170, 70), (520, 330, 160, 80)]:
        add_brick(img, x, y, w, h, (30, 30, 200))


def test_two_touching_pieces_of_different_colours_are_two_pieces_even_when_the_blob_is_not_oversized():
    """A blue plate against a lime brick (a real scan boxed them as one). Big red bricks in the photo make the typical
    blob large, so the blob is NOT an outlier: the colours alone must split it."""
    img = mesh()
    _big_reds(img)
    add_brick(img, 330, 120, 90, 45, (200, 110, 30))                  # blue plate
    add_brick(img, 395, 150, 80, 55, (40, 220, 170))                  # lime brick, overlapping its corner
    for x, y in [(60, 60), (160, 40), (600, 60)]:
        add_brick(img, x, y, 60, 45, (50, 200, 150))
    boxes = reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)
    inside = lambda px, py: sum(1 for x, y, w, h in boxes if x <= px <= x + w and y <= py <= y + h)    # noqa: E731
    assert len(boxes) == 8
    assert inside(350, 135) >= 1 and inside(450, 180) >= 1
    blue = [b for b in boxes if b[0] <= 350 <= b[0] + b[2] and b[1] <= 135 <= b[1] + b[3]]
    lime = [b for b in boxes if b[0] <= 450 <= b[0] + b[2] and b[1] <= 185 <= b[1] + b[3]]
    assert blue and lime and blue[0] != lime[0]


def test_one_long_pale_plate_shading_across_a_hue_boundary_is_one_piece_not_two():
    """A real scan counted a 2x8 plate twice: its yellow-green shading straddled two hue classes."""
    img = mesh()
    _big_reds(img)
    x0, y0, w, h = 600, 150, 150, 50
    cols = np.linspace(22, 38, w)                                      # OpenCV hue 22 -> 38 crosses the 30 boundary
    for i, hue in enumerate(cols):
        px = cv2.cvtColor(np.uint8([[[hue, 150, 235]]]), cv2.COLOR_HSV2BGR)[0, 0]
        img[y0:y0 + h, x0 + i] = px
    boxes = reference.find_pieces_by_reference(img, mesh())
    on_plate = [b for b in boxes if b[0] < x0 + w and b[0] + b[2] > x0 and b[1] < y0 + h and b[1] + b[3] > y0]
    assert len(on_plate) == 1 and len(boxes) == 4


def test_duplicate_boxes_of_one_piece_are_merged_but_neighbours_are_not():
    a, b = (100, 100, 200, 80), (110, 105, 150, 70)                    # b lies inside a
    assert reference.merge_duplicates([a, b], 0.6) == [(100, 100, 200, 80)]
    near = [(100, 100, 100, 80), (190, 100, 100, 80)]                  # padded neighbours overlapping a little
    assert reference.merge_duplicates(near, 0.6) == near


def test_stray_pixels_of_an_in_between_hue_do_not_bridge_two_colours():
    """Blue and lime touching, with a few cyan/green pixels along the seam (highlights, edge blur): still two pieces.
    A real scan failed here because those pixels chained blue to lime through the hue bins."""
    img = mesh()
    _big_reds(img)
    add_brick(img, 330, 120, 90, 45, (200, 110, 30))
    add_brick(img, 395, 150, 80, 55, (40, 220, 170))
    for i, hue in enumerate([60, 70, 80, 90] * 6):                      # in-between hues along the seam
        img[150 + (i % 8), 396 + i * 2] = cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for x, y in [(60, 60), (160, 40), (600, 60)]:
        add_brick(img, x, y, 60, 45, (50, 200, 150))
    boxes = reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)
    blue = [b for b in boxes if b[0] <= 350 <= b[0] + b[2] and b[1] <= 135 <= b[1] + b[3]]
    lime = [b for b in boxes if b[0] <= 450 <= b[0] + b[2] and b[1] <= 185 <= b[1] + b[3]]
    assert len(boxes) == 8 and blue and lime and blue[0] != lime[0]


def test_red_and_orange_pieces_touching_at_a_corner_are_two_pieces():
    img = mesh()
    _big_reds(img)
    add_brick(img, 330, 120, 100, 55, (30, 30, 210))                    # red
    add_brick(img, 425, 170, 80, 50, (20, 130, 235))                    # orange (hue ~12), only its corner touches
    for x, y in [(60, 60), (160, 40), (600, 60)]:
        add_brick(img, x, y, 60, 45, (50, 200, 150))
    boxes = reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)
    assert len(boxes) == 8


def _grey_brick(img):
    """A grey brick on dark mesh: lit top (bright) and two side faces only a little brighter than the mesh, as in a real scan
    where the box covered just the top of the grey brick."""
    top = np.array([[430, 90], [560, 70], [585, 115], [455, 140]], np.int32)
    side = np.array([[455, 140], [585, 115], [585, 150], [455, 175]], np.int32)
    end = np.array([[430, 90], [455, 140], [455, 175], [430, 125]], np.int32)
    cv2.fillPoly(img, [side], (80, 80, 80))
    cv2.fillPoly(img, [end], (76, 76, 76))
    cv2.fillPoly(img, [top], (170, 170, 170))
    return (430, 70, 585, 175)                                           # x0, y0, x1, y1 of the whole brick


def test_a_piece_whose_sides_are_close_to_the_surface_is_boxed_whole_not_just_its_lit_top():
    img = mesh()
    _big_reds(img)
    x0, y0, x1, y1 = _grey_brick(img)
    add_brick(img, 60, 60, 60, 45, (50, 200, 150))
    add_brick(img, 700, 60, 60, 45, (200, 110, 30))
    colour_only = dataclasses.replace(reference.CFG, texture_max_share=0.0)          # structure cues off: colour difference alone
    grown = reference.find_pieces_by_reference(img, mesh(), cfg=colour_only)
    top_only = reference.find_pieces_by_reference(img, mesh(), cfg=dataclasses.replace(colour_only, grow_frac=0))
    pick = lambda boxes: [b for b in boxes if b[0] <= 500 <= b[0] + b[2] and b[1] <= 110 <= b[1] + b[3]]    # noqa: E731
    assert len(pick(grown)) == 1 and len(pick(top_only)) == 1
    gx, gy, gw, gh = pick(grown)[0]
    assert gx <= x0 + 8 and gy <= y0 + 8 and gx + gw >= x1 - 8 and gy + gh >= y1 - 8       # the whole brick, sides included
    assert pick(top_only)[0][2] * pick(top_only)[0][3] < 0.85 * gw * gh                    # without the growth: clearly smaller
    both = pick(reference.find_pieces_by_reference(img, mesh()))[0]                        # structure cues on: whole as well
    assert both[0] <= x0 + 8 and both[0] + both[2] >= x1 - 8


def test_growing_into_weak_differences_does_not_swallow_a_cast_shadow():
    img = mesh()
    _big_reds(img)
    add_brick(img, 300, 120, 100, 60, (50, 200, 150))                  # add_brick casts a soft shadow
    boxes = reference.find_pieces_by_reference(img, mesh())
    b = [b for b in boxes if b[0] <= 350 <= b[0] + b[2] and b[1] <= 150 <= b[1] + b[3]][0]
    assert b[2] <= 100 * 1.4 and b[3] <= 60 * 1.5


def test_yellow_and_lime_bricks_side_by_side_are_two_pieces_but_a_shading_plate_is_one():
    """Yellow (hue 25) and lime (hue 38) are only 13 degrees apart and share a long border, so "close hue + long contact" alone
    merged them. The hue JUMPS across their border; inside one shading plate it does not."""
    for yellow, green in [((20, 200, 240), (40, 220, 170)), ((20, 200, 240), (60, 170, 70)), ((0, 180, 250), (40, 220, 170))]:
        img = mesh()
        _big_reds(img)
        add_brick(img, 300, 120, 90, 55, yellow)
        add_brick(img, 385, 125, 90, 55, green)                           # touching it along a 50 px border
        for x, y in [(60, 60), (160, 40), (600, 60)]:
            add_brick(img, x, y, 60, 45, (200, 110, 30))
        boxes = reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)
        left = [b for b in boxes if b[0] <= 330 <= b[0] + b[2] and b[1] <= 145 <= b[1] + b[3]]
        right = [b for b in boxes if b[0] <= 450 <= b[0] + b[2] and b[1] <= 150 <= b[1] + b[3]]
        assert len(boxes) == 8 and left and right and left[0] != right[0], (yellow, green)


# ---------------------------------------------------------------- pieces the same colour as the surface
def _flat_brick(img, x, y, w, h, grey):
    cv2.rectangle(img, (x, y), (x + w, y + h), (grey, grey, grey), -1)                  # smooth, same average colour as the mesh
    for i in range(3):
        cv2.circle(img, (x + 14 + i * 22, y + 12), 7, (grey - 6, grey - 6, grey - 6), 2)    # studs: faint rims


def test_a_brick_the_same_colour_as_a_textured_surface_is_found_by_its_missing_texture():
    """Black on black mesh: the average colour is the same, but the mesh pattern is gone under smooth plastic."""
    img = mesh()
    _big_reds(img)
    _flat_brick(img, 400, 100, 90, 50, 61)                                               # the mesh averages ~61 grey
    add_brick(img, 60, 60, 60, 45, (50, 200, 150))
    colour_only = dataclasses.replace(reference.CFG, texture_max_share=0.0)
    inside = lambda boxes: [b for b in boxes if b[0] <= 445 <= b[0] + b[2] and b[1] <= 125 <= b[1] + b[3]]    # noqa: E731
    assert inside(reference.find_pieces_by_reference(img, mesh(), cfg=colour_only)) == []     # colour alone cannot see it
    found = inside(reference.find_pieces_by_reference(img, mesh()))
    assert len(found) == 1 and found[0][2] >= 90 and found[0][3] >= 50
    assert len(reference.find_pieces_by_reference(img, mesh())) == 5                     # 3 reds + 1 green + the same-colour brick


def test_a_white_brick_on_plain_white_paper_is_found_by_the_detail_it_brings():
    paper = np.full((H, W, 3), 238, np.uint8)
    paper = np.clip(paper + np.random.default_rng(2).normal(0, 1.2, paper.shape), 0, 255).astype(np.uint8)
    img = paper.copy()
    cv2.rectangle(img, (300, 150), (420, 215), (238, 238, 238), -1)                       # same white as the paper
    for i in range(4):
        cv2.circle(img, (325 + i * 28, 170), 10, (228, 228, 228), 2)                      # stud rims
    cv2.line(img, (300, 215), (420, 215), (200, 200, 200), 3)                             # shaded lower edge
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    boxes = reference.find_pieces_by_reference(img, paper)
    assert len(boxes) == 1 and boxes[0][0] <= 305 and boxes[0][0] + boxes[0][2] >= 415


def test_structure_cues_switch_themselves_off_when_the_photos_do_not_match():
    """A different surface pattern everywhere: the structure cues would flag the whole photo, so they are ignored."""
    ref = mesh()
    other = cv2.flip(mesh(seed=9), 1)
    info = {}
    mask, _ = reference.difference_mask(other, ref, info=info)
    assert info["structure"] is False
    assert reference.find_pieces_by_reference(mesh(), mesh()) == []                       # matching photos: nothing


def test_a_brick_with_two_colours_is_one_piece_unless_splitting_is_asked_for():
    img = mesh()
    _big_reds(img)
    add_brick(img, 300, 120, 100, 55, (20, 200, 240))                                      # yellow half
    add_brick(img, 395, 120, 100, 55, (60, 170, 70), shadow=False)                         # green half, side by side
    add_brick(img, 60, 60, 60, 45, (200, 110, 30))
    assert len(reference.find_pieces_by_reference(img, mesh())) == 5                       # 3 reds + 1 blue + the two-colour brick
    assert len(reference.find_pieces_by_reference(img, mesh(), cfg=SPLIT)) == 6


def test_a_reference_scan_warns_when_the_empty_photo_was_not_empty():
    """Pieces already in the 'empty' photo are invisible to the comparison: the scan must say so instead of silently missing them."""
    import asyncio

    from backend.core import gemini_detect as gd
    not_empty = mesh()
    for x, y, w, h, c in PIECES[:3]:
        add_brick(not_empty, x, y, w, h, c)                              # three pieces were left on the table
    d = asyncio.run(gd.detect_pieces(scene(), "opencv", None, reference=not_empty))
    assert len(d.boxes) == 2 and "WARNING" in d.note
    assert "3 more pieces" in d.note
    ok = asyncio.run(gd.detect_pieces(scene(), "opencv", None, reference=mesh()))
    assert len(ok.boxes) == 5 and "WARNING" not in ok.note and "changed" in ok.note


def _other_weave():
    """The same kind of surface photographed from another position: a different weave scale and angle."""
    rng = np.random.default_rng(11)
    ys, xs = np.mgrid[0:H, 0:W]
    v = 80 + 18 * (np.sin((xs + 0.6 * ys) / 3.1) * np.sin((ys - 0.3 * xs) / 3.1)) + rng.normal(0, 3, (H, W))
    return np.clip(np.dstack([v * 0.95, v, v]), 0, 255).astype(np.uint8)


def test_an_empty_photo_of_another_view_is_recognised_as_not_matching_and_the_normal_detection_is_used():
    """A real scan: the empty photo was brighter, at another angle and zoom than the photo with pieces. Nothing could be compared
    (38% of the frame 'changed'). The match check sees the surface patterns do not correlate and falls back."""
    import asyncio

    from backend.core import gemini_detect as gd
    info = {}
    reference.find_pieces_by_reference(scene(), _other_weave(), info=info)
    assert info["matched"] is False and info["match"] < 0.4
    reference.find_pieces_by_reference(scene(shift=(3, 2), gain=0.9), mesh(), info=info)
    assert info["matched"] is True and info["match"] > 0.9
    d = asyncio.run(gd.detect_pieces(scene(), "opencv", None, reference=_other_weave()))
    assert d.mode == "opencv" and "does not match" in d.note and "WARNING" in d.note        # not "reference": it was not used
    assert len(d.boxes) >= 4                                                               # the normal detection still finds the pieces
    good = asyncio.run(gd.detect_pieces(scene(), "opencv", None, reference=mesh()))
    assert good.mode == "reference" and "does not match" not in good.note
