from pathlib import Path

import cv2
import pytest

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


def test_sample_keeps_the_original_methods_boxes_except_where_a_split_rule_has_evidence():
    img = cv2.imread(str(SAMPLE))
    original = detect.find_pieces_on_plain_background(img)
    assert not detect._implausible(original, img.shape)
    final = detect.find_pieces(img)
    assert len(final) == len(original) == 42
    assert len([b for b in final if b in original]) >= 38     # a few merged/overlapping boxes are re-cut or grown on masks


def white_sheet_with_bridged_pair():
    """A plain white sheet filling the photo: a red and a blue brick 24 px apart joined by a long soft
    shadow, plus a green piece elsewhere. The border is plain, so the background method is trusted, but the
    shadow makes it return the two bricks as ONE box."""
    img = np.full((600, 900, 3), 250, np.uint8)
    cv2.rectangle(img, (150, 200), (300, 320), (60, 60, 200), -1)                 # red
    cv2.rectangle(img, (324, 200), (480, 320), (200, 90, 20), -1)                 # blue, 24 px to the right
    cv2.rectangle(img, (300, 230), (324, 330), (140, 140, 140), -1)               # shadow bridging the gap
    img = cv2.GaussianBlur(img, (0, 0), 2)
    cv2.rectangle(img, (150, 200), (300, 320), (60, 60, 200), -1)
    cv2.rectangle(img, (324, 200), (480, 320), (200, 90, 20), -1)
    cv2.rectangle(img, (650, 380), (780, 470), (60, 170, 60), -1)                 # green
    return img, {"red": (225, 260), "blue": (402, 260), "green": (715, 425)}


def test_combining_both_methods_splits_pieces_the_background_method_merged():
    img, centres = white_sheet_with_bridged_pair()
    background = detect.find_pieces_on_plain_background(img)
    assert not detect._implausible(background, img.shape)                 # plausible: it is trusted...
    held = [b for b in background if sum(1 for cx, cy in centres.values() if b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]) > 1]
    assert held                                                           # ...but it merged two bricks into one box
    boxes = detect.find_pieces(img)
    assert count_per_brick(boxes, centres) == {"red": 1, "blue": 1, "green": 1} and len(boxes) == 3


def test_both_methods_are_run_and_a_plausible_background_result_is_kept_for_pale_pieces():
    img = cv2.imread(str(SAMPLE))
    background = detect.find_pieces_on_plain_background(img)
    colour = detect.find_pieces_by_colour(img)
    assert len(colour) < len(background) == 42                            # colour alone misses the pale pieces
    assert len(detect.find_pieces(img)) == 42                             # combined keeps every one of them


def test_an_implausible_background_result_is_ignored_entirely_including_its_junk_boxes():
    img, centres = towel_on_table()
    background = detect.find_pieces_on_plain_background(img)
    assert detect._implausible(background, img.shape)
    boxes = detect.find_pieces(img)
    colour = detect.find_pieces_by_colour(img)
    assert len(boxes) == len(colour) == 4                                 # the junk boxes are not there...
    for b in boxes:                                                       # ...and each box is a colour box (studs may enlarge it)
        assert max(detect._iou(b, c) for c in colour) > 0.6


def test_combine_boxes_rules():
    shape = (1000, 2000, 3)
    merged_box = (100, 100, 400, 200)                                     # holds two colour boxes: a merge
    two_inside = [(110, 110, 150, 150), (300, 120, 160, 160)]
    lone = (1000, 500, 100, 100)                                          # background-only piece: kept
    colour_only = (1500, 800, 120, 120)                                   # missed by background: added
    duplicate = (1002, 502, 98, 98)                                       # same piece as `lone`: not added again
    giant = (0, 0, 1500, 800)                                             # > 25% of the photo: dropped
    out = detect.combine_boxes([merged_box, lone, giant], two_inside + [colour_only, duplicate, giant], shape)
    assert merged_box not in out and all(b in out for b in two_inside)
    assert lone in out and colour_only in out and duplicate not in out and giant not in out
    assert len(out) == 4
    # one colour box inside one background box is the same piece: the background box stays
    assert detect.combine_boxes([merged_box], [(150, 150, 100, 100)], shape) == [merged_box]
    assert detect.combine_boxes([], [], shape) == []


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


def close_pair_scene(gap=22, crease=True):
    """A yellow brick and a dark brown round piece only `gap` px apart on grey paper, with a dark
    crease between them (as in a user's photo). They are separate pieces and must stay separate."""
    img = np.full((600, 900, 3), 180, np.uint8)
    if crease:
        cv2.line(img, (330, 120), (470, 380), (95, 100, 105), 14)
    cv2.rectangle(img, (150, 200), (300, 330), (110, 240, 235), -1)               # yellow
    cv2.circle(img, (300 + gap + 60, 300), 60, (20, 40, 80), -1)                # dark brown, `gap` px away
    cv2.rectangle(img, (600, 200), (800, 320), (170, 100, 20), -1)               # a blue piece elsewhere
    return img, {"yellow": (225, 265), "brown": (300 + gap + 60, 300), "blue": (700, 260)}


def test_close_pieces_of_different_colours_stay_separate_even_with_a_crease_between():
    for gap in (22, 30, 40):
        img, centres = close_pair_scene(gap)
        boxes = detect.find_pieces_by_colour(img)
        assert count_per_brick(boxes, centres) == {"yellow": 1, "brown": 1, "blue": 1}, gap
        assert len(boxes) == 3, gap


def test_same_colour_fragments_of_a_dark_piece_are_still_one_piece():
    img = np.full((500, 800, 3), 200, np.uint8)
    cv2.rectangle(img, (100, 150), (500, 260), (70, 66, 66), -1)                  # dark grey, V ~ 70
    for xx in range(130, 500, 62):                                                # stud shadows, 27 px apart
        cv2.circle(img, (xx, 205), 17, (25, 24, 24), -1)
    boxes = detect.find_pieces_by_colour(img)
    assert len(boxes) == 1 and boxes[0][2] > 380 and boxes[0][0] <= 100


def test_group_helper_hue_gap_is_circular():
    assert detect._hue_gap(5, 175) == 10 and detect._hue_gap(30, 30) == 0 and detect._hue_gap(0, 90) == 90


def pale_piece_scene():
    """A pale grey piece (neither coloured nor dark) whose only dark parts are two small stud holes on
    top and a larger cavity below, sized like a real piece in a photo (gaps of about 3% of the photo's
    height); on a surface that does not fill the frame, plus a lone dark speck and an ordinary red piece."""
    img = np.full((600, 900, 3), 185, np.uint8)
    cv2.rectangle(img, (0, 0), (900, 40), (35, 70, 115), -1)                      # wood at the top edge
    cv2.rectangle(img, (320, 200), (420, 285), (190, 190, 190), -1)               # pale grey body
    cv2.circle(img, (342, 212), 5, (25, 24, 24), -1)                              # stud holes: ~80 px each, below the
    cv2.circle(img, (398, 212), 5, (25, 24, 24), -1)                              # size that counts as a piece on its own
    cv2.rectangle(img, (335, 240), (405, 272), (30, 28, 28), -1)                  # cavity: the big dark part
    cv2.circle(img, (700, 400), 7, (25, 24, 24), -1)                              # an isolated dark speck
    cv2.rectangle(img, (150, 380), (260, 470), (60, 60, 200), -1)                 # an ordinary red piece
    return img


def test_small_dark_fragments_join_their_piece_so_the_box_covers_the_whole_piece():
    boxes = detect.find_pieces_by_colour(pale_piece_scene())
    pale = [b for b in boxes if b[0] <= 370 <= b[0] + b[2] and b[1] <= 240 <= b[1] + b[3]]
    assert len(pale) == 1
    x, y, w, h = pale[0]
    assert y <= 210 and y + h >= 270 and x <= 336 and x + w >= 405                # includes the stud holes at the top


def test_a_small_dark_fragment_on_its_own_is_not_a_piece():
    boxes = detect.find_pieces_by_colour(pale_piece_scene())
    assert not any(b[0] <= 700 <= b[0] + b[2] and b[1] <= 400 <= b[1] + b[3] for b in boxes)
    assert len(boxes) == 2                                                          # the pale piece and the red one


def wood_table_scene(seed=5):
    """Pieces on a strongly saturated, grainy brown 'wooden table' (as in a user's photo, where 95% of
    the photo was strongly coloured): light and dark grain bands plus noise. Includes a white piece, a
    pale piece and a dark grey piece, which differ from the brown in chroma or lightness."""
    h, w = 700, 1000
    rng = np.random.default_rng(seed)
    base = np.array([35, 85, 150], np.float32)                                   # BGR brown, saturation ~200
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    bands = np.repeat(rng.uniform(0.65, 1.25, (h // 14 + 1, 1)), 14, axis=0)[:h]   # horizontal grain: light/dark rows
    img *= bands[:, :, None]
    img += rng.normal(0, 7, (h, w, 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    pieces = {"blue": ((180, 110, 20), (100, 80, 230, 160)), "white": ((235, 235, 232), (340, 70, 440, 170)),
              "pale green": ((190, 235, 205), (560, 90, 700, 140)), "dark grey": ((75, 70, 68), (120, 300, 360, 380)),
              "yellow": ((40, 210, 245), (480, 280, 600, 380)), "green": ((100, 170, 40), (700, 260, 940, 520))}
    centres = {}
    for name, (colour, (x0, y0, x1, y1)) in pieces.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        centres[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
    return img, centres


def test_a_coloured_wooden_surface_is_recognised_and_the_seed_methods_cannot_cope():
    img, _ = wood_table_scene()
    assert detect.surface_defeats_seeds(img)
    assert not detect.surface_defeats_seeds(np.full((300, 300, 3), 235, np.uint8))                 # white paper
    assert not detect.surface_defeats_seeds(towel_on_table()[0])                                    # towel with some pieces
    colour = detect.find_pieces_by_colour(img)
    assert len(colour) < 6                                                                         # the seed method misses them


def test_pieces_on_a_wooden_surface_are_found_by_modelling_the_surface():
    img, centres = wood_table_scene()
    boxes = detect.find_pieces(img)
    assert count_per_brick(boxes, centres) == {n: 1 for n in centres}
    assert len(boxes) == len(centres)
    assert max(w * h for _, _, w, h in boxes) < 0.25 * img.shape[0] * img.shape[1]


def test_surface_method_is_deterministic_and_ignores_the_table_edge():
    img, centres = wood_table_scene()
    assert detect.find_pieces_by_surface(img) == detect.find_pieces_by_surface(img)
    edge = img.copy()
    cv2.rectangle(edge, (0, 600), (400, 700), (180, 110, 20), -1)                                 # something at the photo edge
    assert len(detect.find_pieces_by_surface(edge)) == len(centres)


def test_on_light_surfaces_the_surface_method_adds_no_junk(monkeypatch):
    """The surface method always runs and may fill gaps, but on photos the other methods already handle it
    must not add boxes: results are the same as without it."""
    for img, expected in ((cv2.imread(str(SAMPLE)), 42), (towel_on_table()[0], 4)):
        with_surface = detect.find_pieces(img)
        monkeypatch.setattr(detect, "find_pieces_by_surface", lambda *a, **k: [])
        without = detect.find_pieces(img)
        monkeypatch.undo()
        assert len(with_surface) == len(without) == expected



def black_cloth_scene(seed=8):
    """Pieces on a dark, slightly mottled cloth (a user's photo: about 75% of it was 'very dark'). Includes a
    NEUTRAL GREY brick, which has the same chroma as the black cloth and differs only in lightness."""
    h, w = 700, 1000
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 30, np.float32) + rng.normal(0, 3, (h, w, 1))
    img += cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 25)[:, :, None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    pieces = {"grey": ((125, 122, 120), (60, 100, 330, 190)), "blue": ((190, 120, 20), (420, 60, 560, 170)),
              "white": ((228, 228, 232), (620, 80, 760, 170)), "yellow": ((40, 200, 245), (80, 300, 200, 400)),
              "green": ((90, 190, 60), (760, 260, 940, 560)), "pale green": ((190, 235, 205), (330, 420, 640, 500))}
    centres = {}
    for name, (colour, (x0, y0, x1, y1)) in pieces.items():
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)
        centres[name] = ((x0 + x1) // 2, (y0 + y1) // 2)
    return img, centres


def test_a_dark_surface_is_recognised_and_pieces_on_it_are_found_including_neutral_grey_ones():
    img, centres = black_cloth_scene()
    assert detect.surface_defeats_seeds(img)
    assert detect.surface_defeats_seeds(wood_table_scene()[0])
    assert not detect.surface_defeats_seeds(cv2.imread(str(SAMPLE)))
    boxes = detect.find_pieces(img)
    assert count_per_brick(boxes, centres) == {n: 1 for n in centres}              # incl. the grey brick on black
    assert len(boxes) == len(centres)


def test_a_light_piece_is_not_part_of_a_dark_surface_even_with_the_same_chroma():
    """Regression: surface clusters were chosen by chroma alone, so a neutral grey piece (L~125) counted as part of
    a neutral black cloth (L~30) and was never found."""
    img, centres = black_cloth_scene()
    grey = centres["grey"]
    boxes = detect.find_pieces_by_surface(img)
    assert any(b[0] <= grey[0] <= b[0] + b[2] and b[1] <= grey[1] <= b[1] + b[3] for b in boxes)
    boxes_chroma_only = detect.find_pieces_by_surface(img, lightness_tol=10_000)   # the old chroma-only behaviour
    assert not any(b[0] <= grey[0] <= b[0] + b[2] and b[1] <= grey[1] <= b[1] + b[3] for b in boxes_chroma_only)


# ------------------------------------------------------------------ dim textured surface, 45 degree photos
def _mesh_scene():
    """Dark mesh with a red brick touching a lime plate, a white slope, a small blue and a grey piece, and a long
    dark-red brick. Returns (image, centres)."""
    from tests.test_reference import H, W, add_brick, mesh
    img = mesh()
    pieces = {"red": ((40, 40, 210), (100, 200, 130, 70)), "lime": ((40, 220, 170), (150, 140, 160, 55)),
              "white": ((225, 228, 232), (400, 100, 70, 60)), "blue": ((200, 120, 30), (600, 80, 45, 45)),
              "grey": ((120, 120, 125), (450, 320, 90, 50)), "dark red": ((30, 30, 150), (600, 300, 130, 60))}
    centres = {}
    for name, (colour, (x, y, w, h)) in pieces.items():
        add_brick(img, x, y, w, h, colour)
        centres[name] = (x + w // 2, y + h // 2)
    return img, centres


def test_a_dim_textured_surface_is_recognised_and_every_piece_gets_its_own_box():
    from backend.core import studs as studmod
    img, centres = _mesh_scene()
    assert detect.surface_is_dim(img) and not detect.surface_is_dim(np.full((100, 100, 3), 220, np.uint8))
    assert not detect.surface_is_dim(cv2.imread(str(SAMPLE)))
    a = detect.analyse(img)
    assert {p.source for p in a.pieces} <= {"surface", "colour"}               # the surface method leads
    assert count_per_brick([p.box for p in a.pieces], centres) == {n: 1 for n in centres}, a.boxes


def test_a_colour_box_holding_two_surface_boxes_is_replaced_by_them():
    shape = (900, 1600, 3)
    surface = [(100, 100, 150, 100), (400, 120, 160, 100), (700, 700, 20, 90)]            # the last is a sliver
    merged = (80, 80, 520, 160)
    assert detect.prefer_surface_over_merged([merged], surface, shape) == surface[:2]
    assert detect.prefer_surface_over_merged([(80, 80, 200, 160)], surface, shape) == [(80, 80, 200, 160)]   # holds only one
    assert detect.prefer_surface_over_merged([(690, 690, 60, 110)], surface, shape) == [(690, 690, 60, 110)]  # a sliver is no piece


def test_a_big_box_holding_pieces_of_clearly_different_colours_is_split_and_small_boxes_are_left_alone():
    img = np.full((600, 900, 3), 70, np.uint8)
    cv2.rectangle(img, (100, 200), (300, 300), (40, 40, 210), -1)               # red
    cv2.rectangle(img, (300, 150), (520, 230), (40, 220, 170), -1)               # lime, touching it
    small = [(600, 50, 60, 60), (700, 50, 60, 60), (600, 150, 60, 60)]
    big = (90, 140, 440, 170)
    out = detect.split_merged_by_colour(img, [big] + small)
    assert len(out) == 5 and all(b in out for b in small) and big not in out
    assert detect.split_merged_by_colour(img, [big, small[0]]) == [big, small[0]]         # too few boxes to know what is typical
    plain = np.full((600, 900, 3), 70, np.uint8)
    cv2.rectangle(plain, (100, 200), (520, 300), (40, 40, 210), -1)              # one red piece: nothing to split
    assert detect.split_merged_by_colour(plain, [(90, 190, 440, 120)] + small) == [(90, 190, 440, 120)] + small


def test_stud_detection_compensates_for_a_45_degree_view():
    from backend.core import studs as studmod
    from tests import stud_scenes as S
    top = S.sheet(emboss=8)
    S.piece(top, 150, 150, 4, 2, (40, 60, 200))
    S.piece(top, 600, 300, 3, 3, (60, 170, 70))
    squashed = cv2.resize(top, (top.shape[1], int(top.shape[0] * 0.7)), interpolation=cv2.INTER_AREA)   # 45 degrees: ~0.7 as tall
    tilt = __import__("dataclasses").replace(studmod.CFG, tilt_stretch=1.43)
    studs_, groups = studmod.find_stud_groups(squashed, tilt)
    assert sorted(g.dims for g in groups) == [(2, 4), (3, 3)]
    for g in groups:                                                           # mapped back into the squashed photo
        x, y, w, h = g.box
        assert 0 <= y and y + h <= squashed.shape[0] and all(0 <= s.y <= squashed.shape[0] for s in g.studs)
    first = min(groups, key=lambda g: g.centre[1])
    assert first.centre[1] == pytest.approx(0.7 * (150 + 28), abs=12)


def test_a_piece_that_only_just_reaches_the_photo_edge_with_a_corner_is_kept_but_a_long_contact_is_not():
    """The surface method ignores what the frame cuts off (the table beyond the sheet), but a piece whose corner touches
    the edge is whole: a real scan lost a brick whose tip reached the bottom edge."""
    from tests.test_reference import H, W, mesh
    img = mesh()
    corner = np.array([[300, 330], [420, 300], [440, 360], [370, H - 1], [310, 410]], np.int32)    # tip touches the bottom edge
    cv2.fillPoly(img, [corner], (40, 40, 210))
    cv2.rectangle(img, (600, 200), (700, 260), (200, 120, 30), -1)
    cv2.rectangle(img, (500, H - 60), (700, H), (50, 200, 150), -1)                                # sits on the edge: 200 px contact
    boxes = detect.find_pieces_by_surface(img)
    inside = lambda px, py: sum(1 for x, y, w, h in boxes if x <= px <= x + w and y <= py <= y + h)   # noqa: E731
    assert inside(360, 360) == 1 and inside(650, 230) == 1
    assert inside(600, H - 20) == 0


def test_the_same_dark_mesh_in_a_brighter_exposure_still_leads_with_the_surface_method_and_keeps_a_white_piece():
    """A real scan: the mesh photographed brighter (median brightness 115, not 70) was not 'dim', so the colour method ran and merged
    pieces. And a white slope lying against a yellow and a red brick must keep a box of its own when the merged box is split by colour."""
    from tests.test_reference import add_brick, mesh
    img = mesh()
    for x, y, w, h, c in [(500, 60, 60, 45, (200, 110, 30)), (640, 90, 60, 45, (50, 200, 150)), (520, 300, 60, 45, (200, 110, 30)),
                          (680, 300, 60, 45, (50, 200, 150)), (400, 200, 60, 45, (200, 110, 30))]:
        add_brick(img, x, y, w, h, c)
    add_brick(img, 100, 110, 70, 70, (232, 232, 236))                    # white, against the bricks below
    add_brick(img, 180, 120, 90, 55, (20, 200, 240), shadow=False)       # yellow
    add_brick(img, 80, 175, 200, 80, (30, 30, 205))                      # red brick touching both
    bright = np.clip(img.astype(np.float32) * 1.85, 0, 255).astype(np.uint8)
    assert detect.surface_is_dim(bright) and not detect.surface_is_dim(np.full((100, 100, 3), 170, np.uint8))
    boxes = [p.box for p in detect.analyse(bright).pieces]
    owners = lambda px, py: [b for b in boxes if b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]]    # noqa: E731
    white, yellow, red = owners(135, 135), owners(225, 148), owners(180, 235)
    assert white and yellow and red
    assert [b for b in white if not b[0] <= 180 <= b[0] + b[2] or b[1] + b[3] <= 235]      # a box for the white piece that stops short of the red brick


def test_a_piece_boxed_twice_becomes_one_box_but_a_small_piece_beside_a_big_one_is_kept():
    """A real scan listed a white slope twice (two boxes with an IoU of 0.97) and a blue 1x1 twice."""
    slope = [(198, 234, 137, 118), (198, 235, 137, 116)]
    blue = [(1041, 122, 66, 77), (1042, 135, 64, 66)]
    out = detect.dedupe_near_identical(slope + blue)
    assert len(out) == 2                                                  # one box per piece, each the union of its two boxes
    for pair in (slope, blue):
        assert any(all(b[0] >= o[0] and b[1] >= o[1] and b[0] + b[2] <= o[0] + o[2] and b[1] + b[3] <= o[1] + o[3] for b in pair) for o in out)
    big, small = (100, 100, 400, 300), (120, 120, 60, 60)                 # a small piece lying inside a big piece's box
    assert sorted(detect.dedupe_near_identical([big, small])) == sorted([big, small])
    neighbours = [(100, 100, 100, 80), (190, 100, 100, 80)]
    assert detect.dedupe_near_identical(neighbours) == neighbours


def test_two_different_coloured_pieces_a_few_pixels_apart_get_a_box_each():
    """A real scan: a grey 1x2 brick and a small blue 1x1, not touching, came out as one box (the surface method closes small
    gaps). A single piece with a printed patch of another colour must stay one box."""
    from tests.test_reference import add_brick, mesh
    img = np.clip(mesh().astype(np.float32) * 1.85, 0, 255).astype(np.uint8)        # dark mesh in a brighter exposure
    for x, y, w, h, c in [(60, 60, 80, 55, (30, 30, 205)), (200, 60, 90, 50, (50, 220, 170)), (560, 60, 80, 55, (30, 160, 235)),
                          (60, 300, 90, 55, (30, 30, 205)), (560, 300, 80, 55, (200, 110, 30))]:
        add_brick(img, x, y, w, h, c)
    for pt0, pt1 in (((330, 190), (410, 250)),):                                     # grey brick with the blue 1x1 8 px above it
        cv2.rectangle(img, pt0, pt1, (190, 190, 192), -1)
    cv2.rectangle(img, (350, 120), (398, 182), (200, 120, 30), -1)
    boxes = [p.box for p in detect.analyse(img).pieces]
    owners = lambda px, py: [b for b in boxes if b[0] <= px <= b[0] + b[2] and b[1] <= py <= b[1] + b[3]]    # noqa: E731
    grey, blue = owners(370, 220), owners(374, 150)
    assert grey and blue and not set(grey) & set(blue), (grey, blue)                  # no box holds both
    # a red piece with a small white print on it is one piece
    one = np.clip(mesh().astype(np.float32) * 1.85, 0, 255).astype(np.uint8)
    for x, y, w, h, c in [(60, 60, 80, 55, (200, 110, 30)), (200, 60, 90, 50, (50, 220, 170)), (560, 60, 80, 55, (30, 160, 235)),
                          (560, 300, 80, 55, (200, 110, 30)), (60, 300, 90, 55, (50, 220, 170))]:
        add_brick(one, x, y, w, h, c)
    cv2.rectangle(one, (330, 170), (460, 240), (30, 30, 205), -1)
    cv2.rectangle(one, (370, 190), (390, 205), (240, 240, 240), -1)
    printed = [p.box for p in detect.analyse(one).pieces if p.box[0] <= 395 <= p.box[0] + p.box[2] and p.box[1] <= 200 <= p.box[1] + p.box[3]]
    assert len(printed) == 1
