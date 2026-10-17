import dataclasses
import time

import cv2
import numpy as np
import pytest

from backend.core import detect, gemini_detect as gd, shadow, studs
from backend.core.studs import Stud, StudInfo
from tests import stud_scenes as S

R = 24.0
PITCH = studs.CFG.pitch_ratio * R                 # 80 px


def grid(x, y, cols, rows, lab=(150, 128, 128), angle=0.0, conf=0.9, r=R, pitch=PITCH):
    th = np.radians(angle)
    ax, ay = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    return [Stud(float(x + (ax * c + ay * rr)[0] * pitch), float(y + (ax * c + ay * rr)[1] * pitch), r, lab, conf)
            for rr in range(rows) for c in range(cols)]


SHAPE = (1000, 1600, 3)


# ---------------------------------------------------------------- grouping
def test_a_stud_grid_becomes_one_group_with_pitch_orientation_dimensions_and_box():
    groups = studs.group_studs(grid(300, 300, 4, 2), shape=SHAPE)
    assert len(groups) == 1
    g = groups[0]
    assert g.dims == (2, 4) and len(g.studs) == 8
    assert g.pitch == pytest.approx(PITCH, rel=0.02) and g.angle == pytest.approx(0, abs=1) or g.angle == pytest.approx(90, abs=1)
    x, y, w, h = g.box                                  # circles expanded by half a pitch on every side
    assert (x, y) == (int(300 - R - PITCH / 2), int(300 - R - PITCH / 2))
    assert w == pytest.approx(3 * PITCH + 2 * R + PITCH, abs=3) and h == pytest.approx(PITCH + 2 * R + PITCH, abs=3)


def test_a_turned_grid_is_one_group_and_its_orientation_is_measured():
    g, = studs.group_studs(grid(500, 300, 3, 3, angle=30), shape=SHAPE)
    assert g.dims == (3, 3) and g.angle == pytest.approx(30, abs=2) and g.aligned > 0.9


def test_two_touching_pieces_of_different_colours_are_two_groups():
    red, blue = (120, 190, 170), (110, 120, 70)                      # Lab-ish: clearly different chroma
    one = grid(300, 300, 3, 2, lab=red)
    two = grid(300 + 3 * PITCH, 300, 3, 2, lab=blue)                  # next lattice position: studs exactly one pitch away
    groups = studs.group_studs(one + two, shape=SHAPE)
    assert sorted(len(g.studs) for g in groups) == [6, 6]
    assert studs.distinct_pieces(groups[0], groups[1])


def test_same_colour_pieces_are_distinct_only_when_their_grids_disagree_and_both_are_big_enough():
    off = (300 + 3.5 * PITCH, 300 + 0.5 * PITCH)                       # half a pitch off: not one lattice
    ga = studs.group_studs(grid(300, 300, 3, 2), shape=SHAPE)[0]
    gb = studs.group_studs(grid(*off, 4, 2), shape=SHAPE)[0]
    assert not studs.aligned_with(ga, gb) and studs.distinct_pieces(ga, gb)
    small = studs.group_studs(grid(*off, 2, 1), shape=SHAPE)[0]        # 2 studs prove nothing (e.g. studs on a side face)
    assert not studs.distinct_pieces(ga, small)
    on_lattice = studs.group_studs(grid(300 + 3 * PITCH, 300, 2, 2), shape=SHAPE)[0]   # the next studs of ga's own lattice
    assert studs.aligned_with(ga, on_lattice) and not studs.distinct_pieces(ga, on_lattice)
    # linked into one group when they really are one lattice
    assert len(studs.group_studs(grid(300, 300, 3, 2) + grid(300 + 3 * PITCH, 300, 2, 2), shape=SHAPE)) == 1


def test_a_1x1_needs_a_very_confident_stud_and_a_lone_false_circle_is_rejected():
    assert studs.group_studs([Stud(500, 500, R, (150, 128, 128), 0.95)], shape=SHAPE)[0].dims == (1, 1)
    assert studs.group_studs([Stud(500, 500, R, (150, 128, 128), 0.6)], shape=SHAPE) == []
    scene = grid(300, 300, 2, 2) + [Stud(1000, 700, R, (150, 128, 128), 0.6)]
    groups = studs.group_studs(scene, shape=SHAPE)
    assert [len(g.studs) for g in groups] == [4]


def test_a_background_coloured_lattice_is_texture_not_a_piece():
    lattice = grid(100, 100, 12, 10)
    for s_ in lattice:
        s_.on_background = True
    assert studs.group_studs(lattice, shape=SHAPE) == []                              # > bg_group_max studs
    pair = grid(100, 100, 2, 1)
    for s_ in pair:
        s_.on_background = True
    assert studs.group_studs(pair, shape=SHAPE) == []                                 # pale-on-pale needs >= bg_min_group


def test_stud_info_for_a_box():
    groups = studs.group_studs(grid(300, 300, 4, 2), shape=SHAPE)
    info = studs.stud_info_for_box((200, 200, 500, 300), groups)
    assert info.stud_count == 8 and info.dims == (2, 4) and info.pitch_px == pytest.approx(PITCH, rel=0.02)
    assert info.radius_px == R and 0 < info.score <= 1
    assert studs.stud_info_for_box((900, 700, 100, 100), groups) == StudInfo()


# ---------------------------------------------------------------- finding studs in images
def scene():
    img = S.sheet(emboss=8)
    S.piece(img, 150, 150, 4, 2, (215, 222, 226), shade=0.15)        # pale warm grey
    S.piece(img, 600, 150, 3, 3, (60, 170, 70))                      # green
    S.piece(img, 200, 450, 2, 4, (222, 228, 230), shade=0.15, angle=15)   # near white, turned
    S.piece(img, 650, 480, 4, 2, (140, 80, 40))                      # blue
    return img


def test_studs_are_found_on_every_piece_whatever_its_colour_and_none_on_embossed_paper():
    img = scene()
    groups = studs.group_studs(studs.find_studs(img), shape=img.shape)
    assert sorted(g.dims for g in groups) == [(2, 4), (2, 4), (2, 4), (3, 3)]
    assert sorted(len(g.studs) for g in groups) == [8, 8, 8, 9]
    blank = S.sheet(emboss=8)
    assert studs.find_studs(blank) == []                              # quilted paper alone: no studs


def test_a_pale_piece_on_a_pale_sheet_is_found_by_the_colour_method_through_its_studs():
    img = S.sheet(emboss=8)
    S.piece(img, 300, 300, 4, 2, (222, 226, 228), shade=0.18)
    groups = studs.group_studs(studs.find_studs(img), shape=img.shape)
    assert len(groups) == 1
    assert detect.find_pieces_by_colour(img) == []                    # no colour seed: colour alone sees nothing
    report = []
    boxes = detect.find_pieces_by_colour(img, stud_groups=groups, report=report)
    assert len(boxes) == 1 and report == ["studs-only"]
    x, y, w, h = boxes[0]
    assert x < 300 < x + w and y < 300 < y + h and x + w > 300 + 3 * S.PITCH


def test_stud_groups_join_the_colour_seed_they_sit_on():
    img = scene()
    groups = studs.group_studs(studs.find_studs(img), shape=img.shape)
    report = []
    boxes = detect.find_pieces_by_colour(img, stud_groups=groups, report=report)
    assert len(boxes) == 4 and report.count("colour+studs") >= 2     # coloured pieces: seeds and studs are one group
    assert "studs-only" in report                                     # pale pieces: studs alone


def test_pale_pieces_found_in_parts_become_one_piece_each():
    """The background method boxes a pale piece AND every stud with a shaded rim; the stud groups say they are one."""
    img = scene()
    off = dataclasses.replace(studs.CFG, enabled=False)
    before = detect.analyse(img, cfg=off)
    after = detect.analyse(img)
    assert len(before.pieces) >= 4 and len(after.pieces) == 4          # the mask merge pass joins the fragments even without studs
    assert sorted(p.studs.dims for p in after.pieces) == [(2, 4), (2, 4), (2, 4), (3, 3)]
    assert all(p.studs.stud_count >= 8 and p.studs.pitch_px == pytest.approx(S.PITCH, rel=0.1) for p in after.pieces)


def test_merge_stud_fragments_rules():
    groups = studs.group_studs(grid(300, 300, 4, 2), shape=SHAPE)
    whole = (200, 200, 400, 250)
    frag = (290, 290, 40, 40)                                         # one stud, inside the whole box
    other = (900, 200, 100, 100)
    assert detect.merge_stud_fragments([whole, frag, other], groups, SHAPE) == detect._reading_order([whole, other])
    parts = [(250, 250, 90, 90), (330, 250, 90, 90), (410, 250, 90, 90)]   # no whole box: replaced by the group's box
    merged = detect.merge_stud_fragments(parts, groups, SHAPE)
    assert len(merged) == 1 and merged[0][2] > 300


def test_split_by_studs_splits_only_what_the_studs_prove():
    red, blue = (120, 190, 170), (110, 120, 70)
    groups = studs.group_studs(grid(200, 300, 3, 2, lab=red) + grid(200 + 3 * PITCH, 300, 3, 2, lab=blue), shape=SHAPE)
    both = (100, 200, 700, 300)
    out = detect.split_by_studs([both], groups, SHAPE)
    assert len(out) == 2 and all(b[2] < 400 for b in out)
    one = studs.group_studs(grid(200, 300, 3, 2), shape=SHAPE)
    assert detect.split_by_studs([(100, 200, 400, 300)], one, SHAPE) == [(100, 200, 400, 300)]
    # a box whose groups cover less than half of it keeps its other contents (e.g. a tile beside two bricks)
    assert detect.split_by_studs([(0, 0, 1600, 1000)], groups, SHAPE) == [(0, 0, 1600, 1000)]
    # combine_boxes uses it for boxes of either method
    assert len(detect.combine_boxes([both], [], SHAPE, stud_groups=groups)) == 2


# ---------------------------------------------------------------- studs as a signal, not a filter
def test_boxes_without_studs_are_kept_unless_small_and_weak():
    img = np.full((1000, 1600, 3), 235, np.uint8)
    cv2.rectangle(img, (600, 400), (700, 460), (30, 30, 200), -1)                       # coloured, no studs: a tile
    pieces = [detect.Piece((600, 400, 100, 60), "background"),                          # kept: strong colour
              detect.Piece((100, 100, 30, 30), "surface"),                              # small, grey, no studs: a blob
              detect.Piece((900, 500, 200, 200), "surface"),                            # weak but not small: kept
              detect.Piece((100, 700, 30, 30), "colour+studs", StudInfo(2, 80, (1, 2), 24, 0.9))]   # small but has studs
    kept = detect._drop_blobs(pieces, img, studs.CFG)
    assert [p.box for p in kept] == [(600, 400, 100, 60), (900, 500, 200, 200), (100, 700, 30, 30)]


# ---------------------------------------------------------------- never fail, kill switch, debug
def test_a_stud_failure_never_fails_the_scan(monkeypatch):
    img = scene()

    def boom(*a, **k):
        raise RuntimeError("stud detector exploded")
    monkeypatch.setattr(studs, "find_studs", boom)
    a = detect.analyse(img)
    assert a.studs == [] and len(a.pieces) > 0 and all(p.studs == StudInfo() for p in a.pieces)
    monkeypatch.undo()

    def boom2(*a, **k):
        raise RuntimeError("stud-assisted step exploded")
    monkeypatch.setattr(detect.studmod, "stud_info_for_box", boom2)
    assert len(detect.analyse(img).pieces) > 0
    monkeypatch.undo()
    monkeypatch.setattr(detect, "split_by_studs", boom2)
    assert len(detect.analyse(img).pieces) > 0


def test_stud_detection_can_be_switched_off():
    off = dataclasses.replace(studs.CFG, enabled=False)
    a = detect.analyse(scene(), cfg=off)
    assert a.studs == [] and a.groups == [] and a.timings["studs"] < 0.01


def test_debug_overlay_is_written(tmp_path):
    img = scene()
    out = tmp_path / "overlay.jpg"
    detect.analyse(img, debug=out)
    shown = cv2.imread(str(out))
    assert shown is not None and shown.shape == img.shape
    assert int(np.abs(shown.astype(int) - img.astype(int)).sum()) > 0
    folder = tmp_path / "dbg"
    detect.analyse(img, debug=folder)
    assert len(list(folder.glob("detect_*.jpg"))) == 1


def test_debug_directory_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("DETECT_DEBUG_DIR", str(tmp_path / "env"))
    detect.analyse(scene())
    assert len(list((tmp_path / "env").glob("*.jpg"))) == 1


def test_find_pieces_still_returns_plain_boxes():
    boxes = detect.find_pieces(scene())
    assert all(isinstance(b, tuple) and len(b) == 4 for b in boxes) and len(boxes) == 4


# ---------------------------------------------------------------- downstream
def test_detection_carries_stud_info_per_box():
    import asyncio
    d = asyncio.run(gd.detect_pieces(scene(), "opencv", None))
    assert len(d.stud_info) == len(d.boxes) == 4 and all(i.stud_count >= 8 for i in d.stud_info)
    d2 = asyncio.run(gd.detect_pieces(scene(), "opencv", None, opencv=lambda im: [(10, 10, 50, 50)]))
    assert d2.stud_info == []                                                           # a boxes-only detector has none


def test_shadow_uses_the_measured_pitch_as_its_ruler():
    img = S.sheet()
    cv2.rectangle(img, (300, 300), (500, 420), (40, 60, 200), -1)                      # a red piece, 200 px wide
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2].astype(np.float32), (0, 0), 2)
    plain = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 110)
    img2 = img.copy()
    img2[300:420, 230:300] = (150, 150, 150)                                            # a 70 px shadow on its left
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    v2 = cv2.GaussianBlur(hsv2[:, :, 2].astype(np.float32), (0, 0), 2)
    plain2 = (hsv2[:, :, 1] < 50) & (hsv2[:, :, 2] > 110)
    own = shadow._measure(hsv2, v2, plain2, (250, 280, 280, 160), (2, 4))["left"]
    measured = shadow._measure(hsv2, v2, plain2, (250, 280, 280, 160), (2, 4), pitch_px=50.0)["left"]
    assert measured == pytest.approx(70 / (50.0 / shadow.STUD_PITCH_MM), rel=0.15)
    assert own != pytest.approx(measured, rel=0.05)                                     # a different ruler gives a different answer


def test_shadow_analyse_accepts_stud_info():
    img = S.sheet()
    assert shadow.analyse(img, [(10, 10, 100, 100)], ["Brick 2 x 4"], stud_info=[StudInfo(8, 56, (2, 4), 17, 0.9)]) == [None]


# ---------------------------------------------------------------- speed
def test_stud_detection_is_fast_on_a_1600_px_photo():
    img = cv2.resize(scene(), (1600, 1120))
    studs.find_studs(img)                                                               # warm up
    start = time.perf_counter()
    found = studs.find_studs(img)
    elapsed = time.perf_counter() - start
    assert len(found) >= 25
    assert elapsed < 0.6, f"find_studs took {elapsed * 1000:.0f} ms"                    # target 300 ms on the server; slack for CI
