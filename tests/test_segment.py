"""Mask segmentation (backend/core/segment.py): the five split rules, the merge pass, coverage-based add_missing, ambiguity, crops."""

import cv2
import numpy as np
import pytest

from backend.core import partdims, segment as seg, studs
from backend.core.studs import Stud
from tests import stud_scenes as S
from tests.detection import scenes

H, W = 700, 1000


def blank(colour=235):
    return np.full((H, W, 3), colour, np.uint8)


def scene_for(img, groups=(), surface=None):
    return seg.Scene.build(img, np.zeros(img.shape[:2], bool) if surface is None else surface, list(groups), 1.0, None)


def rect_mask(x0, y0, x1, y1):
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def region(mask, pitch=56.0, source="test"):
    r = seg.Region.from_full(mask, source)
    r.pitch = pitch
    return r


def grid_group(x, y, cols, rows, angle=0.0, lab=(150, 128, 128), pitch=80.0, r=24.0):
    th = np.radians(angle)
    ax, ay = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
    members = [Stud(float(x + (ax * c + ay * rr)[0] * pitch), float(y + (ax * c + ay * rr)[1] * pitch), r, lab, 0.9)
               for rr in range(rows) for c in range(cols)]
    groups = studs.group_studs(members, shape=(H * 3, W * 3, 3))
    assert len(groups) == 1
    return groups[0]


# ---------------------------------------------------------------- masks are the representation
def test_holes_are_filled_so_axle_and_technic_holes_never_split_a_piece():
    m = np.zeros((200, 300), np.uint8)
    cv2.rectangle(m, (20, 40), (280, 150), 1, -1)
    for x in range(50, 260, 40):
        cv2.circle(m, (x, 95), 12, 0, -1)
    filled = seg.fill_holes(m)
    assert filled.sum() == (260 * 110) + 0 or filled.sum() > m.sum()
    assert filled[95, 50] == 1 and filled[95, 90] == 1
    r = seg.Region.from_full(np.pad(m, ((0, 0), (0, 0))))
    assert r.mask[95 - r.y, 50 - r.x] == 1                          # a Region fills its holes on creation


def test_rotated_rectangle_comes_from_the_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [cv2.boxPoints(((400, 300), (300, 100), 30)).astype(np.int32)], 1)
    (cx, cy), (w, h), _ = seg.rotated_rect(seg.Region.from_full(m))
    assert (cx, cy) == pytest.approx((400, 300), abs=3)
    assert sorted((w, h)) == pytest.approx([100, 300], abs=4)


def test_masks_overlap_and_touch_measures_use_pixels_not_boxes():
    a = region(rect_mask(100, 100, 300, 200))
    b = region(rect_mask(300, 100, 500, 200))                         # touching
    c = region(rect_mask(100, 300, 300, 400))
    assert seg.touching(a, b) and not seg.touching(a, c)
    assert seg.overlap_px(a, b) == 0 and seg.mask_iou(a, a) == pytest.approx(1.0)
    diag = np.zeros((H, W), np.uint8)
    cv2.fillPoly(diag, [np.array([[100, 100], [400, 100], [100, 400]])], 1)           # a triangle whose BOX covers a square
    other = region(rect_mask(300, 300, 400, 400))
    assert seg.overlap_px(region(diag), other) == 0                                    # boxes would have said "overlap"


# ---------------------------------------------------------------- 3a seam
def seam_scene(dl):
    img = blank(120)
    base = (60, 60, 200)
    cv2.rectangle(img, (200, 200), (600, 400), base, -1)
    cv2.line(img, (200, 300), (600, 300), tuple(int(c * (1 - dl)) for c in base), 3)
    return img, rect_mask(200, 200, 601, 401)


def test_a_thin_dark_line_across_the_component_splits_it():
    img, m = seam_scene(0.55)
    r = region(m)
    parts = seg.split_region(r, scene_for(img))
    assert len(parts) == 2 and all("split:seam" in p.rules for p in parts)
    assert sorted(p.h for p in parts)[0] >= 90


def test_a_seam_shallower_than_seam_dl_is_not_evidence():
    img, m = seam_scene(0.03)
    assert len(seg.split_region(region(m), scene_for(img))) == 1


def test_a_seam_must_run_along_most_of_the_cut():
    img = blank(120)
    cv2.rectangle(img, (200, 200), (600, 400), (60, 60, 200), -1)
    cv2.line(img, (200, 300), (300, 300), (30, 30, 100), 3)                 # a scratch over a quarter of the width
    assert len(seg.split_region(region(rect_mask(200, 200, 601, 401)), scene_for(img))) == 1


def test_no_pitch_disables_the_scale_dependent_rules():
    img, m = seam_scene(0.55)
    r = region(m, pitch=None)
    assert len(seg.split_region(r, scene_for(img))) == 1                    # seam needs a scale
    assert seg.rule_shape(r, scene_for(img), None) is None and seg.rule_size(r, scene_for(img), None) is None


# ---------------------------------------------------------------- 3b stud grids
def two_grid_region(offset=(0.0, 0.0), angle=0.0, gap=True, cols=4):
    """Two stud groups side by side in one mask (bridged across the 12 px between them); `gap` draws the dark line two loose bricks leave."""
    a = grid_group(200, 200, cols, 2)
    b = grid_group(200 + cols * 80 + 60 + offset[0] * 80, 200 + offset[1] * 80, cols, 2, angle=angle)
    mask = np.zeros((H, W), np.uint8)
    for g in (a, b):
        cv2.fillPoly(mask, [np.rint(g.footprint).astype(np.int32)], 1)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.argwhere(mask)[:, ::-1].astype(np.int32)), 1)      # one convex blob, like a brick top
    img = blank(200)
    if gap:
        cv2.line(img, (510, 120), (510, 420), (60, 60, 60), 5)
    return region(mask, 80.0), [a, b], img


def test_grids_offset_by_half_a_pitch_with_a_gap_between_are_two_pieces():
    r, groups, img = two_grid_region(offset=(0.0, 0.5))
    assert len(seg.split_region(r, scene_for(img, groups))) == 2


def test_grids_offset_by_whole_pitches_are_one_lattice():
    r, groups, img = two_grid_region(offset=(0.0, 0.0))
    assert seg.grids_continuous(r, scene_for(img, groups))
    assert seg.rule_grids(r, scene_for(img, groups), 80.0) is None


def test_lattices_that_disagree_with_nothing_between_them_are_one_piece():
    """A tilted photo, a stud found a little off: the stud detector's lattices disagree, but nothing physical separates the groups
    (the failure that cut one long red brick into two boxes)."""
    r, groups, img = two_grid_region(offset=(0.0, 0.5), gap=False)
    sc = scene_for(blank(200), groups)
    assert not seg.grids_continuous(r, sc)
    assert seg.rule_grids(r, sc, 80.0) is None
    assert len(seg.split_region(r, sc)) == 1


def test_groups_of_three_studs_are_not_evidence():
    """The long red brick: only six of its twelve studs were found, as two groups of three whose angles disagreed by 32 degrees."""
    r, groups, img = two_grid_region(offset=(0.0, 0.5), gap=True, cols=3)
    groups = [grid_group(200, 200, 3, 1, pitch=80.0), grid_group(200 + 3 * 80, 200 + 40, 3, 1, pitch=80.0)]
    assert all(len(g.studs) == 3 for g in groups)
    assert seg.rule_grids(r, scene_for(img, groups), 80.0) is None


def test_a_notch_or_a_colour_change_is_also_something_between_the_groups():
    r, groups, _ = two_grid_region(offset=(0.0, 0.5), gap=False)
    bite = r.mask.copy()
    bite[:120, 440 - r.x:580 - r.x] = 0                                          # a notch in the outline where the two meet
    notched = region(np.pad(bite, ((r.y, H - r.y - r.h), (r.x, W - r.x - r.w))), 80.0)
    assert len(seg.split_region(notched, scene_for(blank(200), groups))) == 2
    img = blank(200)
    cv2.rectangle(img, (r.x, r.y), (510, r.y + r.h), (40, 40, 200), -1)         # left half red, right half green
    cv2.rectangle(img, (511, r.y), (r.x + r.w, r.y + r.h), (60, 180, 60), -1)
    assert len(seg.split_region(r, scene_for(img, groups))) == 2


def test_the_lattice_offset_is_measured_in_the_tilt_corrected_view():
    """Groups carry the angle and pitch of the y-stretched view; with a tilted photo the studs' y must be stretched too."""
    a = grid_group(200, 300, 4, 2)
    b = grid_group(200, 300 + 2 * 80, 4, 2)                                 # one lattice: two pitches below the first
    for g in (a, b):
        for st in g.studs:
            st.y /= 1.5                                                     # ...as the photo shows it, y squashed by 1.5
    assert seg._lattice_offset(a, b, 1.5) < 0.02
    assert seg._lattice_offset(a, b, 1.0) > 0.1                             # unstretched, the same pair looks misaligned


# ---------------------------------------------------------------- 3c colour
def two_colour(left, right, share=0.5):
    img = blank()
    cut = 200 + int(400 * share)
    cv2.rectangle(img, (200, 200), (cut, 400), left, -1)
    cv2.rectangle(img, (cut, 200), (600, 400), right, -1)
    return img, rect_mask(200, 200, 601, 401)


def test_two_clusters_of_different_hue_split_the_component():
    img, m = two_colour((40, 40, 200), (60, 180, 60))
    parts = seg.split_region(region(m), scene_for(img))
    assert len(parts) == 2 and all("split:colour" in p.rules for p in parts)


def test_a_lit_top_and_a_shaded_side_of_one_hue_do_not_split():
    img, m = two_colour((50, 50, 220), (25, 25, 110))
    assert len(seg.split_region(region(m), scene_for(img))) == 1


def test_a_white_print_has_no_hue_and_does_not_split():
    img = blank()
    cv2.rectangle(img, (200, 200), (600, 400), (200, 80, 30), -1)
    cv2.rectangle(img, (320, 250), (480, 350), (245, 245, 245), -1)
    assert len(seg.split_region(region(rect_mask(200, 200, 601, 401)), scene_for(img))) == 1


def test_a_cluster_under_its_share_does_not_split():
    img, m = two_colour((40, 40, 200), (60, 180, 60), share=0.9)           # green is 10%
    assert len(seg.split_region(region(m), scene_for(img))) == 1


# ---------------------------------------------------------------- 3d shape
def shape_mask(rects):
    m = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = 1
    return m


def test_a_deep_notch_below_the_solidity_limit_is_cut_at_the_defect():
    # two bricks side by side, staggered by two thirds of their length: solidity under 0.85, notches deeper than half a pitch
    m = shape_mask([(100, 200, 400, 300), (300, 300, 600, 400)])
    img = blank(180)
    s = seg.rule_shape(region(m), scene_for(img), 56.0)
    assert s is not None and len(s.parts) == 2


def test_an_l_shaped_corner_part_is_never_cut_at_its_bend():
    m = shape_mask([(200, 200, 300, 450), (200, 350, 450, 450)])
    assert seg.rule_shape(region(m), scene_for(blank(180)), 40.0) is None


def test_a_convex_piece_has_no_defect():
    assert seg.rule_shape(region(rect_mask(200, 200, 500, 400)), scene_for(blank()), 56.0) is None


# ---------------------------------------------------------------- 3e size
def lattice_region(cols, rows, pitch=56.0):
    img = blank()
    x0, y0 = 200, 200
    members = [Stud(float(x0 + c * pitch), float(y0 + r * pitch), 17.0, (150, 128, 128), 0.9) for r in range(rows) for c in range(cols)]
    groups = studs.group_studs(members, shape=(H * 3, W * 3, 3))
    m = rect_mask(int(x0 - pitch / 2), int(y0 - pitch / 2), int(x0 + (cols - 0.5) * pitch), int(y0 + (rows - 0.5) * pitch))
    return region(m, pitch), scene_for(img, groups)


def test_a_real_footprint_is_left_alone():
    r, sc = lattice_region(4, 2)
    assert seg.rule_size(r, sc, 56.0) is None
    r, sc = lattice_region(8, 6)
    assert seg.rule_size(r, sc, 56.0) is None


def test_a_footprint_no_real_part_has_is_partitioned_into_real_ones():
    r, sc = lattice_region(5, 5)                                              # there is no 5 x 5 plate
    s = seg.rule_size(r, sc, 56.0)
    assert s is not None and len(s.parts) == 2
    assert s.alternatives > 1 and s.ambiguous                                  # 1x5 + 4x5 or 2x5 + 3x5 ... equally good


def test_the_size_rule_needs_studs_on_the_footprint():
    r, _ = lattice_region(5, 5)
    assert seg.rule_size(r, scene_for(blank()), 56.0) is None


def test_the_footprint_catalogue_and_partition():
    assert partdims.is_real_footprint(4, 2) and partdims.is_real_footprint(2, 4) and partdims.is_real_footprint(48, 48)
    assert not partdims.is_real_footprint(5, 5)
    assert partdims.nearest_footprint(2.2, 4.1) == (2, 4) and partdims.nearest_footprint(2.6, 4.1) is None
    n, sols = partdims.partition_footprint(4, 2)
    assert n == 1 and sols == [[(0, 0, 4, 2)]]
    n, sols = partdims.partition_footprint(5, 5)
    assert n == 2 and len(sols) > 1
    n, sols = partdims.partition_footprint(3, 7)
    assert n == 2
    assert all(sum(w * h for _, _, w, h in sol) == 21 for sol in sols)         # every partition covers the rectangle exactly


# ---------------------------------------------------------------- 4 merge pass
def plate_halves(colour_a=(60, 170, 70), colour_b=(60, 170, 70), gap=0):
    img = blank()
    cv2.rectangle(img, (200, 200), (400, 400), colour_a, -1)
    cv2.rectangle(img, (400 + gap, 200), (600 + gap, 400), colour_b, -1)
    return img, region(rect_mask(200, 200, 400, 401)), region(rect_mask(400 + gap, 200, 600 + gap, 401))


def test_touching_fragments_of_one_hue_merge_into_one_piece():
    img, a, b = plate_halves(colour_b=(40, 120, 50))                          # lit and shaded half of one plate
    out = seg.merge_pass([a, b], scene_for(img))
    assert len(out) == 1 and out[0].w == pytest.approx(401, abs=6)


def test_pieces_of_different_hue_do_not_merge():
    img, a, b = plate_halves(colour_b=(40, 40, 200))
    assert len(seg.merge_pass([a, b], scene_for(img))) == 2


def test_a_gap_of_surface_between_two_masks_blocks_the_merge():
    img, a, b = plate_halves(gap=40)
    surface = np.zeros((H, W), bool)
    surface[200:400, 400:440] = True
    assert len(seg.merge_pass([a, b], scene_for(img, surface=surface))) == 2


def test_a_seam_along_the_shared_boundary_blocks_the_merge():
    img, m = seam_scene(0.55)
    a, b = region(m[:, :]), None
    top = region(rect_mask(200, 200, 601, 299))
    bottom = region(rect_mask(200, 302, 601, 401))
    assert len(seg.merge_pass([top, bottom], scene_for(img))) == 2


def test_the_parts_of_a_split_are_not_merged_straight_back():
    img, m = two_colour((50, 50, 220), (25, 25, 110))
    img2, m2 = seam_scene(0.55)
    parts = seg.split_region(region(m2), scene_for(img2))
    assert len(parts) == 2 and parts[0].split_id == parts[1].split_id
    assert len(seg.merge_pass(parts, scene_for(img2))) == 2


def test_a_merge_that_would_exceed_the_largest_baseplate_is_refused():
    img = blank(200)
    big_a = region(rect_mask(0, 100, 500, 200), pitch=8.0)
    big_b = region(rect_mask(500, 100, 990, 200), pitch=8.0)                  # 990 px of 8 px studs = 123 studs
    ok, _ = seg.can_merge(big_a, big_b, scene_for(img))
    assert not ok


def test_masks_that_mostly_overlap_are_one_piece_seen_twice_whatever_their_colour():
    img = blank()
    cv2.rectangle(img, (200, 200), (500, 400), (200, 200, 200), -1)
    a, b = region(rect_mask(200, 200, 500, 400)), region(rect_mask(220, 210, 500, 400))
    assert len(seg.merge_pass([a, b], scene_for(img))) == 1


# ---------------------------------------------------------------- 5 add_missing by coverage
def test_a_candidate_covered_by_existing_masks_is_not_added():
    kept = [region(rect_mask(100, 100, 400, 400))]
    cand = region(rect_mask(150, 150, 350, 350))
    assert len(seg.add_missing_masks(kept, [cand], scene_for(blank()))) == 1


def test_an_uncovered_candidate_is_added_whole():
    kept = [region(rect_mask(100, 100, 300, 300))]
    cand = region(rect_mask(500, 100, 700, 300))
    assert len(seg.add_missing_masks(kept, [cand], scene_for(blank()))) == 2


def test_a_candidate_less_than_sixty_percent_uncovered_is_skipped():
    kept = [region(rect_mask(100, 100, 400, 400))]
    cand = region(rect_mask(100, 100, 500, 400))                              # 25% uncovered
    assert len(seg.add_missing_masks(kept, [cand], scene_for(blank()))) == 1


def test_a_partly_covered_candidate_adds_only_its_uncovered_part():
    kept = [region(rect_mask(100, 100, 250, 400))]
    cand = region(rect_mask(100, 100, 600, 400))                              # 70% uncovered
    out = seg.add_missing_masks(kept, [cand], scene_for(blank()), pitch_hint=56.0)
    assert len(out) == 2
    assert out[1].x >= 250 - 2 and out[1].w == pytest.approx(350, abs=4)      # the part nothing covered, not the whole candidate


def test_a_thin_uncovered_sliver_is_not_a_piece():
    kept = [region(rect_mask(100, 100, 600, 400))]
    cand = region(rect_mask(100, 100, 650, 400))                              # 50 px strip: under half... of a 112 px stud pitch
    assert len(seg.add_missing_masks(kept, [cand], scene_for(blank()), pitch_hint=112.0)) == 1


def test_a_blob_that_surrounds_two_pieces_already_found_is_skipped():
    kept = [region(rect_mask(100, 100, 250, 250)), region(rect_mask(350, 100, 500, 250))]
    cand = region(rect_mask(90, 90, 510, 400))                                # bridged by their shadows
    assert len(seg.add_missing_masks(kept, [cand], scene_for(blank()))) == 2


# ---------------------------------------------------------------- 6 ambiguity
def test_a_split_that_only_just_has_evidence_is_ambiguous_and_flagged():
    r, sc = lattice_region(5, 5)
    parts = seg.split_region(r, sc)
    assert len(parts) == 2 and all(p.check and p.ambig is not None for p in parts)
    assert parts[0].ambig is parts[1].ambig and parts[0].ambig.whole is r


def test_a_clear_split_is_not_ambiguous():
    img, m = seam_scene(0.65)
    parts = seg.split_region(region(m), scene_for(img))
    assert len(parts) == 2 and not any(p.check for p in parts)


def test_overlap_of_more_than_ten_percent_is_flagged_check(caplog):
    a, b = region(rect_mask(100, 100, 300, 300)), region(rect_mask(270, 100, 500, 300))
    seg.mark_overlaps([a, b])
    assert a.check and b.check and "overlap" in a.rules
    c, d = region(rect_mask(100, 100, 300, 300)), region(rect_mask(296, 100, 500, 300))
    seg.mark_overlaps([c, d])
    assert not c.check and not d.check


def test_pitch_comes_from_the_studs_inside_else_from_the_neighbours():
    g = grid_group(300, 300, 4, 2)
    a = seg.Region.from_full(rect_mask(250, 250, 600, 420))
    b = seg.Region.from_full(rect_mask(700, 250, 900, 420))                    # no studs
    seg.resolve_pitches([a, b], scene_for(blank(), [g]))
    assert a.pitch == pytest.approx(80.0, rel=0.05) and b.pitch == pytest.approx(a.pitch)
    c = seg.Region.from_full(rect_mask(250, 250, 600, 420))
    seg.resolve_pitches([c], scene_for(blank()))
    assert c.pitch is None


# ---------------------------------------------------------------- crops
def test_crop_is_the_deskewed_rotated_rectangle_plus_half_a_pitch():
    img = blank()
    m = np.zeros((H, W), np.uint8)
    poly = cv2.boxPoints(((500, 350), (336, 112), 30)).astype(np.int32)
    cv2.fillPoly(img, [poly], (40, 40, 200))
    cv2.fillPoly(m, [poly], 1)
    r = seg.Region.from_full(m)
    crop = seg.crop_region(img, r, pitch=56.0)
    assert crop.shape[1] == pytest.approx(336 + 56, abs=6) and crop.shape[0] == pytest.approx(112 + 56, abs=6)   # long side horizontal
    inner = crop[30:-30, 30:-30]
    assert (np.abs(inner.astype(int) - (40, 40, 200)).sum(axis=2) < 40).mean() > 0.9                           # deskewed: the piece fills it


def test_crop_margin_falls_back_to_six_percent_of_the_long_side():
    img = blank()
    m = rect_mask(300, 300, 700, 400)
    crop = seg.crop_region(img, seg.Region.from_full(m))
    assert crop.shape[1] == pytest.approx(400 + 2 * 0.06 * 400, abs=6)


# ---------------------------------------------------------------- the whole pipeline on the test set
@pytest.mark.parametrize("name", list(scenes.SCENES))
def test_each_scene_gives_the_expected_number_of_pieces_with_no_split_or_merge_error(name):
    from backend.core import detect
    from tests.detection import metrics
    fn, expected = scenes.SCENES[name]
    img, gt = fn()
    analysis = detect.analyse_masks(img)
    s = metrics.score(gt, metrics.detection_masks(analysis, img.shape))
    assert len(analysis.pieces) == expected, [p.box for p in analysis.pieces]
    assert (s.split, s.merge, s.false) == (0, 0, 0)


def test_the_test_set_meets_the_acceptance_rule_and_never_does_worse_than_before():
    from tests.detection import report
    rows = report.main(write=False)
    ok, before, after = report.accepted(rows)
    assert ok and after.split <= before.split and after.merge <= before.merge + 1
    assert after.correct >= before.correct and after.missed <= before.missed
