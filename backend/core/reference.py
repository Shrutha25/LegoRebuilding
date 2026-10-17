"""Find pieces by comparing the photo with a photo of the same empty surface (the "reference").

Whatever the surface looks like (dark mesh, wood, towel), it cancels out; only what was added is left. It works when
the reference and the photo were taken from the same place, under the same light. Small differences are corrected:

1. Alignment: ORB features + a homography (RANSAC), else a plain translation (phase correlation), else none; whichever
   leaves the smaller difference on the surface is used. A few pixels of shift would otherwise light up a textured
   surface edge by edge.
2. Exposure: the reference is brought to the photo's overall brightness (median ratio of lightness).
3. Difference in Lab colour, blurred. Cast shadows only darken the surface (same chroma, moderately darker), so those
   pixels are ignored; a dark piece darkens far more and stays.
4. Threshold above the noise of the surface itself (median + k x MAD of the difference, never below `min_diff`),
   cleaned up with morphology; every blob is a piece.
5. Pieces the same colour as the surface (a black brick on black mesh, a white one on white paper) are caught by STRUCTURE,
   not colour: where the empty surface has fine texture, a piece replaces it with smooth plastic, so the local correlation
   with the reference collapses (even when the average colour is identical); where the surface is plain, a piece brings
   edges, studs and shading that were not there (local detail appears). Both cues are switched off for a photo where they
   fire almost everywhere (camera moved, light changed), and the colour difference carries on alone.

Pieces must not touch: one blob is one piece. That keeps a brick with two colours (or one that shades from yellow-green to
green) whole. For the other case, `split_touching=True` cuts an oversized blob along regions of different hue, and an
unsaturated remainder (a white or grey piece lying against a coloured one) becomes a piece of its own.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReferenceConfig:
    work_side: int = 1000          # long side of the copy used for alignment
    orb_features: int = 1500
    min_inliers: int = 30          # ORB homography accepted with at least this many inliers
    blur_sigma: float = 2.0        # smoothing of both photos before differencing
    lightness_weight: float = 0.6  # lightness counts less than chroma (exposure and shadows move it)
    min_diff: float = 22.0         # never threshold below this (Lab units)
    mad_k: float = 6.0             # threshold = median + mad_k * MAD of the difference
    low_factor: float = 0.55       # hysteresis: weaker differences (this share of the threshold, never below the surface's own
                                   # noise) count as part of a piece when they connect to a strong core ...
    grow_frac: float = 0.05        # ... no further than this share of the short side from the core
    shadow_chroma: float = 9.0     # chroma difference below this ...
    shadow_ratio_lo: float = 0.45  # ... and lightness between this share ...
    shadow_ratio_hi: float = 0.97  # ... and this share of the reference's = a cast shadow, not a piece
    open_frac: float = 0.004       # morphology kernel sizes, share of the short side
    close_frac: float = 0.012
    min_area_frac: float = 0.0004  # smaller blobs are noise
    max_area_frac: float = 0.40    # bigger blobs mean the comparison failed
    pad_frac: float = 0.08
    min_match: float = 0.4         # photos whose surface texture correlates less than this (median over the textured pixels) are
                                   # not of the same scene (camera moved or zoomed, light changed): the reference is not used
    min_match_texture: float = 0.02    # ...judged only when at least this share of the photo is textured (a plain surface has no pattern to match)
    split_touching: bool = False   # pieces are expected NOT to touch: a blob is one piece. True = also cut touching pieces by colour
    texture_window_frac: float = 0.012  # structure cues: window size, share of the short side
    texture_min_var: float = 25.0  # the empty surface counts as textured above this local variance (grey levels squared)
    ncc_min: float = 0.5           # textured surface: correlation with the reference below this = something else is there
    var_drop: float = 0.3          # ... or the local variance fell below this share of the reference's (smooth plastic)
    new_var_floor: float = 60.0    # plain surface: local variance above this = edges, studs, shading that were not there
    texture_max_share: float = 0.35    # structure cues firing on more than this share of the photo = misaligned: switch off
    split_outlier: float = 1.8     # a blob this many times bigger than the typical blob may hold several touching pieces
    split_hue_width: int = 10      # hue histogram bin (OpenCV hue units, 0-179, circular)
    split_hue_tol: int = 14        # two coloured regions can be one piece only if their hues are within this ...
    split_contact: float = 1.2     # ... and they touch along a line at least this many times sqrt(area of the smaller):
                                   # a plate shading from yellow-green to green shares its whole width; red and orange pieces
                                   # that touch at a corner do not
    split_step: float = 5.0        # ... and the hue does not JUMP across the border (a gradient inside one piece changes a
                                   # degree or two over a few pixels; a yellow brick beside a lime one jumps by 10 or more)
    split_min_share: float = 0.15  # a coloured region must be at least this share of the blob to count as a piece
    duplicate_overlap: float = 0.6 # a box this much inside another box is the same piece found twice
    split_min_remainder: float = 0.2   # an unsaturated rest (white, grey piece) counts if this share of a typical blob


CFG = ReferenceConfig()


def _gray_small(img, cfg):
    h, w = img.shape[:2]
    s = min(1.0, cfg.work_side / max(h, w))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), s


def _surface_error(a, b):
    """Mean absolute difference of two blurred grey photos, ignoring the 15% largest differences (the pieces)."""
    d = np.abs(cv2.GaussianBlur(a, (0, 0), 1.5).astype(np.float32) - cv2.GaussianBlur(b, (0, 0), 1.5).astype(np.float32))
    return float(np.sort(d.ravel())[: int(0.85 * d.size)].mean())


def align_reference(ref, img, cfg=CFG):
    """-> (reference warped onto the photo, method used: 'homography' | 'translation' | 'none')."""
    h, w = img.shape[:2]
    if ref.shape[:2] != (h, w):
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    g_ref, s = _gray_small(ref, cfg)
    g_img, _ = _gray_small(img, cfg)
    candidates = [("none", ref, _surface_error(g_ref, g_img))]
    try:
        orb = cv2.ORB_create(cfg.orb_features)
        k1, d1 = orb.detectAndCompute(g_ref, None)
        k2, d2 = orb.detectAndCompute(g_img, None)
        if d1 is not None and d2 is not None and len(k1) >= 8 and len(k2) >= 8:
            matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:600]
            if len(matches) >= cfg.min_inliers:
                src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                if H is not None and int(inl.sum()) >= cfg.min_inliers:
                    S = np.diag([1 / s, 1 / s, 1.0])                  # work-copy coordinates -> full-size coordinates
                    Hfull = S @ H @ np.linalg.inv(S)
                    warped = cv2.warpPerspective(ref, Hfull, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    candidates.append(("homography", warped, _surface_error(_gray_small(warped, cfg)[0], g_img)))
    except cv2.error:
        pass
    try:
        (dx, dy), _ = cv2.phaseCorrelate(g_ref.astype(np.float32), g_img.astype(np.float32))
        M = np.float32([[1, 0, dx / s], [0, 1, dy / s]])
        shifted = cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        candidates.append(("translation", shifted, _surface_error(_gray_small(shifted, cfg)[0], g_img)))
    except cv2.error:
        pass
    name, best, _ = min(candidates, key=lambda c: c[2])
    return best, name


def structure_mask(img, ref, cfg=CFG):
    """Pixels where the SURFACE'S STRUCTURE changed, whatever the colour: (textured surface) the local correlation with the
    reference collapsed or the texture vanished, or (plain surface) detail appeared. Returns (mask, share of textured surface)."""
    gi = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gr = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gr = gr * (float(np.median(gi)) / max(1.0, float(np.median(gr))))                 # exposure
    win = max(5, int(min(gi.shape) * cfg.texture_window_frac)) | 1
    box = lambda x: cv2.boxFilter(x, -1, (win, win), normalize=True)                  # noqa: E731
    mi, mr = box(gi), box(gr)
    vi, vr = np.maximum(box(gi * gi) - mi * mi, 0), np.maximum(box(gr * gr) - mr * mr, 0)
    cov = box(gi * gr) - mi * mr
    ncc = cov / np.sqrt(vi * vr + 1e-3)
    textured = vr > cfg.texture_min_var
    changed = textured & ((ncc < cfg.ncc_min) | (vi < cfg.var_drop * vr))
    plain_new = ~textured & (vi > cfg.new_var_floor)
    return changed | plain_new, float(textured.mean())


def match_quality(img, ref, cfg=CFG):
    """How well the surface pattern of the photo and the (aligned) reference agree: the median local correlation over the pixels
    where both are textured, and the share of the photo that is. ~1 for the same scene; ~0 when the camera moved or zoomed, or
    the photos are of different surfaces. (None, share) when there is too little texture to judge (a plain surface)."""
    gi = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gr = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    win = max(5, int(min(gi.shape) * cfg.texture_window_frac)) | 1
    box = lambda x: cv2.boxFilter(x, -1, (win, win), normalize=True)                  # noqa: E731
    mi, mr = box(gi), box(gr)
    vi, vr = np.maximum(box(gi * gi) - mi * mi, 0), np.maximum(box(gr * gr) - mr * mr, 0)
    ncc = (box(gi * gr) - mi * mr) / np.sqrt(vi * vr + 1e-3)
    both = (vr > cfg.texture_min_var) & (vi > cfg.texture_min_var)
    share = float(both.mean())
    if share < cfg.min_match_texture:
        return None, share
    return float(np.median(ncc[both])), share


def difference_mask(img, ref, cfg=CFG, info=None):
    """Boolean mask of pixels that differ from the (aligned) reference, and the difference image. See the module docstring."""
    lab_i = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), cfg.blur_sigma)
    lab_r = cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), cfg.blur_sigma)
    gain = float(np.median(lab_i[:, :, 0]) / max(1.0, np.median(lab_r[:, :, 0])))      # exposure
    lab_r[:, :, 0] = np.clip(lab_r[:, :, 0] * gain, 0, 255)
    dl = lab_i[:, :, 0] - lab_r[:, :, 0]
    dchroma = np.hypot(lab_i[:, :, 1] - lab_r[:, :, 1], lab_i[:, :, 2] - lab_r[:, :, 2])
    diff = np.sqrt(dchroma ** 2 + (cfg.lightness_weight * dl) ** 2)
    ratio = lab_i[:, :, 0] / np.maximum(lab_r[:, :, 0], 1.0)
    shadow = (dchroma < cfg.shadow_chroma) & (ratio > cfg.shadow_ratio_lo) & (ratio < cfg.shadow_ratio_hi)
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    thr = max(cfg.min_diff, med + cfg.mad_k * 1.4826 * mad)
    strong = (diff > thr) & ~shadow
    # Hysteresis: a piece whose sides are close to the surface in brightness (a grey brick on dark mesh) has a strong lit top
    # and weak sides. Weak pixels connected to a strong core, and near it, belong to the same piece.
    low = max(cfg.low_factor * thr, med + 3.0 * 1.4826 * mad)
    colour = strong
    if low < thr and cfg.grow_frac > 0:
        weak = (diff > low) & ~shadow
        n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[strong])] = True
        keep[0] = False
        near = cv2.dilate(strong.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(cfg.grow_frac * min(diff.shape)) + 1,) * 2)) > 0
        colour = strong | (keep[lab] & near)
    structure, textured_share = structure_mask(img, ref, cfg)
    # structure cues that fire nearly everywhere mean the photos do not match (moved camera, new light): ignore them
    structure_ok = float(structure.mean()) <= cfg.texture_max_share
    if info is not None:
        info.update(structure=structure_ok, textured=textured_share)
    return (colour | structure) if structure_ok else colour, diff


def _hue_gap(a, b):
    d = abs(int(a) - int(b))
    return min(d, 180 - d)


def _hue_median(values):
    """Median of OpenCV hues (0-179, a circle): shifted by half a turn so red does not straddle the cut."""
    return (float(np.median((values.astype(int) + 90) % 180)) - 90) % 180


def split_blob(img, blob, median_area, cfg=CFG):
    """Cut one blob (boolean mask, full-photo size) into the pieces it holds. Returns a list of boolean masks, or
    [blob] when it is one piece.

    - Saturated regions of clearly different hue (a lime brick against a blue plate) are separate pieces, however big
      the blob. Hue classes next to each other are ONE colour (a pale plate shading from yellow-green to green).
    - An unsaturated remainder (a white or grey piece lying against them) is a piece of its own, but only in a blob much
      bigger than the typical one, or when the blob already split by colour."""
    h_img, w_img = blob.shape
    short = min(h_img, w_img)
    k = max(3, int(short * cfg.open_frac)) | 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    strong = blob & (hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 80)
    width = cfg.split_hue_width
    n_classes = 180 // width
    classes = ((hsv[:, :, 0].astype(int) + width // 2) % 180) // width              # circular: red wraps round
    area = int(blob.sum())
    # a coloured region counts if it is a good share of the blob, but in a big blob (several pieces) a share would hide
    # a small piece, so it is capped at a quarter of the typical blob
    min_region = max(0.0004 * h_img * w_img, min(cfg.split_min_share * area, 0.25 * median_area))
    counts = np.bincount(classes[strong], minlength=n_classes)
    # Coloured components per hue bin (bins with only stray pixels are ignored, so they cannot bridge two colours) ...
    comps = []                                                       # [mask, hue (0-180 circle), area]
    for c in range(n_classes):
        if counts[c] < 0.25 * min_region:
            continue
        m = cv2.morphologyEx((strong & (classes == c)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 0.25 * min_region:
                comps.append([lab == i, c * width, int(st[i, cv2.CC_STAT_AREA])])
    # ... joined when their hues are close AND they share a long border: one piece shading from one hue to the next.
    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    grown = [cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for m, _, _ in comps]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            d = abs(comps[i][1] - comps[j][1])
            if min(d, 180 - d) <= cfg.split_hue_tol:
                contact = int((grown[i] & comps[j][0]).sum())
                if contact >= cfg.split_contact * np.sqrt(min(comps[i][2], comps[j][2])):
                    near_i, near_j = comps[i][0] & grown[j], comps[j][0] & grown[i]       # each side of the border
                    if near_i.any() and near_j.any():
                        step = _hue_gap(_hue_median(hsv[:, :, 0][near_i]), _hue_median(hsv[:, :, 0][near_j]))
                        if step <= cfg.split_step:
                            parent[find(i)] = find(j)
    merged = {}
    for i, (m, _, _) in enumerate(comps):
        merged[find(i)] = merged.get(find(i), np.zeros(blob.shape, bool)) | m
    regions = [m for m in merged.values() if int(m.sum()) >= min_region]
    kept = []                                          # the same colour found in two overlapping ways
    for r in sorted(regions, key=lambda r: -int(r.sum())):
        if not any(int((r & q).sum()) >= 0.3 * min(int(r.sum()), int(q.sum())) for q in kept):
            kept.append(r)
    parts = list(kept)
    if len(kept) >= 2 or (kept and area >= cfg.split_outlier * median_area):
        covered = cv2.dilate(np.any(kept, axis=0).astype(np.uint8), np.ones((4 * k + 1, 4 * k + 1), np.uint8)) > 0
        rest = (blob & ~covered).astype(np.uint8)
        rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= cfg.split_min_remainder * median_area:
                parts.append(lab == i)
    return parts if len(parts) >= 2 else [blob]


def merge_duplicates(boxes, overlap):
    """Boxes that lie mostly inside one another are one piece found twice: use their union."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a_, b_ = boxes[i], boxes[j]
                ix = max(0, min(a_[0] + a_[2], b_[0] + b_[2]) - max(a_[0], b_[0]))
                iy = max(0, min(a_[1] + a_[3], b_[1] + b_[3]) - max(a_[1], b_[1]))
                if ix * iy >= overlap * min(a_[2] * a_[3], b_[2] * b_[3]):
                    x0, y0 = min(a_[0], b_[0]), min(a_[1], b_[1])
                    boxes[i] = (x0, y0, max(a_[0] + a_[2], b_[0] + b_[2]) - x0, max(a_[1] + a_[3], b_[1] + b_[3]) - y0)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def find_pieces_by_reference(img, ref, min_area_frac=None, pad_frac=None, cfg=CFG, info=None):
    """Return (x, y, w, h) boxes of what is in `img` but not in `ref`. `info`, if a dict, receives the alignment method,
    the threshold-free share of changed pixels and the mask (for debugging)."""
    h_img, w_img = img.shape[:2]
    min_area_frac = cfg.min_area_frac if min_area_frac is None else min_area_frac
    pad_frac = cfg.pad_frac if pad_frac is None else pad_frac
    aligned, method = align_reference(ref, img, cfg)
    mask, _ = difference_mask(img, aligned, cfg, info)
    match, _ = match_quality(img, aligned, cfg)
    matched = match is None or match >= cfg.min_match
    short = min(h_img, w_img)
    ko, kc = max(3, int(short * cfg.open_frac)) | 1, max(5, int(short * cfg.close_frac)) | 1
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)), iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    min_area = min_area_frac * h_img * w_img
    blobs = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area
             and st[i, cv2.CC_STAT_WIDTH] * st[i, cv2.CC_STAT_HEIGHT] <= cfg.max_area_frac * h_img * w_img]
    areas = [int(st[i, cv2.CC_STAT_AREA]) for i in blobs]
    median_area = float(np.median(areas)) if areas else 0.0
    masks = []
    for i, a in zip(blobs, areas):
        blob = lab == i
        masks.extend(split_blob(img, blob, median_area, cfg) if cfg.split_touching else [blob])
    boxes = []
    for b in masks:
        ys, xs = np.nonzero(b)
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        p = int(max(w, h) * pad_frac)
        x0, y0 = max(0, x - p), max(0, y - p)
        boxes.append((x0, y0, min(w_img, x + w + p) - x0, min(h_img, y + h + p) - y0))
    boxes = merge_duplicates(boxes, cfg.duplicate_overlap)
    if info is not None:
        info.update(alignment=method, changed=float(mask.mean()), mask=m, match=match, matched=matched)
    row_h = np.median([b[3] for b in boxes]) if boxes else 1
    boxes.sort(key=lambda b: (int((b[1] + b[3] / 2) // row_h), b[0]))
    return boxes
