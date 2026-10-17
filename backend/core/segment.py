"""Mask-based piece segmentation: one piece per connected region unless there is EVIDENCE to split it.

Every detection is a `Region`: a binary mask (holes filled), its rotated rectangle and, last, the axis-aligned box. All
overlap / duplicate / merge / coverage decisions here use masks, never boxes.

    split_region()     section 3: the five split rules (seam, stud grids, colour, shape, size); each returns WHERE to split
    merge_pass()       section 4: merge neighbours that are really one piece, then re-run the split rules
    add_missing_masks()  section 5: add a candidate only where no existing mask covers it
    mark_overlaps()    section 6: any two masks that still overlap are flagged

Every threshold lives in `SegConfig` (environment overrides: SEG_<NAME>, e.g. SEG_SEAM_DL=14).
"""

import logging
import math
import os
from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from backend.core import partdims

log = logging.getLogger("lego.segment")


@dataclass
class SegConfig:
    # ---- 3a seam: a thin dark line crossing the component
    seam_dl: float = 12.0              # seam is at least this many L* darker than the pixels on both sides
    seam_width_pitch: float = 0.15     # ...and at most this wide (share of the local stud pitch)
    seam_cover: float = 0.70           # ...and runs along at least this share of the cut
    seam_min_rect: float = 0.90        # the component is rectangular (area / rotated rectangle) at least this much: grooves on a round part are no seam
    # ---- 3b stud grids
    grid_offset: float = 0.25          # grids offset from a whole-pitch multiple by more than this (share of pitch) are two pieces
    grid_angle_deg: float = 10.0       # ...or rotated against each other by more than this
    grid_min_studs: int = 4            # a group counts as evidence only with this many studs (the angle of three is guesswork on a tilted photo)
    grids_valley_dl: float = 9.0       # lattices that disagree split a region only if something physical lies between the groups: a dark valley (L*)...
    grids_valley_cover: float = 0.70   # ...along at least this share of the join (continuous, unlike the rims of a stud row)...
    grids_notch_solidity: float = 0.90 # ...or a notch in the outline (solidity below this)...
    grids_hue_gap: float = 20.0        # ...or a clear colour change between the parts (degrees of hue)
    # ---- 3c colour
    colour_spread: float = 8.0         # k-means runs only if the component's (a, b) spread is at least this
    colour_hue_gap: float = 20.0       # the two clusters' hue differs by more than this (degrees)...
    colour_min_chroma: float = 15.0    # ...both have more chroma than this...
    colour_min_share: float = 0.15     # ...and each is at least this share of the component
    colour_surface_chroma: float = 25.0  # ...or when its chroma is this close to the surface's (Lab a, b)
    colour_surface_share: float = 0.60 # a cluster that is this surface-like is shadow or glow, not a second piece
    colour_l_weight: float = 0.30      # lightness weight in the k-means (so a lit and a shaded face do not form two clusters)
    # ---- 3d shape
    solidity: float = 0.85             # area / convex hull area below this...
    shape_min_solidity: float = 0.50   # ...but below this it is a hollow outline (a shadow ring, a rim), not pieces
    defect_pitch: float = 0.50         # ...and a convexity defect deeper than this (share of pitch) = two pieces
    # ---- 3e size
    footprint_tol: float = 0.35        # measured footprint within this many studs of a real one is real
    max_studs: int = partdims.MAX_STUDS   # no piece is wider than this (baseplates)
    size_min_stud_share: float = 0.5   # the rule needs studs on at least this share of the footprint's cells (else the pitch says nothing about its size)
    min_part_pitch2: float = 0.75      # a part smaller than this (pitch^2) is not a piece
    # ---- 4 merge
    merge_dilate: int = 2              # masks this close (px) touch
    merge_surface_frac: float = 0.20   # the strip between two pieces may be at most this surface
    neutral_l_gap: float = 60.0        # two neutral (grey/black/white) pieces are alike if their lightness differs by less
    merge_duplicate: float = 0.10      # masks overlapping by more than this share of the smaller are one piece seen twice, whatever their colours
    merge_rounds: int = 8              # merge/re-split rounds before giving up on stability
    # ---- 5 add_missing
    add_uncovered: float = 0.60        # a candidate is added only if at least this share of it is uncovered
    add_max_surface: float = 0.80      # ...and one whose pixels are this surface-like (a shadow, a glow of reflected colour) is not a piece
    min_sliver_pitch: float = 0.50     # an uncovered remainder thinner than this (share of pitch) is not a piece
    fringe_contact: float = 0.35       # ...and an uncovered part this much of whose outline touches covered pixels is a fringe, not a piece
    min_sliver_frac: float = 0.012     # ...nor thinner than this share of the photo's short side when there is no pitch
    # ---- 6 ambiguity
    ambiguity_margin: float = 0.20     # a split rule within this share of its threshold is ambiguous
    max_tiebreaks: int = 3             # Brickognize tie-breaks per scan
    tiebreak_delta: float = 0.10       # split wins if every part beats the whole crop's top score by this
    overlap_check: float = 0.10        # masks overlapping by more than this share of either are flagged
    # ---- crops
    crop_margin_pitch: float = 0.50    # margin around the rotated rectangle, in stud pitches
    crop_margin_fallback: float = 0.06 # ...or this share of the long side when the pitch is unknown

    def __post_init__(self):
        for f in fields(self):
            raw = os.environ.get("SEG_" + f.name.upper())
            if raw is not None:
                try:
                    setattr(self, f.name, f.type(raw) if callable(f.type) else float(raw))
                except (TypeError, ValueError):
                    pass


SEG = SegConfig()


# ---------------------------------------------------------------- masks
def _ellipse(k):
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask):
    """Fill the holes of a 0/1 mask (axle holes, gaps between studs): redraw its outer contours filled."""
    p = cv2.copyMakeBorder(np.ascontiguousarray(mask, np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(p)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out[1:-1, 1:-1]


class Region:
    """One detection: a tight ROI mask (0/1, holes filled) at (x, y) in the photo, plus what is known about it."""

    def __init__(self, mask, x, y, source="merged"):
        self.mask = np.ascontiguousarray(mask, np.uint8)
        self.x, self.y = int(x), int(y)
        self.source = source
        self.rules = []              # split / merge rules that produced this region (debug labels)
        self.ambig = None            # Ambiguity this region is a part of
        self.check = False           # flag for the UI's "check this"
        self.split_id = None         # parts of one split share this: they are never merged straight back
        self.pitch = None            # local stud pitch (px), None when unknown
        self.grow = 0                # px the method grew the mask beyond the piece's edge
        self._colour = None
        self._hull = None

    @classmethod
    def from_full(cls, full, source="merged", grow=0):
        ys, xs = np.nonzero(full)
        if len(xs) == 0:
            return None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        r = cls(fill_holes(full[y0:y1, x0:x1] > 0), x0, y0, source)
        r.grow = grow
        return r

    @property
    def h(self):
        return self.mask.shape[0]

    @property
    def w(self):
        return self.mask.shape[1]

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self):
        return int(self.mask.sum())

    def full(self, shape):
        out = np.zeros(shape[:2], np.uint8)
        out[self.y:self.y + self.h, self.x:self.x + self.w] = self.mask
        return out

    def derive(self, sub, source=None):
        """A region from a 0/1 mask in this region's ROI coordinates (None if empty)."""
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            return None
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        r = Region(fill_holes(sub[y0:y1, x0:x1] > 0), self.x + x0, self.y + y0, source or self.source)
        r.grow = self.grow
        r.pitch = self.pitch
        return r

    def centroid(self):
        ys, xs = np.nonzero(self.mask)
        return (self.x + float(xs.mean()), self.y + float(ys.mean()))


def _pair(a, b):
    """The overlapping ROI of two regions as (a's pixels, b's pixels), or None."""
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return None
    return a.mask[y0 - a.y:y1 - a.y, x0 - a.x:x1 - a.x], b.mask[y0 - b.y:y1 - b.y, x0 - b.x:x1 - b.x]


def overlap_px(a, b):
    p = _pair(a, b)
    return int(np.count_nonzero(p[0] & p[1])) if p else 0


def mask_iou(a, b):
    inter = overlap_px(a, b)
    return inter / (a.area + b.area - inter) if inter else 0.0


def touching(a, b, dilate=None):
    """Do two masks touch or overlap? Each is dilated by `dilate` px (default SEG.merge_dilate) for the test."""
    d = SEG.merge_dilate if dilate is None else dilate
    if (a.x - d > b.x + b.w or b.x - d > a.x + a.w or a.y - d > b.y + b.h or b.y - d > a.y + a.h):
        return False
    pad = d + 1
    x0, y0 = min(a.x, b.x) - pad, min(a.y, b.y) - pad
    x1, y1 = max(a.x + a.w, b.x + b.w) + pad, max(a.y + a.h, b.y + b.h) + pad
    ca = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cb = np.zeros_like(ca)
    ca[a.y - y0:a.y - y0 + a.h, a.x - x0:a.x - x0 + a.w] = a.mask
    cb[b.y - y0:b.y - y0 + b.h, b.x - x0:b.x - x0 + b.w] = b.mask
    k = _ellipse(2 * d + 1)
    return bool(np.any(cv2.dilate(ca, k) & cv2.dilate(cb, k)))


def union_region(a, b, source="merged", bridge=None):
    """Both masks plus the strip between them (up to `bridge` px wide, default SEG.merge_dilate), holes filled."""
    bridge = max(SEG.merge_dilate, int(bridge or 0))
    pad = bridge + 2
    x0, y0 = min(a.x, b.x) - pad, min(a.y, b.y) - pad
    x1, y1 = max(a.x + a.w, b.x + b.w) + pad, max(a.y + a.h, b.y + b.h) + pad
    c = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for r in (a, b):
        c[r.y - y0:r.y - y0 + r.h, r.x - x0:r.x - x0 + r.w] |= r.mask
    c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, _ellipse(2 * bridge + 1))
    ys, xs = np.nonzero(c)
    cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    r = Region(fill_holes(c[cy0:cy1, cx0:cx1]), x0 + cx0, y0 + cy0, source)
    r.grow = max(a.grow, b.grow)
    return r


def clip_region(r, shape):
    """Drop any part of a region outside the photo (and trim)."""
    h_img, w_img = shape[:2]
    x0, y0 = max(0, r.x), max(0, r.y)
    x1, y1 = min(w_img, r.x + r.w), min(h_img, r.y + r.h)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = r.mask[y0 - r.y:y1 - r.y, x0 - r.x:x1 - r.x]
    out = Region(sub, x0, y0, r.source)
    out.__dict__.update({k: v for k, v in r.__dict__.items() if k not in ("mask", "x", "y", "_colour", "_hull")})
    return out


def rotated_rect(region, stretch=1.0):
    """cv2.minAreaRect of the mask's outline: ((cx, cy), (w, h), angle) in the photo's coordinates. `stretch` is the vertical
    factor that undoes a tilted view (the rectangle is found in the stretched photo and mapped back)."""
    p = cv2.copyMakeBorder(region.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float32) - 1
    pts = pts + [region.x, region.y]
    if abs(stretch - 1.0) > 0.01:
        pts = pts * [1.0, stretch]
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts.astype(np.float32))
    return (cx, cy / stretch), (w, h), ang


def partition_by_seeds(mask, seeds):
    """Split a 0/1 mask among seed masks: every pixel of the mask goes to the nearest seed pixel (so the parts always add up to the whole mask).
    -> list of 0/1 masks, one per seed (same order)."""
    owner_of = np.zeros(mask.shape, np.int32)
    for i, sd in enumerate(seeds, 1):
        owner_of[(sd > 0) & (mask > 0) & (owner_of == 0)] = i
    seed_px = (owner_of > 0).astype(np.uint8)
    if not seed_px.any():
        return [np.zeros(mask.shape, np.uint8) for _ in seeds]
    _, labels = cv2.distanceTransformWithLabels(1 - seed_px, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    owner_by_label = np.zeros(int(labels.max()) + 1, np.int32)
    owner_by_label[labels[seed_px > 0]] = owner_of[seed_px > 0]          # the label of a seed pixel names the seed it belongs to
    nearest = owner_by_label[labels]
    return [((nearest == i) & (mask > 0)).astype(np.uint8) for i in range(1, len(seeds) + 1)]


def parts_from_cut(mask, cut, min_area):
    """Remove the `cut` pixels from a mask; the connected pieces that remain (each at least `min_area`) are parts, and the cut pixels
    go to the nearest part. -> list of 0/1 masks, or [] if fewer than two parts remain."""
    rest = (mask > 0) & ~(cut > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=4)
    seeds = [(lab == i).astype(np.uint8) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(seeds) < 2:
        return []
    return partition_by_seeds(mask, seeds)


# ---------------------------------------------------------------- the photo's pixels
@dataclass
class Scene:
    """What the rules need to know about the photo, computed once."""
    shape: tuple
    lab: np.ndarray                 # true L*a*b* (L 0-100), lightly blurred
    surface: np.ndarray             # bool: looks like the photo's own surface
    groups: list = field(default_factory=list)
    stretch: float = 1.0            # vertical stretch the stud detector applied (tilted view)
    chroma_dist: np.ndarray = None  # distance of each pixel's chroma (a, b) from the surface's: a shadow keeps it, a piece does not

    @classmethod
    def build(cls, img, surface, groups=(), stretch=1.0, chroma_dist=None):
        lab = cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        return cls(img.shape, lab, surface, list(groups), stretch, chroma_dist)


def _hull_mask(region):
    """The region's convex hull as a 0/1 mask (cached): a stud group sits on a piece whose mask may be an outline with the middle missing."""
    if region._hull is None:
        p = cv2.copyMakeBorder(region.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        pts = np.concatenate(cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        hull = np.zeros_like(region.mask)
        cv2.fillConvexPoly(hull, cv2.convexHull(pts) - 1, 1)
        region._hull = hull
    return region._hull


def groups_in(region, scene, min_studs=1):
    """Stud groups sitting on the region: their centre is inside the region's convex hull."""
    out = []
    hull = _hull_mask(region)
    for g in scene.groups:
        if len(g.studs) < min_studs:
            continue
        cx, cy = g.centre
        ix, iy = int(round(cx)) - region.x, int(round(cy)) - region.y
        if 0 <= ix < region.w and 0 <= iy < region.h and hull[iy, ix]:
            out.append(g)
    return out


def local_pitch(region, scene):
    """Median stud spacing inside the region (px), or None when it holds no stud group."""
    gs = groups_in(region, scene)
    if not gs:
        return None
    return float(np.median(np.repeat([g.pitch for g in gs], [len(g.studs) for g in gs])))


def resolve_pitches(regions, scene):
    """Give every region its local pitch; one without studs takes the median of its nearest neighbours' pitches. None if the whole
    photo has no pitch (the scale-dependent rules are then disabled, and that is logged)."""
    for r in regions:
        r.pitch = local_pitch(r, scene)
    known = [r for r in regions if r.pitch]
    if not known:
        if regions:
            log.info("no stud pitch anywhere: seam, shape and size rules are disabled")
        return
    for r in regions:
        if r.pitch is None:
            cx, cy = r.centroid()
            near = sorted(known, key=lambda k: math.hypot(k.centroid()[0] - cx, k.centroid()[1] - cy))[:3]
            r.pitch = float(np.median([k.pitch for k in near]))
            r.rules.append("pitch:neighbours")


@dataclass
class Colour:
    L: float
    a: float
    b: float
    chroma: float
    hue: float


def region_colour(region, scene):
    """Median colour of the region's inner pixels (the rim is blended with the surface)."""
    if region._colour is not None:
        return region._colour
    inner = cv2.erode(region.mask, _ellipse(5)) > 0
    if inner.sum() < 20:
        inner = region.mask > 0
    sub = scene.lab[region.y:region.y + region.h, region.x:region.x + region.w][inner]
    L, a, b = (float(v) for v in np.median(sub, axis=0))
    region._colour = Colour(L, a, b, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360)
    return region._colour


def hue_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def same_hue(ca, cb, cfg=SEG):
    """Same hue per rule 3c. Lightness may differ (lit and shaded faces) for coloured pieces; two neutral pieces (grey, black,
    white have no hue) are alike only when their lightness is too, so a black and a white brick are never one."""
    if ca.chroma > cfg.colour_min_chroma and cb.chroma > cfg.colour_min_chroma:
        return hue_gap(ca.hue, cb.hue) <= cfg.colour_hue_gap
    if ca.chroma <= cfg.colour_min_chroma and cb.chroma <= cfg.colour_min_chroma:
        return abs(ca.L - cb.L) <= cfg.neutral_l_gap
    return False


# ---------------------------------------------------------------- section 3: the split rules
@dataclass
class Split:
    rule: str
    parts: list                     # 0/1 masks in the region's ROI coordinates
    margin: float                   # how far the evidence is past its threshold (share of the threshold)
    alternatives: int = 1           # equally good ways to split (rule e)
    detail: str = ""

    @property
    def ambiguous(self):
        return self.margin <= SEG.ambiguity_margin or self.alternatives > 1


def _margin(value, threshold):
    return abs(value - threshold) / threshold if threshold else 1.0


def _min_part(pitch, region):
    return max(30.0, SEG.min_part_pitch2 * pitch * pitch) if pitch else max(30.0, 0.04 * region.area)


def rule_seam(region, scene, pitch):
    """3a: a thin dark straight line across the component, much darker than both sides, along most of the cut."""
    if not pitch:
        return None
    m = region.mask
    inside = m > 0
    if inside.sum() < 200:
        return None
    (_, _), (rw, rh), _ = rotated_rect(region, scene.stretch)
    if region.area < SEG.seam_min_rect * rw * rh / scene.stretch:
        return None
    L = scene.lab[region.y:region.y + region.h, region.x:region.x + region.w, 0].copy()
    L[~inside] = np.median(L[inside])
    k = max(3, int(round(SEG.seam_width_pitch * pitch))) | 1
    k += 2                                                      # a little wider than the seam so closing bridges it
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, _ellipse(k))
    core = cv2.erode(m, _ellipse(k + 2)) > 0                    # away from the outline
    seam = ((bh >= SEG.seam_dl) & core).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(seam, connectivity=8)
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3 * k:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel())
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        perp = np.abs(-(pts[:, 0] - x0) * vy + (pts[:, 1] - y0) * vx)
        if np.median(perp) > 0.6 * k:
            continue                                            # not straight
        span = float(len(np.unique(np.round(t).astype(int))))      # chord length the seam actually occupies (gaps do not count)
        far = m.shape[0] + m.shape[1]
        line = np.zeros_like(m)
        cv2.line(line, (int(x0 - vx * far), int(y0 - vy * far)), (int(x0 + vx * far), int(y0 + vy * far)), 1, 1)
        chord = line & m
        nc, cl, cs, _ = cv2.connectedComponentsWithStats(chord, connectivity=8)
        if nc < 2:
            continue
        big = 1 + int(np.argmax(cs[1:, cv2.CC_STAT_AREA]))
        cy_, cx_ = np.nonzero(cl == big)
        tc = (cx_ - x0) * vx + (cy_ - y0) * vy
        chord_len = float(tc.max() - tc.min()) + 1
        cover = span / chord_len
        if cover < SEG.seam_cover:
            continue
        cut = cv2.dilate(line & m, _ellipse(3))
        parts = parts_from_cut(m, cut, _min_part(pitch, region))
        if len(parts) < 2:
            continue
        depth = float(np.median(bh[lab == i]))
        margin = min(_margin(depth, SEG.seam_dl), _margin(cover, SEG.seam_cover))
        if best is None or margin > best.margin:
            best = Split("seam", parts, margin, detail=f"dL={depth:.0f} cover={cover:.2f}")
    return best


def _lattice_offset(a, b, stretch=1.0):
    """How far group b's studs sit from a's lattice points, as a share of the pitch (0 = on the lattice, 0.5 = half way)."""
    th = math.radians(a.angle)
    ax = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    k = np.array([1.0, stretch])                                  # a group's angle and pitch live in the y-stretched (tilt-corrected) view
    o = np.array([a.studs[0].x, a.studs[0].y]) * k
    pb = np.array([[s.x, s.y] for s in b.studs]) * k
    frac = ((pb - o) @ ax.T) / a.pitch
    return float(np.median(np.abs(frac - np.rint(frac)), axis=0).max())          # per grid axis, the worse axis


def _grids_disagree(a, b, stretch=1.0):
    """-> (margin, detail) if the two groups cannot be one lattice, else None."""
    off_angle = abs(a.angle - b.angle) % 90
    off_angle = min(off_angle, 90 - off_angle)
    offset = _lattice_offset(a, b, stretch)
    hits = []
    if off_angle > SEG.grid_angle_deg:
        hits.append(_margin(off_angle, SEG.grid_angle_deg))
    if offset > SEG.grid_offset and off_angle <= SEG.grid_angle_deg * 3:
        hits.append(_margin(offset, SEG.grid_offset))
    if not hits:
        return None
    return min(hits), f"angle={off_angle:.0f} offset={offset:.2f}"


def grids_continuous(region, scene):
    """Rule 3b does NOT fire: every pair of stud groups in the region is one lattice."""
    gs = groups_in(region, scene, SEG.grid_min_studs)
    return not any(_grids_disagree(a, b, scene.stretch) for i, a in enumerate(gs) for b in gs[i + 1:])


def _separated(region, scene, parts, pitch):
    """Evidence, besides the lattices, that the parts are different pieces: a darker valley along their join (the gap two loose bricks leave,
    even a faint one), a notch in the outline, or a clear colour change. One piece whose studs were found a little off has none of these."""
    m = region.mask
    hull = float(_hull_mask(region).sum())
    if hull > 0 and region.area / hull < SEG.grids_notch_solidity:
        return True
    cols = []
    for part in parts:
        r = region.derive(part)
        cols.append(region_colour(r, scene) if r is not None else None)
    if all(c is not None for c in cols) and any(
            c.chroma > SEG.colour_min_chroma and d.chroma > SEG.colour_min_chroma and hue_gap(c.hue, d.hue) > SEG.grids_hue_gap
            for i, c in enumerate(cols) for d in cols[i + 1:]):
        return True
    L = scene.lab[region.y:region.y + region.h, region.x:region.x + region.w, 0].copy()
    inside = m > 0
    L[~inside] = np.median(L[inside])
    k = max(3, int(round(0.3 * (pitch or 40)))) | 1
    near = cv2.dilate(cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, _ellipse(k)), _ellipse(9))       # strongest valley within ~4 px
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            edge = (cv2.dilate(parts[i], _ellipse(5)) & parts[j]) | (cv2.dilate(parts[j], _ellipse(5)) & parts[i])
            if edge.sum() < 10:
                continue
            vals = np.where(edge > 0, near, 0.0)
            # a gap between bricks is a CONTINUOUS line along the join; the rims of a stud row are dark in patches only
            along = [float(np.mean(v[(edge > 0).any(axis=ax)] >= SEG.grids_valley_dl)) for v, ax in ((vals.max(axis=1), 1), (vals.max(axis=0), 0))]
            if max(along) >= SEG.grids_valley_cover:
                return True
    return False


def rule_grids(region, scene, pitch):
    """3b: two stud groups whose grids are offset (not a whole number of pitches apart) or turned against each other."""
    gs = groups_in(region, scene, SEG.grid_min_studs)
    if len(gs) < 2:
        return None
    parent = list(range(len(gs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    worst = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            dis = _grids_disagree(gs[i], gs[j], scene.stretch)
            if dis:
                worst.append(dis)
            else:
                parent[find(i)] = find(j)                       # one lattice: same piece
    clusters = {}
    for i in range(len(gs)):
        clusters.setdefault(find(i), []).append(gs[i])
    if len(clusters) < 2:
        return None
    seeds = []
    for members in clusters.values():
        s = np.zeros(region.mask.shape, np.uint8)
        for g in members:
            cv2.fillPoly(s, [np.rint(g.footprint - [region.x, region.y]).astype(np.int32)], 1)
        seeds.append(s & region.mask)
    seeds = [s for s in seeds if s.sum()]
    if len(seeds) < 2:
        return None
    parts = partition_by_seeds(region.mask, seeds)
    if not _separated(region, scene, parts, pitch):
        return None                                              # nothing physical between the groups: the stud detector's lattices just disagree
    return Split("grids", parts, min(m for m, _ in worst), detail=worst[0][1])


def rule_colour(region, scene, pitch):
    """3c: two clusters of pixels of clearly different hue (both properly coloured, both a real share). Hue and chroma only:
    the lit top and shaded side of one piece differ in lightness, not hue."""
    inner = cv2.erode(region.mask, _ellipse(5)) > 0
    if inner.sum() < 100:
        return None
    sub = scene.lab[region.y:region.y + region.h, region.x:region.x + region.w]
    pix = sub[inner]
    if math.hypot(float(pix[:, 1].std()), float(pix[:, 2].std())) < SEG.colour_spread:
        return None
    step = max(1, len(pix) // 4000)
    feats = pix.copy()
    feats[:, 0] *= SEG.colour_l_weight
    cv2.setRNGSeed(0)
    _, labels, centres = cv2.kmeans(feats[::step].astype(np.float32), 2, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    chroma = np.hypot(centres[:, 1], centres[:, 2])
    if chroma.min() <= SEG.colour_min_chroma:
        return None
    hue = np.degrees(np.arctan2(centres[:, 2], centres[:, 1])) % 360
    gap = hue_gap(hue[0], hue[1])
    if gap <= SEG.colour_hue_gap:
        return None
    share = np.bincount(labels.ravel(), minlength=2) / len(labels)
    if share.min() < SEG.colour_min_share:
        return None
    full = np.zeros(inner.shape, np.int8)
    d = np.stack([np.hypot(sub[:, :, 1] - centres[c, 1], sub[:, :, 2] - centres[c, 2]) ** 2
                  + (SEG.colour_l_weight * (sub[:, :, 0] - centres[c, 0] / SEG.colour_l_weight)) ** 2 for c in (0, 1)])
    full = np.argmin(d, axis=0)
    seeds = []
    for c in (0, 1):
        s = cv2.morphologyEx(((full == c) & inner).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(5))
        n, lab, st, _ = cv2.connectedComponentsWithStats(s, connectivity=8)
        if n < 2:
            return None
        s = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        seeds.append(s)
    if min(int(s.sum()) for s in seeds) < _min_part(pitch, region):
        return None
    on_surface = scene.surface[region.y:region.y + region.h, region.x:region.x + region.w]
    if any(float(on_surface[s > 0].mean()) >= SEG.colour_surface_share for s in seeds):
        return None                                              # the second "colour" is the surface seen through a shadow or glow
    if scene.chroma_dist is not None:
        cd = scene.chroma_dist[region.y:region.y + region.h, region.x:region.x + region.w]
        if any(float(np.median(cd[s > 0])) < SEG.colour_surface_chroma for s in seeds):
            return None                                          # a cluster with the surface's own chroma is its shadow
    parts = partition_by_seeds(region.mask, seeds)
    margin = min(_margin(gap, SEG.colour_hue_gap), _margin(float(share.min()), SEG.colour_min_share))
    return Split("colour", parts, margin, detail=f"hue {hue[0]:.0f}/{hue[1]:.0f} share {share.min():.2f}")


def _is_corner_part(region, stretch):
    """Does the outline match an L / corner part: a rectangle with ONE rectangular notch at a corner? (Not to be cut at its bend.)"""
    (cx, cy), (w, h), ang = rotated_rect(region)
    if min(w, h) < 4:
        return False
    full = region.mask
    # deskew the mask to the rectangle's own axes
    M = cv2.getRotationMatrix2D((cx - region.x, cy - region.y), ang, 1.0)
    side = int(math.hypot(region.w, region.h)) + 4
    M[0, 2] += side / 2 - (cx - region.x)
    M[1, 2] += side / 2 - (cy - region.y)
    dsk = cv2.warpAffine(full, M, (side, side), flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(dsk)
    if len(xs) == 0:
        return False
    box = dsk[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    missing = (box == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(missing, connectivity=4)
    if n < 2:
        return False
    notch = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    nx, ny, nw, nh, na = (int(v) for v in st[notch])
    bh, bw = box.shape
    at_corner = (nx == 0 or nx + nw == bw) and (ny == 0 or ny + nh == bh)
    rest = int(missing.sum()) - na
    return at_corner and na >= 0.08 * box.size and na / (nw * nh) >= 0.9 and rest <= 0.03 * box.size and na <= 0.6 * box.size


def _compact(r):
    """A part a real cut leaves is compact (solidity of at least SEG.solidity); a Y or T shaped part cut at one bend is not."""
    if r is None:
        return False
    pc = max(cv2.findContours(cv2.copyMakeBorder(r.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    ha = cv2.contourArea(cv2.convexHull(pc))
    return ha > 0 and r.area / ha >= SEG.solidity


def rule_shape(region, scene, pitch):
    """3d: solidity under SEG.solidity AND a convexity defect deeper than SEG.defect_pitch pitches; cut at the defect.
    L / corner parts are exempt."""
    if not pitch:
        return None
    p = cv2.copyMakeBorder(region.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if hull_area <= 0:
        return None
    solidity = region.area / hull_area
    if solidity >= SEG.solidity or solidity < SEG.shape_min_solidity:
        return None
    hull_idx = cv2.convexHull(c, returnPoints=False)
    try:
        defects = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return None
    if defects is None:
        return None
    defects = defects.reshape(-1, 4)
    depths = defects[:, 3] / 256.0
    k = int(np.argmax(depths))
    depth = float(depths[k])
    if depth <= SEG.defect_pitch * pitch:
        return None
    if float(scene.surface[region.y:region.y + region.h, region.x:region.x + region.w][region.mask > 0].mean()) >= 0.5:
        return None                                              # mostly shadow: an outline of a pale piece, not two pieces
    if _is_corner_part(region, scene.stretch):
        return None
    pts = c[:, 0, :]
    far = int(defects[k, 2])
    P = pts[far]
    n = len(pts)
    idx = np.arange(n)
    arc = np.minimum(np.abs(idx - far), n - np.abs(idx - far))
    cand = arc >= max(8, int(0.15 * n))
    if not cand.any():
        return None
    dist = np.hypot(*(pts - P).T)
    dist[~cand] = np.inf
    tried, parts = [], []
    for k in np.argsort(dist)[:400]:                             # nearest far-away outline points first; a cut must leave compact parts
        Q = pts[int(k)]
        if not np.isfinite(dist[k]) or any(np.hypot(*(Q - t)) < 0.25 * max(depth, 1.0) for t in tried):
            continue
        tried.append(Q)
        if len(tried) > 5:
            break
        cut = np.zeros(p.shape, np.uint8)
        cv2.line(cut, tuple(int(v) for v in P), tuple(int(v) for v in Q), 1, 3)
        found = parts_from_cut(region.mask, cut[1:-1, 1:-1], _min_part(pitch, region))
        if len(found) >= 2 and all(_compact(region.derive(m)) for m in found):
            parts = found
            break
    if len(parts) < 2:
        return None
    margin = min(_margin(solidity, SEG.solidity), _margin(depth, SEG.defect_pitch * pitch))
    if not groups_in(region, scene):
        margin = min(margin, 0.5 * SEG.ambiguity_margin)       # no studs to confirm it: a Y / T shaped part looks the same
    return Split("shape", parts, margin, detail=f"solidity={solidity:.2f} depth={depth / pitch:.2f}p")


def footprint_studs(region, pitch, stretch=1.0):
    """Measured footprint in studs (long, short), from the rotated rectangle and the local pitch. The mask's growth beyond the real edge
    (background method) is taken off each side."""
    (_, _), (w, h), _ = rotated_rect(region, stretch)
    g = 2.0 * region.grow
    a, b = max(1.0, w - g) / pitch, max(1.0, h - g) / pitch
    return max(a, b), min(a, b)


def rule_size(region, scene, pitch):
    """3e: the footprint is not a real one (within SEG.footprint_tol studs) AND the stud grid can be cut into real footprints.
    Splits into the partition with the fewest pieces; several equally good partitions make it ambiguous."""
    if not pitch:
        return None
    long_, short_ = footprint_studs(region, pitch, scene.stretch)
    if partdims.nearest_footprint(long_, short_, SEG.footprint_tol):
        return None
    gs = groups_in(region, scene)
    if gs and grids_continuous(region, scene):
        th = math.radians(gs[0].angle)                           # the studs themselves say how big the piece is: their lattice extent
        axes = np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
        pts = np.array([[st.x, st.y] for g in gs for st in g.studs]) @ axes.T
        lattice = tuple(int(round(float(np.ptp(pts[:, k])) / pitch)) + 1 for k in (0, 1))
        if partdims.is_real_footprint(*lattice):
            return None                                          # one real lattice (glare or shadow only inflates the mask)
    cols, rows = int(round(long_)), int(round(short_))
    if abs(long_ - cols) > SEG.footprint_tol or abs(short_ - rows) > SEG.footprint_tol or rows < 1 or cols < 1:
        return None                                              # not on a whole-stud grid: the measure says nothing about a cut
    (cx, cy), (w, h), ang = rotated_rect(region, scene.stretch)
    if region.area < 0.8 * w * h / scene.stretch:
        return None                                              # not a rectangle: the shape rule's business
    studs_seen = sum(len(g.studs) for g in groups_in(region, scene))
    if studs_seen < SEG.size_min_stud_share * cols * rows:
        return None                                              # the stud grid does not cover the footprint: the measure is not trustworthy
    n, sols = partdims.partition_footprint(cols, rows)
    if not n or n < 2:
        return None
    sol = sols[0]
    # rectangle frame in the (stretched) photo: long side along u
    th = math.radians(ang)
    u = np.array([math.cos(th), math.sin(th)])
    v = np.array([-math.sin(th), math.cos(th)])
    if h > w:
        u, v, w, h = v, -u, h, w                                 # make u the long axis
    seeds = []
    for (c0, r0, cw, rh) in sol:
        corners = []
        for fu, fv in ((c0, r0), (c0 + cw, r0), (c0 + cw, r0 + rh), (c0, r0 + rh)):
            p = np.array([cx, cy * scene.stretch]) + u * (fu / cols - 0.5) * w + v * (fv / rows - 0.5) * h
            corners.append([p[0], p[1] / scene.stretch])
        poly = np.rint(np.array(corners) - [region.x, region.y]).astype(np.int32)
        s = np.zeros(region.mask.shape, np.uint8)
        cv2.fillPoly(s, [poly], 1)
        # shrink so the seeds do not touch: the cut falls between them
        s = cv2.erode(s, _ellipse(max(3, int(0.15 * pitch)))) & region.mask
        seeds.append(s)
    seeds = [s for s in seeds if s.sum()]
    if len(seeds) < 2:
        return None
    parts = partition_by_seeds(region.mask, seeds)
    if scene.stretch > 1.05 and not _separated(region, scene, parts, pitch):
        return None                                              # a tilted view distorts the measured footprint: only cut with physical evidence too
    return Split("size", parts, 1.0, alternatives=len(sols),
                 detail=f"{long_:.1f}x{short_:.1f} studs -> {n} parts")


def _drop_slivers(region, parts, pitch):
    """A part thinner than SEG.min_sliver_pitch pitches (or SEG.min_sliver_frac of the photo's short side) is a strip of shadow or a
    frame-edge remnant, not a piece: it is given back to the real parts. Fewer than two real parts left = no split."""
    if not pitch:
        return parts
    keep = []
    for m in parts:
        r = region.derive(m)
        if r is not None and min(rotated_rect(r)[1]) >= SEG.min_sliver_pitch * pitch:
            keep.append(m)
    if len(keep) == len(parts):
        return parts
    return partition_by_seeds(region.mask, keep) if len(keep) >= 2 else []


RULES = (rule_seam, rule_grids, rule_colour, rule_shape, rule_size)
_split_ids = iter(range(1, 1 << 30))


@dataclass
class Ambiguity:
    """A split the evidence only just supports: the whole region and its parts, for the Brickognize tie-break."""
    whole: Region
    parts: list
    rule: str
    margin: float
    alternatives: int = 1


def split_region(region, scene, depth=0, rules=RULES):
    """Apply the five split rules; a region is split only if one fires. Parts are checked again (a part may hold a third piece).
    -> list of Region (the region itself when nothing fires)."""
    if depth > 4 or region.area < 30:
        return [region]
    pitch = region.pitch
    for rule in rules:
        try:
            s = rule(region, scene, pitch)
        except Exception as e:                                   # noqa: BLE001  a rule failing must never fail a scan
            log.warning("split rule %s failed: %s", rule.__name__, e)
            s = None
        if s is None:
            continue
        sid = next(_split_ids)
        parts = []
        s.parts = _drop_slivers(region, s.parts, pitch)
        for m in s.parts:
            r = region.derive(m)
            if r is None:
                continue
            r.rules = list(region.rules) + [f"split:{s.rule}"]
            r.split_id = sid
            parts.append(r)
        if len(parts) < 2:
            continue
        log.info("split rule %s fired (%s)", s.rule, s.detail)
        out = []
        for r in parts:
            out.extend(split_region(r, scene, depth + 1, rules))
        if s.ambiguous:
            amb = Ambiguity(region, out, s.rule, s.margin, s.alternatives)
            for r in out:
                r.ambig = amb
                r.check = True
        return out
    return [region]


# ---------------------------------------------------------------- section 4: merge
def _strip_surface_frac(a, b, scene, pitch):
    """Share of surface-looking pixels in the strip between two masks (the gap between them, not the masks themselves)."""
    s = max(2, int(round(0.2 * pitch))) if pitch else 6
    pad = s + 2
    x0, y0 = min(a.x, b.x) - pad, min(a.y, b.y) - pad
    x1, y1 = max(a.x + a.w, b.x + b.w) + pad, max(a.y + a.h, b.y + b.h) + pad
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(scene.shape[1], x1), min(scene.shape[0], y1)
    ca = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cb = np.zeros_like(ca)
    ca[a.y - y0:a.y - y0 + a.h, a.x - x0:a.x - x0 + a.w] = a.mask
    cb[b.y - y0:b.y - y0 + b.h, b.x - x0:b.x - x0 + b.w] = b.mask
    k = _ellipse(2 * s + 1)
    strip = (cv2.dilate(ca, k) & cv2.dilate(cb, k)) & (1 - (ca | cb))
    sc = np.zeros(strip.shape, bool)
    sc[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = scene.surface[cy0:cy1, cx0:cx1]
    n = int(strip.sum())
    return float((sc & (strip > 0)).sum()) / n if n >= 8 else 0.0


def _fits_part_size(region, scene, pitch):
    if pitch:
        long_, short_ = footprint_studs(region, pitch, scene.stretch)
        return long_ <= SEG.max_studs + SEG.footprint_tol and short_ <= SEG.max_studs + SEG.footprint_tol
    return region.w * region.h <= 0.40 * scene.shape[0] * scene.shape[1]


def share_stud_group(a, b, scene):
    """One stud group sits on both: a stud with its shaded rim is boxed on its own by the background method, but the group says the
    studs (and the body between them) are one piece."""
    for g in groups_in(a, scene, SEG.grid_min_studs) + groups_in(b, scene, SEG.grid_min_studs):
        in_a = in_b = False
        for st in g.studs:
            for r, flag in ((a, 0), (b, 1)):
                ix, iy = int(round(st.x)) - r.x, int(round(st.y)) - r.y
                if 0 <= ix < r.w and 0 <= iy < r.h and _hull_mask(r)[iy, ix]:
                    in_a, in_b = in_a or flag == 0, in_b or flag == 1
        if in_a and in_b:
            return True
    return False


def can_merge(a, b, scene):
    """Section 4 conditions, other than 'the masks touch' (checked by the caller). -> (ok, union region or None)."""
    if a.split_id is not None and a.split_id == b.split_id:
        return False, None                                       # just separated by evidence
    pitch = a.pitch or b.pitch
    # Two real pieces never overlap in a photo: masks that mostly overlap are one piece seen twice (a pale piece's lit top and its shaded
    # rim differ in lightness, which the hue test would read as two pieces).
    duplicate = (overlap_px(a, b) >= SEG.merge_duplicate * min(a.area, b.area)
                 or _pair_hull(a, b) >= 0.5 * b.area or _pair_hull(b, a) >= 0.5 * a.area   # ...or lies within the other's outline
                 or share_stud_group(a, b, scene))                                         # ...or carries studs of the same lattice
    if not duplicate:
        if not same_hue(region_colour(a, scene), region_colour(b, scene)):
            return False, None
        if _strip_surface_frac(a, b, scene, pitch) >= SEG.merge_surface_frac:
            return False, None
    u = union_region(a, b)
    u.pitch = local_pitch(u, scene) or pitch
    if not _fits_part_size(u, scene, u.pitch):
        return False, None
    if not grids_continuous(u, scene):
        return False, None
    if u.pitch:
        try:
            if rule_seam(u, scene, u.pitch):
                return False, None
        except Exception:                                        # noqa: BLE001
            pass
    if not duplicate and u.pitch and rule_shape(u, scene, u.pitch):
        return False, None                                       # a bridge-shaped union is cut straight back at its neck: two pieces
    return True, u


def merge_pass(regions, scene, rounds=None):
    """Merge neighbours that are one piece (section 4), re-run the split rules on each merged region, and repeat until nothing changes.
    A region the rules split again is never merged straight back (its parts share a `split_id`)."""
    regions = list(regions)
    for _ in range(rounds or SEG.merge_rounds):
        changed = False
        pairs = []
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                a, b = regions[i], regions[j]
                if touching(a, b) or _pair_hull(a, b) >= 0.5 * b.area or _pair_hull(b, a) >= 0.5 * a.area or share_stud_group(a, b, scene):
                    pairs.append((i, j))
        pairs.sort(key=lambda ij: -min(regions[ij[0]].area, regions[ij[1]].area))
        for i, j in pairs:
            a, b = regions[i], regions[j]
            if a is None or b is None:
                continue
            ok, u = can_merge(a, b, scene)
            if not ok:
                continue
            u.rules = sorted(set(a.rules + b.rules)) + ["merge"]
            u.source = a.source if a.area >= b.area else b.source
            u.check = False
            u.ambig = None
            parts = split_region(u, scene)
            regions[i], regions[j] = None, None
            regions.extend(parts)
            changed = True
            # indices moved: recompute the pair list on the next round
            break
        regions = [r for r in regions if r is not None]
        if not changed:
            break
    return regions


def absorb_wrapped(regions, scene, cover=0.7, solidity=0.7):
    """A concave outline (a piece's L-shaped shadow, or the lit rim of a pale piece) whose convex hull holds another region is that
    region's outline: one piece, not two. The wrapped region and its outline become one region before any split rule looks at it."""
    regions = list(regions)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(regions):
            held = [j for j, b in enumerate(regions) if j != i and _pair_hull(a, b) >= 0.35 * b.area]
            ca = region_colour(a, scene)
            for j in held:
                b = regions[j]
                if not same_hue(ca, region_colour(b, scene)):
                    continue                                     # an outline around a piece of another colour is not its outline
                inside = _pair_hull(a, b)
                if (overlap_px(a, b) >= 0.1 * b.area and inside >= 0.35 * b.area) or inside >= cover * b.area:
                    u = union_region(a, b, b.source if b.source != "background" else a.source, bridge=int(a.pitch or b.pitch or 4))
                    u.pitch = a.pitch or b.pitch
                    u.rules = sorted(set(a.rules + b.rules)) + ["absorb"]
                    regions = [r for k, r in enumerate(regions) if k not in (i, j)] + [u]
                    changed = True
                    break
            if changed:
                break
    return regions


def _pair_hull(a, b):
    """Pixels of b inside a's convex hull."""
    h = _hull_mask(a)
    x0, y0 = max(a.x, b.x), max(a.y, b.y)
    x1, y1 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(np.count_nonzero(h[y0 - a.y:y1 - a.y, x0 - a.x:x1 - a.x] & b.mask[y0 - b.y:y1 - b.y, x0 - b.x:x1 - b.x]))


# ---------------------------------------------------------------- section 5: add_missing
def studmod_min_seed():
    from backend.core import studs
    return studs.CFG.min_seed_studs                              # one circle is not a piece's studs


def coverage_canvas(regions, shape):
    c = np.zeros(shape[:2], np.uint8)
    for r in regions:
        c[r.y:r.y + r.h, r.x:r.x + r.w] |= r.mask
    return c


def add_missing_masks(kept, candidates, scene, pitch_hint=None):
    """Add candidate regions only where no kept mask covers them. A candidate is skipped when less than SEG.add_uncovered of it is uncovered;
    when part of it is covered, only the uncovered connected part is added, and only if that part is a plausible piece (big enough,
    not a thin sliver). Returns the new list (kept first)."""
    out = list(kept)
    covered = coverage_canvas(out, scene.shape)
    min_side = max(SEG.min_sliver_pitch * pitch_hint if pitch_hint else 0.0, SEG.min_sliver_frac * min(scene.shape[:2]))
    for c in candidates:
        if sum(1 for k in kept if overlap_px(c, k) >= 0.5 * k.area) >= 2:
            continue                                             # a blob that merely surrounds pieces already found (bridged by shadows)
        sub = covered[c.y:c.y + c.h, c.x:c.x + c.w]
        un = c.mask & (1 - sub)
        frac = float(un.sum()) / max(1, c.area)
        if frac < SEG.add_uncovered:
            continue
        if frac < 0.999:
            un = cv2.morphologyEx(un, cv2.MORPH_OPEN, _ellipse(max(3, int(min_side) | 1)))
            n, lab, st, _ = cv2.connectedComponentsWithStats(un, connectivity=8)
            pieces = []
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < max(30, min_side * min_side):
                    continue
                comp = (lab == i).astype(np.uint8)
                pc = cv2.copyMakeBorder(comp, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)      # the ring may lie outside the candidate's box
                ring = cv2.dilate(pc, _ellipse(5)) & (1 - pc)
                if float((ring & cv2.copyMakeBorder(sub, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)).sum()) / max(1, int(ring.sum())) >= SEG.fringe_contact:
                    continue                                     # wrapped by the covered mask: a halo or a shadow band, not a piece
                pieces.append(comp)
        else:
            pieces = [c.mask]
        for m in pieces:
            r = c.derive(m)
            if r is None or min(r.w, r.h) < min_side:
                continue
            if (float(scene.surface[r.y:r.y + r.h, r.x:r.x + r.w][r.mask > 0].mean()) >= SEG.add_max_surface
                    and not groups_in(r, scene, studmod_min_seed()) and any(touching(r, k, 12) for k in out)):
                continue                                         # looks like the surface itself and lies against a piece: its shadow, or a glow of reflected colour
            r.rules = list(c.rules) + ["add_missing"]
            out.append(r)
            covered[r.y:r.y + r.h, r.x:r.x + r.w] |= r.mask
    return out


# ---------------------------------------------------------------- section 6: overlaps, crops
def mark_overlaps(regions):
    """Any two masks still overlapping by more than SEG.overlap_check of either are flagged 'check' (and logged: a bug signal)."""
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            ov = overlap_px(a, b)
            if ov and ov > SEG.overlap_check * min(a.area, b.area):
                a.check = b.check = True
                a.rules.append("overlap")
                b.rules.append("overlap")
                log.warning("masks overlap by %d px (%.0f%% / %.0f%%) after all passes", ov, 100 * ov / a.area, 100 * ov / b.area)


def padded_box(region, pad_frac, shape):
    """The axis-aligned box of a region, padded the way the original detectors padded theirs."""
    x, y, w, h = region.box
    p = int(max(w, h) * pad_frac)
    h_img, w_img = shape[:2]
    x0, y0 = max(0, x - p), max(0, y - p)
    return (x0, y0, min(w_img, x + w + p) - x0, min(h_img, y + h + p) - y0)


def crop_region(img, region, pitch=None, stretch=1.0):
    """The deskewed crop of a region for recognition: its rotated rectangle plus a margin of half a stud pitch
    (or SEG.crop_margin_fallback of the long side when the pitch is unknown), turned upright with the long side horizontal."""
    (cx, cy), (w, h), ang = rotated_rect(region, stretch)
    if w < h:
        w, h, ang = h, w, ang + 90
    pitch = pitch or region.pitch
    m = SEG.crop_margin_pitch * pitch if pitch else SEG.crop_margin_fallback * max(w, h)
    out_w, out_h = int(round(w + 2 * m)), int(round(h + 2 * m))
    if out_w < 4 or out_h < 4:
        x, y, bw, bh = padded_box(region, 0.08, img.shape)
        return img[y:y + bh, x:x + bw]
    th = math.radians(ang)
    # affine map from the output crop to the photo: crop pixel (u, v) -> centre + R (u - out_w/2, v - out_h/2)
    c, s = math.cos(th), math.sin(th)
    M = np.array([[c, -s, cx - c * out_w / 2 + s * out_h / 2], [s, c, cy - s * out_w / 2 - c * out_h / 2]], np.float64)
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
