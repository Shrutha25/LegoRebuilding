"""Stud detection: the round studs on top of a piece look the same whatever its colour.

    find_studs(img)        circles that look like studs (centre, radius, Lab colour, confidence)
    group_studs(studs)     studs on one piece: a regular grid, one colour. Each group knows its pitch, grid
                           orientation, dimensions (e.g. 2 x 4) and a box.
    stud_info_for_box()    what the stud groups say about a given box (count, pitch, dims, radius, score)

Studs are a *signal*, not a filter: tiles, slopes, round and side-on pieces show none. Nothing in here may fail a
scan, so the callers wrap it and carry on without studs.

Every threshold lives in `StudConfig` below.
"""

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StudConfig:
    enabled: bool = os.environ.get("STUDS_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")   # kill switch
    # ---- working size and candidate circles
    tilt_stretch: float = float(os.environ.get("STUD_TILT_STRETCH", "1.5"))   # photo taken at ~45 degrees: studs are squashed
                                     # vertically, so the photo is stretched by this factor in y before looking for circles (1 = top-down)
    work_side: int = 1024            # long side of the downscaled copy used for detection (results map back)
    min_radius_frac: float = 0.006   # smallest stud radius, share of the work image's long side
    max_radius_frac: float = 0.034   # largest stud radius
    hough_dp: float = 1.5
    hough_canny: float = 150.0       # Hough's upper Canny threshold
    hough_accum: float = 0.65        # HOUGH_GRADIENT_ALT circle quality (0-1): lower = more candidates, validation filters them
    max_candidates: int = 500
    clahe_clip: float = 3.0          # CLAHE contrast limit: gives studs on pale pieces their shading
    # ---- validation of one candidate
    rim_samples: int = 24            # points tested around the circle
    rim_edge: float = 9.0            # gradient (grey levels, after CLAHE) that counts as "an edge here"
    rim_min_continuity: float = 0.55 # share of the rim that must show an edge
    same_plastic_chroma: float = 24.0   # inside vs just outside: max chroma (a,b) difference...
    same_plastic_light: float = 70.0    # ...and lightness difference, for "same plastic"
    mismatch_factor: float = 0.45    # confidence factor when the outside is a different colour
    bg_tol: float = 14.0             # inside this close to a surface cluster (Lab) = looks like the background
    bg_factor: float = 0.85          # confidence factor for studs coloured like the background (pale on pale)
    min_conf: float = 0.45           # below this a candidate is not a stud
    single_conf: float = 0.80        # a stud with no grid neighbour (a 1 x 1 piece) needs at least this
    # ---- size consistency
    radius_tol: float = 0.30         # studs in one photo have the same radius within this (perspective)
    # ---- grid
    pitch_ratio: float = 8.0 / 4.8 * 2.0   # centre-to-centre pitch / stud radius (8 mm pitch, 4.8 mm diameter)
    pitch_tol: float = 0.28          # neighbours at pitch +- this share are on the grid (perspective stretches it)
    neighbour_chroma: float = 30.0   # neighbouring studs of one piece have this alike colour (Lab a,b)...
    neighbour_light: float = 55.0    # ...and lightness
    min_group_studs: int = 2         # a group needs this many studs, or one stud of single_conf
    min_seed_studs: int = 2          # a group seeds a piece in the colour method only with this many studs (1 x 1 is a signal, not a seed)
    min_misaligned_split_studs: int = 4  # two groups of one colour are two pieces only if both have this many studs and their grids disagree (a brick with studs on two faces is one piece)
    min_split_studs: int = 3         # a group counts as evidence of a separate piece (splitting) only with this many studs
    min_coloured_for_radius: int = 3 # radius is learnt from non-background studs when at least this many exist
    aligned_min: float = 0.80        # grid orientation consistency (0-1) below which a group is re-split
    align_tol_deg: float = 12.0      # when re-splitting: edges must lie within this of the grid axes
    bg_min_group: int = 4            # a background-coloured group needs this many studs (pale-on-pale 1x1/1x2 are left to the colour methods)
    bg_group_max: int = 24           # a background-coloured group bigger than this is texture, not a piece
    max_group_frac: float = 0.30     # a group's box over this share of the photo is texture
    # ---- piece box from a group
    box_margin_pitch: float = 0.5    # box = stud circles expanded by this many pitches
    # ---- use as a signal
    drop_small_frac: float = 0.002   # "small": box under this share of the photo...
    drop_weak_colour: float = 0.02   # ...and with fewer than this share of coloured/dark pixels, and no studs = drop


CFG = StudConfig()


@dataclass
class Stud:
    x: float
    y: float
    r: float
    lab: tuple
    conf: float
    on_background: bool = False      # interior looks like the photo's surface


@dataclass
class StudGroup:
    studs: list
    lab: tuple
    pitch: float
    angle: float                     # grid orientation, degrees in [0, 90)
    dims: tuple                      # (short, long) number of studs along the grid axes
    box: tuple                       # (x, y, w, h): stud circles expanded by box_margin_pitch pitches
    aligned: float                   # 0-1 grid consistency
    radius: float
    on_background: bool = False
    footprint: np.ndarray = field(default=None, repr=False)   # polygon (N, 2) of the piece top

    @property
    def centre(self):
        return (float(np.mean([s.x for s in self.studs])), float(np.mean([s.y for s in self.studs])))

    @property
    def score(self):
        """0-1: how stud-like the group is (confidence, size, grid consistency)."""
        conf = float(np.mean([s.conf for s in self.studs]))
        return round(conf * (0.5 + 0.5 * self.aligned) * min(1.0, 0.5 + 0.25 * len(self.studs)), 3)


@dataclass
class StudInfo:
    """What the studs say about one detected piece (all pixel values in the photo's own scale)."""
    stud_count: int = 0
    pitch_px: float = 0.0
    dims: tuple = ()
    radius_px: float = 0.0
    score: float = 0.0


# ---------------------------------------------------------------- candidates
def _sample(arr, xs, ys):
    h, w = arr.shape[:2]
    return arr[np.clip(np.rint(ys).astype(int), 0, h - 1), np.clip(np.rint(xs).astype(int), 0, w - 1)]


def _candidates(gray, cfg):
    h, w = gray.shape
    side = max(h, w)
    rmin = max(3, int(cfg.min_radius_frac * side))
    rmax = max(rmin + 2, int(cfg.max_radius_frac * side))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=cfg.hough_dp, minDist=max(4.0, rmin * 1.2),
                               param1=cfg.hough_canny, param2=cfg.hough_accum, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return np.zeros((0, 3), np.float32)
    return circles[0][:cfg.max_candidates].astype(np.float32)


def _validate(cand, gray, lab, surface, cfg):
    """Vectorised: for each (x, y, r) -> (confidence, interior Lab, on_background)."""
    n = len(cand)
    cx, cy, r = cand[:, 0:1], cand[:, 1:2], cand[:, 2:3]
    # rim evidence: an edge across the circle's boundary, tolerant of slight ellipses (perspective)
    ang = np.linspace(0, 2 * np.pi, cfg.rim_samples, endpoint=False)[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    d = np.maximum(2.0, 0.18 * r)
    best = np.zeros((n, cfg.rim_samples), np.float32)
    for shift in (-0.2, 0.0, 0.2):
        rr = r * (1 + shift)
        m = np.abs(_sample(gray, cx + (rr + d) * cos, cy + (rr + d) * sin).astype(np.float32)
                   - _sample(gray, cx + (rr - d) * cos, cy + (rr - d) * sin).astype(np.float32))
        best = np.maximum(best, m)
    continuity = (best > cfg.rim_edge).mean(axis=1)
    strength = np.minimum(best.mean(axis=1) / (2 * cfg.rim_edge), 1.0)
    rim = np.clip((continuity - 0.4) / 0.45, 0, 1) * (0.5 + 0.5 * strength)
    rim[continuity < cfg.rim_min_continuity] = 0.0

    # inside colour: median of points in the disc (robust to the logo)
    a8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)[None, :]
    xs = np.concatenate([cx, cx + 0.4 * r * np.cos(a8), cx + 0.7 * r * np.cos(a8)], axis=1)
    ys = np.concatenate([cy, cy + 0.4 * r * np.sin(a8), cy + 0.7 * r * np.sin(a8)], axis=1)
    inside = np.median(_sample(lab, xs, ys), axis=1)                                   # (n, 3)
    # colour just outside: the piece top around the stud should be the same plastic
    a16 = np.linspace(0, 2 * np.pi, 16, endpoint=False)[None, :]
    ox = np.concatenate([cx + f * r * np.cos(a16) for f in (1.35, 1.65)], axis=1)
    oy = np.concatenate([cy + f * r * np.sin(a16) for f in (1.35, 1.65)], axis=1)
    out = _sample(lab, ox, oy)                                                          # (n, 32, 3)
    dchroma = np.hypot(out[:, :, 1] - inside[:, None, 1], out[:, :, 2] - inside[:, None, 2])
    dlight = np.abs(out[:, :, 0] - inside[:, None, 0])
    alike = (dchroma < cfg.same_plastic_chroma) & (dlight < cfg.same_plastic_light)
    same = alike.mean(axis=1) >= 0.4
    colour = np.where(same, 1.0, cfg.mismatch_factor)

    on_bg = np.zeros(n, bool)
    if surface is not None:
        centers, idx = surface
        for c in idx:
            dist = np.sqrt((inside[:, 1] - centers[c][1]) ** 2 + (inside[:, 2] - centers[c][2]) ** 2
                           + (0.5 * (inside[:, 0] - centers[c][0])) ** 2)
            on_bg |= dist < cfg.bg_tol
    conf = rim * colour * np.where(on_bg, cfg.bg_factor, 1.0)
    return conf, inside, on_bg


def _nms(cand, conf, keep, r0):
    """Candidates closer than 1.6 radii are the same stud (real neighbours are 3.3 radii apart): keep the most confident, then the one whose radius
    is closest to the photo's dominant stud radius."""
    order = sorted(np.nonzero(keep)[0], key=lambda i: (-round(float(conf[i]), 2), abs(cand[i, 2] - r0)))
    chosen = []
    for i in order:
        if all(np.hypot(cand[i, 0] - cand[j, 0], cand[i, 1] - cand[j, 1]) >= 1.6 * min(cand[i, 2], cand[j, 2])
               for j in chosen):
            chosen.append(i)
    return chosen


def _dominant_radius(r, w):
    """Most supported radius: peak of the confidence-weighted histogram (smoothed over neighbouring bins)."""
    if len(r) == 0:
        return 0.0
    bins = np.arange(0, float(r.max()) + 3, 1.0)
    hist, _ = np.histogram(r, bins=bins, weights=w)
    hist = np.convolve(hist, np.ones(3), mode="same")
    i = int(np.argmax(hist))
    sel = (r >= bins[i] - 1) & (r <= bins[i] + 2)
    return float(np.average(r[sel], weights=w[sel])) if sel.any() else float(bins[i])


def _lab_alike(a, b, cfg):
    return np.hypot(a[1] - b[1], a[2] - b[2]) < cfg.neighbour_chroma and abs(a[0] - b[0]) < cfg.neighbour_light


def find_studs(img, cfg=CFG, surface=None):
    """-> list[Stud] in the photo's own pixel coordinates.

    `surface` is detect._surface_model(img) when the caller already has it; otherwise it is computed here."""
    h_img, w_img = img.shape[:2]
    s = min(1.0, cfg.work_side / max(h_img, w_img))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else img
    gray = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8)).apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    cand = _candidates(gray, cfg)
    if len(cand) == 0:
        return []
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    if surface is None:
        from backend.core.detect import _surface_model
        surface = _surface_model(small)
    conf, inside, on_bg = _validate(cand, gray, lab, surface, cfg)
    ok = conf >= cfg.min_conf
    # Studs in one photo share a radius. Learn it from coloured studs when there are some (a textured pale
    # surface throws up many background-coloured bumps, whereas white pieces alone need the other branch).
    pool = ok & ~on_bg
    if pool.sum() < cfg.min_coloured_for_radius:
        pool = ok
    if not pool.any():
        return []
    r0 = _dominant_radius(cand[pool, 2], conf[pool])
    keep = ok & (np.abs(cand[:, 2] - r0) <= cfg.radius_tol * r0)
    chosen = _nms(cand, conf, keep, r0)
    if not chosen:
        return []

    # grid validation: a real stud has a same-coloured neighbour one pitch away (or is very confident: a 1 x 1)
    pts = cand[chosen, :2]
    dist = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    pitch = cfg.pitch_ratio * (cand[chosen, 2][:, None] + cand[chosen, 2][None, :]) / 2        # per pair: perspective
    near = (np.abs(dist - pitch) <= cfg.pitch_tol * pitch) | (np.abs(dist - pitch * np.sqrt(2)) <= cfg.pitch_tol * pitch)
    studs = []
    for k, i in enumerate(chosen):
        supported = any(near[k, m] and _lab_alike(inside[i], inside[chosen[m]], cfg) for m in range(len(chosen)) if m != k)
        if supported or (conf[i] >= cfg.single_conf and not on_bg[i]):
            studs.append(Stud(float(cand[i, 0] / s), float(cand[i, 1] / s), float(cand[i, 2] / s),
                              tuple(float(v) for v in inside[i]), float(conf[i]), bool(on_bg[i])))
    return studs


# ---------------------------------------------------------------- grouping
def _grid_angle(vectors):
    """Dominant grid orientation (degrees in [0, 90)) and consistency (0-1) of neighbour vectors."""
    if len(vectors) == 0:
        return 0.0, 1.0
    th = np.arctan2(vectors[:, 1], vectors[:, 0]) * 4                  # 4x: 90 degree symmetry
    c, s = np.cos(th).mean(), np.sin(th).mean()
    return float((np.degrees(np.arctan2(s, c)) / 4) % 90), float(np.hypot(c, s))


def _angle_off(vec, angle):
    """How far (degrees) a vector is from the nearest grid axis."""
    a = (np.degrees(np.arctan2(vec[1], vec[0])) - angle) % 90
    return min(a, 90 - a)


def _unite(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def group_studs(studs, cfg=CFG, shape=None):
    """Studs -> list[StudGroup]. Two studs are linked when they sit one pitch apart and have alike colours.
    A group whose links do not share one grid orientation (two touching pieces turned against each other) is
    re-linked using only links along its own grid axes, which usually pulls them apart. Groups too small, or
    texture (a background-coloured lattice, a box over `max_group_frac` of the photo), are dropped."""
    n = len(studs)
    if n == 0:
        return []
    xy = np.array([[s.x, s.y] for s in studs])
    rad = np.array([s.r for s in studs])

    def links(i_idx):
        out = []
        for a in range(len(i_idx)):
            for b in range(a + 1, len(i_idx)):
                i, j = i_idx[a], i_idx[b]
                d = np.hypot(*(xy[i] - xy[j]))
                pitch = cfg.pitch_ratio * (rad[i] + rad[j]) / 2
                if abs(d - pitch) <= cfg.pitch_tol * pitch and _lab_alike(studs[i].lab, studs[j].lab, cfg):
                    out.append((i, j))
        return out

    comps = _unite(n, links(range(n)))
    final = []
    for comp in comps:
        edges = [(i, j) for i, j in links(comp)]
        angle, aligned = _grid_angle(np.array([xy[j] - xy[i] for i, j in edges]))
        if aligned < cfg.aligned_min and len(comp) > 3:
            keep = [(i, j) for i, j in edges if _angle_off(xy[j] - xy[i], angle) <= cfg.align_tol_deg]
            idx = {g: k for k, g in enumerate(comp)}
            for sub in _unite(len(comp), [(idx[i], idx[j]) for i, j in keep]):
                final.append([comp[k] for k in sub])
        else:
            final.append(list(comp))

    groups = []
    for members in final:
        g = _make_group([studs[i] for i in members], cfg, shape)
        if g is not None:
            groups.append(g)
    return groups


def _make_group(members, cfg, shape):
    if len(members) < cfg.min_group_studs:
        if not (len(members) == 1 and members[0].conf >= cfg.single_conf):
            return None
    xy = np.array([[s.x, s.y] for s in members])
    r = float(np.median([s.r for s in members]))
    pitch = cfg.pitch_ratio * r
    vec, gaps = [], []
    for i in range(len(members)):                                    # vectors to the nearest neighbour(s) on the grid
        d = np.hypot(*(xy - xy[i]).T)
        for j in np.nonzero((d > 0) & (np.abs(d - pitch) <= cfg.pitch_tol * pitch))[0]:
            vec.append(xy[j] - xy[i])
            gaps.append(d[j])
    if gaps:
        pitch = float(np.median(gaps))                               # measured spacing beats radius x ratio
    angle, aligned = _grid_angle(np.array(vec)) if vec else (0.0, 1.0)
    th = np.radians(angle)
    ax = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    proj = (xy - xy.mean(axis=0)) @ ax.T
    dims = tuple(sorted(int(round(np.ptp(proj[:, k]) / pitch)) + 1 for k in (0, 1)))
    x0, y0 = (xy - r).min(axis=0)
    x1, y1 = (xy + r).max(axis=0)
    m = cfg.box_margin_pitch * pitch
    h_img, w_img = (shape[:2] if shape is not None else (1 << 30, 1 << 30))
    bx0, by0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    bx1, by1 = min(w_img, int(np.ceil(x1 + m))), min(h_img, int(np.ceil(y1 + m)))
    box = (bx0, by0, bx1 - bx0, by1 - by0)
    on_bg = float(np.mean([s.on_background for s in members])) >= 0.7
    if on_bg and len(members) < cfg.bg_min_group:
        return None
    if shape is not None:
        if on_bg and len(members) > cfg.bg_group_max:
            return None                                              # a lattice of background-coloured bumps
        if box[2] * box[3] > cfg.max_group_frac * h_img * w_img:
            return None
    pts = np.concatenate([xy + [r, 0], xy - [r, 0], xy + [0, r], xy - [0, r]]).astype(np.float32)
    hull = cv2.convexHull(pts)[:, 0, :]
    centre = hull.mean(axis=0)
    d = np.linalg.norm(hull - centre, axis=1, keepdims=True)
    footprint = hull + (hull - centre) / np.maximum(d, 1e-6) * m      # expanded by half a pitch outwards
    lab = tuple(float(v) for v in np.median([s.lab for s in members], axis=0))
    return StudGroup(members, lab, float(pitch), float(angle), dims, box, float(aligned), r, on_bg, footprint)


def _unstretch(group, k):
    """Map a group found in the y-stretched photo back to the photo's own coordinates."""
    for s in group.studs:
        s.y /= k
    x, y, w, h = group.box
    y0, y1 = y / k, (y + h) / k
    group.box = (x, int(y0), w, max(1, int(np.ceil(y1)) - int(y0)))
    group.footprint = group.footprint / np.array([1.0, k])
    return group


def find_stud_groups(img, cfg=CFG, surface=None):
    """find_studs + group_studs for photos taken at an angle: the photo is stretched vertically by `cfg.tilt_stretch`
    (a 45 degree view squashes round studs into ellipses about 0.7 as tall as wide), studs are found and grouped in
    that rectified view, and the results are mapped back. -> (studs, groups) in the photo's own coordinates."""
    k = float(cfg.tilt_stretch or 1.0)
    if abs(k - 1.0) < 0.05:
        studs = find_studs(img, cfg, surface)
        return studs, group_studs(studs, cfg, shape=img.shape)
    h, w = img.shape[:2]
    big = cv2.resize(img, (w, int(round(h * k))), interpolation=cv2.INTER_CUBIC)
    studs = find_studs(big, cfg, surface)
    groups = group_studs(studs, cfg, shape=big.shape)
    for g in groups:
        _unstretch(g, k)                                   # shares the Stud objects with `studs`
    members = {id(s) for g in groups for s in g.studs}
    for s in studs:
        if id(s) not in members:
            s.y /= k
    return studs, groups


# ---------------------------------------------------------------- using groups
def aligned_with(a, b, cfg=CFG):
    """Could two groups be parts of one piece? Same colour, same grid orientation, and studs of one lattice."""
    if not _lab_alike(a.lab, b.lab, cfg):
        return False
    off = abs(a.angle - b.angle) % 90
    if min(off, 90 - off) > cfg.align_tol_deg:
        return False
    th = np.radians(a.angle)
    ax = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    o = np.array([a.studs[0].x, a.studs[0].y])
    pb = np.array([[s.x, s.y] for s in b.studs])
    frac = ((pb - o) @ ax.T) / a.pitch
    resid = np.abs(frac - np.rint(frac))
    return float(resid.mean()) <= 0.2


def distinct_pieces(a, b, cfg=CFG):
    """Do two stud groups sit on different pieces? Different colours always say yes; grids that are not one lattice
    say yes only when both groups are big enough to be sure (a brick with studs on two faces is one piece)."""
    if not _lab_alike(a.lab, b.lab, cfg):
        return True
    return min(len(a.studs), len(b.studs)) >= cfg.min_misaligned_split_studs and not aligned_with(a, b, cfg)


def stud_info_for_box(box, groups, cfg=CFG):
    """Aggregate the groups whose centre lies inside `box` (studs-less boxes get an empty StudInfo)."""
    x, y, w, h = box
    inside = [g for g in groups if x <= g.centre[0] <= x + w and y <= g.centre[1] <= y + h]
    if not inside:
        return StudInfo()
    n = sum(len(g.studs) for g in inside)
    main = max(inside, key=lambda g: len(g.studs))
    score = float(np.mean([g.score for g in inside])) * (1.0 if len(inside) == 1 or all(
        aligned_with(main, g, cfg) for g in inside) else 0.6)
    return StudInfo(n, round(main.pitch, 1), main.dims, round(main.radius, 1), round(score, 3))
