"""Piece detection: find one bounding box per LEGO piece on a plain background.

Two OpenCV methods run on every photo and `find_pieces` combines them:

- the background-colour method compares every pixel with the colour of the photo's border. It is
  right when the plain sheet fills the photo, and it also finds white and transparent pieces;
- the colour method finds pieces from strongly coloured or dark seeds. It copes with uneven light,
  embossed paper, a table showing around the sheet, and neighbouring pieces, but misses pale pieces.

`combine_boxes` keeps the first method's boxes, splits any that hold two or more pieces according to
the second, and adds pieces only the second found. The first method is used only when its result
is plausible; when the photo's border is not the plain surface it produces junk and the colour
method works alone.

When the surface itself is strongly coloured or dark (a wooden table, black cloth), neither the colour seeds nor the
border colour can work, so a third method learns what the surface looks like and finds what differs
from it (`find_pieces_by_surface`).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.core import segment as seg
from backend.core import studs as studmod

log = logging.getLogger("lego.detect")

# A single "piece" bigger than this share of the photo means the background was misjudged.
GIANT_BOX_FRAC = 0.40
ENGINE = os.environ.get("DETECT_ENGINE", "mask").strip().lower()   # mask (segment.py, the default) | legacy (the earlier box logic)
COMPONENT_GROUPING = True       # colour method: seeds of one hue in one connected region are one piece (mask engine only)
USE_BOX_COLOUR_SPLIT = True      # the dim-surface path also runs the box-based colour splitter (`split_merged_by_colour`) before the mask rules


def _reading_order(boxes):
    """Sort in reading order (rows, then left-to-right)."""
    row_h = np.median([b[3] for b in boxes]) if boxes else 1
    boxes.sort(key=lambda b: (int((b[1] + b[3] / 2) // row_h), b[0]))
    return boxes


def _ordered(boxes, labels):
    """Reading order for boxes with a parallel list of labels."""
    row_h = np.median([b[3] for b in boxes]) if boxes else 1
    order = sorted(range(len(boxes)), key=lambda i: (int((boxes[i][1] + boxes[i][3] / 2) // row_h), boxes[i][0]))
    return [boxes[i] for i in order], [labels[i] for i in order]


def _order_regions(regions, pad_frac, shape):
    """Reading order for regions (the same order `_reading_order` gives their padded boxes)."""
    boxes = [seg.padded_box(r, pad_frac, shape) for r in regions]
    if not boxes:
        return []
    row_h = np.median([b[3] for b in boxes])
    order = sorted(range(len(boxes)), key=lambda i: (int((boxes[i][1] + boxes[i][3] / 2) // row_h), boxes[i][0]))
    return [regions[i] for i in order]


def _boxes_of(regions, pad_frac, shape):
    return [seg.padded_box(r, pad_frac, shape) for r in regions]


def _background_regions(img, min_area_frac=0.0005, pad_frac=0.08):
    """The background-colour method's detections as masks (holes filled)."""
    h_img, w_img = img.shape[:2]

    # Estimate background colour from the image border (works for white or any plain surface)
    border = np.concatenate([img[:10].reshape(-1, 3), img[-10:].reshape(-1, 3),
                             img[:, :10].reshape(-1, 3), img[:, -10:].reshape(-1, 3)])
    bg = np.median(border, axis=0)

    # Foreground = pixels that differ from background in colour OR are saturated
    diff = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((diff > 25) | (hsv[:, :, 1] > 40)).astype(np.uint8) * 255

    # Clean up: remove specks, then close gaps so a piece with highlights
    # (or a thin/transparent piece) stays one blob
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = min_area_frac * h_img * w_img

    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Skip very faint blobs (watermarks, shadows, smudges). Even transparent
        # or white pieces have strong edges/shading somewhere; these don't.
        if np.percentile(diff[y:y + h, x:x + w], 99) < 50:
            continue
        roi = np.zeros((h, w), np.uint8)
        cv2.drawContours(roi, [c - [x, y]], -1, 1, -1)          # filled: a piece's holes are part of it
        r = seg.Region(roi, x, y, "background")
        r.grow = (k - 1) // 2
        out.append(r)
    return _order_regions(out, pad_frac, img.shape)


def find_pieces_on_plain_background(img, min_area_frac=0.0005, pad_frac=0.08):
    """Original method. Return a list of (x, y, w, h) boxes, one per piece.

    Assumes the border of the photo is plain surface. Finds transparent and white pieces too."""
    return _boxes_of(_background_regions(img, min_area_frac, pad_frac), pad_frac, img.shape)


def _hue_gap(a, b):
    """Distance between two OpenCV hues (0-179, circular)."""
    d = abs(float(a) - float(b))
    return min(d, 180 - d)


def _colour_regions(img, min_area_frac=0.0005, pad_frac=0.08, sat=110, dark=60, grow=0.02,
                          weak_sat=60, weak_dark=100, join=2.0, dark_join=3.0,
                          hue_tol=8, sat_tol=90, val_tol=80, ring_min=0.5, edge_trim=0.03, stud_groups=None,
                          surface_model=None, report=None, pale_sat=45, pale_val=45, stud_on_seed=0.5):
    """The colour method's detections as masks; see `find_pieces_by_colour`."""
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1
    r = max(3, int(min(h_img, w_img) * grow))

    seeds = ((s_ch > sat) | (v_ch < dark)).astype(np.uint8) * 255
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((seeds > 0).astype(np.uint8), connectivity=8)
    min_seed = 0.0002 * h_img * w_img                         # a piece needs a seed at least this big...
    min_fragment = 0.00005 * h_img * w_img                    # ...but smaller ones can join a piece (stud holes)
    comps = []
    e = max(3, int(min(h_img, w_img) * edge_trim)) | 1
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_fragment:
            continue                                          # specks
        minor = area < min_seed
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            # Touches the photo edge: the table/frame around the sheet, or a piece whose colour spills
            # onto the paper as a thin glow that reaches the edge. Keep only the solid body, and drop it
            # if the body itself touches the edge.
            body = cv2.morphologyEx((lab == i).astype(np.uint8), cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
            ys_b, xs_b = np.nonzero(body)
            if len(xs_b) < min_seed or minor:
                continue
            x, y, w, h = int(xs_b.min()), int(ys_b.min()), int(xs_b.max()) - int(xs_b.min()) + 1, int(ys_b.max()) - int(ys_b.min()) + 1
            if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
                continue
            lab[(lab == i) & (body == 0)] = 0                 # the glow is not part of the piece
        sel = lab == i
        v_med = float(np.median(v_ch[sel]))
        comps.append({"id": i, "bbox": (x, y, w, h), "minor": minor, "dark": v_med < dark + 10, "h": float(np.median(h_ch[sel])),
                      "s": float(np.median(s_ch[sel])), "v": v_med})
    groups_by_id = {}
    next_id = n
    for g in (stud_groups or []):
        if len(g.studs) < studmod.CFG.min_seed_studs:
            continue                                              # one circle is too weak to seed a piece on its own
        fp = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(fp, [np.rint(g.footprint).astype(np.int32)], 1)
        disks = np.zeros((h_img, w_img), np.uint8)
        for st in g.studs:
            cv2.circle(disks, (int(st.x), int(st.y)), max(1, int(st.r * 0.6)), 1, -1)
        if not disks.any():
            continue
        on_seed = np.bincount(lab[(fp > 0) & (lab > 0)], minlength=next_id + 1)      # footprint pixels per colour seed
        overlap = {int(k): float(on_seed[k]) / float(fp.sum()) for k in np.nonzero(on_seed)[0]}
        free = (fp > 0) & (lab == 0)
        if free.sum() < min_fragment:
            continue                                              # entirely on colour seeds: they already cover it
        lab[free] = next_id                                       # the group's own pixels; colour seeds keep theirs
        sel = disks > 0
        v_med = float(np.median(v_ch[sel]))
        ys_f, xs_f = np.nonzero(lab == next_id)
        comps.append({"id": next_id, "bbox": (int(xs_f.min()), int(ys_f.min()), int(xs_f.max() - xs_f.min() + 1),
                                              int(ys_f.max() - ys_f.min() + 1)), "minor": False, "dark": v_med < dark + 10,
                      "h": float(np.median(h_ch[sel])), "s": float(np.median(s_ch[sel])), "v": v_med,
                      "stud": g, "overlap": overlap})
        groups_by_id[next_id] = g
        next_id += 1
    if not any(not c["minor"] for c in comps):
        return []

    # Section 2: what is NOT surface, as connected regions with their holes filled. Seeds of one hue that sit in the same region are
    # ONE piece however far apart they are (a long plate's fragments); the distance limit applies only between regions.
    plain_px = _surface_chroma_distance(img, surface_model or _surface_model(img)) < 25
    nons = ((~plain_px) | (lab > 0)).astype(np.uint8)
    nons = seg.fill_holes(cv2.morphologyEx(nons, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8)))
    n_reg, region_map = cv2.connectedComponents(nons, connectivity=8)
    for c in comps:
        votes = np.bincount(region_map[lab == c["id"]], minlength=n_reg)
        votes[0] = 0                                                # outside every region
        c["region"] = int(votes.argmax()) if votes.max() > 0 else 0

    def compatible(a, b):
        if a.get("stud") is not None and b.get("stud") is not None:
            return studmod.aligned_with(a["stud"], b["stud"])      # two parts of one lattice, nothing else
        if a.get("stud") is not None or b.get("stud") is not None:
            st, other = (a, b) if a.get("stud") is not None else (b, a)
            if st["overlap"].get(other["id"], 0.0) >= stud_on_seed:     # sits on that piece's body: same piece, whatever
                return True                                             # the print or highlights do to its colour
            if st["s"] < pale_sat or other["s"] < pale_sat:         # pale studs: hue is noise, compare lightness
                return abs(st["v"] - other["v"]) <= pale_val and abs(st["s"] - other["s"]) <= sat_tol
        if a["dark"] or b["dark"]:
            return a["dark"] and b["dark"]
        return (_hue_gap(a["h"], b["h"]) <= hue_tol and abs(a["s"] - b["s"]) <= sat_tol
                and abs(a["v"] - b["v"]) <= val_tol)

    def close(a, b):
        gap = (dark_join if a["dark"] and b["dark"] else join) * r
        ax, ay, aw, ah = a["bbox"]
        bx, by, bw, bh = b["bbox"]
        dx, dy = max(bx - (ax + aw), ax - (bx + bw), 0), max(by - (ay + ah), ay - (by + bh), 0)
        if (dx * dx + dy * dy) ** 0.5 > gap:
            return False
        x0, y0 = max(0, min(ax, bx) - 2), max(0, min(ay, by) - 2)
        x1, y1 = min(w_img, max(ax + aw, bx + bw) + 2), min(h_img, max(ay + ah, by + bh) + 2)
        sub = lab[y0:y1, x0:x1]
        dist_a = cv2.distanceTransform((sub != a["id"]).astype(np.uint8), cv2.DIST_L2, 3)
        return bool((dist_a[sub == b["id"]] <= gap).any())

    parent = list(range(len(comps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if compatible(comps[i], comps[j]) and (
                    close(comps[i], comps[j]) or (COMPONENT_GROUPING and comps[i]["region"] and comps[i]["region"] == comps[j]["region"])):
                parent[find(i)] = find(j)

    has_body = {}
    kinds = {}
    for idx, c in enumerate(comps):
        has_body[find(idx)] = has_body.get(find(idx), False) or not c["minor"]
        kinds.setdefault(find(idx), set()).add("studs" if c.get("stud") is not None else "colour")
    markers = np.zeros((h_img, w_img), np.int32)
    group_ids = {}
    for idx, c in enumerate(comps):
        if not has_body[find(idx)]:
            continue                                          # fragments with no piece to belong to
        markers[lab == c["id"]] = group_ids.setdefault(find(idx), len(group_ids) + 1)
    root_of = {gid: root for root, gid in group_ids.items()}
    seed_px = (markers > 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - seed_px, cv2.DIST_L2, 3)
    relief = np.clip(dist / max(1.0, float(dist.max())) * 255, 0, 255).astype(np.uint8)
    owner = cv2.watershed(cv2.cvtColor(relief, cv2.COLOR_GRAY2BGR), markers.copy())     # nearest group wins
    grown = (seed_px > 0) | (((s_ch > weak_sat) | (v_ch < weak_dark)) & (dist <= r))

    plain = plain_px                                           # looks like the photo's own surface
    far = dist > r                                             # farther than the growth distance from any piece
    min_area = min_area_frac * h_img * w_img
    out = []
    for g in group_ids.values():
        own = (owner == g) & grown
        ys, xs = np.nonzero(own)
        if len(xs) < min_area:
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            continue
        m = int(0.15 * max(w, h)) + 4                          # ring just outside the object
        X0, Y0, X1, Y1 = max(0, x - m), max(0, y - m), min(w_img, x + w + m), min(h_img, y + h + m)
        ring = np.ones((Y1 - Y0, X1 - X0), bool)
        ring[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = False
        # Judge only the surroundings proper: ring pixels far from every piece. Near a piece the paper
        # carries its colour cast and glow, and a neighbouring piece is not "background" either.
        ring &= far[Y0:Y1, X0:X1]
        if ring.sum() >= 50 and plain[Y0:Y1, X0:X1][ring].mean() < ring_min:
            continue
        if kinds[root_of[g]] == {"studs"} and ring.sum() < 50:
            continue                                          # a stud-only region must be seen to sit on plain surface
        roi = own[y:y + h, x:x + w].astype(np.uint8)
        roi = cv2.morphologyEx(cv2.copyMakeBorder(roi, k, k, k, k, cv2.BORDER_CONSTANT, value=0), cv2.MORPH_CLOSE,
                               np.ones((k, k), np.uint8))[k:-k, k:-k]        # one body: bridge pinholes, then fill the holes
        label = "studs-only" if kinds[root_of[g]] == {"studs"} else "colour+studs" if "studs" in kinds[root_of[g]] else "colour"
        out.append(seg.Region(seg.fill_holes(roi), x, y, label))
    return _order_regions(out, pad_frac, img.shape)


def find_pieces_by_colour(img, min_area_frac=0.0005, pad_frac=0.08, sat=110, dark=60, grow=0.02,
                          weak_sat=60, weak_dark=100, join=2.0, dark_join=3.0,
                          hue_tol=8, sat_tol=90, val_tol=80, ring_min=0.5, edge_trim=0.03, stud_groups=None,
                          surface_model=None, report=None, pale_sat=45, pale_val=45, stud_on_seed=0.5):
    """Fallback for photos where the plain surface does not fill the frame or is unevenly lit.

    1. Seeds: strongly saturated or very dark pixels. These ignore brightness texture (an embossed
       paper towel) and, unlike every weakly coloured pixel, are not bridged by the faint colour
       cast and long shadows that surround a piece.
    2. Grouping: seed fragments that are close AND alike in colour belong to one piece (the dark
       pockets between the studs of a grey brick; fragments of one pastel plate). Close fragments of
       a different colour stay separate pieces. Distance can't tell these apart on its own: the gap
       between two neighbouring pieces and the gap between the fragments of one dark piece are
       similar, but their colours differ.
    3. Growth: each group takes in paler pixels (weak colour or dark) within `grow` (share of the
       photo's short side) of its own seeds, nearest group first, so groups never absorb each other.
       A piece's box is the bounding box of its group.

    Anything touching the photo edge is ignored (the table or frame around the sheet; a piece cut
    off by the frame can't be identified anyway), and a candidate must be surrounded by light, plain
    surface. Pieces that overlap or touch still merge.

    `stud_groups` (from studs.group_studs) add a second kind of seed: the footprint of a stud group. It merges
    with the colour seeds it overlaps and matches, and a group with no colour seed (a white, pale grey or tan
    piece) becomes a seed group of its own, so pale pieces are found and the watershed gives them their own
    region. Without `stud_groups` the method is as before. `report`, if a list, receives one label per box:
    "colour", "colour+studs" or "studs-only"."""
    regions = _colour_regions(img, min_area_frac, pad_frac, sat, dark, grow, weak_sat, weak_dark, join, dark_join, hue_tol,
                              sat_tol, val_tol, ring_min, edge_trim, stud_groups, surface_model, None, pale_sat, pale_val,
                              stud_on_seed)
    if report is not None:
        report.extend(r.source for r in regions)
    return _boxes_of(regions, pad_frac, img.shape)


def _surface_model(img, clusters=5, chroma_tol=22, lightness_tol=45, min_share=0.0):
    """Learn what the photo's surface looks like. k-means in Lab colour; the surface is the largest
    cluster plus clusters (at least `min_share` of the photo, default any size) of similar chroma (a, b)
    and lightness, so the light and dark grain of wood count as surface. Known limit: a black piece on a
    mid-coloured surface is about the colour of that surface's deep shadows and can be missed.
    Returns (cluster centres, indices of the surface clusters)."""
    h_img, w_img = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    step = max(1, min(h_img, w_img) // 300)
    small = cv2.resize(lab, (max(1, w_img // step), max(1, h_img // step)), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    cv2.setRNGSeed(0)                                             # k-means is random: keep results repeatable
    _, labels, centers = cv2.kmeans(small, clusters, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                    cv2.KMEANS_PP_CENTERS)
    sizes = np.bincount(labels.ravel(), minlength=clusters) / len(labels)
    big = centers[int(np.argmax(sizes))]
    surface = [c for c in range(clusters)
               if (c == int(np.argmax(sizes)) or sizes[c] >= min_share)
               and np.hypot(centers[c][1] - big[1], centers[c][2] - big[2]) < chroma_tol
               and abs(centers[c][0] - big[0]) < lightness_tol]
    return centers, surface


def _surface_chroma_distance(img, model):
    """Per-pixel distance from the surface in chroma only (a, b). Shadows keep the surface's chroma, so
    they count as surface; used to ask "is this ordinary surface?" without assuming it is white."""
    centers, surface = model
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 2)
    dist = np.full(img.shape[:2], np.inf, np.float32)
    for c in surface:
        dist = np.minimum(dist, np.hypot(lab[:, :, 1] - centers[c][1], lab[:, :, 2] - centers[c][2]))
    return dist


def surface_defeats_seeds(img, sat=110, dark=60, frac=0.35):
    """True when a large share of the photo is strongly saturated or very dark, i.e. the surface itself
    is coloured (a wooden table) or dark (black cloth). The seed-based colour method cannot work there:
    the surface looks like one huge piece."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 1] > sat) | (hsv[:, :, 2] < dark)).mean()) > frac


def surface_is_dim(img, median=125):
    """True when the photo's typical brightness (median value) is low: dark mesh or cloth, even in a brighter exposure. Such
    a surface is not dark enough for `surface_defeats_seeds`, but its pockets and the pieces' shadows become colour seeds that
    bridge neighbouring pieces into one box. The surface method leads there. (The same mesh photographed brighter has a median
    of about 115, darker about 70; towels and white sheets are 150 or more.)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.median(hsv[:, :, 2])) < median


def _surface_distance(img, model, lightness_weight=1.0):
    """Per-pixel distance (Lab units, blurred) to the nearest surface cluster; chroma counts fully, lightness `lightness_weight`."""
    centers, surface = model
    lab = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 2)
    dist = np.full(img.shape[:2], np.inf, np.float32)
    for c in surface:
        d = np.sqrt((lab[:, :, 1] - centers[c][1]) ** 2 + (lab[:, :, 2] - centers[c][2]) ** 2
                    + (lightness_weight * (lab[:, :, 0] - centers[c][0])) ** 2)
        dist = np.minimum(dist, d)
    return dist


SURFACE_THRESHOLD = 35


def _surface_regions(img, min_area_frac=0.0005, pad_frac=0.08, clusters=5, chroma_tol=22,
                           lightness_tol=45, threshold=35, lightness_weight=1.0, surface_model=None, edge_contact=0.2):
    """The surface method's detections as masks; see `find_pieces_by_surface`."""
    h_img, w_img = img.shape[:2]
    centers, surface = surface_model or _surface_model(img, clusters, chroma_tol, lightness_tol)

    dist = _surface_distance(img, (centers, surface), lightness_weight)
    mask = (dist > threshold).astype(np.uint8) * 255
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k + 2, k + 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * k + 3, 2 * k + 3), np.uint8), iterations=2)

    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    min_area = min_area_frac * h_img * w_img
    limit = GIANT_BOX_FRAC * h_img * w_img
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area or w * h > limit:
            continue
        if x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img:
            # Touching the photo edge: the table beyond the sheet, or a piece the frame cuts off. A piece that only just
            # reaches the edge with a corner (a short contact) is still a whole piece; a long contact is the table.
            sel = lab[y:y + h, x:x + w] == i
            contact = max(int(sel[0].sum()) if y <= 0 else 0, int(sel[-1].sum()) if y + h >= h_img else 0,
                          int(sel[:, 0].sum()) if x <= 0 else 0, int(sel[:, -1].sum()) if x + w >= w_img else 0)
            if contact > edge_contact * max(w, h):
                continue
        out.append(seg.Region(seg.fill_holes(lab[y:y + h, x:x + w] == i), x, y, "surface"))
    return _order_regions(out, pad_frac, img.shape)


def find_pieces_by_surface(img, min_area_frac=0.0005, pad_frac=0.08, clusters=5, chroma_tol=22,
                           lightness_tol=45, threshold=35, lightness_weight=1.0, surface_model=None, edge_contact=0.2):
    """For photos whose surface is itself coloured or textured (wood grain), where any "is it strongly
    coloured" test fails because the surface is.

    Learns what the surface looks like and treats whatever differs from it as a piece:
    1. k-means on the photo in Lab colour. The surface is the largest cluster plus clusters of similar
       chroma (a, b) and lightness, which is how the light and dark grain of wood count as surface.
    2. Each pixel's distance to the nearest surface cluster (chroma counts fully, lightness a bit
       less, so the grain's light/dark variation does not look like an object).
    3. Pixels further than `threshold` are foreground; components are the pieces. Components touching
       the photo edge (the table's edge) and anything over 25% of the photo are dropped.
    White, pale and dark pieces differ in chroma or lightness from a brown surface and are found."""
    return _boxes_of(_surface_regions(img, min_area_frac, pad_frac, clusters, chroma_tol, lightness_tol, threshold,
                                      lightness_weight, surface_model, edge_contact), pad_frac, img.shape)


def _area(b):
    return b[2] * b[3]


def _overlap(a, b):
    """Intersection area of two (x, y, w, h) boxes."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _iou(a, b):
    inter = _overlap(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union else 0.0


def _implausible(boxes, shape):
    """No boxes, or one 'piece' that is most of the photo: the background was misjudged."""
    if not boxes:
        return True
    h_img, w_img = shape[:2]
    return any(w * h > GIANT_BOX_FRAC * h_img * w_img for _, _, w, h in boxes)


def split_by_studs(boxes, groups, shape, pad_frac=0.08, cover=0.5):
    """A box that holds two or more stud groups that cannot be parts of one piece (different colours, or grids
    that are not aligned / not one lattice) is replaced by the groups' own boxes, the way a box holding two
    colour boxes is split. Only done when the groups account for at least `cover` of the box, so a piece the
    studs say nothing about (a tile) is not thrown away with the split."""
    if not groups:
        return list(boxes)
    h_img, w_img = shape[:2]
    out = []
    for b in boxes:
        x, y, w, h = b
        inner = [g for g in groups if x <= g.centre[0] <= x + w and y <= g.centre[1] <= y + h
                 and len(g.studs) >= studmod.CFG.min_split_studs]
        if len(inner) < 2 or not any(studmod.distinct_pieces(inner[0], g) for g in inner[1:]):
            out.append(b)
            continue
        parts = []
        for g in inner:
            gx, gy, gw, gh = g.box
            p = int(max(gw, gh) * pad_frac)
            x0, y0 = max(0, gx - p), max(0, gy - p)
            parts.append((x0, y0, min(w_img, gx + gw + p) - x0, min(h_img, gy + gh + p) - y0))
        if sum(_area(q) for q in parts) >= cover * _area(b):
            out.extend(parts)
        else:
            out.append(b)
    return out


def combine_boxes(background_boxes, colour_boxes, shape, inside=0.6, duplicate_iou=0.3, duplicate_inside=0.7,
                  stud_groups=None):
    """Merge the two OpenCV detectors' results.

    - The background-colour method's boxes are kept, except that a box which holds two or more
      colour-method boxes is two pieces merged together, and is replaced by those boxes.
    - Colour-method boxes the other method missed (not a duplicate of a kept box) are added.
    - Anything bigger than GIANT_BOX_FRAC of the photo is dropped from both.
    - With `stud_groups`, a box (from either method) holding two or more stud groups that are different
      colours or not one lattice is first split along those groups (see `split_by_studs`)."""
    h_img, w_img = shape[:2]
    limit = GIANT_BOX_FRAC * h_img * w_img
    bg = [b for b in background_boxes if _area(b) <= limit]
    col = [b for b in colour_boxes if _area(b) <= limit]
    if stud_groups:
        bg, col = split_by_studs(bg, stud_groups, shape), split_by_studs(col, stud_groups, shape)
    kept = []
    for b in bg:
        inner = [c for c in col if _overlap(c, b) >= inside * _area(c)]
        # two colour boxes that largely overlap each other (a stud region and a shadow region of ONE piece) are not
        # evidence that the box holds two pieces
        separate = all(_overlap(x, y) < 0.5 * min(_area(x), _area(y)) for i, x in enumerate(inner) for y in inner[i + 1:])
        kept.extend(inner if len(inner) >= 2 and separate else [b])
    for c in col:
        if not any(_iou(c, k) >= duplicate_iou or _overlap(c, k) >= duplicate_inside * _area(c) for k in kept):
            kept.append(c)
    unique = []
    for b in kept:                                              # a colour box can be added by a split and again above
        if b not in unique:
            unique.append(b)
    return _reading_order(unique)


def _is_edge_strip(box, shape, span=0.7):
    """A box hugging the photo edge and spanning most of it: a lighting gradient or vignette, not a piece."""
    h_img, w_img = shape[:2]
    x, y, w, h = box
    touches = x <= 0 or y <= 0 or x + w >= w_img or y + h >= h_img
    return touches and (w >= span * w_img or h >= span * h_img)


def _merge_fragments(boxes, overlap=0.25):
    """Union boxes that substantially overlap (one piece found in several parts, e.g. its stud row and its
    body). Mere neighbours, whose padded boxes overlap only slightly, are left alone."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if _overlap(a, b) >= overlap * min(_area(a), _area(b)):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = (x0, y0, x1 - x0, y1 - y0)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def add_missing(kept, candidates, shape, contains=0.5, duplicate_iou=0.3, duplicate_inside=0.7,
                fragment_overlap=0.25):
    """Add candidate boxes only where nothing was found. A candidate is skipped when it duplicates a kept
    box, lies mostly inside one, or CONTAINS kept boxes (a blob that merely surrounds pieces already found,
    e.g. pieces bridged by their shadows), and anything over GIANT_BOX_FRAC of the photo is dropped."""
    h_img, w_img = shape[:2]
    limit = GIANT_BOX_FRAC * h_img * w_img
    out = list(kept)
    for c in _merge_fragments(candidates):
        if _area(c) > limit or c in out:
            continue
        if any(_iou(c, k) >= duplicate_iou or _overlap(c, k) >= duplicate_inside * _area(c)
               or _overlap(c, k) >= contains * _area(k) for k in kept):
            continue
        # Partly overlapping a box we already have: the same piece found in two parts (its stud row by one
        # method, its body by the other), so join them rather than add a second piece.
        for idx, k in enumerate(out):
            if _overlap(c, k) >= fragment_overlap * min(_area(c), _area(k)):
                x0, y0 = min(c[0], k[0]), min(c[1], k[1])
                x1, y1 = max(c[0] + c[2], k[0] + k[2]), max(c[1] + c[3], k[1] + k[3])
                out[idx] = (x0, y0, x1 - x0, y1 - y0)
                break
        else:
            out.append(c)
    return _reading_order(out)


# ---------------------------------------------------------------- the same steps on masks
def _too_big(r, shape):
    return r.w * r.h > GIANT_BOX_FRAC * shape[0] * shape[1]


def _uniq(regions):
    out = []
    for r in regions:
        if not any(r is o for o in out):
            out.append(r)
    return out


def combine_regions(background, colour, shape, inside=0.6, duplicate_iou=0.3, duplicate_inside=0.7):
    """`combine_boxes` on masks: a background region that holds two or more separate colour regions is two pieces merged and is replaced
    by them; colour regions the background method missed (not a duplicate of a kept region) are added."""
    bg = [r for r in background if not _too_big(r, shape)]
    col = [r for r in colour if not _too_big(r, shape)]
    kept = []
    for b in bg:
        inner = [c for c in col if seg.overlap_px(c, b) >= inside * c.area]
        # two colour regions that largely overlap each other (a stud region and a shadow region of ONE piece) are not
        # evidence that the region holds two pieces
        separate = all(seg.overlap_px(x, y) < 0.5 * min(x.area, y.area) for i, x in enumerate(inner) for y in inner[i + 1:])
        kept.extend(inner if len(inner) >= 2 and separate else [b])
    for c in col:
        if not any(seg.mask_iou(c, k) >= duplicate_iou or seg.overlap_px(c, k) >= duplicate_inside * c.area for k in kept):
            kept.append(c)
    return _uniq(kept)


def prefer_surface_regions(colour, surface, shape, inside=0.6, min_inside=2):
    """`prefer_surface_over_merged` on masks: a colour region holding two or more separate surface regions is a merge."""
    min_side = 0.04 * min(shape[:2])
    surf = [r for r in surface if not _too_big(r, shape) and min(r.w, r.h) >= min_side]
    out = []
    for c in colour:
        held = [s for s in surf if seg.overlap_px(s, c) >= inside * s.area and s.area < 0.8 * c.area]
        separate = all(seg.overlap_px(a, b) < 0.5 * min(a.area, b.area) for i, a in enumerate(held) for b in held[i + 1:])
        if len(held) >= min_inside and separate:
            out.extend(b for b in held if not any(b is o for o in out))
        else:
            out.append(c)
    return out


def grow_stud_only_regions(regions, others, shape, min_overlap=0.25, pitch=None, reach=1.5):
    """A region found from studs alone covers only a piece's top face: where another method saw the whole blob (its mask overlaps, or
    lies within `reach` pitches: a pale piece's body between its studs and its shadow is not seen by anything), use the union."""
    out = []
    for r in regions:
        if r.source == "studs-only":
            for c in others:
                bridge = any(o is not r and seg.overlap_px(o, c) >= 0.5 * o.area for o in regions)   # c also holds other pieces
                if bridge or _too_big(c, shape):
                    continue
                if seg.overlap_px(r, c) >= min_overlap * min(r.area, c.area):
                    r = seg.union_region(r, c, "studs-only")
                elif pitch and seg.touching(r, c, int(reach * pitch)):
                    r = seg.union_region(r, c, "studs-only", bridge=int(reach * pitch))
        out.append(r)
    return out


def dedupe_regions(regions, min_iou=0.5):
    """Two masks that cover almost the same area are ONE piece found twice: keep their union."""
    regions = list(regions)
    changed = True
    while changed:
        changed = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if seg.mask_iou(regions[i], regions[j]) >= min_iou:
                    regions[i] = seg.union_region(regions[i], regions[j], regions[i].source)
                    del regions[j]
                    changed = True
                    break
            if changed:
                break
    return regions


def stud_regions(groups, shape):
    """A stud group's footprint (the piece top) as a mask region."""
    out = []
    h_img, w_img = shape[:2]
    for g in groups:
        if len(g.studs) < studmod.CFG.min_seed_studs:
            continue                                              # one circle is too weak to be a piece on its own
        full = np.zeros((h_img, w_img), np.uint8)
        cv2.fillPoly(full, [np.rint(g.footprint).astype(np.int32)], 1)
        r = seg.Region.from_full(full, "studs-only")
        if r is not None:
            out.append(r)
    return out


def regions_from_split_boxes(regions, in_boxes, out_boxes):
    """`split_merged_by_colour` works on boxes: map its answer back onto masks. Boxes it left alone keep their region; a region it
    split into several boxes is divided between them (every pixel goes to the nearest part)."""
    parts_of = {}
    for ob in out_boxes:
        if ob in in_boxes:
            parts_of.setdefault(in_boxes.index(ob), []).append(None)
            continue
        best, best_px = None, 0
        for i, r in enumerate(regions):
            x0, y0 = max(r.x, ob[0]), max(r.y, ob[1])
            x1, y1 = min(r.x + r.w, ob[0] + ob[2]), min(r.y + r.h, ob[1] + ob[3])
            if x1 > x0 and y1 > y0:
                px = int(r.mask[y0 - r.y:y1 - r.y, x0 - r.x:x1 - r.x].sum())
                if px > best_px:
                    best, best_px = i, px
        if best is not None:
            parts_of.setdefault(best, []).append(ob)
    out = []
    for i, r in enumerate(regions):
        boxes = parts_of.get(i, [])
        if not boxes:
            continue
        if boxes == [None]:
            out.append(r)
            continue
        seeds = []
        for ob in boxes:
            if ob is None:
                continue
            sx, sy = ob[2] // 4, ob[3] // 4
            s = np.zeros(r.mask.shape, np.uint8)
            s[max(0, ob[1] + sy - r.y):max(0, ob[1] + ob[3] - sy - r.y), max(0, ob[0] + sx - r.x):max(0, ob[0] + ob[2] - sx - r.x)] = 1
            seeds.append(s & r.mask)
        seeds = [s for s in seeds if s.any()]
        if len(seeds) < 2:
            out.append(r)
            continue
        for m in seg.partition_by_seeds(r.mask, seeds):
            sub = r.derive(m)
            if sub is not None:
                sub.rules = list(r.rules) + ["split:colour-boxes"]
                out.append(sub)
    return out


# ---------------------------------------------------------------- the whole pipeline
@dataclass
class Piece:
    """One detected piece: its box, which method found it, and what the studs say about it."""
    box: tuple
    source: str                                   # background | colour | colour+studs | studs-only | surface | merged
    studs: studmod.StudInfo = field(default_factory=studmod.StudInfo)
    region: object = None                         # segment.Region: the piece's mask (None for boxes from other detectors)
    rules: list = field(default_factory=list)     # split / merge rules that shaped it (debug labels)
    check: bool = False                           # ambiguous or overlapping: the UI shows "check this"
    pitch: float | None = None                    # local stud pitch in px


@dataclass
class Analysis:
    pieces: list
    studs: list = field(default_factory=list)
    groups: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)   # seconds per stage
    alignment: str = ""                            # reference mode: how the empty photo was aligned
    changed: float = 0.0                           # reference mode: share of the photo that differs from the empty photo
    matched: bool = True                           # reference mode: False = the empty photo is not of this scene (camera moved...)
    match: float | None = None                     # reference mode: surface-pattern correlation with the empty photo
    ambiguities: list = field(default_factory=list)   # segment.Ambiguity: splits the evidence only just supports (tie-break)

    @property
    def boxes(self):
        return [p.box for p in self.pieces]


def _stud_boxes(groups, shape, pad_frac):
    out = []
    h_img, w_img = shape[:2]
    for g in groups:
        x, y, w, h = g.box
        p = int(max(w, h) * pad_frac)
        x0, y0 = max(0, x - p), max(0, y - p)
        out.append((x0, y0, min(w_img, x + w + p) - x0, min(h_img, y + h + p) - y0))
    return out


def _find_studs_safe(img, model, cfg=None):
    """Studs are a signal: any failure here means "no studs", never a failed scan."""
    cfg = cfg or studmod.CFG
    if not cfg.enabled:
        return [], []
    try:
        return studmod.find_stud_groups(img, cfg, surface=model)
    except Exception as e:                                         # noqa: BLE001
        log.warning("stud detection failed (%s); continuing without studs", e)
        return [], []


def merge_stud_fragments(boxes, groups, shape, pad_frac=0.08, whole=0.5, inside=0.7):
    """One piece found in parts: a pale piece whose studs have shaded rims often yields a box for the piece AND a box
    for each stud (or boxes for stud rows). All of them hold studs of ONE stud group, so they are one piece:
    - if one box holds at least `whole` of the group's studs, the other boxes lying mostly inside it are dropped;
    - if none does, boxes lying mostly inside the group's own box are replaced by that box."""
    h_img, w_img = shape[:2]
    out = list(boxes)
    for g in groups:
        if len(g.studs) < studmod.CFG.min_split_studs:
            continue

        def held(b):
            return sum(1 for st in g.studs if b[0] <= st.x <= b[0] + b[2] and b[1] <= st.y <= b[1] + b[3])
        holders = [b for b in out if held(b)]
        if len(holders) < 2:
            continue
        big = max(holders, key=lambda b: (held(b), _area(b)))
        if held(big) >= whole * len(g.studs):
            drop = [b for b in holders if b is not big and _overlap(b, big) >= inside * _area(b)]
        else:
            gx, gy, gw, gh = g.box
            p = int(max(gw, gh) * pad_frac)
            x0, y0 = max(0, gx - p), max(0, gy - p)
            gbox = (x0, y0, min(w_img, gx + gw + p) - x0, min(h_img, gy + gh + p) - y0)
            drop = [b for b in holders if _overlap(b, gbox) >= inside * _area(b)]
            if len(drop) >= 2:
                out = [b for b in out if b not in drop] + [gbox]
                continue
            drop = []
        out = [b for b in out if not any(b is d for d in drop)]
    return _reading_order(out)


def split_merged_by_colour(img, boxes, pad_frac=0.08, outlier=1.8, min_share=0.08, hue_gap=20, sat=110, cover=0.6,
                           small_share=0.15):
    """Surface-method boxes can still hold two touching pieces (a lime plate lying against a red brick). A box much
    bigger than the photo's typical box (`outlier` x the median area) that holds two or more saturated regions of
    clearly different hue, each at least `min_share` of the box, is split along them. Only done when those regions
    account for at least `cover` of the box. A normal-sized box is split too when it holds two clearly separate regions (a grey
    brick and a small blue one a few pixels apart, which the gap-closing of the surface method joined): each region must be a
    real share of the box (`small_share`) and the regions must not overlap each other. Pale pieces in the box (white, light grey: much brighter than the surface and
    unsaturated, like a white slope lying beside a red brick) are not lost in the split: they become boxes of their own."""
    if len(boxes) < 3:
        return list(boxes)
    h_img, w_img = img.shape[:2]
    median = float(np.median([_area(b) for b in boxes]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pale_v = max(1.35 * float(np.median(hsv[:, :, 2])), 150.0)
    k = max(3, int(min(h_img, w_img) * 0.004)) | 1
    out = []
    for b in boxes:
        x, y, w, h = b
        big = _area(b) >= outlier * median
        share = min_share if big else small_share
        sub = hsv[y:y + h, x:x + w]
        strong = (sub[:, :, 1] > sat) & (sub[:, :, 2] > 80)
        hue_class = ((sub[:, :, 0].astype(int) + hue_gap // 2) % 180) // hue_gap      # circular: red wraps round
        regions = []                                     # (bounding box, pixels)
        for c in np.unique(hue_class[strong]):
            mask = (strong & (hue_class == c)).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                bx, by, bw, bh, area = (int(v) for v in st[i])
                if area >= share * 0.5 * w * h:
                    regions.append(((x + bx, y + by, bw, bh), lab == i))
        merged, pixels = [], np.zeros(sub.shape[:2], bool)   # one colour straddling two hue classes is one region
        for r, pix in sorted(regions, key=lambda t: _area(t[0]), reverse=True):
            if not any(_overlap(r, m) >= 0.6 * min(_area(r), _area(m)) for m in merged):
                merged.append(r)
                pixels |= pix
        pale = []
        if merged:
            mask = ((sub[:, :, 2] > pale_v) & (sub[:, :, 1] < 70)).astype(np.uint8)
            mask[cv2.dilate(pixels.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0] = 0   # not what a coloured region covers
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                bx, by, bw, bh, area = (int(v) for v in st[i])
                if area >= (0.15 * median if big else share * w * h) \
                        and not any(_overlap((x + bx, y + by, bw, bh), m) >= 0.5 * bw * bh for m in merged):
                    pale.append((x + bx, y + by, bw, bh))
        found = merged + pale
        separate = all(_overlap(a, c) < 0.2 * min(_area(a), _area(c)) for i, a in enumerate(found) for c in found[i + 1:])
        if len(found) >= 2 and (big or separate):
            parts = []
            for bx, by, bw, bh in found:
                pd = int(max(bw, bh) * pad_frac)
                x0, y0 = max(0, bx - pd), max(0, by - pd)
                parts.append((x0, y0, min(w_img, bx + bw + pd) - x0, min(h_img, by + bh + pd) - y0))
            if len(parts) >= 2 and sum(_area(q) for q in parts) >= (cover if big else 0.4 * cover) * _area(b):     # a normal box is mostly margin
                out.extend(parts)
                continue
        out.append(b)
    return out


def prefer_surface_over_merged(colour_boxes, surface_boxes, shape, inside=0.6, min_inside=2):
    """A colour box that holds two or more of the surface method's boxes is a merge (the colour seeds bridged
    neighbouring pieces through shadows or texture): replace it by the surface boxes it holds."""
    min_side = 0.04 * min(shape[:2])                       # shadows and creases come out as thin slivers: not pieces
    surf = [b for b in surface_boxes if _area(b) <= GIANT_BOX_FRAC * shape[0] * shape[1] and min(b[2], b[3]) >= min_side]
    out = []
    for c in colour_boxes:
        held = [b for b in surf if _overlap(b, c) >= inside * _area(b) and _area(b) < 0.8 * _area(c)]
        separate = all(_overlap(a, b) < 0.5 * min(_area(a), _area(b)) for i, a in enumerate(held) for b in held[i + 1:])
        if len(held) >= min_inside and separate:
            out.extend(b for b in held if b not in out)
        else:
            out.append(c)
    return out


def dedupe_near_identical(boxes, min_iou=0.5):
    """Two boxes that cover almost the same area are ONE piece found twice (e.g. a white slope boxed by the surface method and
    again when a merged box was split): keep their union. Only near-identical boxes (IoU >= `min_iou`) merge, so a small piece
    next to a big one is never swallowed."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _iou(boxes[i], boxes[j]) >= min_iou:
                    a, b = boxes[i], boxes[j]
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    boxes[i] = (x0, y0, max(a[0] + a[2], b[0] + b[2]) - x0, max(a[1] + a[3], b[1] + b[3]) - y0)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return _reading_order(boxes)


def _grow_stud_only(boxes, labelled, others, shape, min_overlap=0.25):
    """A box found from studs alone covers only the studs of its piece (the top face). Where another method saw the
    whole blob (surface or background box), the stud box and that box are two parts of ONE piece: use their union
    instead of leaving the fragment to block the whole box."""
    limit = GIANT_BOX_FRAC * shape[0] * shape[1]
    out = []
    for b in boxes:
        if _source_for(b, labelled) == "studs-only":
            for c in others:
                if _area(c) <= limit and _overlap(b, c) >= min_overlap * min(_area(b), _area(c)):
                    x0, y0 = min(b[0], c[0]), min(b[1], c[1])
                    b = (x0, y0, max(b[0] + b[2], c[0] + c[2]) - x0, max(b[1] + b[3], c[1] + c[3]) - y0)
        out.append(b)
    return out


def _with_studs(fn, default, boxes_or_none, groups, *args, fallback=None):
    """Run a stud-assisted step; if it raises, log it and carry on without studs (a scan must never fail on them).
    Two call shapes: `_with_studs(split_by_studs, boxes, boxes, groups, shape, pad)` returns `boxes` unchanged on
    failure; `_with_studs(lambda g: ..., None, None, groups, fallback=f)` returns `f()` instead."""
    try:
        if boxes_or_none is None:
            return fn(groups)
        return fn(default, groups, *args)
    except Exception as e:                                         # noqa: BLE001
        log.warning("stud-assisted step failed (%s); continuing without studs", e)
        return fallback() if fallback else default


def _stud_info_safe(box, groups, cfg):
    try:
        return studmod.stud_info_for_box(box, groups, cfg)
    except Exception:                                              # noqa: BLE001
        return studmod.StudInfo()


def _source_for(box, candidates):
    """Label of the method whose box best matches `box` (`candidates`: [(box, label)])."""
    best, label = 0.0, "merged"
    for b, name in candidates:
        score = max(_iou(box, b), _overlap(box, b) / max(1, min(_area(box), _area(b))) * 0.9)
        if score > best and score >= 0.45:
            best, label = score, name
    return label


def _drop_blobs(pieces, img, cfg):
    """Studs as a signal: a small box with NO studs and weak colour evidence is most likely a shadow, a fold or a
    texture blob. Boxes with studs, or with real colour or darkness in them, are never dropped for lack of studs."""
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    strong = (hsv[:, :, 1] > 110) | (hsv[:, :, 2] < 60)
    out = []
    for p in pieces:
        x, y, w, h = p.box
        if (p.studs.stud_count == 0 and w * h < cfg.drop_small_frac * h_img * w_img
                and float(strong[y:y + h, x:x + w].mean()) < cfg.drop_weak_colour):
            continue
        out.append(p)
    return out


def _edge_remnant(region, shape, thin=1.5):
    """A part a split cut off against the photo's edge that is only a strip (thinner than `thin` pitches): the frame's lighting gradient
    or the table's edge joined to a piece, not a piece. Regions no rule split are left as the methods found them."""
    if not any(r.startswith("split") for r in region.rules):
        return False
    h_img, w_img = shape[:2]
    if not (region.x <= 0 or region.y <= 0 or region.x + region.w >= w_img or region.y + region.h >= h_img):
        return False
    side = min(seg.rotated_rect(region)[1])
    return side < thin * (region.pitch or 0.06 * min(h_img, w_img))


def _drop_shadow_regions(regions, scene, share=0.7):
    """A region that is almost entirely surface-looking (a piece's cast shadow or glow, found as a ring or strip) lying against another
    region is that region's shadow, not a piece."""
    out = []
    for r in regions:
        on_surface = float(scene.surface[r.y:r.y + r.h, r.x:r.x + r.w][r.mask > 0].mean())
        if on_surface >= share and not seg.groups_in(r, scene) and any(o is not r and seg.touching(r, o, 4) for o in regions):
            continue
        out.append(r)
    return out


def _finish(regions, img, scene, groups, cfg, pad_frac):
    """Sections 3-6 on the regions every method has produced: split where there is evidence, merge what is one piece, flag what is
    unsure. -> (pieces, ambiguities)"""
    regions = dedupe_regions([r for r in regions if r is not None])
    seg.resolve_pitches(regions, scene)
    regions = seg.absorb_wrapped(regions, scene)
    split = []
    for r in regions:
        split.extend(seg.split_region(r, scene))
    regions = seg.merge_pass(split, scene)
    regions = [r for r in regions if not _edge_remnant(r, img.shape)]
    regions = _drop_shadow_regions(regions, scene)
    regions = _order_regions(regions, pad_frac, img.shape)
    seg.mark_overlaps(regions)
    pieces = []
    for r in regions:
        box = seg.padded_box(r, pad_frac, img.shape)
        pieces.append(Piece(box, r.source, _stud_info_safe(box, groups, cfg), r, list(r.rules), r.check, r.pitch))
    if cfg.enabled:
        pieces = _drop_blobs(pieces, img, cfg)
    by_amb = {}
    for p in pieces:
        if p.region.ambig is not None:
            by_amb.setdefault(id(p.region.ambig), (p.region.ambig, []))[1].append(p.region)
    ambiguities = []
    for amb, parts in by_amb.values():
        if len(parts) >= 2:
            amb.parts = parts
            ambiguities.append(amb)
        else:
            for r in parts:
                r.ambig = None
    ambiguities.sort(key=lambda a: a.margin)
    live = {id(r) for a in ambiguities for r in a.parts}
    for p in pieces:
        p.check = p.check and (id(p.region) in live or "overlap" in p.rules)
        p.region.check = p.check
    return pieces, ambiguities


def analyse_masks(img, min_area_frac=0.0005, pad_frac=0.08, debug=None, cfg=None, reference=None):
    """The whole OpenCV pipeline with details: pieces (box, mask, source, stud info), the studs and stud groups, and the
    time each stage took. `find_pieces` is this, reduced to the boxes.

    Every detection is a MASK (see segment.py); boxes are derived from it last. Each method finds regions, the regions are combined by
    mask coverage, and then:
    - each connected region is ONE piece unless a split rule has evidence (seam, stud grids, colour, shape, size);
    - neighbours that are really one piece are merged and re-checked;
    - a split the evidence only just supports is flagged ambiguous (tie-break with Brickognize, then "check this").

    - If the surface is neither strongly coloured nor dark, the two colour-based methods are combined
      (see `combine_regions`); the background-colour method is trusted only when its result is plausible.
    - If it is (wood, black cloth), colour seeds cannot work, so the surface method is used alone, with stud groups
      adding pieces it missed.
    - In the first case the surface method also runs and fills gaps (`seg.add_missing_masks`): pieces the colour
      seeds cannot see, e.g. white or grey ones on a coloured surface.
    - Studs (see studs.py) are found on every scan and work with the colour seeds (`find_pieces_by_colour`), feed the stud-grid split
      rule, give the local pitch every rule is measured in, and give every box a stud score.

    With `reference` (a photo of the same empty surface from the same place) the pieces are what differs from it
    (reference.py) and nothing else runs: no colour seeds, no studs.

    `debug` (a file path, or the DETECT_DEBUG_DIR environment variable's directory) saves an overlay image."""
    if reference is not None:
        return _analyse_reference(img, reference, min_area_frac, pad_frac, debug)
    cfg = cfg or studmod.CFG
    t = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        t[name] = round(now - clock, 4)
        clock = now

    model = _surface_model(img)
    lap("surface_model")
    studs, groups = _find_studs_safe(img, model, cfg)
    lap("studs")
    scene = seg.Scene.build(img, _surface_distance(img, model) < SURFACE_THRESHOLD, groups,
                            float(cfg.tilt_stretch or 1.0) if cfg.enabled else 1.0,
                            _surface_chroma_distance(img, model))
    raw = _background_regions(img, min_area_frac, pad_frac)
    trusted = bool(raw) and not any(_too_big(r, img.shape) for r in raw)       # judged on the raw result: a whole-photo box means failure
    background = [r for r in raw if not _is_edge_strip(seg.padded_box(r, pad_frac, img.shape), img.shape)]
    lap("background")
    surface = _surface_regions(img, min_area_frac, pad_frac, surface_model=model)
    lap("surface")
    useful = [g for g in groups if not g.on_background]         # on a coloured surface, a background-coloured group is grain
    pitch_hint = float(np.median([g.pitch for g in groups])) if groups else None

    def colour_regions():
        return _with_studs(lambda g: _colour_regions(img, min_area_frac, pad_frac, stud_groups=g, surface_model=model), None, None,
                           groups, fallback=lambda: _colour_regions(img, min_area_frac, pad_frac, surface_model=model))

    if surface_defeats_seeds(img):
        found = surface if surface else [r for r in background if not _too_big(r, img.shape)]
        candidates = grow_stud_only_regions(stud_regions(useful, img.shape), found, img.shape, pitch=pitch_hint)
        regions = seg.add_missing_masks(found, candidates, scene, pitch_hint)
    elif surface_is_dim(img):
        # Dim textured surface (dark mesh): the surface method leads. Colour regions are added only where the surface
        # method saw nothing at all (e.g. a black piece on dark grey).
        colour = colour_regions()
        lap("colour")
        min_side = 0.04 * min(img.shape[:2])
        surface = [r for r in surface if min(r.w, r.h) >= min_side]          # shadow slivers are not pieces
        in_boxes = _boxes_of(surface, pad_frac, img.shape)
        if USE_BOX_COLOUR_SPLIT and surface:
            surface = regions_from_split_boxes(surface, in_boxes, split_merged_by_colour(img, in_boxes, pad_frac))
        extra = []
        for c in colour:
            cb = seg.padded_box(c, pad_frac, img.shape)
            if (not _too_big(c, img.shape) and all(seg.overlap_px(c, k) < 0.1 * c.area for k in surface)
                    and cb[0] > 0 and cb[1] > 0 and cb[0] + cb[2] < img.shape[1] and cb[1] + cb[3] < img.shape[0]):   # edge: not a piece
                extra.append(c)
        regions = seg.add_missing_masks(surface, extra, scene, pitch_hint)
    else:
        colour = colour_regions()
        lap("colour")
        if not trusted:
            primary = [r for r in colour if not _too_big(r, img.shape)] or [r for r in background if not _too_big(r, img.shape)]
        else:
            primary = combine_regions(background, colour, img.shape)
        primary = prefer_surface_regions(primary, surface, img.shape)
        primary = grow_stud_only_regions(primary, surface, img.shape, pitch=pitch_hint)
        regions = seg.add_missing_masks(primary, surface, scene, pitch_hint)
    lap("regions")
    pieces, ambiguities = _finish(regions, img, scene, groups, cfg, pad_frac)
    lap("combine")
    result = Analysis(pieces, studs, groups, t, ambiguities=ambiguities)
    _write_debug(img, result, debug)
    return result


def analyse_legacy(img, min_area_frac=0.0005, pad_frac=0.08, debug=None, cfg=None, reference=None):
    """The detector as it was before the mask rewrite (box logic: `combine_boxes`, `add_missing`, `split_by_studs`...). Same result type.

    - If the surface is neither strongly coloured nor dark, the two colour-based methods are combined
      (see `combine_boxes`); the background-colour method is trusted only when its result is plausible.
    - If it is (wood, black cloth), colour seeds cannot work, so the surface method is used alone, with stud groups
      splitting merged surface regions and adding pieces it missed.
    - In the first case the surface method also runs and fills gaps (`add_missing`).
    - Studs (see studs.py) work with the colour seeds, split boxes that hold two pieces, and give every box a stud score."""
    global COMPONENT_GROUPING
    if reference is not None:
        return _analyse_reference(img, reference, min_area_frac, pad_frac, debug)
    cfg = cfg or studmod.CFG
    t = {}
    clock = time.perf_counter()

    def lap(name):
        nonlocal clock
        now = time.perf_counter()
        t[name] = round(now - clock, 4)
        clock = now

    saved, COMPONENT_GROUPING = COMPONENT_GROUPING, False        # the old colour method grouped seeds by distance only
    try:
        model = _surface_model(img)
        lap("surface_model")
        studs, groups = _find_studs_safe(img, model, cfg)
        lap("studs")
        raw = find_pieces_on_plain_background(img, min_area_frac, pad_frac)
        trusted = not _implausible(raw, img.shape)                 # judged on the raw result: a whole-photo box means failure
        background = [b for b in raw if not _is_edge_strip(b, img.shape)]
        lap("background")
        surface = find_pieces_by_surface(img, min_area_frac, pad_frac, surface_model=model)
        surface = _with_studs(split_by_studs, surface, surface, groups, img.shape, pad_frac)
        lap("surface")
        useful = [g for g in groups if not g.on_background]         # on a coloured surface, a background-coloured group is grain
        if surface_defeats_seeds(img):
            stud_boxes = _stud_boxes(useful, img.shape, pad_frac)
            boxes = surface if surface else [b for b in background if not _implausible([b], img.shape)]
            boxes = add_missing(boxes, stud_boxes, img.shape)
            labelled = [(b, "surface") for b in surface] + [(b, "background") for b in background] + \
                       [(b, "studs-only") for b in stud_boxes]
        elif surface_is_dim(img):
            colour = find_pieces_by_colour(img, min_area_frac, pad_frac, stud_groups=groups, surface_model=model)
            lap("colour")
            min_side = 0.04 * min(img.shape[:2])
            surface = [b for b in surface if min(b[2], b[3]) >= min_side]          # shadow slivers are not pieces
            surface = split_merged_by_colour(img, surface, pad_frac)
            extra = [c for c in colour if not _implausible([c], img.shape)
                     and all(_overlap(c, k) < 0.1 * _area(c) for k in surface)
                     and c[0] > 0 and c[1] > 0 and c[0] + c[2] < img.shape[1] and c[1] + c[3] < img.shape[0]]
            boxes = add_missing(list(surface), extra, img.shape)
            labelled = [(b, "surface") for b in surface] + [(b, "colour") for b in colour]
        else:
            report = []

            def colour_with(g):
                del report[:]
                return find_pieces_by_colour(img, min_area_frac, pad_frac, stud_groups=g, surface_model=model, report=report)
            colour = _with_studs(lambda g: colour_with(g), None, None, groups, fallback=lambda: colour_with(None))
            lap("colour")
            labelled = list(zip(colour, report)) + [(b, "background") for b in background] + [(b, "surface") for b in surface]
            if not trusted:
                primary = [b for b in colour if not _implausible([b], img.shape)] or \
                          [b for b in background if not _implausible([b], img.shape)]
            else:
                primary = _with_studs(lambda g: combine_boxes(background, colour, img.shape, stud_groups=g), None, None, groups,
                                      fallback=lambda: combine_boxes(background, colour, img.shape))
            primary = prefer_surface_over_merged(primary, surface, img.shape)
            primary = _grow_stud_only(primary, labelled, surface, img.shape)
            boxes = add_missing(primary, surface, img.shape)
        boxes = _with_studs(merge_stud_fragments, boxes, boxes, groups, img.shape, pad_frac)
        boxes = dedupe_near_identical(boxes)
        pieces = [Piece(b, _source_for(b, labelled), _stud_info_safe(b, groups, cfg)) for b in boxes]
        if cfg.enabled:
            pieces = _drop_blobs(pieces, img, cfg)
        lap("combine")
    finally:
        COMPONENT_GROUPING = saved
    result = Analysis(pieces, studs, groups, t)
    _write_debug(img, result, debug)
    return result


def analyse(img, min_area_frac=0.0005, pad_frac=0.08, debug=None, cfg=None, reference=None):
    """The OpenCV pipeline with details (pieces, studs, timings). `ENGINE` picks which one: "mask" (the default: `analyse_masks`, every detection is a mask, split and merged by evidence)
    or "legacy" (the earlier box logic). Set it with the DETECT_ENGINE environment variable."""
    fn = analyse_masks if ENGINE == "mask" else analyse_legacy
    return fn(img, min_area_frac, pad_frac, debug, cfg, reference)


def _write_debug(img, result, debug):
    target = debug if debug is not None else os.environ.get("DETECT_DEBUG_DIR")
    if not target:
        return
    path = Path(target)
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"detect_{time.time_ns()}.jpg"
    try:
        draw_debug(img, result, path)
    except Exception as e:                                         # noqa: BLE001
        log.warning("debug overlay failed: %s", e)


def _analyse_reference(img, reference, min_area_frac, pad_frac, debug):
    from backend.core import reference as refmod
    start = time.perf_counter()
    info = {}
    boxes = refmod.find_pieces_by_reference(img, reference, min_area_frac, pad_frac, info=info)
    result = Analysis([Piece(b, "reference") for b in boxes], [], [],
                      {"reference": round(time.perf_counter() - start, 4)})
    result.alignment = info.get("alignment") or ""
    result.changed = info.get("changed", 0.0)
    result.matched, result.match = info.get("matched", True), info.get("match")
    _write_debug(img, result, debug)
    return result


def find_pieces_detailed(img, min_area_frac=0.0005, pad_frac=0.08):
    """-> list of Piece (box, source, stud info)."""
    return analyse(img, min_area_frac, pad_frac).pieces


def find_pieces(img, min_area_frac=0.0005, pad_frac=0.08):
    """Return a list of (x, y, w, h) boxes, one per piece, whatever the colour of the (uniform) surface.
    See `analyse` for how they are found."""
    return analyse(img, min_area_frac, pad_frac).boxes


SOURCE_COLOURS = {"gemini": (255, 0, 255), "background": (0, 160, 0), "colour": (255, 120, 0),
                  "colour+studs": (0, 200, 255), "studs-only": (0, 0, 255), "surface": (160, 0, 160),
                  "merged": (120, 120, 120), "reference": (0, 140, 255)}


def draw_debug(img, analysis, path, extra_boxes=None, boxes=None, labels=None):
    """Save an overlay: studs (circles), stud groups (one colour per group), and the final detections. A detection with a mask is drawn
    as its mask (one colour each), its rotated rectangle, and a label with its number, the method that found it, the split / merge
    rules that shaped it and the local stud pitch; one without (other detectors) is a plain box.
    `boxes`/`labels` override the final boxes (e.g. the hybrid result); `extra_boxes` are drawn thin in magenta."""
    out = img.copy()
    th = max(2, img.shape[1] // 500)
    rng = np.random.default_rng(7)
    pieces = analysis.pieces if boxes is None else []
    tint = out.copy()
    palette = [tuple(int(v) for v in c) for c in np.random.default_rng(3).integers(60, 255, (max(1, len(pieces)), 3))]
    for p, c in zip(pieces, palette):
        if p.region is not None:
            r = p.region
            roi = tint[r.y:r.y + r.h, r.x:r.x + r.w]
            roi[r.mask > 0] = c
    if pieces:
        out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)
    for s in analysis.studs:
        cv2.circle(out, (int(s.x), int(s.y)), max(2, int(s.r)), (255, 255, 255), th + 1)
    for g in analysis.groups:
        c = tuple(int(v) for v in rng.integers(40, 230, 3))
        for s in g.studs:
            cv2.circle(out, (int(s.x), int(s.y)), max(2, int(s.r)), c, th)
        cv2.polylines(out, [np.rint(g.footprint).astype(np.int32)], True, c, 1)
        cv2.putText(out, f"{g.dims[0]}x{g.dims[1]}", (int(g.centre[0]) - 20, int(g.centre[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1800, (255, 255, 255), th)
    final = boxes if boxes is not None else analysis.boxes
    names = labels if labels is not None else [p.source for p in analysis.pieces]
    scale = img.shape[1] / 1500
    for i, (b, name) in enumerate(zip(final, names), 1):
        x, y, w, h = b
        p = analysis.pieces[i - 1] if boxes is None and i - 1 < len(analysis.pieces) else None
        c = SOURCE_COLOURS.get(name, (120, 120, 120))
        if p is not None and p.region is not None:
            try:
                (rcx, rcy), (rw, rh), ang = seg.rotated_rect(p.region)
                cv2.polylines(out, [np.rint(cv2.boxPoints(((rcx, rcy), (rw, rh), ang))).astype(np.int32)], True, (0, 0, 0), th + 1)
            except Exception:                                      # noqa: BLE001
                pass
            c = (0, 0, 255) if p.check else c
        cv2.rectangle(out, (x, y), (x + w, y + h), c, th)
        tag = f"{i} {name}"
        if p is not None:
            rules = [r for r in p.rules if not r.startswith("pitch")]
            if rules:
                tag += " " + ",".join(rules)
            if p.pitch:
                tag += f" p{p.pitch:.0f}"
            if p.check:
                tag += " CHECK"
        cv2.putText(out, tag, (x + 4, y + 14 + img.shape[1] // 90), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2)
        cv2.putText(out, tag, (x + 4, y + 14 + img.shape[1] // 90), cv2.FONT_HERSHEY_SIMPLEX, scale, c, th)
    for b in extra_boxes or []:
        cv2.rectangle(out, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), SOURCE_COLOURS["gemini"], th)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
