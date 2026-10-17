"""Detection metrics against ground-truth masks.

correct  a real piece matched one-to-one by a detection with mask IoU >= 0.5
split    a real piece covered by two or more detections (each holding at least SHARE of the piece)
merge    a detection that covers two or more real pieces (each at least SHARE of the piece)
missed   a real piece nothing matched: no detection on it, or only one that is too far off (IoU < 0.5)
false    a detection that matched no piece and holds under SHARE of any (junk, or a stray fragment of a piece)
Detections without a mask (the previous detector's boxes) are scored as their filled rectangle.
"""

from dataclasses import dataclass

import numpy as np

SHARE = 0.25
IOU = 0.5


@dataclass
class Score:
    correct: int = 0
    split: int = 0
    merge: int = 0
    missed: int = 0
    false: int = 0
    pieces: int = 0
    detections: int = 0

    def __add__(self, o):
        return Score(*(a + b for a, b in zip(self.__dict__.values(), o.__dict__.values())))


def box_mask(box, shape):
    m = np.zeros(shape[:2], np.uint8)
    x, y, w, h = box
    m[max(0, y):y + h, max(0, x):x + w] = 1
    return m


def score(gt, dets):
    """gt, dets: lists of full-size 0/1 masks."""
    s = Score(pieces=len(gt), detections=len(dets))
    inter = np.array([[int(np.count_nonzero(g & d)) for d in dets] for g in gt]).reshape(len(gt), len(dets))
    g_area = np.array([int(g.sum()) for g in gt], float)
    d_area = np.array([int(d.sum()) for d in dets], float)
    iou = inter / np.maximum(1.0, g_area[:, None] + d_area[None, :] - inter)
    cover_g = inter / np.maximum(1.0, g_area[:, None])               # share of the real piece a detection holds
    taken = set()
    for i in range(len(gt)):
        parts = [j for j in range(len(dets)) if cover_g[i, j] >= SHARE]
        merged = any(sum(1 for k in range(len(gt)) if cover_g[k, j] >= SHARE) >= 2 for j in parts)
        if len(parts) >= 2:
            s.split += 1
        elif merged:
            s.merge += 1
        elif len(gt) and dets and iou[i].max() >= IOU and int(iou[i].argmax()) not in taken:
            s.correct += 1
            taken.add(int(iou[i].argmax()))
        else:
            s.missed += 1                                                # nothing, or a detection too far off the piece (IoU < 0.5)
    for j in range(len(dets)):
        if j not in taken and not (cover_g[:, j] >= SHARE).any():
            s.false += 1                                                  # matched nothing and holds no real share of any piece: junk or a stray fragment
    # a detection covering two real pieces is ONE merge error however many pieces it swallowed
    merged_dets = sum(1 for j in range(len(dets)) if sum(1 for i in range(len(gt)) if cover_g[i, j] >= SHARE) >= 2)
    s.merge = merged_dets
    return s


def detection_masks(analysis, shape):
    """The final detections of backend.core.detect.analyse as masks."""
    out = []
    for p in analysis.pieces:
        out.append(p.region.full(shape) if p.region is not None else box_mask(p.box, shape))
    return out


def legacy_masks(boxes, shape):
    return [box_mask(b, shape) for b in boxes]
