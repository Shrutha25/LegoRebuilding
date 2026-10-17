"""Manual merge / split corrections, kept for the detection test set.

Each correction is one JSON line in `corrections.jsonl`: when, the photo's hash, what it was (`before`: the pieces involved, as
polygons in the photo's own pixels) and what the user made of it (`after`). The photo is saved once as `<hash>.jpg` so a correction
can become a labelled test photo (tests/detection/).
"""

import hashlib
import json
import time

import cv2

from backend import config


def polygon(region, eps=2.0):
    """A region's outline as [[x, y], ...] in photo pixels."""
    p = cv2.copyMakeBorder(region.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps, True)[:, 0, :] - 1 + [region.x, region.y]
    return [[int(x), int(y)] for x, y in c]


def photo_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def record(job_dir, kind, before, after, directory=None):
    """Append a correction. `before` / `after` are lists of regions. Never raises: a failure to store must not fail the user's action."""
    try:
        d = directory or config.CORRECTIONS_DIR
        d.mkdir(parents=True, exist_ok=True)
        photo = job_dir / "image.jpg"
        h = photo_hash(photo)
        if not (d / f"{h}.jpg").exists():
            (d / f"{h}.jpg").write_bytes(photo.read_bytes())
        row = {"time": time.time(), "photo": h, "kind": kind,
               "before": [polygon(r) for r in before], "after": [polygon(r) for r in after]}
        with open(d / "corrections.jsonl", "a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")
        return row
    except Exception:                                              # noqa: BLE001
        return None
