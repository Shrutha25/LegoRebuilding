"""Before/after table and debug overlays for the detection test set.

    python -m tests.detection.report            writes docs/detection/<scene>.jpg and docs/detection/metrics.md

"Before" is the detector as it was at commit f45a18f (tests/detection/legacy_detect.py, boxes only); "after" is the mask pipeline.
"""

import sys
from pathlib import Path

import cv2

from backend.core import detect, segment, studs
from tests.detection import legacy_detect, metrics, scenes

OUT = Path(__file__).resolve().parents[2] / "docs" / "detection"
COLUMNS = ("correct", "split", "merge", "missed", "false")


def run(name, img, gt, tilt=1.0):
    saved = studs.CFG.tilt_stretch
    studs.CFG.tilt_stretch = tilt                       # the scenes are top-down
    try:
        before = metrics.score(gt, metrics.legacy_masks(legacy_detect.find_pieces(img), img.shape))
        analysis = detect.analyse_masks(img)
    finally:
        studs.CFG.tilt_stretch = saved
    after = metrics.score(gt, metrics.detection_masks(analysis, img.shape))
    return before, after, analysis


def accepted(rows):
    """Section 8's rule: split errors drop and merge errors do not rise by more than 1 across the whole set."""
    b = sum((r[1] for r in rows), metrics.Score())
    a = sum((r[2] for r in rows), metrics.Score())
    return a.split < b.split and a.merge <= b.merge + 1, b, a


def main(write=True):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (img, gt) in scenes.all_scenes().items():
        before, after, analysis = run(name, img, gt)
        rows.append((name, before, after))
        if write:
            overlay = img.copy()
            for g in gt:                                # ground truth: thin white outline
                cnts, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)
            detect.draw_debug(overlay, analysis, OUT / f"{name}.jpg")
    ok, b, a = accepted(rows)
    lines = ["# Detection before / after", "",
             "Per scene: correct / split / merge / missed / false (see `tests/detection/metrics.py`). "
             "Before = detector at commit f45a18f (boxes), after = mask pipeline.", "",
             "| scene | pieces | before | after |", "|---|---|---|---|"]
    for name, before, after in rows:
        fmt = lambda s: " / ".join(str(getattr(s, c)) for c in COLUMNS)
        lines.append(f"| {name} | {before.pieces} | {fmt(before)} | {fmt(after)} |")
    fmt = lambda s: " / ".join(str(getattr(s, c)) for c in COLUMNS)
    lines += [f"| **total** | {b.pieces} | {fmt(b)} | {fmt(a)} |", "",
              f"Acceptance (split errors drop; merge errors rise by at most 1): **{'PASS' if ok else 'FAIL'}**", "",
              "## Thresholds", "", "| name | default |", "|---|---|"]
    for f_ in segment.SegConfig.__dataclass_fields__.values():
        lines.append(f"| `SEG_{f_.name.upper()}` | {getattr(segment.SEG, f_.name)} |")
    text = "\n".join(lines) + "\n"
    if write:
        (OUT / "metrics.md").write_text(text, encoding="utf8")
    print(text)
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
