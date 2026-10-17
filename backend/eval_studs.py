"""Before/after report for stud detection on real photos (OpenCV only, no API calls).

    python -m backend.eval_studs                       # photos in tests/real_photos
    python -m backend.eval_studs --dir some/folder --out some/out

Each photo's TRUE piece count goes at the end of the file name (white_on_white_18.jpg). For every photo it prints the
pieces found without studs and with studs, the stud counts and the time the stud stage took, and saves the debug overlay
(studs, stud groups, boxes labelled by source) as <out>/<name>__studs.jpg.
"""

import argparse
import dataclasses
import re
import time
from collections import Counter
from pathlib import Path

import cv2

from backend.core import detect, studs

ROOT = Path(__file__).resolve().parent.parent
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "tests" / "real_photos"))
    ap.add_argument("--out", default=str(ROOT / "out" / "studs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in Path(args.dir).glob("*") if p.suffix.lower() in EXT)
    if not photos:
        print(f"no photos in {args.dir}")
        return
    off = dataclasses.replace(studs.CFG, enabled=False)
    print(f"{'photo':34s} {'true':>4s} {'before':>6s} {'after':>5s}  {'studs':>5s} {'groups':>6s}  {'stud ms':>7s}  sources")
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = re.search(r"_(\d+)$", p.stem)
        true = m.group(1) if m else "?"
        before = detect.analyse(img, cfg=off)
        t = time.perf_counter()
        after = detect.analyse(img, debug=out / f"{p.stem}__studs.jpg")
        total = time.perf_counter() - t
        ms = 1000 * (after.timings.get("studs", 0))
        print(f"{p.stem[:34]:34s} {true:>4s} {len(before.pieces):6d} {len(after.pieces):5d}  {len(after.studs):5d} "
              f"{len(after.groups):6d}  {ms:7.0f}  {dict(Counter(x.source for x in after.pieces))}  (total {total:.2f}s)")


if __name__ == "__main__":
    main()
