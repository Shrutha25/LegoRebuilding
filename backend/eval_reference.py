"""Try the reference-background method: an empty photo of the surface and a photo with pieces, taken from the same place.

    python -m backend.eval_reference empty.jpg bricks.jpg [--out out/reference]

Prints the number of pieces and how the empty photo was aligned, and saves <out>/<bricks name>__reference.jpg (boxes) and
<bricks name>__mask.png (what differs from the empty photo: if the whole surface lights up, the photos do not match).
OpenCV only: no colour seeds, no studs, no Gemini.
"""

import argparse
import time
from pathlib import Path

import cv2

from backend.core import detect, reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("empty")
    ap.add_argument("bricks")
    ap.add_argument("--out", default="out/reference")
    args = ap.parse_args()
    ref, img = cv2.imread(args.empty), cv2.imread(args.bricks)
    if ref is None or img is None:
        raise SystemExit("could not read one of the photos")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.bricks).stem
    info = {}
    start = time.perf_counter()
    boxes = reference.find_pieces_by_reference(img, ref, info=info)
    took = time.perf_counter() - start
    result = detect.Analysis([detect.Piece(b, "reference") for b in boxes])
    detect.draw_debug(img, result, out / f"{name}__reference.jpg")
    cv2.imwrite(str(out / f"{name}__mask.png"), info["mask"])
    print(f"{len(boxes)} pieces  alignment={info['alignment']}  changed pixels={100 * info['changed']:.1f}%  {took:.2f}s")
    for i, b in enumerate(boxes, 1):
        print(f"  {i:2d}  x={b[0]} y={b[1]} w={b[2]} h={b[3]}")
    print(f"saved {out / (name + '__reference.jpg')} and {out / (name + '__mask.png')}")


if __name__ == "__main__":
    main()
