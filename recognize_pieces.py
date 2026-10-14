"""
Recognize every LEGO piece in one photo of pieces spread on a plain/white background.

Pipeline:
  1. Find each piece with OpenCV (threshold vs background -> contours -> boxes)
  2. Crop each piece (with padding)
  3. Send all crops to Brickognize in parallel
  4. Aggregate into a parts list {(part_id, color): qty}

Usage:
  pip install opencv-python numpy requests
  python recognize_pieces.py photo.jpg
  python recognize_pieces.py photo.jpg --debug      # also saves annotated image + crops

Outputs:
  parts.json        full results per piece (top guesses, confidence, box)
  parts_list.csv    aggregated part_num,color,qty  -> feed this into set matching
  annotated.jpg     (with --debug) boxes + labels drawn on the photo
"""

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2

from backend.core.detect import find_pieces
from backend.core.recognize import recognize_crop, summarize


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--min-score", type=float, default=0.6,
                    help="below this, mark the piece for user review")
    ap.add_argument("--workers", type=int, default=8, help="parallel API calls")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    boxes = find_pieces(img)
    print(f"Found {len(boxes)} pieces")
    crops = [img[y:y + h, x:x + w] for x, y, w, h in boxes]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(recognize_crop, crops))

    pieces, counts = [], Counter()
    for i, (box, res) in enumerate(zip(boxes, raw)):
        s = summarize(res) if "error" not in res else None
        entry = {"index": i, "box": box, "result": s, "error": res.get("error")}
        entry["needs_review"] = s is None or s["score"] < args.min_score
        pieces.append(entry)
        if s:
            counts[(s["part_id"], s["name"], s["color"])] += 1
        label = f'{s["part_id"]} {s["color"] or ""} ({s["score"]})' if s else "??"
        flag = "  <- review" if entry["needs_review"] else ""
        print(f"#{i:02d} {label}{flag}")

    with open("parts.json", "w") as f:
        json.dump(pieces, f, indent=2)
    with open("parts_list.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part_num", "name", "color", "qty"])
        for (pid, name, color), q in counts.most_common():
            w.writerow([pid, name, color, q])
    print(f"\n{sum(counts.values())} recognized, "
          f"{sum(p['needs_review'] for p in pieces)} need review -> parts_list.csv")

    if args.debug:
        os.makedirs("crops", exist_ok=True)
        out = img.copy()
        for p, crop in zip(pieces, crops):
            x, y, w, h = p["box"]
            cv2.imwrite(f"crops/{p['index']:02d}.jpg", crop)
            color = (0, 0, 255) if p["needs_review"] else (0, 160, 0)
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            txt = p["result"]["part_id"] if p["result"] else "??"
            cv2.putText(out, f'{p["index"]}:{txt}', (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite("annotated.jpg", out)
        print("Saved annotated.jpg and crops/")


if __name__ == "__main__":
    main()
