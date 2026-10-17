"""Evaluate piece detection (opencv / gemini / hybrid) on real photos.

    python -m backend.eval_detect                          # photos in tests/real_photos, annotated images in out/
    python -m backend.eval_detect --models gemini-3.8-flash,gemini-3.1-flash-lite   # compare Gemini models
    python -m backend.eval_detect --dir some/folder --out some/out

Each photo needs its TRUE piece count at the end of the file name, e.g. white_on_white_18.jpg.
Optionally `white_on_white_18.points.json` holds the true piece centres, [[x, y], ...] in pixels; then
missed / extra / merged are exact:
    missed = true pieces inside no box        merged = boxes containing 2+ true pieces
    extra  = boxes containing no true piece   found  = number of boxes
Without that file they are estimates from the count and box sizes (marked with ~):
    missed~ = max(0, true - found)    extra~ = max(0, found - true)
    merged~ = boxes over 2.5x the median box area    (and "tiny" = boxes under 0.25x the median)

Gemini results are cached in <out>/cache so re-running is free; delete it to call the API again.
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from backend import config
from backend.core import gemini_detect as gd
from backend.core.detect import find_pieces

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
COUNT_RE = re.compile(r"_(\d+)$")


# ---------------------------------------------------------------- scoring
def score(boxes, true_count, points=None):
    """Compare boxes with the truth. Exact when `points` (true piece centres) are given, else estimated."""
    n = len(boxes)
    if points is not None:
        inside = [[i for i, (x, y, w, h) in enumerate(boxes) if x <= px <= x + w and y <= py <= y + h] for px, py in points]
        missed = sum(1 for v in inside if not v)
        merged = sum(1 for i in range(n) if sum(1 for v in inside if i in v) > 1)
        extra = sum(1 for i in range(n) if not any(i in v for v in inside))
        return {"found": n, "missed": missed, "extra": extra, "merged": merged, "exact": True}
    areas = [w * h for _, _, w, h in boxes] or [0]
    med = statistics.median(areas)
    return {"found": n, "missed": max(0, true_count - n), "extra": max(0, n - true_count),
            "merged": sum(1 for a in areas if med and a > 2.5 * med), "exact": False,
            "tiny": sum(1 for a in areas if med and a < 0.25 * med)}


def fmt(r):
    if r is None:
        return "-"
    if "error" in r:
        return f"ERR {r['error']}"
    m = "" if r["exact"] else "~"
    return f"{r['found']:>3} (-{r['missed']}{m} +{r['extra']}{m} m{r['merged']}{m})"


# ---------------------------------------------------------------- drawing
def annotate(img, boxes, points, path):
    out = img.copy()
    pts = points or []
    for i, (x, y, w, h) in enumerate(boxes, 1):
        held = sum(1 for px, py in pts if x <= px <= x + w and y <= py <= y + h) if pts else 1
        colour = (0, 160, 0) if held == 1 else (0, 0, 255) if held > 1 else (0, 140, 255)   # green ok, red merged, orange extra
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, max(2, img.shape[1] // 400))
        cv2.putText(out, str(i), (x + 4, y + 8 + img.shape[1] // 60), cv2.FONT_HERSHEY_SIMPLEX, img.shape[1] / 1000, colour, 2)
    for px, py in pts:
        cv2.circle(out, (int(px), int(py)), max(4, img.shape[1] // 250), (255, 0, 255), -1)
    cv2.imwrite(str(path), out)


# ---------------------------------------------------------------- Gemini with caching and tiling
async def gemini_boxes(det, img, cache_path):
    """-> {"boxes": [...], "seconds": float, "model": str} or {"error": str}. Cached on disk."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    t0 = time.time()
    try:
        boxes, model = await det.detect(img)
        res = {"boxes": boxes, "seconds": round(time.time() - t0, 1), "model": model}
    except Exception as e:
        msg = str(e)
        code = re.search(r"\b(40\d|429|50\d)\b", msg)
        res = {"error": (code.group(1) if code else msg[:40]), "seconds": round(time.time() - t0, 1)}
        return res                                                   # failures are not cached
    cache_path.write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------- main
def load_photos(folder):
    photos = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        m = COUNT_RE.search(p.stem)
        if not m:
            print(f"skipping {p.name}: no _<count> at the end of the name", file=sys.stderr)
            continue
        pts = p.with_suffix(".points.json")
        photos.append({"path": p, "true": int(m.group(1)),
                       "points": json.loads(pts.read_text()) if pts.exists() else None})
    return photos


async def run(args):
    folder = Path(args.dir)
    if not folder.exists():
        sys.exit(f"{folder} does not exist. Put photos there named like white_on_white_18.jpg "
                 f"(the true piece count at the end), or pass --dir.")
    photos = load_photos(folder)
    if not photos:
        sys.exit(f"no photos with a _<count> suffix in {folder}")
    out = Path(args.out) if args.out else folder / "out"
    (out / "cache").mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models.split(",") if m] or [config.GEMINI_DETECT_MODEL]
    dets = {m: gd.GeminiDetector(config.GEMINI_API_KEY, m, None, timeout=args.timeout) for m in models} if config.GEMINI_API_KEY else {}
    if not dets:
        print("GEMINI_API_KEY is not set: only the opencv column is evaluated.", file=sys.stderr)

    rows, model_rows = [], []
    sem = asyncio.Semaphore(args.concurrency)

    async def one(ph, model):
        img = cv2.imread(str(ph["path"]))
        async with sem:
            g = await gemini_boxes(dets[model], img, out / "cache" / f"{ph['path'].stem}.{model}.json")
        return model, g

    for ph in photos:
        img = cv2.imread(str(ph["path"]))
        ph["img"] = img
        ph["opencv"] = find_pieces(img)
    jobs = [one(ph, m) for ph in photos for m in dets]
    done = await asyncio.gather(*jobs) if jobs else []
    by = {}
    for (ph, m), (_, g) in zip([(ph, m) for ph in photos for m in dets], done):
        by[(ph["path"].name, m)] = g

    for ph in photos:
        img, true, pts, cvb = ph["img"], ph["true"], ph["points"], ph["opencv"]
        shape = img.shape
        row = {"photo": ph["path"].name, "true": true, "opencv": score([b for b in cvb if not gd.too_big(b, shape)], true, pts)}
        annotate(img, cvb, pts, out / f"{ph['path'].stem}__opencv.jpg")
        for mi, m in enumerate(dets):
            g = by[(ph["path"].name, m)]
            tag = m if len(dets) > 1 else "gemini"
            if "error" in g:
                row[tag], row[f"hybrid[{m}]" if len(dets) > 1 else "hybrid"] = {"error": g["error"]}, None
                model_rows.append((ph["path"].name, m, None, g))
                continue
            gb = [gd.pad_box(b, gd.PAD_FRAC, shape) for b in g["boxes"]]
            gb = [b for b in gb if not gd.too_big(b, shape)]
            row[tag] = score(gb, true, pts)
            hyb = gd.merge_boxes(g["boxes"], cvb, shape)
            hyb_lit = gd.merge_boxes(g["boxes"], cvb, shape, contained=None)
            hk = f"hybrid[{m}]" if len(dets) > 1 else "hybrid"
            row[hk] = score(hyb, true, pts)
            row[hk + "-iou-only"] = score(hyb_lit, true, pts)
            row[tag + "_s"] = g["seconds"]
            model_rows.append((ph["path"].name, m, row[tag], g))
            if mi == 0:
                annotate(img, gb, pts, out / f"{ph['path'].stem}__gemini.jpg")
                annotate(img, hyb, pts, out / f"{ph['path'].stem}__hybrid.jpg")
        rows.append(row)

    # ----- tables
    primary = models[0]
    cols = ["opencv", "gemini" if len(models) == 1 else models[0], "hybrid" if len(models) == 1 else f"hybrid[{primary}]"]
    cols_lit = [c + "-iou-only" for c in cols[2:]]
    print("\nlegend: found (-missed +extra mMerged); ~ = estimated from counts (no .points.json)\n")
    head = f"{'photo':28s} {'true':>4s}  " + "  ".join(f"{c:>22s}" for c in cols + cols_lit)
    print(head + f"  {'gemini s':>8s}")
    for r in rows:
        secs = r.get(cols[1] + "_s", "")
        print(f"{r['photo'][:28]:28s} {r['true']:>4d}  " + "  ".join(f"{fmt(r.get(c)):>22s}" for c in cols + cols_lit) + f"  {secs!s:>8s}")
    tot = {c: [sum(r[c][k] for r in rows if r.get(c) and "error" not in r[c]) for k in ("found", "missed", "extra", "merged")] for c in cols + cols_lit}
    print(f"{'TOTAL':28s} {sum(r['true'] for r in rows):>4d}  " + "  ".join(
        f"{('%d (-%d +%d m%d)' % tuple(tot[c])):>22s}" for c in cols + cols_lit))
    if len(models) > 1:
        print("\nGemini detection per model (found (-missed +extra mMerged), seconds):")
        print(f"{'photo':28s} " + "  ".join(f"{m[:24]:>26s}" for m in models))
        for ph in photos:
            cells = []
            for m in models:
                g = by[(ph["path"].name, m)]
                sc = None
                if "error" not in g:
                    shape = ph["img"].shape
                    gb = [b for b in (gd.pad_box(b, gd.PAD_FRAC, shape) for b in g["boxes"]) if not gd.too_big(b, shape)]
                    sc = score(gb, ph["true"], ph["points"])
                cells.append(f"{fmt(sc) if sc else 'ERR ' + g['error']} {g['seconds']}s")
            print(f"{ph['path'].name[:28]:28s} " + "  ".join(f"{c:>26s}" for c in cells))
    print(f"\nannotated images: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "tests" / "real_photos"))
    ap.add_argument("--out", default=None, help="default: <dir>/out")
    ap.add_argument("--models", default="", help="comma-separated Gemini models (default: GEMINI_DETECT_MODEL)")
    ap.add_argument("--timeout", type=float, default=gd.GEMINI_TIMEOUT)
    ap.add_argument("--concurrency", type=int, default=2, help="parallel Gemini calls (high values hit 429 rate limits)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
