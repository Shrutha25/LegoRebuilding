"""Live check of the AI ideas feature against the real Gemini API. Spends a few cents at most.

    python -m backend.live_check --models          # list Flash models your key can use
    python -m backend.live_check                   # call the running backend's /api/ideas twice
    python -m backend.live_check --runs 3 --url http://localhost:8000

The second form needs the backend running. It sends the sample parts list (parts_list.csv, made by
`python recognize_pieces.py lego_pieces.jpg`), reports the wall-clock time per call, validates the
response shape, and prints every idea. The API key is never printed.
"""

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request

from backend import config


def list_models():
    if not config.GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set. Put it in .env in the project root (see .env.example).")
    from google import genai
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    try:
        models = [m for m in client.models.list()]
    except Exception as e:
        sys.exit(f"Could not list models: {type(e).__name__}: {e}")
    print(f"Configured model: {config.GEMINI_MODEL}")
    flash = sorted(m.name.removeprefix("models/") for m in models
                   if "flash" in m.name.lower() and "generateContent" in (m.supported_actions or []))
    print("Flash models that support generateContent:")
    for n in flash:
        print("  ", n, "  <- configured" if n == config.GEMINI_MODEL else "")
    if config.GEMINI_MODEL not in flash:
        print(f"\nWARNING: {config.GEMINI_MODEL!r} is not in that list; set GEMINI_MODEL in .env.")


def sample_parts(path="parts_list.csv"):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        sys.exit(f"{path} not found. Run: python recognize_pieces.py lego_pieces.jpg")
    return [{"part_num": r["part_num"], "name": r["name"], "color": r["color"] or None, "qty": int(r["qty"])}
            for r in rows]


def call_ideas(url, parts):
    req = urllib.request.Request(f"{url}/api/ideas", json.dumps({"parts": parts}).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=150) as r:
            body, status = json.load(r), r.status
    except urllib.error.HTTPError as e:
        body, status = json.loads(e.read() or b"{}"), e.code
    return status, body, time.time() - t0


def check_shape(body):
    problems = []
    if not isinstance(body.get("ideas"), list) or not body["ideas"]:
        return ["no ideas list"]
    for i in body["ideas"]:
        for k in ("name", "description", "pieces_used", "steps"):
            if k not in i:
                problems.append(f"idea missing {k!r}")
        if not 1 <= len(i.get("steps", [])) <= 12:
            problems.append(f"{i.get('name')}: odd number of steps")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="list usable Flash models and exit")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()
    if args.models:
        return list_models()

    parts = sample_parts()
    owned = {p["part_num"].lower(): p["qty"] for p in parts}
    print(f"Sample list: {len(parts)} parts, {sum(p['qty'] for p in parts)} pieces; model {config.GEMINI_MODEL}\n")
    failed = False
    for n in range(1, args.runs + 1):
        status, body, secs = call_ideas(args.url, parts)
        print(f"=== run {n}: HTTP {status} in {secs:.1f}s ===")
        if status != 200:
            print("ERROR:", body.get("detail", body))
            failed = True
            continue
        problems = check_shape(body)
        over = [p["part_num"] for i in body["ideas"] for p in i["pieces_used"]
                if p["part_num"].lower() not in owned or p["qty"] > owned[p["part_num"].lower()]]
        print(f"{len(body['ideas'])} ideas, {len(body.get('dropped', []))} dropped by validation",
              f"| shape problems: {problems or 'none'} | pieces not owned/over quantity: {over or 'none'}")
        for d in body.get("dropped", []):
            print(f"   dropped: {d['name']!r}: {d['reason']}")
        print(json.dumps(body["ideas"], indent=2, ensure_ascii=False))
        print()
        failed |= bool(problems or over)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
