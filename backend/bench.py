"""End-to-end timing on real photos: the same requests the app makes, one photo after another, against the real services, over real HTTP.

    python -m backend.bench bench_photos/*.jpg [--out docs/timings/bench.json] [--repeat 3] [--designs-dir docs/designs] [--legacy] [--colour-off]

A uvicorn server is started in this process. For each photo: POST /api/scan and poll until the job is done (detection + Brickognize), /api/parts/resolve,
then the instant sources fired together (official sets, nearby sets, real sub-builds), then "Designed for your pile" (POST /api/designs/stream, read as it
streams: time to the first and last design, designs kept and removed, Gemini calls and the per-stage counts the server reports; --repeat runs it
that many times per photo). Every design is fetched and drawn on a contact sheet per photo (removed ones marked) and one for all. --legacy also runs the
old recipes and text-ideas requests. Uses the real Brickognize and Gemini, so it spends quota. Gemini requests are queued, not refused: BUILD_MAX_WAIT
is set to 75 s so a request waits for a free slot in the per-minute window (GEMINI_RPM) instead of failing.
"""

import os

os.environ.setdefault("BUILD_MAX_WAIT", "75")                      # wait for a Gemini slot instead of being refused (read by backend.config)

import argparse
import io
import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from backend import config
from backend.timing_report import load, pct, render, summarize


def parts_from(job):
    """Rows like the review screen: one per (part, colour) with its count."""
    rows = {}
    for p in job["pieces"]:
        if not p.get("part_id"):
            continue
        key = (p["part_id"], p.get("color"))
        rows.setdefault(key, {"part_num": p["part_id"], "name": p.get("name"), "color": p.get("color"), "qty": 0})["qty"] += 1
    return list(rows.values())


def timed(client, method, url, **kw):
    t = time.perf_counter()
    r = getattr(client, method)(url, **kw)
    return r, round((time.perf_counter() - t) * 1000)


def start_server():
    """Run the app with uvicorn in a thread on a free port; returns (base url, server)."""
    import uvicorn
    from backend import main as app_main
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app_main.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    return f"http://127.0.0.1:{port}", server


def read_designs(client, parts, match_color):
    """POST the pile to the stream and read it event by event. -> dict with the timings seen by the client and every design."""
    out = {"events": [], "designs": [], "removed": [], "verdicts": {}, "notes": [], "done": None, "error": None}
    t0 = time.perf_counter()
    try:
        with client.stream("POST", "/api/designs/stream", json={"parts": parts, "match_color": match_color}, timeout=httpx.Timeout(180, connect=10)) as r:
            if r.status_code != 200:
                out["error"] = f"HTTP {r.status_code}"
                return out
            event, data = None, []
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif line == "" and event:
                    payload = json.loads("".join(data)) if data else {}
                    ms = round((time.perf_counter() - t0) * 1000)
                    out["events"].append((event, ms))
                    if event == "design":
                        out["designs"].append({**payload["build"], "client_ms": ms})
                    elif event == "verdict":
                        out["verdicts"][payload["id"]] = {k: payload.get(k) for k in ("ok", "choice", "child_yes", "ok_choice", "ok_child")}
                        if not payload["ok"]:
                            out["removed"].append(payload["id"])
                    elif event == "note":
                        out["notes"].append(payload)
                    elif event == "done":
                        out["done"] = {**payload, "client_ms": ms}
                    event, data = None, []
    except Exception as e:                                          # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def design_summary(res):
    kept = [d for d in res["designs"] if d["id"] not in res["removed"]]
    times = [d["client_ms"] for n, d in enumerate(res["designs"])]
    done = res["done"] or {}
    return {"designs_produced": len(res["designs"]), "designs_kept": len(kept), "removed": len(res["removed"]),
            "first_design_ms": min(times) if times else None, "last_design_ms": max(times) if times else None, "done_ms": done.get("client_ms", res.get("total_ms")),
            "gemini_calls": done.get("calls"), "stats": done.get("stats"), "failures": done.get("failures", [])[:6], "notes": [n.get("message") for n in res["notes"]],
            "error": res["error"], "titles": [d["title"] for d in kept], "removed_titles": [d["title"] for d in res["designs"] if d["id"] in res["removed"]]}


def contact_sheet(base, client, items, path, cols=4, cell=(300, 245)):
    """items: [(label, design meta, removed?)] -> a PNG of their thumbnails with the title and part count under each."""
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * (cell[1] + 34)), "white")
    dr = ImageDraw.Draw(sheet)
    for n, (label, d, removed) in enumerate(items):
        x, y = (n % cols) * cell[0], (n // cols) * (cell[1] + 34)
        try:
            png = client.get(d["thumb_url"]).content
            sheet.paste(Image.open(io.BytesIO(png)).convert("RGB").resize(cell), (x, y))
        except Exception:                                           # noqa: BLE001
            dr.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(200, 0, 0))
        dr.text((x + 6, y + cell[1] + 3), f"{label}: {d['title']}", fill=(0, 0, 0))
        dr.text((x + 6, y + cell[1] + 17), f"{d['part_count']} parts, {d['step_count']} steps" + ("  [REMOVED by the look check]" if removed else ""), fill=(180, 0, 0) if removed else (90, 90, 90))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def run_photo(client, base, path, match_color=True, repeat=1, legacy=False, designs_dir=None, timeout=120):
    out = {"photo": str(path)}
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        r, ms = timed(client, "post", "/api/scan", files={"file": (Path(path).name, f.read(), "image/jpeg")})
    job_id = r.json()["job_id"]
    while True:
        job = client.get(f"/api/scan/{job_id}").json()
        if job["status"] in ("done", "error") or time.perf_counter() - t0 > timeout:
            break
        time.sleep(0.15)
    out["scan_ms"] = round((time.perf_counter() - t0) * 1000)
    out["scan_status"], out["pieces"], out["detector"] = job["status"], len(job.get("pieces") or []), job.get("detector")
    if job["status"] != "done":
        out["error"] = job.get("error")
        return out
    parts = parts_from(job)
    out["rows"], out["pieces_identified"] = len(parts), sum(p["qty"] for p in parts)
    r, out["resolve_ms"] = timed(client, "post", "/api/parts/resolve", json={"parts": [p["part_num"] for p in parts], "colors": {p["part_num"]: p["color"] for p in parts}})
    resolved = r.json().get("results", {})
    usable = [p for p in parts if not resolved.get(p["part_num"], {}).get("unmapped")]
    body = {"parts": usable or parts, "match_color": match_color}
    sub_body = {"parts": [{k: v for k, v in p.items() if k in ("part_num", "name", "color", "qty")} for p in parts], "match_color": match_color}
    from concurrent.futures import ThreadPoolExecutor
    t_start = time.perf_counter()
    with ThreadPoolExecutor(3) as pool:                              # the three instant sources, fired together like the results screen does
        f_match = pool.submit(timed, client, "post", "/api/match", json={"parts": body["parts"], "match_color": match_color})
        f_near = pool.submit(timed, client, "post", "/api/sets/nearby", json=body)
        f_sub = pool.submit(timed, client, "post", "/api/subbuilds", json=sub_body)
        (r, out["match_ms"]), (rn, out["nearby_ms"]), (rs, out["subbuilds_ms"]) = f_match.result(), f_near.result(), f_sub.result()
    out["instant_sources_ms"] = round((time.perf_counter() - t_start) * 1000)
    out["sets"] = len(r.json().get("sets", [])) if r.status_code == 200 else 0
    if rs.status_code == 200:
        c = rs.json().get("counts") or {}
        out["subbuilds"] = {"buildable": c.get("buildable", 0), "almost": c.get("almost", 0)}
    runs = []
    for i in range(repeat):
        res = read_designs(client, sub_body["parts"], match_color)
        s = design_summary(res)
        runs.append(s)
        if designs_dir and i == 0:
            items = [(f"{n + 1}", d, d["id"] in res["removed"]) for n, d in enumerate(res["designs"])]
            contact_sheet(base, client, items, Path(designs_dir) / f"{Path(path).stem}.png")
            out["designs_all"] = [{"title": d["title"], "id": d["id"], "parts": d["part_count"], "removed": d["id"] in res["removed"],
                                   "label": f"{Path(path).stem}#{n + 1}", "type": (d.get("design") or {}).get("type"), "concept_sketch": (d.get("design") or {}).get("sketch"),
                                   "verdict": res["verdicts"].get(d["id"]), "outline_iou": (d.get("design") or {}).get("outline_iou"),
                                   "rescued": (d.get("design") or {}).get("rescued"), "use_share": (d.get("design") or {}).get("use_share"), "coverage": (d.get("design") or {}).get("coverage"),
                                   "thumb_url": d["thumb_url"]} for n, d in enumerate(res["designs"])]
    out["designs_runs"] = runs
    if legacy:
        rb = {"parts": sub_body["parts"], "match_color": match_color}
        r, out["recipes_ms"] = timed(client, "post", "/api/builds/recipes", json=rb)
        body_r = r.json() if r.status_code == 200 else {}
        out["recipes"] = {"status": r.status_code, "builds": len(body_r.get("builds", [])), "titles": [b.get("title") for b in body_r.get("builds", [])][:6],
                          "ai_used": body_r.get("ai_used"), "ai_note": body_r.get("ai_note"), "generated": body_r.get("generated")}
        r, out["ideas_ms"] = timed(client, "post", "/api/ideas", json=rb)
        out["ideas"] = {"status": r.status_code, "detail": (r.json().get("detail") if r.status_code != 200 else None) if r.headers.get("content-type", "").startswith("application/json") else None}
    out["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="docs/timings/bench.json")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--designs-dir", default="")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--colour-off", action="store_true")
    args = ap.parse_args(argv)
    base, server = start_server()
    since = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    with httpx.Client(base_url=base, timeout=60) as client:
        for _ in range(240):                                          # the catalogue and the indexes load in the background at start-up
            h = client.get("/api/health").json()
            idx = h.get("indexes") or {}
            if h.get("data_loaded") and h.get("parts_loaded") and idx.get("nearby") and idx.get("omr") and idx.get("service") and idx.get("examples"):
                break
            time.sleep(0.5)
        print("server ready:", h.get("indexes"), flush=True)
        for p in args.photos:
            print(f"== {p}", flush=True)
            res = run_photo(client, base, p, match_color=not args.colour_off, repeat=args.repeat, legacy=args.legacy, designs_dir=args.designs_dir or None)
            results.append(res)
            runs = res.get("designs_runs", [])
            print("   " + "  ".join(f"{k}={v}" for k, v in res.items() if k.endswith("_ms") or k in ("pieces", "sets", "scan_status", "subbuilds")), flush=True)
            if "recipes" in res:
                print(f"   recipes: {res['recipes']}  ideas: {res['ideas']}", flush=True)
            for r in runs:
                print(f"   designs: kept {r['designs_kept']} of {r['designs_produced']}, first {r['first_design_ms']} ms, last {r['last_design_ms']} ms, done {r['done_ms']} ms, "
                      f"gemini calls {r['gemini_calls']} {r['error'] or ''} {r['notes'] or ''}", flush=True)
        if args.designs_dir:
            all_items = [(f"{Path(r['photo']).stem}#{n + 1}", {"title": d["title"], "thumb_url": d["thumb_url"], "part_count": d["parts"], "step_count": 0}, d["removed"])
                         for r in results for n, d in enumerate(r.get("designs_all", []))]
            contact_sheet(base, client, all_items, Path(args.designs_dir) / "all-designs.png", cols=6)
    server.should_exit = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = load(config.TIMING_LOG, since=since) if config.TIMING_LOG and Path(config.TIMING_LOG).exists() else []
    summary = summarize(rows)
    Path(args.out).write_text(json.dumps({"photos": results, "server": {k: {"n": g["n"], "p50": pct(g["total"], .5), "p95": pct(g["total"], .95),
                                                                           "gemini_calls": g["gemini_calls"],
                                                                           "stages": {s: {"p50": pct(v, .5), "p95": pct(v, .95), "n": len(v)} for s, v in g["stages"].items()}}
                                                                       for k, g in summary.items()}}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nserver-side timing for this run (p50 / p95):\n" + render(summary, stages=True))


if __name__ == "__main__":
    main()
