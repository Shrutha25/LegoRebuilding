"""End-to-end timing on real photos: the same requests the app makes, one photo after another, against the real services.

    python -m backend.bench bench_photos/*.jpg [--out docs/timings/baseline.json] [--no-ai] [--colour-off]

For each photo: POST /api/scan and poll until the job is done (detection + Brickognize), /api/parts/resolve, then the "Suggest" sources the results
screen asks for: /api/match (official sets), /api/sets/nearby, /api/builds/recipes (designs) and /api/ideas (text ideas). Every request is timed by
the client (wall clock) and by the server (timing.py stages, appended to the timing log). The run is sequential, one photo at a time; the app runs
the Suggest requests when the tabs open, so this is the sum, not the wait. Uses the real Brickognize and (without --no-ai) the real Gemini, so it
spends quota: GEMINI_RPM / GEMINI_RPD of the budget apply (set GEMINI_RPD high for a benchmark).

The three instant sources (official sets, nearby sets, real sub-builds) are fired together, as the results screen does; `instant_sources_ms` is the
time until the slowest of them answered.
"""

import argparse
import json
import sys
import time
from pathlib import Path

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


def run_photo(client, path, match_color=True, ai=True, timeout=120):
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
    # the three instant sources, fired together like the results screen does on "Suggest"
    t_start = time.perf_counter()
    with ThreadPoolExecutor(3) as pool:
        f_match = pool.submit(timed, client, "post", "/api/match", json={"parts": body["parts"], "match_color": match_color})
        f_near = pool.submit(timed, client, "post", "/api/sets/nearby", json=body)
        f_sub = pool.submit(timed, client, "post", "/api/subbuilds", json=sub_body)
        (r, out["match_ms"]), (rn, out["nearby_ms"]), (rs, out["subbuilds_ms"]) = f_match.result(), f_near.result(), f_sub.result()
    out["instant_sources_ms"] = round((time.perf_counter() - t_start) * 1000)
    out["match_status"], out["sets"] = r.status_code, len(r.json().get("sets", [])) if r.status_code == 200 else 0
    out["nearby_status"] = rn.status_code
    if rn.status_code == 200:
        d = rn.json()
        out["nearby"] = {"can_build": len(d["can_build"]), "almost": len(d["almost"])}
    out["subbuilds_status"] = rs.status_code
    if rs.status_code == 200:
        d = rs.json()
        c = d.get("counts") or {}
        out["subbuilds"] = {"ready": d.get("ready"), "buildable": c.get("buildable", 0), "almost": c.get("almost", 0),
                            "smallest_buildable": c.get("smallest_buildable"), "smallest_almost": c.get("smallest_almost"),
                            "shown_buildable": [f"{b['title']} ({b['set_num']}, {b['parts']})" for b in d.get("buildable", [])[:3]],
                            "shown_almost": [f"{b['title']} ({b['set_num']}, {b['parts']}, missing {b['missing_count']})" for b in d.get("almost", [])[:3]]}
    if ai:
        rb = {"parts": [{k: v for k, v in p.items() if k in ("part_num", "name", "color", "qty")} for p in parts], "match_color": match_color}
        r, out["recipes_ms"] = timed(client, "post", "/api/builds/recipes", json=rb)
        out["recipes_status"] = r.status_code
        if r.status_code == 200:
            d = r.json()
            out["recipes"] = {"builds": len(d["builds"]), "ai_used": d.get("ai_used"), "note": d.get("ai_note"), "concepts": d.get("concepts"),
                              "generated": d.get("generated"), "titles": [b["title"] for b in d["builds"]]}
        r, out["ideas_ms"] = timed(client, "post", "/api/ideas", json=rb)
        out["ideas_status"] = r.status_code
        out["ideas"] = len(r.json().get("ideas", [])) if r.status_code == 200 else r.json().get("detail")
    out["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--out", default="docs/timings/bench.json")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--colour-off", action="store_true")
    args = ap.parse_args(argv)
    from fastapi.testclient import TestClient
    from backend import main as app_main
    since = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    with TestClient(app_main.app) as client:
        for _ in range(240):                                          # the catalogue loads in the background at start-up
            h = client.get("/api/health").json()
            idx = h.get("indexes") or {}
            if h.get("data_loaded") and h.get("parts_loaded") and idx.get("nearby") and idx.get("omr") and idx.get("service"):
                break
            time.sleep(0.5)
        for p in args.photos:
            print(f"== {p}", flush=True)
            res = run_photo(client, p, match_color=not args.colour_off, ai=not args.no_ai)
            results.append(res)
            print("   " + "  ".join(f"{k}={v}" for k, v in res.items() if k.endswith("_ms") or k in ("pieces", "sets", "scan_status", "subbuilds")), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = load(config.TIMING_LOG, since=since) if config.TIMING_LOG and Path(config.TIMING_LOG).exists() else []
    summary = summarize(rows)
    Path(args.out).write_text(json.dumps({"photos": results, "server": {k: {"n": g["n"], "p50": pct(g["total"], .5), "p95": pct(g["total"], .95),
                                                                           "gemini_calls": g["gemini_calls"],
                                                                           "stages": {s: {"p50": pct(v, .5), "p95": pct(v, .95), "n": len(v)} for s, v in g["stages"].items()}}
                                                                       for k, g in summary.items()}}, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nserver-side timing for this run (p50 / p95 over the photos):\n" + render(summary, stages=True))


if __name__ == "__main__":
    main()
