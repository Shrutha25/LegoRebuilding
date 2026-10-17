"""Print where the time goes: p50 / p95 per stage over the timing log.

    python -m backend.timing_report [--log timings.jsonl] [--name /api/match] [--since "2026-10-17 09:00:00"] [--last N] [--stages]

For each kind of request (the route, or "scan_job" for the photo job) it prints the number of requests, the p50 / p95 / max of the whole request
and the number of Gemini calls it made. With --stages it also prints, per stage, how many requests ran it and the p50 / p95 / max of the TOTAL time
spent in that stage per request (stages of one name are added up inside a request; nested stages count in their parents too).
"""

import argparse
import json
import sys
from pathlib import Path

from backend import config


def pct(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(path, name=None, since=None, last=None):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (name is None or row.get("name") == name) and (since is None or row.get("time", "") >= since):
            rows.append(row)
    return rows[-last:] if last else rows


def summarize(rows):
    """{request name: {"n", "total": [ms], "gemini_calls": [n], "stages": {stage: [ms per request]}}}"""
    out = {}
    for r in rows:
        g = out.setdefault(r["name"], {"n": 0, "total": [], "gemini_calls": [], "stages": {}})
        g["n"] += 1
        g["total"].append(r["total_ms"])
        g["gemini_calls"].append(r.get("counters", {}).get("gemini_calls", 0))
        for stage, ms in r.get("sums", {}).items():
            g["stages"].setdefault(stage, []).append(ms)
    return out


def render(summary, stages=True):
    lines = [f"{'request':34} {'n':>4} {'p50 ms':>9} {'p95 ms':>9} {'max ms':>9}   gemini calls p50/max"]
    for name, g in sorted(summary.items(), key=lambda kv: -pct(kv[1]["total"], 0.5)):
        gc = g["gemini_calls"]
        lines.append(f"{name:34} {g['n']:4} {pct(g['total'], .5):9.0f} {pct(g['total'], .95):9.0f} {max(g['total']):9.0f}   {pct(gc, .5):.0f}/{max(gc)}")
        if stages:
            for stage, ms in sorted(g["stages"].items(), key=lambda kv: -pct(kv[1], 0.5)):
                lines.append(f"    {stage:30} {len(ms):4} {pct(ms, .5):9.0f} {pct(ms, .95):9.0f} {max(ms):9.0f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=config.TIMING_LOG or "timings.jsonl")
    ap.add_argument("--name")
    ap.add_argument("--since")
    ap.add_argument("--last", type=int)
    ap.add_argument("--stages", action="store_true", help="also print every stage")
    args = ap.parse_args(argv)
    if not Path(args.log).exists():
        sys.exit(f"{args.log} does not exist: run some requests first (the server writes it; TIMING_LOG= disables it)")
    rows = load(args.log, args.name, args.since, args.last)
    if not rows:
        sys.exit("no matching requests in the log")
    print(render(summarize(rows), args.stages))


if __name__ == "__main__":
    main()
