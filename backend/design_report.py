"""Compare benchmark runs of "Designed for your pile": designs shown per photo, the look-check pass rate under both questions, latency and an estimated cost.

    python -m backend.design_report flash-lite=docs/timings/designs4b-flash-lite.json after=docs/timings/after-cost.json

Every argument is `label=bench.json` (a file written by `python -m backend.bench ... --out`). Output: a Markdown report on stdout (and `--out FILE`).

Cost is an ESTIMATE: Gemini returns no token counts here, so tokens are characters / CHARS_PER_TOKEN (the prompts and answers the pipeline logs in its run stats)
plus IMAGE_TOKENS per contact sheet, priced from PRICES (USD per million tokens, input / output). The prices are assumptions, not read from Google: check them against
the current price list and edit PRICES before relying on the dollar figures. The token counts and the ratios between models are the useful part.
"""

import argparse
import json
import statistics as st
from pathlib import Path

CHARS_PER_TOKEN = 3.5
IMAGE_TOKENS = 1300                                     # one contact sheet image
PRICES = {                                              # label -> (USD per 1M input tokens, USD per 1M output tokens); ASSUMED, edit me
    "flash-lite": (0.10, 0.40),
}
LOOK_PRICE_LABEL = "flash-lite"                         # the look check uses the app's own model in every run


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))] if v else None


def runs_of(bench):
    return [(p["photo"], r) for p in bench["photos"] for r in p.get("designs_runs", [])]


def summarise(label, bench):
    runs = runs_of(bench)
    rs = [r for _, r in runs]
    first = [r["first_design_ms"] for r in rs if r["first_design_ms"] is not None]
    last = [r["last_design_ms"] for r in rs if r["last_design_ms"] is not None]
    done = [r["done_ms"] for r in rs if r["done_ms"] is not None]
    stats = [r.get("stats") or {} for r in rs]

    def total(key):
        return sum(s.get(key, 0) or 0 for s in stats)
    per_photo = {}
    for photo, r in runs:
        per_photo.setdefault(Path(photo).stem, []).append(r)
    concept_in, concept_out = total("concept_in_chars") / CHARS_PER_TOKEN, total("concept_out_chars") / CHARS_PER_TOKEN
    look_in, look_out = total("look_in_chars") / CHARS_PER_TOKEN + IMAGE_TOKENS * sum(1 for s in stats if s.get("look_checked")), total("look_out_chars") / CHARS_PER_TOKEN
    pin, pout = PRICES.get(label, (0, 0))
    lin, lout = PRICES[LOOK_PRICE_LABEL]
    n = max(1, len(rs))
    cost = ((concept_in * pin + concept_out * pout) + (look_in * lin + look_out * lout)) / 1e6 / n
    return {"label": label, "runs": len(rs), "with_design": len(first),
            "shown_per_photo": {k: round(st.mean(r["designs_kept"] for r in v), 2) for k, v in per_photo.items()},
            "shown_mean": round(st.mean(r["designs_kept"] for r in rs), 2) if rs else 0,
            "produced_mean": round(st.mean(r["designs_produced"] for r in rs), 2) if rs else 0,
            "first": (pct(first, .5), pct(first, .95)), "last": (pct(last, .5), pct(last, .95)), "done": (pct(done, .5), pct(done, .95)),
            "calls": round(st.mean(r["gemini_calls"] or 0 for r in rs), 2) if rs else 0,
            "look_checked": total("look_checked"), "look_ok_child": total("look_ok_child"), "look_ok_choice": total("look_ok_choice"),
            "rescued": total("rescued"), "outline_rejected": total("outline_rejected"), "fill_rejected": total("fill_rejected"), "dup_dropped": total("dup_dropped"),
            "stages": {k: total(k) for k in ("concepts", "sketch_ok", "solver_ok", "checks_ok", "diverse_ok", "look_checked", "look_ok")},
            "tokens_in_per_request": round((concept_in + look_in) / n), "tokens_out_per_request": round((concept_out + look_out) / n), "cost_usd_per_request": round(cost, 5)}


def rate(a, b):
    return f"{a / b:.0%}" if b else "-"


def render(results):
    labels = [r["label"] for r in results]
    photos = sorted({p for r in results for p in r["shown_per_photo"]})
    out = ["# Concept model comparison (look check on the app's own model in every run)", "",
           "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]

    def row(name, fn):
        out.append(f"| {name} | " + " | ".join(str(fn(r)) for r in results) + " |")
    for p in photos:
        row(f"designs shown, {p} (mean of repeats)", lambda r, p=p: r["shown_per_photo"].get(p, "-"))
    row("designs shown per request (mean)", lambda r: r["shown_mean"])
    row("designs produced before the look check (mean)", lambda r: r["produced_mean"])
    row("requests with no design", lambda r: f"{r['runs'] - r['with_design']} of {r['runs']}")
    row("look check pass rate, child question (new)", lambda r: f"{rate(r['look_ok_child'], r['look_checked'])} ({r['look_ok_child']}/{r['look_checked']})")
    row("look check pass rate, pick-the-name question (old)", lambda r: f"{rate(r['look_ok_choice'], r['look_checked'])} ({r['look_ok_choice']}/{r['look_checked']})")
    row("time to first design p50 / p95 (ms)", lambda r: f"{r['first'][0]} / {r['first'][1]}")
    row("time to last design p50 / p95 (ms)", lambda r: f"{r['last'][0]} / {r['last'][1]}")
    row("request done p50 / p95 (ms)", lambda r: f"{r['done'][0]} / {r['done'][1]}")
    row("Gemini calls per request (mean)", lambda r: r["calls"])
    row("est. tokens per request in / out", lambda r: f"{r['tokens_in_per_request']} / {r['tokens_out_per_request']}")
    row("est. cost per request (USD, assumed prices)", lambda r: f"{r['cost_usd_per_request']:.4f}")
    out += ["", "## Stage funnel (all requests)", "", "| stage | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    keys = ["concepts", "sketch_ok", "solver_ok", "checks_ok", "diverse_ok", "look_checked", "look_ok"]
    for a, b in [(None, "concepts")] + list(zip(keys, keys[1:])):
        out.append(f"| {b}" + (f" (of {a})" if a else "") + " | " + " | ".join(
            (f"{r['stages'][b]} ({rate(r['stages'][b], r['stages'][a])})" if a else str(r["stages"][b])) for r in results) + " |")
    out += ["", "## Tolerance and diversity (all requests)", "", "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, key in (("designs rescued by the 85% + outline rule", "rescued"), ("rejected because the outline was lost", "outline_rejected"),
                      ("concepts rejected for the wrong size", "fill_rejected"), ("concepts dropped as duplicates (shape or bounding box)", "dup_dropped")):
        out.append(f"| {name} | " + " | ".join(str(r[key]) for r in results) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=bench.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    results = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        results.append(summarise(label, json.loads(Path(path).read_text(encoding="utf-8"))))
    text = render(results)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
