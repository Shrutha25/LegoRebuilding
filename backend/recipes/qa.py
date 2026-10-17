"""Offline QA: does every recipe look like its name?

    python -m backend.recipes.qa                      # all recipes, 20 random inventories each, grids of 4, graded by Gemini vision
    python -m backend.recipes.qa --recipes car house --inventories 12
    python -m backend.recipes.qa --dry-run            # render the grids only (no Gemini)
    python -m backend.recipes.qa --max-calls 2        # never spend more than 2 Gemini calls this run (results are cached; run again later)

For each recipe it generates builds from random realistic inventories (backend/recipes/inventories.py), renders them with
core/render_iso.py into grids of 4 (docs/recipes/qa/<recipe>-gridN.png) and asks a Gemini vision model, per picture,
"does this look like a <name>? yes / partly / no". Several grids go into ONE call (--grids-per-call) to save the daily quota; every call goes
through gemini_budget (purpose "build") and stops cleanly when the budget says no. Verdicts are cached in docs/recipes/qa/cache.json, so a
run that ran out of budget continues where it stopped. The report is docs/recipes/qa-report.md (one section per recipe).
"""

import argparse
import asyncio
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

from backend import config
from backend.core.render_iso import render_png
from backend.recipes import all_recipes, get
from backend.recipes.inventories import hex_of, random_inventory
from backend.recipes.planner import plan_candidates
from backend.recipes.vision import contact_sheet, grade

log = logging.getLogger("lego.recipes.qa")
OUT = config.ROOT / "docs" / "recipes"


def generate(recipe, count, seed):
    """[(label, candidate or None, inventory, png or None)] for `count` random inventories."""
    rows = []
    for i in range(count):
        s = seed * 1000 + i
        rng = random.Random(s)
        inv = random_inventory(rng, rng.randint(40, 120))
        cands = plan_candidates(inv, s, 1, [recipe])
        if not cands:
            rows.append((f"{recipe.id}-{i + 1}", None, inv, None))
            continue
        c = cands[0]
        rows.append((f"{recipe.id}-{i + 1}", c, inv, render_png(c.build, inv, hex_of, (380, 310))))
    return rows


def make_grids(recipe, rows, per_grid=4, out_dir=None):
    """Grids of `per_grid` renders: [(grid name, png, [labels])]. Also written to out_dir when given."""
    built = [r for r in rows if r[3] is not None]
    grids = []
    for g in range(0, len(built), per_grid):
        chunk = built[g:g + per_grid]
        labels = [r[0] for r in chunk]
        png = contact_sheet([r[3] for r in chunk], labels, cols=2)
        name = f"{recipe.id}-grid{g // per_grid + 1}"
        grids.append((name, png, labels))
        if out_dir:
            (out_dir / f"{name}.png").write_bytes(png)
    return grids


async def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    qa_dir = OUT / "qa"
    qa_dir.mkdir(exist_ok=True)
    cache_path = qa_dir / "cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() and not args.fresh else {}
    recipes = [get(r) for r in args.recipes] if args.recipes else all_recipes()
    data = {}
    all_grids = []                                         # (recipe id, grid name, png, labels)
    for r in recipes:
        rows = generate(r, args.inventories, args.seed)
        grids = make_grids(r, rows, args.per_grid, qa_dir)
        data[r.id] = rows
        all_grids += [(r.id, n, png, labels) for n, png, labels in grids]
        print(f"{r.id}: {sum(1 for x in rows if x[1])}/{len(rows)} inventories give a build, {len(grids)} grids")

    # cache keys include a hash of the picture, so a verdict is never reused for a build that has changed since
    key = {r[0]: f"{r[0]}@{hashlib.sha1(r[3]).hexdigest()[:10]}" for rows in data.values() for r in rows if r[3]}
    verdicts = {}                                           # label -> (verdict, note)
    todo = []
    for rid, name, png, labels in all_grids:
        if all(key[l] in cache for l in labels):
            verdicts.update({l: tuple(cache[key[l]]) for l in labels})
        else:
            todo.append((rid, name, png, labels))
    calls, stopped = 0, ""
    if not args.dry_run and todo:
        llm = _llm()
        for k in range(0, len(todo), args.grids_per_call):
            if args.max_calls is not None and calls >= args.max_calls:
                stopped = f"stopped after --max-calls {args.max_calls}"
                break
            batch = todo[k:k + args.grids_per_call]
            names = {}
            for rid, _, _, labels in batch:
                for l in labels:
                    names[l] = _object_name(get(rid))
            try:
                got = await grade(llm, [b[2] for b in batch], names)
                calls += 1
            except Exception as e:                       # noqa: BLE001  budget refused, quota, network: keep what we have
                stopped = f"stopped: {type(e).__name__}: {e}"
                break
            for label, v in got.items():
                cache[key[label]] = list(v)
                verdicts[label] = v
            cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    write_report(OUT / "qa-report.md", recipes, data, verdicts, args, calls, stopped, bool(args.dry_run))
    print(f"report: {OUT / 'qa-report.md'}  ({calls} Gemini calls){' - ' + stopped if stopped else ''}")


def _object_name(recipe):
    return {"vehicle": "toy car", "building": "small house", "tower": "tower", "bridge": "bridge", "spacecraft": "rocket"}.get(
        recipe.category, recipe.nouns[0])


def _llm():
    from backend.core import gemini_budget
    from backend.core.ideas import GeminiLLM
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY is not set: use --dry-run to only render the grids.")
    return GeminiLLM(config.GEMINI_API_KEY, config.GEMINI_MODEL, timeout=90.0, budget=gemini_budget.get_budget())


def write_report(path, recipes, data, verdicts, args, calls, stopped, dry):
    lines = ["# Recipe QA report", "",
             f"Random realistic inventories per recipe: {args.inventories}; grids of {args.per_grid}; model: {config.GEMINI_MODEL}; "
             f"Gemini calls this run: {calls}." + (f" **{stopped}**" if stopped else "") + (" (dry run: nothing graded)" if dry else ""), ""]
    for r in recipes:
        rows = data[r.id]
        built = [x for x in rows if x[1]]
        counts = Counter(verdicts[x[0]][0] for x in built if x[0] in verdicts)
        graded = sum(counts.values())
        lines += [f"## {r.id} (\"{_object_name(r)}\")", "",
                  f"- inventories that give a build: **{len(built)}/{len(rows)}**",
                  f"- graded: {graded}; yes {counts['yes']}, partly {counts['partly']}, no {counts['no']}"
                  + (f" -> **{round(100 * counts['yes'] / graded)}% clearly recognisable, "
                     f"{round(100 * (counts['yes'] + counts['partly']) / graded)}% at least roughly**" if graded else "")]
        bad = [(x, verdicts[x[0]]) for x in built if x[0] in verdicts and verdicts[x[0]][0] == "no"]
        part = [(x, verdicts[x[0]]) for x in built if x[0] in verdicts and verdicts[x[0]][0] == "partly"]
        for tag, group in (("no", bad), ("partly", part)):
            if group:
                lines.append(f"- {tag}:")
                for x, (v, note) in group[:6]:
                    lines.append(f"  - `{x[0]}` {x[1].params} {x[1].part_count} parts, colours {sorted(set(x[1].colours.values()))}: {note}")
        lines.append("")
    lines += ["Pictures: `docs/recipes/qa/<recipe>-gridN.png` (the exact grids sent to the model).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipes", nargs="*")
    ap.add_argument("--inventories", type=int, default=20)
    ap.add_argument("--per-grid", type=int, default=4)
    ap.add_argument("--grids-per-call", type=int, default=5)
    ap.add_argument("--max-calls", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cached verdicts")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
