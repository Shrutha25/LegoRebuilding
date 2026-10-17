"""Five example builds from one test inventory, old output next to new, as contact sheets and a table.

    python -m backend.recipes.examples [--seed 5] [--size 140] [--out docs/examples] [--gemini]

"old" is what the original five recipes (house, car, tower, bridge, rocket) gave for the inventory, named the way their templates named them
("Red House"); their shapes are verified identical to the converted JSON recipes by tests/test_schema_regression.py. The car there stands on plain
bricks as wheels, which is no longer shown. "new" is the concept pipeline (backend/recipes/suggest.py) without Gemini (the recipes' own concepts),
and with --gemini also with Gemini's concepts. Variety is measured as the mean pairwise voxel IoU of the five builds (lower = more varied).
"""

import argparse
import asyncio
import io
import itertools
import json
from pathlib import Path

from PIL import Image, ImageDraw

from backend import config
from backend.core.render_iso import render_png
from backend.recipes import get
from backend.recipes.checks import iou, voxels_of_build
from backend.recipes.facts import facts_of
from backend.recipes.inventories import hex_of, random_inventory
from backend.recipes.planner import plan_candidates
from backend.recipes.suggest import suggest

TILE = (420, 340)
OLD = ("house", "car", "tower", "bridge", "rocket")
OLD_TEMPLATE = {"house": "{} House", "car": "{} Car", "tower": "{} Tower", "bridge": "{} Bridge", "rocket": "{} Rocket"}


def sheet(items, path, inv, title):
    """items: [(label, Build)] -> a PNG contact sheet, one row of up to 5 per line of 3."""
    cols = 3
    rows = (len(items) + cols - 1) // cols
    im = Image.new("RGB", (cols * TILE[0], rows * (TILE[1] + 26) + 30), "white")
    dr = ImageDraw.Draw(im)
    dr.text((8, 8), title, fill=(0, 0, 0))
    for n, (label, build) in enumerate(items):
        x, y = (n % cols) * TILE[0], 30 + (n // cols) * (TILE[1] + 26)
        im.paste(Image.open(io.BytesIO(render_png(build, inv, hex_of, TILE))).convert("RGB"), (x, y + 24))
        dr.text((x + 8, y + 6), label, fill=(0, 0, 0))
    im.save(path)


def variety(builds, inv):
    vox = [voxels_of_build(b, inv) for b in builds]
    pairs = [iou(a, b) for a, b in itertools.combinations(vox, 2)]
    return round(sum(pairs) / len(pairs), 3) if pairs else 0.0


async def main_async(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = random_inventory(args.seed, args.size)
    facts = facts_of(inv)
    report = {"inventory": {"seed": args.seed, "size": args.size, "parts": facts.total_parts, "colours": facts.colours}}

    from backend.recipes import planner
    saved = dict(planner.SUPPORTED_NEEDS)
    planner.SUPPORTED_NEEDS["wheels"] = lambda f: True                    # the old system did offer the car (on plain bricks as wheels)
    try:
        old = plan_candidates(inv, 1, 5, [get(r) for r in OLD])[:5]
    finally:
        planner.SUPPORTED_NEEDS.clear()
        planner.SUPPORTED_NEEDS.update(saved)
    old_rows = [(OLD_TEMPLATE[c.recipe_id].format(c.colours.get(c.recipe.roles[0].name, "")), c) for c in old]
    sheet([(f"{n} ({c.part_count} parts)", c.build) for n, c in old_rows], out / "old.png", inv, "OLD: the five original recipes, template names")
    report["old"] = [{"name": n, "recipe": c.recipe_id, "params": c.params, "parts": c.part_count} for n, c in old_rows]
    report["old_mean_iou"] = variety([c.build for _, c in old_rows], inv)

    for label, llm in (("new", None),) + ((("new_gemini", __import__("backend.core.ideas", fromlist=["x"]).GeminiLLM(
            config.GEMINI_API_KEY, config.GEMINI_MODEL, budget=__import__("backend.core.gemini_budget", fromlist=["x"]).get_budget())),) if args.gemini else ()):
        res = await suggest(inv, llm, seed=1, show=5, timeout=60.0, use_ai=llm is not None, match_color=True)
        rows = [(s.name, s.candidate) for s in res.shown]
        sheet([(f"{n} - {c.recipe_id} ({c.part_count} parts)", c.build) for n, c in rows], out / f"{label}.png", inv,
              f"NEW ({'Gemini concepts' if llm else 'the recipes own concepts, no Gemini'})")
        report[label] = {"ai_used": res.ai_used, "note": res.ai_note, "concepts": len(res.concepts),
                         "builds": [{"name": n, "recipe": c.recipe_id, "category": c.recipe.category, "params": c.params, "parts": c.part_count,
                                     "silhouette": s.concept.silhouette if s.concept else None} for (n, c), s in zip(rows, res.shown)],
                         "categories": sorted({c.recipe.category for _, c in rows}), "mean_iou": variety([c.build for _, c in rows], inv)}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--out", default="docs/examples")
    ap.add_argument("--gemini", action="store_true")
    asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
