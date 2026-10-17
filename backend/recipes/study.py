"""A measured run of the on-demand recipe generation (live Gemini): concepts in, pass rate per stage out.

    python -m backend.recipes.study [--concepts 30] [--seed 5] [--size 140] [--library DIR] [--out report.json] [--no-recognise]

It builds a realistic test inventory (recipes.inventories), asks Gemini for concepts (as many calls of ~15 as needed, categories the inventory
supports), then runs generate_recipe for EVERY concept (even where a built-in recipe exists: this measures the generator), and prints and
saves the pass rate of each stage (schema, fill, inventory, structure, recognizability), the recognisability answers and the failures.
Uses the real Gemini budget (backend/core/gemini_budget.py: GEMINI_RPM / GEMINI_RPD apply). With --library DIR the recipes that pass are stored there.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from backend import config
from backend.core import gemini_budget
from backend.core.ideas import GeminiLLM
from backend.recipes import all_recipes
from backend.recipes.concepts import gemini_concepts, feasible_categories, fallback_concepts
from backend.recipes.facts import facts_of
from backend.recipes.generate import STAGES, call_with_retries, generate_recipe, tally
from backend.recipes.inventories import hex_of, random_inventory
from backend.recipes.library import RecipeLibrary

log = logging.getLogger("lego.recipes.study")


async def collect_concepts(llm, inv, facts, feasible, want, match_color=True):
    out, seen = [], set()
    for round_ in range(4):
        if len(out) >= want:
            break
        try:
            got, dropped = await call_with_retries(lambda: gemini_concepts(llm, inv, facts, feasible, match_color, 15, 60.0, all_recipes()),
                                                   time.perf_counter() + 240)
        except Exception as e:                                                       # noqa: BLE001
            log.warning("concept call failed: %s", e)
            break
        for c in got:
            if c.name.lower() not in seen:
                seen.add(c.name.lower())
                out.append(c)
    if len(out) < want:                                                              # top up with the recipes' own concepts, said openly
        for c in fallback_concepts(feasible, want * 2, 1):
            if len(out) < want and c.name.lower() not in seen:
                seen.add(c.name.lower())
                out.append(c)
    return out[:want]


async def main_async(args):
    config.RECIPE_LIBRARY_DIR = Path(args.library)
    inv = random_inventory(args.seed, args.size)
    facts = facts_of(inv)
    feasible = feasible_categories(facts, all_recipes())
    llm = GeminiLLM(config.GEMINI_API_KEY, config.GEMINI_MODEL, budget=gemini_budget.get_budget())
    library = RecipeLibrary(args.library)
    concepts = await collect_concepts(llm, inv, facts, feasible, args.concepts)
    print(f"{len(concepts)} concepts ({sum(c.source == 'gemini' for c in concepts)} from Gemini) in {sorted({c.category for c in concepts})}")
    results = []
    for i, c in enumerate(concepts, 1):
        t = time.time()
        res = await generate_recipe(llm, c, inv, facts, library, hex_of, recognise=not args.no_recognise, max_repairs=3, timeout=args.timeout)
        results.append(res)
        last = res.attempts[-1]
        print(f"[{i:2}/{len(concepts)}] {c.name!r:34} {c.category:10} {'OK  ' if res.ok else 'FAIL'} attempts={len(res.attempts)} "
              f"{'stored' if res.stored else ('dup: ' + res.store_note if res.ok else 'failed at ' + str(last.stage))}  {time.time() - t:5.1f}s", flush=True)
    rates = tally(results)
    report = {"inventory": {"seed": args.seed, "size": args.size, "parts": facts.total_parts, "colours": facts.colours},
              "concepts": [{"name": r.concept.name, "category": r.concept.category, "silhouette": r.concept.silhouette, "source": r.concept.source,
                            "ok": r.ok, "stored": r.stored, "store_note": r.store_note, "attempts": [
                                {"n": a.number, "stage": a.stage, "errors": a.errors[:3], "answer": a.answer, "reached": a.reached, "passed": a.passed,
                                 "skipped": a.skipped, "service_error": a.service_error}
                                for a in r.attempts]} for r in results],
              "pass_rates": rates}
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nstage            attempts passed/reached     concepts passed (ever)   passed on first try")
    for stage in STAGES:
        r = rates[stage]
        pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a "
        print(f"{stage:16} {r['attempts_passed']:3}/{r['attempts_reached']:<3} {pct(r['attempts_passed'], r['attempts_reached'])}"
              f"        {r['concepts_passed']:3}/{r['concepts']:<3} {pct(r['concepts_passed'], r['concepts'])}"
              f"          {r['concepts_passed_first_try']:3}/{r['concepts']:<3} {pct(r['concepts_passed_first_try'], r['concepts'])}")
    f = rates["final"]
    print(f"\nfinal: {f['ok']}/{f['concepts']} concepts ended with a recipe that passed everything; {f['stored']} stored in the library; "
          f"{f['recipe_calls']} recipe calls to Gemini ({f['gemini_service_errors']} failed because Gemini itself errored, not counted above). Report: {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=30)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--library", default="recipe_library_study")
    ap.add_argument("--out", default="docs/recipe-study.json")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--no-recognise", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not config.GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
