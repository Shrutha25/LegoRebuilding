"""Recipe builds end to end: inventory -> suggestion -> stored builds (LDraw + steps + thumbnail) in the viewer's own format.

The stored metadata is the same JSON the viewer already reads (BuildMeta: kind "ai", steps, model_url ...) plus a `recipe` object, so the
existing viewer opens these builds unchanged. Build ids are content hashes ("ai-<hash>"): the same inventory gives the same builds.
"""

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import replace

from backend import config
from backend.core import aibuild, ldrconv
from backend.core.buildmodel import Build
from backend.core.inventory_check import can_build, pool_inventory, recolour
from backend.core.render_iso import render_png
from backend.recipes import vision
from backend.recipes.library import RecipeLibrary
from backend.recipes.planner import name_for
from backend.recipes.suggest import Shown, suggest

log = logging.getLogger("lego.recipes.service")


def inventory_seed(inv):
    """A stable seed from the inventory, so the same pieces give the same builds."""
    key = "|".join(f"{i.part_id}/{i.colour}/{i.qty}" for i in sorted(inv.parts(), key=lambda i: (i.part_id, i.colour.lower())))
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def finalize(shown, inv, match_color):
    """The last gate before a build is shown. With colour ignored the pooled build gets real colours from the user's stock; then the
    inventory rule is checked on the finished parts. Returns the Shown to store, or None: a build that fails is dropped, never repaired."""
    cand = shown.candidate
    parts = cand.build.parts
    if not match_color:
        parts = recolour(parts, cand.sections, inv)
    if not (can_build(parts, inv, match_color) and can_build(parts, inv, True)):
        log.warning("build %r uses pieces the user does not have: dropped", shown.name)
        return None
    colours = dict(cand.colours)
    name = shown.name
    if not match_color:                                       # real colours per role: the most used colour of its parts
        by_role = {}
        for p in parts:
            by_role.setdefault(cand.sections.get((p.x, p.z, p.y)), Counter())[p.colour] += 1
        colours = {role: c.most_common(1)[0][0] for role, c in by_role.items() if role}
        name = name.replace("Mixed ", "").strip() or name_for(cand.recipe, colours, cand.params)
    cand = replace(cand, build=Build(cand.build.title, cand.build.description, parts), colours=colours)
    return Shown(cand, name, shown.description, shown.from_ai, shown.concept)


def store_candidate(svc, inv, shown, match_color=True):
    """Store one chosen build; returns its metadata with the recipe info and the thumbnail URL."""
    cand = shown.candidate
    build = Build(shown.name, shown.description, cand.build.parts)
    ldr = ldrconv.build_to_ldr(build, inv, author="Brick Finder (recipe build)")
    build_id = "ai-" + hashlib.sha1(ldr.encode("utf-8")).hexdigest()[:12]
    meta = {
        "kind": "ai", "title": shown.name, "description": shown.description, "attribution": None,
        "repairs": [], "warnings": [], "skipped_parts": [],
        "repair_summary": {"dropped": 0, "swapped": 0, "nudged": 0, "steps_regenerated": False, "retried": False,
                           "asked_for": cand.part_count},
        "recipe": {"id": cand.recipe_id, "category": cand.recipe.category, "params": cand.params, "colours": cand.colours,
                   "score": cand.score, "named_by_ai": shown.from_ai,
                   "concept": {"silhouette": shown.concept.silhouette, "bbox": shown.concept.bbox, "features": list(shown.concept.features),
                               "source": shown.concept.source} if shown.concept else None},
        "thumb_url": f"/api/builds/{build_id}/thumb.png",
        "match_colour": match_color, "mixed_colours": not match_color,
    }
    stored = svc._store(build_id, ldr, meta)
    if svc.thumb(build_id) is None:
        svc.save_thumb(build_id, render_png(build, inv, svc.colours.hex))
    return stored


async def recipe_builds(svc, parts, llm, seed=None, use_ai=True, show=None, match_color=True):
    """parts: [{part_num, name?, color?, qty}]. Returns {"builds": [meta], "unlock": [...], "ai_used", "ai_note", "timings", ...}."""
    t0 = time.perf_counter()
    for p in parts:                                      # parts the local rules cannot map: ask Rebrickable for the LDraw id (cached)
        await svc.parts.ldraw_id_async(str(p.get("part_num") or ""))
    inv, skipped = aibuild.build_inventory(parts, svc.parts, svc.colours)
    t1 = time.perf_counter()
    if not inv.parts():
        return {"builds": [], "unlock": [], "ai_used": False, "ai_note": "", "candidate_count": 0, "freeform": False,
                "skipped_parts": [{"part": p, "reason": r} for p, r in skipped],
                "timings_ms": {"inventory": round((t1 - t0) * 1000)}}
    plan_inv = inv if match_color else pool_inventory(inv)       # colour ignored: plan from part counts, colour the result afterwards
    result = await suggest(plan_inv, llm if use_ai else None, inventory_seed(plan_inv) if seed is None else seed, show or config.RECIPE_SHOW,
                           config.RECIPE_AI_TIMEOUT, use_ai, match_color, RecipeLibrary(config.RECIPE_LIBRARY_DIR), svc.colours.hex)
    t2 = time.perf_counter()
    final = [f for f in (finalize(s, inv, match_color) for s in result.shown) if f is not None]    # fewer builds, never invalid ones
    builds = [store_candidate(svc, inv, s, match_color) for s in final]
    freeform = False
    if not builds and use_ai and llm is not None and config.AI_FREEFORM_BUILDS:
        extra = await freeform_fallback(svc, parts, inv, llm, match_color)
        if extra:
            builds, freeform = [extra], True
    t3 = time.perf_counter()
    return {"builds": builds, "match_colour": match_color, "unlock": result.unlock, "concepts": len(result.concepts),
            "unbuilt": [{"concept": n, "why": w} for n, w in result.unbuilt[:12]], "generated": result.generated, "ai_used": result.ai_used, "ai_note": result.ai_note,
            "candidate_count": result.candidate_count, "freeform": freeform,
            "skipped_parts": [{"part": p, "reason": r} for p, r in skipped],
            "timings_ms": {"inventory": round((t1 - t0) * 1000), "plan_and_name": round((t2 - t1) * 1000),
                           "store": round((t3 - t2) * 1000)}}


async def freeform_fallback(svc, parts, inv, llm, match_color=True):
    """The old free-form Gemini placement, only when no recipe fits and AI_FREEFORM_BUILDS is on. The result is checked with a vision
    call ("does it look like its name?"): a build that does not is renamed honestly instead of keeping a false name."""
    try:
        meta = await svc.create_ai_build(parts, None, llm, match_color)
    except Exception as e:                                           # noqa: BLE001  nothing else to offer
        log.info("free-form fallback failed: %s", e)
        return None
    build = ldrconv.ldr_to_build(svc._path(meta["id"], "ldr").read_text(encoding="utf-8"), inv, meta["title"])
    png = render_png(build, inv, svc.colours.hex)
    svc.save_thumb(meta["id"], png)
    verdict = await vision.check_freeform(llm, png, meta["title"])
    meta["thumb_url"] = f"/api/builds/{meta['id']}/thumb.png"
    meta["look_check"] = verdict
    if verdict == "no":
        meta["title"], meta["description"] = "Abstract build", "A free-form AI design. It does not clearly look like a specific object."
    svc._path(meta["id"], "json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
