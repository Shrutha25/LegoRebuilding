"""HTTP endpoints of the 3D building-instructions viewer (additive: nothing existing is changed).

    POST /api/builds/ai                    an AI build (Gemini, placed on a stud grid, validated and repaired locally) for the pieces
    POST /api/sets/{set_num}/build         an official set: its LDraw model if the OMR has one, plus the parts list (have / missing)
    GET  /api/builds/{id}                  build metadata: title, steps with the parts each step adds, attribution
    GET  /api/builds/{id}/model.mpd        the packed model (only the library files it uses); ?lite=1 drops conditional lines
    GET  /api/ldraw/LDConfig.ldr           the LDraw colour table the viewer needs
    POST /api/builds/recipes               recipe builds (backend/recipes): designs the code generates from the pieces, named by Gemini
    GET  /api/builds/{id}/thumb.png        isometric thumbnail of a recipe build
    POST /api/sets/nearby                  sets the pieces can build now, and sets that are almost there (with the missing parts)
    POST /api/designs/stream               "Designed for your pile": concepts from Gemini, fitted by the local solver, streamed (server-sent events)
    POST /api/subbuilds                    real LEGO sub-builds (OMR) the pieces can build now, and ones missing 1-2 parts
    POST /api/subbuilds/build              the 3D viewer build of one of them
    POST /api/sets/instructions            which instructions each set has (3D model / official LEGO link / none); pending ones fill in later
    GET  /api/dev/recipe-gallery           dev only (RECIPE_GALLERY=1): every recipe on random inventories
    GET  /api/builds-status                whether the LDraw library is installed (for the UI)

The LDraw parts library itself is never served.
"""

import csv
import logging
import threading
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend import config, timing
from backend.core import idmap, omr
from backend.core.aibuild import BuildError
from backend.core.buildsvc import BuildService, valid_id
from backend.core.ideas import IdeasError
from backend.core.ldmap import ColourMapper, PartMapper
from backend.core.setmatch import POOL, TOP
from backend.core.ldraw_lib import DEFAULT_ZIP, LDrawLibrary, LibraryMissing

log = logging.getLogger("lego.builds_api")
router = APIRouter()

_deps = {}
_service = None
_setinfo = None


def get_setinfo():
    """Card states (3D model / official instructions / none), created on first use; lookups run in a background thread."""
    global _setinfo
    if _setinfo is None:
        from backend.core.instructions import InstructionsService
        from backend.core.setinfo import SetInfo
        svc = InstructionsService(config.INSTRUCTIONS_DB, config.BRICKSET_API_KEY, config.INSTRUCTIONS_TTL_DAYS,
                                  config.BRICKSET_DAILY_CAP, config.BRICKSET_MIN_INTERVAL)
        _setinfo = SetInfo(svc, config.BUILD_CACHE_DIR, config.OMR_BASE_URL)
    return _setinfo


def rank_by_instructions(rows, top):
    """Annotate match rows with their `instructions` state and sort: state 1, 2, then 3; completeness; part count."""
    from backend.core.setinfo import rank_key
    get_setinfo().annotate(rows)
    return sorted(rows, key=rank_key)[:top]


def configure(store, rb_api, get_llm, slots):
    """Hand the app's shared objects to this module (called once from main.py)."""
    _deps.update(store=store, rb_api=rb_api, get_llm=get_llm, slots=slots)


def reset_service():
    """Forget the cached service (tests, or after the library was downloaded)."""
    global _service
    _service = None


def _colour_rows():
    path = config.DATA_DIR / "colors.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return [(r["id"], r["name"], r["rgb"]) for r in csv.DictReader(f)]


def get_service():
    """The build service, created on first use. 503 with a clear message while the LDraw library is not downloaded."""
    global _service
    if _service is None:
        try:
            lib = LDrawLibrary(config.LDRAW_DIR / DEFAULT_ZIP)
        except LibraryMissing as e:
            raise HTTPException(503, f"3D instructions need the LDraw parts library. {e}")
        store = _deps["store"]
        known = dict(store.known_ids) if store.known_ids else {}
        rel_map = dict(store.index.rel_map) if store.index else {}
        parts = PartMapper(lib, known, rel_map, _deps["rb_api"], config.BUILD_CACHE_DIR / "ldraw_ids.json")
        _service = BuildService(lib, parts, ColourMapper(lib, _colour_rows()), config.BUILD_CACHE_DIR, config.OMR_BASE_URL)
    return _service


class BuildPartIn(BaseModel):
    part_num: str = Field(max_length=40)
    color: str | None = Field(default=None, max_length=40)
    qty: int = Field(default=1, ge=1, le=999)
    name: str | None = Field(default=None, max_length=100)


class IdeaPieceIn(BaseModel):
    part_num: str = Field(max_length=40)
    qty: int = Field(ge=1, le=999)
    colour: str = Field(default="", max_length=40)


class IdeaIn(BaseModel):
    name: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=400)
    pieces_used: list[IdeaPieceIn] = Field(default_factory=list, max_length=60)   # the pieces the idea card lists: the 3D build must use these
    steps: list[str] = Field(default_factory=list, max_length=12)                  # the card's steps: the 3D build follows them


class AiBuildIn(BaseModel):
    parts: list[BuildPartIn] = Field(min_length=1, max_length=500)
    idea: IdeaIn | None = None
    match_color: bool = True          # "Consider color": False counts pieces by part only and colours the build afterwards


class SetBuildIn(BaseModel):
    parts: list[BuildPartIn] = Field(default_factory=list, max_length=500)
    match_color: bool | None = None   # "Consider color": True counts by (part, colour), False by part only, None reports both
    set_name: str | None = Field(default=None, max_length=200)


@router.get("/api/builds-status")
def builds_status():
    ready = (config.LDRAW_DIR / DEFAULT_ZIP).exists()
    return {"ldraw_library": ready, "hint": None if ready else "Run: python -m backend.core.fetch_ldraw"}


@router.post("/api/builds/ai")
async def ai_build(body: AiBuildIn):
    svc = get_service()
    store = _deps["store"]
    names = store.part_names()
    parts = []
    for p in body.parts:
        rid = idmap.resolve_local(p.part_num, store.known_ids) if store.known_ids else None
        parts.append({"part_num": p.part_num, "color": p.color, "qty": p.qty, "name": p.name or (names.get(rid) if rid else None)})
    try:
        async with _deps["slots"]:
            return await svc.create_ai_build(parts, body.idea.model_dump() if body.idea else None, _deps["get_llm"](), body.match_color)
    except (BuildError, IdeasError) as e:
        raise HTTPException(e.status, str(e))


@router.get("/api/builds/{build_id}")
def build_meta(build_id: str):
    meta = get_service().meta(build_id) if valid_id(build_id) else None
    if meta is None:
        raise HTTPException(404, "Unknown build.")
    return meta


@router.get("/api/builds/{build_id}/model.mpd")
def build_model(build_id: str, lite: bool = False):
    text = get_service().mpd(build_id, lite) if valid_id(build_id) else None
    if text is None:
        raise HTTPException(404, "Unknown build.")
    return PlainTextResponse(text, headers={"Cache-Control": "public, max-age=31536000, immutable"})   # ids are content hashes / set numbers


@router.get("/api/ldraw/LDConfig.ldr")
def ldconfig():
    text = get_service().lib.read("LDConfig.ldr")
    if text is None:
        raise HTTPException(404, "LDConfig.ldr is not in the library.")
    return PlainTextResponse(text, headers={"Cache-Control": "public, max-age=86400"})


def set_inventory(set_num, user_parts, store):
    """The set's parts with how many of each the user has: [{part, name, colour, need, have, missing, image}]."""
    index = store.index
    need = (index.req[True].get(set_num) if index else None) or {}
    names = store.part_names()
    colours = {c[1].lower(): c[1] for c in _colour_rows()}
    have = {}
    for p in user_parts:
        rid = idmap.resolve_local(p.part_num, store.known_ids) or p.part_num
        key = index._key(rid, p.color, True)
        have[key] = have.get(key, 0) + p.qty
    rows = []
    for (part, colour), qty in sorted(need.items(), key=lambda kv: (-kv[1], kv[0])):
        got = min(have.get((part, colour), 0), qty)
        rows.append({"part": part, "name": names.get(part, part), "colour": colours.get(colour, colour.title()), "need": qty,
                     "have": got, "missing": qty - got, "image": index.image_for(part, colour)})
    return rows


@router.post("/api/sets/{set_num}/build")
async def set_build(set_num: str, body: SetBuildIn):
    store = _deps["store"]
    if not store.data_loaded:
        raise HTTPException(503, store.error or "Set data is still loading; try again shortly.")
    base = set_num.split("-")[0]
    result = {"set_num": set_num, "name": body.set_name, "available": False, "reason": None, "build": None,
              "instructions_url": f"https://www.lego.com/service/buildinginstructions/{base}",
              "inventory": await run_in_threadpool(set_inventory, set_num, body.parts, store)}
    result["need_total"] = sum(r["need"] for r in result["inventory"])
    result["have_total"] = sum(r["have"] for r in result["inventory"])
    try:
        meta, reason = await get_service().official_build(set_num, body.set_name)
    except HTTPException:
        raise
    except Exception as e:                                                       # noqa: BLE001  the parts list is still useful
        log.warning("official build for %s failed: %s", set_num, e)
        meta, reason = None, "error"
    result["available"], result["reason"], result["build"] = meta is not None, reason, meta
    result["instructions"] = get_setinfo().info(set_num)
    return result


# ------------------------------------------------------------------ recipe builds, nearby sets, gallery (all additive)
class RecipesIn(BaseModel):
    parts: list[BuildPartIn] = Field(min_length=1, max_length=500)
    use_ai: bool = True
    match_color: bool = True


def _named_parts(parts):
    store = _deps["store"]
    names = store.part_names() if store.parts_loaded else {}
    out = []
    for p in parts:
        rid = idmap.resolve_local(p.part_num, store.known_ids) if store.known_ids else None
        out.append({"part_num": p.part_num, "color": p.color, "qty": p.qty, "name": p.name or (names.get(rid) if rid else None)})
    return out


@router.post("/api/builds/recipes")
async def recipe_builds_endpoint(body: RecipesIn):
    from backend.recipes.service import recipe_builds
    svc = get_service()
    llm = None
    if body.use_ai:
        try:
            llm = _deps["get_llm"]()
        except IdeasError:
            llm = None                                           # no API key: standard names, the builds still work
    try:
        async with _deps["slots"]:
            return await recipe_builds(svc, _named_parts(body.parts), llm, use_ai=body.use_ai, match_color=body.match_color)
    except (BuildError, IdeasError) as e:
        raise HTTPException(e.status, str(e))


@router.get("/api/builds/{build_id}/thumb.png")
def build_thumb(build_id: str):
    png = get_service().thumb(build_id) if valid_id(build_id) else None
    if png is None:
        raise HTTPException(404, "No thumbnail for this build.")
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=31536000, immutable"})


class InstructionsIn(BaseModel):
    set_nums: list[str] = Field(min_length=1, max_length=60)


@router.post("/api/sets/instructions")
async def sets_instructions(body: InstructionsIn):
    """{"sets": {set_num: {state: 1|2|3|null, links: [{url, description}], pending}}}: caches only, lookups run in the background."""
    info = get_setinfo()
    nums = [s for s in dict.fromkeys(body.set_nums) if omr.valid_set_num(s)]
    out = {s: info.info(s) for s in nums}
    info.prefetch([s for s in nums if out[s]["pending"]])
    return {"sets": out}


_matcher = None
_omr = {"index": None, "error": None, "bank": None}                       # the OMR sub-build index (backend/core/omr_index.py), built at start-up
_warm = {"nearby": False, "omr": False, "service": False, "examples": False, "seconds": {}}


def warm_indexes(store, wait=600):
    """Build every index the Suggest sources need, once, at start-up (called in a thread): the nearby-sets matcher, the OMR sub-build index and the LDraw
    library behind the part and colour mappers. Waits for the catalogue first. Never raises: a source that fails to warm just answers when asked."""
    t0 = time.time()
    while not (store.data_loaded and store.parts_loaded) and time.time() - t0 < wait and not store.error:
        time.sleep(0.2)
    if not (store.data_loaded and store.parts_loaded):
        return
    for name, fn in (("nearby", lambda: _get_matcher(store)), ("service", get_service), ("omr", lambda: _load_omr(store)), ("examples", lambda: _load_examples())):   # these need the LDraw library
        t = time.time()
        try:
            fn()
            _warm[name] = True
        except HTTPException as e:
            log.info("warm-up of %s skipped: %s", name, e.detail)
        except Exception as e:                                                        # noqa: BLE001
            log.warning("warm-up of %s failed: %s: %s", name, type(e).__name__, e)
        _warm["seconds"][name] = round(time.time() - t, 2)
    log.info("indexes ready after %.1fs: %s", time.time() - t0, _warm)


def start_warmup(store):
    th = threading.Thread(target=warm_indexes, args=(store,), daemon=True, name="warm-indexes")
    th.start()
    return th


def _load_omr(store):
    from backend.core.omr_index import OmrIndex
    names = dict(zip(store.index.sets.set_num, store.index.sets.name)) if store.index is not None else {}
    canon, key = None, None
    try:
        svc = get_service()
        canon = svc.canon
        zip_path = config.LDRAW_DIR / DEFAULT_ZIP
        key = (zip_path.stat().st_mtime_ns, zip_path.stat().st_size)
    except HTTPException:                                                              # no LDraw library: part numbers are used as written
        pass
    _omr["index"] = OmrIndex.load(config.BUILD_CACHE_DIR / "omr", config.BUILD_CACHE_DIR / "omr_index.pkl", names, canon, key)
    return _omr["index"]


def _load_examples():
    from backend.designs.examples import ExampleBank
    svc = get_service()
    zip_path = config.LDRAW_DIR / DEFAULT_ZIP
    _omr["bank"] = ExampleBank.load(_omr["index"], svc.lib, config.BUILD_CACHE_DIR / "omr", config.BUILD_CACHE_DIR / "omr_examples.pkl", svc.canon,
                                    (zip_path.stat().st_mtime_ns, zip_path.stat().st_size))
    return _omr["bank"]


def warm_status():
    idx = _omr["index"]
    return {**{k: v for k, v in _warm.items() if k != "seconds"}, "omr_sub_builds": len(idx) if idx is not None else 0, "seconds": _warm["seconds"]}


def _get_matcher(store):
    global _matcher
    if _matcher is None:
        from backend.core.setmatch import SetMatcher
        _matcher = SetMatcher(store.index, store.parts)
    return _matcher


@router.post("/api/sets/nearby")
async def sets_nearby(body: SetBuildIn):
    store = _deps["store"]
    if not store.data_loaded or not store.parts_loaded:
        raise HTTPException(503, store.error or "Set data is still loading; try again shortly.")
    names = store.part_names()
    parts = []
    for p in body.parts:
        rid = idmap.resolve_local(p.part_num, store.known_ids) or p.part_num
        parts.append({"part_num": rid, "color": p.color, "qty": p.qty})
    with timing.stage("nearby.index"):
        matcher = await run_in_threadpool(_get_matcher, store)
    with timing.stage("nearby.match"):
        result = await run_in_threadpool(matcher.match, parts, names, POOL, body.match_color)
    with timing.stage("nearby.rank"):
        for key in ("can_build", "almost"):
            result[key] = await run_in_threadpool(rank_by_instructions, result[key], TOP)
    for row in result["can_build"] + result["almost"]:                    # the shape SetCard already renders
        row["parts_you_have"] = row["total_parts"] - row["missing_count"]
        row["coverage"] = round(row["parts_you_have"] / max(1, row["total_parts"]), 3)
        row["complete"] = row["missing_count"] == 0
        for m in row["missing"]:
            m["color"] = m["colour"]
    return result


class DesignsIn(BaseModel):
    parts: list[BuildPartIn] = Field(min_length=1, max_length=500)
    match_color: bool = True


def _design_llm():
    """The Gemini client for the concept call: the app's one, or a separate Flash-tier model when GEMINI_DESIGN_MODEL differs from GEMINI_MODEL."""
    llm = _deps["get_llm"]()
    if config.GEMINI_DESIGN_MODEL != getattr(llm, "model", None):
        from backend.core import gemini_budget
        from backend.core.ideas import GeminiLLM
        return GeminiLLM(config.GEMINI_API_KEY, config.GEMINI_DESIGN_MODEL, timeout=90.0, budget=gemini_budget.get_budget())
    return llm


@router.post("/api/designs/stream")
async def designs_stream(body: DesignsIn):
    """"Designed for your pile". The answer is a stream of server-sent events (text/event-stream): start, design (one per design that passed the checks,
    as soon as it is ready), verdict (the look check removes designs), note, done. At most 3 Gemini calls per request."""
    from backend.designs.pipeline import stream_designs
    svc = get_service()
    try:
        llm = _design_llm()
    except IdeasError:
        llm = None
    try:
        look_llm = _deps["get_llm"]()                  # the look check always uses the app's own model, whatever GEMINI_DESIGN_MODEL is
    except IdeasError:
        look_llm = None
    parts = _named_parts(body.parts)

    async def events():
        async with _deps["slots"]:
            async for chunk in stream_designs(svc, parts, llm, body.match_color, bank=_omr["bank"], look_llm=look_llm):
                yield chunk
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SubBuildsIn(BaseModel):
    parts: list[BuildPartIn] = Field(default_factory=list, max_length=500)
    match_color: bool = True


def _clean(desc):
    return " ".join(str(desc or "").lstrip("~=_").split())


def have_for_omr(parts, svc):
    """{(ld part, ld colour or None): qty} for the user's pieces: the same part and colour mappers the 3D builds use (local rules only: no network)."""
    have = {}
    for p in parts:
        ld = svc.parts.ldraw_id(p.part_num)
        if not ld:
            continue
        ld = svc.canon(ld)
        code = svc.colours.code(p.color) if p.color else None
        have[(ld, code)] = have.get((ld.lower(), code), 0) + p.qty
    return have


def subbuild_rows(rows, svc, store, with_missing):
    sets = store.index.sets if store.index is not None else None
    images = dict(zip(sets.set_num, sets.img_url)) if sets is not None else {}
    out = []
    for b, missing in rows:
        row = {"id": b.id, "set_num": b.set_num, "set_name": b.set_name, "title": b.title, "parts": b.parts, "main": b.main,
               "authors": list(b.authors), "licence": list(b.licence), "image": images.get(b.set_num) or None,
               "source_url": f"https://library.ldraw.org/omr/sets?search={b.set_num.split('-')[0]}"}
        if with_missing:
            row["missing_count"] = sum(q for _, _, q in missing)
            row["missing"] = [{"part": part, "name": _clean(svc.lib.part_description(part)) or part,
                               "colour": None if col in (None, 16) else svc.colours.name(col), "qty": q} for part, col, q in missing]
        out.append(row)
    return out


@router.post("/api/subbuilds")
async def subbuilds(body: SubBuildsIn):
    """Real sub-builds from the LDraw Official Model Repository: `buildable` the pieces cover completely, `almost` missing 1-2 parts (listed)."""
    store = _deps["store"]
    idx = _omr["index"]
    if idx is None or not (store.data_loaded and store.parts_loaded):
        return {"ready": False, "buildable": [], "almost": [], "counts": {}}
    if len(idx) == 0:
        return {"ready": True, "available": False, "hint": "Run: python -m backend.core.fetch_omr", "buildable": [], "almost": [], "counts": {}}
    svc = get_service()
    with timing.stage("subbuilds.map"):
        have = await run_in_threadpool(have_for_omr, body.parts, svc)
    with timing.stage("subbuilds.match"):
        res = await run_in_threadpool(idx.match, have, body.match_color)
    with timing.stage("subbuilds.rows"):
        return {"ready": True, "available": True, "match_colour": body.match_color,
                "buildable": subbuild_rows(res["buildable"], svc, store, False), "almost": subbuild_rows(res["almost"], svc, store, True),
                "counts": res["counts"]}


class SubBuildIn(BaseModel):
    id: str = Field(max_length=60)
    title: str | None = Field(default=None, max_length=200)


@router.post("/api/subbuilds/build")
async def subbuild_build(body: SubBuildIn):
    """The 3D viewer build (steps, parts, attribution) of one sub-build found by /api/subbuilds."""
    svc = get_service()
    meta, reason = await run_in_threadpool(svc.official_submodel, body.id, body.title)
    if meta is None:
        raise HTTPException(404 if reason == "none" else 403, "No open 3D model for this build." if reason == "none" else "That model's licence does not allow showing it here.")
    return meta


@router.get("/api/dev/recipe-gallery")
async def recipe_gallery(seed: int = 1, per: int = Query(5, ge=1, le=8)):
    if not config.RECIPE_GALLERY:
        raise HTTPException(404, "Set RECIPE_GALLERY=1 to enable the recipe gallery.")
    import random
    from backend.core.ldrconv import build_to_ldr
    from backend.core.render_iso import render_png
    from backend.recipes import all_recipes
    from backend.recipes.inventories import random_inventory
    from backend.recipes.planner import plan_candidates
    from backend.recipes.suggest import Shown
    from backend.recipes.service import store_candidate
    svc = get_service()
    out = []
    for recipe in all_recipes():
        rows = []
        for i in range(per):
            s = seed * 1000 + i
            inv = random_inventory(random.Random(s), random.Random(s).randint(40, 120))
            cands = await run_in_threadpool(plan_candidates, inv, s, 3, [recipe])
            summary = ", ".join(f"{c.qty} x {c.part_id} {c.colour}" for c in inv.parts()[:6]) + ("..." if len(inv.parts()) > 6 else "")
            if not cands:
                rows.append({"seed": s, "build": None, "inventory": summary, "pieces": sum(c.qty for c in inv.parts())})
                continue
            best = cands[0]
            meta = await run_in_threadpool(store_candidate, svc, inv, Shown(best, best.default_name(), f"{best.part_count} parts", False))
            rows.append({"seed": s, "build": meta, "inventory": summary, "pieces": sum(c.qty for c in inv.parts())})
        out.append({"recipe": recipe.id, "name": recipe.title, "rows": rows})
    return {"recipes": out}
