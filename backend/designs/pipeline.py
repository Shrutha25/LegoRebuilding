"""Designed for your pile: Gemini imagines shapes, a local solver fits the exact bricks, checks judge the result, and every design that passes streams out.

    async for event in stream_designs(svc, parts, llm, match_color): ...

Events (dicts; the API sends them as server-sent events):
    start     the pile as the solver sees it: pieces, kinds, volume, size limits
    design    one design that passed the code checks, the moment it is ready: the viewer build (steps, parts), thumbnail, how much of the pile it uses
    verdict   the batched look check: {"id", "ok", "choice"}; a design with ok = false is removed from the page
    note      anything the user may want to know (the AI is unavailable, nothing fit ...)
    done      counts, Gemini calls (at most 3 per request), per-stage tallies, milliseconds

Order of work (calls: 1 concepts, [2 repair], last: look check; never more than MAX_CALLS):
    1. one streamed concept call (concepts.py): 6-8 concepts as JSON objects; each is parsed the moment it is complete and handed to the solver, in parallel
    2. solver (solver.py, 2 s each) -> exact parts (colours from the real stock afterwards) -> can_build, structure, IoU against the designs already shown
    3. if fewer than 3 designs passed and some concepts failed: ONE repair call for new concepts, solved the same way
    4. ONE batched look check: a numbered contact sheet of all the designs, multiple choice per picture; a design that is not recognised is removed (logged)
"""

import asyncio
import contextvars
import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from backend import config, timing
from backend.core import aibuild, ldrconv
from backend.core import buildmodel as bm
from backend.core.buildmodel import Build, Placement
from backend.core.inventory_check import can_build
from backend.core.render_iso import render_png
from backend.designs import concepts as cn
from backend.designs.affordance import MIN_USE, summarize
from backend.designs import sketch as sk
from backend.designs.solver import solve
from backend.recipes import vision
from backend.recipes.checks import IOU_MAX, iou, structural_problems, voxels_of_build

log = logging.getLogger("lego.designs")

MAX_CALLS = config.DESIGN_MAX_CALLS       # 1: the concept call only; 3 adds one repair call and one batched look check
COVERAGE_CLEAN = 0.95               # a fill of at least this share of the sketch passes as it is ...
COVERAGE_TOLERATED = 0.85           # ... one of at least this share passes too when the sketch's outlines (top, front, side) stay mostly intact:
OUTLINE_IOU_MIN = 0.80              #     the lowest of the three outline IoUs must be at least this
MIN_PARTS = 3                       # ... and a design needs at least this many parts
LOOK_RULE = "child"                 # which look-check answer removes a design: "child" (could a child see this as <name>?) or "choice" (picked its own name from the list)
EXAMPLES_IN_PROMPT = 2                # real OMR models shown next to the fixed teaching examples
SMALL_PILE = 8                      # fewer pieces than this: a friendly "add a few more pieces" message when no design passes
SOLVER_SECONDS = 2.0
ENOUGH = 3                          # fewer designs than this after the first round: one repair call
POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="solver")
NONE_OF_THESE = "none of these"


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def in_thread(fn, *args):
    """Run fn in the solver pool with the request's timing trace (contextvars do not follow executor threads on their own)."""
    ctx = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(POOL, ctx.run, fn, *args)


def pooled_parts(inv):
    """One row per part and colour of the pile: (part_id, colour, dims, qty)."""
    return [(i.part_id, i.colour, i.dims, i.qty) for i in inv.parts()]


def solve_concept(concept, parts, sketch):
    with timing.stage("solver", concept=concept.name):
        return solve(sketch, parts, time_limit=SOLVER_SECONDS, workers=2)


def build_design(res, concept, plan_inv, inv, match_color):
    """(Build, None) from a solver result, or (None, why): steps, the inventory rule (per colour and per part) and the structure check. The solver already
    used the pile's own parts and colours, so nothing is recoloured."""
    placements = [Placement(pid, colour, x, z, y, rot, 1) for pid, colour, x, z, y, rot in res.placements]
    with timing.stage("checks", concept=concept.name):
        coloured = bm.generate_steps(placements, inv)
        build = Build(concept.name, concept.description, coloured)
        if not (can_build(coloured, inv, match_color) and can_build(coloured, inv, True)):
            return None, "the parts it needs are not all in the pile"
        problems = structural_problems(build, inv)
        if problems:
            return None, problems[0]
    return build, None


def store_design(svc, inv, build, extra):
    """Write the viewer files (LDraw, packed model, steps) and the thumbnail; returns the build metadata the viewer reads."""
    with timing.stage("ldr", build=build.title):
        ldr = ldrconv.build_to_ldr(build, inv, author="Brick Finder (designed for your pile)")
    build_id = "ai-" + hashlib.sha1(ldr.encode("utf-8")).hexdigest()[:12]
    meta = {"kind": "ai", "title": build.title, "description": build.description, "attribution": None, "repairs": [], "warnings": [], "skipped_parts": [],
            "repair_summary": {"dropped": 0, "swapped": 0, "nudged": 0, "steps_regenerated": False, "retried": False, "asked_for": len(build.parts)},
            "thumb_url": f"/api/builds/{build_id}/thumb.png", **extra}
    with timing.stage("ldr.pack", build=build.title):
        stored = svc._store(build_id, ldr, meta)
    if svc.thumb(build_id) is None:
        with timing.stage("thumbnail", build=build.title):
            svc.save_thumb(build_id, render_png(build, inv, svc.colours.hex))
    return stored


class Run:
    """One request: the state the orchestration shares."""

    def __init__(self, svc, inv, plan_inv, aff, llm, match_color, bank, look_llm=None):
        self.svc, self.inv, self.plan_inv, self.aff, self.llm, self.match_color, self.bank = svc, inv, plan_inv, aff, llm, match_color, bank
        self.look_llm = look_llm or llm                # the look check may use another model than the concept call
        self.parts = pooled_parts(inv)
        self.queue = asyncio.Queue()
        self.calls = 0
        self.t0 = time.perf_counter()
        self.designs = []                  # {"meta", "voxels", "concept"}
        self.failures = []                 # (concept name, plain reason)
        self.seen = set()
        self.sketches = []                 # concepts whose sketch was accepted for solving (two concepts with the same shape are solved once)
        self.stats = {"concepts": 0, "sketch_ok": 0, "solver_ok": 0, "checks_ok": 0, "diverse_ok": 0, "look_checked": 0, "look_ok": 0,
                      "look_ok_choice": 0, "look_ok_child": 0, "concept_in_chars": 0, "concept_out_chars": 0, "look_in_chars": 0, "look_out_chars": 0, "rescued": 0, "outline_rejected": 0, "fill_rejected": 0, "undersized": 0, "dup_dropped": 0, "examples": [],
                      "first_design_ms": None, "last_design_ms": None}
        self.tasks = []

    def ms(self):
        return round((time.perf_counter() - self.t0) * 1000)

    def emit(self, event, **data):
        self.queue.put_nowait((event, {**data, "t_ms": self.ms()}))

    # -------------------------------------------------------------- one concept: solver, checks, store
    async def handle(self, concept):
        self.stats["solver_ok"] += 0
        lim = self.aff.limits
        sketch, sx, sy = sk.fit_scale(concept.sketch, self.aff.cells, lim["width"], lim["depth"], lim["layers"])          # drawn small, enlarged to the pile
        res = await in_thread(solve_concept, concept, self.parts, sketch)
        filled = sketch.filled
        pile_cells = self.aff.cells
        outline = sk.outline_iou(sketch, res.cells) if res.cells else 0.0
        rescued = COVERAGE_TOLERATED <= res.coverage < COVERAGE_CLEAN and outline >= OUTLINE_IOU_MIN
        enough = res.coverage >= COVERAGE_CLEAN or rescued
        if len(res.placements) < MIN_PARTS or not enough:
            if res.placements and COVERAGE_TOLERATED <= res.coverage < COVERAGE_CLEAN:
                self.stats["outline_rejected"] += 1
                why = f"the pile can fill {res.coverage:.0%} of the sketch but the outline is lost (outline match {outline:.0%}): draw a simpler shape the sizes can make"
            elif res.placements and res.coverage >= COVERAGE_CLEAN:
                why = f"it would be built from only {len(res.placements)} part(s): too plain to be a model, give it more detail"
            elif res.placements:
                why = (f"the pile can fill only {res.coverage:.0%} of the sketch ({filled} cells; the pile's volume is {pile_cells:.0f} cells and its sizes "
                       f"cannot make this shape)")
            else:
                why = "none of the pile's parts fit into the sketch"
            self.failures.append((concept.name, why))
            log.info("design %r: solver %s, coverage %.0f%%, %d parts: dropped", concept.name, res.status, 100 * res.coverage, len(res.placements))
            return
        self.stats["solver_ok"] += 1
        self.stats["rescued"] += rescued
        build, why = await in_thread(self._check, res, concept)
        if build is None:
            self.failures.append((concept.name, why))
            log.info("design %r: checks failed: %s", concept.name, why)
            return
        self.stats["checks_ok"] += 1
        vox = voxels_of_build(build, self.inv)
        for other in self.designs:
            if iou(vox, other["voxels"]) > IOU_MAX:
                self.failures.append((concept.name, f"it has the same shape as {other['meta']['title']!r}"))
                log.info("design %r: IoU with %r above %.1f: dropped", concept.name, other["meta"]["title"], IOU_MAX)
                return
        self.stats["diverse_ok"] += 1
        used = sum(i.dims.long * i.dims.short * i.dims.height / 3 for p in build.parts for i in [self.inv.get(p.part_id, p.colour)])
        extra = {"match_colour": self.match_color, "mixed_colours": not self.match_color,
                 "design": {"concept": concept.name, "sketch": concept.sketch.text(), "scale": [sx, sy], "sketch_cells": filled, "coverage": round(res.coverage, 3), "outline_iou": round(outline, 3), "rescued": rescued, "type": concept.kind, "solver": res.status,
                            "solve_s": round(res.seconds, 2), "colour_wished": res.wish_cells, "colour_met": res.wish_met, "use_share": round(used / max(self.aff.cells, 1e-9), 3), "round": concept.round,
                            "low_use": used < MIN_USE * self.aff.cells}}
        meta = await in_thread(store_design, self.svc, self.inv, build, extra)
        self.designs.append({"meta": meta, "voxels": vox, "concept": concept})
        self.stats["last_design_ms"] = self.ms()
        if self.stats["first_design_ms"] is None:
            self.stats["first_design_ms"] = self.stats["last_design_ms"]
        self.emit("design", build=meta)

    def _check(self, res, concept):
        return build_design(res, concept, self.plan_inv, self.inv, self.match_color)

    # -------------------------------------------------------------- one streamed concept call
    async def concept_round(self, prompt, round_no):
        if self.calls >= MAX_CALLS:
            return
        self.calls += 1
        parser, seen_names = cn.ObjectStream(), {c.name.lower() for c in (d["concept"] for d in self.designs)}
        self.stats["concept_in_chars"] += len(cn.SYSTEM_PROMPT) + len(prompt)
        async for text in self.llm.stream(cn.SYSTEM_PROMPT, prompt, schema=cn.SCHEMA):
            self.stats["concept_out_chars"] += len(text)
            for obj in parser.feed(text):
                self.stats["concepts"] += 1
                c, why = cn.clean(obj, self.aff.limits, round_no, self.aff.colours, self.aff.budget)
                name = (obj.get("name") if isinstance(obj, dict) else None) or "?"
                if c is None:
                    self.stats["fill_rejected"] += "cells" in (why or "") and "sketch fills" in (why or "")
                    self.failures.append((str(name)[:40], why))
                    log.info("concept %r rejected: %s", name, why)
                elif c.key() in self.seen:
                    continue
                elif any(iou(c.sketch.cells, o.sketch.cells) > IOU_MAX for o in self.sketches):          # the same shape twice: the first one is enough
                    twin = next(o for o in self.sketches if iou(c.sketch.cells, o.sketch.cells) > IOU_MAX)
                    self.failures.append((c.name, f"its sketch is the same shape as {twin.name!r}"))
                    self.stats["dup_dropped"] += 1
                elif any(c.bbox == o.bbox for o in self.sketches):                                          # a different bounding box for every concept
                    twin = next(o for o in self.sketches if c.bbox == o.bbox)
                    self.failures.append((c.name, f"its bounding box {c.bbox[0]} x {c.bbox[1]} x {c.bbox[2]} is the same as {twin.name!r}: use a different one"))
                    self.stats["dup_dropped"] += 1
                else:
                    self.seen.add(c.key())
                    self.sketches.append(c)
                    self.stats["sketch_ok"] += 1
                    self.stats["undersized"] += c.undersized
                    self.tasks.append(asyncio.create_task(self.handle(c)))
        for e in parser.errors:
            log.info("concept stream: %s", e)
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    # -------------------------------------------------------------- the batched look check
    async def look_check(self):
        if not self.designs or self.calls >= MAX_CALLS:
            return
        self.calls += 1
        names = [d["meta"]["title"] for d in self.designs]
        pngs = [self.svc.thumb(d["meta"]["id"]) for d in self.designs]
        sheet = vision.contact_sheet(pngs, [str(i + 1) for i in range(len(pngs))], cols=3, cell=(320, 260))
        options = names + [NONE_OF_THESE]
        schema = {"type": "object", "properties": {"answers": {"type": "array", "items": {"type": "object", "properties": {
            "number": {"type": "integer"}, "choice": {"type": "string", "enum": options}, "child_yes": {"type": "boolean"}},
            "required": ["number", "choice", "child_yes"], "additionalProperties": False}}}, "required": ["answers"], "additionalProperties": False}
        listed = "; ".join(f'{i + 1} = "{n}"' for i, n in enumerate(names))
        user = ("The picture shows numbered LEGO models drawn from the front corner (plain bricks, no stickers). Each number has a name: " + listed + ". "
                "For each number answer TWO things. (1) \"choice\": which of these names best says what the model is, or \"none of these\": " +
                "; ".join(f'"{o}"' for o in options) + ". (2) \"child_yes\": could a child see this simple build as the thing named for that number? "
                "(true or false). These are small, simple, playful builds: a few big bricks whose shape and 1 or 2 telling features, together with the name, "
                "tell the story. Accept a reasonable simple representation at this size; refuse only a heap of bricks that suggests nothing of its name.")
        system = ("You judge small LEGO builds made by children and beginners from a handful of bricks. Judge the SHAPE and its telling features, not the "
                  "colours (they come from the pieces the person owns, so a duck need not be yellow) and not the detail.")
        with timing.stage("look_check", designs=len(names)):
            text = await self.look_llm.complete_with_images(system, user, schema, [sheet])
        self.stats["look_in_chars"] += len(system) + len(user)
        self.stats["look_out_chars"] += len(text)
        answers = {a["number"]: a for a in json.loads(text).get("answers", []) if isinstance(a, dict) and isinstance(a.get("number"), int)}
        keep = []
        for i, d in enumerate(self.designs):
            ans = answers.get(i + 1) or {}
            choice, child = ans.get("choice"), ans.get("child_yes")
            ok_choice = choice is None or choice == names[i]            # a number the model skipped is not held against the design
            ok_child = child is None or bool(child)
            ok = ok_child if LOOK_RULE == "child" else ok_choice
            self.stats["look_checked"] += 1
            self.stats["look_ok"] += ok
            self.stats["look_ok_choice"] += ok_choice
            self.stats["look_ok_child"] += ok_child
            self.emit("verdict", id=d["meta"]["id"], title=names[i], ok=ok, choice=choice, child_yes=child, ok_choice=ok_choice, ok_child=ok_child)
            if not ok:
                log.info("look check: %r removed (child_yes=%s, looked like %r)", names[i], child, choice)
            else:
                keep.append(d)
        self.designs = keep

    # -------------------------------------------------------------- the whole request
    async def orchestrate(self):
        try:
            self.emit("start", pieces=self.aff.pieces, kinds=len(self.aff.rows), cells=round(self.aff.cells, 1), limits=self.aff.limits, match_colour=self.match_color)
            seed = int(hashlib.sha1(json.dumps(self.aff.rows, sort_keys=True).encode()).hexdigest()[:8], 16)
            examples = self.bank.pick(self.aff.cells, self.aff.mix, EXAMPLES_IN_PROMPT, seed) if self.bank is not None else []
            self.stats["examples"] = [f"{e['set']} {e['title']}" for e in examples]
            log.info("examples used: %s", self.stats["examples"])
            try:
                with timing.stage("designs.round1"):
                    await self.concept_round(cn.build_prompt(self.aff, self.match_color, examples), 1)
            except Exception as e:                                     # noqa: BLE001  Gemini down or refused: say so, the page keeps its other sections
                log.warning("concept call failed: %s: %s", type(e).__name__, e)
                self.emit("note", kind="ai_unavailable", message=f"AI designs are unavailable right now ({getattr(e, 'status', '')} {e}).".replace("( ", "("))
                return
            if (len(self.designs) < ENOUGH or self.stats["dup_dropped"] >= 2) and self.failures and self.calls < MAX_CALLS - 1:
                try:
                    with timing.stage("designs.repair"):
                        await self.concept_round(cn.repair_prompt(self.aff, self.match_color, self.failures[:8], [d["concept"] for d in self.designs]), 2)
                except Exception as e:                                 # noqa: BLE001
                    log.warning("repair call failed: %s: %s", type(e).__name__, e)
            try:
                await self.look_check()
            except Exception as e:                                     # noqa: BLE001  the check failing is not the design's fault: keep them all
                log.warning("look check failed: %s: %s", type(e).__name__, e)
            if not self.designs and self.aff.pieces < SMALL_PILE:
                self.emit("note", kind="small_pile", message=f"With only {self.aff.pieces} pieces there is not enough to build a recognisable model yet. "
                          f"Add a few more pieces (8 to 10 or more works best) and try again.", reasons=[{"concept": n, "why": w} for n, w in self.failures[:8]])
            elif not self.designs:
                self.emit("note", kind="no_design", message="No design fit this pile this time.", reasons=[{"concept": n, "why": w} for n, w in self.failures[:8]])
        finally:
            self.emit("done", designs=len(self.designs), calls=self.calls, stats=self.stats, failures=[{"concept": n, "why": w} for n, w in self.failures[:12]])
            self.queue.put_nowait(None)


async def stream_designs(svc, parts, llm, match_color, bank=None, trace_name="designs_stream", look_llm=None):
    """The SSE text of a request, event by event. `parts`: [{part_num, name?, color?, qty}]."""
    trace, token = timing.begin(trace_name, match_colour=match_color)
    run = None
    task = None
    try:
        with timing.stage("designs.inventory"):
            for p in parts:
                await svc.parts.ldraw_id_async(str(p.get("part_num") or ""))
            inv, skipped = aibuild.build_inventory(parts, svc.parts, svc.colours)
        if not inv.parts():
            yield sse("note", {"kind": "no_parts", "message": "None of the pieces are plain bricks, plates, tiles, slopes or round parts the solver can place.",
                               "skipped": [{"part": p, "reason": r} for p, r in skipped][:10]})
            yield sse("done", {"designs": 0, "calls": 0, "stats": {}, "failures": []})
            return
        aff = summarize(inv)
        if llm is None:
            yield sse("note", {"kind": "ai_unavailable", "message": "AI designs need a GEMINI_API_KEY."})
            yield sse("done", {"designs": 0, "calls": 0, "stats": {}, "failures": []})
            return
        run = Run(svc, inv, inv, aff, llm, match_color, bank, look_llm)
        task = asyncio.create_task(run.orchestrate())
        while True:
            item = await run.queue.get()
            if item is None:
                break
            yield sse(*item)
    finally:
        if task is not None and not task.done():
            task.cancel()
        if run is not None:
            for t in run.tasks:
                t.cancel()
        row = timing.end(trace, token, designs=len(run.designs) if run else 0, **({"first_design_ms": run.stats["first_design_ms"], "last_design_ms": run.stats["last_design_ms"]} if run else {}))
