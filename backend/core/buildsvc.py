"""Builds for the 3D viewer, official and AI, in ONE format: an LDraw model with `0 STEP` lines, plus metadata.

    create_ai_build(parts, idea, llm)   AI build on the user's pieces (aibuild.py), validated, converted, stored
    official_build(set_num, set_name)   the set's model from the LDraw OMR (omr.py), stored; None if there is none / not usable
    meta(id)  /  mpd(id, lite)          what the API serves: metadata JSON and the packed .mpd (only the files the model needs)

Everything is cached on disk under BUILD_CACHE_DIR/builds/<id>.{ldr,mpd,lite.mpd,json}; the id of an official build is
"set-<set number>", of an AI build "ai-<hash of the model>".
"""

import hashlib
import json
import logging
import re
from pathlib import Path

from backend.core import aibuild, ldrconv, omr
from backend.core.buildmodel import Build
from backend.core.inventory_check import can_build, pool_inventory, recolour
from backend.core.ldraw_lib import pack_mpd

log = logging.getLogger("lego.builds")

_ID = re.compile(r"^(ai-[0-9a-f]{12}|set-[0-9a-z]+(-[0-9]+)?|sub-[0-9a-z]+-[0-9]+-[0-9]+)$", re.I)
SUB_ID = re.compile(r"^sub-([0-9a-z]+-[0-9]+)-([0-9]+)$", re.I)


def valid_id(build_id):
    return bool(_ID.match(build_id or ""))


class BuildService:
    def __init__(self, library, part_mapper, colour_mapper, cache_dir, omr_base_url, rb_names=None):
        self.lib, self.parts, self.colours = library, part_mapper, colour_mapper
        self.dir = Path(cache_dir) / "builds"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.omr_base = omr_base_url
        self.cache_dir = Path(cache_dir)
        self._canon = {}

    # ---- storage
    def _path(self, build_id, ext):
        return self.dir / f"{build_id}.{ext}"

    def meta(self, build_id):
        p = self._path(build_id, "json")
        return json.loads(p.read_text(encoding="utf-8")) if valid_id(build_id) and p.exists() else None

    def mpd(self, build_id, lite=False):
        p = self._path(build_id, "lite.mpd" if lite else "mpd")
        return p.read_text(encoding="utf-8") if valid_id(build_id) and p.exists() else None

    def thumb(self, build_id):
        """The cached isometric thumbnail (PNG bytes) of a build, or None."""
        p = self._path(build_id, "png")
        return p.read_bytes() if valid_id(build_id) and p.exists() else None

    def save_thumb(self, build_id, png):
        if valid_id(build_id):
            self._path(build_id, "png").write_bytes(png)

    def _store(self, build_id, ldr, meta):
        """Write the model, its packed variants and the metadata; returns the metadata (with step list and part counts)."""
        packed = pack_mpd(ldr, self.lib)
        missing = list(pack_mpd.last_missing)
        lite = pack_mpd(ldr, self.lib, drop_conditional=True)
        steps = self.steps_meta(ldr)
        meta = {**meta, "id": build_id, "steps": steps, "step_count": len(steps),
                "part_count": sum(p["qty"] for s in steps for p in s["parts"]),
                "model_url": f"/api/builds/{build_id}/model.mpd", "lite_model_url": f"/api/builds/{build_id}/model.mpd?lite=1",
                "missing_files": missing[:20]}
        if missing:
            log.warning("build %s: %d referenced files are not in the LDraw library: %s", build_id, len(missing), missing[:5])
        self._path(build_id, "ldr").write_text(ldr, encoding="utf-8")
        self._path(build_id, "mpd").write_text(packed, encoding="utf-8")
        self._path(build_id, "lite.mpd").write_text(lite, encoding="utf-8")
        self._path(build_id, "json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return meta

    # ---- step metadata (what each step adds), numbered like the viewer numbers them
    def steps_meta(self, ldr):
        uses = ldrconv.ldr_steps(ldr)
        by_step = {}
        for u in uses:
            by_step.setdefault(u.step, {}).setdefault((u.ld_part, u.colour), 0)
            by_step[u.step][(u.ld_part, u.colour)] += 1
        out = []
        for step in range(max(by_step) + 1 if by_step else 0):
            parts = []
            for (ld_part, colour), qty in sorted(by_step.get(step, {}).items()):
                code = int(colour) if colour.isdigit() else None
                if code is not None:
                    hexv, cname = self.colours.hex(code), self.colours.name(code)
                else:
                    hexv, cname = self._direct_hex(colour), "Custom colour"
                desc = self.lib.part_description(ld_part) or ld_part
                parts.append({"ld_part": ld_part, "name": re.sub(r"\s+", " ", desc.lstrip("~=_")).strip(), "colour_code": colour,
                              "colour_name": cname, "hex": hexv, "qty": qty})
            out.append({"index": step, "parts": parts})
        return out

    @staticmethod
    def _direct_hex(colour):
        m = re.match(r"^0x2([0-9a-fA-F]{6})$", colour)
        return "#" + m.group(1).upper() if m else "#808080"

    # ---- AI
    async def create_ai_build(self, parts, idea, llm, match_color=True):
        for p in parts:                                  # parts the local rules cannot map: ask Rebrickable for the LDraw id (cached)
            await self.parts.ldraw_id_async(str(p.get("part_num") or ""))
        inv, skipped = aibuild.build_inventory(parts, self.parts, self.colours)
        from backend import timing
        with timing.stage("aibuild.generate"):
            result = await aibuild.generate_build(inv if match_color else pool_inventory(inv), idea, llm)
        build = result.build
        if not match_color:                              # planned from part counts; now the real colours
            build = Build(build.title, build.description, recolour(build.parts, {}, inv))
        if not (can_build(build.parts, inv, match_color) and can_build(build.parts, inv, True)):
            raise aibuild.BuildError("That design needs pieces that are not in your inventory, so it was not shown. Try again.", 422)
        ldr = ldrconv.build_to_ldr(build, inv)
        build_id = "ai-" + hashlib.sha1(ldr.encode("utf-8")).hexdigest()[:12]
        meta = {
            "kind": "ai", "title": build.title, "description": build.description, "attribution": None,
            "repairs": result.repairs[:30], "warnings": result.warnings,
            "repair_summary": {"dropped": result.dropped, "swapped": result.swapped, "nudged": result.nudged,
                               "steps_regenerated": result.steps_regenerated, "retried": result.retried,
                               "asked_for": result.original_count},
            "skipped_parts": [{"part": p, "reason": r} for p, r in skipped],
            "match_colour": match_color, "mixed_colours": not match_color,
        }
        return self._store(build_id, ldr, meta)

    def canon(self, part):
        """The current LDraw number of a part: LDraw leaves a `~Moved to 3139b` stub under the old number (3139), OMR models and our mapper use either."""
        key = str(part).strip().lower()
        hit = self._canon.get(key)
        if hit is not None:
            return hit
        out = key
        for _ in range(3):
            m = re.match(r"^~Moved to (\S+)", self.lib.part_description(out) or "")
            if not m:
                break
            out = m.group(1).lower().removesuffix(".dat")
        self._canon[key] = out
        return out

    # ---- a sub-model (or small model) of an OMR file, found by omr_index
    def official_submodel(self, sub_id, title):
        """Metadata of the viewer build for "sub-<set>-<block>", or (None, reason). The model comes from the cached OMR file."""
        from backend.core import omr_index
        m = SUB_ID.match(sub_id or "")
        if not m:
            return None, "none"
        existing = self.meta(sub_id.lower())
        if existing:
            return existing, None
        set_num, block = m.group(1), int(m.group(2))
        path = self.cache_dir / "omr" / f"{set_num}.mpd"
        if not omr.valid_set_num(set_num) or not path.exists():
            return None, "none"
        text = path.read_text(encoding="utf-8", errors="replace")
        authors, licences = omr.parse_header(text)
        if not omr.OmrModel(set_num, text, authors, licences).licence_ok:
            return None, "licence"
        model = omr_index.extract_model(text, block)
        ldr = model if ldrconv.has_steps(model) else ldrconv.add_steps(model)
        meta = {
            "kind": "set", "title": title or f"Part of LEGO set {set_num}", "description": "", "set_num": set_num,
            "attribution": {"authors": authors, "licence": licences, "source_url": f"https://library.ldraw.org/omr/sets?search={set_num.split('-')[0]}",
                            "text": "Model from the LDraw Official Model Repository" + (f", by {', '.join(authors)}" if authors else "") + "."},
            "steps_generated": not ldrconv.has_steps(model), "repairs": [], "warnings": [], "skipped_parts": [],
        }
        return self._store(sub_id.lower(), ldr, meta), None

    # ---- official
    async def official_build(self, set_num, set_name=None):
        """Metadata of the set's viewer model, or (None, reason) when there is none ('none' / 'licence')."""
        build_id = f"set-{set_num.lower()}"
        existing = self.meta(build_id)
        if existing:
            return existing, None
        model = await omr.fetch_set_model(set_num, self.cache_dir, self.omr_base)
        if model is None:
            return None, "none"
        if not model.licence_ok:
            return None, "licence"
        ldr = model.text if ldrconv.has_steps(model.text) else ldrconv.add_steps(model.text)
        generated = not ldrconv.has_steps(model.text)
        meta = {
            "kind": "set", "title": set_name or f"LEGO set {set_num}", "description": "", "set_num": set_num,
            "attribution": {"authors": model.authors, "licence": model.licences, "source_url": model.source_url,
                            "text": "Model from the LDraw Official Model Repository" +
                                    (f", by {', '.join(model.authors)}" if model.authors else "") + "."},
            "steps_generated": generated, "repairs": [], "warnings": [], "skipped_parts": [],
        }
        return self._store(build_id, ldr, meta), None
