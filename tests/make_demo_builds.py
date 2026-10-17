"""Regenerate the two demo builds the viewer renders in tests and screenshots (frontend/public/demo/):

    python -m tests.make_demo_builds

ai-build:       tests/data/ai_build.json (a Gemini-style answer) validated, converted and packed like a real AI build
official-30051: the LDraw OMR model of set 30051-1 (X-wing Fighter, Mini) by Jude Parrill, CCAL 2.0, as a real official build
Needs the LDraw library (python -m backend.core.fetch_ldraw).
"""

import asyncio
import json
import shutil
from pathlib import Path

from backend import config
from backend.core import omr
from backend.core.buildsvc import BuildService
from backend.core.ldmap import ColourMapper, PartMapper
from backend.core.ldraw_lib import get_library
from tests.build_fixtures import inventory

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "public" / "demo"
DATA = Path(__file__).resolve().parent / "data"


class _Llm:
    async def complete(self, system, user, schema):
        return (DATA / "ai_build.json").read_text(encoding="utf-8")


async def main():
    lib = get_library()
    inv = inventory()
    cache = ROOT / "build_cache" / "demo"
    shutil.rmtree(cache, ignore_errors=True)
    svc = BuildService(lib, PartMapper(lib, {i.part_id.lower(): i.part_id for i in inv.parts()}), ColourMapper(lib), cache, "https://x")
    parts = [{"part_num": i.part_id, "name": i.name, "color": i.colour, "qty": i.qty} for i in inv.parts()]
    ai = await svc.create_ai_build(parts, {"name": "Little Red Truck", "description": "a small truck"}, _Llm())

    async def fetch(set_num, cache_dir, base_url, client=None):
        text = (DATA / "omr_30051-1.mpd").read_text(encoding="utf-8")
        authors, licences = omr.parse_header(text)
        return omr.OmrModel(set_num, text, authors, licences, "https://library.ldraw.org/omr/sets?search=30051")
    omr.fetch_set_model = fetch
    official, _ = await svc.official_build("30051-1", "X-wing Fighter - Mini")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, meta in (("ai-build", ai), ("official-30051", official)):
        meta = {**meta, "model_url": f"/demo/{name}.mpd", "lite_model_url": f"/demo/{name}.lite.mpd"}
        (OUT / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / f"{name}.mpd").write_text(svc.mpd(meta["id"]), encoding="utf-8")
        (OUT / f"{name}.lite.mpd").write_text(svc.mpd(meta["id"], lite=True), encoding="utf-8")
    (OUT / "LDConfig.ldr").write_text(lib.read("LDConfig.ldr"), encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    config.BUILD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())
