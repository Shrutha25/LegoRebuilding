import asyncio
import json
from pathlib import Path

import pytest

from backend import builds_api, config
from backend.core import aibuild, ldrconv, omr
from backend.core import buildmodel as bm
from backend.core.buildsvc import BuildService
from backend.core.ldmap import ColourMapper, PartMapper
from backend.core.ldraw_lib import LDrawLibrary, normalise_ref, pack_mpd
from tests.build_fixtures import P, fake_library_zip, inventory, sample_ai_build

DATA = Path(__file__).resolve().parent / "data"
INV = inventory()


@pytest.fixture
def lib(tmp_path):
    return LDrawLibrary(fake_library_zip(tmp_path / "complete.zip"))


# ---------------------------------------------------------------- LDraw library and packing
def test_reference_names_are_normalised_like_threejs():
    assert normalise_ref("s\\3001s01.dat") == "parts/s/3001s01.dat" and normalise_ref("48\\x.dat") == "p/48/x.dat"
    assert normalise_ref("Stud.dat ") == "Stud.dat"


def test_library_lookup_colours_and_descriptions(lib):
    assert lib.part_path("3001") == "parts/3001.dat" and lib.part_path("9999") is None
    assert lib.find("s\\3001s01.dat") == "parts/s/3001s01.dat" and lib.find("STUD.DAT") == "p/stud.dat"
    assert lib.part_description("3003") == "Brick 2 x 2"
    c = lib.colours()
    assert c[4].name == "Red" and c[4].hex == "#B40000" and c[47].transparent and c[71].name == "Light_Bluish_Grey"


def test_pack_embeds_only_what_the_model_uses_under_the_names_threejs_looks_up(lib):
    ldr = ldrconv.build_to_ldr(sample_ai_build(), INV)
    mpd = lib.pack(ldr)
    names = [ln[7:] for ln in mpd.splitlines() if ln.startswith("0 FILE ")]
    assert names[0] == "model.ldr"
    assert set(names[1:]) == {"3001.dat", "3003.dat", "3005.dat", "3020.dat", "3022.dat", "3068b.dat", "stud.dat", "parts/s/3001s01.dat"}
    assert pack_mpd.last_missing == [] and mpd.count("0 STEP") == 8
    lite = lib.pack(ldr + "5 24 0 0 0 1 0 0 0 0 1 0 0 0\n", drop_conditional=True)
    assert "\n5 24" not in lite


def test_pack_reports_files_missing_from_the_library_and_keeps_submodels(lib):
    mpd = "0 FILE main.ldr\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 Sub Model.ldr\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 99999.dat\n0 FILE Sub Model.ldr\n1 4 0 0 0 1 0 0 0 1 0 0 0 1 3003.dat\n"
    out = lib.pack(mpd)
    assert pack_mpd.last_missing == ["99999.dat"]
    assert "0 FILE Sub Model.ldr" in out and "0 FILE 3003.dat" in out


# ---------------------------------------------------------------- mapping
def test_parts_map_through_variants_and_unmapped_are_logged(lib, caplog):
    pm = PartMapper(lib, known={"3001": "3001", "3003": "3003"}, rel_map={"3003pr0001": "3003"})
    assert pm.ldraw_id("3001") == "3001" and pm.ldraw_id("3001pb0002") == "3001"
    assert pm.ldraw_id("3003pr0001") == "3003"                       # via the Rebrickable parent
    with caplog.at_level("WARNING", logger="lego.ldmap"):
        assert pm.ldraw_id("7777777") is None
    assert "7777777" in caplog.text and "7777777" in pm.unmapped
    assert pm.coverage(["3001", "3003", "8888"]) == (2, 3)


def test_part_mapping_is_cached_on_disk(lib, tmp_path):
    path = tmp_path / "ids.json"
    PartMapper(lib, {"3001": "3001"}, cache_path=path).ldraw_id("3001")
    assert json.loads(path.read_text())["3001"] == "3001"
    assert PartMapper(lib, cache_path=path).ldraw_id("3001") == "3001"      # served from the cache without any lookup tables


def test_colours_map_by_name_id_and_nearest_rgb(lib, caplog):
    cm = ColourMapper(lib, [("71", "Light Bluish Gray", "A0A5A9"), ("9999", "Fancy Silver", "970000")])
    assert cm.code("Light Bluish Gray") == 71 and cm.code("light bluish grey") == 71 and cm.code("Medium Azure") == 322
    assert cm.code("4") == 4 and cm.code(1) == 1
    assert cm.code("Fancy Silver") == 4                                       # not in LDraw: nearest by RGB (the red)
    with caplog.at_level("WARNING", logger="lego.ldmap"):
        assert cm.code("Unobtainium") is None
    assert "Unobtainium" in caplog.text
    assert cm.hex(4) == "#B40000" and cm.name(71) == "Light Bluish Grey"


# ---------------------------------------------------------------- the AI build
class FakeLLM:
    def __init__(self, *answers):
        self.answers, self.prompts = list(answers), []

    async def complete(self, system, user, schema):
        self.prompts.append(user)
        out = self.answers.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def fixture_json():
    return (DATA / "ai_build.json").read_text(encoding="utf-8")


def parts_for(inv=INV):
    return [{"part_num": i.part_id, "name": i.name, "color": i.colour, "qty": i.qty} for i in inv.parts()]


def run(coro):
    return asyncio.run(coro)


def test_a_good_ai_build_needs_no_repair_and_one_call():
    llm = FakeLLM(fixture_json())
    res = run(aibuild.generate_build(INV, {"name": "Truck", "description": "a red truck"}, llm))
    assert len(llm.prompts) == 1 and not res.repairs and not res.retried and not res.steps_regenerated
    assert len(res.build.parts) == 12 and "Build this: Truck" in llm.prompts[0] and "3001 Brick 2 x 4 (Red): 4x2 studs, 3 plates tall" in llm.prompts[0]


def test_a_broken_build_is_repaired_without_a_second_call_when_little_is_lost():
    data = json.loads(fixture_json())
    data["build"]["parts"][4]["x"] = 5.0                         # a float coordinate: snapped
    data["build"]["parts"][1]["x"] = 3                            # overlaps the first brick: nudged
    data["build"]["parts"].append({"part_id": "3001", "colour": "Green", "x": 20, "z": 0, "y": 0, "rotation": 0, "step": 9})   # swapped
    llm = FakeLLM(json.dumps(data))
    res = run(aibuild.generate_build(INV, None, llm))
    assert len(llm.prompts) == 1 and res.nudged >= 1 and res.swapped == 1 and bm.validate(res.build, INV).ok


def test_when_too_much_is_dropped_gemini_is_asked_once_more_with_the_problems_listed():
    bad = json.dumps({"build": {"title": "Bad", "description": "", "parts": [
        {"part_id": "9999", "colour": "Red", "x": 0, "z": 0, "y": 0, "rotation": 0, "step": 1},
        {"part_id": "8888", "colour": "Red", "x": 4, "z": 0, "y": 0, "rotation": 0, "step": 1},
        {"part_id": "3001", "colour": "Red", "x": 8, "z": 0, "y": 0, "rotation": 0, "step": 1}]}})
    llm = FakeLLM(bad, fixture_json())
    res = run(aibuild.generate_build(INV, None, llm))
    assert len(llm.prompts) == 2 and res.retried and len(res.build.parts) == 12
    assert "previous attempt had these problems" in llm.prompts[1] and "9999" in llm.prompts[1]


def test_the_retry_happens_at_most_once_and_a_failed_retry_keeps_the_first_result():
    bad = json.dumps({"build": {"title": "Bad", "description": "", "parts": [
        {"part_id": "9999", "colour": "Red", "x": 0, "z": 0, "y": 0, "rotation": 0, "step": 1},
        {"part_id": "3001", "colour": "Red", "x": 8, "z": 0, "y": 0, "rotation": 0, "step": 1}]}})
    llm = FakeLLM(bad, bad, bad)
    res = run(aibuild.generate_build(INV, None, llm))
    assert len(llm.prompts) == 2 and len(res.build.parts) == 1
    llm2 = FakeLLM(bad, RuntimeError("quota"))
    assert len(run(aibuild.generate_build(INV, None, llm2)).build.parts) == 1


def test_invalid_gemini_steps_are_regenerated_locally():
    data = json.loads(fixture_json())
    for p in data["build"]["parts"]:
        p["step"] = 1 if p["y"] > 0 else 5                       # the top before the bottom
    res = run(aibuild.generate_build(INV, None, FakeLLM(json.dumps(data))))
    assert res.steps_regenerated and not bm.check_steps(res.build.parts, INV)
    assert sorted({p.step for p in res.build.parts}) == list(range(1, len({p.step for p in res.build.parts}) + 1))


def test_unusable_answers_become_clear_errors():
    for text in ("not json", json.dumps({"build": {"title": "x", "description": "", "parts": []}}), json.dumps({"nope": 1})):
        with pytest.raises(aibuild.BuildError):
            run(aibuild.generate_build(INV, None, FakeLLM(text)))
    with pytest.raises(aibuild.BuildError) as e:
        run(aibuild.generate_build(bm.Inventory([]), None, FakeLLM("{}")))
    assert e.value.status == 400


def test_the_inventory_keeps_only_plain_bricks_plates_and_tiles_with_ldraw_files(lib):
    pm = PartMapper(lib, {"3001": "3001", "3003": "3003"})
    cm = ColourMapper(lib)
    inv, skipped = aibuild.build_inventory([
        {"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 2}, {"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 1},
        {"part_num": "3003", "name": "Brick 2 x 2", "color": "Unobtainium", "qty": 1},
        {"part_num": "11211", "name": "Brick, Modified 1 x 2 with Studs on 1 Side", "color": "Red", "qty": 1}], pm, cm)
    assert [(i.part_id, i.qty, i.ld_colour) for i in inv.parts()] == [("3001", 3, 4)]
    assert {s[0] for s in skipped} == {"3003", "11211"}


# ---------------------------------------------------------------- the build service (cache, metadata, official models)
def service(lib, tmp_path):
    pm = PartMapper(lib, {i.part_id.lower(): i.part_id for i in INV.parts()})
    return BuildService(lib, pm, ColourMapper(lib), tmp_path / "cache", "https://example.invalid/omr")


def test_an_ai_build_is_stored_with_steps_parts_and_a_packed_model(lib, tmp_path):
    svc = service(lib, tmp_path)
    meta = run(svc.create_ai_build(parts_for(), {"name": "Truck"}, FakeLLM(fixture_json())))
    assert meta["id"].startswith("ai-") and meta["step_count"] == 8 and meta["part_count"] == 12 and meta["kind"] == "ai"
    step1 = meta["steps"][0]["parts"]
    assert step1 == [{"ld_part": "3001", "name": "Brick 2 x 4", "colour_code": "4", "colour_name": "Red", "hex": "#B40000", "qty": 2}]
    assert svc.mpd(meta["id"]).startswith("0 FILE model.ldr") and svc.mpd(meta["id"], lite=True) is not None
    assert svc.meta(meta["id"])["title"] == "Little Red Truck" and svc.meta("ai-000000000000") is None
    again = run(svc.create_ai_build(parts_for(), None, FakeLLM(fixture_json())))
    assert again["id"] == meta["id"]                                         # same model -> same id -> served from the cache


def test_ids_are_validated_so_nothing_outside_the_cache_can_be_read(lib, tmp_path):
    svc = service(lib, tmp_path)
    for bad in ("../x", "ai-../../etc", "set-..", "", "AI-123"):
        assert svc.meta(bad) is None and svc.mpd(bad) is None


def fake_omr(monkeypatch, text=None, licence="Redistributable under CCAL version 2.0 : see CAreadme.txt"):
    text = text or (DATA / "omr_30051-1.mpd").read_text(encoding="utf-8")
    if licence is not None:
        text = text.replace("Redistributable under CCAL version 2.0 : see CAreadme.txt", licence)
    authors, licences = omr.parse_header(text)

    async def fetch(set_num, cache_dir, base_url, client=None):
        return omr.OmrModel(set_num, text, authors, licences, "https://library.ldraw.org/omr/sets?search=30051")
    monkeypatch.setattr(omr, "fetch_set_model", fetch)


def test_an_official_model_keeps_its_steps_and_shows_attribution(lib, tmp_path, monkeypatch):
    fake_omr(monkeypatch)
    meta, reason = run(service(lib, tmp_path).official_build("30051-1", "X-wing Fighter - Mini"))
    assert reason is None and meta["id"] == "set-30051-1" and meta["step_count"] == 28 and not meta["steps_generated"]
    assert meta["attribution"]["authors"] == ["Jude Parrill"] and "CCAL" in meta["attribution"]["licence"][0]
    assert "Jude Parrill" in meta["attribution"]["text"] and meta["part_count"] == 61


def test_an_official_model_without_a_recognised_licence_is_not_used(lib, tmp_path, monkeypatch):
    fake_omr(monkeypatch, licence="Not redistributable: all rights reserved")
    assert run(service(lib, tmp_path).official_build("30051-1")) == (None, "licence")
    fake_omr(monkeypatch, licence=None)
    text = (DATA / "omr_30051-1.mpd").read_text(encoding="utf-8").replace("0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt\n", "")
    fake_omr(monkeypatch, text=text, licence=None)
    assert run(service(lib, tmp_path / "again").official_build("30051-1")) == (None, "licence")


def test_an_official_model_without_steps_gets_generated_steps(lib, tmp_path, monkeypatch):
    stepless = "\n".join(["0 FILE x.ldr", "0 Author: A. Builder", "0 !LICENSE Licensed under CC BY 4.0 : see CAreadme.txt",
                          "1 4 0 -24 0 1 0 0 0 1 0 0 0 1 3001.dat", "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat", "1 4 0 -48 0 1 0 0 0 1 0 0 0 1 3003.dat"])
    fake_omr(monkeypatch, text=stepless, licence=None)
    meta, _ = run(service(lib, tmp_path).official_build("12345-1"))
    assert meta["steps_generated"] and meta["step_count"] >= 2
    assert [p["ld_part"] for p in meta["steps"][0]["parts"]] == ["3001"]               # the lowest part first


def test_no_model_for_a_set_is_reported(lib, tmp_path, monkeypatch):
    async def none(*a, **k):
        return None
    monkeypatch.setattr(omr, "fetch_set_model", none)
    assert run(service(lib, tmp_path).official_build("1-1")) == (None, "none")


# ---------------------------------------------------------------- OMR download, cache and licence
def test_omr_header_parsing_and_licence_rules():
    text = "0 FILE a.ldr\n0 Author: A [a]\n0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt\n0 FILE b.ldr\n0 Author: B\n0 Author: A [a]\n"
    authors, licences = omr.parse_header(text)
    assert authors == ["A [a]", "B"] and omr.OmrModel("1-1", text, authors, licences).licence_ok
    assert omr.OmrModel("1-1", "", ["A"], ["Licensed under CC BY 4.0"]).licence_ok
    assert not omr.OmrModel("1-1", "", ["A"], []).licence_ok and not omr.OmrModel("1-1", "", ["A"], ["All rights reserved"]).licence_ok
    assert omr.valid_set_num("30051-1") and not omr.valid_set_num("../x") and not omr.valid_set_num("a/b")


def test_omr_download_is_cached_and_a_missing_model_is_remembered(tmp_path):
    import httpx
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path.endswith("30051-1.mpd"):
            return httpx.Response(200, content=(DATA / "omr_30051-1.mpd").read_bytes())
        return httpx.Response(404)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            a = await omr.fetch_set_model("30051-1", tmp_path, "https://x/omr", client)
            b = await omr.fetch_set_model("30051-1", tmp_path, "https://x/omr", client)
            c = await omr.fetch_set_model("999-1", tmp_path, "https://x/omr", client)
            d = await omr.fetch_set_model("999-1", tmp_path, "https://x/omr", client)
            e = await omr.fetch_set_model("../etc/passwd", tmp_path, "https://x/omr", client)
            return a, b, c, d, e
    a, b, c, d, e = run(go())
    assert a.licence_ok and a.authors == ["Jude Parrill"] and b.text == a.text and c is None and d is None and e is None
    assert len(calls) == 2                                                    # one download, one 404; everything else from the cache


# ---------------------------------------------------------------- the HTTP API
@pytest.fixture
def client(lib, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main
    from backend.core.index import DataStore
    from tests.fixtures import write_rb_dir
    store = DataStore(str(write_rb_dir(tmp_path / "rb")))
    store._load()
    monkeypatch.setattr(main, "store", store)
    builds_api._deps["store"] = store
    monkeypatch.setattr(store, "start", lambda: None)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "up")
    svc = service(lib, tmp_path)
    monkeypatch.setattr(builds_api, "_service", svc)
    llm = FakeLLM(fixture_json(), fixture_json())
    monkeypatch.setitem(builds_api._deps, "get_llm", lambda: llm)
    with TestClient(main.app) as c:
        c.llm = llm
        yield c


def test_api_ai_build_then_fetch_meta_and_model(client):
    body = {"parts": [{"part_num": i.part_id, "name": i.name, "color": i.colour, "qty": i.qty} for i in INV.parts()],
            "idea": {"name": "Truck", "description": "red"}}
    r = client.post("/api/builds/ai", json=body)
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["step_count"] == 8 and meta["model_url"] == f"/api/builds/{meta['id']}/model.mpd"
    assert client.get(f"/api/builds/{meta['id']}").json()["title"] == "Little Red Truck"
    m = client.get(meta["model_url"])
    assert m.status_code == 200 and m.text.startswith("0 FILE model.ldr") and "immutable" in m.headers["cache-control"]
    assert "0 FILE stud.dat" in client.get(meta["lite_model_url"]).text
    assert client.get("/api/ldraw/LDConfig.ldr").text.startswith("0 LDraw.org Configuration File")
    assert client.get("/api/builds/ai-000000000000").status_code == 404 and client.get("/api/builds/..%2Fx").status_code in (404, 422)


def test_api_ai_build_errors_are_clear(client):
    client.llm.answers[:] = ["not json"]
    r = client.post("/api/builds/ai", json={"parts": [{"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 2}]})
    assert r.status_code == 502 and "not valid JSON" in r.json()["detail"]
    assert client.post("/api/builds/ai", json={"parts": []}).status_code == 422
    r = client.post("/api/builds/ai", json={"parts": [{"part_num": "11211", "name": "Brick, Modified 1 x 2 with Studs on 1 Side", "color": "Red", "qty": 2}]})
    assert r.status_code == 400 and "plain bricks" in r.json()["detail"]


def test_api_set_build_with_a_model_and_the_parts_list(client, monkeypatch):
    fake_omr(monkeypatch)
    r = client.post("/api/sets/10-1/build", json={"parts": [{"part_num": "3001", "color": "Red", "qty": 1}], "set_name": "Small House"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["available"] is False or d["build"]["id"] == "set-10-1"
    assert d["instructions_url"] == "https://www.lego.com/service/buildinginstructions/10"
    assert d["inventory"] and d["need_total"] == sum(row["need"] for row in d["inventory"])
    row = next(r_ for r_ in d["inventory"] if r_["part"] == "3001")
    assert row["have"] == 1 and row["missing"] == row["need"] - 1


def test_api_set_build_without_a_model_still_gives_parts_and_the_official_link(client, monkeypatch):
    async def none(*a, **k):
        return None
    monkeypatch.setattr(omr, "fetch_set_model", none)
    d = client.post("/api/sets/10-1/build", json={"parts": []}).json()
    assert d["available"] is False and d["reason"] == "none" and d["build"] is None
    assert d["instructions_url"].startswith("https://www.lego.com/service/buildinginstructions/") and d["inventory"]


def test_api_reports_a_missing_ldraw_library(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main
    monkeypatch.setattr(config, "LDRAW_DIR", tmp_path / "nothing")
    builds_api.reset_service()
    with TestClient(main.app) as c:
        assert c.get("/api/builds-status").json()["ldraw_library"] is False
        r = c.get("/api/builds/ai-000000000000")
        assert r.status_code == 503 and "fetch_ldraw" in r.json()["detail"]
    builds_api.reset_service()


# ---------------------------------------------------------------- against the real library (skipped without it)
def real_library():
    try:
        from backend.core.ldraw_lib import get_library
        return get_library()
    except Exception:                                                             # noqa: BLE001
        pytest.skip("LDraw library not downloaded (python -m backend.core.fetch_ldraw)")


def test_real_library_packs_the_official_sample_and_the_ai_sample_with_nothing_missing():
    lib = real_library()
    for text in ((DATA / "omr_30051-1.mpd").read_text(encoding="utf-8"), ldrconv.build_to_ldr(sample_ai_build(), INV)):
        mpd = lib.pack(text)
        assert pack_mpd.last_missing == [] and len(mpd) < 400_000
        assert ldrconv.has_steps(mpd)


def test_parts_the_local_rules_cannot_map_get_their_ldraw_id_from_rebrickable_once(lib, tmp_path):
    asked = []

    class FakeApi:
        enabled = True

        async def _get(self, path, params=None):
            asked.append(path)
            return {"external_ids": {"LDraw": ["9999999", "3003"]}} if "/parts/odd-3003/" in path else None
    pm = PartMapper(lib, {"odd-3003": "odd-3003"}, api=FakeApi(), cache_path=tmp_path / "ids.json")
    assert pm.ldraw_id("odd-3003") is None
    assert run(pm.ldraw_id_async("odd-3003")) == "3003"                    # the first external id that exists in the library
    assert pm.ldraw_id("odd-3003") == "3003" and "odd-3003" not in pm.unmapped      # remembered, also on disk
    assert PartMapper(lib, cache_path=tmp_path / "ids.json").ldraw_id("odd-3003") == "3003"
    run(pm.ldraw_id_async("odd-3003"))
    assert asked == ["/parts/odd-3003/"]                                    # one API request in total
    assert run(PartMapper(lib, {}, api=None).ldraw_id_async("nothing")) is None


def test_ai_build_uses_the_rebrickable_lookup_for_unmapped_parts(lib, tmp_path):
    class FakeApi:
        enabled = True

        async def _get(self, path, params=None):
            return {"external_ids": {"LDraw": ["3003"]}} if "/parts/odd-3003/" in path else None
    pm = PartMapper(lib, {"3001": "3001", "odd-3003": "odd-3003"}, api=FakeApi())
    svc = BuildService(lib, pm, ColourMapper(lib), tmp_path / "cache", "https://x")
    parts = [{"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 3}, {"part_num": "odd-3003", "name": "Brick 2 x 2", "color": "Red", "qty": 2}]
    meta = run(svc.create_ai_build(parts, None, FakeLLM(fixture_json())))
    assert meta["skipped_parts"] == [] or all(s["part"] != "odd-3003" for s in meta["skipped_parts"])
