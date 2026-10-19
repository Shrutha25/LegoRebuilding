"""Real LEGO sub-builds from the OMR: parsing, flattening, matching with the colour rule, almost-buildable, the API and the start-up warm-up."""

import pytest

from backend import builds_api, config
from backend.core import omr_index
from backend.core.omr_index import INHERIT, OmrIndex, extract_model, flatten, index_model, split_blocks
from tests.test_builds import client, lib, service  # noqa: F401  (fixtures: a fake LDraw library, the app with a small data set)

LICENCE = "0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt"


def line(colour, part):
    return f"1 {colour} 0 0 0 1 0 0 0 1 0 0 0 1 {part}"


def mpd(licence=LICENCE):
    """A main model (1 red 2x4 and the car, in red) and a car sub-model (2 x 2x4 in the inherited colour, 4 black 2x2, 1 plate 2x4 in colour 7),
    a stepwise sub-model that is not a build, and a sub-model that is too small."""
    return "\n".join([
        "0 FILE 9999-1 - Demo.mpd", "0 Demo Model", "0 Name: 9999-1 - Demo.mpd", "0 Author: Test Author [tester]", "0 !LDRAW_ORG Model", licence,
        line(4, "3001.dat"), line(4, "car.ldr"), line(2, "step 5-9.ldr"), line(2, "tiny.ldr"),
        "0 FILE car.ldr", "0 Car", line(16, "3001.dat"), line(16, "3001.dat"), *[line(0, "3003.dat")] * 4, line(7, "3020.dat"),
        "0 FILE step 5-9.ldr", "0 Step 5-9", *[line(15, "3005.dat")] * 8,
        "0 FILE tiny.ldr", "0 Tiny", line(15, "3005.dat"), line(15, "3005.dat"),
    ]) + "\n"


def test_blocks_flatten_with_inherited_colours_and_nested_models():
    blocks = split_blocks(mpd())
    assert [b[0] for b in blocks] == ["9999-1 - Demo.mpd", "car.ldr", "step 5-9.ldr", "tiny.ldr"]
    flat = flatten(blocks)
    assert flat[1] == {("3001", INHERIT): 2, ("3003", 0): 4, ("3020", 7): 1}                       # the car on its own: colour 16 stays "main"
    assert flat[0][("3001", 4)] == 3 and flat[0][("3003", 0)] == 4 and flat[0][("3020", 7)] == 1  # in the main model the car's 16 became red
    assert sum(flat[0].values()) == 1 + 7 + 8 + 2


def test_index_keeps_open_licence_models_of_6_to_80_parts_and_skips_steps_and_tiny_ones():
    builds = index_model("9999-1", mpd(), "Demo Set")
    assert [(b.id, b.title, b.parts, b.main) for b in builds] == [("sub-9999-1-0", "Demo Set", 18, True), ("sub-9999-1-1", "Car", 7, False)]
    assert builds[1].authors == ("Test Author [tester]",) and "CCAL" in builds[1].licence[0]
    assert index_model("9999-1", mpd("0 !LICENSE Not redistributable: all rights reserved")) == []
    assert index_model("9999-1", mpd("")) == []                                                      # no licence line: not used


def have(**kw):
    return {("3001", 4): 2, ("3003", 0): 4, ("3020", 7): 1, **kw}


def index():
    return OmrIndex(index_model("9999-1", mpd(), "Demo Set"))


def test_buildable_when_the_pieces_cover_it_and_the_inherited_colour_takes_any_colour():
    res = index().match(have(), match_color=True)
    assert [b.id for b, _ in res["buildable"]] == ["sub-9999-1-1"] and res["counts"]["buildable"] == 1 and res["counts"]["smallest_buildable"] == 7
    res = index().match({("3001", 14): 2, ("3003", 0): 4, ("3020", 7): 1}, match_color=True)            # yellow 2x4s: the car's "main colour" is any colour
    assert [b.id for b, _ in res["buildable"]] == ["sub-9999-1-1"]


def test_with_colour_on_a_wrong_colour_is_missing_and_with_colour_off_it_is_not():
    wrong = {("3001", 4): 2, ("3003", 1): 4, ("3020", 7): 1}                                           # the 2x2s are blue, the car needs black
    on = index().match(wrong, match_color=True)
    assert on["buildable"] == [] and on["almost"] == [] and on["counts"]["almost"] == 0                 # 4 missing: too many
    off = index().match(wrong, match_color=False)
    assert [b.id for b, _ in off["buildable"]] == ["sub-9999-1-1"]
    three = {("3001", 4): 2, ("3003", 1): 3, ("3003", 0): 1, ("3020", 7): 1}                           # 3 blue + 1 black: 3 black missing
    assert index().match(three, True)["almost"] == []
    two = {("3001", 4): 2, ("3003", 1): 2, ("3003", 0): 2, ("3020", 7): 1}
    near = index().match(two, True)
    (b, rows), = near["almost"]
    assert b.id == "sub-9999-1-1" and rows == [("3003", 0, 2)] and near["counts"]["smallest_almost"] == 7


def test_shapes_never_substitute_and_missing_parts_are_counted_exactly():
    assert index().match({("3001", 4): 2, ("3003", 0): 4}, True)["almost"][0][1] == [("3020", 7, 1)]       # one plate missing
    assert index().match({("3002", 4): 2, ("3003", 0): 4, ("3020", 7): 1}, False)["buildable"] == []        # 2x3 bricks are not 2x4 bricks
    assert index().match({}, True) == {"buildable": [], "almost": [], "counts": {"buildable": 0, "almost": 0, "smallest_buildable": None, "smallest_almost": None}}


def test_ranking_prefers_bigger_builds_and_limits_each_set():
    builds = [omr_index.SubBuild(f"sub-1-1-{i}", "1-1", i, f"b{i}", 10 + i, (("3001", 4, 10 + i),)) for i in range(5)]
    res = OmrIndex(builds).match({("3001", 4): 99}, True, per_set=2)
    assert [b.parts for b, _ in res["buildable"]] == [14, 13] and res["counts"]["buildable"] == 5


def test_extract_model_makes_a_submodel_standalone_with_what_it_needs():
    text = extract_model(mpd(), 1)
    assert text.startswith("0 FILE car.ldr") and "3003.dat" in text and "tiny.ldr" not in text and "Test Author" in text and "CCAL" in text
    whole = extract_model(mpd(), 0)
    assert "0 FILE car.ldr" in whole and whole.startswith("0 FILE 9999-1 - Demo.mpd")


def test_the_index_is_cached_and_rebuilt_only_when_the_models_change(tmp_path, monkeypatch):
    folder = tmp_path / "omr"
    folder.mkdir()
    (folder / "9999-1.mpd").write_text(mpd(), encoding="utf-8")
    first = OmrIndex.load(folder, tmp_path / "idx.pkl", {"9999-1": "Demo Set"})
    assert len(first) == 2 and (tmp_path / "idx.pkl").exists()
    monkeypatch.setattr(OmrIndex, "build", classmethod(lambda cls, *a, **k: pytest.fail("rebuilt although nothing changed")))
    assert len(OmrIndex.load(folder, tmp_path / "idx.pkl")) == 2
    assert len(OmrIndex.load(tmp_path / "missing", tmp_path / "x.pkl")) == 0


# ---------------------------------------------------------------- the API
def test_the_api_lists_buildable_and_almost_buildable_sub_builds_with_attribution(client, tmp_path, monkeypatch):
    monkeypatch.setitem(builds_api._omr, "index", index())
    parts = [{"part_num": "3001", "color": "Red", "qty": 2}, {"part_num": "3003", "color": "Black", "qty": 4}, {"part_num": "3020", "color": "Light Gray", "qty": 1}]
    r = client.post("/api/subbuilds", json={"parts": parts, "match_color": False})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ready"] and d["available"] and [b["id"] for b in d["buildable"]] == ["sub-9999-1-1"] and d["counts"]["smallest_buildable"] == 7
    b = d["buildable"][0]
    assert b["authors"] == ["Test Author [tester]"] and "CCAL" in b["licence"][0] and b["set_num"] == "9999-1" and b["source_url"].endswith("search=9999")
    short = [dict(parts[0], qty=2), dict(parts[1], qty=2), dict(parts[2])]
    d2 = client.post("/api/subbuilds", json={"parts": short, "match_color": False}).json()
    assert d2["buildable"] == [] and d2["almost"][0]["missing_count"] == 2
    assert d2["almost"][0]["missing"][0]["part"] == "3003" and d2["almost"][0]["missing"][0]["qty"] == 2 and d2["almost"][0]["missing"][0]["name"]


def test_the_api_says_when_the_index_is_not_ready_or_empty(client, monkeypatch):
    monkeypatch.setitem(builds_api._omr, "index", None)
    assert client.post("/api/subbuilds", json={"parts": []}).json()["ready"] is False
    monkeypatch.setitem(builds_api._omr, "index", OmrIndex([]))
    d = client.post("/api/subbuilds", json={"parts": []}).json()
    assert d["ready"] and d["available"] is False and "fetch_omr" in d["hint"]


def test_a_sub_build_opens_in_the_viewer_with_steps_and_attribution(client, tmp_path):
    folder = tmp_path / "cache" / "omr"
    folder.mkdir(parents=True)
    (folder / "9999-1.mpd").write_text(mpd(), encoding="utf-8")
    r = client.post("/api/subbuilds/build", json={"id": "sub-9999-1-1", "title": "Car"})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["id"] == "sub-9999-1-1" and meta["kind"] == "set" and meta["attribution"]["authors"] == ["Test Author [tester]"] and meta["step_count"] >= 1
    assert client.get(meta["model_url"]).status_code == 200
    assert client.get(meta["model_url"]).text.count("0 FILE") >= 1
    assert client.post("/api/subbuilds/build", json={"id": "sub-0000-1-1"}).status_code == 404
    assert client.post("/api/subbuilds/build", json={"id": "../x"}).status_code == 404
    (folder / "8888-1.mpd").write_text(mpd("0 !LICENSE All rights reserved"), encoding="utf-8")
    assert client.post("/api/subbuilds/build", json={"id": "sub-8888-1-1"}).status_code == 403


# ---------------------------------------------------------------- start-up
def test_warm_up_builds_the_nearby_and_omr_indexes_before_the_first_request(tmp_path, monkeypatch):
    from backend.core.index import DataStore
    from tests.fixtures import write_rb_dir
    store = DataStore(str(write_rb_dir(tmp_path / "rb")))
    store._load()
    cache = tmp_path / "bc"
    (cache / "omr").mkdir(parents=True)
    (cache / "omr" / "9999-1.mpd").write_text(mpd(), encoding="utf-8")
    monkeypatch.setattr(config, "BUILD_CACHE_DIR", cache)
    monkeypatch.setattr(builds_api, "_matcher", None)
    monkeypatch.setitem(builds_api._omr, "index", None)
    monkeypatch.setattr(builds_api, "_service", None)
    monkeypatch.setattr(builds_api, "_warm", {"nearby": False, "omr": False, "service": False, "examples": False, "seconds": {}})
    monkeypatch.setattr(builds_api, "_load_examples", lambda: None)
    monkeypatch.setitem(builds_api._deps, "store", store)
    builds_api.warm_indexes(store, wait=5)
    st = builds_api.warm_status()
    assert st["nearby"] and st["omr"] and builds_api._matcher is not None and st["omr_sub_builds"] == 2
    assert set(st["seconds"]) == {"nearby", "omr", "service", "examples"}


def test_a_post_right_after_start_up_does_not_build_anything(client, monkeypatch):
    """The nearby-sets request must find its index ready (built by the warm-up), not build it."""
    from backend.core.setmatch import SetMatcher
    store = builds_api._deps["store"]
    built = []
    monkeypatch.setattr(builds_api, "_matcher", SetMatcher(store.index, store.parts))
    monkeypatch.setattr(SetMatcher, "__init__", lambda *a, **k: built.append(1))
    r = client.post("/api/sets/nearby", json={"parts": [{"part_num": "3001", "color": "Red", "qty": 3}]})
    assert r.status_code == 200 and built == []


def test_renamed_parts_are_matched_under_their_current_number(tmp_path):
    """LDraw leaves a "~Moved to 3005" stub under an old number: OMR models and the user's part mapping may use either."""
    from backend.core.ldmap import ColourMapper, PartMapper
    from backend.core.buildsvc import BuildService
    from backend.core.ldraw_lib import LDrawLibrary
    from tests.build_fixtures import fake_library_zip
    lib = LDrawLibrary(fake_library_zip(tmp_path / "c.zip", {"3005": ("Brick 1 x 1", 10, 24, 10), "3139": ("~Moved to 3005", 10, 24, 10)}))
    svc = BuildService(lib, PartMapper(lib, {"3139": "3139", "3005": "3005"}), ColourMapper(lib), tmp_path / "cache", "https://x.invalid")
    assert svc.canon("3139") == "3005" and svc.canon("3005") == "3005" and svc.canon("nope") == "nope"
    text = mpd().replace("3001.dat", "3139.dat")
    plain = index_model("9999-1", text)[1]
    moved = index_model("9999-1", text, canon=svc.canon)[1]
    assert any(p == "3139" for p, _, _ in plain.need)
    assert not any(p == "3139" for p, _, _ in moved.need)
