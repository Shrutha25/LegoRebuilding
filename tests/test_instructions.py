"""Non-buildable sets, the Brickset instructions cache, the set card state and ranking order."""

import os

import pytest

from backend import config
from backend.core import omr
from backend.core.instructions import InstructionsService, lego_link, parse_response
from backend.core.match import exclusion_report, skipped_theme_ids
from backend.core.setinfo import SetInfo, card_state, rank_key
from backend import main
from backend.core.index import DataStore
from backend.core.rebrickable_api import RebrickableAPI
from fastapi.testclient import TestClient
from tests.fixtures import df, write_rb_dir

LEGO = "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6012.pdf"


@pytest.fixture
def rb_dir(tmp_path):
    return write_rb_dir(tmp_path / "rb")


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = DataStore(str(write_rb_dir(tmp_path / "rbc")))
    store._load()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "rb_api", RebrickableAPI(""))
    monkeypatch.setattr(store, "start", lambda: None)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "up")
    with TestClient(main.app) as c:
        yield c


# ---------------------------------------------------------------- themes and size
def themes():
    return df([["1", "Town", ""], ["2", "Gear", ""], ["3", "Backpacks", "2"], ["4", "Deep", "3"],      # excluded by name, + descendants
               ["5", "Books", ""], ["6", "Star Wars", ""], ["7", "Video Games Tie-in", "6"],          # excluded by word, not its parent
               ["8", "storage boxes", "1"], ["9", "Gearbox Technic", "1"], ["10", "Space", "1"]], ["id", "name", "parent_id"])


def test_excluded_themes_include_all_sub_themes_and_match_names_case_insensitively():
    bad = skipped_theme_ids(themes())
    assert {"2", "3", "4"} <= bad            # Gear and its children, however deep
    assert "5" in bad                         # Books
    assert "7" in bad and "6" not in bad      # a name containing "Video Games"; the parent is a normal theme
    assert "8" in bad                         # "Storage" is matched case-insensitively
    assert "9" in bad                         # contains "Gear"
    assert not ({"1", "6", "10"} & bad)


def test_excluded_theme_lists_come_from_config_and_can_be_overridden():
    assert "gear" in config.EXCLUDED_THEMES and "video games" in config.EXCLUDED_THEME_WORDS
    assert skipped_theme_ids(themes(), names=["space"], words=[]) == {"10"}


def test_exclusion_report_counts_themes_and_small_sets():
    sets = df([["a-1", "A", "2000", "2", "20", ""], ["b-1", "B", "2000", "3", "50", ""], ["c-1", "C", "2000", "10", "5", ""],
               ["d-1", "D", "2000", "10", "500", ""]], ["set_num", "name", "year", "theme_id", "num_parts", "img_url"])
    rep = exclusion_report(sets, themes(), skipped_theme_ids(themes()), 10)
    assert rep["by_theme"] == 2 and rep["small"] == 1 and rep["excluded"] == 3
    assert sorted(rep["top_themes"]) == [("Backpacks", 1), ("Gear", 1)]


def test_min_set_parts_drops_small_sets(rb_dir, monkeypatch):
    from backend.core.index import MatchIndex
    parts = [{"part_num": "3001", "color": "Red", "qty": 20}, {"part_num": "3002", "color": "Blue", "qty": 5}]
    idx = MatchIndex.build(str(rb_dir))
    assert [r["set_num"] for r in idx.match(parts, min_coverage=0.0)] == ["10-1"]      # 5 parts; MIN_SET_PARTS is 1 in tests
    monkeypatch.setattr(config, "MIN_SET_PARTS", 10)
    assert idx.match(parts, min_coverage=0.0) == []


def test_rebrickables_own_part_count_also_excludes(rb_dir, monkeypatch):
    from backend.core.index import MatchIndex
    monkeypatch.setattr(config, "MIN_SET_PARTS", 10)
    idx = MatchIndex.build(str(rb_dir))
    data = {k: v for k, v in idx.__dict__.items() if not k.startswith("_")}
    data["sets"] = idx.sets.assign(num_parts=["5", "500", "500"])
    assert not MatchIndex(data).is_buildable("10-1")


# ---------------------------------------------------------------- Brickset answers
def test_parse_response_keeps_only_lego_links():
    data = {"status": "success", "matches": 3, "instructions": [
        {"URL": LEGO, "description": "Book 1"}, {"URL": LEGO, "description": "dup"},
        {"URL": "https://evil.example/6012.pdf", "description": "x"}, {"URL": "http://www.lego.com/a.pdf", "description": "http"},
        {"URL": "https://notlego.com/a.pdf", "description": "x"}]}
    assert parse_response(data) == [{"url": LEGO, "description": "Book 1"}]
    assert parse_response({"status": "success", "matches": 0, "instructions": []}) == []
    with pytest.raises(ValueError):
        parse_response({"status": "error", "message": "Invalid API key"})
    assert lego_link("https://lego.com/x") and not lego_link("https://lego.com.evil.io/x") and not lego_link("")


# ---------------------------------------------------------------- cache
class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t


def service(tmp_path, fetch, key="k", cap=100, clock=None):
    return InstructionsService(tmp_path / "i.sqlite", key, ttl_days=7, daily_cap=cap, min_interval=0, fetch=fetch, clock=clock or Clock())


def test_cache_hit_does_not_call_the_api_and_uses_the_full_number(tmp_path):
    calls = []
    svc = service(tmp_path, lambda key, num: calls.append(num) or [{"url": LEGO, "description": "Book"}])
    first = svc.refresh("6012-2")
    assert first["available"] and calls == ["6012-2"]                       # suffix kept
    assert svc.refresh("6012-2") == first and calls == ["6012-2"]           # fresh: no second call
    assert svc.get("6012-2")["links"] == first["links"] and svc.get("6012-1")["known"] is False


def test_stale_entry_is_refreshed_after_the_ttl(tmp_path):
    clock, answers = Clock(), [[{"url": LEGO, "description": "old"}], []]
    svc = service(tmp_path, lambda k, n: answers.pop(0), clock=clock)
    assert svc.refresh("1-1")["available"]
    clock.t += 6 * 86400
    assert svc.refresh("1-1")["links"][0]["description"] == "old" and answers   # still fresh
    clock.t += 2 * 86400
    assert svc.needs_refresh("1-1")
    assert svc.refresh("1-1")["available"] is False and not answers              # refreshed: LEGO no longer lists any


def test_api_error_keeps_the_old_value_or_stays_unknown(tmp_path):
    clock = Clock()
    state = {"fail": False}

    def fetch(k, n):
        if state["fail"]:
            raise TimeoutError("slow")
        return [{"url": LEGO, "description": "Book"}]

    svc = service(tmp_path, fetch, clock=clock)
    assert svc.refresh("2-1")["available"]
    clock.t += 30 * 86400
    state["fail"] = True
    assert svc.refresh("2-1")["available"] and svc.get("2-1")["links"]       # stale value survives the error
    assert svc.refresh("3-1") is None
    assert svc.get("3-1") == {"available": False, "links": [], "checked_at": None, "known": False}
    assert not svc.needs_refresh("3-1")                                      # not retried immediately...
    clock.t += 700
    assert svc.needs_refresh("3-1")                                          # ...but later


def test_missing_key_is_unknown_and_never_calls(tmp_path):
    called = []
    svc = service(tmp_path, lambda k, n: called.append(n), key="")
    assert svc.refresh("4-1") is None and not called and svc.get("4-1")["known"] is False


def test_daily_cap_stops_calls_but_keeps_cached_values(tmp_path):
    called = []
    svc = service(tmp_path, lambda k, n: called.append(n) or [], cap=2)
    svc.refresh("1-1"), svc.refresh("2-1")
    assert svc.refresh("3-1") is None and called == ["1-1", "2-1"] and svc.calls_today() == 2
    assert svc.get("1-1")["known"]


# ---------------------------------------------------------------- card state and ranking
def instr(available=False, links=(), known=True):
    return {"available": available, "links": list(links), "known": known}


def test_card_state_priority():
    assert card_state(True, instr(True, [{"url": LEGO}])) == 1              # a 3D model beats official instructions
    assert card_state(False, instr(True, [{"url": LEGO}])) == 2
    assert card_state(False, instr(False)) == 3
    assert card_state(False, instr(known=False)) is None                    # lookup still running
    assert card_state(None, instr(False)) is None                           # OMR not checked yet


def test_set_info_reads_caches_only_and_the_background_check_fills_them(tmp_path):
    svc = service(tmp_path, lambda k, n: [{"url": LEGO, "description": "Book"}] if n == "6012-1" else [])

    async def fetch_omr(num, cache_dir, base, client=None):
        return None

    cache = tmp_path / "omrcache"
    si = SetInfo(svc, cache, "https://x.invalid", fetch_omr=fetch_omr)
    assert si.info("6012-1") == {"state": None, "links": [], "pending": True}
    si.check("6012-1")
    si.check("99-1")
    assert si.info("6012-1")["state"] == 2 and si.info("6012-1")["links"][0]["url"] == LEGO
    assert si.info("99-1")["state"] == 3
    (cache / "omr").mkdir(parents=True)
    (cache / "omr" / "30051-1.mpd").write_text("0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt\n", encoding="utf-8")
    assert omr.cached_status("30051-1", cache) is True and si.info("30051-1")["state"] == 1
    (cache / "omr" / "5-1.mpd").write_text("0 !LICENSE all rights reserved\n", encoding="utf-8")
    assert omr.cached_status("5-1", cache) is False


def test_ranking_state_then_completeness_then_part_count():
    def row(num, state, missing, total):
        return {"set_num": num, "total_parts": total, "missing_count": missing, "instructions": {"state": state}}

    rows = [row("c", 3, 0, 500), row("b2", 2, 0, 100), row("a", 1, 5, 100), row("b1", 2, 0, 300), row("u", None, 0, 900), row("b3", 2, 10, 100)]
    # unknown counts as state 3 (after 1 and 2); inside a state the more complete set, then the bigger one, comes first
    assert [r["set_num"] for r in sorted(rows, key=rank_key)] == ["a", "b1", "b2", "b3", "u", "c"]


# ---------------------------------------------------------------- endpoint
def test_instructions_endpoint_is_additive_and_validates(client):
    r = client.post("/api/sets/instructions", json={"set_nums": ["6012-1", "../x"]})
    assert r.status_code == 200
    assert set(r.json()["sets"]) == {"6012-1"} and r.json()["sets"]["6012-1"]["pending"] is True


@pytest.mark.skipif(not os.environ.get("BRICKSET_LIVE_TEST"), reason="set BRICKSET_LIVE_TEST=1 and BRICKSET_API_KEY to call Brickset")
def test_live_brickset_lookup(tmp_path):
    svc = InstructionsService(tmp_path / "live.sqlite", config.BRICKSET_API_KEY, min_interval=0)
    e = svc.refresh("6012-1")
    assert e is not None and e["available"] and all(lego_link(l["url"]) for l in e["links"])
