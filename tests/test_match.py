from backend.core.match import (build_set_requirements, canonical_map, rank_sets,
                                skipped_theme_ids)
from tests.fixtures import fixture


def test_variant_merging():
    canon = canonical_map(fixture()["part_relationships"])
    assert canon("3001pr2") == "3001"      # chained
    assert canon("3001pr1") == "3001"
    assert canon("9999") == "9999"         # non-variant relation type ignored


def test_latest_inventory_and_spares_and_variants():
    rb = fixture()
    req = build_set_requirements(rb, canonical_map(rb["part_relationships"]), False)
    assert dict(req["10-1"]) == {"3001": 2, "3002": 3}   # v2 only, spare dropped, merged


def test_color_keys():
    rb = fixture()
    req = build_set_requirements(rb, canonical_map(rb["part_relationships"]), True)
    assert req["10-1"][("3001", "red")] == 2


def test_theme_skipping_includes_children():
    assert skipped_theme_ids(fixture()["themes"]) == {"2", "3"}


def test_rank_sets_filters_and_scores():
    rb = fixture()
    canon = canonical_map(rb["part_relationships"])
    req = build_set_requirements(rb, canon, False)
    skip = skipped_theme_ids(rb["themes"])
    res = rank_sets({"3001": 2, "3002": 1}, req, rb["sets"], skip, 150, 1, 0.2, 10)
    assert [r["set_num"] for r in res] == ["10-1"]        # pack themes skipped
    r = res[0]
    assert (r["total_parts"], r["parts_you_have"], r["coverage"]) == (5, 3, 0.6)
    assert r["missing"] == [{"part": "3002", "color": None, "qty": 2}]


def test_rank_sets_size_filter():
    rb = fixture()
    req = build_set_requirements(rb, canonical_map(rb["part_relationships"]), False)
    have = {"3001": 2}
    assert rank_sets(have, req, rb["sets"], set(), 4, 1, 0.0, 10) == []     # 20-1 has 10, 10-1 has 5
    assert len(rank_sets(have, req, rb["sets"], set(), 5, 1, 0.0, 10)) == 1  # 10-1 only


# ------------------------------------------------------------------ complete sets only, biggest first
def _sets(*rows, years=None):
    import pandas as pd
    years = years or {}
    return pd.DataFrame([[n, n, years.get(n, "2020"), "1", ""] for n in rows],
                        columns=["set_num", "name", "year", "theme_id", "img_url"])


def _parts(prefix, n):
    return {f"{prefix}{i}": 1 for i in range(n)}


def test_99_percent_set_is_excluded_and_100_percent_included():
    hundred = _parts("p", 100)
    have_99 = _parts("p", 99)
    assert rank_sets(have_99, {"s-1": hundred}, _sets("s-1"), set(), 150, 5, 1.0, 10) == []
    res = rank_sets(_parts("p", 100), {"s-1": hundred}, _sets("s-1"), set(), 150, 5, 1.0, 10)
    assert [r["set_num"] for r in res] == ["s-1"] and res[0]["complete"] and res[0]["missing"] == []


def test_larger_complete_sets_rank_above_smaller_ones():
    have = _parts("p", 60)
    req = {"small-1": _parts("p", 8), "big-1": _parts("p", 58), "mid-1": _parts("p", 20)}
    res = rank_sets(have, req, _sets("small-1", "big-1", "mid-1"), set(), 150, 5, 1.0, 10)
    assert [r["set_num"] for r in res] == ["big-1", "mid-1", "small-1"]


def test_equal_size_complete_sets_tie_break_on_newest_year():
    have = _parts("p", 12)
    req = {"old-1": _parts("p", 12), "new-1": _parts("p", 12), "mid-1": _parts("p", 12)}
    sets = _sets("old-1", "new-1", "mid-1", years={"old-1": "1999", "new-1": "2024", "mid-1": "2010"})
    res = rank_sets(have, req, sets, set(), 150, 5, 1.0, 10)
    assert [r["set_num"] for r in res] == ["new-1", "mid-1", "old-1"]


def test_min_coverage_is_configurable_and_score_is_have_times_coverage():
    have = {**_parts("p", 99), **_parts("q", 8)}
    req = {"big-1": _parts("p", 100), "small-1": _parts("q", 8)}      # big is 99%, small is 100%
    sets = _sets("big-1", "small-1")
    assert [r["set_num"] for r in rank_sets(have, req, sets, set(), 150, 5, 1.0, 10)] == ["small-1"]
    res = rank_sets(have, req, sets, set(), 150, 5, 0.9, 10)
    # score = have * coverage: 99 * 0.99 = 98 beats 8 * 1.0 = 8, so the big near-complete set leads
    assert [r["set_num"] for r in res] == ["big-1", "small-1"]
    assert [r["complete"] for r in res] == [False, True]


def test_equal_scores_tie_break_on_fewest_missing():
    # a: have 6 of 18 -> 6*0.333 = 2.0, missing 12.  b: have 4 of 8 -> 4*0.5 = 2.0, missing 4.
    a = {**_parts("x", 6), **_parts("a", 12)}
    b = {**_parts("x", 4), **_parts("b", 4)}
    have = _parts("x", 6)
    for order in ({"a-1": a, "b-1": b}, {"b-1": b, "a-1": a}):       # insertion order must not matter
        res = rank_sets(have, order, _sets("a-1", "b-1"), set(), 150, 5, 0.2, 10)
        assert [r["set_num"] for r in res] == ["b-1", "a-1"]


def test_size_and_theme_filters_still_apply_to_complete_sets():
    have = _parts("p", 100)
    req = {"big-1": _parts("p", 100)}
    assert rank_sets(have, req, _sets("big-1"), set(), 60, 5, 1.0, 10) == []          # max_parts
    assert rank_sets(have, req, _sets("big-1"), {"1"}, 150, 5, 1.0, 10) == []         # skipped theme
    assert rank_sets(_parts("p", 3), {"tiny-1": _parts("p", 3)}, _sets("tiny-1"), set(), 150, 5, 1.0, 10) == []  # min_parts


def test_min_coverage_env_parsing():
    from backend.config import parse_min_coverage
    assert parse_min_coverage(None) == 1.0 and parse_min_coverage("") == 1.0 and parse_min_coverage("abc") == 1.0
    assert parse_min_coverage("0.5") == 0.5 and parse_min_coverage("7") == 1.0 and parse_min_coverage("-1") == 0.0


def test_real_10422_inventory_still_returns_10422():
    """Needs the downloaded Rebrickable data (python -m backend.core.fetch_data); skipped otherwise."""
    import os
    import pytest
    from backend.core.index import MatchIndex
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rb_data")
    if not os.path.exists(os.path.join(data, "inventory_parts.csv")):
        pytest.skip("rb_data not downloaded")
    idx = MatchIndex.load(data)
    for mc in (False, True):
        parts = [{"part_num": k[0] if isinstance(k, tuple) else k, "color": k[1] if isinstance(k, tuple) else None, "qty": q}
                 for k, q in idx.req[mc]["10422-1"].items()]
        res = idx.match(parts, match_color=mc, max_parts=150)          # default min_coverage: complete only
        assert "10422-1" in [r["set_num"] for r in res]
        assert all(r["complete"] and r["coverage"] == 1.0 for r in res)
        assert res[0]["total_parts"] == 58                              # biggest buildable first
        assert [r["total_parts"] for r in res] == sorted((r["total_parts"] for r in res), reverse=True)
