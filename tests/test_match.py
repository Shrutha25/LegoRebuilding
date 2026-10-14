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


# ------------------------------------------------------------------ ranking: have * coverage
def _sets(*rows):
    import pandas as pd
    return pd.DataFrame([[n, n, "2020", "1", ""] for n in rows],
                        columns=["set_num", "name", "year", "theme_id", "img_url"])


def test_big_partial_set_outranks_small_complete_set():
    big = {f"b{i}": 1 for i in range(58)}
    small = {f"s{i}": 1 for i in range(8)}
    have = {**{f"b{i}": 1 for i in range(41)}, **{f"s{i}": 1 for i in range(8)}}   # 70.7% of 58, 100% of 8
    res = rank_sets(have, {"big-1": big, "small-1": small}, _sets("big-1", "small-1"),
                    set(), 150, 5, 0.2, 10)
    assert [r["set_num"] for r in res] == ["big-1", "small-1"]
    assert res[0]["coverage"] == 0.707 and res[1]["coverage"] == 1.0   # displayed number unchanged
    assert res[1]["complete"] is True and res[0]["complete"] is False


def test_complete_big_set_beats_complete_small_set():
    big = {f"b{i}": 1 for i in range(20)}
    small = {f"b{i}": 1 for i in range(8)}                 # subset of big, also 100%
    have = {f"b{i}": 1 for i in range(20)}
    res = rank_sets(have, {"small-1": small, "big-1": big}, _sets("big-1", "small-1"), set(), 150, 5, 0.2, 10)
    assert [r["set_num"] for r in res] == ["big-1", "small-1"]


def test_equal_scores_tie_break_on_fewest_missing():
    # a: have 6 of 18 -> 6*0.333 = 2.0, missing 12.  b: have 4 of 8 -> 4*0.5 = 2.0, missing 4.
    a = {**{f"x{i}": 1 for i in range(6)}, **{f"a{i}": 1 for i in range(12)}}
    b = {**{f"x{i}": 1 for i in range(4)}, **{f"b{i}": 1 for i in range(4)}}
    have = {f"x{i}": 1 for i in range(6)}
    for order in ({"a-1": a, "b-1": b}, {"b-1": b, "a-1": a}):       # insertion order must not matter
        res = rank_sets(have, order, _sets("a-1", "b-1"), set(), 150, 5, 0.2, 10)
        assert [r["set_num"] for r in res] == ["b-1", "a-1"]


def test_filters_unchanged_by_new_ranking():
    big = {f"b{i}": 1 for i in range(100)}
    have = {f"b{i}": 1 for i in range(10)}                 # 10% coverage: below min_coverage
    assert rank_sets(have, {"big-1": big}, _sets("big-1"), set(), 150, 5, 0.2, 10) == []
    assert rank_sets({f"b{i}": 1 for i in range(50)}, {"big-1": big}, _sets("big-1"), set(), 60, 5, 0.2, 10) == []  # max_parts


def test_real_10422_inventory_ranks_first():
    """Needs the downloaded Rebrickable data (python -m backend.core.fetch_data); skipped otherwise."""
    import os
    import pytest
    from backend.core.index import MatchIndex
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rb_data")
    if not os.path.exists(os.path.join(data, "inventory_parts.csv")):
        pytest.skip("rb_data not downloaded")
    idx = MatchIndex.load(data)
    parts = [{"part_num": k[0] if isinstance(k, tuple) else k, "color": k[1] if isinstance(k, tuple) else None, "qty": q}
             for k, q in idx.req[True]["10422-1"].items()]
    res = idx.match(parts, match_color=True, max_parts=150)
    assert res[0]["set_num"] == "10422-1" and res[0]["coverage"] == 1.0 and res[0]["complete"]
