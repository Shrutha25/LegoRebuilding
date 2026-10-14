import pandas as pd

from backend.core.match import (build_set_requirements, canonical_map, rank_sets,
                                skipped_theme_ids)


def df(rows, cols):
    return pd.DataFrame(rows, columns=cols, dtype=str)


def fixture():
    rels = df([["3001pr1", "3001", "P"], ["3001pr2", "3001pr1", "P"], ["9999", "1111", "R"]],
              ["child_part_num", "parent_part_num", "rel_type"])
    themes = df([["1", "Town", ""], ["2", "Supplemental", ""], ["3", "Bulk Sub", "2"]],
                ["id", "name", "parent_id"])
    sets = df([["10-1", "Small House", "2020", "1", "http://img/10"],
               ["20-1", "Parts Pack", "2021", "2", "http://img/20"],
               ["30-1", "Sub Pack", "2021", "3", "http://img/30"]],
              ["set_num", "name", "year", "theme_id", "img_url"])
    colors = df([["1", "Red"], ["2", "Blue"]], ["id", "name"])
    inventories = df([["100", "1", "10-1"], ["101", "2", "10-1"], ["200", "1", "20-1"],
                      ["300", "1", "30-1"]], ["id", "version", "set_num"])
    inv_parts = df([
        ["100", "3001", "1", "99", "f"],            # old version: must be ignored
        ["101", "3001pr1", "1", "2", "f"],          # variant -> 3001
        ["101", "3002", "2", "3", "f"],
        ["101", "3003", "2", "5", "t"],             # spare: must be excluded
        ["200", "3001", "1", "10", "f"],
        ["300", "3001", "1", "10", "f"],
    ], ["inventory_id", "part_num", "color_id", "quantity", "is_spare"])
    return {"part_relationships": rels, "themes": themes, "sets": sets, "colors": colors,
            "inventories": inventories, "inventory_parts": inv_parts}


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
