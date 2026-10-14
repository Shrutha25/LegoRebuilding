import pytest

from backend.core import index as index_mod
from backend.core.index import CACHE_NAME, DataStore, MatchIndex
from backend.core.match import build_set_requirements, canonical_map, rank_sets, skipped_theme_ids
from tests.fixtures import fixture, write_rb_dir


@pytest.fixture
def rb_dir(tmp_path):
    return write_rb_dir(tmp_path / "rb")


def test_match_equals_plain_rank_sets(rb_dir):
    idx = MatchIndex.build(str(rb_dir))
    rb = fixture()
    canon = canonical_map(rb["part_relationships"])
    req = build_set_requirements(rb, canon, False)
    expected = rank_sets({"3001": 2, "3002": 1}, req, rb["sets"],
                         skipped_theme_ids(rb["themes"]), 150, 5, 0.2, 10)
    got = idx.match([{"part_num": "3001pr1", "qty": 2}, {"part_num": "3002", "qty": 1}], False, 150)
    assert [r["set_num"] for r in got] == [r["set_num"] for r in expected] == ["10-1"]
    assert got[0]["coverage"] == expected[0]["coverage"] == 0.6
    assert got[0]["rebrickable_url"].endswith("/10-1/")


def test_missing_parts_get_images_and_no_match_returns_empty(rb_dir):
    idx = MatchIndex.build(str(rb_dir))
    r = idx.match([{"part_num": "3001", "color": "red", "qty": 2}], True, 150)
    miss = r[0]["missing"]
    assert miss[0]["part"] == "3002" and miss[0]["image"] == "http://img/3002-blue"
    assert idx.match([{"part_num": "nope", "qty": 1}], False, 150) == []


def test_color_matching_changes_result(rb_dir):
    idx = MatchIndex.build(str(rb_dir))
    wrong = [{"part_num": "3001", "color": "blue", "qty": 2}]
    assert idx.match(wrong, False, 150)          # part-only: matches
    assert idx.match(wrong, True, 150) == []     # exact colour: no match


def test_cache_reused_then_invalidated(rb_dir, monkeypatch):
    real = MatchIndex.build.__func__
    built = []
    monkeypatch.setattr(MatchIndex, "build",
                        classmethod(lambda cls, d: built.append(1) or real(cls, d)))
    MatchIndex.load(str(rb_dir))
    assert (rb_dir / CACHE_NAME).exists() and built == [1]
    MatchIndex.load(str(rb_dir))                 # served from cache
    assert built == [1]
    p = rb_dir / "sets.csv"
    p.write_text(p.read_text() + "40-1,New,2023,1,http://x\n")   # changes size + mtime
    MatchIndex.load(str(rb_dir))
    assert built == [1, 1]


def test_corrupt_cache_is_rebuilt(rb_dir):
    (rb_dir / CACHE_NAME).write_bytes(b"garbage")
    assert MatchIndex.load(str(rb_dir)).req


def test_datastore_loads_and_searches(rb_dir):
    s = DataStore(str(rb_dir))
    s.start().join(30)
    assert s.data_loaded and s.parts_loaded and s.error is None
    assert s.known_ids["3001a"] == "3001a"
    assert [r["part_num"] for r in s.search_parts("3001")][:1] == ["3001"]   # exact number first
    assert s.search_parts("legs")[0]["part_num"] == "41879a"
    assert s.search_parts("") == []


def test_datastore_reports_missing_data(tmp_path):
    s = DataStore(str(tmp_path))
    s.start().join(30)
    assert not s.data_loaded and "fetch_data" in s.error


def test_cache_key_covers_all_sources(rb_dir):
    assert len(index_mod.source_key(str(rb_dir))) == 1 + len(index_mod.RB_FILES)


def test_index_ranking_equals_rank_sets_on_a_larger_catalogue(tmp_path):
    """Randomised catalogue: the index (inverted lookup + prefilter) must return exactly what
    rank_sets returns over every set, in the same order."""
    import random
    import pandas as pd
    rnd = random.Random(7)
    pool = [f"p{i}" for i in range(40)]
    sets, invs, rows = [], [], []
    for n in range(60):
        sn = f"{n}-1"
        sets.append([sn, f"Set {n}", str(2000 + n % 20), "1", ""])
        invs.append([str(n), "1", sn])
        for p in rnd.sample(pool, rnd.randint(3, 30)):
            rows.append([str(n), p, "1", str(rnd.randint(1, 3)), "f", ""])
    rb = fixture()
    rb["sets"] = pd.DataFrame(sets, columns=["set_num", "name", "year", "theme_id", "img_url"])
    rb["inventories"] = pd.DataFrame(invs, columns=["id", "version", "set_num"])
    rb["inventory_parts"] = pd.DataFrame(rows, columns=["inventory_id", "part_num", "color_id", "quantity", "is_spare", "img_url"])
    rb["parts"] = pd.DataFrame([[p, p] for p in pool], columns=["part_num", "name"])
    d = tmp_path / "rb"
    d.mkdir()
    for name, frame in rb.items():
        frame.to_csv(d / f"{name}.csv", index=False)

    idx = MatchIndex.build(str(d))
    canon = canonical_map(rb["part_relationships"])
    req = build_set_requirements(rb, canon, False)
    skip = skipped_theme_ids(rb["themes"])
    for trial in range(10):
        have = {p: rnd.randint(1, 3) for p in rnd.sample(pool, rnd.randint(3, 25))}
        for max_parts in (20, 150):
            expected = rank_sets(have, req, rb["sets"], skip, max_parts, 5, 0.2, 10)
            got = idx.match([{"part_num": p, "qty": q} for p, q in have.items()], False, max_parts)
            key = lambda r: (r["set_num"], r["parts_you_have"], r["coverage"], r["complete"])
            assert [key(r) for r in got] == [key(r) for r in expected], (trial, max_parts)
