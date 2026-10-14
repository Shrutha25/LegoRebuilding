from backend.core import fetch_data


def test_fetch_uses_the_same_directory_the_server_reads(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(fetch_data, "download_file", lambda name, d, refresh=False: seen.append((name, d, refresh)))
    monkeypatch.setenv("RB_DATA_DIR", str(tmp_path / "vol"))
    fetch_data.main([])
    assert {d for _, d, _ in seen} == {str(tmp_path / "vol")}
    assert {n for n, _, _ in seen} == set(fetch_data.RB_FILES)


def test_fetch_defaults_to_rb_data_and_flag_overrides(monkeypatch):
    seen = []
    monkeypatch.setattr(fetch_data, "download_file", lambda name, d, refresh=False: seen.append((d, refresh)))
    monkeypatch.delenv("RB_DATA_DIR", raising=False)
    fetch_data.main([])
    assert {d for d, _ in seen} == {"rb_data"}
    seen.clear()
    monkeypatch.setenv("RB_DATA_DIR", "/from/env")
    fetch_data.main(["--data-dir", "/explicit", "--refresh"])
    assert {d for d, _ in seen} == {"/explicit"} and all(r for _, r in seen)
