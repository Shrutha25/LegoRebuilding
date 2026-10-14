import pytest
from fastapi.testclient import TestClient

from backend import config, main
from backend.core.index import DataStore
from tests.fixtures import write_rb_dir


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = DataStore(str(write_rb_dir(tmp_path / "rb")))
    store._load()                                   # synchronous for the test
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(store, "start", lambda: None)   # lifespan must not reload
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "up")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    h = client.get("/api/health").json()
    assert h["data_loaded"] is True and h["data_error"] is None


def test_match_maps_ids_and_reports_unmapped(client):
    r = client.post("/api/match", json={"parts": [
        {"part_num": "3001pr1", "qty": 2}, {"part_num": "3002", "qty": 1},
        {"part_num": "970c00", "qty": 1}]}).json()
    assert [s["set_num"] for s in r["sets"]] == ["10-1"]
    assert r["unmapped"] == ["970c00"]


def test_match_validation(client):
    assert client.post("/api/match", json={"parts": []}).status_code == 422
    assert client.post("/api/match", json={"parts": [{"part_num": "1", "qty": 0}]}).status_code == 422


def test_match_503_while_loading(client, monkeypatch):
    monkeypatch.setattr(main, "store", DataStore("nowhere"))
    assert client.post("/api/match", json={"parts": [{"part_num": "3001"}]}).status_code == 503
    assert client.get("/api/health").json()["data_loaded"] is False


def test_parts_search(client):
    r = client.get("/api/parts/search", params={"q": "legs"}).json()["results"]
    assert r[0]["part_num"] == "41879a"


def test_scan_rejects_bad_uploads(client):
    assert client.post("/api/scan", files={"file": ("a.txt", b"hi", "text/plain")}).status_code == 415
    assert client.post("/api/scan", files={"file": ("a.jpg", b"junk", "image/jpeg")}).status_code == 400
    assert client.post("/api/scan", files={"file": ("a.mp4", b"x", "video/mp4")}).status_code == 501
    assert client.get("/api/scan/nope").status_code == 404
