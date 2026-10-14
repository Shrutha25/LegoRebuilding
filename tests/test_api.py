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


def test_parts_resolve_flags_unmapped(client):
    r = client.post("/api/parts/resolve", json={"parts": ["3001b", "41879", "970c00"]}).json()["results"]
    assert r["3001b"]["part_num"] == "3001" and not r["3001b"]["unmapped"]
    assert r["3001b"]["name"] == "Brick 2 x 4"
    assert r["41879"]["part_num"] == "41879a"
    assert r["970c00"]["unmapped"] is True


def test_cors_allows_lan_origin_but_not_others(client):
    for origin, ok in [("http://192.168.1.20:3000", True), ("http://localhost:3000", True),
                       ("http://10.0.0.5:3000", True), ("http://evil.example.com", False),
                       ("http://192.169.1.1:3000", False)]:
        h = client.get("/api/health", headers={"Origin": origin}).headers
        assert (h.get("access-control-allow-origin") == origin) is ok, origin


def test_image_proxy_allowlist(client):
    bad = ["http://storage.googleapis.com/brickognize-static/a.webp",       # not https
           "https://evil.com/brickognize-static/a.webp",
           "https://storage.googleapis.com/other-bucket/a.webp",
           "https://storage.googleapis.com/brickognize-static/../x/a.webp",
           "https://storage.googleapis.com/brickognize-static/a.html",
           "https://storage.googleapis.com@evil.com/brickognize-static/a.webp",
           "http://169.254.169.254/latest/meta-data"]
    for u in bad:
        assert client.get("/api/img", params={"url": u}).status_code == 400, u
    assert main._allowed_image("https://storage.googleapis.com/brickognize-static/t/3622/154.webp") == "image/webp"
