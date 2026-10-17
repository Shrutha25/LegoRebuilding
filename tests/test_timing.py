"""Per-request timing traces and the p50/p95 report."""

import asyncio
import json
import time

import pytest

from backend import config, timing
from backend.timing_report import load, pct, render, summarize


def test_stages_nest_add_up_and_are_logged_as_one_json_line(tmp_path, monkeypatch):
    path = tmp_path / "t.jsonl"
    monkeypatch.setattr(config, "TIMING_LOG", str(path))
    with timing.request("demo", who="test"):
        with timing.stage("outer"):
            with timing.stage("inner", n=1):
                time.sleep(0.01)
            with timing.stage("inner", n=2):
                time.sleep(0.01)
        timing.count("gemini_calls")
        timing.count("gemini_calls", 2)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == "demo" and r["meta"] == {"who": "test"} and r["counters"] == {"gemini_calls": 3}
    assert [s["stage"] for s in r["stages"]] == ["inner", "inner", "outer"] and r["stages"][0]["n"] == 1
    assert r["sums"]["inner"] >= 15 and r["sums"]["outer"] >= r["sums"]["inner"] and r["total_ms"] >= r["sums"]["outer"]


def test_without_a_trace_stage_and_count_do_nothing():
    with timing.stage("alone"):
        timing.count("x")
    assert timing.current() is None


def test_the_trace_follows_tasks_and_threads_and_the_file_can_be_switched_off(tmp_path, monkeypatch):
    from starlette.concurrency import run_in_threadpool
    monkeypatch.setattr(config, "TIMING_LOG", "")

    async def go():
        trace, token = timing.begin("async")

        async def child():
            with timing.stage("in task"):
                await asyncio.sleep(0.005)

        def work():
            with timing.stage("in thread"):
                time.sleep(0.005)

        await asyncio.gather(child(), run_in_threadpool(work))
        return timing.end(trace, token)

    row = asyncio.run(go())
    assert {s["stage"] for s in row["stages"]} == {"in task", "in thread"}
    assert not (tmp_path / "timings.jsonl").exists()


def test_every_gemini_request_is_a_stage_and_counts_as_a_call():
    from backend.core import gemini_budget

    class Models:
        async def generate_content(self, model, **kw):
            await asyncio.sleep(0.005)
            return "answer"

    class Client:
        class aio:
            models = Models()

    async def go():
        trace, token = timing.begin("gemini")
        assert await gemini_budget.generate(Client, None, "build", "m", contents="x") == "answer"
        return timing.end(trace, token)

    row = asyncio.run(go())
    assert row["counters"] == {"gemini_calls": 1, "gemini_calls.build": 1}
    assert row["stages"][0]["stage"] == "gemini.build" and row["stages"][0]["model"] == "m"


def test_a_post_to_the_api_gets_a_trace_named_after_its_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main
    path = tmp_path / "t.jsonl"
    monkeypatch.setattr(config, "TIMING_LOG", str(path))
    with TestClient(main.app) as c:
        c.get("/api/health")                                                   # a GET is not traced
        c.post("/api/parts/resolve", json={"parts": ["3001"]})
    names = [json.loads(x)["name"] for x in path.read_text(encoding="utf-8").splitlines()]
    assert names == ["/api/parts/resolve"]


def test_percentiles_and_the_report(tmp_path):
    assert pct([1, 2, 3, 4, 5], 0.5) == 3 and pct([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8) and pct([], 0.5) == 0
    rows = [{"name": "/api/match", "total_ms": ms, "counters": {"gemini_calls": g}, "sums": {"sets.match": ms / 2, "map_parts": 5}}
            for ms, g in ((100, 0), (200, 1), (300, 0), (400, 2))]
    s = summarize(rows)["/api/match"]
    assert s["n"] == 4 and pct(s["total"], .5) == 250 and s["stages"]["sets.match"] == [50, 100, 150, 200]
    text = render(summarize(rows))
    assert "/api/match" in text and "sets.match" in text and "250" in text
    p = tmp_path / "l.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    assert len(load(p)) == 4 and len(load(p, name="nope")) == 0 and len(load(p, last=2)) == 2
