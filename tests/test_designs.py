"""Designed for your pile: the concept stream, the solver in the loop, checks, repair, the look check, the call budget and the SSE endpoint."""

import asyncio
import json

import pytest

from backend import builds_api
from backend.core.ideas import IdeasError
from backend.designs import concepts as cn
from backend.designs import pipeline
from backend.designs.affordance import summarize
from tests.test_builds import INV, client, lib, parts_for, service  # noqa: F401  (fixtures: a fake LDraw library, the app with a small data set)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def three_calls_allowed(monkeypatch):
    """Production allows ONE Gemini call per request (DESIGN_MAX_CALLS=1); most tests here exercise the repair call and the look check, so they allow three."""
    monkeypatch.setattr(pipeline, "MAX_CALLS", 3)


@pytest.fixture(autouse=True)
def unscaled_sketches(monkeypatch):
    """Sketches are enlarged to the pile in production (tested in test_design_solver); here they stay as written so every solve is tiny and exact."""
    from backend.designs import sketch as sk
    monkeypatch.setattr(sk, "fit_scale", lambda s, *a, **k: (s, 1, 1))
    monkeypatch.setattr(cn, "SLACK_LOW", 0.0)                                  # the tiny test sketches are not the pile's exact size budget
    monkeypatch.setattr(cn, "SLACK_HIGH", 1e9)


def concept(name, layers):
    return json.dumps({"name": name, "description": f"a {name.lower()}", "sketch": {"layers": layers}})


WALL = concept("Red Gatepost", [["####", "####"], ["####", "####"], ["..##", "..##"]])           # 4x2 base, a 2x2 on top
TOWER = concept("Little Lighthouse", [["##", "##"], ["##", "##"], ["##", "##"], [".#", ".."]])
STEP = concept("Sunny Steps", [["####", "####"], ["#..#", "#..#"]])
HUGE = concept("Giant Castle Gate", [["#" * 12] * 12] * 4)                                       # far more than the pile owns
FLOATER = concept("Hovering Block", [["#...", "...."], ["...#", "...."]])
BAD = json.dumps({"name": "Broken", "sketch": {"layers": "nope"}})


class FakeLLM:
    """Gemini stand-in: `rounds` are the texts the stream calls return (split into small chunks); `looks` the look check's choices."""

    def __init__(self, rounds, looks=None, fail_stream=False, fail_look=False):
        self.rounds, self.looks, self.fail_stream, self.fail_look = list(rounds), looks, fail_stream, fail_look
        self.stream_calls, self.look_calls, self.prompts, self.schemas = 0, 0, [], []

    async def stream(self, system, user, max_tokens=8192, schema=None):
        self.schemas.append(schema)
        self.stream_calls += 1
        self.prompts.append(user)
        if self.fail_stream:
            raise IdeasError("The AI service is at its usage limit right now.", 429)
        text = self.rounds.pop(0)
        for i in range(0, len(text), 23):                      # chunks that cut objects in the middle
            await asyncio.sleep(0)
            yield text[i:i + 23]

    async def complete_with_images(self, system, user, schema, images):
        self.look_calls += 1
        if self.fail_look:
            raise IdeasError("quota", 429)
        assert images and images[0][:4] == b"\x89PNG"
        names = schema["properties"]["answers"]["items"]["properties"]["choice"]["enum"]
        choices = self.looks(names[:-1]) if self.looks else names[:-1]
        return json.dumps({"answers": [{"number": i + 1, "choice": c, "child_yes": c == n} for i, (c, n) in enumerate(zip(choices, names[:-1]))]})


def collect(svc, llm, parts=None, match_color=True):
    async def go():
        out = []
        async for chunk in pipeline.stream_designs(svc, parts or parts_for(), llm, match_color):
            ev = chunk.split("\n")[0][len("event: "):]
            out.append((ev, json.loads(chunk.split("\n")[1][len("data: "):])))
        return out
    return run(go())


def kinds(events):
    return [e for e, _ in events]


# ---------------------------------------------------------------- the concept stream
def test_object_stream_takes_objects_from_any_layout_and_any_chunking():
    text = '```json\n{"a": 1, "s": "}{ \\" x"}\n{"b": [1, 2,],\n "c": {"d": 3}}\nsome words {"e": 5}```'
    for size in (1, 3, 7, 1000):
        stream = cn.ObjectStream()
        got = []
        for i in range(0, len(text), size):
            got += stream.feed(text[i:i + size])
        assert got == [{"a": 1, "s": '}{ " x'}, {"b": [1, 2], "c": {"d": 3}}, {"e": 5}], size
    bad = cn.ObjectStream()
    assert bad.feed('{"a": nope}') == [] and bad.errors


def test_the_prompt_says_the_pile_is_small_and_mixed_and_gives_the_volume_and_examples():
    aff = summarize(INV)
    ex = [{"title": "Desert Rover", "parts": 35, "filled": 4, "sketch": {"layers": [["##", "##"]]}}]
    text = cn.build_prompt(aff, True, ex)
    assert "1 x 4x2 brick in Red" not in text and "4 x 4x2 brick" in text and "in Red" in text and "cells" in text and "Desert Rover" in text
    assert "JSON array of 7 concepts" in text and "layers" in text and "x=" not in text
    assert "only a wish" in cn.build_prompt(aff, False) and "a colour is used only as often as listed" in text and "palette" in text
    for needle in ("6-20 pieces", "1-2 of each", "duck", "tree", "boat", "house", "robot", "flower", "75%-100%", "few bricks", "like a child", "no wheel parts", "different bounding box"):
        assert needle in cn.SYSTEM_PROMPT, needle
    rep = cn.repair_prompt(aff, True, [("Giant Castle Gate", "the pile can fill only 10% of the sketch")], [], 4)
    assert "Giant Castle Gate" in rep and "only 10%" in rep and "4 NEW concepts" in rep


def test_clean_concept_enforces_the_size_limits_and_names():
    limits = {"width": 6, "depth": 6, "layers": 4}
    ok, why = cn.clean(json.loads(WALL), limits)
    assert ok.name == "Red Gatepost" and ok.sketch.filled == 20 and why is None
    assert "limits" in cn.clean(json.loads(HUGE), limits)[1] and "layers" in cn.clean(json.loads(BAD), limits)[1]
    assert "name" in cn.clean({"name": "x", "sketch": {"layers": [["####"]]}}, limits)[1] and "not an object" in cn.clean([], limits)[1]


# ---------------------------------------------------------------- the pipeline
def test_designs_stream_out_as_they_pass_and_the_look_check_follows(lib, tmp_path):
    svc = service(lib, tmp_path)
    llm = FakeLLM(["\n".join([WALL, TOWER, STEP]) + "\n"])
    ev = collect(svc, llm)
    assert kinds(ev)[0] == "start" and kinds(ev)[-1] == "done"
    assert kinds(ev).count("design") == 3 and kinds(ev).count("verdict") == 3
    assert kinds(ev).index("verdict") > max(i for i, k in enumerate(kinds(ev)) if k == "design")         # the look check is after the designs
    done = ev[-1][1]
    assert done["designs"] == 3 and done["calls"] == 2 and llm.stream_calls == 1 and llm.look_calls == 1       # concepts + look check: no repair needed
    for _, d in (e for e in ev if e[0] == "design"):
        b = d["build"]
        assert b["kind"] == "ai" and b["step_count"] >= 1 and b["thumb_url"] and b["design"]["coverage"] >= 0.6 and b["part_count"] >= 3
        assert svc.mpd(b["id"]).startswith("0 FILE model.ldr")
    times = [d["t_ms"] for e, d in ev]
    assert times == sorted(times) and done["stats"]["first_design_ms"] <= done["stats"]["last_design_ms"]


def test_every_design_uses_only_the_pile_in_both_colour_modes(lib, tmp_path):
    from backend.core.inventory_check import can_build
    from backend.core import ldrconv
    svc = service(lib, tmp_path)
    for match_color in (True, False):
        ev = collect(svc, FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"]), match_color=match_color)
        designs = [d["build"] for e, d in ev if e == "design"]
        assert designs and all(d["mixed_colours"] == (not match_color) for d in designs)
        for d in designs:
            build = ldrconv.ldr_to_build(svc._path(d["id"], "ldr").read_text(encoding="utf-8"), INV, d["title"])
            assert can_build(build.parts, INV, True) and can_build(build.parts, INV, match_color)


def test_a_concept_the_pile_cannot_build_is_dropped_with_a_reason_and_one_repair_call_is_made(lib, tmp_path):
    svc = service(lib, tmp_path)
    llm = FakeLLM([f"{WALL}\n{HUGE}\n{BAD}\n{FLOATER}\n", f"{TOWER}\n{STEP}\n"])
    ev = collect(svc, llm)
    done = ev[-1][1]
    assert llm.stream_calls == 2 and llm.look_calls == 1 and done["calls"] == 3                       # concepts, repair, look check
    assert any("Giant Castle Gate" in f["concept"] for f in done["failures"]) and any(f["concept"] == "Broken" for f in done["failures"])
    assert "repair" not in llm.prompts[0] and "could not be built" in llm.prompts[1] and "Giant Castle Gate" in llm.prompts[1]
    assert {d["build"]["title"] for e, d in ev if e == "design"} >= {"Red Gatepost", "Little Lighthouse", "Sunny Steps"}
    assert [d["build"]["design"]["round"] for e, d in ev if e == "design"][-1] == 2


def test_with_three_designs_in_round_one_there_is_no_repair_call(lib, tmp_path):
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n{HUGE}\n"])
    collect(service(lib, tmp_path), llm)
    assert llm.stream_calls == 1


def test_never_more_than_three_gemini_calls(lib, tmp_path):
    llm = FakeLLM([f"{HUGE}\n", f"{HUGE}\n", f"{HUGE}\n"])
    ev = collect(service(lib, tmp_path), llm)
    assert ev[-1][1]["calls"] <= pipeline.MAX_CALLS and llm.stream_calls <= 2 and llm.look_calls == 0
    assert any(e == "note" and d["kind"] == "no_design" and d["reasons"] for e, d in ev)


def test_a_design_the_look_check_does_not_recognise_is_removed(lib, tmp_path):
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"], looks=lambda names: ["none of these" if n == "Little Lighthouse" else n for n in names])
    ev = collect(service(lib, tmp_path), llm)
    verdicts = {d["title"]: d["ok"] for e, d in ev if e == "verdict"}
    assert verdicts == {"Red Gatepost": True, "Little Lighthouse": False, "Sunny Steps": True}
    assert ev[-1][1]["designs"] == 2 and ev[-1][1]["stats"]["look_ok"] == 2


def test_a_failing_look_check_keeps_the_designs(lib, tmp_path):
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"], fail_look=True))
    assert ev[-1][1]["designs"] == 3 and kinds(ev).count("verdict") == 0


def test_two_concepts_with_the_same_shape_are_shown_once(lib, tmp_path):
    twin = concept("Twin Block", [["####", "####"], ["####", "####"], ["..##", "..##"]])
    ev = collect(service(lib, tmp_path), FakeLLM([f"{WALL}\n{twin}\n{TOWER}\n{STEP}\n"]))
    titles = [d["build"]["title"] for e, d in ev if e == "design"]
    assert "Red Gatepost" in titles and "Twin Block" not in titles
    assert any("same shape" in f["why"] for f in ev[-1][1]["failures"])


def test_when_gemini_is_unavailable_the_stream_says_so_and_ends_cleanly(lib, tmp_path):
    ev = collect(service(lib, tmp_path), FakeLLM([], fail_stream=True))
    assert kinds(ev) == ["start", "note", "done"] and ev[1][1]["kind"] == "ai_unavailable" and ev[-1][1]["designs"] == 0
    ev = collect(service(lib, tmp_path), None)
    assert kinds(ev) == ["note", "done"] and "GEMINI_API_KEY" in ev[0][1]["message"]


def test_pieces_the_solver_cannot_place_are_reported(lib, tmp_path):
    ev = collect(service(lib, tmp_path), FakeLLM([]), parts=[{"part_num": "99999", "name": "Mystery", "color": "Red", "qty": 3}])
    assert ev[0][0] == "note" and ev[0][1]["kind"] == "no_parts" and ev[-1][0] == "done"


# ---------------------------------------------------------------- the endpoint
def test_the_endpoint_streams_server_sent_events(client, monkeypatch):
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"])
    monkeypatch.setattr(builds_api, "_design_llm", lambda: llm)
    r = client.post("/api/designs/stream", json={"parts": parts_for(), "match_color": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-cache"
    events = [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in r.text.strip().split("\n\n")]
    assert events[0][0] == "start" and events[-1][0] == "done" and sum(e == "design" for e, _ in events) == 3
    assert client.get(next(d["build"] for e, d in events if e == "design")["thumb_url"]).content[:4] == b"\x89PNG"
    assert client.post("/api/designs/stream", json={"parts": []}).status_code == 422
