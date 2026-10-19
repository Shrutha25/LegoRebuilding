"""Build suggestions are cheap: one Gemini call per request, minimal thinking, a short prompt, a small JSON schema, and a real recorded answer replayed offline."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from backend import config
from backend.core import gemini_tape
from backend.core.ideas import GeminiLLM
from backend.designs import concepts as cn
from backend.designs import pipeline
from backend.designs.affordance import summarize
from tests.test_builds import INV, client, lib, parts_for, service  # noqa: F401
from tests.test_designs import FakeLLM, HUGE, STEP, TOWER, WALL, collect, kinds  # noqa: F401  (this file does NOT use test_designs' autouse fixtures: real defaults)


def test_the_defaults_are_one_call_no_on_demand_recipe_writing_and_minimal_thinking():
    assert config.DESIGN_MAX_CALLS == 1 and config.RECIPE_GEN_PER_REQUEST == 0 and config.GEMINI_THINKING == "minimal"
    assert pipeline.MAX_CALLS == 1


def test_a_design_request_makes_exactly_one_gemini_call_even_when_nothing_fits(lib, tmp_path, monkeypatch):
    from backend.designs import sketch as sk
    monkeypatch.setattr(sk, "fit_scale", lambda s, *a, **k: (s, 1, 1))
    monkeypatch.setattr(cn, "SLACK_HIGH", 1e9)
    llm = FakeLLM([f"{WALL}\n{TOWER}\n{STEP}\n"])
    ev = collect(service(lib, tmp_path), llm)
    done = ev[-1][1]
    assert llm.stream_calls == 1 and llm.look_calls == 0 and done["calls"] == 1 and "verdict" not in kinds(ev) and kinds(ev).count("design") == 3
    assert llm.schemas == [cn.SCHEMA]
    none = FakeLLM([f"{HUGE}\n", f"{HUGE}\n"])                                                       # nothing fits: still no repair call
    ev = collect(service(lib, tmp_path), none)
    assert none.stream_calls == 1 and ev[-1][1]["calls"] == 1 and any(e == "note" for e, _ in ev)


def test_the_concept_prompt_is_short_and_the_schema_is_small():
    aff = summarize(INV)
    text = cn.SYSTEM_PROMPT + cn.build_prompt(aff, True, [{"title": "Rover", "parts": 12, "filled": 30, "sketch": {"layers": [["####", "####"]]}}] * pipeline.EXAMPLES_IN_PROMPT)
    assert len(text) / 3.5 < 1100                                                                     # about 900 tokens for the 19-piece test pile (it was about 2,300)
    assert len(json.dumps(cn.SCHEMA)) < 700 and cn.SCHEMA["type"] == "array" and set(cn.SCHEMA["items"]["required"]) == {"name", "type", "layers"}


def test_the_flat_answer_format_is_read_like_the_nested_one():
    limits = {"width": 8, "depth": 8, "layers": 6}
    flat, why = cn.clean({"name": "Pond Duck", "type": "animal", "description": "d", "palette": ["a=Red", "b=blue"], "layers": [["aaaa", "aabb"], ["aa..", "...."]]}, limits, colours=["Red", "Blue"])
    assert why is None and flat.kind == "animal" and flat.sketch.colours[(0, 0, 0)] == "Red" and flat.sketch.colours[(2, 1, 0)] == "Blue"
    nested, _ = cn.clean(json.loads(WALL), limits)
    assert nested.name == "Red Gatepost"


def _llm_with(models):
    llm = GeminiLLM("k", "gemini-flash-lite-latest")
    llm.client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return llm


class Models:
    def __init__(self, reject_thinking=False):
        self.configs, self.reject_thinking = [], reject_thinking

    async def generate_content_stream(self, model, **kw):
        from google.genai import errors
        self.configs.append(kw["config"])
        if self.reject_thinking and kw["config"].thinking_config is not None:
            raise errors.ClientError(400, {"error": {"code": 400, "message": "Thinking level minimal is not supported for this model", "status": "INVALID_ARGUMENT"}})

        async def gen():
            yield SimpleNamespace(text="[]")
        return gen()


def stream(llm):
    async def go():
        return [t async for t in llm.stream("s", "u", schema=cn.SCHEMA)]
    return asyncio.run(go())


def test_the_concept_call_asks_for_minimal_thinking_and_the_small_schema():
    m = Models()
    stream(_llm_with(m))
    cfg = m.configs[0]
    assert cfg.thinking_config.thinking_level.value.lower() == "minimal" and cfg.response_mime_type == "application/json" and cfg.response_json_schema == cn.SCHEMA


def test_a_model_that_rejects_the_thinking_level_is_asked_again_without_it_and_remembered():
    m = Models(reject_thinking=True)
    llm = _llm_with(m)
    assert stream(llm) == ["[]"] and [c.thinking_config is None for c in m.configs] == [False, True]
    stream(llm)
    assert [c.thinking_config is None for c in m.configs] == [False, True, True]                      # the next request goes straight to no thinking


def test_thinking_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_THINKING", "none")
    m = Models()
    stream(_llm_with(m))
    assert m.configs[0].thinking_config is None


def test_a_real_recorded_concept_answer_replays_offline_through_the_whole_pipeline(lib, tmp_path):
    """The answer is a real Gemini response, recorded once (GEMINI_LIVE=1 records it: one request). If the prompt changes the recording no longer matches: the
    test skips and says how to record it again."""
    llm = GeminiLLM(config.GEMINI_API_KEY or "test-key", "gemini-flash-lite-latest")          # the key only matters when recording
    aff = summarize(INV)
    prompt = cn.build_prompt(aff, True, [])
    try:
        chunks = asyncio.run(_collect(llm.stream(cn.SYSTEM_PROMPT, prompt, schema=cn.SCHEMA)))
    except gemini_tape.NoRecording:
        pytest.skip("no recording for the current prompt: GEMINI_LIVE=1 python -m pytest tests/test_designs_cost.py -k recorded (one real request)")
    parser = cn.ObjectStream()
    concepts = [cn.clean(o, aff.limits, 1, aff.colours, aff.budget) for obj_chunk in chunks for o in parser.feed(obj_chunk)]
    assert len(concepts) >= 3 and sum(1 for c, _ in concepts if c is not None) >= 1
    ev = collect(service(lib, tmp_path), llm)
    assert ev[-1][0] == "done" and ev[-1][1]["calls"] == 1 and ev[-1][1]["stats"]["concepts"] >= 3


async def _collect(agen):
    return [x async for x in agen]
