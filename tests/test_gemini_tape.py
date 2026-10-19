"""Tests never call Gemini: responses are replayed from a tape keyed by a prompt hash; live calls need GEMINI_LIVE=1 and are capped at 5 per run."""

import asyncio
import socket
from types import SimpleNamespace

import pytest

from backend.core import gemini_tape as tape
from backend.core.gemini_budget import GeminiBudget, BudgetConfig
from backend.core.ideas import GeminiLLM, IdeasError

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


@pytest.fixture
def tape_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tape, "TAPE_DIR", tmp_path / "tape")
    monkeypatch.setattr(tape._State, "live_calls", 0)
    return tmp_path / "tape"


class RealClient:
    """A stand-in for the real SDK client that counts the requests it would have sent."""
    calls = 0

    def __init__(self, *a, **k):
        self.aio = SimpleNamespace(models=self)

    async def generate_content(self, model, **kw):
        RealClient.calls += 1
        return SimpleNamespace(text=f"answer {RealClient.calls}", prompt_feedback=SimpleNamespace(block_reason=None), candidates=[SimpleNamespace(finish_reason="STOP")])

    async def generate_content_stream(self, model, **kw):
        RealClient.calls += 1

        async def gen():
            for t in ("{\"a\"", ": 1}"):
                yield SimpleNamespace(text=t)
        return gen()


def ask(llm, user="hello"):
    return asyncio.run(llm.complete("sys", user, SCHEMA))


def stream(llm, user="hello"):
    async def go():
        return [t async for t in llm.stream("sys", user)]
    return asyncio.run(go())


def test_the_suite_replaces_the_sdk_client_so_nothing_reaches_the_network(tape_dir, monkeypatch):
    real_connect = socket.socket.connect

    def no_network(self, address, *a, **k):
        if address[0] not in ("127.0.0.1", "::1", "localhost"):                 # asyncio's own loopback socket pair is fine
            raise AssertionError(f"a test opened a network connection to {address}")
        return real_connect(self, address, *a, **k)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    llm = GeminiLLM("real-looking-key", "gemini-test-model")
    assert type(llm.client).__name__ == "TapeClient"
    with pytest.raises(tape.NoRecording, match="Tests never call Gemini live"):
        ask(llm)


def test_a_live_run_records_and_a_normal_run_replays_by_prompt_hash(tape_dir, monkeypatch):
    RealClient.calls = 0
    monkeypatch.setenv("GEMINI_LIVE", "1")
    live = GeminiLLM.__new__(GeminiLLM)
    live.client = tape.client_factory(RealClient)("k")
    live.model, live.budget = "m", None
    live._types = __import__("google.genai.types", fromlist=["x"])
    live._errors = __import__("google.genai.errors", fromlist=["x"])
    live._httpx = __import__("httpx")
    first = ask(live)
    chunks = stream(live)
    assert first == "answer 1" and chunks == ['{"a"', ": 1}"] and RealClient.calls == 2 and len(list(tape_dir.glob("*.json"))) == 2
    monkeypatch.delenv("GEMINI_LIVE")                                     # replay: same prompt, same answer, no real request
    again = GeminiLLM("k", "m")
    again.client = tape.client_factory(RealClient)("k")
    assert ask(again) == "answer 1" and stream(again) == ['{"a"', ": 1}"] and RealClient.calls == 2
    with pytest.raises(tape.NoRecording):                                 # another prompt was never recorded
        ask(again, "something else")


def test_live_calls_are_capped_at_five_per_run(tape_dir, monkeypatch):
    RealClient.calls = 0
    monkeypatch.setenv("GEMINI_LIVE", "1")
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.client, llm.model, llm.budget = tape.client_factory(RealClient)("k"), "m", None
    llm._types = __import__("google.genai.types", fromlist=["x"])
    llm._errors = __import__("google.genai.errors", fromlist=["x"])
    llm._httpx = __import__("httpx")
    assert tape.LIVE_CAP == 5
    for i in range(5):
        ask(llm, f"q{i}")
    with pytest.raises(tape.LiveCapExceeded):
        ask(llm, "q5")
    assert RealClient.calls == 5


def test_the_key_depends_on_what_decides_the_answer_only():
    base = tape.request_key("generate", "m", {"contents": "hi", "config": {"a": 1}})
    assert base == tape.request_key("generate", "m", {"contents": "hi", "config": {"a": 1}, "timeout": 5})
    assert base != tape.request_key("generate", "m2", {"contents": "hi", "config": {"a": 1}})
    assert base != tape.request_key("stream", "m", {"contents": "hi", "config": {"a": 1}})
    assert base != tape.request_key("generate", "m", {"contents": "hi!", "config": {"a": 1}})
    a = tape.request_key("generate", "m", {"contents": [b"\x89PNG1", "x"]})
    assert a == tape.request_key("generate", "m", {"contents": [b"\x89PNG1", "x"]}) and a != tape.request_key("generate", "m", {"contents": [b"\x89PNG2", "x"]})


def test_after_a_429_the_model_is_marked_exhausted_and_nothing_else_is_tried(tmp_path):
    """On a quota error gemini_budget marks that model exhausted until the reset; the next request is refused locally: no API call, no other model."""
    from google.genai import errors

    class Models:
        def __init__(self):
            self.models_tried = []

        async def generate_content_stream(self, model, **kw):
            self.models_tried.append(model)
            raise errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    models = Models()
    llm = GeminiLLM("k", "gemini-flash-lite-latest", budget=GeminiBudget(tmp_path / "b.sqlite", BudgetConfig(default_rpm=100, default_rpd=100)))
    llm.client = SimpleNamespace(aio=SimpleNamespace(models=models))
    for _ in range(3):
        with pytest.raises(IdeasError):
            stream(llm)
    assert models.models_tried == ["gemini-flash-lite-latest"]                    # one real attempt, then refused locally; never another model
    assert llm.budget.status(["gemini-flash-lite-latest"])["gemini-flash-lite-latest"]["exhausted_until"]
