"""Record / replay of Gemini responses, so tests never spend quota.

    tests replace `genai.Client` with `client_factory(real_client_class)` (tests/conftest.py, autouse):
        default          every request is answered from tests/fixtures/gemini_tape/<kind>-<hash>.json, keyed by a hash of the model, the system prompt, the
                         contents (images by their bytes' SHA-1), the schema and the generation settings. A request that has no recording raises
                         `NoRecording`: it never reaches the network.
        GEMINI_LIVE=1    requests go to the real API (a real key is needed) and every answer is written to the tape for next time, but at most
                         LIVE_CAP (5) requests per run: the sixth raises `LiveCapExceeded`. The cap is a constant, not a setting.

Only the part of a response the app reads is stored: `.text` (and the chunk texts of a stream); `prompt_feedback` is an empty object.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

TAPE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "gemini_tape"
LIVE_CAP = 5                                     # real requests allowed in one test run when GEMINI_LIVE=1


class NoRecording(AssertionError):
    """A test asked Gemini for something that was never recorded. Record it once with GEMINI_LIVE=1 (at most LIVE_CAP calls per run)."""


class LiveCapExceeded(AssertionError):
    """More than LIVE_CAP real Gemini requests in one run."""


class _State:
    live_calls = 0


def live_enabled():
    return os.environ.get("GEMINI_LIVE") == "1"


def _plain(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (bytes, bytearray)):
        return "bytes:" + hashlib.sha1(bytes(v)).hexdigest()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0])) if x is not None}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    inline = getattr(v, "inline_data", None)
    if inline is not None and getattr(inline, "data", None) is not None:             # a Part holding an image
        return {"image": _plain(inline.data), "mime": getattr(inline, "mime_type", None)}
    if hasattr(v, "model_dump"):
        return _plain(v.model_dump(exclude_none=True))
    return str(v)


def request_key(kind, model, kwargs):
    """The hash a request is stored under: everything that decides the answer, nothing that does not (timeouts, retries)."""
    cfg = kwargs.get("config")
    payload = {"kind": kind, "model": model, "contents": _plain(kwargs.get("contents")), "config": _plain(cfg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:20]


def _path(kind, key):
    return TAPE_DIR / f"{kind}-{key}.json"


def _response(text):
    return SimpleNamespace(text=text, prompt_feedback=SimpleNamespace(block_reason=None), usage_metadata=None, candidates=[SimpleNamespace(finish_reason="STOP")])


class _Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self.chunks:
            yield _response(c)


class _Models:
    def __init__(self, real):
        self.real = real                          # the real client's `aio.models`, or None when replaying only

    async def generate_content(self, model, **kwargs):
        key = request_key("generate", model, kwargs)
        hit = self._load("generate", key)
        if hit is not None:
            return _response(hit["text"])
        resp = await self._live(model, "generate", key, lambda: self.real.generate_content(model=model, **kwargs))
        text = getattr(resp, "text", None)
        self._save("generate", key, model, {"text": text})
        return resp

    async def generate_content_stream(self, model, **kwargs):
        key = request_key("stream", model, kwargs)
        hit = self._load("stream", key)
        if hit is not None:
            return _Stream(hit["chunks"])
        stream = await self._live(model, "stream", key, lambda: self.real.generate_content_stream(model=model, **kwargs))
        chunks = []
        async for chunk in stream:
            try:
                text = chunk.text
            except Exception:                    # a chunk without text
                text = None
            if text:
                chunks.append(text)
        self._save("stream", key, model, {"chunks": chunks})
        return _Stream(chunks)

    # ---- tape
    @staticmethod
    def _load(kind, key):
        p = _path(kind, key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    @staticmethod
    def _save(kind, key, model, body):
        TAPE_DIR.mkdir(parents=True, exist_ok=True)
        _path(kind, key).write_text(json.dumps({"model": model, "kind": kind, **body}, ensure_ascii=False, indent=1), encoding="utf-8")

    async def _live(self, model, kind, key, call):
        if not live_enabled() or self.real is None:
            raise NoRecording(f"no recorded Gemini {kind} response for model {model!r} (key {key}). Tests never call Gemini live. Record it once with "
                              f"GEMINI_LIVE=1 (a real GEMINI_API_KEY; at most {LIVE_CAP} real requests per run).")
        if _State.live_calls >= LIVE_CAP:
            raise LiveCapExceeded(f"GEMINI_LIVE allows {LIVE_CAP} real Gemini requests per run; this would be number {_State.live_calls + 1}")
        _State.live_calls += 1
        return await call()


class TapeClient:
    """Stand-in for `genai.Client`: replays the tape, or (GEMINI_LIVE=1) records real answers within the cap."""

    def __init__(self, *args, real_client=None, **kwargs):
        self.aio = SimpleNamespace(models=_Models(real_client.aio.models if real_client is not None else None))


def client_factory(real_client_class):
    """The callable that replaces `genai.Client` in tests."""
    def make(*args, **kwargs):
        real = None
        if live_enabled():
            real = real_client_class(*args, **kwargs)
        return TapeClient(real_client=real)
    return make


def reset_live_counter():
    _State.live_calls = 0
