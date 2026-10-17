import asyncio
import logging
import threading
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from backend.core import gemini_budget as gb
from backend.core import gemini_detect as gd
from backend.core.gemini_budget import BUILD, DETECTION, BudgetConfig, BudgetRefused, GeminiBudget

LA = ZoneInfo("America/Los_Angeles")
NOON = datetime(2026, 3, 10, 12, 0, tzinfo=LA).timestamp()


class Clock:
    def __init__(self, t=NOON):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, clock=None, **cfg):
    cfg.setdefault("default_rpm", 1000)
    cfg.setdefault("default_rpd", 100)
    cfg.setdefault("detection_daily_share", 1.0)
    cfg.setdefault("build_reserve", 0)
    return GeminiBudget(tmp_path / "b.sqlite", BudgetConfig(**cfg), clock=clock or Clock())


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- counting and limits
def test_usage_is_counted_per_model(tmp_path):
    b = make(tmp_path, default_rpd=3)
    assert all(b.try_acquire(BUILD, "a") for _ in range(3))
    assert not b.try_acquire(BUILD, "a")                       # model a is used up...
    assert b.try_acquire(BUILD, "b")                           # ...model b is not
    s = b.status(["a", "b", "c"])
    assert (s["a"]["used"], s["a"]["remaining"]) == (3, 0)
    assert s["b"]["used"] == 1 and s["c"]["used"] == 0


def test_limits_come_from_config_with_per_model_overrides(tmp_path):
    b = make(tmp_path, model_limits={"small": {"rpm": 2, "rpd": 2}})
    assert b.cfg.limits("small") == (2, 2) and b.cfg.limits("other") == (1000, 100)
    assert b.try_acquire(BUILD, "small") and b.try_acquire(BUILD, "small") and not b.try_acquire(BUILD, "small")


def test_daily_counter_resets_at_midnight_pacific(tmp_path):
    clock = Clock(datetime(2026, 3, 10, 23, 59, tzinfo=LA).timestamp())
    b = make(tmp_path, clock, default_rpd=2)
    assert b.try_acquire(BUILD, "m") and b.try_acquire(BUILD, "m") and not b.try_acquire(BUILD, "m")
    d = b.acquire(BUILD, "m")
    assert "resets 2026-03-11T00:00:00-07:00" in d.reason
    clock.t = datetime(2026, 3, 11, 0, 1, tzinfo=LA).timestamp()
    assert b.try_acquire(BUILD, "m")
    assert b.status()["m"]["used"] == 1


def test_reset_time_is_configurable(tmp_path):
    clock = Clock(datetime(2026, 3, 10, 8, 59, tzinfo=ZoneInfo("UTC")).timestamp())
    b = make(tmp_path, clock, default_rpd=1, reset_time="09:00", reset_tz="UTC")
    assert b.try_acquire(BUILD, "m") and not b.try_acquire(BUILD, "m")
    clock.t += 120
    assert b.try_acquire(BUILD, "m")


def test_detection_is_capped_at_its_share_of_the_daily_limit(tmp_path):
    b = make(tmp_path, default_rpd=100, detection_daily_share=0.3)
    assert sum(b.try_acquire(DETECTION, "m") for _ in range(50)) == 30
    d = b.acquire(DETECTION, "m")
    assert not d and d.reason == "detection budget exhausted"
    assert b.try_acquire(BUILD, "m")                           # builds are unaffected
    assert b.status()["m"]["detection_used"] == 30 and b.status()["m"]["detection_cap"] == 30


def test_detection_cannot_dip_into_the_build_reserve(tmp_path):
    b = make(tmp_path, default_rpd=100, detection_daily_share=1.0, build_reserve=20)
    assert sum(b.try_acquire(DETECTION, "m") for _ in range(100)) == 80
    assert not b.try_acquire(DETECTION, "m")
    assert sum(b.try_acquire(BUILD, "m") for _ in range(100)) == 20     # builds can use what is reserved


def test_reserve_defaults_to_a_share_of_the_daily_limit(tmp_path):
    b = make(tmp_path, default_rpd=50, build_reserve=None, build_reserve_frac=0.2)
    assert b.cfg.reserve(50) == 10
    assert sum(b.try_acquire(DETECTION, "m") for _ in range(50)) == 40


def test_detection_is_refused_at_once_when_the_minute_window_is_full(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock, default_rpm=2)
    assert b.try_acquire(DETECTION, "m") and b.try_acquire(DETECTION, "m")
    assert b.acquire(DETECTION, "m").reason == "rpm window full"
    with pytest.raises(BudgetRefused) as e:
        run(b.acquire_async(DETECTION, "m"))                   # does not wait
    assert e.value.reason == "rpm window full"
    clock.t += 61                                              # sliding window: the old requests have aged out
    assert b.try_acquire(DETECTION, "m")


def test_a_build_waits_briefly_for_an_rpm_slot(tmp_path, monkeypatch):
    clock = Clock()
    b = make(tmp_path, clock, default_rpm=1, build_max_wait=70)
    assert b.try_acquire(BUILD, "m")
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        clock.t += s
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    run(b.acquire_async(BUILD, "m"))
    assert len(slept) == 1 and 59 < slept[0] < 61
    assert b.status()["m"]["used"] == 2


def test_a_build_fails_clearly_when_the_wait_would_be_too_long_or_the_day_is_used_up(tmp_path):
    b = make(tmp_path, default_rpm=1, build_max_wait=5)
    assert b.try_acquire(BUILD, "m")
    with pytest.raises(BudgetRefused) as e:
        run(b.acquire_async(BUILD, "m"))
    assert e.value.reason == "rpm window full" and e.value.retry_after > 5
    b2 = make(tmp_path / "other", default_rpd=1)
    assert b2.try_acquire(BUILD, "m")
    with pytest.raises(BudgetRefused, match="daily limit reached"):
        run(b2.acquire_async(BUILD, "m"))


# ---------------------------------------------------------------- 429 handling
def test_mark_exhausted_blocks_the_model_until_the_given_time_or_the_next_reset(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock)
    b.mark_exhausted("m", clock.t + 30)
    d = b.acquire(BUILD, "m")
    assert not d and d.reason.startswith("model exhausted until") and 29 < d.retry_after <= 30
    assert b.try_acquire(BUILD, "other")
    clock.t += 31
    assert b.try_acquire(BUILD, "m")
    b.mark_exhausted("m")                                      # no delay given: until the daily reset
    assert not b.try_acquire(DETECTION, "m")
    clock.t = datetime(2026, 3, 11, 0, 1, tzinfo=LA).timestamp()
    assert b.try_acquire(BUILD, "m")
    assert b.status()["m"]["exhausted_until"] is None


class QuotaError(Exception):
    code, status = 429, "RESOURCE_EXHAUSTED"

    def __init__(self, message="quota"):
        super().__init__(message)
        self.message = message


class ServerError(Exception):
    code, status, message = 503, "UNAVAILABLE", "overloaded"


class FakeClient:
    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._gen))

    async def _gen(self, model, **kw):
        self.calls.append(model)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_a_429_marks_the_model_exhausted_using_the_retry_delay_when_given(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock)
    client = FakeClient(QuotaError("Quota exceeded. Please retry in 34.5s."))
    with pytest.raises(QuotaError):
        run(gb.generate(client, b, DETECTION, "m"))
    d = b.acquire(BUILD, "m")
    assert not d and 34 < d.retry_after < 35


def test_a_429_without_a_delay_blocks_until_the_next_reset(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock)
    with pytest.raises(QuotaError):
        run(gb.generate(FakeClient(QuotaError()), b, BUILD, "m"))
    assert b.status()["m"]["exhausted_until"].startswith("2026-03-11T00:00:00")


def test_generate_counts_every_attempt_including_retries(tmp_path):
    b = make(tmp_path)
    client = FakeClient(ServerError(), "ok")
    assert run(gb.generate(client, b, BUILD, "m", retries=1)) == "ok"
    assert client.calls == ["m", "m"] and b.status()["m"]["used"] == 2
    client = FakeClient(ServerError(), ServerError())
    with pytest.raises(ServerError):
        run(gb.generate(client, b, BUILD, "m", retries=1))
    assert b.status()["m"]["used"] == 4


def test_generate_makes_no_api_call_when_refused(tmp_path):
    b = make(tmp_path, default_rpd=0)
    client = FakeClient("never")
    with pytest.raises(BudgetRefused):
        run(gb.generate(client, b, BUILD, "m"))
    assert client.calls == []


# ---------------------------------------------------------------- persistence and concurrency
def test_counters_and_exhaustion_survive_a_restart(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock, default_rpd=5)
    for _ in range(3):
        b.try_acquire(DETECTION, "m")
    b.mark_exhausted("x", clock.t + 600)
    again = GeminiBudget(tmp_path / "b.sqlite", b.cfg, clock=clock)
    s = again.status()
    assert s["m"]["used"] == 3 and s["m"]["detection_used"] == 3
    assert not again.try_acquire(BUILD, "x")


def test_concurrent_acquires_never_exceed_the_limits(tmp_path):
    b = make(tmp_path, default_rpd=25, default_rpm=1000)
    others = [GeminiBudget(tmp_path / "b.sqlite", b.cfg, clock=b.clock) for _ in range(3)]   # "other processes"
    granted = []

    def worker(inst):
        for _ in range(20):
            granted.append(inst.try_acquire(BUILD, "m"))
    threads = [threading.Thread(target=worker, args=(i,)) for i in [b, *others]]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(granted) == 25 and b.status()["m"]["used"] == 25


def test_concurrent_acquires_respect_the_minute_window(tmp_path):
    b = make(tmp_path, default_rpd=1000, default_rpm=7)
    granted = []
    threads = [threading.Thread(target=lambda: granted.append(b.try_acquire(DETECTION, "m"))) for _ in range(30)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(granted) == 7


# ---------------------------------------------------------------- integration with detection
class FakeDetector:
    model = "gemini-x"

    def __init__(self, budget, boxes=((10, 10, 100, 100),), fail=None):
        self.budget, self.boxes, self.fail, self.calls = budget, list(boxes), fail, 0

    async def detect(self, img):
        self.calls += 1
        await self.budget.acquire_async(DETECTION, self.model)       # as GeminiDetector does per request
        if self.fail:
            raise self.fail
        return list(self.boxes), self.model


IMG = np.full((400, 400, 3), 255, np.uint8)
CV = lambda img: [(200, 200, 50, 50)]                                # noqa: E731


@pytest.mark.parametrize("mode", ["hybrid", "gemini"])
def test_detection_budget_exhausted_runs_opencv_only_and_logs_why(tmp_path, caplog, mode):
    b = make(tmp_path, default_rpd=10, detection_daily_share=0.0)
    det = FakeDetector(b)
    with caplog.at_level(logging.INFO, logger="lego.budget"), caplog.at_level(logging.INFO, logger="lego.detect"):
        d = run(gd.detect_pieces(IMG, mode, det, opencv=CV))
    assert d.mode == "opencv" and d.boxes == [(200, 200, 50, 50)] and det.calls == 0
    assert d.note == "gemini skipped: detection budget exhausted"
    assert "gemini skipped: detection budget exhausted" in caplog.text


def test_rpm_window_full_and_model_exhausted_reasons_are_reported(tmp_path):
    clock = Clock()
    b = make(tmp_path, clock, default_rpm=1)
    b.try_acquire(BUILD, "gemini-x")
    d = run(gd.detect_pieces(IMG, "hybrid", FakeDetector(b), opencv=CV))
    assert d.note == "gemini skipped: rpm window full" and d.mode == "opencv"
    b2 = make(tmp_path / "2", clock)
    b2.mark_exhausted("gemini-x", clock.t + 90)
    d = run(gd.detect_pieces(IMG, "hybrid", FakeDetector(b2), opencv=CV))
    assert d.note.startswith("gemini skipped: model exhausted until ")


def test_refusal_in_the_middle_of_the_chain_falls_back_to_opencv(tmp_path):
    b = make(tmp_path)
    det = FakeDetector(b, fail=BudgetRefused("detection budget exhausted"))
    d = run(gd.detect_pieces(IMG, "hybrid", det, opencv=CV))
    assert d.mode == "opencv" and d.note == "gemini skipped: detection budget exhausted" and det.calls == 1


def test_with_budget_available_hybrid_still_merges(tmp_path):
    d = run(gd.detect_pieces(IMG, "hybrid", FakeDetector(make(tmp_path)), opencv=CV))
    assert d.mode == "hybrid" and len(d.boxes) == 2


def test_the_real_detector_counts_every_attempt_of_its_fallback_chain(tmp_path):
    pytest.importorskip("google.genai")
    b = make(tmp_path, default_rpd=2)
    det = gd.GeminiDetector("key", "primary", "fallback", budget=b)
    from google.genai import errors
    det.client = FakeClient(errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}}),
                            SimpleNamespace(prompt_feedback=None, text='{"pieces": []}'))
    boxes, model = run(det.detect(IMG))
    assert model == "fallback" and det.client.calls == ["primary", "fallback"]
    assert b.status()["primary"]["used"] == 1 and b.status()["fallback"]["used"] == 1
    det.client = FakeClient("unused")
    b.try_acquire(DETECTION, "primary")
    with pytest.raises(BudgetRefused):                                  # primary is now at its daily limit
        run(det.detect(IMG))
    assert det.client.calls == []


def test_a_refused_build_gives_the_ui_a_clear_error(tmp_path):
    from backend.core.ideas import GeminiLLM, IdeasError
    b = make(tmp_path, default_rpd=0)
    llm = GeminiLLM("key", "m", budget=b)
    with pytest.raises(IdeasError) as e:
        run(llm.complete("sys", "user", {}))
    assert e.value.status == 429 and "usage limit" in str(e.value)
