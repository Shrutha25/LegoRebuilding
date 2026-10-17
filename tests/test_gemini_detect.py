import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from backend import config, jobs
from backend.core import gemini_detect as gd
from backend.core.detect import find_pieces
from backend.core.gemini_detect import (Detection, DetectError, GeminiDetector, boxes_from_gemini, detect_pieces, iou,
                                        merge_boxes, pad_box)

SAMPLE = Path(__file__).resolve().parent.parent / "lego_pieces.jpg"
SHAPE = (1000, 2000, 3)          # h, w, channels: not square, so a swapped x/y shows up


# ---------------------------------------------------------------- box conversion
def test_box_2d_is_y_first_and_normalised_to_1000():
    # [ymin, xmin, ymax, xmax] = [100, 200, 300, 600] on a 2000 px wide, 1000 px high photo
    assert boxes_from_gemini([{"box_2d": [100, 200, 300, 600]}], 2000, 1000) == [(400, 100, 800, 200)]


def test_conversion_clamps_fixes_swapped_corners_and_skips_bad_boxes():
    pieces = [{"box_2d": [-50, -10, 120, 1200]},                    # out of range -> clamped to the photo
              {"box_2d": [500, 800, 100, 200]},                     # min/max swapped -> fixed
              {"box_2d": [10, 10, 11, 11]},                         # degenerate (< 12 px)
              {"box_2d": [1, 2, 3]}, {"box_2d": "nope"}, {"box_2d": ["a", 1, 2, 3]}, {"label": "no box"},
              None, "text", {"box_2d": [100, 100, 300, 300]}]
    out = boxes_from_gemini(pieces, 1000, 1000)
    assert out == [(0, 0, 1000, 120), (200, 100, 600, 400), (100, 100, 200, 200)]
    assert boxes_from_gemini(None, 100, 100) == [] and boxes_from_gemini([], 100, 100) == []


def test_float_and_string_free_coordinates_are_accepted_as_numbers():
    assert boxes_from_gemini([{"box_2d": [100.4, 200.6, 300.2, 600.9]}], 1000, 1000) == [(201, 100, 400, 200)]


def test_pad_box_adds_about_8_percent_and_stays_inside_the_photo():
    assert pad_box((100, 100, 200, 100), 0.08, SHAPE) == (84, 84, 232, 132)           # 16 px each side
    assert pad_box((0, 0, 100, 100), 0.08, SHAPE) == (0, 0, 108, 108)                 # clamped at the corner
    assert pad_box((1950, 950, 50, 50), 0.08, SHAPE) == (1946, 946, 54, 54)


def test_iou():
    a = (0, 0, 100, 100)
    assert iou(a, a) == 1.0 and iou(a, (200, 200, 10, 10)) == 0.0
    assert iou(a, (50, 0, 100, 100)) == pytest.approx(1 / 3)
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------------------------------------------------------------- merging
def test_all_gemini_boxes_are_kept_and_padded():
    g = [(100, 100, 100, 100), (500, 500, 100, 100)]
    out = merge_boxes(g, [], SHAPE)
    assert out == [pad_box(b, gd.PAD_FRAC, SHAPE) for b in g]


def test_opencv_box_is_added_only_where_gemini_found_nothing():
    g = [(100, 100, 100, 100)]
    far = (800, 700, 120, 120)                                   # nowhere near a Gemini box -> added
    same = (96, 96, 108, 108)                                    # same piece found by both -> dropped
    out = merge_boxes(g, [far, same], SHAPE)
    assert far in out and same not in out and len(out) == 2


def test_the_iou_threshold_is_0_3():
    g = [(0, 0, 100, 100)]
    pg = pad_box(g[0], gd.PAD_FRAC, SHAPE)                       # (0, 0, 108, 108) after padding
    shifted_low = (80, 0, 108, 108)                              # IoU ~0.15: below 0.3 and not contained -> added
    shifted_high = (40, 0, 108, 108)                             # IoU ~0.46: duplicate -> dropped
    assert iou(pg, shifted_low) < 0.3 < iou(pg, shifted_high)
    out = merge_boxes(g, [shifted_low, shifted_high], SHAPE)
    assert shifted_low in out and shifted_high not in out


def test_opencv_fragment_inside_a_gemini_box_is_a_duplicate_unless_the_literal_iou_rule_is_used():
    g = [(100, 100, 300, 300)]
    fragment = (150, 150, 60, 60)                                # fully inside, IoU = 0.03
    assert fragment not in merge_boxes(g, [fragment], SHAPE)                       # containment rule (default)
    assert fragment in merge_boxes(g, [fragment], SHAPE, contained=None)           # IoU-only rule adds it


def test_boxes_over_25_percent_of_the_photo_are_dropped_from_both_sources():
    giant_g, giant_cv = (0, 0, 1200, 700), (700, 100, 1200, 800)       # > 25% of 2000x1000
    ok_g, ok_cv = (100, 100, 100, 100), (1500, 800, 100, 100)
    out = merge_boxes([giant_g, ok_g], [giant_cv, ok_cv], SHAPE)
    assert len(out) == 2 and ok_cv in out and pad_box(ok_g, gd.PAD_FRAC, SHAPE) in out
    # a Gemini box that only grows past 25% once padded is dropped too
    edge = (0, 0, 1000, 490)
    assert not gd.too_big(edge, SHAPE) and merge_boxes([edge], [], SHAPE) == []


def test_merged_boxes_are_in_reading_order():
    out = merge_boxes([(1500, 100, 100, 100), (100, 120, 100, 100), (800, 600, 100, 100)], [], SHAPE)
    # first row left to right (x ~100 then ~1500), then the lower piece
    assert [round(b[0], -2) for b in out] == [100, 1500, 800]
    assert out[2][1] > out[0][1] + 300


# ---------------------------------------------------------------- detect_pieces: modes and fallbacks
class StubDetector:
    def __init__(self, boxes=None, error=None, delay=0.0):
        self.boxes, self.error, self.delay, self.calls = boxes or [], error, delay, 0

    async def detect(self, img):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return list(self.boxes), "stub-model"


IMG = np.full((1000, 2000, 3), 230, np.uint8)
CV = lambda img: [(100, 100, 120, 120), (1500, 700, 100, 100)]        # noqa: E731  stub OpenCV detector


def run(mode, detector, opencv=CV):
    return asyncio.run(detect_pieces(IMG, mode, detector, opencv=opencv))


def test_opencv_mode_never_calls_gemini():
    det = StubDetector([(10, 10, 50, 50)])
    d = run("opencv", det)
    assert d.mode == "opencv" and det.calls == 0 and d.boxes == CV(IMG) and d.note == ""


def test_gemini_mode_uses_only_padded_gemini_boxes():
    det = StubDetector([(600, 300, 100, 100)])
    d = run("gemini", det)
    assert d.mode == "gemini" and d.boxes == [pad_box((600, 300, 100, 100), gd.PAD_FRAC, IMG.shape)]
    assert d.gemini_count == 1 and d.opencv_count == 0


def test_hybrid_keeps_gemini_and_adds_opencv_only_where_gemini_missed():
    det = StubDetector([(96, 96, 120, 120)])                       # same piece as OpenCV's first box
    d = run("hybrid", det)
    assert d.mode == "hybrid" and len(d.boxes) == 2                # gemini's + opencv's second (missed) piece
    assert (1500, 700, 100, 100) in d.boxes and (100, 100, 120, 120) not in d.boxes
    assert (d.gemini_count, d.opencv_count) == (1, 2)


@pytest.mark.parametrize("error", [DetectError("timed out after 20s"), RuntimeError("boom"), asyncio.TimeoutError()])
def test_gemini_failure_falls_back_to_opencv_and_is_logged(error, caplog):
    for mode in ("gemini", "hybrid"):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="lego.detect"):
            d = run(mode, StubDetector(error=error))
        assert d.mode == "opencv" and d.boxes == CV(IMG) and "gemini failed" in d.note
        assert any("falling back to opencv" in r.message for r in caplog.records)


def test_no_detector_configured_falls_back_to_opencv(caplog):
    with caplog.at_level(logging.WARNING, logger="lego.detect"):
        d = run("hybrid", None)
    assert d.mode == "opencv" and "not configured" in d.note and caplog.records


def test_unknown_mode_means_hybrid_and_giants_are_removed_from_the_opencv_fallback():
    det = StubDetector([(96, 96, 120, 120)])
    assert run("nonsense", det).mode == "hybrid"
    d = run("opencv", None, opencv=lambda img: [(0, 0, 1900, 900), (100, 100, 100, 100)])
    assert d.boxes == [(100, 100, 100, 100)]


def test_detector_setting_is_validated():
    assert config.parse_detector("Hybrid ") == "hybrid" and config.parse_detector("GEMINI") == "gemini"
    assert config.parse_detector("") == "opencv" and config.parse_detector("bogus") == "opencv"
    assert config.parse_detector(None, default="hybrid") == "hybrid"


# ---------------------------------------------------------------- the 42-piece sample in all three modes
def sample_stub():
    """A Gemini stand-in that finds the sample's 42 pieces: tight boxes (no padding), as Gemini returns them."""
    img = cv2.imread(str(SAMPLE))
    tight = []
    for x, y, w, h in find_pieces(img):
        p = int(max(w, h) * 0.07)
        tight.append((x + p, y + p, w - 2 * p, h - 2 * p))
    return img, StubDetector(tight)


@pytest.mark.parametrize("mode", ["opencv", "gemini", "hybrid"])
def test_sample_image_gives_42_pieces_in_every_mode(mode):
    img, det = sample_stub()
    d = asyncio.run(detect_pieces(img, mode, det))
    assert len(d.boxes) == 42 and d.mode == mode
    assert not any(gd.too_big(b, img.shape) for b in d.boxes)


def test_sample_image_still_42_when_gemini_fails_in_hybrid_mode():
    img, _ = sample_stub()
    d = asyncio.run(detect_pieces(img, "hybrid", StubDetector(error=DetectError("503"))))
    assert len(d.boxes) == 42 and d.mode == "opencv"


# ---------------------------------------------------------------- the real GeminiDetector against a fake client
class FakeModels:
    def __init__(self, behaviours):
        self.behaviours, self.calls = behaviours, []

    async def generate_content(self, **kw):
        self.calls.append(kw)
        b = self.behaviours[min(len(self.calls) - 1, len(self.behaviours) - 1)]
        if isinstance(b, BaseException):
            raise b
        if callable(b):
            return await b()
        return b


def reply(text, block=None):
    return SimpleNamespace(text=text, prompt_feedback=SimpleNamespace(block_reason=block))


def detector(behaviours, **kw):
    d = GeminiDetector("test-key", "primary-model", kw.pop("fallback", "fallback-model"), **kw)
    d.client = SimpleNamespace(aio=SimpleNamespace(models=FakeModels(behaviours)))
    return d, d.client.aio.models


def api_error(code, status=""):
    from google.genai import errors
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": "x", "status": status}})


GOOD = '{"pieces": [{"box_2d": [100, 200, 300, 600], "label": "red brick"}]}'


def test_detector_sends_the_image_prompt_schema_and_converts_boxes_using_the_original_size():
    d, models = detector([reply(GOOD)])
    big = np.full((1200, 3200, 3), 200, np.uint8)                 # larger than SEND_MAX_SIDE: downscaled for the call
    boxes, model = asyncio.run(d.detect(big))
    assert model == "primary-model"
    assert boxes == [(640, 120, 1280, 240)]                       # normalised coords x ORIGINAL 3200x1200
    kw = models.calls[0]
    assert kw["model"] == "primary-model"
    assert kw["config"].response_mime_type == "application/json" and kw["config"].response_json_schema == gd.SCHEMA
    assert "box_2d" in kw["contents"][1] and "one tight bounding box per piece" in kw["contents"][1]
    assert kw["contents"][0].inline_data.mime_type == "image/jpeg"
    assert max(cv2.imdecode(np.frombuffer(kw["contents"][0].inline_data.data, np.uint8), 1).shape[:2]) == gd.SEND_MAX_SIDE
    assert str(kw["config"].thinking_config.thinking_level).endswith("MINIMAL")


@pytest.mark.parametrize("code,status", [(503, "UNAVAILABLE"), (504, "DEADLINE_EXCEEDED")])
def test_503_or_504_retries_once_with_the_fallback_model(code, status):
    d, models = detector([api_error(code, status), reply(GOOD)])
    boxes, model = asyncio.run(d.detect(IMG))
    assert model == "fallback-model" and len(boxes) == 1
    assert [c["model"] for c in models.calls] == ["primary-model", "fallback-model"]


def test_other_errors_are_not_retried_and_become_detect_errors():
    for err in (api_error(400, "INVALID_ARGUMENT"), api_error(429, "RESOURCE_EXHAUSTED"), api_error(403, "PERMISSION_DENIED")):
        d, models = detector([err, reply(GOOD)])
        with pytest.raises(DetectError):
            asyncio.run(d.detect(IMG))
        assert len(models.calls) == 1


def thinking_error():
    from google.genai import errors
    return errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                              "message": "Thinking level MINIMAL is not supported for this model."}})


def test_unsupported_thinking_level_steps_down_then_remembers_what_worked():
    d, models = detector([thinking_error(), reply(GOOD), reply(GOOD)])
    asyncio.run(d.detect(IMG))
    levels = [str(getattr(c["config"].thinking_config, "thinking_level", None)) for c in models.calls]
    assert levels[0].endswith("MINIMAL") and levels[1].endswith("LOW") and len(levels) == 2
    asyncio.run(d.detect(IMG))                                      # the next call goes straight to the level that worked
    assert str(models.calls[2]["config"].thinking_config.thinking_level).endswith("LOW")


def test_thinking_setting_is_dropped_when_no_level_is_accepted():
    d, models = detector([thinking_error(), thinking_error(), reply(GOOD)])
    boxes, _ = asyncio.run(d.detect(IMG))
    assert len(boxes) == 1 and models.calls[2]["config"].thinking_config is None


def test_a_400_that_is_not_about_thinking_is_not_retried_with_other_levels():
    d, models = detector([api_error(400, "INVALID_ARGUMENT"), reply(GOOD)])
    with pytest.raises(DetectError):
        asyncio.run(d.detect(IMG))
    assert len(models.calls) == 1


def test_a_failing_fallback_model_is_a_detect_error():
    d, _ = detector([api_error(503, "UNAVAILABLE"), api_error(503, "UNAVAILABLE")])
    with pytest.raises(DetectError):
        asyncio.run(d.detect(IMG))


def test_no_fallback_model_means_no_retry():
    d, models = detector([api_error(503, "UNAVAILABLE"), reply(GOOD)], fallback=None)
    with pytest.raises(DetectError):
        asyncio.run(d.detect(IMG))
    assert len(models.calls) == 1


def test_timeout_is_a_detect_error_after_the_deadline():
    async def hang():
        await asyncio.sleep(5)
    d, _ = detector([hang], timeout=0.2)
    with pytest.raises(DetectError, match="timed out"):
        asyncio.run(d.detect(IMG))


@pytest.mark.parametrize("response", [reply(""), reply(None), reply("not json {"), reply(GOOD, block="SAFETY")])
def test_empty_invalid_or_blocked_responses_are_detect_errors(response):
    d, _ = detector([response])
    with pytest.raises(DetectError):
        asyncio.run(d.detect(IMG))


def test_a_bare_json_list_is_tolerated_and_bad_boxes_are_skipped():
    d, _ = detector([reply('[{"box_2d": [100, 100, 300, 300]}, {"box_2d": [1, 2]}]')])
    assert asyncio.run(d.detect(IMG))[0] == [(200, 100, 400, 200)]


def test_end_to_end_with_a_stubbed_gemini_through_detect_pieces():
    d, _ = detector([reply(GOOD)])
    out = asyncio.run(detect_pieces(IMG, "hybrid", d, opencv=CV))
    assert out.mode == "hybrid" and out.gemini_count == 1 and len(out.boxes) == 3


# ---------------------------------------------------------------- the scan job uses the configured detector
def scene_image():
    img = np.full((300, 300, 3), 255, np.uint8)
    img[50:120, 50:120] = (0, 0, 200)
    img[180:250, 160:240] = (200, 0, 0)
    return img


def run_job(tmp_path, monkeypatch, mode, detector):
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path)
    job = jobs.new_job()

    async def rec(crops):
        return [{"error": "stub"}] * len(crops)
    asyncio.run(jobs.run_photo_job(job, scene_image(), rec, detector=detector, mode=mode))
    return job


def test_job_records_which_detector_ran_and_survives_a_gemini_failure(tmp_path, monkeypatch):
    job = run_job(tmp_path, monkeypatch, "hybrid", StubDetector(error=DetectError("timed out after 20s")))
    assert job.status == "done" and job.detector == "opencv" and "timed out" in job.detector_note
    assert job.total == 2
    pub = job.public()
    assert pub["detector"] == "opencv" and "timed out" in pub["detector_note"]


def test_job_uses_gemini_boxes_when_available(tmp_path, monkeypatch):
    det = StubDetector([(50, 50, 70, 70), (160, 180, 80, 70)])
    job = run_job(tmp_path, monkeypatch, "gemini", det)
    assert job.status == "done" and job.detector == "gemini" and job.total == 2 and det.calls == 1


def test_job_default_mode_comes_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DETECTOR", "opencv")
    det = StubDetector([(50, 50, 70, 70)])
    job = run_job(tmp_path, monkeypatch, None, det)
    assert job.detector == "opencv" and det.calls == 0
