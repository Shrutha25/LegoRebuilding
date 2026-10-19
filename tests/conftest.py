import pytest

from backend import config
from backend.core import studs


@pytest.fixture(autouse=True)
def top_down_studs(monkeypatch):
    """The stud detector defaults to a 45 degree view (STUD_TILT_STRETCH=1.5); the synthetic test scenes are top-down."""
    monkeypatch.setattr(studs.CFG, "tilt_stretch", 1.0)


@pytest.fixture(autouse=True)
def offline_set_info(monkeypatch, tmp_path):
    """Set cards look up instructions in the background; tests never reach Brickset or the LDraw library."""
    from backend import builds_api
    from backend.core.instructions import InstructionsService
    from backend.core.setinfo import SetInfo

    async def no_model(*a, **k):
        return None

    info = SetInfo(InstructionsService(tmp_path / "instr.sqlite", api_key=""), tmp_path / "omr_cache", "https://x.invalid/omr", fetch_omr=no_model)
    monkeypatch.setattr(builds_api, "_setinfo", info)


@pytest.fixture(autouse=True)
def tiny_fixture_sets(monkeypatch):
    """The test inventories have a handful of parts; the production minimum (MIN_SET_PARTS=10) is tested explicitly."""
    monkeypatch.setattr(config, "MIN_SET_PARTS", 1)


@pytest.fixture(autouse=True)
def no_timing_file(monkeypatch):
    """Requests in tests would otherwise append to the real timings.jsonl."""
    monkeypatch.setattr(config, "TIMING_LOG", "")


@pytest.fixture(autouse=True)
def no_background_warmup(monkeypatch):
    """The app's start-up warm-up (nearby-sets, OMR and LDraw indexes) is tested on its own; other tests must not load the real ones."""
    from backend import builds_api
    monkeypatch.setattr(builds_api, "start_warmup", lambda store: None)


@pytest.fixture(autouse=True)
def gemini_never_live(monkeypatch):
    """Tests never call Gemini: every genai.Client is a tape client that replays tests/fixtures/gemini_tape (backend/core/gemini_tape.py) and raises
    NoRecording for anything not recorded. Live calls only with GEMINI_LIVE=1, at most 5 per run."""
    from google import genai
    from backend.core import gemini_tape
    monkeypatch.setattr(genai, "Client", gemini_tape.client_factory(genai.Client))
