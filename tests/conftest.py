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
