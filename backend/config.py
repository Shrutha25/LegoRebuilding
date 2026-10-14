"""Settings from environment (.env is loaded if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("RB_DATA_DIR", ROOT / "rb_data"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "tmp_uploads"))
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
BRICKOGNIZE_URL = os.environ.get("BRICKOGNIZE_URL", "https://api.brickognize.com/predict/parts/")



def parse_min_coverage(raw, default=1.0):
    """MIN_COVERAGE: fraction of a set that must be covered (1.0 = complete sets only)."""
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return default


MIN_COVERAGE = parse_min_coverage(os.environ.get("MIN_COVERAGE"))

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIDE = 2400          # larger photos are downscaled before detection
JOB_TTL_SECONDS = 3600
MIN_SCORE = 0.6                # below this a piece needs review
