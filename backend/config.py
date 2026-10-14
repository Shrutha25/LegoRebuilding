"""Settings from environment (.env is loaded if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("RB_DATA_DIR", ROOT / "rb_data"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "tmp_uploads"))
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
LLM_EFFORT = os.environ.get("LLM_EFFORT", "medium")   # set empty for models without effort support
BRICKOGNIZE_URL = os.environ.get("BRICKOGNIZE_URL", "https://api.brickognize.com/predict/parts/")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIDE = 2400          # larger photos are downscaled before detection
JOB_TTL_SECONDS = 3600
MIN_SCORE = 0.6                # below this a piece needs review
