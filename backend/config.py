"""Settings from environment (.env is loaded if present)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("RB_DATA_DIR", ROOT / "rb_data"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "tmp_uploads"))
# Manual merge / split corrections (photo hash, before, after) kept for the detection test set; the photo is saved beside them.
CORRECTIONS_DIR = Path(os.environ.get("CORRECTIONS_DIR", ROOT / "corrections"))
REBRICKABLE_API_KEY = os.environ.get("REBRICKABLE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Flash-Lite answered in ~4 s when the larger Flash models (3.5/3.6/3.8) were returning 503 "high demand".
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Piece detection: opencv | gemini | hybrid. Defaults to opencv until the evaluation table has been
# reviewed (see python -m backend.eval_detect); hybrid is the intended default after that.
def parse_detector(raw, default="opencv"):
    raw = (raw or "").strip().lower()
    return raw if raw in ("opencv", "gemini", "hybrid") else default


DETECTOR = parse_detector(os.environ.get("DETECTOR"))
GEMINI_DETECT_MODEL = os.environ.get("GEMINI_DETECT_MODEL", "gemini-3.1-flash-lite")        # separate from the ideas model
GEMINI_DETECT_FALLBACK_MODEL = os.environ.get("GEMINI_DETECT_FALLBACK_MODEL", "gemini-3.1-flash-lite")  # used on 503/504

# ---- Gemini quota budget (see backend/core/gemini_budget.py). The numbers are PLACEHOLDERS: check them against
# Google's current published free-tier limits for each model (the README measured about 20 requests/day per
# model on the free tier). Per-model overrides:
#   GEMINI_MODEL_LIMITS='{"gemini-3.1-flash-lite": {"rpm": 15, "rpd": 500}}'
def _int(raw, default):
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _float(raw, default):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _limits(raw):
    import json
    try:
        data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


GEMINI_RPM = _int(os.environ.get("GEMINI_RPM"), 10)                    # requests per minute, per model
GEMINI_RPD = _int(os.environ.get("GEMINI_RPD"), 20)                   # requests per day, per model
GEMINI_MODEL_LIMITS = _limits(os.environ.get("GEMINI_MODEL_LIMITS"))
DETECTION_DAILY_SHARE = min(1.0, max(0.0, _float(os.environ.get("DETECTION_DAILY_SHARE"), 0.3)))
BUILD_RESERVE = _int(os.environ.get("BUILD_RESERVE"), None) if os.environ.get("BUILD_RESERVE") else None   # absolute count
BUILD_RESERVE_FRAC = min(1.0, max(0.0, _float(os.environ.get("BUILD_RESERVE_FRAC"), 0.2)))                # used when BUILD_RESERVE is unset
GEMINI_RESET_TIME = os.environ.get("GEMINI_RESET_TIME", "00:00")                 # daily reset, local time in GEMINI_RESET_TZ
GEMINI_RESET_TZ = os.environ.get("GEMINI_RESET_TZ", "America/Los_Angeles")
BUILD_MAX_WAIT = _float(os.environ.get("BUILD_MAX_WAIT"), 10.0)                  # seconds a build waits for an RPM slot
GEMINI_BUDGET_DB = Path(os.environ.get("GEMINI_BUDGET_DB", DATA_DIR / "gemini_budget.sqlite"))

# ---- 3D building instructions (LDraw). The parts library is read from a zip (python -m backend.core.fetch_ldraw).
LDRAW_DIR = Path(os.environ.get("LDRAW_DIR", ROOT / "ldraw_data"))
LDRAW_ZIP_URL = os.environ.get("LDRAW_ZIP_URL", "https://library.ldraw.org/library/updates/complete.zip")
OMR_BASE_URL = os.environ.get("OMR_BASE_URL", "https://library.ldraw.org/library/omr")     # <base>/<set number>.mpd
BUILD_CACHE_DIR = Path(os.environ.get("BUILD_CACHE_DIR", ROOT / "build_cache"))          # packed models, OMR downloads, id maps
BUILD_MAX_PARTS = _int(os.environ.get("BUILD_MAX_PARTS"), 80)                              # ask Gemini for at most this many parts
BUILD_MAX_DROP_FRACTION = min(1.0, max(0.0, _float(os.environ.get("BUILD_MAX_DROP_FRACTION"), 0.2)))   # retry Gemini once above this
BUILD_STEP_SIZE = _int(os.environ.get("BUILD_STEP_SIZE"), 3)                               # parts per generated step (1-3)

# ---- recipe builds (backend/recipes): the code designs the shapes, Gemini only picks, names and colours them.
def _flag(raw, default=False):
    return default if raw is None or raw == "" else str(raw).strip().lower() in ("1", "true", "yes", "on")


AI_FREEFORM_BUILDS = _flag(os.environ.get("AI_FREEFORM_BUILDS"))             # old free-form Gemini placement, only as a fallback (default off)
RECIPE_AI_TIMEOUT = _float(os.environ.get("RECIPE_AI_TIMEOUT"), 25.0)         # seconds the pick/name call may take before templates are used
RECIPE_SHOW = _int(os.environ.get("RECIPE_SHOW"), 5)                          # builds shown per request (3-5)
RECIPE_GALLERY = _flag(os.environ.get("RECIPE_GALLERY"))                      # dev page /recipe-gallery needs this endpoint
RECIPE_CONCEPTS = _int(os.environ.get("RECIPE_CONCEPTS"), 12)                 # concepts asked from Gemini per request (10-15)
RECIPE_LIBRARY_DIR = Path(os.environ.get("RECIPE_LIBRARY_DIR", ROOT / "recipe_library"))   # recipes generated on demand that passed every check
RECIPE_GEN_PER_REQUEST = _int(os.environ.get("RECIPE_GEN_PER_REQUEST"), 1)    # new recipes Gemini may write per request when nothing fits (0 = never)
RECIPE_GEN_TIMEOUT = _float(os.environ.get("RECIPE_GEN_TIMEOUT"), 90.0)       # seconds for one on-demand recipe including its repairs
RECIPE_RECOGNISE = _flag(os.environ.get("RECIPE_RECOGNISE"), True)            # vision check ("what is this model?") on generated recipes

BRICKOGNIZE_URL = os.environ.get("BRICKOGNIZE_URL", "https://api.brickognize.com/predict/parts/")



def parse_min_coverage(raw, default=1.0):
    """MIN_COVERAGE: fraction of a set that must be covered (1.0 = complete sets only)."""
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return default


MIN_COVERAGE = parse_min_coverage(os.environ.get("MIN_COVERAGE"))


# ---- which Rebrickable sets count as buildable (backend/core/match.py: skipped_theme_ids)
def _names(raw, default):
    """Comma-separated list from the environment, lower-cased; the default when unset."""
    if raw is None or not raw.strip():
        return tuple(x.lower() for x in default)
    return tuple(x.strip().lower() for x in raw.split(",") if x.strip())


# Themes (and all their sub-themes) that are not builds. Exact name, case-insensitive.
EXCLUDED_THEMES = _names(os.environ.get("EXCLUDED_THEMES"), [
    "Gear", "Key Chains", "Magnets", "Books", "Promotional", "Bulk Bricks", "Service Packs",
    "Supplemental", "Collectible Minifigures"])
# ...and any theme whose name contains one of these words.
EXCLUDED_THEME_WORDS = _names(os.environ.get("EXCLUDED_THEME_WORDS"), [
    "Gear", "Key Chain", "Magnet", "Storage", "Clothing", "Video Games"])
MIN_SET_PARTS = max(0, _int(os.environ.get("MIN_SET_PARTS"), 10))      # smaller sets are never matched

# ---- official building instructions (backend/core/instructions.py, Brickset API v3 getInstructions2)
BRICKSET_API_KEY = os.environ.get("BRICKSET_API_KEY", "")
INSTRUCTIONS_TTL_DAYS = _float(os.environ.get("INSTRUCTIONS_TTL_DAYS"), 7.0)
BRICKSET_DAILY_CAP = _int(os.environ.get("BRICKSET_DAILY_CAP"), 100)    # API calls per UTC day, then the cache is all we use
BRICKSET_MIN_INTERVAL = _float(os.environ.get("BRICKSET_MIN_INTERVAL"), 1.0)   # seconds between two calls
INSTRUCTIONS_DB = Path(os.environ.get("INSTRUCTIONS_DB", DATA_DIR / "instructions.sqlite"))

# per-request timing log (JSON lines, see backend/timing.py); TIMING_LOG= (empty) switches the file off
TIMING_LOG = os.environ.get("TIMING_LOG", str(ROOT / "timings.jsonl"))

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_SIDE = 2400          # larger photos are downscaled before detection
JOB_TTL_SECONDS = 3600
REFERENCE_TTL_SECONDS = 6 * 3600   # an empty-surface photo stays usable for scans this long
MIN_SCORE = 0.6                # below this a piece needs review
