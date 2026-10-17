"""Gemini quota budget: every Gemini request in the app goes through `generate()` in this module.

Free-tier limits are per model, so usage is tracked per model: requests in the last 60 s (RPM) and requests
since the daily reset (RPD). Every actual API request counts, including retries and fallback-model attempts.

Two purposes share a model's quota:
    build      priority (the core feature). May use everything that remains and may wait briefly for an RPM slot.
    detection  best effort. Limited to DETECTION_DAILY_SHARE of the daily limit, refused if it would eat into the
               build reserve, and never waits: if the minute window is full it is refused at once (the caller
               falls back to OpenCV).

State lives in SQLite (WAL, `BEGIN IMMEDIATE` per decision), so counters survive restarts and are exact across
concurrent requests and several worker processes.

The limits below are PLACEHOLDERS. Check them against Google's current published free-tier limits for each model
(https://ai.google.dev/gemini-api/docs/rate-limits) and set GEMINI_MODEL_LIMITS or the defaults in the environment.
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

log = logging.getLogger("lego.budget")

BUILD = "build"
DETECTION = "detection"
PURPOSES = (BUILD, DETECTION)
WINDOW_SECONDS = 60.0


@dataclass
class BudgetConfig:
    default_rpm: int = 10                    # placeholder: requests per minute per model
    default_rpd: int = 20                    # placeholder: requests per day per model
    model_limits: dict = None                # {"model": {"rpm": n, "rpd": n}} overrides the defaults
    detection_daily_share: float = 0.3       # detection may use at most this share of a model's daily limit
    build_reserve: int | None = None         # requests kept back for builds (absolute); None -> build_reserve_frac
    build_reserve_frac: float = 0.2          # ...as a share of the daily limit
    reset_time: str = "00:00"                # local time of the daily reset, HH:MM
    reset_tz: str = "America/Los_Angeles"    # Google resets free-tier daily quotas at midnight Pacific
    build_max_wait: float = 10.0             # seconds a build may wait for an RPM slot before failing

    def limits(self, model):
        o = (self.model_limits or {}).get(model, {})
        return int(o.get("rpm", self.default_rpm)), int(o.get("rpd", self.default_rpd))

    def reserve(self, rpd):
        if self.build_reserve is not None:
            return max(0, int(self.build_reserve))
        return int(round(rpd * self.build_reserve_frac))


class BudgetRefused(Exception):
    """A request was not allowed by the budget. `reason` is short and log-ready; `retry_after` is seconds
    until it could succeed (None: not before the daily reset or unknown)."""

    def __init__(self, reason, retry_after=None):
        super().__init__(reason)
        self.reason, self.retry_after = reason, retry_after


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: float | None = None

    def __bool__(self):
        return self.allowed


def parse_retry_delay(err):
    """Seconds the API asked us to wait (RetryInfo 'retryDelay': '34s' or 'retry in 34.5s'), else None."""
    text = f"{getattr(err, 'details', '')} {getattr(err, 'message', '')} {err}"
    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", text) or \
        re.search(r"retry in (\d+(?:\.\d+)?)\s*s", text, re.I)
    return float(m.group(1)) if m else None


def is_quota_error(err):
    return getattr(err, "code", None) == 429 or (getattr(err, "status", "") or "").upper() == "RESOURCE_EXHAUSTED"


class GeminiBudget:
    def __init__(self, path, config=None, clock=time.time):
        self.path, self.cfg, self.clock = Path(path), config or BudgetConfig(), clock
        self.tz = ZoneInfo(self.cfg.reset_tz)
        hh, mm = (int(p) for p in self.cfg.reset_time.split(":"))
        self._reset_at = dtime(hh, mm)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tx() as c:
            c.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, model TEXT NOT NULL, "
                      "purpose TEXT NOT NULL, ts REAL NOT NULL)")
            c.execute("CREATE INDEX IF NOT EXISTS requests_model_ts ON requests (model, ts)")
            c.execute("CREATE TABLE IF NOT EXISTS exhausted (model TEXT PRIMARY KEY, until REAL NOT NULL)")

    # ---- storage
    def _tx(self):
        return _Tx(self)

    def _connect(self):
        c = sqlite3.connect(str(self.path), timeout=15, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    # ---- time
    def day_window(self, now=None):
        """(start, next reset) of the current quota day, as timestamps."""
        now = self.clock() if now is None else now
        local = datetime.fromtimestamp(now, self.tz)
        start = datetime.combine(local.date(), self._reset_at, tzinfo=self.tz)
        if start > local:
            start -= timedelta(days=1)
        end = datetime.combine((start + timedelta(days=1)).date(), self._reset_at, tzinfo=self.tz)
        return start.timestamp(), end.timestamp()

    # ---- decisions
    def _decide(self, c, purpose, model, now):
        rpm, rpd = self.cfg.limits(model)
        day_start, day_end = self.day_window(now)
        row = c.execute("SELECT until FROM exhausted WHERE model=?", (model,)).fetchone()
        if row and row[0] > now:
            return Decision(False, f"model exhausted until {_fmt(row[0], self.tz)}", min(row[0] - now, day_end - now))
        used_day, = c.execute("SELECT COUNT(*) FROM requests WHERE model=? AND ts>=?", (model, day_start)).fetchone()
        if used_day >= rpd:
            return Decision(False, f"daily limit reached ({used_day}/{rpd}), resets {_fmt(day_end, self.tz)}",
                            day_end - now)
        if purpose == DETECTION:
            det_day, = c.execute("SELECT COUNT(*) FROM requests WHERE model=? AND purpose=? AND ts>=?",
                                 (model, DETECTION, day_start)).fetchone()
            if det_day >= int(self.cfg.detection_daily_share * rpd):
                return Decision(False, "detection budget exhausted", day_end - now)
            if rpd - used_day - 1 < self.cfg.reserve(rpd):
                return Decision(False, "detection budget exhausted (build reserve)", day_end - now)
        times = [r[0] for r in c.execute("SELECT ts FROM requests WHERE model=? AND ts>? ORDER BY ts",
                                         (model, now - WINDOW_SECONDS))]
        if len(times) >= rpm:
            wait = times[len(times) - rpm] + WINDOW_SECONDS - now
            return Decision(False, "rpm window full", max(wait, 0.0))
        return Decision(True)

    def check(self, purpose, model):
        """Would a request be allowed right now? Records nothing."""
        with self._tx() as c:
            return self._decide(c, purpose, model, self.clock())

    def acquire(self, purpose, model):
        """Decide and, if allowed, record the request, atomically across threads and processes."""
        assert purpose in PURPOSES, purpose
        now = self.clock()
        with self._tx() as c:
            d = self._decide(c, purpose, model, now)
            if d.allowed:
                c.execute("INSERT INTO requests (model, purpose, ts) VALUES (?,?,?)", (model, purpose, now))
                c.execute("DELETE FROM requests WHERE ts < ?", (now - 2 * 86400,))
        return d

    def try_acquire(self, purpose, model):
        return bool(self.acquire(purpose, model))

    async def acquire_async(self, purpose, model):
        """Detection: refused at once if not allowed. Build: waits up to build_max_wait for an RPM slot (or a
        short model cool-down), else raises BudgetRefused. Raises BudgetRefused for either purpose when refused."""
        deadline = self.clock() + (self.cfg.build_max_wait if purpose == BUILD else 0.0)
        while True:
            d = await asyncio.to_thread(self.acquire, purpose, model)
            if d.allowed:
                return
            wait = d.retry_after
            if purpose == BUILD and wait is not None and self.clock() + wait <= deadline:
                await asyncio.sleep(wait + 0.05)
                continue
            raise BudgetRefused(d.reason, d.retry_after)

    def mark_exhausted(self, model, until=None):
        """After a 429: stop using `model` until `until` (timestamp) or the next daily reset."""
        now = self.clock()
        until = self.day_window(now)[1] if until is None else until
        with self._tx() as c:
            c.execute("INSERT INTO exhausted (model, until) VALUES (?,?) "
                      "ON CONFLICT(model) DO UPDATE SET until=excluded.until", (model, until))
        log.warning("gemini model %s marked exhausted until %s", model, _fmt(until, self.tz))

    # ---- reporting
    def status(self, models=()):
        now = self.clock()
        day_start, day_end = self.day_window(now)
        with self._tx() as c:
            seen = [r[0] for r in c.execute("SELECT DISTINCT model FROM requests WHERE ts>=?", (day_start,))]
            out = {}
            for model in dict.fromkeys([*models, *seen]):
                rpm, rpd = self.cfg.limits(model)
                used, = c.execute("SELECT COUNT(*) FROM requests WHERE model=? AND ts>=?", (model, day_start)).fetchone()
                det, = c.execute("SELECT COUNT(*) FROM requests WHERE model=? AND purpose=? AND ts>=?",
                                 (model, DETECTION, day_start)).fetchone()
                per_min, = c.execute("SELECT COUNT(*) FROM requests WHERE model=? AND ts>?",
                                     (model, now - WINDOW_SECONDS)).fetchone()
                ex = c.execute("SELECT until FROM exhausted WHERE model=?", (model,)).fetchone()
                out[model] = {"used": used, "limit": rpd, "remaining": max(0, rpd - used),
                              "detection_used": det, "detection_cap": int(self.cfg.detection_daily_share * rpd),
                              "build_reserve": self.cfg.reserve(rpd), "rpm_used": per_min, "rpm_limit": rpm,
                              "exhausted_until": _fmt(ex[0], self.tz) if ex and ex[0] > now else None,
                              "resets_at": _fmt(day_end, self.tz)}
        return out


class _Tx:
    """One short IMMEDIATE transaction on its own connection (SQLite serialises writers across processes)."""

    def __init__(self, budget):
        self.b = budget

    def __enter__(self):
        self.c = self.b._connect()
        self.c.execute("BEGIN IMMEDIATE")
        return self.c

    def __exit__(self, exc_type, *_):
        self.c.execute("ROLLBACK" if exc_type else "COMMIT")
        self.c.close()
        return False


def _fmt(ts, tz):
    return datetime.fromtimestamp(ts, tz).isoformat(timespec="seconds")


# ---------------------------------------------------------------- the one door to the Gemini API
_TRANSIENT = (500, 502, 503, 504)


async def generate(client, budget, purpose, model, *, retries=0, **kwargs):
    """`client.aio.models.generate_content(model=model, **kwargs)`, counted against `budget`.

    Each attempt (including each of `retries` retries after a transient 5xx) acquires budget first. Raises
    BudgetRefused when the budget says no; on a 429 the model is marked exhausted before the error is re-raised.
    `budget=None` disables accounting (tests, ad-hoc scripts)."""
    attempt = 0
    while True:
        if budget is not None:
            await budget.acquire_async(purpose, model)
        try:
            return await client.aio.models.generate_content(model=model, **kwargs)
        except Exception as e:
            if budget is not None and is_quota_error(e):
                delay = parse_retry_delay(e)
                if delay is None and "perminute" in str(e).lower().replace(" ", "").replace("_", ""):
                    delay = WINDOW_SECONDS
                budget.mark_exhausted(model, budget.clock() + delay if delay is not None else None)
            elif attempt < retries and getattr(e, "code", None) in _TRANSIENT:
                attempt += 1
                continue
            raise


# ---------------------------------------------------------------- process-wide instance
_budget = None


def build_config(cfg):
    """BudgetConfig from backend.config (env)."""
    return BudgetConfig(default_rpm=cfg.GEMINI_RPM, default_rpd=cfg.GEMINI_RPD, model_limits=cfg.GEMINI_MODEL_LIMITS,
                        detection_daily_share=cfg.DETECTION_DAILY_SHARE, build_reserve=cfg.BUILD_RESERVE,
                        build_reserve_frac=cfg.BUILD_RESERVE_FRAC, reset_time=cfg.GEMINI_RESET_TIME,
                        reset_tz=cfg.GEMINI_RESET_TZ, build_max_wait=cfg.BUILD_MAX_WAIT)


def get_budget():
    global _budget
    if _budget is None:
        from backend import config
        _budget = GeminiBudget(config.GEMINI_BUDGET_DB, build_config(config))
    return _budget
