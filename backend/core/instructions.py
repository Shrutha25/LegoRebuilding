"""Official LEGO building instructions, looked up through the Brickset API v3 (method getInstructions2).

    GET/POST https://brickset.com/api/v3.asmx/getInstructions2   apiKey=<key>  setNumber=<number-variant, e.g. 6012-1>
    -> {"status": "success", "matches": n, "instructions": [{"URL": ..., "description": ...}]}
    -> {"status": "error", "message": "Invalid API key"}

For a set number this gives {available, links: [{url, description}], checked_at}. Only the links are kept: LEGO's PDFs and
images are never downloaded, stored, proxied or re-hosted, and links that do not point at a lego.com host are dropped.

Results live in SQLite (INSTRUCTIONS_DB), keyed by the full set number ("6012-1", the suffix is never stripped), and are
refreshed after INSTRUCTIONS_TTL_DAYS. If the API fails, times out, is over its daily cap or has no key, the last cached
value is kept; with none, the set is "unknown" (treated as not available) and nothing is cached, so it is tried again later.
Nothing here raises into a request: `cached()` only reads the database, and `refresh()` is for the background worker, which
spaces calls out (BRICKSET_MIN_INTERVAL) and stops at BRICKSET_DAILY_CAP calls per UTC day.
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

log = logging.getLogger("lego.instructions")

API_URL = "https://brickset.com/api/v3.asmx/getInstructions2"
RETRY_AFTER = 600          # seconds before a set whose lookup failed (and has no cached value) is tried again


def lego_link(url):
    """True for an https link on lego.com (or a sub-domain)."""
    try:
        u = urlparse(url or "")
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    return u.scheme == "https" and (host == "lego.com" or host.endswith(".lego.com"))


def parse_response(data):
    """Brickset's JSON -> list of {url, description} on LEGO hosts. Raises ValueError on an error answer."""
    if not isinstance(data, dict) or data.get("status") != "success":
        raise ValueError(f"Brickset: {data.get('message') if isinstance(data, dict) else 'unexpected answer'}")
    links, seen = [], set()
    for item in data.get("instructions") or []:
        url = (item or {}).get("URL") or ""
        if lego_link(url) and url not in seen:
            seen.add(url)
            links.append({"url": url, "description": str((item or {}).get("description") or "").strip()})
    return links


def brickset_fetch(api_key, set_number, timeout=10.0):
    """One real API call -> list of links. Raises on network or API errors."""
    r = httpx.get(API_URL, params={"apiKey": api_key, "setNumber": set_number}, timeout=timeout)
    r.raise_for_status()
    return parse_response(r.json())


class InstructionsService:
    def __init__(self, db_path, api_key="", ttl_days=7.0, daily_cap=100, min_interval=1.0, fetch=brickset_fetch, clock=time.time):
        self.path = Path(db_path)
        self.api_key, self.ttl, self.cap = api_key, ttl_days * 86400, daily_cap
        self.min_interval, self.fetch, self.clock = min_interval, fetch, clock
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._failed = {}                      # set number -> time of the last failed lookup (no cached value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS instructions (set_num TEXT PRIMARY KEY, available INTEGER NOT NULL,"
                       " links TEXT NOT NULL, checked_at REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (day TEXT PRIMARY KEY, n INTEGER NOT NULL)")

    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # ------------------------------------------------------------------ cache
    def cached(self, set_num):
        """The stored {available, links, checked_at} (possibly stale), or None when never looked up. Database read only."""
        try:
            with self._db() as db:
                row = db.execute("SELECT available, links, checked_at FROM instructions WHERE set_num=?", (set_num,)).fetchone()
        except sqlite3.Error as e:
            log.warning("instructions cache unreadable: %s", e)
            return None
        if row is None:
            return None
        return {"available": bool(row[0]), "links": json.loads(row[1]), "checked_at": row[2]}

    def is_fresh(self, entry):
        return entry is not None and self.clock() - entry["checked_at"] < self.ttl

    def needs_refresh(self, set_num):
        """No entry, or an entry older than the TTL; and not a set whose lookup failed a few minutes ago."""
        if self.is_fresh(self.cached(set_num)):
            return False
        return self.clock() - self._failed.get(set_num, 0.0) >= RETRY_AFTER

    def _store(self, set_num, links):
        entry = {"available": bool(links), "links": links, "checked_at": self.clock()}
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO instructions VALUES (?,?,?,?)",
                       (set_num, int(entry["available"]), json.dumps(links), entry["checked_at"]))
        return entry

    # ------------------------------------------------------------------ budget
    def _today(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def calls_today(self):
        with self._db() as db:
            row = db.execute("SELECT n FROM calls WHERE day=?", (self._today(),)).fetchone()
        return row[0] if row else 0

    def _count_call(self):
        with self._db() as db:
            db.execute("INSERT INTO calls VALUES (?,1) ON CONFLICT(day) DO UPDATE SET n=n+1", (self._today(),))

    # ------------------------------------------------------------------ lookup
    def get(self, set_num):
        """What the UI needs now: {available, links, checked_at, known}. `known` is False when nothing was ever stored."""
        e = self.cached(set_num)
        return {"available": e["available"], "links": e["links"], "checked_at": e["checked_at"], "known": True} if e else \
            {"available": False, "links": [], "checked_at": None, "known": False}

    def refresh(self, set_num):
        """Ask Brickset (if the entry is stale, there is a key and the daily cap allows) and store the answer.
        Returns the entry now valid for the set, or None if unknown. Never raises."""
        old = self.cached(set_num)
        if self.is_fresh(old):
            return old
        if not self.api_key:
            log.info("instructions: no BRICKSET_API_KEY, %s stays %s", set_num, "cached" if old else "unknown")
            return old
        with self._lock:
            if self.calls_today() >= self.cap:
                log.info("instructions: daily cap of %d Brickset calls reached; %s stays %s", self.cap, set_num, "cached" if old else "unknown")
                return old
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            self._count_call()
            try:
                links = self.fetch(self.api_key, set_num)
            except Exception as e:                                            # noqa: BLE001  timeouts, HTTP errors, bad JSON, API errors
                log.warning("instructions: lookup of %s failed (%s); keeping %s", set_num, e, "the cached value" if old else "unknown")
                self._failed[set_num] = self.clock()
                return old
            self._failed.pop(set_num, None)
            return self._store(set_num, links)
