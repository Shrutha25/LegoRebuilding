"""Minimal Rebrickable REST v3 client: cached, ~1 request/second, no-op without a key."""

import asyncio
import time

import httpx

BASE = "https://rebrickable.com/api/v3/lego"


class RebrickableAPI:
    def __init__(self, key, min_interval=1.0):
        self.key = key
        self.min_interval = min_interval
        self._cache = {}
        self._lock = asyncio.Lock()
        self._last = 0.0

    @property
    def enabled(self):
        return bool(self.key)

    async def _get(self, path, params=None):
        ck = (path, tuple(sorted((params or {}).items())))
        if ck in self._cache:
            return self._cache[ck]
        async with self._lock:                        # serialise to respect the rate limit
            if ck in self._cache:
                return self._cache[ck]
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    r = await c.get(f"{BASE}{path}", params=params,
                                    headers={"Authorization": f"key {self.key}"})
                self._last = time.monotonic()
                if r.status_code == 404:
                    data = None
                else:
                    r.raise_for_status()
                    data = r.json()
            except Exception:
                self._last = time.monotonic()
                return None                           # transient failures are not cached
            self._cache[ck] = data
            return data

    async def bricklink_to_rebrickable(self, bricklink_id):
        if not self.enabled:
            return None
        data = await self._get("/parts/", {"bricklink_id": bricklink_id})
        results = (data or {}).get("results") or []
        return results[0]["part_num"] if results else None

    async def part_image(self, part_num):
        if not self.enabled:
            return None
        data = await self._get(f"/parts/{part_num}/")
        return (data or {}).get("part_img_url")
