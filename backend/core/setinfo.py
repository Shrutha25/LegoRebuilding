"""Which kind of instructions a set card can offer, and how sets are ranked by that.

    state 1   an open LDraw model (omr.py) exists            -> "View steps" opens our 3D viewer
    state 2   Brickset lists official LEGO instructions      -> "Open LEGO instructions" (instructions.py)
    state 3   neither                                        -> no button; a Rebrickable link instead
    None      not known yet (lookup still running)           -> the card reserves the space and fills in later

`SetInfo.state()` reads caches only, so a request never waits for the network. `prefetch()` queues sets for one background
thread that checks the OMR first (state 1 needs no Brickset call) and then asks Brickset.
"""

import asyncio
import logging
import queue
import threading

from backend.core import omr

log = logging.getLogger("lego.setinfo")


def card_state(has_model, instr):
    """has_model: True/False/None (OMR); instr: InstructionsService.get() dict. -> 1, 2, 3 or None (unknown)."""
    if has_model:
        return 1
    if has_model is None:
        return None
    if instr["available"] and instr["links"]:
        return 2
    return 3 if instr["known"] else None


def rank_key(row):
    """Sort key: instructions state (1, 2, then 3 and unknown), completeness, part count; set number for a stable order."""
    state = (row.get("instructions") or {}).get("state") or 3
    total = max(1, row.get("total_parts") or 1)
    completeness = 1 - (row.get("missing_count") or 0) / total
    return (state, -completeness, -total, row.get("set_num", ""))


class SetInfo:
    def __init__(self, instructions, omr_cache_dir, omr_base_url, fetch_omr=omr.fetch_set_model):
        self.instructions, self.omr_dir, self.omr_base = instructions, omr_cache_dir, omr_base_url
        self.fetch_omr = fetch_omr
        self._queue, self._queued = queue.Queue(), set()
        self._no_model = set()               # checked this run, no usable model (also when the OMR could not be reached)
        self._thread = None
        self._lock = threading.Lock()

    def info(self, set_num):
        """{state, links, pending} for a set, from caches only."""
        has_model = omr.cached_status(set_num, self.omr_dir)
        if has_model is None and set_num in self._no_model:
            has_model = False
        instr = self.instructions.get(set_num)
        state = card_state(has_model, instr)
        return {"state": state, "links": instr["links"] if state == 2 else [], "pending": state is None}

    def annotate(self, rows):
        """Add the optional `instructions` field to match rows and queue the sets not known yet."""
        todo = []
        for row in rows:
            row["instructions"] = self.info(row["set_num"])
            if row["instructions"]["pending"] or self.instructions.needs_refresh(row["set_num"]) and row["instructions"]["state"] != 1:
                todo.append(row["set_num"])
        self.prefetch(todo)
        return rows

    def prefetch(self, set_nums):
        with self._lock:
            for s in set_nums:
                if s not in self._queued:
                    self._queued.add(s)
                    self._queue.put(s)
            if self._queued and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._work, daemon=True, name="set-info")
                self._thread.start()

    def _work(self):
        while True:
            try:
                s = self._queue.get(timeout=30)
            except queue.Empty:
                return
            try:
                self.check(s)
            except Exception as e:                                               # noqa: BLE001  a lookup never kills the worker
                log.warning("set info for %s failed: %s", s, e)
            finally:
                with self._lock:
                    self._queued.discard(s)

    def check(self, set_num):
        """Fill the caches for one set: OMR first, Brickset only when there is no usable model."""
        if omr.cached_status(set_num, self.omr_dir) is None:
            model = asyncio.run(self.fetch_omr(set_num, self.omr_dir, self.omr_base))
            if model is not None and model.licence_ok:
                return
            self._no_model.add(set_num)
        elif omr.cached_status(set_num, self.omr_dir):
            return
        self.instructions.refresh(set_num)
