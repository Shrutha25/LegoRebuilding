"""Per-request timing: where does the time go between "Scan" and the last card?

    with timing.request("scan_job"):            # one trace per HTTP request (middleware) or background job
        with timing.stage("detect", mode="opencv"):
            ...
        timing.count("gemini_calls")

A trace keeps every stage (name, start offset, duration, small metadata) and counters. When the request ends it is logged as one line
(logger "lego.timing") and appended as a JSON line to TIMING_LOG (default <repo>/timings.jsonl; TIMING_LOG= disables the file). Stages nest and
run in threads and asyncio tasks: the trace travels in a contextvar, which asyncio tasks copy and run_in_threadpool preserves. Outside a trace
`stage` costs nothing. Read the log with `python -m backend.timing_report`.

Stage names used by the app (dots group them):
    detect                   piece detection (meta mode)             brickognize          one Brickognize request
    identify                 all crops through Brickognize           crops, save_crops    cutting and writing crop images
    map_parts, sets.match, sets.rank, nearby.match, nearby.rank      the official-set and nearby-set sources
    recipes.inventory        parts -> LDraw inventory (may ask Rebrickable)
    recipes.suggest          the whole concept flow, containing: concepts, solver (per concept), checks (per concept), generate (on demand)
    recipes.store            LDraw conversion (ldr), packing (ldr.pack) and thumbnail rendering (thumbnail) per build
    gemini.<purpose>         one Gemini API request (retries are separate stages); gemini.budget_wait  time waiting for the quota budget
    ideas, aibuild.generate  the AI ideas and free-form AI build endpoints
"""

import contextvars
import json
import logging
import threading
import time
import uuid
from contextlib import contextmanager

log = logging.getLogger("lego.timing")
_current = contextvars.ContextVar("lego_trace", default=None)
_write_lock = threading.Lock()


class Trace:
    def __init__(self, name, **meta):
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.meta = dict(meta)
        self.t0 = time.perf_counter()
        self.wall = time.time()
        self.stages = []                        # {"stage", "start_ms", "ms", ...meta}
        self.counters = {}
        self._lock = threading.Lock()

    def add(self, stage, start, end, meta):
        row = {"stage": stage, "start_ms": round((start - self.t0) * 1000, 1), "ms": round((end - start) * 1000, 1)}
        row.update(meta)
        with self._lock:
            self.stages.append(row)

    def count(self, key, n=1):
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + n

    def sums(self):
        """Total milliseconds per stage name (stages of the same name added up; nested stages are counted in their parents too)."""
        out = {}
        for s in self.stages:
            out[s["stage"]] = round(out.get(s["stage"], 0.0) + s["ms"], 1)
        return out

    def row(self, total_ms):
        return {"id": self.id, "name": self.name, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.wall)), "total_ms": round(total_ms, 1),
                "meta": self.meta, "counters": self.counters, "sums": self.sums(), "stages": self.stages}


def current():
    return _current.get()


def begin(name, **meta):
    trace = Trace(name, **meta)
    return trace, _current.set(trace)


def end(trace, token, **meta):
    """Close a trace: log one line, append the JSON row, return the row."""
    total = (time.perf_counter() - trace.t0) * 1000
    try:
        _current.reset(token)
    except ValueError:                           # reset from another context (a task that outlived its request): the value is simply left alone
        pass
    trace.meta.update(meta)
    row = trace.row(total)
    top = sorted(row["sums"].items(), key=lambda kv: -kv[1])[:6]
    log.info("%s %s %.0f ms  %s  %s", trace.name, trace.id, total, " ".join(f"{k}={v:.0f}" for k, v in top),
             " ".join(f"{k}={v}" for k, v in trace.counters.items()))
    _write(row)
    return row


@contextmanager
def request(name, **meta):
    trace, token = begin(name, **meta)
    try:
        yield trace
    finally:
        end(trace, token)


@contextmanager
def stage(name, **meta):
    trace = _current.get()
    if trace is None:
        yield
        return
    t = time.perf_counter()
    try:
        yield
    finally:
        trace.add(name, t, time.perf_counter(), meta)


def count(key, n=1):
    trace = _current.get()
    if trace is not None:
        trace.count(key, n)


def _write(row):
    from backend import config
    path = config.TIMING_LOG
    if not path:
        return
    try:
        with _write_lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:                          # timing must never break a request
        log.debug("timing log not written: %s", e)
