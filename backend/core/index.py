"""In-memory match index: set requirements + inverted index (part -> sets), cached to disk.

The scoring itself is `rank_sets` from match.py, unchanged; the index only narrows
the candidate sets to those sharing at least one part with the user's pieces.
"""

import logging
import os
import pickle
import threading
import time
from collections import defaultdict

import pandas as pd

from backend import config
from backend.core.match import (RB_FILES, build_set_requirements, canonical_map, exclusion_report, rank_sets,
                                skipped_theme_ids)

log = logging.getLogger("lego.index")

CACHE_VERSION = 2
CACHE_NAME = "index_cache.pkl"
MIN_COVERAGE = 1.0         # library default: only sets that can be built completely
TOP = 10


def _read(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, f"{name}.csv"), dtype=str, keep_default_na=False)


def source_key(data_dir):
    """Cache key: (name, mtime, size) of every source CSV."""
    key = [CACHE_VERSION, config.EXCLUDED_THEMES, config.EXCLUDED_THEME_WORDS]   # the skipped theme ids are cached
    for name in RB_FILES:
        st = os.stat(os.path.join(data_dir, f"{name}.csv"))
        key.append((name, st.st_mtime_ns, st.st_size))
    return key


def _plain(req):
    return {s: dict(d) for s, d in req.items()}


def _inverted(req):
    inv = defaultdict(list)
    for set_num, need in req.items():
        for k in need:
            inv[k].append(set_num)
    return dict(inv)


class MatchIndex:
    def __init__(self, data):
        self.sets = data["sets"]                 # DataFrame, set_num as column
        self.skip = data["skip"]
        self.rel_map = data["rel_map"]           # child -> parent (P/T/M/A relations)
        self.req = data["req"]                   # {match_color: {set: {key: qty}}}
        self.inv = data["inv"]                   # {match_color: {key: [sets]}}
        self.part_imgs = data["part_imgs"]       # {(part, color_name): url}
        self.part_img_any = data["part_img_any"]  # {part: url}
        self._theme = dict(zip(self.sets.set_num, self.sets.theme_id))
        nparts = pd.to_numeric(self.sets["num_parts"], errors="coerce").fillna(0) if "num_parts" in self.sets else None
        self._tiny = set() if nparts is None else set(self.sets.set_num[nparts < config.MIN_SET_PARTS])      # Rebrickable's own count: key rings, polybags...

    def is_buildable(self, set_num):
        """False for sets in an excluded theme or with fewer than MIN_SET_PARTS parts."""
        theme = self._theme.get(set_num)
        return theme is not None and theme not in self.skip and set_num not in self._tiny

    # -- build / cache -------------------------------------------------
    @classmethod
    def build(cls, data_dir):
        rb = {n: _read(data_dir, n) for n in RB_FILES if n != "parts"}
        rels = rb["part_relationships"]
        canon = canonical_map(rels)
        keep = rels[rels.rel_type.isin(["P", "T", "M", "A"])]
        req = {mc: _plain(build_set_requirements(rb, canon, mc)) for mc in (False, True)}

        ip = rb["inventory_parts"]
        colors = dict(zip(rb["colors"].id, rb["colors"].name.str.lower()))
        imgs = ip[ip.img_url != ""].drop_duplicates(["part_num", "color_id"])
        part_imgs, any_img = {}, {}
        for part, cid, url in zip(imgs.part_num, imgs.color_id, imgs.img_url):
            p = canon(part)
            part_imgs.setdefault((p, colors.get(cid, "")), url)
            any_img.setdefault(p, url)
        return cls({
            "sets": rb["sets"], "skip": skipped_theme_ids(rb["themes"]),
            "rel_map": dict(zip(keep.child_part_num, keep.parent_part_num)),
            "req": req, "inv": {mc: _inverted(r) for mc, r in req.items()},
            "part_imgs": part_imgs, "part_img_any": any_img,
        })

    @classmethod
    def load(cls, data_dir):
        idx = cls._load(data_dir)
        try:
            if "num_parts" not in idx.sets:
                return idx
            rep = exclusion_report(idx.sets, _read(data_dir, "themes"), idx.skip, config.MIN_SET_PARTS)
            log.info("set exclusion: %d of %d sets left out (%d by theme, %d with < %d parts); top themes: %s",
                     rep["excluded"], rep["total"], rep["by_theme"], rep["small"], config.MIN_SET_PARTS,
                     "; ".join(f"{t} {n}" for t, n in rep["top_themes"]))
        except Exception as e:                                          # noqa: BLE001  a log line must never stop start-up
            log.warning("exclusion report failed: %s", e)
        return idx

    @classmethod
    def _load(cls, data_dir):
        """Load from the disk cache if it matches the CSVs, else rebuild and re-cache."""
        key = source_key(data_dir)
        path = os.path.join(data_dir, CACHE_NAME)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    blob = pickle.load(f)
                if blob.get("key") == key:
                    return cls(blob["data"])
            except Exception:
                pass                                  # corrupt/incompatible cache: rebuild
        idx = cls.build(data_dir)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"key": key, "data": {k: v for k, v in idx.__dict__.items() if not k.startswith("_")}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return idx

    # -- matching ------------------------------------------------------
    def canon(self, p):
        seen, m = set(), self.rel_map
        while p in m and p not in seen:
            seen.add(p)
            p = m[p]
        return p

    def _key(self, part, color, match_color):
        p = self.canon(part)
        return (p, (color or "").strip().lower()) if match_color else p

    def image_for(self, part, color):
        return (self.part_imgs.get((part, color or "")) if color is not None else None) \
            or self.part_img_any.get(part)

    def match(self, parts, match_color=False, max_parts=150, top=TOP, min_coverage=MIN_COVERAGE):
        """parts: [{part_num (Rebrickable id), color, qty}] -> ranked sets."""
        have = defaultdict(int)
        for p in parts:
            have[self._key(p["part_num"].strip(), p.get("color"), match_color)] += int(p.get("qty") or 1)

        inv = self.inv[match_color]
        cands = set()
        for k in have:
            cands.update(inv.get(k, ()))
        if not cands:
            return []
        # Pre-filter with the same size/theme rules rank_sets applies, so fewer pandas row
        # lookups happen there. Results are identical.
        all_req = self.req[match_color]
        req = {}
        for s in cands:
            if not self.is_buildable(s):
                continue
            if config.MIN_SET_PARTS <= sum(all_req[s].values()) <= max_parts:
                req[s] = all_req[s]
        if not req:
            return []
        sets = self.sets[self.sets.set_num.isin(cands)]
        results = rank_sets(have, req, sets, self.skip, max_parts, config.MIN_SET_PARTS, min_coverage, top)
        for r in results:
            for m in r["missing"]:
                m["image"] = self.image_for(m["part"], m["color"])
            r["rebrickable_url"] = r.pop("rebrickable")
            r["instructions_url"] = r.pop("instructions")
        return results


class DataStore:
    """Holds the parts table and match index; loads them in a background thread."""

    def __init__(self, data_dir):
        self.data_dir = str(data_dir)
        self.parts = None            # DataFrame part_num,name
        self.known_ids = {}
        self.index = None
        self.error = None
        self.load_seconds = None
        self._names = None

    @property
    def parts_loaded(self):
        return self.parts is not None

    @property
    def data_loaded(self):
        return self.index is not None

    def start(self):
        t = threading.Thread(target=self._load, daemon=True, name="rb-loader")
        t.start()
        return t

    def _load(self):
        t0 = time.time()
        try:
            self.parts = _read(self.data_dir, "parts")[["part_num", "name"]]
            self.known_ids = {n.lower(): n for n in self.parts.part_num}
            self.index = MatchIndex.load(self.data_dir)
            self.load_seconds = round(time.time() - t0, 1)
        except FileNotFoundError:
            self.error = ("Rebrickable data not found. Run: "
                          "python -m backend.core.fetch_data")
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def part_names(self):
        if self._names is None and self.parts is not None:
            self._names = dict(zip(self.parts.part_num, self.parts.name))
        return self._names or {}

    def search_parts(self, q, limit=30):
        if self.parts is None or not q.strip():
            return []
        q = q.strip().lower()
        df = self.parts
        num = df.part_num.str.lower()
        hit = df[num.str.startswith(q) | num.str.contains(q, regex=False)
                 | df.name.str.lower().str.contains(q, regex=False)]
        # exact/prefix number matches first
        order = hit.part_num.str.lower().map(lambda n: (n != q, not n.startswith(q), len(n)))
        hit = hit.loc[order.sort_values().index].head(limit)
        any_img = self.index.part_img_any if self.index else {}
        return [{"part_num": r.part_num, "name": r.name, "image": any_img.get(r.part_num)}
                for r in hit.itertuples()]
