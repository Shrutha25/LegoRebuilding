"""Real sets the user can build, or nearly: part -> sets index built from the Rebrickable CSVs the app already loads.

For an inventory it computes, per set, parts owned / parts needed in two modes (exact colour, any colour) with numpy over a
precomputed postings list per part (a few milliseconds), then lists

    can_build   100% of the parts owned (in exact colour, or in any colour: `colour_match` says which)
    almost      missing at most ALMOST_MAX_MISSING parts, or at least ALMOST_MIN_SHARE of the parts owned; the missing parts are listed

Stickers, minifigure parts and prints are left out of every set's parts count (they cannot be matched from a photo of loose bricks).
"""

import logging
import time

import numpy as np

from backend import config

log = logging.getLogger("lego.setmatch")

ALMOST_MAX_MISSING = 3
ALMOST_MIN_SHARE = 0.9
MAX_SET_PARTS = 400
TOP = 12
POOL = 60          # candidates ranked by instructions state before the best TOP are kept
# Rebrickable part categories: stickers, minifigure heads / upper body / lower body / headwear / bodies, Duplo figures
EXCLUDED_CATEGORIES = {"58", "59", "60", "61", "65", "13", "57"}
EXCLUDED_WORDS = ("sticker", " print", "pattern")


def excluded_parts(parts_df):
    """Part numbers that never count towards a set: stickers, minifigure parts, prints and patterns."""
    out = set()
    if parts_df is None:
        return out
    cols = list(parts_df.columns)
    name_col = "name" if "name" in cols else cols[1]
    cat = parts_df["part_cat_id"] if "part_cat_id" in cols else None
    names = parts_df[name_col].str.lower()
    mask = names.apply(lambda n: any(w in n for w in EXCLUDED_WORDS))
    if cat is not None:
        mask = mask | cat.isin(EXCLUDED_CATEGORIES)
    out.update(parts_df.loc[mask, "part_num"])
    return out


class _Mode:
    """Postings for one matching mode: key -> (set index array, quantity array), and each set's total."""

    def __init__(self, req, set_ids, skip_part):
        self.total = np.zeros(len(set_ids), dtype=np.int32)
        post = {}
        for si, s in enumerate(set_ids):
            for key, q in req[s].items():
                part = key if isinstance(key, str) else key[0]
                if part in skip_part:
                    continue
                self.total[si] += q
                post.setdefault(key, ([], []))
                post[key][0].append(si)
                post[key][1].append(q)
        self.post = {k: (np.array(a, dtype=np.int32), np.array(b, dtype=np.int32)) for k, (a, b) in post.items()}

    def owned(self, have):
        out = np.zeros(len(self.total), dtype=np.int32)
        for key, n in have.items():
            hit = self.post.get(key)
            if hit is not None:
                idx, qty = hit
                np.add.at(out, idx, np.minimum(qty, n))
        return out


class SetMatcher:
    def __init__(self, index, parts_df=None):
        t = time.perf_counter()
        self.index = index
        skip_part = excluded_parts(parts_df)
        sets = index.sets
        names = dict(zip(sets.set_num, sets.name))
        years = dict(zip(sets.set_num, sets.year))
        imgs = dict(zip(sets.set_num, sets.img_url))
        any_req, exact_req = index.req[False], index.req[True]
        self.set_ids = [s for s in any_req if s in names and index.is_buildable(s)]
        self.names, self.years, self.imgs = names, years, imgs
        self.any = _Mode(any_req, self.set_ids, skip_part)
        self.exact = _Mode(exact_req, self.set_ids, skip_part)
        self.skip_part = skip_part
        self.build_seconds = time.perf_counter() - t
        log.info("set matcher built in %.1fs (%d sets)", self.build_seconds, len(self.set_ids))

    # ------------------------------------------------------------ matching
    def _have(self, parts):
        """parts: [{part_num (Rebrickable id), color, qty}] -> (have_any {part: qty}, have_exact {(part, colour): qty})."""
        any_, exact = {}, {}
        for p in parts:
            part = self.index.canon(str(p["part_num"]).strip())
            if part in self.skip_part:
                continue
            qty = int(p.get("qty") or 1)
            colour = (p.get("color") or "").strip().lower()
            any_[part] = any_.get(part, 0) + qty
            exact[(part, colour)] = exact.get((part, colour), 0) + qty
        return any_, exact

    def match(self, parts, names=None, top=TOP, match_color=None):
        """-> {"can_build": [...], "almost": [...], "timings_ms": ...}. `names` is {part number: name} for the missing-parts list."""
        t0 = time.perf_counter()
        have_any, have_exact = self._have(parts)
        own_any, own_exact = self.any.owned(have_any), self.exact.owned(have_exact)
        tot_any, tot_exact = self.any.total, self.exact.total
        ok = (tot_any >= config.MIN_SET_PARTS) & (tot_any <= MAX_SET_PARTS) & (own_any > 0)
        miss_any, miss_exact = tot_any - own_any, tot_exact - own_exact
        done_any = ok & (miss_any == 0)
        done_exact = ok & (tot_exact > 0) & (miss_exact == 0)
        share_any = np.where(tot_any > 0, own_any / np.maximum(tot_any, 1), 0)
        near = ok & ~done_any & ((miss_any <= ALMOST_MAX_MISSING) | (share_any >= ALMOST_MIN_SHARE))
        if match_color is not None:        # one colour rule for everything: exact (part, colour) counts, or part counts only
            own, tot = (own_exact, tot_exact) if match_color else (own_any, tot_any)
            miss = tot - own
            done_any = done_exact = ok & (tot > 0) & (miss == 0)
            share = np.where(tot > 0, own / np.maximum(tot, 1), 0)
            near = ok & ~done_any & ((miss <= ALMOST_MAX_MISSING) | (share >= ALMOST_MIN_SHARE))
            miss_any, miss_exact = miss, miss

        can = np.flatnonzero(done_any | done_exact)
        can = sorted(can, key=lambda i: (not done_exact[i], -int(tot_any[i]), self.set_ids[i]))[:top]
        nearby = sorted(np.flatnonzero(near), key=lambda i: (int(miss_any[i]), -int(tot_any[i]), self.set_ids[i]))[:top]
        t1 = time.perf_counter()
        out = {
            "can_build": [self._row(i, own_any, own_exact, tot_any, tot_exact, have_any, have_exact, names, bool(done_exact[i]) and match_color is not False, match_color) for i in can],
            "almost": [self._row(i, own_any, own_exact, tot_any, tot_exact, have_any, have_exact, names, False, match_color) for i in nearby],
        }
        out["timings_ms"] = {"match": round((t1 - t0) * 1000, 1), "total": round((time.perf_counter() - t0) * 1000, 1)}
        return out

    def _missing(self, set_num, have_any, have_exact, exact, names):
        req = self.index.req[exact][set_num]
        have = have_exact if exact else have_any
        rows = []
        for key, need in req.items():
            part, colour = (key, None) if not exact else key
            if part in self.skip_part:
                continue
            short = need - min(have.get(key, 0), need)
            if short > 0:
                rows.append({"part": part, "name": (names or {}).get(part, part), "colour": colour, "qty": short,
                             "image": self.index.image_for(part, colour)})
        rows.sort(key=lambda r: (-r["qty"], r["part"]))
        return rows[:12]

    def _row(self, i, own_any, own_exact, tot_any, tot_exact, have_any, have_exact, names, exact_complete, match_color=None):
        s = self.set_ids[i]
        use_exact = bool(match_color) if match_color is not None else \
            exact_complete or (tot_exact[i] - own_exact[i]) < (tot_any[i] - own_any[i])
        base = s.split("-")[0]
        miss = int(tot_any[i] - own_any[i]) if not use_exact else int(tot_exact[i] - own_exact[i])
        return {
            "set_num": s, "name": self.names[s], "year": self.years.get(s), "image": self.imgs.get(s) or None,
            "total_parts": int(tot_any[i]),
            "any_colour": {"have": int(own_any[i]), "need": int(tot_any[i])},
            "exact_colour": {"have": int(own_exact[i]), "need": int(tot_exact[i])},
            "colour_match": "exact" if exact_complete else "any",
            "missing_count": miss,
            "missing": [] if miss == 0 else self._missing(s, have_any, have_exact, use_exact, names),
            "rebrickable_url": f"https://rebrickable.com/sets/{s}/",
            "instructions_url": f"https://www.lego.com/service/buildinginstructions/{base}",
        }
