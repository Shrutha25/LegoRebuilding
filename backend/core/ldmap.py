"""ONE place that maps our identifiers to LDraw: part numbers and colours. Cached, and every miss is logged.

Parts. Recognition (Brickognize) gives BrickLink-style ids, the catalogue (Rebrickable) its own part numbers, LDraw its file
names, and they mostly coincide. `PartMapper.ldraw_id(part)` tries, in order: the id itself, the Rebrickable id (via the same
local rules the app already uses), the Rebrickable "parent" of a print/pattern/mold variant, trailing-letter and print-suffix
rewrites, and finally (async, optional) the Rebrickable API's external LDraw ids. A part is "mapped" when its file exists in
the downloaded LDraw library.

Colours. `ColourMapper.code(name)` maps a colour name ("Light Bluish Gray", "Medium Azure") or Rebrickable/LDraw id to an LDraw
colour code through LDConfig: normalised name first, then the id, then the nearest colour by RGB. Colours that match nothing
are logged and come back as None.
"""

import json
import logging
import re
from pathlib import Path

from backend.core import idmap

log = logging.getLogger("lego.ldmap")


def _norm(name):
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return s.replace("grey", "gray")


class PartMapper:
    def __init__(self, library, known=None, rel_map=None, api=None, cache_path=None):
        self.lib = library
        self.known = known or {}                  # lower-case part number -> Rebrickable part number
        self.rel_map = rel_map or {}              # child part -> parent part (print / pattern / mold variants)
        self.api = api
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache = {}
        self.unmapped = set()
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}

    def _save(self):
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache, indent=0, sort_keys=True), encoding="utf-8")
            except OSError:
                pass

    def _candidates(self, part):
        part = part.strip().lower()
        seen, out = set(), []

        def add(c):
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        add(part)
        rb = self.known.get(part) or idmap.resolve_local(part, self.known)
        add((rb or "").lower())
        for c in list(out):
            parent = self.rel_map.get(c) or self.rel_map.get(c.upper())
            hops = 0
            while parent and hops < 4:
                add(parent.lower())
                parent = self.rel_map.get(parent)
                hops += 1
        for c in list(out):
            for rewrite in idmap._candidates(c):
                add(rewrite)
        return out

    def ldraw_id(self, part):
        """LDraw part number (the .dat name without extension) for `part`, or None. Logs the first miss of each part."""
        key = (part or "").strip().lower()
        if not key:
            return None
        if key in self._cache:
            return self._cache[key] or None
        found = None
        for c in self._candidates(key):
            if self.lib.part_path(c):
                found = c
                break
        self._cache[key] = found or ""
        if not found:
            self.unmapped.add(key)
            log.warning("ldraw: no LDraw part for %r", part)
        self._save()
        return found

    async def ldraw_id_async(self, part):
        """`ldraw_id`, then the Rebrickable API's external LDraw ids for parts the local rules could not map."""
        found = self.ldraw_id(part)
        if found or self.api is None or not self.api.enabled:
            return found
        rb = self.known.get(part.strip().lower()) or part
        data = await self.api._get(f"/parts/{rb}/")
        for ext in ((data or {}).get("external_ids") or {}).get("LDraw", []) or []:
            if self.lib.part_path(str(ext)):
                self._cache[part.strip().lower()] = str(ext).lower()
                self.unmapped.discard(part.strip().lower())
                self._save()
                return str(ext).lower()
        return None

    def coverage(self, part_numbers):
        """(mapped, total) over catalogue part numbers; used to report how many known parts have an LDraw file."""
        total = mapped = 0
        for p in part_numbers:
            total += 1
            if any(self.lib.part_path(c) for c in self._candidates(p)):
                mapped += 1
        return mapped, total


class ColourMapper:
    def __init__(self, library, rb_colors=None):
        """`rb_colors`: rows (id, name, rgb) of Rebrickable's colors.csv, optional (used for the nearest-colour fallback)."""
        self.ld = library.colours()
        self.by_name = {}
        for c in self.ld.values():
            self.by_name.setdefault(_norm(c.name), c.code)
        self.rb = {}
        for cid, name, rgb in rb_colors or []:
            self.rb[_norm(name)] = (str(cid), rgb)
        self._missing = set()

    @staticmethod
    def _rgb(hexs):
        h = hexs.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def code(self, colour):
        """LDraw colour code for a colour name, Rebrickable id or LDraw id; None if nothing fits (logged)."""
        if colour is None or str(colour).strip() == "":
            return None
        key = _norm(colour)
        if key in self.by_name:
            return self.by_name[key]
        if str(colour).strip().lstrip("-").isdigit() and int(colour) in self.ld:
            return int(colour)
        if key in self.rb:                                       # by Rebrickable name: its id is usually the LDraw code
            cid, rgb = self.rb[key]
            if cid.isdigit() and int(cid) in self.ld:
                return int(cid)
            if rgb:
                return self._nearest(rgb)
        if key not in self._missing:
            self._missing.add(key)
            log.warning("ldraw: no LDraw colour for %r", colour)
        return None

    def _nearest(self, rgb_hex):
        r, g, b = self._rgb(rgb_hex)
        best, best_d = None, 1e18
        for c in self.ld.values():
            if c.transparent or not 0 <= c.code < 512:
                continue
            cr, cg, cb = self._rgb(c.hex)
            d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if d < best_d:
                best, best_d = c.code, d
        return best

    def hex(self, code):
        c = self.ld.get(code)
        return c.hex if c else "#808080"

    def name(self, code):
        c = self.ld.get(code)
        return c.name.replace("_", " ") if c else str(code)
