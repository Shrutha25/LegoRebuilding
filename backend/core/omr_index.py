"""Real LEGO sub-builds you can make from your pieces: an index of the LDraw Official Model Repository (OMR) and a matcher.

Every OMR model is one MPD file: a main model and, usually, several sub-models (a vehicle, a figure, a tree ...), each one a block that starts
with `0 FILE name` and lists its parts and its own sub-models as type-1 lines. This module flattens every model and every sub-model into a parts
list (one instance, sub-models expanded, colours inherited: colour 16 is "the colour of the parent"), keeps those of 6 to 80 parts, and matches
them against an inventory:

    index = OmrIndex.load()                        # pickle cache next to the models, built from build_cache/omr/*.mpd when stale
    res = index.match(have, match_color=True)      # have: {(ld_part, ld_colour or None): qty}
    res["buildable"], res["almost"]                # sub-builds the pieces cover / miss by 1-2 parts, with the missing parts listed

The same rule as every other build (inventory_check.can_build): shape always matters; with colour on the demand is counted per (part, colour), with
colour off per part. A part the model draws in colour 16 at the top level (no colour of its own) can be any colour. Only models with an open licence
(CCAL / CC BY, omr.OmrModel.licence_ok) are indexed; authors and licence stay with each sub-build for attribution. Matching is a loop over a few
thousand small parts lists with an early exit, well under 100 ms.
"""

import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.core import omr

log = logging.getLogger("lego.omr_index")

INDEX_VERSION = 5
MIN_PARTS, MAX_PARTS = 6, 80
MAX_MISSING = 2
INHERIT = 16                       # LDraw "main colour": the colour of the line that refers to this one
_FILE = re.compile(r"^\s*0\s+FILE\s+(.+?)\s*$", re.I)
_NAME_NOISE = re.compile(r"\.(ldr|mpd|dat)$", re.I)
# sub-models that are not builds: the cumulative steps and pages an instruction exporter writes ("21302 - Step 56-9"), bags, sticker sheets
_JUNK = re.compile(r"\b(steps?\d*|pages?\d*|bags?\d*|stickers?|sheets?|instructions?)\b|\bstep\d|\bpage\d|\bsub\s?step", re.I)


@dataclass
class SubBuild:
    id: str
    set_num: str
    block: int                     # 0 = the main model, then the sub-models in file order
    title: str
    parts: int                     # parts in one instance
    need: tuple                    # ((ld part, colour code, qty), ...)
    authors: tuple = ()
    licence: tuple = ()
    set_name: str = ""
    main: bool = False


def _norm(name):
    return _NAME_NOISE.sub("", name.strip().replace("\\", "/").lower())


def split_blocks(text):
    """[(file name as written, [lines])] for an MPD; a plain LDR file is one block."""
    blocks, name, lines = [], None, []
    for raw in text.replace("\r\n", "\n").split("\n"):
        m = _FILE.match(raw)
        if m:
            if name is not None:
                blocks.append((name, lines))
            name, lines = m.group(1), []
        elif name is None:
            name, lines = "model.ldr", [raw]
        else:
            lines.append(raw)
    if name is not None:
        blocks.append((name, lines))
    return blocks


def _refs(lines):
    """[(colour, file)] of the type-1 lines of a block."""
    out = []
    for line in lines:
        t = line.split(None, 14)
        if len(t) == 15 and t[0] == "1":
            try:
                out.append((int(t[1]), t[14].strip()))
            except ValueError:
                continue
    return out


def title_of(name, lines):
    for line in lines[:6]:
        s = line.strip()
        if s.startswith("0 ") and not re.match(r"0\s+(Name:|Author:|!|FILE|BFC|STEP|ROTSTEP|//)", s, re.I):
            t = s[2:].strip()
            if t:
                return t
    return _NAME_NOISE.sub("", name.replace("_", " ")).strip()


def flatten(blocks):
    """{block index: {(part, colour): qty}} for every model block (one instance of each, sub-models expanded)."""
    names = {_norm(n): i for i, (n, _) in enumerate(blocks) if not n.strip().lower().endswith(".dat")}
    refs = [_refs(lines) for _, lines in blocks]
    cache = {}

    def counts(i, stack=()):
        if i in cache:
            return cache[i]
        out = {}
        for colour, ref in refs[i]:
            key = _norm(ref)
            sub = names.get(key)
            if sub is not None and sub not in stack and sub != i:
                for (part, c), q in counts(sub, stack + (i,)).items():
                    c2 = colour if c == INHERIT else c
                    out[(part, c2)] = out.get((part, c2), 0) + q
            elif sub is None:
                part = key.rsplit("/", 1)[-1]
                out[(part, colour)] = out.get((part, colour), 0) + 1
        cache[i] = out
        return out
    return {i: counts(i) for i in names.values()}


def remap(counts, canon):
    """{(part, colour): qty} with every part number passed through `canon` (renamed parts: 3139 -> 3139b), equal ones added up."""
    if canon is None:
        return counts
    out = {}
    for (part, col), q in counts.items():
        k = (canon(part), col)
        out[k] = out.get(k, 0) + q
    return out


def index_model(set_num, text, set_name="", canon=None):
    """SubBuilds of one OMR file (empty when its licence is not an open one). `canon` maps a part number to the current one."""
    authors, licences = omr.parse_header(text)
    if not omr.OmrModel(set_num, "", authors, licences).licence_ok:
        return []
    blocks = split_blocks(text)
    flat = flatten(blocks)
    out, seen = [], set()
    for i, c in sorted(flat.items()):
        c = remap(c, canon)
        n = sum(c.values())
        if not MIN_PARTS <= n <= MAX_PARTS:
            continue
        need = tuple(sorted((p, col, q) for (p, col), q in c.items()))
        if need in seen:
            continue
        seen.add(need)
        title = title_of(*blocks[i])
        if i and (_JUNK.search(title) or _JUNK.search(_NAME_NOISE.sub("", blocks[i][0]))):
            continue
        out.append(SubBuild(f"sub-{set_num}-{i}", set_num, i, set_name if i == 0 and set_name else title, n, need, tuple(authors), tuple(licences),
                            set_name, i == 0))
    return out


def extract_model(text, block):
    """The MPD of one sub-model made standalone: its block first, then the blocks it needs (sub-models and unofficial parts)."""
    blocks = split_blocks(text)
    names = {_norm(n): i for i, (n, _) in enumerate(blocks)}
    refs = [_refs(lines) for _, lines in blocks]
    order, todo = [], [block]
    while todo:
        i = todo.pop(0)
        if i in order:
            continue
        order.append(i)
        todo += [names[_norm(r)] for _, r in refs[i] if _norm(r) in names and names[_norm(r)] not in order]
    keep = [blocks[i] for i in order]
    header = [line for line in blocks[0][1] if line.strip().startswith(("0 Author:", "0 !LICENSE", "0 !LDRAW_ORG", "0 Name:"))][:6] if blocks else []
    out = []
    for k, (name, lines) in enumerate(keep):
        out.append(f"0 FILE {name}")
        out += lines if k else [ln for ln in lines if not ln.strip().startswith("0 Name:")] + [h for h in header if h not in lines]
    return "\n".join(out) + "\n"


def _prep(b):
    """Per part: (part, total qty, ((colour, qty) fixed ...), qty in "inherit" colour)."""
    per = {}
    for part, col, q in b.need:
        e = per.setdefault(part, [0, {}, 0])
        e[0] += q
        if col == INHERIT:
            e[2] += q
        else:
            e[1][col] = e[1].get(col, 0) + q
    return tuple((p, e[0], tuple(e[1].items()), e[2]) for p, e in sorted(per.items(), key=lambda kv: -kv[1][0]))


class OmrIndex:
    def __init__(self, builds=(), set_names=None):
        self.builds = list(builds)
        self._prep = [_prep(b) for b in self.builds]
        self.ready = True

    def __len__(self):
        return len(self.builds)

    # ------------------------------------------------------------ matching
    def missing(self, i, have_part, have_pc, match_color):
        """(count of parts missing, [(part, colour, qty)] missing) for build i, or (None, None) when more than MAX_MISSING are missing."""
        total, rows = 0, []
        for part, qty, fixed, wild in self._prep[i]:
            if not match_color:
                short = qty - have_part.get(part, 0)
                if short > 0:
                    total += short
                    rows.append((part, None, short))
            else:
                free = have_part.get(part, 0)
                for col, q in fixed:
                    h = have_pc.get((part, col), 0)
                    if h < q:
                        total += q - h
                        rows.append((part, col, q - h))
                    free -= min(h, q)
                if wild and free < wild:
                    total += wild - max(0, free)
                    rows.append((part, INHERIT, wild - max(0, free)))
            if total > MAX_MISSING:
                return None, None
        return total, rows

    def match(self, have, match_color=True, top=12, per_set=2):
        """have: {(ld part, ld colour or None): qty}. -> {"buildable": [...], "almost": [...], "counts": {...}} with SubBuild rows and, for almost,
        the missing parts. The lists are ranked (more parts first, at most `per_set` per set); counts are over ALL matches."""
        have_part, have_pc = {}, {}
        for (part, col), q in have.items():
            have_part[part] = have_part.get(part, 0) + q
            have_pc[(part, col)] = have_pc.get((part, col), 0) + q
        full, near = [], []
        for i, b in enumerate(self.builds):
            if b.parts - MAX_MISSING > sum(have_part.values()):          # the pile has fewer pieces than the build needs even when 2 are missing
                continue
            m, rows = self.missing(i, have_part, have_pc, match_color)
            if m is None:
                continue
            (full if m == 0 else near).append((b, rows))

        def rank(items):
            items.sort(key=lambda t: (-t[0].parts, t[0].set_num, t[0].block))
            out, per = [], {}
            for b, rows in items:
                if per.get(b.set_num, 0) >= per_set:
                    continue
                per[b.set_num] = per.get(b.set_num, 0) + 1
                out.append((b, rows))
            return out[:top]
        return {"buildable": rank(full), "almost": rank(near),
                "counts": {"buildable": len(full), "almost": len(near),
                           "smallest_buildable": min((b.parts for b, _ in full), default=None),
                           "smallest_almost": min((b.parts for b, _ in near), default=None)}}

    # ------------------------------------------------------------ building and caching
    @classmethod
    def build(cls, folder, set_names=None, log_every=200, canon=None):
        t = time.perf_counter()
        builds, files = [], sorted(Path(folder).glob("*.mpd"))
        for n, path in enumerate(files, 1):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                builds += index_model(path.stem, text, (set_names or {}).get(path.stem, ""), canon)
            except Exception as e:                                          # noqa: BLE001  one bad file must not stop the index
                log.warning("omr index: %s skipped (%s: %s)", path.name, type(e).__name__, e)
            if log_every and n % log_every == 0:
                log.info("omr index: %d/%d files, %d sub-builds", n, len(files), len(builds))
        log.info("omr index built: %d files -> %d sub-builds of %d-%d parts in %.1fs", len(files), len(builds), MIN_PARTS, MAX_PARTS, time.perf_counter() - t)
        return cls(builds)

    @staticmethod
    def source_key(folder, extra=None):
        files = sorted(Path(folder).glob("*.mpd"))
        return (INDEX_VERSION, extra, len(files), max((f.stat().st_mtime_ns for f in files), default=0), sum(f.stat().st_size for f in files))

    @classmethod
    def load(cls, folder, cache_path=None, set_names=None, canon=None, canon_key=None):
        """The index from its pickle when the model files (and the library `canon` comes from, `canon_key`) have not changed, else rebuilt."""
        folder = Path(folder)
        if not folder.exists():
            return cls([])
        cache = Path(cache_path) if cache_path else folder.parent / "omr_index.pkl"
        key = cls.source_key(folder, canon_key)
        if cache.exists():
            try:
                blob = pickle.loads(cache.read_bytes())
                if blob.get("key") == key:
                    return cls(blob["builds"])
            except Exception:                                               # noqa: BLE001
                pass
        idx = cls.build(folder, set_names, canon=canon)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps({"key": key, "builds": idx.builds}, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(cache)
        return idx
