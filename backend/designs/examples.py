"""Inspiration for the concept prompt: small real LEGO models from the OMR, drawn as the same coarse sketches Gemini is asked to write.

    bank = ExampleBank.load(omr_index, library, folder)       # built once (a couple of seconds), cached as a pickle
    bank.pick(cells, mix, n=4)                                # the n examples that best match the pile in size and mix of parts

A model's sketch comes from its geometry: every part's LDraw bounding box, turned by its matrix, fills the stud x stud x brick cells it covers (a coarse
voxelisation: studs are 20 LDU wide, a brick is 24 LDU tall, y is up when negated). Models bigger than the sketch limits, minifigures and instruction
steps are left out.
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np

from backend.core.ldraw_lib import _matrix, part_bounds
from backend.core.partdims import dims_for
from backend.designs.affordance import role_class
from backend.core.omr_index import split_blocks
from backend.designs import sketch as sk

log = logging.getLogger("lego.designs.examples")
BOARD = (10, 10, 7)                     # largest example: width, depth, layers
SKIP = re.compile(r"minifig|figure|\bstep|page|bag|sticker|sheet|base ?plate|baseplate", re.I)
MIN_PARTS, MAX_PARTS = 6, 40
VERSION = 2                             # the cache key: change when what an example stores changes


def _placements(blocks, block, library, depth=0, matrix=None, out=None):
    """[(part, 4x4 matrix)] of one block, sub-models expanded (at most 6 deep, 300 parts)."""
    matrix = np.eye(4) if matrix is None else matrix
    out = [] if out is None else out
    names = {re.sub(r"\.(ldr|mpd|dat)$", "", n.strip().replace("\\", "/").lower()): i for i, (n, _) in enumerate(blocks) if not n.strip().lower().endswith(".dat")}
    for line in blocks[block][1]:
        t = line.split(None, 14)
        if len(t) != 15 or t[0] != "1" or len(out) > 300:
            continue
        ref = re.sub(r"\.(ldr|mpd|dat)$", "", t[14].strip().replace("\\", "/").lower())
        try:
            m = matrix @ _matrix(t[2:14])
        except ValueError:
            continue
        if ref in names and depth < 6 and names[ref] != block:
            _placements(blocks, names[ref], library, depth + 1, m, out)
        elif ref not in names:
            out.append((ref.rsplit("/", 1)[-1], m))
    return out


def omr_sketch(text, block, library, canon=None, mix=None):
    """A Sketch of one model of an MPD from its geometry, or None: too big or small, or not made of parts laid on the stud grid (tilted parts, parts of
    unknown size). Each part's LDraw bounding box, turned by its matrix, fills the stud x stud x brick cells it covers. `mix`, a dict, is filled with the
    number of parts per role class (affordance.role_class) so an example can be matched to a pile."""
    blocks = split_blocks(text)
    cells, used, skipped = set(), 0, 0
    for part, m in _placements(blocks, block, library):
        b = part_bounds(library, canon(part) if canon else part)
        r = m[:3, :3]
        axis_aligned = b is not None and np.allclose(np.abs(r).max(axis=1), 1, atol=0.02) and abs(abs(r[1, 1]) - 1) < 0.02
        if not axis_aligned:
            skipped += 1
            continue
        if mix is not None:
            c = role_class(dims_for(canon(part) if canon else part))
            mix[c] = mix.get(c, 0) + 1
        lo, hi = b
        corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pts = (corners @ m.T)[:, :3]
        (x0, y0, z0), (x1, y1, z1) = pts.min(axis=0), pts.max(axis=0)
        up0, up1 = -y1, -y0                                         # LDraw's y points down
        xs = range(round(x0 / 20), max(round(x1 / 20), round(x0 / 20) + 1))
        zs = range(round(z0 / 20), max(round(z1 / 20), round(z0 / 20) + 1))
        ks = range(int(np.floor(up0 / 24 + 0.12)), max(int(np.ceil(up1 / 24 - 0.12)), int(np.floor(up0 / 24 + 0.12)) + 1))
        cells.update((x, z, k) for x in xs for z in zs for k in ks)
        used += 1
    if not cells or skipped > 0.2 * (used + skipped):
        return None
    mx, mz, mk = (min(c[i] for c in cells) for i in range(3))
    cells = {(x - mx, z - mz, k - mk) for x, z, k in cells}
    try:
        return sk.parse({"layers": sk.render(sk.Sketch(frozenset(cells), max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1, max(c[2] for c in cells) + 1))["layers"]},
                        BOARD[0], BOARD[1], BOARD[2], min_cells=6)
    except sk.SketchError:
        return None


def plausible(s):
    """A sketch worth showing: compact, one connected piece, every cell above the ground resting on a cell below."""
    cells = s.cells
    if len(cells) < 0.3 * s.width * s.depth * s.layers:
        return False
    if sum(1 for x, z, k in cells if k > 0 and (x, z, k - 1) not in cells) > 0.1 * len(cells):
        return False
    seen, todo = set(), [next(iter(cells))]
    while todo:
        c = todo.pop()
        if c in seen:
            continue
        seen.add(c)
        x, z, k = c
        todo += [n for n in ((x + 1, z, k), (x - 1, z, k), (x, z + 1, k), (x, z - 1, k), (x, z, k + 1), (x, z, k - 1)) if n in cells and n not in seen]
    return len(seen) == len(cells)


class ExampleBank:
    def __init__(self, examples=()):
        self.examples = list(examples)

    def __len__(self):
        return len(self.examples)

    def pick(self, cells, mix=None, n=4, seed=0):
        """The n examples that best match the pile: a size near 75% of its volume and a similar mix of parts (kind and size classes), from different sets.
        The same pile always gets the same examples."""
        total = sum((mix or {}).values()) or 1

        def distance(e):
            em, et = e.get("mix") or {}, sum((e.get("mix") or {}).values()) or 1
            parts = sum(abs(em.get(k, 0) / et - (mix or {}).get(k, 0) / total) for k in set(em) | set(mix or {})) / 2          # 0 (same mix) .. 1
            return abs(e["filled"] - 0.75 * cells) / max(cells, 1) + (parts if mix else 0), e["set"]
        out, used = [], set()
        for e in sorted(self.examples, key=distance):
            if e["set"] not in used:
                used.add(e["set"])
                out.append(e)
            if len(out) == n:
                break
        return out

    @classmethod
    def build(cls, index, library, folder, canon=None, limit=400):
        out = []
        cands = [b for b in index.builds if b.main and MIN_PARTS <= b.parts <= MAX_PARTS and not SKIP.search(b.title)]
        cands.sort(key=lambda b: (b.set_num))
        step = max(1, len(cands) // limit)
        for b in cands[::step][:limit]:
            path = Path(folder) / f"{b.set_num}.mpd"
            if not path.exists():
                continue
            mix = {}
            try:
                s = omr_sketch(path.read_text(encoding="utf-8", errors="replace"), b.block, library, canon, mix)
            except Exception as e:                                       # noqa: BLE001  one odd file must not stop the bank
                log.debug("example %s skipped: %s", b.id, e)
                continue
            if s is not None and 6 <= s.filled <= 150 and plausible(s):
                out.append({"set": b.set_num, "title": b.title, "parts": b.parts, "filled": s.filled, "sketch": sk.render(s), "mix": mix})
        log.info("example bank: %d models sketched", len(out))
        return cls(out)

    @classmethod
    def load(cls, index, library, folder, cache_path, canon=None, key=None):
        cache = Path(cache_path)
        want = (len(index), key, VERSION)
        if cache.exists():
            try:
                blob = pickle.loads(cache.read_bytes())
                if blob.get("key") == want:
                    return cls(blob["examples"])
            except Exception:                                            # noqa: BLE001
                pass
        bank = cls.build(index, library, folder, canon)
        cache.write_bytes(pickle.dumps({"key": want, "examples": bank.examples}))
        return bank
