"""The official LDraw parts library, read straight from its zip (ldraw_data/complete.zip), and a packer.

The viewer never gets the whole library (it has ~37 000 files): `pack_mpd` takes a model (.ldr / .mpd text) and returns ONE
.mpd that embeds the model plus only the part and primitive files it uses, named exactly the way three.js' LDrawLoader looks
them up (lower case, forward slashes, `s\\x.dat` -> `parts/s/x.dat`, `48\\x.dat` -> `p/48/x.dat`). Packed models are cached by
the caller (builds.py).

Also here: the LDConfig colour table, part descriptions, and `part_bounds`, a small geometry walk that measures a part so the
dimension table can be checked against the real geometry.

Download the library once with `python -m backend.core.fetch_ldraw`.
"""

import re
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_ZIP = "complete.zip"

# Where a file referenced from a model or part is searched for, in order (LDraw's own lookup order).
_SEARCH = ("parts/", "p/", "parts/s/", "p/48/", "p/8/", "models/")


class LibraryMissing(Exception):
    """The LDraw library has not been downloaded."""


@dataclass(frozen=True)
class LDColour:
    code: int
    name: str                 # as in LDConfig, e.g. "Light_Bluish_Grey"
    hex: str                  # "#RRGGBB"
    transparent: bool = False


def normalise_ref(name):
    """How LDrawLoader names a referenced file: forward slashes, `s/` and `48/` mapped into parts/ and p/. Not lower-cased."""
    name = name.strip().replace("\\", "/")
    if name.startswith("s/"):
        return "parts/" + name
    if name.startswith("48/"):
        return "p/" + name
    return name


class LDrawLibrary:
    def __init__(self, zip_path):
        self.path = Path(zip_path)
        if not self.path.exists():
            raise LibraryMissing(f"LDraw library not found at {self.path}: run  python -m backend.core.fetch_ldraw")
        self._lock = threading.Lock()
        self._zip = zipfile.ZipFile(self.path)
        self._names = {}                                  # lower-case path inside ldraw/ -> member name
        for n in self._zip.namelist():
            if n.startswith("ldraw/") and not n.endswith("/"):
                self._names[n[6:].lower()] = n
        self._colours = None

    # ---- raw files
    def has(self, relative):
        return relative.lower() in self._names

    def read(self, relative):
        """Text of a library file by path relative to ldraw/ (e.g. 'parts/3001.dat'), or None."""
        member = self._names.get(relative.lower())
        if member is None:
            return None
        with self._lock:
            return self._zip.read(member).decode("utf-8", errors="replace")

    def find(self, reference):
        """Library path of a referenced file name as written in a .dat/.ldr line, or None."""
        ref = normalise_ref(reference).lower()
        if self.has(ref):
            return ref
        for prefix in _SEARCH:
            if self.has(prefix + ref):
                return prefix + ref
        return None

    def part_path(self, ldraw_id):
        """'parts/3001.dat' for an LDraw part number, or None."""
        ldraw_id = ldraw_id.lower().removesuffix(".dat")
        path = f"parts/{ldraw_id}.dat"
        return path if self.has(path) else None

    def part_description(self, ldraw_id):
        path = self.part_path(ldraw_id)
        text = self.read(path) if path else None
        if not text:
            return None
        first = text.splitlines()[0].strip() if text else ""
        return first[2:].strip() if first.startswith("0 ") else None

    # ---- colours
    def colours(self):
        """{code: LDColour} from LDConfig.ldr."""
        if self._colours is None:
            out = {}
            text = self.read("LDConfig.ldr") or ""
            for line in text.splitlines():
                if "!COLOUR" not in line:
                    continue
                m = re.search(r"!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+(#[0-9A-Fa-f]{6})", line)
                if m:
                    out[int(m.group(2))] = LDColour(int(m.group(2)), m.group(1), m.group(3).upper(), "ALPHA" in line)
            self._colours = out
        return self._colours

    # ---- packing
    def pack(self, model_text, drop_conditional=False):
        """One .mpd: the model (an .ldr, or an .mpd whose embedded files are kept) followed by the library files it needs."""
        return pack_mpd(model_text, self, drop_conditional)


_TYPE1 = re.compile(r"^\s*1\s+(\S+)\s+(?:\S+\s+){12}(.+?)\s*$")


def _referenced(text):
    """File names referenced by the type-1 lines of an LDraw file."""
    for line in text.splitlines():
        if line[:1] in ("1", " ") or line.lstrip()[:1] == "1":
            m = _TYPE1.match(line)
            if m:
                yield m.group(2)


def _mpd_files(text):
    """{lower-case name: (name, text)} for the embedded `0 FILE` sections of an MPD, and the name of the main (first) one."""
    files, main, name, buf = {}, None, None, []
    for line in text.replace(chr(13) + chr(10), chr(10)).split(chr(10)):
        if line.startswith("0 FILE "):
            if name is not None:
                files[name.lower()] = (name, chr(10).join(buf))
            name, buf = line[7:].strip(), []
            main = main or name
        elif name is not None:
            buf.append(line)
    if name is not None:
        files[name.lower()] = (name, chr(10).join(buf))
    return files, main


def _strip_conditional(text):
    return "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("5 "))


def pack_mpd(model_text, library, drop_conditional=False):
    """Pack `model_text` (+ the library files it uses) into one .mpd. `pack_mpd.last_missing` lists references that could not be
    found in the library (parts newer than the downloaded library, or typos)."""
    model_text = model_text.replace("\r\n", "\n")
    embedded, main = _mpd_files(model_text)
    if main is None:                                          # a plain .ldr: it becomes the main file
        main, embedded = "model.ldr", {"model.ldr": ("model.ldr", model_text)}
    files = dict(embedded)                                    # key (as LDrawLoader looks it up) -> (name, text)
    main_key = main.lower()
    missing, seen, queue = [], set(), [main_key]
    while queue:
        key = queue.pop()
        if key in seen:
            continue
        seen.add(key)
        for ref in _referenced(files[key][1]):
            nk = normalise_ref(ref).lower()
            if nk not in files:
                path = library.find(ref)
                if path is None:
                    missing.append(ref)
                    continue
                files[nk] = (nk, library.read(path))
            queue.append(nk)
    out = []
    for key in [main_key] + [k for k in files if k != main_key and k in seen]:
        name, text = files[key]
        if drop_conditional:
            text = _strip_conditional(text)
        out.append(f"0 FILE {name}\n{text}\n")
    pack_mpd.last_missing = missing
    return "".join(out)


pack_mpd.last_missing = []


# ---------------------------------------------------------------- geometry
def _matrix(tokens):
    x, y, z, a, b, c, d, e, f, g, h, i = (float(t) for t in tokens)
    m = np.array([[a, b, c, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]], float)
    return m


def part_points(library, ldraw_id, _cache={}):
    """All vertices (N x 3, LDU) of a part's SOLID geometry (triangles and quads), ignoring studs (files whose name starts with 'stud')."""
    key = ldraw_id.lower()
    if key in _cache:
        return _cache[key]
    path = library.part_path(key)
    if path is None:
        return None
    pts = []

    def walk(rel, m, depth, in_stud):
        text = library.read(rel)
        if text is None or depth > 12:
            return
        for line in text.splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == "1" and len(t) >= 15:
                ref = " ".join(t[14:])
                sub = library.find(ref)
                if sub is None:
                    continue
                stud = in_stud or Path(sub).name.lower().startswith("stud")
                walk(sub, m @ _matrix(t[2:14]), depth + 1, stud)
            elif t[0] in ("3", "4") and not in_stud:
                n = 3 if t[0] == "3" else 4
                if len(t) >= 2 + 3 * n:
                    for k in range(n):
                        v = np.array([float(t[2 + 3 * k]), float(t[3 + 3 * k]), float(t[4 + 3 * k]), 1.0])
                        pts.append((m @ v)[:3])

    walk(path, np.eye(4), 0, False)
    _cache[key] = np.array(pts) if pts else None
    return _cache[key]


def part_bounds(library, ldraw_id):
    """Axis-aligned bounds (min xyz, max xyz) in LDU of a part's SOLID geometry, ignoring studs. Used to check footprint and height
    of the dimension table."""
    arr = part_points(library, ldraw_id)
    return None if arr is None else (arr.min(axis=0), arr.max(axis=0))


_library = None


def get_library(zip_path=None):
    """Process-wide library (None path: the configured one). Raises LibraryMissing if it was not downloaded."""
    global _library
    if _library is None:
        if zip_path is None:
            from backend import config
            zip_path = config.LDRAW_DIR / DEFAULT_ZIP
        _library = LDrawLibrary(zip_path)
    return _library
