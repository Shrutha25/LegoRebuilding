"""Official LEGO set models from the LDraw Official Model Repository (OMR): https://library.ldraw.org/omr

A model for set "10001-1" is the file <OMR_BASE_URL>/10001-1.mpd (the same set numbers Rebrickable uses). Each file carries its
author(s) and licence in its header; `licence_ok` accepts only the Creative Commons Attribution family that LDraw's OMR is
published under (CCAL 2.0 / CC BY) and a model without a recognisable licence is NOT used (it is logged, so it can be reviewed).
Attribution (authors, licence, source page) is returned with the model and shown in the viewer.

Downloads are cached on disk (build_cache/omr/), and "no model for this set" is cached too, for a week.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger("lego.omr")

MAX_BYTES = 12 * 1024 * 1024
NEGATIVE_TTL = 7 * 86400
_SAFE = re.compile(r"^[0-9a-z]+(-[0-9]+)?$", re.I)
_OK_LICENCE = re.compile(r"\bCCAL\b|\bCC[ -]BY\b|Creative Commons Attribution", re.I)


@dataclass
class OmrModel:
    set_num: str
    text: str
    authors: list = field(default_factory=list)
    licences: list = field(default_factory=list)
    source_url: str = ""

    @property
    def licence_ok(self):
        return bool(self.licences) and all(_OK_LICENCE.search(line) for line in self.licences)


def parse_header(text):
    """(authors, licence lines) from every `0 Author:` and `0 !LICENSE` line of an MPD (unique, in order)."""
    authors, licences = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if s.startswith("0 Author:"):
            a = s[len("0 Author:"):].strip()
            if a and a not in authors:
                authors.append(a)
        elif s.startswith("0 !LICENSE"):
            lic = s[len("0 !LICENSE"):].strip()
            if lic and lic not in licences:
                licences.append(lic)
    return authors, licences


def valid_set_num(set_num):
    return bool(_SAFE.match(set_num or ""))


async def fetch_set_model(set_num, cache_dir, base_url, client=None):
    """The OMR model for a set, or None when there is none. Never raises for network trouble (returns None and logs)."""
    if not valid_set_num(set_num):
        return None
    folder = Path(cache_dir) / "omr"
    folder.mkdir(parents=True, exist_ok=True)
    path, marker = folder / f"{set_num}.mpd", folder / f"{set_num}.none"
    source = f"https://library.ldraw.org/omr/sets?search={set_num.split('-')[0]}"
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
    elif marker.exists() and time.time() - marker.stat().st_mtime < NEGATIVE_TTL:
        return None
    else:
        try:
            own = client is None
            client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
            try:
                r = await client.get(f"{base_url.rstrip('/')}/{set_num}.mpd")
            finally:
                if own:
                    await client.aclose()
        except Exception as e:                                          # noqa: BLE001
            log.warning("omr: could not fetch %s (%s)", set_num, e)
            return None
        if r.status_code == 404:
            marker.write_text("", encoding="utf-8")
            return None
        if r.status_code != 200 or len(r.content) > MAX_BYTES or b"0 FILE" not in r.content[:4000] and b"\n1 " not in r.content[:20000]:
            log.warning("omr: unusable answer for %s (HTTP %s, %d bytes)", set_num, r.status_code, len(r.content))
            return None
        text = r.content.decode("utf-8", errors="replace").replace(chr(13) + chr(10), chr(10))          # the same text whether just downloaded or read back
        path.write_text(text, encoding="utf-8")
    authors, licences = parse_header(text)
    model = OmrModel(set_num, text, authors, licences, source)
    if not model.licence_ok:
        log.warning("omr: %s has no recognised open licence (%r): not used, needs a manual look", set_num, licences)
    return model


def cached_status(set_num, cache_dir):
    """Whether a usable (open-licence) OMR model exists, from the disk cache only: True / False / None (not checked yet)."""
    if not valid_set_num(set_num):
        return False
    folder = Path(cache_dir) / "omr"
    path, marker = folder / f"{set_num}.mpd", folder / f"{set_num}.none"
    if path.exists():
        _, licences = parse_header(path.read_text(encoding="utf-8", errors="replace"))
        return bool(licences) and all(_OK_LICENCE.search(line) for line in licences)
    if marker.exists() and time.time() - marker.stat().st_mtime < NEGATIVE_TTL:
        return False
    return None
