"""Download the LDraw Official Model Repository (OMR) models into BUILD_CACHE_DIR/omr/ so they can be indexed. Run once:

    python -m backend.core.fetch_omr [--max-parts 1000] [--workers 6] [--limit N]

The list of sets comes from https://library.ldraw.org/omr/sets (25 per page, ?page=N); each model is <OMR_BASE_URL>/<set number>.mpd (the file
omr.fetch_set_model already uses, cached under the same name). Sets with more than --max-parts parts (Rebrickable's count) are skipped: their
files are huge and the app only looks for builds of at most 80 parts, which large sets contain as sub-models too, so use a high limit to
include them. Files already present are kept; a model that does not exist (404) is remembered with a .none marker for a week. Nothing here is
served to users: the files only feed backend/core/omr_index.py, and each indexed model keeps its authors and licence for attribution.
"""

import argparse
import asyncio
import re
import time
from pathlib import Path

import httpx
import pandas as pd

from backend import config
from backend.core import omr

SETS_URL = "https://library.ldraw.org/omr/sets"
THUMB = re.compile(r"omr_models/\d+/conversions/([0-9a-zA-Z\-]+)-thumb")


def list_sets(client, max_pages=300, pause=0.2):
    """Every set number in the OMR, in the order of the site's list."""
    out = []
    for page in range(1, max_pages + 1):
        html = client.get(SETS_URL, params={"page": page}).text
        nums = THUMB.findall(html)
        if not nums:
            break
        out += nums
        time.sleep(pause)
    return list(dict.fromkeys(out))


def part_counts(data_dir):
    path = Path(data_dir) / "sets.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(zip(df.set_num, pd.to_numeric(df.num_parts, errors="coerce")))


async def fetch_all(sets, folder, base_url, workers=6, log=print):
    folder.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(workers)
    stats = {"downloaded": 0, "kept": 0, "missing": 0, "failed": 0}

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        async def one(num):
            path, marker = folder / f"{num}.mpd", folder / f"{num}.none"
            if path.exists():
                stats["kept"] += 1
                return
            if marker.exists() and time.time() - marker.stat().st_mtime < omr.NEGATIVE_TTL:
                stats["missing"] += 1
                return
            async with sem:
                try:
                    r = await client.get(f"{base_url.rstrip('/')}/{num}.mpd")
                except Exception as e:                                        # noqa: BLE001
                    stats["failed"] += 1
                    log(f"  {num}: {type(e).__name__}")
                    return
            if r.status_code == 404:
                marker.write_text("", encoding="utf-8")
                stats["missing"] += 1
            elif r.status_code == 200 and len(r.content) <= omr.MAX_BYTES:
                path.write_text(r.content.decode("utf-8", errors="replace").replace("\r\n", "\n"), encoding="utf-8")
                stats["downloaded"] += 1
            else:
                stats["failed"] += 1
                log(f"  {num}: HTTP {r.status_code}, {len(r.content)} bytes")
            done = sum(stats.values())
            if done % 100 == 0:
                log(f"  {done}/{len(sets)}  {stats}")
        await asyncio.gather(*[one(n) for n in sets])
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parts", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sets (a test run)")
    args = ap.parse_args(argv)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        sets = list_sets(client)
    counts = part_counts(config.DATA_DIR)
    wanted = [s for s in sets if (counts.get(s) is None or counts[s] != counts[s] or counts[s] <= args.max_parts)]
    if args.limit:
        wanted = wanted[:args.limit]
    print(f"{len(sets)} OMR sets; fetching {len(wanted)} with at most {args.max_parts} parts into {config.BUILD_CACHE_DIR / 'omr'}")
    t = time.time()
    stats = asyncio.run(fetch_all(wanted, config.BUILD_CACHE_DIR / "omr", config.OMR_BASE_URL, args.workers))
    print(f"done in {time.time() - t:.0f}s: {stats}")


if __name__ == "__main__":
    main()
