"""Download the official LDraw parts library (~146 MB, one zip) into LDRAW_DIR. Resumes an interrupted download.

    python -m backend.core.fetch_ldraw            # download if missing
    python -m backend.core.fetch_ldraw --refresh  # download again (a new library release)

The library is licensed CC BY 4.0 (parts) / CCAL 2.0 (older files); see CAlicense.txt inside the zip. The viewer only receives the
few files each model needs (see ldraw_lib.pack_mpd), never the library itself.
"""

import argparse
import sys

import httpx

from backend import config
from backend.core.ldraw_lib import DEFAULT_ZIP, LDrawLibrary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    config.LDRAW_DIR.mkdir(parents=True, exist_ok=True)
    target = config.LDRAW_DIR / DEFAULT_ZIP
    if target.exists() and not args.refresh:
        print(f"{target} already there ({target.stat().st_size / 1e6:.0f} MB); use --refresh to download again")
        return
    part = target.with_suffix(".zip.part")
    if args.refresh and part.exists():
        part.unlink()
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with httpx.stream("GET", config.LDRAW_ZIP_URL, headers=headers, follow_redirects=True, timeout=60) as r:
        if r.status_code not in (200, 206):
            sys.exit(f"download failed: HTTP {r.status_code}")
        if r.status_code == 200:
            have = 0                                        # the server ignored the range: start again
        total = int(r.headers.get("content-length", 0)) + have
        with open(part, "ab" if have else "wb") as f:
            done = have
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r{done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    part.replace(target)
    LDrawLibrary(target)                                    # validates that it opens
    print(f"\nsaved {target}")


if __name__ == "__main__":
    main()
