"""Download the Rebrickable CSV dumps. Run once, separately from the server:

    python -m backend.core.fetch_data [--data-dir rb_data] [--refresh]

The default directory is $RB_DATA_DIR if set (the Docker image sets it to the mounted volume),
otherwise ./rb_data, which is the same directory the server reads.
"""

import argparse
import os

from backend.core.match import RB_FILES, download_file


def default_data_dir():
    return os.environ.get("RB_DATA_DIR") or "rb_data"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=default_data_dir())
    ap.add_argument("--refresh", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)
    for name in RB_FILES:
        download_file(name, args.data_dir, args.refresh)
    print(f"Done: {len(RB_FILES)} files in {args.data_dir}/")


if __name__ == "__main__":
    main()
