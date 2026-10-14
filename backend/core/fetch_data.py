"""Download the Rebrickable CSV dumps. Run once, separately from the server:

    python -m backend.core.fetch_data [--data-dir rb_data] [--refresh]
"""

import argparse

from backend.core.match import RB_FILES, download_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="rb_data")
    ap.add_argument("--refresh", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    for name in RB_FILES:
        download_file(name, args.data_dir, args.refresh)
    print(f"Done: {len(RB_FILES)} files in {args.data_dir}/")


if __name__ == "__main__":
    main()
