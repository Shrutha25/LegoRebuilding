"""Build backend/data/part_roles.json: what ROLE each Rebrickable part plays in a model (brick, plate, tile, slope, round, wheel ...).

    python -m backend.core.fetch_part_roles [--data-dir rb_data]

Source: Rebrickable's free CSV dumps parts.csv (part_num, name, part_cat_id) and part_categories.csv (id, name), the same files
fetch_data downloads (part_categories.csv is fetched here when missing). Nothing is scraped. A part's role is its category's role,
refined by its name for the few cases where a category mixes roles (cones in "Bricks Round and Cones", arches in "Bricks Special",
complete vehicle bases with wheels in "Transportation - Land"). The generated file is committed; rerun after a Rebrickable update.
"""

import argparse
import gzip
import io
import json
import re
from pathlib import Path

import pandas as pd
import requests

from backend.core.fetch_data import default_data_dir

OUT = Path(__file__).resolve().parent.parent / "data" / "part_roles.json"
CAT_URL = "https://cdn.rebrickable.com/media/downloads/part_categories.csv.gz"

# Rebrickable part_cat_id -> role
CATEGORY_ROLE = {
    "11": "brick", "5": "brick_special", "8": "technic", "3": "slope", "6": "wedge", "37": "curved",
    "14": "plate", "9": "plate_special", "49": "wedge_plate", "19": "tile", "15": "tile_special", "67": "tile_round",
    "20": "round", "21": "round_plate", "29": "wheel", "36": "vehicle_part", "35": "vehicle_part",
    "16": "window_door", "23": "panel", "32": "fence", "18": "hinge", "1": "baseplate", "47": "windscreen",
    "13": "minifig", "59": "minifig", "60": "minifig", "61": "minifig", "65": "minifig", "70": "minifig", "71": "minifig",
    "72": "minifig", "73": "minifig", "27": "minifig_accessory", "62": "minifig", "63": "minifig", "64": "minifig",
    "28": "animal", "74": "animal", "75": "animal", "76": "plant", "38": "flag", "58": "sticker",
    "4": "duplo", "57": "duplo", "12": "technic", "25": "technic", "26": "technic", "40": "technic", "46": "technic",
    "51": "technic", "52": "technic", "53": "technic", "54": "technic", "55": "technic", "17": "technic",
}
DEFAULT_ROLE = "other"
IGNORED = {"other", "minifig", "minifig_accessory", "duplo", "sticker", "technic", "animal", "plant", "flag"}   # not used to build models

# name refinements, first match wins: (regex on the name, only for these categories or None, role)
REFINE = [
    (r"\bCone\b", {"20", "21", "5"}, "cone"),
    (r"\bArch\b", {"5", "3", "37"}, "arch"),
    (r"\bInverted\b", {"3"}, "slope_inverted"),
    (r"Wheel.*Complete Assembly|Vehicle Base.*Wheels\b(?!.*Holders?)|Racer Base with .* Wheels|Wheel Holder.*with Wheels", {"36"}, "wheelbase"),
    (r"Wheels? Holders?|Wheel Pins?", {"36", "9", "5"}, "wheel_holder"),
    (r"^Tyre|\bTire\b$", {"29"}, "tyre"),
]


def category_names(data_dir):
    path = Path(data_dir) / "part_categories.csv"
    if not path.exists():
        r = requests.get(CAT_URL, timeout=60)
        r.raise_for_status()
        path.write_bytes(gzip.decompress(r.content))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def role_for(name, cat):
    for pattern, cats, role in REFINE:
        if (cats is None or cat in cats) and re.search(pattern, name):
            return role
    return CATEGORY_ROLE.get(cat, DEFAULT_ROLE)


def build(data_dir):
    cats = category_names(data_dir)
    parts = pd.read_csv(Path(data_dir) / "parts.csv", dtype=str, keep_default_na=False)
    roles = {}
    for num, name, cat in zip(parts.part_num, parts.name, parts.part_cat_id):
        role = role_for(name, cat)
        if role not in IGNORED:
            roles[num] = role                                         # parts in no useful role are simply absent: role_of() says "other"
    return {"source": "Rebrickable parts.csv + part_categories.csv",
            "categories": {c: {"name": n, "role": CATEGORY_ROLE.get(c, DEFAULT_ROLE)} for c, n in zip(cats.id, cats.name)},
            "parts": dict(sorted(roles.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=default_data_dir())
    args = ap.parse_args(argv)
    data = build(args.data_dir)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    roles = pd.Series(list(data["parts"].values())).value_counts()
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    print(roles.head(15).to_string())


if __name__ == "__main__":
    main()
