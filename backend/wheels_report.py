"""List the wheel, tyre and wheel-holder parts in our LDraw library and say which of them Rebrickable's part data knows.

    python -m backend.wheels_report [--out docs/wheels-in-ldraw]        writes <out>.md (summary) and <out>.csv (every part)

LDraw side: every parts/*.dat of the library zip (BUILD/LDRAW_DIR) whose description names a wheel, tyre or wheel holder. Rebrickable side:
parts.csv (is the part number there, under which category) and the committed role table (backend/data/part_roles.json). A part counts as
"in Rebrickable" when the same part number is in parts.csv, or when one of its Rebrickable print / pattern / mould children's parent is.
"""

import argparse
import csv
import re
import zipfile
from pathlib import Path

import pandas as pd

from backend import config
from backend.core.ldraw_lib import DEFAULT_ZIP
from backend.core.partroles import role_of

KEY_PARTS = ["4600", "4624", "3641", "6014", "2926", "3139", "4488", "11002", "30027", "4085", "2412", "3176", "6157", "4489", "30028", "2815", "3482", "6015", "4083"]
KINDS = [("tyre", re.compile(r"\b(tyre|tire)\b", re.I)),
         ("wheel holder", re.compile(r"wheels? holders?|wheel pins?|wheel holder|\bwheel\b.*\bholder\b", re.I)),
         ("wheel", re.compile(r"\bwheels?\b", re.I))]


def classify(desc):
    for kind, rx in KINDS:
        if rx.search(desc):
            return kind
    return None


def ldraw_parts(zip_path):
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not (name.startswith("ldraw/parts/") and name.endswith(".dat")) or name.count("/") != 2:
                continue
            with z.open(name) as f:
                first = f.readline().decode("utf-8", errors="replace").strip()
            desc = first[2:].strip() if first.startswith("0") else first
            part = name.rsplit("/", 1)[-1][:-4].lower()
            out[part] = desc
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/wheels-in-ldraw")
    args = ap.parse_args(argv)
    ld = ldraw_parts(config.LDRAW_DIR / DEFAULT_ZIP)
    rb = pd.read_csv(config.DATA_DIR / "parts.csv", dtype=str, keep_default_na=False)
    rb_names = {p.lower(): n for p, n in zip(rb.part_num, rb.name)}
    rb_cat = {p.lower(): c for p, c in zip(rb.part_num, rb.part_cat_id)}
    cats = dict(pd.read_csv(config.DATA_DIR / "part_categories.csv", dtype=str, keep_default_na=False)[["id", "name"]].values)
    rows = []
    for part, desc in sorted(ld.items()):
        if desc.startswith("~Moved"):
            continue
        kind = classify(desc)
        if not kind:
            continue
        rows.append({"part": part, "kind": kind, "ldraw_description": desc, "in_rebrickable": part in rb_names, "rebrickable_name": rb_names.get(part, ""),
                     "rebrickable_category": cats.get(rb_cat.get(part, ""), ""), "app_role": role_of(part) if part in rb_names else ""})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    lines = ["# Wheels, tyres and wheel holders in our LDraw library", "",
             f"Library: `{config.LDRAW_DIR / DEFAULT_ZIP}` ({len(ld)} part files). A part is listed when its LDraw description names a wheel, a tyre or a wheel holder",
             "(`~Moved to ...` stubs left out). \"In Rebrickable\" = the same part number is in `parts.csv`. Every part: `wheels-in-ldraw.csv`.", "",
             "| kind | in LDraw | also in Rebrickable's parts.csv | only in LDraw |", "|---|---|---|---|"]
    for kind in ("wheel", "tyre", "wheel holder"):
        r = by.get(kind, [])
        lines.append(f"| {kind} | {len(r)} | {sum(x['in_rebrickable'] for x in r)} | {sum(not x['in_rebrickable'] for x in r)} |")
    lines += ["", "## The parts you named and other common small-vehicle ones", "", "| part | kind | LDraw description | in Rebrickable | Rebrickable name | Rebrickable category |", "|---|---|---|---|---|---|"]
    for part in KEY_PARTS:
        d = ld.get(part)
        r = next((x for x in rows if x["part"] == part), None)
        if d is None:
            lines.append(f"| {part} | - | not in the LDraw library | {'yes' if part in rb_names else 'no'} | {rb_names.get(part, '')} | {cats.get(rb_cat.get(part, ''), '')} |")
        else:
            lines.append(f"| {part} | {r['kind'] if r else '(not matched by name)'} | {d} | {'yes' if part in rb_names else 'no'} | {rb_names.get(part, '')} | {cats.get(rb_cat.get(part, ''), '')} |")
    lines += ["", "## Rebrickable parts with a wheel role that have no LDraw file", ""]
    miss = [(p, n) for p, n in rb_names.items() if role_of(p) in ("wheel", "tyre", "wheel_holder", "wheelbase") and p not in ld]
    lines.append(f"{len(miss)} of {sum(1 for p in rb_names if role_of(p) in ('wheel', 'tyre', 'wheel_holder', 'wheelbase'))} Rebrickable wheel / tyre / holder / wheel-base parts have no LDraw file under the same number (mostly "
                 "complete assemblies such as `…c01`, which LDraw models as separate parts).")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
