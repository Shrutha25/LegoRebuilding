"""Rebrickable data loading and set matching."""

import gzip
import os
import shutil
from collections import defaultdict

import pandas as pd
import requests

RB_URL = "https://cdn.rebrickable.com/media/downloads/{}.csv.gz"
RB_FILES = ["sets", "themes", "colors", "inventories", "inventory_parts", "part_relationships"]

# Themes that are parts packs, not builds
SKIP_THEMES = ["supplemental", "service packs", "bulk bricks", "key chains", "magnets",
               "gear", "books", "collectible minifigures", "promotional"]


# ---------------------------------------------------------------- data
def load_rebrickable(data_dir, refresh=False):
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    for name in RB_FILES:
        path = os.path.join(data_dir, f"{name}.csv")
        if refresh or not os.path.exists(path):
            print(f"Downloading {name}.csv.gz ...")
            r = requests.get(RB_URL.format(name), stream=True, timeout=120)
            r.raise_for_status()
            gz = path + ".gz"
            with open(gz, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            with gzip.open(gz, "rb") as fi, open(path, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(gz)
        out[name] = pd.read_csv(path, dtype=str, keep_default_na=False)
    return out


def canonical_map(rels):
    """child part -> parent part for printed/patterned/mold/alternate variants."""
    keep = rels[rels.rel_type.isin(["P", "T", "M", "A"])]
    m = dict(zip(keep.child_part_num, keep.parent_part_num))

    def canon(p):
        seen = set()
        while p in m and p not in seen:
            seen.add(p)
            p = m[p]
        return p
    return canon


def skipped_theme_ids(themes):
    themes = themes.set_index("id")
    bad = set()
    for tid, row in themes.iterrows():
        chain, cur = [], tid
        while cur and cur in themes.index:   # walk up to the root theme
            chain.append(themes.at[cur, "name"].lower())
            cur = themes.at[cur, "parent_id"]
        if any(s in name for name in chain for s in SKIP_THEMES):
            bad.add(tid)
    return bad


# ---------------------------------------------------------------- matching
def build_set_requirements(rb, canon, match_color):
    """{set_num: {key: qty}} where key is part or (part, color_name)."""
    inv = rb["inventories"].copy()
    inv["version"] = inv.version.astype(int)
    inv = inv.sort_values("version").drop_duplicates("set_num", keep="last")  # latest inventory
    inv_to_set = dict(zip(inv.id, inv.set_num))

    ip = rb["inventory_parts"]
    ip = ip[ip.inventory_id.isin(inv_to_set) & ~ip.is_spare.str.lower().isin(["t", "true", "1"])]
    colors = dict(zip(rb["colors"].id, rb["colors"].name.str.lower()))

    req = defaultdict(lambda: defaultdict(int))
    for inv_id, part, color_id, qty in zip(ip.inventory_id, ip.part_num, ip.color_id, ip.quantity):
        p = canon(part)
        key = (p, colors.get(color_id, "")) if match_color else p
        req[inv_to_set[inv_id]][key] += int(qty)
    return req


def load_user_parts(path, canon, match_color):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    have = defaultdict(int)
    for _, r in df.iterrows():
        p = canon(r.part_num.strip())
        key = (p, r.get("color", "").strip().lower()) if match_color else p
        have[key] += int(r.qty or 1)
    return have


def rank_sets(have, req, sets, skip_themes, max_parts, min_parts, min_coverage, top):
    sets = sets.set_index("set_num")
    results = []
    for set_num, need in req.items():
        if set_num not in sets.index:
            continue                                  # minifig/other inventories
        s = sets.loc[set_num]
        total = sum(need.values())
        if s.theme_id in skip_themes or not (min_parts <= total <= max_parts):
            continue
        got = sum(min(have.get(k, 0), q) for k, q in need.items())
        if got == 0:
            continue
        cov = got / total
        if cov < min_coverage:
            continue
        missing = {k: q - min(have.get(k, 0), q) for k, q in need.items() if have.get(k, 0) < q}
        base = set_num.split("-")[0]
        results.append({
            "set_num": set_num, "name": s["name"], "year": s.year,
            "total_parts": total, "parts_you_have": got,
            "coverage": round(cov, 3), "missing_count": total - got,
            "missing": [{"part": k if isinstance(k, str) else k[0],
                         "color": None if isinstance(k, str) else k[1], "qty": q}
                        for k, q in sorted(missing.items(), key=lambda x: -x[1])],
            "image": s.img_url,
            "rebrickable": f"https://rebrickable.com/sets/{set_num}/",
            "instructions": f"https://www.lego.com/service/buildinginstructions/{base}",
        })
    # Best coverage first; tie-break on fewest missing pieces, then newer sets
    results.sort(key=lambda r: (-r["coverage"], r["missing_count"], -int(r["year"] or 0)))
    return results[:top]


def write_llm_prompt(user_csv, path="llm_prompt.txt"):
    df = pd.read_csv(user_csv, dtype=str, keep_default_na=False)
    lines = [f'- {r.qty} x {r["name"]} ({r.color or "unknown colour"}), part {r.part_num}'
             for _, r in df.iterrows()]
    prompt = (
        "I have exactly these LEGO pieces:\n" + "\n".join(lines) + "\n\n"
        "Suggest 3 small original models I could build using ONLY these pieces "
        "(I don't need to use all of them). For each: a name, a one-line description, "
        "which pieces it uses, and 4-6 short build steps. Don't invent pieces I don't have."
    )
    with open(path, "w") as f:
        f.write(prompt)
