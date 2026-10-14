"""
Suggest LEGO builds from a recognized parts list.

Input:  parts_list.csv from recognize_pieces.py  (part_num,name,color,qty)
Data:   Rebrickable CSV dumps (downloaded automatically into ./rb_data on first run)
Output: ranked list of official sets you can (mostly) build, with missing parts,
        plus llm_prompt.txt for generating freeform "build your own" ideas.

Usage:
  pip install pandas requests
  python suggest_builds.py parts_list.csv
  python suggest_builds.py parts_list.csv --max-parts 100 --top 15 --match-color

Per set:
  coverage = pieces you have toward the set / total pieces in the set
  (spare parts excluded; print/pattern/mold variants treated as the same part)
Only sets you can build 100% are listed (--min-coverage 1.0). Ranked by parts_you_have * coverage,
which at 100% means biggest set first; then fewest missing, then newest.
"""

import argparse
import json

import pandas as pd

from backend.core.match import (build_set_requirements, canonical_map, load_rebrickable,
                                load_user_parts, rank_sets, skipped_theme_ids)


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


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts_list")
    ap.add_argument("--data-dir", default="rb_data")
    ap.add_argument("--refresh", action="store_true", help="re-download Rebrickable data")
    ap.add_argument("--match-color", action="store_true", help="require exact colours")
    ap.add_argument("--max-parts", type=int, default=150)
    ap.add_argument("--min-parts", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rb = load_rebrickable(args.data_dir, args.refresh)
    canon = canonical_map(rb["part_relationships"])
    req = build_set_requirements(rb, canon, args.match_color)
    have = load_user_parts(args.parts_list, canon, args.match_color)
    skip = skipped_theme_ids(rb["themes"])

    results = rank_sets(have, req, rb["sets"], skip, args.max_parts,
                        args.min_parts, args.min_coverage, args.top)

    print(f"You have {sum(have.values())} pieces ({len(have)} unique).\n")
    if not results:
        print("No official set reaches the coverage threshold - "
              "try --min-coverage 0.5, or use the LLM ideas.")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['set_num']:10s} {r['name']} ({r['year']})  "
              f"{r['coverage']:.0%}  have {r['parts_you_have']}/{r['total_parts']}, "
              f"missing {r['missing_count']}")

    with open("suggestions.json", "w") as f:
        json.dump(results, f, indent=2)
    write_llm_prompt(args.parts_list)
    print("\nSaved suggestions.json and llm_prompt.txt (send to a vision/LLM for freeform ideas)")


if __name__ == "__main__":
    main()
