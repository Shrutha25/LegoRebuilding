"""Fit-to-pile calculation (script only, no app integration): can a pile build each library design?

    experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/fit_to_pile.py
For each one-piece design in library/ (bowl and mug excluded) and each pile (real_pile + the 3 sampled inventories) it finds the largest
number of design bricks the pile can build, with any colour, using these substitutions of a design brick:
  exact      the same brick
  plates     3 stacked plates of the same footprint (the 8 BrickGPT sizes; none of our piles holds plates, so this never fires yet)
  split      the same footprint tiled in one layer by smaller bricks: 1xn by 1xk bricks; 2xn by 2xk bricks, or by two rows of 1xk bricks
             (so 2x4 = two 2x2, 2x6 = three 2x2 or 2x4 + 2x2, 2x2 = two 1x2, ...)
It is an exact integer program (scipy), not a greedy guess. Output: docs/brickgpt/fit-to-pile.md and .json.
Not checked: that a substituted design is still one piece, and that real_pile colours or the 1x3 / 2x3 / modified bricks of the real
pile are used (only the 8 BrickGPT sizes count)."""
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory import PART_DIMS, dims_part, load_fixtures, type_totals  # noqa: E402

ROOT = HERE.parents[1]
PLATE = {"3024": (1, 1), "3023": (1, 2), "3710": (1, 4), "3666": (1, 6), "3460": (1, 8), "3022": (2, 2), "3020": (2, 4), "3795": (2, 6)}
SIZE_PLATE = {v: k for k, v in PLATE.items()}
PARTS = list(PART_DIMS) + list(PLATE)
EXCLUDE = {"bowl", "mug"}
PILES = ["real_pile", "small", "medium", "large"]


def tilings(n, sizes):
    """All multisets of `sizes` that add up to n."""
    out = []

    def go(left, start, cur):
        if left == 0:
            out.append(tuple(cur))
            return
        for i in range(start, len(sizes)):
            if sizes[i] <= left:
                go(left - sizes[i], i, cur + [sizes[i]])

    go(n, 0, [])
    return out


def recipes(part):
    """Every way to build one design brick of this part from pile parts: a list of Counter({pile part: count})."""
    a, b = PART_DIMS[part]
    out = [Counter({part: 1}), Counter({SIZE_PLATE[(a, b)]: 3})]
    row = lambda k: dims_part(1, k)
    if a == 1:
        for t in tilings(b, [1, 2, 4, 6, 8]):
            if t != (b,):
                out.append(Counter(row(k) for k in t))
    else:
        for t in tilings(b, [2, 4, 6]):                               # 2 x n by 2 x k bricks
            if t != (b,):
                out.append(Counter(dims_part(2, k) for k in t))
        rows = [Counter(row(k) for k in t) for t in tilings(b, [1, 2, 4, 6, 8])]
        for r1, r2 in itertools.combinations_with_replacement(range(len(rows)), 2):   # 2 x n as two rows of 1 x k bricks
            out.append(rows[r1] + rows[r2])
    seen, uniq = set(), []
    for c in out:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


RECIPES = {p: recipes(p) for p in PART_DIMS}


def fit(need: Counter, stock: Counter) -> int:
    """Largest number of design bricks buildable. need/stock: {part: count}, colours already ignored."""
    cols = [(d, r) for d in need for r in RECIPES[d]]
    if not cols:
        return 0
    rows, ub = [], []
    for d in need:                                                    # no more bricks of a part than the design has
        rows.append([1.0 if cd == d else 0.0 for cd, _ in cols])
        ub.append(need[d])
    for p in PARTS:                                                   # no more pile bricks of a part than the pile holds
        rows.append([float(r.get(p, 0)) for _, r in cols])
        ub.append(stock.get(p, 0))
    res = milp(c=-np.ones(len(cols)), constraints=LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
               integrality=np.ones(len(cols)), bounds=Bounds(0, np.inf))
    return int(round(-res.fun)) if res.success else 0


def selfcheck():
    """A few known answers, so a broken recipe or constraint cannot go unnoticed."""
    assert fit(Counter({"3001": 1}), Counter({"3003": 2})) == 1            # 2x4 = two 2x2
    assert fit(Counter({"2456": 1}), Counter({"3003": 3})) == 1            # 2x6 = three 2x2
    assert fit(Counter({"2456": 1}), Counter({"3001": 1, "3003": 1})) == 1  # 2x6 = 2x4 + 2x2
    assert fit(Counter({"3001": 1}), Counter({"3020": 3})) == 1            # brick = 3 plates
    assert fit(Counter({"3001": 1}), Counter({"3003": 1})) == 0
    assert fit(Counter({"3001": 2}), Counter({"3003": 3})) == 1            # 3 x 2x2 build only one 2x4
    assert fit(Counter({"3003": 1, "3001": 1}), Counter({"3001": 1})) == 1


def main():
    selfcheck()
    piles = load_fixtures()
    stock = {p: Counter(type_totals(piles[p])) for p in PILES}
    designs = []
    for f in sorted((HERE / "library").glob("*.json")):
        rec = json.loads(f.read_text())
        if not rec["brick_list"] or not rec["single_piece"] or rec["category"] in EXCLUDE:
            continue
        need = Counter(dims_part(b["h"], b["w"]) for b in rec["brick_list"])
        designs.append({"id": rec["id"], "bricks": len(rec["brick_list"]), "need": need})
    designs.sort(key=lambda d: (d["bricks"], d["id"]))
    rows = []
    for d in designs:
        row = {"id": d["id"], "bricks": d["bricks"]}
        for p in PILES:
            cov = fit(d["need"], stock[p])
            row[p] = {"covered": cov, "missing": d["bricks"] - cov}
        rows.append(row)

    pile_bricks = {p: sum(stock[p].values()) for p in PILES}
    out = ["# Fit to pile", "", f"{len(rows)} one-piece designs (bowl and mug excluded) against four piles, any colour. "
           "Cell = bricks the pile covers / bricks missing. Pile sizes (bricks of the 8 BrickGPT sizes): " +
           ", ".join(f"{p} {pile_bricks[p]}" for p in PILES) + ".", "",
           "| design | bricks needed | " + " | ".join(PILES) + " |", "|---|---:|" + "---:|" * len(PILES)]
    for r in rows:
        out.append(f"| {r['id']} | {r['bricks']} | " + " | ".join(f"{r[p]['covered']} / {r[p]['missing']}" for p in PILES) + " |")
    out += ["", "## How many designs can each pile build?", "", "| pile | buildable (0 missing) | nearly (1-5 missing) | within 10 |", "|---|---:|---:|---:|"]
    summary = {}
    for p in PILES:
        miss = [r[p]["missing"] for r in rows]
        summary[p] = {"buildable": sum(m == 0 for m in miss), "nearly": sum(1 <= m <= 5 for m in miss), "within_10": sum(m <= 10 for m in miss)}
        out.append(f"| {p} | {summary[p]['buildable']} | {summary[p]['nearly']} | {summary[p]['within_10']} |")
    sizes = sorted(r["bricks"] for r in rows)
    out += ["", f"Design sizes (bricks): min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}; "
            f"{sum(s < 40 for s in sizes)} under 40, {sum(s <= 60 for s in sizes)} at 60 or fewer, {sum(s <= 120 for s in sizes)} at 120 or fewer."]
    (ROOT / "docs" / "brickgpt" / "fit-to-pile.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (ROOT / "docs" / "brickgpt" / "fit-to-pile.json").write_text(json.dumps({"summary": summary, "pile_bricks": pile_bricks, "rows": rows}, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
