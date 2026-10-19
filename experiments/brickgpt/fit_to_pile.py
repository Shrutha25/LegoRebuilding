"""Fit-to-pile calculation (script only, no app integration): can a pile build each library design?

    experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/fit_to_pile.py
For each one-piece design in library/ (bowl and mug excluded; standard and small) and each pile (real_pile + the 3 sampled inventories) it
finds the largest number of design bricks the pile can build, with any colour, using these substitutions of a design brick:
  exact      the same brick
  plates     3 stacked plates of the same footprint (none of our piles holds plates, so this never fires yet; the geometry is unchanged)
  split      the same footprint tiled in one layer by smaller bricks: 1xn by 1xk bricks; 2xn by 2xk bricks, or by two rows of 1xk bricks
             (so 2x4 = two 2x2, 2x6 = three 2x2 or 2x4 + 2x2, 2x2 = two 1x2, ...)
It is an exact integer program (scipy), not a greedy guess.

One-piece check: for a design with at most NEARLY missing bricks, the chosen substitutions are laid out in real positions (the bricks of a
split are placed side by side along the brick's long axis, order shuffled) and the result is tested with single_piece() (bricks linked through stud
overlaps). Which design bricks are substituted is not fixed by the integer program, so up to TRIES random assignments are tried; the design
counts as "1pc" if any of them is one piece, "BROKEN" if none is. For a design with missing bricks both versions are tested: the missing bricks
bought (as_bought) and left out (as_built). Output: docs/brickgpt/fit-to-pile.md and .json.
Not checked: real_pile colours, or its 1x3 / 2x3 / modified bricks (only the 8 BrickGPT sizes count)."""
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory import PART_DIMS, dims_part, load_fixtures, single_piece, type_totals  # noqa: E402

ROOT = HERE.parents[1]
PLATE = {"3024": (1, 1), "3023": (1, 2), "3710": (1, 4), "3666": (1, 6), "3460": (1, 8), "3022": (2, 2), "3020": (2, 4), "3795": (2, 6)}
SIZE_PLATE = {v: k for k, v in PLATE.items()}
PARTS = list(PART_DIMS) + list(PLATE)
EXCLUDE = {"bowl", "mug"}
PILES = ["real_pile", "small", "medium", "large"]
NEARLY = 5      # "nearly buildable": at most this many bricks missing
TRIES = 300     # random assignments tried per design and pile for the one-piece check


class Br:
    """A brick in the layout, with the fields single_piece() reads."""
    def __init__(self, h, w, x, y, z):
        self.h, self.w, self.x, self.y, self.z = h, w, x, y, z


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
    """Every way to build one design brick of this part: dicts {kind, tiling, counter}. Counter = pile parts used.
    kind exact/plates keep the geometry; seq1 = 1xk bricks in a row; seq2 = 2xk bricks in a row; rows = two lanes of 1xk bricks."""
    a, b = PART_DIMS[part]
    out = [{"kind": "exact", "tiling": None, "counter": Counter({part: 1})},
           {"kind": "plates", "tiling": None, "counter": Counter({SIZE_PLATE[(a, b)]: 3})}]
    row = lambda k: dims_part(1, k)
    if a == 1:
        for t in tilings(b, [1, 2, 4, 6, 8]):
            if t != (b,):
                out.append({"kind": "seq1", "tiling": t, "counter": Counter(row(k) for k in t)})
    else:
        for t in tilings(b, [2, 4, 6]):
            if t != (b,):
                out.append({"kind": "seq2", "tiling": t, "counter": Counter(dims_part(2, k) for k in t)})
        lanes = tilings(b, [1, 2, 4, 6, 8])
        for t1, t2 in itertools.combinations_with_replacement(lanes, 2):
            out.append({"kind": "rows", "tiling": (t1, t2), "counter": Counter(row(k) for k in t1 + t2)})
    seen, uniq = set(), []
    for r in out:
        key = tuple(sorted(r["counter"].items()))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


RECIPES = {p: recipes(p) for p in PART_DIMS}


def layout(brick, recipe, rng):
    """The bricks that replace `brick` (dict h, w, x, y, z) under `recipe`; the footprint is tiled exactly, in the same layer."""
    h, w, x, y, z = brick["h"], brick["w"], brick["x"], brick["y"], brick["z"]
    if recipe["kind"] in ("exact", "plates"):
        return [Br(h, w, x, y, z)]
    along_x = h > w                                    # the long axis; a square brick is laid out along y
    lanes = [recipe["tiling"]] if recipe["kind"] in ("seq1", "seq2") else list(recipe["tiling"])
    width = 2 if recipe["kind"] == "seq2" else 1
    out = []
    for lane, tiling in enumerate(lanes):
        order = list(tiling)
        rng.shuffle(order)
        off = 0
        for k in order:
            out.append(Br(k, width, x + off, y + lane, z) if along_x else Br(width, k, x + lane, y + off, z))
            off += k
    return out


def solve(need: Counter, stock: Counter):
    """-> (design bricks buildable, {(part, recipe index): how many design bricks of that part are built that way})."""
    cols = [(d, i) for d in need for i in range(len(RECIPES[d]))]
    if not cols:
        return 0, {}
    rows, ub = [], []
    for d in need:                                                    # no more bricks of a part than the design has
        rows.append([1.0 if cd == d else 0.0 for cd, _ in cols])
        ub.append(need[d])
    for p in PARTS:                                                   # no more pile bricks of a part than the pile holds
        rows.append([float(RECIPES[d][i]["counter"].get(p, 0)) for d, i in cols])
        ub.append(stock.get(p, 0))
    res = milp(c=-np.ones(len(cols)), constraints=LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
               integrality=np.ones(len(cols)), bounds=Bounds(0, np.inf))
    if not res.success:
        return 0, {}
    plan = {cols[j]: int(round(v)) for j, v in enumerate(res.x) if round(v) > 0}
    return sum(plan.values()), plan


def fit(need: Counter, stock: Counter) -> int:
    return solve(need, stock)[0]


def substitution_check(bricks, stock: Counter, tries=TRIES, seed=0) -> dict:
    """Lay out the integer program's substitutions for `bricks` (list of dicts h, w, x, y, z) and test single_piece().
    -> {"as_bought": any try is one piece with the missing bricks kept, "as_built": any try is one piece without them, "tries": n}"""
    need = Counter(dims_part(b["h"], b["w"]) for b in bricks)
    covered, plan = solve(need, stock)
    by_part = {}
    for i, b in enumerate(bricks):
        by_part.setdefault(dims_part(b["h"], b["w"]), []).append(i)
    rng = random.Random(seed)
    ok_bought = ok_built = False
    for n in range(1, tries + 1):
        built, bought = [], []
        for part, idx in by_part.items():
            idx = idx[:]
            rng.shuffle(idx)
            jobs = [RECIPES[part][i] for (d, i), cnt in plan.items() if d == part for _ in range(cnt)]
            for k, bi in enumerate(idx):
                if k < len(jobs):
                    placed = layout(bricks[bi], jobs[k], rng)
                    built += placed
                    bought += placed
                else:                                                   # not covered by the pile: missing
                    bought.append(Br(bricks[bi]["h"], bricks[bi]["w"], bricks[bi]["x"], bricks[bi]["y"], bricks[bi]["z"]))
        ok_bought = ok_bought or single_piece(bought)["single_piece"]
        ok_built = ok_built or (bool(built) and single_piece(built)["single_piece"])
        if ok_bought and ok_built:
            break
        if covered == len(bricks) and ok_bought:                       # nothing missing: both versions are the same
            ok_built = True
            break
    return {"as_bought": ok_bought, "as_built": ok_built, "tries": n}


def selfcheck():
    """Known answers, so a broken recipe, constraint or layout cannot go unnoticed."""
    assert fit(Counter({"3001": 1}), Counter({"3003": 2})) == 1            # 2x4 = two 2x2
    assert fit(Counter({"2456": 1}), Counter({"3003": 3})) == 1            # 2x6 = three 2x2
    assert fit(Counter({"2456": 1}), Counter({"3001": 1, "3003": 1})) == 1  # 2x6 = 2x4 + 2x2
    assert fit(Counter({"3001": 1}), Counter({"3020": 3})) == 1            # brick = 3 plates
    assert fit(Counter({"3001": 1}), Counter({"3003": 1})) == 0
    assert fit(Counter({"3001": 2}), Counter({"3003": 3})) == 1            # 3 x 2x2 build only one 2x4
    assert fit(Counter({"3003": 1, "3001": 1}), Counter({"3001": 1})) == 1
    rng = random.Random(1)
    for part in PART_DIMS:                                                 # every recipe tiles the footprint exactly: same area, no overlap, same layer
        a, b = PART_DIMS[part]
        for horizontal in (True, False):
            brick = {"h": b, "w": a, "x": 3, "y": 4, "z": 2} if horizontal else {"h": a, "w": b, "x": 3, "y": 4, "z": 2}
            for r in RECIPES[part]:
                cells = [(q.x + i, q.y + j, q.z) for q in layout(brick, r, rng) for i in range(q.h) for j in range(q.w)]
                want = {(brick["x"] + i, brick["y"] + j, 2) for i in range(brick["h"]) for j in range(brick["w"])}
                assert len(cells) == len(set(cells)) and set(cells) == want, (part, r["kind"], r["tiling"])
    two = [{"h": 2, "w": 4, "x": 0, "y": 0, "z": 0}, {"h": 2, "w": 4, "x": 0, "y": 0, "z": 1}]   # two stacked 2x4
    assert substitution_check(two, Counter({"3003": 4}))["as_bought"] is False   # 2x2 only: two separate columns, never one piece
    assert substitution_check(two, Counter({"3001": 1, "3003": 2}))["as_bought"] is True   # one exact + one split: the exact brick joins them


def load_designs():
    designs = []
    for f in sorted((HERE / "library").glob("*.json")):
        rec = json.loads(f.read_text())
        if not rec["brick_list"] or not rec["single_piece"] or rec["category"] in EXCLUDE:
            continue
        designs.append({"id": rec["id"], "tag": rec.get("tag", "standard"), "bricks": len(rec["brick_list"]), "list": rec["brick_list"],
                        "need": Counter(dims_part(b["h"], b["w"]) for b in rec["brick_list"])})
    designs.sort(key=lambda d: (d["bricks"], d["id"]))
    return designs


def mark(cell):
    """Table suffix for a design with at most NEARLY bricks missing: is it still one piece after the substitutions?"""
    if cell["missing"] > NEARLY:
        return ""
    c = cell["one_piece"]
    if c["as_built"] and c["as_bought"]:
        return " (1pc)"
    return " (1pc once bought)" if c["as_bought"] else " (BROKEN)"


def main():
    selfcheck()
    piles = load_fixtures()
    stock = {p: Counter(type_totals(piles[p])) for p in PILES}
    rows = []
    for d in load_designs():
        row = {"id": d["id"], "tag": d["tag"], "bricks": d["bricks"]}
        for p in PILES:
            cov = fit(d["need"], stock[p])
            row[p] = {"covered": cov, "missing": d["bricks"] - cov}
            if row[p]["missing"] <= NEARLY:
                row[p]["one_piece"] = substitution_check(d["list"], stock[p])
        rows.append(row)

    pile_bricks = {p: sum(stock[p].values()) for p in PILES}
    out = ["# Fit to pile", "", f"{len(rows)} one-piece designs (bowl and mug excluded; {sum(r['tag'] == 'small' for r in rows)} of them from the small-design pilot) against four "
           "piles, any colour. Cell = bricks the pile covers / bricks missing. For a design with 5 or fewer bricks missing the cell also says whether it is still one "
           "piece after the substitutions: `1pc`; `1pc once bought` (one piece only if the missing bricks are bought); `BROKEN` (no layout tried was one piece). "
           "Pile sizes (bricks of the 8 BrickGPT sizes): " + ", ".join(f"{p} {pile_bricks[p]}" for p in PILES) + ".", "",
           "| design | tag | bricks needed | " + " | ".join(PILES) + " |", "|---|---|---:|" + "---:|" * len(PILES)]
    for r in rows:
        out.append(f"| {r['id']} | {r['tag']} | {r['bricks']} | " + " | ".join(f"{r[p]['covered']} / {r[p]['missing']}{mark(r[p])}" for p in PILES) + " |")
    out += ["", "## How many designs can each pile build?", "",
            "Buildable = 0 missing, nearly = 1-5 missing. In brackets: how many of them are still one piece after the substitutions (for nearly: once the missing bricks are bought).", "",
            "| pile | designs | buildable | nearly | within 10 missing |", "|---|---|---:|---:|---:|"]
    summary = {}
    for p in PILES:
        for name, sel in (("all", rows), ("standard", [r for r in rows if r["tag"] == "standard"]), ("small", [r for r in rows if r["tag"] == "small"])):
            b = [r for r in sel if r[p]["missing"] == 0]
            n = [r for r in sel if 1 <= r[p]["missing"] <= NEARLY]
            b1 = sum(r[p]["one_piece"]["as_bought"] for r in b)
            n1 = sum(r[p]["one_piece"]["as_bought"] for r in n)
            w10 = sum(r[p]["missing"] <= 10 for r in sel)
            summary[f"{p}/{name}"] = {"designs": len(sel), "buildable": len(b), "buildable_one_piece": b1, "nearly": len(n), "nearly_one_piece_if_bought": n1, "within_10": w10}
            out.append(f"| {p} | {name} ({len(sel)}) | {len(b)} ({b1}) | {len(n)} ({n1}) | {w10} |")
    sizes = sorted(r["bricks"] for r in rows)
    out += ["", f"Design sizes (bricks): min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}; "
            f"{sum(s <= 40 for s in sizes)} at 40 or fewer, {sum(s <= 60 for s in sizes)} at 60 or fewer."]
    (ROOT / "docs" / "brickgpt" / "fit-to-pile.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (ROOT / "docs" / "brickgpt" / "fit-to-pile.json").write_text(json.dumps({"summary": summary, "pile_bricks": pile_bricks, "rows": rows}, indent=1), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
