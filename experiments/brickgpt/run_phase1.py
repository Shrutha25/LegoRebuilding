"""Phase 1 (reduced scope): 7 strict runs + 2 descriptive-prompt runs, seed 42, plus shopping-mode maths on the Phase 0 outputs.

    experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/run_phase1.py
Writes experiments/brickgpt/gallery/<id>.ldr + <id>.meta.json (gitignored; packed for the viewer by pack_gallery.py) and
docs/brickgpt/phase1/results.json + shopping.json. "connected" is BrickGPT's connectivity check (no Gurobi), not a physics check.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brickgpt.data import Brick, BrickStructure
from brickgpt.models import BrickGPTConfig

from inventory import (assign_colours, inv_total, load_fixtures, make_steps, missing_bricks, part_of, read_ldconfig,
                       step_meta, to_ldr)
from inventory_sampler import InventoryBrickGPT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GALLERY = HERE / "gallery"
DOCS = ROOT / "docs" / "brickgpt" / "phase1"
SEED = 42
STRICT_REJECTIONS, STRICT_REGENERATIONS = 100, 10   # bounded so a stuck strict run cannot spin for minutes (upstream: 500 / 100)

D_CHAIR = "a simple chair with four legs, a flat seat and a tall backrest"
D_BENCH = "a long bench with a flat seat and four legs"
RUNS = [  # (id, prompt, inventory name or None for unconstrained)
    ("chair-strict-medium", "a chair", "medium"), ("chair-strict-large", "a chair", "large"),
    ("bench-strict-medium", "a bench", "medium"), ("bench-strict-large", "a bench", "large"),
    ("table-strict-medium", "a table", "medium"), ("table-strict-large", "a table", "large"),
    ("bench-strict-small", "a bench", "small"),
    ("desc-chair-free", D_CHAIR, None), ("desc-bench-free", D_BENCH, None),
]


def shopping(fx):
    out = {}
    for txt in sorted((ROOT / "docs" / "brickgpt" / "phase0").glob("*.txt")):
        bricks = [Brick.from_txt(l) for l in txt.read_text().splitlines() if l.strip()]
        out[txt.stem] = {"bricks": len(bricks), **{name: missing_bricks(bricks, inv) for name, inv in fx.items()}}
        for name, inv in fx.items():
            m = out[txt.stem][name]
            out[txt.stem][name] = {"missing_total": sum(m.values()), "by_type": m, "inventory_bricks": inv_total(inv)}
    return out


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    GALLERY.mkdir(parents=True, exist_ok=True)
    fx = load_fixtures()
    ld = read_ldconfig()
    (DOCS / "shopping.json").write_text(json.dumps(shopping(fx), indent=1))
    print("shopping.json written", flush=True)

    t0 = time.time()
    model = InventoryBrickGPT(BrickGPTConfig(use_gurobi=False))
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    results = []
    for rid, prompt, inv_name in RUNS:
        inv = fx[inv_name] if inv_name else None
        model.set_inventory(inv)
        model.max_brick_rejections, model.max_regenerations = (STRICT_REJECTIONS, STRICT_REGENERATIONS) if inv else (500, 100)
        transformers.set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        out = model(prompt)
        secs = time.time() - t
        bricks = out["bricks"].bricks
        colours = assign_colours(bricks, inv) if inv else ["Medium Lime"] * len(bricks)
        steps = make_steps(bricks)
        (GALLERY / f"{rid}.ldr").write_text(to_ldr(bricks, colours, steps, ld))
        types = Counter(part_of(b) for b in bricks)
        rec = {
            "id": rid, "prompt": prompt, "mode": "strict" if inv else "unconstrained", "inventory": inv_name,
            "inventory_bricks": inv_total(inv) if inv else None, "bricks": len(bricks), "steps": len(steps),
            "end_reason": out["end_reason"] if inv else "finished" if bricks else "empty",
            "connected": bool(out["bricks"].is_connected()) if bricks else False,
            "seconds": round(secs, 1), "rollbacks": out["n_regenerations"], "rejections": out["rejection_reasons"].total(),
            "rejection_reasons": dict(out["rejection_reasons"]), "types": dict(sorted(types.items())),
            "colours": dict(Counter(colours)),
            "over_stock": [k for k, n in Counter(zip(map(part_of, bricks), colours)).items() if inv and n > inv.get(k, 0)],
        }
        meta = {"kind": "ai", "title": f"{prompt} ({rec['mode']}{', ' + inv_name if inv_name else ''})", "description": "BrickGPT, seed 42",
                "attribution": None, "repairs": [], "warnings": [], "id": f"brickgpt-{rid}", "steps": step_meta(bricks, colours, steps, ld),
                "step_count": len(steps), "part_count": len(bricks), "missing_files": [], "run": rec}
        (GALLERY / f"{rid}.meta.json").write_text(json.dumps(meta, indent=1))
        results.append(rec)
        print(json.dumps(rec), flush=True)
        (DOCS / "results.json").write_text(json.dumps(results, indent=2))   # after every run, so a crash loses nothing


if __name__ == "__main__":
    main()
