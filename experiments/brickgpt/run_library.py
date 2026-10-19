"""Offline library: every prompt in prompts.json x seeds 42, 43, 44, unconstrained, BrickGPT's original limits, no Gurobi.

    experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/run_library.py
Writes experiments/brickgpt/library/<id>-s<seed>.json (prompt, category, seed, bricks, numbers) and .ldr (gitignored), and appends
library/runs.jsonl after every run, so a crash loses nothing and a rerun skips finished designs.
"connected" is BrickGPT's check (every brick reaches the ground); "single_piece" is ours (one piece through stud overlaps).
Colours are not tuned: one colour for all bricks."""
import json
import sys
import time
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brickgpt.models import BrickGPT, BrickGPTConfig

from inventory import part_of, single_piece

HERE = Path(__file__).resolve().parent
LIB = HERE / "library"
SEEDS = (42, 43, 44)
CAP_S = 600   # PERMANENT wall-clock cap per design: keep it on every batch. Added after bowl-s42 ran ~37 h (GPU full, never finished); a capped design is recorded as timed_out


class Capped(BrickGPT):
    """Upstream BrickGPT plus a deadline, checked before every brick. Nothing else changes (limits stay at upstream's defaults)."""
    deadline = None

    def generate_brick_with_rejection_sampling(self, prompt=None, bricks=None):
        if self.deadline is not None and time.time() > self.deadline:
            raise TimeoutError
        return super().generate_brick_with_rejection_sampling(prompt, bricks) if bricks is not None else super().generate_brick_with_rejection_sampling(prompt)


def main():
    LIB.mkdir(exist_ok=True)
    prompts = json.loads((HERE / "prompts.json").read_text())
    t_start = time.time()
    model = Capped(BrickGPTConfig(use_gurobi=False))          # original limits: 500 rejections per brick, 100 rollbacks
    print(f"model loaded in {time.time() - t_start:.1f}s", flush=True)
    t_runs = time.time()
    for seed in SEEDS:
        for p in prompts:
            rid = f"{p['id']}-s{seed}"
            if (LIB / f"{rid}.json").exists():
                continue
            transformers.set_seed(seed)
            t = time.time()
            model.deadline = t + CAP_S
            try:
                out = model(p["prompt"])
            except TimeoutError:                                  # saved empty, so a rerun skips it; shown in runs.jsonl
                rec = {"id": rid, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed, "bricks": 0,
                       "seconds": round(time.time() - t, 1), "timed_out": True, "brick_list": []}
                (LIB / f"{rid}.json").write_text(json.dumps(rec))
                with open(LIB / "runs.jsonl", "a") as f:
                    f.write(json.dumps({k: v for k, v in rec.items() if k != "brick_list"}) + "\n")
                print(rid, f"TIMED OUT after {rec['seconds']} s (cap {CAP_S} s)", flush=True)
                continue
            secs = round(time.time() - t, 1)
            bricks = out["bricks"]
            rec = {"id": rid, "prompt_id": p["id"], "category": p["category"], "prompt": p["prompt"], "seed": seed,
                   "bricks": len(bricks), "seconds": secs, "rollbacks": out["n_regenerations"], "rejections": out["rejection_reasons"].total(),
                   "connected": bool(bricks.is_connected()) if len(bricks) else False,
                   **{("single_piece" if k == "single_piece" else "linked_" + k): v for k, v in single_piece(bricks.bricks).items()},
                   "types": {k: sum(1 for b in bricks.bricks if part_of(b) == k) for k in sorted({part_of(b) for b in bricks.bricks})},
                   "brick_list": [{"h": b.h, "w": b.w, "x": b.x, "y": b.y, "z": b.z} for b in bricks.bricks]}
            (LIB / f"{rid}.json").write_text(json.dumps(rec))
            (LIB / f"{rid}.ldr").write_text(bricks.to_ldr())
            with open(LIB / "runs.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in rec.items() if k != "brick_list"}) + "\n")
            print(rid, rec["bricks"], "bricks", secs, "s", "connected" if rec["connected"] else "NOT-grounded",
                  "single-piece" if rec["single_piece"] else f"{rec['linked_pieces']} pieces", flush=True)
    print(f"total generation time {time.time() - t_runs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
