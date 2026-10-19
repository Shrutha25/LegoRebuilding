"""Phase 0 spike: 5 unconstrained prompts, 1 sample each, no Gurobi, no bpy/ImportLDraw.

Run with the experiment venv:  experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/run_phase0.py
HF_TOKEN is read from the environment by huggingface_hub; this script never touches it.
"""
import json
import time
from collections import Counter
from pathlib import Path

import torch
import transformers

from brickgpt.models import BrickGPT, BrickGPTConfig

PROMPTS = ["a chair", "a car", "a table", "a guitar", "a bench"]
SEED = 42
OUT = Path(__file__).resolve().parents[2] / "docs" / "brickgpt" / "phase0"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = BrickGPT(BrickGPTConfig(use_gurobi=False))
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s on {model.device}", flush=True)

    results = []
    for prompt in PROMPTS:
        name = prompt.replace("a ", "").replace(" ", "-")
        transformers.set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        out = model(prompt)
        secs = time.time() - t
        bricks = out["bricks"]
        counts = Counter(f"{b.h}x{b.w}" if b.h <= b.w else f"{b.w}x{b.h}" for b in bricks.bricks)
        (OUT / f"{name}.txt").write_text(bricks.to_txt())
        (OUT / f"{name}.ldr").write_text(bricks.to_ldr())
        rec = {
            "prompt": prompt,
            "seed": SEED,
            "seconds": round(secs, 1),
            "bricks": len(bricks),
            "brick_types": dict(sorted(counts.items())),
            "connected": bool(bricks.is_connected()),  # NOT physically verified (no Gurobi)
            "rollbacks": out["n_regenerations"],
            "rejections": out["rejection_reasons"].total(),
            "rejection_reasons": dict(out["rejection_reasons"]),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)
    (OUT / "results.json").write_text(json.dumps({"model_load_s": round(load_s, 1), "results": results}, indent=2))


if __name__ == "__main__":
    main()
