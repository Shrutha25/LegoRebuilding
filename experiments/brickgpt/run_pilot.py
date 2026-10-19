"""Small-design pilot: the 12 prompts of prompts.json, reworded "a small, simple ...", seeds 42-44 = 36 generations.

    experiments/brickgpt/venv/Scripts/python.exe experiments/brickgpt/run_pilot.py
Same limits as the library run (BrickGPT defaults, no Gurobi, the permanent 10-minute cap per generation) plus an early abort:
as soon as a design passes MAX_BRICKS (60) bricks it is discarded, not truncated (a cut-off shape is not a design).
The model is loaded once. Saves library/<id>-small-s<seed>.json/.ldr with "tag": "small"; aborted and timed-out runs are saved empty
(so a rerun skips them) and logged in runs.jsonl. A progress line follows every design."""
import json
import sys
import time
from pathlib import Path

import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brickgpt.models import BrickGPTConfig

from inventory import part_of, single_piece
from run_library import CAP_S, LIB, SEEDS, Capped

HERE = Path(__file__).resolve().parent
MAX_BRICKS = 60


class TooBig(Exception):
    pass


class Pilot(Capped):
    def generate_brick_with_rejection_sampling(self, prompt=None, bricks=None):
        if bricks is not None and len(bricks.bricks) > MAX_BRICKS:      # the 61st brick was placed: discard the design
            raise TooBig
        return super().generate_brick_with_rejection_sampling(prompt, bricks)


def small(prompt: str) -> str:
    """'a simple chair with ...' -> 'a small, simple chair with ...';  'a long bench ...' -> 'a small, simple long bench ...'"""
    rest = prompt[2:] if prompt.startswith("a ") else prompt
    return "a small, simple " + (rest[len("simple "):] if rest.startswith("simple ") else rest)


def main():
    LIB.mkdir(exist_ok=True)
    prompts = json.loads((HERE / "prompts.json").read_text())
    total = len(prompts) * len(SEEDS)
    t_start = time.time()
    model = Pilot(BrickGPTConfig(use_gurobi=False))
    print(f"model loaded in {time.time() - t_start:.1f}s; {total} generations, abort above {MAX_BRICKS} bricks, cap {CAP_S}s", flush=True)
    t_runs, done = time.time(), 0
    for seed in SEEDS:
        for p in prompts:
            rid = f"{p['id']}-small-s{seed}"
            done += 1
            if (LIB / f"{rid}.json").exists():
                print(f"{done}/{total} skipped (already done)", flush=True)
                continue
            text = small(p["prompt"])
            base = {"id": rid, "prompt_id": p["id"], "category": p["category"], "prompt": text, "seed": seed, "tag": "small"}
            transformers.set_seed(seed)
            t = time.time()
            model.deadline = t + CAP_S
            try:
                out = model(text)
                bricks = out["bricks"]
                rec = {**base, "bricks": len(bricks), "rollbacks": out["n_regenerations"], "rejections": out["rejection_reasons"].total(),
                       "connected": bool(bricks.is_connected()) if len(bricks) else False,
                       **{("single_piece" if k == "single_piece" else "linked_" + k): v for k, v in single_piece(bricks.bricks).items()},
                       "types": {k: sum(1 for b in bricks.bricks if part_of(b) == k) for k in sorted({part_of(b) for b in bricks.bricks})},
                       "brick_list": [{"h": b.h, "w": b.w, "x": b.x, "y": b.y, "z": b.z} for b in bricks.bricks]}
                (LIB / f"{rid}.ldr").write_text(bricks.to_ldr())
                status = f"{rec['bricks']} bricks, " + ("one piece" if rec["single_piece"] else f"{rec['linked_pieces']} pieces")
            except (TooBig, TimeoutError) as e:
                key = "too_big" if isinstance(e, TooBig) else "timed_out"
                rec = {**base, "bricks": 0, key: True, "brick_list": []}
                status = f"DISCARDED (over {MAX_BRICKS} bricks)" if key == "too_big" else f"TIMED OUT (cap {CAP_S} s)"
            rec["seconds"] = secs = round(time.time() - t, 1)
            (LIB / f"{rid}.json").write_text(json.dumps(rec))
            with open(LIB / "runs.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in rec.items() if k != "brick_list"}) + "\n")
            print(f"{done}/{total} done, last took {secs:.0f} s  [{rid}: {status}]", flush=True)
    print(f"total pilot generation time {time.time() - t_runs:.0f}s ({(time.time() - t_runs) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
