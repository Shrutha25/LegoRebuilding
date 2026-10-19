"""StableText2Brick (Hugging Face AvaLovelace/StableText2Brick, MIT) -> the reverse-lookup index. Needs pyarrow<22 (newer ones want NumPy 2).

    venv/Scripts/python.exe ingest_dataset.py            describe the dataset: schema, splits, brick-count percentiles (writes nothing)
    venv/Scripts/python.exe ingest_dataset.py convert    build the index, entries, .ldr files, shortlist and manifest
Everything bulky goes to dataset/ (gitignored); library_manifest.json (small, deterministic) is the only committed output.

convert is resumable and idempotent: a shard whose dataset/index/<shard>.jsonl exists is skipped, an entry or .ldr that exists is not
rewritten, and the merged index, shortlist and manifest are rebuilt from those files every time (--force redoes the shards).
dataset/index/*.jsonl         one light record per dataset structure (no brick list), including structures over the cutoff
dataset/library_index.jsonl   the above + the generated designs from library/, the index reverse_lookup loads
dataset/entries/<id>.json     library/-format record with brick_list, for structures within max_bricks (others: read the shard row)
dataset/ldr/<id>.ldr          plain single-colour LDraw file with "0 STEP" per layer group, same code as pack_library
dataset/shortlist.json        ids for the curation page (see library_index.shortlist)"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inventory import make_steps, read_ldconfig, to_ldr  # noqa: E402
from library_index import (DATASET_DIR, GRID, LBrick, dataset_entry, generated_entry, load_config, parse_bricks,  # noqa: E402,F401
                           shortlist, stability_summary)

DATASET = "AvaLovelace/StableText2Brick"
MANIFEST = HERE / "library_manifest.json"
BATCH = 256
COLOUR = "Medium Lime"


def download():
    return Path(snapshot_download(DATASET, repo_type="dataset", local_dir=DATASET_DIR, allow_patterns=["data/*.parquet", "README.md"]))


def splits(root):
    """{split name: [parquet files]} from the data/<split>-NNNNN-of-NNNNN.parquet names."""
    out = {}
    for f in sorted((root / "data").glob("*.parquet")):
        out.setdefault(f.name.split("-")[0], []).append(f)
    return out


def describe():
    root = download()
    for name, files in splits(root).items():
        table = pq.ParquetFile(files[0])
        print(f"\n== split {name}: {len(files)} file(s), {sum(pq.ParquetFile(f).metadata.num_rows for f in files)} rows")
        if name == next(iter(splits(root))):
            print("schema:", table.schema_arrow)
        counts, bad, size_hist = [], 0, Counter()
        for f in files:
            for bricks in pq.read_table(f, columns=["bricks"]).column("bricks").to_pylist():
                try:
                    bl = parse_bricks(bricks)
                except ValueError:
                    bad += 1
                    continue
                counts.append(len(bl))
                size_hist.update(f"{min(b['h'], b['w'])}x{max(b['h'], b['w'])}" for b in bl)
        a = np.array(counts)
        print("unparsable structures:", bad)
        print("bricks/structure  min/mean/max:", a.min(), round(a.mean(), 1), a.max())
        print("percentiles 5/25/50/75/90/95/99:", [int(np.percentile(a, p)) for p in (5, 25, 50, 75, 90, 95, 99)])
        print("brick sizes:", dict(size_hist.most_common()))


def write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def index_shard(parquet: Path, cfg) -> Counter:
    """dataset/index/<shard>.jsonl for one parquet file. Returns skip reasons (structures that could not be indexed)."""
    skipped, lines = Counter(), []
    pf = pq.ParquetFile(parquet)
    row_no = 0
    for batch in pf.iter_batches(batch_size=BATCH):
        flat = batch.column("stability_scores").flatten().flatten().flatten().to_numpy(zero_copy_only=False)
        n = batch.num_rows
        if len(flat) != n * GRID ** 3:
            raise ValueError(f"{parquet.name}: stability grid is not {GRID}^3")
        flat = flat.reshape(n, GRID ** 3)
        for i, row in enumerate(batch.to_pylist()):
            try:
                bl = parse_bricks(row["bricks"])
                n_bricks = len(bl)
                rec = dataset_entry(row, parquet.stem, row_no + i, cfg, stab=stability_summary(flat[i], bl), analyse=n_bricks <= cfg["analyse_max_bricks"])
            except ValueError as e:
                skipped[str(e).split(":")[0]] += 1
                continue
            lines.append(json.dumps(rec, sort_keys=True))
        row_no += n
    write_atomic(DATASET_DIR / "index" / f"{parquet.stem}.jsonl", "\n".join(lines) + "\n")
    return skipped


def read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generated_records(cfg) -> list:
    out = []
    for f in sorted((HERE / "library").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("brick_list"):
            out.append((generated_entry(rec, cfg), rec))
    return out


def materialise(index, cfg, root):
    """Write dataset/entries/<id>.json and dataset/ldr/<id>.ldr for dataset structures within max_bricks, reading only the shards that
    still have something missing. Returns the number written."""
    ld = read_ldconfig()
    todo = {}
    for r in index:
        if r["source"] == "dataset" and r["total_bricks"] <= cfg["max_bricks"]:
            if not ((DATASET_DIR / "entries" / f"{r['id']}.json").exists() and (DATASET_DIR / "ldr" / f"{r['id']}.ldr").exists()):
                todo.setdefault(r["shard"], {})[r["row"]] = r
    written = 0
    for shard, rows in todo.items():
        pf = pq.ParquetFile(root / "data" / f"{shard}.parquet")
        row_no = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=["bricks"]):
            for i, text in enumerate(batch.column("bricks").to_pylist()):
                r = rows.get(row_no + i)
                if r is None:
                    continue
                bl = parse_bricks(text)
                write_atomic(DATASET_DIR / "entries" / f"{r['id']}.json", json.dumps({**r, "brick_list": bl}))
                bricks = [LBrick(d) for d in bl]
                write_atomic(DATASET_DIR / "ldr" / f"{r['id']}.ldr", to_ldr(bricks, [COLOUR] * len(bricks), make_steps(bricks), ld))
                written += 1
            row_no += batch.num_rows
    return written


def convert(force=False):
    cfg = load_config()
    root = download()
    files = [f for fs in splits(root).values() for f in fs]
    skipped = Counter()
    for f in files:
        if force or not (DATASET_DIR / "index" / f"{f.stem}.jsonl").exists():
            skipped += index_shard(f, cfg)
            print("indexed", f.name, flush=True)
    index = [r for f in files for r in read_jsonl(DATASET_DIR / "index" / f"{f.stem}.jsonl")]
    ids = Counter(r["id"] for r in index)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate ids in dataset index: {dup[:5]}")
    gen = [g for g, _ in generated_records(cfg)]
    index += gen
    write_atomic(DATASET_DIR / "library_index.jsonl", "\n".join(json.dumps(r, sort_keys=True) for r in index) + "\n")
    print("materialised", materialise(index, cfg, root), "entries + .ldr files")
    short = shortlist(index, cfg)
    write_atomic(DATASET_DIR / "shortlist.json", json.dumps([r["id"] for r in short], indent=1))
    ds = [r for r in index if r["source"] == "dataset"]
    cuts = {str(c): sum(r["total_bricks"] <= c for r in ds) for c in cfg["report_cutoffs"]}
    manifest = {
        "dataset": DATASET, "splits": {n: sum(pq.ParquetFile(f).metadata.num_rows for f in fs) for n, fs in splits(root).items()},
        "indexed": {"dataset": len(ds), "brickgpt_generated": len(gen), "skipped_unsupported": dict(skipped) or None},
        "dataset_structures_with_total_bricks_at_most": cuts,
        "within_max_bricks": {"max_bricks": cfg["max_bricks"], "dataset": sum(r["total_bricks"] <= cfg["max_bricks"] for r in ds),
                              "dataset_single_piece": sum(r["total_bricks"] <= cfg["max_bricks"] and r["single_piece"] is True for r in ds)},
        "shortlist": {"total": len(short), "by_source": dict(Counter(r["source"] for r in short)),
                      "by_category": dict(sorted(Counter(r["category"] for r in short).items()))},
        "config": {k: cfg[k] for k in ("max_bricks", "max_bbox", "analyse_max_bricks", "min_stability", "dedupe", "shortlist_max")},
    }
    write_atomic(MANIFEST, json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("indexed", "dataset_structures_with_total_bricks_at_most", "within_max_bricks", "shortlist")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs="?", choices=["describe", "convert"], default="describe")
    ap.add_argument("--force", action="store_true", help="re-index shards that already have an index file")
    a = ap.parse_args()
    convert(a.force) if a.command == "convert" else describe()
