"""Compare the look check with human labels: does it keep what people find recognisable and remove what they don't?

    python -m backend.look_calibration --template labels.csv docs/timings/designs-bench.json     # writes a sheet to fill in: label, title, recognisable
    python -m backend.look_calibration labels.csv docs/timings/designs-bench.json                # agreement report

The labels are the ones printed on the contact sheet (`photo3#2`). In the CSV put `y` (recognisable) or `n` (not) in the last column; rows left empty are skipped.
A bench file from before the child question only knows kept / removed (the old question); a newer one also has both answers per design, and the report
then shows both questions. Nothing here changes the look check: this only measures it.
"""

import argparse
import csv
import json
from pathlib import Path


def designs(bench):
    out = {}
    for p in bench["photos"]:
        for i, d in enumerate(p.get("designs_all", [])):
            label = d.get("label") or f"{Path(p['photo']).stem}#{i + 1}"
            v = d.get("verdict") or {}
            kept = not d.get("removed")
            if v:
                out[label] = {"title": d["title"], "kept": kept, "child": v.get("ok_child"), "choice": v.get("ok_choice")}
            else:                                                           # an older bench: only kept / removed, which was the old question
                out[label] = {"title": d["title"], "kept": kept, "child": None, "choice": kept}
    return out


def template(bench, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "title", "recognisable (y/n)"])
        for label, d in designs(bench).items():
            w.writerow([label, d["title"], ""])


def confusion(labels, decisions):
    """labels {label: bool recognisable}, decisions {label: bool kept}: tp = recognisable and kept ..."""
    tp = sum(1 for k, v in labels.items() if v and decisions.get(k) is True)
    fn = sum(1 for k, v in labels.items() if v and decisions.get(k) is False)
    fp = sum(1 for k, v in labels.items() if not v and decisions.get(k) is True)
    tn = sum(1 for k, v in labels.items() if not v and decisions.get(k) is False)
    n = tp + fn + fp + tn
    return {"n": n, "agreement": round((tp + tn) / n, 3) if n else None, "kept_recognisable": tp, "removed_recognisable": fn, "kept_unrecognisable": fp, "removed_unrecognisable": tn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def load_labels(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[2].strip().lower() in ("y", "n", "yes", "no", "1", "0"):
                out[row[0].strip()] = row[2].strip().lower() in ("y", "yes", "1")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="labels CSV, or the file to write with --template")
    ap.add_argument("bench")
    ap.add_argument("--template", action="store_true")
    args = ap.parse_args(argv)
    bench = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    if args.template:
        template(bench, args.labels)
        print(f"wrote {args.labels}")
        return
    labels, ds = load_labels(args.labels), designs(bench)
    missing = [k for k in labels if k not in ds]
    print(f"{len(labels)} labelled designs, {len(missing)} not in {args.bench}: {missing[:8]}")
    for name, key in (("old question (picked its own name)", "choice"), ("new question (could a child see it?)", "child")):
        dec = {k: d[key] for k, d in ds.items() if d[key] is not None}
        if dec:
            print(f"\n{name}:\n  {json.dumps(confusion({k: v for k, v in labels.items() if k in dec}, dec))}")
    print("\nrecognisable designs the check removed: " + ", ".join(f"{k} ({ds[k]['title']})" for k, v in labels.items() if v and k in ds and not ds[k]["kept"]))
    print("unrecognisable designs the check kept: " + ", ".join(f"{k} ({ds[k]['title']})" for k, v in labels.items() if not v and k in ds and ds[k]["kept"]))


if __name__ == "__main__":
    main()
