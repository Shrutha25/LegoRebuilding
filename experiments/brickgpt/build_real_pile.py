"""Rebuild real_pile.json from the hand-checked ground truth (tests/detection/bricks_only.txt), not from a scan.

    python experiments/brickgpt/build_real_pile.py            (any Python 3)

Only lines of the form "<h>x<w> <colour words> brick- <n>" whose size is one of BrickGPT's 8 types become fixture rows.
Everything else (2x3, 1x3, modified bricks, the technic brick) is listed under "other" and is not used by the BrickGPT matcher.
Colour words -> LDraw/Rebrickable colour names:
  "grey" is read as Dark Bluish Gray (the .txt gives no shade; the brick looks dark slate in the photo - an assumption),
  "bright yellow" as Bright Light Yellow (the pale 2x2; Rebrickable's plain "Yellow" is the saturated 1x3 and 1x6),
  "brown" as Reddish Brown (glare makes it look orange).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TXT = HERE.parents[1] / "tests" / "detection" / "bricks_only.txt"
OUT = HERE / "real_pile.json"

SIZE_PART = {"1x1": "3005", "1x2": "3004", "1x4": "3010", "1x6": "3009", "1x8": "3008", "2x2": "3003", "2x4": "3001", "2x6": "2456"}
COLOURS = {"red": "Red", "grey": "Dark Bluish Gray", "blue": "Blue", "medium azure": "Medium Azure", "bright yellow": "Bright Light Yellow",
           "brown": "Reddish Brown", "yellowish green": "Yellowish Green", "yellow": "Yellow", "lime": "Lime"}
LINE = re.compile(r"^(\d+x\d+)\s+(.+?)\s*-\s*(\d+)\s*$")


def parse(text):
    """-> (rows for the 8 types, other lines, total bricks)"""
    rows, other, total = [], [], 0
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        size, desc, n = m.group(1), m.group(2).lower(), int(m.group(3))
        total += n
        plain = re.fullmatch(r"(.+) brick", desc)       # "modified ...", "technic, brick with hole" do not match
        if size in SIZE_PART and plain and plain.group(1) in COLOURS:
            rows.append({"part": SIZE_PART[size], "colour": COLOURS[plain.group(1)], "qty": n})
        else:
            other.append({"line": line.strip(), "qty": n})
    return rows, other, total


def main():
    rows, other, total = parse(TXT.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps({"source": "tests/detection/bricks_only.txt (hand-checked ground truth), not a scan",
                               "total_bricks_in_photo": total, "rows": rows, "other": other}, indent=1))
    print(sum(r["qty"] for r in rows), "of", total, "bricks are BrickGPT types:", rows)


if __name__ == "__main__":
    main()
