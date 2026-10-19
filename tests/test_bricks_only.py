"""Detection test case: tests/detection/bricks_only.jpeg, a real photo of 16 loose bricks on a textured mat.

Ground truth is bricks_only.txt (16 bricks), written out as part ids in bricks_only.expected.json. The scan result is a recording
(bricks_only.scan.json), so this makes no live Brickognize or detector call. When the detector is fixed, re-record the scan.

Known failure (recorded, deliberately not fixed here; it belongs with the split/merge work): the grey 2x6 is detected as three
overlapping pieces. The colour mistakes on this photo (blue 2x4 read as Medium Azure, brown 2x2 read as Orange) are logged in the
expected file under known_colour_errors only; no test asserts colours.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

D = Path(__file__).parent / "detection"
EXPECTED = json.loads((D / "bricks_only.expected.json").read_text(encoding="utf-8"))
SCAN = json.loads((D / "bricks_only.scan.json").read_text(encoding="utf-8"))


def expected_parts():
    c = Counter()
    for p in EXPECTED["pieces"]:
        c[p["part_id"]] += p["qty"]
    return c


def scanned_parts():
    return Counter(p["part_id"] for p in SCAN["pieces"])


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return min(ax + aw, bx + bw) > max(ax, bx) and min(ay + ah, by + bh) > max(ay, by)


def test_expected_result_is_16_pieces_with_one_grey_2x6():
    assert EXPECTED["pieces_total"] == sum(p["qty"] for p in EXPECTED["pieces"]) == 16
    grey = [p for p in EXPECTED["pieces"] if p["part_id"] == "2456" and p["colour"] == "Dark Bluish Gray"]
    assert len(grey) == 1 and grey[0]["qty"] == 1


@pytest.mark.xfail(strict=True, reason="known failure: the grey 2x6 comes back as three pieces (2x6 + a 2x4 + a wedge); see the split/merge work")
def test_scan_finds_16_pieces_with_the_part_types_of_the_txt():
    assert len(SCAN["pieces"]) == 16
    assert scanned_parts() == expected_parts()


def test_recorded_failure_is_only_the_grey_2x6_split():
    """Pins down what is wrong today, so the xfail above cannot hide a different regression."""
    got, want = scanned_parts(), expected_parts()
    assert len(SCAN["pieces"]) == 18
    assert not (want - got)                                                   # nothing is missed or merged away
    assert got - want == Counter({"3001": 1, "77180": 1})                    # the extra 2x4 and the wedge
    grey = [p["box"] for p in SCAN["pieces"] if p["part_id"] == "2456" and p["color"] == "Light Bluish Gray"]
    extras = [p["box"] for p in SCAN["pieces"] if p["part_id"] == "77180" or (p["part_id"] == "3001" and p["color"] == "Light Bluish Gray")]
    assert len(grey) == 1 and len(extras) == 2
    assert all(overlaps(e, grey[0]) for e in extras)                         # both sit on top of the grey 2x6
