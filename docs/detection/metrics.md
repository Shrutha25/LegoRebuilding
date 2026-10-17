# Detection before / after

Per scene: correct / split / merge / missed / false (see `tests/detection/metrics.py`). Before = detector at commit f45a18f (boxes), after = mask pipeline.

| scene | pieces | before | after |
|---|---|---|---|
| green_plate | 10 | 7 / 1 / 0 / 2 / 0 | 10 / 0 / 0 / 0 / 0 |
| two_same_colour_bricks | 2 | 0 / 0 / 1 / 0 / 0 | 2 / 0 / 0 / 0 / 0 |
| rotated_plate | 1 | 0 / 0 / 0 / 1 / 0 | 1 / 0 / 0 / 0 / 0 |
| beam_and_round_brick | 2 | 1 / 0 / 0 / 1 / 0 | 2 / 0 / 0 / 0 / 0 |
| two_tone_and_printed | 2 | 2 / 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 / 0 |
| lit_and_shaded_faces | 1 | 1 / 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 / 0 |
| quilted_towel | 6 | 4 / 0 / 0 / 2 / 1 | 5 / 0 / 0 / 1 / 0 |
| **total** | 24 | 15 / 1 / 1 / 6 / 1 | 23 / 0 / 0 / 1 / 0 |

Acceptance (split errors drop; merge errors rise by at most 1): **PASS**

## Thresholds

| name | default |
|---|---|
| `SEG_SEAM_DL` | 12.0 |
| `SEG_SEAM_WIDTH_PITCH` | 0.15 |
| `SEG_SEAM_COVER` | 0.7 |
| `SEG_SEAM_MIN_RECT` | 0.9 |
| `SEG_GRID_OFFSET` | 0.25 |
| `SEG_GRID_ANGLE_DEG` | 10.0 |
| `SEG_GRID_MIN_STUDS` | 3 |
| `SEG_COLOUR_SPREAD` | 8.0 |
| `SEG_COLOUR_HUE_GAP` | 20.0 |
| `SEG_COLOUR_MIN_CHROMA` | 15.0 |
| `SEG_COLOUR_MIN_SHARE` | 0.15 |
| `SEG_COLOUR_SURFACE_CHROMA` | 25.0 |
| `SEG_COLOUR_SURFACE_SHARE` | 0.6 |
| `SEG_COLOUR_L_WEIGHT` | 0.3 |
| `SEG_SOLIDITY` | 0.85 |
| `SEG_SHAPE_MIN_SOLIDITY` | 0.5 |
| `SEG_DEFECT_PITCH` | 0.5 |
| `SEG_FOOTPRINT_TOL` | 0.35 |
| `SEG_MAX_STUDS` | 48 |
| `SEG_SIZE_MIN_STUD_SHARE` | 0.5 |
| `SEG_MIN_PART_PITCH2` | 0.75 |
| `SEG_MERGE_DILATE` | 2 |
| `SEG_MERGE_SURFACE_FRAC` | 0.2 |
| `SEG_NEUTRAL_L_GAP` | 60.0 |
| `SEG_MERGE_DUPLICATE` | 0.1 |
| `SEG_MERGE_ROUNDS` | 8 |
| `SEG_ADD_UNCOVERED` | 0.6 |
| `SEG_MIN_SLIVER_PITCH` | 0.5 |
| `SEG_FRINGE_CONTACT` | 0.35 |
| `SEG_MIN_SLIVER_FRAC` | 0.012 |
| `SEG_AMBIGUITY_MARGIN` | 0.2 |
| `SEG_MAX_TIEBREAKS` | 3 |
| `SEG_TIEBREAK_DELTA` | 0.1 |
| `SEG_OVERLAP_CHECK` | 0.1 |
| `SEG_CROP_MARGIN_PITCH` | 0.5 |
| `SEG_CROP_MARGIN_FALLBACK` | 0.06 |
