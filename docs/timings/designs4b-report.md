# Concept model comparison (look check on the app's own model in every run)

| | flash-lite | flash |
|---|---|---|
| designs shown, photo1 (mean of repeats) | 1.67 | 3 |
| designs shown, photo2 (mean of repeats) | 0 | 0 |
| designs shown, photo3 (mean of repeats) | 0 | 0 |
| designs shown, photo4 (mean of repeats) | 0.33 | 1 |
| designs shown, photo5 (mean of repeats) | 0 | 0 |
| designs shown, photo6 (mean of repeats) | 2.33 | 3 |
| designs shown, photo7 (mean of repeats) | 1 | 2 |
| designs shown per request (mean) | 0.76 | 1.29 |
| designs produced before the look check (mean) | 0.86 | 1.29 |
| requests with no design | 12 of 21 | 3 of 7 |
| look check pass rate, child question (new) | 89% (16/18) | 100% (9/9) |
| look check pass rate, pick-the-name question (old) | 72% (13/18) | 100% (9/9) |
| time to first design p50 / p95 (ms) | 5215 / 27967 | 6128 / 13330 |
| time to last design p50 / p95 (ms) | 6582 / 28795 | 19999 / 81585 |
| request done p50 / p95 (ms) | 17407 / 50820 | 22655 / 92284 |
| Gemini calls per request (mean) | 2.29 | 2.43 |
| est. tokens per request in / out | 4122 / 1056 | 4355 / 1160 |
| est. cost per request (USD, assumed prices) | 0.0008 | 0.0040 |

## Stage funnel (all requests)

| stage | flash-lite | flash |
|---|---|---|
| concepts | 192 | 72 |
| sketch_ok (of concepts) | 95 (49%) | 30 (42%) |
| solver_ok (of sketch_ok) | 18 (19%) | 9 (30%) |
| checks_ok (of solver_ok) | 18 (100%) | 9 (100%) |
| diverse_ok (of checks_ok) | 18 (100%) | 9 (100%) |
| look_checked (of diverse_ok) | 18 (100%) | 9 (100%) |
| look_ok (of look_checked) | 16 (89%) | 9 (100%) |

## Tolerance and diversity (all requests)

| | flash-lite | flash |
|---|---|---|
| designs rescued by the 85% + outline rule | 8 | 2 |
| rejected because the outline was lost | 6 | 0 |
| concepts rejected for the wrong size | 39 | 3 |
| concepts dropped as duplicates (shape or bounding box) | 49 | 39 |
