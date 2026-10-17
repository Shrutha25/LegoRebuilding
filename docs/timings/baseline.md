# Baseline timings (before the "Designed for your pile" work)

Measured with `python -m backend.bench bench_photos/photo1.jpg ... photo5.jpg` on 2026-10-17: five real photos of loose pieces on a dark mat
(6, 8, 13, 14 and 18 detected pieces), the app's own requests in the order the results screen makes them, real Brickognize, real Gemini.
Detection is `opencv` (the default). Gemini: `gemini-flash-lite-latest` (the configured `gemini-3.1-flash-lite` had used its local budget of 400 for
the day in earlier recipe-generation studies; a first pass with Gemini refused is in `baseline-no-gemini-quota.*` and shows the non-AI stages are
identical). Budget settings: `GEMINI_RPM=10` (default), `GEMINI_RPD=400`. Raw numbers: `baseline.json`; server stages: `timings.jsonl` rows.
Re-run the table any time with `python -m backend.timing_report --stages`.

## What the user waits for, per photo (client wall clock, ms)

| photo | pieces | scan (detect + identify) | resolve | official sets (`/api/match`) | nearby sets | designs (`/api/builds/recipes`) | text ideas (`/api/ideas`) |
|---|---|---|---|---|---|---|---|
| 1 | 18 | 2640 | 52 | 72 | 1467 (cold index) | 14250 | 2680 |
| 2 | 6 | 2079 | 2 | 16 | 4 | 45541 | 2855 |
| 3 | 8 | 2592 | 1 | 31 | 3 | 11827 | 3570 |
| 4 | 13 | 2748 | 2 | 73 | 4 | 43735 | 4654 |
| 5 | 14 | 2510 | 2 | 79 | 2 | 10635 | 4293 |

## Server stages, p50 / p95 over the five photos (ms; a stage's time is summed inside one request)

| request | stage | p50 | p95 |
|---|---|---|---|
| scan_job | whole job | 2580 | 2714 |
|  | detect | 1077 | 1310 |
|  | identify (all crops through Brickognize, 8 in flight) | 1137 | 1409 |
|  | brickognize (sum of the individual requests) | 7181 | 9726 |
|  | crops + save_crops | 248 | 299 |
| /api/match | whole request | 70 | 76 |
|  | sets.match | 63 | 75 |
| /api/sets/nearby | whole request | 2 | 1173 |
|  | nearby.index (built on first use) | 0 | 1171 |
| /api/builds/recipes | whole request | 14248 | 45178 |
|  | concepts (the concept Gemini call, `gemini.build`) | 2468 | 3492 |
|  | generate (on-demand recipe writing, up to 4 Gemini calls + vision) | 9437 | 41911 |
|  | solver (the filler, all concepts) | 534 | 858 |
|  | thumbnail / ldr.pack / checks / ldr (per build, 0-2 builds) | 36 / 9 / 0 / 0 | 55 / 13 / 1 / 0 |
|  | gemini.budget_wait | 32 | 1541 |
| /api/ideas | whole request | 3567 | 4579 |
|  | gemini.build | 2930 | 3450 |

Gemini calls per request: designs 5 (p50 and max), ideas 1, everything else 0.

## What it says

* Everything that is not Gemini is fast except the very first nearby-sets request: the part-to-sets index is built on first use (1.2-1.4 s) and
  should be built at start-up.
* Photo to inventory costs 2.1-2.7 s (detect about 1.1 s, Brickognize about 1.1 s with the requests in parallel; the sum of the individual
  Brickognize requests is 7-10 s).
* The designs request is the slow one and the variance is the on-demand recipe generation: 5 Gemini calls and 10-45 s per request, for 0-2 cards
  (photos 2, 3 and 5 returned no design at all with 6-14 pieces). The concept call alone is about 2.5 s.
* The text ideas call is about 3.6 s.
