# Step 1 timings: instant sources

Same method as `baseline.md` (`python -m backend.bench`, real Brickognize and Gemini, `gemini-flash-lite-latest`), now on 7 real photos (the 5 of the
baseline plus the only 2 other distinct real photos on this machine). Raw numbers: `step1.json`. The three instant sources (official sets, nearby
sets, real sub-builds) are fired together, as the results screen does.

## Instant sources (client wall clock, ms)

| photo | pieces | official sets | nearby sets | sub-builds | all three together |
|---|---|---|---|---|---|
| 1 | 18 | 155 | 155 | 151 | 156 |
| 2 | 6 | 11 | 34 | 29 | 35 |
| 3 | 8 | 48 | 50 | 48 | 51 |
| 4 | 13 | 63 | 64 | 62 | 65 |
| 5 | 14 | 66 | 67 | 65 | 68 |
| 6 | 17 | 69 | 70 | 69 | 71 |
| 7 | 16 | 62 | 62 | 60 | 63 |

All three are on screen within 160 ms of the request (target: 2 s). Baseline for nearby sets was 1.2-1.4 s on the first request (index built on
first use) and 2-4 ms after; it is now built at start-up.

## Server stages, p50 / p95 over the 7 photos (ms)

| request | stage | baseline p50 / p95 | now p50 / p95 |
|---|---|---|---|
| /api/sets/nearby | whole request | 2 / 1173 | 61 / 128 (three requests run concurrently now) |
|  | nearby.index (build the index) | 0 / 1171 | 8 / 38 (lookup only) |
| /api/match | whole request | 70 / 76 | 61 / 127 |
| /api/subbuilds (new) | whole request | - | 59 / 123 |
|  | subbuilds.map (parts to LDraw numbers and colours) | - | 27 / 79 |
|  | subbuilds.match (5,500 sub-builds) | - | 4 / 24 |
|  | subbuilds.rows (names, attribution) | - | 20 / 31 |
| scan_job | whole job | 2580 / 2714 | 2091 / 2339 |

Start-up warm-up (background thread, `/api/health` `indexes`): nearby-sets index 1.25 s, LDraw library 0.1 s, OMR index 0.17 s from its pickle
(1.4 s to build it from the 1,288 model files the first time; downloading the models is a one-off minute with `fetch_omr`).

The Gemini stages are unchanged by this step (designs: 5 calls, p50 12 s, p95 45 s; ideas: 1 call, p50 2.8 s). On photos 6 and 7 Gemini was refused by
the local limit of 10 requests per minute because the benchmark runs the photos back to back: those two rows have no AI numbers.

## Real sub-builds on the benchmark piles

Matching is correct (a control that feeds the exact parts of 10 indexed sub-builds finds 9; the miss contains a Technic primitive), all pieces of every
photo map to LDraw parts, and the answer is the same in both colour modes: **no sub-build of 6-80 parts is buildable or within 2 parts for any of the
7 piles**.

| photo | pieces | fully buildable | almost buildable (1-2 missing) | smallest | closest build (parts, missing; colour off) |
|---|---|---|---|---|---|
| 1 | 18 | 0 | 0 | - | 8-part "2" (21007-1), 6 missing |
| 2 | 6 | 0 | 0 | - | 9-part "Phase 5-6" (5542-1), 6 missing |
| 3 | 8 | 0 | 0 | - | 8-part "counter" (3796-1), 6 missing |
| 4 | 13 | 0 | 0 | - | 8-part "Torpedo Launcher" (7141-1), 5 missing |
| 5 | 14 | 0 | 0 | - | 8-part "sink" (3061-1), 6 missing |
| 6 | 17 | 0 | 0 | - | 12-part "Ship" (4018-1), 6 missing |
| 7 | 16 | 0 | 0 | - | 9-part "wheelcase" (7740-1), 6 missing |

What would change it (measured offline on the same piles, not enabled): allowing 4 missing parts with colour off gives 1-11 "almost" builds per
pile; lowering the minimum to 3 parts gives 0-6 (colour on) or 7-81 (colour off) "almost" builds, but these are 3-part fragments (a "table" of 3
parts missing 1) and one fully buildable 3-part piece on photo 5, which are not worth showing. The piles are one or two of many different bricks, and
OMR sub-builds repeat a few specific parts.
