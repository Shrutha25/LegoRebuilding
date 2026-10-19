# LEGO "What can I build?"

Upload a photo of LEGO pieces spread on a plain background. The app detects each piece
(OpenCV), identifies it (Brickognize), and suggests official sets ranked by how much of
each set you already own (Rebrickable data).

Status: backend (scan, match, parts search, AI ideas) and mobile-first frontend (upload, review,
results, AI ideas tab) done. Photos only (video is under "Later").

Quick start with Docker: `cp .env.example .env`, add your `GEMINI_API_KEY`, then see "Docker" below.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
cp .env.example .env              # then fill in keys (see below)

python -m backend.core.fetch_data # one-off: downloads ~140 MB of Rebrickable CSVs into rb_data/
uvicorn backend.main:app --port 8000
```

The server starts immediately and loads the set data in a background thread. The first
load builds the match index (a few seconds to a minute); it is cached in
`rb_data/index_cache.pkl`, keyed by the CSV mtimes/sizes, so later restarts take about a
second. `/api/scan` works while the data is loading; `/api/match` returns 503 until
`/api/health` reports `data_loaded: true`.

Run `python -m backend.core.fetch_data --refresh` to update the catalogue. The cache
rebuilds itself when the CSVs change.

### Frontend

```bash
cd frontend
npm install
npm run dev          # binds 0.0.0.0:3000 so a phone on the same Wi-Fi can open it
```

The frontend talks to the backend at `NEXT_PUBLIC_API_URL`; if unset it uses
`http://<same host that served the page>:8000`, so nothing needs configuring for a phone.
Start the backend with `--host 0.0.0.0` (`uvicorn backend.main:app --host 0.0.0.0 --port 8000`).
The backend allows localhost and private-network origins (192.168.x, 10.x, 172.16-31.x) via CORS.

**Testing on a phone:** put the phone on the same Wi-Fi, find the computer's LAN IP
(`ipconfig` / `ifconfig`), open `http://<ip>:3000`. Allow ports 3000 and 8000 through the
firewall if the page does not load. "Take photo" opens the rear camera
(`<input capture="environment">`). Plain http works because this is a file input, not
`getUserMedia`.

`npm test` runs the review-screen logic tests (grouping, corrections, merging); `npx tsc --noEmit`
and `npx eslint .` should both be clean.

### Environment variables

| Variable | Purpose |
|---|---|
| `REBRICKABLE_API_KEY` | Optional. BrickLink→Rebrickable ID lookup when local rules fail. |
| `GEMINI_API_KEY` | Needed for AI build ideas (Google AI Studio key); without it `/api/ideas` returns a clear 503 and the rest of the app works. |
| `GEMINI_MODEL` | Model for ideas (default `gemini-3.1-flash-lite`). `python -m backend.live_check --models` lists the Flash models your key can use. |
| `BRICKOGNIZE_URL` | Recognition endpoint (no key needed). |
| `RB_DATA_DIR`, `UPLOAD_DIR` | Data and temp-upload locations. |
| `DETECTOR` | Piece detection: `opencv` (default for now), `gemini`, or `hybrid` (Gemini boxes plus any OpenCV boxes Gemini missed). See "Piece detection". |
| `GEMINI_DETECT_MODEL`, `GEMINI_DETECT_FALLBACK_MODEL` | Model for detection (default `gemini-3.1-flash-lite`), separate from `GEMINI_MODEL`; the fallback is tried on 503/504. |
| `GEMINI_RPM`, `GEMINI_RPD`, `GEMINI_MODEL_LIMITS` | Per-model Gemini limits (placeholders; check Google's published limits). See "Gemini quota budget". |
| `DETECTION_DAILY_SHARE`, `BUILD_RESERVE`, `BUILD_RESERVE_FRAC`, `BUILD_MAX_WAIT` | How the daily quota is split between detection and AI ideas. |
| `GEMINI_RESET_TIME`, `GEMINI_RESET_TZ`, `GEMINI_BUDGET_DB` | When the daily counters reset (default 00:00 America/Los_Angeles) and where they are stored. |
| `BRICKSET_API_KEY`, `INSTRUCTIONS_TTL_DAYS`, `BRICKSET_DAILY_CAP`, `MIN_SET_PARTS`, `EXCLUDED_THEMES`, `EXCLUDED_THEME_WORDS` | Which sets are matched and whether a set card links to official LEGO instructions (Brickset `getInstructions2`, cached in SQLite, links only). See `.env.example`. |
| `LDRAW_DIR`, `BUILD_CACHE_DIR`, `OMR_BASE_URL`, `BUILD_MAX_PARTS`, `BUILD_MAX_DROP_FRACTION`, `BUILD_STEP_SIZE` | 3D building instructions (see "3D building instructions"); frontend flag `NEXT_PUBLIC_BUILD_ANIMATION`. |
| `STUDS_ENABLED`, `STUD_TILT_STRETCH`, `DETECT_DEBUG_DIR` | Stud detection switch (default on); vertical stretch for photos taken at ~45 degrees (default 1.5, 1 = top-down); folder for debug overlays. See "Stud detection". |
| `MIN_COVERAGE` | Minimum share of a set you must own for it to be suggested (default `1.0` = complete sets only). |
| `CORS_ORIGINS` | Comma-separated allowed origins (default `http://localhost:3000`). |

### Docker

```bash
cp .env.example .env                                                 # add GEMINI_API_KEY (ideas only)
docker compose run --rm backend python -m backend.core.fetch_data   # once: fills the rb_data volume
docker compose up --build                                            # app :3000, API :8000
docker compose run --rm --no-deps backend python -m pytest -q       # backend tests, in the container
```

`rb_data` is a named volume, so the download and the index cache survive rebuilds (first start
builds the index in ~6 s, later starts ~0.5 s). The fetch command writes to `$RB_DATA_DIR`, which
the backend image points at that volume. Open `http://localhost:3000`, or `http://<your-ip>:3000`
from a phone. The browser talks to the API at `http://<same host>:8000` unless you build with
`NEXT_PUBLIC_API_URL`. Local and Docker both use Python 3.14. The `.env` file is read by the
backend container (`env_file`) and is never baked into an image.

### Tests

```bash
python -m pytest -q
```

Tests use small fixture CSVs; the one image test uses `lego_pieces.jpg` (expects 42 pieces).

### Timing

Every POST under `/api` and every scan job writes one trace to `timings.jsonl` (`backend/timing.py`; `TIMING_LOG=` switches it off): stages
for detection, Brickognize, each suggest source, every Gemini request (and its budget wait), the solver, checks, LDraw conversion and thumbnails.

    python -m backend.timing_report --stages                    # p50 / p95 per request and stage
    python -m backend.bench bench_photos/*.jpg --out docs/timings/bench.json   # the whole flow on real photos (spends Gemini quota)

The measured baseline is in `docs/timings/baseline.md`.

## API (curl examples)

```bash
# Health: data_loaded=false while the index is still loading
curl localhost:8000/api/health

# Scan a photo -> {"job_id": "..."}   (jpg/png/webp, <= 50 MB; anything else, incl. video, gets a 415)
curl -F "file=@lego_pieces.jpg;type=image/jpeg" localhost:8000/api/scan

# Optional: a photo of the EMPTY surface, then scan with its id; pieces are found by what changed
#   curl -F file=@empty.jpg http://localhost:8000/api/reference      -> {"reference_id": "..."}
#   curl -F file=@bricks.jpg -F reference_id=... http://localhost:8000/api/scan
#   (unknown or expired id -> 404; the id lasts 6 hours; detector is then "reference")
# Poll the job: status is queued|detecting|recognizing|done|error
curl localhost:8000/api/scan/JOB_ID
#   pieces[]: {index, box:[x,y,w,h], crop_url, part_id, name, color, score,
#              alternatives[], needs_review, error}
#   (+ detector: opencv|gemini|hybrid actually used, detector_note: why it fell back, if it did)
#              (+ width/height of the image)

# The (possibly downscaled) photo and each crop
curl -o photo.jpg localhost:8000/api/scan/JOB_ID/image
curl -o crop0.jpg localhost:8000/api/scan/JOB_ID/crop/0

# Match official sets. part_num may be a Brickognize or Rebrickable id; ids are mapped first.
curl -X POST localhost:8000/api/match -H "Content-Type: application/json" -d '{
  "parts": [{"part_num": "3001", "color": "Red", "qty": 4},
            {"part_num": "41879", "color": "Sand Green", "qty": 1}],
  "match_color": false, "max_parts": 150}'
#   -> {"sets": [{set_num, name, year, image, total_parts, parts_you_have, coverage, complete,
#                 missing_count, missing:[{part, color, qty, image}],   (only complete sets by default)
#                 rebrickable_url, instructions_url}],
#       "unmapped": [ids that could not be mapped], "mapped": {original: rebrickable id}}

# Search parts by number or name (manual-correction UI)
curl "localhost:8000/api/parts/search?q=legs"
```

```bash
# Which recognized ids can be matched? Returns Rebrickable id, name, image, unmapped flag
curl -X POST localhost:8000/api/parts/resolve -H "Content-Type: application/json"   -d '{"parts": ["41879", "970c00"]}'

# Image proxy (allowlisted to Brickognize thumbnails; their host serves a content type
# browsers refuse to render in <img>)
curl -o thumb.webp "localhost:8000/api/img?url=https%3A%2F%2Fstorage.googleapis.com%2Fbrickognize-static%2Fthumbnails%2Fv2.23%2Fpart%2F3622%2F154.webp"
```

```bash
# AI build ideas (needs GEMINI_API_KEY). part_num may be any recognized id; name is optional.
curl -X POST localhost:8000/api/ideas -H "Content-Type: application/json" -d '{
  "parts": [{"part_num": "3001", "name": "Brick 2 x 4", "color": "Red", "qty": 4},
            {"part_num": "3622", "name": "Brick 1 x 3", "color": "Lavender", "qty": 2}]}'
#   -> {"ideas": [{name, description, steps[], pieces_used:[{part_num, name, qty, image}]}],
#       "dropped": [{name, reason}]}      (up to 3 ideas; "dropped" lists ideas that failed validation)
```


### AI ideas

Uses Google Gemini through the official `google-genai` SDK. The model is asked for structured
JSON (`response_mime_type="application/json"` with a `response_json_schema`) and sees the user's
exact parts list. Every idea is then validated server-side: each `pieces_used` entry must exist in
the user's list, quantities must be positive integers, and the total used per part (summed over all
entries and colours) cannot exceed what the user owns. An idea that breaks any rule is dropped
whole. We ask for 4 ideas so dropping one still leaves 3. Part names are sanitised before going
into the prompt, and at most 3 calls run at once.

Failures return a readable error and never crash the server; the UI shows an error card with a
retry button: missing/invalid key, quota or rate limit, safety block (prompt or response),
timeout, connection error, truncated output, invalid JSON, or no valid idea. There is no fallback
model for safety blocks yet. Unit tests use a fake Gemini client; they never call the API.
`python -m backend.live_check` runs the real thing against a running backend (see the file's
docstring).

### Piece detection (opencv / gemini / hybrid)

OpenCV finds pieces by colour, which fails on some real photos (a piece lying on another, pale or
clear pieces, uneven light). `DETECTOR` chooses how pieces are found:

- `opencv` - the colour-based detector only (`backend/core/detect.py`).
- `gemini` - ask Gemini for one bounding box per piece (`box_2d = [ymin, xmin, ymax, xmax]`,
  normalised 0-1000; converted to pixels, y first), via structured JSON output.
- `hybrid` - both, merged: every Gemini box is kept; an OpenCV box is added only where Gemini found
  nothing (IoU < 0.3 *and* not mostly inside a Gemini box); anything over 25% of the photo is dropped;
  Gemini boxes get 8% padding before cropping for Brickognize.

A Gemini failure, a 20 s timeout or a missing key never fails a scan: it falls back to OpenCV, logs a
warning, and records why in the scan result (`detector`, `detector_note`). On a 503/504 the call is
retried once with `GEMINI_DETECT_FALLBACK_MODEL`. The thinking level is adapted per model (not every
model accepts "minimal").

**Evaluating:** put photos in `tests/real_photos/` named with the true piece count at the end
(`white_on_white_18.jpg`), optionally with `white_on_white_18.points.json` (true piece centres,
`[[x, y], ...]`) for exact missed/extra/merged numbers, then run

```bash
python -m backend.eval_detect                      # table + annotated images in tests/real_photos/out/
python -m backend.eval_detect --models gemini-3.1-flash-lite,gemini-3.6-flash   # compare models
```

Results are cached, so re-running costs no API calls. Annotated examples from the first evaluation are
in `docs/detection-eval/`.

**Quota:** a Google API key on the free tier allows only about **20 requests per model per day**. Every
hybrid/gemini scan uses one detection request (plus one for AI ideas if opened, which shares
`gemini-3.1-flash-lite`). See "Gemini quota budget" for how the two share it; enable billing in Google AI
Studio for real use.

### Gemini quota budget

Every Gemini request goes through `backend/core/gemini_budget.py`, which counts requests per model (last 60 s and
since the daily reset, default midnight America/Los_Angeles) in a SQLite file, safe across restarts, threads and
worker processes. Retries and fallback-model attempts each count.

- **build** (AI ideas) has priority: it may use everything that remains and waits up to `BUILD_MAX_WAIT` s for a
  per-minute slot, then fails with a clear 429 message.
- **detection** may use at most `DETECTION_DAILY_SHARE` (0.3) of a model's daily limit, never the last
  `BUILD_RESERVE` (20% of the limit) requests, and is skipped at once when the minute window is full: the scan uses
  OpenCV and `detector_note` says `gemini skipped: <reason>`.
- On HTTP 429 the model is paused until the retry delay in the error, or the next reset.

The limits are placeholders (`GEMINI_RPM`, `GEMINI_RPD`, per model `GEMINI_MODEL_LIMITS`): check them against
Google's current published limits. Usage is logged at startup and shown by `GET /api/debug/gemini-budget`.

### Several photos in one list

A piece the same colour as the surface is easy to miss. After the first photo the review screen has **+ Add photo**:
photograph those pieces on a surface of a different colour, and their pieces join the same list (identical parts across
photos become one row with a quantity). Piece numbers continue across photos, the photo viewer has a tab per photo, and
**Next: find builds** uses the combined list. Only photograph pieces that are not on the list yet, or they are counted twice
(each row's quantity can be edited, and rows deleted). **Start over** clears every photo.

### Dark or textured surfaces, and the reference background

- A dim surface (most of the photo below value 110, e.g. dark grey mesh) makes the **surface method lead**: the colour
  seeds there are the mesh's pockets and the pieces' shadows, which bridge neighbouring pieces. Colour boxes are added
  only where the surface method saw nothing. Surface boxes much bigger than the photo's typical box and holding two
  clearly different colours (a lime plate against a red brick) are split, and shadow slivers are ignored. Known limit:
  a near-black piece on dark mesh.
- On light surfaces, a colour box that holds two or more surface boxes is replaced by them.
- **Reference background** (`backend/core/reference.py`): with a photo of the same empty surface taken from the same
  place, the pieces are what differs from it. The reference is aligned (ORB homography, else translation) and brought to
  the photo's exposure, cast shadows (darker, same chroma) are ignored, and the rest is thresholded above the surface's
  own noise. `detect.analyse(img, reference=empty)` uses it alone (no colour, no studs); try it with
  `python -m backend.eval_reference empty.jpg bricks.jpg`, or through the API (`POST /api/reference`, then `POST /api/scan`
  with `reference_id`). **In the app it is step 1 of every photo**: "Step 1 · Photo of the empty background", then "Step 2 ·
  Photo with the pieces" (disabled until step 1 is done; each photo of pieces uses its own empty photo, so an added photo on a
  different background starts with a new empty photo). It needs both photos from the same spot and light; when a handheld
  pair does not line up, the scan falls back to normal detection and shows a warning.
  **Pieces must not touch**: one blob is one piece, so a brick with two colours stays
  whole (`split_touching=True` in `ReferenceConfig` brings back cutting touching pieces by colour).
  Pieces the same colour as the surface (black on black mesh, white on white paper) are found by structure instead of
  colour: on a textured surface the pattern vanishes under smooth plastic (local correlation with the empty photo
  collapses); on a plain surface the piece brings edges, studs and shading that were not there. These cues switch
  themselves off when they fire on a third of the photo (camera moved, light changed).

### 3D building instructions

After picking a build, **"See it built in 3D, step by step"** (AI idea) or **"View building steps in 3D"** (official set) opens a viewer
(three.js, loaded only when opened) that shows the model step by step: the parts added in this step in full colour with an outline,
earlier parts faded, a callout with a rendered thumbnail, count, name and colour of each new part, a progress bar you can tap or drag,
Previous/Next (or swipe the instruction panel, or arrow keys), drag to rotate (kept from step to step), pinch/scroll to zoom,
Reset view, and a finished-model screen (turning slowly, "Build again" / "Back to suggestions"). `NEXT_PUBLIC_BUILD_ANIMATION=1`
drops new parts in. Phones get a lighter model (no conditional edge lines, no environment map, lower pixel ratio).

Everything is converted to ONE format, an LDraw model with `0 STEP` lines, and rendered by one viewer.

- **Setup:** `python -m backend.core.fetch_ldraw` (one-off, ~146 MB zip of the official LDraw parts library into `ldraw_data/`).
  Without it the endpoints answer 503 with that command. The library is never served: `ldraw_lib.pack_mpd` writes one `.mpd` per
  build with only the files it uses (named the way three.js' LDrawLoader looks them up) and caches it by build id in `build_cache/`.
- **Official sets:** the model comes from the LDraw Official Model Repository (`https://library.ldraw.org/library/omr/<set number>.mpd`,
  same set numbers as Rebrickable). Only models whose header licence is CCAL / CC BY are used; their authors and licence are shown in
  the viewer. A model with no steps gets generated steps (flagged in the viewer). If there is no model the card shows the parts list
  (have / missing against your pieces) and a button to LEGO's own instructions page; LEGO's PDFs and images are never fetched or
  re-hosted. The inventory comes from the Rebrickable CSVs the app already uses (the same data as the API, without rate limits); the
  Rebrickable API (`REBRICKABLE_API_KEY`) is used only to look up LDraw ids for parts the local rules cannot map.
- **AI builds:** Gemini (budget purpose `build`) places parts on a stud grid (`x, z` studs, `y` plates, rotation 0/90/180/270, step);
  `buildmodel.py` validates it locally (inventory and counts, no overlaps, every part resting on studs, step order, one structure or
  reported sub-assemblies) and repairs it (snap, nudge overlaps, swap to an inventory equivalent, lower hanging parts, drop what rests
  on nothing). Only if more than `BUILD_MAX_DROP_FRACTION` (0.2) of the parts had to be dropped is Gemini asked once more, with the
  problems in the prompt. Bad or missing steps are regenerated bottom-up, 1-3 close parts per step. `ldrconv.py` writes the LDraw file
  (1 stud = 20 LDU, 1 plate = 8 LDU, -Y up, origin at the top-middle of each part; checked against the real part geometry in tests).
  Supported parts: plain bricks, plates and tiles (`partdims.py` is the one table of sizes).
- **Mapping** (`ldmap.py`): our part ids -> LDraw files (the id, the Rebrickable id, its parent for print/pattern variants, suffix
  rewrites, Rebrickable external ids; cached in `build_cache/ldraw_ids.json`, misses are logged) and colour names/ids -> LDraw colour
  codes through LDConfig (all 274 Rebrickable colours map). Of the pieces in Rebrickable's sets 97.1% have an LDraw file (43.3% of all
  catalogue part numbers: the rest are stickers, prints and minifigure parts).
- **API (new):** `POST /api/builds/ai`, `POST /api/sets/{set}/build`, `GET /api/builds/{id}`, `GET /api/builds/{id}/model.mpd[?lite=1]`,
  `GET /api/ldraw/LDConfig.ldr`, `GET /api/builds-status`. Existing endpoints are unchanged.
- **Demo:** `/viewer-demo?build=ai|set&step=N` (or `&done=1`) shows two bundled builds without the backend (`python -m
  tests.make_demo_builds` regenerates them; screenshots in `docs/viewer/`).

### Recipe builds (concepts first, the code builds)

Free-form AI placement could not guarantee that a build looks like its name, so Gemini never places a brick. A request goes through
`backend/recipes/suggest.py`:

1. **Feasible categories**: which kinds of models the pieces support (enough parts, a long enough part, enough studs). Vehicles are never
   offered: they need wheel sets the 3D grid cannot place yet, and plain bricks are never shown as wheels.
2. **Concepts (Stage A, `concepts.py`)**: Gemini proposes 10-15 concepts as JSON (name, category, one-line description, silhouette, bounding
   box in studs, required features), limited to the feasible categories and told they must differ in silhouette and structure, not colour.
   A name that is just a colour and a category ("Red House") is dropped. Without Gemini, the recipes' own silhouettes and names are used.
3. **Construction (Stage B)**: a concept goes to a recipe of its category (the recipe library first); the recipe's parameters follow the
   concept's silhouette and bounding box (the nearest allowed setting); the **filler** tiles it from the inventory (`filler.py`, unchanged
   rules: largest part first, running bond, parts resting on studs, counts never exceeded) and now prefers the right part role per region:
   slope bricks for gable roofs (when the inventory has enough), tiles for smooth tops, round bricks and cones for cylinders and trees.
4. **Checks**: inventory (`can_build`), structure (validator, one connected piece) and the features the concept asked for (`checks.py`).
   A build that fails is dropped.

    POST /api/builds/recipes   {"parts": [...], "use_ai": true, "match_color": true}  ->  builds (viewer format + recipe + concept info), hints
    GET  /api/builds/{id}/thumb.png          isometric thumbnail (cached)
    POST /api/sets/nearby      {"parts": [...], "match_color": true}  ->  "can_build" and "almost" with the missing parts
    GET  /api/dev/recipe-gallery             every recipe on random inventories (RECIPE_GALLERY=1); page: /recipe-gallery

**Recipes are JSON** (`backend/recipes/specs/*.json`, schema in `schema.py`): named parts, each a primitive shape with an size in studs given
as an expression, a colour role and a position (`at`, or `attach` on_top_of / in_front_of / behind / beside an earlier part), parameters
with ranges, silhouette presets, required features and symmetry. Expressions are a tiny safe language (`expr.py`). The primitives
(`primitives.py`): box, wall_with_window, tapered_box, slope_roof, arch, pillar, leg, cylinder, cone, wing, stairs, head, posts, checker,
cells, rails and wheel_axle (unsupported). Adding one is a decorated function. There are 17 recipes in 11 buildable categories (building,
tower, bridge, spacecraft, boat, aircraft, robot, animal, plant, castle, furniture) plus the legacy car, which is kept for the regression
test (`tests/test_schema_regression.py`: the five converted recipes give byte-identical regions and builds) and never offered.

**Part roles** (`backend/core/partroles.py`, data `backend/data/part_roles.json`) come from Rebrickable's `parts.csv` and
`part_categories.csv` (`python -m backend.core.fetch_part_roles`, committed): brick, plate, tile, slope, round, cone, arch, wheel, wheelbase ...
Slopes, round bricks, a cone and round tiles are in the grid with dimensions checked against the LDraw geometry (`tests/test_shapes.py`);
slopes carry their LDraw origin and the direction they fall to.

**On-demand recipes** (`generate.py`): when too few concepts could be built, Gemini may write a NEW recipe JSON for a concept. Code checks it
at five stages (schema, fill, inventory, structure+features, recognizability); on failure the specific errors go back for up to 3 repairs;
a recipe that still fails is dropped. Recognizability renders a thumbnail and asks Gemini vision "What is this LEGO model?" without the name or
category; every answer is logged (`recognizability.jsonl` in the library folder). Recipes that pass go to the **library**
(`RECIPE_LIBRARY_DIR`, `library.py`), unless their voxel shape has IoU > 0.6 with a recipe of the same category. Measured run:
`python -m backend.recipes.study --concepts 30` (pass rate per stage, `docs/study-runs/`). Side by side with the old output:
`python -m backend.recipes.examples` (`docs/examples/`).

Config: `RECIPE_CONCEPTS` (concepts asked, 12), `RECIPE_LIBRARY_DIR`, `RECIPE_GEN_PER_REQUEST` (new recipes Gemini may write per request, 1;
0 = never), `RECIPE_GEN_TIMEOUT`, `RECIPE_RECOGNISE`. `AI_FREEFORM_BUILDS` stays off by default and is not developed further.

**Does it look like its name?** `python -m backend.recipes.qa` renders every recipe on 20 random realistic inventories into grids of 4
(`docs/recipes/qa/`), asks a Gemini vision model "yes / partly / no" per picture and writes `docs/recipes/qa-report.md`.

**Unlock hints.** For recipes whose minimum is not met (parts, area, a long enough part) the response lists the smallest additions; recipes
that need wheels are never hinted. **Nearby sets** use a part to sets index built once from the Rebrickable CSVs: about 1 ms per inventory.

### Real LEGO sub-builds (LDraw OMR)

`POST /api/subbuilds {"parts": [...], "match_color": true}` lists real builds from the LDraw Official Model Repository that the pieces can build:
whole small models and the sub-models of bigger ones (a vehicle, a figure, a gate) of 6 to 80 parts, as `buildable` (the pieces cover them) and
`almost` (1-2 parts short, the missing parts listed, shown in a separate section after the buildable ones). The rule is the same as everywhere:
shape always matters; with "Consider color" on the demand is counted per (part, colour), off per part; a part the model draws in colour 16 takes
any colour. Renamed LDraw parts (`~Moved to 3139b`) are matched under their current number. Each result carries the author(s), licence and source of
the model (only CCAL / CC BY models are indexed), and `POST /api/subbuilds/build {"id": "sub-<set>-<block>"}` opens it in the 3D viewer.

    python -m backend.core.fetch_omr [--max-parts 1000]    # once: downloads the models into build_cache/omr/ (about 1,290 files, 110 MB, a minute)

The index (`backend/core/omr_index.py`, about 5,500 sub-builds, a pickle next to the models) and the nearby-sets index are built at start-up in a
background thread together with the LDraw library (`/api/health` shows `indexes`), so the first Suggest never waits for them; a request takes about
60 ms. `python -m backend.wheels_report` lists the wheel, tyre and wheel-holder parts in the LDraw library and whether Rebrickable knows them
(`docs/wheels-in-ldraw.md`).

### Stud detection

Studs look the same whatever the colour of the piece, so `backend/core/studs.py` finds them on every scan: CLAHE +
`HoughCircles` on a 1024 px copy, each circle validated (rim edge all round, same plastic outside as inside,
colour against the surface model), the photo's dominant stud radius learnt, and a stud kept only with a same-coloured
neighbour one pitch (1.67 diameters) away, or with very high confidence (a 1 x 1). Studs are grouped into pieces
(pitch, orientation, dimensions such as 2 x 4, box). They are used to:

- seed the colour method (`find_pieces_by_colour`): a stud group merges with the colour seeds it sits on, and a pale
  piece with no colour seed becomes a seed of its own;
- split boxes that hold two groups of different colour or different lattice, and merge boxes that are parts of one
  group (a pale piece found as its body plus one box per stud);
- give each box a stud score. Boxes without studs (tiles, slopes, side-on pieces) are never dropped for that, except
  small, weakly coloured, stud-less blobs (shadows, folds);
- hand `stud_count`, `pitch_px`, grid dims and stud radius to `Detection.stud_info` (not part of the API JSON) and to
  `shadow.analyse(..., stud_info=)`, which then uses the measured pitch as its ruler.

All thresholds are in one `StudConfig` block at the top of `studs.py`. `STUDS_ENABLED=0` switches it off; a failure in
any stud step is logged and the scan continues without studs. `DETECT_DEBUG_DIR=dir` saves an overlay per scan, and
`python -m backend.eval_studs` prints before/after counts and overlays for the photos in `tests/real_photos/`.

## Matching

Official sets are suggested **only if you can build them 100%** with your pieces.

From `suggest_builds.py`: coverage = pieces you have toward the set / total pieces in the set;
spares excluded; printed/pattern/mold variants merged; latest inventory version per set;
parts-pack themes skipped; sets with 5..`max_parts` parts. The server adds an inverted index
(part → sets) so only sets sharing a part with yours are scored (~0.2 s per request).

**Current rules:**

- **`min_coverage` defaults to 1.0**, set with the `MIN_COVERAGE` environment variable (0-1).
  A 99% set is excluded, a 100% set is included. Lower it later without touching code.
- **Ranking:** by `parts_you_have × coverage` (descending), tie-break fewest missing pieces, then
  newest year. At the default 100% every listed set is complete, so this is simply "biggest
  complete set first". If `MIN_COVERAGE` is lowered, near-complete big sets can outrank small
  complete ones, which is what that score is for. `coverage` stays the number the API reports.
- **Colour matching is off by default** (shape only), with a "Match colours exactly" toggle.
- **UI:** set cards show "Uses X of your Y pieces" plus Instructions and Rebrickable links; no
  coverage bar or missing-parts list. The API still returns `coverage`, `complete` and `missing`
  so they are available if `MIN_COVERAGE` is lowered.
- **Nothing buildable:** the UI shows "No official set can be built 100% with these pieces –
  here's what you can build instead" and switches to the AI ideas tab automatically (once).

### ID mapping (Brickognize → Rebrickable)

Brickognize returns BrickLink-style IDs. In order: exact match in `parts.csv`; strip a
trailing letter (`3001a` → `3001`); strip a BrickLink print suffix (`26603pb509` →
`26603`); append `a` (`41879` → `41879a`); Rebrickable API `bricklink_id` lookup (needs the
key); otherwise flagged unmapped. The print-suffix and append-`a` rules go beyond the
original spec; they came from real misses on the sample image.

## Known limitations

- Touching pieces merge into one detection; spread them apart.
- OpenCV detection runs two methods and combines them. The background-colour method assumes the
  border of the photo is the plain surface; it finds white and transparent pieces, but returns junk
  (e.g. one box for the whole photo) when wood or a table shows around a paper towel, so it is trusted
  only when its result is plausible. The colour method finds pieces from strongly coloured or dark
  seeds, copes with uneven light, embossed paper and close neighbours (a yellow brick and a brown
  round piece 20 px apart stay two pieces, a bar lying on a plate of a different colour stays separate),
  and splits boxes the first method merged. Its limits: pieces that overlap *and* share a colour merge,
  pale pieces (white, pale grey, transparent) may get a partial box or be missed when the first method
  is not trusted, and anything whose solid body touches the photo edge is ignored. For best results
  fill the frame with the sheet and leave a few centimetres between pieces. `DETECTOR=hybrid` adds
  Gemini detection on top (see "Piece detection").
- Background colour: the surface can be light or dark, neutral or coloured, as long as it is one colour
  all over (wood grain and towel texture are fine). Nothing assumes a white sheet: "is this ordinary
  surface?" is always answered against the surface the photo actually has. A *surface method* learns
  it (k-means in Lab colour; the largest cluster plus clusters of similar chroma and lightness, which
  absorbs grain) and treats whatever differs as a piece. It always runs. When over a third of the photo
  is strongly saturated or very dark (wood, black cloth) it is the only method; otherwise it fills gaps
  the colour methods leave (white/grey pieces on a coloured surface) but never adds a blob that merely
  surrounds pieces already found, and overlapping fragments of one piece are joined.
  `tests/background_sweep.py` generates the same nine pieces (white to black, with shadows, noise and a
  lighting gradient) on 62 uniform backgrounds and `tests/test_backgrounds.py` runs them all: 58 are
  perfect. Known limits (also listed in the test): a **black piece on a mid-brightness saturated
  surface** is about the colour of that surface's deep shadows and can be missed; a piece nearly the
  colour of the surface (green on bright green) is missed; a dark grey piece on a mid-dark saturated
  blue surface can split in two. Shadows are included in boxes and pieces that touch can merge.
- Brick vs plate: from straight above the two look alike, so recognition sometimes calls a brick a
  plate (or the reverse) with high confidence. The app does not try to correct this; the user can pick
  the other one under "Change part". (An experimental shadow-based check exists in
  `backend/core/shadow.py` but is not used by the app.)
- Minifigure parts may not match set inventories (e.g. BrickLink `970c00` hips+legs
  assemblies have no Rebrickable equivalent, so they stay unmapped).
- Random piles rarely complete a whole set, so with the 100% rule most mixed piles get no official
  set and go straight to AI ideas. Lower `MIN_COVERAGE` to see near-misses.
- Brickognize's endpoints are marked legacy; the client is isolated in
  `backend/core/recognize.py` so it is easy to swap.
- Brickognize reference thumbnails occasionally 403 upstream; the UI falls back to the Rebrickable
  image from `inventory_parts.csv` (per part and colour), then a neutral placeholder.
- Photos larger than 2400 px on the long side are downscaled before detection.

## Later

- **Video upload** (sample 1 frame per second, drop blurry frames, use the sharpest). Not in the MVP:
  `/api/scan` accepts photos only and answers 415 to anything else.
- A fallback model for AI responses blocked by safety filters.
- Per-set missing-parts view (the data is in the API; it was removed from the cards on purpose).

## Before going public

- **Per-user rate limiting.** `/api/ideas` calls a paid API and `/api/scan` calls Brickognize.
  Today the only guard is a cap of 3 concurrent AI calls for the whole server, which does not stop
  one visitor from using all the quota. Add per-IP or per-session limits (and an upload size/rate
  limit) before exposing the app.
- **HTTPS and CORS.** Plain http is fine on a LAN. For a public deployment serve over https and set
  `CORS_ORIGINS` to the real frontend origin.
- **Keys.** Keep `.env` out of git (it is gitignored), and rotate any key that has been pasted into
  a chat, log or screenshot.
- **Uploads** are stored in a temp dir and deleted after an hour; confirm that is acceptable for
  your privacy policy.

## CLIs

The original scripts still work: `python recognize_pieces.py lego_pieces.jpg --debug`
then `python suggest_builds.py parts_list.csv` (they now import from `backend/core/`).

## Detection: masks, split and merge rules

Every OpenCV detection is a mask (`backend/core/segment.py`). A connected region is one piece unless a split rule has evidence (seam, stud grids, colour, shape, size); neighbours that are really one piece are merged and re-checked; a split the evidence only just supports is flagged and settled by a Brickognize tie-break (at most `SEG_MAX_TIEBREAKS` per scan), otherwise shown as "check this". In the review screen, "Merge" and "Split" fix what is left; each correction is stored in `corrections/` (git-ignored) for the test set. All thresholds, with defaults, are in `docs/detection/metrics.md` and can be overridden with `SEG_<NAME>` environment variables. Before/after metrics and debug overlays: `python -m tests.detection.report`.

The detector engine is chosen with `DETECT_ENGINE`: `mask` (the default: the mask pipeline above, `analyse_masks`) or `legacy` (the earlier box logic). Merge / Split in the review screen work with both.


## Designed for your pile

`POST /api/designs/stream` (server-sent events) builds new models for the pile in `backend/designs/`:

1. one streamed Gemini call (`concepts.py`) returns 6-8 concepts as JSON objects: a name, a description and a coarse layer-by-layer voxel sketch (small, iconic shapes; the prompt says a typical pile is 6-20 pieces with 1-2 of each part). Three short examples from the OMR index inspire it;
2. each concept is solved locally as its line arrives (`solver.py`, OR-Tools CP-SAT, 2 s): the sketch is enlarged to use at least 70% of the pile's volume (never more than it has), then the real bricks, slopes, round parts and wheel sets (4600 holder + 2 rims + 2 tyres, only when the pile has them) are fitted with exact counts, no overlap, support and one connected piece;
3. `can_build`, structure and a duplicate check run at once and each passing design is streamed (`design` events);
4. with `DESIGN_MAX_CALLS=3` a batched contact-sheet look check runs in the background and removes designs that a child could not see as their name (`verdict` events). With `DESIGN_MAX_CALLS=3` there is also at most one repair call, only if fewer than 3 designs passed. The default is ONE Gemini call per request (no repair, no look check).

Benchmark: `GEMINI_MODEL=gemini-flash-lite-latest GEMINI_RPD=400 python -m backend.bench bench_photos/*.jpg --repeat 3 --designs-dir docs/designs` queues requests behind the per-minute limit (`BUILD_MAX_WAIT`) instead of being refused, and writes per-photo and combined contact sheets.

### Tests never call Gemini

`tests/conftest.py` replaces the SDK client with a tape client (`backend/core/gemini_tape.py`): answers are replayed from `tests/fixtures/gemini_tape/`, keyed by a hash of the model, prompts, schema and settings, and a request with no recording raises `NoRecording`. To record a real answer once: `GEMINI_LIVE=1 python -m pytest tests/test_designs_cost.py -k recorded` (needs `GEMINI_API_KEY`; hard cap of 5 real requests per run). Quota: a 429 marks that model exhausted until the daily reset (`gemini_budget`) and nothing else is tried; detection has one fallback model, used only on 503/504.
