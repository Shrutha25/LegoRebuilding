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

## API (curl examples)

```bash
# Health: data_loaded=false while the index is still loading
curl localhost:8000/api/health

# Scan a photo -> {"job_id": "..."}   (jpg/png/webp, <= 50 MB; anything else, incl. video, gets a 415)
curl -F "file=@lego_pieces.jpg;type=image/jpeg" localhost:8000/api/scan

# Poll the job: status is queued|detecting|recognizing|done|error
curl localhost:8000/api/scan/JOB_ID
#   pieces[]: {index, box:[x,y,w,h], crop_url, part_id, name, color, score,
#              alternatives[], needs_review, error}   (+ width/height of the image)

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
- Piece detection first assumes the plain sheet reaches the edges of the photo. If that gives an
  implausible result (e.g. one box covering most of the photo because a wooden table shows around a
  paper towel), it retries with a colour-based method that copes with uneven light and embossed
  paper. That fallback finds coloured and dark pieces but can miss pale ones (white, transparent),
  and ignores anything touching the photo edge. For best results fill the frame with the sheet.
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
