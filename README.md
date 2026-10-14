# LEGO "What can I build?"

Upload a photo of LEGO pieces spread on a plain background. The app detects each piece
(OpenCV), identifies it (Brickognize), and suggests official sets ranked by how much of
each set you already own (Rebrickable data).

Status: backend (scan, match, parts search) and mobile-first frontend (upload, review, results) done.
Next: AI ideas, then video, then polish.

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
| `ANTHROPIC_API_KEY`, `LLM_MODEL` | AI build ideas (not wired up yet). |
| `BRICKOGNIZE_URL` | Recognition endpoint (no key needed). |
| `RB_DATA_DIR`, `UPLOAD_DIR` | Data and temp-upload locations. |
| `CORS_ORIGINS` | Comma-separated allowed origins (default `http://localhost:3000`). |

### Docker

```bash
docker compose build backend
docker compose run --rm backend python -m backend.core.fetch_data   # fills the rb_data volume once
docker compose up backend
docker compose run --rm --no-deps backend python -m pytest -q       # tests, in the container
```

`rb_data` is a named volume, so the download and index cache survive rebuilds.
Local and Docker both use Python 3.14.

### Tests

```bash
python -m pytest -q
```

Tests use small fixture CSVs; the one image test uses `lego_pieces.jpg` (expects 42 pieces).

## API (curl examples)

```bash
# Health: data_loaded=false while the index is still loading
curl localhost:8000/api/health

# Scan a photo -> {"job_id": "..."}   (jpg/png/webp, <= 50 MB; video returns 501 for now)
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
#   -> {"sets": [{set_num, name, year, image, total_parts, parts_you_have, coverage,
#                 missing_count, missing:[{part, color, qty, image}],
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

Not implemented yet: `POST /api/ideas`, video upload.

## Matching

Unchanged from `suggest_builds.py`: coverage = pieces you have toward the set / total
pieces in the set; spares excluded; printed/pattern/mold variants merged; latest inventory
version per set; parts-pack themes skipped; sets with 5..`max_parts` parts and at least 20%
coverage. The server only adds an inverted index (part → sets) so just sets sharing a part
with yours are scored (~0.2 s per request).

### ID mapping (Brickognize → Rebrickable)

Brickognize returns BrickLink-style IDs. In order: exact match in `parts.csv`; strip a
trailing letter (`3001a` → `3001`); strip a BrickLink print suffix (`26603pb509` →
`26603`); append `a` (`41879` → `41879a`); Rebrickable API `bricklink_id` lookup (needs the
key); otherwise flagged unmapped. The print-suffix and append-`a` rules go beyond the
original spec; they came from real misses on the sample image.

## Known limitations

- Touching pieces merge into one detection; spread them apart.
- Minifigure parts may not match set inventories (e.g. BrickLink `970c00` hips+legs
  assemblies have no Rebrickable equivalent, so they stay unmapped).
- Random piles rarely complete a whole set; expect partial coverage.
- Brickognize's endpoints are marked legacy; the client is isolated in
  `backend/core/recognize.py` so it is easy to swap.
- Brickognize reference thumbnails occasionally 403 upstream; the UI falls back to an empty tile.
- Tiny sets (5-10 parts) reach 100% coverage easily and can rank above bigger partial matches.
- Photos larger than 2400 px on the long side are downscaled before detection.

## CLIs

The original scripts still work: `python recognize_pieces.py lego_pieces.jpg --debug`
then `python suggest_builds.py parts_list.csv` (they now import from `backend/core/`).
