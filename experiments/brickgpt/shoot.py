"""Screenshot builds in the dev gallery's real step viewer (finished state).
    shoot.py <out dir relative to repo root> <id> [<id> ...]      e.g. shoot.py docs/brickgpt/phase1 chair-strict-medium
Needs `npm run dev:web -- -p 3100` in frontend/ and a packed gallery (pack_gallery.py). Experiment venv."""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / sys.argv[1]
chrome = next(Path.home().joinpath("AppData/Local/ms-playwright").glob("chromium-*/chrome-win*/chrome.exe"))
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=str(chrome), args=["--use-gl=angle", "--enable-unsafe-swiftshader"])
    page = b.new_page(viewport={"width": 1100, "height": 800})
    for rid in sys.argv[2:]:
        page.goto(f"http://localhost:3100/brickgpt-gallery?open={rid}&done=1", wait_until="networkidle", timeout=180000)
        page.wait_for_function("() => window.__scene", timeout=120000)
        time.sleep(4)
        page.screenshot(path=str(OUT / f"{rid}.png"))
        print("saved", rid, flush=True)
    page.goto("http://localhost:3100/brickgpt-gallery", wait_until="networkidle", timeout=60000)
    page.screenshot(path=str(OUT / "gallery-table.png"), full_page=True)
    b.close()
