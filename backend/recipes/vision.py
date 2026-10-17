"""Does it look like its name? Contact sheets of isometric renders, graded by a Gemini vision model.

Used offline by qa.py (every recipe, many inventories) and, per request, ONLY for free-form fallback builds (AI_FREEFORM_BUILDS on).
Every call goes through `llm.complete_with_images`, i.e. gemini_budget with purpose "build".
"""

import io
import json
import logging

from PIL import Image, ImageDraw

log = logging.getLogger("lego.recipes.vision")

VERDICTS = ("yes", "partly", "no")
SYSTEM_PROMPT = (
    "You judge pictures of small LEGO models drawn as isometric renders (plain bricks, no stickers). Each picture has a label in its "
    "top-left corner. For every labelled picture decide whether it looks like the object named in the list: 'yes' (clearly recognisable), "
    "'partly' (a rough or abstract version) or 'no' (does not look like it). Be strict: a pile of blocks is not a car. Answer for every label."
)
SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {"type": "object", "properties": {
        "label": {"type": "string"}, "verdict": {"type": "string", "enum": list(VERDICTS)}, "note": {"type": "string"}},
        "required": ["label", "verdict", "note"], "additionalProperties": False}}},
    "required": ["answers"],
    "additionalProperties": False,
}


def contact_sheet(pngs, labels, cols=2, cell=(380, 310)):
    """One PNG with the renders in a grid, each with its label in the corner."""
    rows = (len(pngs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), "white")
    dr = ImageDraw.Draw(sheet)
    for n, (png, label) in enumerate(zip(pngs, labels)):
        im = Image.open(io.BytesIO(png)).convert("RGB").resize(cell)
        x, y = (n % cols) * cell[0], (n // cols) * cell[1]
        sheet.paste(im, (x, y))
        dr.rectangle([x, y, x + cell[0] - 1, y + cell[1] - 1], outline=(200, 200, 200))
        dr.rectangle([x + 4, y + 4, x + 74, y + 34], fill=(20, 20, 20))
        dr.text((x + 10, y + 12), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return buf.getvalue()


async def grade(llm, sheets, names):
    """sheets: [png bytes]; names: {label: object name}. Returns {label: (verdict, note)}; labels the model skipped are absent."""
    listing = "\n".join(f"{label}: should look like a {name}" for label, name in names.items())
    text = await llm.complete_with_images(SYSTEM_PROMPT, "Pictures to judge:\n" + listing, SCHEMA, sheets)
    data = json.loads(text)
    out = {}
    for a in data.get("answers", []) if isinstance(data, dict) else []:
        if isinstance(a, dict) and a.get("label") in names and a.get("verdict") in VERDICTS:
            out[a["label"]] = (a["verdict"], str(a.get("note") or "")[:200])
    return out


async def check_freeform(llm, png, title):
    """Verdict for one free-form build: 'yes' | 'partly' | 'no' ('yes' when the check itself fails: never block a build on QA)."""
    try:
        res = await grade(llm, [png], {"1": title})
        return res.get("1", ("yes", ""))[0]
    except Exception as e:                                              # noqa: BLE001
        log.info("free-form look check skipped (%s)", e)
        return "yes"


RECOGNISE_SYSTEM = ("You look at pictures of small LEGO models drawn as isometric renders (plain bricks, no stickers). Say what the model is "
                    "meant to be, as plainly as you can.")
RECOGNISE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}


async def recognise(llm, png):
    """What does a viewer say this model is? The name and category are NOT given. Returns a few words (raises if Gemini fails)."""
    text = await llm.complete_with_images(RECOGNISE_SYSTEM, "What is this LEGO model? Answer in a few words.", RECOGNISE_SCHEMA, [png])
    data = json.loads(text)
    answer = str(data.get("answer") or "") if isinstance(data, dict) else ""
    return " ".join(answer.split())[:120]
