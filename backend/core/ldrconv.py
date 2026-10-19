"""LDraw text: build -> .ldr with `0 STEP` lines (and back), step numbering as three.js' LDrawLoader sees it, and steps for official
models that have none.

LDraw coordinates (what this module writes):
    1 stud = 20 LDU horizontally, 1 plate = 8 LDU vertically, and -Y is UP (the ground is y = 0, a part on layer y has its bottom
    at -8*y). A part's origin is the middle of its footprint at the TOP of its body, so a part whose bottom is on plate layer `y`
    and which is `h` plates tall has its origin at y_ldu = -8 * (y + h). The long side of a brick is along X at rotation 0.
    Rotation about the vertical axis by angle t:  x' = x cos t + z sin t ... see `_rotation`.
"""

import math
import re
from dataclasses import dataclass

import numpy as np

from backend.core.buildmodel import Build, Placement
from backend.core.partdims import PLATE_LDU, STUD_LDU

_TYPE1 = re.compile(r"^\s*1\s+(\S+)\s+((?:\S+\s+){12})(.+?)\s*$")


def _rotation(deg):
    """Matrix entries a..i (row-major) for a turn of `deg` about the vertical axis: local X -> (cos, 0, sin), local Z -> (-sin, 0, cos)."""
    c, s = int(round(math.cos(math.radians(deg)))), int(round(math.sin(math.radians(deg))))
    return [c, 0, -s, 0, 1, 0, s, 0, c]


def build_to_ldr(build, inv, author="Brick Finder (AI-suggested build)"):
    """The validated build as .ldr text; one `0 STEP` after each step's parts."""
    lines = [f"0 {build.title}", "0 Name: build.ldr", f"0 Author: {author}", "0 !LDRAW_ORG Unofficial_Model", ""]
    by_step = {}
    for p in build.parts:
        by_step.setdefault(p.step, []).append(p)
    for step in sorted(by_step):
        for p in sorted(by_step[step], key=lambda q: (q.y, q.z, q.x)):
            item = inv.get(p.part_id, p.colour)
            if item is None:
                raise ValueError(f"{p.part_id}/{p.colour} is not in the inventory")
            d = item.dims
            ox, oz = d.origin(p.rotation)
            x = p.x * STUD_LDU + ox
            z = p.z * STUD_LDU + oz
            y = -(p.y * PLATE_LDU + d.oy) if d.oy is not None else -(p.y + d.height) * PLATE_LDU
            m = " ".join(str(v) for v in _rotation(p.rotation))
            lines.append(f"1 {item.ld_colour} {_num(x)} {_num(y)} {_num(z)} {m} {item.ld_part}.dat")
        lines.append("0 STEP")
    return "\n".join(lines) + "\n"


def _num(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}"


def ldr_to_build(text, inv, title="build"):
    """Read a flat .ldr written by `build_to_ldr` back into a Build (for round-trip tests and for checking). Parts whose
    file/colour match several inventory items take the first."""
    by_key = {(i.ld_part.lower(), i.ld_colour): i for i in inv.parts()}
    parts, step = [], 1
    for raw in text.splitlines():
        if raw.strip().startswith("0 STEP"):
            step += 1
            continue
        m = _TYPE1.match(raw)
        if not m:
            continue
        t = m.group(2).split()
        x, y, z = (float(v) for v in t[:3])
        a, b, c, d, e, f, g, h, i = (float(v) for v in t[3:12])
        ld_part = m.group(3).lower().removesuffix(".dat")
        item = by_key.get((ld_part, int(m.group(1))))
        if item is None:
            raise ValueError(f"part {ld_part} colour {m.group(1)} is not in the inventory")
        rot = round(math.degrees(math.atan2(g, a))) % 360
        ox, oz = item.dims.origin(rot)
        parts.append(Placement(item.part_id, item.colour, round((x - ox) / STUD_LDU), round((z - oz) / STUD_LDU),
                               (round((-y - item.dims.oy) / PLATE_LDU) if item.dims.oy is not None else round(-y / PLATE_LDU) - item.dims.height), rot, step))
    return Build(title, "", parts)


# ---------------------------------------------------------------- steps as three.js numbers them
@dataclass
class PartUse:
    step: int            # 0-based, as LDrawLoader's userData.buildingStep
    ld_part: str         # LDraw part number, lower case, no extension
    colour: str          # LDraw colour code (an int as text) or a direct colour like 0x2FF0000
    file: str


def _files(text):
    """{lower-case name: lines} of an MPD, and the main name; a plain .ldr is its own main file."""
    text = text.replace("\r\n", "\n")
    files, main, name, buf = {}, None, None, []
    for line in text.split("\n"):
        if line.startswith("0 FILE "):
            if name is not None:
                files[name.lower()] = buf
            name, buf = line[7:].strip(), []
            main = main or name
        else:
            buf.append(line)
    if name is None:
        return {"model.ldr": text.split("\n")}, "model.ldr"
    files[name.lower()] = buf
    return files, main


def ldr_steps(text, is_part=None):
    """Parts per step, numbered exactly like LDrawLoader's computeBuildingSteps (depth-first, a `0 STEP` marks the NEXT sub-file
    line as starting a new step). Submodels (files embedded with `0 FILE`) are descended into; everything else is a part.
    Colour 16 inherits the parent's colour. Returns a list of PartUse in build order."""
    files, main = _files(text)
    out, step = [], 0

    def visit(name, parent_colour, depth):
        nonlocal step
        pending = False
        for line in files[name]:
            if line.lstrip().startswith("0 STEP"):                    # (0 ROTSTEP is only a view hint: LDrawLoader ignores it)
                pending = True
                continue
            m = _TYPE1.match(line)
            if not m:
                continue
            if pending:
                step += 1
                pending = False
            colour = parent_colour if m.group(1) == "16" else m.group(1)
            ref = m.group(3).strip().replace("\\", "/")
            key = ref.lower()
            if key in files and depth < 20:
                visit(key, colour, depth + 1)
            else:
                out.append(PartUse(step, ref.lower().removesuffix(".dat").split("/")[-1], colour, ref))

    visit(main.lower(), "16", 0)
    return out


def has_steps(text):
    return any(line.lstrip().startswith("0 STEP") for line in text.splitlines())


def header_lines(text, limit=12):
    """The leading comment/meta lines of the main model (title, author, licence)."""
    files, main = _files(text)
    out = []
    for line in files[main.lower()]:
        if line.startswith("1 "):
            break
        if line.strip():
            out.append(line)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- steps for models that have none
def flatten(text):
    """[(colour, 4x4 matrix, part file name)] of every part in an MPD with submodels expanded (transforms accumulated)."""
    files, main = _files(text)
    out = []

    def visit(name, matrix, parent_colour, depth):
        for line in files[name]:
            m = _TYPE1.match(line)
            if not m:
                continue
            t = [float(v) for v in m.group(2).split()]
            local = np.array([[t[3], t[4], t[5], t[0]], [t[6], t[7], t[8], t[1]], [t[9], t[10], t[11], t[2]], [0, 0, 0, 1]])
            colour = parent_colour if m.group(1) == "16" else m.group(1)
            ref = m.group(3).strip().replace("\\", "/")
            if ref.lower() in files and depth < 20:
                visit(ref.lower(), matrix @ local, colour, depth + 1)
            else:
                out.append((colour, matrix @ local, ref))

    visit(main.lower(), np.eye(4), "16", 0)
    return out


def add_steps(text, max_steps=60):
    """A flat .ldr with `0 STEP` lines for a model that has none: parts bottom-up (lowest first), 1-3 per step and a step's parts
    close together; at most `max_steps` steps (more parts per step in big models). Header lines (author, licence) are kept."""
    parts = flatten(text)
    if not parts:
        return text
    size = max(1, min(8, math.ceil(len(parts) / max_steps)))
    order = sorted(range(len(parts)), key=lambda i: (-parts[i][1][1, 3], parts[i][1][2, 3], parts[i][1][0, 3]))   # LDraw Y points down
    lines = [*header_lines(text), ""]
    remaining = list(order)
    while remaining:
        first = remaining.pop(0)
        group = [first]
        y0 = parts[first][1][1, 3]
        c0 = parts[first][1][[0, 2], 3]
        for i in list(remaining):
            if len(group) >= size:
                break
            same_level = abs(parts[i][1][1, 3] - y0) <= 24
            near = np.linalg.norm(parts[i][1][[0, 2], 3] - c0) <= 8 * STUD_LDU
            if same_level and near:
                group.append(i)
                remaining.remove(i)
        for i in group:
            colour, mtx, ref = parts[i]
            vals = [mtx[0, 3], mtx[1, 3], mtx[2, 3], *mtx[0, :3], *mtx[1, :3], *mtx[2, :3]]
            lines.append("1 " + colour + " " + " ".join(f"{v:.4f}".rstrip("0").rstrip(".") if v else "0" for v in vals) + " " + ref)
        lines.append("0 STEP")
    return "\n".join(lines) + "\n"
