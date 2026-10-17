"""Checks on finished builds: structure, features, and shape similarity (voxel IoU).

    structural_problems(build, inv)          every part is held by studs and the build is one piece
    feature_problems(recipe, tags, report)   the features the concept asked for are really in the build
    voxels_of_build / voxels_of_regions      a build or a recipe's regions as a set of (x, z, y) unit cells, any position
    iou(a, b)                                intersection over union of two voxel sets, aligned at their minimum corner
"""

import re

from backend.core import buildmodel as bm
from backend.recipes.categories import TAG_ALIASES, feature_tags

IOU_MAX = 0.6                  # two builds / recipes whose voxel IoU is higher than this count as the same shape


# ---------------------------------------------------------------- voxels
def normalise(vox):
    """Translate a voxel set so its minimum corner is the origin (position does not matter)."""
    if not vox:
        return frozenset()
    mx, mz, my = min(v[0] for v in vox), min(v[1] for v in vox), min(v[2] for v in vox)
    return frozenset((x - mx, z - mz, y - my) for x, z, y in vox)


def voxels_of_regions(regions):
    """Every stud cell of every layer of the regions as (x, z, plate layer)."""
    out = set()
    for r in regions:
        h = r.layer_height
        for n, cells in enumerate(r.layers):
            y0 = r.y + n * h
            out.update((x, z, y) for x, z in cells for y in range(y0, y0 + h))
    return normalise(out)


def voxels_of_build(build, inv):
    out = set()
    for p in build.parts:
        d = inv.get(p.part_id, p.colour).dims
        fx, fz = d.footprint(p.rotation)
        out.update((p.x + i, p.z + j, p.y + k) for i in range(fx) for j in range(fz) for k in range(d.height))
    return normalise(out)


def iou(a, b):
    a, b = normalise(a), normalise(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def best_iou(a, others):
    """(highest IoU of `a` against any of `others`, index of that one); (0.0, -1) for no others."""
    best = (0.0, -1)
    for i, o in enumerate(others):
        v = iou(a, o)
        if v > best[0]:
            best = (v, i)
    return best


# ---------------------------------------------------------------- structure
def structural_problems(build, inv, max_parts=None):
    """Plain-language problems: parts hanging in the air, overlaps, parts missing from the inventory, more than one piece."""
    rep = bm.validate(build, inv, max_parts)
    out = [i.message for i in rep.errors]
    if rep.components > 1:
        out.append(f"the build falls into {rep.components} separate pieces; every brick must connect to the rest through studs")
    return out


# ---------------------------------------------------------------- features
def _part_present(pattern, report):
    """A part (or a repeated part like "band-{k}") with at least one kept region that has cells filled."""
    base = re.sub(r"\{.*?\}", "", pattern)
    for name, role, kept, coverage in report:
        if (name == pattern or name.startswith(base)) and kept and coverage > 0:
            return True
    return False


def wanted_tags(recipe, concept_features):
    """(tags the concept asks for that the recipe can show, tags it asks for that the recipe cannot offer)."""
    have = set(recipe.features) | {a for t in recipe.features for a in TAG_ALIASES.get(t, ())}
    asked = set()
    for f in concept_features or ():
        asked |= feature_tags(f)
    return {t for t in asked if t in have}, {t for t in asked if t not in have and t in WANT_CHECKED}


# feature tags we refuse to fake: asking for these from a recipe that does not have them fails the concept
WANT_CHECKED = {"wheels", "wings", "legs", "roof", "arch", "head", "tail", "mast", "sail", "gate", "door", "windows", "chimney", "towers"}


def _mandatory(recipe, parts):
    """A feature is always expected when one of its parts is a required part of the recipe; features made only of optional details
    (a chimney, an antenna), and of parts that exist only for some parameters, are expected only when the concept asks for them."""
    specs = {p["name"]: p for p in (recipe.spec or {}).get("parts", [])}
    return any(specs.get(n, {}).get("required", True) is not False and "when" not in specs.get(n, {}) for n in parts)


def feature_problems(recipe, report, concept_features=()):
    """Plain-language problems with the features of a finished build. The recipe's own required features and the ones the concept asked
    for must be in the build (each feature names parts that must have cells in it). `report` is Candidate.regions:
    [(part name, role, kept, coverage)]."""
    out = []
    tags, missing = wanted_tags(recipe, concept_features)
    for t in sorted(missing):
        out.append(f"the concept asks for {t!r} but the {recipe.id} recipe cannot build that")
    spec_features = (recipe.spec or {}).get("features", {})
    asked = set(tags) | {a for t in tags for a in TAG_ALIASES.get(t, ())}
    for tag, parts in sorted(spec_features.items()):
        if (tag in asked or _mandatory(recipe, parts)) and not any(_part_present(p, report) for p in parts):
            out.append(f"feature {tag!r} is missing: none of its parts ({', '.join(parts)}) ended up in the build")
    return out
