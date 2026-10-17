"""The recipe library: every generated recipe that passed all checks, kept as JSON with its category, silhouette and tags.

    lib = RecipeLibrary(config.RECIPE_LIBRARY_DIR)
    lib.add(spec, meta)          -> (True, "") or (False, reason)    rejects a near-duplicate of a recipe already in the category
    lib.recipes()                -> compiled Recipes (checked first when builds are served)
    lib.for_category("boat")

One file per recipe: <id>.json = {"spec": <recipe JSON>, "meta": {category, silhouette, tags, concept, created, answers ...}}.
A near-duplicate is a recipe whose voxel shape (its middle parameters, position ignored) has IoU > checks.IOU_MAX with a recipe of the same
category, built-in or already in the library.
"""

import json
import logging
import re
import time
from pathlib import Path

from backend.recipes import all_recipes
from backend.recipes.checks import IOU_MAX, iou, voxels_of_regions
from backend.recipes.schema import SpecError, compile_spec

log = logging.getLogger("lego.recipes.library")
_SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


def representative(recipe, params=None):
    """The recipe's shape as voxels for the middle of its parameters (or `params`), position ignored."""
    p = params or recipe.default_params()
    return voxels_of_regions(recipe.generate(p, None))


class RecipeLibrary:
    def __init__(self, directory):
        self.dir = Path(directory)
        self._cache = {}                      # file name -> (mtime, Recipe)

    # ------------------------------------------------------------ reading
    def entries(self):
        out = []
        for path in sorted(self.dir.glob("*.json")) if self.dir.exists() else []:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "spec" in data:
                    out.append(data)
            except (OSError, ValueError):
                log.warning("library file %s is unreadable; skipped", path.name)
        return out

    def recipes(self):
        found = []
        for path in sorted(self.dir.glob("*.json")) if self.dir.exists() else []:
            mtime = path.stat().st_mtime
            hit = self._cache.get(path.name)
            if hit is None or hit[0] != mtime:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    recipe = compile_spec(data["spec"])
                    recipe.library_meta = data.get("meta", {})
                    hit = (mtime, recipe)
                except (OSError, ValueError, KeyError, SpecError) as e:
                    log.warning("library recipe %s is invalid (%s); skipped", path.name, e)
                    hit = (mtime, None)
                self._cache[path.name] = hit
            if hit[1] is not None:
                found.append(hit[1])
        return found

    def for_category(self, category):
        return [r for r in self.recipes() if r.category == category]

    def ids(self):
        return {r.id for r in self.recipes()}

    # ------------------------------------------------------------ writing
    def duplicate_of(self, spec):
        """(id of the recipe in the same category with IoU > IOU_MAX, that IoU), or (None, best IoU seen)."""
        recipe = compile_spec(spec)
        mine = representative(recipe)
        best = (None, 0.0)
        for other in list(all_recipes()) + self.recipes():
            if other.category != recipe.category or other.id == recipe.id:
                continue
            v = iou(mine, representative(other))
            if v > best[1]:
                best = (other.id, v)
        return (best[0], best[1]) if best[1] > IOU_MAX else (None, best[1])

    def add(self, spec, meta=None):
        """Store a recipe that passed every check. Returns (True, "") or (False, why not)."""
        rid = (spec or {}).get("id", "")
        if not _SAFE.match(str(rid)):
            return False, f"bad recipe id {rid!r}"
        if rid in {r.id for r in all_recipes()}:
            return False, f"{rid!r} is the id of a built-in recipe"
        try:
            dup, v = self.duplicate_of(spec)
        except SpecError as e:
            return False, f"invalid recipe: {e}"
        if dup:
            return False, f"too similar to {dup!r} (shape IoU {v:.2f} > {IOU_MAX})"
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {"spec": spec, "meta": {**(meta or {}), "category": spec["category"], "tags": spec.get("tags", []),
                                       "silhouettes": sorted(spec.get("silhouettes", {})), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "iou_best": round(v, 3)}}
        tmp = self.dir / f".{rid}.tmp"
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / f"{rid}.json")
        return True, ""
