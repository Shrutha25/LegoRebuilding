"""Recipe-based builds: a recipe describes a shape in regions, the filler turns the regions into parts the user owns.

    from backend.recipes import REGISTRY, all_recipes, get
    from backend.recipes.planner import plan_candidates

Importing this package registers the recipes described by the JSON files in backend/recipes/specs/.
"""

from backend.recipes.base import Recipe

REGISTRY = {}


def register(recipe):
    if recipe.id in REGISTRY:
        raise ValueError(f"duplicate recipe id {recipe.id}")
    REGISTRY[recipe.id] = recipe
    return recipe


def get(recipe_id):
    return REGISTRY.get(recipe_id)


def all_recipes():
    return list(REGISTRY.values())


# one JSON spec per concept (backend/recipes/specs/*.json), compiled by schema.py and registered here
from backend.recipes.schema import load_specs  # noqa: E402

for _recipe in load_specs():
    register(_recipe)
