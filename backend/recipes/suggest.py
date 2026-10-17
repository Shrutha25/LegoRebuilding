"""From an inventory to builds, concept first (nothing here places a brick except the recipes' filler):

    1. feasible categories   which kinds of models the pieces can support (concepts.feasible_categories)
    2. concepts (Stage A)    10-15 evocative concepts from Gemini (name, category, silhouette, bounding box, features), or the recipes' own
                             silhouettes and names without Gemini
    3. construction (Stage B) each concept goes to a recipe of its category (library recipes first, then the built-in ones); the recipe's
                             parameters follow the concept's silhouette and bounding box; the filler tiles it from the inventory
    4. checks                inventory (can_build), structure and the features the concept asked for; a build that fails is dropped
    5. on demand             when too few concepts could be built, Gemini may write a NEW recipe for one (generate.py), which joins the library

Without Gemini (no key, budget refused, error, nonsense answer) step 2 uses the fallback concepts and everything else works unchanged.
"""

import logging
from dataclasses import dataclass, field

from backend import config, timing
from backend.recipes import all_recipes
from backend.recipes.checks import feature_problems, structural_problems
from backend.recipes.concepts import choose_recipes, feasible_categories, get_concepts, nearest_params
from backend.recipes.facts import assign_colours, facts_of
from backend.recipes.planner import Candidate, make_candidate, unlock_hints

log = logging.getLogger("lego.recipes.suggest")


@dataclass
class Shown:
    candidate: Candidate
    name: str
    description: str
    from_ai: bool = False
    concept: object = None


@dataclass
class Suggestion:
    shown: list = field(default_factory=list)       # of Shown
    unlock: list = field(default_factory=list)
    ai_used: bool = False
    ai_note: str = ""
    candidate_count: int = 0
    concepts: list = field(default_factory=list)
    unbuilt: list = field(default_factory=list)     # (concept name, why not)
    generated: list = field(default_factory=list)   # names of concepts a new recipe was written for


def build_concept(concept, recipes, inv, facts, tries=3):
    """(Candidate, None) for the first recipe/parameters that build the concept and pass the structure and feature checks, or (None, why)."""
    why = "no recipe of this category fits the inventory"
    for recipe in choose_recipes(concept, recipes):
        for params in nearest_params(recipe, concept, facts, tries):
            for rot in range(2):
                reasons = []
                try:
                    with timing.stage("solver", recipe=recipe.id):
                        cand = make_candidate(recipe, params, inv, assign_colours(recipe, facts, rot), facts, reasons)
                except ValueError as e:
                    cand, reasons = None, [("fill", str(e))]
                if cand is None:
                    why = reasons[0][1] if reasons else why
                    continue
                with timing.stage("checks", recipe=recipe.id):
                    problems = structural_problems(cand.build, inv) + feature_problems(recipe, cand.regions, concept.features)
                if not problems:
                    return cand, None
                why = problems[0]
    return None, why


async def suggest(inv, llm=None, seed=0, show=5, timeout=40.0, use_ai=True, match_color=True, library=None, hex_of=None):
    facts = facts_of(inv)
    recipes = list(all_recipes())
    if library is not None:
        known = {r.id for r in recipes}
        recipes = [r for r in library.recipes() if r.id not in known] + recipes          # the library is checked first
    out = Suggestion(unlock=unlock_hints(inv))
    feasible = feasible_categories(facts, recipes)
    if not feasible:
        return out
    ai = llm if use_ai else None
    with timing.stage("concepts"):
        concepts, out.ai_used, out.ai_note = await get_concepts(ai, inv, facts, feasible, match_color, config.RECIPE_CONCEPTS, timeout, recipes, seed)
    out.concepts = concepts
    taken, built = set(), []
    for c in concepts:
        cand, why = build_concept(c, recipes, inv, facts)
        if cand is None:
            out.unbuilt.append((c.name, why))
            continue
        key = (cand.recipe_id, tuple(sorted(cand.params.items())))
        if key in taken:                                                         # the same recipe and shape twice would be the same build
            out.unbuilt.append((c.name, "same build as another concept"))
            continue
        taken.add(key)
        built.append(Shown(cand, c.name, c.description, c.source == "gemini", c))
    out.candidate_count = len(built)
    gen_left = config.RECIPE_GEN_PER_REQUEST if ai is not None and library is not None and hex_of is not None else 0
    if len(built) < show and gen_left:
        from backend.recipes.generate import generate_recipe
        failed = [c for c in concepts if c.name in {n for n, _ in out.unbuilt} and c.source == "gemini"]
        for c in failed[:gen_left]:
            with timing.stage("generate", concept=c.name):
                res = await generate_recipe(ai, c, inv, facts, library, hex_of, config.RECIPE_RECOGNISE, timeout=config.RECIPE_GEN_TIMEOUT, recipes=recipes)
            if res.ok and res.candidate is not None:
                out.generated.append(c.name)
                built.append(Shown(res.candidate, c.name, c.description, True, c))
    out.shown = built[:show]
    return out
