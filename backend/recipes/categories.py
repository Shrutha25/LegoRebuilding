"""Words: what a category is called, which feature a phrase talks about, and whether a model's description matches its category.

Used by the concept stage (features a concept asks for), by the feature check, and by the recognisability check (what Gemini says the
model is vs the category it was designed as). The vocabularies grow with the recipes: every recipe's `nouns` count for its category.
"""

import re

_WORD = re.compile(r"[a-z]+")

# words that mean a category, besides the nouns of its recipes
EXTRA_WORDS = {
    "building": ("house", "home", "building", "cottage", "cabin", "hut", "villa", "shop", "store", "inn", "mansion", "barn", "farmhouse", "chalet", "tower block", "apartment"),
    "tower": ("tower", "lighthouse", "turret", "spire", "minaret", "column", "pillar", "chimney", "skyscraper"),
    "bridge": ("bridge", "viaduct", "arch", "walkway", "overpass", "pier"),
    "spacecraft": ("rocket", "spaceship", "spacecraft", "shuttle", "starship", "space", "missile", "lander", "ufo"),
    "boat": ("boat", "ship", "sailboat", "yacht", "ferry", "canoe", "barge", "vessel", "sail"),
    "aircraft": ("plane", "airplane", "aeroplane", "jet", "aircraft", "glider", "airliner", "fighter", "helicopter"),
    "robot": ("robot", "android", "droid", "mech", "cyborg", "figure", "toy"),
    "animal": ("dog", "duck", "bird", "animal", "cat", "horse", "fox", "penguin", "goose", "swan", "pony", "puppy", "creature", "pet", "bear", "wolf", "owl"),
    "plant": ("tree", "bush", "plant", "pine", "fir", "shrub", "hedge", "flower", "oak", "christmas tree", "conifer"),
    "castle": ("castle", "gate", "gatehouse", "wall", "fort", "fortress", "fortification", "rampart", "battlement", "archway"),
    "furniture": ("chair", "table", "bench", "stool", "throne", "desk", "sofa", "couch", "furniture", "seat", "armchair", "shelf"),
    "vehicle": ("car", "truck", "van", "bus", "vehicle", "buggy", "jeep", "wagon", "cart", "tractor", "train"),
}

# a phrase that mentions one of these asks for that feature (tag); a recipe can promise the tag in its "features"
FEATURE_WORDS = {
    "wheels": ("wheel", "wheels", "tyre", "tyres", "tire", "tires", "axle"),
    "roof": ("roof", "gable", "thatch", "eaves", "roofed"),
    "door": ("door", "doorway", "entrance", "gate door"),
    "windows": ("window", "windows"),
    "arch": ("arch", "archway", "arched", "span"),
    "legs": ("leg", "legs", "feet", "paws"),
    "head": ("head", "snout", "face"),
    "tail": ("tail", "tailfin", "rudder"),
    "wings": ("wing", "wings"),
    "fins": ("fin", "fins"),
    "mast": ("mast", "sail", "sails"),
    "sail": ("sail", "sails"),
    "cabin": ("cabin", "cockpit", "cab", "wheelhouse"),
    "chimney": ("chimney", "smokestack"),
    "crenellations": ("crenellation", "crenellations", "battlement", "battlements", "parapet"),
    "gate": ("gate", "portcullis", "gateway"),
    "towers": ("turret", "turrets", "towers"),
    "back": ("backrest", "back rest"),
    "arms": ("arm", "arms", "armrest", "armrests"),
    "neck": ("neck",),
    "ears": ("ear", "ears"),
    "antenna": ("antenna", "aerial"),
    "legs-table": (),
    "seat": ("seat",),
    "engines": ("engine", "engines", "thruster", "thrusters"),
    "cockpit": ("cockpit", "canopy"),
    "nose": ("nose cone", "nosecone"),
    "stripe": ("stripe", "stripes"),
    "bands": ("band", "bands", "banded"),
    "deck": ("deck",),
    "pillars": ("pillar", "pillars", "piers"),
    "rails": ("rail", "rails", "railing", "railings", "handrail"),
    "trunk": ("trunk",),
    "crown": ("crown", "canopy", "foliage", "leaves"),
    "cone": ("cone", "conical"),
    "parapet": ("parapet",),
}
# a tag is promised by a recipe under this other name too
TAG_ALIASES = {"mast": ("sail",), "sail": ("mast",), "cockpit": ("cabin",), "cabin": ("cockpit",), "ears": ("head",)}


def words(text):
    return _WORD.findall((text or "").lower())


def _has(text, phrase):
    t = " " + " ".join(words(text)) + " "
    return (" " + " ".join(words(phrase)) + " ") in t


def feature_tags(text):
    """The feature tags a phrase like "4 wheels and a sloped roof" talks about."""
    return {tag for tag, ws in FEATURE_WORDS.items() if any(_has(text, w) for w in ws)}


def category_vocab(category, recipes=()):
    out = set(EXTRA_WORDS.get(category, ()))
    out.add(category)
    for r in recipes:
        if r.category == category:
            out.update(r.nouns)
    return out


def match_category(answer, category, recipes=()):
    """Does a free-text answer ("a small sailing boat") describe this category? -> (matches, best matching category or None)."""
    best, best_hits = None, 0
    cats = {r.category for r in recipes} | set(EXTRA_WORDS)
    for cat in cats:
        hits = sum(1 for w in category_vocab(cat, recipes) if _has(answer, w))
        if hits > best_hits:
            best, best_hits = cat, hits
    own = sum(1 for w in category_vocab(category, recipes) if _has(answer, w))
    return own > 0 and own >= best_hits, best


def is_colour_category_name(name, category, colours, recipes=()):
    """True for a name that says nothing but a colour and what it is ("Red House", "Blue Tower", "Cottage"): not evocative."""
    ws = set(words(name))
    colour_words = {w for c in colours for w in words(c)} | {"red", "blue", "green", "yellow", "black", "white", "grey", "gray", "orange", "tan",
                                                            "brown", "lime", "pink", "purple", "dark", "light", "bright", "mixed"}
    generic = {w for v in category_vocab(category, recipes) for w in words(v)} | {"the", "a", "an", "little", "small", "big", "mini", "tiny",
                                                                                     "model", "lego", "brick", "build", "my", "simple"}
    return not ws or ws <= (colour_words | generic)
