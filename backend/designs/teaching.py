"""Fixed teaching examples for the concept prompt: simple, playful builds like a child's, where the shape plus the name tells the story.

Each has two or three telling features (a cab and a load, a beak and a tail, a roof and a door) and uses few, big bricks; nothing needs a wheel part, a
vehicle is a chassis and a cab. They are in the same layer-sketch format Gemini answers in, and `cells` is the number of cells each fills (a cell is one
stud wide, one stud deep and one brick tall) so the model sees how a sketch relates to a pile's volume.
"""

from backend.designs import sketch as sk

EXAMPLES = [  # the first three go in the prompt: the truck, a boat and a duck show the style (the rest are kept for tests and docs)
    {"name": "Delivery Truck", "type": "vehicle",
     "description": "a truck: a long chassis of two 2x6 bricks side by side, a two-brick-high cab across the front and a 2x2 load on the back",
     "palette": {"a": "Red", "b": "Blue", "c": "Yellow"},
     "sketch": {"layers": [["aaaaaa", "aaaaaa", "aaaaaa", "aaaaaa"], ["bb....", "bb.cc.", "bb.cc.", "bb...."], ["bb....", "bb....", "bb....", "bb...."]]}},
    {"name": "Sail Boat", "type": "vehicle",
     "description": "a boat: a long hull with a raised bow and stern and a mast carrying a sail",
     "palette": {"a": "Blue", "b": "White"},
     "sketch": {"layers": [["aaaaaa", "aaaaaa"], ["a.b..a", "a....a"], ["..bb..", "......"], ["..bb..", "......"]]}},
    {"name": "Pond Duck", "type": "animal",
     "description": "a duck: a body, a head on a neck, a beak at the front and a tail at the back",
     "palette": {"a": "Yellow", "b": "Orange"},
     "sketch": {"layers": [["aaaaa", "aaaa."], ["a.aab", "..aa."], ["...a.", "....."]]}},
    {"name": "Little House", "type": "building",
     "description": "a house: solid walls with a door gap at the front, a flat roof and a chimney",
     "palette": {"a": "Red", "b": "Brown", "c": "White"},
     "sketch": {"layers": [["aaaa", "aaaa", "aaaa"], ["aaaa", "aaaa", "a..a"], ["bbbb", "bbbb", "bbbb"], ["c...", "....", "...."]]}},
    {"name": "Little Plane", "type": "vehicle",
     "description": "a plane: a long body, two wings across the middle, a tail fin and a cockpit",
     "palette": {"a": "White", "b": "Red"},
     "sketch": {"layers": [["..bb..", "..bb..", "aaaaaa", "..bb..", "..bb.."], ["......", "......", "a...a.", "......", "......"]]}},
]
for _e in EXAMPLES:                                         # the sketches here must be valid and what the prompt says they are
    _e["cells"] = sk.parse(_e["sketch"], 12, 12, 8, min_cells=4, palette=_e["palette"]).filled


IN_PROMPT = 3


def prompt_lines():
    """The examples as JSON lines for the prompt, in the flat answer format: palette as "letter=Colour", layers at the top."""
    import json
    return "\n".join(json.dumps({"name": e["name"], "type": e["type"], "description": e["description"], "palette": [f"{k}={v}" for k, v in e["palette"].items()],
                                 "layers": e["sketch"]["layers"]}, separators=(",", ":")) + f"  ({e['cells']} cells)" for e in EXAMPLES[:IN_PROMPT])
