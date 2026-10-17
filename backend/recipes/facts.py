"""What a recipe may want to know about an inventory, computed once: colours ranked by size, the longest part, which footprints exist."""

from dataclasses import dataclass, field

DARK_WORDS = ("black", "dark", "brown")
NEUTRAL_WORDS = ("gray", "grey", "white", "black", "tan", "dark bluish", "light bluish")


def usable(item):
    """Bricks and plates with studs on top; tiles and anything else are not used by recipes."""
    return item.dims.kind in ("brick", "plate") and item.dims.studs_top


def colour_tone(name):
    n = name.lower()
    return {"dark": any(w in n for w in DARK_WORDS), "neutral": any(w in n for w in NEUTRAL_WORDS)}


@dataclass
class InvFacts:
    items: list                                   # usable InvItems
    colours: list                                 # colour names, biggest first (total studs of bricks/plates)
    colour_area: dict                             # colour name -> total studs
    total_parts: int
    total_area: int
    longest: int                                  # longest side of any brick/plate
    footprints: set = field(default_factory=set)  # {(kind, long, short)}
    slope_area: int = 0                           # studs of slope bricks (they need a roof to go on)
    slope_depths: set = field(default_factory=set)   # depths (studs, the side the surface falls along) of the slopes owned
    tile_area: int = 0
    round_parts: int = 0
    cones: int = 0
    wheel_sets: int = 0                           # complete wheel sets the grid can place: none yet, so vehicles are never offered

    def has(self, kind, long, short):
        return (kind, max(long, short), min(long, short)) in self.footprints

    def has_any(self, long, short):
        return self.has("brick", long, short) or self.has("plate", long, short)

    def has_1x1(self):
        return self.has_any(1, 1)

    def has_thin(self):
        """Anything one stud wide (1 x n): needed for thin walls and fins."""
        return any(f[2] == 1 for f in self.footprints)


def facts_of(inv):
    items = [i for i in inv.parts() if usable(i)]
    area, total, total_area, longest, fp = {}, 0, 0, 0, set()
    for it in items:
        d = it.dims
        a = d.long * d.short * it.qty
        area[it.colour] = area.get(it.colour, 0) + a
        total += it.qty
        total_area += a
        longest = max(longest, d.long)
        fp.add((d.kind, d.long, d.short))
    colours = sorted(area, key=lambda c: (-area[c], c.lower()))
    slope_area = tile_area = rounds = cones = 0
    depths = set()
    for it in inv.parts():
        d = it.dims
        if d.shape == "slope":
            slope_area += it.qty * d.long * d.short
            depths.add(d.footprint(0)[1])
        elif d.kind == "tile":
            tile_area += it.qty * d.long * d.short
        elif d.shape == "round":
            rounds += it.qty
        elif d.shape == "cone":
            cones += it.qty
    return InvFacts(items, colours, area, total, total_area, longest, fp, slope_area, depths, tile_area, rounds, cones)


def assign_colours(recipe, facts, rotation=0):
    """A {role: colour name} for the recipe. `rotation` shifts which big colour becomes the main one (alternative assignments).
    Roles that want 'dark' get the darkest colour, 'neutral' a grey/white/black-ish one; the rest take the biggest colours in turn."""
    if not facts.colours:
        return {}
    cols = list(facts.colours)
    if rotation and len(cols) > 1:
        r = rotation % len(cols)
        cols = cols[r:] + cols[:r]
    out, used = {}, []

    def pick(pred):
        for c in cols:
            if pred(c):
                return c
        return None

    for role in recipe.roles:
        c = None
        first = out.get(recipe.roles[0].name)
        if role.want == "dark":
            c = pick(lambda c: colour_tone(c)["dark"]) or pick(lambda c: c != first and colour_tone(c)["neutral"])                 or next((c for c in reversed(cols) if c != first), None)          # no dark colour: at least not the body colour
        elif role.want == "neutral":
            c = pick(lambda c: colour_tone(c)["neutral"] and c not in used[:1]) or next((c for c in reversed(cols) if c != first), None)
        if c is None:                       # main colours: the biggest colour not used yet, else the biggest
            c = pick(lambda c: c not in used and not (role.want in ("major", "second") and colour_tone(c)["dark"] and len(cols) > 2)) \
                or pick(lambda c: c not in used) or cols[0]
        out[role.name] = c
        if role.want in ("major", "second"):
            used.append(c)
    return out
