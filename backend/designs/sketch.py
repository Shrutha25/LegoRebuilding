"""A coarse layer-by-layer voxel sketch: what a concept looks like before any brick is chosen.

    {"layers": [["####", "####"], ["##..", "##.."]]}

`layers` goes from the bottom up; one layer is one brick tall; each layer is a list of rows (the depth, z, back to front) of characters (the width, x,
left to right): "#" is filled (any colour), "." or " " is empty, and a letter is filled in the colour the concept's palette gives that letter ("a": "Yellow").
A Sketch is the set of filled cells (x, z, layer) and, for the cells with a colour wish, `colours` {(x, z, layer): colour name}. `parse` checks size limits and returns a
Sketch or a plain-language reason (used in the repair prompt). `render` writes a sketch back as text (the OMR examples shown to Gemini).
"""

from dataclasses import dataclass, field

FILLED = set("#X1@*")                 # filled, any colour
EMPTY = set(". _-0")
LETTERS = set("abcdefghijklmnopqrstuvwxyz")        # filled, in the palette colour of that letter (x is a letter here, # is the "any colour" mark)


@dataclass(frozen=True)
class Sketch:
    cells: frozenset                 # (x, z, layer)
    width: int
    depth: int
    layers: int
    colours: dict = field(default_factory=dict)       # (x, z, layer) -> colour name, only for cells with a wish

    @property
    def filled(self):
        return len(self.cells)

    def plate_cells(self):
        """The same cells at plate resolution: (x, z, plate) with 3 plates per layer."""
        return frozenset((x, z, 3 * k + q) for x, z, k in self.cells for q in range(3))

    def text(self):
        return render(self)


class SketchError(ValueError):
    pass


def parse(obj, max_width=12, max_depth=12, max_layers=8, min_cells=4, palette=None):
    """Sketch from the JSON of a concept (a dict with "layers", or the list itself). `palette` {letter: colour name} gives the letters a colour; a letter
    that is not in it is filled in any colour. Raises SketchError with the reason."""
    layers = obj.get("layers") if isinstance(obj, dict) else obj
    palette = {str(k).lower(): v for k, v in (palette or {}).items()}
    if not isinstance(layers, list) or not layers:
        raise SketchError("the sketch has no layers (a list of layers, bottom first, each a list of row strings)")
    cells, wish = set(), {}
    for k, layer in enumerate(layers):
        rows = layer if isinstance(layer, list) else str(layer).split("/") if isinstance(layer, str) else None
        if rows is None or not all(isinstance(r, str) for r in rows):
            raise SketchError(f"layer {k + 1} must be a list of row strings")
        for z, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch in FILLED:
                    cells.add((x, z, k))
                elif ch.lower() in LETTERS:
                    cells.add((x, z, k))
                    if palette.get(ch.lower()):
                        wish[(x, z, k)] = palette[ch.lower()]
                elif ch not in EMPTY:
                    raise SketchError(f"layer {k + 1}, row {z + 1}: character {ch!r} is not allowed; use '#' for filled, '.' for empty or a letter of the palette")
    if len(cells) < min_cells:
        raise SketchError(f"the sketch fills only {len(cells)} cells; it needs at least {min_cells}")
    mx, mz, mk = min(c[0] for c in cells), min(c[1] for c in cells), min(c[2] for c in cells)
    wish = {(x - mx, z - mz, k - mk): c for (x, z, k), c in wish.items()}
    cells = frozenset((x - mx, z - mz, k - mk) for x, z, k in cells)
    w, d, h = max(c[0] for c in cells) + 1, max(c[1] for c in cells) + 1, max(c[2] for c in cells) + 1
    if w > max_width or d > max_depth or h > max_layers:
        raise SketchError(f"the sketch is {w} wide, {d} deep and {h} layers tall; the limits are {max_width} x {max_depth} x {max_layers}")
    return Sketch(cells, w, d, h, wish)


def render(sk):
    out = []
    for k in range(sk.layers):
        out.append([("".join("#" if (x, z, k) in sk.cells else "." for x in range(sk.width))) for z in range(sk.depth)])
    return {"layers": out}


def scale(sk_, s, t):
    """The sketch enlarged by whole numbers: every cell becomes s x s studs and t brick layers."""
    cells = frozenset((x * s + i, z * s + j, k * t + q) for x, z, k in sk_.cells for i in range(s) for j in range(s) for q in range(t))
    colours = {(x * s + i, z * s + j, k * t + q): c for (x, z, k), c in sk_.colours.items() for i in range(s) for j in range(s) for q in range(t)}
    return Sketch(cells, sk_.width * s, sk_.depth * s, sk_.layers * t, colours)


def fit_scale(sk_, pile_cells, max_width=12, max_depth=12, max_layers=8, target=0.85):
    """(sketch, s, t): the sketch enlarged (s horizontally and t in layers, 1 to 3; the same on both when it fits, which keeps the silhouette) so that it
    fills about `target` of the pile's volume without exceeding the volume or the size limits. A sketch already bigger than the pile is left alone (the
    solver then fills what the pile allows)."""
    options = []
    for s in (1, 2, 3):
        for t in (1, 2, 3):
            if sk_.width * s > max_width or sk_.depth * s > max_depth or sk_.layers * t > max_layers:
                continue
            filled = sk_.filled * s * s * t
            if filled > pile_cells and (s, t) != (1, 1):
                continue
            options.append((abs(filled - target * pile_cells) + 0.4 * abs(s - t) + 0.1 * (s + t), s, t))
    _, s, t = min(options) if options else (0, 1, 1)
    return (scale(sk_, s, t) if (s, t) != (1, 1) else sk_), s, t


def outline_iou(sketch, filled):
    """How well a fill keeps the sketch's silhouette: the lowest IoU of the three outlines (top view x/z, front view x/plate, side view z/plate) of the
    sketch's plate cells and `filled`, a set of (x, z, plate) cells. 1.0 = the same outlines."""
    want = sketch.plate_cells()
    worst = 1.0
    for axes in ((0, 1), (0, 2), (1, 2)):
        a = {tuple(c[i] for i in axes) for c in want}
        b = {tuple(c[i] for i in axes) for c in filled}
        worst = min(worst, len(a & b) / len(a | b) if a | b else 1.0)
    return worst
