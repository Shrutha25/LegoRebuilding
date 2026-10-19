"""What a pile can build, as numbers and plain words: the exact parts, their roles, the total volume and the largest model that makes sense.

    aff = summarize(inventory)           # an Inventory of the usable parts (aibuild.build_inventory)
    aff.prompt(match_color=True)         # the text Gemini sees (no coordinates, nothing to place)
    aff.limits                           # sketch size limits: width / depth in studs, height in brick layers

Volume is counted in "cells": one stud wide, one stud deep, one brick (3 plates) tall. A 2 x 4 brick is 8 cells, a 2 x 4 plate 8/3, so a pile's volume
says how much model it can fill; a design should use at least MIN_USE of it where it can and never more than the pile has.
"""

import math
from dataclasses import dataclass, field

MIN_USE = 0.70
FILL_MIN, FILL_MAX = 0.75, 1.0       # a sketch should fill this share of the pile's volume (the prompt says so, the code accepts a little slack)

ROLE_WORDS = {
    ("brick", "rect"): "brick",
    ("plate", "rect"): "plate",
    ("tile", "rect"): "tile",
    ("brick", "round"): "round brick",
    ("plate", "round"): "round plate",
    ("tile", "round"): "round tile",
    ("brick", "cone"): "cone",
    ("brick", "slope"): "slope",
    ("wheelbase", "rect"): "wheel holder",
    ("tyre", "wheel"): "tyre",
    ("rim", "wheel"): "wheel rim",
}

LEGEND = "brick = 3 plates tall, studs on top; plate = 1 plate; tile = smooth top (nothing sits on it); slope = nothing sits on it."
WHEEL_NOTE = ("Wheel set (1 holder + 2 rims + 2 tyres) = one unit stood on the ground, 4 studs long, 2 wide, 2 bricks tall, the body on the holder's studs: sketch a "
              "vehicle whose lowest two layers are filled over a 4 x 2 block per axle; draw no wheels.")


def role_class(d):
    """A coarse class of a part for matching a pile to example models: its kind or shape and whether it is big (6+ studs of footprint) or small."""
    if d is None:
        return "other"
    kind = d.shape if d.shape in ("round", "cone", "slope") else d.kind
    return f"{kind}/{'big' if d.long * d.short >= 6 else 'small'}"


def wheel_sets(inv):
    """How many complete wheel sets (holder, 2 rims, 2 tyres) the pile has."""
    n = {k: sum(i.qty for i in inv.parts() if i.dims.kind == k) for k in ("wheelbase", "rim", "tyre")}
    return min(n["wheelbase"], n["rim"] // 2, n["tyre"] // 2)


def cells_of(item):
    d = item.dims
    return item.qty * d.long * d.short * d.height / 3


@dataclass
class Affordance:
    rows: list = field(default_factory=list)          # [{part, name, colour, qty, size, role, cells}]
    pieces: int = 0
    cells: float = 0.0                                # total volume
    area: int = 0                                     # total studs of footprint
    longest: int = 0
    limits: dict = field(default_factory=dict)        # {"width": n, "depth": n, "layers": n}
    colours: list = field(default_factory=list)
    wheels: int = 0                                   # complete wheel sets
    lengths: list = field(default_factory=list)       # the piece lengths, longest first
    mix: dict = field(default_factory=dict)           # {role class: pieces}

    @property
    def budget(self):
        """(whole cells of the pile, fewest cells a sketch should fill, most it may fill)."""
        whole = int(self.cells)
        return whole, math.ceil(FILL_MIN * whole), whole

    @property
    def ground(self):
        """(most cells the bottom layer can have, longest straight row it can have): the pile's footprint area and its two longest pieces end to end."""
        return self.area, sum(self.lengths[:2])

    def prompt(self, match_color=True):
        lines = []
        for r in self.rows:
            colour = f" in {r['colour']}" if r["colour"] else ""
            lines.append(f"- {r['qty']} x {r['size']} {r['role']}{colour}")
        head = (f"The pile has {self.pieces} pieces of {len(self.rows)} kinds ({LEGEND} "
                f"{'Colours matter: a colour is used only as often as listed' if match_color else 'Colours are only a wish: any colour may stand in for another'}; "
                f"colours: {', '.join(self.colours)}):")
        whole, low, high = self.budget
        area, row = self.ground
        tail = (f"Volume budget: {whole} cells (a cell = 1 stud x 1 stud x 1 brick; a 2x4 brick = 8, a 2x4 plate 2.7). Each sketch must fill {low} to {high} cells. "
                f"At most {self.limits['width']} wide, {self.limits['depth']} deep, {self.limits['layers']} layers. Bottom layer: at most {area} cells and no row "
                f"longer than {row} studs (your longest pieces end to end); each layer above no bigger than the one below unless supported.")
        note = f"\n{WHEEL_NOTE}" if self.wheels else ""
        return head + "\n" + "\n".join(lines) + "\n" + tail + note


def summarize(inv):
    rows, colours = [], {}
    for it in sorted(inv.parts(), key=lambda i: (-cells_of(i), i.part_id, i.colour)):
        d = it.dims
        rows.append({"part": it.part_id, "name": it.name, "colour": it.colour, "qty": it.qty, "size": f"{d.long}x{d.short}",
                     "role": ROLE_WORDS.get((d.kind, d.shape), f"{d.kind}"), "cells": round(cells_of(it), 2)})
        colours[it.colour] = colours.get(it.colour, 0) + it.qty
    pieces = sum(r["qty"] for r in rows)
    cells = sum(cells_of(i) for i in inv.parts())
    area = sum(i.qty * i.dims.long * i.dims.short for i in inv.parts())
    longest = max((i.dims.long for i in inv.parts()), default=0)
    mix = {}
    for i in inv.parts():
        mix[role_class(i.dims)] = mix.get(role_class(i.dims), 0) + i.qty
    lengths = sorted((i.dims.long for i in inv.parts() for _ in range(min(i.qty, 2))), reverse=True)
    side = int(min(12, max(4, math.ceil(math.sqrt(max(area, 1))))))
    layers = int(min(8, max(3, round(math.sqrt(max(cells, 1))))))
    return Affordance(rows, pieces, cells, area, longest, {"width": side, "depth": side, "layers": layers},
                      sorted(colours, key=lambda c: -colours[c]), wheel_sets(inv), lengths, mix)
