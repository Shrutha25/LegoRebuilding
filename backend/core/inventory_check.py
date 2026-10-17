"""Does a build only use pieces the user owns? And, with "Consider color" off, which real colours should its pieces get?

    can_build(parts, inventory, match_color)   the one rule every build must pass before it is shown
    pool_inventory(inventory)                  the inventory with colours merged (what builds are PLANNED from when colour is ignored)
    recolour(parts, sections, inventory)       real colours for a colour-blind build, from the user's actual stock

The size and shape of a piece always has to match: a part is its part number (a 2x3 is never a 2x4). Only the colour is optional.

    match_color=True    counted by (part, colour):  3 red 2x4 + 1 blue 2x3 are two groups; a build needing 4 red 2x4 or a red 2x3 fails
    match_color=False   counted by part only, summed over colours: the same bricks are 3 x 2x4 + 1 x 2x3, in any colour

`parts` are anything with `.part_id` and `.colour` (Placements); `inventory` is an Inventory or an iterable of objects with
`.part_id`, `.colour` and `.qty`.
"""

from collections import Counter
from dataclasses import replace

POOLED = "Mixed"                      # the colour name pooled items carry while a build is planned


def _items(inventory):
    return inventory.parts() if hasattr(inventory, "parts") else list(inventory)


def _key(part_id, colour, match_color):
    part = str(part_id).strip().lower()
    return (part, str(colour).strip().lower()) if match_color else (part,)


def inventory_counts(inventory, match_color):
    counts = Counter()
    for it in _items(inventory):
        counts[_key(it.part_id, it.colour, match_color)] += int(it.qty)
    return counts


def required_counts(parts, match_color):
    return Counter(_key(p.part_id, p.colour, match_color) for p in parts)


def shortfall(parts, inventory, match_color):
    """{(part[, colour]): missing count} for everything the build needs more of than the inventory has (empty = the build is possible)."""
    have = inventory_counts(inventory, match_color)
    return {k: n - have.get(k, 0) for k, n in required_counts(parts, match_color).items() if n > have.get(k, 0)}


def can_build(build_parts, inventory, match_color=True):
    """True when the inventory holds every part the build needs, in the numbers it needs, under the colour rule above."""
    return not shortfall(build_parts, inventory, match_color)


# ---------------------------------------------------------------- colour-blind planning
def pool_inventory(inv):
    """One item per part number: quantities summed over colours, colour POOLED. The LDraw colour is the one the user has most of
    (only used for previews while planning; the final build is recoloured with `recolour`)."""
    from backend.core.buildmodel import Inventory
    groups = {}
    for it in inv.parts():
        groups.setdefault(it.part_id.lower(), []).append(it)
    items = []
    for group in groups.values():
        best = max(group, key=lambda i: (i.qty, i.colour.lower()))
        items.append(replace(best, colour=POOLED, qty=sum(i.qty for i in group)))
    return Inventory(items)


def _footprint(inv, part_id, rotation):
    item = next((i for i in inv.parts() if i.part_id.lower() == part_id.lower()), None)
    return item.dims.footprint(rotation) if item else (1, 1)


def _groups(parts, inv):
    """Indices that must share a colour: parts that mirror each other (left/right, front/back) about the build's centre, same part
    and height. Union-find over both mirror directions, so four wheels form one group."""
    n = len(parts)
    feet = [_footprint(inv, p.part_id, p.rotation) for p in parts]
    min_x = min(p.x for p in parts)
    max_x = max(p.x + feet[i][0] for i, p in enumerate(parts))
    min_z = min(p.z for p in parts)
    max_z = max(p.z + feet[i][1] for i, p in enumerate(parts))
    index = {}
    for i, p in enumerate(parts):
        index.setdefault((p.part_id.lower(), p.y, p.x, p.z, feet[i]), []).append(i)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, p in enumerate(parts):
        fx, fz = feet[i]
        for mx, mz in ((min_x + max_x - (p.x + fx), p.z), (p.x, min_z + max_z - (p.z + fz))):
            for j in index.get((p.part_id.lower(), p.y, mx, mz, (fx, fz)), ()):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def recolour(parts, sections, inv):
    """Real colours for a build planned from the pooled inventory. Returns a new list of Placements (same order, same steps).

    `sections` maps (x, z, y) -> section name (roof, chassis, deck ...); parts missing from it form one section. Rules, in order:
      * parts that mirror each other (wheels, pillars, wings) get the same colour when one colour has enough of that part;
      * the parts of a section share one colour when possible;
      * the largest section is coloured first and so gets the colour the user has most of.
    It never invents stock: a part takes a unit the user really has, and the result passes can_build(..., match_color=True)."""
    if not parts:
        return []
    stock = {}                                                        # part -> {colour name: qty left}
    for it in inv.parts():
        stock.setdefault(it.part_id.lower(), {})[it.colour] = stock.get(it.part_id.lower(), {}).get(it.colour, 0) + it.qty
    area = Counter()
    for it in inv.parts():
        area[it.colour] += it.qty * it.dims.long * it.dims.short
    rank = {c: r for r, (c, _) in enumerate(sorted(area.items(), key=lambda kv: (-kv[1], kv[0].lower())))}

    section_of = [sections.get((p.x, p.z, p.y), "") for p in parts]
    size = Counter()
    for i, p in enumerate(parts):
        fx, fz = _footprint(inv, p.part_id, p.rotation)
        size[section_of[i]] += fx * fz
    order = sorted(size, key=lambda s: (-size[s], s))
    groups = _groups(parts, inv)
    by_section = {s: [] for s in order}
    for g in groups:
        by_section[Counter(section_of[i] for i in g).most_common(1)[0][0]].append(g)

    result = [None] * len(parts)
    taken_by = {}                                                     # colour -> sections already using it

    def covers(colour, idxs):
        """How many of the parts `colour` can still supply."""
        need = Counter(parts[i].part_id.lower() for i in idxs)
        return sum(min(n, stock.get(part, {}).get(colour, 0)) for part, n in need.items())

    for s in order:
        members = [i for g in by_section[s] for i in g]
        colours = sorted({c for i in members for c, q in stock[parts[i].part_id.lower()].items() if q > 0} or rank,
                         key=lambda c: (-covers(c, members), len(taken_by.get(c, ())), rank.get(c, 99), c.lower()))
        main = colours[0] if colours else None
        taken_by.setdefault(main, set()).add(s)
        for g in sorted(by_section[s], key=lambda g: (-len(g), g[0])):
            need = Counter(parts[i].part_id.lower() for i in g)
            whole = [c for c in [main] + colours if all(stock.get(part, {}).get(c, 0) >= n for part, n in need.items())]
            if whole:                                                  # the section colour, else the first colour that fits the group
                for i in g:
                    stock[parts[i].part_id.lower()][whole[0]] -= 1
                    result[i] = replace(parts[i], colour=whole[0])
                continue
            for i in g:                                                # not enough of one colour: part by part
                options = stock.get(parts[i].part_id.lower(), {})
                if not any(q > 0 for q in options.values()):                # more than the user owns: left as is, can_build rejects it
                    result[i] = parts[i]
                    continue
                pick = max((c for c, q in options.items() if q > 0), key=lambda c: (c == main, options[c], -rank.get(c, 99), c))
                options[pick] -= 1
                result[i] = replace(parts[i], colour=pick)
    return result
