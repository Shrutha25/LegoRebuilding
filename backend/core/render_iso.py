"""A small isometric renderer for builds: every part is a box with stud dots. Used for thumbnails and for the offline "does it look
like its name?" check. Pure PIL, no LDraw library needed.

The camera looks from the +x / +z side and above. A part's top face is drawn in its colour, the face towards +z a little darker and
the face towards +x darker still, studs as small cylinders. Parts are drawn far to near by a topological sort of "is behind".
"""

import io
import math

from PIL import Image, ImageDraw

SS = 2                           # supersampling factor
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
PLATE = 0.4                      # a plate is 8 LDU high, a stud cell 20 LDU wide
STUD_R = 0.3
STUD_H = 0.17


def _rgb(hexv):
    h = hexv.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _boxes(build, inv):
    out = []
    for p in build.parts:
        item = inv.get(p.part_id, p.colour)
        if item is None:
            continue
        d = item.dims
        fx, fz = d.footprint(p.rotation)
        out.append((p, item, p.x, p.z, p.y, fx, fz, d.height))
    return out


def _order(boxes):
    """Far to near. A is before B when A is wholly behind B along x or z, or wholly under B."""
    n = len(boxes)
    before = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            _, _, ax, az, ay, afx, afz, ah = boxes[i]
            _, _, bx, bz, by, bfx, bfz, bh = boxes[j]
            if ax + afx <= bx or az + afz <= bz or ay + ah <= by:
                before[j].add(i)
    done, order = set(), []
    remaining = set(range(n))
    while remaining:
        ready = sorted(i for i in remaining if not (before[i] & remaining))
        if not ready:                                   # a cycle (cannot happen for a grid model): fall back to a depth key
            ready = [min(remaining, key=lambda i: (boxes[i][2] + boxes[i][3] + boxes[i][4], i))]
        i = ready[0]
        order.append(i)
        remaining.discard(i)
    return order


def render_png(build, inv, hex_of, size=(420, 340), background=(255, 255, 255)):
    """PNG bytes of the build. `hex_of(ld_colour_code) -> "#RRGGBB"`."""
    boxes = _boxes(build, inv)
    if not boxes:
        img = Image.new("RGB", size, background)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()

    def proj(x, z, y):
        return ((x - z) * C30, (x + z) * S30 - y * PLATE)

    pts = []
    for _, _, x, z, y, fx, fz, h in boxes:
        for cx in (x, x + fx):
            for cz in (z, z + fz):
                for cy in (y, y + h):
                    pts.append(proj(cx, cz, cy + (STUD_H / PLATE if cy == y + h else 0)))
    minx, maxx = min(p[0] for p in pts), max(p[0] for p in pts)
    miny, maxy = min(p[1] for p in pts), max(p[1] for p in pts)
    W, H = size[0] * SS, size[1] * SS
    margin = 0.1
    scale = min(W * (1 - 2 * margin) / max(1e-6, maxx - minx), H * (1 - 2 * margin) / max(1e-6, maxy - miny))
    ox = (W - (maxx - minx) * scale) / 2 - minx * scale
    oy = (H - (maxy - miny) * scale) / 2 - miny * scale

    def P(x, z, y):
        a, b = proj(x, z, y)
        return (a * scale + ox, b * scale + oy)

    img = Image.new("RGB", (W, H), background)
    dr = ImageDraw.Draw(img)
    line = max(1, int(scale * 0.03))
    for i in _order(boxes):
        p, item, x, z, y, fx, fz, h = boxes[i]
        base = _rgb(hex_of(item.ld_colour))
        top = y + h
        edge = _shade(base, 0.45)
        dr.polygon([P(x, z + fz, top), P(x + fx, z + fz, top), P(x + fx, z + fz, y), P(x, z + fz, y)], fill=_shade(base, 0.82), outline=edge)
        dr.polygon([P(x + fx, z, top), P(x + fx, z + fz, top), P(x + fx, z + fz, y), P(x + fx, z, y)], fill=_shade(base, 0.66), outline=edge)
        dr.polygon([P(x, z, top), P(x + fx, z, top), P(x + fx, z + fz, top), P(x, z + fz, top)], fill=base, outline=edge, width=line)
        if item.dims.studs_top:
            for a in range(fx):
                for b in range(fz):
                    cx, cz = x + a + 0.5, z + b + 0.5
                    r = STUD_R
                    for k, (col, lift) in enumerate(((_shade(base, 0.78), 0.0), (_shade(base, 1.08), STUD_H / PLATE))):
                        c = P(cx, cz, top + lift)
                        rx, ry = r * C30 * 1.15 * scale, r * S30 * 1.55 * scale
                        if k == 0:
                            dr.ellipse([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry], fill=col)
                        else:
                            base_c = P(cx, cz, top)
                            dr.rectangle([c[0] - rx, c[1], c[0] + rx, base_c[1]], fill=_shade(base, 0.9))
                            dr.ellipse([c[0] - rx, c[1] - ry, c[0] + rx, c[1] + ry], fill=col, outline=_shade(base, 0.6))
    img = img.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
