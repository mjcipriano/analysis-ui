"""Parts of a whole: pie / donut and frequency bar charts."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

from .base import Plot, Z_FILL, Z_TEXT, contrast_text, mix, rgba
from .categorical import edge_color


# ===================================================================== pie
def pie(p: Plot):
    o, ax = p.o, p.ax
    sl = p.frames["slices"]
    names = [str(v) for v in sl["label"]]
    values = sl["value"].to_numpy(dtype=float)
    pct = sl["pct"].to_numpy(dtype=float) if "pct" in sl else values / values.sum() * 100
    texts = [str(t) if t is not None else "" for t in sl["text"]] if "text" in sl else [""] * len(names)
    hole = float(np.clip(p.num("hole", 0.0), 0.0, 0.95))
    gap = p.num("slice_gap", 1.0) if len(names) > 1 else 0.0
    outside = (o.get("label_position") or "inside") == "outside"
    fs = p.fonts["tick"]
    ax.set_axis_off()
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    p.title_span = "rect"
    p.placer = lambda l, b, r, t: p.place_square(ax, l, b, r, t)

    frac = values / values.sum()
    start = p.num("start_angle", 90.0)
    ends = start - np.cumsum(frac) * 360.0           # clockwise
    begins = np.concatenate([[start], ends[:-1]])
    r_in = (1 + hole) / 2 if hole > 0 else 0.64
    out_labels = []                                   # (mid angle, text, colour)
    for name, a0, a1, f, pc, txt in zip(names, begins, ends, frac, pct, texts):
        col = p.color(name)
        ax.add_patch(Wedge((0, 0), 1.0, a1, a0, width=(1 - hole) if hole > 0 else None, facecolor=col,
                           edgecolor=p.bg if gap > 0 else "none", linewidth=gap, zorder=Z_FILL, joinstyle="round"))
        if not txt:
            continue
        mid = math.radians((a0 + a1) / 2)
        # an inside label must fit its slice: arc length at the label radius vs. text width
        tw, th = p.text_size(txt, fs)
        arc_px = f * 2 * math.pi * r_in * 0.5 * min(p.W, p.H) * p.dpi * 0.42
        ring_px = (1 - hole if hole > 0 else 0.7) * 0.5 * min(p.W, p.H) * p.dpi * 0.42
        fits = arc_px > max(th * 0.9, min(tw, th * 2.2) * 0.75) and tw < ring_px * (2.2 if hole == 0 else 1.05)
        if outside or not fits:
            out_labels.append((mid, txt, col))
        else:
            ax.text(r_in * math.cos(mid), r_in * math.sin(mid), txt, ha="center", va="center", fontsize=fs,
                    color=contrast_text(col), zorder=Z_TEXT, linespacing=1.1)
    if out_labels:
        _outside_labels(p, out_labels, fs)
    for name in names:
        p.lg_patch(name, p.color(name))
    named = (o.get("slice_labels") or "percent") in ("label", "label_percent", "label_value")
    p.legend_auto = "none" if named else "right"


def _outside_labels(p: Plot, labels: list, fs: float):
    """Labels around the pie with leader lines; labels on the same side are
    pushed apart vertically so they never overlap."""
    ax = p.ax
    R = 1.17
    arts = []
    for mid, txt, _col in labels:
        c, s = math.cos(mid), math.sin(mid)
        arts.append(ax.text(R * c, R * s, txt, ha="left" if c >= 0 else "right", va="center", fontsize=fs,
                            color=p.ink, clip_on=False, zorder=Z_TEXT, linespacing=1.1))
    leader = LineCollection([], colors=mix(p.ink, p.bg, 0.45), linewidths=0.7, zorder=Z_TEXT - 1, clip_on=False)
    ax.add_collection(leader, autolim=False)

    def arrange():
        unit = ax.bbox.height / 2.06                 # pixels per data unit
        segs = []
        for side in (1, -1):
            idx = [i for i, (mid, _, _) in enumerate(labels) if (math.cos(mid) >= 0) == (side > 0)]
            idx.sort(key=lambda i: -math.sin(labels[i][0]))
            if not idx:
                continue
            hs = [p.text_size(labels[i][1], fs)[1] / unit * 1.02 for i in idx]
            want = [R * math.sin(labels[i][0]) for i in idx]
            # resolve overlaps by merging neighbours into blocks centred on their wanted positions
            blocks = [[[k], want[k], hs[k]] for k in range(len(idx))]          # members, centre, height
            merged = True
            while merged:
                merged = False
                for b in range(len(blocks) - 1):
                    up, dn = blocks[b], blocks[b + 1]
                    if up[1] - up[2] / 2 < dn[1] + dn[2] / 2 - 1e-9:
                        members = up[0] + dn[0]
                        blocks[b:b + 2] = [[members, sum(want[k] for k in members) / len(members), up[2] + dn[2]]]
                        merged = True
                        break
            ys = [0.0] * len(idx)
            for members, centre, height in blocks:
                top = centre + height / 2
                for k in members:
                    ys[k] = top - hs[k] / 2
                    top -= hs[k]
            for i, y in zip(idx, ys):
                mid = labels[i][0]
                x = side * max(math.sqrt(max(R * R - y * y, 0.0)), 0.35 if abs(y) > R else 0.0)
                x = side * max(abs(x), abs(R * math.cos(mid)) * 0.6, 0.12)
                arts[i].set_position((x, y))
                segs.append([(math.cos(mid) * 1.0, math.sin(mid) * 1.0), (math.cos(mid) * 1.06, math.sin(mid) * 1.06),
                             (x - side * 0.03, y)])
        leader.set_segments(segs)
    arrange()
    p.hooks[1].append(arrange)


# =============================================================== count_bar
def count_bar(p: Plot):
    o = p.o
    df = p.frames["counts"]
    xs = [str(v) for v in p.computed.get("x_levels", [])]
    gs = [str(v) for v in p.computed.get("groups", [])]
    has_group = bool(p.computed.get("has_group"))
    position = p.computed.get("position") or "dodge"
    col_name = "pct" if p.computed.get("value") == "pct" else "count"
    alpha, outline, olw = p.num("fill_alpha", 1.0), o.get("outline") or "match", p.num("outline_width", 1.0)
    bw = p.num("bar_width", 0.75)
    p.ref_lines(categorical=True)
    xi = {x: i for i, x in enumerate(xs)}
    gi = {g: j for j, g in enumerate(gs)}
    X = df["x"].astype(str).to_numpy()
    G = df["group"].astype(str).to_numpy()
    V = np.nan_to_num(df[col_name].to_numpy(dtype=float))
    T = [str(t) if t is not None else "" for t in df["text"]] if "text" in df else [""] * len(df)
    ng = len(gs) if has_group else 1
    stacked = position in ("stack", "fill") and has_group
    pos, base, width = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    acc: dict = {}
    order = np.lexsort((np.array([gi.get(g, 0) for g in G]), np.array([xi.get(x, 0) for x in X])))
    for k in order:
        i = xi.get(X[k], 0)
        if stacked:
            pos[k], width[k], base[k] = i, bw, acc.get(i, 0.0)
            acc[i] = base[k] + V[k]
        elif has_group:
            pos[k], width[k] = i + (gi.get(G[k], 0) - (ng - 1) / 2) * bw / ng, bw / ng
        else:
            pos[k], width[k] = i, bw
    colors = [p.color(g) for g in G]
    lw = olw if outline != "none" else 0.0
    p.bars(pos, base + V, width, base=base, color=[rgba(c, alpha) for c in colors],
           edgecolor=[edge_color(outline, c, p.ink) for c in colors] if lw > 0 else "none",
           linewidth=lw, zorder=Z_FILL)
    has_text = any(T)
    if has_text:
        fs = p.fonts["tick"] * 0.9
        total = float((base + V).max()) or 1.0
        for k in range(len(df)):
            if not T[k]:
                continue
            if stacked:
                if V[k] < total * 0.055:             # segment too thin for a label
                    continue
                p.ax.text(*p.xy(pos[k], base[k] + V[k] / 2), T[k], ha="center", va="center", fontsize=fs,
                          color=contrast_text(colors[k]) if alpha > 0.5 else p.ink, zorder=Z_TEXT)
            else:
                p.ax.annotate(T[k], p.xy(pos[k], V[k]), xytext=(3, 0) if p.flip else (0, 2),
                              textcoords="offset points", ha="left" if p.flip else "center",
                              va="center" if p.flip else "bottom", fontsize=fs, color=p.ink, zorder=Z_TEXT)
    for g in gs:
        col = p.color(g)
        p.lg_patch(g, col, alpha, ec=edge_color(outline, col, p.ink) if lw > 0 else "none", lw=min(lw, 1.5))
    if not has_group:
        p.legend_auto = "none"
    p.cat_axis(xs)
    if position == "fill" and has_group:
        p.value_axis(zero=True, lo=0.0, hi=100.0)
    else:
        p.value_axis(zero=True, pad=0.1 if (has_text and not stacked) else 0.05)
    p.axis_titles()


DRAWERS = {"pie": pie, "count_bar": count_bar}
