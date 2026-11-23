"""Heat maps (also used for correlation matrices)."""
from __future__ import annotations

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection

from .base import Plot, Z_FILL, Z_TEXT, contrast_text

MAX_CELL_TEXTS = 4000


def heatmap(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    rows, cols = [str(r) for r in c.get("rows", [])], [str(v) for v in c.get("cols", [])]
    nr, nc = len(rows), len(cols)
    vals = np.array([[np.nan if v is None else v for v in row] for row in c.get("values", [])], dtype=float)
    vals = vals.reshape(nr, nc) if vals.size == nr * nc else np.full((nr, nc), np.nan)
    text = c.get("text") or []
    vmin, vmax = float(c.get("vmin", 0.0)), float(c.get("vmax", 1.0))
    if not vmax > vmin:
        vmin, vmax = vmin - 0.5, vmin + 0.5
    cmap = mcolors.LinearSegmentedColormap.from_list("aui", o.get("_colormap") or ["#440154", "#21918C", "#FDE725"])
    norm = mcolors.Normalize(vmin, vmax, clip=True)

    # one rectangle per non-empty cell (empty cells stay background)
    ii, jj = np.nonzero(np.isfinite(vals))
    face = cmap(norm(vals[ii, jj]))
    x0, y0 = jj - 0.5, ii - 0.5
    verts = np.stack([np.column_stack([x0, y0]), np.column_stack([x0 + 1, y0]),
                      np.column_stack([x0 + 1, y0 + 1]), np.column_stack([x0, y0 + 1])], axis=1)
    border = o.get("cell_border") or "white"
    bcol = {"white": p.panel, "gray": "#9A9A9A", "black": p.ink}.get(border)
    mesh = PolyCollection(verts, facecolors=face, edgecolors=bcol if bcol else "face",
                          linewidths=0.8 if bcol else 0.3, zorder=Z_FILL, joinstyle="miter")
    ax.add_collection(mesh, autolim=False)

    # axes: categories on both sides, first row on top, no frame
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    p.cat_axis(cols, which="x", pad=0.5)
    p.cat_axis(rows, which="y", pad=0.5, invert=True)
    ax.set_xlabel(p.labels.get("x", ""))
    ax.set_ylabel(p.labels.get("y", ""))

    # cell labels
    labels = []
    if text and nr * nc <= MAX_CELL_TEXTS:
        fixed = o.get("value_size")
        fs0 = float(fixed) if fixed else p.fonts["tick"] * 0.85
        for k, (i, j) in enumerate(zip(ii, jj)):
            t = text[i][j] if i < len(text) and j < len(text[i]) else ""
            if t:
                labels.append(ax.text(j, i, t, ha="center", va="center", fontsize=fs0, linespacing=0.95,
                                      color=contrast_text(face[k][:3]), zorder=Z_TEXT, clip_on=True))

    def fit():
        """Thin borders, shrink labels to the cells and hide what cannot fit."""
        cw, ch = ax.bbox.width / max(nc, 1), ax.bbox.height / max(nr, 1)
        if bcol:
            mesh.set_linewidth(float(np.clip(min(cw, ch) / 14 / p.px, 0.0, 1.0)))
        if labels and not o.get("value_size"):
            longest = max((max(len(s) for s in t.get_text().split("\n")) for t in labels), default=1)
            lines = max(t.get_text().count("\n") + 1 for t in labels)
            fs = min(p.fonts["tick"] * 0.9, cw / p.px / (0.62 * longest + 0.6), ch / p.px / (1.25 * lines))
            for t in labels:
                t.set_fontsize(max(fs, 1.0))
                t.set_visible(fs >= 3.5)
        tick = p.fonts["tick"]
        for axis, cell in ((ax.yaxis, ch), (ax.xaxis, cw)):
            need = tick * p.px * 1.12
            if cell < need:                          # more labels than room: shrink, then drop
                fs = cell / p.px / 1.12
                for t in axis.get_ticklabels():
                    t.set_fontsize(max(fs, 1.0))
                    t.set_visible(fs >= 4.0)
    p.hooks[0].append(fit)
    p.hooks[1].append(fit)

    # colour bar
    cax = None
    if o.get("show_colorbar", True):
        cax = p.fig.add_axes([0.9, 0.2, 0.02, 0.6])
        p.axes.append(cax)
        cb = p.fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.outline.set_visible(False)
        cax.tick_params(axis="y", length=2.5, width=0.6, color=p.axis_color, labelcolor=p.ink,
                        labelsize=p.fonts["tick"] * 0.9, pad=2)
        label = c.get("legend_label") or ""
        if label:
            cb.set_label(label, fontsize=p.fonts["legend"], color=p.ink, labelpad=5)
    square = bool(o.get("square"))

    def place(l, b, r, t):
        bar_w, gap = 0.13 * p.dpi, 0.14 * p.dpi
        mr = r - (bar_w + gap if cax is not None else 0)
        box = (l, b, mr, t)
        if square and nr and nc:
            w, h = mr - l, t - b
            cell = max(min(w / nc, h / nr), 1.0)
            cx, cy = (l + mr) / 2, (b + t) / 2
            box = (cx - cell * nc / 2, cy - cell * nr / 2, cx + cell * nc / 2, cy + cell * nr / 2)
        p.set_pos(ax, *box)
        if cax is not None:
            h = min(box[3] - box[1], 2.6 * p.dpi)
            mid = (box[1] + box[3]) / 2
            p.set_pos(cax, box[2] + gap, mid - h / 2, box[2] + gap + bar_w, mid + h / 2)
    p.placer = place


DRAWERS = {"heatmap": heatmap}
