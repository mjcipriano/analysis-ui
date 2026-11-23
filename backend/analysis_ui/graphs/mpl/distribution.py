"""Distribution graphs: histogram, density (overlay / ridgeline), QQ, ECDF."""
from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection

from .base import Plot, Z_BAND, Z_FILL, Z_LINE, Z_POINT, darken, luminance, point_style, rgba


def _groups(p: Plot, frame, col: str = "group") -> list[str]:
    return [str(g) for g in (p.computed.get("groups") or frame[col].astype(str).unique())]


def _rug(p: Plot, ax, values, color, height: float = 0.03):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return
    segs = np.stack([np.column_stack([v, np.zeros(len(v))]), np.column_stack([v, np.full(len(v), height)])], axis=1)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=0.7, alpha=0.6,
                                     transform=ax.get_xaxis_transform(), zorder=Z_LINE + 1), autolim=False)


# =============================================================== histogram
def histogram(p: Plot):
    o = p.o
    bins = p.frames["bins"]
    groups = _groups(p, bins)
    curve, main = p.frames.get("curve"), p.frames.get("main")
    style = o.get("hist_style") or "overlay"
    if len(groups) < 2:
        style = "overlay"
    alpha = p.num("fill_alpha", 0.6)
    outline = o.get("outline") or "white"
    lw_curve = p.num("line_width", 1.5)
    bgrp = bins["group"].astype(str).to_numpy()
    left, right = bins["left"].to_numpy(dtype=float), bins["right"].to_numpy(dtype=float)
    height = bins["height"].to_numpy(dtype=float)
    nb = max(int((bgrp == groups[0]).sum()), 1)
    edge_lw = 0.0 if outline == "none" else float(np.clip(p.W * 72 / nb / 9, 0.25, 0.8))
    if style == "dodge":
        edge_lw = min(edge_lw, float(np.clip(p.W * 72 / nb / len(groups) / 6, 0.15, 0.8)))

    def edge(col):
        return {"white": p.panel, "match": col, "black": p.ink}.get(outline, "none")

    # one panel per group for "facet", otherwise everything in the main axes
    panels = {g: p.ax for g in groups}
    if style == "facet":
        for g in groups[1:]:
            panels[g] = p.new_axes(sharex=p.ax, sharey=p.ax)
    for ax in dict.fromkeys(panels.values()):
        p.ref_lines(ax=ax)

    y_top = 0.0
    stack_base = np.zeros(nb)
    for j, g in enumerate(groups):
        ax, col = panels[g], p.color(g)
        m = bgrp == g
        l, r, h = left[m], right[m], np.nan_to_num(height[m])
        w = r - l
        kw = dict(color=rgba(col, alpha), edgecolor=edge(col) if edge_lw else "none", linewidth=edge_lw,
                  zorder=Z_FILL + j * 0.01, align="edge")
        if style == "stack" and len(h) == nb:
            ax.bar(l, h, width=w, bottom=stack_base, **kw)
            stack_base = stack_base + h
            y_top = max(y_top, float(stack_base.max(initial=0)))
        elif style == "dodge":
            ax.bar(l + w * j / len(groups), h, width=w / len(groups), **kw)
            y_top = max(y_top, float(h.max(initial=0)))
        else:
            ax.bar(l, h, width=w, **kw)
            y_top = max(y_top, float(h.max(initial=0)))
        if curve is not None and len(curve):
            cv = curve[curve["group"].astype(str) == g]
            kinds = list(dict.fromkeys(cv["kind"].astype(str)))
            for kind in kinds:
                c = cv[cv["kind"].astype(str) == kind]
                dashed = kind == "normal" and "kde" in kinds
                ax.plot(c["x"].to_numpy(dtype=float), c["y"].to_numpy(dtype=float),
                        color=darken(col, 0.25) if alpha > 0.3 else col, linewidth=lw_curve,
                        linestyle=(0, (4, 2)) if dashed else "-", zorder=Z_LINE + 1, solid_capstyle="round")
                y_top = max(y_top, float(np.nanmax(c["y"].to_numpy(dtype=float), initial=0)))
        if o.get("show_rug") and main is not None:
            _rug(p, ax, main.loc[main["group"].astype(str) == g, "value"], col)
        if style == "facet":
            ax.text(0.985, 0.93, g, transform=ax.transAxes, ha="right", va="top", fontsize=p.fonts["tick"],
                    fontweight="bold", color=darken(col, 0.25) if luminance(p.panel) > 0.5 else col, zorder=Z_LINE + 2)
        p.lg_patch(g, col, alpha, ec=edge(col) if outline in ("match", "black") else "none",
                   lw=0.8 if outline in ("match", "black") else 0.0)

    lo_x, hi_x = float(np.nanmin(left)), float(np.nanmax(right))
    p.num_axis("x", "x", data=(lo_x, hi_x), pad=0.03)
    p.num_axis("y", "y", data=(0.0, y_top or 1.0), zero=True, pad=0.06)
    if style == "facet":
        _facet_layout(p, [panels[g] for g in groups])
        p.legend_auto = "none"
    else:
        p.axis_titles()


def _facet_layout(p: Plot, axes: list):
    """Stack the panels vertically (first group on top) with shared x."""
    n = len(axes)
    gap = 0.75 * p.fonts["base"] * p.px
    xl, yl = p.labels.get("x", ""), p.labels.get("y", "")
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(xl)
    mid = axes[n // 2]                               # even n: the title is shifted up half a gap
    mid.set_ylabel(yl)
    for ax in axes[1:]:                              # inherit scale/limits, keep our tick styling
        ax.xaxis.offsetText.set_visible(False)

    def place(l, b, r, t):
        h = max((t - b - gap * (n - 1)) / n, 4)
        for i, ax in enumerate(axes):
            top = t - i * (h + gap)
            p.set_pos(ax, l, top - h, r, top)
        if n % 2 == 0:                               # centre the y title on the whole stack
            mid.yaxis.label.set_y(1 + gap / (2 * h))
    p.placer = place


# ================================================================= density
def density(p: Plot):
    o, ax = p.o, p.ax
    dens = p.frames["density"]
    groups = _groups(p, dens)
    main = p.frames.get("main")
    alpha, lw = p.num("fill_alpha", 0.35), p.num("line_width", 1.5)
    dgrp = dens["group"].astype(str).to_numpy()
    X, Y = dens["x"].to_numpy(dtype=float), np.nan_to_num(dens["y"].to_numpy(dtype=float))
    ridge = (p.computed.get("style") or o.get("density_style")) == "ridgeline"
    p.ref_lines()
    if not ridge:
        for j, g in enumerate(groups):
            m, col = dgrp == g, p.color(g)
            if alpha > 0:
                ax.fill_between(X[m], 0, Y[m], color=col, alpha=alpha, linewidth=0, zorder=Z_BAND + j * 0.01)
            ax.plot(X[m], Y[m], color=col, linewidth=lw, zorder=Z_LINE, solid_joinstyle="round")
            if o.get("show_rug") and main is not None:
                _rug(p, ax, main.loc[main["group"].astype(str) == g, "value"], col)
            p.lg_patch(g, col, max(alpha, 0.25), ec=col, lw=min(lw, 1.5))
        p.num_axis("x", "x", data=(float(X.min()), float(X.max())), pad=0.0)
        p.num_axis("y", "y", data=(0.0, float(Y.max()) or 1.0), zero=True, pad=0.05)
        p.axis_titles()
        return
    # ridgeline: one row per group, first group on top, rows overlap by ``ridge_overlap``
    n = len(groups)
    overlap = p.num("ridge_overlap", 1.5)
    k = overlap / (float(Y.max()) or 1.0)
    for j, g in enumerate(groups):
        m, col = dgrp == g, p.color(g)
        base = float(n - 1 - j)
        x, y = X[m], base + Y[m] * k
        z = Z_FILL + j * 0.1                         # lower rows in front of the rows above
        ax.fill_between(x, base, y, color=p.panel, linewidth=0, zorder=z)
        ax.fill_between(x, base, y, color=col, alpha=alpha, linewidth=0, zorder=z + 0.01)
        ax.plot(x, y, color=darken(col, 0.15), linewidth=lw, zorder=z + 0.02, solid_joinstyle="round")
        ax.plot([x.min(), x.max()], [base, base], color=darken(col, 0.15), linewidth=max(lw * 0.5, 0.6), zorder=z + 0.02)
        if o.get("show_rug") and main is not None:
            v = main.loc[main["group"].astype(str) == g, "value"].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            segs = np.stack([np.column_stack([v, np.full(len(v), base)]),
                             np.column_stack([v, np.full(len(v), base - 0.07)])], axis=1)
            ax.add_collection(LineCollection(segs, colors=col, linewidths=0.7, alpha=0.7, zorder=z + 0.03),
                              autolim=False)
        p.lg_patch(g, col, alpha)
    p.legend_auto = "none"
    p.num_axis("x", "x", data=(float(X.min()), float(X.max())), pad=0.0)
    ax.set_ylim(-0.12 if o.get("show_rug") else -0.04, n - 1 + overlap * 1.04)
    ax.set_yticks(np.arange(n), groups[::-1])
    ax.tick_params(axis="y", length=0)
    if p.theme.get("grid") in ("y", "both"):
        ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    p.axis_titles()


# ====================================================================== qq
def qq(p: Plot):
    o, ax = p.o, p.ax
    df, lines = p.frames["qq"], p.frames.get("line")
    groups = _groups(p, df)
    p.ref_lines()
    g_col = df["group"].astype(str).to_numpy()
    T, S = df["theoretical"].to_numpy(dtype=float), df["sample"].to_numpy(dtype=float)
    lw = p.num("line_width", 1.2)
    span = np.array([T.min(), T.max()]) + np.array([-0.15, 0.15]) * (np.ptp(T) or 1.0) * 0.3
    for g in groups:
        m, col = g_col == g, p.color(g)
        st = point_style(o.get("point_shape") or "circle", col, p.num("point_size", 5), p.num("point_alpha", 0.85),
                         ink=p.ink)
        if o.get("show_line", True) and lines is not None:
            row = lines[lines["group"].astype(str) == g]
            if len(row):
                s, i = float(row["slope"].iloc[0]), float(row["intercept"].iloc[0])
                if np.isfinite(s) and np.isfinite(i):
                    ax.plot(span, i + s * span, color=col if len(groups) > 1 else p.ink, linewidth=lw,
                            alpha=0.9, zorder=Z_LINE, linestyle="-" if len(groups) > 1 else (0, (5, 2.5)))
        ax.scatter(T[m], S[m], s=st["d"] ** 2, marker=st["marker"], facecolors=st["fc"], edgecolors=st["ec"],
                   linewidths=st["lw"], alpha=st["alpha"], zorder=Z_POINT)
        p.lg_point(g, st)
    p.num_axis("x", "x", data=(float(T.min()), float(T.max())), pad=0.06)
    p.num_axis("y", "y", data=(float(np.nanmin(S)), float(np.nanmax(S))), pad=0.06)
    p.axis_titles()


# ==================================================================== ecdf
def ecdf(p: Plot):
    ax = p.ax
    df = p.frames["ecdf"]
    groups = _groups(p, df)
    p.ref_lines()
    g_col = df["group"].astype(str).to_numpy()
    X, Y = df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    lw = p.num("line_width", 1.6)
    top = 100.0 if p.computed.get("percent") else 1.0
    for g in groups:
        m, col = g_col == g, p.color(g)
        ax.plot(X[m], Y[m], drawstyle="steps-post", color=col, linewidth=lw, zorder=Z_LINE, solid_joinstyle="miter")
        p.lg_line(g, col, lw)
    p.num_axis("x", "x", data=(float(np.nanmin(X)), float(np.nanmax(X))), pad=0.04)
    p.num_axis("y", "y", data=(0.0, top), lo=0.0, hi=top, pad_hi=0.03)
    p.axis_titles()


DRAWERS = {"histogram": histogram, "density": density, "qq": qq, "ecdf": ecdf}
