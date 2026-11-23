"""Kaplan–Meier survival curves with censor marks, CI bands and a risk table."""
from __future__ import annotations

import numpy as np

from .base import Plot, Z_BAND, Z_LINE, Z_POINT, Z_TEXT, darken, luminance, mix


def survival(p: Plot):
    o, ax = p.o, p.ax
    curve, censor, risk = p.frames["curve"], p.frames.get("censor"), p.frames.get("risk")
    groups = [str(g) for g in (p.computed.get("groups") or curve["group"].astype(str).unique())]
    top = 100.0 if p.computed.get("percent") else 1.0
    cumulative = bool(p.computed.get("cumulative"))
    lw = p.num("line_width", 1.8)
    p.ref_lines()
    cg = curve["group"].astype(str).to_numpy()
    t_max = 0.0
    for g in groups:
        m, col = cg == g, p.color(g)
        t, s = curve["time"].to_numpy(dtype=float)[m], curve["surv"].to_numpy(dtype=float)[m]
        if not len(t):
            continue
        t_max = max(t_max, float(np.nanmax(t)))
        if o.get("show_ci"):
            lo, hi = curve["lo"].to_numpy(dtype=float)[m], curve["hi"].to_numpy(dtype=float)[m]
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.sum() > 1:
                ax.fill_between(t[ok], lo[ok], hi[ok], step="post", color=col, alpha=0.16, linewidth=0, zorder=Z_BAND)
        ax.plot(t, s, drawstyle="steps-post", color=col, linewidth=lw, zorder=Z_LINE, solid_joinstyle="miter")
        if o.get("show_censors", True) and censor is not None and len(censor):
            c = censor[censor["group"].astype(str) == g]
            if len(c):
                t_max = max(t_max, float(c["time"].max()))
                ax.scatter(c["time"].to_numpy(dtype=float), c["surv"].to_numpy(dtype=float), marker="|",
                           s=(max(lw * 3.6, 6.0)) ** 2, color=col, linewidths=max(lw * 0.6, 0.8), zorder=Z_POINT)
        p.lg_line(g, col, lw)
    if o.get("median_line"):
        half = top / 2
        meds = [(str(d["group"]), d.get("median")) for d in (p.computed.get("medians") or [])]
        meds = [(g, float(v)) for g, v in meds if v is not None and np.isfinite(v)]
        if meds:
            kw = dict(linewidth=max(0.7, lw * 0.45), linestyle=(0, (3, 2.5)), zorder=Z_LINE - 0.5)
            ax.plot([0, max(v for _, v in meds)], [half, half], color=mix(p.ink, p.panel, 0.35), **kw)
            for g, v in meds:
                ax.plot([v, v], [0, half], color=p.color(g), **kw)
    if o.get("show_p", True) and p.computed.get("p_label"):
        ax.text(0.04, 0.95 if cumulative else 0.06, p.computed["p_label"], transform=ax.transAxes,
                ha="left", va="top" if cumulative else "bottom", fontsize=p.fonts["tick"], color=p.ink, zorder=Z_TEXT)
    p.num_axis("x", "x", data=(0.0, t_max or 1.0), lo=0.0, pad_hi=0.03)
    p.num_axis("y", "y", data=(0.0, top), lo=0.0, hi=top, pad_hi=0.03)
    p.axis_titles()
    if o.get("risk_table") and risk is not None and len(risk):
        _risk_table(p, risk, groups)


def _risk_table(p: Plot, risk, groups: list[str]):
    """Numbers at risk under the x axis, one row per group, columns aligned
    with ``computed.risk_times``."""
    ax = p.ax
    fs = p.fonts["tick"] * 0.92
    tab = p.new_axes(sharex=ax)
    tab.set_facecolor("none")
    tab.grid(False)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.tick_params(axis="both", which="both", length=0, labelbottom=False, labelsize=fs)
    n = len(groups)
    tab.set_ylim(n - 0.5, -0.5)
    tab.set_yticks(np.arange(n), groups)
    dark_bg = luminance(p.bg) < 0.5
    cols = [p.color(g) if dark_bg else darken(p.color(g), 0.1) for g in groups]
    for lab, col in zip(tab.get_yticklabels(), cols):
        lab.set_color(col)
    x0, x1 = sorted(ax.get_xlim())
    rg = risk["group"].astype(str).to_numpy()
    for i, g in enumerate(groups):
        r = risk[rg == g]
        for t, k in zip(r["time"].to_numpy(dtype=float), r["n"].to_numpy()):
            if x0 - 1e-9 <= t <= x1 + 1e-9 and np.isfinite(k):
                tab.text(t, i, f"{int(k)}", ha="center", va="center", fontsize=fs, color=cols[i], clip_on=False)
    tab.text(0.0, 1.0, "Number at risk", transform=tab.transAxes, ha="left", va="bottom", fontsize=fs,
             fontweight="bold", color=p.ink)
    row_h = fs * p.px * 1.45
    title_h = fs * p.px * 1.7

    def place(l, b, r, t):
        table_h = n * row_h
        below = max(ax.bbox.y0 - ax.get_tightbbox(p.renderer).y0, 0)      # x tick labels + axis title
        split = b + table_h + title_h + below + 0.3 * row_h
        if t - split < 0.3 * (t - b):               # tiny figure: squeeze the curves rather than vanish
            split = b + 0.7 * (t - b)
        p.set_pos(tab, l, b, r, b + table_h)
        p.set_pos(ax, l, split, r, t)
    p.placer = place


DRAWERS = {"survival": survival}
