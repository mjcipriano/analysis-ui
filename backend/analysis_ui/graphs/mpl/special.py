"""Specialised graphs: Bland–Altman, ROC, volcano, forest, waterfall."""
from __future__ import annotations

import numpy as np

from .base import Plot, Z_BAND, Z_FILL, Z_LINE, Z_POINT, Z_REF, mix, point_style, rgba, tidy_labels
from .categorical import edge_color


def _fmt(v: float) -> str:
    a = abs(v)
    s = f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 0.1 else f"{v:.3g}"
    return s.replace("-", "−")


def _scatter(ax, x, y, st, zorder=Z_POINT):
    return ax.scatter(x, y, s=st["d"] ** 2, marker=st["marker"], facecolors=st["fc"], edgecolors=st["ec"],
                      linewidths=st["lw"], alpha=st["alpha"], zorder=zorder)


# ============================================================ bland_altman
def bland_altman(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    df = p.frames["main"]
    x, y = df["mean"].to_numpy(dtype=float), df["diff"].to_numpy(dtype=float)
    col = p.color("Points")
    lw = p.num("line_width", 1.2)
    p.ref_lines()
    lines = [("Mean", c.get("bias"), c.get("bias_ci"), "-"),
             ("+1.96 SD", c.get("loa_hi"), c.get("loa_hi_ci"), (0, (5, 3))),
             ("−1.96 SD", c.get("loa_lo"), c.get("loa_lo_ci"), (0, (5, 3)))]
    extent = [np.nanmin(y), np.nanmax(y)]
    soft = mix(p.ink, p.panel, 0.25)
    for name, v, ci, ls in lines:
        if v is None or not np.isfinite(v):
            continue
        if o.get("show_ci", True) and ci and all(np.isfinite(ci)):
            ax.axhspan(ci[0], ci[1], color=col if name == "Mean" else soft, alpha=0.13, linewidth=0, zorder=Z_BAND)
            extent += [ci[0], ci[1]]
        ax.axhline(v, color=p.ink if name == "Mean" else soft, linewidth=lw, linestyle=ls, zorder=Z_LINE)
        extent.append(v)
        if o.get("show_line_labels", True):
            ax.annotate(f"{name}\n{_fmt(v)}", (1.0, v), xycoords=ax.get_yaxis_transform(), xytext=(5, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=p.fonts["tick"] * 0.9,
                        color=p.ink, linespacing=1.15, annotation_clip=False)
    st = point_style(o.get("point_shape") or "circle", col, p.num("point_size", 5.5), p.num("point_alpha", 0.85),
                     ink=p.ink)
    _scatter(ax, x, y, st)
    p.num_axis("x", "x", data=(float(np.nanmin(x)), float(np.nanmax(x))), pad=0.06)
    p.num_axis("y", "y", data=(float(min(extent)), float(max(extent))), pad=0.08)
    p.axis_titles()


# ===================================================================== roc
def roc(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    df = p.frames["roc"]
    curves = c.get("curves") or []
    top = 100.0 if c.get("percent") else 1.0
    lw = p.num("line_width", 1.8)
    names = df["name"].astype(str).to_numpy()
    if o.get("show_diagonal", True):
        ax.plot([0, top], [0, top], color=mix(p.ink, p.panel, 0.55), linewidth=max(0.8, lw * 0.5),
                linestyle=(0, (4, 3)), zorder=Z_REF)
    p.ref_lines()
    for cv in curves:
        name = str(cv["name"])
        m, col = names == name, p.color(name)
        f, t = df["fpr"].to_numpy(dtype=float)[m], df["tpr"].to_numpy(dtype=float)[m]
        if o.get("fill_area"):
            ax.fill_between(f, 0, t, color=col, alpha=0.12, linewidth=0, zorder=Z_BAND)
        ax.plot(f, t, color=col, linewidth=lw, zorder=Z_LINE, solid_joinstyle="round", clip_on=False)
        p.lg_line(cv.get("label") or name if o.get("show_auc", True) else name, col, lw)
    p.num_axis("x", None, lo=0.0, hi=top, pad_lo=0.015, pad_hi=0.015)
    p.num_axis("y", None, lo=0.0, hi=top, pad_lo=0.015, pad_hi=0.015)
    angle = str(o.get("x_tick_angle") or "auto")
    if angle not in ("auto", "0"):
        p.hooks[0].append(lambda: p.rotate_xticks(ax, float(angle)))
    p.axis_titles()
    # ROC convention: square panel, legend in the empty lower-right corner
    p.placer = lambda l, b, r, t: p.place_square(ax, l, b, r, t)
    p.legend_auto = "inside_br"


# ================================================================= volcano
def volcano(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    df = p.frames["main"]
    x, y = df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    status = df["status"].astype(str).to_numpy()
    counts = c.get("counts") or {}
    size, alpha = p.num("point_size", 3.5), p.num("point_alpha", 0.75)
    p.ref_lines()
    if o.get("show_thresholds", True):
        kw = dict(color=mix(p.ink, p.panel, 0.45), linewidth=0.8, linestyle=(0, (4, 3)), zorder=Z_REF)
        xt, yt = c.get("x_threshold"), c.get("y_threshold")
        if xt:
            ax.axvline(xt, **kw)
            ax.axvline(-xt, **kw)
        if yt is not None and np.isfinite(yt):
            ax.axhline(yt, **kw)
    for z, key in enumerate(["Not significant", "Down", "Up"]):       # NS underneath
        m = status == key
        st = point_style("circle", p.color(key), size, alpha, ink=p.ink)
        if m.any():
            _scatter(ax, x[m], y[m], st, zorder=Z_POINT + z * 0.1)
    for key in ["Down", "Not significant", "Up"]:
        n = counts.get(key, int((status == key).sum()))
        p.lg_point(f"{key} ({n})", point_style("circle", p.color(key), max(size, 5), 1.0, ink=p.ink))
    p.legend_auto = "right"
    labelled = False
    if "label" in df:
        labs = [("" if (l is None or l != l) else str(l)) for l in df["label"]]
        if any(labs):
            labelled = True
            p.label_points(x, y, labs, float(o.get("label_size") or p.fonts["tick"] * 0.82), radius_pt=size / 2,
                           color=p.ink, avoid=np.column_stack([x, y]) if len(x) <= 20000 else None)
    p.num_axis("x", "x", data=(float(np.nanmin(x)), float(np.nanmax(x))), pad=0.06, symmetric=True)
    p.num_axis("y", "y", data=(0.0, float(np.nanmax(y)) or 1.0), zero=True, pad=0.07,
               pad_hi=0.14 if labelled else None)                  # head-room for labels above the top hits
    p.axis_titles()


# ================================================================== forest
def forest(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    df = p.frames["main"]
    labels = [str(l) for l in df["label"]]
    n = len(labels)
    est = df["est"].to_numpy(dtype=float)
    lo, hi = df["lo"].to_numpy(dtype=float), df["hi"].to_numpy(dtype=float)
    is_log = (o.get("x_scale") or "linear") in ("log10", "log2")
    if is_log:
        est, lo, hi = (np.where(a > 0, a, np.nan) for a in (est, lo, hi))
    col = p.color("Estimate")
    lw = p.num("line_width", 1.4)
    rows = np.arange(n, dtype=float)
    ref = c.get("ref_line")
    try:
        ref = float(ref) if ref is not None else None
    except (TypeError, ValueError):
        ref = None
    if ref is not None and np.isfinite(ref) and not (is_log and ref <= 0):
        ax.axvline(ref, color=mix(p.ink, p.panel, 0.35), linewidth=max(0.8, float(p.theme.get("axis_width") or 0.8) * 0.8),
                   zorder=Z_REF)
    else:
        ref = None
    p.ref_lines()
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.hlines(rows[ok], lo[ok], hi[ok], colors=col, linewidths=lw, zorder=Z_LINE)
        cap = p.num("error_cap", 0)
        if cap > 0:
            ends_x, ends_y = np.concatenate([lo[ok], hi[ok]]), np.concatenate([rows[ok], rows[ok]])
            ax.scatter(ends_x, ends_y, marker="|", s=cap ** 2, color=col, linewidths=lw, zorder=Z_LINE)
    st = point_style(o.get("point_shape") or "square", col, p.num("point_size", 7), 1.0, ink=p.ink)
    _scatter(ax, est, rows, st)
    p.cat_axis(labels, which="y", pad=0.6, invert=True)
    ax.tick_params(axis="y", length=0)
    if p.theme.get("grid") in ("y", "both"):
        ax.grid(False, axis="y")
    allv = np.concatenate([est, lo, hi, [ref] if ref is not None else []]).astype(float)
    allv = allv[np.isfinite(allv)]
    data = (float(allv.min()), float(allv.max())) if len(allv) else None
    p.num_axis("x", "x", data=data, pad=0.06)
    if c.get("show_text", True) and "text" in df:
        for i, t in enumerate(df["text"]):
            if t:
                ax.annotate(str(t), (1.0, i), xycoords=ax.get_yaxis_transform(), xytext=(9, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=p.fonts["tick"],
                            color=p.ink, annotation_clip=False)
    ax.set_xlabel(p.labels.get("x", ""))
    ax.set_ylabel(p.labels.get("y", ""))


# =============================================================== waterfall
def waterfall(p: Plot):
    o, ax, c = p.o, p.ax, p.computed
    df = p.frames["main"]
    labels = tidy_labels(df["label"])
    v = df["value"].to_numpy(dtype=float)
    grp = df["group"].astype(str).to_numpy()
    groups = [str(g) for g in (c.get("groups") or [])]
    n = len(v)
    alpha, outline = p.num("fill_alpha", 1.0), o.get("outline") or "match"
    lw = p.num("outline_width", 1.0) if outline != "none" else 0.0
    lw = min(lw, max(p.W * 72 / max(n, 1) / 8, 0.2))                   # thin outlines for many bars
    colors = [p.color(g) for g in grp]
    p.ref_lines(categorical=True)
    ax.bar(np.arange(n), v, width=p.num("bar_width", 0.85), color=[rgba(k, alpha) for k in colors],
           edgecolor=[edge_color(outline, k, p.ink) for k in colors] if lw > 0 else "none", linewidth=lw,
           zorder=Z_FILL)
    if np.nanmin(v) < 0:
        ax.axhline(0, color=p.axis_color, linewidth=max(float(p.theme.get("axis_width") or 0.8), 0.6), zorder=Z_FILL + 0.5)
    if c.get("show_labels"):
        p.cat_axis(labels, which="x", pad=0.7)
    else:
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_xticks([])
    if p.theme.get("grid") in ("x", "both"):
        ax.grid(False, axis="x")
    p.num_axis("y", "y", zero=True, pad=0.06)
    for g in groups:
        k = p.color(g)
        p.lg_patch(g, k, alpha, ec=edge_color(outline, k, p.ink) if lw > 0 else "none", lw=min(lw, 1.5))
    p.legend_auto = "right" if c.get("has_group") and len(groups) > 1 else "none"
    ax.set_xlabel(p.labels.get("x", ""))
    ax.set_ylabel(p.labels.get("y", ""))


DRAWERS = {"bland_altman": bland_altman, "roc": roc, "volcano": volcano, "forest": forest, "waterfall": waterfall}
