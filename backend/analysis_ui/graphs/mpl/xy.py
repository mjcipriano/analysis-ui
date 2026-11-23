"""XY graphs: scatter, line, dose–response, bubble (render type ``xy``)."""
from __future__ import annotations

import numpy as np
from matplotlib import ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .base import (LINE_CYCLE, LINE_STYLES, SHAPE_CYCLE, Plot, Z_BAND, Z_LINE, Z_POINT, Z_SUMMARY, mix,
                   point_style)


def wrap_fit_label(label: str) -> str:
    """Keep fit results narrow enough for a legend beside the plot."""
    label = str(label).replace(" (R²", "\n(R²")
    first = label.split("\n")[0]
    if len(first) > 26 and ", " in first:
        label = label.replace(", ", ",\n", 1)
    return label


def _rug(p: Plot, x, y, color):
    ax = p.ax
    k = max(p.fonts["base"] / 11, 0.7)
    for vals, trans, horizontal in ((x, ax.get_xaxis_transform(), False), (y, ax.get_yaxis_transform(), True)):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        z, tip = np.zeros(len(vals)), np.full(len(vals), 0.025)
        a = np.column_stack([z, vals] if horizontal else [vals, z])
        b = np.column_stack([tip, vals] if horizontal else [vals, tip])
        ax.add_collection(LineCollection(np.stack([a, b], axis=1), colors=color, linewidths=0.7 * k, alpha=0.6,
                                         transform=trans, zorder=Z_LINE), autolim=False)


def xy(p: Plot):
    o, ax = p.o, p.ax
    p.ref_lines()
    main = p.frames["main"]
    summ, fit = p.frames.get("summary"), p.frames.get("fit")
    series = [str(s) for s in p.computed.get("series", [])]
    mode = p.computed.get("mode", "points")
    has_summary = mode in ("summary", "both") and summ is not None and len(summ)
    raw = mode in ("points", "both") or not has_summary
    show_points = bool(o.get("show_points", True))
    has_size = bool(p.computed.get("has_size")) and "size" in main.columns
    has_label = bool(p.computed.get("has_label")) and "label" in main.columns
    size, alpha, lw = p.num("point_size", 6), p.num("point_alpha", 0.9), p.num("line_width", 1.5)
    shape_opt, ls_opt, connect = o.get("point_shape") or "circle", o.get("line_style") or "solid", o.get("connect") or "none"
    log_x, log_y = (o.get("x_scale") in ("log10", "log2")), (o.get("y_scale") in ("log10", "log2"))

    grp = main["group"].astype(str).to_numpy()
    X, Y = main["x"].to_numpy(dtype=float), main["y"].to_numpy(dtype=float)
    S = main["size"].to_numpy(dtype=float) if has_size else None
    S_all = p.full_frames["main"]["size"].to_numpy(dtype=float) if has_size else None     # same scale in all facets
    smax = float(np.nanmax(np.where(S_all > 0, S_all, np.nan))) if has_size and np.any(S_all > 0) else None
    d_max = p.num("size_max", 22)
    fit_labels = {str(f["group"]): f["label"] for f in (p.computed.get("fits") or [])}
    sgrp = summ["group"].astype(str).to_numpy() if has_summary else None
    fgrp = fit["group"].astype(str).to_numpy() if fit is not None and len(fit) else None
    xs_all, ys_all, fit_entries = [], [], []

    for i, s in enumerate(series):
        col = p.color(s)
        shape = SHAPE_CYCLE[i % len(SHAPE_CYCLE)] if shape_opt == "auto" else shape_opt
        ls = LINE_CYCLE[i % len(LINE_CYCLE)] if ls_opt == "auto" else LINE_STYLES.get(ls_opt, "-")
        st = point_style(shape, col, size, alpha, ink=p.ink)
        m = grp == s
        x, y = X[m], Y[m]
        sm = summ[sgrp == s] if has_summary else None
        # --- connecting line
        if connect in ("line", "step"):
            if has_summary:
                lx, ly = sm["x"].to_numpy(dtype=float), sm["center"].to_numpy(dtype=float)
            else:
                lx, ly = x, y
            order = np.argsort(lx, kind="stable")
            ax.plot(lx[order], ly[order], color=col, linewidth=lw, linestyle=ls, zorder=Z_LINE,
                    drawstyle="steps-post" if connect == "step" else "default", solid_joinstyle="round")
        # --- summary: error bars / band and centre markers
        if has_summary and len(sm):
            cx, cc = sm["x"].to_numpy(dtype=float), sm["center"].to_numpy(dtype=float)
            lo, hi = sm["lo"].to_numpy(dtype=float), sm["hi"].to_numpy(dtype=float)
            if log_y:
                lo = np.where(lo > 0, lo, np.nan)
            ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(cc)
            if ok.any():
                if (o.get("error_style") or "bars") == "band":
                    order = np.argsort(cx[ok])
                    ax.fill_between(cx[ok][order], lo[ok][order], hi[ok][order], color=col, alpha=0.2,
                                    linewidth=0, zorder=Z_BAND)
                else:
                    cap = p.num("error_cap", 3)
                    elw = max(0.8, lw * 0.7)
                    ax.errorbar(cx[ok], cc[ok], yerr=[np.maximum(cc[ok] - lo[ok], 0), np.maximum(hi[ok] - cc[ok], 0)],
                                fmt="none", ecolor=col, elinewidth=elw, capsize=cap / 2, capthick=elw,
                                zorder=Z_SUMMARY - 0.5)
                ys_all += [lo[ok], hi[ok]]
            if show_points:
                ax.scatter(cx, cc, s=st["d"] ** 2, marker=st["marker"], facecolors=st["fc"], edgecolors=st["ec"],
                           linewidths=st["lw"], zorder=Z_SUMMARY)
            xs_all.append(cx); ys_all.append(cc)
        # --- individual points
        if raw and len(x):
            xs_all.append(x); ys_all.append(y)
            if show_points:
                if has_size and smax:
                    sz = S[m]
                    d = np.where(np.isfinite(sz) & (sz > 0), d_max * np.sqrt(np.clip(sz, 0, None) / smax), 1.5)
                    order = np.argsort(-d, kind="stable")          # small bubbles on top
                    ax.scatter(x[order], y[order], s=np.maximum(d[order], 1.5) ** 2, marker=st["marker"],
                               facecolors=st["fc"], edgecolors=p.panel if st["fc"] != "none" else st["ec"],
                               linewidths=0.6 if st["fc"] != "none" else st["lw"], alpha=alpha, zorder=Z_POINT)
                else:
                    faint = has_summary                            # replicates behind their mean
                    rst = point_style(shape, col, size * (0.65 if faint else 1.0),
                                      alpha * (0.45 if faint else 1.0), ink=p.ink)
                    ax.scatter(x, y, s=rst["d"] ** 2, marker=rst["marker"], facecolors=rst["fc"],
                               edgecolors=rst["ec"], linewidths=rst["lw"], alpha=rst["alpha"],
                               zorder=Z_POINT - (0.5 if faint else 0))
            if o.get("marginal_rug"):
                _rug(p, x, y, col)
        # --- fitted curve
        has_fit = False
        if fgrp is not None:
            ft = fit[fgrp == s]
            if len(ft):
                has_fit = True
                fx, fy = ft["x"].to_numpy(dtype=float), ft["y"].to_numpy(dtype=float)
                flo, fhi = ft["lo"].to_numpy(dtype=float), ft["hi"].to_numpy(dtype=float)
                if log_y:
                    fy = np.where(fy > 0, fy, np.nan)
                    flo = np.where(flo > 0, flo, np.nan)
                band = np.isfinite(flo) & np.isfinite(fhi)
                if o.get("fit_band", True) and band.sum() > 1:
                    ax.fill_between(fx[band], flo[band], fhi[band], color=col, alpha=0.16, linewidth=0, zorder=Z_BAND)
                ax.plot(fx, fy, color=col, linewidth=lw, linestyle=ls if connect == "none" else "-",
                        zorder=Z_LINE + 0.2, solid_capstyle="round")
                xs_all.append(fx)
                fit_entries.append((s, col, ls if connect == "none" else "-"))
                ys_all.append(("fit", fy))
        # --- legend entry for the series
        with_line = connect != "none" or (has_fit and not show_points)
        if show_points and with_line:
            p.lg_line(s, col, lw, ls, st=st)
        elif with_line or not show_points:
            p.lg_line(s, col, lw, ls)
        else:
            p.lg_point(s, st)

    # ---- legend extras: fit results and bubble sizes
    extra = False
    if fit_entries and o.get("fit_legend", True) and p.faceted:
        # fit results differ between panels: write them inside each panel instead of the legend
        lines = [(str(fit_labels[s_]).replace(" (R²", ", R²").rstrip(")") if " (R²" in str(fit_labels[s_])
                  else str(fit_labels[s_]), col) for s_, col, _ in fit_entries if s_ in fit_labels]
        fs = p.fonts["tick"] * 0.85
        notes = []
        for txt, col in lines:
            notes.append(ax.annotate(txt, (0, 1), xycoords="axes fraction", xytext=(0, 0), textcoords="offset points",
                                     ha="left", va="top", fontsize=fs, color=p.accent(col, 0.25),
                                     zorder=Z_SUMMARY + 1))
            notes[-1].set_in_layout(False)           # must not push the panels apart

        def fit_notes(ax=ax):                        # shrink to the panel width, then stack the lines
            widest = max(p.text_size(t.get_text(), fs)[0] for t in notes)
            size = max(min(fs, fs * (ax.bbox.width - fs * p.px) / max(widest, 1.0)), 4.0)
            for k, t in enumerate(notes):
                t.set_fontsize(size)
                t.xyann = (0.5 * size, -0.5 * size - k * size * 1.25)
        fit_notes()
        p.hooks[2].append(fit_notes)
    elif fit_entries and o.get("fit_legend", True):
        if len(series) == 1:
            p.legend_items = []
        for s, col, ls in fit_entries:
            if s in fit_labels:
                p.lg_line(wrap_fit_label(fit_labels[s]), col, lw, ls)
                extra = True
    if has_size and smax and show_points and raw:
        smin = float(np.nanmin(np.where(S_all > 0, S_all, np.nan)))
        vals = [v for v in mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(smin, smax)
                if smin * 0.999 <= v <= smax * 1.001 and v > 0]
        vals = [vals[0], vals[len(vals) // 2], vals[-1]] if len(vals) > 3 else (vals or [smax])
        if vals:
            if len(series) == 1:
                p.legend_items = []
            gray = mix(p.ink, p.panel, 0.45)
            p.legend_items.append((Patch(facecolor="none", edgecolor="none"), str(o.get("size") or "Size")))
            for v in vals:
                d = d_max * np.sqrt(v / smax)
                p.legend_items.append((Line2D([], [], linestyle="none", marker="o", markersize=d,
                                              markerfacecolor=gray, markeredgecolor="none", alpha=0.6), f"{v:g}"))
            fs = p.fonts["legend"]
            p.legend_kw.update(labelspacing=max(0.5, d_max / fs * 0.62 - 0.35), handlelength=max(1.1, d_max / fs),
                               handletextpad=0.8)
            extra = True
    if extra:
        p.legend_auto = "right"

    # ---- labels
    if has_label and raw:
        r = (d_max * 0.35) if has_size else size / 2
        p.label_points(X, Y, main["label"].to_numpy(dtype=object), p.fonts["tick"] * 0.85, radius_pt=r,
                       color=p.ink, avoid=np.column_stack([X, Y]))

    # ---- axes: limits follow the data; fitted curves may not blow them up
    def extent(parts, log):
        core = [np.asarray(a, dtype=float) for a in parts if not isinstance(a, tuple)]
        v = np.concatenate(core) if core else np.array([])
        v = v[np.isfinite(v) & ((v > 0) if log else True)]
        if not len(v):
            return None
        lo, hi = float(v.min()), float(v.max())
        span = (hi - lo) or abs(hi) or 1.0
        for a in parts:
            if isinstance(a, tuple):                 # fitted values: allowed to widen the range a little
                f = np.asarray(a[1], dtype=float)
                f = f[np.isfinite(f) & ((f > 0) if log else True)]
                if len(f):
                    if log:
                        lo, hi = min(lo, max(f.min(), lo / 10)), max(hi, min(f.max(), hi * 10))
                    else:
                        lo, hi = min(lo, max(f.min(), lo - 0.5 * span)), max(hi, min(f.max(), hi + 0.5 * span))
        return lo, hi

    pad = 0.09 if has_size else 0.05
    p.num_axis("x", "x", data=extent(xs_all, log_x), pad=pad)
    p.num_axis("y", "y", data=extent(ys_all, log_y), pad=pad)
    p.axis_titles()


DRAWERS = {"xy": xy}
