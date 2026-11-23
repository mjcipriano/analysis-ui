"""Column and grouped graphs: one numeric axis, one categorical axis.

Both families are drawn from the same *cells*: a cell is one group of values
at one position on the categorical axis. Column graphs have one cell per
category; grouped graphs dodge several cells inside each category. Every
element (bars, boxes, violins, points, summaries) is drawn for a list of
cells, so the two families share all their code and ``flip`` comes for free
through the flip-aware helpers of :class:`Plot`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import (Plot, Z_FILL, Z_LINE, Z_POINT, Z_SUMMARY, beeswarm, darken, kde_curve, mix,
                   point_style, rgba)

UP_COLOR, DOWN_COLOR = "#D7402B", "#2C6FBB"


@dataclass
class Cell:
    pos: float                    # centre on the categorical axis
    slot: float                   # width available to this cell
    key: str                      # colour key
    color: str
    values: np.ndarray
    center: float = np.nan
    lo: float = np.nan
    hi: float = np.nan
    x: str = ""                   # category label (grouped graphs)
    index: int = 0
    top: float = field(default=-np.inf)   # highest drawn value (for brackets)

    def raise_top(self, *vals):
        for v in vals:
            if v is not None and np.isfinite(v):
                self.top = max(self.top, float(v))


# ------------------------------------------------------------------- cells
def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def column_cells(p: Plot) -> tuple[list[Cell], list[str]]:
    main, summ = p.frames["main"], p.frames.get("summary")
    groups = [str(g) for g in (p.computed.get("groups") or main["group"].astype(str).unique())]
    vals = main["value"].to_numpy(dtype=float)
    grp = main["group"].astype(str).to_numpy()
    rows = {str(r.group): r for r in summ.itertuples()} if summ is not None else {}
    cells = []
    for i, g in enumerate(groups):
        v = vals[grp == g]
        r = rows.get(g)
        cells.append(Cell(pos=float(i), slot=1.0, key=g, color=p.color(g), values=v[np.isfinite(v)], index=i,
                          center=_num(getattr(r, "center", np.nan)), lo=_num(getattr(r, "lo", np.nan)),
                          hi=_num(getattr(r, "hi", np.nan)), x=g))
    return cells, groups


def grouped_cells(p: Plot, dodge: float, stacked: bool = False) -> tuple[list[Cell], list[str], list[str]]:
    main, summ = p.frames["main"], p.frames.get("summary")
    xs = [str(v) for v in p.computed.get("x_levels", [])]
    gs = [str(v) for v in p.computed.get("groups", [])]
    vals = main["value"].to_numpy(dtype=float)
    xcol, gcol = main["x"].astype(str).to_numpy(), main["group"].astype(str).to_numpy()
    rows = {(str(r.x), str(r.group)): r for r in summ.itertuples()} if summ is not None else {}
    ng = max(len(gs), 1)
    slot = dodge if stacked else dodge / ng
    cells = []
    for i, x in enumerate(xs):
        in_x = xcol == x
        for j, g in enumerate(gs):
            v = vals[in_x & (gcol == g)]
            v = v[np.isfinite(v)]
            r = rows.get((x, g))
            if not len(v) and r is None:
                continue
            pos = float(i) if stacked else i + (j - (ng - 1) / 2) * slot
            cells.append(Cell(pos=pos, slot=slot, key=g, color=p.color(g), values=v, index=len(cells), x=x,
                              center=_num(getattr(r, "center", np.nan)), lo=_num(getattr(r, "lo", np.nan)),
                              hi=_num(getattr(r, "hi", np.nan))))
    return cells, xs, gs


# ---------------------------------------------------------------- elements
def edge_color(outline: str, color: str, ink: str) -> str:
    return {"match": color, "black": ink, "white": "#FFFFFF"}.get(outline, "none")


def draw_bars(p: Plot, cells: list[Cell], base: float, width_frac: float, alpha: float, outline: str,
              outline_width: float):
    cells = [c for c in cells if np.isfinite(c.center)]
    if not cells:
        return
    lw = outline_width if outline != "none" else 0.0
    p.bars([c.pos for c in cells], [c.center for c in cells], [c.slot * width_frac for c in cells], base=base,
           color=[rgba(c.color, alpha) for c in cells],
           edgecolor=[edge_color(outline, c.color, p.ink) for c in cells] if lw > 0 else "none",
           linewidth=lw, zorder=Z_FILL)
    for c in cells:
        c.raise_top(c.center, base)


def draw_errors(p: Plot, cells: list[Cell], cap_frac: float, lw: float, color=None, direction: str = "both"):
    if not cells:
        return
    colors = color if color is not None else [c.color for c in cells]
    p.errorbars([c.pos for c in cells], [c.center for c in cells], [c.lo for c in cells], [c.hi for c in cells],
                cap=0.0, lw=lw, color=colors, direction=direction)
    # caps separately so that each cell can have its own cap width
    for c in cells:
        c.raise_top(c.hi, c.center)
    if cap_frac > 0:
        C0, V0, C1, V1, cols = [], [], [], [], []
        for i, c in enumerate(cells):
            if not np.isfinite(c.center):
                continue
            ends = [c.hi] if direction == "up" else [c.lo, c.hi]
            for e in ends:
                if np.isfinite(e):
                    h = cap_frac * c.slot / 2
                    C0.append(c.pos - h); C1.append(c.pos + h); V0.append(e); V1.append(e)
                    cols.append(color if color is not None else c.color)
        if C0:
            p.segments(C0, V0, C1, V1, colors=cols, linewidths=lw, zorder=Z_SUMMARY)


def draw_center_lines(p: Plot, cells: list[Cell], width_frac: float, lw: float, color=None):
    cells = [c for c in cells if np.isfinite(c.center)]
    if not cells:
        return
    h = np.array([c.slot * width_frac / 2 for c in cells])
    pos, cen = np.array([c.pos for c in cells]), np.array([c.center for c in cells])
    p.segments(pos - h, cen, pos + h, cen, colors=color if color is not None else [c.color for c in cells],
               linewidths=lw, zorder=Z_SUMMARY)
    for c in cells:
        c.raise_top(c.center)


def cell_style(p: Plot, c: Cell, on_fill: bool = False, size: float | None = None, alpha: float | None = None):
    return point_style(p.opt("point_shape", "circle"), c.color, p.num("point_size", 5) if size is None else size,
                       p.num("point_alpha", 0.9) if alpha is None else alpha, p.opt("point_color", "group"),
                       ink=p.ink, on_fill=on_fill)


def draw_points(p: Plot, cells: list[Cell], layout: str, half: float | None = None, on_fill: bool = False,
                offset: float = 0.0, size: float | None = None, alpha: float | None = None, zorder=Z_POINT):
    """Individual values. ``half`` is the largest half-spread in category
    units (None: a fraction of each cell's slot); ``offset`` shifts the points
    sideways (raincloud)."""
    cells = [c for c in cells if len(c.values)]
    if not cells:
        return
    for c in cells:
        c.raise_top(c.values.max())
        p.note(c.pos + offset, [c.values.min(), c.values.max()])
    spread = {c.index: (0.38 * c.slot if half is None else half) for c in cells}
    styles = {c.index: cell_style(p, c, on_fill, size, alpha) for c in cells}

    def emit(offsets: dict):
        by_color: dict = {}
        for c in cells:
            by_color.setdefault(c.color, []).append(c)
        for group in by_color.values():
            pos = np.concatenate([c.pos + offset + offsets[c.index] for c in group])
            p.scatter(pos, np.concatenate([c.values for c in group]), styles[group[0].index], zorder=zorder)

    if layout == "aligned" or all(s <= 0 for s in spread.values()):
        emit({c.index: np.zeros(len(c.values)) for c in cells})
    elif layout == "jitter":
        emit({c.index: np.random.default_rng(7919 + c.index).uniform(-spread[c.index], spread[c.index],
                                                                    len(c.values)) for c in cells})
    else:                                            # beeswarm needs the final pixel geometry
        def run():
            ax = p.ax
            lim = ax.get_ylim() if p.flip else ax.get_xlim()
            cat_px = (ax.bbox.height if p.flip else ax.bbox.width) / max(abs(lim[1] - lim[0]), 1e-9)
            offsets = {}
            for c in cells:
                st = styles[c.index]
                if len(c.values) > 1500:             # too many to swarm: fall back to jitter
                    offsets[c.index] = np.random.default_rng(7919 + c.index).uniform(
                        -spread[c.index], spread[c.index], len(c.values))
                    continue
                pts = ax.transData.transform(np.column_stack(p.xy(np.zeros(len(c.values)), c.values)))
                v_px = pts[:, 0] if p.flip else pts[:, 1]
                d_px = (st["d"] + st["lw"] * 0.5) * p.px * 1.06
                offsets[c.index] = beeswarm(v_px, d_px, spread[c.index] * cat_px) / cat_px
            emit(offsets)
        p.hooks[2].append(run)


def box_stats(v: np.ndarray, mode: str) -> dict:
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    if mode == "minmax":
        lo, hi = v.min(), v.max()
    elif mode == "p5_95":
        lo, hi = np.percentile(v, [5, 95])
    elif mode == "p10_90":
        lo, hi = np.percentile(v, [10, 90])
    else:
        iqr = q3 - q1
        inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        lo, hi = (inside.min(), inside.max()) if len(inside) else (q1, q3)
    lo, hi = min(lo, q1), max(hi, q3)
    return {"q1": q1, "med": med, "q3": q3, "lo": lo, "hi": hi, "out": v[(v < lo) | (v > hi)]}


def draw_boxes(p: Plot, cells: list[Cell], *, whiskers: str = "tukey", outliers: bool = True, mean: bool = False,
               notch: bool = False, width_frac: float = 0.6, alpha: float = 0.6, outline: str = "match",
               outline_width: float = 1.0, caps: bool = True, zorder=Z_FILL):
    cells = [c for c in cells if len(c.values)]
    if not cells:
        return
    lw = max(outline_width, 0.0) if outline != "none" else 0.0
    line_lw = lw if lw > 0 else 1.0
    shapes, faces, edges = [], [], []
    W = {"c0": [], "v0": [], "c1": [], "v1": [], "col": []}       # whiskers and caps
    M = {"c0": [], "v0": [], "c1": [], "v1": [], "col": []}       # medians
    out_c, out_v, out_col, mean_c, mean_v = [], [], [], [], []
    for c in cells:
        s = box_stats(c.values, whiskers)
        w = c.slot * width_frac
        l, r = c.pos - w / 2, c.pos + w / 2
        ec = edge_color(outline, c.color, p.ink)
        line_col = c.color if ec == "none" else ec
        med_half = w / 2
        if notch and len(c.values) > 1 and s["q3"] > s["q1"]:
            ci = 1.57 * (s["q3"] - s["q1"]) / np.sqrt(len(c.values))
            nlo, nhi = max(s["med"] - ci, s["q1"]), min(s["med"] + ci, s["q3"])
            med_half = w / 4
            shapes.append(([l, r, r, c.pos + med_half, r, r, l, l, c.pos - med_half, l],
                           [s["q1"], s["q1"], nlo, s["med"], nhi, s["q3"], s["q3"], nhi, s["med"], nlo]))
        else:
            shapes.append(([l, r, r, l], [s["q1"], s["q1"], s["q3"], s["q3"]]))
        faces.append(rgba(c.color, alpha))
        edges.append(ec)
        cap = w * 0.22 if caps else 0.0
        for a, b in ((s["lo"], s["q1"]), (s["q3"], s["hi"])):
            W["c0"].append(c.pos); W["v0"].append(a); W["c1"].append(c.pos); W["v1"].append(b); W["col"].append(line_col)
        if cap:
            for e in (s["lo"], s["hi"]):
                W["c0"].append(c.pos - cap); W["v0"].append(e); W["c1"].append(c.pos + cap); W["v1"].append(e)
                W["col"].append(line_col)
        # the median must stand out from the fill whatever the outline is
        med_col = p.ink if outline == "black" else darken(c.color, 0.45 if alpha > 0.5 else 0.15)
        M["c0"].append(c.pos - med_half); M["v0"].append(s["med"]); M["c1"].append(c.pos + med_half)
        M["v1"].append(s["med"]); M["col"].append(med_col)
        if outliers and len(s["out"]):
            out_c += [c.pos] * len(s["out"]); out_v += list(s["out"]); out_col += [line_col] * len(s["out"])
        if mean:
            mean_c.append(c.pos); mean_v.append(float(np.mean(c.values)))
        c.raise_top(s["hi"], c.values.max() if outliers else s["hi"])
    p.polys(shapes, facecolors=faces, edgecolors=edges if lw > 0 else "none", linewidths=lw, zorder=zorder,
            joinstyle="miter")
    p.segments(W["c0"], W["v0"], W["c1"], W["v1"], colors=W["col"], linewidths=line_lw, zorder=zorder + 0.1)
    p.segments(M["c0"], M["v0"], M["c1"], M["v1"], colors=M["col"], linewidths=max(line_lw * 1.6, 1.6),
               zorder=zorder + 0.2)
    if out_c:
        x, y = p.xy(np.array(out_c), np.array(out_v))
        d = max(min(p.num("point_size", 4.0), 5.0), 2.5)
        p.ax.scatter(x, y, s=d * d, marker="o", facecolors="none", edgecolors=out_col, linewidths=0.9,
                     zorder=zorder + 0.3)
    if mean_c:
        x, y = p.xy(np.array(mean_c), np.array(mean_v))
        p.ax.scatter(x, y, s=7.0 ** 2, marker="+", color=p.ink, linewidths=1.3, zorder=Z_SUMMARY)


def draw_violins(p: Plot, cells: list[Cell], *, inner: str = "box", scale: str = "width", trim: bool = True,
                 bw_adjust: float = 1.0, width_frac: float = 0.8, alpha: float = 0.6, outline: str = "match",
                 outline_width: float = 1.0, side: int = 0, offset: float = 0.0):
    """Violins (``side`` 0) or half violins (``side`` ±1, growing towards that
    side of ``pos + offset``)."""
    cells = [c for c in cells if len(c.values)]
    if not cells:
        return
    lw = max(outline_width, 0.0) if outline != "none" else 0.0
    curves = {c.index: kde_curve(c.values, bw_adjust, trim) for c in cells}
    peak = max((k[1].max() for k in curves.values() if k is not None), default=1.0) or 1.0
    nmax = max(len(c.values) for c in cells)
    shapes, faces, edges = [], [], []
    Q = {"c0": [], "v0": [], "c1": [], "v1": [], "col": [], "ls": []}
    flat = {"c0": [], "v0": [], "c1": [], "v1": [], "col": []}
    inner_cells = []
    for c in cells:
        k = curves[c.index]
        full = c.slot * width_frac * (1.0 if side else 0.5)
        centre = c.pos + offset
        ec = edge_color(outline, c.color, p.ink)
        if k is None:                                # no spread: a flat line at the value
            v = float(c.values[0])
            flat["c0"].append(centre - (0 if side > 0 else full) * 0.5)
            flat["c1"].append(centre + (0 if side < 0 else full) * 0.5)
            flat["v0"].append(v); flat["v1"].append(v); flat["col"].append(c.color)
            c.raise_top(v)
            continue
        grid, dens = k
        if scale == "width":
            hw = dens / dens.max() * full
        else:
            hw = dens / peak * full * (len(c.values) / nmax if scale == "count" else 1.0)
        left = centre - hw if side <= 0 else np.full_like(hw, centre)
        right = centre + hw if side >= 0 else np.full_like(hw, centre)
        shapes.append((np.concatenate([left, right[::-1]]), np.concatenate([grid, grid[::-1]])))
        faces.append(rgba(c.color, alpha))
        edges.append(ec)
        c.raise_top(grid.max())
        if inner == "quartiles":
            qs = np.percentile(c.values, [25, 50, 75])
            for q, ls in zip(qs, [(0, (3, 2)), "-", (0, (3, 2))]):
                h = float(np.interp(q, grid, hw))
                Q["c0"].append(centre - (h if side <= 0 else 0)); Q["c1"].append(centre + (h if side >= 0 else 0))
                Q["v0"].append(q); Q["v1"].append(q); Q["ls"].append(ls)
                Q["col"].append(p.ink if outline == "black" else darken(c.color, 0.45))
        elif inner in ("box", "mean_sd"):
            inner_cells.append(c)
    p.polys(shapes, facecolors=faces, edgecolors=edges if lw > 0 else "none", linewidths=lw, zorder=Z_FILL,
            joinstyle="round")
    if flat["c0"]:
        p.segments(flat["c0"], flat["v0"], flat["c1"], flat["v1"], colors=flat["col"], linewidths=max(lw, 1.2),
                   zorder=Z_FILL)
    if Q["c0"]:
        p.segments(Q["c0"], Q["v0"], Q["c1"], Q["v1"], colors=Q["col"], linewidths=max(lw, 1.0) * 1.1,
                   linestyles=Q["ls"], zorder=Z_FILL + 0.2)
    if inner_cells and side == 0:
        pos = np.array([c.pos + offset for c in inner_cells])
        dark = [p.ink if outline == "black" else darken(c.color, 0.6) for c in inner_cells]
        k = max(p.fonts["base"] / 11, 0.7)
        if inner == "box":
            st = [box_stats(c.values, "tukey") for c in inner_cells]
            p.segments(pos, [s["lo"] for s in st], pos, [s["hi"] for s in st], colors=dark, linewidths=1.1 * k,
                       zorder=Z_FILL + 0.2)
            p.segments(pos, [s["q1"] for s in st], pos, [s["q3"] for s in st], colors=dark, linewidths=4.2 * k,
                       zorder=Z_FILL + 0.3)
            mid = [s["med"] for s in st]
        else:
            mean = np.array([c.values.mean() for c in inner_cells])
            sd = np.array([c.values.std(ddof=1) if len(c.values) > 1 else 0.0 for c in inner_cells])
            p.segments(pos, mean - sd, pos, mean + sd, colors=dark, linewidths=1.6 * k, zorder=Z_FILL + 0.2)
            mid = mean
        x, y = p.xy(pos, np.asarray(mid, dtype=float))
        p.ax.scatter(x, y, s=(3.4 * k) ** 2, marker="o", facecolors="#FFFFFF", edgecolors=dark,
                     linewidths=0.8 * k, zorder=Z_FILL + 0.4)


# ----------------------------------------------------------------- shared
def _brackets(p: Plot, cells: list[Cell], grouped: bool):
    brs = p.computed.get("brackets") or []
    for c in cells:
        p.note_top(c.pos, c.top)
    if not brs:
        return
    if grouped:
        where = {(c.x, c.key): c.pos for c in cells}
        spans = [(where.get((str(b.get("x")), str(b["g1"]))), where.get((str(b.get("x")), str(b["g2"]))),
                  b.get("label")) for b in brs]
    else:
        where = {c.key: c.pos for c in cells}
        spans = [(where.get(str(b["g1"])), where.get(str(b["g2"])), b.get("label")) for b in brs]
    p.add_brackets([s for s in spans if s[0] is not None and s[1] is not None and s[2]])


def _wrap_up(p: Plot, cells: list[Cell], labels: list[str], *, zero: bool = False, grouped: bool = False,
             brackets: bool = True, pad: float = 0.05):
    p.cat_axis(labels)
    p.value_axis(zero=zero, pad=pad)
    p.axis_titles()
    if brackets:
        _brackets(p, cells, grouped)


def _legend(p: Plot, keys: list[str], kind: str, *, alpha: float = 1.0, auto: str | None = None, **kw):
    """One legend entry per colour key."""
    for k in keys:
        col = p.color(k)
        if kind == "patch":
            outline = kw.get("outline", "match")
            lw = kw.get("outline_width", 1.0) if outline != "none" else 0.0
            p.lg_patch(k, col, alpha, ec=edge_color(outline, col, p.ink) if lw > 0 else "none", lw=min(lw, 1.5))
        elif kind == "point":
            p.lg_point(k, point_style(p.opt("point_shape", "circle"), col, min(p.num("point_size", 5), 9),
                                      1.0, p.opt("point_color", "group"), ink=p.ink))
        else:
            p.lg_line(k, col, kw.get("lw", 1.5), st=kw.get("st", {}).get(k))
    p.legend_auto = auto


def _show_points(p: Plot, default: bool = True) -> bool:
    return bool(p.o.get("show_points", default))


# ================================================================== COLUMN
def column_scatter(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    draw_points(p, cells, p.opt("point_layout", "swarm"), half=p.num("jitter", 0.18))
    if p.opt("show_summary", True):
        lw = p.num("error_width", 1.0)
        draw_center_lines(p, cells, p.num("summary_width", 0.5), max(lw * 1.5, 1.2), color=p.ink)
        draw_errors(p, cells, p.num("error_cap", 0.25), lw, color=p.ink)
    _legend(p, groups, "point", auto="none")
    _wrap_up(p, cells, groups)


def column_bar(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    alpha, outline, olw = p.num("fill_alpha", 1.0), p.opt("outline", "match"), p.num("outline_width", 1.0)
    base = p.bar_base([c.center for c in cells] + [c.lo for c in cells])
    draw_bars(p, cells, base, p.num("bar_width", 0.65), alpha, outline, olw)
    draw_errors(p, cells, p.num("error_cap", 0.25), p.num("error_width", 1.0), color=p.ink,
                direction=p.opt("error_direction", "both"))
    if _show_points(p):
        draw_points(p, cells, p.opt("point_layout", "swarm"),
                    half=min(p.num("jitter", 0.18), p.num("bar_width", 0.65) / 2), on_fill=alpha > 0.35)
    _legend(p, groups, "patch", alpha=alpha, outline=outline, outline_width=olw, auto="none")
    _wrap_up(p, cells, groups, zero=True)


def box(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    pts = _show_points(p, False)
    alpha, outline, olw = p.num("fill_alpha", 0.6), p.opt("outline", "match"), p.num("outline_width", 1.0)
    draw_boxes(p, cells, whiskers=p.opt("whiskers", "tukey"), outliers=bool(p.opt("show_outliers", True)) and not pts,
               mean=bool(p.opt("show_mean", False)), notch=bool(p.opt("notch", False)),
               width_frac=p.num("bar_width", 0.6), alpha=alpha, outline=outline, outline_width=olw)
    if pts:
        draw_points(p, cells, p.opt("point_layout", "swarm"),
                    half=min(p.num("jitter", 0.18), p.num("bar_width", 0.6) / 2), on_fill=alpha > 0.35)
    _legend(p, groups, "patch", alpha=alpha, outline=outline, outline_width=olw, auto="none")
    _wrap_up(p, cells, groups)


def violin(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    alpha, outline, olw = p.num("fill_alpha", 0.6), p.opt("outline", "match"), p.num("outline_width", 1.0)
    draw_violins(p, cells, inner=p.opt("violin_inner", "box"), scale=p.opt("violin_scale", "width"),
                 trim=bool(p.opt("violin_trim", True)), bw_adjust=p.num("bw_adjust", 1.0),
                 width_frac=p.num("bar_width", 0.8), alpha=alpha, outline=outline, outline_width=olw)
    if _show_points(p, False):
        draw_points(p, cells, p.opt("point_layout", "swarm"),
                    half=min(p.num("jitter", 0.18), p.num("bar_width", 0.8) / 2), on_fill=alpha > 0.35,
                    zorder=Z_FILL + 0.15)
    _legend(p, groups, "patch", alpha=alpha, outline=outline, outline_width=olw, auto="none")
    _wrap_up(p, cells, groups)


def raincloud(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    alpha = p.num("fill_alpha", 0.6)
    s = -1 if p.flip else 1                          # the cloud points up on horizontal graphs
    draw_violins(p, cells, inner="none", scale="width", trim=False, bw_adjust=p.num("bw_adjust", 1.0),
                 width_frac=0.42, alpha=alpha, outline="match", outline_width=1.0, side=s, offset=0.07 * s)
    if p.opt("show_box", True):
        draw_boxes(p, cells, whiskers="tukey", outliers=False, width_frac=0.09, alpha=min(alpha + 0.2, 1.0),
                   outline="match", outline_width=1.0, caps=False, zorder=Z_LINE)
    draw_points(p, cells, "jitter", half=0.10, offset=-0.20 * s)
    _legend(p, groups, "patch", alpha=alpha, auto="none")
    _wrap_up(p, cells, groups)


def before_after(p: Plot):
    p.ref_lines(categorical=True)
    cells, groups = column_cells(p)
    main = p.frames["main"]
    mode = p.opt("line_color", "gray")
    if "subject" in main.columns and len(groups) > 1:
        wide = main.assign(group=main["group"].astype(str)).pivot_table(
            index="subject", columns="group", values="value", aggfunc="first", observed=False)
        m = wide.reindex(columns=groups).to_numpy(dtype=float)
        k = len(groups)
        c0 = np.tile(np.arange(k - 1, dtype=float), (len(m), 1))
        v0, v1 = m[:, :-1], m[:, 1:]
        if mode == "direction":
            neutral = "#9A9A9A"
            cols = np.where(v1 > v0, UP_COLOR, np.where(v1 < v0, DOWN_COLOR, neutral)).ravel().tolist()
        else:
            cols = [p.ink if mode == "black" else mix("#8C8C8C", p.panel, 0.1)] * v0.size
        p.segments(c0.ravel(), v0.ravel(), c0.ravel() + 1, v1.ravel(), colors=cols,
                   linewidths=p.num("line_width", 1.0), alpha=p.num("line_alpha", 0.6), zorder=Z_LINE,
                   capstyle="round")
    draw_points(p, cells, "aligned")
    if p.opt("show_summary", False):
        lw = p.num("error_width", 1.0)
        draw_center_lines(p, cells, 0.34, max(lw * 1.6, 1.6), color=p.ink)
        draw_errors(p, cells, p.num("error_cap", 0.25) * 0.7, lw, color=p.ink)
    _legend(p, groups, "point", auto="none")
    if mode == "direction" and len(groups) > 1:
        p.legend_items = []
        p.lg_line("Increase", UP_COLOR, max(p.num("line_width", 1.0), 1.2))
        p.lg_line("Decrease", DOWN_COLOR, max(p.num("line_width", 1.0), 1.2))
        p.legend_auto = "right"
    _wrap_up(p, cells, groups)


# ================================================================= GROUPED
def grouped_bar(p: Plot):
    p.ref_lines(categorical=True)
    stacked = p.opt("position", "dodge") == "stack"
    dodge = p.num("dodge", 0.8)
    cells, xs, gs = grouped_cells(p, dodge, stacked)
    alpha, outline, olw = p.num("fill_alpha", 1.0), p.opt("outline", "match"), p.num("outline_width", 1.0)
    wf = p.num("bar_width", 0.9)
    if stacked:
        pos_top: dict = {}
        neg_top: dict = {}
        ok = [c for c in cells if np.isfinite(c.center)]
        bases = []
        for c in ok:
            acc = pos_top if c.center >= 0 else neg_top
            bases.append(acc.get(c.x, 0.0))
            acc[c.x] = bases[-1] + c.center
        if ok:
            lw = olw if outline != "none" else 0.0
            tops = np.array(bases) + np.array([c.center for c in ok])
            p.bars([c.pos for c in ok], tops, [c.slot * wf for c in ok], base=np.array(bases),
                   color=[rgba(c.color, alpha) for c in ok],
                   edgecolor=[edge_color(outline, c.color, p.ink) for c in ok] if lw > 0 else "none",
                   linewidth=lw, zorder=Z_FILL)
    else:
        base = p.bar_base([c.center for c in cells] + [c.lo for c in cells])
        draw_bars(p, cells, base, wf, alpha, outline, olw)
        draw_errors(p, cells, p.num("error_cap", 0.25), p.num("error_width", 1.0), color=p.ink)
        if _show_points(p):
            draw_points(p, cells, "swarm", on_fill=alpha > 0.35)
    _legend(p, gs, "patch", alpha=alpha, outline=outline, outline_width=olw)
    _wrap_up(p, cells, xs, zero=True, grouped=True, brackets=not stacked)


def grouped_box(p: Plot):
    p.ref_lines(categorical=True)
    cells, xs, gs = grouped_cells(p, p.num("dodge", 0.8))
    pts = _show_points(p, False)
    alpha, outline, olw = p.num("fill_alpha", 0.6), p.opt("outline", "match"), p.num("outline_width", 1.0)
    draw_boxes(p, cells, whiskers=p.opt("whiskers", "tukey"), outliers=bool(p.opt("show_outliers", True)) and not pts,
               width_frac=p.num("bar_width", 0.85), alpha=alpha, outline=outline, outline_width=olw)
    if pts:
        draw_points(p, cells, "swarm", on_fill=alpha > 0.35)
    _legend(p, gs, "patch", alpha=alpha, outline=outline, outline_width=olw)
    _wrap_up(p, cells, xs, grouped=True)


def grouped_violin(p: Plot):
    p.ref_lines(categorical=True)
    cells, xs, gs = grouped_cells(p, p.num("dodge", 0.8))
    alpha, outline, olw = p.num("fill_alpha", 0.6), p.opt("outline", "match"), p.num("outline_width", 1.0)
    draw_violins(p, cells, inner=p.opt("violin_inner", "box"), bw_adjust=p.num("bw_adjust", 1.0),
                 width_frac=p.num("bar_width", 0.9), alpha=alpha, outline=outline, outline_width=olw)
    if _show_points(p, False):
        draw_points(p, cells, "swarm", half=None, on_fill=alpha > 0.35, zorder=Z_FILL + 0.15)
    _legend(p, gs, "patch", alpha=alpha, outline=outline, outline_width=olw)
    _wrap_up(p, cells, xs, grouped=True)


def grouped_points(p: Plot):
    p.ref_lines(categorical=True)
    cells, xs, gs = grouped_cells(p, p.num("dodge", 0.8))
    half = p.num("jitter", 0.08)
    draw_points(p, cells, "swarm" if half > 0 else "aligned", half=half)
    if p.opt("show_summary", True):
        lw = p.num("error_width", 1.0)
        draw_center_lines(p, cells, 0.7, max(lw * 1.5, 1.2), color=p.ink)
        draw_errors(p, cells, p.num("error_cap", 0.25), lw, color=p.ink)
    _legend(p, gs, "point")
    _wrap_up(p, cells, xs, grouped=True)


def interaction_line(p: Plot):
    p.ref_lines(categorical=True)
    cells, xs, gs = grouped_cells(p, 1.0)
    ng = max(len(gs), 1)
    spread = p.num("dodge", 0.15)
    offs = {g: (j - (ng - 1) / 2) * (spread / max(ng - 1, 1)) for j, g in enumerate(gs)}
    xi = {x: i for i, x in enumerate(xs)}
    for c in cells:                                  # all groups share the category centre, plus a small offset
        c.pos, c.slot = xi[c.x] + offs[c.key], 0.4
    lw = p.num("line_width", 1.5)
    styles = {}
    for g in gs:
        mine = sorted((c for c in cells if c.key == g and np.isfinite(c.center)), key=lambda c: c.pos)
        col = p.color(g)
        st = point_style(p.opt("point_shape", "circle"), col, p.num("point_size", 6), 1.0,
                         p.opt("point_color", "group"), ink=p.ink)
        styles[g] = st
        if mine:
            p.line([c.pos for c in mine], [c.center for c in mine], color=col, linewidth=lw, zorder=Z_LINE,
                   solid_joinstyle="round")
            draw_errors(p, mine, p.num("error_cap", 0.25), p.num("error_width", 1.0))
            p.scatter([c.pos for c in mine], [c.center for c in mine], st, zorder=Z_SUMMARY + 0.1)
            for c in mine:
                c.raise_top(c.center)
    if p.opt("show_points", False):
        draw_points(p, cells, "swarm", half=min(0.06, 0.4 / ng), size=max(p.num("point_size", 6) * 0.55, 1.5),
                    alpha=min(p.num("point_alpha", 0.9), 0.55), zorder=Z_LINE - 0.2)
    for g in gs:
        p.lg_line(g, p.color(g), lw, st=styles[g])
    _wrap_up(p, cells, xs, grouped=True, brackets=False)


DRAWERS = {"column_scatter": column_scatter, "column_bar": column_bar, "box": box, "violin": violin,
           "raincloud": raincloud, "before_after": before_after, "grouped_bar": grouped_bar,
           "grouped_box": grouped_box, "grouped_violin": grouped_violin, "grouped_points": grouped_points,
           "interaction_line": interaction_line}
