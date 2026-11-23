"""Shared machinery of the matplotlib engine.

:class:`Plot` owns one figure for one render. ``begin()`` builds the themed
figure/axes, the type-specific drawer adds its artists, ``finish()`` builds the
legend and lays everything out, ``save()`` encodes the image.

Layout is done by hand instead of with constrained layout: it is faster, it is
deterministic, and it lets us reserve exact room for the things journals care
about (outside legends, significance brackets, risk tables, colour bars).
The algorithm: reserve header / caption / legend strips, then repeatedly place
the axes and measure their tight bounding box until the decorations fit.

Geometry that depends on the final size of the axes in pixels (beeswarms,
bracket stacking, automatic tick rotation, label de-overlapping) is registered
as a *hook* and runs between layout passes:

    phase 0   may change the size of axis decorations a lot (tick rotation)
    phase 1   may change axis limits (significance brackets, label fitting)
    phase 2   only draws inside the axes (beeswarm points, point labels)
"""
from __future__ import annotations

import io
import math
import warnings

import matplotlib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox

from ...core.errors import AnalysisError

FALLBACK_COLOR = "#3B6FE0"
Z_REF, Z_BAND, Z_FILL, Z_LINE, Z_POINT, Z_SUMMARY, Z_TEXT = 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0


# ------------------------------------------------------------------ colours
def rgb(c) -> np.ndarray:
    try:
        return np.array(mcolors.to_rgb(c))
    except ValueError:
        return np.array(mcolors.to_rgb(FALLBACK_COLOR))


def darken(c, amount: float = 0.35) -> str:
    """Mix a colour with black (amount 0–1)."""
    return mcolors.to_hex(rgb(c) * (1 - amount))


def lighten(c, amount: float = 0.5) -> str:
    """Mix a colour with white (amount 0–1)."""
    v = rgb(c)
    return mcolors.to_hex(v + (1 - v) * amount)


def mix(c1, c2, t: float) -> str:
    return mcolors.to_hex(rgb(c1) * (1 - t) + rgb(c2) * t)


def rgba(c, alpha: float = 1.0) -> tuple:
    return (*rgb(c), float(alpha))


def luminance(c) -> float:
    r, g, b = rgb(c)
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_text(bg) -> str:
    """Black or white, whichever reads better on ``bg``."""
    return "#111111" if luminance(bg) > 0.58 else "#FFFFFF"


# ------------------------------------------------------------------ markers
_MARKERS = {"circle": ("o", 1.0), "square": ("s", 0.88), "triangle": ("^", 1.08), "diamond": ("D", 0.76)}
SHAPE_CYCLE = ["circle", "square", "triangle", "diamond", "circle_open", "square_open", "triangle_open"]
LINE_CYCLE = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (5, 2, 1, 2)), (0, (8, 2, 2, 2))]
LINE_STYLES = {"solid": "-", "dashed": (0, (5, 2)), "dotted": (0, (1, 1.6))}


def point_style(shape: str, color, size: float, alpha: float = 1.0, point_color: str = "group",
                ink: str = "#111111", on_fill: bool = False) -> dict:
    """Resolve the point options into concrete marker properties.

    ``size`` is the marker diameter in points. ``on_fill`` marks points drawn
    on top of a filled shape of the same colour (bars, boxes): they get a
    darker rim so they stay visible.
    """
    shape = shape or "circle"
    is_open = shape.endswith("_open")
    marker, k = _MARKERS.get(shape.replace("_open", ""), _MARKERS["circle"])
    d = max(float(size), 0.1) * k
    base = {"group": color, "dark": darken(color, 0.4), "black": ink, "gray": "#8C8C8C",
            "white": color}.get(point_color, color)
    rim = max(0.5, min(1.5, d / 6))
    if is_open:
        fc, ec, lw = "none", base, max(0.8, min(1.8, d / 4.5))
    elif point_color == "white":
        fc, ec, lw = "#FFFFFF", color, max(0.8, min(1.6, d / 5))
    elif on_fill:
        fc, ec, lw = base, darken(base, 0.55), rim
    else:
        fc, ec, lw = base, "none", 0.0
    return {"marker": marker, "d": d, "fc": fc, "ec": ec, "lw": lw, "alpha": float(alpha)}


def line_marker_kw(st: dict) -> dict:
    """``point_style`` result as Line2D keyword arguments."""
    return {"marker": st["marker"], "markersize": st["d"], "markerfacecolor": st["fc"],
            "markeredgecolor": st["ec"], "markeredgewidth": st["lw"]}


# ------------------------------------------------------------------- scales
def scale_fwd(scale: str):
    return {"log10": np.log10, "log2": np.log2,
            "sqrt": lambda v: np.sqrt(np.maximum(v, 0.0))}.get(scale, lambda v: np.asarray(v, dtype=float))


def scale_inv(scale: str):
    return {"log10": lambda t: np.power(10.0, t), "log2": lambda t: np.power(2.0, t),
            "sqrt": lambda t: np.square(np.maximum(t, 0.0))}.get(scale, lambda t: np.asarray(t, dtype=float))


def _plain(v: float) -> str:
    return format(v, ".10g").replace("-", "−")


def _log_formatter(base: int, lo: float, hi: float):
    plain = (lo >= 0.9e-4 and hi <= 1.1e6) if base == 10 else (lo >= 1 / 2048 and hi <= 70000)

    def fmt(v, _pos=None):
        if not v > 0:
            return ""
        if plain:
            return _plain(v)
        e = math.log(v, base)
        if abs(e - round(e)) < 1e-9:
            return rf"$\mathdefault{{{base}^{{{int(round(e))}}}}}$"
        e10 = math.floor(math.log10(v))
        return rf"$\mathdefault{{{v / 10 ** e10:g}{{\times}}10^{{{e10}}}}}$"
    return mticker.FuncFormatter(fmt)


# -------------------------------------------------------------- geometry
def beeswarm(v_px: np.ndarray, d_px: float, max_half_px: float) -> np.ndarray:
    """Offsets (px) across the category axis that make a non-overlapping swarm.

    Points are visited in order of value; each one takes the free position
    closest to the centre line among "centre" and "touching an already placed
    neighbour". If the swarm becomes wider than ``max_half_px`` it is squeezed
    (points then overlap sideways, like every compact beeswarm does).
    """
    n = len(v_px)
    out = np.zeros(n)
    if n < 2 or d_px <= 0:
        return out
    order = np.argsort(v_px, kind="stable")
    y = np.asarray(v_px, dtype=float)[order]
    x = np.zeros(n)
    d2 = d_px * d_px
    start = 0
    for i in range(1, n):
        while y[i] - y[start] >= d_px:
            start += 1
        if start == i:
            continue
        nx, dy2 = x[start:i], (y[i] - y[start:i]) ** 2
        off = np.sqrt(np.maximum(d2 - dy2, 0.0))
        cands = np.concatenate(([0.0], nx - off, nx + off))
        cands = cands[np.argsort(np.abs(cands), kind="stable")]
        ok = ((cands[:, None] - nx[None, :]) ** 2 + dy2[None, :] >= d2 - 1e-6).all(axis=1)
        x[i] = cands[int(np.argmax(ok))] if ok.any() else 0.0
    m = np.abs(x).max()
    if m > max_half_px > 0:
        x *= max_half_px / m
    elif max_half_px <= 0:
        x[:] = 0
    out[order] = x
    return out


def kde_curve(vals: np.ndarray, adjust: float = 1.0, trim: bool = True, cut: float = 2.0, n: int = 128):
    """Gaussian KDE (Scott bandwidth × ``adjust``) on a grid: ``(grid, density)``,
    or ``None`` when the values have no spread."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return None
    sd = v.std(ddof=1)
    if not sd > 0:
        return None
    bw = sd * len(v) ** (-0.2) * max(float(adjust), 0.05)
    lo, hi = (v.min(), v.max()) if trim else (v.min() - cut * bw, v.max() + cut * bw)
    grid = np.linspace(lo, hi, n)
    src = v if len(v) <= 4000 else np.quantile(v, np.linspace(0, 1, 4000))
    z = (grid[:, None] - src[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (len(src) * bw * math.sqrt(2 * math.pi))
    return grid, dens


# ---------------------------------------------------------------- rc params
_RC_BASE: dict | None = None
_RC_SKIP = {"backend", "backend_fallback", "interactive", "toolbar", "figure.raise_window", "tk.window_focus",
            "webagg.port", "webagg.address", "webagg.port_retries", "webagg.open_in_browser", "timezone"}


def build_rc(fonts: dict, theme: dict, editable: bool) -> dict:
    """A complete rc dict: matplotlib's factory defaults plus our style, so the
    result never depends on what other code did to the global rcParams."""
    global _RC_BASE
    if _RC_BASE is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RC_BASE = {k: v for k, v in matplotlib.rcParamsDefault.items()
                        if k not in _RC_SKIP and not k.startswith("_")}
    family = fonts.get("family") or "DejaVu Sans"
    rc = dict(_RC_BASE)
    rc.update({
        "font.family": [family, "DejaVu Sans"], "font.size": fonts["base"],
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.default": "regular",
        "mathtext.cal": family, "mathtext.sf": family, "mathtext.tt": family,
        "axes.unicode_minus": True, "axes.formatter.use_mathtext": True,
        "svg.fonttype": "none" if editable else "path", "pdf.fonttype": 42 if editable else 3,
        "svg.hashsalt": "analysis-ui",
        "text.color": theme["text"], "axes.labelcolor": theme["text"], "axes.edgecolor": theme["axis"],
        "xtick.color": theme["axis"], "ytick.color": theme["axis"],
        "xtick.labelcolor": theme["text"], "ytick.labelcolor": theme["text"],
        "figure.dpi": 100, "figure.facecolor": theme["bg"], "axes.facecolor": theme["panel_bg"],
        "savefig.facecolor": "auto", "savefig.edgecolor": "none",
        "legend.frameon": False, "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.0, "hatch.linewidth": 0.6, "path.simplify": True,
    })
    return rc


# ====================================================================== Plot
class Plot:
    def __init__(self, payload: dict):
        self.payload = payload
        self.type = payload["type"]
        self.o = o = payload["options"]
        self.frames = payload.get("frames") or {}
        self.computed = payload.get("computed") or {}
        self.theme, self.fonts, self.labels = o["_theme"], o["_fonts"], o["_labels"]
        self.colors = dict(zip([str(n) for n in o.get("_color_names", [])], o.get("_color_values", [])))
        self.ink, self.axis_color = self.theme["text"], self.theme["axis"]
        self.bg, self.panel = self.theme["bg"], self.theme["panel_bg"]
        self.flip = bool(o.get("flip"))
        self.W = min(max(float(o.get("width") or 5.0), 1.0), 60.0)
        self.H = min(max(float(o.get("height") or 4.0), 1.0), 60.0)
        self.dpi = 100.0
        self.px = self.dpi / 72.0                    # pixels per typographic point
        self.fig = Figure(figsize=(self.W, self.H), dpi=self.dpi, facecolor=self.bg)
        self.canvas = FigureCanvasAgg(self.fig)
        self._renderer = None
        self.axes: list = []                         # every axes taking part in the layout
        self.ax = None
        self.scales = {"x": "linear", "y": "linear"}
        self.legend_items: list[tuple] = []          # (handle, label)
        self.legend_auto: str | None = None          # what "auto" means; None = by item count
        self.legend_kw: dict = {}
        self.legend = None
        self.legend_pos = "none"
        self.hooks: tuple[list, list, list] = ([], [], [])
        self.placer = None                           # callable(l, b, r, t) in px -> positions the axes
        self.title_span = "axes"
        self.tops: list[tuple[float, float]] = []    # (category position, highest drawn value)
        self._texts: dict = {}
        self._inner = None

    # ------------------------------------------------------------ small helpers
    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = self.canvas.get_renderer()
        return self._renderer

    def color(self, key) -> str:
        return self.colors.get(str(key)) or FALLBACK_COLOR

    def opt(self, name, default=None):
        v = self.o.get(name)
        return default if v is None else v

    def num(self, name, default: float) -> float:
        try:
            v = float(self.o.get(name))
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def xy(self, c, v):
        """(category, value) -> (x, y), honouring ``flip``."""
        return (v, c) if self.flip else (c, v)

    @property
    def vname(self) -> str:
        """Which displayed axis carries the values of a categorical graph."""
        return "x" if self.flip else "y"

    @property
    def cname(self) -> str:
        return "y" if self.flip else "x"

    def text_size(self, text: str, size: float, bold: bool = False) -> tuple[float, float]:
        """Width and height of a (possibly multi-line) string in pixels."""
        from matplotlib.font_manager import FontProperties
        fp = FontProperties(size=size, weight="bold" if bold else "normal")
        w = h = 0.0
        for line in str(text).split("\n"):
            lw, lh, _ = self.renderer.get_text_width_height_descent(line or " ", fp, False)
            w, h = max(w, lw), h + max(lh, size * self.px * 1.0) * 1.15
        return w, h

    # ---------------------------------------------------------------- begin
    def begin(self):
        f, th = self.fonts, self.theme
        fig = self.fig
        ha = "center" if th.get("title_align") == "center" else "left"
        if self.labels.get("title"):
            self._texts["title"] = fig.text(0.5, 1, self.labels["title"], ha=ha, va="top", fontsize=f["title"],
                                            fontweight="bold" if th.get("title_bold") else "normal", color=self.ink)
        if self.labels.get("subtitle"):
            self._texts["subtitle"] = fig.text(0.5, 1, self.labels["subtitle"], ha=ha, va="top",
                                               fontsize=f["subtitle"], color=mix(self.ink, self.bg, 0.25))
        if self.labels.get("caption"):
            self._texts["caption"] = fig.text(0, 0, self.labels["caption"], ha="left", va="bottom",
                                              fontsize=f["caption"], color=mix(self.ink, self.bg, 0.3))
        self.ax = self.new_axes()
        self.placer = self._place_single
        return self

    def new_axes(self, **kw):
        ax = self.fig.add_axes([0.16, 0.16, 0.74, 0.7], **kw)
        self.style_axes(ax)
        self.axes.append(ax)
        return ax

    def style_axes(self, ax):
        th, f = self.theme, self.fonts
        ax.set_facecolor(self.panel)
        ax.set_axisbelow(True)
        lw = float(th.get("axis_width") or 0)
        frame = th.get("frame", "L")
        visible = {"L": ("left", "bottom"), "box": ("left", "bottom", "top", "right")}.get(frame, ())
        for side, sp in ax.spines.items():
            sp.set_visible(side in visible and lw > 0)
            sp.set_linewidth(lw)
            sp.set_color(self.axis_color)
            sp.set_capstyle("projecting")
        tick_dir = th.get("tick_dir", "out")
        k = max(f["base"] / 11.0, 0.6)
        length = 0 if tick_dir == "none" else 4.0 * k
        tw = lw if lw > 0 else 0.6
        common = dict(direction="out" if tick_dir == "none" else tick_dir, color=self.axis_color,
                      labelcolor=self.ink, labelsize=f["tick"], pad=3.0 * k)
        ax.tick_params(axis="both", which="major", length=length, width=tw, **common)
        ax.tick_params(axis="both", which="minor", length=length * 0.55, width=tw * 0.75, **common)
        grid = th.get("grid", "none")
        if self.flip and grid in ("x", "y"):         # grid lines follow the value axis
            grid = "x" if grid == "y" else "y"
        ax.grid(False)
        if grid in ("x", "y", "both"):
            ax.grid(True, axis=grid, which="major", color=th.get("grid_color", "#E6E6E6"),
                    linewidth=max(0.5, 0.75 * k), zorder=0)
        weight = "bold" if th.get("axis_title_bold") else "normal"
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set(fontsize=f["axis_title"], fontweight=weight, color=self.ink)
            axis.labelpad = 4.0 * k
            axis.get_offset_text().set(fontsize=f["tick"], color=self.ink)

    def axis_titles(self, x: str | None = None, y: str | None = None, ax=None):
        """Axis titles from ``_labels`` (x = category/horizontal role, y = value
        role; they swap sides on flipped graphs)."""
        ax = ax or self.ax
        x = self.labels.get("x", "") if x is None else x
        y = self.labels.get("y", "") if y is None else y
        if self.flip:
            x, y = y, x
        ax.set_xlabel(x or "")
        ax.set_ylabel(y or "")

    def ref_lines(self, categorical: bool = False, ax=None):
        """User reference lines. ``hlines`` are positions on the value (Y)
        axis, ``vlines`` on the X axis (1-based slots on categorical axes)."""
        ax = ax or self.ax
        kw = dict(color=self.axis_color, linewidth=max(0.6, 0.8 * self.fonts["base"] / 11), linestyle=(0, (4, 3)),
                  zorder=Z_REF)
        off = 1.0 if categorical else 0.0
        for v in self.o.get("_hlines") or []:
            (ax.axvline if self.flip else ax.axhline)(v, **kw)
        for v in self.o.get("_vlines") or []:
            (ax.axhline if self.flip else ax.axvline)(v - off, **kw)

    # ------------------------------------------------------------ axis set-up
    def cat_axis(self, labels: list[str], ax=None, pad: float = 0.6, which: str | None = None,
                 invert: bool | None = None, auto_angle: bool = True):
        """Categorical axis with one slot per label at 0..n-1."""
        ax = ax or self.ax
        which = which or self.cname
        n = max(len(labels), 1)
        ticks = np.arange(len(labels))
        if which == "x":
            ax.set_xlim(-pad, n - 1 + pad)
            ax.set_xticks(ticks, [str(l) for l in labels])
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            angle = str(self.o.get("x_tick_angle") or "auto")
            if angle == "auto":
                if auto_angle:
                    self.hooks[0].append(lambda: self._auto_angle(ax))
            else:
                self.rotate_xticks(ax, float(angle))
        else:
            invert = True if invert is None else invert
            ax.set_ylim((n - 1 + pad, -pad) if invert else (-pad, n - 1 + pad))
            ax.set_yticks(ticks, [str(l) for l in labels])
            ax.yaxis.set_minor_locator(mticker.NullLocator())

    @staticmethod
    def rotate_xticks(ax, angle: float):
        for t in ax.get_xticklabels():
            if angle == 0:
                t.set(rotation=0, ha="center", va="top", rotation_mode="default")
            elif angle >= 90:
                t.set(rotation=90, ha="right", va="center", rotation_mode="anchor")
            else:
                t.set(rotation=angle, ha="right", va="top", rotation_mode="anchor")

    def _auto_angle(self, ax):
        """Rotate category labels only when they would run into each other."""
        labels = [t for t in ax.get_xticklabels() if t.get_text()]
        if len(labels) < 2:
            return
        ticks = ax.get_xticks()
        x0, x1 = ax.get_xlim()
        step = float(np.min(np.diff(np.sort(ticks)))) if len(ticks) > 1 else 1.0
        slot = ax.bbox.width * step / max(abs(x1 - x0), 1e-9)
        fs = labels[0].get_fontsize()
        sample = sorted(labels, key=lambda t: -max(len(s) for s in t.get_text().split("\n")))[:6]
        widest = max(self.text_size(t.get_text(), fs)[0] for t in sample)
        if widest + 0.6 * fs * self.px <= slot:
            return
        line_h = fs * self.px * 1.2
        self.rotate_xticks(ax, 45 if line_h * 1.3 <= slot else 90)
        if line_h > slot:                            # even vertical labels collide: shrink, then thin out
            new = max(slot / self.px / 1.2, 4.5)
            keep = max(int(math.ceil(new * self.px * 1.2 / slot)), 1)
            for i, t in enumerate(ax.get_xticklabels()):
                t.set_fontsize(new)
                t.set_visible(i % keep == 0)

    def num_axis(self, which: str, prefix: str | None = None, *, zero: bool = False, pad: float = 0.05,
                 lo: float | None = None, hi: float | None = None, pad_lo: float | None = None,
                 pad_hi: float | None = None, ax=None, data: tuple | None = None, symmetric: bool = False):
        """Configure a numeric axis from the ``{prefix}_min/_max/_scale/_tick_step``
        options and the data limits collected so far.

        which   displayed axis ("x" | "y");  prefix  option prefix (None = no user options)
        zero    bar-like: the axis includes 0 and there is no gap between 0 and the bars
        lo/hi   fixed automatic limits (the user's min/max still win)
        data    explicit (min, max) instead of the axes' data limits
        """
        ax = ax or self.ax
        o = self.o
        scale = (o.get(f"{prefix}_scale") if prefix else None) or "linear"
        umin = o.get(f"{prefix}_min") if prefix else None
        umax = o.get(f"{prefix}_max") if prefix else None
        step = o.get(f"{prefix}_tick_step") if prefix else None
        is_log = scale in ("log10", "log2")
        if data is not None:
            d0, d1 = data
            minpos = d0 if d0 > 0 else np.inf
        elif which == "x":
            (d0, d1), minpos = ax.dataLim.intervalx, ax.dataLim.minposx
        else:
            (d0, d1), minpos = ax.dataLim.intervaly, ax.dataLim.minposy
        if not (np.isfinite(d0) and np.isfinite(d1)):
            d0, d1 = (1.0, 10.0) if is_log else (0.0, 1.0)
        if is_log:
            if d0 <= 0:
                d0 = minpos if np.isfinite(minpos) and minpos > 0 else (d1 / 100 if d1 > 0 else 1.0)
            if d1 <= 0 or d1 < d0:
                d1 = d0 * 10
        fwd, inv = scale_fwd(scale), scale_inv(scale)
        if lo is not None and not (is_log and lo <= 0):
            d0 = lo
        if hi is not None and not (is_log and hi <= 0):
            d1 = hi
        if symmetric and not is_log and scale != "sqrt" and d0 < 0 < d1:
            m = max(-d0, d1)
            d0, d1 = -m, m
        t0, t1 = float(fwd(d0)), float(fwd(d1))
        if not t1 > t0:
            if is_log:
                t0, t1 = t0 - 0.5, t0 + 0.5
            else:
                dv = abs(d0) * 0.1 or 0.5
                t0, t1 = float(fwd(d0 - dv)) if scale != "sqrt" else max(t0 - 0.5, 0), float(fwd(d0 + dv))
                if not t1 > t0:
                    t1 = t0 + 1.0
        span = t1 - t0
        a = t0 - (pad if pad_lo is None else pad_lo) * span
        b = t1 + (pad if pad_hi is None else pad_hi) * span
        if zero and not is_log:
            if d0 >= 0:
                a = 0.0
            if d1 <= 0:
                b = 0.0
        if lo is not None and pad_lo is None and not (is_log and lo <= 0):
            a = float(fwd(lo))
        if hi is not None and pad_hi is None and not (is_log and hi <= 0):
            b = float(fwd(hi))
        if scale == "sqrt":
            a = max(a, 0.0)
        v0, v1 = float(inv(a)), float(inv(b))
        if umin is not None and not (is_log and umin <= 0) and not (scale == "sqrt" and umin < 0):
            v0 = float(umin)
        if umax is not None and not (is_log and umax <= 0):
            v1 = float(umax)
        if not v1 > v0:                              # contradictory user limits: keep the one just set
            if umax is not None and umin is None:
                v0 = v1 / 10 if is_log else v1 - (abs(v1) or 1.0)
            else:
                v1 = v0 * 10 if is_log else v0 + (abs(v0) or 1.0)
        axis = ax.xaxis if which == "x" else ax.yaxis
        set_scale = ax.set_xscale if which == "x" else ax.set_yscale
        if scale == "log10":
            set_scale("log", base=10)
        elif scale == "log2":
            set_scale("log", base=2)
        elif scale == "sqrt":
            set_scale("function", functions=(scale_fwd("sqrt"), scale_inv("sqrt")))
        (ax.set_xlim if which == "x" else ax.set_ylim)(v0, v1)
        if ax is self.ax:
            self.scales[which] = scale
        self._ticks(axis, scale, v0, v1, step)
        if which == "x" and prefix is not None:
            angle = str(o.get("x_tick_angle") or "auto")
            if angle not in ("auto", "0"):
                # tick labels do not exist before the first draw: rotate through tick params
                a = float(angle)
                ax.tick_params(axis="x", labelrotation=a)
                self.hooks[0].append(lambda: self.rotate_xticks(ax, a))
        return v0, v1

    def _ticks(self, axis, scale: str, lo: float, hi: float, step):
        if scale in ("log10", "log2"):
            base = 10 if scale == "log10" else 2
            span = math.log(hi / lo, base)
            minor = mticker.NullLocator()
            if base == 10 and span >= 1.2:
                major = mticker.LogLocator(base=10, subs=(1.0,), numticks=10)
                if span <= 8:
                    minor = mticker.LogLocator(base=10, subs=np.arange(2, 10) * 1.0, numticks=40)
            elif base == 10 and span >= 0.45:
                major = mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=10)
                minor = mticker.LogLocator(base=10, subs=np.arange(1, 10) * 1.0, numticks=40)
            elif base == 2 and span >= 1.5:
                major = mticker.LogLocator(base=2, subs=(1.0,), numticks=10)
            else:
                major = mticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])
            axis.set_major_locator(major)
            axis.set_major_formatter(_log_formatter(base, lo, hi))
            axis.set_minor_locator(minor)
            axis.set_minor_formatter(mticker.NullFormatter())
            return
        try:
            step = float(step) if step else 0.0
        except (TypeError, ValueError):
            step = 0.0
        if step > 0 and (hi - lo) / step <= 60:
            axis.set_major_locator(mticker.MultipleLocator(step))
        else:
            axis.set_major_locator(mticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10]))
        fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        fmt.set_powerlimits((-4, 6))
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(mticker.NullLocator())

    def value_axis(self, **kw):
        """The numeric axis of a categorical graph (options are always ``y_*``)."""
        return self.num_axis(self.vname, "y", **kw)

    def bar_base(self, values) -> float:
        """Where bars start: 0, or a sensible positive floor on log axes."""
        scale = self.o.get("y_scale") or "linear"
        if scale not in ("log10", "log2"):
            return 0.0
        umin = self.o.get("y_min")
        if umin is not None and umin > 0:
            return float(umin)
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if not len(v):
            return 1.0
        base = 10.0 if scale == "log10" else 2.0
        return float(base ** math.floor(math.log(v.min() * 0.8, base)))

    # --------------------------------------------- flip-aware drawing helpers
    def bars(self, c, v, width, base=0.0, ax=None, **kw):
        ax = ax or self.ax
        c, v = np.asarray(c, dtype=float), np.asarray(v, dtype=float)
        base = np.broadcast_to(np.asarray(base, dtype=float), v.shape)
        if self.flip:
            return ax.barh(c, v - base, height=width, left=base, **kw)
        return ax.bar(c, v - base, width=width, bottom=base, **kw)

    def segments(self, c0, v0, c1, v1, ax=None, **kw):
        """Many straight segments (category, value) -> (category, value) as one collection."""
        ax = ax or self.ax
        c0, v0, c1, v1 = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (c0, v0, c1, v1))
        c0, v0, c1, v1 = np.broadcast_arrays(c0, v0, c1, v1)
        ok = np.isfinite(c0) & np.isfinite(v0) & np.isfinite(c1) & np.isfinite(v1)
        if not ok.any():
            return None
        a, b = self.xy(c0[ok], v0[ok]), self.xy(c1[ok], v1[ok])
        segs = np.stack([np.column_stack(a), np.column_stack(b)], axis=1)
        for key in ("colors", "color"):              # per-segment colours must follow the mask
            col = kw.get(key)
            if isinstance(col, (list, np.ndarray)) and len(col) == len(ok) and not isinstance(col, str):
                kw[key] = [cc for cc, k in zip(col, ok) if k]
        kw.setdefault("capstyle", "butt")
        lc = LineCollection(segs, **kw)
        ax.add_collection(lc, autolim=False)
        ax.update_datalim(segs.reshape(-1, 2))
        return lc

    def polys(self, shapes: list, ax=None, **kw):
        """Polygons given as (category array, value array) pairs."""
        ax = ax or self.ax
        verts = [np.column_stack(self.xy(np.asarray(c, dtype=float), np.asarray(v, dtype=float))) for c, v in shapes]
        if not verts:
            return None
        pc = PolyCollection(verts, **kw)
        ax.add_collection(pc, autolim=False)
        ax.update_datalim(np.vstack(verts))
        return pc

    def line(self, c, v, ax=None, **kw):
        return (ax or self.ax).plot(*self.xy(np.asarray(c, dtype=float), np.asarray(v, dtype=float)), **kw)

    def scatter(self, c, v, st: dict, ax=None, zorder=Z_POINT, **kw):
        """Scatter with a ``point_style`` dict (size = marker diameter in pt)."""
        x, y = self.xy(np.asarray(c, dtype=float), np.asarray(v, dtype=float))
        return (ax or self.ax).scatter(x, y, s=st["d"] ** 2, marker=st["marker"], facecolors=st["fc"],
                                       edgecolors=st["ec"], linewidths=st["lw"], alpha=st["alpha"],
                                       zorder=zorder, **kw)

    def note(self, c, v, ax=None):
        """Tell the autoscaler about data that will only be drawn in a hook."""
        c, v = np.atleast_1d(np.asarray(c, dtype=float)), np.atleast_1d(np.asarray(v, dtype=float))
        c, v = np.broadcast_arrays(c, v)
        ok = np.isfinite(c) & np.isfinite(v)
        if ok.any():
            (ax or self.ax).update_datalim(np.column_stack(self.xy(c[ok], v[ok])))

    def errorbars(self, c, center, lo, hi, cap: float, lw: float, color, direction: str = "both",
                  ax=None, zorder=Z_SUMMARY):
        """Error bars along the value axis. ``cap`` is the full cap width in
        category units; NaN ends are simply not drawn."""
        c, center, lo, hi = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (c, center, lo, hi))
        lo_d = center if direction == "up" else np.where(np.isfinite(lo), lo, center)
        hi_d = np.where(np.isfinite(hi), hi, center)
        ok = np.isfinite(center) & (np.isfinite(lo) | np.isfinite(hi))
        if not ok.any():
            return
        cs, ls, hs = c[ok], lo_d[ok], hi_d[ok]
        C0, V0, C1, V1 = [cs], [ls], [cs], [hs]
        if cap > 0:
            up = np.isfinite(hi[ok])
            C0.append(cs[up] - cap / 2); V0.append(hs[up]); C1.append(cs[up] + cap / 2); V1.append(hs[up])
            if direction != "up":
                dn = np.isfinite(lo[ok])
                C0.append(cs[dn] - cap / 2); V0.append(ls[dn]); C1.append(cs[dn] + cap / 2); V1.append(ls[dn])
        self.segments(np.concatenate(C0), np.concatenate(V0), np.concatenate(C1), np.concatenate(V1),
                      ax=ax, colors=color, linewidths=lw, zorder=zorder)

    # --------------------------------------------------------------- legend
    def lg_patch(self, label, color, alpha: float = 1.0, ec="none", lw: float = 0.0):
        self.legend_items.append((Patch(facecolor=rgba(color, alpha), edgecolor=ec, linewidth=lw), str(label)))

    def lg_point(self, label, st: dict):
        self.legend_items.append((Line2D([], [], linestyle="none", alpha=st["alpha"], **line_marker_kw(st)),
                                  str(label)))

    def lg_line(self, label, color, lw: float, ls="-", st: dict | None = None, alpha: float = 1.0):
        kw = line_marker_kw(st) if st else {}
        self.legend_items.append((Line2D([], [], color=color, linewidth=lw, linestyle=ls, alpha=alpha, **kw),
                                  str(label)))

    def _make_legend(self, pos: str, ncols: int):
        f = self.fonts
        handles = [h for h, _ in self.legend_items]
        labels = [l for _, l in self.legend_items]
        has_line = any(isinstance(h, Line2D) and h.get_linestyle() not in ("None", "none") for h in handles)
        kw = dict(fontsize=f["legend"], title=self.labels.get("legend") or None, title_fontsize=f["legend"],
                  frameon=False, borderpad=0.0, borderaxespad=0.0, handlelength=2.1 if has_line else 1.1,
                  handleheight=0.9, handletextpad=0.55, labelspacing=0.5, columnspacing=1.4, ncols=ncols,
                  labelcolor=self.ink, alignment="left")
        kw.update(self.legend_kw)
        if pos.startswith("inside"):
            loc = {"inside_tr": "upper right", "inside_tl": "upper left", "inside_br": "lower right",
                   "inside_bl": "lower left"}.get(pos, "upper right")
            kw.update(borderaxespad=0.7, frameon=True, framealpha=0.75, facecolor=self.panel, edgecolor="none",
                      borderpad=0.3, fancybox=False)
            leg = self.ax.legend(handles, labels, loc=loc, **kw)
            leg.set_zorder(Z_TEXT + 1)
        else:
            loc = {"right": "center left", "left": "center right", "top": "lower center",
                   "bottom": "upper center"}[pos]
            leg = self.fig.legend(handles, labels, loc=loc, bbox_to_anchor=(0.5, 0.5),
                                  bbox_transform=self.fig.transFigure, **kw)
        if leg.get_title() is not None:
            leg.get_title().set_color(self.ink)
        return leg

    def _build_legend(self):
        pos = str(self.o.get("legend_position") or "auto")
        if pos == "auto":
            pos = self.legend_auto or ("right" if len(self.legend_items) > 1 else "none")
        if pos == "none" or not self.legend_items or pos not in (
                "right", "left", "top", "bottom", "inside_tr", "inside_tl", "inside_br", "inside_bl"):
            self.legend_pos = "none"
            return
        self.legend_pos = pos
        n = len(self.legend_items)
        fw, fh = self.fig.bbox.width, self.fig.bbox.height
        if pos in ("top", "bottom"):
            ncols = n
            while True:
                leg = self._make_legend(pos, ncols)
                if ncols == 1 or leg.get_window_extent(self.renderer).width <= fw * 0.94:
                    break
                w = leg.get_window_extent(self.renderer).width
                leg.remove()
                ncols = max(1, min(ncols - 1, int(ncols * fw * 0.94 / w)))
        else:
            leg = self._make_legend(pos, 1)
            limit = fh * (0.8 if not pos.startswith("inside") else 0.55)
            h = leg.get_window_extent(self.renderer).height
            if h > limit and n > 2:
                leg.remove()
                leg = self._make_legend(pos, min(int(math.ceil(h / limit)), 4))
        self.legend = leg

    # --------------------------------------------------------------- layout
    def _place_single(self, l, b, r, t):
        self.set_pos(self.ax, l, b, r, t)

    def set_pos(self, ax, l, b, r, t):
        fw, fh = self.fig.bbox.width, self.fig.bbox.height
        ax.set_position([l / fw, b / fh, max(r - l, 4) / fw, max(t - b, 4) / fh])

    def place_square(self, ax, l, b, r, t, aspect: float = 1.0):
        """Largest box with height/width = ``aspect`` centred in the rectangle."""
        w, h = r - l, t - b
        if h / max(w, 1e-9) > aspect:
            nh = w * aspect
            b, t = b + (h - nh) / 2, b + (h + nh) / 2
        else:
            nw = h / aspect
            l, r = l + (w - nw) / 2, l + (w + nw) / 2
        self.set_pos(ax, l, b, r, t)
        return l, b, r, t

    def _axes_box(self) -> Bbox:
        return Bbox.union([ax.bbox for ax in self.axes])

    def _layout(self):
        r = self.renderer
        fw, fh = self.fig.bbox.width, self.fig.bbox.height
        base = self.fonts["base"] * self.px
        pad = 0.65 * base
        L, B, R, T = pad, pad, fw - pad, fh - pad
        header = False
        for key in ("title", "subtitle"):
            art = self._texts.get(key)
            if art is not None:
                art.set_y(T / fh)
                T -= art.get_window_extent(r).height + 0.3 * base
                header = True
        if header:
            T -= 0.35 * base
        cap = self._texts.get("caption")
        if cap is not None:
            cap.set_position((pad / fw, B / fh))
            B += cap.get_window_extent(r).height + 0.6 * base
        leg, pos = self.legend, self.legend_pos
        gap = 0.9 * base
        if leg is not None and not pos.startswith("inside"):
            ext = leg.get_window_extent(r)
            if pos == "right":
                R -= ext.width + gap
            elif pos == "left":
                L += ext.width + gap
            elif pos == "top":
                T -= ext.height + gap * 0.8
            else:
                B += ext.height + gap * 0.8
        min_w, min_h = 0.25 * fw, 0.25 * fh
        inner = self._inner or (L + 4 * base, B + 3 * base, R - base, T - base)
        for _ in range(4):
            self.placer(*inner)
            box = self._axes_box()
            tight = Bbox.union([b for b in (ax.get_tightbbox(r) for ax in self.axes) if b is not None] or [box])
            ml, mb = max(box.x0 - tight.x0, 0), max(box.y0 - tight.y0, 0)
            mr, mt = max(tight.x1 - box.x1, 0), max(tight.y1 - box.y1, 0)
            l, b_, r_, t = L + ml, B + mb, R - mr, T - mt
            if r_ - l < min_w:                       # hopeless: keep a drawable axes, let things clip
                l, r_ = min(l, fw - pad - min_w), max(r_, min(l, fw - pad - min_w) + min_w)
            if t - b_ < min_h:
                b_, t = min(b_, fh - pad - min_h), max(t, min(b_, fh - pad - min_h) + min_h)
            new = (l, b_, r_, t)
            done = max(abs(p - q) for p, q in zip(new, inner)) < 0.75
            inner = new
            if done:
                break
        self._inner = inner
        self.placer(*inner)
        box = self._axes_box()
        span = (box.x0, box.x1) if self.title_span == "axes" else (L, R)
        for key in ("title", "subtitle"):
            art = self._texts.get(key)
            if art is None:
                continue
            w = art.get_window_extent(r).width
            if art.get_ha() == "center":
                x = min(max((span[0] + span[1]) / 2, pad + w / 2), fw - pad - w / 2)
                if w > fw - 2 * pad:
                    x = fw / 2
            else:
                x = max(min(span[0], fw - pad - w), pad)
            art.set_x(x / fw)
        if leg is not None and not pos.startswith("inside"):
            cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
            ext = leg.get_window_extent(r)
            cx = min(max(cx, pad + ext.width / 2), fw - pad - ext.width / 2)
            cy = min(max(cy, pad + ext.height / 2), fh - pad - ext.height / 2)
            anchor = {"right": (R + gap, cy), "left": (L - gap, cy), "top": (cx, T + gap * 0.8),
                      "bottom": (cx, B - gap * 0.8)}[pos]
            leg.set_bbox_to_anchor((anchor[0] / fw, anchor[1] / fh), transform=self.fig.transFigure)

    def finish(self):
        self._build_legend()
        self._layout()
        for phase in (0, 1):
            if self.hooks[phase]:
                for fn in self.hooks[phase]:
                    fn()
                self._layout()
        for fn in self.hooks[2]:
            fn()

    # ---------------------------------------------------- significance brackets
    def note_top(self, c: float, v: float):
        if v is not None and np.isfinite(v):
            self.tops.append((float(c), float(v)))

    def add_brackets(self, spans: list[tuple[float, float, str]]):
        """Queue significance brackets between category positions (c1, c2, label)."""
        spans = [(min(a, b), max(a, b), str(lab)) for a, b, lab in spans if lab is not None]
        if spans:
            self.hooks[1].append(lambda: self._draw_brackets(spans))

    def _draw_brackets(self, spans):
        ax, f, px = self.ax, self.fonts, self.px
        scale = self.scales[self.vname]
        fwd, inv = scale_fwd(scale), scale_inv(scale)
        lim = ax.get_xlim() if self.flip else ax.get_ylim()
        lo_t, hi0_t = float(fwd(lim[0])), float(fwd(lim[1]))
        H = ax.bbox.width if self.flip else ax.bbox.height
        cat_px = (ax.bbox.height if self.flip else ax.bbox.width) / max(
            abs(np.diff(ax.get_ylim() if self.flip else ax.get_xlim())[0]), 1e-9)
        fs = f["tick"]
        lw = max(0.7, min(float(self.theme.get("axis_width") or 0.8), 1.4))
        tick, gap, level_gap = 3.0 * px, 4.5 * px, 2.5 * px
        inset = min(3.0 * px / cat_px, 0.12)
        tops_c = np.array([c for c, _ in self.tops] or [0.0])
        tops_t = fwd(np.array([v for _, v in self.tops] or [lim[0]]))

        def is_stars(lab):
            return bool(lab) and set(lab) <= {"*"}

        def label_px(lab):                           # room taken along the value axis
            if self.flip:
                return self.text_size(lab, fs * (1.2 if is_stars(lab) else 1.0))[1] * (0.55 if is_stars(lab) else 0.9) \
                    + 2 * px
            return (fs * 0.62 if is_stars(lab) else fs * 1.12) * px + 1.5 * px

        items = sorted(spans, key=lambda s: (round(s[1] - s[0], 6), s[0]))
        hi_t, placed = hi0_t, []
        for _ in range(8):
            u = (hi_t - lo_t) / max(H, 1.0)          # scaled units per pixel
            placed = []
            for a, b, lab in items:
                under = (tops_c >= a - 0.26) & (tops_c <= b + 0.26)
                y = (tops_t[under].max() if under.any() else lo_t) + gap * u
                for pa, pb, _, _, ptop in placed:
                    if a < pb - 1e-6 and b > pa + 1e-6:
                        y = max(y, ptop + level_gap * u)
                line = y + tick * u
                placed.append((a, b, lab, line, line + label_px(lab) * u))
            need = max(p[4] for p in placed) + 2 * px * u
            new_hi = max(hi0_t, need)
            if abs(new_hi - hi_t) < 0.25 * u or (new_hi - lo_t) > 6 * (hi0_t - lo_t):
                hi_t = new_hi
                break
            hi_t = new_hi
        (ax.set_xlim if self.flip else ax.set_ylim)(lim[0], float(inv(hi_t)))
        u = (hi_t - lo_t) / max(H, 1.0)
        C0, V0, C1, V1 = [], [], [], []
        for a, b, lab, line, _ in placed:
            a2, b2 = (a + inset, b - inset) if b - a > 2.5 * inset else (a, b)
            v_line, v_tick = float(inv(line)), float(inv(line - tick * u))
            C0 += [a2, a2, b2]; V0 += [v_tick, v_line, v_line]
            C1 += [a2, b2, b2]; V1 += [v_line, v_line, v_tick]
            stars = is_stars(lab)
            size = fs * (1.2 if stars else 1.0)
            xy = self.xy((a + b) / 2, v_line)
            if self.flip:
                ax.annotate(lab, xy, xytext=(1.5 if not stars else -0.12 * size, 0), textcoords="offset points",
                            ha="left", va="center", rotation=270, rotation_mode="anchor" if False else "default",
                            fontsize=size, color=self.ink, zorder=Z_TEXT, annotation_clip=False)
            else:
                ax.annotate(lab, xy, xytext=(0, 1.5 if not stars else -0.36 * size), textcoords="offset points",
                            ha="center", va="bottom", fontsize=size, color=self.ink, zorder=Z_TEXT,
                            annotation_clip=False)
        self.segments(C0, V0, C1, V1, colors=self.ink, linewidths=lw, zorder=Z_TEXT, capstyle="projecting",
                      clip_on=False)

    # ---------------------------------------------------------- point labels
    def label_points(self, x, y, texts, size: float, radius_pt: float = 3.0, color=None,
                     avoid: np.ndarray | None = None, max_labels: int = 250):
        """Label points (data coords) with a greedy search for a free spot
        around each point; labels pushed away get a thin connector."""
        self.hooks[2].append(lambda: self._label_points(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                                                        list(texts), size, radius_pt, color, avoid, max_labels))

    def _label_points(self, x, y, texts, size, radius_pt, color, avoid, max_labels):
        ax, px = self.ax, self.px
        keep = [i for i, t in enumerate(texts) if t not in (None, "") and np.isfinite(x[i]) and np.isfinite(y[i])]
        keep = keep[:max_labels]
        if not keep:
            return
        pts = ax.transData.transform(np.column_stack([x, y]))
        obstacles = pts if avoid is None else ax.transData.transform(avoid)
        obstacles = obstacles[np.isfinite(obstacles).all(axis=1)]
        box = ax.bbox
        placed = np.empty((0, 4))
        r0 = (radius_pt + 1.5) * px
        angles = [45, 135, 315, 225, 90, 270, 0, 180, 20, 160, 340, 200, 70, 110, 290, 250]
        dirs = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in angles]
        rings = [r0, r0 + 8 * px, r0 + 18 * px, r0 + 30 * px, r0 + 46 * px]
        color = color or self.ink
        m = 1.5 * px                                 # breathing space around every label
        for i in keep:
            w, h = self.text_size(texts[i], size)
            best = None
            for ring_i, rr in enumerate(rings):
                for ux, uy in dirs:
                    sx = 0 if abs(ux) < 0.3 else (1 if ux > 0 else -1)
                    sy = 0 if abs(uy) < 0.3 else (1 if uy > 0 else -1)
                    ox, oy = ux * rr, uy * rr
                    x0 = pts[i, 0] + ox - (w if sx < 0 else w / 2 if sx == 0 else 0)
                    y0 = pts[i, 1] + oy - (h if sy < 0 else h / 2 if sy == 0 else 0)
                    x1, y1 = x0 + w, y0 + h
                    cost = ring_i * 0.8
                    if x0 < box.x0 or x1 > box.x1 or y0 < box.y0 or y1 > box.y1:
                        cost += 60
                    if len(placed):
                        cost += 100 * ((x0 - m < placed[:, 2]) & (x1 + m > placed[:, 0])
                                       & (y0 - m < placed[:, 3]) & (y1 + m > placed[:, 1])).sum()
                    cost += ((obstacles[:, 0] > x0 - m) & (obstacles[:, 0] < x1 + m)
                             & (obstacles[:, 1] > y0 - m) & (obstacles[:, 1] < y1 + m)).sum()
                    if best is None or cost < best[0]:
                        best = (cost, ox, oy, sx, sy, (x0, y0, x1, y1), ring_i)
                if best[0] < 0.8 * (ring_i + 1):     # a free spot in this ring: stop searching
                    break
            _, ox, oy, sx, sy, rect, ring_i = best
            placed = np.vstack([placed, rect])
            ax.annotate(texts[i], (x[i], y[i]), xytext=(ox / px, oy / px), textcoords="offset points",
                        ha="right" if sx < 0 else "center" if sx == 0 else "left",
                        va="top" if sy < 0 else "center" if sy == 0 else "bottom",
                        fontsize=size, color=color, zorder=Z_TEXT,
                        arrowprops=(dict(arrowstyle="-", linewidth=0.5, color=mix(color, self.panel, 0.4),
                                         shrinkA=0.5, shrinkB=radius_pt) if ring_i > 0 else None))

    # ----------------------------------------------------------------- save
    def save(self) -> bytes:
        fmt = str(self.payload.get("format") or "svg").lower()
        dpi = int(self.payload.get("dpi") or 300)
        transparent = bool(self.theme.get("transparent")) and fmt != "jpg"
        if transparent:
            self.fig.patch.set_alpha(0.0)
            for ax in self.fig.axes:
                ax.patch.set_alpha(0.0)
        kw: dict = {"format": "jpeg" if fmt == "jpg" else fmt, "facecolor": "none" if transparent else self.bg,
                    "edgecolor": "none"}
        if fmt == "svg":
            kw["metadata"] = {"Date": None}
        elif fmt == "pdf":
            kw["metadata"] = {"CreationDate": None}
        else:
            kw["dpi"] = max(dpi, 20)
            if fmt == "tiff":
                kw["pil_kwargs"] = {"compression": "tiff_lzw"}
            elif fmt == "jpg":
                kw["pil_kwargs"] = {"quality": 95}
        buf = io.BytesIO()
        self.fig.savefig(buf, **kw)
        return buf.getvalue()


def tidy_labels(labels) -> list[str]:
    """Row labels as text; whole numbers that arrive as floats ("14.0") lose the ".0"."""
    out = [str(l) for l in labels]
    if out and all(s.endswith(".0") and s[:-2].lstrip("-").isdigit() for s in out):
        out = [s[:-2] for s in out]
    return out


def require(frame, name: str = "data"):
    """Friendly error for an empty frame."""
    if frame is None or len(frame) == 0:
        raise AnalysisError("There is nothing to plot yet.")
    return frame
