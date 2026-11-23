"""Graph type definitions: option schemas + data preparation.

Each ``_prep_*`` function documents the *payload* (frames + computed) that
the drawing engines receive for that render type. Engines never see the
user's raw data, only these tidy frames.

Conventions for all payloads
----------------------------
* categorical frame columns are pandas Categoricals / R factors whose level
  order is the display order;
* ``options["_color_names"]`` / ``["_color_values"]`` give the colour of every
  colour key (group / series / slice), in legend order;
* missing numbers are NaN (NA in R).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from ..core import options as opt
from ..core.context import Context
from ..core.errors import AnalysisError
from ..core.results import stars
from . import compute
from .registry import Prepared, graph, variant

def _label_text(v):
    """Cell value -> label string (numbers without a trailing .0)."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):g}"
    return str(v)


# ---------------------------------------------------------------- option kits
CENTER = [("mean", "Mean"), ("median", "Median"), ("geomean", "Geometric mean")]
ERROR = [("sd", "SD"), ("sem", "SEM"), ("ci95", "95% CI"), ("iqr", "Interquartile range"),
         ("range", "Range (min–max)"), ("none", "None")]
SHAPES = [("circle", "● Circle"), ("square", "■ Square"), ("triangle", "▲ Triangle"),
          ("diamond", "◆ Diamond"), ("circle_open", "○ Open circle"),
          ("square_open", "□ Open square"), ("triangle_open", "△ Open triangle")]
SORT = [("data", "As in data"), ("alpha", "Alphabetical"), ("asc", "Ascending by mean"),
        ("desc", "Descending by mean")]


def _summary_opts(center="mean", error="sd", section="Summary & error bars"):
    return [
        opt.choice("center", "Show", CENTER, default=center, section=section),
        opt.choice("error", "Error bars", ERROR, default=error, section=section),
        opt.number("error_cap", "Error bar cap width", 0.25, min=0, max=1, step=0.05, section=section,
                   help="As a fraction of the group width; 0 for no caps."),
        opt.number("error_width", "Error bar thickness (pt)", 1.0, min=0.1, max=6, step=0.1, section=section),
    ]


def _point_opts(show=True, size=5.0, alpha=0.9, section="Points", toggle=True, layout=True):
    o = []
    if toggle:
        o.append(opt.boolean("show_points", "Show individual points", show, section=section))
    if layout:
        o += [opt.choice("point_layout", "Arrangement", [("jitter", "Random jitter"),
                                                        ("swarm", "Beeswarm (no overlap)"),
                                                        ("aligned", "Aligned in a line")],
                         default="swarm", section=section),
              opt.number("jitter", "Spread width", 0.18, min=0, max=0.5, step=0.02, section=section)]
    o += [opt.number("point_size", "Point size (pt)", size, min=0.5, max=30, step=0.5, section=section),
          opt.choice("point_shape", "Point shape", SHAPES, default="circle", section=section),
          opt.number("point_alpha", "Point opacity", alpha, min=0.05, max=1, step=0.05, section=section),
          opt.choice("point_color", "Point color", [("group", "Group color"), ("dark", "Darker shade of group color"),
                                                    ("black", "Black"), ("gray", "Gray"), ("white", "White fill, group outline")],
                     default="group", section=section)]
    return o


def _stat_opts(paired=False, grouped=False, section="Significance"):
    tests = [("none", "None")]
    if paired:
        tests += [("paired_t", "Paired t test"), ("wilcoxon", "Wilcoxon matched-pairs")]
    tests += [("welch", "Welch's t test"), ("t", "Student's t test"), ("mannwhitney", "Mann–Whitney test")]
    comps = [("all", "All pairs"), ("control", "Each vs. a control group")]
    if not grouped:
        comps += [("adjacent", "Neighbouring groups"), ("custom", "Custom pairs")]
    o = [opt.choice("stat_test", "Pairwise test", tests, default="none", section=section,
                    help="Draws significance brackets. For a full report use the Analyze menu."),
         opt.choice("stat_comparisons", "Comparisons", comps, default="all", section=section)]
    if grouped:
        o.append(opt.level("stat_control", "Control group", ["group", "columns"], section=section,
                           visible={"stat_comparisons": "control"}))
    else:
        o += [opt.level("stat_control", "Control group", ["group", "columns"], section=section,
                        visible={"stat_comparisons": "control"}),
              opt.text("stat_custom", "Pairs", placeholder="A vs B; A vs C", section=section,
                       visible={"stat_comparisons": "custom"})]
    o += [opt.choice("stat_adjust", "Multiple-comparison correction", opt.P_ADJUST, default="holm", section=section),
          opt.choice("stat_label", "Label", [("stars", "Stars (*, **, ***)"), ("p", "P value")],
                     default="stars", section=section),
          opt.boolean("stat_hide_ns", "Hide non-significant", False, section=section)]
    return o


def _bar_opts(section="Bars", width=0.65, alpha=1.0):
    return [opt.number("bar_width", "Width", width, min=0.05, max=1, step=0.05, section=section),
            opt.number("fill_alpha", "Fill opacity", alpha, min=0, max=1, step=0.05, section=section),
            opt.choice("outline", "Outline", [("match", "Same as fill"), ("black", "Black"), ("none", "None")],
                       default="match", section=section),
            opt.number("outline_width", "Outline thickness (pt)", 1.0, min=0, max=6, step=0.25, section=section)]


COLOR_BY = opt.choice("color_by", "Color", [("group", "Each group its own color"),
                                            ("single", "One color for all groups")],
                      default="group", section="Colors & theme")

COLUMN_FEATURES = {"axes", "y_num", "x_cat", "palette", "flip", "legend"}


# ================================================================== FACETS
FACET_OPTS = [
    opt.variable("facet", "Split into panels by", opt.ANY, required=False, section="Data",
                 help="Draws one small panel per value of this column."),
    opt.integer("facet_cols", "Panels per row", None, min=1, max=12, section="Panels", placeholder="auto"),
    opt.choice("facet_axes", "Panel axes", [("shared", "Same scales in every panel"),
                                            ("free_y", "Independent Y scales"),
                                            ("free", "Independent X and Y scales")],
               default="shared", section="Panels"),
    opt.boolean("facet_labels", "Show panel titles", True, section="Panels"),
]


def faceted(prep):
    """Wrap a ``_prep_*`` function so the graph can be split into panels.

    Without a ``facet`` variable the payload is unchanged. With one:

    * every frame gains a ``facet`` factor column (level order = panel order);
    * ``computed["facets"]`` lists the panel names;
    * every dict in ``computed["brackets"]`` and ``computed["fits"]`` gains a
      ``"facet"`` key — statistics are computed separately inside each panel;
    * ``computed["facet_cols"]`` (int or None = automatic), ``computed["facet_axes"]``
      ("shared" | "free_y" | "free") and ``computed["facet_labels"]`` (bool).
    """
    def wrapped(ctx: Context) -> Prepared:
        fv = ctx.var("facet", required=False)
        full = prep(ctx)
        if not fv:
            return full
        f = ctx.factor(fv)
        levels = [str(l) for l in f.cat.categories]
        if len(levels) > 30:
            raise AnalysisError(f"“{fv}” would create {len(levels)} panels; the limit is 30.")
        frames: dict[str, list[pd.DataFrame]] = {}
        brackets, fits, used, warnings = [], [], [], list(full.warnings)
        for lev in levels:
            mask = (f.astype(object) == lev).to_numpy()
            sub_opts = ctx.options
            if "density" in full.frames:      # same smoothing grid in every panel
                gx = full.frames["density"]["x"]
                sub_opts = dict(ctx.options, _grid=(float(gx.min()), float(gx.max())))
            sub = Context(ctx.df[mask].reset_index(drop=True), sub_opts)
            sub._labels = ctx._labels
            try:
                part = prep(sub)
            except AnalysisError as exc:
                warnings.append(f"Panel “{lev}” was left out: {exc}")
                continue
            used.append(lev)
            for name, df in part.frames.items():
                df = df.copy()
                ref = full.frames.get(name)
                for col in df.columns:        # keep one level order across panels
                    if isinstance(df[col].dtype, pd.CategoricalDtype) and ref is not None and col in ref.columns \
                            and isinstance(ref[col].dtype, pd.CategoricalDtype):
                        cats = [str(c) for c in ref[col].cat.categories]
                        if set(map(str, df[col].cat.categories)) <= set(cats):
                            df[col] = pd.Categorical(df[col].astype(str).where(df[col].notna()), categories=cats)
                df["facet"] = lev
                frames.setdefault(name, []).append(df)
            brackets += [dict(b, facet=lev) for b in part.computed.get("brackets", [])]
            fits += [dict(b, facet=lev) for b in part.computed.get("fits", [])]
            warnings += [f"{lev}: {w}" for w in part.warnings if w not in full.warnings]
        if not used:
            raise AnalysisError(f"None of the panels defined by “{fv}” has enough data to draw.")
        out = {}
        for name, parts in frames.items():
            df = pd.concat(parts, ignore_index=True)
            df["facet"] = pd.Categorical(df["facet"], categories=used)
            out[name] = df
        computed = dict(full.computed)
        if "brackets" in computed:
            computed["brackets"] = brackets
        if "fits" in computed:
            computed["fits"] = fits
        computed.update(facets=used, facet_cols=ctx.opt("facet_cols"), facet_axes=ctx.opt("facet_axes", "shared"),
                        facet_labels=bool(ctx.opt("facet_labels", True)))
        full.frames, full.computed, full.warnings = out, computed, warnings
        return full
    return wrapped


# ================================================================== COLUMN
def _prep_column(ctx: Context, paired: bool = False) -> Prepared:
    """Render types: column_scatter, column_bar, box, violin, raincloud, before_after.

    frames
      main     value (num), group (factor) [, subject (chr) for before_after]
      summary  group, n, center, lo, hi   (lo/hi = error-bar ends, NaN if none)
    computed
      groups    [names in display order]
      brackets  [{g1, g2, p, label}]  significance brackets to draw
      center, error   the chosen summary types (for legends/captions)
    """
    long = ctx.long(with_subject=paired)
    if long.empty:
        raise AnalysisError("There are no numeric values to plot yet.")
    order = [str(c) for c in long["group"].cat.categories]
    long["group"] = long["group"].astype(str)
    sort = ctx.opt("sort_groups", "data")
    if sort == "alpha":
        order = sorted(order, key=str.lower)
    elif sort in ("asc", "desc"):
        means = long.groupby("group")["value"].mean()
        order = sorted(order, key=lambda g: means[g], reverse=(sort == "desc"))
    long["group"] = pd.Categorical(long["group"], categories=order)
    groups = {g: long.loc[long["group"] == g, "value"].to_numpy(dtype=float) for g in order}
    center, error = ctx.opt("center", "mean"), ctx.opt("error", "sd")
    rows = [dict(group=g, **compute.summarize(v, center, error)) for g, v in groups.items()]
    summary = pd.DataFrame(rows)
    summary["group"] = pd.Categorical(summary["group"], categories=order)
    for c in ("center", "lo", "hi"):
        summary[c] = pd.to_numeric(summary[c], errors="coerce")
    paired_frame = None
    if paired:
        paired_frame = long.pivot_table(index="subject", columns="group", values="value",
                                        aggfunc="first", observed=False)
        paired_frame.columns = [str(c) for c in paired_frame.columns]
    brackets = compute.pairwise_brackets(
        groups, ctx.opt("stat_test", "none"), ctx.opt("stat_comparisons", "all"),
        ctx.opt("stat_control", ""), ctx.opt("stat_custom", ""), ctx.opt("stat_adjust", "holm"),
        ctx.opt("stat_label", "stars"), bool(ctx.opt("stat_hide_ns", False)), paired=paired_frame)
    wide = ctx.opt("layout") == "wide"
    p = Prepared(frames={"main": long, "summary": summary},
                 computed={"groups": order, "brackets": brackets, "center": center, "error": error},
                 color_keys=order,
                 x_label="" if wide else (ctx.opt("group") or ""),
                 y_label="Value" if wide else (ctx.opt("dep") or ""))
    p.single_color = ctx.opt("color_by", "group") == "single"
    return p


def _column_data(paired=False):
    return [*opt.group_layout(paired=paired, min_groups=1),
            opt.choice("sort_groups", "Group order", SORT, default="data", section="Data"),
            *copy.deepcopy(FACET_OPTS)]


graph(id="column_scatter", name="Scatter dot plot", category="Column",
      description="Every value as a dot, with a line at the mean or median and error bars.",
      features=COLUMN_FEATURES, prepare=faceted(_prep_column),
      options=[*_column_data(),
               *_point_opts(toggle=False, size=6),
               opt.boolean("show_summary", "Show mean/median line & error bars", True, section="Summary & error bars"),
               *_summary_opts(),
               opt.number("summary_width", "Mean/median line width", 0.5, min=0.05, max=1, step=0.05,
                          section="Summary & error bars", help="As a fraction of the group width."),
               *_stat_opts(), COLOR_BY])

graph(id="column_bar", name="Bar graph", category="Column",
      description="Bars at the mean or median with error bars; optionally overlay the individual points.",
      features=COLUMN_FEATURES, prepare=faceted(_prep_column),
      options=[*_column_data(), *_bar_opts(), *_summary_opts(error="sem"),
               opt.choice("error_direction", "Error bar direction", [("both", "Both"), ("up", "Up only")],
                          default="both", section="Summary & error bars"),
               *_point_opts(show=True, size=4.5, alpha=0.85), *_stat_opts(), COLOR_BY])

graph(id="box", name="Box & whiskers", category="Column",
      description="Median, quartiles and whiskers; optionally overlay the individual points.",
      features=COLUMN_FEATURES, prepare=faceted(_prep_column),
      options=[*_column_data(),
               opt.choice("whiskers", "Whiskers", [("tukey", "Tukey (1.5 × IQR)"), ("minmax", "Min to max"),
                                                   ("p5_95", "5th–95th percentile"), ("p10_90", "10th–90th percentile")],
                          default="tukey", section="Box"),
               opt.boolean("show_outliers", "Show outliers", True, section="Box",
                           help="Ignored when individual points are shown."),
               opt.boolean("show_mean", "Mark the mean (+)", False, section="Box"),
               opt.boolean("notch", "Notched boxes", False, section="Box"),
               *_bar_opts(section="Box", width=0.6, alpha=0.6),
               *_point_opts(show=False, size=4, alpha=0.8), *_stat_opts(), COLOR_BY])

graph(id="violin", name="Violin plot", category="Column",
      description="Distribution shape of every group, with optional box, quartiles and points.",
      features=COLUMN_FEATURES, prepare=faceted(_prep_column),
      options=[*_column_data(),
               opt.choice("violin_inner", "Inside the violin", [("box", "Mini box plot"), ("quartiles", "Quartile lines"),
                                                               ("mean_sd", "Mean ± SD"), ("none", "Nothing")],
                          default="box", section="Violin"),
               opt.choice("violin_scale", "Violin widths", [("width", "All the same width"), ("area", "Same area"),
                                                            ("count", "Proportional to n")],
                          default="width", section="Violin"),
               opt.boolean("violin_trim", "Trim at the data range", True, section="Violin"),
               opt.number("bw_adjust", "Smoothing", 1.0, min=0.1, max=5, step=0.1, section="Violin"),
               *_bar_opts(section="Violin", width=0.8, alpha=0.6),
               *_point_opts(show=False, size=3.5, alpha=0.7), *_stat_opts(), COLOR_BY])

graph(id="raincloud", name="Raincloud plot", category="Column",
      description="Half violin, box plot and raw points side by side — the full picture of each group.",
      features=COLUMN_FEATURES, prepare=faceted(_prep_column),
      options=[*_column_data(),
               opt.number("bw_adjust", "Smoothing", 1.0, min=0.1, max=5, step=0.1, section="Cloud"),
               opt.number("fill_alpha", "Fill opacity", 0.6, min=0, max=1, step=0.05, section="Cloud"),
               opt.boolean("show_box", "Show box plot", True, section="Cloud"),
               *_point_opts(toggle=False, size=3.5, alpha=0.7, layout=False), *_stat_opts(), COLOR_BY])

graph(id="before_after", name="Before–after (paired)", category="Column",
      description="Paired values joined by a line for each subject.",
      features=COLUMN_FEATURES, prepare=faceted(lambda ctx: _prep_column(ctx, paired=True)),
      options=[*_column_data(paired=True),
               opt.number("line_width", "Line thickness (pt)", 1.0, min=0.1, max=6, step=0.1, section="Lines"),
               opt.number("line_alpha", "Line opacity", 0.6, min=0.05, max=1, step=0.05, section="Lines"),
               opt.choice("line_color", "Line color", [("gray", "Gray"), ("black", "Black"),
                                                       ("direction", "By direction (up / down)")],
                          default="gray", section="Lines"),
               opt.boolean("show_summary", "Overlay mean & error bars", False, section="Summary & error bars"),
               *_summary_opts(),
               *_point_opts(toggle=False, size=6, layout=False), *_stat_opts(paired=True), COLOR_BY])


# ================================================================= GROUPED
def _grouped_frame(ctx: Context) -> tuple[pd.DataFrame, str, str, str]:
    if ctx.opt("layout", "long") == "wide":
        cols = ctx.vars("columns", min=1)
        lab = ctx.var("label", required=False)
        if lab:
            x = ctx.factor(lab)
        else:
            x = pd.Series(pd.Categorical(["All"] * len(ctx.df)), index=ctx.df.index)
        parts = [pd.DataFrame({"value": ctx.numeric(c).to_numpy(), "x": x.to_numpy(), "group": c}) for c in cols]
        df = pd.concat(parts, ignore_index=True)
        df["x"] = pd.Categorical(df["x"], categories=list(x.cat.categories))
        df["group"] = pd.Categorical(df["group"], categories=cols)
        labels = (lab or "", "Value", "")
    else:
        dep, xv, gv = ctx.var("dep"), ctx.var("x"), ctx.var("group", required=False)
        xf = ctx.factor(xv)
        gf = ctx.factor(gv) if gv else pd.Series(pd.Categorical([dep] * len(ctx.df)), index=ctx.df.index)
        df = pd.DataFrame({"value": ctx.numeric(dep).to_numpy(), "x": xf.to_numpy(), "group": gf.to_numpy()})
        df["x"] = pd.Categorical(df["x"], categories=list(xf.cat.categories))
        df["group"] = pd.Categorical(df["group"], categories=list(gf.cat.categories))
        labels = (xv, dep, gv or "")
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise AnalysisError("There are no numeric values to plot yet.")
    df["x"] = df["x"].cat.remove_unused_categories()
    df["group"] = df["group"].cat.remove_unused_categories()
    return df, *labels


def _prep_grouped(ctx: Context) -> Prepared:
    """Render types: grouped_bar, grouped_box, grouped_violin, grouped_points, interaction_line.

    frames
      main     value (num), x (factor: categories on the axis), group (factor: colour / legend)
      summary  x, group, n, center, lo, hi
    computed
      x_levels, groups   display orders
      brackets  [{x, g1, g2, p, label}]  comparisons between groups *within* one x category
    """
    df, xl, yl, ll = _grouped_frame(ctx)
    center, error = ctx.opt("center", "mean"), ctx.opt("error", "sd")
    xs = [str(c) for c in df["x"].cat.categories]
    gs = [str(c) for c in df["group"].cat.categories]
    rows = []
    for x in xs:
        for g in gs:
            v = df.loc[(df["x"] == x) & (df["group"] == g), "value"].to_numpy(dtype=float)
            if len(v):
                rows.append(dict(x=x, group=g, **compute.summarize(v, center, error)))
    summary = pd.DataFrame(rows)
    summary["x"] = pd.Categorical(summary["x"], categories=xs)
    summary["group"] = pd.Categorical(summary["group"], categories=gs)
    for c in ("center", "lo", "hi"):
        summary[c] = pd.to_numeric(summary[c], errors="coerce")
    brackets = []
    test = ctx.opt("stat_test", "none")
    if test != "none" and len(gs) > 1:
        raw = []
        for x in xs:
            groups = {g: df.loc[(df["x"] == x) & (df["group"] == g), "value"].to_numpy(dtype=float) for g in gs}
            groups = {g: v for g, v in groups.items() if len(v)}
            for b in compute.pairwise_brackets(groups, test, ctx.opt("stat_comparisons", "all"),
                                               ctx.opt("stat_control", ""), "", "none", "stars", False):
                raw.append(dict(b, x=x))
        ok = [b for b in raw if b["p"] is not None]
        for b, p in zip(ok, compute._adjust([b["p"] for b in ok], ctx.opt("stat_adjust", "holm"))):
            b["p"] = float(p)
        for b in raw:
            p = b["p"]
            if ctx.opt("stat_hide_ns", False) and not (p is not None and p < 0.05):
                continue
            b["label"] = compute.p_label(float("nan") if p is None else p, ctx.opt("stat_label", "stars"))
            brackets.append(b)
    return Prepared(frames={"main": df, "summary": summary},
                    computed={"x_levels": xs, "groups": gs, "brackets": brackets,
                              "center": center, "error": error},
                    color_keys=gs, x_label=xl, y_label=yl, legend_title=ll)


def _grouped_data():
    s = "Data"
    return [opt.choice("layout", "Data layout", [("long", "Values + two grouping columns"),
                                                 ("wide", "Row labels + one column per subgroup")],
                       default="long", section=s),
            opt.variable("dep", "Values", opt.CONTINUOUS, section=s, visible={"layout": "long"}),
            opt.variable("x", "Categories on the X axis", opt.ANY, section=s, visible={"layout": "long"}),
            opt.variable("group", "Subgroups (colors)", opt.ANY, required=False, section=s, visible={"layout": "long"}),
            opt.variable("label", "Row labels (X axis categories)", opt.ANY, required=False, section=s,
                         visible={"layout": "wide"}),
            opt.variables("columns", "Subgroups (one column each)", opt.CONTINUOUS, min=1, section=s,
                          visible={"layout": "wide"}),
            *copy.deepcopy(FACET_OPTS)]


GROUPED_FEATURES = {"axes", "y_num", "x_cat", "palette", "flip", "legend"}
_dodge = opt.number("dodge", "Spacing between subgroups", 0.8, min=0.2, max=1, step=0.05, section="Layout",
                    help="Total width used by the subgroups of one category.")

graph(id="grouped_bar", name="Grouped bar graph", category="Grouped",
      description="Bars for every subgroup, side by side or stacked, with error bars.",
      features=GROUPED_FEATURES, prepare=faceted(_prep_grouped),
      options=[*_grouped_data(),
               opt.choice("position", "Arrangement", [("dodge", "Side by side"), ("stack", "Stacked")],
                          default="dodge", section="Layout"), _dodge,
               *_bar_opts(width=0.9), *_summary_opts(error="sem"),
               *_point_opts(show=True, size=3.5, alpha=0.85, layout=False),
               *_stat_opts(grouped=True)])

graph(id="grouped_box", name="Grouped box plot", category="Grouped",
      description="Box & whiskers for every subgroup within each category.",
      features=GROUPED_FEATURES, prepare=faceted(_prep_grouped),
      options=[*_grouped_data(), _dodge,
               opt.choice("whiskers", "Whiskers", [("tukey", "Tukey (1.5 × IQR)"), ("minmax", "Min to max")],
                          default="tukey", section="Box"),
               opt.boolean("show_outliers", "Show outliers", True, section="Box"),
               *_bar_opts(section="Box", width=0.85, alpha=0.6),
               *_point_opts(show=False, size=3, alpha=0.8, layout=False), *_stat_opts(grouped=True)])

graph(id="grouped_violin", name="Grouped violin plot", category="Grouped",
      description="Violins for every subgroup within each category.",
      features=GROUPED_FEATURES, prepare=faceted(_prep_grouped),
      options=[*_grouped_data(), _dodge,
               opt.choice("violin_inner", "Inside the violin", [("box", "Mini box plot"), ("quartiles", "Quartile lines"),
                                                               ("none", "Nothing")], default="box", section="Violin"),
               opt.number("bw_adjust", "Smoothing", 1.0, min=0.1, max=5, step=0.1, section="Violin"),
               *_bar_opts(section="Violin", width=0.9, alpha=0.6),
               *_point_opts(show=False, size=3, alpha=0.7, layout=False), *_stat_opts(grouped=True)])

graph(id="grouped_points", name="Grouped scatter", category="Grouped",
      description="Individual points for every subgroup with mean/median line and error bars.",
      features=GROUPED_FEATURES, prepare=faceted(_prep_grouped),
      options=[*_grouped_data(), _dodge,
               *_point_opts(toggle=False, size=4.5, layout=False),
               opt.number("jitter", "Spread width", 0.08, min=0, max=0.4, step=0.01, section="Points"),
               opt.boolean("show_summary", "Show mean/median line & error bars", True, section="Summary & error bars"),
               *_summary_opts(), *_stat_opts(grouped=True)])

graph(id="interaction_line", name="Interaction (means) plot", category="Grouped",
      description="Group means joined by lines across categories — the classic two-way ANOVA plot.",
      features=GROUPED_FEATURES, prepare=faceted(_prep_grouped),
      options=[*_grouped_data(),
               opt.number("dodge", "Horizontal offset between lines", 0.15, min=0, max=0.8, step=0.05, section="Layout"),
               opt.number("line_width", "Line thickness (pt)", 1.5, min=0.1, max=6, step=0.1, section="Lines"),
               *_summary_opts(error="sem"),
               opt.boolean("show_points", "Show individual points", False, section="Points"),
               *_point_opts(toggle=False, size=6, layout=False)])


# ====================================================================== XY
def _prep_xy(ctx: Context) -> Prepared:
    """Render type: xy.

    frames
      main     x, y (num), group (factor = series) [, size (num), label (chr)]
      summary  x, group, n, center, lo, hi   one row per distinct x per series
               (present when ``xy_mode`` is "summary" or "both")
      fit      group, x, y, lo, hi           fitted curve points (lo/hi NaN if no band);
               present when computed.fits is non-empty
    computed
      series        [names]
      mode          "points" | "summary" | "both"
      fits          [{group, label}]   legend text for each fitted curve
      has_size, has_label
    """
    xv = ctx.var("x")
    ys = ctx.vars("y", min=1)
    gv = ctx.var("group", required=False)
    sv = ctx.var("size", required=False)
    lv = ctx.var("label", required=False)
    if gv and len(ys) > 1:
        raise AnalysisError("Choose a single Y variable when a grouping variable is used "
                            "(or remove the grouping variable to plot several Y columns).")
    x = ctx.numeric(xv).to_numpy()
    parts = []
    for yname in ys:
        part = pd.DataFrame({"x": x, "y": ctx.numeric(yname).to_numpy()})
        part["group"] = ctx.factor(gv).astype(object).to_numpy() if gv else yname
        if sv:
            part["size"] = ctx.numeric(sv).to_numpy()
        if lv:
            part["label"] = ctx.column(lv).astype(object).map(_label_text).to_numpy()
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["x"].notna() & df["y"].notna() & df["group"].notna()].reset_index(drop=True)
    warnings = []
    log_x = ctx.opt("x_scale", "linear") in ("log10", "log2")
    log_y = ctx.opt("y_scale", "linear") in ("log10", "log2")
    if log_x and (df["x"] <= 0).any():
        warnings.append(f"{int((df['x'] <= 0).sum())} point(s) with X ≤ 0 cannot be shown on a log axis.")
        df = df[df["x"] > 0]
    if log_y and (df["y"] <= 0).any():
        warnings.append(f"{int((df['y'] <= 0).sum())} point(s) with Y ≤ 0 cannot be shown on a log axis.")
        df = df[df["y"] > 0]
    if df.empty:
        raise AnalysisError("There are no complete X–Y pairs to plot yet.")
    series = list(ctx.factor(gv).cat.categories) if gv else list(ys)
    series = [s for s in series if (df["group"] == s).any()]
    df["group"] = pd.Categorical(df["group"], categories=series)
    df = df.reset_index(drop=True)
    frames = {"main": df}
    mode = ctx.opt("xy_mode", "points")
    if mode in ("summary", "both"):
        rows = []
        for (g, xval), sub in df.groupby(["group", "x"], observed=True):
            rows.append(dict(x=xval, group=g, **compute.summarize(sub["y"], ctx.opt("center", "mean"), ctx.opt("error", "sd"))))
        sm = pd.DataFrame(rows)
        sm["group"] = pd.Categorical(sm["group"], categories=series)
        for c in ("center", "lo", "hi"):
            sm[c] = pd.to_numeric(sm[c], errors="coerce")
        frames["summary"] = sm.sort_values(["group", "x"]).reset_index(drop=True)
    fits, fit_rows = [], []
    model = ctx.opt("fit", "none")
    if model != "none":
        for g in series:
            sub = df[df["group"] == g]
            try:
                f = compute.fit_series(sub["x"], sub["y"], model, log_x=log_x,
                                       span=float(ctx.opt("loess_span", 0.75)),
                                       band=bool(ctx.opt("fit_band", True)))
            except AnalysisError as exc:
                warnings.append(f"{g}: {exc}")
                continue
            if f is None:
                continue
            n = len(f["x"])
            fit_rows.append(pd.DataFrame({
                "group": g, "x": f["x"], "y": f["y"],
                "lo": f["lo"] if f["lo"] is not None else np.full(n, np.nan),
                "hi": f["hi"] if f["hi"] is not None else np.full(n, np.nan)}))
            fits.append({"group": g, "label": f["label"]})
    if fit_rows:
        ft = pd.concat(fit_rows, ignore_index=True)
        ft["group"] = pd.Categorical(ft["group"], categories=series)
        ft = ft[np.isfinite(ft["y"])]
        frames["fit"] = ft.reset_index(drop=True)
    return Prepared(frames=frames,
                    computed={"series": series, "mode": mode, "fits": fits,
                              "has_size": bool(sv), "has_label": bool(lv), "size_label": sv or "",
                              "center": ctx.opt("center", "mean"), "error": ctx.opt("error", "sd")},
                    color_keys=series, x_label=xv, y_label=ys[0] if len(ys) == 1 else "",
                    legend_title=gv or "", warnings=warnings)


XY_FEATURES = {"axes", "x_num", "y_num", "palette", "legend"}
_xy_options = [
    opt.variable("x", "X", opt.CONTINUOUS, section="Data"),
    opt.variables("y", "Y (one or more series)", opt.CONTINUOUS, min=1, section="Data"),
    opt.variable("group", "Group into series by", opt.ANY, required=False, section="Data"),
    opt.variable("size", "Bubble size", opt.CONTINUOUS, required=False, section="Data"),
    opt.variable("label", "Point labels", opt.ANY, required=False, section="Data"),
    *copy.deepcopy(FACET_OPTS),
    opt.choice("xy_mode", "Replicates at the same X", [("points", "Plot every point"),
                                                       ("summary", "Plot mean/median with error bars"),
                                                       ("both", "Both")], default="points", section="Points & lines"),
    opt.choice("center", "Summary", CENTER, default="mean", section="Points & lines",
               visible={"xy_mode": ["summary", "both"]}),
    opt.choice("error", "Error bars", ERROR, default="sd", section="Points & lines",
               visible={"xy_mode": ["summary", "both"]}),
    opt.choice("error_style", "Error style", [("bars", "Bars"), ("band", "Shaded band")], default="bars",
               section="Points & lines", visible={"xy_mode": ["summary", "both"]}),
    opt.number("error_cap", "Error bar cap width (pt)", 3, min=0, max=20, step=0.5, section="Points & lines",
               visible={"xy_mode": ["summary", "both"]}),
    opt.boolean("show_points", "Show points", True, section="Points & lines"),
    opt.number("point_size", "Point size (pt)", 6, min=0.5, max=30, step=0.5, section="Points & lines"),
    opt.choice("point_shape", "Point shape", SHAPES + [("auto", "Different shape per series")],
               default="circle", section="Points & lines"),
    opt.number("point_alpha", "Point opacity", 0.9, min=0.05, max=1, step=0.05, section="Points & lines"),
    opt.choice("connect", "Connect points", [("none", "No line"), ("line", "Straight lines"), ("step", "Steps")],
               default="none", section="Points & lines"),
    opt.number("line_width", "Line thickness (pt)", 1.5, min=0.1, max=8, step=0.1, section="Points & lines"),
    opt.choice("line_style", "Line style", [("solid", "Solid"), ("dashed", "Dashed"), ("dotted", "Dotted"),
                                            ("auto", "Different per series")], default="solid", section="Points & lines"),
    opt.number("size_max", "Largest bubble (pt)", 22, min=4, max=80, step=1, section="Points & lines"),
    opt.choice("fit", "Fit a curve", compute.FIT_CHOICES, default="none", section="Curve fit"),
    opt.boolean("fit_band", "Show 95% confidence band", True, section="Curve fit"),
    opt.boolean("fit_legend", "Show fit results in the legend", True, section="Curve fit"),
    opt.number("loess_span", "LOWESS span", 0.75, min=0.15, max=1, step=0.05, section="Curve fit",
               visible={"fit": "loess"}),
    opt.boolean("marginal_rug", "Rug marks on the axes", False, section="Points & lines"),
]

graph(id="xy", name="XY scatter", category="XY",
      description="Points for X–Y data, with optional lines, error bars and fitted curves.",
      features=XY_FEATURES, prepare=faceted(_prep_xy), options=_xy_options)
variant(of="xy", id="xy_line", name="Line graph",
        description="Points joined by lines. Replicates at the same X are averaged with error bars.",
        defaults={"connect": "line", "xy_mode": "summary", "error": "sem"})
variant(of="xy", id="xy_regression", name="Scatter with regression line",
        description="Scatter plot with a fitted straight line and its 95% confidence band.",
        defaults={"fit": "linear"})
variant(of="xy", id="dose_response", name="Dose–response curve",
        description="Log-concentration axis with a fitted 4-parameter logistic curve.",
        defaults={"fit": "4pl", "x_scale": "log10", "xy_mode": "summary", "error": "sem", "fit_band": False})
variant(of="xy", id="bubble", name="Bubble chart",
        description="Scatter plot where a third variable sets the size of each point.",
        defaults={"point_alpha": 0.6})


# ============================================================ DISTRIBUTION
def _dist_long(ctx: Context) -> pd.DataFrame:
    long = ctx.long()
    if long.empty:
        raise AnalysisError("There are no numeric values to plot yet.")
    return long


def _kde(values: np.ndarray, grid: np.ndarray, adjust: float) -> np.ndarray:
    from scipy.stats import gaussian_kde
    if len(values) < 2 or np.ptp(values) == 0:
        return np.zeros_like(grid)
    k = gaussian_kde(values)
    k.set_bandwidth(k.factor * adjust)
    return k(grid)


def _prep_histogram(ctx: Context) -> Prepared:
    """Render type: histogram.

    frames
      bins    group, left, right, mid, height   (height already in the chosen unit)
      curve   group, x, y, kind ("kde" | "normal")   overlay curves in the same unit
      main    value, group                      (for rug marks)
    computed
      groups, stat ("count" | "percent" | "density"), edges [bin edges]
    """
    long = _dist_long(ctx)
    groups = [str(c) for c in long["group"].cat.categories]
    v = long["value"].to_numpy(dtype=float)
    nb = ctx.opt("bins")
    edges = np.histogram_bin_edges(v, bins=int(nb) if nb else "auto")
    if len(edges) > 201:
        edges = np.histogram_bin_edges(v, bins=200)
    width = np.diff(edges)
    stat = ctx.opt("stat", "count")
    rows, curves = [], []
    grid = np.linspace(edges[0] - 0.05 * np.ptp(edges), edges[-1] + 0.05 * np.ptp(edges), 256)
    for g in groups:
        vals = long.loc[long["group"] == g, "value"].to_numpy(dtype=float)
        counts, _ = np.histogram(vals, bins=edges)
        n = max(len(vals), 1)
        if stat == "percent":
            h, scale = counts / n * 100, 100 * width.mean()
        elif stat == "density":
            h, scale = counts / n / width, 1.0
        else:
            h, scale = counts.astype(float), n * width.mean()
        rows.append(pd.DataFrame({"group": g, "left": edges[:-1], "right": edges[1:],
                                  "mid": (edges[:-1] + edges[1:]) / 2, "height": h}))
        if ctx.opt("kde", False) and len(vals) > 1:
            curves.append(pd.DataFrame({"group": g, "x": grid, "y": _kde(vals, grid, float(ctx.opt("bw_adjust", 1.0))) * scale,
                                        "kind": "kde"}))
        if ctx.opt("normal_curve", False) and len(vals) > 1 and np.std(vals, ddof=1) > 0:
            from scipy.stats import norm
            curves.append(pd.DataFrame({"group": g, "x": grid,
                                        "y": norm.pdf(grid, np.mean(vals), np.std(vals, ddof=1)) * scale,
                                        "kind": "normal"}))
    bins = pd.concat(rows, ignore_index=True)
    bins["group"] = pd.Categorical(bins["group"], categories=groups)
    frames = {"bins": bins, "main": long}
    if curves:
        cv = pd.concat(curves, ignore_index=True)
        cv["group"] = pd.Categorical(cv["group"], categories=groups)
        frames["curve"] = cv
    wide = ctx.opt("layout") == "wide"
    return Prepared(frames=frames, computed={"groups": groups, "stat": stat, "edges": edges},
                    color_keys=groups,
                    x_label=(groups[0] if len(groups) == 1 else "Value") if wide else (ctx.opt("dep") or ""),
                    y_label={"count": "Count", "percent": "Percent", "density": "Density"}[stat],
                    legend_title="" if wide else (ctx.opt("group") or ""))


DIST_FEATURES = {"axes", "x_num", "y_num", "palette", "legend"}
graph(id="histogram", name="Histogram", category="Distribution",
      description="Frequency distribution, with optional density and normal curves.",
      features=DIST_FEATURES, prepare=_prep_histogram,
      options=[*opt.group_layout(min_groups=1),
               opt.integer("bins", "Number of bins", None, min=2, max=200, section="Bins", placeholder="auto"),
               opt.choice("stat", "Y axis shows", [("count", "Count"), ("percent", "Percent"), ("density", "Density")],
                          default="count", section="Bins"),
               opt.choice("hist_style", "Several groups", [("overlay", "Overlaid"), ("stack", "Stacked"),
                                                           ("dodge", "Side by side"), ("facet", "Separate panels")],
                          default="overlay", section="Bins"),
               opt.number("fill_alpha", "Fill opacity", 0.6, min=0, max=1, step=0.05, section="Bins"),
               opt.choice("outline", "Outline", [("white", "White"), ("match", "Same as fill"), ("black", "Black"),
                                                 ("none", "None")], default="white", section="Bins"),
               opt.boolean("kde", "Overlay density curve", False, section="Curves"),
               opt.boolean("normal_curve", "Overlay normal curve", False, section="Curves"),
               opt.number("bw_adjust", "Density smoothing", 1.0, min=0.1, max=5, step=0.1, section="Curves"),
               opt.number("line_width", "Curve thickness (pt)", 1.5, min=0.1, max=6, step=0.1, section="Curves"),
               opt.boolean("show_rug", "Rug marks", False, section="Curves")])


def _prep_density(ctx: Context) -> Prepared:
    """Render type: density.

    frames
      density  group, x, y        kernel density estimate per group
      main     value, group
    computed
      groups, style ("overlay" | "ridgeline")
    """
    long = _dist_long(ctx)
    groups = [str(c) for c in long["group"].cat.categories]
    v = long["value"].to_numpy(dtype=float)
    pad = 0.15 * (np.ptp(v) or 1.0)
    lo, hi = ctx.options.get("_grid") or (v.min() - pad, v.max() + pad)
    grid = np.linspace(lo, hi, 300)
    rows = []
    for g in groups:
        vals = long.loc[long["group"] == g, "value"].to_numpy(dtype=float)
        rows.append(pd.DataFrame({"group": g, "x": grid, "y": _kde(vals, grid, float(ctx.opt("bw_adjust", 1.0)))}))
    dens = pd.concat(rows, ignore_index=True)
    dens["group"] = pd.Categorical(dens["group"], categories=groups)
    wide = ctx.opt("layout") == "wide"
    return Prepared(frames={"density": dens, "main": long},
                    computed={"groups": groups, "style": ctx.opt("density_style", "overlay")},
                    color_keys=groups,
                    x_label=(groups[0] if len(groups) == 1 else "Value") if wide else (ctx.opt("dep") or ""),
                    y_label="Density" if ctx.opt("density_style", "overlay") == "overlay" else "",
                    legend_title="" if wide else (ctx.opt("group") or ""))


graph(id="density", name="Density plot", category="Distribution",
      description="Smooth kernel density curves, overlaid or as a ridgeline plot.",
      features=DIST_FEATURES, prepare=faceted(_prep_density),
      options=[*opt.group_layout(min_groups=1), *copy.deepcopy(FACET_OPTS),
               opt.choice("density_style", "Style", [("overlay", "Overlaid curves"), ("ridgeline", "Ridgeline (stacked rows)")],
                          default="overlay", section="Density"),
               opt.number("bw_adjust", "Smoothing", 1.0, min=0.1, max=5, step=0.1, section="Density"),
               opt.number("fill_alpha", "Fill opacity", 0.35, min=0, max=1, step=0.05, section="Density"),
               opt.number("line_width", "Line thickness (pt)", 1.5, min=0.1, max=6, step=0.1, section="Density"),
               opt.number("ridge_overlap", "Ridgeline overlap", 1.5, min=0.5, max=5, step=0.1, section="Density",
                          visible={"density_style": "ridgeline"}),
               opt.boolean("show_rug", "Rug marks", False, section="Density")])
variant(of="density", id="ridgeline", name="Ridgeline plot",
        description="One density curve per group, stacked in overlapping rows.",
        defaults={"density_style": "ridgeline", "fill_alpha": 0.7})


def _prep_qq(ctx: Context) -> Prepared:
    """Render type: qq.

    frames
      qq     group, theoretical, sample
      line   group, slope, intercept     reference line through the quartiles
    computed
      groups
    """
    from scipy import stats as st
    long = _dist_long(ctx)
    groups = [str(c) for c in long["group"].cat.categories]
    rows, lines = [], []
    for g in groups:
        vals = np.sort(long.loc[long["group"] == g, "value"].to_numpy(dtype=float))
        n = len(vals)
        if n < 3:
            continue
        theo = st.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        q1, q3 = np.percentile(vals, [25, 75])
        slope = (q3 - q1) / (st.norm.ppf(0.75) - st.norm.ppf(0.25))
        lines.append({"group": g, "slope": slope, "intercept": q1 - slope * st.norm.ppf(0.25)})
        rows.append(pd.DataFrame({"group": g, "theoretical": theo, "sample": vals}))
    if not rows:
        raise AnalysisError("A QQ plot needs at least 3 values per group.")
    qq = pd.concat(rows, ignore_index=True)
    keep = [g for g in groups if (qq["group"] == g).any()]
    qq["group"] = pd.Categorical(qq["group"], categories=keep)
    ln = pd.DataFrame(lines)
    ln["group"] = pd.Categorical(ln["group"], categories=keep)
    return Prepared(frames={"qq": qq, "line": ln}, computed={"groups": keep}, color_keys=keep,
                    x_label="Theoretical normal quantiles", y_label="Sample quantiles")


graph(id="qq", name="Normal QQ plot", category="Distribution",
      description="Compare the distribution of the data with a normal distribution.",
      features=DIST_FEATURES, prepare=faceted(_prep_qq),
      options=[*opt.group_layout(min_groups=1), *copy.deepcopy(FACET_OPTS),
               opt.number("point_size", "Point size (pt)", 5, min=0.5, max=30, step=0.5, section="Points"),
               opt.choice("point_shape", "Point shape", SHAPES, default="circle", section="Points"),
               opt.number("point_alpha", "Point opacity", 0.85, min=0.05, max=1, step=0.05, section="Points"),
               opt.boolean("show_line", "Show reference line", True, section="Points"),
               opt.number("line_width", "Line thickness (pt)", 1.2, min=0.1, max=6, step=0.1, section="Points")])


def _prep_ecdf(ctx: Context) -> Prepared:
    """Render type: ecdf.

    frames
      ecdf   group, x, y     step-function vertices (y in 0–1, or percent when
                             computed.percent), sorted by x within group
    computed
      groups, percent (bool)
    """
    long = _dist_long(ctx)
    groups = [str(c) for c in long["group"].cat.categories]
    pct = ctx.opt("ecdf_unit", "fraction") == "percent"
    rows = []
    for g in groups:
        vals = np.sort(long.loc[long["group"] == g, "value"].to_numpy(dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals) * (100 if pct else 1)
        rows.append(pd.DataFrame({"group": g, "x": np.r_[vals[0], vals], "y": np.r_[0, y]}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = pd.Categorical(df["group"], categories=groups)
    wide = ctx.opt("layout") == "wide"
    return Prepared(frames={"ecdf": df}, computed={"groups": groups, "percent": pct}, color_keys=groups,
                    x_label=(groups[0] if len(groups) == 1 else "Value") if wide else (ctx.opt("dep") or ""),
                    y_label="Cumulative percent" if pct else "Cumulative fraction")


graph(id="ecdf", name="Cumulative distribution", category="Distribution",
      description="Empirical cumulative frequency distribution of each group.",
      features=DIST_FEATURES, prepare=faceted(_prep_ecdf),
      options=[*opt.group_layout(min_groups=1), *copy.deepcopy(FACET_OPTS),
               opt.choice("ecdf_unit", "Y axis shows", [("fraction", "Fraction (0–1)"), ("percent", "Percent")],
                          default="percent", section="Lines"),
               opt.number("line_width", "Line thickness (pt)", 1.6, min=0.1, max=6, step=0.1, section="Lines")])


# ================================================================ SURVIVAL
def event_indicator(ctx: Context, name: str, event_value: str) -> np.ndarray:
    """Boolean event array from a numeric or categorical status column
    (NaN where missing -> returned as float array with NaN)."""
    s = ctx.column(name)
    want = (event_value or "1").strip()
    if isinstance(s.dtype, pd.CategoricalDtype):
        labels = s.astype(object)
        levels = [str(l) for l in s.cat.categories]
        if want not in levels:
            lower = {l.lower(): l for l in levels}
            for cand in (want.lower(), "1", "yes", "true", "event", "dead", "died", "death", "deceased"):
                if cand in lower:
                    want = lower[cand]
                    break
            else:
                raise AnalysisError(
                    f"The event column “{name}” has the values {', '.join(levels)}. "
                    f"Type which of them means “event occurred” in the “Event value” box.")
        return np.array([np.nan if pd.isna(v) else float(str(v) == want) for v in labels])
    try:
        target = float(want)
    except ValueError:
        target = 1.0
    vals = s.to_numpy(dtype=float)
    return np.where(np.isnan(vals), np.nan, (vals == target).astype(float))


def _prep_survival(ctx: Context) -> Prepared:
    """Render type: survival.

    frames
      curve    group, time, surv, lo, hi     Kaplan–Meier step vertices (start at time 0,
               surv 1). Values are fractions, or percent when computed.percent.
               When computed.cumulative the values are already 1 − S.
      censor   group, time, surv             tick positions for censored subjects
      risk     group, time, n                numbers at risk at computed.risk_times
    computed
      groups, percent, cumulative
      risk_times  [x positions of the risk-table columns]
      medians     [{group, median}]  (median None when not reached)
      p, p_label  log-rank test across groups (None with one group)
      summary     [{group, n, events}]
    """
    tv, ev = ctx.var("time"), ctx.var("event")
    gv = ctx.var("group", required=False)
    df = pd.DataFrame({"time": ctx.numeric(tv).to_numpy(),
                       "event": event_indicator(ctx, ev, ctx.opt("event_value", "1"))})
    df["group"] = ctx.factor(gv).astype(object).to_numpy() if gv else "All"
    df = df.dropna().reset_index(drop=True)
    df = df[df["time"] >= 0]
    if df.empty:
        raise AnalysisError("There are no complete time/event rows to plot yet.")
    groups = [g for g in (list(ctx.factor(gv).cat.categories) if gv else ["All"]) if (df["group"] == g).any()]
    pct = bool(ctx.opt("percent", True))
    cum = bool(ctx.opt("cumulative", False))
    mult = 100.0 if pct else 1.0

    def tr(a):
        a = np.asarray(a, dtype=float)
        return (1 - a if cum else a) * mult
    from matplotlib.ticker import MaxNLocator
    tmax = float(ctx.opt("x_max") or df["time"].max())
    ticks = [float(t) for t in MaxNLocator(nbins=6).tick_values(0, tmax) if 0 <= t <= tmax * 1.0001]
    curves, cens, risk, medians, summ = [], [], [], [], []
    for g in groups:
        sub = df[df["group"] == g]
        km = compute.km_curve(sub["time"], sub["event"])
        lo, hi = (tr(km["hi"]), tr(km["lo"])) if cum else (tr(km["lo"]), tr(km["hi"]))
        curves.append(pd.DataFrame({"group": g, "time": km["time"], "surv": tr(km["surv"]), "lo": lo, "hi": hi}))
        cens.append(pd.DataFrame({"group": g, "time": km["censor_time"], "surv": tr(km["censor_surv"])}))
        risk.append(pd.DataFrame({"group": g, "time": ticks, "n": compute.at_risk(sub["time"], ticks)}))
        medians.append({"group": g, "median": km["median"]})
        summ.append({"group": g, "n": km["n"], "events": km["events"]})

    def cat(parts):
        out = pd.concat(parts, ignore_index=True)
        out["group"] = pd.Categorical(out["group"], categories=groups)
        return out
    p = compute.logrank_p(df["time"], df["event"], df["group"]) if len(groups) > 1 else None
    ylab = ("Cumulative incidence" if cum else "Survival probability") + (" (%)" if pct else "")
    return Prepared(frames={"curve": cat(curves), "censor": cat(cens), "risk": cat(risk)},
                    computed={"groups": groups, "percent": pct, "cumulative": cum, "risk_times": ticks,
                              "medians": medians, "summary": summ, "p": p,
                              "p_label": None if p is None else ("Log-rank p < 0.0001" if p < 0.0001 else f"Log-rank p = {p:.4f}")},
                    color_keys=groups, x_label=tv, y_label=ylab, legend_title=gv or "")


graph(id="survival", name="Kaplan–Meier survival curve", category="Survival",
      description="Survival curves with censoring marks, confidence bands and a numbers-at-risk table.",
      features={"axes", "x_num", "y_num", "palette", "legend"}, prepare=_prep_survival,
      options=[opt.variable("time", "Time", opt.CONTINUOUS, section="Data"),
               opt.variable("event", "Event / status", opt.ANY, section="Data"),
               opt.text("event_value", "Event value", "1", section="Data",
                        help="The value in the status column that means the event happened (others are censored)."),
               opt.variable("group", "Groups", opt.ANY, required=False, section="Data"),
               opt.boolean("percent", "Show as percent", True, section="Curves"),
               opt.boolean("cumulative", "Plot cumulative incidence (1 − survival)", False, section="Curves"),
               opt.number("line_width", "Line thickness (pt)", 1.8, min=0.1, max=6, step=0.1, section="Curves"),
               opt.boolean("show_ci", "95% confidence band", False, section="Curves"),
               opt.boolean("show_censors", "Mark censored subjects", True, section="Curves"),
               opt.boolean("median_line", "Mark median survival", False, section="Curves"),
               opt.boolean("show_p", "Show log-rank P value", True, section="Curves"),
               opt.boolean("risk_table", "Numbers-at-risk table", False, section="Curves")])


# ========================================================== PARTS OF WHOLE
def _prep_pie(ctx: Context) -> Prepared:
    """Render type: pie.

    frames
      slices   label (factor), value, pct (0–100), text (ready-made slice label, may be "")
    computed
      total
    """
    if ctx.opt("pie_data", "counts") == "counts":
        cv = ctx.var("category")
        counts = ctx.factor(cv).value_counts(sort=False)
        labels, values = [str(i) for i in counts.index], counts.to_numpy(dtype=float)
    else:
        lv, vv = ctx.var("label"), ctx.var("value")
        sub = pd.DataFrame({"label": ctx.column(lv).astype(object), "value": ctx.numeric(vv)}).dropna()
        agg = sub.groupby("label", sort=False)["value"].sum()
        labels, values = [str(i) for i in agg.index], agg.to_numpy(dtype=float)
    keep = values > 0
    labels, values = [l for l, k in zip(labels, keep) if k], values[keep]
    if len(values) == 0:
        raise AnalysisError("There are no positive values to plot yet.")
    order = np.arange(len(values))
    if ctx.opt("sort_slices", "data") == "desc":
        order = np.argsort(-values, kind="stable")
    elif ctx.opt("sort_slices", "data") == "asc":
        order = np.argsort(values, kind="stable")
    labels, values = [labels[i] for i in order], values[order]
    total = float(values.sum())
    pct = values / total * 100
    mode = ctx.opt("slice_labels", "percent")

    def text(l, v, p):
        vt = f"{v:g}"
        return {"none": "", "percent": f"{p:.1f}%", "value": vt, "label": l,
                "label_percent": f"{l}\n{p:.1f}%", "label_value": f"{l}\n{vt}"}[mode]
    df = pd.DataFrame({"label": pd.Categorical(labels, categories=labels), "value": values, "pct": pct,
                       "text": [text(l, v, p) for l, v, p in zip(labels, values, pct)]})
    return Prepared(frames={"slices": df}, computed={"total": total}, color_keys=labels)


_pie_options = [
    opt.choice("pie_data", "Data", [("counts", "Count the categories of one column"),
                                    ("values", "A label column and a value column")], default="counts", section="Data"),
    opt.variable("category", "Category", opt.ANY, section="Data", visible={"pie_data": "counts"}),
    opt.variable("label", "Labels", opt.ANY, section="Data", visible={"pie_data": "values"}),
    opt.variable("value", "Values", opt.CONTINUOUS, section="Data", visible={"pie_data": "values"}),
    opt.choice("sort_slices", "Slice order", [("data", "As in data"), ("desc", "Largest first"), ("asc", "Smallest first")],
               default="data", section="Slices"),
    opt.number("hole", "Hole size (donut)", 0.0, min=0, max=0.9, step=0.05, section="Slices"),
    opt.choice("slice_labels", "Slice labels", [("percent", "Percent"), ("value", "Value"), ("label", "Name"),
                                                ("label_percent", "Name and percent"), ("label_value", "Name and value"),
                                                ("none", "None")], default="percent", section="Slices"),
    opt.choice("label_position", "Label position", [("inside", "Inside"), ("outside", "Outside")],
               default="inside", section="Slices"),
    opt.number("start_angle", "Start angle (°)", 90, min=0, max=360, step=5, section="Slices"),
    opt.number("slice_gap", "Gap between slices (pt)", 1.0, min=0, max=10, step=0.5, section="Slices"),
]
graph(id="pie", name="Pie chart", category="Parts of whole",
      description="Fractions of a whole as slices.", features={"palette", "legend"},
      prepare=_prep_pie, options=_pie_options)
variant(of="pie", id="donut", name="Donut chart", description="A pie chart with a hole in the middle.",
        defaults={"hole": 0.55})


def _prep_count_bar(ctx: Context) -> Prepared:
    """Render type: count_bar.

    frames
      counts   x (factor), group (factor), count, pct (0–100, within x), text (bar label or "")
    computed
      x_levels, groups, has_group (bool; when False ``group`` equals ``x``),
      position ("dodge" | "stack" | "fill"),  value ("count" | "pct") = which column is the bar height
    """
    xv = ctx.var("x")
    gv = ctx.var("group", required=False)
    df = pd.DataFrame({"x": ctx.factor(xv).to_numpy()})
    df["x"] = pd.Categorical(df["x"], categories=list(ctx.factor(xv).cat.categories))
    if gv:
        df["group"] = pd.Categorical(ctx.factor(gv).to_numpy(), categories=list(ctx.factor(gv).cat.categories))
    else:
        df["group"] = df["x"]
    df = df.dropna()
    if df.empty:
        raise AnalysisError("There are no values to count yet.")
    xs = [str(c) for c in df["x"].cat.categories if (df["x"] == c).any()]
    gs = [str(c) for c in df["group"].cat.categories if (df["group"] == c).any()]
    tab = df.groupby(["x", "group"], observed=True).size().reset_index(name="count")
    tab["x"] = pd.Categorical(tab["x"].astype(str), categories=xs)
    tab["group"] = pd.Categorical(tab["group"].astype(str), categories=gs)
    if gv:
        tab["pct"] = tab["count"] / tab.groupby("x", observed=True)["count"].transform("sum") * 100
    else:
        tab["pct"] = tab["count"] / tab["count"].sum() * 100
    position = ctx.opt("position", "dodge")
    value = "pct" if (position == "fill" or ctx.opt("count_unit", "count") == "percent") else "count"
    lab = ctx.opt("bar_labels", "none")
    tab["text"] = [{"none": "", "count": f"{int(c)}", "percent": f"{p:.0f}%"}[lab] for c, p in zip(tab["count"], tab["pct"])]
    return Prepared(frames={"counts": tab.sort_values(["x", "group"]).reset_index(drop=True)},
                    computed={"x_levels": xs, "groups": gs, "has_group": bool(gv), "position": position, "value": value},
                    color_keys=gs, x_label=xv, y_label="Percent" if value == "pct" else "Count", legend_title=gv or "")


graph(id="count_bar", name="Frequency bar chart", category="Parts of whole",
      description="Counts or percentages of categories; stack or split by a second variable.",
      features={"axes", "y_num", "x_cat", "palette", "flip", "legend"}, prepare=_prep_count_bar,
      options=[opt.variable("x", "Categories", opt.ANY, section="Data"),
               opt.variable("group", "Split by", opt.ANY, required=False, section="Data"),
               opt.choice("position", "Arrangement", [("dodge", "Side by side"), ("stack", "Stacked"),
                                                      ("fill", "Stacked to 100%")], default="dodge", section="Bars"),
               opt.choice("count_unit", "Bar height", [("count", "Count"), ("percent", "Percent")], default="count", section="Bars"),
               opt.choice("bar_labels", "Labels on bars", [("none", "None"), ("count", "Count"), ("percent", "Percent")],
                          default="none", section="Bars"),
               *_bar_opts(width=0.75)])


# ================================================================== MATRIX
def _matrix_payload(rows, cols, values, text) -> tuple[pd.DataFrame, dict]:
    recs = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            recs.append({"row": r, "col": c, "value": values[i][j], "text": text[i][j]})
    cells = pd.DataFrame(recs)
    cells["row"] = pd.Categorical(cells["row"], categories=rows)
    cells["col"] = pd.Categorical(cells["col"], categories=cols)
    return cells, {"rows": rows, "cols": cols, "values": values, "text": text}


def _prep_heatmap(ctx: Context) -> Prepared:
    """Render types: heatmap (and corr_matrix, same payload).

    frames
      cells   row (factor), col (factor), value (NaN = empty cell), text ("" = no label)
    computed
      rows, cols      display order (first row at the TOP)
      values, text    the same data as nested lists [row][col]
      vmin, vmax      colour-scale limits
      legend_label    title for the colour bar
    Colours: options["_colormap"] (hex stops from low to high).
    """
    cols = ctx.vars("columns", min=2)
    lab = ctx.var("label", required=False)
    m = np.column_stack([ctx.numeric(c).to_numpy() for c in cols])
    if lab:
        rows = [("" if pd.isna(v) else str(v)) for v in ctx.column(lab).astype(object)]
    else:
        rows = [str(i + 1) for i in range(m.shape[0])]
    keep = ~np.all(np.isnan(m), axis=1)
    m, rows = m[keep], [r for r, k in zip(rows, keep) if k]
    if m.shape[0] == 0:
        raise AnalysisError("There are no numeric values to plot yet.")
    if m.shape[0] > 400:
        raise AnalysisError("A heat map is limited to 400 rows; this table has more.")
    seen: dict[str, int] = {}
    for i, r in enumerate(rows):          # row labels must be unique for factor levels
        seen[r] = seen.get(r, 0) + 1
        if seen[r] > 1:
            rows[i] = f"{r} ({seen[r]})"
    scale = ctx.opt("scale", "none")
    legend = "Value"
    with np.errstate(all="ignore"):
        if scale == "row":
            m = (m - np.nanmean(m, axis=1, keepdims=True)) / np.nanstd(m, axis=1, ddof=1, keepdims=True)
            legend = "Row z-score"
        elif scale == "column":
            m = (m - np.nanmean(m, axis=0, keepdims=True)) / np.nanstd(m, axis=0, ddof=1, keepdims=True)
            legend = "Column z-score"
    cols = list(cols)
    if ctx.opt("cluster_rows", False):
        order = compute.cluster_order(m)
        m, rows = m[order], [rows[i] for i in order]
    if ctx.opt("cluster_cols", False):
        order = compute.cluster_order(m.T)
        m, cols = m[:, order], [cols[i] for i in order]
    finite = m[np.isfinite(m)]
    vmin = float(finite.min()) if len(finite) else 0.0
    vmax = float(finite.max()) if len(finite) else 1.0
    if ctx.opt("center_zero", False) or scale != "none":
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim
    if ctx.opt("color_min") is not None:
        vmin = float(ctx.opt("color_min"))
    if ctx.opt("color_max") is not None:
        vmax = float(ctx.opt("color_max"))
    digits = int(ctx.opt("value_digits", 2))
    show = bool(ctx.opt("show_values", False))
    values = [[None if not math.isfinite(v) else float(v) for v in row] for row in m]
    text = [[(f"{v:.{digits}f}" if (show and v is not None) else "") for v in row] for row in values]
    cells, comp = _matrix_payload(rows, cols, values, text)
    comp.update(vmin=vmin, vmax=vmax, legend_label=legend)
    return Prepared(frames={"cells": cells}, computed=comp, legend_title=legend)


_matrix_cell_opts = [
    opt.boolean("show_values", "Show values in cells", False, section="Cells"),
    opt.integer("value_digits", "Decimals", 2, min=0, max=6, section="Cells"),
    opt.number("value_size", "Value text size (pt)", None, min=3, max=30, step=0.5, section="Cells", placeholder="auto"),
    opt.choice("cell_border", "Cell borders", [("white", "White"), ("gray", "Gray"), ("black", "Black"), ("none", "None")],
               default="white", section="Cells"),
    opt.boolean("square", "Square cells", False, section="Cells"),
    opt.boolean("show_colorbar", "Show color scale", True, section="Cells"),
]
graph(id="heatmap", name="Heat map", category="Matrix",
      description="A color-coded table of values, with optional clustering and z-scores.",
      features={"axes", "x_cat", "colormap"}, prepare=_prep_heatmap,
      options=[opt.variables("columns", "Columns", opt.CONTINUOUS, min=2, section="Data"),
               opt.variable("label", "Row labels", opt.ANY, required=False, section="Data"),
               opt.choice("scale", "Standardize", [("none", "No"), ("row", "Z-score within rows"),
                                                   ("column", "Z-score within columns")], default="none", section="Data"),
               opt.boolean("cluster_rows", "Cluster rows", False, section="Data"),
               opt.boolean("cluster_cols", "Cluster columns", False, section="Data"),
               opt.boolean("center_zero", "Center the color scale on zero", False, section="Cells"),
               opt.number("color_min", "Color scale minimum", None, section="Cells", placeholder="auto"),
               opt.number("color_max", "Color scale maximum", None, section="Cells", placeholder="auto"),
               *_matrix_cell_opts])


def _prep_corr(ctx: Context) -> Prepared:
    """Render type: heatmap (payload identical to :func:`_prep_heatmap`);
    cells outside the chosen triangle have value NaN and text ""."""
    from scipy import stats as st
    cols = ctx.vars("columns", min=2)
    df = pd.DataFrame({c: ctx.numeric(c).to_numpy() for c in cols})
    method = ctx.opt("method", "pearson")
    k = len(cols)
    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                r[i, j] = 1.0
                continue
            sub = df[[cols[i], cols[j]]].dropna()
            if len(sub) < 3 or sub.iloc[:, 0].nunique() < 2 or sub.iloc[:, 1].nunique() < 2:
                continue
            fn = {"pearson": st.pearsonr, "spearman": st.spearmanr, "kendall": st.kendalltau}[method]
            res = fn(sub.iloc[:, 0], sub.iloc[:, 1])
            r[i, j], p[i, j] = float(res[0]), float(res[1])
    order = list(range(k))
    if ctx.opt("cluster", False):
        order = compute.cluster_order(np.nan_to_num(r))
        r, p, cols = r[np.ix_(order, order)], p[np.ix_(order, order)], [cols[i] for i in order]
    shape = ctx.opt("shape", "full")
    digits = int(ctx.opt("value_digits", 2))
    show, sig = bool(ctx.opt("show_values", True)), bool(ctx.opt("mark_sig", False))
    values, text = [], []
    for i in range(k):
        vr, tr = [], []
        for j in range(k):
            hidden = (shape == "lower" and j > i) or (shape == "upper" and j < i) or not math.isfinite(r[i, j])
            if hidden:
                vr.append(None); tr.append(""); continue
            vr.append(float(r[i, j]))
            t = f"{r[i, j]:.{digits}f}" if show else ""
            if sig and i != j and math.isfinite(p[i, j]) and p[i, j] < 0.05:
                t += ("\n" if t else "") + stars(p[i, j])
            tr.append(t)
        values.append(vr); text.append(tr)
    cells, comp = _matrix_payload(list(cols), list(cols), values, text)
    name = {"pearson": "Pearson r", "spearman": "Spearman ρ", "kendall": "Kendall τ"}[method]
    comp.update(vmin=-1.0, vmax=1.0, legend_label=name)
    return Prepared(frames={"cells": cells}, computed=comp, legend_title=name)


_corr = graph(id="corr_matrix", name="Correlation matrix", category="Matrix",
              description="Heat map of the correlations between every pair of variables.",
              features={"axes", "x_cat", "colormap"}, prepare=_prep_corr,
              options=[opt.variables("columns", "Variables", opt.CONTINUOUS, min=2, section="Data"),
                       opt.choice("method", "Coefficient", [("pearson", "Pearson r"), ("spearman", "Spearman ρ"),
                                                            ("kendall", "Kendall τ")], default="pearson", section="Data"),
                       opt.choice("shape", "Show", [("full", "Full matrix"), ("lower", "Lower triangle"),
                                                    ("upper", "Upper triangle")], default="full", section="Data"),
                       opt.boolean("cluster", "Order variables by clustering", False, section="Data"),
                       opt.boolean("mark_sig", "Mark significant correlations with stars", False, section="Cells"),
                       *copy.deepcopy(_matrix_cell_opts)])
_corr.render_type = "heatmap"
for _o in _corr.options:
    if _o["name"] == "colormap":
        _o["default"] = "rdbu"
    if _o["name"] in ("show_values", "square"):
        _o["default"] = True


# ============================================================= SPECIALIZED
def _prep_bland_altman(ctx: Context) -> Prepared:
    """Render type: bland_altman.

    frames
      main   mean, diff
    computed
      bias, sd, loa_lo, loa_hi, bias_ci [lo, hi], loa_lo_ci [lo, hi], loa_hi_ci [lo, hi], n,
      percent (bool: differences are % of the mean)
    """
    from scipy import stats as st
    a, b = ctx.var("method_a"), ctx.var("method_b")
    sub = pd.DataFrame({"a": ctx.numeric(a), "b": ctx.numeric(b)}).dropna()
    if len(sub) < 3:
        raise AnalysisError("A Bland–Altman plot needs at least 3 complete pairs.")
    mean = (sub["a"] + sub["b"]) / 2
    pct = ctx.opt("diff_type", "absolute") == "percent"
    diff = (sub["a"] - sub["b"]) / mean * 100 if pct else sub["a"] - sub["b"]
    ok = np.isfinite(diff)
    mean, diff = mean[ok], diff[ok]
    n = len(diff)
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    t = st.t.ppf(0.975, n - 1)
    se_b, se_l = sd / math.sqrt(n), sd * math.sqrt(3 / n)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    return Prepared(frames={"main": pd.DataFrame({"mean": mean.to_numpy(), "diff": diff.to_numpy()})},
                    computed={"bias": bias, "sd": sd, "loa_lo": lo, "loa_hi": hi, "n": n, "percent": pct,
                              "bias_ci": [bias - t * se_b, bias + t * se_b],
                              "loa_lo_ci": [lo - t * se_l, lo + t * se_l],
                              "loa_hi_ci": [hi - t * se_l, hi + t * se_l]},
                    color_keys=["Points"], x_label=f"Mean of {a} and {b}",
                    y_label=f"Difference ({a} − {b})" + (", % of mean" if pct else ""))


graph(id="bland_altman", name="Bland–Altman plot", category="Specialized",
      description="Agreement between two measurement methods: difference against mean with limits of agreement.",
      features={"axes", "x_num", "y_num", "palette"}, prepare=_prep_bland_altman,
      options=[opt.variable("method_a", "Method A", opt.CONTINUOUS, section="Data"),
               opt.variable("method_b", "Method B", opt.CONTINUOUS, section="Data"),
               opt.choice("diff_type", "Difference", [("absolute", "A − B"), ("percent", "Percent of the mean")],
                          default="absolute", section="Data"),
               opt.boolean("show_ci", "Shade 95% CIs of bias and limits", True, section="Appearance"),
               opt.boolean("show_line_labels", "Label the lines with their values", True, section="Appearance"),
               opt.number("point_size", "Point size (pt)", 5.5, min=0.5, max=30, step=0.5, section="Appearance"),
               opt.choice("point_shape", "Point shape", SHAPES, default="circle", section="Appearance"),
               opt.number("point_alpha", "Point opacity", 0.85, min=0.05, max=1, step=0.05, section="Appearance"),
               opt.number("line_width", "Line thickness (pt)", 1.2, min=0.1, max=6, step=0.1, section="Appearance")])


def _binary_outcome(ctx: Context, name: str, positive: str) -> tuple[np.ndarray, str, str]:
    f = ctx.factor(name)
    levels = [str(l) for l in f.cat.categories]
    if len(levels) != 2:
        raise AnalysisError(f"The outcome “{name}” must have exactly two values, but it has {len(levels)}.")
    pos = positive if positive in levels else levels[1]
    neg = levels[0] if pos == levels[1] else levels[1]
    y = np.array([np.nan if pd.isna(v) else float(str(v) == pos) for v in f.astype(object)])
    return y, pos, neg


def _prep_roc(ctx: Context) -> Prepared:
    """Render type: roc.

    frames
      roc   name (factor = predictor), fpr, tpr     curve vertices (0–1), from (0,0) to (1,1)
    computed
      curves  [{name, auc, lo, hi, label}]   label e.g. "Marker A (AUC = 0.87)"
      percent (bool)  axes should read 0–100 %; frame values are then already ×100
    """
    y, pos, _ = _binary_outcome(ctx, ctx.var("outcome"), ctx.opt("positive", ""))
    preds = ctx.vars("predictors", min=1)
    pct = bool(ctx.opt("percent", False))
    rows, curves = [], []
    for pname in preds:
        s = ctx.numeric(pname).to_numpy()
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() < 3 or len(np.unique(y[ok])) < 2:
            continue
        rc = compute.roc_curve(y[ok] == 1, s[ok])
        m = 100.0 if pct else 1.0
        rows.append(pd.DataFrame({"name": pname, "fpr": rc["fpr"] * m, "tpr": rc["tpr"] * m}))
        curves.append({"name": pname, "auc": rc["auc"], "lo": rc["lo"], "hi": rc["hi"],
                       "label": f"{pname} (AUC = {rc['auc']:.3f})"})
    if not rows:
        raise AnalysisError("A ROC curve needs a binary outcome and at least one numeric predictor with data in both outcome groups.")
    names = [c["name"] for c in curves]
    df = pd.concat(rows, ignore_index=True)
    df["name"] = pd.Categorical(df["name"], categories=names)
    unit = "%" if pct else ""
    return Prepared(frames={"roc": df}, computed={"curves": curves, "percent": pct, "positive": pos},
                    color_keys=names,
                    x_label=f"1 − Specificity{' (%)' if pct else ''} (false positive rate)",
                    y_label=f"Sensitivity{' (%)' if pct else ''} (true positive rate)")


graph(id="roc", name="ROC curve", category="Specialized",
      description="Sensitivity against 1 − specificity for one or more diagnostic markers.",
      features={"axes", "palette", "legend"}, prepare=_prep_roc,
      options=[opt.variable("outcome", "Outcome (two groups)", opt.ANY, section="Data"),
               opt.level("positive", "Positive / diseased group", ["outcome"], section="Data"),
               opt.variables("predictors", "Markers / predictors", opt.CONTINUOUS, min=1, section="Data"),
               opt.boolean("percent", "Axes in percent", False, section="Appearance"),
               opt.boolean("show_auc", "Show AUC in the legend", True, section="Appearance"),
               opt.boolean("show_diagonal", "Show line of identity", True, section="Appearance"),
               opt.boolean("fill_area", "Shade the area under the curve", False, section="Appearance"),
               opt.number("line_width", "Line thickness (pt)", 1.8, min=0.1, max=6, step=0.1, section="Appearance")])


def _prep_volcano(ctx: Context) -> Prepared:
    """Render type: volcano.

    frames
      main   x (effect, e.g. log2 fold change), y (−log10 p), status (factor: "Down",
             "Not significant", "Up"), label (chr; "" when the point is not labelled)
    computed
      x_threshold (≥ 0, draw vertical lines at ±x_threshold when > 0),
      y_threshold (horizontal line at −log10 of the p cut-off),
      counts {Down, "Not significant", Up}
    Colour keys are the three status levels.
    """
    ev, pv = ctx.var("effect"), ctx.var("p")
    lv = ctx.var("label", required=False)
    df = pd.DataFrame({"x": ctx.numeric(ev).to_numpy(), "p": ctx.numeric(pv).to_numpy()})
    df["label"] = (ctx.column(lv).astype(object).map(lambda v: "" if pd.isna(v) else str(v)).to_numpy() if lv else "")
    df = df[np.isfinite(df["x"]) & np.isfinite(df["p"]) & (df["p"] >= 0)].reset_index(drop=True)
    if df.empty:
        raise AnalysisError("There are no complete effect / P value rows to plot yet.")
    if ctx.opt("p_is_log", False):
        df["y"] = df["p"]
    else:
        floor = df.loc[df["p"] > 0, "p"].min() if (df["p"] > 0).any() else 1e-300
        df["y"] = -np.log10(df["p"].clip(lower=floor * 0.1))
    fc = abs(float(ctx.opt("fc_threshold", 1.0)))
    y_thr = -math.log10(float(ctx.opt("p_threshold", 0.05)))
    sig = df["y"] >= y_thr
    status = np.where(sig & (df["x"] >= fc) & (df["x"] > 0), "Up",
                      np.where(sig & (df["x"] <= -fc) & (df["x"] < 0), "Down", "Not significant"))
    levels = ["Down", "Not significant", "Up"]
    df["status"] = pd.Categorical(status, categories=levels)
    top = int(ctx.opt("label_top", 10) or 0)
    show = pd.Series(False, index=df.index)
    if lv and top > 0:
        cand = df[df["status"] != "Not significant"]
        score = cand["y"] * (cand["x"].abs() + 1e-9) ** 0.5
        show.loc[score.sort_values(ascending=False).head(top).index] = True
    df["label"] = np.where(show, df["label"], "")
    p = Prepared(frames={"main": df[["x", "y", "status", "label"]]},
                 computed={"x_threshold": fc, "y_threshold": y_thr,
                           "counts": {l: int((df["status"] == l).sum()) for l in levels}},
                 color_keys=levels, x_label=ev, y_label="−log10 (P value)")
    p.default_colors = {"Down": "#2C6FBB", "Not significant": "#B8BDC4", "Up": "#D7402B"}
    return p


graph(id="volcano", name="Volcano plot", category="Specialized",
      description="Effect size against significance for many comparisons (e.g. differential expression).",
      features={"axes", "x_num", "y_num", "palette", "legend"}, prepare=_prep_volcano,
      options=[opt.variable("effect", "Effect (e.g. log2 fold change)", opt.CONTINUOUS, section="Data"),
               opt.variable("p", "P value", opt.CONTINUOUS, section="Data"),
               opt.boolean("p_is_log", "P column is already −log10(P)", False, section="Data"),
               opt.variable("label", "Labels (e.g. gene)", opt.ANY, required=False, section="Data"),
               opt.number("fc_threshold", "Effect cut-off (±)", 1.0, min=0, step=0.1, section="Thresholds"),
               opt.number("p_threshold", "P value cut-off", 0.05, min=1e-300, max=1, step=0.01, section="Thresholds"),
               opt.boolean("show_thresholds", "Draw cut-off lines", True, section="Thresholds"),
               opt.integer("label_top", "Label the top N hits", 10, min=0, max=200, section="Thresholds"),
               opt.number("point_size", "Point size (pt)", 3.5, min=0.5, max=30, step=0.5, section="Points"),
               opt.number("point_alpha", "Point opacity", 0.75, min=0.05, max=1, step=0.05, section="Points"),
               opt.number("label_size", "Label text size (pt)", None, min=3, max=30, step=0.5, section="Points", placeholder="auto")])


def _prep_forest(ctx: Context) -> Prepared:
    """Render type: forest.

    frames
      main   label (factor; first level is drawn at the TOP), est, lo, hi, text
             (e.g. "0.85 (0.70 to 1.02)")
    computed
      ref_line (number or None), show_text (bool)
    """
    lv, ev = ctx.var("label"), ctx.var("estimate")
    lo, hi = ctx.var("lower"), ctx.var("upper")
    df = pd.DataFrame({"label": ctx.column(lv).astype(object).map(_label_text),
                       "est": ctx.numeric(ev), "lo": ctx.numeric(lo), "hi": ctx.numeric(hi)})
    df = df[df["label"].notna() & df["est"].notna()].reset_index(drop=True)
    if df.empty:
        raise AnalysisError("There are no estimates to plot yet.")
    if df["label"].duplicated().any():
        df["label"] = [f"{l} ({i + 1})" if d else l for i, (l, d) in enumerate(zip(df["label"], df["label"].duplicated(keep=False)))]
    if ctx.opt("sort_rows", "data") != "data":
        df = df.sort_values("est", ascending=(ctx.opt("sort_rows") == "asc")).reset_index(drop=True)
    df["label"] = pd.Categorical(df["label"], categories=list(df["label"]))
    d = int(ctx.opt("value_digits", 2))
    df["text"] = [f"{e:.{d}f}" + (f" ({l:.{d}f} to {h:.{d}f})" if math.isfinite(l) and math.isfinite(h) else "")
                  for e, l, h in zip(df["est"], df["lo"], df["hi"])]
    return Prepared(frames={"main": df},
                    computed={"ref_line": ctx.opt("ref_line"), "show_text": bool(ctx.opt("show_text", True))},
                    color_keys=["Estimate"], x_label=ev, y_label="")


graph(id="forest", name="Forest plot", category="Specialized",
      description="Estimates with confidence intervals, one row each — for odds ratios, hazard ratios, effect sizes.",
      features={"axes", "x_num", "palette"}, prepare=_prep_forest,
      options=[opt.variable("label", "Row labels", opt.ANY, section="Data"),
               opt.variable("estimate", "Estimate", opt.CONTINUOUS, section="Data"),
               opt.variable("lower", "Lower confidence limit", opt.CONTINUOUS, section="Data"),
               opt.variable("upper", "Upper confidence limit", opt.CONTINUOUS, section="Data"),
               opt.choice("sort_rows", "Row order", [("data", "As in data"), ("asc", "Ascending"), ("desc", "Descending")],
                          default="data", section="Data"),
               opt.number("ref_line", "Reference line at", 1.0, nullable=True, section="Appearance", placeholder="none",
                          help="Use 1 for ratios and 0 for differences."),
               opt.boolean("show_text", "Print estimates beside the graph", True, section="Appearance"),
               opt.integer("value_digits", "Decimals", 2, min=0, max=6, section="Appearance"),
               opt.number("point_size", "Marker size (pt)", 7, min=0.5, max=30, step=0.5, section="Appearance"),
               opt.choice("point_shape", "Marker shape", SHAPES, default="square", section="Appearance"),
               opt.number("line_width", "Line thickness (pt)", 1.4, min=0.1, max=6, step=0.1, section="Appearance"),
               opt.number("error_cap", "Cap size (pt)", 0, min=0, max=20, step=0.5, section="Appearance")])


def _prep_waterfall(ctx: Context) -> Prepared:
    """Render type: waterfall.

    frames
      main   label (factor in display order, left to right), value, group (factor)
    computed
      groups, has_group (bool), show_labels (bool: draw x tick labels)
    """
    vv = ctx.var("value")
    lv, gv = ctx.var("label", required=False), ctx.var("group", required=False)
    df = pd.DataFrame({"value": ctx.numeric(vv).to_numpy()})
    df["label"] = (ctx.column(lv).astype(object).map(_label_text).to_numpy()
                   if lv else [str(i + 1) for i in range(len(df))])
    df["group"] = ctx.factor(gv).astype(object).to_numpy() if gv else "All"
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise AnalysisError("There are no values to plot yet.")
    if ctx.opt("sort_bars", "desc") != "data":
        df = df.sort_values("value", ascending=(ctx.opt("sort_bars") == "asc"), kind="stable").reset_index(drop=True)
    if df["label"].duplicated().any():
        df["label"] = [f"{l} ({i + 1})" for i, l in enumerate(df["label"])]
    df["label"] = pd.Categorical(df["label"], categories=list(df["label"]))
    groups = [g for g in (list(ctx.factor(gv).cat.categories) if gv else ["All"]) if (df["group"] == g).any()]
    df["group"] = pd.Categorical(df["group"], categories=groups)
    return Prepared(frames={"main": df},
                    computed={"groups": groups, "has_group": bool(gv),
                              "show_labels": bool(lv) and bool(ctx.opt("show_labels", True))},
                    color_keys=groups, x_label="", y_label=vv, legend_title=gv or "")


graph(id="waterfall", name="Waterfall plot", category="Specialized",
      description="One bar per subject, sorted by value — e.g. best change in tumor size.",
      features={"axes", "y_num", "x_cat", "palette", "legend"}, prepare=_prep_waterfall,
      options=[opt.variable("value", "Values", opt.CONTINUOUS, section="Data"),
               opt.variable("label", "Labels (e.g. subject ID)", opt.ANY, required=False, section="Data"),
               opt.variable("group", "Color by", opt.ANY, required=False, section="Data"),
               opt.choice("sort_bars", "Order", [("desc", "Largest first"), ("asc", "Smallest first"), ("data", "As in data")],
                          default="desc", section="Data"),
               opt.boolean("show_labels", "Show labels on the X axis", True, section="Bars"),
               *_bar_opts(width=0.85)])
