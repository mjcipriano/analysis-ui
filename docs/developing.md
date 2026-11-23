# Developer guide

Analysis UI is schema-driven, like jamovi: an analysis or graph type declares
its options in Python; the web app builds the options panel from that schema
and renders whatever results come back. **Adding an analysis or a graph type
never requires touching the frontend.**

```
backend/analysis_ui/
  core/        options schema, Context (data access), Results, registry, R bridge, curve fitting
  analyses/    one module per family; each registers analyses (Python or R)
  graphs/      types.py (schemas + data preparation), compute.py (shared statistics),
               style.py (themes/palettes/fonts), mpl/ (matplotlib engine)
  r/           worker.R (persistent R process), lib/ (helpers), analyses/*.R, graphs/*.R (ggplot2 engine)
  shared/      palettes.json, themes.json (used by both engines)
frontend/      React + TypeScript single-page app
scripts/       make_samples.py, render_gallery.py
```

Run the tests with `cd backend && .venv/bin/python -m pytest -q`.

## 1. Writing a Python analysis

```python
# backend/analysis_ui/analyses/my_family.py   (every module here is auto-imported)
from ..core import analysis, opt, Context, Results, AnalysisError, new_figure, COLORS, stars, fmt_p, p_text

@analysis(
    id="ttest_unpaired", name="Unpaired t test", category="Compare two groups",
    description="Compare the means of two independent groups.",
    keywords="student welch two sample",
    options=[
        *opt.group_layout(),                       # layout / columns / dep / group
        opt.choice("test", "Test", [("welch", "Welch's t test"), ("student", "Student's t test")]),
        opt.boolean("show_plot", "Estimation plot", True, section="Output"),
        opt.alpha(section="Output"),
    ])
def ttest_unpaired(ctx: Context) -> Results:
    groups = ctx.groups(min_groups=2, min_n=2)     # {"Control": array, "Drug": array}
    if len(groups) != 2:
        raise AnalysisError(f"This test compares exactly two groups, but {len(groups)} were selected.")
    r = Results()
    t = r.table("Unpaired t test", [("stat", "t", "num"), ("df", "df", "num"), ("p", "P value", "p")])
    t.add_row(stat=..., df=..., p=...)
    t.note("Two-tailed.")
    r.summary("The means differ significantly (P = 0.0031).")
    if ctx.opt("show_plot"):
        fig, ax = new_figure(5, 3.6)
        ...
        r.plot("Estimation plot", fig)
    r.graph("Create a column graph of these data", "column_scatter",
            {k: ctx.options[k] for k in ("layout", "columns", "dep", "group")})
    return r
```

Categories (fixed list, `core/registry.py`): Describe & explore · Compare two
groups · Compare three or more groups · Factorial & repeated measures ·
Correlation & regression · Curve fitting · Categorical data · Survival ·
Diagnostics & agreement · Multivariate · Planning.

### Options (`core/options.py`)

`opt.variable(name, label, permitted, required=True)`, `opt.variables(name,
label, permitted, min=1, max=None)`, `opt.choice(name, label, [(value,
label)...], default)`, `opt.boolean`, `opt.number` (`default=None` → "auto",
nullable), `opt.integer`, `opt.text`, `opt.level(name, label, sources=[...])`
(pick a level of the variable chosen in another option, or one of the column
names of a `variables` option; "" = first), `opt.alpha()`.
`permitted` ⊂ `("continuous", "nominal", "ordinal")` — shortcuts
`opt.CONTINUOUS`, `opt.CATEGORICAL`, `opt.ANY`.

Every helper accepts `section="…"` (collapsible group in the panel; options
without a section appear first), `visible={"other": value | [values]}` and
`help="…"`.

`opt.group_layout(paired=False, min_groups=2)` adds the standard way of
describing groups, supporting both GraphPad-style column tables
(`layout="wide"`, `columns=[…]`) and tidy data (`layout="long"`, `dep`,
`group`, and `subject` when paired). Always use it for "groups of numbers"
analyses and read the data with the Context helpers below, never directly.
Conventional option names for other roles: `dep` (outcome), `group`,
`factors`, `covariates`, `x`, `y`, `time`, `event`, `subject`.

### Context (`core/context.py`)

| call | returns |
|---|---|
| `ctx.opt(name, default)` | option value (default when None) |
| `ctx.var(name, required=True)` | column name of a `variable` option (raises a friendly error when missing) |
| `ctx.vars(name, min=1)` | list of column names |
| `ctx.numeric(col)` | float Series (categorical labels parsed as numbers) |
| `ctx.factor(col)` | Categorical Series, unused levels dropped, level order = display order |
| `ctx.frame([cols])` | complete-case sub-frame |
| `ctx.groups(min_groups, min_n)` | ordered `{label: float array}` (NaN removed) for either layout |
| `ctx.long(with_subject=False)` | DataFrame `value`, `group` [, `subject`] |
| `ctx.paired()` | one row per subject, one column per condition, complete cases |
| `ctx.df` | the full DataFrame (continuous → float64, nominal/ordinal → Categorical) |

Raise `AnalysisError("message for the user")` for anything the user can fix.

### Results (`core/results.py`)

`r.heading(text)`, `r.table(title, [(key, label, format)…])` → `.add_row(**values)`
/ `.note(text)`, `r.text(content, title=None)`, `r.summary(text)` (plain-language
conclusion, shown prominently), `r.warn(text)`, `r.plot(title, fig)` (matplotlib
figure from `new_figure(width, height, nrows, ncols)`; use `COLORS` for the house
palette), `r.graph(button_label, graph_type, options)` (offers a one-click
editable graph; pass options of that graph type).
Column formats: `text`, `num` (smart significant digits), `int`, `p` (P value;
`<0.0001` when tiny), `pct` (0–1 fraction shown as %), `stars`. Return raw
numbers; the UI formats them. NaN/inf become empty cells.

House rules for results: report what GraphPad Prism reports for the same
analysis (test statistic, df, exact P value, P value summary stars, effect
size with CI, descriptive rows per group), state the test actually run and
assumptions in table footnotes, and end with a one- or two-sentence
`r.summary()` in plain language. Put assumption checks and optional tables
behind boolean options in an "Output" or "Assumption checks" section.

## 2. Writing an R analysis

Declare the schema in Python, compute in R:

```python
# backend/analysis_ui/analyses/r_anova.py
from ..core import r_analysis, opt, prepare_long

r_analysis(id="anova_oneway", name="One-way ANOVA", category="Compare three or more groups",
           script="anova_oneway.R", requires=["emmeans"], prepare=prepare_long,
           options=[*opt.group_layout(min_groups=3), ...])
```

`prepare(ctx) -> (DataFrame, options)` runs in Python first. Ready-made:
`prepare_long` (R receives `data` with `value`, `group`), `prepare_long_subject`
(`value`, `group`, `subject`; complete subjects only). Without `prepare`, R gets
all selected columns with their original names. You can write your own prepare
function next to the `r_analysis` call (e.g. to rename columns to safe names and
pass the mapping in options).

```r
# backend/analysis_ui/r/analyses/anova_oneway.R
run <- function(data, options) {
  fit <- aov(value ~ group, data)
  tab <- as.data.frame(summary(fit)[[1]])
  aui_table("ANOVA table", data.frame(Source = rownames(tab), SS = tab$`Sum Sq`, df = tab$Df,
                                      F = tab$`F value`, p = tab$`Pr(>F)`),
            formats = c(df = "int", p = "p"), labels = c(p = "P value"),
            footnotes = "Type I sums of squares.")
  aui_summary("…")
  aui_plot("Residuals", ggplot2::ggplot(...) + aui_theme(), width = 6, height = 4)
}
```

R helpers (`r/lib/00_core.R`): `aui_opt(options, name, default)`,
`aui_flag(options, name)`, `aui_num(options, name)`, `aui_vars(options, name)`,
`aui_q(colname)` (backtick-quote for formulas), `aui_table(title, df, formats,
labels, footnotes)`, `aui_text`, `aui_summary`, `aui_warn`, `aui_heading`,
`aui_plot(title, ggplot_or_function, width, height)`, `aui_graph(label, type,
options)`, `aui_stop("message for the user")`, `aui_need("pkg")`, `aui_stars(p)`,
`aui_fmt_p(p)`, `aui_theme()`, `AUI_COLORS`. Warnings raised by R are collected
and shown under the results automatically. Factor columns arrive as factors
with the level order chosen in the UI; numeric columns as doubles. With
jsonlite, single-element option vectors arrive as scalars and empty lists as
`list()` — use `aui_vars()` for multi-variable options.

Only packages installed from the distribution are guaranteed (see
`r/worker.R` `info` for the list: ggplot2, emmeans, lme4, lmerTest, car,
survival, multcomp, afex, rstatix, ggpubr, ggbeeswarm, ggsignif, pROC,
patchwork, ggrepel, scales, minpack.lm, nlme, survminer, ggridges, effectsize,
broom, MASS, nortest, pwr, psych, coin, epitools, …). List what you need in
`requires=[…]`; the analysis is greyed out when a package is missing.

## 3. Graphs

A graph spec is `{"type", "engine": "matplotlib" | "ggplot2", "options"}`.
`graphs/types.py` defines each type: its options (data roles + geometry; the
shared look-and-feel options from `graphs/style.py` are appended
automatically) and a `_prep_*` function that converts the user's data into
tidy **frames** and **computed** statistics. The docstring of each `_prep_*`
function is the payload contract for that render type. Engines only draw.

Engine entry points:

* Python: `analysis_ui.graphs.mpl.render(payload) -> bytes`
* R: `aui_render_graph(req)` (in `r/lib/`), which must write the image to `req$file`

`payload` / `req` keys: `type` (render type), `options`, `frames` (pandas
DataFrames / R data frames after `aui_frame()`), `computed`, `format`
(svg | png | pdf | tiff | jpg), `dpi`, `editable_text` (SVG/PDF text stays
text instead of outlines).

Resolved style keys inside `options` (see `style.resolve`):

| key | content |
|---|---|
| `_theme` | `bg`, `panel_bg`, `text`, `axis`, `grid_color`, `grid` (none/x/y/both), `frame` (L/box/none), `tick_dir` (out/in/none), `axis_width` (pt), `title_align` (left/center), `title_bold`, `axis_title_bold`, `transparent` |
| `_fonts` | `family`, and sizes in pt: `base`, `title`, `subtitle`, `axis_title`, `tick`, `legend`, `caption` |
| `_color_names`, `_color_values` | parallel lists: colour for each group / series / slice |
| `_colormap`, `_colormap_diverging` | hex stops low→high for continuous scales |
| `_labels` | final `title`, `subtitle`, `caption`, `x`, `y`, `legend` strings ("" = omit) |
| `_hlines`, `_vlines` | reference line positions |

Plus the plain options `width`, `height` (inches), `x_min/x_max/y_min/y_max`,
`x_scale/y_scale` (linear/log10/log2/sqrt), `x_tick_step/y_tick_step`,
`x_tick_angle` (auto/0/30/45/60/90), `flip`, `legend_position`
(auto/right/top/bottom/left/inside_tr/inside_tl/inside_br/inside_bl/none).

To add a graph type: add a `graph(...)` definition with a `_prep_*` function in
`types.py`, then a renderer for its `render_type` in both engines, and a case in
`scripts/render_gallery.py`.

## 4. HTTP API

| endpoint | purpose |
|---|---|
| `GET /api/catalog` | analyses, graph types (with option schemas), palettes, themes, engines |
| `POST /api/analyses/run` | `{analysis, options, data}` → `{status, results: {elements, warnings}}` |
| `POST /api/graphs/render` | `{graph, data}` → `{status, svg, meta: {groups: [{name, color}], warnings}}` |
| `POST /api/graphs/export` | `{graph, data, format, dpi, editable_text}` → image file |
| `POST /api/import/file`, `/api/import/text` | → `{sheets: [{name, columns}]}` |
| `POST /api/transform` | column transforms and wide/long reshaping |
| `GET /api/samples`, `/api/samples/{id}` | bundled example datasets |

Data on the wire: `{"columns": [{"name", "type": "continuous" | "nominal" |
"ordinal", "values": [...], "levels": [...]}]}`.
