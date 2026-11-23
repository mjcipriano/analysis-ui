# Analysis UI

Point-and-click statistics and publication-quality graphs in your browser,
computed by **R** and **Python**.

Analysis UI aims to be as easy as GraphPad Prism — paste a table, pick an
analysis, get a results sheet and an editable graph — while the numbers come
from the open-source statistical ecosystems (R: `emmeans`, `afex`, `lme4`,
`survival`, `car` …; Python: `scipy`, `statsmodels`, `lifelines`,
`scikit-learn` …) and the figures are drawn by `ggplot2` or `matplotlib`.
Its architecture follows [jamovi](https://www.jamovi.org): a spreadsheet,
analyses whose option panels are generated from a declarative schema, and
results that update live as you change options or data.

## What it does

* **Data** — a fast spreadsheet (type, paste from Excel, fill, undo/redo).
  Import CSV/TSV, Excel, SPSS, Stata, SAS, R (`.rds`/`.RData`), JSON and
  Parquet, or drop a file on the window. Columns are numeric, categorical or
  ordered; types are detected automatically. Both data layouts work
  everywhere: *one column per group* (Prism style) **or** *values + grouping
  column* (tidy / jamovi style). Transform, normalize, compute with formulas,
  and reshape wide ↔ long.
* **Analyses** — t tests and their nonparametric counterparts, one-way to
  N-way ANOVA, repeated-measures and mixed ANOVA, mixed-effects models,
  multiple comparisons (Tukey, Dunnett, Šidák, Holm, Games–Howell, Dunn, FDR
  …), correlation, linear / multiple / logistic / Poisson / Cox regression,
  nonlinear curve fitting (dose–response, enzyme kinetics, exponentials,
  growth …) with interpolation, contingency tables, survival analysis, ROC
  curves, Bland–Altman, PCA and clustering, power and sample size, and more.
  Every result sheet has exact P values, effect sizes with confidence
  intervals, assumption checks, and a plain-language conclusion.
  → [full list](docs/analyses.md)
* **Graphs** — 30+ graph types (scatter/bar/box/violin/raincloud column
  graphs, grouped graphs, XY with fitted curves, survival curves, histograms,
  heat maps, volcano, forest, waterfall, ROC …). Every graph can be drawn by
  **matplotlib** or **ggplot2** — switch with one click. A *Format graph*
  panel controls everything: themes, journal color palettes and per-group
  colors, fonts, axes (limits, log scales, ticks), error bars, point shapes
  and layouts, significance brackets, reference lines, legend, size. Double-
  click the title, an axis or the data to jump to its settings; copy a
  graph's style and paste it onto another. Export PNG / TIFF / JPEG up to
  1200 dpi, or vector PDF / SVG with editable text. Split any column, grouped
  or XY graph into panels by a variable; add text and arrows by clicking and
  dragging on the graph; combine graphs into multi-panel figures (A, B, C …).
  → [full list](docs/graphs.md)
* **Projects** — everything (tables, analyses, graphs) is kept in one project
  file (`.aui.json`); the session is also auto-saved in the browser.
  Result sheets can be saved as HTML or printed; tables copy straight into
  Excel or Word.

## Quick start

### Docker (nothing else to install)

```bash
docker compose up --build        # then open http://localhost:8765
```

### From source

Requirements: Python ≥ 3.10, Node ≥ 18 (to build the web app once), and —
recommended — R ≥ 4.1.

```bash
./run.sh                         # first run sets everything up, then opens the browser
```

or step by step:

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -e "backend[dev]"
(cd frontend && npm install && npm run build)     # builds into backend/analysis_ui/static
backend/.venv/bin/analysis-ui                     # http://localhost:8765
```

#### Installing R

Without R the Python analyses and matplotlib graphs still work; analyses that
need R are greyed out. To enable everything:

* Ubuntu / Debian: see the `apt-get install` line in the [Dockerfile](Dockerfile)
* any platform: install R, then `Rscript scripts/install_r_packages.R`

If `Rscript` is not on your `PATH`, set `AUI_RSCRIPT=/path/to/Rscript`.

## How it works

```
 browser (React + TypeScript)                 Python (FastAPI)                    R
┌───────────────────────────┐   JSON    ┌──────────────────────────┐  JSON   ┌───────────────────┐
│ spreadsheet · options     │ ───────►  │ registry of analyses and │ ──────► │ persistent worker │
│ panels built from schema  │           │ graph types (schemas)    │         │ analyses/*.R      │
│ results · graph editor    │ ◄───────  │ scipy / statsmodels / …  │ ◄────── │ ggplot2 engine    │
└───────────────────────────┘ tables,   │ matplotlib engine        │ tables, └───────────────────┘
                              SVG       └──────────────────────────┘ SVG
```

* The backend is stateless: the browser sends the columns an analysis needs
  together with its options, and gets back tables, text and plots.
* An analysis is a Python function (or an R script) plus a declarative list
  of options. The options panel, defaults, validation and results rendering
  are generic — **adding an analysis or graph type needs no frontend code.**
* For graphs, Python computes every statistic once (error bars, significance
  brackets, fitted curves, Kaplan–Meier steps …) and hands tidy data to the
  chosen drawing engine, so matplotlib and ggplot2 show identical numbers.

See the [developer guide](docs/developing.md) for how to add analyses and
graph types.

## Development

```bash
backend/.venv/bin/analysis-ui --no-browser --reload   # API on :8765
(cd frontend && npm run dev)                          # app on :5173 with hot reload (proxies /api)

(cd backend && .venv/bin/python -m pytest -q)         # test-suite (R tests are skipped without R)
backend/.venv/bin/python scripts/render_gallery.py    # render every graph type with both engines
backend/.venv/bin/python scripts/make_docs.py         # regenerate docs/analyses.md, docs/graphs.md
```

Settings (environment variables): `AUI_RSCRIPT` (path to Rscript),
`AUI_R_WORKERS` (R processes, default 2), `AUI_R_TIMEOUT` (seconds, default
120).

## Notes and limitations

* Analysis UI is a local, single-user application: it has no accounts or
  access control, so do not expose the server to an untrusted network.
* Statistical results are validated in the test-suite against reference
  implementations (R, scipy, statsmodels, lifelines), but this is young
  software — check critical results independently.
* Where a method differs from GraphPad Prism's default, the table footnote names
  the method that was used.

## License

MIT. Analysis UI is an independent project; it is not affiliated with
GraphPad Software or the jamovi project, and contains no code from either.
