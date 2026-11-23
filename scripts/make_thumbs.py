"""Render the small previews shown in the "New graph" gallery
(frontend/public/thumbs/<graph id>.png)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gallery as rg  # noqa: E402

from analysis_ui.graphs import GRAPHS, render_graph  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "thumbs"
OUT.mkdir(parents=True, exist_ok=True)
STYLE = {"width": 3.0, "height": 2.4, "font_size": 8, "legend_position": "none", "title": "", "subtitle": "",
         "caption": "", "theme": "classic", "x_label": " ", "y_label": " "}
EXTRA = {
    "volcano": {"label_top": 0, "point_size": 2.5},
    "survival": {"risk_table": False, "show_p": False},
    "heatmap": {"show_colorbar": False, "label": None},
    "corr_matrix": {"show_colorbar": False, "value_size": 6, "mark_sig": False},
    "forest": {"show_text": False},
    "waterfall": {"show_labels": False},
    "column_scatter": {"stat_test": "none"}, "column_bar": {"stat_test": "none"},
    "before_after": {"stat_test": "none"}, "grouped_bar": {"stat_test": "none"},
    "pie": {"slice_labels": "none"}, "donut": {"slice_labels": "none", "label_position": "inside"},
    "count_bar": {"bar_labels": "none"}, "bland_altman": {"show_line_labels": False},
    "xy": {"point_size": 4}, "bubble": {"size_max": 14}, "roc": {"show_auc": False},
}

done = set()
for case in rg.CASES:
    gid, sname, options = case[0], case[1], case[2]
    if gid in done or gid not in GRAPHS:
        continue
    data = rg.sample(sname)
    if options is None:
        data = rg.sample("dose_response")
        options = {"x": "Concentration (µM)", "y": ["Response (%)"], "group": "Compound", "x_scale": "log10"}
    known = {o["name"] for o in GRAPHS[gid].options}
    opts = {**options, **{k: v for k, v in {**STYLE, **EXTRA.get(gid, {})}.items() if k in known}}
    img, _ = render_graph({"type": gid, "engine": "matplotlib", "options": opts}, data, fmt="png", dpi=110)
    (OUT / f"{gid}.png").write_bytes(img)
    done.add(gid)
missing = set(GRAPHS) - done
print(f"{len(done)} thumbnails written" + (f"; missing: {sorted(missing)}" if missing else ""))
