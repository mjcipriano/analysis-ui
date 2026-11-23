"""Render every graph type with both engines into ./gallery_out/ (PNG).

    python scripts/render_gallery.py                 # everything
    python scripts/render_gallery.py matplotlib box  # one engine / ids containing "box"

Used for visual QA of the engines; also a handy set of example specs.
"""
import json
import sys
import time
from pathlib import Path

from analysis_ui import importer
from analysis_ui.graphs import GRAPHS, render_graph

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "backend" / "analysis_ui" / "sample_data"
OUT = ROOT / "gallery_out"

W = {"layout": "wide", "columns": ["Control", "Drug A", "Drug B", "Drug C"]}
G = {"layout": "long", "dep": "Response", "x": "Treatment", "group": "Sex"}
# (graph id, sample, options, optional variant tag)
CASES = [
    ("column_scatter", "treatment_groups", {**W, "stat_test": "welch", "stat_comparisons": "control"}),
    ("column_bar", "treatment_groups", {**W, "stat_test": "welch"}),
    ("column_bar", "treatment_groups", {**W, "theme": "dark", "palette": "set2", "flip": True, "title": "Dark, flipped"}, "dark"),
    ("box", "treatment_groups", {**W, "show_points": True}),
    ("violin", "treatment_groups", {**W, "show_points": True}),
    ("raincloud", "treatment_groups", W),
    ("before_after", "paired_bp", {"layout": "wide", "columns": ["Before", "After"], "stat_test": "paired_t"}),
    ("grouped_bar", "two_way", {**G, "stat_test": "welch"}),
    ("grouped_bar", "two_way", {**G, "position": "stack", "show_points": False, "theme": "minimal"}, "stack"),
    ("grouped_box", "two_way", {**G, "show_points": True}),
    ("grouped_violin", "two_way", G),
    ("grouped_points", "two_way", G),
    ("interaction_line", "two_way", G),
    ("xy", "regression", {"x": "Age", "y": ["Systolic BP"], "group": "Sex"}),
    ("xy_regression", "regression", {"x": "Age", "y": ["Systolic BP"]}),
    ("xy_line", "repeated_measures", None),
    ("dose_response", "dose_response", {"x": "Concentration (µM)", "y": ["Response (%)"], "group": "Compound"}),
    ("bubble", "regression", {"x": "Age", "y": ["Systolic BP"], "size": "BMI", "group": "Sex"}),
    ("xy", "enzyme_kinetics", {"x": "Substrate (mM)", "y": ["Velocity"], "fit": "michaelis_menten", "theme": "boxed"}, "mm"),
    ("histogram", "regression", {"layout": "long", "dep": "Systolic BP", "group": "Sex", "kde": True}),
    ("histogram", "regression", {"layout": "wide", "columns": ["Age"], "normal_curve": True, "theme": "gray"}, "single"),
    ("density", "treatment_groups", W),
    ("ridgeline", "treatment_groups", W),
    ("qq", "treatment_groups", W),
    ("ecdf", "treatment_groups", W),
    ("survival", "survival", {"time": "Months", "event": "Status", "group": "Arm", "risk_table": True, "show_ci": True}),
    ("pie", "contingency", {"pie_data": "counts", "category": "Outcome"}),
    ("donut", "biomarker_panel", {"pie_data": "counts", "category": "Subtype", "slice_labels": "label_percent", "label_position": "outside"}),
    ("count_bar", "contingency", {"x": "Treatment", "group": "Outcome", "bar_labels": "count"}),
    ("count_bar", "contingency", {"x": "Treatment", "group": "Outcome", "position": "fill", "bar_labels": "percent"}, "fill"),
    ("heatmap", "biomarker_panel", {"columns": ["IL-6", "TNF-α", "CRP", "IL-10", "IFN-γ", "VEGF"], "scale": "column", "cluster_rows": True, "colormap": "rdbu"}),
    ("corr_matrix", "biomarker_panel", {"columns": ["IL-6", "TNF-α", "CRP", "IL-10", "IFN-γ", "VEGF"], "mark_sig": True}),
    ("bland_altman", "method_comparison", {"method_a": "Assay A", "method_b": "Assay B"}),
    ("roc", "diagnostic", {"outcome": "Status", "positive": "Disease", "predictors": ["Marker A", "Marker B"]}),
    ("volcano", "volcano", {"effect": "log2FC", "p": "P value", "label": "Gene"}),
    ("forest", "subgroups", {"label": "Subgroup", "estimate": "HR", "lower": "Lower", "upper": "Upper"}),
    ("column_bar", "two_way", {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex", "stat_test": "welch",
                               "annotations": [{"kind": "text", "x": 0.3, "y": 0.2, "text": "Note", "bold": True},
                                               {"kind": "arrow", "x": 0.3, "y": 0.23, "x2": 0.22, "y2": 0.42}]}, "facet"),
    ("box", "two_way", {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex", "show_points": True, "facet_axes": "free_y"}, "facet"),
    ("grouped_bar", "regression", {"layout": "long", "dep": "Systolic BP", "x": "Sex", "group": "Hypertension", "facet": "Sex"}, "facet"),
    ("dose_response", "dose_response", {"x": "Concentration (µM)", "y": ["Response (%)"], "facet": "Compound"}, "facet"),
    ("xy_regression", "regression", {"x": "Age", "y": ["Systolic BP"], "group": "Sex", "facet": "Hypertension", "facet_axes": "free"}, "facet"),
    ("density", "two_way", {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex"}, "facet"),
    ("ecdf", "two_way", {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex"}, "facet"),
    ("qq", "two_way", {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex", "facet_cols": 1}, "facet"),
    ("interaction_line", "two_way", {"layout": "long", "dep": "Response", "x": "Treatment", "facet": "Sex"}, "facet"),
    ("waterfall", "tumor_response", {"value": "Best change (%)", "label": "Patient", "group": "Arm", "hlines": "-30, 20"}),
]


def sample(name):
    sheet = importer.read_file(f"{name}.csv", (SAMPLES / f"{name}.csv").read_bytes())[0]
    return {"columns": sheet["columns"]}


def main():
    args = sys.argv[1:]
    engines = [a for a in args if a in ("matplotlib", "ggplot2")] or ["matplotlib", "ggplot2"]
    filt = [a for a in args if a not in ("matplotlib", "ggplot2")]
    OUT.mkdir(exist_ok=True)
    failures = 0
    for case in CASES:
        gid, sname, options = case[0], case[1], case[2]
        tag = case[3] if len(case) > 3 else ""
        if filt and not any(f in gid for f in filt):
            continue
        data = sample(sname)
        if options is None and gid == "xy_line":       # wide -> long demo for line graphs
            data = sample("dose_response")
            options = {"x": "Concentration (µM)", "y": ["Response (%)"], "group": "Compound", "x_scale": "log10"}
        for engine in engines:
            name = f"{gid}{'-' + tag if tag else ''}.{engine}.png"
            t = time.time()
            try:
                img, meta = render_graph({"type": gid, "engine": engine, "options": options}, data, fmt="png", dpi=110)
                (OUT / name).write_bytes(img)
                print(f"ok    {name:45s} {time.time() - t:5.2f}s {' | '.join(meta['warnings'])}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name:45s} {type(exc).__name__}: {str(exc)[:300]}")
    print("failures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
