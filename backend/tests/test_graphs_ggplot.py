"""ggplot2 graph engine: every gallery case, every option value, formats and edge cases.

The whole module is skipped when R (or ggplot2) is not available.
"""
from __future__ import annotations

import importlib.util
import io
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from analysis_ui.core import rbridge
from analysis_ui.core.errors import AnalysisError
from analysis_ui.graphs import GRAPHS, render_graph
from analysis_ui.graphs import style
from analysis_ui.graphs.registry import build_payload

_info = rbridge.r_info()
if not (_info.get("available") and _info.get("packages", {}).get("ggplot2")):
    pytest.skip("R with ggplot2 is not available", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("render_gallery", ROOT / "scripts" / "render_gallery.py")
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

SMALL = {"width": 3.4, "height": 2.8}          # small figures keep the suite fast
XY_LINE = ("dose_response", {"x": "Concentration (µM)", "y": ["Response (%)"], "group": "Compound",
                             "x_scale": "log10"})


@lru_cache(maxsize=None)
def sample(name: str) -> dict:
    return gallery.sample(name)


def case_data(case) -> tuple[str, dict, dict]:
    gid, sname, options = case[0], case[1], case[2]
    if options is None and gid == "xy_line":
        sname, options = XY_LINE
    return gid, sample(sname), dict(options or {})


def case_id(case) -> str:
    return case[0] + ("-" + case[3] if len(case) > 3 else "")


def render(gid, data, options, fmt="svg", **kw) -> bytes:
    img, _meta = render_graph({"type": gid, "engine": "ggplot2", "options": options}, data, fmt=fmt, **kw)
    return img


def svg_size(svg: bytes) -> tuple[float, float]:
    head = svg[:2000].decode("utf-8", "replace")
    tag = re.search(r"<svg[^>]*>", head)
    assert tag, "no <svg> element"
    w = re.search(r"""\swidth=['"]([\d.]+)pt['"]""", tag.group(0))
    h = re.search(r"""\sheight=['"]([\d.]+)pt['"]""", tag.group(0))
    assert w and h, f"SVG size is not given in points: {tag.group(0)}"
    return float(w.group(1)) / 72, float(h.group(1)) / 72


def assert_svg(svg: bytes, width: float, height: float):
    assert svg.lstrip().startswith(b"<?xml") or svg.lstrip().startswith(b"<svg")
    assert svg.rstrip().endswith(b"</svg>")
    assert len(svg) > 3000, "suspiciously small SVG"
    w, h = svg_size(svg)
    assert w == pytest.approx(width, abs=0.01) and h == pytest.approx(height, abs=0.01)
    assert svg.count(b"<path") + svg.count(b"<rect") + svg.count(b"<line") + svg.count(b"<poly") > 5


# ------------------------------------------------------------------ gallery
@pytest.mark.parametrize("case", gallery.CASES, ids=[case_id(c) for c in gallery.CASES])
def test_gallery_case(case):
    gid, data, options = case_data(case)
    svg = render(gid, data, options)
    assert_svg(svg, float(options.get("width", 5.0)), float(options.get("height", 4.0)))


# ------------------------------------------------------- every option value
def _base_cases() -> dict[str, tuple[dict, dict]]:
    """First gallery case of every graph id."""
    out: dict[str, tuple[dict, dict]] = {}
    for case in gallery.CASES:
        gid, data, options = case_data(case)
        out.setdefault(gid, (data, options))
    return out


BASE = _base_cases()
# variants share the option schema of their base type: toggle options only where a different code path is drawn
TOGGLE_IDS = [g for g in BASE if GRAPHS[g].render_type == g] + ["corr_matrix", "ridgeline", "bubble", "donut"]
# richer starting points so that toggles reach the summary / fit / curve code paths
TOGGLE_EXTRA = {"xy": {"xy_mode": "both", "fit": "linear", "connect": "line", "label": "Sex"},
                "histogram": {"kde": True, "normal_curve": True},
                "survival": {"median_line": True}, "box": {"stat_test": "welch"}}
STYLE_NAMES = {o["name"] for o in style.style_options({"x_num", "y_num", "x_cat", "axes", "legend", "palette",
                                                       "colormap", "flip"})}
DATA_ROLES = {"layout", "pie_data"}             # choices that change which data options are required
# graph ids on which the shared style options are exercised (one per family of axes / legend)
STYLE_IDS = ["column_bar", "grouped_box", "xy", "survival", "pie", "heatmap", "forest"]
# style choices are tested exhaustively on the first id, a spot check on the others
SPOT = {"theme": ["dark", "boxed"], "palette": ["colorblind"], "colormap": ["rdbu"], "legend_position": ["top", "inside_bl"],
        "font_family": ["DejaVu Serif"], "x_tick_angle": ["45", "90"], "x_scale": ["log10"], "y_scale": ["log10", "sqrt"],
        "grid": ["both"], "frame": ["box", "none"], "tick_direction": ["in"]}
# options whose values only change the numbers computed in Python, not what the engine has to draw
PREP_ONLY = {"fit": ["4pl", "loess"], "stat_adjust": ["none"], "sort_groups": ["desc"], "center": ["median"],
             "error": ["ci95", "none"], "stat_test": ["welch"]}


def _variations(opt: dict, current, exhaustive=True):
    kind = opt.get("type")
    name = opt["name"]
    if kind == "boolean":
        yield not bool(current if current is not None else opt.get("default"))
    elif kind == "choice":
        values = [c["value"] if isinstance(c, dict) else c[0] for c in opt.get("choices", [])]
        base = current if current is not None else opt.get("default")
        for v in values:
            if v == base or (name in PREP_ONLY and v not in PREP_ONLY[name]):
                continue
            if exhaustive or name not in SPOT or v in SPOT[name]:
                yield v
    elif kind in ("number", "integer") and name not in STYLE_NAMES:
        for key in ("min", "max"):
            v = opt.get(key)
            if v is not None and v != current and abs(v) < 1e6 and (v > 0 or key == "min"):
                yield v


def _toggle_params():
    params = []
    for gid in TOGGLE_IDS:
        options = {**BASE[gid][1], **TOGGLE_EXTRA.get(gid, {})}
        for opt in GRAPHS[gid].options:
            name = opt["name"]
            if name in DATA_ROLES:
                continue
            is_style = name in STYLE_NAMES
            if is_style and gid not in STYLE_IDS:
                continue
            exhaustive = not is_style or gid == STYLE_IDS[0]
            for value in _variations(opt, options.get(name), exhaustive):
                params.append(pytest.param(gid, name, value, id=f"{gid}-{name}={value}"))
    return params


@pytest.mark.parametrize("gid,name,value", _toggle_params())
def test_option_value(gid, name, value):
    data, options = BASE[gid]
    options = {**options, **TOGGLE_EXTRA.get(gid, {}), **SMALL, name: value}
    try:
        svg = render(gid, data, options)
    except AnalysisError as exc:
        try:
            build_payload({"type": gid, "engine": "ggplot2", "options": options}, data)
        except AnalysisError:                  # the combination is rejected before any drawing
            pytest.skip(f"not drawable with this data: {exc}")
        raise
    assert svg.rstrip().endswith(b"</svg>") and len(svg) > 2000


AXIS_STRESS = [
    {"y_scale": "log10"}, {"y_scale": "log2"}, {"y_scale": "sqrt"},
    {"x_scale": "log10"}, {"x_scale": "sqrt"},
    {"y_min": 1, "y_max": 50}, {"x_min": 1, "x_max": 20}, {"y_tick_step": 7}, {"x_tick_step": 3},
    {"hlines": "0, 50; 80", "vlines": "2, 10"}, {"flip": True, "y_scale": "log10", "legend_position": "bottom"},
    {"title": "Title µ", "subtitle": "ρ − ≥ α γ", "caption": "A footnote", "x_label": " ", "y_label": "Custom Y",
     "legend_title": "Legend"},
    {"transparent": True, "background": "#FFEEDD", "axis_width": 0, "font_size": 6, "title_size": 20},
]


@pytest.mark.parametrize("gid", sorted(BASE))
def test_axis_and_label_options(gid):
    data, options = BASE[gid]
    known = {o["name"] for o in GRAPHS[gid].options}
    for extra in AXIS_STRESS:
        extra = {k: v for k, v in extra.items() if k in known}
        if not extra:
            continue
        try:
            svg = render(gid, data, {**options, **SMALL, **extra})
        except AnalysisError as exc:
            pytest.fail(f"{gid} with {extra}: {exc}")
        assert svg.rstrip().endswith(b"</svg>")


# ------------------------------------------------------------------ formats
MAGIC = {"png": b"\x89PNG", "pdf": b"%PDF", "jpg": b"\xff\xd8\xff", "tiff": (b"II*\x00", b"MM\x00*")}


@pytest.mark.parametrize("fmt", ["png", "pdf", "jpg", "tiff"])
@pytest.mark.parametrize("gid", ["column_scatter", "survival", "pie"])
def test_raster_and_pdf_formats(gid, fmt):
    data, options = BASE[gid]
    img = render(gid, data, {**options, "width": 4, "height": 3}, fmt=fmt, dpi=100)
    magic = MAGIC[fmt]
    assert img.startswith(magic if isinstance(magic, tuple) else (magic,))
    if fmt in ("png", "jpg", "tiff"):
        from PIL import Image
        assert Image.open(io.BytesIO(img)).size == (400, 300)


def test_png_dpi_and_transparency():
    from PIL import Image
    data, options = BASE["column_bar"]
    im = Image.open(io.BytesIO(render("column_bar", data, {**options, "transparent": True}, fmt="png", dpi=72)))
    assert im.size == (360, 288)
    assert im.convert("RGBA").getpixel((1, 1))[3] == 0
    im = Image.open(io.BytesIO(render("column_bar", data, {**options, "theme": "dark"}, fmt="png", dpi=72)))
    assert im.convert("RGB").getpixel((1, 1)) == (0x11, 0x18, 0x27)


@pytest.mark.parametrize("gid", ["column_bar", "xy", "survival", "heatmap", "pie"])
def test_editable_text_svg(gid):
    data, options = BASE[gid]
    options = {**options, "title": "Editable µ title"}
    svg = render(gid, data, options, editable_text=True)
    assert_svg(svg, 5.0, 4.0)
    text = svg.decode("utf-8")
    assert "<text" in text and "Editable µ title" in text and "Liberation Sans" in text
    outlined = render(gid, data, options).decode("utf-8")
    assert "Editable" not in outlined            # default: glyph outlines, identical everywhere


def test_pdf_editable_and_fonts():
    data, options = BASE["xy"]
    for family in ("DejaVu Serif", "Liberation Mono"):
        pdf = render("xy", data, {**options, "font_family": family}, fmt="pdf", editable_text=True)
        assert pdf.startswith(b"%PDF") and family.replace(" ", "").encode() in pdf


def test_marker_diameter_is_in_points():
    """point_size is the marker diameter in pt on the page."""
    data, options = BASE["column_scatter"]
    svg = render("column_scatter", data, {**options, "point_size": 12, "show_summary": False, "stat_test": "none",
                                          "point_layout": "aligned", "point_alpha": 1}).decode()
    body = svg[svg.index("</defs>"):]
    widths = []
    for d in re.findall(r'<path [^>]*fill-rule="nonzero"[^>]* d="(M [^"]*C [^"]*)"', body):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", d)]
        widths.append(max(nums[0::2]) - min(nums[0::2]))
    assert widths and all(abs(w - 12) < 0.15 for w in widths)


# --------------------------------------------------------------- edge cases
def wide(columns: dict) -> dict:
    return {"columns": [{"name": k, "type": "continuous", "values": list(map(float, v))} for k, v in columns.items()]}


COLUMN_IDS = ["column_scatter", "column_bar", "box", "violin", "raincloud"]
DIST_IDS = ["histogram", "density", "ridgeline", "ecdf"]
_rng = np.random.default_rng(7)
EDGE_DATA = {
    "single_group": wide({"Only": [3.1, 4.2, 5.5, 2.2, 6.0]}),
    "single_point_each": wide({"A": [1.5], "B": [2.5], "C": [0.5]}),
    "constant": wide({"A": [2, 2, 2, 2], "B": [2, 2, 2, 2]}),
    "negative": wide({"A": [-5, -3, -8, 1], "B": [-1, -2, -0.5, -4]}),
    "thirty_groups": wide({f"Group {i + 1:02d} with a long name": _rng.normal(10 + i, 2, 6) for i in range(30)}),
    "one_big_one_tiny": wide({"A": _rng.normal(0, 1, 300), "B": [1.0]}),
}


@pytest.mark.parametrize("gid", COLUMN_IDS + DIST_IDS)
@pytest.mark.parametrize("name", sorted(EDGE_DATA))
def test_edge_case_groups(gid, name):
    data = EDGE_DATA[name]
    options = {"layout": "wide", "columns": [c["name"] for c in data["columns"]], "show_points": True, **SMALL}
    if gid in COLUMN_IDS and len(data["columns"]) in (2, 3):
        options["stat_test"] = "welch"
    try:
        svg = render(gid, data, options)
    except AnalysisError as exc:
        # only data problems reported by the Python preparation step are acceptable
        try:
            build_payload({"type": gid, "engine": "ggplot2", "options": options}, data)
        except AnalysisError:
            pytest.skip(str(exc))
        raise
    assert svg.rstrip().endswith(b"</svg>")


@pytest.mark.parametrize("extra", [{"flip": True}, {"y_scale": "log10"}, {"x_tick_angle": "90"},
                                   {"legend_position": "bottom"}, {"stat_test": "welch", "stat_comparisons": "adjacent"}])
def test_thirty_groups_variants(extra):
    data = EDGE_DATA["thirty_groups"]
    options = {"layout": "wide", "columns": [c["name"] for c in data["columns"]], **extra}
    for gid in ("column_bar", "box"):
        assert_svg(render(gid, data, options), 5.0, 4.0)


def test_before_after_with_missing_pairs():
    data = wide({"Before": [1, 2, 3, 4], "After": [2, 3, 5, 3], "Later": [3, 3, 6, 2]})
    data["columns"][1]["values"][2] = None
    svg = render("before_after", data, {"layout": "wide", "columns": ["Before", "After", "Later"],
                                        "line_color": "direction", "show_summary": True, "stat_test": "paired_t"})
    assert_svg(svg, 5.0, 4.0)


def test_grouped_with_missing_cells_and_many_subgroups():
    rng = np.random.default_rng(3)
    xs, gs, vals = [], [], []
    for xi in range(4):
        for gi in range(6):
            if (xi, gi) in ((0, 0), (3, 5)):        # empty cells
                continue
            n = 1 if (xi, gi) == (1, 1) else 5
            xs += [f"Dose {xi}"] * n
            gs += [f"Line {gi}"] * n
            vals += list(rng.normal(5 + xi + gi, 1, n))
    data = {"columns": [{"name": "v", "type": "continuous", "values": vals},
                        {"name": "x", "type": "nominal", "values": xs},
                        {"name": "g", "type": "nominal", "values": gs}]}
    options = {"layout": "long", "dep": "v", "x": "x", "group": "g", "stat_test": "welch", **SMALL}
    for gid in ("grouped_bar", "grouped_box", "grouped_violin", "grouped_points", "interaction_line"):
        svg = render(gid, data, {**options, "show_points": True})
        assert svg.rstrip().endswith(b"</svg>")
        svg = render(gid, data, {**options, "flip": True, "legend_position": "top"})
        assert svg.rstrip().endswith(b"</svg>")


def test_xy_edge_cases():
    one = {"columns": [{"name": "x", "type": "continuous", "values": [1.0]},
                       {"name": "y", "type": "continuous", "values": [2.0]}]}
    assert_svg(render("xy", one, {"x": "x", "y": ["y"]}), 5.0, 4.0)
    flat = {"columns": [{"name": "x", "type": "continuous", "values": [1.0, 2.0, 3.0, 4.0]},
                        {"name": "y", "type": "continuous", "values": [5.0, 5.0, 5.0, 5.0]},
                        {"name": "z", "type": "continuous", "values": [1.0, None, 3.0, 2.0]}]}
    for extra in ({}, {"connect": "line", "fit": "linear"}, {"y": ["y", "z"], "point_shape": "auto", "line_style": "auto",
                                                             "connect": "step"},
                  {"size": "z"}, {"x_scale": "log10", "y_scale": "log10", "marginal_rug": True}):
        svg = render("xy", flat, {"x": "x", "y": ["y"], **extra})
        assert svg.rstrip().endswith(b"</svg>")


def test_unknown_render_type_is_a_friendly_error():
    with pytest.raises(AnalysisError, match="cannot draw"):
        rbridge.pool().call({"kind": "graph", "type": "no_such_graph", "options": {"width": 3, "height": 3},
                             "frames": {}, "computed": {}, "format": "svg", "dpi": 72, "editable_text": False,
                             "file": "/tmp/unused.svg"})


def test_worker_survives_errors_and_keeps_rendering():
    """A failed render must not leave a graphics device open or poison the worker."""
    for _ in range(3):
        with pytest.raises(AnalysisError):
            rbridge.pool().call({"kind": "graph", "type": "pie", "options": {"width": 3, "height": 3},
                                 "frames": {}, "computed": {}, "format": "svg", "dpi": 72,
                                 "editable_text": False, "file": "/nonexistent-dir/x.svg"})
    data, options = BASE["pie"]
    assert_svg(render("pie", data, options), 5.0, 4.0)


# ------------------------------------------------------------------- facets
def _facet_data(n_panels: int = 2, long_names: bool = False) -> dict:
    rng = np.random.default_rng(11)
    per = 18
    panel = np.repeat(np.arange(n_panels), per)
    names = [f"Panel {i + 1}" + (" with quite a long descriptive name" if long_names and i == 1 else "") for i in panel]
    return {"columns": [
        {"name": "v", "type": "continuous", "values": list(map(float, rng.normal(20, 4, len(panel)) + panel * 5))},
        {"name": "w", "type": "continuous", "values": list(map(float, rng.normal(22, 4, len(panel)) + panel * 5))},
        {"name": "x", "type": "continuous", "values": list(map(float, rng.uniform(1, 100, len(panel))))},
        {"name": "g", "type": "nominal", "values": (["A", "B", "C"] * (len(panel) // 3 + 1))[:len(panel)]},
        {"name": "h", "type": "nominal", "values": (["u"] * 9 + ["t"] * 9) * n_panels},
        {"name": "f", "type": "nominal", "values": names}]}


COLUMN_FACET = {"layout": "long", "dep": "v", "group": "g", "facet": "f"}
GROUPED_FACET = {"layout": "long", "dep": "v", "x": "g", "group": "h", "facet": "f"}
XY_FACET = {"x": "x", "y": ["v"], "group": "g", "facet": "f"}
FACET_SPECS = {
    **{gid: COLUMN_FACET for gid in ("column_scatter", "column_bar", "box", "violin", "raincloud", "density",
                                     "ridgeline", "ecdf", "qq")},
    "before_after": {"layout": "wide", "columns": ["v", "w"], "facet": "f"},
    **{gid: GROUPED_FACET for gid in ("grouped_bar", "grouped_box", "grouped_violin", "grouped_points",
                                      "interaction_line")},
    **{gid: XY_FACET for gid in ("xy", "xy_line", "xy_regression", "bubble")},
}
FACET_VARIANTS = {
    "default": {},
    "free_y": {"facet_axes": "free_y", "legend_position": "bottom"},
    "free": {"facet_axes": "free", "facet_labels": False, "theme": "dark"},
    "flip": {"flip": True},
    "flip_free": {"flip": True, "facet_axes": "free_y", "facet_cols": 1},
    "stats": {"stat_test": "welch", "show_points": True, "fit": "linear", "y_scale": "log10", "hlines": "20"},
    "limits": {"y_min": 5, "y_max": 60, "facet_axes": "free", "x_tick_angle": "45", "title": "T", "caption": "c"},
}


def _facet_options(gid: str, extra: dict) -> dict:
    known = {o["name"] for o in GRAPHS[gid].options}
    return {**FACET_SPECS[gid], **{k: v for k, v in extra.items() if k in known}}


def test_facets_are_offered_for_every_documented_type():
    for gid in FACET_SPECS:
        assert "facet" in {o["name"] for o in GRAPHS[gid].options}, gid


@pytest.mark.parametrize("variant", sorted(FACET_VARIANTS))
@pytest.mark.parametrize("gid", sorted(FACET_SPECS))
def test_facets(gid, variant):
    data = _facet_data(2)
    options = {**_facet_options(gid, FACET_VARIANTS[variant]), "width": 4.5, "height": 3.2}
    svg = render(gid, data, options)
    assert_svg(svg, 4.5, 3.2)


@pytest.mark.parametrize("gid", ["column_bar", "box", "grouped_bar", "xy", "density", "interaction_line"])
def test_facets_many_panels_and_long_names(gid):
    data = _facet_data(7, long_names=True)
    for extra in ({"stat_test": "welch", "fit": "linear"}, {"facet_axes": "free", "facet_cols": 2, "flip": True}):
        svg = render(gid, data, {**_facet_options(gid, extra), "width": 6, "height": 4.5})
        assert_svg(svg, 6, 4.5)


def test_single_facet_and_png_size():
    from PIL import Image
    data = _facet_data(1)
    for gid in ("column_scatter", "xy", "ecdf"):
        assert_svg(render(gid, data, _facet_options(gid, {"stat_test": "welch"})), 5.0, 4.0)
    img = render("box", _facet_data(3), {**COLUMN_FACET, "width": 4, "height": 3}, fmt="png", dpi=100)
    assert Image.open(io.BytesIO(img)).size == (400, 300)


def test_facet_panels_show_strip_titles_and_per_panel_fit_text():
    data = _facet_data(2)
    options = {**XY_FACET, "fit": "linear", "group": None, "width": 7, "height": 3.5}
    text = render("xy", data, options, editable_text=True).decode()
    assert "Panel 1" in text and "Panel 2" in text
    assert text.count("R²") == 2                       # one equation inside each panel
    hidden = render("xy", data, {**options, "facet_labels": False}, editable_text=True).decode()
    assert "Panel 1" not in hidden


def test_facet_options_without_facet_variable_change_nothing():
    data, options = BASE["column_bar"]
    plain = render("column_bar", data, options)
    same = render("column_bar", data, {**options, "facet_cols": 3, "facet_axes": "free", "facet_labels": False})
    assert plain == same


# -------------------------------------------------------------- annotations
ANNOTATIONS = [
    {"kind": "text", "x": 0.25, "y": 0.2, "text": "Annotated µ", "bold": True, "italic": True, "angle": 30,
     "size": 13, "color": "#CC0000", "align": "left"},
    {"kind": "text", "x": 0.9, "y": 0.9, "text": "plain", "align": "right"},
    {"kind": "arrow", "x": 0.3, "y": 0.25, "x2": 0.5, "y2": 0.5, "width": 2.5, "color": "#0055AA"},
    {"kind": "line", "x": 0.1, "y": 0.95, "x2": 0.9, "y2": 0.95},
]


@pytest.mark.parametrize("gid", ["column_bar", "xy", "pie", "heatmap", "survival", "forest", "column_bar-facet"])
def test_annotations(gid):
    if gid.endswith("-facet"):
        gid, data, options = "column_bar", sample("two_way"), {"layout": "long", "dep": "Response", "group": "Treatment",
                                                                "facet": "Sex"}
    else:
        data, options = BASE[gid]
    plain = render(gid, data, options)
    svg = render(gid, data, {**options, "annotations": ANNOTATIONS})
    assert_svg(svg, 5.0, 4.0)
    assert len(svg) > len(plain)                       # drawn on top, nothing replaced
    text = render(gid, data, {**options, "annotations": ANNOTATIONS}, editable_text=True).decode()
    assert "Annotated µ" in text and "plain" in text
    for fmt in ("png", "pdf"):
        assert len(render(gid, data, {**options, "annotations": ANNOTATIONS}, fmt=fmt, dpi=72)) > 1000


def test_annotation_position_is_in_figure_fractions_from_top_left():
    data, options = BASE["xy"]
    notes = [{"kind": "text", "x": 0.25, "y": 0.1, "text": "HERE", "align": "left", "size": 10}]
    text = render("xy", data, {**options, "annotations": notes, "width": 6, "height": 4}, editable_text=True).decode()
    m = re.search(r"<text x='([\d.]+)' y='([\d.]+)'[^>]*>HERE</text>", text)
    assert m, "annotation text not found"
    assert float(m.group(1)) == pytest.approx(0.25 * 6 * 72, abs=1.0)
    assert float(m.group(2)) == pytest.approx(0.1 * 4 * 72, abs=6.0)       # baseline of vertically centred text


def test_annotations_do_not_change_the_layout():
    """The plot underneath is identical with and without annotations."""
    data, options = BASE["column_scatter"]
    plain = render("column_scatter", data, options).decode()
    noted = render("column_scatter", data, {**options, "annotations": [ANNOTATIONS[3]]}).decode()
    assert noted.startswith(plain[:plain.rindex("</g>")][:len(plain) // 2])
