"""matplotlib graph engine: every gallery case renders, every option of every
graph type can be changed without an exception, and edge-case data degrade
gracefully."""
from __future__ import annotations

import importlib.util
import re
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from analysis_ui.core.errors import AnalysisError
from analysis_ui.graphs import GRAPHS, render_graph

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("render_gallery", ROOT / "scripts" / "render_gallery.py")
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)


@lru_cache(maxsize=None)
def sample(name: str) -> dict:
    return gallery.sample(name)


def _case(case):
    gid, sname, options = case[0], case[1], case[2]
    data = sample(sname)
    if options is None and gid == "xy_line":
        data = sample("dose_response")
        options = {"x": "Concentration (µM)", "y": ["Response (%)"], "group": "Compound", "x_scale": "log10"}
    return gid, data, dict(options or {})


CASE_IDS = [c[0] + ("-" + c[3] if len(c) > 3 else "") for c in gallery.CASES]
# one representative data set + options per graph id (first gallery case wins)
BASE: dict[str, tuple] = {}
for _c in gallery.CASES:
    BASE.setdefault(_c[0], _c)


def render(gid, data, options, fmt="svg", **kw) -> bytes:
    with warnings.catch_warnings():
        warnings.simplefilter("error")               # the engine must not emit warnings either
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"(pyparsing|dateutil).*")
        img, _meta = render_graph({"type": gid, "engine": "matplotlib", "options": options}, data, fmt=fmt, **kw)
    return img


def svg_size(svg: bytes) -> tuple[float, float]:
    m = re.search(rb'<svg[^>]*?width="([\d.]+)pt"[^>]*?height="([\d.]+)pt"', svg, re.S) or \
        re.search(rb'<svg[^>]*?height="([\d.]+)pt"[^>]*?width="([\d.]+)pt"', svg, re.S)
    assert m, "no size on the <svg> element"
    a, b = float(m.group(1)), float(m.group(2))
    return (a, b) if b"width" in m.group(0).split(b"height")[0] else (b, a)


def check_svg(svg: bytes, width: float, height: float, min_size: int = 3000):
    assert svg.lstrip().startswith(b"<?xml") and b"</svg>" in svg[-200:]
    assert len(svg) > min_size
    w, h = svg_size(svg)
    assert abs(w - width * 72) < 0.5 and abs(h - height * 72) < 0.5


# ------------------------------------------------------------------ gallery
@pytest.mark.parametrize("case", gallery.CASES, ids=CASE_IDS)
def test_gallery_case_renders_svg(case):
    gid, data, options = _case(case)
    options.update(width=4.5, height=3.5)
    svg = render(gid, data, options)
    check_svg(svg, 4.5, 3.5)
    assert b"<text" not in svg                       # text is converted to outlines by default


def test_every_graph_type_has_a_gallery_case():
    assert set(GRAPHS) <= set(BASE) | {"donut", "ridgeline"} or set(GRAPHS) == set(BASE)
    from analysis_ui.graphs import mpl
    assert {g.render_type for g in GRAPHS.values()} <= set(mpl.DRAWERS)


# ------------------------------------------------- every option, one at a time
# Data roles are not appearance options; toggling them needs different data.
SKIP_TYPES = {"variable", "variables", "level", "group_colors"}
NUMBER_VALUES = {"width": [3.0], "height": [2.5], "font_size": [7], "title_size": [16], "axis_title_size": [8],
                 "tick_size": [6], "legend_size": [7], "axis_width": [0, 2.5], "x_tick_step": [10],
                 "y_tick_step": [10], "bins": [5, 60], "hole": [0.5], "start_angle": [0], "slice_gap": [0, 4],
                 "point_size": [1, 14], "jitter": [0, 0.4], "bar_width": [0.2, 1], "fill_alpha": [0, 1],
                 "outline_width": [0, 3], "error_cap": [0, 1], "error_width": [3], "dodge": [0.3, 1],
                 "bw_adjust": [0.3, 3], "line_width": [0.3, 4], "line_alpha": [0.2], "point_alpha": [0.2],
                 "summary_width": [1], "size_max": [50], "loess_span": [0.3], "ridge_overlap": [0.5, 4],
                 "label_top": [0, 40], "fc_threshold": [0, 2], "p_threshold": [0.001], "label_size": [5],
                 "value_digits": [0, 4], "value_size": [5], "color_min": [-1], "color_max": [1],
                 "ref_line": [None, 0.8], "x_min": [0.5], "x_max": [1000], "y_min": [1], "y_max": [500]}
TEXT_VALUES = {"title": "A fairly long title for this graph", "subtitle": "Subtitle text", "caption": "Source: test",
               "x_label": "Custom X", "y_label": " ", "legend_title": "Legend", "hlines": "0, 50; 100",
               "vlines": "1, 2.5", "background": "#FFF8E7", "stat_custom": "Control vs Drug A; Drug A vs Drug C",
               "event_value": "1"}


def option_variants():
    out = []
    for gid, case in BASE.items():
        base_opts = _case(case)[2]
        for o in GRAPHS[gid].options:
            name, kind = o["name"], o.get("type")
            if kind in SKIP_TYPES or name in ("layout", "pie_data"):
                continue
            current = base_opts.get(name, o.get("default"))
            if kind in ("boolean", "bool"):
                values = [not bool(current)]
            elif kind == "choice":
                values = [c["value"] if isinstance(c, dict) else c[0] for c in o.get("choices", o.get("options", []))]
                values = [v for v in values if v != current]
            elif kind in ("number", "integer"):
                values = NUMBER_VALUES.get(name, [])
            elif kind in ("text", "color"):
                values = [TEXT_VALUES[name]] if name in TEXT_VALUES else []
            else:
                values = []
            for v in values:
                out.append(pytest.param(gid, name, v, id=f"{gid}-{name}={v}"))
    return out


def _variant_options(gid, name, value) -> dict:
    options = _case(BASE[gid])[2]
    options.update(width=3.6, height=2.8)
    options[name] = value
    # options that only act together with another one
    if name in ("stat_comparisons", "stat_adjust", "stat_label", "stat_hide_ns", "stat_custom"):
        options.setdefault("stat_test", "welch")
        if name == "stat_custom":
            options["stat_comparisons"] = "custom"
    if name in ("error_style", "center", "error") and gid in ("xy", "xy_regression", "bubble"):
        options["xy_mode"] = "summary"
    if name == "loess_span":
        options["fit"] = "loess"
    if name in ("point_layout", "jitter", "point_shape", "point_color", "point_size", "point_alpha"):
        options["show_points"] = True
    if name == "ridge_overlap":
        options["density_style"] = "ridgeline"
    if name in ("value_size", "value_digits"):
        options["show_values"] = True
    return options


def test_option_schema_was_understood():
    """Guard for the generic option sweep below: it must actually find choices and booleans."""
    kinds = {p.values[1] for p in option_variants()}
    assert {"theme", "flip", "legend_position", "whiskers", "hist_style", "x_scale", "grid", "frame"} <= kinds
    assert len(option_variants()) > 500


@pytest.mark.parametrize("gid,name,value", option_variants())
def test_each_option_value_renders(gid, name, value):
    options = _variant_options(gid, name, value)
    data = _case(BASE[gid])[1]
    try:
        svg = render(gid, data, options)
    except AnalysisError:
        return                                       # a friendly refusal from data preparation is fine
    check_svg(svg, options["width"], options["height"], min_size=1500)


# ----------------------------------------------------------- formats & text
@pytest.mark.parametrize("fmt,magic", [("png", b"\x89PNG"), ("pdf", b"%PDF"), ("tiff", (b"II*\x00", b"MM\x00*")),
                                       ("jpg", b"\xff\xd8")])
def test_raster_and_pdf_formats(fmt, magic):
    gid, data, options = _case(BASE["column_bar"])
    img = render(gid, data, {**options, "width": 4, "height": 3}, fmt=fmt, dpi=100)
    assert img.startswith(magic) and len(img) > 2000
    if fmt == "png":
        w, h = int.from_bytes(img[16:20], "big"), int.from_bytes(img[20:24], "big")
        assert (w, h) == (400, 300)                  # exactly width × height inches


@pytest.mark.parametrize("gid", ["column_bar", "xy_regression", "dose_response", "heatmap", "survival", "pie"])
def test_editable_text(gid):
    _, data, options = _case(BASE[gid])
    options.update(title="Editable title", width=4, height=3)
    svg = render(gid, data, options, editable_text=True)
    assert b"<text" in svg and b"Editable title" in svg
    assert b"<text" not in render(gid, data, options)
    pdf = render(gid, data, options, fmt="pdf", editable_text=True)
    assert pdf.startswith(b"%PDF") and b"/Type /Font" in pdf and b"/Subtype /Type3" not in pdf


def test_transparent_background():
    from io import BytesIO

    from PIL import Image
    gid, data, options = _case(BASE["box"])
    opaque = Image.open(BytesIO(render(gid, data, {**options, "width": 3, "height": 2.5}, fmt="png", dpi=60)))
    clear = Image.open(BytesIO(render(gid, data, {**options, "width": 3, "height": 2.5, "transparent": True},
                                      fmt="png", dpi=60)))
    assert opaque.convert("RGBA").getpixel((1, 1))[3] == 255
    assert clear.convert("RGBA").getpixel((1, 1))[3] == 0
    svg = render(gid, data, {**options, "transparent": True, "theme": "dark"})
    assert b"#111827" not in svg                     # no opaque theme background left


def test_render_is_deterministic():
    gid, data, options = _case(BASE["column_scatter"])
    options.update(point_layout="jitter", width=4, height=3)
    assert render(gid, data, options) == render(gid, data, options)


def test_theme_colours_reach_the_svg():
    gid, data, options = _case(BASE["grouped_bar"])
    svg = render(gid, data, {**options, "theme": "dark", "width": 4, "height": 3}).lower()
    assert b"#111827" in svg                         # dark background
    for colour in (b"#3b6fe0", b"#e4572e"):          # first two palette colours
        assert colour in svg


# --------------------------------------------------------------- edge cases
def wide(columns: dict) -> tuple[dict, dict]:
    data = {"columns": [{"name": k, "type": "continuous", "values": list(v)} for k, v in columns.items()]}
    return data, {"layout": "wide", "columns": list(columns)}


COLUMN_TYPES = ["column_scatter", "column_bar", "box", "violin", "raincloud"]
DIST_TYPES = ["histogram", "density", "ridgeline", "qq", "ecdf"]
RNG = np.random.default_rng(3)
EDGE_DATA = {
    "single_group": {"Only": RNG.normal(10, 2, 9).round(2)},
    "single_point": {"A": [5.0, None, None], "B": [7.0, 8.0, 9.5]},
    "all_equal": {"A": [3.0] * 6, "B": [3.0] * 6, "C": [4.0] * 6},
    "negative": {"A": RNG.normal(-20, 5, 8).round(2), "B": RNG.normal(4, 5, 8).round(2)},
    "thirty_groups": {f"A rather long group name {i}": RNG.normal(i, 1, 5).round(2) for i in range(30)},
    "huge_values": {"A": RNG.normal(4e7, 1e7, 12).round(0), "B": RNG.normal(9e7, 1e7, 12).round(0)},
    "many_points": {"A": RNG.normal(0, 1, 2500).round(3), "B": RNG.normal(1, 1, 400).round(3)},
}


@pytest.mark.parametrize("gid", COLUMN_TYPES + DIST_TYPES)
@pytest.mark.parametrize("shape", list(EDGE_DATA))
def test_edge_case_data(gid, shape):
    if shape == "many_points" and gid not in ("column_scatter", "box", "histogram"):
        pytest.skip("covered by three representative types")
    data, options = wide(EDGE_DATA[shape])
    options.update(width=3.6, height=2.8, show_points=True)
    if gid in COLUMN_TYPES and shape != "thirty_groups":
        options.update(stat_test="welch")
    try:
        svg = render(gid, data, options)
    except AnalysisError:
        return                                       # e.g. a QQ plot needs three values per group
    check_svg(svg, 3.6, 2.8, min_size=1500)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("scale", ["log10", "log2", "sqrt"])
@pytest.mark.parametrize("gid", ["column_bar", "box", "grouped_bar", "grouped_points"])
def test_scales_with_brackets_and_flip(gid, scale, flip):
    _, data, options = _case(BASE[gid])
    options.update(y_scale=scale, flip=flip, stat_test="welch", width=3.6, height=2.8)
    check_svg(render(gid, data, options), 3.6, 2.8)


def test_log_scale_with_non_positive_values():
    data, options = wide(EDGE_DATA["negative"])
    for gid in ("column_bar", "column_scatter", "box"):
        check_svg(render(gid, data, {**options, "y_scale": "log10", "width": 3.6, "height": 2.8}), 3.6, 2.8,
                  min_size=1500)


def test_contradictory_axis_limits_do_not_crash():
    _, data, options = _case(BASE["xy"])
    check_svg(render("xy", data, {**options, "x_min": 80, "x_max": 20, "y_min": 500, "width": 4, "height": 3}), 4, 3)
    check_svg(render("xy", data, {**options, "x_scale": "log10", "x_min": -5, "y_tick_step": 0.0001,
                                  "width": 4, "height": 3}), 4, 3)


@pytest.mark.parametrize("size", [(1.5, 1.5), (30, 3), (3, 12)])
def test_extreme_figure_sizes(size):
    for gid in ("grouped_bar", "survival", "heatmap", "pie", "roc"):
        _, data, options = _case(BASE[gid])
        check_svg(render(gid, data, {**options, "width": size[0], "height": size[1], "title": "Title"}),
                  *size, min_size=1500)


def test_nothing_is_clipped_outside_the_canvas():
    """Layout check on the artists themselves: titles, tick labels, legends and
    brackets must lie inside the figure."""
    from analysis_ui.graphs import mpl
    from analysis_ui.graphs.mpl.base import Plot, build_rc
    from analysis_ui.graphs.registry import build_payload
    from matplotlib import rc_context
    from matplotlib.text import Text
    cases = [("column_bar", {"stat_test": "welch", "title": "Title", "caption": "Caption", "legend_position": "right"}),
             ("column_bar", {"stat_test": "welch", "flip": True, "legend_position": "bottom"}),
             ("grouped_box", {"legend_position": "left", "x_tick_angle": "60", "subtitle": "Sub"}),
             ("survival", {"legend_position": "top"}), ("forest", {}), ("bland_altman", {}),
             ("donut", {}), ("corr_matrix", {"x_tick_angle": "90"}), ("waterfall", {})]
    for gid, extra in cases:
        _, data, options = _case(BASE[gid])
        options.update(extra, width=4.5, height=3.5)
        payload, _ = build_payload({"type": gid, "options": options}, data)
        payload.update(format="svg", dpi=100, editable_text=False)
        o = payload["options"]
        with rc_context(rc=build_rc(o["_fonts"], o["_theme"], False)):
            p = Plot(payload).begin()
            mpl.DRAWERS[payload["type"]](p)
            p.finish()
            r = p.renderer
            fig_box = p.fig.bbox
            boxes = [ax.get_tightbbox(r) for ax in p.axes]
            boxes += [t.get_window_extent(r) for t in p.fig.findobj(Text) if t.get_visible() and t.get_text().strip()
                      and t.figure is p.fig and t.axes is None]
            if p.legend is not None:
                boxes.append(p.legend.get_window_extent(r))
            for b in boxes:
                assert b.x0 >= fig_box.x0 - 1.5 and b.y0 >= fig_box.y0 - 1.5, (gid, extra, b)
                assert b.x1 <= fig_box.x1 + 1.5 and b.y1 <= fig_box.y1 + 1.5, (gid, extra, b)


def test_beeswarm_has_no_overlaps_and_is_deterministic():
    from analysis_ui.graphs.mpl.base import beeswarm
    v = np.random.default_rng(0).normal(0, 30, 120)
    x = beeswarm(v, 6.0, 1e9)
    assert np.array_equal(x, beeswarm(v, 6.0, 1e9))
    d = np.hypot(x[:, None] - x[None, :], v[:, None] - v[None, :])
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 6.0 - 1e-3
    assert np.abs(beeswarm(v, 6.0, 10.0)).max() <= 10.0 + 1e-9      # squeezed into the allowed width


def test_unknown_render_type_is_a_friendly_error():
    from analysis_ui.graphs import mpl
    with pytest.raises(AnalysisError):
        mpl.render({"type": "nope", "options": {}, "frames": {}, "computed": {}, "format": "svg"})
