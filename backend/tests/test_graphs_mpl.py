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
from analysis_ui.graphs import GRAPHS, style

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


def render(gid, data, options, fmt="svg", dpi=300, editable_text=False) -> bytes:
    """Like ``render_graph`` with the matplotlib engine, but any warning raised
    while *drawing* fails the test (data preparation may warn, e.g. scipy on
    constant data)."""
    from analysis_ui.core.plotstyle import MPL_LOCK
    from analysis_ui.graphs import mpl
    from analysis_ui.graphs.registry import build_payload
    payload, _meta = build_payload({"type": gid, "engine": "matplotlib", "options": options}, data)
    payload.update(format=fmt, dpi=dpi, editable_text=editable_text)
    with MPL_LOCK, warnings.catch_warnings():
        warnings.simplefilter("error")
        return mpl.render(payload)


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
    assert set(GRAPHS) == set(BASE)
    from analysis_ui.graphs import mpl
    assert {g.render_type for g in GRAPHS.values()} <= set(mpl.DRAWERS)


# ------------------------------------------------- every option, one at a time
# Data roles are not appearance options; toggling them needs different data.
SKIP_TYPES = {"variable", "variables", "level", "group_colors"}
NUMBER_VALUES = {"width": [3.0], "height": [2.5], "font_size": [7], "title_size": [16], "axis_title_size": [8],
                 "tick_size": [6], "legend_size": [7], "axis_width": [0], "x_tick_step": [10],
                 "y_tick_step": [10], "bins": [60], "hole": [0.5], "start_angle": [0], "slice_gap": [0],
                 "point_size": [14], "jitter": [0], "bar_width": [1], "fill_alpha": [0],
                 "outline_width": [0], "error_cap": [0], "error_width": [3], "dodge": [0.3],
                 "bw_adjust": [0.3], "line_width": [4], "line_alpha": [0.2], "point_alpha": [0.2],
                 "summary_width": [1], "size_max": [50], "loess_span": [0.3], "ridge_overlap": [4],
                 "label_top": [40], "fc_threshold": [0], "p_threshold": [0.001], "label_size": [5],
                 "value_digits": [0], "value_size": [5], "color_min": [-1], "color_max": [1],
                 "ref_line": [None, 0.8], "x_min": [0.5], "x_max": [1000], "y_min": [1], "y_max": [500]}
TEXT_VALUES = {"title": "A fairly long title for this graph", "subtitle": "Subtitle text", "caption": "Source: test",
               "x_label": "Custom X", "y_label": " ", "legend_title": "Legend", "hlines": "0, 50; 100",
               "vlines": "1, 2.5", "background": "#FFF8E7", "stat_custom": "Control vs Drug A; Drug A vs Drug C",
               "event_value": "1"}


COMMON = {o["name"] for o in style.style_options({"axes", "x_num", "y_num", "x_cat", "legend", "palette",
                                                  "colormap", "flip"})}
# The shared look-and-feel options are swept completely on a few representative
# layouts; on every other type only the values that change the layout a lot.
COMMON_SWEEP = ("column_bar", "xy", "heatmap")
CRITICAL = {"flip": [True], "legend_position": ["top", "inside_tl"], "theme": ["dark"], "x_scale": ["log10"],
            "y_scale": ["log10"], "x_tick_angle": ["45"]}
# Options drawn by code shared between many types: every value once (on the
# first type that has the option), a single value on the other types.
SHARED = {"point_shape", "point_color", "stat_adjust", "stat_test", "stat_comparisons", "stat_label", "error",
          "center", "sort_groups", "outline", "point_layout"}
VARIANTS = {"xy_regression", "xy_line", "dose_response", "bubble", "donut", "ridgeline"}   # same options as their base


def _values(o: dict, current):
    name, kind = o["name"], o.get("type")
    if kind == "bool":
        return [not bool(current)]
    if kind == "choice":
        return [c["value"] for c in o["choices"] if c["value"] != current]
    if kind in ("number", "integer"):
        return NUMBER_VALUES.get(name, [])
    if kind in ("text", "color"):
        return [TEXT_VALUES[name]] if name in TEXT_VALUES else []
    return []


def option_variants():
    out, seen = [], set()
    for gid, case in BASE.items():
        if gid in VARIANTS:
            continue
        base_opts = _case(case)[2]
        for o in GRAPHS[gid].options:
            name = o["name"]
            if o.get("type") in SKIP_TYPES or name in ("layout", "pie_data") or name.startswith("facet"):
                continue                             # facets have their own tests below
            values = _values(o, base_opts.get(name, o.get("default")))
            if name in COMMON and gid not in COMMON_SWEEP:
                values = [v for v in values if v in CRITICAL.get(name, [])] if o.get("type") != "text" else \
                    (values if name in CRITICAL else [])
            if name in SHARED:
                values = values[:1] if name in seen else values
                seen.add(name)
            for v in values:
                out.append(pytest.param(gid, name, v, id=f"{gid}-{name}={v}"))
    return out


def _variant_options(gid, name, value) -> dict:
    options = _case(BASE[gid])[2]
    options.update(width=3.4, height=2.6)
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
    if gid == "heatmap":                             # 150 rows: keep the sweep fast
        options.setdefault("show_values", False)
        options.setdefault("square", False)
    if name in ("value_size", "value_digits"):
        options["show_values"] = True
    return options


def test_option_schema_was_understood():
    """Guard for the generic option sweep below: it must actually find choices and booleans."""
    kinds = {p.values[1] for p in option_variants()}
    assert {"theme", "flip", "legend_position", "whiskers", "hist_style", "x_scale", "grid", "frame"} <= kinds
    assert len(option_variants()) > 400


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


@pytest.mark.parametrize("size", [(1.5, 1.5), (14, 2), (2.5, 9)])
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
            boxes += [t.get_window_extent(r) for t in p._texts.values()]        # title, subtitle, caption
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


# ------------------------------------------------------------------- facets
def build(gid, data, options):
    """Draw and lay out a graph, returning the finished ``Plot`` for inspection."""
    from analysis_ui.core.plotstyle import MPL_LOCK
    from analysis_ui.graphs import mpl
    from analysis_ui.graphs.mpl.base import Plot, build_rc
    from analysis_ui.graphs.registry import build_payload
    from matplotlib import rc_context
    payload, _ = build_payload({"type": gid, "options": options}, data)
    payload.update(format="svg", dpi=100, editable_text=False)
    o = payload["options"]
    with MPL_LOCK, warnings.catch_warnings(), rc_context(rc=build_rc(o["_fonts"], o["_theme"], False)):
        warnings.simplefilter("error")
        p = Plot(payload).begin()
        draw = mpl.DRAWERS[payload["type"]]
        if p.computed.get("facets"):
            p.draw_facets(draw)
        else:
            draw(p)
        p.finish()
        assert len(p.save()) > 1500
        return p


def assert_inside_canvas(p, tol=1.5):
    r, fig = p.renderer, p.fig.bbox
    boxes = [ax.get_tightbbox(r) for ax in p.axes] + [t.get_window_extent(r) for t in p._texts.values()]
    if p.legend is not None:
        boxes.append(p.legend.get_window_extent(r))
    for b in boxes:
        assert b.x0 >= fig.x0 - tol and b.y0 >= fig.y0 - tol and b.x1 <= fig.x1 + tol and b.y1 <= fig.y1 + tol, b


COLUMN_FACET = {"layout": "long", "dep": "Response", "group": "Treatment", "facet": "Sex"}
GROUPED_FACET = {"layout": "long", "dep": "Systolic BP", "x": "Sex", "group": "Hypertension", "facet": "Sex"}
XY_FACET = {"x": "Age", "y": ["Systolic BP"], "group": "Sex", "facet": "Hypertension"}
FACET_CASES = {
    **{g: ("two_way", COLUMN_FACET) for g in ("column_scatter", "column_bar", "box", "violin", "raincloud",
                                               "density", "ridgeline", "ecdf", "qq")},
    "before_after": ("two_way", {"layout": "long", "dep": "Response", "group": "Sex", "facet": "Treatment"}),
    **{g: ("regression", GROUPED_FACET) for g in ("grouped_bar", "grouped_box", "grouped_violin",
                                                   "grouped_points", "interaction_line")},
    "xy": ("regression", XY_FACET), "xy_regression": ("regression", XY_FACET),
    "xy_line": ("regression", XY_FACET), "bubble": ("regression", {**XY_FACET, "size": "BMI"}),
    "dose_response": ("dose_response", {"x": "Concentration (µM)", "y": ["Response (%)"], "facet": "Compound"}),
}
FLIPPABLE = [g for g in FACET_CASES if any(o["name"] == "flip" for o in GRAPHS[g].options)]
STATS = [g for g in FACET_CASES if any(o["name"] == "stat_test" for o in GRAPHS[g].options)]


def facet_case(gid, **extra):
    sname, options = FACET_CASES[gid]
    return gid, sample(sname), {**options, "width": 4.6, "height": 3.4, **extra}


@pytest.mark.parametrize("gid", list(FACET_CASES))
def test_facets_draw_one_panel_per_level(gid):
    gid, data, options = facet_case(gid, title="Faceted", caption="Caption")
    p = build(gid, data, options)
    names = p.computed["facets"]
    assert len(names) >= 2 and len(p.axes) == len(names)
    assert [ax.get_title() for ax in p.axes] == [str(n) for n in names]
    assert len({ax.get_xlim() for ax in p.axes}) == 1 and len({ax.get_ylim() for ax in p.axes}) == 1   # shared
    assert all(not ax.get_xlabel() and not ax.get_ylabel() for ax in p.axes)      # one shared title per axis
    assert_inside_canvas(p)
    pos = [ax.get_position() for ax in p.axes]
    for i, a in enumerate(pos):                      # panels never overlap
        for b in pos[i + 1:]:
            assert a.x1 <= b.x0 + 1e-6 or b.x1 <= a.x0 + 1e-6 or a.y1 <= b.y0 + 1e-6 or b.y1 <= a.y0 + 1e-6
    check_svg(render(gid, data, options), 4.6, 3.4)


@pytest.mark.parametrize("gid", FLIPPABLE)
def test_facets_flipped(gid):
    gid, data, options = facet_case(gid, flip=True, legend_position="bottom")
    p = build(gid, data, options)
    assert len(p.axes) == len(p.computed["facets"])
    assert_inside_canvas(p)


@pytest.mark.parametrize("mode", ["free_y", "free"])
@pytest.mark.parametrize("gid", ["column_bar", "box", "grouped_points", "xy", "density", "ecdf", "qq"])
def test_facets_free_axes(gid, mode):
    gid, data, options = facet_case(gid, facet_axes=mode)
    p = build(gid, data, options)
    xl, yl = {ax.get_xlim() for ax in p.axes}, {ax.get_ylim() for ax in p.axes}
    if gid != "ecdf":                                # (an ECDF always runs from 0 to 100 %)
        assert len(yl) > 1                           # the two panels hold different data
    if gid in ("xy", "density"):
        assert (len(xl) == 1) == (mode == "free_y")
    labelled = [ax for ax in p.axes if any(t.get_visible() and t.get_text() for t in ax.get_yticklabels())]
    assert len(labelled) == len(p.axes)              # free scales: every panel shows its own tick labels
    assert_inside_canvas(p)


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("gid", STATS)
def test_facets_with_brackets(gid, flip):
    gid, data, options = facet_case(gid, stat_test="welch", flip=flip, y_scale="log10" if flip else "linear")
    p = build(gid, data, options)
    assert p.computed["brackets"] and all("facet" in b for b in p.computed["brackets"])
    labels = {b["label"] for b in p.computed["brackets"]}
    for ax, name in zip(p.axes, p.computed["facets"]):
        mine = [b for b in p.computed["brackets"] if str(b["facet"]) == str(name)]
        drawn = [t.get_text() for t in ax.texts if t.get_text() in labels]
        assert len(drawn) == len(mine)               # each panel shows its own brackets, exactly once
    assert len({(ax.get_xlim() if flip else ax.get_ylim()) for ax in p.axes}) == 1
    assert_inside_canvas(p)


@pytest.mark.parametrize("fit", ["linear", "loess"])
def test_facets_with_fits(fit):
    gid, data, options = facet_case("xy", fit=fit, facet_axes="free")
    p = build(gid, data, options)
    assert p.computed["fits"] and all("facet" in f for f in p.computed["fits"])
    for ax, name in zip(p.axes, p.computed["facets"]):
        notes = [t.get_text() for t in ax.texts]
        assert len(notes) == sum(str(f["facet"]) == str(name) for f in p.computed["fits"])
    assert_inside_canvas(p)


@pytest.mark.parametrize("extra", [{"facet_cols": 1}, {"facet_cols": 12}, {"facet_labels": False},
                                   {"legend_position": "inside_tr", "theme": "dark"}, {"width": 2.5, "height": 2.2},
                                   {"legend_position": "left", "x_tick_angle": "90", "hlines": "50"}])
def test_facet_layout_options(extra):
    gid, data, options = facet_case("grouped_bar", **extra)
    p = build(gid, data, options)
    n = len(p.computed["facets"])
    cols = len({round(ax.get_position().x0, 4) for ax in p.axes})
    if "facet_cols" in extra:
        assert cols == min(extra["facet_cols"], n)
    if extra.get("facet_labels") is False:
        assert not any(ax.get_title() for ax in p.axes)
    assert_inside_canvas(p)


def test_many_facets_and_long_names():
    rng = np.random.default_rng(1)
    names = ["A rather long panel name number %d" % i for i in range(7)]
    data = {"columns": [{"name": "Site", "type": "nominal", "values": [n for n in names for _ in range(12)]},
                        {"name": "Arm", "type": "nominal", "values": ["A", "B", "C"] * 28},
                        {"name": "Y", "type": "continuous", "values": list(rng.normal(50, 9, 84).round(1))}]}
    options = {"layout": "long", "dep": "Y", "group": "Arm", "facet": "Site", "stat_test": "welch",
               "width": 6, "height": 5}
    for gid in ("column_scatter", "box"):
        p = build(gid, data, options)
        assert len(p.axes) == 7
        rows = len({round(ax.get_position().y0, 4) for ax in p.axes})
        assert 2 <= rows <= 4                        # a sensible automatic grid
        assert_inside_canvas(p)
    one = {"columns": [dict(c, values=c["values"][:12]) for c in data["columns"]]}
    assert len(build("column_bar", one, options).axes) == 1


def test_without_facet_variable_nothing_changes():
    gid, data, options = facet_case("column_bar")
    plain = {k: v for k, v in options.items() if k != "facet"}
    p = build(gid, data, {**plain, "facet_cols": 3, "facet_axes": "free", "facet_labels": True})
    assert len(p.axes) == 1 and p.ax.get_ylabel() and not p.ax.get_title()
    assert render(gid, data, plain) == render(gid, data, {**plain, "facet_axes": "free", "facet_cols": 2})


# -------------------------------------------------------------- annotations
ANNOTATIONS = [
    {"kind": "text", "x": 0.25, "y": 0.2, "text": "Hello note", "bold": True, "italic": True, "angle": 30,
     "align": "left", "size": 14, "color": "#AA3377"},
    {"kind": "text", "x": 0.5, "y": 0.5, "text": "plain"},
    {"kind": "arrow", "x": 0.2, "y": 0.3, "x2": 0.6, "y2": 0.7, "width": 2.5, "color": "#228833"},
    {"kind": "line", "x": 0.1, "y": 0.9, "x2": 0.9, "y2": 0.9},
    {"kind": "arrow", "x": 0.4, "y": 0.4, "x2": 0.4, "y2": 0.4},          # zero length must not crash
]


@pytest.mark.parametrize("gid", ["column_bar", "xy", "pie", "heatmap", "survival", "histogram", "forest", "roc"])
def test_annotations_render_on_every_kind_of_graph(gid):
    _, data, options = _case(BASE[gid])
    options.update(width=4, height=3, annotations=ANNOTATIONS)
    svg = render(gid, data, options, editable_text=True)
    check_svg(svg, 4, 3)
    assert b"Hello note" in svg and b"#aa3377" in svg.lower() and b"#228833" in svg.lower()


def test_annotations_use_figure_fractions_from_the_top_left_and_do_not_move_the_layout():
    _, data, options = _case(BASE["column_bar"])
    options.update(width=4, height=3)
    plain = build("column_bar", data, options)
    noted = build("column_bar", data, {**options, "annotations": ANNOTATIONS})
    assert np.allclose(plain.ax.get_position().bounds, noted.ax.get_position().bounds)
    text = next(t for t in noted.fig.texts if t.get_text() == "Hello note")
    assert np.allclose(text.get_position(), (0.25, 0.8))         # y is measured downwards
    assert text.get_ha() == "left" and text.get_va() == "center" and text.get_rotation() == 30
    assert text.get_fontsize() == 14 and text.get_fontweight() == "bold" and text.get_fontstyle() == "italic"
    default = next(t for t in noted.fig.texts if t.get_text() == "plain")
    assert default.get_fontsize() == noted.fonts["base"] and default.get_ha() == "center"
    from matplotlib.patches import FancyArrowPatch
    arrows = [a for a in noted.fig.artists if isinstance(a, FancyArrowPatch)]
    assert len(arrows) == 1 and arrows[0].get_linewidth() == 2.5
    top = max(a.get_zorder() for a in noted.fig.axes)
    assert all(a.get_zorder() > top for a in arrows + [text])    # on top of everything
