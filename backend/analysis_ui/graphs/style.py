"""The look-and-feel options every graph shares, and their resolution into
concrete values (theme dict, fonts, colours) handed to the drawing engines."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..core import options as opt

SHARED = Path(__file__).resolve().parent.parent / "shared"

FONT_CANDIDATES = [
    ("Liberation Sans", "Arial / Helvetica (Liberation Sans)"),
    ("Arial", "Arial"),
    ("Helvetica", "Helvetica"),
    ("DejaVu Sans", "DejaVu Sans"),
    ("Open Sans", "Open Sans"),
    ("Roboto", "Roboto"),
    ("Lato", "Lato"),
    ("Liberation Serif", "Times (Liberation Serif)"),
    ("Times New Roman", "Times New Roman"),
    ("DejaVu Serif", "DejaVu Serif"),
    ("Liberation Mono", "Courier (Liberation Mono)"),
]


@lru_cache(maxsize=1)
def palettes() -> dict:
    return json.loads((SHARED / "palettes.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def themes() -> list[dict]:
    return json.loads((SHARED / "themes.json").read_text(encoding="utf-8"))["themes"]


@lru_cache(maxsize=1)
def fonts() -> list[tuple[str, str]]:
    """Font families that are actually installed (first one is the default)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    out = [(name, label) for name, label in FONT_CANDIDATES if name in have]
    return out or [("DejaVu Sans", "DejaVu Sans")]


def style_options(features: set[str]) -> list[dict]:
    """Common appearance options. ``features`` selects the relevant ones:

    ``x_num`` / ``y_num``  numeric axis (limits, scale, tick step)
    ``x_cat``              categorical x axis (tick-label angle)
    ``axes``               has axes at all (grid, frame, flip, reference lines)
    ``legend``             has a legend
    ``palette``            discrete colours per group
    ``colormap``           continuous colour scale
    """
    o: list[dict] = []
    s = "Titles & labels"
    o += [opt.text("title", "Title", section=s),
          opt.text("subtitle", "Subtitle", section=s)]
    if "axes" in features:
        o += [opt.text("x_label", "X axis title", section=s, placeholder="automatic",
                       help="Leave empty for automatic; type a single space to hide."),
              opt.text("y_label", "Y axis title", section=s, placeholder="automatic")]
    if "legend" in features:
        o += [opt.text("legend_title", "Legend title", section=s, placeholder="none")]
    o += [opt.text("caption", "Caption / footnote", section=s)]

    s = "Colors & theme"
    o += [opt.choice("theme", "Theme", [(t["id"], t["label"]) for t in themes()],
                     default="publication", section=s)]
    if "palette" in features:
        o += [opt.choice("palette", "Color palette",
                         [(p["id"], p["label"]) for p in palettes()["palettes"]],
                         default="vivid", section=s),
              opt.boolean("palette_reverse", "Reverse palette", False, section=s),
              {"type": "group_colors", "name": "group_colors", "label": "Individual colors",
               "default": {}, "section": s,
               "help": "Click a swatch to give one group its own color."}]
    if "colormap" in features:
        o += [opt.choice("colormap", "Color scale",
                         [(c["id"], c["label"]) for c in palettes()["colormaps"]],
                         default="viridis", section=s),
              opt.boolean("colormap_reverse", "Reverse color scale", False, section=s)]
    o += [opt.color("background", "Background color", "", section=s,
                    help="Overrides the theme background."),
          opt.boolean("transparent", "Transparent background", False, section=s)]

    if "axes" in features:
        s = "Axes"
        scales = [("linear", "Linear"), ("log10", "Log 10"), ("log2", "Log 2"), ("sqrt", "Square root")]
        if "x_num" in features:
            o += [opt.number("x_min", "X minimum", None, section=s, placeholder="auto"),
                  opt.number("x_max", "X maximum", None, section=s, placeholder="auto"),
                  opt.choice("x_scale", "X scale", scales, default="linear", section=s),
                  opt.number("x_tick_step", "X tick interval", None, min=0, section=s, placeholder="auto")]
        if "y_num" in features:
            o += [opt.number("y_min", "Y minimum", None, section=s, placeholder="auto"),
                  opt.number("y_max", "Y maximum", None, section=s, placeholder="auto"),
                  opt.choice("y_scale", "Y scale", scales, default="linear", section=s),
                  opt.number("y_tick_step", "Y tick interval", None, min=0, section=s, placeholder="auto")]
        o += [opt.choice("x_tick_angle", "X tick label angle",
                         [("auto", "Automatic"), ("0", "Horizontal"), ("30", "30°"), ("45", "45°"),
                          ("60", "60°"), ("90", "Vertical")], default="auto", section=s),
              opt.choice("grid", "Grid lines", [("theme", "Theme default"), ("none", "None"),
                                                ("y", "Horizontal"), ("x", "Vertical"), ("both", "Both")],
                         default="theme", section=s),
              opt.choice("frame", "Frame", [("theme", "Theme default"), ("L", "Left & bottom axes"),
                                            ("box", "Full box"), ("none", "No axis lines")],
                         default="theme", section=s),
              opt.number("axis_width", "Axis line thickness (pt)", None, min=0, max=6, step=0.25,
                         section=s, placeholder="theme"),
              opt.choice("tick_direction", "Tick marks", [("out", "Outside"), ("in", "Inside"),
                                                          ("none", "None")], default="out", section=s)]
        if "flip" in features:
            o += [opt.boolean("flip", "Horizontal (swap X and Y)", False, section=s)]
        o += [opt.text("hlines", "Horizontal reference lines", section=s, placeholder="e.g. 0, 50",
                       help="Y values separated by commas."),
              opt.text("vlines", "Vertical reference lines", section=s, placeholder="e.g. 10")]

    if "legend" in features:
        o += [opt.choice("legend_position", "Legend position", [
            ("auto", "Automatic"), ("right", "Right"), ("top", "Top"), ("bottom", "Bottom"),
            ("left", "Left"), ("inside_tr", "Inside, top right"), ("inside_tl", "Inside, top left"),
            ("inside_br", "Inside, bottom right"), ("inside_bl", "Inside, bottom left"),
            ("none", "Hide legend")], default="auto", section="Legend")]

    s = "Fonts"
    o += [opt.choice("font_family", "Font", [("theme", "Default")] + [(n, l) for n, l in fonts()],
                     default="theme", section=s),
          opt.number("font_size", "Base font size (pt)", 11, min=4, max=40, step=0.5, section=s),
          opt.number("title_size", "Title size (pt)", None, min=4, max=60, step=0.5, section=s, placeholder="auto"),
          opt.number("axis_title_size", "Axis title size (pt)", None, min=4, max=60, step=0.5, section=s, placeholder="auto"),
          opt.number("tick_size", "Tick label size (pt)", None, min=4, max=60, step=0.5, section=s, placeholder="auto"),
          opt.number("legend_size", "Legend text size (pt)", None, min=4, max=60, step=0.5, section=s, placeholder="auto")]

    o += [{"type": "annotations", "name": "annotations", "label": "Text and arrows", "default": [],
           "section": "Annotations",
           "help": "Use the Text and Arrow buttons above the graph, then drag items into place."}]

    s = "Size"
    o += [opt.number("width", "Width (inches)", 5.0, min=1.5, max=30, step=0.25, section=s),
          opt.number("height", "Height (inches)", 4.0, min=1.5, max=30, step=0.25, section=s)]
    return o


def clean_annotations(items) -> list[dict]:
    """Validated free-form annotations for the engines.

    Coordinates are fractions of the whole figure with the origin at the
    TOP-LEFT corner (x to the right, y downwards), so they do not depend on
    the data or the engine.

    ``{"kind": "text", "x", "y", "text", "size" (pt or None = base font size),
       "color" ("" = theme text colour), "bold", "italic", "angle" (degrees,
       counter-clockwise), "align" ("left" | "center" | "right")}``
    ``{"kind": "arrow" | "line", "x", "y", "x2", "y2", "color", "width" (pt)}``
       — an arrow has its head at (x2, y2).
    """
    out = []
    for a in items or []:
        if not isinstance(a, dict):
            continue
        try:
            x, y = float(a.get("x")), float(a.get("y"))
        except (TypeError, ValueError):
            continue
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        color = a.get("color") if isinstance(a.get("color"), str) else ""
        kind = a.get("kind")
        if kind == "text":
            text = str(a.get("text") or "").strip()
            if not text:
                continue
            try:
                size = float(a["size"]) if a.get("size") not in (None, "") else None
            except (TypeError, ValueError):
                size = None
            try:
                angle = float(a.get("angle") or 0)
            except (TypeError, ValueError):
                angle = 0.0
            out.append({"kind": "text", "x": x, "y": y, "text": text, "size": size, "color": color,
                        "bold": bool(a.get("bold")), "italic": bool(a.get("italic")), "angle": angle,
                        "align": a.get("align") if a.get("align") in ("left", "center", "right") else "center"})
        elif kind in ("arrow", "line"):
            try:
                x2, y2 = float(a.get("x2")), float(a.get("y2"))
                width = float(a.get("width") or 1.2)
            except (TypeError, ValueError):
                continue
            out.append({"kind": kind, "x": x, "y": y, "x2": min(max(x2, 0.0), 1.0), "y2": min(max(y2, 0.0), 1.0),
                        "color": color, "width": min(max(width, 0.1), 12.0)})
    return out[:100]


def _cycle(colors: list[str], n: int, sequential: bool) -> list[str]:
    if n <= 0:
        return []
    if sequential and n > 1 and len(colors) > 1:
        # spread evenly across the ramp
        import matplotlib.colors as mc
        cmap = mc.LinearSegmentedColormap.from_list("p", colors)
        return [mc.to_hex(cmap(i / (n - 1))) for i in range(n)]
    return [colors[i % len(colors)] for i in range(n)]


def resolve(o: dict, color_keys: list[str], single: bool = False,
            default_colors: dict | None = None) -> dict:
    """Add the private, fully resolved style keys to the options dict.

    ``_theme``        dict: bg, panel_bg, text, axis, grid_color, grid
                      (none|x|y|both), frame (L|box|none), tick_dir
                      (out|in|none), axis_width (pt), title_align
                      (left|center), title_bold, axis_title_bold
    ``_fonts``        dict: family, base, title, subtitle, axis_title, tick,
                      legend, caption (sizes in pt)
    ``_color_names``  / ``_color_values``: parallel lists, one colour per key
    ``_colormap``     list of hex stops (low -> high), ``_colormap_diverging``
    """
    o = dict(o)
    theme = dict(next((t for t in themes() if t["id"] == o.get("theme")), themes()[0]))
    if o.get("grid") and o["grid"] != "theme":
        theme["grid"] = o["grid"]
    if o.get("frame") and o["frame"] != "theme":
        theme["frame"] = o["frame"]
    if o.get("axis_width") is not None:
        theme["axis_width"] = float(o["axis_width"])
    theme["tick_dir"] = o.get("tick_direction") or theme.get("tick_dir", "out")
    if o.get("background"):
        theme["bg"] = theme["panel_bg"] = o["background"]
    theme["transparent"] = bool(o.get("transparent"))
    o["_theme"] = theme

    family = o.get("font_family")
    if not family or family == "theme":
        family = fonts()[0][0]
    base = float(o.get("font_size") or 11) * float(theme.get("font_scale", 1.0))

    def size(key, factor):
        v = o.get(key)
        return float(v) if v else round(base * factor, 2)
    o["_fonts"] = {"family": family, "base": base, "title": size("title_size", 1.2),
                   "subtitle": round(base * 0.95, 2), "axis_title": size("axis_title_size", 1.0),
                   "tick": size("tick_size", 0.9), "legend": size("legend_size", 0.9),
                   "caption": round(base * 0.8, 2)}

    pal = next((p for p in palettes()["palettes"] if p["id"] == o.get("palette")),
               palettes()["palettes"][0])
    colors = list(pal["colors"])
    if o.get("palette_reverse"):
        colors = colors[::-1]
    values = _cycle(colors, len(color_keys), bool(pal.get("sequential")))
    if single and values:
        values = [values[0]] * len(values)
    custom = dict(default_colors or {})
    custom.update({k: v for k, v in (o.get("group_colors") or {}).items() if v})
    values = [custom.get(k) or v for k, v in zip(color_keys, values)]
    o["_color_names"] = list(color_keys)
    o["_color_values"] = values

    cm = next((c for c in palettes()["colormaps"] if c["id"] == o.get("colormap")),
              palettes()["colormaps"][0])
    stops = list(cm["colors"])
    if o.get("colormap_reverse"):
        stops = stops[::-1]
    o["_colormap"] = stops
    o["_colormap_diverging"] = bool(cm.get("diverging"))

    def parse_lines(text):
        out = []
        for part in str(text or "").replace(";", ",").split(","):
            try:
                out.append(float(part))
            except ValueError:
                pass
        return out
    o["_annotations"] = clean_annotations(o.get("annotations"))
    o["_hlines"] = parse_lines(o.get("hlines"))
    o["_vlines"] = parse_lines(o.get("vlines"))
    return o
