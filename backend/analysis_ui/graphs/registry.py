"""Graph registry and the render pipeline.

    spec = {"type": "column_bar", "engine": "matplotlib" | "ggplot2",
            "options": {...}}
    image_bytes, meta = render_graph(spec, data, fmt="svg")

Pipeline: options are cleaned against the type's schema -> the type's
``prepare`` function turns the user's data into tidy *frames* plus
*computed* statistics (error bars, significance brackets, fitted curves,
Kaplan–Meier steps ...) -> shared style is resolved -> the chosen engine only
has to draw.
"""
from __future__ import annotations

import copy
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..core.context import Context
from ..core.dataset import frame_to_r, jsonable, to_frame
from ..core.errors import AnalysisError
from ..core.registry import clean_options
from . import style

log = logging.getLogger("analysis_ui")

CATEGORIES = ["Column", "Grouped", "XY", "Distribution", "Survival",
              "Parts of whole", "Matrix", "Specialized"]
ENGINES = ("matplotlib", "ggplot2")
FORMATS = {"svg": "image/svg+xml", "png": "image/png", "pdf": "application/pdf",
           "tiff": "image/tiff", "jpg": "image/jpeg"}


@dataclass
class Prepared:
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    computed: dict = field(default_factory=dict)
    color_keys: list[str] = field(default_factory=list)
    x_label: str = ""
    y_label: str = ""
    legend_title: str = ""
    warnings: list[str] = field(default_factory=list)
    single_color: bool = False                 # all colour keys share one colour
    default_colors: dict = field(default_factory=dict)   # key -> hex, beats the palette


@dataclass
class GraphDef:
    id: str
    name: str
    category: str
    description: str
    render_type: str
    options: list[dict]
    prepare: Callable[[Context], Prepared]
    icon: str = ""

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "category": self.category,
                "description": self.description, "render_type": self.render_type,
                "icon": self.icon or self.render_type, "options": self.options}


GRAPHS: dict[str, GraphDef] = {}


def graph(*, id: str, name: str, category: str, description: str, options: list[dict],
          features: set[str], prepare: Callable[[Context], Prepared], icon: str = "") -> GraphDef:
    if id in GRAPHS:
        raise ValueError(f"Duplicate graph id: {id}")
    if category not in CATEGORIES:
        raise ValueError(f"Unknown graph category: {category}")
    full = list(options) + style.style_options(features)
    names = [o["name"] for o in full]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"Duplicate option names in graph {id}: {dup}")
    d = GraphDef(id=id, name=name, category=category, description=description,
                 render_type=id, options=full, prepare=prepare, icon=icon)
    GRAPHS[id] = d
    return d


def variant(*, of: str, id: str, name: str, description: str, defaults: dict,
            category: str | None = None, icon: str = "") -> GraphDef:
    """A gallery entry that is another type with different default options."""
    base = GRAPHS[of]
    opts = copy.deepcopy(base.options)
    known = {o["name"] for o in opts}
    for k in defaults:
        if k not in known:
            raise ValueError(f"variant {id}: unknown option {k}")
    for o in opts:
        if o["name"] in defaults:
            o["default"] = defaults[o["name"]]
    d = GraphDef(id=id, name=name, category=category or base.category, description=description,
                 render_type=base.render_type, options=opts, prepare=base.prepare, icon=icon or id)
    GRAPHS[id] = d
    return d


def _label(user: str | None, default: str) -> str:
    if user is None or user == "":
        return default or ""
    return user.strip()        # a single space hides the label


def build_payload(spec: dict, data: dict | None) -> tuple[dict, dict]:
    """Everything an engine needs (``payload``) plus ``meta`` for the UI."""
    d = GRAPHS.get(spec.get("type", ""))
    if d is None:
        raise AnalysisError(f"Unknown graph type: {spec.get('type')}")
    opts = clean_options(d.options, spec.get("options"))
    ctx = Context(to_frame(data), opts)
    ctx._labels = {o["name"]: o["label"] for o in d.options}
    prep = d.prepare(ctx)
    o = style.resolve(opts, prep.color_keys, single=prep.single_color,
                      default_colors=prep.default_colors)
    o["_labels"] = {
        "title": (o.get("title") or "").strip(),
        "subtitle": (o.get("subtitle") or "").strip(),
        "caption": (o.get("caption") or "").strip(),
        "x": _label(o.get("x_label"), prep.x_label),
        "y": _label(o.get("y_label"), prep.y_label),
        "legend": _label(o.get("legend_title"), prep.legend_title),
    }
    payload = {"type": d.render_type, "options": o, "frames": prep.frames,
               "computed": prep.computed}
    meta = {"groups": [{"name": n, "color": c}
                       for n, c in zip(o["_color_names"], o["_color_values"])],
            "warnings": prep.warnings, "type": d.id}
    return payload, meta


def render_graph(spec: dict, data: dict | None, fmt: str = "svg", dpi: int = 300,
                 editable_text: bool = False) -> tuple[bytes, dict]:
    """Render to ``fmt`` (svg | png | pdf | tiff | jpg). Raises AnalysisError
    with a user-facing message when the graph cannot be drawn."""
    if fmt not in FORMATS:
        raise AnalysisError(f"Unsupported image format: {fmt}")
    engine = spec.get("engine") or "matplotlib"
    if engine not in ENGINES:
        raise AnalysisError(f"Unknown graph engine: {engine}")
    payload, meta = build_payload(spec, data)
    payload.update(format=fmt, dpi=int(dpi), editable_text=bool(editable_text))
    meta["engine"] = engine
    if engine == "matplotlib":
        from ..core.plotstyle import MPL_LOCK
        from .mpl import render as mpl_render
        with MPL_LOCK:
            return mpl_render(payload), meta
    from ..core.rbridge import pool
    fd, path = tempfile.mkstemp(suffix="." + fmt, prefix="aui-graph-")
    os.close(fd)
    try:
        pool().call({
            "kind": "graph", "type": payload["type"], "options": jsonable(payload["options"]),
            "frames": {k: frame_to_r(v) for k, v in payload["frames"].items()},
            "computed": jsonable(payload["computed"]),
            "format": fmt, "dpi": int(dpi), "editable_text": bool(editable_text), "file": path})
        with open(path, "rb") as fh:
            return fh.read(), meta
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
