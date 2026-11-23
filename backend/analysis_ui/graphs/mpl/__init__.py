"""matplotlib drawing engine for editable graphs.

    render(payload) -> bytes

``payload`` is produced by :func:`analysis_ui.graphs.registry.build_payload`
(the ``_prep_*`` docstrings in ``graphs/types.py`` are the per-type contract).
The engine only draws: every number on the canvas comes from the payload.

Modules
-------
base         ``Plot``: theme, axes, scales, legend, manual layout, significance
             brackets, beeswarm, label placement, saving
categorical  column_* / box / violin / raincloud / before_after / grouped_* /
             interaction_line (shared "cell" model)
xy           xy (scatter, line, dose–response, bubble)
distribution histogram, density (+ ridgeline), qq, ecdf
survival     Kaplan–Meier curves with risk table
parts        pie / donut, count_bar
matrix       heatmap (+ correlation matrix)
special      bland_altman, roc, volcano, forest, waterfall
"""
from __future__ import annotations

from matplotlib import rc_context

from ...core.errors import AnalysisError
from . import categorical, distribution, matrix, parts, special, survival, xy
from .base import Plot, build_rc

DRAWERS: dict = {}
for _mod in (categorical, xy, distribution, survival, parts, matrix, special):
    DRAWERS.update(_mod.DRAWERS)

FORMATS = ("svg", "png", "pdf", "tiff", "jpg")


def render(payload: dict) -> bytes:
    """Draw one graph. The caller holds the global matplotlib lock."""
    draw = DRAWERS.get(payload.get("type"))
    if draw is None:
        raise AnalysisError(f"The matplotlib engine cannot draw “{payload.get('type')}”.")
    if str(payload.get("format") or "svg").lower() not in FORMATS:
        raise AnalysisError(f"Unsupported image format: {payload.get('format')}")
    o = payload["options"]
    with rc_context(rc=build_rc(o["_fonts"], o["_theme"], bool(payload.get("editable_text")))):
        p = Plot(payload).begin()
        if p.computed.get("facets"):
            p.draw_facets(draw)
        else:
            draw(p)
        p.finish()
        return p.save()
