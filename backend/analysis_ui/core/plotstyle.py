"""House style for the small diagnostic plots embedded in analysis results.

(Editable, publication-quality graphs live in ``analysis_ui.graphs``.)
"""
from __future__ import annotations

import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#3B6FE0", "#E4572E", "#17A589", "#F2A541", "#8E5BD9", "#D6456F", "#4FB0E5", "#7A8B99"]

_RC = {
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.prop_cycle": matplotlib.cycler(color=COLORS),
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "svg.fonttype": "path",
    "figure.dpi": 100,
}


# The house style is the process-wide default; the graph engine overrides
# everything it needs inside its own rc_context. All matplotlib work must hold
# MPL_LOCK because pyplot and rcParams are global state.
from matplotlib import font_manager as _fm  # noqa: E402

_have = {f.name for f in _fm.fontManager.ttflist}
_RC["font.family"] = [f for f in _RC["font.family"] if f in _have] or ["DejaVu Sans"]   # avoids findfont warnings
matplotlib.rcParams.update(_RC)
MPL_LOCK = threading.RLock()


def new_figure(width: float = 5.5, height: float = 3.8, nrows: int = 1, ncols: int = 1, **kw):
    """``fig, ax = new_figure()`` — a styled figure for ``Results.plot``.

    With ``nrows``/``ncols`` > 1 ``ax`` is an array, as in ``plt.subplots``.
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=(width, height),
                           constrained_layout=True, **kw)
    return fig, ax
