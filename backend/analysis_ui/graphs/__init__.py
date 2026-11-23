"""Editable, publication-quality graphs.

``types`` registers every graph type (schema + data preparation);
``mpl`` draws with matplotlib/seaborn and ``r/graphs`` with ggplot2.
"""
from .registry import GRAPHS, render_graph, build_payload  # noqa: F401
from . import types  # noqa: F401  (registers the graph types)
