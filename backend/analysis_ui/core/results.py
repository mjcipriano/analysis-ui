"""Builders for the results an analysis returns.

A result is a flat list of *elements* which the frontend renders in order:

``heading``  a sub-title
``table``    columns (key / label / format) + rows (dicts) + footnotes
``text``     a paragraph; ``kind`` is "note", "summary", "warning"
``plot``     an SVG image (diagnostic plots)
``graph``    a button that creates an editable graph from the same data

Column formats: ``text``, ``num`` (smart significant digits), ``int``,
``p`` (p value, shown as <0.0001 when tiny), ``pct`` (fraction 0-1 shown as
percent), ``stars`` (significance summary computed from a p value).
"""
from __future__ import annotations

import io
from typing import Any, Sequence

from .dataset import jsonable


def stars(p: float | None) -> str:
    if p is None or p != p:
        return ""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fmt_p(p: float | None) -> str:
    if p is None or p != p:
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def p_text(p: float | None) -> str:
    """For sentences: ``"P = 0.0123"`` or ``"P < 0.0001"``."""
    if p is None or p != p:
        return "P not available"
    return "P < 0.0001" if p < 0.0001 else f"P = {p:.4f}"


class Table:
    def __init__(self, title: str, columns: Sequence[tuple | dict]):
        self.title = title
        self.columns = []
        for c in columns:
            if isinstance(c, dict):
                self.columns.append(c)
            else:
                key, label, *rest = c
                self.columns.append({"key": key, "label": label,
                                     "format": rest[0] if rest else "num"})
        self.rows: list[dict] = []
        self.footnotes: list[str] = []

    def add_row(self, _row: dict | None = None, **values: Any) -> "Table":
        row = dict(_row or {})
        row.update(values)
        self.rows.append(row)
        return self

    def note(self, text: str) -> "Table":
        self.footnotes.append(text)
        return self

    def to_dict(self) -> dict:
        return {"type": "table", "title": self.title, "columns": self.columns,
                "rows": self.rows, "footnotes": self.footnotes}


class Results:
    def __init__(self):
        self.elements: list[Any] = []
        self.warnings: list[str] = []

    def heading(self, text: str) -> None:
        self.elements.append({"type": "heading", "text": text})

    def table(self, title: str, columns: Sequence[tuple | dict]) -> Table:
        t = Table(title, columns)
        self.elements.append(t)
        return t

    def text(self, content: str, title: str | None = None, kind: str = "note") -> None:
        self.elements.append({"type": "text", "title": title, "content": content, "kind": kind})

    def summary(self, content: str) -> None:
        """A plain-language interpretation shown prominently."""
        self.text(content, kind="summary")

    def warn(self, content: str) -> None:
        self.text(content, kind="warning")

    def plot(self, title: str, fig, width: float | None = None, height: float | None = None) -> None:
        """Attach a matplotlib figure (it is closed afterwards)."""
        import matplotlib.pyplot as plt
        if width and height:
            fig.set_size_inches(width, height)
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        self.elements.append({"type": "plot", "title": title, "svg": buf.getvalue()})

    def graph(self, label: str, graph_type: str, options: dict | None = None) -> None:
        """Offer a one-click editable graph of the analysed data."""
        self.elements.append({"type": "graph", "label": label,
                              "graph": {"type": graph_type, "options": options or {}}})

    def to_dict(self) -> dict:
        els = [e.to_dict() if isinstance(e, Table) else e for e in self.elements]
        return jsonable({"elements": els, "warnings": self.warnings})


def normalize(result: Any) -> dict:
    """Make results coming back from R (or hand-built dicts) well-formed."""
    if isinstance(result, Results):
        return result.to_dict()
    result = dict(result or {})
    els = result.get("elements") or []
    if isinstance(els, dict):
        els = [els]
    out = []
    for e in els:
        if not isinstance(e, dict) or "type" not in e:
            continue
        e = dict(e)
        if e["type"] == "table":
            rows = e.get("rows") or []
            if isinstance(rows, dict):
                rows = [rows]
            e["rows"] = rows
            cols = e.get("columns") or []
            if isinstance(cols, dict):
                cols = [cols]
            e["columns"] = cols
            fn = e.get("footnotes") or []
            if isinstance(fn, str):
                fn = [fn]
            e["footnotes"] = fn
        out.append(e)
    warnings = result.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    return jsonable({"elements": out, "warnings": warnings})
