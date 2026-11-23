"""Analysis registry.

Python analysis::

    @analysis(id="ttest", name="t test", category="Compare two groups",
              options=[*opt.group_layout(), ...])
    def run(ctx: Context) -> Results: ...

R analysis (schema declared in Python, computation in ``r/analyses/<script>``)::

    r_analysis(id="anova_oneway", name="One-way ANOVA", category="...",
               script="anova_oneway.R", options=[...], prepare=prepare_long,
               requires=["emmeans"])

``prepare(ctx) -> (DataFrame, options)`` lets Python normalise the data
layout before R sees it; see :func:`prepare_long` and friends.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import options as opt
from .context import Context
from .dataset import frame_to_r, jsonable, to_frame
from .errors import AnalysisError
from .results import Results, normalize

log = logging.getLogger("analysis_ui")

# Order in which categories appear in the Analyze menu.
CATEGORIES = [
    "Describe & explore",
    "Compare two groups",
    "Compare three or more groups",
    "Factorial & repeated measures",
    "Correlation & regression",
    "Curve fitting",
    "Categorical data",
    "Survival",
    "Diagnostics & agreement",
    "Multivariate",
    "Planning",
]


@dataclass
class AnalysisDef:
    id: str
    name: str
    category: str
    engine: str                      # "python" | "r"
    options: list[dict]
    description: str = ""
    func: Callable[[Context], Results] | None = None
    script: str | None = None
    prepare: Callable[[Context], tuple[pd.DataFrame, dict]] | None = None
    requires: list[str] = field(default_factory=list)
    keywords: str = ""

    def public(self) -> dict:
        from .rbridge import r_has, r_info
        available, reason = True, None
        if self.engine == "r":
            if not r_info()["available"]:
                available, reason = False, "R is not installed"
            elif not r_has(*self.requires):
                missing = [p for p in self.requires if not r_has(p)]
                available, reason = False, "Missing R package(s): " + ", ".join(missing)
        return {"id": self.id, "name": self.name, "category": self.category,
                "engine": self.engine, "description": self.description,
                "keywords": self.keywords, "options": self.options,
                "available": available, "unavailable_reason": reason}


ANALYSES: dict[str, AnalysisDef] = {}


def _register(d: AnalysisDef):
    if d.id in ANALYSES:
        raise ValueError(f"Duplicate analysis id: {d.id}")
    names = [o["name"] for o in d.options]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate option names in analysis {d.id}")
    if d.category not in CATEGORIES:
        raise ValueError(f"Unknown category {d.category!r} for analysis {d.id}")
    ANALYSES[d.id] = d


def analysis(*, id: str, name: str, category: str, options: list[dict],
             description: str = "", keywords: str = ""):
    def deco(func):
        _register(AnalysisDef(id=id, name=name, category=category, engine="python",
                              options=options, description=description, func=func,
                              keywords=keywords))
        return func
    return deco


def r_analysis(*, id: str, name: str, category: str, script: str, options: list[dict],
               description: str = "", prepare=None, requires: list[str] | None = None,
               keywords: str = ""):
    _register(AnalysisDef(id=id, name=name, category=category, engine="r",
                          options=options, description=description, script=script,
                          prepare=prepare, requires=list(requires or []),
                          keywords=keywords))


# ---- ready-made prepare functions for R analyses --------------------------

def prepare_long(ctx: Context):
    """Data frame with ``value`` and ``group`` (factor) columns."""
    return ctx.long(), ctx.options


def prepare_long_subject(ctx: Context):
    """``value``, ``group`` and ``subject`` columns (complete subjects only)."""
    wide = ctx.paired(complete=True)
    if len(wide) < 2:
        raise AnalysisError("At least two subjects with values in every condition are needed.")
    long = wide.reset_index().melt(id_vars="subject", var_name="group", value_name="value")
    long["group"] = pd.Categorical(long["group"], categories=list(wide.columns))
    long["subject"] = pd.Categorical(long["subject"].astype(str))
    return long, ctx.options


# ---- running --------------------------------------------------------------

def clean_options(schema: list[dict], given: dict | None) -> dict:
    """Fill defaults, drop unknown keys and coerce simple types."""
    given = given or {}
    out: dict[str, Any] = {}
    for o in schema:
        name, kind = o["name"], o["type"]
        v = given.get(name, o["default"])
        if kind in ("number", "integer"):
            if v in ("", None):
                v = None if o.get("nullable") else o["default"]
            else:
                try:
                    v = float(v)
                    if kind == "integer":
                        v = int(round(v))
                except (TypeError, ValueError):
                    v = o["default"]
                if v is not None and o.get("min") is not None and v < o["min"]:
                    v = o["min"]
                if v is not None and o.get("max") is not None and v > o["max"]:
                    v = o["max"]
        elif kind == "bool":
            v = bool(v)
        elif kind == "choice":
            if v not in [c["value"] for c in o["choices"]]:
                v = o["default"]
        elif kind == "variables":
            v = [x for x in (v or []) if x]
        elif kind in ("text", "level", "color"):
            v = "" if v is None else str(v)
        elif kind == "variable":
            v = v or None
        out[name] = v
    # preserve private keys (used internally, e.g. "_brackets")
    for k, v in given.items():
        if k.startswith("_"):
            out[k] = v
    return out


def discover():
    """Import every module in ``analysis_ui.analyses`` (registers analyses)."""
    from .. import analyses
    for mod in pkgutil.iter_modules(analyses.__path__):
        importlib.import_module(f"{analyses.__name__}.{mod.name}")


def run_analysis(analysis_id: str, data: dict | None, options: dict | None) -> dict:
    """Run one analysis. Always returns a dict; never raises for user errors."""
    d = ANALYSES.get(analysis_id)
    if d is None:
        return {"status": "error", "message": f"Unknown analysis: {analysis_id}"}
    try:
        opts = clean_options(d.options, options)
        ctx = Context(to_frame(data), opts)
        ctx._labels = {o["name"]: o["label"] for o in d.options}
        if d.engine == "python":
            assert d.func is not None
            from .plotstyle import MPL_LOCK
            with MPL_LOCK:
                res = normalize(d.func(ctx))
        else:
            from .rbridge import pool
            if d.prepare:
                df, opts = d.prepare(ctx)
            else:
                df = ctx.df
            res = normalize(pool().call({
                "kind": "analysis", "script": d.script,
                "data": frame_to_r(df), "options": jsonable(opts)}))
        return {"status": "ok", "results": res}
    except AnalysisError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:  # unexpected: keep the UI alive, log the trace
        log.error("Analysis %s failed:\n%s", analysis_id, traceback.format_exc())
        return {"status": "error",
                "message": f"{type(exc).__name__}: {exc}"}
