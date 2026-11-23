"""Column transforms (Prism "Transform / Normalize", jamovi "Compute")."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core.dataset import frame_to_wire, to_frame
from .core.errors import AnalysisError

# id -> (label, suffix, needs numeric)
TRANSFORMS = [
    {"id": "log10", "label": "Log10(Y)", "group": "Math"},
    {"id": "ln", "label": "Ln(Y)", "group": "Math"},
    {"id": "log2", "label": "Log2(Y)", "group": "Math"},
    {"id": "exp10", "label": "10^Y", "group": "Math"},
    {"id": "sqrt", "label": "Square root", "group": "Math"},
    {"id": "square", "label": "Y squared", "group": "Math"},
    {"id": "reciprocal", "label": "1 / Y", "group": "Math"},
    {"id": "abs", "label": "Absolute value", "group": "Math"},
    {"id": "zscore", "label": "Z-score (standardize)", "group": "Normalize"},
    {"id": "norm_minmax", "label": "Normalize: min = 0%, max = 100%", "group": "Normalize"},
    {"id": "norm_max", "label": "Percent of the column maximum", "group": "Normalize"},
    {"id": "norm_first", "label": "Percent of the first value", "group": "Normalize"},
    {"id": "norm_mean", "label": "Percent of the column mean", "group": "Normalize"},
    {"id": "norm_sum", "label": "Fraction of the column total", "group": "Normalize"},
    {"id": "baseline", "label": "Subtract the first value (baseline)", "group": "Normalize"},
    {"id": "rank", "label": "Rank", "group": "Other"},
    {"id": "cumsum", "label": "Cumulative sum", "group": "Other"},
    {"id": "diff", "label": "Difference from previous row", "group": "Other"},
    {"id": "formula", "label": "Custom formula…", "group": "Other"},
]
_SUFFIX = {"log10": "log10", "ln": "ln", "log2": "log2", "exp10": "pow10", "sqrt": "sqrt",
           "square": "sq", "reciprocal": "recip", "abs": "abs", "zscore": "z",
           "norm_minmax": "norm", "norm_max": "pct max", "norm_first": "pct first",
           "norm_mean": "pct mean", "norm_sum": "fraction", "baseline": "baseline",
           "rank": "rank", "cumsum": "cumsum", "diff": "diff"}


def _apply(op: str, s: pd.Series) -> pd.Series:
    v = s.astype(float)
    with np.errstate(all="ignore"):
        if op == "log10":
            return np.log10(v.where(v > 0))
        if op == "ln":
            return np.log(v.where(v > 0))
        if op == "log2":
            return np.log2(v.where(v > 0))
        if op == "exp10":
            return np.power(10.0, v)
        if op == "sqrt":
            return np.sqrt(v.where(v >= 0))
        if op == "square":
            return v ** 2
        if op == "reciprocal":
            return 1 / v.where(v != 0)
        if op == "abs":
            return v.abs()
        if op == "zscore":
            return (v - v.mean()) / v.std(ddof=1)
        if op == "norm_minmax":
            return (v - v.min()) / (v.max() - v.min()) * 100
        if op == "norm_max":
            return v / v.max() * 100
        first = v.dropna().iloc[0] if v.notna().any() else np.nan
        if op == "norm_first":
            return v / first * 100
        if op == "norm_mean":
            return v / v.mean() * 100
        if op == "norm_sum":
            return v / v.sum()
        if op == "baseline":
            return v - first
        if op == "rank":
            return v.rank()
        if op == "cumsum":
            return v.cumsum()
        if op == "diff":
            return v.diff()
    raise AnalysisError(f"Unknown transform: {op}")


def transform(data: dict, op: str, columns: list[str], formula: str = "", name: str = "") -> dict:
    """Returns ``{"columns": [new wire columns]}``."""
    df = to_frame(data)
    if op == "formula":
        if not formula.strip():
            raise AnalysisError("Type a formula, for example:  log10(`Dose`) * 2  or  A / B")
        num = pd.DataFrame({c: (pd.to_numeric(df[c].astype(object), errors="coerce")
                                if isinstance(df[c].dtype, pd.CategoricalDtype) else df[c]) for c in df.columns})
        try:
            res = num.eval(formula, engine="python")
        except Exception as exc:
            raise AnalysisError(
                f"The formula could not be evaluated: {exc}. Put column names that contain spaces "
                f"or symbols in backticks, e.g. `Drug A` / `Control`.") from None
        if np.isscalar(res):
            res = pd.Series([res] * len(df))
        out = pd.DataFrame({name.strip() or "Computed": pd.to_numeric(pd.Series(res), errors="coerce").replace([np.inf, -np.inf], np.nan)})
        return frame_to_wire(out)
    if not columns:
        raise AnalysisError("Choose at least one column to transform.")
    out = {}
    for c in columns:
        if c not in df.columns:
            raise AnalysisError(f"Column “{c}” was not found.")
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            s = pd.to_numeric(s.astype(object), errors="coerce")
        res = _apply(op, s).replace([np.inf, -np.inf], np.nan)
        out[f"{c} ({_SUFFIX[op]})"] = res.to_numpy()
    return frame_to_wire(pd.DataFrame(out))


def reshape(data: dict, op: str, columns: list[str], id_column: str | None = None,
            value_column: str | None = None) -> dict:
    """``stack``: wide -> long (Group, Value [, id]); ``unstack``: long -> wide."""
    df = to_frame(data)
    if op == "stack":
        if len(columns) < 1:
            raise AnalysisError("Choose the columns to stack.")
        ids = [id_column] if id_column else []
        long = df[ids + columns].melt(id_vars=ids, var_name="Group", value_name="Value")
        long = long[long["Value"].notna()].reset_index(drop=True)
        long["Group"] = pd.Categorical(long["Group"], categories=columns)
        return frame_to_wire(long)
    if op == "unstack":
        if not value_column or not id_column:
            raise AnalysisError("Choose the values column and the grouping column.")
        g = df[id_column]
        levels = list(g.cat.categories) if isinstance(g.dtype, pd.CategoricalDtype) else sorted(g.dropna().unique())
        parts = {str(l): df.loc[g == l, value_column].dropna().reset_index(drop=True) for l in levels}
        return frame_to_wire(pd.DataFrame(parts))
    raise AnalysisError(f"Unknown reshape: {op}")
