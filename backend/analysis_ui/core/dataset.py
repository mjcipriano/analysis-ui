"""Conversion between the JSON wire format and pandas data frames.

Wire format (what the browser sends and receives)::

    {"columns": [
        {"name": "Dose", "type": "continuous", "values": [1, 2.5, null]},
        {"name": "Sex",  "type": "nominal",    "values": ["F", "M", null],
         "levels": ["F", "M"]}          # optional explicit level order
    ]}

Column types follow jamovi: ``continuous``, ``ordinal`` and ``nominal``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .errors import AnalysisError

TYPES = ("continuous", "nominal", "ordinal")


def _clean_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer():
            return str(int(v))
        return repr(v)
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    s = str(v).strip()
    return s if s != "" else None


def _to_number(v: Any) -> float:
    if v is None or isinstance(v, bool):
        return math.nan if v is None else float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return math.nan
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return math.nan


def to_frame(data: dict | None) -> pd.DataFrame:
    """Wire format -> DataFrame (float columns and pandas Categoricals)."""
    if not data or not data.get("columns"):
        return pd.DataFrame()
    cols = data["columns"]
    n = max((len(c.get("values") or []) for c in cols), default=0)
    out: dict[str, Any] = {}
    for c in cols:
        name = str(c["name"])
        if name in out:
            raise AnalysisError(f"Duplicate column name: {name!r}")
        values = list(c.get("values") or [])
        values += [None] * (n - len(values))
        ctype = c.get("type", "continuous")
        if ctype == "continuous":
            out[name] = np.array([_to_number(v) for v in values], dtype=float)
        else:
            labels = [_clean_label(v) for v in values]
            levels = c.get("levels")
            if levels:
                levels = [x for x in (_clean_label(l) for l in levels) if x is not None]
                seen = set(levels)
                for lab in labels:  # tolerate values missing from the level list
                    if lab is not None and lab not in seen:
                        levels.append(lab)
                        seen.add(lab)
            else:
                levels = list(dict.fromkeys(l for l in labels if l is not None))
                # numeric-looking labels sort numerically, like jamovi
                try:
                    levels = sorted(levels, key=float)
                except ValueError:
                    pass
            out[name] = pd.Categorical(labels, categories=levels,
                                       ordered=(ctype == "ordinal"))
    return pd.DataFrame(out)


def column_type(series: pd.Series) -> str:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "ordinal" if series.cat.ordered else "nominal"
    return "continuous"


def jsonable(x: Any) -> Any:
    """Recursively convert numpy / pandas objects into JSON-safe values."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, (int,)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.Series):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, pd.DataFrame):
        return frame_to_wire(x)
    if x is pd.NA or x is pd.NaT:
        return None
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    return str(x)


def frame_to_wire(df: pd.DataFrame, infer: bool = False) -> dict:
    """DataFrame -> wire format. With ``infer`` column types are guessed
    (used by file import)."""
    cols = []
    for name in df.columns:
        s = df[name]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cols.append({
                "name": str(name),
                "type": "ordinal" if s.cat.ordered else "nominal",
                "values": [None if pd.isna(v) else str(v) for v in s.tolist()],
                "levels": [str(l) for l in s.cat.categories],
            })
            continue
        if pd.api.types.is_bool_dtype(s):
            cols.append({"name": str(name), "type": "nominal",
                         "values": [None if pd.isna(v) else ("TRUE" if v else "FALSE") for v in s.tolist()]})
            continue
        if pd.api.types.is_numeric_dtype(s):
            vals = [jsonable(float(v)) if not pd.isna(v) else None for v in s.tolist()]
            # keep integers as integers for a tidy grid
            if pd.api.types.is_integer_dtype(s) or all(v is None or float(v).is_integer() for v in vals):
                vals = [None if v is None else int(v) for v in vals]
            cols.append({"name": str(name), "type": "continuous", "values": vals})
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            cols.append({"name": str(name), "type": "nominal",
                         "values": [None if pd.isna(v) else str(v) for v in s.tolist()]})
            continue
        # object column: numeric if (almost) everything parses as a number
        raw = s.tolist()
        labels = [_clean_label(v) for v in raw]
        present = [l for l in labels if l is not None]
        if infer and present:
            nums = [_to_number(l) for l in present]
            ok = sum(1 for v in nums if not math.isnan(v))
            if ok == len(present):
                vals = [None if l is None else _to_number(l) for l in labels]
                vals = [None if v is None else (int(v) if float(v).is_integer() else v) for v in vals]
                cols.append({"name": str(name), "type": "continuous", "values": vals})
                continue
        cols.append({"name": str(name), "type": "nominal", "values": labels})
    return {"columns": cols}


def frame_to_r(df: pd.DataFrame) -> dict:
    """DataFrame -> compact column payload for the R worker."""
    cols = []
    for name in df.columns:
        s = df[name]
        if isinstance(s.dtype, pd.CategoricalDtype):
            cols.append({"name": str(name), "kind": "factor",
                         "ordered": bool(s.cat.ordered),
                         "levels": [str(l) for l in s.cat.categories],
                         "values": [None if pd.isna(v) else str(v) for v in s.tolist()]})
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            cols.append({"name": str(name), "kind": "numeric",
                         "values": [None if pd.isna(v) or not math.isfinite(float(v)) else float(v)
                                    for v in s.tolist()]})
        else:
            cols.append({"name": str(name), "kind": "character",
                         "values": [None if pd.isna(v) else str(v) for v in s.tolist()]})
    return {"nrow": int(len(df)), "columns": cols}
