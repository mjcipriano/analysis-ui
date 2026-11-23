"""The object handed to every analysis / graph ``prepare`` function."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import AnalysisError


class Context:
    def __init__(self, df: pd.DataFrame, options: dict):
        self.df = df
        self.options = options

    # ------------------------------------------------------------ options
    def opt(self, name: str, default: Any = None) -> Any:
        v = self.options.get(name, default)
        return default if v is None else v

    def has(self, name: str) -> bool:
        v = self.options.get(name)
        return v not in (None, "", [])

    # ------------------------------------------------------------ columns
    def column(self, name: str) -> pd.Series:
        if name not in self.df.columns:
            raise AnalysisError(f"Column “{name}” was not found in the data.")
        return self.df[name]

    def var(self, option: str, required: bool = True) -> str | None:
        """Column name stored in a ``variable`` option."""
        name = self.options.get(option)
        if not name:
            if required:
                raise AnalysisError(f"Please choose a variable for “{self._label(option)}”.")
            return None
        self.column(name)
        return name

    def vars(self, option: str, min: int = 1) -> list[str]:
        names = [n for n in (self.options.get(option) or []) if n]
        if len(names) < min:
            what = "variable" if min == 1 else "variables"
            raise AnalysisError(
                f"Please choose at least {min} {what} for “{self._label(option)}”.")
        for n in names:
            self.column(n)
        return names

    def numeric(self, name: str) -> pd.Series:
        """A column as floats (categorical labels are parsed as numbers)."""
        s = self.column(name)
        if isinstance(s.dtype, pd.CategoricalDtype):
            return pd.to_numeric(s.astype(object), errors="coerce").astype(float)
        return s.astype(float)

    def factor(self, name: str) -> pd.Series:
        """A column as an (unordered-safe) Categorical with unused levels dropped."""
        s = self.column(name)
        if not isinstance(s.dtype, pd.CategoricalDtype):
            vals = s.tolist()
            labels = [None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v))
                      for v in vals]
            levels = sorted({l for l in labels if l is not None}, key=float)
            s = pd.Series(pd.Categorical(labels, categories=levels), index=s.index, name=name)
        return s.cat.remove_unused_categories()

    _labels: dict[str, str] = {}

    def _label(self, option: str) -> str:
        return self._labels.get(option, option)

    # ------------------------------------------------------- group layouts
    def long(self, dropna: bool = True, with_subject: bool = False) -> pd.DataFrame:
        """Groups as a tidy frame with columns ``value``, ``group``
        (Categorical, level order = display order) and, when
        ``with_subject``, ``subject``."""
        layout = self.opt("layout", "wide")
        if layout == "wide":
            cols = self.vars("columns", min=1)
            parts = []
            for c in cols:
                part = pd.DataFrame({"value": self.numeric(c).to_numpy(), "group": c})
                if with_subject:
                    part["subject"] = np.arange(len(part)).astype(str)
                parts.append(part)
            out = pd.concat(parts, ignore_index=True)
            out["group"] = pd.Categorical(out["group"], categories=cols)
        else:
            dep = self.var("dep")
            grp = self.var("group")
            out = pd.DataFrame({"value": self.numeric(dep).to_numpy(),
                                "group": self.factor(grp).to_numpy()})
            out["group"] = pd.Categorical(out["group"],
                                          categories=list(self.factor(grp).cat.categories))
            if with_subject:
                subj = self.var("subject", required=False)
                if subj:
                    out["subject"] = self.column(subj).astype(object).map(
                        lambda v: None if pd.isna(v) else str(v)).to_numpy()
                else:
                    out["subject"] = out.groupby("group", observed=True).cumcount().astype(str)
                    out.loc[out["group"].isna(), "subject"] = None
        if dropna:
            keep = out["value"].notna() & out["group"].notna()
            if with_subject:
                keep &= out["subject"].notna()
            out = out[keep].reset_index(drop=True)
            out["group"] = out["group"].cat.remove_unused_categories()
        return out

    def groups(self, min_groups: int = 1, min_n: int = 1) -> dict[str, np.ndarray]:
        """Ordered mapping ``group label -> 1-D float array`` (NaNs removed)."""
        long = self.long()
        out: dict[str, np.ndarray] = {}
        for g in long["group"].cat.categories:
            vals = long.loc[long["group"] == g, "value"].to_numpy(dtype=float)
            if len(vals):
                out[str(g)] = vals
        if len(out) < min_groups:
            raise AnalysisError(
                f"This needs at least {min_groups} groups with data, but found {len(out)}.")
        for g, v in out.items():
            if len(v) < min_n:
                raise AnalysisError(
                    f"Group “{g}” has only {len(v)} value(s); at least {min_n} are needed.")
        return out

    def paired(self, complete: bool = True) -> pd.DataFrame:
        """Matched data: one row per subject, one column per condition."""
        long = self.long(dropna=True, with_subject=True)
        if long.duplicated(["subject", "group"]).any():
            raise AnalysisError(
                "Some subjects have more than one value in the same group, so "
                "the rows cannot be paired. Choose a Subject / ID column that "
                "identifies each subject exactly once per group.")
        wide = long.pivot(index="subject", columns="group", values="value")
        wide = wide[[c for c in long["group"].cat.categories if c in wide.columns]]
        wide.columns = [str(c) for c in wide.columns]
        if complete:
            wide = wide.dropna()
        return wide

    # --------------------------------------------------------------- misc
    def frame(self, names: list[str], dropna: bool = True) -> pd.DataFrame:
        """Sub-frame of the given columns with complete cases."""
        for n in names:
            self.column(n)
        out = self.df[list(dict.fromkeys(names))].copy()
        if dropna:
            out = out.dropna().reset_index(drop=True)
            for c in out.columns:
                if isinstance(out[c].dtype, pd.CategoricalDtype):
                    out[c] = out[c].cat.remove_unused_categories()
        return out
