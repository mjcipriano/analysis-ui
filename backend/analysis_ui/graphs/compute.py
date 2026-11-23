"""Statistics computed once in Python for graphs, so that the matplotlib and
ggplot2 engines draw exactly the same numbers."""
from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from ..core.curvefit import MODELS, fit_model
from ..core.errors import AnalysisError
from ..core.results import stars


# ---------------------------------------------------------------- summaries
def summarize(values, center: str = "mean", error: str = "sd") -> dict:
    """``{"n", "center", "lo", "hi"}`` — lo/hi are the error-bar ends
    (None when ``error == "none"`` or not computable)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "center": None, "lo": None, "hi": None}
    if center == "median":
        c = float(np.median(v))
    elif center == "geomean":
        pos = v[v > 0]
        c = float(np.exp(np.mean(np.log(pos)))) if len(pos) else float("nan")
    else:
        c = float(np.mean(v))
    lo = hi = None
    if error == "range":
        lo, hi = float(np.min(v)), float(np.max(v))
    elif error == "iqr":
        lo, hi = (float(q) for q in np.percentile(v, [25, 75]))
    elif error in ("sd", "sem", "ci95") and n > 1:
        if center == "geomean":
            pos = v[v > 0]
            if len(pos) > 1:
                ls = float(np.std(np.log(pos), ddof=1))
                if error == "sem":
                    ls /= math.sqrt(len(pos))
                elif error == "ci95":
                    ls = ls / math.sqrt(len(pos)) * stats.t.ppf(0.975, len(pos) - 1)
                lo, hi = c / math.exp(ls), c * math.exp(ls)
        elif center == "median" and error == "ci95":
            # distribution-free CI of the median from order statistics
            k = int(stats.binom.ppf(0.025, n, 0.5))
            s = np.sort(v)
            lo, hi = float(s[max(k - 1, 0)]), float(s[min(n - k, n - 1)])
        else:
            sd = float(np.std(v, ddof=1))
            half = sd
            if error == "sem":
                half = sd / math.sqrt(n)
            elif error == "ci95":
                half = sd / math.sqrt(n) * stats.t.ppf(0.975, n - 1)
            lo, hi = c - half, c + half
    return {"n": n, "center": c, "lo": lo, "hi": hi}


ERROR_LABEL = {"sd": "SD", "sem": "SEM", "ci95": "95% CI", "iqr": "IQR", "range": "range", "none": ""}
CENTER_LABEL = {"mean": "Mean", "median": "Median", "geomean": "Geometric mean"}


# ---------------------------------------------------------------- brackets
def _adjust(pvals: list[float], method: str) -> list[float]:
    if method in (None, "", "none") or len(pvals) <= 1:
        return list(pvals)
    from statsmodels.stats.multitest import multipletests
    name = {"holm": "holm", "bonferroni": "bonferroni", "sidak": "sidak", "fdr_bh": "fdr_bh"}.get(method, "holm")
    return list(multipletests(pvals, method=name)[1])


def _test(a: np.ndarray, b: np.ndarray, test: str) -> float:
    try:
        if test == "t":
            return float(stats.ttest_ind(a, b, equal_var=True).pvalue)
        if test == "welch":
            return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        if test == "mannwhitney":
            return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if test == "paired_t":
            return float(stats.ttest_rel(a, b).pvalue)
        if test == "wilcoxon":
            return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")
    raise AnalysisError(f"Unknown test: {test}")


def p_label(p: float, style: str) -> str:
    if p is None or p != p:
        return "n/a"
    if style == "p":
        return "p < 0.0001" if p < 0.0001 else f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.4f}"
    return stars(p)


def _parse_custom(text: str, names: list[str]) -> list[tuple[str, str]]:
    pairs = []
    lookup = {n.lower(): n for n in names}
    for chunk in re.split(r"[;\n]+", text or ""):
        parts = re.split(r"\s+vs\.?\s+|\s*[-–—]\s*|\s*,\s*", chunk.strip(), maxsplit=1)
        if len(parts) == 2:
            a, b = (lookup.get(p.strip().lower()) for p in parts)
            if a and b and a != b:
                pairs.append((a, b))
    return pairs


def pairwise_brackets(groups: dict[str, np.ndarray], test: str, comparisons: str = "all",
                      control: str = "", custom: str = "", adjust: str = "holm",
                      label: str = "stars", hide_ns: bool = False,
                      paired: pd.DataFrame | None = None) -> list[dict]:
    """Significance brackets ``[{"g1", "g2", "p", "label"}]`` between groups."""
    names = list(groups)
    if test in (None, "", "none") or len(names) < 2:
        return []
    if comparisons == "control":
        ctrl = control if control in names else names[0]
        pairs = [(ctrl, n) for n in names if n != ctrl]
    elif comparisons == "adjacent":
        pairs = list(zip(names[:-1], names[1:]))
    elif comparisons == "custom":
        pairs = _parse_custom(custom, names)
    else:
        pairs = list(itertools.combinations(names, 2))
    if len(pairs) > 45:
        raise AnalysisError("Too many comparisons to draw (more than 45). "
                            "Compare against a control or list custom pairs instead.")
    ps = []
    for a, b in pairs:
        if test in ("paired_t", "wilcoxon"):
            if paired is None:
                raise AnalysisError("Paired tests need paired data.")
            sub = paired[[a, b]].dropna()
            ps.append(_test(sub[a].to_numpy(), sub[b].to_numpy(), test) if len(sub) > 1 else float("nan"))
        else:
            ps.append(_test(groups[a], groups[b], test) if min(len(groups[a]), len(groups[b])) > 1 else float("nan"))
    finite = [i for i, p in enumerate(ps) if p == p]
    adj = _adjust([ps[i] for i in finite], adjust)
    for i, p in zip(finite, adj):
        ps[i] = float(p)
    out = []
    for (a, b), p in zip(pairs, ps):
        if hide_ns and not (p == p and p < 0.05):
            continue
        out.append({"g1": a, "g2": b, "p": None if p != p else p, "label": p_label(p, label)})
    return out


# -------------------------------------------------------------------- fits
FIT_CHOICES = [("none", "None")] + [(m.id, m.name) for m in MODELS.values()] + [
    ("loess", "LOWESS smoother"), ("spline", "Smoothing spline"), ("connect_means", "Point-to-point through means")]


def fit_series(x, y, model: str, log_x: bool = False, span: float = 0.75,
               n_grid: int = 200, band: bool = True) -> dict | None:
    """Fitted curve for one series: ``{"x", "y", "lo", "hi", "label", "r2", "params"}``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if log_x:
        ok &= x > 0
    x, y = x[ok], y[ok]
    if model in (None, "", "none") or len(x) < 2:
        return None
    lo_x, hi_x = float(np.min(x)), float(np.max(x))
    if lo_x == hi_x:
        return None
    grid = np.geomspace(lo_x, hi_x, n_grid) if (log_x and lo_x > 0) else np.linspace(lo_x, hi_x, n_grid)
    if model == "loess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xs = np.log10(x) if log_x else x
        sm = lowess(y, xs, frac=min(max(span, 0.15), 1.0), return_sorted=True)
        ux, idx = np.unique(sm[:, 0], return_index=True)
        gx = np.log10(grid) if log_x else grid
        return {"x": grid, "y": np.interp(gx, ux, sm[idx, 1]), "lo": None, "hi": None,
                "label": "LOWESS", "r2": None, "params": {}}
    if model == "spline":
        from scipy.interpolate import UnivariateSpline
        xs = np.log10(x) if log_x else x
        d = pd.DataFrame({"x": xs, "y": y}).groupby("x", as_index=False).mean().sort_values("x")
        if len(d) < 4:
            return None
        spl = UnivariateSpline(d["x"], d["y"], k=3, s=len(d) * float(np.var(d["y"])) * 0.05)
        gx = np.log10(grid) if log_x else grid
        return {"x": grid, "y": spl(gx), "lo": None, "hi": None, "label": "Spline", "r2": None, "params": {}}
    if model == "connect_means":
        d = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean().sort_values("x")
        return {"x": d["x"].to_numpy(), "y": d["y"].to_numpy(), "lo": None, "hi": None,
                "label": "Means", "r2": None, "params": {}}
    fit = fit_model(model, x, y)
    with np.errstate(all="ignore"):
        yg = fit.predict(grid)
        lo, hi = fit.band(grid) if band else (None, None)
    if model == "linear":
        label = f"Y = {fit.params['Slope']:.4g}·X {'+' if fit.params['Intercept'] >= 0 else '−'} {abs(fit.params['Intercept']):.4g}"
    elif "EC50" in fit.params:
        label = f"EC50 = {fit.params['EC50']:.4g}"
    elif "LogEC50" in fit.params:
        label = f"LogEC50 = {fit.params['LogEC50']:.4g}"
    elif "Km" in fit.params:
        label = f"Vmax = {fit.params['Vmax']:.4g}, Km = {fit.params['Km']:.4g}"
    elif "K" in fit.params and model.startswith("exp"):
        label = f"K = {fit.params['K']:.4g}"
    else:
        label = fit.model.name.split(":")[0]
    if fit.r2 == fit.r2:
        label += f" (R² = {fit.r2:.3f})"
    return {"x": grid, "y": yg, "lo": lo, "hi": hi, "label": label, "r2": fit.r2,
            "params": fit.params}


# ---------------------------------------------------------------- survival
def km_curve(time, event) -> dict:
    """Kaplan–Meier estimate with log-log 95 % CI. Arrays start at t = 0."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    n = len(t)
    times, surv, lo, hi, at_risk_list = [0.0], [1.0], [1.0], [1.0], [n]
    s, var_sum = 1.0, 0.0
    cens_t, cens_s = [], []
    for ut in np.unique(t):
        at_risk = int(np.sum(t >= ut))
        d = int(np.sum((t == ut) & e))
        c = int(np.sum((t == ut) & ~e))
        if d > 0:
            s *= 1 - d / at_risk
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
            times.append(float(ut))
            surv.append(s)
            at_risk_list.append(at_risk)
            if 0 < s < 1 and var_sum > 0:
                se = math.sqrt(var_sum) / abs(math.log(s))
                lo.append(s ** math.exp(1.96 * se))
                hi.append(s ** math.exp(-1.96 * se))
            else:
                lo.append(s)
                hi.append(s)
        if c > 0:
            cens_t.append(float(ut))
            cens_s.append(s)
    last = float(np.max(t)) if n else 0.0
    if times[-1] < last:      # extend the step to the last follow-up time
        times.append(last); surv.append(s); lo.append(lo[-1]); hi.append(hi[-1])
    below = [tt for tt, ss in zip(times, surv) if ss <= 0.5]
    return {"time": times, "surv": surv, "lo": lo, "hi": hi,
            "censor_time": cens_t, "censor_surv": cens_s,
            "n": n, "events": int(e.sum()), "median": below[0] if below else None}


def at_risk(time, ticks) -> list[int]:
    t = np.asarray(time, dtype=float)
    return [int(np.sum(t >= k)) for k in ticks]


def logrank_p(time, event, group) -> float | None:
    from lifelines.statistics import multivariate_logrank_test
    try:
        return float(multivariate_logrank_test(time, group, event).p_value)
    except Exception:
        return None


# --------------------------------------------------------------------- ROC
def roc_curve(y_true, score) -> dict:
    """ROC for a binary outcome (True = positive). Higher score = positive;
    the direction is flipped automatically when AUC < 0.5."""
    from sklearn.metrics import roc_auc_score, roc_curve as _roc
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    auc = float(roc_auc_score(y, s))
    flipped = auc < 0.5
    if flipped:
        s = -s
        auc = 1 - auc
    fpr, tpr, thr = _roc(y, s)
    n1, n0 = int(y.sum()), int((~y).sum())
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = math.sqrt(max((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0), 0))
    return {"fpr": fpr, "tpr": tpr, "threshold": np.where(np.isfinite(thr), -thr if flipped else thr, np.nan),
            "auc": auc, "se": se, "lo": max(auc - 1.96 * se, 0.0), "hi": min(auc + 1.96 * se, 1.0),
            "n_pos": n1, "n_neg": n0, "flipped": flipped}


# ------------------------------------------------------------------ matrix
def cluster_order(m: np.ndarray) -> list[int]:
    """Leaf order of an average-linkage hierarchical clustering of rows."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    if m.shape[0] < 3:
        return list(range(m.shape[0]))
    filled = np.where(np.isfinite(m), m, np.nanmean(m))
    try:
        return [int(i) for i in leaves_list(linkage(filled, method="average", optimal_ordering=True))]
    except Exception:
        return list(range(m.shape[0]))
