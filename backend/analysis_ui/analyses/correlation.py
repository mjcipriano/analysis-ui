"""Correlation & regression: correlation (pair or matrix), simple linear
regression (with comparison of lines between groups) and Deming regression."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..core import analysis, opt, Context, Results, AnalysisError, new_figure, COLORS, stars, fmt_p
from .descriptive import conf_opt, conf_level, pct, fnum, p_text, is_constant, parse_numbers

CAT = "Correlation & regression"

_METHODS = [("pearson", "Pearson r"), ("spearman", "Spearman ρ (ranks)"), ("kendall", "Kendall τ-b (ranks)")]
_SYMBOL = {"pearson": "r", "spearman": "ρ", "kendall": "τ"}
_NAME = {"pearson": "Pearson correlation", "spearman": "Spearman rank correlation",
         "kendall": "Kendall rank correlation"}


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

def corr_pair(x: np.ndarray, y: np.ndarray, method: str, conf: float = 0.95) -> dict:
    """Correlation of two complete vectors with Fisher-z confidence interval."""
    n = len(x)
    out = {"n": n, "r": math.nan, "lo": math.nan, "hi": math.nan, "p": math.nan, "r2": math.nan}
    if n < 3 or is_constant(x) or is_constant(y):
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            res = stats.pearsonr(x, y)
            se = 1 / math.sqrt(n - 3) if n > 3 else math.nan
        elif method == "spearman":
            res = stats.spearmanr(x, y)
            se = math.sqrt(1.06 / (n - 3)) if n > 3 else math.nan
        else:
            res = stats.kendalltau(x, y)
            se = math.sqrt(0.437 / (n - 4)) if n > 4 else math.nan
    r = float(res.statistic)
    out.update(r=r, p=float(res.pvalue), r2=r * r)
    if np.isfinite(se) and abs(r) < 1:
        z = math.atanh(r)
        zc = stats.norm.ppf(0.5 + conf / 2)
        out.update(lo=math.tanh(z - zc * se), hi=math.tanh(z + zc * se))
    elif abs(r) >= 1:
        out.update(lo=r, hi=r)
    return out


def _strength(r: float) -> str:
    a = abs(r)
    word = ("very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6
            else "strong" if a < 0.8 else "very strong")
    return f"{word} {'positive' if r > 0 else 'negative'}"


@analysis(
    id="correlation", name="Correlation", category=CAT,
    description="Pearson, Spearman or Kendall correlation between two variables, or a correlation matrix of many.",
    keywords="pearson spearman kendall r rho tau correlation matrix association",
    options=[
        opt.variables("variables", "Variables", opt.CONTINUOUS, min=2),
        opt.choice("method", "Coefficient", _METHODS, default="pearson"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_p", "Matrix of P values", True, section="Output"),
        opt.boolean("show_n", "Matrix of sample sizes", True, section="Output"),
        opt.boolean("show_ci", "Table of all pairs with confidence intervals", False, section="Output"),
        opt.boolean("show_plot", "Scatter plot / heat map", True, section="Output"),
    ])
def correlation(ctx: Context) -> Results:
    names = list(dict.fromkeys(ctx.vars("variables", min=2)))
    if len(names) < 2:
        raise AnalysisError("Please choose at least 2 different variables.")
    method = ctx.opt("method", "pearson")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    sym = _SYMBOL[method]
    data = {n: ctx.numeric(n).to_numpy(float) for n in names}

    def pair(a, b):
        ok = np.isfinite(data[a]) & np.isfinite(data[b])
        return corr_pair(data[a][ok], data[b][ok], method, conf), data[a][ok], data[b][ok]

    method_note = {
        "pearson": "Pearson correlation; two-tailed P value from the t distribution with n − 2 df. "
                   f"{cl} CI by Fisher's z transformation. Assumes both variables are sampled from Gaussian "
                   "distributions and are linearly related.",
        "spearman": "Spearman rank correlation (tied values get average ranks); two-tailed P value from the t "
                    f"approximation. Approximate {cl} CI by Fisher's z transformation with the Fieller correction.",
        "kendall": "Kendall's τ-b (adjusted for ties); two-tailed P value (exact for small samples without ties, "
                   f"otherwise normal approximation). Approximate {cl} CI by Fisher's z with the Fieller correction.",
    }[method]

    r = Results()
    if len(names) == 2:
        a, b = names
        res, x, y = pair(a, b)
        if res["n"] < 3:
            raise AnalysisError(f"Only {res['n']} rows have a value for both “{a}” and “{b}”; at least 3 are needed.")
        if not np.isfinite(res["r"]):
            const = a if is_constant(x) else b
            raise AnalysisError(f"All values of “{const}” are identical, so a correlation cannot be computed.")
        cols = [("pair", "Variables", "text"), ("r", f"{_NAME[method].split()[0]} {sym}", "num"),
                ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num")]
        if method == "pearson":
            cols.append(("r2", "R squared", "num"))
        cols += [("p", "P value (two-tailed)", "p"), ("stars", "P value summary", "text"),
                 ("sig", f"Significant (α = {alpha:g})?", "text"), ("n", "Number of XY pairs", "int")]
        t = r.table(_NAME[method], cols)
        t.add_row(pair=f"{a} vs. {b}", stars=stars(res["p"]), sig="Yes" if res["p"] < alpha else "No", **res)
        t.note(method_note)
        dropped = len(data[a]) - res["n"]
        if dropped:
            t.note(f"{dropped} row(s) with a missing value were excluded.")
        ci_txt = f"; {cl} CI {fnum(res['lo'], 3)} to {fnum(res['hi'], 3)}" if np.isfinite(res["lo"]) else ""
        if res["p"] < alpha:
            r.summary(f"There is a statistically significant {_strength(res['r'])} correlation between {a} and {b} "
                      f"({sym} = {fnum(res['r'], 3)}{ci_txt}; {p_text(res['p'])}; n = {res['n']}). "
                      "Correlation does not by itself show causation.")
        else:
            r.summary(f"The correlation between {a} and {b} is not statistically significant "
                      f"({sym} = {fnum(res['r'], 3)}{ci_txt}; {p_text(res['p'])}; n = {res['n']}).")
        if ctx.opt("show_plot"):
            fig, ax = new_figure(4.6, 3.8)
            ax.scatter(x, y, s=20, color=COLORS[0], alpha=0.8, linewidths=0)
            if method == "pearson":
                slope, inter = np.polyfit(x, y, 1)
                xx = np.array([x.min(), x.max()])
                ax.plot(xx, inter + slope * xx, color=COLORS[1], lw=1.4)
            ax.set_xlabel(a)
            ax.set_ylabel(b)
            ax.set_title(f"{sym} = {res['r']:.3f}")
            r.plot("Scatter plot", fig)
        r.graph("Create a scatter graph", "xy_regression", {"x": a, "y": [b]})
        return r

    k = len(names)
    R = np.full((k, k), math.nan)
    P = np.full((k, k), math.nan)
    N = np.zeros((k, k), int)
    pairs_rows = []
    for i in range(k):
        ok = np.isfinite(data[names[i]])
        N[i, i] = int(ok.sum())
        if N[i, i] >= 2 and not is_constant(data[names[i]][ok]):
            R[i, i] = 1.0
        for j in range(i + 1, k):
            res, _, _ = pair(names[i], names[j])
            R[i, j] = R[j, i] = res["r"]
            P[i, j] = P[j, i] = res["p"]
            N[i, j] = N[j, i] = res["n"]
            pairs_rows.append(dict(pair=f"{names[i]} vs. {names[j]}", stars=stars(res["p"]), **res))
    if not np.isfinite(R[~np.eye(k, dtype=bool)]).any():
        raise AnalysisError("No correlation could be computed: each pair of variables needs at least 3 rows "
                            "with both values, and the values must not all be identical.")

    def matrix(title, M, fmt):
        t = r.table(title, [("var", "", "text"), *[(f"c{j}", n, fmt) for j, n in enumerate(names)]])
        for i, n in enumerate(names):
            t.add_row(var=n, **{f"c{j}": (float(M[i, j]) if fmt != "int" else int(M[i, j])) for j in range(k)})
        return t

    t = matrix(f"Correlation matrix ({_NAME[method].split()[0]} {sym})", R, "num")
    t.note(method_note.split(";")[0] + ". Each coefficient uses all rows with both values (pairwise deletion).")
    bad = [n for i, n in enumerate(names) if not np.isfinite(R[i, i])]
    if bad:
        t.note("No correlation could be computed (too few or identical values) for: " + ", ".join(bad) + ".")
    if ctx.opt("show_p"):
        tp = matrix("P values (two-tailed)", P, "p")
        tp.note("P values are not corrected for multiple comparisons: with "
                f"{k * (k - 1) // 2} correlations some small P values are expected by chance.")
    if ctx.opt("show_n"):
        matrix("Number of XY pairs", N, "int")
    if ctx.opt("show_ci"):
        cols = [("pair", "Variables", "text"), ("r", sym, "num"), ("lo", f"{cl} CI lower", "num"),
                ("hi", f"{cl} CI upper", "num"), ("p", "P value", "p"), ("stars", "Summary", "text"), ("n", "n", "int")]
        tc = r.table("All pairs", cols)
        for row in pairs_rows:
            tc.add_row(**row)
        tc.note(method_note)
    finite = [row for row in pairs_rows if np.isfinite(row["r"])]
    top = max(finite, key=lambda row: abs(row["r"]))
    n_sig = sum(1 for row in finite if row["p"] < alpha)
    r.summary(f"{n_sig} of {len(finite)} pairwise correlations are statistically significant at α = {alpha:g} "
              f"(uncorrected). The strongest is {top['pair']} ({sym} = {fnum(top['r'], 3)}; {p_text(top['p'])}; n = {top['n']}).")
    if ctx.opt("show_plot"):
        size = min(8, 2.2 + 0.55 * k)
        fig, ax = new_figure(size + 0.8, size)
        im = ax.imshow(np.ma.masked_invalid(R), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        if k <= 10:
            for i in range(k):
                for j in range(k):
                    if np.isfinite(R[i, j]):
                        ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if abs(R[i, j]) > 0.6 else "#222222")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, shrink=0.8, label=f"{_NAME[method].split()[0]} {sym}")
        r.plot("Correlation heat map", fig)
    r.graph("Create a correlation matrix graph", "corr_matrix", {"columns": names, "method": method})
    return r


# --------------------------------------------------------------------------
# Simple linear regression
# --------------------------------------------------------------------------

def linfit(x: np.ndarray, y: np.ndarray, origin: bool = False, conf: float = 0.95) -> dict:
    """Ordinary least-squares straight line with everything Prism reports."""
    n = len(x)
    xbar, ybar = x.mean(), y.mean()
    if origin:
        sxx = float(np.sum(x * x))
        slope = float(np.sum(x * y) / sxx)
        inter = 0.0
        df = n - 1
        sst = float(np.sum(y * y))
    else:
        sxx = float(np.sum((x - xbar) ** 2))
        slope = float(np.sum((x - xbar) * (y - ybar)) / sxx)
        inter = float(ybar - slope * xbar)
        df = n - 2
        sst = float(np.sum((y - ybar) ** 2))
    resid = y - (inter + slope * x)
    rss = float(np.sum(resid ** 2))
    mse = rss / df if df > 0 else math.nan
    syx = math.sqrt(mse) if df > 0 else math.nan
    se_slope = syx / math.sqrt(sxx) if df > 0 else math.nan
    se_inter = math.nan if origin else (syx * math.sqrt(1 / n + xbar ** 2 / sxx) if df > 0 else math.nan)
    tc = stats.t.ppf(0.5 + conf / 2, df) if df > 0 else math.nan
    f = (slope / se_slope) ** 2 if df > 0 and se_slope > 0 else (math.inf if df > 0 and slope != 0 else math.nan)
    p = float(stats.f.sf(f, 1, df)) if np.isfinite(f) else (0.0 if f == math.inf else math.nan)
    out = {"n": n, "slope": slope, "inter": inter, "df": df, "rss": rss, "sxx": sxx, "xbar": float(xbar),
           "ybar": float(ybar), "syx": syx, "se_slope": se_slope, "se_inter": se_inter,
           "slope_lo": slope - tc * se_slope, "slope_hi": slope + tc * se_slope,
           "inter_lo": inter - tc * se_inter, "inter_hi": inter + tc * se_inter,
           "r2": 1 - rss / sst if sst > 0 else math.nan, "f": f, "p": p, "tc": tc, "origin": origin,
           "resid": resid, "xint": math.nan, "xint_lo": math.nan, "xint_hi": math.nan,
           "inv_slope": 1 / slope if slope != 0 else math.nan}
    if not origin and slope != 0:
        out["xint"] = -inter / slope
        # Fieller's theorem for the CI of the X intercept
        if df > 0:
            g = tc * tc * mse
            A = slope ** 2 - g / sxx
            B = 2 * slope * ybar
            C = ybar ** 2 - g / n
            disc = B * B - 4 * A * C
            if A > 0 and disc >= 0:
                r1 = xbar + (-B - math.sqrt(disc)) / (2 * A)
                r2 = xbar + (-B + math.sqrt(disc)) / (2 * A)
                out["xint_lo"], out["xint_hi"] = min(r1, r2), max(r1, r2)
    return out


def predict(fit: dict, x0: np.ndarray, kind: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Fitted values and half-width of the confidence (or prediction) band."""
    x0 = np.asarray(x0, float)
    yhat = fit["inter"] + fit["slope"] * x0
    if fit["df"] <= 0:
        return yhat, np.full_like(yhat, math.nan)
    if fit["origin"]:
        var = x0 ** 2 / fit["sxx"]
    else:
        var = 1 / fit["n"] + (x0 - fit["xbar"]) ** 2 / fit["sxx"]
    if kind == "prediction":
        var = var + 1
    return yhat, fit["tc"] * fit["syx"] * np.sqrt(var)


def runs_test(resid: np.ndarray) -> dict:
    """Wald–Wolfowitz runs test on the signs of residuals (ordered by X).
    One-tailed exact P value: probability of this few runs or fewer."""
    s = np.sign(resid[np.abs(resid) > 1e-12 * max(1.0, float(np.max(np.abs(resid))) if len(resid) else 1.0)])
    n1, n2 = int(np.sum(s > 0)), int(np.sum(s < 0))
    out = {"above": n1, "below": n2, "runs": math.nan, "p": math.nan}
    if n1 == 0 or n2 == 0:
        return out
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    out["runs"] = runs
    total = math.comb(n1 + n2, n1)
    cum = 0
    for rr in range(2, runs + 1):
        k = rr // 2
        if rr % 2 == 0:
            cum += 2 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1)
        else:
            cum += (math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
                    + math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k))
    out["p"] = min(1.0, cum / total)
    return out


def _equation(fit: dict, xn: str = "X", yn: str = "Y") -> str:
    if fit["origin"]:
        return f"{yn} = {fnum(fit['slope'])} × {xn}"
    sign = "+" if fit["inter"] >= 0 else "−"
    return f"{yn} = {fnum(fit['slope'])} × {xn} {sign} {fnum(abs(fit['inter']))}"


def compare_lines(sets: dict[str, tuple[np.ndarray, np.ndarray]], fits: dict[str, dict]) -> dict:
    """Are the slopes equal? If so, are the elevations (intercepts) equal?"""
    k = len(sets)
    N = sum(len(x) for x, _ in sets.values())
    rss_sep = sum(f["rss"] for f in fits.values())
    df_sep = N - 2 * k
    sxx = sum(f["sxx"] for f in fits.values())
    sxy = sum(float(np.sum((x - x.mean()) * (y - y.mean()))) for x, y in sets.values())
    syy = sum(float(np.sum((y - y.mean()) ** 2)) for _, y in sets.values())
    pooled_slope = sxy / sxx
    rss_par = syy - sxy ** 2 / sxx
    df_par = N - k - 1
    allx = np.concatenate([x for x, _ in sets.values()])
    ally = np.concatenate([y for _, y in sets.values()])
    one = linfit(allx, ally)
    out = {"pooled_slope": pooled_slope, "k": k, "N": N,
           "f_slope": math.nan, "p_slope": math.nan, "dfn": k - 1, "dfd_slope": df_sep,
           "f_elev": math.nan, "p_elev": math.nan, "dfd_elev": df_par,
           "pooled_inter": one["inter"], "one_slope": one["slope"],
           "intercepts": {g: float(y.mean() - pooled_slope * x.mean()) for g, (x, y) in sets.items()}}
    if df_sep > 0 and rss_sep > 0:
        out["f_slope"] = max(0.0, (rss_par - rss_sep) / (k - 1)) / (rss_sep / df_sep)
        out["p_slope"] = float(stats.f.sf(out["f_slope"], k - 1, df_sep))
    if df_par > 0 and rss_par > 0:
        out["f_elev"] = max(0.0, (one["rss"] - rss_par) / (k - 1)) / (rss_par / df_par)
        out["p_elev"] = float(stats.f.sf(out["f_elev"], k - 1, df_par))
    return out


@analysis(
    id="linear_regression", name="Simple linear regression", category=CAT,
    description="Fit a straight line through XY data; optionally compare the lines of several groups.",
    keywords="line slope intercept regression least squares ancova interpolate standard curve",
    options=[
        opt.variable("x", "X", opt.CONTINUOUS),
        opt.variable("y", "Y", opt.CONTINUOUS),
        opt.variable("group", "Group (one line per group)", opt.ANY, required=False,
                     help="Fits a line to each group and tests whether slopes and intercepts differ."),
        opt.boolean("origin", "Force the line through the origin (0, 0)", False, section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.text("interp_x", "Interpolate Y at these X values", "", placeholder="e.g. 2.5, 10", section="Interpolate"),
        opt.text("interp_y", "Interpolate X at these Y values", "", placeholder="e.g. 0.5", section="Interpolate"),
        opt.boolean("runs", "Runs test (departure from linearity)", True, section="Assumption checks"),
        opt.boolean("show_resid", "Residual plot", True, section="Assumption checks"),
        opt.boolean("show_plot", "Graph of the fit with confidence band", True, section="Output"),
    ])
def linear_regression(ctx: Context) -> Results:
    xn, yn = ctx.var("x"), ctx.var("y")
    gn = ctx.var("group", required=False)
    if xn == yn:
        raise AnalysisError("X and Y must be two different variables.")
    if gn in (xn, yn):
        raise AnalysisError("The grouping variable must be different from X and Y.")
    origin = bool(ctx.opt("origin"))
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    df_all = pd.DataFrame({"x": ctx.numeric(xn).to_numpy(float), "y": ctx.numeric(yn).to_numpy(float)})
    if gn:
        fac = ctx.factor(gn)
        df_all["g"] = fac.astype(object).to_numpy()
        levels = [str(l) for l in fac.cat.categories]
    df_all = df_all[np.isfinite(df_all["x"]) & np.isfinite(df_all["y"])]
    if gn:
        df_all = df_all[df_all["g"].notna()]
        sets = {l: (df_all.loc[df_all["g"] == l, "x"].to_numpy(), df_all.loc[df_all["g"] == l, "y"].to_numpy())
                for l in levels}
        sets = {l: v for l, v in sets.items() if len(v[0])}
    else:
        sets = {"All data": (df_all["x"].to_numpy(), df_all["y"].to_numpy())}
    if not sets:
        raise AnalysisError("No rows have a value for both X and Y.")
    min_n = 2 if origin else 3
    fits, skipped = {}, []
    for name, (x, y) in sets.items():
        if len(x) < min_n:
            skipped.append(f"{name} (only {len(x)} point{'s' if len(x) != 1 else ''})")
        elif is_constant(x) and not origin:
            skipped.append(f"{name} (all X values identical)")
        elif origin and np.all(x == 0):
            skipped.append(f"{name} (all X values are zero)")
        else:
            fits[name] = linfit(x, y, origin, conf)
    if not fits:
        raise AnalysisError("A line cannot be fitted: " + "; ".join(skipped) + ". At least "
                            f"{min_n} points with different X values are needed.")
    sets = {k: v for k, v in sets.items() if k in fits}
    multi = len(fits) > 1
    lab = "Group" if gn else "Data"
    r = Results()
    if skipped:
        r.warn("Not fitted: " + "; ".join(skipped) + ".")

    t = r.table("Best-fit values",
                [("g", lab, "text"), ("slope", "Slope", "num"), ("inter", "Y-intercept", "num"),
                 ("xint", "X-intercept", "num"), ("inv", "1/slope", "num"), ("eq", "Equation", "text")])
    for name, f in fits.items():
        t.add_row(g=name, slope=f["slope"], inter=f["inter"], xint=f["xint"], inv=f["inv_slope"],
                  eq=_equation(f, xn, yn))
    t.note("Ordinary least squares: minimises the sum of squared vertical (Y) distances from the line"
           + (", constrained to pass through the origin." if origin else "."))
    se = r.table(f"Standard errors and {cl} confidence intervals",
                 [("g", lab, "text"), ("se_slope", "Slope SE", "num"), ("slope_lo", "Slope CI lower", "num"),
                  ("slope_hi", "Slope CI upper", "num"), ("se_inter", "Y-intercept SE", "num"),
                  ("inter_lo", "Y-intercept CI lower", "num"), ("inter_hi", "Y-intercept CI upper", "num"),
                  ("xint_lo", "X-intercept CI lower", "num"), ("xint_hi", "X-intercept CI upper", "num")])
    for name, f in fits.items():
        se.add_row(g=name, **{k: f[k] for k in ("se_slope", "slope_lo", "slope_hi", "se_inter", "inter_lo",
                                                 "inter_hi", "xint_lo", "xint_hi")})
    if not origin:
        se.note("The CI of the X-intercept uses Fieller's theorem; it is empty when the slope is not "
                "significantly different from zero (the interval is then unbounded).")
    gof = r.table("Goodness of fit",
                  [("g", lab, "text"), ("r2", "R squared", "num"), ("syx", "Sy.x", "num"),
                   ("n", "Number of points", "int"), ("df", "Residual df", "int")])
    for name, f in fits.items():
        gof.add_row(g=name, r2=f["r2"], syx=f["syx"], n=f["n"], df=f["df"])
    gof.note("Sy.x is the standard deviation of the residuals (standard error of the estimate)."
             + (" With the line forced through the origin, R squared is computed relative to Y = 0 "
                "(uncentred), as in R; it is not comparable with the R squared of an ordinary fit." if origin else ""))
    sl = r.table("Is the slope significantly non-zero?",
                 [("g", lab, "text"), ("f", "F", "num"), ("dfn", "DFn", "int"), ("dfd", "DFd", "int"),
                  ("p", "P value", "p"), ("stars", "P value summary", "text"),
                  ("sig", "Deviation from zero?", "text")])
    for name, f in fits.items():
        sl.add_row(g=name, f=f["f"], dfn=1, dfd=f["df"], p=f["p"], stars=stars(f["p"]),
                   sig=("Significant" if f["p"] < alpha else "Not significant") if np.isfinite(f["p"]) else "")
    sl.note("F test of the null hypothesis that the slope is zero (equivalent to the two-tailed t test of the slope).")
    perfect = [n for n, f in fits.items() if f["df"] > 0 and f["rss"] <= 1e-20 * max(1.0, f["ybar"] ** 2)]
    if perfect:
        r.warn("The points lie exactly on a line for: " + ", ".join(perfect) + ". Standard errors are zero and "
               "P values are not meaningful.")

    if ctx.opt("runs") and not origin:
        rt = r.table("Runs test (departure from linearity)",
                     [("g", lab, "text"), ("above", "Points above line", "int"), ("below", "Points below line", "int"),
                      ("runs", "Number of runs", "int"), ("p", "P value (runs test)", "p"),
                      ("dev", "Deviation from linearity", "text")])
        for name, f in fits.items():
            x = sets[name][0]
            rn = runs_test(f["resid"][np.argsort(x, kind="stable")])
            rt.add_row(g=name, dev=("Significant" if rn["p"] < alpha else "Not significant") if np.isfinite(rn["p"]) else "not tested", **rn)
        rt.note("A run is a series of consecutive points (ordered by X) on the same side of the line. "
                "One-tailed exact P value: the chance of so few runs if the points were randomly scattered around "
                "the line. A small P value suggests the relationship is curved.")

    cmp_res = None
    if multi and not origin:
        cmp_res = compare_lines(sets, fits)
        ct = r.table("Are the lines different?",
                     [("q", "Question", "text"), ("f", "F", "num"), ("dfn", "DFn", "int"), ("dfd", "DFd", "int"),
                      ("p", "P value", "p"), ("stars", "P value summary", "text"), ("concl", "Conclusion", "text")])
        ps = cmp_res["p_slope"]
        ct.add_row(q="Are the slopes equal?", f=cmp_res["f_slope"], dfn=cmp_res["dfn"], dfd=cmp_res["dfd_slope"],
                   p=ps, stars=stars(ps),
                   concl=("Slopes differ significantly" if ps < alpha else
                          f"Slopes not significantly different; pooled slope = {fnum(cmp_res['pooled_slope'])}")
                   if np.isfinite(ps) else "not tested")
        pe = cmp_res["p_elev"]
        if np.isfinite(ps) and ps < alpha:
            ct.add_row(q="Are the elevations or intercepts equal?",
                       concl="Not tested, because the slopes differ significantly")
        else:
            ct.add_row(q="Are the elevations or intercepts equal?", f=cmp_res["f_elev"], dfn=cmp_res["dfn"],
                       dfd=cmp_res["dfd_elev"], p=pe, stars=stars(pe),
                       concl=("Elevations/intercepts differ significantly" if pe < alpha else
                              f"Not significantly different; pooled intercept = {fnum(cmp_res['pooled_inter'])}")
                       if np.isfinite(pe) else "not tested")
        ct.note("Equivalent to analysis of covariance (ANCOVA). Slopes: F test of the X × group interaction. "
                "Elevations: F test of the group effect in a model with one shared slope; it is only meaningful "
                "when the slopes do not differ.")
    elif multi and origin:
        r.text("Slopes and intercepts of the groups are not compared when the lines are forced through the origin.")

    # interpolation
    xs = parse_numbers(ctx.opt("interp_x", ""), "X values to interpolate")
    ys = parse_numbers(ctx.opt("interp_y", ""), "Y values to interpolate")
    if xs or ys:
        it = r.table("Interpolated values",
                     [("g", lab, "text"), ("x", xn, "num"), ("y", yn, "num"),
                      ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num"),
                      ("plo", f"{cl} prediction lower", "num"), ("phi", f"{cl} prediction upper", "num"),
                      ("what", "Interpolated", "text"), ("note", "Note", "text")])
        extrap = False
        for name, f in fits.items():
            x = sets[name][0]
            for x0 in xs:
                yh, hw = predict(f, [x0])
                _, pw = predict(f, [x0], "prediction")
                out = x0 < x.min() or x0 > x.max()
                extrap |= out
                it.add_row(g=name, x=x0, y=float(yh[0]), lo=float(yh[0] - hw[0]), hi=float(yh[0] + hw[0]),
                           plo=float(yh[0] - pw[0]), phi=float(yh[0] + pw[0]), what="Y from X",
                           note="outside the range of the data" if out else "")
            for y0 in ys:
                if f["slope"] == 0:
                    it.add_row(g=name, y=y0, what="X from Y", note="slope is zero")
                    continue
                x0 = (y0 - f["inter"]) / f["slope"]
                out = x0 < x.min() or x0 > x.max()
                extrap |= out
                it.add_row(g=name, x=float(x0), y=y0, what="X from Y",
                           note="outside the range of the data" if out else "")
        it.note("For Y from X: the CI is for the mean Y at that X (the confidence band of the line); the prediction "
                "interval is where a single new observation is expected. X from Y is read off the fitted line, "
                "without an interval.")
        if extrap:
            it.note("Values outside the range of the data are extrapolations and should be treated with caution.")

    # summary
    sentences = []
    for name, f in list(fits.items())[:4]:
        prefix = f"{name}: " if gn else ""
        if not np.isfinite(f["p"]):
            sentences.append(f"{prefix}{_equation(f, xn, yn)}.")
            continue
        sentences.append(
            prefix + (f"the slope is significantly different from zero" if f["p"] < alpha else
                      "the slope is not significantly different from zero")
            + f" ({p_text(f['p'])}; F({1}, {f['df']}) = {fnum(f['f'])}). {_equation(f, xn, yn)}; "
              f"slope {fnum(f['slope'])} ({cl} CI {fnum(f['slope_lo'])} to {fnum(f['slope_hi'])}); "
              f"R² = {fnum(f['r2'], 3)}.")
    if sentences and not gn:
        sentences[0] = sentences[0][0].upper() + sentences[0][1:]
    if cmp_res is not None and np.isfinite(cmp_res["p_slope"]):
        if cmp_res["p_slope"] < alpha:
            sentences.append(f"The slopes of the groups differ significantly ({p_text(cmp_res['p_slope'])}).")
        else:
            s = f"The slopes are not significantly different ({p_text(cmp_res['p_slope'])})"
            if np.isfinite(cmp_res["p_elev"]):
                s += ("; the elevations (intercepts) differ significantly" if cmp_res["p_elev"] < alpha else
                      "; the elevations (intercepts) are not significantly different either")
                s += f" ({p_text(cmp_res['p_elev'])})"
            sentences.append(s + ".")
    r.summary(" ".join(sentences))

    if ctx.opt("show_plot"):
        fig, ax = new_figure(5.4, 3.9)
        for i, (name, f) in enumerate(fits.items()):
            x, y = sets[name]
            c = COLORS[i % len(COLORS)]
            ax.scatter(x, y, s=20, color=c, alpha=0.8, linewidths=0, label=name if multi else None, zorder=3)
            lo = min(x.min(), 0.0) if origin else x.min()
            xx = np.linspace(lo, x.max(), 100)
            yh, hw = predict(f, xx)
            ax.plot(xx, yh, color=c if multi else COLORS[1], lw=1.5, zorder=4)
            if np.all(np.isfinite(hw)):
                ax.fill_between(xx, yh - hw, yh + hw, color=c if multi else COLORS[1], alpha=0.15, linewidth=0)
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        if multi:
            ax.legend(title=gn)
        r.plot(f"Linear regression with {cl} confidence band", fig)
    if ctx.opt("show_resid"):
        fig, ax = new_figure(5.4, 3.2)
        for i, (name, f) in enumerate(fits.items()):
            ax.scatter(sets[name][0], f["resid"], s=20, color=COLORS[i % len(COLORS)], alpha=0.8, linewidths=0,
                       label=name if multi else None)
        ax.axhline(0, color="#888888", lw=1, ls="--")
        ax.set_xlabel(xn)
        ax.set_ylabel(f"Residual ({yn})")
        if multi:
            ax.legend(title=gn)
        r.plot("Residual plot", fig)
    r.graph("Create an XY graph with regression line", "xy_regression", {"x": xn, "y": [yn], "group": gn})
    return r


# --------------------------------------------------------------------------
# Deming regression
# --------------------------------------------------------------------------

def deming_fit(x: np.ndarray, y: np.ndarray, delta: float = 1.0) -> tuple[float, float]:
    """Deming slope and intercept. ``delta`` = error variance of Y / error variance of X."""
    xbar, ybar = x.mean(), y.mean()
    sxx = float(np.sum((x - xbar) ** 2))
    syy = float(np.sum((y - ybar) ** 2))
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    if sxy == 0:
        return math.nan, math.nan
    u = syy - delta * sxx
    slope = (u + math.sqrt(u * u + 4 * delta * sxy * sxy)) / (2 * sxy)
    return float(slope), float(ybar - slope * xbar)


@analysis(
    id="deming", name="Deming regression", category=CAT,
    description="Fit a line when both X and Y are measured with error (method comparison).",
    keywords="deming orthogonal errors in variables method comparison model II regression",
    options=[
        opt.variable("x", "X (method 1)", opt.CONTINUOUS),
        opt.variable("y", "Y (method 2)", opt.CONTINUOUS),
        opt.number("ratio", "Ratio of error variances (Y / X)", 1, min=1e-6, max=1e6, nullable=False,
                   help="Variance of the measurement error of Y divided by that of X. "
                        "1 = both methods equally precise (orthogonal regression)."),
        conf_opt(section="Options"),
        opt.boolean("show_plot", "Graph of the fit", True, section="Output"),
    ])
def deming(ctx: Context) -> Results:
    xn, yn = ctx.var("x"), ctx.var("y")
    if xn == yn:
        raise AnalysisError("X and Y must be two different variables.")
    delta = float(ctx.opt("ratio", 1))
    conf = conf_level(ctx)
    cl = pct(conf)
    d = pd.DataFrame({"x": ctx.numeric(xn).to_numpy(float), "y": ctx.numeric(yn).to_numpy(float)})
    d = d[np.isfinite(d["x"]) & np.isfinite(d["y"])]
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    n = len(x)
    if n < 4:
        raise AnalysisError(f"Deming regression needs at least 4 complete XY pairs (found {n}).")
    if is_constant(x) or is_constant(y):
        raise AnalysisError("All values of " + (f"“{xn}”" if is_constant(x) else f"“{yn}”")
                            + " are identical, so a line cannot be fitted.")
    slope, inter = deming_fit(x, y, delta)
    if not np.isfinite(slope):
        raise AnalysisError("X and Y are completely uncorrelated, so the Deming slope is undefined.")
    # leave-one-out jackknife (Linnet 1990)
    idx = np.arange(n)
    jk = np.array([deming_fit(x[idx != i], y[idx != i], delta) for i in range(n)])
    pseudo = n * np.array([slope, inter]) - (n - 1) * jk
    se = np.nanstd(pseudo, axis=0, ddof=1) / math.sqrt(n)
    df = n - 2
    tc = stats.t.ppf(0.5 + conf / 2, df)
    r = Results()
    t = r.table("Deming regression",
                [("param", "Parameter", "text"), ("est", "Best-fit value", "num"), ("se", "Std. error", "num"),
                 ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num"),
                 ("ref", "Compared with", "num"), ("p", "P value", "p"), ("concl", "Conclusion", "text")])
    rows = []
    for name, est, s, ref in (("Slope", slope, se[0], 1.0), ("Y-intercept", inter, se[1], 0.0)):
        lo, hi = est - tc * s, est + tc * s
        p = float(2 * stats.t.sf(abs(est - ref) / s, df)) if s > 0 else math.nan
        inside = lo <= ref <= hi
        concl = (f"CI includes {ref:g}" if inside else f"CI excludes {ref:g}") if s > 0 else ""
        t.add_row(param=name, est=est, se=s, lo=lo, hi=hi, ref=ref, p=p, concl=concl)
        rows.append((name, est, lo, hi, inside, p))
    t.note(f"Deming regression with error-variance ratio (Y / X) = {delta:g}"
           + (" (orthogonal regression: perpendicular distances are minimised)." if delta == 1 else "."))
    t.note(f"Standard errors by the leave-one-out jackknife (Linnet 1990); CIs and P values use the t distribution "
           f"with n − 2 = {df} df. P values test slope = 1 and intercept = 0 (identity line, i.e. no proportional "
           "and no constant difference between the methods).")
    extra = r.table("Other results",
                    [("n", "Number of XY pairs", "int"), ("xint", "X-intercept", "num"), ("inv", "1/slope", "num"),
                     ("r", "Pearson r", "num"), ("ols", "Ordinary least-squares slope", "num"), ("eq", "Equation", "text")])
    ols = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    eq = f"{yn} = {fnum(slope)} × {xn} {'+' if inter >= 0 else '−'} {fnum(abs(inter))}"
    extra.add_row(n=n, xint=-inter / slope if slope else math.nan, inv=1 / slope if slope else math.nan,
                  r=float(np.corrcoef(x, y)[0, 1]), ols=ols, eq=eq)
    extra.note("Ordinary least squares assumes X is measured without error and gives a slope biased towards zero "
               "when it is not.")
    s_ok, i_ok = rows[0][4], rows[1][4]
    r.summary(
        f"{eq}. Slope {fnum(slope)} ({cl} CI {fnum(rows[0][2])} to {fnum(rows[0][3])}), intercept {fnum(inter)} "
        f"({cl} CI {fnum(rows[1][2])} to {fnum(rows[1][3])}). "
        + ("The slope is not significantly different from 1" if s_ok else "The slope differs significantly from 1")
        + f" ({p_text(rows[0][5])}) and "
        + ("the intercept is not significantly different from 0" if i_ok else "the intercept differs significantly from 0")
        + f" ({p_text(rows[1][5])})"
        + ("; the data are consistent with the two methods agreeing." if s_ok and i_ok else
           "; there is evidence of a " + " and a ".join(
               filter(None, ["proportional" if not s_ok else "", "constant" if not i_ok else ""]))
           + " difference between the methods."))
    if ctx.opt("show_plot"):
        fig, ax = new_figure(4.8, 4.0)
        ax.scatter(x, y, s=20, color=COLORS[0], alpha=0.8, linewidths=0, zorder=3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        xx = np.array([x.min(), x.max()])
        ax.plot([lo, hi], [lo, hi], color="#999999", ls="--", lw=1, label="Identity (Y = X)")
        ax.plot(xx, inter + slope * xx, color=COLORS[1], lw=1.6, label="Deming fit")
        ax.set_xlabel(xn)
        ax.set_ylabel(yn)
        ax.legend(loc="best")
        r.plot("Deming regression", fig)
    r.graph("Create an XY graph", "xy_regression", {"x": xn, "y": [yn]})
    return r
