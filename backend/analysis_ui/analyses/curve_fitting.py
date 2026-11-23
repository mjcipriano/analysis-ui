"""Curve fitting: nonlinear regression, standard-curve interpolation and
area under the curve."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core import (AnalysisError, COLORS, Context, Results, analysis, fmt_p,
                    new_figure, opt, stars)
from ..core import curvefit
from ..core.curvefit import MODELS, Fit, compare_fits, fit_model

CATEGORY = "Curve fitting"


# ------------------------------------------------------------------ helpers
def parse_numbers(text: str, what: str = "values") -> list[float]:
    """Numbers typed by the user, separated by commas, semicolons or spaces."""
    out = []
    for tok in re.split(r"[,;\s]+", (text or "").strip()):
        if not tok:
            continue
        try:
            v = float(tok.replace("−", "-"))
        except ValueError:
            raise AnalysisError(f"“{tok}” in the list of {what} is not a number. "
                                "Separate numbers with commas or spaces.") from None
        if not math.isfinite(v):
            raise AnalysisError(f"The list of {what} must contain finite numbers only.")
        out.append(v)
    return out


def p_phrase(p: float) -> str:
    """``P = 0.0123`` / ``P < 0.0001`` for sentences."""
    txt = fmt_p(p)
    return f"P {txt}" if txt.startswith("<") else f"P = {txt}"


def _xy_datasets(ctx: Context) -> tuple[str, dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """``(x name, {data set: (x, y)}, rows dropped for missing values)``.

    Either every Y column is a data set, or one Y column is split by ``group``."""
    xname = ctx.var("x")
    ynames = ctx.vars("y", min=1)
    gname = ctx.var("group", required=False)
    x = ctx.numeric(xname).to_numpy()
    sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dropped = 0
    if gname:
        if len(ynames) > 1:
            raise AnalysisError("Choose either several Y columns or one Y column with a grouping "
                                "variable, not both.")
        if gname in (xname, ynames[0]):
            raise AnalysisError("The grouping variable must differ from the X and Y variables.")
        y = ctx.numeric(ynames[0]).to_numpy()
        g = ctx.factor(gname)
        for lev in g.cat.categories:
            m = (g == lev).to_numpy()
            ok = m & np.isfinite(x) & np.isfinite(y)
            dropped += int(m.sum() - ok.sum())
            if ok.any():
                sets[str(lev)] = (x[ok], y[ok])
    else:
        for yn in ynames:
            if yn == xname:
                raise AnalysisError("X and Y must be different variables.")
            y = ctx.numeric(yn).to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            dropped += int((np.isfinite(x) | np.isfinite(y)).sum() - ok.sum())
            if ok.any():
                sets[yn] = (x[ok], y[ok])
    if not sets:
        raise AnalysisError("There are no rows with both an X and a Y value.")
    return xname, sets, dropped


def _means_at_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False)["y"].mean().sort_values("x")
    return d["x"].to_numpy(), d["y"].to_numpy()


def _use_log_x(choice: str, model: curvefit.Model | None, xs: list[np.ndarray]) -> bool:
    allx = np.concatenate(xs) if xs else np.array([])
    pos = allx[allx > 0]
    if len(pos) < 2:
        return False
    if choice == "log":
        return True
    if choice == "linear":
        return False
    if model is not None and model.log_dose:
        return True
    return bool(np.all(allx > 0) and pos.max() / pos.min() > 100)


def _grid(lo: float, hi: float, log: bool, n: int = 300) -> np.ndarray:
    if log and lo > 0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def _weighted_residuals(fit: Fit) -> np.ndarray:
    return fit.residuals if fit.weights is None else fit.residuals * np.sqrt(fit.weights)


def runs_test(residuals_in_x_order: np.ndarray) -> dict | None:
    """Wald–Wolfowitz runs test, exact one-tailed P (too few runs)."""
    s = np.sign(residuals_in_x_order)
    s = s[s != 0]
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    if n1 == 0 or n2 == 0:
        return None
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    total = math.comb(n1 + n2, n1)

    def prob(r: int) -> float:
        k = r // 2
        if r % 2 == 0:
            return 2 * math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k - 1) / total
        return (math.comb(n1 - 1, k) * math.comb(n2 - 1, k - 1)
                + math.comb(n1 - 1, k - 1) * math.comb(n2 - 1, k)) / total

    p = min(1.0, sum(prob(r) for r in range(2, runs + 1)))
    return {"above": n1, "below": n2, "runs": runs, "p": p}


def lack_of_fit(fit: Fit) -> dict | None:
    """Replicates test: is the scatter around the curve larger than expected
    from the scatter among replicates?"""
    w = fit.weights if fit.weights is not None else np.ones(fit.n)
    d = pd.DataFrame({"x": fit.x, "y": fit.y, "w": w})
    ss_pe = 0.0
    for _, g in d.groupby("x"):
        ybar = np.average(g["y"], weights=g["w"])
        ss_pe += float(np.sum(g["w"] * (g["y"] - ybar) ** 2))
    n_x = d["x"].nunique()
    df_pe = fit.n - n_x
    df_lof = n_x - fit.k
    if df_pe <= 0 or df_lof <= 0 or ss_pe <= 0:
        return None
    ss_lof = max(fit.ss_res - ss_pe, 0.0)
    F = (ss_lof / df_lof) / (ss_pe / df_pe)
    return {"F": F, "df1": df_lof, "df2": df_pe, "p": float(stats.f.sf(F, df_lof, df_pe)),
            "sd_replicates": math.sqrt(ss_pe / df_pe), "sd_lof": math.sqrt(ss_lof / df_lof)}


def derived_values(fit: Fit) -> list[dict]:
    """Quantities computed from the fitted parameters, with 95 % CIs."""
    p, fixed = fit.params, fit.model.fixed
    names = list(fit.model.params)
    tcrit = stats.t.ppf(0.975, fit.dof) if fit.dof > 0 else float("nan")
    out: list[dict] = []

    def val(name):
        return p[name] if name in p else fixed.get(name)

    def mono(label, param, fn, valid=lambda v: True):
        """Monotone transform of one parameter: transform the CI limits."""
        if param not in p or not valid(p[param]):
            return
        lo, hi = fit.ci[param]
        ends = sorted(fn(v) if valid(v) else float("nan") for v in (lo, hi))
        if any(e != e for e in ends):
            ends = [float("nan"), float("nan")]
        out.append({"name": label, "value": fn(p[param]), "lo": ends[0], "hi": ends[1]})

    def delta(label, fn):
        """Function of several parameters: delta-method CI."""
        v0 = fn(dict(p, **fixed))
        grad = np.zeros(len(names))
        for i, nm in enumerate(names):
            h = 1e-6 * max(abs(p[nm]), 1e-6)
            up, dn = dict(p, **fixed), dict(p, **fixed)
            up[nm] += h
            dn[nm] -= h
            grad[i] = (fn(up) - fn(dn)) / (2 * h)
        var = float(grad @ fit.pcov @ grad) if np.all(np.isfinite(fit.pcov)) else float("nan")
        se = math.sqrt(var) if var >= 0 else float("nan")
        out.append({"name": label, "value": v0, "lo": v0 - tcrit * se, "hi": v0 + tcrit * se})

    positive = lambda v: v > 0  # noqa: E731
    mono("LogEC50", "EC50", math.log10, positive)
    mono("EC50", "LogEC50", lambda v: 10 ** v, lambda v: abs(v) < 300)
    if val("Bottom") is not None and val("Top") is not None and ("Bottom" in p or "Top" in p):
        delta("Span (Top − Bottom)", lambda q: q["Top"] - q["Bottom"])
    mid = fit.model.id
    if mid in ("exp_decay", "exp_assoc"):
        mono("Half-life (ln 2 / K)" if mid == "exp_decay" else "Half-time (ln 2 / K)", "K",
             lambda v: math.log(2) / v, positive)
        mono("Tau (1 / K)", "K", lambda v: 1 / v, positive)
        if val("Y0") is not None and val("Plateau") is not None and ("Y0" in p or "Plateau" in p):
            delta("Span", (lambda q: q["Y0"] - q["Plateau"]) if mid == "exp_decay"
                  else (lambda q: q["Plateau"] - q["Y0"]))
    elif mid == "exp_decay2":
        mono("Half-life (fast)", "KFast", lambda v: math.log(2) / v, positive)
        mono("Half-life (slow)", "KSlow", lambda v: math.log(2) / v, positive)
        if all(val(k) is not None for k in ("SpanFast", "SpanSlow")) and ("SpanFast" in p or "SpanSlow" in p):
            delta("Percent fast", lambda q: 100 * q["SpanFast"] / (q["SpanFast"] + q["SpanSlow"]))
    elif mid == "exp_growth":
        mono("Doubling time (ln 2 / k)", "k", lambda v: math.log(2) / v, positive)
    elif mid in ("logistic_growth", "gompertz"):
        pass
    elif mid == "linear" and val("Slope") not in (None, 0) and val("Intercept") is not None:
        delta("X intercept (−Intercept / Slope)", lambda q: -q["Intercept"] / q["Slope"])
        mono("1 / Slope", "Slope", lambda v: 1 / v, lambda v: v != 0)
        if out[-1]["lo"] == out[-1]["lo"] and fit.ci["Slope"][0] * fit.ci["Slope"][1] <= 0:
            out[-1]["lo"] = out[-1]["hi"] = float("nan")   # CI of the slope spans zero
    elif mid == "michaelis_menten" and val("Vmax") is not None and val("Km") is not None:
        delta("Vmax / Km", lambda q: q["Vmax"] / q["Km"])
    return out


def _fit_problem(fit: Fit) -> str | None:
    se = np.array(list(fit.se.values()), dtype=float)
    if not np.all(np.isfinite(se)):
        return ("the fit is ambiguous: at least one parameter could not be determined from these "
                "data (its standard error is infinite)")
    return None


def _try_fit(model, x, y, weighting, fixed=None) -> tuple[Fit | None, str | None]:
    try:
        return fit_model(model, x, y, weighting=weighting, fixed=fixed), None
    except AnalysisError as exc:
        return None, str(exc)
    except Exception as exc:  # numerical trouble inside the optimiser
        return None, f"numerical problem during fitting ({type(exc).__name__}: {exc})"


def _short(name: str) -> str:
    return name.split(":")[0] if name.startswith("Dose") else name


_X_AXIS = [("auto", "Automatic"), ("linear", "Linear"), ("log", "Logarithmic")]
_WEIGHTS = [("none", "No weighting"), ("1/y", "Weight by 1/Y"), ("1/y2", "Weight by 1/Y² (relative)")]


# ====================================================== nonlinear regression
@analysis(
    id="nonlinear_regression", name="Nonlinear regression (curve fit)", category=CATEGORY,
    description="Fit a model (dose–response, Michaelis–Menten, exponential, polynomial …) to XY data.",
    keywords="curve fit ec50 ic50 dose response 4pl logistic michaelis menten exponential decay "
             "half-life polynomial least squares global fit",
    options=[
        opt.variable("x", "X", opt.CONTINUOUS),
        opt.variables("y", "Y (each column is a data set)", opt.CONTINUOUS, min=1),
        opt.variable("group", "Split Y into data sets by", opt.ANY, required=False,
                     help="Optional. Use with a single Y column."),
        opt.choice("model", "Model", curvefit.model_choices(), default="4pl"),
        opt.text("constraints", "Fix parameters to constants", "", placeholder="e.g. Bottom=0, Top=100",
                 section="Fitting", help="Name=value pairs separated by commas."),
        opt.choice("weighting", "Weighting", _WEIGHTS, default="none", section="Fitting"),
        opt.choice("replicates", "Replicates at the same X",
                   [("individual", "Fit every point"), ("means", "Fit the mean at each X")],
                   default="individual", section="Fitting"),
        opt.choice("compare", "Comparison", [
            ("none", "None"),
            ("shared", "Do the data sets share one curve? (extra sum-of-squares F test)"),
            ("aicc", "Compare with a second model (AICc)")], default="none", section="Compare"),
        opt.choice("model2", "Second model", curvefit.model_choices(), default="3pl",
                   section="Compare", visible={"compare": "aicc"}),
        opt.text("interp_y", "Interpolate X at these Y values", "", placeholder="e.g. 50, 90",
                 section="Interpolate"),
        opt.text("interp_x", "Predict Y at these X values", "", placeholder="e.g. 0.1, 1, 10",
                 section="Interpolate"),
        opt.boolean("show_derived", "Derived values (LogEC50, half-life, span …)", True, section="Output"),
        opt.boolean("runs_test", "Runs test", True, section="Output"),
        opt.boolean("replicates_test", "Replicates test for lack of fit", True, section="Output"),
        opt.boolean("normality", "Normality tests of residuals", False, section="Output"),
        opt.boolean("show_plot", "Plot of data and fitted curves", True, section="Plots"),
        opt.boolean("show_band", "95% confidence band", False, section="Plots"),
        opt.boolean("show_residuals", "Residual plot", True, section="Plots"),
        opt.choice("x_axis", "X axis", _X_AXIS, default="auto", section="Plots"),
    ])
def nonlinear_regression(ctx: Context) -> Results:
    xname, sets, dropped = _xy_datasets(ctx)
    model = MODELS[ctx.opt("model", "4pl")]
    fixed = curvefit.parse_constraints(ctx.opt("constraints", ""), model)
    cmodel = curvefit.constrain_model(model, fixed)
    weighting = ctx.opt("weighting", "none")
    use_means = ctx.opt("replicates") == "means"
    multi = len(sets) > 1

    r = Results()
    fits: dict[str, Fit] = {}
    failed: dict[str, str] = {}
    data_used: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (x, y) in sets.items():
        if use_means:
            x, y = _means_at_x(x, y)
        data_used[name] = (x, y)
        fit, err = _try_fit(model, x, y, weighting, fixed)
        if fit is None:
            failed[name] = err or "unknown problem"
        else:
            fits[name] = fit
    if not fits:
        if len(failed) == 1:
            raise AnalysisError(next(iter(failed.values())))
        raise AnalysisError("None of the data sets could be fitted. " +
                            " ".join(f"{k}: {v}" for k, v in failed.items()))
    for name, why in failed.items():
        r.warn(f"Data set “{name}” could not be fitted and is left out: {why}")
    for name, fit in fits.items():
        prob = _fit_problem(fit)
        if prob:
            r.warn(f"Data set “{name}”: {prob}. Consider fixing a parameter or a simpler model.")
        n_in = len(data_used[name][0])
        if fit.n < n_in:
            r.warn(f"Data set “{name}”: {n_in - fit.n} point(s) with X ≤ 0 were excluded because "
                   "this model needs positive X values.")
    if dropped:
        r.warn(f"{dropped} row(s) with a missing X or Y value were ignored.")

    r.text(f"{model.name}\n{cmodel.equation}" + (f"\n{model.description}" if model.description else ""),
           title="Model")

    ds_col = [("dataset", "Data set", "text")] if multi else []

    # ---- best-fit values
    t = r.table("Best-fit values", ds_col + [
        ("param", "Parameter", "text"), ("value", "Best-fit value", "num"), ("se", "Std. error", "num"),
        ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
    for name, fit in fits.items():
        for pn in fit.model.params:
            t.add_row(dataset=name, param=pn, value=fit.params[pn], se=fit.se[pn],
                      lo=fit.ci[pn][0], hi=fit.ci[pn][1])
        for pn, v in fit.model.fixed.items():
            t.add_row(dataset=name, param=f"{pn} (fixed)", value=v)
    t.note("Least-squares fit (Levenberg–Marquardt / trust-region). Confidence intervals are symmetrical "
           "(asymptotic): best-fit value ± t × standard error.")
    if weighting != "none":
        t.note(f"Weighted by {'1/Y' if weighting == '1/y' else '1/Y²'} (weights computed from the observed Y values).")
    if use_means:
        t.note("Fitted to the mean Y at each X value.")

    # ---- derived
    if ctx.opt("show_derived", True):
        rows = [(name, d) for name, fit in fits.items() for d in derived_values(fit)]
        if rows:
            t = r.table("Derived values", ds_col + [
                ("param", "Quantity", "text"), ("value", "Value", "num"),
                ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
            for name, d in rows:
                t.add_row(dataset=name, param=d["name"], value=d["value"], lo=d["lo"], hi=d["hi"])
            t.note("CIs of transformed parameters (log, reciprocal, 10^x) are the transformed limits of the "
                   "parameter's CI; CIs of combinations of parameters use the delta method.")

    # ---- goodness of fit
    t = r.table("Goodness of fit", ds_col + [
        ("n", "Points analysed", "int"), ("df", "Degrees of freedom", "int"), ("r2", "R²", "num"),
        ("adj", "Adjusted R²", "num"), ("ss", "Sum of squares", "num"), ("syx", "Sy.x", "num"),
        ("aicc", "AICc", "num")])
    for name, fit in fits.items():
        t.add_row(dataset=name, n=fit.n, df=fit.dof, r2=fit.r2, adj=fit.adj_r2, ss=fit.ss_res,
                  syx=fit.sy_x, aicc=fit.aicc)
    t.note("Sy.x = √(sum of squares / df), the standard deviation of the residuals. "
           "AICc = N·ln(SS/N) + 2K + 2K(K+1)/(N−K−1) with K = number of parameters + 1.")
    if weighting != "none":
        t.note("Sum of squares, Sy.x and R² are weighted.")

    # ---- residual diagnostics
    if ctx.opt("runs_test", True):
        t = r.table("Runs test", ds_col + [
            ("above", "Points above curve", "int"), ("below", "Points below curve", "int"),
            ("runs", "Number of runs", "int"), ("p", "P value", "p"), ("verdict", "Deviation from model", "text")])
        any_row = False
        for name, fit in fits.items():
            rx, rres = _means_at_x(fit.x, _weighted_residuals(fit))
            rt = runs_test(rres)
            if rt:
                any_row = True
                t.add_row(dataset=name, **rt, verdict="Significant" if rt["p"] < 0.05 else "Not significant")
        if any_row:
            t.note("Wald–Wolfowitz runs test on the signs of the residuals in X order (mean residual at each X); "
                   "exact one-tailed P value for observing this few runs or fewer. A small P value suggests "
                   "the curve deviates systematically from the data.")
        else:
            r.elements.remove(t)
    if ctx.opt("replicates_test", True) and not use_means:
        rows = [(name, lack_of_fit(fit)) for name, fit in fits.items()]
        rows = [(n_, l) for n_, l in rows if l]
        if rows:
            t = r.table("Replicates test for lack of fit", ds_col + [
                ("sd_replicates", "SD replicates", "num"), ("sd_lof", "SD lack of fit", "num"),
                ("F", "F", "num"), ("df1", "DFn", "int"), ("df2", "DFd", "int"), ("p", "P value", "p"),
                ("verdict", "Evidence of inadequate model?", "text")])
            for name, l in rows:
                t.add_row(dataset=name, **l, verdict="Yes" if l["p"] < 0.05 else "No")
            t.note("Compares the scatter of the means around the curve (lack of fit) with the scatter among "
                   "replicates at the same X (pure error) by an F test.")
    if ctx.opt("normality", False):
        t = r.table("Normality of residuals", ds_col + [
            ("test", "Test", "text"), ("stat", "Statistic", "num"), ("p", "P value", "p"),
            ("verdict", "Passed normality test (α = 0.05)?", "text")])
        for name, fit in fits.items():
            res = _weighted_residuals(fit)
            if len(res) >= 3 and np.ptp(res) > 0:
                w, p = stats.shapiro(res)
                t.add_row(dataset=name, test="Shapiro–Wilk (W)", stat=w, p=p, verdict="Yes" if p >= 0.05 else "No")
            if len(res) >= 8 and np.ptp(res) > 0:
                k2, p = stats.normaltest(res)
                t.add_row(dataset=name, test="D'Agostino–Pearson omnibus (K²)", stat=k2, p=p,
                          verdict="Yes" if p >= 0.05 else "No")
        t.note("D'Agostino–Pearson needs at least 8 residuals; Shapiro–Wilk at least 3.")

    # ---- comparison
    compare = ctx.opt("compare", "none")
    cmp_sentence = ""
    if compare == "shared":
        cmp_sentence = _compare_shared(r, cmodel, model, fixed, fits, weighting)
    elif compare == "aicc":
        cmp_sentence = _compare_aicc(r, ctx, model, fits, weighting, multi)

    # ---- interpolation
    ys = parse_numbers(ctx.opt("interp_y", ""), "Y values to interpolate")
    xs = parse_numbers(ctx.opt("interp_x", ""), "X values")
    if ys:
        t = r.table("Interpolated X values", ds_col + [("y", "Y", "num"), ("x", f"Interpolated {xname}", "num"),
                                                       ("note", "Note", "text")])
        for name, fit in fits.items():
            lo, hi = float(fit.x.min()), float(fit.x.max())
            for yv in ys:
                xv = fit.solve_x(yv)
                note = ("The curve does not reach this Y value" if xv != xv
                        else "Extrapolated beyond the X range of the data" if not lo <= xv <= hi else "")
                t.add_row(dataset=name, y=yv, x=xv, note=note)
        t.note("X where the fitted curve equals Y (searched within and slightly beyond the range of the data).")
    if xs:
        t = r.table("Predicted Y values", ds_col + [("x", xname, "num"), ("y", "Predicted Y", "num"),
                                                    ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
        for name, fit in fits.items():
            with np.errstate(all="ignore"):
                yp = fit.predict(xs)
                blo, bhi = fit.band(xs)
            for xv, yv, a, b in zip(xs, yp, blo, bhi):
                t.add_row(dataset=name, x=xv, y=yv, lo=a, hi=b)
        t.note("95% confidence interval of the curve (delta method), not a prediction interval for new points.")

    # ---- summary
    r.summary(_nlr_summary(model, fits, failed) + (" " + cmp_sentence if cmp_sentence else ""))

    # ---- plots
    log_x = _use_log_x(ctx.opt("x_axis", "auto"), model, [d[0] for d in data_used.values()])
    yname = ctx.options["y"][0] if len(ctx.options["y"]) == 1 else "Y"
    if ctx.opt("show_plot", True):
        fig, ax = new_figure(6, 4)
        for i, (name, (x, y)) in enumerate(data_used.items()):
            c = COLORS[i % len(COLORS)]
            m = x > 0 if log_x else np.ones(len(x), bool)
            ax.scatter(x[m], y[m], s=18, color=c, alpha=0.8, label=name if multi else None, zorder=3)
            fit = fits.get(name)
            if fit is not None:
                g = _grid(float(fit.x.min()), float(fit.x.max()), log_x)
                with np.errstate(all="ignore"):
                    ax.plot(g, fit.predict(g), color=c, lw=1.6, zorder=2)
                    if ctx.opt("show_band", False):
                        blo, bhi = fit.band(g)
                        if np.all(np.isfinite(blo)):
                            ax.fill_between(g, blo, bhi, color=c, alpha=0.15, lw=0)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_title(_short(model.name))
        if multi:
            ax.legend()
        r.plot("Data and fitted curve" + ("s" if multi else ""), fig)
    if ctx.opt("show_residuals", True):
        fig, ax = new_figure(6, 3.2)
        for i, (name, fit) in enumerate(fits.items()):
            ax.scatter(fit.x, _weighted_residuals(fit), s=18, color=COLORS[i % len(COLORS)], alpha=0.8,
                       label=name if multi else None)
        ax.axhline(0, color="#555555", lw=0.8, ls="--")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xname)
        ax.set_ylabel("Weighted residual" if weighting != "none" else "Residual")
        if multi:
            ax.legend()
        r.plot("Residuals", fig)

    gopts = {"x": xname, "y": list(ctx.options["y"]), "group": ctx.options.get("group"), "fit": model.id}
    if log_x:
        gopts["x_scale"] = "log10"
    r.graph("Create an XY graph with this fit", "xy", gopts)
    return r


def _nlr_summary(model, fits: dict[str, Fit], failed: dict) -> str:
    key = next((k for k in ("EC50", "LogEC50", "Km", "Kd", "K", "k", "Slope") if k in next(iter(fits.values())).params), None)
    parts = []
    for name, fit in fits.items():
        lead = f"{name}: " if len(fits) + len(failed) > 1 else ""
        if key and math.isfinite(fit.se[key]):
            lo, hi = fit.ci[key]
            parts.append(f"{lead}{key} = {fit.params[key]:.4g} (95% CI {lo:.4g} to {hi:.4g}), R² = {fit.r2:.4f}")
        else:
            parts.append(f"{lead}R² = {fit.r2:.4f}, Sy.x = {fit.sy_x:.4g}")
    s = f"Fitted “{model.name}”. " + "; ".join(parts) + "."
    if failed:
        s += f" {len(failed)} data set(s) could not be fitted."
    return s


def _compare_shared(r: Results, cmodel, model, fixed, fits: dict[str, Fit], weighting: str) -> str:
    if len(fits) < 2:
        r.warn("Comparing curves needs at least two fitted data sets.")
        return ""
    x = np.concatenate([f.x for f in fits.values()])
    y = np.concatenate([f.y for f in fits.values()])
    glob, err = _try_fit(model, x, y, weighting, fixed)
    if glob is None:
        r.warn(f"The single shared curve could not be fitted, so the comparison was skipped: {err}")
        return ""
    ss_sep = sum(f.ss_res for f in fits.values())
    df_sep = sum(f.dof for f in fits.values())
    df1 = glob.dof - df_sep
    if df1 <= 0 or df_sep <= 0 or ss_sep <= 0:
        r.warn("There are too few points to compare the shared curve with separate curves.")
        return ""
    F = ((glob.ss_res - ss_sep) / df1) / (ss_sep / df_sep)
    p = float(stats.f.sf(F, df1, df_sep)) if F > 0 else 1.0
    t = r.table("Do the data sets share one curve?", [
        ("model", "Model", "text"), ("ss", "Sum of squares", "num"), ("df", "df", "int"),
        ("F", "F", "num"), ("dfn", "DFn", "int"), ("dfd", "DFd", "int"), ("p", "P value", "p"), ("s", "Summary", "text")])
    t.add_row(model="Null hypothesis: one curve for all data sets", ss=glob.ss_res, df=glob.dof)
    t.add_row(model="Alternative: a separate curve for each data set", ss=ss_sep, df=df_sep,
              F=F, dfn=df1, dfd=df_sep, p=p, s=stars(p))
    t.note("Extra sum-of-squares F test: all parameters shared between data sets (null) versus "
           "all parameters fitted separately (alternative).")
    t2 = r.table("Shared (global) curve: best-fit values", [
        ("param", "Parameter", "text"), ("value", "Best-fit value", "num"), ("se", "Std. error", "num"),
        ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
    for pn in glob.model.params:
        t2.add_row(param=pn, value=glob.params[pn], se=glob.se[pn], lo=glob.ci[pn][0], hi=glob.ci[pn][1])
    if p < 0.05:
        return (f"The data sets are not adequately described by one shared curve: separate curves fit "
                f"significantly better (F({df1}, {df_sep}) = {F:.3g}, {p_phrase(p)}).")
    return (f"There is no significant evidence that the data sets need different curves "
            f"(F({df1}, {df_sep}) = {F:.3g}, {p_phrase(p)}); one shared curve is adequate.")


def _compare_aicc(r: Results, ctx: Context, model, fits: dict[str, Fit], weighting: str, multi: bool) -> str:
    model2 = MODELS[ctx.opt("model2", "3pl")]
    if model2.id == model.id:
        r.warn("The second model is the same as the first; choose a different model to compare.")
        return ""
    ds_col = [("dataset", "Data set", "text")] if multi else []
    t = r.table("Comparison of models (AICc)", ds_col + [
        ("a1", f"AICc: {_short(model.name)}", "num"), ("a2", f"AICc: {_short(model2.name)}", "num"),
        ("delta", "Difference in AICc", "num"), ("p1", "Probability model 1 is correct", "pct"),
        ("p2", "Probability model 2 is correct", "pct"), ("pref", "Preferred model", "text")])
    sentences = []
    for name, fit in fits.items():
        f2, err = _try_fit(model2, fit.x, fit.y, weighting)
        if f2 is None or f2.n != fit.n or not math.isfinite(f2.aicc) or not math.isfinite(fit.aicc):
            why = err or ("the two models use different points (X ≤ 0 excluded by one of them)"
                          if f2 is not None and f2.n != fit.n else "too few points to compute AICc")
            r.warn(f"Data set “{name}”: the models could not be compared — {why}")
            continue
        delta = f2.aicc - fit.aicc
        p1 = 1 / (1 + math.exp(-0.5 * delta)) if abs(delta) < 1400 else float(delta > 0)
        pref = model if delta > 0 else model2
        t.add_row(dataset=name, a1=fit.aicc, a2=f2.aicc, delta=delta, p1=p1, p2=1 - p1, pref=_short(pref.name))
        sentences.append(f"{name + ': ' if multi else ''}“{_short(pref.name)}” is preferred "
                         f"({100 * max(p1, 1 - p1):.1f}% probability of being the better model)")
    t.note("Difference = AICc(model 2) − AICc(model 1). Probabilities are Akaike weights, "
           "exp(−Δ/2) / (1 + exp(−Δ/2)). The lower AICc is preferred; any constraints apply to model 1 only.")
    return ("AICc comparison: " + "; ".join(sentences) + ".") if sentences else ""


# ================================================ interpolate standard curve
def _band_crossings(fit: Fit, y0: float, xhat: float) -> tuple[float, float]:
    """Inverse-prediction CI: the X values around ``xhat`` where the 95 %
    confidence band of the curve crosses ``y0``."""
    lo, hi = float(fit.x.min()), float(fit.x.max())
    if fit.model.positive_x:
        grid = np.geomspace(max(lo / 10, 1e-300), hi * 10, 3000)
    else:
        pad = (hi - lo) * 0.5
        grid = np.linspace(lo - pad, hi + pad, 3000)
    with np.errstate(all="ignore"):
        blo, bhi = fit.band(grid)
    below, above = float("nan"), float("nan")
    for band_idx, band in enumerate((blo, bhi)):
        f = band - y0
        ok = np.isfinite(f)
        g, f = grid[ok], f[ok]
        for i in np.where(np.diff(np.sign(f)) != 0)[0]:
            try:
                root = optimize.brentq(
                    lambda v: float(fit.band([v])[band_idx][0]) - y0, g[i], g[i + 1])
            except (ValueError, RuntimeError):
                root = float(g[i])
            if root < xhat and not (below > root):
                below = root            # closest crossing below xhat
            elif root > xhat and not (above < root):
                above = root
    return below, above


@analysis(
    id="interpolate_standard_curve", name="Interpolate a standard curve", category=CATEGORY,
    description="Fit a standard curve to known concentrations and read unknown samples off it.",
    keywords="elisa calibration standard curve unknowns interpolation 4pl 5pl linear assay",
    options=[
        opt.variable("x", "Known concentration (X)", opt.CONTINUOUS),
        opt.variable("y", "Signal (Y)", opt.CONTINUOUS,
                     help="Rows with a signal but no concentration are treated as unknowns."),
        opt.variable("unknown_y", "Signals of unknown samples (optional column)", opt.CONTINUOUS, required=False),
        opt.text("unknown_values", "…or type unknown signals", "", placeholder="e.g. 0.45, 1.2, 2.05"),
        opt.variable("label", "Sample names (optional)", opt.ANY, required=False),
        opt.choice("model", "Standard curve model", [
            (m, MODELS[m].name) for m in ("4pl", "5pl", "linear", "quadratic", "cubic", "3pl", "4pl_log",
                                          "michaelis_menten", "hill", "exp_assoc", "exp_decay", "power", "log_line")],
            default="4pl"),
        opt.choice("weighting", "Weighting", _WEIGHTS, default="none", section="Fitting"),
        opt.boolean("show_fit", "Standard-curve parameters", True, section="Output"),
        opt.boolean("show_ci", "95% CI of interpolated values", True, section="Output"),
        opt.boolean("show_plot", "Plot of the standard curve", True, section="Output"),
        opt.choice("x_axis", "X axis", _X_AXIS, default="auto", section="Output"),
    ])
def interpolate_standard_curve(ctx: Context) -> Results:
    xname, yname = ctx.var("x"), ctx.var("y")
    if xname == yname:
        raise AnalysisError("The concentration and the signal must be different variables.")
    x = ctx.numeric(xname).to_numpy()
    y = ctx.numeric(yname).to_numpy()
    labname = ctx.var("label", required=False)
    labels = None
    if labname:
        labels = [None if pd.isna(v) else str(v) for v in ctx.column(labname).astype(object)]
    std = np.isfinite(x) & np.isfinite(y)
    if std.sum() == 0:
        raise AnalysisError("There are no standards: no rows have both a concentration and a signal.")

    unknowns: list[tuple[str, float]] = []
    for i in np.where(~np.isfinite(x) & np.isfinite(y))[0]:
        unknowns.append(((labels[i] if labels and labels[i] else f"Row {i + 1}"), float(y[i])))
    uname = ctx.var("unknown_y", required=False)
    if uname and uname not in (xname, yname):
        u = ctx.numeric(uname).to_numpy()
        for i in np.where(np.isfinite(u))[0]:
            unknowns.append(((labels[i] if labels and labels[i] else f"{uname}, row {i + 1}"), float(u[i])))
    for j, v in enumerate(parse_numbers(ctx.opt("unknown_values", ""), "unknown signals")):
        unknowns.append((f"Typed value {j + 1}", v))

    model = MODELS[ctx.opt("model", "4pl")]
    fit = fit_model(model, x[std], y[std], weighting=ctx.opt("weighting", "none"))
    r = Results()
    if fit.n < int(std.sum()):
        r.warn(f"{int(std.sum()) - fit.n} standard(s) with concentration ≤ 0 were excluded because this "
               "model needs positive X values. (Subtract the blank from the signals instead.)")
    prob = _fit_problem(fit)
    if prob:
        r.warn(f"Standard curve: {prob}. Interpolated values may be unreliable; try a simpler model.")

    if ctx.opt("show_fit", True):
        r.text(f"{model.name}\n{model.equation}", title="Standard curve model")
        t = r.table("Standard curve: best-fit values", [
            ("param", "Parameter", "text"), ("value", "Best-fit value", "num"), ("se", "Std. error", "num"),
            ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
        for pn in model.params:
            t.add_row(param=pn, value=fit.params[pn], se=fit.se[pn], lo=fit.ci[pn][0], hi=fit.ci[pn][1])
        t.note(f"{fit.n} standards, {fit.dof} degrees of freedom, R² = {fit.r2:.4f}, Sy.x = {fit.sy_x:.4g}.")

    x_lo, x_hi = float(fit.x.min()), float(fit.x.max())
    y_lo, y_hi = float(fit.y.min()), float(fit.y.max())
    want_ci = ctx.opt("show_ci", True) and fit.dof > 0 and np.all(np.isfinite(fit.pcov))
    rows = []
    for name, yv in unknowns:
        xv = fit.solve_x(yv)
        lo = hi = float("nan")
        if xv != xv:
            note = "Out of range: the standard curve does not reach this signal"
        else:
            if want_ci:
                lo, hi = _band_crossings(fit, yv, xv)
            if not x_lo <= xv <= x_hi:
                note = "Extrapolated: outside the range of the standards"
            elif not y_lo <= yv <= y_hi:
                note = "Signal outside the range of the standards' signals"
            else:
                note = ""
        rows.append({"name": name, "y": yv, "x": xv, "lo": lo, "hi": hi, "note": note})

    if unknowns:
        cols = [("name", "Sample", "text"), ("y", yname, "num"), ("x", f"Interpolated {xname}", "num")]
        if want_ci:
            cols += [("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")]
        cols.append(("note", "Flag", "text"))
        t = r.table("Interpolated values", cols)
        cap = 500
        for row in rows[:cap]:
            t.add_row(row)
        if len(rows) > cap:
            t.note(f"Only the first {cap} of {len(rows)} unknowns are shown.")
        if want_ci:
            t.note("95% CI by inverse prediction: the X values at which the 95% confidence band of the fitted "
                   "curve equals the signal. It reflects uncertainty of the curve only, not the measurement "
                   "error of the unknown itself. An empty limit means the band does not cross the signal.")
        elif ctx.opt("show_ci", True):
            t.note("Confidence intervals are omitted because the curve has no residual degrees of freedom.")
        ok = sum(1 for row in rows if row["note"] == "")
        r.summary(f"The standard curve ({_short(model.name)}, R² = {fit.r2:.4f}, {fit.n} standards) was used to "
                  f"interpolate {len(rows)} unknown sample(s); {ok} fell within the range of the standards"
                  + (f" and {len(rows) - ok} are flagged." if len(rows) - ok else "."))
    else:
        r.summary(f"The standard curve ({_short(model.name)}) was fitted to {fit.n} standards (R² = {fit.r2:.4f}). "
                  "No unknowns were found: leave the concentration empty on the rows of unknown samples, "
                  "choose a column of unknown signals, or type the signals.")

    if ctx.opt("show_plot", True):
        log_x = _use_log_x(ctx.opt("x_axis", "auto"), model, [fit.x])
        fig, ax = new_figure(6, 4)
        g = _grid(x_lo, x_hi, log_x)
        with np.errstate(all="ignore"):
            ax.plot(g, fit.predict(g), color=COLORS[0], lw=1.6, label="Standard curve", zorder=2)
        ax.scatter(fit.x, fit.y, s=22, color=COLORS[0], label="Standards", zorder=3)
        good = [row for row in rows if row["x"] == row["x"] and (row["x"] > 0 or not log_x)]
        if good:
            ax.scatter([g_["x"] for g_ in good], [g_["y"] for g_ in good], s=46, marker="D",
                       facecolor="white", edgecolor=COLORS[1], linewidth=1.4, label="Unknowns", zorder=4)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.legend()
        r.plot("Standard curve", fig)
    return r


# ======================================================================= AUC
def auc_regions(x: np.ndarray, y: np.ndarray, baseline: float) -> list[dict]:
    """Split a point-to-point curve into regions above / below ``baseline``.

    Each region: ``{"sign", "area", "x0", "x1", "peak_x", "peak_y", "n"}``;
    region boundaries are where the connecting lines cross the baseline."""
    d = y - baseline
    px, pd_, orig = [float(x[0])], [float(d[0])], [True]
    for i in range(1, len(x)):
        if d[i - 1] * d[i] < 0:            # crossing between two points
            f = d[i - 1] / (d[i - 1] - d[i])
            px.append(float(x[i - 1] + f * (x[i] - x[i - 1])))
            pd_.append(0.0)
            orig.append(False)
        px.append(float(x[i]))
        pd_.append(float(d[i]))
        orig.append(True)
    regions: list[dict] = []
    cur: dict | None = None
    for i in range(1, len(px)):
        a, b = pd_[i - 1], pd_[i]
        sign = 1 if (a > 0 or b > 0) else -1 if (a < 0 or b < 0) else 0
        if sign == 0:
            cur = None
            continue
        area = (a + b) / 2 * (px[i] - px[i - 1])
        if cur is None or cur["sign"] != sign:
            cur = {"sign": sign, "area": 0.0, "x0": px[i - 1], "x1": px[i], "pts": set()}
            regions.append(cur)
        cur["area"] += area
        cur["x1"] = px[i]
        for j in (i - 1, i):
            if orig[j] and pd_[j] * sign > 0:
                cur["pts"].add(j)
        if b == 0:
            cur = None
    for reg in regions:
        idx = sorted(reg.pop("pts"))
        k = max(idx, key=lambda j: pd_[j] * reg["sign"])
        reg.update(peak_x=px[k], peak_y=pd_[k] + baseline, height=abs(pd_[k]), n=len(idx))
    return regions


@analysis(
    id="auc", name="Area under the curve", category=CATEGORY,
    description="Trapezoid-rule area under an XY curve, with peaks relative to a baseline.",
    keywords="auc trapezoid peak area pharmacokinetics glucose tolerance baseline",
    options=[
        opt.variable("x", "X", opt.CONTINUOUS),
        opt.variables("y", "Y (each column is a curve)", opt.CONTINUOUS, min=1),
        opt.variable("group", "Split Y into curves by", opt.ANY, required=False,
                     help="Optional. Use with a single Y column."),
        opt.choice("baseline", "Baseline", [
            ("zero", "Y = 0"), ("first", "First Y value"), ("first_k", "Mean of the first points"),
            ("last_k", "Mean of the last points"), ("value", "A value I type")], default="zero"),
        opt.integer("baseline_k", "Number of points to average", 2, min=1, max=1000,
                    visible={"baseline": ["first_k", "last_k"]}),
        opt.number("baseline_value", "Baseline Y", 0.0, visible={"baseline": "value"}),
        opt.number("min_height_pct", "Ignore peaks lower than this % of the Y range", 10.0, min=0, max=100,
                   step=1, section="Peaks", help="Height from the baseline, as a percentage of (max Y − min Y)."),
        opt.integer("min_points", "Ignore peaks with fewer adjacent points than", 1, min=1, max=1000,
                    section="Peaks"),
        opt.boolean("negative_peaks", "Also count peaks below the baseline", False, section="Peaks"),
        opt.boolean("show_peaks", "Table of peaks", True, section="Output"),
        opt.boolean("show_plot", "Plot with shaded area", True, section="Output"),
    ])
def auc(ctx: Context) -> Results:
    xname, sets, dropped = _xy_datasets(ctx)
    multi = len(sets) > 1
    r = Results()
    if dropped:
        r.warn(f"{dropped} row(s) with a missing X or Y value were ignored.")
    mode = ctx.opt("baseline", "zero")
    k = int(ctx.opt("baseline_k", 2))
    min_pct = float(ctx.opt("min_height_pct", 10.0))
    min_pts = int(ctx.opt("min_points", 1))
    negative = bool(ctx.opt("negative_peaks", False))

    ds_col = [("dataset", "Curve", "text")] if multi else []
    summary = r.table("Area under the curve", ds_col + [
        ("baseline", "Baseline", "num"), ("total", "Total area", "num"), ("se", "Std. error", "num"),
        ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num"),
        ("peak_area", "Total peak area", "num"), ("n_peaks", "Number of peaks", "int"),
        ("n_x", "X values", "int")])
    peak_rows, curves, skipped, any_se = [], {}, [], False
    for name, (x, y) in sets.items():
        d = pd.DataFrame({"x": x, "y": y}).groupby("x")["y"].agg(["mean", "count", "std"]).sort_index()
        if len(d) < 2:
            skipped.append(name)
            continue
        ux, my = d.index.to_numpy(dtype=float), d["mean"].to_numpy()
        if mode == "first":
            base = float(my[0])
        elif mode == "first_k":
            base = float(np.mean(my[:min(k, len(my))]))
        elif mode == "last_k":
            base = float(np.mean(my[-min(k, len(my)):]))
        elif mode == "value":
            base = float(ctx.opt("baseline_value", 0.0))
        else:
            base = 0.0
        regions = auc_regions(ux, my, base)
        yrange = float(my.max() - my.min())
        pos_area = sum(g["area"] for g in regions if g["sign"] > 0)
        neg_area = sum(g["area"] for g in regions if g["sign"] < 0)
        total = pos_area + (neg_area if negative else 0.0)
        peaks = [g for g in regions
                 if (g["sign"] > 0 or negative) and g["n"] >= min_pts
                 and g["height"] >= min_pct / 100 * yrange]
        # Gagnon & Peterson (1998): AUC is a weighted sum of the means at each X
        se = lo = hi = float("nan")
        if (d["count"] >= 2).all():
            w = np.empty(len(ux))
            w[0], w[-1] = (ux[1] - ux[0]) / 2, (ux[-1] - ux[-2]) / 2
            w[1:-1] = (ux[2:] - ux[:-2]) / 2
            se = float(np.sqrt(np.sum(w ** 2 * d["std"].to_numpy() ** 2 / d["count"].to_numpy())))
            if negative or neg_area == 0:
                lo, hi = total - 1.96 * se, total + 1.96 * se
                any_se = True
            else:
                se = float("nan")   # the SE describes the net area, which is not what is reported
        summary.add_row(dataset=name, baseline=base, total=total, se=se, lo=lo, hi=hi,
                        peak_area=sum(g["area"] for g in peaks), n_peaks=len(peaks), n_x=len(ux))
        tot_peak = sum(abs(g["area"]) for g in peaks)
        for i, g in enumerate(peaks, 1):
            peak_rows.append(dict(dataset=name, peak=i, area=g["area"],
                                  pct=abs(g["area"]) / tot_peak if tot_peak > 0 else float("nan"),
                                  peak_x=g["peak_x"], peak_y=g["peak_y"], x0=g["x0"], x1=g["x1"], n=g["n"]))
        curves[name] = (ux, my, base, peaks)
    if not curves:
        raise AnalysisError("An area needs at least two different X values with Y values.")
    for name in skipped:
        r.warn(f"Curve “{name}” has fewer than two different X values and was skipped.")

    summary.note("Trapezoid rule on the mean Y at each X. Total area = area between the curve and the baseline"
                 + (", with regions below the baseline counted as negative (net area)." if negative
                    else "; regions below the baseline are ignored."))
    summary.note("Total peak area = sum of the peaks that pass the height and width filters.")
    if any_se:
        summary.note("Standard error from the SEM of the replicates at each X (Gagnon & Peterson, 1998); "
                     "95% CI = area ± 1.96 × SE. The baseline is treated as a fixed constant.")
    else:
        summary.note("A standard error and CI need at least two replicate Y values at every X"
                     + ("" if negative else " and a curve that stays on or above the baseline") + ".")

    if ctx.opt("show_peaks", True) and peak_rows:
        t = r.table("Peaks", ds_col + [
            ("peak", "Peak", "int"), ("area", "Area", "num"), ("pct", "% of peak area", "pct"),
            ("peak_x", "X at peak", "num"), ("peak_y", "Y at peak", "num"),
            ("x0", "First X", "num"), ("x1", "Last X", "num"), ("n", "Points", "int")])
        for row in peak_rows[:200]:
            t.add_row(row)
        t.note("A peak is a region where the curve stays on one side of the baseline; First and Last X are "
               "where the connecting lines cross the baseline (linear interpolation).")
        if len(peak_rows) > 200:
            t.note(f"Only the first 200 of {len(peak_rows)} peaks are shown.")

    parts = []
    for row in summary.rows:
        s = f"{row['dataset'] + ': ' if multi else ''}area = {row['total']:.4g}"
        if row["lo"] == row["lo"]:
            s += f" (95% CI {row['lo']:.4g} to {row['hi']:.4g})"
        parts.append(s)
    r.summary("Area under the curve by the trapezoid rule — " + "; ".join(parts) + ".")

    if ctx.opt("show_plot", True):
        fig, ax = new_figure(6, 4)
        for i, (name, (ux, my, base, peaks)) in enumerate(curves.items()):
            c = COLORS[i % len(COLORS)]
            ax.plot(ux, my, "-o", color=c, ms=3.5, lw=1.4, label=name if multi else None, zorder=3)
            ax.fill_between(ux, my, base, where=(my >= base) | negative, interpolate=True,
                            color=c, alpha=0.18, lw=0)
            ax.axhline(base, color=c if multi else "#555555", lw=0.8, ls="--")
            for g in peaks:
                ax.plot([g["peak_x"]], [g["peak_y"]], marker="v" if g["sign"] > 0 else "^", color=c, ms=6,
                        zorder=4, mec="white")
        ax.set_xlabel(xname)
        ax.set_ylabel(ctx.options["y"][0] if len(ctx.options["y"]) == 1 else "Y")
        if multi:
            ax.legend()
        r.plot("Area under the curve", fig)
    return r
