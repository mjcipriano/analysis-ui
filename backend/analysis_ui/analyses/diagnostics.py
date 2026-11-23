"""Diagnostics & agreement: ROC curves, Bland–Altman, intraclass correlation
and Cohen's kappa."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from ..core import (AnalysisError, COLORS, Context, Results, analysis, fmt_p,
                    new_figure, opt, stars)
from .curve_fitting import p_phrase
from ..graphs.types import _binary_outcome

CATEGORY = "Diagnostics & agreement"
Z95 = stats.norm.ppf(0.975)


# ======================================================================= ROC
def delong(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their covariance matrix by DeLong, DeLong & Clarke-Pearson (1988).

    ``pos_scores`` (m × K) and ``neg_scores`` (n × K) hold K markers measured
    on the same m positive and n negative subjects (higher = more positive).
    Uses the mid-rank algorithm of Sun & Xu (2014)."""
    pos = np.atleast_2d(np.asarray(pos_scores, dtype=float).T).T
    neg = np.atleast_2d(np.asarray(neg_scores, dtype=float).T).T
    m, n = pos.shape[0], neg.shape[0]
    k = pos.shape[1]
    v10 = np.empty((m, k))
    v01 = np.empty((n, k))
    for j in range(k):
        both = stats.rankdata(np.concatenate([pos[:, j], neg[:, j]]))
        v10[:, j] = (both[:m] - stats.rankdata(pos[:, j])) / n
        v01[:, j] = 1 - (both[m:] - stats.rankdata(neg[:, j])) / m
    auc = v10.mean(axis=0)
    s10 = np.atleast_2d(np.cov(v10, rowvar=False)) if m > 1 else np.zeros((k, k))
    s01 = np.atleast_2d(np.cov(v01, rowvar=False)) if n > 1 else np.zeros((k, k))
    return auc, s10 / m + s01 / n


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    z2 = Z95 ** 2
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def cutoff_table(pos: np.ndarray, neg: np.ndarray, higher_positive: bool) -> pd.DataFrame:
    """Sensitivity / specificity at every cut-off midway between adjacent
    observed values. A subject tests positive when its value is beyond the cut-off."""
    values = np.unique(np.concatenate([pos, neg]))
    cuts = (values[:-1] + values[1:]) / 2
    rows = []
    for c in cuts:
        tp = int(np.sum(pos > c) if higher_positive else np.sum(pos < c))
        tn = int(np.sum(neg <= c) if higher_positive else np.sum(neg >= c))
        sens, spec = tp / len(pos), tn / len(neg)
        slo, shi = wilson(tp, len(pos))
        plo, phi = wilson(tn, len(neg))
        rows.append({"cutoff": float(c), "sens": sens, "sens_lo": slo, "sens_hi": shi,
                     "spec": spec, "spec_lo": plo, "spec_hi": phi,
                     "lr": sens / (1 - spec) if spec < 1 else float("nan"),
                     "youden": sens + spec - 1})
    return pd.DataFrame(rows)


@analysis(
    id="roc", name="ROC curve analysis", category=CATEGORY,
    description="Area under the ROC curve, sensitivity and specificity at each cut-off, and comparison of markers.",
    keywords="roc auc receiver operating characteristic sensitivity specificity cutoff youden delong diagnostic test",
    options=[
        opt.variable("outcome", "Outcome (two groups)", opt.ANY),
        opt.level("positive", "Positive / diseased group", ["outcome"]),
        opt.variables("predictors", "Markers / predictors", opt.CONTINUOUS, min=1),
        opt.choice("direction", "Which values indicate a positive result?", [
            ("auto", "Decide automatically (AUC ≥ 0.5)"), ("higher", "Higher values"), ("lower", "Lower values")],
            default="auto", section="Method"),
        opt.choice("se_method", "Standard error of the AUC", [
            ("delong", "DeLong (nonparametric)"), ("hanley", "Hanley & McNeil (1982)")],
            default="delong", section="Method"),
        opt.boolean("show_cutoffs", "Table of cut-offs (sensitivity & specificity)", True, section="Output"),
        opt.integer("max_rows", "Most cut-offs to list per marker", 40, min=5, max=1000, section="Output"),
        opt.boolean("compare", "Compare the ROC curves (DeLong's test)", True, section="Output"),
        opt.boolean("show_plot", "ROC plot", True, section="Output"),
    ])
def roc(ctx: Context) -> Results:
    oname = ctx.var("outcome")
    preds = [p for p in ctx.vars("predictors", min=1) if p != oname]
    if not preds:
        raise AnalysisError("The marker must be a different variable from the outcome.")
    y, pos_label, neg_label = _binary_outcome(ctx, oname, ctx.opt("positive", ""))
    direction = ctx.opt("direction", "auto")
    se_method = ctx.opt("se_method", "delong")
    r = Results()

    t = r.table("Area under the ROC curve", [
        ("name", "Marker", "text"), ("n_pos", f"n {pos_label}", "int"), ("n_neg", f"n {neg_label}", "int"),
        ("auc", "AUC", "num"), ("se", "Std. error", "num"), ("lo", "95% CI lower", "num"),
        ("hi", "95% CI upper", "num"), ("z", "z", "num"), ("p", "P value (AUC = 0.5)", "p"),
        ("s", "Summary", "text"), ("dir", "Positive when", "text")])
    info: dict[str, dict] = {}
    for pn in preds:
        s = ctx.numeric(pn).to_numpy()
        ok = np.isfinite(s) & np.isfinite(y)
        pos, neg = s[ok & (y == 1)], s[ok & (y == 0)]
        if len(pos) < 2 or len(neg) < 2:
            r.warn(f"“{pn}” was skipped: it needs at least two values in each outcome group "
                   f"(found {len(pos)} {pos_label} and {len(neg)} {neg_label}).")
            continue
        if np.ptp(np.concatenate([pos, neg])) == 0:
            r.warn(f"“{pn}” was skipped: all of its values are identical.")
            continue
        auc_hi = float(delong(pos, neg)[0][0])
        higher = auc_hi >= 0.5 if direction == "auto" else direction == "higher"
        sign = 1.0 if higher else -1.0
        aucs, cov = delong(sign * pos, sign * neg)
        auc = float(aucs[0])
        se = math.sqrt(max(cov[0, 0], 0.0)) if se_method == "delong" else hanley_mcneil_se(auc, len(pos), len(neg))
        z = (auc - 0.5) / se if se > 0 else float("nan")
        p = float(2 * stats.norm.sf(abs(z))) if z == z else float("nan")
        info[pn] = dict(pos=pos, neg=neg, higher=higher, auc=auc, se=se, p=p, ok=ok, sign=sign,
                        lo=max(0.0, auc - Z95 * se), hi=min(1.0, auc + Z95 * se))
        t.add_row(name=pn, n_pos=len(pos), n_neg=len(neg), auc=auc, se=se, lo=info[pn]["lo"], hi=info[pn]["hi"],
                  z=z, p=p, s=stars(p), dir="higher values" if higher else "lower values")
    if not info:
        raise AnalysisError("None of the markers could be analysed: each needs at least two values in "
                            "both outcome groups and some variation.")
    t.note(f"Positive group: {pos_label}. AUC = probability that a randomly chosen {pos_label} subject has a more "
           f"extreme value than a randomly chosen {neg_label} subject (ties count ½).")
    t.note(("Standard error by the nonparametric method of DeLong et al. (1988)" if se_method == "delong"
            else "Standard error by Hanley & McNeil (1982)")
           + "; 95% CI = AUC ± 1.96 × SE (truncated to 0–1); z = (AUC − 0.5) / SE, two-tailed P value.")

    # ---- cut-offs
    best_cut = {}
    for pn, d in info.items():
        ct = cutoff_table(d["pos"], d["neg"], d["higher"])
        if len(ct) == 0:
            continue
        ibest = int(ct["youden"].idxmax())
        best_cut[pn] = ct.loc[ibest]
        if not ctx.opt("show_cutoffs", True):
            continue
        cap = int(ctx.opt("max_rows", 40))
        keep = np.arange(len(ct))
        if len(ct) > cap:
            keep = np.unique(np.append(np.round(np.linspace(0, len(ct) - 1, cap - 1)).astype(int), ibest))
        sym = ">" if d["higher"] else "<"
        tc = r.table(f"Sensitivity and specificity: {pn}", [
            ("cut", f"Positive if {pn}", "text"), ("sens", "Sensitivity", "pct"), ("sens_lo", "95% CI lower", "pct"),
            ("sens_hi", "95% CI upper", "pct"), ("spec", "Specificity", "pct"), ("spec_lo", "95% CI lower", "pct"),
            ("spec_hi", "95% CI upper", "pct"), ("lr", "Likelihood ratio", "num"), ("youden", "Youden index", "num"),
            ("mark", "", "text")])
        for i in keep:
            row = ct.loc[i].to_dict()
            cut = row.pop("cutoff")
            tc.add_row(row, cut=f"{sym} {cut:.4g}", cutoff=cut,
                       mark="◄ highest Youden index" if i == ibest else "")
        tc.note("Cut-offs lie midway between adjacent observed values. 95% CIs by the Wilson score method. "
                "Likelihood ratio = sensitivity / (1 − specificity). Youden index = sensitivity + specificity − 1.")
        if len(ct) > cap:
            tc.note(f"{len(keep)} of {len(ct)} possible cut-offs are shown (evenly spaced, plus the one with the "
                    "highest Youden index).")

    # ---- comparison of correlated curves
    cmp_sentence = ""
    names = list(info)
    if len(names) >= 2 and ctx.opt("compare", True):
        ok = np.isfinite(y)
        for pn in names:
            ok &= np.isfinite(ctx.numeric(pn).to_numpy())
        ypos, yneg = ok & (y == 1), ok & (y == 0)
        if ypos.sum() >= 2 and yneg.sum() >= 2:
            mat = np.column_stack([info[pn]["sign"] * ctx.numeric(pn).to_numpy() for pn in names])
            aucs, cov = delong(mat[ypos], mat[yneg])
            tcmp = r.table("Comparison of ROC curves (DeLong's test)", [
                ("pair", "Comparison", "text"), ("a1", "AUC 1", "num"), ("a2", "AUC 2", "num"),
                ("diff", "Difference", "num"), ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num"),
                ("z", "z", "num"), ("p", "P value", "p"), ("s", "Summary", "text")])
            pair_res = []
            for i, j in itertools.combinations(range(len(names)), 2):
                diff = float(aucs[i] - aucs[j])
                var = float(cov[i, i] + cov[j, j] - 2 * cov[i, j])
                se = math.sqrt(var) if var > 0 else float("nan")
                z = diff / se if se == se else float("nan")
                p = float(2 * stats.norm.sf(abs(z))) if z == z else float("nan")
                pair_res.append((names[i], names[j], diff, p))
                tcmp.add_row(pair=f"{names[i]} vs. {names[j]}", a1=aucs[i], a2=aucs[j], diff=diff,
                             lo=diff - Z95 * se, hi=diff + Z95 * se, z=z, p=p, s=stars(p))
            n_used = int(ypos.sum() + yneg.sum())
            tcmp.note(f"DeLong's test for two correlated ROC curves (markers measured on the same subjects), "
                      f"two-tailed; based on the {n_used} subjects with values for every marker.")
            if len(names) > 2:
                L = np.zeros((len(names) - 1, len(names)))
                for i in range(len(names) - 1):
                    L[i, i], L[i, i + 1] = 1, -1
                try:
                    stat = float((L @ aucs) @ np.linalg.solve(L @ cov @ L.T, L @ aucs))
                    p_all = float(stats.chi2.sf(stat, len(names) - 1))
                    tcmp.note(f"Overall test that all {len(names)} AUCs are equal: χ²({len(names) - 1}) = {stat:.4g}, "
                              f"{p_phrase(p_all)}. Pairwise P values are not adjusted for multiple comparisons.")
                except np.linalg.LinAlgError:
                    pass
            a, b, diff, p = pair_res[0]
            if p == p:
                cmp_sentence = (f" The AUCs of {a} and {b} "
                                + ("differ significantly" if p < 0.05 else "do not differ significantly")
                                + f" (difference {diff:.3f}, DeLong {p_phrase(p)}).")
        else:
            r.warn("The ROC curves could not be compared: too few subjects have values for every marker.")

    parts = []
    for pn, d in info.items():
        s = f"{pn}: AUC = {d['auc']:.3f} (95% CI {d['lo']:.3f} to {d['hi']:.3f}, {p_phrase(d['p'])})"
        if pn in best_cut:
            bc = best_cut[pn]
            s += (f"; the cut-off {'>' if d['higher'] else '<'} {bc['cutoff']:.4g} maximises the Youden index "
                  f"(sensitivity {100 * bc['sens']:.1f}%, specificity {100 * bc['spec']:.1f}%)")
        parts.append(s)
    r.summary(f"Discrimination of {pos_label} from {neg_label} — " + ". ".join(parts) + "." + cmp_sentence)

    if ctx.opt("show_plot", True):
        fig, ax = new_figure(4.6, 4.4)
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
        for i, (pn, d) in enumerate(info.items()):
            ct = cutoff_table(d["pos"], d["neg"], d["higher"])
            fpr = np.concatenate([[1.0], 1 - ct["spec"].to_numpy(), [0.0]])
            tpr = np.concatenate([[1.0], ct["sens"].to_numpy(), [0.0]])
            order = np.lexsort((tpr, fpr))
            ax.plot(fpr[order], tpr[order], color=COLORS[i % len(COLORS)], lw=1.7,
                    label=f"{pn} (AUC = {d['auc']:.3f})")
            if pn in best_cut:
                ax.plot([1 - best_cut[pn]["spec"]], [best_cut[pn]["sens"]], "o", ms=5,
                        color=COLORS[i % len(COLORS)], mec="white")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("1 − Specificity (false positive rate)")
        ax.set_ylabel("Sensitivity (true positive rate)")
        ax.legend(loc="lower right")
        r.plot("ROC curve" + ("s" if len(info) > 1 else ""), fig)
    r.graph("Create a ROC graph", "roc", {"outcome": oname, "positive": pos_label, "predictors": names})
    return r


# ============================================================== Bland–Altman
@analysis(
    id="bland_altman", name="Bland–Altman method comparison", category=CATEGORY,
    description="Agreement between two measurement methods: bias and 95% limits of agreement.",
    keywords="bland altman agreement bias limits of agreement method comparison tukey mean difference",
    options=[
        opt.variable("method_a", "Method A", opt.CONTINUOUS),
        opt.variable("method_b", "Method B", opt.CONTINUOUS),
        opt.choice("diff_type", "Difference", [
            ("absolute", "A − B"), ("percent", "Percent of the mean: 100 × (A − B) / mean"),
            ("ratio", "Ratio A / B (analysed on a log scale)")], default="absolute"),
        opt.boolean("prop_bias", "Test for proportional bias", True, section="Output"),
        opt.boolean("show_plot", "Bland–Altman plot", True, section="Output"),
        opt.boolean("show_ci", "Shade 95% CIs of bias and limits", True, section="Output"),
    ])
def bland_altman(ctx: Context) -> Results:
    a_name, b_name = ctx.var("method_a"), ctx.var("method_b")
    if a_name == b_name:
        raise AnalysisError("Choose two different columns for the two methods.")
    d = ctx.frame([a_name, b_name])
    dropped = len(ctx.df) - len(d)
    a, b = d[a_name].to_numpy(dtype=float), d[b_name].to_numpy(dtype=float)
    n = len(a)
    if n < 3:
        raise AnalysisError(f"At least 3 pairs of measurements are needed, but only {n} complete pair(s) were found.")
    kind = ctx.opt("diff_type", "absolute")
    mean = (a + b) / 2
    if kind == "percent":
        if np.any(mean == 0):
            raise AnalysisError("A percent difference cannot be computed because the mean of some pairs is zero.")
        diff = 100 * (a - b) / mean
        what, unit = "Difference, % of mean", "%"
    elif kind == "ratio":
        if np.any(a <= 0) or np.any(b <= 0):
            raise AnalysisError("Ratios are analysed on a log scale, so all measurements must be positive.")
        diff = np.log(a / b)
        what, unit = "Ratio A / B", ""
    else:
        diff = a - b
        what, unit = f"Difference ({a_name} − {b_name})", ""
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    if sd == 0:
        raise AnalysisError("All differences are identical, so limits of agreement cannot be estimated.")
    tcrit = stats.t.ppf(0.975, n - 1)
    se_bias, se_loa = sd / math.sqrt(n), sd * math.sqrt(3 / n)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    back = np.exp if kind == "ratio" else (lambda v: v)
    t_one = stats.ttest_1samp(diff, 0.0)

    r = Results()
    if dropped:
        r.warn(f"{dropped} row(s) with a missing value were ignored.")
    t = r.table("Bland–Altman analysis", [
        ("stat", "", "text"), ("value", "Estimate", "num"), ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")])
    t.add_row(stat="Number of pairs", value=n)
    bias_label = "Bias (geometric mean ratio A / B)" if kind == "ratio" else f"Bias (mean {what[0].lower() + what[1:]})"
    t.add_row(stat=bias_label, value=back(bias), lo=back(bias - tcrit * se_bias), hi=back(bias + tcrit * se_bias))
    t.add_row(stat="SD of bias" + (" (of log ratios)" if kind == "ratio" else ""), value=sd)
    t.add_row(stat="Lower 95% limit of agreement", value=back(lo), lo=back(lo - tcrit * se_loa), hi=back(lo + tcrit * se_loa))
    t.add_row(stat="Upper 95% limit of agreement", value=back(hi), lo=back(hi - tcrit * se_loa), hi=back(hi + tcrit * se_loa))
    t.note("Limits of agreement = bias ± 1.96 × SD. CI of the bias from the t distribution; CIs of the limits use "
           "the approximate standard error √(3·SD²/n) of Bland & Altman (1986).")
    if kind == "ratio":
        t.note("Computed on natural-log ratios and back-transformed, so bias and limits are ratios (1 = no difference).")
    null = "1" if kind == "ratio" else "0"
    t.note(f"One-sample t test of bias = {null}: t({n - 1}) = {t_one.statistic:.4g}, {p_phrase(float(t_one.pvalue))}.")

    summary = (f"On average, {a_name} reads "
               + (f"{back(bias):.4g} times {b_name}" if kind == "ratio"
                  else f"{abs(bias):.4g}{unit} {'higher' if bias > 0 else 'lower'} than {b_name}")
               + f" (95% CI of the bias {back(bias - tcrit * se_bias):.4g} to {back(bias + tcrit * se_bias):.4g}); "
               f"95% of differences are expected to lie between {back(lo):.4g} and {back(hi):.4g}"
               + (" (ratio scale)." if kind == "ratio" else f"{unit}."))

    slope = intercept = float("nan")
    if ctx.opt("prop_bias", True) and np.ptp(mean) > 0:
        reg = stats.linregress(mean, diff)
        slope, intercept = float(reg.slope), float(reg.intercept)
        ts = stats.t.ppf(0.975, n - 2)
        tp = r.table("Proportional bias", [
            ("term", "", "text"), ("est", "Estimate", "num"), ("se", "Std. error", "num"),
            ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num"), ("p", "P value", "p"), ("s", "Summary", "text")])
        tp.add_row(term="Slope", est=slope, se=reg.stderr, lo=slope - ts * reg.stderr, hi=slope + ts * reg.stderr,
                   p=reg.pvalue, s=stars(reg.pvalue))
        tp.add_row(term="Intercept", est=intercept, se=reg.intercept_stderr,
                   lo=intercept - ts * reg.intercept_stderr, hi=intercept + ts * reg.intercept_stderr)
        tp.note("Linear regression of the " + ("log ratio" if kind == "ratio" else "difference")
                + " on the mean of the two methods. A slope different from zero means the disagreement "
                  f"changes with the size of the measurement (r = {reg.rvalue:.3f}).")
        summary += (" There is evidence of proportional bias" if reg.pvalue < 0.05
                    else " There is no significant evidence of proportional bias") + \
                   f" (slope = {slope:.3g}, {p_phrase(float(reg.pvalue))})."
    r.summary(summary)

    if ctx.opt("show_plot", True):
        fig, ax = new_figure(6, 4)
        yv = back(diff)
        ax.scatter(mean, yv, s=20, color=COLORS[0], alpha=0.8, zorder=3)
        for val, se, c, lab in ((bias, se_bias, COLORS[1], "Bias"), (lo, se_loa, "#555555", "−1.96 SD"),
                                (hi, se_loa, "#555555", "+1.96 SD")):
            ax.axhline(back(val), color=c, lw=1.2, ls="-" if lab == "Bias" else "--", zorder=2)
            if ctx.opt("show_ci", True):
                ax.axhspan(back(val - tcrit * se), back(val + tcrit * se), color=c, alpha=0.10, lw=0, zorder=1)
            ax.annotate(f"{lab}: {back(val):.3g}", xy=(1.0, back(val)), xycoords=("axes fraction", "data"),
                        xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=c)
        ax.axhline(1.0 if kind == "ratio" else 0.0, color="#bbbbbb", lw=0.8, zorder=1)
        if kind == "ratio":
            ax.set_yscale("log")
        ax.set_xlabel(f"Mean of {a_name} and {b_name}")
        ax.set_ylabel({"absolute": f"{a_name} − {b_name}", "percent": "Difference (% of mean)",
                       "ratio": f"{a_name} / {b_name}"}[kind])
        r.plot("Bland–Altman plot", fig)
    r.graph("Create a Bland–Altman graph", "bland_altman",
            {"method_a": a_name, "method_b": b_name, "diff_type": "percent" if kind == "percent" else "absolute"})
    return r


# ======================================================================= ICC
def icc_all(data: np.ndarray, alpha: float = 0.05) -> tuple[list[dict], dict]:
    """The six Shrout & Fleiss intraclass correlations of an n × k matrix
    (subjects × raters) with F tests and confidence intervals."""
    n, k = data.shape
    grand = data.mean()
    ss_total = float(((data - grand) ** 2).sum())
    ss_rows = float(k * ((data.mean(axis=1) - grand) ** 2).sum())
    ss_cols = float(n * ((data.mean(axis=0) - grand) ** 2).sum())
    ss_err = max(ss_total - ss_rows - ss_cols, 0.0)
    msb = ss_rows / (n - 1)
    msw = (ss_total - ss_rows) / (n * (k - 1))
    msj = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    anova = {"MSB": msb, "MSW": msw, "MSJ": msj, "MSE": mse, "n": n, "k": k}
    with np.errstate(divide="ignore", invalid="ignore"):
        icc1 = (msb - msw) / (msb + (k - 1) * msw)
        icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
        icc3 = (msb - mse) / (msb + (k - 1) * mse)
        icc1k = (msb - msw) / msb
        icc2k = (msb - mse) / (msb + (msj - mse) / n)
        icc3k = (msb - mse) / msb
        f1 = msb / msw if msw > 0 else float("inf")
        f3 = msb / mse if mse > 0 else float("inf")
    d1n, d1d, d3d = n - 1, n * (k - 1), (n - 1) * (k - 1)
    q = 1 - alpha / 2
    f1l, f1u = f1 / stats.f.ppf(q, d1n, d1d), f1 * stats.f.ppf(q, d1d, d1n)
    f3l, f3u = f3 / stats.f.ppf(q, d1n, d3d), f3 * stats.f.ppf(q, d3d, d1n)
    l1, u1 = (f1l - 1) / (f1l + k - 1), (f1u - 1) / (f1u + k - 1)
    l3, u3 = (f3l - 1) / (f3l + k - 1), (f3u - 1) / (f3u + k - 1)
    # ICC(2,1): Satterthwaite-type df (Shrout & Fleiss 1979; McGraw & Wong 1996)
    l2 = u2 = float("nan")
    if mse > 0 and math.isfinite(icc2):
        fj = msj / mse
        vn = (k - 1) * (n - 1) * (k * icc2 * fj + n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        vd = (n - 1) * k ** 2 * icc2 ** 2 * fj ** 2 + (n * (1 + (k - 1) * icc2) - k * icc2) ** 2
        v = vn / vd if vd > 0 else float("nan")
        if v == v and v > 0:
            fu, fl = stats.f.ppf(q, n - 1, v), stats.f.ppf(q, v, n - 1)
            l2 = n * (msb - fu * mse) / (fu * (k * msj + (k * n - k - n) * mse) + n * msb)
            u2 = n * (fl * msb - mse) / (k * msj + (k * n - k - n) * mse + n * fl * msb)
    p1 = float(stats.f.sf(f1, d1n, d1d))
    p3 = float(stats.f.sf(f3, d1n, d3d))

    def avg(v):
        return k * v / (1 + (k - 1) * v)

    rows = [
        dict(type="ICC(1,1)", desc="One-way random, single measurement", icc=icc1, F=f1, df1=d1n, df2=d1d, p=p1, lo=l1, hi=u1),
        dict(type="ICC(2,1)", desc="Two-way random, absolute agreement, single measurement", icc=icc2, F=f3, df1=d1n, df2=d3d, p=p3, lo=l2, hi=u2),
        dict(type="ICC(3,1)", desc="Two-way mixed, consistency, single measurement", icc=icc3, F=f3, df1=d1n, df2=d3d, p=p3, lo=l3, hi=u3),
        dict(type="ICC(1,k)", desc="One-way random, average of k measurements", icc=icc1k, F=f1, df1=d1n, df2=d1d, p=p1,
             lo=1 - 1 / f1l, hi=1 - 1 / f1u),
        dict(type="ICC(2,k)", desc="Two-way random, absolute agreement, average of k", icc=icc2k, F=f3, df1=d1n, df2=d3d, p=p3,
             lo=avg(l2), hi=avg(u2)),
        dict(type="ICC(3,k)", desc="Two-way mixed, consistency, average of k", icc=icc3k, F=f3, df1=d1n, df2=d3d, p=p3,
             lo=1 - 1 / f3l, hi=1 - 1 / f3u),
    ]
    return rows, anova


def lins_ccc(x: np.ndarray, y: np.ndarray) -> dict:
    """Lin's concordance correlation coefficient with a 95 % CI (Fisher z; Lin 1989, 2000)."""
    n = len(x)
    sx2, sy2 = float(np.var(x)), float(np.var(y))       # 1/n variances, as in Lin (1989)
    sxy = float(np.mean((x - x.mean()) * (y - y.mean())))
    ccc = 2 * sxy / (sx2 + sy2 + (x.mean() - y.mean()) ** 2)
    rho = sxy / math.sqrt(sx2 * sy2)
    cb = ccc / rho if rho != 0 else float("nan")
    lo = hi = float("nan")
    if n > 2 and abs(ccc) < 1 and rho != 0:
        u = (x.mean() - y.mean()) / (sx2 * sy2) ** 0.25
        var_z = ((1 - rho ** 2) * ccc ** 2 / ((1 - ccc ** 2) * rho ** 2)
                 + 2 * ccc ** 3 * (1 - ccc) * u ** 2 / (rho * (1 - ccc ** 2) ** 2)
                 - ccc ** 4 * u ** 4 / (2 * rho ** 2 * (1 - ccc ** 2) ** 2)) / (n - 2)
        if var_z > 0:
            z = math.atanh(ccc)
            lo, hi = math.tanh(z - Z95 * math.sqrt(var_z)), math.tanh(z + Z95 * math.sqrt(var_z))
    return {"ccc": ccc, "lo": lo, "hi": hi, "r": rho, "cb": cb}


def _icc_word(v: float) -> str:
    return "poor" if v < 0.5 else "moderate" if v < 0.75 else "good" if v < 0.9 else "excellent"


@analysis(
    id="icc_agreement", name="Intraclass correlation (ICC)", category=CATEGORY,
    description="Reliability of measurements made by two or more raters or methods on the same subjects.",
    keywords="icc intraclass correlation reliability raters agreement shrout fleiss mcgraw wong lin concordance ccc",
    options=[
        opt.variables("variables", "Raters / measurements (one column each)", opt.CONTINUOUS, min=2),
        opt.choice("focus", "Model to highlight in the summary", [
            ("icc2", "Two-way random, absolute agreement — ICC(2,1)"),
            ("icc3", "Two-way mixed, consistency — ICC(3,1)"),
            ("icc1", "One-way random — ICC(1,1)")], default="icc2"),
        opt.boolean("show_anova", "ANOVA mean squares", False, section="Output"),
        opt.boolean("show_ccc", "Lin's concordance correlation (two columns)", True, section="Output"),
        opt.alpha(section="Output"),
    ])
def icc_agreement(ctx: Context) -> Results:
    cols = list(dict.fromkeys(ctx.vars("variables", min=2)))
    if len(cols) < 2:
        raise AnalysisError("Please choose at least 2 different columns.")
    for c in cols:
        ctx.numeric(c)
    d = pd.DataFrame({c: ctx.numeric(c) for c in cols})
    full = len(d.dropna(how="all"))
    d = d.dropna()
    n, k = d.shape
    if n < 3:
        raise AnalysisError(f"At least 3 subjects (rows) with a value in every column are needed; found {n}.")
    data = d.to_numpy(dtype=float)
    if np.ptp(data) == 0:
        raise AnalysisError("All values are identical, so reliability cannot be estimated.")
    alpha = float(ctx.opt("alpha", 0.05))
    if np.ptp(data.mean(axis=1)) == 0:
        raise AnalysisError("The subjects do not differ from each other (every row has the same mean), "
                            "so an intraclass correlation cannot be estimated.")
    rows, anova = icc_all(data, alpha)
    r = Results()
    if full > n:
        r.warn(f"{full - n} row(s) with a missing value were excluded (complete rows only).")
    level = f"{100 * (1 - alpha):g}%"
    t = r.table("Intraclass correlation coefficients", [
        ("type", "Type", "text"), ("desc", "Model", "text"), ("icc", "ICC", "num"), ("lo", f"{level} CI lower", "num"),
        ("hi", f"{level} CI upper", "num"), ("F", "F", "num"), ("df1", "df1", "int"), ("df2", "df2", "int"),
        ("p", "P value", "p")])
    for row in rows:
        t.add_row(row)
    t.note(f"{n} subjects × {k} raters. Definitions of Shrout & Fleiss (1979); confidence intervals after "
           "Shrout & Fleiss (1979) and McGraw & Wong (1996). The F test examines the null hypothesis ICC = 0.")
    t.note("Use ICC(2,·) when the raters are a random sample of possible raters and systematic differences "
           "between them matter (absolute agreement); ICC(3,·) when these are the only raters of interest "
           "(consistency); ICC(1,·) when each subject is rated by different raters.")
    if ctx.opt("show_anova", False):
        ta = r.table("Mean squares (two-way ANOVA)", [("src", "Source", "text"), ("ms", "Mean square", "num"), ("df", "df", "int")])
        ta.add_row(src="Between subjects (MSB)", ms=anova["MSB"], df=n - 1)
        ta.add_row(src="Within subjects (MSW)", ms=anova["MSW"], df=n * (k - 1))
        ta.add_row(src="Between raters (MSJ)", ms=anova["MSJ"], df=k - 1)
        ta.add_row(src="Residual (MSE)", ms=anova["MSE"], df=(n - 1) * (k - 1))
    focus = {"icc1": 0, "icc2": 1, "icc3": 2}[ctx.opt("focus", "icc2")]
    f = rows[focus]
    summary = (f"{f['type']} = {f['icc']:.3f} ({level} CI {f['lo']:.3f} to {f['hi']:.3f}), which is conventionally "
               f"read as {_icc_word(f['icc'])} reliability of a single measurement (Koo & Li, 2016). "
               f"For the average of the {k} measurements, {rows[focus + 3]['type']} = {rows[focus + 3]['icc']:.3f}.")
    if k == 2 and ctx.opt("show_ccc", True):
        c = lins_ccc(data[:, 0], data[:, 1])
        tc = r.table("Lin's concordance correlation coefficient", [
            ("ccc", "CCC", "num"), ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num"),
            ("r", "Pearson r (precision)", "num"), ("cb", "Bias correction factor Cb (accuracy)", "num")])
        tc.add_row(c)
        tc.note("CCC = r × Cb measures how closely the pairs fall on the line of identity (Lin, 1989). "
                "95% CI from the Fisher z-transformation.")
        summary += f" Lin's concordance correlation is {c['ccc']:.3f} (95% CI {c['lo']:.3f} to {c['hi']:.3f})."
    r.summary(summary)
    return r


# ===================================================================== kappa
def _kappa_word(v: float) -> str:
    if v < 0:
        return "less than chance"
    return ("slight" if v <= 0.2 else "fair" if v <= 0.4 else "moderate" if v <= 0.6
            else "substantial" if v <= 0.8 else "almost perfect")


@analysis(
    id="cohen_kappa", name="Cohen's kappa (agreement between two raters)", category=CATEGORY,
    description="Chance-corrected agreement between two raters who classify the same subjects.",
    keywords="kappa cohen agreement raters weighted kappa inter-rater reliability categorical",
    options=[
        opt.variable("rater_a", "Rater A", opt.ANY),
        opt.variable("rater_b", "Rater B", opt.ANY),
        opt.choice("weights", "Weighting", [
            ("none", "Unweighted"), ("linear", "Linear weights (ordered categories)"),
            ("quadratic", "Quadratic weights (ordered categories)")], default="none"),
        opt.boolean("show_table", "Agreement table", True, section="Output"),
    ])
def cohen_kappa(ctx: Context) -> Results:
    from statsmodels.stats.inter_rater import cohens_kappa
    a_name, b_name = ctx.var("rater_a"), ctx.var("rater_b")
    if a_name == b_name:
        raise AnalysisError("Choose two different columns for the two raters.")
    fa, fb = ctx.factor(a_name), ctx.factor(b_name)
    la, lb = [str(c) for c in fa.cat.categories], [str(c) for c in fb.cat.categories]
    levels = la + [c for c in lb if c not in la]
    if all(_is_num(c) for c in levels):
        levels = sorted(levels, key=float)
    va = fa.astype(object).map(lambda v: None if pd.isna(v) else str(v))
    vb = fb.astype(object).map(lambda v: None if pd.isna(v) else str(v))
    ok = va.notna() & vb.notna()
    n = int(ok.sum())
    if n < 2:
        raise AnalysisError("At least two subjects rated by both raters are needed.")
    levels = [c for c in levels if ((va[ok] == c) | (vb[ok] == c)).any()]
    q = len(levels)
    if q < 2:
        raise AnalysisError("Both raters used a single category for every subject, so kappa is undefined.")
    if q > 30:
        raise AnalysisError(f"The ratings have {q} different categories; kappa is meant for a small number "
                            "of categories (at most 30 here).")
    tab = pd.crosstab(pd.Categorical(va[ok], categories=levels), pd.Categorical(vb[ok], categories=levels),
                      dropna=False).reindex(index=levels, columns=levels, fill_value=0).to_numpy()
    r = Results()
    if n < len(va):
        r.warn(f"{len(va) - n} row(s) with a missing rating were ignored.")
    wt = ctx.opt("weights", "none")
    kinds = [("none", "Unweighted kappa")]
    if wt != "none":
        kinds.append((wt, f"Weighted kappa ({wt} weights)"))
    t = r.table("Cohen's kappa", [
        ("kind", "", "text"), ("kappa", "Kappa", "num"), ("se", "Std. error", "num"), ("lo", "95% CI lower", "num"),
        ("hi", "95% CI upper", "num"), ("z", "z", "num"), ("p", "P value", "p"), ("s", "Summary", "text")])
    main = None
    for kind, label in kinds:
        with np.errstate(all="ignore"):
            res = cohens_kappa(tab, wt=None if kind == "none" else kind)
        kappa = float(res.kappa)
        if kappa != kappa:
            raise AnalysisError("Kappa is undefined for these ratings (each rater used one category only).")
        se, se0 = float(np.sqrt(res.var_kappa)), float(np.sqrt(res.var_kappa0))
        z = kappa / se0 if se0 > 0 else float("nan")
        p = float(2 * stats.norm.sf(abs(z))) if z == z else float("nan")
        main = dict(kind=label, kappa=kappa, se=se, lo=max(-1.0, kappa - Z95 * se), hi=min(1.0, kappa + Z95 * se),
                    z=z, p=p, s=stars(p))
        t.add_row(main)
    agree = float(np.trace(tab) / n)
    pe = float((tab.sum(axis=1) / n) @ (tab.sum(axis=0) / n))
    t.note(f"{n} subjects, {q} categories. Observed agreement {100 * agree:.1f}%; agreement expected by chance "
           f"{100 * pe:.1f}%.")
    t.note("Standard error and 95% CI from the large-sample variance of Fleiss, Cohen & Everitt (1969); "
           "z and P test the null hypothesis kappa = 0 using the standard error under that hypothesis.")
    if wt != "none":
        t.note("Weighted kappa gives partial credit for near-misses and uses the category order: "
               + " < ".join(levels) + ".")
    if ctx.opt("show_table", True):
        ta = r.table(f"Agreement table (rows: {a_name}; columns: {b_name})",
                     [("row", "", "text")] + [(f"c{j}", lev, "int") for j, lev in enumerate(levels)] + [("total", "Total", "int")])
        for i, lev in enumerate(levels):
            ta.add_row(row=lev, total=int(tab[i].sum()), **{f"c{j}": int(tab[i, j]) for j in range(q)})
        ta.add_row(row="Total", total=n, **{f"c{j}": int(tab[:, j].sum()) for j in range(q)})
    assert main is not None
    r.summary(f"The raters agreed on {100 * agree:.1f}% of {n} subjects. {main['kind']} = {main['kappa']:.3f} "
              f"(95% CI {main['lo']:.3f} to {main['hi']:.3f}), conventionally described as "
              f"{_kappa_word(main['kappa'])} agreement beyond chance (Landis & Koch, 1977); "
              f"{p_phrase(main['p'])} for the null hypothesis of chance agreement.")
    return r


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
