"""Compare two groups: unpaired / paired t tests, Mann–Whitney, Wilcoxon
matched pairs, two-sample Kolmogorov–Smirnov and multiple t tests.

Convention (as in GraphPad Prism): differences are *second group − first
group*, so with columns "Control", "Treated" a positive difference means the
treated values are higher.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core import analysis, opt, Context, Results, AnalysisError, new_figure, COLORS, stars, fmt_p
from .descriptive import (p_text, conf_opt, conf_level, pct, fnum, layout_options, tails_label, describe,
                          is_constant, signed_rank, strip_axis, value_label, _t_pvalue)

CAT = "Compare two groups"

TAILS2 = [("two-sided", "Two-tailed"),
          ("greater", "One-tailed (second group is greater)"),
          ("less", "One-tailed (second group is smaller)")]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def two_groups(ctx: Context, min_n: int = 2) -> tuple[str, np.ndarray, str, np.ndarray]:
    groups = ctx.groups(min_groups=1)
    if len(groups) != 2:
        raise AnalysisError(f"This test compares exactly two groups, but {len(groups)} "
                            f"{'was' if len(groups) == 1 else 'were'} found with data. "
                            "Select two columns, or a grouping variable with two levels.")
    for name, x in groups.items():
        if len(x) < min_n:
            raise AnalysisError(f"Group “{name}” has only {len(x)} value{'s' if len(x) != 1 else ''}; "
                                f"at least {min_n} are needed.")
    (na, a), (nb, b) = groups.items()
    return na, a, nb, b


def pairs(ctx: Context, min_n: int = 2) -> tuple[str, np.ndarray, str, np.ndarray, int]:
    """Matched values of exactly two conditions → (nameA, a, nameB, b, n_dropped)."""
    full = ctx.paired(complete=False)
    if full.shape[1] != 2:
        raise AnalysisError(f"This test compares exactly two paired conditions, but {full.shape[1]} "
                            f"{'was' if full.shape[1] == 1 else 'were'} found with data. "
                            "Select two columns, or a grouping variable with two levels.")
    wide = full.dropna()
    dropped = len(full) - len(wide)
    if len(wide) < min_n:
        raise AnalysisError(f"Only {len(wide)} complete pair{'s' if len(wide) != 1 else ''} found; at least {min_n} "
                            "are needed. Each subject needs a value in both conditions.")
    na, nb = wide.columns
    return na, wide[na].to_numpy(float), nb, wide[nb].to_numpy(float), dropped


def nct_ci(t_obs: float, df: float, conf: float) -> tuple[float, float]:
    """CI of the non-centrality parameter of a t statistic (pivot method)."""
    if not np.isfinite(t_obs) or df < 1:
        return math.nan, math.nan
    a = (1 - conf) / 2

    def solve(target):
        f = lambda ncp: stats.nct.cdf(t_obs, df, ncp) - target  # noqa: E731
        lo, hi = t_obs - 5, t_obs + 5
        for _ in range(60):
            if f(lo) > 0:
                break
            lo -= 5
        for _ in range(60):
            if f(hi) < 0:
                break
            hi += 5
        return optimize.brentq(f, lo, hi, xtol=1e-10)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = solve(1 - a), solve(a)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return math.nan, math.nan
        return float(lo), float(hi)
    except Exception:
        return math.nan, math.nan


def descriptives_table(r: Results, named: dict[str, np.ndarray], conf: float, title: str = "Descriptive statistics"):
    cl = pct(conf)
    t = r.table(title, [("group", "Group", "text"), ("n", "n", "int"), ("mean", "Mean", "num"),
                        ("sd", "SD", "num"), ("sem", "SEM", "num"),
                        ("ci_lo", f"{cl} CI lower", "num"), ("ci_hi", f"{cl} CI upper", "num"),
                        ("median", "Median", "num"), ("min", "Minimum", "num"), ("max", "Maximum", "num")])
    for name, x in named.items():
        t.add_row(group=name, **describe(x, conf))
    return t


def shapiro_table(r: Results, named: dict[str, np.ndarray], alpha: float, title: str = "Normality (Shapiro–Wilk)"):
    t = r.table(title, [("group", "Data", "text"), ("n", "n", "int"), ("w", "W", "num"), ("p", "P value", "p"),
                        ("passed", f"Passed (α = {alpha:g})?", "text")])
    for name, x in named.items():
        if len(x) < 3 or is_constant(x) or len(x) > 5000:
            t.add_row(group=name, n=len(x), passed="not tested")
            continue
        w, p = stats.shapiro(x)
        t.add_row(group=name, n=len(x), w=float(w), p=float(p), passed="Yes" if p >= alpha else "No")
    t.note("A small P value suggests the values are not sampled from a Gaussian distribution; consider a "
           "nonparametric test or a transformation. With small samples the test has little power.")
    return t


def tail_note(alt: str, na: str, nb: str) -> str:
    if alt == "greater":
        return f"One-tailed P value (alternative: {nb} > {na})."
    if alt == "less":
        return f"One-tailed P value (alternative: {nb} < {na})."
    return "Two-tailed P value."


@lru_cache(maxsize=32)
def _mwu_cdf(m: int, n: int) -> np.ndarray:
    """Exact null CDF of the Mann–Whitney U statistic (no ties)."""
    rows = [[np.ones(1) for _ in range(n + 1)]]
    for i in range(1, m + 1):
        row = [np.ones(1)]
        for j in range(1, n + 1):
            f = np.zeros(i * j + 1)
            up = rows[i - 1][j]                 # largest value is in sample 1 → U gains j
            f[j:j + len(up)] += i / (i + j) * up
            left = row[j - 1]                   # largest value is in sample 2
            f[:len(left)] += j / (i + j) * left
            row.append(f)
        rows.append(row)
    return np.cumsum(rows[m][n])


def hodges_lehmann(a: np.ndarray, b: np.ndarray, conf: float, exact: bool) -> dict:
    """Hodges–Lehmann estimate of the shift (b − a) and its CI."""
    diffs = np.sort((b[None, :] - a[:, None]).ravel())
    n1, n2 = len(a), len(b)
    N = n1 * n2
    out = {"est": float(np.median(diffs)), "lo": math.nan, "hi": math.nan, "conf": math.nan}
    alpha = 1 - conf
    if exact:
        cdf = _mwu_cdf(min(n1, n2), max(n1, n2))
        qu = int(np.searchsorted(cdf, alpha / 2 - 1e-12, side="left"))
        if qu == 0:
            qu = 1
        achieved = 1 - 2 * cdf[qu - 1]
    else:
        z = stats.norm.ppf(1 - alpha / 2)
        qu = max(int(math.floor(N / 2 - z * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0))), 1)
        achieved = conf
    ql = N - qu
    if achieved > 0 and qu - 1 < N and 0 <= ql < N and qu - 1 <= ql:
        out.update(lo=float(diffs[qu - 1]), hi=float(diffs[ql]), conf=float(achieved))
    return out


def mann_whitney_core(a: np.ndarray, b: np.ndarray, alt: str = "two-sided") -> dict:
    """Mann–Whitney test. ``alt`` refers to the second sample (b) vs the first."""
    n1, n2 = len(a), len(b)
    ranks = stats.rankdata(np.concatenate([a, b]))
    r1, r2 = float(ranks[:n1].sum()), float(ranks[n1:].sum())
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    ties = len(np.unique(np.concatenate([a, b]))) < n1 + n2
    exact = (not ties) and n1 <= 50 and n2 <= 50
    res = stats.mannwhitneyu(b, a, alternative=alt, method="exact" if exact else "asymptotic",
                             use_continuity=True)
    return {"r1": r1, "r2": r2, "u1": float(u1), "u2": float(u2), "u": float(min(u1, u2)),
            "p": float(res.pvalue), "exact": exact, "ties": ties,
            "rb": float((u2 - u1) / (n1 * n2))}


# --------------------------------------------------------------------------
# Unpaired t test
# --------------------------------------------------------------------------

def unpaired_t(a: np.ndarray, b: np.ndarray, welch: bool, alt: str, conf: float) -> dict:
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    diff = m2 - m1
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if welch:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if se > 0 else math.nan
    else:
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    t = diff / se if se > 0 else math.nan
    p = _t_pvalue(t, df, alt) if se > 0 else math.nan
    half = stats.t.ppf(0.5 + conf / 2, df) * se if se > 0 else math.nan
    sp = math.sqrt(sp2)
    d = diff / sp if sp > 0 else math.nan
    dfp = n1 + n2 - 2
    j = 1 - 3 / (4 * dfp - 1)
    scale = math.sqrt(n1 * n2 / (n1 + n2))
    ncp_lo, ncp_hi = nct_ci(d * scale, dfp, conf) if sp > 0 else (math.nan, math.nan)
    return {"diff": float(diff), "se": float(se), "t": float(t), "df": float(df), "p": float(p),
            "lo": float(diff - half), "hi": float(diff + half), "d": float(d), "g": float(d * j),
            "d_lo": ncp_lo / scale, "d_hi": ncp_hi / scale,
            "r2": float(t * t / (t * t + df)) if se > 0 else math.nan}


@analysis(
    id="ttest_unpaired", name="Unpaired t test", category=CAT,
    description="Compare the means of two independent groups (Student's or Welch's t test).",
    keywords="student welch two sample independent t-test means",
    options=[
        *opt.group_layout(),
        opt.choice("test", "Test", [("welch", "Welch's t test (unequal variances)"),
                                    ("student", "Student's t test (equal variances)")], default="welch"),
        opt.choice("tails", "P value", TAILS2, default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("f_test", "F test to compare variances", True, section="Assumption checks"),
        opt.boolean("normality", "Normality test (Shapiro–Wilk)", False, section="Assumption checks"),
        opt.boolean("show_descriptives", "Descriptive statistics", True, section="Output"),
        opt.boolean("show_plot", "Estimation plot", True, section="Output"),
    ])
def ttest_unpaired(ctx: Context) -> Results:
    na, a, nb, b = two_groups(ctx, min_n=2)
    welch = ctx.opt("test", "welch") == "welch"
    alt = ctx.opt("tails", "two-sided")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    if is_constant(a) and is_constant(b):
        raise AnalysisError("Both groups have no variability (all values within each group are identical), "
                            "so a t test cannot be computed.")
    res = unpaired_t(a, b, welch, alt, conf)
    test_name = "Welch's t test" if welch else "Student's t test"
    r = Results()
    t = r.table(f"Unpaired t test ({'Welch' if welch else 'Student'})",
                [("t", "t", "num"), ("df", "df", "num"), ("p", "P value", "p"), ("stars", "P value summary", "text"),
                 ("sig", f"Significant (α = {alpha:g})?", "text"), ("tails", "Tails", "text"),
                 ("r2", "R² (eta squared)", "num")])
    t.add_row(t=res["t"], df=res["df"], p=res["p"], stars=stars(res["p"]),
              sig="Yes" if res["p"] < alpha else "No", tails=tails_label(alt), r2=res["r2"])
    t.note(("Welch's unequal-variances t test (Welch–Satterthwaite degrees of freedom)." if welch else
            "Student's t test, assuming both populations have the same SD (pooled variance).")
           + " " + tail_note(alt, na, nb))
    t.note("Assumes independent observations sampled from Gaussian populations.")
    e = r.table("How big is the difference?",
                [("label", "Difference", "text"), ("diff", "Difference between means", "num"),
                 ("se", "SE of difference", "num"), ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num"),
                 ("d", "Cohen's d", "num"), ("d_lo", f"d {cl} CI lower", "num"), ("d_hi", f"d {cl} CI upper", "num"),
                 ("g", "Hedges' g", "num")])
    e.add_row(label=f"{nb} − {na}", **{k: res[k] for k in ("diff", "se", "lo", "hi", "d", "d_lo", "d_hi", "g")})
    e.note("Cohen's d = difference / pooled SD; Hedges' g applies the small-sample correction. "
           "The CI of d is based on the noncentral t distribution (assumes equal variances).")
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    f_p = math.nan
    if ctx.opt("f_test"):
        ft = r.table("F test to compare variances",
                     [("f", "F", "num"), ("dfn", "DFn", "int"), ("dfd", "DFd", "int"), ("p", "P value", "p"),
                      ("stars", "P value summary", "text"), ("sig", "Variances differ significantly?", "text")])
        if min(v1, v2) > 0:
            (big, nbig), (small, nsmall) = sorted([(v1, len(a)), (v2, len(b))], reverse=True)
            f = big / small
            f_p = float(min(1.0, 2 * stats.f.sf(f, nbig - 1, nsmall - 1)))
            ft.add_row(f=f, dfn=nbig - 1, dfd=nsmall - 1, p=f_p, stars=stars(f_p), sig="Yes" if f_p < alpha else "No")
            ft.note("F = larger variance / smaller variance; two-tailed P value. This test is sensitive to "
                    "departures from normality.")
        else:
            ft.add_row(sig="not tested")
            ft.note("One group has zero variance, so the F test cannot be computed.")
    if ctx.opt("show_descriptives"):
        descriptives_table(r, {na: a, nb: b}, conf)
    if ctx.opt("normality"):
        shapiro_table(r, {na: a, nb: b}, alpha)
    if not welch and np.isfinite(f_p) and f_p < alpha:
        r.warn("The variances of the two groups differ significantly (F test). "
               "Welch's t test does not assume equal variances and may be more appropriate.")

    direction = "higher" if res["diff"] > 0 else "lower"
    if res["p"] < alpha:
        lead = (f"The difference between the means of {na} and {nb} is statistically significant "
                f"({p_text(res['p'])}; {test_name}{', one-tailed' if alt != 'two-sided' else ''}).")
    else:
        lead = (f"The difference between the means of {na} and {nb} is not statistically significant "
                f"({p_text(res['p'])}; {test_name}{', one-tailed' if alt != 'two-sided' else ''}); "
                "the data do not provide convincing evidence of a difference.")
    r.summary(f"{lead} The mean of {nb} is {fnum(abs(res['diff']))} {direction} than that of {na} "
              f"({cl} CI of the difference {fnum(res['lo'])} to {fnum(res['hi'])}; Cohen's d = {fnum(res['d'], 3)}).")

    if ctx.opt("show_plot"):
        fig, axes = new_figure(6.2, 3.6, 1, 2, gridspec_kw={"width_ratios": [2, 1]})
        ax = axes[0]
        groups = {na: a, nb: b}
        strip_axis(ax, groups)
        for i, x in enumerate(groups.values()):
            ax.hlines(x.mean(), i - 0.25, i + 0.25, color="#222222", lw=1.6, zorder=4)
        ax.set_ylabel(value_label(ctx))
        ax.set_title("Data and means")
        ax2 = axes[1]
        ax2.errorbar([0], [res["diff"]], yerr=[[res["diff"] - res["lo"]], [res["hi"] - res["diff"]]],
                     fmt="o", color=COLORS[1], capsize=4, lw=1.5, ms=6)
        ax2.axhline(0, color="#888888", ls="--", lw=1)
        ax2.set_xlim(-1, 1)
        ax2.set_xticks([0])
        ax2.set_xticklabels([f"{nb} − {na}"])
        ax2.set_title(f"Difference ({cl} CI)")
        r.plot("Estimation plot", fig)
    r.graph("Create a column graph of these data", "column_scatter", layout_options(ctx))
    return r


# --------------------------------------------------------------------------
# Mann–Whitney
# --------------------------------------------------------------------------

@analysis(
    id="mann_whitney", name="Mann–Whitney test", category=CAT,
    description="Nonparametric comparison of two independent groups based on ranks.",
    keywords="wilcoxon rank sum nonparametric u test medians",
    options=[
        *opt.group_layout(),
        opt.choice("tails", "P value", TAILS2, default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_descriptives", "Group medians and ranks", True, section="Output"),
        opt.boolean("show_plot", "Plot with medians", True, section="Output"),
    ])
def mann_whitney(ctx: Context) -> Results:
    na, a, nb, b = two_groups(ctx, min_n=1)
    if len(a) + len(b) < 4:
        raise AnalysisError("The Mann–Whitney test needs at least 4 values in total.")
    alt = ctx.opt("tails", "two-sided")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    if is_constant(np.concatenate([a, b])):
        raise AnalysisError("All values in both groups are identical, so the ranks carry no information.")
    mw = mann_whitney_core(a, b, alt)
    hl = hodges_lehmann(a, b, conf, mw["exact"])
    r = Results()
    t = r.table("Mann–Whitney test",
                [("u", "Mann–Whitney U", "num"), ("p", "P value", "p"), ("stars", "P value summary", "text"),
                 ("sig", f"Significant (α = {alpha:g})?", "text"), ("method", "P value method", "text"),
                 ("tails", "Tails", "text"), ("rb", "Rank-biserial r", "num"), ("ps", f"P({nb} > {na})", "num")])
    t.add_row(u=mw["u"], p=mw["p"], stars=stars(mw["p"]), sig="Yes" if mw["p"] < alpha else "No",
              method="Exact" if mw["exact"] else "Normal approximation", tails=tails_label(alt),
              rb=mw["rb"], ps=mw["u2"] / (len(a) * len(b)))
    t.note("U is the smaller of the two U values. " + tail_note(alt, na, nb))
    if mw["exact"]:
        t.note("Exact P value (no tied values, both n ≤ 50).")
    else:
        t.note("P value from the normal approximation with continuity correction"
               + (", corrected for ties" if mw["ties"] else "")
               + (" (an exact P value is not computed when values are tied)." if mw["ties"] else " (n > 50)."))
    t.note(f"Rank-biserial r = (U₂ − U₁)/(n₁n₂), from −1 to 1; positive when {nb} tends to be higher. "
           f"P({nb} > {na}) is the probability that a random value from {nb} exceeds one from {na} (ties count half).")
    e = r.table("Difference between medians",
                [("label", "Difference", "text"), ("med_diff", "Difference of medians", "num"),
                 ("hl", "Hodges–Lehmann estimate", "num"), ("lo", "CI lower", "num"), ("hi", "CI upper", "num"),
                 ("actual", "Actual confidence level", "pct")])
    e.add_row(label=f"{nb} − {na}", med_diff=float(np.median(b) - np.median(a)), hl=hl["est"],
              lo=hl["lo"], hi=hl["hi"], actual=hl["conf"])
    e.note("The Hodges–Lehmann estimate is the median of all pairwise differences; its confidence interval "
           + (f"is exact, so its actual confidence level is at least the requested {cl}."
              if mw["exact"] else f"uses the normal approximation ({cl}).")
           + " It estimates a shift in location, assuming the two distributions have the same shape.")
    if ctx.opt("show_descriptives"):
        d = r.table("Descriptive statistics",
                    [("group", "Group", "text"), ("n", "n", "int"), ("median", "Median", "num"),
                     ("q1", "25% percentile", "num"), ("q3", "75% percentile", "num"),
                     ("rank_sum", "Sum of ranks", "num"), ("mean_rank", "Mean rank", "num"), ("u", "U", "num")])
        from .descriptive import quantile
        for name, x, rs, u in ((na, a, mw["r1"], mw["u1"]), (nb, b, mw["r2"], mw["u2"])):
            d.add_row(group=name, n=len(x), median=float(np.median(x)), q1=quantile(x, 0.25), q3=quantile(x, 0.75),
                      rank_sum=rs, mean_rank=rs / len(x), u=u)
    sig = mw["p"] < alpha
    r.summary(
        (f"The distributions of {na} and {nb} differ significantly" if sig else
         f"The difference between {na} and {nb} is not statistically significant")
        + f" ({p_text(mw['p'])}; Mann–Whitney test, {'exact' if mw['exact'] else 'normal approximation'}"
        + (", one-tailed" if alt != "two-sided" else "") + "). "
        + f"Medians: {na} {fnum(float(np.median(a)))}, {nb} {fnum(float(np.median(b)))}; "
        + f"Hodges–Lehmann difference {fnum(hl['est'])}"
        + (f" (CI {fnum(hl['lo'])} to {fnum(hl['hi'])})." if np.isfinite(hl["lo"]) else "."))
    if ctx.opt("show_plot"):
        fig, ax = new_figure(3.8, 3.6)
        groups = {na: a, nb: b}
        strip_axis(ax, groups)
        for i, x in enumerate(groups.values()):
            ax.hlines(np.median(x), i - 0.25, i + 0.25, color="#222222", lw=1.6, zorder=4)
        ax.set_ylabel(value_label(ctx))
        r.plot("Values with medians", fig)
    r.graph("Create a column graph of these data", "column_scatter", layout_options(ctx))
    return r


# --------------------------------------------------------------------------
# Paired t test
# --------------------------------------------------------------------------

def _before_after_plot(r: Results, na, a, nb, b, title="Before–after plot", ylabel="Value", log=False):
    fig, ax = new_figure(3.6, 3.6)
    for ya, yb in zip(a, b):
        ax.plot([0, 1], [ya, yb], color="#9AA5B1", lw=0.9, alpha=0.8, zorder=2)
    ax.scatter(np.zeros(len(a)), a, s=18, color=COLORS[0], zorder=3)
    ax.scatter(np.ones(len(b)), b, s=18, color=COLORS[1], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([na, nb])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    r.plot(title, fig)


def _pairing_note(dropped: int) -> str | None:
    if dropped:
        return (f"{dropped} subject{'s' if dropped != 1 else ''} with a missing value in one condition "
                f"{'were' if dropped != 1 else 'was'} excluded.")
    return None


@analysis(
    id="ttest_paired", name="Paired t test", category=CAT,
    description="Compare two matched measurements (before/after, matched pairs) with a paired or ratio paired t test.",
    keywords="paired dependent matched repeated before after ratio t-test",
    options=[
        *opt.group_layout(paired=True),
        opt.choice("test", "Test", [("paired", "Paired t test (differences)"),
                                    ("ratio", "Ratio paired t test (log ratios)")], default="paired"),
        opt.choice("tails", "P value", TAILS2, default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("normality", "Normality of differences (Shapiro–Wilk)", False, section="Assumption checks"),
        opt.boolean("show_descriptives", "Descriptive statistics", True, section="Output"),
        opt.boolean("show_plot", "Before–after plot", True, section="Output"),
    ])
def ttest_paired(ctx: Context) -> Results:
    na, a, nb, b, dropped = pairs(ctx, min_n=2)
    ratio = ctx.opt("test", "paired") == "ratio"
    alt = ctx.opt("tails", "two-sided")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    if ratio:
        if np.any(a <= 0) or np.any(b <= 0):
            raise AnalysisError("The ratio paired t test needs all values to be greater than zero "
                                "(it analyses the logarithm of the ratios).")
        d = np.log10(b / a)
    else:
        d = b - a
    n = len(d)
    if is_constant(d):
        raise AnalysisError("All pairs have exactly the same " + ("ratio" if ratio else "difference")
                            + ", so there is no variability and a paired t test cannot be computed.")
    mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    sem_d = sd_d / math.sqrt(n)
    df = n - 1
    tv = mean_d / sem_d
    p = _t_pvalue(tv, df, alt)
    half = stats.t.ppf(0.5 + conf / 2, df) * sem_d
    lo, hi = mean_d - half, mean_d + half
    dz = mean_d / sd_d
    ncp_lo, ncp_hi = nct_ci(tv, df, conf)
    test_name = "ratio paired t test" if ratio else "paired t test"

    r = Results()
    t = r.table("Ratio paired t test" if ratio else "Paired t test",
                [("t", "t", "num"), ("df", "df", "int"), ("p", "P value", "p"), ("stars", "P value summary", "text"),
                 ("sig", f"Significant (α = {alpha:g})?", "text"), ("tails", "Tails", "text"),
                 ("n", "Number of pairs", "int")])
    t.add_row(t=tv, df=df, p=p, stars=stars(p), sig="Yes" if p < alpha else "No", tails=tails_label(alt), n=n)
    t.note(tail_note(alt, na, nb) + (" The test is computed on log10(ratio) and asks whether the mean log ratio "
                                     "differs from 0 (ratio of 1)." if ratio else ""))
    t.note("Assumes the " + ("log ratios" if ratio else "differences") + " are sampled from a Gaussian distribution.")
    for note in filter(None, [_pairing_note(dropped)]):
        t.note(note)
    if ratio:
        e = r.table("How big is the difference?",
                    [("label", "Ratio", "text"), ("gm", "Geometric mean of ratios", "num"),
                     ("gm_lo", f"{cl} CI lower", "num"), ("gm_hi", f"{cl} CI upper", "num"),
                     ("mean", "Mean of log10(ratio)", "num"), ("sd", "SD of log10(ratio)", "num"),
                     ("sem", "SEM of log10(ratio)", "num"), ("dz", "Cohen's dz", "num")])
        e.add_row(label=f"{nb} / {na}", gm=10 ** mean_d, gm_lo=10 ** lo, gm_hi=10 ** hi,
                  mean=mean_d, sd=sd_d, sem=sem_d, dz=dz)
        e.note("Cohen's dz = mean / SD of the log ratios.")
    else:
        e = r.table("How big is the difference?",
                    [("label", "Difference", "text"), ("mean", "Mean of differences", "num"),
                     ("sd", "SD of differences", "num"), ("sem", "SEM of differences", "num"),
                     ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num"),
                     ("dz", "Cohen's dz", "num"), ("dz_lo", f"dz {cl} CI lower", "num"),
                     ("dz_hi", f"dz {cl} CI upper", "num"), ("r2", "R² (partial eta squared)", "num")])
        e.add_row(label=f"{nb} − {na}", mean=mean_d, sd=sd_d, sem=sem_d, lo=lo, hi=hi, dz=dz,
                  dz_lo=ncp_lo / math.sqrt(n), dz_hi=ncp_hi / math.sqrt(n), r2=tv * tv / (tv * tv + df))
        e.note("Cohen's dz = mean difference / SD of the differences; its CI is based on the noncentral t distribution.")
    # effectiveness of pairing
    pr = r.table("How effective was the pairing?",
                 [("r", "Correlation coefficient (r)", "num"), ("p", "P value (one-tailed)", "p"),
                  ("stars", "P value summary", "text"), ("effective", "Was the pairing significantly effective?", "text")])
    xa, xb = (np.log10(a), np.log10(b)) if ratio else (a, b)
    if n >= 3 and not is_constant(xa) and not is_constant(xb):
        rr = float(np.corrcoef(xa, xb)[0, 1])
        tr = rr * math.sqrt((n - 2) / max(1e-300, 1 - rr * rr))
        p_r = float(stats.t.sf(tr, n - 2))
        pr.add_row(r=rr, p=p_r, stars=stars(p_r), effective="Yes" if p_r < alpha else "No")
    else:
        pr.add_row(effective="not tested")
    pr.note("Pearson correlation between the two conditions" + (" (log values)" if ratio else "")
            + "; one-tailed P value for a positive correlation. Pairing helps when the paired values are positively correlated.")
    if ctx.opt("show_descriptives"):
        descriptives_table(r, {na: a, nb: b, ("log10 ratio" if ratio else "Difference") + f" ({nb} {'/' if ratio else '−'} {na})": d}, conf)
    if ctx.opt("normality"):
        shapiro_table(r, {("Log ratios" if ratio else "Differences"): d}, alpha,
                      title="Normality of the " + ("log ratios" if ratio else "differences") + " (Shapiro–Wilk)")
    tail_txt = ", one-tailed" if alt != "two-sided" else ""
    if ratio:
        gm = 10 ** mean_d
        r.summary(
            (f"The ratio {nb} / {na} differs significantly from 1" if p < alpha else
             f"The ratio {nb} / {na} is not significantly different from 1")
            + f" ({p_text(p)}; {test_name}{tail_txt}). On average {nb} is {fnum(gm)} times {na} "
              f"(geometric mean ratio; {cl} CI {fnum(10 ** lo)} to {fnum(10 ** hi)}).")
    else:
        r.summary(
            (f"The mean difference between {nb} and {na} is statistically significant" if p < alpha else
             f"The mean difference between {nb} and {na} is not statistically significant")
            + f" ({p_text(p)}; {test_name}{tail_txt}). Mean of differences ({nb} − {na}) {fnum(mean_d)} "
              f"({cl} CI {fnum(lo)} to {fnum(hi)}; n = {n} pairs).")
    if ctx.opt("show_plot"):
        _before_after_plot(r, na, a, nb, b, ylabel=value_label(ctx), log=ratio)
    r.graph("Create a before–after graph", "before_after", layout_options(ctx, extra=("subject",)))
    return r


# --------------------------------------------------------------------------
# Wilcoxon matched pairs
# --------------------------------------------------------------------------

@analysis(
    id="wilcoxon_paired", name="Wilcoxon matched-pairs signed rank test", category=CAT,
    description="Nonparametric test for two matched measurements.",
    keywords="wilcoxon signed rank paired nonparametric before after matched",
    options=[
        *opt.group_layout(paired=True),
        opt.choice("tails", "P value", TAILS2, default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_descriptives", "Descriptive statistics", True, section="Output"),
        opt.boolean("show_plot", "Before–after plot", True, section="Output"),
    ])
def wilcoxon_paired(ctx: Context) -> Results:
    na, a, nb, b, dropped = pairs(ctx, min_n=2)
    alt = ctx.opt("tails", "two-sided")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    d = b - a
    sr = signed_rank(d, alt, conf)
    if sr["n_used"] == 0:
        raise AnalysisError("All paired differences are zero, so the signed-rank test cannot be computed.")
    p = sr["p"]
    r = Results()
    t = r.table("Wilcoxon matched-pairs signed rank test",
                [("w", "W (sum of signed ranks)", "num"), ("t_pos", "Sum of positive ranks", "num"),
                 ("t_neg", "Sum of negative ranks", "num"), ("p", "P value", "p"),
                 ("stars", "P value summary", "text"), ("sig", f"Significant (α = {alpha:g})?", "text"),
                 ("method", "P value method", "text"), ("tails", "Tails", "text"),
                 ("n", "Number of pairs", "int"), ("n_zero", "Zero differences", "int"),
                 ("rb", "Rank-biserial r", "num")])
    tot = sr["t_pos"] + sr["t_neg"]
    t.add_row(w=sr["w"], t_pos=sr["t_pos"], t_neg=-sr["t_neg"], p=p, stars=stars(p),
              sig="Yes" if p < alpha else "No", method="Exact" if sr["exact"] else "Normal approximation",
              tails=tails_label(alt), n=sr["n"], n_zero=sr["n_zero"], rb=sr["w"] / tot if tot else math.nan)
    t.note(f"Differences are {nb} − {na}. " + tail_note(alt, na, nb))
    t.note("Exact P value (no tied ranks, n ≤ 50)." if sr["exact"] else
           "P value from the normal approximation with tie and continuity correction "
           + ("(tied ranks present)." if sr["ties"] else "(n > 50)."))
    if sr["n_zero"]:
        t.note(f"{sr['n_zero']} pair(s) with a difference of zero were ignored (Wilcoxon's method).")
    t.note("Rank-biserial r = W / (sum of all ranks), from −1 to 1.")
    for note in filter(None, [_pairing_note(dropped)]):
        t.note(note)
    e = r.table("Median of differences",
                [("label", "Difference", "text"), ("median", "Median of differences", "num"),
                 ("lo", "CI lower", "num"), ("hi", "CI upper", "num"), ("actual", "Actual confidence level", "pct"),
                 ("hl", "Hodges–Lehmann estimate", "num"), ("hl_lo", "HL CI lower", "num"),
                 ("hl_hi", "HL CI upper", "num"), ("hl_actual", "HL actual confidence level", "pct")])
    e.add_row(label=f"{nb} − {na}", median=sr["median"], lo=sr["med_lo"], hi=sr["med_hi"], actual=sr["med_conf"],
              hl=sr["hl"], hl_lo=sr["hl_lo"], hl_hi=sr["hl_hi"], hl_actual=sr["hl_conf"])
    e.note(f"The CI of the median is distribution-free (order statistics); its actual confidence level is at least "
           f"the requested {cl} and it is empty when there are too few pairs.")
    e.note("The Hodges–Lehmann estimate (pseudo-median) is the median of all pairwise averages of the differences; "
           "its CI is the one that corresponds to the signed-rank test and assumes symmetric differences.")
    pr = r.table("How effective was the pairing?",
                 [("rs", "Spearman r", "num"), ("p", "P value (one-tailed)", "p"),
                  ("stars", "P value summary", "text"), ("effective", "Was the pairing significantly effective?", "text")])
    if len(a) >= 3 and not is_constant(a) and not is_constant(b):
        sp = stats.spearmanr(a, b, alternative="greater")
        pr.add_row(rs=float(sp.statistic), p=float(sp.pvalue), stars=stars(float(sp.pvalue)),
                   effective="Yes" if sp.pvalue < alpha else "No")
    else:
        pr.add_row(effective="not tested")
    pr.note("Spearman rank correlation between the two conditions; one-tailed P value for a positive correlation "
            "(t approximation).")
    if ctx.opt("show_descriptives"):
        descriptives_table(r, {na: a, nb: b, f"Difference ({nb} − {na})": d}, conf)
    r.summary(
        (f"The difference between {nb} and {na} is statistically significant" if p < alpha else
         f"The difference between {nb} and {na} is not statistically significant")
        + f" ({p_text(p)}; Wilcoxon matched-pairs signed rank test, "
        + ("exact" if sr["exact"] else "normal approximation") + (", one-tailed" if alt != "two-sided" else "") + "). "
        + f"Median of differences ({nb} − {na}) {fnum(sr['median'])}"
        + (f" (CI {fnum(sr['med_lo'])} to {fnum(sr['med_hi'])})." if np.isfinite(sr["med_lo"]) else "."))
    if ctx.opt("show_plot"):
        _before_after_plot(r, na, a, nb, b, ylabel=value_label(ctx))
    r.graph("Create a before–after graph", "before_after", layout_options(ctx, extra=("subject",)))
    return r


# --------------------------------------------------------------------------
# Kolmogorov–Smirnov
# --------------------------------------------------------------------------

@analysis(
    id="ks_two_sample", name="Kolmogorov–Smirnov test (two samples)", category=CAT,
    description="Compare the cumulative distributions of two independent groups.",
    keywords="ks kolmogorov smirnov distribution ecdf cumulative nonparametric",
    options=[
        *opt.group_layout(),
        opt.alpha(section="Options"),
        opt.boolean("show_descriptives", "Descriptive statistics", True, section="Output"),
        opt.boolean("show_plot", "Cumulative distribution plot", True, section="Output"),
    ])
def ks_two_sample(ctx: Context) -> Results:
    na, a, nb, b = two_groups(ctx, min_n=2)
    alpha = float(ctx.opt("alpha", 0.05))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
    exact = max(len(a), len(b)) <= 10000  # scipy's rule for method="auto"
    p, dstat = float(res.pvalue), float(res.statistic)
    loc = float(getattr(res, "statistic_location", math.nan))
    r = Results()
    t = r.table("Kolmogorov–Smirnov test",
                [("d", "Kolmogorov–Smirnov D", "num"), ("p", "P value", "p"), ("stars", "P value summary", "text"),
                 ("sig", f"Significant (α = {alpha:g})?", "text"), ("method", "P value method", "text"),
                 ("loc", "Largest gap at value", "num"), ("n1", f"n ({na})", "int"), ("n2", f"n ({nb})", "int")])
    t.add_row(d=dstat, p=p, stars=stars(p), sig="Yes" if p < alpha else "No",
              method="Exact" if exact else "Asymptotic", loc=loc, n1=len(a), n2=len(b))
    t.note("D is the largest vertical distance between the two cumulative distributions. Two-tailed P value. "
           "The test is sensitive to any difference in distribution (location, spread or shape).")
    if len(np.unique(np.concatenate([a, b]))) < len(a) + len(b):
        t.note("Tied values are present; the P value assumes continuous data and is then conservative.")
    if ctx.opt("show_descriptives"):
        descriptives_table(r, {na: a, nb: b}, 0.95)
    r.summary(
        (f"The distributions of {na} and {nb} differ significantly" if p < alpha else
         f"The distributions of {na} and {nb} are not significantly different")
        + f" ({p_text(p)}; Kolmogorov–Smirnov test, D = {fnum(dstat)}).")
    if ctx.opt("show_plot"):
        fig, ax = new_figure(5, 3.6)
        for i, (name, x) in enumerate(((na, a), (nb, b))):
            xs = np.sort(x)
            ax.step(np.concatenate([[xs[0]], xs]), np.arange(0, len(xs) + 1) / len(xs), where="post",
                    color=COLORS[i], lw=1.6, label=name)
        if np.isfinite(loc):
            fa, fb = np.mean(a <= loc), np.mean(b <= loc)
            ax.plot([loc, loc], [fa, fb], color="#222222", lw=1.2, ls=":", label=f"D = {dstat:.3g}")
        ax.set_xlabel(value_label(ctx))
        ax.set_ylabel("Cumulative fraction")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
        r.plot("Cumulative distributions", fig)
    r.graph("Create a column graph of these data", "column_scatter", layout_options(ctx))
    return r


# --------------------------------------------------------------------------
# Multiple t tests
# --------------------------------------------------------------------------

_MT_METHODS = [("fdr_bh", "FDR (Benjamini–Hochberg)"), ("fdr_by", "FDR (Benjamini–Yekutieli)"),
               ("holm-sidak", "Holm–Šidák"), ("bonferroni", "Bonferroni"), ("none", "None (unadjusted P values)")]


@analysis(
    id="multiple_ttests", name="Multiple t tests (one per variable)", category=CAT,
    description="Compare two groups on many variables at once, with correction for multiple comparisons.",
    keywords="multiple comparisons fdr q value discovery volcano holm sidak bonferroni benjamini hochberg",
    options=[
        opt.variables("variables", "Variables (one test each)", opt.CONTINUOUS, min=1),
        opt.variable("group", "Grouping variable (2 levels)", opt.ANY),
        opt.choice("test", "Test", [("welch", "Welch's t test"), ("student", "Student's t test"),
                                    ("mann_whitney", "Mann–Whitney test")], default="welch"),
        opt.choice("correction", "Multiple comparisons", _MT_METHODS, default="fdr_bh", section="Options"),
        opt.number("alpha", "Threshold (α or FDR Q)", 0.05, min=0.0001, max=0.5, step=0.01, section="Options"),
        opt.boolean("show_plot", "Volcano plot", True, section="Output"),
    ])
def multiple_ttests(ctx: Context) -> Results:
    variables = ctx.vars("variables", min=1)
    gname = ctx.var("group")
    if gname in variables:
        raise AnalysisError("The grouping variable cannot also be one of the tested variables.")
    fac = ctx.factor(gname)
    levels = [str(l) for l in fac.cat.categories]
    if len(levels) != 2:
        raise AnalysisError(f"The grouping variable “{gname}” must have exactly 2 levels, but it has {len(levels)}"
                            + (f" ({', '.join(levels[:6])}{'…' if len(levels) > 6 else ''})." if levels else "."))
    la, lb = levels
    test = ctx.opt("test", "welch")
    method = ctx.opt("correction", "fdr_bh")
    alpha = float(ctx.opt("alpha", 0.05))
    gvals = fac.astype(object).to_numpy()
    rows = []
    for v in variables:
        x = ctx.numeric(v).to_numpy(float)
        a = x[(gvals == la) & np.isfinite(x)]
        b = x[(gvals == lb) & np.isfinite(x)]
        row = {"variable": v, "n1": len(a), "n2": len(b),
               "mean1": float(a.mean()) if len(a) else math.nan, "mean2": float(b.mean()) if len(b) else math.nan,
               "diff": math.nan, "se": math.nan, "stat": math.nan, "df": math.nan, "p": math.nan, "note": ""}
        if len(a) < 2 or len(b) < 2:
            row["note"] = "fewer than 2 values in a group"
        elif is_constant(np.concatenate([a, b])) or (is_constant(a) and is_constant(b)):
            row["note"] = "no variability"
            row["diff"] = row["mean2"] - row["mean1"]
        elif test == "mann_whitney":
            mw = mann_whitney_core(a, b)
            row.update(diff=float(np.median((b[None, :] - a[:, None]).ravel())), stat=mw["u"], p=mw["p"],
                       mean1=float(np.median(a)), mean2=float(np.median(b)))
        else:
            res = unpaired_t(a, b, test == "welch", "two-sided", 0.95)
            row.update(diff=res["diff"], se=res["se"], stat=res["t"], df=res["df"], p=res["p"])
        rows.append(row)
    pvals = np.array([row["p"] for row in rows])
    ok = np.isfinite(pvals)
    if not ok.any():
        raise AnalysisError("None of the variables could be tested: each group needs at least 2 values "
                            "and some variability.")
    adj = np.full(len(rows), math.nan)
    if method == "none":
        adj[ok] = pvals[ok]
    else:
        from statsmodels.stats.multitest import multipletests
        adj[ok] = multipletests(pvals[ok], alpha=alpha, method=method)[1]
    fdr = method.startswith("fdr")
    mw = test == "mann_whitney"
    centre = "Median" if mw else "Mean"
    r = Results()
    cols = [("variable", "Variable", "text"), ("n1", f"n ({la})", "int"), ("n2", f"n ({lb})", "int"),
            ("mean1", f"{centre} ({la})", "num"), ("mean2", f"{centre} ({lb})", "num"),
            ("diff", "Hodges–Lehmann difference" if mw else "Difference", "num")]
    if not mw:
        cols += [("se", "SE of difference", "num"), ("stat", "t", "num"), ("df", "df", "num")]
    else:
        cols += [("stat", "Mann–Whitney U", "num")]
    cols += [("p", "P value", "p")]
    if method != "none":
        cols.append(("adj", "q value" if fdr else "Adjusted P value", "p"))
    cols += [("hit", "Discovery?" if fdr else "Significant?", "text"), ("note", "Note", "text")]
    t = r.table("Multiple " + ("Mann–Whitney tests" if mw else "t tests"), cols)
    hits = []
    for row, q in zip(rows, adj):
        hit = bool(np.isfinite(q) and q < alpha)
        if hit:
            hits.append(row["variable"])
        t.add_row(adj=q, hit=("Yes" if hit else "No") if np.isfinite(q) else "", **row)
    test_label = {"welch": "Welch's t test (unequal variances)", "student": "Student's t test (pooled variance within each variable)",
                  "mann_whitney": "Mann–Whitney test (exact P value without ties and n ≤ 50, else normal approximation)"}[test]
    t.note(f"Each row is a separate two-tailed {test_label}; difference = {lb} − {la}. "
           "Missing values are excluded variable by variable.")
    m = int(ok.sum())
    corr_label = dict(_MT_METHODS)[method]
    if fdr:
        t.note(f"q values: false discovery rate, {corr_label} procedure across {m} tests"
               + (" (valid under arbitrary dependence)" if method == "fdr_by" else "")
               + f". A row is a discovery when q < {alpha:g}.")
    elif method == "none":
        t.note(f"No correction for multiple comparisons; significant when P < {alpha:g}. With {m} tests, "
               "expect some small P values by chance alone.")
    else:
        t.note(f"Adjusted P values: {corr_label} method across {m} tests (family-wise error rate); "
               f"significant when adjusted P < {alpha:g}.")
    what = "discoveries" if fdr else "significant differences"
    if hits:
        r.summary(f"{len(hits)} of {m} variables {'are discoveries' if fdr else 'differ significantly'} between {la} and {lb} "
                  f"({corr_label}, threshold {alpha:g}): " + ", ".join(hits[:8]) + ("…" if len(hits) > 8 else "") + ".")
    else:
        r.summary(f"No {what} between {la} and {lb} among {m} variables ({corr_label}, threshold {alpha:g}).")
    if ctx.opt("show_plot") and ok.any():
        fig, ax = new_figure(5.2, 3.8)
        diffs = np.array([row["diff"] for row in rows])
        y = -np.log10(np.clip(pvals, 1e-300, 1))
        is_hit = np.isfinite(adj) & (adj < alpha)
        ax.scatter(diffs[ok & ~is_hit], y[ok & ~is_hit], s=22, color="#9AA5B1", label="Not significant", linewidths=0)
        if is_hit.any():
            ax.scatter(diffs[is_hit], y[is_hit], s=28, color=COLORS[1], label="Discovery" if fdr else "Significant",
                       linewidths=0)
        order = [i for i in np.argsort(-np.where(ok, y, -1)) if ok[i]][:8]
        for i in order:
            ax.annotate(rows[i]["variable"], (diffs[i], y[i]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.axvline(0, color="#BBBBBB", lw=0.8, zorder=0)
        ax.set_xlabel(f"Difference ({lb} − {la})")
        ax.set_ylabel("−log10(P value)")
        if len(rows) > 1:
            ax.legend(loc="best")
        r.plot("Volcano plot", fig)
    return r
