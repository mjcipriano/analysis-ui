"""Describe & explore: descriptive statistics, normality tests, outliers,
frequency distributions and one-sample tests.

The small helpers at the top (confidence level option, mean CI, signed-rank
test, number parsing, ...) are shared with the other analysis modules.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np
from scipy import stats

from ..core import analysis, opt, Context, Results, AnalysisError, new_figure, COLORS, stars, fmt_p

CAT = "Describe & explore"

TAILS = [("two-sided", "Two-tailed"),
         ("greater", "One-tailed (greater)"),
         ("less", "One-tailed (less)")]


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def conf_opt(section: str | None = "Options") -> dict:
    return opt.number("conf_level", "Confidence level (%)", 95, min=50, max=99.99,
                      step=1, section=section)


def conf_level(ctx: Context) -> float:
    """Confidence level as a fraction (0.95)."""
    return float(ctx.opt("conf_level", 95)) / 100.0


def pct(conf: float) -> str:
    """'95%' / '99.5%' label for a confidence fraction."""
    return f"{conf * 100:g}%"


def fnum(x: float, digits: int = 4) -> str:
    """Compact number for plain-language summaries."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    if abs(x) >= 10 ** digits or abs(x) < 1e-4:
        return f"{x:.{digits - 1}e}"
    return f"{x:.{digits}g}"


def p_text(p: float) -> str:
    """'P = 0.0031' or 'P < 0.0001' for plain-language summaries."""
    s = fmt_p(p)
    return f"P {s[0]} {s[1:]}" if s.startswith("<") else f"P = {s}"


def layout_options(ctx: Context, extra: tuple = ()) -> dict:
    """The group-layout options, for ``r.graph`` links."""
    keys = ("layout", "columns", "dep", "group") + tuple(extra)
    return {k: ctx.options.get(k) for k in keys if k in ctx.options}


def tails_label(alt: str) -> str:
    return "Two-tailed" if alt == "two-sided" else "One-tailed"


def describe(x: np.ndarray, conf: float = 0.95) -> dict:
    """n, mean, SD, SEM and CI of the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.mean(x)) if n else math.nan
    sd = float(np.std(x, ddof=1)) if n > 1 else math.nan
    sem = sd / math.sqrt(n) if n > 1 else math.nan
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * sem if n > 1 else math.nan
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "ci_lo": mean - half, "ci_hi": mean + half,
            "median": float(np.median(x)) if n else math.nan,
            "min": float(np.min(x)) if n else math.nan,
            "max": float(np.max(x)) if n else math.nan}


def quantile(x: np.ndarray, q: float) -> float:
    """Percentile by the (n+1)p rule (what GraphPad Prism uses)."""
    return float(np.quantile(np.asarray(x, dtype=float), q, method="weibull"))


def is_constant(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    return len(x) == 0 or float(np.ptp(x)) <= 1e-12 * max(1.0, float(np.max(np.abs(x))))


def parse_numbers(text: str, what: str = "values") -> list[float]:
    """Parse '1, 2.5; 3' typed by the user."""
    out = []
    for tok in str(text or "").replace(";", ",").replace("\n", ",").replace("\t", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        pct_sign = tok.endswith("%")
        try:
            v = float(tok[:-1] if pct_sign else tok)
        except ValueError:
            raise AnalysisError(f"Could not read “{tok}” as a number in the {what}. "
                                "Type numbers separated by commas.")
        if not math.isfinite(v):
            raise AnalysisError(f"“{tok}” is not a finite number in the {what}.")
        out.append(v)
    return out


def median_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Distribution-free CI of the median from order statistics.
    Returns (lo, hi, actual confidence level); NaNs when n is too small."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    a2 = (1 - conf) / 2
    k = 0
    while k < n // 2 and stats.binom.cdf(k, n, 0.5) <= a2:
        k += 1
    # k = number of order statistics trimmed from each end + 1
    if k == 0:
        return math.nan, math.nan, math.nan
    return float(x[k - 1]), float(x[n - k]), float(1 - 2 * stats.binom.cdf(k - 1, n, 0.5))


@lru_cache(maxsize=64)
def _signrank_cdf(n: int) -> np.ndarray:
    """Exact null CDF of the Wilcoxon signed-rank statistic T+ (no ties)."""
    pmf = np.zeros(n * (n + 1) // 2 + 1)
    pmf[0] = 1.0
    for k in range(1, n + 1):
        shifted = np.zeros_like(pmf)
        shifted[k:] = pmf[:-k]
        pmf = 0.5 * (pmf + shifted)
    return np.cumsum(pmf)


def signed_rank(d: np.ndarray, alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Wilcoxon signed-rank test of differences ``d`` against zero.

    Zero differences are dropped (Wilcoxon's method). The P value is exact
    when there are no ties and at most 50 non-zero differences, otherwise a
    normal approximation with tie and continuity correction is used.
    Also returns the Hodges–Lehmann pseudo-median with its CI and the plain
    median with an order-statistic CI.
    """
    d_all = np.asarray(d, dtype=float)
    nz = d_all[d_all != 0]
    n = len(nz)
    out = {"n": len(d_all), "n_used": n, "n_zero": len(d_all) - n,
           "t_pos": math.nan, "t_neg": math.nan, "w": math.nan, "p": math.nan,
           "exact": False, "ties": False, "z": math.nan}
    if n >= 1:
        ranks = stats.rankdata(np.abs(nz))
        out["t_pos"] = float(ranks[nz > 0].sum())
        out["t_neg"] = float(ranks[nz < 0].sum())
        out["w"] = out["t_pos"] - out["t_neg"]
        out["ties"] = len(np.unique(np.abs(nz))) < n
        out["exact"] = (not out["ties"]) and n <= 50
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(nz, zero_method="wilcox", alternative=alternative,
                                 method="exact" if out["exact"] else "approx",
                                 correction=not out["exact"])
        out["p"] = float(res.pvalue)
        if not out["exact"]:
            out["z"] = float(getattr(res, "zstatistic", math.nan))
    # Hodges–Lehmann estimate (median of Walsh averages) and its CI
    m = len(d_all)
    i, j = np.triu_indices(m)
    walsh = np.sort((d_all[i] + d_all[j]) / 2.0)
    out["hl"] = float(np.median(walsh)) if m else math.nan
    out["hl_lo"] = out["hl_hi"] = out["hl_conf"] = math.nan
    alpha = 1 - conf
    total = m * (m + 1) // 2
    if m >= 2:
        if m <= 50:
            cdf = _signrank_cdf(m)
            qu = int(np.searchsorted(cdf, alpha / 2 - 1e-12, side="left"))  # qsignrank
            if qu == 0:
                qu = 1
            achieved = 1 - 2 * cdf[qu - 1]
        else:
            z = stats.norm.ppf(1 - alpha / 2)
            qu = int(math.floor(total / 2 - z * math.sqrt(m * (m + 1) * (2 * m + 1) / 24.0)))
            qu = max(qu, 1)
            achieved = conf
        ql = total - qu
        if achieved > 0 and 0 <= qu - 1 < total and 0 <= ql < total:
            out["hl_lo"], out["hl_hi"], out["hl_conf"] = float(walsh[qu - 1]), float(walsh[ql]), float(achieved)
    out["median"] = float(np.median(d_all)) if m else math.nan
    out["med_lo"], out["med_hi"], out["med_conf"] = median_ci(d_all, conf) if m else (math.nan,) * 3
    return out


def groups_with_rows(ctx: Context) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Like ``ctx.groups()`` but also returns the 1-based data row of every value."""
    long = ctx.long(dropna=False)
    nrow = len(ctx.df)
    if ctx.opt("layout", "wide") == "wide":
        rows = np.arange(len(long)) % max(nrow, 1) + 1
    else:
        rows = np.arange(len(long)) + 1
    long = long.assign(_row=rows)
    long = long[long["value"].notna() & long["group"].notna()]
    out = {}
    for g in long["group"].cat.categories:
        sub = long[long["group"] == g]
        if len(sub):
            out[str(g)] = (sub["value"].to_numpy(dtype=float), sub["_row"].to_numpy())
    if not out:
        raise AnalysisError("No numeric values were found in the selected data.")
    return out


def group_columns(groups) -> list[tuple]:
    """Table columns for a 'one column per group' (Prism-style) table."""
    return [(f"g{i}", name, "num") for i, name in enumerate(groups)]


def jitter(n: int, width: float = 0.12, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def strip_axis(ax, groups: dict, *, seed: int = 1, alpha: float = 0.75, size: float = 16):
    """Draw every value as a jittered dot, one x position per group."""
    for i, (name, x) in enumerate(groups.items()):
        ax.scatter(i + jitter(len(x), seed=seed + i), x, s=size, color=COLORS[i % len(COLORS)],
                   alpha=alpha, linewidths=0, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(list(groups), rotation=0 if len(groups) <= 4 else 30,
                       ha="center" if len(groups) <= 4 else "right")
    ax.set_xlim(-0.6, len(groups) - 0.4)


def value_label(ctx: Context) -> str:
    return ctx.options.get("dep") or "Value" if ctx.opt("layout", "wide") == "long" else "Value"


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

_STAT_SECTION = "Statistics"


@analysis(
    id="descriptives", name="Descriptive statistics", category=CAT,
    description="Mean, SD, SEM, confidence interval, median, quartiles and more for each group.",
    keywords="column statistics summary mean median sd sem cv quartile skewness kurtosis geometric",
    options=[
        *opt.group_layout(min_groups=1),
        opt.boolean("show_n", "Number of values", True, section=_STAT_SECTION),
        opt.boolean("show_mean", "Mean", True, section=_STAT_SECTION),
        opt.boolean("show_sd", "SD and SEM", True, section=_STAT_SECTION),
        opt.boolean("show_ci", "CI of the mean", True, section=_STAT_SECTION),
        opt.boolean("show_median", "Median", True, section=_STAT_SECTION),
        opt.boolean("show_quartiles", "Quartiles and IQR", True, section=_STAT_SECTION),
        opt.boolean("show_minmax", "Minimum, maximum, range", True, section=_STAT_SECTION),
        opt.boolean("show_cv", "Coefficient of variation", False, section=_STAT_SECTION),
        opt.boolean("show_geo", "Geometric mean and SD factor", False, section=_STAT_SECTION),
        opt.boolean("show_shape", "Skewness and kurtosis", False, section=_STAT_SECTION),
        opt.boolean("show_sum", "Sum", False, section=_STAT_SECTION),
        conf_opt(section="Options"),
        opt.boolean("show_plot", "Box plot with values", True, section="Output"),
    ])
def descriptives(ctx: Context) -> Results:
    groups = ctx.groups(min_groups=1)
    conf = conf_level(ctx)
    names = list(groups)
    r = Results()
    t = r.table("Descriptive statistics", [("stat", "Statistic", "text"), *group_columns(names)])

    def row(label, fn):
        vals = {}
        for i, x in enumerate(groups.values()):
            try:
                v = fn(x)
            except Exception:
                v = math.nan
            vals[f"g{i}"] = v
        t.add_row(stat=label, **vals)

    d = [describe(x, conf) for x in groups.values()]
    by = {id(x): di for x, di in zip(groups.values(), d)}
    get = lambda key: (lambda x: by[id(x)][key])  # noqa: E731

    if ctx.opt("show_n"):
        row("Number of values", lambda x: len(x))
    if ctx.opt("show_minmax"):
        row("Minimum", np.min)
    if ctx.opt("show_quartiles"):
        row("25% percentile", lambda x: quantile(x, 0.25))
    if ctx.opt("show_median"):
        row("Median", np.median)
    if ctx.opt("show_quartiles"):
        row("75% percentile", lambda x: quantile(x, 0.75))
    if ctx.opt("show_minmax"):
        row("Maximum", np.max)
        row("Range", np.ptp)
    if ctx.opt("show_quartiles"):
        row("Interquartile range", lambda x: quantile(x, 0.75) - quantile(x, 0.25))
    if ctx.opt("show_mean"):
        row("Mean", get("mean"))
    if ctx.opt("show_sd"):
        row("Std. deviation", get("sd"))
        row("Std. error of mean", get("sem"))
    if ctx.opt("show_ci"):
        row(f"Lower {pct(conf)} CI of mean", get("ci_lo"))
        row(f"Upper {pct(conf)} CI of mean", get("ci_hi"))
    if ctx.opt("show_cv"):
        row("Coefficient of variation (%)",
            lambda x: 100 * by[id(x)]["sd"] / by[id(x)]["mean"] if by[id(x)]["mean"] != 0 else math.nan)
    non_positive = []
    if ctx.opt("show_geo"):
        def geo(x, what):
            if np.any(x <= 0):
                return math.nan
            lx = np.log(x)
            if what == "mean":
                return float(np.exp(lx.mean()))
            if len(x) < 2:
                return math.nan
            gsd = float(np.exp(lx.std(ddof=1)))
            if what == "sd":
                return gsd
            half = stats.t.ppf(0.5 + conf / 2, len(x) - 1) * lx.std(ddof=1) / math.sqrt(len(x))
            return float(np.exp(lx.mean() - half if what == "lo" else lx.mean() + half))
        non_positive = [n for n, x in groups.items() if np.any(x <= 0)]
        row("Geometric mean", lambda x: geo(x, "mean"))
        row("Geometric SD factor", lambda x: geo(x, "sd"))
        row(f"Lower {pct(conf)} CI of geo. mean", lambda x: geo(x, "lo"))
        row(f"Upper {pct(conf)} CI of geo. mean", lambda x: geo(x, "hi"))
    if ctx.opt("show_shape"):
        row("Skewness", lambda x: float(stats.skew(x, bias=False)) if len(x) >= 3 and not is_constant(x) else math.nan)
        row("Excess kurtosis", lambda x: float(stats.kurtosis(x, bias=False)) if len(x) >= 4 and not is_constant(x) else math.nan)
    if ctx.opt("show_sum"):
        row("Sum", np.sum)
    if not t.rows:
        raise AnalysisError("Choose at least one statistic to report.")

    if ctx.opt("show_quartiles"):
        t.note("Percentiles are computed with the (n+1)p rule, as in GraphPad Prism.")
    if ctx.opt("show_shape"):
        t.note("Skewness and excess kurtosis are the bias-corrected sample estimates (as in Prism, Excel and SAS); "
               "both are 0 for a Gaussian distribution.")
    if non_positive:
        t.note("Geometric statistics are left empty for groups with zero or negative values: "
               + ", ".join(non_positive) + ".")
    single = [n for n, x in groups.items() if len(x) < 2]
    if single:
        t.note("SD, SEM and CI need at least two values and are empty for: " + ", ".join(single) + ".")

    parts = [f"{n}: mean {fnum(di['mean'])}" + (f" (SD {fnum(di['sd'])}, n = {di['n']})" if di["n"] > 1 else " (n = 1)")
             for n, di in zip(names[:6], d)]
    r.summary("; ".join(parts) + ("; …" if len(names) > 6 else "") + ".")

    if ctx.opt("show_plot"):
        fig, ax = new_figure(max(3.2, min(9, 1.6 + 1.0 * len(groups))), 3.6)
        bp = ax.boxplot(list(groups.values()), positions=range(len(groups)), widths=0.5,
                        showfliers=False, patch_artist=True, medianprops={"color": "#222222", "linewidth": 1.4})
        for i, box in enumerate(bp["boxes"]):
            box.set(facecolor=COLORS[i % len(COLORS)], alpha=0.22, edgecolor="#444444")
        strip_axis(ax, groups)
        ax.set_ylabel(value_label(ctx))
        r.plot("Box plot with individual values", fig)
    r.graph("Create a column graph of these data", "column_scatter", layout_options(ctx))
    return r


# --------------------------------------------------------------------------
# Normality
# --------------------------------------------------------------------------

def _normality_tests(x: np.ndarray, which: dict) -> list[dict]:
    """Run the requested normality tests on one sample."""
    from statsmodels.stats.diagnostic import lilliefors, normal_ad
    n = len(x)
    const = is_constant(x)
    rows = []

    def add(key, name, symbol, min_n, fn, max_n=None):
        if not which.get(key):
            return
        row = {"test": name, "symbol": symbol, "stat": math.nan, "p": math.nan, "note": ""}
        if n < min_n:
            row["note"] = f"n too small (needs ≥ {min_n})"
        elif max_n and n > max_n:
            row["note"] = f"n too large (needs ≤ {max_n})"
        elif const:
            row["note"] = "all values identical"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s, p = fn(x)
                row["stat"], row["p"] = float(s), float(p)
            except Exception as exc:  # pragma: no cover - defensive
                row["note"] = f"could not be computed ({type(exc).__name__})"
        rows.append(row)

    add("ad", "Anderson–Darling", "A²*", 8, _anderson_darling)
    add("dagostino", "D'Agostino–Pearson omnibus", "K²", 8, lambda v: tuple(stats.normaltest(v)))
    add("shapiro", "Shapiro–Wilk", "W", 3, lambda v: tuple(stats.shapiro(v)), max_n=5000)
    add("ks", "Kolmogorov–Smirnov (Lilliefors)", "D", 5, lambda v: lilliefors(v, dist="norm", pvalmethod="table"))
    return rows


def _anderson_darling(x: np.ndarray) -> tuple[float, float]:
    """Anderson–Darling test for normality (mean and SD estimated), with
    Stephens' small-sample adjustment A²* and D'Agostino & Stephens' P value
    (same as R ``nortest::ad.test``)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    z = (x - x.mean()) / x.std(ddof=1)
    logp1 = stats.norm.logcdf(z)
    logp2 = stats.norm.logsf(z[::-1])
    i = np.arange(1, n + 1)
    a2 = -n - np.mean((2 * i - 1) * (logp1 + logp2))
    aa = a2 * (1 + 0.75 / n + 2.25 / n ** 2)
    if aa < 0.2:
        p = 1 - math.exp(-13.436 + 101.14 * aa - 223.73 * aa ** 2)
    elif aa < 0.34:
        p = 1 - math.exp(-8.318 + 42.796 * aa - 59.938 * aa ** 2)
    elif aa < 0.6:
        p = math.exp(0.9177 - 4.279 * aa - 1.38 * aa ** 2)
    elif aa < 10:
        p = math.exp(1.2937 - 5.709 * aa + 0.0186 * aa ** 2)
    else:
        p = 3.7e-24
    return float(aa), float(p)


@analysis(
    id="normality", name="Normality & lognormality tests", category=CAT,
    description="Test whether each group is consistent with a Gaussian (or lognormal) distribution.",
    keywords="shapiro wilk dagostino pearson anderson darling kolmogorov smirnov lilliefors qq plot gaussian",
    options=[
        *opt.group_layout(min_groups=1),
        opt.boolean("shapiro", "Shapiro–Wilk", True, section="Tests"),
        opt.boolean("dagostino", "D'Agostino–Pearson omnibus", True, section="Tests"),
        opt.boolean("ad", "Anderson–Darling", True, section="Tests"),
        opt.boolean("ks", "Kolmogorov–Smirnov (Lilliefors)", True, section="Tests"),
        opt.boolean("lognormal", "Also test lognormality", False, section="Options",
                    help="Runs the same tests on the logarithms of the values."),
        opt.alpha(section="Options"),
        opt.boolean("show_plot", "QQ plots", True, section="Output"),
    ])
def normality(ctx: Context) -> Results:
    groups = ctx.groups(min_groups=1)
    alpha = float(ctx.opt("alpha", 0.05))
    which = {k: bool(ctx.opt(k)) for k in ("shapiro", "dagostino", "ad", "ks")}
    if not any(which.values()):
        raise AnalysisError("Choose at least one normality test.")
    if all(len(x) < 3 for x in groups.values()):
        raise AnalysisError("Normality tests need at least 3 values in a group "
                            "(8 for the D'Agostino–Pearson and Anderson–Darling tests).")
    r = Results()
    cols = [("group", "Group", "text"), ("test", "Test", "text"), ("n", "n", "int"),
            ("symbol", "Statistic", "text"), ("stat", "Value", "num"), ("p", "P value", "p"),
            ("passed", f"Passed (α = {alpha:g})?", "text"), ("stars", "Summary", "text"),
            ("note", "Note", "text")]

    def run(title, data, log=False):
        t = r.table(title, cols)
        verdict = {}
        for name, x in data.items():
            for row in _normality_tests(x, which):
                ok = None if not np.isfinite(row["p"]) else row["p"] >= alpha
                verdict.setdefault(name, []).append(ok)
                t.add_row(group=name, n=len(x), passed="" if ok is None else ("Yes" if ok else "No"),
                          stars=stars(row["p"]) if ok is not None else "", **row)
        t.note("Null hypothesis: the values were sampled from a "
               + ("lognormal" if log else "Gaussian") + " distribution. A small P value is evidence against it; "
               "a large P value does not prove the distribution is " + ("lognormal." if log else "Gaussian."))
        if which["ks"]:
            t.note("Kolmogorov–Smirnov uses the Lilliefors correction (mean and SD estimated from the data); "
                   "its P value is interpolated from simulated tables and is limited to the range 0.001 to 0.99.")
        if which["ad"]:
            t.note("Anderson–Darling: A²* is the statistic adjusted for sample size (Stephens); "
                   "P value from D'Agostino & Stephens (1986).")
        return verdict

    verdict = run("Normality tests", groups)
    log_verdict = None
    if ctx.opt("lognormal"):
        ok_groups = {n: np.log(x) for n, x in groups.items() if np.all(x > 0)}
        bad = [n for n in groups if n not in ok_groups]
        if bad:
            r.warn("Lognormality cannot be tested for groups with zero or negative values: " + ", ".join(bad) + ".")
        if ok_groups:
            log_verdict = run("Lognormality tests (tests on ln of the values)", ok_groups, log=True)

    def phrase(v, kind):
        failed = [n for n, oks in v.items() if any(o is False for o in oks)]
        tested = [n for n, oks in v.items() if any(o is not None for o in oks)]
        if not tested:
            return f"No {kind} test could be computed."
        if not failed:
            return (f"No test rejected {kind} at α = {alpha:g} "
                    f"({'all ' + str(len(tested)) + ' groups' if len(tested) > 1 else tested[0]}); "
                    f"the data are consistent with a {'lognormal' if kind == 'lognormality' else 'Gaussian'} distribution.")
        return (f"At least one test rejected {kind} at α = {alpha:g} for: " + ", ".join(failed) + ".")

    text = phrase(verdict, "normality")
    if log_verdict is not None:
        text += " " + phrase(log_verdict, "lognormality")
    r.summary(text)

    if ctx.opt("show_plot"):
        show = {n: x for n, x in groups.items() if len(x) >= 3 and not is_constant(x)}
        if show:
            k = len(show)
            ncols = min(k, 4)
            nrows = math.ceil(k / ncols)
            fig, axes = new_figure(2.6 * ncols + 0.4, 2.6 * nrows + 0.2, nrows, ncols, squeeze=False)
            for ax in axes.ravel()[k:]:
                ax.set_visible(False)
            for i, (name, x) in enumerate(show.items()):
                ax = axes.ravel()[i]
                (osm, osr), (slope, inter, _) = stats.probplot(x, dist="norm")
                ax.scatter(osm, osr, s=14, color=COLORS[i % len(COLORS)], alpha=0.8, linewidths=0)
                xx = np.array([osm.min(), osm.max()])
                ax.plot(xx, inter + slope * xx, color="#444444", lw=1)
                ax.set_title(name)
                ax.set_xlabel("Theoretical quantile (z)")
                if i % ncols == 0:
                    ax.set_ylabel("Observed value")
            r.plot("Normal QQ plots", fig)
    return r


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------

def _grubbs_once(x: np.ndarray) -> tuple[int, float, float]:
    """Index of the most extreme value, Grubbs' G and its two-sided P value."""
    n = len(x)
    sd = x.std(ddof=1)
    dev = np.abs(x - x.mean())
    i = int(np.argmax(dev))
    g = float(dev[i] / sd)
    denom = (n - 1) ** 2 - n * g * g
    if denom <= 1e-12:
        return i, g, 0.0
    tval = math.sqrt(n * (n - 2) * g * g / denom)
    return i, g, float(min(1.0, 2 * n * stats.t.sf(tval, n - 2)))


def _grubbs_critical(n: int, alpha: float) -> float:
    tc = stats.t.isf(alpha / (2 * n), n - 2)
    return (n - 1) / math.sqrt(n) * math.sqrt(tc * tc / (n - 2 + tc * tc))


def _find_outliers(x: np.ndarray, method: str, alpha: float, max_out: int | None, k: float) -> tuple[list[dict], str]:
    """Outliers of one sample → list of {index, stat, crit, p} and a note."""
    n = len(x)
    found: list[dict] = []
    if n < 3:
        return found, "fewer than 3 values"
    if is_constant(x):
        return found, ""
    idx = np.arange(n)
    if method in ("grubbs", "grubbs_iter"):
        keep = np.ones(n, bool)
        while keep.sum() >= 3 and not is_constant(x[keep]):
            i, g, p = _grubbs_once(x[keep])
            if p >= alpha:
                break
            j = idx[keep][i]
            found.append({"index": int(j), "stat": g, "crit": _grubbs_critical(int(keep.sum()), alpha), "p": p})
            keep[j] = False
            if method == "grubbs":
                break
        return found, ""
    if method == "gesd":
        r_max = max_out if max_out else max(1, min(10, n // 5))
        r_max = int(min(r_max, n - 3)) if n > 3 else 0
        if r_max < 1:
            return found, "too few values for the generalized ESD test (needs ≥ 4)"
        keep = np.ones(n, bool)
        steps = []
        for i in range(1, r_max + 1):
            sub = x[keep]
            if is_constant(sub):
                break
            dev = np.abs(sub - sub.mean())
            a = int(np.argmax(dev))
            ri = float(dev[a] / sub.std(ddof=1))
            pp = 1 - alpha / (2 * (n - i + 1))
            tc = stats.t.ppf(pp, n - i - 1)
            lam = (n - i) * tc / math.sqrt((n - i - 1 + tc * tc) * (n - i + 1))
            j = int(idx[keep][a])
            steps.append({"index": j, "stat": ri, "crit": float(lam), "p": math.nan})
            keep[j] = False
        n_out = max([i + 1 for i, s in enumerate(steps) if s["stat"] > s["crit"]], default=0)
        return steps[:n_out], f"up to {r_max} outliers tested"
    if method == "tukey":
        q1, q3 = quantile(x, 0.25), quantile(x, 0.75)
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        for j in idx[(x < lo) | (x > hi)]:
            found.append({"index": int(j), "stat": float(x[j]), "crit": float(lo if x[j] < lo else hi), "p": math.nan})
        return found, f"fences {fnum(lo)} to {fnum(hi)}"
    if method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            return found, "MAD is zero (more than half of the values are identical)"
        m = 0.6745 * (x - med) / mad
        for j in idx[np.abs(m) > 3.5]:
            found.append({"index": int(j), "stat": float(m[j]), "crit": 3.5, "p": math.nan})
        return found, ""
    raise AnalysisError(f"Unknown outlier method: {method}")


_OUTLIER_METHODS = [
    ("grubbs", "Grubbs' test (one outlier)"),
    ("grubbs_iter", "Iterative Grubbs' test"),
    ("gesd", "Generalized ESD (Rosner)"),
    ("tukey", "Tukey fences (IQR rule)"),
    ("mad", "Robust: modified Z score (median/MAD)"),
]
_OUTLIER_STAT = {"grubbs": ("G", "Critical G"), "grubbs_iter": ("G", "Critical G"),
                 "gesd": ("R", "Critical λ"), "tukey": ("Value", "Fence crossed"),
                 "mad": ("Modified Z", "Cut-off")}


@analysis(
    id="outliers", name="Identify outliers", category=CAT,
    description="Find outliers in each group with Grubbs' test, generalized ESD, Tukey fences or a robust rule.",
    keywords="grubbs rosner esd rout tukey iqr extreme values clean",
    options=[
        *opt.group_layout(min_groups=1),
        opt.choice("method", "Method", _OUTLIER_METHODS, default="grubbs"),
        opt.alpha(section="Options"),
        opt.integer("max_outliers", "Maximum number of outliers", None, min=1, max=100, section="Options",
                    visible={"method": "gesd"}, placeholder="auto",
                    help="Generalized ESD needs an upper bound; auto = n/5, at most 10."),
        opt.number("iqr_k", "Fence multiplier (× IQR)", 1.5, min=0.5, max=10, step=0.5, section="Options",
                   visible={"method": "tukey"}),
        opt.boolean("show_plot", "Plot with outliers highlighted", True, section="Output"),
    ])
def outliers(ctx: Context) -> Results:
    data = groups_with_rows(ctx)
    method = ctx.opt("method", "grubbs")
    alpha = float(ctx.opt("alpha", 0.05))
    max_out = ctx.opt("max_outliers")
    k = float(ctx.opt("iqr_k", 1.5))
    if all(len(v[0]) < 3 for v in data.values()):
        raise AnalysisError("Outlier detection needs at least 3 values in a group.")
    r = Results()
    label = dict(_OUTLIER_METHODS)[method]
    summ = r.table("Outlier summary", [("group", "Group", "text"), ("n", "n", "int"),
                                       ("n_out", "Outliers", "int"), ("values", "Outlier values", "text"),
                                       ("mean", "Mean (all)", "num"), ("mean_clean", "Mean (cleaned)", "num"),
                                       ("sd", "SD (all)", "num"), ("sd_clean", "SD (cleaned)", "num"),
                                       ("note", "Note", "text")])
    s_lab, c_lab = _OUTLIER_STAT[method]
    cols = [("group", "Group", "text"), ("row", "Row", "int"), ("value", "Value", "num"),
            ("stat", s_lab, "num"), ("crit", c_lab, "num")]
    if method.startswith("grubbs"):
        cols.append(("p", "P value", "p"))
    detail_rows = []
    masks = {}
    total = 0
    for name, (x, rows) in data.items():
        found, note = _find_outliers(x, method, alpha, int(max_out) if max_out else None, k)
        mask = np.zeros(len(x), bool)
        for f in found:
            mask[f["index"]] = True
            detail_rows.append(dict(group=name, row=int(rows[f["index"]]), value=float(x[f["index"]]),
                                    stat=f["stat"], crit=f["crit"], p=f["p"]))
        masks[name] = mask
        total += int(mask.sum())
        clean = x[~mask]
        summ.add_row(group=name, n=len(x), n_out=int(mask.sum()),
                     values=", ".join(fnum(v, 6) for v in x[mask]) if mask.any() else "—",
                     mean=float(x.mean()), mean_clean=float(clean.mean()) if len(clean) else math.nan,
                     sd=float(x.std(ddof=1)) if len(x) > 1 else math.nan,
                     sd_clean=float(clean.std(ddof=1)) if len(clean) > 1 else math.nan, note=note)
    notes = {
        "grubbs": f"Grubbs' test (extreme studentized deviate), two-sided, α = {alpha:g}. Tests only the single most extreme value in each group.",
        "grubbs_iter": f"Grubbs' test applied repeatedly (α = {alpha:g}): the most extreme value is removed and the test repeated until no outlier is found. "
                       "Repeated testing can miss outliers that mask each other; the generalized ESD test handles that better.",
        "gesd": f"Generalized extreme studentized deviate test (Rosner 1983), α = {alpha:g}.",
        "tukey": f"Values beyond {k:g} × IQR below the 25th or above the 75th percentile (percentiles by the (n+1)p rule).",
        "mad": "Modified Z score = 0.6745 × (value − median) / MAD; values with |Z| > 3.5 are flagged (Iglewicz & Hoaglin 1993).",
    }
    summ.note(notes[method])
    if method in ("grubbs", "grubbs_iter", "gesd"):
        summ.note("These tests assume that the values, apart from the outliers, come from a Gaussian distribution.")
    if detail_rows:
        det = r.table("Outliers found", cols)
        for row in detail_rows:
            det.add_row(**row)
        det.note("Row = row of the data table.")
    if total == 0:
        r.summary(f"No outliers were identified ({label}).")
    else:
        where = ", ".join(f"{n} ({int(m.sum())})" for n, m in masks.items() if m.any())
        r.summary(f"{total} outlier{'s' if total != 1 else ''} identified with {label}: {where}. "
                  "An outlier is not necessarily a mistake; check how the value arose before excluding it.")
    if ctx.opt("show_plot"):
        fig, ax = new_figure(max(3.2, min(9, 1.6 + 1.0 * len(data))), 3.6)
        for i, (name, (x, _)) in enumerate(data.items()):
            jit = i + jitter(len(x), seed=1 + i)
            m = masks[name]
            ax.scatter(jit[~m], x[~m], s=16, color=COLORS[0], alpha=0.7, linewidths=0, zorder=3)
            ax.scatter(jit[m], x[m], s=34, color=COLORS[1], marker="D", edgecolors="#222222",
                       linewidths=0.6, zorder=4, label="Outlier" if m.any() else None)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(list(data), rotation=0 if len(data) <= 4 else 30,
                           ha="center" if len(data) <= 4 else "right")
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.set_ylabel(value_label(ctx))
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:1], labels[:1], loc="best")
        r.plot("Values with outliers highlighted", fig)
    return r


# --------------------------------------------------------------------------
# Frequency distribution
# --------------------------------------------------------------------------

def _nice_width(raw: float) -> float:
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    exp = math.floor(math.log10(raw))
    frac = raw / 10 ** exp
    for nice in (1, 2, 2.5, 5, 10):
        if frac <= nice * (1 + 1e-9):
            return nice * 10 ** exp
    return 10 ** (exp + 1)


def _bin_edges(pooled: np.ndarray, width: float | None, n_bins: int | None, start: float | None) -> np.ndarray:
    lo, hi = float(pooled.min()), float(pooled.max())
    if width is None:
        if n_bins:
            if hi == lo:
                width = 1.0
            elif start is None:
                return np.linspace(lo, hi, int(n_bins) + 1)
            else:
                width = (hi - start) / int(n_bins)
        else:
            k = len(np.histogram_bin_edges(pooled, bins="auto")) - 1 if hi > lo else 1
            k = min(max(k, 1), 50)
            width = _nice_width((hi - lo) / k) if hi > lo else 1.0
    if width <= 0:
        raise AnalysisError("The bin width must be greater than zero.")
    if start is None:
        start = math.floor(lo / width + 1e-9) * width
    if start > lo:
        raise AnalysisError(f"The first bin starts at {start:g}, above the smallest value ({lo:g}). "
                            "Choose a smaller start or leave it empty.")
    k = int(math.floor((hi - start) / width + 1e-9)) + 1
    if n_bins and _ends_on_edge(hi, start, width, int(n_bins)):
        k = int(n_bins)
    if k > 500:
        raise AnalysisError(f"This bin width would create {k} bins. Choose a wider bin (at most 500 bins).")
    return start + width * np.arange(k + 1)


def _ends_on_edge(hi: float, start: float, width: float, n_bins: int) -> bool:
    """True when the largest value sits exactly on the last edge of ``n_bins`` bins."""
    return abs(start + width * n_bins - hi) <= 1e-9 * max(1.0, abs(hi))


@analysis(
    id="frequency", name="Frequency distribution", category=CAT,
    description="Tabulate how many values fall in each bin and draw a histogram.",
    keywords="histogram bins counts relative cumulative distribution",
    options=[
        *opt.group_layout(min_groups=1),
        opt.number("bin_width", "Bin width", None, min=0, section="Bins", placeholder="auto"),
        opt.integer("n_bins", "Number of bins", None, min=1, max=500, section="Bins", placeholder="auto",
                    help="Used when no bin width is given."),
        opt.number("bin_start", "First bin starts at", None, section="Bins", placeholder="auto"),
        opt.choice("tabulate", "Tabulate", [("count", "Number of values"),
                                            ("percent", "Relative frequency (%)"),
                                            ("fraction", "Relative frequency (fraction)")],
                   default="count", section="Options"),
        opt.boolean("cumulative", "Cumulative", False, section="Options"),
        opt.boolean("show_plot", "Histogram", True, section="Output"),
    ])
def frequency(ctx: Context) -> Results:
    groups = ctx.groups(min_groups=1)
    pooled = np.concatenate(list(groups.values()))
    width = ctx.opt("bin_width")
    if width is not None and float(width) <= 0:
        width = None
    edges = _bin_edges(pooled, None if width is None else float(width), ctx.opt("n_bins"),
                       None if ctx.opt("bin_start") is None else float(ctx.opt("bin_start")))
    mode = ctx.opt("tabulate", "count")
    cumulative = bool(ctx.opt("cumulative"))
    names = list(groups)
    k = len(edges) - 1
    table_vals = {}
    for name, x in groups.items():
        counts, _ = np.histogram(x, bins=edges)
        v = counts.astype(float)
        if mode != "count":
            v = v / len(x) * (100 if mode == "percent" else 1)
        if cumulative:
            v = np.cumsum(v)
        table_vals[name] = v
    unit = {"count": "number of values", "percent": "% of values", "fraction": "fraction of values"}[mode]
    r = Results()
    fmt = "int" if mode == "count" else "num"
    t = r.table(("Cumulative frequency" if cumulative else "Frequency") + f" distribution ({unit})",
                [("bin", "Bin", "text"), ("center", "Bin center", "num"),
                 *[(f"g{i}", n, fmt) for i, n in enumerate(names)]])
    for b in range(k):
        last = b == k - 1
        t.add_row(bin=f"{edges[b]:.6g} to {'≤' if last else '<'} {edges[b + 1]:.6g}",
                  center=float((edges[b] + edges[b + 1]) / 2),
                  **{f"g{i}": float(table_vals[n][b]) for i, n in enumerate(names)})
    t.note(f"{k} bins of width {edges[1] - edges[0]:.6g}. A value on a bin boundary is counted in the higher bin "
           "(the last bin includes its upper limit).")
    if mode != "count":
        t.note("Relative frequencies are computed within each group (n = "
               + ", ".join(f"{len(x)}" for x in groups.values()) + ").")
    modal = []
    for name, x in list(groups.items())[:4]:
        c, _ = np.histogram(x, bins=edges)
        b = int(np.argmax(c))
        modal.append(f"{name}: most values ({int(c[b])} of {len(x)}) fall between {edges[b]:.4g} and {edges[b + 1]:.4g}")
    r.summary("; ".join(modal) + ("; …" if len(groups) > 4 else "") + ".")

    if ctx.opt("show_plot"):
        fig, ax = new_figure(5.8, 3.6)
        g = len(names)
        w = (edges[1] - edges[0])
        for i, name in enumerate(names):
            if cumulative:
                ax.step(edges, np.concatenate([[0], table_vals[name]]), where="post",
                        color=COLORS[i % len(COLORS)], lw=1.6, label=name)
            else:
                bw = w / g * (0.9 if g > 1 else 1.0)
                ax.bar(edges[:-1] + (i + 0.5) * w / g, table_vals[name], width=bw,
                       color=COLORS[i % len(COLORS)], alpha=0.85 if g > 1 else 0.8,
                       edgecolor="white", linewidth=0.5, label=name)
        ax.set_xlabel(value_label(ctx))
        ax.set_ylabel(("Cumulative " if cumulative else "") + {"count": "number of values", "percent": "% of values",
                                                              "fraction": "fraction of values"}[mode])
        if g > 1:
            ax.legend()
        r.plot("Cumulative distribution" if cumulative else "Histogram", fig)
    return r


# --------------------------------------------------------------------------
# One-sample t test & Wilcoxon
# --------------------------------------------------------------------------

@analysis(
    id="one_sample", name="One-sample t test & Wilcoxon", category=CAT,
    description="Compare the mean (t test) or median (Wilcoxon signed-rank test) of each group with a hypothetical value.",
    keywords="one sample t test wilcoxon signed rank hypothetical value theoretical mean",
    options=[
        *opt.group_layout(min_groups=1),
        opt.number("mu", "Hypothetical value", 0, nullable=False),
        opt.boolean("t_test", "One-sample t test", True, section="Tests"),
        opt.boolean("wilcoxon", "Wilcoxon signed-rank test", True, section="Tests"),
        opt.choice("tails", "P value", [("two-sided", "Two-tailed"),
                                        ("greater", "One-tailed (mean/median > hypothetical)"),
                                        ("less", "One-tailed (mean/median < hypothetical)")],
                   default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_plot", "Plot of values vs. hypothetical value", True, section="Output"),
    ])
def one_sample(ctx: Context) -> Results:
    groups = ctx.groups(min_groups=1)
    mu = float(ctx.opt("mu", 0))
    alt = ctx.opt("tails", "two-sided")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    do_t, do_w = bool(ctx.opt("t_test")), bool(ctx.opt("wilcoxon"))
    if not (do_t or do_w):
        raise AnalysisError("Choose at least one test (t test or Wilcoxon).")
    if all(len(x) < 2 for x in groups.values()):
        raise AnalysisError("A one-sample test needs at least 2 values in a group.")
    r = Results()
    sentences = []
    skipped = []
    cl = pct(conf)
    if do_t:
        t = r.table(f"One-sample t test (hypothetical value = {mu:g})",
                    [("group", "Group", "text"), ("n", "n", "int"), ("mean", "Mean", "num"),
                     ("sd", "SD", "num"), ("sem", "SEM", "num"),
                     ("diff", "Discrepancy", "num"), ("lo", f"{cl} CI lower", "num"), ("hi", f"{cl} CI upper", "num"),
                     ("t", "t", "num"), ("df", "df", "int"), ("p", "P value", "p"), ("stars", "Summary", "text"),
                     ("d", "Cohen's d", "num")])
        for name, x in groups.items():
            d = describe(x, conf)
            if d["n"] < 2:
                skipped.append(f"{name} (n = {d['n']})")
                t.add_row(group=name, n=d["n"], mean=d["mean"], diff=d["mean"] - mu)
                continue
            if d["sd"] == 0 or is_constant(x):
                t.add_row(group=name, n=d["n"], mean=d["mean"], sd=0.0, sem=0.0, diff=d["mean"] - mu)
                skipped.append(f"{name} (all values identical)")
                continue
            tv = (d["mean"] - mu) / d["sem"]
            df = d["n"] - 1
            p = _t_pvalue(tv, df, alt)
            t.add_row(group=name, n=d["n"], mean=d["mean"], sd=d["sd"], sem=d["sem"], diff=d["mean"] - mu,
                      lo=d["ci_lo"] - mu, hi=d["ci_hi"] - mu, t=tv, df=df, p=p, stars=stars(p),
                      d=(d["mean"] - mu) / d["sd"])
            sentences.append(
                f"{name}: the mean ({fnum(d['mean'])}) "
                + ("differs significantly from" if p < alpha else "is not significantly different from")
                + f" {mu:g} ({p_text(p)}; one-sample t test; discrepancy {fnum(d['mean'] - mu)}, "
                  f"{cl} CI {fnum(d['ci_lo'] - mu)} to {fnum(d['ci_hi'] - mu)}).")
        t.note(f"{tails_label(alt)} P value. Discrepancy = mean − hypothetical value, with its {cl} confidence interval. "
               "Cohen's d = discrepancy / SD. Assumes the values are sampled from a Gaussian distribution.")
    if do_w:
        w = r.table(f"Wilcoxon signed-rank test (hypothetical value = {mu:g})",
                    [("group", "Group", "text"), ("n", "n", "int"), ("median", "Median", "num"),
                     ("diff", "Discrepancy", "num"),
                     ("lo", "CI of median, lower", "num"), ("hi", "CI of median, upper", "num"),
                     ("actual", "Actual confidence level", "pct"),
                     ("t_pos", "Sum of positive ranks", "num"), ("t_neg", "Sum of negative ranks", "num"),
                     ("w", "W (sum of signed ranks)", "num"), ("p", "P value", "p"),
                     ("stars", "Summary", "text"), ("method", "P value method", "text")])
        any_zero = False
        for name, x in groups.items():
            if len(x) < 2:
                if not do_t:
                    skipped.append(f"{name} (n = {len(x)})")
                w.add_row(group=name, n=len(x), median=float(np.median(x)))
                continue
            sr = signed_rank(x - mu, alt, conf)
            any_zero |= sr["n_zero"] > 0
            if sr["n_used"] == 0:
                w.add_row(group=name, n=len(x), median=sr["median"] + mu, diff=sr["median"],
                          method="all values equal the hypothetical value")
                continue
            w.add_row(group=name, n=len(x), median=sr["median"] + mu, diff=sr["median"],
                      lo=sr["med_lo"] + mu, hi=sr["med_hi"] + mu, actual=sr["med_conf"],
                      t_pos=sr["t_pos"], t_neg=-sr["t_neg"], w=sr["w"], p=sr["p"], stars=stars(sr["p"]),
                      method="Exact" if sr["exact"] else "Normal approximation")
            if not do_t:
                sentences.append(
                    f"{name}: the median ({fnum(sr['median'] + mu)}) "
                    + ("differs significantly from" if sr["p"] < alpha else "is not significantly different from")
                    + f" {mu:g} ({p_text(sr['p'])}; Wilcoxon signed-rank test).")
        w.note(f"{tails_label(alt)} P value; exact when there are no tied ranks and n ≤ 50, otherwise a normal "
               "approximation with tie and continuity correction.")
        w.note("The CI of the median is distribution-free (from order statistics), so its actual confidence level "
               f"is at least the requested {cl}; it is empty when n is too small.")
        if any_zero:
            w.note("Values equal to the hypothetical value were ignored (Wilcoxon's method).")
    if skipped:
        r.warn("Not tested: " + "; ".join(dict.fromkeys(skipped)) + ".")
    r.summary(" ".join(sentences[:6]) if sentences else "No group could be tested.")

    if ctx.opt("show_plot"):
        fig, ax = new_figure(max(3.2, min(9, 1.6 + 1.0 * len(groups))), 3.6)
        strip_axis(ax, groups)
        for i, x in enumerate(groups.values()):
            d = describe(x, conf)
            if d["n"] > 1:
                ax.errorbar(i + 0.3, d["mean"], yerr=[[d["mean"] - d["ci_lo"]], [d["ci_hi"] - d["mean"]]],
                            fmt="o", color="#222222", ms=4, capsize=3, lw=1.2, zorder=5)
        ax.axhline(mu, color=COLORS[1], ls="--", lw=1.1)
        ax.annotate(f"hypothetical = {mu:g}", (1, mu), xycoords=("axes fraction", "data"),
                    xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=8, color=COLORS[1])
        ax.set_ylabel(value_label(ctx))
        r.plot(f"Values with mean and {cl} CI", fig)
    return r


def _t_pvalue(t: float, df: float, alternative: str) -> float:
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    return float(2 * stats.t.sf(abs(t), df))
