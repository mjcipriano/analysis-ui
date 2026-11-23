"""Planning: sample size and power calculations (no data needed)."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import optimize, stats

from ..core import (AnalysisError, COLORS, Context, Results, analysis,
                    new_figure, opt)
from .curve_fitting import parse_numbers

CATEGORY = "Planning"
_TAILS = [("two", "Two-tailed"), ("one", "One-tailed")]
_N_MAX = 1e7


# ------------------------------------------------------------------ helpers
def _prob(ctx: Context, name: str, label: str, lo: float = 0.0, hi: float = 1.0) -> float:
    v = float(ctx.opt(name))
    if not (lo < v < hi):
        raise AnalysisError(f"{label} must be between {lo:g} and {hi:g} (exclusive).")
    return v


def _solve_n(power_at: Callable[[float], float], target: float, n_min: float) -> float:
    """Smallest (fractional) n with ``power_at(n) >= target``."""
    if power_at(n_min) >= target:
        return n_min
    hi = max(n_min * 2, 4.0)
    while power_at(hi) < target:
        hi *= 2
        if hi > _N_MAX:
            raise AnalysisError("The required sample size is larger than ten million; the effect is too small "
                                "to detect with these settings.")
    return float(optimize.brentq(lambda n: power_at(n) - target, n_min, hi, xtol=1e-10, rtol=1e-12))


def _solve_effect(power_at: Callable[[float], float], target: float, hi: float = 1.0, cap: float = 1e6) -> float:
    """Smallest effect with ``power_at(effect) >= target``."""
    lo = 1e-9
    while power_at(hi) < target:
        hi *= 2
        if hi > cap:
            raise AnalysisError("No effect size reaches the requested power with this sample size.")
    return float(optimize.brentq(lambda e: power_at(e) - target, lo, hi, xtol=1e-12))


def _curve(r: Results, title: str, ns: np.ndarray, powers: dict[str, np.ndarray], xlabel: str,
           mark: tuple[float, float] | None = None, target: float | None = None) -> None:
    fig, ax = new_figure(5.4, 3.6)
    for i, (label, pw) in enumerate(powers.items()):
        ax.plot(ns, pw, color=COLORS[i % len(COLORS)], lw=1.7, label=label if len(powers) > 1 else None)
    if target is not None:
        ax.axhline(target, color="#999999", lw=0.8, ls="--")
    if mark is not None and all(math.isfinite(v) for v in mark):
        ax.plot([mark[0]], [mark[1]], "o", ms=6, color=COLORS[1], mec="white", zorder=5)
        ax.axvline(mark[0], color="#bbbbbb", lw=0.8, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power")
    if len(powers) > 1:
        ax.legend(loc="lower right")
    r.plot(title, fig)


def _n_grid(n_star: float, n_min: float) -> np.ndarray:
    hi = max(n_min + 8, math.ceil(n_star * 2))
    return np.unique(np.round(np.linspace(n_min, hi, 80)))


def _kv_table(r: Results, title: str, rows: list[tuple[str, object, str]]):
    t = r.table(title, [("item", "", "text"), ("value", "Value", "num"), ("text", "", "text")])
    for label, value, extra in rows:
        if isinstance(value, str):
            t.add_row(item=label, text=value)
        else:
            t.add_row(item=label, value=value, text=extra)
    return t


# ===================================================================== t test
def ttest_power(d: float, n1: float, alpha: float, design: str = "two", ratio: float = 1.0,
                two_sided: bool = True) -> float:
    """Power of a t test from the noncentral t distribution (statsmodels)."""
    from statsmodels.stats.power import TTestIndPower, TTestPower
    alt = "two-sided" if two_sided else "larger"
    if design == "two":
        return float(TTestIndPower().power(abs(d), n1, alpha, ratio=ratio, alternative=alt))
    return float(TTestPower().power(abs(d), n1, alpha, alternative=alt))


@analysis(
    id="power_ttest", name="Sample size & power: t tests", category=CATEGORY,
    description="How many subjects are needed to compare means with a t test, or the power of a planned study.",
    keywords="power sample size t test effect size cohen d paired unpaired planning",
    options=[
        opt.choice("design", "Design", [("two", "Two independent groups (unpaired t test)"),
                                        ("paired", "Paired / matched (paired t test)"),
                                        ("one", "One sample vs. a hypothetical mean")], default="two"),
        opt.choice("solve_for", "Calculate", [("n", "Sample size"), ("power", "Power"),
                                              ("effect", "Smallest detectable difference")], default="n"),
        opt.choice("effect_mode", "Describe the effect as", [("raw", "Difference in means and SD"),
                                                             ("d", "Standardized effect size (Cohen's d)")],
                   default="raw", visible={"solve_for": ["n", "power"]}),
        opt.number("diff", "Difference in means to detect", 5.0, visible={"solve_for": ["n", "power"], "effect_mode": "raw"}),
        opt.number("sd", "Standard deviation", 10.0, min=0,
                   help="Within-group SD; for a paired design, the SD of the paired differences."),
        opt.number("d", "Effect size d", 0.5, visible={"solve_for": ["n", "power"], "effect_mode": "d"}),
        opt.number("power", "Power", 0.8, min=0.01, max=0.9999, step=0.05, visible={"solve_for": ["n", "effect"]}),
        opt.number("n", "Sample size (group 1 / pairs)", 20, min=2, visible={"solve_for": ["power", "effect"]}),
        opt.alpha(),
        opt.choice("tails", "Tails", _TAILS, default="two"),
        opt.number("ratio", "Allocation ratio (n₂ / n₁)", 1.0, min=0.01, max=100, visible={"design": "two"}),
        opt.boolean("show_plot", "Power curve", True, section="Output"),
    ])
def power_ttest(ctx: Context) -> Results:
    design, solve = ctx.opt("design", "two"), ctx.opt("solve_for", "n")
    alpha = _prob(ctx, "alpha", "The significance level")
    two_sided = ctx.opt("tails", "two") == "two"
    ratio = float(ctx.opt("ratio", 1.0)) if design == "two" else 1.0
    if ratio <= 0:
        raise AnalysisError("The allocation ratio must be positive.")
    sd = float(ctx.opt("sd", 10.0))
    raw = solve == "effect" or ctx.opt("effect_mode", "raw") == "raw"
    if raw and sd <= 0:
        raise AnalysisError("The standard deviation must be greater than zero.")
    n_min = 2.0 if design != "two" else max(2.0, 2.0 / ratio)

    def pw(d, n):
        return ttest_power(d, n, alpha, design, ratio, two_sided)

    d = n1 = power = float("nan")
    if solve != "effect":
        d = float(ctx.opt("diff", 5.0)) / sd if raw else float(ctx.opt("d", 0.5))
        if d == 0 or not math.isfinite(d):
            raise AnalysisError("The difference to detect must not be zero: a zero difference can never be detected.")
    if solve == "n":
        power = _prob(ctx, "power", "Power")
        if power <= alpha:
            raise AnalysisError("The power must be larger than the significance level.")
        n1 = _solve_n(lambda n: pw(d, n), power, n_min)
    else:
        n1 = float(ctx.opt("n", 20))
        if n1 < n_min:
            raise AnalysisError(f"The sample size must be at least {math.ceil(n_min)}.")
        if solve == "power":
            power = pw(d, n1)
        else:
            power = _prob(ctx, "power", "Power")
            if power <= alpha:
                raise AnalysisError("The power must be larger than the significance level.")
            d = _solve_effect(lambda e: pw(e, n1), power)

    unit = {"two": "per group", "paired": "pairs", "one": "subjects"}[design]
    name = {"two": "Unpaired t test (two independent groups)", "paired": "Paired t test", "one": "One-sample t test"}[design]
    r = Results()
    rows: list[tuple[str, object, str]] = [("Test", name, ""),
                                           ("Significance level (α)", alpha, "two-tailed" if two_sided else "one-tailed"),
                                           ("Effect size (Cohen's d)", abs(d), "")]
    if raw:
        rows.append(("Difference in means", abs(d) * sd, "detectable difference" if solve == "effect" else ""))
        rows.append(("Standard deviation", sd, "of the paired differences" if design == "paired" else ""))
    if solve == "n":
        n1c = math.ceil(n1 - 1e-9)
        n2c = math.ceil(n1c * ratio - 1e-9) if design == "two" else 0
        rows.append(("Target power", power, ""))
        rows.append((f"Required sample size ({unit if design != 'two' else 'group 1'})", n1c, f"exact solution {n1:.2f}"))
        if design == "two":
            rows.append(("Required sample size (group 2)", n2c, ""))
            rows.append(("Total sample size", n1c + n2c, ""))
        rows.append(("Actual power with these sample sizes", pw(d, n1c), ""))
        concl = (f"To detect {'a difference of ' + format(abs(d) * sd, '.4g') + ' (SD ' + format(sd, '.4g') + ')' if raw else 'an effect size of d = ' + format(abs(d), '.3g')} "
                 f"with {100 * power:.0f}% power at α = {alpha:g} ({'two' if two_sided else 'one'}-tailed), "
                 + (f"{n1c} subjects in group 1 and {n2c} in group 2 are needed ({n1c + n2c} in total)." if design == "two" and ratio != 1
                    else f"{n1c} subjects per group are needed ({2 * n1c} in total)." if design == "two"
                    else f"{n1c} {unit} are needed."))
    else:
        rows.append((f"Sample size ({unit if design != 'two' else 'group 1'})", n1, ""))
        if design == "two":
            rows.append(("Sample size (group 2)", n1 * ratio, ""))
        rows.append(("Power", power, "" if solve == "power" else "target"))
        if solve == "power":
            concl = (f"With {n1:g} {unit}, the study has {100 * power:.1f}% power to detect "
                     f"{'a difference of ' + format(abs(d) * sd, '.4g') if raw else 'an effect size of d = ' + format(abs(d), '.3g')} "
                     f"at α = {alpha:g} ({'two' if two_sided else 'one'}-tailed).")
        else:
            concl = (f"With {n1:g} {unit}, the smallest difference detectable with {100 * power:.0f}% power at "
                     f"α = {alpha:g} is {abs(d) * sd:.4g} (d = {abs(d):.3g}).")
    t = _kv_table(r, "Sample size and power: t test", rows)
    t.note("Power from the noncentral t distribution (statsmodels), as R's power.t.test. "
           "Sample sizes are rounded up to whole subjects.")
    r.summary(concl)
    if ctx.opt("show_plot", True):
        ns = _n_grid(n1, math.ceil(n_min))
        _curve(r, "Power curve", ns, {"power": np.array([pw(d, n) for n in ns])},
               f"Sample size ({unit if design != 'two' else 'group 1'})",
               mark=(math.ceil(n1 - 1e-9) if solve == "n" else n1, pw(d, math.ceil(n1 - 1e-9) if solve == "n" else n1)),
               target=power if solve != "power" else None)
    return r


# ====================================================================== ANOVA
def anova_power(f: float, n_per_group: float, k: int, alpha: float) -> float:
    """Power of the one-way ANOVA F test (noncentral F; Cohen's f)."""
    df1, df2 = k - 1, k * (n_per_group - 1)
    if df2 <= 0:
        return float("nan")
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(crit, df1, df2, f * f * k * n_per_group))


@analysis(
    id="power_anova", name="Sample size & power: one-way ANOVA", category=CATEGORY,
    description="Subjects per group needed for a one-way ANOVA, or the power of a planned study.",
    keywords="power sample size anova effect size cohen f groups planning",
    options=[
        opt.choice("solve_for", "Calculate", [("n", "Sample size per group"), ("power", "Power"),
                                              ("effect", "Smallest detectable effect size")], default="n"),
        opt.choice("effect_mode", "Describe the effect as", [("f", "Effect size f (Cohen)"),
                                                             ("means", "Expected group means and common SD")],
                   default="f", visible={"solve_for": ["n", "power"]}),
        opt.integer("k", "Number of groups", 3, min=2, max=100,
                    help="Ignored when you type the expected group means."),
        opt.number("f", "Effect size f", 0.25, min=0, visible={"solve_for": ["n", "power"], "effect_mode": "f"},
                   help="Conventionally 0.10 = small, 0.25 = medium, 0.40 = large."),
        opt.text("means", "Expected group means", "10, 12, 15", visible={"solve_for": ["n", "power"], "effect_mode": "means"}),
        opt.number("sd", "Common within-group SD", 5.0, min=0, visible={"solve_for": ["n", "power"], "effect_mode": "means"}),
        opt.number("power", "Power", 0.8, min=0.01, max=0.9999, step=0.05, visible={"solve_for": ["n", "effect"]}),
        opt.number("n", "Sample size per group", 20, min=2, visible={"solve_for": ["power", "effect"]}),
        opt.alpha(),
        opt.boolean("show_plot", "Power curve", True, section="Output"),
    ])
def power_anova(ctx: Context) -> Results:
    solve = ctx.opt("solve_for", "n")
    alpha = _prob(ctx, "alpha", "The significance level")
    use_means = solve != "effect" and ctx.opt("effect_mode", "f") == "means"
    k = int(ctx.opt("k", 3))
    f = float("nan")
    rows: list[tuple[str, object, str]] = [("Test", "One-way ANOVA (F test, equal group sizes)", "")]
    if use_means:
        means = parse_numbers(ctx.opt("means", ""), "group means")
        sd = float(ctx.opt("sd", 5.0))
        if len(means) < 2:
            raise AnalysisError("Type at least two expected group means, separated by commas.")
        if sd <= 0:
            raise AnalysisError("The standard deviation must be greater than zero.")
        k = len(means)
        f = float(np.std(means)) / sd
        rows += [("Expected group means", ", ".join(f"{m:g}" for m in means), ""), ("Common SD", sd, "")]
    elif solve != "effect":
        f = float(ctx.opt("f", 0.25))
    if solve != "effect" and not f > 0:
        raise AnalysisError("The effect size must be greater than zero"
                            + (" (the expected group means are all equal)." if use_means else "."))
    rows.append(("Number of groups", k, ""))
    rows.append(("Significance level (α)", alpha, ""))
    if solve == "n":
        power = _prob(ctx, "power", "Power")
        if power <= alpha:
            raise AnalysisError("The power must be larger than the significance level.")
        n = _solve_n(lambda m: anova_power(f, m, k, alpha), power, 2.0)
        nc = math.ceil(n - 1e-9)
        rows += [("Effect size f", f, ""), ("Target power", power, ""),
                 ("Required sample size per group", nc, f"exact solution {n:.2f}"),
                 ("Total sample size", nc * k, ""), ("Actual power", anova_power(f, nc, k, alpha), "")]
        concl = (f"To detect an effect size of f = {f:.3g} among {k} groups with {100 * power:.0f}% power at "
                 f"α = {alpha:g}, {nc} subjects per group are needed ({nc * k} in total).")
        mark_n = nc
    else:
        n = float(ctx.opt("n", 20))
        if n < 2:
            raise AnalysisError("At least 2 subjects per group are needed.")
        if solve == "power":
            power = anova_power(f, n, k, alpha)
            concl = (f"With {n:g} subjects in each of {k} groups, the ANOVA has {100 * power:.1f}% power to detect "
                     f"an effect size of f = {f:.3g} at α = {alpha:g}.")
        else:
            power = _prob(ctx, "power", "Power")
            if power <= alpha:
                raise AnalysisError("The power must be larger than the significance level.")
            f = _solve_effect(lambda e: anova_power(e, n, k, alpha), power)
            concl = (f"With {n:g} subjects in each of {k} groups, the smallest effect detectable with "
                     f"{100 * power:.0f}% power at α = {alpha:g} is f = {f:.3g}.")
        rows += [("Effect size f", f, "detectable" if solve == "effect" else ""), ("Sample size per group", n, ""),
                 ("Total sample size", n * k, ""), ("Power", power, "" if solve == "power" else "target")]
        mark_n = n
    r = Results()
    t = _kv_table(r, "Sample size and power: one-way ANOVA", rows)
    t.note("Power from the noncentral F distribution with noncentrality f² × total N (as R's pwr.anova.test). "
           "f = SD of the group means ÷ common within-group SD; η² = f² / (1 + f²)"
           + (f" = {f * f / (1 + f * f):.3g}." if f == f else "."))
    r.summary(concl)
    if ctx.opt("show_plot", True):
        ns = _n_grid(n, 2)
        _curve(r, "Power curve", ns, {"power": np.array([anova_power(f, m, k, alpha) for m in ns])},
               "Sample size per group", mark=(mark_n, anova_power(f, mark_n, k, alpha)),
               target=power if solve != "power" else None)
    return r


# ================================================================ proportions
def _z_alpha(alpha: float, two_sided: bool) -> float:
    return float(stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))


def prop_power_arcsine(p1: float, p2: float, n1: float, alpha: float, ratio: float = 1.0, two_sided: bool = True) -> float:
    """Power using Cohen's effect size h = 2·asin√p1 − 2·asin√p2."""
    h = abs(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))
    z = h * math.sqrt(n1 * ratio / (1 + ratio))
    za = _z_alpha(alpha, two_sided)
    return float(stats.norm.sf(za - z) + (stats.norm.cdf(-za - z) if two_sided else 0.0))


def prop_power_pooled(p1: float, p2: float, n1: float, alpha: float, ratio: float = 1.0, two_sided: bool = True) -> float:
    """Power of the z (chi-square) test with pooled variance under the null."""
    pbar = (p1 + ratio * p2) / (1 + ratio)
    za = _z_alpha(alpha, two_sided)
    num = abs(p1 - p2) * math.sqrt(n1) - za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio))
    return float(stats.norm.cdf(num / math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)))


def prop_n_pooled(p1: float, p2: float, alpha: float, power: float, ratio: float = 1.0, two_sided: bool = True) -> float:
    pbar = (p1 + ratio * p2) / (1 + ratio)
    za, zb = _z_alpha(alpha, two_sided), float(stats.norm.ppf(power))
    return (za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio))
            + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2 / (p1 - p2) ** 2


def prop_n_cc(n: float, p1: float, p2: float, ratio: float = 1.0) -> float:
    """Fleiss' continuity correction of a group-1 sample size."""
    return n / 4 * (1 + math.sqrt(1 + 2 * (ratio + 1) / (n * ratio * abs(p1 - p2)))) ** 2


def prop_n_uncc(n_cc: float, p1: float, p2: float, ratio: float = 1.0) -> float:
    """Inverse of :func:`prop_n_cc`."""
    a = (ratio + 1) / (ratio * abs(p1 - p2))
    root = math.sqrt(n_cc) - a / (2 * math.sqrt(n_cc))
    return root * root if root > 0 else 0.0


@analysis(
    id="power_proportions", name="Sample size & power: two proportions", category=CATEGORY,
    description="Subjects needed to compare two proportions (chi-square / Fisher), or the power of a planned study.",
    keywords="power sample size proportions chi-square fisher binary response rate planning",
    options=[
        opt.choice("solve_for", "Calculate", [("n", "Sample size"), ("power", "Power")], default="n"),
        opt.number("p1", "Expected proportion in group 1", 0.3, min=0, max=1, step=0.05),
        opt.number("p2", "Expected proportion in group 2", 0.5, min=0, max=1, step=0.05),
        opt.number("power", "Power", 0.8, min=0.01, max=0.9999, step=0.05, visible={"solve_for": "n"}),
        opt.number("n", "Sample size of group 1", 50, min=2, visible={"solve_for": "power"}),
        opt.alpha(),
        opt.choice("tails", "Tails", _TAILS, default="two"),
        opt.number("ratio", "Allocation ratio (n₂ / n₁)", 1.0, min=0.01, max=100),
        opt.choice("method", "Method to highlight", [
            ("pooled", "Normal approximation, pooled variance (chi-square test)"),
            ("cc", "The same with continuity correction (≈ Fisher's exact test)"),
            ("arcsine", "Arcsine transformation (Cohen's h)")], default="pooled"),
        opt.boolean("show_plot", "Power curve", True, section="Output"),
    ])
def power_proportions(ctx: Context) -> Results:
    p1 = _prob(ctx, "p1", "The proportion in group 1")
    p2 = _prob(ctx, "p2", "The proportion in group 2")
    if p1 == p2:
        raise AnalysisError("The two proportions are equal, so there is no difference to detect.")
    alpha = _prob(ctx, "alpha", "The significance level")
    two_sided = ctx.opt("tails", "two") == "two"
    ratio = float(ctx.opt("ratio", 1.0))
    solve, method = ctx.opt("solve_for", "n"), ctx.opt("method", "pooled")
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    labels = {"pooled": "Normal approximation (pooled variance)", "cc": "With continuity correction",
              "arcsine": "Arcsine transformation (Cohen's h)"}

    def power_by(m, n1):
        if m == "arcsine":
            return prop_power_arcsine(p1, p2, n1, alpha, ratio, two_sided)
        if m == "cc":
            n1 = prop_n_uncc(n1, p1, p2, ratio)
            if n1 <= 0:
                return 0.0
        return prop_power_pooled(p1, p2, n1, alpha, ratio, two_sided)

    r = Results()
    _kv_table(r, "Sample size and power: two proportions", [
        ("Proportion in group 1", p1, ""), ("Proportion in group 2", p2, ""),
        ("Difference (p₁ − p₂)", p1 - p2, ""), ("Effect size h", abs(h), "arcsine difference"),
        ("Significance level (α)", alpha, "two-tailed" if two_sided else "one-tailed"),
        ("Allocation ratio (n₂ / n₁)", ratio, "")])
    if solve == "n":
        power = _prob(ctx, "power", "Power")
        if power <= alpha:
            raise AnalysisError("The power must be larger than the significance level.")
        n_pooled = prop_n_pooled(p1, p2, alpha, power, ratio, two_sided)
        za, zb = _z_alpha(alpha, two_sided), float(stats.norm.ppf(power))
        exact = {"pooled": n_pooled, "cc": prop_n_cc(n_pooled, p1, p2, ratio),
                 "arcsine": ((za + zb) / h) ** 2 * (1 + ratio) / ratio}
        if two_sided:   # include the (tiny) far tail, as pwr.2p.test does
            exact["arcsine"] = _solve_n(lambda n: prop_power_arcsine(p1, p2, n, alpha, ratio, True), power, 1.0)
        t = r.table("Required sample size", [
            ("method", "Method", "text"), ("exact", "Exact solution (group 1)", "num"), ("n1", "Group 1", "int"),
            ("n2", "Group 2", "int"), ("total", "Total", "int"), ("mark", "", "text")])
        for m in ("pooled", "cc", "arcsine"):
            n1c = math.ceil(exact[m] - 1e-9)
            n2c = math.ceil(n1c * ratio - 1e-9)
            t.add_row(method=labels[m], exact=exact[m], n1=n1c, n2=n2c, total=n1c + n2c,
                      mark="◄ highlighted" if m == method else "", key=m)
        n1 = exact[method]
        n1c = math.ceil(n1 - 1e-9)
        n2c = math.ceil(n1c * ratio - 1e-9)
        concl = (f"To detect a difference between proportions of {p1:g} and {p2:g} with {100 * power:.0f}% power at "
                 f"α = {alpha:g} ({'two' if two_sided else 'one'}-tailed), {n1c} subjects are needed in group 1 and "
                 f"{n2c} in group 2 ({n1c + n2c} in total; {labels[method].lower()}).")
    else:
        n1 = float(ctx.opt("n", 50))
        t = r.table("Power", [("method", "Method", "text"), ("power", "Power", "pct"), ("mark", "", "text")])
        for m in ("pooled", "cc", "arcsine"):
            t.add_row(method=labels[m], power=power_by(m, n1), mark="◄ highlighted" if m == method else "", key=m)
        power = power_by(method, n1)
        concl = (f"With {n1:g} subjects in group 1 and {n1 * ratio:g} in group 2, the study has {100 * power:.1f}% power "
                 f"to detect a difference between proportions of {p1:g} and {p2:g} at α = {alpha:g} "
                 f"({labels[method].lower()}).")
    t.note("Pooled variance: n₁ = [z₍α₎·√(p̄q̄(1 + 1/κ)) + z₍β₎·√(p₁q₁ + p₂q₂/κ)]² / (p₁ − p₂)², κ = n₂/n₁ "
           "(as R's power.prop.test for equal groups). Continuity correction after Fleiss, which approximates the "
           "sample size for Fisher's exact test. Arcsine: Cohen's h with the normal distribution (as R's pwr.2p.test).")
    r.summary(concl)
    if ctx.opt("show_plot", True):
        ns = _n_grid(n1, 2)
        _curve(r, "Power curve", ns, {labels[m]: np.array([power_by(m, n) for n in ns]) for m in ("pooled", "cc", "arcsine")},
               "Sample size of group 1",
               mark=(math.ceil(n1 - 1e-9) if solve == "n" else n1, power_by(method, math.ceil(n1 - 1e-9) if solve == "n" else n1)),
               target=power if solve == "n" else None)
    return r


# =================================================================== survival
def logrank_events(hr: float, alpha: float, power: float, ratio: float = 1.0, two_sided: bool = True) -> float:
    """Total events needed by the log-rank test (Schoenfeld, 1981)."""
    za, zb = _z_alpha(alpha, two_sided), float(stats.norm.ppf(power))
    return (za + zb) ** 2 * (1 + ratio) ** 2 / (ratio * math.log(hr) ** 2)


def logrank_power(hr: float, events: float, alpha: float, ratio: float = 1.0, two_sided: bool = True) -> float:
    z = abs(math.log(hr)) * math.sqrt(events * ratio) / (1 + ratio)
    return float(stats.norm.cdf(z - _z_alpha(alpha, two_sided)))


@analysis(
    id="power_survival", name="Sample size & power: survival (log-rank)", category=CATEGORY,
    description="Number of events and subjects needed to compare two survival curves with the log-rank test.",
    keywords="power sample size survival log-rank hazard ratio events schoenfeld freedman planning",
    options=[
        opt.choice("solve_for", "Calculate", [("n", "Number of events and subjects"), ("power", "Power")], default="n"),
        opt.choice("effect_mode", "Describe the effect as", [("hr", "Hazard ratio"), ("medians", "Two median survival times")],
                   default="hr"),
        opt.number("hr", "Hazard ratio to detect", 0.6, min=0.001, max=1000, visible={"effect_mode": "hr"}),
        opt.number("median1", "Median survival, control group", 12.0, min=0, visible={"effect_mode": "medians"}),
        opt.number("median2", "Median survival, treatment group", 20.0, min=0, visible={"effect_mode": "medians"}),
        opt.number("power", "Power", 0.8, min=0.01, max=0.9999, step=0.05, visible={"solve_for": "n"}),
        opt.number("events", "Total number of events", 100, min=1, visible={"solve_for": "power"}),
        opt.alpha(),
        opt.choice("tails", "Tails", _TAILS, default="two"),
        opt.number("ratio", "Allocation ratio (n₂ / n₁)", 1.0, min=0.01, max=100),
        opt.number("event_prob", "Expected overall probability of the event", None, min=0.0001, max=1,
                   placeholder="optional", help="The fraction of all subjects expected to have the event during "
                                                "the study. Needed to convert events into subjects."),
        opt.boolean("show_plot", "Power curve", True, section="Output"),
    ])
def power_survival(ctx: Context) -> Results:
    alpha = _prob(ctx, "alpha", "The significance level")
    two_sided = ctx.opt("tails", "two") == "two"
    ratio = float(ctx.opt("ratio", 1.0))
    rows: list[tuple[str, object, str]] = [("Test", "Log-rank test (two groups)", "")]
    if ctx.opt("effect_mode", "hr") == "medians":
        m1, m2 = float(ctx.opt("median1", 12.0)), float(ctx.opt("median2", 20.0))
        if m1 <= 0 or m2 <= 0:
            raise AnalysisError("Median survival times must be greater than zero.")
        hr = m1 / m2
        rows += [("Median survival, control", m1, ""), ("Median survival, treatment", m2, ""),
                 ("Hazard ratio (treatment / control)", hr, "assuming exponential survival")]
    else:
        hr = float(ctx.opt("hr", 0.6))
        if hr <= 0:
            raise AnalysisError("The hazard ratio must be greater than zero.")
        rows.append(("Hazard ratio", hr, ""))
    if abs(math.log(hr)) < 1e-12:
        raise AnalysisError("A hazard ratio of 1 (equal survival) means there is no difference to detect.")
    rows += [("Significance level (α)", alpha, "two-tailed" if two_sided else "one-tailed"),
             ("Allocation ratio (n₂ / n₁)", ratio, "")]
    pe = ctx.opt("event_prob", None)
    solve = ctx.opt("solve_for", "n")
    if solve == "n":
        power = _prob(ctx, "power", "Power")
        if power <= alpha:
            raise AnalysisError("The power must be larger than the significance level.")
        d = logrank_events(hr, alpha, power, ratio, two_sided)
        za, zb = _z_alpha(alpha, two_sided), float(stats.norm.ppf(power))
        d_freedman = (za + zb) ** 2 * (1 + ratio * hr) ** 2 / (ratio * (1 - hr) ** 2)
        dc = math.ceil(d - 1e-9)
        rows += [("Target power", power, ""), ("Required number of events (Schoenfeld)", dc, f"exact solution {d:.2f}"),
                 ("Required number of events (Freedman)", math.ceil(d_freedman - 1e-9), f"exact solution {d_freedman:.2f}")]
        concl = (f"To detect a hazard ratio of {hr:.3g} with {100 * power:.0f}% power at α = {alpha:g} "
                 f"({'two' if two_sided else 'one'}-tailed), {dc} events are needed in total (Schoenfeld)")
        if pe:
            n_tot = d / float(pe)
            n1c = math.ceil(n_tot / (1 + ratio) - 1e-9)
            n2c = math.ceil(n_tot * ratio / (1 + ratio) - 1e-9)
            rows += [("Expected probability of the event", float(pe), ""), ("Required subjects, group 1", n1c, ""),
                     ("Required subjects, group 2", n2c, ""), ("Total subjects", n1c + n2c, "")]
            concl += f"; with {100 * float(pe):.0f}% of subjects expected to have the event, enrol {n1c + n2c} subjects."
        else:
            concl += ". Enter the expected probability of the event to convert events into a number of subjects."
        mark = (dc, logrank_power(hr, dc, alpha, ratio, two_sided))
        d_ref = d
    else:
        d = float(ctx.opt("events", 100))
        power = logrank_power(hr, d, alpha, ratio, two_sided)
        rows += [("Total number of events", d, ""), ("Power", power, "")]
        if pe:
            rows.append(("Total subjects implied", math.ceil(d / float(pe) - 1e-9), f"event probability {float(pe):g}"))
        concl = (f"With {d:g} events, the log-rank test has {100 * power:.1f}% power to detect a hazard ratio of "
                 f"{hr:.3g} at α = {alpha:g}.")
        mark = (d, power)
        d_ref = d
    r = Results()
    t = _kv_table(r, "Sample size and power: log-rank test", rows)
    t.note("Schoenfeld (1981): events = (z₍α₎ + z₍β₎)² (1 + κ)² / (κ (ln HR)²), κ = n₂/n₁. Power depends on the "
           "number of events, not the number of subjects. Assumes proportional hazards. "
           "Freedman (1982) is shown for comparison.")
    r.summary(concl)
    if ctx.opt("show_plot", True):
        ds = _n_grid(d_ref, 2)
        _curve(r, "Power curve", ds, {"power": np.array([logrank_power(hr, x, alpha, ratio, two_sided) for x in ds])},
               "Total number of events", mark=mark, target=power if solve == "n" else None)
    return r


# ================================================================ correlation
def correlation_power(rho: float, n: float, alpha: float, two_sided: bool = True) -> float:
    """Power of the t test of r = 0, by the algorithm of R's ``pwr.r.test``
    (Cohen 1988): the critical r comes from the t distribution with n − 2 df,
    and the sampling distribution of r is approximated on the Fisher z scale
    with the bias correction r / (2(n − 1))."""
    r = abs(rho)
    tcrit = stats.t.isf(alpha / 2 if two_sided else alpha, n - 2)
    rc = math.sqrt(tcrit ** 2 / (tcrit ** 2 + n - 2))
    zr = math.atanh(r) + r / (2 * (n - 1))
    zrc = math.atanh(rc)
    root = math.sqrt(n - 3)
    power = stats.norm.cdf((zr - zrc) * root)
    if two_sided:
        power += stats.norm.cdf((-zr - zrc) * root)
    return float(power)


@analysis(
    id="power_correlation", name="Sample size & power: correlation", category=CATEGORY,
    description="Subjects needed to show that a correlation differs from zero, or the power of a planned study.",
    keywords="power sample size correlation pearson r fisher z planning",
    options=[
        opt.choice("solve_for", "Calculate", [("n", "Sample size"), ("power", "Power")], default="n"),
        opt.number("r", "Expected correlation (r)", 0.3, min=-0.999, max=0.999, step=0.05),
        opt.number("power", "Power", 0.8, min=0.01, max=0.9999, step=0.05, visible={"solve_for": "n"}),
        opt.number("n", "Sample size", 50, min=4, visible={"solve_for": "power"}),
        opt.alpha(),
        opt.choice("tails", "Tails", _TAILS, default="two"),
        opt.boolean("show_plot", "Power curve", True, section="Output"),
    ])
def power_correlation(ctx: Context) -> Results:
    rho = float(ctx.opt("r", 0.3))
    if rho == 0 or abs(rho) >= 1:
        raise AnalysisError("The expected correlation must be between −1 and 1 and not zero.")
    alpha = _prob(ctx, "alpha", "The significance level")
    two_sided = ctx.opt("tails", "two") == "two"
    rows: list[tuple[str, object, str]] = [
        ("Test", "Pearson correlation different from zero (t test)", ""), ("Expected correlation (r)", rho, ""),
        ("Significance level (α)", alpha, "two-tailed" if two_sided else "one-tailed")]
    solve = ctx.opt("solve_for", "n")
    if solve == "n":
        power = _prob(ctx, "power", "Power")
        if power <= alpha:
            raise AnalysisError("The power must be larger than the significance level.")
        n = _solve_n(lambda m: correlation_power(rho, m, alpha, two_sided), power, 4.0)
        nc = max(4, math.ceil(n - 1e-9))
        rows += [("Target power", power, ""), ("Required sample size", nc, f"exact solution {n:.2f}"),
                 ("Actual power", correlation_power(rho, nc, alpha, two_sided), "")]
        concl = (f"To detect a correlation of r = {rho:g} with {100 * power:.0f}% power at α = {alpha:g} "
                 f"({'two' if two_sided else 'one'}-tailed), {nc} subjects are needed.")
        mark_n = nc
    else:
        n = float(ctx.opt("n", 50))
        if n < 4:
            raise AnalysisError("At least 4 subjects are needed.")
        power = correlation_power(rho, n, alpha, two_sided)
        rows += [("Sample size", n, ""), ("Power", power, "")]
        concl = (f"With {n:g} subjects, the study has {100 * power:.1f}% power to detect a correlation of "
                 f"r = {rho:g} at α = {alpha:g}.")
        mark_n = n
    r = Results()
    t = _kv_table(r, "Sample size and power: correlation", rows)
    t.note("Method of Cohen (1988) as implemented in R's pwr.r.test: the critical correlation is taken from the "
           "t distribution with n − 2 df, and power from the Fisher z transformation of r with the bias "
           "correction r / (2(n − 1)).")
    r.summary(concl)
    if ctx.opt("show_plot", True):
        ns = _n_grid(n, 4)
        _curve(r, "Power curve", ns, {"power": np.array([correlation_power(rho, m, alpha, two_sided) for m in ns])},
               "Sample size", mark=(mark_n, correlation_power(rho, mark_n, alpha, two_sided)),
               target=power if solve == "n" else None)
    return r
