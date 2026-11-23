"""Survival analysis: Kaplan–Meier estimates and comparison of survival curves."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from ..core import (AnalysisError, COLORS, Context, Results, analysis, fmt_p,
                    new_figure, opt, stars)
from ..graphs.types import event_indicator
from .curve_fitting import p_phrase, parse_numbers

Z95 = stats.norm.ppf(0.975)


# ------------------------------------------------------------- computations
def km_table(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """Kaplan–Meier estimate at every distinct observed time.

    Columns: time, at_risk, events, censored, surv, se (Greenwood), lo, hi
    (95 % CI, log(−log) transform — NaN once survival reaches 0 or while it is 1)."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    rows = []
    s, var_sum = 1.0, 0.0
    for ut in np.unique(t):
        n = int(np.sum(t >= ut))
        d = int(np.sum((t == ut) & e))
        c = int(np.sum((t == ut) & ~e))
        if d:
            s *= 1 - d / n
            var_sum = var_sum + d / (n * (n - d)) if n > d else float("inf")
        if 0 < s < 1 and math.isfinite(var_sum):
            se = s * math.sqrt(var_sum)
            g = Z95 * math.sqrt(var_sum) / abs(math.log(s))
            lo, hi = s ** math.exp(g), s ** math.exp(-g)
        elif s == 1.0:
            se, lo, hi = 0.0, 1.0, 1.0
        else:
            se = lo = hi = float("nan")
        rows.append((float(ut), n, d, c, s, se, lo, hi))
    return pd.DataFrame(rows, columns=["time", "at_risk", "events", "censored", "surv", "se", "lo", "hi"])


def _first_at_or_below(times: np.ndarray, curve: np.ndarray, level: float) -> float:
    idx = np.where(curve <= level + 1e-12)[0]
    return float(times[idx[0]]) if len(idx) else float("nan")


def median_survival(km: pd.DataFrame) -> tuple[float, float, float]:
    """Median survival and its 95 % CI (Brookmeyer & Crowley, using the
    log(−log) confidence band). NaN = not reached / undefined."""
    ev = km[km["events"] > 0]
    times, surv = ev["time"].to_numpy(), ev["surv"].to_numpy()
    idx = np.where(surv <= 0.5 + 1e-12)[0]
    if len(idx) == 0:
        med = float("nan")
    else:
        i = idx[0]
        med = float(times[i])
        if abs(surv[i] - 0.5) < 1e-12 and i + 1 < len(times):
            med = (med + float(times[i + 1])) / 2      # curve is flat at exactly 50 %
    lower = _first_at_or_below(times, ev["lo"].to_numpy(), 0.5)
    upper = _first_at_or_below(times, ev["hi"].to_numpy(), 0.5)
    return med, lower, upper


def restricted_mean(km: pd.DataFrame, tau: float) -> tuple[float, float]:
    """Area under the KM curve up to ``tau`` and its standard error."""
    ev = km[(km["events"] > 0) & (km["time"] <= tau)]
    times = np.concatenate([[0.0], ev["time"].to_numpy(), [tau]])
    surv = np.concatenate([[1.0], ev["surv"].to_numpy()])
    widths = np.diff(times)
    rmst = float(np.sum(widths * surv))
    # area to the right of each event time
    tail = np.cumsum((widths * surv)[::-1])[::-1][1:]
    n, d = ev["at_risk"].to_numpy(dtype=float), ev["events"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, tail ** 2 * d / (n * (n - d)), 0.0)
    return rmst, float(np.sqrt(np.sum(terms)))


def weighted_logrank(time, event, codes, k: int, weights: str = "logrank") -> dict:
    """K-sample (weighted) log-rank test.

    ``weights``: "logrank" (1, Mantel–Cox), "gehan" (number at risk,
    Gehan–Breslow–Wilcoxon) or "tarone" (√ number at risk, Tarone–Ware)."""
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    g = np.asarray(codes, dtype=int)
    U = np.zeros(k)
    V = np.zeros((k, k))
    O = np.zeros(k)
    E = np.zeros(k)
    for ut in np.unique(t[e]):
        at = t >= ut
        n = float(at.sum())
        nj = np.bincount(g[at], minlength=k).astype(float)
        dj = np.bincount(g[(t == ut) & e], minlength=k).astype(float)
        d = float(dj.sum())
        w = 1.0 if weights == "logrank" else n if weights == "gehan" else math.sqrt(n)
        ej = d * nj / n
        O += dj
        E += ej
        U += w * (dj - ej)
        if n > 1:
            f = d * (n - d) / (n - 1)
            p = nj / n
            V += w * w * f * (np.diag(p) - np.outer(p, p))
    out = {"O": O, "E": E, "U": U, "V": V, "chi2": float("nan"), "df": k - 1, "p": float("nan")}
    try:
        Vr = V[:-1, :-1]
        if k >= 2 and np.linalg.matrix_rank(Vr) == k - 1:
            chi2 = float(U[:-1] @ np.linalg.solve(Vr, U[:-1]))
            out.update(chi2=chi2, p=float(stats.chi2.sf(chi2, k - 1)))
    except np.linalg.LinAlgError:
        pass
    return out


def logrank_trend(lr: dict, scores: np.ndarray | None = None) -> dict:
    """Log-rank test for trend across ordered groups (scores 1, 2, 3 … by default)."""
    k = len(lr["U"])
    s = np.arange(1, k + 1, dtype=float) if scores is None else np.asarray(scores, dtype=float)
    num = float(s @ lr["U"])
    var = float(s @ lr["V"] @ s)
    if var <= 0:
        return {"chi2": float("nan"), "p": float("nan"), "z": float("nan")}
    return {"chi2": num ** 2 / var, "p": float(stats.chi2.sf(num ** 2 / var, 1)), "z": num / math.sqrt(var)}


def hazard_ratios(lr: dict) -> dict:
    """Hazard ratio (first / second group) by the Mantel–Haenszel and the
    log-rank (O/E) methods, each with a 95 % CI."""
    O, E, U, V = lr["O"], lr["E"], lr["U"], lr["V"]
    out = {}
    v = V[0, 0]
    if v > 0:
        ln = U[0] / v
        se = 1 / math.sqrt(v)
        out["mh"] = (math.exp(ln), math.exp(ln - Z95 * se), math.exp(ln + Z95 * se))
    if min(O) > 0 and min(E) > 0:
        ln = math.log((O[0] / E[0]) / (O[1] / E[1]))
        se = math.sqrt(1 / E[0] + 1 / E[1])
        out["lr"] = (math.exp(ln), math.exp(ln - Z95 * se), math.exp(ln + Z95 * se))
    return out


def _survival_data(ctx: Context) -> tuple[pd.DataFrame, list[str], int]:
    """Clean frame with ``time``, ``event`` (bool), ``group`` (code) plus group names."""
    tname, ename = ctx.var("time"), ctx.var("event")
    gname = ctx.var("group", required=False)
    if ename == tname:
        raise AnalysisError("The time and the event variables must be different columns.")
    time = ctx.numeric(tname).to_numpy()
    want = ctx.opt("event_value", "1")
    ev = event_indicator(ctx, ename, want)
    raw = ctx.column(ename)
    if not isinstance(raw.dtype, pd.CategoricalDtype):
        present = sorted(set(raw.dropna().tolist()))
        try:
            target = float((want or "1").strip())
        except ValueError:
            raise AnalysisError(f"The event value “{want}” is not a number, but “{ename}” is numeric. "
                                f"It contains the values {', '.join(f'{v:g}' for v in present[:8])}.") from None
        if present and target not in present:
            raise AnalysisError(
                f"No row of “{ename}” equals the event value {target:g}. It contains the values "
                f"{', '.join(f'{v:g}' for v in present[:8])}; type the one that means “event occurred”.")
    if gname:
        if gname in (tname, ename):
            raise AnalysisError("The grouping variable must differ from the time and event variables.")
        fac = ctx.factor(gname)
        names = [str(c) for c in fac.cat.categories]
        codes = fac.cat.codes.to_numpy().astype(float)
        codes[codes < 0] = np.nan
    else:
        names, codes = ["All subjects"], np.zeros(len(time))
    ok = np.isfinite(time) & np.isfinite(ev) & np.isfinite(codes)
    dropped = int(len(time) - ok.sum())
    d = pd.DataFrame({"time": time[ok], "event": ev[ok] == 1, "group": codes[ok].astype(int)})
    if len(d) == 0:
        raise AnalysisError("There are no rows with a time, an event status" + (" and a group." if gname else "."))
    if (d["time"] < 0).any():
        raise AnalysisError("Survival times cannot be negative.")
    # drop empty groups, keep order
    used = [i for i in range(len(names)) if (d["group"] == i).any()]
    remap = {old: new for new, old in enumerate(used)}
    d["group"] = d["group"].map(remap).astype(int)
    names = [names[i] for i in used]
    if not d["event"].any():
        raise AnalysisError("No events were observed (every subject is censored), so survival cannot be "
                            "analysed. Check the “Event value” setting.")
    return d, names, dropped


_ADJ = {"holm": "holm", "bonferroni": "bonferroni", "sidak": "sidak", "fdr_bh": "fdr_bh"}


@analysis(
    id="kaplan_meier", name="Kaplan–Meier survival analysis", category="Survival",
    description="Survival curves, median survival, and comparison of groups by the log-rank test.",
    keywords="kaplan meier survival logrank log-rank mantel cox gehan breslow wilcoxon hazard ratio "
             "median survival censored time to event",
    options=[
        opt.variable("time", "Time", opt.CONTINUOUS),
        opt.variable("event", "Event / status", opt.ANY),
        opt.text("event_value", "Event value", "1",
                 help="The value in the status column that means the event happened (others are censored)."),
        opt.variable("group", "Groups", opt.ANY, required=False),
        opt.boolean("show_tests", "Comparison of survival curves", True, section="Compare groups"),
        opt.boolean("tarone_ware", "Tarone–Ware test", False, section="Compare groups"),
        opt.boolean("trend", "Log-rank test for trend (3+ ordered groups)", True, section="Compare groups"),
        opt.boolean("show_hr", "Hazard ratio (two groups)", True, section="Compare groups"),
        opt.boolean("pairwise", "Pairwise comparisons (3+ groups)", True, section="Compare groups"),
        opt.choice("p_adjust", "Correct pairwise P values by", opt.P_ADJUST, default="holm",
                   section="Compare groups"),
        opt.text("times", "Survival proportions at these times", "", placeholder="e.g. 12, 24, 36",
                 section="Survival table"),
        opt.boolean("full_table", "Survival proportions at every event time", False, section="Survival table"),
        opt.boolean("show_rmst", "Restricted mean survival time", False, section="Survival table"),
        opt.boolean("show_plot", "Survival plot", True, section="Plot"),
        opt.boolean("show_ci", "95% confidence bands", False, section="Plot"),
        opt.boolean("show_risk", "Numbers at risk below the plot", True, section="Plot"),
        opt.boolean("percent", "Show survival as percent", True, section="Plot"),
    ])
def kaplan_meier(ctx: Context) -> Results:
    d, names, dropped = _survival_data(ctx)
    k = len(names)
    tname = ctx.options["time"]
    r = Results()
    if dropped:
        r.warn(f"{dropped} row(s) with a missing time, status or group were ignored.")

    kms = {}
    for i, name in enumerate(names):
        sub = d[d["group"] == i]
        kms[name] = km_table(sub["time"].to_numpy(), sub["event"].to_numpy())

    # ---- summary per group
    show_rmst = bool(ctx.opt("show_rmst", False))
    tau = min(float(km["time"].max()) for km in kms.values())
    cols = [("group", "Group", "text"), ("n", "Subjects", "int"), ("events", "Events", "int"),
            ("censored", "Censored", "int"), ("median", "Median survival", "num"),
            ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num")]
    if show_rmst:
        cols += [("rmst", "Restricted mean", "num"), ("rmst_se", "SE", "num")]
    t = r.table("Summary of survival", cols)
    medians = {}
    for name, km in kms.items():
        med, lo, hi = median_survival(km)
        medians[name] = med
        row = dict(group=name, n=int(km["events"].sum() + km["censored"].sum()),
                   events=int(km["events"].sum()), censored=int(km["censored"].sum()), median=med, lo=lo, hi=hi)
        if show_rmst:
            row["rmst"], row["rmst_se"] = restricted_mean(km, tau)
        t.add_row(row)
    t.note("Median survival = the earliest time at which the Kaplan–Meier survival estimate is 50% or less "
           "(empty = not reached). 95% CI by the method of Brookmeyer & Crowley from the log(−log) "
           "confidence band of the survival curve; an empty limit could not be estimated.")
    if show_rmst:
        t.note(f"Restricted mean survival time = area under the survival curve up to {tname} = {tau:g} "
               "(the shortest of the groups' longest follow-up times).")

    # ---- comparisons
    sentences = []
    events_per_group = [int(kms[n]["events"].sum()) for n in names]
    if k >= 2 and ctx.opt("show_tests", True):
        lr = weighted_logrank(d["time"], d["event"], d["group"], k, "logrank")
        tests = [("Log-rank (Mantel–Cox)", lr),
                 ("Gehan–Breslow–Wilcoxon", weighted_logrank(d["time"], d["event"], d["group"], k, "gehan"))]
        if ctx.opt("tarone_ware", False):
            tests.append(("Tarone–Ware", weighted_logrank(d["time"], d["event"], d["group"], k, "tarone")))
        t = r.table("Comparison of survival curves", [
            ("test", "Test", "text"), ("chi2", "Chi-square", "num"), ("df", "df", "int"),
            ("p", "P value", "p"), ("s", "Summary", "text")])
        for label, res in tests:
            t.add_row(test=label, chi2=res["chi2"], df=res["df"], p=res["p"], s=stars(res["p"]))
        if k >= 3 and ctx.opt("trend", True):
            tr = logrank_trend(lr)
            t.add_row(test="Log-rank test for trend", chi2=tr["chi2"], df=1, p=tr["p"], s=stars(tr["p"]))
            t.note("The test for trend assumes the groups are in a meaningful order (scores 1, 2, 3 …).")
        t.note("The log-rank test weights all time points equally; the Gehan–Breslow–Wilcoxon test weights "
               "each event time by the number of subjects still at risk, giving more weight to early events"
               + ("; Tarone–Ware uses the square root of the number at risk." if ctx.opt("tarone_ware", False) else "."))
        t2 = r.table("Observed and expected events (log-rank)", [
            ("group", "Group", "text"), ("o", "Observed", "int"), ("e", "Expected", "num"), ("oe", "O / E", "num")])
        for i, name in enumerate(names):
            t2.add_row(group=name, o=int(lr["O"][i]), e=lr["E"][i],
                       oe=lr["O"][i] / lr["E"][i] if lr["E"][i] > 0 else float("nan"))
        p = lr["p"]
        if p == p:
            if p < 0.05:
                sentences.append(f"The survival curves differ significantly (log-rank χ²({k - 1}) = "
                                 f"{lr['chi2']:.3g}, {p_phrase(p)}).")
            else:
                sentences.append(f"The survival curves do not differ significantly (log-rank χ²({k - 1}) = "
                                 f"{lr['chi2']:.3g}, {p_phrase(p)}).")
        else:
            r.warn("The log-rank test could not be computed (too few events).")

        if k == 2 and ctx.opt("show_hr", True):
            a, b = names
            hr = hazard_ratios(lr)
            t = r.table("Hazard ratio", [
                ("method", "Method", "text"), ("ratio", f"{a} / {b}", "num"), ("lo", "95% CI lower", "num"),
                ("hi", "95% CI upper", "num"), ("recip", f"{b} / {a}", "num"),
                ("rlo", "95% CI lower", "num"), ("rhi", "95% CI upper", "num")])
            for key, label in (("mh", "Mantel–Haenszel"), ("lr", "Log-rank")):
                if key in hr:
                    v, lo, hi = hr[key]
                    t.add_row(method=label, ratio=v, lo=lo, hi=hi, recip=1 / v, rlo=1 / hi, rhi=1 / lo)
            t.note("Mantel–Haenszel: exp((O−E)/V) with CI exp((O−E)/V ± 1.96/√V). Log-rank: (O₁/E₁)/(O₂/E₂) with "
                   "SE of the log ratio √(1/E₁ + 1/E₂). Both assume proportional hazards.")
            if "mh" in hr:
                v, lo, hi = hr["mh"]
                sentences.append(f"The hazard ratio ({a} relative to {b}) is {v:.3g} (95% CI {lo:.3g} to {hi:.3g}; "
                                 "Mantel–Haenszel).")
            ma, mb = medians[a], medians[b]
            if ma == ma and mb == mb and mb > 0 and ma > 0 and min(events_per_group) > 0:
                ratio = ma / mb
                se = math.sqrt(1 / events_per_group[0] + 1 / events_per_group[1])
                lo, hi = ratio * math.exp(-Z95 * se), ratio * math.exp(Z95 * se)
                t = r.table("Ratio of median survival times", [
                    ("ratio", f"{a} / {b}", "num"), ("lo", "95% CI lower", "num"), ("hi", "95% CI upper", "num"),
                    ("recip", f"{b} / {a}", "num"), ("rlo", "95% CI lower", "num"), ("rhi", "95% CI upper", "num")])
                t.add_row(ratio=ratio, lo=lo, hi=hi, recip=1 / ratio, rlo=1 / hi, rhi=1 / lo)
                t.note("The CI assumes exponential survival: SE of the log ratio = √(1/events₁ + 1/events₂).")

        if k >= 3 and ctx.opt("pairwise", True):
            pairs = list(itertools.combinations(range(k), 2))
            raw = []
            for i, j in pairs:
                sub = d[d["group"].isin([i, j])]
                res = weighted_logrank(sub["time"], sub["event"], (sub["group"] == j).astype(int), 2)
                raw.append(res)
            method = ctx.opt("p_adjust", "holm")
            ps = np.array([x["p"] for x in raw])
            adj = ps.copy()
            fin = np.isfinite(ps)
            if method in _ADJ and fin.sum() > 1:
                from statsmodels.stats.multitest import multipletests
                adj[fin] = multipletests(ps[fin], method=_ADJ[method])[1]
            t = r.table("Pairwise comparisons (log-rank)", [
                ("pair", "Comparison", "text"), ("chi2", "Chi-square", "num"), ("p", "P value", "p"),
                ("padj", "Adjusted P value", "p"), ("s", "Summary", "text")])
            for (i, j), res, pa in zip(pairs, raw, adj):
                t.add_row(pair=f"{names[i]} vs. {names[j]}", chi2=res["chi2"], p=res["p"], padj=pa, s=stars(pa))
            label = dict(opt.P_ADJUST).get(method, "None")
            t.note("Each comparison is a log-rank (Mantel–Cox) test using only the two groups involved. "
                   + (f"P values adjusted for {len(pairs)} comparisons by the {label} method."
                      if method in _ADJ else "P values are not adjusted for multiple comparisons."))
    elif k == 1:
        (name, km), = kms.items()
        med, lo, hi = median_survival(km)
        if med == med:
            ci = f" (95% CI {lo:.4g} to {'not reached' if hi != hi else format(hi, '.4g')})" if lo == lo else ""
            sentences.append(f"Median survival is {med:.4g}{ci}; {events_per_group[0]} of "
                             f"{len(d)} subjects had the event.")
        else:
            sentences.append(f"Median survival was not reached; {events_per_group[0]} of {len(d)} subjects "
                             "had the event.")
    if k >= 2:
        meds = ", ".join(f"{n}: {'not reached' if m != m else format(m, '.4g')}" for n, m in medians.items())
        sentences.append(f"Median survival — {meds}.")
    r.summary(" ".join(sentences))

    # ---- survival proportions
    times = parse_numbers(ctx.opt("times", ""), "times")
    surv_cols = [("group", "Group", "text"), ("time", tname, "num"), ("at_risk", "At risk", "int"),
                 ("surv", "Survival", "pct"), ("se", "Std. error", "num"),
                 ("lo", "95% CI lower", "pct"), ("hi", "95% CI upper", "pct")]
    if times:
        t = r.table("Survival proportions at chosen times", surv_cols)
        for i, name in enumerate(names):
            km = kms[name]
            gt = d.loc[d["group"] == i, "time"].to_numpy()
            for tt in times:
                if tt > km["time"].max():
                    t.add_row(group=name, time=tt, at_risk=0)
                    continue
                prev = km[km["time"] <= tt]
                row = prev.iloc[-1] if len(prev) else None
                t.add_row(group=name, time=tt, at_risk=int(np.sum(gt >= tt)),
                          surv=1.0 if row is None else row["surv"], se=0.0 if row is None else row["se"],
                          lo=1.0 if row is None else row["lo"], hi=1.0 if row is None else row["hi"])
        t.note("Kaplan–Meier estimates; standard error by Greenwood's formula; 95% CI from the log(−log) "
               "transform. Empty cells: the time is beyond the last follow-up of that group. "
               "At risk = subjects still under observation just before that time.")
    if ctx.opt("full_table", False):
        cap = 300
        t = r.table("Survival proportions at each event time",
                    surv_cols[:3] + [("events", "Events", "int"), ("censored", "Censored", "int")] + surv_cols[3:])
        total = 0
        for name, km in kms.items():
            for _, row in km.iterrows():
                total += 1
                if total <= cap:
                    t.add_row(group=name, **{c: row[c] for c in km.columns})
        t.note("Kaplan–Meier estimates; standard error by Greenwood's formula; 95% CI from the log(−log) transform.")
        if total > cap:
            t.note(f"Only the first {cap} of {total} rows are shown.")

    if ctx.opt("show_plot", True):
        _km_plot(r, ctx, d, names, kms, tname)
    r.graph("Create a survival graph", "survival",
            {key: ctx.options.get(key) for key in ("time", "event", "event_value", "group")})
    return r


def _km_plot(r: Results, ctx: Context, d: pd.DataFrame, names: list[str], kms: dict, tname: str) -> None:
    k = len(names)
    show_risk = bool(ctx.opt("show_risk", True))
    scale = 100.0 if ctx.opt("percent", True) else 1.0
    if show_risk:
        fig, axes = new_figure(6.2, 4.2 + 0.24 * k, nrows=2,
                               gridspec_kw={"height_ratios": [1, 0.07 * (k + 1.5)]})
        ax, rax = axes
    else:
        fig, ax = new_figure(6.2, 4.0)
        rax = None
    tmax = float(d["time"].max())
    for i, name in enumerate(names):
        km = kms[name]
        c = COLORS[i % len(COLORS)]
        ev = km[km["events"] > 0]
        tt = np.concatenate([[0.0], ev["time"].to_numpy(), [float(km["time"].max())]])
        ss = np.concatenate([[1.0], ev["surv"].to_numpy(), [float(km["surv"].iloc[-1])]])
        ax.step(tt, ss * scale, where="post", color=c, lw=1.7, label=name)
        if ctx.opt("show_ci", False):
            lo = np.concatenate([[1.0], ev["lo"].to_numpy(), [ev["lo"].iloc[-1] if len(ev) else 1.0]])
            hi = np.concatenate([[1.0], ev["hi"].to_numpy(), [ev["hi"].iloc[-1] if len(ev) else 1.0]])
            ax.fill_between(tt, lo * scale, hi * scale, step="post", color=c, alpha=0.14, lw=0)
        cens = km[km["censored"] > 0]
        ax.plot(cens["time"], cens["surv"] * scale, linestyle="none", marker="|", ms=7, mew=1.2, color=c)
    ax.set_ylim(0, 1.04 * scale)
    ax.set_xlim(0, tmax * 1.03 if tmax > 0 else 1)
    ax.set_ylabel("Percent survival" if scale == 100 else "Probability of survival")
    if k > 1:
        ax.legend(loc="best")
    if rax is None:
        ax.set_xlabel(tname)
    else:
        ticks = [tk for tk in ax.get_xticks() if 0 <= tk <= tmax * 1.03]
        ax.set_xticks(ticks)
        rax.set_xlim(ax.get_xlim())
        rax.set_ylim(-0.5, k - 0.5)
        rax.invert_yaxis()
        for i, name in enumerate(names):
            gt = d.loc[d["group"] == i, "time"].to_numpy()
            for tk in ticks:
                rax.text(tk, i, str(int(np.sum(gt >= tk))), ha="center", va="center", fontsize=8.5,
                         color=COLORS[i % len(COLORS)])
        rax.set_yticks(range(k))
        rax.set_yticklabels(names, fontsize=8.5)
        rax.set_xticks(ticks)
        rax.tick_params(length=0, labelbottom=True)
        rax.tick_params(axis="y", pad=16)
        for sp in rax.spines.values():
            sp.set_visible(False)
        rax.set_title("Number at risk", fontsize=9, fontweight="normal")
        rax.set_xlabel(tname)
    r.plot("Kaplan–Meier survival curve" + ("s" if k > 1 else ""), fig)
