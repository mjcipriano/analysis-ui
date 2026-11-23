"""Categorical data: contingency tables (chi-square, Fisher), goodness of
fit / binomial test, McNemar's test and confidence intervals of a proportion.

Contingency data can be entered in two forms (option ``data_form``):

``raw``     two categorical variables, one subject per data row
``counts``  a ready-made table: an optional label column plus two or more
            numeric columns holding the counts (GraphPad Prism style)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from ..core import analysis, opt, Context, Results, AnalysisError, new_figure, COLORS, stars, fmt_p
from .descriptive import conf_opt, conf_level, pct, fnum, p_text, parse_numbers

CAT = "Categorical data"

_FORMS = [("raw", "Raw data (one subject per row)"), ("counts", "Table of counts")]


# --------------------------------------------------------------------------
# Confidence intervals of proportions
# --------------------------------------------------------------------------

def ci_wald(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    p = k / n
    h = stats.norm.ppf(0.5 + conf / 2) * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def ci_wilson(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - h), min(1.0, centre + h)


def ci_clopper_pearson(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def ci_agresti_coull(k: float, n: float, conf: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + conf / 2)
    nt = n + z * z
    pt = (k + z * z / 2) / nt
    h = z * math.sqrt(pt * (1 - pt) / nt)
    return max(0.0, pt - h), min(1.0, pt + h)


# --------------------------------------------------------------------------
# Reading a contingency table
# --------------------------------------------------------------------------

def table_options(min_cols: int = 2) -> list[dict]:
    return [
        opt.choice("data_form", "Data are", _FORMS, default="raw", section="Data"),
        opt.variable("rows", "Rows (groups / exposure)", opt.ANY, section="Data", visible={"data_form": "raw"}),
        opt.variable("cols", "Columns (outcome)", opt.ANY, section="Data", visible={"data_form": "raw"}),
        opt.variable("labels", "Row labels", opt.ANY, required=False, section="Data",
                     visible={"data_form": "counts"}),
        opt.variables("count_cols", "Columns of counts", opt.CONTINUOUS, min=min_cols, section="Data",
                      visible={"data_form": "counts"}),
    ]


def contingency_table(ctx: Context) -> tuple[pd.DataFrame, str, str]:
    """The table of counts as a DataFrame (index = row labels, columns =
    column labels), plus titles for the row and column dimension.
    Empty rows and columns are dropped."""
    if ctx.opt("data_form", "raw") == "raw":
        rn, cn = ctx.var("rows"), ctx.var("cols")
        if rn == cn:
            raise AnalysisError("Rows and columns must be two different variables.")
        rf, cf = ctx.factor(rn), ctx.factor(cn)
        ok = rf.notna() & cf.notna()
        if not ok.any():
            raise AnalysisError("No rows have a value for both variables.")
        tab = pd.crosstab(rf[ok].to_numpy(), cf[ok].to_numpy(), dropna=False)
        tab = tab.reindex(index=[l for l in rf.cat.categories if l in tab.index],
                          columns=[l for l in cf.cat.categories if l in tab.columns], fill_value=0)
        row_title, col_title = rn, cn
    else:
        cols = list(dict.fromkeys(ctx.vars("count_cols", min=2)))
        if len(cols) < 2:
            raise AnalysisError("Please choose at least 2 different columns of counts.")
        ln = ctx.var("labels", required=False)
        if ln in cols:
            raise AnalysisError("The row-label column cannot also be a column of counts.")
        vals = pd.DataFrame({c: ctx.numeric(c).to_numpy(float) for c in cols})
        keep = vals.notna().any(axis=1).to_numpy().copy()
        if ln:
            lab = ctx.column(ln).astype(object).map(
                lambda v: None if pd.isna(v) else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)))
            keep &= lab.notna().to_numpy()
            labels = lab[keep].tolist()
        else:
            labels = [f"Row {i + 1}" for i in np.flatnonzero(keep)]
        vals = vals[keep]
        if vals.empty:
            raise AnalysisError("The selected columns contain no counts.")
        if vals.isna().any().any():
            raise AnalysisError("The table of counts has empty cells. Enter 0 where nothing was observed.")
        arr = vals.to_numpy()
        if (arr < 0).any():
            raise AnalysisError("Counts cannot be negative.")
        if not np.allclose(arr, np.round(arr)):
            raise AnalysisError("The table must contain counts (whole numbers of subjects), "
                                "not percentages, means or normalised values.")
        tab = pd.DataFrame(np.round(arr).astype(int), columns=cols)
        tab.index = labels
        if tab.index.has_duplicates:              # same label twice → add up
            tab = tab.groupby(level=0, sort=False).sum()
        row_title, col_title = (ln or "Row"), "Column"
    tab = tab.loc[tab.sum(axis=1) > 0, tab.sum(axis=0) > 0]
    tab.index = [str(i) for i in tab.index]
    tab.columns = [str(c) for c in tab.columns]
    return tab.astype(int), row_title, col_title


def _need_2d(tab: pd.DataFrame):
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        raise AnalysisError(f"A contingency table needs at least 2 rows and 2 columns with data, but this one "
                            f"has {tab.shape[0]} row(s) and {tab.shape[1]} column(s) with non-zero counts.")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class _TooManyTables(Exception):
    pass


def fisher_rxc(table: np.ndarray, max_tables: int = 150_000) -> float:
    """Exact Fisher (Freeman–Halton) P value of an r×c table by complete
    enumeration of all tables with the observed margins."""
    t = np.asarray(table, int)
    if t.shape[0] < t.shape[1]:
        t = t.T                                   # enumerate over the shorter dimension
    rows = [int(v) for v in t.sum(axis=1)]
    cols = [int(v) for v in t.sum(axis=0)]
    n = sum(rows)
    lg = gammaln(np.arange(n + 2)).tolist()       # lg[v] = log((v-1)!)  → log(v!) = lg[v+1]
    const = sum(lg[r + 1] for r in rows) + sum(lg[c + 1] for c in cols) - lg[n + 1]
    logp_obs = const - sum(lg[int(v) + 1] for v in t.ravel())
    thresh = logp_obs + 1e-7
    nrow, ncol = len(rows), len(cols)
    state = {"count": 0, "p": 0.0}

    def rec_col(j, rem, acc):
        if j == ncol - 1:
            state["count"] += 1
            if state["count"] > max_tables:
                raise _TooManyTables
            lp = const + acc - sum(lg[v + 1] for v in rem)
            if lp <= thresh:
                state["p"] += math.exp(lp)
            return
        suffix = [0] * (nrow + 1)
        for i in range(nrow - 1, -1, -1):
            suffix[i] = suffix[i + 1] + rem[i]

        def rec_row(i, left, acc2, new_rem):
            if i == nrow - 1:
                if left <= rem[i]:
                    rec_col(j + 1, new_rem + [rem[i] - left], acc2 - lg[left + 1])
                return
            for v in range(max(0, left - suffix[i + 1]), min(left, rem[i]) + 1):
                rec_row(i + 1, left - v, acc2 - lg[v + 1], new_rem + [rem[i] - v])

        rec_row(0, cols[j], acc, [])

    rec_col(0, rows, 0.0)
    return float(min(1.0, state["p"]))


def fisher_test(table: np.ndarray) -> tuple[float, str]:
    """Two-sided Fisher exact P value and a description of how it was obtained."""
    t = np.asarray(table, int)
    if t.shape == (2, 2):
        return float(stats.fisher_exact(t, alternative="two-sided")[1]), "exact"
    try:
        return fisher_rxc(t), "exact"
    except (_TooManyTables, RecursionError):
        pass
    try:
        res = stats.fisher_exact(t, method=stats.MonteCarloMethod(n_resamples=19999,
                                                                   rng=np.random.default_rng(20240501)))
        return float(res.pvalue), "Monte Carlo"
    except Exception:
        return math.nan, "unavailable"


def trend_test(counts: np.ndarray) -> tuple[float, float]:
    """Cochran–Armitage chi-square test for trend. ``counts`` is 2 × k with
    ordered columns (scores 1..k). Returns (chi-square with 1 df, P)."""
    c = np.asarray(counts, float)
    n_j = c.sum(axis=0)
    N = n_j.sum()
    s = np.arange(1, c.shape[1] + 1, dtype=float)
    pbar = c[0].sum() / N
    tstat = np.sum(s * (c[0] - n_j * pbar))
    var = pbar * (1 - pbar) * (np.sum(n_j * s * s) - np.sum(n_j * s) ** 2 / N)
    if var <= 0:
        return math.nan, math.nan
    chi2 = float(tstat ** 2 / var)
    return chi2, float(stats.chi2.sf(chi2, 1))


def two_by_two(a: int, b: int, c: int, d: int, conf: float) -> dict:
    """Effect sizes of a 2×2 table  [[a, b], [c, d]]  (rows = groups,
    first column = outcome of interest)."""
    z = stats.norm.ppf(0.5 + conf / 2)
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    out = {"p1": p1, "p2": p2, "corrected": False}
    # relative risk (Katz log method) and odds ratio (Woolf logit method)
    aa, bb, cc, dd = a, b, c, d
    if min(a, b, c, d) == 0:
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        out["corrected"] = True
    rr = (aa / (aa + bb)) / (cc / (cc + dd))
    se_rr = math.sqrt(1 / aa - 1 / (aa + bb) + 1 / cc - 1 / (cc + dd))
    out["rr"] = (rr, rr * math.exp(-z * se_rr), rr * math.exp(z * se_rr))
    orr = (aa * dd) / (bb * cc)
    se_or = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    out["or"] = (orr, orr * math.exp(-z * se_or), orr * math.exp(z * se_or))
    # risk difference, Newcombe hybrid score interval
    l1, u1 = ci_wilson(a, n1, conf)
    l2, u2 = ci_wilson(c, n2, conf)
    rd = p1 - p2
    rd_lo = rd - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    rd_hi = rd + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    out["rd"] = (rd, rd_lo, rd_hi)
    if rd != 0:
        nnt = 1 / abs(rd)
        if rd_lo > 0 or rd_hi < 0:
            lo, hi = sorted([1 / abs(rd_lo), 1 / abs(rd_hi)])
            out["nnt"] = (nnt, lo, hi)
        else:
            out["nnt"] = (nnt, math.nan, math.nan)
    else:
        out["nnt"] = (math.inf, math.nan, math.nan)
    # diagnostic-test view: rows = test result (+/−), columns = condition (present/absent)
    def prop(k, n):
        return (k / n, *ci_wilson(k, n, conf)) if n > 0 else (math.nan, math.nan, math.nan)
    out["sens"] = prop(a, a + c)
    out["spec"] = prop(d, b + d)
    out["ppv"] = prop(a, a + b)
    out["npv"] = prop(d, c + d)
    out["acc"] = prop(a + d, a + b + c + d)

    def lr(num_k, num_n, den_k, den_n):
        if min(num_k, den_k) == 0 or num_n == 0 or den_n == 0:
            v = (num_k / num_n) / (den_k / den_n) if den_k and num_n and den_n else math.nan
            return (v, math.nan, math.nan)
        v = (num_k / num_n) / (den_k / den_n)
        se = math.sqrt(1 / num_k - 1 / num_n + 1 / den_k - 1 / den_n)
        return (v, v * math.exp(-z * se), v * math.exp(z * se))
    out["lr_pos"] = lr(a, a + c, b, b + d)
    out["lr_neg"] = lr(c, a + c, d, b + d)
    return out


def _pick(labels: list[str], wanted: str, what: str) -> int:
    """Index of the level chosen in a ``level`` option ('' = first)."""
    if not wanted:
        return 0
    if wanted in labels:
        return labels.index(wanted)
    raise AnalysisError(f"“{wanted}” is not one of the {what} of the table ({', '.join(labels)}).")


def _matrix_table(r: Results, title: str, tab: pd.DataFrame, corner: str, fmt: str = "int",
                  totals: bool = False, values: np.ndarray | None = None,
                  row_tot: np.ndarray | None = None, col_tot: np.ndarray | None = None, grand: float | None = None):
    cols = [("row", corner, "text"), *[(f"c{j}", c, fmt) for j, c in enumerate(tab.columns)]]
    if totals:
        cols.append(("total", "Total", fmt))
    t = r.table(title, cols)
    vals = tab.to_numpy() if values is None else values
    cast = (lambda v: int(v)) if fmt == "int" else (lambda v: float(v))
    for i, name in enumerate(tab.index):
        row = {f"c{j}": cast(vals[i, j]) for j in range(tab.shape[1])}
        if totals:
            row["total"] = cast(row_tot[i])
        t.add_row(row=name, **row)
    if totals:
        row = {f"c{j}": cast(col_tot[j]) for j in range(tab.shape[1])}
        row["total"] = cast(grand)
        t.add_row(row="Total", **row)
    return t


# --------------------------------------------------------------------------
# Contingency table
# --------------------------------------------------------------------------

@analysis(
    id="contingency", name="Contingency table: chi-square & Fisher's exact", category=CAT,
    description="Test the association between two categorical variables; relative risk, odds ratio and "
                "diagnostic-test statistics for 2×2 tables.",
    keywords="chi square fisher exact contingency crosstab relative risk odds ratio sensitivity specificity "
             "cramer phi yates trend cochran armitage g test",
    options=[
        *table_options(),
        opt.boolean("yates", "Yates' continuity correction (2×2)", True, section="Tests"),
        opt.boolean("fisher", "Fisher's exact test", True, section="Tests"),
        opt.boolean("gtest", "Likelihood-ratio (G) test", False, section="Tests"),
        opt.boolean("trend", "Chi-square test for trend (ordered categories)", False, section="Tests",
                    help="For 2×k tables whose k categories have a natural order."),
        opt.boolean("effects", "Relative risk, odds ratio, risk difference", True, section="2×2 tables"),
        opt.boolean("diagnostic", "Sensitivity, specificity, predictive values", False, section="2×2 tables"),
        opt.level("ref_row", "Row of interest (exposed / test positive)", sources=["rows", "labels"],
                  section="2×2 tables"),
        opt.level("event_col", "Column of interest (event / disease)", sources=["cols", "count_cols"],
                  section="2×2 tables"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_expected", "Expected counts", False, section="Output"),
        opt.boolean("show_row_pct", "Percentages of row totals", True, section="Output"),
        opt.boolean("show_col_pct", "Percentages of column totals", False, section="Output"),
        opt.boolean("show_total_pct", "Percentages of grand total", False, section="Output"),
        opt.boolean("show_plot", "Stacked bar plot", True, section="Output"),
    ])
def contingency(ctx: Context) -> Results:
    tab, row_title, col_title = contingency_table(ctx)
    _need_2d(tab)
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    obs = tab.to_numpy().astype(float)
    N = obs.sum()
    rt, ct = obs.sum(axis=1), obs.sum(axis=0)
    expected = np.outer(rt, ct) / N
    nr, nc = obs.shape
    is22 = (nr, nc) == (2, 2)
    corner = f"{row_title} \\ {col_title}" if ctx.opt("data_form", "raw") == "raw" else row_title

    r = Results()
    _matrix_table(r, "Observed counts", tab, corner, "int", True, obs, rt, ct, N)
    if ctx.opt("show_expected"):
        _matrix_table(r, "Expected counts (if rows and columns were independent)", tab, corner, "num",
                      False, expected)
    if ctx.opt("show_row_pct"):
        _matrix_table(r, "Percentage of row total", tab, corner, "pct", True, obs / rt[:, None],
                      np.ones(nr), ct / N, 1.0)
    if ctx.opt("show_col_pct"):
        _matrix_table(r, "Percentage of column total", tab, corner, "pct", True, obs / ct[None, :],
                      rt / N, np.ones(nc), 1.0)
    if ctx.opt("show_total_pct"):
        _matrix_table(r, "Percentage of grand total", tab, corner, "pct", True, obs / N, rt / N, ct / N, 1.0)

    # ---- tests
    tt = r.table("Tests of association",
                 [("test", "Test", "text"), ("stat", "Statistic", "num"), ("df", "df", "int"),
                  ("p", "P value", "p"), ("stars", "P value summary", "text"),
                  ("sig", f"Significant (α = {alpha:g})?", "text")])

    def add(test, stat, df, p):
        tt.add_row(test=test, stat=stat, df=df, p=p, stars=stars(p),
                   sig=("Yes" if p < alpha else "No") if np.isfinite(p) else "")

    chi2, p_chi, dof, _ = stats.chi2_contingency(obs, correction=False)
    add("Pearson chi-square", float(chi2), int(dof), float(p_chi))
    p_main, main_name = float(p_chi), "chi-square test"
    if is22 and ctx.opt("yates"):
        chi2_y, p_y, _, _ = stats.chi2_contingency(obs, correction=True)
        add("Chi-square with Yates' correction", float(chi2_y), 1, float(p_y))
        p_main, main_name = float(p_y), "chi-square test with Yates' correction"
    if ctx.opt("gtest"):
        g, p_g, _, _ = stats.chi2_contingency(obs, correction=False, lambda_="log-likelihood")
        add("Likelihood-ratio G test", float(g), int(dof), float(p_g))
    if ctx.opt("fisher"):
        p_f, how = fisher_test(obs.astype(int))
        if how == "unavailable":
            tt.add_row(test="Fisher's exact test", sig="not available for this table")
        else:
            add("Fisher's exact test" + ("" if is22 else " (Freeman–Halton)")
                + (" — Monte Carlo estimate" if how == "Monte Carlo" else ""), math.nan, None, p_f)
            if how == "Monte Carlo":
                tt.note("The table has too many possible arrangements for complete enumeration; Fisher's P value "
                        "is a Monte Carlo estimate from 19,999 random tables with the observed margins.")
            if is22 or how == "exact":
                p_main, main_name = p_f, "Fisher's exact test"
    p_trend = math.nan
    if ctx.opt("trend"):
        if min(nr, nc) == 2 and max(nr, nc) >= 3:
            arr = obs if nr == 2 else obs.T
            chi_t, p_trend = trend_test(arr)
            add("Chi-square test for trend (Cochran–Armitage)", chi_t, 1, p_trend)
            ordered = tab.columns if nr == 2 else tab.index
            tt.note("Trend test: the categories " + " < ".join(ordered) + " are scored 1, 2, 3, … in this order; "
                    "it asks whether the proportion changes linearly across them.")
        else:
            tt.add_row(test="Chi-square test for trend", sig="needs a 2×k table (k ≥ 3)")
    tt.note("All P values are two-sided. Null hypothesis: the row and column classifications are independent.")
    low = int((expected < 5).sum())
    if low:
        share = low / expected.size
        msg = (f"{low} of {expected.size} cells have an expected count below 5 (smallest {expected.min():.2f}); "
               "the chi-square approximation may be inaccurate")
        if is22 or share > 0.2 or expected.min() < 1:
            r.warn(msg + ". Prefer Fisher's exact test.")
        else:
            tt.note(msg + " when more than 20% of cells are affected.")

    es = r.table("Strength of association",
                 [("measure", "Measure", "text"), ("value", "Value", "num"), ("note", "Interpretation", "text")])
    v = math.sqrt(chi2 / (N * min(nr - 1, nc - 1)))
    if is22:
        a, b, c, d = obs.ravel()
        phi = (a * d - b * c) / math.sqrt(rt[0] * rt[1] * ct[0] * ct[1])
        es.add_row(measure="Phi coefficient", value=float(phi), note="−1 to 1; 0 = no association")
    es.add_row(measure="Cramér's V", value=float(v), note="0 = no association, 1 = perfect association")
    es.add_row(measure="Contingency coefficient", value=float(math.sqrt(chi2 / (chi2 + N))), note="")

    # ---- 2×2 extras
    summary_extra = ""
    if is22 and (ctx.opt("effects") or ctx.opt("diagnostic")):
        ri = _pick(list(tab.index), ctx.opt("ref_row", ""), "rows")
        ci = _pick(list(tab.columns), ctx.opt("event_col", ""), "columns")
        t2 = tab.iloc[[ri, 1 - ri], [ci, 1 - ci]]
        a, b, c, d = (int(v_) for v_ in t2.to_numpy().ravel())
        e = two_by_two(a, b, c, d, conf)
        r1, r2 = t2.index
        c1, c2 = t2.columns
        cols = [("measure", "Measure", "text"), ("value", "Value", "num"), ("lo", f"{cl} CI lower", "num"),
                ("hi", f"{cl} CI upper", "num"), ("method", "CI method", "text")]
        if ctx.opt("effects"):
            ef = r.table(f"Effect sizes: risk of “{c1}” in {r1} vs. {r2}", cols)
            ef.add_row(measure=f"Risk in {r1}", value=e["p1"], lo=ci_wilson(a, a + b, conf)[0],
                       hi=ci_wilson(a, a + b, conf)[1], method="Wilson")
            ef.add_row(measure=f"Risk in {r2}", value=e["p2"], lo=ci_wilson(c, c + d, conf)[0],
                       hi=ci_wilson(c, c + d, conf)[1], method="Wilson")
            for key, label, method in (("rr", "Relative risk", "Katz log"), ("or", "Odds ratio", "Woolf logit"),
                                       ("rd", "Risk difference (attributable risk)", "Newcombe hybrid score"),
                                       ("nnt", "Number needed to treat (or harm)", "from the risk difference")):
                val, lo, hi = e[key]
                ef.add_row(measure=label, value=val, lo=lo, hi=hi, method=method)
            ef.note(f"Risks are the proportion of “{c1}” within each row; ratios and differences compare {r1} with {r2}. "
                    "Change the row and column of interest under “2×2 tables”.")
            if e["corrected"]:
                ef.note("A cell is zero, so 0.5 was added to every cell to compute the relative risk and odds ratio "
                        "(Haldane–Anscombe correction).")
            if not np.isfinite(e["nnt"][1]) and np.isfinite(e["nnt"][0]):
                ef.note("The CI of the risk difference includes zero, so the CI of the NNT is not bounded "
                        "(it extends to infinity in both directions) and is left empty.")
            rr = e["rr"]
            summary_extra = (f" The risk of {c1} is {fnum(e['p1'] * 100, 3)}% in {r1} and {fnum(e['p2'] * 100, 3)}% in {r2}: "
                             f"relative risk {fnum(rr[0], 3)} ({cl} CI {fnum(rr[1], 3)} to {fnum(rr[2], 3)}), "
                             f"odds ratio {fnum(e['or'][0], 3)} ({cl} CI {fnum(e['or'][1], 3)} to {fnum(e['or'][2], 3)}).")
        if ctx.opt("diagnostic"):
            dg = r.table(f"Diagnostic test: “{r1}” as a test for “{c1}”", cols)
            for key, label, method in (("sens", "Sensitivity", "Wilson"), ("spec", "Specificity", "Wilson"),
                                       ("ppv", "Positive predictive value", "Wilson"),
                                       ("npv", "Negative predictive value", "Wilson"),
                                       ("acc", "Accuracy (fraction correct)", "Wilson"),
                                       ("lr_pos", "Positive likelihood ratio", "log method"),
                                       ("lr_neg", "Negative likelihood ratio", "log method")):
                val, lo, hi = e[key]
                dg.add_row(measure=label, value=val, lo=lo, hi=hi, method=method)
            dg.note(f"Rows are read as the test result (“{r1}” = positive) and columns as the true condition "
                    f"(“{c1}” = present). Proportions are shown as fractions. Predictive values apply only when the "
                    "prevalence in the table matches the population the test will be used in.")

    assoc = (f"{row_title} and {col_title}" if ctx.opt("data_form", "raw") == "raw" else "the rows and columns")
    if p_main < alpha:
        text = f"There is a statistically significant association between {assoc} ({p_text(p_main)}; {main_name})."
    else:
        text = (f"The association between {assoc} is not statistically significant ({p_text(p_main)}; {main_name}); "
                "the data do not provide convincing evidence that the proportions differ.")
    if np.isfinite(p_trend):
        text += (" There is a significant linear trend across the ordered categories" if p_trend < alpha else
                 " The linear trend across the ordered categories is not significant") + f" ({p_text(p_trend)})."
    r.summary(text + summary_extra)

    if ctx.opt("show_plot"):
        fig, ax = new_figure(max(3.6, min(8, 1.8 + 0.9 * nr)), 3.6)
        frac = obs / rt[:, None] * 100
        bottom = np.zeros(nr)
        for j, cname in enumerate(tab.columns):
            ax.bar(range(nr), frac[:, j], bottom=bottom, color=COLORS[j % len(COLORS)], width=0.65,
                   edgecolor="white", linewidth=0.6, label=cname)
            if nr * nc <= 24:
                for i in range(nr):
                    if frac[i, j] >= 8:
                        ax.text(i, bottom[i] + frac[i, j] / 2, f"{int(obs[i, j])}", ha="center", va="center",
                                fontsize=8, color="white")
            bottom += frac[:, j]
        ax.set_xticks(range(nr))
        ax.set_xticklabels(tab.index, rotation=0 if nr <= 4 else 30, ha="center" if nr <= 4 else "right")
        ax.set_ylabel("% of row total")
        ax.set_ylim(0, 100)
        ax.legend(title=col_title if ctx.opt("data_form", "raw") == "raw" else None,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
        r.plot("Row percentages (numbers are counts)", fig)
    if ctx.opt("data_form", "raw") == "raw":
        r.graph("Create a frequency bar chart", "count_bar",
                {"x": ctx.options.get("rows"), "group": ctx.options.get("cols")})
    return r


# --------------------------------------------------------------------------
# Binomial test & chi-square goodness of fit
# --------------------------------------------------------------------------

def _observed_counts(ctx: Context) -> tuple[list[str], np.ndarray]:
    if ctx.opt("data_form", "raw") == "raw":
        f = ctx.factor(ctx.var("variable"))
        counts = f.value_counts(sort=False)
        labels = [str(l) for l in counts.index]
        obs = counts.to_numpy().astype(int)
        keep = obs > 0
        return [l for l, k in zip(labels, keep) if k], obs[keep]
    vals = parse_numbers(ctx.opt("observed", ""), "observed counts")
    if not vals:
        raise AnalysisError("Type the observed counts, separated by commas (for example: 18, 32).")
    if any(v < 0 or abs(v - round(v)) > 1e-9 for v in vals):
        raise AnalysisError("Observed counts must be whole numbers that are zero or greater.")
    names = [s.strip() for s in str(ctx.opt("names", "")).replace(";", ",").split(",") if s.strip()]
    if names and len(names) != len(vals):
        raise AnalysisError(f"You typed {len(vals)} counts but {len(names)} category names.")
    labels = names or [f"Category {i + 1}" for i in range(len(vals))]
    return labels, np.array([int(round(v)) for v in vals])


@analysis(
    id="binomial", name="Binomial & chi-square goodness of fit", category=CAT,
    description="Compare observed counts with expected proportions: exact binomial test for two categories, "
                "chi-square goodness of fit for more.",
    keywords="binomial sign test goodness of fit chi square expected observed proportion mendelian",
    options=[
        opt.choice("data_form", "Data are", [("raw", "A categorical variable"), ("counts", "Counts typed here")],
                   default="raw", section="Data"),
        opt.variable("variable", "Variable", opt.ANY, section="Data", visible={"data_form": "raw"}),
        opt.text("observed", "Observed counts", "", placeholder="e.g. 18, 32", section="Data",
                 visible={"data_form": "counts"}),
        opt.text("names", "Category names (optional)", "", placeholder="e.g. Heads, Tails", section="Data",
                 visible={"data_form": "counts"}),
        opt.text("expected", "Expected proportions", "", placeholder="equal",
                 help="One per category, in order, separated by commas — proportions, percentages or ratios "
                      "(e.g. 9, 3, 3, 1). Empty = all categories equally likely."),
        opt.level("success", "Category tested (binomial)", sources=["variable"], section="Options",
                  visible={"data_form": "raw"}),
        opt.choice("tails", "P value (binomial)", [("two-sided", "Two-tailed"),
                                                   ("greater", "One-tailed (more than expected)"),
                                                   ("less", "One-tailed (fewer than expected)")],
                   default="two-sided", section="Options"),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
        opt.boolean("show_plot", "Observed vs. expected plot", True, section="Output"),
    ])
def binomial(ctx: Context) -> Results:
    labels, obs = _observed_counts(ctx)
    k = len(obs)
    n = int(obs.sum())
    if k < 2:
        raise AnalysisError(f"At least 2 categories are needed, but only {k} "
                            f"{'was' if k == 1 else 'were'} found. If a category was never observed, "
                            "type the counts (including the 0) instead of using a variable.")
    if n == 0:
        raise AnalysisError("All observed counts are zero.")
    exp_raw = parse_numbers(ctx.opt("expected", ""), "expected proportions")
    rescaled = False
    if exp_raw:
        if len(exp_raw) != k:
            raise AnalysisError(f"There are {k} categories ({', '.join(labels)}) but {len(exp_raw)} expected "
                                "proportions were typed. Type one per category, in the same order.")
        if any(v < 0 for v in exp_raw) or sum(exp_raw) <= 0:
            raise AnalysisError("Expected proportions must be zero or greater, and not all zero.")
        tot = sum(exp_raw)
        rescaled = abs(tot - 1) > 1e-9
        prop = np.array(exp_raw) / tot
    else:
        prop = np.full(k, 1 / k)
    if np.any((prop == 0) & (obs > 0)):
        raise AnalysisError("A category with an expected proportion of zero was observed, "
                            "so the data are impossible under the hypothesis.")
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    expected = n * prop
    r = Results()
    t = r.table("Observed and expected counts",
                [("cat", "Category", "text"), ("obs", "Observed", "int"), ("obs_pct", "Observed %", "pct"),
                 ("exp", "Expected", "num"), ("exp_pct", "Expected %", "pct"),
                 ("resid", "Observed − expected", "num"), ("std", "Std. residual", "num")])
    for i in range(k):
        t.add_row(cat=labels[i], obs=int(obs[i]), obs_pct=obs[i] / n, exp=float(expected[i]), exp_pct=float(prop[i]),
                  resid=float(obs[i] - expected[i]),
                  std=float((obs[i] - expected[i]) / math.sqrt(expected[i])) if expected[i] > 0 else math.nan)
    t.add_row(cat="Total", obs=n, obs_pct=1.0, exp=float(n), exp_pct=1.0)
    t.note("Expected counts follow " + ("the proportions you typed" + (", rescaled to add up to 100%" if rescaled else "")
                                        if exp_raw else "equal proportions in every category")
           + ". Standardised residual = (observed − expected) / √expected.")
    if k == 2:
        si = _pick(labels, ctx.opt("success", "") if ctx.opt("data_form", "raw") == "raw" else "", "categories")
        x, p0 = int(obs[si]), float(prop[si])
        alt = ctx.opt("tails", "two-sided")
        if not 0 < p0 < 1:
            raise AnalysisError("For the binomial test the expected proportion must be between 0 and 1 (exclusive).")
        res = stats.binomtest(x, n, p0, alternative=alt)
        p = float(res.pvalue)
        bt = r.table("Exact binomial test",
                     [("cat", "Category tested", "text"), ("x", "Observed count", "int"), ("n", "Total", "int"),
                      ("prop", "Observed proportion", "num"), ("p0", "Expected proportion", "num"),
                      ("p", "P value", "p"), ("stars", "P value summary", "text"),
                      ("sig", f"Significant (α = {alpha:g})?", "text"), ("tails", "Tails", "text")])
        bt.add_row(cat=labels[si], x=x, n=n, prop=x / n, p0=p0, p=p, stars=stars(p),
                   sig="Yes" if p < alpha else "No", tails="Two-tailed" if alt == "two-sided" else "One-tailed")
        bt.note("Exact binomial test. " + (
            "The two-tailed P value sums the probabilities of all outcomes that are as likely as, or less likely "
            "than, the observed one (method of small P values, as in R)." if alt == "two-sided" else
            f"One-tailed: probability of observing {'at least' if alt == 'greater' else 'at most'} {x} of {n}."))
        ci = r.table(f"{cl} confidence interval of the observed proportion",
                     [("method", "Method", "text"), ("prop", "Proportion", "num"), ("lo", "Lower", "num"),
                      ("hi", "Upper", "num")])
        w = ci_wilson(x, n, conf)
        cp = ci_clopper_pearson(x, n, conf)
        ci.add_row(method="Wilson (score)", prop=x / n, lo=w[0], hi=w[1])
        ci.add_row(method="Clopper–Pearson (exact)", prop=x / n, lo=cp[0], hi=cp[1])
        r.summary(
            f"{labels[si]} was observed {x} times out of {n} ({fnum(100 * x / n, 3)}%; {cl} CI {fnum(100 * w[0], 3)}% to "
            f"{fnum(100 * w[1], 3)}%, Wilson). This "
            + ("differs significantly from" if p < alpha else "is not significantly different from")
            + f" the expected {fnum(100 * p0, 3)}% ({p_text(p)}; exact binomial test"
            + (", one-tailed" if alt != "two-sided" else "") + ").")
    else:
        pos = prop > 0
        chi2 = float(np.sum((obs[pos] - expected[pos]) ** 2 / expected[pos]))
        df = int(pos.sum()) - 1
        if df < 1:
            raise AnalysisError("At least two categories need a non-zero expected proportion.")
        p = float(stats.chi2.sf(chi2, df))
        ct = r.table("Chi-square goodness of fit",
                     [("chi2", "Chi-square", "num"), ("df", "df", "int"), ("p", "P value", "p"),
                      ("stars", "P value summary", "text"), ("sig", f"Significant (α = {alpha:g})?", "text"),
                      ("n", "Total count", "int")])
        ct.add_row(chi2=chi2, df=df, p=p, stars=stars(p), sig="Yes" if p < alpha else "No", n=n)
        ct.note("Pearson chi-square test of the null hypothesis that the observed counts follow the expected "
                "proportions.")
        low = int((expected[pos] < 5).sum())
        if low:
            r.warn(f"{low} of {int(pos.sum())} categories have an expected count below 5; the chi-square "
                   "approximation may be inaccurate. Consider combining categories.")
        r.summary(
            ("The observed distribution differs significantly from the expected proportions" if p < alpha else
             "The observed distribution is not significantly different from the expected proportions")
            + f" ({p_text(p)}; chi-square goodness of fit, χ² = {fnum(chi2)}, df = {df}).")
    if ctx.opt("show_plot"):
        fig, ax = new_figure(max(3.6, min(8, 1.6 + 0.9 * k)), 3.4)
        xs = np.arange(k)
        ax.bar(xs - 0.2, obs, width=0.4, color=COLORS[0], label="Observed")
        ax.bar(xs + 0.2, expected, width=0.4, color="#B8C0CC", label="Expected")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=0 if k <= 4 else 30, ha="center" if k <= 4 else "right")
        ax.set_ylabel("Count")
        ax.legend()
        r.plot("Observed vs. expected counts", fig)
    return r


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------

@analysis(
    id="mcnemar", name="McNemar's test (paired proportions)", category=CAT,
    description="Compare two paired yes/no measurements on the same subjects (before/after, two raters, matched pairs).",
    keywords="mcnemar paired proportions matched binary discordant before after",
    options=[
        opt.variable("var1", "First measurement", opt.ANY),
        opt.variable("var2", "Second measurement", opt.ANY),
        conf_opt(section="Options"),
        opt.alpha(section="Options"),
    ])
def mcnemar(ctx: Context) -> Results:
    n1, n2 = ctx.var("var1"), ctx.var("var2")
    if n1 == n2:
        raise AnalysisError("Choose two different variables (the two paired measurements).")
    f1, f2 = ctx.factor(n1), ctx.factor(n2)
    ok = (f1.notna() & f2.notna()).to_numpy()
    v1, v2 = f1.astype(object).to_numpy()[ok], f2.astype(object).to_numpy()[ok]
    dropped = int((~ok).sum())
    if len(v1) == 0:
        raise AnalysisError("No rows have a value for both measurements.")
    levels = [str(l) for l in f1.cat.categories if l in set(v1) | set(v2)]
    levels += [str(l) for l in f2.cat.categories if str(l) not in levels and l in set(v2)]
    if len(levels) != 2:
        raise AnalysisError(f"McNemar's test needs two measurements with the same two categories, but "
                            f"{len(levels)} categor{'y was' if len(levels) == 1 else 'ies were'} found"
                            f" ({', '.join(levels[:6])}{'…' if len(levels) > 6 else ''}).")
    l1, l2 = levels
    v1, v2 = v1.astype(str), v2.astype(str)
    a = int(np.sum((v1 == l1) & (v2 == l1)))
    b = int(np.sum((v1 == l1) & (v2 == l2)))
    c = int(np.sum((v1 == l2) & (v2 == l1)))
    d = int(np.sum((v1 == l2) & (v2 == l2)))
    n = a + b + c + d
    conf = conf_level(ctx)
    alpha = float(ctx.opt("alpha", 0.05))
    cl = pct(conf)
    r = Results()
    t = r.table("Paired table (number of subjects)",
                [("row", f"{n1} \\ {n2}", "text"), ("c0", l1, "int"), ("c1", l2, "int"), ("total", "Total", "int")])
    t.add_row(row=l1, c0=a, c1=b, total=a + b)
    t.add_row(row=l2, c0=c, c1=d, total=c + d)
    t.add_row(row="Total", c0=a + c, c1=b + d, total=n)
    if dropped:
        t.note(f"{dropped} row(s) with a missing value were excluded.")
    dt = r.table("Discordant pairs",
                 [("what", "Pairs", "text"), ("count", "Number", "int"), ("frac", "% of all pairs", "pct")])
    dt.add_row(what=f"{l1} → {l2}", count=b, frac=b / n)
    dt.add_row(what=f"{l2} → {l1}", count=c, frac=c / n)
    dt.add_row(what="Concordant (no change)", count=a + d, frac=(a + d) / n)
    dt.note("Only the discordant pairs (subjects whose two measurements differ) carry information about a change.")
    m = b + c
    tt = r.table("McNemar's test",
                 [("test", "Test", "text"), ("stat", "Chi-square", "num"), ("df", "df", "int"), ("p", "P value", "p"),
                  ("stars", "P value summary", "text"), ("sig", f"Significant (α = {alpha:g})?", "text")])
    if m == 0:
        p_exact = 1.0
        tt.add_row(test="Exact (binomial)", p=1.0, stars=stars(1.0), sig="No")
        tt.note("There are no discordant pairs, so there is no evidence of any change.")
    else:
        p_exact = float(stats.binomtest(b, m, 0.5).pvalue)
        chi_cc = (abs(b - c) - 1) ** 2 / m
        chi = (b - c) ** 2 / m
        tt.add_row(test="Exact (binomial)", p=p_exact, stars=stars(p_exact), sig="Yes" if p_exact < alpha else "No")
        for name, s in (("Chi-square with continuity correction", chi_cc), ("Chi-square, uncorrected", chi)):
            p = float(stats.chi2.sf(s, 1))
            tt.add_row(test=name, stat=float(s), df=1, p=p, stars=stars(p), sig="Yes" if p < alpha else "No")
        tt.note("Two-sided P values. The exact test is a binomial test of the discordant pairs (expected 50:50) "
                "and is preferred when there are fewer than about 25 discordant pairs.")
    ef = r.table("Size of the change",
                 [("measure", "Measure", "text"), ("value", "Value", "num"), ("lo", f"{cl} CI lower", "num"),
                  ("hi", f"{cl} CI upper", "num"), ("method", "CI method", "text")])
    p1, p2 = (a + b) / n, (a + c) / n
    ef.add_row(measure=f"Proportion “{l1}” in {n1}", value=p1, lo=ci_wilson(a + b, n, conf)[0],
               hi=ci_wilson(a + b, n, conf)[1], method="Wilson")
    ef.add_row(measure=f"Proportion “{l1}” in {n2}", value=p2, lo=ci_wilson(a + c, n, conf)[0],
               hi=ci_wilson(a + c, n, conf)[1], method="Wilson")
    diff = p2 - p1
    se = math.sqrt(max(0.0, b + c - (b - c) ** 2 / n)) / n
    z = stats.norm.ppf(0.5 + conf / 2)
    ef.add_row(measure="Difference in proportions (second − first)", value=diff, lo=diff - z * se, hi=diff + z * se,
               method="Wald (paired)")
    if m > 0 and c > 0 and b > 0:
        lo_p, hi_p = ci_clopper_pearson(b, m, conf)
        ef.add_row(measure=f"Odds ratio of discordant pairs ({l1} → {l2}) / ({l2} → {l1})", value=b / c,
                   lo=lo_p / (1 - lo_p), hi=hi_p / (1 - hi_p) if hi_p < 1 else math.inf,
                   method="exact conditional (Clopper–Pearson)")
    else:
        ef.add_row(measure=f"Odds ratio of discordant pairs ({l1} → {l2}) / ({l2} → {l1})",
                   value=math.nan, method="not defined (a discordant count is zero)")
    r.summary(
        (f"The proportion of “{l1}” differs significantly between {n1} and {n2}" if p_exact < alpha else
         f"The change in the proportion of “{l1}” between {n1} and {n2} is not statistically significant")
        + f" ({p_text(p_exact)}; exact McNemar test). {fnum(100 * p1, 3)}% vs. {fnum(100 * p2, 3)}%; "
          f"{b} subject{'s' if b != 1 else ''} changed from {l1} to {l2} and {c} from {l2} to {l1} (n = {n} pairs).")
    return r


# --------------------------------------------------------------------------
# CI of a proportion
# --------------------------------------------------------------------------

@analysis(
    id="proportions_ci", name="Confidence interval of a proportion", category=CAT,
    description="Wilson, Clopper–Pearson (exact), Agresti–Coull and Wald confidence intervals of a proportion.",
    keywords="proportion confidence interval wilson clopper pearson exact agresti coull wald binomial percentage",
    options=[
        opt.choice("data_form", "Data are", [("counts", "Numbers typed here"), ("raw", "A categorical variable")],
                   default="counts", section="Data"),
        opt.integer("numerator", "Numerator (number of events)", None, min=0, section="Data",
                    visible={"data_form": "counts"}),
        opt.integer("denominator", "Denominator (total)", None, min=1, section="Data",
                    visible={"data_form": "counts"}),
        opt.variable("variable", "Variable", opt.ANY, section="Data", visible={"data_form": "raw"}),
        opt.level("success", "Category counted", sources=["variable"], section="Data",
                  visible={"data_form": "raw"}),
        conf_opt(section="Options"),
    ])
def proportions_ci(ctx: Context) -> Results:
    if ctx.opt("data_form", "counts") == "raw":
        vn = ctx.var("variable")
        f = ctx.factor(vn)
        labels = [str(l) for l in f.cat.categories]
        n = int(f.notna().sum())
        if n == 0:
            raise AnalysisError(f"“{vn}” has no values.")
        level = labels[_pick(labels, ctx.opt("success", ""), "categories")]
        k = int((f.astype(object) == level).sum())
        what = f" for “{level}” in {vn}"
    else:
        k, n = ctx.opt("numerator"), ctx.opt("denominator")
        if k is None or n is None:
            raise AnalysisError("Type the numerator (number of events) and the denominator (total).")
        k, n = int(k), int(n)
        if n < 1:
            raise AnalysisError("The denominator must be at least 1.")
        if k > n:
            raise AnalysisError(f"The numerator ({k}) cannot be larger than the denominator ({n}).")
        what = ""
    conf = conf_level(ctx)
    cl = pct(conf)
    p = k / n
    r = Results()
    t = r.table(f"{cl} confidence interval of a proportion ({k} / {n})",
                [("method", "Method", "text"), ("prop", "Proportion", "num"), ("lo", "Lower limit", "num"),
                 ("hi", "Upper limit", "num"), ("width", "Width", "num"), ("pct", "As percentages", "text")])
    cis = {}
    for name, fn in (("Wilson (score)", ci_wilson), ("Clopper–Pearson (exact)", ci_clopper_pearson),
                     ("Agresti–Coull", ci_agresti_coull), ("Wald (normal approximation)", ci_wald)):
        lo, hi = fn(k, n, conf)
        cis[name] = (lo, hi)
        t.add_row(method=name, prop=p, lo=lo, hi=hi, width=hi - lo,
                  pct=f"{100 * p:.4g}% ({100 * lo:.4g}% to {100 * hi:.4g}%)")
    t.note("Wilson's score interval is recommended for general use. Clopper–Pearson is 'exact' but conservative "
           "(wider than necessary). The Wald interval is the textbook formula p ± z·√(p(1−p)/n); it is unreliable "
           "for small n or proportions near 0 or 1 and is shown for comparison only.")
    if k in (0, n):
        t.note("The observed proportion is exactly " + ("0" if k == 0 else "1")
               + ", so the Wald interval has zero width and should not be used.")
    w = cis["Wilson (score)"]
    r.summary(f"{k} of {n} = {fnum(100 * p, 4)}%{what}; {cl} CI {fnum(100 * w[0], 4)}% to "
              f"{fnum(100 * w[1], 4)}% (Wilson score method).")
    return r
