"""Tests for the R-powered analyses (ANOVA family, regression models, Cox
regression, reliability and factor analysis).

Key numbers are compared with independent reference computations (scipy,
statsmodels, lifelines, numpy formulas); user-error paths must give a helpful
plain-English message instead of a raw R error.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis_ui import importer
from analysis_ui.core import registry, run_analysis
from analysis_ui.core.rbridge import r_info

pytestmark = pytest.mark.skipif(not r_info()["available"], reason="R is not available")

registry.discover()
SAMPLES = Path(importer.__file__).parent / "sample_data"

ALL_IDS = ["anova_oneway", "kruskal_wallis", "anova_rm_oneway", "friedman", "anova_factorial",
           "anova_mixed", "mixed_model", "nested_anova", "multiple_regression", "logistic_regression",
           "poisson_regression", "cox_regression", "reliability", "factor_analysis"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def sample(name: str) -> dict:
    return importer.read_file(name, (SAMPLES / name).read_bytes())[0]


def col(name, values, kind="continuous"):
    return {"name": name, "type": kind, "values": list(values)}


def frame(data: dict) -> pd.DataFrame:
    return pd.DataFrame({c["name"]: c["values"] for c in data["columns"]})


def to_long(data: dict, value="Value", group="Group") -> dict:
    vals, grp = [], []
    for c in data["columns"]:
        for v in c["values"]:
            if v is not None:
                vals.append(v)
                grp.append(c["name"])
    return {"columns": [col(value, vals), col(group, grp, "nominal")]}


def ok(aid: str, data: dict, options: dict) -> dict:
    out = run_analysis(aid, data, options)
    assert out["status"] == "ok", out.get("message")
    return out["results"]


def err(aid: str, data: dict, options: dict) -> str:
    out = run_analysis(aid, data, options)
    assert out["status"] == "error", f"expected an error, got {out['status']}"
    msg = out["message"]
    assert len(msg) > 25, msg
    # never a raw R / Python error
    for bad in ("Error in", "object '", "subscript", "Traceback", "non-numeric argument", "KeyError",
                "contrasts can be applied", "NA/NaN/Inf", "TypeError", "ValueError"):
        assert bad not in msg, msg
    return msg


def table(res: dict, title: str) -> list[dict]:
    tables = [e for e in res["elements"] if e["type"] == "table"]
    hits = ([e for e in tables if e["title"].lower() == title.lower()]
            or [e for e in tables if e["title"].lower().startswith(title.lower())]
            or [e for e in tables if title.lower() in e["title"].lower()])
    assert hits, f"no table matching {title!r}; have {[e.get('title') for e in res['elements'] if e['type'] == 'table']}"
    return hits[0]["rows"]


def has_table(res: dict, title: str) -> bool:
    return any(e["type"] == "table" and title.lower() in e["title"].lower() for e in res["elements"])


def row(rows: list[dict], key: str, value: str) -> dict:
    hits = [r for r in rows if str(r.get(key, "")).startswith(value)]
    assert hits, f"no row with {key} starting with {value!r}: {[r.get(key) for r in rows]}"
    return hits[0]


def measure(res: dict, title: str, name: str, key: str = "value"):
    return row(table(res, title), "Measure", name)[key]


def summary(res: dict) -> str:
    s = [e["content"] for e in res["elements"] if e["type"] == "text" and e.get("kind") == "summary"]
    assert len(s) == 1
    return s[0]


def warnings_of(res: dict) -> str:
    return " ".join([e["content"] for e in res["elements"] if e["type"] == "text" and e.get("kind") == "warning"]
                    + list(res.get("warnings") or []))


def n_plots(res: dict) -> int:
    return sum(e["type"] == "plot" for e in res["elements"])


def close(a, b, rel=1e-6, abs_=1e-9):
    assert a == pytest.approx(b, rel=rel, abs=abs_), f"{a} != {b}"


@pytest.fixture(scope="module")
def groups():
    return sample("treatment_groups.csv")


@pytest.fixture(scope="module")
def unbalanced(groups):
    d = copy.deepcopy(groups)
    d["columns"][1]["values"][2] = None
    d["columns"][1]["values"][7] = None
    d["columns"][3]["values"][0] = None
    return d


def arrays(data):
    return {c["name"]: np.array([v for v in c["values"] if v is not None], float) for c in data["columns"]}


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_all_registered_and_available():
    for aid in ALL_IDS:
        d = registry.ANALYSES[aid]
        assert d.engine == "r"
        pub = d.public()
        assert pub["available"], pub["unavailable_reason"]
        assert (Path(registry.__file__).parent.parent / "r" / "analyses" / d.script).exists()
        names = [o["name"] for o in d.options]
        assert "alpha" in names


# ---------------------------------------------------------------------------
# one-way ANOVA
# ---------------------------------------------------------------------------

def test_oneway_ordinary_matches_scipy(unbalanced):
    a = arrays(unbalanced)
    cols = list(a)
    res = ok("anova_oneway", unbalanced, {"columns": cols, "qq_plot": True, "trend": True})
    f, p = stats.f_oneway(*a.values())
    t = table(res, "One-way ANOVA")
    close(t[0]["F"], f); close(t[0]["p"], p, rel=1e-6)
    assert t[0]["df"] == 3 and t[1]["df"] == sum(map(len, a.values())) - 4
    allv = np.concatenate(list(a.values()))
    close(t[2]["SS"], ((allv - allv.mean()) ** 2).sum())
    eff = table(res, "Effect size")
    close(eff[0]["value"], t[0]["SS"] / t[2]["SS"])
    # descriptives
    d = table(res, "Descriptive statistics")
    for r_, (name, v) in zip(d, a.items()):
        assert r_["Group"] == name and r_["n"] == len(v)
        close(r_["mean"], v.mean()); close(r_["sd"], v.std(ddof=1))
    # Tukey-Kramer against scipy
    tk = stats.tukey_hsd(*a.values())
    ci = tk.confidence_interval(0.95)
    mc = table(res, "Multiple comparisons")
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    assert len(mc) == 6
    for r_, (i, j) in zip(mc, pairs):
        assert r_["Comparison"] == f"{cols[i]} − {cols[j]}"
        close(r_["diff"], a[cols[i]].mean() - a[cols[j]].mean())
        close(r_["p"], tk.pvalue[i, j], rel=1e-5, abs_=1e-9)
        close(r_["lower"], ci.low[i, j], rel=1e-5); close(r_["upper"], ci.high[i, j], rel=1e-5)
    # assumption checks
    eq = table(res, "Equal variances")
    lev = stats.levene(*a.values(), center="median")
    close(eq[0]["statistic"], lev.statistic); close(eq[0]["p"], lev.pvalue)
    bart = stats.bartlett(*a.values())
    close(eq[1]["statistic"], bart.statistic); close(eq[1]["p"], bart.pvalue)
    resid = np.concatenate([v - v.mean() for v in a.values()])
    norm = table(res, "Normality")
    sw = stats.shapiro(resid)
    close(row(norm, "Test", "Shapiro")["statistic"], sw.statistic, rel=1e-5)
    close(row(norm, "Test", "Shapiro")["p"], sw.pvalue, rel=1e-4)
    k2 = stats.normaltest(resid)
    close(row(norm, "Test", "D'Agostino")["statistic"], k2.statistic, rel=1e-8)
    close(row(norm, "Test", "D'Agostino")["p"], k2.pvalue, rel=1e-8)
    # linear trend: contrast with equally spaced coefficients
    n = np.array([len(v) for v in a.values()]); m = np.array([v.mean() for v in a.values()])
    c = np.arange(4) - 1.5
    mse = t[1]["MS"]
    f_lin = (c @ m) ** 2 / (c ** 2 / n).sum() / mse
    tr = table(res, "linear trend")
    close(tr[0]["F"], f_lin); close(tr[0]["p"], stats.f.sf(f_lin, 1, t[1]["df"]))
    assert n_plots(res) == 2
    assert "differ significantly" in summary(res) and "P " in summary(res)
    graphs = [e for e in res["elements"] if e["type"] == "graph"]
    assert graphs and graphs[0]["graph"]["options"]["columns"] == cols


def test_oneway_long_layout_equals_wide(unbalanced):
    cols = [c["name"] for c in unbalanced["columns"]]
    wide = ok("anova_oneway", unbalanced, {"columns": cols})
    long = ok("anova_oneway", to_long(unbalanced), {"layout": "long", "dep": "Value", "group": "Group"})
    tw, tl = table(wide, "One-way ANOVA"), table(long, "One-way ANOVA")
    close(tw[0]["F"], tl[0]["F"]); close(tw[0]["p"], tl[0]["p"])
    assert [r["Comparison"] for r in table(wide, "Multiple comparisons")] == \
           [r["Comparison"] for r in table(long, "Multiple comparisons")]


def test_oneway_welch_and_brown_forsythe(unbalanced):
    from statsmodels.stats.oneway import anova_oneway
    a = arrays(unbalanced); cols = list(a)
    for variant, use_var, title in (("welch", "unequal", "Welch"), ("brown_forsythe", "bf", "Brown")):
        res = ok("anova_oneway", unbalanced, {"columns": cols, "variant": variant, "compare": "none"})
        ref = anova_oneway(list(a.values()), use_var=use_var, welch_correction=True)
        t = table(res, title)[0]
        close(t["F"], ref.statistic); close(t["df2"], ref.df_denom)
        # numerator df = k - 1 as in Brown & Forsythe (1974), SPSS and Prism (statsmodels also
        # applies Mehrotra's correction to the numerator df for the BF test)
        close(t["p"], ref.pvalue if variant == "welch" else stats.f.sf(ref.statistic, 3, ref.df_denom))


def test_oneway_dunnett_matches_scipy(unbalanced):
    a = arrays(unbalanced); cols = list(a)
    res = ok("anova_oneway", unbalanced, {"columns": cols, "compare": "control", "control": "Drug B",
                                          "posthoc": "dunnett"})
    others = [c for c in cols if c != "Drug B"]
    ref = stats.dunnett(*[a[c] for c in others], control=a["Drug B"], random_state=np.random.default_rng(1))
    mc = table(res, "each group vs Drug B")
    ci = ref.confidence_interval(0.95)
    for r_, name, p, lo, hi in zip(mc, others, ref.pvalue, ci.low, ci.high):
        assert r_["Comparison"] == f"{name} − Drug B"
        close(r_["p"], p, rel=0.02, abs_=2e-4)
        close(r_["lower"], lo, rel=5e-3, abs_=0.05); close(r_["upper"], hi, rel=5e-3, abs_=0.05)


def test_oneway_other_corrections(unbalanced):
    from statsmodels.stats.multitest import multipletests
    a = arrays(unbalanced); cols = list(a)
    n = {k: len(v) for k, v in a.items()}
    dfe = sum(n.values()) - 4
    mse = sum(((v - v.mean()) ** 2).sum() for v in a.values()) / dfe
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    tt = np.array([(a[cols[i]].mean() - a[cols[j]].mean()) / math.sqrt(mse * (1 / n[cols[i]] + 1 / n[cols[j]]))
                   for i, j in pairs])
    raw = 2 * stats.t.sf(np.abs(tt), dfe)
    for method, sm in (("bonferroni", "bonferroni"), ("sidak", "sidak"), ("holm_sidak", "holm-sidak"),
                       ("fdr", "fdr_bh"), ("lsd", None)):
        res = ok("anova_oneway", unbalanced, {"columns": cols, "posthoc": method, "assumptions": False,
                                              "show_plot": False})
        got = [r["p"] for r in table(res, "Multiple comparisons")]
        want = raw if sm is None else multipletests(raw, method=sm)[1]
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-12)
    # Games-Howell: Welch SE and df with the studentized range distribution
    res = ok("anova_oneway", unbalanced, {"columns": cols, "posthoc": "games_howell"})
    for r_, (i, j) in zip(table(res, "Multiple comparisons"), pairs):
        x, y = a[cols[i]], a[cols[j]]
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        se = math.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        q = abs(x.mean() - y.mean()) / se * math.sqrt(2)
        close(r_["df"], df); close(r_["p"], stats.studentized_range.sf(q, 4, df), rel=1e-4, abs_=1e-8)


def test_oneway_two_groups_equals_t_test(groups):
    a = arrays(groups)
    res = ok("anova_oneway", groups, {"columns": ["Control", "Drug B"]})
    t = stats.ttest_ind(a["Control"], a["Drug B"])
    close(table(res, "One-way ANOVA")[0]["F"], t.statistic ** 2)
    close(table(res, "One-way ANOVA")[0]["p"], t.pvalue)


def test_oneway_labels_with_symbols():
    rng = np.random.default_rng(0)
    labels = ["Drug A (10 µM)", "Vehicle + 0.1% DMSO", "β-blocker / high: 5×", "it's `quoted` \"x\""]
    vals, grp = [], []
    for i, lab in enumerate(labels):
        vals += list(rng.normal(10 + i, 1, 6).round(3)); grp += [lab] * 6
    data = {"columns": [col("Réponse (%)", vals), col("Grp № 1", grp, "nominal")]}
    res = ok("anova_oneway", data, {"layout": "long", "dep": "Réponse (%)", "group": "Grp № 1",
                                    "compare": "control", "control": "Vehicle + 0.1% DMSO"})
    assert [r["Group"] for r in table(res, "Descriptive")] == labels
    mc = table(res, "Multiple comparisons")
    assert mc[0]["Comparison"] == "Drug A (10 µM) − Vehicle + 0.1% DMSO" and len(mc) == 3
    res = ok("kruskal_wallis", data, {"layout": "long", "dep": "Réponse (%)", "group": "Grp № 1"})
    assert [r["Group"] for r in table(res, "Medians")] == labels


def test_oneway_errors(groups):
    cols = [c["name"] for c in groups["columns"]]
    assert "choose" in err("anova_oneway", groups, {}).lower()
    assert "two or more groups" in err("anova_oneway", groups, {"columns": cols[:1]})
    single = {"columns": [col("a", [1.0]), col("b", [2.0]), col("c", [3.0])]}
    assert "only one value" in err("anova_oneway", single, {"columns": ["a", "b", "c"]})
    const = {"columns": [col("a", [1.0, 1.0]), col("b", [1.0, 1.0]), col("c", [1.0, 1.0])]}
    assert "identical" in err("anova_oneway", const, {"columns": ["a", "b", "c"]})
    assert "identical" in err("kruskal_wallis", const, {"columns": ["a", "b", "c"]})
    nowithin = {"columns": [col("a", [1.0, 1.0]), col("b", [2.0, 2.0]), col("c", [3.0, 3.0])]}
    assert "no variation within" in err("anova_oneway", nowithin, {"columns": ["a", "b", "c"]})
    one = {"columns": [col("a", [1.0, None, None]), col("b", [2.0, 2.5, 3]), col("c", [3.0, 3.5, 4])]}
    assert "at least two values" in err("anova_oneway", one, {"columns": ["a", "b", "c"], "variant": "welch"})
    assert "Grouping variable" in err("anova_oneway", groups, {"layout": "long", "dep": "Control"})
    # a group with a single value is fine for the ordinary ANOVA
    res = ok("anova_oneway", one, {"columns": ["a", "b", "c"]})
    assert table(res, "Descriptive")[0]["n"] == 1


# ---------------------------------------------------------------------------
# Kruskal-Wallis and Friedman
# ---------------------------------------------------------------------------

def test_kruskal_wallis_and_dunn(unbalanced):
    a = arrays(unbalanced); cols = list(a)
    a["Drug A"][0] = a["Control"][0]          # force a tie
    data = {"columns": [col(k, list(v) + [None] * (12 - len(v))) for k, v in a.items()]}
    ref = stats.kruskal(*a.values())
    allv = np.concatenate(list(a.values())); N = len(allv)
    ranks = stats.rankdata(allv)
    idx = np.cumsum([0] + [len(v) for v in a.values()])
    mean_rank = [ranks[idx[i]:idx[i + 1]].mean() for i in range(4)]
    _, counts = np.unique(allv, return_counts=True)
    sigma2 = N * (N + 1) / 12 - (counts ** 3 - counts).sum() / (12 * (N - 1))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    z = np.array([(mean_rank[i] - mean_rank[j]) / math.sqrt(sigma2 * (1 / len(a[cols[i]]) + 1 / len(a[cols[j]])))
                  for i, j in pairs])
    raw = 2 * stats.norm.sf(np.abs(z))
    for layout in ("wide", "long"):
        d, o = (data, {"columns": cols}) if layout == "wide" else (
            to_long(data), {"layout": "long", "dep": "Value", "group": "Group"})
        res = ok("kruskal_wallis", d, o)
        t = table(res, "Kruskal")[0]
        close(t["H"], ref.statistic); close(t["p"], ref.pvalue); assert t["df"] == 3
        close(t["eps2"], ref.statistic / (N - 1))
        dunn = table(res, "Dunn")
        np.testing.assert_allclose([r["z"] for r in dunn], z, rtol=1e-9)
        np.testing.assert_allclose([r["p"] for r in dunn], np.minimum(1, raw * 6), rtol=1e-9)
        med = table(res, "Medians")
        np.testing.assert_allclose([r["median"] for r in med], [np.median(v) for v in a.values()])
        np.testing.assert_allclose([r["mean_rank"] for r in med], mean_rank)
    res = ok("kruskal_wallis", data, {"columns": cols, "compare": "control", "control": "Drug C", "adjust": "holm"})
    dunn = table(res, "Dunn")
    assert [r["Comparison"] for r in dunn] == [f"{c} − Drug C" for c in cols[:3]]
    from statsmodels.stats.multitest import multipletests
    raw_c = np.array([raw[pairs.index((i, 3))] for i in range(3)])
    np.testing.assert_allclose([r["p"] for r in dunn], multipletests(raw_c, method="holm")[1], rtol=1e-9)
    assert "Kruskal" in summary(res)


def test_friedman_matches_scipy():
    rm = sample("repeated_measures.csv")
    df = frame(rm)
    cols = ["Baseline", "Week 4", "Week 8"]
    x = df[cols].to_numpy(float)
    ref = stats.friedmanchisquare(*x.T)
    res = ok("friedman", rm, {"columns": cols})
    t = table(res, "Friedman test")[0]
    close(t["chi2"], ref.statistic); close(t["p"], ref.pvalue); assert t["n"] == 14
    close(t["W"], ref.statistic / (14 * 2))
    rs = stats.rankdata(x, axis=1).sum(axis=0)
    dunn = table(res, "Dunn")
    pairs = [(0, 1), (0, 2), (1, 2)]
    z = [(rs[i] - rs[j]) / math.sqrt(14 * 3 * 4 / 6) for i, j in pairs]
    np.testing.assert_allclose([r["z"] for r in dunn], z, rtol=1e-9)
    np.testing.assert_allclose([r["p"] for r in dunn], np.minimum(1, 3 * 2 * stats.norm.sf(np.abs(z))), rtol=1e-9)
    # long layout with a subject column, rows shuffled
    long = df.melt(id_vars="Subject", var_name="Visit", value_name="Score").sample(frac=1, random_state=1)
    data = {"columns": [col("Subject", long["Subject"], "nominal"), col("Visit", long["Visit"], "nominal"),
                        col("Score", long["Score"])]}
    data["columns"][1]["levels"] = cols
    res2 = ok("friedman", data, {"layout": "long", "dep": "Score", "group": "Visit", "subject": "Subject"})
    close(table(res2, "Friedman test")[0]["chi2"], ref.statistic)
    # incomplete subjects are dropped and reported
    rm2 = copy.deepcopy(rm); rm2["columns"][2]["values"][0] = None
    res3 = ok("friedman", rm2, {"columns": cols})
    assert table(res3, "Friedman test")[0]["n"] == 13
    msg = err("friedman", {"columns": [col("a", [1.0, None, 3.0]), col("b", [2.0, 3.0, None]),
                                       col("c", [None, 4.0, 5.0])]}, {"columns": ["a", "b", "c"]})
    assert "every condition" in msg


# ---------------------------------------------------------------------------
# repeated-measures one-way ANOVA
# ---------------------------------------------------------------------------

def _gg_epsilon(x: np.ndarray) -> float:
    """Greenhouse-Geisser epsilon from the double-centred covariance matrix."""
    k = x.shape[1]
    s = np.cov(x, rowvar=False)
    num = k ** 2 * (np.diag(s).mean() - s.mean()) ** 2
    den = (k - 1) * ((s ** 2).sum() - 2 * k * (s.mean(axis=1) ** 2).sum() + k ** 2 * s.mean() ** 2)
    return num / den


def test_rm_oneway_matches_statsmodels():
    from statsmodels.stats.anova import AnovaRM
    rm = sample("repeated_measures.csv")
    df = frame(rm); cols = ["Baseline", "Week 4", "Week 8"]
    x = df[cols].to_numpy(float); n, k = x.shape
    long = df.melt(id_vars="Subject", var_name="Visit", value_name="Score")
    ref = AnovaRM(long, "Score", "Subject", within=["Visit"]).fit().anova_table
    f = ref["F Value"].iloc[0]
    eps = _gg_epsilon(x)
    res = ok("anova_rm_oneway", rm, {"columns": cols})
    t = table(res, "Repeated-measures one-way ANOVA")
    close(t[0]["F"], f); assert t[0]["df"] == 2 and t[2]["df"] == 26 and t[1]["df"] == 13
    close(t[0]["p"], stats.f.sf(f, 2 * eps, 26 * eps), rel=1e-6)
    close(t[0]["dfn"], 2 * eps)
    sph = table(res, "Sphericity")[0]
    close(sph["gg"], eps)
    hf = (n * (k - 1) * eps - 2) / ((k - 1) * (n - 1 - (k - 1) * eps))
    close(sph["hf"], min(1, hf))
    close(sph["lb"], 0.5)
    # Mauchly's W for k = 3 from the orthonormal contrasts
    c = np.array([[1, -1, 0], [1, 1, -2]], float); c /= np.linalg.norm(c, axis=1, keepdims=True)
    s = c @ np.cov(x, rowvar=False) @ c.T
    w = np.linalg.det(s) / (np.trace(s) / 2) ** 2
    close(sph["W"], w)
    # matching effective?
    grand = x.mean()
    ss_s = k * ((x.mean(axis=1) - grand) ** 2).sum()
    close(t[1]["SS"], ss_s)
    close(t[1]["p"], stats.f.sf((ss_s / 13) / t[2]["MS"], 13, 26))
    # comparisons: paired t tests referred to the studentized range distribution (as Prism)
    mc = table(res, "Multiple comparisons")
    for r_, (i, j) in zip(mc, [(0, 1), (0, 2), (1, 2)]):
        tt = stats.ttest_rel(x[:, i], x[:, j])
        close(r_["t"], tt.statistic)
        close(r_["p"], stats.studentized_range.sf(abs(tt.statistic) * math.sqrt(2), 3, n - 1), rel=1e-5)
        assert r_["df"] == n - 1
    # sphericity assumed
    res = ok("anova_rm_oneway", rm, {"columns": cols, "sphericity": "assume", "posthoc": "bonferroni"})
    t = table(res, "Repeated-measures one-way ANOVA")
    close(t[0]["p"], ref["Pr > F"].iloc[0])
    mc = table(res, "Multiple comparisons")
    se = math.sqrt(2 * t[2]["MS"] / n)
    close(mc[0]["se"], se); assert mc[0]["df"] == 26
    close(mc[0]["p"], min(1, 3 * 2 * stats.t.sf(abs(mc[0]["diff"] / se), 26)))
    # Huynh-Feldt
    res = ok("anova_rm_oneway", rm, {"columns": cols, "sphericity": "hf", "compare": "none"})
    close(table(res, "Repeated-measures one-way ANOVA")[0]["p"], stats.f.sf(f, 2 * min(1, hf), 26 * min(1, hf)))
    # long layout, shuffled, subject IDs
    sh = long.sample(frac=1, random_state=3)
    data = {"columns": [col("Subject", sh["Subject"], "nominal"), col("Visit", sh["Visit"], "nominal"),
                        col("Score", sh["Score"])]}
    data["columns"][1]["levels"] = cols
    res = ok("anova_rm_oneway", data, {"layout": "long", "dep": "Score", "group": "Visit", "subject": "Subject"})
    close(table(res, "Repeated-measures one-way ANOVA")[0]["F"], f)


def test_rm_oneway_missing_values_and_mixed_model():
    rm = sample("repeated_measures.csv"); cols = ["Baseline", "Week 4", "Week 8"]
    full = ok("anova_rm_oneway", rm, {"columns": cols, "sphericity": "assume"})
    f_full = table(full, "Repeated-measures one-way ANOVA")[0]
    # complete data: the mixed model reproduces the sphericity-assumed ANOVA
    mm = ok("anova_rm_oneway", rm, {"columns": cols, "missing": "mixed"})
    t = table(mm, "Mixed-effects model")[0]
    close(t["F"], f_full["F"], rel=1e-5); close(t["dfd"], 26, rel=1e-4); close(t["p"], f_full["p"], rel=1e-4)
    # one missing value: classic analysis drops the subject and says so
    rm2 = copy.deepcopy(rm); rm2["columns"][2]["values"][3] = None
    res = ok("anova_rm_oneway", rm2, {"columns": cols})
    t = table(res, "Repeated-measures one-way ANOVA")
    assert t[1]["df"] == 12
    notes = " ".join(e for el in res["elements"] if el["type"] == "table" for e in el["footnotes"])
    assert "1 subject with missing values was excluded" in notes
    # mixed model keeps all 14 subjects; compare with statsmodels MixedLM (REML)
    import statsmodels.formula.api as smf
    df = frame(rm2).melt(id_vars="Subject", var_name="Visit", value_name="Score").dropna()
    ref = smf.mixedlm("Score ~ C(Visit)", df, groups=df["Subject"]).fit(reml=True)
    res = ok("anova_rm_oneway", rm2, {"columns": cols, "missing": "mixed"})
    vc = table(res, "Random effects")
    close(vc[0]["Variance"], float(ref.cov_re.iloc[0, 0]), rel=2e-3)
    close(vc[1]["Variance"], ref.scale, rel=2e-3)
    emm = table(res, "Estimated marginal means")
    close(emm[1]["emmean"] - emm[0]["emmean"], ref.params["C(Visit)[T.Week 4]"], rel=2e-3)
    assert table(res, "Descriptive")[1]["n"] == 13
    msg = err("anova_rm_oneway", {"columns": [col("a", [1.0, 2.0, 3.0]), col("b", [2.0, 3.0, 4.0]),
                                              col("c", [3.0, 4.0, 5.0])]}, {"columns": ["a", "b", "c"]})
    assert "no residual variation" in msg
    assert "at least 2 conditions" in err("anova_rm_oneway", rm, {"columns": ["Baseline"]})


# ---------------------------------------------------------------------------
# factorial ANOVA
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def twoway_unbalanced():
    d = sample("two_way.csv")
    d = copy.deepcopy(d)
    for i in (0, 1, 9, 20, 21, 22, 40):
        d["columns"][0]["values"][i] = None
    return d


def _sm_anova(df, formula, typ):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    return sm.stats.anova_lm(smf.ols(formula, df).fit(), typ=typ)


def test_factorial_type3_and_type2_unbalanced(twoway_unbalanced):
    df = frame(twoway_unbalanced).dropna()
    opts = {"dep": "Response", "factors": ["Treatment", "Sex"], "posthoc": "both", "resid_plots": True}
    res = ok("anova_factorial", twoway_unbalanced, opts)
    ref = _sm_anova(df, "Response ~ C(Treatment, Sum) * C(Sex, Sum)", 3)
    t = table(res, "ANOVA table")
    for name, key in (("Treatment", "C(Treatment, Sum)"), ("Sex", "C(Sex, Sum)"),
                      ("Treatment × Sex", "C(Treatment, Sum):C(Sex, Sum)")):
        r_ = [x for x in t if x["Source"] == name][0]
        close(r_["SS"], ref.loc[key, "sum_sq"], rel=1e-8); close(r_["F"], ref.loc[key, "F"], rel=1e-8)
        close(r_["p"], ref.loc[key, "PR(>F)"], rel=1e-7)
        close(r_["eta_p"], ref.loc[key, "sum_sq"] / (ref.loc[key, "sum_sq"] + ref.loc["Residual", "sum_sq"]))
    close(row(t, "Source", "Residual")["SS"], ref.loc["Residual", "sum_sq"])
    assert "7 rows with missing values were excluded" in " ".join(
        f for e in res["elements"] if e["type"] == "table" for f in e["footnotes"])
    # type II
    res2 = ok("anova_factorial", twoway_unbalanced, dict(opts, ss_type="2", posthoc="none"))
    ref2 = _sm_anova(df, "Response ~ C(Treatment) * C(Sex)", 2)
    t2 = table(res2, "ANOVA table")
    close(row(t2, "Source", "Treatment")["SS"], ref2.loc["C(Treatment)", "sum_sq"], rel=1e-8)
    close(row(t2, "Source", "Sex")["F"], ref2.loc["C(Sex)", "F"], rel=1e-8)
    assert abs(row(t2, "Source", "Sex")["SS"] - row(t, "Source", "Sex")["SS"]) > 1e-6   # they differ when unbalanced
    # descriptives and estimated marginal means of the full model = cell means
    cells = df.groupby(["Treatment", "Sex"], observed=True)["Response"].agg(["count", "mean", "std"])
    desc = table(res, "Descriptive statistics")
    assert len(desc) == 6
    for r_ in desc:
        c = cells.loc[(r_["Treatment"], r_["Sex"])]
        assert r_["n"] == c["count"]; close(r_["mean"], c["mean"]); close(r_["sd"], c["std"])
    emm = table(res, "Estimated marginal means: Treatment × Sex")
    for r_ in emm:
        close(r_["emmean"], cells.loc[(r_["Treatment"], r_["Sex"]), "mean"])
    # marginal mean of a factor = unweighted mean of the cell means
    em_t = table(res, "Estimated marginal means: Treatment")
    assert [r_["Treatment"] for r_ in em_t] == ["Vehicle", "Low dose", "High dose"]      # level order preserved
    close(em_t[0]["emmean"], cells.loc["Vehicle", "mean"].mean())
    # simple effects: Treatment within Sex (default), 3 comparisons per level of Sex
    se = table(res, "Simple effects: Treatment within each level of Sex")
    assert len(se) == 6 and {r_["Sex"] for r_ in se} == {"Female", "Male"}
    mse = ref.loc["Residual", "sum_sq"] / ref.loc["Residual", "df"]
    r0 = [r_ for r_ in se if r_["Sex"] == "Female" and r_["Comparison"] == "Vehicle − Low dose"][0]
    n1, n2 = cells.loc[("Vehicle", "Female"), "count"], cells.loc[("Low dose", "Female"), "count"]
    diff = cells.loc[("Vehicle", "Female"), "mean"] - cells.loc[("Low dose", "Female"), "mean"]
    close(r0["diff"], diff); close(r0["se"], math.sqrt(mse * (1 / n1 + 1 / n2)))
    close(r0["p"], stats.studentized_range.sf(abs(diff / r0["se"]) * math.sqrt(2), 3, ref.loc["Residual", "df"]),
          rel=1e-4)
    # the main-effect comparison of a two-level factor equals its F test
    me = table(res, "Main effect comparisons: Sex")
    close(me[0]["p"], row(t, "Source", "Sex")["p"], rel=1e-6)
    lev = table(res, "Equal variances")[0]
    ref_lev = stats.levene(*[g["Response"].to_numpy() for _, g in df.groupby(["Treatment", "Sex"], observed=True)],
                           center="median")
    close(lev["F"], ref_lev.statistic); close(lev["p"], ref_lev.pvalue)
    assert n_plots(res) == 2
    g = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert g["type"] == "grouped_bar" and g["options"] == {"layout": "long", "dep": "Response", "x": "Treatment",
                                                           "group": "Sex"}
    # explicit simple-effects choice and Dunnett vs a named control
    res3 = ok("anova_factorial", twoway_unbalanced, dict(opts, posthoc="both", simple_factor="Sex",
                                                         simple_by="Treatment", compare="control",
                                                         control="Low dose", posthoc_method="dunnett"))
    se3 = table(res3, "Simple effects: Sex within each level of Treatment")
    assert len(se3) == 3
    me3 = table(res3, "Main effect comparisons: Treatment")
    assert [r_["Comparison"] for r_ in me3] == ["Vehicle − Low dose", "High dose − Low dose"]


def test_factorial_main_effects_ancova_and_three_way(twoway_unbalanced):
    rng = np.random.default_rng(4)
    d = copy.deepcopy(twoway_unbalanced)
    n = len(d["columns"][0]["values"])
    d["columns"].append(col("Body weight (g)", rng.normal(25, 3, n).round(2)))
    d["columns"].append(col("Batch", rng.choice(["b1", "b2"], n), "nominal"))
    df = frame(d).dropna().rename(columns={"Body weight (g)": "bw"})
    res = ok("anova_factorial", d, {"dep": "Response", "factors": ["Treatment", "Sex"],
                                    "covariates": ["Body weight (g)"], "model": "main"})
    ref = _sm_anova(df, "Response ~ C(Treatment, Sum) + C(Sex, Sum) + bw", 3)
    t = table(res, "ANCOVA table")
    close(row(t, "Source", "Treatment")["F"], ref.loc["C(Treatment, Sum)", "F"], rel=1e-8)
    close(row(t, "Source", "Body weight (g)")["p"], ref.loc["bw", "PR(>F)"], rel=1e-7)
    res = ok("anova_factorial", d, {"dep": "Response", "factors": ["Treatment", "Sex", "Batch"], "model": "twoway",
                                    "posthoc": "none"})
    ref = _sm_anova(df, "Response ~ (C(Treatment, Sum) + C(Sex, Sum) + C(Batch, Sum))**2", 3)
    t = table(res, "ANOVA table")
    close(row(t, "Source", "Sex × Batch")["F"], ref.loc["C(Sex, Sum):C(Batch, Sum)", "F"], rel=1e-8)
    assert not any("Treatment × Sex × Batch" == r_["Source"] for r_ in t)
    res = ok("anova_factorial", d, {"dep": "Response", "factors": ["Treatment", "Sex", "Batch"]})
    ref = _sm_anova(df, "Response ~ C(Treatment, Sum) * C(Sex, Sum) * C(Batch, Sum)", 3)
    close(row(table(res, "ANOVA table"), "Source", "Treatment × Sex × Batch")["F"],
          ref.loc["C(Treatment, Sum):C(Sex, Sum):C(Batch, Sum)", "F"], rel=1e-8)
    assert "Three-way ANOVA" in summary(res)


def test_factorial_one_factor_equals_oneway(groups):
    long = to_long(groups)
    a = arrays(groups)
    res = ok("anova_factorial", long, {"dep": "Value", "factors": ["Group"]})
    f, p = stats.f_oneway(*a.values())
    r_ = row(table(res, "ANOVA table"), "Source", "Group")
    close(r_["F"], f); close(r_["p"], p)


def test_factorial_errors(twoway_unbalanced):
    tw = sample("two_way.csv")
    assert "Factors" in err("anova_factorial", tw, {"dep": "Response"})
    assert "more than one role" in err("anova_factorial", tw, {"dep": "Response", "factors": ["Treatment", "Treatment"]})
    cells = [(t, s) for t in "ABC" for s in "XY"]
    one = {"columns": [col("T", [c[0] for c in cells], "nominal"), col("S", [c[1] for c in cells], "nominal"),
                       col("y", [1.0, 2.5, 2.0, 4.0, 3.5, 7.0])]}
    msg = err("anova_factorial", one, {"dep": "y", "factors": ["T", "S"]})
    assert "one observation per combination" in msg and "Main effects only" in msg
    res = ok("anova_factorial", one, {"dep": "y", "factors": ["T", "S"], "model": "main"})   # additive model works
    assert row(table(res, "ANOVA table"), "Source", "Residual")["df"] == 2
    empty = copy.deepcopy(tw)
    for i, (t, s) in enumerate(zip(tw["columns"][1]["values"], tw["columns"][2]["values"])):
        if t == "Low dose" and s == "Male":
            empty["columns"][0]["values"][i] = None
    msg = err("anova_factorial", empty, {"dep": "Response", "factors": ["Treatment", "Sex"]})
    assert "no data" in msg and "Low dose" in msg
    res = ok("anova_factorial", empty, {"dep": "Response", "factors": ["Treatment", "Sex"], "model": "main"})
    assert "no data" in warnings_of(res)
    const = copy.deepcopy(tw); const["columns"][0]["values"] = [5.0] * 48
    assert "identical" in err("anova_factorial", const, {"dep": "Response", "factors": ["Treatment"]})
    single = copy.deepcopy(tw); single["columns"][2]["values"] = ["Male"] * 48
    assert "only 1 level" in err("anova_factorial", single, {"dep": "Response", "factors": ["Treatment", "Sex"]})


# ---------------------------------------------------------------------------
# repeated-measures / mixed ANOVA
# ---------------------------------------------------------------------------

def _split_plot(seed=1, n_per=8, labels=("Drug A (10 µM)", "Placebo"), times=("Pre", "Week 4", "Week-8")):
    rng = np.random.default_rng(seed)
    rows = []
    for g, lab in enumerate(labels):
        for s in range(n_per):
            b = rng.normal(0, 2)
            for j, t in enumerate(times):
                rows.append((f"S{g}_{s}", lab, t, round(10 + b + j * (1.5 if g == 0 else 0.2) + rng.normal(), 3)))
    df = pd.DataFrame(rows, columns=["Subj", "Arm", "Time", "Score"])
    data = {"columns": [col("Subj", df["Subj"], "nominal"), col("Arm", df["Arm"], "nominal"),
                        col("Time", df["Time"], "nominal"), col("Score (a.u.)", df["Score"])]}
    return df, data


def _split_plot_reference(df):
    """Textbook sums of squares for a balanced one-between one-within design."""
    wide = df.pivot(index="Subj", columns="Time", values="Score")
    arm = df.drop_duplicates("Subj").set_index("Subj")["Arm"].loc[wide.index]
    x = wide.to_numpy(); n, k = x.shape
    grand = x.mean()
    groups = [x[(arm == a).to_numpy()] for a in arm.unique()]
    g = len(groups)
    ss_a = sum(len(xx) * k * (xx.mean() - grand) ** 2 for xx in groups)
    ss_subj = sum(k * ((xx.mean(axis=1) - xx.mean()) ** 2).sum() for xx in groups)
    ss_t = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_cells = sum(len(xx) * ((xx.mean(axis=0) - grand) ** 2).sum() for xx in groups)
    ss_at = ss_cells - ss_a - ss_t
    ss_err = ((x - grand) ** 2).sum() - ss_a - ss_subj - ss_t - ss_at
    dfs, dfe = n - g, (n - g) * (k - 1)
    return {"Arm": (ss_a / (g - 1)) / (ss_subj / dfs), "Time": (ss_t / (k - 1)) / (ss_err / dfe),
            "Arm × Time": (ss_at / ((g - 1) * (k - 1))) / (ss_err / dfe), "dfs": dfs, "dfe": dfe,
            "ss": {"Arm": ss_a, "Time": ss_t, "Arm × Time": ss_at}, "ss_subj": ss_subj, "ss_err": ss_err}


def test_mixed_anova_long_matches_textbook():
    df, data = _split_plot()
    ref = _split_plot_reference(df)
    opts = {"data_form": "long", "dep": "Score (a.u.)", "subject": "Subj", "within": ["Time"], "between": ["Arm"],
            "sphericity": "none", "posthoc": "simple", "simple_factor": "Arm", "simple_by": "Time"}
    res = ok("anova_mixed", data, opts)
    t = table(res, "Repeated-measures ANOVA")
    for name in ("Arm", "Time", "Arm × Time"):
        r_ = [x for x in t if x["Source"] == name][0]
        close(r_["F"], ref[name], rel=1e-8); close(r_["SS"], ref["ss"][name], rel=1e-8)
        dfe = ref["dfs"] if name == "Arm" else ref["dfe"]
        assert r_["dfe"] == dfe
        close(r_["p"], stats.f.sf(ref[name], r_["df"], dfe), rel=1e-7)
    # generalized eta squared (all factors manipulated): SS / (SS + all error SS)
    close(row(t, "Source", "Time")["ges"], ref["ss"]["Time"] / (ref["ss"]["Time"] + ref["ss_subj"] + ref["ss_err"]))
    # level labels with spaces / symbols survive, in the original order
    desc = table(res, "Descriptive statistics")
    assert [r_["Time"] for r_ in desc[:2]] == ["Pre", "Pre"] and desc[0]["Arm"] == "Drug A (10 µM)"
    assert {r_["Time"] for r_ in desc} == {"Pre", "Week 4", "Week-8"}
    se = table(res, "Simple effects: Arm within each level of Time")
    assert [r_["Time"] for r_ in se] == ["Pre", "Week 4", "Week-8"]
    for r_ in se:      # between-group comparison at one time point = Student t test
        a = df[(df.Time == r_["Time"]) & (df.Arm == "Drug A (10 µM)")]["Score"]
        b = df[(df.Time == r_["Time"]) & (df.Arm == "Placebo")]["Score"]
        tt = stats.ttest_ind(a, b)
        assert r_["Comparison"] == "Drug A (10 µM) − Placebo"
        close(r_["t"], tt.statistic, rel=1e-6); close(r_["p"], tt.pvalue, rel=1e-6)
    # Greenhouse-Geisser: P of within effects uses eps-scaled df
    res_gg = ok("anova_mixed", data, dict(opts, sphericity="gg", posthoc="none"))
    r_ = row(table(res_gg, "Repeated-measures ANOVA"), "Source", "Time")
    assert 0.5 <= r_["eps"] <= 1
    close(r_["p"], stats.f.sf(ref["Time"], 2 * r_["eps"], ref["dfe"] * r_["eps"]), rel=1e-6)
    close(table(res_gg, "Sphericity")[0]["gg"], r_["eps"])
    # the mixed-effects model reproduces the sphericity-assumed ANOVA for complete balanced data
    res_l = ok("anova_mixed", data, dict(opts, engine="lmm", posthoc="none"))
    tl = table(res_l, "tests of fixed effects")
    for name in ("Arm", "Time", "Arm × Time"):
        close(row(tl, "Source", name)["F"], ref[name], rel=1e-4)
    # missing value: ANOVA drops the subject, mixed model keeps it
    d2 = copy.deepcopy(data); d2["columns"][3]["values"][4] = None
    r1 = ok("anova_mixed", d2, dict(opts, posthoc="none"))
    assert row(table(r1, "Repeated-measures ANOVA"), "Source", "Arm")["dfe"] == ref["dfs"] - 1
    assert "excluded" in " ".join(f for e in r1["elements"] if e["type"] == "table" for f in e["footnotes"])
    r2 = ok("anova_mixed", d2, dict(opts, engine="lmm", posthoc="none"))
    assert "16 subjects, 47 observations" in " ".join(
        f for e in r2["elements"] if e["type"] == "table" for f in e["footnotes"])


def test_mixed_anova_wide_layout():
    rm = sample("repeated_measures.csv"); cols = ["Baseline", "Week 4", "Week 8"]
    one = ok("anova_rm_oneway", rm, {"columns": cols})
    f_ref = table(one, "Repeated-measures one-way ANOVA")[0]
    res = ok("anova_mixed", rm, {"data_form": "wide", "rm_columns": cols, "within_name": "Visit"})
    t = table(res, "Repeated-measures ANOVA")
    assert t[0]["Source"] == "Visit"
    close(t[0]["F"], f_ref["F"]); close(t[0]["p"], f_ref["p"], rel=1e-6)
    mc = table(res, "Main effect comparisons: Visit")
    ref_mc = table(one, "Multiple comparisons")
    np.testing.assert_allclose([r_["p"] for r_ in mc], [r_["p"] for r_ in ref_mc], rtol=1e-5)
    assert [r_["Comparison"] for r_ in mc] == [r_["Comparison"] for r_ in ref_mc]
    # with a between-subjects factor
    d = copy.deepcopy(rm)
    d["columns"].append(col("Sex", ["F", "M"] * 7, "nominal"))
    res = ok("anova_mixed", d, {"data_form": "wide", "rm_columns": cols, "between": ["Sex"], "posthoc": "both"})
    t = table(res, "Repeated-measures ANOVA")
    df = frame(d)
    m = df[cols].mean(axis=1)
    tt = stats.ttest_ind(m[df.Sex == "F"], m[df.Sex == "M"])
    close(row(t, "Source", "Sex")["F"], tt.statistic ** 2, rel=1e-8)
    assert has_table(res, "Simple effects: Condition within each level of Sex")


def test_mixed_anova_two_within_factors():
    from statsmodels.stats.anova import AnovaRM
    rng = np.random.default_rng(7)
    rows = [(f"s{s}", a, b, rng.normal(5 + (a == "hi") + 0.5 * (b == "t2") + s * 0.1, 1))
            for s in range(10) for a in ("lo", "hi") for b in ("t1", "t2", "t3")]
    df = pd.DataFrame(rows, columns=["id", "dose", "time", "y"])
    data = {"columns": [col("id", df["id"], "nominal"), col("dose", df["dose"], "nominal"),
                        col("time", df["time"], "nominal"), col("y", df["y"])]}
    ref = AnovaRM(df, "y", "id", within=["dose", "time"]).fit().anova_table
    res = ok("anova_mixed", data, {"dep": "y", "subject": "id", "within": ["dose", "time"], "sphericity": "none"})
    t = table(res, "Repeated-measures ANOVA")
    close(row(t, "Source", "dose")["F"], ref.loc["dose", "F Value"], rel=1e-8)
    close(row(t, "Source", "time")["F"], ref.loc["time", "F Value"], rel=1e-8)
    close(row(t, "Source", "dose × time")["p"], ref.loc["dose:time", "Pr > F"], rel=1e-7)


def test_mixed_anova_errors():
    df, data = _split_plot()
    base = {"data_form": "long", "dep": "Score (a.u.)", "subject": "Subj", "within": ["Time"]}
    assert "Within-subjects factors" in err("anova_mixed", data, {"dep": "Score (a.u.)", "subject": "Subj"})
    msg = err("anova_mixed", data, dict(base, within=["Arm"]))        # Arm does not vary within subjects
    assert "every within-subjects condition" in msg
    msg = err("anova_mixed", data, dict(base, between=["Time"]))
    assert "more than one role" in msg or "not a between-subjects factor" in msg
    d2 = copy.deepcopy(data); d2["columns"].append(col("Phase", list(df["Time"]), "nominal"))
    assert "not a between-subjects factor" in err("anova_mixed", d2, dict(base, between=["Phase"]))
    assert "at least 2" in err("anova_mixed", sample("repeated_measures.csv"),
                               {"data_form": "wide", "rm_columns": ["Baseline"]})


# ---------------------------------------------------------------------------
# linear mixed model and nested ANOVA
# ---------------------------------------------------------------------------

def _clustered(seed=3, n_animals=4, reps=5, sd_animal=1.5):
    rng = np.random.default_rng(seed)
    rows = []
    for g, gm in (("Control", 10), ("Drug", 13), ("Drug B", 12)):
        for a in range(n_animals):
            am = rng.normal(0, sd_animal)
            for _ in range(reps):
                dose = rng.uniform(0, 5)
                rows.append((g, f"A{a + 1}", f"{g}-{a + 1}", round(dose, 3), round(gm + am + 0.4 * dose + rng.normal(), 3)))
    df = pd.DataFrame(rows, columns=["Trt", "Animal", "AnimalID", "Dose", "Y"])
    data = {"columns": [col("Trt", df["Trt"], "nominal"), col("Animal", df["Animal"], "nominal"),
                        col("Animal ID", df["AnimalID"], "nominal"), col("Dose (mg)", df["Dose"]), col("Y", df["Y"])]}
    return df, data


def test_mixed_model_matches_statsmodels():
    import statsmodels.formula.api as smf
    df, data = _clustered()
    ref = smf.mixedlm("Y ~ C(Trt) + Dose", df, groups=df["AnimalID"]).fit(reml=True, method="lbfgs")
    res = ok("mixed_model", data, {"dep": "Y", "factors": ["Trt"], "covariates": ["Dose (mg)"],
                                   "random": ["Animal ID"], "assumptions": True})
    co = table(res, "Fixed-effect coefficients")
    assert [r_["Term"] for r_ in co] == ["Intercept", "Trt [Drug]", "Trt [Drug B]", "Dose (mg)"]
    for r_, key in zip(co, ["Intercept", "C(Trt)[T.Drug]", "C(Trt)[T.Drug B]", "Dose"]):
        close(r_["est"], ref.params[key], rel=2e-3, abs_=2e-3)
        close(r_["se"], ref.bse[key], rel=5e-3)
    vc = table(res, "Random effects")
    v_a, v_e = float(ref.cov_re.iloc[0, 0]), ref.scale
    close(vc[0]["Variance"], v_a, rel=5e-3); close(vc[1]["Variance"], v_e, rel=5e-3)
    close(vc[0]["share"], v_a / (v_a + v_e), rel=5e-3)
    # Nakagawa R2 from the reference fit
    xb = ref.model.exog @ ref.fe_params.to_numpy()
    vf = xb.var(ddof=1)
    close(measure(res, "Model fit", "R² marginal"), vf / (vf + v_a + v_e), rel=5e-3)
    close(measure(res, "Model fit", "R² conditional"), (vf + v_a) / (vf + v_a + v_e), rel=5e-3)
    assert measure(res, "Model fit", "Observations") == 60 and measure(res, "Model fit", "Levels of Animal ID") == 12
    fx = table(res, "Tests of fixed effects")
    cov_row = row(fx, "Source", "Dose (mg)")
    close(cov_row["F"], row(co, "Term", "Dose (mg)")["t"] ** 2, rel=1e-6)     # 1-df F = t squared
    assert has_table(res, "Multiple comparisons: Trt") and has_table(res, "Estimated marginal means: Trt")
    assert n_plots(res) >= 1
    # ML estimation and a random slope run and report sensible output
    res = ok("mixed_model", data, {"dep": "Y", "factors": ["Trt"], "covariates": ["Dose (mg)"],
                                   "random": ["Animal ID"], "slope": "Dose (mg)", "estimation": "ml",
                                   "interactions": "twoway", "posthoc": "none"})
    vc = table(res, "Random effects")
    assert any(r_["Effect"].startswith("Slope: Dose (mg)") for r_ in vc)
    assert has_table(res, "Tests of fixed effects")
    assert any(r_["Source"] == "Trt × Dose (mg)" for r_ in table(res, "Tests of fixed effects"))


def test_mixed_model_errors_and_singular_fit():
    df, data = _clustered()
    assert "Grouping variables" in err("mixed_model", data, {"dep": "Y", "factors": ["Trt"]})
    msg = err("mixed_model", data, {"dep": "Y", "factors": ["Trt"], "random": ["Animal ID"], "slope": "Animal"})
    assert "random-slope variable" in msg
    d = copy.deepcopy(data); d["columns"].append(col("Row", [f"r{i}" for i in range(60)], "nominal"))
    assert "Every row has a different value" in err("mixed_model", d, {"dep": "Y", "factors": ["Trt"], "random": ["Row"]})
    # no between-animal variation -> singular fit explained in plain language
    df0, data0 = _clustered(seed=11, sd_animal=0.0)
    res = ok("mixed_model", data0, {"dep": "Y", "factors": ["Trt"], "random": ["Animal ID"]})
    if table(res, "Random effects")[0]["Variance"] < 1e-8:
        assert "Singular fit" in warnings_of(res)
    # confounded fixed effects
    d = copy.deepcopy(data); d["columns"].append(col("Trt copy", list(df["Trt"]), "nominal"))
    msg = err("mixed_model", d, {"dep": "Y", "factors": ["Trt", "Trt copy"], "random": ["Animal ID"]})
    assert "confounded" in msg


def test_nested_anova_balanced_equals_subgroup_means():
    df, data = _clustered()
    means = df.groupby(["Trt", "Animal"], sort=False)["Y"].mean()
    f, p = stats.f_oneway(*[means.loc[g].to_numpy() for g in ("Control", "Drug", "Drug B")])
    res = ok("nested_anova", data, {"dep": "Y", "group": "Trt", "nested": "Animal"})
    t = table(res, "Nested one-way ANOVA")[0]
    close(t["F"], f, rel=1e-5); close(t["p"], p, rel=1e-4); close(t["df2"], 9, rel=1e-4)
    cmp_ = table(res, "For comparison")[0]
    close(cmp_["F"], f, rel=1e-8)
    gm = table(res, "Group means")
    close(gm[0]["mean"], means.loc["Control"].mean()); assert gm[0]["subgroups"] == 4 and gm[0]["n"] == 20
    assert len(table(res, "Subgroup means")) == 12
    # variance components: balanced ANOVA estimators
    msw = df.groupby("AnimalID")["Y"].var(ddof=1).mean()
    msb = 5 * sum(((means.loc[g] - means.loc[g].mean()) ** 2).sum() for g in ("Control", "Drug", "Drug B")) / 9
    vc = table(res, "Variance components")
    close(vc[1]["Variance"], msw, rel=1e-4); close(vc[0]["Variance"], (msb - msw) / 5, rel=1e-3)
    assert has_table(res, "Multiple comparisons: Trt")
    # nested t test (two groups)
    two = {"columns": [dict(c, values=c["values"][:40]) for c in data["columns"]]}
    res = ok("nested_anova", two, {"dep": "Y", "group": "Trt", "nested": "Animal"})
    t = table(res, "Nested t test")[0]
    tt = stats.ttest_ind(means.loc["Control"], means.loc["Drug"])
    close(t["t"], tt.statistic, rel=1e-5); close(t["p"], tt.pvalue, rel=1e-4)
    assert t["Comparison"] == "Control − Drug"


def test_nested_anova_errors():
    df, data = _clustered()
    d = copy.deepcopy(data); d["columns"].append(col("One", ["x"] * 60, "nominal"))
    assert "only one subgroup" in err("nested_anova", d, {"dep": "Y", "group": "Trt", "nested": "One"})
    d["columns"].append(col("Row", [f"r{i}" for i in range(60)], "nominal"))
    assert "single value" in err("nested_anova", d, {"dep": "Y", "group": "Trt", "nested": "Row"})
    assert "only 1 level" in err("nested_anova", d, {"dep": "Y", "group": "One", "nested": "Animal"})


# ---------------------------------------------------------------------------
# multiple linear regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg():
    return sample("regression.csv")


def test_multiple_regression_matches_statsmodels(reg):
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    df = frame(reg).rename(columns={"Systolic BP": "sbp"})
    df["Sex"] = pd.Categorical(df["Sex"], categories=["Male", "Female"])
    ref = smf.ols("sbp ~ Age + BMI + C(Sex)", df).fit()
    res = ok("multiple_regression", reg, {"dep": "Systolic BP", "covariates": ["Age", "BMI"], "factors": ["Sex"],
                                          "influence": True})
    co = table(res, "Coefficients")
    assert [r_["Term"] for r_ in co] == ["Intercept", "Age", "BMI", "Sex [Female]"]
    keys = ["Intercept", "Age", "BMI", "C(Sex)[T.Female]"]
    ci = ref.conf_int()
    for r_, k in zip(co, keys):
        close(r_["est"], ref.params[k], rel=1e-8); close(r_["se"], ref.bse[k], rel=1e-8)
        close(r_["t"], ref.tvalues[k], rel=1e-8); close(r_["p"], ref.pvalues[k], rel=1e-7, abs_=1e-300)
        close(r_["lower"], ci.loc[k, 0], rel=1e-8); close(r_["upper"], ci.loc[k, 1], rel=1e-8)
    x = ref.model.exog
    for r_, k in zip(co[1:], keys[1:]):
        i = ref.model.exog_names.index(k)
        close(r_["vif"], variance_inflation_factor(x, i), rel=1e-8)
        close(r_["beta"], ref.params[k] * x[:, i].std(ddof=1) / df["sbp"].std(ddof=1), rel=1e-8)
    ms = {r_["Measure"]: r_["value"] for r_ in table(res, "Model summary")}
    close(float(ms["R²"]), ref.rsquared, rel=1e-3); close(float(ms["Adjusted R²"]), ref.rsquared_adj, rel=1e-3)
    close(float(ms["F"]), ref.fvalue, rel=1e-3); close(float(ms["Residual SD (Sy.x)"]), math.sqrt(ref.mse_resid), rel=1e-3)
    close(float(ms["AIC"]), ref.aic + 2, rel=1e-4)     # R counts the error variance as a parameter
    assert ms["Observations"] == "60"
    import statsmodels.api as sm
    a2 = sm.stats.anova_lm(ref, typ=2)
    an = table(res, "ANOVA table")
    close(row(an, "Source", "Age")["SS"], a2.loc["Age", "sum_sq"], rel=1e-8)
    close(row(an, "Source", "Sex")["F"], a2.loc["C(Sex)", "F"], rel=1e-8)
    # Breusch-Pagan (Cook-Weisberg score test on the fitted values)
    lm = het_breuschpagan(ref.resid, np.column_stack([np.ones(len(df)), ref.fittedvalues]), robust=False)
    bp = table(res, "Constant variance")[0]
    close(bp["chi2"], lm[0], rel=1e-6); close(bp["p"], lm[1], rel=1e-6)
    sw = stats.shapiro(ref.resid)
    close(row(table(res, "Normality"), "Test", "Shapiro")["p"], sw.pvalue, rel=1e-4)
    infl = ref.get_influence()
    cook = infl.cooks_distance[0]
    it = table(res, "Influential points")
    assert it[0]["row"] == int(np.argmax(cook)) + 1
    close(it[0]["cook"], cook.max(), rel=1e-8)
    close(it[0]["rstu"], infl.resid_studentized_external[int(np.argmax(cook))], rel=1e-8)
    assert n_plots(res) == 2
    assert "Reference levels: Sex = Male" in " ".join(
        f for e in res["elements"] if e["type"] == "table" for f in e["footnotes"])


def test_multiple_regression_interactions_weights_no_intercept(reg):
    import statsmodels.formula.api as smf
    df = frame(reg).rename(columns={"Systolic BP": "sbp"})
    df["Sex"] = pd.Categorical(df["Sex"], categories=["Male", "Female"])
    base = {"dep": "Systolic BP", "covariates": ["Age", "BMI"], "factors": ["Sex"], "diag_plots": False,
            "show_plot": False, "assumptions": False}
    # all two-way interactions
    ref = smf.ols("sbp ~ (Age + BMI + C(Sex))**2", df).fit()
    res = ok("multiple_regression", reg, dict(base, interactions="twoway"))
    co = table(res, "Coefficients")
    close(row(co, "Term", "Age × BMI")["est"], ref.params["Age:BMI"], rel=1e-7)
    close(row(co, "Term", "Age × Sex [Female]")["p"], ref.pvalues["Age:C(Sex)[T.Female]"], rel=1e-6)
    # custom formula with a transformation, Type III table
    ref = smf.ols("sbp ~ Age * C(Sex, Sum) + I(BMI**2)", df).fit()
    res = ok("multiple_regression", reg, dict(base, interactions="custom", formula="`Age`*`Sex` + I(`BMI`^2)",
                                              ss_type="3"))
    import statsmodels.api as sm
    a3 = sm.stats.anova_lm(ref, typ=3)
    an = table(res, "ANOVA table")
    close(row(an, "Source", "I(BMI^2)")["F"], a3.loc["I(BMI ** 2)", "F"], rel=1e-7)
    close(row(an, "Source", "Sex")["F"], a3.loc["C(Sex, Sum)", "F"], rel=1e-7)
    close(row(an, "Source", "Age × Sex")["p"], a3.loc["Age:C(Sex, Sum)", "PR(>F)"], rel=1e-6)
    # weights
    d = copy.deepcopy(reg)
    w = np.linspace(0.5, 2, 60).round(3)
    d["columns"].append(col("w", w))
    ref = smf.wls("sbp ~ Age + BMI", df, weights=w).fit()
    res = ok("multiple_regression", d, dict(base, factors=[], weights="w"))
    co = table(res, "Coefficients")
    close(row(co, "Term", "Age")["est"], ref.params["Age"], rel=1e-8)
    close(row(co, "Term", "BMI")["se"], ref.bse["BMI"], rel=1e-8)
    # no intercept
    ref = smf.ols("sbp ~ Age + BMI - 1", df).fit()
    res = ok("multiple_regression", reg, dict(base, factors=[], intercept=False))
    co = table(res, "Coefficients")
    assert [r_["Term"] for r_ in co] == ["Age", "BMI"]
    close(co[0]["est"], ref.params["Age"], rel=1e-8)
    close(float(row(table(res, "Model summary"), "Measure", "R²")["value"]), ref.rsquared, rel=1e-3)
    # rows with NaN are dropped and counted
    d = copy.deepcopy(reg); d["columns"][0]["values"][5] = None; d["columns"][2]["values"][8] = None
    res = ok("multiple_regression", d, dict(base))
    assert row(table(res, "Model summary"), "Measure", "Observations")["value"] == "58"
    assert "2 rows with missing values were excluded" in " ".join(
        f for e in res["elements"] if e["type"] == "table" for f in e["footnotes"])


def test_multiple_regression_errors(reg):
    base = {"dep": "Systolic BP"}
    assert "at least one predictor" in err("multiple_regression", reg, base)
    d = copy.deepcopy(reg)
    d["columns"].append(col("Age (months)", [v * 12 for v in reg["columns"][0]["values"]]))
    d["columns"].append(col("K", [3.0] * 60))
    d["columns"].append(col("Y2", [2 * v + 1 for v in reg["columns"][0]["values"]]))
    msg = err("multiple_regression", d, dict(base, covariates=["Age", "Age (months)"]))
    assert "collinearity" in msg and "Age (months)" in msg
    assert "same value in every row" in err("multiple_regression", d, dict(base, covariates=["Age", "K"]))
    assert "exactly" in err("multiple_regression", d, {"dep": "Y2", "covariates": ["Age"]})
    assert "more than one role" in err("multiple_regression", d, dict(base, covariates=["Systolic BP"]))
    c = dict(base, covariates=["Age"], factors=["Sex"], interactions="custom")
    assert "Custom model" in err("multiple_regression", reg, dict(c, formula=""))
    assert "not one of the selected predictors" in err("multiple_regression", reg, dict(c, formula="`Age` + `BMI`"))
    assert "not one of the selected predictors" in err("multiple_regression", reg, dict(c, formula='system("ls")'))
    assert "not understood" in err("multiple_regression", reg, dict(c, formula="`Age`; q()"))
    assert "categorical predictor cannot" in err("multiple_regression", reg, dict(c, formula="log(`Sex`)"))
    tiny = {"columns": [col("y", [1.0, 2.0, 4.0]), col("a", [1.0, 2.0, 3.0]), col("b", [2.0, 1.0, 5.0])]}
    assert "as many coefficients" in err("multiple_regression", tiny, {"dep": "y", "covariates": ["a", "b"]})
    assert "negative" in err("multiple_regression", dict(columns=reg["columns"] + [col("w", [-1.0] * 60)]),
                             dict(base, covariates=["Age"], weights="w"))


# ---------------------------------------------------------------------------
# logistic regression
# ---------------------------------------------------------------------------

def test_logistic_regression_matches_statsmodels(reg):
    import statsmodels.formula.api as smf
    df = frame(reg)
    df["y"] = (df["Hypertension"] == "Yes").astype(int)
    df["Sex"] = pd.Categorical(df["Sex"], categories=["Male", "Female"])
    ref = smf.logit("y ~ Age + BMI + C(Sex)", df).fit(disp=0)
    res = ok("logistic_regression", reg, {"dep": "Hypertension", "covariates": ["Age", "BMI"], "factors": ["Sex"],
                                          "ci_method": "wald"})
    co = table(res, "Coefficients")
    keys = ["Intercept", "Age", "BMI", "C(Sex)[T.Female]"]
    for r_, k in zip(co, keys):
        close(r_["est"], ref.params[k], rel=1e-5); close(r_["se"], ref.bse[k], rel=1e-4)
        close(r_["p"], ref.pvalues[k], rel=1e-3)
    orr = table(res, "Odds ratios")
    ci = np.exp(ref.conf_int())
    assert [r_["Term"] for r_ in orr] == ["Age", "BMI", "Sex [Female]"]
    close(orr[0]["ratio"], math.exp(ref.params["Age"]), rel=1e-5)
    close(orr[0]["lower"], ci.loc["Age", 0], rel=1e-4); close(orr[0]["upper"], ci.loc["Age", 1], rel=1e-4)
    t = table(res, "Model summary")
    assert measure(res, "Model summary", "Observations") == 60
    assert measure(res, "Model summary", "Outcome = Yes") == int(df["y"].sum())
    close(row(t, "Measure", "Likelihood-ratio")["value"], ref.llr, rel=1e-6)
    close(row(t, "Measure", "Likelihood-ratio")["p"], ref.llr_pvalue, rel=1e-5)
    close(measure(res, "Model summary", "McFadden"), ref.prsquared, rel=1e-6)
    close(measure(res, "Model summary", "Log-likelihood"), ref.llf, rel=1e-7)
    close(measure(res, "Model summary", "AIC"), ref.aic, rel=1e-7)
    ph = ref.predict()
    close(measure(res, "Model summary", "Tjur"), ph[df.y == 1].mean() - ph[df.y == 0].mean(), rel=1e-5)
    n = len(df); cs = 1 - math.exp(-ref.llr / n)
    close(measure(res, "Model summary", "Nagelkerke"), cs / (1 - math.exp(2 * ref.llnull / n)), rel=1e-6)
    # AUC = Mann-Whitney U / (n1 n0)
    u = stats.mannwhitneyu(ph[df.y == 1], ph[df.y == 0]).statistic
    close(table(res, "ROC curve")[0]["auc"], u / ((df.y == 1).sum() * (df.y == 0).sum()), rel=1e-8)
    cl = table(res, "Classification table")
    assert cl[0]["p1"] == int(((ph >= 0.5) & (df.y == 1)).sum()) and cl[1]["p0"] == int(((ph < 0.5) & (df.y == 0)).sum())
    # Hosmer-Lemeshow with 10 equal-sized groups
    order = np.argsort(ph, kind="stable")
    chunks = np.array_split(order, 10)
    hl = sum((df.y.to_numpy()[c].sum() - ph[c].sum()) ** 2 / (ph[c].sum() * (1 - ph[c].mean())) for c in chunks)
    h = table(res, "Hosmer")[0]
    close(h["chi2"], hl, rel=1e-5); assert h["df"] == 8
    close(h["p"], stats.chi2.sf(hl, 8), rel=1e-5)
    assert "events per predictor" in warnings_of(res).lower()
    # profile-likelihood intervals contain the estimate and differ from Wald
    res_p = ok("logistic_regression", reg, {"dep": "Hypertension", "covariates": ["Age", "BMI"], "factors": ["Sex"]})
    orp = table(res_p, "Odds ratios")
    assert orp[0]["lower"] < orp[0]["ratio"] < orp[0]["upper"]
    assert abs(orp[0]["upper"] - orr[0]["upper"]) > 1e-4
    assert "profile-likelihood" in " ".join(f for e in res_p["elements"] if e["type"] == "table" for f in e["footnotes"])
    # modelling the other outcome flips the sign
    res_n = ok("logistic_regression", reg, {"dep": "Hypertension", "positive": "No", "covariates": ["Age"],
                                            "ci_method": "wald"})
    res_y = ok("logistic_regression", reg, {"dep": "Hypertension", "covariates": ["Age"], "ci_method": "wald"})
    close(table(res_n, "Coefficients")[1]["est"], -table(res_y, "Coefficients")[1]["est"], rel=1e-6)
    assert n_plots(res_y) == 2          # ROC + predicted-probability curve for a single continuous predictor


def test_logistic_regression_separation_and_errors(reg):
    sep = {"columns": [col("x", [1.0, 2, 3, 4, 5, 6, 7, 8]), col("y", list("nnnnyyyy"), "nominal")]}
    res = ok("logistic_regression", sep, {"dep": "y", "covariates": ["x"]})
    assert "separation" in warnings_of(res).lower()
    assert "separation" in summary(res)
    assert "exactly two values" in err("logistic_regression", reg, {"dep": "Age", "covariates": ["BMI"]})
    assert "not a value" in err("logistic_regression", reg, {"dep": "Sex", "covariates": ["Age"], "positive": "zzz"})
    assert "at least one predictor" in err("logistic_regression", reg, {"dep": "Sex"})
    assert "cannot also be a predictor" in err("logistic_regression", reg, {"dep": "Sex", "factors": ["Sex"]})
    # numeric 0/1 outcome works; last level (1) is modelled
    d = copy.deepcopy(reg)
    d["columns"].append(col("HT", [1.0 if v == "Yes" else 0.0 for v in reg["columns"][4]["values"]]))
    r1 = ok("logistic_regression", d, {"dep": "HT", "covariates": ["Age"], "ci_method": "wald"})
    r2 = ok("logistic_regression", reg, {"dep": "Hypertension", "covariates": ["Age"], "ci_method": "wald"})
    close(table(r1, "Coefficients")[1]["est"], table(r2, "Coefficients")[1]["est"], rel=1e-8)


# ---------------------------------------------------------------------------
# Poisson / negative-binomial regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(5); n = 150
    x = rng.normal(size=n).round(3); g = rng.choice(["A", "B", "C"], n); expo = rng.uniform(1, 5, n).round(2)
    mu = expo * np.exp(0.3 + 0.5 * x + 0.4 * (g == "B"))
    y = rng.negative_binomial(2, 2 / (2 + mu))
    df = pd.DataFrame({"x": x, "g": pd.Categorical(g, categories=["A", "B", "C"]), "T": expo, "y": y})
    data = {"columns": [col("x", x), dict(col("Group", g, "nominal"), levels=["A", "B", "C"]),
                        col("Person-years", expo), col("Events (n)", [float(v) for v in y])]}
    return df, data


def test_poisson_regression_matches_statsmodels(counts):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    df, data = counts
    opts = {"dep": "Events (n)", "covariates": ["x"], "factors": ["Group"], "offset": "Person-years",
            "ci_method": "wald"}
    ref = smf.glm("y ~ x + C(g)", df, family=sm.families.Poisson(), offset=np.log(df["T"])).fit()
    res = ok("poisson_regression", data, opts)
    co = table(res, "Coefficients")
    assert [r_["Term"] for r_ in co] == ["Intercept", "x", "Group [B]", "Group [C]"]
    for r_, k in zip(co, ["Intercept", "x", "C(g)[T.B]", "C(g)[T.C]"]):
        close(r_["est"], ref.params[k], rel=1e-6); close(r_["se"], ref.bse[k], rel=1e-5)
    irr = table(res, "Incidence rate ratios")
    ci = np.exp(ref.conf_int())
    close(irr[0]["ratio"], math.exp(ref.params["x"]), rel=1e-6)
    close(irr[0]["lower"], ci.loc["x", 0], rel=1e-5); close(irr[0]["upper"], ci.loc["x", 1], rel=1e-5)
    close(measure(res, "Model summary", "Residual deviance"), ref.deviance, rel=1e-7)
    close(measure(res, "Model summary", "Residual deviance", "p"), stats.chi2.sf(ref.deviance, ref.df_resid), rel=1e-5,
          abs_=1e-300)
    close(measure(res, "Model summary", "Pearson"), ref.pearson_chi2, rel=1e-6)
    disp = ref.pearson_chi2 / ref.df_resid
    close(measure(res, "Model summary", "Dispersion"), disp, rel=1e-6)
    close(measure(res, "Model summary", "AIC"), ref.aic, rel=1e-7)
    assert disp > 1.5
    text = " ".join(e["content"] for e in res["elements"] if e["type"] == "text")
    assert "overdispersed" in text and "negative binomial" in text
    # quasi-Poisson: same coefficients, SE scaled by sqrt(dispersion)
    rq = ok("poisson_regression", data, dict(opts, family="quasipoisson"))
    cq = table(rq, "Coefficients")
    close(cq[1]["est"], co[1]["est"], rel=1e-8); close(cq[1]["se"], co[1]["se"] * math.sqrt(disp), rel=1e-6)
    # negative binomial against statsmodels' ML fit
    nb = smf.negativebinomial("y ~ x + C(g)", df, offset=np.log(df["T"])).fit(disp=0, maxiter=200)
    rn = ok("poisson_regression", data, dict(opts, family="negbin"))
    cn = table(rn, "Coefficients")
    for r_, k in zip(cn, ["Intercept", "x", "C(g)[T.B]", "C(g)[T.C]"]):
        close(r_["est"], nb.params[k], rel=2e-3, abs_=2e-3)
        # MASS::glm.nb reports SEs conditional on theta; statsmodels uses the joint Hessian
        close(r_["se"], nb.bse[k], rel=5e-2)
    close(measure(rn, "Model summary", "Theta"), 1 / nb.params["alpha"], rel=1e-2)
    close(measure(rn, "Model summary", "Log-likelihood"), nb.llf, rel=1e-5)
    # without offset, profile intervals (default)
    r0 = ok("poisson_regression", data, {"dep": "Events (n)", "covariates": ["x"]})
    ref0 = smf.glm("y ~ x", df, family=sm.families.Poisson()).fit()
    close(table(r0, "Coefficients")[1]["est"], ref0.params["x"], rel=1e-6)
    i0 = table(r0, "Incidence rate ratios")[0]
    assert i0["lower"] < i0["ratio"] < i0["upper"]


def test_poisson_regression_errors(counts, reg):
    df, data = counts
    assert "non-integer" in err("poisson_regression", reg, {"dep": "BMI", "covariates": ["Age"]})
    ok("poisson_regression", reg, {"dep": "BMI", "covariates": ["Age"], "family": "quasipoisson"})
    d = copy.deepcopy(data); d["columns"][3]["values"][0] = -1.0
    assert "negative" in err("poisson_regression", d, {"dep": "Events (n)", "covariates": ["x"]})
    d = copy.deepcopy(data); d["columns"][2]["values"][0] = 0.0
    assert "greater than zero" in err("poisson_regression", d, {"dep": "Events (n)", "covariates": ["x"],
                                                               "offset": "Person-years"})
    # equidispersed counts: the negative-binomial model falls back gracefully
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    pois = {"columns": [col("x", x), col("y", [float(v) for v in rng.poisson(np.exp(0.2 + 0.3 * x))])]}
    res = ok("poisson_regression", pois, {"dep": "y", "covariates": ["x"], "family": "negbin"})
    assert has_table(res, "Incidence rate ratios")


# ---------------------------------------------------------------------------
# Cox regression
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surv():
    d = sample("survival.csv")
    rng = np.random.default_rng(2)
    n = len(d["columns"][0]["values"])
    d["columns"].append(col("Age (y)", rng.normal(60, 8, n).round(1)))
    d["columns"].append(col("Site", rng.choice(["North", "South"], n), "nominal"))
    return d


def test_cox_regression_matches_lifelines(surv):
    from lifelines import CoxPHFitter
    df = frame(surv)
    ref_df = pd.DataFrame({"T": df["Months"], "E": df["Status"].astype(int), "age": df["Age (y)"],
                           "trt": (df["Arm"] == "Treatment").astype(int)})
    cph = CoxPHFitter().fit(ref_df, "T", "E")
    res = ok("cox_regression", surv, {"time": "Months", "event": "Status", "covariates": ["Age (y)"],
                                      "factors": ["Arm"]})
    hr = table(res, "Hazard ratios")
    assert [r_["Term"] for r_ in hr] == ["Age (y)", "Arm [Treatment]"]
    s = cph.summary
    for r_, k in zip(hr, ["age", "trt"]):
        close(r_["coef"], s.loc[k, "coef"], rel=1e-4, abs_=1e-6); close(r_["se"], s.loc[k, "se(coef)"], rel=1e-4)
        close(r_["hr"], s.loc[k, "exp(coef)"], rel=1e-4); close(r_["p"], s.loc[k, "p"], rel=1e-3)
        close(r_["lower"], s.loc[k, "exp(coef) lower 95%"], rel=1e-4)
        close(r_["upper"], s.loc[k, "exp(coef) upper 95%"], rel=1e-4)
    assert measure(res, "Model summary", "Observations") == 80
    assert measure(res, "Model summary", "Events") == int(ref_df.E.sum())
    close(measure(res, "Model summary", "Concordance"), cph.concordance_index_, rel=1e-6)
    lr = cph.log_likelihood_ratio_test()
    close(measure(res, "Model summary", "Likelihood-ratio"), lr.test_statistic, rel=1e-5)
    close(measure(res, "Model summary", "Likelihood-ratio", "p"), lr.p_value, rel=1e-4)
    ph = table(res, "proportional-hazards")
    assert [r_["Term"] for r_ in ph] == ["Age (y)", "Arm", "Global"]
    assert all(0 <= r_["p"] <= 1 for r_ in ph)
    assert n_plots(res) == 2
    g = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert g["type"] == "survival" and g["options"]["group"] == "Arm"
    # one binary predictor: the score test is the log-rank test
    from lifelines.statistics import logrank_test
    a, b = ref_df[ref_df.trt == 0], ref_df[ref_df.trt == 1]
    lrk = logrank_test(a["T"], b["T"], a["E"], b["E"])
    res1 = ok("cox_regression", surv, {"time": "Months", "event": "Status", "factors": ["Arm"], "ties": "breslow"})
    # Breslow ties against statsmodels
    from statsmodels.duration.hazard_regression import PHReg
    ph_ref = PHReg(ref_df["T"], ref_df[["trt"]], status=ref_df["E"], ties="breslow").fit()
    r_ = table(res1, "Hazard ratios")[0]
    close(r_["coef"], ph_ref.params[0], rel=1e-5); close(r_["se"], ph_ref.bse[0], rel=1e-4)
    res_e = ok("cox_regression", surv, {"time": "Months", "event": "Status", "factors": ["Arm"]})
    close(measure(res_e, "Model summary", "Score"), lrk.test_statistic, rel=2e-2)
    # stratified model against lifelines
    ref_df["site"] = df["Site"]
    cs = CoxPHFitter().fit(ref_df, "T", "E", strata=["site"])
    res_s = ok("cox_regression", surv, {"time": "Months", "event": "Status", "covariates": ["Age (y)"],
                                        "factors": ["Arm"], "strata": "Site"})
    hs = table(res_s, "Hazard ratios")
    assert len(hs) == 2
    close(hs[1]["coef"], cs.summary.loc["trt", "coef"], rel=1e-4, abs_=1e-6)
    # categorical event column with a named event value
    d = copy.deepcopy(surv)
    d["columns"][1] = col("Status", ["dead" if v == 1 else "alive" for v in surv["columns"][1]["values"]], "nominal")
    res_c = ok("cox_regression", d, {"time": "Months", "event": "Status", "event_value": "dead", "factors": ["Arm"]})
    close(table(res_c, "Hazard ratios")[0]["coef"], table(res_e, "Hazard ratios")[0]["coef"], rel=1e-10)


def test_cox_regression_errors(surv):
    base = {"time": "Months", "event": "Status"}
    assert "at least one predictor" in err("cox_regression", surv, base)
    assert "No events were found" in err("cox_regression", surv, dict(base, factors=["Arm"], event_value="7"))
    d = copy.deepcopy(surv); d["columns"][0]["values"][0] = -2.0
    assert "negative times" in err("cox_regression", d, dict(base, factors=["Arm"]))
    # a group without events -> infinite hazard ratio explained in plain language
    d = copy.deepcopy(surv)
    d["columns"][1]["values"] = [0 if a == "Treatment" else s
                                 for a, s in zip(surv["columns"][2]["values"], surv["columns"][1]["values"])]
    res = ok("cox_regression", d, dict(base, factors=["Arm"]))
    assert "zero or infinite" in warnings_of(res)
    d = copy.deepcopy(surv); d["columns"].append(col("Age2", [2 * v for v in surv["columns"][3]["values"]]))
    assert "collinearity" in err("cox_regression", d, dict(base, covariates=["Age (y)", "Age2"]))


# ---------------------------------------------------------------------------
# reliability and factor analysis
# ---------------------------------------------------------------------------

ITEMS = ["IL-6", "TNF-α", "CRP", "IL-10", "IFN-γ", "VEGF"]


def _alpha(x: np.ndarray) -> float:
    k = x.shape[1]
    return k / (k - 1) * (1 - x.var(axis=0, ddof=1).sum() / x.sum(axis=1).var(ddof=1))


def test_reliability_matches_formulas():
    bm = sample("biomarker_panel.csv")
    x = frame(bm)[ITEMS].to_numpy(float); n, k = x.shape
    res = ok("reliability", bm, {"variables": ITEMS, "show_plot": True})
    a = _alpha(x)
    r_ = row(table(res, "Scale reliability"), "Measure", "Cronbach")
    close(r_["value"], a, rel=1e-10)
    close(r_["lower"], 1 - (1 - a) * stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1)), rel=1e-8)
    close(r_["upper"], 1 - (1 - a) * stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1)), rel=1e-8)
    rr = np.corrcoef(x, rowvar=False); rbar = rr[np.triu_indices(k, 1)].mean()
    close(measure(res, "Scale reliability", "Standardized alpha"), k * rbar / (1 + (k - 1) * rbar), rel=1e-10)
    assert measure(res, "Scale reliability", "Number of items") == 6
    items = table(res, "Item statistics")
    assert [r_["Item"] for r_ in items] == ITEMS
    for j, r_ in enumerate(items):
        rest = np.delete(x, j, axis=1)
        close(r_["rest"], np.corrcoef(x[:, j], rest.sum(axis=1))[0, 1], rel=1e-9)
        close(r_["dropped"], _alpha(rest), rel=1e-9, abs_=1e-12)
        close(r_["mean"], x[:, j].mean())
    assert "negatively related" in warnings_of(res) and "IL-10" in warnings_of(res)
    assert n_plots(res) == 1
    # automatic reverse scoring of IL-10 and VEGF
    res = ok("reliability", bm, {"variables": ITEMS, "reverse": True})
    xr = x.copy()
    for j in (3, 5):
        xr[:, j] = xr[:, j].min() + xr[:, j].max() - xr[:, j]
    close(measure(res, "Scale reliability", "Cronbach"), _alpha(xr), rel=1e-10)
    assert "negatively related" not in warnings_of(res)
    om = measure(res, "Scale reliability", "McDonald")
    assert 0 < om < 1
    # two items: alpha = Spearman-Brown of their correlation when variances are equal (standardized alpha)
    res = ok("reliability", bm, {"variables": ["IL-6", "TNF-α"]})
    r12 = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
    close(measure(res, "Scale reliability", "Standardized alpha"), 2 * r12 / (1 + r12), rel=1e-10)
    # errors
    assert "at least 2" in err("reliability", bm, {"variables": ["IL-6"]})
    d = copy.deepcopy(bm); d["columns"].append(col("K", [1.0] * 90))
    assert "same value in every row" in err("reliability", d, {"variables": ["IL-6", "K"]})


def test_factor_analysis():
    bm = sample("biomarker_panel.csv")
    x = frame(bm)[ITEMS].to_numpy(float); n, p = x.shape
    r = np.corrcoef(x, rowvar=False)
    res = ok("factor_analysis", bm, {"variables": ITEMS, "hide_below": 0, "sort_loadings": False})
    # Bartlett's test and KMO from their definitions
    chi2 = -(n - 1 - (2 * p + 5) / 6) * math.log(np.linalg.det(r))
    ad = table(res, "Sampling adequacy")
    close(ad[1]["statistic"], chi2, rel=1e-8); assert ad[1]["df"] == 15
    close(ad[1]["p"], stats.chi2.sf(chi2, 15), rel=1e-6, abs_=1e-300)
    inv = np.linalg.inv(r); d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    off = ~np.eye(p, dtype=bool)
    kmo = (r[off] ** 2).sum() / ((r[off] ** 2).sum() + (partial[off] ** 2).sum())
    close(ad[0]["statistic"], kmo, rel=1e-8)
    msa = table(res, "KMO for each variable")
    r0, p0 = r * off, partial * off
    np.testing.assert_allclose([m["msa"] for m in msa], (r0 ** 2).sum(0) / ((r0 ** 2).sum(0) + (p0 ** 2).sum(0)), rtol=1e-8)
    ev = table(res, "Eigenvalues")
    np.testing.assert_allclose([e["eigen"] for e in ev], np.sort(np.linalg.eigvalsh(r))[::-1], rtol=1e-8)
    # two factors by parallel analysis; loadings reproduce the correlation matrix
    load = table(res, "Factor loadings")
    assert [l_["Variable"] for l_ in load] == ITEMS
    assert set(load[0]) >= {"F1", "F2", "communality", "uniqueness"} and "F3" not in load[0]
    L = np.array([[l_["F1"], l_["F2"]] for l_ in load]); u = np.array([l_["uniqueness"] for l_ in load])
    np.testing.assert_allclose((L ** 2).sum(1), [l_["communality"] for l_ in load], atol=1e-4)
    np.testing.assert_allclose((L ** 2).sum(1) + u, 1, atol=1e-4)
    resid = (L @ L.T + np.diag(u) - r)[off]
    assert np.sqrt((resid ** 2).mean()) < 0.03
    var = table(res, "Variance explained")
    close(var[0]["ss"], (L[:, 0] ** 2).sum(), rel=1e-6); close(var[1]["cum"], (L ** 2).sum() / p, rel=1e-6)
    close(var[0]["prop"], var[0]["ss"] / p, rel=1e-9)
    assert var[0]["ss"] >= var[1]["ss"]
    # rotation leaves communalities unchanged; unrotated ML solution has the same uniquenesses
    res_n = ok("factor_analysis", bm, {"variables": ITEMS, "rotation": "none", "hide_below": 0,
                                       "sort_loadings": False, "n_method": "fixed", "n_factors": 2})
    np.testing.assert_allclose([l_["uniqueness"] for l_ in table(res_n, "Factor loadings")], u, atol=1e-6)
    # oblique rotations (oblimin falls back to promax when GPArotation is missing), minres, hidden loadings
    res_o = ok("factor_analysis", bm, {"variables": ITEMS, "rotation": "oblimin", "extraction": "minres",
                                       "n_method": "fixed", "n_factors": 2})
    assert has_table(res_o, "pattern matrix") and has_table(res_o, "Correlations between factors")
    lo = table(res_o, "Factor loadings")
    assert any(l_.get("F1") is None or l_.get("F2") is None for l_ in lo)       # small loadings hidden
    np.testing.assert_allclose(sorted(l_["uniqueness"] for l_ in lo), sorted(u), atol=0.05)
    res_k = ok("factor_analysis", bm, {"variables": ITEMS, "n_method": "kaiser"})
    assert "2 factors" in summary(res_k)
    # errors
    assert "at least 3" in err("factor_analysis", bm, {"variables": ITEMS[:2]})
    assert "at most" in err("factor_analysis", bm, {"variables": ITEMS, "n_method": "fixed", "n_factors": 5})
    dd = copy.deepcopy(bm); dd["columns"].append(col("IL-6 (pg)", [v * 1000 for v in bm["columns"][1]["values"]]))
    assert "perfectly correlated" in err("factor_analysis", dd, {"variables": ITEMS + ["IL-6 (pg)"]})
    dd = copy.deepcopy(bm)
    dd["columns"].append(col("Total", list(frame(bm)[ITEMS].sum(axis=1))))
    assert "singular" in err("factor_analysis", dd, {"variables": ITEMS + ["Total"]})


# ---------------------------------------------------------------------------
# cross-cutting behaviour
# ---------------------------------------------------------------------------

def test_missing_required_variables_give_friendly_errors():
    data = sample("two_way.csv")
    for aid in ALL_IDS:
        msg = err(aid, data, {})
        assert "choose" in msg.lower(), (aid, msg)


def test_paired_bp_two_conditions():
    bp = sample("paired_bp.csv")
    x = frame(bp)
    tt = stats.ttest_rel(x["Before"], x["After"])
    res = ok("anova_rm_oneway", bp, {"columns": ["Before", "After"]})
    t = table(res, "Repeated-measures one-way ANOVA")[0]
    close(t["F"], tt.statistic ** 2); close(t["p"], tt.pvalue)
    res = ok("friedman", bp, {"columns": ["Before", "After"]})
    d = (x["Before"] - x["After"]).to_numpy()
    pos, neg = (d > 0).sum(), (d < 0).sum()          # two conditions: Friedman = sign test (ties get mid-ranks)
    ranks = np.where(d[:, None] == 0, 1.5, np.where(d[:, None] > 0, [2, 1], [1, 2]))
    rs = ranks.sum(axis=0); n = len(d)
    ties = (d == 0).sum()
    chi2 = 12 * ((rs - n * 1.5) ** 2).sum() / (n * 2 * 3) / (1 - ties * 6 / (n * 2 * 3))
    close(table(res, "Friedman test")[0]["chi2"], chi2, rel=1e-8)
    assert table(res, "Friedman test")[0]["n"] == 15


def test_alpha_option_changes_confidence_level(groups):
    cols = [c["name"] for c in groups["columns"]]
    res = ok("anova_oneway", groups, {"columns": cols, "alpha": 0.01})
    labels = [c["label"] for e in res["elements"] if e["type"] == "table" for c in e["columns"]]
    assert "99% CI lower" in labels
    a = arrays(groups)
    ci = stats.tukey_hsd(*a.values()).confidence_interval(0.99)
    close(table(res, "Multiple comparisons")[0]["lower"], ci.low[0, 1], rel=1e-5)


def test_speed(groups):
    import time
    cols = [c["name"] for c in groups["columns"]]
    ok("anova_oneway", groups, {"columns": cols})            # warm the worker
    t0 = time.time()
    ok("anova_oneway", groups, {"columns": cols})
    assert time.time() - t0 < 2.0
