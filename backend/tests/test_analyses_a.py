"""Tests for descriptive.py, two_groups.py, correlation.py and categorical.py.

Reference numbers marked "R:" were produced with R 4.x on the bundled sample
data (the R call is given next to each); the others are recomputed here with
scipy / statsmodels / numpy.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis_ui.core import registry, run_analysis

registry.discover()

SAMPLES = Path(registry.__file__).resolve().parents[1] / "sample_data"

MY_ANALYSES = [
    "descriptives", "normality", "outliers", "frequency", "one_sample",
    "ttest_unpaired", "mann_whitney", "ttest_paired", "wilcoxon_paired", "ks_two_sample", "multiple_ttests",
    "correlation", "linear_regression", "deming",
    "contingency", "binomial", "mcnemar", "proportions_ci",
]


# ------------------------------------------------------------------ helpers

def wire(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            cols.append({"name": str(c), "type": "continuous",
                         "values": [None if pd.isna(v) else float(v) for v in s]})
        else:
            cols.append({"name": str(c), "type": "nominal",
                         "values": [None if pd.isna(v) else str(v) for v in s]})
    return {"columns": cols}


def col(name, values, kind="continuous"):
    return {"name": name, "type": kind, "values": list(values)}


def data(*cols):
    return {"columns": list(cols)}


def to_long(named: dict) -> dict:
    """{"A": [...], "B": [...]} → tidy wire data with columns value / grp."""
    vals, grp = [], []
    for k, v in named.items():
        vals += list(v)
        grp += [k] * len(v)
    return data(col("value", vals), col("grp", grp, "nominal"))


def wide(named: dict) -> dict:
    n = max(len(v) for v in named.values())
    return data(*[col(k, list(v) + [None] * (n - len(v))) for k, v in named.items()])


def run_ok(aid, d, options):
    res = run_analysis(aid, d, options)
    assert res["status"] == "ok", res.get("message")
    return res["results"]


def run_err(aid, d, options):
    res = run_analysis(aid, d, options)
    assert res["status"] == "error", "expected an error"
    msg = res["message"]
    # a friendly message, not an exception dump
    assert not re.match(r"^[A-Za-z]+(Error|Exception|Warning):", msg), msg
    assert "Traceback" not in msg
    return msg


def tables(results):
    return [e for e in results["elements"] if e["type"] == "table"]


def table(results, title):
    for t in tables(results):
        if title.lower() in t["title"].lower():
            return t
    raise AssertionError(f"no table titled like {title!r}; have {[t['title'] for t in tables(results)]}")


def rows(results, title):
    return table(results, title)["rows"]


def row(results, title, **match):
    for r in rows(results, title):
        if all(r.get(k) == v for k, v in match.items()):
            return r
    raise AssertionError(f"no row {match} in {title!r}")


def summary(results):
    s = [e["content"] for e in results["elements"] if e["type"] == "text" and e["kind"] == "summary"]
    assert len(s) == 1 and s[0].strip()
    return s[0]


def kinds(results):
    return [e["type"] for e in results["elements"]]


def close(a, b, rel=1e-6, abs_=1e-9):
    assert a is not None, "value is missing"
    assert a == pytest.approx(b, rel=rel, abs=abs_), (a, b)


def stat_row(results, label, title="Descriptive statistics"):
    """Row of a 'one column per group' table → list of values."""
    r = row(results, title, stat=label)
    return [r[k] for k in sorted((k for k in r if k.startswith("g")), key=lambda k: int(k[1:]))]


# ------------------------------------------------------------------ fixtures

TG = pd.read_csv(SAMPLES / "treatment_groups.csv")
BP = pd.read_csv(SAMPLES / "paired_bp.csv")
REG = pd.read_csv(SAMPLES / "regression.csv")
CT = pd.read_csv(SAMPLES / "contingency.csv")
BIO = pd.read_csv(SAMPLES / "biomarker_panel.csv")
MC = pd.read_csv(SAMPLES / "method_comparison.csv")

A = TG["Control"].to_numpy()
B = TG["Drug B"].to_numpy()
C = TG["Drug A"].to_numpy()
AB_WIDE = (wire(TG), {"layout": "wide", "columns": ["Control", "Drug B"]})
AB_LONG = (to_long({"Control": A, "Drug B": B}), {"layout": "long", "dep": "value", "group": "grp"})
BOTH = pytest.mark.parametrize("d,o", [AB_WIDE, AB_LONG], ids=["wide", "long"])

BEFORE, AFTER = BP["Before"].to_numpy(), BP["After"].to_numpy()
PAIR_WIDE = (wire(BP), {"layout": "wide", "columns": ["Before", "After"]})
_pl = pd.DataFrame({"bp": np.concatenate([BEFORE, AFTER]),
                    "time": ["Before"] * len(BP) + ["After"] * len(BP),
                    "id": list(BP["Patient"]) * 2})
# shuffle the rows so that pairing must come from the subject column
PAIR_LONG = (wire(_pl.sample(frac=1, random_state=3).sort_values("time", ascending=False, kind="stable")),
             {"layout": "long", "dep": "bp", "group": "time", "subject": "id"})
PAIRED = pytest.mark.parametrize("d,o", [PAIR_WIDE, PAIR_LONG], ids=["wide", "long"])


# ================================================================ general

def test_all_registered():
    for aid in MY_ANALYSES:
        assert aid in registry.ANALYSES, aid
        d = registry.ANALYSES[aid]
        assert d.engine == "python" and d.description and d.options


@pytest.mark.parametrize("aid", MY_ANALYSES)
def test_no_variables_chosen_is_a_friendly_error(aid):
    for d in (None, wire(TG), wire(CT)):
        msg = run_err(aid, d, {})
        assert len(msg) > 10


@pytest.mark.parametrize("aid", MY_ANALYSES)
def test_empty_columns_do_not_crash(aid):
    """Columns that exist but hold no data at all."""
    d = data(col("x", [None, None, None]), col("y", [None, None, None]), col("g", [None, None, None], "nominal"),
             col("h", [None, None, None], "nominal"))
    opts = {"columns": ["x", "y"], "variables": ["x", "y"], "x": "x", "y": "y", "group": "g", "rows": "g",
            "cols": "h", "variable": "g", "var1": "g", "var2": "h", "data_form": "raw"}
    run_err(aid, d, opts)


@pytest.mark.parametrize("aid", ["descriptives", "normality", "outliers", "frequency", "one_sample",
                                 "ttest_unpaired", "mann_whitney", "ks_two_sample", "ttest_paired",
                                 "wilcoxon_paired"])
def test_unicode_labels_and_missing_values(aid):
    rng = np.random.default_rng(5)
    a = list(rng.normal(10, 2, 12)) + [None, None]
    b = [None] + list(rng.normal(12, 2, 13))
    d = data(col("Kontrolle (µg/mL) – α", a), col("Wirkstoff № 2 β+", b))
    res = run_ok(aid, d, {"columns": ["Kontrolle (µg/mL) – α", "Wirkstoff № 2 β+"]})
    assert tables(res)
    summary(res)


# ============================================================ descriptives

def test_descriptives_wide_matches_numpy_and_r():
    res = run_ok("descriptives", wire(TG), {"columns": list(TG.columns), "show_geo": True, "show_shape": True,
                                            "show_cv": True, "show_sum": True})
    t = table(res, "Descriptive statistics")
    assert [c["label"] for c in t["columns"]] == ["Statistic", *TG.columns]
    close(stat_row(res, "Mean")[0], A.mean())
    close(stat_row(res, "Std. deviation")[0], A.std(ddof=1))
    close(stat_row(res, "Std. error of mean")[0], stats.sem(A))
    lo, hi = stats.t.interval(0.95, len(A) - 1, A.mean(), stats.sem(A))
    close(stat_row(res, "Lower 95% CI of mean")[0], lo)
    close(stat_row(res, "Upper 95% CI of mean")[0], hi)
    close(stat_row(res, "Lower 95% CI of mean")[0], 88.5933009243)        # R: t.test(a)$conf.int
    close(stat_row(res, "Median")[0], np.median(A))
    close(stat_row(res, "25% percentile")[0], 92.275)                     # R: quantile(a, .25, type = 6)
    close(stat_row(res, "75% percentile")[0], 109.05)
    close(stat_row(res, "Interquartile range")[0], 109.05 - 92.275)
    close(stat_row(res, "Minimum")[0], A.min())
    close(stat_row(res, "Range")[0], np.ptp(A))
    close(stat_row(res, "Skewness")[0], -0.00015027774434, rel=1e-5)      # R: e1071::skewness(a, type = 2)
    close(stat_row(res, "Excess kurtosis")[0], 1.3122745379)              # R: e1071::kurtosis(a, type = 2)
    close(stat_row(res, "Geometric mean")[0], stats.gmean(A))
    close(stat_row(res, "Geometric SD factor")[0], stats.gstd(A))
    close(stat_row(res, "Coefficient of variation (%)")[0], 100 * A.std(ddof=1) / A.mean())
    close(stat_row(res, "Sum")[0], A.sum())
    assert stat_row(res, "Number of values") == [12, 12, 12, 12]
    assert "plot" in kinds(res) and "graph" in kinds(res)
    assert "Control" in summary(res)


def test_descriptives_long_equals_wide_and_conf_level():
    d, o = AB_LONG
    res = run_ok("descriptives", d, {**o, "conf_level": 99, "show_plot": False})
    lo, hi = stats.t.interval(0.99, len(B) - 1, B.mean(), stats.sem(B))
    close(stat_row(res, "Lower 99% CI of mean")[1], lo)
    close(stat_row(res, "Upper 99% CI of mean")[1], hi)
    assert "plot" not in kinds(res)


def test_descriptives_unequal_n_single_value_and_nonpositive():
    d = wide({"a": [1.0, 2.0, 3.0, 4.0], "one": [5.0], "neg": [-1.0, 0.0, 2.0]})
    res = run_ok("descriptives", d, {"columns": ["a", "one", "neg"], "show_geo": True, "show_shape": True})
    assert stat_row(res, "Number of values") == [4, 1, 3]
    assert stat_row(res, "Std. deviation")[1] is None            # NaN → empty cell
    assert stat_row(res, "Geometric mean")[2] is None
    notes = " ".join(table(res, "Descriptive")["footnotes"])
    assert "one" in notes and "neg" in notes


def test_descriptives_needs_a_statistic():
    off = {k: False for k in ("show_n", "show_mean", "show_sd", "show_ci", "show_median", "show_quartiles",
                              "show_minmax")}
    assert "statistic" in run_err("descriptives", wire(TG), {"columns": ["Control"], **off})


def test_descriptives_text_in_numeric_column():
    d = data(col("x", ["1.5", "abc", "2.5", "", "3,000"]))
    res = run_ok("descriptives", d, {"columns": ["x"]})
    assert stat_row(res, "Number of values") == [3]
    close(stat_row(res, "Mean")[0], (1.5 + 2.5 + 3000) / 3)


# =============================================================== normality

def test_normality_reference_values():
    from statsmodels.stats.diagnostic import lilliefors, normal_ad
    res = run_ok("normality", wire(TG), {"columns": ["Control", "Drug A"]})
    r = row(res, "Normality tests", group="Control", test="Shapiro–Wilk")
    close(r["stat"], 0.948972768083)                              # R: shapiro.test(a)
    close(r["p"], 0.622018410071, rel=1e-5)
    assert r["passed"] == "Yes"
    r = row(res, "Normality tests", group="Control", test="Anderson–Darling")
    close(r["p"], 0.323005845653, rel=1e-6)                       # R: nortest::ad.test(a)$p.value
    close(r["p"], normal_ad(A)[1], rel=1e-6)
    r = row(res, "Normality tests", group="Control", test="D'Agostino–Pearson omnibus")
    k2, p = stats.normaltest(A)
    close(r["stat"], k2)
    close(r["p"], p)
    r = row(res, "Normality tests", group="Control", test="Kolmogorov–Smirnov (Lilliefors)")
    close(r["stat"], 0.146922452053)                              # R: nortest::lillie.test(a)$statistic
    close(r["p"], lilliefors(A)[1])
    assert "plot" in kinds(res)
    assert "normality" in summary(res)


def test_normality_detects_skewed_data_and_lognormal_passes():
    x = np.exp(np.random.default_rng(1).normal(0, 1, 60))
    for d, o in ((wide({"x": x}), {"columns": ["x"]}),
                 (to_long({"x": x}), {"layout": "long", "dep": "value", "group": "grp"})):
        res = run_ok("normality", d, {**o, "lognormal": True})
        r = row(res, "Normality tests", test="Shapiro–Wilk")
        close(r["p"], stats.shapiro(x).pvalue)
        assert r["passed"] == "No"
        r = row(res, "Lognormality tests", test="Shapiro–Wilk")
        close(r["p"], stats.shapiro(np.log(x)).pvalue)
        assert r["passed"] == "Yes"
        s = summary(res)
        assert "rejected normality" in s and "lognormal" in s


def test_normality_small_constant_and_negative_groups():
    d = wide({"small": [1.0, 2.0, 4.0, 3.5], "const": [2.0] * 9, "neg": [-3.0, 1.0, 2.0, 5.0, 4.0, 2.5, 3.1, 0.5]})
    res = run_ok("normality", d, {"columns": ["small", "const", "neg"], "lognormal": True})
    r = row(res, "Normality tests", group="small", test="D'Agostino–Pearson omnibus")
    assert r["p"] is None and "n too small" in r["note"]
    assert row(res, "Normality tests", group="small", test="Shapiro–Wilk")["p"] is not None
    r = row(res, "Normality tests", group="const", test="Shapiro–Wilk")
    assert r["p"] is None and "identical" in r["note"]
    warn = [e["content"] for e in res["elements"] if e["type"] == "text" and e["kind"] == "warning"]
    assert warn and "neg" in warn[0]
    assert "at least 3" in run_err("normality", wide({"a": [1.0, 2.0]}), {"columns": ["a"]})
    assert "at least one" in run_err("normality", wire(TG), {"columns": ["Control"], "shapiro": False,
                                                              "dagostino": False, "ad": False, "ks": False})


# ================================================================ outliers

OUT = [9.8, 10.1, 10.4, 9.9, None, 10.0, 10.2, 9.7, 25.0, 10.3, 9.6, 10.05]


@pytest.mark.parametrize("method", ["grubbs", "grubbs_iter", "gesd", "tukey", "mad"])
def test_outliers_planted_value_found_in_both_layouts(method):
    clean = [v for v in OUT if v is not None and v != 25.0]
    d = data(col("ok", [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 10.4, 9.7, 10.0, 9.6]), col("bad", OUT))
    res = run_ok("outliers", d, {"columns": ["ok", "bad"], "method": method})
    r = row(res, "Outlier summary", group="bad")
    assert r["n"] == 11 and r["n_out"] == 1 and r["values"] == "25"
    close(r["mean_clean"], np.mean(clean))
    assert row(res, "Outlier summary", group="ok")["n_out"] == 0
    found = rows(res, "Outliers found")
    assert len(found) == 1 and found[0]["row"] == 9 and found[0]["value"] == 25.0
    assert "1 outlier" in summary(res)
    # long layout: the row number is the row of the tidy table
    vals = [v for v in OUT]
    dl = data(col("v", vals), col("g", ["bad"] * len(vals), "nominal"))
    res = run_ok("outliers", dl, {"layout": "long", "dep": "v", "group": "g", "method": method})
    assert rows(res, "Outliers found")[0]["row"] == 9


def test_grubbs_statistic_and_p_value():
    x = np.array([v for v in OUT if v is not None])
    res = run_ok("outliers", wide({"x": x}), {"columns": ["x"], "method": "grubbs"})
    f = rows(res, "Outliers found")[0]
    n = len(x)
    g = (25.0 - x.mean()) / x.std(ddof=1)
    close(f["stat"], g)
    t = math.sqrt(n * (n - 2) * g * g / ((n - 1) ** 2 - n * g * g))
    close(f["p"], min(1, 2 * n * stats.t.sf(t, n - 2)))
    tc = stats.t.isf(0.05 / (2 * n), n - 2)
    close(f["crit"], (n - 1) / math.sqrt(n) * math.sqrt(tc ** 2 / (n - 2 + tc ** 2)))
    assert f["stat"] > f["crit"]


def test_gesd_rosner_example():
    # Rosner (1983) data as used in the NIST/SEMATECH handbook, section 1.3.5.17.3: three outliers.
    x = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69,
         1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26, 2.35,
         2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64,
         5.34, 5.42, 6.01]
    res = run_ok("outliers", wide({"x": x}), {"columns": ["x"], "method": "gesd", "max_outliers": 10})
    found = rows(res, "Outliers found")
    assert sorted(f["value"] for f in found) == [5.34, 5.42, 6.01]
    first = max(found, key=lambda f: f["value"])
    close(first["stat"], 3.118, abs_=1e-3)                        # NIST: R1 = 3.118, lambda1 = 3.158
    close(first["crit"], 3.158, abs_=1e-3)
    # masking: the single Grubbs test finds nothing here
    res = run_ok("outliers", wide({"x": x}), {"columns": ["x"], "method": "grubbs"})
    assert row(res, "Outlier summary", group="x")["n_out"] == 0
    assert "No outliers" in summary(res)


def test_outliers_edge_cases():
    res = run_ok("outliers", wide({"const": [3.0] * 6, "tiny": [1.0, 50.0]}), {"columns": ["const", "tiny"]})
    assert row(res, "Outlier summary", group="const")["n_out"] == 0
    assert "fewer than 3" in row(res, "Outlier summary", group="tiny")["note"]
    assert "at least 3" in run_err("outliers", wide({"a": [1.0, 2.0]}), {"columns": ["a"]})


# =============================================================== frequency

def test_frequency_counts_and_bins():
    res = run_ok("frequency", wire(TG), {"columns": list(TG.columns)})
    t = table(res, "Frequency distribution")
    keys = [c["key"] for c in t["columns"] if c["key"].startswith("g")]
    for k, name in zip(keys, TG.columns):
        assert sum(r[k] for r in t["rows"]) == TG[name].notna().sum()
    centers = [r["center"] for r in t["rows"]]
    width = centers[1] - centers[0]
    edges = [centers[0] - width / 2 + i * width for i in range(len(centers) + 1)]
    assert edges[0] <= TG.min().min() and edges[-1] >= TG.max().max()
    expected, _ = np.histogram(A, bins=edges)
    assert [r["g0"] for r in t["rows"]] == list(expected)


def test_frequency_manual_width_percent_cumulative_long():
    d, o = AB_LONG
    res = run_ok("frequency", d, {**o, "bin_width": 10, "bin_start": 40, "tabulate": "percent", "cumulative": True})
    t = table(res, "Cumulative frequency")
    assert t["rows"][0]["bin"].startswith("40 to")
    assert t["rows"][-1]["g0"] == pytest.approx(100) and t["rows"][-1]["g1"] == pytest.approx(100)
    edges = 40 + 10 * np.arange(len(t["rows"]) + 1)
    exp = np.cumsum(np.histogram(B, bins=edges)[0]) / len(B) * 100
    assert [r["g1"] for r in t["rows"]] == pytest.approx(list(exp))
    res = run_ok("frequency", d, {**o, "n_bins": 4})
    assert len(rows(res, "Frequency distribution")) == 4


def test_frequency_edge_cases():
    res = run_ok("frequency", wide({"c": [5.0, 5.0, 5.0]}), {"columns": ["c"]})
    assert sum(r["g0"] for r in rows(res, "Frequency")) == 3
    # a value exactly on the last edge is still counted
    res = run_ok("frequency", wide({"x": [0.0, 1.0, 2.0, 10.0]}), {"columns": ["x"], "bin_width": 5})
    assert sum(r["g0"] for r in rows(res, "Frequency")) == 4
    assert "bins" in run_err("frequency", wide({"x": [0.0, 1e6]}), {"columns": ["x"], "bin_width": 0.5})
    assert "above the smallest" in run_err("frequency", wide({"x": [1.0, 2.0]}), {"columns": ["x"], "bin_start": 1.5,
                                                                                  "bin_width": 1})


# ============================================================== one sample

@BOTH
def test_one_sample_t_and_wilcoxon(d, o):
    res = run_ok("one_sample", d, {**o, "mu": 80})
    r = row(res, "One-sample t test", group="Control")
    close(r["t"], 3.78836882481)                                  # R: t.test(a, mu = 80)
    close(r["p"], 0.00300319160847)
    close(r["lo"] + 80, 88.5933009243)
    close(r["hi"] + 80, 112.423365742)
    close(r["diff"], A.mean() - 80)
    close(r["d"], (A.mean() - 80) / A.std(ddof=1))
    assert r["df"] == 11 and r["stars"] == "**"
    w = row(res, "Wilcoxon signed-rank", group="Control")
    close(w["t_pos"], 74)                                          # R: wilcox.test(a, mu = 80) → V = 74
    close(w["p"], 0.00341796875)
    close(w["w"], 74 - 4)
    assert w["method"] == "Exact"
    assert w["lo"] <= np.median(A) <= w["hi"] and w["actual"] >= 0.95
    assert "P = 0.0030" in summary(res)


def test_one_sample_one_tailed_and_ties():
    res = run_ok("one_sample", wire(TG), {"columns": ["Control"], "mu": 80, "tails": "greater"})
    close(row(res, "One-sample t", group="Control")["p"], stats.ttest_1samp(A, 80, alternative="greater").pvalue)
    close(row(res, "Wilcoxon", group="Control")["p"], stats.wilcoxon(A - 80, alternative="greater").pvalue)
    x = np.array([1, 1, 2, 2, 3, 3, 4, 5, 5, 0, 0], float)        # ties and zeros
    res = run_ok("one_sample", wide({"x": x}), {"columns": ["x"], "mu": 0})
    w = row(res, "Wilcoxon", group="x")
    assert w["method"] == "Normal approximation"
    close(w["p"], stats.wilcoxon(x[x != 0], correction=True, method="approx").pvalue)
    assert any("ignored" in f for f in table(res, "Wilcoxon")["footnotes"])


def test_one_sample_degenerate_groups():
    d = wide({"ok": [1.0, 2.0, 3.0, 5.0], "one": [4.0], "const": [2.0, 2.0, 2.0]})
    res = run_ok("one_sample", d, {"columns": ["ok", "one", "const"], "mu": 2})
    assert row(res, "One-sample t", group="one").get("p") is None
    assert row(res, "One-sample t", group="const").get("p") is None
    assert row(res, "One-sample t", group="ok")["p"] is not None
    warn = " ".join(e["content"] for e in res["elements"] if e["type"] == "text" and e["kind"] == "warning")
    assert "one" in warn and "const" in warn
    assert "at least 2" in run_err("one_sample", wide({"a": [1.0]}), {"columns": ["a"]})


# ========================================================= unpaired t test

@BOTH
def test_ttest_unpaired_welch(d, o):
    res = run_ok("ttest_unpaired", d, {**o, "normality": True})
    r = rows(res, "Unpaired t test")[0]
    close(r["t"], -5.00146675015)                                  # R: t.test(b, a)
    close(r["df"], 18.9526714772)
    close(r["p"], 7.98063110467e-05)
    assert r["stars"] == "****" and r["sig"] == "Yes"
    e = rows(res, "How big is the difference")[0]
    assert e["label"] == "Drug B − Control"
    close(e["diff"], B.mean() - A.mean())
    close(e["lo"], -45.8901827904)
    close(e["hi"], -18.8098172096)
    sp = math.sqrt((A.var(ddof=1) + B.var(ddof=1)) / 2)
    close(e["d"], (B.mean() - A.mean()) / sp)
    close(e["g"], e["d"] * (1 - 3 / (4 * 22 - 1)))
    assert e["d_lo"] < e["d"] < e["d_hi"]
    # noncentral-t CI: the limits reproduce the tail probabilities
    close(stats.nct.cdf(e["d"] * math.sqrt(6), 22, e["d_lo"] * math.sqrt(6)), 0.975, rel=1e-5)
    close(stats.nct.cdf(e["d"] * math.sqrt(6), 22, e["d_hi"] * math.sqrt(6)), 0.025, rel=1e-4)
    f = rows(res, "F test")[0]
    close(f["f"], 2.33879548229)                                   # R: var.test(a, b)
    close(f["p"], 0.174495337843)
    assert (f["dfn"], f["dfd"]) == (11, 11)
    close(row(res, "Descriptive statistics", group="Drug B")["mean"], B.mean())
    close(row(res, "Normality", group="Control")["p"], 0.622018410071, rel=1e-5)
    s = summary(res)
    assert "statistically significant" in s and "P < 0.0001" in s and "Welch" in s
    assert kinds(res).count("plot") == 1
    g = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert g["type"] == "column_scatter" and g["options"]["layout"] == o["layout"]


@BOTH
def test_ttest_unpaired_student_and_one_tailed(d, o):
    res = run_ok("ttest_unpaired", d, {**o, "test": "student", "conf_level": 90})
    r = rows(res, "Unpaired t test")[0]
    close(r["t"], -5.00146675015)                                  # R: t.test(b, a, var.equal = TRUE)
    assert r["df"] == 22
    close(r["p"], 5.24973455937e-05)
    ref = stats.ttest_ind(B, A).confidence_interval(0.90)
    e = rows(res, "How big is the difference")[0]
    close(e["lo"], ref.low)
    close(e["hi"], ref.high)
    res = run_ok("ttest_unpaired", d, {**o, "tails": "less"})
    close(rows(res, "Unpaired t test")[0]["p"], 3.99031555233e-05)  # R: t.test(b, a, alternative = "less")
    res = run_ok("ttest_unpaired", d, {**o, "tails": "greater"})
    close(rows(res, "Unpaired t test")[0]["p"], 1 - 3.99031555233e-05)
    assert "one-tailed" in summary(res)


def test_ttest_unpaired_unequal_n_and_nan():
    a = [5.1, 4.9, None, 6.2, 5.8, 5.5, None]
    b = [7.0, 6.5, 7.7, None]
    res = run_ok("ttest_unpaired", data(col("a", a), col("b", b + [None] * 3)), {"columns": ["a", "b"]})
    aa, bb = [v for v in a if v is not None], [v for v in b if v is not None]
    ref = stats.ttest_ind(bb, aa, equal_var=False)
    r = rows(res, "Unpaired t test")[0]
    close(r["t"], ref.statistic)
    close(r["p"], ref.pvalue)
    close(r["df"], ref.df)
    assert row(res, "Descriptive statistics", group="a")["n"] == 5


def test_two_group_error_paths():
    three = {"layout": "wide", "columns": ["Control", "Drug A", "Drug B"]}
    for aid in ("ttest_unpaired", "mann_whitney", "ks_two_sample", "ttest_paired", "wilcoxon_paired"):
        assert "exactly two" in run_err(aid, wire(TG), three), aid
        assert "exactly two" in run_err(aid, wire(TG), {"columns": ["Control"]}), aid
    d = wide({"a": [1.0, 2.0, 3.0], "b": [4.0]})
    assert "only 1 value" in run_err("ttest_unpaired", d, {"columns": ["a", "b"]})
    assert "only 1 value" in run_err("ks_two_sample", d, {"columns": ["a", "b"]})
    d = wide({"a": [2.0, 2.0, 2.0], "b": [4.0, 4.0, 4.0]})
    assert "variability" in run_err("ttest_unpaired", d, {"columns": ["a", "b"]})
    d = wide({"a": [2.0, 2.0, 2.0], "b": [2.0, 2.0, 2.0]})
    assert "identical" in run_err("mann_whitney", d, {"columns": ["a", "b"]})
    # long layout with three levels
    d3 = to_long({"x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0], "z": [5.0, 6.0, 7.0]})
    assert "exactly two" in run_err("ttest_unpaired", d3, {"layout": "long", "dep": "value", "group": "grp"})
    assert "Grouping variable" in run_err("ttest_unpaired", d3, {"layout": "long", "dep": "value"})


def test_ttest_one_constant_group_still_works():
    d = wide({"a": [2.0, 2.0, 2.0, 2.0], "b": [3.0, 4.0, 5.0, 4.5]})
    res = run_ok("ttest_unpaired", d, {"columns": ["a", "b"]})
    x = np.array([3.0, 4.0, 5.0, 4.5])
    t = (x.mean() - 2.0) / (x.std(ddof=1) / 2)                    # Welch with one zero variance: df = n − 1
    close(rows(res, "Unpaired t test")[0]["t"], t)
    close(rows(res, "Unpaired t test")[0]["df"], 3)
    close(rows(res, "Unpaired t test")[0]["p"], 2 * stats.t.sf(t, 3))
    assert rows(res, "F test")[0].get("p") is None


# ============================================================ Mann–Whitney

@BOTH
def test_mann_whitney_exact(d, o):
    res = run_ok("mann_whitney", d, o)
    r = rows(res, "Mann–Whitney test")[0]
    close(r["u"], 11)                                              # R: wilcox.test(b, a, conf.int = TRUE) → W = 11
    close(r["p"], 0.000144222448705)
    assert r["method"] == "Exact"
    close(r["rb"], (11 - 133) / 144)
    e = rows(res, "Difference between medians")[0]
    close(e["hl"], -34.1)
    close(e["lo"], -45.3)
    close(e["hi"], -19.2)
    assert e["actual"] >= 0.95
    close(e["med_diff"], np.median(B) - np.median(A))
    dr = row(res, "Descriptive statistics", group="Control")
    close(dr["rank_sum"], 211)
    close(dr["u"], 133)
    assert "Mann–Whitney" in summary(res)


def test_mann_whitney_conf_level_ties_and_tails():
    res = run_ok("mann_whitney", wire(TG), {"columns": ["Control", "Drug A"], "conf_level": 90})
    r = rows(res, "Mann–Whitney test")[0]
    close(r["u"], 34)                                              # R: wilcox.test(cc, a, conf.int = T, conf.level = .9)
    close(r["p"], 0.0284206976225)
    e = rows(res, "Difference between medians")[0]
    close(e["hl"], -18.55)
    close(e["lo"], -28.6)
    close(e["hi"], -5.3)
    a = np.array([1, 2, 2, 3, 4, 5, 5, 6], float)
    b = np.array([4, 5, 6, 6, 7, 8, 9], float)
    res = run_ok("mann_whitney", wide({"a": a, "b": b}), {"columns": ["a", "b"], "tails": "greater"})
    r = rows(res, "Mann–Whitney test")[0]
    assert r["method"] == "Normal approximation"
    close(r["p"], stats.mannwhitneyu(b, a, alternative="greater", method="asymptotic").pvalue)
    close(r["ps"], stats.mannwhitneyu(b, a).statistic / (len(a) * len(b)))
    # exact U distribution used for the CI agrees with scipy's exact P value
    from analysis_ui.analyses.two_groups import _mwu_cdf
    cdf = _mwu_cdf(5, 7)
    assert cdf[-1] == pytest.approx(1.0)
    x, y = np.array([1.5, 2.5, 9.0, 3.5, 0.5]), np.array([2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0])
    u = stats.mannwhitneyu(x, y, method="exact")
    close(2 * cdf[int(u.statistic)], u.pvalue)


def test_mann_whitney_tiny_samples():
    assert "at least 4" in run_err("mann_whitney", wide({"a": [1.0], "b": [2.0, 3.0]}), {"columns": ["a", "b"]})
    res = run_ok("mann_whitney", wide({"a": [1.0, 2.0], "b": [3.0, 4.0]}), {"columns": ["a", "b"]})
    close(rows(res, "Mann–Whitney test")[0]["p"], 1 / 3)
    assert rows(res, "Difference between medians")[0]["hl"] == 2.0


# ================================================================== paired

@PAIRED
def test_ttest_paired(d, o):
    res = run_ok("ttest_paired", d, {**o, "normality": True})
    r = rows(res, "Paired t test")[0]
    close(r["t"], -3.54524099859)                                  # R: t.test(After, Before, paired = TRUE)
    close(r["p"], 0.00323132330056)
    assert r["df"] == 14 and r["n"] == 15
    e = rows(res, "How big is the difference")[0]
    assert e["label"] == "After − Before"
    close(e["mean"], -6.06666666667)
    close(e["lo"], -9.73685605912)
    close(e["hi"], -2.39647727422)
    dd = AFTER - BEFORE
    close(e["sd"], dd.std(ddof=1))
    close(e["dz"], dd.mean() / dd.std(ddof=1))
    assert e["dz_lo"] < e["dz"] < e["dz_hi"]
    pr = rows(res, "effective")[0]
    ref = stats.pearsonr(BEFORE, AFTER, alternative="greater")
    close(pr["r"], ref.statistic)
    close(pr["p"], ref.pvalue, rel=1e-5)
    assert pr["effective"] == "Yes"
    close(rows(res, "Normality of the differences")[0]["p"], stats.shapiro(dd).pvalue)
    assert "P = 0.0032" in summary(res) and "paired t test" in summary(res)
    g = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert g["type"] == "before_after" and g["options"]["layout"] == o["layout"]
    if o["layout"] == "long":
        assert g["options"]["subject"] == "id"


@PAIRED
def test_ttest_ratio_paired(d, o):
    res = run_ok("ttest_paired", d, {**o, "test": "ratio"})
    r = rows(res, "Ratio paired t test")[0]
    close(r["t"], -3.66754380202)                                  # R: t.test(log10(After / Before))
    close(r["p"], 0.00253516503841)
    e = rows(res, "How big is the difference")[0]
    close(e["gm"], 0.956778037659)
    close(e["gm_lo"], 0.932372700785)
    close(e["gm_hi"], 0.981822196826)
    assert "ratio" in summary(res)


@PAIRED
def test_wilcoxon_paired_with_ties(d, o):
    res = run_ok("wilcoxon_paired", d, o)
    r = rows(res, "Wilcoxon matched-pairs")[0]
    close(r["t_pos"], 15)                                          # R: wilcox.test(After, Before, paired = TRUE) → V = 15
    close(r["p"], 0.0114234652452, rel=1e-6)
    assert r["method"] == "Normal approximation" and r["n"] == 15
    close(r["w"], 15 - 105)
    e = rows(res, "Median of differences")[0]
    close(e["median"], np.median(AFTER - BEFORE))
    assert e["lo"] <= e["median"] <= e["hi"]
    sp = stats.spearmanr(BEFORE, AFTER, alternative="greater")
    close(rows(res, "effective")[0]["rs"], sp.statistic)
    close(rows(res, "effective")[0]["p"], sp.pvalue)
    assert "Wilcoxon" in summary(res)


def test_wilcoxon_paired_exact_matches_r():
    x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]     # R ?wilcox.test depression data
    y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]
    res = run_ok("wilcoxon_paired", wide({"x": x, "y": y}), {"columns": ["x", "y"]})
    r = rows(res, "Wilcoxon matched-pairs")[0]
    close(r["t_pos"], 5)                                           # R: wilcox.test(y, x, paired = T, conf.int = T)
    close(r["p"], 0.0390625)
    assert r["method"] == "Exact"
    e = rows(res, "Median of differences")[0]
    close(e["hl"], -0.46)
    close(e["hl_lo"], -0.786)
    close(e["hl_hi"], -0.01)
    assert e["hl_actual"] >= 0.95
    res = run_ok("wilcoxon_paired", wide({"x": x, "y": y}), {"columns": ["x", "y"], "tails": "less"})
    close(rows(res, "Wilcoxon matched-pairs")[0]["p"],
          stats.wilcoxon(np.array(y) - np.array(x), alternative="less").pvalue)


def test_paired_missing_values_and_errors():
    b = [10.0, 12.0, None, 11.0, 14.0, 9.0, 13.0]
    a = [11.0, 14.5, 12.0, None, 15.0, 9.5, 15.5]
    d = data(col("before", b), col("after", a))
    res = run_ok("ttest_paired", d, {"columns": ["before", "after"]})
    bb = np.array([10.0, 12.0, 14.0, 9.0, 13.0])
    aa = np.array([11.0, 14.5, 15.0, 9.5, 15.5])
    ref = stats.ttest_rel(aa, bb)
    r = rows(res, "Paired t test")[0]
    close(r["t"], ref.statistic)
    close(r["p"], ref.pvalue)
    assert r["n"] == 5
    assert any("excluded" in f for f in table(res, "Paired t test")["footnotes"])
    run_ok("wilcoxon_paired", d, {"columns": ["before", "after"]})
    # constant differences
    d = wide({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]})
    assert "same difference" in run_err("ttest_paired", d, {"columns": ["a", "b"]})
    assert "zero" in run_err("wilcoxon_paired", wide({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}),
                             {"columns": ["a", "b"]})
    # too few pairs
    d = data(col("a", [1.0, None, 3.0]), col("b", [None, 2.0, 4.0]))
    assert "complete pair" in run_err("ttest_paired", d, {"columns": ["a", "b"]})
    # ratio test with non-positive values
    d = wide({"a": [1.0, 2.0, 3.0], "b": [0.0, 3.0, 5.0]})
    assert "greater than zero" in run_err("ttest_paired", d, {"columns": ["a", "b"], "test": "ratio"})
    # duplicated subject in long layout
    dl = data(col("v", [1.0, 2.0, 3.0, 4.0]), col("g", ["a", "a", "b", "b"], "nominal"),
              col("id", ["s1", "s1", "s1", "s2"], "nominal"))
    assert "more than one value" in run_err("ttest_paired", dl, {"layout": "long", "dep": "v", "group": "g",
                                                                 "subject": "id"})
    # long layout without subject: paired by order within group
    dl = data(col("v", [10.0, 12.0, 14.0, 11.0, 14.5, 15.0]), col("g", ["a"] * 3 + ["b"] * 3, "nominal"))
    res = run_ok("ttest_paired", dl, {"layout": "long", "dep": "v", "group": "g"})
    close(rows(res, "Paired t test")[0]["p"], stats.ttest_rel([11.0, 14.5, 15.0], [10.0, 12.0, 14.0]).pvalue)


# ====================================================================== KS

@BOTH
def test_ks_two_sample(d, o):
    res = run_ok("ks_two_sample", d, o)
    r = rows(res, "Kolmogorov–Smirnov test")[0]
    close(r["d"], 0.75)                                            # R: ks.test(a, b)
    close(r["p"], 0.00149695505733)
    assert r["n1"] == 12 and r["n2"] == 12
    assert "D = 0.75" in summary(res)
    assert "plot" in kinds(res)


# ========================================================= multiple t tests

BIO2 = BIO[BIO["Subtype"].isin(["Type I", "Type II"])].reset_index(drop=True)
MARKERS = [c for c in BIO.columns if c != "Subtype"]


@pytest.mark.parametrize("test", ["welch", "student", "mann_whitney"])
@pytest.mark.parametrize("corr,sm", [("fdr_bh", "fdr_bh"), ("fdr_by", "fdr_by"), ("holm-sidak", "holm-sidak"),
                                     ("bonferroni", "bonferroni"), ("none", None)])
def test_multiple_ttests(test, corr, sm):
    from statsmodels.stats.multitest import multipletests
    res = run_ok("multiple_ttests", wire(BIO2), {"variables": MARKERS, "group": "Subtype", "test": test,
                                                 "correction": corr})
    rr = rows(res, "Multiple")
    assert [r["variable"] for r in rr] == MARKERS
    pv = []
    for m in MARKERS:
        a = BIO2.loc[BIO2["Subtype"] == "Type I", m].to_numpy()
        b = BIO2.loc[BIO2["Subtype"] == "Type II", m].to_numpy()
        if test == "mann_whitney":
            pv.append(stats.mannwhitneyu(b, a).pvalue)
        else:
            pv.append(stats.ttest_ind(b, a, equal_var=test == "student").pvalue)
    for r, p in zip(rr, pv):
        close(r["p"], p)
    adj = pv if sm is None else multipletests(pv, method=sm)[1]
    for r, q in zip(rr, adj):
        close(r["adj"], q)
        assert r["hit"] == ("Yes" if q < 0.05 else "No")
    if test == "welch":
        a = BIO2.loc[BIO2["Subtype"] == "Type I", "IL-6"]
        b = BIO2.loc[BIO2["Subtype"] == "Type II", "IL-6"]
        close(rr[0]["diff"], b.mean() - a.mean())
        close(rr[0]["stat"], stats.ttest_ind(b, a, equal_var=False).statistic)
    assert "plot" in kinds(res)
    assert str(sum(r["hit"] == "Yes" for r in rr)) in summary(res)


def test_multiple_ttests_errors_and_partial_data():
    msg = run_err("multiple_ttests", wire(BIO), {"variables": MARKERS, "group": "Subtype"})
    assert "exactly 2 levels" in msg and "3" in msg
    d = data(col("ok", [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]), col("sparse", [1.0, None, None, 2.0, 3.0, 4.0]),
             col("const", [1.0] * 6), col("g", ["a", "a", "a", "b", "b", "b"], "nominal"))
    res = run_ok("multiple_ttests", d, {"variables": ["ok", "sparse", "const"], "group": "g"})
    rr = {r["variable"]: r for r in rows(res, "Multiple")}
    assert rr["ok"]["p"] is not None
    assert rr["sparse"]["p"] is None and "fewer than 2" in rr["sparse"]["note"]
    assert rr["const"]["p"] is None and "variability" in rr["const"]["note"]
    close(rr["ok"]["adj"], rr["ok"]["p"])                      # only one valid test → no adjustment
    assert "None of the variables" in run_err("multiple_ttests", d, {"variables": ["sparse", "const"], "group": "g"})
    assert "cannot also be" in run_err("multiple_ttests", d, {"variables": ["ok", "g"], "group": "g"})
    # numeric 0/1 grouping column
    d = data(col("y", [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]), col("grp", [0, 0, 0, 1, 1, 1]))
    res = run_ok("multiple_ttests", d, {"variables": ["y"], "group": "grp"})
    close(rows(res, "Multiple")[0]["p"], stats.ttest_ind([4, 5, 7], [1, 2, 3], equal_var=False).pvalue)


# ============================================================= correlation

def test_correlation_two_variables():
    res = run_ok("correlation", wire(REG), {"variables": ["Age", "Systolic BP"]})
    r = rows(res, "Pearson correlation")[0]
    close(r["r"], 0.666469549967)                                  # R: cor.test(Age, SBP)
    close(r["p"], 6.15155979863e-09, rel=1e-5)
    close(r["lo"], 0.496583143467)
    close(r["hi"], 0.787177737642)
    close(r["r2"], 0.666469549967 ** 2)
    assert r["n"] == 60 and r["sig"] == "Yes"
    s = summary(res)
    assert "significant" in s and "positive" in s and "P < 0.0001" in s
    assert "plot" in kinds(res)
    res = run_ok("correlation", wire(REG), {"variables": ["Age", "Systolic BP"], "method": "spearman"})
    r = rows(res, "Spearman")[0]
    close(r["r"], 0.696873611683)                                  # R: cor.test(method = "spearman")
    close(r["p"], 6.23389033868e-10, rel=1e-5)
    res = run_ok("correlation", wire(REG), {"variables": ["Age", "Systolic BP"], "method": "kendall"})
    r = rows(res, "Kendall")[0]
    close(r["r"], 0.51581377131)                                   # R: cor.test(method = "kendall")
    close(r["p"], stats.kendalltau(REG["Age"], REG["Systolic BP"]).pvalue)
    assert r["lo"] < r["r"] < r["hi"]


def test_correlation_matrix_pairwise_complete():
    df = REG[["Age", "BMI", "Systolic BP"]].copy()
    df.loc[[0, 5, 9], "BMI"] = np.nan
    df.loc[[2], "Age"] = np.nan
    res = run_ok("correlation", wire(df), {"variables": list(df.columns), "show_ci": True})
    m = rows(res, "Correlation matrix")
    ref = df.corr(method="pearson")
    for i, a in enumerate(df.columns):
        assert m[i]["var"] == a
        for j, b in enumerate(df.columns):
            close(m[i][f"c{j}"], ref.loc[a, b])
    pm = rows(res, "P values")
    ok = df["Age"].notna() & df["BMI"].notna()
    close(pm[0]["c1"], stats.pearsonr(df.loc[ok, "Age"], df.loc[ok, "BMI"]).pvalue)
    assert pm[0]["c0"] is None
    nm = rows(res, "Number of XY pairs")
    assert nm[0]["c1"] == 56 and nm[1]["c1"] == 57 and nm[0]["c2"] == 59
    assert len(rows(res, "All pairs")) == 3
    g = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert g["type"] == "corr_matrix" and g["options"]["columns"] == list(df.columns)
    assert "strongest" in summary(res)


def test_correlation_errors_and_degenerate():
    assert "at least 2" in run_err("correlation", wire(REG), {"variables": ["Age"]})
    assert "different" in run_err("correlation", wire(REG), {"variables": ["Age", "Age"]})
    d = data(col("x", [1.0, 2.0, 3.0, 4.0]), col("c", [5.0, 5.0, 5.0, 5.0]), col("y", [2.0, 1.0, 4.0, 3.0]))
    assert "identical" in run_err("correlation", d, {"variables": ["x", "c"]})
    d2 = data(col("x", [1.0, 2.0, None, None]), col("y", [None, 1.0, 4.0, 3.0]))
    assert "at least 3" in run_err("correlation", d2, {"variables": ["x", "y"]})
    res = run_ok("correlation", d, {"variables": ["x", "c", "y"]})      # constant column inside a matrix
    m = rows(res, "Correlation matrix")
    assert m[0]["c1"] is None and m[1]["c1"] is None and m[0]["c2"] is not None
    assert any("c" in f for f in table(res, "Correlation matrix")["footnotes"])
    # perfect correlation does not blow up
    d3 = data(col("x", [1.0, 2.0, 3.0, 4.0, 5.0]), col("y", [2.0, 4.0, 6.0, 8.0, 10.0]))
    res = run_ok("correlation", d3, {"variables": ["x", "y"]})
    assert rows(res, "Pearson")[0]["r"] == pytest.approx(1.0)


# ======================================================= linear regression

def test_linear_regression_matches_r():
    res = run_ok("linear_regression", wire(REG), {"x": "Age", "y": "Systolic BP", "interp_x": "40, 200",
                                                  "interp_y": "130"})
    b = rows(res, "Best-fit values")[0]
    close(b["slope"], 0.530971496258)                              # R: lm(SBP ~ Age)
    close(b["inter"], 99.3937455345)
    close(b["xint"], -99.3937455345 / 0.530971496258)
    close(b["inv"], 1 / 0.530971496258)
    s = rows(res, "Standard errors")[0]
    close(s["se_slope"], 0.0779907928039)
    close(s["se_inter"], 3.69893483676)
    close(s["slope_lo"], 0.374855962701)
    close(s["slope_hi"], 0.687087029816)
    close(s["inter_lo"], 91.9895229991)
    close(s["inter_hi"], 106.79796807)
    assert s["xint_lo"] < b["xint"] < s["xint_hi"]
    g = rows(res, "Goodness of fit")[0]
    close(g["r2"], 0.444181661033)
    close(g["syx"], 9.28808894954)
    assert g["n"] == 60 and g["df"] == 58
    f = rows(res, "slope significantly non-zero")[0]
    close(f["f"], 46.3506410886)
    close(f["p"], 6.15155979863e-09, rel=1e-5)
    it = rows(res, "Interpolated values")
    close(it[0]["y"], 120.632605385)                               # R: predict(m, data.frame(Age = 40), interval = ...)
    close(it[0]["lo"], 118.114995336)
    close(it[0]["hi"], 123.150215433)
    close(it[0]["plo"], 101.870791508)
    close(it[0]["phi"], 139.394419262)
    assert it[1]["note"] == "outside the range of the data"
    close(it[2]["x"], (130 - 99.3937455345) / 0.530971496258)
    assert kinds(res).count("plot") == 2
    gr = [e for e in res["elements"] if e["type"] == "graph"][0]["graph"]
    assert gr == {"type": "xy_regression", "options": {"x": "Age", "y": ["Systolic BP"], "group": None}}
    assert "P < 0.0001" in summary(res) and "R²" in summary(res)


def test_linear_regression_fieller_and_runs():
    from analysis_ui.analyses.correlation import linfit, runs_test
    x, y = REG["Age"].to_numpy(), REG["Systolic BP"].to_numpy()
    f = linfit(x, y)
    # Fieller limits are exactly the X values where the confidence band of the line touches Y = 0
    for x0 in (f["xint_lo"], f["xint_hi"]):
        yhat = f["inter"] + f["slope"] * x0
        half = f["tc"] * f["syx"] * math.sqrt(1 / f["n"] + (x0 - f["xbar"]) ** 2 / f["sxx"])
        close(abs(yhat), half, rel=1e-8)
    # runs test: exact distribution sums to 1 and matches brute force
    import itertools
    n1, n2 = 4, 3
    counts = {}
    for pos in itertools.combinations(range(n1 + n2), n1):
        s = np.full(n1 + n2, -1.0)
        s[list(pos)] = 1.0
        runs = 1 + int(np.sum(s[1:] != s[:-1]))
        counts[runs] = counts.get(runs, 0) + 1
    total = sum(counts.values())
    for runs in sorted(counts):
        s = np.array([1.0] * n1 + [-1.0] * n2)
        # build a sequence with exactly `runs` runs to query the function
        seq = next(np.array([1.0 if i in pos else -1.0 for i in range(n1 + n2)])
                   for pos in map(set, itertools.combinations(range(n1 + n2), n1))
                   if 1 + sum(1 for i in range(1, n1 + n2) if (i in pos) != (i - 1 in pos)) == runs)
        res = runs_test(seq)
        assert res["runs"] == runs
        close(res["p"], sum(c for r_, c in counts.items() if r_ <= runs) / total)
    # curved data → few runs → small P
    xx = np.linspace(0, 10, 30)
    res = run_ok("linear_regression", data(col("x", xx), col("y", xx ** 2)), {"x": "x", "y": "y"})
    rt = rows(res, "Runs test")[0]
    assert rt["runs"] == 3 and rt["p"] < 0.001 and rt["dev"] == "Significant"


def test_linear_regression_groups_ancova():
    res = run_ok("linear_regression", wire(REG), {"x": "Age", "y": "Systolic BP", "group": "Sex"})
    for sex in ("Male", "Female"):
        sub = REG[REG["Sex"] == sex]
        ref = stats.linregress(sub["Age"], sub["Systolic BP"])
        close(row(res, "Best-fit values", g=sex)["slope"], ref.slope)
        close(row(res, "Best-fit values", g=sex)["inter"], ref.intercept)
        close(row(res, "Standard errors", g=sex)["se_slope"], ref.stderr)
        close(row(res, "Standard errors", g=sex)["se_inter"], ref.intercept_stderr)
        close(row(res, "Goodness of fit", g=sex)["r2"], ref.rvalue ** 2)
        close(row(res, "non-zero", g=sex)["p"], ref.pvalue, rel=1e-5)
    c = rows(res, "Are the lines different")
    close(c[0]["f"], 1.26031450482)                                # R: anova(lm(SBP ~ Age + Sex), lm(SBP ~ Age * Sex))
    close(c[0]["p"], 0.266381806957)
    assert (c[0]["dfn"], c[0]["dfd"]) == (1, 56)
    close(c[1]["f"], 13.8045474947)                                # R: anova(lm(SBP ~ Age), lm(SBP ~ Age + Sex))
    close(c[1]["p"], 0.000463838183583)
    assert "0.5052" in c[0]["concl"]                               # pooled slope, R: coef(lm(SBP ~ Age + Sex))[2]
    s = summary(res)
    assert "slopes are not significantly different" in s and "elevations" in s


def test_linear_regression_origin():
    res = run_ok("linear_regression", wire(REG), {"x": "Age", "y": "Systolic BP", "origin": True})
    close(rows(res, "Best-fit values")[0]["slope"], 2.5134850775)   # R: lm(SBP ~ Age - 1)
    assert rows(res, "Best-fit values")[0]["inter"] == 0
    close(rows(res, "Standard errors")[0]["se_slope"], 0.0919287688351)
    close(rows(res, "Goodness of fit")[0]["r2"], 0.926850405776)
    close(rows(res, "Goodness of fit")[0]["syx"], 33.7722825774)
    assert rows(res, "Goodness of fit")[0]["df"] == 59


def test_linear_regression_edge_cases():
    assert "different" in run_err("linear_regression", wire(REG), {"x": "Age", "y": "Age"})
    d = data(col("x", [1.0, 1.0, 1.0, 1.0]), col("y", [1.0, 2.0, 3.0, 4.0]))
    assert "identical" in run_err("linear_regression", d, {"x": "x", "y": "y"})
    d = data(col("x", [1.0, 2.0, None, 4.0]), col("y", [1.0, None, 3.0, 4.0]))
    assert "At least 3" in run_err("linear_regression", d, {"x": "x", "y": "y"})
    assert "abc" in run_err("linear_regression", wire(REG), {"x": "Age", "y": "BMI", "interp_x": "1, abc"})
    # perfect line: no crash, warning shown
    d = data(col("x", [1.0, 2.0, 3.0, 4.0]), col("y", [3.0, 5.0, 7.0, 9.0]))
    res = run_ok("linear_regression", d, {"x": "x", "y": "y"})
    close(rows(res, "Best-fit values")[0]["slope"], 2.0)
    assert any(e["type"] == "text" and e["kind"] == "warning" for e in res["elements"])
    # flat line (slope exactly 0)
    d = data(col("x", [1.0, 2.0, 3.0, 4.0]), col("y", [1.0, 2.0, 2.0, 1.0]))
    res = run_ok("linear_regression", d, {"x": "x", "y": "y", "interp_y": "5"})
    assert rows(res, "Best-fit values")[0]["slope"] == pytest.approx(0)
    assert rows(res, "Best-fit values")[0]["xint"] is None
    assert rows(res, "non-zero")[0]["p"] == pytest.approx(1.0)
    # a group that is too small is skipped with a warning, the others are fitted
    d = data(col("x", [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0]), col("y", [1.1, 2.3, 2.9, 4.2, 5.1, 3.0, 4.0]),
             col("g", ["a"] * 5 + ["b"] * 2, "nominal"))
    res = run_ok("linear_regression", d, {"x": "x", "y": "y", "group": "g"})
    assert len(rows(res, "Best-fit values")) == 1
    assert any("b (only 2 points)" in e["content"] for e in res["elements"] if e["type"] == "text")


# ================================================================== Deming

def test_deming_orthogonal_equals_first_principal_axis():
    x, y = MC["Assay A"].to_numpy(), MC["Assay B"].to_numpy()
    res = run_ok("deming", wire(MC), {"x": "Assay A", "y": "Assay B"})
    sl = row(res, "Deming regression", param="Slope")
    ic = row(res, "Deming regression", param="Y-intercept")
    w, v = np.linalg.eigh(np.cov(x, y))
    pc = v[:, np.argmax(w)]
    close(sl["est"], pc[1] / pc[0])
    close(ic["est"], y.mean() - pc[1] / pc[0] * x.mean())
    # jackknife standard errors recomputed independently
    from analysis_ui.analyses.correlation import deming_fit
    n = len(x)
    jk = np.array([deming_fit(np.delete(x, i), np.delete(y, i), 1.0) for i in range(n)])
    pseudo = n * np.array(deming_fit(x, y, 1.0)) - (n - 1) * jk
    se = pseudo.std(axis=0, ddof=1) / math.sqrt(n)
    close(sl["se"], se[0])
    close(ic["se"], se[1])
    tc = stats.t.ppf(0.975, n - 2)
    close(sl["lo"], sl["est"] - tc * se[0])
    close(ic["hi"], ic["est"] + tc * se[1])
    close(sl["p"], 2 * stats.t.sf(abs(sl["est"] - 1) / se[0], n - 2))
    assert "Deming" in table(res, "Deming regression")["footnotes"][0]
    assert summary(res)


def test_deming_ratio_limits_and_odr():
    from scipy import odr
    x, y = MC["Assay A"].to_numpy(), MC["Assay B"].to_numpy()
    ols_yx = stats.linregress(x, y).slope
    ols_xy = 1 / stats.linregress(y, x).slope
    res = run_ok("deming", wire(MC), {"x": "Assay A", "y": "Assay B", "ratio": 1e6})
    close(row(res, "Deming regression", param="Slope")["est"], ols_yx, rel=1e-4)
    res = run_ok("deming", wire(MC), {"x": "Assay A", "y": "Assay B", "ratio": 1e-6})
    close(row(res, "Deming regression", param="Slope")["est"], ols_xy, rel=1e-4)
    # general ratio against scipy's orthogonal distance regression with weights
    delta = 4.0
    res = run_ok("deming", wire(MC), {"x": "Assay A", "y": "Assay B", "ratio": delta})
    fit = odr.ODR(odr.RealData(x, y, sx=np.full(len(x), 1.0), sy=np.full(len(x), math.sqrt(delta))),
                  odr.Model(lambda b, xx: b[0] * xx + b[1]), beta0=[1.0, 0.0]).run()
    close(row(res, "Deming regression", param="Slope")["est"], fit.beta[0], rel=1e-6)
    close(row(res, "Deming regression", param="Y-intercept")["est"], fit.beta[1], rel=1e-5)


def test_deming_errors():
    d = data(col("x", [1.0, 2.0, 3.0]), col("y", [1.0, 2.0, 3.5]))
    assert "at least 4" in run_err("deming", d, {"x": "x", "y": "y"})
    d = data(col("x", [1.0, 1.0, 1.0, 1.0, 1.0]), col("y", [1.0, 2.0, 3.5, 2.0, 1.0]))
    assert "identical" in run_err("deming", d, {"x": "x", "y": "y"})
    assert "different" in run_err("deming", wire(MC), {"x": "Assay A", "y": "Assay A"})


# ============================================================= contingency

RAW = (wire(CT), {"data_form": "raw", "rows": "Treatment", "cols": "Outcome"})
COUNTS = (data(col("Arm", ["Placebo", "Drug"], "nominal"), col("Non-responder", [39, 28]), col("Responder", [21, 32])),
          {"data_form": "counts", "labels": "Arm", "count_cols": ["Non-responder", "Responder"]})
FORMS = pytest.mark.parametrize("d,o", [RAW, COUNTS], ids=["raw", "counts"])


@FORMS
def test_contingency_2x2(d, o):
    res = run_ok("contingency", d, {**o, "gtest": True, "diagnostic": True, "show_expected": True,
                                    "show_col_pct": True, "show_total_pct": True})
    obs = rows(res, "Observed counts")
    assert [obs[0]["c0"], obs[0]["c1"], obs[1]["c0"], obs[1]["c1"]] == [39, 21, 28, 32]
    assert obs[2]["total"] == 120 and obs[0]["total"] == 60 and obs[2]["c0"] == 67
    close(rows(res, "Expected counts")[0]["c0"], 33.5)
    close(rows(res, "Percentage of row total")[0]["c0"], 39 / 60)
    close(rows(res, "Percentage of column total")[0]["c0"], 39 / 67)
    close(rows(res, "Percentage of grand total")[1]["c1"], 32 / 120)
    r = row(res, "Tests of association", test="Pearson chi-square")
    close(r["stat"], 4.08898901718)                                # R: chisq.test(tb, correct = FALSE)
    close(r["p"], 0.0431634526298)
    assert r["df"] == 1
    r = row(res, "Tests of association", test="Chi-square with Yates' correction")
    close(r["stat"], 3.37932976626)                                # R: chisq.test(tb)
    close(r["p"], 0.0660188970444)
    close(row(res, "Tests of association", test="Fisher's exact test")["p"], 0.065554672613)   # R: fisher.test(tb)
    g = stats.chi2_contingency([[39, 21], [28, 32]], correction=False, lambda_="log-likelihood")
    close(row(res, "Tests of association", test="Likelihood-ratio G test")["stat"], g[0])
    close(row(res, "Strength", measure="Cramér's V")["value"], math.sqrt(4.08898901718 / 120))
    close(row(res, "Strength", measure="Phi coefficient")["value"], math.sqrt(4.08898901718 / 120))
    rr = row(res, "Effect sizes", measure="Relative risk")
    close(rr["value"], 1.39285714286)                              # R: epitools::riskratio(method = "wald")
    close(rr["lo"], 1.00326907459)
    close(rr["hi"], 1.9337295144)
    orr = row(res, "Effect sizes", measure="Odds ratio")
    close(orr["value"], 2.12244897959)                             # R: epitools::oddsratio(method = "wald")
    close(orr["lo"], 1.01880649874)
    close(orr["hi"], 4.42163421273)
    from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint
    rd = row(res, "Effect sizes", measure="Risk difference (attributable risk)")
    lo, hi = confint_proportions_2indep(39, 60, 28, 60, method="newcomb", compare="diff")
    close(rd["value"], 39 / 60 - 28 / 60)
    close(rd["lo"], lo)
    close(rd["hi"], hi)
    nnt = row(res, "Effect sizes", measure="Number needed to treat (or harm)")
    close(nnt["value"], 1 / (39 / 60 - 28 / 60))
    close(nnt["lo"], 1 / hi)
    close(nnt["hi"], 1 / lo)
    sens = row(res, "Diagnostic test", measure="Sensitivity")
    close(sens["value"], 39 / 67)
    wl, wh = proportion_confint(39, 67, method="wilson")
    close(sens["lo"], wl)
    close(sens["hi"], wh)
    close(row(res, "Diagnostic test", measure="Specificity")["value"], 32 / 53)
    close(row(res, "Diagnostic test", measure="Positive predictive value")["value"], 39 / 60)
    close(row(res, "Diagnostic test", measure="Negative predictive value")["value"], 32 / 60)
    lrp = row(res, "Diagnostic test", measure="Positive likelihood ratio")
    close(lrp["value"], (39 / 67) / (21 / 53))
    se = math.sqrt(1 / 39 - 1 / 67 + 1 / 21 - 1 / 53)
    close(lrp["lo"], lrp["value"] * math.exp(-1.959963984540054 * se))
    s = summary(res)
    assert "not statistically significant" in s and "P = 0.0656" in s and "Fisher" in s and "relative risk" in s
    assert "plot" in kinds(res)
    has_graph = any(e["type"] == "graph" and e["graph"]["type"] == "count_bar" for e in res["elements"])
    assert has_graph == (o["data_form"] == "raw")


@FORMS
def test_contingency_choose_row_and_column_of_interest(d, o):
    res = run_ok("contingency", d, {**o, "ref_row": "Drug", "event_col": "Responder"})
    rr = row(res, "Effect sizes", measure="Relative risk")
    close(rr["value"], (32 / 60) / (21 / 60))
    orr = row(res, "Effect sizes", measure="Odds ratio")
    close(orr["value"], 2.12244897959)                 # the odds ratio is symmetric under swapping both
    assert "Responder" in table(res, "Effect sizes")["title"] and "Drug vs. Placebo" in table(res, "Effect sizes")["title"]
    assert "not one of" in run_err("contingency", d, {**o, "ref_row": "Nope"})


def test_contingency_rxc_fisher_and_trend():
    tb3 = data(col("r", ["a", "b", "c"], "nominal"), col("x", [3, 1, 4]), col("y", [5, 7, 4]), col("z", [2, 9, 1]))
    res = run_ok("contingency", tb3, {"data_form": "counts", "labels": "r", "count_cols": ["x", "y", "z"],
                                      "trend": True})
    f = [r for r in rows(res, "Tests of association") if r["test"].startswith("Fisher")][0]
    close(f["p"], 0.0802262490289, rel=1e-6)                       # R: fisher.test(tb3)
    c = row(res, "Tests of association", test="Pearson chi-square")
    close(c["stat"], 8.25245098039)
    assert c["df"] == 4
    assert any(e["type"] == "text" and e["kind"] == "warning" and "expected count" in e["content"]
               for e in res["elements"])
    assert "needs a 2×k" in row(res, "Tests of association", test="Chi-square test for trend")["sig"]
    assert not any("Effect sizes" in t["title"] for t in tables(res))
    # 3 × 2 with ordered rows
    tb4 = data(col("dose", ["Low", "Mid", "High"], "nominal"), col("Yes", [3, 8, 15]), col("No", [20, 14, 9]))
    res = run_ok("contingency", tb4, {"data_form": "counts", "labels": "dose", "count_cols": ["Yes", "No"],
                                      "trend": True})
    t = row(res, "Tests of association", test="Chi-square test for trend (Cochran–Armitage)")
    close(t["stat"], 12.2445551952)                                # R: prop.trend.test(c(3, 8, 15), c(23, 22, 24))
    close(t["p"], 0.000466617755144)
    f = [r for r in rows(res, "Tests of association") if r["test"].startswith("Fisher")][0]
    close(f["p"], 0.00207621992211, rel=1e-6)                      # R: fisher.test(tb4)
    assert "trend" in summary(res)
    # the same table transposed (2 × k) gives the same trend test
    tb5 = data(col("o", ["Yes", "No"], "nominal"), col("Low", [3, 20]), col("Mid", [8, 14]), col("High", [15, 9]))
    res = run_ok("contingency", tb5, {"data_form": "counts", "labels": "o", "count_cols": ["Low", "Mid", "High"],
                                      "trend": True})
    close(row(res, "Tests of association", test="Chi-square test for trend (Cochran–Armitage)")["stat"], 12.2445551952)


def test_fisher_rxc_enumeration_and_monte_carlo_fallback():
    from analysis_ui.analyses.categorical import fisher_rxc, fisher_test, _TooManyTables
    # 2×2 special case agrees with scipy's exact test
    for tb in ([[3, 7], [8, 2]], [[0, 5], [6, 1]], [[10, 10], [10, 10]]):
        close(fisher_rxc(np.array(tb)), stats.fisher_exact(tb)[1])
    big = np.array([[12, 9, 15, 11, 8], [10, 14, 9, 13, 12], [8, 11, 12, 10, 16], [13, 9, 10, 12, 9]])
    with pytest.raises(_TooManyTables):
        fisher_rxc(big)
    p, how = fisher_test(big)
    assert how == "Monte Carlo" and 0 < p <= 1
    assert fisher_test(big)[0] == p                                 # seeded → reproducible
    chi_p = stats.chi2_contingency(big)[1]
    assert abs(p - chi_p) < 0.1


def test_contingency_zero_cell_and_missing():
    d = data(col("r", ["a", "b"], "nominal"), col("x", [0, 6]), col("y", [9, 4]))
    res = run_ok("contingency", d, {"data_form": "counts", "labels": "r", "count_cols": ["x", "y"]})
    orr = row(res, "Effect sizes", measure="Odds ratio")
    close(orr["value"], (0.5 * 4.5) / (9.5 * 6.5))
    assert any("0.5 was added" in f for f in table(res, "Effect sizes")["footnotes"])
    close(row(res, "Tests of association", test="Fisher's exact test")["p"], stats.fisher_exact([[0, 9], [6, 4]])[1])
    # raw data with missing cells and a numeric-coded variable
    d = data(col("exposed", [1, 1, 1, 0, 0, 0, None, 1, 0, 0]),
             col("sick", ["y", "y", "n", "n", "n", "y", "y", None, "n", "y"], "nominal"))
    res = run_ok("contingency", d, {"rows": "exposed", "cols": "sick"})
    obs = rows(res, "Observed counts")
    assert obs[-1]["total"] == 8
    assert [o_["row"] for o_ in obs] == ["0", "1", "Total"]


def test_contingency_errors():
    d = data(col("r", ["a", "b"], "nominal"), col("x", [1.5, 6]), col("y", [9, 4]))
    o = {"data_form": "counts", "labels": "r", "count_cols": ["x", "y"]}
    assert "whole numbers" in run_err("contingency", d, o)
    d = data(col("r", ["a", "b"], "nominal"), col("x", [-1, 6]), col("y", [9, 4]))
    assert "negative" in run_err("contingency", d, o)
    d = data(col("r", ["a", "b"], "nominal"), col("x", [None, 6]), col("y", [9, 4]))
    assert "empty cells" in run_err("contingency", d, o)
    d = data(col("r", ["a", "b"], "nominal"), col("x", [0, 0]), col("y", [9, 4]))
    assert "at least 2 rows and 2 columns" in run_err("contingency", d, o)
    assert "at least 2" in run_err("contingency", d, {"data_form": "counts", "count_cols": ["x"]})
    d = data(col("a", ["x", "x", "x"], "nominal"), col("b", ["p", "q", "p"], "nominal"))
    assert "at least 2 rows" in run_err("contingency", d, {"rows": "a", "cols": "b"})
    assert "different" in run_err("contingency", d, {"rows": "a", "cols": "a"})
    # counts without a label column get generic row names
    d = data(col("x", [5, 6]), col("y", [9, 4]))
    res = run_ok("contingency", d, {"data_form": "counts", "count_cols": ["x", "y"]})
    assert [r["row"] for r in rows(res, "Observed counts")] == ["Row 1", "Row 2", "Total"]


# ================================================================ binomial

def test_binomial_two_categories_raw_and_typed():
    for d, o in ((wire(CT), {"variable": "Outcome"}),
                 (None, {"data_form": "counts", "observed": "67, 53", "names": "Non-responder, Responder"})):
        res = run_ok("binomial", d, o)
        b = rows(res, "Exact binomial test")[0]
        assert b["x"] == 67 and b["n"] == 120 and b["cat"] == "Non-responder"
        close(b["p"], 0.235203367973)                              # R: binom.test(67, 120, 0.5)
        cp = row(res, "confidence interval", method="Clopper–Pearson (exact)")
        close(cp["lo"], 0.464826996316)
        close(cp["hi"], 0.648891889396)
        from statsmodels.stats.proportion import proportion_confint
        w = row(res, "confidence interval", method="Wilson (score)")
        lo, hi = proportion_confint(67, 120, method="wilson")
        close(w["lo"], lo)
        close(w["hi"], hi)
        o_ = rows(res, "Observed and expected")
        close(o_[0]["exp"], 60)
        assert o_[-1]["obs"] == 120
        assert "P = 0.2352" in summary(res) and "not significantly different" in summary(res)


def test_binomial_expected_and_success_level_and_tails():
    res = run_ok("binomial", None, {"data_form": "counts", "observed": "7;13", "expected": "60%, 40%"})
    close(rows(res, "Exact binomial test")[0]["p"], 0.0369900902678)   # R: binom.test(7, 20, 0.6)
    res = run_ok("binomial", None, {"data_form": "counts", "observed": "7, 13", "expected": "3, 2", "tails": "less"})
    close(rows(res, "Exact binomial test")[0]["p"], stats.binomtest(7, 20, 0.6, alternative="less").pvalue)
    res = run_ok("binomial", wire(CT), {"variable": "Outcome", "success": "Responder", "expected": "0.6, 0.4"})
    b = rows(res, "Exact binomial test")[0]
    assert b["cat"] == "Responder" and b["x"] == 53
    close(b["p"], stats.binomtest(53, 120, 0.4).pvalue)


def test_chisquare_goodness_of_fit():
    res = run_ok("binomial", None, {"data_form": "counts", "observed": "315,108,101,32", "expected": "9,3,3,1"})
    g = rows(res, "Chi-square goodness of fit")[0]
    close(g["chi2"], 0.470023980815)                               # R: chisq.test(c(315,108,101,32), p = c(9,3,3,1)/16)
    close(g["p"], 0.925425895104)
    assert g["df"] == 3 and g["n"] == 556
    close(rows(res, "Observed and expected")[0]["exp"], 312.75)
    assert "rescaled" in table(res, "Observed and expected")["footnotes"][0]
    # raw variable with three levels, equal expected
    res = run_ok("binomial", wire(BIO), {"variable": "Subtype"})
    counts = BIO["Subtype"].value_counts(sort=False).to_numpy()
    ref = stats.chisquare(counts)
    close(rows(res, "Chi-square goodness of fit")[0]["chi2"], ref.statistic, abs_=1e-12)
    close(rows(res, "Chi-square goodness of fit")[0]["p"], ref.pvalue)
    # small expected counts → warning
    res = run_ok("binomial", None, {"data_form": "counts", "observed": "3, 1, 2"})
    assert any(e["type"] == "text" and e["kind"] == "warning" for e in res["elements"])


def test_binomial_errors():
    c = {"data_form": "counts"}
    assert "Type the observed counts" in run_err("binomial", None, c)
    assert "abc" in run_err("binomial", None, {**c, "observed": "3, abc"})
    assert "whole numbers" in run_err("binomial", None, {**c, "observed": "3.5, 2"})
    assert "At least 2 categories" in run_err("binomial", None, {**c, "observed": "12"})
    assert "3 expected" in run_err("binomial", None, {**c, "observed": "5, 6", "expected": "1,2,3"})
    assert "zero" in run_err("binomial", None, {**c, "observed": "0, 0"})
    assert "impossible" in run_err("binomial", None, {**c, "observed": "5, 6, 2", "expected": "1, 1, 0"})
    assert "2 counts but 3" in run_err("binomial", None, {**c, "observed": "5, 6", "names": "a, b, c"})
    d = data(col("v", ["a", "a", "a"], "nominal"))
    assert "At least 2 categories" in run_err("binomial", d, {"variable": "v"})


# ================================================================= McNemar

def _mc_data(a, b, c, d, l1="yes", l2="no"):
    first = [l1] * (a + b) + [l2] * (c + d)
    second = [l1] * a + [l2] * b + [l1] * c + [l2] * d
    return data(col("before", first, "nominal"), col("after", second, "nominal"))


def test_mcnemar_matches_r_and_statsmodels():
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
    res = run_ok("mcnemar", _mc_data(4, 6, 1, 9), {"var1": "before", "var2": "after"})
    t = rows(res, "Paired table")
    assert (t[0]["c0"], t[0]["c1"], t[1]["c0"], t[1]["c1"], t[2]["total"]) == (4, 6, 1, 9, 20)
    close(row(res, "McNemar's test", test="Exact (binomial)")["p"], 0.125)           # R: binom.test(6, 7)
    cc = row(res, "McNemar's test", test="Chi-square with continuity correction")
    close(cc["stat"], 2.28571428571)                               # R: mcnemar.test(mm)
    close(cc["p"], 0.130570018116)
    close(row(res, "McNemar's test", test="Chi-square, uncorrected")["p"], 0.0587817213554)
    tb = [[4, 6], [1, 9]]
    close(row(res, "McNemar's test", test="Exact (binomial)")["p"], sm_mcnemar(tb, exact=True).pvalue)
    close(cc["p"], sm_mcnemar(tb, exact=False, correction=True).pvalue)
    d_ = rows(res, "Discordant pairs")
    assert d_[0]["count"] == 6 and d_[1]["count"] == 1 and d_[2]["count"] == 13
    orr = [r for r in rows(res, "Size of the change") if r["measure"].startswith("Odds ratio")][0]
    close(orr["value"], 6.0)
    ci = stats.binomtest(6, 7).proportion_ci(0.95, method="exact")
    close(orr["lo"], ci.low / (1 - ci.low))
    close(orr["hi"], ci.high / (1 - ci.high))
    diff = [r for r in rows(res, "Size of the change") if r["measure"].startswith("Difference")][0]
    close(diff["value"], 5 / 20 - 10 / 20)
    assert "P = 0.1250" in summary(res) and "not statistically significant" in summary(res)


def test_mcnemar_edge_cases():
    res = run_ok("mcnemar", _mc_data(5, 0, 0, 7), {"var1": "before", "var2": "after"})
    assert row(res, "McNemar's test", test="Exact (binomial)")["p"] == 1.0
    assert any("no discordant" in f for f in table(res, "McNemar's test")["footnotes"])
    res = run_ok("mcnemar", _mc_data(5, 8, 0, 7), {"var1": "before", "var2": "after"})
    close(row(res, "McNemar's test", test="Exact (binomial)")["p"], stats.binomtest(8, 8, 0.5).pvalue)
    # second variable only uses one of the two levels; missing values dropped
    d = data(col("b", ["y", "n", "y", "n", None, "y"], "nominal"), col("a", ["y", "y", "y", "y", "y", None], "nominal"))
    res = run_ok("mcnemar", d, {"var1": "b", "var2": "a"})
    assert rows(res, "Paired table")[2]["total"] == 4
    # errors
    d = data(col("b", ["y", "n", "m"], "nominal"), col("a", ["y", "y", "n"], "nominal"))
    assert "same two categories" in run_err("mcnemar", d, {"var1": "b", "var2": "a"})
    d = data(col("b", ["y", "y"], "nominal"), col("a", ["y", "y"], "nominal"))
    assert "same two categories" in run_err("mcnemar", d, {"var1": "b", "var2": "a"})
    assert "different" in run_err("mcnemar", d, {"var1": "b", "var2": "b"})


# ========================================================== proportions CI

@pytest.mark.parametrize("k,n,conf", [(12, 40, 95), (0, 25, 95), (25, 25, 95), (1, 7, 90), (480, 1000, 99)])
def test_proportions_ci_against_statsmodels(k, n, conf):
    from statsmodels.stats.proportion import proportion_confint
    res = run_ok("proportions_ci", None, {"numerator": k, "denominator": n, "conf_level": conf})
    alpha = 1 - conf / 100
    for label, method in (("Wilson (score)", "wilson"), ("Clopper–Pearson (exact)", "beta"),
                          ("Agresti–Coull", "agresti_coull"), ("Wald (normal approximation)", "normal")):
        r = row(res, "confidence interval", method=label)
        lo, hi = proportion_confint(k, n, alpha=alpha, method=method)
        close(r["prop"], k / n)
        close(r["lo"], max(0.0, lo), abs_=1e-10)
        close(r["hi"], min(1.0, hi), abs_=1e-10)
        assert 0 <= r["lo"] <= k / n <= r["hi"] <= 1
    assert f"{k} of {n}" in summary(res)


def test_proportions_ci_wilson_matches_r_and_raw_form():
    res = run_ok("proportions_ci", None, {"numerator": 12, "denominator": 40})
    w = row(res, "confidence interval", method="Wilson (score)")
    close(w["lo"], 0.180748452297)                                 # R: prop.test(12, 40, correct = FALSE)$conf.int
    close(w["hi"], 0.454300188181)
    res = run_ok("proportions_ci", wire(CT), {"data_form": "raw", "variable": "Outcome", "success": "Responder"})
    assert "53 / 120" in table(res, "confidence interval")["title"]
    res = run_ok("proportions_ci", wire(CT), {"data_form": "raw", "variable": "Outcome"})
    assert "67 / 120" in table(res, "confidence interval")["title"]


def test_proportions_ci_errors():
    assert "numerator" in run_err("proportions_ci", None, {"numerator": 5})
    assert "cannot be larger" in run_err("proportions_ci", None, {"numerator": 9, "denominator": 5})
    assert "Variable" in run_err("proportions_ci", wire(CT), {"data_form": "raw"})
    assert "not one of" in run_err("proportions_ci", wire(CT), {"data_form": "raw", "variable": "Outcome",
                                                               "success": "Maybe"})


# ================================================================ sanity

def test_results_contain_no_nan_and_are_json_serialisable():
    import json
    runs = [("descriptives", *AB_WIDE), ("ttest_unpaired", *AB_LONG), ("ttest_paired", *PAIR_LONG),
            ("contingency", *RAW), ("linear_regression", wire(REG), {"x": "Age", "y": "BMI", "group": "Sex"}),
            ("correlation", wire(REG), {"variables": ["Age", "BMI", "Systolic BP"]})]
    for aid, d, o in runs:
        res = run_ok(aid, d, o)
        text = json.dumps(res, allow_nan=False)
        assert "NaN" not in text
        for el in res["elements"]:
            if el["type"] == "plot":
                assert el["svg"].lstrip().startswith("<?xml") or "<svg" in el["svg"][:300]
            if el["type"] == "table":
                keys = {c["key"] for c in el["columns"]}
                assert el["title"]
                for r in el["rows"]:
                    assert set(r) <= keys, (aid, el["title"], set(r) - keys)
