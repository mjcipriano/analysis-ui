"""Tests for curve fitting, survival, diagnostics, multivariate and planning
analyses. Reference values come from R 4.3 (nls, survival, pROC, psych, pwr,
power.t.test, power.prop.test — the R code is quoted next to each value) or
are computed here with scipy / lifelines / scikit-learn / statsmodels."""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis_ui.core import registry, run_analysis
from analysis_ui.importer import read_file

registry.discover()

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / "analysis_ui" / "sample_data"
X_DR, Y_DR = "Concentration (µM)", "Response (%)"
PANEL = ["IL-6", "TNF-α", "CRP", "IL-10", "IFN-γ", "VEGF"]


# ------------------------------------------------------------------ helpers
def sample(name: str) -> dict:
    return read_file(name, (SAMPLES / name).read_bytes())[0]


def frame(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLES / name)


def wire(**cols) -> dict:
    """Wire-format data; lists of str become nominal columns."""
    out = []
    for name, values in cols.items():
        values = list(values)
        nominal = any(isinstance(v, str) for v in values)
        out.append({"name": name, "type": "nominal" if nominal else "continuous",
                    "values": [None if (isinstance(v, float) and math.isnan(v)) else v for v in values]})
    return {"columns": out}


def run(analysis_id: str, data, options: dict | None = None) -> dict:
    res = run_analysis(analysis_id, data, options or {})
    assert res["status"] == "ok", res.get("message")
    return res["results"]


def fail(analysis_id: str, data, options: dict | None = None) -> str:
    res = run_analysis(analysis_id, data, options or {})
    assert res["status"] == "error", "expected a user error"
    msg = res["message"]
    assert msg and "Traceback" not in msg
    # a crash would be reported as "SomeError: ..." by run_analysis
    assert not msg.split(":")[0].endswith(("Error", "Exception", "Warning")), msg
    return msg


def table(results: dict, title: str) -> list[dict]:
    for e in results["elements"]:
        if e["type"] == "table" and e["title"].startswith(title):
            return e["rows"]
    raise AssertionError(f"no table {title!r}; have {[e['title'] for e in results['elements'] if e['type'] == 'table']}")


def has_table(results: dict, title: str) -> bool:
    return any(e["type"] == "table" and e["title"].startswith(title) for e in results["elements"])


def texts(results: dict, kind: str) -> list[str]:
    return [e["content"] for e in results["elements"] if e["type"] == "text" and e["kind"] == kind]


def summary(results: dict) -> str:
    s = texts(results, "summary")
    assert len(s) == 1 and len(s[0]) > 20
    return s[0]


def plots(results: dict) -> list[str]:
    return [e["title"] for e in results["elements"] if e["type"] == "plot"]


def graph(results: dict) -> dict:
    g = [e["graph"] for e in results["elements"] if e["type"] == "graph"]
    assert len(g) == 1
    return g[0]


def param(rows: list[dict], name: str, dataset: str | None = None, key: str = "param") -> dict:
    for r in rows:
        if r[key] == name and (dataset is None or r.get("dataset") == dataset):
            return r
    raise AssertionError(f"{name} not in {[r[key] for r in rows]}")


def kv(rows: list[dict], label: str):
    for r in rows:
        if r["item"].startswith(label):
            return r.get("value") if r.get("value") is not None else r.get("text")
    raise AssertionError(f"{label} not in {[r['item'] for r in rows]}")


def approx(v, rel=1e-4, abs_=1e-8):
    return pytest.approx(v, rel=rel, abs=abs_)


def test_registry_has_all_analyses():
    ids = {"nonlinear_regression", "interpolate_standard_curve", "auc", "kaplan_meier", "roc", "bland_altman",
           "icc_agreement", "cohen_kappa", "pca", "kmeans", "hclust", "power_ttest", "power_anova",
           "power_proportions", "power_survival", "power_correlation"}
    assert ids <= set(registry.ANALYSES)
    for i in ids:
        assert registry.ANALYSES[i].public()["options"]


# ====================================================== nonlinear regression
NLR = {"x": X_DR, "y": [Y_DR], "group": "Compound"}


def test_nlr_4pl_matches_r_nls():
    # R: nls(y ~ b + (t-b)/(1+(e/x)^h), CompoundA, start=list(b=0,t=100,e=0.1,h=1))
    res = run("nonlinear_regression", sample("dose_response.csv"), NLR)
    best = table(res, "Best-fit values")
    ref = {"Bottom": (3.3870489869, 1.647495294442), "Top": (98.3540467867, 1.419471981432),
           "EC50": (0.1156072363, 0.008934210714), "HillSlope": (1.1994260622, 0.101626277010)}
    for name, (est, se) in ref.items():
        row = param(best, name, "Compound A")
        assert row["value"] == approx(est, 1e-5)
        assert row["se"] == approx(se, 1e-4)
        tcrit = stats.t.ppf(0.975, 26)
        assert row["lo"] == approx(est - tcrit * se, 1e-4, 1e-4)
        assert row["hi"] == approx(est + tcrit * se, 1e-4, 1e-4)
    gof = param(table(res, "Goodness of fit"), "Compound A", key="dataset")
    assert gof["ss"] == approx(412.1618, 1e-5) and gof["df"] == 26 and gof["n"] == 30
    assert gof["syx"] == approx(math.sqrt(412.1618 / 26), 1e-5)
    assert 0.99 < gof["r2"] < 1 and gof["adj"] < gof["r2"]
    # AICc as defined by Prism: N ln(SS/N) + 2K + 2K(K+1)/(N-K-1), K = parameters + 1
    assert gof["aicc"] == approx(30 * math.log(412.1618 / 30) + 10 + 60 / 24, 1e-5)
    derived = table(res, "Derived values")
    assert param(derived, "LogEC50", "Compound A")["value"] == approx(math.log10(0.1156072363), 1e-5)
    assert param(derived, "Span (Top − Bottom)", "Compound B")["value"] > 80
    assert "EC50" in summary(res) and "Compound B" in summary(res)
    assert plots(res) == ["Data and fitted curves", "Residuals"]
    g = graph(res)
    assert g["type"] == "xy" and g["options"] == {"x": X_DR, "y": [Y_DR], "group": "Compound", "fit": "4pl",
                                                  "x_scale": "log10"}
    assert has_table(res, "Runs test") and has_table(res, "Replicates test")


def test_nlr_constraints_and_weighting_match_r():
    data = sample("dose_response.csv")
    # R: nls(y ~ 0 + (100-0)/(1+(e/x)^h), ...)
    res = run("nonlinear_regression", data, {**NLR, "constraints": "bottom = 0, Top=100"})
    best = table(res, "Best-fit values")
    assert param(best, "EC50", "Compound A")["value"] == approx(0.1101387031, 1e-5)
    assert param(best, "EC50", "Compound A")["se"] == approx(0.006727802747, 1e-4)
    assert param(best, "HillSlope", "Compound A")["value"] == approx(1.0592345297, 1e-5)
    assert param(best, "Bottom (fixed)", "Compound A")["value"] == 0
    assert param(table(res, "Goodness of fit"), "Compound A", key="dataset")["df"] == 28
    # R: nls(..., weights = 1/y^2)
    res = run("nonlinear_regression", data, {**NLR, "weighting": "1/y2"})
    best = table(res, "Best-fit values")
    assert param(best, "EC50", "Compound A")["value"] == approx(0.1023829822, 1e-4)
    assert param(best, "EC50", "Compound A")["se"] == approx(0.02466773997, 1e-3)
    assert param(best, "Bottom", "Compound A")["value"] == approx(2.3794386852, 1e-4)

    assert "not a parameter" in fail("nonlinear_regression", data, {**NLR, "constraints": "Botom=0"})
    assert "Name=value" in fail("nonlinear_regression", data, {**NLR, "constraints": "Bottom 0"})
    assert "number" in fail("nonlinear_regression", data, {**NLR, "constraints": "Bottom=abc"})
    assert "nothing left" in fail("nonlinear_regression", sample("enzyme_kinetics.csv"),
                                  {"x": "Substrate (mM)", "y": ["Velocity"], "model": "michaelis_menten",
                                   "constraints": "Vmax=100, Km=5"})


def test_nlr_michaelis_menten_and_aicc():
    # R: nls(y ~ V*x/(K+x), start=list(V=100,K=5))
    res = run("nonlinear_regression", sample("enzyme_kinetics.csv"),
              {"x": "Substrate (mM)", "y": ["Velocity"], "model": "michaelis_menten", "compare": "aicc",
               "model2": "hill", "normality": True, "interp_y": "59.22", "interp_x": "5.910492005"})
    best = table(res, "Best-fit values")
    assert param(best, "Vmax")["value"] == approx(118.441760282, 1e-5)
    assert param(best, "Vmax")["se"] == approx(2.0772891183, 1e-4)
    assert param(best, "Km")["value"] == approx(5.910492005, 1e-5)
    assert param(best, "Km")["se"] == approx(0.3486102392, 1e-4)
    cmp_ = table(res, "Comparison of models")[0]
    assert cmp_["delta"] == approx(cmp_["a2"] - cmp_["a1"])
    assert cmp_["p1"] == approx(1 / (1 + math.exp(-0.5 * cmp_["delta"])))
    assert cmp_["p1"] + cmp_["p2"] == approx(1)
    # Y at X = Km is Vmax / 2, and interpolation is the inverse of prediction
    pred = table(res, "Predicted Y values")[0]
    assert pred["y"] == approx(118.441760282 / 2, 1e-5) and pred["lo"] < pred["y"] < pred["hi"]
    assert table(res, "Interpolated X values")[0]["x"] == approx(5.9104, 1e-3)
    norm = table(res, "Normality of residuals")
    assert {r["test"].split(" ")[0] for r in norm} == {"Shapiro–Wilk", "D'Agostino–Pearson"}


def test_nlr_shared_curve_f_test():
    from analysis_ui.core.curvefit import fit_model
    d = frame("dose_response.csv")
    res = run("nonlinear_regression", sample("dose_response.csv"), {**NLR, "compare": "shared"})
    rows = table(res, "Do the data sets share one curve?")
    ss_sep = sum(fit_model("4pl", g[X_DR], g[Y_DR]).ss_res for _, g in d.groupby("Compound"))
    ss_glob = fit_model("4pl", d[X_DR], d[Y_DR]).ss_res
    F = ((ss_glob - ss_sep) / 4) / (ss_sep / 52)
    assert rows[0]["ss"] == approx(ss_glob) and rows[0]["df"] == 56
    assert rows[1]["ss"] == approx(ss_sep) and rows[1]["dfn"] == 4 and rows[1]["dfd"] == 52
    assert rows[1]["F"] == approx(F) and rows[1]["p"] == approx(stats.f.sf(F, 4, 52), 1e-6, 1e-300)
    assert "separate curves fit significantly better" in summary(res)
    # with a single data set the comparison is skipped with a warning, not an error
    res = run("nonlinear_regression", sample("enzyme_kinetics.csv"),
              {"x": "Substrate (mM)", "y": ["Velocity"], "model": "michaelis_menten", "compare": "shared"})
    assert any("at least two" in w for w in texts(res, "warning"))


def test_nlr_linear_matches_scipy_and_derived_values():
    rng = np.random.default_rng(3)
    x = np.arange(1.0, 21)
    y = 3 + 2 * x + rng.normal(0, 1, 20)
    lr = stats.linregress(x, y)
    res = run("nonlinear_regression", wire(x=x, y=y), {"x": "x", "y": ["y"], "model": "linear"})
    best = table(res, "Best-fit values")
    assert param(best, "Slope")["value"] == approx(lr.slope) and param(best, "Slope")["se"] == approx(lr.stderr)
    assert param(best, "Intercept")["se"] == approx(lr.intercept_stderr)
    assert param(table(res, "Goodness of fit"), 20, key="n")["r2"] == approx(lr.rvalue ** 2)
    assert param(table(res, "Derived values"), "X intercept (−Intercept / Slope)")["value"] == approx(-lr.intercept / lr.slope)
    assert "x_scale" not in graph(res)["options"]

    t = np.linspace(0, 10, 30)
    yd = 80 * np.exp(-0.5 * t) + 20 + rng.normal(0, 0.5, 30)
    res = run("nonlinear_regression", wire(t=t, y=yd), {"x": "t", "y": ["y"], "model": "exp_decay"})
    k = param(table(res, "Best-fit values"), "K")
    assert k["value"] == approx(0.5, 0.05)
    half = param(table(res, "Derived values"), "Half-life (ln 2 / K)")
    assert half["value"] == approx(math.log(2) / k["value"])
    assert half["lo"] == approx(math.log(2) / k["hi"]) and half["hi"] == approx(math.log(2) / k["lo"])
    assert param(table(res, "Derived values"), "Tau (1 / K)")["value"] == approx(1 / k["value"])


def test_nlr_multiple_y_columns_means_and_missing_cells():
    rng = np.random.default_rng(0)
    x = np.repeat([0.01, 0.1, 1, 10, 100, 1000], 3).astype(float)
    y1 = 100 / (1 + (5 / x)) + rng.normal(0, 2, len(x))
    y2 = 100 / (1 + (50 / x)) + rng.normal(0, 2, len(x))
    y2[4] = np.nan
    x2 = x.copy()
    x2[0] = np.nan
    data = wire(**{"Dose (nM)": x2, "Drug α": y1, "Drug β / ctrl": y2})
    opts = {"x": "Dose (nM)", "y": ["Drug α", "Drug β / ctrl"]}
    res = run("nonlinear_regression", data, opts)
    gof = table(res, "Goodness of fit")
    assert param(gof, "Drug α", key="dataset")["n"] == 17 and param(gof, "Drug β / ctrl", key="dataset")["n"] == 16
    assert any("missing" in w for w in texts(res, "warning"))
    assert param(table(res, "Best-fit values"), "EC50", "Drug α")["value"] == approx(5, 0.3)
    res = run("nonlinear_regression", data, {**opts, "replicates": "means"})
    assert param(table(res, "Goodness of fit"), "Drug α", key="dataset")["n"] == 6
    assert not has_table(res, "Replicates test")
    msg = fail("nonlinear_regression", wire(x=x, a=y1, b=y2, g=["u", "v"] * 9), {"x": "x", "y": ["a", "b"], "group": "g"})
    assert "not both" in msg


def test_nlr_user_errors_and_partial_failures():
    data = sample("dose_response.csv")
    assert "choose" in fail("nonlinear_regression", data, {"y": [Y_DR]}).lower()
    assert "at least 1" in fail("nonlinear_regression", data, {"x": X_DR})
    assert "different" in fail("nonlinear_regression", data, {"x": X_DR, "y": [X_DR]})
    # too few points
    msg = fail("nonlinear_regression", wire(x=[1.0, 2, 3], y=[1.0, 2, 4]), {"x": "x", "y": ["y"]})
    assert "needs at least 5 points" in msg
    # all X identical
    msg = fail("nonlinear_regression", wire(x=[2.0] * 8, y=np.arange(8.0)), {"x": "x", "y": ["y"], "model": "linear"})
    assert "identical" in msg
    # no complete rows
    msg = fail("nonlinear_regression", wire(x=[1.0, None, 3], y=[None, 2.0, None]), {"x": "x", "y": ["y"]})
    assert "no rows" in msg.lower()
    # X <= 0 with a model that needs positive X: points excluded with a warning
    x = np.array([0, 0, 0.1, 0.1, 1, 1, 10, 10, 100, 100, 1000, 1000.0])
    y = 100 / (1 + 10 / np.maximum(x, 1e-9)) + np.tile([1.0, -1.0], 6)
    res = run("nonlinear_regression", wire(x=x, y=y), {"x": "x", "y": ["y"]})
    assert any("X ≤ 0" in w for w in texts(res, "warning"))
    # one data set too small: the other is still reported
    g = ["big"] * 12 + ["tiny"] * 3
    res = run("nonlinear_regression", wire(x=np.concatenate([x, [1.0, 10, 100]]), y=np.concatenate([y, [5.0, 50, 90]]), g=g),
              {"x": "x", "y": ["y"], "group": "g"})
    assert any("tiny" in w and "could not be fitted" in w for w in texts(res, "warning"))
    assert {r["dataset"] for r in table(res, "Best-fit values")} == {"big"}
    assert "1 data set(s) could not be fitted" in summary(res)


def test_nlr_fit_that_cannot_be_determined_is_flagged_not_silent():
    # a flat line carries no information about EC50 / slope
    x = np.repeat([0.01, 0.1, 1, 10, 100], 2).astype(float)
    res = run_analysis("nonlinear_regression", wire(x=x, y=np.full(10, 5.0)), {"x": "x", "y": ["y"]})
    assert res["status"] == "error" and "identical" in res["message"]
    # a 4PL through data that only cover the bottom plateau and the start of the rise:
    # whatever is reported must be finite or flagged
    y = np.array([1, 1.2, 0.9, 1.1, 1.0, 1.05, 1.3, 1.2, 2.0, 2.1])
    res = run_analysis("nonlinear_regression", wire(x=x, y=y), {"x": "x", "y": ["y"]})
    if res["status"] == "ok":
        ec50 = param(table(res["results"], "Best-fit values"), "EC50")
        wide = ec50["se"] is None or ec50["se"] > ec50["value"]
        assert not wide or any("ambiguous" in w or "uncertain" in w for w in texts(res["results"], "warning"))
    # noise with an exponential-growth model must not crash either
    rng = np.random.default_rng(1)
    res = run_analysis("nonlinear_regression", wire(x=np.arange(12.0), y=rng.normal(0, 1, 12)),
                       {"x": "x", "y": ["y"], "model": "exp_decay2"})
    assert res["status"] in ("ok", "error") and (res["status"] == "ok" or "Traceback" not in res["message"])


def test_runs_test_exact_probability():
    from analysis_ui.analyses.curve_fitting import runs_test
    # + + + - - - : 2 runs; P(R <= 2) = 2 / C(6,3) = 0.1
    out = runs_test(np.array([1, 2, 1, -1, -2, -1.0]))
    assert out["runs"] == 2 and out["p"] == approx(0.1)
    # alternating signs: the maximum number of runs -> P = 1
    assert runs_test(np.array([1, -1, 1, -1, 1, -1.0]))["p"] == approx(1.0)
    assert runs_test(np.array([1.0, 2, 3])) is None


# ================================================ interpolate standard curve
def test_standard_curve_linear_inverse_prediction():
    conc = np.array([0, 0, 1, 1, 2, 2, 4, 4, 8, 8.0])
    rng = np.random.default_rng(5)
    sig = 0.1 + 0.25 * conc + rng.normal(0, 0.02, 10)
    lr = stats.linregress(conc, sig)
    x = np.concatenate([conc, [np.nan, np.nan, np.nan]])
    y = np.concatenate([sig, [0.6, 1.5, 2.6]])
    ids = [f"S{i}" for i in range(10)] + ["Patient 1", "Patient 2", "Patient 3"]
    res = run("interpolate_standard_curve", wire(**{"Conc (ng/mL)": x, "OD 450": y, "ID": ids}),
              {"x": "Conc (ng/mL)", "y": "OD 450", "label": "ID", "model": "linear", "unknown_values": "1.0"})
    rows = table(res, "Interpolated values")
    assert [r["name"] for r in rows] == ["Patient 1", "Patient 2", "Patient 3", "Typed value 1"]
    for r, y0 in zip(rows, [0.6, 1.5, 2.6, 1.0]):
        assert r["x"] == approx((y0 - lr.intercept) / lr.slope, 1e-6)
    # Fieller-type interval from the confidence band: solve band(x) = y0 directly
    n, xbar, sxx = 10, conc.mean(), ((conc - conc.mean()) ** 2).sum()
    s = math.sqrt(((sig - lr.intercept - lr.slope * conc) ** 2).sum() / 8)
    tc = stats.t.ppf(0.975, 8)

    def upper(xv):
        return lr.intercept + lr.slope * xv + tc * s * math.sqrt(1 / n + (xv - xbar) ** 2 / sxx)

    assert upper(rows[0]["lo"]) == approx(0.6, 1e-5)          # increasing curve: upper band gives the lower limit
    assert rows[0]["lo"] < rows[0]["x"] < rows[0]["hi"]
    assert rows[0]["note"] == "" and "Extrapolated" in rows[2]["note"]
    assert "Standard curve" in plots(res)
    assert "3 fell within" in summary(res)


def test_standard_curve_4pl_unknown_column_and_flags():
    d = frame("dose_response.csv")
    a = d[d["Compound"] == "Compound A"]
    n = len(a)
    unknown = [50.0, 20.0, 150.0] + [None] * (n - 3)
    data = wire(conc=a[X_DR].to_numpy(), signal=a[Y_DR].to_numpy(), unk=[np.nan if v is None else v for v in unknown])
    res = run("interpolate_standard_curve", data, {"x": "conc", "y": "signal", "unknown_y": "unk"})
    rows = table(res, "Interpolated values")
    b, t, e, h = 3.3870489869, 98.3540467867, 0.1156072363, 1.1994260622     # R nls fit (see above)
    for r, y0 in zip(rows[:2], [50.0, 20.0]):
        assert r["x"] == approx(e / ((t - b) / (y0 - b) - 1) ** (1 / h), 1e-4)
        assert r["lo"] < r["x"] < r["hi"]
    assert rows[2]["x"] is None and "does not reach" in rows[2]["note"] and rows[2]["lo"] is None
    assert param(table(res, "Standard curve: best-fit values"), "EC50")["value"] == approx(e, 1e-5)
    # no unknowns: still a helpful result
    res = run("interpolate_standard_curve", data, {"x": "conc", "y": "signal"})
    assert "No unknowns" in summary(res)
    assert "no standards" in fail("interpolate_standard_curve", wire(x=[None, None], y=[1.0, 2.0]), {"x": "x", "y": "y"}).lower()
    assert "not a number" in fail("interpolate_standard_curve", data, {"x": "conc", "y": "signal", "unknown_values": "1, abc"})
    assert "needs at least" in fail("interpolate_standard_curve", wire(x=[1.0, 2, 3], y=[1.0, 2, 3]), {"x": "x", "y": "y"})


# ======================================================================= AUC
def test_auc_matches_trapezoid_and_gagnon_se():
    d = frame("enzyme_kinetics.csv")
    g = d.groupby("Substrate (mM)")["Velocity"].agg(["mean", "std", "count"])
    x, m = g.index.to_numpy(), g["mean"].to_numpy()
    area = float(np.sum((m[1:] + m[:-1]) / 2 * np.diff(x)))
    w = np.empty(len(x))
    w[0], w[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
    w[1:-1] = (x[2:] - x[:-2]) / 2
    se = math.sqrt(np.sum(w ** 2 * g["std"] ** 2 / g["count"]))
    assert np.sum(w * m) == approx(area)
    res = run("auc", sample("enzyme_kinetics.csv"), {"x": "Substrate (mM)", "y": ["Velocity"]})
    row = table(res, "Area under the curve")[0]
    assert row["total"] == approx(area) and row["se"] == approx(se)
    assert row["lo"] == approx(area - 1.96 * se) and row["hi"] == approx(area + 1.96 * se)
    assert row["n_peaks"] == 1 and row["n_x"] == 8
    peak = table(res, "Peaks")[0]
    assert peak["peak_x"] == 64 and peak["peak_y"] == approx(m.max()) and peak["area"] == approx(area)
    assert plots(res) == ["Area under the curve"]


def test_auc_baseline_peaks_and_filters():
    x = np.arange(0, 11.0)
    y = np.array([0, 2, 4, 2, 0, -1, 0, 1, 0.2, 0, 0.0])
    res = run("auc", wire(t=x, y=y), {"x": "t", "y": ["y"], "min_height_pct": 0})
    row = table(res, "Area under the curve")[0]
    assert row["total"] == approx(8 + 0.5 + 0.6 + 0.1) and row["n_peaks"] == 2 and row["se"] is None
    peaks = table(res, "Peaks")
    assert [p["peak_x"] for p in peaks] == [2, 7] and peaks[0]["x0"] == 0 and peaks[0]["x1"] == 4
    assert peaks[0]["area"] == approx(8) and peaks[1]["area"] == approx(1.2)
    assert sum(p["pct"] for p in peaks) == approx(1)
    # default 10 % filter removes nothing here; a 30 % filter drops the small peak (height 1 of range 5)
    res = run("auc", wire(t=x, y=y), {"x": "t", "y": ["y"], "min_height_pct": 30})
    row = table(res, "Area under the curve")[0]
    assert row["n_peaks"] == 1 and row["peak_area"] == approx(8) and row["total"] == approx(9.2)
    # negative peaks: net area
    res = run("auc", wire(t=x, y=y), {"x": "t", "y": ["y"], "min_height_pct": 0, "negative_peaks": True})
    assert table(res, "Area under the curve")[0]["total"] == approx(9.2 - 1.0)
    assert [p["area"] for p in table(res, "Peaks")] == [approx(8), approx(-1), approx(1.2)]
    # baseline typed: crossing points are interpolated
    res = run("auc", wire(t=x, y=y), {"x": "t", "y": ["y"], "baseline": "value", "baseline_value": 1, "min_height_pct": 0})
    row = table(res, "Area under the curve")[0]
    assert row["baseline"] == 1 and row["total"] == approx(0.5 * 3 * 3)      # triangle between x = 0.5 and 3.5
    p = table(res, "Peaks")[0]
    assert (p["x0"], p["x1"]) == (approx(0.5), approx(3.5))
    res = run("auc", wire(t=x, y=y + 3), {"x": "t", "y": ["y"], "baseline": "first", "min_height_pct": 0})
    assert table(res, "Area under the curve")[0]["total"] == approx(9.2)


def test_auc_groups_and_errors():
    x = np.tile(np.arange(5.0), 4)
    y = np.concatenate([[0, 1, 2, 1, 0], [0, 1.2, 2.2, 1.2, 0], [0, 2, 4, 2, 0], [0, 2, 4, 2, np.nan]])
    g = ["low dose"] * 10 + ["high dose"] * 10
    res = run("auc", wire(x=x, y=y, g=g), {"x": "x", "y": ["y"], "group": "g"})
    rows = table(res, "Area under the curve")
    assert param(rows, "low dose", key="dataset")["total"] == approx(4.3)
    assert param(rows, "high dose", key="dataset")["total"] == approx(8.0)
    assert param(rows, "low dose", key="dataset")["se"] > 0
    assert param(rows, "high dose", key="dataset")["se"] is None       # one X has a single replicate
    assert any("missing" in w for w in texts(res, "warning"))
    assert "two different X" in fail("auc", wire(x=[1.0, 1, 1], y=[1.0, 2, 3]), {"x": "x", "y": ["y"]})
    assert "choose" in fail("auc", wire(x=[1.0, 2], y=[1.0, 2]), {"y": ["y"]}).lower()


# ================================================================== survival
KM = {"time": "Months", "event": "Status", "group": "Arm"}


def test_km_matches_r_survival():
    res = run("kaplan_meier", sample("survival.csv"), {**KM, "times": "12, 24", "show_rmst": True, "tarone_ware": True})
    # R: survfit(Surv(Months,Status)~Arm, conf.type="log-log"); summary(..., rmean=27.7)
    summ = table(res, "Summary of survival")
    c, t = param(summ, "Control", key="group"), param(summ, "Treatment", key="group")
    assert (c["n"], c["events"], c["censored"]) == (40, 28, 12) and (t["n"], t["events"], t["censored"]) == (40, 25, 15)
    assert (c["median"], c["lo"], c["hi"]) == (9.8, 4.8, 16.4)
    assert (t["median"], t["lo"], t["hi"]) == (18.2, 10.0, 24.7)
    assert c["rmst"] == approx(12.22563688, 1e-6) and c["rmst_se"] == approx(1.603464691, 1e-6)
    assert t["rmst"] == approx(15.970384, 1e-6) and t["rmst_se"] == approx(1.6606794, 1e-6)
    # R: survdiff(Surv(Months,Status)~Arm)
    tests = table(res, "Comparison of survival curves")
    lr = param(tests, "Log-rank (Mantel–Cox)", key="test")
    assert lr["chi2"] == approx(3.113576918, 1e-7) and lr["df"] == 1
    assert lr["p"] == approx(stats.chi2.sf(3.113576918, 1), 1e-6)
    oe = table(res, "Observed and expected")
    assert oe[0]["e"] == approx(21.82568416, 1e-7) and oe[1]["e"] == approx(31.17431584, 1e-7)
    # summary(survfit, times = c(12, 24))
    sp = table(res, "Survival proportions at chosen times")
    ref = [(12, 0.41971925, 0.082316921, 0.258788036, 0.57258827), (2, 0.15898456, 0.085500904, 0.038306995, 0.35463522),
           (19, 0.55733333, 0.08111328, 0.38585977, 0.69846341), (9, 0.36786555, 0.08779943, 0.20260566, 0.53450192)]
    for row, (risk, s, se, lo, hi) in zip(sp, ref):
        assert row["at_risk"] == risk
        assert (row["surv"], row["se"], row["lo"], row["hi"]) == (approx(s, 1e-6), approx(se, 1e-6), approx(lo, 1e-5), approx(hi, 1e-5))
    # hazard ratios from O, E and V (V = 12.24385237 from survdiff)
    hr = table(res, "Hazard ratio")
    u, v = 28 - 21.82568416, 12.24385237
    mh = param(hr, "Mantel–Haenszel", key="method")
    assert mh["ratio"] == approx(math.exp(u / v), 1e-6)
    assert mh["lo"] == approx(math.exp(u / v - 1.959964 / math.sqrt(v)), 1e-6)
    assert mh["recip"] == approx(1 / mh["ratio"]) and mh["rlo"] == approx(1 / mh["hi"])
    lrk = param(hr, "Log-rank", key="method")
    assert lrk["ratio"] == approx((28 / 21.82568416) / (25 / 31.17431584), 1e-6)
    assert lrk["hi"] == approx(lrk["ratio"] * math.exp(1.959964 * math.sqrt(1 / 21.82568416 + 1 / 31.17431584)), 1e-6)
    med = table(res, "Ratio of median survival times")[0]
    assert med["ratio"] == approx(9.8 / 18.2)
    assert med["lo"] == approx(9.8 / 18.2 * math.exp(-1.959964 * math.sqrt(1 / 28 + 1 / 25)), 1e-6)
    assert "do not differ significantly" in summary(res) and "P = 0.0776" in summary(res)
    assert plots(res) == ["Kaplan–Meier survival curves"]
    assert graph(res) == {"type": "survival", "options": {"time": "Months", "event": "Status", "event_value": "1", "group": "Arm"}}


def test_km_weighted_tests_match_lifelines():
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    d = frame("survival.csv")
    a, b = d[d["Arm"] == "Control"], d[d["Arm"] == "Treatment"]
    res = run("kaplan_meier", sample("survival.csv"), {**KM, "tarone_ware": True})
    tests = table(res, "Comparison of survival curves")
    for label, w in (("Log-rank (Mantel–Cox)", None), ("Gehan–Breslow–Wilcoxon", "wilcoxon"), ("Tarone–Ware", "tarone-ware")):
        ref = logrank_test(a["Months"], b["Months"], a["Status"], b["Status"], weightings=w)
        row = param(tests, label, key="test")
        assert row["chi2"] == approx(ref.test_statistic, 1e-8) and row["p"] == approx(ref.p_value, 1e-7)

    # three groups, pairwise comparisons and trend
    rng = np.random.default_rng(11)
    n = 45
    grp = np.repeat(["Low", "Mid", "High"], n)
    time = np.concatenate([rng.exponential(s, n) for s in (8, 12, 20)]).round(2)
    event = (rng.random(3 * n) < 0.75).astype(int)
    data = {"columns": [{"name": "t", "type": "continuous", "values": time.tolist()},
                        {"name": "e", "type": "continuous", "values": event.tolist()},
                        {"name": "dose group", "type": "ordinal", "values": grp.tolist(), "levels": ["Low", "Mid", "High"]}]}
    res = run("kaplan_meier", data, {"time": "t", "event": "e", "group": "dose group", "p_adjust": "bonferroni"})
    ref = multivariate_logrank_test(time, grp, event)
    tests = table(res, "Comparison of survival curves")
    row = param(tests, "Log-rank (Mantel–Cox)", key="test")
    assert row["df"] == 2 and row["chi2"] == approx(ref.test_statistic, 1e-8) and row["p"] == approx(ref.p_value, 1e-7)
    ref_w = multivariate_logrank_test(time, grp, event, weightings="wilcoxon")
    assert param(tests, "Gehan–Breslow–Wilcoxon", key="test")["chi2"] == approx(ref_w.test_statistic, 1e-8)
    trend = param(tests, "Log-rank test for trend", key="test")
    assert trend["df"] == 1 and 0 < trend["chi2"] <= row["chi2"] + 1e-9
    pw = table(res, "Pairwise comparisons")
    assert [r["pair"] for r in pw] == ["Low vs. Mid", "Low vs. High", "Mid vs. High"]
    for r, (g1, g2) in zip(pw, [("Low", "Mid"), ("Low", "High"), ("Mid", "High")]):
        m1, m2 = grp == g1, grp == g2
        ref = logrank_test(time[m1], time[m2], event[m1], event[m2])
        assert r["p"] == approx(ref.p_value, 1e-7) and r["padj"] == approx(min(1, 3 * ref.p_value), 1e-7)
    assert not has_table(res, "Hazard ratio")


def test_km_trend_test_matches_r():
    # R: survdiff trend via scores is not built in; reference computed from the
    # score vector and covariance of survdiff: (s'U)^2 / (s'Vs) for s = 1, 2, 3
    from analysis_ui.analyses.survival import logrank_trend, weighted_logrank
    rng = np.random.default_rng(2)
    time = rng.exponential(10, 90).round(1) + 0.1
    event = rng.random(90) < 0.7
    g = np.repeat([0, 1, 2], 30)
    lr = weighted_logrank(time, event, g, 3)
    assert lr["U"].sum() == approx(0, abs_=1e-9) and lr["O"].sum() == approx(lr["E"].sum())
    s = np.array([1.0, 2, 3])
    assert logrank_trend(lr)["chi2"] == approx((s @ lr["U"]) ** 2 / (s @ lr["V"] @ s))
    # with two groups the trend test equals the log-rank test
    lr2 = weighted_logrank(time[:60], event[:60], g[:60], 2)
    assert logrank_trend(lr2)["chi2"] == approx(lr2["chi2"])


def test_km_single_group_categorical_event_and_missing_values():
    d = frame("survival.csv")
    status = ["Dead" if s == 1 else "Alive" for s in d["Status"]]
    months = d["Months"].tolist()
    months[0] = None
    status[1] = None
    data = {"columns": [{"name": "Time (months)", "type": "continuous", "values": months},
                        {"name": "Vital status", "type": "nominal", "values": status}]}
    res = run("kaplan_meier", data, {"time": "Time (months)", "event": "Vital status", "event_value": "Dead",
                                     "full_table": True})
    row = table(res, "Summary of survival")[0]
    assert row["group"] == "All subjects" and row["n"] == 78
    from lifelines import KaplanMeierFitter
    dd = d.iloc[2:]
    kmf = KaplanMeierFitter().fit(dd["Months"], dd["Status"])
    assert row["median"] == approx(kmf.median_survival_time_)
    full = table(res, "Survival proportions at each event time")
    ref = kmf.survival_function_["KM_estimate"]
    for r in full[:40]:
        assert r["surv"] == approx(ref.loc[r["time"]], 1e-9)
    assert not has_table(res, "Comparison of survival curves")
    assert any("ignored" in w for w in texts(res, "warning"))
    assert "Median survival is" in summary(res)
    # the default event value "1" falls back to a recognisable label
    run("kaplan_meier", data, {"time": "Time (months)", "event": "Vital status"})


def test_km_user_errors():
    data = sample("survival.csv")
    assert "choose" in fail("kaplan_meier", data, {"event": "Status"}).lower()
    assert "No row" in fail("kaplan_meier", data, {**KM, "event_value": "2"})
    assert "not a number" in fail("kaplan_meier", data, {**KM, "event_value": "dead"})
    assert "not a number" in fail("kaplan_meier", data, {**KM, "times": "12; x"})
    t = [5.0, 8, 12, 3, 9, 14]
    censored = {"columns": [{"name": "t", "type": "continuous", "values": t},
                            {"name": "e", "type": "nominal", "values": ["alive"] * 6, "levels": ["alive", "dead"]}]}
    assert "No events" in fail("kaplan_meier", censored, {"time": "t", "event": "e", "event_value": "dead"})
    msg = fail("kaplan_meier", wire(t=t, e=["alive", "lost"] * 3), {"time": "t", "event": "e"})
    assert "Event value" in msg
    assert "negative" in fail("kaplan_meier", wire(t=[-1.0, 2, 3], e=[1.0, 1, 0]), {"time": "t", "event": "e"})
    assert "no rows" in fail("kaplan_meier", wire(t=[None, None], e=[1.0, 1]), {"time": "t", "event": "e"}).lower()
    # a group without events and a tiny group: results, not a crash
    res = run("kaplan_meier", wire(t=[1.0, 2, 3, 4, 5, 6, 7], e=[1.0, 1, 1, 0, 0, 0, 0], g=list("aaabbbb")),
              {"time": "t", "event": "e", "group": "g"})
    assert param(table(res, "Summary of survival"), "b", key="group")["median"] is None
    # times beyond follow-up give empty cells
    res = run("kaplan_meier", data, {**KM, "times": "0, 1000"})
    sp = table(res, "Survival proportions at chosen times")
    assert sp[0]["surv"] == 1 and sp[1].get("surv") is None


def test_km_median_when_curve_is_flat_at_half():
    # S(t) = 0.5 exactly from t = 2 until the next event at t = 6 -> median (2 + 6) / 2, as R and Prism
    res = run("kaplan_meier", wire(t=[1.0, 2, 6, 8], e=[1.0, 1, 1, 1]), {"time": "t", "event": "e"})
    assert table(res, "Summary of survival")[0]["median"] == approx(4.0)


# ======================================================================= ROC
ROC = {"outcome": "Status", "positive": "Disease", "predictors": ["Marker A", "Marker B"]}


def test_roc_matches_proc_delong():
    from sklearn.metrics import roc_auc_score
    d = frame("diagnostic.csv")
    res = run("roc", sample("diagnostic.csv"), ROC)
    rows = table(res, "Area under the ROC curve")
    a, b = param(rows, "Marker A", key="name"), param(rows, "Marker B", key="name")
    assert a["auc"] == approx(roc_auc_score(d["Status"] == "Disease", d["Marker A"]), 1e-10)
    assert b["auc"] == approx(roc_auc_score(d["Status"] == "Disease", d["Marker B"]), 1e-10)
    # R: pROC::ci.auc(roc(Status, `Marker A`, levels=c("Healthy","Disease"), direction="<")); sqrt(var(roc))
    assert a["se"] == approx(0.04301075245, 1e-7)
    assert (a["lo"], a["hi"]) == (approx(0.7719227, 1e-6), approx(0.9405217, 1e-6))
    assert (b["lo"], b["hi"]) == (approx(0.5162098, 1e-6), approx(0.745568, 1e-6))
    assert (a["n_pos"], a["n_neg"]) == (45, 50)
    assert a["p"] == approx(2 * stats.norm.sf((a["auc"] - 0.5) / a["se"]), 1e-6, 1e-300)
    # R: roc.test(r1, r2, method="delong", paired=TRUE)
    cmp_ = table(res, "Comparison of ROC curves")[0]
    assert cmp_["z"] == approx(3.342493501, 1e-7) and cmp_["p"] == approx(0.0008302929965, 1e-6)
    assert cmp_["diff"] == approx(a["auc"] - b["auc"]) and cmp_["lo"] < cmp_["diff"] < cmp_["hi"]
    s = summary(res)
    assert "differ significantly" in s and "P < 0.0001" in s and "P = <" not in s
    assert plots(res) == ["ROC curves"]
    assert graph(res) == {"type": "roc", "options": ROC}


def test_roc_cutoffs_wilson_and_youden():
    from statsmodels.stats.proportion import proportion_confint
    d = frame("diagnostic.csv")
    pos, neg = d.loc[d["Status"] == "Disease", "Marker A"], d.loc[d["Status"] == "Healthy", "Marker A"]
    res = run("roc", sample("diagnostic.csv"), {**ROC, "predictors": ["Marker A"], "max_rows": 1000})
    rows = table(res, "Sensitivity and specificity: Marker A")
    assert len(rows) == d["Marker A"].nunique() - 1
    for r in rows[::7]:
        c = r["cutoff"]
        assert r["sens"] == approx((pos > c).mean()) and r["spec"] == approx((neg <= c).mean())
        lo, hi = proportion_confint((pos > c).sum(), len(pos), method="wilson")
        assert (r["sens_lo"], r["sens_hi"]) == (approx(lo, 1e-6), approx(hi, 1e-6))
        if r["spec"] < 1:
            assert r["lr"] == approx(r["sens"] / (1 - r["spec"]))
        assert r["youden"] == approx(r["sens"] + r["spec"] - 1)
    best = [r for r in rows if r["mark"]]
    assert len(best) == 1 and best[0]["youden"] == approx(max(r["youden"] for r in rows))
    assert not has_table(res, "Comparison of ROC curves")
    # capped table keeps the optimum
    res = run("roc", sample("diagnostic.csv"), {**ROC, "predictors": ["Marker A"], "max_rows": 6})
    capped = table(res, "Sensitivity and specificity: Marker A")
    assert len(capped) <= 6 and any(r["mark"] for r in capped)


def test_roc_direction_hanley_and_three_markers():
    d = frame("diagnostic.csv")
    inv = (100 - d["Marker A"]).tolist()
    noise = np.random.default_rng(0).normal(size=len(d)).tolist()
    data = wire(Status=d["Status"].tolist(), **{"Marker A": d["Marker A"].tolist(), "Inverse": inv, "Noise": noise})
    opts = {"outcome": "Status", "positive": "Disease", "predictors": ["Marker A", "Inverse", "Noise"]}
    res = run("roc", data, {**opts, "se_method": "hanley"})
    rows = table(res, "Area under the ROC curve")
    a, i = param(rows, "Marker A", key="name"), param(rows, "Inverse", key="name")
    assert i["auc"] == approx(a["auc"]) and i["dir"] == "lower values" and a["dir"] == "higher values"
    auc, n1, n0 = a["auc"], 45, 50
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    assert a["se"] == approx(math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0)))
    cmp_ = table(res, "Comparison of ROC curves")
    assert len(cmp_) == 3
    same = param(cmp_, "Marker A vs. Inverse", key="pair")
    assert same["diff"] == approx(0, abs_=1e-12) and same["p"] is None      # identical curves: no variance
    # fixed direction keeps AUC below 0.5 instead of flipping
    res = run("roc", data, {**opts, "predictors": ["Inverse"], "direction": "higher"})
    assert table(res, "Area under the ROC curve")[0]["auc"] == approx(1 - a["auc"])
    # choosing the other positive level mirrors the AUC of a fixed-direction analysis
    res = run("roc", data, {**opts, "positive": "Healthy", "predictors": ["Marker A"], "direction": "higher"})
    assert table(res, "Area under the ROC curve")[0]["auc"] == approx(1 - a["auc"])


def test_roc_user_errors():
    d = frame("diagnostic.csv")
    three = ["A", "B", "C"] * 31 + ["A", "B"]
    msg = fail("roc", wire(out=three, m=d["Marker A"].tolist()), {"outcome": "out", "predictors": ["m"]})
    assert "exactly two" in msg and "3" in msg
    assert "at least 1" in fail("roc", sample("diagnostic.csv"), {"outcome": "Status"})
    assert "choose" in fail("roc", sample("diagnostic.csv"), {"predictors": ["Marker A"]}).lower()
    const = wire(out=["y", "n"] * 10, m=[3.0] * 20)
    assert "None of the markers" in fail("roc", const, {"outcome": "out", "predictors": ["m"]})
    tiny = wire(out=["y", "n", "n", "n"], m=[1.0, 2, 3, 4])
    assert "None of the markers" in fail("roc", tiny, {"outcome": "out", "predictors": ["m"]})
    # one bad marker does not hide the good one; NaNs are ignored
    vals = d["Marker A"].tolist()
    vals[3] = None
    res = run("roc", wire(Status=d["Status"].tolist(), good=vals, flat=[1.0] * len(d)),
              {"outcome": "Status", "predictors": ["good", "flat"]})
    assert len(table(res, "Area under the ROC curve")) == 1
    assert any("flat" in w for w in texts(res, "warning"))
    assert table(res, "Area under the ROC curve")[0]["n_neg"] == 49


def test_delong_against_brute_force():
    from analysis_ui.analyses.diagnostics import delong
    rng = np.random.default_rng(4)
    pos, neg = rng.normal(1, 1, (12, 2)).round(1), rng.normal(0, 1, (15, 2)).round(1)     # with ties
    auc, cov = delong(pos, neg)
    for j in range(2):
        psi = (pos[:, j][:, None] > neg[:, j][None, :]) + 0.5 * (pos[:, j][:, None] == neg[:, j][None, :])
        assert auc[j] == approx(psi.mean())
    psi = [(pos[:, j][:, None] > neg[:, j][None, :]) + 0.5 * (pos[:, j][:, None] == neg[:, j][None, :]) for j in range(2)]
    v10 = np.column_stack([p.mean(axis=1) for p in psi])
    v01 = np.column_stack([p.mean(axis=0) for p in psi])
    assert cov == approx(np.cov(v10, rowvar=False) / 12 + np.cov(v01, rowvar=False) / 15)


# ============================================================== Bland–Altman
def test_bland_altman_absolute():
    d = frame("method_comparison.csv")
    diff = d["Assay A"] - d["Assay B"]
    mean = (d["Assay A"] + d["Assay B"]) / 2
    n, bias, sd = len(diff), diff.mean(), diff.std(ddof=1)
    res = run("bland_altman", sample("method_comparison.csv"), {"method_a": "Assay A", "method_b": "Assay B"})
    rows = table(res, "Bland–Altman analysis")
    get = lambda label: next(r for r in rows if r["stat"].startswith(label))  # noqa: E731
    assert get("Number of pairs")["value"] == 45
    lo, hi = stats.t.interval(0.95, n - 1, loc=bias, scale=sd / math.sqrt(n))
    b = get("Bias")
    assert (b["value"], b["lo"], b["hi"]) == (approx(bias), approx(lo), approx(hi))
    assert get("SD of bias")["value"] == approx(sd)
    lower, upper = get("Lower 95% limit"), get("Upper 95% limit")
    assert lower["value"] == approx(bias - 1.96 * sd) and upper["value"] == approx(bias + 1.96 * sd)
    half = stats.t.ppf(0.975, n - 1) * sd * math.sqrt(3 / n)
    assert lower["lo"] == approx(bias - 1.96 * sd - half) and upper["hi"] == approx(bias + 1.96 * sd + half)
    lr = stats.linregress(mean, diff)
    slope = table(res, "Proportional bias")[0]
    assert (slope["est"], slope["se"], slope["p"]) == (approx(lr.slope), approx(lr.stderr), approx(lr.pvalue))
    s = summary(res)
    assert "lower than Assay B" in s and "proportional bias" in s
    assert plots(res) == ["Bland–Altman plot"]
    assert graph(res)["options"] == {"method_a": "Assay A", "method_b": "Assay B", "diff_type": "absolute"}


def test_bland_altman_percent_ratio_and_errors():
    d = frame("method_comparison.csv")
    a, b = d["Assay A"], d["Assay B"]
    res = run("bland_altman", sample("method_comparison.csv"),
              {"method_a": "Assay A", "method_b": "Assay B", "diff_type": "percent"})
    pct = 100 * (a - b) / ((a + b) / 2)
    rows = table(res, "Bland–Altman analysis")
    assert rows[1]["value"] == approx(pct.mean()) and rows[2]["value"] == approx(pct.std(ddof=1))
    assert graph(res)["options"]["diff_type"] == "percent"
    res = run("bland_altman", sample("method_comparison.csv"),
              {"method_a": "Assay A", "method_b": "Assay B", "diff_type": "ratio"})
    lg = np.log(a / b)
    rows = table(res, "Bland–Altman analysis")
    assert rows[1]["value"] == approx(math.exp(lg.mean()))
    assert rows[3]["value"] == approx(math.exp(lg.mean() - 1.96 * lg.std(ddof=1)))
    assert rows[1]["lo"] < rows[1]["value"] < rows[1]["hi"]
    assert "times Assay B" in summary(res)

    assert "different" in fail("bland_altman", sample("method_comparison.csv"), {"method_a": "Assay A", "method_b": "Assay A"})
    assert "choose" in fail("bland_altman", sample("method_comparison.csv"), {"method_a": "Assay A"}).lower()
    assert "At least 3" in fail("bland_altman", wire(a=[1.0, 2, None], b=[1.5, 2.5, 3]), {"method_a": "a", "method_b": "b"})
    assert "identical" in fail("bland_altman", wire(a=[1.0, 2, 3, 4], b=[2.0, 3, 4, 5]), {"method_a": "a", "method_b": "b"})
    assert "positive" in fail("bland_altman", wire(a=[1.0, -2, 3, 4], b=[2.0, 3, 5, 5]),
                              {"method_a": "a", "method_b": "b", "diff_type": "ratio"})
    res = run("bland_altman", wire(a=[1.0, 2, 3.5, 4, None], b=[2.0, 3.5, 4, 5.5, 1]), {"method_a": "a", "method_b": "b"})
    assert table(res, "Bland–Altman analysis")[0]["value"] == 4 and any("missing" in w for w in texts(res, "warning"))


# ======================================================================= ICC
# R: psych::ICC(x, lmer=FALSE)$results  -> (ICC, F, df1, df2, lower, upper)
ICC_REF_2 = {"ICC(1,1)": (0.97330418, 73.918094, 44, 45, 0.95215432, 0.98520543),
             "ICC(2,1)": (0.97335880, 87.331836, 44, 44, 0.94297649, 0.98654253),
             "ICC(3,1)": (0.97735811, 87.331836, 44, 44, 0.95917722, 0.98749362),
             "ICC(1,k)": (0.98647151, 73.918094, 44, 45, 0.97549083, 0.99254759),
             "ICC(2,k)": (0.98649957, 87.331836, 44, 44, 0.97065147, 0.99322568),
             "ICC(3,k)": (0.98854942, 87.331836, 44, 44, 0.97916330, 0.99370746)}
ICC_REF_3 = {"ICC(1,1)": (0.35115563, 2.6236049, 89, 180, 0.22058078, 0.48334973),
             "ICC(2,1)": (0.43335048, 7.9665635, 89, 178, 0.05554127, 0.68338802),
             "ICC(3,1)": (0.69899354, 7.9665635, 89, 178, 0.60578577, 0.77884591),
             "ICC(1,k)": (0.61884504, 2.6236049, 89, 180, 0.45917290, 0.73730085),
             "ICC(2,k)": (0.69644333, 7.9665635, 89, 178, 0.14996528, 0.86622647),
             "ICC(3,k)": (0.87447536, 7.9665635, 89, 178, 0.82174927, 0.91353372)}


@pytest.mark.parametrize("name,cols,ref", [("method_comparison.csv", ["Assay A", "Assay B"], ICC_REF_2),
                                           ("biomarker_panel.csv", ["IL-6", "TNF-α", "CRP"], ICC_REF_3)])
def test_icc_matches_psych(name, cols, ref):
    res = run("icc_agreement", sample(name), {"variables": cols, "show_anova": True})
    rows = table(res, "Intraclass correlation coefficients")
    assert [r["type"] for r in rows] == list(ref)
    for r in rows:
        icc, F, df1, df2, lo, hi = ref[r["type"]]
        assert (r["icc"], r["F"], r["df1"], r["df2"]) == (approx(icc, 1e-6), approx(F, 1e-6), df1, df2)
        assert (r["lo"], r["hi"]) == (approx(lo, 1e-5), approx(hi, 1e-5))
        assert r["p"] == approx(stats.f.sf(F, df1, df2), 1e-5, 1e-300)
    assert len(table(res, "Mean squares")) == 4
    assert has_table(res, "Lin's concordance") == (len(cols) == 2)
    assert "ICC(2,1)" in summary(res)


def test_lins_ccc_and_icc_errors():
    d = frame("method_comparison.csv")
    x, y = d["Assay A"].to_numpy(), d["Assay B"].to_numpy()
    ccc = 2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    res = run("icc_agreement", sample("method_comparison.csv"), {"variables": ["Assay A", "Assay B"], "focus": "icc3"})
    row = table(res, "Lin's concordance")[0]
    assert row["ccc"] == approx(ccc) and row["r"] == approx(stats.pearsonr(x, y)[0])
    assert row["cb"] == approx(ccc / row["r"]) and row["lo"] < ccc < row["hi"] < 1
    assert "ICC(3,1)" in summary(res)
    # perfect agreement shifted by a constant: consistency 1, absolute agreement < 1
    base = np.array([1.0, 3, 4, 7, 9, 12])
    res = run("icc_agreement", wire(r1=base, r2=base + 2), {"variables": ["r1", "r2"]})
    rows = {r["type"]: r for r in table(res, "Intraclass correlation coefficients")}
    assert rows["ICC(3,1)"]["icc"] == approx(1.0) and rows["ICC(2,1)"]["icc"] < 0.95
    assert "at least 2" in fail("icc_agreement", sample("method_comparison.csv"), {"variables": ["Assay A"]})
    assert "At least 3 subjects" in fail("icc_agreement", wire(a=[1.0, 2, None], b=[1.0, None, 3]), {"variables": ["a", "b"]})
    assert "identical" in fail("icc_agreement", wire(a=[2.0] * 5, b=[2.0] * 5), {"variables": ["a", "b"]})
    assert "do not differ" in fail("icc_agreement", wire(a=[1.0] * 5, b=[2.0] * 5), {"variables": ["a", "b"]})
    res = run("icc_agreement", wire(a=[1.0, 2, 3, 4, None, 6], b=[1.1, 2.3, 2.9, 4.2, 5, 6.1]), {"variables": ["a", "b"]})
    assert any("missing" in w for w in texts(res, "warning"))


# ===================================================================== kappa
K1 = ["a"] * 20 + ["b"] * 15 + ["c"] * 10 + ["a"] * 5 + ["b"] * 6 + ["c"] * 4
K2 = ["a"] * 20 + ["b"] * 15 + ["c"] * 10 + ["b"] * 5 + ["c"] * 6 + ["a"] * 4


def test_kappa_matches_psych_and_sklearn():
    from sklearn.metrics import cohen_kappa_score
    data = wire(**{"Rater 1": K1, "Rater 2": K2})
    opts = {"rater_a": "Rater 1", "rater_b": "Rater 2"}
    res = run("cohen_kappa", data, opts)
    row = table(res, "Cohen's kappa")[0]
    # R: psych::cohen.kappa(table(k1, k2))$confid
    assert row["kappa"] == approx(0.6179966044, 1e-8) == approx(cohen_kappa_score(K1, K2))
    assert (row["lo"], row["hi"]) == (approx(0.4524778502, 1e-6), approx(0.7835153586, 1e-6))
    assert row["p"] < 1e-6
    tab = table(res, "Agreement table")
    assert tab[0]["c0"] == 20 and tab[0]["c1"] == 5 and tab[2]["c0"] == 4 and tab[3]["total"] == 60
    assert "75.0%" in summary(res) and "substantial" in summary(res)
    res = run("cohen_kappa", data, {**opts, "weights": "quadratic"})
    rows = table(res, "Cohen's kappa")
    assert len(rows) == 2 and rows[1]["kappa"] == approx(0.6450482033, 1e-8) == approx(cohen_kappa_score(K1, K2, weights="quadratic"))
    assert (rows[1]["lo"], rows[1]["hi"]) == (approx(0.4421798977, 1e-5), approx(0.8479165090, 1e-5))
    res = run("cohen_kappa", data, {**opts, "weights": "linear"})
    assert table(res, "Cohen's kappa")[1]["kappa"] == approx(cohen_kappa_score(K1, K2, weights="linear"))


def test_kappa_edge_cases():
    # numeric ratings, a category used by one rater only, and missing cells
    a = [1.0, 2, 3, 1, 2, 3, 1, 2, None, 3]
    b = [1.0, 2, 3, 1, 2, 2, 1, 4, 2, None]
    from sklearn.metrics import cohen_kappa_score
    res = run("cohen_kappa", wire(a=a, b=b), {"rater_a": "a", "rater_b": "b"})
    assert table(res, "Cohen's kappa")[0]["kappa"] == approx(cohen_kappa_score([int(v) for v in a[:8]], [int(v) for v in b[:8]]))
    assert len(table(res, "Agreement table")) == 5 and any("missing" in w for w in texts(res, "warning"))
    assert "single category" in fail("cohen_kappa", wire(a=["x"] * 5, b=["x"] * 5), {"rater_a": "a", "rater_b": "b"})
    assert "different" in fail("cohen_kappa", wire(a=K1, b=K2), {"rater_a": "a", "rater_b": "a"})
    assert "choose" in fail("cohen_kappa", wire(a=K1, b=K2), {"rater_a": "a"}).lower()
    assert "At least two" in fail("cohen_kappa", wire(a=["x", None], b=[None, "y"]), {"rater_a": "a", "rater_b": "b"})
    # complete disagreement is a valid (negative) kappa
    res = run("cohen_kappa", wire(a=["x", "y"] * 5, b=["y", "x"] * 5), {"rater_a": "a", "rater_b": "b"})
    assert table(res, "Cohen's kappa")[0]["kappa"] == approx(-1.0)


# ======================================================================= PCA
def test_pca_matches_sklearn():
    from sklearn.decomposition import PCA
    d = frame("biomarker_panel.csv")
    x = d[PANEL].to_numpy()
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    ref = PCA().fit(z)
    res = run("pca", sample("biomarker_panel.csv"), {"variables": PANEL, "color": "Subtype", "select": "fixed",
                                                     "n_components": 6, "show_vectors": True, "show_scores": True})
    eig = table(res, "Eigenvalues")
    assert [r["eig"] for r in eig] == approx(ref.explained_variance_, 1e-9)
    assert [r["pct"] for r in eig] == approx(ref.explained_variance_ratio_, 1e-9)
    assert eig[-1]["cum"] == approx(1) and sum(r["eig"] for r in eig) == approx(6)
    assert np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))[::-1] == approx([r["eig"] for r in eig], 1e-9)
    vec = np.array([[r[f"pc{i}"] for i in range(6)] for r in table(res, "Eigenvectors")])
    assert np.abs(vec) == approx(np.abs(ref.components_.T), 1e-7, 1e-9)
    assert (np.abs(vec).max(axis=0) == vec.max(axis=0)).all()               # sign convention
    load = np.array([[r[f"pc{i}"] for i in range(6)] for r in table(res, "Loadings")])
    assert load == approx(vec * np.sqrt(ref.explained_variance_), 1e-9)
    scores = np.array([[r[f"pc{i}"] for i in range(6)] for r in table(res, "PC scores")])
    assert np.abs(scores) == approx(np.abs(ref.transform(z)), 1e-7, 1e-9)
    # a loading is the correlation between variable and component score
    assert load[2, 0] == approx(np.corrcoef(z[:, 2], scores[:, 0])[0, 1], 1e-9)
    assert plots(res) == ["Scree plot", "Score plot", "Loading plot"]
    assert "PC1 explains 45.0%" in summary(res)


def test_pca_component_selection_rules():
    data = sample("biomarker_panel.csv")
    res = run("pca", data, {"variables": PANEL})                    # parallel analysis by default
    eig = table(res, "Eigenvalues")
    assert [r["sel"] for r in eig] == ["Yes", "Yes", "", "", "", ""]
    thr = [r["thr"] for r in eig]
    assert thr == sorted(thr, reverse=True) and 1.2 < thr[0] < 1.8 and thr[-1] < 1
    assert eig[1]["eig"] > thr[1] and eig[2]["eig"] < thr[2]
    again = table(run("pca", data, {"variables": PANEL}), "Eigenvalues")
    assert [r["thr"] for r in again] == thr                         # reproducible (seeded)
    assert len(table(res, "Loadings")[0]) == 3                      # Variable + 2 retained PCs
    res = run("pca", data, {"variables": PANEL, "select": "kaiser"})
    assert sum(r["sel"] == "Yes" for r in table(res, "Eigenvalues")) == sum(r["eig"] > 1 for r in eig)
    res = run("pca", data, {"variables": PANEL, "select": "variance", "variance_pct": 90})
    sel = [r for r in table(res, "Eigenvalues") if r["sel"]]
    assert len(sel) == 3 and sel[-1]["cum"] >= 0.9 > sel[-2]["cum"]
    # covariance PCA
    x = frame("biomarker_panel.csv")[PANEL].to_numpy()
    res = run("pca", data, {"variables": PANEL, "scaling": "center", "select": "kaiser", "show_biplot": True})
    assert [r["eig"] for r in table(res, "Eigenvalues")] == approx(np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1], 1e-9)
    assert "Biplot" in plots(res)


def test_pca_missing_rows_and_errors():
    d = frame("biomarker_panel.csv")
    cols = {c: d[c].tolist() for c in PANEL[:3]}
    cols["IL-6"][0] = None
    cols["CRP"][5] = None
    cols["CRP"][0] = None
    res = run("pca", wire(**cols), {"variables": PANEL[:3]})
    assert any("2 row(s)" in w for w in texts(res, "warning"))
    assert len(table(res, "Eigenvalues")) == 3
    notes = next(e for e in res["elements"] if e["type"] == "table" and e["title"] == "Eigenvalues")["footnotes"]
    assert notes[0].startswith("88 rows, 3 variables")
    assert "at least 2" in fail("pca", sample("biomarker_panel.csv"), {"variables": ["IL-6"]})
    assert "at least 2" in fail("pca", sample("biomarker_panel.csv"), {"variables": ["IL-6", "IL-6"]}).lower()
    msg = fail("pca", wire(a=[1.0, 2, 3, 4], b=[5.0, 5, 5, 5]), {"variables": ["a", "b"]})
    assert "same value in every row" in msg and "“b”" in msg
    assert "complete rows" in fail("pca", wire(a=[1.0, None, 3, None], b=[None, 2.0, None, 4]), {"variables": ["a", "b"]})
    # more variables than rows: warning, not a crash
    rng = np.random.default_rng(0)
    res = run("pca", wire(**{f"v{i}": rng.normal(size=4) for i in range(6)}), {"variables": [f"v{i}" for i in range(6)]})
    assert any("unstable" in w for w in texts(res, "warning"))
    assert len(table(res, "Eigenvalues")) == 4


# ================================================================== k-means
def test_kmeans_matches_sklearn():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    d = frame("biomarker_panel.csv")
    x = d[PANEL].to_numpy()
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    ref = KMeans(n_clusters=3, n_init=25, random_state=1).fit(z)
    res = run("kmeans", sample("biomarker_panel.csv"), {"variables": PANEL, "k_mode": "fixed", "k": 3, "show_members": True})
    clusters = table(res, "Clusters")
    assert sorted(r["n"] for r in clusters) == sorted(np.bincount(ref.labels_).tolist())
    assert [r["n"] for r in clusters] == sorted((r["n"] for r in clusters), reverse=True)
    ss = table(res, "Sums of squares")
    assert ss[1]["ss"] == approx(ref.inertia_, 1e-6) and ss[2]["ss"] == approx((z ** 2).sum())
    assert ss[0]["ss"] + ss[1]["ss"] == approx(ss[2]["ss"])
    members = table(res, "Cluster membership")
    lab = np.array([r["cluster"] for r in members])
    assert len(lab) == 90
    assert np.mean([r["sil"] for r in members]) == approx(silhouette_score(z, ref.labels_), 1e-6)
    centres = table(res, "Cluster centres")
    assert centres[0]["c0"] == approx(x[lab == 1, 0].mean())
    assert plots(res) == ["Clusters on the first two principal components"]
    assert "3 clusters" in summary(res)


def test_kmeans_auto_k_and_errors():
    rng = np.random.default_rng(7)
    blobs = np.vstack([rng.normal(c, 0.4, (25, 2)) for c in ((0, 0), (6, 0), (0, 6), (6, 6))])
    data = wire(x=blobs[:, 0], y=blobs[:, 1])
    res = run("kmeans", data, {"variables": ["x", "y"]})
    choice = table(res, "Choice of the number of clusters")
    assert [r["k"] for r in choice if r["best"]] == [4]
    assert [r["n"] for r in table(res, "Clusters")] == [25, 25, 25, 25]
    assert len(plots(res)) == 2
    assert "at least 2" in fail("kmeans", data, {"variables": ["x"]})
    assert "identical" in fail("kmeans", wire(a=[1.0] * 6, b=[2.0] * 6), {"variables": ["a", "b"], "standardize": False})
    assert "same value" in fail("kmeans", wire(a=[1.0] * 6, b=[2.0] * 6), {"variables": ["a", "b"]})
    assert "only 3 distinct" in fail("kmeans", wire(a=[1.0, 2, 3, 3], b=[1.0, 5, 2, 2]),
                                     {"variables": ["a", "b"], "k_mode": "fixed", "k": 4})
    assert "complete rows" in fail("kmeans", wire(a=[1.0, None, 3], b=[1.0, 2, None]), {"variables": ["a", "b"]})
    # tiny data set with automatic k
    res = run("kmeans", wire(a=[1.0, 1.1, 5, 5.2], b=[1.0, 0.9, 5, 5.1]), {"variables": ["a", "b"]})
    assert [r["n"] for r in table(res, "Clusters")] == [2, 2]


# ==================================================== hierarchical clustering
def test_hclust_matches_scipy():
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    d = frame("biomarker_panel.csv")
    x = d[PANEL].to_numpy()
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    Z = linkage(z, method="ward")
    ref = fcluster(Z, 3, criterion="maxclust")
    res = run("hclust", sample("biomarker_panel.csv"), {"variables": PANEL, "k": 3, "label": "Subtype", "show_steps": True})
    sizes = [r["n"] for r in table(res, "Cluster sizes")]
    assert sorted(sizes) == sorted(np.bincount(ref)[1:].tolist())
    member = np.array([r["cluster"] for r in table(res, "Cluster membership")])
    # same partition up to relabelling
    assert len(set(zip(member.tolist(), ref.tolist()))) == 3
    info = {r["item"]: r["value"] for r in table(res, "Hierarchical clustering")}
    assert float(info["Cophenetic correlation"]) == approx(cophenet(Z, pdist(z))[0], abs_=6e-4)
    steps = table(res, "Agglomeration schedule")
    assert len(steps) == 20 and steps[-1]["h"] == approx(Z[-1, 2]) and steps[-1]["size"] == 90
    assert plots(res) == ["Dendrogram"]
    # without a cut there is no membership table
    res = run("hclust", sample("biomarker_panel.csv"), {"variables": PANEL, "linkage": "average", "distance": "manhattan"})
    assert not has_table(res, "Cluster sizes") and "average" in summary(res)


def test_hclust_columns_and_errors():
    from scipy.cluster.hierarchy import fcluster, linkage
    data = sample("biomarker_panel.csv")
    x = frame("biomarker_panel.csv")[PANEL].to_numpy()
    res = run("hclust", data, {"variables": PANEL, "what": "columns", "distance": "correlation", "linkage": "complete", "k": 2})
    member = {r["item"]: r["cluster"] for r in table(res, "Cluster membership")}
    ref = fcluster(linkage(x.T, method="complete", metric="correlation"), 2, criterion="maxclust")
    assert set(member) == set(PANEL)
    assert len({(member[v], int(c)) for v, c in zip(PANEL, ref)}) == 2
    assert "Ward" in fail("hclust", data, {"variables": PANEL, "distance": "correlation"})
    assert "at least 3" in fail("hclust", data, {"variables": PANEL[:2], "what": "columns"}).lower()
    assert "fewer" in fail("hclust", data, {"variables": PANEL, "what": "columns", "linkage": "average", "k": 6})
    assert "complete rows" in fail("hclust", wire(a=[1.0, None, 3], b=[1.0, 2, None]), {"variables": ["a", "b"]})
    assert "identical" in fail("hclust", wire(a=[1.0] * 5, b=[2.0] * 5), {"variables": ["a", "b"], "standardize": False})
    run("hclust", wire(a=[1.0, 2, 3, 5], b=[1.0, 2, 4, 6], c=[1.0, 2, 3, 4]),
        {"variables": ["a", "b", "c"], "standardize": False, "distance": "correlation", "linkage": "average"})
    # missing rows are dropped with a warning; many rows hide the labels but still plot
    rng = np.random.default_rng(0)
    big = rng.normal(size=(200, 3))
    big[0, 0] = np.nan
    res = run("hclust", wire(a=big[:, 0], b=big[:, 1], c=big[:, 2]), {"variables": ["a", "b", "c"], "k": 4})
    assert sum(r["n"] for r in table(res, "Cluster sizes")) == 199
    assert any("1 row(s)" in w for w in texts(res, "warning"))


# ================================================================== planning
def test_power_ttest_matches_r():
    # R: power.t.test(delta=5, sd=10, power=0.8)$n = 63.76576
    res = run("power_ttest", None, {})
    rows = table(res, "Sample size and power: t test")
    assert kv(rows, "Required sample size (group 1)") == 64 and kv(rows, "Total sample size") == 128
    assert "63.77" in next(r["text"] for r in rows if r["item"].startswith("Required sample size (group 1)"))
    assert kv(rows, "Effect size") == approx(0.5)
    assert "64 subjects per group" in summary(res) and plots(res) == ["Power curve"]
    # R: power.t.test(delta=5, sd=10, power=0.9, type="paired")$n = 43.99552
    rows = table(run("power_ttest", None, {"design": "paired", "power": 0.9}), "Sample size")
    assert kv(rows, "Required sample size (pairs)") == 44
    # R: power.t.test(delta=5, sd=10, power=0.8, alternative="one.sided", type="one.sample")$n = 26.13751
    rows = table(run("power_ttest", None, {"design": "one", "tails": "one"}), "Sample size")
    assert kv(rows, "Required sample size (subjects)") == 27
    # R: power.t.test(delta=5, sd=10, n=20)$power = 0.3377084 (R ignores the far tail; tolerance covers it)
    rows = table(run("power_ttest", None, {"solve_for": "power", "n": 20}), "Sample size")
    assert kv(rows, "Power") == approx(0.3377083656, 2e-3)
    # R: power.t.test(n=20, sd=10, power=0.8)$delta = 9.091306
    rows = table(run("power_ttest", None, {"solve_for": "effect", "n": 20}), "Sample size")
    assert kv(rows, "Difference in means") == approx(9.091305661, 1e-4)
    # R: pwr::pwr.t2n.test(n1=30, n2=60, d=0.5)$power = 0.5993611
    rows = table(run("power_ttest", None, {"solve_for": "power", "effect_mode": "d", "d": 0.5, "n": 30, "ratio": 2}), "Sample size")
    assert kv(rows, "Power") == approx(0.599361091, 1e-5) and kv(rows, "Sample size (group 2)") == 60
    # unequal allocation, solving for n: achieved power reaches the target
    rows = table(run("power_ttest", None, {"ratio": 2}), "Sample size")
    assert kv(rows, "Required sample size (group 2)") == 2 * kv(rows, "Required sample size (group 1)")
    assert 0.8 <= kv(rows, "Actual power") < 0.82
    # a negative difference is the same problem
    rows = table(run("power_ttest", None, {"diff": -5}), "Sample size")
    assert kv(rows, "Required sample size (group 1)") == 64


def test_power_ttest_errors():
    assert "zero" in fail("power_ttest", None, {"diff": 0})
    assert "standard deviation" in fail("power_ttest", None, {"sd": 0}).lower()
    assert "larger than the significance" in fail("power_ttest", None, {"power": 0.02, "alpha": 0.05})
    assert "too small" in fail("power_ttest", None, {"effect_mode": "d", "d": 1e-6})
    # empty data set object instead of None
    run("power_ttest", {"columns": []}, {})


def test_power_anova_matches_pwr():
    # R: pwr.anova.test(k=3, f=0.25, power=0.8)$n = 52.3966
    res = run("power_anova", None, {})
    rows = table(res, "Sample size and power: one-way ANOVA")
    assert kv(rows, "Required sample size per group") == 53 and kv(rows, "Total sample size") == 159
    assert "52.40" in next(r["text"] for r in rows if r["item"].startswith("Required sample size"))
    # R: pwr.anova.test(k=4, f=0.3, n=20)$power = 0.5785956
    rows = table(run("power_anova", None, {"solve_for": "power", "k": 4, "f": 0.3, "n": 20}), "Sample size")
    assert kv(rows, "Power") == approx(0.5785956356, 1e-6)
    from statsmodels.stats.power import FTestAnovaPower
    assert kv(rows, "Power") == approx(FTestAnovaPower().power(0.3, 80, 0.05, k_groups=4), 1e-6)
    # means + SD -> f = SD(means, population) / SD
    rows = table(run("power_anova", None, {"effect_mode": "means", "means": "10, 12, 15", "sd": 5}), "Sample size")
    assert kv(rows, "Effect size f") == approx(np.std([10, 12, 15]) / 5) and kv(rows, "Number of groups") == 3
    # detectable effect is the inverse of power
    rows = table(run("power_anova", None, {"solve_for": "effect", "k": 4, "n": 20, "power": 0.5785956356}), "Sample size")
    assert kv(rows, "Effect size f") == approx(0.3, 1e-5)
    assert "all equal" in fail("power_anova", None, {"effect_mode": "means", "means": "5, 5, 5"})
    assert "at least two" in fail("power_anova", None, {"effect_mode": "means", "means": "5"}).lower()
    assert "not a number" in fail("power_anova", None, {"effect_mode": "means", "means": "5, x"})
    assert "greater than zero" in fail("power_anova", None, {"f": 0})


def test_power_proportions_matches_r():
    res = run("power_proportions", None, {})
    rows = {r["key"]: r for r in table(res, "Required sample size")}
    # R: power.prop.test(p1=0.3, p2=0.5, power=0.8)$n = 92.99884 ; pwr.2p.test(h=ES.h(0.3,0.5), power=0.8)$n = 92.69608
    assert rows["pooled"]["exact"] == approx(92.99884482, 1e-6) and rows["pooled"]["n1"] == 93
    assert rows["arcsine"]["exact"] == approx(92.69608024, 1e-6)
    n = 92.99884482      # Fleiss continuity correction
    assert rows["cc"]["exact"] == approx(n / 4 * (1 + math.sqrt(1 + 4 / (n * 0.2))) ** 2, 1e-6)
    assert rows["cc"]["total"] == 206 and "93 subjects are needed in group 1" in summary(res)
    res = run("power_proportions", None, {"solve_for": "power", "n": 50})
    rows = {r["key"]: r for r in table(res, "Power")}
    # R: power.prop.test(p1=0.3, p2=0.5, n=50)$power = 0.5330844
    assert rows["pooled"]["power"] == approx(0.5330843962, 1e-6)
    assert rows["cc"]["power"] < rows["pooled"]["power"]
    # R: pwr.2p2n.test(h=ES.h(0.3,0.5), n1=50, n2=100)$power = 0.6612765
    res = run("power_proportions", None, {"solve_for": "power", "n": 50, "ratio": 2, "method": "arcsine"})
    rows = {r["key"]: r for r in table(res, "Power")}
    assert rows["arcsine"]["power"] == approx(0.6612765126, 1e-6)
    from statsmodels.stats.power import NormalIndPower
    h = 2 * math.asin(math.sqrt(0.3)) - 2 * math.asin(math.sqrt(0.5))
    assert rows["arcsine"]["power"] == approx(NormalIndPower().power(abs(h), 50, 0.05, ratio=2), 1e-6)
    assert "66.1%" in summary(res)
    # power at the computed n returns the target for every method (self-consistency, unequal groups)
    need = {r["key"]: r["exact"] for r in table(run("power_proportions", None, {"ratio": 1.5, "p1": 0.1, "p2": 0.25}), "Required")}
    for m, n1 in need.items():
        got = {r["key"]: r["power"] for r in table(run("power_proportions", None, {
            "solve_for": "power", "n": n1, "ratio": 1.5, "p1": 0.1, "p2": 0.25}), "Power")}
        assert got[m] == approx(0.8, 1e-4)
    assert "equal" in fail("power_proportions", None, {"p1": 0.4, "p2": 0.4})
    assert "between 0 and 1" in fail("power_proportions", None, {"p1": 0, "p2": 0.4})


def test_power_survival_schoenfeld():
    # textbook: HR 0.5, two-sided alpha 0.05, power 0.8, 1:1 -> 4 (1.95996 + 0.84162)^2 / ln(2)^2 = 65.35 -> 66 events
    res = run("power_survival", None, {"hr": 0.5, "event_prob": 0.6})
    rows = table(res, "Sample size and power: log-rank test")
    exact = 4 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) ** 2 / math.log(2) ** 2
    assert kv(rows, "Required number of events (Schoenfeld)") == 66 == math.ceil(exact)
    assert kv(rows, "Total subjects") == 2 * math.ceil(exact / 0.6 / 2)
    assert "66 events" in summary(res) and plots(res) == ["Power curve"]
    # medians 12 vs 24 months are the same problem (HR = 0.5 under exponential survival)
    rows = table(run("power_survival", None, {"effect_mode": "medians", "median1": 12, "median2": 24}), "Sample size")
    assert kv(rows, "Hazard ratio") == approx(0.5) and kv(rows, "Required number of events (Schoenfeld)") == 66
    # HR and 1/HR need the same number of events; unequal allocation needs more
    rows2 = table(run("power_survival", None, {"hr": 2.0}), "Sample size")
    assert kv(rows2, "Required number of events (Schoenfeld)") == 66
    rows3 = table(run("power_survival", None, {"hr": 0.5, "ratio": 2}), "Sample size")
    assert kv(rows3, "Required number of events (Schoenfeld)") == math.ceil(exact * 9 / 8)
    # power is the inverse
    rows = table(run("power_survival", None, {"solve_for": "power", "hr": 0.5, "events": exact}), "Sample size")
    assert kv(rows, "Power") == approx(0.8, 1e-9)
    assert "no difference" in fail("power_survival", None, {"hr": 1})
    assert "greater than zero" in fail("power_survival", None, {"effect_mode": "medians", "median1": 0})


def test_power_correlation():
    res = run("power_correlation", None, {})
    rows = table(res, "Sample size and power: correlation")
    exact = ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / math.atanh(0.3)) ** 2 + 3
    assert kv(rows, "Required sample size") == math.ceil(exact) == 85
    # R: pwr.r.test(r=0.3, power=0.8)$n = 84.07 (exact method) — the Fisher approximation is within one subject
    assert abs(exact - 84.07363978) < 1
    assert kv(rows, "Actual power") >= 0.8
    rows = table(run("power_correlation", None, {"solve_for": "power", "n": 85, "r": -0.3}), "Sample size")
    z = math.atanh(0.3) * math.sqrt(82)
    assert kv(rows, "Power") == approx(stats.norm.sf(1.959964 - z) + stats.norm.cdf(-1.959964 - z), 1e-6)
    assert "not zero" in fail("power_correlation", None, {"r": 0})
    rows = table(run("power_correlation", None, {"tails": "one", "r": 0.5}), "Sample size")
    assert kv(rows, "Required sample size") == math.ceil(((stats.norm.ppf(0.95) + stats.norm.ppf(0.8)) / math.atanh(0.5)) ** 2 + 3)


# =========================================================== cross-cutting
@pytest.mark.parametrize("analysis_id", ["nonlinear_regression", "interpolate_standard_curve", "auc", "kaplan_meier",
                                         "roc", "bland_altman", "icc_agreement", "cohen_kappa", "pca", "kmeans", "hclust"])
def test_no_variables_chosen_gives_a_friendly_error(analysis_id):
    for data in (sample("biomarker_panel.csv"), None):
        msg = fail(analysis_id, data, {})
        assert "choose" in msg.lower()


def test_plots_can_be_switched_off():
    off = {"show_plot": False, "show_residuals": False, "show_scree": False, "show_score_plot": False,
           "show_loading_plot": False, "show_silhouette_plot": False}
    cases = [("nonlinear_regression", "dose_response.csv", NLR), ("kaplan_meier", "survival.csv", KM),
             ("roc", "diagnostic.csv", ROC), ("pca", "biomarker_panel.csv", {"variables": PANEL}),
             ("kmeans", "biomarker_panel.csv", {"variables": PANEL}), ("hclust", "biomarker_panel.csv", {"variables": PANEL}),
             ("bland_altman", "method_comparison.csv", {"method_a": "Assay A", "method_b": "Assay B"}),
             ("auc", "enzyme_kinetics.csv", {"x": "Substrate (mM)", "y": ["Velocity"]}),
             ("power_ttest", None, {}), ("power_survival", None, {})]
    for aid, f, opts in cases:
        res = run(aid, sample(f) if f else None, {**opts, **off})
        assert plots(res) == [], aid
        summary(res)
