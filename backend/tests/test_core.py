"""Tests for the framework itself: data conversion, layouts, options, API."""
import io
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from analysis_ui import importer, transforms
from analysis_ui.app import app
from analysis_ui.core import AnalysisError, Context, opt, registry
from analysis_ui.core.curvefit import MODELS, compare_fits, fit_model
from analysis_ui.core.dataset import frame_to_wire, jsonable, to_frame
from analysis_ui.graphs import GRAPHS, build_payload
from analysis_ui.graphs import compute

client = TestClient(app)

WIDE = {"columns": [
    {"name": "Control", "type": "continuous", "values": [1, 2, 3, 4, None]},
    {"name": "Drug A", "type": "continuous", "values": [2, 4, 6, 8, 10]},
]}
LONG = {"columns": [
    {"name": "y", "type": "continuous", "values": [1, 2, 3, 4, 2, 4, 6, 8, 10, None]},
    {"name": "g", "type": "nominal", "values": ["b", "b", "b", "b", "a", "a", "a", "a", "a", "a"], "levels": ["b", "a"]},
    {"name": "id", "type": "nominal", "values": ["s1", "s2", "s3", "s4", "s1", "s2", "s3", "s4", "s5", "s6"]},
]}


def test_to_frame_types_and_levels():
    df = to_frame({"columns": [
        {"name": "x", "type": "continuous", "values": [1, "2.5", "abc", None, "1,000"]},
        {"name": "g", "type": "nominal", "values": ["b", "a", None, "", 3.0]},
        {"name": "o", "type": "ordinal", "values": ["lo", "hi", "lo", "hi", "lo"], "levels": ["lo", "hi"]},
    ]})
    assert df["x"].tolist()[:2] == [1.0, 2.5] and math.isnan(df["x"][2]) and df["x"][4] == 1000
    assert list(df["g"].cat.categories) == ["b", "a", "3"]          # order of appearance, 3.0 -> "3"
    assert df["g"].isna().sum() == 2
    assert df["o"].cat.ordered and list(df["o"].cat.categories) == ["lo", "hi"]


def test_numeric_levels_sort_numerically():
    df = to_frame({"columns": [{"name": "dose", "type": "nominal", "values": [10, 2, 1, 10]}]})
    assert list(df["dose"].cat.categories) == ["1", "2", "10"]


def test_duplicate_column_names_rejected():
    with pytest.raises(AnalysisError):
        to_frame({"columns": [{"name": "a", "type": "continuous", "values": [1]},
                              {"name": "a", "type": "continuous", "values": [2]}]})


def test_groups_wide_and_long_agree():
    w = Context(to_frame(WIDE), {"layout": "wide", "columns": ["Control", "Drug A"]}).groups()
    l = Context(to_frame(LONG), {"layout": "long", "dep": "y", "group": "g"}).groups()
    assert list(w) == ["Control", "Drug A"] and list(l) == ["b", "a"]     # UI level order kept
    assert w["Control"].tolist() == l["b"].tolist() == [1, 2, 3, 4]
    assert w["Drug A"].tolist() == l["a"].tolist() == [2, 4, 6, 8, 10]


def test_paired_layouts():
    w = Context(to_frame(WIDE), {"layout": "wide", "columns": ["Control", "Drug A"]}).paired()
    assert len(w) == 4 and list(w.columns) == ["Control", "Drug A"]
    l = Context(to_frame(LONG), {"layout": "long", "dep": "y", "group": "g", "subject": "id"}).paired()
    assert len(l) == 4 and l.loc["s2", "a"] == 4 and l.loc["s2", "b"] == 2
    with pytest.raises(AnalysisError):
        dup = json.loads(json.dumps(LONG))
        dup["columns"][2]["values"][1] = "s1"
        Context(to_frame(dup), {"layout": "long", "dep": "y", "group": "g", "subject": "id"}).paired()


def test_missing_variable_is_a_friendly_error():
    ctx = Context(to_frame(LONG), {"layout": "long", "dep": None, "group": "g"})
    ctx._labels = {"dep": "Values"}
    with pytest.raises(AnalysisError, match="Values"):
        ctx.long()


def test_clean_options_coerces_and_clamps():
    schema = [opt.number("a", "A", 0.05, min=0, max=1), opt.integer("n", "N", None, min=2),
              opt.choice("c", "C", [("x", "X"), ("y", "Y")]), opt.boolean("b", "B", True),
              opt.variables("v", "V"), opt.text("t", "T")]
    out = registry.clean_options(schema, {"a": "5", "n": "", "c": "zzz", "b": 0, "v": None, "t": None,
                                          "junk": 1, "_private": 2})
    assert out == {"a": 1, "n": None, "c": "x", "b": False, "v": [], "t": "", "_private": 2}


def test_jsonable_handles_numpy_and_nan():
    out = jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": np.array([1.5, np.inf]), "d": pd.NA})
    assert out == {"a": None, "b": 3, "c": [1.5, None], "d": None}
    json.dumps(out, allow_nan=False)


def test_every_registered_schema_is_well_formed():
    registry.discover()
    kinds = {"variable", "variables", "choice", "bool", "number", "integer", "text", "level", "color", "group_colors"}
    for d in list(registry.ANALYSES.values()) + list(GRAPHS.values()):
        names = {o["name"] for o in d.options}
        for o in d.options:
            assert o["type"] in kinds, (d.id, o)
            assert o["label"], (d.id, o)
            for dep in (o.get("visible") or {}):
                assert dep in names, f"{d.id}.{o['name']}: visible refers to unknown option {dep}"
            if o["type"] == "choice":
                assert o["default"] in [c["value"] for c in o["choices"]], (d.id, o["name"])
            if o["type"] == "level":
                assert all(s in names for s in o["sources"]), (d.id, o["name"])
        json.dumps(d.public())


# ------------------------------------------------------------- curve fitting
def test_4pl_recovers_parameters():
    x = np.repeat([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], 3)
    rng = np.random.default_rng(1)
    y = 5 + 95 / (1 + (0.2 / x) ** 1.3) + rng.normal(0, 1.5, len(x))
    fit = fit_model("4pl", x, y)
    assert fit.params["EC50"] == pytest.approx(0.2, rel=0.12)
    assert fit.params["HillSlope"] == pytest.approx(1.3, rel=0.12)
    assert fit.ci["EC50"][0] < 0.2 < fit.ci["EC50"][1]
    assert fit.r2 > 0.99
    assert fit.solve_x(52.5) == pytest.approx(0.2, rel=0.15)
    lo, hi = fit.band([0.2])
    assert lo[0] < fit.predict([0.2])[0] < hi[0]


def test_linear_fit_matches_ols_and_f_test():
    x = np.arange(20.0)
    y = 3 + 2 * x + np.random.default_rng(2).normal(0, 1, 20)
    fit = fit_model("linear", x, y)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit.params["Slope"] == pytest.approx(slope) and fit.params["Intercept"] == pytest.approx(intercept)
    cmp = compare_fits(fit, fit_model("quadratic", x, y))
    assert cmp["df1"] == 1 and 0 <= cmp["p"] <= 1


@pytest.mark.parametrize("model", list(MODELS))
def test_every_model_fits_its_own_curve(model):
    m = MODELS[model]
    x = np.linspace(0.5, 12, 40)
    truth = {"linear": [1, 2], "quadratic": [1, 2, .3], "cubic": [1, 2, .3, .02], "4pl": [2, 90, 3, 1.5],
             "3pl": [2, 90, 3], "5pl": [2, 90, 3, 1.5, 1.2], "4pl_log": [2, 90, 5, .8], "michaelis_menten": [100, 4],
             "hill": [100, 1.6, 4], "one_site_total": [80, 3, 1.5, 4], "exp_decay": [100, 10, .4],
             "exp_decay2": [5, 60, 1.5, 40, .15], "exp_assoc": [3, 90, .35], "exp_growth": [2, .3],
             "logistic_growth": [100, 3, .9], "gompertz": [100, 3, .5], "gaussian": [50, 6, 1.5],
             "power": [3, 1.4], "log_line": [4, 7]}[model]
    y = m.func(x, *truth)
    fit = fit_model(model, x, y + np.random.default_rng(3).normal(0, 0.01 * np.ptp(y), len(x)))
    assert fit.r2 > 0.98, (model, fit.params)


def test_fit_needs_enough_points():
    with pytest.raises(AnalysisError, match="at least"):
        fit_model("4pl", [1, 2, 3], [1, 2, 3])


# ------------------------------------------------------------ graph compute
def test_summarize_variants():
    v = [2, 4, 4, 4, 5, 5, 7, 9]
    s = compute.summarize(v, "mean", "sd")
    assert s["center"] == 5 and s["hi"] - s["center"] == pytest.approx(np.std(v, ddof=1))
    s = compute.summarize(v, "median", "iqr")
    assert s["center"] == 4.5 and (s["lo"], s["hi"]) == tuple(np.percentile(v, [25, 75]))
    assert compute.summarize([3], "mean", "sem") == {"n": 1, "center": 3.0, "lo": None, "hi": None}
    assert compute.summarize([], "mean", "sd")["n"] == 0


def test_brackets_and_adjustment():
    rng = np.random.default_rng(4)
    groups = {"a": rng.normal(0, 1, 12), "b": rng.normal(3, 1, 12), "c": rng.normal(0.2, 1, 12)}
    raw = compute.pairwise_brackets(groups, "welch", adjust="none")
    adj = compute.pairwise_brackets(groups, "welch", adjust="bonferroni")
    assert len(raw) == 3 and all(a["p"] >= r["p"] for a, r in zip(adj, raw))
    assert [(b["g1"], b["g2"]) for b in compute.pairwise_brackets(groups, "t", "control", control="b")] == [("b", "a"), ("b", "c")]
    assert [(b["g1"], b["g2"]) for b in compute.pairwise_brackets(groups, "t", "custom", custom="a vs c; B - A")] == [("a", "c"), ("b", "a")]
    assert all(b["label"] != "ns" for b in compute.pairwise_brackets(groups, "t", hide_ns=True))
    assert compute.pairwise_brackets(groups, "none") == []


def test_km_curve_against_lifelines():
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(5)
    t, e = rng.exponential(10, 60).round(1), rng.random(60) < 0.7
    km = compute.km_curve(t, e)
    ref = KaplanMeierFitter().fit(t, e)
    for tt, ss in list(zip(km["time"], km["surv"]))[1:]:      # [0] is the artificial start (0, 1)
        assert ss == pytest.approx(float(ref.predict(tt)), abs=1e-9)
    assert km["median"] == pytest.approx(ref.median_survival_time_)
    assert km["events"] == int(e.sum())


def test_every_graph_type_prepares_a_json_safe_payload():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import render_gallery as rg
    for case in rg.CASES:
        gid, sname, options = case[0], case[1], case[2]
        if options is None:
            continue
        payload, meta = build_payload({"type": gid, "options": options}, rg.sample(sname))
        json.dumps(jsonable(payload["computed"]), allow_nan=False)
        json.dumps(jsonable(payload["options"]), allow_nan=False)
        assert payload["frames"], gid
        assert payload["options"]["_theme"]["bg"] and payload["options"]["_fonts"]["family"]


def test_group_colors_override_palette():
    payload, meta = build_payload({"type": "column_bar", "options": {
        "layout": "wide", "columns": ["Control", "Drug A"], "group_colors": {"Drug A": "#123456"}, "palette": "npg"}}, WIDE)
    assert meta["groups"] == [{"name": "Control", "color": "#E64B35"}, {"name": "Drug A", "color": "#123456"}]


# ------------------------------------------------------------------ import
def test_read_text_detects_header_separator_and_types():
    sheet = importer.read_text("Group\tDose (mg)\tResp\nA\t1\t3.5\nB\t2\t\nA\t3\t5,5\n")[0]
    cols = {c["name"]: c for c in sheet["columns"]}
    assert cols["Group"]["type"] == "nominal" and cols["Dose (mg)"]["type"] == "continuous"
    assert cols["Dose (mg)"]["values"] == [1, 2, 3] and cols["Resp"]["values"][1] is None


def test_read_text_without_header():
    sheet = importer.read_text("1,2\n3,4\n")[0]
    assert [c["name"] for c in sheet["columns"]] == ["Column 1", "Column 2"]
    assert sheet["columns"][0]["values"] == [1, 3]


def test_read_excel_roundtrip():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xl:
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(xl, sheet_name="One", index=False)
        pd.DataFrame({"c": [1.5]}).to_excel(xl, sheet_name="Two", index=False)
    sheets = importer.read_file("book.xlsx", buf.getvalue())
    assert [s["name"] for s in sheets] == ["One", "Two"] and sheets[0]["columns"][1]["type"] == "nominal"


def test_unsupported_file_type():
    with pytest.raises(AnalysisError, match="not supported"):
        importer.read_file("x.docx", b"")


# -------------------------------------------------------------- transforms
def test_transforms():
    out = transforms.transform(WIDE, "log10", ["Drug A"])["columns"][0]
    assert out["name"] == "Drug A (log10)" and out["values"][4] == pytest.approx(1.0)
    out = transforms.transform(WIDE, "formula", [], formula="`Drug A` / Control", name="Ratio")["columns"][0]
    assert out["name"] == "Ratio" and out["values"][:4] == [2, 2, 2, 2] and out["values"][4] is None
    with pytest.raises(AnalysisError):
        transforms.transform(WIDE, "formula", [], formula="nonsense +")
    long = transforms.reshape(WIDE, "stack", ["Control", "Drug A"])
    assert [c["name"] for c in long["columns"]] == ["Group", "Value"] and len(long["columns"][1]["values"]) == 9
    wide = transforms.reshape(LONG, "unstack", [], id_column="g", value_column="y")
    assert [c["name"] for c in wide["columns"]] == ["b", "a"]


# --------------------------------------------------------------------- API
def test_api_catalog_and_samples():
    cat = client.get("/api/catalog").json()
    assert cat["engines"]["python"] is True and len(cat["graphs"]) >= 30 and cat["themes"] and cat["palettes"]
    samples = client.get("/api/samples").json()
    assert len(samples) >= 10
    for s in samples:
        got = client.get(f"/api/samples/{s['id']}").json()
        assert got["status"] == "ok" and got["sheets"][0]["columns"], s["id"]
    assert client.get("/api/samples/nope").status_code == 404


def test_api_errors_are_reported_not_raised():
    r = client.post("/api/analyses/run", json={"analysis": "does_not_exist", "options": {}, "data": WIDE}).json()
    assert r["status"] == "error"
    r = client.post("/api/graphs/render", json={"graph": {"type": "column_bar", "options": {"layout": "wide", "columns": []}}, "data": WIDE}).json()
    assert r["status"] == "error" and "Groups" in r["message"]
    r = client.post("/api/graphs/export", json={"graph": {"type": "nope"}, "data": WIDE, "format": "png"})
    assert r.status_code == 400
    r = client.post("/api/import/text", json={"text": "  "}).json()
    assert r["status"] == "error"
    r = client.post("/api/transform", json={"data": WIDE, "op": "zscore", "columns": ["Control"]}).json()
    assert r["status"] == "ok" and r["columns"][0]["name"] == "Control (z)"
