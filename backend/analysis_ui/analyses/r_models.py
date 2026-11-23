"""Regression models, survival regression and multivariate analyses computed
in R: multiple linear / logistic / Poisson regression, Cox regression,
reliability analysis and exploratory factor analysis.

See ``r_anova.py`` for the data hand-over convention (safe column names,
``options["_names"]``, ``options["_dropped"]``).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..core import AnalysisError, Context, opt, r_analysis
from .r_anova import check_factor, check_numeric, model_frame, with_private

REG = "Correlation & regression"
MODEL = "Model"
CHECKS = "Assumption checks"
OUT = "Output"


# ---------------------------------------------------------------------------
# Predictors and custom model formulas
# ---------------------------------------------------------------------------

def _predictor_opts(min_note: str = "") -> list[dict]:
    return [
        opt.variables("covariates", "Continuous predictors", opt.CONTINUOUS, min=0),
        opt.variables("factors", "Categorical predictors", opt.ANY, min=0,
                      help="Each level is compared with the first (reference) level."),
        opt.choice("interactions", "Interactions", [
            ("none", "None (main effects only)"),
            ("twoway", "All two-way interactions"),
            ("custom", "Custom model…")], section=MODEL),
        opt.text("formula", "Custom model terms", "", section=MODEL, visible={"interactions": "custom"},
                 placeholder="`Age` + `Sex` + `Age`:`Sex`",
                 help="Right-hand side of the model. Put column names in backticks; use + to add a term, "
                      ": for an interaction, * for main effects plus interaction, and I(`x`^2), log(`x`), "
                      "sqrt(`x`) or poly(`x`, 2) for transformations."),
    ]


_FORMULA_FUNCS = {"I", "log", "log2", "log10", "sqrt", "exp", "poly", "abs"}
_TOKEN = re.compile(r"\s+|`([^`]+)`|(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z_.][A-Za-z0-9_.]*)|([-+*:^()/,])")


def translate_formula(text: str, safe_by_name: dict[str, str]) -> str:
    """Translate the user's right-hand side into one that uses the safe column
    names. Only predictor names, numbers, the operators ``+ - * : ^ / ( ) ,``
    and a short list of functions are accepted, so arbitrary R code can never
    be injected."""
    text = (text or "").strip()
    if "~" in text:
        text = text.split("~", 1)[1].strip()
    if not text:
        raise AnalysisError("Type the model terms in the “Custom model terms” box, for example "
                            "`Age` + `Sex` + `Age`:`Sex`, or choose another interaction setting.")
    available = ", ".join(f"`{n}`" for n in safe_by_name) or "none selected"
    out: list[str] = []
    used = False
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise AnalysisError(
                f"The custom model contains something that is not understood at “{text[pos:pos + 12]}”. "
                "Use column names in backticks combined with + : * and parentheses.")
        pos = m.end()
        quoted, number, ident, op = m.groups()
        if quoted is not None or ident is not None:
            name = quoted if quoted is not None else ident
            if name in safe_by_name:
                out.append(safe_by_name[name])
                used = True
            elif quoted is None and name in _FORMULA_FUNCS and text[pos:].lstrip().startswith("("):
                out.append(name)
            else:
                raise AnalysisError(
                    f"“{name}” in the custom model is not one of the selected predictors. "
                    f"Add it to the predictors first. Available: {available}.")
        elif number is not None:
            out.append(number)
        elif op is not None:
            out.append(op)
    if not used:
        raise AnalysisError("The custom model does not contain any of the selected predictors.")
    rhs = " ".join(out)
    if rhs.count("(") != rhs.count(")"):
        raise AnalysisError("The parentheses in the custom model are not balanced.")
    return rhs


def _predictor_frame(ctx: Context, first: list[tuple[str, str, str]], extra: dict | None = None,
                     optional: list[tuple[str, str, str]] | None = None, need_predictor: bool = True):
    """Model frame with the outcome role(s) in ``first`` plus x1.. / f1..."""
    covs = ctx.vars("covariates", min=0)
    factors = ctx.vars("factors", min=0)
    if need_predictor and not covs and not factors:
        raise AnalysisError("Choose at least one predictor (continuous or categorical).")
    roles = (list(first) + [(f"x{i + 1}", c, "num") for i, c in enumerate(covs)]
             + [(f"f{i + 1}", c, "factor") for i, c in enumerate(factors)] + list(optional or []))
    df, names, dropped = model_frame(ctx, roles, extra=extra)
    if len(df) < 3:
        raise AnalysisError("Too few complete rows remain to fit a model. Check for missing values "
                            "in the selected variables.")
    cov_safe = [f"x{i + 1}" for i in range(len(covs))]
    fac_safe = [f"f{i + 1}" for i in range(len(factors))]
    for s, c in zip(cov_safe, covs):
        check_numeric(df, s, c, "predictor")
    for s, c in zip(fac_safe, factors):
        check_factor(df, s, c, what="categorical predictor")
    private = {"covariates": cov_safe, "factors": fac_safe}
    if ctx.opt("interactions", "none") == "custom":
        private["rhs"] = translate_formula(ctx.opt("formula", ""),
                                           {c: s for s, c in zip(cov_safe + fac_safe, covs + factors)})
    return df, names, dropped, private


# ---------------------------------------------------------------------------
# Multiple linear regression
# ---------------------------------------------------------------------------

def prepare_multiple_regression(ctx: Context):
    dep = ctx.var("dep")
    weights = ctx.var("weights", required=False)
    optional = [("w", weights, "num")] if weights else []
    df, names, dropped, private = _predictor_frame(ctx, [("y", dep, "num")], optional=optional)
    check_numeric(df, "y", dep, "outcome")
    df["row_id"] = df.attrs["rows"].astype(float)
    if weights:
        if (df["w"] < 0).any():
            raise AnalysisError(f"The weights in “{weights}” must not be negative.")
        n_zero = int((df["w"] == 0).sum())
        df = df[df["w"] > 0].reset_index(drop=True)
        dropped += n_zero
        if len(df) < 3:
            raise AnalysisError(f"Fewer than three rows have a positive weight in “{weights}”.")
    return df, with_private(ctx.options, names, dropped, weighted=bool(weights), **private)


r_analysis(
    id="multiple_regression", name="Multiple linear regression", category=REG,
    script="multiple_regression.R",
    description="Model a continuous outcome from several continuous and categorical predictors, "
                "with interactions, diagnostics and collinearity checks.",
    keywords="linear model lm ols predictors coefficients vif standardized beta residuals cook ancova",
    requires=["ggplot2", "car"], prepare=prepare_multiple_regression,
    options=[
        opt.variable("dep", "Dependent variable (outcome)", opt.CONTINUOUS),
        *_predictor_opts(),
        opt.boolean("intercept", "Include an intercept", True, section=MODEL),
        opt.variable("weights", "Weights (optional)", opt.CONTINUOUS, required=False, section=MODEL,
                     help="Weighted least squares: each row counts in proportion to its weight."),
        opt.choice("ss_type", "Sums of squares in the ANOVA table", [("2", "Type II"), ("3", "Type III")],
                   section=MODEL),
        opt.boolean("assumptions", "Normality and constant-variance tests", True, section=CHECKS),
        opt.boolean("diag_plots", "Diagnostic plots (residuals, QQ, scale–location, Cook's distance)", True,
                    section=CHECKS),
        opt.boolean("influence", "Table of influential points", False, section=CHECKS),
        opt.boolean("anova_table", "ANOVA table", True, section=OUT),
        opt.boolean("standardized", "Standardized coefficients (beta)", True, section=OUT),
        opt.boolean("vif", "Variance inflation factors (collinearity)", True, section=OUT),
        opt.boolean("show_plot", "Actual vs predicted plot", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------------

def prepare_logistic(ctx: Context):
    dep = ctx.var("dep")
    outcome = ctx.factor(dep)
    levels = [str(l) for l in outcome.dropna().cat.remove_unused_categories().cat.categories]
    if len(levels) != 2:
        raise AnalysisError(
            f"Logistic regression needs an outcome with exactly two values, but “{dep}” has "
            f"{len(levels)}" + (f" ({', '.join(levels[:6])}{'…' if len(levels) > 6 else ''})" if levels else "") + ".")
    positive = (ctx.opt("positive", "") or "").strip()
    if positive and positive not in levels:
        raise AnalysisError(f"“{positive}” is not a value of “{dep}” (values: {', '.join(levels)}).")
    if not positive:
        positive = levels[-1]
    negative = [l for l in levels if l != positive][0]
    y = np.where(outcome.isna().to_numpy(), np.nan, (outcome.astype(object) == positive).astype(float))
    df, names, dropped, private = _predictor_frame(ctx, [], extra={"y": y})
    names["y"] = dep
    if dep in ctx.vars("covariates", min=0) + ctx.vars("factors", min=0):
        raise AnalysisError(f"“{dep}” is the outcome and cannot also be a predictor.")
    n1 = int(df["y"].sum())
    if n1 == 0 or n1 == len(df):
        raise AnalysisError(
            f"After removing rows with missing values, every remaining row has the same outcome "
            f"(“{positive if n1 else negative}”), so nothing can be modelled.")
    return df, with_private(ctx.options, names, dropped, positive=positive, negative=negative, **private)


GLM_CI = [("profile", "Profile likelihood (recommended)"), ("wald", "Wald (estimate ± z × SE)")]

r_analysis(
    id="logistic_regression", name="Logistic regression", category=REG, script="logistic_regression.R",
    description="Model a yes/no outcome from continuous and categorical predictors: odds ratios, "
                "goodness of fit, classification table and ROC curve.",
    keywords="binary outcome odds ratio logit glm roc auc hosmer lemeshow classification",
    requires=["ggplot2", "pROC", "MASS"], prepare=prepare_logistic,
    options=[
        opt.variable("dep", "Outcome (two values)", opt.ANY),
        opt.level("positive", "Outcome value to predict", sources=["dep"],
                  help="The “event” or “positive” value. Empty = the last value (e.g. Yes, 1)."),
        *_predictor_opts(),
        opt.choice("ci_method", "Confidence intervals", GLM_CI, section=MODEL),
        opt.boolean("hosmer", "Hosmer–Lemeshow goodness-of-fit test", True, section=CHECKS),
        opt.boolean("classification", "Classification table", True, section=OUT),
        opt.number("cutoff", "Classification cut-off (predicted probability)", 0.5, min=0.01, max=0.99,
                   step=0.05, section=OUT),
        opt.boolean("roc", "ROC curve and AUC", True, section=OUT),
        opt.boolean("show_plot", "Predicted-probability plot (single continuous predictor)", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Poisson / negative-binomial regression
# ---------------------------------------------------------------------------

def prepare_poisson(ctx: Context):
    dep = ctx.var("dep")
    offset = ctx.var("offset", required=False)
    optional = [("expo", offset, "num")] if offset else []
    df, names, dropped, private = _predictor_frame(ctx, [("y", dep, "num")], optional=optional)
    if (df["y"] < 0).any():
        raise AnalysisError(f"“{dep}” contains negative values, but counts cannot be negative.")
    check_numeric(df, "y", dep, "outcome")
    if offset and (df["expo"] <= 0).any():
        raise AnalysisError(
            f"The exposure variable “{offset}” must be greater than zero in every row (its logarithm "
            "is used as the offset).")
    nonint = bool((np.abs(df["y"] - np.round(df["y"])) > 1e-8).any())
    return df, with_private(ctx.options, names, dropped, has_offset=bool(offset), noninteger=nonint, **private)


r_analysis(
    id="poisson_regression", name="Poisson / negative-binomial regression (counts)", category=REG,
    script="poisson_regression.R",
    description="Model counts or rates: incidence rate ratios, goodness of fit and a check for "
                "overdispersion.",
    keywords="count data rate incidence rate ratio glm overdispersion quasi-poisson negative binomial offset exposure",
    requires=["ggplot2", "MASS"], prepare=prepare_poisson,
    options=[
        opt.variable("dep", "Outcome (counts)", opt.CONTINUOUS),
        *_predictor_opts(),
        opt.variable("offset", "Exposure (optional)", opt.CONTINUOUS, required=False, section=MODEL,
                     help="Person-time, area, volume… Counts are modelled as rates per unit of exposure "
                          "(log exposure is used as an offset)."),
        opt.choice("family", "Distribution", [
            ("poisson", "Poisson"),
            ("quasipoisson", "Quasi-Poisson (allows overdispersion)"),
            ("negbin", "Negative binomial (allows overdispersion)")], section=MODEL),
        opt.choice("ci_method", "Confidence intervals", GLM_CI, section=MODEL),
        opt.boolean("show_plot", "Observed vs predicted counts", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Cox proportional hazards regression
# ---------------------------------------------------------------------------

def prepare_cox(ctx: Context):
    from ..graphs.types import event_indicator
    time = ctx.var("time")
    event = ctx.var("event")
    strata = ctx.var("strata", required=False)
    status = event_indicator(ctx, event, ctx.opt("event_value", "1"))
    optional = [("strata", strata, "factor")] if strata else []
    df, names, dropped, private = _predictor_frame(ctx, [("time", time, "num")], extra={"status": status},
                                                   optional=optional)
    names["status"] = event
    if (df["time"] < 0).any():
        raise AnalysisError(f"“{time}” contains negative times; follow-up times must be zero or positive.")
    n_events = int(df["status"].sum())
    if n_events == 0:
        raise AnalysisError(
            f"No events were found: no row of “{event}” equals the event value "
            f"“{ctx.opt('event_value', '1') or '1'}”. Check the “Event value” setting.")
    if n_events == len(df) and len(df) < 3:
        raise AnalysisError("Too few observations for a Cox model.")
    return df, with_private(ctx.options, names, dropped, has_strata=bool(strata), **private)


r_analysis(
    id="cox_regression", name="Cox proportional hazards regression", category="Survival",
    script="cox_regression.R",
    description="Model time-to-event data with several predictors: hazard ratios, global tests, "
                "concordance and a check of the proportional-hazards assumption.",
    keywords="survival hazard ratio coxph time to event censored proportional hazards schoenfeld forest plot",
    requires=["ggplot2", "survival"], prepare=prepare_cox,
    options=[
        opt.variable("time", "Time to event or censoring", opt.CONTINUOUS),
        opt.variable("event", "Event indicator", opt.ANY),
        opt.text("event_value", "Event value", "1", help="The value of the event column that means "
                 "“the event occurred”. Everything else is treated as censored."),
        *_predictor_opts(),
        opt.variable("strata", "Stratify by (optional)", opt.ANY, required=False, section=MODEL,
                     help="A separate baseline hazard is used in each stratum; no hazard ratio is "
                          "estimated for this variable."),
        opt.choice("ties", "Handling of tied event times", [("efron", "Efron (recommended)"),
                                                            ("breslow", "Breslow")], section=MODEL),
        opt.boolean("ph_test", "Test of proportional hazards", True, section=CHECKS),
        opt.boolean("forest", "Forest plot of hazard ratios", True, section=OUT),
        opt.boolean("curves", "Adjusted survival curves by the first categorical predictor", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Multivariate: reliability and factor analysis
# ---------------------------------------------------------------------------

def _items(ctx: Context, minimum: int, what: str):
    cols = ctx.vars("variables", min=minimum)
    if len(set(cols)) != len(cols):
        raise AnalysisError("A variable was selected more than once.")
    df, names, dropped = model_frame(ctx, [(f"x{i + 1}", c, "num") for i, c in enumerate(cols)])
    if len(df) < max(3, minimum):
        raise AnalysisError(f"Too few complete rows for a {what} (rows with a missing value in any "
                            "selected variable are excluded).")
    for safe, col in names.items():
        check_numeric(df, safe, col, "variable")
    return df, names, dropped, [f"x{i + 1}" for i in range(len(cols))]


def prepare_reliability(ctx: Context):
    df, names, dropped, items = _items(ctx, 2, "reliability analysis")
    return df, with_private(ctx.options, names, dropped, items=items)


def prepare_factor_analysis(ctx: Context):
    df, names, dropped, items = _items(ctx, 3, "factor analysis")
    if len(df) <= len(items):
        raise AnalysisError(
            f"Factor analysis needs more complete rows ({len(df)}) than variables ({len(items)}).")
    return df, with_private(ctx.options, names, dropped, items=items)


r_analysis(
    id="reliability", name="Reliability analysis (Cronbach's alpha)", category="Multivariate",
    script="reliability.R",
    description="Internal consistency of a scale: Cronbach's alpha with confidence interval, "
                "McDonald's omega and item statistics.",
    keywords="cronbach alpha internal consistency scale items questionnaire omega item-rest",
    requires=["ggplot2"], prepare=prepare_reliability,
    options=[
        opt.variables("variables", "Items", opt.CONTINUOUS, min=2),
        opt.boolean("reverse", "Reverse-score negatively related items automatically", False,
                    help="Items that correlate negatively with the rest of the scale are flipped "
                         "(new value = minimum + maximum − value)."),
        opt.boolean("omega", "McDonald's omega (total)", True, section=OUT),
        opt.boolean("item_stats", "Item statistics", True, section=OUT),
        opt.boolean("show_plot", "Inter-item correlation heat map", False, section=OUT),
        opt.alpha(section=OUT),
    ])

r_analysis(
    id="factor_analysis", name="Exploratory factor analysis", category="Multivariate",
    script="factor_analysis.R",
    description="Find the latent factors behind a set of correlated variables: loadings, "
                "variance explained, sampling adequacy and scree plot.",
    keywords="efa latent factors loadings rotation varimax promax oblimin kmo bartlett scree parallel analysis",
    requires=["ggplot2", "psych"], prepare=prepare_factor_analysis,
    options=[
        opt.variables("variables", "Variables", opt.CONTINUOUS, min=3),
        opt.choice("n_method", "Number of factors", [("parallel", "Parallel analysis (recommended)"),
                                                     ("kaiser", "Eigenvalues greater than 1"),
                                                     ("fixed", "Fixed number")]),
        opt.integer("n_factors", "Number of factors", 2, min=1, max=20, visible={"n_method": "fixed"}),
        opt.choice("extraction", "Extraction", [("ml", "Maximum likelihood"),
                                                ("minres", "Minimum residual (minres)")], section=MODEL),
        opt.choice("rotation", "Rotation", [("varimax", "Varimax (uncorrelated factors)"),
                                            ("promax", "Promax (correlated factors)"),
                                            ("oblimin", "Oblimin (correlated factors)"),
                                            ("none", "None")], section=MODEL),
        opt.number("hide_below", "Hide loadings smaller than", 0.3, min=0, max=0.9, step=0.05, section=OUT),
        opt.boolean("sort_loadings", "Sort variables by loading", True, section=OUT),
        opt.boolean("adequacy", "KMO and Bartlett's test", True, section=CHECKS),
        opt.boolean("show_plot", "Scree plot", True, section=OUT),
        opt.alpha(section=OUT),
    ])
