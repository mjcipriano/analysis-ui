"""ANOVA family computed in R: one-way / Kruskal–Wallis / repeated measures /
Friedman, factorial and mixed designs, linear mixed models, nested ANOVA.

The option schemas and the data preparation live here; the statistics are in
``r/analyses/*.R`` (shared helpers in ``r/lib/20_stats.R``).

Data hand-over convention: every ``prepare`` function gives R a data frame
with *safe* column names (``y``, ``f1``, ``x1``, …) and stores the user's
column names in ``options["_names"]`` (safe → display name) and the number of
incomplete rows that were removed in ``options["_dropped"]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import AnalysisError, Context, opt, r_analysis

CAT3 = "Compare three or more groups"
FACTORIAL = "Factorial & repeated measures"


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------

def model_frame(ctx: Context, roles: list[tuple[str, str, str]], dropna: bool = True,
                keep: list[str] | None = None, extra: dict | None = None):
    """Build the data frame handed to R.

    ``roles`` is a list of ``(safe_name, column, kind)`` with kind ``"num"`` or
    ``"factor"``. Returns ``(frame, names, dropped)`` where ``names`` maps safe
    names to the user's column names and ``dropped`` counts incomplete rows
    (rows that are empty in every selected column are ignored silently, they
    are just padding). Columns listed in ``keep`` may contain missing values;
    ``extra`` adds pre-computed columns (arrays in the original row order).
    """
    seen: dict[str, str] = {}
    cols: dict[str, pd.Series] = {}
    names: dict[str, str] = {}
    for safe, col, kind in roles:
        if col in seen:
            raise AnalysisError(
                f"The column “{col}” is used in more than one role. "
                "Each variable can be used only once in this analysis.")
        seen[col] = safe
        s = ctx.numeric(col) if kind == "num" else ctx.factor(col)
        cols[safe] = s.reset_index(drop=True)
        names[safe] = col
    df = pd.DataFrame(cols)
    for safe, values in (extra or {}).items():      # pre-computed columns (same row order)
        df[safe] = np.asarray(values)
    df = df[~df.isna().all(axis=1)]
    n0 = len(df)
    if dropna:
        need = [c for c in df.columns if c not in (keep or [])]
        df = df.dropna(subset=need)
    rows = df.index.to_numpy() + 1          # 1-based row numbers in the user's table
    df = df.reset_index(drop=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].cat.remove_unused_categories()
    df.attrs["rows"] = rows
    return df, names, n0 - len(df)


def check_factor(df: pd.DataFrame, safe: str, label: str, min_levels: int = 2,
                 what: str = "factor"):
    n_levels = len(df[safe].cat.categories)
    if n_levels < min_levels:
        raise AnalysisError(
            f"The {what} “{label}” has only {n_levels} level{'s' if n_levels != 1 else ''} "
            f"with data; at least {min_levels} are needed.")
    if n_levels > 2 and n_levels >= len(df):
        raise AnalysisError(
            f"“{label}” has a different value in every row, so it cannot be used as a {what}. "
            "If it is a measurement, use it as a covariate instead.")


def check_numeric(df: pd.DataFrame, safe: str, label: str, role: str = "covariate"):
    v = df[safe].to_numpy(dtype=float)
    if len(v) and np.nanmax(v) == np.nanmin(v):
        if role == "outcome":
            raise AnalysisError(
                f"All values of “{label}” are identical ({v[0]:g}), so there is no variation to analyse.")
        raise AnalysisError(
            f"The {role} “{label}” has the same value in every row, so its effect cannot be estimated. "
            "Remove it from the model.")


def with_private(options: dict, names: dict, dropped: int, **extra) -> dict:
    out = dict(options)
    out["_names"] = names
    out["_dropped"] = int(dropped)
    for k, v in extra.items():
        out["_" + k] = v
    return out


# ---------------------------------------------------------------------------
# Option groups
# ---------------------------------------------------------------------------

MC = "Multiple comparisons"
CHECKS = "Assumption checks"
OUT = "Output"

RANK_ADJUST = [("bonferroni", "Bonferroni (Dunn's test)"), ("holm", "Holm"), ("sidak", "Šidák"),
               ("fdr", "FDR (Benjamini–Hochberg)"), ("none", "None (uncorrected)")]

EMM_METHODS = [("tukey", "Tukey"), ("bonferroni", "Bonferroni"), ("holm", "Holm"),
               ("sidak", "Šidák"), ("dunnett", "Dunnett (vs control)"), ("none", "None (uncorrected)")]


def _compare_opts(control_sources: list[str], default: str = "pairs") -> list[dict]:
    return [
        opt.choice("compare", "Compare", [("none", "No multiple comparisons"),
                                          ("pairs", "Every group with every other group"),
                                          ("control", "Every group with a control group")],
                   default=default, section=MC),
        opt.level("control", "Control group", sources=control_sources, section=MC,
                  visible={"compare": "control"}, help="Leave empty to use the first group."),
    ]


def _factorial_posthoc_opts(sources: list[str], default: str = "main") -> list[dict]:
    """Main-effect and simple-effect comparisons on estimated marginal means."""
    return [
        opt.choice("posthoc", "Compare", [
            ("none", "No multiple comparisons"),
            ("main", "Main effects (levels of each factor)"),
            ("simple", "Simple effects (one factor within each level of another)"),
            ("both", "Main effects and simple effects")], default=default, section=MC),
        opt.level("simple_factor", "Simple effects: compare the levels of", sources=sources, section=MC,
                  visible={"posthoc": ["simple", "both"]}, help="Empty = the first factor."),
        opt.level("simple_by", "…within each level of", sources=sources, section=MC,
                  visible={"posthoc": ["simple", "both"]}, help="Empty = the second factor."),
        opt.choice("compare", "Which comparisons", [("pairs", "All pairs of levels"),
                                                    ("control", "Each level with a control level")],
                   section=MC),
        opt.text("control", "Control level", placeholder="first level", section=MC,
                 visible={"compare": "control"},
                 help="Name of the control level. Empty = the first level of each factor."),
        opt.choice("posthoc_method", "Correction for multiple comparisons", EMM_METHODS, section=MC),
    ]


# ---------------------------------------------------------------------------
# One-way ANOVA and Kruskal–Wallis
# ---------------------------------------------------------------------------

def _group_names(ctx: Context) -> dict:
    if ctx.opt("layout", "wide") == "long":
        return {"value": ctx.var("dep"), "group": ctx.var("group")}
    return {"value": "Value", "group": "Group"}


def prepare_groups(ctx: Context):
    """``value`` / ``group`` for independent groups, with basic checks."""
    names = _group_names(ctx)
    long = ctx.long()
    counts = long["group"].value_counts()
    if (counts > 0).sum() < 2:
        raise AnalysisError(
            f"This analysis compares two or more groups, but only {(counts > 0).sum()} group with data "
            "was found. Select more groups.")
    if len(long) <= (counts > 0).sum():
        raise AnalysisError(
            "Every group has only one value, so the variation within groups cannot be estimated. "
            "At least one group needs two or more values.")
    return long, with_private(ctx.options, names, 0)


ONEWAY_METHODS = [("tukey", "Tukey (recommended for all pairs)"), ("dunnett", "Dunnett (vs control)"),
                  ("bonferroni", "Bonferroni"), ("sidak", "Šidák"), ("holm_sidak", "Holm–Šidák"),
                  ("lsd", "Fisher's LSD (uncorrected)"),
                  ("games_howell", "Games–Howell (unequal SDs)"),
                  ("fdr", "FDR (Benjamini–Hochberg)")]

r_analysis(
    id="anova_oneway", name="One-way ANOVA", category=CAT3, script="anova_oneway.R",
    description="Compare the means of three or more independent groups (ordinary, Welch or "
                "Brown–Forsythe ANOVA) with multiple comparisons.",
    keywords="anova f test tukey dunnett bonferroni sidak games-howell welch brown-forsythe post hoc trend",
    requires=["ggplot2", "multcomp"], prepare=prepare_groups,
    options=[
        *opt.group_layout(min_groups=2),
        opt.choice("variant", "Assume equal SDs?", [
            ("ordinary", "Yes – ordinary ANOVA"),
            ("welch", "No – Welch's ANOVA"),
            ("brown_forsythe", "No – Brown–Forsythe ANOVA")]),
        *_compare_opts(["group", "columns"]),
        opt.choice("posthoc", "Test", ONEWAY_METHODS, section=MC,
                   help="Without the equal-SD assumption, Tukey becomes Games–Howell and "
                        "Dunnett becomes Dunnett's T3."),
        opt.boolean("trend", "Test for linear trend (groups in order)", False, section=MC),
        opt.boolean("assumptions", "Equal variances and normality tests", True, section=CHECKS),
        opt.boolean("qq_plot", "QQ plot of residuals", False, section=CHECKS),
        opt.boolean("descriptives", "Descriptive statistics", True, section=OUT),
        opt.boolean("show_plot", "Plot of means with confidence intervals", True, section=OUT),
        opt.alpha(section=OUT),
    ])

r_analysis(
    id="kruskal_wallis", name="Kruskal–Wallis test", category=CAT3, script="kruskal_wallis.R",
    description="Nonparametric comparison of three or more independent groups, with Dunn's "
                "multiple comparisons.",
    keywords="nonparametric rank dunn anova on ranks",
    requires=["ggplot2"], prepare=prepare_groups,
    options=[
        *opt.group_layout(min_groups=2),
        *_compare_opts(["group", "columns"]),
        opt.choice("adjust", "Correction for multiple comparisons", RANK_ADJUST, section=MC),
        opt.boolean("descriptives", "Medians and mean ranks", True, section=OUT),
        opt.boolean("show_plot", "Box plot", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Repeated-measures one-way ANOVA and Friedman
# ---------------------------------------------------------------------------

def _matched(ctx: Context, complete: bool, min_conditions: int = 2):
    """Long ``value`` / ``group`` / ``subject`` frame from matched data."""
    names = _group_names(ctx)
    if ctx.opt("layout", "wide") == "long" and ctx.has("subject"):
        names["subject"] = ctx.var("subject")
    else:
        names["subject"] = "Subject"
    wide = ctx.paired(complete=False)
    wide = wide.dropna(how="all")
    if wide.shape[1] < min_conditions:
        raise AnalysisError(
            f"Repeated-measures analyses need at least {min_conditions} conditions "
            f"(columns or groups), but {wide.shape[1]} was found.")
    n_all = len(wide)
    full = wide.dropna()
    if len(full) < 2:
        raise AnalysisError(
            "At least two subjects with a value in every condition are needed. "
            "Check that each row (or each subject ID) has a value in every group.")
    used = full if complete else wide
    long = (used.reset_index().melt(id_vars="subject", var_name="group", value_name="value")
            .dropna(subset=["value"]).reset_index(drop=True))
    long["group"] = pd.Categorical(long["group"], categories=list(wide.columns))
    long["subject"] = pd.Categorical(long["subject"].astype(str),
                                     categories=[str(s) for s in used.index])
    extra = {"n_subjects": int(len(used)), "n_incomplete": int(n_all - len(full))}
    return long, with_private(ctx.options, names, n_all - len(used), **extra)


def prepare_rm_oneway(ctx: Context):
    return _matched(ctx, complete=ctx.opt("missing", "exclude") != "mixed")


def prepare_friedman(ctx: Context):
    return _matched(ctx, complete=True)


RM_METHODS = [("tukey", "Tukey"), ("bonferroni", "Bonferroni"), ("holm", "Holm"),
              ("sidak", "Šidák"), ("dunnett", "Dunnett (vs control)"), ("none", "None (uncorrected)")]

r_analysis(
    id="anova_rm_oneway", name="Repeated-measures one-way ANOVA", category=CAT3,
    script="anova_rm_oneway.R",
    description="Compare three or more matched measurements (each subject measured in every "
                "condition), with sphericity corrections and multiple comparisons.",
    keywords="repeated measures within subject matched greenhouse geisser sphericity mauchly mixed model",
    requires=["ggplot2"], prepare=prepare_rm_oneway,
    options=[
        *opt.group_layout(paired=True, min_groups=2),
        opt.choice("sphericity", "Sphericity", [
            ("gg", "Do not assume – Greenhouse–Geisser correction"),
            ("hf", "Do not assume – Huynh–Feldt correction"),
            ("assume", "Assume sphericity (no correction)")]),
        opt.choice("missing", "Subjects with missing values", [
            ("exclude", "Exclude them (classic repeated-measures ANOVA)"),
            ("mixed", "Keep them – fit a mixed-effects model")],
            help="The mixed-effects model gives the same answer as the ANOVA when nothing is missing "
                 "and sphericity is assumed."),
        *_compare_opts(["group", "columns"]),
        opt.choice("posthoc", "Test", RM_METHODS, section=MC),
        opt.boolean("assumptions", "Sphericity and normality tests", True, section=CHECKS),
        opt.boolean("qq_plot", "QQ plot of residuals", False, section=CHECKS),
        opt.boolean("descriptives", "Descriptive statistics", True, section=OUT),
        opt.boolean("show_plot", "Plot of means with confidence intervals", True, section=OUT),
        opt.alpha(section=OUT),
    ])

r_analysis(
    id="friedman", name="Friedman test", category=CAT3, script="friedman.R",
    description="Nonparametric comparison of three or more matched measurements, with Dunn's "
                "multiple comparisons.",
    keywords="nonparametric repeated measures rank matched kendall w dunn",
    requires=["ggplot2"], prepare=prepare_friedman,
    options=[
        *opt.group_layout(paired=True, min_groups=2),
        *_compare_opts(["group", "columns"]),
        opt.choice("adjust", "Correction for multiple comparisons", RANK_ADJUST, section=MC),
        opt.boolean("descriptives", "Medians and rank sums", True, section=OUT),
        opt.boolean("show_plot", "Plot of each subject's values", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Factorial ANOVA / ANCOVA
# ---------------------------------------------------------------------------

def prepare_factorial(ctx: Context):
    dep = ctx.var("dep")
    factors = ctx.vars("factors", min=1)
    covs = ctx.vars("covariates", min=0)
    if len(factors) > 4:
        raise AnalysisError("At most four factors are supported. Remove some factors.")
    roles = ([("y", dep, "num")] + [(f"f{i + 1}", c, "factor") for i, c in enumerate(factors)]
             + [(f"x{i + 1}", c, "num") for i, c in enumerate(covs)])
    df, names, dropped = model_frame(ctx, roles)
    if len(df) < 3:
        raise AnalysisError("Too few complete rows to fit the model (fewer than three).")
    check_numeric(df, "y", dep, "outcome")
    for i, c in enumerate(factors):
        check_factor(df, f"f{i + 1}", c)
    for i, c in enumerate(covs):
        check_numeric(df, f"x{i + 1}", c)
    return df, with_private(ctx.options, names, dropped,
                            factors=[f"f{i + 1}" for i in range(len(factors))],
                            covariates=[f"x{i + 1}" for i in range(len(covs))])


r_analysis(
    id="anova_factorial", name="Two-way / N-way ANOVA", category=FACTORIAL,
    script="anova_factorial.R",
    description="ANOVA with one to four between-subjects factors, interactions and optional "
                "covariates (ANCOVA); estimated marginal means and post hoc comparisons.",
    keywords="two-way three-way factorial ancova interaction simple effects emmeans type III",
    requires=["ggplot2", "car", "emmeans"], prepare=prepare_factorial,
    options=[
        opt.variable("dep", "Dependent variable", opt.CONTINUOUS),
        opt.variables("factors", "Factors", opt.ANY, min=1, max=4),
        opt.variables("covariates", "Covariates (ANCOVA)", opt.CONTINUOUS, min=0),
        opt.choice("model", "Model terms", [("full", "Full factorial (all interactions)"),
                                            ("twoway", "Main effects + two-way interactions"),
                                            ("main", "Main effects only")], section="Model"),
        opt.choice("ss_type", "Sums of squares", [("3", "Type III"), ("2", "Type II")], section="Model"),
        *_factorial_posthoc_opts(["factors"]),
        opt.boolean("assumptions", "Equal variances and normality tests", True, section=CHECKS),
        opt.boolean("resid_plots", "Residual plots", False, section=CHECKS),
        opt.boolean("descriptives", "Descriptive statistics for every cell", True, section=OUT),
        opt.boolean("emm", "Estimated marginal means", True, section=OUT),
        opt.boolean("show_plot", "Interaction plot", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Repeated-measures / mixed ANOVA
# ---------------------------------------------------------------------------

def prepare_anova_mixed(ctx: Context):
    form = ctx.opt("data_form", "long")
    between = ctx.vars("between", min=0)
    use_lmm = ctx.opt("engine", "anova") == "lmm"
    names: dict[str, str] = {}
    if form == "wide":
        cols = ctx.vars("rm_columns", min=2)
        overlap = [c for c in between if c in cols]
        if overlap:
            raise AnalysisError(f"“{overlap[0]}” is selected both as a repeated measurement and as a "
                                "between-subjects factor.")
        wname = (ctx.opt("within_name", "") or "").strip() or "Condition"
        parts = []
        for c in cols:
            part = pd.DataFrame({"subject": np.arange(len(ctx.df)).astype(str),
                                 "y": ctx.numeric(c).to_numpy(), "w1": c})
            for j, b in enumerate(between):
                part[f"b{j + 1}"] = ctx.factor(b).to_numpy()
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        df["w1"] = pd.Categorical(df["w1"], categories=cols)
        for j, b in enumerate(between):
            df[f"b{j + 1}"] = pd.Categorical(df[f"b{j + 1}"], categories=list(ctx.factor(b).cat.categories))
        names.update({"y": "Value", "subject": "Subject", "w1": wname})
        within_safe = ["w1"]
        # rows without any measurement are padding
        has_any = df.groupby("subject")["y"].transform(lambda s: s.notna().any())
        df = df[has_any]
    else:
        dep = ctx.var("dep")
        subject = ctx.var("subject")
        within = ctx.vars("within", min=1)
        roles = ([("y", dep, "num"), ("subject", subject, "factor")]
                 + [(f"w{i + 1}", c, "factor") for i, c in enumerate(within)]
                 + [(f"b{i + 1}", c, "factor") for i, c in enumerate(between)])
        df, names, _ = model_frame(ctx, roles, dropna=False)
        df["subject"] = df["subject"].astype(object)
        within_safe = [f"w{i + 1}" for i in range(len(within))]
    between_safe = [f"b{i + 1}" for i in range(len(between))]
    for j, b in enumerate(between):
        names[f"b{j + 1}"] = b

    # rows that cannot be used at all
    df = df.dropna(subset=["subject"] + within_safe)
    bad_between = df[between_safe].isna().any(axis=1) if between_safe else pd.Series(False, index=df.index)
    df = df[~bad_between & df["y"].notna()].copy()
    if df.empty:
        raise AnalysisError("No complete observations were found for the selected variables.")
    df["subject"] = df["subject"].astype(str)

    # between-subject factors must be constant within a subject
    for safe in between_safe:
        nun = df.groupby("subject", observed=True)[safe].nunique()
        if (nun > 1).any():
            raise AnalysisError(
                f"“{names[safe]}” takes more than one value within some subjects (for example subject "
                f"{nun[nun > 1].index[0]}), so it is not a between-subjects factor. "
                "Move it to the within-subjects factors.")

    # several observations per subject and cell are averaged (as afex does)
    keys = ["subject"] + within_safe
    aggregated = bool(df.duplicated(keys).any())
    if aggregated:
        first = {b: "first" for b in between_safe}
        df = df.groupby(keys, observed=True, as_index=False).agg({"y": "mean", **first})

    for safe in within_safe + between_safe:
        df[safe] = pd.Categorical(df[safe]).remove_unused_categories() \
            if not isinstance(df[safe].dtype, pd.CategoricalDtype) else df[safe].cat.remove_unused_categories()
        check_factor(df, safe, names[safe])

    n_cells = int(np.prod([len(df[w].cat.categories) for w in within_safe]))
    per_subject = df.groupby("subject", observed=True).size()
    incomplete = per_subject[per_subject < n_cells].index
    n_complete = int(len(per_subject) - len(incomplete))
    if use_lmm:
        if len(per_subject) < 3:
            raise AnalysisError("A mixed-effects model needs at least three subjects.")
    else:
        if n_complete < 2:
            raise AnalysisError(
                "Fewer than two subjects have a value in every within-subjects condition. "
                "Check the subject column and the within-subjects factors, or switch the method to "
                "the mixed-effects model, which also uses subjects with missing values.")
        df = df[~df["subject"].isin(incomplete)]
    df = df.reset_index(drop=True)
    for safe in within_safe + between_safe:
        df[safe] = df[safe].cat.remove_unused_categories()
        check_factor(df, safe, names[safe])
    df["subject"] = pd.Categorical(df["subject"])
    check_numeric(df, "y", names["y"], "outcome")
    return df, with_private(ctx.options, names, 0, within=within_safe, between=between_safe,
                            n_incomplete=int(len(incomplete)), n_subjects=int(df["subject"].nunique()),
                            aggregated=aggregated)


r_analysis(
    id="anova_mixed", name="Repeated-measures / mixed ANOVA (two-way and beyond)", category=FACTORIAL,
    script="anova_mixed.R",
    description="ANOVA with within-subjects (repeated) factors and optional between-subjects "
                "factors; sphericity corrections, or a mixed-effects model when values are missing.",
    keywords="repeated measures split plot within between mixed anova sphericity greenhouse geisser afex",
    requires=["ggplot2", "afex", "emmeans", "lmerTest"], prepare=prepare_anova_mixed,
    options=[
        opt.choice("data_form", "Data layout", [
            ("long", "Long: one row per measurement"),
            ("wide", "Wide: one row per subject, one column per repeated measurement")], section="Data"),
        opt.variable("dep", "Dependent variable", opt.CONTINUOUS, section="Data", visible={"data_form": "long"}),
        opt.variable("subject", "Subject / ID", opt.ANY, section="Data", visible={"data_form": "long"}),
        opt.variables("within", "Within-subjects factors", opt.ANY, min=1, max=3, section="Data",
                      visible={"data_form": "long"}),
        opt.variables("rm_columns", "Repeated measurements (columns)", opt.CONTINUOUS, min=2, section="Data",
                      visible={"data_form": "wide"}),
        opt.text("within_name", "Name of the repeated factor", "Condition", section="Data",
                 visible={"data_form": "wide"}),
        opt.variables("between", "Between-subjects factors", opt.ANY, min=0, max=3, section="Data"),
        opt.choice("engine", "Method", [
            ("anova", "Repeated-measures ANOVA (complete subjects only)"),
            ("lmm", "Mixed-effects model (handles missing values)")], section="Model"),
        opt.choice("sphericity", "Sphericity correction", [
            ("gg", "Greenhouse–Geisser"), ("hf", "Huynh–Feldt"), ("none", "None (assume sphericity)")],
            section="Model", visible={"engine": "anova"}),
        *_factorial_posthoc_opts(["within", "between"]),
        opt.boolean("assumptions", "Sphericity, equal variances and normality tests", True, section=CHECKS),
        opt.boolean("descriptives", "Descriptive statistics for every cell", True, section=OUT),
        opt.boolean("emm", "Estimated marginal means", True, section=OUT),
        opt.boolean("show_plot", "Plot of means", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Linear mixed-effects model
# ---------------------------------------------------------------------------

def prepare_mixed_model(ctx: Context):
    dep = ctx.var("dep")
    factors = ctx.vars("factors", min=0)
    covs = ctx.vars("covariates", min=0)
    random = ctx.vars("random", min=1)
    slope = ctx.var("slope", required=False)
    roles = ([("y", dep, "num")] + [(f"f{i + 1}", c, "factor") for i, c in enumerate(factors)]
             + [(f"x{i + 1}", c, "num") for i, c in enumerate(covs)]
             + [(f"g{i + 1}", c, "factor") for i, c in enumerate(random)])
    slope_safe = None
    if slope:
        by_col = {c: s for s, c, _ in roles}
        if slope not in factors and slope not in covs:
            raise AnalysisError(
                f"The random-slope variable “{slope}” must also be one of the fixed-effect factors "
                "or covariates.")
        slope_safe = by_col[slope]
    df, names, dropped = model_frame(ctx, roles)
    if len(df) < 4:
        raise AnalysisError("Too few complete rows to fit a mixed model.")
    check_numeric(df, "y", dep, "outcome")
    for i, c in enumerate(factors):
        check_factor(df, f"f{i + 1}", c)
    for i, c in enumerate(covs):
        check_numeric(df, f"x{i + 1}", c)
    for i, c in enumerate(random):
        n_lev = len(df[f"g{i + 1}"].cat.categories)
        if n_lev < 2:
            raise AnalysisError(f"The grouping variable “{c}” has only one level, so no random effect "
                                "can be estimated for it.")
        if n_lev >= len(df):
            raise AnalysisError(
                f"Every row has a different value of “{c}”, so its random effect cannot be separated "
                "from the residual variation. Choose a grouping variable with repeated observations "
                "(for example subject, animal, batch).")
    return df, with_private(ctx.options, names, dropped,
                            factors=[f"f{i + 1}" for i in range(len(factors))],
                            covariates=[f"x{i + 1}" for i in range(len(covs))],
                            random=[f"g{i + 1}" for i in range(len(random))],
                            slope=slope_safe)


r_analysis(
    id="mixed_model", name="Linear mixed-effects model", category=FACTORIAL, script="mixed_model.R",
    description="Linear model with fixed effects and random intercepts (and an optional random "
                "slope) for clustered or repeated data.",
    keywords="lmm lmer random effects multilevel hierarchical random intercept slope icc satterthwaite",
    requires=["ggplot2", "lme4", "lmerTest", "emmeans"], prepare=prepare_mixed_model,
    options=[
        opt.variable("dep", "Dependent variable", opt.CONTINUOUS),
        opt.variables("factors", "Fixed factors", opt.ANY, min=0),
        opt.variables("covariates", "Fixed covariates", opt.CONTINUOUS, min=0),
        opt.variables("random", "Grouping variables (random intercepts)", opt.ANY, min=1,
                      help="For example subject, animal, plate or site."),
        opt.variable("slope", "Random slope for (optional)", opt.ANY, required=False,
                     help="One of the fixed effects whose effect may differ between the levels of the "
                          "first grouping variable."),
        opt.choice("interactions", "Interactions between fixed effects", [
            ("none", "None (main effects only)"), ("twoway", "All two-way interactions"),
            ("full", "All interactions")], section="Model"),
        opt.choice("estimation", "Estimation", [("reml", "REML (recommended)"), ("ml", "Maximum likelihood")],
                   section="Model"),
        opt.choice("posthoc", "Compare levels of each factor", [("none", "No"), ("main", "Yes")],
                   default="main", section=MC),
        opt.choice("compare", "Which comparisons", [("pairs", "All pairs of levels"),
                                                    ("control", "Each level with a control level")],
                   section=MC),
        opt.text("control", "Control level", placeholder="first level", section=MC,
                 visible={"compare": "control"}),
        opt.choice("posthoc_method", "Correction for multiple comparisons", EMM_METHODS, section=MC),
        opt.boolean("resid_plots", "Residual diagnostic plots", True, section=CHECKS),
        opt.boolean("assumptions", "Normality test of residuals", False, section=CHECKS),
        opt.boolean("coefficients", "Fixed-effect coefficients", True, section=OUT),
        opt.boolean("emm", "Estimated marginal means", True, section=OUT),
        opt.alpha(section=OUT),
    ])


# ---------------------------------------------------------------------------
# Nested t test / nested one-way ANOVA
# ---------------------------------------------------------------------------

def prepare_nested(ctx: Context):
    dep, group, nested = ctx.var("dep"), ctx.var("group"), ctx.var("nested")
    df, names, dropped = model_frame(ctx, [("y", dep, "num"), ("group", group, "factor"),
                                           ("nested", nested, "factor")])
    if len(df) < 4:
        raise AnalysisError("Too few complete rows for a nested analysis.")
    check_numeric(df, "y", dep, "outcome")
    check_factor(df, "group", group, what="grouping variable")
    # label subgroups uniquely within their group (IDs are often reused across groups)
    sub = df["group"].astype(str) + " / " + df["nested"].astype(str)
    order = list(dict.fromkeys(sub[np.lexsort((df["nested"].cat.codes, df["group"].cat.codes))]))
    df["sub"] = pd.Categorical(sub, categories=order)
    per_group = df.groupby("group", observed=True)["sub"].nunique()
    if (per_group < 2).any():
        g = per_group[per_group < 2].index[0]
        raise AnalysisError(
            f"Group “{g}” contains only one subgroup of “{nested}”. A nested analysis needs at least "
            "two subgroups (e.g. animals, replicates) in every group.")
    if df["sub"].nunique() >= len(df):
        raise AnalysisError(
            f"Every subgroup of “{nested}” contains a single value, so there is nothing nested. "
            "Use an ordinary t test or one-way ANOVA instead.")
    return df, with_private(ctx.options, names, dropped)


r_analysis(
    id="nested_anova", name="Nested t test / nested one-way ANOVA", category=FACTORIAL,
    script="nested_anova.R",
    description="Compare groups when the values are technical replicates nested within "
                "subgroups (e.g. cells within animals within treatments).",
    keywords="nested hierarchical pseudoreplication technical replicates subgroups mixed model",
    requires=["ggplot2", "lme4", "lmerTest", "emmeans"], prepare=prepare_nested,
    options=[
        opt.variable("dep", "Dependent variable", opt.CONTINUOUS),
        opt.variable("group", "Group (treatment)", opt.ANY),
        opt.variable("nested", "Subgroup nested within group", opt.ANY,
                     help="For example animal, donor, dish or replicate experiment."),
        opt.choice("compare", "Compare", [("none", "No multiple comparisons"),
                                          ("pairs", "Every group with every other group"),
                                          ("control", "Every group with a control group")],
                   default="pairs", section=MC),
        opt.level("control", "Control group", sources=["group"], section=MC, visible={"compare": "control"}),
        opt.choice("posthoc_method", "Correction for multiple comparisons", EMM_METHODS, section=MC),
        opt.boolean("descriptives", "Subgroup means", True, section=OUT),
        opt.boolean("show_plot", "Plot of subgroups", True, section=OUT),
        opt.alpha(section=OUT),
    ])
