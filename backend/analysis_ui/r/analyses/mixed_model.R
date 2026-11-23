# Linear mixed-effects model (lme4 / lmerTest).
# data: y, f1.. (fixed factors), x1.. (fixed covariates), g1.. (grouping variables)

# Nakagawa & Schielzeth R2 (marginal / conditional), valid for random slopes
# (Johnson 2014): random variance = mean over observations of z' Sigma z.
.lmm_r2 <- function(fit) {
  x <- lme4::getME(fit, "X")
  vf <- stats::var(as.vector(x %*% lme4::fixef(fit)))
  vc <- lme4::VarCorr(fit)
  mm <- lme4::getME(fit, "mmList")
  vr <- 0
  for (i in seq_along(vc)) {
    sigma <- as.matrix(vc[[i]]); z <- mm[[i]]
    vr <- vr + mean(rowSums((z %*% sigma) * z))
  }
  ve <- stats::sigma(fit)^2
  c(marginal = vf / (vf + vr + ve), conditional = (vf + vr) / (vf + vr + ve))
}

.lmm_warnings <- function(w) {
  if (!length(w)) return(invisible(NULL))
  if (any(grepl("converge|Hessian|eigenvalue|gradient", w, ignore.case = TRUE))) {
    aui_warn("The model did not converge cleanly, so the estimates may be unreliable. This usually means the random-effects structure is too complex for the data: try removing the random slope or a grouping variable, or rescale covariates with very large or very small values.")
  }
  other <- w[!grepl("converge|Hessian|eigenvalue|gradient|singular", w, ignore.case = TRUE)]
  for (o in unique(other)) aui_warn(o)
}

run <- function(data, options) {
  aui_need("ggplot2", "lme4", "lmerTest", "emmeans")
  names <- st_names(options)
  alpha <- st_alpha(options)
  factors <- aui_vars(options, "_factors")
  covs <- aui_vars(options, "_covariates")
  random <- aui_vars(options, "_random")
  slope <- aui_opt(options, "_slope", NULL)
  fixed <- c(factors, covs)
  reml <- !identical(aui_opt(options, "estimation", "reml"), "ml")

  rand <- paste0("(1 | ", random, ")")
  if (!is.null(slope)) rand[1] <- paste0("(1 + ", slope, " | ", random[1], ")")
  rhs <- st_rhs(fixed, aui_opt(options, "interactions", "none"))
  form <- stats::as.formula(paste("y ~", rhs, "+", paste(rand, collapse = " + ")))
  fail <- paste0("The mixed model could not be fitted. ",
                 if (!is.null(slope)) "The random slope may be too much for these data (it needs several observations per level of the grouping variable); try removing it."
                 else "Check that each grouping variable has several observations per level.")
  q <- st_quiet(st_try(st_with_sum_contrasts(lmerTest::lmer(form, data = data, REML = reml)), fail))
  fit <- q$value
  .lmm_warnings(q$warnings)
  n_coef_full <- ncol(stats::model.matrix(stats::as.formula(paste("~", rhs)), data))
  if (length(lme4::fixef(fit)) < n_coef_full) {
    aui_stop("Some fixed effects cannot be estimated because they are perfectly confounded (for example empty combinations of factor levels, or a predictor that duplicates another). Remove an interaction or one of the confounded predictors.")
  }
  singular <- lme4::isSingular(fit)
  fit_t <- if (length(factors)) st_quiet(st_with_treatment_contrasts(lmerTest::lmer(form, data = data, REML = reml)))$value else fit

  # ---- model fit
  r2 <- .lmm_r2(fit)
  ng <- vapply(random, function(g) nlevels(droplevels(data[[g]])), numeric(1))
  aui_table("Model fit", data.frame(
    Measure = c("Observations", paste0("Levels of ", vapply(random, function(g) st_name(names, g), "")),
                "R² marginal (fixed effects)", "R² conditional (fixed + random effects)",
                "Residual SD", "Log-likelihood", "AIC", "BIC"),
    value = c(nrow(data), ng, r2[["marginal"]], r2[["conditional"]], stats::sigma(fit),
              as.numeric(stats::logLik(fit)), stats::AIC(fit), stats::BIC(fit)), stringsAsFactors = FALSE),
    labels = c(value = "Value"),
    footnotes = c(paste0("Fitted by ", if (reml) "restricted maximum likelihood (REML)" else "maximum likelihood (ML)", " with lme4. Model: ",
                         st_pretty(gsub("\\+ \\(", "+ (", paste(deparse(form, width.cutoff = 500), collapse = "")), names), "."),
                  "R² after Nakagawa & Schielzeth: marginal = variance explained by the fixed effects; conditional = by fixed and random effects together.",
                  if (reml) "Use ML rather than REML when comparing AIC/BIC of models with different fixed effects.",
                  st_dropped_note(options)))

  # ---- fixed effects
  a <- NULL
  if (length(fixed)) {
    a <- as.data.frame(st_quiet(stats::anova(fit, type = 3, ddf = "Satterthwaite"))$value)
    aui_table("Tests of fixed effects", data.frame(
      Source = st_pretty(rownames(a), names), SS = a$`Sum Sq`, MS = a$`Mean Sq`, df1 = a$NumDF, df2 = a$DenDF,
      F = a$`F value`, p = a$`Pr(>F)`, stars = aui_stars(a$`Pr(>F)`), stringsAsFactors = FALSE),
      formats = c(p = "p", stars = "stars", df1 = "int"),
      labels = c(df1 = "df numerator", df2 = "df denominator", p = "P value", stars = "Summary"),
      footnotes = "Type III F tests with Satterthwaite's approximation for the denominator degrees of freedom (sum-to-zero contrasts).")
  }
  if (aui_flag(options, "coefficients", TRUE)) {
    co <- as.data.frame(stats::coef(summary(fit_t)))
    crit <- stats::qt(1 - alpha / 2, co$df)
    aui_table("Fixed-effect coefficients", data.frame(
      Term = st_coef_labels(stats::terms(fit_t), stats::model.matrix(fit_t), data, names),
      est = co$Estimate, se = co$`Std. Error`, df = co$df, t = co$`t value`, p = co$`Pr(>|t|)`,
      lower = co$Estimate - crit * co$`Std. Error`, upper = co$Estimate + crit * co$`Std. Error`, stringsAsFactors = FALSE),
      formats = c(p = "p"),
      labels = c(est = "Estimate", se = "SE", p = "P value", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
      footnotes = c("t tests with Satterthwaite degrees of freedom; Wald confidence intervals.",
                    st_reference_note(data, factors, names)))
  }

  # ---- random effects
  vc <- st_emit_varcomp(fit, names, footnotes = c(
    if (is.null(slope)) "The share of variance of a grouping variable is its intraclass correlation (ICC): the correlation between two observations from the same level, after accounting for the fixed effects."
    else "With a random slope the between-group variance depends on the value of the slope variable; the shares shown refer to the intercepts only.",
    "Random effects are assumed to be normally distributed."))
  if (singular) {
    aui_warn(paste0("Singular fit: at least one random-effect variance was estimated as zero (or a correlation as ±1). ",
                    "The data do not support all the random effects in the model; the fixed-effect results are usually still usable, ",
                    "but consider removing the random ", if (!is.null(slope)) "slope" else "effect with zero variance", "."))
  }

  # ---- estimated marginal means and comparisons
  if (length(factors) && aui_flag(options, "emm", TRUE)) {
    aui_heading("Estimated marginal means")
    for (f in factors) st_emit_emm(fit, f, names, alpha,
                                   footnotes = if (length(fixed) > 1) "Averaged over the levels of other factors and evaluated at the mean of the covariates.")
  }
  if (length(factors) && identical(aui_opt(options, "posthoc", "main"), "main")) {
    aui_heading("Multiple comparisons")
    for (f in factors) {
      st_emit_emm_comparisons(fit, f, NULL, names, type = aui_opt(options, "compare", "pairs"),
                              control = aui_opt(options, "control", ""), method = aui_opt(options, "posthoc_method", "tukey"),
                              alpha = alpha, extra_notes = "Satterthwaite degrees of freedom.")
    }
  }

  # ---- diagnostics
  res <- stats::residuals(fit)
  if (aui_flag(options, "assumptions")) st_emit_normality(res, alpha)
  if (aui_flag(options, "resid_plots", TRUE)) {
    st_emit_residual_plots(stats::fitted(fit), res, "Residual diagnostics")
    re <- lme4::ranef(fit)[[random[1]]][, 1]
    if (length(re) >= 5 && !singular) {
      aui_plot(paste0("QQ plot of the random intercepts (", st_name(names, random[1]), ")"),
               st_qq_plot(re, "Random intercepts") + ggplot2::labs(y = "Estimated random intercept"), width = 4.6, height = 3.8)
    }
  }

  icc <- vc$vcov[vc$grp == random[1] & vc$var1 == "(Intercept)" & is.na(vc$var2)] /
    sum(vc$vcov[is.na(vc$var2) & (is.na(vc$var1) | vc$var1 == "(Intercept)")])
  lead <- sprintf("Linear mixed model with %d observations in %s (ICC for %s = %s; marginal R² = %s, conditional R² = %s).",
                  nrow(data), paste(sprintf("%d levels of %s", ng, vapply(random, function(g) st_name(names, g), "")), collapse = " and "),
                  st_name(names, random[1]), formatC(icc, format = "f", digits = 3),
                  formatC(r2[["marginal"]], format = "f", digits = 3), formatC(r2[["conditional"]], format = "f", digits = 3))
  if (!is.null(a)) aui_summary(st_effects_summary(lead, rownames(a), a$`F value`, a$NumDF, a$DenDF, a$`Pr(>F)`, alpha, names))
  else aui_summary(paste(lead, "The model has no fixed effects other than the intercept."))
}
