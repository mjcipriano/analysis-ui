# Poisson / quasi-Poisson / negative-binomial regression for counts.
# data: y (counts), x1.., f1.., optional expo (exposure > 0)
run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  covs <- aui_vars(options, "_covariates")
  factors <- aui_vars(options, "_factors")
  family <- aui_opt(options, "family", "poisson")
  has_offset <- isTRUE(options[["_has_offset"]])
  yname <- st_name(names, "y")
  if (isTRUE(options[["_noninteger"]])) {
    if (family != "quasipoisson") aui_stop("“", yname, "” contains non-integer values. Poisson and negative-binomial regression are for counts (0, 1, 2, …). ",
                                         "For rates, model the counts and give the denominator as the exposure; for other non-integer data choose quasi-Poisson.")
  }

  rhs <- st_rhs(c(covs, factors), aui_opt(options, "interactions", "none"), options[["_rhs"]])
  form <- stats::as.formula(paste("y ~", rhs, if (has_offset) "+ offset(log(expo))" else ""))
  null_form <- stats::as.formula(paste("y ~ 1", if (has_offset) "+ offset(log(expo))" else ""))
  fit_one <- function(f, fam) {
    if (fam == "negbin") { aui_need("MASS"); MASS::glm.nb(f, data = data) }
    else stats::glm(f, data = data, family = if (fam == "quasipoisson") stats::quasipoisson() else stats::poisson())
  }
  msg <- "The model could not be fitted. Check the custom model terms."
  pois <- st_quiet(st_try(st_with_treatment_contrasts(fit_one(form, "poisson")), msg))$value
  mm <- stats::model.matrix(pois)
  labels <- st_coef_labels(stats::terms(pois), mm, data, names)
  st_check_aliased(stats::coef(pois), labels)
  if (stats::df.residual(pois) < 1) aui_stop("The model has as many coefficients as observations. Use fewer predictors or more data.")
  pearson <- sum(stats::residuals(pois, type = "pearson")^2)
  dfr <- stats::df.residual(pois)
  disp <- pearson / dfr

  nb_note <- NULL
  if (family == "poisson") {
    fit <- pois
  } else {
    q <- st_quiet(tryCatch(st_with_treatment_contrasts(fit_one(form, family)), error = function(e) NULL))
    fit <- q$value
    if (is.null(fit) || (family == "negbin" && (any(grepl("iteration limit|alternation limit", q$warnings)) || fit$theta > 1e5))) {
      if (family == "negbin") {
        nb_note <- "The negative-binomial model could not estimate extra-Poisson variation (the counts are not overdispersed), so its results coincide with the Poisson model shown here."
        fit <- pois; family <- "poisson"
      } else aui_stop(msg)
    }
  }
  fam_name <- switch(family, poisson = "Poisson", quasipoisson = "Quasi-Poisson", negbin = "Negative-binomial")
  co <- stats::coef(summary(fit))
  stat_name <- if (colnames(co)[3] == "t value") "t" else "z"
  n <- nrow(data); k <- length(stats::coef(fit)) - 1

  # ---- model summary
  dev <- stats::deviance(fit)
  tab <- data.frame(Measure = c("Observations", "Sum of counts", "Residual deviance", "Residual df", "Pearson χ²",
                                "Dispersion (Pearson χ² / df, Poisson model)"),
                    value = c(n, sum(data$y), dev, dfr, sum(stats::residuals(fit, type = "pearson")^2), disp),
                    p = NA_real_, stringsAsFactors = FALSE)
  lr_txt <- ""
  if (family != "quasipoisson") {
    null <- st_quiet(tryCatch(fit_one(null_form, family), error = function(e) NULL))$value
    if (!is.null(null)) {
      lr <- max(0, 2 * (as.numeric(stats::logLik(fit)) - as.numeric(stats::logLik(null))))
      lr_p <- stats::pchisq(lr, k, lower.tail = FALSE)
      tab <- rbind(tab, data.frame(Measure = c(sprintf("Likelihood-ratio χ² vs intercept-only model (df = %d)", as.integer(k)), "Log-likelihood", "AIC"),
                                   value = c(lr, as.numeric(stats::logLik(fit)), stats::AIC(fit)), p = c(lr_p, NA, NA), stringsAsFactors = FALSE))
      lr_txt <- sprintf(" The predictors together %s (likelihood-ratio χ² = %s, df = %d, %s).",
                        if (lr_p < alpha) "improve the fit significantly" else "do not improve the fit significantly",
                        formatC(lr, format = "f", digits = 2), as.integer(k), st_p_text(lr_p))
    }
    if (family == "poisson") tab$p[tab$Measure == "Residual deviance"] <- stats::pchisq(dev, dfr, lower.tail = FALSE)
  } else {
    tab <- rbind(tab, data.frame(Measure = "Estimated dispersion parameter", value = summary(fit)$dispersion, p = NA, stringsAsFactors = FALSE))
  }
  if (family == "negbin") tab <- rbind(tab, data.frame(Measure = c("Theta (negative-binomial size)", "SE of theta"), value = c(fit$theta, fit$SE.theta), p = NA, stringsAsFactors = FALSE))
  aui_table("Model summary", tab, formats = c(p = "p"), labels = c(value = "Value", p = "P value"),
            footnotes = c(paste0(fam_name, " regression with log link", if (has_offset) paste0("; log(", st_name(names, "expo"), ") is an offset, so coefficients describe rates per unit of ", st_name(names, "expo")) else "",
                                 ". Model: ", st_pretty(paste("y ~", rhs), names), "."),
                          if (family == "poisson") "Deviance goodness-of-fit test: a small P value means the Poisson model does not fit (often because of overdispersion).",
                          if (family == "negbin") "Variance = mean + mean² / theta; small theta means strong overdispersion.",
                          if (family == "quasipoisson") "Quasi-Poisson keeps the Poisson coefficients but multiplies their standard errors by √dispersion; no likelihood, so AIC and likelihood-ratio tests are not available.",
                          nb_note, st_dropped_note(options)))

  # ---- overdispersion advice
  advice <- if (disp > 1.5) {
    if (family == "poisson") sprintf("The counts are overdispersed (dispersion = %s; 1 is expected for Poisson data): the Poisson standard errors are too small and P values too optimistic. Switch the distribution to quasi-Poisson or negative binomial.", st_fmt(disp, 3))
    else sprintf("The counts are overdispersed relative to a Poisson model (dispersion = %s), so the %s model is an appropriate choice.", st_fmt(disp, 3), tolower(fam_name))
  } else if (disp < 0.6) {
    sprintf("The counts vary less than a Poisson model expects (dispersion = %s, underdispersion); Poisson standard errors are then conservative. Quasi-Poisson adjusts for this.", st_fmt(disp, 3))
  } else {
    sprintf("No important overdispersion (dispersion = %s; values near 1 are consistent with a Poisson model)%s.", st_fmt(disp, 3),
            if (family != "poisson") "; the simpler Poisson model would also be adequate" else "")
  }
  aui_text(advice, title = "Overdispersion check")

  # ---- coefficients and rate ratios
  ci <- st_glm_ci(fit, alpha, aui_opt(options, "ci_method", "profile"))
  ci_name <- if (ci$method == "profile") "profile-likelihood" else "Wald"
  ctab <- data.frame(Term = labels, est = co[, 1], se = co[, 2], stat = co[, 3], p = co[, 4], stars = aui_stars(co[, 4]),
                     lower = ci$ci[, 1], upper = ci$ci[, 2], stringsAsFactors = FALSE)
  aui_table("Coefficients (log scale)", ctab, formats = c(p = "p", stars = "stars"),
            labels = c(est = "Estimate (log rate ratio)", se = "SE", stat = stat_name, p = "P value", stars = "Summary",
                       lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
            footnotes = c(paste0("Wald ", stat_name, " tests; ", ci_name, " confidence intervals."),
                          st_reference_note(data, intersect(factors, all.vars(form)), names)))
  irr <- st_ratio_table(labels, co[, 1], co[, 2], co[, 3], co[, 4], ci$ci)
  aui_table("Incidence rate ratios", irr, formats = c(p = "p", stars = "stars"),
            labels = c(ratio = "Rate ratio (IRR)", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper"),
                       p = "P value", stars = "Summary"),
            footnotes = paste0("IRR = exp(coefficient), with ", ci_name, " confidence intervals: the multiplicative change in the expected ",
                               if (has_offset) "rate" else "count", " per one-unit increase of a continuous predictor, or relative to the reference level of a categorical predictor. 1 means no association."))

  if (aui_flag(options, "show_plot", TRUE)) {
    d <- data.frame(pred = stats::fitted(fit), y = data$y)
    pl <- ggplot2::ggplot(d, ggplot2::aes(pred, y)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, colour = "#7A8B99", linetype = "dashed") +
      ggplot2::geom_point(colour = AUI_COLORS[1], alpha = 0.75, size = 1.9) +
      ggplot2::labs(x = "Predicted count", y = paste("Observed", yname)) + aui_theme()
    aui_plot("Observed vs predicted counts (dashed: line of identity)", pl, width = 4.8, height = 4.2)
  }

  sig <- irr[irr$p < alpha, , drop = FALSE]
  txt <- sprintf("%s regression of %s (n = %d).%s", fam_name, yname, n, lr_txt)
  txt <- paste0(txt, if (nrow(sig)) paste0(" Significant predictors: ", paste(sprintf("%s (IRR %s, %s CI %s to %s)", sig$Term, st_fmt(sig$ratio, 3), st_pct(alpha),
                                                                                    st_fmt(sig$lower, 3), st_fmt(sig$upper, 3)), collapse = "; "), ".")
                     else " No individual predictor is statistically significant.")
  if (family == "poisson" && disp > 1.5) txt <- paste0(txt, " Caution: the counts are overdispersed, so these P values are too small; use quasi-Poisson or negative binomial.")
  aui_summary(txt)
}
