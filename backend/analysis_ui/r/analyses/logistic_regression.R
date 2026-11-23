# Binary logistic regression.
# data: y (0/1), x1.., f1..

# Hosmer-Lemeshow test with g groups of (nearly) equal size.
.hosmer_lemeshow <- function(y, p, g = 10) {
  n <- length(y)
  g <- min(g, floor(n / 2), length(unique(p)))
  if (g < 3) return(NULL)
  grp <- cut(rank(p, ties.method = "first"), breaks = seq(0, n, length.out = g + 1), labels = FALSE)
  obs <- tapply(y, grp, sum); expd <- tapply(p, grp, sum); ng <- tapply(y, grp, length)
  chi2 <- sum((obs - expd)^2 / (expd * (1 - expd / ng)))
  list(chi2 = chi2, df = g - 2, p = stats::pchisq(chi2, g - 2, lower.tail = FALSE), g = g,
       sparse = min(c(expd, ng - expd)) < 1)
}

.separation_message <- function(warns, se) {
  sep <- any(grepl("fitted probabilities numerically 0 or 1|did not converge", warns)) || any(se > 100, na.rm = TRUE)
  if (!sep) return(FALSE)
  aui_warn(paste0("Complete or quasi-complete separation: some predictor (or combination of predictors) predicts the outcome perfectly, ",
                  "so the affected coefficients drift towards infinity and their standard errors, P values and confidence intervals are not trustworthy. ",
                  "Look for coefficients with huge values or standard errors. Remedies: remove or merge the offending predictor/categories, or collect more data."))
  TRUE
}

run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  covs <- aui_vars(options, "_covariates")
  factors <- aui_vars(options, "_factors")
  pos <- options[["_positive"]]; neg <- options[["_negative"]]
  yname <- st_name(names, "y")

  rhs <- st_rhs(c(covs, factors), aui_opt(options, "interactions", "none"), options[["_rhs"]])
  form <- stats::as.formula(paste("y ~", rhs))
  q <- st_quiet(st_try(st_with_treatment_contrasts(stats::glm(form, data = data, family = stats::binomial())),
                       "The model could not be fitted. Check the custom model terms."))
  fit <- q$value
  mm <- stats::model.matrix(fit)
  labels <- st_coef_labels(stats::terms(fit), mm, data, names)
  st_check_aliased(stats::coef(fit), labels)
  n <- nrow(data); n1 <- sum(data$y); n0 <- n - n1
  k <- length(stats::coef(fit)) - 1
  if (stats::df.residual(fit) < 1) aui_stop("The model has as many coefficients as observations. Use fewer predictors or more data.")
  co <- stats::coef(summary(fit))
  separated <- .separation_message(q$warnings, co[, 2])
  epv <- min(n1, n0) / max(1, k)
  if (epv < 10) {
    aui_warn(sprintf("Few events per predictor: the rarer outcome occurs %d times for %d model coefficient%s (%.1f per coefficient). With fewer than about 10, estimates tend to be unstable and too extreme; interpret with caution or simplify the model.",
                     as.integer(min(n1, n0)), as.integer(k), if (k == 1) "" else "s", epv))
  }

  # ---- model summary
  null <- stats::glm(y ~ 1, data = data, family = stats::binomial())
  ll <- as.numeric(stats::logLik(fit)); ll0 <- as.numeric(stats::logLik(null))
  lr <- 2 * (ll - ll0); lr_p <- stats::pchisq(lr, k, lower.tail = FALSE)
  phat <- stats::fitted(fit)
  mcfadden <- 1 - ll / ll0
  cox_snell <- 1 - exp(-lr / n)
  nagelkerke <- cox_snell / (1 - exp(2 * ll0 / n))
  tjur <- mean(phat[data$y == 1]) - mean(phat[data$y == 0])
  aui_table("Model summary", data.frame(
    Measure = c("Observations", paste0("Outcome = ", pos, " (modelled)"), paste0("Outcome = ", neg),
                sprintf("Likelihood-ratio \u03c7\u00b2 vs intercept-only model (df = %d)", as.integer(k)),
                "McFadden's pseudo-R\u00b2", "Tjur's R\u00b2 (coefficient of discrimination)", "Nagelkerke's pseudo-R\u00b2",
                "Log-likelihood", "AIC"),
    value = c(n, n1, n0, lr, mcfadden, tjur, nagelkerke, ll, stats::AIC(fit)),
    p = c(rep(NA, 3), lr_p, rep(NA, 5)), stringsAsFactors = FALSE),
    formats = c(p = "p"), labels = c(value = "Value", p = "P value"),
    footnotes = c(paste0("Binary logistic regression (maximum likelihood) for the probability that ", yname, " = ", pos, ". Model: ",
                         st_formula_text(form, names), "."),
                  "Pseudo-R\u00b2 values are not the fraction of variance explained and are typically much lower than R\u00b2 of a linear regression.",
                  st_dropped_note(options)))

  # ---- coefficients and odds ratios
  ci <- st_glm_ci(fit, alpha, if (separated) "wald" else aui_opt(options, "ci_method", "profile"))
  ci_name <- if (ci$method == "profile") "profile-likelihood" else "Wald"
  aui_table("Coefficients (log-odds)", data.frame(
    Term = labels, est = co[, 1], se = co[, 2], z = co[, 3], p = co[, 4], stars = aui_stars(co[, 4]),
    lower = ci$ci[, 1], upper = ci$ci[, 2], stringsAsFactors = FALSE),
    formats = c(p = "p", stars = "stars"),
    labels = c(est = "Estimate (log-odds)", se = "SE", p = "P value", stars = "Summary",
               lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
    footnotes = c(paste0("Wald z tests; ", ci_name, " confidence intervals."), st_reference_note(data, intersect(factors, all.vars(form)), names)))
  or <- st_ratio_table(labels, co[, 1], co[, 2], co[, 3], co[, 4], ci$ci)
  aui_table("Odds ratios", or, formats = c(p = "p", stars = "stars"),
            labels = c(ratio = "Odds ratio", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper"),
                       p = "P value", stars = "Summary"),
            footnotes = c(paste0("Odds ratio = exp(coefficient), with ", ci_name, " confidence intervals. For a continuous predictor it is the multiplicative change in the odds of ",
                                 yname, " = ", pos, " per one-unit increase; for a categorical predictor it compares a level with the reference level. Each is adjusted for the other predictors."),
                          "An odds ratio of 1 means no association."))

  # ---- goodness of fit
  if (aui_flag(options, "hosmer", TRUE)) {
    hl <- .hosmer_lemeshow(data$y, phat)
    if (is.null(hl)) {
      aui_text("Too few distinct predicted probabilities for the Hosmer\u2013Lemeshow test.", title = "Hosmer\u2013Lemeshow test")
    } else {
      aui_table("Hosmer\u2013Lemeshow goodness-of-fit test", data.frame(
        chi2 = hl$chi2, df = hl$df, p = hl$p, stringsAsFactors = FALSE),
        formats = c(df = "int", p = "p"), labels = c(chi2 = "\u03c7\u00b2", p = "P value"),
        footnotes = c(sprintf("Observed and predicted numbers of events are compared in %d groups of similar predicted probability. A small P value is evidence of poor calibration; a large P value does not prove that the model is correct (the test has low power in small samples).", hl$g),
                      if (hl$sparse) "Some groups have an expected count below 1, so the \u03c7\u00b2 approximation is unreliable here; do not over-interpret this P value."))
    }
  }

  # ---- classification
  auc_txt <- ""
  if (aui_flag(options, "classification", TRUE)) {
    cut <- aui_num(options, "cutoff", 0.5)
    if (!is.finite(cut)) cut <- 0.5
    pred <- as.integer(phat >= cut)
    tp <- sum(pred == 1 & data$y == 1); fn <- sum(pred == 0 & data$y == 1)
    fp <- sum(pred == 1 & data$y == 0); tn <- sum(pred == 0 & data$y == 0)
    aui_table(sprintf("Classification table (cut-off %s)", format(cut)), data.frame(
      Observed = c(pos, neg), p1 = c(tp, fp), p0 = c(fn, tn), correct = c(tp / (tp + fn), tn / (tn + fp)),
      stringsAsFactors = FALSE),
      formats = c(p1 = "int", p0 = "int", correct = "pct"),
      labels = c(p1 = paste0("Predicted ", pos), p0 = paste0("Predicted ", neg), correct = "% correct"),
      footnotes = c(sprintf("Sensitivity %.1f%%, specificity %.1f%%, positive predictive value %s, negative predictive value %s, overall accuracy %.1f%%.",
                            100 * tp / (tp + fn), 100 * tn / (tn + fp),
                            if (tp + fp > 0) sprintf("%.1f%%", 100 * tp / (tp + fp)) else "n/a",
                            if (tn + fn > 0) sprintf("%.1f%%", 100 * tn / (tn + fn)) else "n/a", 100 * (tp + tn) / n),
                    "Evaluated on the same data that were used to fit the model, so performance on new data will usually be somewhat lower."))
  }
  if (aui_flag(options, "roc", TRUE) && requireNamespace("pROC", quietly = TRUE)) {
    roc <- st_quiet(pROC::roc(response = data$y, predictor = phat, levels = c(0, 1), direction = "<"))$value
    auc <- as.numeric(pROC::auc(roc))
    aci <- tryCatch(as.numeric(st_quiet(pROC::ci.auc(roc, conf.level = 1 - alpha))$value), error = function(e) c(NA, auc, NA))
    aui_table("ROC curve", data.frame(auc = auc, lower = aci[1], upper = aci[3], stringsAsFactors = FALSE),
              labels = c(auc = "Area under the curve (AUC)", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
              footnotes = "AUC of the predicted probabilities (0.5 = no discrimination, 1 = perfect); DeLong confidence interval. Computed on the data used to fit the model.")
    auc_txt <- sprintf(" The model discriminates with an AUC of %s.", formatC(auc, format = "f", digits = 3))
    d <- data.frame(fpr = 1 - roc$specificities, tpr = roc$sensitivities)
    d <- d[order(d$fpr, d$tpr), ]
    pl <- ggplot2::ggplot(d, ggplot2::aes(fpr, tpr)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, colour = "#7A8B99", linetype = "dashed") +
      ggplot2::geom_step(colour = AUI_COLORS[1], linewidth = 0.9) +
      ggplot2::coord_equal() +
      ggplot2::labs(x = "1 \u2212 specificity (false-positive rate)", y = "Sensitivity (true-positive rate)",
                    title = sprintf("AUC = %s", formatC(auc, format = "f", digits = 3))) + aui_theme()
    aui_plot("ROC curve", pl, width = 4.4, height = 4.4)
  }

  # ---- predicted probability for a single continuous predictor
  if (aui_flag(options, "show_plot", TRUE) && length(covs) == 1 && !length(factors) && identical(all.vars(form), c("y", covs))) {
    xs <- seq(min(data[[covs]]), max(data[[covs]]), length.out = 200)
    nd <- stats::setNames(data.frame(xs), covs)
    pr <- stats::predict(fit, nd, type = "link", se.fit = TRUE)
    z <- stats::qnorm(1 - alpha / 2)
    curve <- data.frame(x = xs, p = stats::plogis(pr$fit), lo = stats::plogis(pr$fit - z * pr$se.fit), hi = stats::plogis(pr$fit + z * pr$se.fit))
    pts <- data.frame(x = data[[covs]], y = data$y)
    pl <- ggplot2::ggplot(curve, ggplot2::aes(x, p)) +
      ggplot2::geom_ribbon(ggplot2::aes(ymin = lo, ymax = hi), fill = AUI_COLORS[1], alpha = 0.18) +
      ggplot2::geom_line(colour = AUI_COLORS[1], linewidth = 0.9) +
      ggplot2::geom_point(data = pts, ggplot2::aes(x, y), alpha = 0.5, size = 1.8, colour = "#444444",
                          position = ggplot2::position_jitter(width = 0, height = 0.025, seed = 1)) +
      ggplot2::scale_y_continuous(limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25)) +
      ggplot2::labs(x = st_name(names, covs), y = paste0("Probability of ", yname, " = ", pos)) + aui_theme()
    aui_plot(paste0("Predicted probability with ", st_pct(alpha), " confidence band"), pl, width = 5.4, height = 4)
  }

  sig <- or$Term[or$p < alpha]
  txt <- sprintf("Logistic regression for %s = %s (n = %d, %d events): the model %s the intercept-only model (likelihood-ratio \u03c7\u00b2 = %s, df = %d, %s).",
                 yname, pos, n, as.integer(n1), if (lr_p < alpha) "fits significantly better than" else "does not fit significantly better than",
                 formatC(lr, format = "f", digits = 2), as.integer(k), st_p_text(lr_p))
  if (length(sig)) {
    best <- or[or$p < alpha, , drop = FALSE]
    txt <- paste0(txt, " Significant predictors: ", paste(sprintf("%s (OR %s, %s CI %s to %s)", best$Term, st_fmt(best$ratio, 3), st_pct(alpha),
                                                                 st_fmt(best$lower, 3), st_fmt(best$upper, 3)), collapse = "; "), ".")
  } else txt <- paste0(txt, " No individual predictor is statistically significant.")
  if (separated) txt <- paste0(txt, " Caution: the data show (quasi-)complete separation, so the Wald P values and confidence intervals of the affected predictors are not reliable.")
  aui_summary(paste0(txt, auc_txt))
}
