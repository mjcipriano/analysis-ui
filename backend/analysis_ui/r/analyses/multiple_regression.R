# Multiple linear regression.
# data: y, x1.. (continuous predictors), f1.. (categorical predictors), optional w (weights), row_id
run <- function(data, options) {
  aui_need("ggplot2", "car")
  names <- st_names(options)
  alpha <- st_alpha(options)
  covs <- aui_vars(options, "_covariates")
  factors <- aui_vars(options, "_factors")
  weighted <- isTRUE(options[["_weighted"]])
  intercept <- aui_flag(options, "intercept", TRUE)
  ylab <- st_name(names, "y")

  rhs <- st_rhs(c(covs, factors), aui_opt(options, "interactions", "none"), options[["_rhs"]])
  form <- stats::as.formula(paste("y ~", rhs, if (!intercept) "- 1" else ""))
  fit_model <- function() if (weighted) stats::lm(form, data = data, weights = w) else stats::lm(form, data = data)
  fit <- st_try(st_with_treatment_contrasts(fit_model()),
                "The model could not be fitted. Check the custom model terms (for example, a categorical predictor cannot be squared or log-transformed).")
  mm <- stats::model.matrix(fit)
  labels <- st_coef_labels(stats::terms(fit), mm, data, names)
  if (length(stats::coef(fit)) != length(labels)) labels <- st_pretty(names(stats::coef(fit)), names)
  st_check_aliased(stats::coef(fit), labels)
  n <- nrow(data); p_coef <- length(stats::coef(fit))
  if (stats::df.residual(fit) < 1) {
    aui_stop("The model has as many coefficients (", p_coef, ") as there are complete rows (", n,
             "), so nothing is left to estimate the error. Use fewer predictors or interactions, or more data.")
  }
  res <- stats::residuals(fit)
  ss_res <- sum((if (weighted) data$w else 1) * res^2)
  ss_tot <- sum((if (weighted) data$w else 1) * (data$y - if (intercept) stats::weighted.mean(data$y, if (weighted) data$w else rep(1, n)) else 0)^2)
  if (ss_res <= ss_tot * 1e-20) {
    aui_stop("The predictors reproduce the outcome exactly (the residual variation is zero), so standard errors and P values cannot be computed. ",
             "Check whether a predictor is a copy or an exact function of the outcome.")
  }
  s <- summary(fit)
  fs <- s$fstatistic
  p_model <- if (is.null(fs)) NA else stats::pf(fs[1], fs[2], fs[3], lower.tail = FALSE)

  aui_table("Model summary", data.frame(
    Measure = c("Observations", "R²", "Adjusted R²", "Residual SD (Sy.x)", "RMSE", "F", "df (model, residual)", "P value (overall model)", "AIC", "BIC"),
    value = c(as.character(n), st_fmt(s$r.squared), st_fmt(s$adj.r.squared), st_fmt(s$sigma), st_fmt(sqrt(mean(res^2))),
              if (is.null(fs)) "" else st_fmt(fs[1]), if (is.null(fs)) "" else sprintf("%d, %d", as.integer(fs[2]), as.integer(fs[3])),
              aui_fmt_p(p_model), st_fmt(stats::AIC(fit), 5), st_fmt(stats::BIC(fit), 5)),
    stringsAsFactors = FALSE),
    labels = c(value = "Value"),
    footnotes = c(paste0("Ordinary least squares", if (weighted) paste0(" weighted by ", st_name(names, "w")) else "", ". Model: ", st_formula_text(form, names), "."),
                  if (!intercept) "The model has no intercept: R² is computed about zero rather than about the mean and is not comparable with R² of models that have an intercept.",
                  st_dropped_note(options)))

  # ---- ANOVA table
  type <- if (identical(as.character(aui_opt(options, "ss_type", "2")), "3")) 3L else 2L
  if (aui_flag(options, "anova_table", TRUE) && length(attr(stats::terms(fit), "term.labels"))) {
    a <- NULL
    if (type == 3L) {
      a <- tryCatch(as.data.frame(car::Anova(st_with_sum_contrasts(fit_model()), type = 3)), error = function(e) NULL)
      if (is.null(a)) type <- 2L
    }
    if (is.null(a)) a <- tryCatch(as.data.frame(car::Anova(fit, type = 2)), error = function(e) NULL)
    if (!is.null(a)) {
      a <- a[rownames(a) != "(Intercept)", , drop = FALSE]
      is_res <- rownames(a) == "Residuals"
      ssr <- a$`Sum Sq`[is_res]
      aui_table("ANOVA table", data.frame(
        Source = ifelse(is_res, "Residual", st_pretty(rownames(a), names)), SS = a$`Sum Sq`, df = a$Df,
        MS = a$`Sum Sq` / a$Df, F = a$`F value`, p = a$`Pr(>F)`, stars = ifelse(is_res, "", aui_stars(a$`Pr(>F)`)),
        eta_p = ifelse(is_res, NA, a$`Sum Sq` / (a$`Sum Sq` + ssr)), stringsAsFactors = FALSE),
        formats = c(df = "int", p = "p", stars = "stars"),
        labels = c(p = "P value", stars = "Summary", eta_p = "Partial η²"),
        footnotes = paste0("Type ", if (type == 3) "III sums of squares (each term adjusted for all others; sum-to-zero contrasts)."
                           else "II sums of squares (each term adjusted for all others except interactions that contain it)."))
    }
  }

  # ---- coefficients
  co <- stats::coef(s)
  crit <- stats::qt(1 - alpha / 2, stats::df.residual(fit))
  tab <- data.frame(Term = labels, est = co[, 1], se = co[, 2], t = co[, 3], p = co[, 4],
                    lower = co[, 1] - crit * co[, 2], upper = co[, 1] + crit * co[, 2], stringsAsFactors = FALSE)
  notes <- c(st_reference_note(data, intersect(factors, all.vars(form)), names))
  if (aui_flag(options, "standardized", TRUE) && !weighted) {
    sdx <- apply(mm, 2, stats::sd)
    tab$beta <- ifelse(colnames(mm) == "(Intercept)", NA, co[, 1] * sdx / stats::sd(data$y))
    notes <- c(notes, "Standardized beta = coefficient × SD(predictor) / SD(outcome): the change in the outcome, in SDs, per one-SD change in the predictor.")
  }
  if (aui_flag(options, "vif", TRUE) && intercept && ncol(mm) > 2) {
    tab$vif <- st_vif(mm)
    notes <- c(notes, "VIF (variance inflation factor) above about 5–10 signals strong collinearity among predictors. Interaction and polynomial terms inflate VIFs by construction.")
    if (any(tab$vif > 10, na.rm = TRUE) && identical(aui_opt(options, "interactions", "none"), "none")) aui_warn("Some predictors are strongly collinear (VIF > 10): their coefficients are estimated imprecisely and can change a lot when other predictors are added or removed.")
  }
  aui_table("Coefficients", tab, formats = c(p = "p"),
            labels = c(est = "Estimate (B)", se = "SE", p = "P value", lower = paste(st_pct(alpha), "CI lower"),
                       upper = paste(st_pct(alpha), "CI upper"), beta = "Standardized β", vif = "VIF"),
            footnotes = notes)

  # ---- assumption checks
  h <- stats::hatvalues(fit)
  rstd <- suppressWarnings(stats::rstandard(fit)); rstu <- suppressWarnings(stats::rstudent(fit))
  cook <- suppressWarnings(stats::cooks.distance(fit))
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    st_emit_normality(if (weighted) res * sqrt(data$w) else res, alpha)
    bp <- tryCatch(car::ncvTest(fit), error = function(e) NULL)
    if (!is.null(bp)) {
      aui_table("Constant variance (homoscedasticity)", data.frame(
        Test = "Breusch–Pagan (score test)", chi2 = bp$ChiSquare, df = bp$Df, p = bp$p, stringsAsFactors = FALSE),
        formats = c(df = "int", p = "p"), labels = c(chi2 = "χ²", p = "P value"),
        footnotes = "Tests whether the residual variance changes with the fitted values (car::ncvTest). A small P value is evidence of heteroscedasticity.")
    }
  }
  if (aui_flag(options, "diag_plots", TRUE)) {
    d <- data.frame(fitted = stats::fitted(fit), res = res, rstd = rstd, cook = cook, idx = seq_len(n))
    d <- d[is.finite(d$rstd), , drop = FALSE]
    p1 <- st_resid_plot(d$fitted, d$res)
    p2 <- st_qq_plot(d$rstd, "Normal QQ plot") + ggplot2::labs(y = "Standardized residuals")
    p3 <- st_resid_plot(d$fitted, sqrt(abs(d$rstd)), "Scale–location", expression(sqrt("|standardized residual|")))
    p3$layers[[1]] <- NULL
    p4 <- ggplot2::ggplot(d, ggplot2::aes(idx, cook)) +
      ggplot2::geom_segment(ggplot2::aes(xend = idx, yend = 0), colour = AUI_COLORS[1]) +
      ggplot2::geom_hline(yintercept = 4 / n, linetype = "dashed", colour = AUI_COLORS[2]) +
      ggplot2::labs(x = "Observation (row of the analysed data)", y = "Cook's distance", title = "Cook's distance (dashed: 4/n)") + aui_theme()
    if (requireNamespace("patchwork", quietly = TRUE)) {
      aui_plot("Diagnostic plots", patchwork::wrap_plots(p1, p2, p3, p4, ncol = 2), width = 8.6, height = 7)
    } else {
      aui_plot("Residuals vs fitted values", p1); aui_plot("Normal QQ plot", p2)
      aui_plot("Scale–location", p3); aui_plot("Cook's distance", p4)
    }
  }
  if (aui_flag(options, "influence")) {
    flag <- which((is.finite(cook) & cook > 4 / n) | (is.finite(rstu) & abs(rstu) > 3) | h > 3 * p_coef / n)
    if (length(flag)) {
      flag <- flag[order(-cook[flag])]
      aui_table("Influential points", data.frame(
        row = data$row_id[flag] %||% flag, y = data$y[flag], fitted = stats::fitted(fit)[flag], res = res[flag],
        rstu = rstu[flag], lev = h[flag], cook = cook[flag], stringsAsFactors = FALSE),
        formats = c(row = "int"),
        labels = c(row = "Row", y = ylab, fitted = "Predicted", res = "Residual", rstu = "Studentized residual",
                   lev = "Leverage", cook = "Cook's distance"),
        footnotes = sprintf("Rows with Cook's distance > 4/n (%s), |studentized residual| > 3 or leverage > 3p/n (%s). Influential points are not necessarily errors; check them and report the analysis with and without them if they matter.",
                            st_fmt(4 / n, 3), st_fmt(3 * p_coef / n, 3)))
    } else {
      aui_text("No point exceeds the usual thresholds (Cook's distance > 4/n, |studentized residual| > 3, leverage > 3p/n).", title = "Influential points")
    }
  }

  if (aui_flag(options, "show_plot", TRUE)) {
    d <- data.frame(pred = stats::fitted(fit), y = data$y)
    pl <- ggplot2::ggplot(d, ggplot2::aes(pred, y)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, colour = "#7A8B99", linetype = "dashed") +
      ggplot2::geom_point(colour = AUI_COLORS[1], alpha = 0.8, size = 1.9) +
      ggplot2::labs(x = paste("Predicted", ylab), y = paste("Actual", ylab)) + aui_theme()
    aui_plot("Actual vs predicted (dashed: line of identity)", pl, width = 4.8, height = 4.2)
  }

  sig <- which(tab$p < alpha & tab$Term != "Intercept")
  txt <- sprintf("The model explains %s%% of the variation in %s (R² = %s, adjusted R² = %s)%s.",
                 formatC(100 * s$r.squared, format = "f", digits = 1), ylab, formatC(s$r.squared, format = "f", digits = 3),
                 formatC(s$adj.r.squared, format = "f", digits = 3),
                 if (is.null(fs)) "" else sprintf("; overall %s, %s", st_fmt_f(fs[1], fs[2], fs[3]), st_p_text(p_model)))
  txt <- paste(txt, if (length(sig)) paste0("Coefficients significantly different from zero (P < ", format(alpha), "): ",
                                           paste(tab$Term[sig], collapse = ", "), ".")
                    else "No individual coefficient is significantly different from zero.")
  aui_summary(txt)
}
