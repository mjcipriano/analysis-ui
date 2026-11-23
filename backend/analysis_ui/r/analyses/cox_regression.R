# Cox proportional hazards regression.
# data: time, status (0/1), x1.., f1.., optional strata (factor)
run <- function(data, options) {
  aui_need("ggplot2", "survival")
  names <- st_names(options)
  alpha <- st_alpha(options)
  covs <- aui_vars(options, "_covariates")
  factors <- aui_vars(options, "_factors")
  has_strata <- isTRUE(options[["_has_strata"]])
  ties <- if (identical(aui_opt(options, "ties", "efron"), "breslow")) "breslow" else "efron"
  tname <- st_name(names, "time")

  rhs <- st_rhs(c(covs, factors), aui_opt(options, "interactions", "none"), options[["_rhs"]])
  env <- new.env(parent = globalenv())
  env$Surv <- survival::Surv; env$strata <- survival::strata
  form <- stats::as.formula(paste("Surv(time, status) ~", rhs, if (has_strata) "+ strata(strata)" else ""), env = env)
  q <- st_quiet(st_try(st_with_treatment_contrasts(survival::coxph(form, data = data, ties = ties, x = TRUE, model = TRUE)),
                       "The Cox model could not be fitted. Check the custom model terms."))
  fit <- q$value
  n <- fit$n; nev <- fit$nevent
  mm <- stats::model.matrix(fit)
  plain_terms <- stats::delete.response(stats::terms(stats::as.formula(paste("~", rhs))))
  labels <- tryCatch({
    lm_mm <- st_with_treatment_contrasts(stats::model.matrix(plain_terms, data))
    lab <- st_coef_labels(plain_terms, lm_mm, data, names)
    lab <- lab[colnames(lm_mm) != "(Intercept)"]
    if (length(lab) == length(stats::coef(fit))) lab else st_pretty(names(stats::coef(fit)), names)
  }, error = function(e) st_pretty(names(stats::coef(fit)), names))
  st_check_aliased(stats::coef(fit), labels)
  s <- summary(fit, conf.int = 1 - alpha)
  co <- s$coefficients; cint <- s$conf.int
  k <- length(stats::coef(fit))
  if (any(grepl("converged before|Ran out of iterations|infinite", q$warnings)) || any(co[, "se(coef)"] > 100)) {
    aui_warn(paste0("At least one hazard ratio is estimated as zero or infinite (monotone likelihood): in some group no events occurred, or a predictor separates the subjects with and without events perfectly. ",
                    "The Wald tests and confidence intervals for the affected terms are not trustworthy; rely on the likelihood-ratio test, or merge or remove the offending category."))
  }
  if (nev / max(1, k) < 10) {
    aui_warn(sprintf("Few events per predictor: %d events for %d model coefficient%s (%.1f per coefficient). With fewer than about 10 the estimates tend to be unstable; interpret with caution or simplify the model.",
                     as.integer(nev), as.integer(k), if (k == 1) "" else "s", nev / k))
  }

  conc <- s$concordance
  aui_table("Model summary", data.frame(
    Measure = c("Observations", "Events", "Censored", "Concordance (Harrell's C)", "SE of concordance",
                "Likelihood-ratio test", "Wald test", "Score (log-rank) test"),
    value = c(n, nev, n - nev, conc[1], conc[2], s$logtest[1], s$waldtest[1], s$sctest[1]),
    df = c(NA, NA, NA, NA, NA, s$logtest[2], s$waldtest[2], s$sctest[2]),
    p = c(NA, NA, NA, NA, NA, s$logtest[3], s$waldtest[3], s$sctest[3]), stringsAsFactors = FALSE),
    formats = c(df = "int", p = "p"), labels = c(value = "Value / χ²", p = "P value"),
    footnotes = c(paste0("Cox proportional hazards regression; ", if (ties == "efron") "Efron" else "Breslow", " method for tied event times",
                         if (has_strata) paste0("; stratified by ", st_name(names, "strata"), " (separate baseline hazard in each stratum)") else "",
                         ". Model: ", tname, " ~ ", st_pretty(rhs, names), "."),
                  "The three global tests ask whether all coefficients are zero (no predictor is related to the hazard). Concordance is the probability that, of two subjects, the one with the higher predicted risk has the event first (0.5 = chance, 1 = perfect).",
                  st_dropped_note(options)))

  hr <- data.frame(Term = labels, hr = cint[, 1], lower = cint[, 3], upper = cint[, 4], coef = co[, "coef"],
                   se = co[, "se(coef)"], z = co[, "z"], p = co[, "Pr(>|z|)"], stars = aui_stars(co[, "Pr(>|z|)"]),
                   stringsAsFactors = FALSE)
  aui_table("Hazard ratios", hr, formats = c(p = "p", stars = "stars"),
            labels = c(hr = "Hazard ratio", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper"),
                       coef = "Coefficient (log HR)", se = "SE", p = "P value", stars = "Summary"),
            footnotes = c("Hazard ratio = exp(coefficient) with Wald confidence intervals and z tests. HR > 1 means a higher event rate (shorter survival) per one-unit increase of a continuous predictor, or relative to the reference level of a categorical predictor; each is adjusted for the other predictors.",
                          st_reference_note(data, intersect(factors, all.vars(stats::as.formula(paste("~", rhs)))), names)))

  ph_txt <- ""
  if (aui_flag(options, "ph_test", TRUE)) {
    zph <- tryCatch(st_quiet(survival::cox.zph(fit))$value, error = function(e) NULL)
    if (is.null(zph)) {
      aui_text("The proportional-hazards test could not be computed for this model.", title = "Proportional hazards")
    } else {
      tb <- as.data.frame(zph$table)
      rn <- rownames(tb)
      tb_p <- tb$p
      glob <- tb_p[rn == "GLOBAL"]
      viol <- rn[rn != "GLOBAL" & !is.na(tb_p) & tb_p < alpha]
      reading <- if (!length(viol) && (length(glob) == 0 || is.na(glob) || glob >= alpha))
        "No evidence against proportional hazards: the hazard ratios appear to be reasonably constant over time."
      else paste0("Evidence that the hazard ratio changes over time",
                  if (length(viol)) paste0(" for ", paste(st_pretty(viol, names), collapse = ", ")) else " (global test)",
                  ". A single hazard ratio is then an average over follow-up; consider stratifying by that variable or modelling a time-varying effect.")
      aui_table("Test of the proportional-hazards assumption", data.frame(
        Term = ifelse(rn == "GLOBAL", "Global", st_pretty(rn, names)), chi2 = tb$chisq, df = tb$df, p = tb_p, stringsAsFactors = FALSE),
        formats = c(p = "p"), labels = c(chi2 = "χ²", p = "P value"),
        footnotes = c("Grambsch–Therneau test based on scaled Schoenfeld residuals (survival::cox.zph, Kaplan–Meier time transform). A small P value is evidence that the effect of that term changes over time.",
                      reading))
      ph_txt <- paste0(" ", sub("\\..*$", ".", reading))
    }
  }

  if (aui_flag(options, "forest", TRUE)) {
    d <- hr[is.finite(hr$hr) & is.finite(hr$lower) & is.finite(hr$upper) & hr$upper < 1e6 & hr$lower > 1e-6, , drop = FALSE]
    if (nrow(d)) {
      d$Term <- factor(d$Term, levels = rev(unique(d$Term)))
      pl <- ggplot2::ggplot(d, ggplot2::aes(hr, Term)) +
        ggplot2::geom_vline(xintercept = 1, linetype = "dashed", colour = "#7A8B99") +
        ggplot2::geom_errorbar(ggplot2::aes(xmin = lower, xmax = upper), width = 0.18, colour = AUI_COLORS[1], linewidth = 0.7, orientation = "y") +
        ggplot2::geom_point(size = 2.8, shape = 15, colour = AUI_COLORS[1]) +
        ggplot2::scale_x_log10() +
        ggplot2::labs(x = paste0("Hazard ratio (", st_pct(alpha), " CI, log scale)"), y = NULL) + aui_theme()
      aui_plot("Forest plot of hazard ratios", pl, width = 6.2, height = max(2.2, 1.2 + 0.42 * nrow(d)))
    }
  }

  used <- all.vars(stats::as.formula(paste("~", rhs)))
  if (aui_flag(options, "curves", TRUE) && length(intersect(factors, used)) && !has_strata) {
    f1 <- intersect(factors, used)[1]
    lv <- levels(data[[f1]])
    nd <- data.frame(row.names = seq_along(lv))
    for (v in used) {
      nd[[v]] <- if (v == f1) factor(lv, levels = lv)
                 else if (is.factor(data[[v]])) factor(rep(levels(data[[v]])[1], length(lv)), levels = levels(data[[v]]))
                 else rep(mean(data[[v]]), length(lv))
    }
    sf <- tryCatch(st_quiet(survival::survfit(fit, newdata = nd))$value, error = function(e) NULL)
    if (!is.null(sf)) {
      surv <- matrix(sf$surv, ncol = length(lv))
      d <- do.call(rbind, lapply(seq_along(lv), function(j) data.frame(time = c(0, sf$time), surv = c(1, surv[, j]), level = lv[j])))
      d$level <- factor(d$level, levels = lv)
      others <- setdiff(used, f1)
      pl <- ggplot2::ggplot(d, ggplot2::aes(time, surv, colour = level)) +
        ggplot2::geom_step(linewidth = 0.9) +
        ggplot2::scale_colour_manual(values = st_colors(length(lv)), name = st_name(names, f1)) +
        ggplot2::scale_y_continuous(limits = c(0, 1), labels = function(v) paste0(100 * v, "%")) +
        ggplot2::labs(x = tname, y = "Predicted survival") + aui_theme() +
        ggplot2::theme(legend.title = ggplot2::element_text())
      aui_plot(paste0("Adjusted survival curves by ", st_name(names, f1),
                      if (length(others)) " (other predictors at their mean / reference level)" else " (model-based)"),
               pl, width = 6, height = 4.2)
    }
  }

  sig <- hr[!is.na(hr$p) & hr$p < alpha, , drop = FALSE]
  txt <- sprintf("Cox regression with %d subjects and %d events: the predictors are %s related to the hazard overall (likelihood-ratio χ² = %s, df = %d, %s; concordance = %s).",
                 as.integer(n), as.integer(nev), if (s$logtest[3] < alpha) "significantly" else "not significantly",
                 formatC(s$logtest[1], format = "f", digits = 2), as.integer(s$logtest[2]), st_p_text(s$logtest[3]),
                 formatC(conc[1], format = "f", digits = 3))
  txt <- paste0(txt, if (nrow(sig)) paste0(" Significant predictors: ", paste(sprintf("%s (HR %s, %s CI %s to %s)", sig$Term, st_fmt(sig$hr, 3), st_pct(alpha),
                                                                                    st_fmt(sig$lower, 3), st_fmt(sig$upper, 3)), collapse = "; "), ".")
                     else " No individual predictor is statistically significant.", ph_txt)
  aui_summary(txt)
  if (length(factors)) {
    aui_graph("Create a Kaplan–Meier graph of these data", "survival",
              list(time = tname, event = st_name(names, "status"), event_value = aui_opt(options, "event_value", "1"),
                   group = st_name(names, factors[1])))
  }
}
