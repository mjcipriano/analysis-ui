# Factorial (N-way) ANOVA / ANCOVA.
# data: y, f1..fk (factors), x1..xm (covariates)
run <- function(data, options) {
  aui_need("ggplot2", "car", "emmeans")
  names <- st_names(options)
  alpha <- st_alpha(options)
  factors <- aui_vars(options, "_factors")
  covs <- aui_vars(options, "_covariates")
  model <- aui_opt(options, "model", "full")
  type <- if (identical(as.character(aui_opt(options, "ss_type", "3")), "2")) 2L else 3L
  ylab <- st_name(names, "y")
  nf <- length(factors)

  rhs <- st_rhs(factors, switch(model, full = "full", twoway = "twoway", "none"))
  if (length(covs)) rhs <- paste(rhs, "+", paste(covs, collapse = " + "))
  form <- stats::as.formula(paste("y ~", rhs))

  empty <- st_empty_cells(data, factors, names)
  fit <- st_with_sum_contrasts(stats::lm(form, data = data))
  if (anyNA(stats::coef(fit))) {
    if (length(empty)) {
      aui_stop("Some combinations of factor levels have no data (", paste(utils::head(empty, 3), collapse = "; "),
               if (length(empty) > 3) "; …" else "", "), so not all interactions can be estimated. ",
               "Choose “Main effects only” (or fewer interactions) under Model, or remove a factor.")
    }
    aui_stop("The model terms are perfectly confounded with one another (for example two factors that define ",
             "the same groups, or a covariate that is constant within groups), so their effects cannot be separated. Remove one of them.")
  }
  if (stats::df.residual(fit) < 1) {
    aui_stop("There is only one observation per combination of factor levels, so no residual variation is left to test ",
             "the full model. Choose “Main effects only” under Model (the interaction then serves as the error term), or add replicates.")
  }
  ss_res <- sum(stats::residuals(fit)^2)
  ss_tot <- sum((data$y - mean(data$y))^2)
  if (ss_res <= ss_tot * 1e-14) aui_stop("The model reproduces the data exactly (no residual variation), so F ratios cannot be computed.")
  if (length(empty)) {
    aui_warn(paste0("Some combinations of factor levels have no data (", paste(utils::head(empty, 3), collapse = "; "),
                    if (length(empty) > 3) "; …" else "", "). Estimated marginal means involving these cells are not available."))
  }

  a <- as.data.frame(st_try(car::Anova(fit, type = type), "The ANOVA table could not be computed for this design."))
  a <- a[rownames(a) != "(Intercept)", , drop = FALSE]
  is_res <- rownames(a) == "Residuals"
  df_res <- a$Df[is_res]; ms_res <- a$`Sum Sq`[is_res] / df_res
  eta_p <- ifelse(is_res, NA, a$`Sum Sq` / (a$`Sum Sq` + a$`Sum Sq`[is_res]))
  tab <- data.frame(
    Source = c(ifelse(is_res, "Residual", st_pretty(rownames(a), names)), "Total (corrected)"),
    SS = c(a$`Sum Sq`, ss_tot), df = c(a$Df, nrow(data) - 1),
    MS = c(a$`Sum Sq` / a$Df, NA), F = c(a$`F value`, NA), p = c(a$`Pr(>F)`, NA),
    stars = c(ifelse(is_res, "", aui_stars(a$`Pr(>F)`)), ""), eta_p = c(eta_p, NA), stringsAsFactors = FALSE)
  balanced <- length(unique(as.vector(table(data[, factors, drop = FALSE])))) == 1
  title <- paste0(if (length(covs)) "ANCOVA" else "ANOVA", " table")
  aui_table(title, tab, formats = c(df = "int", p = "p", stars = "stars"),
            labels = c(p = "P value", stars = "Summary", eta_p = "Partial η²"),
            footnotes = c(
              paste0("Type ", if (type == 3) "III" else "II", " sums of squares",
                     if (type == 3) " (each effect adjusted for all others; sum-to-zero contrasts)" else " (each effect adjusted for the others, ignoring interactions that contain it)",
                     if (balanced && !length(covs)) "; the design is balanced, so Type II and III agree." else if (!balanced) "; the design is unbalanced, so the sums of squares need not add up to the total." else "."),
              "Partial η² = SS effect / (SS effect + SS residual).",
              if (length(covs)) "Covariate effects are slopes adjusted for the factors; factor effects are adjusted for the covariates.",
              st_dropped_note(options)))

  if (aui_flag(options, "descriptives", TRUE)) {
    aui_table("Descriptive statistics", st_cell_descriptives(data$y, data, factors, names),
              formats = c(n = "int"), labels = c(mean = "Mean", sd = "SD", sem = "SEM"))
  }

  emm_note <- c(if (nf > 1) "Marginal means give equal weight to every level of the other factors.",
                if (length(covs)) paste0("Evaluated at the mean of ", paste(vapply(covs, function(v) st_name(names, v), ""), collapse = ", "), "."))
  if (aui_flag(options, "emm", TRUE)) {
    aui_heading("Estimated marginal means")
    for (f in factors) st_emit_emm(fit, f, names, alpha, footnotes = emm_note)
    if (nf > 1) st_emit_emm(fit, factors, names, alpha,
                            footnotes = if (length(covs)) emm_note[length(emm_note)] else NULL)
  }

  st_factorial_posthoc(fit, factors, names, options, alpha)

  res <- stats::residuals(fit)
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    cell <- interaction(data[, factors, drop = FALSE], drop = TRUE)
    lv <- st_levene(if (length(covs)) res else data$y, cell, stats::median)
    aui_table("Equal variances", data.frame(
      Test = "Brown–Forsythe (Levene, median-centred)", F = lv$F, df1 = lv$df1, df2 = lv$df2, p = lv$p,
      stringsAsFactors = FALSE), formats = c(df1 = "int", df2 = "int", p = "p"), labels = c(p = "P value"),
      footnotes = c(paste0("Compares the spread of ", if (length(covs)) "the residuals" else "the values", " across all ",
                           nlevels(cell), " cells of the design. A small P value is evidence of unequal variances."),
                    if (is.na(lv$F)) "Not available: every cell needs more than one value."))
    st_emit_normality(res, alpha)
  }
  if (aui_flag(options, "resid_plots")) st_emit_residual_plots(stats::fitted(fit), res)

  if (aui_flag(options, "show_plot", TRUE)) {
    pf <- factors[seq_len(min(3, nf))]
    em <- st_emit_emm(fit, pf, names, alpha, emit = FALSE)
    aui_plot(if (nf > 1) paste0("Interaction plot (estimated marginal means ± ", st_pct(alpha), " CI)")
             else paste0("Estimated marginal means ± ", st_pct(alpha), " CI"),
             st_emm_plot(em, pf, names, ylab, alpha), width = if (nf > 2) 8 else 5.6, height = 4)
  }

  eff <- !is_res
  parts <- sprintf("%s: %s, %s", st_pretty(rownames(a)[eff], names),
                   mapply(st_fmt_f, a$`F value`[eff], a$Df[eff], df_res), vapply(a$`Pr(>F)`[eff], st_p_text, ""))
  sig <- a$`Pr(>F)`[eff] < alpha
  inter <- grepl(":", rownames(a)[eff], fixed = TRUE)
  txt <- paste0(if (length(covs)) "ANCOVA" else paste0(c("One", "Two", "Three", "Four")[nf], "-way ANOVA"),
                " (Type ", if (type == 3) "III" else "II", "). ", paste(parts, collapse = "; "), ". ")
  txt <- paste0(txt, if (!any(sig)) "No effect is statistically significant."
                else paste0("Significant: ", paste(st_pretty(rownames(a)[eff][sig], names), collapse = ", "), "."))
  if (any(inter & sig)) txt <- paste0(txt, " Because an interaction is significant, the effect of one factor depends on the level of the other; look at the simple effects rather than the main effects alone.")
  aui_summary(txt)

  if (nf >= 2) {
    aui_graph("Create a grouped bar graph of these data", "grouped_bar",
              list(layout = "long", dep = st_name(names, "y"), x = st_name(names, factors[1]), group = st_name(names, factors[2])))
  }
}
