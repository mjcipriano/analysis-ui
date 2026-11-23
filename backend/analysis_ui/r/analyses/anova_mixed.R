# Repeated-measures / mixed ANOVA (afex) or the equivalent mixed-effects model.
# data: y, subject, w1.. (within-subjects factors), b1.. (between-subjects factors)

.mixed_anova_classic <- function(d, within, between, factors, names, levmap, options, alpha) {
  aui_need("afex")
  corr <- aui_opt(options, "sphericity", "gg")
  fit <- st_quiet(st_try(
    afex::aov_ez(id = "subject", dv = "y", data = d, within = within,
                 between = if (length(between)) between else NULL, type = 3, include_aov = FALSE,
                 anova_table = list(es = "ges", correction = "none")),
    "The repeated-measures ANOVA could not be computed for this design."))$value
  s <- st_quiet(summary(fit))$value
  ut <- unclass(s$univariate.tests)
  ut <- ut[rownames(ut) != "(Intercept)", , drop = FALSE]
  terms <- rownames(ut)
  adj <- s$pval.adjustments
  sph <- s$sphericity.tests
  has_adj <- !is.null(adj) && length(adj) > 0 && nrow(adj) > 0
  eps <- rep(NA_real_, length(terms)); p <- ut[, "Pr(>F)"]
  if (has_adj && corr %in% c("gg", "hf")) {
    hit <- match(terms, rownames(adj))
    e <- adj[, if (corr == "gg") "GG eps" else "HF eps"][hit]
    eps <- pmin(1, e)
    pc <- adj[, if (corr == "gg") "Pr(>F[GG])" else "Pr(>F[HF])"][hit]
    p <- ifelse(is.na(pc), p, pc)
  }
  at <- as.data.frame(st_quiet(stats::anova(fit, es = "ges", correction = "none"))$value)
  ges <- at$ges[match(terms, rownames(at))]
  df1 <- ut[, "num Df"]; df2 <- ut[, "den Df"]
  f <- ut[, "F value"]
  if (any(!is.finite(f))) aui_stop("There is no residual variation for at least one effect, so F ratios cannot be computed.")
  corr_name <- switch(corr, gg = "Greenhouse\u2013Geisser", hf = "Huynh\u2013Feldt", "none")
  aui_table("Repeated-measures ANOVA", data.frame(
    Source = st_pretty(terms, names),
    Type = ifelse(vapply(strsplit(terms, ":", fixed = TRUE), function(t) any(t %in% within), logical(1)), "Within", "Between"),
    SS = ut[, "Sum Sq"], df = df1, SSe = ut[, "Error SS"], dfe = df2,
    F = f, eps = eps, p = p, stars = aui_stars(p), ges = ges, stringsAsFactors = FALSE),
    formats = c(p = "p", stars = "stars", df = "int", dfe = "int"),
    labels = c(Type = "Effect type", SSe = "SS error", dfe = "df error", eps = "\u03b5 applied", p = "P value", stars = "Summary",
               ges = "Generalized \u03b7\u00b2"),
    footnotes = c(
      "Type III sums of squares; each effect is tested against its own error term.",
      if (corr == "none" || !has_adj) "Sphericity assumed: P values use the uncorrected degrees of freedom."
      else paste0("P values of within-subjects effects with more than two levels are ", corr_name,
                  "-corrected: both degrees of freedom are multiplied by \u03b5."),
      sprintf("%d subjects with complete data.", nlevels(droplevels(d$subject))),
      if ((options[["_n_incomplete"]] %||% 0) > 0) sprintf("%d subject(s) with missing values in some condition were excluded; the mixed-effects model option can use them.", options[["_n_incomplete"]]),
      if (isTRUE(options[["_aggregated"]])) "Several observations per subject and condition were averaged before the analysis."))

  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    if (!is.null(sph) && length(sph) > 0 && nrow(sph) > 0) {
      sph <- unclass(sph)
      aui_table("Sphericity", data.frame(
        Effect = st_pretty(rownames(sph), names), W = sph[, "Test statistic"], p = sph[, "p-value"],
        gg = adj[match(rownames(sph), rownames(adj)), "GG eps"],
        hf = pmin(1, adj[match(rownames(sph), rownames(adj)), "HF eps"]), stringsAsFactors = FALSE),
        formats = c(p = "p"),
        labels = c(W = "Mauchly's W", p = "P value", gg = "Greenhouse\u2013Geisser \u03b5", hf = "Huynh\u2013Feldt \u03b5"),
        footnotes = "Mauchly's test: a small P value is evidence that sphericity is violated. Epsilon = 1 means perfect sphericity.")
    } else {
      aui_text("No within-subjects factor has more than two levels, so sphericity always holds.", title = "Sphericity")
    }
    if (length(between)) {
      sm <- stats::aggregate(d$y, d[, c("subject", between), drop = FALSE], mean)
      lv <- st_levene(sm$x, interaction(sm[, between, drop = FALSE], drop = TRUE), stats::median)
      aui_table("Equal variances between groups", data.frame(
        Test = "Brown\u2013Forsythe (Levene, median-centred)", F = lv$F, df1 = lv$df1, df2 = lv$df2, p = lv$p,
        stringsAsFactors = FALSE), formats = c(df1 = "int", df2 = "int", p = "p"), labels = c(p = "P value"),
        footnotes = "Computed on each subject's mean across the repeated measurements, across the between-subjects groups.")
    }
    # residuals after removing condition (cell) effects and each subject's own level
    cell <- interaction(d[, factors, drop = FALSE], drop = TRUE)
    bcell <- if (length(between)) interaction(d[, between, drop = FALSE], drop = TRUE) else factor(rep("all", nrow(d)))
    res <- d$y - stats::ave(d$y, cell) - stats::ave(d$y, d$subject) + stats::ave(d$y, bcell)
    st_emit_normality(res, alpha, title = "Normality of residuals (subject and condition effects removed)")
  }
  list(fit = fit, terms = terms, f = f, df1 = df1 * ifelse(is.na(eps), 1, eps), df2 = df2 * ifelse(is.na(eps), 1, eps), p = p,
       lead = paste0("Repeated-measures ANOVA", if (corr != "none" && has_adj) paste0(" (", corr_name, " correction).") else "."),
       emm_note = "Based on the multivariate model: every comparison uses its own error term (no sphericity assumption).")
}

.mixed_anova_lmm <- function(d, within, between, factors, names, levmap, options, alpha) {
  aui_need("lme4", "lmerTest")
  form <- stats::as.formula(paste("y ~", paste(c(between, within), collapse = " * "), "+ (1 | subject)"))
  q <- st_quiet(st_try(st_with_sum_contrasts(lmerTest::lmer(form, data = d, REML = TRUE)),
                       "The mixed-effects model could not be fitted. Some combinations of factor levels may have no data."))
  fit <- q$value
  full_rank <- ncol(stats::model.matrix(stats::as.formula(paste("~", paste(c(between, within), collapse = " * "))), d))
  if (length(lme4::fixef(fit)) < full_rank) {
    aui_stop("Some combinations of factor levels have no data, so not all effects can be estimated. Remove a factor or collect the missing combinations.")
  }
  a <- as.data.frame(st_quiet(stats::anova(fit, type = 3, ddf = "Satterthwaite"))$value)
  terms <- rownames(a)
  aui_table("Mixed-effects model: tests of fixed effects", data.frame(
    Source = st_pretty(terms, names), SS = a$`Sum Sq`, MS = a$`Mean Sq`, df1 = a$NumDF, df2 = a$DenDF,
    F = a$`F value`, p = a$`Pr(>F)`, stars = aui_stars(a$`Pr(>F)`), stringsAsFactors = FALSE),
    formats = c(p = "p", stars = "stars", df1 = "int"),
    labels = c(df1 = "df numerator", df2 = "df denominator", p = "P value", stars = "Summary"),
    footnotes = c("Linear mixed-effects model fitted by REML with a random intercept for each subject; Type III F tests with Satterthwaite degrees of freedom.",
                  sprintf("%d subjects, %d observations; %s.", nlevels(droplevels(d$subject)), nrow(d),
                          if ((options[["_n_incomplete"]] %||% 0) > 0) sprintf("%d subject(s) have missing values and contribute the values available", options[["_n_incomplete"]])
                          else "no subject has missing values"),
                  "Assumes compound symmetry (sphericity). The results are valid when values are missing at random.",
                  if (isTRUE(options[["_aggregated"]])) "Several observations per subject and condition were averaged before the analysis."))
  st_emit_varcomp(fit, c(names, list(subject = st_name(names, "subject"))))
  if (lme4::isSingular(fit)) aui_warn("The estimated variation between subjects is zero (singular fit): subjects do not differ consistently once the factors are accounted for.")
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    st_emit_normality(stats::residuals(fit), alpha)
  }
  list(fit = fit, terms = terms, f = a$`F value`, df1 = a$NumDF, df2 = a$DenDF, p = a$`Pr(>F)`,
       lead = "Mixed-effects model (random intercept per subject).",
       emm_note = "Based on the mixed model (pooled error, Satterthwaite df).")
}

run <- function(data, options) {
  aui_need("ggplot2", "emmeans")
  names <- st_names(options)
  alpha <- st_alpha(options)
  within <- aui_vars(options, "_within")
  between <- aui_vars(options, "_between")
  factors <- c(within, between)
  data$subject <- droplevels(data$subject)
  for (f in factors) data[[f]] <- droplevels(data[[f]])

  empty <- st_empty_cells(data, factors, names)
  if (length(empty) && length(between) > 1) {
    tab <- table(unique(data[, c("subject", between), drop = FALSE])[, between, drop = FALSE])
    if (any(tab == 0)) aui_stop("Some combinations of the between-subjects factors contain no subjects, so their interaction cannot be estimated. Remove one of the between-subjects factors.")
  }
  if (length(between)) {
    per <- table(unique(data[, c("subject", between), drop = FALSE])[, between, drop = FALSE])
    if (any(per == 1)) aui_warn("At least one between-subjects group contains a single subject; its variability cannot be estimated and results rest on the other groups.")
  }
  rc <- st_recode_levels(data, factors)
  d <- rc$data; levmap <- rc$levmap
  lmm <- identical(aui_opt(options, "engine", "anova"), "lmm")

  if (aui_flag(options, "descriptives", TRUE)) {
    desc <- st_cell_descriptives(d$y, d, factors, names, levmap)
  }
  out <- if (lmm) .mixed_anova_lmm(d, within, between, factors, names, levmap, options, alpha)
         else .mixed_anova_classic(d, within, between, factors, names, levmap, options, alpha)
  fit <- out$fit

  if (aui_flag(options, "descriptives", TRUE)) {
    aui_table("Descriptive statistics", desc, formats = c(n = "int"), labels = c(mean = "Mean", sd = "SD", sem = "SEM"))
  }
  if (aui_flag(options, "emm", TRUE)) {
    aui_heading("Estimated marginal means")
    for (f in factors) st_emit_emm(fit, f, names, alpha, levmap)
    if (length(factors) > 1) st_emit_emm(fit, factors, names, alpha, levmap)
  }
  st_factorial_posthoc(fit, factors, names, options, alpha, levmap)
  if (!identical(aui_opt(options, "posthoc", "none"), "none")) aui_text(out$emm_note, title = "About these comparisons")

  if (aui_flag(options, "show_plot", TRUE)) {
    pf <- factors[seq_len(min(3, length(factors)))]
    em <- st_emit_emm(fit, pf, names, alpha, levmap, emit = FALSE)
    aui_plot(paste0("Estimated marginal means \u00b1 ", st_pct(alpha), " CI"),
             st_emm_plot(em, pf, names, st_name(names, "y"), alpha), width = if (length(pf) > 2) 8 else 5.6, height = 4)
  }
  aui_summary(st_effects_summary(out$lead, out$terms, out$f, out$df1, out$df2, out$p, alpha, names))
}
