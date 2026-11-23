# Repeated-measures one-way ANOVA (classic, with sphericity corrections) or a
# mixed-effects model when subjects with missing values are kept.
# data: value, group (factor: conditions), subject (factor)

.rm_classic <- function(data, options, names, alpha) {
  group <- data$group
  lev <- levels(group)
  wide <- tapply(data$value, list(data$subject, group), function(v) v[1])
  wide <- wide[stats::complete.cases(wide), , drop = FALSE]
  n <- nrow(wide); k <- ncol(wide)
  if (n < 2) aui_stop("At least two subjects with values in every condition are needed.")
  grand <- mean(wide)
  ss_t <- n * sum((colMeans(wide) - grand)^2)
  ss_s <- k * sum((rowMeans(wide) - grand)^2)
  ss_tot <- sum((wide - grand)^2)
  ss_r <- ss_tot - ss_t - ss_s
  df_t <- k - 1; df_s <- n - 1; df_r <- df_t * df_s
  if (ss_tot <= 0) aui_stop("All values are identical, so there is no variation to analyse.")
  if (ss_r <= ss_tot * 1e-14) {
    aui_stop("After removing subject and condition effects there is no residual variation ",
             "(every subject changes by exactly the same amount), so an F ratio cannot be computed.")
  }
  ms_t <- ss_t / df_t; ms_s <- ss_s / df_s; ms_r <- ss_r / df_r
  f_t <- ms_t / ms_r; f_s <- ms_s / ms_r
  sph <- st_sphericity(wide)
  mode <- aui_opt(options, "sphericity", "gg")
  if (k == 2) mode <- "assume"
  eps <- switch(mode, gg = sph$gg, hf = sph$hf, 1)
  p_t <- stats::pf(f_t, df_t * eps, df_r * eps, lower.tail = FALSE)
  p_s <- stats::pf(f_s, df_s, df_r, lower.tail = FALSE)
  corr_name <- switch(mode, gg = "Greenhouse\u2013Geisser", hf = "Huynh\u2013Feldt", "none")

  aui_table("Repeated-measures one-way ANOVA", data.frame(
    Source = c("Treatment (between conditions)", "Subjects (matching)", "Residual", "Total"),
    SS = c(ss_t, ss_s, ss_r, ss_tot), df = c(df_t, df_s, df_r, n * k - 1),
    MS = c(ms_t, ms_s, ms_r, NA), F = c(f_t, f_s, NA, NA),
    dfn = c(df_t * eps, df_s, NA, NA), dfd = c(df_r * eps, df_r, NA, NA),
    p = c(p_t, p_s, NA, NA), stars = c(aui_stars(p_t), aui_stars(p_s), "", ""), stringsAsFactors = FALSE),
    formats = c(df = "int", p = "p", stars = "stars"),
    labels = c(dfn = "df numerator (for P)", dfd = "df denominator (for P)", p = "P value", stars = "Summary"),
    footnotes = c(
      if (mode == "assume") "Sphericity assumed: P values use the uncorrected degrees of freedom."
      else sprintf("Sphericity not assumed: the treatment P value uses degrees of freedom multiplied by the %s epsilon (%s).",
                   corr_name, formatC(eps, format = "f", digits = 4)),
      sprintf("%d subjects \u00d7 %d conditions.", n, k),
      st_dropped_note(options, "subject")))

  eta_p <- ss_t / (ss_t + ss_r)
  aui_table("Effect size and matching", data.frame(
    Measure = c("Partial eta squared (treatment)", "R\u00b2 (treatment / total)", "Was the matching effective? F", "Matching: P value"),
    value = c(eta_p, ss_t / ss_tot, f_s, p_s), stringsAsFactors = FALSE),
    labels = c(value = "Value"),
    footnotes = if (p_s < alpha) "Subjects differ significantly from one another, so matching (repeated measures) removed real between-subject variation and increased power."
                else "No significant variation among subjects was detected; the matching did not clearly help in this experiment (keep it if it was part of the design).")

  if (aui_flag(options, "descriptives", TRUE)) {
    d <- st_describe_groups(as.vector(wide), factor(rep(lev, each = n), levels = lev), alpha)
    names(d)[1] <- "Condition"
    aui_table("Descriptive statistics", d, formats = c(n = "int"), labels = ST_DESC_LABELS(alpha))
  }

  compare <- aui_opt(options, "compare", "pairs")
  n_sig <- NA; n_cmp <- 0
  if (compare != "none") {
    ctrl <- if (compare == "control") st_control_index(lev, aui_opt(options, "control", "")) else 1L
    cmp <- st_paired_comparisons(wide, compare, ctrl, aui_opt(options, "posthoc", "tukey"),
                                 pooled = mode == "assume", ms_resid = ms_r, df_resid = df_r, alpha = alpha)
    st_emit_comparisons(if (compare == "control") paste0("Multiple comparisons: each condition vs ", lev[ctrl])
                        else "Multiple comparisons: all pairs", cmp, alpha)
    n_sig <- sum(cmp$p < alpha, na.rm = TRUE); n_cmp <- nrow(cmp)
  }

  res <- wide - outer(rowMeans(wide), rep(1, k)) - outer(rep(1, n), colMeans(wide)) + grand
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    if (k >= 3) {
      aui_table("Sphericity", data.frame(
        W = sph$W, chi2 = sph$chi2, df = sph$df, p = sph$p, gg = sph$gg, hf = sph$hf, lb = sph$lb,
        stringsAsFactors = FALSE),
        formats = c(df = "int", p = "p"),
        labels = c(W = "Mauchly's W", chi2 = "\u03c7\u00b2", p = "P value", gg = "Greenhouse\u2013Geisser \u03b5",
                   hf = "Huynh\u2013Feldt \u03b5", lb = "Lower-bound \u03b5"),
        footnotes = c("Mauchly's test: a small P value is evidence that sphericity (equal variances of all pairwise differences) is violated. The test has little power in small samples, so correcting by default is the safer choice.",
                      "Epsilon = 1 means perfect sphericity; smaller values mean stronger violation.",
                      if (is.na(sph$W)) "Mauchly's test needs more subjects than conditions."))
    } else {
      aui_text("With only two conditions sphericity always holds; no correction is needed.", title = "Sphericity")
    }
    st_emit_normality(as.vector(res), alpha)
  }
  if (aui_flag(options, "qq_plot")) aui_plot("QQ plot of residuals", st_qq_plot(as.vector(res)), width = 4.6, height = 3.8)

  txt <- sprintf("Repeated-measures ANOVA%s: the means of the %d conditions %s (%s, %s; partial \u03b7\u00b2 = %s).",
                 if (mode == "assume") "" else paste0(" with ", corr_name, " correction"), k,
                 if (p_t < alpha) "differ significantly" else "do not differ significantly",
                 st_fmt_f(f_t, df_t * eps, df_r * eps), st_p_text(p_t), formatC(eta_p, format = "f", digits = 3))
  list(text = txt, n_sig = n_sig, n_cmp = n_cmp)
}

.rm_mixed <- function(data, options, names, alpha) {
  aui_need("lme4", "lmerTest", "emmeans")
  lev <- levels(data$group)
  k <- length(lev)
  q <- st_quiet(st_try(st_with_sum_contrasts(lmerTest::lmer(value ~ group + (1 | subject), data = data, REML = TRUE)),
                       "The mixed-effects model could not be fitted."))
  fit <- q$value
  n_inc <- as.integer(options[["_n_incomplete"]] %||% 0)
  a <- as.data.frame(st_quiet(stats::anova(fit, type = 3, ddf = "Satterthwaite"))$value)
  f_t <- a$`F value`[1]; p_t <- a$`Pr(>F)`[1]
  aui_table("Mixed-effects model: fixed effect of treatment", data.frame(
    Source = "Treatment (between conditions)", F = f_t, dfn = a$NumDF[1], dfd = a$DenDF[1], p = p_t,
    stars = aui_stars(p_t), stringsAsFactors = FALSE),
    formats = c(p = "p", stars = "stars"),
    labels = c(dfn = "df numerator", dfd = "df denominator", p = "P value", stars = "Summary"),
    footnotes = c("Linear mixed-effects model fitted by REML with a random intercept for each subject (compound symmetry); F test with Satterthwaite degrees of freedom.",
                  sprintf("%d subjects, %d observations; %s.", nlevels(droplevels(data$subject)), nrow(data),
                          if (n_inc == 0) "no subject has missing values"
                          else sprintf("%d subject%s missing values and contribute%s the values available", n_inc,
                                       if (n_inc == 1) " has" else "s have", if (n_inc == 1) "s" else "")),
                  "With no missing values this gives the same F and P as a repeated-measures ANOVA that assumes sphericity. The results are valid when values are missing at random."))
  vc <- as.data.frame(lme4::VarCorr(fit))
  v_s <- vc$vcov[vc$grp == "subject"]; v_e <- vc$vcov[vc$grp == "Residual"]
  aui_table("Random effects", data.frame(
    Component = c("Subjects (random intercept)", "Residual"), Variance = c(v_s, v_e), SD = sqrt(c(v_s, v_e)),
    stringsAsFactors = FALSE),
    footnotes = sprintf("Intraclass correlation (share of the variance due to differences between subjects) = %s.",
                        formatC(v_s / (v_s + v_e), format = "f", digits = 3)))
  if (lme4::isSingular(fit)) aui_warn("The estimated variation between subjects is zero (singular fit): subjects do not differ consistently, so the model behaves like an ordinary one-way ANOVA.")

  if (aui_flag(options, "descriptives", TRUE)) {
    d <- st_describe_groups(data$value, data$group, alpha)
    names(d)[1] <- "Condition"
    aui_table("Descriptive statistics (observed values)", d, formats = c(n = "int"), labels = ST_DESC_LABELS(alpha))
    st_emit_emm(fit, "group", list(group = "Condition"), alpha, title = "Estimated marginal means (model-based)",
                footnotes = "Model-based means; they differ from the observed means when values are missing.")
  }
  compare <- aui_opt(options, "compare", "pairs")
  n_sig <- NA; n_cmp <- 0
  if (compare != "none") {
    method <- aui_opt(options, "posthoc", "tukey")
    cmp <- st_emit_emm_comparisons(fit, "group", NULL, list(group = "Condition"), type = compare,
                                   control = aui_opt(options, "control", ""), method = method, alpha = alpha,
                                   title = if (compare == "control") "Multiple comparisons with the control condition" else "Multiple comparisons: all pairs",
                                   extra_notes = "Comparisons use the pooled error of the mixed model (Satterthwaite df).")
    if (!is.null(cmp)) { n_sig <- sum(cmp$p < alpha, na.rm = TRUE); n_cmp <- nrow(cmp) }
  }
  res <- stats::residuals(fit)
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    st_emit_normality(res, alpha)
  }
  if (aui_flag(options, "qq_plot")) aui_plot("QQ plot of residuals", st_qq_plot(res), width = 4.6, height = 3.8)
  txt <- sprintf("Mixed-effects model: the means of the %d conditions %s (%s, %s).", k,
                 if (p_t < alpha) "differ significantly" else "do not differ significantly",
                 st_fmt_f(f_t, a$NumDF[1], a$DenDF[1]), st_p_text(p_t))
  list(text = txt, n_sig = n_sig, n_cmp = n_cmp)
}

run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  data$group <- droplevels(data$group)
  data$subject <- droplevels(data$subject)
  k <- nlevels(data$group)
  if (k < 2) aui_stop("Repeated-measures ANOVA needs at least two conditions.")
  mixed <- identical(aui_opt(options, "missing", "exclude"), "mixed")
  out <- if (mixed) .rm_mixed(data, options, names, alpha) else .rm_classic(data, options, names, alpha)

  if (aui_flag(options, "show_plot", TRUE)) {
    aui_plot(paste0("Condition means with ", st_pct(alpha), " confidence intervals"),
             st_means_plot(data$value, data$group, alpha, ylab = st_name(names, "value"), connect = TRUE),
             width = max(4.2, min(9, 1.6 + 0.9 * k)), height = 3.9)
  }
  txt <- out$text
  if (!is.na(out$n_sig)) txt <- paste0(txt, sprintf(" %d of %d comparison%s significant at \u03b1 = %s.",
                                                    out$n_sig, out$n_cmp, if (out$n_cmp == 1) " is" else "s are", format(alpha)))
  aui_summary(txt)
  aui_graph("Create a before\u2013after graph of these data", "before_after", st_layout_options(options))
}
