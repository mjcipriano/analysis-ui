# One-way ANOVA (ordinary / Welch / Brown-Forsythe) with multiple comparisons.
# data: value (numeric), group (factor)

.oneway_tests <- function(value, group) {
  n <- as.numeric(tapply(value, group, length))
  mu <- as.numeric(tapply(value, group, mean))
  v <- as.numeric(tapply(value, group, stats::var))
  k <- length(n); N <- sum(n)
  grand <- mean(value)
  ssb <- sum(n * (mu - grand)^2)
  ssw <- sum((value - stats::ave(value, group))^2)
  list(n = n, mean = mu, var = v, k = k, N = N, ssb = ssb, ssw = ssw, sst = ssb + ssw,
       dfb = k - 1, dfw = N - k)
}

# Brown-Forsythe ANOVA: F* = SSB / sum((1 - n_i/N) s_i^2), Satterthwaite df.
.brown_forsythe_anova <- function(s) {
  w <- (1 - s$n / s$N) * s$var
  f <- s$ssb / sum(w)
  cc <- w / sum(w)
  df2 <- 1 / sum(cc^2 / (s$n - 1))
  list(F = f, df1 = s$k - 1, df2 = df2, p = stats::pf(f, s$k - 1, df2, lower.tail = FALSE))
}

.welch_anova <- function(s) {
  w <- s$n / s$var
  mw <- sum(w * s$mean) / sum(w)
  a <- sum(w * (s$mean - mw)^2) / (s$k - 1)
  lam <- sum((1 - w / sum(w))^2 / (s$n - 1))
  f <- a / (1 + 2 * (s$k - 2) / (s$k^2 - 1) * lam)
  df2 <- (s$k^2 - 1) / (3 * lam)
  list(F = f, df1 = s$k - 1, df2 = df2, p = stats::pf(f, s$k - 1, df2, lower.tail = FALSE))
}

run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  value <- data$value
  group <- droplevels(data$group)
  lev <- levels(group)
  variant <- aui_opt(options, "variant", "ordinary")
  s <- .oneway_tests(value, group)
  ylab <- st_name(names, "value")

  if (s$sst <= 0) aui_stop("All values are identical, so there is no variation to analyse.")
  if (s$ssw <= s$sst * 1e-14) {
    aui_stop("There is no variation within the groups (every value equals its group mean), ",
             "so an F ratio cannot be computed.")
  }
  if (variant != "ordinary") {
    if (any(s$n < 2)) aui_stop("Welch's and Brown\u2013Forsythe ANOVA need at least two values in every group. ",
                               "Group \u201c", lev[which(s$n < 2)[1]], "\u201d has only one.")
    if (any(s$var <= 0)) aui_stop("Group \u201c", lev[which(s$var <= 0)[1]], "\u201d has no variation (all its values are identical), ",
                                  "so Welch's and Brown\u2013Forsythe ANOVA cannot be computed.")
  }

  msb <- s$ssb / s$dfb; msw <- s$ssw / s$dfw
  f_ord <- msb / msw
  p_ord <- stats::pf(f_ord, s$dfb, s$dfw, lower.tail = FALSE)
  eta2 <- s$ssb / s$sst
  omega2 <- max(0, (s$ssb - s$dfb * msw) / (s$sst + msw))

  # ---- main test
  if (variant == "ordinary") {
    main <- list(F = f_ord, df1 = s$dfb, df2 = s$dfw, p = p_ord)
    test_name <- "ordinary one-way ANOVA"
    aui_table("One-way ANOVA", data.frame(
      Source = c("Between groups (treatment)", "Within groups (residual)", "Total"),
      SS = c(s$ssb, s$ssw, s$sst), df = c(s$dfb, s$dfw, s$N - 1), MS = c(msb, msw, NA),
      F = c(f_ord, NA, NA), p = c(p_ord, NA, NA), stars = c(aui_stars(p_ord), "", ""),
      stringsAsFactors = FALSE),
      formats = c(df = "int", p = "p", stars = "stars"), labels = c(p = "P value", stars = "Summary"),
      footnotes = "Ordinary one-way ANOVA; assumes that the groups are sampled from normal populations with equal SDs.")
  } else {
    main <- if (variant == "welch") .welch_anova(s) else .brown_forsythe_anova(s)
    test_name <- if (variant == "welch") "Welch's ANOVA" else "Brown\u2013Forsythe ANOVA"
    aui_table(paste0(if (variant == "welch") "Welch's" else "Brown\u2013Forsythe", " ANOVA"), data.frame(
      F = main$F, df1 = main$df1, df2 = main$df2, p = main$p, stars = aui_stars(main$p), stringsAsFactors = FALSE),
      formats = c(df1 = "int", p = "p", stars = "stars"),
      labels = c(F = if (variant == "welch") "W (Welch's F)" else "F* (Brown\u2013Forsythe)", df1 = "df (numerator)",
                 df2 = "df (denominator)", p = "P value", stars = "Summary"),
      footnotes = paste0(test_name, "; does not assume equal SDs. The denominator df is ",
                         if (variant == "welch") "Welch's" else "Satterthwaite's", " approximation."))
    aui_table("Sums of squares", data.frame(
      Source = c("Between groups (treatment)", "Within groups (residual)", "Total"),
      SS = c(s$ssb, s$ssw, s$sst), df = c(s$dfb, s$dfw, s$N - 1), MS = c(msb, msw, NA), stringsAsFactors = FALSE),
      formats = c(df = "int"))
  }
  ci <- st_eta2_ci(f_ord, s$dfb, s$dfw, 1 - alpha)
  aui_table("Effect size", data.frame(
    Measure = c("R\u00b2 (eta squared)", "Omega squared"), value = c(eta2, omega2),
    lower = c(ci[1], NA), upper = c(ci[2], NA), stringsAsFactors = FALSE),
    labels = c(value = "Value", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
    footnotes = c("R\u00b2 = SS between / SS total: the fraction of the total variation explained by group membership. Omega squared is a less biased estimate of the same quantity.",
                  "Confidence interval from the noncentral F distribution."))

  if (aui_flag(options, "descriptives", TRUE)) {
    aui_table("Descriptive statistics", st_describe_groups(value, group, alpha),
              formats = c(n = "int"), labels = ST_DESC_LABELS(alpha))
  }

  # ---- multiple comparisons
  compare <- aui_opt(options, "compare", "pairs")
  n_sig <- NA
  if (compare != "none" && s$k >= 2) {
    ctrl <- if (compare == "control") st_control_index(lev, aui_opt(options, "control", "")) else 1L
    cmp <- st_group_comparisons(value, group, type = compare, control = ctrl,
                                method = aui_opt(options, "posthoc", "tukey"),
                                pooled = variant == "ordinary", alpha = alpha)
    st_emit_comparisons(
      if (compare == "control") paste0("Multiple comparisons: each group vs ", lev[ctrl]) else "Multiple comparisons: all pairs",
      cmp, alpha)
    n_sig <- sum(cmp$p < alpha, na.rm = TRUE)
    n_cmp <- nrow(cmp)
    cmp_method <- attr(cmp, "label")
  }

  # ---- linear trend
  if (aui_flag(options, "trend")) {
    if (s$k < 3) {
      aui_text("The test for linear trend needs at least three groups.", title = "Test for linear trend")
    } else {
      cc <- seq_len(s$k) - mean(seq_len(s$k))
      l <- sum(cc * s$mean)
      ss_lin <- l^2 / sum(cc^2 / s$n)
      f_lin <- ss_lin / msw
      ss_dev <- max(0, s$ssb - ss_lin)
      f_dev <- (ss_dev / (s$k - 2)) / msw
      slope <- sum(s$n * (seq_len(s$k) - sum(s$n * seq_len(s$k)) / s$N) * s$mean) /
        sum(s$n * (seq_len(s$k) - sum(s$n * seq_len(s$k)) / s$N)^2)
      aui_table("Test for linear trend", data.frame(
        Source = c("Linear trend", "Departure from linear trend"),
        SS = c(ss_lin, ss_dev), df = c(1, s$k - 2), MS = c(ss_lin, ss_dev / (s$k - 2)),
        F = c(f_lin, f_dev),
        p = stats::pf(c(f_lin, f_dev), c(1, s$k - 2), s$dfw, lower.tail = FALSE), stringsAsFactors = FALSE),
        formats = c(df = "int", p = "p"), labels = c(p = "P value"),
        footnotes = c(sprintf("Slope = %s per step from one group to the next (groups in the order shown, treated as equally spaced).", st_fmt(slope)),
                      "Linear contrast of the group means, tested against the within-group mean square of the ordinary ANOVA."))
    }
  }

  # ---- assumptions
  res <- value - stats::ave(value, group)
  if (aui_flag(options, "assumptions", TRUE)) {
    aui_heading("Assumption checks")
    bf <- st_levene(value, group, stats::median)
    bt <- if (all(s$n >= 2) && all(s$var > 0)) tryCatch(stats::bartlett.test(value, group), error = function(e) NULL) else NULL
    aui_table("Equal variances", data.frame(
      Test = c("Brown\u2013Forsythe (Levene, median-centred)", "Bartlett"),
      stat_name = c("F", "K\u00b2"),
      statistic = c(bf$F, if (is.null(bt)) NA else unname(bt$statistic)),
      df = c(sprintf("%d, %d", bf$df1, bf$df2), if (is.null(bt)) "" else as.character(unname(bt$parameter))),
      p = c(bf$p, if (is.null(bt)) NA else bt$p.value), stringsAsFactors = FALSE),
      formats = c(p = "p"), labels = c(stat_name = "Statistic", statistic = "Value", p = "P value"),
      footnotes = c("A small P value is evidence that the group SDs differ. Bartlett's test is sensitive to non-normality; the Brown\u2013Forsythe test is more robust.",
                    if (variant == "ordinary" && !is.na(bf$p) && bf$p < alpha) "The SDs appear to differ: consider Welch's ANOVA with Games\u2013Howell comparisons."))
    st_emit_normality(res, alpha)
  }
  if (aui_flag(options, "qq_plot")) aui_plot("QQ plot of residuals", st_qq_plot(res), width = 4.6, height = 3.8)

  if (aui_flag(options, "show_plot", TRUE)) {
    aui_plot(paste0("Group means with ", st_pct(alpha), " confidence intervals"),
             st_means_plot(value, group, alpha, ylab = ylab,
                           xlab = if (identical(st_name(names, "group"), "Group")) NULL else st_name(names, "group")),
             width = max(4.2, min(9, 1.6 + 0.9 * s$k)), height = 3.9)
  }

  # ---- summary
  txt <- sprintf("%s: the group means %s (%s, P = %s; R\u00b2 = %s).",
                 sub("^(.)", "\\U\\1", test_name, perl = TRUE),
                 if (main$p < alpha) "differ significantly" else "do not differ significantly",
                 st_fmt_f(main$F, main$df1, main$df2), aui_fmt_p(main$p), formatC(eta2, format = "f", digits = 3))
  if (!is.na(n_sig)) {
    txt <- paste0(txt, sprintf(" %d of %d comparison%s %s significant at \u03b1 = %s (%s).",
                               n_sig, n_cmp, if (n_cmp == 1) "" else "s", if (n_sig == 1) "is" else "are",
                               format(alpha), sub("\\s*\\(.*$", "", cmp_method)))
  }
  aui_summary(txt)

  keep <- st_layout_options(options)
  aui_graph("Create a column graph of these data", "column_scatter", keep)
  aui_graph("Create a bar graph of these data", "column_bar", keep)
}
