# Shared statistical helpers for the R analyses (ANOVA family, regression
# models, survival, multivariate). Sourced once when the worker starts.
#
# Conventions
#   * Python `prepare` functions hand R a data frame with safe column names
#     (y, f1, x1, ...) and put the user's column names in options$`_names`
#     (safe -> display). `st_pretty()` turns model terms back into display text.
#   * options$`_dropped` is the number of incomplete rows removed in Python.

# ---- names -----------------------------------------------------------------

st_names <- function(options) {
  n <- options[["_names"]]
  if (is.null(n)) list() else as.list(n)
}

st_name <- function(names, key) {
  v <- names[[key]]
  if (is.null(v)) key else as.character(v)
}

# Replace safe names inside model terms with display names; ":" becomes " \u00d7 ".
st_pretty <- function(x, names) {
  x <- gsub(":", " \u00d7 ", as.character(x), fixed = TRUE)
  if (length(names) && length(x)) {
    keys <- names(names)[order(-nchar(names(names)))]
    m <- gregexpr(paste0("\\b(", paste(keys, collapse = "|"), ")\\b"), x, perl = TRUE)
    regmatches(x, m) <- lapply(regmatches(x, m), function(k) {
      if (length(k)) vapply(k, function(i) as.character(names[[i]]), "") else character(0)
    })
  }
  x
}

st_dropped_note <- function(options, what = "row") {
  n <- suppressWarnings(as.integer(options[["_dropped"]] %||% 0L))
  if (is.na(n) || n <= 0) return(NULL)
  sprintf("%d %s%s with missing values %s excluded.", n, what, if (n == 1) "" else "s",
          if (n == 1) "was" else "were")
}

st_alpha <- function(options) {
  a <- aui_num(options, "alpha", 0.05)
  if (!is.finite(a) || a <= 0 || a >= 1) 0.05 else a
}

st_pct <- function(alpha) paste0(format(100 * (1 - alpha), digits = 4), "%")

# ---- running code quietly ---------------------------------------------------

# Evaluate `expr`, swallowing messages and collecting warnings.
st_quiet <- function(expr) {
  warns <- character(0)
  value <- withCallingHandlers(
    suppressMessages(expr),
    warning = function(w) {
      warns <<- c(warns, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  list(value = value, warnings = unique(warns))
}

# Evaluate `expr`; on error stop with a plain-language message.
st_try <- function(expr, message) {
  tryCatch(expr, error = function(e) {
    aui_stop(message, " (Details: ", trimws(conditionMessage(e)), ")")
  })
}

# Evaluate with sum-to-zero contrasts (needed for Type III sums of squares).
st_with_sum_contrasts <- function(expr) {
  old <- options(contrasts = c("contr.sum", "contr.poly"))
  on.exit(options(old), add = TRUE)
  force(expr)
}

st_with_treatment_contrasts <- function(expr) {
  old <- options(contrasts = c("contr.treatment", "contr.poly"))
  on.exit(options(old), add = TRUE)
  force(expr)
}

# ---- multiplicity ------------------------------------------------------------

ST_ADJUST_LABELS <- c(
  none = "no correction for multiple comparisons",
  lsd = "Fisher's LSD (no correction for multiple comparisons)",
  bonferroni = "Bonferroni", sidak = "\u0160id\u00e1k", holm = "Holm",
  holm_sidak = "Holm\u2013\u0160id\u00e1k", fdr = "Benjamini\u2013Hochberg FDR",
  tukey = "Tukey", dunnett = "Dunnett", games_howell = "Games\u2013Howell")

st_adjust_label <- function(method) {
  if (method %in% names(ST_ADJUST_LABELS)) ST_ADJUST_LABELS[[method]] else method
}

# Adjust a family of P values. Methods: none/lsd, bonferroni, sidak, holm,
# holm_sidak, fdr.
st_padjust <- function(p, method) {
  m <- sum(!is.na(p))
  if (m <= 1 || method %in% c("none", "lsd")) return(p)
  switch(method,
    bonferroni = pmin(1, p * m),
    sidak = 1 - (1 - p)^m,
    holm = stats::p.adjust(p, "holm"),
    holm_sidak = {
      out <- rep(NA_real_, length(p))
      ok <- which(!is.na(p))
      o <- ok[order(p[ok])]
      adj <- 1 - (1 - p[o])^(m - seq_along(o) + 1)
      out[o] <- pmin(1, cummax(adj))
      out
    },
    fdr = stats::p.adjust(p, "BH"),
    aui_stop("Unknown multiple-comparison correction: ", method))
}

# Per-comparison alpha giving simultaneous intervals for single-step methods.
# Step-down methods use the matching single-step interval; FDR has none.
st_ci_alpha <- function(alpha, m, method) {
  switch(method,
    bonferroni = , holm = alpha / m,
    sidak = , holm_sidak = 1 - (1 - alpha)^(1 / m),
    alpha)
}

st_ci_note <- function(method, alpha) {
  lvl <- st_pct(alpha)
  switch(method,
    holm = paste0("Confidence intervals are simultaneous Bonferroni intervals (", lvl,
                  " family-wise); the step-down Holm procedure itself does not define intervals."),
    holm_sidak = paste0("Confidence intervals are simultaneous \u0160id\u00e1k intervals (", lvl,
                        " family-wise); the step-down Holm\u2013\u0160id\u00e1k procedure itself does not define intervals."),
    fdr = paste0("Confidence intervals are ordinary ", lvl, " intervals, not adjusted for multiple comparisons."),
    none = , lsd = paste0("Confidence intervals are ordinary ", lvl, " intervals, not adjusted for multiple comparisons."),
    paste0("Confidence intervals are simultaneous (", lvl, " family-wise)."))
}

# Index pairs to compare. type = "pairs" (all pairs, i < j) or "control".
st_pairs <- function(k, type = "pairs", control = 1L) {
  if (identical(type, "control")) {
    others <- setdiff(seq_len(k), control)
    cbind(others, rep(control, length(others)))
  } else {
    t(utils::combn(k, 2))
  }
}

# Resolve the control level ("" / unknown -> first level).
st_control_index <- function(levels, control) {
  if (is.null(control) || !nzchar(control)) return(1L)
  i <- match(control, levels)
  if (is.na(i)) i <- match(tolower(trimws(control)), tolower(trimws(levels)))
  if (is.na(i)) {
    aui_warn(paste0("The control group \u201c", control, "\u201d was not found; using \u201c",
                    levels[1], "\u201d instead."))
    return(1L)
  }
  i
}

# Two-sided Dunnett P values / critical value for t statistics that share an
# error df. `corr` is the correlation matrix of the comparisons.
st_dunnett_corr <- function(n_treat, n_ctrl) {
  b <- sqrt(n_treat / (n_treat + n_ctrl))
  r <- outer(b, b)
  diag(r) <- 1
  r
}

st_dunnett_p <- function(t, df, corr) {
  aui_need("mvtnorm")
  m <- length(t)
  if (m == 1) return(2 * stats::pt(-abs(t), df))
  set.seed(20240229)
  vapply(abs(t), function(q) {
    if (!is.finite(q)) return(NA_real_)
    p <- 1 - mvtnorm::pmvt(lower = rep(-q, m), upper = rep(q, m), df = max(1, round(df)),
                           corr = corr, algorithm = mvtnorm::GenzBretz(abseps = 1e-5, maxpts = 1e5))[1]
    min(1, max(0, p))
  }, numeric(1))
}

st_dunnett_crit <- function(alpha, df, corr) {
  aui_need("mvtnorm")
  if (nrow(corr) == 1) return(stats::qt(1 - alpha / 2, df))
  set.seed(20240229)
  mvtnorm::qmvt(1 - alpha, tail = "both.tails", df = max(1, round(df)), corr = corr)$quantile
}

# Studentized maximum modulus distribution (m independent comparisons sharing
# df), used for Dunnett's T3.
st_psmm <- function(q, m, df) {
  f <- function(s) (2 * stats::pnorm(q * s) - 1)^m * 2 * s * stats::dchisq(df * s^2, df) * df
  stats::integrate(f, 0, Inf, rel.tol = 1e-8)$value
}

st_qsmm <- function(p, m, df) {
  stats::uniroot(function(q) st_psmm(q, m, df) - p, c(0.5, 200), tol = 1e-7)$root
}

# ---- comparisons among independent group means -------------------------------

# value: numeric, group: factor. Returns a data frame of comparisons.
#   method: tukey, dunnett, bonferroni, sidak, holm_sidak, holm, lsd, fdr,
#           games_howell, none
#   pooled: TRUE = pooled (ANOVA) error term; FALSE = individual variances
st_group_comparisons <- function(value, group, type = "pairs", control = 1L,
                                 method = "tukey", pooled = TRUE, alpha = 0.05) {
  lev <- levels(group)
  k <- length(lev)
  n <- as.numeric(tapply(value, group, length))
  mu <- as.numeric(tapply(value, group, mean))
  v <- as.numeric(tapply(value, group, stats::var))
  pairs <- st_pairs(k, type, control)
  i <- pairs[, 1]; j <- pairs[, 2]
  m <- nrow(pairs)
  diff <- mu[i] - mu[j]
  notes <- character(0)

  # method availability
  if (method == "tukey" && type == "control") {
    method <- "dunnett"
    notes <- c(notes, "Tukey's test applies to all pairwise comparisons; Dunnett's test was used for comparisons with a control.")
  }
  if (method == "dunnett" && type == "pairs") {
    method <- "tukey"
    notes <- c(notes, "Dunnett's test applies to comparisons with a control; Tukey's test was used for all pairwise comparisons.")
  }
  if (method == "games_howell") pooled <- FALSE
  if (!pooled && method == "tukey") method <- "games_howell"

  if (pooled) {
    dfe <- sum(n) - k
    mse <- sum((n - 1) * v) / dfe
    se <- sqrt(mse * (1 / n[i] + 1 / n[j]))
    df <- rep(dfe, m)
  } else {
    if (any(n[unique(c(i, j))] < 2)) aui_stop("Comparisons that do not assume equal SDs need at least two values in every group.")
    se <- sqrt(v[i] / n[i] + v[j] / n[j])
    df <- (v[i] / n[i] + v[j] / n[j])^2 /
      ((v[i] / n[i])^2 / (n[i] - 1) + (v[j] / n[j])^2 / (n[j] - 1))
  }
  t <- diff / se
  p_raw <- 2 * stats::pt(-abs(t), df)

  if (method %in% c("tukey", "games_howell")) {
    p <- stats::ptukey(abs(t) * sqrt(2), k, df, lower.tail = FALSE)
    crit <- stats::qtukey(1 - alpha, k, df) / sqrt(2)
    label <- if (method == "tukey") {
      if (length(unique(n)) > 1) "Tukey\u2013Kramer test (pooled SD, unequal sample sizes)" else "Tukey's test (pooled SD)"
    } else "Games\u2013Howell test (does not assume equal SDs; Welch df)"
  } else if (method == "dunnett") {
    if (pooled) {
      corr <- st_dunnett_corr(n[i], n[j][1])
      p <- st_dunnett_p(t, df[1], corr)
      crit <- rep(st_dunnett_crit(alpha, df[1], corr), m)
      label <- "Dunnett's test (pooled SD)"
    } else {
      p <- vapply(seq_len(m), function(q) 1 - st_psmm(abs(t[q]), m, df[q]), numeric(1))
      crit <- vapply(seq_len(m), function(q) st_qsmm(1 - alpha, m, df[q]), numeric(1))
      label <- "Dunnett's T3 test (does not assume equal SDs; Welch df)"
    }
  } else {
    p <- st_padjust(p_raw, method)
    crit <- stats::qt(1 - st_ci_alpha(alpha, m, method) / 2, df)
    base <- if (pooled) "t tests with the pooled SD from the ANOVA" else "Welch t tests (individual variances)"
    label <- if (method %in% c("none", "lsd")) {
      if (pooled) "Fisher's LSD test (pooled SD; no correction for multiple comparisons)"
      else "Welch t tests, no correction for multiple comparisons"
    } else paste0(base, ", ", st_adjust_label(method), " correction")
  }
  p <- pmin(1, pmax(0, p))
  out <- data.frame(
    Comparison = paste0(lev[i], " \u2212 ", lev[j]),
    diff = diff, se = se, lower = diff - crit * se, upper = diff + crit * se,
    t = t, df = df, p = p, stars = aui_stars(p), stringsAsFactors = FALSE)
  attr(out, "label") <- label
  attr(out, "method") <- method
  attr(out, "notes") <- c(notes, if (method %in% c("holm", "holm_sidak", "fdr", "none", "lsd")) st_ci_note(method, alpha)
                                 else st_ci_note("tukey", alpha))
  out
}

# Emit a comparison table produced by st_group_comparisons / paired variants.
st_emit_comparisons <- function(title, cmp, alpha = 0.05, stat_label = "t", extra_notes = NULL,
                                diff_label = "Mean diff.") {
  labels <- c(diff = diff_label, se = "SE", lower = paste(st_pct(alpha), "CI lower"),
              upper = paste(st_pct(alpha), "CI upper"), t = stat_label, p = "Adjusted P",
              stars = "Summary", z = "z", p_raw = "P (unadjusted)")
  aui_table(title, cmp, formats = c(p = "p", p_raw = "p", stars = "stars"), labels = labels,
            footnotes = c(attr(cmp, "label"), attr(cmp, "notes"), extra_notes))
}

# ---- Dunn's test --------------------------------------------------------------

# Dunn's test on mean ranks with tie correction (as GraphPad Prism).
st_dunn <- function(value, group, type = "pairs", control = 1L, method = "bonferroni") {
  lev <- levels(group)
  N <- length(value)
  r <- rank(value)
  n <- as.numeric(tapply(r, group, length))
  rbar <- as.numeric(tapply(r, group, mean))
  ties <- table(value)
  tie_term <- sum(ties^3 - ties) / (12 * (N - 1))
  pairs <- st_pairs(length(lev), type, control)
  i <- pairs[, 1]; j <- pairs[, 2]
  se <- sqrt((N * (N + 1) / 12 - tie_term) * (1 / n[i] + 1 / n[j]))
  z <- (rbar[i] - rbar[j]) / se
  p_raw <- 2 * stats::pnorm(-abs(z))
  p <- st_padjust(p_raw, method)
  out <- data.frame(Comparison = paste0(lev[i], " \u2212 ", lev[j]),
                    diff = rbar[i] - rbar[j], z = z, p_raw = p_raw, p = p,
                    stars = aui_stars(p), stringsAsFactors = FALSE)
  attr(out, "label") <- paste0("Dunn's test on mean ranks (with tie correction), ",
                               if (method %in% c("none", "lsd")) "no correction for multiple comparisons"
                               else paste0(st_adjust_label(method), " correction"), ".")
  out
}

# ---- assumption checks ----------------------------------------------------------

# D'Agostino-Pearson omnibus K2 test (same algorithm as scipy.stats.normaltest).
st_dagostino <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 8) return(list(statistic = NA_real_, p = NA_real_))
  d <- x - mean(x)
  m2 <- mean(d^2)
  if (m2 <= 0) return(list(statistic = NA_real_, p = NA_real_))
  g1 <- mean(d^3) / m2^1.5
  b2 <- mean(d^4) / m2^2
  y <- g1 * sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
  beta2 <- 3 * (n^2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
  w2 <- -1 + sqrt(2 * (beta2 - 1))
  delta <- 1 / sqrt(0.5 * log(w2))
  a <- sqrt(2 / (w2 - 1))
  if (y == 0) y <- 1
  zs <- delta * log(y / a + sqrt((y / a)^2 + 1))
  e <- 3 * (n - 1) / (n + 1)
  vb2 <- 24 * n * (n - 2) * (n - 3) / ((n + 1)^2 * (n + 3) * (n + 5))
  xx <- (b2 - e) / sqrt(vb2)
  sb1 <- 6 * (n^2 - 5 * n + 2) / ((n + 7) * (n + 9)) * sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
  A <- 6 + 8 / sb1 * (2 / sb1 + sqrt(1 + 4 / sb1^2))
  term1 <- 1 - 2 / (9 * A)
  denom <- 1 + xx * sqrt(2 / (A - 4))
  term2 <- sign(denom) * ((1 - 2 / A) / abs(denom))^(1 / 3)
  zk <- (term1 - term2) / sqrt(2 / (9 * A))
  k2 <- zs^2 + zk^2
  list(statistic = k2, p = stats::pchisq(k2, 2, lower.tail = FALSE))
}

# Table of normality tests for a vector (usually residuals).
st_normality_table <- function(x, alpha = 0.05) {
  x <- x[is.finite(x)]
  n <- length(x)
  rows <- list()
  add <- function(test, stat_name, stat, p) {
    rows[[length(rows) + 1]] <<- data.frame(
      Test = test, stat_name = stat_name, statistic = stat, p = p,
      Passed = if (is.na(p)) "" else if (p > alpha) "Yes" else "No", stringsAsFactors = FALSE)
  }
  constant <- n < 3 || stats::sd(x) < 1e-12 * max(1, abs(mean(x)))
  if (!constant && n <= 5000) {
    sw <- stats::shapiro.test(x)
    add("Shapiro\u2013Wilk", "W", unname(sw$statistic), sw$p.value)
  }
  if (!constant && n >= 8) {
    dp <- st_dagostino(x)
    add("D'Agostino\u2013Pearson omnibus", "K\u00b2", dp$statistic, dp$p)
    if (requireNamespace("nortest", quietly = TRUE)) {
      ad <- nortest::ad.test(x)
      add("Anderson\u2013Darling", "A\u00b2", unname(ad$statistic), ad$p.value)
    }
  }
  if (!constant && n >= 5 && requireNamespace("nortest", quietly = TRUE)) {
    ks <- nortest::lillie.test(x)
    add("Kolmogorov\u2013Smirnov (Lilliefors)", "D", unname(ks$statistic), ks$p.value)
  }
  if (!length(rows)) return(NULL)
  do.call(rbind, rows)
}

st_emit_normality <- function(x, alpha = 0.05, title = "Normality of residuals") {
  tab <- st_normality_table(x, alpha)
  if (is.null(tab)) {
    aui_text("Too few (or constant) residuals to test for normality.", title = title)
    return(invisible(NULL))
  }
  aui_table(title, tab, formats = c(p = "p"),
            labels = c(stat_name = "Statistic", statistic = "Value", p = "P value",
                       Passed = paste0("Consistent with normality (\u03b1 = ", alpha, ")?")),
            footnotes = c("A small P value is evidence that the residuals are not sampled from a normal distribution.",
                          if (length(x) < 8) "The D'Agostino\u2013Pearson and Anderson\u2013Darling tests need at least 8 values."))
  invisible(tab)
}

# Levene-type test: one-way ANOVA on absolute deviations from the group
# centre (median = Brown-Forsythe test, mean = original Levene test).
st_levene <- function(value, group, center = stats::median) {
  group <- droplevels(group)
  z <- abs(value - stats::ave(value, group, FUN = center))
  k <- nlevels(group); N <- length(z)
  if (k < 2 || N - k < 1) return(list(F = NA_real_, df1 = k - 1, df2 = N - k, p = NA_real_))
  zbar <- stats::ave(z, group)
  ssb <- sum((zbar - mean(z))^2)
  ssw <- sum((z - zbar)^2)
  if (ssw <= 0) return(list(F = NA_real_, df1 = k - 1, df2 = N - k, p = NA_real_))
  f <- (ssb / (k - 1)) / (ssw / (N - k))
  list(F = f, df1 = k - 1, df2 = N - k, p = stats::pf(f, k - 1, N - k, lower.tail = FALSE))
}

# Confidence interval for (partial) eta squared from the noncentral F
# distribution.
st_eta2_ci <- function(f, df1, df2, conf = 0.95) {
  if (!is.finite(f) || f <= 0) return(c(0, NA_real_))
  lims <- c((1 + conf) / 2, (1 - conf) / 2)
  ncp <- vapply(lims, function(q) {
    g <- function(ncp) stats::pf(f, df1, df2, ncp) - q
    if (g(0) < 0) return(0)
    hi <- max(10, f * df1 * 3)
    for (it in 1:40) { if (g(hi) < 0) break; hi <- hi * 2 }
    tryCatch(stats::uniroot(g, c(0, hi), tol = 1e-8)$root, error = function(e) NA_real_)
  }, numeric(1))
  ncp / (ncp + df1 + df2 + 1)
}

# ---- diagnostic plots -----------------------------------------------------------

st_qq_plot <- function(res, title = "Normal QQ plot of residuals") {
  res <- res[is.finite(res)]
  d <- data.frame(sample = sort(res), theo = stats::qnorm(stats::ppoints(length(res))))
  q <- stats::quantile(res, c(0.25, 0.75), names = FALSE)
  slope <- diff(q) / diff(stats::qnorm(c(0.25, 0.75)))
  ggplot2::ggplot(d, ggplot2::aes(theo, sample)) +
    ggplot2::geom_abline(intercept = q[1] - slope * stats::qnorm(0.25), slope = slope,
                         colour = "#7A8B99", linetype = "dashed") +
    ggplot2::geom_point(colour = AUI_COLORS[1], size = 1.9, alpha = 0.85) +
    ggplot2::labs(x = "Theoretical quantiles", y = "Residuals", title = title) + aui_theme()
}

st_resid_plot <- function(fitted, res, title = "Residuals vs fitted values", ylab = "Residuals") {
  d <- data.frame(fitted = fitted, res = res)
  ggplot2::ggplot(d, ggplot2::aes(fitted, res)) +
    ggplot2::geom_hline(yintercept = 0, colour = "#7A8B99", linetype = "dashed") +
    ggplot2::geom_point(colour = AUI_COLORS[1], size = 1.9, alpha = 0.85) +
    ggplot2::labs(x = "Fitted values", y = ylab, title = title) + aui_theme()
}

# Residual-vs-fitted and QQ plots side by side.
st_emit_residual_plots <- function(fitted, res, title = "Residual plots") {
  p1 <- st_resid_plot(fitted, res)
  p2 <- st_qq_plot(res)
  if (requireNamespace("patchwork", quietly = TRUE)) {
    aui_plot(title, patchwork::wrap_plots(p1, p2, nrow = 1), width = 8.4, height = 3.8)
  } else {
    aui_plot("Residuals vs fitted values", p1, width = 5, height = 3.8)
    aui_plot("Normal QQ plot of residuals", p2, width = 5, height = 3.8)
  }
}

st_colors <- function(n) {
  if (n <= length(AUI_COLORS)) AUI_COLORS[seq_len(n)] else grDevices::colorRampPalette(AUI_COLORS)(n)
}

# Group means with CI and the individual points.
st_means_plot <- function(value, group, alpha = 0.05, ylab = "Value", xlab = NULL, connect = FALSE) {
  d <- data.frame(value = value, group = group)
  n <- tapply(value, group, length); mu <- tapply(value, group, mean); s <- tapply(value, group, stats::sd)
  half <- stats::qt(1 - alpha / 2, pmax(n - 1, 1)) * s / sqrt(n)
  half[n < 2] <- NA
  sm <- data.frame(group = factor(levels(group), levels = levels(group)), mean = as.numeric(mu),
                   lower = as.numeric(mu - half), upper = as.numeric(mu + half))
  k <- nlevels(group)
  p <- ggplot2::ggplot(d, ggplot2::aes(group, value, colour = group)) +
    ggplot2::geom_point(position = ggplot2::position_jitter(width = 0.12, height = 0, seed = 1),
                        alpha = 0.55, size = 1.8)
  if (connect) p <- p + ggplot2::geom_line(data = sm, ggplot2::aes(group, mean, group = 1),
                                           colour = "#7A8B99", inherit.aes = FALSE)
  p + ggplot2::geom_errorbar(data = sm[!is.na(sm$lower), , drop = FALSE],
                             ggplot2::aes(x = group, ymin = lower, ymax = upper, colour = group),
                             width = 0.18, linewidth = 0.7, inherit.aes = FALSE) +
    ggplot2::geom_point(data = sm, ggplot2::aes(group, mean, colour = group), size = 3.2, shape = 18,
                        inherit.aes = FALSE) +
    ggplot2::scale_colour_manual(values = st_colors(k), guide = "none") +
    ggplot2::labs(x = xlab, y = ylab) + aui_theme() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = if (k > 5 || max(nchar(levels(group))) > 12) 30 else 0,
                                                       hjust = if (k > 5 || max(nchar(levels(group))) > 12) 1 else 0.5))
}

# Group descriptives: n, mean, SD, SEM, CI.
st_describe_groups <- function(value, group, alpha = 0.05, median = FALSE) {
  n <- as.numeric(tapply(value, group, length))
  mu <- as.numeric(tapply(value, group, mean))
  s <- as.numeric(tapply(value, group, stats::sd))
  sem <- s / sqrt(n)
  half <- stats::qt(1 - alpha / 2, pmax(n - 1, 1)) * sem
  out <- data.frame(Group = levels(group), n = n, mean = mu, sd = s, sem = sem,
                    lower = mu - half, upper = mu + half, stringsAsFactors = FALSE)
  if (median) out$median <- as.numeric(tapply(value, group, stats::median))
  out
}

ST_DESC_LABELS <- function(alpha) c(mean = "Mean", sd = "SD", sem = "SEM", median = "Median",
                                    lower = paste(st_pct(alpha), "CI lower"),
                                    upper = paste(st_pct(alpha), "CI upper"))

# ---- emmeans helpers --------------------------------------------------------------

# levmap: optional list factor -> named character vector (code -> label), for
# models fitted with recoded factor levels.
st_relabel <- function(v, f, levmap) {
  v <- as.character(v)
  map <- levmap[[f]]
  if (is.null(map)) v else unname(ifelse(v %in% names(map), map[v], v))
}

st_emmeans <- function(fit, specs, by = NULL, ...) {
  aui_need("emmeans")
  args <- list(fit, specs = specs, by = by, ...)
  if (inherits(fit, "merMod")) args$lmer.df <- "satterthwaite"
  st_quiet(do.call(emmeans::emmeans, args))$value
}

# Emit a table of estimated marginal means; returns the summary data frame
# (factor columns relabelled; columns emmean, SE, df, lower, upper).
st_emit_emm <- function(fit, specs, names, alpha = 0.05, levmap = list(), title = NULL,
                        footnotes = NULL, emit = TRUE, ...) {
  emm <- st_emmeans(fit, specs, ...)
  s <- as.data.frame(st_quiet(summary(emm, level = 1 - alpha, infer = c(TRUE, FALSE)))$value)
  est <- intersect(c("emmean", "response", "prob", "rate"), names(s))[1]
  lo <- intersect(c("lower.CL", "asymp.LCL"), names(s))[1]
  hi <- intersect(c("upper.CL", "asymp.UCL"), names(s))[1]
  out <- data.frame(row.names = seq_len(nrow(s)))
  for (f in specs) out[[f]] <- st_relabel(s[[f]], f, levmap)
  out$emmean <- s[[est]]; out$SE <- s$SE; out$df <- s$df
  out$lower <- s[[lo]]; out$upper <- s[[hi]]
  if (length(specs) > 1) {
    out <- out[do.call(order, lapply(specs, function(f) match(out[[f]], unique(out[[f]])))), , drop = FALSE]
  }
  out <- out[!is.na(out$emmean), , drop = FALSE]
  if (emit) {
    shown <- out
    names(shown)[seq_along(specs)] <- make.unique(vapply(specs, function(f) st_name(names, f), ""))
    aui_table(title %||% paste0("Estimated marginal means: ", st_pretty(paste(specs, collapse = ":"), names)),
              shown, labels = c(emmean = "Mean", lower = paste(st_pct(alpha), "CI lower"),
                                upper = paste(st_pct(alpha), "CI upper")),
              footnotes = footnotes)
  }
  invisible(out)
}

# Comparisons among the levels of `factor` (optionally within each level of
# `by`) from a fitted model, via emmeans.
#   type: "pairs" | "control";  method: tukey, dunnett, bonferroni, sidak, holm, none
st_emit_emm_comparisons <- function(fit, factor, by = NULL, names, type = "pairs", control = "",
                                    method = "tukey", alpha = 0.05, levmap = list(),
                                    title = NULL, extra_notes = NULL, ...) {
  emm <- st_emmeans(fit, factor, by = by, ...)
  codes <- emm@levels[[factor]]
  lev <- st_relabel(codes, factor, levmap)
  k <- length(lev)
  if (k < 2) return(invisible(NULL))
  notes <- character(0)
  if (method == "tukey" && type == "control") {
    method <- "dunnett"
    notes <- c(notes, "Tukey's test applies to all pairwise comparisons; Dunnett's test was used for comparisons with a control.")
  }
  if (method == "dunnett" && type == "pairs") {
    method <- "tukey"
    notes <- c(notes, "Dunnett's test applies to comparisons with a control; Tukey's test was used for all pairwise comparisons.")
  }
  ctrl <- if (type == "control") st_control_index(lev, control) else 1L
  con <- if (type == "control") emmeans::contrast(emm, "trt.vs.ctrl", ref = ctrl)
         else emmeans::contrast(emm, "pairwise")
  adj <- switch(method, tukey = "tukey", dunnett = "mvt", bonferroni = "bonferroni",
                sidak = "sidak", holm = "holm", none = "none", "none")
  set.seed(20240229)
  s <- as.data.frame(st_quiet(summary(con, infer = c(TRUE, TRUE), adjust = adj, level = 1 - alpha))$value)
  pairs <- st_pairs(k, type, ctrl)
  n_by <- if (is.null(by)) 1L else nrow(unique(s[, by, drop = FALSE]))
  labels <- paste0(lev[pairs[, 1]], " \u2212 ", lev[pairs[, 2]])
  comparison <- if (nrow(s) == length(labels) * n_by) rep(labels, times = n_by)
                else gsub(" - ", " \u2212 ", as.character(s$contrast), fixed = TRUE)
  stat <- intersect(c("t.ratio", "z.ratio"), names(s))[1]
  lo <- intersect(c("lower.CL", "asymp.LCL"), names(s))[1]
  hi <- intersect(c("upper.CL", "asymp.UCL"), names(s))[1]
  out <- data.frame(row.names = seq_len(nrow(s)))
  for (b in by) out[[b]] <- st_relabel(s[[b]], b, levmap)
  out$Comparison <- comparison
  out$diff <- s$estimate; out$se <- s$SE; out$df <- s$df
  out$lower <- s[[lo]]; out$upper <- s[[hi]]
  out$t <- s[[stat]]; out$p <- s$p.value; out$stars <- aui_stars(s$p.value)
  out <- out[!is.na(out$diff), , drop = FALSE]
  if (!nrow(out)) return(invisible(NULL))
  names(out)[seq_along(by)] <- make.unique(vapply(by, function(f) st_name(names, f), ""))
  m <- nrow(out) / n_by
  fam <- if (!is.null(by)) " Each level of the conditioning factor is treated as a separate family of comparisons." else ""
  method_note <- if (m <= 1) "t test on the estimated marginal means (a single comparison needs no correction)." else switch(method,
    tukey = "Tukey's test on the estimated marginal means.",
    dunnett = "Dunnett's test (multivariate t distribution) on the estimated marginal means.",
    none = "t tests on the estimated marginal means, no correction for multiple comparisons.",
    paste0("t tests on the estimated marginal means, ", st_adjust_label(method), " correction."))
  ci_note <- if (m <= 1) NULL else st_ci_note(if (method %in% c("tukey", "dunnett", "bonferroni", "sidak")) "tukey" else method, alpha)
  labs <- c(diff = "Difference", se = "SE", lower = paste(st_pct(alpha), "CI lower"),
            upper = paste(st_pct(alpha), "CI upper"), t = if (identical(stat, "z.ratio")) "z" else "t",
            p = if (method == "none" || m <= 1) "P value" else "Adjusted P", stars = "Summary")
  aui_table(title %||% paste0("Multiple comparisons: ", st_name(names, factor)), out,
            formats = c(p = "p", stars = "stars"), labels = labs,
            footnotes = c(paste0(method_note, fam), notes, ci_note, extra_notes))
  invisible(out)
}

# ---- model helpers ------------------------------------------------------------------

# Human-readable labels for the coefficients of a model fitted with treatment
# (reference-level) coding. Falls back to prettified raw names.
st_coef_labels <- function(fit_terms, mm, data, names) {
  raw <- colnames(mm)
  out <- st_pretty(raw, names)
  out[raw == "(Intercept)"] <- "Intercept"
  assign <- attr(mm, "assign")
  labs <- attr(fit_terms, "term.labels")
  if (is.null(assign)) return(out)
  for (ti in seq_along(labs)) {
    idx <- which(assign == ti)
    if (!length(idx)) next
    parts <- strsplit(labs[ti], ":", fixed = TRUE)[[1]]
    pieces <- lapply(parts, function(v) {
      if (v %in% names(data) && is.factor(data[[v]])) {
        paste0(st_name(names, v), " [", levels(data[[v]])[-1], "]")
      } else st_pretty(v, names)
    })
    grid <- do.call(expand.grid, c(pieces, stringsAsFactors = FALSE))
    lab <- apply(grid, 1, paste, collapse = " \u00d7 ")
    if (length(lab) == length(idx)) out[idx] <- lab
  }
  out
}

# Sentence naming the reference level of every factor in the model.
st_reference_note <- function(data, factors, names) {
  if (!length(factors)) return(NULL)
  paste0("Reference levels: ", paste(vapply(factors, function(f) {
    paste0(st_name(names, f), " = ", levels(data[[f]])[1])
  }, ""), collapse = "; "), ". Coefficients of a factor compare each level with its reference level.")
}

# Right-hand side of a model formula from options.
#   interactions: "none"/"main", "twoway", "full", "custom"
st_rhs <- function(terms, interactions = "none", custom = NULL) {
  if (identical(interactions, "custom")) {
    if (is.null(custom) || !nzchar(trimws(custom))) aui_stop("Type the model terms in the \u201cCustom model\u201d box, or choose another interaction setting.")
    return(custom)
  }
  if (!length(terms)) return("1")
  if (length(terms) == 1 || interactions %in% c("none", "main")) return(paste(terms, collapse = " + "))
  if (interactions == "twoway") return(paste0("(", paste(terms, collapse = " + "), ")^2"))
  paste(terms, collapse = " * ")
}

# Cross-tabulation check for factors: returns labels of empty cells.
st_empty_cells <- function(data, factors, names) {
  if (length(factors) < 2) return(character(0))
  tab <- as.data.frame(table(data[, factors, drop = FALSE]), stringsAsFactors = FALSE)
  empty <- tab[tab$Freq == 0, factors, drop = FALSE]
  if (!nrow(empty)) return(character(0))
  apply(empty, 1, function(r) paste(paste0(vapply(factors, function(f) st_name(names, f), ""), " = ", r), collapse = ", "))
}

st_fmt <- function(x, digits = 4) trimws(formatC(as.numeric(x), digits = digits, format = "g"))

# "F(2, 27) = 4.56"
st_fmt_f <- function(f, df1, df2) {
  d <- function(v) if (abs(v - round(v)) < 1e-6) as.character(round(v)) else formatC(v, format = "f", digits = 2)
  sprintf("F(%s, %s) = %s", d(df1), d(df2), formatC(f, format = "f", digits = if (abs(f) >= 100) 1 else 2))
}

st_sig <- function(p, alpha) if (is.na(p)) "could not be tested" else if (p < alpha) "statistically significant" else "not statistically significant"

# Options for the "create a graph" links of group-layout analyses (keeps
# `columns` a JSON array even when a single column is selected).
st_layout_options <- function(options) {
  keep <- options[intersect(c("layout", "dep", "group", "subject"), names(options))]
  keep <- keep[!vapply(keep, is.null, logical(1))]
  if (!is.null(options$columns)) keep$columns <- as.list(aui_vars(options, "columns"))
  keep
}

# "P = 0.0123" / "P < 0.0001" for sentences.
st_p_text <- function(p) {
  if (is.na(p)) "P not available" else if (p < 1e-4) "P < 0.0001" else paste0("P = ", formatC(p, format = "f", digits = 4))
}

st_cap <- function(x) sub("^(.)", "\\U\\1", x, perl = TRUE)

# Comparisons among matched conditions. wide: matrix subjects x conditions.
#   pooled = TRUE : pooled residual MS of the RM ANOVA (assumes sphericity)
#   pooled = FALSE: every comparison uses the SD of its own paired differences
st_paired_comparisons <- function(wide, type = "pairs", control = 1L, method = "tukey",
                                  pooled = FALSE, ms_resid = NA, df_resid = NA, alpha = 0.05) {
  lev <- colnames(wide)
  k <- ncol(wide); n <- nrow(wide)
  pairs <- st_pairs(k, type, control)
  i <- pairs[, 1]; j <- pairs[, 2]; m <- nrow(pairs)
  notes <- character(0)
  if (method == "tukey" && type == "control") {
    method <- "dunnett"
    notes <- c(notes, "Tukey's test applies to all pairwise comparisons; Dunnett's test was used for comparisons with a control.")
  }
  if (method == "dunnett" && type == "pairs") {
    method <- "tukey"
    notes <- c(notes, "Dunnett's test applies to comparisons with a control; Tukey's test was used for all pairwise comparisons.")
  }
  diff <- colMeans(wide)[i] - colMeans(wide)[j]
  if (pooled) {
    se <- rep(sqrt(2 * ms_resid / n), m); df <- df_resid
  } else {
    se <- vapply(seq_len(m), function(q) stats::sd(wide[, i[q]] - wide[, j[q]]) / sqrt(n), numeric(1)); df <- n - 1
  }
  t <- diff / se
  p_raw <- 2 * stats::pt(-abs(t), df)
  if (method == "tukey") {
    p <- stats::ptukey(abs(t) * sqrt(2), k, df, lower.tail = FALSE)
    crit <- stats::qtukey(1 - alpha, k, df) / sqrt(2)
  } else if (method == "dunnett") {
    corr <- matrix(0.5, m, m); diag(corr) <- 1
    p <- st_dunnett_p(t, df, corr)
    crit <- st_dunnett_crit(alpha, df, corr)
  } else {
    p <- st_padjust(p_raw, method)
    crit <- stats::qt(1 - st_ci_alpha(alpha, m, method) / 2, df)
  }
  out <- data.frame(Comparison = paste0(lev[i], " \u2212 ", lev[j]), diff = unname(diff), se = se,
                    lower = unname(diff) - crit * se, upper = unname(diff) + crit * se, t = unname(t),
                    df = df, p = pmin(1, p), stars = aui_stars(p), stringsAsFactors = FALSE)
  name <- switch(method, tukey = "Tukey's test", dunnett = "Dunnett's test",
                 none = "Paired t tests, no correction for multiple comparisons",
                 paste0("Paired t tests with ", st_adjust_label(method), " correction"))
  attr(out, "label") <- paste0(name, if (pooled)
    sprintf("; pooled error term of the ANOVA (assumes sphericity), df = %s.", format(df))
    else sprintf("; each comparison uses the SD of its own paired differences (no sphericity assumption), df = n \u2212 1 = %d.", n - 1))
  attr(out, "notes") <- c(notes, st_ci_note(if (method %in% c("holm", "none")) method else "tukey", alpha))
  out
}

# Sphericity diagnostics from a subjects x conditions matrix.
st_sphericity <- function(wide) {
  k <- ncol(wide); n <- nrow(wide); p <- k - 1
  C <- stats::contr.helmert(k)
  C <- sweep(C, 2, sqrt(colSums(C^2)), "/")
  S <- t(C) %*% stats::cov(wide) %*% C
  ev <- eigen(S, symmetric = TRUE, only.values = TRUE)$values
  gg <- sum(ev)^2 / (p * sum(ev^2))
  hf <- min(1, (n * p * gg - 2) / (p * (n - 1 - p * gg)))
  out <- list(gg = gg, hf = max(hf, gg * 0 + 1 / p), lb = 1 / p, W = NA_real_, chi2 = NA_real_, df = NA_real_, p = NA_real_)
  if (p >= 2 && n > k && all(ev > 0)) {
    W <- exp(sum(log(ev)) - p * log(mean(ev)))
    f <- 1 - (2 * p^2 + p + 2) / (6 * p * (n - 1))
    out$W <- W
    out$chi2 <- -(n - 1) * f * log(W)
    out$df <- p * (p + 1) / 2 - 1
    out$p <- stats::pchisq(out$chi2, out$df, lower.tail = FALSE)
  }
  out
}

# ---- factorial designs: shared output blocks ------------------------------------

# Display column name -> safe factor name ("" / unknown -> default).
st_resolve_factor <- function(display, factors, names, default) {
  if (is.null(display) || !nzchar(display)) return(default)
  hit <- factors[vapply(factors, function(f) identical(st_name(names, f), display), logical(1))]
  if (length(hit)) hit[1] else default
}

# Main-effect and simple-effect comparisons according to the standard options
# (posthoc, simple_factor, simple_by, compare, control, posthoc_method).
# `available`: optional function(factor, by) -> TRUE when the comparison can be made.
st_factorial_posthoc <- function(fit, factors, names, options, alpha, levmap = list(), ...) {
  what <- aui_opt(options, "posthoc", "none")
  if (identical(what, "none")) return(invisible(NULL))
  type <- aui_opt(options, "compare", "pairs")
  method <- aui_opt(options, "posthoc_method", "tukey")
  control <- aui_opt(options, "control", "")
  aui_heading("Multiple comparisons")
  if (what %in% c("main", "both")) {
    for (f in factors) {
      st_emit_emm_comparisons(
        fit, f, NULL, names, type = type, control = control, method = method, alpha = alpha, levmap = levmap,
        title = paste0("Main effect comparisons: ", st_name(names, f)),
        extra_notes = if (length(factors) > 1) "Levels are compared after averaging over the other factors; interpret with care when this factor is involved in an interaction.",
        ...)
    }
  }
  if (what %in% c("simple", "both")) {
    if (length(factors) < 2) {
      aui_text("Simple effects need at least two factors.", title = "Simple effects")
    } else {
      f <- st_resolve_factor(aui_opt(options, "simple_factor", ""), factors, names, factors[1])
      by <- st_resolve_factor(aui_opt(options, "simple_by", ""), factors, names, setdiff(factors, f)[1])
      if (identical(by, f)) by <- setdiff(factors, f)[1]
      others <- setdiff(factors, c(f, by))
      st_emit_emm_comparisons(
        fit, f, by, names, type = type, control = control, method = method, alpha = alpha, levmap = levmap,
        title = paste0("Simple effects: ", st_name(names, f), " within each level of ", st_name(names, by)),
        extra_notes = if (length(others)) paste0("Averaged over the levels of ", paste(vapply(others, function(o) st_name(names, o), ""), collapse = ", "), "."),
        ...)
    }
  }
  invisible(NULL)
}

# Plot of estimated marginal means: x = first factor, colour = second, facet = third.
st_emm_plot <- function(emm, factors, names, ylab, alpha = 0.05) {
  factors <- factors[seq_len(min(3, length(factors)))]
  for (f in factors) emm[[f]] <- factor(emm[[f]], levels = unique(emm[[f]]))
  x <- factors[1]
  if (length(factors) > 1) {
    g <- factors[2]
    dodge <- ggplot2::position_dodge(width = 0.25)
    p <- ggplot2::ggplot(emm, ggplot2::aes(.data[[x]], emmean, colour = .data[[g]], group = .data[[g]])) +
      ggplot2::geom_line(position = dodge, linewidth = 0.7) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower, ymax = upper), width = 0.12, position = dodge, linewidth = 0.6) +
      ggplot2::geom_point(position = dodge, size = 2.6) +
      ggplot2::scale_colour_manual(values = st_colors(nlevels(emm[[g]])), name = st_name(names, g)) +
      aui_theme() + ggplot2::theme(legend.title = ggplot2::element_text())
  } else {
    p <- ggplot2::ggplot(emm, ggplot2::aes(.data[[x]], emmean, group = 1)) +
      ggplot2::geom_line(linewidth = 0.7, colour = AUI_COLORS[1]) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = lower, ymax = upper), width = 0.12, linewidth = 0.6, colour = AUI_COLORS[1]) +
      ggplot2::geom_point(size = 2.6, colour = AUI_COLORS[1]) + aui_theme()
  }
  p <- p + ggplot2::labs(x = st_name(names, x), y = ylab)
  if (length(factors) > 2) p <- p + ggplot2::facet_wrap(stats::as.formula(paste("~", factors[3])))
  if (nlevels(emm[[x]]) > 5 || max(nchar(levels(emm[[x]]))) > 12) {
    p <- p + ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
  }
  p
}

# n / mean / SD / SEM for every cell of a factorial design.
st_cell_descriptives <- function(y, data, factors, names, levmap = list()) {
  key <- interaction(data[, factors, drop = FALSE], drop = TRUE, lex.order = TRUE)
  first <- !duplicated(key)
  ord <- order(key[first])
  out <- data.frame(row.names = seq_len(sum(first)))
  for (f in factors) out[[f]] <- st_relabel(data[[f]][first][ord], f, levmap)
  lv <- levels(key)
  n <- as.numeric(tapply(y, key, length)); s <- as.numeric(tapply(y, key, stats::sd))
  out$n <- n[match(as.character(key[first][ord]), lv)]
  out$mean <- as.numeric(tapply(y, key, mean))[match(as.character(key[first][ord]), lv)]
  out$sd <- s[match(as.character(key[first][ord]), lv)]
  out$sem <- out$sd / sqrt(out$n)
  names(out)[seq_along(factors)] <- make.unique(vapply(factors, function(f) st_name(names, f), ""))
  out
}

# One-sentence-per-effect summary of an ANOVA table.
st_effects_summary <- function(lead, terms, f, df1, df2, p, alpha, names) {
  lab <- st_pretty(terms, names)
  parts <- sprintf("%s: %s, %s", lab, mapply(st_fmt_f, f, df1, df2), vapply(p, st_p_text, ""))
  sig <- !is.na(p) & p < alpha
  txt <- paste0(lead, " ", paste(parts, collapse = "; "), ". ",
                if (!any(sig)) "No effect is statistically significant."
                else paste0("Significant: ", paste(lab[sig], collapse = ", "), "."))
  if (any(sig & grepl(":", terms, fixed = TRUE))) {
    txt <- paste0(txt, " Because an interaction is significant, the effect of one factor depends on the level of another; look at the simple effects rather than the main effects alone.")
  }
  txt
}

# Recode the levels of the given factors to safe codes (L01, L02, ...).
# Returns list(data, levmap) where levmap[[f]] maps code -> original label.
st_recode_levels <- function(data, factors) {
  levmap <- list()
  for (f in factors) {
    lv <- levels(data[[f]])
    codes <- sprintf("L%02d", seq_along(lv))
    levmap[[f]] <- stats::setNames(lv, codes)
    data[[f]] <- factor(codes[as.integer(data[[f]])], levels = codes)
  }
  list(data = data, levmap = levmap)
}

# Variance components table of an lmer fit; returns the VarCorr data frame.
st_emit_varcomp <- function(fit, names, title = "Random effects (variance components)", footnotes = NULL) {
  vc <- as.data.frame(lme4::VarCorr(fit))
  is_var <- is.na(vc$var2)
  total <- sum(vc$vcov[is_var & (is.na(vc$var1) | vc$var1 == "(Intercept)")])
  grp <- ifelse(vc$grp == "Residual", "Residual", st_pretty(vc$grp, names))
  what <- ifelse(vc$grp == "Residual", "",
                 ifelse(is_var, ifelse(vc$var1 == "(Intercept)", "Intercept", paste0("Slope: ", st_pretty(vc$var1, names))),
                        paste0("Correlation: ", ifelse(vc$var1 == "(Intercept)", "Intercept", st_pretty(vc$var1, names)),
                               " ~ ", st_pretty(vc$var2, names))))
  aui_table(title, data.frame(
    Group = grp, Effect = what, Variance = ifelse(is_var, vc$vcov, NA), SD = ifelse(is_var, vc$sdcor, NA),
    Correlation = as.numeric(ifelse(is_var, NA, vc$sdcor)),
    share = ifelse(is_var & (is.na(vc$var1) | vc$var1 == "(Intercept)"), vc$vcov / total, NA),
    stringsAsFactors = FALSE),
    formats = c(share = "pct"), labels = c(share = "% of variance (intercepts + residual)"), footnotes = footnotes)
  invisible(vc)
}

# ---- regression helpers -------------------------------------------------------------

# Stop with a clear message when coefficients are aliased (perfect collinearity).
st_check_aliased <- function(coefs, labels) {
  bad <- is.na(coefs)
  if (!any(bad)) return(invisible(NULL))
  aui_stop("Perfect collinearity: the following term", if (sum(bad) > 1) "s" else "", " cannot be estimated because ",
           if (sum(bad) > 1) "they are exact combinations" else "it is an exact combination", " of other predictors: ",
           paste(labels[bad], collapse = ", "), ". Remove the redundant predictor",
           " (for example one of two variables that carry the same information, or a category that is fully determined by another variable).")
}

# Variance inflation factor of every non-intercept column of a model matrix.
st_vif <- function(mm) {
  out <- rep(NA_real_, ncol(mm))
  keep <- colnames(mm) != "(Intercept)"
  if (sum(keep) < 2 || all(keep)) return(out)
  r <- tryCatch(solve(stats::cor(mm[, keep, drop = FALSE])), error = function(e) NULL)
  if (!is.null(r)) out[keep] <- diag(r)
  out
}

# Model formula text with display names.
st_formula_text <- function(form, names) st_pretty(paste(deparse(form, width.cutoff = 500), collapse = " "), names)

# Coefficient table for GLM-type models on the link scale plus ratio scale.
#   ci: matrix (lower, upper) on the link scale
st_ratio_table <- function(labels, est, se, stat, p, ci, drop_intercept = TRUE) {
  d <- data.frame(Term = labels, ratio = exp(est), lower = exp(ci[, 1]), upper = exp(ci[, 2]), p = p,
                  stars = aui_stars(p), stringsAsFactors = FALSE)
  if (drop_intercept) d <- d[labels != "Intercept", , drop = FALSE]
  d
}

# Confidence intervals for a glm: profile likelihood with Wald fallback.
# Returns list(ci = matrix, method = "profile" | "wald").
st_glm_ci <- function(fit, alpha, method = "profile") {
  co <- stats::coef(summary(fit))
  est <- co[, 1]; se <- co[, 2]
  crit <- if (colnames(co)[3] == "t value") stats::qt(1 - alpha / 2, stats::df.residual(fit)) else stats::qnorm(1 - alpha / 2)
  wald <- cbind(est - crit * se, est + crit * se)
  if (identical(method, "profile")) {
    ci <- tryCatch(st_quiet(stats::confint(fit, level = 1 - alpha))$value, error = function(e) NULL)
    if (!is.null(ci)) {
      ci <- matrix(ci, ncol = 2)
      if (nrow(ci) == length(est) && !anyNA(ci)) return(list(ci = ci, method = "profile"))
    }
  }
  list(ci = wald, method = "wald")
}
