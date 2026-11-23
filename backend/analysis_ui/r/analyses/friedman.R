# Friedman test with Dunn's multiple comparisons on rank sums.
# data: value, group (factor: conditions), subject (factor); complete subjects only
run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  group <- droplevels(data$group)
  lev <- levels(group)
  wide <- tapply(data$value, list(data$subject, group), function(v) v[1])
  wide <- wide[stats::complete.cases(wide), , drop = FALSE]
  n <- nrow(wide); k <- ncol(wide)
  if (k < 2) aui_stop("The Friedman test needs at least two matched conditions.")
  if (n < 2) aui_stop("At least two subjects with values in every condition are needed.")
  ranks <- t(apply(wide, 1, rank))
  if (all(apply(wide, 1, function(r) length(unique(r)) == 1))) {
    aui_stop("Every subject has identical values in all conditions, so the conditions cannot be ranked.")
  }
  fr <- stats::friedman.test(wide)
  chi2 <- unname(fr$statistic); df <- unname(fr$parameter); p <- fr$p.value
  w <- chi2 / (n * (k - 1))
  aui_table("Friedman test", data.frame(
    chi2 = chi2, df = df, p = p, stars = aui_stars(p), n = n, W = w, stringsAsFactors = FALSE),
    formats = c(df = "int", n = "int", p = "p", stars = "stars"),
    labels = c(chi2 = "Friedman statistic (χ²)", p = "P value", stars = "Summary",
               n = "Subjects", W = "Kendall's W"),
    footnotes = c("Approximate P value from the chi-square distribution; the statistic is corrected for ties within subjects.",
                  "Kendall's W (coefficient of concordance) = χ² / (n (k − 1)); 0 = no agreement among subjects on the ordering of conditions, 1 = perfect agreement.",
                  st_dropped_note(options, "subject"),
                  if (n < 6) "With few subjects the chi-square approximation may be inaccurate."))

  rsum <- colSums(ranks)
  if (aui_flag(options, "descriptives", TRUE)) {
    aui_table("Medians and rank sums", data.frame(
      Condition = lev, n = n, median = apply(wide, 2, stats::median),
      q1 = apply(wide, 2, stats::quantile, probs = 0.25, names = FALSE),
      q3 = apply(wide, 2, stats::quantile, probs = 0.75, names = FALSE),
      rank_sum = rsum, mean_rank = rsum / n, stringsAsFactors = FALSE),
      formats = c(n = "int"),
      labels = c(median = "Median", q1 = "25th percentile", q3 = "75th percentile",
                 rank_sum = "Sum of ranks", mean_rank = "Mean rank"),
      footnotes = "Values are ranked within each subject.")
  }

  compare <- aui_opt(options, "compare", "pairs")
  n_sig <- NA
  if (compare != "none") {
    ctrl <- if (compare == "control") st_control_index(lev, aui_opt(options, "control", "")) else 1L
    method <- aui_opt(options, "adjust", "bonferroni")
    pairs <- st_pairs(k, compare, ctrl)
    i <- pairs[, 1]; j <- pairs[, 2]
    z <- (rsum[i] - rsum[j]) / sqrt(n * k * (k + 1) / 6)
    p_raw <- 2 * stats::pnorm(-abs(z))
    p_adj <- st_padjust(p_raw, method)
    aui_table(if (compare == "control") paste0("Dunn's multiple comparisons: each condition vs ", lev[ctrl])
              else "Dunn's multiple comparisons: all pairs",
              data.frame(Comparison = paste0(lev[i], " − ", lev[j]), diff = unname(rsum[i] - rsum[j]),
                         z = unname(z), p_raw = p_raw, p = p_adj, stars = aui_stars(p_adj), stringsAsFactors = FALSE),
              formats = c(p_raw = "p", p = "p", stars = "stars"),
              labels = c(diff = "Rank sum diff.", p_raw = "P (unadjusted)", p = "Adjusted P", stars = "Summary"),
              footnotes = paste0("Dunn's test on the within-subject rank sums: z = difference / √(n k (k + 1) / 6), ",
                                 if (method == "none") "no correction for multiple comparisons." else paste0(st_adjust_label(method), " correction.")))
    n_sig <- sum(p_adj < alpha); n_cmp <- length(p_adj)
  }

  if (aui_flag(options, "show_plot", TRUE)) {
    d <- data.frame(value = as.vector(wide), group = factor(rep(lev, each = n), levels = lev),
                    subject = rep(rownames(wide), times = k))
    med <- data.frame(group = factor(lev, levels = lev), value = apply(wide, 2, stats::median))
    pl <- ggplot2::ggplot(d, ggplot2::aes(group, value)) +
      ggplot2::geom_line(ggplot2::aes(group = subject), colour = "#7A8B99", alpha = 0.45) +
      ggplot2::geom_point(colour = "#7A8B99", alpha = 0.6, size = 1.5) +
      ggplot2::geom_line(data = med, ggplot2::aes(group = 1), colour = AUI_COLORS[2], linewidth = 1) +
      ggplot2::geom_point(data = med, colour = AUI_COLORS[2], size = 3) +
      ggplot2::labs(x = NULL, y = st_name(names, "value")) + aui_theme()
    aui_plot("Each subject's values (medians in colour)", pl, width = max(4.2, min(9, 1.6 + 0.9 * k)), height = 3.9)
  }

  txt <- sprintf("Friedman test: the %d matched conditions %s (χ² = %s, df = %d, %s; Kendall's W = %s, n = %d subjects).",
                 k, if (p < alpha) "differ significantly" else "do not differ significantly",
                 formatC(chi2, format = "f", digits = 2), df, st_p_text(p), formatC(w, format = "f", digits = 3), n)
  if (!is.na(n_sig)) txt <- paste0(txt, sprintf(" Dunn's test: %d of %d comparison%s significant at α = %s.",
                                                n_sig, n_cmp, if (n_cmp == 1) " is" else "s are", format(alpha)))
  aui_summary(txt)
  aui_graph("Create a before\u2013after graph of these data", "before_after", st_layout_options(options))
}
