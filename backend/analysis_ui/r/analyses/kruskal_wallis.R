# Kruskal-Wallis test with Dunn's multiple comparisons.
# data: value (numeric), group (factor)
run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  value <- data$value
  group <- droplevels(data$group)
  lev <- levels(group)
  k <- nlevels(group); N <- length(value)
  if (length(unique(value)) < 2) aui_stop("All values are identical, so the groups cannot be ranked.")

  kw <- stats::kruskal.test(value, group)
  h <- unname(kw$statistic); df <- unname(kw$parameter); p <- kw$p.value
  eps2 <- h / ((N^2 - 1) / (N + 1))
  eta2h <- max(0, (h - k + 1) / (N - k))
  has_ties <- any(duplicated(value))
  aui_table("Kruskal–Wallis test", data.frame(
    H = h, df = df, p = p, stars = aui_stars(p), eps2 = eps2, eta2 = eta2h, stringsAsFactors = FALSE),
    formats = c(df = "int", p = "p", stars = "stars"),
    labels = c(H = "Kruskal–Wallis H", p = "P value", stars = "Summary", eps2 = "Epsilon²", eta2 = "Eta² (H)"),
    footnotes = c(paste0("Approximate P value from the chi-square distribution", if (has_ties) "; H is corrected for ties" else "", "."),
                  "Epsilon² = H / (N − 1) and eta²(H) = (H − k + 1) / (N − k) are rank-based effect sizes (0 = no effect, 1 = complete separation).",
                  if (min(table(group)) < 5) "Some groups have fewer than five values; the chi-square approximation may be inaccurate."))

  r <- rank(value)
  if (aui_flag(options, "descriptives", TRUE)) {
    q <- function(pr) as.numeric(tapply(value, group, stats::quantile, probs = pr, names = FALSE))
    aui_table("Medians and mean ranks", data.frame(
      Group = lev, n = as.numeric(table(group)), median = q(0.5), q1 = q(0.25), q3 = q(0.75),
      mean_rank = as.numeric(tapply(r, group, mean)), rank_sum = as.numeric(tapply(r, group, sum)),
      stringsAsFactors = FALSE),
      formats = c(n = "int"),
      labels = c(median = "Median", q1 = "25th percentile", q3 = "75th percentile",
                 mean_rank = "Mean rank", rank_sum = "Sum of ranks"))
  }

  compare <- aui_opt(options, "compare", "pairs")
  n_sig <- NA
  if (compare != "none") {
    ctrl <- if (compare == "control") st_control_index(lev, aui_opt(options, "control", "")) else 1L
    cmp <- st_dunn(value, group, compare, ctrl, aui_opt(options, "adjust", "bonferroni"))
    aui_table(if (compare == "control") paste0("Dunn's multiple comparisons: each group vs ", lev[ctrl])
              else "Dunn's multiple comparisons: all pairs",
              cmp, formats = c(p_raw = "p", p = "p", stars = "stars"),
              labels = c(diff = "Mean rank diff.", p_raw = "P (unadjusted)", p = "Adjusted P", stars = "Summary"),
              footnotes = attr(cmp, "label"))
    n_sig <- sum(cmp$p < alpha); n_cmp <- nrow(cmp)
  }

  if (aui_flag(options, "show_plot", TRUE)) {
    d <- data.frame(value = value, group = group)
    pl <- ggplot2::ggplot(d, ggplot2::aes(group, value, colour = group)) +
      ggplot2::geom_boxplot(outlier.shape = NA, width = 0.55, fill = NA) +
      ggplot2::geom_point(position = ggplot2::position_jitter(width = 0.12, height = 0, seed = 1), alpha = 0.6, size = 1.8) +
      ggplot2::scale_colour_manual(values = st_colors(k), guide = "none") +
      ggplot2::labs(x = NULL, y = st_name(names, "value")) + aui_theme() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = if (k > 5 || max(nchar(lev)) > 12) 30 else 0,
                                                         hjust = if (k > 5 || max(nchar(lev)) > 12) 1 else 0.5))
    aui_plot("Box plot (median and quartiles)", pl, width = max(4.2, min(9, 1.6 + 0.9 * k)), height = 3.9)
  }

  txt <- sprintf("Kruskal–Wallis test: the distributions of the %d groups %s (H = %s, df = %d, %s).",
                 k, if (p < alpha) "differ significantly" else "do not differ significantly",
                 formatC(h, format = "f", digits = 2), df, st_p_text(p))
  if (!is.na(n_sig)) txt <- paste0(txt, sprintf(" Dunn's test: %d of %d comparison%s significant at α = %s.",
                                                n_sig, n_cmp, if (n_cmp == 1) " is" else "s are", format(alpha)))
  aui_summary(txt)
  aui_graph("Create a column graph of these data", "column_scatter", st_layout_options(options))
}
