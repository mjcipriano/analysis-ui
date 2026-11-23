# Reliability analysis: Cronbach's alpha, McDonald's omega, item statistics.
# data: x1..xk (items), complete cases

.cronbach <- function(cv) {
  k <- ncol(cv)
  k / (k - 1) * (1 - sum(diag(cv)) / sum(cv))
}

# McDonald's omega (total) from a one-factor maximum-likelihood model.
.omega_total <- function(x) {
  k <- ncol(x)
  if (k < 3) return(NA_real_)
  fa <- tryCatch(st_quiet(stats::factanal(covmat = stats::cor(x), factors = 1, n.obs = nrow(x)))$value, error = function(e) NULL)
  if (is.null(fa)) return(NA_real_)
  sds <- apply(x, 2, stats::sd)
  l <- as.numeric(fa$loadings[, 1]) * sds
  psi <- fa$uniquenesses * sds^2
  sum(l)^2 / (sum(l)^2 + sum(psi))
}

run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  items <- aui_vars(options, "_items")
  x <- as.matrix(data[, items, drop = FALSE])
  lab <- vapply(items, function(i) st_name(names, i), "")
  n <- nrow(x); k <- ncol(x)
  r <- stats::cor(x)

  # direction of every item relative to the scale (sign of first principal component)
  pc <- eigen(r, symmetric = TRUE)$vectors[, 1]
  if (sum(pc) < 0) pc <- -pc
  negative <- pc < -1e-8
  reversed <- rep(FALSE, k)
  if (aui_flag(options, "reverse") && any(negative)) {
    for (j in which(negative)) x[, j] <- min(x[, j]) + max(x[, j]) - x[, j]
    reversed <- negative
    r <- stats::cor(x)
  }
  cv <- stats::cov(x)
  total <- rowSums(x)
  if (stats::var(total) <= 0) aui_stop("The total score is the same for every row, so reliability cannot be estimated.")
  a <- .cronbach(cv)
  rbar <- mean(r[lower.tri(r)])
  a_std <- k * rbar / (1 + (k - 1) * rbar)
  df1 <- n - 1; df2 <- (n - 1) * (k - 1)
  lo <- 1 - (1 - a) * stats::qf(1 - alpha / 2, df1, df2)
  hi <- 1 - (1 - a) * stats::qf(alpha / 2, df1, df2)
  quality <- if (a >= 0.9) "excellent" else if (a >= 0.8) "good" else if (a >= 0.7) "acceptable" else if (a >= 0.6) "questionable" else if (a >= 0.5) "poor" else "unacceptable"

  rows <- data.frame(Measure = c("Cronbach's alpha", "Standardized alpha", "Average inter-item correlation"),
                     value = c(a, a_std, rbar), lower = c(lo, NA, NA), upper = c(hi, NA, NA), stringsAsFactors = FALSE)
  if (aui_flag(options, "omega", TRUE)) {
    om <- .omega_total(x)
    rows <- rbind(rows, data.frame(Measure = "McDonald's omega (total)", value = om, lower = NA, upper = NA, stringsAsFactors = FALSE))
  }
  rows <- rbind(rows, data.frame(Measure = c("Number of items", "Complete rows", "Scale mean (sum of items)", "Scale SD"),
                                 value = c(k, n, mean(total), stats::sd(total)), lower = NA, upper = NA, stringsAsFactors = FALSE))
  aui_table("Scale reliability", rows,
            labels = c(value = "Value", lower = paste(st_pct(alpha), "CI lower"), upper = paste(st_pct(alpha), "CI upper")),
            footnotes = c("Cronbach's alpha from the item covariances; confidence interval by Feldt's method (F distribution). Standardized alpha is based on the correlations (items given equal variance).",
                          if (aui_flag(options, "omega", TRUE)) (if (k < 3) "McDonald's omega needs at least three items." else "McDonald's omega (total) from a one-factor maximum-likelihood model; unlike alpha it does not assume that all items measure the construct equally well."),
                          if (any(reversed)) paste0("Reverse-scored items (value \u2192 minimum + maximum \u2212 value): ", paste(lab[reversed], collapse = ", "), "."),
                          st_dropped_note(options)))
  if (any(negative) && !any(reversed)) {
    aui_warn(paste0("Some items are negatively related to the rest of the scale (", paste(lab[negative], collapse = ", "),
                    "). If they are worded in the opposite direction, reverse-score them (option \u201cReverse-score negatively related items\u201d); otherwise alpha is misleadingly low."))
  }

  if (aui_flag(options, "item_stats", TRUE)) {
    rest <- vapply(seq_len(k), function(j) stats::cor(x[, j], total - x[, j]), numeric(1))
    dropped <- vapply(seq_len(k), function(j) if (k > 2) .cronbach(cv[-j, -j, drop = FALSE]) else NA_real_, numeric(1))
    it <- data.frame(Item = lab, mean = colMeans(x), sd = apply(x, 2, stats::sd), rest = rest, dropped = dropped,
                     note = ifelse(reversed, "reversed", ifelse(!is.na(dropped) & dropped > a + 1e-9, "alpha rises if dropped", "")),
                     stringsAsFactors = FALSE)
    aui_table("Item statistics", it,
              labels = c(mean = "Mean", sd = "SD", rest = "Item\u2013rest correlation", dropped = "Alpha if item dropped", note = "Note"),
              footnotes = c("Item\u2013rest correlation: correlation of the item with the sum of the other items (corrected item\u2013total correlation); values below about 0.3 suggest the item fits the scale poorly.",
                            if (k < 3) "Alpha if item dropped needs at least three items."))
  }

  if (aui_flag(options, "show_plot")) {
    d <- expand.grid(a = factor(lab, levels = lab), b = factor(lab, levels = rev(lab)))
    d$r <- as.vector(r)
    pl <- ggplot2::ggplot(d, ggplot2::aes(a, b, fill = r)) + ggplot2::geom_tile(colour = "white") +
      ggplot2::geom_text(ggplot2::aes(label = formatC(r, format = "f", digits = 2)), size = 3) +
      ggplot2::scale_fill_gradient2(low = AUI_COLORS[2], mid = "white", high = AUI_COLORS[1], limits = c(-1, 1), name = "r") +
      ggplot2::labs(x = NULL, y = NULL) + aui_theme() +
      ggplot2::theme(legend.position = "right", legend.title = ggplot2::element_text(),
                     axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))
    aui_plot("Inter-item correlations", pl, width = min(9, 2.6 + 0.62 * k), height = min(8, 2 + 0.55 * k))
  }

  aui_summary(sprintf("Cronbach's alpha = %s (%s CI %s to %s) for %d items and %d complete rows: %s internal consistency by the usual rule of thumb.%s",
                      formatC(a, format = "f", digits = 3), st_pct(alpha), formatC(lo, format = "f", digits = 3),
                      formatC(hi, format = "f", digits = 3), k, n, quality,
                      if (a < 0) " A negative alpha means the items do not measure a common construct in the same direction." else ""))
}
