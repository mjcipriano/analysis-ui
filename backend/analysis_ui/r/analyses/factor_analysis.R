# Exploratory factor analysis.
# data: x1..xp (variables), complete cases

# Kaiser-Meyer-Olkin measure of sampling adequacy (overall and per variable).
.kmo <- function(r) {
  inv <- solve(r)
  d <- sqrt(diag(inv))
  pr <- -inv / outer(d, d)          # partial correlations
  diag(pr) <- 0
  r0 <- r; diag(r0) <- 0
  list(overall = sum(r0^2) / (sum(r0^2) + sum(pr^2)),
       items = colSums(r0^2) / (colSums(r0^2) + colSums(pr^2)))
}

.bartlett_sphericity <- function(r, n) {
  p <- ncol(r)
  chi2 <- -(n - 1 - (2 * p + 5) / 6) * as.numeric(determinant(r, logarithm = TRUE)$modulus)
  df <- p * (p - 1) / 2
  list(chi2 = chi2, df = df, p = stats::pchisq(chi2, df, lower.tail = FALSE))
}

# Horn's parallel analysis: 95th percentile of the eigenvalues of correlation
# matrices of random normal data with the same n and p.
.parallel <- function(n, p, reps = 100) {
  set.seed(20240229)
  ev <- vapply(seq_len(reps), function(i) {
    eigen(stats::cor(matrix(stats::rnorm(n * p), n, p)), symmetric = TRUE, only.values = TRUE)$values
  }, numeric(p))
  apply(ev, 1, stats::quantile, probs = 0.95, names = FALSE)
}

.max_factors_ml <- function(p) {
  k <- 0
  while (((p - (k + 1))^2 - (p + k + 1)) / 2 >= 0) k <- k + 1
  k
}

.rotate <- function(l, rotation) {
  k <- ncol(l)
  phi <- diag(k)
  note <- NULL
  if (k > 1 && rotation != "none") {
    if (rotation == "oblimin") {
      if (requireNamespace("GPArotation", quietly = TRUE)) {
        g <- GPArotation::oblimin(l)
        l <- unclass(g$loadings); phi <- g$Phi
      } else {
        rotation <- "promax"
        note <- "Oblimin rotation needs the R package GPArotation, which is not installed; promax (another oblique rotation, which usually gives very similar results) was used instead."
      }
    }
    if (rotation == "varimax") {
      l <- unclass(stats::varimax(l)$loadings)
    } else if (rotation == "promax") {
      pm <- stats::promax(l, m = 4)
      l <- unclass(pm$loadings)
      phi <- solve(crossprod(pm$rotmat))
    }
  }
  # order factors by size and make the larger loadings positive
  ss <- colSums(l^2)
  o <- order(-ss)
  l <- l[, o, drop = FALSE]; phi <- phi[o, o, drop = FALSE]
  sgn <- ifelse(colSums(l) < 0, -1, 1)
  l <- sweep(l, 2, sgn, "*"); phi <- phi * outer(sgn, sgn)
  list(loadings = l, phi = phi, rotation = if (k > 1) rotation else "none", note = note)
}

run <- function(data, options) {
  aui_need("ggplot2")
  names <- st_names(options)
  alpha <- st_alpha(options)
  items <- aui_vars(options, "_items")
  x <- as.matrix(data[, items, drop = FALSE])
  lab <- vapply(items, function(i) st_name(names, i), "")
  n <- nrow(x); p <- ncol(x)
  r <- stats::cor(x)
  off <- r; diag(off) <- 0
  if (any(abs(off) > 1 - 1e-8)) {
    w <- which(abs(off) > 1 - 1e-8, arr.ind = TRUE)[1, ]
    aui_stop("\u201c", lab[w[1]], "\u201d and \u201c", lab[w[2]], "\u201d are perfectly correlated, so they carry the same information. Remove one of them.")
  }
  ev <- eigen(r, symmetric = TRUE, only.values = TRUE)$values
  if (min(ev) < 1e-10) aui_stop("The correlation matrix is singular: at least one variable is an exact combination of the others (for example a total score together with its parts). Remove the redundant variable.")

  extraction <- aui_opt(options, "extraction", "ml")
  n_method <- aui_opt(options, "n_method", "parallel")
  par <- .parallel(n, p)
  k_max <- if (extraction == "ml") .max_factors_ml(p) else p - 1
  if (k_max < 1) aui_stop("Too few variables for a factor analysis with this extraction method. Select more variables.")
  k_note <- NULL
  if (n_method == "fixed") {
    k <- as.integer(aui_num(options, "n_factors", 2))
    if (!is.finite(k) || k < 1) k <- 1L
    if (k > k_max) aui_stop("With ", p, " variables at most ", k_max, " factor", if (k_max == 1) "" else "s", " can be extracted by ",
                            if (extraction == "ml") "maximum likelihood" else "this method", ". Lower the number of factors or add variables.")
    k_note <- sprintf("Number of factors fixed at %d.", k)
  } else {
    k <- if (n_method == "kaiser") sum(ev > 1) else { above <- ev > par; if (all(above)) p else which(!above)[1] - 1 }
    rule <- if (n_method == "kaiser") "the eigenvalue > 1 rule" else "parallel analysis (eigenvalues larger than the 95th percentile of eigenvalues from random data)"
    k_note <- sprintf("%d factor%s suggested by %s.", k, if (k == 1) "" else "s", rule)
    if (k < 1) { k <- 1; k_note <- paste0("No factor passed ", rule, " (the variables are only weakly correlated); one factor was extracted anyway.") }
    if (k > k_max) { k_note <- paste0(k_note, sprintf(" Reduced to %d, the most that can be estimated from %d variables.", k_max, p)); k <- k_max }
  }

  # ---- extraction (unrotated)
  fit_stat <- NULL
  if (extraction == "ml") {
    fa <- tryCatch(st_quiet(stats::factanal(covmat = r, factors = k, n.obs = n, rotation = "none"))$value, error = function(e) e)
    if (inherits(fa, "error")) {
      aui_stop("The maximum-likelihood factor model with ", k, " factor", if (k == 1) "" else "s", " could not be estimated (the optimiser did not converge). ",
               "Try fewer factors or the minimum-residual extraction method.")
    }
    l <- unclass(fa$loadings); uniq <- fa$uniquenesses
    if (!is.null(fa$STATISTIC) && fa$dof > 0) fit_stat <- list(chi2 = unname(fa$STATISTIC), df = fa$dof, p = unname(fa$PVAL))
  } else {
    aui_need("psych")
    fa <- tryCatch(st_quiet(psych::fa(r = r, nfactors = k, n.obs = n, rotate = "none", fm = "minres", warnings = FALSE))$value, error = function(e) e)
    if (inherits(fa, "error")) aui_stop("The minimum-residual factor model with ", k, " factors could not be estimated. Try fewer factors.")
    l <- unclass(fa$loadings); uniq <- fa$uniquenesses
  }
  l <- matrix(l, nrow = p, ncol = k)
  rot <- .rotate(l, aui_opt(options, "rotation", "varimax"))
  l <- rot$loadings; phi <- rot$phi
  oblique <- rot$rotation %in% c("promax", "oblimin")
  if (!is.null(rot$note)) aui_warn(rot$note)
  fnames <- paste("Factor", seq_len(k))
  heywood <- any(uniq <= 0.0051)

  # ---- loadings
  hide <- aui_num(options, "hide_below", 0.3)
  if (!is.finite(hide)) hide <- 0
  shown <- l
  shown[abs(shown) < hide] <- NA
  tab <- data.frame(Variable = lab, stringsAsFactors = FALSE)
  for (j in seq_len(k)) tab[[paste0("F", j)]] <- shown[, j]
  communality <- 1 - uniq
  tab$communality <- communality; tab$uniqueness <- uniq
  if (aui_flag(options, "sort_loadings", TRUE)) {
    primary <- apply(abs(l), 1, which.max)
    tab <- tab[order(primary, -apply(abs(l), 1, max)), , drop = FALSE]
  }
  rot_name <- switch(rot$rotation, varimax = "varimax rotation (orthogonal)", promax = "promax rotation (oblique)",
                     oblimin = "oblimin rotation (oblique)", "no rotation")
  aui_table(if (oblique) "Factor loadings (pattern matrix)" else "Factor loadings", tab,
            labels = c(stats::setNames(fnames, paste0("F", seq_len(k))), communality = "Communality", uniqueness = "Uniqueness"),
            footnotes = c(paste0(if (extraction == "ml") "Maximum-likelihood" else "Minimum-residual (minres)", " extraction from the correlation matrix, ", rot_name, ". ", k_note),
                          if (hide > 0) sprintf("Loadings smaller than %s in absolute value are hidden.", format(hide)),
                          "Communality is the share of a variable's variance explained by the factors; uniqueness = 1 \u2212 communality.",
                          if (heywood) "At least one uniqueness is at its lower bound (a Heywood case): the solution may be unstable; try fewer factors.",
                          st_dropped_note(options)))

  ss <- colSums(l^2)
  aui_table("Variance explained", data.frame(
    Factor = fnames, ss = ss, prop = ss / p, cum = cumsum(ss / p), stringsAsFactors = FALSE),
    formats = c(prop = "pct", cum = "pct"),
    labels = c(ss = "Sum of squared loadings", prop = "% of variance", cum = "Cumulative %"),
    footnotes = if (oblique) "With an oblique rotation the factors are correlated, so their contributions overlap and these shares are approximate.")
  if (oblique && k > 1) {
    pt <- data.frame(Factor = fnames, stringsAsFactors = FALSE)
    for (j in seq_len(k)) pt[[paste0("F", j)]] <- phi[, j]
    aui_table("Correlations between factors", pt, labels = stats::setNames(fnames, paste0("F", seq_len(k))))
  }
  if (!is.null(fit_stat)) {
    aui_table("Test that the factors are sufficient", data.frame(chi2 = fit_stat$chi2, df = fit_stat$df, p = fit_stat$p, stringsAsFactors = FALSE),
              formats = c(df = "int", p = "p"), labels = c(chi2 = "\u03c7\u00b2", p = "P value"),
              footnotes = sprintf("Likelihood-ratio test of the %d-factor model against an unrestricted correlation matrix. A small P value suggests that more factors are needed (with large samples the test rejects even for trivial misfit).", k))
  }

  # ---- sampling adequacy
  if (aui_flag(options, "adequacy", TRUE)) {
    aui_heading("Is the data suitable for factor analysis?")
    kmo <- .kmo(r); bt <- .bartlett_sphericity(r, n)
    verdict <- if (kmo$overall >= 0.9) "marvellous" else if (kmo$overall >= 0.8) "meritorious" else if (kmo$overall >= 0.7) "middling" else if (kmo$overall >= 0.6) "mediocre" else if (kmo$overall >= 0.5) "miserable" else "unacceptable"
    aui_table("Sampling adequacy and sphericity", data.frame(
      Test = c("Kaiser\u2013Meyer\u2013Olkin (KMO) measure", "Bartlett's test of sphericity"),
      stat_name = c("KMO", "\u03c7\u00b2"), statistic = c(kmo$overall, bt$chi2), df = c(NA, bt$df), p = c(NA, bt$p), stringsAsFactors = FALSE),
      formats = c(df = "int", p = "p"), labels = c(stat_name = "Statistic", statistic = "Value", p = "P value"),
      footnotes = c(sprintf("KMO = %s (\u201c%s\u201d on Kaiser's scale; at least 0.6 is usually wanted).", formatC(kmo$overall, format = "f", digits = 3), verdict),
                    "Bartlett's test: a small P value means the variables are correlated enough for factor analysis to be meaningful."))
    aui_table("KMO for each variable", data.frame(Variable = lab, msa = kmo$items, stringsAsFactors = FALSE),
              labels = c(msa = "Measure of sampling adequacy"),
              footnotes = "Variables below about 0.5 share little with the others and are candidates for removal.")
  }

  et <- data.frame(Component = seq_len(p), eigen = ev, prop = ev / p, random = par, stringsAsFactors = FALSE)
  aui_table("Eigenvalues of the correlation matrix", et, formats = c(Component = "int", prop = "pct"),
            labels = c(eigen = "Eigenvalue", prop = "% of variance", random = "Parallel analysis threshold"),
            footnotes = "Parallel analysis threshold: 95th percentile of the eigenvalues of 100 random data sets with the same number of rows and variables.")
  if (aui_flag(options, "show_plot", TRUE)) {
    d <- rbind(data.frame(i = seq_len(p), v = ev, what = "Data"), data.frame(i = seq_len(p), v = par, what = "Random data (95th percentile)"))
    pl <- ggplot2::ggplot(d, ggplot2::aes(i, v, colour = what, linetype = what)) +
      ggplot2::geom_hline(yintercept = 1, colour = "#C8CED3") +
      ggplot2::geom_line(linewidth = 0.8) + ggplot2::geom_point(size = 2.2) +
      ggplot2::scale_colour_manual(values = c("Data" = AUI_COLORS[1], "Random data (95th percentile)" = AUI_COLORS[2])) +
      ggplot2::scale_linetype_manual(values = c("Data" = "solid", "Random data (95th percentile)" = "dashed")) +
      ggplot2::scale_x_continuous(breaks = seq_len(p)) +
      ggplot2::labs(x = "Component number", y = "Eigenvalue") + aui_theme()
    aui_plot("Scree plot with parallel analysis", pl, width = 5.6, height = 4)
  }

  aui_summary(sprintf("%d factor%s extracted from %d variables (%d rows) by %s with %s, together explaining %s%% of the variance. %s",
                      k, if (k == 1) " was" else "s were", p, n, if (extraction == "ml") "maximum likelihood" else "minimum residuals", rot_name,
                      formatC(100 * sum(ss) / p, format = "f", digits = 1), k_note))
}
