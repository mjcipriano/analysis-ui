# Nested t test / nested one-way ANOVA.
# data: y, group (treatment), nested (subgroup label), sub (unique subgroup id, "group / label")
run <- function(data, options) {
  aui_need("ggplot2", "lme4", "lmerTest", "emmeans")
  names <- st_names(options)
  alpha <- st_alpha(options)
  data$group <- droplevels(data$group); data$sub <- droplevels(data$sub)
  lev <- levels(data$group); k <- length(lev)
  gname <- st_name(names, "group"); sname <- st_name(names, "nested"); ylab <- st_name(names, "y")

  q <- st_quiet(st_try(st_with_sum_contrasts(lmerTest::lmer(y ~ group + (1 | sub), data = data, REML = TRUE)),
                       "The nested model could not be fitted."))
  fit <- q$value
  singular <- lme4::isSingular(fit)
  a <- as.data.frame(st_quiet(stats::anova(fit, type = 3, ddf = "Satterthwaite"))$value)
  f <- a$`F value`[1]; p <- a$`Pr(>F)`[1]; df1 <- a$NumDF[1]; df2 <- a$DenDF[1]
  nm <- list(group = gname)

  title <- if (k == 2) "Nested t test" else "Nested one-way ANOVA"
  method_note <- paste0("Mixed-effects model fitted by REML: ", gname, " is a fixed effect and ", sname,
                        " (nested within ", gname, ") is a random effect; Satterthwaite degrees of freedom.")
  if (k == 2) {
    ct <- st_quiet(as.data.frame(summary(emmeans::contrast(st_emmeans(fit, "group"), "pairwise"),
                                         infer = c(TRUE, TRUE), level = 1 - alpha)))$value
    aui_table(title, data.frame(
      Comparison = paste0(lev[1], " \u2212 ", lev[2]), diff = ct$estimate, se = ct$SE, lower = ct$lower.CL, upper = ct$upper.CL,
      t = ct$t.ratio, df = ct$df, p = ct$p.value, stars = aui_stars(ct$p.value), stringsAsFactors = FALSE),
      formats = c(p = "p", stars = "stars"),
      labels = c(diff = "Difference between means", se = "SE", lower = paste(st_pct(alpha), "CI lower"),
                 upper = paste(st_pct(alpha), "CI upper"), p = "P value", stars = "Summary"),
      footnotes = c(method_note, sprintf("Equivalent F test: %s.", st_fmt_f(f, df1, df2)), st_dropped_note(options)))
  } else {
    aui_table(title, data.frame(
      Source = paste0(gname, " (fixed effect)"), F = f, df1 = df1, df2 = df2, p = p, stars = aui_stars(p), stringsAsFactors = FALSE),
      formats = c(p = "p", stars = "stars", df1 = "int"),
      labels = c(df1 = "df numerator", df2 = "df denominator", p = "P value", stars = "Summary"),
      footnotes = c(method_note, st_dropped_note(options)))
  }

  vc <- as.data.frame(lme4::VarCorr(fit))
  v_sub <- vc$vcov[vc$grp == "sub"]; v_res <- vc$vcov[vc$grp == "Residual"]
  lrt <- tryCatch({
    ll0 <- as.numeric(stats::logLik(stats::lm(y ~ group, data = data), REML = TRUE))
    stat <- max(0, 2 * (as.numeric(stats::logLik(fit)) - ll0))
    c(stat, 0.5 * stats::pchisq(stat, 1, lower.tail = FALSE))
  }, error = function(e) c(NA, NA))
  aui_table("Variance components", data.frame(
    Component = c(paste0("Between subgroups (", sname, " within ", gname, ")"), "Within subgroups (residual)"),
    Variance = c(v_sub, v_res), SD = sqrt(c(v_sub, v_res)), share = c(v_sub, v_res) / (v_sub + v_res),
    stringsAsFactors = FALSE),
    formats = c(share = "pct"), labels = c(share = "% of random variation"),
    footnotes = c(sprintf("Do the subgroups differ? Likelihood-ratio test of the subgroup variance: \u03c7\u00b2 = %s, %s (P halved because a variance cannot be negative).",
                          formatC(lrt[1], format = "f", digits = 3), st_p_text(lrt[2])),
                  if (singular) "The variation between subgroups was estimated as zero: subgroups within a group do not differ more than expected from the scatter of individual values. The test then treats all values as independent."))

  # ---- subgroup means
  sm <- stats::aggregate(y ~ sub + nested + group, data = data, FUN = mean)
  sn <- stats::aggregate(y ~ sub + nested + group, data = data, FUN = length)
  sm$n <- sn$y
  sm <- sm[order(sm$group, sm$sub), ]
  n_sub <- as.numeric(tapply(sm$y, sm$group, length))
  gm <- data.frame(Group = lev, subgroups = n_sub, n = as.numeric(table(data$group)),
                   mean = as.numeric(tapply(sm$y, sm$group, mean)), sd = as.numeric(tapply(sm$y, sm$group, stats::sd)),
                   stringsAsFactors = FALSE)
  gm$sem <- gm$sd / sqrt(gm$subgroups)
  names(gm)[1] <- gname
  aui_table("Group means (means of the subgroup means)", gm, formats = c(subgroups = "int", n = "int"),
            labels = c(subgroups = "Subgroups", n = "Values", mean = "Mean of subgroup means", sd = "SD of subgroup means", sem = "SEM"))
  if (aui_flag(options, "descriptives", TRUE)) {
    d <- data.frame(g = as.character(sm$group), s = as.character(sm$nested), n = sm$n, mean = sm$y, stringsAsFactors = FALSE)
    names(d)[1:2] <- make.unique(c(gname, sname))
    aui_table("Subgroup means", d, formats = c(n = "int"), labels = c(mean = "Mean"))
  }

  balanced <- length(unique(sm$n)) == 1
  if (all(n_sub >= 2)) {
    sa <- stats::anova(stats::lm(y ~ group, data = sm))
    aui_table("For comparison: ordinary analysis of the subgroup means", data.frame(
      Test = if (k == 2) "Unpaired t test on subgroup means (as F)" else "One-way ANOVA on subgroup means",
      F = sa$`F value`[1], df1 = sa$Df[1], df2 = sa$Df[2], p = sa$`Pr(>F)`[1], stringsAsFactors = FALSE),
      formats = c(p = "p", df1 = "int", df2 = "int"), labels = c(df1 = "df numerator", df2 = "df denominator", p = "P value"),
      footnotes = if (balanced && !singular) "Every subgroup has the same number of values, so the nested analysis and the analysis of subgroup means are equivalent (same F and P)."
                  else if (balanced) "With equal subgroup sizes the two analyses agree whenever the between-subgroup variance is estimated as positive; here it is zero, so the nested test has more degrees of freedom."
                  else "Subgroup sizes differ, so the nested analysis (which weights subgroups by their precision) and the analysis of subgroup means can give somewhat different answers.")
  }

  compare <- aui_opt(options, "compare", "pairs")
  if (k > 2 && compare != "none") {
    st_emit_emm_comparisons(fit, "group", NULL, nm, type = compare, control = aui_opt(options, "control", ""),
                            method = aui_opt(options, "posthoc_method", "tukey"), alpha = alpha,
                            title = paste0("Multiple comparisons: ", gname), extra_notes = "Based on the nested mixed model (Satterthwaite df).")
  }

  if (aui_flag(options, "show_plot", TRUE)) {
    sm$x <- as.numeric(sm$group) + (stats::ave(seq_along(sm$sub), sm$group, FUN = function(i) seq_along(i) - (length(i) + 1) / 2)) *
      (0.7 / max(n_sub))
    data$x <- sm$x[match(data$sub, sm$sub)]
    pl <- ggplot2::ggplot(data, ggplot2::aes(x, y, colour = group)) +
      ggplot2::geom_point(alpha = 0.5, size = 1.5) +
      ggplot2::geom_point(data = sm, ggplot2::aes(x, y, colour = group), shape = 95, size = 9) +
      ggplot2::scale_x_continuous(breaks = seq_len(k), labels = lev, limits = c(0.5, k + 0.5)) +
      ggplot2::scale_colour_manual(values = st_colors(k), guide = "none") +
      ggplot2::labs(x = gname, y = ylab) + aui_theme()
    aui_plot(paste0("Values by subgroup (bars: subgroup means; one column per ", sname, ")"), pl,
             width = max(4.6, min(9, 1.8 + 1.3 * k)), height = 3.9)
  }

  aui_summary(sprintf("%s: the %s means %s (%s, %s). Variation between subgroups accounts for %s%% of the random variation; %d subgroups, %d values.",
                      title, gname, if (p < alpha) "differ significantly" else "do not differ significantly",
                      st_fmt_f(f, df1, df2), st_p_text(p), formatC(100 * v_sub / (v_sub + v_res), format = "f", digits = 1),
                      nrow(sm), nrow(data)))
}
