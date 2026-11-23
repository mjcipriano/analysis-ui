# Kaplan–Meier curves with censor marks, CI bands, median lines, log-rank P
# and an optional numbers-at-risk table aligned with the time axis.

# step-function ribbon vertices (ggplot has no stepped ribbon)
step_band <- function(d) {
  do.call(rbind, lapply(split(d, d$key, drop = TRUE), function(g) {
    g <- g[order(g$time), , drop = FALSE]
    n <- nrow(g)
    if (n < 2) return(NULL)
    i <- c(rep(seq_len(n - 1), each = 2), n)
    data.frame(time = c(g$time[1], rep(g$time[-1], each = 2)), lo = g$lo[i], hi = g$hi[i], key = g$key[1])
  }))
}

render_survival <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed
  curve <- ctx$frames$curve
  if (!has_rows(curve)) aui_stop("There are no complete time/event rows to plot yet.")
  curve <- keyed(curve)
  keys <- levels(curve$group); k <- length(keys)
  percent <- opt_flag(comp, "percent", TRUE); cumulative <- opt_flag(comp, "cumulative")
  top <- if (percent) 100 else 1
  risk <- ctx$frames$risk
  show_risk <- opt_flag(o, "risk_table") && has_rows(risk)
  risk_times <- num_vec(comp$risk_times)
  ax_x <- axis_spec(ctx, "x", finite_range(0, curve$time, if (show_risk) risk_times), zero = TRUE, flush = TRUE,
                    mult = c(0, 0.04))
  ax_y <- axis_spec(ctx, "y", c(0, top), fixed = c(0, top * 1.04))
  legend <- legend_position(ctx, if (k > 1) "right" else "none")
  P <- xy_panel(ctx, ax_y, legend, keys, key_w = 2)
  line_pt <- opt_num(o, "line_width", 1.8)
  curve <- curve[order(curve$group, curve$time, if (cumulative) curve$surv else -curve$surv), , drop = FALSE]
  layers <- list()
  add <- function(...) layers <<- c(layers, list(...))

  if (opt_flag(o, "show_ci")) {
    band <- step_band(curve[is.finite(curve$lo) & is.finite(curve$hi), , drop = FALSE])
    if (has_rows(band)) add(geom_ribbon(aes(x = .data$time, ymin = .data$lo, ymax = .data$hi, fill = .data$key,
                                            group = .data$key), data = band, alpha = 0.16, colour = NA, show.legend = FALSE))
  }
  if (opt_flag(o, "median_line")) {
    med <- records(comp$medians, c("group", "median"))
    med$median <- suppressWarnings(as.numeric(med$median))
    med <- med[is.finite(med$median) & med$group %in% keys, , drop = FALSE]
    if (nrow(med)) {
      half <- top / 2
      add(geom_segment(aes(x = ax_x$lim[1], xend = max(med$median), y = half, yend = half), data = med[1, , drop = FALSE],
                       colour = ctx$muted, linewidth = lw(0.8), linetype = "33"),
          geom_segment(aes(x = .data$median, xend = .data$median, y = half, yend = ax_y$lim[1], colour = .data$group),
                       data = med, linewidth = lw(0.8), linetype = "33", show.legend = FALSE))
    }
  }
  add(geom_step(aes(x = .data$time, y = .data$surv, colour = .data$key, group = .data$key), data = curve,
                direction = "hv", linewidth = lw(line_pt), show.legend = TRUE))
  cens <- ctx$frames$censor
  if (opt_flag(o, "show_censors", TRUE) && has_rows(cens)) {
    cens <- keyed(cens)
    cens <- cens[is.finite(cens$time) & is.finite(cens$surv), , drop = FALSE]
    h <- max(3, line_pt * 2.2) / P$h * diff(ax_y$lim)            # tick half-height in data units
    add(geom_segment(aes(x = .data$time, xend = .data$time, y = .data$surv - h, yend = .data$surv + h,
                         colour = .data$key), data = cens, linewidth = lw(max(0.8, line_pt * 0.6)), show.legend = FALSE))
  }
  p_label <- opt_chr(comp, "p_label", "")
  if (opt_flag(o, "show_p", TRUE) && nzchar(p_label)) {
    at_top <- cumulative
    add(annotate("text", x = ax_x$lim[1] + diff(ax_x$lim) * 0.035,
                 y = if (at_top) ax_y$lim[2] - diff(ax_y$lim) * 0.05 else ax_y$lim[1] + diff(ax_y$lim) * 0.05,
                 label = p_label, hjust = 0, vjust = if (at_top) 1 else 0, size = ts(ctx$f$tick),
                 colour = ctx$th$text, family = ctx$f$family))
  }
  step <- opt_num(o, "y_tick_step")
  ybreaks <- if (is.na(step) && ax_y$scale == "linear" && !any(ax_y$user)) seq(0, top, length.out = 5) else NULL
  xbreaks <- if (show_risk && length(risk_times) && is.na(opt_num(o, "x_tick_step")) && ax_x$scale == "linear")
    risk_times[risk_times >= ax_x$lim[1] & risk_times <= ax_x$lim[2]] else NULL
  angle <- tick_angle(opt_chr(o, "x_tick_angle", "auto"))
  x_scale <- function() if (is.null(xbreaks)) axis_scale(ctx, ax_x) else axis_scale(ctx, ax_x, breaks = xbreaks, labels = fmt_num(xbreaks))
  p <- ggplot() + ref_lines(ctx, ax_x, ax_y) + layers + x_scale() +
    (if (is.null(ybreaks)) axis_scale(ctx, ax_y) else axis_scale(ctx, ax_y, breaks = ybreaks, labels = fmt_num(ybreaks))) +
    colour_scales(ctx, keys) +
    coord_cartesian(xlim = ax_x$lim, ylim = ax_y$lim, expand = FALSE) +
    gg_labs(ctx) + gg_theme(ctx, legend = legend, x_angle = angle, key_w = 2)
  if (!show_risk) return(p)

  # ---- numbers at risk
  risk <- keyed(risk)
  risk <- risk[is.finite(risk$time) & risk$time >= ax_x$lim[1] & risk$time <= ax_x$lim[2] & risk$key %in% keys, , drop = FALSE]
  risk$row <- k + 1 - as.integer(risk$group)                     # first group at the top
  f <- ctx$f
  row_pt <- f$tick * 1.35
  cols <- group_cols(ctx, keys)
  table_theme <- gg_theme(ctx, legend = "none") + theme(
    axis.line = element_blank(), axis.ticks = element_blank(), panel.border = element_blank(),
    panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
    panel.background = element_blank(), axis.text.x = element_blank(), axis.title = element_blank(),
    axis.ticks.length = grid::unit(0, "pt"),
    plot.title = element_text(size = f$tick, face = "bold", hjust = 0, colour = ctx$th$text, margin = margin(b = 2)),
    plot.title.position = "panel",
    plot.margin = margin(2, max(5, f$base * 0.55) + 3, max(5, f$base * 0.55), max(5, f$base * 0.55)))
  table_theme <- table_theme + suppressWarnings(theme(
    axis.text.y = element_text(colour = rev(unname(cols)), size = f$tick, face = "bold", hjust = 1)))
  tab <- ggplot(risk) +
    geom_text(aes(x = .data$time, y = .data$row, label = .data$n), size = ts(f$tick), colour = ctx$th$text,
              family = f$family) +
    x_scale() +
    scale_y_continuous(breaks = seq_len(k), labels = rev(keys), expand = c(0, 0)) +
    coord_cartesian(xlim = ax_x$lim, ylim = c(0.4, k + 0.6), expand = FALSE, clip = "off") +
    labs(title = "Number at risk", caption = if (nzchar(ctx$labels$caption)) ctx$labels$caption else NULL) +
    table_theme
  p <- p + labs(caption = NULL) + theme(plot.margin = margin(max(5, f$base * 0.55), max(5, f$base * 0.55) + 3, 2,
                                                              max(5, f$base * 0.55)))
  patchwork::wrap_plots(p, tab, ncol = 1, heights = grid::unit(c(1, k * row_pt), c("null", "pt")))
}
register("survival", render_survival)
