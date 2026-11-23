# Specialised graphs: Bland–Altman, ROC, volcano, forest.

render_bland_altman <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed; f <- ctx$f
  d <- ctx$frames$main
  if (!has_rows(d)) aui_stop("A Bland–Altman plot needs at least 3 complete pairs.")
  d <- d[is.finite(d$mean) & is.finite(d$diff), , drop = FALSE]
  bias <- opt_num(comp, "bias"); lo <- opt_num(comp, "loa_lo"); hi <- opt_num(comp, "loa_hi")
  ci <- list(bias = num_vec(comp$bias_ci), lo = num_vec(comp$loa_lo_ci), hi = num_vec(comp$loa_hi_ci))
  show_ci <- opt_flag(o, "show_ci", TRUE)
  show_lab <- opt_flag(o, "show_line_labels", TRUE)
  col <- if (length(ctx$cols)) unname(ctx$cols[1]) else ctx$ink
  line_pt <- opt_num(o, "line_width", 1.2)
  digits <- function(v) { r <- max(abs(c(bias, lo, hi)), na.rm = TRUE); if (!is.finite(r) || r == 0) 2 else max(0, 2 - floor(log10(r))) }
  fmt <- function(v) minus(formatC(v, format = "f", digits = min(digits(v), 6)))
  labels <- c(sprintf("+1.96 SD: %s", fmt(hi)), sprintf("Mean: %s", fmt(bias)), sprintf("−1.96 SD: %s", fmt(lo)))
  ax_x <- axis_spec(ctx, "x", finite_range(d$mean), mult = c(0.05, 0.05))
  ax_y <- axis_spec(ctx, "y", finite_range(d$diff, bias, lo, hi, if (show_ci) unlist(ci)), mult = c(0.07, 0.12))
  layers <- list()
  add <- function(...) layers <<- c(layers, list(...))
  band <- function(r, fill, alpha) {
    if (length(r) == 2 && all(is.finite(r))) add(annotate("rect", xmin = -Inf, xmax = Inf, ymin = min(r), ymax = max(r),
                                                          fill = fill, alpha = alpha))
  }
  if (show_ci && ax_y$scale == "linear") {
    band(ci$bias, col, 0.14); band(ci$lo, ctx$muted, 0.16); band(ci$hi, ctx$muted, 0.16)
  }
  if (!length(ctx$hlines) && ax_y$scale == "linear") add(geom_hline(yintercept = if (isTRUE(comp$ratio)) 1 else 0, colour = ctx$muted, linewidth = lw(0.6)))
  if (is.finite(bias)) add(geom_hline(yintercept = bias, colour = col, linewidth = lw(line_pt)))
  loa <- c(lo, hi)[is.finite(c(lo, hi))]
  if (length(loa)) add(geom_hline(yintercept = loa, colour = ctx$ink, linewidth = lw(line_pt), linetype = "42"))
  spec <- marker_spec(ctx, mode = "group")
  key <- if (length(ctx$cols)) names(ctx$cols)[1] else "Points"
  d <- d[on_scale(ax_x, d$mean) & on_scale(ax_y, d$diff), , drop = FALSE]
  add(marker_layer(spec, data.frame(x = d$mean, y = d$diff, key = key)))
  if (show_lab) {
    ld <- data.frame(y = c(hi, bias, lo), label = labels)
    ld <- ld[is.finite(ld$y), , drop = FALSE]
    add(geom_text(aes(x = Inf, y = .data$y, label = .data$label), data = ld, hjust = 1.04, vjust = -0.5,
                  size = ts(f$tick * 0.92), colour = ctx$th$text, family = f$family))
  }
  xy_assemble(ctx, ax_x, ax_y, layers, "none", key)
}
register("bland_altman", render_bland_altman)

render_roc <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed
  d <- ctx$frames$roc
  if (!has_rows(d)) aui_stop("A ROC curve needs a binary outcome and at least one numeric predictor.")
  d <- keyed(d, "name")
  keys <- levels(d$name)
  top <- if (opt_flag(comp, "percent")) 100 else 1
  ax_x <- axis_spec(ctx, "x", c(0, top), fixed = c(0, top))
  ax_y <- axis_spec(ctx, "y", c(0, top), fixed = c(0, top))
  curves <- records(comp$curves, c("name", "label"))
  show_auc <- opt_flag(o, "show_auc", TRUE) && nrow(curves) > 0
  labels <- stats::setNames(keys, keys)
  if (show_auc) {
    m <- match(keys, curves$name)
    labels[!is.na(m)] <- curves$label[m[!is.na(m)]]
    labels[is.na(labels)] <- keys[is.na(labels)]
  }
  legend <- legend_position(ctx, if (length(keys) > 1 || show_auc) "inside_br" else "none")
  line_pt <- opt_num(o, "line_width", 1.8)
  layers <- list()
  if (opt_flag(o, "fill_area")) {
    layers <- c(layers, list(geom_ribbon(aes(x = .data$fpr, ymin = 0, ymax = .data$tpr, fill = .data$key, group = .data$key),
                                         data = d, alpha = if (length(keys) > 1) 0.1 else 0.16, colour = NA,
                                         show.legend = FALSE)))
  }
  if (opt_flag(o, "show_diagonal", TRUE)) {
    layers <- c(layers, list(annotate("segment", x = 0, y = 0, xend = top, yend = top, colour = ctx$muted,
                                      linewidth = lw(0.9), linetype = "42")))
  }
  layers <- c(layers, list(geom_path(aes(x = .data$fpr, y = .data$tpr, colour = .data$key, group = .data$key), data = d,
                                     linewidth = lw(line_pt), lineend = "round", linejoin = "round", show.legend = TRUE)))
  br <- seq(0, top, length.out = 6)
  xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, key_labels = labels, key_w = 2, x_breaks = br, y_breaks = br,
              coord = coord_fixed(xlim = ax_x$lim, ylim = ax_y$lim, expand = FALSE, clip = "off"))
}
register("roc", render_roc)

render_volcano <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed; f <- ctx$f
  d <- ctx$frames$main
  if (!has_rows(d)) aui_stop("There are no complete effect / P value rows to plot yet.")
  d <- keyed(d, "status")
  d <- d[is.finite(d$x) & is.finite(d$y), , drop = FALSE]
  keys <- levels(d$status)
  counts <- comp$counts %||% list()
  labels <- stats::setNames(vapply(keys, function(kk) {
    n <- opt_num(counts, kk, sum(d$key == kk))
    sprintf("%s (%s)", kk, format(n, big.mark = ","))
  }, character(1)), keys)
  xt <- opt_num(comp, "x_threshold", 0); yt <- opt_num(comp, "y_threshold", NA)
  xr <- finite_range(d$x)
  symmetric <- is.na(opt_num(o, "x_min")) && is.na(opt_num(o, "x_max")) && opt_chr(o, "x_scale", "linear") == "linear" &&
    all(is.finite(xr)) && xr[1] < 0 && xr[2] > 0
  if (symmetric) xr <- c(-1, 1) * max(abs(xr), xt * 1.15)
  ax_x <- axis_spec(ctx, "x", xr, mult = c(0.06, 0.06))
  has_labels <- any(!is.na(d$label) & nzchar(d$label))
  ax_y <- axis_spec(ctx, "y", finite_range(d$y, 0), zero = TRUE, flush = TRUE, mult = c(0, if (has_labels) 0.14 else 0.06))
  legend <- legend_position(ctx, "right")
  d <- d[on_scale(ax_x, d$x) & (on_scale(ax_y, d$y) | !ax_y$is_log), , drop = FALSE]
  layers <- list()
  if (opt_flag(o, "show_thresholds", TRUE)) {
    lwd <- lw(0.8)
    if (is.finite(yt) && on_scale(ax_y, yt)) layers <- c(layers, list(geom_hline(yintercept = yt, colour = ctx$muted, linewidth = lwd, linetype = "42")))
    v <- c(-xt, xt); v <- v[is.finite(v) & v != 0 & on_scale(ax_x, v)]
    if (xt > 0 && length(v)) layers <- c(layers, list(geom_vline(xintercept = v, colour = ctx$muted, linewidth = lwd, linetype = "42")))
  }
  dia <- opt_num(o, "point_size", 3.5); alpha <- opt_num(o, "point_alpha", 0.75)
  ns <- d$key == "Not significant" | !(d$key %in% c("Up", "Down"))
  d <- rbind(d[ns, , drop = FALSE], d[!ns, , drop = FALSE])     # significant points on top
  layers <- c(layers, list(geom_point(aes(x = .data$x, y = .data$y, colour = .data$key), data = d, shape = 16,
                                      size = pt_size(dia, 16), stroke = 0, alpha = alpha, show.legend = TRUE)))
  lab <- d[!is.na(d$label) & nzchar(d$label), , drop = FALSE]
  if (nrow(lab)) {
    size_pt <- opt_num(o, "label_size", f$tick * 0.85)
    layers <- c(layers, list(ggrepel::geom_text_repel(
      aes(x = .data$x, y = .data$y, label = .data$label), data = lab, size = ts(size_pt), colour = ctx$th$text,
      family = f$family, seed = 1, max.overlaps = Inf, max.time = 0.3, max.iter = 4000, min.segment.length = 0.2,
      segment.size = lw(0.5), segment.colour = ctx$muted, box.padding = 0.3, point.padding = dia / 72.27 / 2 + 0.01,
      show.legend = FALSE)))
  }
  xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, key_labels = labels, dark = FALSE) +
    guides(colour = guide_legend(override.aes = list(size = pt_size(max(dia, 6), 16), alpha = 1), reverse = TRUE),
           fill = "none")
}
register("volcano", render_volcano)

render_forest <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed; f <- ctx$f
  d <- ctx$frames$main
  if (!has_rows(d)) aui_stop("There are no estimates to plot yet.")
  if (!is.factor(d$label)) d$label <- factor(as.character(d$label), levels = unique(as.character(d$label)))
  d$label <- droplevels(d$label)
  rows <- levels(d$label); n <- length(rows)
  d$row <- n + 1 - as.integer(d$label)                       # first row at the top
  d$text <- ifelse(is.na(d$text), "", as.character(d$text))
  ref <- opt_num(comp, "ref_line", NA)
  show_text <- opt_flag(comp, "show_text", TRUE) && any(nzchar(d$text))
  ax_x <- axis_spec(ctx, "x", finite_range(d$est, d$lo, d$hi, ref), mult = c(0.06, 0.06))
  col <- if (length(ctx$cols)) unname(ctx$cols[1]) else ctx$ink
  key <- if (length(ctx$cols)) names(ctx$cols)[1] else "Estimate"
  text_w_pt <- if (show_text) max_text_w(ctx, d$text, f$tick) + 8 else 0
  P <- panel_size(ctx, left_w = max_text_w(ctx, rows, f$tick), right = text_w_pt)
  row_pt <- P$h / n
  d <- d[on_scale(ax_x, d$est), , drop = FALSE]
  line_pt <- opt_num(o, "line_width", 1.4)
  layers <- list()
  add <- function(...) layers <<- c(layers, list(...))
  if (is.finite(ref) && on_scale(ax_x, ref)) add(geom_vline(xintercept = ref, colour = ctx$muted, linewidth = lw(0.9), linetype = "42"))
  ci <- d[is.finite(d$lo) | is.finite(d$hi), , drop = FALSE]
  if (nrow(ci)) {
    ci$lo[!on_scale(ax_x, ci$lo)] <- NA; ci$hi[!on_scale(ax_x, ci$hi)] <- NA
    a <- ifelse(is.finite(ci$lo), ci$lo, ci$est); b <- ifelse(is.finite(ci$hi), ci$hi, ci$est)
    seg <- data.frame(x = a, xend = b, y = ci$row, yend = ci$row)
    cap <- opt_num(o, "error_cap", 0)
    if (cap > 0) {
      h <- cap / row_pt                                       # half cap height in row units
      ends <- c(ci$lo[is.finite(ci$lo)], ci$hi[is.finite(ci$hi)])
      rws <- c(ci$row[is.finite(ci$lo)], ci$row[is.finite(ci$hi)])
      seg <- rbind(seg, data.frame(x = ends, xend = ends, y = rws - h, yend = rws + h))
    }
    add(geom_segment(aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend), data = seg, colour = col,
                     linewidth = lw(line_pt), lineend = "butt"))
  }
  spec <- marker_spec(ctx, size = opt_num(o, "point_size", 7), shape = opt_chr(o, "point_shape", "square"),
                      alpha = 1, mode = "group")
  add(marker_layer(spec, data.frame(x = d$est, y = d$row, key = key)))
  txt <- character(n); txt[d$row] <- d$text
  sec <- if (show_text) dup_axis(name = NULL, breaks = seq_len(n), labels = txt) else waiver()
  angle <- tick_angle(opt_chr(o, "x_tick_angle", "auto"))
  ggplot() + ref_lines(ctx, ax_x, NULL) + layers +
    axis_scale(ctx, ax_x) +
    scale_y_continuous(breaks = seq_len(n), labels = rev(rows), expand = c(0, 0), sec.axis = sec) +
    colour_scales(ctx, key) +
    coord_cartesian(xlim = ax_x$lim, ylim = c(0.4, n + 0.6), expand = FALSE) +
    gg_labs(ctx) + gg_theme(ctx, legend = "none", x_angle = angle) +
    theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(), panel.grid.major.y = element_blank(),
          axis.ticks.length.y = grid::unit(0, "pt"))
}
register("forest", render_forest)
