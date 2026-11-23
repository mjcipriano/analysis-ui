# XY graphs: scatter, line graph, dose-response, bubble (render type "xy").

LINETYPES <- c("solid", "42", "13", "4212", "62", "2262")
SHAPE_CYCLE <- c(16, 15, 17, 18, 1, 0, 2, 5)

# shared tail of every graph with two numeric axes
xy_assemble <- function(ctx, ax_x, ax_y, layers, legend, keys, key_labels = NULL, key_w = 1, extra_cols = NULL,
                        scales = list(), coord = NULL, theme_extra = NULL, dark = TRUE,
                        x_breaks = NULL, y_breaks = NULL) {
  angle <- tick_angle(opt_chr(ctx$o, "x_tick_angle", "auto"))
  ggplot() + ref_lines(ctx, ax_x, ax_y) + layers +
    axis_scale(ctx, ax_x, breaks = x_breaks, labels = if (!is.null(x_breaks)) fmt_num(x_breaks)) +
    axis_scale(ctx, ax_y, breaks = y_breaks, labels = if (!is.null(y_breaks)) fmt_num(y_breaks)) +
    colour_scales(ctx, keys, labels = key_labels, extra = c(if (dark) dark_extra(ctx, keys), extra_cols)) + scales +
    (coord %||% coord_cartesian(xlim = ax_x$lim, ylim = ax_y$lim, expand = FALSE)) +
    gg_labs(ctx) + gg_theme(ctx, legend = legend, x_angle = angle, key_w = key_w) + theme_extra
}

# panel size for two numeric axes
xy_panel <- function(ctx, ax_y, legend = "none", legend_labels = NULL, key_w = 1, right = 0) {
  left_w <- max_text_w(ctx, axis_breaks(ctx, ax_y, minor_ok = FALSE)$labels, ctx$f$tick)
  panel_size(ctx, left_w = left_w, legend = legend, legend_labels = legend_labels, key_w = key_w, right = right)
}

render_xy <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed
  main <- ctx$frames$main
  if (!has_rows(main)) aui_stop("There are no complete X–Y pairs to plot yet.")
  if (!is.factor(main$group)) main$group <- factor(as.character(main$group), levels = unique(as.character(main$group)))
  keys <- levels(main$group)
  summ <- ctx$frames$summary; fit <- ctx$frames$fit
  mode <- opt_chr(comp, "mode", "points")
  if (!has_rows(summ)) mode <- "points"
  has_fit <- has_rows(fit)
  show_points <- opt_flag(o, "show_points", TRUE)
  connect <- opt_chr(o, "connect", "none")
  line_pt <- opt_num(o, "line_width", 1.5)
  style <- opt_chr(o, "line_style", "solid")
  err_style <- opt_chr(o, "error_style", "bars")
  bubbles <- opt_flag(comp, "has_size") && !is.null(main$size) && any(is.finite(main$size)) && mode == "points"

  # ---- axes
  yv <- if (mode == "summary") c(summ$center, summ$lo, summ$hi) else c(main$y, if (mode == "both") c(summ$lo, summ$hi))
  yr <- finite_range(yv)
  if (has_fit && all(is.finite(yr))) {              # let the curve in, but not runaway extrapolation
    slack <- max(diff(yr), abs(yr[1]) * 0.05, 1e-9) * 0.5
    fy <- fit$y[is.finite(fit$y)]
    yr <- range(yr, pmin(pmax(fy, yr[1] - slack), yr[2] + slack))
  }
  ax_x <- axis_spec(ctx, "x", finite_range(main$x, if (has_fit) fit$x))
  ax_y <- axis_spec(ctx, "y", yr, mult = if (bubbles) c(0.09, 0.09) else c(0.05, 0.05))
  if (bubbles) ax_x <- axis_spec(ctx, "x", finite_range(main$x, if (has_fit) fit$x), mult = c(0.07, 0.07))

  # ---- legend
  fits <- records(comp$fits, c("group", "label"))
  fit_labels <- NULL
  if (has_fit && opt_flag(o, "fit_legend", TRUE) && nrow(fits)) {
    lab <- stats::setNames(fits$label, fits$group)
    lab <- lab[!is.na(lab) & names(lab) %in% keys]
    if (length(lab)) fit_labels <- if (length(keys) == 1) lab else stats::setNames(paste0(names(lab), "\n", lab), names(lab))
  }
  legend_labels <- keys
  if (!is.null(fit_labels)) legend_labels[match(names(fit_labels), keys)] <- fit_labels
  default <- if (length(keys) > 1 || !is.null(fit_labels) || bubbles) "right" else "none"
  if (default == "right" && !is.null(fit_labels) &&
      max_text_w(ctx, legend_labels, ctx$f$legend) > ctx$W * 72 * 0.3) default <- "bottom"
  legend <- legend_position(ctx, default)
  has_lines <- connect != "none" || has_fit
  key_w <- if (has_lines) 2.2 else 1
  P <- xy_panel(ctx, ax_y, legend, legend_labels, key_w)

  on_xy <- function(d, xc = "x", yc = "y") d[on_scale(ax_x, d[[xc]]) & on_scale(ax_y, d[[yc]]), , drop = FALSE]
  main <- on_xy(main)
  main$key <- as.character(main$group)
  if (mode != "points") {
    summ <- on_xy(summ, "x", "center")
    summ$key <- as.character(summ$group)
    bad <- function(v) !is.na(v) & !on_scale(ax_y, v)
    summ$lo[bad(summ$lo)] <- NA; summ$hi[bad(summ$hi)] <- NA
  }
  layers <- list()
  add <- function(...) layers <<- c(layers, list(...))
  lt_map <- style == "auto"
  lt_const <- switch(style, dashed = "42", dotted = "13", "solid")
  line_geom <- function(geom, data, mapping, ...) {
    if (lt_map) { mapping$linetype <- quote(.data$key); geom(mapping, data = data, linewidth = lw(line_pt), ...) }
    else geom(mapping, data = data, linewidth = lw(line_pt), linetype = lt_const, ...)
  }

  # ---- bands
  if (has_fit) {
    fit <- fit[on_scale(ax_x, fit$x) & on_scale(ax_y, fit$y), , drop = FALSE]
    fit$key <- as.character(fit$group)
    fit <- fit[order(fit$group, fit$x), , drop = FALSE]
    band <- fit[is.finite(fit$lo) & is.finite(fit$hi), , drop = FALSE]
    if (opt_flag(o, "fit_band", TRUE) && nrow(band)) {
      if (ax_y$is_log) band$lo <- pmax(band$lo, ax_y$lim[1] * 1e-3)
      add(geom_ribbon(aes(x = .data$x, ymin = .data$lo, ymax = .data$hi, fill = .data$key, group = .data$key), data = band,
                      alpha = 0.18, colour = NA, show.legend = FALSE))
    }
  }
  centre <- if (mode == "points") NULL else summ[order(summ$group, summ$x), , drop = FALSE]
  if (!is.null(centre) && err_style == "band") {
    eb <- centre[is.finite(centre$lo) & is.finite(centre$hi), , drop = FALSE]
    if (nrow(eb)) add(geom_ribbon(aes(x = .data$x, ymin = .data$lo, ymax = .data$hi, fill = .data$key, group = .data$key),
                                  data = eb, alpha = 0.22, colour = NA, show.legend = FALSE))
  }
  # ---- lines
  if (connect != "none") {
    ld <- if (is.null(centre)) data.frame(x = main$x, y = main$y, key = main$key, g = main$group)
          else data.frame(x = centre$x, y = centre$center, key = centre$key, g = centre$group)
    ld <- ld[order(ld$g, ld$x), , drop = FALSE]
    m <- aes(x = .data$x, y = .data$y, colour = .data$key, group = .data$key)
    if (connect == "step") add(line_geom(geom_step, ld, m, direction = "hv", show.legend = TRUE))
    else add(line_geom(geom_line, ld, m, show.legend = TRUE))
  }
  if (has_fit && nrow(fit)) {
    add(line_geom(geom_line, fit, aes(x = .data$x, y = .data$y, colour = .data$key, group = .data$key), show.legend = TRUE))
  }
  # ---- points
  shape_opt <- opt_chr(o, "point_shape", "circle")
  d_pt <- opt_num(o, "point_size", 6)
  spec <- marker_spec(ctx, mode = "group")
  point_layer <- function(df, spec, show.legend) {
    if (shape_opt == "auto" && !bubbles) {
      geom_point(aes(x = .data$x, y = .data$y, colour = .data$key, shape = .data$key), data = df,
                 size = pt_size(d_pt, 16), stroke = max(0.6, min(d_pt / 6, 1.6)) / 1.417, alpha = spec$alpha,
                 show.legend = show.legend)
    } else marker_layer(spec, df, show.legend = show.legend)
  }
  if (mode == "both" && show_points) {
    raw <- marker_spec(ctx, size = max(d_pt * 0.6, 2), alpha = min(spec$alpha, 0.45), mode = "group")
    add(point_layer(data.frame(x = main$x, y = main$y, key = main$key), raw, FALSE))
  }
  if (!is.null(centre) && err_style != "band") {
    eb <- centre[is.finite(centre$lo) | is.finite(centre$hi), , drop = FALSE]
    eb$lo[!is.finite(eb$lo)] <- eb$center[!is.finite(eb$lo)]
    eb$hi[!is.finite(eb$hi)] <- eb$center[!is.finite(eb$hi)]
    cap <- 2 * opt_num(o, "error_cap", 3) / P$w * diff(ax_x$tlim)
    if (nrow(eb)) add(geom_errorbar(aes(x = .data$x, ymin = .data$lo, ymax = .data$hi, colour = .data$key), data = eb,
                                    width = cap, linewidth = lw(max(0.75, min(line_pt * 0.7, 1.5))), show.legend = FALSE))
  }
  if (show_points) {
    if (is.null(centre)) {
      pd <- data.frame(x = main$x, y = main$y, key = main$key)
      if (bubbles) {
        pd$size <- main$size
        pd <- pd[is.finite(pd$size) & pd$size >= 0, , drop = FALSE]
        pd <- pd[order(-pd$size), , drop = FALSE]            # small bubbles on top
        add(geom_point(aes(x = .data$x, y = .data$y, fill = .data$key, size = .data$size), data = pd, shape = 21,
                       colour = ctx$th$panel_bg, stroke = 0.35, alpha = spec$alpha, show.legend = TRUE))
      } else add(point_layer(pd, spec, TRUE))
    } else {
      add(point_layer(data.frame(x = centre$x, y = centre$center, key = centre$key), spec, TRUE))
    }
  }
  # ---- labels and rug
  if (opt_flag(comp, "has_label") && !is.null(main$label) && mode == "points") {
    ld <- main[!is.na(main$label) & nzchar(main$label), , drop = FALSE]
    if (nrow(ld)) add(ggrepel::geom_text_repel(
      aes(x = .data$x, y = .data$y, label = .data$label), data = ld, size = ts(ctx$f$tick * 0.9), colour = ctx$th$text,
      family = ctx$f$family, seed = 1, max.overlaps = if (nrow(ld) > 80) 8 else 20, max.time = 0.25, max.iter = 3000,
      min.segment.length = 0.3, segment.size = lw(0.5), segment.colour = ctx$muted,
      box.padding = 0.25, point.padding = d_pt / 72.27 / 2, show.legend = FALSE))
  }
  if (opt_flag(o, "marginal_rug")) {
    add(geom_rug(aes(x = .data$x, y = .data$y, colour = .data$key), data = main, sides = "bl", alpha = 0.55,
                 linewidth = lw(0.6), length = grid::unit(min(0.03, 6 / P$h), "npc"), show.legend = FALSE))
  }

  title <- if (nzchar(ctx$labels$legend)) ctx$labels$legend else NULL
  scales <- list()
  if (lt_map) scales <- c(scales, list(scale_linetype_manual(
    values = stats::setNames(LINETYPES[(seq_along(keys) - 1) %% length(LINETYPES) + 1], keys),
    breaks = keys, labels = legend_labels, name = title, drop = FALSE)))
  if (shape_opt == "auto" && !bubbles) scales <- c(scales, list(scale_shape_manual(
    values = stats::setNames(SHAPE_CYCLE[(seq_along(keys) - 1) %% length(SHAPE_CYCLE) + 1], keys),
    breaks = keys, labels = legend_labels, name = title, drop = FALSE)))
  if (bubbles) {
    size_title <- opt_chr(o, "size", "Size")
    scales <- c(scales, list(
      scale_size_area(max_size = pt_size(opt_num(o, "size_max", 22), 21, 0.35), name = size_title,
                      guide = guide_legend(order = 2, override.aes = list(fill = ctx$muted, alpha = 1))),
      guides(fill = if (length(keys) > 1) guide_legend(order = 1, override.aes = list(size = pt_size(8, 21, 0.35), alpha = 1))
                    else "none", colour = "none")))
  }
  p <- xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, key_labels = stats::setNames(legend_labels, keys),
                   key_w = key_w, scales = scales)
  if (bubbles) p <- p + theme(legend.title = element_text(colour = ctx$th$text, size = ctx$f$legend, face = "bold"))
  if (legend == "bottom" && !is.null(fit_labels)) {
    g <- guide_legend(ncol = 1, byrow = TRUE)
    p <- p + guides(colour = g, fill = g, linetype = if (lt_map) g else "none",
                    shape = if (shape_opt == "auto" && !bubbles) g else "none") +
      theme(legend.spacing.y = grid::unit(ctx$f$legend * 0.45, "pt"))
  }
  p
}
register("xy", render_xy)
