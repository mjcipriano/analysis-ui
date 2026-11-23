# Distribution graphs: histogram, density (overlay / ridgeline), QQ, ECDF.

keyed <- function(df, col = "group") {
  if (!is.factor(df[[col]])) df[[col]] <- factor(as.character(df[[col]]), levels = unique(as.character(df[[col]])))
  df$key <- as.character(df[[col]])
  df
}

render_histogram <- function(ctx) {
  o <- ctx$o
  bins <- ctx$frames$bins
  if (!has_rows(bins)) aui_stop("There are no numeric values to plot yet.")
  bins <- keyed(bins)
  keys <- levels(bins$group); k <- length(keys)
  style <- if (k > 1) opt_chr(o, "hist_style", "overlay") else "overlay"
  curve <- ctx$frames$curve
  has_curve <- has_rows(curve)
  if (has_curve) curve <- keyed(curve)
  bins <- bins[is.finite(bins$height) & is.finite(bins$left) & is.finite(bins$right), , drop = FALSE]
  bins$ymin <- 0; bins$ymax <- bins$height
  bins$xmin <- bins$left; bins$xmax <- bins$right
  gi <- as.integer(bins$group)
  if (style == "stack") {                                   # first group on top, like the legend
    ord <- order(bins$left, -gi)
    bins <- bins[ord, , drop = FALSE]
    top <- stats::ave(bins$height, bins$left, FUN = cumsum)
    bins$ymax <- top; bins$ymin <- top - bins$height
  } else if (style == "dodge") {
    w <- (bins$right - bins$left) / k
    bins$xmin <- bins$left + (gi - 1) * w; bins$xmax <- bins$xmin + w
  }
  ax_x <- axis_spec(ctx, "x", finite_range(bins$left, bins$right, if (has_curve) curve$x),
                    mult = if (has_curve) c(0, 0) else c(0.03, 0.03))
  ax_y <- axis_spec(ctx, "y", finite_range(bins$ymax, if (has_curve) curve$y, 0), zero = TRUE, flush = TRUE,
                    mult = c(0, 0.06))
  legend <- legend_position(ctx, if (k > 1 && style != "facet") "right" else "none")
  if (style == "facet") legend <- "none"
  P <- xy_panel(ctx, ax_y, legend, keys)

  bins <- bins[on_scale(ax_x, bins$xmin) & bins$ymax > bins$ymin, , drop = FALSE]
  if (ax_y$is_log) bins$ymin <- pmax(bins$ymin, ax_y$lim[1])
  outline <- opt_chr(o, "outline", "white")
  bin_pt <- if (nrow(bins)) stats::median(ax_x$tr(bins$xmax) - ax_x$tr(bins$xmin)) / diff(ax_x$tlim) * P$w else 10
  line_pt <- if (!is.finite(bin_pt) || bin_pt >= 5) 0.75 else if (bin_pt >= 2.5) 0.4 else 0
  m <- aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$ymax, fill = .data$key)
  args <- list(data = bins, alpha = opt_num(o, "fill_alpha", 0.6), linewidth = lw(max(line_pt, 0.01)),
               linejoin = "mitre", show.legend = c(fill = TRUE, colour = FALSE, linetype = FALSE))
  if (line_pt == 0) outline <- "none"
  if (outline == "match") m$colour <- quote(.data$key)
  else args$colour <- switch(outline, white = ctx$th$panel_bg, black = ctx$ink, NA)
  args$mapping <- m
  layers <- list(do.call(geom_rect, args))
  scales <- list()
  if (has_curve) {
    curve <- curve[on_scale(ax_x, curve$x) & on_scale(ax_y, curve$y), , drop = FALSE]
    curve$ckey <- paste0(curve$key, DARK_SUFFIX)
    curve$kind <- as.character(curve$kind %||% "kde")
    layers <- c(layers, list(geom_line(
      aes(x = .data$x, y = .data$y, colour = .data$ckey, linetype = .data$kind,
          group = interaction(.data$key, .data$kind)), data = curve, linewidth = lw(opt_num(o, "line_width", 1.5)),
      show.legend = c(colour = FALSE, fill = FALSE, linetype = TRUE))))
    kinds <- unique(curve$kind)
    scales <- list(scale_linetype_manual(
      values = c(kde = "solid", normal = "42"), breaks = c("kde", "normal"), labels = c("Density", "Normal"),
      name = NULL, guide = if (length(kinds) > 1 && legend != "none") guide_legend(order = 2, override.aes = list(colour = ctx$ink)) else "none"))
  }
  if (opt_flag(o, "show_rug") && has_rows(ctx$frames$main)) {
    rug <- keyed(ctx$frames$main)
    rug <- rug[on_scale(ax_x, rug$value), , drop = FALSE]
    rug$ckey <- paste0(rug$key, DARK_SUFFIX)
    layers <- c(layers, list(geom_rug(aes(x = .data$value, colour = .data$ckey), data = rug, sides = "b", alpha = 0.6,
                                      linewidth = lw(0.6), length = grid::unit(min(0.035, 7 / P$h), "npc"),
                                      show.legend = FALSE)))
  }
  theme_extra <- NULL
  if (style == "facet") {
    rows <- max(1, floor(P$h / (ctx$f$tick * 5.5)))            # panels need a minimal height
    ncol <- ceiling(k / min(rows, k))
    layers <- c(layers, list(facet_wrap(~group, ncol = ncol)))
    ax_y <- axis_on_page(ax_y, P$h / ceiling(k / ncol) - ctx$f$tick * 2.2, FALSE)
    ax_x <- axis_on_page(ax_x, P$w / ncol - 6, TRUE)
    if (ctx$th$frame == "L" && ctx$th$axis_width > 0 && !ax_y$is_log) {
      layers <- c(layers, list(geom_hline(yintercept = ax_y$lim[1], colour = ctx$th$axis, linewidth = lw(ctx$th$axis_width) * 2)))
    }
    theme_extra <- theme(panel.spacing.y = grid::unit(ctx$f$tick * 0.6, "pt"))
  }
  p <- xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, scales = scales, theme_extra = theme_extra)
  if (legend != "none") p <- p + guides(fill = legend_guide(ctx, legend, keys, order = 1), colour = "none")
  p
}
register("histogram", render_histogram)

render_density <- function(ctx) {
  o <- ctx$o
  dens <- ctx$frames$density
  if (!has_rows(dens)) aui_stop("There are no numeric values to plot yet.")
  dens <- keyed(dens)
  dens <- dens[is.finite(dens$x) & is.finite(dens$y), , drop = FALSE]
  keys <- levels(dens$group); k <- length(keys)
  ridges <- opt_chr(ctx$computed, "style", "overlay") == "ridgeline"
  alpha <- opt_num(o, "fill_alpha", 0.35); line_pt <- opt_num(o, "line_width", 1.5)
  ax_x <- axis_spec(ctx, "x", finite_range(dens$x), mult = c(0, 0))
  dens <- dens[on_scale(ax_x, dens$x), , drop = FALSE]
  dens <- dens[order(dens$group, dens$x), , drop = FALSE]
  rug <- NULL
  if (opt_flag(o, "show_rug") && has_rows(ctx$frames$main)) {
    rug <- keyed(ctx$frames$main)
    rug <- rug[on_scale(ax_x, rug$value) & rug$key %in% keys, , drop = FALSE]
  }

  if (!ridges) {
    ax_y <- axis_spec(ctx, "y", finite_range(dens$y, 0), zero = TRUE, flush = TRUE, mult = c(0, 0.06))
    axes_ready(ctx, "any", x = ax_x, y = ax_y)
    legend <- legend_position(ctx, if (k > 1) "right" else "none")
    P <- xy_panel(ctx, ax_y, legend, keys)
    base <- if (ax_y$is_log) ax_y$lim[1] else 0
    dens <- dens[on_scale(ax_y, dens$y) | !ax_y$is_log, , drop = FALSE]
    layers <- list(
      geom_ribbon(aes(x = .data$x, ymin = base, ymax = .data$y, fill = .data$key, group = .data$key), data = dens,
                  alpha = alpha, colour = NA, show.legend = TRUE),
      geom_line(aes(x = .data$x, y = .data$y, colour = .data$key, group = .data$key), data = dens,
                linewidth = lw(line_pt), show.legend = FALSE))
    if (has_rows(rug)) layers <- c(layers, list(geom_rug(
      aes(x = .data$value, colour = .data$key), data = rug, sides = "b", alpha = 0.6, linewidth = lw(0.6),
      length = grid::unit(min(0.035, 7 / P$h), "npc"), show.legend = FALSE)))
    return(xy_assemble(ctx, ax_x, ax_y, layers, legend, keys))
  }

  # ---- ridgeline: one row per group, first group at the top
  overlap <- opt_num(o, "ridge_overlap", 1.5)
  peak <- max(dens$y, 0)
  if (!is.finite(peak) || peak <= 0) peak <- 1
  gi <- as.integer(dens$group)
  dens$gi <- gi
  dens$base <- k - gi
  dens$top <- dens$base + dens$y / peak * overlap
  ymax <- max(dens$top, k - 1 + 0.3)
  ylim <- c(-0.12 - (if (has_rows(rug)) 0.1 else 0), ymax + (ymax + 0.12) * 0.03)
  # the rows live on a plain linear pseudo-axis (so panels of a facet grid can share or free it)
  ax_y <- ctx$fixed_axes$y %||% list(which = "y", scale = "linear", tr = identity, inv = identity, trans = "identity",
                                     lim = ylim, tlim = ylim, user = c(FALSE, FALSE), is_log = FALSE)
  ylim <- ax_y$lim
  axes_ready(ctx, "any", x = ax_x, y = ax_y)
  legend <- legend_position(ctx, "none")
  lab_w <- max_text_w(ctx, keys, ctx$f$tick)
  P <- panel_size(ctx, left_w = lab_w, legend = legend, legend_labels = keys)
  layers <- list(
    geom_hline(yintercept = k - seq_len(k), colour = ctx$th$grid_color, linewidth = lw(0.6)),
    geom_ribbon(aes(x = .data$x, ymin = .data$base, ymax = .data$top, fill = .data$key, colour = .data$key,
                    group = .data$gi), data = dens, alpha = alpha, linewidth = lw(line_pt), outline.type = "upper",
                show.legend = legend != "none"))
  if (has_rows(rug)) {
    rug$base <- k - as.integer(rug$group)
    tick <- 5 / P$h * diff(ylim)
    layers <- c(layers, list(geom_segment(aes(x = .data$value, xend = .data$value, y = .data$base - tick * 0.25,
                                              yend = .data$base - tick * 1.25, colour = .data$key), data = rug,
                                          alpha = 0.7, linewidth = lw(0.6), show.legend = FALSE)))
  }
  xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, dark = FALSE,
              y_scale = scale_y_continuous(breaks = k - seq_len(k), labels = keys, expand = c(0, 0)),
              theme_extra = theme(axis.line.y = element_blank(), axis.ticks.y = element_blank(),
                                  panel.grid.major.y = element_blank(),
                                  axis.text.y = element_text(vjust = 0, size = ctx$f$tick)))
}
register("density", function(ctx) with_facets(ctx, render_density, xy_finish))

render_qq <- function(ctx) {
  o <- ctx$o
  qq <- ctx$frames$qq
  if (!has_rows(qq)) aui_stop("A QQ plot needs at least 3 values per group.")
  qq <- keyed(qq)
  keys <- levels(qq$group)
  ax_x <- axis_spec(ctx, "x", finite_range(qq$theoretical))
  ax_y <- axis_spec(ctx, "y", finite_range(qq$sample))
  axes_ready(ctx, "any", x = ax_x, y = ax_y)
  legend <- legend_position(ctx, if (length(keys) > 1) "right" else "none")
  qq <- qq[on_scale(ax_x, qq$theoretical) & on_scale(ax_y, qq$sample), , drop = FALSE]
  layers <- list()
  ln <- ctx$frames$line
  if (opt_flag(o, "show_line", TRUE) && has_rows(ln) && ax_x$scale == "linear" && ax_y$scale == "linear") {
    ln <- keyed(ln)
    ln <- ln[is.finite(ln$slope) & is.finite(ln$intercept), , drop = FALSE]
    single <- length(keys) == 1
    args <- list(mapping = aes(slope = .data$slope, intercept = .data$intercept), data = ln,
                 linewidth = lw(opt_num(o, "line_width", 1.2)), show.legend = FALSE)
    if (single) { args$colour <- ctx$ink; args$linetype <- "42" } else args$mapping$colour <- quote(.data$key)
    if (nrow(ln)) layers <- list(do.call(geom_abline, args))
  }
  spec <- marker_spec(ctx, mode = "group")
  layers <- c(layers, list(marker_layer(spec, data.frame(x = qq$theoretical, y = qq$sample, key = qq$key),
                                        show.legend = TRUE)))
  xy_assemble(ctx, ax_x, ax_y, layers, legend, keys)
}
register("qq", function(ctx) with_facets(ctx, render_qq, xy_finish))

render_ecdf <- function(ctx) {
  o <- ctx$o
  d <- ctx$frames$ecdf
  if (!has_rows(d)) aui_stop("There are no numeric values to plot yet.")
  d <- keyed(d)
  keys <- levels(d$group)
  top <- if (opt_flag(ctx$computed, "percent")) 100 else 1
  ax_x <- axis_spec(ctx, "x", finite_range(d$x), mult = c(0.04, 0.04))
  ax_y <- axis_spec(ctx, "y", c(0, top), fixed = c(0, top * 1.03))
  axes_ready(ctx, "any", x = ax_x, y = ax_y)
  legend <- legend_position(ctx, if (length(keys) > 1) "right" else "none")
  d <- d[on_scale(ax_x, d$x) & (on_scale(ax_y, d$y) | !ax_y$is_log), , drop = FALSE]
  d <- d[order(d$group, d$x, d$y), , drop = FALSE]
  # extend every curve to the right edge at its final value
  ends <- do.call(rbind, lapply(split(d, d$group, drop = TRUE), function(g) {
    e <- g[nrow(g), , drop = FALSE]; e$x <- max(d$x); e
  }))
  d <- rbind(d, ends)
  d <- d[order(d$group, d$x, d$y), , drop = FALSE]
  layers <- list(geom_step(aes(x = .data$x, y = .data$y, colour = .data$key, group = .data$key), data = d,
                           direction = "hv", linewidth = lw(opt_num(o, "line_width", 1.6)), show.legend = TRUE))
  xy_assemble(ctx, ax_x, ax_y, layers, legend, keys, key_w = 2)
}
register("ecdf", function(ctx) with_facets(ctx, render_ecdf, xy_finish))
