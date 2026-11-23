# Heat map and correlation matrix (same payload).

render_heatmap <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed; f <- ctx$f
  cells <- ctx$frames$cells
  if (!has_rows(cells)) aui_stop("There are no numeric values to plot yet.")
  fac <- function(v) if (is.factor(v)) v else factor(as.character(v), levels = unique(as.character(v)))
  cells$row <- fac(cells$row); cells$col <- fac(cells$col)
  rows <- levels(cells$row); cols <- levels(cells$col)
  nr <- length(rows); nc <- length(cols)
  cells$x <- as.integer(cells$col)
  cells$y <- nr + 1 - as.integer(cells$row)                 # first row at the top
  cells$text <- ifelse(is.na(cells$text), "", as.character(cells$text))
  vmin <- opt_num(comp, "vmin", NA); vmax <- opt_num(comp, "vmax", NA)
  rng <- finite_range(cells$value)
  if (is.na(vmin)) vmin <- if (is.finite(rng[1])) rng[1] else 0
  if (is.na(vmax)) vmax <- if (is.finite(rng[2])) rng[2] else 1
  if (vmax <= vmin) vmax <- vmin + 1
  stops <- chr_vec(o[["_colormap"]])
  if (length(stops) < 2) stops <- c("#440154", "#21908C", "#FDE725")
  show_bar <- opt_flag(o, "show_colorbar", TRUE)
  legend <- if (show_bar) legend_position(ctx, "right") else "none"
  bar_title <- if (nzchar(ctx$labels$legend)) ctx$labels$legend else opt_chr(comp, "legend_label", "")

  # ---- sizes (pt): room taken by labels and the colour bar, then the cells
  row_w <- max_text_w(ctx, rows, f$tick)
  col_w <- max_text_w(ctx, cols, f$tick)
  bar_w <- if (legend %in% c("right", "left")) f$legend * 1.1 + max(max_text_w(ctx, fmt_num(c(vmin, vmax, -0.55)), f$legend),
                                                                   max_text_w(ctx, bar_title, f$legend, TRUE) - f$legend * 1.1) + f$legend * 2 else 0
  bar_h <- if (legend %in% c("top", "bottom")) f$legend * 4.2 else 0
  P0 <- panel_size(ctx, left_w = row_w, bottom_h = f$tick * 1.1, right = bar_w, top = if (legend == "top") bar_h else 0)
  cw <- P0$w / nc
  choice <- opt_chr(o, "x_tick_angle", "auto")
  angle <- if (choice == "auto") {
    if (col_w <= cw * 0.92) NULL else if (f$tick * 1.5 > cw) tick_angle(90) else tick_angle(45)
  } else tick_angle(choice)
  a <- if (is.null(angle)) 0 else angle$angle
  bottom_h <- if (a == 0) f$tick * 1.15 else col_w * sin(a * pi / 180) + f$tick * cos(a * pi / 180)
  P <- panel_size(ctx, left_w = row_w, bottom_h = bottom_h + (if (legend == "bottom") bar_h else 0), right = bar_w,
                  top = if (legend == "top") bar_h else 0)
  square <- opt_flag(o, "square")
  cw <- P$w / nc; ch <- P$h / nr
  if (square) cw <- ch <- min(cw, ch)

  # ---- tiles
  tiles <- cells[is.finite(cells$value), , drop = FALSE]
  border <- opt_chr(o, "cell_border", "white")
  border_pt <- if (min(cw, ch) >= 6) 0.75 else if (min(cw, ch) >= 3) 0.35 else 0
  if (border_pt == 0) border <- "none"
  m <- aes(x = .data$x, y = .data$y, fill = .data$value)
  targs <- list(data = tiles, width = 1, height = 1, linejoin = "mitre")
  if (border == "none") { m$colour <- quote(.data$value); targs$linewidth <- lw(0.25) }   # hides hairline seams
  else {
    targs$colour <- switch(border, gray = if (ctx$dark) "#6B7280" else "#B5B5B5", black = ctx$ink, ctx$th$panel_bg)
    targs$linewidth <- lw(border_pt)
  }
  targs$mapping <- m
  layers <- list(do.call(geom_tile, targs))

  # ---- values
  lab <- tiles[nzchar(tiles$text), , drop = FALSE]
  if (nrow(lab)) {
    size_pt <- opt_num(o, "value_size", NA)
    if (is.na(size_pt)) {
      n_lines <- max(lengths(strsplit(lab$text, "\n", fixed = TRUE)))
      w10 <- max_text_w(ctx, lab$text, 10)
      size_pt <- min(f$tick, (cw - 3) / max(w10, 1) * 10, (ch - 2) / (n_lines * 1.15))
    }
    if (size_pt >= 3.5) {
      pal <- scales::gradient_n_pal(stops)
      shade <- pal(pmin(pmax((lab$value - vmin) / (vmax - vmin), 0), 1))
      lab$col <- contrast_col(shade)
      layers <- c(layers, list(geom_text(aes(x = .data$x, y = .data$y, label = .data$text), data = lab, colour = lab$col,
                                         size = ts(size_pt), family = f$family, lineheight = 0.9)))
    }
  }

  # ---- axis labels: thin them out when the cells are smaller than the text
  thin <- function(labels, cell_pt) {
    every <- max(1, ceiling(f$tick * 1.05 / max(cell_pt, 0.01)))
    i <- seq_along(labels)
    if (every > 1) labels[(i - 1) %% every != 0] <- ""
    labels
  }
  horizontal <- legend %in% c("top", "bottom")
  bar_len <- if (horizontal) min(P$w * 0.6, 220) else min(max(P$h * 0.55, 60), 200)
  guide <- if (!show_bar) "none" else guide_colourbar(
    title = if (nzchar(bar_title)) bar_title else NULL, title.position = "top",
    barwidth = grid::unit(if (horizontal) bar_len else f$legend * 1.1, "pt"),
    barheight = grid::unit(if (horizontal) f$legend * 1.1 else bar_len, "pt"),
    frame.colour = NA, ticks.colour = ctx$th$panel_bg, ticks.linewidth = lw(0.6), draw.ulim = TRUE, draw.llim = TRUE)
  fill_args <- list(colours = stops, limits = c(vmin, vmax), oob = scales::squish, na.value = ctx$th$panel_bg,
                    labels = fmt_num)
  xlim <- c(0.5, nc + 0.5); ylim <- c(0.5, nr + 0.5)
  p <- ggplot() + layers +
    do.call(scale_fill_gradientn, c(fill_args, list(guide = guide))) +
    do.call(scale_colour_gradientn, c(fill_args, list(guide = "none"))) +
    scale_x_continuous(breaks = seq_len(nc), labels = thin(cols, if (a == 0) cw * 0 + col_w * 0 + cw else cw), expand = c(0, 0)) +
    scale_y_continuous(breaks = seq_len(nr), labels = rev(thin(rows, ch)), expand = c(0, 0)) +
    (if (square) coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE) else coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE)) +
    gg_labs(ctx) + gg_theme(ctx, legend = legend, x_angle = angle) +
    theme(axis.line = element_blank(), axis.ticks = element_blank(), panel.border = element_blank(),
          panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
          axis.ticks.length = grid::unit(1.5, "pt"),
          legend.title = element_text(colour = ctx$th$text, size = f$legend, face = "bold"))
  p
}
register("heatmap", render_heatmap)
