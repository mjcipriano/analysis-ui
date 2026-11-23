# Parts of a whole and other simple bar charts: pie / donut, frequency bars,
# waterfall.

render_pie <- function(ctx) {
  o <- ctx$o; f <- ctx$f
  sl <- ctx$frames$slices
  if (!has_rows(sl)) aui_stop("There are no positive values to plot yet.")
  sl <- sl[is.finite(sl$value) & sl$value > 0, , drop = FALSE]
  if (!nrow(sl)) aui_stop("There are no positive values to plot yet.")
  sl$key <- as.character(sl$label)
  keys <- sl$key
  n <- nrow(sl)
  sl$text <- ifelse(is.na(sl$text), "", as.character(sl$text))
  hole <- min(max(opt_num(o, "hole", 0), 0), 0.95)
  outside <- opt_chr(o, "label_position", "inside") == "outside"
  # angles: 0 = 12 o'clock, clockwise (ggforce); the option is degrees counter-clockwise from 3 o'clock
  a0 <- (90 - opt_num(o, "start_angle", 90)) * pi / 180
  frac <- sl$value / sum(sl$value)
  sl$end <- a0 + cumsum(frac) * 2 * pi
  sl$start <- sl$end - frac * 2 * pi
  sl$mid <- (sl$start + sl$end) / 2
  cols <- group_cols(ctx, keys)

  names_in_labels <- opt_chr(o, "slice_labels", "percent") %in% c("label", "label_percent", "label_value")
  legend <- legend_position(ctx, if (names_in_labels) "none" else "right")
  P <- panel_size(ctx, left_w = 0, bottom_h = 0, legend = legend, legend_labels = keys,
                  has_x_title = FALSE, has_y_title = FALSE)
  P$w <- P$w + f$tick * 0.6; P$h <- P$h + f$tick * 0.6           # no axes: ticks and gaps are free space
  lab_pt <- f$legend
  has_text <- nzchar(sl$text)
  tw <- text_w(ctx, sl$text, lab_pt); tw[!has_text] <- 0
  n_lines <- lengths(strsplit(sl$text, "\n", fixed = TRUE)); n_lines[!has_text] <- 0
  th_pt <- n_lines * lab_pt * 1.15
  reach <- 1.12                                                    # outside labels start at this radius
  R <- min(P$w, P$h) / 2 * 0.97                                    # radius in points
  if (outside && any(has_text)) {
    R <- min((P$w / 2 - max(tw) - 6) / (reach + 0.06), (P$h / 2 - max(th_pt) / 2 - 2) / (reach - 0.02), R)
  }
  R <- max(R, 18)
  xlim <- c(-1, 1) * P$w / 2 / R; ylim <- c(-1, 1) * P$h / 2 / R   # data units: radius = 1

  gap_pt <- opt_num(o, "slice_gap", 1)
  sep <- if (ctx$th$transparent) "#FFFFFF" else ctx$th$bg
  arc_args <- list(mapping = aes(x0 = 0, y0 = 0, r0 = hole, r = 1, start = .data$start, end = .data$end, fill = .data$key),
                   data = sl, show.legend = TRUE, n = 360)
  if (n > 1 && gap_pt > 0) { arc_args$colour <- sep; arc_args$linewidth <- lw(gap_pt) }
  else { arc_args$mapping$colour <- quote(.data$key); arc_args$linewidth <- lw(0.3) }
  layers <- list(do.call(ggforce::geom_arc_bar, arc_args))

  if (any(has_text)) {
    r_in <- if (hole > 0.05) (hole + 1) / 2 else 0.62
    d <- sl[has_text, , drop = FALSE]
    d$tw <- tw[has_text]; d$th <- th_pt[has_text]
    # does the label fit inside its slice?
    arc_pt <- (d$end - d$start) * r_in * R
    fits <- arc_pt >= pmin(d$tw, d$th * 1.6) + 2 & (1 - hole) * R >= pmin(d$tw, d$th) + 3
    if (n == 1) fits <- TRUE
    inside <- d[!outside & fits, , drop = FALSE]
    out <- d[outside | !fits, , drop = FALSE]
    if (nrow(inside)) {
      inside$x <- r_in * sin(inside$mid); inside$y <- r_in * cos(inside$mid)
      if (n == 1 && hole <= 0.05) { inside$x <- 0; inside$y <- 0 }
      inside$col <- contrast_col(cols[inside$key])
      layers <- c(layers, list(geom_text(aes(x = .data$x, y = .data$y, label = .data$text), data = inside,
                                         colour = inside$col, size = ts(lab_pt), family = f$family, lineheight = 0.95)))
    }
    if (nrow(out)) {
      out$right <- sin(out$mid) >= 0
      out$ex <- sin(out$mid); out$ey <- cos(out$mid)                # point on the rim
      out$ly <- (reach - 0.02) * out$ey                             # label height, then de-overlapped per side
      for (side in c(TRUE, FALSE)) {
        i <- which(out$right == side)
        if (length(i) < 2) next
        i <- i[order(out$ly[i])]
        need <- (out$th[i] / R) * 0.5
        for (j in seq_along(i)[-1]) {
          min_y <- out$ly[i[j - 1]] + need[j - 1] + need[j] + 2 / R
          if (out$ly[i[j]] < min_y) out$ly[i[j]] <- min_y
        }
        over <- max(out$ly[i] + need) - (ylim[2] - 2 / R)          # shift back into the panel if pushed out
        if (over > 0) out$ly[i] <- out$ly[i] - over
      }
      dir <- ifelse(out$right, 1, -1)
      out$kx <- pmax(abs((reach - 0.04) * out$ex), sqrt(pmax(1.03^2 - pmin(abs(out$ly), 1.03)^2, 0))) * dir
      out$lx <- pmax(abs(out$kx), reach * abs(out$ex)) * dir + 0.06 * dir
      seg <- rbind(data.frame(x = 1.0 * out$ex, y = 1.0 * out$ey, xend = out$kx, yend = out$ly),
                   data.frame(x = out$kx, y = out$ly, xend = out$lx - 0.015 * dir, yend = out$ly))
      layers <- c(layers, list(
        geom_segment(aes(x = .data$x, y = .data$y, xend = .data$xend, yend = .data$yend), data = seg,
                     colour = ctx$muted, linewidth = lw(0.6), lineend = "round"),
        geom_text(aes(x = .data$lx + 0.02 * dir, y = .data$ly, label = .data$text), data = out,
                  hjust = ifelse(out$right, 0, 1), colour = ctx$th$text, size = ts(lab_pt), family = f$family,
                  lineheight = 0.95)))
    }
  }
  ggplot() + layers +
    colour_scales(ctx, keys) +
    scale_x_continuous(expand = c(0, 0)) + scale_y_continuous(expand = c(0, 0)) +
    coord_fixed(xlim = xlim, ylim = ylim, expand = FALSE, clip = "off") +
    gg_labs(ctx, x = "", y = "") + gg_theme(ctx, legend = legend, void = TRUE) +
    guides(colour = "none")
}
register("pie", render_pie)

# a minimal stand-in for cat_setup() for graphs that only need the axis part
cat_axis <- function(labels, groups = character(0)) list(xl = labels, gl = groups, n = length(labels))

render_count_bar <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed; f <- ctx$f
  d <- ctx$frames$counts
  if (!has_rows(d)) aui_stop("There are no values to count yet.")
  d <- keyed(d)
  if (!is.factor(d$x)) d$x <- factor(as.character(d$x), levels = unique(as.character(d$x)))
  xl <- levels(d$x); keys <- levels(d$group); k <- length(keys)
  has_group <- opt_flag(comp, "has_group")
  position <- opt_chr(comp, "position", "dodge")
  if (!has_group) position <- "stack"                      # one bar per category
  vcol <- opt_chr(comp, "value", "count")
  d$v <- d[[vcol]] %||% d$count
  d <- d[is.finite(d$v) & !is.na(d$x), , drop = FALSE]
  d$text <- ifelse(is.na(d$text), "", as.character(d$text))
  bw <- min(max(opt_num(o, "bar_width", 0.75), 0.02), 1)
  xi <- as.integer(d$x); gi <- as.integer(d$group)
  if (position == "dodge") {
    slot <- bw / k
    cx <- xi + (gi - (k + 1) / 2) * slot
    d$xmin <- cx - slot / 2; d$xmax <- cx + slot / 2; d$ymin <- 0; d$ymax <- d$v
  } else {
    ord <- order(xi, -gi)                                  # first group on top, like the legend
    d <- d[ord, , drop = FALSE]; xi <- xi[ord]
    topv <- stats::ave(d$v, xi, FUN = cumsum)
    d$ymax <- topv; d$ymin <- topv - d$v
    d$xmin <- xi - bw / 2; d$xmax <- xi + bw / 2
  }
  has_text <- any(nzchar(d$text))
  full <- position == "fill"
  ax <- if (full) axis_spec(ctx, "y", c(0, 100), fixed = c(0, 100))
        else axis_spec(ctx, "y", finite_range(d$ymax, 0), zero = TRUE, flush = TRUE,
                       mult = c(0, if (has_text && position == "dodge") 0.1 else 0.05))
  S <- cat_axis(xl, keys)
  legend <- legend_position(ctx, if (has_group && k > 1) "right" else "none")
  M <- cat_metrics(ctx, S, ax, legend)
  if (ax$is_log) d$ymin <- pmax(d$ymin, ax$lim[1])
  d <- d[d$ymax > d$ymin, , drop = FALSE]
  alpha <- opt_num(o, "fill_alpha", 1)
  outline <- opt_chr(o, "outline", "match"); ow <- opt_num(o, "outline_width", 1)
  layers <- list(outlined_geom(geom_rect, aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$ymax,
                                              fill = .data$key), d, outline, ow, ctx$ink, alpha = alpha,
                               show.legend = TRUE, linejoin = "mitre"))
  lab <- d[nzchar(d$text), , drop = FALSE]
  if (nrow(lab)) {
    size_pt <- f$tick * 0.92
    u <- diff(ax$tlim) / M$vlen                             # value units per point
    lab$x <- (lab$xmin + lab$xmax) / 2
    if (position == "dodge") {
      slot_pt <- (lab$xmax - lab$xmin) * M$unit_pt
      tw <- text_w(ctx, lab$text, size_pt)
      if (!ctx$flip && any(tw > slot_pt * 1.15)) size_pt <- max(size_pt * min(slot_pt * 1.15 / tw), 5)
      lab$y <- ax$inv(ax$tr(lab$ymax) + 3 * u)
      layers <- c(layers, list(geom_text(aes(x = .data$x, y = .data$y, label = .data$text), data = lab, size = ts(size_pt),
                                         colour = ctx$th$text, family = f$family, hjust = if (ctx$flip) 0 else 0.5,
                                         vjust = if (ctx$flip) 0.5 else 0)))
    } else {
      seg_pt <- (ax$tr(lab$ymax) - ax$tr(lab$ymin)) / u
      need <- if (ctx$flip) text_w(ctx, lab$text, size_pt) + 3 else size_pt * 1.05
      lab <- lab[seg_pt >= need, , drop = FALSE]
      if (nrow(lab)) {
        lab$y <- ax$inv((ax$tr(lab$ymax) + ax$tr(lab$ymin)) / 2)
        lab$col <- contrast_col(group_cols(ctx, lab$key), alpha, ctx$th$panel_bg)
        layers <- c(layers, list(geom_text(aes(x = .data$x, y = .data$y, label = .data$text), data = lab, size = ts(size_pt),
                                           colour = lab$col, family = f$family)))
      }
    }
  }
  cat_assemble(ctx, S, ax, M, layers, legend, keys)
}
register("count_bar", render_count_bar)

render_waterfall <- function(ctx) {
  o <- ctx$o; comp <- ctx$computed
  d <- ctx$frames$main
  if (!has_rows(d)) aui_stop("There are no values to plot yet.")
  d <- keyed(d)
  d <- d[is.finite(d$value), , drop = FALSE]
  if (!is.factor(d$label)) d$label <- factor(as.character(d$label), levels = unique(as.character(d$label)))
  d$label <- droplevels(d$label)
  keys <- levels(d$group)
  S <- cat_axis(levels(d$label), keys)
  ax <- axis_spec(ctx, "y", finite_range(d$value, 0, ctx$hlines), zero = TRUE, flush = TRUE)
  has_group <- opt_flag(comp, "has_group")
  legend <- legend_position(ctx, if (has_group && length(keys) > 1) "right" else "none")
  show_labels <- opt_flag(comp, "show_labels")
  if (!show_labels) S$xl <- rep("", S$n)
  M <- cat_metrics(ctx, S, ax, legend)
  bw <- min(max(opt_num(o, "bar_width", 0.85), 0.02), 1)
  xi <- as.integer(d$label)
  base <- if (ax$is_log) ax$lim[1] else 0
  d$xmin <- xi - bw / 2; d$xmax <- xi + bw / 2
  d$ymin <- pmin(base, d$value); d$ymax <- pmax(base, d$value)
  d <- d[on_scale(ax, d$value), , drop = FALSE]
  ow <- opt_num(o, "outline_width", 1)
  if (bw * M$unit_pt < 4) ow <- min(ow, 0.4)                 # thin bars: a heavy outline would swallow the fill
  layers <- list(outlined_geom(geom_rect, aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin, ymax = .data$ymax,
                                              fill = .data$key), d, opt_chr(o, "outline", "match"), ow, ctx$ink,
                               alpha = opt_num(o, "fill_alpha", 1), show.legend = TRUE, linejoin = "mitre"))
  if (!ax$is_log && ax$lim[1] < 0 && ax$lim[2] > 0) {       # zero line when bars go both ways
    layers <- c(layers, list(geom_hline(yintercept = 0, colour = ctx$th$axis, linewidth = lw(max(ctx$th$axis_width * 0.75, 0.6)))))
  }
  extra <- if (!show_labels) theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
                                   axis.ticks.length.x = grid::unit(0, "pt"))
  cat_assemble(ctx, S, ax, M, layers, legend, keys, show_cat_labels = show_labels, theme_extra = extra)
}
register("waterfall", render_waterfall)
