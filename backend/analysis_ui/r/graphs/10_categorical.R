# Column and grouped graphs: one numeric value axis, one categorical axis.
#
# Both families share one model. A *cell* is one position on the categorical
# axis: a group (column graphs) or a subgroup inside a category (grouped
# graphs, dodged). Every drawing primitive works on cells, so e.g. `box` and
# `grouped_box` are the same code.
#
# The categorical axis is a continuous scale with category i at x = i
# (reversed when flipped so the first category is at the top).

CAT_PAD <- 0.6      # free space outside the first / last category (category units)

cat_setup <- function(ctx, grouped) {
  main <- ctx$frames$main
  summ <- ctx$frames$summary
  if (!has_rows(main)) aui_stop("There are no numeric values to plot yet.")
  if (!has_rows(summ)) summ <- data.frame(group = main$group[0], n = numeric(0), center = numeric(0),
                                          lo = numeric(0), hi = numeric(0))
  fac <- function(v, levels = NULL) if (is.factor(v)) droplevels(v) else factor(as.character(v), levels = levels %||% unique(as.character(v)))
  main$group <- if (is.factor(main$group)) main$group else fac(main$group)
  gl <- levels(main$group)
  summ$group <- factor(as.character(summ$group), levels = gl)
  if (grouped) {
    main$x <- if (is.factor(main$x)) main$x else fac(main$x)
    xl <- levels(main$x)
    summ$x <- factor(as.character(summ$x), levels = xl)
    k <- length(gl)
    total <- min(max(opt_num(ctx$o, "dodge", 0.8), 0), 1)
    slot <- total / k
    pos <- function(x, g) as.integer(x) + (as.integer(g) - (k + 1) / 2) * slot
  } else {
    xl <- gl; k <- 1; slot <- 1
    main$x <- main$group; summ$x <- summ$group
    pos <- function(x, g) as.numeric(as.integer(g))
  }
  cell_id <- function(x, g) paste(as.integer(x), as.integer(g), sep = ":")
  main <- main[is.finite(main$value) & !is.na(main$group) & !is.na(main$x), , drop = FALSE]
  summ <- summ[!is.na(summ$group) & !is.na(summ$x), , drop = FALSE]
  main$cx <- pos(main$x, main$group); main$cell <- cell_id(main$x, main$group)
  summ$cx <- pos(summ$x, summ$group); summ$cell <- cell_id(summ$x, summ$group)
  main$key <- as.character(main$group); summ$key <- as.character(summ$group)
  tf <- axis_trans(opt_chr(ctx$o, "y_scale", "linear"))
  list(main = main, summ = summ, xl = xl, gl = gl, k = k, slot = slot, n = length(xl),
       grouped = grouped, pos = pos, cell_id = cell_id, tr = tf$tr, inv = tf$inv,
       is_log = tf$name %in% c("log10", "log2"))
}

# per-cell split of the raw values: list(cell = list(v, cx, key, x, group))
cat_cells <- function(S) {
  lapply(split(S$main, factor(S$main$cell, levels = unique(S$main$cell))), function(d) {
    list(v = d$value, cx = d$cx[1], key = d$key[1], cell = d$cell[1], x = d$x[1], group = d$group[1])
  })
}

# Panel metrics in points (see panel_size) + the resolved tick-label angle.
cat_metrics <- function(ctx, S, ax, legend, key_w = 1) {
  f <- ctx$f
  val_w <- max_text_w(ctx, axis_breaks(ctx, ax, minor_ok = FALSE)$labels, f$tick)
  cat_w <- max_text_w(ctx, S$xl, f$tick)
  n_lines <- max(1, lengths(strsplit(S$xl, "\n", fixed = TRUE)))
  span <- S$n - 1 + 2 * CAT_PAD
  choice <- opt_chr(ctx$o, "x_tick_angle", "auto")
  if (ctx$flip) {
    ps <- panel_size(ctx, left_w = cat_w, bottom_h = f$tick * 1.1, legend = legend, legend_labels = S$gl, key_w = key_w)
    angle <- if (choice == "auto") NULL else tick_angle(choice)
  } else {
    ps0 <- panel_size(ctx, left_w = val_w, legend = legend, legend_labels = S$gl, key_w = key_w)
    slot_pt <- ps0$w / span
    if (choice == "auto") {
      angle <- if (cat_w <= slot_pt * 0.94) NULL
               else if (f$tick * 1.6 > slot_pt) tick_angle(90) else tick_angle(45)
    } else {
      angle <- tick_angle(choice)
    }
    a <- if (is.null(angle)) 0 else angle$angle
    bottom_h <- if (a == 0) f$tick * 1.15 * n_lines else cat_w * sin(a * pi / 180) + f$tick * cos(a * pi / 180)
    ps <- panel_size(ctx, left_w = val_w, bottom_h = bottom_h, legend = legend, legend_labels = S$gl, key_w = key_w)
  }
  vlen <- if (ctx$flip) ps$w else ps$h
  clen <- if (ctx$flip) ps$h else ps$w
  list(w = ps$w, h = ps$h, vlen = vlen, clen = clen, unit_pt = clen / span, angle = angle)
}

# points spread around their cell centre
cat_point_layer <- function(ctx, S, ax, M, spec, layout, half, df = S$main, show.legend = FALSE) {
  df <- df[on_scale(ax, df$value), , drop = FALSE]
  if (!nrow(df)) return(list())
  dx <- spec$d / M$unit_pt
  dy <- spec$d / M$vlen * diff(ax$tlim)
  off <- spread_points(ax$tr(df$value), df$cell, layout, half, dx, dy)
  d <- data.frame(x = df$cx + off, y = df$value, key = spec$key(df$group))
  list(marker_layer(spec, d, show.legend = show.legend))
}

# geom with the outline either mapped to the group ("match") or constant
outlined_geom <- function(.geom, .mapping, .data, .outline, .width_pt, .ink, ...) {
  args <- list(data = .data, ...)
  if (.width_pt <= 0) .outline <- "none"
  if (.outline == "match") .mapping$colour <- quote(.data$key)
  else args$colour <- if (.outline == "none") NA else .ink
  args$linewidth <- lw(max(.width_pt, 0.01))
  args$mapping <- .mapping
  do.call(.geom, args)
}

summary_frame <- function(S, ax) {
  s <- S$summ
  if (!has_rows(s)) return(data.frame(x = numeric(0), center = numeric(0), lo = numeric(0), hi = numeric(0),
                                      key = character(0), cell = character(0)))
  s <- s[on_scale(ax, s$center), , drop = FALSE]
  bad <- function(v) !is.na(v) & !on_scale(ax, v)
  s$lo[bad(s$lo)] <- NA; s$hi[bad(s$hi)] <- NA
  data.frame(x = s$cx, center = s$center, lo = s$lo, hi = s$hi, key = s$key, cell = s$cell,
             cat = as.integer(s$x), group = s$group)
}

quartile_stats <- function(v, whiskers = "tukey") {
  q <- stats::quantile(v, c(0.25, 0.5, 0.75), names = FALSE, type = 7)
  iqr <- q[3] - q[1]
  w <- switch(whiskers,
              minmax = range(v),
              p5_95 = stats::quantile(v, c(0.05, 0.95), names = FALSE),
              p10_90 = stats::quantile(v, c(0.10, 0.90), names = FALSE),
              c(min(v[v >= q[1] - 1.5 * iqr]), max(v[v <= q[3] + 1.5 * iqr])))
  w <- c(min(w[1], q[1]), max(w[2], q[3]))
  list(q1 = q[1], med = q[2], q3 = q[3], lo = w[1], hi = w[2], n = length(v),
       out = v[v < w[1] | v > w[2]])
}

# kernel density of one cell on the axis scale; NULL when it cannot be estimated
cell_density <- function(S, v, adjust, trim) {
  tv <- S$tr(v)
  tv <- tv[is.finite(tv)]
  if (length(tv) < 2 || diff(range(tv)) == 0) return(NULL)
  d <- tryCatch(stats::density(tv, adjust = adjust, n = 160, cut = if (trim) 0 else 3), error = function(e) NULL)
  if (is.null(d)) return(NULL)
  list(t = d$x, d = d$y, n = length(tv))
}

# ============================================================== kinds
# Each kind returns list(range, tops (named by cell, data units), zero, flush,
# layers = function(ax, M) -> list of layers, legend_keys = TRUE/FALSE).

kind_scatter <- function(ctx, S) {
  o <- ctx$o
  show_summary <- opt_flag(o, "show_summary", TRUE)
  s <- S$summ
  tops <- tapply(S$main$value, S$main$cell, max)
  if (show_summary && nrow(s)) {
    st <- tapply(pmax(s$hi, s$center, na.rm = TRUE), s$cell, max)
    tops <- tapply(c(tops, st), c(names(tops), names(st)), max, na.rm = TRUE)
  }
  list(range = finite_range(S$main$value, if (show_summary) c(s$lo, s$hi, s$center)), tops = tops,
       layers = function(ax, M) {
         spec <- marker_spec(ctx)
         layout <- opt_chr(o, "point_layout", "swarm")
         half <- min(opt_num(o, "jitter", 0.18), 0.48)
         if (S$grouped) half <- min(half, S$slot * 0.46)
         out <- cat_point_layer(ctx, S, ax, M, spec, layout, half, show.legend = S$grouped)
         if (show_summary) {
           sf <- summary_frame(S, ax)
           ew <- opt_num(o, "error_width", 1)
           sw <- opt_num(o, "summary_width", 0.5) * S$slot
           out <- c(out, errorbar_layers(sf, opt_num(o, "error_cap", 0.25) * S$slot, ew, colour = ctx$ink))
           if (nrow(sf)) out <- c(out, list(geom_segment(
             aes(x = .data$x - sw / 2, xend = .data$x + sw / 2, y = .data$center, yend = .data$center),
             data = sf, colour = ctx$ink, linewidth = lw(ew * 1.5), lineend = "butt", show.legend = FALSE)))
         }
         out
       })
}

kind_bar <- function(ctx, S) {
  o <- ctx$o
  stacked <- S$grouped && opt_chr(o, "position", "dodge") == "stack"
  show_points <- opt_flag(o, "show_points", TRUE) && !stacked
  s <- S$summ
  bw <- min(max(opt_num(o, "bar_width", 0.65), 0.02), 1) * S$slot
  alpha <- opt_num(o, "fill_alpha", 1)
  outline <- opt_chr(o, "outline", "match"); ow <- opt_num(o, "outline_width", 1)
  if (stacked) {
    s <- s[is.finite(s$center), , drop = FALSE]
    s <- s[order(as.integer(s$x), -as.integer(s$group)), , drop = FALSE]      # first group ends on top
    s$y0 <- 0; s$y1 <- 0
    for (xi in unique(as.integer(s$x))) {
      i <- which(as.integer(s$x) == xi)
      pos <- s$center[i] >= 0
      up <- cumsum(ifelse(pos, s$center[i], 0)); dn <- cumsum(ifelse(!pos, s$center[i], 0))
      s$y1[i] <- ifelse(pos, up, dn); s$y0[i] <- s$y1[i] - s$center[i]
    }
    width <- min(max(opt_num(o, "bar_width", 0.9), 0.02), 1) * min(max(opt_num(o, "dodge", 0.8), 0.2), 1)
    tops <- tapply(s$y1, as.integer(s$x), max)
    return(list(range = finite_range(s$y0, s$y1, 0), tops = NULL, zone_tops = tops, zero = TRUE, flush = TRUE,
                layers = function(ax, M) {
                  d <- data.frame(xmin = as.integer(s$x) - width / 2, xmax = as.integer(s$x) + width / 2,
                                  ymin = s$y0, ymax = s$y1, key = s$key)
                  if (S$is_log) { d$ymin <- pmax(d$ymin, ax$lim[1]); d <- d[d$ymax > d$ymin, , drop = FALSE] }
                  list(outlined_geom(geom_rect, aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin,
                                                    ymax = .data$ymax, fill = .data$key),
                                     d, outline, ow, ctx$ink, alpha = alpha,
                                     show.legend = TRUE, linejoin = "mitre"))
                }))
  }
  vals <- c(s$center, s$lo, s$hi, if (show_points) S$main$value)
  tops <- tapply(pmax(s$hi, s$center, na.rm = TRUE), s$cell, max)
  if (show_points) {
    pt <- tapply(S$main$value, S$main$cell, max)
    tops <- tapply(c(tops, pt), c(names(tops), names(pt)), max, na.rm = TRUE)
  }
  tops[!is.finite(tops) | tops < 0] <- 0
  list(range = finite_range(vals, 0), tops = tops, zero = TRUE, flush = TRUE,
       layers = function(ax, M) {
         sf <- summary_frame(S, ax)
         base <- if (S$is_log) ax$lim[1] else 0
         d <- data.frame(xmin = sf$x - bw / 2, xmax = sf$x + bw / 2, ymin = base, ymax = sf$center, key = sf$key)
         out <- list(outlined_geom(geom_rect, aes(xmin = .data$xmin, xmax = .data$xmax, ymin = .data$ymin,
                                                  ymax = .data$ymax, fill = .data$key),
                                   d, outline, ow, ctx$ink, alpha = alpha, show.legend = TRUE, linejoin = "mitre"))
         err <- errorbar_layers(sf, opt_num(o, "error_cap", 0.25) * S$slot, opt_num(o, "error_width", 1),
                                colour = ctx$ink, direction = opt_chr(o, "error_direction", "both"))
         pts <- list()
         if (show_points) {
           spec <- marker_spec(ctx, outlined = TRUE)
           layout <- opt_chr(o, "point_layout", "swarm")
           half <- if (S$grouped) bw * 0.36 else min(opt_num(o, "jitter", 0.18), bw * 0.48)
           pts <- cat_point_layer(ctx, S, ax, M, spec, layout, half)
         }
         c(out, pts, err)
       })
}

kind_box <- function(ctx, S) {
  o <- ctx$o
  cells <- cat_cells(S)
  mode <- opt_chr(o, "whiskers", "tukey")
  show_points <- opt_flag(o, "show_points", FALSE)
  show_out <- opt_flag(o, "show_outliers", TRUE) && !show_points
  st <- lapply(cells, function(cl) c(cl, quartile_stats(cl$v, mode), mean = mean(cl$v)))
  tops <- vapply(st, function(b) if (show_points || show_out) max(b$v) else b$hi, numeric(1))
  bots <- vapply(st, function(b) if (show_points || show_out) min(b$v) else b$lo, numeric(1))
  bw <- min(max(opt_num(o, "bar_width", 0.6), 0.02), 1) * S$slot
  outline <- opt_chr(o, "outline", "match"); ow <- opt_num(o, "outline_width", 1)
  if (ow <= 0) outline <- "none"
  list(range = finite_range(tops, bots), tops = tops,
       layers = function(ax, M) {
         num <- function(k) vapply(st, function(b) as.numeric(b[[k]]), numeric(1))
         d <- data.frame(x = num("cx"), cell = names(st), key = vapply(st, function(b) b$key, ""),
                         q1 = num("q1"), med = num("med"), q3 = num("q3"), lo = num("lo"), hi = num("hi"),
                         n = num("n"), mean = num("mean"))
         d <- d[on_scale(ax, d$q1) & on_scale(ax, d$lo), , drop = FALSE]
         if (!nrow(d)) return(list())
         line_pt <- if (outline == "none") 1 else ow
         wcol <- if (outline == "match") NULL else ctx$ink
         cap <- bw * 0.4
         wseg <- rbind(data.frame(x = d$x, xend = d$x, y = d$q3, yend = d$hi, key = d$key),
                       data.frame(x = d$x, xend = d$x, y = d$q1, yend = d$lo, key = d$key),
                       data.frame(x = d$x - cap / 2, xend = d$x + cap / 2, y = d$hi, yend = d$hi, key = d$key),
                       data.frame(x = d$x - cap / 2, xend = d$x + cap / 2, y = d$lo, yend = d$lo, key = d$key))
         wm <- aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend)
         whisk <- if (is.null(wcol)) {
           wm$colour <- quote(.data$key)
           geom_segment(wm, data = wseg, linewidth = lw(line_pt), show.legend = FALSE)
         } else geom_segment(wm, data = wseg, linewidth = lw(line_pt), colour = wcol, show.legend = FALSE)
         notch <- opt_flag(o, "notch")
         m <- aes(x = .data$x, group = .data$cell, ymin = .data$q1, lower = .data$q1, middle = .data$med,
                  upper = .data$q3, ymax = .data$q3, fill = .data$key)
         if (notch) {
           half <- 1.58 * (d$q3 - d$q1) / sqrt(d$n)
           d$nlo <- pmax(d$med - half, d$q1); d$nhi <- pmin(d$med + half, d$q3)
           m$notchlower <- quote(.data$nlo); m$notchupper <- quote(.data$nhi)
         }
         box <- outlined_geom(geom_boxplot, m, d, outline, ow, ctx$ink, stat = "identity", width = bw,
                              alpha = opt_num(o, "fill_alpha", 0.6), notch = notch, fatten = 1.8,
                              show.legend = TRUE, key_glyph = "rect")
         out <- list(whisk, box)
         if (outline == "none") {
           hw <- bw / 2 * (if (notch) 0.5 else 1)
           out <- c(out, list(geom_segment(aes(x = .data$x - hw, xend = .data$x + hw, y = .data$med, yend = .data$med),
                                           data = d, colour = ctx$ink, linewidth = lw(1.5), show.legend = FALSE)))
         }
         if (show_out) {
           od <- do.call(rbind, lapply(st, function(b) if (length(b$out)) data.frame(x = b$cx, y = b$out, key = b$key)))
           if (has_rows(od)) {
             od <- od[on_scale(ax, od$y), , drop = FALSE]
             dia <- max(2.5, opt_num(o, "point_size", 4) * 0.8)
             args <- list(mapping = aes(x = .data$x, y = .data$y), data = od, shape = 1, stroke = 0.8 / 1.417,
                          size = pt_size(dia, 1, 0.8 / 1.417), show.legend = FALSE)
             if (outline == "match") args$mapping$colour <- quote(.data$key) else args$colour <- ctx$ink
             out <- c(out, list(do.call(geom_point, args)))
           }
         }
         if (show_points) {
           spec <- marker_spec(ctx, outlined = TRUE)
           half <- if (S$grouped) bw * 0.36 else min(opt_num(o, "jitter", 0.18), bw * 0.48)
           out <- c(out, cat_point_layer(ctx, S, ax, M, spec, opt_chr(o, "point_layout", "swarm"), half))
         }
         if (opt_flag(o, "show_mean")) {
           md <- d[on_scale(ax, d$mean), , drop = FALSE]
           out <- c(out, list(geom_point(aes(x = .data$x, y = .data$mean), data = md, shape = 3, colour = ctx$ink,
                                         size = pt_size(max(5, bw * M$unit_pt * 0.22), 1), stroke = 1.1 / 1.417,
                                         show.legend = FALSE)))
         }
         out
       })
}

# polygons for (half) violins. side: 0 = both, 1 = towards larger x, -1 = smaller x.
violin_polys <- function(S, dens, centres, half, keys, side = 0, offset = 0) {
  do.call(rbind, lapply(names(dens), function(id) {
    d <- dens[[id]]
    if (is.null(d)) return(NULL)
    w <- d$w * half
    y <- S$inv(d$t)
    cx <- centres[[id]]
    if (side == 0) data.frame(x = c(cx - w, rev(cx + w)), y = c(y, rev(y)), cell = id, key = keys[[id]])
    else data.frame(x = c(cx + side * (offset + w), rep(cx + side * offset, length(y))), y = c(y, rev(y)),
                    cell = id, key = keys[[id]])
  }))
}

# densities of all cells with widths scaled 0..1 according to `scale`
scaled_densities <- function(S, cells, adjust, trim, scale = "width") {
  dens <- lapply(cells, function(cl) cell_density(S, cl$v, adjust, trim))
  peak <- vapply(dens, function(d) if (is.null(d)) NA_real_ else max(d$d), numeric(1))
  npk <- vapply(dens, function(d) if (is.null(d)) NA_real_ else max(d$d) * d$n, numeric(1))
  for (id in names(dens)) {
    d <- dens[[id]]
    if (is.null(d)) next
    dens[[id]]$w <- switch(scale,
                           area = d$d / max(peak, na.rm = TRUE),
                           count = d$d * d$n / max(npk, na.rm = TRUE),
                           d$d / max(d$d))
  }
  dens
}

kind_violin <- function(ctx, S) {
  o <- ctx$o
  cells <- cat_cells(S)
  trim <- opt_flag(o, "violin_trim", TRUE)
  dens <- scaled_densities(S, cells, opt_num(o, "bw_adjust", 1), trim, opt_chr(o, "violin_scale", "width"))
  ext <- lapply(names(cells), function(id) range(c(cells[[id]]$v, if (!is.null(dens[[id]])) S$inv(range(dens[[id]]$t)))))
  names(ext) <- names(cells)
  tops <- vapply(ext, function(r) r[2], numeric(1))
  bw <- min(max(opt_num(o, "bar_width", 0.8), 0.02), 1) * S$slot
  outline <- opt_chr(o, "outline", "match"); ow <- opt_num(o, "outline_width", 1)
  inner <- opt_chr(o, "violin_inner", "box")
  show_points <- opt_flag(o, "show_points", FALSE)
  centres <- lapply(cells, function(cl) cl$cx); keys <- lapply(cells, function(cl) cl$key)
  list(range = finite_range(unlist(ext)), tops = tops,
       layers = function(ax, M) {
         out <- list()
         poly <- violin_polys(S, dens, centres, bw / 2, keys)
         if (has_rows(poly)) {
           out <- c(out, list(outlined_geom(geom_polygon, aes(x = .data$x, y = .data$y, group = .data$cell, fill = .data$key),
                                            poly, outline, ow, ctx$ink, alpha = opt_num(o, "fill_alpha", 0.6),
                                            show.legend = TRUE, key_glyph = "rect")))
         }
         # cells without a density (n < 2 or constant): a short line at the value
         flat <- names(dens)[vapply(dens, is.null, logical(1))]
         if (length(flat)) {
           fd <- do.call(rbind, lapply(flat, function(id) data.frame(x = centres[[id]], y = stats::median(cells[[id]]$v), key = keys[[id]])))
           fd <- fd[on_scale(ax, fd$y), , drop = FALSE]
           if (nrow(fd)) out <- c(out, list(geom_segment(aes(x = .data$x - bw * 0.25, xend = .data$x + bw * 0.25, y = .data$y,
                                                             yend = .data$y, colour = .data$key), data = fd,
                                                         linewidth = lw(1.5), show.legend = FALSE)))
         }
         if (show_points) {
           spec <- marker_spec(ctx, outlined = TRUE)
           half <- if (S$grouped) bw * 0.3 else min(opt_num(o, "jitter", 0.18), bw * 0.45)
           out <- c(out, cat_point_layer(ctx, S, ax, M, spec, opt_chr(o, "point_layout", "swarm"), half))
         }
         has_d <- names(dens)[!vapply(dens, is.null, logical(1))]
         if (inner == "box" && length(has_d)) {
           q <- do.call(rbind, lapply(has_d, function(id) {
             b <- quartile_stats(cells[[id]]$v, "tukey")
             data.frame(x = centres[[id]], q1 = b$q1, med = b$med, q3 = b$q3, lo = b$lo, hi = b$hi)
           }))
           q <- q[on_scale(ax, q$lo), , drop = FALSE]
           bx <- min(bw * 0.07, 5 / M$unit_pt)
           box_col <- if (ctx$dark) "#E5E7EB" else "#222222"
           out <- c(out, list(
             geom_segment(aes(x = .data$x, xend = .data$x, y = .data$lo, yend = .data$hi), data = q, colour = box_col,
                          linewidth = lw(1), show.legend = FALSE),
             geom_rect(aes(xmin = .data$x - bx, xmax = .data$x + bx, ymin = .data$q1, ymax = .data$q3), data = q,
                       fill = box_col, colour = NA, show.legend = FALSE),
             geom_point(aes(x = .data$x, y = .data$med), data = q, shape = 21, fill = "white", colour = box_col,
                        size = pt_size(min(max(2 * bx * M$unit_pt * 0.9, 3), 6), 21, 0.4), stroke = 0.4, show.legend = FALSE)))
         } else if (inner == "quartiles" && length(has_d)) {
           q <- do.call(rbind, lapply(has_d, function(id) {
             d <- dens[[id]]
             qs <- stats::quantile(cells[[id]]$v, c(0.25, 0.5, 0.75), names = FALSE)
             hw <- stats::approx(d$t, d$w, xout = S$tr(qs), rule = 2)$y * bw / 2
             data.frame(x = centres[[id]] - hw, xend = centres[[id]] + hw, y = qs, which = c("q", "m", "q"))
           }))
           q <- q[on_scale(ax, q$y), , drop = FALSE]
           qcol <- if (outline == "match") ctx$ink else ctx$ink
           out <- c(out, list(
             geom_segment(aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$y), data = q[q$which == "q", ],
                          colour = qcol, linewidth = lw(0.8), linetype = "22", show.legend = FALSE),
             geom_segment(aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$y), data = q[q$which == "m", ],
                          colour = qcol, linewidth = lw(1.3), show.legend = FALSE)))
         } else if (inner == "mean_sd") {
           q <- do.call(rbind, lapply(names(cells), function(id) {
             v <- cells[[id]]$v
             s <- if (length(v) > 1) stats::sd(v) else NA
             data.frame(x = centres[[id]], center = mean(v), lo = mean(v) - s, hi = mean(v) + s)
           }))
           q <- q[on_scale(ax, q$center), , drop = FALSE]
           q$lo[!on_scale(ax, q$lo)] <- NA
           out <- c(out, errorbar_layers(q, 0, 1.3, colour = ctx$ink),
                    list(geom_point(aes(x = .data$x, y = .data$center), data = q, shape = 21, fill = "white",
                                    colour = ctx$ink, size = pt_size(5.5, 21, 0.6), stroke = 0.6, show.legend = FALSE)))
         }
         out
       })
}

kind_raincloud <- function(ctx, S) {
  o <- ctx$o
  cells <- cat_cells(S)
  dens <- scaled_densities(S, cells, opt_num(o, "bw_adjust", 1), trim = FALSE, "width")
  # tails of an untrimmed density carry no information far from the data: cut at 1 % of the peak
  dens <- lapply(dens, function(d) {
    if (is.null(d)) return(NULL)
    keep <- range(which(d$w >= 0.01))
    i <- seq(keep[1], keep[2])
    list(t = d$t[i], d = d$d[i], w = d$w[i], n = d$n)
  })
  ext <- lapply(names(cells), function(id) range(c(cells[[id]]$v, if (!is.null(dens[[id]])) S$inv(range(dens[[id]]$t)))))
  tops <- stats::setNames(vapply(ext, function(r) r[2], numeric(1)), names(cells))
  side <- if (ctx$flip) -1 else 1                 # cloud on the right, or on top when horizontal
  shift <- -side * 0.1                            # centre the cloud + rain around the tick
  centres <- lapply(cells, function(cl) cl$cx + shift); keys <- lapply(cells, function(cl) cl$key)
  list(range = finite_range(unlist(ext)), tops = tops,
       layers = function(ax, M) {
         out <- list()
         poly <- violin_polys(S, dens, centres, 0.42, keys, side = side, offset = 0.07)
         if (has_rows(poly)) {
           out <- c(out, list(geom_polygon(aes(x = .data$x, y = .data$y, group = .data$cell, fill = .data$key,
                                               colour = .data$key), data = poly, alpha = opt_num(o, "fill_alpha", 0.6),
                                           linewidth = lw(0.9), show.legend = TRUE, key_glyph = "rect")))
         }
         spec <- marker_spec(ctx)
         df <- S$main[on_scale(ax, S$main$value), , drop = FALSE]
         if (nrow(df)) {
           jit <- with_seed(20240229L, stats::runif(nrow(df), -0.09, 0.09))
           pd <- data.frame(x = df$cx + shift - side * 0.19 + jit, y = df$value, key = spec$key(df$group))
           out <- c(out, list(marker_layer(spec, pd)))
         }
         if (opt_flag(o, "show_box", TRUE)) {
           q <- do.call(rbind, lapply(names(cells), function(id) {
             b <- quartile_stats(cells[[id]]$v, "tukey")
             data.frame(x = centres[[id]], q1 = b$q1, med = b$med, q3 = b$q3, lo = b$lo, hi = b$hi, key = keys[[id]])
           }))
           q <- q[on_scale(ax, q$lo), , drop = FALSE]
           bx <- 0.045
           out <- c(out, list(
             geom_segment(aes(x = .data$x, xend = .data$x, y = .data$lo, yend = .data$hi), data = q, colour = ctx$ink,
                          linewidth = lw(1), show.legend = FALSE),
             geom_rect(aes(xmin = .data$x - bx, xmax = .data$x + bx, ymin = .data$q1, ymax = .data$q3, fill = .data$key),
                       data = q, colour = ctx$ink, linewidth = lw(1), show.legend = FALSE),
             geom_segment(aes(x = .data$x - bx, xend = .data$x + bx, y = .data$med, yend = .data$med), data = q,
                          colour = ctx$ink, linewidth = lw(1.6), lineend = "butt", show.legend = FALSE)))
         }
         out
       })
}

kind_before_after <- function(ctx, S) {
  o <- ctx$o
  show_summary <- opt_flag(o, "show_summary", FALSE)
  s <- S$summ
  tops <- tapply(S$main$value, S$main$cell, max)
  list(range = finite_range(S$main$value, if (show_summary) c(s$lo, s$hi)), tops = tops,
       layers = function(ax, M) {
         df <- S$main[on_scale(ax, S$main$value), , drop = FALSE]
         out <- list()
         if (!is.null(df$subject) && nrow(df)) {
           df <- df[order(df$subject, df$cx), , drop = FALSE]
           mode <- opt_chr(o, "line_color", "gray")
           lwd <- lw(opt_num(o, "line_width", 1)); la <- opt_num(o, "line_alpha", 0.6)
           ld <- df[!is.na(df$subject), , drop = FALSE]
           if (mode == "direction" && nrow(ld)) {
             delta <- tapply(seq_len(nrow(ld)), ld$subject, function(i) ld$value[i[length(i)]] - ld$value[i[1]])
             dl <- delta[as.character(ld$subject)]
             ld$dir <- ifelse(is.na(dl) | dl == 0, "flat", ifelse(dl > 0, "up", "down"))
             for (d in c("flat", "down", "up")) {
               sub <- ld[ld$dir == d, , drop = FALSE]
               if (nrow(sub)) out <- c(out, list(geom_line(
                 aes(x = .data$cx, y = .data$value, group = .data$subject), data = sub, linewidth = lwd, alpha = la,
                 colour = switch(d, up = "#D7402B", down = "#2C6FBB", ctx$muted), show.legend = FALSE)))
             }
           } else if (nrow(ld)) {
             out <- c(out, list(geom_line(aes(x = .data$cx, y = .data$value, group = .data$subject), data = ld,
                                          linewidth = lwd, alpha = la,
                                          colour = if (mode == "black") ctx$ink else ctx$muted, show.legend = FALSE)))
           }
         }
         spec <- marker_spec(ctx)
         if (nrow(df)) out <- c(out, list(marker_layer(spec, data.frame(x = df$cx, y = df$value, key = spec$key(df$group)))))
         if (show_summary) {
           sf <- summary_frame(S, ax)
           ew <- opt_num(o, "error_width", 1)
           out <- c(out, errorbar_layers(sf, opt_num(o, "error_cap", 0.25) * 0.6, ew * 1.3, colour = ctx$ink))
           if (nrow(sf)) out <- c(out, list(geom_segment(
             aes(x = .data$x - 0.2, xend = .data$x + 0.2, y = .data$center, yend = .data$center), data = sf,
             colour = ctx$ink, linewidth = lw(ew * 2), show.legend = FALSE)))
         }
         out
       })
}

kind_interaction <- function(ctx, S) {
  o <- ctx$o
  s <- S$summ
  show_points <- opt_flag(o, "show_points", FALSE)
  vals <- c(s$center, s$lo, s$hi, if (show_points) S$main$value)
  list(range = finite_range(vals), tops = NULL, key_w = 2.2,
       layers = function(ax, M) {
         sf <- summary_frame(S, ax)
         out <- list()
         spec <- marker_spec(ctx)
         if (show_points) {
           raw <- marker_spec(ctx, size = max(spec$d * 0.55, 2), alpha = min(spec$alpha, 0.4))
           out <- c(out, cat_point_layer(ctx, S, ax, M, raw, "jitter", max(S$slot * 0.3, 0.04)))
         }
         if (nrow(sf)) {
           sf <- sf[order(sf$group, sf$x), , drop = FALSE]
           out <- c(out, list(geom_line(aes(x = .data$x, y = .data$center, group = .data$key, colour = .data$key),
                                        data = sf, linewidth = lw(opt_num(o, "line_width", 1.5)), show.legend = TRUE)))
           out <- c(out, errorbar_layers(sf, opt_num(o, "error_cap", 0.25) * 0.3, opt_num(o, "error_width", 1)))
           out <- c(out, list(marker_layer(spec, data.frame(x = sf$x, y = sf$center, key = spec$key(sf$group)),
                                           show.legend = TRUE)))
         }
         out
       })
}

# ============================================================== assembly
render_categorical <- function(ctx, kind_fn, grouped) {
  o <- ctx$o
  S <- cat_setup(ctx, grouped)
  K <- kind_fn(ctx, S)
  ax <- axis_spec(ctx, "y", K$range, zero = isTRUE(K$zero), flush = isTRUE(K$flush))
  legend <- legend_position(ctx, if (grouped && S$k > 1) "right" else "none")
  key_w <- K$key_w %||% 1
  M <- cat_metrics(ctx, S, ax, legend, key_w)

  # significance brackets
  bl <- list(ax = ax, layers = list())
  br <- records(ctx$computed$brackets, c("x", "g1", "g2", "label"))
  if (nrow(br) && !is.null(K$tops)) {
    gi <- function(g) match(g, S$gl)
    if (grouped) {
      xi <- match(br$x, S$xl)
      ok <- !is.na(xi) & !is.na(gi(br$g1)) & !is.na(gi(br$g2))
      br <- br[ok, , drop = FALSE]; xi <- xi[ok]
      x1 <- xi + (gi(br$g1) - (S$k + 1) / 2) * S$slot
      x2 <- xi + (gi(br$g2) - (S$k + 1) / 2) * S$slot
      cell_x <- as.integer(sub(":.*$", "", names(K$tops)))
      zone_top <- tapply(K$tops, cell_x, max, na.rm = TRUE)
      top <- as.numeric(zone_top[as.character(xi)])
      zone <- xi
    } else {
      ok <- !is.na(gi(br$g1)) & !is.na(gi(br$g2))
      br <- br[ok, , drop = FALSE]
      x1 <- gi(br$g1); x2 <- gi(br$g2)
      top <- rep(max(K$tops, na.rm = TRUE), nrow(br))
      zone <- rep(1L, nrow(br))
    }
    if (nrow(br)) {
      top <- pmin(pmax(ax$tr(top), ax$tlim[1]), ax$tlim[2])
      top[!is.finite(top)] <- ax$tlim[2]
      bd <- data.frame(x1 = x1, x2 = x2, label = br$label, zone = zone, top = top)
      bl <- bracket_layers(ctx, bd, ax, M$vlen)
      ax <- bl$ax
    }
  }

  cat_assemble(ctx, S, ax, M, c(K$layers(ax, M), bl$layers), legend, S$gl, key_w)
}

# Put a categorical-axis graph together: reference lines, the layers, both
# position scales, colours, coordinate system (flipped or not), labels, theme.
cat_assemble <- function(ctx, S, ax, M, layers, legend, keys, key_w = 1, extra_cols = NULL, key_labels = NULL,
                         show_cat_labels = TRUE, theme_extra = NULL) {
  xlim <- c(1 - CAT_PAD, S$n + CAT_PAD)
  ggplot() + ref_lines(ctx, NULL, ax) + layers +
    scale_x_continuous(breaks = if (show_cat_labels) seq_len(S$n) else NULL, labels = if (show_cat_labels) S$xl else NULL,
                       expand = c(0, 0), trans = if (ctx$flip) "reverse" else "identity") +
    axis_scale(ctx, ax) +
    colour_scales(ctx, keys, labels = key_labels, extra = c(dark_extra(ctx, keys), extra_cols)) +
    (if (ctx$flip) coord_flip(xlim = rev(xlim), ylim = ax$lim, expand = FALSE)
     else coord_cartesian(xlim = xlim, ylim = ax$lim, expand = FALSE)) +
    gg_labs(ctx) + gg_theme(ctx, legend = legend, x_angle = M$angle, key_w = key_w) + theme_extra
}

register("column_scatter", function(ctx) render_categorical(ctx, kind_scatter, FALSE))
register("column_bar", function(ctx) render_categorical(ctx, kind_bar, FALSE))
register("box", function(ctx) render_categorical(ctx, kind_box, FALSE))
register("violin", function(ctx) render_categorical(ctx, kind_violin, FALSE))
register("raincloud", function(ctx) render_categorical(ctx, kind_raincloud, FALSE))
register("before_after", function(ctx) render_categorical(ctx, kind_before_after, FALSE))
register("grouped_bar", function(ctx) render_categorical(ctx, kind_bar, TRUE))
register("grouped_box", function(ctx) render_categorical(ctx, kind_box, TRUE))
register("grouped_violin", function(ctx) render_categorical(ctx, kind_violin, TRUE))
register("grouped_points", function(ctx) render_categorical(ctx, kind_scatter, TRUE))
register("interaction_line", function(ctx) render_categorical(ctx, kind_interaction, TRUE))
