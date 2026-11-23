# ggplot2 engine — shared machinery: options, units, theme, axes, legends,
# points, error bars, significance brackets, output devices and dispatch.
#
# Every renderer is a function `render_<type>(ctx)` returning a ggplot (or a
# patchwork). `ctx` (see make_ctx) carries the options, frames, computed
# values, resolved theme/fonts/colours and the figure size.
#
# Conventions
#   * all "pt" options are typographic points; convert with lw() for line
#     widths, ts() for text sizes and pt_size() for marker diameters;
#   * numeric axes are fully resolved by axis_spec() (limits known up front,
#     scales never expand), which lets brackets, swarms and annotations be laid
#     out in physical units;
#   * categorical axes are continuous scales with one unit per category, so
#     dodging, half-violins and brackets share one coordinate system.

PT <- 72.27 / 25.4                      # ggplot2::.pt
lw <- function(pt) pt / 2.134           # line width: pt -> ggplot linewidth
ts <- function(pt) pt / PT              # text size:  pt -> geom_text size

# ---- option access (exact matching, tolerant of NULL / NA / "") -------------
opt_num <- function(o, name, default = NA_real_) {
  v <- o[[name]]
  if (is.null(v) || length(v) != 1) return(default)
  v <- suppressWarnings(as.numeric(v))
  if (is.na(v) || !is.finite(v)) default else v
}
opt_chr <- function(o, name, default = "") {
  v <- o[[name]]
  if (is.null(v) || length(v) != 1 || is.na(v) || !nzchar(as.character(v))) default else as.character(v)
}
opt_flag <- function(o, name, default = FALSE) {
  v <- o[[name]]
  if (is.null(v) || length(v) != 1 || is.na(v)) default else isTRUE(as.logical(v))
}
chr_vec <- function(x) if (is.null(x) || length(x) == 0) character(0) else as.character(unlist(x))
num_vec <- function(x) {
  if (is.null(x) || length(x) == 0) return(numeric(0))
  v <- suppressWarnings(as.numeric(unlist(lapply(x, function(e) if (is.null(e)) NA else e))))
  v
}
# list of records (JSON array of objects) -> data.frame of character/numeric columns
records <- function(x, fields) {
  if (is.null(x) || length(x) == 0 || !is.list(x)) x <- list()
  if (length(x) && !is.null(names(x)) && !is.list(x[[1]])) x <- list(x)   # a single record
  out <- lapply(fields, function(f) vapply(x, function(r) {
    v <- r[[f]]
    if (is.null(v) || length(v) == 0 || is.na(v[1])) NA_character_ else as.character(v[1])
  }, character(1)))
  names(out) <- fields
  as.data.frame(out, stringsAsFactors = FALSE)
}
has_rows <- function(df) is.data.frame(df) && nrow(df) > 0
finite_range <- function(...) {
  v <- c(...)
  v <- v[is.finite(v)]
  if (length(v)) range(v) else c(NA_real_, NA_real_)
}

# ---- colour helpers ----------------------------------------------------------
col_rgb <- function(col) grDevices::col2rgb(col) / 255
luminance <- function(col) {
  m <- col_rgb(col)
  as.numeric(0.2126 * m[1, ] + 0.7152 * m[2, ] + 0.0722 * m[3, ])
}
mix_col <- function(col, with, amount) {
  a <- col_rgb(col); b <- col_rgb(with)
  amount <- rep_len(amount, ncol(a))
  b <- b[, rep_len(seq_len(ncol(b)), ncol(a)), drop = FALSE]
  m <- t(t(a) * (1 - amount) + t(b) * amount)
  grDevices::rgb(m[1, ], m[2, ], m[3, ])
}
darken <- function(col, amount = 0.35) mix_col(col, "black", amount)
# black or white text, whichever reads better on `fill` (optionally a
# translucent fill over `bg`)
contrast_col <- function(fill, alpha = 1, bg = "white") {
  eff <- if (all(alpha >= 1)) fill else mix_col(fill, bg, 1 - alpha)
  ifelse(luminance(eff) > 0.52, "#111111", "#FFFFFF")
}

# ---- context -------------------------------------------------------------
make_ctx <- function(req) {
  o <- req$options %||% list()
  th <- o[["_theme"]] %||% list()
  f <- o[["_fonts"]] %||% list()
  thv <- function(k, d) { v <- th[[k]]; if (is.null(v) || length(v) != 1 || is.na(v)) d else v }
  th <- list(bg = thv("bg", "#FFFFFF"), panel_bg = thv("panel_bg", "#FFFFFF"),
             text = thv("text", "#111111"), axis = thv("axis", "#111111"),
             grid_color = thv("grid_color", "#E6E6E6"), grid = thv("grid", "none"),
             frame = thv("frame", "L"), tick_dir = thv("tick_dir", "out"),
             axis_width = as.numeric(thv("axis_width", 1)), title_align = thv("title_align", "center"),
             title_bold = isTRUE(thv("title_bold", TRUE)), axis_title_bold = isTRUE(thv("axis_title_bold", FALSE)),
             transparent = isTRUE(thv("transparent", FALSE)))
  base <- opt_num(f, "base", 11)
  fz <- function(k, mult) opt_num(f, k, round(base * mult, 2))
  fonts <- list(family = opt_chr(f, "family", "sans"), base = base, title = fz("title", 1.2),
                subtitle = fz("subtitle", 0.95), axis_title = fz("axis_title", 1),
                tick = fz("tick", 0.9), legend = fz("legend", 0.9), caption = fz("caption", 0.8))
  lab <- o[["_labels"]] %||% list()
  labels <- lapply(c(title = "title", subtitle = "subtitle", caption = "caption", x = "x", y = "y",
                     legend = "legend"), function(k) opt_chr(lab, k, ""))
  frames <- lapply(req$frames %||% list(), aui_frame)
  nm <- chr_vec(o[["_color_names"]]); vals <- chr_vec(o[["_color_values"]])
  n <- min(length(nm), length(vals))
  cols <- stats::setNames(vals[seq_len(n)], nm[seq_len(n)])
  cols <- cols[!duplicated(names(cols))]
  dark <- luminance(th$panel_bg) < 0.4
  list(type = req$type, o = o, th = th, f = fonts, labels = labels, frames = frames,
       computed = req$computed %||% list(), cols = cols,
       W = opt_num(o, "width", 5), H = opt_num(o, "height", 4),
       flip = opt_flag(o, "flip"), dark = dark,
       ink = th$text,                                  # "black" that also works on dark themes
       muted = mix_col(th$text, th$panel_bg, 0.45),    # gray for secondary marks
       hlines = num_vec(o[["_hlines"]]), vlines = num_vec(o[["_vlines"]]))
}

# colour of group names missing from the palette (should not happen) -> ink
group_cols <- function(ctx, keys) {
  keys <- as.character(keys)
  out <- ctx$cols[keys]
  out[is.na(out)] <- ctx$ink
  stats::setNames(as.character(out), keys)
}

# ---- text measurement (points) ---------------------------------------------
text_w <- function(ctx, strings, size, bold = FALSE) {
  strings <- as.character(strings)
  if (!length(strings)) return(numeric(0))
  lines <- strsplit(strings, "\n", fixed = TRUE)
  flat <- unlist(lapply(lines, function(l) if (length(l)) l else ""))
  w <- tryCatch(
    systemfonts::string_width(flat, family = ctx$f$family, bold = bold, size = size, res = 72),
    error = function(e) nchar(flat) * size * 0.55)
  idx <- rep(seq_along(lines), pmax(lengths(lines), 1L))
  as.numeric(tapply(w, idx, max))
}
max_text_w <- function(ctx, strings, size, bold = FALSE) {
  w <- text_w(ctx, strings, size, bold)
  if (length(w)) max(w, na.rm = TRUE) else 0
}

# ---- numeric axes ------------------------------------------------------------
minus <- function(x) sub("^-", "−", x)
fmt_num <- function(x) {
  if (!length(x)) return(character(0))
  a <- abs(x[is.finite(x) & x != 0])
  lab <- if (length(a) && (max(a) >= 1e6 || max(a) < 1e-4)) {
    format(x, scientific = TRUE, digits = 3, trim = TRUE)
  } else {
    format(round(x, 10), scientific = FALSE, trim = TRUE, digits = 8)
  }
  lab[!is.finite(x)] <- ""
  minus(lab)
}
fmt_each <- function(x) minus(vapply(x, function(v) format(v, scientific = FALSE, trim = TRUE,
                                                          drop0trailing = TRUE, digits = 8), character(1)))

axis_trans <- function(scale) {
  switch(scale,
         log10 = list(name = "log10", tr = log10, inv = function(x) 10^x),
         log2 = list(name = "log2", tr = log2, inv = function(x) 2^x),
         sqrt = list(name = "sqrt", tr = function(x) sqrt(pmax(x, 0)), inv = function(x) pmax(x, 0)^2),
         list(name = "identity", tr = identity, inv = identity))
}

# Resolve one numeric axis.
#   rng     data range to show (data units)
#   zero    make sure 0 is included (bars)
#   flush   no padding on a side whose data bound is exactly 0 (bars sit on the axis)
#   mult    padding as fractions of the (transformed) span: c(low, high)
#   fixed   exact limits that ignore the data (e.g. c(0, 100))
# Returns list(which, scale, tr, inv, lim (data units), tlim (transformed), user = c(min, max set?)).
axis_spec <- function(ctx, which, rng, zero = FALSE, flush = FALSE, mult = c(0.05, 0.05), fixed = NULL) {
  if (!is.null(ctx$fixed_axes[[which]])) return(ctx$fixed_axes[[which]])   # axis shared between facet panels
  o <- ctx$o
  scale <- opt_chr(o, paste0(which, "_scale"), "linear")
  if (!scale %in% c("linear", "log10", "log2", "sqrt")) scale <- "linear"
  tf <- axis_trans(scale)
  is_log <- scale %in% c("log10", "log2")
  rng <- suppressWarnings(as.numeric(rng))
  rng <- rng[is.finite(rng)]
  if (is_log) rng <- rng[rng > 0]
  if (scale == "sqrt") rng <- pmax(rng, 0)
  if (!length(rng)) rng <- if (is_log) c(1, 10) else c(0, 1)
  rng <- range(rng)
  if (zero && !is_log) rng <- range(rng, 0)
  t <- tf$tr(rng)
  span <- diff(t)
  if (!is.finite(span) || span <= max(abs(t), 1e-300) * 1e-9) {   # constant data: open a sensible window
    t <- rep(mean(t), 2)
    pad <- if (is_log) 0.5 else if (rng[1] == 0) 1 else abs(rng[1]) * 0.1
    if (scale == "sqrt") pad <- max(sqrt(abs(rng[1])) * 0.1, 0.5)
    t <- t + c(-pad, pad)
    if (flush && !is_log && rng[1] >= 0) t[1] <- max(t[1], 0)
    span <- diff(t)
    lo <- t[1]; hi <- t[2]
  } else {
    lo <- t[1] - mult[1] * span
    hi <- t[2] + mult[2] * span
  }
  if (flush && !is_log) {
    if (rng[1] == 0) lo <- tf$tr(0)
    if (rng[2] == 0) hi <- tf$tr(0)
  }
  if (scale == "sqrt") lo <- max(lo, 0)
  if (!is.null(fixed)) {
    tl <- suppressWarnings(tf$tr(fixed))
    if (is.finite(tl[1])) lo <- tl[1]
    if (is.finite(tl[2])) hi <- tl[2]
  }
  umin <- opt_num(o, paste0(which, "_min")); umax <- opt_num(o, paste0(which, "_max"))
  if (is_log) { if (!is.na(umin) && umin <= 0) umin <- NA; if (!is.na(umax) && umax <= 0) umax <- NA }
  if (scale == "sqrt") { if (!is.na(umin) && umin < 0) umin <- 0; if (!is.na(umax) && umax <= 0) umax <- NA }
  if (!is.na(umin) && !is.na(umax) && umin >= umax) { umin <- NA; umax <- NA }
  user <- c(!is.na(umin), !is.na(umax))
  if (user[1]) lo <- tf$tr(umin)
  if (user[2]) hi <- tf$tr(umax)
  if (hi <= lo) {                                   # one user limit beyond the data
    w <- if (span > 0) span else 1
    if (user[1] && !user[2]) hi <- lo + w else lo <- hi - w
    if (scale == "sqrt") lo <- max(lo, 0)
  }
  list(which = which, scale = scale, tr = tf$tr, inv = tf$inv, trans = tf$name,
       lim = tf$inv(c(lo, hi)), tlim = c(lo, hi), user = user, is_log = is_log)
}

# raise the upper limit (transformed units) unless the user fixed it
axis_extend_top <- function(ax, t_hi) {
  if (!ax$user[2] && is.finite(t_hi) && t_hi > ax$tlim[2]) {
    ax$tlim[2] <- t_hi
    ax$lim <- ax$inv(ax$tlim)
  }
  ax
}

axis_breaks <- function(ctx, ax, minor_ok = TRUE) {
  lim <- ax$lim
  step <- opt_num(ctx$o, paste0(ax$which, "_tick_step"))
  eps <- diff(ax$tlim) * 1e-6
  inside <- function(b) b[ax$tr(b) >= ax$tlim[1] - eps & ax$tr(b) <= ax$tlim[2] + eps]
  if (ax$scale == "log10") {
    k <- seq(ceiling(ax$tlim[1] - 1e-9), floor(ax$tlim[2] + 1e-9))
    if (length(k) >= 2 && k[1] <= k[length(k)]) {
      every <- ceiling(length(k) / 8)
      kk <- k[k %% every == 0]
      if (length(kk) < 2) kk <- k
      major <- 10^kk
      wide <- min(kk) < -4 || max(kk) > 5
      labs <- if (wide) parse(text = sprintf("10^%d", kk)) else fmt_each(major)
      if (minor_ok && every == 1 && length(k) <= 6) {
        kmin <- seq(min(k) - 1, max(k))
        minor <- inside(setdiff(as.vector(outer(2:9, 10^kmin)), major))
        if (length(minor)) {
          br <- c(major, minor)
          lb <- if (wide) parse(text = c(sprintf("10^%d", kk), rep('""', length(minor))))
                else c(labs, rep("", length(minor)))
          ord <- order(br)
          return(list(breaks = br[ord], labels = lb[ord]))
        }
      }
      return(list(breaks = major, labels = labs))
    }
    br <- inside(scales::log_breaks(n = 6)(lim))
    if (length(br) < 2) br <- inside(scales::extended_breaks(n = 5)(lim))
    return(list(breaks = br, labels = fmt_each(br)))
  }
  if (ax$scale == "log2") {
    k <- seq(ceiling(ax$tlim[1] - 1e-9), floor(ax$tlim[2] + 1e-9))
    if (length(k) >= 2 && k[1] <= k[length(k)]) {
      every <- ceiling(length(k) / 9)
      br <- 2^k[k %% every == 0]
    } else {
      br <- inside(scales::extended_breaks(n = 5)(lim))
    }
    return(list(breaks = br, labels = fmt_each(br)))
  }
  br <- NULL
  if (!is.na(step) && step > 0 && diff(lim) / step <= 150) {
    first <- ceiling(lim[1] / step - 1e-9) * step
    if (first <= lim[2]) br <- round(seq(first, lim[2] + step * 1e-9, by = step), 12)
  }
  if (is.null(br)) {
    # fewer ticks when the labels would not fit along the axis (ax$len_pt, when known)
    for (n in c(5, 4, 3, 2)) {
      br <- inside(scales::extended_breaks(n = n)(lim))
      if (is.null(ax$len_pt) || length(br) <= 2) break
      need <- if (isTRUE(ax$horizontal)) sum(text_w(ctx, fmt_num(br), ctx$f$tick)) + length(br) * ctx$f$tick * 0.9
              else length(br) * ctx$f$tick * 1.25
      if (need <= ax$len_pt) break
    }
  } else br <- inside(br)
  list(breaks = br, labels = fmt_num(br))
}

# tell an axis how long it is on the page (pt) and whether its labels run along it
axis_on_page <- function(ax, len_pt, horizontal) {
  ax$len_pt <- len_pt; ax$horizontal <- horizontal
  ax
}

# continuous position scale for a resolved axis (never expands: the limits are final)
axis_scale <- function(ctx, ax, breaks = NULL, labels = NULL, minor_ok = TRUE, ...) {
  if (is.null(breaks)) {
    b <- axis_breaks(ctx, ax, minor_ok = minor_ok && !grid_on(ctx, ax$which))
    breaks <- b$breaks
    if (is.null(labels)) labels <- b$labels
  }
  fn <- if (ax$which == "x") scale_x_continuous else scale_y_continuous
  if (!length(breaks) || length(labels) != length(breaks)) {
    if (!length(breaks)) return(fn(trans = ax$trans, breaks = NULL, expand = c(0, 0), ...))
    labels <- fmt_num(breaks)
  }
  fn(trans = ax$trans, breaks = breaks, labels = labels, expand = c(0, 0), ...)
}

# keep only values that can be drawn on the axis' scale
on_scale <- function(ax, v) if (ax$is_log) is.finite(v) & v > 0 else if (ax$scale == "sqrt") is.finite(v) & v >= 0 else is.finite(v)

# ---- theme -------------------------------------------------------------------
# which grid lines are drawn, as seen on the page: the theme default follows
# the value axis when the graph is flipped; an explicit choice is literal.
grid_mode <- function(ctx) {
  g <- ctx$th$grid
  if (ctx$flip && opt_chr(ctx$o, "grid", "theme") == "theme") g <- switch(g, x = "y", y = "x", g)
  g
}
# is the grid drawn across the *aesthetic* axis `which` ("x"/"y" before flipping)?
grid_on <- function(ctx, which) {
  g <- grid_mode(ctx)
  page <- if (ctx$flip) (if (which == "x") "y" else "x") else which
  g == "both" || g == page
}

legend_position <- function(ctx, default = "right") {
  p <- opt_chr(ctx$o, "legend_position", "auto")
  if (p == "auto") default else p
}

# angle spec for bottom tick labels
tick_angle <- function(angle) {
  angle <- suppressWarnings(as.numeric(angle))
  if (is.na(angle) || angle == 0) return(NULL)
  if (angle >= 90) list(angle = 90, hjust = 1, vjust = 0.5) else list(angle = angle, hjust = 1, vjust = 1)
}

# The complete theme.
#   legend    resolved legend position
#   x_angle   NULL or tick_angle() spec for the bottom axis labels
#   key_w     legend key width as a multiple of the key height (lines need more)
#   void      no axes at all (pie)
gg_theme <- function(ctx, legend = "none", x_angle = NULL, key_w = 1, void = FALSE) {
  th <- ctx$th; f <- ctx$f
  alw <- lw(th$axis_width)
  has_line <- alw > 0
  axis_line <- if (has_line) element_line(colour = th$axis, linewidth = alw, lineend = "square") else element_blank()
  tick_len <- max(2.5, f$tick * 0.32)
  ticks_on <- th$tick_dir != "none" && th$frame != "none" && has_line
  tl <- if (!ticks_on) 0 else if (th$tick_dir == "in") -tick_len else tick_len
  gap <- max(2, f$tick * 0.25) + (if (ticks_on && th$tick_dir == "in") tick_len else 0)
  g <- grid_mode(ctx)
  grid_line <- element_line(colour = th$grid_color, linewidth = lw(0.6), lineend = "butt")
  bg_fill <- if (th$transparent) NA else th$bg
  panel_fill <- if (th$transparent) NA else th$panel_bg
  title_h <- if (th$title_align == "left") 0 else 0.5
  pad <- max(5, f$base * 0.55)
  key <- f$legend * 1.25

  t <- theme_grey(base_size = f$base, base_family = f$family) + theme(
    text = element_text(colour = th$text, family = f$family, size = f$base),
    line = element_line(colour = th$axis),
    plot.background = element_rect(fill = bg_fill, colour = NA),
    panel.background = element_rect(fill = panel_fill, colour = NA),
    panel.border = if (th$frame == "box" && has_line)
      element_rect(fill = NA, colour = th$axis, linewidth = alw) else element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = if (g %in% c("x", "both")) grid_line else element_blank(),
    panel.grid.major.y = if (g %in% c("y", "both")) grid_line else element_blank(),
    axis.line = if (th$frame == "L") axis_line else element_blank(),
    axis.ticks = if (ticks_on) element_line(colour = th$axis, linewidth = alw, lineend = "butt") else element_blank(),
    axis.ticks.length = grid::unit(tl, "pt"),
    axis.text = element_text(colour = th$text, size = f$tick),
    axis.text.x = element_text(margin = margin(t = gap), vjust = 1),
    axis.text.y = element_text(margin = margin(r = gap), hjust = 1),
    axis.text.y.right = element_text(margin = margin(l = gap), hjust = 0),
    axis.title = element_text(colour = th$text, size = f$axis_title,
                              face = if (th$axis_title_bold) "bold" else "plain"),
    axis.title.x = element_text(margin = margin(t = f$axis_title * 0.5)),
    axis.title.y = element_text(margin = margin(r = f$axis_title * 0.5), angle = 90),
    plot.title = element_text(colour = th$text, size = f$title, hjust = title_h,
                              face = if (th$title_bold) "bold" else "plain",
                              margin = margin(b = f$title * 0.5)),
    plot.subtitle = element_text(colour = mix_col(th$text, th$bg, 0.2), size = f$subtitle, hjust = title_h,
                                 margin = margin(b = f$subtitle * 0.6)),
    plot.caption = element_text(colour = mix_col(th$text, th$bg, 0.3), size = f$caption, hjust = 0,
                                margin = margin(t = f$caption * 0.8)),
    plot.title.position = if (th$title_align == "left") "plot" else "panel",
    plot.caption.position = "plot",
    plot.margin = margin(pad, pad + 3, pad, pad),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.box.background = element_blank(),
    legend.text = element_text(colour = th$text, size = f$legend),
    legend.title = if (nzchar(ctx$labels$legend))
      element_text(colour = th$text, size = f$legend, face = "bold") else element_blank(),
    legend.key.height = grid::unit(key, "pt"),
    legend.key.width = grid::unit(key * key_w, "pt"),
    legend.spacing.x = grid::unit(f$legend * 0.4, "pt"),
    legend.spacing.y = grid::unit(f$legend * 0.3, "pt"),
    legend.margin = margin(2, 2, 2, 2),
    legend.box.spacing = grid::unit(f$legend * 0.8, "pt"),
    strip.background = element_blank(),
    strip.text = element_text(colour = th$text, size = f$tick, face = "bold", hjust = 0,
                              margin = margin(b = 3, t = 2)),
    strip.clip = "off"
  )
  if (!is.null(ctx$facet)) {
    t <- t + theme(strip.text = if (ctx$facet$labels) element_text(colour = th$text, size = f$tick, face = "bold", hjust = 0.5,
                                                                   margin = margin(b = 4, t = 2)) else element_blank(),
                   panel.spacing = grid::unit(ctx$facet$spacing, "pt"))
  }
  if (!is.null(x_angle)) {
    t <- t + theme(axis.text.x = element_text(angle = x_angle$angle, hjust = x_angle$hjust,
                                              vjust = x_angle$vjust, margin = margin(t = gap)))
  }
  if (void) {
    t <- t + theme(axis.line = element_blank(), axis.ticks = element_blank(), axis.text = element_blank(),
                   axis.text.x = element_blank(), axis.text.y = element_blank(),
                   axis.title = element_blank(), axis.title.x = element_blank(), axis.title.y = element_blank(),
                   panel.grid.major.x = element_blank(),
                   panel.grid.major.y = element_blank(), panel.border = element_blank(),
                   panel.background = element_blank(), axis.ticks.length = grid::unit(0, "pt"))
  }
  t + legend_theme(ctx, legend)
}

legend_theme <- function(ctx, pos) {
  inside <- function(x, y) {
    args <- list(legend.justification = c(x, y), legend.box.margin = margin(4, 4, 4, 4),
                 legend.background = element_rect(fill = grDevices::adjustcolor(ctx$th$panel_bg, 0.75), colour = NA))
    if (utils::packageVersion("ggplot2") >= "3.5.0") {
      args$legend.position <- "inside"; args$legend.position.inside <- c(x, y)
    } else {
      args$legend.position <- c(x, y)
    }
    if (ctx$th$transparent) args$legend.background <- element_blank()
    do.call(theme, args)
  }
  switch(pos,
         none = theme(legend.position = "none"),
         top = theme(legend.position = "top", legend.direction = "horizontal",
                     legend.box.spacing = grid::unit(ctx$f$legend * 0.3, "pt")),
         bottom = theme(legend.position = "bottom", legend.direction = "horizontal"),
         left = theme(legend.position = "left"),
         inside_tr = inside(1, 1), inside_tl = inside(0, 1),
         inside_br = inside(1, 0), inside_bl = inside(0, 0),
         theme(legend.position = "right"))
}

# titles; "" -> NULL so ggplot leaves no gap
gg_labs <- function(ctx, x = ctx$labels$x, y = ctx$labels$y) {
  nz <- function(s) if (is.null(s) || !nzchar(s)) NULL else s
  L <- ctx$labels
  labs(title = nz(L$title), subtitle = nz(L$subtitle), caption = nz(L$caption), x = nz(x), y = nz(y))
}

# manual colour + fill scales sharing breaks and labels, so legends merge.
#   extra   additional named colours that may be mapped but never appear in the legend
#   labels  named character vector overriding legend labels
colour_scales <- function(ctx, keys = names(ctx$cols), labels = NULL, extra = NULL, aesthetics = c("colour", "fill"),
                          legend = "none", key_w = 1) {
  vals <- c(group_cols(ctx, keys), extra)
  lab <- if (is.null(labels)) keys else { l <- labels[keys]; l[is.na(l)] <- keys[is.na(l)]; unname(l) }
  title <- if (nzchar(ctx$labels$legend)) ctx$labels$legend else NULL
  guide <- legend_guide(ctx, legend, lab, key_w)
  lapply(aesthetics, function(a) {
    scale_discrete_manual(aesthetics = a, values = vals, breaks = keys, labels = lab, name = title,
                          drop = FALSE, na.value = ctx$muted, guide = guide)
  })
}

# A legend guide whose number of columns makes the keys fit on the page:
# side legends wrap into columns when they are taller than the figure,
# top / bottom legends use as many columns as the width allows.
legend_guide <- function(ctx, legend, labels, key_w = 1, ...) {
  n <- length(labels)
  if (legend == "none" || n < 2) return(guide_legend(...))
  f <- ctx$f
  key <- f$legend * 1.25
  lines <- max(1, lengths(strsplit(as.character(labels), "\n", fixed = TRUE)))
  entry_w <- max_text_w(ctx, labels, f$legend) + key * key_w + f$legend * 1.3
  if (legend %in% c("top", "bottom")) {
    title_w <- if (nzchar(ctx$labels$legend)) max_text_w(ctx, ctx$labels$legend, f$legend, TRUE) + f$legend else 0
    ncol <- max(1, min(n, floor((ctx$W * 72 - 16 - title_w) / entry_w)))
    return(guide_legend(ncol = ncol, byrow = TRUE, ...))
  }
  rows <- max(1, floor((ctx$H * 72 * (if (grepl("^inside", legend)) 0.6 else 0.85)) / (max(key, f$legend * 1.15 * lines) + f$legend * 0.1)))
  if (n <= rows) return(guide_legend(...))
  guide_legend(ncol = ceiling(n / rows), ...)
}

# ---- size of the panel, estimated (points) -----------------------------------
# Used to convert physical sizes (marker diameters, text heights, caps in pt)
# into data units before the plot is built. `left_w`: width of the left tick
# labels; `bottom_h`: height of the bottom tick labels; `right`: extra space
# at the right (legend, secondary labels).
panel_size <- function(ctx, left_w = NULL, bottom_h = NULL, right = 0, top = 0, legend = "none",
                       legend_labels = NULL, key_w = 1, has_x_title = NULL, has_y_title = NULL) {
  f <- ctx$f; L <- ctx$labels
  tick_len <- max(2.5, f$tick * 0.32); gap <- max(2, f$tick * 0.25); pad <- max(5, f$base * 0.55)
  xt <- if (is.null(has_x_title)) nzchar(if (ctx$flip) L$y else L$x) else has_x_title
  yt <- if (is.null(has_y_title)) nzchar(if (ctx$flip) L$x else L$y) else has_y_title
  left <- pad + (if (yt) f$axis_title * 1.65 else 0) + (left_w %||% (f$tick * 2.2)) + gap + tick_len
  bottom <- pad + (if (xt) f$axis_title * 1.65 else 0) + (bottom_h %||% (f$tick * 1.1)) + gap + tick_len
  topm <- pad + top + (if (nzchar(L$title)) f$title * 1.6 else 0) + (if (nzchar(L$subtitle)) f$subtitle * 1.7 else 0)
  if (nzchar(L$caption)) bottom <- bottom + f$caption * 1.9
  rightm <- pad + 3 + right
  if (legend %in% c("right", "left", "top", "bottom")) {
    labs <- legend_labels %||% names(ctx$cols)
    key <- f$legend * 1.25
    lw_each <- text_w(ctx, labs, f$legend) + key * key_w + f$legend * 0.8
    title_w <- if (nzchar(L$legend)) max_text_w(ctx, L$legend, f$legend, bold = TRUE) else 0
    n <- length(labs)
    if (legend %in% c("right", "left")) {
      rows <- max(1, floor(ctx$H * 72 * 0.85 / (key + f$legend * 0.1)))
      w <- max(c(max(lw_each, 0) * ceiling(n / rows), title_w), 0) + f$legend * 0.8 + 6
      if (legend == "right") rightm <- rightm + w else left <- left + w
    } else {
      ncol <- max(1, min(n, floor((ctx$W * 72 - 16 - title_w) / (max(lw_each, 1) + f$legend * 0.5))))
      h <- key * ceiling(n / ncol) + f$legend * 0.8 + 6
      if (legend == "top") topm <- topm + h else bottom <- bottom + h
    }
  }
  w <- ctx$W * 72 - left - rightm; h <- ctx$H * 72 - topm - bottom
  Fc <- ctx$facet
  if (!is.null(Fc)) {                                # the plotting area is shared by a grid of panels
    axis_v <- (left_w %||% (f$tick * 2.2)) + gap + tick_len
    axis_h <- (bottom_h %||% (f$tick * 1.1)) + gap + tick_len
    w <- (w - (Fc$ncol - 1) * (Fc$spacing + if (Fc$free_v) axis_v else 0)) / Fc$ncol
    h <- (h - Fc$nrow * Fc$strip - (Fc$nrow - 1) * (Fc$spacing + if (Fc$free_h) axis_h else 0)) / Fc$nrow
  }
  list(w = max(w, 20), h = max(h, 20), left = left, right = rightm)
}

# ---- markers -----------------------------------------------------------------
SHAPE_SOLID <- c(circle = 16, square = 15, triangle = 17, diamond = 18)
SHAPE_FILLED <- c(circle = 21, square = 22, triangle = 24, diamond = 23)
SHAPE_OPEN <- c(circle = 1, square = 0, triangle = 2, diamond = 5)
# R draws pch 15/0/5 larger and pch 18 smaller than a circle of the same
# "size"; these factors give every marker the area of the circle.
SHAPE_SCALE <- c(`16` = 1, `15` = 0.886, `17` = 1, `18` = 1.2533, `21` = 1, `22` = 1, `23` = 1, `24` = 1,
                 `1` = 1, `0` = 0.886, `2` = 1, `5` = 0.886)

# ggplot `size` for a marker whose outer diameter is `d` pt (circle-equivalent).
# A circle is 0.75 x fontsize wide; fontsize = size * .pt + stroke * .stroke / 2;
# the outline adds stroke * 1.417 pt to the diameter.
pt_size <- function(d, shape = 16, stroke = 0) {
  sc <- SHAPE_SCALE[as.character(shape)]
  sc[is.na(sc)] <- 1
  line_pt <- stroke * 1.417
  pmax((pmax(d - line_pt, 0.1) / 0.75 * sc - stroke * 1.8898) / PT, 0.05)
}

# Marker description from the point options.
#   outlined = TRUE draws filled markers with a thin ink outline (used on top of
#   bars / boxes / violins where a same-coloured marker would disappear).
# Returns list(shape, size, stroke, alpha, mode, fill_const, colour_const, key)
# where key(group) gives the colour-scale key for each point.
marker_spec <- function(ctx, size = NULL, shape = NULL, alpha = NULL, mode = NULL, outlined = FALSE) {
  o <- ctx$o
  d <- size %||% opt_num(o, "point_size", 5)
  sh <- shape %||% opt_chr(o, "point_shape", "circle")
  if (sh == "auto") sh <- "circle"
  mode <- mode %||% opt_chr(o, "point_color", "group")
  alpha <- alpha %||% opt_num(o, "point_alpha", 0.9)
  open <- grepl("_open$", sh)
  base <- sub("_open$", "", sh)
  if (!base %in% names(SHAPE_SOLID)) base <- "circle"
  edge <- max(0.6, min(d / 6, 1.6))                 # outline thickness in pt
  spec <- list(alpha = alpha, mode = mode, d = d, fill_const = NULL, colour_const = NULL, fill_group = FALSE)
  if (open) {
    spec$shape <- SHAPE_OPEN[[base]]; spec$stroke <- edge / 1.417
  } else if (mode == "white") {
    spec$shape <- SHAPE_FILLED[[base]]; spec$stroke <- edge / 1.417
    spec$fill_const <- if (ctx$dark) ctx$th$panel_bg else "#FFFFFF"
  } else if (outlined) {
    spec$shape <- SHAPE_FILLED[[base]]; spec$stroke <- min(edge, 0.75) / 1.417
    spec$fill_group <- TRUE
    spec$colour_const <- ctx$ink
  } else {
    spec$shape <- SHAPE_SOLID[[base]]; spec$stroke <- 0
  }
  spec$size <- pt_size(d, spec$shape, spec$stroke)
  const <- switch(mode, black = ctx$ink, gray = if (ctx$dark) "#B8BDC4" else "#7F7F7F", NULL)
  if (!is.null(const)) {
    if (spec$fill_group) { spec$fill_group <- FALSE; spec$fill_const <- const } else spec$colour_const <- const
  }
  spec$key <- function(group) {
    g <- as.character(group)
    if (mode == "dark") paste0(g, DARK_SUFFIX) else g
  }
  spec
}
DARK_SUFFIX <- "​(dark)"
dark_extra <- function(ctx, keys = names(ctx$cols)) {
  stats::setNames(darken(group_cols(ctx, keys), 0.35), paste0(keys, DARK_SUFFIX))
}

# geom_point for a marker spec. `df` needs columns x, y and `key` (colour key).
marker_layer <- function(spec, df, mapping = aes(x = .data$x, y = .data$y), show.legend = FALSE, ...) {
  args <- list(data = df, shape = spec$shape, size = spec$size, stroke = spec$stroke, alpha = spec$alpha,
               show.legend = show.legend, ...)
  m <- mapping
  if (is.null(spec$colour_const)) m$colour <- quote(.data$key) else args$colour <- spec$colour_const
  if (spec$fill_group) m$fill <- quote(.data$key) else if (!is.null(spec$fill_const)) args$fill <- spec$fill_const
  args$mapping <- m
  do.call(geom_point, args)
}

# deterministic random numbers: re-rendering must not move jittered points
with_seed <- function(seed, expr) {
  old <- if (exists(".Random.seed", envir = globalenv())) get(".Random.seed", envir = globalenv()) else NULL
  on.exit(if (is.null(old)) suppressWarnings(rm(".Random.seed", envir = globalenv()))
          else assign(".Random.seed", old, envir = globalenv()))
  set.seed(seed)
  expr
}

# Horizontal offsets for points sharing a categorical position.
#   v        values (already transformed to the axis scale)
#   cell     id of the position each value belongs to
#   layout   jitter | swarm | aligned
#   half     maximal half-width (category units)
#   dx, dy   marker diameter in category units / in transformed value units
spread_points <- function(v, cell, layout, half, dx, dy) {
  off <- numeric(length(v))
  if (!length(v) || layout == "aligned" || half <= 0) return(off)
  if (layout == "jitter") {
    return(with_seed(20240229L, stats::runif(length(v), -half, half)))
  }
  for (idx in split(seq_along(v), cell)) {
    if (length(idx) < 2) next
    sx <- tryCatch(beeswarm::swarmx(rep(0, length(idx)), v[idx], xsize = dx * 1.05, ysize = dy * 1.05,
                                    cex = 1, priority = "density")$x,
                   error = function(e) NULL)
    if (is.null(sx)) sx <- with_seed(7L, stats::runif(length(idx), -half, half))
    m <- max(abs(sx))
    if (m > half) sx <- sx * half / m          # squeeze into the allowed band
    off[idx] <- sx
  }
  off
}

# ---- error bars on a linear position axis --------------------------------------
# df: x, lo, hi, center [, key]. Drawn as segments so that one-sided bars and
# missing ends work. `cap` is the full cap width in x units.
errorbar_layers <- function(df, cap, width_pt, colour = NULL, direction = "both") {
  if (!has_rows(df)) return(list())
  df <- df[is.finite(df$x) & (is.finite(df$lo) | is.finite(df$hi)), , drop = FALSE]
  if (!nrow(df)) return(list())
  if (direction == "up") {                       # away from zero only
    neg <- is.finite(df$center) & df$center < 0
    df$hi[neg] <- NA; df$lo[!neg] <- NA
  }
  a <- ifelse(is.finite(df$lo), df$lo, df$center)
  b <- ifelse(is.finite(df$hi), df$hi, df$center)
  stem <- data.frame(x = df$x, xend = df$x, y = a, yend = b, key = df$key %||% NA_character_)
  caps <- NULL
  if (cap > 0) {
    lo <- df[is.finite(df$lo), , drop = FALSE]; hi <- df[is.finite(df$hi), , drop = FALSE]
    caps <- rbind(
      data.frame(x = lo$x - cap / 2, xend = lo$x + cap / 2, y = lo$lo, yend = lo$lo, key = lo$key %||% rep(NA_character_, nrow(lo))),
      data.frame(x = hi$x - cap / 2, xend = hi$x + cap / 2, y = hi$hi, yend = hi$hi, key = hi$key %||% rep(NA_character_, nrow(hi))))
  }
  seg <- rbind(stem[is.finite(stem$y) & is.finite(stem$yend), , drop = FALSE], caps)
  if (!has_rows(seg)) return(list())
  m <- aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend)
  if (is.null(colour)) {
    m$colour <- quote(.data$key)
    list(geom_segment(m, data = seg, linewidth = lw(width_pt), lineend = "butt", show.legend = FALSE))
  } else {
    list(geom_segment(m, data = seg, linewidth = lw(width_pt), lineend = "butt", colour = colour, show.legend = FALSE))
  }
}

# ---- reference lines ---------------------------------------------------------
ref_lines <- function(ctx, ax_x = NULL, ax_y = NULL) {
  out <- list()
  ok <- function(v, ax) if (is.null(ax)) v[is.finite(v)] else v[on_scale(ax, v)]
  h <- ok(ctx$hlines, ax_y); v <- ok(ctx$vlines, ax_x)
  lwd <- lw(max(0.6, ctx$th$axis_width * 0.6))
  if (length(h)) out <- c(out, list(geom_hline(yintercept = h, colour = ctx$th$axis, linewidth = lwd, linetype = "22")))
  if (length(v)) out <- c(out, list(geom_vline(xintercept = v, colour = ctx$th$axis, linewidth = lwd, linetype = "22")))
  out
}

# ---- significance brackets -----------------------------------------------------
# br: data.frame x1, x2 (positions), label, zone (brackets of one zone stack
# together), top (transformed value the bracket has to clear).
# Brackets are laid out in points (level height follows the label size), then
# converted to data units, growing the axis until everything fits.
# Returns list(ax = updated axis, layers = list of geoms).
bracket_layers <- function(ctx, br, ax, panel_len_pt) {
  if (!has_rows(br)) return(list(ax = ax, layers = list()))
  br$label[is.na(br$label)] <- ""
  swap <- br$x1 > br$x2
  br[swap, c("x1", "x2")] <- br[swap, c("x2", "x1")]
  is_star <- grepl("^[*∗]+$", br$label)
  size_pt <- ctx$f$tick * ifelse(is_star, 1.25, 0.92)
  text_h <- ifelse(br$label == "", 0, ifelse(is_star, ctx$f$tick * 0.5, ctx$f$tick * 0.95))
  step <- max(text_h) + ctx$f$tick * 0.75 + 3      # pt between stacked brackets
  gap0 <- ctx$f$tick * 0.7 + 2                     # pt between the data and the first bracket
  tip <- min(4, ctx$f$tick * 0.35)                 # pt length of the bracket ends

  # greedy levels within each zone: shorter spans first
  br$level <- 0L
  inset <- 0.06
  for (z in unique(br$zone)) {
    idx <- which(br$zone == z)
    idx <- idx[order(br$x2[idx] - br$x1[idx], br$x1[idx])]
    placed <- integer(0)
    for (i in idx) {
      lev <- 0L
      repeat {
        clash <- any(br$level[placed] == lev &
                       br$x1[placed] + inset < br$x2[i] - inset & br$x2[placed] - inset > br$x1[i] + inset)
        if (!clash) break
        lev <- lev + 1L
      }
      br$level[i] <- lev
      placed <- c(placed, i)
    }
  }
  # never give more than 55 % of the axis to brackets: with very many of them the levels are squeezed
  max_level <- max(br$level)
  budget <- panel_len_pt * 0.55 - gap0 - max(text_h) - 4
  if (max_level > 0 && max_level * step > budget) step <- max(budget / max_level, 1)
  off_pt <- gap0 + br$level * step                 # bracket line above its zone top
  # the axis has to end at `hi` with top + need * (hi - lo) / panel_len <= hi for every bracket
  frac <- pmin((off_pt + text_h + 4) / panel_len_pt, 0.8)
  lo <- ax$tlim[1]
  hi <- max(ax$tlim[2], (br$top - frac * lo) / (1 - frac))
  ax <- axis_extend_top(ax, hi)
  u <- diff(ax$tlim) / panel_len_pt
  y <- br$top + off_pt * u
  seg <- rbind(
    data.frame(x = br$x1 + inset, xend = br$x2 - inset, y = y, yend = y),
    data.frame(x = br$x1 + inset, xend = br$x1 + inset, y = y, yend = y - tip * u),
    data.frame(x = br$x2 - inset, xend = br$x2 - inset, y = y, yend = y - tip * u))
  seg$y <- ax$inv(seg$y); seg$yend <- ax$inv(seg$yend)
  # stars sit high in their text box: pull them down towards the line
  ty <- y + ifelse(is_star, -ctx$f$tick * 0.24, ctx$f$tick * 0.25) * u
  txt <- data.frame(x = (br$x1 + br$x2) / 2, y = ax$inv(ty), label = br$label, size = ts(size_pt))
  txt <- txt[txt$label != "", , drop = FALSE]
  layers <- list(geom_segment(aes(x = .data$x, xend = .data$xend, y = .data$y, yend = .data$yend), data = seg,
                              colour = ctx$ink, linewidth = lw(max(0.7, ctx$th$axis_width * 0.75)),
                              lineend = "square", show.legend = FALSE))
  if (nrow(txt)) {
    layers <- c(layers, list(geom_text(
      aes(x = .data$x, y = .data$y, label = .data$label), data = txt, size = txt$size, colour = ctx$ink,
      family = ctx$f$family, vjust = 0, hjust = 0.5, angle = if (ctx$flip) -90 else 0, show.legend = FALSE)))
  }
  list(ax = ax, layers = layers)
}

# ---- devices -----------------------------------------------------------------
open_device <- function(req, ctx) {
  fmt <- req$format %||% "svg"
  w <- ctx$W; h <- ctx$H
  dpi <- opt_num(req, "dpi", 300)
  bg <- if (ctx$th$transparent) "transparent" else ctx$th$bg
  fam <- ctx$f$family
  switch(fmt,
    svg = if (isTRUE(req$editable_text)) svglite::svglite(req$file, width = w, height = h, bg = bg)
          else grDevices::svg(req$file, width = w, height = h, bg = bg, family = fam),
    pdf = grDevices::cairo_pdf(req$file, width = w, height = h, bg = bg, family = fam),
    png = ragg::agg_png(req$file, width = w, height = h, units = "in", res = dpi, background = bg),
    tiff = ragg::agg_tiff(req$file, width = w, height = h, units = "in", res = dpi, background = bg,
                          compression = "lzw"),
    jpg = ragg::agg_jpeg(req$file, width = w, height = h, units = "in", res = dpi, quality = 95,
                         background = if (ctx$th$transparent) "white" else ctx$th$bg),
    aui_stop("Unsupported image format: ", fmt))
  invisible(grDevices::dev.cur())
}

RENDERERS <- new.env()
register <- function(type, fn) assign(type, fn, envir = RENDERERS)

render <- function(req) {
  ctx <- make_ctx(req)
  fn <- RENDERERS[[req$type %||% ""]]
  if (is.null(fn)) aui_stop("The ggplot2 engine cannot draw '", req$type, "' yet.")
  if (!is.finite(ctx$W) || !is.finite(ctx$H) || ctx$W <= 0 || ctx$H <= 0) aui_stop("The figure size is not valid.")
  dev <- open_device(req, ctx)
  on.exit(if (dev %in% grDevices::dev.list()) grDevices::dev.off(dev), add = TRUE)
  plot <- fn(ctx)
  if (inherits(plot, "grob")) grid::grid.draw(plot) else print(plot)
  draw_annotations(ctx)
  grDevices::dev.off(dev)
  if (identical(req$format %||% "svg", "svg") && !isTRUE(req$editable_text)) svg_point_units(req$file)
  list(ok = TRUE)
}

# Cairo writes the SVG size as bare numbers (= CSS pixels in a browser) although
# its user unit is the point; declare the unit so the figure has its true size.
svg_point_units <- function(path) {
  size <- file.info(path)$size
  if (is.na(size) || size <= 0) return(invisible(FALSE))
  txt <- readChar(path, size, useBytes = TRUE)
  head <- substr(txt, 1, 600)
  fixed <- sub('(<svg[^>]*?[[:space:]]width=")([0-9.]+)("[[:space:]]+height=")([0-9.]+)(")', "\\1\\2pt\\3\\4pt\\5", head, perl = TRUE)
  if (!identical(fixed, head)) {
    con <- file(path, open = "wb")
    on.exit(close(con))
    writeChar(paste0(fixed, substr(txt, 601, nchar(txt, type = "bytes"))), con, eos = NULL, useBytes = TRUE)
  }
  invisible(TRUE)
}


# ---- free-form annotations ---------------------------------------------------
# Drawn last, on top of the finished figure, in figure coordinates (fractions
# from the top-left corner), so they never influence the layout.
draw_annotations <- function(ctx) {
  items <- ctx$o[["_annotations"]]
  if (is.null(items) || !is.list(items) || !length(items)) return(invisible(NULL))
  if (!is.null(names(items)) && !is.list(items[[1]])) items <- list(items)     # a single record
  Wp <- ctx$W * 72; Hp <- ctx$H * 72                                           # figure size in points
  pt <- function(v) grid::unit(v, "bigpts")
  for (a in items) {
    if (!is.list(a)) next
    kind <- opt_chr(a, "kind", "")
    x <- opt_num(a, "x", NA); y <- opt_num(a, "y", NA)
    if (is.na(x) || is.na(y)) next
    col <- opt_chr(a, "color", ctx$th$text)
    col <- tryCatch({ grDevices::col2rgb(col); col }, error = function(e) ctx$th$text)
    if (kind == "text") {
      label <- opt_chr(a, "text", "")
      if (!nzchar(label)) next
      bold <- opt_flag(a, "bold"); italic <- opt_flag(a, "italic")
      align <- opt_chr(a, "align", "center")
      grid::grid.draw(grid::textGrob(
        label, x = pt(x * Wp), y = pt((1 - y) * Hp), rot = opt_num(a, "angle", 0),
        hjust = switch(align, left = 0, right = 1, 0.5), vjust = 0.5,
        gp = grid::gpar(col = col, fontsize = opt_num(a, "size", ctx$f$base), fontfamily = ctx$f$family, lineheight = 0.95,
                        fontface = if (bold && italic) "bold.italic" else if (bold) "bold" else if (italic) "italic" else "plain")))
    } else if (kind %in% c("arrow", "line")) {
      x2 <- opt_num(a, "x2", NA); y2 <- opt_num(a, "y2", NA)
      if (is.na(x2) || is.na(y2)) next
      width <- opt_num(a, "width", 1.2)
      p0 <- c(x * Wp, (1 - y) * Hp); p1 <- c(x2 * Wp, (1 - y2) * Hp)
      len <- sqrt(sum((p1 - p0)^2))
      if (len < 0.01) next
      u <- (p1 - p0) / len                              # unit vector towards the head
      end <- p1
      if (kind == "arrow") {
        head <- min(max(5, width * 4.2), max(len * 0.9, 2))    # head length (pt), scales with the line width
        half <- head * 0.42
        base <- p1 - u * head
        n <- c(-u[2], u[1])
        tri <- rbind(p1, base + n * half, base - n * half)
        end <- p1 - u * head * 0.85                     # the shaft stops inside the head: no blunt end at the tip
        grid::grid.draw(grid::polygonGrob(x = pt(tri[, 1]), y = pt(tri[, 2]), gp = grid::gpar(fill = col, col = NA)))
      }
      grid::grid.draw(grid::segmentsGrob(pt(p0[1]), pt(p0[2]), pt(end[1]), pt(end[2]),
                                         gp = grid::gpar(col = col, lwd = width * 96 / 72, lineend = "butt")))
    }
  }
  invisible(NULL)
}

# ---- facets (multi-panel graphs) ---------------------------------------------
# A faceted graph is the unfaceted renderer run once per panel on that panel's
# subset: every panel computes its own swarm, brackets, error bars, curves ...
# and its layers are tagged with the `facet` column. The panels are then put
# into one ggplot with facet_wrap, so there is one legend and one set of titles.
#
# Protocol between with_facets() and a renderer:
#   * ctx$facet       grid description (see facet_layout); panel_size() returns
#                     the size of ONE panel when it is set;
#   * ctx$fixed_axes  axes shared by all panels, returned as-is by axis_spec();
#   * ctx$stop_at     "base" / "final": axes_ready() aborts the renderer at that
#                     point and hands back the panel's axes (cheap first passes
#                     used to work out shared limits);
#   * ctx$collect     the *_assemble() functions return their arguments instead
#                     of a plot; facet_plot() merges them.

axes_ready <- function(ctx, stage, x = NULL, y = NULL) {
  if (!is.null(ctx$stop_at) && (ctx$stop_at == stage || stage == "any")) {
    stop(structure(class = c("aui_axes", "error", "condition"),
                   list(message = "axes ready", call = NULL, x = x, y = y)))
  }
  invisible(NULL)
}

facet_layout <- function(ctx, levels, free_x_ok = TRUE) {
  comp <- ctx$computed
  n <- length(levels)
  ncol <- opt_num(comp, "facet_cols", NA)
  if (is.na(ncol) || ncol < 1) {                       # automatic: panels as square as possible
    cand <- seq_len(n)
    aspect <- (ctx$W / cand) / (ctx$H / ceiling(n / cand))
    ncol <- cand[which.min(abs(log(aspect / 1.05)) + 0.02 * ceiling(n / cand))]
  }
  ncol <- max(1, min(n, round(ncol)))
  nrow <- ceiling(n / ncol)
  mode <- opt_chr(comp, "facet_axes", "shared")
  free_y <- mode %in% c("free_y", "free")              # in terms of the x / y aesthetics
  free_x <- mode == "free" && free_x_ok
  free_h <- if (ctx$flip) free_y else free_x           # ... and as seen on the page
  free_v <- if (ctx$flip) free_x else free_y
  labels <- opt_flag(comp, "facet_labels", TRUE)
  lines <- max(1, lengths(strsplit(levels, "\n", fixed = TRUE)))
  list(levels = levels, n = n, ncol = ncol, nrow = nrow, free_x = free_x, free_y = free_y,
       free_h = free_h, free_v = free_v, labels = labels,
       scales = if (free_h && free_v) "free" else if (free_h) "free_x" else if (free_v) "free_y" else "fixed",
       spacing = max(8, ctx$f$base * 0.9), strip = if (labels) ctx$f$tick * 1.2 * lines + 6 else 0)
}

# the context of one panel: frames, brackets and fits restricted to that facet
facet_ctx <- function(ctx, level) {
  ctx$frames <- lapply(ctx$frames, function(df) {
    if (is.data.frame(df) && !is.null(df$facet)) df[!is.na(df$facet) & as.character(df$facet) == level, , drop = FALSE] else df
  })
  keep <- function(items) {
    if (is.null(items) || !is.list(items) || !length(items)) return(list())
    if (!is.null(names(items)) && !is.list(items[[1]])) items <- list(items)
    Filter(function(r) identical(as.character(r[["facet"]] %||% ""), level), items)
  }
  ctx$computed$brackets <- keep(ctx$computed$brackets)
  ctx$computed$fits <- keep(ctx$computed$fits)
  ctx$facet_level <- level
  ctx
}

flatten_layers <- function(x) {
  if (inherits(x, "Layer")) return(list(x))
  if (is.list(x)) return(unlist(lapply(x, flatten_layers), recursive = FALSE))
  list()
}
# send layers to one panel
tag_layers <- function(layers, level, levels) {
  layers <- flatten_layers(layers)
  for (l in layers) if (is.data.frame(l$data)) l$data$facet <- factor(rep(level, nrow(l$data)), levels = levels)
  layers
}

# Panels produce the same kinds of layers; drawing 7 x 8 separate layers is slow,
# so layers that only differ in their data are combined into one. Constant
# aesthetics given per row (e.g. text colours) are concatenated with the data.
merge_layers <- function(layers) {
  if (length(layers) < 2) return(layers)
  signature <- function(l) {
    n <- if (is.data.frame(l$data)) nrow(l$data) else -1L
    per_row <- vapply(l$aes_params, function(v) length(v) > 1 && length(v) == n, logical(1))
    const <- l$aes_params[!per_row]
    paste(class(l$geom)[1], class(l$stat)[1], class(l$position)[1],
          paste(names(l$mapping), vapply(l$mapping, function(m) paste(deparse(m), collapse = ""), ""), collapse = ";"),
          paste(names(const), vapply(const, function(v) paste(format(v), collapse = ","), ""), collapse = ";"),
          paste(sort(names(l$aes_params)[per_row]), collapse = ","),
          paste(deparse(l$geom_params[setdiff(names(l$geom_params), "na.rm")]), collapse = ""),
          paste(deparse(l$stat_params), collapse = ""), paste(deparse(l$show.legend), collapse = ""),
          if (is.data.frame(l$data)) paste(names(l$data), collapse = ",") else "nodata", sep = "|")
  }
  sig <- vapply(layers, signature, character(1))
  keep <- list()
  for (s in unique(sig)) {
    group <- layers[sig == s]
    first <- group[[1]]
    if (length(group) > 1 && is.data.frame(first$data)) {
      n <- nrow(first$data)
      for (nm in names(first$aes_params)) {
        if (length(first$aes_params[[nm]]) > 1 && length(first$aes_params[[nm]]) == n) {
          first$aes_params[[nm]] <- unlist(lapply(group, function(l) l$aes_params[[nm]]))
        }
      }
      first$data <- do.call(rbind, lapply(group, function(l) l$data))
    }
    keep <- c(keep, list(first))
  }
  keep
}

union_axes <- function(axes) {
  axes <- Filter(Negate(is.null), axes)
  if (!length(axes)) return(NULL)
  ax <- axes[[1]]
  ax$tlim <- c(min(vapply(axes, function(a) a$tlim[1], numeric(1))), max(vapply(axes, function(a) a$tlim[2], numeric(1))))
  ax$lim <- ax$inv(ax$tlim)
  ax
}

# Position scale for an axis that differs between panels. The limits of each
# panel come from invisible points at its resolved limits (facet_limits);
# breaks and labels are worked out per panel from those limits.
free_axis_scale <- function(ctx, ax, ...) {
  cache <- new.env()
  key <- function(b) paste(signif(b, 12), collapse = "|")
  minor_ok <- !grid_on(ctx, ax$which)
  brk <- function(lim) {
    a <- ax
    a$lim <- lim; a$tlim <- suppressWarnings(a$tr(lim))
    if (!all(is.finite(a$tlim))) return(numeric(0))
    b <- axis_breaks(ctx, a, minor_ok = minor_ok)
    assign(key(b$breaks), b$labels, envir = cache)
    b$breaks
  }
  lab <- function(b) {
    l <- if (length(b)) cache[[key(b)]] else NULL
    if (is.null(l) || length(l) != length(b)) fmt_num(b) else l
  }
  umin <- opt_num(ctx$o, paste0(ax$which, "_min")); umax <- opt_num(ctx$o, paste0(ax$which, "_max"))
  args <- list(trans = ax$trans, breaks = brk, labels = lab, expand = c(0, 0), ...)
  if (any(ax$user)) {                                  # user limits win over the trained range; nothing is dropped
    args$limits <- function(l) c(if (ax$user[1]) ax$lim[1] else l[1], if (ax$user[2]) ax$lim[2] else l[2])
    args$oob <- scales::oob_keep
  }
  do.call(if (ax$which == "x") scale_x_continuous else scale_y_continuous, args)
}
facet_limits <- function(axes, which, levels) {
  d <- do.call(rbind, lapply(seq_along(axes), function(i) {
    data.frame(v = axes[[i]]$lim, facet = factor(levels[i], levels = levels))
  }))
  geom_blank(if (which == "x") aes(x = .data$v) else aes(y = .data$v), data = d, inherit.aes = FALSE)
}

# facet_wrap + the pieces every faceted graph needs
facet_layers <- function(ctx) {
  Fc <- ctx$facet
  out <- list(facet_wrap(~facet, ncol = Fc$ncol, scales = Fc$scales, drop = FALSE))
  if (ctx$th$frame == "L" && ctx$th$axis_width > 0) {
    # ggplot draws shared axes on the outer panels only: give every panel its left and bottom line
    out <- c(out, list(annotation_custom(grid::polylineGrob(
      x = c(0, 0, 1), y = c(1, 0, 0), gp = grid::gpar(col = ctx$th$axis, lwd = ctx$th$axis_width * 96 / 72 * 2,
                                                      lineend = "square", linejoin = "mitre")))))
  }
  out
}

# Run `builder` (an ordinary renderer) per panel and combine the panels.
#   finish(ctx, parts)  builds the plot from merged parts (parts$panels = per-panel parts)
with_facets <- function(ctx, builder, finish, free_x_ok = TRUE) {
  levels <- chr_vec(ctx$computed$facets)
  has_col <- any(vapply(ctx$frames, function(df) is.data.frame(df) && !is.null(df$facet), logical(1)))
  if (!length(levels) || !has_col) return(builder(ctx))
  ctx$facet <- facet_layout(ctx, levels, free_x_ok)
  Fc <- ctx$facet
  panels <- lapply(levels, function(l) facet_ctx(ctx, l))
  axes_pass <- function(stage, fixed) lapply(panels, function(cp) {
    cp$stop_at <- stage; cp$fixed_axes <- fixed
    tryCatch(builder(cp), aui_axes = function(e) e)
  })
  got <- function(res, which) lapply(res, function(r) if (inherits(r, "aui_axes")) r[[which]] else NULL)
  fixed <- list()
  if (!Fc$free_x || !Fc$free_y) {
    for (stage in c("base", "final")) {                # "final" lets brackets raise the shared top
      res <- axes_pass(stage, fixed)
      if (!Fc$free_x) fixed$x <- union_axes(got(res, "x")) %||% fixed$x
      if (!Fc$free_y) fixed$y <- union_axes(got(res, "y")) %||% fixed$y
    }
  }
  parts <- lapply(panels, function(cp) { cp$collect <- TRUE; cp$fixed_axes <- fixed; builder(cp) })
  merged <- parts[[1]]
  merged$layers <- merge_layers(unlist(lapply(seq_along(parts), function(i) {
    tag_layers(parts[[i]]$layers, levels[i], levels)
  }), recursive = FALSE))
  merged$panels <- parts
  ctx$fixed_axes <- fixed
  finish(ctx, merged)
}
