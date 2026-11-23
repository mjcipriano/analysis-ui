# Core helpers available to every R analysis and graph script.

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# A user-facing error (message is shown verbatim in the results panel).
aui_stop <- function(...) stop(paste0(...), call. = FALSE)

aui_need <- function(...) {
  for (p in c(...)) {
    if (!requireNamespace(p, quietly = TRUE)) {
      aui_stop("The R package '", p, "' is required for this but is not installed. ",
               "Install it in R with install.packages(\"", p, "\").")
    }
  }
  invisible(TRUE)
}

# ---- data ----------------------------------------------------------------

# Column payload (see dataset.frame_to_r) -> data.frame. Numeric columns are
# doubles, factor columns are factors with the level order chosen in the UI.
aui_frame <- function(x) {
  if (is.null(x) || is.null(x$columns) || length(x$columns) == 0) return(data.frame())
  n <- x$nrow %||% 0
  cols <- list()
  for (col in x$columns) {
    v <- col$values
    if (is.null(v) || length(v) == 0) v <- rep(NA, n)
    if (is.list(v)) v <- vapply(v, function(e) if (is.null(e)) NA else e, FUN.VALUE = if (identical(col$kind, "numeric")) numeric(1) else character(1))
    if (identical(col$kind, "numeric")) {
      v <- as.numeric(v)
    } else if (identical(col$kind, "factor")) {
      v <- factor(as.character(v), levels = as.character(unlist(col$levels)),
                  ordered = FALSE)
    } else {
      v <- as.character(v)
    }
    cols[[col$name]] <- v
  }
  as.data.frame(cols, check.names = FALSE, stringsAsFactors = FALSE)
}

# Option accessors -----------------------------------------------------------
aui_opt <- function(options, name, default = NULL) {
  v <- options[[name]]
  if (is.null(v) || (length(v) == 1 && is.character(v) && !nzchar(v))) default else v
}
aui_flag <- function(options, name, default = FALSE) isTRUE(aui_opt(options, name, default))
aui_num <- function(options, name, default = NA_real_) {
  v <- aui_opt(options, name, NULL)
  if (is.null(v)) default else suppressWarnings(as.numeric(v))
}
# multi-variable options arrive as character vectors (or list() when empty)
aui_vars <- function(options, name) as.character(unlist(options[[name]] %||% character(0)))

# Backtick-quote a column name for use in formulas.
aui_q <- function(x) paste0("`", gsub("`", "", x), "`")

# ---- results -------------------------------------------------------------

.aui_res <- new.env()

aui_results_begin <- function() {
  .aui_res$elements <- list()
  invisible(NULL)
}

aui_results_end <- function(out = NULL) {
  # A script may either use the aui_* emitters (recommended) or return
  # list(elements = ...) itself.
  if (is.list(out) && !is.null(out$elements)) return(out)
  list(elements = .aui_res$elements)
}

.aui_push <- function(el) {
  .aui_res$elements[[length(.aui_res$elements) + 1]] <- el
  invisible(NULL)
}

aui_heading <- function(text) .aui_push(list(type = "heading", text = text))

aui_text <- function(content, title = NULL, kind = "note") {
  .aui_push(list(type = "text", title = title, content = paste(content, collapse = "\n"), kind = kind))
}
aui_summary <- function(content) aui_text(content, kind = "summary")
aui_warn <- function(content) aui_text(content, kind = "warning")

# aui_table("Title", df, formats = c(p = "p", n = "int"), labels = c(est = "Estimate"),
#           footnotes = c("..."))
# Every data-frame column becomes a table column. Default format: "num" for
# numeric columns, "text" otherwise. Formats: text, num, int, p, pct, stars.
aui_table <- function(title, df, formats = NULL, labels = NULL, footnotes = NULL) {
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  rownames(df) <- NULL
  cols <- lapply(names(df), function(k) {
    fmt <- if (!is.null(formats) && k %in% names(formats)) formats[[k]]
           else if (is.numeric(df[[k]])) "num" else "text"
    lab <- if (!is.null(labels) && k %in% names(labels)) labels[[k]] else k
    list(key = k, label = lab, format = fmt)
  })
  for (k in names(df)) {
    if (is.factor(df[[k]])) df[[k]] <- as.character(df[[k]])
    if (is.numeric(df[[k]])) df[[k]][!is.finite(df[[k]])] <- NA
  }
  rows <- lapply(seq_len(nrow(df)), function(i) {
    r <- lapply(names(df), function(k) { v <- df[[k]][i]; if (is.na(v)) NULL else v })
    names(r) <- names(df)
    r
  })
  .aui_push(list(type = "table", title = title, columns = cols, rows = rows,
                 footnotes = as.list(as.character(footnotes %||% character(0)))))
}

# Attach a ggplot (or a function that draws with base graphics) as an SVG.
aui_plot <- function(title, plot, width = 6, height = 4.2) {
  path <- tempfile(fileext = ".svg")
  on.exit(unlink(path), add = TRUE)
  grDevices::svg(path, width = width, height = height)
  ok <- tryCatch({
    if (is.function(plot)) plot() else print(plot)
    TRUE
  }, finally = grDevices::dev.off())
  svg <- paste(readLines(path, warn = FALSE), collapse = "\n")
  .aui_push(list(type = "plot", title = title, svg = svg))
}

# Offer a one-click editable graph of the same data.
aui_graph <- function(label, graph_type, options = list()) {
  .aui_push(list(type = "graph", label = label,
                 graph = list(type = graph_type, options = options)))
}

aui_stars <- function(p) {
  ifelse(is.na(p), "", ifelse(p < 1e-4, "****", ifelse(p < 1e-3, "***",
    ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "ns")))))
}

aui_fmt_p <- function(p) {
  ifelse(is.na(p), "n/a", ifelse(p < 1e-4, "<0.0001", formatC(p, format = "f", digits = 4)))
}

# House style for diagnostic plots shown inside results.
aui_theme <- function(base_size = 11) {
  ggplot2::theme_classic(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = base_size + 1),
      axis.text = ggplot2::element_text(colour = "#222222"),
      axis.line = ggplot2::element_line(linewidth = 0.5, colour = "#222222"),
      axis.ticks = ggplot2::element_line(linewidth = 0.5, colour = "#222222"),
      legend.position = "top",
      legend.title = ggplot2::element_blank(),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold"))
}
AUI_COLORS <- c("#3B6FE0", "#E4572E", "#17A589", "#F2A541", "#8E5BD9", "#D6456F", "#4FB0E5", "#7A8B99")

# Read a data file with R (rds / RData / sav / ...). Returns columns.
aui_read_file <- function(path) {
  ext <- tolower(tools::file_ext(path))
  df <- if (ext == "rds") readRDS(path)
        else if (ext %in% c("rdata", "rda")) { e <- new.env(); nm <- load(path, envir = e); e[[nm[1]]] }
        else aui_stop("Unsupported file type for the R reader: ", ext)
  df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
  cols <- lapply(names(df), function(k) {
    v <- df[[k]]
    if (is.factor(v)) list(name = k, type = if (is.ordered(v)) "ordinal" else "nominal",
                           values = as.list(as.character(v)), levels = as.list(levels(v)))
    else if (is.numeric(v)) list(name = k, type = "continuous", values = as.list(ifelse(is.finite(v), v, NA)))
    else list(name = k, type = "nominal", values = as.list(as.character(v)))
  })
  list(columns = cols)
}
