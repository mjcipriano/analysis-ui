# ggplot2 graph engine — entry point.
#
# The engine itself lives in r/graphs/*.R. Those files are sourced, in name
# order, into ONE shared environment (so helpers in 00_common.R are visible
# to every renderer) and are re-sourced automatically whenever any of them
# changes, which makes editing a renderer hot-reloadable without restarting
# the worker.
#
#   aui_render_graph(req)   req = list(type, options, frames, computed,
#                                      format, dpi, editable_text, file)

# Text in graphs is UTF-8 (µ, ρ, −, ≥ ...). In a C/POSIX locale R would
# escape such characters when measuring or drawing text.
if (!isTRUE(l10n_info()[["UTF-8"]])) {
  for (.loc in c("C.UTF-8", "C.utf8", "en_US.UTF-8", "en_US.utf8")) {
    if (nzchar(suppressWarnings(Sys.setlocale("LC_CTYPE", .loc)))) break
  }
  rm(.loc)
}

.aui_gg <- new.env()

# Attach ggplot2 and pre-load the namespaces the renderers use, so the first
# graph is not slowed down by package loading.
.aui_gg$ready <- requireNamespace("ggplot2", quietly = TRUE)
if (.aui_gg$ready) {
  suppressWarnings(suppressPackageStartupMessages(library(ggplot2)))
  for (.p in c("scales", "systemfonts", "ragg", "beeswarm", "ggrepel", "patchwork", "svglite")) {
    try(suppressWarnings(suppressPackageStartupMessages(loadNamespace(.p))), silent = TRUE)
  }
  rm(.p)
}

aui_gg_env <- function() {
  dir <- file.path(AUI_R_DIR, "graphs")
  files <- sort(list.files(dir, pattern = "\\.R$", full.names = TRUE))
  stamp <- paste(files, as.numeric(file.info(files)$mtime), collapse = "|")
  if (!identical(.aui_gg$stamp, stamp)) {
    env <- new.env(parent = globalenv())
    for (f in files) sys.source(f, envir = env)
    .aui_gg$env <- env
    .aui_gg$stamp <- stamp
  }
  .aui_gg$env
}

aui_render_graph <- function(req) {
  if (!isTRUE(.aui_gg$ready)) {
    aui_stop("The R package 'ggplot2' is required for the ggplot2 graph engine but is not installed.")
  }
  aui_gg_env()$render(req)
}
