# Persistent R worker for Analysis UI.
#
# Protocol: one line on stdin = path of a JSON request file. The reply is
# written to request$out and a sentinel line "@@AUI-DONE <id>" is printed.
# Everything else R prints goes to stderr.

suppressWarnings(suppressPackageStartupMessages({
  library(jsonlite)
  library(stats)
  library(utils)
}))

AUI_R_DIR <- Sys.getenv("AUI_R_DIR", unset = ".")
AUI_SHARED_DIR <- Sys.getenv("AUI_SHARED_DIR", unset = file.path(AUI_R_DIR, "..", "shared"))
options(warn = 1, stringsAsFactors = FALSE, scipen = 0)

real_stdout <- stdout()
sink(stderr(), type = "output")

for (f in sort(list.files(file.path(AUI_R_DIR, "lib"), pattern = "\\.R$", full.names = TRUE))) {
  source(f, local = FALSE, encoding = "UTF-8")
}

.aui_cache <- new.env()

# Scripts are loaded once into their own environment (whose parent is the
# global env, so lib helpers are visible) and re-sourced when the file changes.
aui_load <- function(subdir, script) {
  path <- file.path(AUI_R_DIR, subdir, script)
  if (!file.exists(path)) stop(sprintf("R script not found: %s/%s", subdir, script))
  key <- paste(subdir, script)
  mtime <- as.numeric(file.info(path)$mtime)
  hit <- .aui_cache[[key]]
  if (is.null(hit) || hit$mtime != mtime) {
    env <- new.env(parent = globalenv())
    source(path, local = env, encoding = "UTF-8")
    hit <- list(env = env, mtime = mtime)
    .aui_cache[[key]] <- hit
  }
  hit$env
}

aui_handle <- function(req) {
  kind <- req$kind
  if (identical(kind, "ping")) return("pong")
  if (identical(kind, "info")) {
    pkgs <- c("ggplot2", "svglite", "emmeans", "lme4", "lmerTest", "car", "survival",
              "multcomp", "afex", "ggpubr", "ggbeeswarm", "ggsignif", "pROC", "rstatix",
              "patchwork", "ggrepel", "viridis", "scales", "minpack.lm", "nlme",
              "survminer", "ggridges", "effectsize", "broom", "pheatmap", "ggforce",
              "cowplot", "MASS", "nortest", "ggthemes", "ggsci", "pwr", "psych", "ragg",
              "gridExtra", "coin", "epitools", "haven", "readxl")
    have <- vapply(pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
    return(list(version = paste(R.version$major, R.version$minor, sep = "."),
                packages = as.list(have)))
  }
  if (identical(kind, "analysis")) {
    env <- aui_load("analyses", req$script)
    data <- aui_frame(req$data)
    res <- aui_results_begin()
    out <- env$run(data, req$options)
    return(aui_results_end(out))
  }
  if (identical(kind, "graph")) {
    return(aui_render_graph(req))
  }
  if (identical(kind, "read")) {
    return(aui_read_file(req$path))
  }
  stop(sprintf("Unknown request kind: %s", kind))
}

con <- file("stdin", open = "r")
repeat {
  line <- readLines(con, n = 1, warn = FALSE)
  if (length(line) == 0) break
  line <- trimws(line)
  if (!nzchar(line)) next
  req <- NULL
  reply <- tryCatch({
    req <- jsonlite::fromJSON(line, simplifyVector = TRUE, simplifyDataFrame = FALSE,
                              simplifyMatrix = FALSE)
    warns <- character(0)
    result <- withCallingHandlers(
      aui_handle(req),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    if (is.list(result) && !is.null(result$elements) && length(warns)) {
      result$warnings <- as.list(unique(c(unlist(result$warnings), warns)))
    }
    list(status = "ok", result = result)
  }, error = function(e) {
    list(status = "error", message = conditionMessage(e))
  })
  out_path <- if (!is.null(req) && !is.null(req$out)) req$out else sub("\\.req\\.json$", ".out.json", line)
  rid <- if (!is.null(req) && !is.null(req$id)) req$id else sub("\\.req\\.json$", "", basename(line))
  json <- tryCatch(
    jsonlite::toJSON(reply, auto_unbox = TRUE, na = "null", null = "null", digits = NA, force = TRUE),
    error = function(e) jsonlite::toJSON(list(status = "error",
      message = paste("Could not serialise the R result:", conditionMessage(e))), auto_unbox = TRUE))
  writeLines(json, out_path, useBytes = TRUE)
  cat(sprintf("@@AUI-DONE %s\n", rid), file = real_stdout)
  flush(real_stdout)
}
