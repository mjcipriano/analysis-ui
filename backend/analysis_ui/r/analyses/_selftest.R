# Used by the test-suite to check the Python <-> R bridge.
run <- function(data, options) {
  aui_table("Means", aggregate(value ~ group, data, mean), formats = c(value = "num"))
  aui_text(sprintf("n = %d", nrow(data)))
  aui_need("ggplot2")
  p <- ggplot2::ggplot(data, ggplot2::aes(group, value)) + ggplot2::geom_boxplot() + aui_theme()
  aui_plot("Box", p)
}
