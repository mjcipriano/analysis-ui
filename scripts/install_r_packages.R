# Install the R packages used by Analysis UI (run once):  Rscript scripts/install_r_packages.R
pkgs <- c("jsonlite", "ggplot2", "svglite", "ragg", "systemfonts", "scales", "patchwork", "cowplot",
          "gridExtra", "ggbeeswarm", "ggsignif", "ggrepel", "ggridges", "ggforce", "ggpubr", "survminer",
          "viridis", "emmeans", "lme4", "lmerTest", "car", "afex", "multcomp", "rstatix", "effectsize",
          "broom", "survival", "nlme", "MASS", "minpack.lm", "nortest", "pROC", "psych", "pwr", "coin",
          "epitools", "dplyr", "tidyr")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All R packages are already installed.")
}
