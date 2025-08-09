#!/usr/bin/env Rscript
############################################################
# 13_summarize_findings.R
#
# Summarize final pipeline results:
#  - If row=sample and there's a Condition col, do a quick PCA
#    on the numeric columns (excluding Condition).
#  - Save "pca_final.png" in the results folder if feasible.
#
# Usage:
#   Rscript 13_summarize_findings.R
############################################################

suppressMessages({
  library(data.table)
  library(ggplot2)
  library(dplyr)
})

cat("=== Summarize Findings Script ===\n")

epi_root <- "/Volumes/T9/EpiMECoV"
results_dir <- file.path(epi_root, "results")

final_csv_1 <- file.path(epi_root, "processed_data", "transformed_data.csv")
final_csv_2 <- file.path(epi_root, "processed_data", "filtered_biomarker_matrix.csv")

use_file <- NA
if (file.exists(final_csv_1)) {
  use_file <- final_csv_1
} else if (file.exists(final_csv_2)) {
  use_file <- final_csv_2
} else {
  stop("[ERROR] No final CSV found (neither transformed_data.csv nor filtered_biomarker_matrix.csv).")
}

cat("[INFO] Using =>", use_file, "\n")

df <- fread(use_file)
cat("[INFO] dimension =>", dim(df), "\n")

if (!("Condition" %in% colnames(df))) {
  cat("[ERROR] no Condition col => cannot do PCA by group.\n")
  quit(status=1)
}

# The typical pipeline has row=sample, so #rows = ~678, #cols = 64 (or 4470).
if (nrow(df) < ncol(df)) {
  cat("[INFO] row=sample => attempting PCA.\n")
  
  # Exclude Condition + any non-numeric columns:
  numeric_cols <- df %>%
    select(where(is.numeric)) %>%
    colnames()

  # if Condition was numeric for some reason, remove:
  if ("Condition" %in% numeric_cols) {
    numeric_cols <- numeric_cols[numeric_cols != "Condition"]
  }
  
  if (length(numeric_cols) < 2) {
    cat("[WARN] Not enough numeric columns for PCA.\n")
    quit(status=0)
  }
  
  mat <- as.matrix(df[, ..numeric_cols])

  # Must ensure the row count > col count for typical prcomp usage:
  # If you have more columns than rows, you can still do prcomp, but let's proceed:
  pca_res <- prcomp(mat, scale. = TRUE)
  pc_df <- as.data.frame(pca_res$x[,1:2])
  colnames(pc_df) <- c("PC1","PC2")

  pc_df$Condition <- df$Condition

  p <- ggplot(pc_df, aes(x=PC1, y=PC2, color=Condition)) +
    geom_point(alpha=0.7) +
    theme_minimal() +
    labs(title="PCA of Final Data", x="PC1", y="PC2")

  out_png <- file.path(results_dir, "pca_final.png")
  ggsave(out_png, p, width=6, height=5)
  cat("[SAVED PCA plot]", out_png, "\n")
  
} else {
  cat("[INFO] Data suggests row=probes => skipping PCA.\n")
}

cat("=== Summarize Findings Done. ===\n")