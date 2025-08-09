#!/usr/bin/env Rscript
############################################################
# 08_prepare_final_data.R
#
# This script:
# 1) Loads Beta_Transposed_with_Condition.csv (row=samples, columns=probes + Condition)
# 2) Removes rows with too many NAs or extremely low variance
#    (but uses a less strict filter than before).
# 3) Saves "filtered_biomarker_matrix.csv"
#
# Usage:
#   Rscript 08_prepare_final_data.R
############################################################

library(data.table)
library(dplyr)

cat("=== Step 8) Additional data filtering (less aggressive) ===\n")

epi_root <- "/Volumes/T9/EpiMECoV"
beta_path <- file.path(epi_root, "processed_data", "Beta_Transposed_with_Condition.csv")
if (!file.exists(beta_path)) {
  stop("Beta CSV not found => ", beta_path)
}

# FIX: Use read.csv with row.names=1 so that the sample IDs (rownames) are preserved.
df <- read.csv(beta_path, row.names = 1, check.names = FALSE)
cat("[INFO] dimension =>", dim(df), "\n")

if (!("Condition" %in% colnames(df))) {
  cat("[WARN] 'Condition' column not found. We will proceed but the pipeline may need it.\n")
} else {
  cat("[INFO] 'Condition' column found.\n")
}

cond_vector <- NULL
if ("Condition" %in% colnames(df)) {
  cond_vector <- df$Condition
}

# Remove the Condition column to work on numeric data only.
df_noCond <- df[, !colnames(df) %in% "Condition", drop = FALSE]
cat("[INFO] dimension without Condition =>", dim(df_noCond), "\n")

# Filtering: allow up to 70% missing values per sample (row).
threshold_na <- 0.9
row_na_frac <- apply(df_noCond, 1, function(x) mean(is.na(x)))
keep_na <- which(row_na_frac < threshold_na)
df_filtered <- df_noCond[keep_na, ]

# Filter rows (samples) with extremely low variance.
threshold_var <- 1e-9
rowvars <- apply(df_filtered, 1, var, na.rm = TRUE)
keep_var <- which(rowvars >= threshold_var)
df_filtered <- df_filtered[keep_var, ]

cat("[INFO] After filtering =>", dim(df_filtered), "\n")

# Re-append the Condition vector.
if (!is.null(cond_vector)) {
  final_indices <- keep_na[keep_var]
  df_filtered$Condition <- cond_vector[final_indices]
}

out_path <- file.path(epi_root, "processed_data", "filtered_biomarker_matrix.csv")
# Write with row names so that later scripts can recover the sample IDs.
write.csv(df_filtered, out_path, row.names = TRUE, quote = FALSE)
cat("[SAVED] =>", out_path, "\n")
cat("=== Done preparing final data with a less strict filter. ===\n")