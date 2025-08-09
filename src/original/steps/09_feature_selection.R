#!/usr/bin/env Rscript
###############################################################################
# 09_feature_selection.R
#
# Step 9) Feature Selection + Condition Restoration
#
# This script:
#   (1) Reads the "filtered_biomarker_matrix.csv" (from step 8) – a large matrix
#       with row=probes, columns=samples, plus a "Condition" column repeated in
#       each row.
#   (2) Reads the top DMPs from the three contrasts (DMP_ME_vs_Control.csv, etc.)
#   (3) Combines (union) the top CpG IDs across these DMP results
#   (4) Subsets the big matrix to those CpGs only
#   (5) Re-appends the Condition vector as the final column
#   (6) Writes out "feature_selected_matrix.csv"
#
#   NEW: Also writes out a separate "all_cpg_matrix.csv" that keeps ALL CpGs
#
# USAGE:
#   Rscript 09_feature_selection.R
###############################################################################

suppressPackageStartupMessages({
  library(dplyr)
  # Note: We now use base R’s read.csv() instead of data.table::fread()
})

cat("=== Step 9) Feature Selection + Condition Restoration ===\n\n")

# 1) Define paths
epi_root <- "/Volumes/T9/EpiMECoV"
processed_dir <- file.path(epi_root, "processed_data")
results_dir   <- file.path(epi_root, "results")

filtered_csv <- file.path(processed_dir, "filtered_biomarker_matrix.csv")
if (!file.exists(filtered_csv)) {
  stop("[ERROR] Could not find filtered_biomarker_matrix.csv =>", filtered_csv)
}

# 2) Read the big Beta matrix using read.csv (instead of fread)
cat("[INFO] Reading Beta data using read.csv ...\n")
beta_big <- read.csv(filtered_csv, row.names = 1, stringsAsFactors = FALSE)
cat("[INFO] Beta data dimension:", dim(beta_big), "\n")

# === NEW PART: also save a copy with ALL CpGs (no subsetting) ===
all_cpg_out <- file.path(processed_dir, "all_cpg_matrix.csv")
write.csv(beta_big, all_cpg_out, quote = FALSE)
cat("[INFO] Also wrote out full CpG data to =>", all_cpg_out, "\n")

# 3) Read the top DMP CpG IDs from the three contrast files and take their union
dmp_me_ctrl_file <- file.path(results_dir, "DMP_ME_vs_Control.csv")
dmp_lc_ctrl_file <- file.path(results_dir, "DMP_LC_vs_Control.csv")
dmp_me_lc_file   <- file.path(results_dir, "DMP_ME_vs_LC.csv")

cpg_union <- c()
if (file.exists(dmp_me_ctrl_file)) {
  dmp1 <- read.csv(dmp_me_ctrl_file, row.names = 1, stringsAsFactors = FALSE)
  cpg1 <- rownames(dmp1)
  if (!is.null(cpg1)) {
    cpg_union <- unique(c(cpg_union, cpg1))
  }
}
if (file.exists(dmp_lc_ctrl_file)) {
  dmp2 <- read.csv(dmp_lc_ctrl_file, row.names = 1, stringsAsFactors = FALSE)
  cpg2 <- rownames(dmp2)
  cpg_union <- unique(c(cpg_union, cpg2))
}
if (file.exists(dmp_me_lc_file)) {
  dmp3 <- read.csv(dmp_me_lc_file, row.names = 1, stringsAsFactors = FALSE)
  cpg3 <- rownames(dmp3)
  cpg_union <- unique(c(cpg_union, cpg3))
}
cat("[INFO] Unique DMP CpGs:", length(cpg_union), "\n")

if (length(cpg_union) < 1) {
  cat("[WARN] No DMPs found => skipping feature selection.\n")
  out_path <- file.path(processed_dir, "feature_selected_matrix.csv")
  write.csv(beta_big, out_path, quote = FALSE)
  cat("[SAVED] =>", out_path, "\n")
  quit(status = 0)
}

# 4) Subset the big Beta matrix using the row names as probe IDs
probe_ids <- rownames(beta_big)
if (length(probe_ids) == 0) {
  stop("[ERROR] Could not identify the probe IDs in the row names.")
}

keep_idx <- which(probe_ids %in% cpg_union)
subset_mat <- beta_big[keep_idx, , drop = FALSE]
cat("[INFO] Post-subset dimension:", dim(subset_mat), "\n")

# 5) Re-append the Condition vector.
# (Assuming the original beta_big has a "Condition" column)
cond_vec <- beta_big$Condition
final_df <- subset_mat
final_df$Condition <- cond_vec[keep_idx]

# 6) Write out the final feature-selected matrix.
out_file <- file.path(processed_dir, "feature_selected_matrix.csv")
write.csv(final_df, out_file, quote = FALSE)
cat("[SAVED] =>", out_file, "\n")

cat("\n=== Feature Selection step complete. ===\n")
