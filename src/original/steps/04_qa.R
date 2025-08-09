#!/usr/bin/env Rscript
###############################################################################
# 04_qa_FAST.R (a faster drop-in replacement for 04_qa.R)
#
# PURPOSE:
#   Minimal QA: verifies that the merged GenomicMethylSet loads correctly,
#   and then does a quick read of Beta_Transposed_with_Condition.csv to
#   confirm dimension and distribution of Condition. 
#
#   This version uses data.table::fread to handle extremely wide CSVs quickly
#   (the user had 3 x ~415K columns, which can be painfully slow with base R).
#
# USAGE:
#   Rscript 04_qa_FAST.R
#
#   (Or rename it back to 04_qa.R if you prefer the same file name.)
###############################################################################

# Clear workspace and console
rm(list=ls())
cat("\014")

cat("=== Minimal QA Only Script Start ===\n", as.character(Sys.time()), "\n\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (1) Load needed packages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", repos="https://cloud.r-project.org/")
}
if (!requireNamespace("minfi", quietly = TRUE)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", repos="https://cloud.r-project.org/")
  }
  BiocManager::install("minfi")
}
library(data.table)
library(minfi)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (2) Define your paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
epi_root <- "/Volumes/T9/EpiMECoV"
processed_dir <- file.path(epi_root, "processed_data")

gmset_path <- file.path(processed_dir, "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds")
beta_csv   <- file.path(processed_dir, "Beta_Transposed_with_Condition.csv")

cat("[STEP] Loading RDS =>", gmset_path, "\n")
if (!file.exists(gmset_path)) {
  stop("Could not find GenomicMethylSet RDS => ", gmset_path)
}

gmset <- readRDS(gmset_path)
cat("[INFO] Dimension =>", nrow(gmset), "x", ncol(gmset), "\n")
cat("[INFO] Condition distribution =>\n\n")
print(table(colData(gmset)$Condition))
cat("\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (3) Read the huge Beta CSV using data.table::fread (multi-threaded)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("[STEP] Loading Beta CSV =>", beta_csv, "\n\n")
if (!file.exists(beta_csv)) {
  stop("Beta CSV not found => ", beta_csv)
}

# Adjust nThread= to match your Mac’s core count (e.g., 8, 10, 16).
# data.table automatically uses a default that is often good enough, 
# but you can override it explicitly:
nCores <- 16  # or something that matches your Apple M4 Max
cat("[INFO] Using data.table::fread with nThread =", nCores, "\n")

# Because your CSV is 3 rows × ~415,000 columns, the main challenge 
# is the extremely wide dimension. data.table's fread is highly optimized 
# for this scenario.
DT <- fread(
  input       = beta_csv,
  sep         = ",",
  header      = TRUE,
  nThread     = nCores,
  verbose     = FALSE,
  showProgress= TRUE
)

cat("[INFO] Dimension =>", nrow(DT), "x", ncol(DT), "\n\n")

# If 'Condition' is your last column, let’s confirm distribution:
if (!("Condition" %in% names(DT))) {
  cat("[WARNING] No 'Condition' col found in CSV.\n")
} else {
  cond_vec <- DT[["Condition"]]
  cat("[INFO] Condition distribution =>\n")
  print(table(cond_vec))
  cat("\n")
}

cat("=== Minimal QA is complete. ===\n\n")