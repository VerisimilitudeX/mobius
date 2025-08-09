#!/usr/bin/env Rscript
###############################################################################
# 03_verify.R
#
# Purpose:
#   Verify the final Beta CSV (row=samples, columns=probes + Condition)
#   matches the merged GenomicMethylSet in dimension, sample order, and Condition.
#
# Usage:
#   Rscript 03_verify.R
###############################################################################

rm(list=ls())
cat("\014")

library(minfi)

epi_root <- "/Volumes/T9/EpiMECoV"
genomic_rds_path <- file.path(epi_root,"processed_data","Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds")
beta_csv_updated <- file.path(epi_root,"processed_data","Beta_Transposed_with_Condition.csv")

cat("=== Checking merged GenomicMethylSet vs. Beta CSV ===\n\n")

if (!file.exists(genomic_rds_path)) {
  stop("[ERROR] RDS file not found => ", genomic_rds_path)
}
if (!file.exists(beta_csv_updated)) {
  stop("[ERROR] CSV file not found => ", beta_csv_updated)
}

mergedSet <- readRDS(genomic_rds_path)
cat("[INFO] mergedSet dimension =>", dim(mergedSet), "\n")
cat("Condition distribution =>\n")
print(table(colData(mergedSet)$Condition))

num_samples <- ncol(mergedSet)
num_probes  <- nrow(mergedSet)

beta_df <- read.csv(beta_csv_updated, row.names=1, check.names=FALSE)
cat("\n[INFO] Beta CSV dimension =>", dim(beta_df), "\n")

if (!("Condition" %in% colnames(beta_df))) {
  stop("[ERROR] No 'Condition' col found in Beta CSV.")
}

# Check #samples
if (nrow(beta_df) != num_samples) {
  warning("Mismatch in sample count: CSV has", nrow(beta_df),
          "where mergedSet has", num_samples)
} else {
  cat("[OK] #samples match.\n")
}
# Check #probes
if ((ncol(beta_df)-1) != num_probes) {
  warning("Mismatch in probe count: CSV has", (ncol(beta_df)-1),
          "where mergedSet has", num_probes)
} else {
  cat("[OK] #probes match (accounting for Condition as last col).\n")
}

# Check Condition alignment
csvCond <- as.character(beta_df$Condition)
setCond <- as.character(colData(mergedSet)$Condition)
if (length(csvCond) == length(setCond) && all(csvCond == setCond)) {
  cat("[OK] Condition vectors match exactly.\n")
} else {
  cat("[WARNING] Condition mismatch or partial mismatch.\n")
  idx <- which(csvCond != setCond)
  if (length(idx)>0) {
    cat("First mismatch at index =>", idx[1], 
        "Beta CSV cond=", csvCond[idx[1]], 
        "mergedSet cond=", setCond[idx[1]], "\n")
  }
}

cat("\n=== 03_verify done. ===\n")