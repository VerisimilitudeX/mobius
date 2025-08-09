###############################################################################
# fix_condition_only.R
#
# If Condition is missing or misaligned in your Beta CSV, this script merges
# them from your GenomicMethylSet's colData. 
###############################################################################

rm(list=ls())
cat("\014")

library(minfi)

merged_rds_path  <- "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds"
beta_csv_path    <- "Merged_450K_EPIC_BetaValues_with_Condition.csv"

cat("=== Loading merged GenomicMethylSet ===\n")
if (!file.exists(merged_rds_path)) {
  stop("Merged RDS file not found: ", merged_rds_path)
}
mergedSet <- readRDS(merged_rds_path)
cat("Loaded mergedSet. Dimension:", dim(mergedSet), "\n")

cat("Condition distribution:\n")
print(table(colData(mergedSet)$Condition))

cat("\n=== Loading Beta CSV ===\n")
if (!file.exists(beta_csv_path)) {
  stop("Beta CSV file not found: ", beta_csv_path)
}
beta_values <- read.csv(beta_csv_path, row.names=1, check.names=FALSE)
cat("Loaded Beta CSV. Dimension:", dim(beta_values), "\n")

if (!("Condition" %in% colnames(beta_values))) {
  cat("No Condition column found. Manually appending...\n")
  beta_values_t <- t(beta_values)
  cat("After transpose => dimension:", dim(beta_values_t), "\n")
  df <- as.data.frame(beta_values_t)
  df$Condition <- as.character(colData(mergedSet)$Condition)
  out_path <- "Merged_450K_EPIC_BetaValues_with_Condition_Updated.csv"
  write.csv(df, out_path, quote=FALSE)
  cat("Saved updated =>", out_path, "\n")
} else {
  cat("Condition column already present.\n")
}
cat("Done.\n")