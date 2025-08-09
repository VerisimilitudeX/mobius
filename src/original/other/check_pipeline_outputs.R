#!/usr/bin/env Rscript

################################################################################
# check_pipeline_outputs.R
# Purpose: Automated checks to confirm the pipeline ran successfully and
#          generated all expected outputs.
#
# Usage: Rscript check_pipeline_outputs.R
################################################################################

library(minfi)

expected_files <- c(
  "450K_BetaValues.csv",
  "450K_Combined_RGChannelSet.rds",
  "450K_Final_GenomicMethylSet.rds",
  "EPIC_BetaValues.csv",
  "EPIC_Combined_RGChannelSet.rds",
  "EPIC_Final_GenomicMethylSet.rds",
  "Merged_450K_EPIC_BetaValues_with_Condition.csv",
  "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds"
)

missing_files <- character()
cat("Checking for expected output files...\n")
for (f in expected_files) {
  if (!file.exists(f)) {
    cat(sprintf("[MISSING] %s\n", f))
    missing_files <- c(missing_files, f)
  } else {
    cat(sprintf("[OK]      %s\n", f))
  }
}

if (length(missing_files) > 0) {
  cat("\nWARNING: Some expected files are missing!\n")
  print(missing_files)
  cat("Please verify your pipeline.\n\n")
} else {
  cat("\nAll expected files found.\n\n")
}

safeCheckRDS <- function(filepath) {
  cat(sprintf("   Checking RDS file: %s\n", filepath))
  obj <- readRDS(filepath)
  if (inherits(obj, "RGChannelSet")) {
    cat(sprintf("   --> RGChannelSet found. Dimensions: %d features x %d samples\n",
                nrow(obj), ncol(obj)))
  } else if (inherits(obj, "GenomicMethylSet")) {
    cat(sprintf("   --> GenomicMethylSet found. Dimensions: %d features x %d samples\n",
                nrow(obj), ncol(obj)))
    col_info <- colData(obj)
    if ("Condition" %in% colnames(col_info)) {
      cat("   --> 'Condition' column detected in colData.\n")
    } else {
      cat("   --> WARNING: 'Condition' column not found in colData!\n")
    }
  } else {
    cat("   --> Unrecognized object type.\n")
}

safeCheckCSV <- function(filepath, check_header = TRUE) {
  cat(sprintf("   Checking CSV file: %s\n", filepath))
  df <- read.csv(filepath, header = check_header, row.names = 1)
  cat(sprintf("   --> CSV read success. Dimensions: %d rows x %d cols\n",
              nrow(df), ncol(df)))
  cat("   --> First few row names:\n")
  print(head(rownames(df)))
  cat("\n")
}

for (f in expected_files) {
  if (!file.exists(f)) next
  ext <- tools::file_ext(f)
  if (tolower(ext) == "rds") {
    safeCheckRDS(f)
  } else if (tolower(ext) == "csv") {
    safeCheckCSV(f, check_header = TRUE)
  }
}

cat("Done. If no major warnings, your outputs look good!\n")