#!/usr/bin/env Rscript
###############################################################################
# 09_fix_step_10.R
#
# Optional debugging script to show overlaps between DMP probe IDs and
# your Beta matrix. Not used in the normal pipeline, but helpful if 
# you get "no overlap" or "Condition mismatch" errors.
###############################################################################

library(data.table)
library(dplyr)

cat("=== 09_fix_step_10 debugging script ===\n")

epi_root <- "/Volumes/T9/EpiMECoV"

# 1) load filtered beta (row=probes, columns=samples, possibly with a 'CpG' col)
beta_file <- file.path(epi_root, "processed_data", "filtered_biomarker_matrix.csv")
beta_data <- fread(beta_file)

if ("V1" %in% colnames(beta_data)) {
  setnames(beta_data, old="V1", new="CpG", skip_absent=TRUE)
}
cat("Beta data dimension =>", dim(beta_data), "\n")
cat("Beta data first row =>\n")
print(head(beta_data,1))

# 2) load DMP
results_dir <- file.path(epi_root,"results")
dmp_me_ctrl_file <- file.path(results_dir,"DMP_ME_vs_Control.csv")
dmp_lc_ctrl_file <- file.path(results_dir,"DMP_LC_vs_Control.csv")
dmp_me_lc_file   <- file.path(results_dir,"DMP_ME_vs_LC.csv")

dmp_me_ctrl <- fread(dmp_me_ctrl_file)
dmp_lc_ctrl <- fread(dmp_lc_ctrl_file)
dmp_me_lc   <- fread(dmp_me_lc_file)

for (dmp_df in list(dmp_me_ctrl, dmp_lc_ctrl, dmp_me_lc)) {
  if ("V1" %in% colnames(dmp_df)) {
    setnames(dmp_df, old="V1", new="CpG", skip_absent=TRUE)
  }
}

# top 2k
topN <- 2000
top_me_ctrl <- head(dmp_me_ctrl[order(dmp_me_ctrl$P.Value)], topN)
top_lc_ctrl <- head(dmp_lc_ctrl[order(dmp_lc_ctrl$P.Value)], topN)
top_me_lc   <- head(dmp_me_lc[order(dmp_me_lc$P.Value)],   topN)

all_cpgs <- unique(c(top_me_ctrl$CpG, top_lc_ctrl$CpG, top_me_lc$CpG))
cat("Unique CpGs from top hits =>", length(all_cpgs), "\n")

if (!"CpG" %in% colnames(beta_data)) {
  cat("[ERROR] Beta data lacks 'CpG' col => can't do overlap.\n")
} else {
  overlap <- intersect(all_cpgs, beta_data$CpG)
  cat("Overlap =>", length(overlap), "CpGs in Beta matrix.\n")
  cat("Example of overlap:\n")
  print(head(overlap, 10))
}

cat("=== Debug script done. ===\n")