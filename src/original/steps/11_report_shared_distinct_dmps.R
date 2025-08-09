#!/usr/bin/env Rscript
###############################################################################
# report_shared_distinct_dmps.R
#
# Purpose:
#   - Reads your three DMP CSVs (ME vs Control, LC vs Control, ME vs LC).
#   - Identifies:
#       * set of ME_CpGs
#       * set of LC_CpGs
#       * set of ME_LC_CpGs
#     and prints:
#       * Shared in all three
#       * Distinct to ME vs Ctrl only
#       * Distinct to LC vs Ctrl only
#       * Distinct to ME vs LC only
#   - Saves them in separate text files for your reference.
#
# Usage:
#   Rscript report_shared_distinct_dmps.R
###############################################################################
library(data.table)

epi_root <- "/Volumes/T9/EpiMECoV"
res_dir  <- file.path(epi_root, "results")

dmp_me_ctrl <- file.path(res_dir, "DMP_ME_vs_Control.csv")
dmp_lc_ctrl <- file.path(res_dir, "DMP_LC_vs_Control.csv")
dmp_me_lc   <- file.path(res_dir, "DMP_ME_vs_LC.csv")

if (!all(file.exists(c(dmp_me_ctrl, dmp_lc_ctrl, dmp_me_lc)))) {
  stop("One or more DMP CSVs are missing from results/ folder.")
}

df_me_ctrl <- fread(dmp_me_ctrl)
df_lc_ctrl <- fread(dmp_lc_ctrl)
df_me_lc   <- fread(dmp_me_lc)

# We assume the first column is the CpG or rownames
cpg_me_ctrl <- df_me_ctrl[[1]]
cpg_lc_ctrl <- df_lc_ctrl[[1]]
cpg_me_lc   <- df_me_lc[[1]]

set_me_ctrl <- unique(cpg_me_ctrl)
set_lc_ctrl <- unique(cpg_lc_ctrl)
set_me_lc   <- unique(cpg_me_lc)

# Intersection / union
shared_all   <- Reduce(intersect, list(set_me_ctrl, set_lc_ctrl, set_me_lc))
distinct_me_ctrl <- set_me_ctrl[ !(set_me_ctrl %in% c(set_lc_ctrl, set_me_lc)) ]
distinct_lc_ctrl <- set_lc_ctrl[ !(set_lc_ctrl %in% c(set_me_ctrl, set_me_lc)) ]
distinct_me_lc   <- set_me_lc[   !(set_me_lc   %in% c(set_me_ctrl, set_lc_ctrl)) ]

cat("\n=== Shared in all three comparisons:", length(shared_all), "CpGs ===\n")
cat("Some examples:\n")
print(head(shared_all, 10))

cat("\nME vs Ctrl only (distinct):", length(distinct_me_ctrl), "\n")
cat("LC vs Ctrl only (distinct):", length(distinct_lc_ctrl), "\n")
cat("ME vs LC only (distinct):",   length(distinct_me_lc),   "\n")

fwrite(data.table(CpG=shared_all), file.path(res_dir, "Shared_in_all_three.txt"))
fwrite(data.table(CpG=distinct_me_ctrl), file.path(res_dir, "Distinct_ME_Ctrl_only.txt"))
fwrite(data.table(CpG=distinct_lc_ctrl), file.path(res_dir, "Distinct_LC_Ctrl_only.txt"))
fwrite(data.table(CpG=distinct_me_lc),   file.path(res_dir, "Distinct_ME_LC_only.txt"))

cat("\n=== Shared/Distinct DMP lists saved in 'results/' directory. ===\n")