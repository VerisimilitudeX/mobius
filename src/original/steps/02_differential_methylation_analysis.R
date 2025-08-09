#!/usr/bin/env Rscript
###############################################################################
# 02_differential_methylation_analysis.R
#
# Purpose:
#   Identify differentially methylated probes (DMPs) using 'limma'
#   among (ME, LC, Control).
#   1) Loads Beta_Transposed_with_Condition.csv (samples x probes + Condition).
#   2) Transposes => row=probes, col=samples for limma.
#   3) Fits a linear model with design matrix ~0+Condition.
#   4) Defines contrasts (ME-Control, LC-Control, ME-LC).
#   5) Applies filtering to keep only significant probes (adjusted P-value < 0.05 and |logFC| > 0.2).
#   6) Saves the filtered DMP results to the results directory.
#
# Usage:
#   Rscript 02_differential_methylation_analysis.R
###############################################################################

rm(list=ls())
cat("\014")  # Clear console

cat("=== Differential Methylation Analysis (limma) START ===\n\n")

# (A) Ensure required packages are available
if (!requireNamespace("limma", quietly = TRUE)) {
  install.packages("limma", repos = "https://cloud.r-project.org/")
}
suppressPackageStartupMessages(library(limma))

if (!requireNamespace("vroom", quietly = TRUE)) {
  install.packages("vroom", repos = "https://cloud.r-project.org/")
}
suppressPackageStartupMessages(library(vroom))

if (!requireNamespace("BiocParallel", quietly = TRUE)) {
  install.packages("BiocParallel", repos = "https://cloud.r-project.org/")
}
suppressPackageStartupMessages(library(BiocParallel))
# Use all available cores minus one for parallel processing in modeling.
num_cores <- max(1, parallel::detectCores() - 1)
BPPARAM <- MulticoreParam(workers = num_cores)
cat("[INFO] Using", num_cores, "cores for parallel processing.\n")

# (B) Define file paths
epi_root <- "/Volumes/T9/EpiMECoV"
beta_csv <- file.path(epi_root, "processed_data", "Beta_Transposed_with_Condition.csv")
results_dir <- file.path(epi_root, "results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
}

cat("[INFO] Reading CSV =>", beta_csv, "\n")
if (!file.exists(beta_csv)) {
  stop("File not found: ", beta_csv)
}

# (C) Read the beta matrix using vroom::vroom for faster multi-threaded CSV reading
cat("[DATA] Reading beta matrix using vroom::vroom()...\n")
beta_df <- tryCatch({
  # vroom returns a tibble; convert it to a data.frame for consistency.
  as.data.frame(vroom(beta_csv, delim = ",", col_names = TRUE, progress = FALSE))
}, error = function(e) {
  cat("[WARN] vroom() failed with error:\n", conditionMessage(e), "\nFalling back to read.csv()...\n")
  read.csv(beta_csv, header = TRUE, sep = ",", check.names = FALSE)
})
cat("[INFO] Data reading complete; data converted to data.frame.\n")
gc()

cat("[INFO] Setting row names from the first column...\n")
rownames(beta_df) <- beta_df[[1]]
beta_df[[1]] <- NULL
cat("[INFO] Data dimensions (including Condition):", dim(beta_df), "\n")

# (D) Extract Condition and convert numeric data to a matrix
if (!("Condition" %in% colnames(beta_df))) {
  stop("[ERROR] 'Condition' column not found in the CSV. Cannot proceed.")
}
Condition <- as.factor(beta_df$Condition)
cat("[INFO] Converting beta values to numeric matrix (excluding 'Condition')...\n")
beta_numeric <- beta_df[, !colnames(beta_df) %in% "Condition", drop = FALSE]
beta_numeric <- as.matrix(sapply(beta_numeric, as.numeric))
cat("[INFO] Numeric beta matrix dimensions:", paste(dim(beta_numeric), collapse = " x "), "\n")
rm(beta_df)
gc()

# (E) Transpose the beta matrix for limma (row=probes, col=samples)
cat("\n[STEP] Transposing numeric matrix: now row=probes, col=samples.\n")
beta_t <- t(beta_numeric)
cat("[INFO] Transposed dimensions:", paste(dim(beta_t), collapse = " x "), "\n")
rm(beta_numeric)
gc()

# (F) Build design matrix
cat("\n[STEP] Building design matrix (model.matrix(~0 + Condition)).\n")
design <- model.matrix(~0 + Condition)
colnames(design) <- levels(Condition)
print(head(design))

# (G) Fit linear model and compute contrasts
cat("\n[STEP] Fitting linear model with limma (parallelized)...\n")
fit <- lmFit(beta_t, design, BPPARAM = BPPARAM)
contrast.matrix <- makeContrasts(
  ME_vs_Control = ME - Control,
  LC_vs_Control = LC - Control,
  ME_vs_LC      = ME - LC,
  levels = design
)
fit2 <- contrasts.fit(fit, contrast.matrix)
fit2 <- eBayes(fit2)

cat("[INFO] Extracting topTable for each contrast.\n\n")

# (H) Filter DMP results before saving
# Set thresholds: adjusted P-value < 0.05 and absolute logFC > 0.2
pval_thresh <- 0.05
logfc_thresh <- 0.2

top_ME <- topTable(fit2, coef = "ME_vs_Control", number = Inf, 
                   adjust.method = "BH", sort.by = "P")
sig_ME <- top_ME[top_ME$adj.P.Val < pval_thresh & abs(top_ME$logFC) > logfc_thresh, ]
cat(" => ME vs Control: Found", nrow(sig_ME), "significant probes (of", nrow(top_ME), ").\n")
out_ME <- file.path(results_dir, "DMP_ME_vs_Control.csv")
write.csv(sig_ME, out_ME, row.names = TRUE, quote = FALSE)

top_LC <- topTable(fit2, coef = "LC_vs_Control", number = Inf,
                   adjust.method = "BH", sort.by = "P")
sig_LC <- top_LC[top_LC$adj.P.Val < pval_thresh & abs(top_LC$logFC) > logfc_thresh, ]
cat(" => LC vs Control: Found", nrow(sig_LC), "significant probes (of", nrow(top_LC), ").\n")
out_LC <- file.path(results_dir, "DMP_LC_vs_Control.csv")
write.csv(sig_LC, out_LC, row.names = TRUE, quote = FALSE)

top_ME_LC <- topTable(fit2, coef = "ME_vs_LC", number = Inf,
                      adjust.method = "BH", sort.by = "P")
sig_ME_LC <- top_ME_LC[top_ME_LC$adj.P.Val < pval_thresh & abs(top_ME_LC$logFC) > logfc_thresh, ]
cat(" => ME vs LC: Found", nrow(sig_ME_LC), "significant probes (of", nrow(top_ME_LC), ").\n")
out_ME_LC <- file.path(results_dir, "DMP_ME_vs_LC.csv")
write.csv(sig_ME_LC, out_ME_LC, row.names = TRUE, quote = FALSE)

cat("\n=== Differential Methylation Analysis Complete. ===\n")
cat("Results saved in the 'results' folder.\n")