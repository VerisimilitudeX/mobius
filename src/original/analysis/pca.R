###############################################################################
# pca_analysis.R
#
# Purpose:
#   A more comprehensive PCA script for your epigenetic Beta matrix, leveraging
#   'irlba' for large-scale data. It:
#     1) Accepts user-defined arguments (via command line).
#     2) Loads your Beta matrix CSV (rows = samples, columns = probes, last col = Condition).
#     3) Performs PCA using 'irlba'.
#     4) Outputs a PCA plot (PC1 vs. PC2), coloring by Condition.
#     5) Saves numeric PCA results in CSV.
#     6) Saves the plot as a PNG file.
#
# Usage:
#   Rscript pca_analysis.R --csv=Merged_450K_EPIC_BetaValues_with_Condition_Updated.csv --pcs=10
###############################################################################

rm(list=ls())
cat("\014")

if (!requireNamespace("argparse", quietly = TRUE)) {
  install.packages("argparse", repos="https://cloud.r-project.org/")
}
library(argparse)

if (!requireNamespace("irlba", quietly = TRUE)) {
  install.packages("irlba", repos="https://cloud.r-project.org/")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos="https://cloud.r-project.org/")
}
library(irlba)
library(ggplot2)

parser <- ArgumentParser(description="Perform PCA on a Beta matrix with optional subsampling.")
parser$add_argument("--csv", default="Merged_450K_EPIC_BetaValues_with_Condition_Updated.csv",
                    help="Path to the Beta matrix CSV.")
parser$add_argument("--pcs", type="integer", default=10,
                    help="Number of principal components (default=10).")
parser$add_argument("--subsample", type="integer", default=0,
                    help="Number of random probes to select. If 0, use all.")
parser$add_argument("--output", default="PCA_irlba",
                    help="Basename for output files.")
args <- parser$parse_args()

beta_csv   <- args$csv
n_pcs      <- args$pcs
subsample  <- args$subsample
output_tag <- args$output

cat("=== PCA Analysis ===\n")
cat("CSV file   =", beta_csv, "\n")
cat("PCs        =", n_pcs, "\n")
cat("Subsample  =", subsample, "\n")
cat("Output tag =", output_tag, "\n\n")

if (!file.exists(beta_csv)) {
  stop("CSV file not found: ", beta_csv)
}

beta_df <- read.csv(beta_csv, row.names=1, check.names=FALSE)
if (!("Condition" %in% colnames(beta_df))) {
  stop("No 'Condition' column found.")
}
Condition <- as.factor(beta_df$Condition)
beta_mat <- beta_df[, -ncol(beta_df), drop=FALSE]

cat("Samples =", nrow(beta_mat), "\nProbes  =", ncol(beta_mat), "\n\n")

if (subsample > 0 && subsample < ncol(beta_mat)) {
  set.seed(123)
  chosen_cols <- sample(colnames(beta_mat), subsample, replace=FALSE)
  beta_mat <- beta_mat[, chosen_cols, drop=FALSE]
  cat("Subsampled => new dimension =", nrow(beta_mat), "x", ncol(beta_mat), "\n")
}

beta_mat <- as.matrix(beta_mat)
cat("\nRunning PCA with IRLBA, #PCs =", n_pcs, "\n")
pca_res <- irlba::prcomp_irlba(beta_mat, n=n_pcs, center=TRUE, scale.=TRUE)

cat("PCA done. Summarizing variance:\n")
var_explained <- summary(pca_res)$importance[2, ]
print(var_explained)

pca_df <- data.frame(PC1=pca_res$x[,1], PC2=pca_res$x[,2], Condition=Condition)

pca_plot <- ggplot(pca_df, aes(x=PC1, y=PC2, color=Condition)) +
  geom_point(alpha=0.7, size=2) +
  theme_minimal() +
  labs(title=paste("PCA with IRLBA (", n_pcs, " PCs)", sep=""),
       x="PC1", y="PC2")

plot_file <- paste0(output_tag, "_PC1_PC2.png")
cat("Saving PCA plot =>", plot_file, "\n")
ggsave(filename=plot_file, plot=pca_plot, width=8, height=6, dpi=300)

pc_coords <- as.data.frame(pca_res$x)
pc_coords$Condition <- Condition
out_csv <- paste0(output_tag, "_PC_Coords.csv")
write.csv(pc_coords, file=out_csv, row.names=TRUE, quote=FALSE)
cat("Saved numeric PCA coords =>", out_csv, "\n")

cat("\n=== Script Complete ===\n")