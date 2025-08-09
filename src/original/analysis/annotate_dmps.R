###############################################################################
# annotate_all_dmps.R
#
# Purpose:
#   Annotate your three DMP CSV files:
#     - DMP_ME_vs_Control.csv
#     - DMP_LC_vs_Control.csv
#     - DMP_ME_vs_LC.csv
#   Each file has row names for CpG IDs plus columns from limma's topTable.
#   We'll create a "CpG" column from row names, then run champ.annot().
#
# Usage:
#   Rscript annotate_all_dmps.R --array=450K  (or EPIC)
###############################################################################

rm(list=ls())
cat("\014")

if (!requireNamespace("argparse", quietly=TRUE)) {
  install.packages("argparse", repos="https://cloud.r-project.org/")
}
library(argparse)

# We need BiocManager for ChAMP + minfi if not installed
if (!requireNamespace("BiocManager", quietly=TRUE)) {
  install.packages("BiocManager", repos="https://cloud.r-project.org/")
}
if (!requireNamespace("ChAMP", quietly=TRUE)) {
  BiocManager::install("ChAMP")
}
if (!requireNamespace("minfi", quietly=TRUE)) {
  BiocManager::install("minfi")
}

library(ChAMP)
library(minfi)

parser <- ArgumentParser(description="Annotate the 3 main DMP CSV files.")
parser$add_argument("--array", default="450K",
                    help="Array type: 450K or EPIC.")
args <- parser$parse_args()
array_type <- args$array

cat("=== Bulk Annotation for 3 DMP Files ===\n")
cat("Array type =", array_type, "\n\n")

# We'll define the three expected DMP files
dmp_files <- c("DMP_ME_vs_Control.csv", "DMP_LC_vs_Control.csv", "DMP_ME_vs_LC.csv")

for (dmp_file in dmp_files) {
  if (!file.exists(dmp_file)) {
    cat("[WARNING] File not found:", dmp_file, "Skipping.\n")
    next
  }
  
  cat("\n--- Processing:", dmp_file, "---\n")
  dmp_data <- read.csv(dmp_file, row.names=1, stringsAsFactors=FALSE)
  cat("Read", dmp_file, "dimension:", nrow(dmp_data), "x", ncol(dmp_data), "\n")
  
  dmp_data$CpG <- rownames(dmp_data)
  
  dummyBeta <- matrix(nrow=nrow(dmp_data), ncol=1)
  rownames(dummyBeta) <- dmp_data$CpG
  colnames(dummyBeta) <- "Dummy_Sample"
  
  cat("Running champ.annot()...\n")
  annotRes <- champ.annot(beta=dummyBeta, arraytype=array_type)
  annotDF <- annotRes$bedFile
  
  if (!("probeID" %in% colnames(annotDF))) {
    annotDF$probeID <- rownames(annotDF)
  }
  
  final_annotated <- merge(dmp_data, annotDF, by.x="CpG", by.y="probeID", all.x=TRUE)
  
  outFile <- paste0("Annotated_", dmp_file)
  cat("Writing =>", outFile, "\nDimension:", nrow(final_annotated), "x", ncol(final_annotated), "\n")
  write.csv(final_annotated, outFile, row.names=FALSE, quote=FALSE)
}

cat("\n=== All annotation completed. Check Annotated_*.csv files. ===\n")