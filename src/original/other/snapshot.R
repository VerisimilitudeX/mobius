################################################################################
# snapshot.R
# Automated R script to tally IDAT/CSV files and estimate sample sizes
################################################################################

data_dir <- "/Volumes/T9/EpiMECoV/data"
output_file <- file.path(data_dir, "output_log.txt")

sink(output_file)

all_files <- list.files(path = data_dir, pattern = NULL, full.names = TRUE, recursive = TRUE)

idat_files <- all_files[grepl("\\.idat$", all_files, ignore.case = TRUE)]
gBase <- sub("_Grn\\.idat$", "", basename(idat_files), ignore.case = TRUE)
rBase <- sub("_Red\\.idat$", "", basename(idat_files), ignore.case = TRUE)
common_samples <- intersect(gBase, rBase)

cat("\nIDAT file check\n")
cat("==============================\n")
cat("Total IDAT files found:", length(idat_files), "\n")
cat("Estimated unique samples (assuming Grn/Red pairs):", length(common_samples), "\n")

sample_sheet_candidates <- all_files[grepl("samplesheet\\.csv$", all_files, ignore.case = TRUE)]
if(length(sample_sheet_candidates) > 0) {
  cat("\nPotential sample sheet(s):\n")
  print(sample_sheet_candidates)
  cat("\nReading first one:\n")
  sample_sheet <- read.csv(sample_sheet_candidates[1], header = TRUE, stringsAsFactors = FALSE)
  cat("\nSample sheet (head):\n")
  print(head(sample_sheet))
  cat("\nTotal rows in sample sheet:", nrow(sample_sheet), "\n")
} else {
  cat("\nNo obvious SampleSheet.csv found.\n")
}

cat("\nPlatform determination usually requires reading the IDAT data.\n")

csv_or_tsv_files <- all_files[grepl("\\.csv$|\\.tsv$|\\.txt$", all_files, ignore.case = TRUE)]
cat("\nCSV/TSV file check\n")
cat("==============================\n")
cat("Total CSV/TSV files found:", length(csv_or_tsv_files), "\n")

for(f in csv_or_tsv_files) {
  file_con <- file(f, "r")
  first_lines <- readLines(file_con, n = 5)
  close(file_con)
  suspicion <- FALSE
  if(any(grepl("beta", first_lines, ignore.case = TRUE))) suspicion <- TRUE
  if(any(grepl("Methylation", first_lines, ignore.case = TRUE))) suspicion <- TRUE
  if(any(grepl("^cg", first_lines))) suspicion <- TRUE
  
  if(suspicion) {
    cat("\nPotential methylation file:", basename(f), "\n")
    cat("First lines:\n")
    cat(paste(first_lines, collapse = "\n"), "\n")
  }
}

cat("\nDone.\n")
sink()