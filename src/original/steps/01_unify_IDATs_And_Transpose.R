#!/usr/bin/env Rscript
###############################################################################
# unify_IDATs_And_Transpose.R
#
# PURPOSE:
#   A single, combined script that:
#   (1) Installs + loads necessary Bioconductor and CRAN packages.
#   (2) Detects and merges IDAT files from subfolders (ME, LC, controls).
#   (3) Filters by detection p-values, performs Noob normalization, maps to genome.
#   (4) Removes cross-reactive + SNP-affected probes (if desired).
#   (5) Merges 450K and EPIC sets into a single GenomicMethylSet.
#   (6) Verifies Condition labeling, then saves the merged set + Beta matrix.
#   (7) Transposes that Beta matrix into “samples x probes,” appends Condition 
#       as the last column.
#   (8) **New:** Applies BMIQ to correct type II probe bias and then uses RUVM 
#       (via RUVfit/RUVadj) to adjust for batch effects.
#
# USAGE:
#   Rscript unify_IDATs_And_Transpose.R
###############################################################################

cat("=== unify_IDATs_And_Transpose START ===\n", as.character(Sys.time()), "\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (1) Install and Load Packages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[1] Installing/loading required packages...\n")

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

# Personal R library path (optional). Adjust as you wish:
personal_lib <- "~/R/library"
if (!dir.exists(personal_lib)) {
  dir.create(personal_lib, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(personal_lib, .libPaths()))

pkg_list <- c(
  "minfi",
  "IlluminaHumanMethylation450kmanifest",
  "IlluminaHumanMethylationEPICmanifest",
  "BiocParallel",
  "minfiData",
  "IlluminaHumanMethylation450kanno.ilmn12.hg19",
  "maxprobes",   # to handle cross-reactive or SNP-affected probes if desired
  "wateRmelon",  # for BMIQ normalization
  "sva"          # for ComBat batch-effect correction
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (pkg %in% c("maxprobes")) {
      # 'maxprobes' is not on CRAN/Bioc => install from GitHub
      remotes::install_github("markgene/maxprobes",
                              lib = personal_lib,
                              upgrade = "never")
    } else {
      BiocManager::install(pkg, ask = FALSE, lib = personal_lib)
    }
  }
  library(pkg, character.only = TRUE, lib.loc = personal_lib)
}

for (p in pkg_list) {
  install_if_missing(p)
}

library(BiocParallel)
num_cores <- max(1, parallel::detectCores() - 1)
register(MulticoreParam(num_cores))
cat("[INFO] Using", num_cores, "cores.\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (2) Define Data and Output Directories + max_idats
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# You can control how many IDATs to sample with this variable:
# - If set to -1, it uses ALL IDATs found (forces balanced sampling)
# - Otherwise it picks exactly that many IDATs in a balanced way across conditions
max_idats <- 6   # <--- changed here

epi_root <- "/Volumes/T9/EpiMECoV"
data_root_dir <- file.path(epi_root, "data")

# Condition subfolders: "ME", "LC", "controls"
cond_names <- c("ME", "LC", "Control")
dir_names  <- c("ME", "LC", "controls") 
condition_dirs <- file.path(data_root_dir, dir_names)
names(condition_dirs) <- cond_names

for (cn in names(condition_dirs)) {
  if (!dir.exists(condition_dirs[cn])) {
    stop(paste("No folder for condition", cn, "=>", condition_dirs[cn]))
  }
  cat("[OK]", cn, "found at:", condition_dirs[cn], "\n")
}

processed_dir <- file.path(epi_root, "processed_data")
if (!dir.exists(processed_dir)) {
  dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
  cat("[INFO] Created 'processed_data' folder =>", processed_dir, "\n")
} else {
  cat("[INFO] Using existing output dir =>", processed_dir, "\n")
}

# Helper function to pick a balanced subset across conditions
balancedSubset <- function(sheet, desired_total) {
  conds <- unique(sheet$Condition)
  k <- length(conds)
  if (k <= 1) {
    cat("[INFO] Only one condition => returning entire sheet.\n")
    return(sheet)
  }
  
  each <- floor(desired_total / k)
  out_list <- list()
  
  # pick an equal number from each condition
  for (cn in conds) {
    subdf <- sheet[sheet$Condition == cn, ]
    n_pick <- min(nrow(subdf), each)
    if (n_pick > 0) {
      idx <- sample(seq_len(nrow(subdf)), n_pick)
      out_list[[cn]] <- subdf[idx, ]
    } else {
      out_list[[cn]] <- subdf[0, ]
    }
  }
  
  out_df <- do.call(rbind, out_list)
  used_ct <- nrow(out_df)
  leftover <- desired_total - used_ct
  
  # If leftover > 0, pick random from entire set that wasn't used yet
  if (leftover > 0) {
    used_names <- out_df$Sample_Name
    remain <- sheet[!(sheet$Sample_Name %in% used_names), ]
    if (nrow(remain) < leftover) leftover <- nrow(remain)
    if (leftover > 0) {
      idx2 <- sample(seq_len(nrow(remain)), leftover)
      out_df <- rbind(out_df, remain[idx2, ])
    }
  }
  
  return(out_df)
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (3) IDAT Detection & Sample Sheet
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[3] Scanning IDAT files for each condition...\n")

findIDATsForCondition <- function(dirPath, condName) {
  # Recursively locate all *.idat
  all_idats <- list.files(dirPath, pattern = "\\.idat$", 
                          full.names = TRUE, recursive = TRUE)
  gPaths <- all_idats[grepl("Grn\\.idat$", all_idats, ignore.case = TRUE)]
  rPaths <- all_idats[grepl("Red\\.idat$", all_idats, ignore.case = TRUE)]
  gBase <- sub("_Grn\\.idat$", "", basename(gPaths), ignore.case = TRUE)
  rBase <- sub("_Red\\.idat$", "", basename(rPaths), ignore.case = TRUE)
  common <- intersect(gBase, rBase)
  if (length(common) == 0) {
    cat("[WARNING] No matching Grn/Red for condition:", condName, "\n")
    return(NULL)
  }
  df <- data.frame(
    Sample_Name = common,
    Basename    = rep("", length(common)),
    Condition   = rep(condName, length(common)),
    stringsAsFactors = FALSE
  )
  for (i in seq_along(common)) {
    sid <- common[i]
    # Locate matching *Grn.idat
    gMatch <- gPaths[basename(gPaths) == paste0(sid, "_Grn.idat")]
    if (length(gMatch) == 1) {
      basePath <- sub("_Grn\\.idat$", "", gMatch, ignore.case = TRUE)
      df$Basename[i] <- basePath
    } else {
      df$Basename[i] <- NA
    }
  }
  df <- df[!is.na(df$Basename), ]
  return(df)
}

all_df_list <- lapply(names(condition_dirs), function(cn) {
  findIDATsForCondition(condition_dirs[cn], cn)
})
sample_sheet <- do.call(rbind, all_df_list)
sample_sheet <- sample_sheet[!is.na(sample_sheet$Basename), ]
cat("Total samples across all conditions:", nrow(sample_sheet), "\n")

cat("[INFO] Picking a balanced subset across conditions.\n")
set.seed(123)
# Force balanced sampling even when max_idats is -1.
if (max_idats < 1) {
  counts <- table(sample_sheet$Condition)
  min_count <- min(counts)
  balanced_total <- min_count * length(counts)
  actual_idats <- balanced_total
  cat("[INFO] max_idats = -1, so using", min_count, "samples per condition (total =", balanced_total, ").\n")
} else {
  actual_idats <- max_idats
}
sample_sheet <- balancedSubset(sample_sheet, actual_idats)
cat("Now using", nrow(sample_sheet), "samples.\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (4) Detect 450K vs EPIC array
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[4] Checking array platform for each sample...\n")

detectPlatform <- function(bPath) {
  rg <- tryCatch(
    read.metharray(bPath, verbose = FALSE, force = TRUE),
    error = function(e) { cat("[ERR] Could not read:", bPath, "\n"); return(NULL) }
  )
  if (is.null(rg)) {
    return("Unknown")
  }
  arr <- annotation(rg)[["array"]]
  if (is.null(arr) || arr == "") arr <- "Unknown"
  return(arr)
}

library(parallel)  # or use BiocParallel
platforms <- bplapply(sample_sheet$Basename, detectPlatform)
sample_sheet$Platform <- unlist(platforms)

tbl_platform <- table(sample_sheet$Platform, sample_sheet$Condition)
cat("Platform counts:\n")
print(tbl_platform)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (5) Build RGChannelSets for each platform in parallel
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[5] Building RGChannelSets (450K, EPIC) in parallel...\n")

sheet_450k <- sample_sheet[sample_sheet$Platform == "IlluminaHumanMethylation450k", ]
sheet_epic <- sample_sheet[sample_sheet$Platform == "IlluminaHumanMethylationEPIC", ]

buildRGChannelSet <- function(sheet, arrayname="450K") {
  if (nrow(sheet) == 0) {
    cat("[INFO]", arrayname, " => no samples.\n")
    return(NULL)
  }
  cat("[INFO] Reading", nrow(sheet), "samples =>", arrayname, "\n")
  rgSet <- read.metharray.exp(targets = sheet, force = TRUE, verbose = FALSE)
  # Store Condition in pData
  pData(rgSet)$Condition <- sheet$Condition
  cat("[INFO]", arrayname, " => read", sum(!is.null(rgSet)), "non-null.\n")
  return(rgSet)
}

rgSet_450k <- buildRGChannelSet(sheet_450k, arrayname="450K")
if (!is.null(rgSet_450k)) {
  out_rds_450k <- file.path(processed_dir, "450K_Combined_RGChannelSet.rds")
  saveRDS(rgSet_450k, out_rds_450k)
  cat("[SAVED]", out_rds_450k, "\n")
}

rgSet_epic <- buildRGChannelSet(sheet_epic, arrayname="EPIC")
if (!is.null(rgSet_epic)) {
  out_rds_epic <- file.path(processed_dir, "EPIC_Combined_RGChannelSet.rds")
  saveRDS(rgSet_epic, out_rds_epic)
  cat("[SAVED]", out_rds_epic, "\n")
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (6) Preprocess each platform (detectionP, Noob, mapToGenome, BMIQ, RUVM)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[6] Preprocessing each platform (detectionP, Noob, mapToGenome, BMIQ, RUVM)...\n")

preprocessPlatform <- function(rgSet, array_label="450K") {
  if (is.null(rgSet)) {
    return(NULL)
  }
  cat("[INFO]", array_label, "=> detection p-value filter...\n")
  detP <- detectionP(rgSet)
  keep <- colMeans(detP < 0.01) > 0.98
  rgSet <- rgSet[, keep, drop=FALSE]
  
  cat("[INFO]", array_label, "=> preprocessNoob (background correction) ...\n")
  gmSet <- preprocessNoob(rgSet)
  
  # Optional quantile normalization if desired per paper phrasing
  cat("[INFO]", array_label, "=> preprocessQuantile (quantile normalization) ...\n")
  gmSet <- preprocessQuantile(gmSet)
  
  cat("[INFO]", array_label, "=> mapToGenome...\n")
  gmSet <- mapToGenome(gmSet)
  
  # --- Apply BMIQ Normalization ---
  cat("[INFO] Applying BMIQ normalization...\n")
  betas <- getBeta(gmSet)
  # Get probe design information from annotation
  anno <- getAnnotation(gmSet)
  # Convert design type ("I" or "II") to numeric (1 for type I, 2 for type II)
  design <- ifelse(anno$Type == "I", 1, 2)
  betas_bmiq <- betas  # initialize matrix for BMIQ-adjusted betas
  for (i in 1:ncol(betas)) {
    sample_beta <- betas[, i]
    res <- BMIQ(beta.v = sample_beta, design.v = design, nfit = 5000,
                th1.v = c(0.2, 0.75), th2.v = NULL, niter = 5, tol = 0.001,
                plots = FALSE, pri = FALSE)
    betas_bmiq[, i] <- res$nbeta
    cat("[INFO] BMIQ normalization applied to sample", i, "\n")
  }
  # Update the GenomicMethylSet with BMIQ-normalized beta values
  gmSet_bmiq <- gmSet
  assay(gmSet_bmiq, "Beta") <- betas_bmiq
  outFileCSV_BMIQ <- file.path(processed_dir, paste0(array_label, "_BetaValues_BMIQ.csv"))
  write.csv(betas_bmiq, outFileCSV_BMIQ, quote = FALSE)
  cat("[SAVED]", outFileCSV_BMIQ, "\n")
  
  # --- Apply ComBat Batch Correction (sva) ---
  cat("[INFO] Applying ComBat batch-effect correction (sva)...\n")
  betas_bmiq <- assay(gmSet_bmiq, "Beta")
  # Build batch vector: if 'Slide' info exists in pData, use it; otherwise use 'Array'
  pd <- pData(gmSet_bmiq)
  batch <- NULL
  if (!is.null(pd$Slide)) {
    batch <- as.factor(pd$Slide)
  } else if (!is.null(pd$Array)) {
    batch <- as.factor(pd$Array)
  } else {
    # fallback: use Condition as batch proxy (not ideal, but avoids failure)
    batch <- as.factor(pd$Condition)
  }
  # ComBat expects genes x samples matrix (features x samples)
  require(sva)
  combat_corrected <- ComBat(dat = betas_bmiq, batch = batch, par.prior = TRUE, prior.plots = FALSE)
  
  # Update GenomicMethylSet with ComBat-corrected betas
  gmSet_corrected <- gmSet_bmiq
  assay(gmSet_corrected, "Beta") <- combat_corrected
  
  outFileRDS_corrected <- file.path(processed_dir, paste0(array_label, "_Final_GenomicMethylSet_ComBat.rds"))
  saveRDS(gmSet_corrected, outFileRDS_corrected)
  cat("[SAVED]", outFileRDS_corrected, "\n")
  
  outFileCSV_corrected <- file.path(processed_dir, paste0(array_label, "_BetaValues_ComBat.csv"))
  write.csv(combat_corrected, outFileCSV_corrected, quote = FALSE)
  cat("[SAVED]", outFileCSV_corrected, "\n")
  
  return(gmSet_corrected)
}

gmSet_450k <- preprocessPlatform(rgSet_450k, "450K")
gmSet_epic <- preprocessPlatform(rgSet_epic, "EPIC")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (7) Removes cross-reactive + SNP-affected probes (if desired).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Example usage (commented out):
# if (!is.null(gmSet_450k)) {
#   gmSet_450k <- removeCrossReactive(gmSet_450k)
# }
# if (!is.null(gmSet_epic)) {
#   gmSet_epic <- removeCrossReactive(gmSet_epic)
# }

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (8) Merge 450K & EPIC if both exist
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cat("\n[8] Merging 450K & EPIC if both exist...\n")
if (!is.null(gmSet_450k) && !is.null(gmSet_epic)) {
  cat("[convertArray] Casting EPIC to IlluminaHumanMethylation450k\n")
  gmSet_epic_450k <- convertArray(gmSet_epic,
                                  outType = "IlluminaHumanMethylation450k",
                                  verbose = FALSE)
  combined <- combineArrays(gmSet_450k, gmSet_epic_450k)
  
  cond_450k <- pData(gmSet_450k)$Condition
  cond_epic <- pData(gmSet_epic_450k)$Condition
  new_cond <- c(cond_450k, cond_epic)
  if (length(new_cond) == ncol(combined)) {
    pData(combined)$Condition <- new_cond
  }
  
  out_merged_rds <- file.path(processed_dir, "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds")
  saveRDS(combined, out_merged_rds)
  cat("[SAVED]", out_merged_rds, "\n")
  
  combined_betas <- getBeta(combined)
  out_merged_csv <- file.path(processed_dir, "Merged_450K_EPIC_BetaValues_with_Condition.csv")
  write.csv(combined_betas, out_merged_csv, quote = FALSE)
  cat("[SAVED]", out_merged_csv, "\n")
  
  cat("\nNow => generating transposed Beta with Condition as final column.\n")
  cond_vec <- pData(combined)$Condition
  if (is.null(cond_vec)) cond_vec <- rep("Unknown", ncol(combined_betas))
  
  beta_t <- t(combined_betas)
  beta_df <- as.data.frame(beta_t)
  beta_df$Condition <- cond_vec
  
  outFileTransposed <- file.path(processed_dir, "Beta_Transposed_with_Condition.csv")
  write.csv(beta_df, outFileTransposed, row.names = TRUE, quote = FALSE)
  cat("[INFO] Saving =>", outFileTransposed, "\n")
  
} else if (!is.null(gmSet_450k)) {
  cat("[INFO] Only 450K => saving Beta_Transposed_with_Condition for that alone.\n")
  combined <- gmSet_450k
  combined_betas <- getBeta(combined)
  
  out_merged_rds <- file.path(processed_dir, "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds")
  saveRDS(combined, out_merged_rds)
  
  out_merged_csv <- file.path(processed_dir, "Merged_450K_EPIC_BetaValues_with_Condition.csv")
  write.csv(combined_betas, out_merged_csv, quote = FALSE)
  
  cond_vec <- pData(combined)$Condition
  beta_t <- t(combined_betas)
  beta_df <- as.data.frame(beta_t)
  beta_df$Condition <- cond_vec
  
  outFileTransposed <- file.path(processed_dir, "Beta_Transposed_with_Condition.csv")
  write.csv(beta_df, outFileTransposed, row.names = TRUE, quote = FALSE)
  cat("[INFO] Saving =>", outFileTransposed, "\n")
  
} else if (!is.null(gmSet_epic)) {
  cat("[INFO] Only EPIC => saving Beta_Transposed_with_Condition for that alone.\n")
  combined <- gmSet_epic
  combined_betas <- getBeta(combined)
  
  out_merged_rds <- file.path(processed_dir, "Merged_450K_EPIC_GenomicMethylSet_with_Condition.rds")
  saveRDS(combined, out_merged_rds)
  
  out_merged_csv <- file.path(processed_dir, "Merged_450K_EPIC_BetaValues_with_Condition.csv")
  write.csv(combined_betas, out_merged_csv, quote = FALSE)
  
  cond_vec <- pData(combined)$Condition
  beta_t <- t(combined_betas)
  beta_df <- as.data.frame(beta_t)
  beta_df$Condition <- cond_vec
  
  outFileTransposed <- file.path(processed_dir, "Beta_Transposed_with_Condition.csv")
  write.csv(beta_df, outFileTransposed, row.names = TRUE, quote = FALSE)
  cat("[INFO] Saving =>", outFileTransposed, "\n")
  
} else {
  cat("[ERROR] No GMsets created => no data.\n")
}

cat("\n=== unify_IDATs_And_Transpose DONE ===\n", as.character(Sys.time()), "\n")