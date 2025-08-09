#!/usr/bin/env Rscript
###############################################################################
# run_pipeline_master.R (Updated)
#
# This script orchestrates the epigenetic biomarker discovery pipeline,
# now with an optional final step calling "SOTA_Transformer_Classifier.py"
# for mandatory self-supervised pretraining + classification.
#
# Additionally:
#  • If --web_mode=TRUE, we read from a config file that the new interactive
#    dashboard might generate, specifying the folder structure for the 3 conditions.
#
# Tradeoffs of using a Transformer:
#  - Transformers offer powerful sequence modeling but are computationally intensive
#    and may overfit on small datasets. To address these issues, we incorporated
#    self-supervised pretraining, dropout, ReZero connections, and LoRA modules.
#
# USAGE:
#   Rscript run_pipeline_master.R [startStep] [--web_mode=TRUE|FALSE]
###############################################################################

rm(list = ls())
cat("\014")  # Clear console

args <- commandArgs(trailingOnly = TRUE)
startStep <- 0
web_mode <- FALSE
if (length(args) > 0) {
  maybeNum <- suppressWarnings(as.numeric(args[1]))
  if (!is.na(maybeNum)) {
    startStep <- maybeNum
  }
  if (any(grepl("--web_mode=", args))) {
    wm_val <- gsub("--web_mode=", "", args[grepl("--web_mode=", args)])
    if (length(wm_val)==1 && tolower(wm_val)=="true") {
      web_mode <- TRUE
    }
  }
}

cat("=== Run Pipeline Master Script (Updated) ===\n")
cat("Will begin at step:", startStep, "\n")
cat("Web mode:", web_mode, "\n\n")

N <- 15  # total pipeline steps
library(progress)
pb <- progress_bar$new(format = "  Step :current/:total [:bar] :percent Elapsed: :elapsed ETA: :eta",
                       total = N, clear = FALSE, width = 60)

required_dirs <- c("processed_data","results")
for (dd in required_dirs) {
  if (!dir.exists(dd)) {
    dir.create(dd, recursive = TRUE)
  }
}

if (web_mode) {
  cat("[INFO] Web mode => reading config from 'dashboard_config.json' if it exists.\n")
  json_path <- "dashboard_config.json"
  if (file.exists(json_path)) {
    cat("Found dashboard_config.json => parsing.\n")
    conf_txt <- readLines(json_path)
    library(jsonlite)
    conf_list <- fromJSON(paste(conf_txt, collapse=""))
    Sys.setenv(PIPELINE_COND1=conf_list$condition1)
    Sys.setenv(PIPELINE_COND2=conf_list$condition2)
    Sys.setenv(PIPELINE_COND3=conf_list$condition3)
  } else {
    cat("[WARN] No dashboard_config.json found. Using default.\n")
  }
}

if (startStep <= 0) {
  cat("[STEP 0] IDAT Verification (optional)\n")
  # (Optional step)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 0\n")
}

if (startStep <= 1) {
  cat("\n[STEP 1] unify_IDATs_And_Transpose.R\n")
  system("Rscript src/original/steps/01_unify_IDATs_And_Transpose.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 1\n")
}

if (startStep <= 2) {
  cat("\n[STEP 2] 04_qa.R minimal QA\n")
  system("Rscript src/original/steps/04_qa.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 2\n")
}

if (startStep <= 3) {
  cat("\n[STEP 3] 02_differential_methylation_analysis.R\n")
  system("Rscript src/original/steps/02_differential_methylation_analysis.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 3\n")
}

if (startStep <= 4) {
  cat("\n[STEP 4] 08_prepare_final_data.R\n")
  system("Rscript src/original/steps/08_prepare_final_data.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 4\n")
}

if (startStep <= 5) {
  cat("\n[STEP 5] 09_feature_selection.R\n")
  system("Rscript src/original/steps/09_feature_selection.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 5\n")
}

if (startStep <= 6) {
  cat("\n[STEP 6] 05_preprocessing.py\n")
  cmd <- "python3 src/original/steps/05_preprocessing.py --csv ./processed_data/filtered_biomarker_matrix.csv --out ./processed_data/cleaned_data.csv --method auto"
  system(cmd, intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 6\n")
}

if (startStep <= 7) {
  cat("\n[STEP 7] 06_feature_engineer.py (VAE-based dimensionality reduction)\n")
  cmd <- "python3 src/original/steps/06_feature_engineer.py --csv ./processed_data/cleaned_data.csv --out ./processed_data/transformed_data.csv --latent_dim 64 --epochs 20 --batch_size 64 --lr 0.001 --dropout 0.1 --use_scale True"
  system(cmd, intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 7\n")
}

if (startStep <= 8) {
  cat("\n[STEP 8] 10_baseline_classification.py\n")
  system("python3 src/original/steps/10_baseline_classification.py", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 8\n")
}

if (startStep <= 9) {
  cat("\n[STEP 9] 11_transformer_classifier.py & 12_evaluate_results.py\n")
  system("python3 src/original/steps/transformer_classifier.py", intern = TRUE)
  system("python3 src/original/steps/12_evaluate_results.py", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 9\n")
}

if (startStep <= 10) {
  cat("\n[STEP 10] 13_summarize_findings.R\n")
  system("Rscript src/original/steps/13_summarize_findings.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 10\n")
}

if (startStep <= 10.5) {
  cat("\n[STEP 10.5] 11_report_shared_distinct_dmps.R\n")
  system("Rscript src/original/steps/11_report_shared_distinct_dmps.R", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 10.5\n")
}

if (startStep <= 11) {
  cat("\n[STEP 11] 14_visualize_results.py\n")
  system("python3 src/original/steps/14_visualize_results.py", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 11\n")
}

if (startStep <= 12) {
  cat("\n[STEP 12] Other advanced analyses (e.g., random forest hyperparam search, shap analysis)\n")
  # Optional steps
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 12\n")
}

if (startStep <= 13) {
  cat("\n[STEP 13] **New** SOTA_Transformer_Classifier (with convergence graphs) - mandatory self-supervised pretraining\n")
  system("python3 src/original/steps/transformer_classifier.py", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 13\n")
}

if (startStep <= 14) {
  cat("\n[STEP 14] 07_ensemble_ml.py (Ensemble Machine Learning)\n")
  system("python3 src/original/other/random_forest_hyperparam_search.py", intern = TRUE)
  try(pb$tick(), silent=TRUE)
} else {
  cat("Skipping Step 14\n")
}

cat("\n=== Pipeline complete! ===\n")