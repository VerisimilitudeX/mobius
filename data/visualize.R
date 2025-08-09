# Function to analyze methylation patterns by genomic regions
analyze_genomic_regions <- function(data_list, results_dir) {
  mSet <- data_list$mSet
  sample_sheet <- data_list$sample_sheet
  
  # Get annotation
  anno <- getAnnotation(mSet)
  
  # Gene region analysis (TSS1500, TSS200, 5'UTR, 1stExon, Body, 3'UTR)
  gene_regions <- c("TSS1500", "TSS200", "5'UTR", "1stExon", "Body", "3'UTR")
  
  # Create a matrix to store region means by condition
  conditions <- unique(sample_sheet$Sample_Group)
  region_means <- matrix(NA, nrow = length(gene_regions), ncol = length(conditions),
                        dimnames = list(gene_regions, conditions))
  
  # Calculate mean methylation by gene region for each condition
  for (region in gene_regions) {
    # Look for region in UCSC_RefGene_Group column (handles multiple regions in semicolon-separated values)
    region_probes <- rownames(anno)[grep(region, anno$UCSC_RefGene_Name)]
    
    if (length(region_probes) > 0) {
      region_beta <- getBeta(mSet)[region_probes, ]
      
      for (cond in conditions) {
        cond_samples <- sample_sheet$Sample_Name[sample_sheet$Sample_Group == cond]
        if (length(cond_samples) > 0) {
          region_means[region, cond] <- mean(region_beta[, cond_samples], na.rm = TRUE)
        }
      }
    }
  }
  
  # CpG island region analysis (Island, Shore, Shelf, Open Sea)
  island_regions <- c("Island", "N_Shore", "S_Shore", "N_Shelf", "S_Shelf", "OpenSea")
  
  # Create a matrix to store island region means by condition
  island_means <- matrix(NA, nrow = length(island_regions), ncol = length(conditions),
                         dimnames = list(island_regions, conditions))
  
  # Calculate mean methylation by CpG island region for each condition
  for (region in island_regions) {
    if (region == "Island") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "Island"]
    } else if (region == "N_Shore") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "N_Shore"]
    } else if (region == "S_Shore") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "S_Shore"]
    } else if (region == "N_Shelf") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "N_Shelf"]
    } else if (region == "S_Shelf") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "S_Shelf"]
    } else if (region == "OpenSea") {
      region_probes <- rownames(anno)[anno$Relation_to_Island == "" | is.na(anno$Relation_to_Island)]
    }
    
    if (length(region_probes) > 0) {
      region_beta <- getBeta(mSet)[region_probes, ]
      
      for (cond in conditions) {
        cond_samples <- sample_sheet$Sample_Name[sample_sheet$Sample_Group == cond]
        if (length(cond_samples) > 0) {
          island_means[region, cond] <- mean(region_beta[, cond_samples], na.rm = TRUE)
        }
      }
    }
  }
  
  # Create barplots for region methylation
  pdf(file.path(results_dir, "13_gene_region_methylation.pdf"), width = 12, height = 8)
  barplot(region_means, beside = TRUE, 
          col = rainbow(length(conditions)),
          main = "Mean Methylation by Gene Region",
          xlab = "Gene Region", ylab = "Mean Beta Value",
          ylim = c(0, 1))
  legend("topright", legend = conditions, fill = rainbow(length(conditions)))
  dev.off()
  
  pdf(file.path(results_dir, "14_cpg_island_methylation.pdf"), width = 12, height = 8)
  barplot(island_means, beside = TRUE, 
          col = rainbow(length(conditions)),
          main = "Mean Methylation by CpG Island Region",
          xlab = "CpG Island Region", ylab = "Mean Beta Value",
          ylim = c(0, 1))
  legend("topright", legend = conditions, fill = rainbow(length(conditions)))
  dev.off()
  
  # Create a combined heatmap of region methylation
  pdf(file.path(results_dir, "15_region_methylation_heatmap.pdf"), width = 10, height = 8)
  # Combine region and island data
  all_regions <- rbind(region_means, island_means)
  
  # Heatmap of region methylation
  pheatmap(all_regions,
           main = "Methylation by Genomic Region",
           color = colorRampPalette(c("navy", "white", "firebrick3"))(100),
           display_numbers = TRUE,
           number_format = "%.2f",
           fontsize_number = 8)
  dev.off()
  
  # Return region results
  return(list(
    region_methyl = region_means,
    island_methyl = island_means
  ))
}

# Function to check and create directory structure
check_directory_structure <- function(root_dir) {
  # Check if the root directory exists
  if (!dir.exists(root_dir)) {
    stop(paste("Root directory does not exist:", root_dir))
  }
  
  # Check if condition directories exist
  condition_dirs <- c("ME", "LC", "controls")
  existing_conditions <- condition_dirs[sapply(condition_dirs, function(cond) {
    dir.exists(file.path(root_dir, cond))
  })]
  
  if (length(existing_conditions) == 0) {
    stop("None of the expected condition directories (ME, LC, controls) exist in the root directory")
  }
  
  message("Found condition directories: ", paste(existing_conditions, collapse = ", "))
  
  # Return list of existing condition directories
  return(existing_conditions)
}

# Main execution flow
main <- function() {
  # Check directory structure
  message("Checking directory structure...")
  tryCatch({
    existing_conditions <- check_directory_structure(root_dir)
    
    # Step 1: Find IDAT files with a limit of 10 samples per array type
    message("Scanning directories for IDAT files...")
    samples_df <- find_idat_files(root_dir, max_samples_per_type = 1000)
    
    # Check if we found any samples
    if (nrow(samples_df) == 0) {
      stop("No IDAT files found in the specified directories")
    }
    
    # Print summary of found samples
    message("Found ", nrow(samples_df), " samples:")
    message("  EPIC array: ", sum(samples_df$array_type == "IlluminaHumanMethylationEPIC"))
    message("  450K array: ", sum(samples_df$array_type == "IlluminaHumanMethylation450k"))
    
    # Save the sample information
    write.csv(samples_df, file = file.path(results_dir, "sample_info.csv"), row.names = FALSE)
    
    # Process each array type separately
    array_types <- unique(samples_df$array_type)
    
    for (array_type in array_types) {
      message("\nProcessing ", array_type, " samples...")
      
      # Create a directory for this array type
      array_result_dir <- file.path(results_dir, gsub("IlluminaHuman", "", array_type))
      dir.create(array_result_dir, showWarnings = FALSE, recursive = TRUE)
      
      # Filter samples for current array type
      current_samples <- samples_df[samples_df$array_type == array_type, ]
      
      # Step 2: Process IDAT files
      message("Processing IDAT files...")
      data_list <- tryCatch({
        process_idat_files(current_samples, array_result_dir)
      }, error = function(e) {
        message("Error processing IDAT files: ", e$message)
        return(NULL)
      })
      
      if (is.null(data_list)) {
        message("Skipping further analysis for ", array_type, " due to processing error")
        next
      }
      
      # Step 3: Quality control
      message("Performing quality control...")
      qc_results <- tryCatch({
        quality_control(data_list, array_result_dir)
      }, error = function(e) {
        message("Error in quality control: ", e$message)
        return(NULL)
      })
      
      # Step 4: Visualize methylation patterns
      message("Visualizing methylation patterns...")
      viz_results <- tryCatch({
        visualize_methylation_patterns(data_list, array_result_dir)
      }, error = function(e) {
        message("Error visualizing methylation patterns: ", e$message)
        return(NULL)
      })
      
      # Step 5: Find DMPs
      message("Finding differentially methylated positions...")
      dmp_results <- tryCatch({
        find_dmps(data_list, array_result_dir)
      }, error = function(e) {
        message("Error finding DMPs: ", e$message)
        return(NULL)
      })
      
      # Step 6: Visualize top DMPs
      message("Visualizing top DMPs...")
      tryCatch({
        visualize_top_dmps(data_list, dmp_results, array_result_dir)
      }, error = function(e) {
        message("Error visualizing top DMPs: ", e$message)
      })
      
      # Step 7: Compare datasets
      message("Comparing datasets...")
      dataset_comparison <- tryCatch({
        compare_datasets(data_list, array_result_dir)
      }, error = function(e) {
        message("Error comparing datasets: ", e$message)
        return(NULL)
      })
      
      # Step 8: Analyze genomic regions
      message("Analyzing genomic regions...")
      region_results <- tryCatch({
        analyze_genomic_regions(data_list, array_result_dir)
      }, error = function(e) {
        message("Error analyzing genomic regions: ", e$message)
        return(NULL)
      })
      
      # Step 9: Analyze functional enrichment
      message("Performing functional enrichment analysis...")
      enrichment_results <- tryCatch({
        analyze_functional_enrichment(dmp_results, array_result_dir)
      }, error = function(e) {
        message("Error in functional enrichment analysis: ", e$message)
        return(NULL)
      })
      
      # Step 10: Generate report
      message("Generating report...")
      tryCatch({
        generate_report(data_list, dmp_results, region_results, enrichment_results, 
                      array_result_dir, array_type)
      }, error = function(e) {
        message("Error generating report: ", e$message)
      })
      
      message("Analysis for ", array_type, " completed successfully\n")
    }
    
    message("All analyses completed successfully")
  }, error = function(e) {
    message("Error in main execution: ", e$message)
  })
}

# Run the main function
main()