# Pipeline Test Plan

## Prerequisites
- Ensure all required R and Python packages are installed
- Verify IDAT data files exist in correct locations
- Check disk space for intermediate files

## Test Cases

### 1. Fresh Run Test
1. Delete all files in processed_data/ and results/
2. Run pipeline from start: `Rscript run_pipeline_master_fixed.R`
3. Verify each step creates expected output files
4. Check log for any errors or warnings

### 2. Dependency Check Test
1. Delete specific intermediate files
2. Run pipeline from different starting points
3. Verify proper error messages when dependencies missing
4. Confirm pipeline stops on critical failures

### 3. Partial Run Test
1. Run pipeline with different startStep values
2. Verify correct steps are skipped
3. Confirm existing files are properly validated

### 4. Web Mode Test
1. Create test dashboard_config.json
2. Run pipeline with --web_mode=TRUE
3. Verify condition mappings are applied correctly

### 5. Error Recovery Test
1. Introduce intentional errors (e.g., corrupt input files)
2. Verify error messages are clear and helpful
3. Confirm pipeline stops at appropriate points
4. Check error logs for debugging information

## Expected Outputs

Each step should produce specific files:

1. Step 1 (IDAT Processing):
   - processed_data/Beta_Transposed_with_Condition.csv

2. Step 2 (Differential Analysis):
   - results/DMP_*.csv files

3. Steps 5-6 (Processing):
   - processed_data/cleaned_data.csv
   - processed_data/transformed_data.csv

4. Final Steps:
   - results/transformer_model.pth
   - Various visualization outputs

## Validation Criteria

- All files created with expected formats
- No duplicate steps executed
- Proper error handling demonstrated
- Dependencies correctly checked
- Progress bar updates accurately
- Clear logging of each step

## Performance Metrics

Monitor and record:
- Execution time per step
- Memory usage
- Disk space requirements
- CPU utilization

## Recovery Procedures

Document steps to:
1. Clean up after failed runs
2. Resume from specific steps
3. Verify data integrity
4. Handle partial completions