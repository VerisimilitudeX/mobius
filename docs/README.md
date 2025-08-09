# EpiMECoV Analysis Pipeline

## Overview

EpiMECoV is a comprehensive analysis pipeline for epigenetic biomarker discovery, combining methylation data analysis with advanced machine learning techniques. The pipeline includes:

- Methylation data preprocessing and quality control
- Differential methylation analysis
- Feature engineering and selection
- Advanced machine learning models including transformer-based classification
- Interactive visualization and result analysis

## Project Structure

```
EpiMECoV/
├── web/                    # Web interface for pipeline execution
│   ├── css/               # Styling files
│   ├── js/               # JavaScript for interactivity
│   ├── server.py         # WebSocket server
│   └── index.html        # Main interface
├── src/
│   └── original/         # Core pipeline components
│       ├── steps/        # Individual analysis steps
│       ├── analysis/     # Analysis utilities
│       └── other/        # Additional tools
├── docs/                 # Documentation
├── processed_data/       # Intermediate processing results
├── results/             # Final analysis outputs
└── visuals/             # Generated visualizations
```

## Setup Instructions

1. Install Dependencies:
```bash
# Python dependencies
cd web
pip install -r requirements.txt

# R dependencies
Rscript -e "install.packages(c('progress', 'jsonlite'))"
```

2. Start the Web Interface:
```bash
cd web
python server.py
```

3. Access the Interface:
- Open your browser to http://localhost:8080
- The interface will load with all styles and scripts

## Pipeline Steps

1. **IDAT Verification (Step 0)**
   - Validates input IDAT files
   - Checks data integrity

2. **Data Unification (Step 1)**
   - Combines IDAT files
   - Performs initial data organization

3. **Quality Analysis (Step 2)**
   - Basic quality metrics
   - Data validation checks

4. **Differential Methylation (Step 3)**
   - Identifies differentially methylated positions
   - Statistical analysis

5. **Data Preparation (Step 4)**
   - Formats data for analysis
   - Initial filtering

6. **Feature Selection (Step 5)**
   - Identifies key methylation sites
   - Statistical feature importance

7. **Preprocessing (Step 6)**
   - Data normalization
   - Missing value handling

8. **Feature Engineering (Step 7)**
   - Advanced feature creation
   - Dimensionality reduction

9. **Clinical Feature Integration (Step 8)**
   - Combines methylation and clinical data
   - Feature correlation analysis

10. **Classification (Steps 9-10)**
    - Baseline models
    - Advanced transformer classification

11. **Results Analysis (Steps 11-13)**
    - Statistical summaries
    - Visualization generation

12. **Advanced Analysis (Steps 14-15)**
    - Ensemble methods
    - Additional validations

## Using the Web Interface

1. **Configuration**
   - Enter paths for your three conditions
   - Select starting step (0-15)
   - Configure any additional parameters

2. **Execution**
   - Click "Start Analysis" to begin
   - Monitor progress in real-time
   - View logs and status updates

3. **Results**
   - View quality metrics
   - Examine classification results
   - Explore visualizations

## Data Requirements

- Input data should be in IDAT format
- Three conditions are required for analysis
- File structure should follow:
```
condition1/
├── sample1_Red.idat
├── sample1_Grn.idat
└── ...
condition2/
└── ...
condition3/
└── ...
```

## Scientific Background

The pipeline implements rigorous statistical and machine learning methods for epigenetic analysis:

1. **Methylation Analysis**
   - Beta value calculation
   - M-value transformation
   - Batch effect correction

2. **Feature Selection**
   - Statistical significance testing
   - Effect size estimation
   - Multiple testing correction

3. **Machine Learning**
   - Transformer architecture
   - Multimodal integration
   - Ensemble methods

4. **Validation**
   - Cross-validation
   - Independent test sets
   - Biological validation

## Output Files

The pipeline generates several key outputs:

1. **Quality Control**
   - Sample quality metrics
   - Detection p-values
   - Control probe performance

2. **Analysis Results**
   - Differential methylation tables
   - Feature importance scores
   - Classification metrics

3. **Visualizations**
   - PCA plots
   - Heatmaps
   - ROC curves
   - Confusion matrices

## Troubleshooting

Common issues and solutions:

1. **WebSocket Connection Issues**
   - Check server is running
   - Verify port 8080 is available
   - Check browser console for errors

2. **Pipeline Errors**
   - Verify input data format
   - Check file permissions
   - Ensure sufficient disk space

3. **Visualization Issues**
   - Verify results directory exists
   - Check file permissions
   - Ensure R libraries are installed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the terms of the LICENSE file in the docs directory.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{epimecov,
  title = {EpiMECoV: Epigenetic Biomarker Discovery Pipeline},
  year = {2025},
  author = {[Authors]},
  url = {[Repository URL]}
}
```

## Contact

For questions or support, please [create an issue](https://github.com/[username]/EpiMECoV/issues) in the repository.
