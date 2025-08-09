#!/usr/bin/env python3
"""
This script recursively processes IDAT files from a specified base directory,
extracts beta values using methylprep, and computes the fractal dimension of
each sample using a box-counting algorithm. The results (including log–log plots)
are saved for further analysis.

If required packages are not installed, the script attempts to install them automatically.
"""

import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["methylprep", "numpy", "matplotlib", "pandas"]

# Check for each package and install if missing
for package in required_packages:
    try:
        __import__(package)
    except ModuleNotFoundError:
        print(f"Package '{package}' not found. Installing {package}...")
        install(package)

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from methylprep import run_pipeline  # For processing IDAT files

def box_count(points, box_size, x_min, x_max, y_min, y_max):
    """
    Count the number of boxes of side length `box_size` needed to cover the set of points.
    Uses a 2D histogram to count boxes with at least one point.
    
    Parameters:
        points (np.ndarray): Array of shape (N,2) with (x,y) coordinates.
        box_size (float): The side length of the square box.
        x_min, x_max, y_min, y_max (float): Bounds of the data.
    
    Returns:
        count (int): Number of boxes that contain at least one point.
    """
    x_bins = np.arange(x_min, x_max + box_size, box_size)
    y_bins = np.arange(y_min, y_max + box_size, box_size)
    H, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[x_bins, y_bins])
    count = np.sum(H > 0)
    return count

def compute_fractal_dimension(points, box_sizes):
    """
    Estimate the fractal dimension of a set of points using the box-counting method.
    
    Parameters:
        points (np.ndarray): Array of shape (N,2) of points in the normalized domain.
        box_sizes (np.ndarray): Array of box sizes to use for the analysis.
    
    Returns:
        fractal_dim (float): Estimated fractal dimension (slope from log–log regression).
        bs_used (np.ndarray): The box sizes used.
        counts (list): List of box counts corresponding to each box size.
        coeffs (np.ndarray): Coefficients of the linear fit (slope and intercept).
    """
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    
    counts = []
    for box_size in box_sizes:
        count = box_count(points, box_size, x_min, x_max, y_min, y_max)
        counts.append(count)
    
    counts = np.array(counts)
    valid = counts > 0
    log_counts = np.log(counts[valid])
    log_inv_box_sizes = np.log(1 / box_sizes[valid])
    
    coeffs = np.polyfit(log_inv_box_sizes, log_counts, 1)
    fractal_dim = coeffs[0]
    
    return fractal_dim, box_sizes, counts, coeffs

def main():
    # Define the base directory containing the IDAT files (including subfolders)
    base_dir = "/Volumes/T9/EpiMECoV/data"
    print("Running methylprep pipeline on directory:", base_dir)
    
    # Process all IDAT files; returns a DataFrame with beta values where
    # rows correspond to probes and columns correspond to samples.
    df = run_pipeline(base_dir, export=False)
    print("Loaded beta values with shape:", df.shape)
    
    results = {}
    # Define a range of box sizes for the analysis (domain is [0,1]x[0,1])
    bs = np.linspace(0.01, 0.1, 20)
    
    # Process each sample (each column in the DataFrame)
    for sample in df.columns:
        beta_values = df[sample].values
        # x coordinates: equally spaced indices normalized to [0,1]
        x = np.linspace(0, 1, len(beta_values))
        # y coordinates are the beta values (assumed to be in [0,1])
        y = beta_values
        points = np.column_stack((x, y))
        
        # Compute fractal dimension using the box-counting method
        fractal_dim, bs_used, counts, coeffs = compute_fractal_dimension(points, bs)
        results[sample] = fractal_dim
        
        # Generate and save a log–log plot for diagnostic purposes
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(1 / bs_used), np.log(counts), 'o-', label=f'Fractal dim = {fractal_dim:.3f}')
        plt.xlabel('log(1/box size)')
        plt.ylabel('log(box count)')
        plt.title(f'Box-Counting Analysis for Sample {sample}')
        plt.legend()
        plot_filename = f'box_count_{sample}.png'
        plt.savefig(plot_filename)
        plt.close()
        print(f"Sample '{sample}' fractal dimension: {fractal_dim:.3f} (plot saved as {plot_filename})")
    
    # Save fractal dimension results to a text file
    with open("fractal_dimensions.txt", "w") as f:
        for sample, fd in results.items():
            f.write(f"{sample}\t{fd:.4f}\n")
    print("Fractal dimension analysis complete. Results saved to 'fractal_dimensions.txt'.")

if __name__ == '__main__':
    main()