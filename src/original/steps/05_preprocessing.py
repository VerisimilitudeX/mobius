#!/usr/bin/env python3
"""
05_preprocessing.py

Purpose:
  1) Load a CSV that typically has (row=sample) x (feature-columns) plus a "Condition" column.
  2) Remove duplicates.
  3) Check that "Condition" is present.
  4) Convert non-numeric columns (except "Condition") to numeric (coercing errors), cast numeric columns to float32,
     and fill numeric NaNs using column medians.
  5) Drop any columns that remain entirely NaN after median fill, but do NOT drop all columns if that leads to zero.
  6) Save cleaned CSV.

Usage example:
  python 05_preprocessing.py --csv /path/to/filtered_biomarker_matrix.csv \
                             --out /path/to/cleaned_data.csv \
                             --method auto --chunksize 100000
"""

import argparse
import logging
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def read_csv_auto(csv_path: str, chunksize: int = 100000) -> pd.DataFrame:
    logging.info("Reading header to count columns...")
    try:
        with open(csv_path, 'r') as f:
            header = f.readline().strip().split(',')
    except Exception as e:
        logging.error(f"Error reading header: {e}")
        sys.exit(1)
    num_cols = len(header)
    logging.info(f"Detected {num_cols} columns.")

    if num_cols > 10000:
        try:
            import polars as pl
            logging.info("Detected wide CSV; using Polars for fast reading.")
            df = pl.read_csv(csv_path).to_pandas()
        except ImportError:
            logging.warning("Polars not installed; falling back to pandas with low_memory=False.")
            df = pd.read_csv(csv_path, low_memory=False)
    else:
        logging.info(f"Using chunked pandas read (chunksize={chunksize}).")
        chunks = []
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            chunk = chunk.drop_duplicates()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    return df

def force_numeric_conversion(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    for col in df.columns:
        if col in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.info(f"Converted column '{col}' to numeric (non-numeric => NaN).")
    return df

def fill_numeric_medians(df: pd.DataFrame, condition_col: str = "Condition") -> pd.DataFrame:
    logging.info("Extracting numeric columns for median imputation...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        logging.warning("No numeric columns found!")
        return df

    # Convert to float32
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    logging.info(f"Numeric data shape => {df[numeric_cols].shape}")

    # Compute column medians
    numeric_data = df[numeric_cols].values
    medians = np.nanmedian(numeric_data, axis=0)
    median_series = pd.Series(medians, index=numeric_cols)
    df[numeric_cols] = df[numeric_cols].fillna(median_series)

    # Now drop columns that remain all-NaN
    all_na_cols = df[numeric_cols].columns[df[numeric_cols].isna().all()]
    if len(all_na_cols) > 0:
        # But ensure we do not drop *all* numeric columns
        if len(all_na_cols) == len(numeric_cols):
            logging.warning("All numeric columns are fully NaN. Preserving them to avoid zero columns.")
        else:
            df.drop(columns=all_na_cols, inplace=True)
            logging.info(f"Dropped {len(all_na_cols)} columns that were all-NaN after median fill.")

    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing (fill NA, drop duplicates).")
    parser.add_argument("--csv", required=True,
                        help="Input CSV with row=sample, columns=features, last col=Condition.")
    parser.add_argument("--out", required=True,
                        help="Output CSV path for cleaned data.")
    parser.add_argument("--method", default="auto", choices=["auto", "chunked", "polars"],
                        help="Reading method. 'auto' tries best approach automatically.")
    parser.add_argument("--chunksize", type=int, default=100000,
                        help="Chunk size for reading CSV if chunked approach is used.")
    args = parser.parse_args()

    start_time = time.time()

    if not os.path.exists(args.csv):
        logging.error(f"CSV not found: {args.csv}")
        sys.exit(1)

    if args.method == "auto":
        df = read_csv_auto(args.csv, chunksize=args.chunksize)
    elif args.method == "polars":
        try:
            import polars as pl
            logging.info("Using Polars to read CSV with default threading.")
            df_pl = pl.read_csv(args.csv)
            df = df_pl.to_pandas()
            df = df.drop_duplicates()
        except ImportError:
            logging.warning("Polars not installed; falling back to auto mode.")
            df = read_csv_auto(args.csv, chunksize=args.chunksize)
    else:  # chunked
        chunks = []
        logging.info(f"Using chunked pandas read (chunksize={args.chunksize}).")
        for chunk in pd.read_csv(args.csv, chunksize=args.chunksize):
            chunk = chunk.drop_duplicates()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    logging.info(f"After reading, shape = {df.shape}")

    # Drop any duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logging.info(f"Dropped {before - after} duplicate rows; new shape = {df.shape}")

    if "Condition" not in df.columns:
        logging.error("'Condition' column not found; aborting.")
        sys.exit(1)
    df["Condition"] = df["Condition"].astype(str)

    # Convert everything except Condition => numeric
    df = force_numeric_conversion(df, exclude_cols=["Condition"])
    df = fill_numeric_medians(df, "Condition")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logging.info(f"Created directory {out_dir}.")

    logging.info(f"Saving cleaned CSV to {args.out} with final shape {df.shape}")
    df.to_csv(args.out, index=False)

    elapsed_time = time.time() - start_time
    logging.info(f"Preprocessing complete in {elapsed_time:.2f} seconds.")
    logging.info("=== Done Preprocessing. ===")

if __name__ == "__main__":
    main()