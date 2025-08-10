#!/usr/bin/env python3
"""
Hugging Face dataset loader utilities for methylation CSV-style datasets.

Supports two modes:
- Hub dataset repos (arrow/parquet): load via datasets.load_dataset(<repo_id>, split=...)
- Direct CSV files: load via datasets.load_dataset('csv', data_files=...)

Returns a pandas DataFrame with a required 'Condition' column.
"""

from __future__ import annotations

import os
from typing import Optional, Union, Dict, Any

import pandas as pd


def _lazy_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face 'datasets' package is required. Add it to requirements and install."
        ) from e
    return load_dataset


def load_methylation_dataframe(
    *,
    hf_dataset: Optional[str] = None,
    hf_split: str = "train",
    hf_config: Optional[str] = None,
    data_files: Optional[Union[str, Dict[str, Union[str, list]]]] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a methylation dataset into a pandas DataFrame with a 'Condition' column.

    Args:
        hf_dataset: HF hub dataset repo id (e.g., "user/methylation-betas").
        hf_split: Split name to load (default: "train").
        hf_config: Optional config name on the hub.
        data_files: For CSV/JSON loaders; may be a path/URL or dict of split->path(s).
        csv_path: Local CSV path fallback.

    Returns:
        pandas.DataFrame
    """
    # Local CSV takes precedence if provided explicitly
    if csv_path:
        df = pd.read_csv(csv_path)
        if "Condition" not in df.columns:
            raise ValueError("Expected a 'Condition' column in the CSV.")
        return df

    load_dataset = _lazy_import_datasets()

    if hf_dataset:
        # Load an arrow dataset from the hub then convert to pandas
        ds = load_dataset(hf_dataset, name=hf_config, split=hf_split)
        df = ds.to_pandas()
        if "Condition" not in df.columns:
            raise ValueError("HF dataset missing required 'Condition' column.")
        return df

    if data_files:
        # CSV/JSON/Parquet by explicit files (local or remote URL)
        # Try CSV first; datasets auto-detects by extension.
        ds = load_dataset(path="csv", data_files=data_files, split=hf_split if isinstance(data_files, dict) else None)
        df = ds.to_pandas()
        if "Condition" not in df.columns:
            raise ValueError("Loaded data missing required 'Condition' column.")
        return df

    raise ValueError("Provide either csv_path, hf_dataset, or data_files to load the dataset.")

