#!/usr/bin/env python3
import argparse
from pprint import pprint

from src.utils.hf_data import load_methylation_dataframe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", required=True, help="HF dataset repo id, e.g. VerisimilitudeX/mobius-data")
    ap.add_argument("--hf-split", default="train", help="Split name (default: train)")
    ap.add_argument("--hf-config", default=None, help="Config name (optional)")
    args = ap.parse_args()

    df = load_methylation_dataframe(hf_dataset=args.hf_dataset,
                                    hf_split=args.hf_split,
                                    hf_config=args.hf_config)
    print("Loaded DataFrame:")
    print(f" - shape: {df.shape}")
    print(f" - columns: {list(df.columns)[:10]}{'...' if len(df.columns)>10 else ''}")
    assert 'Condition' in df.columns, "Expected a 'Condition' column in the dataset"
    print(" - unique Condition values:")
    pprint(sorted(df['Condition'].astype(str).unique()))
    print("\nPreview:")
    print(df.head(3))


if __name__ == "__main__":
    main()

