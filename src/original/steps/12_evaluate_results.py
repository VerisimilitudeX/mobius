#!/usr/bin/env python3
"""
12_evaluate_results.py

Purpose:
  Loads the saved advanced transformer model and evaluates it on the dataset.
  Prints the confusion matrix and classification report.
  
Usage:
  python 12_evaluate_results.py
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from transformer_classifier import (
    MethylationChunkedDataset,
    methylation_collate_fn,
    TransformerClassifier,
    CHUNK_SIZE
)

def evaluate_transformer(model_path, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MethylationChunkedDataset(csv_path)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=methylation_collate_fn)
    seq_len_init = max(len(chunks) for chunks in dataset.data_chunks)
    num_classes = len(dataset.classes)
    model = TransformerClassifier(seq_len=seq_len_init, num_classes=num_classes).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            _, predicted = torch.max(out, dim=1)
            preds_all.append(predicted.cpu().numpy())
            labels_all.append(y_batch.cpu().numpy())
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    cm = confusion_matrix(labels_all, preds_all)
    print("[CONFUSION MATRIX]\n", cm)
    print("\n[CLASSIFICATION REPORT]\n",
          classification_report(labels_all, preds_all, digits=4))

def main():
    model_path = "/Volumes/T9/EpiMECoV/results/transformer_model.pth"
    data_csv = "/Volumes/T9/EpiMECoV/processed_data/filtered_biomarker_matrix.csv"
    evaluate_transformer(model_path, data_csv)

if __name__ == "__main__":
    main()