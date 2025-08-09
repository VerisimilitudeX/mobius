#!/usr/bin/env python3
"""
10_baseline_classification.py

Purpose:
  - Loads a CSV with row=sample, columns=selected features, last col=Condition.
  - Removes classes with fewer than 2 samples.
  - Splits data (train/test) and runs a quick RandomForest.
  - Prints confusion matrix & classification report.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

def main():
    csv_path = "./processed_data/filtered_biomarker_matrix.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] No CSV at {csv_path}. Did you run '09_feature_selection.R'?")
        return

    df = pd.read_csv(csv_path, index_col=0)
    if "Condition" not in df.columns:
        print("[ERROR] 'Condition' is missing in the final CSV.")
        return

    # Remove duplicates in columns or rows if any
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df[~df.index.duplicated()].copy()

    # X,y
    X = df.drop("Condition", axis=1).values
    y_str = df["Condition"].values.astype(str)

    # Remove classes with <2 samples
    class_counts = pd.Series(y_str).value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    mask_valid = pd.Series(y_str).isin(valid_classes)
    X = X[mask_valid]
    y_str = y_str[mask_valid]

    if len(np.unique(y_str)) < 2:
        print("[ERROR] After removing rare classes, not enough classes to classify.")
        return

    classes = sorted(np.unique(y_str))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[val] for val in y_str], dtype=int)

    n_samples = len(X)
    n_classes = len(classes)
    test_size = 0.3

    # Ensure we have enough test samples for all classes:
    # => test_size * n_samples >= n_classes
    if int(round(test_size * n_samples)) < n_classes:
        # pick the smallest fraction that yields at least n_classes in test
        recommended = float(n_classes + 1) / n_samples
        # clamp recommended to 0.5 max, so we keep some train
        recommended = min(recommended, 0.5)
        print(f"[INFO] Adjusting test_size from {test_size} to {recommended:.2f} to accommodate {n_classes} classes.")
        test_size = recommended

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, target_names=classes, digits=4))

    if len(classes) == 2:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_proba)
        print("[Binary] ROC AUC =>", auc_val)

    print("\nBaseline classification done.")

if __name__ == "__main__":
    main()