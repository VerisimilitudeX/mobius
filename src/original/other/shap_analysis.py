#!/usr/bin/env python3
"""
shap_analysis.py

Purpose:
  - Loads or trains a multi-class RandomForest model.
  - Evaluates on test set.
  - Uses shap.Explainer for multi-class RandomForest.
  - Additionally applies Integrated Gradients (via Captum) for model interpretability.
  - Saves separate SHAP beeswarm plots and an overall bar chart, plus integrated gradients plots.

Usage:
  python shap_analysis.py
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional: Captum for Integrated Gradients
try:
    from captum.attr import IntegratedGradients
    HAVE_CAPTUM = True
except ImportError:
    HAVE_CAPTUM = False

CSV_DATA = "transformed_data.csv"
RF_MODEL_PATH = "best_rf.joblib"
OUTPUT_PREFIX = "shap_summary"
CLASS_NAMES = ["Control", "LC", "ME"]  # adjust as needed

def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Condition" not in df.columns:
        raise ValueError("[ERROR] 'Condition' column not found.")
    feat_cols = [c for c in df.columns if c != "Condition"]
    X = df[feat_cols].values
    conds = df["Condition"].values
    classes = sorted(np.unique(conds))
    c_map = {c: i for i, c in enumerate(classes)}
    y = np.array([c_map[v] for v in conds], dtype=int)
    return X, y, feat_cols, classes

def train_and_save_rf(X, y, out_path=RF_MODEL_PATH):
    print(f"[INFO] No existing RF => training new RandomForest.")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, out_path)
    print("[INFO] Done training & saved model as", out_path)

def main():
    X, y, feat_cols, detected_classes = load_data(CSV_DATA)
    if CLASS_NAMES and len(CLASS_NAMES) == len(detected_classes):
        class_names = CLASS_NAMES
    else:
        class_names = detected_classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if os.path.exists(RF_MODEL_PATH):
        print(f"[INFO] Found existing RF model. Loading it.")
        rf = joblib.load(RF_MODEL_PATH)
    else:
        train_and_save_rf(X_train_s, y_train, RF_MODEL_PATH)
        rf = joblib.load(RF_MODEL_PATH)
    preds_test = rf.predict(X_test_s)
    acc = np.mean(preds_test == y_test)
    print(f"[INFO] Test Accuracy => {acc:.4f}")
    cm = confusion_matrix(y_test, preds_test)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, preds_test, target_names=class_names))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("RF Confusion Matrix")
    plt.savefig("rf_confusion_matrix.png", dpi=300)
    plt.close()
    # SHAP analysis
    explainer = shap.Explainer(rf, X_train_s)
    shap_values = explainer(X_test_s)
    # Generate beeswarm plots for each class
    n_classes = shap_values.values.shape[1]
    for c_idx in range(n_classes):
        class_label = class_names[c_idx] if c_idx < len(class_names) else f"Class{c_idx}"
        shap.plots.beeswarm(shap_values[:, c_idx, :], max_display=20, show=False)
        plt.title(f"SHAP Beeswarm: {class_label}")
        out_beeswarm = f"{OUTPUT_PREFIX}_beeswarm_class_{c_idx}.png"
        plt.savefig(out_beeswarm, dpi=300, bbox_inches="tight")
        plt.close()
    shap_abs_mean = np.abs(shap_values.values).mean(axis=0)
    plt.figure(figsize=(8,6))
    sorted_idx = np.argsort(shap_abs_mean)[::-1][:20]
    sns.barplot(x=shap_abs_mean[sorted_idx], y=np.array(feat_cols)[sorted_idx], palette="viridis")
    plt.title("SHAP Bar (Average |SHAP|)")
    out_bar = f"{OUTPUT_PREFIX}_bar_overall.png"
    plt.savefig(out_bar, dpi=300, bbox_inches="tight")
    plt.close()
    # Integrated Gradients analysis (if Captum available)
    if HAVE_CAPTUM:
        print("[INFO] Running Integrated Gradients analysis using Captum.")
        # Wrap the RF model in a simple PyTorch model
        # Uses a simple linear approximation.
        class SimpleRFWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # x: [batch, features]
                # Using model.predict_proba as a proxy; here we simply convert input to tensor.
                x_np = x.detach().cpu().numpy()
                preds = self.model.predict_proba(x_np)
                return torch.tensor(preds, dtype=torch.float32, device=x.device)
        rf_wrapper = SimpleRFWrapper(rf)
        ig = IntegratedGradients(rf_wrapper)
        attributions, delta = ig.attribute(torch.tensor(X_test_s, dtype=torch.float32), 
                                           target=1, return_convergence_delta=True)
        plt.figure(figsize=(10,4))
        plt.bar(range(len(feat_cols)), attributions.mean(dim=0).cpu().numpy())
        plt.xticks(range(len(feat_cols)), feat_cols, rotation=90)
        plt.title("Integrated Gradients (Average Attribution for Class 1)")
        plt.tight_layout()
        plt.savefig("integrated_gradients_class1.png", dpi=300)
        plt.close()
        print("Integrated Gradients plot saved as integrated_gradients_class1.png")
    else:
        print("[INFO] Captum not installed; skipping Integrated Gradients analysis.")
    print("=== SHAP analysis complete. ===")

if __name__ == "__main__":
    main()
