#!/usr/bin/env python3
"""
EPIGENETICS ANALYSIS + VISUALS FOR PAPER (MODIFIED)
==================================================
This script reads your 'filtered_biomarker_matrix.csv' from EpiMECoV/processed_data,
performs:
  • Data inspection
  • Baseline comparisons (Logistic Regression, RandomForest)
  • Transformer-based classification
  • Robust cross-validation, hold-out set
  • Sensitivity analysis (RF n_estimators)

Figures & CSV outputs are placed in EpiMECoV/results/.

Usage:
  python make_paper_figures.py

Modified to include:
  1) A hold-out set (20%) for final evaluation,
  2) Stratified K-Fold CV on the training portion,
  3) Logistic Regression baseline,
  4) Sensitivity Analysis example,
  5) Extended clarity on diagnostic prints.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA, TruncatedSVD

# ---------------------------------------------------------
# Resolve repository root dynamically to avoid hard-coded paths
# ---------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PROCESSED_DATA_DIR = os.path.join(REPO_ROOT, "processed_data")
BETA_FILE = os.path.join(PROCESSED_DATA_DIR, "filtered_biomarker_matrix.csv")

# Optional DMP results (if they exist):
DMP_ME_CTRL = os.path.join(RESULTS_DIR, "DMP_ME_vs_Control.csv")
DMP_LC_CTRL = os.path.join(RESULTS_DIR, "DMP_LC_vs_Control.csv")
DMP_ME_LC   = os.path.join(RESULTS_DIR, "DMP_ME_vs_LC.csv")

# ---------------------------------------------------------
# Helper Classes / Functions
# ---------------------------------------------------------
class DnamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TransformerClassifier(nn.Module):
    """Lightweight transformer baseline classifier used for figure generation."""
    def __init__(self, seq_len=128, num_heads=4, ff_dim=256, num_classes=3, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(seq_len, seq_len)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, seq_len))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=seq_len,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(seq_len, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        x = x.unsqueeze(-1)          # [batch_size, seq_len, 1]
        x = x.transpose(1, 2)        # [batch_size, 1, seq_len]
        seq_len = x.size(-1)

        # Simple positional encoding
        pe = self.pos_encoding[:, :seq_len, :].transpose(1,2)
        x = x + pe[..., :1]

        enc = self.transformer_encoder(x)
        enc = enc.mean(dim=-1)  # global avg pooling
        out = self.classifier(enc)
        return out

def plot_confusion_matrix(cm, classes, out_path, title="Confusion Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------------------------------------------------
# RandomForest sensitivity analysis utility
# ---------------------------------------------------------
def random_forest_sensitivity_analysis(X_train, y_train, X_valid, y_valid):
    """Evaluate macro-F1 across different n_estimators settings for RandomForest."""
    results = []
    n_estimators_list = [50, 100, 150, 200, 300, 500]
    for ne in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=ne, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_valid)
        f1 = f1_score(y_valid, preds, average='macro')
        results.append((ne, f1))

    # Plot or log the results
    fig, ax = plt.subplots(figsize=(6,4))
    n_list = [r[0] for r in results]
    f1_list = [r[1] for r in results]
    ax.plot(n_list, f1_list, marker='o')
    ax.set_title("Sensitivity: #Estimators vs. Macro-F1")
    ax.set_xlabel("#Estimators")
    ax.set_ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rf_sensitivity_analysis.png"))
    plt.close()

    # Print the table
    for (ne, score) in results:
        print(f"  [RF sensitivity] n_estimators={ne}, Macro-F1={score:.3f}")

# ---------------------------------------------------------
# Main Script
# ---------------------------------------------------------
def main():
    # 1) LOAD & INSPECT
    if not os.path.exists(BETA_FILE):
        raise FileNotFoundError(f"Cannot find {BETA_FILE}; please confirm path.")
    df = pd.read_csv(BETA_FILE, index_col=0)
    if 'Condition' not in df.columns:
        raise ValueError("No 'Condition' column in the CSV. Cannot proceed.")
    print("Loaded data with shape:", df.shape)
    print("Unique conditions BEFORE fix:", df['Condition'].unique())

    # Example fix merging 'Noel ME' into 'ME' if relevant:
    df.loc[df['Condition'] == "Noel ME", "Condition"] = "ME"
    print("Unique conditions AFTER fix:", df['Condition'].unique())
    print("Head:\n", df.head())

    # Convert condition => numeric
    cond_map = {c: i for i, c in enumerate(sorted(df['Condition'].unique()))}
    y = df['Condition'].map(cond_map).values
    X = df.drop(columns=['Condition']).values

    class_labels = sorted(cond_map.keys())
    print("Label distribution (by numeric code):")
    for k in np.unique(y):
        print(f"  Class {k} => {np.sum(y==k)} samples  (Condition={class_labels[k]})")

    # 2) Simple Summaries + PCA
    summary_cols = min(X.shape[1], 10)
    summary_stats = pd.DataFrame(X[:, :summary_cols]).describe()
    summary_stats.to_csv(os.path.join(RESULTS_DIR, "summary_stats_sampleFeatures.csv"))
    print("Saved sample stats to summary_stats_sampleFeatures.csv")

    # Quick PCA for visualization
    pca = PCA(n_components=2)
    max_cols_for_pca = 2000
    X_sub = X if X.shape[1] <= max_cols_for_pca else X[:, :max_cols_for_pca]
    X_pca = pca.fit_transform(X_sub)
    plt.figure(figsize=(7,5))
    scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.title("PCA (Top 2 Components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pca_plot.png"))
    plt.close()

    # 3) SPLIT into HOLD-OUT (for final testing) + everything else
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train/Val shape => {X_trainval.shape}, Hold-Out => {X_holdout.shape}")

    # 4) BASELINE COMPARISONS: (A) LOGISTIC REGRESSION, (B) RANDOM FOREST, (C) XGBoost
    #    Use STRATIFIED 10-FOLD on the training portion only (per paper)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ---- (A) Logistic Regression
    log_f1_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
        X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
        y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
        log_clf = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto', random_state=42)
        log_clf.fit(X_tr, y_tr)
        va_preds = log_clf.predict(X_va)
        f1_v = f1_score(y_va, va_preds, average='macro')
        log_f1_scores.append(f1_v)
        print(f"[LogReg fold {fold_i}] Macro-F1={f1_v:.3f}")
    avg_log_f1 = np.mean(log_f1_scores)
    print(f"Logistic Regression CV Macro-F1 => {avg_log_f1:.3f}")
    
    # ---- (B) Random Forest
    rf_f1_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
        X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
        y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
        rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_clf.fit(X_tr, y_tr)
        va_preds = rf_clf.predict(X_va)
        f1_v = f1_score(y_va, va_preds, average='macro')
        rf_f1_scores.append(f1_v)
        print(f"[RF fold {fold_i}] Macro-F1={f1_v:.3f}")
    avg_rf_f1 = np.mean(rf_f1_scores)
    print(f"Random Forest CV Macro-F1 => {avg_rf_f1:.3f}")

    # ---- (C) XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        xgb_f1_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
            X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
            y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            xgb.fit(X_tr, y_tr)
            va_preds = xgb.predict(X_va)
            f1_v = f1_score(y_va, va_preds, average='macro')
            xgb_f1_scores.append(f1_v)
            print(f"[XGB fold {fold_i}] Macro-F1={f1_v:.3f}")
        avg_xgb_f1 = np.mean(xgb_f1_scores)
        print(f"XGBoost CV Macro-F1 => {avg_xgb_f1:.3f}")
    except Exception as e:
        print("[WARN] XGBoost not available:", e)

    # Train final LogReg on entire trainval, evaluate on hold-out
    final_log = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='auto', random_state=42)
    final_log.fit(X_trainval, y_trainval)
    hold_preds = final_log.predict(X_holdout)
    hold_f1 = f1_score(y_holdout, hold_preds, average='macro')
    try:
        hold_auc = roc_auc_score(pd.get_dummies(y_holdout),
                                 final_log.predict_proba(X_holdout),
                                 average='macro', multi_class='ovr')
    except:
        hold_auc = np.nan
    print(f"[LogReg] HOLD-OUT => Macro-F1={hold_f1:.3f}, AUC={hold_auc:.3f}")
    cm_log = confusion_matrix(y_holdout, hold_preds)
    plot_confusion_matrix(cm_log, class_labels,
                          os.path.join(RESULTS_DIR, "logreg_holdout_confusion.png"),
                          title="Logistic Regression (Hold-Out)")

    # ---- (B) Random Forest (100 trees, tuned depth optional)
    rf_f1_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), start=1):
        X_tr, X_va = X_trainval[tr_idx], X_trainval[va_idx]
        y_tr, y_va = y_trainval[tr_idx], y_trainval[va_idx]
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_tr, y_tr)
        va_preds = rf_clf.predict(X_va)
        f1_v = f1_score(y_va, va_preds, average='macro')
        rf_f1_scores.append(f1_v)
        print(f"[RF fold {fold_i}] Macro-F1={f1_v:.3f}")
    avg_rf_f1 = np.mean(rf_f1_scores)
    print(f"RandomForest CV Macro-F1 => {avg_rf_f1:.3f}")

    # Train final RF on entire trainval, evaluate on hold-out
    final_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    final_rf.fit(X_trainval, y_trainval)
    hold_preds_rf = final_rf.predict(X_holdout)
    hold_f1_rf = f1_score(y_holdout, hold_preds_rf, average='macro')
    try:
        hold_auc_rf = roc_auc_score(pd.get_dummies(y_holdout),
                                    final_rf.predict_proba(X_holdout),
                                    average='macro', multi_class='ovr')
    except:
        hold_auc_rf = np.nan
    print(f"[RF] HOLD-OUT => Macro-F1={hold_f1_rf:.3f}, AUC={hold_auc_rf:.3f}")
    cm_rf = confusion_matrix(y_holdout, hold_preds_rf)
    plot_confusion_matrix(cm_rf, class_labels,
                          os.path.join(RESULTS_DIR, "rf_holdout_confusion.png"),
                          title="Random Forest (Hold-Out)")

    # (B.1) Optional Sensitivity Analysis for RandomForest
    # We'll do this using the train/val approach again
    # Here we do a single train/val split from the trainval set
    X_subtrain, X_subval, y_subtrain, y_subval = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=99, stratify=y_trainval
    )
    print("\n=== RandomForest Sensitivity Analysis ===")
    random_forest_sensitivity_analysis(X_subtrain, y_subtrain, X_subval, y_subval)

    # 5) Transformer baseline with Monte Carlo Dropout; train on trainval, eval on hold-out.
    from sklearn.decomposition import TruncatedSVD
    reduced_dim = 128
    # SVD to reduce dimensionality (if needed)
    if X_trainval.shape[1] > reduced_dim:
        svd = TruncatedSVD(n_components=reduced_dim, random_state=42)
        X_trainval_svd = svd.fit_transform(X_trainval)
        X_holdout_svd  = svd.transform(X_holdout)
        actual_seq_len = reduced_dim
    else:
        X_trainval_svd = X_trainval
        X_holdout_svd  = X_holdout
        actual_seq_len = X_trainval.shape[1]

    train_data = DnamDataset(X_trainval_svd, y_trainval)
    hold_data  = DnamDataset(X_holdout_svd,  y_holdout)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    hold_loader  = DataLoader(hold_data, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trans = TransformerClassifier(seq_len=actual_seq_len,
                                        num_classes=len(class_labels)).to(device)
    optimizer = optim.Adam(model_trans.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    epochs = 15
    for epoch in range(epochs):
        model_trans.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model_trans(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Transformer] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")

    # MC-Dropout Inference
    model_trans.train()  # keep dropout in training mode
    mc_samples = 20
    hold_preds_list = []
    with torch.no_grad():
        for _ in range(mc_samples):
            tmp_preds = []
            for bx, by in hold_loader:
                bx = bx.to(device)
                out_logits = model_trans(bx)
                preds = torch.argmax(out_logits, dim=1).cpu().numpy()
                tmp_preds.append(preds)
            hold_preds_list.append(np.concatenate(tmp_preds))
    hold_preds_array = np.stack(hold_preds_list, axis=0)
    # majority vote
    final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=hold_preds_array)

    # Evaluate
    trans_f1 = f1_score(y_holdout, final_preds, average='macro')
    try:
        # approximate mean prob for AUC
        with torch.no_grad():
            mc_prob_sum = np.zeros((len(X_holdout_svd), len(class_labels)), dtype=np.float32)
            for _ in range(mc_samples):
                preds_prob = []
                start_idx = 0
                for bx, by in hold_loader:
                    bx = bx.to(device)
                    out_logits = model_trans(bx)
                    prob = nn.Softmax(dim=1)(out_logits).cpu().numpy()
                    batch_size = prob.shape[0]
                    mc_prob_sum[start_idx:start_idx+batch_size, :] += prob
                    start_idx += batch_size
            mc_prob_mean = mc_prob_sum / mc_samples
        trans_auc = roc_auc_score(pd.get_dummies(y_holdout), mc_prob_mean, average='macro', multi_class='ovr')
    except:
        trans_auc = np.nan
    print(f"[Transformer] HOLD-OUT => Macro-F1={trans_f1:.3f}, AUC={trans_auc:.3f}")
    cm_trans = confusion_matrix(y_holdout, final_preds)
    plot_confusion_matrix(cm_trans, class_labels,
                          os.path.join(RESULTS_DIR, "transformer_holdout_confusion.png"),
                          title="Transformer (Hold-Out)")

    # 6) If DMP results exist, optionally load them
    if os.path.exists(DMP_ME_CTRL):
        df_me_ctrl = pd.read_csv(DMP_ME_CTRL)
        print("Sample ME vs Control DMPs:\n", df_me_ctrl.head())
    if os.path.exists(DMP_LC_CTRL):
        df_lc_ctrl = pd.read_csv(DMP_LC_CTRL)
        print("Sample LC vs Control DMPs:\n", df_lc_ctrl.head())
    if os.path.exists(DMP_ME_LC):
        df_me_lc = pd.read_csv(DMP_ME_LC)
        print("Sample ME vs LC DMPs:\n", df_me_lc.head())

    print("All done. Results in EpiMECoV/results/ folder.")

if __name__ == "__main__":
    main()
