#!/usr/bin/env python3
"""
epigenetic_transformer_analysis.py

Title: Epigenetic Profiling: A Transformer Approach

Steps:
  1) Loads data from "transformed_data.csv" (with columns AE_0..AE_n, Condition).
  2) Splits into train/val/test, trains a custom Transformer.
  3) Evaluates performance, confusion matrix, classification report.
  4) Stats-based feature analysis (ANOVA) => identifies differential features.

Outputs:
  - Plots + textual summary in "epigenetic_analysis_report.txt"
  - "epigenetic_stats.csv" with p-values and effect sizes
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "transformed_data.csv"
EPOCHS = 400
BATCH_SIZE = 32
LR = 1e-4
HID_DIM = 64
N_LAYERS = 2
N_HEADS = 4
DROP_PROB = 0.1
TEST_SIZE = 0.2
VAL_SIZE = 0.2
SEED = 42
REPORT_TXT = "epigenetic_analysis_report.txt"
REPORT_CSV = "epigenetic_stats.csv"


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out, _ = self.mha(x, x, x)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out = self.norm(residual + out)
        return out


class TabularTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_layers=2, n_heads=4, d_ff=256, dropout=0.1, num_classes=3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim, d_model))

        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            attn = SelfAttentionBlock(d_model, n_heads, dropout)
            ff = FeedForwardBlock(d_model, d_ff, dropout)
            self.blocks.append(nn.ModuleList([attn, ff]))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        bsz, seq_len = x.shape
        x = x.unsqueeze(-1)
        x = self.embed(x)
        x = x + self.pos_embed[:, :seq_len, :]
        for attn, ff in self.blocks:
            x = attn(x)
            x = ff(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if "Condition" not in df.columns:
        print("[ERROR] Condition col missing.")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c != "Condition"]
    X = df[feature_cols].values.astype(np.float32)
    conds = df["Condition"].values.astype(str)
    class_names = sorted(list(np.unique(conds)))
    c2i = {cname: i for i, cname in enumerate(class_names)}
    y = np.array([c2i[v] for v in conds], dtype=np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=y_trainval,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformer(
        input_dim=X_train_s.shape[1],
        d_model=HID_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=HID_DIM * 4,
        dropout=DROP_PROB,
        num_classes=len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    def make_loader(Xf, yf):
        tX = torch.from_numpy(Xf)
        ty = torch.from_numpy(yf)
        ds = torch.utils.data.TensorDataset(tX, ty)
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    train_loader = make_loader(X_train_s, y_train)
    val_loader   = make_loader(X_val_s, y_val)

    best_val_loss = float("inf")
    best_sd = None
    patience = 50
    no_improve = 0
    train_losses, val_losses = [], []

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        model.eval()
        total_val = 0
        correct = 0
        total_samp = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                v_log = model(bx)
                v_loss = criterion(v_log, by)
                total_val += v_loss.item()
                _, preds = torch.max(v_log, 1)
                correct += (preds == by).sum().item()
                total_samp += by.size(0)

        avg_val = total_val / len(val_loader)
        val_acc = correct / total_samp

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if ep % 50 == 0 or ep == 1 or ep == EPOCHS:
            print(f"[Epoch {ep}/{EPOCHS}] train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, val_acc={val_acc:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_sd = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] No improvement {patience} epochs => stop at ep={ep}")
                break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    # Evaluate
    test_ds = make_loader(X_test_s, y_test)
    model.eval()
    preds_all = []
    truths_all = []
    for bx, by in test_ds:
        bx, by = bx.to(device), by.to(device)
        with torch.no_grad():
            out = model(bx)
            predicted = torch.argmax(out, dim=1)
            preds_all.append(predicted.cpu().numpy())
            truths_all.append(by.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    truths_all = np.concatenate(truths_all)

    acc_test = np.mean(preds_all == truths_all)
    print(f"[TEST ACCURACY] => {acc_test:.4f}")

    cm = confusion_matrix(truths_all, preds_all)
    print("Confusion Matrix:\n", cm)

    cr = classification_report(truths_all, preds_all, target_names=class_names)
    print("Classification Report:\n", cr)

    # Confusion matrix plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Transformer Conf Matrix")
    plt.savefig("transformer_confusion_matrix.png", dpi=300)
    plt.close()

    # Stats-based feature diffs
    X_full_s = scaler.fit_transform(X)
    results = []
    for i, feat in enumerate(feature_cols):
        groups = []
        for cls_i in range(len(class_names)):
            groups.append(X_full_s[y==cls_i, i])
        try:
            fval, pval = stats.f_oneway(*groups)
            k = len(class_names)
            N = sum(len(g) for g in groups)
            eta_sq = (fval * (k - 1)) / ((fval * (k - 1)) + (N - k)) if fval > 0 else 0
            if pval < 1e-14:
                pval = 1e-14
            results.append((feat, pval, eta_sq))
        except:
            hval, pval = stats.kruskal(*groups)
            results.append((feat, pval, 0.0))

    res_sorted = sorted(results, key=lambda x: x[1])
    df_stats = pd.DataFrame(res_sorted, columns=["Feature","p_value","eta_sq_est"])
    df_stats.to_csv("epigenetic_stats.csv", index=False)
    print("[INFO] Wrote stats => epigenetic_stats.csv")

    p_thresh = 0.05
    differential = [r for r in res_sorted if r[1] < p_thresh]
    similar = [r for r in res_sorted if r[1] >= p_thresh]

    top_20 = differential[:20]
    if len(top_20) > 0:
        feats = [x[0] for x in top_20]
        pvals = [x[1] for x in top_20]
        plt.figure(figsize=(6,5))
        sns.barplot(x=-np.log10(pvals), y=feats, color="red")
        plt.title("Top 20 Diff Features")
        plt.savefig("top20_differential_features.png", dpi=300)
        plt.close()

    lines = []
    lines.append("=== Epigenetic Transformer Analysis ===")
    lines.append(f"Data shape => {X.shape}, #features={X.shape[1]}, classes={class_names}")
    lines.append(f"Transformer Test Accuracy => {acc_test:.4f}")
    lines.append("Confusion Matrix:\n" + str(cm))
    lines.append("Classification Report:\n" + cr)
    lines.append(f"\n=== Stats-based Feature Differences (ANOVA p<{p_thresh}) ===")
    lines.append(f"Found {len(differential)} features significantly differ.")
    lines.append(f"Found {len(similar)} features appear 'similar'.")
    lines.append("Sample top 10 diffs [feat, p_val, eta_sq]:")
    for x in differential[:10]:
        lines.append(f" - {x[0]} => p={x[1]:.4e}, eta^2~{x[2]:.3f}")

    lines.append("\n=== Potential Similarities ===")
    for x in similar[:10]:
        lines.append(f" - {x[0]} => p={x[1]:.4e}")

    with open("epigenetic_analysis_report.txt","w") as f:
        f.write("\n".join(lines))

    print("[DONE] Wrote epigenetic_analysis_report.txt\n")

if __name__ == "__main__":
    main()