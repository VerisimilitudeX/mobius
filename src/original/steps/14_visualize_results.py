#!/usr/bin/env python3
"""
14_visualize_results.py

Purpose:
  Generate a suite of detailed visuals (PCA, t-SNE, UMAP, heatmaps, etc.)
  from the final data. This version has been optimized by using fast,
  vectorized imputation (handling NaN and infinite values), subsampling when
  datasets are huge, and safety checks for expensive computations.
  
Usage:
  python3 14_visualize_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import scipy.cluster.hierarchy as sch
import networkx as nx

try:
    from upsetplot import UpSet
    HAVE_UPSETPLOT = True
except ImportError:
    HAVE_UPSETPLOT = False

try:
    import umap
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except ImportError:
    HAVE_PLOTLY = False

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# ----------------------------
# Helper: safe_call to catch exceptions
# ----------------------------
def safe_call(func, *args, **kwargs):
    """Wrapper to run a plotting function and catch any exceptions."""
    try:
        func(*args, **kwargs)
    except Exception as e:
        print(f"[ERROR] {func.__name__} failed: {e}")

# ----------------------------
# Optimized impute_and_scale using median fill (vectorized)
# ----------------------------
def impute_and_scale(df, feat_cols):
    """
    Imputes missing values in the specified feature columns using the median and
    applies StandardScaler. Replaces infinities and ensures no NaN remains.
    """
    if len(feat_cols) == 0:
        print("[ERROR] No numeric features. Skipping impute_and_scale.")
        return None
    feat_cols_in_df = [c for c in feat_cols if c in df.columns]
    if len(feat_cols_in_df) == 0:
        print("[ERROR] None of feat_cols exist in df.columns. Skipping.")
        return None

    X_orig = df[feat_cols_in_df]
    # Replace infinities with NaN and fill missing values with median
    X_orig = X_orig.replace([np.inf, -np.inf], np.nan)
    X_imputed = X_orig.fillna(X_orig.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed.values)
    # Ensure any remaining NaN or infinities are set to zero
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return X_scaled

# ----------------------------
# Plotting functions
# ----------------------------
def plot_volcano(dmp_csv, out_path):
    df = pd.read_csv(dmp_csv)
    if 'P.Value' not in df.columns or 'logFC' not in df.columns:
        raise ValueError("CSV file must contain 'P.Value' and 'logFC' columns.")
    df['-log10(P.Value)'] = -np.log10(df['P.Value'])

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='logFC', y='-log10(P.Value)', hue='-log10(P.Value)',
                    palette="viridis", alpha=0.7)
    plt.title("Volcano Plot")
    plt.xlabel("Log Fold Change")
    plt.ylabel("-Log10(P.Value)")
    plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='P=0.05')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Volcano plot => {out_path}")

def plot_scatter_box(df, scatter_out, box_out):
    # Reset index to ensure numeric positions for boxplot
    df = df.reset_index(drop=True)
    feat_cols = [col for col in df.columns if col != "Condition"]
    if not feat_cols:
        print("[INFO] No feature columns available for scatter/box plots.")
        return
    feature = feat_cols[0]

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df.index, y=df[feature], hue=df["Condition"], alpha=0.7)
    plt.title(f"Scatter Plot for {feature}")
    plt.xlabel("Sample Index")
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.savefig(scatter_out, dpi=300)
    plt.close()
    print(f"[SAVED] Scatter plot => {scatter_out}")

    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x="Condition", y=feature)
    plt.title(f"Box Plot for {feature}")
    plt.xlabel("Condition")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(box_out, dpi=300)
    plt.close()
    print(f"[SAVED] Box plot => {box_out}")

def plot_age_distribution(df, out_path):
    if "Age" not in df.columns:
        print("[INFO] No 'Age' column found; skipping age distribution plot.")
        return
    plt.figure(figsize=(7,5))
    sns.histplot(df["Age"].dropna(), kde=True, bins=30, color="steelblue")
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Age distribution => {out_path}")

def plot_sex_distribution(df, out_path):
    if "Sex" not in df.columns:
        print("[INFO] No 'Sex' column found; skipping sex distribution plot.")
        return
    plt.figure(figsize=(7,5))
    sns.countplot(x=df["Sex"], palette="pastel")
    plt.title("Sex Distribution")
    plt.xlabel("Sex")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] Sex distribution => {out_path}")

def plot_tsne(Xs, y, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for t-SNE.")
        return

    n_samples = Xs.shape[0]
    if n_samples > 1000:
        np.random.seed(42)
        indices = np.random.choice(n_samples, 1000, replace=False)
        Xs = Xs[indices, :]
        y = y[indices]

    # Ensure Xs is finite
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    perplexity_value = max(1, min(Xs.shape[0] - 1, 30))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value,
                n_iter=250, init='random')
    X2d = tsne.fit_transform(Xs)

    plt.figure(figsize=(7,5))
    for cond in np.unique(y):
        idx = (y == cond)
        plt.scatter(X2d[idx,0], X2d[idx,1], label=cond, alpha=0.7)
    plt.title("t-SNE on Final Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] t-SNE => {out_path}")

def plot_tsne_with_perplexity(Xs, y, perplexity, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for t-SNE perplexity test.")
        return

    n_samples = Xs.shape[0]
    if perplexity >= n_samples:
        perplexity = n_samples - 1
    if perplexity < 1:
        print("[WARN] Not enough samples for this perplexity.")
        return

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                n_iter=250, init='random')
    X2d = tsne.fit_transform(Xs)

    plt.figure(figsize=(7,5))
    for cond in np.unique(y):
        idx = (y == cond)
        plt.scatter(X2d[idx,0], X2d[idx,1], label=cond, alpha=0.7)
    plt.title(f"t-SNE (Perplexity={perplexity})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] t-SNE (p={perplexity}) => {out_path}")

def plot_pca(Xs, y, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for PCA.")
        return

    if Xs.shape[0] > 1000:
        np.random.seed(42)
        indices = np.random.choice(Xs.shape[0], 1000, replace=False)
        Xs = Xs[indices, :]
        y = y[indices]

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(Xs)

    plt.figure(figsize=(7,5))
    for cond in np.unique(y):
        idx = (y == cond)
        plt.scatter(X_pca[idx,0], X_pca[idx,1], label=cond, alpha=0.7)
    plt.title("PCA on Final Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] PCA => {out_path}")

def plot_pca_scree(Xs, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for PCA Scree.")
        return

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(random_state=42)
    pca.fit(Xs)
    var_exp = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(var_exp)+1), var_exp, marker='o')
    plt.title("PCA Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] PCA scree => {out_path}")

def plot_pca_biplot(Xs, y, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for PCA biplot.")
        return

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(Xs)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette="viridis", alpha=0.7)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    for i in range(loadings.shape[0]):
        plt.arrow(0, 0, loadings[i,0], loadings[i,1], color='red', alpha=0.3)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Biplot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] PCA biplot => {out_path}")

def plot_umap_visualization(Xs, y, out_path):
    if not HAVE_UMAP:
        print("[INFO] UMAP not installed; skipping.")
        return
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for UMAP.")
        return

    if Xs.shape[0] > 1000:
        np.random.seed(42)
        indices = np.random.choice(Xs.shape[0], 1000, replace=False)
        Xs = Xs[indices, :]
        y = y[indices]

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(Xs)

    plt.figure(figsize=(7,5))
    for cond in np.unique(y):
        idx = (y == cond)
        plt.scatter(X_umap[idx,0], X_umap[idx,1], label=cond, alpha=0.7)
    plt.title("UMAP on Final Data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] UMAP => {out_path}")

def plot_correlation_heatmap(Xs, out_path):
    if Xs is None or Xs.shape[1] < 2:
        print("[WARN] Not enough features for correlation heatmap.")
        return

    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(Xs, rowvar=False)
    if np.isnan(corr).all():
        print("[WARN] correlation is all NaN; skipping heatmap.")
        return

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Features)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] correlation_heatmap => {out_path}")

def plot_hierarchical_dendrogram(Xs, out_path):
    if Xs is None or Xs.shape[0] < 2 or Xs.shape[1] == 0:
        print("[WARN] Not enough data for dendrogram.")
        return

    max_samples = 500
    if Xs.shape[0] > max_samples:
        np.random.seed(42)
        indices = np.random.choice(Xs.shape[0], max_samples, replace=False)
        Xs_sample = Xs[indices, :]
    else:
        Xs_sample = Xs

    Xs_sample = np.nan_to_num(Xs_sample, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        linkage = sch.linkage(Xs_sample, method='ward')
    except ValueError as e:
        print(f"[ERROR] Cannot cluster: {e}")
        return

    plt.figure(figsize=(10,7))
    sch.dendrogram(linkage)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] hierarchical_dendrogram => {out_path}")

def plot_box_feature(df, feature, out_path):
    if feature not in df.columns:
        print(f"[WARN] feature {feature} not in df; skipping box plot.")
        return

    plt.figure(figsize=(7,5))
    sns.boxplot(data=df, x="Condition", y=feature)
    plt.title(f"Boxplot of {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] boxplot_feature => {out_path}")

def plot_violin_feature(df, feature, out_path):
    if feature not in df.columns:
        print(f"[WARN] feature {feature} not in df; skipping violin plot.")
        return

    plt.figure(figsize=(7,5))
    sns.violinplot(data=df, x="Condition", y=feature)
    plt.title(f"Violin of {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] violin_feature => {out_path}")

def plot_pairplot(df, features, out_path):
    # Ensure all requested features are in df
    for feat in features:
        if feat not in df.columns:
            print(f"[WARN] feature {feat} not in df; skipping pairplot.")
            return

    sns_plot = sns.pairplot(df[features + ["Condition"]], hue="Condition")
    sns_plot.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] pairplot => {out_path}")

def plot_kde_feature(df, feature, out_path):
    if feature not in df.columns:
        print(f"[WARN] feature {feature} not found for KDE.")
        return

    plt.figure(figsize=(7,5))
    sns.kdeplot(data=df, x=feature, hue="Condition", fill=True)
    plt.title(f"KDE for {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] kde_plot => {out_path}")

def plot_condition_count(df, out_path):
    if "Condition" not in df.columns:
        print("[WARN] No Condition col for condition_count.")
        return

    plt.figure(figsize=(7,5))
    count_series = df["Condition"].value_counts()
    sns.barplot(x=count_series.index, y=count_series.values)
    plt.title("Sample Count per Condition")
    plt.xlabel("Condition")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] condition_count => {out_path}")

def plot_rf_feature_importance(feature_names, importances, top_n, out_path):
    idxs = np.argsort(importances)[::-1]
    top_idx = idxs[:top_n]

    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[top_idx], y=np.array(feature_names)[top_idx], palette="viridis")
    plt.title("Top Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] rf_importance => {out_path}")

def plot_classifier_confusion_matrix(cm, classes, out_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] confusion_matrix => {out_path}")

def plot_roc_curve(y_true, y_score, out_path):
    if len(np.unique(y_true)) > 2:
        print("[WARN] ROC not valid for multi-class. Skipping.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC (area={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] roc_curve => {out_path}")

def plot_precision_recall_curve(y_true, y_score, out_path):
    if len(np.unique(y_true)) > 2:
        print("[WARN] PR curve not valid for multi-class. Skipping.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"PR (area={pr_auc:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] pr_curve => {out_path}")

def plot_calibration_curve(y_true, y_prob, out_path):
    if len(np.unique(y_true)) > 2:
        print("[WARN] calibration curve not valid for multi-class.")
        return

    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(7,5))
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] calibration_curve => {out_path}")

def plot_sankey_diagram(out_path):
    if not HAVE_PLOTLY:
        print("[INFO] Plotly not installed; skipping Sankey.")
        return
    try:
        import kaleido
    except ImportError:
        print("[ERROR] kaleido not installed => skipping Sankey.")
        return

    labels = ["Start","GroupA","GroupB","End"]
    source = [0,0,1,2]
    target = [1,2,3,3]
    value  = [8,4,2,6]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Sankey Example", font_size=10)

    try:
        fig.write_image(out_path)
        print(f"[SAVED] sankey_diagram => {out_path}")
    except Exception as e:
        print(f"[ERROR] plot_sankey_diagram => {e}")

def plot_pca_3d(Xs, y, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for 3D PCA.")
        return
    if Xs.shape[0] < 2:
        print("[INFO] Not enough samples for 3D PCA.")
        return

    from mpl_toolkits.mplot3d import Axes3D
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(Xs)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for cond in np.unique(y):
        idx = (y == cond)
        ax.scatter(X_3d[idx,0], X_3d[idx,1], X_3d[idx,2], label=cond, alpha=0.7)
    ax.set_title("3D PCA Plot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] pca_3d_plot => {out_path}")

def plot_tsne_3d(Xs, y, out_path):
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] No valid features for 3D t-SNE.")
        return

    n_samples = Xs.shape[0]
    if n_samples < 3:
        print("[INFO] Not enough samples for 3D t-SNE.")
        return

    from mpl_toolkits.mplot3d import Axes3D
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    perp = min(30, max(1, n_samples-1))
    tsne = TSNE(n_components=3, random_state=42, perplexity=perp, n_iter=250, init='random')
    X_3d = tsne.fit_transform(Xs)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for cond in np.unique(y):
        idx = (y == cond)
        ax.scatter(X_3d[idx,0], X_3d[idx,1], X_3d[idx,2], label=cond, alpha=0.7)
    ax.set_title("3D t-SNE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] tsne_3d_plot => {out_path}")

def plot_multi_panel_summary(imputed_df, out_path):
    """
    Creates a multi-panel summary figure with PCA, t-SNE, UMAP, Scree plot, correlation heatmap,
    dendrogram, and box/violin plots for the first numeric feature, plus condition counts.
    """
    if imputed_df.shape[0] > 1000:
        np.random.seed(42)
        imputed_df = imputed_df.sample(n=1000, random_state=42)

    numeric_cols = [c for c in imputed_df.columns if c != "Condition"]
    if len(numeric_cols) == 0:
        print("[WARN] No numeric columns for multi-panel summary.")
        return

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    n_samples = imputed_df.shape[0]
    if n_samples < 2:
        print("[WARN] Not enough samples for multi-panel summary.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(15,15))

    # PCA (2D)
    X_mat = imputed_df[numeric_cols].values
    pca = PCA(n_components=2, random_state=42)
    try:
        X_pca = pca.fit_transform(X_mat)
    except Exception as e:
        print(f"[ERROR] PCA in multi-panel summary failed: {e}")
        return

    ax = axes[0,0]
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=imputed_df["Condition"], ax=ax)
    ax.set_title("PCA Plot")

    # t-SNE
    if n_samples > 2:
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, n_samples-1), n_iter=250, init='random')
        X_tsne = tsne.fit_transform(X_mat)
        ax = axes[0,1]
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=imputed_df["Condition"], ax=ax)
        ax.set_title("t-SNE")
    else:
        axes[0,1].text(0.5,0.5,"Not enough samples",ha='center')
        axes[0,1].set_title("t-SNE")

    # UMAP
    ax = axes[0,2]
    if HAVE_UMAP and n_samples > 2:
        import umap
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_mat)
        sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=imputed_df["Condition"], ax=ax)
        ax.set_title("UMAP")
    else:
        ax.text(0.5,0.5,"UMAP not installed or not enough samples",
                ha='center', va='center')
        ax.set_title("UMAP")

    # PCA Scree
    pca_full = PCA(random_state=42)
    pca_full.fit(X_mat)
    var_exp = pca_full.explained_variance_ratio_*100
    ax = axes[1,0]
    ax.plot(range(1,len(var_exp)+1), var_exp, marker='o')
    ax.set_title("PCA Scree")

    # Correlation heatmap (top 50 features)
    ax = axes[1,1]
    max_for_corr = min(50, len(numeric_cols))
    corr_data = X_mat[:, :max_for_corr]
    corr = np.corrcoef(corr_data, rowvar=False)
    if not np.isnan(corr).all():
        sns.heatmap(corr, ax=ax, cmap="coolwarm", cbar=False)
        ax.set_title("Corr Heatmap (top 50 features)")
    else:
        ax.text(0.5,0.5,"All NaN corr",ha='center')
        ax.set_title("Corr Heatmap")

    # Dendrogram
    ax = axes[1,2]
    try:
        link = sch.linkage(X_mat, method='ward')
        sch.dendrogram(link, ax=ax, no_labels=True)
        ax.set_title("Dendrogram")
    except Exception as e:
        ax.text(0.5,0.5,f"Dendrogram failed:\n{e}",ha='center')
        ax.set_title("Dendrogram")

    # Box + Violin for first numeric feature
    ax = axes[2,0]
    first_feat = numeric_cols[0]
    sns.boxplot(data=imputed_df, x="Condition", y=first_feat, ax=ax)
    ax.set_title(f"Box: {first_feat}")

    ax = axes[2,1]
    sns.violinplot(data=imputed_df, x="Condition", y=first_feat, ax=ax)
    ax.set_title(f"Violin: {first_feat}")

    # Condition count
    ax = axes[2,2]
    ccount = imputed_df["Condition"].value_counts()
    sns.barplot(x=ccount.index, y=ccount.values, ax=ax)
    ax.set_title("Condition Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] multi_panel_summary => {out_path}")

def plot_upset(dmp_files, out_path):
    """
    If upsetplot is available, we attempt to show an UpSet plot of overlapping DMP sets.
    DMP CSVs must have at least one column with CpG IDs (named 'CpG' or else we use first col).
    """
    if not HAVE_UPSETPLOT:
        print("[INFO] upsetplot not installed; skipping UpSet.")
        return

    sets = {}
    for key, path in dmp_files.items():
        if not os.path.exists(path):
            print(f"[WARN] {path} missing => skipping in UpSet.")
            continue
        df = pd.read_csv(path)
        if 'CpG' in df.columns:
            cpgs = df['CpG'].dropna().astype(str).values
        else:
            cpgs = df.iloc[:,0].dropna().astype(str).values
        sets[key] = set(cpgs)

    if len(sets)==0:
        print("[WARN] No DMP sets for UpSet.")
        return

    all_union = set().union(*sets.values())
    if len(all_union)==0:
        print("[WARN] No CpGs in union => skip upset.")
        return

    data = {}
    for key, s in sets.items():
        data[key] = [1 if x in s else 0 for x in all_union]

    upset_df = pd.DataFrame(data, index=list(all_union))
    from upsetplot import from_indicators, plot as upset_plot
    upset_data = from_indicators(upset_df)

    fig, ax = plt.subplots(figsize=(8,6))
    upset_plot(upset_data, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] intersection_plot => {out_path}")

def plot_network(dmp_csv, out_path):
    """
    Currently a 'dummy' chain network linking top 50 CpGs from the DMP CSV.
    For a biologically meaningful approach, you might compute co-methylation
    edges or shared pathway edges, etc.
    """
    if not os.path.exists(dmp_csv):
        raise FileNotFoundError(dmp_csv)

    df = pd.read_csv(dmp_csv)
    if 'CpG' in df.columns:
        cpgs = df['CpG'].head(50).tolist()
    else:
        cpgs = df.iloc[:,0].head(50).tolist()

    G = nx.Graph()
    G.add_nodes_from(cpgs)
    # Dummy chain edges:
    for i in range(len(cpgs)-1):
        G.add_edge(cpgs[i], cpgs[i+1])

    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", font_size=8)
    plt.title("Network Diagram (Top 50 DMPs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] network_diagram => {out_path}")

# ----------------------------
# Main function
# ----------------------------
def main():
    results_dir = "/Volumes/T9/EpiMECoV/visuals"
    os.makedirs(results_dir, exist_ok=True)

    data_csv = "/Volumes/T9/EpiMECoV/processed_data/transformed_data.csv"
    if not os.path.exists(data_csv):
        print("[WARN] No transformed_data.csv. Falling back to cleaned_data.csv.")
        data_csv = "/Volumes/T9/EpiMECoV/processed_data/cleaned_data.csv"
    if not os.path.exists(data_csv):
        print("[WARN] No cleaned_data.csv. Exiting visualization early.")
        return

    df = pd.read_csv(data_csv)
    if "Condition" not in df.columns:
        print("[ERROR] Condition missing in data. Exiting.")
        return

    # Example fix: Merge any 'Noel ME' labels into 'ME'
    print("Unique conditions BEFORE fix:", df['Condition'].unique())
    df.loc[df['Condition'] == "Noel ME", "Condition"] = "ME"
    print("Unique conditions AFTER fix:", df['Condition'].unique())

    feat_cols = [c for c in df.columns if c != "Condition"]

    # If extremely high dimensional, do a quick PCA -> 64D
    if len(feat_cols) > 1000:
        print("[INFO] High-dimensional data detected => applying PCA to reduce to 64 features.")
        X_orig = df[feat_cols].values
        pca_reducer = PCA(n_components=64, random_state=42)
        X_reduced = pca_reducer.fit_transform(X_orig)
        latent_cols = [f"AE_{i}" for i in range(64)]
        df_new = pd.DataFrame(X_reduced, columns=latent_cols)
        df_new["Condition"] = df["Condition"].values
        df = df_new
        feat_cols = latent_cols

    # Impute + scale
    Xs = impute_and_scale(df, feat_cols)
    if Xs is None or Xs.shape[1] == 0:
        print("[WARN] impute_and_scale failed; skipping visualizations that require numeric matrix.")
        return

    y = df["Condition"].values
    imputed_df = pd.DataFrame(Xs, columns=[f"Imputed_{i}" for i in range(Xs.shape[1])])
    imputed_df["Condition"] = y

    # We'll do a small random forest to see feature importances for demonstration
    # (In real usage, you might skip this if you don't want a quick internal model.)
    from sklearn.ensemble import RandomForestClassifier
    top_rf_cols = [c for c in feat_cols[:100] if c in df.columns]
    if len(top_rf_cols) > 0:
        X_for_rf = df[top_rf_cols].values
        y_rf = np.array([str(x) for x in df["Condition"].values])
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_for_rf, y_rf)
        preds_rf = clf.predict(X_for_rf)
        cm_rf = confusion_matrix(y_rf, preds_rf)
    else:
        X_for_rf = None
        cm_rf = None

    # DMP CSVs if they exist
    dmp_files = {
        "ME_vs_Control": os.path.join("/Volumes/T9/EpiMECoV/results", "DMP_ME_vs_Control.csv"),
        "LC_vs_Control": os.path.join("/Volumes/T9/EpiMECoV/results", "DMP_LC_vs_Control.csv"),
        "ME_vs_LC":      os.path.join("/Volumes/T9/EpiMECoV/results", "DMP_ME_vs_LC.csv")
    }

    # Generate visuals
    safe_call(plot_tsne, Xs, y, os.path.join(results_dir, "tsne_plot.png"))
    safe_call(plot_pca, Xs, y, os.path.join(results_dir, "pca_plot.png"))
    safe_call(plot_age_distribution, df, os.path.join(results_dir, "age_distribution.png"))
    safe_call(plot_sex_distribution, df, os.path.join(results_dir, "sex_distribution.png"))
    safe_call(plot_scatter_box, df,
              os.path.join(results_dir, "scatter_cpg.png"),
              os.path.join(results_dir, "box_cpg.png"))
    safe_call(plot_volcano,
              os.path.join("/Volumes/T9/EpiMECoV/results", "DMP_ME_vs_Control.csv"),
              os.path.join(results_dir, "volcano_plot.png"))
    safe_call(plot_upset, dmp_files, os.path.join(results_dir, "intersection_plot.png"))
    safe_call(plot_network,
              os.path.join("/Volumes/T9/EpiMECoV/results", "DMP_ME_vs_Control.csv"),
              os.path.join(results_dir, "network_diagram.png"))
    safe_call(plot_pca_scree, Xs, os.path.join(results_dir, "pca_scree_plot.png"))
    safe_call(plot_pca_biplot, Xs, y, os.path.join(results_dir, "pca_biplot.png"))
    safe_call(plot_tsne_with_perplexity, Xs, y, 10,
              os.path.join(results_dir, "tsne_plot_p10.png"))
    safe_call(plot_tsne_with_perplexity, Xs, y, 20,
              os.path.join(results_dir, "tsne_plot_p20.png"))
    safe_call(plot_umap_visualization, Xs, y,
              os.path.join(results_dir, "umap_plot.png"))
    safe_call(plot_correlation_heatmap, Xs,
              os.path.join(results_dir, "correlation_heatmap.png"))
    safe_call(plot_hierarchical_dendrogram, Xs,
              os.path.join(results_dir, "hierarchical_dendrogram.png"))

    # Box & violin for the first feature, if any
    first_feat = feat_cols[0] if len(feat_cols) > 0 else None
    if first_feat is not None:
        safe_call(plot_box_feature, df, first_feat,
                  os.path.join(results_dir, "boxplot_feature.png"))
        safe_call(plot_violin_feature, df, first_feat,
                  os.path.join(results_dir, "violin_feature.png"))

    # Pairplot & KDE for up to first 5 features
    n_plot_feats = min(5, len(feat_cols))
    subset_feats = [c for c in feat_cols[:n_plot_feats] if c in df.columns]
    if len(subset_feats) > 0 and "Condition" in df.columns:
        safe_call(plot_pairplot, df, subset_feats,
                  os.path.join(results_dir, "pairplot.png"))
        safe_call(plot_kde_feature, df, subset_feats[0],
                  os.path.join(results_dir, "kde_plot.png"))

    # Condition count bar
    safe_call(plot_condition_count, df,
              os.path.join(results_dir, "condition_count.png"))

    # If we have a small random forest trained above, show feature importances & confusion
    if X_for_rf is not None and cm_rf is not None:
        safe_call(plot_rf_feature_importance, top_rf_cols, clf.feature_importances_, 20,
                  os.path.join(results_dir, "rf_feature_importances.png"))
        safe_call(plot_classifier_confusion_matrix, cm_rf, np.unique(df["Condition"].values),
                  os.path.join(results_dir, "confusion_matrix.png"))

    # Sankey example
    safe_call(plot_sankey_diagram,
              os.path.join(results_dir, "sankey_diagram.png"))

    # 3D PCA / t-SNE
    safe_call(plot_pca_3d, Xs, y,
              os.path.join(results_dir, "pca_3d_plot.png"))
    safe_call(plot_tsne_3d, Xs, y,
              os.path.join(results_dir, "tsne_3d_plot.png"))

    # Multi-panel summary
    safe_call(plot_multi_panel_summary, imputed_df,
              os.path.join(results_dir, "multi_panel_summary.png"))

    # Optionally do a smaller pairplot subset for feats [5:10] if we have >10 feats
    if len(feat_cols) > 10:
        subset2 = [c for c in feat_cols[5:10] if c in df.columns]
        if len(subset2) > 0:
            safe_call(plot_pairplot, df, subset2,
                      os.path.join(results_dir, "pairplot_subset.png"))

    print("=== All visualization steps completed. ===")


if __name__ == "__main__":
    main()