#!/usr/bin/env python3
"""
random_forest_hyperparam_search.py

Purpose:
  1) Load your epigenetic CSV and parse Condition => y.
  2) Use Optuna for Bayesian hyperparameter optimization of RandomForestClassifier.
  3) Evaluate the best model on a hold-out test set.
  4) Save confusion matrix & top-20 feature importances.

Usage:
  python random_forest_hyperparam_search.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def objective(trial, X_train, y_train, X_val, y_val):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=42
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc

def plot_confusion_matrix(cm, class_names, filename="rf_confusion_matrix.png"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (XGB Hyperparam Search)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_feature_importances(feature_names, importances, top_n=20, filename="rf_feature_importances.png"):
    idxs = np.argsort(importances)[::-1]
    top_idx = idxs[:top_n]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[top_idx], y=[feature_names[i] for i in top_idx], orient="h", palette="viridis")
    plt.title("RF Feature Importances (Optuna Optimized)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    CSV_DATA = "transformed_data.csv"
    if not os.path.exists(CSV_DATA):
        raise FileNotFoundError(f"[ERROR] Could not find {CSV_DATA}")
    df = pd.read_csv(CSV_DATA)
    if "Condition" not in df.columns:
        raise ValueError("[ERROR] Condition column not found.")
    feature_names = [c for c in df.columns if c != "Condition"]
    X = df[feature_names].values
    y_str = df["Condition"].values
    classes = sorted(np.unique(y_str))
    class_map = {c: i for i, c in enumerate(classes)}
    y = np.array([class_map[v] for v in y_str], dtype=np.int64)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=20)
    print("Best Params:", study.best_trial.params)
    best_rf = RandomForestClassifier(
        n_estimators=int(study.best_trial.params["n_estimators"]),
        max_depth=study.best_trial.params["max_depth"],
        min_samples_split=study.best_trial.params["min_samples_split"],
        min_samples_leaf=study.best_trial.params["min_samples_leaf"],
        bootstrap=study.best_trial.params["bootstrap"],
        random_state=42
    )
    best_rf.fit(X_trainval, y_trainval)
    preds_test = best_rf.predict(X_test)
    acc = accuracy_score(y_test, preds_test)
    print(f"Test Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, preds_test)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, preds_test, target_names=classes))
    plot_confusion_matrix(cm, classes, filename="rf_confusion_matrix.png")
    # Feature importance using permutation importance could be used,
    # but here we use the built-in feature_importances_
    importances = best_rf.feature_importances_
    plot_feature_importances(feature_names, importances, filename="rf_feature_importances.png")
    print("\nDone with Random Forest hyperparam search. See:")
    print(" • rf_confusion_matrix.png")
    print(" • rf_feature_importances.png")

if __name__ == "__main__":
    main()