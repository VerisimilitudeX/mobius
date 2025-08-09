#!/usr/bin/env python3
"""
07_ensemble_ml.py

Manual Hard-Vote Ensemble:

Steps:
  1) Loads 'transformed_data.csv' (or whichever CSV from the autoencoder).
  2) Splits into train/test.
  3) Trains XGB, LGBM, CatBoost individually.
  4) Hard-votes to get final predictions.
  5) Prints confusion matrix & classification report.

Optional: Ray Tune for XGB hyperparam search if --tune is used.
"""

import argparse
import os

import numpy as np
import pandas as pd
import ray
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from ray import tune
from ray.air import session
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def xgb_tune(config, X_train, y_train, X_val, y_val):
    """Helper for Ray Tune to evaluate XGB hyperparams."""
    model = XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        learning_rate=config["learning_rate"],
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds_val = model.predict(X_val)
    acc = accuracy_score(y_val, preds_val)
    session.report({"accuracy": acc})


def tune_xgb(X_train, y_train, X_test, y_test):
    """Runs random search via Ray Tune for XGB, then returns best model."""
    X_sub, X_val, y_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=999, stratify=y_train
    )
    space = {
        "n_estimators": tune.randint(50, 300),
        "learning_rate": tune.loguniform(1e-3, 1e-1),
    }
    tuner = tune.run(
        tune.with_parameters(
            xgb_tune, X_train=X_sub, y_train=y_sub, X_val=X_val, y_val=y_val
        ),
        metric="accuracy",
        mode="max",
        num_samples=8,
        verbose=1,
    )
    best_trial = tuner.get_best_trial("accuracy", mode="max")
    best_cfg = best_trial.config
    print("[HYPERPARAM] Best config:", best_cfg)
    final_xgb = XGBClassifier(
        n_estimators=int(best_cfg["n_estimators"]),
        learning_rate=best_cfg["learning_rate"],
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
    )
    final_xgb.fit(X_train, y_train)
    preds = final_xgb.predict(X_test)
    acc_test = accuracy_score(y_test, preds)
    print(f"[HYPERPARAM] Final XGB => test acc: {acc_test:.4f}")
    return final_xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="/Volumes/T9/EpiMECoV/processed_data/transformed_data.csv",
        help="Input CSV with features + Condition.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Tune XGB hyperparams with Ray.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"No CSV => {args.csv}")

    df = pd.read_csv(args.csv)
    if "Condition" not in df.columns:
        raise ValueError("Need Condition col in CSV.")

    # X,y
    feat_cols = [c for c in df.columns if c != "Condition"]
    X = df[feat_cols].values
    conds = df["Condition"].unique()
    cond_map = {c: i for i, c in enumerate(conds)}
    y = np.array([cond_map[v] for v in df["Condition"].values])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # XGB
    if args.tune:
        model_xgb = tune_xgb(X_train, y_train, X_test, y_test)
    else:
        model_xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
        model_xgb.fit(X_train, y_train)

    # LightGBM
    model_lgb = LGBMClassifier(random_state=42)
    model_lgb.fit(X_train, y_train)

    # CatBoost
    model_cat = CatBoostClassifier(verbose=0, random_state=42)
    model_cat.fit(X_train, y_train)

    preds_xgb = model_xgb.predict(X_test)
    preds_lgb = model_lgb.predict(X_test)
    preds_cat = model_cat.predict(X_test)

    # Hard-vote ensemble
    ensemble_preds = []
    for i in range(len(y_test)):
        votes = [preds_xgb[i], preds_lgb[i], preds_cat[i]]
        voted = np.bincount(votes).argmax()
        ensemble_preds.append(voted)
    ensemble_preds = np.array(ensemble_preds, dtype=int)

    # Evaluate
    cm = confusion_matrix(y_test, ensemble_preds)
    acc = accuracy_score(y_test, ensemble_preds)
    print("\nEnsemble Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    inv_map = {v:k for k,v in cond_map.items()}
    label_names = [inv_map[i] for i in sorted(inv_map.keys())]
    print(classification_report(y_test, ensemble_preds, target_names=label_names))
    print("Ensemble Accuracy:", acc)


if __name__ == "__main__":
    main()