#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heart Disease Modelling — MLflow Project (Ultimate CI Version)
Author  : Yoga Fatiqurrahman
Notes   :
    - Fully MLflow Project Compatible
    - Auto-detect target column (condition/target/label/result/output)
    - Auto logging: params, metrics, confusion matrix, model
    - Production-grade, CI-safe, no active run conflict
"""

import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier


# ============================================================
#  AUTO-DETECT TARGET COLUMN (ANTI ERROR)
# ============================================================
def detect_target_column(df: pd.DataFrame):
    possible = ["target", "condition", "label", "output", "result", "y"]
    for col in df.columns:
        if col.lower() in [x.lower() for x in possible]:
            return col
    raise ValueError(
        f"[ERROR] Tidak ditemukan kolom target. Kolom tersedia: {list(df.columns)}"
    )


# ============================================================
#  LOAD DATASET
# ============================================================
def load_dataset(data_dir: str):
    paths = {
        "train": os.path.join(data_dir, "train.csv"),
        "val": os.path.join(data_dir, "val.csv"),
        "test": os.path.join(data_dir, "test.csv"),
    }

    for name, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] File {name}.csv tidak ditemukan di {data_dir}")

    return (
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["val"]),
        pd.read_csv(paths["test"]),
    )


# ============================================================
#  PREPROCESSING (TRAIN)
# ============================================================
def preprocess_train(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    return X, y, imputer, scaler


# ============================================================
#  PREPROCESSING (VAL/TEST)
# ============================================================
def preprocess_apply(df: pd.DataFrame, target_col: str, imputer, scaler):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = imputer.transform(X)
    X = scaler.transform(X)

    return X, y


# ============================================================
#  MODEL TRAINING
# ============================================================
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
#  EVALUATION
# ============================================================
def evaluate(model, X, y, prefix):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, probs),
    }

    return metrics, confusion_matrix(y, preds)


# ============================================================
#  MAIN PIPELINE (CI SAFE)
# ============================================================
def main(args):

    mlflow.set_experiment("Heart Disease — CI Training")

    # ALWAYS SAFE: MLflow creates new run cleanly
    with mlflow.start_run(run_name="CI_Retrain_Model"):

        # Load dataset
        train_df, val_df, test_df = load_dataset(args.data_dir)

        # Auto-detect target
        target_col = detect_target_column(train_df)
        mlflow.log_param("target_column", target_col)

        # Preprocessing
        X_train, y_train, imputer, scaler = preprocess_train(train_df, target_col)
        X_val, y_val = preprocess_apply(val_df, target_col, imputer, scaler)
        X_test, y_test = preprocess_apply(test_df, target_col, imputer, scaler)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate
        val_metrics, val_cm = evaluate(model, X_val, y_val, "val")
        test_metrics, test_cm = evaluate(model, X_test, y_test, "test")

        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)

        mlflow.log_dict({"val_confusion_matrix": val_cm.tolist()}, "val_cm.json")
        mlflow.log_dict({"test_confusion_matrix": test_cm.tolist()}, "test_cm.json")

        # Save transformers
        joblib.dump(imputer, "imputer.pkl")
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("imputer.pkl")
        mlflow.log_artifact("scaler.pkl")

        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HeartDisease_CI_Model"
        )

        print("\n=== TRAINING SELESAI TANPA ERROR ===")
        print("Validation Metrics:", val_metrics)
        print("Test Metrics:", test_metrics)


# ============================================================
#  CLI ENTRY
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder dataset hasil preprocessing (train.csv/val.csv/test.csv)"
    )

    args = parser.parse_args()
    main(args)