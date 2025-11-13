#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Heart Disease Modelling — MLflow Project (CI Version)
Author  : Yoga Fatiqurrahman
Level   : Skilled/Advanced (Dicoding)
Notes   :
    - Fully compatible with MLflow Project + GitHub Actions CI
    - NO nested mlflow.start_run() → solves CI error
    - Automatically logs metrics, params, model, confusion matrix
    - Accepts dataset folder via --data_dir
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


def load_dataset(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    val_path   = os.path.join(data_dir, "val.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"[ERROR] train.csv tidak ditemukan di {data_dir}")

    return (
        pd.read_csv(train_path),
        pd.read_csv(val_path),
        pd.read_csv(test_path)
    )


def preprocess(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    return X, y, imputer, scaler


def preprocess_with_existing(df, imputer, scaler):
    X = df.drop(columns=["target"])
    y = df["target"]
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


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


def main(args):

    # MLflow project memiliki run otomatis → tidak pakai start_run()
    mlflow.set_experiment("Heart Disease — CI Training")

    # Ambil run aktif
    run = mlflow.active_run()
    print(f"[INFO] Active MLflow Run: {run.info.run_id}")

    # Load dataset
    train_df, val_df, test_df = load_dataset(args.data_dir)

    # Preprocess
    X_train, y_train, imputer, scaler = preprocess(train_df)
    X_val, y_val = preprocess_with_existing(val_df, imputer, scaler)
    X_test, y_test = preprocess_with_existing(test_df, imputer, scaler)

    # Params
    mlflow.log_param("dataset_dir", args.data_dir)

    # Training
    model = train_model(X_train, y_train)

    # Evaluation
    val_metrics, val_cm = evaluate(model, X_val, y_val, "val")
    test_metrics, test_cm = evaluate(model, X_test, y_test, "test")

    mlflow.log_metrics(val_metrics)
    mlflow.log_metrics(test_metrics)

    mlflow.log_dict({"validation_cm": val_cm.tolist()}, "validation_cm.json")
    mlflow.log_dict({"test_cm": test_cm.tolist()}, "test_cm.json")

    # Save model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="HeartDiseaseModel_CI"
    )

    # Save preprocessing objects
    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("imputer.pkl")
    mlflow.log_artifact("scaler.pkl")

    print("\n=== Training selesai tanpa error ===")
    print("VAL METRICS:", val_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder dataset preprocessing (train.csv, val.csv, test.csv)"
    )
    args = parser.parse_args()
    main(args)