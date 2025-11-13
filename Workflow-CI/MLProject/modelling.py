#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier


def load_dataset(data_dir: str):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, val_df, test_df


def preprocess(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    return X, y, imputer, scaler


def preprocess_apply(df, imputer, scaler):
    X = df.drop(columns=["target"])
    y = df["target"]
    X = imputer.transform(X)
    X = scaler.transform(X)
    return X, y


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y)
    return model


def evaluate(model, X, y, prefix):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, preds),
        f"{prefix}_precision": precision_score(y, preds),
        f"{prefix}_recall": recall_score(y, preds),
        f"{prefix}_f1": f1_score(y, preds),
        f"{prefix}_roc_auc": roc_auc_score(y, probs)
    }, confusion_matrix(y, preds)


def main(args):
    mlflow.set_experiment("Heart Disease — CI Training")

    # ===========================
    # FIX UTAMA PALING PENTING
    # ===========================
    run = mlflow.active_run()
    if run:
        print(f"[INFO] Active MLflow Run from MLflow Project: {run.info.run_id}")
    else:
        print("[WARN] Tidak ada active_run dari MLflow Project. Logging tetap dilanjutkan tanpa start_run().")

    # MULAI TRAINING
    train_df, val_df, test_df = load_dataset(args.data_dir)

    X_train, y_train, imputer, scaler = preprocess(train_df)
    X_val, y_val = preprocess_apply(val_df, imputer, scaler)
    X_test, y_test = preprocess_apply(test_df, imputer, scaler)

    mlflow.log_param("dataset_dir", args.data_dir)

    model = train_model(X_train, y_train)

    val_metrics, val_cm = evaluate(model, X_val, y_val, "val")
    test_metrics, test_cm = evaluate(model, X_test, y_test, "test")

    mlflow.log_metrics(val_metrics)
    mlflow.log_metrics(test_metrics)

    mlflow.log_dict({"val_cm": val_cm.tolist()}, "val_cm.json")
    mlflow.log_dict({"test_cm": test_cm.tolist()}, "test_cm.json")

    mlflow.sklearn.log_model(model, artifact_path="model")

    joblib.dump(imputer, "imputer.pkl")
    joblib.dump(scaler, "scaler.pkl")

    mlflow.log_artifact("imputer.pkl")
    mlflow.log_artifact("scaler.pkl")

    print("\n=== CI TRAINING SUCCESS ===")
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
