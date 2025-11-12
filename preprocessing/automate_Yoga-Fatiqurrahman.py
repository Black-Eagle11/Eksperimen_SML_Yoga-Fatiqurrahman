#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated preprocessing — Heart Disease (tabular, binary)
Author  : Yoga Fatiqurrahman
License : MIT
Notes   : Cross-platform (Windows/Linux/Colab). Compatible with Dicoding SMSML (Kriteria 1 Skilled/Advanced).
"""

import os
import sys
import json
import time
import argparse
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def infer_target_col(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "condition" in df.columns:
        return "condition"
    raise ValueError("Kolom target tidak ditemukan. Wajib ada salah satu: 'target' atau 'condition'.")


def split_60_20_20(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split 60/20/20 dengan stratify."""
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, "Fraksi split tidak valid."
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_frac + test_frac, random_state=seed, stratify=y
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, random_state=seed, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    report_dir: Path
    seed: int = 42
    val_size: float = 0.20
    test_size: float = 0.20


# ---------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------
def run(cfg: Config, tracking_uri: str | None = None) -> None:
    t0 = time.time()
    setup_logger()
    set_global_seed(cfg.seed)

    # Ensure dirs
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    logging.info("============================================================")
    logging.info("Memuat Dataset Mentah")
    logging.info("============================================================")

    if not cfg.raw_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {cfg.raw_dir}")

    csv_files = sorted([p for p in cfg.raw_dir.iterdir() if p.suffix.lower() == ".csv"])
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file .csv di: {cfg.raw_dir}")

    raw_path = csv_files[0]  # ambil yang pertama
    logging.info(f"✅ File dataset ditemukan : {raw_path}")
    logging.info(f"🔑 SHA256                 : {sha256_of_file(raw_path)}")

    # Load
    df = pd.read_csv(raw_path)
    n0 = len(df)
    logging.info(f"Jumlah data awal         : {df.shape[0]} baris × {df.shape[1]} kolom")

    # Basic cleaning
    logging.info("Membersihkan Data")
    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates()
    logging.info(f"Duplikat dihapus         : {dups}")

    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_all = [c for c in df.columns if c not in num_cols_all]
        if num_cols_all:
            # median untuk numerik (robust)
            df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median(numeric_only=True))
        for c in cat_cols_all:
            # mode untuk kategorikal
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        logging.info(f"Nilai kosong ditangani   : {missing_total}")
    else:
        logging.info("Tidak ada nilai kosong")

    # Target
    target_col = infer_target_col(df)
    y = df[target_col].astype(int).to_numpy()
    X = df.drop(columns=[target_col]).copy()

    # Pastikan hanya numerik yang dipakai (Heart UCI tabular)
    feat_cols: List[str] = X.select_dtypes(include=np.number).columns.tolist()
    X = X[feat_cols].copy()

    # Distribusi kelas (sebelum split)
    class_counts = pd.Series(y).value_counts().to_dict()

    # Split 60/20/20 (stratify)
    X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(
        X, y, seed=cfg.seed, val_frac=cfg.val_size, test_frac=cfg.test_size
    )

    # Transform
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    X_train_sc = scaler.fit_transform(X_train_imp)
    X_val_sc = scaler.transform(X_val_imp)
    X_test_sc = scaler.transform(X_test_imp)

    # Kembali ke DataFrame dengan kolom konsisten
    train_df = pd.DataFrame(X_train_sc, columns=feat_cols)
    val_df = pd.DataFrame(X_val_sc, columns=feat_cols)
    test_df = pd.DataFrame(X_test_sc, columns=feat_cols)
    train_df[target_col] = y_train
    val_df[target_col] = y_val
    test_df[target_col] = y_test

    # Output paths
    train_path = cfg.out_dir / "train.csv"
    val_path = cfg.out_dir / "val.csv"
    test_path = cfg.out_dir / "test.csv"
    meta_path = cfg.out_dir / "meta.json"
    scaler_path = cfg.out_dir / "scaler.pkl"
    imputer_path = cfg.out_dir / "imputer.pkl"
    schema_path = cfg.out_dir / "schema.json"
    feature_names_path = cfg.out_dir / "feature_names.json"

    # Save artifacts
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)

    # Schema & feature names
    schema = {c: str(train_df[c].dtype) for c in train_df.columns}
    json.dump(schema, open(schema_path, "w"), indent=2)
    json.dump({"feature_names_in_order": feat_cols}, open(feature_names_path, "w"), indent=2)

    logging.info(f"✅ Data latih tersimpan   : {train_path}")
    logging.info(f"✅ Data validasi tersimpan: {val_path}")
    logging.info(f"✅ Data uji tersimpan     : {test_path}")
    logging.info(f"✅ Simpan scaler/imputer  : {scaler_path} | {imputer_path}")

    # Report
    report = {
        "source_csv": str(raw_path).replace("\\", "/"),
        "source_sha256": sha256_of_file(raw_path),
        "rows_before": n0,
        "rows_after_dropdup": int(len(df)),
        "target_col": target_col,
        "class_distribution_before": class_counts,
        "split": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "feature_count": len(feat_cols),
        "feature_names": feat_cols,
        "seed": cfg.seed,
        "duplicates_removed": dups,
        "missing_filled": missing_total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "mlflow": mlflow.__version__,
        },
    }
    report_path = cfg.report_dir / "preprocessing_report.json"
    json.dump(report, open(report_path, "w"), indent=2)
    json.dump(report, open(meta_path, "w"), indent=2)  # meta yang ikut di folder dataset
    logging.info(f"📄 Laporan preprocessing  : {report_path}")

    # MLflow logging
    if tracking_uri is None:
        # mlruns sejajar dengan root repo (naik 1 level dari folder 'preprocessing')
        base_dir = Path(__file__).resolve().parents[1]
        mlruns_dir = (base_dir / "mlruns").resolve()
        mlruns_dir.mkdir(exist_ok=True)
        tracking_uri = mlruns_dir.as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Heart Disease — Automate Preprocessing")
    logging.info(f"🧭 MLflow Tracking URI   : {tracking_uri}")

    with mlflow.start_run(run_name="Yoga_Fatiqurrahman_Preprocessing"):
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "val_size": cfg.val_size,
                "test_size": cfg.test_size,
                "rows_before": n0,
                "rows_after_dropdup": len(df),
                "n_features": len(feat_cols),
                "duplicates_removed": dups,
                "missing_filled": missing_total,
                "target_col": target_col,
            }
        )
        # ringkas class distribution
        for cls, cnt in class_counts.items():
            mlflow.log_metric(f"class_count_before_{cls}", cnt)

        # log artifacts
        mlflow.log_artifacts(str(cfg.out_dir))
        mlflow.log_artifact(str(report_path))

        # durasi
        duration = round(time.time() - t0, 3)
        mlflow.log_metric("preprocessing_duration_sec", duration)
        logging.info(f"⏱️  Durasi preprocessing  : {duration:.3f} s")

    logging.info("============================================================")
    logging.info("✅ Proses preprocessing selesai tanpa error.")
    logging.info("Dataset siap digunakan untuk modelling & tuning.")
    logging.info("============================================================")


# ---------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]  # root repo (sejajar dgn 'preprocessing')
    parser = argparse.ArgumentParser(description="Automated preprocessing (60/20/20, scaling & imputation).")
    parser.add_argument("--raw_dir", type=str, default=str(base_dir / "namadataset_raw"))
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "preprocessing" / "namadataset_preprocessing"))
    parser.add_argument("--report_dir", type=str, default=str(base_dir / "preprocessing" / "reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--tracking_uri", type=str, default=None, help="MLflow tracking URI (opsional).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        report_dir=Path(args.report_dir),
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    run(cfg, tracking_uri=args.tracking_uri)