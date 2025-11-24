#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated preprocessing — Heart Disease (tabular, binary)
Author  : Yoga Fatiqurrahman
License : MIT
Notes   : Cross-platform (Windows/Linux/Colab).
          Compatible with Dicoding SMSML (Kriteria 1 Skilled/Advanced).
"""

import os
import sys
import json
import time
import argparse
import hashlib
import logging
import yaml
import getpass
import platform
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

import joblib
from joblib import Parallel, delayed

import mlflow


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
    import random

    np.random.seed(seed)
    random.seed(seed)


def infer_target_col(df: pd.DataFrame) -> str:
    if "target" in df.columns:
        return "target"
    if "condition" in df.columns:
        return "condition"
    raise ValueError(
        "Kolom target tidak ditemukan. Wajib ada salah satu: 'target' atau 'condition'."
    )


def validate_schema(df: pd.DataFrame, expected_min_cols: int = 2) -> None:
    logging.info("Memvalidasi skema dataset ...")

    if df.shape[1] < expected_min_cols:
        raise ValueError(
            f"Dataset hanya memiliki {df.shape[1]} kolom, minimal {expected_min_cols} dibutuhkan."
        )

    unsupported = [
        c
        for c in df.columns
        if not (
            is_numeric_dtype(df[c])
            or is_string_dtype(df[c])
            or is_categorical_dtype(df[c])
        )
    ]
    if unsupported:
        raise TypeError(f"Kolom dengan tipe tidak didukung: {unsupported}")

    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        miss_per_col = df.isnull().sum().astype(int).to_dict()
        logging.warning(f"Terdapat {total_missing} nilai kosong: {miss_per_col}")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols):
        high_outliers: Dict[str, int] = {}
        for col in num_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_limit = q3 + 3 * iqr
            lower_limit = q1 - 3 * iqr
            extreme = ((df[col] < lower_limit) | (df[col] > upper_limit)).sum()
            if int(extreme) > 0:
                high_outliers[col] = int(extreme)
        if high_outliers:
            logging.warning(f"Terdapat nilai ekstrem di kolom: {high_outliers}")

    type_summary = df.dtypes.astype(str).to_dict()
    logging.info(f"Tipe kolom: {json.dumps(type_summary, indent=2)}")
    logging.info("Skema dataset valid — lanjut ke tahap berikutnya.")


def split_60_20_20(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray
]:
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1, (
        "Fraksi split tidak valid."
    )
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=y,
    )
    rel_test = test_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=rel_test,
        random_state=seed,
        stratify=y_tmp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    method: str = "up",
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    logging.info("Memeriksa keseimbangan kelas ...")

    unique_classes = np.unique(y)
    if len(unique_classes) <= 1:
        logging.warning("Dataset hanya memiliki satu kelas, balancing dilewati.")
        return X, y

    df_all = X.copy()
    df_all["target"] = y

    dist_before = df_all["target"].value_counts(normalize=True).to_dict()
    imbalance_ratio = min(dist_before.values()) / max(dist_before.values())
    logging.info(
        f"Distribusi awal kelas: {dist_before} (rasio ≈ {imbalance_ratio:.3f})"
    )

    if imbalance_ratio >= 0.5:
        logging.info("Distribusi kelas sudah cukup seimbang, tidak perlu balancing.")
        return X, y

    maj_class = df_all["target"].value_counts().idxmax()
    min_class = df_all["target"].value_counts().idxmin()
    df_major = df_all[df_all["target"] == maj_class]
    df_minor = df_all[df_all["target"] == min_class]

    if method == "up":
        df_minor_resampled = resample(
            df_minor,
            replace=True,
            n_samples=len(df_major),
            random_state=seed,
        )
        df_balanced = pd.concat([df_major, df_minor_resampled], ignore_index=True)
        action = "Oversampling"
    else:
        df_major_resampled = resample(
            df_major,
            replace=False,
            n_samples=len(df_minor),
            random_state=seed,
        )
        df_balanced = pd.concat([df_major_resampled, df_minor], ignore_index=True)
        action = "Undersampling"

    df_balanced = df_balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    y_bal = df_balanced["target"].to_numpy()
    X_bal = df_balanced.drop(columns=["target"])

    dist_after = pd.Series(y_bal).value_counts(normalize=True).to_dict()
    ratio_after = min(dist_after.values()) / max(dist_after.values())
    logging.info(f"{action} selesai → distribusi baru: {dist_after}")

    try:
        mlflow.log_metric("class_balance_ratio_before", float(imbalance_ratio))
        mlflow.log_metric("class_balance_ratio_after", float(ratio_after))
    except Exception as e:
        logging.warning(f"Gagal mencatat metrik balancing ke MLflow: {e}")

    return X_bal, y_bal


def save_metadata_yaml(
    raw_path: Path,
    cfg: "Config",
    schema: Dict[str, str],
    feature_names: List[str],
    tracking_uri: str,
    report_dir: Path,
) -> Path:
    logging.info("Menyimpan metadata preprocessing ke metadata.yaml ...")

    metadata = {
        "dataset_info": {
            "source_file": str(raw_path.name),
            "absolute_path": str(raw_path.resolve()),
            "num_features": len(feature_names),
            "features": feature_names,
            "schema": schema,
        },
        "preprocessing_config": {
            "seed": cfg.seed,
            "val_size": cfg.val_size,
            "test_size": cfg.test_size,
        },
        "environment": {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "tracking_uri": tracking_uri,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    yaml_path = report_dir / "metadata.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)

    try:
        mlflow.log_artifact(str(yaml_path))
        logging.info("metadata.yaml berhasil disimpan dan di-log ke MLflow.")
    except Exception as e:
        logging.warning(f"Gagal log metadata.yaml ke MLflow: {e}")

    return yaml_path


def quick_profile(df: pd.DataFrame, out_path: Path, sample_limit: int = 5000) -> Path:
    logging.info("Membuat laporan profil dataset (quick profiling)...")

    profile: Dict[str, object] = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isnull().sum().astype(int).to_dict(),
    }

    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        stats = numeric_cols.describe().T.round(3).to_dict()
        profile["numeric_stats"] = stats

    sample_cols: Dict[str, List[str]] = {}
    for col in df.columns:
        uniques = df[col].dropna().unique()[:10]
        sample_cols[col] = [str(u) for u in uniques]
    profile["sample_values"] = sample_cols

    if len(df) > sample_limit:
        preview_df = df.sample(n=sample_limit, random_state=42)
    else:
        preview_df = df.copy()
    profile["preview"] = preview_df.head(5).to_dict(orient="records")

    json_path = out_path / "data_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    try:
        mlflow.log_artifact(str(json_path))
        mlflow.log_metric("n_rows", int(df.shape[0]))
        mlflow.log_metric("n_columns", int(df.shape[1]))
        mlflow.log_metric("missing_values_total", int(df.isnull().sum().sum()))
        logging.info("Profil dataset berhasil dibuat & di-log ke MLflow.")
    except Exception as e:
        logging.warning(f"Gagal log profil ke MLflow: {e}")

    return json_path


def parallel_fillna(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    logging.info("Memulai proses pengisian nilai kosong (parallel fillna)...")
    start_time = time.time()

    def fill_column(col_name: str) -> pd.Series:
        col = df[col_name]
        if col.isnull().any():
            if np.issubdtype(col.dtype, np.number):
                fill_value = col.median()
            else:
                mode_vals = col.mode()
                fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            return col.fillna(fill_value)
        return col

    missing_before = int(df.isnull().sum().sum())

    try:
        filled_cols = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fill_column)(col) for col in df.columns
        )
        df_filled = pd.concat(filled_cols, axis=1)
        missing_after = int(df_filled.isnull().sum().sum())
        missing_filled = missing_before - missing_after

        duration = round(time.time() - start_time, 3)
        logging.info("Proses parallel fillna selesai tanpa error.")
        logging.info(f"Durasi pengisian: {duration:.3f} detik")
        logging.info(f"Jumlah nilai kosong yang diisi: {missing_filled}")

        try:
            mlflow.log_metric("fillna_duration_sec", duration)
            mlflow.log_metric("missing_values_filled", missing_filled)
        except Exception as e:
            logging.warning(f"Gagal mencatat metrik fillna ke MLflow: {e}")

        return df_filled

    except Exception as e:
        logging.warning(
            f"Gagal melakukan parallel fillna, fallback ke mode normal: {e}"
        )
        num_cols_all = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols_all = [c for c in df.columns if c not in num_cols_all]

        if num_cols_all:
            df[num_cols_all] = df[num_cols_all].fillna(df[num_cols_all].median())
        for c in cat_cols_all:
            mode_vals = df[c].mode()
            if not mode_vals.empty:
                df[c] = df[c].fillna(mode_vals.iloc[0])
            else:
                df[c] = df[c].fillna("unknown")

        duration = round(time.time() - start_time, 3)
        missing_after = int(df.isnull().sum().sum())
        missing_filled = missing_before - missing_after
        logging.info(
            f"Fallback fillna selesai dalam {duration:.3f} detik (mode normal)."
        )

        try:
            mlflow.log_metric("fillna_duration_sec_fallback", duration)
            mlflow.log_metric("missing_values_filled_fallback", missing_filled)
        except Exception:
            pass

        return df


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    report_dir: Path
    seed: int = 42
    val_size: float = 0.20
    test_size: float = 0.20


def run(cfg: Config, tracking_uri: Optional[str] = None) -> None:
    t0 = time.time()
    setup_logger()
    set_global_seed(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    if tracking_uri is None:
        base_dir = Path(__file__).resolve().parents[1]
        mlruns_dir = (base_dir / "mlruns").resolve()
        mlruns_dir.mkdir(exist_ok=True)
        tracking_uri = mlruns_dir.as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Heart Disease — Automate Preprocessing")
    logging.info(f"MLflow Tracking URI : {tracking_uri}")

    try:
        if mlflow.active_run():
            logging.warning("Menutup MLflow run lama yang masih aktif ...")
            mlflow.end_run()
    except Exception as e:
        logging.warning(f"Melewati error saat menutup run lama: {e}")

    with mlflow.start_run(run_name="Yoga_Fatiqurrahman_Preprocessing"):
        mlflow.set_tag("author", "Yoga Fatiqurrahman")
        mlflow.set_tag("project", "Heart Disease — Automated Preprocessing")
        mlflow.log_param("git_commit", os.getenv("GITHUB_SHA", "local_run"))
        mlflow.log_param("run_env", platform.system())

        logging.info("Memuat dataset mentah ...")

        if not cfg.raw_dir.exists():
            raise FileNotFoundError(f"Folder tidak ditemukan: {cfg.raw_dir}")

        csv_files = sorted(
            [p for p in cfg.raw_dir.iterdir() if p.suffix.lower() == ".csv"]
        )
        if not csv_files:
            raise FileNotFoundError(f"Tidak ada file .csv di: {cfg.raw_dir}")

        raw_path = csv_files[0]
        logging.info(f"File dataset ditemukan : {raw_path}")
        raw_sha256 = sha256_of_file(raw_path)
        logging.info(f"SHA256                 : {raw_sha256}")

        df = pd.read_csv(raw_path)
        n0 = int(len(df))

        validate_schema(df)
        quick_profile(df, cfg.report_dir)

        logging.info(
            f"Jumlah data awal       : {df.shape[0]} baris × {df.shape[1]} kolom"
        )

        logging.info("Membersihkan data ...")
        dups = int(df.duplicated().sum())
        if dups:
            df = df.drop_duplicates().reset_index(drop=True)
        logging.info(f"Duplikat dihapus       : {dups}")

        missing_total = int(df.isnull().sum().sum())
        if missing_total > 0:
            df = parallel_fillna(df)
            logging.info(f"Nilai kosong ditangani : {missing_total}")
        else:
            logging.info("Tidak ada nilai kosong")

        target_col = infer_target_col(df)
        y = df[target_col].astype(int).to_numpy()
        X_full = df.drop(columns=[target_col]).copy()

        feat_cols: List[str] = X_full.select_dtypes(include=np.number).columns.tolist()
        if not feat_cols:
            raise ValueError("Tidak ada fitur numerik yang ditemukan untuk modelling.")

        X = X_full[feat_cols].copy()

        class_counts_before = pd.Series(y).value_counts().to_dict()
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "val_size": cfg.val_size,
                "test_size": cfg.test_size,
                "rows_before": n0,
                "rows_after_dropdup": int(len(df)),
                "n_features_numeric": len(feat_cols),
                "duplicates_removed": dups,
                "missing_filled": missing_total,
                "target_col": target_col,
            }
        )
        for cls, cnt in class_counts_before.items():
            mlflow.log_metric(f"class_count_before_{cls}", float(cnt))

        X_bal, y_bal = balance_dataset(X, y, method="up", seed=cfg.seed)
        class_counts_after = pd.Series(y_bal).value_counts().to_dict()
        for cls, cnt in class_counts_after.items():
            mlflow.log_metric(f"class_count_after_{cls}", float(cnt))

        X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(
            X_bal,
            y_bal,
            seed=cfg.seed,
            val_frac=cfg.val_size,
            test_frac=cfg.test_size,
        )

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        X_train_sc = scaler.fit_transform(X_train_imp)
        X_val_sc = scaler.transform(X_val_imp)
        X_test_sc = scaler.transform(X_test_imp)

        train_df = pd.DataFrame(X_train_sc, columns=feat_cols)
        val_df = pd.DataFrame(X_val_sc, columns=feat_cols)
        test_df = pd.DataFrame(X_test_sc, columns=feat_cols)

        train_df[target_col] = y_train
        val_df[target_col] = y_val
        test_df[target_col] = y_test

        train_path = cfg.out_dir / "train.csv"
        val_path = cfg.out_dir / "val.csv"
        test_path = cfg.out_dir / "test.csv"
        meta_path = cfg.out_dir / "meta.json"
        scaler_path = cfg.out_dir / "scaler.pkl"
        imputer_path = cfg.out_dir / "imputer.pkl"
        schema_path = cfg.out_dir / "schema.json"
        feature_names_path = cfg.out_dir / "feature_names.json"
        report_path = cfg.report_dir / "preprocessing_report.json"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)

        schema = {c: str(train_df[c].dtype) for c in train_df.columns}
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        with open(feature_names_path, "w", encoding="utf-8") as f:
            json.dump({"feature_names_in_order": feat_cols}, f, indent=2)

        logging.info(f"Data latih tersimpan   : {train_path}")
        logging.info(f"Data validasi tersimpan: {val_path}")
        logging.info(f"Data uji tersimpan     : {test_path}")
        logging.info(f"Simpan scaler/imputer  : {scaler_path} | {imputer_path}")

        report = {
            "source_csv": str(raw_path).replace("\\", "/"),
            "source_sha256": raw_sha256,
            "rows_before": n0,
            "rows_after_dropdup": int(len(df)),
            "target_col": target_col,
            "class_distribution_before": class_counts_before,
            "class_distribution_after": class_counts_after,
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

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Laporan preprocessing  : {report_path}")

        save_metadata_yaml(
            raw_path=raw_path,
            cfg=cfg,
            schema=schema,
            feature_names=feat_cols,
            tracking_uri=tracking_uri,
            report_dir=cfg.report_dir,
        )

        try:
            mlflow.log_artifacts(str(cfg.out_dir))
            mlflow.log_artifact(str(report_path))
            mlflow.log_artifact(str(schema_path))
            mlflow.log_artifact(str(feature_names_path))
        except Exception as e:
            logging.warning(f"Gagal log artefak utama ke MLflow: {e}")

        assert train_df.shape[0] > 0 and val_df.shape[0] > 0 and test_df.shape[0] > 0, (
            "Dataset split tidak valid!"
        )

        duration = round(time.time() - t0, 3)
        mlflow.log_metric("preprocessing_duration_sec", duration)
        logging.info(f"Durasi preprocessing   : {duration:.3f} s")
        logging.info(
            "Semua dataset hasil split valid dan siap digunakan untuk modelling."
        )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Automated preprocessing (60/20/20, imputation, scaling, balancing)."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=str(base_dir / "namadataset_raw"),
        help="Folder berisi file .csv mentah.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(base_dir / "preprocessing" / "namadataset_preprocessing"),
        help="Folder output untuk train/val/test dan artefak preprocessing.",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=str(base_dir / "preprocessing" / "reports"),
        help="Folder untuk laporan profil dan metadata.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI (opsional, default: mlruns lokal).",
    )
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
