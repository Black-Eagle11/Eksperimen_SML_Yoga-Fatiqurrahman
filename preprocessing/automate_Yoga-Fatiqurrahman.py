import os
import pandas as pd
import numpy as np
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import time

print("============================================================")
print("Memuat Dataset Mentah")
print("============================================================")

CONFIG = {
    "RAW_DIR": "namadataset_raw",
    "OUT_DIR": "preprocessing/namadataset_preprocessing",
    "REPORT_DIR": "preprocessing/reports",
    "TEST_SIZE": 0.2,
    "SEED": 42
}

os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
os.makedirs(CONFIG["REPORT_DIR"], exist_ok=True)

# otomatis cari file CSV di folder raw
csv_files = [f for f in os.listdir(CONFIG["RAW_DIR"]) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError("Tidak ada file CSV ditemukan di folder 'namadataset_raw'.")
raw_path = os.path.join(CONFIG["RAW_DIR"], csv_files[0])

print(f"✅ File dataset ditemukan: {raw_path}")

df = pd.read_csv(raw_path)
print(f"Jumlah data awal: {df.shape[0]} baris, {df.shape[1]} kolom")

# --- Data Preprocessing ---
print("\n============================================================")
print("Membersihkan Data")
print("============================================================")

dups = df.duplicated().sum()
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"Duplikat dihapus: {dups} baris")
else:
    print("Tidak ada duplikat")

missing = df.isnull().sum().sum()
if missing > 0:
    df.fillna(df.median(numeric_only=True), inplace=True)
    print(f"Nilai kosong ditangani: {missing}")
else:
    print("Tidak ada nilai kosong")

df = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
scaled = scaler.fit_transform(df.values)
df_scaled = pd.DataFrame(scaled, columns=df.columns)

train_df, test_df = train_test_split(df_scaled, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["SEED"])

train_path = os.path.join(CONFIG["OUT_DIR"], "train.csv")
test_path = os.path.join(CONFIG["OUT_DIR"], "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Data latih tersimpan: {train_path}")
print(f"✅ Data uji tersimpan: {test_path}")

# --- Simpan laporan ---
report = {
    "rows_total": int(len(df)),
    "cols_total": int(df.shape[1]),
    "duplicates_removed": int(dups),
    "missing_filled": int(missing),
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "timestamp": time.ctime()
}

report_path = os.path.join(CONFIG["REPORT_DIR"], "preprocessing_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"📄 Laporan preprocessing disimpan: {report_path}")

# --- Logging MLflow ---
mlflow.set_tracking_uri("file:///content/mlruns")
mlflow.set_experiment("Heart Disease — Automate Preprocessing")

with mlflow.start_run(run_name="Yoga_Fatiqurrahman_Preprocessing"):
    mlflow.log_param("rows_total", len(df))
    mlflow.log_param("cols_total", df.shape[1])
    mlflow.log_param("duplicates_removed", int(dups))
    mlflow.log_param("missing_filled", int(missing))
    mlflow.log_param("train_rows", len(train_df))
    mlflow.log_param("test_rows", len(test_df))
    mlflow.log_artifact(train_path)
    mlflow.log_artifact(test_path)
    mlflow.log_artifact(report_path)
print("\n============================================================")
print("✅ Proses preprocessing selesai tanpa error.")
print("Dataset siap digunakan untuk modelling.")
print("============================================================")