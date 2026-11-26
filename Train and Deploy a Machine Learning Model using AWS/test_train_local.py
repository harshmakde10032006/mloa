"""
Local data preparation and test script for Breast Cancer dataset
Validates Step 3 & Step 4 logic locally before submitting to AWS SageMaker.
"""

import os
import subprocess
import sys
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 1. Load dataset
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer(as_frame=True)
df = data.frame

print(f"Dataset shape: {df.shape} (569 rows, 30 features + 1 target)")
print(f"Target counts: Malignant(0)={sum(df['target'] == 0)}, Benign(1)={sum(df['target'] == 1)}")

# 2. Stratified 80/20 split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["target"],
    random_state=42
)

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape:  {test_df.shape}")
print(f"Train Malignant(0): {sum(train_df['target'] == 0)}, Benign(1): {sum(train_df['target'] == 1)}")
print(f"Test Malignant(0):  {sum(test_df['target'] == 0)}, Benign(1):  {sum(test_df['target'] == 1)}")

# Save CSVs
train_csv_path = os.path.join(TRAIN_DIR, "train.csv")
test_csv_path = os.path.join(TEST_DIR, "test.csv")
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
print(f"Saved: {train_csv_path}")
print(f"Saved: {test_csv_path}")

# 3. Test train.py locally across 3 hyperparameter configurations
configs = [
    {"max_depth": 4, "n_estimators": 100, "name": "run-depth4-est100"},
    {"max_depth": 8, "n_estimators": 100, "name": "run-depth8-est100"},
    {"max_depth": 8, "n_estimators": 200, "name": "run-depth8-est200"},
]

for cfg in configs:
    model_dir = os.path.join(BASE_DIR, "models", cfg["name"])
    os.makedirs(model_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(BASE_DIR, "train.py"),
        f"--n-estimators={cfg['n_estimators']}",
        f"--max-depth={cfg['max_depth']}",
        f"--model-dir={model_dir}",
        f"--train={TRAIN_DIR}",
        f"--test={TEST_DIR}"
    ]
    print(f"\nRunning local check for: {cfg['name']}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
