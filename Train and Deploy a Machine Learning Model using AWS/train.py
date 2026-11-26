"""
SageMaker Script Mode Training Script for Breast Cancer Wisconsin Dataset
Algorithm: Random Forest Classifier (Scikit-Learn)
"""

import argparse
import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Random Forest Training Script")

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum depth of the trees")

    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "."))

    return parser.parse_args()


def load_dataset(channel_dir, filename="train.csv"):
    file_path = os.path.join(channel_dir, filename)
    if not os.path.exists(file_path):
        # Look for any csv file in channel directory
        csv_files = [f for f in os.listdir(channel_dir) if f.endswith(".csv")]
        if csv_files:
            file_path = os.path.join(channel_dir, csv_files[0])
        else:
            raise FileNotFoundError(f"No CSV file found in channel directory: {channel_dir}")

    df = pd.read_csv(file_path)
    if "target" in df.columns:
        X = df.drop(columns=["target"])
        y = df["target"]
    else:
        # Fallback: assume first column is target if unnamed, or last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    return X, y


def train():
    args = parse_args()
    print("=" * 60)
    print("Starting SageMaker Random Forest Training")
    print(f"Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Channels: Train={args.train}, Test={args.test}")
    print(f"Model Output Dir: {args.model_dir}")
    print("=" * 60)

    # 1. Load train and test data
    X_train, y_train = load_dataset(args.train, "train.csv")
    X_test, y_test = load_dataset(args.test, "test.csv")
    print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"Testing samples: {len(X_test)}, Features: {X_test.shape[1]}")
    print(f"Train class distribution: Malignant(0)={sum(y_train == 0)}, Benign(1)={sum(y_train == 1)}")
    print(f"Test class distribution:  Malignant(0)={sum(y_test == 0)}, Benign(1)={sum(y_test == 1)}")

    # 2. Fit Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    print("Model training completed successfully.")

    # 3. Evaluate on test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Confusion Matrix:
    # 0 = Malignant (Negative class in sklearn binary convention), 1 = Benign (Positive class)
    # tn: True Malignant, fp: False Benign (Critical medical error!)
    # fn: False Malignant, tp: True Benign
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    malignant_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 60)
    print("EVALUATION METRICS:")
    print(f"test_accuracy={acc:.4f}")
    print(f"test_precision={prec:.4f}")
    print(f"test_recall={rec:.4f}")
    print(f"test_f1={f1:.4f}")
    print(f"test_auc={auc_score:.4f}")
    print(f"test_tn={tn}")
    print(f"test_fp={fp}")
    print(f"test_fn={fn}")
    print(f"test_tp={tp}")
    print(f"test_malignant_recall={malignant_recall:.4f}")
    print("=" * 60)

    # 4. Save model artifact
    os.makedirs(args.model_dir, exist_ok=True)
    model_output_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(clf, model_output_path)
    print(f"Model saved to: {model_output_path}")


# ==========================================
# SageMaker Hosting Inference Handlers
# ==========================================

def model_fn(model_dir):
    """Load model from model_dir for hosting/inference."""
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Parse input payload into numpy array or dataframe."""
    print(f"Received request with ContentType: {request_content_type}")
    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df.to_numpy()
    elif request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "features" in data:
            return np.array([data["features"]])
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("Unsupported JSON payload format. Expected {'features': [...]} or nested array.")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate predictions using the loaded model."""
    predictions = model.predict(input_data)
    return predictions


def output_fn(prediction, accept):
    """Serialize model predictions into response format."""
    print(f"Serializing response for Accept: {accept}")
    if accept in ["application/json", "*/*"]:
        return json.dumps(prediction.tolist()), "application/json"
    elif accept == "text/csv":
        output = io.StringIO()
        pd.DataFrame(prediction).to_csv(output, header=False, index=False)
        return output.getvalue(), "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


if __name__ == "__main__":
    train()
