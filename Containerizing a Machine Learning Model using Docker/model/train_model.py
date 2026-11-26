import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def train_and_evaluate():
    # Define paths relative to the project root
    base_dir = Path(__file__).resolve().parent.parent
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    model_path = base_dir / "model.joblib"
    eval_file_path = outputs_dir / "model_evaluation.txt"

    print("==================================================")
    print("      MLOps Assignment: Iris Model Training      ")
    print("==================================================")

    # 1. Load the Iris dataset
    print("[1/5] Loading Iris dataset from sklearn.datasets...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"      Features: {feature_names}")
    print(f"      Target Classes: {list(target_names)}")
    print(f"      Dataset Shape: X={X.shape}, y={y.shape}")

    # 2. Split dataset into train and test sets (test_size=0.2, random_state=42)
    print("[2/5] Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 3. Train RandomForestClassifier (n_estimators=100, random_state=42)
    print("[3/5] Training RandomForestClassifier (n_estimators=100, random_state=42)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("      Model training completed successfully.")

    # 4. Evaluate the model
    print("[4/5] Evaluating model performance on test set...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:\n" + report)
    print("Confusion Matrix:\n" + str(matrix))

    # Save evaluation metrics to outputs/model_evaluation.txt
    eval_content = f"""MLOps Assignment - Model Evaluation Report
=============================================
Course: MLOps (Machine Learning Operations)
Unit: III - Model Deployment and Serving
Model: RandomForestClassifier(n_estimators=100, random_state=42)
Dataset: Iris Plants Database

Dataset Split:
- Total Samples: {len(X)}
- Training Samples: {len(X_train)}
- Testing Samples: {len(X_test)}
- Test Size: 0.20
- Random State: 42

Evaluation Results:
-------------------
Accuracy Score: {accuracy:.4f} ({accuracy * 100:.2f}%)

Classification Report:
{report}

Confusion Matrix:
{matrix}

Target Classes:
0: {target_names[0]}
1: {target_names[1]}
2: {target_names[2]}
=============================================
"""
    with open(eval_file_path, "w", encoding="utf-8") as f:
        f.write(eval_content)
    print(f"\n[INFO] Evaluation results saved to: {eval_file_path}")

    # 5. Serialize the trained model using Joblib
    print("[5/5] Serializing model using Joblib...")
    joblib.dump(clf, model_path)
    print(f"[SUCCESS] Trained model saved at project root: {model_path}")
    print("==================================================")

    return accuracy

if __name__ == "__main__":
    train_and_evaluate()
