import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    os.makedirs("model", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    # 1. Load Iris dataset
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Standardize column names
    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width"
    }
    df = df.rename(columns=column_mapping)
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    X = df[feature_cols]
    y = df["target"]
    target_names = list(iris.target_names)

    # 2. Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Compute per-feature training stats (mean and std)
    training_stats = {
        "features": feature_cols,
        "target_names": target_names,
        "stats": {
            col: {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std(ddof=1))
            }
            for col in feature_cols
        }
    }

    # 4. Train RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred, target_names=target_names)

    eval_output = (
        "===============================================\n"
        "MODEL EVALUATION REPORT (Iris Classification)\n"
        "===============================================\n"
        f"Model: RandomForestClassifier(n_estimators=100, random_state=42)\n"
        f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n\n"
        "Classification Report:\n"
        f"{report_str}\n"
        "Training Set Feature Statistics:\n"
    )
    for col, stat in training_stats["stats"].items():
        eval_output += f"  - {col}: mean = {stat['mean']:.4f}, std = {stat['std']:.4f}\n"

    print(eval_output)

    # 6. Save artifacts
    joblib.dump(clf, "model/model.joblib")
    with open("model/training_stats.json", "w") as f:
        json.dump(training_stats, f, indent=2)

    with open("report/evaluation.txt", "w") as f:
        f.write(eval_output)

    print("Model saved to model/model.joblib")
    print("Training stats saved to model/training_stats.json")
    print("Evaluation report saved to report/evaluation.txt")

if __name__ == "__main__":
    train()
