import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Create MLflow Experiment
# ----------------------------
mlflow.set_experiment("Iris_Classification_MLflow")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/iris.csv")

X = df.drop("target", axis=1)
y = df["target"]

# ----------------------------
# Preprocessing
# ----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Models
# ----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

best_accuracy = 0
best_model = None
best_model_name = ""

# ----------------------------
# Train and Log Experiments
# ----------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        # Log Parameters
        mlflow.log_param("model", name)

        if hasattr(model, "max_depth"):
            mlflow.log_param("max_depth", model.max_depth)

        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", model.n_estimators)

        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Confusion Matrix
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} Confusion Matrix")

        cm_file = f"{name}_confusion_matrix.png"
        plt.savefig(cm_file)
        plt.close()

        mlflow.log_artifact(cm_file)

        # Classification Report
        report = classification_report(y_test, predictions)

        report_file = f"{name}_classification_report.txt"

        with open(report_file, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_file)

        # Save Model
        mlflow.sklearn.log_model(model, "model")

        print(f"{name}: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

# ----------------------------
# Register Best Model
# ----------------------------
with mlflow.start_run(run_name="Best_Model_Registration"):

    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="Best_Iris_Model"
    )

    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_param("best_model", best_model_name)

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)