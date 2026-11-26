from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load Iris dataset
X, y = load_iris(return_X_y=True)

print("Iris dataset loaded successfully.")

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

print("Model trained successfully.")

# Get project root directory
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Save model in project root
model_path = os.path.join(project_dir, "iris_model.pkl")

joblib.dump(model, model_path)

print("Model saved successfully!")
print("Location:", model_path)