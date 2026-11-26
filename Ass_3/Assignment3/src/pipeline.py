import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from preprocessing import build_preprocessor
from feature_selection import select_features

# Load Dataset
df = pd.read_csv("../data/titanic.csv")

print("Dataset Loaded Successfully")
print("Original Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

# Remove duplicates
df = df.drop_duplicates()
print("\nDuplicate Records Removed")

# Separate target
target = "Survived"
X = df.drop(columns=[target])
y = df[target]

# Build preprocessing pipeline
preprocessor, num_cols, cat_cols = build_preprocessor(X)

# Transform data
X_processed = preprocessor.fit_transform(X)

# Feature selection
selector, X_selected = select_features(X_processed, y, k=10)

# Full reusable pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, "../models/feature_pipeline.pkl")
joblib.dump(selector, "../models/feature_selector.pkl")

# Save transformed dataset
feature_names = preprocessor.get_feature_names_out()
selected_features = feature_names[selector.get_support()]

pd.DataFrame(X_selected, columns=selected_features).to_csv(
    "../output/transformed_data.csv",
    index=False
)

print("\nMissing Values Handled")
print("Categorical Features Encoded")
print("Numerical Features Scaled")

print("\nSelected Features:")
for feature in selected_features:
    print("-", feature)

print("\nTransformed Dataset Shape:", X_selected.shape)
print("Pipeline Saved Successfully")

# Reuse pipeline
loaded_pipeline = joblib.load("../models/feature_pipeline.pkl")

sample = X.head(3)
prediction = loaded_pipeline.predict(sample)

print("\nPipeline Reused Successfully")
print("Sample Predictions:", prediction)