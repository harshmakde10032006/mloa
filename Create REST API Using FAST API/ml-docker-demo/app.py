from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("iris_model.pkl")

# Iris class names
class_names = ["setosa", "versicolor", "virginica"]


# Home page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Iris Model API is running"
    })


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.get_json()

    # Get the four input features
    features = np.array(data["features"]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Convert numerical prediction to class name
    predicted_class = class_names[int(prediction[0])]

    return jsonify({
        "prediction": predicted_class
    })


# Start Flask server
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )   