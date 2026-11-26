import os
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ---------------------------------------------------------
# Exact Required Prometheus Metrics
# ---------------------------------------------------------
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)

ml_prediction_requests_total = Counter(
    "ml_prediction_requests_total",
    "Total count of ML prediction requests",
    ["endpoint", "status"]
)

ml_prediction_errors_total = Counter(
    "ml_prediction_errors_total",
    "Total count of ML prediction errors (422, 500, etc.)",
    ["endpoint"]
)

ml_prediction_latency_seconds = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction request latency in seconds",
    ["endpoint"],
    buckets=LATENCY_BUCKETS
)

ml_predictions_total = Counter(
    "ml_predictions_total",
    "Total count of successful ML predictions by predicted class",
    ["predicted_class"]
)

# ---------------------------------------------------------
# Additional Rich Metrics for Dashboard & Drift Monitoring
# ---------------------------------------------------------
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests handled by method, endpoint, and status",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Total HTTP request latency in seconds",
    ["endpoint"],
    buckets=LATENCY_BUCKETS
)

model_prediction_confidence = Histogram(
    "model_prediction_confidence",
    "Confidence score distribution of predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0)
)

model_feature_drift_zscore = Gauge(
    "model_feature_drift_zscore",
    "Z-score of rolling mean feature values against training baseline",
    ["feature"]
)

model_feedback_total = Counter(
    "model_feedback_total",
    "Ground truth feedback count by outcome",
    ["outcome"]
)

model_inference_errors_total = Counter(
    "model_inference_errors_total",
    "Internal inference exceptions encountered"
)

model_info = Gauge(
    "model_info",
    "Metadata about the loaded ML model",
    ["version", "model_type"]
)

# ---------------------------------------------------------
# State & Data structures
# ---------------------------------------------------------
MODEL_VERSION = "1.0.0"
model = None
training_stats: Dict[str, Any] = {}
feature_history: Dict[str, deque] = {
    "sepal_length": deque(maxlen=100),
    "sepal_width": deque(maxlen=100),
    "petal_length": deque(maxlen=100),
    "petal_width": deque(maxlen=100),
}

# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class IrisInput(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0.0, le=20.0)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0.0, le=20.0)
    petal_length: float = Field(..., description="Petal length in cm", ge=0.0, le=20.0)
    petal_width: float = Field(..., description="Petal width in cm", ge=0.0, le=20.0)

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    model_version: str

class FeedbackInput(BaseModel):
    prediction: Optional[str] = None
    actual: Optional[str] = None
    outcome: Optional[str] = None  # 'correct' or 'incorrect'

# ---------------------------------------------------------
# Lifespan
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, training_stats
    model_path = os.getenv("MODEL_PATH", "model/model.joblib")
    stats_path = os.getenv("STATS_PATH", "model/training_stats.json")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")

    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            training_stats = json.load(f)
        print(f"Loaded training stats from {stats_path}")
    else:
        print(f"Warning: Stats file not found at {stats_path}")

    model_info.labels(version=MODEL_VERSION, model_type="RandomForestClassifier").set(1)

    # Initialize drift gauges to 0.0
    for feat in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        model_feature_drift_zscore.labels(feature=feat).set(0.0)

    yield

app = FastAPI(
    title="ML Model Serving & Monitoring API",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# ---------------------------------------------------------
# Middleware for Exact Request & Error Tracking
# ---------------------------------------------------------
@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    raw_path = request.url.path
    if raw_path in ["/predict", "/feedback", "/health", "/metrics"]:
        endpoint = raw_path
    else:
        endpoint = "unmatched"

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as exc:
        status_code = 500
        raise exc
    finally:
        latency = time.time() - start_time
        status_str = str(status_code)

        # Track HTTP general metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=status_str
        ).inc()
        http_request_duration_seconds.labels(endpoint=endpoint).observe(latency)

        # Track assignment required prediction metrics
        if endpoint == "/predict":
            ml_prediction_requests_total.labels(
                endpoint="/predict",
                status=status_str
            ).inc()
            
            if status_code >= 400:
                ml_prediction_errors_total.labels(endpoint="/predict").inc()
                
            ml_prediction_latency_seconds.labels(endpoint="/predict").observe(latency)

    return response

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_version": MODEL_VERSION
    }

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: IrisInput):
    if model is None:
        model_inference_errors_total.inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )

    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    
    try:
        # Perform inference using DataFrame to match feature names
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        df_input = pd.DataFrame([features], columns=feature_names)
        probs = model.predict_proba(df_input)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        target_names = training_stats.get("target_names", ["setosa", "versicolor", "virginica"])
        predicted_class = target_names[pred_idx] if pred_idx < len(target_names) else str(pred_idx)

        # Record required metric
        ml_predictions_total.labels(predicted_class=predicted_class).inc()

        # Record rich metrics
        model_prediction_confidence.observe(confidence)

        # Drift tracking (sliding window of 100, calculate z-score when n >= 20)
        feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        stats_dict = training_stats.get("stats", {})

        for name, val in zip(feature_names, features):
            q = feature_history[name]
            q.append(val)
            n = len(q)
            if n >= 20 and name in stats_dict:
                rolling_mean = np.mean(q)
                train_mean = stats_dict[name]["mean"]
                train_std = stats_dict[name]["std"]
                se = train_std / np.sqrt(n) if train_std > 0 else 1e-6
                z = (rolling_mean - train_mean) / se
                model_feature_drift_zscore.labels(feature=name).set(float(z))

        return PredictionResponse(
            prediction=predicted_class,
            confidence=round(confidence, 4),
            model_version=MODEL_VERSION
        )

    except Exception as e:
        model_inference_errors_total.inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/feedback")
async def feedback(data: FeedbackInput):
    outcome = data.outcome
    if outcome is None:
        if data.prediction and data.actual:
            outcome = "correct" if data.prediction.strip().lower() == data.actual.strip().lower() else "incorrect"
        else:
            outcome = "correct"
            
    norm_outcome = "correct" if outcome.lower() == "correct" else "incorrect"
    model_feedback_total.labels(outcome=norm_outcome).inc()
    
    return {
        "status": "feedback_recorded",
        "outcome": norm_outcome
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
