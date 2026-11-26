# Iris Classification REST API with FastAPI

A production-ready REST API built with **FastAPI**, **scikit-learn**, and **Pydantic** to serve real-time predictions from a machine learning model trained on the Iris dataset.

---

## 1. Project Overview & Architecture

### Purpose
An ML model developed inside a Jupyter Notebook cannot be directly consumed by frontend web applications, mobile apps, or enterprise microservices. Exposing the model via a REST API provides a standardized, platform-independent HTTP interface.

### Processing Workflow
```
[ Client / Web App / Mobile ]
            │  (HTTP POST JSON)
            ▼
    [ FastAPI Server ]
            │
            ▼
   [ Pydantic Validation ]  ──(Invalid)──▶ [ 422 Unprocessable Content ]
            │ (Validated Features)
            ▼
    [ Loaded ML Model ]
(RandomForest in app.state)
            │
            ▼
   [ Class & Probability ]
            │
            ▼
[ JSON Response with Confidence ]
```

---

## 2. Project Directory Structure

```
fastapi-ml-assignment/
├── artifacts/
│   └── iris_model.joblib      # Serialized trained model & metadata bundle
├── tests/
│   └── test_api.py            # Automated pytest suite covering 8 test scenarios
├── main.py                    # FastAPI application & endpoint definitions
├── train_model.py             # Script to train, evaluate, and save model bundle
├── requirements.txt           # Project dependencies
└── README.md                  # Comprehensive documentation
```

---

## 3. Setup and Installation

### Step 1: Create and Activate Virtual Environment
**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Train the Machine Learning Model

Run the training script to train a `RandomForestClassifier` on the Iris flower dataset and save the artifact bundle:

```bash
python train_model.py
```

**Expected Output:**
```
Model saved at: artifacts/iris_model.joblib
Test accuracy: 0.9000
```

The artifact bundle (`artifacts/iris_model.joblib`) packages:
- The trained scikit-learn model
- Target class names (`setosa`, `versicolor`, `virginica`)
- Feature names
- Model test accuracy
- Model version (`1.0.0`)

---

## 5. Running the FastAPI Application

Start the development server using the FastAPI CLI or Uvicorn:

```bash
uvicorn main:app --reload
```
or
```bash
fastapi dev main.py
```

The server starts locally at: **`http://127.0.0.1:8000`**

### Available Documentation Interfaces
- **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI JSON Schema:** [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json)

---

## 6. API Endpoints Specification

### 1. Root Endpoint
- **URL:** `GET /`
- **Purpose:** Welcome message, links to documentation and health check.
- **Response (200 OK):**
```json
{
  "message": "Welcome to the Iris Classification API",
  "documentation": "/docs",
  "health_check": "/health"
}
```

### 2. Health-Check Endpoint
- **URL:** `GET /health`
- **Purpose:** Verifies server availability and confirms the ML model bundle is loaded in memory.
- **Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Model Information Endpoint
- **URL:** `GET /model-info`
- **Purpose:** Retrieves metadata about the loaded machine learning model.
- **Response (200 OK):**
```json
{
  "model_type": "RandomForestClassifier",
  "model_version": "1.0.0",
  "feature_names": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "target_names": [
    "setosa",
    "versicolor",
    "virginica"
  ],
  "test_accuracy": 0.9
}
```

### 4. Prediction Endpoint
- **URL:** `POST /predict`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
- **Validation Rules:**
  - `sepal_length`: float, `> 0` and `<= 20`
  - `sepal_width`: float, `> 0` and `<= 20`
  - `petal_length`: float, `> 0` and `<= 20`
  - `petal_width`: float, `> 0` and `<= 20`
- **Response (200 OK):**
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "confidence": 1.0,
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  },
  "model_version": "1.0.0"
}
```

---

## 7. Testing with cURL

### Health Check:
```bash
curl http://127.0.0.1:8000/health
```

### Model Info:
```bash
curl http://127.0.0.1:8000/model-info
```

### Valid Prediction:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

### Invalid Prediction (Negative Value - triggers 422):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": -5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

---

## 8. Automated Testing

The automated test suite in `tests/test_api.py` uses pytest and FastAPI's `TestClient` (backed by `httpx`) to test the application in-process without spinning up an external network port.

Run the test suite:
```bash
pytest -v
```

### Test Cases Matrix
| Test Case | Method & Path | Input / Condition | Expected Status |
|---|---|---|---|
| `test_root_endpoint` | `GET /` | None | `200 OK` |
| `test_health_endpoint` | `GET /health` | Application running & model in state | `200 OK` |
| `test_model_info_endpoint` | `GET /model-info` | Model metadata loaded | `200 OK` |
| `test_valid_prediction` | `POST /predict` | All measurements valid float values | `200 OK` |
| `test_negative_feature_value` | `POST /predict` | `sepal_length` = -5.1 (< 0) | `422 Unprocessable` |
| `test_missing_feature` | `POST /predict` | `petal_width` omitted | `422 Unprocessable` |
| `test_incorrect_data_type` | `POST /predict` | String provided instead of float | `422 Unprocessable` |
| `test_unknown_endpoint` | `GET /unknown` | Non-existent route | `404 Not Found` |

---

## 9. Containerization and AWS Deployment Architecture

### Docker Containerization
To deploy this application consistently across different environments, package it into a Docker image using a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained artifacts
COPY artifacts/ artifacts/
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run locally:
```bash
docker build -t iris-fastapi-app:latest .
docker run -p 8000:8000 iris-fastapi-app:latest
```

### Deployment Options on AWS

1. **AWS App Runner / Elastic Beanstalk (PaaS):**
   - Direct deployment from source or container image.
   - Handles auto-scaling, SSL termination, and health checks automatically.

2. **Amazon ECS (Elastic Container Service) with AWS Fargate (Serverless Containers):**
   - Docker image pushed to Amazon ECR (Elastic Container Registry).
   - ECS Task Definition runs container on Fargate without managing underlying EC2 VMs.
   - Application Load Balancer (ALB) routes traffic across tasks and performs `/health` checks.

3. **Amazon EC2 (IaaS):**
   - Virtual machine instance running Ubuntu.
   - Nginx reverse proxy routing traffic to Uvicorn running behind `systemd` or `supervisor`.
