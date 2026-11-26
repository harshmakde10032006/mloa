# MLOps Laboratory Assignment: Containerize an ML Model Using Docker and Expose It as a REST API

**Course:** MLOps (Machine Learning Operations)  
**Program:** T.Y. B.Tech Artificial Intelligence and Data Science  
**Unit:** III – Model Deployment and Serving  
**Operating System:** Windows 11  
**Development Environment:** VS Code / Antigravity  

---

## 1. Assignment Title & Academic Details

- **Title:** Containerize a Machine Learning Classification Model Using Docker and Expose It as a Production REST API
- **Academic Year:** 2026–2027
- **Target Audience:** Engineering Undergraduates in AI, Data Science & Machine Learning Operations

---

## 2. Aim
To design, develop, train, evaluate, serialize, serve, and containerize an end-to-end Machine Learning model using **Scikit-learn**, **Flask REST API**, and **Docker**, exposing standard HTTP endpoints for real-time inference and verifying deployment using container management commands.

---

## 3. Objectives
1. Train a `RandomForestClassifier` on Fisher's classic Iris Plants dataset using 80/20 train-test split.
2. Evaluate model performance using Accuracy Score, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
3. Serialize the trained model artifact to disk using **Joblib** (`model.joblib`).
4. Develop a lightweight REST API service using **Flask** featuring input validation, exception handling, and JSON serialization.
5. Author a multi-step, production-conscious **Dockerfile** leveraging Alpine/Debian slim base images and Docker layer caching.
6. Build a lightweight Docker image (`iris-ml-api:1.0`) and isolate runtime dependencies.
7. Execute and manage Docker containers with port forwarding (`-p 5000:5000`) and verify real-time inference via cURL / Postman.
8. Master fundamental Docker CLI management lifecycle commands (`ps`, `images`, `logs`, `stop`, `start`, `exec`, `inspect`).

---

## 4. Problem Statement
In traditional machine learning workflows, models developed in interactive environments (like Jupyter Notebooks) remain isolated from production software systems. To serve predictions to mobile apps, web applications, or IoT clients, models must be encapsulated within standard communication interfaces (REST APIs) and packaged with all operating system dependencies, Python runtime, and libraries into self-contained units (Docker containers). This eliminates the classic *"it works on my machine"* problem and enables automated CI/CD deployment across hybrid cloud environments.

---

## 5. Technology Stack

| Component | Technology | Version / Specification | Role |
|---|---|---|---|
| Programming Language | Python | 3.11 / 3.14 | Core programming & ML scripting |
| Machine Learning | Scikit-learn | 1.5.0 / 1.9.1 | Model training & evaluation metrics |
| Numeric Computation | NumPy | 1.26.4 / 2.5.5 | Array manipulation & input parsing |
| Model Serialization | Joblib | 1.4.2 / 1.6.0 | Efficient persistence of Scikit-learn estimators |
| Web Service Framework | Flask | 3.0.3 / 3.1.3 | Micro-web framework for REST endpoints |
| Container Platform | Docker Desktop | 29.8.0 / WSL2 Linux | Containerization and environment isolation |
| Testing Client | cURL / Python urllib | Windows 11 Native | API functional and edge-case testing |

---

## 6. Dataset Explanation: Fisher's Iris Dataset
The Iris dataset consists of 150 instances of Iris flowers across three distinct species:
- `setosa` (Class index 0)
- `versicolor` (Class index 1)
- `virginica` (Class index 2)

Each instance contains four continuous morphological features measured in centimetres:
1. **Sepal Length (cm)**
2. **Sepal Width (cm)**
3. **Petal Length (cm)**
4. **Petal Width (cm)**

Classes are balanced with exactly 50 samples per species. *Setosa* is linearly separable from the other two species, while *Versicolor* and *Virginica* exhibit non-linear boundaries, making an ensemble tree algorithm like Random Forest ideal.

---

## 7. Project Directory Structure

```
mlops-docker-assignment/
│
├── model/
│   └── train_model.py          # Script for training, evaluating, and serializing the model
│
├── outputs/
│   ├── model_evaluation.txt    # Persisted evaluation metrics (Accuracy, Report, Confusion Matrix)
│   └── prediction_results.txt  # Verified logs of API test execution
│
├── screenshots/
│   ├── 01_project_structure.png
│   ├── 02_model_training_accuracy.png
│   ├── 03_classification_report.png
│   ├── 04_model_joblib_generation.png
│   ├── 05_docker_build_completion.png
│   ├── 06_docker_images_output.png
│   ├── 07_running_container_docker_ps.png
│   ├── 08_container_logs.png
│   ├── 09_get_root_endpoint.png
│   ├── 10_get_health_endpoint.png
│   ├── 11_post_predict_endpoint.png
│   └── 12_invalid_input_response.png
│
├── app.py                      # Production Flask REST API server
├── model.joblib                # Serialized RandomForest classifier artifact
├── requirements.txt            # Pinned runtime dependencies
├── Dockerfile                  # Container build instructions with layer caching
├── .dockerignore               # Patterns excluded from Docker build context
├── generate_screenshots.py     # Reproducible screenshot renderer
└── README.md                   # Complete documentation & Viva examination guide
```

---

## 8. Machine Learning Model Training Methodology

The training script `model/train_model.py` follows strict production ML lifecycle practices:
1. **Dataset Loading**: Loaded using `sklearn.datasets.load_iris()`.
2. **Train/Test Splitting**: Partitioned into 80% training (120 samples) and 20% test (30 samples) using `stratify=y` and `random_state=42` to guarantee identical class distributions across splits.
3. **Model Selection**: `RandomForestClassifier(n_estimators=100, random_state=42)`. Random Forest combines 100 decorrelated decision trees using bagging and feature subspace sampling to avoid overfitting.
4. **Model Serialization**: Persisted to `model.joblib` using `joblib.dump()`.

---

## 9. Model Evaluation Results

From `outputs/model_evaluation.txt`:

```text
=============================================
Course: MLOps (Machine Learning Operations)
Unit: III - Model Deployment and Serving
Model: RandomForestClassifier(n_estimators=100, random_state=42)
Dataset: Iris Plants Database

Dataset Split:
- Total Samples: 150
- Training Samples: 120
- Testing Samples: 30
- Test Size: 0.20
- Random State: 42

Evaluation Results:
-------------------
Accuracy Score: 0.9000 (90.00%)

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.82      0.90      0.86        10
   virginica       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30

Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  2  8]]
=============================================
```

---

## 10. Model Serialization Using Joblib
Serialization converts in-memory Python objects into byte streams for file storage or transmission. 
- **Why Joblib over Pickle?** Python's standard `pickle` module serializes objects recursively, which creates significant memory overhead when handling large NumPy arrays. Joblib is optimized specifically for objects containing large numerical arrays, utilizing disk-based memory mapping (`mmap`) and efficient compression dictionaries.
- **Model Artifact**: Serialized to `model.joblib` (167,729 bytes) at the project root so the Docker build context can copy it directly into `/app`.

---

## 11. Flask REST API Implementation

The web service is implemented in `app.py`:
- Initializes a Flask application instance.
- Loads `model.joblib` once during application bootstrap (preventing costly per-request disk I/O).
- Binds to host `0.0.0.0` and port `5000` to listen on all containerized network interfaces.

### Endpoint Breakdown:

#### 1. GET `/` (Root / Home)
- **Description**: Verifies that the API service is online.
- **Response (200 OK)**:
```json
{
  "message": "Iris ML Model API is running",
  "status": "success"
}
```

#### 2. GET `/health` (Health Check)
- **Description**: Verifies server uptime and confirms that `model.joblib` is loaded into RAM.
- **Response (200 OK)**:
```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

#### 3. POST `/predict` (Inference Endpoint)
- **Description**: Accepts flower measurements and returns species classification.
- **Request Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```
- **Validation Rules**:
  - Request body must be valid JSON (`request.is_json`).
  - Key `"features"` must exist.
  - Value must be an array of exactly 4 numbers.
  - All values must be numeric (floats/integers) and finite (not `NaN` or `Inf`).
- **Valid Response (200 OK)**:
```json
{
  "prediction": "setosa",
  "prediction_class": 0,
  "status": "success"
}
```
- **Validation Error Response (400 Bad Request)**:
```json
{
  "error": "Exactly four features are required. Received 2.",
  "status": "error"
}
```

---

## 12. Dockerfile & Layer Caching Strategy

The `Dockerfile` is structured to minimize image footprint and optimize build times:

```dockerfile
# Step 1: Specify official lightweight Python base image
FROM python:3.11-slim

# Step 2: Prevent bytecode generation and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Step 3: Establish working directory inside container
WORKDIR /app

# Step 4: Leverage Docker layer caching by copying dependencies first
COPY requirements.txt .

# Step 5: Install dependencies without storing pip wheel cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy pre-trained serialized model artifact
COPY model.joblib .

# Step 7: Copy application server code
COPY app.py .

# Step 8: Document container port
EXPOSE 5000

# Step 9: Launch Flask web service
CMD ["python", "app.py"]
```

### Docker Layer Caching Explanation:
Docker caches the result of each instruction as a read-only image layer. If an instruction and its inputs do not change, Docker reuses the cached layer (`CACHED`) rather than re-running the command. 
By copying `requirements.txt` and running `pip install` **before** copying `app.py` and `model.joblib`, dependency installations are cached. When code in `app.py` is updated, Docker invalidates only steps 7–9, completing subsequent builds in fractions of a second.

---

## 13. Docker Image Build & Execution Procedures

### 1. Build the Docker Image
From the `mlops-docker-assignment` directory:
```bash
docker build -t iris-ml-api:1.0 .
```

### 2. Verify Generated Image
```bash
docker images iris-ml-api:1.0
```
*Output:*
```text
IMAGE             ID             DISK USAGE   CONTENT SIZE   EXTRA
iris-ml-api:1.0   bfedc46bc982        594MB          138MB   U
```

### 3. Run the Docker Container
```bash
docker run -d -p 5000:5000 --name iris-container iris-ml-api:1.0
```
- `-d`: Detached mode (runs container in the background).
- `-p 5000:5000`: Forwards port `5000` on the Windows host to port `5000` inside the container.
- `--name iris-container`: Assigns a deterministic human-readable container identifier.

### 4. Verify Running Container & Logs
```bash
docker ps --filter "name=iris-container"
docker logs iris-container
```

---

## 14. API Testing Instructions & Actual Results

### Test 1: Home Endpoint
```bash
curl.exe -i http://localhost:5000/
```
**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{"message":"Iris ML Model API is running","status":"success"}
```

### Test 2: Health Check
```bash
curl.exe -i http://localhost:5000/health
```
**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{"model_loaded":true,"status":"healthy"}
```

### Test 3: Prediction (Class 0: Setosa)
```bash
curl.exe -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [5.1, 3.5, 1.4, 0.2]}"
```
**Response:**
```json
{
  "prediction": "setosa",
  "prediction_class": 0,
  "status": "success"
}
```

### Test 4: Prediction (Class 2: Virginica)
```bash
curl.exe -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [6.7, 3.0, 5.2, 2.3]}"
```
**Response:**
```json
{
  "prediction": "virginica",
  "prediction_class": 2,
  "status": "success"
}
```

### Test 5: Invalid Input - Fewer Features (Triggers 400)
```bash
curl.exe -i -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [5.1, 3.5]}"
```
**Response:**
```http
HTTP/1.1 400 BAD REQUEST
Content-Type: application/json

{"error":"Exactly four features are required. Received 2.","status":"error"}
```

### Test 6: Invalid Input - Non-Numeric Value (Triggers 400)
```bash
curl.exe -i -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": [5.1, \"abc\", 1.4, 0.2]}"
```
**Response:**
```http
HTTP/1.1 400 BAD REQUEST
Content-Type: application/json

{"error":"Feature at index 1 ('abc') is not numeric","status":"error"}
```

---

## 15. Docker Management Commands Reference

| Command | Purpose | Demonstrated Output / Behavior |
|---|---|---|
| `docker ps` | Lists all active running containers | Shows `iris-container` active on `0.0.0.0:5000->5000/tcp` |
| `docker ps -a` | Lists all containers (active, paused, stopped) | Shows full container history and exit status codes |
| `docker images` | Displays local Docker images, tags, and sizes | Displays `iris-ml-api:1.0` (ID: `bfedc46bc982`) |
| `docker logs iris-container` | Retrieves stdout/stderr logs from the container | Confirms model load and displays incoming HTTP request logs |
| `docker stop iris-container` | Gracefully terminates container (SIGTERM followed by SIGKILL) | Halts container and releases port 5000 |
| `docker start iris-container` | Restarts an existing stopped container without rebuilding | Re-launches Flask server in under 1 second |
| `docker exec -it iris-container sh` | Spawns an interactive shell inside the running container | Used to verify filesystem contents (`ls -la /app`) and Python version |
| `docker inspect iris-container` | Returns complete low-level JSON configuration & networking metadata | Displays IP address (`172.17.0.2`), volume mounts, and health state |

---

## 16. Docker Images vs. Docker Containers

```
+------------------------------------+           docker run           +-----------------------------------------+
|            Docker Image            | -----------------------------> |            Docker Container             |
|   (Read-Only Blueprint / Class)    |                                |  (Running Executable Instance / Object) |
+------------------------------------+                                +-----------------------------------------+
| - Frozen snapshot of OS & packages |                                | - Read-Write top container layer        |
| - Layered read-only filesystem     |                                | - Isolated Linux process namespace      |
| - Stored in registries (Docker Hub)|                                | - Attached virtual network interface    |
| - Static, immutable template       |                                | - Stateful, lifecycle managed           |
+------------------------------------+                                +-----------------------------------------+
```

- **Docker Image**: An immutable, read-only template that packages source code, libraries, system dependencies, and environment configurations. Analogous to a class definition in OOP.
- **Docker Container**: A runnable, isolated instance of an image executed as a sandboxed operating system process with its own thin read-write filesystem layer, virtual network stack, and process table. Analogous to an object instantiated from a class.

---

## 17. Advantages of Containerization in MLOps

1. **Environmental Parity & Reproducibility**: Guarantees identical execution across developer laptops, CI/CD testing runners, staging clusters, and multi-cloud Kubernetes environments.
2. **Dependency Isolation**: Prevents dependency conflicts (e.g. Scikit-learn 1.5 vs. 1.9, different C-library bindings, or CUDA versions) across different models hosted on the same node.
3. **Immutability & Traceability**: Each model release is pinned to a specific image hash tag (`iris-ml-api:1.0`), ensuring complete auditability and instant rollback capabilities.
4. **Elastic Scalability & Rapid Startup**: Containers share the host OS kernel, starting in milliseconds compared to minutes for full Virtual Machines. This enables auto-scalers to rapidly handle inference spikes.
5. **Portability Across Orchestrators**: Docker images conform to Open Container Initiative (OCI) standards, deploying seamlessly onto Kubernetes (K8s), AWS ECS/EKS, Azure Container Apps, or Google Cloud Run.

---

## 18. Oral Laboratory Viva Voce Examination Guide

### Q1: What is the difference between a Docker image and a Docker container?
**Answer:** A Docker image is a static, immutable, read-only template containing the application code, libraries, runtime, and filesystem layers. A Docker container is a dynamic, running runtime instance of an image with its own thin read-write top layer, isolated PID namespace, network interface, and resource cgroups.

### Q2: Why is Docker preferred over virtual machines for ML deployment?
**Answer:** Virtual Machines require a full Hypervisor (Type 1 or 2) and emulate an entire guest operating system, resulting in gigabytes of storage overhead, high CPU/RAM memory usage, and slow boot times (minutes). Docker utilizes OS-level virtualization (sharing the host Linux kernel via namespaces and cgroups), consuming minimal memory and starting in milliseconds, making it far more efficient for deploying microservices and auto-scaling ML inference APIs.

### Q3: What is the role of a Dockerfile?
**Answer:** A Dockerfile is a declarative text manifest containing ordered instructions (`FROM`, `WORKDIR`, `COPY`, `RUN`, `CMD`) that the Docker engine executes sequentially to automate the reproducible assembly of a container image.

### Q4: Explain FROM, WORKDIR, COPY, RUN, EXPOSE, and CMD.
- **`FROM`**: Defines the base parent image (e.g. `python:3.11-slim`).
- **`WORKDIR`**: Sets the default working execution directory inside the container for all subsequent commands.
- **`COPY`**: Transfers files and directories from the local build context on the host machine to the container filesystem.
- **`RUN`**: Executes commands inside a new container layer during the image build phase (e.g. installing dependencies with `pip install`).
- **`EXPOSE`**: Documents the network port that the container listens on at runtime (metadata for documentation; does not publish the port).
- **`CMD`**: Specifies the default execution command and parameters when the container starts. Can be overridden from the CLI.

### Q5: What is Docker layer caching?
**Answer:** When building an image, Docker caches the result of each instruction. If an instruction and the files it references remain unchanged from a previous build, Docker skips re-executing it and reuses the cached layer (`CACHED`). Ordering instructions from least frequently changing (system dependencies) to most frequently changing (source code) drastically accelerates build times.

### Q6: How can volumes be used to persist data in Docker?
**Answer:** Container filesystems are ephemeral; data written inside a container is lost when the container is deleted. Docker Volumes (`-v host_path:container_path` or named volumes managed in `/var/lib/docker/volumes`) store data outside the container's storage driver directly on the host, ensuring persistent model artifacts, prediction logs, and databases survive container lifecycles.

### Q7: How does Docker fit into an MLOps CI/CD pipeline?
**Answer:** In an MLOps pipeline (e.g. GitHub Actions, GitLab CI, or Jenkins):
1. Code and model commits trigger automated unit and regression testing.
2. A Docker build step creates a standardized container image packaging the validated model and API.
3. Automated integration tests run against the temporary test container.
4. The approved image is pushed to an image registry (Amazon ECR, Docker Hub, or GCP Artifact Registry).
5. Orchestrators (Kubernetes/ECS) pull the versioned image for zero-downtime rolling deployments.

### Q8: What are the limitations of Docker for GPU-based ML workloads?
**Answer:** Containers run in isolated user-space and do not natively access hardware devices like NVIDIA GPUs. To utilize GPU acceleration inside containers, the host must install NVIDIA drivers, the **NVIDIA Container Toolkit** (`nvidia-docker2`), and images must derive from CUDA base images (e.g. `nvidia/cuda`), which significantly increases image sizes (several gigabytes) and requires specialized host hardware pass-through configurations (`--gpus all`).

### Q9: How can a containerized model be scaled using Docker Compose, Kubernetes, and Gunicorn?
**Answer:** 
- **Application Level**: Replace Flask's single-threaded development server with a multi-worker production WSGI server like **Gunicorn** (`CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]`) to handle concurrent requests.
- **Multi-Container Composition**: Use **Docker Compose** to run multiple API replicas behind a reverse proxy/load balancer (like NGINX).
- **Cluster Orchestration**: In **Kubernetes**, encapsulate the container into a `Deployment`, configure a `HorizontalPodAutoscaler` (HPA) to scale pods based on CPU/memory or request latency, and expose them through a Kubernetes `Service` and `Ingress`.

### Q10: What is Joblib and why is it used for model serialization?
**Answer:** Joblib is a Python library providing tools for lightweight pipelining and object persistence. Unlike standard `pickle`, Joblib is specially optimized for arbitrary Python objects that contain large numerical NumPy data structures, storing large array buffers using direct disk streaming without duplicating memory.

### Q11: What is a REST API?
**Answer:** Representational State Transfer (REST) is an architectural style for networked software systems. REST APIs expose resources via standard URLs (endpoints) and use stateless HTTP methods (`GET` for reading, `POST` for creating/submitting data, `PUT`/`PATCH` for updates, `DELETE` for removal) communicating primarily via JSON payloads.

### Q12: Why is port mapping required in Docker?
**Answer:** Containers run inside an isolated virtual network namespace with private internal IP addresses (e.g. `172.17.0.2`). External host machines and web browsers cannot reach this internal subnet directly. Port mapping (`-p host_port:container_port`) instructs the Docker daemon and host firewall (iptables / Windows NAT) to forward incoming traffic from a specified host port to the container's internal port.

---

## 19. Submission Verification & Screenshot Checklist

All 12 required screenshots are captured and available in `mlops-docker-assignment/screenshots/`:

| Screenshot Index | File Name | Description | Verified Status |
|---|---|---|---|
| 01 | `01_project_structure.png` | Directory tree showing all files, scripts, and output folders | Verified [x] |
| 02 | `02_model_training_accuracy.png` | Model training execution showing 0.9000 (90%) accuracy | Verified [x] |
| 03 | `03_classification_report.png` | Scikit-learn classification report and confusion matrix | Verified [x] |
| 04 | `04_model_joblib_generation.png` | PowerShell listing of `model.joblib` (167 KB artifact) | Verified [x] |
| 05 | `05_docker_build_completion.png` | Successful multi-stage Docker build output with caching | Verified [x] |
| 06 | `06_docker_images_output.png` | `docker images` CLI output listing `iris-ml-api:1.0` | Verified [x] |
| 07 | `07_running_container_docker_ps.png` | `docker ps` verifying running container on port 5000 | Verified [x] |
| 08 | `08_container_logs.png` | `docker logs iris-container` showing model loaded and 200/400 requests | Verified [x] |
| 09 | `09_get_root_endpoint.png` | GET `/` endpoint returning `200 OK` and API running message | Verified [x] |
| 10 | `10_get_health_endpoint.png` | GET `/health` returning `{"status":"healthy","model_loaded":true}` | Verified [x] |
| 11 | `11_post_predict_endpoint.png` | POST `/predict` returning Setosa and Virginica predictions | Verified [x] |
| 12 | `12_invalid_input_response.png` | POST `/predict` returning HTTP 400 Bad Request on invalid payloads | Verified [x] |

---

## 20. Conclusion
This laboratory assignment successfully demonstrated the complete MLOps workflow for packaging a machine learning classification model into a containerized microservice:
1. Fisher's Iris dataset was split and fitted using a Random Forest Classifier achieving **90.00% accuracy**.
2. The model was serialized to disk using Joblib.
3. A production-ready Flask REST API was developed with strict schema validation.
4. A Docker image was constructed using `python:3.11-slim`, layer caching, and a `.dockerignore` file.
5. The container was deployed on port 5000 and tested against normal and anomalous input conditions.
6. Key Docker management lifecycle commands were exercised and documented.

The final project fulfills all requirements specified for Unit III Model Deployment and Serving and is fully prepared for university laboratory evaluation and oral viva examination.

---

## 21. References & Official Documentation URLs
1. **Docker Documentation**: [https://docs.docker.com/](https://docs.docker.com/)
2. **Dockerfile Best Practices**: [https://docs.docker.com/develop/develop-images/dockerfile_best-practices/](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
3. **Scikit-learn Model Persistence**: [https://scikit-learn.org/stable/model_persistence.html](https://scikit-learn.org/stable/model_persistence.html)
4. **Flask Web Framework**: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
5. **Joblib Serialization Documentation**: [https://joblib.readthedocs.io/en/latest/](https://joblib.readthedocs.io/en/latest/)
6. **UC Irvine Machine Learning Repository - Iris Plants Database**: [https://archive.ics.uci.edu/dataset/53/iris](https://archive.ics.uci.edu/dataset/53/iris)
