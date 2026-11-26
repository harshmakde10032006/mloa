# MLOps Assignment 8: Monitor a Machine Learning Model Using Prometheus and Grafana

An end-to-end, production-grade telemetry and observability stack for serving, monitoring, and benchmarking a Scikit-Learn machine learning classifier.

---

## 1. Project Overview & Architecture

This project deploys an Iris Flower Classification microservice built with **FastAPI**, fully instrumented using `prometheus_client`. The service is integrated into a multi-container monitoring topology orchestrated with **Docker Compose**, including **Prometheus (v2.51.0)** for metric collection and dynamic alerting, **Alertmanager (v0.27.0)** for alert routing, **Node Exporter (v1.7.0)** for system host telemetry, and **Grafana (v10.4.1)** for real-time visualization across three structured rows: Service Health, ML Performance, and Infrastructure.

### Architecture Dataflow
```
[ Traffic Simulator (simulate.py) ]
             │  (HTTP POST inference & feedback)
             ▼
[ FastAPI Model Service (:8000) ] ◄── (Loads model.joblib & training_stats.json)
             │
             │  (Pull scrape /metrics every 5s)
             ▼
   [ Prometheus Server (:9090) ] ◄── [ Node Exporter (:9100) ]
        │                   │
        │ (PromQL queries)  │ (Dispatches triggered alerts)
        ▼                   ▼
[ Grafana (:3001) ]   [ Alertmanager (:9093) ]
```

---

## 2. Directory Structure

```
mlops-monitoring/
├── app.py                                # FastAPI serving app with Prometheus middleware & drift tracking
├── train.py                              # Scikit-learn training script (Iris dataset, 80/20 split)
├── simulate.py                           # Traffic simulator (normal, drift, errors, batch modes)
├── run_experiments.py                    # Automated test harness & screenshot orchestrator
├── generate_report.py                    # ReportLab script generating the submission PDF
├── generate_architecture.py              # Architecture diagram generator (matplotlib)
├── requirements.txt                      # Pinned Python dependencies
├── Dockerfile                            # Production Dockerfile (python:3.11-slim)
├── docker-compose.yml                    # Multi-container orchestration (5 pinned services)
├── .env.example                          # Environment variable template
├── .env                                  # Grafana credentials & port mapping (.gitignored)
├── .gitignore                            # Ignores .env, cache, virtual environments
├── README.md                             # Comprehensive project documentation
├── model/
│   ├── model.joblib                      # Trained RandomForestClassifier
│   └── training_stats.json               # Baseline feature statistics (mean, std)
├── prometheus/
│   ├── prometheus.yml                    # Prometheus scrape configurations (5s interval)
│   └── alert_rules.yml                   # 5 dynamic alerting rules
├── alertmanager/
│   └── alertmanager.yml                  # Minimal alert routing configuration
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/prometheus.yml    # Auto-provisioned Prometheus datasource
│   │   └── dashboards/dashboards.yml     # Auto-provisioned dashboard provider
│   └── dashboards/
│       └── ml-model-monitoring.json      # Complete 3-row Grafana dashboard JSON
├── screenshots/                          # 15 real browser captures of running stack
│   ├── 01-model-api-docs-or-sample-request.png
│   ├── 02-metrics-endpoint.png
│   ├── 03-prometheus-targets-up.png
│   ├── 04-prometheus-query-request-rate.png
│   ├── 05-prometheus-alerts-page-normal.png
│   ├── 06-grafana-datasource.png
│   ├── 07-grafana-dashboard-normal.png
│   ├── 08-grafana-dashboard-errors.png
│   ├── 09-prometheus-alert-higherrorrate-firing.png
│   ├── 10-grafana-dashboard-drift.png
│   ├── 11-prometheus-alert-featuredrift-firing.png
│   ├── 12-grafana-dashboard-outage-down.png
│   ├── 13-prometheus-alert-modelservicedown-firing.png
│   ├── 14-grafana-dashboard-recovery.png
│   └── 15-terminal-docker-compose-ps.png
└── report/
    ├── MLOps_Assignment8_Report.pdf      # Final comprehensive submission PDF report
    ├── evaluation.txt                    # Baseline test set evaluation output
    ├── architecture_diagram.png          # High-resolution architecture diagram
    └── experiment_metrics.json           # Raw unsimulated measurements from live stack
```

---

## 3. Service Ports & Public Exposure Warning

| Service | Container Port | Host Port | Purpose |
| :--- | :---: | :---: | :--- |
| **model-service** | `8000` | `8000` | FastAPI ML serving, `/predict`, `/feedback`, `/health`, `/metrics` |
| **prometheus** | `9090` | `9090` | Prometheus TSDB, Targets (`/targets`), Alerts (`/alerts`), PromQL |
| **alertmanager** | `9093` | `9093` | Alertmanager routing and notification dashboard |
| **node-exporter** | `9100` | `9100` | Host hardware CPU and Memory metrics |
| **grafana** | `3000` | `3001` | Grafana visual dashboard portal (mapped to 3001 to prevent conflicts) |

> ⚠️ **SECURITY NOTICE**: Prometheus (`:9090`), Alertmanager (`:9093`), and Grafana (`:3001`) are internal operational observability tools. **They must never be exposed publicly to the open internet without an authenticating reverse proxy (e.g. Nginx/OAuth2-Proxy), mTLS, or VPN restriction**, as raw operational endpoints expose sensitive architectural internals, traffic volumes, and runtime configurations.

---

## 4. Setup and Execution Instructions

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (Docker Desktop on Windows/macOS or Docker Engine on Linux)

### Step 1: Train the ML Model
```bash
python train.py
```
This trains a `RandomForestClassifier` on Fisher's Iris dataset (80/20 stratified split), prints the classification report, and outputs `model/model.joblib`, `model/training_stats.json`, and `report/evaluation.txt`.

### Step 2: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default Grafana credentials are set to `admin` / `admin`, accessible at `http://localhost:3001`.

### Step 3: Launch Docker Stack
```bash
docker compose up -d --build
```
Verify all 5 containers are running:
```bash
docker compose ps
```

Verify service endpoints:
- FastAPI Swagger UI: `http://localhost:8000/docs`
- Prometheus Targets: `http://localhost:9090/targets`
- Prometheus Alerts: `http://localhost:9090/alerts`
- Grafana Dashboard: `http://localhost:3001/d/ml-model-monitoring/ml-model-monitoring`

---

## 5. Traffic Simulation Modes (`simulate.py`)

The simulator tests operational resilience, data drift, error interception, and exact counter validation:

- **Normal Traffic**: Sends valid Iris vectors with 30% delayed feedback (~95% accuracy):
  ```bash
  python simulate.py --mode normal --rps 10 --duration 120
  ```
- **Error Injection (~30% invalid payloads)**: Sends ill-formed JSON, missing fields, strings, and negative values to trigger HTTP 422:
  ```bash
  python simulate.py --mode errors --rps 10 --duration 90
  ```
- **Data Drift**: Shifts `petal_length` +2.0 and `sepal_length` +1.0, reducing accuracy to ~60%:
  ```bash
  python simulate.py --mode drift --rps 10 --duration 90
  ```
- **Deterministic Batch Validation**: Sends exact counts of valid and invalid requests:
  ```bash
  python simulate.py --mode batch --valid 100 --invalid 20
  ```

---

## 6. Monitored Prometheus Metrics & PromQL Queries

### Required & Custom Metrics
- `ml_prediction_requests_total` (Counter, labels: `endpoint`, `status`): Captured via HTTP middleware to count all requests, including pre-handler 422 validation failures.
- `ml_prediction_errors_total` (Counter, labels: `endpoint`): Increments on HTTP 422 or 500 status codes.
- `ml_prediction_latency_seconds` (Histogram, labels: `endpoint`, buckets: 0.005 to 2.5): Request execution latency.
- `ml_predictions_total` (Counter, labels: `predicted_class`): Counts successful inferences partitioned by Iris class.
- `model_feature_drift_zscore` (Gauge, labels: `feature`): Rolling 100-sample Z-score vs. training baseline:  
  $$Z = \frac{\bar{X}_{\text{rolling}} - \mu_{\text{train}}}{\sigma_{\text{train}} / \sqrt{n}}$$
- `model_feedback_total` (Counter, labels: `outcome`): Ground truth feedback (`correct` vs `incorrect`).
- `up{job="ml_api"}` (Gauge): Scrape reachability status.

### Key PromQL Expressions
- **Total Requests**: `sum(ml_prediction_requests_total)`
- **Request Rate (req/s)**: `sum(rate(ml_prediction_requests_total[5m]))`
- **Average Latency (s)**: `sum(rate(ml_prediction_latency_seconds_sum[5m])) / sum(rate(ml_prediction_latency_seconds_count[5m]))`
- **P95 Latency (s)**: `histogram_quantile(0.95, sum by (le) (rate(ml_prediction_latency_seconds_bucket[5m])))`
- **Error Rate (%)**: `sum(rate(ml_prediction_errors_total[5m])) / clamp_min(sum(rate(ml_prediction_requests_total[5m])), 1e-9)`
- **Service Availability**: `up{job="ml_api"}` (Mapped to 1=UP, 0=DOWN)
- **Rolling Accuracy (5m)**: `sum(rate(model_feedback_total{outcome="correct"}[5m])) / clamp_min(sum(rate(model_feedback_total[5m])), 1e-9)`

---

## 7. Dynamic Alerting Rules (`prometheus/alert_rules.yml`)

1. **ModelServiceDown**: `up{job="ml_api"} == 0` for `30s` (Severity: `critical`)
2. **HighErrorRate**: Error ratio > `5%` for `1m` (Severity: `warning`)
3. **HighP95Latency**: P95 latency > `0.5s` for `2m` (Severity: `warning`)
4. **FeatureDriftDetected**: `abs(model_feature_drift_zscore) > 3` for `1m` (Severity: `warning`)
5. **LowModelAccuracy**: Ground truth accuracy < `85%` for `2m` (Severity: `critical`)

---

## 8. Stopping the Stack

To cleanly terminate all running containers and free all ports:
```bash
docker compose down
```
To remove persistent container volumes:
```bash
docker compose down -v
```
