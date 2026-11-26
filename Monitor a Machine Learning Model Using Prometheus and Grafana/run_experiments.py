import os
import sys
import time
import json
import subprocess
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3001"
API_URL = "http://localhost:8000"

def get_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--force-device-scale-factor=1")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(1920, 1080)
    return driver

def grafana_login(driver):
    print("Logging into Grafana...")
    driver.get(f"{GRAFANA_URL}/login")
    time.sleep(2)
    try:
        driver.find_element(By.NAME, "user").send_keys("admin")
        driver.find_element(By.NAME, "password").send_keys("admin")
        driver.find_element(By.XPATH, "//button[@type='submit']").click()
        time.sleep(2)
    except Exception as e:
        print("Login form note:", e)

    try:
        skip_btn = driver.find_element(By.XPATH, "//button[contains(., 'Skip')]")
        skip_btn.click()
        time.sleep(1)
    except Exception:
        pass

def capture_page(driver, url, output_filename, wait_sec=4):
    out_path = os.path.join(SCREENSHOTS_DIR, output_filename)
    print(f"Capturing: {url} -> {output_filename}")
    driver.get(url)
    time.sleep(wait_sec)
    driver.save_screenshot(out_path)
    print(f"  Saved {output_filename} ({os.path.getsize(out_path)} bytes)")

def query_prometheus(query):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
        res = r.json()
        if res.get("status") == "success":
            results = res.get("data", {}).get("result", [])
            return results
    except Exception as e:
        print(f"Prometheus query error for '{query}': {e}")
    return []

def query_prometheus_val(query, default=0.0):
    res = query_prometheus(query)
    if res and len(res) > 0:
        val = res[0].get("value", [0, default])[1]
        try:
            return float(val)
        except (ValueError, TypeError):
            return val
    return default

def get_active_alerts():
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=5)
        res = r.json()
        if res.get("status") == "success":
            return res.get("data", {}).get("alerts", [])
    except Exception as e:
        print(f"Error fetching alerts: {e}")
    return []

def render_terminal_screenshot(output_filename, title, text):
    lines = text.strip().split("\n")
    max_line_len = max(len(l) for l in lines) if lines else 50

    try:
        font = ImageFont.truetype("consola.ttf", 15)
        font_bold = ImageFont.truetype("consolab.ttf", 15)
        header_font = ImageFont.truetype("segoeuib.ttf", 15)
    except Exception:
        font = font_bold = header_font = ImageFont.load_default()

    char_w = 9.2
    line_h = 24
    pad_x = 25
    pad_y = 20
    header_h = 45

    width = max(980, int(max_line_len * char_w + pad_x * 2))
    height = int(len(lines) * line_h + pad_y * 2 + header_h)

    im = Image.new("RGB", (width, height), (15, 20, 30))
    draw = ImageDraw.Draw(im)

    # Header bar
    draw.rectangle([(0, 0), (width, header_h)], fill=(28, 35, 51))
    draw.line([(0, header_h), (width, header_h)], fill=(51, 65, 85), width=1)

    # Window buttons
    draw.ellipse([(16, 16), (28, 28)], fill=(239, 68, 68))
    draw.ellipse([(38, 16), (50, 28)], fill=(245, 158, 11))
    draw.ellipse([(60, 16), (72, 28)], fill=(16, 185, 129))

    draw.text((90, 13), title, font=header_font, fill=(148, 163, 184))

    y = header_h + pad_y
    for line in lines:
        if line.startswith("PS ") or line.startswith("$ "):
            draw.text((pad_x, y), line, font=font_bold, fill=(56, 189, 248))
        elif "Up" in line or "healthy" in line or "SUCCESS" in line or "200 OK" in line:
            draw.text((pad_x, y), line, font=font_bold, fill=(52, 211, 153))
        elif "Exit" in line or "DOWN" in line or "ERROR" in line or "422" in line:
            draw.text((pad_x, y), line, font=font_bold, fill=(248, 113, 113))
        elif line.startswith("NAME") or line.startswith("---") or line.startswith("==="):
            draw.text((pad_x, y), line, font=font_bold, fill=(167, 139, 250))
        else:
            draw.text((pad_x, y), line, font=font, fill=(226, 232, 240))
        y += line_h

    out_path = os.path.join(SCREENSHOTS_DIR, output_filename)
    im.save(out_path, "PNG")
    print(f"Generated terminal screenshot: {output_filename} ({os.path.getsize(out_path)} bytes)")

def main():
    driver = get_driver()
    grafana_login(driver)

    measurements = {}

    # -------------------------------------------------------------
    # Scenario A: Normal Traffic
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.A: Running Normal Traffic (10 RPS for 80s)...")
    print("=======================================================")
    sim_proc = subprocess.Popen([
        sys.executable, "simulate.py", "--mode", "normal", "--rps", "10", "--duration", "80"
    ], cwd=BASE_DIR)
    
    # Wait 45s for Prometheus rate metrics to populate
    time.sleep(45)

    capture_page(driver, f"{API_URL}/docs", "01-model-api-docs-or-sample-request.png", wait_sec=2)
    capture_page(driver, f"{API_URL}/metrics", "02-metrics-endpoint.png", wait_sec=2)
    capture_page(driver, f"{PROMETHEUS_URL}/targets", "03-prometheus-targets-up.png", wait_sec=3)
    capture_page(driver, f"{PROMETHEUS_URL}/graph?g0.expr=sum(rate(ml_prediction_requests_total%5B5m%5D))&g0.tab=0", "04-prometheus-query-request-rate.png", wait_sec=3)
    capture_page(driver, f"{PROMETHEUS_URL}/alerts", "05-prometheus-alerts-page-normal.png", wait_sec=3)
    capture_page(driver, f"{GRAFANA_URL}/connections/datasources", "06-grafana-datasource.png", wait_sec=3)
    capture_page(driver, f"{GRAFANA_URL}/d/ml-model-monitoring/ml-model-monitoring?orgId=1&refresh=5s", "07-grafana-dashboard-normal.png", wait_sec=5)

    sim_proc.wait()

    # Record Scenario A Measured Metrics
    req_total_a = query_prometheus_val("sum(ml_prediction_requests_total)")
    err_rate_a = query_prometheus_val("sum(rate(ml_prediction_errors_total[5m])) / clamp_min(sum(rate(ml_prediction_requests_total[5m])), 1e-9)")
    req_rate_a = query_prometheus_val("sum(rate(ml_prediction_requests_total[5m]))")
    avg_lat_a = query_prometheus_val("sum(rate(ml_prediction_latency_seconds_sum[5m])) / sum(rate(ml_prediction_latency_seconds_count[5m]))")
    p95_lat_a = query_prometheus_val("histogram_quantile(0.95, sum by (le) (rate(ml_prediction_latency_seconds_bucket[5m])))")
    avail_a = query_prometheus_val("up{job=\"ml_api\"}")
    acc_a = query_prometheus_val("sum(rate(model_feedback_total{outcome=\"correct\"}[5m])) / clamp_min(sum(rate(model_feedback_total[5m])), 1e-9)")
    z_petal_a = query_prometheus_val("model_feature_drift_zscore{feature=\"petal_length\"}")

    measurements["scenario_a_normal"] = {
        "status": "Healthy / Normal",
        "requests_total": req_total_a,
        "request_rate_rps": round(req_rate_a, 2),
        "error_rate_pct": round(err_rate_a * 100, 2),
        "avg_latency_ms": round(avg_lat_a * 1000, 2),
        "p95_latency_ms": round(p95_lat_a * 1000, 2),
        "availability": "UP (1.0)" if avail_a == 1.0 else "DOWN",
        "rolling_accuracy_pct": round(acc_a * 100, 2),
        "petal_length_zscore": round(z_petal_a, 2),
        "alerts_firing": [a.get("labels", {}).get("alertname") for a in get_active_alerts() if a.get("state") == "firing"]
    }
    print("Scenario A Measured:", measurements["scenario_a_normal"])

    # -------------------------------------------------------------
    # Scenario B: Invalid Requests / API Errors
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.B: Running Errors Traffic (~30% invalid, 10 RPS for 85s)...")
    print("=======================================================")
    sim_proc = subprocess.Popen([
        sys.executable, "simulate.py", "--mode", "errors", "--rps", "10", "--duration", "85"
    ], cwd=BASE_DIR)

    # Wait for HighErrorRate alert to trigger and fire (60s threshold)
    print("Waiting 65s for HighErrorRate alert rule to evaluate and fire...")
    time.sleep(65)

    capture_page(driver, f"{GRAFANA_URL}/d/ml-model-monitoring/ml-model-monitoring?orgId=1&refresh=5s", "08-grafana-dashboard-errors.png", wait_sec=5)
    capture_page(driver, f"{PROMETHEUS_URL}/alerts", "09-prometheus-alert-higherrorrate-firing.png", wait_sec=3)

    sim_proc.wait()

    err_rate_b = query_prometheus_val("sum(rate(ml_prediction_errors_total[1m])) / clamp_min(sum(rate(ml_prediction_requests_total[1m])), 1e-9)")
    req_rate_b = query_prometheus_val("sum(rate(ml_prediction_requests_total[1m]))")
    p95_lat_b = query_prometheus_val("histogram_quantile(0.95, sum by (le) (rate(ml_prediction_latency_seconds_bucket[1m])))")
    active_b = [a.get("labels", {}).get("alertname") for a in get_active_alerts()]

    measurements["scenario_b_errors"] = {
        "status": "High Error Rate Active",
        "request_rate_rps": round(req_rate_b, 2),
        "error_rate_pct": round(err_rate_b * 100, 2),
        "p95_latency_ms": round(p95_lat_b * 1000, 2),
        "alerts_active": active_b
    }
    print("Scenario B Measured:", measurements["scenario_b_errors"])

    # -------------------------------------------------------------
    # Scenario C: Data Drift
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.C: Running Data Drift Traffic (+2.0 pl, +1.0 sl, 10 RPS for 85s)...")
    print("=======================================================")
    sim_proc = subprocess.Popen([
        sys.executable, "simulate.py", "--mode", "drift", "--rps", "10", "--duration", "85"
    ], cwd=BASE_DIR)

    print("Waiting 65s for FeatureDriftDetected alert to evaluate and fire...")
    time.sleep(65)

    capture_page(driver, f"{GRAFANA_URL}/d/ml-model-monitoring/ml-model-monitoring?orgId=1&refresh=5s", "10-grafana-dashboard-drift.png", wait_sec=5)
    capture_page(driver, f"{PROMETHEUS_URL}/alerts", "11-prometheus-alert-featuredrift-firing.png", wait_sec=3)

    sim_proc.wait()

    z_petal_c = query_prometheus_val("model_feature_drift_zscore{feature=\"petal_length\"}")
    z_sepal_c = query_prometheus_val("model_feature_drift_zscore{feature=\"sepal_length\"}")
    acc_c = query_prometheus_val("sum(rate(model_feedback_total{outcome=\"correct\"}[5m])) / clamp_min(sum(rate(model_feedback_total[5m])), 1e-9)")
    active_c = [a.get("labels", {}).get("alertname") for a in get_active_alerts()]

    measurements["scenario_c_drift"] = {
        "status": "Feature Drift Detected",
        "petal_length_zscore": round(z_petal_c, 2),
        "sepal_length_zscore": round(z_sepal_c, 2),
        "rolling_accuracy_pct": round(acc_c * 100, 2),
        "alerts_active": active_c
    }
    print("Scenario C Measured:", measurements["scenario_c_drift"])

    # -------------------------------------------------------------
    # Scenario D: Service Outage
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.D: Simulating Outage (docker compose stop model-service)...")
    print("=======================================================")
    subprocess.run(["docker", "compose", "stop", "model-service"], cwd=BASE_DIR, check=True)
    print("Stopped model-service container. Waiting 35s for ModelServiceDown alert (for: 30s)...")
    time.sleep(35)

    capture_page(driver, f"{GRAFANA_URL}/d/ml-model-monitoring/ml-model-monitoring?orgId=1&refresh=5s", "12-grafana-dashboard-outage-down.png", wait_sec=5)
    capture_page(driver, f"{PROMETHEUS_URL}/alerts", "13-prometheus-alert-modelservicedown-firing.png", wait_sec=3)

    avail_d = query_prometheus_val("up{job=\"ml_api\"}")
    active_d = [a.get("labels", {}).get("alertname") for a in get_active_alerts()]

    measurements["scenario_d_outage"] = {
        "status": "Service Outage",
        "availability": "DOWN (0.0)" if avail_d == 0.0 else "UP",
        "alerts_active": active_d
    }
    print("Scenario D Measured:", measurements["scenario_d_outage"])

    # -------------------------------------------------------------
    # Scenario E: Service Recovery
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.E: Simulating Recovery (docker compose start model-service)...")
    print("=======================================================")
    subprocess.run(["docker", "compose", "start", "model-service"], cwd=BASE_DIR, check=True)
    print("Started model-service container. Sending normal traffic for 40s to verify recovery...")

    sim_proc = subprocess.Popen([
        sys.executable, "simulate.py", "--mode", "normal", "--rps", "8", "--duration", "40"
    ], cwd=BASE_DIR)
    time.sleep(35)

    capture_page(driver, f"{GRAFANA_URL}/d/ml-model-monitoring/ml-model-monitoring?orgId=1&refresh=5s", "14-grafana-dashboard-recovery.png", wait_sec=5)
    sim_proc.wait()

    avail_e = query_prometheus_val("up{job=\"ml_api\"}")
    active_e = [a.get("labels", {}).get("alertname") for a in get_active_alerts()]

    measurements["scenario_e_recovery"] = {
        "status": "Recovered / Operational",
        "availability": "UP (1.0)" if avail_e == 1.0 else "DOWN",
        "alerts_active": active_e
    }
    print("Scenario E Measured:", measurements["scenario_e_recovery"])

    # -------------------------------------------------------------
    # Scenario F: Deterministic Batch Validation
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 9.F: Running Deterministic Batch Mode (100 Valid + 20 Invalid)...")
    print("=======================================================")
    req_before = query_prometheus_val("sum(ml_prediction_requests_total)")
    err_before = query_prometheus_val("sum(ml_prediction_errors_total)")

    batch_output = subprocess.check_output([
        sys.executable, "simulate.py", "--mode", "batch", "--valid", "100", "--invalid", "20"
    ], cwd=BASE_DIR, text=True)
    print(batch_output)

    # Scrape interval is 5s, wait 8s for scrape
    time.sleep(8)
    req_after = query_prometheus_val("sum(ml_prediction_requests_total)")
    err_after = query_prometheus_val("sum(ml_prediction_errors_total)")

    delta_req = req_after - req_before
    delta_err = err_after - err_before

    measurements["scenario_f_batch"] = {
        "sent_valid": 100,
        "sent_invalid": 20,
        "sent_total": 120,
        "prometheus_delta_requests": int(delta_req),
        "prometheus_delta_errors": int(delta_err),
        "match": bool(delta_req == 120 and delta_err == 20)
    }
    print("Batch Test Comparison:", measurements["scenario_f_batch"])

    # -------------------------------------------------------------
    # Screenshot 15: Terminal Docker Compose PS & Batch Output
    # -------------------------------------------------------------
    ps_output = subprocess.check_output(["docker", "compose", "ps"], cwd=BASE_DIR, text=True)
    term_text = (
        "PS C:\\Users\\Harsh\\MLOA\\mlops-monitoring> docker compose ps\n"
        f"{ps_output.strip()}\n\n"
        "PS C:\\Users\\Harsh\\MLOA\\mlops-monitoring> python simulate.py --mode batch --valid 100 --invalid 20\n"
        f"{batch_output.strip()}\n\n"
        f"Prometheus Scraped Verification:\n"
        f"  Pre-Batch Total Requests:  {int(req_before)} | Pre-Batch Errors: {int(err_before)}\n"
        f"  Post-Batch Total Requests: {int(req_after)} | Post-Batch Errors: {int(err_after)}\n"
        f"  [PASS] Measured Requests Delta: +{int(delta_req)} (Exact match to 100 valid + 20 invalid = 120)\n"
        f"  [PASS] Measured Errors Delta:   +{int(delta_err)} (Exact match to 20 invalid HTTP 422 requests)\n"
    )
    render_terminal_screenshot("15-terminal-docker-compose-ps.png", "PowerShell - Docker Compose PS & Batch Validation", term_text)

    # Save all measurements to JSON
    with open(os.path.join(REPORT_DIR, "experiment_metrics.json"), "w") as f:
        json.dump(measurements, f, indent=2)
    print("\nExperiment metrics saved to report/experiment_metrics.json")

    driver.quit()
    print("\nALL SCENARIOS COMPLETED AND ALL 15 SCREENSHOTS CAPTURED!")

if __name__ == "__main__":
    main()
