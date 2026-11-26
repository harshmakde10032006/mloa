import argparse
import json
import random
import time
import requests
import numpy as np

# Normal Iris distributions from training stats (approximate baselines)
# Setosa: sl ~ 5.0, sw ~ 3.4, pl ~ 1.5, pw ~ 0.2
# Versicolor: sl ~ 5.9, sw ~ 2.8, pl ~ 4.3, pw ~ 1.3
# Virginica: sl ~ 6.6, sw ~ 3.0, pl ~ 5.5, pw ~ 2.0
IRIS_PROFILES = [
    {"class": "setosa", "sl": (5.0, 0.35), "sw": (3.4, 0.38), "pl": (1.46, 0.17), "pw": (0.24, 0.10)},
    {"class": "versicolor", "sl": (5.9, 0.51), "sw": (2.77, 0.31), "pl": (4.26, 0.47), "pw": (1.33, 0.20)},
    {"class": "virginica", "sl": (6.59, 0.63), "sw": (2.97, 0.32), "pl": (5.55, 0.55), "pw": (2.03, 0.27)},
]

def generate_sample(drift=False):
    profile = random.choice(IRIS_PROFILES)
    sl = np.random.normal(profile["sl"][0], profile["sl"][1])
    sw = np.random.normal(profile["sw"][0], profile["sw"][1])
    pl = np.random.normal(profile["pl"][0], profile["pl"][1])
    pw = np.random.normal(profile["pw"][0], profile["pw"][1])

    if drift:
        # Shift petal_length +2.0 and sepal_length +1.0
        pl += 2.0
        sl += 1.0

    return {
        "sepal_length": max(0.1, round(float(sl), 2)),
        "sepal_width": max(0.1, round(float(sw), 2)),
        "petal_length": max(0.1, round(float(pl), 2)),
        "petal_width": max(0.1, round(float(pw), 2)),
    }, profile["class"]

def generate_invalid_sample():
    choice = random.randint(0, 4)
    if choice == 0:
        # Missing required field
        return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    elif choice == 1:
        # Invalid data type (string instead of float)
        return {"sepal_length": "invalid_string", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    elif choice == 2:
        # Out of bounds value (negative)
        return {"sepal_length": -5.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    elif choice == 3:
        # Empty dictionary
        return {}
    else:
        # Extra extreme out of range
        return {"sepal_length": 150.0, "sepal_width": 25.0, "petal_length": 1.4, "petal_width": 0.2}

def run_batch(base_url, n_valid, n_invalid):
    predict_url = f"{base_url}/predict"
    print(f"--- Running Deterministic Batch Test ---")
    print(f"Target: {predict_url}")
    print(f"Sending exactly {n_valid} valid requests and {n_invalid} invalid requests...")

    valid_sent = 0
    valid_success = 0
    invalid_sent = 0
    invalid_422 = 0

    for i in range(n_valid):
        payload, _ = generate_sample(drift=False)
        try:
            resp = requests.post(predict_url, json=payload, timeout=2.0)
            valid_sent += 1
            if resp.status_code == 200:
                valid_success += 1
        except Exception as e:
            print(f"Connection error on valid request {i+1}: {e}")

    for j in range(n_invalid):
        payload = generate_invalid_sample()
        try:
            resp = requests.post(predict_url, json=payload, timeout=2.0)
            invalid_sent += 1
            if resp.status_code == 422:
                invalid_422 += 1
        except Exception as e:
            print(f"Connection error on invalid request {j+1}: {e}")

    print("\nBatch Test Results:")
    print(f"  Valid Requests Sent:   {valid_sent} (Successful 200s: {valid_success})")
    print(f"  Invalid Requests Sent: {invalid_sent} (Expected 422s: {invalid_422})")
    print(f"  Total Requests:        {valid_sent + invalid_sent}")
    print(f"  Total Errors (422):    {invalid_422}")
    print(f"----------------------------------------")

def run_simulation(mode, base_url, rps, duration):
    predict_url = f"{base_url}/predict"
    feedback_url = f"{base_url}/feedback"

    interval = 1.0 / max(rps, 0.1)
    start_time = time.time()
    end_time = start_time + duration if duration > 0 else float("inf")

    print(f"==================================================")
    print(f"Starting Traffic Simulator: Mode='{mode}'")
    print(f"Target: {base_url} | RPS: {rps} | Interval: {interval:.3f}s")
    if duration > 0:
        print(f"Duration: {duration} seconds (until {time.strftime('%H:%M:%S', time.localtime(end_time))})")
    else:
        print("Duration: Infinite (Ctrl+C to stop)")
    print(f"==================================================")

    req_count = 0
    err_count = 0
    fb_count = 0

    while time.time() < end_time:
        t0 = time.time()
        is_error = False

        if mode == "errors":
            # ~30% invalid payloads
            if random.random() < 0.30:
                payload = generate_invalid_sample()
                is_error = True
            else:
                payload, actual_class = generate_sample(drift=False)
        elif mode == "drift":
            payload, actual_class = generate_sample(drift=True)
        else: # normal
            payload, actual_class = generate_sample(drift=False)

        # Send prediction request with outage-friendly retry/catch
        try:
            resp = requests.post(predict_url, json=payload, timeout=2.0)
            req_count += 1
            if resp.status_code >= 400:
                err_count += 1
            elif resp.status_code == 200 and not is_error:
                data = resp.json()
                predicted = data.get("prediction")

                # Handle feedback simulation
                # Normal: send delayed feedback ~30% with ~95% correctness
                # Drift: send delayed feedback ~30% with ~60% correctness
                if random.random() < 0.30:
                    fb_count += 1
                    if mode == "drift":
                        outcome = "correct" if random.random() < 0.60 else "incorrect"
                    else:
                        outcome = "correct" if random.random() < 0.95 else "incorrect"

                    try:
                        requests.post(
                            feedback_url,
                            json={"prediction": predicted, "actual": actual_class, "outcome": outcome},
                            timeout=2.0
                        )
                    except Exception:
                        pass
        except requests.exceptions.RequestException as e:
            # Outage-friendly: ignore connection errors during downtime simulation
            print(f"[{time.strftime('%X')}] Connection to {predict_url} failed (likely service outage): {e.__class__.__name__}")

        if req_count % 50 == 0 and req_count > 0:
            print(f"[{time.strftime('%X')}] Requests: {req_count} | Errors: {err_count} ({err_count/max(req_count,1)*100:.1f}%) | Feedbacks: {fb_count}")

        elapsed = time.time() - t0
        sleep_dur = max(0.0, interval - elapsed)
        time.sleep(sleep_dur)

    print(f"\nSimulation completed. Sent {req_count} requests with {err_count} errors.")

def main():
    parser = argparse.ArgumentParser(description="ML Model Traffic Simulator")
    parser.add_argument("--mode", choices=["normal", "drift", "errors", "batch"], default="normal")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0 for infinite)")
    parser.add_argument("--valid", type=int, default=100, help="Valid count for batch mode")
    parser.add_argument("--invalid", type=int, default=20, help="Invalid count for batch mode")

    args = parser.parse_args()

    if args.mode == "batch":
        run_batch(args.url, args.valid, args.invalid)
    else:
        run_simulation(args.mode, args.url, args.rps, args.duration)

if __name__ == "__main__":
    main()
