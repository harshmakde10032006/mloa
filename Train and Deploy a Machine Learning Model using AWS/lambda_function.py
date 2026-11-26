"""
AWS Lambda Proxy Function for SageMaker Inference Endpoint
Runtime: Python 3.12
Contract:
- Reads ENDPOINT_NAME from environment variables.
- Parses event['body'] JSON: {"features": [30 numeric values]}
- Validates input: exactly 30 numeric features.
- Calls SageMaker Runtime invoke_endpoint (ContentType='text/csv', Accept='application/json').
- Maps prediction: 1 -> "benign", 0 -> "malignant".
- Returns HTTP 200 with prediction result.
- Returns HTTP 400 on invalid input.
- Returns HTTP 500 on unexpected runtime errors without leaking internal stack traces.
"""

import json
import os
import boto3
from botocore.exceptions import ClientError

# Read SageMaker endpoint name and region from environment variable
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "breast-cancer-rf-endpoint")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"))

# Initialize sagemaker-runtime client with explicit region fallback
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token",
}


def build_response(status_code, body):
    """Utility to build standard API Gateway proxy responses."""
    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    """AWS Lambda entry point for API Gateway POST /predict."""
    try:
        # Handle CORS preflight request
        http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method")
        if http_method == "OPTIONS":
            return build_response(200, {"message": "CORS preflight OK"})

        # 1. Parse request body
        raw_body = event.get("body")
        if not raw_body:
            return build_response(400, {"error": "Missing request body"})

        if isinstance(raw_body, str):
            try:
                payload = json.loads(raw_body)
            except Exception:
                return build_response(400, {"error": "Malformed JSON payload"})
        elif isinstance(raw_body, dict):
            payload = raw_body
        else:
            return build_response(400, {"error": "Invalid body type"})

        # 2. Extract features
        if "features" not in payload:
            return build_response(400, {"error": "Expected payload containing 'features' array"})

        features = payload["features"]
        if not isinstance(features, list):
            return build_response(400, {"error": "'features' must be a list of numbers"})

        # Validate feature count (Wisconsin Breast Cancer has 30 features)
        if len(features) != 30:
            return build_response(400, {"error": "Expected 30 features"})

        # Validate all items are numeric
        try:
            numeric_features = [float(x) for x in features]
        except (ValueError, TypeError):
            return build_response(400, {"error": "All 30 features must be numeric values"})

        # 3. Format payload as CSV string for SageMaker endpoint
        csv_payload = ",".join(str(val) for val in numeric_features)

        # 4. Invoke SageMaker real-time endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Accept="application/json",
            Body=csv_payload,
        )

        # 5. Parse endpoint response
        response_body = response["Body"].read().decode("utf-8")
        raw_pred = json.loads(response_body)

        # Handle list vs scalar output
        if isinstance(raw_pred, list):
            pred_val = int(raw_pred[0])
        else:
            pred_val = int(raw_pred)

        # Map binary label: 1 -> "benign", 0 -> "malignant"
        label_map = {1: "benign", 0: "malignant"}
        prediction_label = label_map.get(pred_val, "unknown")

        return build_response(
            200,
            {
                "prediction": prediction_label,
                "class_id": pred_val,
                "status": "success",
            },
        )

    except ClientError as e:
        # SageMaker invocation error
        error_code = e.response.get("Error", {}).get("Code", "ClientError")
        print(f"[ERROR] AWS ClientError: {error_code} - {str(e)}")
        return build_response(500, {"error": "Inference endpoint error"})

    except Exception as e:
        # Unexpected error (do not leak internal stack traces)
        print(f"[FATAL] Unexpected error in lambda_handler: {type(e).__name__} - {str(e)}")
        return build_response(500, {"error": "Internal server error"})
