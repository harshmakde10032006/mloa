"""
Unit test for lambda_function.py local contract and validation.
"""

import json
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import lambda_function

def run_tests():
    print("Testing Lambda function validation...")

    # Test 1: Missing body
    res = lambda_function.lambda_handler({}, None)
    assert res["statusCode"] == 400
    print("Test 1 Passed: Missing body returns 400")

    # Test 2: Invalid feature count (29 features instead of 30)
    event_29 = {"body": json.dumps({"features": [1.0] * 29})}
    res = lambda_function.lambda_handler(event_29, None)
    assert res["statusCode"] == 400
    assert "Expected 30 features" in json.loads(res["body"])["error"]
    print("Test 2 Passed: 29 features returns 400 Expected 30 features")

    # Test 3: Non-numeric feature
    bad_features = [1.0] * 29 + ["invalid_string"]
    event_bad = {"body": json.dumps({"features": bad_features})}
    res = lambda_function.lambda_handler(event_bad, None)
    assert res["statusCode"] == 400
    assert "numeric" in json.loads(res["body"])["error"]
    print("Test 3 Passed: Non-numeric feature returns 400")

    # Test 4: Mocked SageMaker invocation with valid 30 features
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps([1]).encode("utf-8")
    
    with patch.object(lambda_function.sagemaker_runtime, "invoke_endpoint", return_value={"Body": mock_body}):
        event_valid = {"body": json.dumps({"features": [1.0] * 30})}
        res = lambda_function.lambda_handler(event_valid, None)
        assert res["statusCode"] == 200
        body = json.loads(res["body"])
        assert body["prediction"] == "benign"
        assert body["class_id"] == 1
        print("Test 4 Passed: Valid 30 features returns 200 with prediction 'benign'")

    # Test 5: Malignant prediction mapping
    mock_body_0 = MagicMock()
    mock_body_0.read.return_value = json.dumps([0]).encode("utf-8")
    with patch.object(lambda_function.sagemaker_runtime, "invoke_endpoint", return_value={"Body": mock_body_0}):
        event_valid = {"body": json.dumps({"features": [1.0] * 30})}
        res = lambda_function.lambda_handler(event_valid, None)
        assert res["statusCode"] == 200
        body = json.loads(res["body"])
        assert body["prediction"] == "malignant"
        assert body["class_id"] == 0
        print("Test 5 Passed: Prediction 0 mapped to 'malignant'")

    print("\nALL LAMBDA CONTRACT TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
