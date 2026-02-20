#!/usr/bin/env python3
"""Test the /predict endpoint directly."""

import requests
import json
import numpy as np
import time

# Test data - create a valid 48x188 sequence
sequence = np.random.randn(48, 188).astype(np.float32).tolist()

payload = {
    "sequence": sequence
}

print("[TEST] Testing /predict endpoint...")
print(f"[TEST] Payload shape: 48x188 = {48*188} features")

try:
    response = requests.post(
        "http://localhost:5000/predict",
        json=payload,
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"[TEST] ✓ Request successful!")
        print(f"[TEST] Response: {json.dumps(result, indent=2)}")
    else:
        print(f"[TEST] ✗ Request failed with status {response.status_code}")
        print(f"[TEST] Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print(f"[TEST] ✗ Connection error - is the Flask app running?")
    print(f"[TEST] Try running: python app.py")
except Exception as e:
    print(f"[TEST] ✗ Error: {e}")
