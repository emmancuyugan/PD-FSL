"""
Activity Mode Diagnostics & GPU Verification
Tests Activity Mode functionality and GPU usage
"""

import torch
import requests
import json
import time
import numpy as np
from gpu_profiler import JetsonDiagnostics, GPUProfiler, AdvancedOptimizer

def test_backend_connectivity():
    """Test if backend is reachable"""
    print("\n" + "="*60)
    print("BACKEND CONNECTIVITY TEST")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:5000/ping", timeout=5)
        if response.status_code == 200:
            print("✓ Backend is reachable")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot reach backend: {e}")
        print("  Make sure app.py is running with: python app.py")
        return False


def test_prediction_endpoint(mode="predict"):
    """Test prediction endpoint with dummy data"""
    print("\n" + "="*60)
    print(f"TESTING /{mode} ENDPOINT")
    print("="*60)
    
    # Create dummy sequence (48 frames × 188 features)
    dummy_sequence = np.random.randn(48, 188).tolist()
    
    payload = {
        "sequence": dummy_sequence,
        "source": "activity"
    }
    
    try:
        url = f"http://localhost:5000/{mode}"
        print(f"POST {url}")
        print(f"Payload size: {len(json.dumps(payload)) / 1024:.2f} KB")
        
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=30)
        latency = (time.perf_counter() - start) * 1000
        
        print(f"Request latency: {latency:.2f} ms")
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction successful!")
            print(f"  Prediction: {result.get('prediction', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 'N/A'):.2%}")
            return True
        else:
            print(f"✗ Request failed: {response.status_code}")
            print(f"  Error: {response.text[:200]}")
            return False
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out (>30s)")
        print("  Inference might be too slow - check GPU")
        return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False


def test_gpu_inference():
    """Test local GPU inference"""
    print("\n" + "="*60)
    print("LOCAL GPU INFERENCE TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == 'cpu':
        print("⚠ Running on CPU - GPU not available for local test")
        return True
    
    try:
        from model import ModifiedLSTM
        
        model = ModifiedLSTM(188, 256, 2, 29, dropout=0.35, use_layernorm=True).to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 48, 188, device=device)
        
        # Profile inference
        times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.inference_mode():
                _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000
        print(f"✓ GPU inference working")
        print(f"  Average latency: {avg_time:.2f} ms")
        print(f"  Throughput: {1000/avg_time:.1f} req/sec")
        return True
    except Exception as e:
        print(f"✗ GPU inference failed: {e}")
        return False


def test_activity_mode_flow():
    """Test complete Activity Mode flow"""
    print("\n" + "="*60)
    print("ACTIVITY MODE FLOW TEST")
    print("="*60)
    
    print("\n1. Testing backend connectivity...")
    if not test_backend_connectivity():
        print("✗ Cannot proceed without backend")
        return False
    
    print("\n2. Testing /predict endpoint...")
    if not test_prediction_endpoint("predict"):
        print("⚠ /predict endpoint failed")
        print("  Activity Mode will not work")
        return False
    
    print("\n3. Testing /api/results endpoint...")
    try:
        response = requests.get("http://localhost:5000/api/results", timeout=5)
        if response.status_code in [200, 401]:  # 401 is expected if not logged in
            print("✓ /api/results endpoint is accessible")
        else:
            print(f"⚠ /api/results returned {response.status_code}")
    except Exception as e:
        print(f"⚠ /api/results test failed: {e}")
    
    print("\n✓ Activity Mode should work!")
    return True


def print_gpu_verification_report():
    """Print comprehensive GPU verification report"""
    print("\n" + "="*60)
    print("COMPLETE GPU VERIFICATION REPORT")
    print("="*60)
    
    # Hardware diagnostics
    JetsonDiagnostics.check_hardware()
    
    # GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiler = GPUProfiler(device)
    profiler.print_cuda_info()
    
    # Test inference
    print("\nTesting local inference...")
    test_gpu_inference()
    
    # Performance estimates
    JetsonDiagnostics.estimate_jetson_performance()
    
    print("\n" + "="*60)
    print("END OF REPORT")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    print("\n" + "🔍 FSL ACTIVITY MODE & GPU DIAGNOSTICS 🔍".center(60))
    
    if len(sys.argv) > 1:
        test = sys.argv[1]
        if test == "backend":
            test_backend_connectivity()
        elif test == "predict":
            test_prediction_endpoint("predict")
        elif test == "activity":
            test_activity_mode_flow()
        elif test == "gpu":
            print_gpu_verification_report()
        else:
            print(f"Unknown test: {test}")
            print("\nAvailable tests:")
            print("  - backend    : Test backend connectivity")
            print("  - predict    : Test /predict endpoint")
            print("  - activity   : Test Activity Mode flow")
            print("  - gpu        : Full GPU verification report")
    else:
        # Run all tests
        print("\nRunning all diagnostics...\n")
        
        test_backend_connectivity()
        test_prediction_endpoint("predict")
        test_activity_mode_flow()
        print_gpu_verification_report()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("If tests passed:")
        print("  ✓ Activity Mode should work")
        print("  ✓ GPU is being utilized if available")
        print("  ✓ Inference latency within expectations")
        print("\nIf tests failed:")
        print("  1. Check backend is running: python app.py")
        print("  2. Verify model file: run35.pth exists")
        print("  3. Check GPU: torch.cuda.is_available()")
        print("  4. Review app.py logs for errors")
        print("="*60)
