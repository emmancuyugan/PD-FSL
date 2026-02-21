# Activity Mode Debugging & GPU Verification Guide

## Quick Troubleshooting

### Activity Mode Not Working?

**Step 1: Check if backend is running**
```bash
# Terminal 1: Start the backend
python app.py

# Terminal 2: Test backend connectivity
python test_activity_mode.py backend
```

Expected output:
```
✓ Backend is reachable
Response: {"message": "Backend is reachable ✅"}
```

**Step 2: Test prediction endpoint**
```bash
python test_activity_mode.py predict
```

Expected output:
```
✓ Prediction successful!
Prediction: One
Confidence: 95.23%
Request latency: 15.32 ms
```

**Step 3: Run full Activity Mode flow test**
```bash
python test_activity_mode.py activity
```

This will test:
- Backend connectivity
- `/predict` endpoint
- Activity Mode flow
- Results API

### If Tests Fail

#### Backend connectivity fails
```
✗ Cannot reach backend
```
**Solution**: Start backend in another terminal:
```bash
python app.py
```

#### Prediction endpoint returns error
```
✗ Request failed: 500
```
**Solutions**:
1. Check `run35.pth` exists in c:\FSL\PD-FSL\
2. Check app.py logs for Python errors
3. Verify PyTorch is installed: `python -c "import torch; print(torch.__version__)"`
4. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`

#### Activity Mode slow (> 30 seconds)
```
✗ Request timed out (>30s)
```
**Solutions**:
1. Check GPU is actually being used (see GPU Verification below)
2. Reduce inference complexity (see Optimization section)
3. Check if Jetson Orin is thermally throttling

---

## GPU Verification & Usage

### Check GPU Availability

```bash
python test_activity_mode.py gpu
```

This will print:
- Device information (NVIDIA GPU model, compute capability)
- CUDA version and cuDNN version
- Memory available
- Optimization flags enabled (TF32, cuBLASLt, etc.)
- Expected performance estimates for Jetson Orin Nano

### Verify GPU is Being Used During Inference

#### Option 1: Watch GPU usage in real-time (Jetson)
```bash
# Terminal 1: Run backend
python app.py

# Terminal 2: Monitor GPU
watch nvidia-smi

# Terminal 3: Trigger inference
python test_activity_mode.py predict
```

Look for:
- ✓ GPU Memory Usage increasing (should be 100-300 MB)
- ✓ GPU Utilization percentage increasing (should be 80-100%)
- ✓ Temperature increasing slightly (should stay < 60°C)

#### Option 2: Check logs during inference
```bash
python app.py 2>&1 | grep -E "GPU|CUDA|inference"
```

Look for messages like:
```
[GPU] predict | Latency: 15.23ms | Avg: 15.45ms | Peak Memory: 245MB
[JETSON] Model ready for inference on cuda:0
```

#### Option 3: Python direct check
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Allocated memory:", torch.cuda.memory_allocated() / 1024**2, "MB")
```

---

## Additional Optimizations

### For Slower Devices (Jetson Nano, not Super)

Add these environment variables before running:
```bash
# Reduce precision further
export JETSON_OPTIMIZED=true

# Limit memory
export CUDA_VISIBLE_DEVICES=0

# Run with lower timer settings
python app.py
```

### Enable Quantization (2-4x speedup)

Add to app.py after model loads:
```python
# Quantize model to INT8
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Profile Model Performance

```bash
# Test model latency on your hardware
python gpu_profiler.py
```

Output will show:
- Device information
- Expected performance on Jetson Orin
- CUDA capabilities

### Monitor Temperature & Thermal Throttling

```bash
# On Jetson Orin
watch -n 1 "nvidia-smi | grep -E 'Temp|Power'"

# If temperature > 65°C, add cooling
# If seeing "Thermal Throttling", device is overheating
```

---

## Activity Mode Specific Debugging

### Issue: "Activity Mode doesn't start"

**Check browser console** (F12 → Console):
```javascript
// Should NOT see errors like:
// Fetch error
// Connection refused
// 404 Not Found
```

**Check app logs** (where app.py is running):
```bash
# Look for Activity Mode specific logs
[PREDICT] Input tensor shape: ...
[ACTIVITY] ...

# If no logs appear, the request isn't reaching backend
```

### Issue: "Predictions are incorrect"

1. **Check input dimensions**: Should be 48 frames × 188 features
2. **Verify model**: `run35.pth` should be the correct trained model
3. **Test with benchmark**:
   ```bash
   python benchmark_jetson.py
   ```
   This shows baseline accuracy on dummy data

### Issue: "Gestures not being detected"

1. Check MediaPipe is working (camera feed shows skeleton)
2. Check poses are being captured (progress bar fills)
3. Verify gesture database has your gestures defined

---

## Performance Metrics

### Expected Performance (Development PC)

```
Device: Ryzen 5 5600G (CPU)
Latency: 1.92 ms per inference
Throughput: 519.6 req/sec
```

### Expected Performance (Jetson Orin Nano Super)

```
Device: Jetson Orin Nano Super (GPU)
FP32 Baseline:      30-50 ms, 20-33 req/sec
FP16 (Recommended): 10-25 ms, 40-100 req/sec ← Current setting
FP16 + TorchScript: 8-15 ms, 67-125 req/sec
INT8 Quantized:     5-10 ms, 100-200 req/sec
```

### If Performance is Different

If actual latency is significantly different from expected:

1. **Much slower than expected** (> 100 ms)
   - GPU might not be using (check with nvidia-smi)
   - Model might be on CPU (check logs)
   - Jetson might be thermally throttling

2. **Much faster than expected** (< 1 ms)
   - Might be cached response
   - Check input dimensions are correct
   - Verify model is actually running

---

## Comprehensive Test Suite

### Run all Activity Mode tests
```bash
python test_activity_mode.py activity
```

### Run all GPU tests
```bash
python test_activity_mode.py gpu
```

### Run specific backend test
```bash
python test_activity_mode.py backend
```

### View GPU profiling during runtime
```bash
python app.py 2>&1 | grep "GPU\|JETSON"
```

This will show every inference with latency and memory stats.

---

## When to Contact Support

Gather this information:
1. Output of: `python test_activity_mode.py gpu`
2. Output of: `python app.py` (first 50 lines showing CUDA/device info)
3. Output of: `python test_activity_mode.py activity`
4. Browser console errors (F12)
5. Hardware: Is it Jetson? Dev PC? What model?

With this info, we can diagnose the exact issue.

---

## Summary

To verify Activity Mode is working with GPU optimization:

```bash
# Terminal 1: Start backend with GPU monitoring
python app.py 2>&1 | grep -E "GPU|JETSON|CUDA"

# Terminal 2: Run diagnostics
python test_activity_mode.py activity

# Expected output:
# ✓ All tests pass
# ✓ Latency < 30 ms (Jetson) or < 2ms (Dev PC)
# ✓ GPU memory usage > 100 MB if using GPU
```

If all tests pass and Activity Mode works, you're good! You can now optimize further with INT8 quantization or TorchScript compilation if needed.
