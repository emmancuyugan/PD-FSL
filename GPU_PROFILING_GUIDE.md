# GPU Profiling & Activity Mode Debugging - What's New

## New Files Created

### 1. **gpu_profiler.py** (Production-ready GPU diagnostics)
   - `GPUProfiler`: Track inference latency, memory, throughput
   - `AdvancedOptimizer`: Enable Tensor Cores, cuBLASLt, memory optimization
   - `JetsonDiagnostics`: Check hardware, estimate performance
   
   Usage:
   ```python
   from gpu_profiler import JetsonDiagnostics, GPUProfiler
   
   # Check hardware
   JetsonDiagnostics.check_hardware()
   
   # Profile inference
   profiler = GPUProfiler(device)
   with profiler.profile_inference(label="predict"):
       output = model(input)
   print(profiler.get_summary())
   ```

### 2. **test_activity_mode.py** (Complete diagnostic test suite)
   - Tests backend connectivity
   - Tests `/predict` endpoint
   - Tests Activity Mode flow
   - Runs full GPU verification

   Usage:
   ```bash
   # Test specific component
   python test_activity_mode.py backend      # Check if backend is running
   python test_activity_mode.py predict      # Test prediction endpoint
   python test_activity_mode.py activity     # Full Activity Mode test
   python test_activity_mode.py gpu          # GPU verification report
   
   # Or run everything
   python test_activity_mode.py
   ```

### 3. **ACTIVITY_MODE_DEBUG.md** (Comprehensive troubleshooting guide)
   - Step-by-step debugging procedures
   - Performance verification
   - Common issues and solutions
   - GPU usage verification
   - Expected performance metrics

---

## Updates to Existing Files

### app.py
**Added:**
- GPU profiler initialization
- GPU profiling to all prediction endpoints
- Advanced GPU optimizations (Tensor Cores, cuBLASLt)
- Hardware diagnostics at startup
- Detailed GPU usage logging

**Lines changed:**
- Imports: Added gpu_profiler imports
- Init: Added JetsonDiagnostics.check_hardware()
- Jetson setup: Added AdvancedOptimizer calls
- /predict endpoint: Wrapped inference in profiler context

**Result:** Every prediction now logs:
```
[GPU] predict | Latency: 15.23ms | Avg: 15.45ms | Peak Memory: 245MB
```

---

## How to Verify Activity Mode & GPU Usage

### **Option 1: Quick Test (1 minute)**
```bash
# Terminal 1
python app.py

# Terminal 2
python test_activity_mode.py activity
```

Expected output:
```
✓ Backend is reachable
✓ Prediction successful!
✓ Activity Mode should work!
```

### **Option 2: GPU Verification (5 minutes)**
```bash
python test_activity_mode.py gpu
```

Will show:
- ✓ CUDA available: True
- ✓ Device: RTX 4090 / Jetson Orin Nano Super
- ✓ Compute Capability: 9.0
- ✓ Total Memory: 24 GB / 8 GB
- ✓ Optimizations enabled (Tensor Cores, cuBLASLt, etc.)
- Expected performance estimates for Jetson

### **Option 3: Real-time GPU Monitoring (Jetson)**
```bash
# Terminal 1
python app.py 2>&1 | grep -E "GPU|JETSON"

# Terminal 2
watch nvidia-smi

# Terminal 3
python test_activity_mode.py predict
```

Watch nvidia-smi for:
- ✓ GPU Memory increasing (100-300 MB)
- ✓ GPU Utilization at 80-100%
- ✓ Temperature rising (should stay < 60°C)

---

## If Activity Mode Doesn't Work

**Step 1: Check backend is running**
```bash
python app.py
# Look for: [INFO] Using device: cuda or [INFO] Using device: cpu
```

**Step 2: Check GPU is available**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Step 3: Test prediction endpoint directly**
```bash
python test_activity_mode.py predict
```

**Step 4: Check browser console**
Open your browser → F12 → Console
Look for any red errors when clicking "Start Activity!"

**Step 5: Full diagnostic**
```bash
python test_activity_mode.py
# Will test everything and show detailed report
```

---

## GPU Optimization Verification

### Verify These Are Enabled

When app.py starts, look for:
```
[JETSON] Initializing Jetson Orin Nano Super optimizations...
[GPU] TF32 precision enabled for Tensor Cores
[GPU] cuBLASLt enabled
[GPU] Memory fraction limited to 50% to prevent thermal throttling
```

### Check During Inference

Every `/predict` request logs:
```
[GPU] predict | Latency: 15.23ms | Avg: 15.45ms | Peak Memory: 245MB
```

If you see similar, GPU is being properly utilized!

### Performance Comparison

**Expected latency:**
- Dev PC (CPU): 1-2 ms
- Jetson FP32: 30-50 ms
- Jetson FP16 (current): 10-25 ms ← You are here
- Jetson FP16+TorchScript: 8-15 ms

If your latency is:
- Much lower: Might be cached/dummy data
- Much higher: GPU might not be used, check with nvidia-smi

---

## Advanced: Further Optimization

### Enable INT8 Quantization (2-4x speedup)
```python
# Add to app.py after model loads
import torch.quantization
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

Expected: 5-10 ms latency on Jetson Orin Nano Super

### Enable TorchScript Compilation (1.5x speedup)
```python
# Add to app.py
dummy_input = torch.randn(1, 48, 188, device=device)
model = torch.jit.trace(model, dummy_input)
```

Expected: 8-15 ms latency

### Reduce Model Complexity
```python
# Change HIDDEN_SIZE from 256 to 128
HIDDEN_SIZE = 128  # Smaller = faster but less accurate
model = ModifiedLSTM(INPUT_SIZE, 128, NUM_LAYERS, NUM_CLASSES)
```

Expected 20-30% speedup with minimal accuracy loss

---

## Summary

**New Tools:**
- `gpu_profiler.py` - Production-ready GPU diagnostics
- `test_activity_mode.py` - Complete test suite
- `ACTIVITY_MODE_DEBUG.md` - Troubleshooting guide

**Key Commands:**
```bash
# Verify Activity Mode works
python test_activity_mode.py activity

# Verify GPU is being used
python test_activity_mode.py gpu

# Monitor during runtime
python app.py 2>&1 | grep "GPU\|JETSON"
```

**Expected Results:**
- ✓ Activity Mode starts and records gestures
- ✓ Prediction latency < 25 ms (Jetson FP16)
- ✓ GPU memory usage > 100 MB
- ✓ GPU utilization 80-100% during inference

If you get these, optimization is working! 🎉

---

*For detailed troubleshooting, see ACTIVITY_MODE_DEBUG.md*
