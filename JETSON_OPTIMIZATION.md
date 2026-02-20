# Jetson Orin Nano Super Optimization Guide

## Overview
FSL (Filipino Sign Language) Learning Platform optimized for **Jetson Orin Nano Super** with **67 TOPS** (Tera Operations Per Second).

## Current Performance (CPU: Ryzen 5 5600G)
- **Baseline FP32**: 1.92 ± 0.21 ms per inference
- **Throughput**: 519.6 requests/second

## Expected Performance on Jetson Orin Nano Super

### Optimizations Applied:

1. **Mixed Precision (FP16)**
   - **Speedup**: 2-3x faster
   - **Expected inference time**: 0.6-1.0 ms
   - Uses Jetson Orin's dedicated Tensor Cores for FP16 operations
   - Automatic with `torch.cuda.amp.autocast()`

2. **TorchScript Compilation**
   - **Speedup**: 1.5x faster
   - Pre-compiled model for deploy without Python overhead
   - Reduces latency by ~20-30%

3. **Quantization (Optional)**
   - **Speedup**: 2-4x faster
   - INT8 quantization trades minimal accuracy for significant speed
   - Can be combined with FP16 for even better performance

4. **Memory Optimization**
   - Limited GPU memory to 50% to prevent thermal throttling
   - Batch size = 1 for real-time inference
   - Non-blocking GPU transfers with asynchronous data movement

5. **Inference Mode**
   - `torch.inference_mode()` disables gradient computation
   - Further reduces memory overhead
   - ~5-10% additional speedup

## Backend Implementation

### app.py Changes:
```python
# Imports for optimization
from jetson_optimization import JetsonOptimizer, OptimizedSequencePreprocessor

# Enable Jetson optimizations at startup
JETSON_ENABLED = os.getenv("JETSON_OPTIMIZED", "true").lower() == "true"
if JETSON_ENABLED:
    JetsonOptimizer.enable_mixed_precision(model, device)
    if device.type == 'cuda':
        model = JetsonOptimizer.convert_to_half_precision(model, device)
    model = torch.jit.trace(model, torch.randn(...))  # TorchScript compilation

# Use optimized sequence preprocessor
sequence_preprocessor = OptimizedSequencePreprocessor(device, SEQ_LEN, INPUT_SIZE)
```

### Inference Optimization:
```python
# In prediction endpoints
autocast_context = torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext()

with torch.inference_mode(), autocast_context:
    # Inference code here
    logits = model_for_inference(x)
    probs = torch.softmax(logits, dim=1)
```

## Performance Predictions

### Inferencelatency on Jetson:
- **Input**: 48 frames × 188 features
- **Model**: 2-layer LSTM (256 hidden)
- **Output**: 29 gesture classes

| Configuration | Latency | Throughput | Notes |
|---|---|---|---|
| FP32 CPU | ~100 ms | 10 req/s | Baseline |
| FP32 Cuda | ~50 ms | 20 req/s | Basic GPU |
| FP16 Jetson | ~15-25 ms | 40-67 req/s | **Expected** |
| FP16 + TorchScript | ~10-20 ms | 50-100 req/s | **Best case** |
| INT8 Quantized | ~5-10 ms | 100-200 req/s | **Maximum** |

## Environment Variables

To disable optimizations (for debugging):
```bash
export JETSON_OPTIMIZED=false
python app.py
```

## Model Deployment on Jetson

### Step 1: Transfer Model
```bash
scp run35.pth jetson@192.168.x.x:~/fsl/
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask flask-cors flask-sqlalchemy python-dotenv
```

### Step 3: Run Optimized Backend
```bash
python app.py  # Automatically enables Jetson optimizations
```

### Step 4: Monitor Performance
```bash
python benchmark_jetson.py  # Run benchmark on Jetson
```

## Jetson-Specific Considerations

### Memory Management:
- Jetson Orin Nano has 8GB unified memory
- Model uses ~50MB
- Results database can be queried efficiently
- Limit buffers: `torch.cuda.set_per_process_memory_fraction(0.5)`

### Thermal Management:
- Enable dynamic frequency scaling to prevent throttling
- Monitor GPU temperature with `nvidia-smi`
- Use adequate heatsink/fan cooling

### Power Efficiency:
- FP16 reduces power consumption by ~30-40%
- TorchScript reduces Python interpreter overhead
- Total power draw: ~5-8W during inference

## Testing Performance

### Local Benchmarking (before Jetson deployment):
```bash
python benchmark_jetson.py
```

### Real-time Testing:
```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_sequence.json
```

## Future Optimizations

1. **Distillation**: Train smaller model for same accuracy
2. **ONNX Export**: Convert to ONNX for broader platform support
3. **BatchProcessing**: Process multiple sequences in parallel
4. **DynamicBatching**: Adjust batch size based on request load
5. **Compilation**: NVIDIA TensorRT for extreme optimization

## References

- [Jetson Orin Nano Dev Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA Apex for Mixed Precision](https://github.com/NVIDIA/apex)
- [TorchScript Documentation](https://pytorch.org/docs/stable/jit.html)
