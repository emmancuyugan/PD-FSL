# Jetson Orin Nano - Backend & System Configuration Guide

## app.py Optimizations

The Flask backend is already quite efficient, but here are recommendations for Jetson-specific tuning:

### 1. PyTorch GPU Acceleration
Currently configured to use GPU if available. The model loading is efficient. Consider:

```python
# In app.py - Optional GPU optimization
import torch

# Ensure we're using GPU when available (already done)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optional: Set GPU memory growth (prevents OOM)
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% GPU memory
    torch.cuda.empty_cache()  # Clear cache at startup
```

### 2. Model Optimization

Consider quantization for even better performance:

```python
# Optional: Quantize model for faster inference (some accuracy loss)
from torch.quantization import quantize_dynamic

if torch.cuda.is_available():
    # Convert to 8-bit for 2-4x faster inference
    quantized_model = quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
```

### 3. Request Handling Optimization

Current implementation is good, but ensure:

```python
# In app.run() - Optimize for single device
app.run(
    host="0.0.0.0",        # Allow local network access
    port=5000, 
    debug=False,           # Already disabled - good!
    use_reloader=False,    # Already disabled - good!
    threaded=True,         # Enable threading for concurrent requests
    processes=1            # Single process for GPU access
)
```

### 4. Database Optimization

The SQLAlchemy setup is fine. For Jetson, no additional tuning needed unless you see slowdown:

```python
# Optional: Add connection pooling tuning
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 10,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
}
```

## System-Level Configuration for Jetson Orin Nano

### Power & Performance Modes

Check current power mode:
```bash
sudo /usr/sbin/nvpmodel -m
# Output: NVIDIA Jetson Orin Nano Developer Kit Mode: <MODE>
```

Available modes:
- 0: 15W (default, recommended for web app)
- 1: 25W (higher performance, more power draw)

Switch to 25W mode for maximum performance:
```bash
sudo /usr/sbin/nvpmodel -m 1
sudo systemctl restart nvpmodel
```

Set maximum CPU frequency (optional):
```bash
# Check current frequencies
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies

# Set to maximum (auto-scales down as needed)
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### GPU Clock Optimization

Check available GPU clocks:
```bash
cat /sys/devices/virtual/thermal/cooling_device*/cur_state
```

For web app (recommended balanced):
```bash
# Balanced performance (don't force max, let thermal management work)
# System will auto-scale based on thermal limits
```

### Memory & Swap Configuration

Check available resources:
```bash
nvidia-smi  # GPU memory (typically 4GB-8GB)
free -h     # System RAM (typically 6GB-8GB)
swapon --show  # Check if swap is enabled
```

If swap is needed (not recommended but possible):
```bash
# Create 2GB swap file (only if needed, SD cards have limited writes)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Network & Flask Configuration

For optimal network handling:

```python
# In app.py - Optional TCP optimization
import socket

# Set TCP keep-alive for WebSocket stability
socket.setdefaulttimeout(60)  # 60 second timeout for connections
```

Enable HTTP/2 if using a production server:

```bash
# Install gunicorn with h2 support
pip install gunicorn[h2]

# Run with gunicorn instead of Flask dev server
gunicorn -b 0.0.0.0:5000 -w 1 --worker-class sync \
         -t 120 --access-logfile - app:app
```

### TensorRT Optimization (Advanced)

If you want to further optimize the PyTorch model:

```bash
# Install TensorRT
pip install tensorrt

# Convert LSTM model to TensorRT format for 2-3x faster inference
# (Advanced - requires model export to ONNX first)
```

## Monitoring on Jetson

### Real-time Performance Monitoring

Create a monitoring script:

```bash
#!/bin/bash
# monitor_jetson.sh

while true; do
    clear
    echo "==== Jetson Orin Nano Metrics ===="
    nvidia-smi
    echo ""
    echo "==== CPU Load ===="
    top -bn1 | grep "Cpu(s)"
    echo ""
    echo "==== Memory ===="
    free -h | grep Mem
    sleep 5
done
```

Run in separate terminal:
```bash
bash monitor_jetson.sh
```

### Application Performance Metrics

In Flask app logs:
```python
import logging
import psutil
import time

# Log resource usage periodically
def log_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"CPU: {cpu_percent}% | Memory: {memory.percent}% | Disk: {disk.percent}%")
```

## Recommended Settings for Jetson Orin Nano

```
Device: NVIDIA Jetson Orin Nano (8GB variant)
Power Mode: 15W (default) or 25W (high-perf)
GPU Memory: Auto (usually 2-4GB allocated)
CPU Frequency: Auto-scaling (balanced)
Flask Workers: 1 thread
Mediapipe Model: Lite (modelComplexity: 1)
Camera Resolution: 480x360
Frame Skip: 2-3
Expected FPS: 15-20
GPU Utilization: 40-60% (leaves headroom)
```

## Testing Performance

1. Open browser to device on local network:
   ```
   http://jetson-ip:5000
   ```

2. Monitor in separate SSH terminal:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. Test each page:
   - Click "Select & Practice" → Start capturing frames
   - Watch GPU/CPU usage in monitoring terminal
   - Check browser console (F12) for FPS logs

4. Expected results:
   - GPU usage: 40-60% when capturing
   - CPU: <20%
   - Memory: <80%
   - FPS: 15-20

## If Performance is Still Slow

### Easy Fixes (try in order):
1. Reduce camera resolution to 360p by modifying select.html/auto.html
2. Increase frameSkip from 2 to 3 or 4
3. Use 15W power mode instead of 25W

### Medium Fixes:
1. Use Ultra-Lite model: `modelComplexity: 0`
2. Disable visualization (`showVisuals = false`)
3. Switch to 25W power mode and max GPU clock

### Advanced Fixes:
1. Quantize the LSTM model to int8
2. Use TensorRT to convert models
3. Run Flask with gunicorn + uWSGI for better performance
4. Use nginx as reverse proxy & load balancer

## Notes

- These settings are optimized for the developer variant
- Production variants may have different thermal characteristics
- Always monitor thermals - Jetson has smart thermal management
- Avoid sustained 100% GPU usage to prevent thermal throttling
