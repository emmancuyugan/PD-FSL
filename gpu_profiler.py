"""
GPU Monitoring & Profiling for Jetson Deployment
Tracks memory, latency, temperature, and utilization
"""

import torch
import time
import psutil
import os
from contextlib import contextmanager

class GPUProfiler:
    """Profile GPU usage and inference latency"""
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.is_cuda = device.type == "cuda"
        self.metrics = {
            'total_inference_time': 0,
            'total_inferences': 0,
            'peak_memory': 0,
            'current_memory': 0,
        }
    
    def reset_peak_memory(self):
        """Reset peak memory tracking"""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    
    @contextmanager
    def profile_inference(self, label="inference"):
        """Context manager to profile a single inference"""
        if self.is_cuda:
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_mb()
        
        try:
            yield
        finally:
            if self.is_cuda:
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            peak_memory = self._get_peak_memory_mb()
            
            self.metrics['total_inference_time'] += elapsed
            self.metrics['total_inferences'] += 1
            self.metrics['peak_memory'] = max(self.metrics['peak_memory'], peak_memory)
            
            # Log per-inference metrics
            if self.metrics['total_inferences'] % 10 == 0:
                avg_latency = (self.metrics['total_inference_time'] / 
                             self.metrics['total_inferences']) * 1000
                print(f"[GPU] {label} | Latency: {elapsed*1000:.2f}ms "
                      f"| Avg: {avg_latency:.2f}ms | Peak Memory: {peak_memory}MB")
    
    def _get_memory_mb(self):
        """Get current GPU/CPU memory in MB"""
        if self.is_cuda:
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _get_peak_memory_mb(self):
        """Get peak GPU/CPU memory in MB"""
        if self.is_cuda:
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def get_summary(self):
        """Get profiling summary"""
        if self.metrics['total_inferences'] == 0:
            return "No inferences profiled yet"
        
        avg_latency = (self.metrics['total_inference_time'] / 
                      self.metrics['total_inferences']) * 1000
        throughput = self.metrics['total_inferences'] / self.metrics['total_inference_time']
        
        summary = f"""
GPU PROFILING SUMMARY
=====================
Device: {self.device}
Total Inferences: {self.metrics['total_inferences']}
Average Latency: {avg_latency:.2f} ms
Peak Latency: {self.metrics.get('max_latency', 0)*1000:.2f} ms
Throughput: {throughput:.1f} req/sec
Peak Memory: {self.metrics['peak_memory']:.1f} MB
"""
        if self.is_cuda:
            summary += f"CUDA Device: {torch.cuda.get_device_name(0)}\n"
            summary += f"Compute Capability: {torch.cuda.get_device_capability(0)}\n"
        
        return summary
    
    def print_cuda_info(self):
        """Print detailed CUDA information"""
        if not self.is_cuda:
            print("CUDA not available - running on CPU")
            return
        
        print("\n" + "="*60)
        print("CUDA DEVICE INFORMATION")
        print("="*60)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Available Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f} GB")
        print("="*60 + "\n")


class AdvancedOptimizer:
    """Advanced optimizations for Jetson Orin"""
    
    @staticmethod
    def enable_tensor_cores(device):
        """Enable NVIDIA Tensor Cores if available"""
        if device.type == 'cuda':
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[GPU] TF32 precision enabled for Tensor Cores")
            except:
                pass
    
    @staticmethod
    def enable_cu_blas_lt(device):
        """Enable cuBLASLt for better performance on Jetson"""
        if device.type == 'cuda':
            try:
                torch.backends.cuda.cublasLt_enabled = True
                print("[GPU] cuBLASLt enabled")
            except:
                pass
    
    @staticmethod
    def optimize_memory(device, fraction=0.5):
        """Configure GPU memory growth and limits"""
        if device.type == 'cuda':
            # Set memory growth (Jetson doesn't support this like desktop GPUs)
            # But we can pre-allocate a fraction to prevent fragmentation
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, device=0)
                print(f"[GPU] Memory fraction limited to {fraction*100}% to prevent thermal throttling")
            except:
                pass
    
    @staticmethod
    def profile_model(model, device, input_size=(1, 48, 188)):
        """Profile model inference latency"""
        import numpy as np
        
        model.eval()
        dummy_input = torch.randn(*input_size, device=device)
        
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.inference_mode():
                _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        times = np.array(times[1:]) * 1000  # Skip first, convert to ms
        
        print(f"\nMODEL PROFILING ({device})")
        print(f"Input Shape: {input_size}")
        print(f"Avg Latency: {np.mean(times):.2f} ms")
        print(f"Min Latency: {np.min(times):.2f} ms")
        print(f"Max Latency: {np.max(times):.2f} ms")
        print(f"Std Dev: {np.std(times):.2f} ms")
        print(f"Throughput: {1000/np.mean(times):.1f} req/sec\n")
        
        return {
            'avg': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times),
            'throughput': 1000/np.mean(times)
        }


class JetsonDiagnostics:
    """Diagnose Jetson hardware capabilities"""
    
    @staticmethod
    def check_hardware():
        """Check Jetson hardware"""
        print("\n" + "="*60)
        print("JETSON HARDWARE DIAGNOSTICS")
        print("="*60)
        
        # Check GPU
        if torch.cuda.is_available():
            print("✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("✗ CUDA is NOT available (CPU-only mode)")
        
        # Check PyTorch
        print(f"\n✓ PyTorch {torch.version.__version__}")
        print(f"  CUDA Toolkit: {torch.version.cuda}")
        print(f"  cuDNN: {torch.backends.cudnn.version()}")
        
        # Check CPU
        print(f"\n✓ CPU: {psutil.cpu_count()} cores")
        print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        
        # Check optimization flags
        print(f"\n✓ Optimizations:")
        print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        try:
            print(f"  TF32 (Tensor Cores): {torch.backends.cuda.matmul.allow_tf32}")
        except:
            print(f"  TF32: Not supported")
        
        print("="*60 + "\n")
    
    @staticmethod
    def estimate_jetson_performance(model_name="ModifiedLSTM"):
        """Estimate expected performance on Jetson Orin Nano"""
        print("\n" + "="*60)
        print("EXPECTED JETSON PERFORMANCE ESTIMATES")
        print("="*60)
        
        estimates = {
            "ModifiedLSTM": {
                "FP32": {"latency_ms": 45, "throughput": 22},
                "FP16": {"latency_ms": 18, "throughput": 56},
                "FP16+TorchScript": {"latency_ms": 12, "throughput": 83},
                "INT8": {"latency_ms": 8, "throughput": 125},
            }
        }
        
        if model_name not in estimates:
            print(f"No estimates for {model_name}")
            return
        
        configs = estimates[model_name]
        print(f"Model: {model_name}\n")
        print(f"{'Config':<20} {'Latency':<15} {'Throughput':<15}")
        print("-" * 50)
        for config, metrics in configs.items():
            print(f"{config:<20} {metrics['latency_ms']:<15.1f}ms {metrics['throughput']:<15.1f}req/s")
        
        print("\nRecommendation: Use FP16 for balanced speed/memory/power")
        print("="*60 + "\n")


def export_summary_report(profiler, filepath="gpu_profile_report.txt"):
    """Export profiling report to file"""
    with open(filepath, 'w') as f:
        f.write(profiler.get_summary())
        f.write("\n\nGenerated at: " + time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check hardware
    JetsonDiagnostics.check_hardware()
    
    # Show performance estimates
    JetsonDiagnostics.estimate_jetson_performance()
    
    # Print CUDA info if available
    profiler = GPUProfiler(device)
    profiler.print_cuda_info()
