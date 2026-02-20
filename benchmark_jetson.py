#!/usr/bin/env python3
"""
Jetson FSL Optimization Benchmark
Compares inference performance with and without optimizations
"""
import sys
import time
import numpy as np

sys.path.insert(0, '.')

import torch
from model import ModifiedLSTM
from jetson_optimization import JetsonOptimizer
from pathutils import resource_path

def benchmark_model(model, device, input_shape=(1, 48, 188), num_iterations=100, 
                   use_mixed_precision=False, model_name="Model"):
    """Run inference benchmark"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    print(f"  Warming up {model_name}...")
    with torch.inference_mode():
        for _ in range(5):
            if use_mixed_precision and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"  Benchmarking {model_name} ({num_iterations} iterations)...")
    times = []
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.inference_mode():
            if use_mixed_precision and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = 1000.0 / avg_time  # Requests per second
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'throughput_rps': throughput
    }

def main():
    print("=" * 70)
    print("FSL JETSON OPTIMIZATION BENCHMARK")
    print("=" * 70)
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    
    # Load model
    print("\nLoading model...")
    MODEL_PATH = resource_path("run35.pth")
    INPUT_SIZE = 188
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    NUM_CLASSES = 29
    
    # Test 1: Baseline FP32
    print("\n" + "=" * 70)
    print("TEST 1: Baseline FP32 Model")
    print("=" * 70)
    model_fp32 = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                              dropout=0.35, use_layernorm=True).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model_fp32.load_state_dict(state_dict)
    
    results_fp32 = benchmark_model(model_fp32, device, model_name="FP32 Model")
    print(f"  Average Inference Time: {results_fp32['avg_time_ms']:.2f} ± {results_fp32['std_time_ms']:.2f} ms")
    print(f"  Throughput: {results_fp32['throughput_rps']:.1f} requests/second")
    
    # Test 2: Mixed Precision (FP16)
    if device.type == 'cuda':
        print("\n" + "=" * 70)
        print("TEST 2: Mixed Precision (FP16) Model")
        print("=" * 70)
        model_fp16 = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                                  dropout=0.35, use_layernorm=True).to(device)
        model_fp16.load_state_dict(state_dict)
        model_fp16 = JetsonOptimizer.convert_to_half_precision(model_fp16, device)
        
        results_fp16 = benchmark_model(model_fp16, device, use_mixed_precision=True, 
                                      model_name="FP16 Model")
        print(f"  Average Inference Time: {results_fp16['avg_time_ms']:.2f} ± {results_fp16['std_time_ms']:.2f} ms")
        print(f"  Throughput: {results_fp16['throughput_rps']:.1f} requests/second")
        
        # Calculate speedup
        speedup = results_fp32['avg_time_ms'] / results_fp16['avg_time_ms']
        print(f"\n  ➤ FP16 Speedup: {speedup:.2f}x faster than FP32")
        
        # Test 3: TorchScript Compilation
        print("\n" + "=" * 70)
        print("TEST 3: TorchScript Compiled Model")
        print("=" * 70)
        try:
            model_scripted = torch.jit.trace(model_fp32, 
                                            torch.randn(1, 48, INPUT_SIZE, device=device))
            results_scripted = benchmark_model(model_scripted, device, model_name="TorchScript Model")
            print(f"  Average Inference Time: {results_scripted['avg_time_ms']:.2f} ± {results_scripted['std_time_ms']:.2f} ms")
            print(f"  Throughput: {results_scripted['throughput_rps']:.1f} requests/second")
            
            speedup_scripted = results_fp32['avg_time_ms'] / results_scripted['avg_time_ms']
            print(f"\n  ➤ TorchScript Speedup: {speedup_scripted:.2f}x faster than FP32")
        except Exception as e:
            print(f"  TorchScript compilation failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print("\nFor Jetson Orin Nano Super (67 TOPS):")
    print("  1. Use FP16 (half precision) for ~2-3x speedup")
    print("  2. Enable mixed precision with torch.cuda.amp.autocast()")
    print("  3. Use batch_size=1 for real-time inference")
    print("  4. Keep model evaluation mode (model.eval())")
    print("  5. Use TorchScript compilation when possible (~1.5x speedup)")
    print("  6. Monitor memory usage with torch.cuda.set_per_process_memory_fraction()")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
