#!/usr/bin/env python3
"""Quick test to verify model inference works."""

import torch
import numpy as np
from model import ModifiedLSTM

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TEST] Using device: {device}")

# Create model
INPUT_SIZE = 188
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 30

model = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                     dropout=0.35, use_layernorm=True).to(device)
model.eval()

print(f"[TEST] Model created successfully")
print(f"[TEST] Model device: {next(model.parameters()).device}")

# Create dummy input: (batch_size=1, seq_len=48, features=188)
dummy_input = torch.randn(1, 48, 188, dtype=torch.float32).to(device)
print(f"[TEST] Input shape: {dummy_input.shape}")
print(f"[TEST] Input device: {dummy_input.device}")

# Test forward pass
try:
    with torch.no_grad():
        output = model(dummy_input)
    print(f"[TEST] ✓ Forward pass successful!")
    print(f"[TEST] Output shape: {output.shape}")
    print(f"[TEST] Output: {output[0]}")
    print(f"[TEST] Max output value: {output.max().item():.4f}")
    print(f"[TEST] Min output value: {output.min().item():.4f}")
except Exception as e:
    print(f"[TEST] ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n[TEST] All tests passed! Model is working correctly.")
