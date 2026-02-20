#!/usr/bin/env python3
"""Test full inference pipeline with model weights."""

import torch
import numpy as np
from model import ModifiedLSTM
from pathutils import resource_path
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[TEST] Using device: {device}")

# Model constants
INPUT_SIZE = 188
HIDDEN_SIZE = 256
NUM_LAYERS = 2
CLASSES = [
    "Color_Black", "Color_Blue", "Color_Green", "Color_Orange", "Color_Pink", "Color_Red",
    "Color_White", "Color_Yellow",
    "Family_Daughter", "Family_Father", "Family_Grandfather", "Family_Grandmother", "Family_Mother", "Family_Son",
    "Numbers_Five", "Numbers_Four", "Numbers_One", "Numbers_Three", "Numbers_Two",
    "Relationship_Boy", "Relationship_Girl", "Relationship_Man", "Relationship_Woman",
    "Survival_Correct", "Survival_Don'tUnderstand", "Survival_No", "Survival_Understand", "Survival_Wrong", "Survival_Yes",
]
NUM_CLASSES = len(CLASSES)

# Load model
print("\n[TEST] Loading model...")
model = ModifiedLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
                     dropout=0.35, use_layernorm=True).to(device)
model.eval()

# Load weights
MODEL_PATH = resource_path("run35.pth")
print(f"[TEST] Loading weights from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"[TEST] ✗ Model file not found: {MODEL_PATH}")
    exit(1)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
print(f"[TEST] ✓ Weights loaded successfully")
print(f"[TEST] State dict keys: {list(state_dict.keys())[:3]}... (showing first 3)")

# Test inference
print("\n[TEST] Testing inference...")
dummy_input = torch.randn(1, 48, 188, dtype=torch.float32).to(device)
print(f"[TEST] Input shape: {dummy_input.shape}, device: {dummy_input.device}")

try:
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
    
    print(f"[TEST] ✓ Inference successful!")
    print(f"[TEST] Logits shape: {logits.shape}")
    print(f"[TEST] Probs shape: {probs.shape}")
    
    # Get prediction
    pred_idx = int(torch.argmax(probs[0]))
    confidence = float(probs[0, pred_idx])
    predicted_label = CLASSES[pred_idx]
    
    print(f"\n[TEST] Prediction results:")
    print(f"[TEST]   Predicted: {predicted_label}")
    print(f"[TEST]   Confidence: {confidence:.4f}")
    print(f"[TEST]   Top 5 probabilities:")
    top5_probs, top5_indices = torch.topk(probs[0], 5)
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"[TEST]     {i+1}. {CLASSES[idx]}: {prob.item():.4f}")
    
    print(f"\n[TEST] ✓ All tests passed!")
    
except Exception as e:
    print(f"[TEST] ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
