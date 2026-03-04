import torch
import numpy as np
from model import ModifiedLSTM

# =============================
# SETTINGS
# =============================
MODEL_PATH = "run16.pt"   # <-- make sure this is your NEW trained checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# LOAD CHECKPOINT
# =============================
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
config = checkpoint["config"]

CLASSES      = config["CLASSES"]
INPUT_SIZE   = config["FEATURE_DIM"]
HIDDEN_SIZE  = config["HIDDEN_SIZE"]
NUM_LAYERS   = config["NUM_LAYERS"]
DROPOUT      = config["DROPOUT"]
SEQ_LEN      = config.get("SEQ_LEN", config.get("SEQUENCE_LENGTH", 48))

print("Loaded config:")
print(f"Hidden: {HIDDEN_SIZE}")
print(f"Layers: {NUM_LAYERS}")
print(f"Dropout: {DROPOUT}")
print(f"Seq Len: {SEQ_LEN}")
print(f"Classes: {len(CLASSES)}")

# =============================
# BUILD MODEL
# =============================
model = ModifiedLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    len(CLASSES),
    dropout=DROPOUT,
    use_layernorm=True
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.\n")

# =============================
# PREDICTION FUNCTION
# =============================
def inspect_sequence(sample_path):
    arr = np.load(sample_path).astype(np.float32)
    T, D = arr.shape

    # Pad or truncate to SEQ_LEN
    if T > SEQ_LEN:
        arr = arr[:SEQ_LEN]
    elif T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, D), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs.max().item()

    return {
        "predicted_class": CLASSES[pred_idx],
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    path = r"C:\Users\ACER NITRO\OneDrive\Documents\GitHub\FSL-PROJECT-DESIGN\BETTER\Training\KEYPOINTS\Family_Father\15.npy"

    result = inspect_sequence(path)
    print("Prediction Result:")
    print(result)