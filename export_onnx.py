import torch
import pathlib, platform
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PurePosixPath

from model import ModifiedLSTM

MODEL_PATH = "run20.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

config = checkpoint["config"]

INPUT_SIZE  = config["FEATURE_DIM"]
HIDDEN_SIZE = config["HIDDEN_SIZE"]
NUM_LAYERS  = config["NUM_LAYERS"]
DROPOUT     = config["DROPOUT"]
CLASSES     = config["CLASSES"]
SEQ_LEN     = config.get("SEQ_LEN", config.get("SEQUENCE_LENGTH", 48))

print("Input size:", INPUT_SIZE)
print("Seq len:", SEQ_LEN)
print("Hidden:", HIDDEN_SIZE)

# Build model  — always export in FP32 (TensorRT handles its own quantisation)
model = ModifiedLSTM(
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    len(CLASSES),
    dropout=DROPOUT,
    use_layernorm=True
).to(device).float()

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Dummy input (FP32)
dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE, dtype=torch.float32).to(device)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ ONNX export successful → model.onnx")
print()
print("Next steps — build the TensorRT engine on your Jetson:")
print("  Option A (FP32, most accurate):")
print(f"    trtexec --onnx=model.onnx --saveEngine=model.engine")
print()
print("  Option B (FP16, faster but may lose some LSTM accuracy):")
print(f"    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
print()
print("  Option C (mixed — FP16 where safe, FP32 for sensitive layers):")
print(f"    trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 --precisionConstraints=prefer")