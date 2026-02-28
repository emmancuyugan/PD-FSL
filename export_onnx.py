import torch
from model import ModifiedLSTM  # make sure this file exists

MODEL_PATH = "run2.pt"  # change if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

config = checkpoint["config"]

INPUT_SIZE  = config["FEATURE_DIM"]
HIDDEN_SIZE = config["HIDDEN_SIZE"]
NUM_LAYERS  = config["NUM_LAYERS"]
DROPOUT     = config["DROPOUT"]
CLASSES     = config["CLASSES"]
SEQ_LEN     = config["SEQUENCE_LENGTH"]

print("Input size:", INPUT_SIZE)
print("Seq len:", SEQ_LEN)
print("Hidden:", HIDDEN_SIZE)

# Build model
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

# Dummy input
dummy_input = torch.randn(1, SEQ_LEN, INPUT_SIZE).to(device)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("✅ ONNX export successful.")