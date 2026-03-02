import torch
import numpy as np
from model import ModifiedLSTM

MODEL_PATH = "run2.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
config = checkpoint["config"]

CLASSES = config["CLASSES"]
INPUT_SIZE = config["FEATURE_DIM"]
HIDDEN_SIZE = config["HIDDEN_SIZE"]
NUM_LAYERS = config["NUM_LAYERS"]
DROPOUT = config["DROPOUT"]
SEQ_LEN = config.get("SEQ_LEN", config.get("SEQUENCE_LENGTH", 48))

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

def inspect_sequence(sample_path):
    arr = np.load(sample_path).astype(np.float32)
    T, D = arr.shape
    
    if T > SEQ_LEN:
        arr = arr[:SEQ_LEN]
    elif T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, D), np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        out = x
        for i, lstm in enumerate(model.lstm_layers):
            out, _ = lstm(out)
            out = model.act(out)
            out = model.drop(out)

        logits_per_frame = model.fc(out.squeeze(0))
        preds = logits_per_frame.argmax(dim=1).cpu().numpy()

    return preds


def predict_last_frame(sample_path):
    arr = np.load(sample_path).astype(np.float32)
    T, D = arr.shape
    
    if T > SEQ_LEN:
        arr = arr[:SEQ_LEN]
    elif T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, D), np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        out = x
        for i, lstm in enumerate(model.lstm_layers):
            out, _ = lstm(out)
            out = model.act(out)
            out = model.drop(out)

        last = out[:, -1, :]
        logits = model.fc(last)
        pred = logits.argmax(1).item()

    return CLASSES[pred]


if __name__ == "__main__":
    path = r"C:\Users\ACER NITRO\OneDrive\Documents\GitHub\FSL-PROJECT-DESIGN\BETTER\Training\KEYPOINTS\Family_Father\18.npy"
    
    print("Per-frame predictions:")
    print(inspect_sequence(path))
    
    print("\nLast-frame prediction:")
    print(predict_last_frame(path))