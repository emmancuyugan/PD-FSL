"""
Diagnose confidence for Father, Mother, Don't Understand.
Loads the model, picks 5 random samples per class, runs inference,
and shows the top-3 predicted classes + confidence for each.
"""
import torch
import numpy as np
import glob, os, random
from pathlib import Path
from model import ModifiedLSTM

random.seed(42)
np.random.seed(42)

# ── Config ──
MODEL_PATH = "run18.pt"
KEYPOINTS  = Path("Training") / "KEYPOINTS"
TARGET_CLASSES = ["Family_Father", "Family_Mother", "Survival_Don't Understand"]
N_TRIALS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ──
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
cfg  = ckpt["config"]

CLASSES   = cfg["CLASSES"]
SEQ_LEN   = cfg.get("SEQ_LEN", cfg.get("SEQUENCE_LENGTH", 48))
FEAT_DIM  = cfg["FEATURE_DIM"]
FLAG_S    = cfg["FLAG_START"]
FLAG_E    = cfg["FLAG_END"]

model = ModifiedLSTM(
    FEAT_DIM, cfg["HIDDEN_SIZE"], cfg["NUM_LAYERS"],
    len(CLASSES), dropout=cfg["DROPOUT"], use_layernorm=True
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
if device.type == "cuda":
    model.half()
model.eval()

print(f"Model: {MODEL_PATH}  |  {len(CLASSES)} classes  |  SEQ_LEN={SEQ_LEN}")
print(f"Device: {device}\n")

# ── Helpers ──
def load_and_pad(path):
    arr = np.load(path).astype(np.float32)
    T, D = arr.shape
    if T > SEQ_LEN:
        arr = arr[:SEQ_LEN]
    elif T < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - T, D), np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr

def infer(arr):
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    if device.type == "cuda":
        x = x.half()
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# ── Run diagnosis ──
for cls_name in TARGET_CLASSES:
    cls_dir = KEYPOINTS / cls_name
    if not cls_dir.exists():
        print(f"[SKIP] {cls_name} — directory not found at {cls_dir}")
        continue

    files = sorted(glob.glob(str(cls_dir / "*.npy")))
    if not files:
        print(f"[SKIP] {cls_name} — no .npy files")
        continue

    # True label index
    true_idx = CLASSES.index(cls_name) if cls_name in CLASSES else -1

    selected = random.sample(files, min(N_TRIALS, len(files)))

    print("=" * 70)
    print(f"  {cls_name}  (true index={true_idx}, {len(files)} total samples)")
    print("=" * 70)

    all_confs = []
    all_correct = 0

    for i, fpath in enumerate(selected):
        arr   = load_and_pad(fpath)
        probs = infer(arr)

        pred_idx  = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        true_conf = float(probs[true_idx]) if true_idx >= 0 else 0.0
        correct   = pred_idx == true_idx

        all_confs.append(true_conf)
        if correct:
            all_correct += 1

        # Top 5 competing classes
        top5_idx = np.argsort(probs)[::-1][:5]

        mark = "✓" if correct else "✗"
        print(f"\n  Trial {i+1}/{N_TRIALS}  {mark}  ({Path(fpath).name})")
        print(f"  True class conf: {true_conf:.4f}  |  Predicted: {CLASSES[pred_idx]} ({pred_conf:.4f})")
        print(f"  {'Rank':<6} {'Class':<30} {'Confidence':>10}")
        print(f"  {'-'*6} {'-'*30} {'-'*10}")
        for rank, idx in enumerate(top5_idx, 1):
            tag = " ◄ TRUE" if idx == true_idx else ""
            print(f"  {rank:<6} {CLASSES[idx]:<30} {probs[idx]:>10.4f}{tag}")

    avg_conf = np.mean(all_confs)
    print(f"\n  >>> {cls_name}: {all_correct}/{len(selected)} correct, avg true-class conf = {avg_conf:.4f}")
    print()

print("\nDone.")
