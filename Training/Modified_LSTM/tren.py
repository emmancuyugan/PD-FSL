# %% [markdown]
# # Libraries

# %%
import os, re, glob, json, time, random
from pathlib import Path
from copy import deepcopy
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from torch.amp import autocast, GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from math import sqrt
from model import build_modified_lstm

# Directories
if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parents[2]
# If running in notebook
else:
    BASE_DIR = Path.cwd().parents[1]

print("BASE_DIR:", BASE_DIR)

DATASET_DIR = BASE_DIR / "Training" / "KEYPOINTS"
EXPORT_ROOT = BASE_DIR / "Training" / "Modified_LSTM" / "ModifiedLSTM_Exports"
BEST_DIR    = BASE_DIR / "Training" / "Modified_LSTM" / "ModifiedLSTM_best"

EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset exists:", DATASET_DIR.exists())
print("Total classes:", len([d for d in DATASET_DIR.iterdir() if d.is_dir()]))

# %%
print("Base:", BASE_DIR)
print("Dataset exists:", DATASET_DIR.exists())
print("Classes:", sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()]))

# %% [markdown]
# # Configuration and Helpers

# %%
try:
    import psutil
except Exception:
    psutil = None

SEED = 2025
import numpy as np, torch, random, time, os
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Handle CUDA compatibility issues
try:
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        # Test if CUDA actually works
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device("cpu")
except Exception as cuda_error:
    print(f"[WARNING] CUDA error: {cuda_error}")
    print("[WARNING] Falling back to CPU")
    device = torch.device("cpu")

print(f"Using device: {device}")

RUN_STAMP = time.strftime("%Y%m%d-%H%M%S")
EXPORT_DIR = EXPORT_ROOT / "HybridLSTMGRU" / RUN_STAMP
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_DIR = BASE_DIR / "Training" / "KEYPOINTS"

CONFIG = {
    "DATA_DIRS": [str(DATASET_DIR)],
    "SEQUENCE_LENGTH": 48,
    "BATCH_SIZE": 16,
    "EPOCHS": 150,
    "LR": 3e-4,
    "WEIGHT_DECAY": 1e-4,
    "DROPOUT": 0.30,
    "PATIENCE": 10,
    "VAL_SPLIT": 0.15,
    "TEST_SPLIT": 0.15,
    "NORM_MODE": "none",
    "AUG_ON": True,
    "EXPORT_DIR": EXPORT_DIR,
    "BEST_DIR": BEST_DIR,
}

print("Dataset dir:", DATASET_DIR)
CONFIG["EXPORT_DIR"] = EXPORT_DIR
CONFIG["BEST_DIR"] = BEST_DIR
from pathlib import Path
Path(BEST_DIR).mkdir(parents=True, exist_ok=True)
CONFIG["BEST_DIR"] = BEST_DIR

DATASET_DIR = BASE_DIR / "KEYPOINTS"
if not DATASET_DIR.exists():
    fallback_dir = Path(CONFIG["DATA_DIRS"][0])
    if fallback_dir.exists():
        DATASET_DIR = fallback_dir
    else:
        raise FileNotFoundError(f"KEYPOINTS directory not found at {DATASET_DIR} or {fallback_dir}")

actions = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
CONFIG["CLASSES"] = actions
print("Discovered classes:", CONFIG["CLASSES"])

# Should be same feature as our extraction
SELECT_JOINTS = [0, 4, 8, 12, 16, 20]
DERIVED_PER_JOINT = 5
DERIVED_DIM = len(SELECT_JOINTS) * 2 * DERIVED_PER_JOINT
BASE_HAND_DIM = 42 * 3
APPEND_FLAGS = True
FEATURE_DIM = BASE_HAND_DIM + DERIVED_DIM + (2 if APPEND_FLAGS else 0)  # 188 total
CONFIG["FEATURE_DIM"] = FEATURE_DIM

# flag indices
FLAG_START = FEATURE_DIM - 2
FLAG_END   = FEATURE_DIM
CONFIG["FLAG_START"] = FLAG_START
CONFIG["FLAG_END"]   = FLAG_END

print(f"Feature dimension set to {FEATURE_DIM} (flags at {FLAG_START}:{FLAG_END})")


# %%
# Helpers
def _next_run_id(dir_path, prefix="run"):
    """
    Scan dir for files or subfolders named like 'run<number>.*' or 'run<number>'.
    Returns the next integer id (starting at 1).
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)")
    seen = []
    for item in p.iterdir():
        m = pat.match(item.stem)  
        if m:
            try:
                seen.append(int(m.group(1)))
            except ValueError:
                pass
    return (max(seen) + 1) if seen else 1

def _latest_run_id(dir_path, prefix="run"):
    p = Path(dir_path)
    if not p.exists():
        return None
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)")
    seen = []
    for item in p.iterdir():
        m = pat.match(item.stem)
        if m:
            try:
                seen.append(int(m.group(1)))
            except ValueError:
                pass
    return max(seen) if seen else None

# Ensure directory exists
Path(CONFIG["BEST_DIR"]).mkdir(parents=True, exist_ok=True)
RUN_PREFIX = "run"

def _find_latest_run_id(dir_path, prefix=RUN_PREFIX):
    latest = _latest_run_id(dir_path, prefix)
    return latest if latest is not None else 0  # 0 means none yet

CURRENT_RUN_ID = _find_latest_run_id(CONFIG["BEST_DIR"], prefix=RUN_PREFIX)

SKIP_VIDEO_TESTS = False

class Chrono:
    def __enter__(self): 
        self.t0 = time.time(); 
        return self
    def __exit__(self, *exc): 
        self.dt = time.time() - self.t0

def human_time(sec):
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else (f"{m:d}m {s:02d}s" if m else f"{s:d}s")

def report_memory(prefix=""):
    gpu_mem = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_mem = torch.cuda.memory_allocated() / (1024**2)
    ram = psutil.Process().memory_info().rss / (1024**2) if psutil else None
    msg = []
    if ram is not None: msg.append(f"RAM: {ram:.1f} MB")
    if gpu_mem is not None: msg.append(f"GPU: {gpu_mem:.1f} MB")
    print((prefix+" ").strip(), " | ".join(msg))

report_memory("Init mem")
print(f"Exports will be saved to: {CONFIG['EXPORT_DIR']}")

# %% [markdown]
# # Dataset

# %%
# Utilities
FLAG_START = CONFIG["FLAG_START"]
FLAG_END   = CONFIG["FLAG_END"]
FEATURE_DIM = CONFIG["FEATURE_DIM"]

def infer_label_from_path(path, classes):
    p = Path(path)
    for c in classes:
        if c in p.parts:
            return classes.index(c)
    stem = p.stem
    for c in classes:
        if stem.startswith(c):
            return classes.index(c)
    raise ValueError(f"Cannot infer label for {path}")

def list_npy_files(data_dirs):
    files = []
    for d in data_dirs:
        if not os.path.isdir(d): continue
        files.extend(glob.glob(str(Path(d) / "**" / "*.npy"), recursive=True))
    return sorted(files)

def effective_length(arr):
    T, D = arr.shape
    if D >= FEATURE_DIM:
        flags = arr[:, FLAG_START:FLAG_END]
        valid = (flags.sum(axis=1) > 0.0)
        if valid.any(): return int(valid.sum())
    coords = arr[:, :min(BASE_HAND_DIM, D)].reshape(T, -1)
    if T <= 1:
        return 1 if np.abs(coords).sum() > 1e-6 else 0
    diffs = np.abs(np.diff(coords, axis=0)).reshape(T-1, -1)
    mag = diffs.mean(axis=1)
    eps = 1e-6
    active = np.zeros(T, dtype=bool)
    active[0] = (mag[0] > eps) if len(mag) else (np.abs(coords[0]).sum() > eps)
    active[1:] = mag > eps
    if active.any(): return int(active.sum())
    nonzero = (np.abs(coords).sum(axis=1) > 1e-6)
    return int(nonzero.sum()) if nonzero.any() else (T if coords.any() else 0)

def _presence_score(arr):
    T, D = arr.shape
    if D >= FEATURE_DIM:
        flags = arr[:, FLAG_START:FLAG_END]
        return ((flags[:,0] > 0.5) | (flags[:,1] > 0.5)).astype(np.float32)
    coords = arr[:, :min(BASE_HAND_DIM, D)]
    if T < 2:
        return (np.abs(coords).sum(axis=1) > 1e-6).astype(np.float32)
    diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    diffs = np.concatenate([[0.0], diffs])
    rng = diffs.ptp()
    return (diffs - diffs.min()) / (rng + 1e-6)

def _random_crop_start(arr, L):
    T = len(arr)
    if T <= L: return 0
    score = _presence_score(arr)
    k = T - L + 1
    window_scores = np.zeros(k, dtype=np.float32)
    csum = np.concatenate([[0.0], np.cumsum(score)])
    for s in range(k):
        window_scores[s] = csum[s+L] - csum[s]
    probs = window_scores + 1e-6
    probs /= probs.sum()
    return int(np.random.choice(np.arange(k), p=probs))

# %%
FLAG_START = CONFIG["FLAG_START"]
FLAG_END   = CONFIG["FLAG_END"]
FEATURE_DIM = CONFIG["FEATURE_DIM"]

# Data augmentation
def _resample(coords, flags, L):
    T = coords.shape[0]
    if T == L: return coords, flags
    idx = np.linspace(0, T-1, L)
    lo  = np.floor(idx).astype(int)
    hi  = np.clip(lo+1, 0, T-1)
    w   = (idx - lo)[:,None,None]
    c   = (1-w)*coords[lo] + w*coords[hi]
    f   = flags[np.round(idx).astype(int)]
    return c.astype(np.float32), f.astype(np.float32)

def temporal_shift(x, max_shift=3):
    if max_shift <= 0: return x
    T, D = x.shape
    s = np.random.randint(-max_shift, max_shift+1)
    if s == 0: return x
    pad = np.zeros((abs(s), D), np.float32)
    return (np.concatenate([pad, x], 0)[:T] if s > 0
            else np.concatenate([x[-s:], pad], 0)[:T])

def temporal_stretch(x, rmin=0.90, rmax=1.12):
    T, D = x.shape
    r = np.random.uniform(rmin, rmax)
    Lnew = max(8, int(round(T*r)))
    coords = x[:, :BASE_HAND_DIM].reshape(T, 42, 3)
    flags  = x[:, FLAG_START:FLAG_END] if D >= FEATURE_DIM else np.zeros((T, 2), np.float32)
    c, f = _resample(coords, flags, Lnew)
    y = np.zeros((T, D), np.float32)
    Lc = min(Lnew, T)
    y[:Lc, :BASE_HAND_DIM] = c.reshape(Lnew, BASE_HAND_DIM)[:Lc]
    if D >= FEATURE_DIM: y[:Lc, FLAG_START:FLAG_END] = f[:Lc]
    return y

def temporal_dropout(x, max_runs=1, run_len=(1, 2)):
    T, D = x.shape
    coords = x[:, :BASE_HAND_DIM].reshape(T, 42, 3)
    flags  = x[:, FLAG_START:FLAG_END] if D >= FEATURE_DIM else np.zeros((T, 2), np.float32)
    runs = np.random.randint(0, max_runs+1)
    for _ in range(runs):
        k = np.random.randint(run_len[0], run_len[1]+1)
        s = np.random.randint(0, max(1, T-k))
        coords[s:s+k] = 0.0
        if D >= FEATURE_DIM: flags[s:s+k] = 0.0
    x[:, :BASE_HAND_DIM] = coords.reshape(T, BASE_HAND_DIM)
    if D >= FEATURE_DIM: x[:, FLAG_START:FLAG_END] = flags
    return x

def spatial_jitter(x, rot_deg=7.0, scale_j=0.08, trans_j=0.03, noise_std=0.01):
    T, D = x.shape
    coords = x[:, :BASE_HAND_DIM].reshape(T, 42, 3)
    flags  = x[:, FLAG_START:FLAG_END] if D >= FEATURE_DIM else np.zeros((T, 2), np.float32)

    theta = np.deg2rad(np.random.uniform(-rot_deg, rot_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.asarray([[c, -s], [s, c]], np.float32)
    sc = 1.0 + np.random.uniform(-scale_j, scale_j)
    tr = np.random.uniform(-trans_j, trans_j, size=(2,)).astype(np.float32)

    xy = coords[..., :2]
    present = (flags.sum(1) > 0.5)[:, None, None] if D >= FEATURE_DIM else np.ones((T,1,1),bool)
    xy = np.where(present, (xy @ R.T) * sc + tr, xy)
    coords[..., :2] = xy

    noise = np.random.normal(0.0, noise_std, size=coords.shape).astype(np.float32)
    coords = np.where(present, coords + noise, coords)

    x[:, :BASE_HAND_DIM] = coords.reshape(T, BASE_HAND_DIM)
    if D >= FEATURE_DIM: x[:, FLAG_START:FLAG_END] = flags
    return x

def random_augment(x):
    if np.random.rand() < 0.8: x = temporal_shift(x, 3)
    if np.random.rand() < 0.6: x = temporal_stretch(x, 0.90, 1.12)
    if np.random.rand() < 0.4: x = temporal_dropout(x, 1, (1, 2))
    if np.random.rand() < 0.7: x = spatial_jitter(x, 7.0, 0.08, 0.03, 0.01)
    return x

# %%
class KeypointSequenceDataset(Dataset):
    def __init__(self, files, classes, seq_len, split="train",
                 mean_vec=None, std_vec=None):
        self.files = files
        self.classes = classes
        self.seq_len = seq_len
        self.split = split
        self.mean_vec = mean_vec
        self.std_vec = std_vec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        arr = np.load(f).astype(np.float32)  # (T, D)
        T, D = arr.shape

        # normalize (optional)
        if CONFIG.get("NORM_MODE") == "zscore" and self.mean_vec is not None:
            arr = (arr - self.mean_vec) / self.std_vec

        # apply augmentation (only train)
        if self.split == "train" and CONFIG.get("AUG_ON", True):
            arr = random_augment(arr)

        # pad or trim
        L = self.seq_len
        if T > L:
            arr = arr[:L]
        elif T < L:
            pad = np.zeros((L - T, D), np.float32)
            arr = np.concatenate([arr, pad], axis=0)

        y = infer_label_from_path(f, self.classes)
        return {
            "x": torch.from_numpy(arr),
            "len": torch.tensor(min(T, L)),
            "y": torch.tensor(y, dtype=torch.long),
            "path": f,
        }


def collate_batch(samples):
    xs = torch.stack([s["x"] for s in samples])
    lens = torch.stack([s["len"] for s in samples])
    ys = torch.stack([s["y"] for s in samples])
    paths = [s["path"] for s in samples]
    return xs, lens, ys, paths

# %%
TEST_KEEP_AUG = False

FLAG_START = CONFIG["FLAG_START"]
FLAG_END   = CONFIG["FLAG_END"]
FEATURE_DIM = CONFIG["FEATURE_DIM"]

def base_group(path: str) -> str:
    p = Path(path)
    label = p.parent.name
    stem  = re.sub(r"_aug\d+$", "", p.stem, flags=re.IGNORECASE)
    return f"{label}::{stem}"

def is_augmented(path: str) -> bool:
    return bool(re.search(r"_aug\d+$", Path(path).stem.lower()))

all_files = list_npy_files(CONFIG["DATA_DIRS"])

usable, labels = [], []
for f in all_files:
    try:
        arr = np.load(f, mmap_mode='r')
        if arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
            usable.append(f)
            labels.append(infer_label_from_path(f, CONFIG["CLASSES"]))
    except Exception:
        pass

assert len(usable) > 0, f"No usable .npy found — expected feature dim {FEATURE_DIM}."
print(f"Total usable samples (pre-filter): {len(usable)}")

# Drop weak clips before splitting
MIN_ACTIVE = 6
M_THR      = 0.02

def motion_active_len(fp, m_thr=M_THR):
    a = np.load(fp)
    if a.ndim != 2 or a.shape[1] != FEATURE_DIM:
        return 0
    T, D = a.shape
    if T < 2: return 0
    coords = a[:, :BASE_HAND_DIM].reshape(T, 42, 3)
    v = np.diff(coords, axis=0)
    m = np.linalg.norm(v, axis=(1,2))
    rng = np.ptp(m)
    m = (m - m.min()) / (rng + 1e-6)
    return int((m > m_thr).sum())

usable_filtered = []
for f in usable:
    try:
        if motion_active_len(f) >= MIN_ACTIVE:
            usable_filtered.append(f)
    except Exception:
        pass

if len(usable_filtered) != len(usable):
    print(f"Filtered out {len(usable) - len(usable_filtered)} weak/idle clips.")
usable = usable_filtered
labels = [infer_label_from_path(f, CONFIG["CLASSES"]) for f in usable]

# build dataframe & group-aware splits
df = pd.DataFrame({"path": usable, "label": labels})
df["group"]  = df["path"].apply(base_group)
df["is_aug"] = df["path"].apply(is_augmented)

df_base = df[~df["is_aug"]].copy()
assert len(df_base) > 0, "No base files found."

grp_lbl_unique = df_base.groupby("group")["label"].nunique()
if not (grp_lbl_unique == 1).all():
    bad = grp_lbl_unique[grp_lbl_unique != 1]
    raise ValueError(f"Mixed labels within base groups: {bad.index.tolist()[:8]} ...")

group_labels_df = df_base.groupby("group")["label"].first().reset_index()
all_groups = group_labels_df["group"].tolist()
all_group_labels = group_labels_df["label"].tolist()

from sklearn.model_selection import train_test_split
TEST_SPLIT = CONFIG["TEST_SPLIT"]
VAL_SPLIT  = CONFIG["VAL_SPLIT"]

grp_trainval, grp_test = train_test_split(
    all_groups,
    test_size=TEST_SPLIT,
    stratify=all_group_labels,
    random_state=SEED
)

remain_mask   = group_labels_df["group"].isin(grp_trainval)
remain_groups = group_labels_df.loc[remain_mask, "group"].tolist()
remain_labels = group_labels_df.loc[remain_mask, "label"].tolist()

val_ratio_of_trainval = VAL_SPLIT / (1.0 - TEST_SPLIT)
grp_train, grp_val = train_test_split(
    remain_groups,
    test_size=val_ratio_of_trainval,
    stratify=remain_labels,
    random_state=SEED
)

grp_train, grp_val, grp_test = set(grp_train), set(grp_val), set(grp_test)

train_files = df[df["group"].isin(grp_train) & (~df["is_aug"])]["path"].tolist()
val_files   = df[df["group"].isin(grp_val)   & (~df["is_aug"])]["path"].tolist()
test_files  = df[df["group"].isin(grp_test)  & (~df["is_aug"])]["path"].tolist()
print("Split sizes (no group overlap):")
print("  Train:", len(train_files))
print("  Val:  ", len(val_files))
print("  Test: ", len(test_files))

def files_to_groups(files): return {base_group(p) for p in files}
assert set(train_files).isdisjoint(val_files)
assert set(train_files).isdisjoint(test_files)
assert set(val_files).isdisjoint(test_files)

def compute_feature_stats(paths, max_files=None):
    xs = []
    for i, f in enumerate(paths):
        if (max_files is not None) and (i >= max_files): break
        a = np.load(f).astype(np.float32)
        xs.append(a)
    X = np.concatenate(xs, axis=0) if len(xs) else np.zeros((1, FEATURE_DIM), np.float32)
    mv = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    sv = X.std(axis=0,  dtype=np.float64).astype(np.float32)
    sv[sv < 1e-6] = 1.0
    return mv, sv

train_mean, train_std = compute_feature_stats(train_files)

train_ds = KeypointSequenceDataset(train_files, CONFIG["CLASSES"], CONFIG["SEQUENCE_LENGTH"],
                                   split="train", mean_vec=train_mean, std_vec=train_std)
val_ds   = KeypointSequenceDataset(val_files,   CONFIG["CLASSES"], CONFIG["SEQUENCE_LENGTH"],
                                   split="val",   mean_vec=train_mean, std_vec=train_std)
test_ds  = KeypointSequenceDataset(test_files,  CONFIG["CLASSES"], CONFIG["SEQUENCE_LENGTH"],
                                   split="test",  mean_vec=train_mean, std_vec=train_std)

train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                          collate_fn=collate_batch, num_workers=0, pin_memory=False,
                          persistent_workers=False)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                          collate_fn=collate_batch, num_workers=0, pin_memory=False,
                          persistent_workers=False)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                          collate_fn=collate_batch, num_workers=0, pin_memory=False,
                          persistent_workers=False)

cnt = Counter([infer_label_from_path(p, CONFIG["CLASSES"]) for p in train_files])
class_weights = torch.tensor([1.0/max(1, cnt[i]) for i in range(len(CONFIG["CLASSES"]))], dtype=torch.float32)

# %% [markdown]
# # Modified LSTM

# %%
INPUT_SIZE  = CONFIG["FEATURE_DIM"]
NUM_CLASSES = len(CONFIG["CLASSES"])

model = build_modified_lstm(NUM_CLASSES, input_size=INPUT_SIZE,
                            hidden_size=256, num_layers=2,
                            dropout=CONFIG["DROPOUT"], use_layernorm=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"ModifiedLSTM params: {total_params/1e3:.1f}K  |  Input dim = {INPUT_SIZE}")

# %%
FLAG_START = CONFIG["FLAG_START"]
FLAG_END   = CONFIG["FLAG_END"]

# === Training + Inference Time + Last-5 Avg + Model Size (MB) ===
def train_run(TRAIN_FROM_SCRATCH=True, prefix="run"):
    CONFIG["BEST_DIR"] = r"C:\Users\ACER NITRO\FSL\Training\Modified_LSTM\ModifiedLSTM_best"
    best_dir = Path(CONFIG["BEST_DIR"]); best_dir.mkdir(parents=True, exist_ok=True)

    run_id = _next_run_id(best_dir, prefix=prefix) if TRAIN_FROM_SCRATCH else _latest_run_id(best_dir, prefix=prefix) or 1
    ckpt_name, weights_name = f"{prefix}{run_id}.pt", f"{prefix}{run_id}.pth"
    ckpt_path, weights_path = best_dir / ckpt_name, best_dir / weights_name
    export_dir = Path(CONFIG["EXPORT_DIR"]) / f"{prefix}{run_id}"; export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training for {CONFIG['EPOCHS']} epochs… ({prefix}{run_id})")

    criterion  = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer  = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    use_cuda   = torch.cuda.is_available()
    scaler     = GradScaler(enabled=use_cuda)
    history = defaultdict(list)
    best_val_loss, best_epoch, no_improve = float("inf"), -1, 0

    def run_one_epoch(loader, train=True):
        model.train(train)
        total_loss = 0; total = 0; correct = 0; inf_times = []
        for x, lens, y, _paths in loader:
            x, y = x.to(device).float(), y.to(device).long()
            reset_mask = (x[..., FLAG_START:FLAG_END].sum(dim=-1) > 0).float()
            if train: optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_cuda):
                t0 = time.time()
                logits = model(x, reset_mask=reset_mask)
                t1 = time.time()
                loss = criterion(logits, y)
            if not train: inf_times.append(t1 - t0)
            if train:
                if use_cuda:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(1)
                correct += (pred == y).sum().item(); total += y.size(0)
                total_loss += loss.item() * y.size(0)
        inf_ms = (np.mean(inf_times)*1000) if inf_times else 0
        return total_loss/max(1,total), correct/max(1,total), inf_ms

    print("\n=== Training started ===")
    t0 = time.time()
    for epoch in range(CONFIG["EPOCHS"]):
        tr_loss, tr_acc, _ = run_one_epoch(train_loader, True)
        with torch.no_grad():
            va_loss, va_acc, va_inf = run_one_epoch(val_loader, False)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history.setdefault("val_inf_time", []).append(va_inf)
        tag = ""
        if va_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch, no_improve = va_loss, epoch, 0
            torch.save(model.state_dict(), weights_path)
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG, "best_val_loss": float(best_val_loss),
                "history": {k:[float(v) for v in vals] for k,vals in history.items()},
            }, ckpt_path)
            tag = f"  <-- BEST (saved {ckpt_name})"
        else:
            no_improve += 1
            if no_improve >= CONFIG["PATIENCE"]:
                print(f"Early stopping (no improve {CONFIG['PATIENCE']} epochs, best@{best_epoch})")
                break
        print(f"Epoch {epoch+1:03d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | inf {va_inf:.2f}ms{tag}")

    print("\n=== [SUMMARY] ===")
    print(f"Best epoch: {best_epoch}, best val loss: {best_val_loss:.4f}")
    print("========================")
    with open(export_dir / "train_history.json", "w") as f:
        json.dump({k:[float(v) for v in vls] for k,vls in history.items()}, f)

train_run(TRAIN_FROM_SCRATCH=True)

# %% [markdown]
# # Training with hyperparameter tuning

# %%
HPO_SPACE = {
    "optimizer": ["adam", "adamw"],
    "lr": [7e-4, 5e-4, 3e-4, 2e-4, 1e-4],
    "dropout": [0.40, 0.30, 0.35],
    "weight_decay": [0.0, 1e-6, 1e-5],
    "hidden_size": [212, 234, 256, 280, 300],
    "num_layers": [2],
    "label_smoothing": [0.0, 0.05],
    "scheduler": ["cosine", "plateau"],
}

def sample_candidates(space, n=20, seed=2025):
    rng = random.Random(seed)
    keys = list(space.keys())
    return [{k: rng.choice(space[k]) for k in keys} for _ in range(n)]

candidates = sample_candidates(HPO_SPACE, n=50)
print(f"Testing {len(candidates)} candidate hyperparameter sets")

def quick_eval(hparams, max_epochs=80):
    local_config = deepcopy(CONFIG)
    local_config.update({
        "LR": hparams["lr"],
        "WEIGHT_DECAY": hparams["weight_decay"],
        "DROPOUT": hparams["dropout"],
    })

    model = build_modified_lstm(
        len(local_config["CLASSES"]),
        input_size=local_config["FEATURE_DIM"],
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        use_layernorm=True
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=hparams["label_smoothing"]
    )

    if hparams["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )

    if hparams["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=hparams["lr"] * 0.1
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4
        )

    scaler = GradScaler(enabled=torch.cuda.is_available())
    best_val_loss = float("inf")
    patience = 15
    bad_epochs = 0
    start = time.time()

    for epoch in range(max_epochs):
        model.train()
        for x, lens, y, _paths in train_loader:
            x, y = x.to(device).float(), y.to(device).long()
            reset_mask = (x[..., CONFIG["FLAG_START"]:CONFIG["FLAG_END"]].sum(dim=-1) > 0).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                logits = model(x, reset_mask=reset_mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for x, lens, y, _paths in val_loader:
                x, y = x.to(device).float(), y.to(device).long()
                reset_mask = (x[..., CONFIG["FLAG_START"]:CONFIG["FLAG_END"]].sum(dim=-1) > 0).float()
                logits = model(x, reset_mask=reset_mask)
                loss = criterion(logits, y)
                total_loss += loss.item() * y.size(0)
                total += y.size(0)
        val_loss = total_loss / max(1, total)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stop @ epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{max_epochs} val_loss={val_loss:.4f}")

    elapsed = time.time() - start
    return best_val_loss, elapsed

# %% [markdown]
# The best hyperparameters found were
# Best hyperparameters found:
# {
#     "optimizer": "adamw",
#     "lr": 0.0003,
#     "dropout": 0.35,
#     "weight_decay": 1e-05,
#     "hidden_size": 384,
#     "num_layers": 2,
#     "label_smoothing": 0.0,
#     "scheduler": "plateau"
# }
# Best validation loss: 0.1485 (trained 18.9s)

# %%
hpo_start = time.time()
results = []

for i, hparams in enumerate(candidates, 1):
    print(f"\nCandidate {i}/{len(candidates)}")
    print(hparams)

    candidate_start = time.time()
    val_loss, elapsed = quick_eval(hparams, max_epochs=80)
    candidate_time = time.time() - candidate_start

    results.append((val_loss, hparams, candidate_time))

    print(f"Validation loss: {val_loss:.4f}")
    print(f"Candidate time: {candidate_time:.2f} sec")

hpo_total_time = time.time() - hpo_start

results.sort(key=lambda x: x[0])
best_loss, best_hparams, best_time = results[0]

print(json.dumps(best_hparams, indent=4))
print(f"Best validation loss: {best_loss:.4f}")
print(f"Best candidate training time: {best_time:.2f} sec")
print(f"Total HPO time: {hpo_total_time/60:.2f} minutes")

# %%
FLAG_START = CONFIG["FLAG_START"]
FLAG_END   = CONFIG["FLAG_END"]

def train_run(best_hparams, TRAIN_FROM_SCRATCH=True, prefix="run"):
    CONFIG["BEST_DIR"] = r"C:\Users\ACER NITRO\FSL\Training\Modified_LSTM\ModifiedLSTM_best"
    best_dir = Path(CONFIG["BEST_DIR"])
    best_dir.mkdir(parents=True, exist_ok=True)

    run_id = _next_run_id(best_dir, prefix=prefix) if TRAIN_FROM_SCRATCH \
             else _latest_run_id(best_dir, prefix=prefix) or 1

    ckpt_name = f"{prefix}{run_id}.pt"
    weights_name = f"{prefix}{run_id}.pth"
    ckpt_path = best_dir / ckpt_name
    weights_path = best_dir / weights_name

    export_dir = Path(CONFIG["EXPORT_DIR"]) / f"{prefix}{run_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 Apply best hyperparameters from HPO
    CONFIG.update({
        "LR": best_hparams["lr"],
        "DROPOUT": best_hparams["dropout"],
        "WEIGHT_DECAY": best_hparams["weight_decay"],
        "HIDDEN_SIZE": best_hparams["hidden_size"],
        "NUM_LAYERS": best_hparams["num_layers"],
        "LABEL_SMOOTHING": best_hparams["label_smoothing"],
        "OPTIMIZER": best_hparams["optimizer"],
        "SCHEDULER": best_hparams["scheduler"],
        "EPOCHS": 120,
        "PATIENCE": 20,
    })

    # Build model
    model = build_modified_lstm(
        len(CONFIG["CLASSES"]),
        input_size=CONFIG["FEATURE_DIM"],
        hidden_size=CONFIG["HIDDEN_SIZE"],
        num_layers=CONFIG["NUM_LAYERS"],
        dropout=CONFIG["DROPOUT"],
        use_layernorm=True
    ).to(device)

    print(f"\nTraining run {prefix}{run_id}")
    print(json.dumps(best_hparams, indent=4))

    # Loss
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=CONFIG["LABEL_SMOOTHING"]
    )

    # Optimizer
    if CONFIG["OPTIMIZER"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=CONFIG["LR"],
            weight_decay=CONFIG["WEIGHT_DECAY"]
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["LR"],
            weight_decay=CONFIG["WEIGHT_DECAY"]
        )

    # Scheduler
    if CONFIG["SCHEDULER"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=CONFIG["LR"] * 0.1
        )
    elif CONFIG["SCHEDULER"] == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5
        )
    else:
        scheduler = None

    use_cuda = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_cuda)

    history = defaultdict(list)
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    print("\nTraining started")
    t0 = time.time()

    for epoch in range(CONFIG["EPOCHS"]):

        epoch_start = time.time()
        # ---- TRAIN ----
        model.train()
        total_loss, total, correct = 0, 0, 0

        for x, lens, y, _paths in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            reset_mask = (x[..., CONFIG["FLAG_START"]:CONFIG["FLAG_END"]]
                          .sum(dim=-1) > 0).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_cuda):
                logits = model(x, reset_mask=reset_mask)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

        tr_loss = total_loss / total
        tr_acc = correct / total

        # ---- VALIDATION ----
        model.eval()
        total_loss, total, correct = 0, 0, 0

        with torch.no_grad():
            for x, lens, y, _paths in val_loader:
                x = x.to(device).float()
                y = y.to(device).long()

                reset_mask = (x[..., CONFIG["FLAG_START"]:CONFIG["FLAG_END"]]
                              .sum(dim=-1) > 0).float()

                logits = model(x, reset_mask=reset_mask)
                loss = criterion(logits, y)

                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                total_loss += loss.item() * y.size(0)

        va_loss = total_loss / total
        va_acc = correct / total

        # ---- Scheduler step (correctly handled) ----
        if scheduler is not None:
            if CONFIG["SCHEDULER"] == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step(epoch)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        tag = ""
        if va_loss < best_val_loss - 1e-4:
            best_val_loss = va_loss
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), weights_path)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG,
                "best_val_loss": float(best_val_loss),
                "history": history,
            }, ckpt_path)

            tag = " <-- best saved"
        else:
            no_improve += 1
            if no_improve >= CONFIG["PATIENCE"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:03d} | "
            f"train {tr_loss:.4f}/{tr_acc:.4f} | "
            f"val {va_loss:.4f}/{va_acc:.4f} | "
            f"{epoch_time:.2f}s{tag}")

    total_time = time.time() - t0
    avg_epoch_time = total_time / max(1, (epoch+1))

    last5 = history["val_acc"][-5:] if len(history["val_acc"]) >= 5 else history["val_acc"]
    last5_avg = sum(last5) / len(last5)

    model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Last-5 Val Acc Avg: {last5_avg:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average epoch time: {avg_epoch_time:.2f} sec")
    print(f"Model size: {model_size_mb:.2f} MB")

best_loss, best_hparams, _ = results[0]
train_run(best_hparams, TRAIN_FROM_SCRATCH=True)

# %% [markdown]
# # Results

# %%
def _coerce_float_list(v):
    try:
        return [float(x) for x in (v if isinstance(v, (list, tuple)) else [])]
    except Exception:
        return []

def load_history_json(json_path):
    """Safe loader for train_history.json when you DO want to load from disk."""
    try:
        with open(json_path, "r") as f:
            hist = json.load(f)
        hist_out = {}
        for k in ("train_acc","val_acc","train_loss","val_loss"):
            hist_out[k] = _coerce_float_list(hist.get(k, []))
        return hist_out
    except Exception as e:
        print(f"[plot] Could not load history from {json_path}: {e}")
        return None

def plot_history(history=None, json_path=None, save_dir=None, title_suffix=""):
    """
    Plot accuracy & loss using decimals (0..1). Prefer `history` (in-memory).
    If `history` is None, tries to load from `json_path`.
    Titles have no run IDs.
    """
    if history is None:
        history = load_history_json(json_path)
    if not history:
        print("[plot] No history found to plot.")
        return

    train_acc = _coerce_float_list(history.get("train_acc"))
    val_acc   = _coerce_float_list(history.get("val_acc"))
    train_ls  = _coerce_float_list(history.get("train_loss"))
    val_ls    = _coerce_float_list(history.get("val_loss"))

    n = max(len(train_acc), len(val_acc), len(train_ls), len(val_ls))
    if n == 0:
        print("[plot] History lists are empty.")
        return
    epochs = list(range(1, n+1))

    def _pad(arr, n):
        arr = list(arr)
        if len(arr) < n:
            arr = arr + [np.nan]*(n-len(arr))
        return arr[:n]

    train_acc = _pad(train_acc, n)
    val_acc   = _pad(val_acc,   n)
    train_ls  = _pad(train_ls,  n)
    val_ls    = _pad(val_ls,    n)

    # --- Accuracy ---
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, train_acc, label="Train Acc")
    plt.plot(epochs, val_acc,   label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (fraction)")
    plt.title("Accuracies" + (f" — {title_suffix}" if title_suffix else ""))
    plt.grid(True, alpha=0.3); plt.legend(loc="lower right")
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        acc_path = str(Path(save_dir) / "acc_curve.png")
        plt.tight_layout(); plt.savefig(acc_path, dpi=150)
        print("[plot] Saved:", acc_path)
    plt.show()

    # --- Loss ---
    plt.figure(figsize=(7,4.5))
    plt.plot(epochs, train_ls, label="Train Loss")
    plt.plot(epochs, val_ls,   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Losses" + (f" — {title_suffix}" if title_suffix else ""))
    plt.grid(True, alpha=0.3); plt.legend(loc="upper right")
    if save_dir:
        loss_path = str(Path(save_dir) / "loss_curve.png")
        plt.tight_layout(); plt.savefig(loss_path, dpi=150)
        print("[plot] Saved:", loss_path)
    plt.show()

cands = glob.glob(os.path.join(CONFIG["EXPORT_DIR"], "run*", "train_history.json"))
if not cands:
    print("[plot] No train_history.json found under", CONFIG["EXPORT_DIR"])
else:
    json_path = max(cands, key=os.path.getmtime)
    save_dir  = os.path.dirname(json_path)
    print("[plot] Using history:", json_path)
    plot_history(history=None, json_path=json_path, save_dir=save_dir, title_suffix="Modified LSTM")

# %%
def latest_ckpt(best_dir, prefix="run"):
    best_dir = Path(best_dir)
    if not best_dir.exists():
        return None, None
    cands = []
    for f in best_dir.iterdir():
        m = re.match(rf"^{re.escape(prefix)}(\d+)\.pt$", f.name)
        if m:
            cands.append((int(m.group(1)), f))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[0])
    return cands[-1]

BEST_DIR = CONFIG["BEST_DIR"]
run_id, ckpt_path = latest_ckpt(BEST_DIR)

assert ckpt_path is not None, f"No checkpoint found in {BEST_DIR}"

print(f"[eval] Loading checkpoint: {ckpt_path.name}")

checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

saved_config = checkpoint["config"]

# Build model exactly as trained
test_model = build_modified_lstm(
    len(saved_config["CLASSES"]),
    input_size=saved_config["FEATURE_DIM"],
    hidden_size=saved_config["HIDDEN_SIZE"],
    num_layers=saved_config["NUM_LAYERS"],
    dropout=saved_config["DROPOUT"],
    use_layernorm=True
).to(device)

test_model.load_state_dict(checkpoint["model_state_dict"])
test_model.eval()

print(f"[eval] Loaded run {run_id:03d}")
print("Model params:",
      f"hidden={saved_config['HIDDEN_SIZE']},",
      f"layers={saved_config['NUM_LAYERS']},",
      f"dropout={saved_config['DROPOUT']}")

def _tta_variants(x_np, n_nonflip=3):
    T, D = x_np.shape
    FEAT = BASE_HAND_DIM
    coords = x_np[:, :FEAT].reshape(T, 42, 3).copy()
    extra = x_np[:, FEAT:].copy() if D > FEAT else None
    outs = [x_np.astype(np.float32)]
    for _ in range(n_nonflip):
        c = coords.copy()
        ang = np.deg2rad(np.random.uniform(-5, 5))
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        xy = c[..., :2].reshape(-1, 2) @ R.T
        c[..., :2] = xy.reshape(T, 42, 2)
        s = np.random.uniform(0.97, 1.05)
        c *= np.array([s, s, 1.0], np.float32)
        flat = np.concatenate([c.reshape(T, FEAT), extra], axis=1) if extra is not None else c.reshape(T, FEAT)
        outs.append(flat.astype(np.float32))
    return outs

FLAG_START, FLAG_END = CONFIG["FLAG_START"], CONFIG["FLAG_END"]
all_preds, all_labels = [], []

with torch.no_grad():
    for x, lens, y, _ in test_loader:
        B, T, D = x.shape
        x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
        logits_sum = torch.zeros((B, len(CONFIG["CLASSES"])), device=device)
        for b in range(B):
            for v in _tta_variants(x_np[b]):
                v_t = torch.from_numpy(v).float().unsqueeze(0).to(device)
                rm = torch.from_numpy((v[:, FLAG_START:FLAG_END].sum(axis=1) > 0).astype(np.float32)).unsqueeze(0).to(device)
                logits_sum[b] += test_model(v_t, reset_mask=rm).squeeze(0)
        all_preds.append(logits_sum.argmax(1).cpu().numpy())
        all_labels.append(y_np)

all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
test_acc = (all_preds == all_labels).mean() * 100
print(f"TTA Test Accuracy: {test_acc:.2f}%")

labels_names = CONFIG["CLASSES"]
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels_names))))
fig, ax = plt.subplots(figsize=(9, 9))
disp = ConfusionMatrixDisplay(cm, display_labels=labels_names)
disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
plt.xticks(rotation=60, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.title(f"Confusion Matrix (TTA) — Run {run_id:03d} (Acc {test_acc:.2f}%)", pad=20)
plt.tight_layout()
cm_path = Path(CONFIG["EXPORT_DIR"]) / f"cm_tta_run{run_id:03d}.png"
plt.savefig(cm_path, dpi=200)
plt.show()

# --- 95 % CI and report ---
n = len(all_labels); p = test_acc / 100; z = 1.96
den = 1 + z*z/n
center = (p + z*z/(2*n)) / den
half = z * sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / den
print(f"Test Acc 95% CI (Wilson): {(center-half)*100:.2f}% .. {(center+half)*100:.2f}%\n")
print("Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=labels_names, digits=3))
print("Saved confusion matrix to:", cm_path)

# %%
def latest_ckpt(best_dir, prefix="run"):
    best_dir = Path(best_dir)
    if not best_dir.exists():
        return None, None

    candidates = []
    for f in best_dir.iterdir():
        m = re.match(rf"^{re.escape(prefix)}(\d+)\.(pt|pth)$", f.name)
        if m:
            candidates.append((int(m.group(1)), f))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


# --- locate latest checkpoint ---
BEST_DIR = CONFIG["BEST_DIR"]
run_id, ckpt_path = latest_ckpt(BEST_DIR)
assert ckpt_path is not None, f"No checkpoint found under {BEST_DIR}"

print(f"[eval] Loading checkpoint: {ckpt_path.name}")


# --- load safely (PyTorch 2.6 fix) ---
checkpoint = torch.load(
    str(ckpt_path),
    map_location=device,
    weights_only=False
)

# determine format
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    saved_config = checkpoint["config"]
else:
    # pure .pth weights only
    state_dict = checkpoint
    saved_config = CONFIG  # fallback


# --- rebuild model using saved hyperparameters ---
test_model = build_modified_lstm(
    len(saved_config["CLASSES"]),
    input_size=saved_config["FEATURE_DIM"],
    hidden_size=saved_config["HIDDEN_SIZE"],
    num_layers=saved_config["NUM_LAYERS"],
    dropout=saved_config["DROPOUT"],
    use_layernorm=True
).to(device)

test_model.load_state_dict(state_dict)
test_model.eval()

print(
    f"[eval] Run {run_id:03d} | "
    f"hidden={saved_config['HIDDEN_SIZE']} | "
    f"layers={saved_config['NUM_LAYERS']} | "
    f"dropout={saved_config['DROPOUT']}"
)


# --- evaluate ---
FLAG_START, FLAG_END = saved_config["FLAG_START"], saved_config["FLAG_END"]

all_preds, all_labels = [], []

with torch.no_grad():
    for x, lens, y, _ in test_loader:
        x, y = x.to(device).float(), y.to(device).long()
        reset_mask = (x[..., FLAG_START:FLAG_END].sum(dim=-1) > 0).float()
        logits = test_model(x, reset_mask=reset_mask)

        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

acc = (all_preds == all_labels).mean() * 100
print(f"\nPlain Test Accuracy: {acc:.2f}%")


# --- confusion matrix ---
labels_names = saved_config["CLASSES"]
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(labels_names))))

fig, ax = plt.subplots(figsize=(9, 9))
disp = ConfusionMatrixDisplay(cm, display_labels=labels_names)
disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")

plt.xticks(rotation=60, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.title(f"Confusion Matrix — Run {run_id:03d} (Acc {acc:.2f}%)", pad=20)
plt.tight_layout()

cm_path = Path(CONFIG["EXPORT_DIR"]) / f"cm_plain_run{run_id:03d}.png"
plt.savefig(cm_path, dpi=200)
plt.show()

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=labels_names, digits=3))
print("Saved confusion matrix to:", cm_path)


