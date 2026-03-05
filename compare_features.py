"""
Compare training .npy features with live inference features.
Run this AFTER doing a live prediction — it reads /tmp_live_seq.npy saved by app.py.
"""
import numpy as np
import os, glob

KEYPOINTS_DIR = r"Training\KEYPOINTS\Family_Mother"
LIVE_FILE = "tmp_live_seq.npy"

def describe(arr, label):
    """Print summary stats for a (seq_len, feat_dim) array."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  shape={arr.shape}  dtype={arr.dtype}")
    print(f"{'='*60}")
    mid = arr.shape[0] // 2
    frame = arr[mid]
    
    # Feature layout: L_hand(0:63), R_hand(63:126), derived(126:198), flags(198:200)
    L  = frame[0:63]
    R  = frame[63:126]
    D  = frame[126:198]
    fl = frame[198:200]
    
    print(f"\n  Middle frame (idx={mid}):")
    print(f"  L_hand  mean={L.mean():.4f}  std={L.std():.4f}  min={L.min():.4f}  max={L.max():.4f}  nonzero={np.count_nonzero(L)}/63")
    print(f"  R_hand  mean={R.mean():.4f}  std={R.std():.4f}  min={R.min():.4f}  max={R.max():.4f}  nonzero={np.count_nonzero(R)}/63")
    print(f"  Derived mean={D.mean():.4f}  std={D.std():.4f}  min={D.min():.4f}  max={D.max():.4f}  nonzero={np.count_nonzero(D)}/72")
    print(f"  Flags   = [{fl[0]:.0f}, {fl[1]:.0f}]")
    
    # Print derived in 6-value chunks (12 joints × 6 features)
    # Layout: for each of 12 joints (6 L + 6 R): [dx_chin, dy_chin, dy_lipU, dy_brow, dy_fore, dz_nose]
    print(f"\n  Derived breakdown (6 per joint, 12 joints):")
    names = ["dx_chin", "dy_chin", "dy_lipU", "dy_brow", "dy_fore", "dz_nose"]
    for i in range(12):
        chunk = D[i*6:(i+1)*6]
        side = "L" if i < 6 else "R"
        ji = i if i < 6 else i - 6
        joint_idx = [0,4,8,12,16,20][ji]
        vals = "  ".join(f"{names[k]}={chunk[k]:+.4f}" for k in range(6))
        print(f"    {side} joint[{joint_idx:2d}]: {vals}")
    
    # Sequence-level stats
    print(f"\n  Sequence stats (all frames):")
    print(f"  Overall  mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"  Per-frame norm: min={np.linalg.norm(arr,axis=1).min():.2f}  max={np.linalg.norm(arr,axis=1).max():.2f}")

def main():
    # Load training samples
    npy_files = sorted(glob.glob(os.path.join(KEYPOINTS_DIR, "*.npy")))
    if not npy_files:
        print(f"[!] No .npy files found in {KEYPOINTS_DIR}")
        return
    
    # Show first training sample
    train = np.load(npy_files[0])
    describe(train, f"TRAINING: {npy_files[0]}")
    
    # Average across all training samples for Mother
    all_mids = []
    for f in npy_files:
        d = np.load(f)
        mid = d.shape[0] // 2
        all_mids.append(d[mid])
    avg = np.mean(all_mids, axis=0)
    print(f"\n{'='*60}")
    print(f"  AVG across {len(npy_files)} training Mother samples (middle frame)")
    D_avg = avg[126:198]
    names = ["dx_chin", "dy_chin", "dy_lipU", "dy_brow", "dy_fore", "dz_nose"]
    for i in range(12):
        chunk = D_avg[i*6:(i+1)*6]
        side = "L" if i < 6 else "R"
        ji = i if i < 6 else i - 6
        joint_idx = [0,4,8,12,16,20][ji]
        vals = "  ".join(f"{names[k]}={chunk[k]:+.4f}" for k in range(6))
        print(f"    {side} joint[{joint_idx:2d}]: {vals}")
    
    # Load live prediction if available
    if os.path.exists(LIVE_FILE):
        live = np.load(LIVE_FILE)
        describe(live, f"LIVE INFERENCE: {LIVE_FILE}")
        
        # Compare middle frames
        mid_t = train.shape[0] // 2
        mid_l = live.shape[0] // 2
        diff = live[mid_l] - train[mid_t]
        print(f"\n{'='*60}")
        print(f"  DIFF (live - training) middle frame")
        print(f"  L_hand  diff_mean={diff[0:63].mean():.4f}   diff_max={np.abs(diff[0:63]).max():.4f}")
        print(f"  R_hand  diff_mean={diff[63:126].mean():.4f}  diff_max={np.abs(diff[63:126]).max():.4f}")
        print(f"  Derived diff_mean={diff[126:198].mean():.4f}  diff_max={np.abs(diff[126:198]).max():.4f}")
        
        D_diff = diff[126:198]
        print(f"\n  Derived DIFF per joint:")
        for i in range(12):
            chunk = D_diff[i*6:(i+1)*6]
            side = "L" if i < 6 else "R"
            ji = i if i < 6 else i - 6
            joint_idx = [0,4,8,12,16,20][ji]
            vals = "  ".join(f"{names[k]}={chunk[k]:+.4f}" for k in range(6))
            print(f"    {side} joint[{joint_idx:2d}]: {vals}")
    else:
        print(f"\n[!] No live data yet. Do a prediction first, then re-run this script.")
        print(f"    (app.py will save to {LIVE_FILE})")

if __name__ == "__main__":
    main()
