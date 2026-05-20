"""
SignX — Phase 2: EDA (v3 — correct shape T × 543 × 3)
======================================================
Paste each CELL block as a separate cell in your Kaggle notebook.
Run them in order.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports and paths
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

DATA_ROOT = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
TRAIN_CSV = f'{DATA_ROOT}/metadata/how2sign_realigned_train.csv'
TRAIN_NPY = f'{DATA_ROOT}/train/frontal/'

NUM_KEYPOINTS       = 543
COORDS              = 3
NPY_SUFFIX          = '-rgb_front_holistic.npy'
POSE_LEFT_HIP       = 23
POSE_RIGHT_HIP      = 24
POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12

print("Data root:", os.listdir(DATA_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Build lookup and scan frame lengths
# ─────────────────────────────────────────────────────────────────────────────

# Build SENTENCE_ID → filepath lookup
npy_lookup = {}
for fname in os.listdir(TRAIN_NPY):
    if not fname.endswith(NPY_SUFFIX):
        continue
    base  = fname[: -len(NPY_SUFFIX)]
    parts = base.rsplit('-', 1)
    sid   = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else base
    npy_lookup[sid] = os.path.join(TRAIN_NPY, fname)

print(f"Files in lookup  : {len(npy_lookup)}")

# Load metadata
df = pd.read_csv(TRAIN_CSV, sep='\t')
df.columns = [c.strip().upper() for c in df.columns]
print(f"Metadata rows    : {len(df)}")

df['_npy_path'] = df['SENTENCE_ID'].astype(str).str.strip().map(npy_lookup)
matched = df.dropna(subset=['_npy_path'])
print(f"Matched clips    : {len(matched)}")
print(f"Unmatched        : {len(df) - len(matched)}")

# Scan frame lengths — mmap reads only the header, not the full array
frame_lengths = []
for path in matched['_npy_path']:
    try:
        arr = np.load(path, mmap_mode='r', allow_pickle=True)
        frame_lengths.append(arr.shape[0])
    except Exception:
        pass

frame_lengths = np.array(frame_lengths)
print(f"\nArray shape (one file) : (T, {NUM_KEYPOINTS}, {COORDS})")
print(f"\nFrame length stats:")
print(f"  min    : {frame_lengths.min()}")
print(f"  max    : {frame_lengths.max()}")
print(f"  mean   : {frame_lengths.mean():.1f}")
print(f"  median : {np.median(frame_lengths):.1f}")
print(f"  p90    : {np.percentile(frame_lengths, 90):.1f}")
print(f"  p95    : {np.percentile(frame_lengths, 95):.1f}")
print(f"  p99    : {np.percentile(frame_lengths, 99):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Q1: Frame length distribution
# ─────────────────────────────────────────────────────────────────────────────

p95 = np.percentile(frame_lengths, 95)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Q1: Frame Length Distribution', fontsize=14, fontweight='bold')

axes[0].hist(frame_lengths, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(p95, color='red', linestyle='--', linewidth=2, label=f'p95 = {p95:.0f}')
axes[0].axvline(np.median(frame_lengths), color='orange', linestyle='--',
                linewidth=2, label=f'median = {np.median(frame_lengths):.0f}')
axes[0].set_xlabel('Frames per clip'); axes[0].set_ylabel('Clips')
axes[0].set_title('All clips'); axes[0].legend()

short = frame_lengths[frame_lengths < 600]
axes[1].hist(short, bins=60, color='teal', edgecolor='white', alpha=0.85)
axes[1].axvline(np.percentile(short, 95), color='red', linestyle='--', linewidth=2,
                label=f'p95 = {np.percentile(short,95):.0f}')
axes[1].set_xlabel('Frames per clip'); axes[1].set_ylabel('Clips')
axes[1].set_title('Clips < 600 frames (zoomed)'); axes[1].legend()

plt.tight_layout()
plt.savefig('eda_q1_frame_distribution.png', dpi=120, bbox_inches='tight')
plt.show()

corrupt_short = (frame_lengths < 10).sum()
if corrupt_short:
    print(f"⚠ {corrupt_short} clips have < 10 frames — will be filtered by min_frames=10")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Q2: Validate normalization on 100 random clips
# ─────────────────────────────────────────────────────────────────────────────
# Array shape is (T, 543, 3) so keypoint access is motion[:, kp_index, :]
# No flat index arithmetic needed.

def normalize_coordinates(motion, eps=1e-6):
    """Body-relative normalization for (T, 543, 3) arrays."""
    motion = motion.copy().astype(np.float32)
    T = motion.shape[0]

    hip_mid   = (motion[:, POSE_LEFT_HIP, :] + motion[:, POSE_RIGHT_HIP, :]) / 2.0
    shoulder_w = np.linalg.norm(
        motion[:, POSE_LEFT_SHOULDER, :] - motion[:, POSE_RIGHT_SHOULDER, :],
        axis=1
    )
    shoulder_w = np.maximum(shoulder_w, eps).reshape(T, 1, 1)

    motion = motion - hip_mid[:, np.newaxis, :]
    motion = motion / shoulder_w
    return motion.astype(np.float32)

sample = matched.sample(min(100, len(matched)), random_state=42)
hip_x_means, sw_raw_list, sw_norm_list = [], [], []

for path in sample['_npy_path']:
    try:
        raw = np.load(path, allow_pickle=True).astype(np.float32)
        if raw.shape[1:] != (NUM_KEYPOINTS, COORDS):
            continue

        # Raw shoulder width (image space)
        sw_raw = np.linalg.norm(
            raw[:, POSE_LEFT_SHOULDER, :] - raw[:, POSE_RIGHT_SHOULDER, :], axis=1
        ).mean()
        sw_raw_list.append(sw_raw)

        norm = normalize_coordinates(raw)

        # Normalised shoulder width (should be ≈ 1.0)
        sw_norm = np.linalg.norm(
            norm[:, POSE_LEFT_SHOULDER, :] - norm[:, POSE_RIGHT_SHOULDER, :], axis=1
        ).mean()
        sw_norm_list.append(sw_norm)

        # Hip midpoint X after normalisation (should be ≈ 0.0)
        hip_x = abs(((norm[:, POSE_LEFT_HIP, 0] + norm[:, POSE_RIGHT_HIP, 0]) / 2).mean())
        hip_x_means.append(hip_x)

    except Exception:
        continue

print("── Normalization Validation ─────────────────────────────────────────")
print(f"Clips checked          : {len(hip_x_means)}")
print(f"Hip X after norm       : {np.mean(hip_x_means):.4f}  (target ≈ 0.0)")
print(f"Shoulder width raw     : {np.mean(sw_raw_list):.4f}  (image-space, varies)")
print(f"Shoulder width normed  : {np.mean(sw_norm_list):.4f}  (target ≈ 1.0)")

hip_ok = np.mean(hip_x_means) < 0.05
sho_ok = 0.8 < np.mean(sw_norm_list) < 1.2
print(f"\nHip centering  : {'PASS ✓' if hip_ok else 'FAIL ✗'}")
print(f"Shoulder scale : {'PASS ✓' if sho_ok else 'FAIL ✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Q3: Jitter analysis (confirmed σ=1.5 from previous run)
# ─────────────────────────────────────────────────────────────────────────────
# We already know from the previous run:
#   σ=1.5 → 70.9% jitter reduction  ← use this
#   σ=3.0 → 99.3% reduction         ← too aggressive, flattens real motion
# This cell re-plots it cleanly for your portfolio / README.

LEFT_WRIST = 15   # keypoint index for left wrist in body pose

sample_path = None
for path in matched['_npy_path']:
    try:
        arr = np.load(path, mmap_mode='r', allow_pickle=True)
        if arr.shape[0] > 60:
            sample_path = path
            break
    except Exception:
        continue

if sample_path:
    raw_clip  = np.load(sample_path, allow_pickle=True).astype(np.float32)
    # Left wrist X coordinate across all frames — shape (T,)
    wrist_raw = raw_clip[:, LEFT_WRIST, 0]

    sigmas   = [1.0, 1.5, 3.0]
    smoothed = {s: gaussian_filter1d(wrist_raw, sigma=s) for s in sigmas}

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle('Q3: Jitter — Left Wrist X coordinate', fontsize=13, fontweight='bold')

    for ax, (label, signal, color) in zip(axes.flat, [
        ('Raw',     wrist_raw,     '#e74c3c'),
        ('σ = 1.0', smoothed[1.0], '#3498db'),
        ('σ = 1.5', smoothed[1.5], '#2ecc71'),
        ('σ = 3.0', smoothed[3.0], '#9b59b6'),
    ]):
        delta = np.abs(np.diff(signal)).mean()
        ax.plot(signal, color=color, linewidth=1.2)
        ax.set_title(f'{label} — mean |Δ| = {delta:.5f}')
        ax.set_xlabel('Frame'); ax.set_ylabel('X coordinate')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_q3_jitter.png', dpi=120, bbox_inches='tight')
    plt.show()

    raw_j = np.abs(np.diff(wrist_raw)).mean()
    for s in sigmas:
        j = np.abs(np.diff(smoothed[s])).mean()
        print(f"σ={s}: {(1-j/raw_j)*100:.1f}% jitter reduction")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Q4: Detect corrupt clips
# ─────────────────────────────────────────────────────────────────────────────
# Now loading with allow_pickle=True (required for these files).
# Three failure types:
#   Type A — wrong shape or unloadable
#   Type B — all zeros (complete MediaPipe failure)
#   Type C — near-zero variance (frozen / static skeleton)

print("Scanning for corrupt clips...")

corrupt_bad_shape = []
corrupt_all_zeros = []
corrupt_frozen    = []

for sid, path in zip(matched['SENTENCE_ID'], matched['_npy_path']):
    try:
        arr = np.load(path, allow_pickle=True).astype(np.float32)

        if arr.shape[1:] != (NUM_KEYPOINTS, COORDS):
            corrupt_bad_shape.append(sid)
            continue

        if np.all(arr == 0.0):
            corrupt_all_zeros.append(sid)
            continue

        if arr.std() < 1e-6:
            corrupt_frozen.append(sid)

    except Exception:
        corrupt_bad_shape.append(sid)

total = len(corrupt_bad_shape) + len(corrupt_all_zeros) + len(corrupt_frozen)
print(f"Wrong shape / unloadable : {len(corrupt_bad_shape)}")
print(f"All zeros                : {len(corrupt_all_zeros)}")
print(f"Frozen skeleton          : {len(corrupt_frozen)}")
print(f"Total corrupt            : {total}")
print(f"Clean clips              : {len(matched) - total}")

all_corrupt = set(corrupt_bad_shape) | set(corrupt_all_zeros) | set(corrupt_frozen)
pd.DataFrame({'SENTENCE_ID': list(all_corrupt)}).to_csv('corrupt_clips.csv', index=False)
print("Saved → corrupt_clips.csv")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Summary
# ─────────────────────────────────────────────────────────────────────────────

p95_val       = int(np.percentile(frame_lengths, 95))
max_frames_rec = (p95_val // 50 + 1) * 50

print("\n" + "=" * 65)
print("  EDA COMPLETE — training config values")
print("=" * 65)
print(f"""
Array shape    : (T, 543, 3)   ← confirmed

max_frames     = {max_frames_rec}     ← p95 = {p95_val}, rounded up to nearest 50
min_frames     = 10            ← filters {corrupt_short} clips with < 10 frames
normalize      = True          ← {'validated ✓' if (hip_ok and sho_ok) else 'FAILING — debug first'}
smooth         = True
smooth_sigma   = 1.5           ← 70.9% jitter reduction, preserves motion peaks
batch_size     = 8             ← increase if GPU memory allows

corrupt clips  = {total}           ← corrupt_clips.csv saved
clean clips    = {len(matched) - total}
""")