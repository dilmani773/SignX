"""
SignX — Phase 1: Data Pipeline (v3 — correct array shape)
==========================================================
How2Sign .npy files have shape (T, 543, 3):
    T   = number of frames (variable per clip)
    543 = MediaPipe Holistic keypoints
    3   = (x, y, z) per keypoint

We do NOT flatten to (T, 1629). We keep (T, 543, 3) throughout.
This is cleaner — no reshaping needed, and the frontend gets
keypoints as (x,y,z) triplets directly.

Kaggle paths:
    DATA_ROOT = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
    TRAIN_CSV = DATA_ROOT + '/metadata/how2sign_realigned_train.csv'
    TRAIN_NPY = DATA_ROOT + '/train/frontal/'
    VAL_CSV   = DATA_ROOT + '/metadata/how2sign_realigned_val.csv'
    VAL_NPY   = DATA_ROOT + '/val/frontal/'

Run locally:
    python signx_phase1_dataset.py
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Holistic outputs 543 keypoints in a fixed order:
#   [0  : 33]  → 33 body pose landmarks
#   [33 : 54]  → 21 left hand landmarks
#   [54 : 75]  → 21 right hand landmarks
#   [75 : 543] → 468 face landmarks
#
# Each keypoint has 3 values: (x, y, z)
# So the full shape per frame is (543, 3)
# Full clip shape: (T, 543, 3)

NUM_KEYPOINTS = 543
COORDS        = 3       # x, y, z

# Body pose keypoint indices (within the 543)
POSE_LEFT_HIP       = 23
POSE_RIGHT_HIP      = 24
POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12

# How2Sign filename suffix
NPY_SUFFIX = '-rgb_front_holistic.npy'

# Skeleton connections for the 3D frontend (Phase 8)
# Each tuple is (keypoint_index_A, keypoint_index_B)
SKELETON_CONNECTIONS = {
    "body": [
        (11, 12), (11, 23), (12, 24), (23, 24),   # torso
        (11, 13), (13, 15),                         # left arm
        (12, 14), (14, 16),                         # right arm
        (23, 25), (25, 27),                         # left leg
        (24, 26), (26, 28),                         # right leg
    ],
    "left_hand": [
        (33, 34), (34, 35), (35, 36),
        (33, 37), (37, 38), (38, 39), (39, 40),
        (33, 41), (41, 42), (42, 43), (43, 44),
        (33, 45), (45, 46), (46, 47), (47, 48),
        (33, 49), (49, 50), (50, 51), (51, 52),
    ],
    "right_hand": [
        (54, 55), (55, 56), (56, 57),
        (54, 58), (58, 59), (59, 60), (60, 61),
        (54, 62), (62, 63), (63, 64), (64, 65),
        (54, 66), (66, 67), (67, 68), (68, 69),
        (54, 70), (70, 71), (71, 72), (72, 73),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FILENAME LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
# How2Sign filenames: {SENTENCE_ID}-{N}-rgb_front_holistic.npy
# N is an unpredictable integer — we can't derive it from the CSV.
# Fix: scan the directory once and build { SENTENCE_ID → full_path }.

def build_npy_lookup(npy_dir: str) -> Dict[str, str]:
    """Scan npy_dir → { sentence_id: full_file_path }"""
    lookup: Dict[str, str] = {}
    for fname in os.listdir(npy_dir):
        if not fname.endswith(NPY_SUFFIX):
            continue
        base  = fname[: -len(NPY_SUFFIX)]       # strip suffix
        parts = base.rsplit('-', 1)              # split at last dash
        sid   = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else base
        lookup[sid] = os.path.join(npy_dir, fname)
    log.info(f"NPY lookup built: {len(lookup)} files in '{npy_dir}'")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def smooth_motion(motion: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Gaussian smoothing along the time axis (axis=0).
    Input/output shape: (T, 543, 3)

    WHY: MediaPipe detects keypoints per-frame independently — no memory
    of the previous frame — causing high-frequency jitter. σ=1.5 removes
    ~70% of jitter (confirmed by EDA) without flattening real motion peaks.

    WHY GAUSSIAN over moving average: moving average introduces temporal
    lag — it shifts peaks forward in time. For ASL the exact frame where
    a handshape peaks is linguistically significant. Gaussian smoothing is
    symmetric so it does not shift peaks.

    Applied BEFORE normalize_coordinates (see below for why).
    """
    from scipy.ndimage import gaussian_filter1d
    # axis=0 smooths along time, independently per keypoint per coordinate
    return gaussian_filter1d(motion, sigma=sigma, axis=0).astype(np.float32)


def normalize_coordinates(
    motion: np.ndarray,     # (T, 543, 3)
    eps:    float = 1e-6,
) -> np.ndarray:
    """
    Express all keypoints relative to the body:
      Translate → subtract hip midpoint    (body centre becomes origin)
      Scale     → divide by shoulder width (body size normalised to ~1.0)

    WHY: Raw MediaPipe coordinates are in image/camera space.
    Two signers signing the same phrase will have different raw values
    if they stand at different distances from the camera. After this
    normalization, coordinates describe HOW the body moves, not WHERE
    it was in the frame.

    Input/output shape: (T, 543, 3)
    Applied AFTER smooth_motion because shoulder-width (our scale
    reference) should be computed on smooth data, not jittery data.
    """
    motion = motion.copy()      # never mutate the caller's array
    T = motion.shape[0]         # number of frames

    # Hip midpoint: average of left and right hip positions
    # Shape: (T, 3)
    hip_mid = (
        motion[:, POSE_LEFT_HIP, :]   +
        motion[:, POSE_RIGHT_HIP, :]
    ) / 2.0

    # Shoulder width: Euclidean distance between shoulders per frame
    # Shape: (T,) → (T, 1, 1) for broadcasting over (543, 3)
    shoulder_w = np.linalg.norm(
        motion[:, POSE_LEFT_SHOULDER, :] - motion[:, POSE_RIGHT_SHOULDER, :],
        axis=1,                 # distance along xyz
    )                           # (T,)
    shoulder_w = np.maximum(shoulder_w, eps).reshape(T, 1, 1)  # (T, 1, 1)

    # Translate: subtract hip midpoint
    # hip_mid is (T, 3) → reshape to (T, 1, 3) to broadcast over 543 keypoints
    motion = motion - hip_mid[:, np.newaxis, :]     # (T, 543, 3)

    # Scale: divide by shoulder width
    motion = motion / shoulder_w                    # (T, 543, 3)

    return motion.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATASET
# ─────────────────────────────────────────────────────────────────────────────

class How2SignDataset(Dataset):
    """
    PyTorch Dataset for How2Sign Holistic.

    One item = (text: str, motion: FloatTensor[T, 543, 3])
    T varies per clip — padding handled by collate_fn.
    """

    def __init__(
        self,
        csv_path:     str,
        npy_dir:      str,
        normalize:    bool          = True,
        smooth:       bool          = True,
        smooth_sigma: float         = 1.5,
        max_frames:   Optional[int] = 450,
        min_frames:   int           = 10,   # filter corrupt short clips
    ) -> None:
        super().__init__()
        self.npy_dir      = npy_dir
        self.normalize    = normalize
        self.smooth       = smooth
        self.smooth_sigma = smooth_sigma
        self.max_frames   = max_frames
        self.min_frames   = min_frames

        # Build filename lookup once at startup
        self.npy_lookup = build_npy_lookup(npy_dir)

        self.df = self._load_csv(csv_path)
        self.df = self._filter(self.df)

        log.info(
            f"Dataset ready — {len(self.df)} clips | "
            f"normalize={normalize} | smooth={smooth} | "
            f"max_frames={max_frames} | min_frames={min_frames}"
        )

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path, sep='\t')
            if df.shape[1] < 2:
                raise ValueError
        except Exception:
            log.warning("TSV parse failed — retrying with comma.")
            df = pd.read_csv(csv_path)

        df.columns = [c.strip().upper() for c in df.columns]
        log.info(f"CSV: {len(df)} rows | columns: {list(df.columns)}")

        self.id_col = next(
            (c for c in ['SENTENCE_ID', 'VIDEO_ID', 'CLIP_ID', 'ID']
             if c in df.columns), None
        )
        self.text_col = next(
            (c for c in ['SENTENCE', 'ENGLISH_SENTENCE', 'TEXT', 'TRANSLATION']
             if c in df.columns), None
        )
        if not self.id_col:
            raise KeyError(f"No ID column. Got: {list(df.columns)}")
        if not self.text_col:
            raise KeyError(f"No text column. Got: {list(df.columns)}")

        before = len(df)
        df = df.dropna(subset=[self.id_col, self.text_col]).reset_index(drop=True)
        if len(df) < before:
            log.warning(f"Dropped {before-len(df)} rows with null ID or text.")
        return df

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only clips that exist in the lookup."""
        df   = df.copy()
        mask = df[self.id_col].astype(str).str.strip().isin(self.npy_lookup)
        log.warning(f"Filtered {(~mask).sum()} clips — no .npy found.")
        return df[mask].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.FloatTensor]:
        """
        Returns:
            text   : raw English sentence string
            motion : FloatTensor (T, 543, 3)
        """
        row  = self.df.iloc[idx]
        text = str(row[self.text_col]).strip()
        path = self.npy_lookup[str(row[self.id_col]).strip()]

        try:
            # allow_pickle=True because How2Sign files require it
            motion = np.load(path, allow_pickle=True).astype(np.float32)
        except Exception as e:
            log.error(f"Load failed {path}: {e} — returning zero clip.")
            motion = np.zeros((1, NUM_KEYPOINTS, COORDS), dtype=np.float32)

        # Validate shape: must be (T, 543, 3)
        if motion.ndim != 3 or motion.shape[1:] != (NUM_KEYPOINTS, COORDS):
            log.warning(f"Bad shape {motion.shape} — returning zero clip.")
            motion = np.zeros((1, NUM_KEYPOINTS, COORDS), dtype=np.float32)

        # Filter corrupt clips with too few frames
        if motion.shape[0] < self.min_frames:
            motion = np.zeros((1, NUM_KEYPOINTS, COORDS), dtype=np.float32)

        # Truncate very long clips to save GPU memory
        if self.max_frames and motion.shape[0] > self.max_frames:
            motion = motion[: self.max_frames]

        # Smooth THEN normalize (smooth first so scale reference is clean)
        if self.smooth:
            motion = smooth_motion(motion, sigma=self.smooth_sigma)
        if self.normalize:
            motion = normalize_coordinates(motion)

        return text, torch.from_numpy(motion)   # (T, 543, 3) FloatTensor

    def frame_stats(self) -> Dict[str, float]:
        """Scan .npy headers (fast) and return frame length statistics."""
        lengths = []
        for sid in self.df[self.id_col].astype(str).str.strip():
            path = self.npy_lookup.get(sid)
            if path:
                try:
                    arr = np.load(path, mmap_mode='r', allow_pickle=True)
                    lengths.append(arr.shape[0])
                except Exception:
                    pass
        L = np.array(lengths)
        return {
            'count':  len(L),
            'min':    int(L.min()),
            'max':    int(L.max()),
            'mean':   round(float(L.mean()), 1),
            'median': round(float(np.median(L)), 1),
            'std':    round(float(L.std()), 1),
            'p95':    round(float(np.percentile(L, 95)), 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Tuple[str, torch.Tensor]]) -> Dict:
    """
    Pad variable-length (T, 543, 3) tensors to max_T in the batch.

    Returns:
        texts          : List[str]
        motion_tensors : FloatTensor (B, max_T, 543, 3) — padded with 0.0
        motion_masks   : BoolTensor  (B, max_T)         — True = real frame
        lengths        : LongTensor  (B,)               — real frame counts
    """
    texts, motions = zip(*batch)
    B       = len(motions)
    lengths = torch.tensor([m.shape[0] for m in motions], dtype=torch.long)
    max_T   = int(lengths.max().item())

    # (B, max_T, 543, 3)
    padded = torch.zeros(B, max_T, NUM_KEYPOINTS, COORDS, dtype=torch.float32)
    mask   = torch.zeros(B, max_T, dtype=torch.bool)

    for i, (motion, L) in enumerate(zip(motions, lengths.tolist())):
        padded[i, :L] = motion
        mask[i,   :L] = True

    return {
        'texts':          list(texts),
        'motion_tensors': padded,     # (B, max_T, 543, 3)
        'motion_masks':   mask,       # (B, max_T)
        'lengths':        lengths,    # (B,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloader(
    csv_path:    str,
    npy_dir:     str,
    batch_size:  int           = 8,
    shuffle:     bool          = True,
    num_workers: int           = 0,
    normalize:   bool          = True,
    smooth:      bool          = True,
    max_frames:  Optional[int] = 450,
    min_frames:  int           = 10,
) -> Tuple[DataLoader, How2SignDataset]:
    dataset = How2SignDataset(
        csv_path=csv_path, npy_dir=npy_dir,
        normalize=normalize, smooth=smooth,
        max_frames=max_frames, min_frames=min_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    return loader, dataset


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile, random

    print("\n" + "=" * 65)
    print("  SignX Phase 1 — Smoke Test  (shape: T × 543 × 3)")
    print("=" * 65 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        npy_dir  = os.path.join(tmpdir, 'train', 'frontal')
        meta_dir = os.path.join(tmpdir, 'metadata')
        os.makedirs(npy_dir);  os.makedirs(meta_dir)

        random.seed(42)
        rows = []

        # Dummy files with the REAL shape (T, 543, 3) and naming convention
        for i in range(20):
            sid      = f'train_{i:04d}'
            n        = random.randint(1, 9)
            fname    = f'{sid}-{n}{NPY_SUFFIX}'
            n_frames = random.randint(15, 200)
            # Random coords in [0,1] — mimics raw MediaPipe image-space values
            motion   = np.random.rand(n_frames, NUM_KEYPOINTS, COORDS).astype(np.float32)
            np.save(os.path.join(npy_dir, fname), motion)
            rows.append({'SENTENCE_ID': sid, 'SENTENCE': f'ASL sentence {i}.'})

        # 3 rows with no matching file (filter test)
        for i in range(3):
            rows.append({'SENTENCE_ID': f'missing_{i}', 'SENTENCE': 'No file.'})

        csv_path = os.path.join(meta_dir, 'how2sign_realigned_train.csv')
        pd.DataFrame(rows).to_csv(csv_path, sep='\t', index=False)

        # ── Test 1: instantiation ──────────────────────────────────────
        print("── Test 1: Instantiation ───────────────────────────────────")
        loader, dataset = build_dataloader(
            csv_path, npy_dir, batch_size=4, num_workers=0
        )
        assert len(dataset) == 20
        print(f"Dataset size : {len(dataset)}  (3 missing filtered) ✓")

        # ── Test 2: single item ────────────────────────────────────────
        print("\n── Test 2: Single item ─────────────────────────────────────")
        text, motion = dataset[0]
        print(f"Text   : '{text}'")
        print(f"Motion : shape={tuple(motion.shape)}  dtype={motion.dtype}")
        assert motion.ndim == 3
        assert motion.shape[1] == NUM_KEYPOINTS
        assert motion.shape[2] == COORDS
        assert motion.dtype == torch.float32
        print("Shape (T, 543, 3) and dtype ✓")

        # ── Test 3: batch ──────────────────────────────────────────────
        print("\n── Test 3: Batch ───────────────────────────────────────────")
        batch = next(iter(loader))
        print(f"motion_tensors : {tuple(batch['motion_tensors'].shape)}")
        print(f"motion_masks   : {tuple(batch['motion_masks'].shape)}")
        print(f"lengths        : {batch['lengths'].tolist()}")
        assert batch['motion_tensors'].shape[2] == NUM_KEYPOINTS
        assert batch['motion_tensors'].shape[3] == COORDS
        for i, L in enumerate(batch['lengths'].tolist()):
            assert batch['motion_masks'][i, :L].all()
            assert not batch['motion_masks'][i, L:].any()
        print("Shape (B, max_T, 543, 3) and mask integrity ✓")

        # ── Test 4: normalization sanity check ─────────────────────────
        print("\n── Test 4: Normalization sanity ────────────────────────────")
        _, motion_norm = dataset[0]
        m = motion_norm.numpy()
        hip_mid_x = ((m[:, POSE_LEFT_HIP, 0] + m[:, POSE_RIGHT_HIP, 0]) / 2).mean()
        sw = np.linalg.norm(
            m[:, POSE_LEFT_SHOULDER, :] - m[:, POSE_RIGHT_SHOULDER, :], axis=1
        ).mean()
        print(f"Hip midpoint X (should be ≈ 0.0) : {hip_mid_x:.4f}")
        print(f"Shoulder width  (should be ≈ 1.0) : {sw:.4f}")
        assert abs(hip_mid_x) < 0.05, "Normalization failed — hip not centred"
        assert 0.8 < sw < 1.2,        "Normalization failed — shoulder width wrong"
        print("Normalization ✓")

        # ── Test 5: frame stats ────────────────────────────────────────
        print("\n── Test 5: Frame statistics ────────────────────────────────")
        for k, v in dataset.frame_stats().items():
            print(f"  {k:8s}: {v}")

        print("\n" + "=" * 65)
        print("  All tests PASSED ✓")
        print("=" * 65)

    print("""
── Kaggle Usage ──────────────────────────────────────────────────────
DATA_ROOT  = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
TRAIN_CSV  = DATA_ROOT + '/metadata/how2sign_realigned_train.csv'
TRAIN_NPY  = DATA_ROOT + '/train/frontal/'
VAL_CSV    = DATA_ROOT + '/metadata/how2sign_realigned_val.csv'
VAL_NPY    = DATA_ROOT + '/val/frontal/'

train_loader, train_ds = build_dataloader(TRAIN_CSV, TRAIN_NPY, batch_size=8)
val_loader,   val_ds   = build_dataloader(VAL_CSV,   VAL_NPY,   batch_size=8, shuffle=False)

# Always check your real data stats before training:
print(train_ds.frame_stats())
""")