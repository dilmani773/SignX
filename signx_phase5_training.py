"""
SignX — Phase 5: Training Loop
================================
Combines all phases into a complete training pipeline.

Components:
    SignXLoss        — MSE + velocity + length loss
    train_one_epoch  — single training epoch
    validate         — validation with MPJPE metric
    save_checkpoint  — save model + optimizer state
    load_checkpoint  — resume training
    run_training     — full training orchestration

On Kaggle — paste everything (Phase 1 + 3 + 4 + this file)
into one notebook and run.

Run locally (dummy data smoke test):
    python signx_phase5_training.py
"""

import os
import math
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

class SignXLoss(nn.Module):
    """
    Three-component loss for text-to-motion generation.

    total_loss = motion_loss
               + lambda_vel    * velocity_loss
               + lambda_length * length_loss

    Component 1 — Motion Loss (MSE on coordinates):
        Penalizes wrong keypoint positions.
        Applied ONLY on real frames (mask=True), not padding.
        WHY: Padding frames are zeros — including them would
        dominate the loss and teach the model to predict zeros.

    Component 2 — Velocity Loss (MSE on frame differences):
        Penalizes jerky, non-smooth motion.
        velocity[t] = frame[t+1] - frame[t]
        Model learns HOW SMOOTHLY hands move, not just WHERE they go.
        WHY: A model minimizing MSE alone can produce individually
        correct frames that look like a strobe light when animated.

    Component 3 — Length Loss (L1 on predicted vs actual frame count):
        Penalizes wrong animation duration.
        WHY L1 not MSE: L1 is more robust to outliers.
        A clip predicted as 50 frames when it should be 200 is a big
        error — L1 keeps gradients linear, MSE would explode them.
    """

    def __init__(
        self,
        lambda_vel:    float = 0.1,   # weight for velocity loss
        lambda_length: float = 0.01,  # weight for length loss
    ) -> None:
        super().__init__()
        self.lambda_vel    = lambda_vel
        self.lambda_length = lambda_length

        # reduction='none' lets us apply the mask manually
        self.mse = nn.MSELoss(reduction='none')
        self.l1  = nn.L1Loss()

    def forward(
        self,
        pred_motion:   torch.Tensor,   # (B, T, 543, 3) predicted
        target_motion: torch.Tensor,   # (B, T, 543, 3) ground truth
        motion_mask:   torch.Tensor,   # (B, T) True=real, False=pad
        pred_length:   torch.Tensor,   # (B,)   predicted frame count
        true_length:   torch.Tensor,   # (B,)   actual frame count
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss : scalar tensor (backpropagate this)
            components : dict of individual loss values (for logging)
        """

        # ── Component 1: Motion Loss ───────────────────────────────────────
        # MSE per element: (B, T, 543, 3)
        motion_error = self.mse(pred_motion, target_motion)

        # Expand mask to match motion dimensions for broadcasting:
        # (B, T) → (B, T, 1, 1) → broadcasts over (543, 3)
        mask_expanded = motion_mask.unsqueeze(-1).unsqueeze(-1).float()

        # Zero out padded frames, sum over real frames only
        masked_error  = motion_error * mask_expanded

        # Divide by number of real elements (not total elements)
        # to get true mean over real frames only
        num_real = mask_expanded.sum() * 543 * 3   # total real values
        motion_loss = masked_error.sum() / (num_real + 1e-8)

        # ── Component 2: Velocity Loss ─────────────────────────────────────
        # Velocity = difference between consecutive frames
        # Shape: (B, T-1, 543, 3)
        pred_vel   = pred_motion[:, 1:, :, :] - pred_motion[:, :-1, :, :]
        target_vel = target_motion[:, 1:, :, :] - target_motion[:, :-1, :, :]

        # Mask for velocity: a transition t→t+1 is real only if BOTH
        # frame t AND frame t+1 are real frames
        # mask[:, :-1] = is frame t real?
        # mask[:, 1:]  = is frame t+1 real?
        vel_mask = (motion_mask[:, :-1] & motion_mask[:, 1:])  # (B, T-1)
        vel_mask_exp = vel_mask.unsqueeze(-1).unsqueeze(-1).float()

        vel_error    = self.mse(pred_vel, target_vel) * vel_mask_exp
        num_vel_real = vel_mask_exp.sum() * 543 * 3
        velocity_loss = vel_error.sum() / (num_vel_real + 1e-8)

        # ── Component 3: Length Loss ───────────────────────────────────────
        # Compare predicted frame count to actual frame count
        # true_length is LongTensor — cast to float for L1
        length_loss = self.l1(pred_length, true_length.float())

        # ── Total ──────────────────────────────────────────────────────────
        total = (
            motion_loss
            + self.lambda_vel    * velocity_loss
            + self.lambda_length * length_loss
        )

        components = {
            'motion_loss':   motion_loss.item(),
            'velocity_loss': velocity_loss.item(),
            'length_loss':   length_loss.item(),
            'total_loss':    total.item(),
        }

        return total, components


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MPJPE METRIC
# ─────────────────────────────────────────────────────────────────────────────

def compute_mpjpe(
    pred_motion:   torch.Tensor,   # (B, T, 543, 3)
    target_motion: torch.Tensor,   # (B, T, 543, 3)
    motion_mask:   torch.Tensor,   # (B, T) True=real
) -> float:
    """
    Mean Per Joint Position Error — standard pose estimation metric.

    For each real frame, for each keypoint, compute the Euclidean
    distance between predicted and ground truth position.
    Return the mean across all keypoints and all real frames.

    WHY MPJPE over MSE as metric:
        MSE squares errors — one bad prediction dominates.
        MPJPE is directly interpretable: "on average, each joint
        is X units away from where it should be."
        This is what pose estimation papers report.

    Returns: float (lower is better)
    """
    with torch.no_grad():
        # Euclidean distance per keypoint per frame: (B, T, 543)
        dist = torch.norm(
            pred_motion - target_motion,
            dim=-1,             # norm over (x,y,z) dimension
        )

        # Apply mask — only real frames
        mask_float = motion_mask.float()        # (B, T)
        masked_dist = dist * mask_float.unsqueeze(-1)  # (B, T, 543)

        # Mean over real frames only
        total_dist  = masked_dist.sum()
        total_real  = mask_float.sum() * 543    # real frames × keypoints
        mpjpe = (total_dist / (total_real + 1e-8)).item()

    return mpjpe


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    loader:    DataLoader,
    encoder:   nn.Module,
    decoder:   nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn:   SignXLoss,
    device:    torch.device,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    """
    Train for one full pass over the dataset.

    Args:
        loader    : DataLoader yielding batches from Phase 1
        encoder   : EnglishTextEncoder (Phase 3)
        decoder   : ContinuousMotionDecoder (Phase 4)
        optimizer : AdamW
        loss_fn   : SignXLoss
        device    : cuda or cpu
        clip_grad : max gradient norm (prevents exploding gradients)

    Returns:
        dict of average loss components for this epoch
    """
    encoder.train()
    decoder.train()

    # Accumulators for averaging over all batches
    total_losses = {
        'motion_loss': 0.0,
        'velocity_loss': 0.0,
        'length_loss': 0.0,
        'total_loss': 0.0,
    }
    num_batches = 0

    for batch_idx, batch in enumerate(loader):

        # ── Move batch to device ───────────────────────────────────────────
        motion_tensors = batch['motion_tensors'].to(device)  # (B,T,543,3)
        motion_masks   = batch['motion_masks'].to(device)    # (B,T)
        lengths        = batch['lengths'].to(device)         # (B,)
        texts          = batch['texts']                      # List[str]

        # ── Forward pass ───────────────────────────────────────────────────
        # Step 1: Text → embedding
        text_memory = encoder(texts)                         # (B, 512)

        # Step 2: Embedding + motion → predictions
        # decoder.forward() uses teacher forcing internally
        pred_motion, pred_length = decoder(
            text_memory=text_memory,
            target_motion=motion_tensors,
            motion_mask=motion_masks,
        )                          # pred_motion: (B,T,543,3), pred_length: (B,)

        # ── Compute loss ───────────────────────────────────────────────────
        total_loss, components = loss_fn(
            pred_motion=pred_motion,
            target_motion=motion_tensors,
            motion_mask=motion_masks,
            pred_length=pred_length,
            true_length=lengths,
        )

        # ── Backward pass ──────────────────────────────────────────────────
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping: if gradients explode (norm > clip_grad),
        # scale them down. Common in Transformer training.
        # WHY: Transformers with cross-attention can produce large gradient
        # spikes early in training, especially with continuous outputs.
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=clip_grad,
        )

        optimizer.step()

        # ── Accumulate losses ──────────────────────────────────────────────
        for k in total_losses:
            total_losses[k] += components[k]
        num_batches += 1

        # ── Log every 50 batches ───────────────────────────────────────────
        if (batch_idx + 1) % 50 == 0:
            log.info(
                f"  Batch {batch_idx+1:4d} | "
                f"loss={components['total_loss']:.4f} | "
                f"motion={components['motion_loss']:.4f} | "
                f"vel={components['velocity_loss']:.4f}"
            )

    # Average over all batches
    return {k: v / num_batches for k, v in total_losses.items()}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — VALIDATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    loader:  DataLoader,
    encoder: nn.Module,
    decoder: nn.Module,
    loss_fn: SignXLoss,
    device:  torch.device,
) -> Dict[str, float]:
    """
    Evaluate on validation set.
    No gradients, no weight updates.
    Reports both loss and MPJPE.
    """
    encoder.eval()
    decoder.eval()

    total_losses = {
        'motion_loss': 0.0, 'velocity_loss': 0.0,
        'length_loss': 0.0, 'total_loss':    0.0,
    }
    total_mpjpe = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            motion_tensors = batch['motion_tensors'].to(device)
            motion_masks   = batch['motion_masks'].to(device)
            lengths        = batch['lengths'].to(device)
            texts          = batch['texts']

            text_memory  = encoder(texts)
            pred_motion, pred_length = decoder(
                text_memory=text_memory,
                target_motion=motion_tensors,
                motion_mask=motion_masks,
            )

            _, components = loss_fn(
                pred_motion=pred_motion,
                target_motion=motion_tensors,
                motion_mask=motion_masks,
                pred_length=pred_length,
                true_length=lengths,
            )

            mpjpe = compute_mpjpe(pred_motion, motion_tensors, motion_masks)

            for k in total_losses:
                total_losses[k] += components[k]
            total_mpjpe += mpjpe
            num_batches += 1

    results = {k: v / num_batches for k, v in total_losses.items()}
    results['mpjpe'] = total_mpjpe / num_batches
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    encoder:    nn.Module,
    decoder:    nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    val_mpjpe:  float,
    path:       str = 'signx_best.pth',
) -> None:
    """
    Save model weights + optimizer state + training metadata.

    WHY save optimizer state?
    Optimizer (AdamW) maintains momentum and adaptive learning rate
    per parameter. If you resume training without it, the optimizer
    starts cold — first few batches are unstable.
    Saving it lets training resume exactly where it left off.

    WHY save val_mpjpe?
    So when resuming, you know the best score so far and can
    decide whether the new epoch improved it.
    """
    checkpoint = {
        'epoch':             epoch,
        'val_mpjpe':         val_mpjpe,
        'encoder_state':     encoder.state_dict(),
        'decoder_state':     decoder.state_dict(),
        'optimizer_state':   optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    log.info(f"Checkpoint saved → {path}  (epoch={epoch}, mpjpe={val_mpjpe:.4f})")


def load_checkpoint(
    path:      str,
    encoder:   nn.Module,
    decoder:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device:    torch.device = torch.device('cpu'),
) -> Dict:
    """
    Load a checkpoint and restore model + optimizer states.

    Returns the checkpoint dict so caller can read epoch, val_mpjpe etc.
    """
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    log.info(
        f"Checkpoint loaded ← {path}  "
        f"(epoch={checkpoint['epoch']}, mpjpe={checkpoint['val_mpjpe']:.4f})"
    )
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FULL TRAINING ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    encoder:         nn.Module,
    decoder:         nn.Module,
    num_epochs:      int   = 30,
    learning_rate:   float = 1e-4,
    weight_decay:    float = 1e-2,
    clip_grad:       float = 1.0,
    checkpoint_path: str   = 'signx_best.pth',
    resume_from:     Optional[str] = None,
    device:          torch.device  = torch.device('cpu'),
) -> None:
    """
    Full training loop with:
        - AdamW optimizer with weight decay
        - Cosine annealing learning rate schedule
        - Best model checkpointing by val MPJPE
        - Early stopping after 5 epochs without improvement
        - Resumable from checkpoint

    WHY AdamW over Adam?
        Adam has a weight decay bug — it applies decay to all params
        uniformly including the adaptive learning rate scaling.
        AdamW fixes this by decoupling weight decay from the gradient
        update. Better generalization, standard in modern Transformers.

    WHY Cosine Annealing?
        Constant LR: model oscillates around minimum, never converges.
        Step decay: abrupt drops can destabilize training.
        Cosine annealing: smoothly decreases LR following a cosine curve.
        Near end of training LR approaches zero → model settles into
        the loss minimum cleanly.
    """
    # ── Optimizer ──────────────────────────────────────────────────────────
    # Only pass trainable parameters — frozen DistilBERT params excluded
    trainable_params = (
        list(filter(lambda p: p.requires_grad, encoder.parameters())) +
        list(decoder.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ── LR Scheduler ───────────────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    # ── Loss function ───────────────────────────────────────────────────────
    loss_fn = SignXLoss(lambda_vel=0.1, lambda_length=0.01)

    # ── Resume from checkpoint if provided ─────────────────────────────────
    start_epoch   = 0
    best_mpjpe    = float('inf')
    no_improve    = 0

    if resume_from and os.path.exists(resume_from):
        ckpt        = load_checkpoint(resume_from, encoder, decoder, optimizer, device)
        start_epoch = ckpt['epoch'] + 1
        best_mpjpe  = ckpt['val_mpjpe']
        log.info(f"Resuming from epoch {start_epoch}")

    # ── Training loop ───────────────────────────────────────────────────────
    log.info(f"Starting training: {num_epochs} epochs on {device}")
    log.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            loader=train_loader, encoder=encoder, decoder=decoder,
            optimizer=optimizer, loss_fn=loss_fn,
            device=device, clip_grad=clip_grad,
        )

        # Validate
        val_metrics = validate(
            loader=val_loader, encoder=encoder, decoder=decoder,
            loss_fn=loss_fn, device=device,
        )

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # ── Log epoch summary ──────────────────────────────────────────────
        log.info(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"time={epoch_time:.0f}s | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_metrics['total_loss']:.4f} | "
            f"val_loss={val_metrics['total_loss']:.4f} | "
            f"val_mpjpe={val_metrics['mpjpe']:.4f}"
        )

        # ── Save best checkpoint ───────────────────────────────────────────
        if val_metrics['mpjpe'] < best_mpjpe:
            best_mpjpe = val_metrics['mpjpe']
            no_improve = 0
            save_checkpoint(
                encoder, decoder, optimizer,
                epoch=epoch,
                val_mpjpe=best_mpjpe,
                path=checkpoint_path,
            )
            log.info(f"  ↑ New best MPJPE: {best_mpjpe:.4f}")
        else:
            no_improve += 1
            log.info(f"  No improvement ({no_improve}/5)")

        # ── Early stopping ─────────────────────────────────────────────────
        # WHY early stopping?
        # Kaggle gives 9 hours. If validation MPJPE stops improving for
        # 5 consecutive epochs, continuing wastes GPU time and risks
        # overfitting. Stop early and keep the best checkpoint.
        if no_improve >= 5:
            log.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    log.info(f"Training complete. Best val MPJPE: {best_mpjpe:.4f}")
    log.info(f"Best model saved at: {checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import tempfile
    import random
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader

    # Add current directory to path so we can import other phases
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from signx_phase1_dataset import (
        How2SignDataset, collate_fn,
        NPY_SUFFIX, NUM_KEYPOINTS, COORDS
    )
    from signx_phase3_encoder import EnglishTextEncoder
    from signx_phase4_decoder import ContinuousMotionDecoder

    print("\n" + "=" * 65)
    print("  SignX Phase 5 — Training Loop Smoke Test")
    print("=" * 65 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Build synthetic dataset ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        npy_dir  = os.path.join(tmpdir, 'frontal')
        meta_dir = os.path.join(tmpdir, 'meta')
        os.makedirs(npy_dir); os.makedirs(meta_dir)

        random.seed(42)
        rows = []
        for i in range(32):     # 32 clips → 4 batches of 8
            sid      = f'clip_{i:04d}'
            n        = random.randint(1, 9)
            fname    = f'{sid}-{n}{NPY_SUFFIX}'
            n_frames = random.randint(20, 100)
            motion   = np.random.rand(
                n_frames, NUM_KEYPOINTS, COORDS
            ).astype(np.float32)
            np.save(os.path.join(npy_dir, fname), motion)
            rows.append({'SENTENCE_ID': sid, 'SENTENCE': f'Test sentence {i}.'})

        csv_path = os.path.join(meta_dir, 'train.csv')
        pd.DataFrame(rows).to_csv(csv_path, sep='\t', index=False)

        # Split: 24 train, 8 val
        train_dataset = How2SignDataset(
            csv_path=csv_path, npy_dir=npy_dir,
            normalize=True, smooth=False, max_frames=450,
        )
        # Simple split for smoke test
        from torch.utils.data import Subset
        train_ds = Subset(train_dataset, list(range(24)))
        val_ds   = Subset(train_dataset, list(range(24, 32)))

        train_loader = DataLoader(
            train_ds, batch_size=4, shuffle=True,
            collate_fn=collate_fn, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=4, shuffle=False,
            collate_fn=collate_fn, drop_last=True,
        )

        # ── Instantiate models ─────────────────────────────────────────────
        encoder = EnglishTextEncoder(
            decoder_dim=512, freeze_bert=True
        ).to(device)

        decoder = ContinuousMotionDecoder(
            decoder_dim=512, num_heads=8,
            num_layers=2,    # use 2 layers for fast smoke test
            ffn_dim=512,
        ).to(device)

        # ── Test 1: Loss function ──────────────────────────────────────────
        print("── Test 1: Loss function ───────────────────────────────────")
        loss_fn = SignXLoss(lambda_vel=0.1, lambda_length=0.01)
        batch   = next(iter(train_loader))

        mt = batch['motion_tensors'].to(device)
        mm = batch['motion_masks'].to(device)
        ln = batch['lengths'].to(device)

        with torch.no_grad():
            tm  = encoder(batch['texts'])
        pm, pl = decoder(tm, mt, mm)

        total_loss, comps = loss_fn(pm, mt, mm, pl, ln)
        print(f"motion_loss   : {comps['motion_loss']:.4f}")
        print(f"velocity_loss : {comps['velocity_loss']:.4f}")
        print(f"length_loss   : {comps['length_loss']:.4f}")
        print(f"total_loss    : {comps['total_loss']:.4f}")
        assert total_loss.item() > 0
        print("Loss values ✓")

        # ── Test 2: MPJPE ──────────────────────────────────────────────────
        print("\n── Test 2: MPJPE metric ────────────────────────────────────")
        mpjpe = compute_mpjpe(pm, mt, mm)
        print(f"MPJPE (untrained): {mpjpe:.4f}")
        assert mpjpe >= 0
        print("MPJPE ✓")

        # ── Test 3: One training epoch ─────────────────────────────────────
        print("\n── Test 3: One training epoch ──────────────────────────────")
        trainable = (
            list(filter(lambda p: p.requires_grad, encoder.parameters())) +
            list(decoder.parameters())
        )
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)

        metrics = train_one_epoch(
            train_loader, encoder, decoder,
            optimizer, loss_fn, device,
        )
        print(f"Epoch train loss: {metrics['total_loss']:.4f}")
        print("Training epoch ✓")

        # ── Test 4: Validation ─────────────────────────────────────────────
        print("\n── Test 4: Validation ──────────────────────────────────────")
        val_metrics = validate(val_loader, encoder, decoder, loss_fn, device)
        print(f"Val loss  : {val_metrics['total_loss']:.4f}")
        print(f"Val MPJPE : {val_metrics['mpjpe']:.4f}")
        print("Validation ✓")

        # ── Test 5: Checkpoint save/load ───────────────────────────────────
        print("\n── Test 5: Checkpoint save / load ──────────────────────────")
        ckpt_path = os.path.join(tmpdir, 'test_checkpoint.pth')
        save_checkpoint(encoder, decoder, optimizer,
                        epoch=0, val_mpjpe=val_metrics['mpjpe'],
                        path=ckpt_path)

        # Load into fresh models and verify weights match
        enc2 = EnglishTextEncoder(decoder_dim=512, freeze_bert=True).to(device)
        dec2 = ContinuousMotionDecoder(
            decoder_dim=512, num_heads=8, num_layers=2, ffn_dim=512
        ).to(device)
        opt2 = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, enc2.parameters())) +
            list(dec2.parameters()), lr=1e-4
        )
        load_checkpoint(ckpt_path, enc2, dec2, opt2, device)

        # Verify output is identical after loading.
        # MUST call .eval() on both models before comparing —
        # Dropout in the projection layer is stochastic during train mode,
        # producing different values each forward pass.
        # .eval() disables Dropout → deterministic → outputs must match.
        encoder.eval()
        enc2.eval()

        with torch.no_grad():
            out1 = encoder(["Test sentence."])
            out2 = enc2(["Test sentence."])
        assert torch.allclose(out1, out2, atol=1e-4), \
            "Loaded model produces different output!"
        print("Checkpoint save/load ✓")

        print("\n" + "=" * 65)
        print("  All tests PASSED ✓")
        print("=" * 65)

    # ── Kaggle training script ─────────────────────────────────────────────
    print("""
── Kaggle Training Script ────────────────────────────────────────────

DATA_ROOT = '/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features'
TRAIN_CSV = DATA_ROOT + '/metadata/how2sign_realigned_train.csv'
TRAIN_NPY = DATA_ROOT + '/train/frontal/'
VAL_CSV   = DATA_ROOT + '/metadata/how2sign_realigned_val.csv'
VAL_NPY   = DATA_ROOT + '/val/frontal/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, _ = build_dataloader(TRAIN_CSV, TRAIN_NPY,
                                   batch_size=8, smooth=False)
val_loader,   _ = build_dataloader(VAL_CSV,   VAL_NPY,
                                   batch_size=8, shuffle=False, smooth=False)

encoder = EnglishTextEncoder(decoder_dim=512, freeze_bert=True).to(device)
decoder = ContinuousMotionDecoder(decoder_dim=512).to(device)

run_training(
    train_loader    = train_loader,
    val_loader      = val_loader,
    encoder         = encoder,
    decoder         = decoder,
    num_epochs      = 30,
    learning_rate   = 1e-4,
    checkpoint_path = '/kaggle/working/signx_best.pth',
    device          = device,
)
# Download signx_best.pth from Kaggle output after training.
""")