"""
SignX — Phase 4: Continuous Motion Decoder
==========================================
Takes text embedding (B, 512) from Phase 3 and generates
continuous 3D motion (B, T, 543, 3).

Components:
    MotionLengthEstimator  — MLP: predicts frame count from text
    PositionalEncoding     — sinusoidal time stamps for frames
    ContinuousMotionDecoder — Transformer decoder:
                              self-attention  (motion context)
                              cross-attention (text memory)

Run locally:
    python signx_phase4_decoder.py
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CONSTANTS  (must match Phase 1 and Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

NUM_KEYPOINTS = 543
COORDS        = 3
MOTION_DIM    = NUM_KEYPOINTS * COORDS   # 1629 — flattened per-frame features
DECODER_DIM   = 512                      # must match EnglishTextEncoder.decoder_dim


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MOTION LENGTH ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class MotionLengthEstimator(nn.Module):
    """
    Predicts how many frames a signing clip should have,
    given the text embedding.

    WHY a separate module?
      The decoder needs to know L (frame count) before generating.
      We learn this from text because longer sentences → more frames.
      It's a weak but useful prior — signing speed varies, so we treat
      this as a soft guide, not a hard rule.

    Input  : (B, decoder_dim)
    Output : raw float (B,) for loss  |  rounded int (B,) for inference
    """

    def __init__(
        self,
        decoder_dim: int = DECODER_DIM,
        hidden_dim:  int = 256,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
            # WHY Softplus over ReLU:
            # ReLU hard-zeros negatives → derivative discontinuity at 0.
            # Softplus = smooth, differentiable, always positive.
            # Frame count must be positive — Softplus guarantees this
            # while keeping stable gradients near zero.
        )

    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Returns raw (float) predicted length for use in loss calculation.
        Shape: (B,)
        """
        return self.mlp(text_embedding).squeeze(-1)   # (B,)

    def predict(self, text_embedding: torch.Tensor, min_frames: int = 10) -> torch.Tensor:
        """
        Returns integer frame counts for use during inference.
        Clamped to at least min_frames so we always generate something.
        Shape: (B,) LongTensor
        """
        with torch.no_grad():
            raw = self.forward(text_embedding)             # (B,) float
        return torch.clamp(raw.long(), min=min_frames)     # (B,) int


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal position information into frame sequences.

    WHY sinusoidal (not learned)?
      Learned positional embeddings only work up to the max sequence
      length seen during training. Sinusoidal generalizes beyond it —
      if training used max_frames=450 but inference generates 600 frames,
      sinusoidal still produces valid encodings. Learned embeddings would
      fail (index out of range).

    WHY sin AND cos?
      Using both gives the model two independent signals per position.
      More importantly: PE(pos+k) can be expressed as a linear function
      of PE(pos) — the model can easily learn relative positions.

    Input/output shape: (B, T, decoder_dim) — adds encoding in-place.
    """

    def __init__(
        self,
        decoder_dim: int   = DECODER_DIM,
        max_len:     int   = 5000,      # large enough for any sequence
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the encoding table once — shape (max_len, decoder_dim)
        # We register it as a buffer (not a parameter) because:
        #   - It's fixed (not learned)
        #   - It should move to GPU with .to(device) automatically
        #   - It should be saved with the model checkpoint
        pe     = torch.zeros(max_len, decoder_dim)               # (L, D)
        pos    = torch.arange(0, max_len).unsqueeze(1).float()   # (L, 1)

        # Denominator: 10000^(2i/d) — creates different frequencies
        # per dimension so each dimension encodes a different "wavelength"
        div    = torch.exp(
            torch.arange(0, decoder_dim, 2).float()
            * (-math.log(10000.0) / decoder_dim)
        )

        pe[:, 0::2] = torch.sin(pos * div)   # even dimensions → sin
        pe[:, 1::2] = torch.cos(pos * div)   # odd  dimensions → cos

        # Add batch dimension: (1, max_len, decoder_dim)
        # The 1 broadcasts over any batch size automatically
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, decoder_dim)
        Returns:
            x + positional encoding, same shape
        """
        # x.size(1) = T — slice only as many positions as we need
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CONTINUOUS MOTION DECODER
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousMotionDecoder(nn.Module):
    """
    Transformer decoder that generates a sequence of 3D keypoint coordinates
    conditioned on a text embedding.

    Architecture:
        Input projection  : (B, T, MOTION_DIM) → (B, T, decoder_dim)
        Positional Encoding: inject time information
        Transformer Decoder: self-attention + cross-attention to text
        Output projection : (B, T, decoder_dim) → (B, T, MOTION_DIM)
        Reshape           : (B, T, MOTION_DIM) → (B, T, 543, 3)

    Two modes:
        forward() — training, uses teacher forcing
        generate() — inference, autoregressive frame-by-frame
    """

    def __init__(
        self,
        decoder_dim:  int   = DECODER_DIM,
        num_heads:    int   = 8,
        num_layers:   int   = 4,
        ffn_dim:      int   = 2048,
        dropout:      float = 0.1,
        max_len:      int   = 500,
    ) -> None:
        """
        Args:
            decoder_dim : Hidden dimension. Must match EnglishTextEncoder output.
            num_heads   : Attention heads. decoder_dim must be divisible by this.
                          8 heads × 64 dim each = 512. Standard setup.
            num_layers  : Transformer decoder layers. 4 is a good start —
                          deep enough to learn complex motion, light enough
                          for Kaggle GPU memory.
            ffn_dim     : Feed-forward network inner dimension.
                          Convention: 4 × decoder_dim = 4 × 512 = 2048.
            dropout     : Regularization. 0.1 is standard for Transformers.
            max_len     : Max sequence length for positional encoding.
        """
        super().__init__()

        self.decoder_dim = decoder_dim
        self.motion_dim  = MOTION_DIM    # 1629 (543 × 3)
        self.num_kp      = NUM_KEYPOINTS # 543
        self.coords      = COORDS        # 3

        # ── Input Projection ───────────────────────────────────────────────
        # Motion frames are (T, 543, 3) → flatten to (T, 1629) → project to
        # (T, decoder_dim=512) so the Transformer can process them.
        # WHY flatten then project (not process (543,3) directly)?
        # The Transformer processes one token per timestep.
        # Each token = one frame = all 543 keypoints together.
        # Flattening lets the model learn inter-keypoint relationships
        # within a single frame implicitly through the linear projection.
        self.input_proj = nn.Sequential(
            nn.Linear(MOTION_DIM, decoder_dim),
            nn.ReLU(),
        )

        # ── Positional Encoding ────────────────────────────────────────────
        self.pos_encoding = PositionalEncoding(
            decoder_dim=decoder_dim,
            max_len=max_len,
            dropout=dropout,
        )

        # ── Transformer Decoder ────────────────────────────────────────────
        # PyTorch's TransformerDecoderLayer has:
        #   1. Masked self-attention  → attends to previous frames only
        #   2. Cross-attention        → attends to text embedding (memory)
        #   3. Feed-forward network   → per-position transformation
        #
        # WHY batch_first=True?
        # Default PyTorch Transformer expects (T, B, D) — time first.
        # batch_first=True uses (B, T, D) — batch first.
        # (B, T, D) is consistent with everything else in our pipeline.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LayerNorm: more stable training
            # WHY norm_first=True (Pre-LN) over Post-LN?
            # Post-LN (original Transformer) has vanishing gradient problems
            # in deep networks. Pre-LN normalizes BEFORE attention —
            # gradients flow more cleanly. Standard in modern Transformers.
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(decoder_dim),
        )

        # ── Output Projection ──────────────────────────────────────────────
        # Map decoder hidden state → actual coordinate predictions
        # (B, T, decoder_dim) → (B, T, MOTION_DIM=1629)
        self.output_proj = nn.Linear(decoder_dim, MOTION_DIM)

        # ── Length Estimator ───────────────────────────────────────────────
        self.length_estimator = MotionLengthEstimator(decoder_dim=decoder_dim)

        # ── Parameter initialization ───────────────────────────────────────
        # Xavier uniform init for linear layers: keeps activation variance
        # stable at initialization → faster convergence.
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ── Causal Mask Helper ─────────────────────────────────────────────────

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """
        Upper-triangular mask that prevents frame t from attending to
        frames t+1, t+2, ... (future frames).

        Returns: (T, T) BoolTensor — True = IGNORE this position
        PyTorch's attention interprets True as "mask out" (set to -inf).

        Example for T=4:
            [[F, T, T, T],   frame 0 sees only itself
             [F, F, T, T],   frame 1 sees 0,1
             [F, F, F, T],   frame 2 sees 0,1,2
             [F, F, F, F]]   frame 3 sees all
        """
        mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    # ── Training Forward Pass (Teacher Forcing) ────────────────────────────

    def forward(
        self,
        text_memory:   torch.Tensor,            # (B, 512) from Phase 3
        target_motion: torch.Tensor,            # (B, T, 543, 3) ground truth
        motion_mask:   Optional[torch.Tensor] = None,  # (B, T) True=real
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass using teacher forcing.

        Teacher forcing: feed ground truth frames as input, predict
        next frames as output. This stabilizes early training.

        Args:
            text_memory   : (B, 512) text embedding from Phase 3
            target_motion : (B, T, 543, 3) ground truth motion
            motion_mask   : (B, T) True=real frame, False=padding

        Returns:
            pred_motion   : (B, T-1, 543, 3) predicted frames
            pred_length   : (B,) predicted frame count (float, for loss)
        """
        B, T, K, C = target_motion.shape

        # ── Step 1: Flatten (T, 543, 3) → (T, 1629) ──────────────────────
        motion_flat = target_motion.reshape(B, T, self.motion_dim)  # (B,T,1629)

        # ── Step 2: Teacher forcing shift ─────────────────────────────────
        # Input  = frames [0 .. T-2] (all except last)
        # Target = frames [1 .. T-1] (all except first)
        # Frame 0 = zero tensor (start-of-sequence token)
        start_token = torch.zeros(B, 1, self.motion_dim, device=target_motion.device)
        decoder_input = torch.cat([start_token, motion_flat[:, :-1, :]], dim=1)
        # decoder_input shape: (B, T, 1629)

        # ── Step 3: Project to decoder_dim ────────────────────────────────
        x = self.input_proj(decoder_input)       # (B, T, 512)

        # ── Step 4: Add positional encoding ───────────────────────────────
        x = self.pos_encoding(x)                 # (B, T, 512)

        # ── Step 5: Expand text memory for cross-attention ─────────────────
        # TransformerDecoder cross-attention expects memory shape (B, S, D)
        # Our text_memory is (B, 512) — a single vector per sentence.
        # Unsqueeze to (B, 1, 512) — one "memory token" per sentence.
        # The decoder can attend to this single rich token.
        memory = text_memory.unsqueeze(1)        # (B, 1, 512)

        # ── Step 6: Causal mask ────────────────────────────────────────────
        causal_mask = self._causal_mask(T, target_motion.device)

        # ── Step 7: Transformer decoder ───────────────────────────────────
        # tgt          = current motion frames (what we're decoding)
        # memory       = text embedding (what we condition on)
        # tgt_mask     = causal mask (no peeking at future frames)
        out = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
        )                                        # (B, T, 512)

        # ── Step 8: Project to motion coordinates ─────────────────────────
        pred_flat = self.output_proj(out)        # (B, T, 1629)

        # ── Step 9: Reshape to (B, T, 543, 3) ─────────────────────────────
        pred_motion = pred_flat.reshape(B, T, self.num_kp, self.coords)

        # ── Step 10: Predict sequence length ──────────────────────────────
        pred_length = self.length_estimator(text_memory)   # (B,) float

        return pred_motion, pred_length

    # ── Inference: Autoregressive Generation ──────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        text_memory: torch.Tensor,    # (B, 512) from Phase 3
        max_frames:  int = 450,
    ) -> torch.Tensor:
        """
        Autoregressively generate motion frame by frame.

        At each step:
            1. Feed all frames generated SO FAR into the decoder
            2. Take only the LAST output token (next frame prediction)
            3. Append to generated sequence
            4. Repeat until predicted length reached

        WHY autoregressive (not all-at-once)?
          Each frame depends on previous frames — a signer's hand
          position at frame t depends on where it was at t-1.
          Autoregressive generation respects this temporal dependency.

        Args:
            text_memory : (B, 512)
            max_frames  : safety ceiling (never generate more than this)

        Returns:
            generated : (B, L, 543, 3) where L = predicted frame count
        """
        B      = text_memory.shape[0]
        device = text_memory.device

        # Predict how many frames to generate
        lengths = self.length_estimator.predict(text_memory)   # (B,) int
        L       = min(int(lengths.max().item()), max_frames)

        memory = text_memory.unsqueeze(1)   # (B, 1, 512)

        # Start with a single zero frame (start token)
        generated = torch.zeros(B, 1, self.motion_dim, device=device)  # (B,1,1629)

        for step in range(L):
            # Project all frames generated so far
            x = self.input_proj(generated)           # (B, step+1, 512)
            x = self.pos_encoding(x)                 # (B, step+1, 512)

            T_cur = generated.shape[1]
            causal_mask = self._causal_mask(T_cur, device)

            out = self.transformer_decoder(
                tgt=x,
                memory=memory,
                tgt_mask=causal_mask,
            )                                        # (B, step+1, 512)

            # Take only the last position — next frame prediction
            next_frame = self.output_proj(out[:, -1:, :])  # (B, 1, 1629)

            # Append to sequence
            generated = torch.cat([generated, next_frame], dim=1)  # (B, step+2, 1629)

        # Remove the initial start token (index 0)
        generated = generated[:, 1:, :]              # (B, L, 1629)

        # Reshape to (B, L, 543, 3)
        return generated.reshape(B, L, self.num_kp, self.coords)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — UNIFIED MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
# WHY a wrapper?
# In the training loop we always use encoder + decoder together.
# A wrapper makes the API clean: one object, one forward call.
# Also makes saving/loading checkpoints simpler — one state_dict.

class SignXModel(nn.Module):
    """
    Full SignX model: text → motion.
    Wraps EnglishTextEncoder + ContinuousMotionDecoder.
    """

    def __init__(
        self,
        decoder_dim: int   = DECODER_DIM,
        num_heads:   int   = 8,
        num_layers:  int   = 4,
        ffn_dim:     int   = 2048,
        dropout:     float = 0.1,
        freeze_bert: bool  = True,
    ) -> None:
        super().__init__()

        # Import here to avoid circular imports when files are separate
        # On Kaggle you'll have all phases in one notebook so this works fine
        from signx_phase3_encoder import EnglishTextEncoder

        self.encoder = EnglishTextEncoder(
            decoder_dim=decoder_dim,
            freeze_bert=freeze_bert,
            dropout=dropout,
        )
        self.decoder = ContinuousMotionDecoder(
            decoder_dim=decoder_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(
        self,
        texts:         list,
        target_motion: torch.Tensor,
        motion_mask:   Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass."""
        text_memory  = self.encoder(texts)
        pred_motion, pred_length = self.decoder(
            text_memory, target_motion, motion_mask
        )
        return pred_motion, pred_length

    @torch.no_grad()
    def generate(self, texts: list, max_frames: int = 450) -> torch.Tensor:
        """Inference: text → generated motion."""
        text_memory = self.encoder(texts)
        return self.decoder.generate(text_memory, max_frames=max_frames)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  SignX Phase 4 — ContinuousMotionDecoder Smoke Test")
    print("=" * 65 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Instantiate decoder only (no encoder needed for unit test) ─────────
    decoder = ContinuousMotionDecoder(
        decoder_dim=512,
        num_heads=8,
        num_layers=4,
        ffn_dim=2048,
        dropout=0.1,
    ).to(device)

    total     = sum(p.numel() for p in decoder.parameters())
    trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Decoder parameters   : {total:,}")
    print(f"All trainable        : {trainable:,}")

    # ── Test 1: Training forward pass ──────────────────────────────────────
    print("\n── Test 1: Training forward pass (teacher forcing) ─────────────")
    B, T = 4, 80         # batch of 4, 80 frames each
    dummy_memory = torch.randn(B, 512, device=device)          # fake text emb
    dummy_motion = torch.randn(B, T, NUM_KEYPOINTS, COORDS, device=device)
    dummy_mask   = torch.ones(B, T, dtype=torch.bool, device=device)

    pred_motion, pred_length = decoder(dummy_memory, dummy_motion, dummy_mask)

    print(f"Input  text_memory   : {tuple(dummy_memory.shape)}")
    print(f"Input  target_motion : {tuple(dummy_motion.shape)}")
    print(f"Output pred_motion   : {tuple(pred_motion.shape)}")
    print(f"Output pred_length   : {tuple(pred_length.shape)}")

    assert pred_motion.shape == (B, T, NUM_KEYPOINTS, COORDS), \
        f"Expected {(B, T, NUM_KEYPOINTS, COORDS)}, got {pred_motion.shape}"
    assert pred_length.shape == (B,)
    print("Shapes ✓")

    # ── Test 2: Loss backward ──────────────────────────────────────────────
    print("\n── Test 2: Loss backward (gradients flow?) ─────────────────────")
    loss = pred_motion.mean() + pred_length.mean()
    loss.backward()

    # Check output projection has gradients
    grad_norm = decoder.output_proj.weight.grad.norm().item()
    print(f"Output projection gradient norm : {grad_norm:.6f}")
    assert grad_norm > 0, "No gradients in output projection!"
    print("Gradients flowing ✓")

    # ── Test 3: Autoregressive generation ─────────────────────────────────
    print("\n── Test 3: Autoregressive generation (inference) ───────────────")
    dummy_memory_inf = torch.randn(2, 512, device=device)

    generated = decoder.generate(dummy_memory_inf, max_frames=450)

    print(f"Input  text_memory : {tuple(dummy_memory_inf.shape)}")
    print(f"Output generated   : {tuple(generated.shape)}")
    assert generated.ndim == 4
    assert generated.shape[0] == 2
    assert generated.shape[2] == NUM_KEYPOINTS
    assert generated.shape[3] == COORDS
    print(f"Generated {generated.shape[1]} frames per clip ✓")

    # ── Test 4: Causal mask ────────────────────────────────────────────────
    print("\n── Test 4: Causal mask shape ───────────────────────────────────")
    mask = decoder._causal_mask(5, device)
    print("Causal mask (5×5):")
    print(mask.int())
    # Expected:
    # [[0, 1, 1, 1, 1],
    #  [0, 0, 1, 1, 1],
    #  [0, 0, 0, 1, 1],
    #  [0, 0, 0, 0, 1],
    #  [0, 0, 0, 0, 0]]
    assert mask[0, 1] == True   # frame 0 cannot see frame 1
    assert mask[1, 0] == False  # frame 1 CAN see frame 0
    print("Causal masking ✓")

    # ── Test 5: Parameter count summary ───────────────────────────────────
    print("\n── Test 5: Full model parameter summary ────────────────────────")
    print(f"  Length estimator : "
          f"{sum(p.numel() for p in decoder.length_estimator.parameters()):,}")
    print(f"  Input projection : "
          f"{sum(p.numel() for p in decoder.input_proj.parameters()):,}")
    print(f"  Transformer      : "
          f"{sum(p.numel() for p in decoder.transformer_decoder.parameters()):,}")
    print(f"  Output projection: "
          f"{sum(p.numel() for p in decoder.output_proj.parameters()):,}")
    print(f"  ─────────────────")
    print(f"  Total decoder    : {total:,}")

    print("\n" + "=" * 65)
    print("  All tests PASSED ✓")
    print("=" * 65)

    print("""
── Usage in training loop (Phase 5) ─────────────────────────────────
decoder = ContinuousMotionDecoder().to(device)

# Training step:
pred_motion, pred_length = decoder(
    text_memory   = encoder(batch['texts']),     # (B, 512)
    target_motion = batch['motion_tensors'],     # (B, T, 543, 3)
    motion_mask   = batch['motion_masks'],       # (B, T)
)

# Inference:
generated = decoder.generate(
    text_memory = encoder(["How are you?"]),     # (1, 512)
    max_frames  = 450,
)
# generated shape: (1, L, 543, 3)
""")