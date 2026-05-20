"""
SignX — Phase 3: English Text Encoder
======================================
Takes a batch of English strings and outputs dense sentence embeddings.

Flow:
    List[str]
        ↓  HuggingFace tokenizer
    token ids, attention mask  (B, seq_len)
        ↓  DistilBERT
    last_hidden_state          (B, seq_len, 768)
        ↓  extract [CLS] token (index 0)
    cls_embedding              (B, 768)
        ↓  nn.Linear(768 → decoder_dim)
    projected_embedding        (B, decoder_dim)   ← fed into Phase 4 decoder

Run locally:
    python signx_phase3_encoder.py
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ENCODER MODULE
# ─────────────────────────────────────────────────────────────────────────────

class EnglishTextEncoder(nn.Module):
    """
    Wraps DistilBERT to produce sentence-level embeddings from raw text,
    then projects them to the decoder's hidden dimension.

    Why two separate steps (CLS extraction + linear projection)?
      - CLS extraction: takes the sentence summary DistilBERT already computed
      - Linear projection: learned compression from 768 → decoder_dim (512)
        This lets the model learn WHICH parts of the language embedding
        matter most for generating sign motion — not all 768 dimensions
        are equally useful for this task.

    Output shape: (B, decoder_dim)
    """

    def __init__(
        self,
        model_name:   str   = 'distilbert-base-uncased',
        decoder_dim:  int   = 512,
        freeze_bert:  bool  = True,
        dropout:      float = 0.1,
    ) -> None:
        """
        Args:
            model_name  : HuggingFace model identifier.
                          'distilbert-base-uncased' — 66M params, fast,
                          97% of BERT quality at 60% of the compute cost.
            decoder_dim : Output dimension. Must match the motion decoder's
                          hidden_dim (set to 512 in Phase 4).
            freeze_bert : If True, DistilBERT weights are frozen — gradients
                          do NOT flow through it during training.
                          Why freeze: we have ~30K clips but DistilBERT was
                          pre-trained on 3B words. Fine-tuning it risks
                          overfitting and consumes GPU memory we need for
                          the motion decoder. The projection layer is still
                          trainable and adapts the embeddings for our task.
            dropout     : Applied before the projection layer to regularize
                          the compressed representation.
        """
        super().__init__()

        # ── Tokenizer ──────────────────────────────────────────────────────
        # Converts raw strings → token IDs that DistilBERT understands.
        # Loaded once, lives in CPU memory (no GPU needed for tokenization).
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ── DistilBERT backbone ────────────────────────────────────────────
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size   # 768 for DistilBERT

        # ── Freeze backbone if requested ───────────────────────────────────
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(
                f"[EnglishTextEncoder] DistilBERT frozen — "
                f"only the projection layer will train."
            )
        else:
            print(
                f"[EnglishTextEncoder] DistilBERT trainable — "
                f"use a very low LR (1e-5) to avoid destroying pre-trained weights."
            )

        # ── Projection: 768 → decoder_dim ─────────────────────────────────
        # Why not just pass 768-d directly to the decoder?
        #   1. The decoder is designed for decoder_dim=512 — changing it
        #      would double the parameter count of every attention layer.
        #   2. This layer is learnable — it learns which dimensions of the
        #      language embedding matter most for sign motion generation.
        #   3. It acts as a bottleneck forcing a compact representation.
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert_dim, decoder_dim),
            nn.GELU(),
            # WHY GELU over ReLU: GELU doesn't hard-zero negative values.
            # It softly gates them. Works better with transformer-style
            # representations where negative activations carry information.
        )

        self.decoder_dim = decoder_dim

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Convert a batch of English sentences to decoder-ready embeddings.

        Args:
            texts : List[str] of length B

        Returns:
            embeddings : FloatTensor (B, decoder_dim)
        """
        # ── Step 1: Tokenize ───────────────────────────────────────────────
        # padding=True        → pad all sequences to the longest in this batch
        # truncation=True     → clip sequences > model's max length (512 tokens)
        # return_tensors='pt' → return PyTorch tensors
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )

        # ── Step 2: Move to same device as model ───────────────────────────
        # Tokenizer always runs on CPU. We must move its output to the same
        # device as the model weights before the forward pass.
        device = next(self.bert.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # ── Step 3: DistilBERT forward pass ───────────────────────────────
        # When frozen, wrap in torch.no_grad() to skip gradient computation.
        # This saves significant GPU memory during training.
        if not any(p.requires_grad for p in self.bert.parameters()):
            with torch.no_grad():
                bert_out = self.bert(**encoded)
        else:
            bert_out = self.bert(**encoded)

        last_hidden = bert_out.last_hidden_state   # (B, seq_len, 768)

        # ── Step 4: Extract [CLS] token ────────────────────────────────────
        # Index 0 along the sequence dimension.
        # DistilBERT was pre-trained to accumulate the full sentence meaning
        # into this special token — it's the best single-vector summary
        # of the entire input sentence.
        cls_embedding = last_hidden[:, 0, :]       # (B, 768)

        # ── Step 5: Project to decoder_dim ────────────────────────────────
        projected = self.projection(cls_embedding) # (B, decoder_dim)

        return projected


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "=" * 65)
    print("  SignX Phase 3 — EnglishTextEncoder Smoke Test")
    print("=" * 65 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Instantiate ────────────────────────────────────────────────────────
    encoder = EnglishTextEncoder(
        model_name='distilbert-base-uncased',
        decoder_dim=512,
        freeze_bert=True,
        dropout=0.1,
    ).to(device)

    # ── Parameter counts ───────────────────────────────────────────────────
    total     = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}  (projection layer only)")
    print(f"Frozen parameters    : {total - trainable:,}  (DistilBERT backbone)")

    # ── Test 1: basic forward pass ─────────────────────────────────────────
    print("\n── Test 1: Forward pass ────────────────────────────────────────")
    dummy_texts = [
        "How are you doing today?",
        "I want to go to the store.",
        "The weather is nice outside.",
        "Please sign slowly so I can understand.",
    ]
    with torch.no_grad():
        embeddings = encoder(dummy_texts)

    print(f"Input  : {len(dummy_texts)} sentences")
    print(f"Output : {embeddings.shape}")
    assert embeddings.shape == (4, 512), \
        f"Expected (4, 512), got {embeddings.shape}"
    print("Output shape (4, 512) ✓")

    # ── Test 2: semantic similarity sanity check ───────────────────────────
    # Two semantically similar sentences should produce closer embeddings
    # than two unrelated sentences. Validates the CLS token is meaningful.
    print("\n── Test 2: Semantic similarity ─────────────────────────────────")
    sentences = [
        "How are you?",
        "How are you doing?",       # semantically close to sentence 0
        "The cat sat on the mat.",  # unrelated
    ]
    with torch.no_grad():
        embs = encoder(sentences)

    cos = nn.CosineSimilarity(dim=0)
    sim_01 = cos(embs[0], embs[1]).item()
    sim_02 = cos(embs[0], embs[2]).item()
    print(f"'How are you?' vs 'How are you doing?' : {sim_01:.4f}  (expect higher)")
    print(f"'How are you?' vs 'The cat sat...'     : {sim_02:.4f}  (expect lower)")
    assert sim_01 > sim_02, "Semantic similarity ordering failed"
    print("Semantic ordering ✓")

    # ── Test 3: single sentence (inference mode) ───────────────────────────
    print("\n── Test 3: Single sentence ─────────────────────────────────────")
    with torch.no_grad():
        single = encoder(["I love sign language."])
    assert single.shape == (1, 512)
    print(f"Shape: {single.shape} ✓")

    # ── Test 4: gradient flow ──────────────────────────────────────────────
    # Projection layer SHOULD have gradients.
    # DistilBERT backbone should NOT (frozen).
    print("\n── Test 4: Gradient flow ───────────────────────────────────────")
    enc2 = EnglishTextEncoder(freeze_bert=True).to(device)
    out  = enc2(["Gradient flow test."])
    out.sum().backward()

    proj_grad = enc2.projection[1].weight.grad
    assert proj_grad is not None, "Projection layer missing gradient!"
    print(f"Projection gradient norm : {proj_grad.norm().item():.6f} ✓")

    bert_grad = list(enc2.bert.parameters())[0].grad
    assert bert_grad is None, "DistilBERT should be frozen!"
    print("DistilBERT backbone frozen (no gradients) ✓")

    print("\n" + "=" * 65)
    print("  All tests PASSED ✓")
    print("=" * 65)

    print("""
── Usage in training loop (Phase 5) ─────────────────────────────────
encoder = EnglishTextEncoder(
    model_name  = 'distilbert-base-uncased',
    decoder_dim = 512,
    freeze_bert = True,
    dropout     = 0.1,
).to(device)

# In the training loop:
text_embeddings = encoder(batch['texts'])   # (B, 512)
# Pass text_embeddings to the motion decoder as memory
""")