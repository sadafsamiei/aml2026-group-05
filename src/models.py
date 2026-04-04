"""
models.py
---------
Two model classes used in this project:

  AudioTransformer              — Model 2 (neural baseline, mean pooling)
  AttentivePoolingTransformer   — Model 3 (main model, query-based pooling)

Both share the identical Transformer encoder backbone. The only difference
is in how the frame-level encoder outputs are pooled into a single vector
before the classification head.

Pooling comparison
------------------
Mean pooling  : pool = mean(enc_outputs, over real frames)
               Simple, parameter-free, treats all frames equally.

Attentive pooling : pool = weighted_sum(enc_outputs)
                    weights = softmax(score(query, enc_output))
                    The query biases which frames matter most.

Query source (controlled by use_external_meta)
----------------------------------------------
  use_external_meta = False  (current default)
      A single learnable parameter vector acts as the query.
      This isolates the architecture contribution of attentive pooling
      from any metadata effect. Use this when metadata = participant
      identity (current state: 1 participant with PNGs, age = identity).

  use_external_meta = True
      Metadata (age + gender) is projected to the query vector.
      Use this once the full dataset has >= 4 participants and age/gender
      do not uniquely identify individuals.
      Switch: set USE_EXTERNAL_META = True in config.py and pass
      meta tensors through the dataset and training loop.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared encoder builder
# ---------------------------------------------------------------------------

def _build_encoder(d_model: int, nhead: int, num_layers: int,
                   dropout: float) -> nn.TransformerEncoder:
    """Build a pre-norm TransformerEncoder (shared by both model classes)."""
    assert d_model % nhead == 0, \
        f"d_model ({d_model}) must be divisible by nhead ({nhead})"
    layer = nn.TransformerEncoderLayer(
        d_model         = d_model,
        nhead           = nhead,
        dim_feedforward = d_model * 4,
        dropout         = dropout,
        activation      = "relu",
        batch_first     = True,
        norm_first      = True,   # pre-LN: more stable at small N
    )
    return nn.TransformerEncoder(
        layer, num_layers=num_layers, enable_nested_tensor=False
    )


def _init_weights(module):
    """Xavier-uniform for Linear, normal for Embedding."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


# ---------------------------------------------------------------------------
# Model 2: AudioTransformer (mean pooling — neural baseline)
# ---------------------------------------------------------------------------

class AudioTransformer(nn.Module):
    """
    Transformer encoder with masked mean pooling.
    Implements Model 2 (neural baseline) from PROBLEMSETTING.md.

    Architecture
    ------------
    (B, T, F) → input_proj → pos_enc → TransformerEncoder
              → masked mean pool → dropout → Linear → (B, num_classes)
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = nn.Embedding(max_len, d_model)
        self.encoder    = _build_encoder(d_model, nhead, num_layers, dropout)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x            : (B, T, F_DIM)
        padding_mask : (B, T) bool — True = padded frame (ignored)
        returns      : (B, num_classes) logits
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.input_proj(x) + self.pos_enc(pos)
        x   = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask_f = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)

        return self.classifier(self.dropout(pooled))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        layer = self.encoder.layers[0]
        print("=" * 52)
        print("AudioTransformer  (mean pooling)")
        print("=" * 52)
        print(f"  input_proj  : {self.input_proj.in_features} → {self.input_proj.out_features}")
        print(f"  d_model     : {layer.self_attn.embed_dim}")
        print(f"  nhead       : {layer.self_attn.num_heads}")
        print(f"  num_layers  : {len(self.encoder.layers)}")
        print(f"  ffn_dim     : {layer.linear1.out_features}")
        print(f"  dropout     : {layer.dropout.p}")
        print(f"  num_classes : {self.classifier.out_features}")
        print(f"  pooling     : mean (masked)")
        print(f"  Parameters  : {self.count_parameters():,}")
        print("=" * 52)


# ---------------------------------------------------------------------------
# Model 3: AttentivePoolingTransformer (query pooling — main model)
# ---------------------------------------------------------------------------

class AttentivePoolingTransformer(nn.Module):
    """
    Transformer encoder with query-based attentive pooling.
    Implements Model 3 (main model) from PROBLEMSETTING.md.

    Architecture
    ------------
    (B, T, F) → input_proj → pos_enc → TransformerEncoder → enc  (B, T, d)
    query     → (from meta_proj(meta) or learned_query)           (B, 1, d)
    scores    = dot(enc, query)                                    (B, T, 1)
    weights   = softmax(scores, masked)                            (B, T, 1)
    pooled    = sum(enc * weights, dim=T)                          (B, d)
    logits    = Linear(dropout(pooled))                            (B, num_classes)

    Query source
    ------------
    use_external_meta = False  (default — current dataset)
        self.learned_query : nn.Parameter of shape (1, 1, d_model)
        Trained end-to-end. Isolates pooling architecture effect from
        metadata. Use when metadata = participant identity.

    use_external_meta = True   (switch when full dataset arrives)
        self.meta_proj : Linear(meta_dim → d_model) + LayerNorm + Tanh
        meta vector (age_normalised, gender_binary) → query.
        Use when >= 4 participants and age/gender are not collinear
        with participant identity.

    Parameters
    ----------
    input_dim        : F_DIM (frequency bins from spectrogram audit)
    d_model          : Transformer hidden dimension
    nhead            : attention heads (d_model must be divisible by nhead)
    num_layers       : encoder depth
    dropout          : applied inside encoder and before head
    num_classes      : output classes (4 for full dataset, 2 currently)
    max_len          : positional embedding table size (> MAX_LEN is safe)
    use_external_meta: False = learned query; True = metadata-driven query
    meta_dim         : metadata vector size (2 = [age, gender]; ignored if
                       use_external_meta=False)
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, max_len: int = 512,
                 use_external_meta: bool = False, meta_dim: int = 2):
        super().__init__()

        self.use_external_meta = use_external_meta

        # Shared backbone (identical to AudioTransformer)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = nn.Embedding(max_len, d_model)
        self.encoder    = _build_encoder(d_model, nhead, num_layers, dropout)

        # ── Query source ─────────────────────────────────────────────────
        if use_external_meta:
            # Metadata → query vector
            # LayerNorm stabilises the projection; Tanh bounds the query
            # so it doesn't dominate the dot product with encoder outputs.
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Tanh(),
            )
        else:
            # Single learned query parameter — one vector shared across
            # all examples in a batch, broadcast to (B, 1, d_model).
            # This is the default for the current dataset.
            self.learned_query = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.learned_query, std=0.02)

        # ── Classification head ───────────────────────────────────────────
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialise all Linear and Embedding layers
        # (learned_query is initialised above; meta_proj is covered here)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                meta: torch.Tensor = None) -> tuple:
        """
        Parameters
        ----------
        x            : (B, T, F_DIM) spectrogram frames
        padding_mask : (B, T) bool — True = padded, ignored in attention
        meta         : (B, meta_dim) float — required if use_external_meta=True,
                       ignored otherwise

        Returns
        -------
        logits  : (B, num_classes)
        weights : (B, T) attention weights (for analysis and visualisation)
        """
        B, T, _ = x.shape

        # ── Encoder ───────────────────────────────────────────────────────
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.input_proj(x) + self.pos_enc(pos)              # (B, T, d)
        enc = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d)

        # ── Query ─────────────────────────────────────────────────────────
        if self.use_external_meta:
            assert meta is not None, \
                "meta tensor required when use_external_meta=True"
            query = self.meta_proj(meta).unsqueeze(1)              # (B, 1, d)
        else:
            query = self.learned_query.expand(B, -1, -1)           # (B, 1, d)

        # ── Attentive pooling ─────────────────────────────────────────────
        # Scaled dot product: query · enc^T / sqrt(d)
        d = enc.shape[-1]
        scores = (enc * query).sum(dim=-1, keepdim=True) / (d ** 0.5)  # (B, T, 1)

        # Mask padded positions to -inf before softmax
        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )

        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        pooled  = (enc * weights).sum(dim=1)        # (B, d)

        logits = self.classifier(self.dropout(pooled))  # (B, num_classes)
        return logits, weights.squeeze(-1)           # weights: (B, T)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        layer = self.encoder.layers[0]
        mode  = ("external metadata" if self.use_external_meta
                 else "learned query (no external metadata)")
        print("=" * 52)
        print("AttentivePoolingTransformer")
        print("=" * 52)
        print(f"  input_proj  : {self.input_proj.in_features} → {self.input_proj.out_features}")
        print(f"  d_model     : {layer.self_attn.embed_dim}")
        print(f"  nhead       : {layer.self_attn.num_heads}")
        print(f"  num_layers  : {len(self.encoder.layers)}")
        print(f"  ffn_dim     : {layer.linear1.out_features}")
        print(f"  dropout     : {layer.dropout.p}")
        print(f"  num_classes : {self.classifier.out_features}")
        print(f"  pooling     : attentive ({mode})")
        print(f"  Parameters  : {self.count_parameters():,}")
        print("=" * 52)
