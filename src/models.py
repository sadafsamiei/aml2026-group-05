"""
models.py
---------
Two model classes used in this project:

  AudioTransformer              — Model 2 (neural baseline, mean pooling)
  AttentivePoolingTransformer   — Model 3 (main model, query-based pooling)

Both share the identical Transformer encoder backbone. The only difference
is in how the frame-level encoder outputs are pooled into a single vector
before the classification head.

Extension (v2): separate dropout rates at four independent locations
--------------------------------------------------------------------
Previously a single `dropout` rate was shared between the encoder layers
and the head. Following reviewer feedback, v2 introduces per-location rates:

  dropout_encoder   — inside TransformerEncoderLayer (attention + FFN)
                      same semantic as the original `dropout` parameter
  dropout_input     — applied after input_proj + pos_enc, before encoder
                      regularises the token embeddings as they enter the
                      encoder stack
  dropout_head      — applied after pooling, before the classifier head
                      directly regularises the pooled representation
  dropout_meta      — (Model 3 only) applied to the metadata query after
                      meta_proj; forces the model not to over-rely on any
                      single metadata dimension

Backward compatibility
----------------------
All new parameters default to their v1 equivalents so existing configs
and saved checkpoints are not broken:
  dropout_encoder = dropout   (same as before)
  dropout_input   = 0.0       (new; default = no extra regularisation)
  dropout_head    = dropout   (same as before)
  dropout_meta    = 0.0       (new; default = no extra regularisation)

The original `dropout` kwarg is preserved as the fallback for
dropout_encoder and dropout_head when the new params are not supplied.

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

  use_external_meta = True
      Metadata (age + gender) is projected to the query vector.
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
        dropout         = dropout,       # dropout_encoder goes here
        activation      = "relu",
        batch_first     = True,
        norm_first      = True,          # pre-LN: more stable at small N
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

    Architecture (v2 — separate dropout rates)
    -------------------------------------------
    (B, T, F) → input_proj → pos_enc
              → dropout_input                   ← NEW in v2
              → TransformerEncoder (dropout_encoder inside each layer)
              → masked mean pool
              → dropout_head                    ← was `dropout` in v1
              → Linear → (B, num_classes)

    Parameters
    ----------
    dropout         : float — legacy fallback for dropout_encoder + dropout_head
    dropout_encoder : float — inside TransformerEncoderLayer (default=dropout)
    dropout_input   : float — after input proj, before encoder    (default=0.0)
    dropout_head    : float — after pooling, before classifier    (default=dropout)
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, max_len: int = 512,
                 # ── v2 separate dropout rates ──
                 dropout_encoder: float = None,
                 dropout_input:   float = 0.0,
                 dropout_head:    float = None):
        super().__init__()

        # Resolve defaults — fall back to `dropout` for backward compat
        d_enc  = dropout if dropout_encoder is None else dropout_encoder
        d_head = dropout if dropout_head    is None else dropout_head

        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_enc       = nn.Embedding(max_len, d_model)
        self.dropout_input = nn.Dropout(dropout_input)   # NEW v2
        self.encoder       = _build_encoder(d_model, nhead, num_layers, d_enc)
        self.dropout_head  = nn.Dropout(d_head)          # was self.dropout
        self.classifier    = nn.Linear(d_model, num_classes)
        self.apply(_init_weights)

        # Store for introspection / printing
        self._d_enc   = d_enc
        self._d_input = dropout_input
        self._d_head  = d_head

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x            : (B, T, F_DIM)
        padding_mask : (B, T) bool — True = padded frame (ignored)
        returns      : (B, num_classes) logits
        """
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        x = self.input_proj(x) + self.pos_enc(pos)    # (B, T, d)
        x = self.dropout_input(x)                      # dropout_input — v2
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            mask_f = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = x.mean(1)

        return self.classifier(self.dropout_head(pooled))    # dropout_head — v2

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        layer = self.encoder.layers[0]
        print("=" * 56)
        print("AudioTransformer  (mean pooling)  [v2 — multi-dropout]")
        print("=" * 56)
        print(f"  input_proj      : {self.input_proj.in_features} → {self.input_proj.out_features}")
        print(f"  d_model         : {layer.self_attn.embed_dim}")
        print(f"  nhead           : {layer.self_attn.num_heads}")
        print(f"  num_layers      : {len(self.encoder.layers)}")
        print(f"  ffn_dim         : {layer.linear1.out_features}")
        print(f"  dropout_input   : {self._d_input}")
        print(f"  dropout_encoder : {self._d_enc}")
        print(f"  dropout_head    : {self._d_head}")
        print(f"  num_classes     : {self.classifier.out_features}")
        print(f"  pooling         : mean (masked)")
        print(f"  Parameters      : {self.count_parameters():,}")
        print("=" * 56)


# ---------------------------------------------------------------------------
# Model 3: AttentivePoolingTransformer (query pooling — main model)
# ---------------------------------------------------------------------------

class AttentivePoolingTransformer(nn.Module):
    """
    Transformer encoder with query-based attentive pooling.
    Implements Model 3 (main model) from PROBLEMSETTING.md.

    Architecture (v2 — separate dropout rates)
    -------------------------------------------
    (B, T, F) → input_proj → pos_enc
              → dropout_input                   ← NEW in v2
              → TransformerEncoder (dropout_encoder inside each layer)
              → enc   (B, T, d)
    query     → meta_proj(meta) or learned_query
              → dropout_meta                    ← NEW in v2 (Model 3 only)
    scores    = dot(enc, query) · exp(log_tau)
    weights   = softmax(scores, masked)
    pooled    = sum(enc * weights)
              → dropout_head                    ← was `dropout` in v1
              → Linear → (B, num_classes)

    Parameters
    ----------
    dropout         : float — legacy fallback for dropout_encoder + dropout_head
    dropout_encoder : float — inside TransformerEncoderLayer  (default=dropout)
    dropout_input   : float — after input proj, before encoder (default=0.0)
    dropout_head    : float — after pooling, before classifier (default=dropout)
    dropout_meta    : float — after meta_proj, on the query    (default=0.0)
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 4, max_len: int = 512,
                 use_external_meta: bool = False, meta_dim: int = 2,
                 # ── v2 separate dropout rates ──
                 dropout_encoder: float = None,
                 dropout_input:   float = 0.0,
                 dropout_head:    float = None,
                 dropout_meta:    float = 0.0):
        super().__init__()

        # Resolve defaults — fall back to `dropout` for backward compat
        d_enc  = dropout if dropout_encoder is None else dropout_encoder
        d_head = dropout if dropout_head    is None else dropout_head

        self.use_external_meta = use_external_meta

        # Shared backbone
        self.input_proj    = nn.Linear(input_dim, d_model)
        self.pos_enc       = nn.Embedding(max_len, d_model)
        self.dropout_input = nn.Dropout(dropout_input)   # NEW v2
        self.encoder       = _build_encoder(d_model, nhead, num_layers, d_enc)

        # ── Query source ─────────────────────────────────────────────────
        if use_external_meta:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.dropout_meta = nn.Dropout(dropout_meta)  # NEW v2
        else:
            self.learned_query = nn.Parameter(
                torch.randn(1, 1, d_model) * (d_model ** -0.5)
            )

        # Learned temperature
        self.log_tau = nn.Parameter(torch.zeros(1))

        # ── Classification head ───────────────────────────────────────────
        self.dropout_head  = nn.Dropout(d_head)          # was self.dropout
        self.classifier    = nn.Linear(d_model, num_classes)

        self.apply(_init_weights)

        # Store for introspection
        self._d_enc   = d_enc
        self._d_input = dropout_input
        self._d_head  = d_head
        self._d_meta  = dropout_meta

    def forward(self, x: torch.Tensor,
                padding_mask: torch.Tensor = None,
                meta: torch.Tensor = None) -> tuple:
        """
        Parameters
        ----------
        x            : (B, T, F_DIM) spectrogram frames
        padding_mask : (B, T) bool — True = padded, ignored in attention
        meta         : (B, meta_dim) float — required if use_external_meta=True

        Returns
        -------
        logits  : (B, num_classes)
        weights : (B, T) attention weights (for analysis)
        """
        B, T, _ = x.shape

        # ── Encoder ───────────────────────────────────────────────────────
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x   = self.input_proj(x) + self.pos_enc(pos)              # (B, T, d)
        x   = self.dropout_input(x)                                # dropout_input — v2
        enc = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d)

        # ── Query ─────────────────────────────────────────────────────────
        if self.use_external_meta:
            assert meta is not None, \
                "meta tensor required when use_external_meta=True"
            query = self.dropout_meta(                             # dropout_meta — v2
                self.meta_proj(meta)
            ).unsqueeze(1)                                         # (B, 1, d)
        else:
            query = self.learned_query.expand(B, -1, -1)           # (B, 1, d)

        # ── Attentive pooling ─────────────────────────────────────────────
        tau    = torch.exp(self.log_tau)
        scores = (enc * query).sum(dim=-1, keepdim=True) * tau    # (B, T, 1)

        if padding_mask is not None:
            scores = scores.masked_fill(
                padding_mask.unsqueeze(-1), float("-inf")
            )

        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        pooled  = (enc * weights).sum(dim=1)        # (B, d)

        logits = self.classifier(self.dropout_head(pooled))       # dropout_head — v2
        return logits, weights.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        layer = self.encoder.layers[0]
        mode  = ("external metadata" if self.use_external_meta
                 else "learned query (no external metadata)")
        print("=" * 56)
        print("AttentivePoolingTransformer  [v2 — multi-dropout]")
        print("=" * 56)
        print(f"  input_proj      : {self.input_proj.in_features} → {self.input_proj.out_features}")
        print(f"  d_model         : {layer.self_attn.embed_dim}")
        print(f"  nhead           : {layer.self_attn.num_heads}")
        print(f"  num_layers      : {len(self.encoder.layers)}")
        print(f"  ffn_dim         : {layer.linear1.out_features}")
        print(f"  dropout_input   : {self._d_input}")
        print(f"  dropout_encoder : {self._d_enc}")
        print(f"  dropout_head    : {self._d_head}")
        print(f"  dropout_meta    : {self._d_meta}  (external meta only)")
        print(f"  num_classes     : {self.classifier.out_features}")
        print(f"  pooling         : attentive ({mode})")
        print(f"  Parameters      : {self.count_parameters():,}")
        print("=" * 56)