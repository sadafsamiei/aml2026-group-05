"""
models.py
---------
AudioTransformer: small Transformer encoder with masked mean pooling.

This implements the 'neural baseline: audio-only Transformer' from
PROBLEMSETTING.md. The attentive pooling variant (main model) will be
added in a separate module once this baseline is confirmed working.
"""

import math
import torch
import torch.nn as nn


class AudioTransformer(nn.Module):
    """
    Small Transformer encoder over spectrogram time frames with
    masked mean pooling and linear classification head.

    Architecture
    ------------
    Input    : (B, T, F_DIM)  — batch of frame sequences
    Project  : Linear(F_DIM → d_model)  + learned positional embedding
    Encode   : TransformerEncoder  (pre-norm, num_layers layers)
    Pool     : Masked mean over non-padded frames
    Classify : Dropout → Linear(d_model → num_classes)
    Output   : (B, num_classes)  logits

    Parameters
    ----------
    input_dim  : number of frequency bins F (== F_DIM from audit)
    d_model    : Transformer hidden dimension
    nhead      : number of attention heads (d_model must be divisible by nhead)
    num_layers : number of TransformerEncoderLayer blocks
    dropout    : dropout probability (applied inside encoder and before head)
    num_classes: number of output classes (4 for EXTEV/INTER/MOVE/PLTOY)
    max_len    : maximum sequence length (for positional embedding table size)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int    = 64,
        nhead: int      = 4,
        num_layers: int = 2,
        dropout: float  = 0.3,
        num_classes: int = 4,
        max_len: int    = 512,
    ):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        # --- Input projection ---
        self.input_proj = nn.Linear(input_dim, d_model)

        # --- Learned positional encoding ---
        # More robust than sinusoidal for short variable-length sequences
        self.pos_enc = nn.Embedding(max_len, d_model)

        # --- Transformer encoder ---
        # norm_first=True (pre-LN) is more stable with small datasets and
        # deep models than the original post-LN formulation.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward = d_model * 4,
            dropout        = dropout,
            activation     = "relu",
            batch_first    = True,   # input is (B, T, d_model) not (T, B, d_model)
            norm_first     = True,   # pre-norm: more stable at small N
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
            # enable_nested_tensor=False avoids a PyTorch warning with
            # norm_first=True in some versions
            enable_nested_tensor = False,
        )

        # --- Classification head ---
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # --- Weight initialisation ---
        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform for linear layers; normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x            : (B, T, F_DIM) float tensor — spectrogram frames
        padding_mask : (B, T) bool tensor — True at padded positions
                       (same convention as nn.TransformerEncoder's
                       src_key_padding_mask)

        Returns
        -------
        logits : (B, num_classes)
        """
        B, T, _ = x.shape

        # Positional indices: (1, T) → broadcast to (B, T)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # Project input + add positional encoding
        x = self.input_proj(x) + self.pos_enc(pos)   # (B, T, d_model)

        # Transformer encoder
        # padding_mask (B, T): True = ignore this position
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # Masked mean pooling — exclude padded frames from the average
        if padding_mask is not None:
            # real positions: 1.0; padded: 0.0
            mask_float = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)
            pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)    # (B, d_model)

        logits = self.classifier(self.dropout(pooled))   # (B, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_architecture(self):
        """Print a concise summary of the model."""
        print("=" * 50)
        print("AudioTransformer")
        print("=" * 50)
        print(f"  input_proj   : {self.input_proj.in_features} → "
              f"{self.input_proj.out_features}")
        layer = self.encoder.layers[0]
        print(f"  d_model      : {layer.self_attn.embed_dim}")
        print(f"  nhead        : {layer.self_attn.num_heads}")
        print(f"  num_layers   : {len(self.encoder.layers)}")
        print(f"  ffn_dim      : {layer.linear1.out_features}")
        print(f"  dropout      : {layer.dropout.p}")
        print(f"  num_classes  : {self.classifier.out_features}")
        print(f"  Parameters   : {self.count_parameters():,}")
        print("=" * 50)
