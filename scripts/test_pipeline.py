"""
test_pipeline.py
----------------
Quick end-to-end functionality test.
Runs in < 60 seconds without a GPU using 2 epochs and a tiny model.
Use this to verify everything works before launching a full training run.

Usage (from repo root):
    python scripts/test_pipeline.py

Expected output:
    [1] Master table built       ✓
    [2] Spectrograms audited     ✓
    [3] Splits constructed       ✓
    [4] Dataset __getitem__      ✓
    [5] DataLoader batch         ✓
    [6] Model forward pass       ✓
    [7] Loss computed            ✓
    [8] Backward pass            ✓
    [9] 2-epoch training loop    ✓
    [10] Evaluation              ✓
    All tests passed ✓
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from data_utils import (
    build_master_table, audit_spectrograms,
    determine_eval_protocol, build_splits, SpectrogramDataset,
)
from models import AudioTransformer
from train_utils import (
    train_one_epoch, evaluate,
    sanity_check_forward, sanity_check_dataloader,
)


def check(step: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗ FAILED"
    msg    = f"  [{step}] {status}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    if not passed:
        sys.exit(1)


def main():
    print("=" * 55)
    print("PIPELINE FUNCTIONALITY TEST")
    print("=" * 55)

    device = torch.device("cpu")   # always CPU for the quick test
    cfg.make_output_dirs()

    # [1] Master table
    print("\n--- Data loading ---")
    try:
        df = build_master_table(verbose=False)
        df = df[df["png_exists"]].copy()
        check("1", len(df) > 0,
              f"{len(df)} usable examples with PNGs found")
    except Exception as e:
        check("1", False, str(e))

    # [2] Spectrogram audit
    try:
        stats   = audit_spectrograms(df["png_path"].tolist()[:10], verbose=False)
        f_dim   = stats["f_dim"]
        max_len = stats["max_len_p95"]
        check("2", f_dim > 0 and max_len > 0,
              f"F={f_dim}, MAX_LEN={max_len}")
    except Exception as e:
        check("2", False, str(e))

    # [3] Splits
    print("\n--- Split construction ---")
    try:
        protocol = determine_eval_protocol(df)
        folds    = build_splits(df, protocol, save=False)
        check("3", len(folds) > 0,
              f"protocol='{protocol}', {len(folds)} fold(s)")
    except Exception as e:
        check("3", False, str(e))

    train_df, val_df, _ = folds[0]

    # [4] Dataset __getitem__
    print("\n--- Dataset and DataLoader ---")
    try:
        ds = SpectrogramDataset(
            train_df, cfg.LABEL_ENCODER, max_len, f_dim)
        x, mask, y = ds[0]
        ok = (x.shape == (max_len, f_dim) and
              mask.shape == (max_len,) and
              mask.dtype == torch.bool and
              y.dtype == torch.int64 and
              0.0 <= x.min().item() <= x.max().item() <= 1.0)
        check("4", ok, f"x={x.shape}, mask={mask.shape}, y={y.item()}")
    except Exception as e:
        check("4", False, str(e))

    # [5] DataLoader batch
    try:
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        sanity_check_dataloader(loader, f_dim, max_len)
        check("5", True, "batch shapes verified")
    except Exception as e:
        check("5", False, str(e))

    # [6] Model forward pass
    print("\n--- Model ---")
    try:
        model = AudioTransformer(
            input_dim=f_dim, d_model=16, nhead=4,
            num_layers=1, dropout=0.1,
            num_classes=cfg.N_CLASSES, max_len=max_len + 50,
        ).to(device)
        ok = sanity_check_forward(model, f_dim, max_len,
                                  cfg.N_CLASSES, device)
        check("6", ok, f"{model.count_parameters():,} parameters")
    except Exception as e:
        check("6", False, str(e))

    # [7] Loss
    print("\n--- Training mechanics ---")
    try:
        weights   = ds.class_weights().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        x_b, m_b, y_b = next(iter(loader))
        x_b, m_b, y_b = x_b.to(device), m_b.to(device), y_b.to(device)
        logits = model(x_b, padding_mask=m_b)
        loss   = criterion(logits, y_b)
        ok     = (not torch.isnan(loss).item() and loss.item() > 0)
        check("7", ok, f"loss={loss.item():.4f}")
    except Exception as e:
        check("7", False, str(e))

    # [8] Backward pass
    try:
        loss.backward()
        grad_ok = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters() if p.requires_grad
        )
        check("8", grad_ok, "gradients are finite")
    except Exception as e:
        check("8", False, str(e))

    # [9] 2-epoch training loop
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.zero_grad()

        losses = []
        for epoch in range(2):
            stats = train_one_epoch(
                model, loader, optimizer, criterion, device)
            losses.append(stats["loss"])
            print(f"    Epoch {epoch+1}: loss={stats['loss']:.4f}")

        check("9", all(np.isfinite(l) for l in losses),
              f"2 epochs completed, losses={[round(l,4) for l in losses]}")
    except Exception as e:
        check("9", False, str(e))

    # [10] Evaluation
    try:
        val_ds     = SpectrogramDataset(
            val_df, cfg.LABEL_ENCODER, max_len, f_dim)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        result     = evaluate(model, val_loader, device, return_preds=True)
        ok = (0.0 <= result["macro_f1"] <= 1.0 and
              len(result["preds"]) == len(val_df))
        check("10", ok,
              f"val macro-F1={result['macro_f1']:.4f}, "
              f"n_preds={len(result['preds'])}")
    except Exception as e:
        check("10", False, str(e))

    print(f"\n{'='*55}")
    print("All 10 tests passed ✓")
    print("Pipeline is functional. Run the full training with:")
    print("  python scripts/train_transformer_audio.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
