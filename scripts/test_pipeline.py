"""
test_pipeline.py  [FIXED]
----------------
Quick end-to-end functionality test (< 60 seconds, no GPU needed).

Usage (from repo root):
    python scripts/test_pipeline.py

If test [1] fails, run first:
    python scripts/diagnose_paths.py
to understand the exact filename mismatch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from data_utils import (
    build_master_table,
    audit_spectrograms,
    determine_eval_protocol,
    build_splits,
    SpectrogramDataset,
)
from models import AudioTransformer
from train_utils import (
    train_one_epoch,
    evaluate,
    sanity_check_forward,
    sanity_check_dataloader,
)


def check(step: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗ FAILED"
    print(f"  [{step:>2}] {status}  {detail}")
    if not passed:
        sys.exit(1)


def main():
    print("=" * 55)
    print("PIPELINE FUNCTIONALITY TEST")
    print("=" * 55)

    cfg.print_config()
    device = torch.device("cpu")
    cfg.make_output_dirs()

    # ── [1] Master table ──────────────────────────────────────────────
    print("\n--- Data loading ---")
    try:
        df = build_master_table(verbose=True)
        n_with_png = df["png_exists"].sum()
        n_total    = len(df)

        if n_with_png == 0:
            print(f"\n  ✗ FAILED: 0 PNGs found out of {n_total} examples.")
            print(f"\n  Diagnosis steps:")
            print(f"    1. Run:  python scripts/diagnose_paths.py")
            print(f"    2. Check SPEC_DIR in src/config.py: {cfg.SPEC_DIR}")
            print(f"    3. Run:  ls {cfg.SPEC_DIR} | head -10")
            sys.exit(1)

        if n_with_png < n_total:
            print(f"\n  ⚠  {n_total - n_with_png} examples have no matching PNG "
                  f"({n_with_png}/{n_total} found). "
                  f"Run diagnose_paths.py for details.")

        df_ok = df[df["png_exists"]].copy()
        check("1", len(df_ok) > 0,
              f"{len(df_ok)}/{n_total} examples with PNGs found")
    except FileNotFoundError as e:
        print(f"\n  ✗ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ✗ FAILED: {e}")
        sys.exit(1)

    # ── [2] Spectrogram audit ─────────────────────────────────────────
    try:
        sample_paths = df_ok["png_path"].tolist()[:min(20, len(df_ok))]
        stats   = audit_spectrograms(sample_paths, verbose=True)
        f_dim   = stats["f_dim"]
        max_len = stats["max_len_p95"]
        check("2", f_dim > 0 and max_len > 0,
              f"F={f_dim}, MAX_LEN={max_len}")
    except Exception as e:
        check("2", False, str(e))

    # ── [3] Splits ────────────────────────────────────────────────────
    print("\n--- Split construction ---")
    try:
        protocol = determine_eval_protocol(df_ok)
        folds    = build_splits(df_ok, protocol, save=False)
        check("3", len(folds) > 0,
              f"protocol='{protocol}', {len(folds)} fold(s)")
    except Exception as e:
        check("3", False, str(e))

    train_df, val_df, _ = folds[0]

    # ── [4] Dataset __getitem__ ───────────────────────────────────────
    print("\n--- Dataset and DataLoader ---")
    try:
        ds = SpectrogramDataset(train_df, cfg.LABEL_ENCODER, max_len, f_dim)
        x, mask, y = ds[0]
        ok = (
            x.shape    == (max_len, f_dim) and
            mask.shape == (max_len,) and
            mask.dtype == torch.bool and
            y.dtype    == torch.int64 and
            0.0 <= x.min().item() and x.max().item() <= 1.0
        )
        check("4", ok,
              f"x={x.shape}, mask={mask.shape}, y={y.item()}, "
              f"x_range=[{x.min():.3f}, {x.max():.3f}]")
    except Exception as e:
        check("4", False, str(e))

    # ── [5] DataLoader batch ─────────────────────────────────────────
    try:
        loader = DataLoader(ds, batch_size=min(4, len(ds)),
                            shuffle=True, num_workers=0)
        sanity_check_dataloader(loader, f_dim, max_len)
        check("5", True, "batch shapes verified")
    except Exception as e:
        check("5", False, str(e))

    # ── [6] Model forward pass ────────────────────────────────────────
    print("\n--- Model ---")
    try:
        # Tiny model for the test
        nhead = 4 if f_dim >= 4 else 1
        model = AudioTransformer(
            input_dim=f_dim, d_model=16, nhead=nhead,
            num_layers=1, dropout=0.1,
            num_classes=cfg.N_CLASSES, max_len=max_len + 50,
        ).to(device)
        ok = sanity_check_forward(model, f_dim, max_len, cfg.N_CLASSES, device)
        check("6", ok, f"{model.count_parameters():,} parameters")
    except Exception as e:
        check("6", False, str(e))

    # ── [7] Loss ──────────────────────────────────────────────────────
    print("\n--- Training mechanics ---")
    try:
        weights   = ds.class_weights().to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        x_b, m_b, y_b = next(iter(loader))
        logits = model(x_b.to(device), padding_mask=m_b.to(device))
        loss   = criterion(logits, y_b.to(device))
        ok     = not torch.isnan(loss).item() and loss.item() > 0
        check("7", ok, f"loss={loss.item():.4f}")
    except Exception as e:
        check("7", False, str(e))

    # ── [8] Backward pass ─────────────────────────────────────────────
    try:
        loss.backward()
        grad_ok = all(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters() if p.requires_grad
        )
        check("8", grad_ok, "all gradients finite")
    except Exception as e:
        check("8", False, str(e))

    # ── [9] 2-epoch training ──────────────────────────────────────────
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.zero_grad()
        losses = []
        for epoch in range(2):
            stats = train_one_epoch(model, loader, optimizer, criterion, device)
            losses.append(stats["loss"])
            print(f"    Epoch {epoch+1}: loss={stats['loss']:.4f}")
        check("9", all(np.isfinite(l) for l in losses),
              f"losses={[round(l,4) for l in losses]}")
    except Exception as e:
        check("9", False, str(e))

    # ── [10] Evaluation ───────────────────────────────────────────────
    try:
        val_ds     = SpectrogramDataset(val_df, cfg.LABEL_ENCODER, max_len, f_dim)
        val_loader = DataLoader(val_ds, batch_size=min(16, len(val_ds)),
                                shuffle=False, num_workers=0)
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
