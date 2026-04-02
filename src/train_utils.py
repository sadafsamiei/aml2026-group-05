"""
train_utils.py
--------------
Training loop, evaluation, early stopping, and results logging
for the audio-only Transformer baseline.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import CLASS_NAMES, LABEL_DECODER


# ---------------------------------------------------------------------------
# Training: one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict:
    """
    Run one training epoch.

    Returns
    -------
    dict with keys: 'loss', 'grad_norm'
    """
    model.train()
    total_loss  = 0.0
    total_grad  = 0.0
    n_batches   = 0

    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x, padding_mask=mask)
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping — prevents exploding gradients at small batch sizes
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_grad += grad_norm.item()
        n_batches  += 1

    return {
        "loss"     : total_loss / max(n_batches, 1),
        "grad_norm": total_grad / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    return_preds: bool = False,
) -> dict:
    """
    Evaluate the model on a data loader.

    Returns
    -------
    dict with keys: 'macro_f1', 'per_class_f1', and optionally 'preds', 'labels'
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, mask, y in loader:
            x, mask = x.to(device), mask.to(device)
            logits  = model(x, padding_mask=mask)
            preds   = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.tolist())

    macro_f1     = f1_score(all_labels, all_preds,
                            average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds,
                            average=None, zero_division=0,
                            labels=list(range(len(CLASS_NAMES))))

    result = {
        "macro_f1"    : macro_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1.tolist())),
    }
    if return_preds:
        result["preds"]  = all_preds
        result["labels"] = all_labels

    return result


# ---------------------------------------------------------------------------
# Full training loop with early stopping
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    lr: float          = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int        = 100,
    patience: int      = 15,
    save_path: Path    = None,
    verbose: bool      = True,
) -> dict:
    """
    Train with AdamW + ReduceLROnPlateau scheduler + early stopping.

    Early stopping monitors val macro-F1 (higher is better).
    Best checkpoint is saved to save_path if provided.

    Returns
    -------
    dict with keys:
        'best_val_f1'  : float
        'stopped_epoch': int
        'history'      : dict of lists (train_loss, val_f1 per epoch)
        'elapsed_s'    : float
    """
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_f1    = -1.0
    patience_count = 0
    stopped_epoch  = epochs
    history        = {"train_loss": [], "val_f1": [], "grad_norm": [], "lr": []}
    t0             = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        # --- Validate ---
        val_stats = evaluate(model, val_loader, device)
        val_f1    = val_stats["macro_f1"]

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Record history ---
        history["train_loss"].append(train_stats["loss"])
        history["val_f1"].append(val_f1)
        history["grad_norm"].append(train_stats["grad_norm"])
        history["lr"].append(current_lr)

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1

        # --- Logging ---
        if verbose and (epoch % 5 == 0 or epoch == 1 or
                        patience_count >= patience):
            per_cls = " | ".join(
                f"{c}:{v:.2f}"
                for c, v in val_stats["per_class_f1"].items()
            )
            print(
                f"  Epoch {epoch:3d} | "
                f"loss: {train_stats['loss']:.4f} | "
                f"val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                f"patience: {patience_count}/{patience} | "
                f"lr: {current_lr:.2e}"
            )
            if verbose and epoch % 20 == 0:
                print(f"           per-class → {per_cls}")

        if patience_count >= patience:
            if verbose:
                print(f"  ↳ Early stopping at epoch {epoch}")
            stopped_epoch = epoch
            break

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Training complete: best val macro-F1 = {best_val_f1:.4f} "
              f"(epoch {stopped_epoch - patience}, {elapsed:.1f}s)")

    return {
        "best_val_f1"  : best_val_f1,
        "stopped_epoch": stopped_epoch,
        "history"      : history,
        "elapsed_s"    : elapsed,
    }


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def print_classification_report(preds, labels, split_name: str = ""):
    """
    Print a per-class classification report.

    Uses only the integer class IDs actually present in labels+preds so the
    report never crashes when the current data subset contains fewer classes
    than the full label set (e.g. only INTER and PLTOY when MOVE and EXTEV
    have no spectrograms yet).
    """
    present_ids   = sorted(set(labels) | set(preds))
    present_names = [LABEL_DECODER.get(i, str(i)) for i in present_ids]

    if len(present_ids) < len(CLASS_NAMES):
        missing = [LABEL_DECODER.get(i, str(i))
                   for i in range(len(CLASS_NAMES)) if i not in present_ids]
        print(f"  \u2139  Classes absent from this split "
              f"(no spectrograms yet): {missing}")

    title = f"Classification Report \u2014 {split_name}" if split_name else \
            "Classification Report"
    print(f"\n" + "="*55)
    print(title)
    print("="*55)
    print(classification_report(
        labels, preds,
        labels       = present_ids,
        target_names = present_names,
        digits       = 3,
        zero_division= 0,
    ))


def save_results(results: dict, path: Path):
    """Save a results dict as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to Python natives for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# Sanity checks (call before any training run)
# ---------------------------------------------------------------------------

def sanity_check_forward(model, f_dim: int, max_len: int, n_classes: int,
                          device: torch.device):
    """
    Run a dummy forward pass to verify:
    - Output shape is correct
    - No NaN in output
    - Padding mask is handled
    Prints PASS or FAIL for each check.
    """
    print("\n--- Forward pass sanity checks ---")
    model.eval()
    B = 4
    dummy_x    = torch.randn(B, max_len, f_dim).to(device)
    dummy_mask = torch.zeros(B, max_len, dtype=torch.bool).to(device)
    dummy_mask[:, -20:] = True   # simulate padding on the last 20 frames

    try:
        with torch.no_grad():
            logits = model(dummy_x, padding_mask=dummy_mask)

        checks = [
            ("Output shape == (B, n_classes)",
             logits.shape == (B, n_classes)),
            ("No NaN in output",
             not torch.isnan(logits).any().item()),
            ("No Inf in output",
             not torch.isinf(logits).any().item()),
            ("Output is float",
             logits.dtype == torch.float32),
        ]
        all_pass = True
        for name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {name}")
            if not passed:
                all_pass = False

        print(f"  Parameter count: {model.count_parameters():,}")
        if model.count_parameters() > 500_000:
            print("  ⚠  Parameter count > 500k — consider reducing d_model "
                  "or num_layers for this dataset size")

    except Exception as e:
        print(f"  ✗ Forward pass crashed: {e}")
        all_pass = False

    print(f"{'All checks passed ✓' if all_pass else 'Some checks FAILED ✗'}")
    return all_pass


def sanity_check_dataloader(loader, f_dim: int, max_len: int):
    """Verify a DataLoader produces correctly shaped, NaN-free batches."""
    print("\n--- DataLoader sanity checks ---")
    x, mask, y = next(iter(loader))
    checks = [
        ("x shape == (B, MAX_LEN, F_DIM)",
         x.shape[1] == max_len and x.shape[2] == f_dim),
        ("mask shape == (B, MAX_LEN)",
         mask.shape[1] == max_len),
        ("y is int64",
         y.dtype == torch.int64),
        ("No NaN in x",
         not torch.isnan(x).any().item()),
        ("x values in [0, 1]",
         x.min().item() >= 0.0 and x.max().item() <= 1.0),
        ("Mask is bool",
         mask.dtype == torch.bool),
    ]
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
    print(f"  Batch: x={x.shape}, mask={mask.shape}, y={y.shape}")