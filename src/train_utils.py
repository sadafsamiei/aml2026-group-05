"""
train_utils.py
--------------
Training loop, evaluation, sanity checks, and results logging.
Supports both the audio-only Transformer (Model 2) and the attentive
pooling Transformer (Model 3) via a use_meta flag.

"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import CLASS_NAMES, LABEL_DECODER

torch.backends.cudnn.benchmark = True
_ALL_CLASS_IDS = list(range(len(CLASS_NAMES)))



def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_meta: bool = False, grad_clip: float = 1.0,
                    use_amp: bool = True) -> dict:
    
    model.train()
    total_loss = 0.0
    total_grad = 0.0
    n_batches  = 0

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype   = torch.bfloat16

    for batch in loader:
        x, mask, y = batch[0], batch[1], batch[2]
        meta = batch[3] if len(batch) == 4 else None

        if use_meta and meta is not None:
            x    = x   .to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y    = y   .to(device, non_blocking=True)
            meta = meta.to(device, non_blocking=True)
        else:
            x    = x   .to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y    = y   .to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=amp_enabled):
            if use_meta and meta is not None:
                logits, _ = model(x, padding_mask=mask, meta=meta)
            else:
                out = model(x, padding_mask=mask)
                logits = out[0] if isinstance(out, tuple) else out

            loss = criterion(logits, y)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_grad += grad_norm.item()
        n_batches  += 1

    return {
        "loss"     : total_loss / max(n_batches, 1),
        "grad_norm": total_grad / max(n_batches, 1),
    }


def evaluate(model, loader, device,
             use_meta: bool = False, return_preds: bool = False,
             use_amp: bool = True) -> dict:
    model.eval()
    all_preds   = []
    all_labels  = []
    all_weights = []   

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype   = torch.bfloat16

    with torch.no_grad():
        for batch in loader:
            x, mask, y = batch[0], batch[1], batch[2]
            meta = batch[3] if len(batch) == 4 else None

            x    = x   .to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if use_meta and meta is not None:
                meta = meta.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype,
                                    enabled=amp_enabled):
                if use_meta and meta is not None:
                    out = model(x, padding_mask=mask, meta=meta)
                else:
                    out = model(x, padding_mask=mask)


            if isinstance(out, tuple):
                logits, weights = out

                all_weights.append(weights.float().cpu())
            else:
                logits = out

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(y.tolist())

    macro_f1     = f1_score(all_labels, all_preds,
                            average="macro", zero_division=0,
                            labels=_ALL_CLASS_IDS)
    per_class_f1 = f1_score(all_labels, all_preds,
                            average=None, zero_division=0,
                            labels=_ALL_CLASS_IDS)

    result = {
        "macro_f1"    : macro_f1,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1.tolist())),
    }
    if return_preds:
        result["preds"]  = all_preds
        result["labels"] = all_labels
    if all_weights:
        result["attn_weights"] = torch.cat(all_weights, dim=0)

    return result



def train_model(model, train_loader, val_loader, criterion, device,
                lr: float = 1e-3, weight_decay: float = 1e-4,
                epochs: int = 100, patience: int = 15,
                save_path: Path = None, verbose: bool = True,
                use_meta: bool = False) -> dict:
  
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_f1    = -1.0
    patience_count = 0
    stopped_epoch  = epochs
    history = {"train_loss": [], "val_f1": [], "grad_norm": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_meta=use_meta
        )
        val_stats = evaluate(model, val_loader, device, use_meta=use_meta)
        val_f1    = val_stats["macro_f1"]

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_stats["loss"])
        history["val_f1"].append(val_f1)
        history["grad_norm"].append(train_stats["grad_norm"])
        history["lr"].append(current_lr)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or
                        patience_count >= patience):
            per_cls = " | ".join(
                f"{c}:{v:.2f}" for c, v in val_stats["per_class_f1"].items()
            )
            print(f"  Epoch {epoch:3d} | "
                  f"loss: {train_stats['loss']:.4f} | "
                  f"val F1: {val_f1:.4f} (best: {best_val_f1:.4f}) | "
                  f"patience: {patience_count}/{patience} | "
                  f"lr: {current_lr:.2e}")
            if epoch % 20 == 0:
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

    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    elif save_path and verbose:
        print(f"  ⚠ No checkpoint found at {save_path}; model is in last-epoch state.")

    return {
        "best_val_f1"  : best_val_f1,
        "stopped_epoch": stopped_epoch,
        "history"      : history,
        "elapsed_s"    : elapsed,
    }

def analyse_attention_weights(weights: torch.Tensor,
                               padding_mask: torch.Tensor = None) -> dict:
    if padding_mask is not None:
        real_mask = (~padding_mask).float()
        weights   = weights * real_mask

    std_per_example = weights.std(dim=1)  
    uniform_std     = (1.0 / weights.shape[1]) ** 0.5 * 0.1

    stats = {
        "mean_weight_std"  : float(std_per_example.mean()),
        "min_weight_std"   : float(std_per_example.min()),
        "max_weight"       : float(weights.max()),
        "uniform_std_proxy": float(uniform_std),
        "is_non_uniform"   : bool(std_per_example.mean() > uniform_std),
    }
    return stats


def print_classification_report(preds, labels, split_name: str = ""):
    present_ids   = sorted(set(labels) | set(preds))
    present_names = [LABEL_DECODER.get(i, str(i)) for i in present_ids]

    if len(present_ids) < len(CLASS_NAMES):
        missing = [LABEL_DECODER.get(i, str(i))
                   for i in range(len(CLASS_NAMES)) if i not in present_ids]
        print(f"  ℹ  Classes absent from this split "
              f"(no spectrograms yet): {missing}")

    title = f"Classification Report — {split_name}" if split_name else \
            "Classification Report"
    print(f"\n{'='*55}")
    print(title)
    print("=" * 55)
    print(classification_report(
        labels, preds,
        labels       = present_ids,
        target_names = present_names,
        digits       = 3,
        zero_division= 0,
    ))


def save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, torch.Tensor): return obj.tolist()
        return obj

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"Results saved → {path}")


def sanity_check_forward(model, f_dim: int, max_len: int, n_classes: int,
                         device: torch.device,
                         use_meta: bool = False, meta_dim: int = 2):
    print("\n--- Forward pass sanity checks ---")
    model.eval()
    B = 4
    dummy_x    = torch.randn(B, max_len, f_dim).to(device)
    dummy_mask = torch.zeros(B, max_len, dtype=torch.bool).to(device)
    dummy_mask[:, -20:] = True
    dummy_meta = torch.randn(B, meta_dim).to(device) if use_meta else None

    try:
        with torch.no_grad():
            out = model(dummy_x, padding_mask=dummy_mask,
                        **({'meta': dummy_meta} if use_meta else {}))

        is_tuple   = isinstance(out, tuple)
        logits     = out[0] if is_tuple else out
        attn_wts   = out[1] if is_tuple else None

        checks = [
            ("Output shape == (B, n_classes)",
             logits.shape == (B, n_classes)),
            ("No NaN in logits",
             not torch.isnan(logits).any().item()),
            ("No Inf in logits",
             not torch.isinf(logits).any().item()),
            ("logits dtype is float32",
             logits.dtype == torch.float32),
        ]

        if attn_wts is not None:
            checks += [
                ("Attn weights shape == (B, T)",
                 attn_wts.shape == (B, max_len)),
                ("Attn weights sum ≈ 1 per example",
                 attn_wts.sum(dim=1).allclose(
                     torch.ones(B, device=device), atol=1e-3)),
                ("Padded positions have near-zero weight",
                 attn_wts[:, -20:].max().item() < 1e-3),
                ("Weights are non-uniform",
                 attn_wts.std(dim=1).mean().item() > 1e-4),
            ]

        all_pass = True
        for name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {name}")
            if not passed:
                all_pass = False

        print(f"  Parameter count: {model.count_parameters():,}")

    except Exception as e:
        print(f"  ✗ Forward pass crashed: {e}")
        import traceback; traceback.print_exc()
        all_pass = False

    print(f"  {'All checks passed ✓' if all_pass else 'Some checks FAILED ✗'}")
    return all_pass


def sanity_check_dataloader(loader, f_dim: int, max_len: int,
                             use_meta: bool = False):
    print("\n--- DataLoader sanity checks ---")
    batch = next(iter(loader))

    if len(batch) == 4:
        x, mask, y, meta = batch
    else:
        x, mask, y = batch
        meta = None

    if use_meta and meta is not None:
        checks = [
            ("meta shape == (B, meta_dim)", meta.ndim == 2),
            ("meta dtype is float32",       meta.dtype == torch.float32),
        ]
    else:
        checks = []

    checks = [
        ("x shape == (B, MAX_LEN, F_DIM)",
         x.shape[1] == max_len and x.shape[2] == f_dim),
        ("mask shape == (B, MAX_LEN)", mask.shape[1] == max_len),
        ("y is int64",                 y.dtype == torch.int64),
        ("No NaN in x",                not torch.isnan(x).any().item()),
        ("x values are finite",
         torch.isfinite(x).all().item()),
        ("mask is bool",               mask.dtype == torch.bool),
    ] + checks

    for name, passed in checks:
        print(f"  {'✓ PASS' if passed else '✗ FAIL'}  {name}")

    shape_str = f"x={x.shape}, mask={mask.shape}, y={y.shape}"
    if use_meta and meta is not None:
        shape_str += f", meta={meta.shape}"
    print(f"  Batch: {shape_str}")