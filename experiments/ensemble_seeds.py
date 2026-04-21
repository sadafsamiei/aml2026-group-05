"""
scripts/ensemble_seeds.py
-------------------------
Ensemble the 5 per-seed checkpoints for Models 2 and 3 on wav2vec2.

For each model:
  1. Load the 5 per-seed checkpoints.
  2. Run each on the held-out test set, record softmax probabilities.
  3. Average probabilities across seeds, argmax → ensemble prediction.
  4. Compute macro-F1, per-class F1, confusion matrix.
  5. Also record the per-seed individual test F1 for comparison.

Canonical checkpoints (hard-coded because we want to be exact about WHICH
checkpoints are the "canonical" ones per the current README). If any of
these paths go stale, update them here — do not take a more recent file
from the same seed by accident.

Assumes USE_PRETRAINED = True in config.py (wav2vec2 path). If it isn't
the test dataset will load mel spectrograms instead and everything
breaks silently. Asserts this at the top.

Outputs
-------
  RES_DIR/results/ensemble_wav2vec2_<timestamp>.json
      Aggregated ensemble metrics for both models.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import config as cfg
from config import (
    CLASS_NAMES, LABEL_ENCODER, LABEL_DECODER, N_CLASSES,
    RESULTS_DIR, RES_DIR,
)
from data_utils import (
    build_master_table, determine_eval_protocol, load_splits,
    SpectrogramDataset, MetadataSpectrogramDataset,
    check_metadata_viability, compute_meta_stats,
)
from models import AudioTransformer, AttentivePoolingTransformer


# ---------------------------------------------------------------------------
# Canonical checkpoints — update here if the canonical runs change
# ---------------------------------------------------------------------------

MODEL2_CKPTS = [
    # (seed, path) — all d=64, L=2, drop=0.5, from 2026-04-21 multiseed run
    (42,   "/home/renku/work/data/models/transformer_audio_audio_seed42_fold0_20260421_121308.pt"),
    (123,  "/home/renku/work/data/models/transformer_audio_audio_seed123_fold0_20260421_122138.pt"),
    (456,  "/home/renku/work/data/models/transformer_audio_audio_seed456_fold0_20260421_122957.pt"),
    (789,  "/home/renku/work/data/models/transformer_audio_audio_seed789_fold0_20260421_124116.pt"),
    (2024, "/home/renku/work/data/models/transformer_audio_audio_seed2024_fold0_20260421_124946.pt"),
]

MODEL2_HPARAMS = {"d_model": 64, "nhead": 4, "num_layers": 2, "dropout": 0.5}

MODEL3_CKPTS = [
    # (seed, path) — all d=192, L=1, drop=0.3, from 2026-04-20/21 multiseed run
    (42,   "/home/renku/work/data/models/transformer_attn_attn_seed42_fold0_20260420_234541.pt"),
    (123,  "/home/renku/work/data/models/transformer_attn_attn_seed123_fold0_20260420_234958.pt"),
    (456,  "/home/renku/work/data/models/transformer_attn_attn_seed456_fold0_20260420_235421.pt"),
    (789,  "/home/renku/work/data/models/transformer_attn_attn_seed789_fold0_20260420_235821.pt"),
    (2024, "/home/renku/work/data/models/transformer_attn_attn_seed2024_fold0_20260421_000237.pt"),
]

MODEL3_HPARAMS = {"d_model": 192, "nhead": 4, "num_layers": 1, "dropout": 0.3}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_test_probs(model, loader, device, use_meta: bool) -> tuple:
    """Run model on loader, return (probs, labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if use_meta:
                x, mask, y, meta = batch
                x = x.to(device); mask = mask.to(device); meta = meta.to(device)
                logits, _ = model(x, padding_mask=mask, meta=meta)
            else:
                x, mask, y = batch
                x = x.to(device); mask = mask.to(device)
                logits = model(x, padding_mask=mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def metrics_from_preds(preds, labels) -> dict:
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    per_class = {}
    report = classification_report(labels, preds, output_dict=True,
                                    zero_division=0, labels=list(range(N_CLASSES)))
    for i, cname in enumerate(CLASS_NAMES):
        per_class[cname] = float(report[str(i)]["f1-score"])
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)),
                           normalize="true").tolist()
    return {
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class,
        "confusion_matrix_rownorm": cm,
    }


# ---------------------------------------------------------------------------
# Model constructors (factored out so each seed rebuilds fresh)
# ---------------------------------------------------------------------------

def build_model2(f_dim, max_len, device):
    return AudioTransformer(
        input_dim=f_dim, d_model=MODEL2_HPARAMS["d_model"],
        nhead=MODEL2_HPARAMS["nhead"], num_layers=MODEL2_HPARAMS["num_layers"],
        dropout=MODEL2_HPARAMS["dropout"], num_classes=N_CLASSES,
        max_len=max_len + 50,
    ).to(device)


def build_model3(f_dim, max_len, device, use_meta: bool):
    return AttentivePoolingTransformer(
        input_dim=f_dim, d_model=MODEL3_HPARAMS["d_model"],
        nhead=MODEL3_HPARAMS["nhead"], num_layers=MODEL3_HPARAMS["num_layers"],
        dropout=MODEL3_HPARAMS["dropout"], num_classes=N_CLASSES,
        max_len=max_len + 50,
        use_external_meta=use_meta, meta_dim=2,
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert cfg.USE_PRETRAINED, \
        "USE_PRETRAINED must be True in config.py for the wav2vec2 ensemble"

    cfg.print_config()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    # Data & splits
    print("\n[1/4] Loading test split...")
    df = build_master_table(verbose=False)
    df = df[df["png_exists"]].copy()
    protocol = determine_eval_protocol(df)
    folds = load_splits(protocol)
    train_df, val_df, test_df = folds[0]

    f_dim   = cfg.F_DIM
    max_len = cfg.MAX_LEN
    print(f"  Test n = {len(test_df)}, f_dim = {f_dim}, max_len = {max_len}")

    # Metadata stats (needed for Model 3)
    meta_check = check_metadata_viability(df, verbose=False)
    use_meta   = meta_check["USE_EXTERNAL_META"]
    meta_stats = compute_meta_stats(train_df)

    # Build test loaders — one without meta (Model 2), one with meta (Model 3)
    test_ds_audio = SpectrogramDataset(test_df, LABEL_ENCODER, max_len, f_dim,
                                        train=False)
    test_ds_attn  = MetadataSpectrogramDataset(
        test_df, LABEL_ENCODER, max_len, f_dim,
        meta_stats=meta_stats, use_external_meta=use_meta, train=False)
    test_loader_audio = DataLoader(test_ds_audio, batch_size=128, shuffle=False)
    test_loader_attn  = DataLoader(test_ds_attn,  batch_size=128, shuffle=False)

    results = {}

    # ── Model 2 ───────────────────────────────────────────────────────
    print("\n[2/4] Running Model 2 (audio, mean pool) ensemble...")
    probs_by_seed_m2 = []
    per_seed_m2 = []
    labels_m2 = None
    for seed, ckpt_path in MODEL2_CKPTS:
        model = build_model2(f_dim, max_len, device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd)
        probs, labels = get_test_probs(model, test_loader_audio, device,
                                        use_meta=False)
        if labels_m2 is None: labels_m2 = labels
        else: assert np.array_equal(labels, labels_m2), "label order differs across seeds"
        probs_by_seed_m2.append(probs)
        individual_preds = probs.argmax(axis=-1)
        individual_m = metrics_from_preds(individual_preds, labels)
        per_seed_m2.append({"seed": seed, "test_f1": individual_m["macro_f1"]})
        print(f"  seed {seed:>5}: individual test F1 = {individual_m['macro_f1']:.4f}")

    ensemble_probs_m2 = np.mean(np.stack(probs_by_seed_m2, axis=0), axis=0)
    ensemble_preds_m2 = ensemble_probs_m2.argmax(axis=-1)
    m2_metrics = metrics_from_preds(ensemble_preds_m2, labels_m2)

    print(f"\n  Model 2 individual mean: "
          f"{np.mean([s['test_f1'] for s in per_seed_m2]):.4f} "
          f"± {np.std([s['test_f1'] for s in per_seed_m2]):.4f}")
    print(f"  Model 2 ensemble test F1: {m2_metrics['macro_f1']:.4f}")
    print(f"  Model 2 ensemble per-class: {m2_metrics['per_class_f1']}")

    results["model2"] = {
        "hparams": MODEL2_HPARAMS,
        "checkpoints": [str(p) for _, p in MODEL2_CKPTS],
        "per_seed_test_f1": per_seed_m2,
        "individual_mean_f1": float(np.mean([s["test_f1"] for s in per_seed_m2])),
        "individual_std_f1":  float(np.std ([s["test_f1"] for s in per_seed_m2])),
        "ensemble_test_f1":   m2_metrics["macro_f1"],
        "ensemble_per_class": m2_metrics["per_class_f1"],
        "ensemble_cm_rownorm": m2_metrics["confusion_matrix_rownorm"],
    }

    # ── Model 3 ───────────────────────────────────────────────────────
    print("\n[3/4] Running Model 3 (attn pool, d=192) ensemble...")
    probs_by_seed_m3 = []
    per_seed_m3 = []
    labels_m3 = None
    for seed, ckpt_path in MODEL3_CKPTS:
        model = build_model3(f_dim, max_len, device, use_meta=use_meta)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd)
        probs, labels = get_test_probs(model, test_loader_attn, device,
                                        use_meta=use_meta)
        if labels_m3 is None: labels_m3 = labels
        else: assert np.array_equal(labels, labels_m3), "label order differs across seeds"
        probs_by_seed_m3.append(probs)
        individual_preds = probs.argmax(axis=-1)
        individual_m = metrics_from_preds(individual_preds, labels)
        per_seed_m3.append({"seed": seed, "test_f1": individual_m["macro_f1"]})
        print(f"  seed {seed:>5}: individual test F1 = {individual_m['macro_f1']:.4f}")

    ensemble_probs_m3 = np.mean(np.stack(probs_by_seed_m3, axis=0), axis=0)
    ensemble_preds_m3 = ensemble_probs_m3.argmax(axis=-1)
    m3_metrics = metrics_from_preds(ensemble_preds_m3, labels_m3)

    print(f"\n  Model 3 individual mean: "
          f"{np.mean([s['test_f1'] for s in per_seed_m3]):.4f} "
          f"± {np.std([s['test_f1'] for s in per_seed_m3]):.4f}")
    print(f"  Model 3 ensemble test F1: {m3_metrics['macro_f1']:.4f}")
    print(f"  Model 3 ensemble per-class: {m3_metrics['per_class_f1']}")

    results["model3"] = {
        "hparams": MODEL3_HPARAMS,
        "checkpoints": [str(p) for _, p in MODEL3_CKPTS],
        "per_seed_test_f1": per_seed_m3,
        "individual_mean_f1": float(np.mean([s["test_f1"] for s in per_seed_m3])),
        "individual_std_f1":  float(np.std ([s["test_f1"] for s in per_seed_m3])),
        "ensemble_test_f1":   m3_metrics["macro_f1"],
        "ensemble_per_class": m3_metrics["per_class_f1"],
        "ensemble_cm_rownorm": m3_metrics["confusion_matrix_rownorm"],
    }

    # ── Save & print summary ───────────────────────────────────────────
    print("\n[4/4] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ensemble_wav2vec2_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")

    print(f"\n{'='*62}")
    print("ENSEMBLE SUMMARY — wav2vec2, 5 seeds")
    print(f"{'='*62}")
    m2 = results["model2"]
    m3 = results["model3"]
    print(f"  {'Model 2 (d=64, L=2)':<30s}  individual: {m2['individual_mean_f1']:.4f} ± {m2['individual_std_f1']:.4f}  ensemble: {m2['ensemble_test_f1']:.4f}  (Δ = {m2['ensemble_test_f1'] - m2['individual_mean_f1']:+.4f})")
    print(f"  {'Model 3 (d=192, L=1)':<30s}  individual: {m3['individual_mean_f1']:.4f} ± {m3['individual_std_f1']:.4f}  ensemble: {m3['ensemble_test_f1']:.4f}  (Δ = {m3['ensemble_test_f1'] - m3['individual_mean_f1']:+.4f})")
    print(f"\n  Ensemble gap (Model 3 − Model 2): {m3['ensemble_test_f1'] - m2['ensemble_test_f1']:+.4f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()