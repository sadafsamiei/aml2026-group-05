"""
scripts/attention_analysis.py
-----------------------------
Compare attention weight profiles for Model 3 between:
  - d=32 mel (grid-selected, 26K params) — canonical Apr 16 run, test F1 0.221±0.013
  - d=192 wav2vec2 (re-gridded, 648K params) — canonical Apr 21 run, test F1 0.246±0.016

Produces figures showing whether the attentive mechanism actually learns
to focus on specific frames, or whether it collapses to near-uniform
(equivalent to mean pooling). This directly addresses the README's Step 4
claim: "the attentive mechanism can identify diagnostic regions on
wav2vec2 but not on mel".

Sampling protocol
-----------------
For each target class we pick ONE representative validation clip (the
first val-set occurrence, deterministic). Both configs see the SAME
clip — we load the same audio_ID through the mel pipeline for d=32 and
through the wav2vec2 pipeline for d=192. Same child, same utterance,
different input representations.

Output
------
  RES_DIR/figures/attn_analysis_compare_<timestamp>.png
      4-panel figure (one panel per class) showing 5-seed overlayed
      attention weights for both configs side-by-side.

  RES_DIR/figures/attn_analysis_summary_<timestamp>.png
      2-panel figure: distribution of per-seed mean-weight-std across
      the val set. The non-uniformity metric from README's design
      decisions section. Visually confirms whether attention is
      non-uniform in practice.

  RES_DIR/results/attn_analysis_<timestamp>.json
      Per-class, per-config, per-seed mean-weight-std values. Raw data
      for the figures.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from config import (
    CLASS_NAMES, LABEL_ENCODER, N_CLASSES,
    RESULTS_DIR, FIGURES_DIR, PRETRAINED_DIR, SPEC_DIR,
)
from data_utils import (
    build_master_table, determine_eval_protocol, load_splits,
    MetadataSpectrogramDataset, check_metadata_viability, compute_meta_stats,
    load_png,
)
from models import AttentivePoolingTransformer


# ---------------------------------------------------------------------------
# Canonical checkpoints
# ---------------------------------------------------------------------------

# d=32 mel (canonical from attn_multiseed_20260416_230142.json, test F1 0.221±0.013)
D32_MEL_CKPTS = [
    (42,   "/home/renku/work/data/models/transformer_attn_attn_seed42_fold0_20260416_224008.pt"),
    (123,  "/home/renku/work/data/models/transformer_attn_attn_seed123_fold0_20260416_224431.pt"),
    (456,  "/home/renku/work/data/models/transformer_attn_attn_seed456_fold0_20260416_224847.pt"),
    (789,  "/home/renku/work/data/models/transformer_attn_attn_seed789_fold0_20260416_225300.pt"),
    (2024, "/home/renku/work/data/models/transformer_attn_attn_seed2024_fold0_20260416_225713.pt"),
]
D32_HP = {"d_model": 32, "nhead": 4, "num_layers": 1, "dropout": 0.3}
D32_F_DIM   = 128
D32_MAX_LEN = 230

# d=192 wav2vec2 (canonical from attn_multiseed_20260421_000648.json, test F1 0.246±0.016)
D192_W2V_CKPTS = [
    (42,   "/home/renku/work/data/models/transformer_attn_attn_seed42_fold0_20260420_234541.pt"),
    (123,  "/home/renku/work/data/models/transformer_attn_attn_seed123_fold0_20260420_234958.pt"),
    (456,  "/home/renku/work/data/models/transformer_attn_attn_seed456_fold0_20260420_235421.pt"),
    (789,  "/home/renku/work/data/models/transformer_attn_attn_seed789_fold0_20260420_235821.pt"),
    (2024, "/home/renku/work/data/models/transformer_attn_attn_seed2024_fold0_20260421_000237.pt"),
]
D192_HP = {"d_model": 192, "nhead": 4, "num_layers": 1, "dropout": 0.3}
D192_F_DIM   = 768
D192_MAX_LEN = 230


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(f_dim, max_len, hp, use_meta, device):
    return AttentivePoolingTransformer(
        input_dim=f_dim, d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], dropout=hp["dropout"],
        num_classes=N_CLASSES, max_len=max_len + 50,
        use_external_meta=use_meta, meta_dim=2,
    ).to(device)


def real_frame_count(audio_id, f_dim, max_len):
    """Return actual (non-padded) number of frames for this clip."""
    if f_dim == 128:
        # Mel path — derive from the PNG
        png_path = SPEC_DIR / f"{audio_id}.png"
        spec = load_png(str(png_path), target_height=f_dim)
        return min(spec.shape[1], max_len)
    else:
        # wav2vec2 path — derive from the npy
        emb_path = PRETRAINED_DIR / f"{audio_id}.npy"
        emb = np.load(emb_path)  # (768, T)
        # Real frames = T up to the first all-zero column (if padded) or full T
        # Most wav2vec2 embeddings are unpadded but cap at max_len
        return min(emb.shape[1], max_len)


def attention_weights_for(model, x, mask, meta, use_meta, device):
    """Return 1-D numpy array of attention weights (T,) for a single clip."""
    model.eval()
    x_b    = x.unsqueeze(0).to(device)
    mask_b = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        if use_meta:
            meta_b = meta.unsqueeze(0).to(device)
            _, w = model(x_b, padding_mask=mask_b, meta=meta_b)
        else:
            _, w = model(x_b, padding_mask=mask_b)
    return w.squeeze(0).cpu().numpy()


def compute_weight_std(model, loader, use_meta, device):
    """
    Mean std of attention weights across all val clips.
    High std → attention is non-uniform (peaked on specific frames).
    Low std → near-uniform (equivalent to mean pooling).

    Compare against the "uniform std proxy": if attention were perfectly
    uniform at w=1/T for each clip, std=0. But for a length distribution
    with max_len T=230, a uniform-like baseline has tiny std ≈ sqrt(1/T) / T
    which your training code already prints as ~0.00659 for the d=32 mel
    runs. Values above that indicate learned non-uniformity.
    """
    stds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if use_meta:
                x, mask, _, meta = batch
                x = x.to(device); mask = mask.to(device); meta = meta.to(device)
                _, w = model(x, padding_mask=mask, meta=meta)
            else:
                x, mask, _ = batch
                x = x.to(device); mask = mask.to(device)
                _, w = model(x, padding_mask=mask)
            # Per-clip std along time dim, then average
            # Only include non-padded positions
            pad = mask  # True = padded
            w_np = w.cpu().numpy()
            for i in range(w_np.shape[0]):
                real = (~pad[i]).sum().item()
                stds.append(float(np.std(w_np[i, :real])))
    return stds


def pick_representative_clips(val_df, n_per_class=1):
    """For each class, pick the first val example deterministically."""
    picked = {}
    for cname in CLASS_NAMES:
        cls_df = val_df[val_df["class4"] == cname]
        if len(cls_df) == 0:
            picked[cname] = None
            continue
        picked[cname] = cls_df.iloc[0].to_dict()
    return picked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    cfg.print_config()

    # We need the val split to run analyses on. Both configs use the same
    # participant-disjoint split.
    df = build_master_table(verbose=False)
    df = df[df["png_exists"]].copy()
    protocol = determine_eval_protocol(df)
    folds = load_splits(protocol)
    train_df, val_df, test_df = folds[0]
    meta_check = check_metadata_viability(df, verbose=False)
    use_meta = meta_check["USE_EXTERNAL_META"]
    meta_stats = compute_meta_stats(train_df)

    print(f"\nVal set: {len(val_df)} clips")

    # Pick one representative clip per class
    reps = pick_representative_clips(val_df)
    print("\nRepresentative clips:")
    for cname, row in reps.items():
        if row is None:
            print(f"  {cname}: NO examples in val")
        else:
            print(f"  {cname}: audio_ID={row['audio_ID']}, child={row['child']}")

    # Two datasets — one mel (d=32 config), one wav2vec2 (d=192 config)
    # Both loaded from val_df containing the same rows.
    # The USE_PRETRAINED flag inside the Dataset determines which file gets loaded.
    # Since we can't flip config mid-run without reimporting, we handle this by
    # temporarily overriding cfg at load time for each config.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "d32_mel":     {"per_class_attn": {}, "weight_stds": {}},
        "d192_w2v":    {"per_class_attn": {}, "weight_stds": {}},
    }

    # ── d=32 mel ──────────────────────────────────────────────────────
    print(f"\n{'='*62}\nConfig 1: d=32, mel spectrogram (F_DIM=128)\n{'='*62}")

    # Temporarily patch config for the mel path
    orig_USE_PRETRAINED = cfg.USE_PRETRAINED
    orig_USE_MFCC       = cfg.USE_MFCC
    orig_F_DIM          = cfg.F_DIM
    orig_USE_SPECAUG    = cfg.USE_SPECAUGMENT
    cfg.USE_PRETRAINED  = False
    cfg.USE_MFCC        = False
    cfg.F_DIM           = 128
    cfg.USE_SPECAUGMENT = False  # no augmentation at inference
    # Also patch the module-level globals in data_utils which were imported
    # from config at import time
    import data_utils as du
    du.USE_PRETRAINED = False
    du.USE_MFCC = False
    du.USE_SPECAUGMENT = False

    val_ds_mel = MetadataSpectrogramDataset(
        val_df, LABEL_ENCODER, D32_MAX_LEN, D32_F_DIM,
        meta_stats=meta_stats, use_external_meta=use_meta, train=False)
    val_loader_mel = torch.utils.data.DataLoader(val_ds_mel, batch_size=64,
                                                   shuffle=False)

    # Build index from audio_ID to dataset position for fast lookup
    audio_id_to_idx_mel = {r["audio_ID"]: i for i, r in val_ds_mel.records.iterrows()}

    for seed, ckpt_path in D32_MEL_CKPTS:
        model = build_model(D32_F_DIM, D32_MAX_LEN, D32_HP, use_meta, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                          weights_only=False))

        # Per-class attention profile on the representative clip
        for cname, row in reps.items():
            if row is None: continue
            idx = audio_id_to_idx_mel[row["audio_ID"]]
            x, mask, _, meta = val_ds_mel[idx]
            w = attention_weights_for(model, x, mask, meta, use_meta, device)
            real = real_frame_count(row["audio_ID"], D32_F_DIM, D32_MAX_LEN)
            results["d32_mel"]["per_class_attn"].setdefault(cname, {})[seed] = {
                "weights": w[:real].tolist(),
                "T_real": int(real),
            }

        # Aggregate weight std across val set
        stds = compute_weight_std(model, val_loader_mel, use_meta, device)
        results["d32_mel"]["weight_stds"][seed] = stds
        print(f"  seed {seed:>5}: mean weight std over val = "
              f"{np.mean(stds):.5f} (median={np.median(stds):.5f})")

    # ── d=192 wav2vec2 ────────────────────────────────────────────────
    print(f"\n{'='*62}\nConfig 2: d=192, wav2vec2 (F_DIM=768)\n{'='*62}")

    cfg.USE_PRETRAINED  = True
    cfg.USE_MFCC        = False
    cfg.F_DIM           = 768
    cfg.USE_SPECAUGMENT = False
    du.USE_PRETRAINED = True
    du.USE_MFCC = False
    du.USE_SPECAUGMENT = False

    val_ds_w2v = MetadataSpectrogramDataset(
        val_df, LABEL_ENCODER, D192_MAX_LEN, D192_F_DIM,
        meta_stats=meta_stats, use_external_meta=use_meta, train=False)
    val_loader_w2v = torch.utils.data.DataLoader(val_ds_w2v, batch_size=64,
                                                   shuffle=False)

    audio_id_to_idx_w2v = {r["audio_ID"]: i for i, r in val_ds_w2v.records.iterrows()}

    for seed, ckpt_path in D192_W2V_CKPTS:
        model = build_model(D192_F_DIM, D192_MAX_LEN, D192_HP, use_meta, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                          weights_only=False))

        for cname, row in reps.items():
            if row is None: continue
            idx = audio_id_to_idx_w2v[row["audio_ID"]]
            x, mask, _, meta = val_ds_w2v[idx]
            w = attention_weights_for(model, x, mask, meta, use_meta, device)
            real = real_frame_count(row["audio_ID"], D192_F_DIM, D192_MAX_LEN)
            results["d192_w2v"]["per_class_attn"].setdefault(cname, {})[seed] = {
                "weights": w[:real].tolist(),
                "T_real": int(real),
            }

        stds = compute_weight_std(model, val_loader_w2v, use_meta, device)
        results["d192_w2v"]["weight_stds"][seed] = stds
        print(f"  seed {seed:>5}: mean weight std over val = "
              f"{np.mean(stds):.5f} (median={np.median(stds):.5f})")

    # Restore original config state
    cfg.USE_PRETRAINED  = orig_USE_PRETRAINED
    cfg.USE_MFCC        = orig_USE_MFCC
    cfg.F_DIM           = orig_F_DIM
    cfg.USE_SPECAUGMENT = orig_USE_SPECAUG
    du.USE_PRETRAINED = orig_USE_PRETRAINED
    du.USE_MFCC = orig_USE_MFCC
    du.USE_SPECAUGMENT = orig_USE_SPECAUG

    # ── Save raw JSON ─────────────────────────────────────────────────
    # Convert seed keys (int) to strings for JSON compatibility
    j_results = {
        "d32_mel": {
            "per_class_attn": {c: {str(s): v for s, v in d.items()}
                                 for c, d in results["d32_mel"]["per_class_attn"].items()},
            "weight_stds":    {str(s): v for s, v in results["d32_mel"]["weight_stds"].items()},
        },
        "d192_w2v": {
            "per_class_attn": {c: {str(s): v for s, v in d.items()}
                                 for c, d in results["d192_w2v"]["per_class_attn"].items()},
            "weight_stds":    {str(s): v for s, v in results["d192_w2v"]["weight_stds"].items()},
        },
        "representative_clips": {c: {"audio_ID": r["audio_ID"], "child": r["child"]}
                                   for c, r in reps.items() if r is not None},
    }
    out_json = RESULTS_DIR / f"attn_analysis_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(j_results, f, indent=2)
    print(f"\nRaw data → {out_json}")

    # ── Figure 1: per-class attention comparison ──────────────────────
    print("\nBuilding figures...")
    n_classes = sum(1 for c in CLASS_NAMES if reps.get(c) is not None)
    fig, axes = plt.subplots(n_classes, 2, figsize=(13, 2.8 * n_classes),
                              squeeze=False)
    cmap = plt.get_cmap("tab10")

    row = 0
    for cname in CLASS_NAMES:
        if reps.get(cname) is None: continue
        audio_id = reps[cname]["audio_ID"]
        child    = reps[cname]["child"]

        for col, (cfg_key, config_label, hp_str) in enumerate([
            ("d32_mel",  "d=32, mel (Model 3 grid-selected)",  "26K params"),
            ("d192_w2v", "d=192, wav2vec2 (Model 3 regridded)", "648K params"),
        ]):
            ax = axes[row, col]
            class_attn = results[cfg_key]["per_class_attn"].get(cname, {})
            if not class_attn:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            T_real = next(iter(class_attn.values()))["T_real"]
            for i, (seed, entry) in enumerate(sorted(class_attn.items())):
                w = np.array(entry["weights"])
                ax.plot(range(len(w)), w, color=cmap(i), lw=1.3, alpha=0.85,
                        label=f"seed {seed}")

            uniform = 1.0 / T_real
            ax.axhline(uniform, ls="--", color="grey", lw=1.0,
                       label=f"uniform ({uniform:.4f})")
            ax.set_xlim(0, T_real)
            ax.set_xlabel("Time frame")
            if col == 0: ax.set_ylabel("Attention weight")
            ax.set_title(f"{cname} ({audio_id}, {child}) — {config_label}",
                          fontsize=9.5)
            ax.grid(alpha=0.3)
            if row == 0: ax.legend(fontsize=7, loc="upper right", ncol=2)

        row += 1

    fig.suptitle("Model 3 attention weight profiles — d=32 mel vs d=192 wav2vec2"
                 "\n(same clip per class, 5 seeds overlaid)", y=1.005, fontsize=11)
    plt.tight_layout()
    out_fig1 = FIGURES_DIR / f"attn_analysis_compare_{timestamp}.png"
    plt.savefig(out_fig1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure → {out_fig1}")

    # ── Figure 2: weight-std distribution summary ─────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (cfg_key, label, color) in zip(
        axes2,
        [("d32_mel",  "d=32, mel",      "steelblue"),
         ("d192_w2v", "d=192, wav2vec2", "darkorange")],
    ):
        # Aggregate across all seeds
        all_stds = []
        for seed, stds in results[cfg_key]["weight_stds"].items():
            all_stds.extend(stds)
        ax.hist(all_stds, bins=30, color=color, alpha=0.75, edgecolor="black")
        # Uniform proxy — same across both configs for T=230
        uniform_std = np.sqrt(1/230) / 230  # tiny
        ax.axvline(np.mean(all_stds), ls="--", color="black", lw=1.2,
                   label=f"mean = {np.mean(all_stds):.5f}")
        ax.set_xlabel("Per-clip attention weight std")
        ax.set_ylabel("Number of val clips × seeds")
        ax.set_title(f"{label}\n(5 seeds × {len(val_df)} val clips)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig2.suptitle("Attention non-uniformity across the val set — "
                  "higher std = more peaked attention", y=1.02, fontsize=11)
    plt.tight_layout()
    out_fig2 = FIGURES_DIR / f"attn_analysis_summary_{timestamp}.png"
    plt.savefig(out_fig2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Summary figure    → {out_fig2}")

    # ── Printed summary ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("ATTENTION ANALYSIS SUMMARY")
    print(f"{'='*62}")
    for cfg_key, label in [("d32_mel", "d=32 mel"),
                             ("d192_w2v", "d=192 wav2vec2")]:
        all_stds = []
        for seed, stds in results[cfg_key]["weight_stds"].items():
            all_stds.extend(stds)
        print(f"  {label:<20s}  mean std = {np.mean(all_stds):.5f}  "
              f"median = {np.median(all_stds):.5f}  "
              f"max = {np.max(all_stds):.5f}")

    mel_mean = np.mean([s for ss in results["d32_mel"]["weight_stds"].values() for s in ss])
    w2v_mean = np.mean([s for ss in results["d192_w2v"]["weight_stds"].values() for s in ss])
    ratio = w2v_mean / mel_mean if mel_mean > 0 else float("inf")
    print(f"\n  wav2vec2 / mel std ratio: {ratio:.2f}x")
    if ratio > 1.3:
        print(f"  → wav2vec2 attention is MORE peaked than mel by {ratio:.1f}x — "
              f"consistent with README's Step 4 claim.")
    elif ratio < 0.8:
        print(f"  → wav2vec2 attention is LESS peaked than mel — "
              f"the README's Step 4 claim needs revision.")
    else:
        print(f"  → attention non-uniformity is comparable between configs — "
              f"Model 3's wav2vec2 advantage comes from elsewhere.")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()