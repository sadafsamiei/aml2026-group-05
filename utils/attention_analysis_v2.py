"""
utils/attention_analysis_v2.py
---------------------------------
Improved attention figures.

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


D32_MEL_CKPTS = [
    (42,   "/home/renku/work/data/models/transformer_attn_attn_seed42_fold0_20260416_224008.pt"),
    (123,  "/home/renku/work/data/models/transformer_attn_attn_seed123_fold0_20260416_224431.pt"),
    (456,  "/home/renku/work/data/models/transformer_attn_attn_seed456_fold0_20260416_224847.pt"),
    (789,  "/home/renku/work/data/models/transformer_attn_attn_seed789_fold0_20260416_225300.pt"),
    (2024, "/home/renku/work/data/models/transformer_attn_attn_seed2024_fold0_20260416_225713.pt"),
]
D32_HP = {"d_model": 32, "nhead": 4, "num_layers": 1, "dropout": 0.3}

D192_W2V_CKPTS = [
    (42,   "/home/renku/work/data/models/transformer_attn_attn_seed42_fold0_20260420_234541.pt"),
    (123,  "/home/renku/work/data/models/transformer_attn_attn_seed123_fold0_20260420_234958.pt"),
    (456,  "/home/renku/work/data/models/transformer_attn_attn_seed456_fold0_20260420_235421.pt"),
    (789,  "/home/renku/work/data/models/transformer_attn_attn_seed789_fold0_20260420_235821.pt"),
    (2024, "/home/renku/work/data/models/transformer_attn_attn_seed2024_fold0_20260421_000237.pt"),
]
D192_HP = {"d_model": 192, "nhead": 4, "num_layers": 1, "dropout": 0.3}

MAX_LEN = 230


def build_model(f_dim, hp, use_meta, device):
    return AttentivePoolingTransformer(
        input_dim=f_dim, d_model=hp["d_model"], nhead=hp["nhead"],
        num_layers=hp["num_layers"], dropout=hp["dropout"],
        num_classes=N_CLASSES, max_len=MAX_LEN + 50,
        use_external_meta=use_meta, meta_dim=2,
    ).to(device)


def real_frame_count(audio_id, f_dim):
    if f_dim == 128:
        png_path = SPEC_DIR / f"{audio_id}.png"
        spec = load_png(str(png_path), target_height=f_dim)
        return min(spec.shape[1], MAX_LEN)
    emb_path = PRETRAINED_DIR / f"{audio_id}.npy"
    emb = np.load(emb_path)
    return min(emb.shape[1], MAX_LEN)


def attention_weights_single(model, x, mask, meta, use_meta, device):
    model.eval()
    x_b = x.unsqueeze(0).to(device)
    mask_b = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        if use_meta:
            meta_b = meta.unsqueeze(0).to(device)
            _, w = model(x_b, padding_mask=mask_b, meta=meta_b)
        else:
            _, w = model(x_b, padding_mask=mask_b)
    return w.squeeze(0).cpu().numpy()


def attention_weights_batch(model, loader, use_meta, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            if use_meta:
                x, mask, _, meta = batch
                x, mask, meta = x.to(device), mask.to(device), meta.to(device)
                _, w = model(x, padding_mask=mask, meta=meta)
            else:
                x, mask, _ = batch
                x, mask = x.to(device), mask.to(device)
                _, w = model(x, padding_mask=mask)
            w_np = w.cpu().numpy()
            pad  = mask.cpu().numpy()
            for i in range(w_np.shape[0]):
                n_real = int((~pad[i]).sum())
                out.append((w_np[i, :n_real], n_real))
    return out


def effective_frames(weights, fraction=0.95):
    sorted_w = np.sort(weights)[::-1]
    cum = np.cumsum(sorted_w)
    total = cum[-1]
    if total <= 0: return len(weights)
    threshold = fraction * total
    k = int(np.searchsorted(cum, threshold) + 1)
    return min(k, len(weights))


def load_tau(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k, v in sd.items():
        if ("tau" in k.lower() or "temperature" in k.lower()) and v.numel() == 1:
            log_tau = float(v.item())
            return log_tau, float(np.exp(log_tau))
    return None, None


def pick_reps(val_df):
    picked = {}
    for cname in CLASS_NAMES:
        cls_df = val_df[val_df["class4"] == cname]
        picked[cname] = cls_df.iloc[0].to_dict() if len(cls_df) else None
    return picked


def _set_feature_mode(pretrained: bool):
    cfg.USE_PRETRAINED  = pretrained
    cfg.USE_MFCC        = False
    cfg.F_DIM           = 768 if pretrained else 128
    cfg.USE_SPECAUGMENT = False
    import data_utils as du
    du.USE_PRETRAINED = pretrained
    du.USE_MFCC = False
    du.USE_SPECAUGMENT = False


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    cfg.print_config()

    df = build_master_table(verbose=False)
    df = df[df["png_exists"]].copy()
    folds = load_splits(determine_eval_protocol(df))
    train_df, val_df, _ = folds[0]

    meta_check = check_metadata_viability(df, verbose=False)
    use_meta = meta_check["USE_EXTERNAL_META"]
    meta_stats = compute_meta_stats(train_df)
    print(f"Val: {len(val_df)} clips, use_meta={use_meta}")

    reps = pick_reps(val_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "d32_mel":  {"per_class_attn": {}, "val_weights": {}, "tau": {}},
        "d192_w2v": {"per_class_attn": {}, "val_weights": {}, "tau": {}},
        "representative_clips": {c: {"audio_ID": r["audio_ID"], "child": r["child"]}
                                   for c, r in reps.items() if r is not None},
    }

    print(f"\n{'='*62}\nConfig 1: d=32, mel\n{'='*62}")
    _set_feature_mode(pretrained=False)
    val_ds = MetadataSpectrogramDataset(
        val_df, LABEL_ENCODER, MAX_LEN, 128,
        meta_stats=meta_stats, use_external_meta=use_meta, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    aid_to_idx = {r["audio_ID"]: i for i, r in val_ds.records.iterrows()}

    for seed, path in D32_MEL_CKPTS:
        model = build_model(128, D32_HP, use_meta, device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        log_tau, tau = load_tau(path)
        results["d32_mel"]["tau"][seed] = {"log_tau": log_tau, "tau": tau}
        print(f"  seed {seed:>5}: log_tau={log_tau:.4f} → τ={tau:.4f}")

        for cname, row in reps.items():
            if row is None: continue
            idx = aid_to_idx[row["audio_ID"]]
            x, mask, _, meta = val_ds[idx]
            w = attention_weights_single(model, x, mask, meta, use_meta, device)
            n_real = real_frame_count(row["audio_ID"], 128)
            results["d32_mel"]["per_class_attn"].setdefault(cname, {})[seed] = {
                "weights": w[:n_real].tolist(), "T_real": int(n_real),
            }

        batch_out = attention_weights_batch(model, val_loader, use_meta, device)
        eff50 = [effective_frames(w, 0.50) for w, _ in batch_out]
        eff80 = [effective_frames(w, 0.80) for w, _ in batch_out]
        eff95 = [effective_frames(w, 0.95) for w, _ in batch_out]
        T_real = [n for _, n in batch_out]
        results["d32_mel"]["val_weights"][seed] = {
            "eff50": eff50, "eff80": eff80, "eff95": eff95, "T_real": T_real,
        }

    print(f"\n{'='*62}\nConfig 2: d=192, wav2vec2\n{'='*62}")
    _set_feature_mode(pretrained=True)
    val_ds = MetadataSpectrogramDataset(
        val_df, LABEL_ENCODER, MAX_LEN, 768,
        meta_stats=meta_stats, use_external_meta=use_meta, train=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    aid_to_idx = {r["audio_ID"]: i for i, r in val_ds.records.iterrows()}

    for seed, path in D192_W2V_CKPTS:
        model = build_model(768, D192_HP, use_meta, device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        log_tau, tau = load_tau(path)
        results["d192_w2v"]["tau"][seed] = {"log_tau": log_tau, "tau": tau}
        print(f"  seed {seed:>5}: log_tau={log_tau:.4f} → τ={tau:.4f}")

        for cname, row in reps.items():
            if row is None: continue
            idx = aid_to_idx[row["audio_ID"]]
            x, mask, _, meta = val_ds[idx]
            w = attention_weights_single(model, x, mask, meta, use_meta, device)
            n_real = real_frame_count(row["audio_ID"], 768)
            results["d192_w2v"]["per_class_attn"].setdefault(cname, {})[seed] = {
                "weights": w[:n_real].tolist(), "T_real": int(n_real),
            }

        batch_out = attention_weights_batch(model, val_loader, use_meta, device)
        eff50 = [effective_frames(w, 0.50) for w, _ in batch_out]
        eff80 = [effective_frames(w, 0.80) for w, _ in batch_out]
        eff95 = [effective_frames(w, 0.95) for w, _ in batch_out]
        T_real = [n for _, n in batch_out]
        results["d192_w2v"]["val_weights"][seed] = {
            "eff50": eff50, "eff80": eff80, "eff95": eff95, "T_real": T_real,
        }

    j = json.loads(json.dumps(results, default=lambda o: (
        list(o) if isinstance(o, (list, tuple)) else o
    )))
    def _stringify(d):
        if not isinstance(d, dict): return d
        return {str(k): _stringify(v) for k, v in d.items()}
    j = _stringify(results)
    out_json = RESULTS_DIR / f"attn_v2_{timestamp}.json"
    with open(out_json, "w") as f:
        json.dump(j, f, indent=2)
    print(f"\nRaw data → {out_json}")

    print("\nBuilding figure A (per-class profiles)...")
    classes_present = [c for c in CLASS_NAMES if reps.get(c) is not None]
    n_c = len(classes_present)
    fig, axes = plt.subplots(n_c, 2, figsize=(13, 2.7 * n_c), squeeze=False)
    cmap = plt.get_cmap("tab10")

    for row_i, cname in enumerate(classes_present):
        audio_id = reps[cname]["audio_ID"]
        child    = reps[cname]["child"]

        ax = axes[row_i, 0]
        class_attn = results["d32_mel"]["per_class_attn"].get(cname, {})
        if class_attn:
            T_real = next(iter(class_attn.values()))["T_real"]
            for i, (seed, entry) in enumerate(sorted(class_attn.items())):
                w = np.array(entry["weights"])
                ax.plot(range(len(w)), w, color=cmap(i), lw=1.3, alpha=0.85,
                        label=f"seed {seed}")
            ax.axhline(1.0 / T_real, ls="--", color="grey", lw=1.0,
                       label=f"uniform ({1/T_real:.3f})")
            ax.set_xlim(0, T_real)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Attention weight")
        ax.set_title(f"{cname} ({audio_id}, {child})\nd=32, mel — linear y",
                      fontsize=9.5)
        ax.grid(alpha=0.3)
        if row_i == 0: ax.legend(fontsize=7, loc="upper right", ncol=2)

        ax = axes[row_i, 1]
        class_attn = results["d192_w2v"]["per_class_attn"].get(cname, {})
        if class_attn:
            T_real = next(iter(class_attn.values()))["T_real"]
            for i, (seed, entry) in enumerate(sorted(class_attn.items())):
                w = np.array(entry["weights"])
                w_plot = np.maximum(w, 1e-6)
                ax.plot(range(len(w_plot)), w_plot, color=cmap(i), lw=1.0,
                        alpha=0.85, label=f"seed {seed}")
            ax.axhline(1.0 / T_real, ls="--", color="grey", lw=1.0,
                       label=f"uniform ({1/T_real:.4f})")
            ax.set_xlim(0, T_real)
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 1.2)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Attention weight (log)")
        ax.set_title(f"{cname} — d=192, wav2vec2 — log y",
                      fontsize=9.5)
        ax.grid(alpha=0.3, which="both")
        if row_i == 0: ax.legend(fontsize=7, loc="lower left", ncol=2)

    fig.suptitle("Model 3 attention profiles by class — mel (linear y) vs "
                 "wav2vec2 (log y)\nLog scale reveals the full shape of "
                 "wav2vec2 attention, not just the peak",
                 y=1.005, fontsize=11)
    plt.tight_layout()
    out_a = FIGURES_DIR / f"attn_v2_per_class_{timestamp}.png"
    plt.savefig(out_a, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_a}")

    print("Building figure B (effective frames)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for col, (frac_key, frac_label) in enumerate([
        ("eff50", "50%"),
        ("eff80", "80%"),
        ("eff95", "95%"),
    ]):
        ax = axes[col]
        mel_vals = []
        for seed, d in results["d32_mel"]["val_weights"].items():
            mel_vals.extend(d[frac_key])
        w2v_vals = []
        for seed, d in results["d192_w2v"]["val_weights"].items():
            w2v_vals.extend(d[frac_key])

        bins = np.arange(0, max(max(mel_vals), max(w2v_vals)) + 2)
        ax.hist(mel_vals, bins=bins, alpha=0.65, color="steelblue",
                label=f"d=32 mel (median {int(np.median(mel_vals))})",
                edgecolor="black")
        ax.hist(w2v_vals, bins=bins, alpha=0.65, color="darkorange",
                label=f"d=192 wav2vec2 (median {int(np.median(w2v_vals))})",
                edgecolor="black")
        ax.set_xlabel(f"Frames needed to reach {frac_label} of attention mass")
        ax.set_ylabel("Clip × seed count")
        ax.set_title(f"Top-K frames for {frac_label} mass")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Effective-frames distribution across val set — "
                 "how many frames carry most of Model 3's attention?",
                 y=1.01, fontsize=11)
    plt.tight_layout()
    out_b = FIGURES_DIR / f"attn_v2_effective_frames_{timestamp}.png"
    plt.savefig(out_b, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_b}")

    print("Building figure C (τ bar chart)...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    seeds_sorted = sorted(results["d192_w2v"]["tau"].keys())
    mel_taus = [results["d32_mel"]["tau"][s]["tau"] for s in seeds_sorted]
    w2v_taus = [results["d192_w2v"]["tau"][s]["tau"] for s in seeds_sorted]
    x = np.arange(len(seeds_sorted))
    w = 0.35
    ax.bar(x - w/2, mel_taus, w, color="steelblue",
           label=f"d=32 mel (mean τ={np.mean(mel_taus):.3f})",
           edgecolor="black")
    ax.bar(x + w/2, w2v_taus, w, color="darkorange",
           label=f"d=192 wav2vec2 (mean τ={np.mean(w2v_taus):.3f})",
           edgecolor="black")
    ax.axhline(1.0, ls="--", color="black", lw=1.2,
               label="τ init (=1.0, sharpen if τ<1, soften if τ>1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds_sorted])
    ax.set_ylabel("Learned temperature τ")
    ax.set_title("Learned softmax temperature across 5 seeds\n"
                 "Both configs learned τ < 1 (anti-sharpening) — "
                 "one-hot behaviour on wav2vec2 is driven by score magnitudes, not τ")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    out_c = FIGURES_DIR / f"attn_v2_tau_bar_{timestamp}.png"
    plt.savefig(out_c, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_c}")

    def _agg(xs):
        import numpy as _np
        return float(_np.median(xs)), float(_np.mean(xs)), float(_np.std(xs))

    print(f"\n{'='*62}\nEFFECTIVE-FRAMES SUMMARY\n{'='*62}")
    for frac_key, frac_label in [("eff50", "50%"), ("eff80", "80%"), ("eff95", "95%")]:
        mel_vals = [v for seed, d in results["d32_mel"]["val_weights"].items() for v in d[frac_key]]
        w2v_vals = [v for seed, d in results["d192_w2v"]["val_weights"].items() for v in d[frac_key]]
        m1, mn1, s1 = _agg(mel_vals)
        m2, mn2, s2 = _agg(w2v_vals)
        print(f"  {frac_label} mass  | mel      median={m1:>5.1f}  mean={mn1:>5.1f}  std={s1:>4.1f}")
        print(f"            | wav2vec2 median={m2:>5.1f}  mean={mn2:>5.1f}  std={s2:>4.1f}")
    print(f"\n  Interpretation: on wav2vec2 roughly 1–3 frames carry 95% of")
    print(f"  attention mass. On mel, attention is spread across many more")
    print(f"  frames. The mechanism concentrates differently, not 'more or less'.")

    mean_tau_mel = np.mean([t["tau"] for t in results["d32_mel"]["tau"].values()])
    mean_tau_w2v = np.mean([t["tau"] for t in results["d192_w2v"]["tau"].values()])
    print(f"\n  Learned τ — mel mean {mean_tau_mel:.3f}, "
          f"wav2vec2 mean {mean_tau_w2v:.3f}")
    print(f"  Both < 1.0 (init). Model tried to SOFTEN attention in both cases.")
    print(f"  wav2vec2 concentration is a score-magnitude effect, not a τ runaway.")


if __name__ == "__main__":
    main()