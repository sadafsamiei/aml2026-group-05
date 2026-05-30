"""
scripts/train_hydra.py
----------------------
Hydra-enabled training wrapper for Models 2 and 3.
Loads config from YAML files

Usage
-----
    python scripts/train_hydra.py --config-name audio_mel

    python scripts/train_hydra.py --config-name attn_wav2vec2

    python scripts/train_hydra.py --config-name attn_mel \\
        model.dropout_encoder=0.4 \\
        model.dropout_head=0.2 \\
        model.dropout_input=0.1 \\
        model.dropout_meta=0.15 \\
        training.lr=5e-4

    python scripts/train_hydra.py --config-name audio_mel training.seed=42 / 123 / 456 / 789 / 2024


"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    print("ERROR: hydra-core is not installed. Run:  pip install hydra-core omegaconf")
    sys.exit(1)

import config as cfg
from config import (
    N_CLASSES, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, RES_DIR, RANDOM_SEED,
)
from data_utils import (
    build_master_table, audit_spectrograms,
    determine_eval_protocol, build_splits, load_splits,
    SpectrogramDataset,
)
from models import AudioTransformer, AttentivePoolingTransformer
from train_utils import train_model, evaluate, save_results


def build_model(model_cfg: DictConfig, f_dim: int, max_len: int) -> torch.nn.Module:
    """Instantiate the correct model class from the Hydra model config."""
    features = model_cfg.get("features", "mel")
    is_attn  = model_cfg.get("use_external_meta", False)

    d_enc   = model_cfg.get("dropout_encoder", 0.3)
    d_input = model_cfg.get("dropout_input",   0.0)
    d_head  = model_cfg.get("dropout_head",    0.3)

    if is_attn:
        d_meta = model_cfg.get("dropout_meta", 0.0)
        return AttentivePoolingTransformer(
            input_dim         = f_dim,
            d_model           = model_cfg.d_model,
            nhead             = model_cfg.nhead,
            num_layers        = model_cfg.num_layers,
            dropout_encoder   = d_enc,
            dropout_input     = d_input,
            dropout_head      = d_head,
            dropout_meta      = d_meta,
            use_external_meta = True,
            meta_dim          = model_cfg.get("meta_dim", 2),
            num_classes       = N_CLASSES,
            max_len           = max_len,
        )
    else:
        return AudioTransformer(
            input_dim       = f_dim,
            d_model         = model_cfg.d_model,
            nhead           = model_cfg.nhead,
            num_layers      = model_cfg.num_layers,
            dropout_encoder = d_enc,
            dropout_input   = d_input,
            dropout_head    = d_head,
            num_classes     = N_CLASSES,
            max_len         = max_len,
        )


def run_single_hydra(model_cfg: DictConfig, train_cfg: DictConfig,
                     folds, f_dim: int, max_len: int, device):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features  = model_cfg.get("features", "mel")
    model_tag = "attn" if model_cfg.get("use_external_meta", False) else "audio"

    all_results = []
    for fold_i, (train_df, val_df, test_df) in enumerate(folds):
        fold_tag = f"{model_tag}_{features}_seed{train_cfg.seed}_fold{fold_i}"
        print(f"\n── Fold {fold_i} ({fold_tag}) ──")

        ds_kwargs = {"features": features} if features != "mel" else {}
        train_ds  = SpectrogramDataset(train_df, augment=True,  **ds_kwargs)
        val_ds    = SpectrogramDataset(val_df,   augment=False, **ds_kwargs)
        test_ds   = SpectrogramDataset(test_df,  augment=False, **ds_kwargs)

        bs = int(train_cfg.batch_size)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

        model = build_model(model_cfg, f_dim, max_len).to(device)
        model.print_architecture()

        criterion = torch.nn.CrossEntropyLoss(
            weight=train_ds.class_weights().to(device),
            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        )

        save_path = MODELS_DIR / f"{fold_tag}.pt"
        train_result = train_model(
            model, train_loader, val_loader, criterion, device,
            lr=float(model_cfg.lr),
            weight_decay=float(train_cfg.weight_decay),
            epochs=int(train_cfg.epochs),
            patience=int(train_cfg.patience),
            save_path=save_path,
            verbose=True,
        )

        model.load_state_dict(torch.load(save_path, map_location=device))
        val_result  = evaluate(model, val_loader,  device)
        test_result = evaluate(model, test_loader, device)

        print(f"  Val  macro-F1 : {val_result['macro_f1']:.4f}")
        print(f"  Test macro-F1 : {test_result['macro_f1']:.4f}")

        fold_summary = {
            "fold"          : fold_i,
            "seed"          : int(train_cfg.seed),
            "model_cfg"     : OmegaConf.to_container(model_cfg),
            "train_cfg"     : OmegaConf.to_container(train_cfg),
            "val_macro_f1"  : val_result["macro_f1"],
            "test_macro_f1" : test_result["macro_f1"],
            "stopped_epoch" : train_result.get("stopped_epoch", -1),
        }
        all_results.append(fold_summary)
        save_results(fold_summary, RESULTS_DIR / f"fold_results_{fold_tag}_{timestamp}.json")

    val_f1s  = [r["val_macro_f1"]  for r in all_results]
    test_f1s = [r["test_macro_f1"] for r in all_results]
    summary = {
        "model_kind"        : model_tag,
        "features"          : features,
        "seed"              : int(train_cfg.seed),
        "model_cfg"         : OmegaConf.to_container(model_cfg),
        "train_cfg"         : OmegaConf.to_container(train_cfg),
        "val_macro_f1_mean" : float(np.mean(val_f1s)),
        "val_macro_f1_std"  : float(np.std(val_f1s)),
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std" : float(np.std(test_f1s)),
    }
    save_results(summary, RESULTS_DIR / f"{model_tag}_{features}_seed{train_cfg.seed}_{timestamp}.json")
    return summary

@hydra.main(config_path="../configs/model", config_name="audio_mel",
            version_base=None)
def main(model_cfg: DictConfig):

    from omegaconf import OmegaConf
    train_cfg_path = Path(__file__).parent.parent / "configs" / "training" / "default.yaml"
    train_cfg = OmegaConf.load(train_cfg_path)

    seed = int(train_cfg.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n── Model config ──")
    print(OmegaConf.to_yaml(model_cfg))
    print("── Training config ──")
    print(OmegaConf.to_yaml(train_cfg))

    cfg.print_config()
    cfg.make_output_dirs()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice: {device}")

    print("\n[1/3] Building master table...")
    df = build_master_table(verbose=True)
    df = df[df["png_exists"]].copy()

    print("\n[2/3] Auditing spectrograms...")
    audit_spectrograms(df["png_path"].tolist(), verbose=False)
    f_dim   = cfg.F_DIM
    max_len = cfg.MAX_LEN

    print("\n[3/3] Constructing splits...")
    protocol = determine_eval_protocol(df)
    folds    = build_splits(df, protocol, save=True)

    result = run_single_hydra(model_cfg, train_cfg, folds, f_dim, max_len, device)

    print(f"\n── Final result ──")
    print(f"  Val  macro-F1 : {result['val_macro_f1_mean']:.4f} "
          f"± {result['val_macro_f1_std']:.4f}")
    print(f"  Test macro-F1 : {result['test_macro_f1_mean']:.4f} "
          f"± {result['test_macro_f1_std']:.4f}")


if __name__ == "__main__":
    main()
