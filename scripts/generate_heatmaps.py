"""
generate_spectrograms.py
------------------------
Generate viridis mel spectrogram PNGs.
Saves to RES_DIR/spectrograms/audio_XXXX.png

Usage:
    python scripts/generate_heatmaps.py
    python scripts/generate_heatmaps.py --limit 10
    python scripts/generate_heatmaps.py --skip_existing
"""

import argparse
import io
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as cfg


def wav_to_png(wav_path: Path, out_path: Path) -> None:
    """Convert one WAV file to a viridis spectrogram PNG."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    y, _ = librosa.load(str(wav_path), sr=cfg.SPEC_SR, mono=True)
    if len(y) == 0:
        raise ValueError("Empty audio file")

    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.SPEC_SR,
        n_fft=cfg.SPEC_N_FFT, hop_length=cfg.SPEC_HOP,
        n_mels=cfg.SPEC_N_MELS, fmin=cfg.SPEC_FMIN, fmax=cfg.SPEC_FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)

    fig, ax = plt.subplots(figsize=(9, 3), dpi=110)
    img_plot = librosa.display.specshow(
        log_mel,
        sr=cfg.SPEC_SR, hop_length=cfg.SPEC_HOP,
        x_axis="time", y_axis="mel",
        fmin=cfg.SPEC_FMIN, fmax=cfg.SPEC_FMAX,
        cmap="viridis", vmin=-80, vmax=0, ax=ax,
    )
    cbar = fig.colorbar(img_plot, ax=ax, format="%+.0f dB", pad=0.01)
    cbar.set_label("dB", rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Hz", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf).convert("RGB")
    pil_img = pil_img.resize((1000, 400), Image.LANCZOS)
    pil_img.save(str(out_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit",         type=int,  default=None)
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    cfg.print_config()

    # ── Dependency check ─────────────────────────────────────────────────
    missing = []
    for pkg in ["librosa", "matplotlib", "PIL"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n✗ Missing packages: {missing}")
        print(f"  Run: pip install {' '.join(missing)}")
        sys.exit(1)
    print("Dependencies: librosa ✓  matplotlib ✓  PIL ✓")

    # ── Locate WAV files ─────────────────────────────────────────────────
    if not cfg.WAV_DIR.exists():
        print(f"\n✗ WAV dir not found: {cfg.WAV_DIR}")
        sys.exit(1)

    print(f"Scanning {cfg.WAV_DIR} ...", end=" ", flush=True)
    wav_files = sorted(cfg.WAV_DIR.glob("*.wav"))
    print(f"found {len(wav_files)} files")

    if not wav_files:
        print("✗ No WAV files found.")
        sys.exit(1)

    if args.limit:
        wav_files = wav_files[:args.limit]

    # ── Create output dir ────────────────────────────────────────────────
    cfg.SPEC_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir : {cfg.SPEC_DIR}  (exists={cfg.SPEC_DIR.exists()})")
    print(f"Generating {len(wav_files)} spectrograms ...\n")

    # ── Test first file immediately ──────────────────────────────────────
    test_wav = wav_files[0]
    test_out = cfg.SPEC_DIR / (test_wav.stem + ".png")
    print(f"[TEST] First file: {test_wav.name} → {test_out.name}", flush=True)
    try:
        wav_to_png(test_wav, test_out)
        print(f"[TEST] ✓ Saved  size={test_out.stat().st_size} bytes\n")
    except Exception:
        print(f"[TEST] ✗ FAILED")
        traceback.print_exc()
        sys.exit(1)

    # ── Process remaining files ──────────────────────────────────────────
    from tqdm import tqdm
    generated = 1   # first file already done
    skipped   = 0
    failed    = 0

    for wav_path in tqdm(wav_files[1:], desc="Generating"):
        out_path = cfg.SPEC_DIR / (wav_path.stem + ".png")

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            wav_to_png(wav_path, out_path)
            generated += 1
        except Exception as e:
            tqdm.write(f"  ⚠ {wav_path.name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Generated : {generated}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"\nNext: python scripts/train_transformer_audio.py")


if __name__ == "__main__":
    main()