"""
generate_spectrograms.py
------------------------
Generate publication-quality viridis mel spectrogram PNGs with Hz axis,
time axis, and dB colorbar — matching the format of the existing pilot
spectrograms in examples/spectrograms/.

Run ONCE before any training script. Safe to re-run with --skip_existing.

Usage:
    python scripts/generate_spectrograms.py              # all 1009 files
    python scripts/generate_spectrograms.py --limit 10  # test on 10 files
    python scripts/generate_spectrograms.py --skip_existing  # safe re-run

Output format per clip:
    File      : RES_DIR/spectrograms/audio_XXXX.png
    Mode      : RGB
    Size      : 1000 × 400 pixels (width × height) — fixed for all clips
    Colormap  : viridis  (purple=quiet → yellow=loud)
    Y-axis    : Frequency in Hz (0 to fmax)
    X-axis    : Time in seconds
    Colorbar  : dB scale (−80 dB to 0 dB)

After running, config.py already has the correct values:
    F_DIM   = 400
    MAX_LEN = 1000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tqdm import tqdm

import config as cfg

# Fixed output image dimensions (matches pilot spectrogram format)
OUT_WIDTH  = 1000
OUT_HEIGHT = 400


def wav_to_spectrogram_png(wav_path: Path, out_path: Path) -> None:
    """
    Compute a log-power mel spectrogram and save as a labelled, viridis-
    coloured RGB PNG with Hz Y-axis, time X-axis, and dB colorbar.

    Steps
    -----
    1. Load and resample audio to SPEC_SR.
    2. Compute mel spectrogram → convert to dB (top_db=80, ref=np.max).
    3. Render with librosa.display.specshow using viridis colormap.
    4. Add frequency axis (Hz), time axis (s), and dB colorbar.
    5. Save figure to in-memory buffer → resize to OUT_WIDTH × OUT_HEIGHT.
    6. Write final RGB PNG to out_path.
    """
    import io
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    from PIL import Image

    # ── 1. Load audio ──────────────────────────────────────────────────
    y, _ = librosa.load(str(wav_path), sr=cfg.SPEC_SR, mono=True)
    if len(y) == 0:
        raise ValueError(f"Empty audio: {wav_path.name}")

    # ── 2. Mel spectrogram → log dB ────────────────────────────────────
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.SPEC_SR,
        n_fft=cfg.SPEC_N_FFT, hop_length=cfg.SPEC_HOP,
        n_mels=cfg.SPEC_N_MELS, fmin=cfg.SPEC_FMIN, fmax=cfg.SPEC_FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)
    # Shape: (n_mels, T), range: [−80, 0] dB

    # ── 3. Render with librosa.display.specshow ─────────────────────────
    # figsize chosen so the rendered image lands close to 1000×400
    fig, ax = plt.subplots(figsize=(9, 3), dpi=110)

    img = librosa.display.specshow(
        log_mel,
        sr          = cfg.SPEC_SR,
        hop_length  = cfg.SPEC_HOP,
        x_axis      = "time",           # seconds on X
        y_axis      = "mel",            # Hz on Y
        fmin        = cfg.SPEC_FMIN,
        fmax        = cfg.SPEC_FMAX,
        cmap        = "viridis",
        vmin        = -80,              # consistent dB floor across all clips
        vmax        = 0,                # consistent dB ceiling
        ax          = ax,
    )

    # ── 4. Colorbar, labels ─────────────────────────────────────────────
    cbar = fig.colorbar(img, ax=ax, format="%+.0f dB", pad=0.01)
    cbar.set_label("dB", rotation=270, labelpad=12, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Hz", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

    plt.tight_layout(pad=0.4)

    # ── 5. Save to buffer → resize to fixed output dimensions ───────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf).convert("RGB")
    pil_img = pil_img.resize((OUT_WIDTH, OUT_HEIGHT), Image.LANCZOS)

    # ── 6. Write PNG ─────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_path))


def main():
    p = argparse.ArgumentParser(
        description="Generate labelled viridis mel spectrogram PNGs")
    p.add_argument("--limit",         type=int, default=None,
                   help="Process only first N files (for testing)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip files whose PNG already exists")
    args = p.parse_args()

    cfg.print_config()

    # Dependency check
    for pkg in ["librosa", "matplotlib", "PIL"]:
        try:
            __import__(pkg)
        except ImportError:
            print(f"✗ {pkg} not installed.  Run:  pip install {pkg}")
            sys.exit(1)

    if not cfg.WAV_DIR.exists():
        print(f"\n✗ WAV directory not found: {cfg.WAV_DIR}")
        print("  Check that training_data/anonymized_audios/ is mounted.")
        sys.exit(1)

    wav_files = sorted(cfg.WAV_DIR.glob("*.wav"))
    if not wav_files:
        print(f"✗ No WAV files in {cfg.WAV_DIR}")
        sys.exit(1)

    if args.limit:
        wav_files = wav_files[:args.limit]

    cfg.SPEC_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nWAV files        : {len(wav_files)}")
    print(f"Output dir       : {cfg.SPEC_DIR}")
    print(f"Output size      : {OUT_WIDTH} × {OUT_HEIGHT} px  (W × H, fixed)")
    print(f"Colormap         : viridis  (purple = quiet, yellow = loud)")
    print(f"dB range         : −80 dB to 0 dB  (fixed, consistent across all clips)")
    print(f"Axes             : Hz (Y) and Time/s (X), dB colorbar")
    print(f"sr={cfg.SPEC_SR}, n_mels={cfg.SPEC_N_MELS}, "
          f"hop={cfg.SPEC_HOP}, n_fft={cfg.SPEC_N_FFT}\n")

    skipped = generated = failed = 0

    for wav_path in tqdm(wav_files, desc="Generating"):
        out_path = cfg.SPEC_DIR / (wav_path.stem + ".png")
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        try:
            wav_to_spectrogram_png(wav_path, out_path)
            generated += 1
        except Exception as e:
            print(f"\n  ⚠ Failed: {wav_path.name} — {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Generated : {generated}")
    print(f"Skipped   : {skipped}  (already existed)")
    print(f"Failed    : {failed}")
    print(f"\nAll images: {OUT_WIDTH} × {OUT_HEIGHT} px, RGB, viridis + dB labels")
    print(f"\nconfig.py values to confirm:")
    print(f"  F_DIM   = {OUT_HEIGHT}   (image height)")
    print(f"  MAX_LEN = {OUT_WIDTH}    (image width, all same size)")
    print(f"\nNext: python scripts/train_transformer_audio.py")
    print("=" * 55)


if __name__ == "__main__":
    main()