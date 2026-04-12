"""
generate_spectrograms.py
------------------------
Generate clean grayscale mel spectrogram PNGs for model training.
Saves to RES_DIR/spectrograms/audio_XXXX.png

Each PNG is a true grayscale image of shape (SPEC_N_MELS, T) — no axes,
no colorbar, no matplotlib decoration. Pixel values map linearly from
the dB range [-80, 0] to [0, 255]. The model loads these at native
resolution and resizes to (MAX_LEN, F_DIM) inside SpectrogramDataset.

Usage:
    python scripts/generate_spectrograms.py
    python scripts/generate_spectrograms.py --limit 10
    python scripts/generate_spectrograms.py --skip_existing
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config as cfg


def wav_to_png(wav_path: Path, out_path: Path) -> None:
    """
    Convert one WAV file to a clean grayscale mel spectrogram PNG.

    Pipeline:
      WAV → mono float32 → mel spectrogram → dB scale → [0,255] uint8 → PNG

    The output image contains only acoustic content — no axes, labels, or
    colorbar. Shape: (SPEC_N_MELS=128 rows, T cols) where T depends on
    clip duration. SpectrogramDataset resizes to (F_DIM, MAX_LEN) at load time.
    """
    import librosa
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
    # Convert to dB, clipped to [-80, 0] range
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80)   # (128, T), range [-80, 0]

    # Map [-80, 0] → [0, 255] linearly
    # 0 dB (loudest) → 255 (white), -80 dB (silence) → 0 (black)
    img_arr = ((log_mel + 80.0) / 80.0 * 255.0).clip(0, 255).astype(np.uint8)

    # img_arr shape is (n_mels=128, T) — PIL expects (height, width)
    # Low frequencies are at row 0; flip vertically so low freqs are at bottom
    # (consistent with standard spectrogram orientation)
    img_arr = np.flipud(img_arr)

    Image.fromarray(img_arr, mode="L").save(str(out_path))


def main():
    p = argparse.ArgumentParser(
        description="Generate clean grayscale mel spectrogram PNGs for training"
    )
    p.add_argument("--limit",         type=int,  default=None,
                   help="Process only the first N files (for testing)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip files that already have a PNG")
    args = p.parse_args()

    cfg.print_config()

    # ── Dependency check ─────────────────────────────────────────────────
    missing = []
    for pkg in ["librosa", "PIL"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n✗ Missing packages: {missing}")
        print(f"  Run: pip install {' '.join(missing)}")
        sys.exit(1)
    print("Dependencies: librosa ✓  PIL ✓")

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
    print(f"Format     : grayscale PNG, {cfg.SPEC_N_MELS} mel bins × T frames")
    print(f"             (no axes/labels — clean acoustic content only)")
    print(f"Generating {len(wav_files)} spectrograms ...\n")

    # ── Test first file immediately ──────────────────────────────────────
    import numpy as np
    from PIL import Image

    test_wav = wav_files[0]
    test_out = cfg.SPEC_DIR / (test_wav.stem + ".png")
    print(f"[TEST] First file: {test_wav.name} → {test_out.name}", flush=True)
    try:
        wav_to_png(test_wav, test_out)
        img = Image.open(test_out)
        arr = np.array(img)
        print(f"[TEST] ✓  size={test_out.stat().st_size} bytes | "
              f"shape={arr.shape} | mode={img.mode} | "
              f"pixel range=[{arr.min()}, {arr.max()}]\n")
        assert img.mode == "L",          f"Expected grayscale, got {img.mode}"
        assert arr.shape[0] == cfg.SPEC_N_MELS, \
            f"Expected {cfg.SPEC_N_MELS} rows, got {arr.shape[0]}"
        assert arr.max() > 0,            "Silent clip — all-zero image"
    except Exception:
        print(f"[TEST] ✗ FAILED")
        traceback.print_exc()
        sys.exit(1)

    # ── Process remaining files ──────────────────────────────────────────
    from tqdm import tqdm
    generated = 1
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
    print(f"\nSpectrograms saved to {cfg.SPEC_DIR}")
    print(f"Next: python scripts/train_transformer_audio.py")


if __name__ == "__main__":
    main()