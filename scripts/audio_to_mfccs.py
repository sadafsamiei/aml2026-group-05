"""
audio_to_mfccs.py
-----------------
Extract transformer-compatible MFCC features from denoised audio files.

Produces (128, 230) arrays per clip to match the existing mel spectrogram
pipeline (F_DIM=128, MAX_LEN=230) so no config changes are needed:
  - 40 MFCCs + 40 deltas + 40 delta-deltas  →  120 frequency bins
  - Zero-padded from 120 → 128 on the frequency axis
  - Padded / truncated to 230 time frames

Usage
-----
    # Default: matches pipeline dimensions exactly (128 × 230)
    python scripts/audio_to_mfccs.py

    # Custom settings
    python scripts/audio_to_mfccs.py --n_mfcc 20 --max_frames 300
"""

import os
import re
import sys
import argparse
import numpy as np

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import config as cfg



def extract_number(file_name: str) -> int:
    match = re.search(r"\d+", file_name)
    return int(match.group()) if match else float("inf")


def extract_mfcc_features(
    file_path: str,
    sr: int = 22050,
    n_mfcc: int = 40,
    target_freq_bins: int = 128,
    max_frames: int = 230,
) -> np.ndarray:

    import librosa

    audio, _ = librosa.load(file_path, sr=sr)

    mfccs  = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc) 
    delta1 = librosa.feature.delta(mfccs)                         
    delta2 = librosa.feature.delta(mfccs, order=2)               

    features = np.vstack([mfccs, delta1, delta2])                  

    freq_bins = features.shape[0]
    if freq_bins < target_freq_bins:
        pad_rows = target_freq_bins - freq_bins
        features = np.pad(features, ((0, pad_rows), (0, 0)))
    elif freq_bins > target_freq_bins:
        features = features[:target_freq_bins, :]

    T = features.shape[1]
    if T < max_frames:
        features = np.pad(features, ((0, 0), (0, max_frames - T)))
    else:
        features = features[:, :max_frames]

    return features.astype(np.float32)                            


def main():
    parser = argparse.ArgumentParser(
        description="Extract transformer-ready MFCC features from .wav files. "
                    "Output shape matches mel spectrograms (128×230) by default."
    )
    parser.add_argument(
        "--audio_dir", type=str, default=None,
        help=f"Folder containing denoised .wav files "
             f"(default: {{DATA_DIR}}/denoised_audios)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help=f"Folder to save .npy MFCC arrays "
             f"(default: {{RES_DIR}}/mfccs)",
    )
    parser.add_argument("--sr", type=int, default=22050,
                        help="Sample rate (default: 22050)")
    parser.add_argument("--n_mfcc", type=int, default=40,
                        help="Number of MFCCs (default: 40, stacked with Δ+ΔΔ → 120)")
    parser.add_argument("--target_freq_bins", type=int, default=cfg.F_DIM,
                        help=f"Pad/truncate frequency axis to this size "
                             f"(default: F_DIM={cfg.F_DIM})")
    parser.add_argument("--max_frames", type=int, default=cfg.MAX_LEN,
                        help=f"Pad/truncate time axis to this size "
                             f"(default: MAX_LEN={cfg.MAX_LEN})")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip clips that already have a .npy file")
    args = parser.parse_args()

    audio_dir  = args.audio_dir or str(cfg.WAV_DIR)
    output_dir = args.output_dir or str(cfg.RES_DIR / "mfccs")

    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(
        [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")],
        key=extract_number,
    )

    print(f"MFCC Extraction")
    print(f"  Audio dir     : {audio_dir}")
    print(f"  Output dir    : {output_dir}")
    print(f"  n_mfcc        : {args.n_mfcc}  (stacked: {args.n_mfcc * 3} → padded to {args.target_freq_bins})")
    print(f"  Output shape  : ({args.target_freq_bins}, {args.max_frames})")
    print(f"  Clips found   : {len(wav_files)}")
    print(f"  Skip existing : {args.skip_existing}")
    print()

    n_processed = 0
    n_skipped   = 0

    for i, fname in enumerate(wav_files):
        out_name = os.path.splitext(fname)[0] + ".npy"
        out_path = os.path.join(output_dir, out_name)

        if args.skip_existing and os.path.exists(out_path):
            n_skipped += 1
            continue

        src = os.path.join(audio_dir, fname)
        features = extract_mfcc_features(
            src,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            target_freq_bins=args.target_freq_bins,
            max_frames=args.max_frames,
        )
        np.save(out_path, features)
        n_processed += 1

        if n_processed % 100 == 0 or (i + 1) == len(wav_files):
            print(f"  [{i + 1:>{len(str(len(wav_files)))}}/{len(wav_files)}] "
                  f"{fname} → {out_name}  {features.shape}")

    print(f"\nDone. {n_processed} files written, {n_skipped} skipped.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
