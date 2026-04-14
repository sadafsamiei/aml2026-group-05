import librosa
import pandas as pd
import os

print(f"{'='*10} PROJECT SETUP CHECK {'='*10}")

# 1. Define Paths
paths = {
    "Raw": "training_data/anonymized_audios/audio_0002.wav",
    "Denoised": "training_data/denoised_audios/audio_0002.wav",
    "CSV": "training_data/mfccs_070426.csv"
}

# 2. Check Audio Files
for label in ["Raw", "Denoised"]:
    path = paths[label]
    if os.path.exists(path):
        y, sr = librosa.load(path, duration=1.0)
        print(f"SUCCESS - {label:8}: Found and loaded (SR: {sr}Hz)")
    else:
        print(f"FAILURE - {label:8}: File not found at {path}")

# 3. Check CSV Data
csv_path = paths["CSV"]
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"SUCCESS - CSV     : Found {csv_path} ({len(df)} rows)")
else:
    print(f"FAILURE - CSV     : File not found at {csv_path}")

print(f"{'='*33}")