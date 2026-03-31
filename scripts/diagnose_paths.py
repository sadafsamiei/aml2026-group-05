"""
diagnose_paths.py  [FIXED v2]
-----------------
Diagnose the positional PNG-to-CSV mapping.

Usage:
    python scripts/diagnose_paths.py
"""

import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
from config import CONTEXTS_CSV, SPEC_DIR, LANG_FILTER


def numeric_sort_key(p):
    nums = re.findall(r'\d+', p.stem)
    return int(nums[0]) if nums else 0


def main():
    print("=" * 65)
    print("PATH DIAGNOSTIC  (positional mapping)")
    print("=" * 65)

    # [1] Directory
    print(f"\n[1] Spectrograms directory: {SPEC_DIR}")
    if not SPEC_DIR.exists():
        print(f"    ✗ DOES NOT EXIST — fix DATA_DIR in src/config.py")
        return
    print("    ✓ exists")

    # [2] Unique PNGs (deduplicated, numerically sorted)
    all_pngs  = list(SPEC_DIR.rglob("*.png"))
    seen, unique = set(), []
    for p in all_pngs:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); unique.append(p)
    unique.sort(key=numeric_sort_key)

    print(f"\n[2] Unique PNG files (deduplicated): {len(unique)}")
    if not unique:
        print("    ✗ NO PNGs FOUND"); return
    print(f"    First : {unique[0].name}")
    print(f"    Last  : {unique[-1].name}")
    print(f"    Naming: {unique[0].stem!r} → integer = "
          f"{numeric_sort_key(unique[0])}")

    # [3] CSV rows
    df = pd.read_csv(CONTEXTS_CSV)
    if LANG_FILTER:
        df = df[df["lang"] == LANG_FILTER].copy()
    print(f"\n[3] CSV rows after language filter '{LANG_FILTER}': {len(df)}")

    # [4] Positional mapping
    n_png = len(unique)
    n_csv = len(df)
    print(f"\n[4] Positional mapping")
    print(f"    PNGs : {n_png}")
    print(f"    Rows : {n_csv}")

    if n_png == n_csv:
        print(f"    ✓ Counts match — perfect 1-to-1 positional mapping")
    elif n_png < n_csv:
        print(f"    ⚠  {n_csv - n_png} CSV rows will have no PNG.")
        print(f"       This is expected if more spectrograms are being generated.")
        print(f"       Only the first {n_png} rows will be usable for training.")
    else:
        print(f"    ⚠  More PNGs than CSV rows — first {n_csv} PNGs will be used.")

    print(f"\n    Mapping preview (first 5):")
    audio_files = df["audio_file"].tolist()
    for i in range(min(5, n_png, n_csv)):
        print(f"      {unique[i].name:25s}  ←  {audio_files[i]}")

    print(f"\n    Mapping preview (last 3 with PNG):")
    for i in range(max(0, min(n_png, n_csv) - 3), min(n_png, n_csv)):
        print(f"      {unique[i].name:25s}  ←  {audio_files[i]}")

    # [5] PNG format
    print(f"\n[5] PNG format check:")
    from PIL import Image
    img = Image.open(unique[0])
    w, h = img.size
    print(f"    File  : {unique[0].name}")
    print(f"    Mode  : {img.mode}  "
          f"{'(will be converted to grayscale)' if img.mode in ('RGB','RGBA') else '(grayscale, ready)'}")
    print(f"    Width : {w}  (time frames T)")
    print(f"    Height: {h}  (frequency bins F)")
    print(f"\n    >>> F_DIM   = {h}")
    print(f"    >>> MAX_LEN ≈ {w}  (verify with audit_spectrograms() across all files)")

    # [6] Class coverage for mapped rows
    MOVE  = {'MOVE2','MOVE3','MOVE5'}
    INTER = {'INTER1','INTER2','INTER4','INTER5','INTER6','INTER7'}
    EXTEV = {'EXTEV1','EXTEV2','EXTEV3','EXTEV6'}
    def ml(lbl):
        if lbl == 'PLTOY': return 'PLTOY'
        if lbl in MOVE: return 'MOVE'
        if lbl in INTER: return 'INTER'
        if lbl in EXTEV: return 'EXTEV'
        return None

    df_mapped = df.iloc[:min(n_png, n_csv)].copy()
    df_mapped["class4"] = df_mapped["context_label"].apply(ml)
    df_usable = df_mapped[df_mapped["class4"].notna()]
    print(f"\n[6] Class coverage in mapped rows ({len(df_usable)} usable):")
    print(f"    {df_usable['class4'].value_counts().to_dict()}")
    print(f"    Participants: {df_usable['child'].unique().tolist()}")

    print("\n" + "=" * 65)
    print("If mapping preview looks correct, the fix is working.")
    print("Run:  python scripts/test_pipeline.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
