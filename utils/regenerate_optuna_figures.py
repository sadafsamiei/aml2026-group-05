"""
regenerate_optuna_figures.py
-------------------------------------
Regenerates the Optuna extension figures and fills the three previously
missing cells in the canonical-vs-Optuna chart from the added JSON files:

  • M2 MFCC canonical: audio_canonical_multiseed_M2_MFCC.json
  • M2 Mel Optuna:     audio_multiseed_optuna_M2_Mel.json (5-seed rerun; preferred)
  • M3 MFCC canonical: attn_canonical_multiseed_M3_MFCC.json

Usage
-----
From the project root:

    python utils/regenerate_optuna_figures_complete.py

or from the directory containing the JSON files:

    python regenerate_optuna_figures_complete.py --data-dir .

Outputs
-------
    <FIGURES_DIR>/presentation/fig5_complexity_vs_f1.png
    <FIGURES_DIR>/presentation/fig8_optuna_summary.png
    <FIGURES_DIR>/presentation/fig9_canonical_vs_optuna.png

The script reads the added JSONs when present, preferring the corrected
M2 Mel Optuna 5-seed rerun JSON over the single-trial search JSON. It also
keeps audited fallback values so the figures can still be regenerated if a
JSON is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Project path / output directory handling
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

# Works both when this file is placed in utils/ and when it is run standalone.
for candidate in [HERE.parent / "src", HERE / "src"]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))

try:
    from config import RESULTS_DIR, FIGURES_DIR  # type: ignore
except Exception:
    # Standalone fallback for running directly next to the uploaded JSON files.
    RESULTS_DIR = HERE
    FIGURES_DIR = HERE / "figures"

OUT_DIR = Path(FIGURES_DIR) / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    test_mean: float
    test_std: float | None
    n_params: int | None
    label: str = ""
    val_mean: float | None = None
    val_std: float | None = None

    def as_tuple3(self) -> tuple[float, float | None, int | None]:
        return (self.test_mean, self.test_std, self.n_params)

    def as_tuple4(self) -> tuple[float, float | None, int | None, str]:
        return (self.test_mean, self.test_std, self.n_params, self.label)


Cell = tuple[str, str]

CELL_ORDER: list[Cell] = [
    ("LR", "MFCC"),     ("LR", "Mel"),     ("LR", "wav2vec2"),
    ("M2", "MFCC"),     ("M2", "Mel"),     ("M2", "wav2vec2"),
    ("M3", "MFCC"),     ("M3", "Mel"),     ("M3", "wav2vec2"),
]

# ---------------------------------------------------------------------------
# Single source of truth for cells not supplied by the three new JSONs
# ---------------------------------------------------------------------------
# Values are test macro-F1 mean/std and approximate parameter counts used in
# the report figure.  The three previously missing entries are filled from JSON
# in load_results().

CANONICAL_FALLBACK: dict[Cell, Result | None] = {
    # LR — deterministic lbfgs, so std = 0
    ("LR", "MFCC"):     Result(0.253, 0.000,    164),
    ("LR", "Mel"):      Result(0.277, 0.000,   1028),
    ("LR", "wav2vec2"): Result(0.263, 0.000,   6148),

    # M2 / AudioTransformer
    ("M2", "MFCC"):     None,  # filled from audio_canonical_multiseed_M2_MFCC.json
    ("M2", "Mel"):      Result(0.249, 0.020, 251_140),
    ("M2", "wav2vec2"): Result(0.219, 0.018, 333_060),

    # M3 / AttentivePoolingTransformer
    ("M3", "MFCC"):     None,  # filled from attn_canonical_multiseed_M3_MFCC.json
    ("M3", "Mel"):      Result(0.221, 0.013,  38_789),
    ("M3", "wav2vec2"): Result(0.246, 0.016, 117_701),
}

OPTUNA_FALLBACK: dict[Cell, Result | None] = {
    # LR — best C from line search
    ("LR", "MFCC"):     Result(0.239, 0.000,    164, "C=0.01735"),
    ("LR", "Mel"):      Result(0.311, 0.000,   1028, "C=0.00492"),
    ("LR", "wav2vec2"): Result(0.285, 0.000,   6148, "C=0.01498"),

    # M2 — Optuna winners / reruns
    ("M2", "MFCC"):     Result(0.219, 0.029, 251_140, "d=128,L=1"),
    ("M2", "Mel"):      Result(0.226, 0.018, 251_140, "d=32,L=2"),  # overwritten by audio_multiseed_optuna_M2_Mel.json when present
    ("M2", "wav2vec2"): Result(0.251, 0.020, 333_060, "d=128,L=1"),

    # M3 — Optuna winners / reruns
    ("M3", "MFCC"):     Result(0.217, 0.038,  38_789, "d=32,L=2"),
    ("M3", "Mel"):      Result(0.233, 0.014,  38_789, "d=64,L=2"),
    ("M3", "wav2vec2"): Result(0.241, 0.021, 117_701, "d=64,L=1"),
}

# Parameter-count fallbacks for JSONs that do not store n_params.
# These are used only for log-complexity plotting; fig9 itself does not depend
# on n_params.
PARAM_FALLBACK: dict[Cell, int] = {
    ("M2", "MFCC"): 251_140,
    ("M2", "Mel"): 251_140,
    ("M3", "MFCC"): 38_789,
}

# ---------------------------------------------------------------------------
# Colours and visual conventions
# ---------------------------------------------------------------------------

COLOR_M1 = "#1f77b4"
COLOR_M2 = "#ff7f0e"
COLOR_M3 = "#2ca02c"
COLOR_OPT = "#e91e63"       # pink = genuine gain
COLOR_DEG = "#888888"       # grey = genuine loss
COLOR_CHANCE = "#bbbbbb"
COLOR_CANON = "lightsteelblue"
COLOR_NOISE = "lightcoral"

CHANCE_LEVEL = 0.245
CHANCE_BAND = 0.027

# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def _candidate_dirs(data_dir: Path | None) -> list[Path]:
    dirs = []
    for d in [data_dir, HERE, Path(RESULTS_DIR), Path.cwd()]:
        if d is None:
            continue
        d = Path(d).resolve()
        if d.exists() and d not in dirs:
            dirs.append(d)
    return dirs


def find_json(filename: str, data_dir: Path | None = None) -> Path | None:
    for d in _candidate_dirs(data_dir):
        exact = d / filename
        if exact.exists():
            return exact
        matches = sorted(d.rglob(filename)) if d.is_dir() else []
        if matches:
            return matches[-1]
    return None


def latest_json(patterns: Iterable[str], data_dir: Path | None = None) -> Path | None:
    matches: list[Path] = []
    for d in _candidate_dirs(data_dir):
        for pattern in patterns:
            matches.extend(d.glob(pattern))
    if not matches:
        return None
    return sorted(set(matches), key=lambda p: p.stat().st_mtime)[-1]


def _metric_pair(data: dict, key: str) -> tuple[float, float | None]:
    metric = data[key]
    if isinstance(metric, dict):
        return float(metric["mean"]), None if metric.get("std") is None else float(metric["std"])
    return float(metric), None


def result_from_multiseed_json(path: Path, cell: Cell) -> Result:
    with path.open() as f:
        data = json.load(f)
    test_mean, test_std = _metric_pair(data, "test_macro_f1")
    val_mean, val_std = _metric_pair(data, "val_macro_f1")
    return Result(
        test_mean=test_mean,
        test_std=test_std,
        n_params=PARAM_FALLBACK.get(cell),
        label="5-seed canonical",
        val_mean=val_mean,
        val_std=val_std,
    )


def result_from_optuna_json(path: Path, cell: Cell) -> Result:
    """Read either a single-trial Optuna JSON or a 5-seed Optuna rerun JSON."""
    with path.open() as f:
        data = json.load(f)

    # Corrected M2 Mel file is a multiseed rerun of the Optuna-selected config.
    if isinstance(data.get("test_macro_f1"), dict):
        test_mean, test_std = _metric_pair(data, "test_macro_f1")
        val_mean, val_std = _metric_pair(data, "val_macro_f1")
        hp = data.get("hparams", {}) or {}
        d_model = hp.get("d_model")
        num_layers = hp.get("num_layers")
        label = f"d={d_model},L={num_layers}" if d_model is not None and num_layers is not None else "Optuna 5-seed"
        return Result(test_mean=test_mean, test_std=test_std, n_params=PARAM_FALLBACK.get(cell),
                      label=label, val_mean=val_mean, val_std=val_std)

    params = data.get("best_params", {}) or {}
    d_model = params.get("d_model")
    num_layers = params.get("num_layers")
    if d_model is not None and num_layers is not None:
        label = f"d={d_model},L={num_layers}"
    else:
        label = "Optuna best"

    return Result(
        test_mean=float(data["best_test_f1"]),
        test_std=None,
        n_params=PARAM_FALLBACK.get(cell),
        label=label,
        val_mean=float(data["best_val_f1"]),
        val_std=None,
    )


def load_results(data_dir: Path | None = None) -> tuple[dict[Cell, Result | None], dict[Cell, Result | None], dict[str, Path]]:
    canonical = dict(CANONICAL_FALLBACK)
    optuna = dict(OPTUNA_FALLBACK)
    used: dict[str, Path] = {}

    # Fill the three cells that were missing in the previous figure.
    m2_mfcc_can = find_json("audio_canonical_multiseed_M2_MFCC.json", data_dir)
    if m2_mfcc_can:
        canonical[("M2", "MFCC")] = result_from_multiseed_json(m2_mfcc_can, ("M2", "MFCC"))
        used["M2 MFCC canonical"] = m2_mfcc_can

    m3_mfcc_can = find_json("attn_canonical_multiseed_M3_MFCC.json", data_dir)
    if m3_mfcc_can:
        canonical[("M3", "MFCC")] = result_from_multiseed_json(m3_mfcc_can, ("M3", "MFCC"))
        used["M3 MFCC canonical"] = m3_mfcc_can

    m2_mel_opt = find_json("audio_multiseed_optuna_M2_Mel.json", data_dir) or find_json("audio_optuna_M2_Mel.json", data_dir)
    if m2_mel_opt:
        optuna[("M2", "Mel")] = result_from_optuna_json(m2_mel_opt, ("M2", "Mel"))
        used["M2 Mel Optuna"] = m2_mel_opt

    return canonical, optuna, used

# ---------------------------------------------------------------------------
# Shared classification and plotting helpers
# ---------------------------------------------------------------------------

def classify_delta(canonical: Result | None, optuna: Result | None) -> str:
    """Classify the Optuna-vs-canonical delta using the report's |Δ|/σ rule."""
    if canonical is None or optuna is None:
        return "missing"
    delta = optuna.test_mean - canonical.test_mean
    sigma = max(canonical.test_std or 0.0, optuna.test_std or 0.0, 1e-6)
    ratio = abs(delta) / sigma
    if delta > 0 and ratio >= 1.5:
        return "genuine_gain"
    if delta < 0 and ratio >= 1.5:
        return "genuine_loss"
    return "within_noise"


def optuna_color(cls: str) -> str:
    if cls == "genuine_gain":
        return COLOR_OPT
    if cls == "genuine_loss":
        return COLOR_DEG
    if cls == "missing":
        return "white"
    return COLOR_NOISE


def label_for(cell: Cell) -> str:
    return f"{cell[0]}\n{cell[1]}"

# ---------------------------------------------------------------------------
# Figure 5 — model complexity vs F1
# ---------------------------------------------------------------------------

def fig5_complexity_vs_f1(canonical: dict[Cell, Result | None], optuna: dict[Cell, Result | None]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {"LR": "o", "M2": "^", "M3": "s"}
    colors = {"LR": COLOR_M1, "M2": COLOR_M2, "M3": COLOR_M3}

    for (model, feat), entry in canonical.items():
        if entry is None or entry.n_params is None:
            continue
        ax.scatter(entry.n_params, entry.test_mean, s=140,
                   marker=markers[model], facecolor=colors[model],
                   edgecolor="black", linewidth=0.6, zorder=3)
        if entry.test_std and entry.test_std > 0:
            ax.errorbar(entry.n_params, entry.test_mean, yerr=entry.test_std,
                        color=colors[model], capsize=3, lw=0.8,
                        alpha=0.6, zorder=2)

    # Overlay stars for any Optuna change that exceeds the noise rule.
    for cell, o in optuna.items():
        c = canonical.get(cell)
        if c is None or o is None or o.n_params is None:
            continue
        cls = classify_delta(c, o)
        if cls not in {"genuine_gain", "genuine_loss"}:
            continue
        color = COLOR_OPT if cls == "genuine_gain" else COLOR_DEG
        edge = "darkred" if cls == "genuine_gain" else "black"
        if cls == "genuine_gain":
            ax.annotate("", xy=(o.n_params, o.test_mean), xytext=(c.n_params, c.test_mean),
                        arrowprops=dict(arrowstyle="->", color=COLOR_OPT, lw=1.4))
        ax.scatter(o.n_params, o.test_mean, s=300, marker="*",
                   facecolor=color, edgecolor=edge, linewidth=0.8, zorder=5)

    ax.axhline(CHANCE_LEVEL, color=COLOR_CHANCE, ls="--", lw=1.2, zorder=1,
               label=f"Chance ≈ {CHANCE_LEVEL:.3f}")
    ax.axhspan(CHANCE_LEVEL - CHANCE_BAND, CHANCE_LEVEL + CHANCE_BAND,
               color=COLOR_CHANCE, alpha=0.2)

    ax.annotate("Best: LR Mel 0.311\n(Optuna C=0.00492)",
                xy=(1028, 0.311), xytext=(120, 0.34),
                arrowprops=dict(arrowstyle="->", color=COLOR_OPT),
                fontsize=10, color=COLOR_OPT, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Model complexity [log10(parameters)]")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Model complexity vs test macro-F1\nStars mark Optuna changes beyond seed noise", fontsize=11)
    ax.set_ylim(0.18, 0.35)
    ax.grid(alpha=0.3, zorder=0)

    legend_elems = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=COLOR_M1,
                   markeredgecolor="k", markersize=10, label="Model 1 (LR)"),
        plt.Line2D([], [], marker="^", color="w", markerfacecolor=COLOR_M2,
                   markeredgecolor="k", markersize=10, label="Model 2"),
        plt.Line2D([], [], marker="s", color="w", markerfacecolor=COLOR_M3,
                   markeredgecolor="k", markersize=10, label="Model 3"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor=COLOR_OPT,
                   markeredgecolor="darkred", markersize=15, label="Optuna gain"),
        plt.Line2D([], [], marker="*", color="w", markerfacecolor=COLOR_DEG,
                   markeredgecolor="black", markersize=15, label="Optuna loss"),
        plt.Line2D([], [], color=COLOR_CHANCE, ls="--", label="Chance"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "fig5_complexity_vs_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

# ---------------------------------------------------------------------------
# Figure 8 — Optuna summary
# ---------------------------------------------------------------------------

def _plot_trials(ax, jsonpath: Path | None, color: str, label: str) -> None:
    if not jsonpath or not jsonpath.exists():
        return
    with jsonpath.open() as f:
        data = json.load(f)
    trials = data.get("all_trials", [])
    vals = [t["val_macro_f1"] for t in trials if t.get("test_macro_f1") is not None]
    tests = [t["test_macro_f1"] for t in trials if t.get("test_macro_f1") is not None]
    if vals and tests:
        ax.scatter(vals, tests, c=color, alpha=0.5, s=35, label=f"{label} trials")


def fig8_optuna_summary(
    canonical: dict[Cell, Result | None],
    optuna: dict[Cell, Result | None],
    data_dir: Path | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: LR Mel C search if the JSON exists; otherwise leave a clear placeholder.
    lr_mel_json = latest_json(["optuna_lr_mel_*.json", "*lr*mel*optuna*.json"], data_dir)
    if lr_mel_json and lr_mel_json.exists():
        with lr_mel_json.open() as f:
            data = json.load(f)
        trials = data.get("all_trials", [])
        Cs = [t.get("C") for t in trials if t.get("C") is not None]
        inner_cv = [t.get("inner_cv") for t in trials if t.get("C") is not None]
        if Cs and inner_cv:
            axes[0].scatter(np.log10(Cs), inner_cv, c="lightblue",
                            edgecolor="navy", s=40, alpha=0.7, label="Optuna trials")
        axes[0].axvline(np.log10(0.00492), color=COLOR_OPT, lw=1.5, ls="--")
        axes[0].scatter([np.log10(0.00492)], [0.214], marker="*", s=300,
                        facecolor=COLOR_OPT, edgecolor="darkred", zorder=5,
                        label="Best C=0.00492")
        axes[0].set_xlabel("log10(C)")
        axes[0].set_ylabel("Inner LOPO-CV macro-F1")
        axes[0].set_title("LR Mel — continuous C search", fontsize=10)
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5,
                     "LR Mel Optuna JSON not found\n(panel populated when available)",
                     ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    # Panel 2: Transformer Mel val-vs-test scatter.
    m2_mel_json = find_json("audio_optuna_M2_Mel.json", data_dir) or latest_json(["optuna_search_audio_mel_*.json"], data_dir)
    m3_mel_json = latest_json(["optuna_search_attn_mel_*.json", "*attn*mel*optuna*.json"], data_dir)
    _plot_trials(axes[1], m2_mel_json, "tab:green", "M2 Mel")
    _plot_trials(axes[1], m3_mel_json, "tab:orange", "M3 Mel")

    for cell, color, name in [(("M2", "Mel"), "tab:green", "M2 Mel canonical"),
                              (("M3", "Mel"), "tab:orange", "M3 Mel canonical")]:
        c = canonical.get(cell)
        if c is not None:
            axes[1].scatter([c.val_mean if c.val_mean is not None else 0.306],
                            [c.test_mean], marker="D", s=160,
                            facecolor=color, edgecolor="black", zorder=4,
                            label=name)
    axes[1].plot([0.15, 0.40], [0.15, 0.40], "k--", lw=1, alpha=0.5, label="val = test")
    axes[1].set_xlabel("Val macro-F1")
    axes[1].set_ylabel("Test macro-F1")
    axes[1].set_title("Transformer Mel Optuna trials\nVal F1 does not guarantee test F1", fontsize=10)
    axes[1].set_xlim(0.17, 0.40)
    axes[1].set_ylim(0.15, 0.40)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="lower right")

    # Panel 3: compact canonical-vs-Optuna summary for every cell.
    x = np.arange(len(CELL_ORDER))
    width = 0.36
    canon_vals = [canonical[k].test_mean if canonical.get(k) else 0 for k in CELL_ORDER]
    opt_vals = [optuna[k].test_mean if optuna.get(k) else 0 for k in CELL_ORDER]
    classes = [classify_delta(canonical.get(k), optuna.get(k)) for k in CELL_ORDER]

    axes[2].bar(x - width / 2, canon_vals, width, label="Canonical",
                color=COLOR_CANON, edgecolor="black", lw=0.5)
    for i, (cls, value) in enumerate(zip(classes, opt_vals)):
        axes[2].bar(x[i] + width / 2, value, width, color=optuna_color(cls),
                    edgecolor="black", lw=0.5)

    for i, cell in enumerate(CELL_ORDER):
        c = canonical.get(cell)
        o = optuna.get(cell)
        if c is None or o is None:
            continue
        d = o.test_mean - c.test_mean
        cls = classes[i]
        clr = "darkgreen" if cls == "genuine_gain" else "darkred" if cls == "genuine_loss" else "dimgray"
        axes[2].text(x[i], max(canon_vals[i], opt_vals[i]) + 0.012,
                     f"{d:+.3f}", ha="center", fontsize=7,
                     color=clr, fontweight="bold")

    axes[2].axhline(CHANCE_LEVEL, color="grey", ls=":", lw=1, label="Chance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([label_for(k) for k in CELL_ORDER], fontsize=8)
    axes[2].set_ylabel("Test macro-F1")
    axes[2].set_title("Canonical vs Optuna across all 9 cells\nAll added JSON-backed cells included", fontsize=10)
    axes[2].set_ylim(0, 0.36)
    axes[2].grid(alpha=0.3, axis="y")

    legend_p3 = [
        Patch(facecolor=COLOR_CANON, edgecolor="black", label="Canonical"),
        Patch(facecolor=COLOR_OPT, edgecolor="black", label="Optuna gain"),
        Patch(facecolor=COLOR_NOISE, edgecolor="black", label="Optuna within noise"),
        Patch(facecolor=COLOR_DEG, edgecolor="black", label="Optuna loss"),
    ]
    axes[2].legend(handles=legend_p3, fontsize=7, loc="upper left")

    fig.suptitle("Optuna Bayesian hyperparameter search — canonical vs Optuna", fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig8_optuna_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

# ---------------------------------------------------------------------------
# Figure 9 — detailed canonical-vs-Optuna chart
# ---------------------------------------------------------------------------

def fig9_canonical_vs_optuna_detailed(
    canonical: dict[Cell, Result | None],
    optuna: dict[Cell, Result | None],
) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(CELL_ORDER))
    width = 0.38

    canon_vals = [canonical[k].test_mean if canonical.get(k) else 0 for k in CELL_ORDER]
    canon_std = [canonical[k].test_std or 0 if canonical.get(k) else 0 for k in CELL_ORDER]
    opt_vals = [optuna[k].test_mean if optuna.get(k) else 0 for k in CELL_ORDER]
    opt_std = [optuna[k].test_std or 0 if optuna.get(k) else 0 for k in CELL_ORDER]
    classes = [classify_delta(canonical.get(k), optuna.get(k)) for k in CELL_ORDER]

    ax.bar(x - width / 2, canon_vals, width, yerr=canon_std,
           label="Canonical (mean ± std where available)",
           color=COLOR_CANON, edgecolor="black", lw=0.6,
           error_kw={"lw": 0.7, "capsize": 3})

    for i, (cls, value, err) in enumerate(zip(classes, opt_vals, opt_std)):
        ax.bar(x[i] + width / 2, value, width, yerr=err,
               color=optuna_color(cls), edgecolor="black", lw=0.6,
               error_kw={"lw": 0.7, "capsize": 3})

    # Δ annotations for every cell where both bars exist.
    for i, cell in enumerate(CELL_ORDER):
        c = canonical.get(cell)
        o = optuna.get(cell)
        if c is None or o is None:
            ax.text(x[i], 0.015, "missing", ha="center", fontsize=7, rotation=90, color="gray")
            continue
        d = o.test_mean - c.test_mean
        cls = classes[i]
        clr = "darkgreen" if cls == "genuine_gain" else "darkred" if cls == "genuine_loss" else "dimgray"
        ax.text(x[i], max(canon_vals[i], opt_vals[i]) + 0.022,
                f"Δ = {d:+.3f}", ha="center", fontsize=8.5,
                color=clr, fontweight="bold")

    ax.axhline(CHANCE_LEVEL, color="grey", ls=":", lw=1.2,
               label=f"Chance ≈ {CHANCE_LEVEL:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(k) for k in CELL_ORDER], fontsize=10)
    ax.set_ylabel("Test macro-F1", fontsize=11)
    ax.set_title("Canonical vs Optuna for every (model, feature) combination\n"
                 "Includes M2 MFCC canonical, M2 Mel Optuna 5-seed, and M3 MFCC canonical from added JSONs",
                 fontsize=12)
    ax.set_ylim(0, 0.40)

    legend = [
        Patch(facecolor=COLOR_CANON, edgecolor="black", label="Canonical"),
        Patch(facecolor=COLOR_OPT, edgecolor="black", label="Optuna (genuine gain)"),
        Patch(facecolor=COLOR_NOISE, edgecolor="black", label="Optuna (within seed noise)"),
        Patch(facecolor=COLOR_DEG, edgecolor="black", label="Optuna (genuine loss)"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUT_DIR / "fig9_canonical_vs_optuna.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory containing result JSON files. Defaults to RESULTS_DIR, script dir, and cwd search.")
    parser.add_argument("--only", choices=["fig5", "fig8", "fig9", "all"], default="all",
                        help="Regenerate only one figure or all figures.")
    args = parser.parse_args()

    canonical, optuna, used = load_results(args.data_dir)

    print("Regenerating Optuna extension figures…")
    if used:
        print("Loaded added JSONs:")
        for name, path in used.items():
            print(f"  {name}: {path}")
    else:
        print("No added JSONs found; using fallback values where available.")

    outputs: list[Path] = []
    if args.only in {"fig5", "all"}:
        outputs.append(fig5_complexity_vs_f1(canonical, optuna))
    if args.only in {"fig8", "all"}:
        outputs.append(fig8_optuna_summary(canonical, optuna, args.data_dir))
    if args.only in {"fig9", "all"}:
        outputs.append(fig9_canonical_vs_optuna_detailed(canonical, optuna))

    print("\nWrote:")
    for out in outputs:
        print(f"  {out}")


if __name__ == "__main__":
    main()
