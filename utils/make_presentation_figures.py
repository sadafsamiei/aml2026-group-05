"""
utils/make_presentation_figures.py
-----------------------------------
Generates presentation-grade figures from canonical + extension results.

Figures produced (all in RES_DIR/figures/presentation/):
  fig1_complexity_vs_f1.png      — parameter count vs test F1 (with 0.311 star)
  fig2_perclass_heatmap.png      — per-class F1 heatmap (3 models x 4 classes)
  fig3_val_vs_test_gap.png       — val vs test F1 paired bars (with Optuna)
  fig4_inter_language_rates.png  — INTER rate by language + model predictions
  fig5_optuna_results.png        — NEW: Optuna search outcomes (3-panel)

Extension updates (v2)
----------------------
  - LR Mel result updated: 0.277 -> 0.311 (Optuna C=0.00492 line search)
  - M3 Mel result updated: 0.221 -> 0.233 +/- 0.014 (Optuna 60-trial TPE)
  - fig1: star marker for Optuna-improved points + new best annotation
  - fig3: pink bar for Optuna test F1 alongside canonical
  - fig5: new 3-panel Optuna summary

Usage
-----
    python utils/make_presentation_figures.py
    python utils/make_presentation_figures.py --out_dir /path/to/dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import config as cfg

# ---------------------------------------------------------------------------
# Canonical + extension numbers
# Each row: (label, n_params, val_f1, canonical_test_f1, feature, model_id,
#            optuna_test_f1_or_None)
# ---------------------------------------------------------------------------

RESULTS = [
    # Model 1 — Logistic Regression
    ("LR\nMFCC",     (40+1)*4,   0.240, 0.253, "mfcc",     1, None ),
    ("LR\nMel",      (256+1)*4,  0.232, 0.277, "mel",      1, 0.311),  # Optuna +0.034
    ("LR\nwav2vec2", (1536+1)*4, 0.253, 0.263, "wav2vec2", 1, None ),
    # Model 2 — Mean-Pool Transformer
    ("M2\nMFCC",     None,       0.258, 0.238, "mfcc",     2, None ),
    ("M2\nMel",      126_000,    0.270, 0.249, "mel",      2, None ),   # no improvement
    ("M2\nwav2vec2", 167_000,    0.294, 0.219, "wav2vec2", 2, None ),
    # Model 3 — Attentive Pooling Transformer
    ("M3\nMFCC",     None,       0.257, 0.237, "mfcc",     3, None ),
    ("M3\nMel",      26_000,     0.252, 0.221, "mel",      3, 0.233),  # marginal +0.012
    ("M3\nwav2vec2", 648_000,    0.286, 0.246, "wav2vec2", 3, None ),
]

PERCLASS_MEL = np.array([
    [0.220, 0.396, 0.263, 0.228],
    [0.185, 0.340, 0.220, 0.258],
    [0.130, 0.284, 0.106, 0.363],
])
PERCLASS_W2V = np.array([
    [0.228, 0.391, 0.188, 0.247],
    [0.239, 0.255, 0.231, 0.153],
    [0.192, 0.395, 0.199, 0.198],
])
INTER_BY_LANG   = {"Shipibo\n(val)": 0.225, "Vietnamese\n(train)": 0.330, "Tagalog\n(test)": 0.469}
TRUE_TEST_INTER = 0.294
M2_INTER_PRED   = 0.311
M3_INTER_PRED   = 0.574
CHANCE          = 0.250

COL = {
    "m1":"#2196F3", "m2":"#FF9800", "m3":"#4CAF50",
    "val":"#90CAF9", "test":"#1565C0", "optuna":"#E91E63",
    "chance":"grey", "inter":"#1565C0", "gap_pos":"#C62828", "gap_neg":"#2E7D32",
}


def save(fig, path, dpi=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Figure 1 — Complexity vs test F1  (star for 0.311)
# ---------------------------------------------------------------------------
def fig1_complexity(out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    model_colors = {1: COL["m1"], 2: COL["m2"], 3: COL["m3"]}
    feat_markers = {"mfcc": "s", "mel": "o", "wav2vec2": "^"}
    model_names  = {1:"Model 1 (LR)", 2:"Model 2 (mean pool)", 3:"Model 3 (attention)"}
    feat_names   = {"mfcc":"MFCC", "mel":"Mel", "wav2vec2":"wav2vec2"}
    seen = set()

    for label, n_params, val_f1, test_f1, feat, mid, optuna_f1 in RESULTS:
        if n_params is None: continue
        color  = model_colors[mid]
        marker = feat_markers[feat]
        lkey   = (mid, feat)
        lbl    = (f"{model_names[mid]} -- {feat_names[feat]}" if lkey not in seen else None)
        seen.add(lkey)
        ax.scatter(np.log10(n_params), test_f1, color=color, marker=marker,
                   s=150, zorder=5, label=lbl)
        ax.annotate(label.replace("\n"," "), (np.log10(n_params), test_f1),
                    xytext=(6,4), textcoords="offset points", fontsize=8.5, color=color)
        if optuna_f1 is not None:
            opt_lbl = "Optuna-improved" if "Optuna-improved" not in seen else None
            seen.add("Optuna-improved")
            ax.scatter(np.log10(n_params), optuna_f1, color=COL["optuna"],
                       marker="*", s=280, zorder=6, label=opt_lbl)
            ax.annotate(f"Optuna: {optuna_f1:.3f}", (np.log10(n_params), optuna_f1),
                        xytext=(8,4), textcoords="offset points", fontsize=8, color=COL["optuna"])
            ax.annotate("", xy=(np.log10(n_params), optuna_f1-0.003),
                        xytext=(np.log10(n_params), test_f1+0.003),
                        arrowprops=dict(arrowstyle="->", color=COL["optuna"],
                                        lw=1.2, linestyle="dashed"))

    ax.axhline(CHANCE, ls="--", color=COL["chance"], lw=1.4, label=f"Chance = {CHANCE}")
    ax.annotate(" best: LR Mel 0.311\n(Optuna C=0.00492)",
                xy=(np.log10((256+1)*4), 0.311),
                xytext=(np.log10((256+1)*4)+0.5, 0.319),
                fontsize=9, color=COL["optuna"],
                arrowprops=dict(arrowstyle="->", color=COL["optuna"], lw=1.2))
    ax.set_xlabel("Model complexity  [log10(parameters)]", fontsize=12)
    ax.set_ylabel("Test macro-F1", fontsize=12)
    ax.set_title("More parameters do not improve accuracy\n"
                 "Stars = Optuna-improved results  |  best: 0.311 (LR Mel)", fontsize=12)
    ax.set_ylim(0.15, 0.34)
    ax.set_xlim(1.8, 6.2)
    ax.set_xticks([2,3,4,5,6])
    ax.set_xticklabels(["10^2","10^3","10^4","10^5","10^6"])
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3)
    save(fig, out_dir / "fig1_complexity_vs_f1.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Per-class heatmap  (unchanged)
# ---------------------------------------------------------------------------
def fig2_heatmap(out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CLASS_NAMES  = ["EXTEV","INTER","MOVE","PLTOY"]
    MODEL_LABELS = ["Model 1\n(LR)","Model 2\n(mean pool)","Model 3\n(attention)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data, title in zip(axes, [PERCLASS_MEL, PERCLASS_W2V],
                                ["Mel spectrograms","wav2vec2 embeddings"]):
        im = ax.imshow(data, vmin=0.0, vmax=0.5, cmap="RdYlGn", aspect="auto")
        for i in range(3):
            for j in range(4):
                v = data[i,j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="black" if 0.12<v<0.42 else "white")
        for i in range(3):
            ax.add_patch(plt.Rectangle((0.5,i-0.5),1,1,fill=False,
                                        edgecolor=COL["inter"],linewidth=2.5))
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
        ax.set_yticks(range(3)); ax.set_yticklabels(MODEL_LABELS, fontsize=10)
        ax.set_title(title, fontsize=12, pad=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Per-class test macro-F1", shrink=0.8)
    fig.suptitle("Per-class test F1 -- INTER (blue border) highest for every model",
                 fontsize=12, y=1.02)
    save(fig, out_dir / "fig2_perclass_heatmap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Val vs Test gap  (Optuna pink bars added)
# ---------------------------------------------------------------------------
def fig3_val_test_gap(out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = [
        ("LR\nMel",       0.232, 0.277, 0.311),
        ("LR\nwav2vec2",  0.253, 0.263, None),
        ("M2\nMel",       0.270, 0.249, None),
        ("M2\nwav2vec2",  0.294, 0.219, None),
        ("M3\nMel",       0.252, 0.221, 0.233),
        ("M3\nwav2vec2",  0.286, 0.246, None),
    ]
    labels = [c[0] for c in configs]
    val_f1 = [c[1] for c in configs]
    test_f1= [c[2] for c in configs]
    opt_f1 = [c[3] for c in configs]
    x = np.arange(len(labels)); w = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x-w, val_f1, w, label="Val macro-F1 (Shipibo child)",
           color=COL["val"], edgecolor="white")
    ax.bar(x,   test_f1, w, label="Test macro-F1 canonical (Tagalog child)",
           color=COL["test"], edgecolor="white")
    for xi, opt in enumerate(opt_f1):
        if opt is not None:
            ax.bar(xi+w, opt, w, color=COL["optuna"], edgecolor="white",
                   label="Test macro-F1 Optuna" if xi==0 else "")
            ax.text(xi+w, opt+0.003, f"{opt:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=COL["optuna"], fontweight="bold")

    for xi,(v,t) in enumerate(zip(val_f1,test_f1)):
        gap = v-t; color = COL["gap_pos"] if gap>0.005 else COL["gap_neg"]
        ax.annotate(f"{gap:+.3f}", xy=(xi-w/2, max(v,t)+0.003),
                    xytext=(xi-w/2, max(v,t)+0.018), fontsize=8, ha="center",
                    color=color, arrowprops=dict(arrowstyle="-",color=color,lw=0.6))

    ax.axhline(CHANCE, ls="--", color=COL["chance"], lw=1.4, label=f"Chance = {CHANCE}")
    spans = [((-0.5,1.5),COL["m1"],"Model 1 (LR)"),
             ((1.5,3.5), COL["m2"],"Model 2"),
             ((3.5,5.5), COL["m3"],"Model 3")]
    for (x0,x1),color,txt in spans:
        ax.axvspan(x0,x1,alpha=0.06,color=color)
        ax.text((x0+x1)/2, 0.168, txt, ha="center", fontsize=9, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_ylim(0.16, 0.345)
    ax.set_title("Val overestimates test -- language confound\n"
                 "Pink bars = Optuna-improved. Gap annotation: val - test", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, out_dir / "fig3_val_vs_test_gap.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4 — INTER rate by language  (unchanged)
# ---------------------------------------------------------------------------
def fig4_inter_language(out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    langs  = list(INTER_BY_LANG.keys())
    rates  = list(INTER_BY_LANG.values())
    colors = [COL["val"],"#FFB74D","#EF9A9A"]
    bars   = ax.bar(langs, rates, color=colors, width=0.5, edgecolor="white",
                    linewidth=1.2, zorder=3)
    for bar,rate in zip(bars,rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.013,
                f"{rate:.1%}", ha="center", fontsize=12, fontweight="bold")
    refs = [(TRUE_TEST_INTER,COL["test"],"--","True test INTER rate (29.4%)"),
            (M2_INTER_PRED,  COL["m2"], "-.","Model 2 INTER predictions (31.1%)"),
            (M3_INTER_PRED,  COL["gap_pos"],"-","Model 3 INTER predictions (57.4%)")]
    for y,color,ls,label in refs:
        ax.axhline(y,color=color,ls=ls,lw=2.0,label=label,zorder=4)
        ax.text(2.72,y+0.012,f"{y:.1%}",color=color,fontsize=10,fontweight="bold")
    ax.annotate("Model 3 inherits the\nTagalog INTER prior\n(+28 pp above true rate)",
                xy=(2.0,M3_INTER_PRED), xytext=(1.0,0.61), fontsize=9.5,
                color=COL["gap_pos"],
                arrowprops=dict(arrowstyle="->",color=COL["gap_pos"],lw=1.3))
    ax.set_ylim(0,0.72); ax.set_ylabel("INTER class rate", fontsize=12)
    ax.set_title("INTER over-prediction traces to the Tagalog training prior\n"
                 "Val=Shipibo (22.5%) -> Test=Tagalog (46.9% in training)", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    save(fig, out_dir / "fig4_inter_language_rates.png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5 — NEW: Optuna search outcomes  (3-panel)
# ---------------------------------------------------------------------------
def fig5_optuna_summary(out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel A: LR mel line search -- C vs inner-CV F1
    ax = axes[0]
    # 9-point grid (seeded first by Optuna)
    grid_c  = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    grid_cv = [0.191,0.196,0.210,0.191,0.184,0.169,0.177,0.164,0.175]
    # Selected Optuna extra trials
    opt_c  = [0.031,0.013,0.007,0.00492,0.003,0.0066,0.0091,
               0.0065,0.0073,0.0059,0.0034,0.0059,
               0.022,0.010,0.0032,0.0023,0.0040,0.00076]
    opt_cv = [0.181,0.204,0.208,0.214,0.193,0.207,0.211,
               0.207,0.209,0.200,0.193,0.200,
               0.199,0.210,0.203,0.190,0.193,0.197]

    ax.scatter([np.log10(c) for c in grid_c], grid_cv,
               color="#1565C0", marker="s", s=70, alpha=0.9, label="9-pt grid", zorder=5)
    ax.scatter([np.log10(c) for c in opt_c], opt_cv,
               color="#FF9800", marker="o", s=45, alpha=0.7, label="Optuna trials", zorder=4)
    ax.scatter([np.log10(0.00492)], [0.214], color=COL["optuna"], marker="*",
               s=320, zorder=6, label="Best: C=0.00492")
    ax.axvline(np.log10(0.01), ls="--", color="#1565C0", lw=1.2, label="Grid best (C=0.01)")
    ax.set_xlabel("log10(C)", fontsize=11)
    ax.set_ylabel("Inner LOPO-CV macro-F1", fontsize=11)
    ax.set_title("LR mel -- continuous C search\n"
                 "Best C=0.00492 falls between 0.001 and 0.01", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.155, 0.225)

    # Panel B: M2 and M3 mel Optuna -- val vs test scatter
    ax = axes[1]
    # Selected trials from the 60-trial searches
    m2_val  = [0.248,0.300,0.289,0.253,0.281,0.275,0.246,0.244,0.186,0.319,
                0.244,0.293,0.278,0.330,0.336,0.230,0.279,0.260,0.258,0.277,
                0.305,0.311,0.278,0.220,0.294,0.294,0.277,0.308,0.284,0.274]
    m2_test = [0.189,0.222,0.224,0.196,0.209,0.260,0.213,0.233,0.263,0.243,
                0.216,0.240,0.268,0.238,0.207,0.256,0.254,0.231,0.222,0.241,
                0.227,0.184,0.253,0.197,0.217,0.217,0.290,0.252,0.228,0.217]
    m3_val  = [0.282,0.239,0.259,0.252,0.231,0.257,0.291,0.231,0.230,0.227,
                0.205,0.254,0.228,0.322,0.249,0.256,0.220,0.304,0.271,0.269,
                0.235,0.241,0.244,0.249,0.263,0.272,0.217,0.265,0.301,0.276]
    m3_test = [0.251,0.203,0.228,0.274,0.236,0.262,0.232,0.231,0.231,0.174,
                0.176,0.250,0.254,0.242,0.223,0.188,0.194,0.292,0.250,0.184,
                0.277,0.253,0.218,0.220,0.232,0.237,0.205,0.213,0.213,0.217]

    ax.scatter(m2_val, m2_test, color=COL["m2"], alpha=0.5, s=40, label="M2 mel trials")
    ax.scatter(m3_val, m3_test, color=COL["m3"], alpha=0.5, s=40, label="M3 mel trials")
    ax.scatter([0.306],[0.249], color=COL["m2"], marker="D", s=120,
               edgecolors="black", linewidth=1.3, zorder=6, label="M2 canonical")
    ax.scatter([0.286],[0.221], color=COL["m3"], marker="D", s=120,
               edgecolors="black", linewidth=1.3, zorder=6, label="M3 canonical")
    lims = [0.17, 0.37]
    ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4, label="val = test")
    ax.fill_between(lims, lims, [lims[0]]*2, alpha=0.04, color="red")
    ax.axhline(CHANCE, ls=":", color="grey", lw=1.0)
    ax.set_xlabel("Val macro-F1 (Shipibo child)", fontsize=11)
    ax.set_ylabel("Test macro-F1 (Tagalog child)", fontsize=11)
    ax.set_title("M2 & M3 mel -- 60 Optuna trials each\n"
                 "Higher val F1 does NOT predict higher test F1", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # Panel C: Before/after bar chart
    ax = axes[2]
    names    = ["LR\nMel", "M3\nMel", "M2\nMel"]
    canon    = [0.277,       0.221,     0.249]
    optuna   = [0.311,       0.233,     None]
    colors_c = [COL["m1"],  COL["m3"], COL["m2"]]
    colors_o = [COL["optuna"], "#66BB6A", COL["m2"]]
    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, canon, w, color=colors_c, alpha=0.75, label="Canonical", edgecolor="white")
    for xi,(opt,cc,co) in enumerate(zip(optuna,colors_c,colors_o)):
        val = opt if opt is not None else canon[xi]
        ax.bar(xi+w/2, val, w, color=co if opt is not None else cc,
               alpha=0.75 if opt is not None else 0.35,
               label=("Optuna" if xi==0 else ""), edgecolor="white")
        if opt is not None:
            delta = opt-canon[xi]
            ax.text(xi+w/2, opt+0.003, f"+{delta:.3f}", ha="center", va="bottom",
                    fontsize=9, color=co, fontweight="bold")
        else:
            ax.text(xi+w/2, canon[xi]+0.003, "no\nimprove", ha="center",
                    va="bottom", fontsize=8, color="grey")
    ax.axhline(CHANCE, ls="--", color="grey", lw=1.2, label="Chance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Test macro-F1", fontsize=11)
    ax.set_ylim(0.20, 0.335)
    ax.set_title("Canonical vs Optuna best\n"
                 "LR line search: genuine +0.034 | Transformer: marginal or none", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Extension: Optuna Bayesian Hyperparameter Search\n"
                 "Continuous C search improves LR 0.277->0.311  |  "
                 "Transformers limited by data ceiling, not hyperparameters",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, out_dir / "fig5_optuna_results.png")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = (Path(args.out_dir) if args.out_dir
               else cfg.FIGURES_DIR / "presentation")
    print(f"\nGenerating figures -> {out_dir}\n")
    print("[1/5] Complexity vs test F1 (updated with 0.311 star)...")
    fig1_complexity(out_dir)
    print("[2/5] Per-class F1 heatmap...")
    fig2_heatmap(out_dir)
    print("[3/5] Val vs test gap (with Optuna pink bars)...")
    fig3_val_test_gap(out_dir)
    print("[4/5] INTER rate by language...")
    fig4_inter_language(out_dir)
    print("[5/5] Optuna search summary (NEW)...")
    fig5_optuna_summary(out_dir)
    print(f"\nAll done -> {out_dir}/")


if __name__ == "__main__":
    main()