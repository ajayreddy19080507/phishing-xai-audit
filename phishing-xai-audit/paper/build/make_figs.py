"""Figures for the paper. All numbers come from results/main_run_r6 (main run) and results/controls_ctl (controls)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("PAPER_DATA") or os.path.normpath(os.path.join(W, "..", "..", "results"))
R6 = f"{DATA}/main_run_r6"; CT = f"{DATA}/controls_ctl"; OUT = f"{W}/figs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "mathtext.fontset": "stix",
})
# Okabe-Ito colours; every series also gets its own marker / line style / hatch so the
# figures stay readable in grayscale.
C = dict(LIME="#0072B2", SHAP="#E69F00", IG="#009E73", HYB="#D55E00", RAND="#555555",
         LR="#999999", DIS="#CC79A7", ROB="#0072B2")
MOD = {"distilroberta-base": "DistilRoBERTa", "roberta-base": "RoBERTa-base"}
rng = np.random.RandomState(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    m = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------------ Fig. 1 pipeline
def fig1():
    meta = json.load(open(f"{R6}/meta.json"))["audit"]
    boxes = [
        ("Kaggle Phishing Email corpus", f"{meta['n_raw']:,} emails ({100*meta['n_phish']/meta['n_raw']:.1f}% phishing)", 0),
        ("Duplicate audit and removal", f"{meta['n_rows_in_duplicate_groups']:,} rows in duplicate groups; {meta['n_dedup']:,} kept", 0),
        ("Stratified 70/10/20 split", "3 seeds; raw and deduplicated protocols", 0),
        ("Detectors", "TF-IDF LR / SVM / RF, DistilRoBERTa, RoBERTa-base", 0),
        ("Explainers (word level)", "LIME, SHAP, Integrated Gradients, LIME–IG hybrid", 1),
        ("Evaluation and controls", "Comp./Suff., agreement, IG/SHAP controls, obfuscation, transfer", 2),
    ]
    fig, ax = plt.subplots(figsize=(3.45, 3.55)); ax.set_xlim(0, 10); ax.set_ylim(0, 6 * 1.0 + 0.2); ax.axis("off")
    h, gap = 0.78, 0.22
    face = ["#EEF2FA", "#FDEDE8", "#EAF5EE"]
    for i, (t, s, f) in enumerate(boxes):
        y = 6.1 - (i + 1) * (h + gap) + gap
        ax.add_patch(FancyBboxPatch((0.15, y), 9.7, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=face[f], ec="#4A5568", lw=0.7))
        ax.text(5, y + h * 0.68, t, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(5, y + h * (0.28 if "\n" not in s else 0.30), s, ha="center", va="center", fontsize=6.6, color="#333333",
                linespacing=1.05)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(5, y - gap + 0.02), xytext=(5, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#4A5568"))
    fig.savefig(f"{OUT}/fig1_pipeline.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


# ------------------------------------------------------------------ Fig. 2 deletion curves
def fig2():
    cur = json.load(open(f"{CT}/curves_export.json"))
    series = [("LIME", "LIME", C["LIME"], "o", "-"), ("SHAP", "SHAP", C["SHAP"], "s", "--"),
              ("IG", "IG (pad baseline)", C["IG"], "^", "-."), ("LITA-absmax(a=.5)", "Hybrid, α = 0.5", C["HYB"], "D", "-"),
              ("Random", "Random order", C["RAND"], "x", ":")]
    fig, axs = plt.subplots(2, 2, figsize=(3.45, 3.55), sharex=True)
    ks = np.arange(1, 11)
    for j, m in enumerate(MOD):
        for i, (key, ylab) in enumerate([("rem", "Comprehensiveness"), ("keep", "Sufficiency")]):
            ax = axs[i, j]
            for name, lab, col, mk, ls in series:
                d = cur[f"{m}|{name}"]; mu = np.array(d[f"{key}_mean"]); sd = np.array(d[f"{key}_sd"]); n = d["n"]
                ax.plot(ks, mu, color=col, marker=mk, ls=ls, lw=1.0, ms=3, label=lab, markevery=1)
                ax.fill_between(ks, mu - 1.96 * sd / np.sqrt(n), mu + 1.96 * sd / np.sqrt(n), color=col, alpha=0.10, lw=0)
            if i == 0: ax.set_title(MOD[m])
            if j == 0: ax.set_ylabel(ylab + (" ↑" if key == "rem" else " ↓"))
            if i == 1: ax.set_xlabel("k (top-ranked words)")
            ax.set_xticks([1, 3, 5, 7, 9, 10] if False else [2, 4, 6, 8, 10]); ax.grid(alpha=0.25, lw=0.4)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), columnspacing=0.9,
               handlelength=1.8, fontsize=6.6)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    fig.savefig(f"{OUT}/fig2_curves.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


# ------------------------------------------------------------------ Fig. 3 IG / SHAP controls
def fig3():
    P = pd.read_csv(f"{CT}/ig_controls_per_email.csv")
    order = [("Random", "Random order"), ("IG_prob30 (paper)", "IG: prob., 30 steps, pad (default)"),
             ("IG_prob300", "IG: prob., 300 steps, pad"), ("IG_logit100", "IG: logit, 100 steps, pad"),
             ("GradxInput_logit", "Gradient×Input: logit"), ("IG_logit100_maskbase", "IG: logit, 100 steps, mask base"),
             ("LIME", "LIME")]
    order = order[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    off = {"distilroberta-base": 0.16, "roberta-base": -0.16}
    mk = {"distilroberta-base": ("o", C["DIS"]), "roberta-base": ("s", C["ROB"])}
    for m in MOD:
        for yi, (v, lab) in enumerate(order):
            x = P[(P.model == m) & (P.variant == v)].sort_values("email").comp.values
            lo, hi = boot_ci(x)
            ax.errorbar(x.mean(), yi + off[m], xerr=[[x.mean() - lo], [hi - x.mean()]], fmt=mk[m][0], color=mk[m][1],
                        ms=3.6, lw=0.9, capsize=1.6, label=MOD[m] if yi == 0 else None, mfc=mk[m][1] if m == "roberta-base" else "white")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order])
    ax.set_xlabel("Comprehensiveness, mean over k = 1–10 (95% bootstrap CI)")
    ax.axvline(0, color="#888", lw=0.5); ax.grid(axis="x", alpha=0.25, lw=0.4); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#ccc")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_ig_controls.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


# ------------------------------------------------------------------ Fig. 4 robustness
def fig4():
    R = pd.read_csv(f"{R6}/robustness.csv"); RC = pd.read_csv(f"{CT}/robust_controls.csv")
    models = [("TFIDF-LR", "TF-IDF LR", "#999999", "//"), ("distilroberta-base", "DistilRoBERTa", C["DIS"], "\\\\"),
              ("roberta-base", "RoBERTa-base", C["ROB"], "..")]
    atk = [("none", "None"), ("homoglyph", "Homoglyph"), ("typo", "Typo"), ("zero_width", "Zero-width")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.45, 3.7), gridspec_kw=dict(height_ratios=[1, 1.05]))
    w = 0.26
    for i, (k, lab, col, hat) in enumerate(models):
        v = [100 * R[(R.model == k) & (R.attack == a)].fpr.iloc[0] for a, _ in atk]
        b = a1.bar(np.arange(4) + (i - 1) * w, v, w, color=col, hatch=hat, edgecolor="white" if False else "#222", lw=0.4, label=lab)
        for x, y in zip(np.arange(4) + (i - 1) * w, v): a1.text(x, y + 0.12, f"{y:.1f}", ha="center", va="bottom", fontsize=6)
    a1.set_xticks(range(4)); a1.set_xticklabels([l for _, l in atk]); a1.set_ylabel("False-positive rate (%)")
    a1.set_ylim(0, 10.2); a1.legend(frameon=False, ncol=3, loc="upper left", fontsize=6.4, handlelength=1.4, columnspacing=0.8)
    a1.set_title("(a) all 3,499 dedup seed-42 test emails, 256 tokens", fontsize=7, loc="left")
    # (b) homoglyph controls, transformers only
    groups = [("Clean,\n256 tok.", lambda m: RC[(RC.model == m) & (RC.attack == "none") & (RC.max_len == 256)].fpr.iloc[0]),
              ("Homoglyph,\n256 tok.", lambda m: RC[(RC.model == m) & (RC.attack == "homoglyph") & (RC.max_len == 256)].fpr.iloc[0]),
              ("Homoglyph,\n512 tok.", lambda m: RC[(RC.model == m) & (RC.attack == "homoglyph") & (RC.max_len == 512)].fpr.iloc[0]),
              ("Short emails:\nclean", lambda m: RC[(RC.model == m) & (RC.attack == "homoglyph_fits256_only")].fpr_clean_same_subset.iloc[0]),
              ("Short emails:\nhomoglyph", lambda m: RC[(RC.model == m) & (RC.attack == "homoglyph_fits256_only")].fpr.iloc[0])]
    w = 0.36
    for i, (k, lab, col, hat) in enumerate(models[1:]):
        v = [100 * g(k) for _, g in groups]
        a2.bar(np.arange(5) + (i - 0.5) * w, v, w, color=col, hatch=hat, edgecolor="#222", lw=0.4, label=lab)
        for x, y in zip(np.arange(5) + (i - 0.5) * w, v): a2.text(x, y + 0.12, f"{y:.1f}", ha="center", va="bottom", fontsize=6)
    a2.set_xticks(range(5)); a2.set_xticklabels([l for l, _ in groups], fontsize=6.2); a2.set_ylabel("False-positive rate (%)")
    a2.set_ylim(0, 10.2); a2.set_title("(b) truncation controls (n = 3,499; short emails n = 1,609)", fontsize=7, loc="left")
    fig.tight_layout(h_pad=0.8); fig.savefig(f"{OUT}/fig4_robustness.png", dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4): f()
    print(sorted(os.listdir(OUT)))
