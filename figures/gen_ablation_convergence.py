#!/usr/bin/env python3
"""
Generate convergence_ablation.pdf — ablation convergence curves for VLDB paper.
Shows best-SIEVE-Score-so-far vs iteration for all 4 mechanistic ablation conditions.
The key story: w/o History stagnates completely after iteration ~5, gaining
only 0.1 Score points over 45 remaining iterations and reaching Score>=299 in 0/3 seeds.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
BASE = "/work/hdd/bdjd/vdms_workflow/semantic_vdms/FINAL_RESULTS/hico_det/results"

CONDITIONS = {
    "LLM Agent (Full)": ["llm_seed42", "llm_seed99", "llm_seed200"],
    "w/o History cond.": [
        "ablation_no_history_seed42",
        "ablation_no_history_seed99",
        "ablation_no_history_seed200",
    ],
    "w/o Phase struct.": [
        "ablation_no_phases_seed42",
        "ablation_no_phases_seed99",
        "ablation_no_phases_seed200",
    ],
    "w/o Both": [
        "ablation_no_history_no_phases_seed42",
        "ablation_no_history_no_phases_seed99",
        "ablation_no_history_no_phases_seed200",
    ],
}

def load_bsf_curves(fnames):
    """Return (n_seeds, n_iters) array of best-so-far Score curves."""
    curves = []
    for f in fnames:
        data = json.load(open(f"{BASE}/{f}.json"))
        raw = [r["score"] or 0.0 for r in data["results"]]
        bsf, best = [], 0.0
        for s in raw:
            best = max(best, s)
            bsf.append(best)
        curves.append(bsf)
    return np.array(curves)  # (n_seeds, n_iters)

curves = {label: load_bsf_curves(files) for label, files in CONDITIONS.items()}
N = curves["LLM Agent (Full)"].shape[1]  # 50 iterations
iters = np.arange(1, N + 1)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLES = {
    "LLM Agent (Full)":   dict(color="#1a1a1a", lw=2.2, ls="-",   zorder=5),
    "w/o History cond.":  dict(color="#d62728", lw=1.8, ls="--",  zorder=4),
    "w/o Phase struct.":  dict(color="#1f77b4", lw=1.8, ls="-.",  zorder=3),
    "w/o Both":           dict(color="#ff7f0e", lw=1.8, ls=":",   zorder=2),
}
BAND_ALPHA = 0.12

# Phase boundary annotations (for full agent with N=50)
T_EXP  = 20   # end of Exploration
T_EXPL = 38   # end of Exploitation

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.9))

# Shade phase regions (light background bands)
ax.axvspan(1,        T_EXP,  alpha=0.03, color="#4daf4a", lw=0)
ax.axvspan(T_EXP+1,  T_EXPL, alpha=0.03, color="#377eb8", lw=0)
ax.axvspan(T_EXPL+1, N,      alpha=0.03, color="#984ea3", lw=0)

# Phase boundary lines (faint vertical)
for t in [T_EXP, T_EXPL]:
    ax.axvline(t + 0.5, color="#aaaaaa", lw=0.7, ls="--", zorder=1)

for label, arr in curves.items():
    mean = arr.mean(0)
    std  = arr.std(0)
    st   = STYLES[label]
    ax.plot(iters, mean, label=label, **st)
    ax.fill_between(iters, mean - std, mean + std,
                    color=st["color"], alpha=BAND_ALPHA, linewidth=0)

# ── Stagnation annotation for w/o History ─────────────────────────────────────
arr_nh = curves["w/o History cond."]
mean_nh = arr_nh.mean(0)
# Mark stagnation onset at iteration 5 (where it plateaus)
stag_iter = 5
stag_y    = mean_nh[stag_iter - 1]
ax.annotate(
    "stagnates\n(+0.1 pts\nover iters 5–50)",
    xy=(stag_iter, stag_y),
    xytext=(16, 291.5),
    fontsize=5.5,
    color="#d62728",
    arrowprops=dict(
        arrowstyle="->",
        color="#d62728",
        lw=0.8,
        connectionstyle="arc3,rad=0.25",
    ),
    ha="center",
)

# ── Phase labels (top of figure) ─────────────────────────────────────────────
phase_y = 302.0
ax.text(10.5,       phase_y, "Exploration",   ha="center", va="bottom",
        fontsize=5.5, color="#4daf4a", style="italic")
ax.text(29,         phase_y, "Exploitation",  ha="center", va="bottom",
        fontsize=5.5, color="#377eb8", style="italic")
ax.text(44.5,       phase_y, "Fine-tuning",   ha="center", va="bottom",
        fontsize=5.5, color="#984ea3", style="italic")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Iteration", fontsize=8, fontweight="bold")
ax.set_ylabel("Best SIEVE Score (QPS)", fontsize=8, fontweight="bold")
ax.set_xlim(1, N)
ax.set_ylim(80, 305)

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))
ax.grid(True, which="major", alpha=0.30, linewidth=0.6)
ax.grid(True, which="minor", alpha=0.12, linewidth=0.3)

# Reference line at Score=299
ax.axhline(299, color="#aaaaaa", lw=0.7, ls=":", zorder=1)
ax.text(N + 0.3, 299, "299", va="center", fontsize=5.5, color="#888888")

ax.legend(
    fontsize=6,
    loc="lower right",
    framealpha=0.92,
    edgecolor="#cccccc",
    handlelength=2.2,
    labelspacing=0.35,
)

plt.tight_layout(pad=0.6)

OUT = "/work/hdd/bdjd/69baaca9e33b6a3148342511/figures/convergence_ablation.pdf"
plt.savefig(OUT, format="pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")

# ── Print table values for LaTeX ─────────────────────────────────────────────
print("\n--- Table values ---")
for label, arr in curves.items():
    mean10 = arr.mean(0)[9]
    mean50 = arr.mean(0)[49]
    std50  = arr.std(0)[49]
    gain   = mean50 - mean10
    n_above = int((arr[:, 49] >= 299).sum())
    delta_pct = (mean50 - 300.29) / 300.29 * 100
    print(f"{label:<22}: Score@10={mean10:.1f}  Score@50={mean50:.1f}"
          f"  sigma={std50:.1f}  gain={gain:+.1f}  seeds>=299={n_above}/3"
          f"  delta={delta_pct:+.1f}%")
