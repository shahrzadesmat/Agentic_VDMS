#!/usr/bin/env python3
"""
Generate convergence_ablation.pdf — ablation convergence curves for VLDB paper.
Shows best-SIEVE-Score-so-far vs iteration for all 4 mechanistic ablation conditions.
Y-axis zoomed to 274-305 to show the critical post-iteration-5 behaviour.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    curves = []
    for f in fnames:
        data = json.load(open(f"{BASE}/{f}.json"))
        raw = [r["score"] or 0.0 for r in data["results"]]
        bsf, best = [], 0.0
        for s in raw:
            best = max(best, s)
            bsf.append(best)
        curves.append(bsf)
    return np.array(curves)

curves = {label: load_bsf_curves(files) for label, files in CONDITIONS.items()}
N = curves["LLM Agent (Full)"].shape[1]
iters = np.arange(1, N + 1)

# ── Style ─────────────────────────────────────────────────────────────────────
STYLES = {
    "LLM Agent (Full)":  dict(color="#1a1a1a", lw=2.0, ls="-",  zorder=5),
    "w/o History cond.": dict(color="#d62728", lw=1.6, ls="--", zorder=4),
    "w/o Phase struct.": dict(color="#1f77b4", lw=1.6, ls="-.", zorder=3),
    "w/o Both":          dict(color="#ff7f0e", lw=1.6, ls=":",  zorder=2),
}
BAND_ALPHA = 0.12

T_EXP  = 20
T_EXPL = 38

Y_LO, Y_HI = 274, 305

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.3, 1.9))

# Phase shading (very subtle)
ax.axvspan(1,        T_EXP,  alpha=0.04, color="#4daf4a", lw=0)
ax.axvspan(T_EXP+1,  T_EXPL, alpha=0.04, color="#377eb8", lw=0)
ax.axvspan(T_EXPL+1, N,      alpha=0.04, color="#984ea3", lw=0)

# Phase boundary lines (faint vertical)
for t in [T_EXP, T_EXPL]:
    ax.axvline(t + 0.5, color="#bbbbbb", lw=0.6, ls="--", zorder=1)

# Phase labels: placed near top, x-centred in each region — no overlap risk
for x, label, col in [
        (10.5, "Expl.", "#4daf4a"),
        (29,   "Exploit.", "#377eb8"),
        (44.5, "Fine-t.", "#984ea3")]:
    ax.text(x, Y_HI - 0.4, label, ha="center", va="top",
            fontsize=4.5, color=col, style="italic")

for label, arr in curves.items():
    mean = arr.mean(0)
    st   = STYLES[label]
    ax.plot(iters, mean, label=label, **st, clip_on=True)

# ── Stagnation annotation: arrow from plateau into clear bottom space ─────────
arr_nh  = curves["w/o History cond."]
mean_nh = arr_nh.mean(0)
stag_iter = 7
stag_y    = mean_nh[stag_iter - 1]

ax.annotate(
    "stagnates\n(+0.1 pts,\niters 5–50)",
    xy=(stag_iter, min(stag_y, Y_HI)),
    xytext=(19, 276.5),
    fontsize=4.8,
    color="#d62728",
    arrowprops=dict(
        arrowstyle="->",
        color="#d62728",
        lw=0.7,
        connectionstyle="arc3,rad=-0.10",
    ),
    ha="center", va="bottom",
)

# ── Reference line at Score=299 ───────────────────────────────────────────────
ax.axhline(299, color="#aaaaaa", lw=0.7, ls=":", zorder=1)
ax.text(7, 299.4, "299", va="bottom", ha="left", fontsize=4.8, color="#888888")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlabel("Iteration", fontsize=7, fontweight="bold")
ax.set_ylabel("Best SIEVE Score (QPS)", fontsize=7, fontweight="bold")
ax.set_xlim(6, N)
ax.set_ylim(Y_LO, Y_HI)

ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.tick_params(axis="both", labelsize=6)
ax.grid(True, which="major", alpha=0.25, linewidth=0.5)
ax.grid(True, which="minor", alpha=0.00)  # no minor grid — too busy at 5-pt spacing

ax.legend(
    fontsize=5.5,
    loc="lower right",
    framealpha=0.92,
    edgecolor="#cccccc",
    handlelength=2.0,
    labelspacing=0.3,
    borderpad=0.4,
)

plt.tight_layout(pad=0.4)

OUT = "/work/hdd/bdjd/69baaca9e33b6a3148342511/figures/convergence_ablation.pdf"
plt.savefig(OUT, format="pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")

# ── Table values for LaTeX ────────────────────────────────────────────────────
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
