#!/usr/bin/env python3
"""
Two-panel ablation figure for VLDB paper.
Left: BSF convergence curves (zoomed y=293-302).
Right: per-seed final Score at N=50 with 299 threshold.
The right panel makes the 0/3 vs 3/3 reliability story instantly visible.
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = "/work/hdd/bdjd/vdms_workflow/semantic_vdms/FINAL_RESULTS/hico_det/results"

CONDITIONS = {
    "LLM Agent (Full)": ["llm_seed42", "llm_seed99", "llm_seed200"],
    "w/o History":      ["ablation_no_history_seed42",
                         "ablation_no_history_seed99",
                         "ablation_no_history_seed200"],
    "w/o Phase":        ["ablation_no_phases_seed42",
                         "ablation_no_phases_seed99",
                         "ablation_no_phases_seed200"],
    "w/o Both":         ["ablation_no_history_no_phases_seed42",
                         "ablation_no_history_no_phases_seed99",
                         "ablation_no_history_no_phases_seed200"],
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

STYLES = {
    "LLM Agent (Full)": dict(color="#1a1a1a", lw=2.0, ls="-",  zorder=5),
    "w/o History":      dict(color="#d62728", lw=1.6, ls="--", zorder=4),
    "w/o Phase":        dict(color="#1f77b4", lw=1.6, ls="-.", zorder=3),
    "w/o Both":         dict(color="#ff7f0e", lw=1.6, ls=":",  zorder=2),
}

T_EXP, T_EXPL = 20, 38
Y_LO, Y_HI = 293, 302
THRESH = 299

fig, (ax_l, ax_r) = plt.subplots(
    1, 2, figsize=(3.3, 1.30),
    gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.08}
)

# ── LEFT: convergence curves ──────────────────────────────────────────────────
# Success zone shading
ax_l.axhspan(THRESH, Y_HI, alpha=0.07, color="#4daf4a", lw=0)

# Phase boundary lines (very faint)
for t in [T_EXP, T_EXPL]:
    ax_l.axvline(t + 0.5, color="#cccccc", lw=0.5, ls="--", zorder=1)

for label, arr in curves.items():
    mean = arr.mean(0)
    st = STYLES[label]
    ax_l.plot(iters, mean, label=label, **st, clip_on=True)

ax_l.axhline(THRESH, color="#444444", lw=0.8, ls="--", zorder=3)

ax_l.set_xlim(6, N)
ax_l.set_ylim(Y_LO, Y_HI)
ax_l.set_xlabel("Iteration", fontsize=7, fontweight="bold")
ax_l.set_ylabel("Best SIEVE Score (QPS)", fontsize=7, fontweight="bold")
ax_l.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_l.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax_l.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_l.tick_params(axis="both", labelsize=5.5)
ax_l.grid(True, which="major", alpha=0.20, linewidth=0.5)

ax_l.legend(fontsize=4.8, loc="lower right", framealpha=0.92,
            edgecolor="#cccccc", handlelength=1.8, labelspacing=0.25,
            borderpad=0.4)

# ── RIGHT: per-seed final score dot plot ──────────────────────────────────────
ax_r.axhspan(THRESH, Y_HI, alpha=0.07, color="#4daf4a", lw=0)
ax_r.axhline(THRESH, color="#444444", lw=0.8, ls="--", zorder=3)

labels_short = ["Full", "−Hist.", "−Phase", "−Both"]
x_pos = np.arange(len(CONDITIONS))

for xi, (label, col_short) in enumerate(zip(CONDITIONS.keys(), labels_short)):
    final_scores = curves[label][:, -1]
    col = STYLES[label]["color"]
    for s in final_scores:
        ax_r.plot(xi, s, "o", color=col, ms=3.5, zorder=5,
                  markeredgewidth=0.3, markeredgecolor="white")

ax_r.set_xlim(-0.5, len(CONDITIONS) - 0.5)
ax_r.set_ylim(Y_LO, Y_HI)
ax_r.set_xticks(x_pos)
ax_r.set_xticklabels(labels_short, fontsize=4.8, rotation=30, ha="right")
ax_r.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax_r.tick_params(axis="y", labelsize=5.5)
ax_r.tick_params(axis="x", length=0)
ax_r.grid(True, axis="y", which="major", alpha=0.20, linewidth=0.5)
ax_r.set_ylabel("")
ax_r.yaxis.set_ticklabels([])  # share y-axis visually

# Label the threshold on the right side
ax_r.text(len(CONDITIONS) - 0.5 + 0.05, THRESH + 0.15, "299",
          va="bottom", ha="left", fontsize=4.5, color="#444444")

# Seeds ≥299 count annotation above each column
above_counts = {label: int((curves[label][:, -1] >= THRESH).sum())
                for label in CONDITIONS}
for xi, label in enumerate(CONDITIONS):
    n = above_counts[label]
    ax_r.text(xi, Y_HI - 0.15, f"{n}/3",
              ha="center", va="top", fontsize=4.5,
              color=STYLES[label]["color"], fontweight="bold")

plt.tight_layout(pad=0.35)

OUT = "/work/hdd/bdjd/69baaca9e33b6a3148342511/figures/convergence_ablation.pdf"
plt.savefig(OUT, format="pdf", bbox_inches="tight", dpi=300)
print(f"Saved: {OUT}")

print("\n--- Final scores per seed ---")
for label, arr in curves.items():
    finals = arr[:, -1]
    print(f"{label:<22}: {finals}  mean={finals.mean():.1f}  seeds>=299={above_counts[label]}/3")
