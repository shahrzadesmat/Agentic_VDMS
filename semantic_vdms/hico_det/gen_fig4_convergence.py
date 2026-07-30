#!/usr/bin/env python3
"""
Generate Figure 4: Best SIEVE Score vs. Iteration on HICO-DET.
100% data integrity from finalized 150-run results (llm/optuna/gpbo/random/grid _seed*.json).
SIEVE: Score = QPS if mAP >= tau else 0  (tau = 0.15).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "/path/to/Agentic_VDMS/semantic_vdms/FINAL_RESULTS/hico_det/results"
TAU  = 0.15

# ── Data loading ──────────────────────────────────────────────────────

def sieve_scores(fname):
    with open(f"{BASE}/{fname}") as f:
        data = json.load(f)
    return [
        r["benchmark"]["qps"] if r["benchmark"]["map_score"] >= TAU else 0.0
        for r in data["results"]
    ]

def best_so_far(scores):
    best, cur = [], float("-inf")
    for s in scores:
        cur = max(cur, s)
        best.append(cur)
    return np.array(best)

def load_method(files):
    curves = [best_so_far(sieve_scores(f)) for f in files]
    n = min(len(c) for c in curves)
    arr = np.array([c[:n] for c in curves])
    return arr          # shape (n_seeds, n_iters)

llm_arr    = load_method(["llm_seed42.json",    "llm_seed99.json",    "llm_seed200.json"])
optuna_arr = load_method(["optuna_seed42.json", "optuna_seed99.json", "optuna_seed200.json"])
gpbo_arr   = load_method(["gpbo_seed42.json",   "gpbo_seed99.json",   "gpbo_seed200.json"])
rnd_arr    = load_method(["random_seed42.json", "random_seed99.json", "random_seed200.json"])
grid_arr   = load_method(["grid.json"])          # single run

N = llm_arr.shape[1]   # 50
iters = np.arange(1, N + 1)

# Derived statistics
llm_mean  = llm_arr.mean(0);   llm_std   = llm_arr.std(0)
opt_mean  = optuna_arr.mean(0)
gpbo_mean = gpbo_arr.mean(0)
rnd_mean  = rnd_arr.mean(0)
grid_curve = grid_arr[0]

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.4))

# Colour palette (colour-blind-friendly)
C_LLM    = "#1a1a1a"    # near-black
C_OPT    = "#2166ac"    # blue
C_GPBO   = "#4dac26"    # green
C_RND    = "#d6604d"    # coral-red
C_GRID   = "#7b2d8b"    # purple

MK = 5       # marker every N iters
MS = 5.5     # marker size
MEW = 0.6    # marker edge width

# ── LLM Agent — mean + ±1σ band ──────────────────────────────────────
ax.fill_between(iters, llm_mean - llm_std, llm_mean + llm_std,
                color=C_LLM, alpha=0.12, linewidth=0, label="_nolegend_")
ax.plot(iters, llm_mean,
        color=C_LLM, lw=2.2, linestyle="-",
        marker="o", markersize=MS, markevery=MK, markeredgewidth=MEW,
        markerfacecolor=C_LLM, markeredgecolor="white",
        label="LLM Agent (Ours)")

# ── Optuna TPE ───────────────────────────────────────────────────────
ax.plot(iters, opt_mean,
        color=C_OPT, lw=1.8, linestyle="--",
        marker="s", markersize=MS, markevery=MK, markeredgewidth=MEW,
        markerfacecolor=C_OPT, markeredgecolor="white",
        label="Optuna TPE")

# ── GP-BO ─────────────────────────────────────────────────────────────
ax.plot(iters, gpbo_mean,
        color=C_GPBO, lw=1.8, linestyle="-.",
        marker="^", markersize=MS, markevery=MK, markeredgewidth=MEW,
        markerfacecolor=C_GPBO, markeredgecolor="white",
        label="GP-BO")

# ── Random Search ────────────────────────────────────────────────────
ax.plot(iters, rnd_mean,
        color=C_RND, lw=1.8, linestyle=":",
        marker="D", markersize=MS-0.5, markevery=MK, markeredgewidth=MEW,
        markerfacecolor=C_RND, markeredgecolor="white",
        label="Random Search")

# ── Grid Search ───────────────────────────────────────────────────────
ax.plot(iters, grid_curve,
        color=C_GRID, lw=1.8, linestyle=(0, (4, 1, 1, 1)),
        marker="P", markersize=MS, markevery=MK, markeredgewidth=MEW,
        markerfacecolor=C_GRID, markeredgecolor="white",
        label="Grid Search")

# ── Annotations ──────────────────────────────────────────────────────
# Vertical dashed line at iter 23; label sits in the clean gap above LLM plateau
ax.axvline(x=23, color=C_LLM, lw=0.8, linestyle="--", alpha=0.45)
ax.text(24.2, 305, "iter 23\n(LLM peak)", fontsize=7.5, color=C_LLM,
        va="bottom", ha="left", alpha=0.80)

# Horizontal dotted reference at LLM plateau
ax.axhline(y=300.3, color=C_LLM, lw=0.6, linestyle=":", alpha=0.35)

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
ax.set_ylabel("Best SIEVE Score (QPS)", fontsize=12, fontweight="bold")

ax.set_xlim(1, 50)
ax.set_ylim(bottom=0, top=340)

ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

# Professional grid: horizontal-only, light, no minor clutter
ax.grid(True, axis="y", which="major", alpha=0.22, linewidth=0.6,
        color="#888888", linestyle="-")
ax.set_axisbelow(True)

ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95,
          handlelength=2.5, labelspacing=0.40, ncol=3, columnspacing=0.8,
          edgecolor="#cccccc", handler_map={})

ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=0.8)
ax.tick_params(axis="both", which="minor", length=0)

# Clean spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

plt.tight_layout(pad=0.8)

OUT = "/path/to/paper/figures/convergence_hico_det.pdf"
plt.savefig(OUT, format="pdf", bbox_inches="tight")
print(f"Saved: {OUT}")

# ── Verification printout ─────────────────────────────────────────────
print(f"\nFinal values (iter 50):")
print(f"  LLM:    {llm_mean[-1]:.1f} ± {llm_std[-1]:.1f}")
print(f"  Optuna: {opt_mean[-1]:.1f}")
print(f"  GP-BO:  {gpbo_mean[-1]:.1f}")
print(f"  Random: {rnd_mean[-1]:.1f}")
print(f"  Grid:   {grid_curve[-1]:.1f}")

# When LLM crosses each baseline
for name, curve in [("Optuna",opt_mean),("GP-BO",gpbo_mean),
                    ("Random",rnd_mean),("Grid",grid_curve)]:
    target = curve[-1]
    cross = next((i+1 for i,v in enumerate(llm_mean) if v >= target), None)
    print(f"  LLM crosses {name} ({target:.1f}) at iter {cross}")
