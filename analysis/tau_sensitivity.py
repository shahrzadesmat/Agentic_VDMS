"""
tau_sensitivity.py
------------------
Re-scores every logged (config, benchmark) entry from the HICO-DET and SIFT1M
experiments at multiple tau thresholds.  No new experiments are run; the
optimizer paths are fixed — only the SIEVE step function changes.

HICO-DET SIEVE: Score = QPS  if  mAP >= tau  else  0
SIFT1M   SIEVE: Score = QPS  if  Recall@10 >= tau  else  0

Output: one table per dataset, rows = methods, cols = tau values.
"""

import json
import os
import numpy as np

# ── file paths ────────────────────────────────────────────────────────────────

BASE = "/work/hdd/bdjd/Agentic_VDMS/results"

HICO_FILES = {
    "LLM Agent": [
        f"{BASE}/hico_det/llm/seed42.json",
        f"{BASE}/hico_det/llm/seed99.json",
        f"{BASE}/hico_det/llm/seed200.json",
    ],
    "GP-BO": [
        f"{BASE}/hico_det/gpbo/seed42.json",
        f"{BASE}/hico_det/gpbo/seed99.json",
        f"{BASE}/hico_det/gpbo/seed200.json",
    ],
    "Optuna TPE": [
        f"{BASE}/hico_det/optuna/seed42.json",
        f"{BASE}/hico_det/optuna/seed99.json",
        f"{BASE}/hico_det/optuna/seed200.json",
    ],
    "Random Search": [
        f"{BASE}/hico_det/random/seed42.json",
        f"{BASE}/hico_det/random/seed99.json",
        f"{BASE}/hico_det/random/seed200.json",
    ],
    "Grid Search": [
        f"{BASE}/hico_det/grid/seed42.json",  # single run
    ],
}

SIFT1M_FILES = {
    "LLM Agent":    [f"{BASE}/sift1m/llm/seed42.json"],
    "GP-BO":        [f"{BASE}/sift1m/gpbo/seed42.json"],
    "Optuna TPE":   [f"{BASE}/sift1m/optuna/seed42.json"],
    "Random Search":[f"{BASE}/sift1m/random/seed42.json"],
    "Grid Search":  [f"{BASE}/sift1m/grid/seed42.json"],
}

# tau grids
HICO_TAUS  = [0.10, 0.12, 0.15, 0.18, 0.20]
SIFT1M_TAUS = [0.80, 0.85, 0.90, 0.93, 0.95]


# ── core function ─────────────────────────────────────────────────────────────

def best_score_at_tau(filepath, tau, quality_key):
    """
    Re-apply SIEVE with threshold `tau` to every iteration stored in `filepath`.
    Returns the maximum score achievable (best-so-far at end of 50 iterations).
    Also returns the iteration index at which the best was first reached.
    quality_key: 'map_score' for HICO-DET/GLDv2, 'recall_at_10' for SIFT1M.
    """
    with open(filepath) as f:
        data = json.load(f)

    best = 0.0
    iter_to_best = None
    for r in data["results"]:
        b = r["benchmark"]
        quality = b[quality_key]
        qps = b["qps"]
        score = qps if quality >= tau else 0.0
        if score > best:
            best = score
            iter_to_best = r["iteration"]

    return best, iter_to_best


def table_for_dataset(files_dict, tau_list, quality_key, dataset_name, unit):
    print(f"\n{'='*68}")
    print(f"  {dataset_name}  —  SIEVE Score = QPS if {unit} >= tau else 0")
    print(f"{'='*68}")

    col_w = 10
    header = f"{'Method':<16}" + "".join(f"  tau={t:.2f}" for t in tau_list)
    print(header)
    print("-" * len(header))

    for method, filepaths in files_dict.items():
        row_scores = []
        for tau in tau_list:
            seed_bests = []
            for fp in filepaths:
                if not os.path.isfile(fp):
                    raise FileNotFoundError(fp)
                best, _ = best_score_at_tau(fp, tau, quality_key)
                seed_bests.append(best)
            row_scores.append(np.mean(seed_bests))

        # Highlight the LLM advantage at each tau
        row_str = f"{method:<16}"
        for s in row_scores:
            row_str += f"  {s:8.1f}"
        print(row_str)

    print()

    # ── secondary table: LLM lead over best non-LLM at each tau ──────────────
    print(f"  LLM advantage over best non-LLM baseline:")
    llm_files = files_dict["LLM Agent"]
    other_methods = [k for k in files_dict if k != "LLM Agent"]

    print(f"{'tau':<8}", end="")
    for tau in tau_list:
        print(f"  tau={tau:.2f}", end="")
    print()
    print("-" * (8 + 12 * len(tau_list)))

    llm_means = []
    best_baseline_means = []
    for tau in tau_list:
        llm_seeds = [best_score_at_tau(fp, tau, quality_key)[0] for fp in llm_files]
        llm_mean = np.mean(llm_seeds)
        llm_means.append(llm_mean)

        baseline_means = []
        for m in other_methods:
            s = np.mean([best_score_at_tau(fp, tau, quality_key)[0]
                         for fp in files_dict[m]])
            baseline_means.append(s)
        best_baseline_means.append(max(baseline_means))

    print(f"{'LLM':<8}", end="")
    for v in llm_means:
        print(f"  {v:8.1f}", end="")
    print()

    print(f"{'Best base':<8}", end="")
    for v in best_baseline_means:
        print(f"  {v:8.1f}", end="")
    print()

    print(f"{'Lead%':<8}", end="")
    for llm, base in zip(llm_means, best_baseline_means):
        if base > 0:
            pct = (llm - base) / base * 100
            print(f"  {pct:+7.1f}%", end="")
        else:
            print(f"  {'N/A':>8}", end="")
    print()


# ── iters-to-best under each tau (LLM only) ──────────────────────────────────

def iter_to_best_table(files_dict, tau_list, quality_key, dataset_name):
    print(f"\n  Iters→Best for LLM Agent under each tau ({dataset_name}):")
    llm_files = files_dict["LLM Agent"]
    header = f"{'seed':<8}" + "".join(f"  tau={t:.2f}" for t in tau_list)
    print(header)
    print("-" * len(header))

    for fp in llm_files:
        seed_label = os.path.basename(fp).replace(".json", "")
        row = f"{seed_label:<8}"
        for tau in tau_list:
            _, itb = best_score_at_tau(fp, tau, quality_key)
            row += f"  {str(itb):>8}"
        print(row)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    table_for_dataset(HICO_FILES,  HICO_TAUS,   "map_score",    "HICO-DET", "mAP")
    iter_to_best_table(HICO_FILES, HICO_TAUS,   "map_score",    "HICO-DET")

    table_for_dataset(SIFT1M_FILES, SIFT1M_TAUS, "recall_at_10", "SIFT1M",  "Recall@10")
    iter_to_best_table(SIFT1M_FILES, SIFT1M_TAUS, "recall_at_10", "SIFT1M")
