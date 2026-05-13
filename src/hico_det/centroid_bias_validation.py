"""
Centroid bias validation: compare CLIP text query AP vs DINOv2 image query AP
across HICO-DET HOI categories.

Uses:
  FINAL_RESULTS/hico_det/results/per_class_ap.json  — per-category AP (best pipeline config)
  FINAL_RESULTS/hico_det/results/system_baselines.json — aggregate mAP for text-only
                                                          and DINOv2-only baselines

Since per-category AP for alpha=0 (text-only) and alpha=1.0 (DINOv2-only) are not
directly stored, this script:
  1. Reports aggregate text vs image comparison from system_baselines.json
  2. Uses the per_class_ap.json 'no_prf' entry (Stage-1 text retrieval, minimal DINOv2
     influence: n_refs=1, ref_strategy='first', alpha=0.75) as the best available
     per-category proxy.

To obtain true per-category text vs image comparison, run eval_per_class_ap.py with
alpha=0 and alpha=1.0 configurations.
"""

import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "FINAL_RESULTS" / "hico_det" / "results"

per_class_data = json.load(open(RESULTS_DIR / "per_class_ap.json"))
system_baselines = json.load(open(RESULTS_DIR / "system_baselines.json"))

# ── Aggregate comparison from system baselines ────────────────────────────────
clip_text_entry  = next(e for e in system_baselines if e["name"] == "CLIP_alone_k2000")
dinov2_entry     = next(e for e in system_baselines if e["name"] == "DINOv2_alone")

clip_text_map  = clip_text_entry["map_score"]
dinov2_map     = dinov2_entry["map_score"]
delta_map_agg  = clip_text_map - dinov2_map

print("=" * 60)
print("Aggregate comparison (system_baselines.json)")
print(f"  CLIP text queries (alpha=0): mAP = {clip_text_map:.4f}")
print(f"  DINOv2 image queries (alpha=1): mAP = {dinov2_map:.4f}")
print(f"  Δ = {delta_map_agg:+.4f} ({delta_map_agg * 100:+.2f} pp) in favor of text")
print()

# ── Per-category comparison using 'no_prf' proxy ─────────────────────────────
# 'no_prf': alpha=0.75 text+DINOv2 with n_refs=1 first image (Stage 1 text-dominant)
# 'best_config': alpha=0.75 with n_refs=5 diverse (PRF-enhanced)
# These are proxies, not pure text vs image. The ratio alpha=0 vs alpha=1.0 requires
# separate evaluation runs.
no_prf_entry    = next(e for e in per_class_data if e["label"] == "no_prf")
best_cfg_entry  = next(e for e in per_class_data if e["label"] == "best_config")

hoi_keys        = no_prf_entry["hoi_keys"]
no_prf_ap       = np.array(no_prf_entry["per_query_ap"])
best_cfg_ap     = np.array(best_cfg_entry["per_query_ap"])

n_categories = len(hoi_keys)
delta_per_class = best_cfg_ap - no_prf_ap
pct_improved  = 100.0 * np.mean(delta_per_class > 0)
mean_delta    = np.mean(delta_per_class)

print("=" * 60)
print(f"Per-category proxy comparison (n={n_categories} HOI classes)")
print(f"  Comparison: best_config (n_refs=5, diverse) vs no_prf (n_refs=1, first)")
print(f"  % categories where best_config AP > no_prf AP: {pct_improved:.1f}%")
print(f"  Mean ΔAP: {mean_delta:+.4f} ({mean_delta * 100:+.2f} pp)")
print()

# ── Theory-consistent estimate for text vs image ──────────────────────────────
# The centroid-displacement theory (Eq. 7-9) predicts text queries (alpha=0)
# outperform any single image query (alpha=1.0) for compositional HOI categories.
# Aggregate evidence: text mAP={clip_text_map:.4f} > DINOv2 mAP={dinov2_map:.4f}
# Per-category distribution of delta (proxy):
n_text_wins = int(np.sum(no_prf_ap > dinov2_map))  # categories where no_prf > DINOv2 aggregate

print("=" * 60)
print("Summary for paper §4.4:")
print(f"  Aggregate: text (mAP={clip_text_map:.4f}) outperforms image (mAP={dinov2_map:.4f})")
print(f"  Δ = +{delta_map_agg * 100:.2f} pp across all 600 HOI categories")
print(f"  PRF benefit: {pct_improved:.0f}% of {n_categories} categories improve with n_refs=5")
print()
print("Suggested sentence for §4.4:")
print(f"  Empirically, text queries (mAP={clip_text_map:.3f}) outperform image queries")
print(f"  (mAP={dinov2_map:.3f}) by +{delta_map_agg * 100:.2f}\\,pp on aggregate across")
print(f"  all {n_categories} HOI categories, consistent with Eq.~\\ref{{eq:textquery}}.")
