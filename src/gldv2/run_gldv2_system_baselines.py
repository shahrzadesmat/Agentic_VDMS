"""
System Baseline Benchmarks — GLDv2 Landmark Retrieval
======================================================
Runs 4 fixed configs through VDMS (FaissFlat = exact brute-force) to measure
QPS and compute Score = mAP x QPS for Table 1 (system baselines).

Configs:
  1. CLIP alone       (alpha=0.0, k=500)   — fast retrieval
  2. CLIP alone       (alpha=0.0, k=2000)  — large-k brute-force
  3. CLIP+DINOv2 best (alpha=0.9, k=2000)  — best brute-force fusion (α from optimizer)
  4. DINOv2 alone     (alpha=1.0, k=2000)  — pure reranking upper bound

All use FaissFlat (exact, no approximation) and n_refs=1, ref_strategy=first.

Usage:
    python run_gldv2_system_baselines.py --port 55598 --dataset-dir /path/to/datasets
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gldv2_agent_optimizer import run_benchmark


BASELINE_CONFIGS = [
    {
        "name":         "CLIP_alone_k500",
        "description":  "CLIP ViT-L/14 only — FaissFlat exact, k=500, alpha=0.0",
        "engine":       "FaissFlat",
        "params":       {},
        "k_neighbors":  500,
        "alpha":        0.0,
        "n_refs":       1,
        "ref_strategy": "first",
        "n_aqe":        1,
        "aqe_weight":   0.0,
    },
    {
        "name":         "CLIP_alone_k2000",
        "description":  "CLIP ViT-L/14 only — FaissFlat exact, k=2000, alpha=0.0",
        "engine":       "FaissFlat",
        "params":       {},
        "k_neighbors":  2000,
        "alpha":        0.0,
        "n_refs":       1,
        "ref_strategy": "first",
        "n_aqe":        1,
        "aqe_weight":   0.0,
    },
    {
        "name":         "CLIP_DINOv2_best",
        "description":  "CLIP + DINOv2 best fusion — FaissFlat exact, k=2000, alpha=0.9",
        "engine":       "FaissFlat",
        "params":       {},
        "k_neighbors":  2000,
        "alpha":        0.9,
        "n_refs":       1,
        "ref_strategy": "first",
        "n_aqe":        1,
        "aqe_weight":   0.0,
    },
    {
        "name":         "DINOv2_alone",
        "description":  "DINOv2 alone — FaissFlat exact, k=2000, alpha=1.0",
        "engine":       "FaissFlat",
        "params":       {},
        "k_neighbors":  2000,
        "alpha":        1.0,
        "n_refs":       1,
        "ref_strategy": "first",
        "n_aqe":        1,
        "aqe_weight":   0.0,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",        type=int,  required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output",      type=str,
                        default="gldv2/gldv2_system_baselines.json")
    args = parser.parse_args()

    print("=" * 70)
    print("GLDv2 System Baseline Benchmarks")
    print(f"  Backbone:    CLIP ViT-L/14 (768-d) + DINOv2 ViT-L/14-reg4")
    print(f"  Index type:  FaissFlat (exact brute-force)")
    print(f"  Corpus:      761,757 vectors")
    print(f"  Queries:     ~1,100 GLDv2 dev queries")
    print(f"  Port:        {args.port}")
    print("=" * 70)

    results = []
    for i, cfg in enumerate(BASELINE_CONFIGS):
        name = cfg["name"]
        desc = cfg["description"]
        config = {k: v for k, v in cfg.items() if k not in ("name", "description")}
        config["_iter"] = i + 1

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(BASELINE_CONFIGS)}] {name}")
        print(f"  {desc}")
        print(f"  k={config['k_neighbors']}  alpha={config['alpha']:.2f}")
        print(f"{'='*70}")

        benchmark = run_benchmark(
            args.port, str(args.dataset_dir),
            config, iteration=i + 1,
        )

        score = benchmark.score()
        print(f"\n  Score (mAP x QPS): {score:.4f}")
        print(f"  mAP:               {benchmark.map_score:.4f}")
        print(f"  QPS:               {benchmark.qps:.2f}")
        print(f"  P@10:              {benchmark.precision_at_10:.4f}")
        print(f"  nDCG@10:           {benchmark.ndcg_at_10:.4f}")
        print(f"  Recall@10:         {benchmark.recall_at_10:.4f}")
        print(f"  Index build:       {benchmark.index_build_s:.1f}s")

        results.append({
            "name":            name,
            "description":     desc,
            "config":          config,
            "score":           score,
            "map_score":       benchmark.map_score,
            "qps":             benchmark.qps,
            "precision_at_10": benchmark.precision_at_10,
            "ndcg_at_10":      benchmark.ndcg_at_10,
            "recall_at_10":    benchmark.recall_at_10,
            "latency_ms":      benchmark.latency_ms,
            "t_clip_avg_ms":   benchmark.t_clip_avg_ms,
            "t_rerank_avg_ms": benchmark.t_rerank_avg_ms,
            "index_build_s":   benchmark.index_build_s,
        })

    print(f"\n{'='*70}")
    print("SYSTEM BASELINES SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<25} {'Score':>8} {'mAP':>8} {'QPS':>8} {'R@10':>8}")
    print("  " + "-" * 63)
    for r in results:
        print(f"  {r['name']:<25} {r['score']:>8.4f} {r['map_score']:>8.4f} "
              f"{r['qps']:>8.2f} {r['recall_at_10']:>8.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
