# Agentic VDMS — Supplementary Materials

**Paper:** *LLM-Guided ANN Index Optimization for Human-Object Interaction Retrieval*
**Venue:** VLDB 2027 submission

![Optimization Framework](assets/workflow.png)

> **Figure:** Five optimizer methods propose VDMS configurations over N=50 iterations.
> Adaptive methods (LLM Agent, Optuna TPE, GP-BO) receive SIEVE score feedback;
> Grid and Random Search do not.

---

## Repository Structure

```
├── src/                         # Python source code
│   ├── hico_det/
│   │   ├── hico_agent_optimizer.py      # Main LLM agent optimizer (all methods)
│   │   ├── benchmark_hico_det.py        # Standalone benchmark harness
│   │   ├── centroid_bias_validation.py  # Text-query centroid geometry proof
│   │   ├── extract_hico_det.py          # HICO-DET dataset preparation (images, SIFT, DINOv2)
│   │   ├── extract_hico_text_queries.py # MobileCLIP-S2 HOI text embeddings
│   │   ├── extract_hico_clip_features.py # CLIP image+text embeddings (MobileCLIP-S2 & ViT-L/14)
│   │   └── run_hico_system_baselines.py # UniIR / FaissFlat baselines
│   ├── gldv2/
│   │   ├── gldv2_agent_optimizer.py     # GLDv2 optimizer (cross-domain)
│   │   ├── extract_gldv2_features.py    # CLIP + DINOv2 feature extraction
│   │   ├── benchmark_gldv2.py           # Standalone benchmark harness
│   │   └── run_gldv2_system_baselines.py
│   ├── sift1m/
│   │   └── sift1m_agent_optimizer.py    # SIFT1M optimizer (pure ANN)
│   ├── milvus/
│   │   ├── milvus_hico_optimizer.py     # LLM/baseline optimizers on Milvus (HICO-DET)
│   │   ├── milvus_gldv2_optimizer.py    # LLM/baseline optimizers on Milvus (GLDv2)
│   │   └── milvus_sift1m_optimizer.py   # LLM/baseline optimizers on Milvus (SIFT1M)
│   └── vdtuner_ehvi.py                  # VDTuner baseline (EHVI multi-objective)
│
├── experiments/                 # SLURM scripts to reproduce every run
│   ├── hico_det/
│   │   ├── run_llm_seed{42,99,200}.sh         # LLM agent (3 seeds)
│   │   ├── run_gpbo_seed{42,99,200}.sh        # GP-BO baseline
│   │   ├── run_optuna_seed{42,99,200}.sh      # Optuna TPE baseline
│   │   ├── run_random_seed{42,99,200}.sh      # Random Search baseline
│   │   ├── run_grid.sh                        # Grid Search (1 run)
│   │   ├── run_ablation_no_history_seed{42,99,200}.sh
│   │   ├── run_ablation_no_phases_seed{42,99,200}.sh
│   │   ├── run_ablation_no_history_no_phases_seed{42,99,200}.sh
│   │   ├── run_qds_seed{42,99,200}.sh         # Query Difficulty-weighted objective
│   │   ├── run_backbone_gpt4omini_seed42.sh   # GPT-4o-mini backbone ablation
│   │   ├── run_backbone_llama_seed42.sh       # Llama-3.3-70B backbone ablation
│   │   └── no_threshold/                      # Exploratory runs without SIEVE threshold
│   ├── gldv2/
│   │   └── run_{llm,gpbo,optuna,random,grid,vdtuner}_seed{42,99,200}.sh
│   └── sift1m/
│       └── run_{llm,gpbo,optuna,random,grid,vdtuner}_seed{42,99,200}.sh
│
├── results/                     # Canonical JSON result files
│   ├── hico_det/
│   │   ├── llm/seed{42,99,200}.json
│   │   ├── gpbo/seed{42,99,200}.json
│   │   ├── optuna/seed{42,99,200}.json
│   │   ├── random/seed{42,99,200}.json
│   │   ├── grid/seed42.json
│   │   ├── ablations/no_history_seed{42,99,200}.json
│   │   ├── ablations/no_phases_seed{42,99,200}.json
│   │   ├── ablations/no_history_no_phases_seed{42,99,200}.json
│   │   ├── ablations/phase_boundary_{A,B}_seed42.json  # Phase schedule sensitivity
│   │   ├── qds/seed{42,99,200}.json
│   │   ├── backbone/gpt4o_mini_seed42.json
│   │   ├── backbone/llama_seed42.json
│   │   ├── system_baselines.json           # UniIR / FaissFlat system baselines
│   │   └── system_baselines_full.json      # Extended baselines with per-query detail
│   ├── gldv2/{llm,gpbo,optuna,random,grid,vdtuner}/seed{42,99,200}.json
│   └── sift1m/{llm,gpbo,optuna,random,grid,vdtuner}/seed{42,99,200}.json
│
├── analysis/
│   ├── coupling_metric.py       # Quantifies parameter coupling across datasets (supports Section 5)
│   └── tau_sensitivity.py       # Re-scores all runs at multiple τ thresholds (SIEVE sensitivity)
│
├── milvus/                      # Milvus replication: all methods × 3 datasets × 3 seeds
│   ├── hico_det/{results/,run_milvus_*.sh}
│   ├── gldv2/{results/,run_milvus_*.sh}
│   └── sift1m/{results/,run_milvus_*.sh}
│
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU (tested on A40, CUDA 11.8)
- [Intel VDMS](https://github.com/IntelLabs/vdms) (via Apptainer container)
- OpenRouter API key (for LLM agent runs)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data preparation (HICO-DET)
```bash
# 1. Extract images, SIFT, DINOv2 features, and ground truth
python src/hico_det/extract_hico_det.py --data-dir /path/to/hico_det

# 2. Extract MobileCLIP-S2 text embeddings (hico_clip_text_queries.npy)
python src/hico_det/extract_hico_text_queries.py --data-dir /path/to/hico_det

# 3. Extract CLIP image + text embeddings for both backbones
#    Produces: hico_clip.npy (MobileCLIP-S2, 512-d)
#              hico_clipvitl14_db.npy (ViT-L/14, 768-d)
#              hico_clipvitl14_text_queries.npy (ViT-L/14, 768-d)
python src/hico_det/extract_hico_clip_features.py --dataset-dir /path/to/hico_det
```

---

## Reproducing Results

Set your OpenRouter API key before running any LLM experiment:
```bash
export OPENROUTER_API_KEY="your-key-here"
```

Submit a SLURM job (example — LLM agent on HICO-DET, seed 42):
```bash
sbatch experiments/hico_det/run_llm_seed42.sh
```

Each script writes its result JSON to a path specified by `OUTPUT` inside the script. To reproduce the exact paper numbers, run all three seeds and average the `best_score` field from each output JSON.

---

## Result Format

Each JSON result file contains:
```json
{
  "summary": {
    "best_score": 300.5,
    "best_map": 0.160,      // "best_recall" for SIFT1M (Recall@10, not mAP)
    "best_qps": 300.5,
    "best_config": { ... },
    "iterations_used": 50
  },
  "results": [ ... ]
}
```

> **Note:** SIFT1M summaries use `best_recall` (Recall@10) in place of `best_map`. Per-iteration records in `results[]` always use `benchmark.recall_at_10` (SIFT1M) or `benchmark.map_score` (HICO-DET/GLDv2).

The **SIEVE score** = `best_qps` when the quality metric ≥ τ (τ = 0.15 for HICO-DET/GLDv2, τ = 0.90 for SIFT1M), else 0. All canonical results in `results/` already meet the threshold.

---

## Key Results (Tables 2–4 in paper)

| Method       | HICO-DET Score (QPS) | GLDv2 Score  | SIFT1M Score |
|--------------|----------------------|--------------|--------------|
| LLM Agent    | **300.3** (±0.3)     | 271.45       | **1184.5**   |
| VDTuner      | 223.8                | **273.91**   | 1150.9       |
| Optuna TPE   | 225.2                | 272.35       | 1160.7       |
| Grid Search  | 183.0                | 272.98       | 1174.4       |
| Random       | 111.4                | 265.47       | 1143.6       |
| GP-BO        | 199.3                | 56.57        | 755.4        |

Score = QPS if mAP ≥ τ, else 0 (SIEVE objective).
All scores are 3-seed means (seeds 42, 99, 200); Grid Search is 1 run (seed 42).

Phase boundary sensitivity (t_exp/t_expl): all variants within ±0.53% of default, confirming robustness.
