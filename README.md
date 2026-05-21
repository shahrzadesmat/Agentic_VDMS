# Agentic VDMS — Supplementary Materials

**Paper:** *LLM-Guided ANN Index Optimization for Human-Object Interaction Retrieval*
**Authors:** Shahrzad Esmat, Chaunté W. Lacewell, Sameh Gobriel, Nilesh Jain, Ali Jannesari
**Venue:** Under review at PVLDB Vol. 20, 2027

<img src="assets/workflow_fig.png" alt="Optimization Framework"><br>
<em>Five optimizer methods propose VDMS configurations over N=50 iterations. Adaptive methods (LLM Agent, Optuna TPE, GP-BO) receive SIEVE score feedback; Grid and Random Search do not.</em>

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
│   │   ├── run_vdtuner_seed{42,99,200}.sh     # VDTuner baseline
│   │   ├── run_grid.sh                        # Grid Search (1 run, seed 42)
│   │   ├── run_ablation_no_history_seed{42,99,200}.sh
│   │   ├── run_ablation_no_phases_seed{42,99,200}.sh
│   │   ├── run_ablation_no_history_no_phases_seed{42,99,200}.sh
│   │   ├── run_qds_seed{42,99,200}.sh         # Query Difficulty-weighted objective
│   │   ├── run_phase_boundary_{A,B}_seed42.sh # Phase schedule sensitivity
│   │   ├── run_backbone_gpt4omini_seed42.sh   # GPT-4o-mini backbone ablation
│   │   ├── run_backbone_llama_seed42.sh       # Llama-3.3-70B backbone ablation
│   │   └── no_threshold/                      # Exploratory runs with --no-threshold (Score = QPS, no mAP gate)
│   ├── gldv2/
│   │   └── run_{llm,gpbo,optuna,random,vdtuner}_seed{42,99,200}.sh + run_grid_seed42.sh
│   └── sift1m/
│       └── run_{llm,gpbo,optuna,random,vdtuner}_seed{42,99,200}.sh + run_grid_seed42.sh
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

## Datasets

| Dataset | Scale | Source |
|---------|-------|--------|
| HICO-DET | 47,776 images, 600 HOI categories | [zhimeng/hico_det on Hugging Face](https://huggingface.co/datasets/zhimeng/hico_det) — auto-downloaded by `extract_hico_det.py` |
| GLDv2 | 762K gallery images, 1,129 queries | [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark) — download the 100 index tar archives + query images |
| SIFT1M | 1M vectors, 128-d | [INRIA Texmex corpus](http://corpus-texmex.irisa.fr/) — download `sift.tar.gz` and place extracted files under `$DATASET_DIR/sift/` |

---

## Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU (tested on A40, CUDA 11.8)
- [Apptainer](https://apptainer.org/) (for the VDMS container)
- OpenRouter API key (for LLM agent runs)

### VDMS container

All experiments run VDMS inside an Apptainer container. Build the `.sif` image from the [Intel VDMS](https://github.com/IntelLabs/vdms) source:

```bash
git clone https://github.com/IntelLabs/vdms.git
cd vdms
apptainer build vdms_latest.sif docker://intellabs/vdms:latest
```

Then set `CONTAINER=/path/to/vdms_latest.sif` in each SLURM script's USER CONFIG block.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Data preparation (HICO-DET)
```bash
# 1. Extract images, SIFT, DINOv2 features, and ground truth
#    Dataset is auto-downloaded from HuggingFace (zhimeng/hico_det)
python src/hico_det/extract_hico_det.py --output-dir /path/to/hico_det

# 2. Extract MobileCLIP-S2 text embeddings (hico_clip_text_queries.npy)
python src/hico_det/extract_hico_text_queries.py --dataset-dir /path/to/hico_det

# 3. Extract CLIP image + text embeddings for both backbones
#    Produces: hico_clip.npy (MobileCLIP-S2, 512-d)
#              hico_clipvitl14_db.npy (ViT-L/14, 768-d)
#              hico_clipvitl14_text_queries.npy (ViT-L/14, 768-d)
python src/hico_det/extract_hico_clip_features.py --dataset-dir /path/to/hico_det
```

### Data preparation (GLDv2)
Download the 100 index tar archives from [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark) and the query images. Then extract CLIP ViT-L/14 + DINOv2 features (~4–5 h on A40):
```bash
# Extract index image features (762K images, streamed — no disk extraction needed)
python src/gldv2/extract_gldv2_features.py \
    --index-dir /path/to/gldv2/index \
    --output-dir /path/to/datasets/gldv2

# Extract query features (requires gldv2_queries.json — see note below)
python src/gldv2/extract_gldv2_features.py --extract-queries \
    --gldv2-dir /path/to/gldv2 \
    --output-dir /path/to/datasets/gldv2
```

> **Note:** Query extraction requires `gldv2_queries.json` (produced by `compute_gldv2_gt.py`). This script is not yet in the repo; contact the authors for the ground-truth preparation script or build `gldv2_queries.json` from the [GLDv2 retrieval split](https://github.com/cvdfoundation/google-landmark).

### Data preparation (SIFT1M)
Download [`sift.tar.gz`](http://corpus-texmex.irisa.fr/) from the INRIA Texmex corpus and extract into `$DATASET_DIR/sift/`:
```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz -C /path/to/datasets/
# Expected files: sift/sift_base.fvecs  sift/sift_query.fvecs  sift/sift_groundtruth.ivecs
```
No feature extraction script is needed — the SIFT1M optimizer reads `.fvecs` files directly.

---

## Reproducing Results

Set your OpenRouter API key before running any LLM experiment:
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### Via SLURM (recommended)

Edit the four variables in the `USER CONFIG` block at the top of each script, then submit:
```bash
sbatch experiments/hico_det/run_llm_seed42.sh
```

Each script writes its result JSON to a path specified by `OUTPUT` inside the script. To reproduce the exact paper numbers, run all three seeds and average the `best_score` field from each output JSON.

### Direct Python execution (no SLURM)

Start VDMS manually, then invoke the optimizer directly:
```bash
# 1. Start VDMS container (adjust paths to match your environment)
DB_ROOT=/tmp/vdms_db && mkdir -p $DB_ROOT
printf '{"port":55630,"db_root_path":"/db","max_simultaneous_clients":100}' > /tmp/vdms.json
apptainer instance start --bind $DB_ROOT:/db vdms_latest.sif vdms_inst
apptainer exec instance://vdms_inst /vdms/build/vdms -cfg /tmp/vdms.json &
sleep 5  # wait for VDMS to become ready

# 2. Run the optimizer (example: LLM agent on HICO-DET, seed 42)
python src/hico_det/hico_agent_optimizer.py \
    --port 55630 \
    --dataset-dir /path/to/datasets \
    --method hyperparameter_only \
    --iterations 50 \
    --seed 42 \
    --model "minimax/minimax-m2.1" \
    --patience 0 \
    --clip-backbone vitl14 \
    --output results/hico_det/llm/seed42.json

# 3. Stop VDMS when done
apptainer instance stop vdms_inst
```

Available `--method` values: `hyperparameter_only` (LLM), `gpbo`, `optuna`, `random`, `grid`, `vdtuner`.

### `no_threshold/` exploratory scripts

The scripts in `experiments/hico_det/no_threshold/` run with `--no-threshold`, meaning Score = QPS with no mAP quality gate. These were used during development to understand raw optimizer behavior without the SIEVE constraint. Their output JSONs are not included in `results/` because these runs are not reported in the paper.

### Reproducing system baselines (UniIR / FaissFlat)

`results/hico_det/system_baselines.json` was produced by `run_hico_system_baselines.py`:
```bash
# Requires VDMS running on --port (start as shown above)
python src/hico_det/run_hico_system_baselines.py \
    --port 55630 \
    --dataset-dir /path/to/datasets \
    --output results/hico_det/system_baselines.json
```

---

## Running the Analysis Scripts

The `analysis/` scripts re-process the canonical result JSONs in `results/` — no new experiments needed.

```bash
# SIEVE threshold sensitivity (Table 5 in paper)
# Prints one table per dataset showing Score at multiple τ values
python analysis/tau_sensitivity.py

# Parameter coupling metric (Section 5 in paper)
# Prints Spearman rho and feasibility rates across datasets
python analysis/coupling_metric.py
```

> **Note:** Both scripts have a hardcoded `BASE` path at the top (currently set to the cluster path). If you clone the repo to a different location, update the `BASE` variable in each script to point to your local `results/` directory.

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

---

## Citation

If you use this work, please cite:

```bibtex
@misc{esmat2027llm,
  author    = {Shahrzad Esmat and
               Chaunt{\'{e}} W. Lacewell and
               Sameh Gobriel and
               Nilesh Jain and
               Ali Jannesari},
  title     = {LLM-Guided {ANN} Index Optimization for Human-Object Interaction Retrieval},
  note      = {Under review at PVLDB Vol. 20},
  year      = {2027},
  url       = {https://github.com/shahrzadesmat/Agentic_VDMS}
}
```

> **Note:** This entry will be updated to `@article` with journal/volume/doi fields upon acceptance.
