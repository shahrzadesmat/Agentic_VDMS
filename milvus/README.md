# Milvus Replication Study

This directory contains a full replication of the paper's main comparison on [Milvus](https://milvus.io/) HNSW, demonstrating that the findings generalize beyond VDMS.

All five optimizer methods (LLM Agent, Optuna TPE, GP-BO, Random Search, VDTuner) are re-run across all three datasets × 3 seeds using `src/milvus/milvus_hico_optimizer.py`, `milvus_gldv2_optimizer.py`, and `milvus_sift1m_optimizer.py`. The LLM agent ranks first on all three datasets under Milvus (Table 7 in paper).

## Structure

```
milvus/
├── hico_det/
│   ├── results/          # JSON results (same schema as top-level results/)
│   │   ├── milvus_{method}_s{seed}.json   # main comparison (5 methods × 3 seeds)
│   │   ├── milvus_grid.json               # Grid Search (1 run)
│   │   └── milvus_qds_s42.json            # QDS objective variant (seed 42 only)
│   └── run_milvus_*.sh   # SLURM scripts mirroring experiments/hico_det/
├── gldv2/
│   ├── results/          # 5 methods × 3 seeds
│   └── run_milvus_*.sh
└── sift1m/
    ├── results/          # 5 methods × 3 seeds
    └── run_milvus_*.sh
```

## Running

Each script follows the same USER CONFIG pattern as `experiments/`. Edit `BASE_DIR`, `DATASET_DIR`, `CONTAINER`, and `PYTHON` at the top, then submit:

```bash
sbatch milvus/hico_det/run_milvus_llm_s42.sh
```

## Notes

- **`milvus_qds_s42.json`** — A single-seed run of the QDS (Query Difficulty-weighted Scoring) objective variant on Milvus, included for completeness. Not reported separately in the paper tables.
- Milvus result JSONs use a flat schema (`method`, `seed`, `iterations`, `map_threshold`, `best_score`, `best_map`, `best_qps`, `best_config`, `all_results`) rather than the nested `summary` / `results` structure used in the VDMS results.
