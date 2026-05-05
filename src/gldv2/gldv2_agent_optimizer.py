"""
LLM-Guided VDMS Hyperparameter Optimizer for GLDv2 Landmark Retrieval.

Ports hico_agent_optimizer.py (HICO-DET) to Google Landmarks Dataset v2
(GLDv2) with an image-to-image two-stage pipeline:
  Stage 1: CLIP ViT-L/14 image query → CLIP_Set (VDMS ANN, agent-optimized config)
  Stage 2: DINOv2 ViT-L/14-reg4 weighted-sum rerank (agent-tuned alpha)

Key differences from HICO-DET:
  - Queries are IMAGE embeddings (not text).  762K DB images (~17x larger).
  - No PMGD metadata → no constraint_strategy dimension.
  - n_refs and ref_strategy: back in search space (matches HICO-DET parameter set).
    Prior tests at M=32 efC=200 k=100 showed n_refs>1 hurt; interactions with k/efC unexplored.
  - AQE (Alpha Query Expansion) added to search space: n_aqe ∈ {1,3,5,10}, aqe_weight ∈ {0.1,0.2,0.3,0.5}.
  - HNSW index build ~4.5 min at 762K (batch_properties=500, fast).

Score function: QPS if mAP ≥ τ else 0 (SIEVE-style constrained, τ set via --map-threshold, default 0.15)

Agent tunes jointly:
  engine          : FaissHNSWFlat only (IVFFlat QPS~5 at 762K — excluded)
  M, efC, efS     : HNSW params
  k_neighbors     : Stage-1 candidate pool size (primary QPS lever)
  alpha           : DINOv2 rerank weight
  n_refs          : DINOv2 PRF reference count (1 = no PRF)
  ref_strategy    : PRF aggregation strategy (first / centroid / diverse)

Usage:
    python gldv2_agent_optimizer.py \\
        --port 55596 \\
        --dataset-dir /work/hdd/bdjd/vdms/datasets \\
        --method hyperparameter_only \\
        --iterations 30 \\
        --output gldv2_llm_results.json
"""

import sys
import time
import json
import random
import asyncio
import re
import os
import argparse
import subprocess
import requests
import numpy as np
import vdms
import optuna
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_gldv2 import (
    load_gldv2_data,
    setup_clip_index,
    run_gldv2_queries,
)


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================


# Constrained optimization threshold τ (SIEVE-style): Score = QPS if mAP ≥ τ else 0.
# Set at startup from --map-threshold CLI arg. Default 0.15 (above Grid baseline).
_MAP_THRESHOLD: float = 0.15


@dataclass
class BenchmarkResult:
    qps:             float
    map_score:       float   # mAP over all GLDv2 queries
    latency_ms:      float   # avg total wall-clock per query (Stage-1 + Stage-2), ms
    t_clip_avg_ms:   float   # avg Stage-1 VDMS FindDescriptor time per query, ms
    t_rerank_avg_ms: float   # avg Stage-2 DINOv2 numpy rerank time per query, ms
    index_build_s:   float   # CLIP_Set index build time, seconds (one-time cost)
    precision_at_10: float
    ndcg_at_10:      float
    recall_at_10:    float
    engine:          str
    params:          Dict
    k_neighbors:     int
    alpha:           float
    timestamp:       float
    per_query_ap:    List[float] = field(default_factory=list)

    def score(self) -> float:
        """Constrained optimization (SIEVE-style): Score = QPS if mAP ≥ τ else 0."""
        if float(self.map_score) < _MAP_THRESHOLD:
            return 0.0
        return float(self.qps)


@dataclass
class IterationResult:
    iteration:     int
    config:        Dict
    benchmark:     BenchmarkResult
    llm_reasoning: str
    search_method: str


# ===========================================================================
# DATA CACHE (avoid reloading ~3 GB per iteration)
# ===========================================================================

_GLDV2_CACHE: Dict = {}


def _load_gldv2_data_cached(dataset_dir: str) -> Dict:
    global _GLDV2_CACHE
    if not _GLDV2_CACHE:
        _GLDV2_CACHE = load_gldv2_data(dataset_dir)
    return _GLDV2_CACHE


# ===========================================================================
# CONFIG PROVIDER
# ===========================================================================

class ConfigProvider:
    ENGINES = ["FaissFlat", "FaissHNSWFlat", "FaissIVFFlat"]

    # ---------------------------------------------------------------
    # CANONICAL SEARCH SPACE — identical for random, grid, and LLM agent.
    # Every value here must appear verbatim in the LLM system prompt and
    # in grid_search_configs().  Do not add values to only one place.
    # ---------------------------------------------------------------
    ENGINE_PARAMS = {
        "FaissFlat": {},
        "FaissHNSWFlat": {
            "M":              [8, 16, 32, 48, 64],
            "efConstruction": [100, 200, 400],              # build quality: higher=better graph=higher recall
            "efSearch":       [32, 64, 100, 150, 200, 300, 500],
        },
        "FaissIVFFlat": {
            "nlist":  [256, 512, 1024, 2048],               # tuned for 762K vectors
            "nprobe": [4, 8, 16, 32, 64],
        },
    }

    K_NEIGHBORS_VALUES   = [50, 100, 150, 200, 300, 500, 750, 1000]
    ALPHA_VALUES         = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                            0.92, 0.95]
    N_REFS_VALUES        = [1, 3, 5, 10]
    REF_STRATEGY_VALUES  = ["first", "centroid", "diverse"]
    N_AQE_VALUES         = [1, 3, 5, 10]          # 1 = AQE disabled
    AQE_WEIGHT_VALUES    = [0.1, 0.2, 0.3, 0.5]   # only used when n_aqe > 1

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    # Practical engine set for random/Optuna: exclude IVFFlat (QPS~5 at 762K,
    # empirically useless) so all methods compare on an equal footing with Grid.
    SEARCH_ENGINES = ["FaissHNSWFlat"]

    def get_random_config(self) -> Dict:
        engine = self.rng.choice(self.SEARCH_ENGINES)
        params = self._sample_params(engine)
        ref_strategy = str(self.rng.choice(self.REF_STRATEGY_VALUES))
        # centroid uses ALL k candidates regardless of n_refs — normalize to 1 to avoid
        # generating configs that are identical in execution but look different in the dedup key.
        n_refs = 1 if ref_strategy == "centroid" else int(self.rng.choice(self.N_REFS_VALUES))
        n_aqe = int(self.rng.choice(self.N_AQE_VALUES))
        aqe_weight = float(self.rng.choice(self.AQE_WEIGHT_VALUES)) if n_aqe > 1 else 0.0
        return {
            "engine":       engine,
            "params":       params,
            "k_neighbors":  int(self.rng.choice(self.K_NEIGHBORS_VALUES)),
            "alpha":        float(self.rng.choice(self.ALPHA_VALUES)),
            "n_refs":       n_refs,
            "ref_strategy": ref_strategy,
            "n_aqe":        n_aqe,
            "aqe_weight":   aqe_weight,
        }

    def _sample_params(self, engine: str) -> Dict:
        params = {}
        if engine in self.ENGINE_PARAMS:
            for pname, pvals in self.ENGINE_PARAMS[engine].items():
                params[pname] = int(self.rng.choice(pvals))
        return params

    def grid_search_configs(self) -> List[Dict]:
        # Pure 1D axis sweep — ONE parameter varies at a time, all others at baseline.
        # NO cross-term combinations. LLM must discover joint interactions on its own.
        #
        # Baseline: M=32, efC=400, efS=500, k=100, α=0.90, n_refs=1, centroid, n_aqe=1, aqe_w=0.0
        #
        #   S1: M sweep (5)         M=8/16/32/48/64
        #   S3: k sweep (6)         k=50/100/150/200/300/500
        #   S4: alpha sweep (7)     α=0.0/0.30/0.50/0.70/0.80/0.85/0.95
        #   S5: n_refs/first (3)    (3,first)/(5,first)/(10,first)
        #   S6: ref_strategy (3)    (1,centroid)/(3,diverse)/(5,diverse)
        #   S7: AQE sweep (6)       n_aqe/aqe_w combos at baseline config
        #
        # Total: 5+6+7+3+3+6 = 30
        configs = []

        def _gc(**kw):
            kw.setdefault("n_aqe", 1)
            kw.setdefault("aqe_weight", 0.0)
            return kw

        BASE = dict(engine="FaissHNSWFlat",
                    params={"M": 32, "efConstruction": 400, "efSearch": 500},
                    k_neighbors=100, alpha=0.90, n_refs=1, ref_strategy="centroid")

        # ---- S1: M sweep (5) ----
        for M in [8, 16, 32, 48, 64]:
            configs.append(_gc(**{**BASE,
                "params": {"M": M, "efConstruction": 400, "efSearch": 500}}))

        # ---- S3: k sweep (6) ----
        for k in [50, 100, 150, 200, 300, 500]:
            configs.append(_gc(**{**BASE, "k_neighbors": k}))

        # ---- S4: alpha sweep (7) ----
        for alpha in [0.0, 0.30, 0.50, 0.70, 0.80, 0.85, 0.95]:
            configs.append(_gc(**{**BASE, "alpha": alpha}))

        # ---- S5: n_refs sweep with first (3) ----
        for n_refs in [3, 5, 10]:
            configs.append(_gc(**{**BASE, "n_refs": n_refs, "ref_strategy": "first"}))

        # ---- S6: ref_strategy sweep (3) ----
        for n_refs, ref_strategy in [(1, "centroid"), (3, "diverse"), (5, "diverse")]:
            configs.append(_gc(**{**BASE, "n_refs": n_refs, "ref_strategy": ref_strategy}))

        # ---- S7: AQE sweep (6) — n_aqe/aqe_weight at baseline ----
        for n_aqe, aqe_weight in [(3, 0.1), (3, 0.3), (5, 0.2), (5, 0.5), (10, 0.2), (10, 0.5)]:
            configs.append({**BASE, "n_aqe": n_aqe, "aqe_weight": aqe_weight})

        # Total: 5+6+7+3+3+6 = 30
        assert len(configs) == 30, f"Grid must have exactly 30 configs, got {len(configs)}"
        return configs


# ===========================================================================
# LLM PROVIDER
# ===========================================================================

class LLMProvider:
    def __init__(self, api_key: str, model: str = "minimax/minimax-m2.1",
                 seed: int = 42):
        self.api_key = api_key
        self.model   = model
        self.seed    = seed

    async def query_llm(self, iteration: int, history: List[IterationResult],
                        total_iterations: int,
                        consecutive_alpha_only: int = 0,
                        iters_since_improvement: int = 0,
                        dedup_hint: str = "") -> Tuple[Dict, str]:

        exp_end  = max(1, int(round(total_iterations * 0.40)))
        expl_end = max(exp_end + 1, int(round(total_iterations * 0.75)))
        phase = ("EXPLORATION" if iteration <= exp_end else
                 "EXPLOITATION" if iteration <= expl_end else "FINE-TUNING")

        current_best_score = max(r.benchmark.score() for r in history) if history else 0.0

        if history:
            last_run    = history[-1]
            last_score  = last_run.benchmark.score()
            last_map    = last_run.benchmark.map_score
            last_qps    = last_run.benchmark.qps
            last_k      = last_run.config.get("k_neighbors", 200)
            last_alpha  = last_run.config.get("alpha", 0.0)
            last_engine = last_run.config["engine"]
            last_params = last_run.config.get("params", {})

            last_n_refs      = last_run.config.get("n_refs", 1)
            last_ref_strategy= last_run.config.get("ref_strategy", "first")

            if last_score >= current_best_score * 0.98:
                if phase == "FINE-TUNING":
                    guidance = (
                        f"EXCELLENT. Near peak. {last_engine} k={last_k} alpha={last_alpha:.2f} "
                        f"n_refs={last_n_refs} ref_strategy={last_ref_strategy}. "
                        f"Score={last_score:.4f}. Fine-tuning: change ONE parameter by ONE step: "
                        f"alpha +/-0.05, OR k_neighbors one step, OR try n_refs/ref_strategy variant.")
                else:
                    guidance = (
                        f"Near best so far ({last_engine} k={last_k} alpha={last_alpha:.2f} "
                        f"n_refs={last_n_refs} Score={last_score:.4f}). Phase={phase}: do NOT fine-tune yet. "
                        f"KEEP EXPLORING — try a structurally DIFFERENT region: "
                        f"ref_strategy=centroid n_refs=1 at k=150/200 (centroid uses ALL k — ALWAYS n_refs=1, change k to vary PRF), "
                        f"OR n_refs=3/5 with ref_strategy=first or diverse (PRF with top-N candidates), "
                        f"OR efConstruction=400 (better graph quality, +1-3% recall, same QPS), "
                        f"OR M=48 efC=400 + centroid, OR alpha=0.92/0.95. "
                        f"PRIMARY unexplored: centroid at k=150-200, efC=400, n_refs×first/diverse.")
            elif last_map < _MAP_THRESHOLD:
                guidance = (
                    f"Score=0 (mAP={last_map:.4f} below threshold τ={_MAP_THRESHOLD:.2f}). "
                    f"RESET to known-safe baseline: HNSW M=32 efC=400 efS=500 k=50 α=0.90 n_refs=1/centroid "
                    f"→ reliably achieves mAP≈0.161 (above τ), QPS≈273. "
                    f"Do NOT try to salvage the current config — it is in an infeasible region. "
                    f"After re-establishing feasibility at k=50, tune α=0.92/0.95 and ref_strategy "
                    f"to maximize QPS while staying above τ.")
            elif last_map >= 0.05 and last_qps < 2.0:
                guidance = (
                    f"mAP acceptable ({last_map:.4f}) but QPS critically low ({last_qps:.1f}). "
                    f"This is likely FaissFlat or very large K. Switch to HNSW M=32 efS=500 "
                    f"with k=100-200. Do NOT reduce efSearch — it does not help QPS at 762K.")
            elif last_score < current_best_score * 0.90:
                guidance = (
                    f"Score={last_score:.4f} significantly below best={current_best_score:.4f}. "
                    f"Last config: {last_engine} k={last_k} alpha={last_alpha:.2f} "
                    f"n_refs={last_n_refs} ref_strategy={last_ref_strategy} params={last_params}. "
                    f"Return closer to best config and make targeted adjustments.")
            else:
                guidance = (
                    f"Score={last_score:.2f} (QPS). mAP={last_map:.4f} (≥τ={_MAP_THRESHOLD:.2f} ✓). QPS={last_qps:.1f}. "
                    f"mAP floor is met — now maximize QPS. Primary lever: reduce k_neighbors. "
                    f"Remember: efSearch barely affects QPS at 762K scale — do NOT reduce it for QPS. "
                    f"Real levers: try ref_strategy=centroid n_refs=1 at k=150/200 (centroid PRF — n_refs MUST be 1), "
                    f"OR n_refs=3/5 with ref_strategy=first/diverse, "
                    f"efConstruction=400 (zero QPS cost, better recall), or alpha=0.92/0.95.")
        else:
            guidance = (
                f"First iteration — cold start. Objective: Score = QPS if mAP ≥ τ={_MAP_THRESHOLD:.2f} else 0. "
                "Feasibility floor τ must be met first; then maximize QPS. "
                "CRITICAL PHYSICS (empirically measured in batch mode): "
                "At 762K scale, all 1,129 queries are issued in ONE batch VDMS call. "
                "k IS the dominant QPS lever — measured k→QPS curve: "
                "k=50→272 QPS, k=100→115 QPS, k=150→76 QPS, k=200→51 QPS, k=300→28 QPS, k=500→13 QPS. "
                "efSearch has MINIMAL impact on QPS in batch mode (VDMS amortises overhead). Use efS=500. "
                "FaissIVFFlat: AVOID — QPS~5 regardless of k due to PMGD overhead. "
                "TAU BOUNDARY WARNING: k=50 is the QPS sweet spot (~272 QPS) BUT mAP is sensitive to "
                "graph quality at k=50. M=32 efC=400 k=50 → mAP≈0.161 (safe, 1.1pp above τ=0.15). "
                "Lower M or efC at k=50 risks mAP<τ → score=0. Always pair k=50 with M=32 efC≥200. "
                "TARGET mAP≥0.17 to maintain a safety buffer above τ=0.15. "
                "STEP 1 — SAFE BASELINE (start here): "
                "HNSW M=32 efC=400 efS=500 k=50 α=0.90 n_refs=1/centroid → mAP≈0.161, QPS≈273. "
                "STEP 2 — FINE-TUNE AT k=50: k=50 is the minimum available — QPS is already near peak. "
                "Tune alpha=0.92/0.95, ref_strategy=centroid/diverse, n_refs=3/5 for marginal gains. "
                "CRITICAL: centroid ALWAYS requires n_refs=1. Change k to vary PRF quality. "
                "START: HNSW M=32 efC=400 efS=500 k=50 α=0.90 n_refs=1/centroid.")

        system_context = f"""
SYSTEM ARCHITECTURE:
1. TARGET SYSTEM: VDMS (Visual Data Management System)
   - C++ middleware for persistent multimodal vector storage.
   - Indexing Backends: FaissHNSWFlat, FaissIVFFlat, FaissFlat.
   - Communication: Client-Server via TCP sockets.
   - Indexing: batch_properties=500 vectors per AddDescriptor call (FAISS bulk insert).
     762,460 vectors → 1,525 batch calls. PMGD ops ≈ 1001 per call, well within 2MB journal limit.

2. DATASET: Google Landmarks Dataset v2 (GLDv2)
   - 762,460 DB images indexed in VDMS (CLIP ViT-L/14, 768-d, L2-normed).
   - 1,129 query images (image-to-image retrieval — NOT text queries).
   - Ground truth: positive and junk image sets per query (Oxford/Paris style).
   - NOTE: This is 17x larger than HICO-DET (47K images). Engine behavior differs at this scale.

3. RETRIEVAL PIPELINE (TWO STAGES — you tune BOTH stages simultaneously):

   Stage 1 — VDMS ANN Search  [CLIP ViT-L/14 — Image-to-Image Semantic Retrieval]:
     WHAT CLIP IS: A vision-language model. Even without text, it maps images into a 768-d
       semantic space where visually and conceptually similar images cluster together.
     WHAT IT IS GOOD AT: Finding images that depict the SAME landmark or scene category.
       e.g., query of the Eiffel Tower → returns images of the Eiffel Tower from all angles.
       It operates at the concept/category level and understands visual semantics.
     WHAT IT IS NOT: A fine-grained visual descriptor. It finds the right semantic neighborhood
       but may struggle with precise instance-level matching (e.g., specific building facade).
     Query: 768-d CLIP image embedding of the query landmark image
     Index: CLIP_Set (you choose engine + engine params)
     Output: top-K candidate images (you choose k_neighbors)
     Cost: ANN search cost amortised across all 1,129 queries in one batch call. IVFFlat ~185ms/query — avoid.

   Stage 2 — DINOv2 Rerank  [DINOv2 ViT-L/14-reg4 — Visual Appearance Refinement]:
     WHAT DINOv2 IS: A self-supervised vision model with strong spatial/local feature understanding.
       It encodes fine-grained visual appearance (textures, patterns, building details) into 1024-d.
     WHAT IT IS GOOD AT: Distinguishing SAME-instance images from same-category distractors.
       For landmarks, it excels at ranking images of the EXACT building/location over similar ones.
     WHY FUSION HELPS: CLIP finds semantically correct images (same landmark category) but may
       rank lookalike buildings too highly. DINOv2 corrects that via fine-grained appearance.
     NOTE: n_refs>1 tested at M=32 efC=200 k=100 hurt performance. ROOT CAUSE HYPOTHESIS:
       k=100 → ~18 true positives in top-100 on average → centroid is dominated by 82 false positives
       → corrupted DINOv2 query. At k=150-200, more true positives in pool → cleaner centroid.
       Also: efC=200 graph misses some true positives → efC=400 may fix PRF input quality.
       CENTROID RULE: centroid uses ALL k candidates regardless of n_refs. ALWAYS use n_refs=1 with centroid.
       Change k (not n_refs) to vary centroid PRF quality.
       PRIMARY UNEXPLORED HYPOTHESIS: centroid n_refs=1 at k=150-200 with efC=400.
     Fuse: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm  (WEIGHTED SUM, lower=better)
       where clip_norm   = CLIP L2 distance normalized to [0,1] within K candidates
             dinov2_norm = DINOv2 L2 distance normalized to [0,1] within K candidates
     alpha=0 → pure CLIP semantic ranking.
     alpha=1 → pure DINOv2 visual ranking (not in allowed list).
     Cost: ~5-15ms/query (negligible vs Stage-1 VDMS cost).

4. OPTIMIZATION OBJECTIVE: Constrained maximization (SIEVE-style)
   Score = QPS  if mAP ≥ {_MAP_THRESHOLD:.2f}  else  0
   - The threshold τ={_MAP_THRESHOLD:.2f} is the quality floor. Configs below it score 0 (infeasible).
   - Above the floor, maximize QPS (throughput). This prevents degenerate low-mAP/high-QPS solutions.
   - Standard baseline: HNSW M=32 efC=400 efS=500 k=50 α=0.90 → mAP≈0.161, QPS≈273.
   Current best score in this session: {current_best_score:.2f}

5. SCALE NOTES (GLDv2 vs HICO-DET) — EMPIRICALLY MEASURED:
   - 762K DB images (vs 47K in HICO-DET). All 1,129 queries issued in ONE batch VDMS call.
     * Batch mode amortises socket overhead — QPS now reflects pure ANN search cost.
     * FaissFlat exact search: low QPS at any k (brute-force over 762K). Recall ceiling only.
     * FaissHNSWFlat M=32: efSearch has MINIMAL impact on QPS (VDMS overhead dominates within batch).
       Always use efS=500 for maximum CLIP recall at negligible QPS cost.
     * FaissIVFFlat: DO NOT USE. QPS~5 due to PMGD transaction overhead at 762K.
     * k_neighbors is the PRIMARY QPS lever: smaller k = faster batch = higher throughput.
     * HNSW INDEX BUILD TIME: M=32 → ~4.5 min; M=64 → ~8 min (batch_properties=500). Build is one-time per iter.
       PREFER M=32 for most iterations. M=48/64 can be tried later.
"""

        engine_definitions = f"""
PARAMETER DEFINITIONS AND PHYSICS:

============================================================
GROUP A — STAGE-1 ENGINE PARAMETERS (affect VDMS search speed and CLIP recall)
============================================================

[A1. engine — Index Type]
  Choices: FaissHNSWFlat | FaissIVFFlat | FaissFlat
  - FaissFlat:     exact brute-force. Highest recall at any K. No tunable params.
                   At 762K scale: QPS~1-5 depending on K. Useful for recall ceiling measurement.
                   Try with small K (100-200) for high QPS.
  - FaissHNSWFlat: graph-based. THE ONLY PRACTICAL ENGINE at 762K scale.
                   efSearch barely affects QPS here (fixed VDMS overhead dominates). Use efS=500.
                   BUILD NOTE: M=32 at 762K takes ~4.5 min (batch_properties=500).
  - FaissIVFFlat:  DO NOT USE. Empirically gives QPS~5 at 762K due to PMGD transaction overhead
                   (~185ms/query regardless of nprobe). Far slower than HNSW (~45ms).

[A2. M — HNSW graph connectivity] (FaissHNSWFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 8, 16, 32, 48, 64
  - Higher M = denser graph = higher CLIP recall at same efSearch.
  - Higher M = longer index build at 762K scale. Measured: M=32→4.5 min, M=48→6 min, M=64→8 min.
  - RECOMMENDED: M=32 for early baseline. Try M=48/64 once good k/alpha/efC found.
  - RELATIONSHIP WITH efConstruction: Higher M requires higher efC to build a good graph.

[A3. efConstruction — HNSW build quality] (FaissHNSWFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 100, 200, 400
  - Controls the quality of the HNSW graph during index construction.
  - Higher efC = better-connected graph = higher recall at the same efSearch and M.
  - efC=100: ~3 min build at 762K. Faster but graph may miss some shortcuts → lower recall.
  - efC=200: ~4.5 min build. Standard quality. SIEVE baseline: k=100 α=0.90 → Score≈115 QPS in batch mode (mAP≈0.179 ≥ τ).
  - efC=400: ~8 min build. Best graph quality — recall improvement over efC=200 is 1-3% absolute.
  - RECOMMENDATION: after establishing k/alpha baseline, try efC=400 to see if better graph helps.
  - INTERACTION WITH M: higher M benefits more from higher efC (more edges to optimize).
    M=48 efC=400 may outperform M=32 efC=200 for recall at same QPS.

[A4. efSearch — HNSW search depth] (FaissHNSWFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 32, 64, 100, 150, 200, 300, 500
  - At 762K scale in batch mode: efSearch has MINIMAL impact on QPS (VDMS overhead dominates within batch).
    Variation across efS=32..500 is small — the ANN graph traversal cost is not the bottleneck.
  - RECOMMENDATION: ALWAYS use efSearch=500 for maximum CLIP recall at negligible QPS cost.
  - Do NOT reduce efSearch to gain QPS — it does not work at this scale.

[A5. nlist — IVF cluster count] (FaissIVFFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 256, 512, 1024, 2048
  - Tuned for 762K vectors. Sqrt(762K) ≈ 873, so nlist=512-1024 is the target range.
  - Higher nlist = faster search (fewer items per cell) but lower recall at same nprobe.
  - RELATIONSHIP WITH nprobe: always tune nprobe alongside nlist.

[A6. nprobe — IVF cells visited] (FaissIVFFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 4, 8, 16, 32, 64
  - QPS ∝ 1/nprobe (linear relationship).
  - Recall increases with nprobe. nprobe=nlist → exact search.
  - For nlist=512 use nprobe=8-32. For nlist=1024 use nprobe=16-64.

============================================================
GROUP B — STAGE-2 RERANKING PARAMETERS (affect DINOv2 fusion)
============================================================

[B1. k_neighbors — Candidate pool size]
  ALLOWED VALUES (you MUST choose exactly one): 50, 100, 150, 200, 300, 500, 750, 1000
  - Stage-1 retrieves top-K from VDMS. Stage-2 reranks exactly these K images.
  - LARGER K = higher mAP ceiling (more true positives can surface through reranking).
  - k IS THE PRIMARY QPS LEVER at 762K scale in batch mode.
    Smaller k = fewer neighbors computed per query in the batch = higher throughput.
    Larger k = more candidates for AQE and DINOv2 reranking = higher mAP ceiling.
  - NOTE: efSearch does NOT affect QPS meaningfully — VDMS overhead dominates even in batch mode.
  - SIEVE reference: k=100 α=0.90 → Score≈batch QPS when mAP≥τ (exact number established at iter 1).
  - k=50 is high priority for QPS gain; k=150 is worth testing for AQE reference quality.
    Always re-tune alpha when changing k.
  - RELATIONSHIP WITH centroid: centroid uses ALL k candidates. At k=50, centroid gets fewer
    candidates but query is faster. Test k=50 centroid n_refs=1 α=0.85 as first priority.
  - RELATIONSHIP WITH alpha: k=50 centroid may need slightly lower alpha (0.80-0.85) since
    fewer candidates make the centroid estimate noisier.

[B2. alpha — DINOv2 rerank weight in Weighted Sum]
  ALLOWED VALUES (you MUST choose exactly one):
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95
  - Fusion formula: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm  (lower=better)
  - For LANDMARK retrieval, DINOv2 is especially useful: CLIP finds the right landmark category
    but DINOv2 distinguishes the exact building/location from near-duplicates.
  - SIEVE reference: α=0.90 gives Score≈115 QPS at k=100 in batch mode (mAP≈0.179). α=0.85 slightly lower mAP — both meet τ.
  - The optimal alpha near the optimum region (0.85-0.95) is still unknown precisely.

[B3. n_refs — DINOv2 Pseudo-Relevance Feedback (PRF) count]
  ALLOWED VALUES (you MUST choose exactly one): 1, 3, 5, 10
  - n_refs=1: use query image DINOv2 feature directly (standard, no PRF).
  - n_refs>1: augment the query DINOv2 vector by averaging it with DINOv2 features of
    the top-(n_refs-1) Stage-1 CLIP candidates. This creates an enriched query vector.
  - RULE: n_refs>1 with first/diverse PROVEN BAD (all scored <3.1 in prior 30-iteration run).
    Do NOT use n_refs>1 with first or diverse strategy. Always use n_refs=1.
  - CENTROID RULE: centroid uses ALL k candidates. n_refs is IGNORED — ALWAYS use n_refs=1 with centroid.
    To vary centroid PRF quality, change k (not n_refs): k=50 vs k=100 centroid n_refs=1.
  - INTERACTION WITH k: k=50 centroid (QPS gain) and k=150 centroid (more candidates for high-H queries) are both worth testing.
  - INTERACTION WITH efC: better graph (efC=400) → more relevant Stage-1 candidates → better recall.

[B4. ref_strategy — PRF aggregation strategy]
  ALLOWED VALUES (you MUST choose exactly one): "first", "centroid", "diverse"
  - "first":    use top-(n_refs-1) candidates by CLIP distance rank. Default.
  - "centroid": use centroid of ALL Stage-1 CLIP candidates' DINOv2 features.
                Captures the "average look" of the retrieved set. Best when Stage-1 is precise.
  - "diverse":  greedy farthest-point subset — maximizes coverage of visual space.
                Best when the target landmark has diverse appearances (e.g., many viewpoints).
  - CRITICAL REDUNDANCY + IMPLEMENTATION NOTE:
    "centroid" uses ALL k candidates regardless of n_refs — the code ALWAYS stores centroid
    configs with n_refs=1 in history, regardless of what n_refs you suggest.
    THIS MEANS: if you see "n_refs=1 centroid k=150" in history, it means centroid at k=150
    was already tested — do NOT suggest n_refs=3/5/10 centroid at the same k, it will be
    normalized to n_refs=1 and rejected as a duplicate, wasting your iteration budget.
    RULE: Always suggest n_refs=1 when using ref_strategy=centroid.
    To vary centroid behavior, change k (more candidates = better centroid quality).
  - "first" and "diverse" use exactly n_refs-1 candidates — here n_refs DOES matter.
  - RECOMMENDED: centroid is the most stable choice for landmark retrieval.

[B5. n_aqe / aqe_weight — ALPHA QUERY EXPANSION (AQE)]
  AQE expands the CLIP query by blending it with the mean CLIP embedding of the top-(n_aqe-1)
  retrieved images from Stage-1 (pseudo-relevance feedback):
    q_expanded = (1 - aqe_weight) * q_original + aqe_weight * mean(clip_db[top_(n_aqe-1)])
  This sharpens the query toward the visual cluster of retrieved images — proven effective for landmarks.

  n_aqe:       [1, 3, 5, 10]   — 1 = AQE disabled; >1 = expand using top-(n_aqe-1) images
  aqe_weight:  [0.1, 0.2, 0.3, 0.5]  — blend strength (ONLY used when n_aqe > 1)
  RULE: n_aqe=1 → aqe_weight MUST be 0.0. n_aqe>1 → choose aqe_weight from [0.1, 0.2, 0.3, 0.5].
  GUIDANCE:
    - n_aqe=5, aqe_weight=0.2 is a safe starting point (moderate feedback, 4 reference images)
    - Higher n_aqe = stronger feedback; risk: drift if top-k retrieval contains many distractors
    - Higher aqe_weight = stronger blend toward retrieved cluster
    - AQE requires k ≥ n_aqe (always true given k≥50 and n_aqe≤10)
    - AQE + centroid ref_strategy: AQE sharpens Stage-1 query → centroid aggregates better Stage-2 refs
    - AQE + higher alpha: improved Stage-1 → more reliable DINOv2 reranking signal

============================================================
CROSS-PARAMETER INTERACTION SUMMARY (READ CAREFULLY):
============================================================
  efSearch       → DOES NOT AFFECT QPS at 762K scale in batch mode. Always use efS=500 (free max recall).
  efConstruction → AFFECTS RECALL (not QPS). Higher efC = better HNSW graph = more true positives in top-k.
                   efC=400 costs ~8 min build but may push mAP from baseline by 1-3pp. Worth testing.
  IVFFlat        → AVOID. QPS~5 regardless of nprobe due to PMGD overhead at 762K.
  centroid × k   → centroid uses ALL k candidates (n_refs IGNORED — always use n_refs=1 with centroid).
                   k=50: lower QPS cost, fewer centroid candidates. k=150: more candidates for better centroid quality.
  n_refs × k     → for first/diverse ONLY: n_refs>1 with first/diverse ALL scored <3.1 in prior runs (proven bad).
                   Do NOT use n_refs>1 with first or diverse strategy. centroid always uses n_refs=1.
  n_refs × efC   → KEY INTERACTION: efC=400 gives better Stage-1 candidates → better graph recall.
                   (centroid n_refs=1, M=32/48, efC=400, k=50/100) is a promising joint hypothesis.
  ref_strategy   → "centroid" aggregates all k candidates; most stable for landmark retrieval.
                   "diverse" good when landmark has many visual appearances (varied viewpoints).
  n_aqe × k      → AQE uses top-(n_aqe-1) images; k must be ≥ n_aqe (always true here).
                   More k candidates → better AQE reference pool → stronger feedback.
  n_aqe × alpha  → AQE improves Stage-1 recall → cleaner centroid → higher alpha may work better.
  n_aqe=1        → AQE disabled, aqe_weight must be 0.0.
  K ↓            → QPS ↑ (fewer neighbors per query in batch = faster). Exact numbers established at iter 1.
  K ↑            → mAP ↑, QPS ↓. k=150/200 may improve mAP and AQE quality enough to keep Score high — test empirically.
  alpha × k      → At k=50 (fewer centroid candidates), the centroid estimate is noisier → optimal
                   alpha may shift slightly down (0.80-0.85) vs k=100 optimum (0.85-0.90).
                   Always re-tune alpha when changing k.
  efC × M        → M=48 efC=400 may outperform M=32 efC=200 for recall at same QPS.
"""

        scenario_guide = f"""
PHASE-SPECIFIC STRATEGY (current phase: {phase}):

EXPLORATION (iterations 1-{exp_end}): Establish baseline, then explore AQE, k variations, and alpha.
  - PRIORITY 1: k=100 α=0.90 n_refs=1 centroid M=32 efC=400 efS=500 n_aqe=1 aqe_w=0.0 — establish baseline score.
  - PRIORITY 2: n_aqe=5 aqe_weight=0.2 at baseline config — AQE sharpens Stage-1 query (moderate feedback).
  - PRIORITY 3: n_aqe=5 aqe_weight=0.5 — stronger AQE blend; test if pseudo-relevance helps or drifts.
  - PRIORITY 4: k=50 n_refs=1 centroid α=0.92 n_aqe=5 aqe_w=0.2 — higher QPS (~136 QPS est.) + AQE sharpening.
  - PRIORITY 5: k=150 n_refs=1 centroid α=0.90 n_aqe=5 aqe_w=0.2 — more candidates + AQE; test mAP gain.
  - PRIORITY 6: n_aqe=10 aqe_weight=0.2 — stronger feedback from top-9 images.
  - AVOID: IVFFlat (QPS~5 at 762K), k≥300 (QPS too low), n_aqe=1 with aqe_weight>0 (invalid).
  - Goal: determine if AQE improves score and which n_aqe/aqe_weight/k/alpha combination is optimal.

EXPLOITATION (iterations {exp_end+1}-{expl_end}): Narrow around best config found.
  - Fix engine/M/efC to best found values.
  - Sweep AQE at best (k, alpha, ref_strategy): try n_aqe ∈ {{3,5,10}} × aqe_weight ∈ {{0.1,0.2,0.3,0.5}}.
  - Sweep alpha ±0.05 around best alpha found.
  - Sweep k within {{50, 100, 150}} at best AQE config.
  - Goal: find exact (k, alpha, n_aqe, aqe_weight, ref_strategy) optimum.

FINE-TUNING (iterations {expl_end+1}-{total_iterations}): ONE parameter change at a time.
  - Change only ONE of: alpha, n_aqe, aqe_weight, k, M, ref_strategy.
  - Goal: squeeze final improvement from AQE × alpha × k interactions.

SCENARIO DECISION MATRIX:

1. "Establish baseline" → k=100 n_refs=1 centroid M=32 efC=400 efS=500 α=0.90 n_aqe=1 aqe_w=0.0
   Run FIRST. n_aqe=1 disables AQE — baseline gives ~115 QPS (batch mode, k=100). AQE adds a second batch
   call (~2× overhead), so AQE must meaningfully raise mAP to keep Score competitive vs no-AQE.

2. "Moderate AQE" → n_aqe=5 aqe_weight=0.2 at best (k, alpha) — safe starting point for AQE.
   5 images for expansion, moderate blend. Should improve Stage-1 precision → better centroid.

3. "Strong AQE" → n_aqe=10 aqe_weight=0.3 — more pseudo-relevance images, stronger feedback.
   Risk: if top-10 CLIP results contain distractors, AQE may drift. Test after moderate AQE.

4. "AQE + QPS gain" → k=50 n_aqe=5 aqe_weight=0.2 α=0.92 centroid — ~136 QPS est. (2 batches at k=50) + AQE sharpening.
   AQE compensates for reduced k (better query → recall maintained despite smaller pool).

5. "AQE + more candidates" → k=150 n_aqe=5 aqe_weight=0.2 α=0.90 centroid — more pool for AQE refs.
   n_aqe=5 at k=150 uses top-4 from 150 candidates → better AQE reference images.

6. AVOID:
   - IVFFlat (QPS~5 at 762K), k≥300 (est. QPS ~6 at k=500 with AQE — too slow)
   - n_aqe=1 with aqe_weight>0 (invalid — AQE requires n_aqe>1 to activate)
   - n_refs>1 with first or diverse (proven bad in prior runs — scored below τ or very low QPS)
"""

        def _snap_note(cfg):
            notes = []
            if "_raw_alpha" in cfg:
                notes.append(f"sugg_a={cfg['_raw_alpha']:.2f}")
            if "_raw_k" in cfg:
                notes.append(f"sugg_k={cfg['_raw_k']}")
            if "_raw_n_aqe" in cfg:
                notes.append(f"sugg_aqe={cfg['_raw_n_aqe']}")
            return f" [{', '.join(notes)}]" if notes else ""

        all_configs_tried = "\n".join(
            f"  iter{h.config.get('_iter','?'):02d}: {h.config['engine']} "
            f"p={h.config.get('params',{})} k={h.config.get('k_neighbors',200)} "
            f"a={h.config.get('alpha',0.0):.2f} "
            f"refs={h.config.get('n_refs',1)}/{h.config.get('ref_strategy','first')} "
            f"aqe={h.config.get('n_aqe',1)}/{h.config.get('aqe_weight',0.0):.1f}"
            f"{_snap_note(h.config)} → Score={h.benchmark.score():.4f} "
            f"mAP={h.benchmark.map_score:.4f} QPS={h.benchmark.qps:.1f}"
            for h in history
        )

        untried_alpha_hint = ""
        untried_refs_hint  = ""
        if history and phase in ("EXPLOITATION", "FINE-TUNING"):
            best_h = max(history, key=lambda r: r.benchmark.score())
            best_engine      = best_h.config["engine"]
            best_k           = best_h.config.get("k_neighbors", 200)
            best_params      = best_h.config.get("params", {})
            best_alpha       = best_h.config.get("alpha", 0.0)
            best_n_refs      = best_h.config.get("n_refs", 1)
            best_ref_strategy= best_h.config.get("ref_strategy", "first")

            tried_alphas_here = {
                round(r.config.get("alpha", 0.0), 2)
                for r in history
                if r.config["engine"] == best_engine
                and r.config.get("k_neighbors", 200) == best_k
                and r.config.get("params", {}) == best_params
                and r.config.get("n_refs", 1) == best_n_refs
                and r.config.get("ref_strategy", "first") == best_ref_strategy
            }
            untried_alphas = [a for a in ConfigProvider.ALPHA_VALUES
                              if round(a, 2) not in tried_alphas_here]
            if untried_alphas and phase == "FINE-TUNING":
                untried_alpha_hint = (
                    f"\nUNTRIED alpha values at best config "
                    f"({best_engine} k={best_k} n_refs={best_n_refs} "
                    f"ref_strategy={best_ref_strategy} params={best_params}): {untried_alphas}"
                    f"\n  → These are valid next probes. Pick one if you want to tune alpha."
                )

            tried_refs_here = {
                (r.config.get("n_refs", 1), r.config.get("ref_strategy", "first"))
                for r in history
                if r.config["engine"] == best_engine
                and r.config.get("k_neighbors", 200) == best_k
                and r.config.get("params", {}) == best_params
                and round(r.config.get("alpha", 0.0), 2) == round(best_alpha, 2)
            }
            untried_refs = [
                (nr, rs)
                for nr in ConfigProvider.N_REFS_VALUES
                for rs in ConfigProvider.REF_STRATEGY_VALUES
                if (nr, rs) not in tried_refs_here
                and not (rs == "centroid" and nr > 1)  # centroid normalizes to n_refs=1
            ]
            if untried_refs:
                untried_refs_hint = (
                    f"\nUNTRIED (n_refs, ref_strategy) combos at best (engine/params/k/alpha): "
                    f"{untried_refs[:8]}"
                    f"\n  → These are valid next probes. "
                    f"centroid n_refs=1 (vary k) and n_refs=3-5 with first/diverse are the highest-value probes."
                )

        history_json = json.dumps([{
            "eng":   h.config["engine"],
            "p":     h.config.get("params", {}),
            "k":     h.config.get("k_neighbors", 200),
            "alp":   round(h.config.get("alpha", 0.0), 2),
            "refs":  h.config.get("n_refs", 1),
            "ref_s": h.config.get("ref_strategy", "first"),
            "n_aqe": h.config.get("n_aqe", 1),
            "aqe_w": round(h.config.get("aqe_weight", 0.0), 2),
            "sc":    round(h.benchmark.score(), 4),
            "map":   round(h.benchmark.map_score, 4),
            "qps":   round(h.benchmark.qps, 1),
        } for h in history[-5:]], indent=1)

        prompt = f"""You are an expert VDMS Optimization Engine for GLDv2 Landmark Retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

{system_context}

DIAGNOSIS OF LAST RUN:
{guidance}

{engine_definitions}

{scenario_guide}

ALL CONFIGS TRIED SO FAR (do NOT repeat any of these exact combinations):
{all_configs_tried if all_configs_tried else "  (none yet)"}
{untried_alpha_hint}{untried_refs_hint}{("⚠️  DEDUP REJECTION — YOUR LAST SUGGESTION WAS REJECTED:\n" + dedup_hint + "\nYou MUST suggest a genuinely different config.\n\n") if dedup_hint else ""}LAST 5 RESULTS (detailed):
{history_json}

AVAILABLE PARAMETER RANGES:
  FaissFlat:     params={{}}  (no tunable params — exact brute-force)
  FaissHNSWFlat: M={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['M']}
                 efConstruction={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efConstruction']}
                 efSearch={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efSearch']}
  FaissIVFFlat:  DO NOT USE (QPS~5 due to PMGD overhead at 762K)
  k_neighbors:          {ConfigProvider.K_NEIGHBORS_VALUES}
  alpha:                {ConfigProvider.ALPHA_VALUES}
  n_refs:               {ConfigProvider.N_REFS_VALUES}
  ref_strategy:         {ConfigProvider.REF_STRATEGY_VALUES}
  n_aqe:                {ConfigProvider.N_AQE_VALUES}  (1=disabled; >1=AQE active — expands query by blending top-(n_aqe-1) retrieved images)
  aqe_weight:           {ConfigProvider.AQE_WEIGHT_VALUES}  (blend strength: q_exp=(1-w)*q_orig+w*mean(top); only when n_aqe>1)

DECISION RULES (apply in order):
1. Read DIAGNOSIS — it tells you the single most important fix.
2. Use CROSS-PARAMETER INTERACTIONS to avoid breaking other metrics.
3. In EXPLORATION phase: probe centroid n_refs=1 at k=150/200, efConstruction=400, and their combination.
   CENTROID RULE: ALWAYS n_refs=1 with centroid. Do NOT suggest n_refs>1 with centroid.
4. In FINE-TUNING phase: change only ONE parameter by ONE step.
5. n_refs and ref_strategy are FREE to vary — they are primary search dimensions.
6. n_aqe and aqe_weight are FREE to vary — AQE can sharpen queries at cost of extra CLIP lookup.
   When n_aqe=1, aqe_weight MUST be 0.0. When n_aqe>1, set aqe_weight from {ConfigProvider.AQE_WEIGHT_VALUES}.
7. Never repeat a config already in HISTORY.
8. ANTI-COLLAPSE: {f"*** MANDATORY *** Your last {consecutive_alpha_only} consecutive iterations changed ONLY alpha (same engine/params/k/n_refs/ref_strategy). You MUST change efConstruction, OR M, OR k_neighbors, OR n_refs, OR ref_strategy in this iteration." if consecutive_alpha_only >= 2 else f"[{consecutive_alpha_only} consecutive alpha-only iter(s). Change efC/M/k/n_refs/ref_strategy if this reaches 2.]"}
9. STAGNATION: {f"⚠ WARNING: No improvement for {iters_since_improvement} iterations (best={current_best_score:.4f}). MANDATORY: Try a completely different (n_refs, ref_strategy) combination not yet tested, OR change k. Do NOT retry configs near the stuck point." if iters_since_improvement >= 5 else f"[{iters_since_improvement} iter(s) without improvement — OK]"}

OUTPUT JSON ONLY (no markdown, no explanation outside the JSON):
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 32, "efConstruction": 400, "efSearch": 500}},
  "k_neighbors": 150,
  "alpha": 0.90,
  "n_refs": 1,
  "ref_strategy": "centroid",
  "n_aqe": 5,
  "aqe_weight": 0.2,
  "reasoning": "AQE with n_aqe=5 sharpens the CLIP query by blending top-4 retrieved images before DINOv2 reranking. aqe_weight=0.2 is conservative — 80% original query preserved. k=150 gives a good AQE reference pool (n_aqe-1=4 images from 150 candidates). efC=400 for quality index."
}}
"""

        try:
            content = None
            for _api_attempt in range(3):
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model":      self.model,
                        "messages":   [{"role": "user", "content": prompt}],
                        "max_tokens": 2000,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if content:
                    break
                print(f"LLM Warning: Empty response on API attempt {_api_attempt+1}/3, retrying...")
                time.sleep(5)
            if not content:
                raise ValueError("Empty LLM response (content=None) after 3 attempts")

            # Robust JSON extraction
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            if not json_str.strip().startswith("{"):
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0)

            config_data = json.loads(json_str.strip())

            # Validation + snap-to-canonical layer.
            def snap(value, allowed):
                allowed = sorted(allowed)
                return min(allowed, key=lambda x: abs(x - value))

            provider = ConfigProvider()
            engine = config_data.get("engine")
            params = config_data.get("params", {})

            if engine not in provider.ENGINE_PARAMS:
                return provider.get_random_config(), f"Fallback (invalid engine: {engine})"

            allowed_params = provider.ENGINE_PARAMS[engine]
            clean_params = {}
            for k, allowed_vals in allowed_params.items():
                if k in params:
                    clean_params[k] = snap(int(params[k]), allowed_vals)
                elif allowed_vals:
                    clean_params[k] = allowed_vals[len(allowed_vals) // 2]

            k_neighbors  = snap(int(config_data.get("k_neighbors", 200)),
                                provider.K_NEIGHBORS_VALUES)
            alpha        = snap(float(config_data.get("alpha", 0.0)),
                                provider.ALPHA_VALUES)
            ref_strategy_raw = str(config_data.get("ref_strategy", "first"))
            ref_strategy = (ref_strategy_raw if ref_strategy_raw in provider.REF_STRATEGY_VALUES
                            else "first")
            n_refs_raw   = int(config_data.get("n_refs", 1))
            # centroid ignores n_refs — normalize to 1 to avoid dedup collisions
            n_refs = 1 if ref_strategy == "centroid" else snap(n_refs_raw, provider.N_REFS_VALUES)
            n_aqe_raw    = int(config_data.get("n_aqe", 1))
            n_aqe        = snap(n_aqe_raw, provider.N_AQE_VALUES)
            if n_aqe > 1:
                aqe_weight_raw = float(config_data.get("aqe_weight", 0.0))
                aqe_weight = snap(aqe_weight_raw, provider.AQE_WEIGHT_VALUES)
            else:
                aqe_weight = 0.0

            final_config = {
                "engine":       engine,
                "params":       clean_params,
                "k_neighbors":  k_neighbors,
                "alpha":        alpha,
                "n_refs":       n_refs,
                "ref_strategy": ref_strategy,
                "n_aqe":        n_aqe,
                "aqe_weight":   aqe_weight,
            }

            # Store raw suggested values so history can show what was snapped
            raw_alpha_val = float(config_data.get("alpha", 0.0))
            if abs(raw_alpha_val - alpha) > 1e-6:
                final_config["_raw_alpha"] = raw_alpha_val
            raw_k_val = int(config_data.get("k_neighbors", 200))
            if raw_k_val != k_neighbors:
                final_config["_raw_k"] = raw_k_val
            raw_naqe_val = int(config_data.get("n_aqe", 1))
            if raw_naqe_val != n_aqe:
                final_config["_raw_n_aqe"] = raw_naqe_val

            reasoning_text = config_data.get("reasoning", "(no reasoning field)")
            raw_engine  = config_data.get("engine", "?")
            raw_params  = config_data.get("params", {})
            raw_k       = config_data.get("k_neighbors", "?")
            raw_alpha_p = config_data.get("alpha", "?")
            snapped = []
            if raw_engine != engine:
                snapped.append(f"engine {raw_engine!r}→{engine!r}")
            for pk in clean_params:
                raw_v = raw_params.get(pk)
                if raw_v is not None and int(raw_v) != clean_params[pk]:
                    snapped.append(f"{pk} {raw_v}→{clean_params[pk]}")
            if isinstance(raw_k, (int, float)) and int(raw_k) != k_neighbors:
                snapped.append(f"k_neighbors {raw_k}→{k_neighbors}")
            if isinstance(raw_alpha_p, (int, float)) and abs(float(raw_alpha_p) - alpha) > 1e-6:
                snapped.append(f"alpha {raw_alpha_p}→{alpha}")
            snap_note = f"  [SNAPPED: {', '.join(snapped)}]" if snapped else ""
            print(f"\n[LLM] ── Suggested config ──────────────────────────────")
            print(f"[LLM]   engine={engine}  params={clean_params}")
            print(f"[LLM]   k_neighbors={k_neighbors}  alpha={alpha:.2f}  "
                  f"n_refs={n_refs}/{ref_strategy}  "
                  f"n_aqe={n_aqe}  aqe_weight={aqe_weight:.2f}{snap_note}")
            print(f"[LLM]   Reasoning: {reasoning_text}")
            print(f"[LLM] ────────────────────────────────────────────────────\n")

            return final_config, content

        except Exception as e:
            print(f"LLM Error: {e}")
            return ConfigProvider(seed=self.seed).get_random_config(), f"Error: {e}"


# ===========================================================================
# BENCHMARK EXECUTION
# ===========================================================================

def run_benchmark(port: int, dataset_dir: str, config: Dict,
                  iteration: int = 0) -> BenchmarkResult:
    """
    Connect to VDMS, build CLIP_Set, run two-stage image+DINOv2 retrieval,
    return BenchmarkResult with mAP and QPS.

    Uses a unique set name per iteration to avoid DeleteDescriptorSet issues.
    """
    engine        = config["engine"]
    engine_params = config.get("params", {})
    k_neighbors   = int(config.get("k_neighbors", 200))
    alpha         = float(config.get("alpha", 0.0))
    n_refs        = int(config.get("n_refs", 1))
    ref_strategy  = str(config.get("ref_strategy", "first"))
    n_aqe         = int(config.get("n_aqe", 1))
    aqe_weight    = float(config.get("aqe_weight", 0.0))
    set_name      = f"CLIP_Set_{iteration}"

    data = _load_gldv2_data_cached(dataset_dir)

    print(f"\n[Iter {iteration}] engine={engine} params={engine_params} "
          f"k={k_neighbors} alpha={alpha:.2f} n_refs={n_refs}/{ref_strategy} "
          f"n_aqe={n_aqe} aqe_w={aqe_weight:.2f}")

    db = vdms.vdms()
    db.connect("localhost", port)
    res, _ = db.query([{"GetSchema": {}}])
    if not res:
        raise RuntimeError(f"Cannot connect to VDMS on port {port}")

    print(f"[Iter {iteration}] Building {set_name} ...")
    t_build_start = time.time()
    setup_clip_index(db, data["clip_db"], engine, engine_params,
                     set_name=set_name)
    index_build_s = time.time() - t_build_start
    print(f"[Iter {iteration}] Index built in {index_build_s:.1f}s ({index_build_s/60:.1f}min)")

    print(f"[Iter {iteration}] Running {len(data['query_ids'])} queries...")
    metrics = run_gldv2_queries(
        db,
        data["clip_q"],
        data["dinov2_q"],
        data["dinov2_db"],
        data["rel_sets"],
        data["junk_sets"],
        alpha=alpha,
        k_neighbors=k_neighbors,
        n_refs=n_refs,
        ref_strategy=ref_strategy,
        clip_name=set_name,
        query_ids=data["query_ids"],
        clip_db=data["clip_db"],
        n_aqe=n_aqe,
        aqe_weight=aqe_weight,
    )

    print(f"[Iter {iteration}] Timing breakdown:")
    print(f"  Index build:      {index_build_s:.1f}s  (one-time)")
    print(f"  Query (Stage-1):  {metrics['t_clip_avg_ms']:.2f}ms/query  (VDMS ANN)")
    print(f"  Rerank (Stage-2): {metrics['t_rerank_avg_ms']:.2f}ms/query  (DINOv2 numpy)")
    print(f"  Total latency:    {metrics['latency_avg_ms']:.2f}ms/query")
    print(f"  QPS:              {metrics['qps']:.1f}")

    return BenchmarkResult(
        qps=             float(metrics["qps"]),
        map_score=       float(metrics["mAP"]),
        latency_ms=      float(metrics["latency_avg_ms"]),
        t_clip_avg_ms=   float(metrics["t_clip_avg_ms"]),
        t_rerank_avg_ms= float(metrics["t_rerank_avg_ms"]),
        index_build_s=   float(index_build_s),
        precision_at_10= float(metrics["precision_at_10"]),
        ndcg_at_10=      float(metrics["ndcg_at_10"]),
        recall_at_10=    float(metrics["recall_at_10"]),
        engine=          engine,
        params=          engine_params,
        k_neighbors=     k_neighbors,
        alpha=           alpha,
        timestamp=       time.time(),
        per_query_ap=    [float(a) for a in metrics.get("per_query_ap", [])],
    )


# ===========================================================================
# OPTIMIZATION LOOP
# ===========================================================================

def run_optimization(method: str, iterations: int, port: int, dataset_dir: Path,
                     seed: int = 42, api_key: Optional[str] = None,
                     model: str = "minimax/minimax-m2.1",
                     vdms_restart_cmd: Optional[str] = None,
                     patience: int = 3) -> List[IterationResult]:

    print(f"Starting {method} optimization. Iterations: {iterations}, Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    config_provider = ConfigProvider(seed=seed)
    llm_provider    = LLMProvider(api_key, model=model, seed=seed) if api_key else None

    if method == "grid":
        configs = config_provider.grid_search_configs()
        if len(configs) < iterations:
            configs += [config_provider.get_random_config()
                        for _ in range(iterations - len(configs))]
        configs = configs[:iterations]
    elif method == "random":
        configs = [config_provider.get_random_config() for _ in range(iterations)]
    else:
        configs = None

    optuna_study = None
    optuna_trial = None
    if method == "optuna":
        optuna_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
    elif method == "gp_bo":
        optuna_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GPSampler(seed=seed),
        )

    vdtuner_opt = None
    if method == "vdtuner":
        _alpha_vals = ConfigProvider.ALPHA_VALUES
        _encoder = KnobEncoder([
            {"name": "M",              "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]},
            {"name": "efConstruction", "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"]},
            {"name": "k_neighbors",    "type": "enum",
             "values": ConfigProvider.K_NEIGHBORS_VALUES},
            {"name": "alpha",          "type": "enum", "values": _alpha_vals},
            {"name": "n_refs",         "type": "enum",
             "values": ConfigProvider.N_REFS_VALUES},
            {"name": "ref_strategy",   "type": "enum",
             "values": ConfigProvider.REF_STRATEGY_VALUES},
            {"name": "n_aqe",          "type": "enum",
             "values": ConfigProvider.N_AQE_VALUES},
            {"name": "aqe_weight",     "type": "enum",
             "values": ConfigProvider.AQE_WEIGHT_VALUES},
        ])
        vdtuner_opt = VDTunerOptimizer(_encoder, seed=seed)

    def _norm_refs(cfg):
        """Normalize n_refs=1 for centroid (centroid ignores n_refs)."""
        rs = cfg.get("ref_strategy", "first")
        nr = 1 if rs == "centroid" else cfg.get("n_refs", 1)
        return nr, rs

    results: List[IterationResult] = []
    best_score_ever = 0.0
    iters_since_improvement = 0

    for i in range(iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{iterations} ({method.upper()})")
        print(f"{'='*80}")

        # Restart VDMS before each iteration (except first — already started by shell script)
        if i > 0 and vdms_restart_cmd:
            print(f"[VDMS] Restarting for clean DB (iteration {i+1})...")
            subprocess.run(vdms_restart_cmd, shell=True, check=True)
            time.sleep(3)
            print(f"[VDMS] Restart complete.")

        if method == "hyperparameter_only":

            tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 r.config.get("k_neighbors", 200),
                 f"{r.config.get('alpha', 0.0):.2f}",
                 *_norm_refs(r.config),
                 r.config.get("n_aqe", 1),
                 f"{r.config.get('aqe_weight', 0.0):.1f}")
                for r in results
            }

            consecutive_alpha_only = 0
            if len(results) >= 2:
                ref_cfg = results[-1].config
                ref_struct = (ref_cfg.get("engine"),
                              str(sorted(ref_cfg.get("params", {}).items())),
                              ref_cfg.get("k_neighbors", 200),
                              ref_cfg.get("n_refs", 1),
                              ref_cfg.get("ref_strategy", "first"))
                for h in reversed(results):
                    h_struct = (h.config.get("engine"),
                                str(sorted(h.config.get("params", {}).items())),
                                h.config.get("k_neighbors", 200),
                                h.config.get("n_refs", 1),
                                h.config.get("ref_strategy", "first"))
                    if h_struct == ref_struct:
                        consecutive_alpha_only += 1
                    else:
                        break

            dedup_hint = ""
            for _attempt in range(4):
                config, reasoning = asyncio.run(
                    llm_provider.query_llm(i + 1, results, iterations,
                                           consecutive_alpha_only=consecutive_alpha_only,
                                           iters_since_improvement=iters_since_improvement,
                                           dedup_hint=dedup_hint))
                config["_iter"] = i + 1
                key = (config["engine"],
                       str(sorted(config.get("params", {}).items())),
                       config.get("k_neighbors", 200),
                       f"{config.get('alpha', 0.0):.2f}",
                       *_norm_refs(config),
                       config.get("n_aqe", 1),
                       f"{config.get('aqe_weight', 0.0):.1f}")
                if key not in tried_keys:
                    break
                # Build an informative hint explaining exactly why this config was rejected
                nr_raw = config.get("n_refs", 1)
                rs = config.get("ref_strategy", "first")
                nr_norm, rs_norm = _norm_refs(config)
                if rs == "centroid" and nr_raw != nr_norm:
                    dedup_hint = (
                        f"You suggested n_refs={nr_raw}, ref_strategy=centroid, "
                        f"k={config.get('k_neighbors')}, alpha={config.get('alpha'):.2f}, "
                        f"params={config.get('params')}.\n"
                        f"  → centroid ALWAYS normalizes to n_refs=1 (centroid uses ALL k candidates, "
                        f"n_refs is ignored). The normalized config "
                        f"(n_refs=1, centroid, k={config.get('k_neighbors')}) was already tried.\n"
                        f"  → Do NOT suggest centroid with n_refs>1. To vary centroid, change k instead.\n"
                        f"  → If you want to test PRF with n_refs>1, use ref_strategy=first or diverse."
                    )
                else:
                    # Check if stuck on the same k value repeatedly
                    recent_k = [r.config.get("k_neighbors") for r in results[-6:]]
                    suggested_k = config.get("k_neighbors")
                    k_fixation = recent_k.count(suggested_k) >= 3 if recent_k else False
                    if k_fixation:
                        tried_k_values = sorted(set(r.config.get("k_neighbors") for r in results))
                        untried_k = [k for k in [50, 100, 150] if k not in tried_k_values]
                        dedup_hint = (
                            f"You suggested k={suggested_k} again — you have used k={suggested_k} "
                            f"in {recent_k.count(suggested_k)} of the last {len(recent_k)} iterations. "
                            f"You are STUCK on k={suggested_k}.\n"
                            f"  → k values already tried: {tried_k_values}\n"
                            f"  → k values NOT yet tried from [50, 100]: {untried_k if untried_k else 'both tried — vary alpha or efC instead'}\n"
                            f"  → RULE: k=50 is the highest-priority unexplored config (QPS≈272 vs 115 at k=100). "
                            f"Try k=50 with your current best (M, efC, alpha, ref_strategy).\n"
                            f"  → Do NOT suggest k={suggested_k} again until you have tried k=50."
                        )
                    else:
                        dedup_hint = (
                            f"You suggested engine={config.get('engine')}, "
                            f"params={config.get('params')}, k={config.get('k_neighbors')}, "
                            f"alpha={config.get('alpha'):.2f}, n_refs={nr_norm}, "
                            f"ref_strategy={rs_norm} — this exact combination was already evaluated. "
                            f"Check the 'ALL CONFIGS TRIED' list above and suggest something new."
                        )
                print(f"[Dedup] LLM repeated a tried config (attempt {_attempt+1}), retrying...")
            else:
                config = ConfigProvider(seed=seed + i).get_random_config()
                config["_iter"] = i + 1
                reasoning = "Fallback random (LLM repeated configs 4x)"
                print(f"[Dedup] Falling back to random config after 4 duplicate attempts.")

        elif method == "grid":
            config    = configs[i]
            config["_iter"] = i + 1
            reasoning = "Baseline (Grid)"

        elif method in ("optuna", "gp_bo"):
            opt_tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 r.config.get("k_neighbors", 200),
                 f"{r.config.get('alpha', 0.0):.2f}",
                 *_norm_refs(r.config),
                 r.config.get("n_aqe", 1),
                 f"{r.config.get('aqe_weight', 0.0):.1f}")
                for r in results
            }
            optuna_trial = None
            for _dedup_attempt in range(4):
                _trial = optuna_study.ask()
                engine = _trial.suggest_categorical(
                    "engine", ConfigProvider.SEARCH_ENGINES)
                params = {}
                if engine == "FaissHNSWFlat":
                    params["M"]              = _trial.suggest_categorical(
                        "M", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"])
                    params["efConstruction"] = _trial.suggest_categorical(
                        "efConstruction", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"])
                    params["efSearch"]       = 500  # fixed: negligible QPS effect at 762K scale
                k_neighbors  = _trial.suggest_categorical(
                    "k_neighbors", ConfigProvider.K_NEIGHBORS_VALUES)
                alpha        = _trial.suggest_categorical(
                    "alpha", ConfigProvider.ALPHA_VALUES)
                ref_strategy = _trial.suggest_categorical(
                    "ref_strategy", ConfigProvider.REF_STRATEGY_VALUES)
                # centroid ignores n_refs — fix to 1 so TPE doesn't waste budget on redundant configs
                if ref_strategy == "centroid":
                    n_refs = 1
                else:
                    n_refs = _trial.suggest_categorical(
                        "n_refs", ConfigProvider.N_REFS_VALUES)
                n_aqe = _trial.suggest_categorical(
                    "n_aqe", ConfigProvider.N_AQE_VALUES)
                if n_aqe > 1:
                    aqe_weight = _trial.suggest_categorical(
                        "aqe_weight", ConfigProvider.AQE_WEIGHT_VALUES)
                else:
                    aqe_weight = 0.0
                _nr_norm = 1 if ref_strategy == "centroid" else int(n_refs)
                opt_key = (engine, str(sorted(params.items())),
                           int(k_neighbors), f"{float(alpha):.2f}",
                           _nr_norm, str(ref_strategy),
                           int(n_aqe), f"{float(aqe_weight):.1f}")
                if opt_key not in opt_tried_keys:
                    optuna_trial = _trial
                    break
                dup_score = next(
                    (r.benchmark.score() for r in results
                     if (r.config["engine"],
                         str(sorted(r.config.get("params", {}).items())),
                         r.config.get("k_neighbors", 200),
                         f"{r.config.get('alpha', 0.0):.2f}",
                         *_norm_refs(r.config),
                         r.config.get("n_aqe", 1),
                         f"{r.config.get('aqe_weight', 0.0):.1f}") == opt_key),
                    0.0
                )
                optuna_study.tell(_trial, dup_score)
                print(f"[Optuna Dedup] Attempt {_dedup_attempt+1}: duplicate, told score={dup_score:.2f}, retrying...")
            else:
                config = ConfigProvider(seed=seed + i).get_random_config()
                config["_iter"] = i + 1
                reasoning = "Fallback random (Optuna repeated configs 4x)"
                optuna_trial = None
                print("[Optuna Dedup] Falling back to random config after 4 duplicate attempts.")

            if optuna_trial is not None:
                config = {
                    "engine":       engine,
                    "params":       params,
                    "k_neighbors":  int(k_neighbors),
                    "alpha":        float(alpha),
                    "n_refs":       int(n_refs),
                    "ref_strategy": str(ref_strategy),
                    "n_aqe":        int(n_aqe),
                    "aqe_weight":   float(aqe_weight),
                    "_iter":        i + 1,
                }
                reasoning = "Baseline (Optuna TPE)" if method == "optuna" else "Baseline (GP-BO)"

        elif method == "vdtuner":
            vt_raw = vdtuner_opt.ask()
            # Apply conditional constraints: centroid ignores n_refs; n_aqe=1 disables aqe_weight
            n_refs = 1 if vt_raw["ref_strategy"] == "centroid" else vt_raw["n_refs"]
            aqe_weight = 0.0 if vt_raw["n_aqe"] == 1 else vt_raw["aqe_weight"]
            config = {
                "engine": "FaissHNSWFlat",
                "params": {
                    "M":              vt_raw["M"],
                    "efConstruction": vt_raw["efConstruction"],
                    "efSearch":       500,
                },
                "k_neighbors":  vt_raw["k_neighbors"],
                "alpha":        vt_raw["alpha"],
                "n_refs":       n_refs,
                "ref_strategy": vt_raw["ref_strategy"],
                "n_aqe":        vt_raw["n_aqe"],
                "aqe_weight":   aqe_weight,
                "_iter":        i + 1,
            }
            reasoning = f"VDTuner EHVI iter {i+1}"

        else:
            config    = configs[i]
            config["_iter"] = i + 1
            reasoning = "Baseline (Random)"

        print(f"  engine:       {config['engine']}")
        print(f"  params:       {config.get('params', {})}")
        print(f"  k_neighbors:  {config.get('k_neighbors', 200)}")
        print(f"  alpha:        {config.get('alpha', 0.0):.2f}")
        print(f"  n_refs:       {config.get('n_refs', 1)}/{config.get('ref_strategy', 'first')}")

        try:
            benchmark = run_benchmark(port, str(dataset_dir), config, iteration=i + 1)

            print(f"\n[Results iter {i+1}]")
            score_label = f"Score (QPS|mAP≥{_MAP_THRESHOLD:.2f})"
            print(f"  {score_label}: {benchmark.score():.2f}")
            print(f"  mAP:               {benchmark.map_score:.4f}")
            print(f"  QPS:               {benchmark.qps:.2f}")
            print(f"  P@10:              {benchmark.precision_at_10:.4f}")
            print(f"  nDCG@10:           {benchmark.ndcg_at_10:.4f}")

            results.append(IterationResult(i + 1, config, benchmark, reasoning, method))

            if method in ("optuna", "gp_bo") and optuna_trial is not None:
                optuna_study.tell(optuna_trial, benchmark.score())
                optuna_trial = None
            if method == "vdtuner":
                vdtuner_opt.tell(config, benchmark.map_score, benchmark.qps)

            if benchmark.score() > best_score_ever:
                best_score_ever = benchmark.score()
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            if patience > 0 and best_score_ever > 0.0 and iters_since_improvement >= patience:
                print(f"\n[Early Stop] No improvement for {patience} consecutive iterations. "
                      f"Best score: {best_score_ever:.4f}. Stopping at iteration {i+1}/{iterations}.")
                break

        except Exception as e:
            print(f"\n[ERROR] Benchmark failed at iteration {i+1}: {e}")
            import traceback
            traceback.print_exc()
            if method in ("optuna", "gp_bo") and optuna_trial is not None:
                optuna_study.tell(optuna_trial, 0.0)
                optuna_trial = None
            if method == "vdtuner" and vdtuner_opt._last_vec is not None:
                vdtuner_opt.tell(config, 0.0, 0.0)
            sys.exit(1)

    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if results:
        best = max(results, key=lambda r: r.benchmark.score())
        print(f"Best Score:  {best.benchmark.score():.4f}")
        print(f"Best mAP:    {best.benchmark.map_score:.4f}")
        print(f"Best QPS:    {best.benchmark.qps:.2f}")
        print(f"Best Config: {best.config}")
    if method == "vdtuner" and vdtuner_opt is not None:
        vt_cfg, vt_q, vt_qps = vdtuner_opt.select(_MAP_THRESHOLD)
        print(f"\n[VDTuner] ε-constraint selection (mAP≥{_MAP_THRESHOLD}):")
        print(f"  Config:  {vt_cfg}")
        print(f"  mAP:     {vt_q:.4f}  QPS: {vt_qps:.2f}")

    return results


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided VDMS Optimizer for GLDv2 Landmark Retrieval")
    parser.add_argument("--port", type=int, default=55596)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Root dataset dir (e.g. /work/hdd/bdjd/vdms/datasets)")
    parser.add_argument("--method",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"],
                        required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--model", type=str, default="minimax/minimax-m2.1",
                        help="OpenRouter model ID")
    parser.add_argument("--vdms-restart-cmd", type=str, default=None,
                        help="Shell command to restart VDMS between iterations for clean DB")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop if no improvement for this many consecutive iterations (0=disabled)")
    parser.add_argument("--map-threshold", type=float, default=0.15,
                        help="Constrained optimization threshold τ: Score=QPS if mAP≥τ else 0. "
                             "Default 0.15. GLDv2 baseline mAP≈0.179.")
    args = parser.parse_args()

    global _MAP_THRESHOLD
    _MAP_THRESHOLD = args.map_threshold
    print(f"[Score] Constrained objective: Score = QPS  if mAP ≥ {_MAP_THRESHOLD:.3f} else 0")

    api_key = None
    if args.method == "hyperparameter_only":
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)

    results = run_optimization(
        method=           args.method,
        iterations=       args.iterations,
        port=             args.port,
        dataset_dir=      args.dataset_dir,
        seed=             args.seed,
        api_key=          api_key,
        model=            args.model,
        vdms_restart_cmd= args.vdms_restart_cmd,
        patience=         args.patience,
    )

    best_result = max(results, key=lambda r: r.benchmark.score()) if results else None

    output_data = {
        "summary": {
            "best_score":      best_result.benchmark.score() if best_result else 0.0,
            "best_map":        best_result.benchmark.map_score if best_result else 0.0,
            "best_qps":        best_result.benchmark.qps if best_result else 0.0,
            "best_config":     best_result.config if best_result else {},
            "iterations_used": len(results),
        },
        "results": [
            {
                "iteration": r.iteration,
                "config":    r.config,
                "benchmark": asdict(r.benchmark),
                "score":     r.benchmark.score(),
                "reasoning": r.llm_reasoning,
            } for r in results
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Results saved to {args.output}")


if __name__ == "__main__":
    main()
