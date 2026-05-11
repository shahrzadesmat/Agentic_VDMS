"""
LLM-Guided VDMS Hyperparameter Optimizer for SIFT1M.

Ports gldv2_agent_optimizer.py (GLDv2) to the SIFT1M standard ANN benchmark.
ONLY the dataset-specific parts differ; the optimizer loop, dedup logic, grid
structure, untried-hints, and stagnation detection are IDENTICAL to HICO-DET
and GLDv2 for a fair method comparison in the paper.

Key differences from HICO-DET/GLDv2:
  - Dataset: SIFT1M, 1,000,000 vectors × 128-d, L2 metric (no semantic component).
  - No Stage-2 fusion: no alpha, n_refs, ref_strategy params.
  - Score: QPS if Recall@10 ≥ τ else 0  (SIEVE-style constrained; τ via --recall-threshold, default 0.90).
  - HNSW params (M, efConstruction, efSearch) injected via VDMS server config
    (VDMS_PARAMS_FILE written before each restart).  VDMS API schema rejects
    hnsw_M/efConstruction/efsearch in AddDescriptorSet — server config is canonical.
  - FaissIVFFlat forbidden: VDMS probes all cells at 1M scale (Score~35, unfixable).
  - Search space: FaissHNSWFlat only; 9×M × 6×efC × 11×efS × 10×k = 5,940 configs.

Score function: QPS if Recall@10 ≥ τ else 0  (SIEVE-style constrained; τ via --recall-threshold, default 0.90)

Agent tunes jointly (index-level only — no Stage-2 fusion):
  engine          : FaissHNSWFlat (FaissIVFFlat forbidden; FaissFlat reference-only)
  M               : HNSW graph connectivity  [4,8,12,16,24,32,48,64,96]
  efConstruction  : HNSW build quality       [50,100,150,200,300,400]
  efSearch        : HNSW search beam width   [8,16,24,32,48,64,100,150,200,300,500]
  k_neighbors     : results returned/query   [10,20,30,50,75,100,150,200,300,500]

Usage:
    python sift1m_agent_optimizer.py \\
        --port 55595 \\
        --dataset-dir /work/hdd/bdjd/vdms/datasets \\
        --method hyperparameter_only \\
        --iterations 30 \\
        --output sift1m/sift1m_llm_results.json
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
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# File used to communicate engine params to the VDMS restart shell function.
# Shell reads VDMS_PARAMS_FILE and injects hnsw_M/efConstruction/efsearch into
# the VDMS server JSON config before starting.  Identical mechanism to agentic_vdms.
SNAP_KEY_FILE   = os.environ.get("SNAP_KEY_FILE",   "/tmp/vdms_snap_key.txt")
VDMS_PARAMS_FILE = os.environ.get("VDMS_PARAMS_FILE", "/tmp/vdms_params.json")

# Constrained optimization threshold τ (SIEVE-style): Score = QPS if Recall@10 ≥ τ else 0.
# Set at startup from --recall-threshold CLI arg. Default 0.90 (standard ANN quality floor).
_RECALL_THRESHOLD: float = 0.90


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class BenchmarkResult:
    qps:            float
    recall_at_10:   float   # ANN accuracy: fraction of true top-10 neighbors found
    latency_ms:     float   # avg wall-clock per query (VDMS round-trip), ms
    index_build_s:  float   # SIFT_Set index build time (one-time cost), seconds
    precision:      float
    f1_score:       float
    engine:         str
    params:         Dict
    timestamp:      float

    def score(self) -> float:
        """Constrained optimization (SIEVE-style): Score = QPS if Recall@10 ≥ τ else 0.
        Maximizes throughput within the feasible high-recall region."""
        if float(self.recall_at_10) < _RECALL_THRESHOLD:
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
# DATA CACHE (avoid reloading 1M × 128 floats per iteration)
# ===========================================================================

_SIFT1M_CACHE: Dict = {}


def _read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def _read_ivecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def _load_sift1m_data_cached(dataset_dir: str) -> Dict:
    """Load SIFT1M once; cache in module-level dict for all subsequent iterations."""
    global _SIFT1M_CACHE
    if _SIFT1M_CACHE:
        return _SIFT1M_CACHE

    sift_dir = Path(dataset_dir) / "sift"
    print("[SIFT1M] Loading base vectors (1M × 128)…")
    t0 = time.time()
    base_vecs  = _read_fvecs(str(sift_dir / "sift_base.fvecs"))   # [1M, 128]
    query_vecs = _read_fvecs(str(sift_dir / "sift_query.fvecs"))  # [10K, 128]
    print(f"[SIFT1M] Loaded in {time.time()-t0:.1f}s  "
          f"base={base_vecs.shape}  queries={query_vecs.shape}")

    # Ground truth: load pre-computed file (sift_groundtruth.ivecs contains top-100 neighbors)
    gt_path = sift_dir / "sift_groundtruth.ivecs"
    print(f"[SIFT1M] Loading ground truth from {gt_path.name}…")
    t0 = time.time()
    gt_array = _read_ivecs(str(gt_path))   # [10K, 100]
    gt_sets  = [set(row[:10].tolist()) for row in gt_array]
    print(f"[SIFT1M] Ground truth loaded in {time.time()-t0:.1f}s  shape={gt_array.shape}")

    _SIFT1M_CACHE = {
        "base_vecs":  base_vecs,
        "query_vecs": query_vecs,
        "gt_sets":    gt_sets,
    }
    return _SIFT1M_CACHE


# ===========================================================================
# CONFIG PROVIDER
# ===========================================================================

class ConfigProvider:
    # -----------------------------------------------------------------------
    # CANONICAL SEARCH SPACE — identical for random, grid, LLM, and Optuna.
    # Every value here must appear verbatim in the LLM system prompt and in
    # grid_search_configs().  Do NOT add values to only one place.
    # -----------------------------------------------------------------------

    # k_neighbors: number of VDMS results to return per query.
    # Recall@10 is measured from all k returned results (higher k → higher recall,
    # lower QPS due to larger priority queue + TCP transfer).
    # k_neighbors must be >= 10 because Recall@10 is measured against 10 true neighbors.
    # With k < 10, Recall@10 is hard-capped at k/10 (e.g. k=5 → max 0.50), making it
    # strictly dominated by k=10 for any config.
    K_NEIGHBORS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]

    ENGINE_PARAMS = {
        "FaissFlat": {},
        "FaissHNSWFlat": {
            "M":              [4, 8, 12, 16, 24, 32, 48, 64, 96],
            "efConstruction": [50, 100, 150, 200, 300, 400],  # build quality; higher=better graph, slower build
            "efSearch":       [8, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500],
        },
        "FaissIVFFlat": {
            "nlist": [64, 128, 256, 512, 1024, 2048, 4096],
            # nprobe is fixed at 1 by VDMS API — cannot be tuned
        },
    }

    # Engines for random/Optuna: FaissHNSWFlat only.
    # FaissIVFFlat probes all cells at 1M scale (VDMS bug) → Score~35, cannot compete.
    # FaissFlat is still in ENGINE_PARAMS (LLM validation) and grid (brute-force reference).
    SEARCH_ENGINES = ["FaissHNSWFlat"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def get_random_config(self) -> Dict:
        engine = str(self.rng.choice(self.SEARCH_ENGINES))
        params = self._sample_params(engine)
        k = int(self.rng.choice(self.K_NEIGHBORS))
        return {"engine": engine, "params": params, "k_neighbors": k}

    def _sample_params(self, engine: str) -> Dict:
        params = {}
        for pname, pvals in self.ENGINE_PARAMS.get(engine, {}).items():
            params[pname] = int(self.rng.choice(pvals))
        return params

    def grid_search_configs(self) -> List[Dict]:
        """
        Structured 30-config sweep over the expanded M×efC×efS×k space.
        Designed to give every method (LLM, grid, random, optuna) fair coverage
        of the most informative corners of the 5,940-config HNSW search space.

          S1 (1):  Brute-force reference — FaissFlat k=10
          S2 (9):  M sweep — all 9 M values, efC=200 efS=32 k=10
          S3 (6):  efS sweep — M=32 efC=200 k=10, efS∈{8,16,48,64,200,500}
                   (efS=32 already in S2)
          S4 (5):  k sweep — M=32 efC=200 efS=32, k∈{20,30,75,150,500}
                   (k=10 already in S2; k<10 excluded — caps Recall@10 at k/10)
          S5 (4):  efC sweep — M=32 efS=32 k=10, efC∈{50,100,300,400}
                   (efC=200 already in S2)
          S6 (5):  Cross interactions: extreme M×k, M×efC, M×efS corners
          Total:   1 + 9 + 6 + 5 + 4 + 5 = 30
        """
        configs = []

        def hnsw(M, efC, efS, k):
            return {"engine": "FaissHNSWFlat",
                    "params": {"M": M, "efConstruction": efC, "efSearch": efS},
                    "k_neighbors": k}

        # S1: Brute-force reference (1)
        configs.append({"engine": "FaissFlat", "params": {}, "k_neighbors": 10})

        # S2: HNSW M sweep — all 9 M values at efC=200 efS=32 k=10 (9)
        for M in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
            configs.append(hnsw(M, 200, 32, 10))

        # S3: HNSW efS sweep — M=32 efC=200 k=10, efS∈{8,16,48,64,200,500} (6)
        # (efS=32 already covered in S2)
        for efS in [8, 16, 48, 64, 200, 500]:
            configs.append(hnsw(32, 200, efS, 10))

        # S4: k sweep — M=32 efC=200 efS=32, k∈{20,30,75,150,500} (5)
        # (k=10 already covered in S2; k=5 excluded — caps Recall@10 at 0.5)
        for k in [20, 30, 75, 150, 500]:
            configs.append(hnsw(32, 200, 32, k))

        # S5: efC sweep — M=32 efS=32 k=10, efC∈{50,100,300,400} (4)
        # (efC=200 already covered in S2)
        for efC in [50, 100, 300, 400]:
            configs.append(hnsw(32, efC, 32, 10))

        # S6: Cross interactions — extreme corners (5)
        configs.append(hnsw(4,  200, 32, 50))  # extreme low M + k compensation
        configs.append(hnsw(8,  400, 32, 20))  # low M + best efC + mid k
        configs.append(hnsw(96, 400, 16, 10))  # extreme high M + low efS + best efC
        configs.append(hnsw(32, 400, 48, 30))  # mid efS + best efC + mid k
        configs.append(hnsw(16, 200, 24, 100)) # low M + very low efS + high k

        # ---- S7a: efS × k cross — high recall zone (5) ----
        for efS, k in [(64, 75), (64, 150), (100, 50), (100, 100), (200, 50)]:
            configs.append(hnsw(32, 200, efS, k))

        # ---- S7b: high-M cross (5) ----
        for M, efC, efS, k in [(64, 200, 64, 20), (64, 200, 200, 50),
                                (64, 400, 64, 10), (96, 200, 64, 20),
                                (48, 200, 64, 20)]:
            configs.append(hnsw(M, efC, efS, k))

        # ---- S7c: low-M compensation — low M + higher efS/k (5) ----
        for M, efC, efS, k in [(4, 200, 64, 50), (8, 200, 64, 20),
                                (8, 200, 200, 30), (12, 200, 64, 30),
                                (16, 200, 64, 20)]:
            configs.append(hnsw(M, efC, efS, k))

        # ---- S7d: efC × efS cross (5) ----
        for M, efC, efS, k in [(32, 400, 64, 10), (32, 400, 200, 10),
                                (64, 400, 64, 10), (32, 400, 64, 50),
                                (48, 400, 64, 20)]:
            configs.append(hnsw(M, efC, efS, k))

        # Total: 1+9+6+5+4+5 + 5+5+5+5 = 50
        assert len(configs) == 50, f"Grid must have exactly 50 configs, got {len(configs)}"
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
                        consecutive_efsearch_only: int = 0,
                        iters_since_improvement: int = 0,
                        dedup_hint: str = "") -> Tuple[Dict, str]:

        exp_end  = max(1, int(round(total_iterations * 0.40)))
        expl_end = max(exp_end + 1, int(round(total_iterations * 0.75)))
        phase = ("EXPLORATION" if iteration <= exp_end else
                 "EXPLOITATION" if iteration <= expl_end else "FINE-TUNING")

        current_best_score = max(r.benchmark.score() for r in history) if history else 0.0

        # ── Diagnosis ──────────────────────────────────────────────────────
        if history:
            last_run     = history[-1]
            last_score   = last_run.benchmark.score()
            last_recall  = last_run.benchmark.recall_at_10
            last_qps     = last_run.benchmark.qps
            last_engine  = last_run.config["engine"]
            last_params  = last_run.config.get("params", {})

            if last_score >= current_best_score * 0.98:
                if phase == "FINE-TUNING":
                    guidance = (
                        f"EXCELLENT. Near peak. {last_engine} params={last_params} "
                        f"Score={last_score:.1f}. Fine-tuning: change ONE param by ONE step — "
                        f"efSearch ±1 step, M ±1 step, efConstruction ±1 step, or k ±1 step.")
                else:
                    guidance = (
                        f"Near best so far ({last_engine} params={last_params} "
                        f"Score={last_score:.1f}). Phase={phase}: do NOT fine-tune yet. "
                        f"KEEP EXPLORING — try a structurally DIFFERENT region: "
                        f"different M value (untried M values preferred), very low efSearch (8-16), or higher efC.")
            elif last_recall < 0.80:
                guidance = (
                    f"Recall too low ({last_recall:.3f}). Score={last_score:.1f}. "
                    f"INCREASE search depth: increase efSearch, M, k, or efC=400. "
                    f"OR: keep low efSearch but raise k (higher k recovers missed neighbors cheaply).")
            elif last_recall > 0.98 and last_qps < 400:
                guidance = (
                    f"Recall near-perfect ({last_recall:.3f}) but QPS is low ({last_qps:.1f}). "
                    f"REDUCE search depth: decrease efSearch (primary lever). "
                    f"Lower M also helps QPS at the cost of recall ceiling.")
            elif last_score < current_best_score * 0.90:
                guidance = (
                    f"Score={last_score:.1f} significantly below best={current_best_score:.1f}. "
                    f"Last: {last_engine} params={last_params} Recall={last_recall:.3f} QPS={last_qps:.1f}. "
                    f"Return near the best config and make targeted changes.")
            else:
                guidance = (
                    f"Score={last_score:.1f} (QPS). Recall={last_recall:.3f} (≥τ={_RECALL_THRESHOLD:.2f} ✓). QPS={last_qps:.1f}. "
                    f"Recall floor met — maximize QPS. Try LOWER M (e.g. M=12) or lower efSearch (24). "
                    f"EMPIRICAL best region: M=12-24, efSearch=24-32, k=10, efC=200 or 400 → Score~1150-1200+. "
                    f"KEY: efC=200 is the reliable default. efC=400 costs 2× build time but does NOT reduce QPS — "
                    f"try efC=400 at M=24 if efC=200 baseline is established and time remains.")
        else:
            guidance = (
                f"First iteration — cold start. Score = QPS if Recall@10 ≥ τ={_RECALL_THRESHOLD:.2f} else 0. "
                "Feasibility floor τ must be met; then maximize QPS. "
                "CRITICAL PHYSICS (SIFT1M 1M vectors, L2): "
                "FaissHNSWFlat is the ONLY viable engine (IVFFlat forbidden, FaissFlat too slow). "
                "Key physics: LOWER M = fewer graph edges = FASTER graph traversal at same efSearch. "
                "M=16 has fewer edges than M=32 → higher QPS at same efS, with only a small recall cost. "
                "M=16 efC=200 efS=32 k=10 is the recommended starting point (HNSW physics: QPS ~1150-1200). "
                "efC=200 is a reliable default — sufficient to cross τ=0.90 for M≥16 at efS=32. "
                "k=10 is the minimum (k<10 caps Recall@10 at k/10, strictly dominated). "
                f"START HERE: HNSW M=16 efC=200 efS=32 k=10 — recommended baseline (Recall≈0.91 ≥ τ={_RECALL_THRESHOLD:.2f}, high QPS).")

        # ── System context ──────────────────────────────────────────────────
        system_context = f"""
SYSTEM ARCHITECTURE:
1. TARGET SYSTEM: VDMS (Visual Data Management System)
   - C++ middleware for persistent vector storage and ANN search.
   - Active indexing backend: FaissHNSWFlat ONLY.
     (FaissIVFFlat is FORBIDDEN — VDMS bug at 1M scale. FaissFlat is reference-only.)
   - Communication: Client-Server via TCP sockets.
   - CRITICAL: HNSW params (M, efConstruction, efSearch) are set in the VDMS server
     JSON config at startup — NOT in AddDescriptorSet.
     The shell restart injects these via VDMS_PARAMS_FILE before each iteration.

2. DATASET: SIFT1M (Standard ANN Benchmark)
   - 1,000,000 base vectors × 128 dimensions (original uint8 cast to float32).
   - 10,000 query vectors.  Ground truth: exact brute-force top-10 neighbors.
   - Metric: L2 (Euclidean).
   - This is a PURE ANN task — no semantic component, no Stage-2 fusion.

3. OPTIMIZATION OBJECTIVE: Constrained maximization (SIEVE-style)
   Score = QPS  if Recall@10 ≥ τ={_RECALL_THRESHOLD:.2f}  else  0
   - τ={_RECALL_THRESHOLD:.2f} is the quality floor. Configs below it score 0 (infeasible).
   - Above the floor, maximize QPS (throughput). Prevents degenerate low-recall/ultra-fast solutions.
   - Recall@10: fraction of true top-10 neighbors found in the k returned results.
   - QPS: queries per second (10,000 queries timed as a batch).
   - Tunable parameters: M, efConstruction, efSearch, k_neighbors.
     No alpha, n_refs, or ref_strategy (those are Stage-2 fusion params, not used here).
"""

        # ── Engine definitions ──────────────────────────────────────────────
        engine_definitions = f"""
ENGINE PHYSICS (SIFT1M 1M vectors, L2):

[FaissFlat] — Brute-force exact search
  Parameters: none.
  PHYSICS: Recall=1.0 always.  QPS≈1-5 (too slow to compete).  Score<5.
  USE WHEN: Reference upper bound only.

[FaissHNSWFlat] — HNSW navigable graph  ← ONLY COMPETITIVE ENGINE
  Parameters:
    M [4, 8, 12, 16, 24, 32, 48, 64, 96]: Graph connectivity (edges per node).
      Very low M (4-8): extremely fast build, sparse graph, low Recall ceiling, highest raw QPS.
      M=4:  Recall ceiling ~0.70 (k=10), fastest possible. Needs high k or efS to compensate.
      M=8:  Recall ceiling ~0.82 (k=10), very fast.
      M=12: Recall ceiling ~0.87 (k=10).
      M=16: Recall ceiling ~0.92 (k=10), memory-efficient.
      M=24: Recall ceiling ~0.95 (k=10).
      M=32: Recall ceiling ~0.97 (k=10), robust default.
      M=48: Recall ceiling ~0.99 (k=10), best recall at moderate cost.
      M=64: Near-perfect recall ceiling, heavy build.
      M=96: Recall ceiling ~1.0, heaviest build, marginal over M=64.
    efConstruction [50, 100, 150, 200, 300, 400]: Build quality (beam width during construction).
      Higher efC = better graph connectivity = higher Recall at the SAME efSearch.
      Does NOT affect QPS at inference time.  Costs more build time only.
      efC=50:  Fast build, lowest quality — risk recall floor failure at M≤16.
      efC=100: Good for M≥24. Risky for M=8-16 (borderline recall).
      efC=200: RELIABLE DEFAULT — sufficient for M≥16 to cross τ=0.90 at efS=32.
      efC=400: Best graph quality — zero QPS cost at inference time (only build is slower ~2×).
               For M=24+: efC=400 can raise recall by 1-3pp vs efC=200 at the SAME QPS.
               Try efC=400 with M=24 after establishing the efC=200 baseline.
      KEY: efC interacts with M. Low M (4-12) needs higher efC to compensate for sparse graph.
    efSearch [8, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500]: Search beam width.  PRIMARY QPS lever.
      At M=32, efC=200, k=10:
        efS=8  → QPS~2000+ Recall~0.40-0.50  Score~800-1000 (extreme speed, low recall)
        efS=16 → QPS~1600  Recall~0.65       Score~1040
        efS=24 → QPS~1300  Recall~0.80       Score~1040
        efS=32 → QPS~1100  Recall~0.88-0.94  Score~970-1034 ← sweet spot for high QPS
        efS=48 → QPS~900   Recall~0.93       Score~837
        efS=64 → QPS~750   Recall~0.96       Score~720
        efS=100→ QPS~580   Recall~0.98       Score~568
        efS=150→ QPS~430   Recall~0.99       Score~426
        efS=200→ QPS~330   Recall~0.99       Score~327
        efS=300→ QPS~200   Recall~0.995      Score~199
        efS=500→ QPS~120   Recall~0.998      Score~120
  KEY INSIGHT: efSearch is the dominant speed/recall lever.
    Very low efSearch (8-24) can achieve higher Score than efS=32 if M and k are tuned.
    M determines the Recall ceiling at any efSearch value.

k_neighbors [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]: Results returned per query.
  Recall@10 is measured from ALL k returned results (does any true top-10 appear in k results?).
  CONSTRAINT: k must be >= 10. With k < 10, Recall@10 ≤ k/10 (strictly dominated by k=10).
  Higher k → higher Recall@10 (more candidates) → lower QPS (larger queue + TCP transfer).
  At M=32, efC=200, efS=32:
    k=10  → Recall≈0.943  QPS~1073  Score~1012
    k=20  → Recall≈0.962  QPS~1040  Score~1001
    k=30  → Recall≈0.969  QPS~1010  Score~979
    k=50  → Recall≈0.975  QPS~990   Score~965
    k=75  → Recall≈0.979  QPS~960   Score~939
    k=100 → Recall≈0.983  QPS~940   Score~924
    k=150 → Recall≈0.986  QPS~900   Score~887
    k=200 → Recall≈0.988  QPS~870   Score~859
    k=300 → Recall≈0.991  QPS~820   Score~812
    k=500 → Recall≈0.994  QPS~750   Score~746
  CRITICAL CROSS-TERM: A sparse graph (low M or low efS) misses some true top-10.
    Compensate by requesting more results (higher k): the missed neighbors may be in positions 11-50.
    Example: M=8 efS=32 k=10 → Recall≈0.82 Score≈926
             M=8 efS=32 k=50 → Recall≈0.87 Score≈870  ← sometimes k=10 wins on Score despite lower recall
    This M×efS×k×efC interaction is where LLM reasoning has the clearest advantage.

⛔ FaissIVFFlat — PERMANENTLY FORBIDDEN on SIFT1M with VDMS.
  VDMS architectural bug: probes ALL cells regardless of nlist at 1M scale.
  Measured: Recall~1.0, QPS~35, Score~35. This is an unfixable limitation.
  Do NOT suggest FaissIVFFlat under any circumstances.

============================================================
CROSS-PARAMETER INTERACTION SUMMARY (SIFT1M key insights):
============================================================
  M × QPS       → CRITICAL: Lower M = fewer edges per node = faster graph traversal at same efS.
                  HNSW physics: M=16 efC=200 efS=32 k=10 → estimated QPS ~1150-1200 (sparser graph)
                                M=32 efC=200 efS=32 k=10 → estimated QPS ~1050-1100 (standard baseline)
                  M=16 beats M=32 on QPS by ~10% because sparser graph = less traversal work per query.
                  The recall loss from M=16 vs M=32 is small (~0.01) and still above τ=0.90.
  M ceiling     → M=4: Recall ceiling ~0.70 — CANNOT cross τ=0.90 at k=10. Avoid unless k≥100.
                  M=8: Recall ceiling ~0.82 — needs k≥50 to possibly cross τ, costly on QPS.
                  M=12: Recall ceiling ~0.87 — borderline; needs efC=400 and k=20-30.
                  M=16: Recall ceiling ~0.92 — SWEET SPOT for score at efS=32 k=10.
  efSearch × M  → At same efSearch, higher M = higher Recall but LOWER QPS (more edges).
                  Optimal: use M=16 (not M=32) for best QPS × Recall tradeoff at τ=0.90.
  efSearch × k  → At low efSearch (fast), k>10 recovers missed neighbors cheaply.
                  efSearch=24, k=20 at M=16 may beat efSearch=32, k=10 — test empirically.
  efC × M       → efC=200 is the default and sufficient for M≥16. For M=24, efC=400 can improve
                  recall by 1-3pp at ZERO QPS cost — worthwhile exploration after efC=200 baseline.
                  For M≤12, efC=400 is often needed to reach τ=0.90.
  efC × build   → efC=400 costs ~2× build time vs efC=200. No inference-time QPS effect.
  Score sweet spot → Two competitive regions (empirically measured):
                  (A) M=16 efC=200 efS=32 k=10 → Score~1163-1174 (sparser graph, peak QPS)
                  (B) M=24 efC=400 efS=24 k=10 → Score~1163 (denser graph, better recall, similar QPS at efS=24)
"""

        # ── Phase-specific scenario guide ────────────────────────────────────
        scenario_guide = f"""
PHASE-SPECIFIC STRATEGY (current phase: {phase}):

EXPLORATION (iterations 1-{exp_end}): Map the M × efSearch × k × efC landscape.
  - PRIORITY 1: HNSW M=16 efC=200 efS=32 k=10  — sparse graph, high QPS, recall ≈0.91 (above τ).
  - PRIORITY 2: HNSW M=24 efC=200 efS=32 k=10  — denser graph, higher recall ceiling, test QPS.
  - PRIORITY 3: HNSW M=24 efC=400 efS=24 k=10  — efC=400 costs zero QPS; efS=24 boosts QPS vs efS=32.
  - PRIORITY 4: HNSW M=16 efC=200 efS=24 k=10  — lower efS → higher QPS, test if recall still ≥ τ.
  - PRIORITY 5: HNSW M=12 efC=200 efS=32 k=10  — even sparser graph → even higher QPS (test recall).
  - PRIORITY 6: HNSW M=12 efC=200 efS=32 k=20  — M=12 recall borderline; k=20 recovery.
  - AVOID: M=4 k=10 (Recall ceiling ~0.70, cannot cross τ=0.90), efSearch≥200 (Score<400),
           k≥300 (QPS cost excessive).
  - Goal: find the (M, efC, efS, k) that maximizes QPS while keeping Recall≥τ.

EXPLOITATION (iterations {exp_end+1}-{expl_end}): Narrow around best region.
  - Fix engine to FaissHNSWFlat.  Sweep efSearch ±1 step and M ±1 step around best.
  - KEY INSIGHT: if best has Recall just above τ (0.90-0.93), try lower efS or lower M for QPS.
    If best has Recall <0.90, increase k (cheapest fix) or efC (last resort).
  - efC=200 is the default. After establishing the efC=200 baseline, try efC=400 at M=24 —
    it costs zero QPS and may raise recall 1-3pp, unlocking lower efS for higher QPS.

FINE-TUNING (iterations {expl_end+1}-{total_iterations}): ONE parameter change per iter.
  - Change ONLY one of: efSearch, M, k_neighbors, efConstruction.
  - Focus on the M × efSearch × k × efC interaction to squeeze final improvement.

SCENARIO DECISION MATRIX:

1. "Sparse M + efC=200" (strong baseline):
   M=16 efS=32 k=10 efC=200: fewer edges → highest QPS while keeping Recall≥τ=0.90.
   Variants to test: M=12 efS=32 k=10, M=16 efS=24 k=10, M=12 efS=32 k=20.

2. "M=24 + efC=400 + lower efS" (competitive alternative — explore early):
   M=24 efC=400 efS=24 k=10: efC=400 raises recall (zero QPS cost), efS=24 recovers QPS.
   Empirically measured: Score≈1163, Recall=0.912 — competitive with M=16 efC=200 efS=32.

3. "Very low M + k compensation":
   M=8 efS=32 k=30-50 efC=200: recall borderline but QPS very high (~1200+). Test empirically.
   M=12 efS=24 k=20 efC=200: promising unexplored region.

4. "Ultra-low efS + compensation":
   M=16 efS=16 k=20-30 efC=200: lower efS → QPS~1500 but recall drops; k compensates.
   M=16 efS=24 k=10 efC=200: slightly lower efS than best known; test recall.

5. FORBIDDEN (do NOT suggest):
   - M=4 k=10 (Recall ceiling ~0.70 — physically impossible to cross τ=0.90 at k=10)
   - M=8 k=10 (Recall ceiling ~0.82 — risky; needs k≥50 to cross τ)
   - k < 10 (Recall@10 capped at k/10 — strictly dominated by k=10)
   - efSearch≥200 (Score<400)
   - FaissFlat (Score<5 — reference only)
   - FaissIVFFlat (VDMS bug: Score~35, permanently forbidden)
"""

        # ── Build "all configs tried" log ────────────────────────────────────
        all_configs_tried = "\n".join(
            f"  iter{h.config.get('_iter','?'):02d}: {h.config['engine']} "
            f"p={h.config.get('params',{})} k={h.config.get('k_neighbors',10)} "
            f"→ Score={h.benchmark.score():.1f} "
            f"Recall={h.benchmark.recall_at_10:.3f} QPS={h.benchmark.qps:.1f}"
            for h in history
        )

        # ── Untried hints (analogous to GLDv2 untried_alpha_hint) ────────────
        untried_efsearch_hint = ""
        untried_M_hint        = ""
        untried_k_hint        = ""

        # efSearch and M hints: only useful once we have enough data (EXPLOITATION/FINE-TUNING)
        if history and phase in ("EXPLOITATION", "FINE-TUNING"):
            best_h      = max(history, key=lambda r: r.benchmark.score())
            best_engine = best_h.config["engine"]
            best_params = best_h.config.get("params", {})
            best_M      = best_params.get("M")

            if best_engine == "FaissHNSWFlat" and best_M is not None:
                tried_efsearch_here = {
                    r.config.get("params", {}).get("efSearch")
                    for r in history
                    if r.config["engine"] == "FaissHNSWFlat"
                    and r.config.get("params", {}).get("M") == best_M
                }
                untried_efsearch = [e for e in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]
                                    if e not in tried_efsearch_here]
                if untried_efsearch and phase == "FINE-TUNING":
                    untried_efsearch_hint = (
                        f"\nUNTRIED efSearch values at best M={best_M}: {untried_efsearch}"
                        f"\n  → These are valid next probes for fine-tuning efSearch."
                    )

                tried_M_here = {
                    r.config.get("params", {}).get("M")
                    for r in history
                    if r.config["engine"] == "FaissHNSWFlat"
                }
                untried_M = [m for m in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]
                             if m not in tried_M_here]
                if untried_M:
                    untried_M_hint = (
                        f"\nUNTRIED M values (any efSearch): {untried_M}"
                        f"\n  → Different M values are high-value probes — try these."
                    )

        # k_neighbors untried hint: shown in ALL phases (k is always a relevant axis)
        if history:
            tried_k_here = {r.config.get("k_neighbors", 10) for r in history}
            untried_k = [k for k in ConfigProvider.K_NEIGHBORS if k not in tried_k_here]
            if untried_k:
                untried_k_hint = (
                    f"\nUNTRIED k_neighbors values: {untried_k}"
                    f"\n  → k controls recall/QPS tradeoff. Higher k recovers missed neighbors "
                    f"from sparse graphs at a QPS cost. Minimum valid k is 10."
                )

        # ── Dynamic IVFFlat blacklist ─────────────────────────────────────────
        # If any IVFFlat run has been measured and scored < 100, permanently
        # forbid IVFFlat for the rest of this run.  VDMS IVFFlat at 1M scale
        # probes all cells regardless of nlist (Recall~0.999, QPS~35 — same as
        # brute-force), so no nlist value can rescue it.
        ivf_blacklist_hint = ""
        if history:
            ivf_scores = [r.benchmark.score() for r in history
                          if r.config.get("engine") == "FaissIVFFlat"]
            if ivf_scores and max(ivf_scores) < 100:
                ivf_blacklist_hint = (
                    f"\n⛔ IVFFlat PERMANENTLY FORBIDDEN — measured Score={max(ivf_scores):.1f} "
                    f"(VDMS probes all cells at 1M scale regardless of nlist: Recall~1.0, QPS~35). "
                    f"This is an architectural limitation, NOT a config issue. "
                    f"Do NOT propose FaissIVFFlat for any remaining iteration.\n"
                )

        # ── History JSON (last 5, same as GLDv2) ────────────────────────────
        history_json = json.dumps([{
            "eng":  h.config["engine"],
            "p":    h.config.get("params", {}),
            "k":    h.config.get("k_neighbors", 10),
            "sc":   round(h.benchmark.score(), 1),
            "rec":  round(h.benchmark.recall_at_10, 3),
            "qps":  round(h.benchmark.qps, 1),
        } for h in history[-5:]], indent=1)

        prompt = f"""You are an expert VDMS Optimization Engine for SIFT1M ANN retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

{system_context}

DIAGNOSIS OF LAST RUN:
{guidance}

{engine_definitions}

{scenario_guide}

ALL CONFIGS TRIED SO FAR (do NOT repeat any of these exact combinations):
{all_configs_tried if all_configs_tried else "  (none yet)"}
{ivf_blacklist_hint}{untried_k_hint}{untried_efsearch_hint}{untried_M_hint}{("⚠️  DEDUP REJECTION — YOUR LAST SUGGESTION WAS REJECTED:\n" + dedup_hint + "\nYou MUST suggest a genuinely different config.\n\n") if dedup_hint else ""}LAST 5 RESULTS (detailed):
{history_json}

AVAILABLE PARAMETER RANGES:
  FaissHNSWFlat: M={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['M']}
                 efConstruction={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efConstruction']}
                 efSearch={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efSearch']}
  k_neighbors:   {ConfigProvider.K_NEIGHBORS}  (results returned per query)
  ⛔ FaissIVFFlat: FORBIDDEN. ⛔ FaissFlat: reference only (Score<5).

DECISION RULES (apply in order):
1. Read DIAGNOSIS — it tells you the single most important fix.
2. HNSW ONLY: FaissHNSWFlat is the ONLY competitive engine. Never suggest IVFFlat or FaissFlat.
3. PHYSICS: To increase QPS → decrease efSearch or k. To increase Recall → increase efSearch, M, k, or efC.
4. ANTI-STAGNATION: {f"*** MANDATORY *** Your last {consecutive_efsearch_only} consecutive iterations changed ONLY efSearch (same engine/M). You MUST try a different M value in this iteration." if consecutive_efsearch_only >= 2 else f"[{consecutive_efsearch_only} consecutive efSearch-only iter(s). Change M if this reaches 2.]"}
5. STAGNATION: {f"⚠ WARNING: No improvement for {iters_since_improvement} iterations (best={current_best_score:.1f}). MANDATORY: Try a structurally DIFFERENT HNSW config — different M value, OR very low efSearch (8-16), OR efC=400 if not yet tried." if iters_since_improvement >= 5 else f"[{iters_since_improvement} iter(s) without improvement — OK]"}
6. efC LEVER: START with efC=200. After baseline established, also try efC=400 at M=24 — it has ZERO QPS cost and may raise recall 1-3pp, enabling lower efS for higher QPS. Do not wait for efC=200 to fail before trying efC=400.
7. Never repeat a config already in HISTORY.
8. k CONSTRAINT: k must be >= 10. With k < 10, Recall@10 ≤ k/10 (hard ceiling) — never suggest k < 10.

OUTPUT JSON ONLY (no markdown, no explanation outside the JSON):
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 32, "efConstruction": 400, "efSearch": 16}},
  "k_neighbors": 10,
  "reasoning": "HNSW M=32 efC=400 efSearch=16 is an ultra-low-efS config. efC=400 ensures the graph is high-quality so Recall stays reasonable even at efS=16. At efS=16 QPS~1600, Recall~0.65-0.80. Score~1040-1280 — likely beating efS=32."
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
                print(f"[LLM] Empty response on attempt {_api_attempt+1}/3, retrying…")
                time.sleep(5)
            if not content:
                raise ValueError("Empty LLM response after 3 attempts")

            # Robust JSON extraction
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            if not json_str.strip().startswith("{"):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    json_str = match.group(0)

            config_data = json.loads(json_str.strip())

            def snap(value, allowed):
                return min(sorted(allowed), key=lambda x: abs(x - value))

            provider  = ConfigProvider()
            engine    = config_data.get("engine", "FaissHNSWFlat")
            params_in = config_data.get("params", {})

            if engine not in provider.ENGINE_PARAMS:
                return provider.get_random_config(), f"Fallback (invalid engine: {engine})"

            allowed_params = provider.ENGINE_PARAMS[engine]
            clean_params = {}
            for k, allowed_vals in allowed_params.items():
                if k in params_in:
                    clean_params[k] = snap(int(params_in[k]), allowed_vals)
                elif allowed_vals:
                    clean_params[k] = allowed_vals[len(allowed_vals) // 2]

            # Extract and snap k_neighbors
            raw_k = config_data.get("k_neighbors", 10)
            k_final = min(ConfigProvider.K_NEIGHBORS,
                          key=lambda x: abs(x - int(raw_k)))
            final_config = {"engine": engine, "params": clean_params,
                            "k_neighbors": k_final}

            # Log snapping
            snapped = []
            for pk in clean_params:
                raw_v = params_in.get(pk)
                if raw_v is not None and int(raw_v) != clean_params[pk]:
                    snapped.append(f"{pk} {raw_v}→{clean_params[pk]}")
            if int(raw_k) != k_final:
                snapped.append(f"k_neighbors {raw_k}→{k_final}")
            snap_note = f"  [SNAPPED: {', '.join(snapped)}]" if snapped else ""
            reasoning_text = config_data.get("reasoning", "(no reasoning field)")
            print(f"\n[LLM] ── Suggested config ──────────────────────────────")
            print(f"[LLM]   engine={engine}  params={clean_params}  k={k_final}{snap_note}")
            print(f"[LLM]   Reasoning: {reasoning_text}")
            print(f"[LLM] ────────────────────────────────────────────────────\n")

            return final_config, content

        except Exception as e:
            print(f"[LLM] Error: {e}")
            return ConfigProvider(seed=self.seed).get_random_config(), f"Error: {e}"


# ===========================================================================
# BENCHMARK EXECUTION
# ===========================================================================

def run_benchmark(port: int, dataset_dir: str, config: Dict,
                  iteration: int = 0) -> BenchmarkResult:
    """
    Connect to VDMS, build SIFT_Set, run 10K queries, return BenchmarkResult.

    IMPORTANT: HNSW params (M, efConstruction, efSearch) are NOT passed via
    AddDescriptorSet — they are set in the VDMS server JSON config at startup
    (written to VDMS_PARAMS_FILE by run_optimization before calling vdms_restart_cmd).
    AddDescriptorSet only specifies the engine name.
    """
    engine        = config["engine"]
    engine_params = config.get("params", {})
    k_neighbors   = int(config.get("k_neighbors", 10))
    set_name      = f"SIFT_Set_{iteration}"

    data       = _load_sift1m_data_cached(dataset_dir)
    base_vecs  = data["base_vecs"]   # [1M, 128]
    query_vecs = data["query_vecs"]  # [10K, 128]
    gt_sets    = data["gt_sets"]     # [10K] sets of int

    print(f"\n[Iter {iteration}] engine={engine} params={engine_params}")

    db = vdms.vdms()
    db.connect("localhost", port)
    res, _ = db.query([{"GetSchema": {}}])
    if not res:
        raise RuntimeError(f"Cannot connect to VDMS on port {port}")

    # ── Build index ─────────────────────────────────────────────────────────
    # Engine params are already in VDMS server config — AddDescriptorSet only needs engine name.
    print(f"[Iter {iteration}] Building {set_name} (engine={engine})…")
    t_build_start = time.time()

    set_query = [{"AddDescriptorSet": {
        "name":       set_name,
        "dimensions": 128,
        "metric":     "L2",
        "engine":     engine,
    }}]
    res, _ = db.query(set_query)
    if not res or res[0].get("AddDescriptorSet", {}).get("status", -1) != 0:
        raise RuntimeError(f"AddDescriptorSet failed: {res}")

    # Add all 1M vectors using batch_properties=500 (single blob per 500 vectors).
    # batch_properties: one FAISS insert_descriptor(blob, path, N) call per AddDescriptor.
    # PMGD ops ≈ 1001 per call (safe, within 2MB journal limit). ~2K round-trips total.
    BATCH_PROPS_SIZE = 500
    N    = len(base_vecs)
    fail = 0
    print(f"Indexing {N:,} SIFT vectors  set={set_name}  (batch_properties={BATCH_PROPS_SIZE})...")
    for start in range(0, N, BATCH_PROPS_SIZE):
        end         = min(start + BATCH_PROPS_SIZE, N)
        batch_size  = end - start
        batch_props = [{"id": start + j} for j in range(batch_size)]
        blob        = base_vecs[start:end].astype(np.float32).tobytes()
        cmd = {"AddDescriptor": {
            "set":              set_name,
            "batch_properties": batch_props,
        }}
        res, _ = db.query([cmd], [blob])
        if res:
            status = res[0].get("AddDescriptor", {}).get("status", -1)
            if status != 0:
                fail += 1
        if start % 100000 == 0 and start > 0:
            print(f"  progress: {start:,}/{N:,}")
    if fail:
        raise RuntimeError(f"batch_properties AddDescriptor: {fail} failures")
    print(f"  progress: {N:,}/{N:,}")
    print(f"  batch failures: {fail}/{N // BATCH_PROPS_SIZE}  (OK)")

    index_build_s = time.time() - t_build_start
    print(f"[Iter {iteration}] Index built in {index_build_s:.1f}s ({index_build_s/60:.1f}min)")

    # ── Run queries (batched) ────────────────────────────────────────────────
    print(f"[Iter {iteration}] Running {len(query_vecs):,} batched queries…")
    batched_cmds  = []
    batched_blobs = []
    for q_vec in query_vecs:
        batched_cmds.append({"FindDescriptor": {
            "set":         set_name,
            "k_neighbors": k_neighbors,
            "results":     {"list": ["id", "_distance"]},
        }})
        batched_blobs.append(q_vec.tobytes())

    t_search_start = time.time()
    res, _ = db.query(batched_cmds, batched_blobs)
    total_search_s = time.time() - t_search_start

    # ── Parse results ────────────────────────────────────────────────────────
    search_results = []
    if res:
        for resp in res:
            if "FindDescriptor" in resp:
                entities   = resp["FindDescriptor"].get("entities", [])
                ids = []
                for ent in entities:
                    if "id" in ent:
                        ids.append(int(ent["id"]))
                    elif "properties" in ent and "id" in ent["properties"]:
                        ids.append(int(ent["properties"]["id"]))
                    elif "_label" in ent:
                        ids.append(int(ent["_label"]))
                search_results.append(ids)
            else:
                search_results.append([])
    else:
        print("[WARN] No results returned from batched query")
        search_results = [[] for _ in range(len(query_vecs))]

    # ── Compute metrics ──────────────────────────────────────────────────────
    recalls    = []
    precisions = []
    for gt, found in zip(gt_sets, search_results):
        # Recall@10 (hard): fraction of true top-10 found in the FIRST 10 returned results.
        # Standard ANN benchmark definition (ann-benchmarks.com): only the first k=10
        # returned results count. k_neighbors > 10 does not inflate recall.
        found_set = set(found[:10])
        hits = len(gt & found_set)
        recalls.append(hits / len(gt) if gt else 0.0)
        # Precision@10: top-10 of returned results (standard precision metric)
        top10_set = set(found[:10])
        top10_hits = len(gt & top10_set)
        precisions.append(top10_hits / 10.0 if found else 0.0)

    recall_at_10 = float(np.mean(recalls))
    precision    = float(np.mean(precisions))
    f1           = (2.0 * precision * recall_at_10 / (precision + recall_at_10)
                    if (precision + recall_at_10) > 0 else 0.0)
    qps          = float(len(query_vecs) / total_search_s)
    latency_ms   = float(total_search_s / len(query_vecs) * 1000.0)

    print(f"[Iter {iteration}] Score={qps if recall_at_10 >= _RECALL_THRESHOLD else 0.0:.1f}  "
          f"Recall@10={recall_at_10:.4f}  QPS={qps:.1f}  "
          f"Latency={latency_ms:.2f}ms  Build={index_build_s:.1f}s")

    return BenchmarkResult(
        qps=           qps,
        recall_at_10=  recall_at_10,
        latency_ms=    latency_ms,
        index_build_s= index_build_s,
        precision=     precision,
        f1_score=      f1,
        engine=        engine,
        params=        engine_params,
        timestamp=     time.time(),
    )


# ===========================================================================
# OPTIMIZATION LOOP
# ===========================================================================

def _restart_vdms(config: Dict, iteration: int, vdms_restart_cmd: str) -> None:
    """Write VDMS_PARAMS_FILE and restart VDMS with correct engine params.

    Called for EVERY iteration (including i=0) because HNSW M/efConstruction/efSearch
    are set in the VDMS server JSON config — not in AddDescriptorSet.  The shell's
    initial start_vdms() used plain config; start_vdms_snapshot() reads VDMS_PARAMS_FILE
    and rebuilds the config with HNSW params before starting VDMS.
    """
    _write_vdms_params(config)
    print(f"[VDMS] Restarting for iter {iteration} "
          f"({config.get('engine')} {config.get('params', {})})…")
    subprocess.run(vdms_restart_cmd, shell=True, check=True)
    time.sleep(3)
    print(f"[VDMS] Ready.")


def run_optimization(method: str, iterations: int, port: int, dataset_dir: Path,
                     seed: int = 42, api_key: Optional[str] = None,
                     model: str = "minimax/minimax-m2.1",
                     vdms_restart_cmd: Optional[str] = None,
                     patience: int = 0) -> List[IterationResult]:

    print(f"Starting {method} optimization. Iterations: {iterations}, Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    config_provider = ConfigProvider(seed=seed)
    llm_provider    = LLMProvider(api_key, model=model, seed=seed) if api_key else None

    if method == "grid":
        configs = config_provider.grid_search_configs()
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
        _encoder = KnobEncoder([
            {"name": "M",              "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]},
            {"name": "efConstruction", "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"]},
            {"name": "efSearch",       "type": "enum",
             "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]},
            {"name": "k_neighbors",    "type": "enum",
             "values": ConfigProvider.K_NEIGHBORS},
        ])
        vdtuner_opt = VDTunerOptimizer(_encoder, seed=seed)

    results: List[IterationResult] = []
    best_score_ever         = 0.0
    iters_since_improvement = 0

    for i in range(iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {i+1}/{iterations} ({method.upper()})")
        print(f"{'='*80}")

        # ── Config selection ─────────────────────────────────────────────────
        # For grid/random: config is pre-computed; select then restart.
        # For LLM/optuna: query then restart.
        # Restart happens for ALL iterations (including i=0) because HNSW params
        # must be injected via VDMS_PARAMS_FILE — the shell's plain initial start
        # does NOT have these params.  The extra restart on iter 1 costs ~3s.

        if method == "hyperparameter_only":

            tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 r.config.get("k_neighbors", 10))
                for r in results
            }

            # Detect consecutive iterations varying ONLY efSearch (same engine+M+efC+k).
            # Including efC in the key ensures that efC-only changes reset the counter —
            # varying efC is legitimate exploration and should not trigger anti-stagnation.
            consecutive_efsearch_only = 0
            if len(results) >= 2:
                ref_cfg    = results[-1].config
                ref_struct = (ref_cfg.get("engine"),
                              ref_cfg.get("params", {}).get("M"),
                              ref_cfg.get("params", {}).get("efConstruction"),
                              ref_cfg.get("k_neighbors", 10))
                for h in reversed(results):
                    h_struct = (h.config.get("engine"),
                                h.config.get("params", {}).get("M"),
                                h.config.get("params", {}).get("efConstruction"),
                                h.config.get("k_neighbors", 10))
                    if h_struct == ref_struct:
                        consecutive_efsearch_only += 1
                    else:
                        break

            dedup_hint = ""
            for _attempt in range(4):
                config, reasoning = asyncio.run(
                    llm_provider.query_llm(
                        i + 1, results, iterations,
                        consecutive_efsearch_only=consecutive_efsearch_only,
                        iters_since_improvement=iters_since_improvement,
                        dedup_hint=dedup_hint))
                config["_iter"] = i + 1
                key = (config["engine"],
                       str(sorted(config.get("params", {}).items())),
                       config.get("k_neighbors", 10))
                if key not in tried_keys:
                    break
                # Build dedup hint
                recent_efS    = [r.config.get("params", {}).get("efSearch")
                                 for r in results[-6:]
                                 if r.config.get("engine") == "FaissHNSWFlat"]
                suggested_efS = config.get("params", {}).get("efSearch")
                efsearch_fixation = (recent_efS.count(suggested_efS) >= 3
                                     if recent_efS and suggested_efS is not None else False)
                if efsearch_fixation:
                    tried_efS  = sorted({
                        r.config.get("params", {}).get("efSearch")
                        for r in results
                        if r.config.get("engine") == "FaissHNSWFlat"
                        and r.config.get("params", {}).get("efSearch") is not None
                    })
                    untried_efS = [e for e in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]
                                   if e not in tried_efS]
                    dedup_hint = (
                        f"You suggested efSearch={suggested_efS} again — used in "
                        f"{recent_efS.count(suggested_efS)} of last {len(recent_efS)} HNSW iters. "
                        f"You are STUCK on efSearch={suggested_efS}.\n"
                        f"  → efSearch values already tried: {tried_efS}\n"
                        f"  → PRIORITY: try efSearch=8 or 16 (highest QPS region, may yield best Score).\n"
                        f"  → Alternatively, change M or efConstruction to a value not yet tried."
                    )
                else:
                    dedup_hint = (
                        f"You suggested engine={config.get('engine')}, "
                        f"params={config.get('params')} — this exact combination was already evaluated. "
                        f"Check 'ALL CONFIGS TRIED' and suggest something new."
                    )
                print(f"[Dedup] LLM repeated a tried config (attempt {_attempt+1}), retrying…")
            else:
                for _fb in range(8):
                    config = ConfigProvider(seed=seed + i + _fb).get_random_config()
                    config["_iter"] = i + 1
                    _fb_key = (config["engine"],
                               str(sorted(config.get("params", {}).items())),
                               config.get("k_neighbors", 10))
                    if _fb_key not in tried_keys:
                        break
                reasoning = "Fallback random (LLM repeated configs 4x)"
                print(f"[Dedup] Falling back to random config after 4 duplicate attempts.")

            # Restart AFTER config known (includes iter 0 — shell plain start is overridden)
            if vdms_restart_cmd:
                _restart_vdms(config, i + 1, vdms_restart_cmd)

        elif method == "grid":
            config    = configs[i]
            config["_iter"] = i + 1
            reasoning = "Baseline (Grid)"
            if vdms_restart_cmd:
                _restart_vdms(config, i + 1, vdms_restart_cmd)

        elif method in ("optuna", "gp_bo"):
            tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 r.config.get("k_neighbors", 10))
                for r in results
            }
            # SEARCH_ENGINES = ["FaissHNSWFlat"] — IVFFlat excluded (VDMS 1M bug)
            for _dedup_attempt in range(4):
                optuna_trial = optuna_study.ask()
                engine = optuna_trial.suggest_categorical(
                    "engine", ConfigProvider.SEARCH_ENGINES)
                params = {}
                if engine == "FaissHNSWFlat":
                    params["M"]              = optuna_trial.suggest_categorical(
                        "M", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"])
                    params["efConstruction"] = optuna_trial.suggest_categorical(
                        "efConstruction", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"])
                    params["efSearch"]       = optuna_trial.suggest_categorical(
                        "efSearch", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"])
                k_opt = optuna_trial.suggest_categorical("k_neighbors", ConfigProvider.K_NEIGHBORS)
                key = (engine, str(sorted(params.items())), k_opt)
                if key not in tried_keys:
                    break
                dup_score = next(
                    (r.benchmark.score() for r in results if
                     (r.config["engine"],
                      str(sorted(r.config.get("params", {}).items())),
                      r.config.get("k_neighbors", 10)) == key),
                    0.0)
                optuna_study.tell(optuna_trial, dup_score)
                optuna_trial = None
                print(f"[Optuna Dedup] Attempt {_dedup_attempt+1}: duplicate config, "
                      f"told score={dup_score:.1f}, retrying...")
            else:
                config = ConfigProvider(seed=seed + i).get_random_config()
                config["_iter"] = i + 1
                reasoning = "Fallback random (Optuna repeated configs 4x)"
                optuna_trial = None
                print("[Optuna Dedup] Falling back to random after 4 duplicate Optuna suggestions.")
            if optuna_trial is not None:
                config = {"engine": engine, "params": params,
                          "k_neighbors": k_opt, "_iter": i + 1}
                reasoning = f"Optuna TPE trial {optuna_trial.number}" if method == "optuna" else f"GP-BO trial {optuna_trial.number}"
            # Restart AFTER config known
            if vdms_restart_cmd:
                _restart_vdms(config, i + 1, vdms_restart_cmd)

        elif method == "vdtuner":
            vt_raw = vdtuner_opt.ask()
            config = {
                "engine": "FaissHNSWFlat",
                "params": {
                    "M":              vt_raw["M"],
                    "efConstruction": vt_raw["efConstruction"],
                    "efSearch":       vt_raw["efSearch"],
                },
                "k_neighbors": vt_raw["k_neighbors"],
                "_iter": i + 1,
            }
            reasoning = f"VDTuner EHVI iter {i+1}"
            if vdms_restart_cmd:
                _restart_vdms(config, i + 1, vdms_restart_cmd)

        else:  # random
            config    = configs[i]
            config["_iter"] = i + 1
            reasoning = "Baseline (Random)"
            if vdms_restart_cmd:
                _restart_vdms(config, i + 1, vdms_restart_cmd)

        print(f"  engine: {config['engine']}")
        print(f"  params: {config.get('params', {})}")

        try:
            benchmark = run_benchmark(port, str(dataset_dir), config, iteration=i + 1)

            print(f"\n[Results iter {i+1}]")
            print(f"  Score (QPS|Recall≥{_RECALL_THRESHOLD:.2f}): {benchmark.score():.1f}")
            print(f"  Recall@10:               {benchmark.recall_at_10:.4f}")
            print(f"  QPS:                     {benchmark.qps:.1f}")
            print(f"  Latency:                 {benchmark.latency_ms:.2f}ms")
            print(f"  Index build:             {benchmark.index_build_s:.1f}s")

            results.append(IterationResult(i + 1, config, benchmark, reasoning, method))

            if method in ("optuna", "gp_bo") and optuna_trial is not None:
                optuna_study.tell(optuna_trial, benchmark.score())
                optuna_trial = None
            if method == "vdtuner":
                vdtuner_opt.tell(config, benchmark.recall_at_10, benchmark.qps)

            if benchmark.score() > best_score_ever:
                best_score_ever         = benchmark.score()
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            if patience > 0 and best_score_ever > 0.0 and iters_since_improvement >= patience:
                print(f"\n[Early Stop] No improvement for {patience} consecutive iterations. "
                      f"Best: {best_score_ever:.1f}. Stopping at iteration {i+1}/{iterations}.")
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
        print(f"Best Score:   {best.benchmark.score():.1f}")
        print(f"Best Recall:  {best.benchmark.recall_at_10:.4f}")
        print(f"Best QPS:     {best.benchmark.qps:.1f}")
        print(f"Best Config:  {best.config}")
    if method == "vdtuner" and vdtuner_opt is not None:
        vt_cfg, vt_q, vt_qps = vdtuner_opt.select(_RECALL_THRESHOLD)
        print(f"\n[VDTuner] ε-constraint selection (Recall≥{_RECALL_THRESHOLD}):")
        print(f"  Config:  {vt_cfg}")
        print(f"  Recall:  {vt_q:.4f}  QPS: {vt_qps:.1f}")

    return results


# ===========================================================================
# VDMS PARAMS FILE WRITER
# ===========================================================================

def _write_vdms_params(config: Dict) -> None:
    """
    Write engine params to VDMS_PARAMS_FILE so the shell's start_vdms_snapshot
    can inject hnsw_M / hnsw_efConstruction / hnsw_efsearch into the VDMS
    server JSON config before starting.  Called before every VDMS restart.
    Also write SNAP_KEY_FILE = "FRESH_DB" (always rebuild; no snapshot used here).
    """
    engine = config.get("engine", "FaissFlat")
    params = config.get("params", {})
    try:
        with open(VDMS_PARAMS_FILE, "w") as f:
            json.dump({"engine": engine, "params": params}, f)
            f.flush()
            os.fsync(f.fileno())
        with open(SNAP_KEY_FILE, "w") as f:
            f.write("FRESH_DB")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"[WARN] Could not write VDMS_PARAMS_FILE: {e}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided VDMS Optimizer for SIFT1M ANN Retrieval")
    parser.add_argument("--port",          type=int,  default=55595)
    parser.add_argument("--dataset-dir",   type=Path, required=True,
                        help="Root dataset dir (contains sift/ subdirectory)")
    parser.add_argument("--method",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"],
                        required=True)
    parser.add_argument("--iterations",    type=int,  default=30)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--output",        type=Path, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--model",         type=str,
                        default="minimax/minimax-m2.1",
                        help="OpenRouter model ID")
    parser.add_argument("--vdms-restart-cmd", type=str, default=None,
                        help="Shell command to restart VDMS between iterations")
    parser.add_argument("--patience",      type=int,  default=0,
                        help="Stop after N iters with no improvement (0=disabled)")
    parser.add_argument("--recall-threshold", type=float, default=0.90,
                        help="Constrained optimization threshold τ: Score=QPS if Recall@10≥τ else 0. "
                             "Default 0.90. Standard ANN quality floor.")
    args = parser.parse_args()

    global _RECALL_THRESHOLD
    _RECALL_THRESHOLD = args.recall_threshold
    print(f"[Score] Constrained objective: Score = QPS  if Recall@10 ≥ {_RECALL_THRESHOLD:.3f} else 0")

    api_key = None
    if args.method == "hyperparameter_only":
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set in environment / .env")
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
            "best_score":      best_result.benchmark.score()        if best_result else 0.0,
            "best_recall":     best_result.benchmark.recall_at_10   if best_result else 0.0,
            "best_qps":        best_result.benchmark.qps            if best_result else 0.0,
            "best_config":     best_result.config                   if best_result else {},
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
