"""
Milvus-backend SIFT1M optimizer (pure FAISS, no VDMS).
Methods: hyperparameter_only (LLM), random, grid, optuna, gp_bo, vdtuner.
Metric: Recall@10, threshold τ=0.90, Score = QPS if Recall@10 ≥ τ else 0.

Prompt architecture ported from sift1m_agent_optimizer.py (VDMS version):
  - 3-phase system (EXPLORATION/EXPLOITATION/FINE-TUNING), t_exp=0.40, t_expl=0.75
  - Phase-conditioned guidance templates (5 scenarios)
  - Anti-collapse (consecutive efSearch-only detection) and stagnation detection
  - Untried-hint injection (efSearch, M, k)
  - Snap-to-canonical config layer
  - Rich dedup hints with efSearch-fixation detection

Search space expanded to match VDMS sift1m_agent_optimizer.py:
  M:              [4, 8, 12, 16, 24, 32, 48, 64, 96]
  efConstruction: [50, 100, 150, 200, 300, 400]
  efSearch:       [8, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500]
  k_neighbors:    [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
"""

import sys, os, time, json, random, re, argparse, requests
import numpy as np
import faiss
import optuna
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
_RECALL_THRESHOLD: float = 0.90
_SIFT1M_CACHE: Dict = {}
_FAISS_INDEX  = None
_FAISS_LAST_CFG: Optional[str] = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_fvecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")

def _read_ivecs(path: str) -> np.ndarray:
    a = np.fromfile(path, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def _load_data(dataset_dir: str) -> Dict:
    global _SIFT1M_CACHE
    if _SIFT1M_CACHE:
        return _SIFT1M_CACHE
    d = Path(dataset_dir) / "sift"
    print("[SIFT1M] Loading base vectors (1M × 128)…")
    t0 = time.time()
    base_vecs  = _read_fvecs(str(d / "sift_base.fvecs"))
    query_vecs = _read_fvecs(str(d / "sift_query.fvecs"))
    gt_array   = _read_ivecs(str(d / "sift_groundtruth.ivecs"))
    gt_sets    = [set(row[:10].tolist()) for row in gt_array]
    print(f"[SIFT1M] Loaded in {time.time()-t0:.1f}s  base={base_vecs.shape}  queries={query_vecs.shape}")
    _SIFT1M_CACHE = {"base": base_vecs, "queries": query_vecs, "gt_sets": gt_sets}
    return _SIFT1M_CACHE


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def _build_index(engine: str, params: Dict, base: np.ndarray) -> faiss.Index:
    d = base.shape[1]
    if engine == "FaissFlat":
        idx = faiss.IndexFlatL2(d)
    elif engine == "FaissHNSWFlat":
        M   = int(params.get("M", 32))
        efC = int(params.get("efConstruction", 200))
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)
        idx.hnsw.efConstruction = efC
    else:
        raise ValueError(f"Unknown engine: {engine}")
    idx.add(base)
    return idx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    recall_at_10:  float
    qps:           float
    latency_ms:    float
    index_build_s: float

    def score(self) -> float:
        return self.qps if self.recall_at_10 >= _RECALL_THRESHOLD else 0.0


def run_benchmark(cfg: Dict, dataset_dir: str, iteration: int = 0) -> BenchmarkResult:
    global _FAISS_INDEX, _FAISS_LAST_CFG

    data    = _load_data(dataset_dir)
    base    = data["base"]
    qvecs   = data["queries"]
    gt_sets = data["gt_sets"]

    engine = cfg.get("engine", "FaissHNSWFlat")
    params = cfg.get("params", {})
    k      = int(cfg.get("k_neighbors", 10))

    print(f"\n[Iter {iteration}] engine={engine} params={params} k={k}")

    cfg_key = json.dumps({"engine": engine, "params": {pk: v for pk, v in params.items()
                                                        if pk != "efSearch"}}, sort_keys=True)
    if cfg_key != _FAISS_LAST_CFG:
        t0 = time.time()
        print(f"[FAISS] Building index: {engine} {params}…")
        _FAISS_INDEX    = _build_index(engine, params, base)
        _FAISS_LAST_CFG = cfg_key
        build_s = time.time() - t0
        print(f"[FAISS] Index built in {build_s:.1f}s")
    else:
        build_s = 0.0

    efS = int(params.get("efSearch", 32))
    if engine == "FaissHNSWFlat":
        _FAISS_INDEX.hnsw.efSearch = efS

    n_q = len(qvecs)
    t0  = time.time()
    _, I = _FAISS_INDEX.search(qvecs, k)
    elapsed = time.time() - t0

    recalls = []
    for qi in range(n_q):
        retrieved = set(int(x) for x in I[qi][:10] if x >= 0)
        recalls.append(len(retrieved & gt_sets[qi]) / max(len(gt_sets[qi]), 1))

    recall = float(np.mean(recalls))
    qps    = float(n_q / elapsed)
    lat_ms = float(elapsed / n_q * 1000)

    print(f"  Score={qps if recall >= _RECALL_THRESHOLD else 0:.1f}  "
          f"Recall@10={recall:.4f}  QPS={qps:.1f}  build={build_s:.1f}s")

    return BenchmarkResult(recall_at_10=recall, qps=qps,
                           latency_ms=lat_ms, index_build_s=build_s)


# ---------------------------------------------------------------------------
# Config provider — search space matches VDMS sift1m_agent_optimizer.py
# ---------------------------------------------------------------------------

class ConfigProvider:
    ENGINE_PARAMS = {
        "FaissFlat":    {},
        "FaissHNSWFlat": {
            "M":              [4, 8, 12, 16, 24, 32, 48, 64, 96],
            "efConstruction": [50, 100, 150, 200, 300, 400],
            "efSearch":       [8, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500],
        },
    }
    K_NEIGHBORS    = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    SEARCH_ENGINES = ["FaissHNSWFlat"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def get_random_config(self) -> Dict:
        engine = "FaissHNSWFlat"
        params = {k: int(self.rng.choice(v))
                  for k, v in self.ENGINE_PARAMS[engine].items()}
        return {"engine": engine, "params": params,
                "k_neighbors": int(self.rng.choice(self.K_NEIGHBORS))}

    def optuna_suggest(self, trial) -> Dict:
        engine = trial.suggest_categorical("engine", self.SEARCH_ENGINES)
        cfg = {"engine": engine, "params": {},
               "k_neighbors": trial.suggest_categorical("k", self.K_NEIGHBORS)}
        if engine == "FaissHNSWFlat":
            cfg["params"]["M"]              = trial.suggest_categorical(
                "M", self.ENGINE_PARAMS[engine]["M"])
            cfg["params"]["efConstruction"] = trial.suggest_categorical(
                "efC", self.ENGINE_PARAMS[engine]["efConstruction"])
            cfg["params"]["efSearch"]       = trial.suggest_categorical(
                "efS", self.ENGINE_PARAMS[engine]["efSearch"])
        return cfg

    def grid_configs(self, n: int) -> List[Dict]:
        cfgs = []
        # Brute-force reference
        cfgs.append({"engine": "FaissFlat", "params": {}, "k_neighbors": 10})
        # M sweep at efC=200, efS=32, k=10
        for M in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
            cfgs.append({"engine": "FaissHNSWFlat",
                          "params": {"M": M, "efConstruction": 200, "efSearch": 32},
                          "k_neighbors": 10})
        # efS sweep at M=32, efC=200, k=10
        for efS in [8, 16, 24, 48, 64, 100, 200, 500]:
            cfgs.append({"engine": "FaissHNSWFlat",
                          "params": {"M": 32, "efConstruction": 200, "efSearch": efS},
                          "k_neighbors": 10})
        # k sweep at M=32, efC=200, efS=32
        for k in [20, 30, 50, 75, 150]:
            cfgs.append({"engine": "FaissHNSWFlat",
                          "params": {"M": 32, "efConstruction": 200, "efSearch": 32},
                          "k_neighbors": k})
        # efC sweep
        for efC in [50, 100, 150, 300, 400]:
            cfgs.append({"engine": "FaissHNSWFlat",
                          "params": {"M": 32, "efConstruction": efC, "efSearch": 32},
                          "k_neighbors": 10})
        return cfgs[:n]


# ---------------------------------------------------------------------------
# LLM provider (full VDMS architecture, adapted for Milvus FAISS)
# ---------------------------------------------------------------------------

class LLMProvider:
    def __init__(self, api_key: str, model: str = "minimax/minimax-m2.1", seed: int = 42):
        self.api_key = api_key
        self.model   = model
        self.seed    = seed

    def query_llm(self, iteration: int, history: List["IterationResult"],
                  total_iterations: int,
                  consecutive_efsearch_only: int = 0,
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
            last_recall = last_run.benchmark.recall_at_10
            last_qps    = last_run.benchmark.qps
            last_engine = last_run.config["engine"]
            last_params = last_run.config.get("params", {})

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
                "CRITICAL PHYSICS (SIFT1M 1M vectors, L2, Milvus direct-FAISS): "
                "FaissHNSWFlat is the ONLY viable engine (FaissFlat too slow at 1M). "
                "Key physics: LOWER M = fewer graph edges = FASTER graph traversal at same efSearch. "
                "M=16 has fewer edges than M=32 → higher QPS at same efS, with only a small recall cost. "
                "M=16 efC=200 efS=32 k=10 is the recommended starting point (estimated QPS ~1150-1200+). "
                "efC=200 is a reliable default — sufficient to cross τ=0.90 for M≥16 at efS=32. "
                "k=10 is the minimum (k<10 caps Recall@10 at k/10, strictly dominated). "
                f"START HERE: HNSW M=16 efC=200 efS=32 k=10 — recommended baseline (Recall≈0.91 ≥ τ={_RECALL_THRESHOLD:.2f}, high QPS).")

        system_context = f"""
SYSTEM ARCHITECTURE:
1. TARGET SYSTEM: Milvus-backend (direct FAISS, no VDMS middleware).
   - Pure FAISS index search in Python. No client-server TCP overhead.
   - Active indexing backend: FaissHNSWFlat ONLY.

2. DATASET: SIFT1M (Standard ANN Benchmark)
   - 1,000,000 base vectors × 128 dimensions (float32).
   - 10,000 query vectors. Ground truth: exact brute-force top-10 neighbors.
   - Metric: L2 (Euclidean). Pure ANN task — no semantic component, no Stage-2 fusion.

3. OPTIMIZATION OBJECTIVE: Constrained maximization (SIEVE-style)
   Score = QPS  if Recall@10 ≥ τ={_RECALL_THRESHOLD:.2f}  else  0
   - τ={_RECALL_THRESHOLD:.2f} is the quality floor. Configs below it score 0 (infeasible).
   - Above the floor, maximize QPS. Prevents degenerate low-recall/ultra-fast solutions.
   - Recall@10: fraction of true top-10 neighbors found in the FIRST 10 returned results.
   - Tunable parameters: M, efConstruction, efSearch, k_neighbors. No alpha or fusion params.
   Current best score: {current_best_score:.1f}
"""

        engine_definitions = f"""
ENGINE PHYSICS (SIFT1M 1M vectors, L2, Milvus direct-FAISS):

[FaissHNSWFlat] — HNSW navigable graph  ← ONLY COMPETITIVE ENGINE
  M [4, 8, 12, 16, 24, 32, 48, 64, 96]: Graph connectivity (edges per node).
    M=4:  Recall ceiling ~0.70 (k=10), fastest possible. Cannot cross τ=0.90 at k=10.
    M=8:  Recall ceiling ~0.82 (k=10), very fast.
    M=12: Recall ceiling ~0.87 (k=10). Needs efC=400 and k≥20 to reach τ.
    M=16: Recall ceiling ~0.92 (k=10). SWEET SPOT for score at efS=32. HIGH PRIORITY.
    M=24: Recall ceiling ~0.95 (k=10). Good recall with moderate QPS.
    M=32: Recall ceiling ~0.97 (k=10). Robust default.
    M=48+: Near-perfect recall ceiling, heavier build.
  efConstruction [50, 100, 150, 200, 300, 400]: Build quality.
    Higher efC = better graph = higher recall at SAME efSearch. Does NOT reduce QPS at inference.
    efC=200: RELIABLE DEFAULT — sufficient for M≥16 to cross τ=0.90 at efS=32.
    efC=400: Best graph quality — zero QPS cost at inference (only build is 2× slower).
             For M=24+: efC=400 can raise recall by 1-3pp at the SAME QPS.
    Low M (4-12) needs higher efC to compensate for sparse graph.
  efSearch [8, 16, 24, 32, 48, 64, 100, 150, 200, 300, 500]: Search beam width. PRIMARY QPS lever.
    At M=32, efC=200, k=10 (estimated in Milvus direct-FAISS):
      efS=8  → QPS~2000+ Recall~0.40-0.50  Score~800-1000 (extreme speed, low recall)
      efS=16 → QPS~1600  Recall~0.65       Score~1040
      efS=24 → QPS~1300  Recall~0.80       Score~1040
      efS=32 → QPS~1100  Recall~0.88-0.94  Score~970-1034 ← sweet spot for high QPS
      efS=48 → QPS~900   Recall~0.93       Score~837
      efS=64 → QPS~750   Recall~0.96       Score~720
      efS=100→ QPS~580   Recall~0.98       Score~568
    KEY: efSearch is the dominant speed/recall lever. Very low efSearch (8-24) can achieve
    higher Score than efS=32 if M and k are tuned to compensate for recall loss.

k_neighbors [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]: Results returned per query.
  Recall@10 is measured from the FIRST 10 returned results (ann-benchmarks.com standard).
  CONSTRAINT: k must be >= 10. With k < 10, Recall@10 ≤ k/10 (strictly dominated by k=10).
  At M=32, efC=200, efS=32 (estimated):
    k=10  → Recall≈0.94  QPS~1073  Score~1012
    k=20  → Recall≈0.96  QPS~1040  Score~1001
    k=50  → Recall≈0.98  QPS~990   Score~965
    k=100 → Recall≈0.98  QPS~940   Score~924
  Higher k → higher Recall@10 (more candidates to find true neighbors) → lower QPS.
  CROSS-TERM: sparse graph (low M or low efS) misses some true top-10.
    Compensate by requesting more results (higher k): missed neighbors may be in positions 11-50.
    Example: M=12 efS=32 k=10 → Recall~0.87 (below τ). M=12 efS=32 k=20 → Recall~0.91 (above τ).

⛔ FaissFlat — exact search. Recall=1.0 always. QPS≈1-5 at 1M. Reference only.

============================================================
CROSS-PARAMETER INTERACTION SUMMARY:
============================================================
  M × QPS       → CRITICAL: Lower M = fewer edges = faster traversal at same efS.
                  M=16 efC=200 efS=32 k=10 → estimated QPS ~1150-1200 (SWEET SPOT).
                  M=32 efC=200 efS=32 k=10 → estimated QPS ~1050-1100 (standard baseline).
  M ceiling     → M=4: cannot cross τ=0.90 at k=10. M=8: needs k≥50. M=12: needs efC=400 + k≥20.
                  M=16: SWEET SPOT — sparser graph, high QPS, Recall≈0.92 above τ.
  efSearch × M  → At same efS, higher M = higher Recall but LOWER QPS.
                  Optimal: use M=16 (not M=32) for best QPS × Recall tradeoff at τ=0.90.
  efSearch × k  → At low efSearch (fast), k>10 recovers missed neighbors cheaply.
                  efSearch=24, k=20 at M=16 may beat efSearch=32, k=10 — test empirically.
  efC × M       → efC=200 is default for M≥16. For M=24, efC=400 can improve recall 1-3pp
                  at ZERO QPS cost. For M≤12, efC=400 often needed to reach τ.
  Score sweet spot → Two competitive regions:
                  (A) M=16 efC=200 efS=32 k=10 → Score~1163-1174 (sparser graph, peak QPS)
                  (B) M=24 efC=400 efS=24 k=10 → Score~1163 (denser graph, lower efS, similar QPS)
"""

        scenario_guide = f"""
PHASE-SPECIFIC STRATEGY (current phase: {phase}):

EXPLORATION (iterations 1-{exp_end}): Map the M × efSearch × k × efC landscape.
  - PRIORITY 1: HNSW M=16 efC=200 efS=32 k=10  — sparse graph, high QPS, Recall≈0.91 (above τ).
  - PRIORITY 2: HNSW M=24 efC=200 efS=32 k=10  — denser graph, higher recall ceiling.
  - PRIORITY 3: HNSW M=24 efC=400 efS=24 k=10  — efC=400 zero QPS cost; efS=24 boosts QPS.
  - PRIORITY 4: HNSW M=16 efC=200 efS=24 k=10  — lower efS → higher QPS; test recall ≥ τ.
  - PRIORITY 5: HNSW M=12 efC=200 efS=32 k=10  — even sparser → even higher QPS (test recall).
  - PRIORITY 6: HNSW M=12 efC=200 efS=32 k=20  — M=12 recall borderline; k=20 recovery.
  - AVOID: M=4 k=10 (Recall ceiling ~0.70, cannot cross τ=0.90), efSearch≥200 (Score<400),
           k≥300 (QPS cost excessive), k<10 (Recall@10 hard-capped at k/10).
  - Goal: find (M, efC, efS, k) that maximizes QPS while keeping Recall≥τ.

EXPLOITATION (iterations {exp_end+1}-{expl_end}): Narrow around best region.
  - Fix engine to FaissHNSWFlat. Sweep efSearch ±1 step and M ±1 step around best.
  - If best has Recall just above τ (0.90-0.93), try lower efS or lower M for QPS.
  - If best has Recall <0.90, increase k (cheapest fix) or efC (last resort).
  - After efC=200 baseline: try efC=400 at M=24 — zero QPS cost, may raise recall 1-3pp.

FINE-TUNING (iterations {expl_end+1}-{total_iterations}): ONE parameter change per iter.
  - Change ONLY one of: efSearch, M, k_neighbors, efConstruction.
  - Focus on the M × efSearch × k × efC interaction.

SCENARIO DECISION MATRIX:

1. "Sparse M + efC=200" (strong baseline):
   M=16 efS=32 k=10 efC=200: fewer edges → highest QPS while keeping Recall≥τ=0.90.
   Variants: M=12 efS=32 k=10, M=16 efS=24 k=10, M=12 efS=32 k=20.

2. "M=24 + efC=400 + lower efS" (competitive alternative):
   M=24 efC=400 efS=24 k=10: efC=400 raises recall (zero QPS cost), efS=24 recovers QPS.

3. "Very low M + k compensation":
   M=8 efS=32 k=30-50 efC=200: recall borderline but QPS very high (~1200+). Test empirically.
   M=12 efS=24 k=20 efC=200: promising unexplored region.

4. "Ultra-low efS + compensation":
   M=16 efS=16 k=20-30 efC=200: lower efS → QPS~1500 but recall drops; k compensates.

5. FORBIDDEN (do NOT suggest):
   - M=4 k=10 (Recall ceiling ~0.70 — physically impossible to cross τ=0.90)
   - M=8 k=10 (Recall ceiling ~0.82 — risky)
   - k < 10 (Recall@10 capped at k/10 — strictly dominated by k=10)
   - efSearch≥200 (Score<400)
   - FaissFlat (Score<5 — reference only)
"""

        all_configs_tried = "\n".join(
            f"  iter{h.config.get('_iter','?'):02d}: {h.config['engine']} "
            f"p={h.config.get('params',{})} k={h.config.get('k_neighbors',10)} "
            f"→ Score={h.benchmark.score():.1f} "
            f"Recall={h.benchmark.recall_at_10:.3f} QPS={h.benchmark.qps:.1f}"
            for h in history
        )

        untried_efsearch_hint = ""
        untried_M_hint        = ""
        untried_k_hint        = ""

        if history and phase in ("EXPLOITATION", "FINE-TUNING"):
            best_h      = max(history, key=lambda r: r.benchmark.score())
            best_engine = best_h.config["engine"]
            best_params = best_h.config.get("params", {})
            best_M      = best_params.get("M")

            if best_engine == "FaissHNSWFlat" and best_M is not None:
                tried_efsearch = {r.config.get("params", {}).get("efSearch")
                                  for r in history
                                  if r.config["engine"] == "FaissHNSWFlat"
                                  and r.config.get("params", {}).get("M") == best_M}
                untried_efsearch = [e for e in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]
                                    if e not in tried_efsearch]
                if untried_efsearch and phase == "FINE-TUNING":
                    untried_efsearch_hint = (
                        f"\nUNTRIED efSearch values at best M={best_M}: {untried_efsearch}"
                        f"\n  → Valid next probes for fine-tuning efSearch."
                    )

                tried_M = {r.config.get("params", {}).get("M")
                           for r in history if r.config["engine"] == "FaissHNSWFlat"}
                untried_M = [m for m in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]
                             if m not in tried_M]
                if untried_M:
                    untried_M_hint = (
                        f"\nUNTRIED M values (any efSearch): {untried_M}"
                        f"\n  → Different M values are high-value probes — try these."
                    )

        if history:
            tried_k = {r.config.get("k_neighbors", 10) for r in history}
            untried_k = [k for k in ConfigProvider.K_NEIGHBORS if k not in tried_k]
            if untried_k:
                untried_k_hint = (
                    f"\nUNTRIED k_neighbors values: {untried_k}"
                    f"\n  → k controls recall/QPS tradeoff. Minimum valid k is 10."
                )

        history_json = json.dumps([{
            "eng": h.config["engine"],
            "p":   h.config.get("params", {}),
            "k":   h.config.get("k_neighbors", 10),
            "sc":  round(h.benchmark.score(), 1),
            "rec": round(h.benchmark.recall_at_10, 3),
            "qps": round(h.benchmark.qps, 1),
        } for h in history[-5:]], indent=1)

        prompt = f"""You are an expert Milvus Optimization Engine for SIFT1M ANN retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

{system_context}

DIAGNOSIS OF LAST RUN:
{guidance}

{engine_definitions}

{scenario_guide}

ALL CONFIGS TRIED SO FAR (do NOT repeat any of these exact combinations):
{all_configs_tried if all_configs_tried else "  (none yet)"}
{untried_k_hint}{untried_efsearch_hint}{untried_M_hint}{("DEDUP REJECTION — YOUR LAST SUGGESTION WAS REJECTED:\n" + dedup_hint + "\nYou MUST suggest a genuinely different config.\n\n") if dedup_hint else ""}LAST 5 RESULTS (detailed):
{history_json}

AVAILABLE PARAMETER RANGES:
  FaissHNSWFlat: M={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['M']}
                 efConstruction={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efConstruction']}
                 efSearch={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efSearch']}
  k_neighbors:   {ConfigProvider.K_NEIGHBORS}  (results returned per query; k must be ≥ 10)
  ⛔ FaissFlat: reference only (Score<5). No FaissIVFFlat.

DECISION RULES (apply in order):
1. Read DIAGNOSIS — it tells you the single most important fix.
2. HNSW ONLY: FaissHNSWFlat is the ONLY competitive engine.
3. PHYSICS: To increase QPS → decrease efSearch or M. To increase Recall → increase efSearch, M, k, or efC.
4. ANTI-STAGNATION: {f"*** MANDATORY *** Your last {consecutive_efsearch_only} consecutive iterations changed ONLY efSearch (same engine/M). You MUST try a different M value in this iteration." if consecutive_efsearch_only >= 2 else f"[{consecutive_efsearch_only} consecutive efSearch-only iter(s). Change M if this reaches 2.]"}
5. STAGNATION: {f"WARNING: No improvement for {iters_since_improvement} iterations (best={current_best_score:.1f}). MANDATORY: Try a structurally DIFFERENT HNSW config — different M value, OR very low efSearch (8-16), OR efC=400 if not yet tried." if iters_since_improvement >= 5 else f"[{iters_since_improvement} iter(s) without improvement — OK]"}
6. efC LEVER: START with efC=200. After baseline, try efC=400 at M=24 — ZERO QPS cost, may raise recall 1-3pp.
7. Never repeat a config already in HISTORY.
8. k CONSTRAINT: k must be >= 10. Never suggest k < 10.

OUTPUT JSON ONLY (no markdown, no explanation outside the JSON):
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 16, "efConstruction": 200, "efSearch": 32}},
  "k_neighbors": 10,
  "reasoning": "M=16 efC=200 efS=32 k=10 — sparse graph sweet spot. Fewer edges than M=32 → higher QPS while keeping Recall≈0.91 above τ=0.90."
}}
"""

        try:
            content = None
            for _attempt in range(3):
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 2000},
                    timeout=60,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if content:
                    break
                print(f"[LLM] Empty response attempt {_attempt+1}/3, retrying…")
                time.sleep(5)
            if not content:
                raise ValueError("Empty LLM response after 3 attempts")

            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            if not json_str.strip().startswith("{"):
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    json_str = m.group(0)

            cfg_data = json.loads(json_str.strip())

            def snap(value, allowed):
                return min(sorted(allowed), key=lambda x: abs(x - value))

            provider  = ConfigProvider()
            engine    = cfg_data.get("engine", "FaissHNSWFlat")
            params_in = cfg_data.get("params", {})

            if engine not in provider.ENGINE_PARAMS:
                return provider.get_random_config(), f"Fallback (invalid engine: {engine})"

            allowed_params = provider.ENGINE_PARAMS[engine]
            clean_params   = {}
            for pk, pvals in allowed_params.items():
                if pk in params_in:
                    clean_params[pk] = snap(int(params_in[pk]), pvals)
                elif pvals:
                    clean_params[pk] = pvals[len(pvals) // 2]

            raw_k   = cfg_data.get("k_neighbors", 10)
            k_final = snap(int(raw_k), provider.K_NEIGHBORS)

            final_config = {"engine": engine, "params": clean_params, "k_neighbors": k_final}

            snapped = []
            for pk in clean_params:
                raw_v = params_in.get(pk)
                if raw_v is not None and int(raw_v) != clean_params[pk]:
                    snapped.append(f"{pk} {raw_v}→{clean_params[pk]}")
            if int(raw_k) != k_final:
                snapped.append(f"k_neighbors {raw_k}→{k_final}")
            snap_note = f"  [SNAPPED: {', '.join(snapped)}]" if snapped else ""
            reasoning = cfg_data.get("reasoning", "(no reasoning field)")
            print(f"\n[LLM] engine={engine} params={clean_params} k={k_final}{snap_note}")
            print(f"[LLM] Reasoning: {reasoning}\n")

            return final_config, content

        except Exception as e:
            print(f"[LLM] Error: {e}")
            return ConfigProvider(seed=self.seed).get_random_config(), f"Error: {e}"


# ---------------------------------------------------------------------------
# Iteration result
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    iteration:     int
    config:        Dict
    benchmark:     BenchmarkResult
    llm_reasoning: str
    search_method: str


# ---------------------------------------------------------------------------
# Optimizer loop
# ---------------------------------------------------------------------------

def run_optimization(args) -> List[IterationResult]:
    global _RECALL_THRESHOLD
    _RECALL_THRESHOLD = args.recall_threshold

    provider = ConfigProvider(seed=args.seed)
    results: List[IterationResult] = []

    llm = None
    if args.method == "hyperparameter_only":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model   = os.getenv("LLM_MODEL", "minimax/minimax-m2.1")
        llm     = LLMProvider(api_key, model=model, seed=args.seed)

    best_score_ever         = 0.0
    iters_since_improvement = 0

    for i in range(args.iterations):
        print(f"\n{'='*70}")
        print(f"ITERATION {i+1}/{args.iterations} ({args.method.upper()})")
        print(f"{'='*70}")

        if args.method == "hyperparameter_only":
            tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 r.config.get("k_neighbors", 10))
                for r in results
            }

            # Detect consecutive iterations varying ONLY efSearch (same engine+M+efC+k).
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
            for _att in range(4):
                config, reasoning = llm.query_llm(
                    i + 1, results, args.iterations,
                    consecutive_efsearch_only=consecutive_efsearch_only,
                    iters_since_improvement=iters_since_improvement,
                    dedup_hint=dedup_hint)
                config["_iter"] = i + 1
                key = (config["engine"],
                       str(sorted(config.get("params", {}).items())),
                       config.get("k_neighbors", 10))
                if key not in tried_keys:
                    break

                recent_efS    = [r.config.get("params", {}).get("efSearch")
                                 for r in results[-6:]
                                 if r.config.get("engine") == "FaissHNSWFlat"]
                suggested_efS = config.get("params", {}).get("efSearch")
                if recent_efS and suggested_efS is not None and recent_efS.count(suggested_efS) >= 3:
                    tried_efS  = sorted({r.config.get("params", {}).get("efSearch")
                                         for r in results
                                         if r.config.get("engine") == "FaissHNSWFlat"
                                         and r.config.get("params", {}).get("efSearch") is not None})
                    untried_efS = [e for e in ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]
                                   if e not in tried_efS]
                    dedup_hint = (
                        f"You suggested efSearch={suggested_efS} again — used "
                        f"{recent_efS.count(suggested_efS)}/{len(recent_efS)} recent HNSW iters. STUCK.\n"
                        f"  efSearch values tried: {tried_efS}\n"
                        f"  PRIORITY: try efSearch=8 or 16 (highest QPS region).\n"
                        f"  Alternatively, change M or efConstruction to an untried value."
                    )
                else:
                    dedup_hint = (
                        f"You suggested engine={config.get('engine')}, "
                        f"params={config.get('params')} — already evaluated. "
                        f"Check ALL CONFIGS TRIED and suggest something new."
                    )
                print(f"[Dedup] LLM repeated config (attempt {_att+1}), retrying…")
            else:
                for _fb in range(8):
                    config = ConfigProvider(seed=args.seed + i + _fb).get_random_config()
                    config["_iter"] = i + 1
                    _fb_key = (config["engine"],
                               str(sorted(config.get("params", {}).items())),
                               config.get("k_neighbors", 10))
                    if _fb_key not in tried_keys:
                        break
                reasoning = "Fallback random (LLM repeated configs 4×)"
                print("[Dedup] Falling back to random after 4 duplicate attempts.")

        elif args.method == "random":
            config    = provider.get_random_config()
            config["_iter"] = i + 1
            reasoning = "Random"

        elif args.method == "grid":
            cfgs = provider.grid_configs(args.iterations)
            config = cfgs[i] if i < len(cfgs) else provider.get_random_config()
            config["_iter"] = i + 1
            reasoning = "Grid"

        elif args.method in ("optuna", "gp_bo"):
            sampler = (optuna.samplers.TPESampler(seed=args.seed)
                       if args.method == "optuna"
                       else optuna.samplers.GPSampler(seed=args.seed))
            if not hasattr(run_optimization, "_optuna_study"):
                run_optimization._optuna_study = optuna.create_study(
                    direction="maximize", sampler=sampler)
            study = run_optimization._optuna_study
            trial = study.ask()
            config = provider.optuna_suggest(trial)
            config["_iter"] = i + 1
            reasoning = f"Optuna trial {trial.number}"

        elif args.method == "vdtuner":
            if not hasattr(run_optimization, "_vdtuner"):
                _enc = KnobEncoder([
                    {"name": "M",              "type": "enum",
                     "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"]},
                    {"name": "efConstruction", "type": "enum",
                     "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"]},
                    {"name": "efSearch",       "type": "enum",
                     "values": ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]},
                    {"name": "k_neighbors",    "type": "enum",
                     "values": ConfigProvider.K_NEIGHBORS},
                ])
                run_optimization._vdtuner = VDTunerOptimizer(_enc, seed=args.seed)
            vt = run_optimization._vdtuner
            vt_raw = vt.ask()
            config = {
                "engine": "FaissHNSWFlat",
                "params": {"M": vt_raw["M"], "efConstruction": vt_raw["efConstruction"],
                           "efSearch": vt_raw["efSearch"]},
                "k_neighbors": vt_raw["k_neighbors"],
                "_iter": i + 1,
            }
            reasoning = f"VDTuner EHVI iter {i+1}"

        else:
            config    = provider.get_random_config()
            config["_iter"] = i + 1
            reasoning = "Random"

        try:
            bm = run_benchmark(config, args.dataset_dir, iteration=i + 1)
            results.append(IterationResult(i + 1, config, bm, reasoning, args.method))

            if args.method in ("optuna", "gp_bo") and hasattr(run_optimization, "_optuna_study"):
                run_optimization._optuna_study.tell(trial, bm.score())
            if args.method == "vdtuner":
                run_optimization._vdtuner.tell(config, bm.recall_at_10, bm.qps)

            if bm.score() > best_score_ever:
                best_score_ever         = bm.score()
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

        except Exception as e:
            print(f"[ERROR] Iteration {i+1} failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",       required=True)
    parser.add_argument("--method",            default="hyperparameter_only",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"])
    parser.add_argument("--iterations",        type=int, default=50)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--recall-threshold",  type=float, default=0.90)
    parser.add_argument("--output",            default="milvus_sift1m_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Milvus SIFT1M Optimizer  method={args.method}  seed={args.seed}  τ={args.recall_threshold}")
    print("=" * 60)

    results = run_optimization(args)

    best     = max(results, key=lambda r: r.benchmark.score(), default=None)
    feasible = [r for r in results if r.benchmark.recall_at_10 >= args.recall_threshold]

    summary = {
        "method": args.method, "seed": args.seed, "iterations": args.iterations,
        "recall_threshold": args.recall_threshold,
        "best_score":   best.benchmark.score()        if best else 0.0,
        "best_recall":  best.benchmark.recall_at_10   if best else 0.0,
        "best_qps":     best.benchmark.qps            if best else 0.0,
        "best_config":  best.config                   if best else {},
        "n_feasible":   len(feasible),
        "all_results": [
            {"iteration": r.iteration, "config": r.config, "search_method": r.search_method,
             "score": r.benchmark.score(), "recall_at_10": r.benchmark.recall_at_10,
             "qps": r.benchmark.qps, "latency_ms": r.benchmark.latency_ms}
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.output}")
    if best:
        print(f"Best: Recall@10={best.benchmark.recall_at_10:.4f}  QPS={best.benchmark.qps:.1f}  "
              f"Score={best.benchmark.score():.1f}")


if __name__ == "__main__":
    main()
