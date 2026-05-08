"""
Milvus-backend GLDv2 optimizer (pure FAISS, no VDMS).
Methods: hyperparameter_only (LLM), random, grid, optuna, gp_bo, vdtuner.
Stage 1: FAISS IP search on CLIP features.
Stage 1b: AQE (numpy).
Stage 2: DINOv2 reranking (numpy).
Metric: mAP, threshold τ=0.15, Score = QPS if mAP ≥ τ else 0.

Prompt architecture ported from gldv2_agent_optimizer.py (VDMS version):
  - 3-phase system (EXPLORATION/EXPLOITATION/FINE-TUNING), t_exp=0.40, t_expl=0.75
  - Phase-conditioned guidance templates (5 scenarios)
  - Anti-collapse and stagnation detection
  - Untried-hint injection (alpha, refs)
  - Snap-to-canonical config layer
  - Rich dedup hints with k-fixation detection

Key Milvus vs VDMS physics difference:
  - In VDMS batch mode, efSearch had MINIMAL QPS impact (PMGD round-trip dominated).
  - In Milvus direct-FAISS, efSearch IS a real secondary QPS lever (pure ANN traversal cost).
  - k remains the PRIMARY QPS lever at 762K scale.
"""

import sys, os, time, json, random, re, argparse, requests
import numpy as np
import faiss
import optuna
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../gldv2"))
from benchmark_gldv2 import (
    load_gldv2_data,
    build_prf_query, rerank_with_dinov2,
    compute_ap, compute_precision_at_k, compute_recall_at_k, compute_ndcg_at_k,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vdtuner_ehvi import VDTunerOptimizer, KnobEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
_MAP_THRESHOLD: float = 0.15
_GLDV2_CACHE: Dict = {}
_FAISS_INDEX  = None
_FAISS_LAST_CFG: Optional[str] = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_data(dataset_dir: str) -> Dict:
    global _GLDV2_CACHE
    if not _GLDV2_CACHE:
        _GLDV2_CACHE = load_gldv2_data(dataset_dir)
    return _GLDV2_CACHE


# ---------------------------------------------------------------------------
# FAISS index (IP metric for CLIP)
# ---------------------------------------------------------------------------

def _build_index(engine: str, params: Dict, clip_db: np.ndarray) -> faiss.Index:
    d = clip_db.shape[1]
    if engine == "FaissFlat":
        idx = faiss.IndexFlatIP(d)
    elif engine == "FaissHNSWFlat":
        M   = int(params.get("M", 32))
        efC = int(params.get("efConstruction", 200))
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = efC
    else:
        raise ValueError(f"Unknown engine: {engine}")
    vecs = clip_db / (np.linalg.norm(clip_db, axis=1, keepdims=True) + 1e-10)
    idx.add(vecs.astype(np.float32))
    return idx


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    map_score:       float
    qps:             float
    latency_ms:      float
    index_build_s:   float
    precision_at_10: float = 0.0
    recall_at_10:    float = 0.0

    def score(self) -> float:
        return self.qps if self.map_score >= _MAP_THRESHOLD else 0.0


def run_benchmark(cfg: Dict, dataset_dir: str, iteration: int = 0) -> BenchmarkResult:
    global _FAISS_INDEX, _FAISS_LAST_CFG

    data      = _load_data(dataset_dir)
    clip_db   = data["clip_db"]
    clip_q    = data["clip_q"]
    dinov2_db = data["dinov2_db"]
    dinov2_q  = data["dinov2_q"]
    rel_sets  = data["rel_sets"]
    junk_sets = data["junk_sets"]

    engine    = cfg.get("engine", "FaissHNSWFlat")
    params    = cfg.get("params", {})
    k         = int(cfg.get("k_neighbors", 100))
    alpha     = float(cfg.get("alpha", 0.30))
    n_refs    = int(cfg.get("n_refs", 1))
    ref_strat = str(cfg.get("ref_strategy", "first"))
    n_aqe     = int(cfg.get("n_aqe", 1))
    aqe_w     = float(cfg.get("aqe_weight", 0.0))

    print(f"\n[Iter {iteration}] engine={engine} params={params} k={k} "
          f"alpha={alpha:.2f} n_refs={n_refs}/{ref_strat} n_aqe={n_aqe} aqe_w={aqe_w:.2f}")

    cfg_key = json.dumps({"engine": engine,
                          "params": {k2: v for k2, v in params.items() if k2 != "efSearch"}},
                         sort_keys=True)
    if cfg_key != _FAISS_LAST_CFG:
        t0 = time.time()
        print(f"[FAISS] Building index: {engine} {params}…")
        _FAISS_INDEX    = _build_index(engine, params, clip_db)
        _FAISS_LAST_CFG = cfg_key
        build_s = time.time() - t0
        print(f"[FAISS] Index built in {build_s:.1f}s")
    else:
        build_s = 0.0

    efS = int(params.get("efSearch", 200))
    if engine == "FaissHNSWFlat":
        _FAISS_INDEX.hnsw.efSearch = efS

    q_norm  = clip_q  / (np.linalg.norm(clip_q,  axis=1, keepdims=True) + 1e-10)
    db_norm = clip_db / (np.linalg.norm(clip_db, axis=1, keepdims=True) + 1e-10)

    n_queries    = len(q_norm)
    search_start = time.time()

    D, I = _FAISS_INDEX.search(q_norm.astype(np.float32), k)

    aps, precs, recs, ndcgs = [], [], [], []
    t_rerank_total = 0.0

    for qi in range(n_queries):
        cand_ids   = [int(x) for x in I[qi] if x >= 0]
        cand_dists = [float(D[qi][j]) for j in range(len(cand_ids))]
        clip_results = [{"id": cand_ids[j], "distance": -cand_dists[j]}
                        for j in range(len(cand_ids))]

        cq = q_norm[qi]
        if n_aqe > 1 and aqe_w > 0.0 and len(cand_ids) >= n_aqe:
            n_refs_aqe = min(n_aqe - 1, len(cand_ids))
            q_exp = ((1.0 - aqe_w) * cq
                     + aqe_w * db_norm[np.array(cand_ids[:n_refs_aqe])].mean(axis=0))
            q_exp /= (np.linalg.norm(q_exp) + 1e-10)
            new_sims = db_norm[np.array(cand_ids)] @ q_exp
            clip_results = sorted(
                [{"id": cand_ids[j], "distance": -float(new_sims[j])}
                 for j in range(len(cand_ids))],
                key=lambda r: r["distance"],
            )

        t0    = time.perf_counter()
        dq    = dinov2_q[qi]
        aug_dq   = build_prf_query(dq, clip_results, dinov2_db, n_refs, ref_strat)
        reranked = rerank_with_dinov2(clip_results, aug_dq, dinov2_db, alpha)
        t_rerank_total += (time.perf_counter() - t0) * 1000

        ranked_ids = [r["id"] for r in reranked]
        aps.append(compute_ap(ranked_ids, rel_sets[qi], junk_sets[qi]))
        precs.append(compute_precision_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))
        recs.append(compute_recall_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))
        ndcgs.append(compute_ndcg_at_k(ranked_ids, rel_sets[qi], junk_sets[qi], k=10))

    total_time = time.time() - search_start
    map_score  = float(np.mean(aps))
    qps        = float(n_queries / total_time)
    lat_ms     = float(total_time / n_queries * 1000)

    print(f"  Score={qps if map_score >= _MAP_THRESHOLD else 0:.1f}  "
          f"mAP={map_score:.4f}  QPS={qps:.1f}  build={build_s:.1f}s")

    return BenchmarkResult(
        map_score=map_score, qps=qps, latency_ms=lat_ms,
        index_build_s=build_s,
        precision_at_10=float(np.mean(precs)),
        recall_at_10=float(np.mean(recs)),
    )


# ---------------------------------------------------------------------------
# Config provider
# ---------------------------------------------------------------------------

class ConfigProvider:
    ENGINE_PARAMS = {
        "FaissFlat":    {},
        "FaissHNSWFlat": {
            "M":              [8, 16, 32, 48, 64],
            "efConstruction": [100, 200, 400],
            "efSearch":       [32, 64, 100, 150, 200, 300, 500],
        },
    }
    K_NEIGHBORS_VALUES  = [50, 100, 150, 200, 300, 500, 750, 1000]
    ALPHA_VALUES        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                           0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
                           0.92, 0.95]
    N_REFS_VALUES       = [1, 3, 5, 10]
    REF_STRATEGY_VALUES = ["first", "centroid", "diverse"]
    N_AQE_VALUES        = [1, 3, 5, 10]
    AQE_WEIGHT_VALUES   = [0.1, 0.2, 0.3, 0.5]
    SEARCH_ENGINES      = ["FaissHNSWFlat"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def get_random_config(self) -> Dict:
        engine    = "FaissHNSWFlat"
        params    = {k: int(self.rng.choice(v))
                     for k, v in self.ENGINE_PARAMS[engine].items()}
        n_aqe     = int(self.rng.choice(self.N_AQE_VALUES))
        ref_strat = str(self.rng.choice(self.REF_STRATEGY_VALUES))
        n_refs    = 1 if ref_strat == "centroid" else int(self.rng.choice(self.N_REFS_VALUES))
        return {
            "engine":       engine,
            "params":       params,
            "k_neighbors":  int(self.rng.choice(self.K_NEIGHBORS_VALUES)),
            "alpha":        float(self.rng.choice(self.ALPHA_VALUES)),
            "n_refs":       n_refs,
            "ref_strategy": ref_strat,
            "n_aqe":        n_aqe,
            "aqe_weight":   float(self.rng.choice(self.AQE_WEIGHT_VALUES)) if n_aqe > 1 else 0.0,
        }

    def optuna_suggest(self, trial) -> Dict:
        M         = trial.suggest_categorical("M",   self.ENGINE_PARAMS["FaissHNSWFlat"]["M"])
        efC       = trial.suggest_categorical("efC", self.ENGINE_PARAMS["FaissHNSWFlat"]["efConstruction"])
        efS       = trial.suggest_categorical("efS", self.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"])
        k         = trial.suggest_categorical("k",   self.K_NEIGHBORS_VALUES)
        alpha     = trial.suggest_categorical("alpha", self.ALPHA_VALUES)
        ref_strat = trial.suggest_categorical("ref_strat", self.REF_STRATEGY_VALUES)
        n_refs    = 1 if ref_strat == "centroid" else trial.suggest_categorical("n_refs", self.N_REFS_VALUES)
        n_aqe     = trial.suggest_categorical("n_aqe", self.N_AQE_VALUES)
        aqe_w     = trial.suggest_categorical("aqe_w", self.AQE_WEIGHT_VALUES) if n_aqe > 1 else 0.0
        return {
            "engine": "FaissHNSWFlat",
            "params": {"M": M, "efConstruction": efC, "efSearch": efS},
            "k_neighbors": k, "alpha": alpha,
            "n_refs": n_refs, "ref_strategy": ref_strat,
            "n_aqe": n_aqe, "aqe_weight": aqe_w,
        }

    def grid_configs(self, n: int) -> List[Dict]:
        cfgs = []
        for M in [32, 64]:
            for efS in [64, 200, 500]:
                for k in [100, 200]:
                    for alpha in [0.30, 0.70, 0.90]:
                        cfgs.append({
                            "engine": "FaissHNSWFlat",
                            "params": {"M": M, "efConstruction": 200, "efSearch": efS},
                            "k_neighbors": k, "alpha": alpha,
                            "n_refs": 1, "ref_strategy": "centroid",
                            "n_aqe": 1, "aqe_weight": 0.0,
                        })
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
                  consecutive_alpha_only: int = 0,
                  iters_since_improvement: int = 0,
                  dedup_hint: str = "") -> Tuple[Dict, str]:

        exp_end  = max(1, int(round(total_iterations * 0.40)))
        expl_end = max(exp_end + 1, int(round(total_iterations * 0.75)))
        phase = ("EXPLORATION" if iteration <= exp_end else
                 "EXPLOITATION" if iteration <= expl_end else "FINE-TUNING")

        current_best_score = max(r.benchmark.score() for r in history) if history else 0.0

        if history:
            last_run      = history[-1]
            last_score    = last_run.benchmark.score()
            last_map      = last_run.benchmark.map_score
            last_qps      = last_run.benchmark.qps
            last_k        = last_run.config.get("k_neighbors", 200)
            last_alpha    = last_run.config.get("alpha", 0.0)
            last_engine   = last_run.config["engine"]
            last_params   = last_run.config.get("params", {})
            last_n_refs   = last_run.config.get("n_refs", 1)
            last_ref_strat= last_run.config.get("ref_strategy", "first")

            if last_score >= current_best_score * 0.98:
                if phase == "FINE-TUNING":
                    guidance = (
                        f"EXCELLENT. Near peak. {last_engine} k={last_k} alpha={last_alpha:.2f} "
                        f"n_refs={last_n_refs} ref_strategy={last_ref_strat}. "
                        f"Score={last_score:.4f}. Fine-tuning: change ONE parameter by ONE step: "
                        f"alpha +/-0.05, OR k_neighbors one step, OR try n_refs/ref_strategy variant.")
                else:
                    guidance = (
                        f"Near best so far ({last_engine} k={last_k} alpha={last_alpha:.2f} "
                        f"n_refs={last_n_refs} Score={last_score:.4f}). Phase={phase}: do NOT fine-tune yet. "
                        f"KEEP EXPLORING — try a structurally DIFFERENT region: "
                        f"ref_strategy=centroid n_refs=1 at k=150/200 (centroid uses ALL k — ALWAYS n_refs=1), "
                        f"OR n_refs=3/5 with ref_strategy=first or diverse, "
                        f"OR efConstruction=400 (better graph quality, +1-3% recall, same QPS), "
                        f"OR M=48 efC=400 + centroid, OR alpha=0.92/0.95. "
                        f"PRIMARY unexplored: centroid at k=150-200, efC=400, n_refs×first/diverse.")
            elif last_map < _MAP_THRESHOLD:
                guidance = (
                    f"Score=0 (mAP={last_map:.4f} below threshold τ={_MAP_THRESHOLD:.2f}). "
                    f"RESET to known-safe baseline: HNSW M=32 efC=400 efS=200 k=100 α=0.90 n_refs=1/centroid "
                    f"→ reliably achieves mAP above τ. "
                    f"Do NOT try to salvage the current config — it is in an infeasible region. "
                    f"After re-establishing feasibility, tune α=0.92/0.95 and reduce k for QPS.")
            elif last_map >= 0.05 and last_qps < 2.0:
                guidance = (
                    f"mAP acceptable ({last_map:.4f}) but QPS critically low ({last_qps:.1f}). "
                    f"This is likely FaissFlat or very large K. Switch to HNSW M=32 efS=200 "
                    f"with k=100-200.")
            elif last_score < current_best_score * 0.90:
                guidance = (
                    f"Score={last_score:.4f} significantly below best={current_best_score:.4f}. "
                    f"Last config: {last_engine} k={last_k} alpha={last_alpha:.2f} "
                    f"n_refs={last_n_refs} ref_strategy={last_ref_strat} params={last_params}. "
                    f"Return closer to best config and make targeted adjustments.")
            else:
                guidance = (
                    f"Score={last_score:.2f} (QPS). mAP={last_map:.4f} (≥τ={_MAP_THRESHOLD:.2f} ✓). QPS={last_qps:.1f}. "
                    f"mAP floor is met — now maximize QPS. Primary lever: reduce k_neighbors. "
                    f"Secondary lever: reduce efSearch (Milvus direct-FAISS: efSearch IS a real QPS lever). "
                    f"centroid ALWAYS requires n_refs=1. Change k to vary PRF quality. "
                    f"Try ref_strategy=centroid n_refs=1 at k=100/150, "
                    f"OR n_refs=3/5 with ref_strategy=first/diverse, "
                    f"OR efConstruction=400 (zero QPS cost, better recall), or alpha=0.92/0.95.")
        else:
            guidance = (
                f"First iteration — cold start. Objective: Score = QPS if mAP ≥ τ={_MAP_THRESHOLD:.2f} else 0. "
                "Feasibility floor τ must be met first; then maximize QPS. "
                "CRITICAL PHYSICS (762K Milvus FAISS IP vectors, batch mode): "
                "k IS the dominant QPS lever — smaller k = fewer ANN candidates = higher throughput. "
                "efSearch is a REAL secondary QPS lever (unlike VDMS where batch overhead dominated): "
                "lower efSearch = faster graph traversal = higher QPS (at recall cost). "
                "FaissIVFFlat: AVOID — excluded from Milvus port. "
                "TARGET mAP≥0.17 to maintain a safety buffer above τ=0.15. "
                "STEP 1 — SAFE BASELINE (start here): "
                "HNSW M=32 efC=400 efS=200 k=100 α=0.90 n_refs=1/centroid → mAP well above τ. "
                "STEP 2 — TUNE: reduce k for QPS, reduce efSearch for secondary QPS gain, "
                "tune alpha=0.92/0.95, explore AQE (n_aqe=5 aqe_weight=0.2). "
                "CENTROID RULE: centroid uses ALL k candidates; ALWAYS use n_refs=1 with centroid. "
                "START: HNSW M=32 efC=400 efS=200 k=100 α=0.90 n_refs=1/centroid n_aqe=1.")

        system_context = f"""
SYSTEM ARCHITECTURE:
1. TARGET SYSTEM: Milvus-backend (direct FAISS, no VDMS middleware).
   - Pure FAISS index search in Python. No client-server TCP overhead.
   - Indexing Backends: FaissHNSWFlat (primary), FaissFlat (reference only).
   - KEY PHYSICS DIFFERENCE FROM VDMS: In Milvus direct-FAISS mode, efSearch IS a real
     secondary QPS lever (pure graph traversal cost). In VDMS batch mode, TCP/PMGD overhead
     dominated, making efSearch negligible. Here: lower efSearch → higher QPS (real tradeoff).

2. DATASET: Google Landmarks Dataset v2 (GLDv2)
   - 762,460 DB images indexed (CLIP ViT-L/14, 768-d, L2-normed → IP search).
   - 1,129 query images (image-to-image retrieval — NOT text queries).
   - Ground truth: positive and junk image sets per query (Oxford/Paris style).
   - This is 17× larger than HICO-DET (47K images). Engine behavior differs at this scale.

3. RETRIEVAL PIPELINE (TWO STAGES — you tune BOTH stages simultaneously):

   Stage 1 — FAISS IP Search [CLIP ViT-L/14 — Image-to-Image Semantic Retrieval]:
     Query: 768-d CLIP image embedding.
     Index: HNSW IP index (you choose engine + engine params).
     Output: top-K candidate images (you choose k_neighbors).
     Cost: batch FAISS search over all 1,129 queries. k and efSearch both affect QPS.

   Stage 1b — AQE [Alpha Query Expansion]:
     Expands CLIP query by blending with top-(n_aqe-1) retrieved images:
       q_exp = (1-aqe_w)*q_orig + aqe_w*mean(clip_db[top_(n_aqe-1)])
     Sharpens query toward retrieved visual cluster. AQE adds a second FAISS pass (~2× cost).

   Stage 2 — DINOv2 Rerank [DINOv2 ViT-L/14-reg4 — Visual Appearance Refinement]:
     Fuse: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm (lower=better)
     alpha=0 → pure CLIP. alpha→1 → mostly DINOv2. Cost: ~5-15ms/query (negligible).
     PRF: n_refs>1 with first/diverse augments the DINOv2 query vector.
     CENTROID RULE: centroid uses ALL k candidates regardless of n_refs. ALWAYS use n_refs=1 with centroid.

4. OPTIMIZATION OBJECTIVE: Constrained maximization (SIEVE-style)
   Score = QPS  if mAP ≥ {_MAP_THRESHOLD:.2f}  else  0
   Current best score in this session: {current_best_score:.2f}

5. SCALE NOTES (GLDv2 762K — Milvus FAISS):
   - k_neighbors IS the primary QPS lever: smaller k = fewer neighbors computed = higher throughput.
   - efSearch IS a real secondary QPS lever (unlike VDMS): lower efSearch = faster traversal.
     Tradeoff: lower efSearch reduces recall. Use efSearch=200 for baseline, reduce for QPS gain.
   - FaissFlat: exact search at 762K → very low QPS (~1-5). Reference only.
"""

        engine_definitions = f"""
PARAMETER DEFINITIONS AND PHYSICS:

============================================================
GROUP A — STAGE-1 ENGINE PARAMETERS (affect FAISS search speed and CLIP recall)
============================================================

[A1. engine — Index Type]
  Choices: FaissHNSWFlat | FaissFlat
  - FaissFlat:     exact brute-force. Highest recall. No tunable params. QPS~1-5 at 762K.
  - FaissHNSWFlat: graph-based. THE ONLY PRACTICAL ENGINE at 762K scale.
                   efSearch has REAL (secondary) QPS impact in Milvus direct-FAISS mode.

[A2. M — HNSW graph connectivity] (FaissHNSWFlat only)
  ALLOWED VALUES: 8, 16, 32, 48, 64
  - Higher M = denser graph = higher CLIP recall at same efSearch. Higher M = longer build.
  - RECOMMENDED: M=32 for early baseline. Try M=48/64 once good k/alpha/efC found.
  - RELATIONSHIP WITH efConstruction: Higher M requires higher efC for a good graph.

[A3. efConstruction — HNSW build quality] (FaissHNSWFlat only)
  ALLOWED VALUES: 100, 200, 400
  - Higher efC = better-connected graph = higher recall at the same efSearch and M.
  - Does NOT affect QPS at inference time. Only costs more build time.
  - efC=200: standard quality. efC=400: best graph quality (recall +1-3pp vs efC=200).
  - RECOMMENDATION: try efC=400 after establishing k/alpha baseline — zero QPS cost.

[A4. efSearch — HNSW search beam width] (FaissHNSWFlat only)
  ALLOWED VALUES: 32, 64, 100, 150, 200, 300, 500
  - In Milvus direct-FAISS: efSearch has MODERATE real impact on QPS.
    Lower efSearch → faster graph traversal → higher QPS (but lower recall).
    Higher efSearch → more recall → lower QPS.
  - Unlike VDMS batch mode (where TCP overhead dominated), efSearch IS a tunable QPS lever here.
  - RECOMMENDATION: Start at efSearch=200. Reduce to 100-150 for QPS gain after establishing baseline.
    Do NOT drop below 64 without verifying mAP stays above τ.

============================================================
GROUP B — STAGE-2 RERANKING PARAMETERS (affect DINOv2 fusion)
============================================================

[B1. k_neighbors — Candidate pool size]
  ALLOWED VALUES: 50, 100, 150, 200, 300, 500, 750, 1000
  - PRIMARY QPS LEVER at 762K scale: smaller k = fewer neighbors per query = higher throughput.
  - Larger K = higher mAP ceiling (more true positives can surface through reranking).
  - SIEVE reference: k=100 α=0.90 efS=200 → establishes baseline score at iter 1.
  - k=50 is high priority for QPS gain. Always re-tune alpha when changing k.
  - CENTROID RULE: centroid uses ALL k candidates. ALWAYS n_refs=1 with centroid.

[B2. alpha — DINOv2 rerank weight]
  ALLOWED VALUES:
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95
  - Fusion: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm  (lower=better)
  - For landmark retrieval, DINOv2 helps distinguish exact location from near-duplicates.
  - Optimal alpha is typically 0.85-0.92 for GLDv2 landmark retrieval.

[B3. n_refs — DINOv2 PRF count]
  ALLOWED VALUES: 1, 3, 5, 10
  - n_refs=1: use query DINOv2 feature directly (no PRF).
  - n_refs>1 with first/diverse: augment query with top-(n_refs-1) candidates.
  - NOTE: n_refs>1 with first/diverse has shown poor results at k=100. Test cautiously.
  - CENTROID RULE: centroid uses ALL k candidates. ALWAYS use n_refs=1 with centroid.

[B4. ref_strategy — PRF aggregation]
  ALLOWED VALUES: "first", "centroid", "diverse"
  - "centroid": centroid of ALL k candidates' DINOv2 features. ALWAYS n_refs=1.
    To vary centroid quality, change k (more candidates = better centroid).
  - "first": top-(n_refs-1) candidates by CLIP rank.
  - "diverse": greedy farthest-point subset.
  - CENTROID is the most stable for landmark retrieval.

[B5. n_aqe / aqe_weight — ALPHA QUERY EXPANSION]
  n_aqe: [1, 3, 5, 10] — 1=disabled; >1=expand using top-(n_aqe-1) images
  aqe_weight: [0.1, 0.2, 0.3, 0.5] — blend strength (only when n_aqe>1)
  RULE: n_aqe=1 → aqe_weight MUST be 0.0. n_aqe>1 → choose aqe_weight from [0.1, 0.2, 0.3, 0.5].
  GUIDANCE:
    - n_aqe=5, aqe_weight=0.2 is a safe starting point
    - AQE adds ~2× search cost (extra FAISS pass) — must meaningfully raise mAP to keep Score competitive
    - AQE + centroid: AQE sharpens Stage-1 query → centroid aggregates better Stage-2 refs

============================================================
CROSS-PARAMETER INTERACTION SUMMARY (READ CAREFULLY):
============================================================
  k ↓         → QPS ↑ (PRIMARY lever). Test k=50 vs k=100 — large QPS difference.
  efSearch ↓  → QPS ↑ (SECONDARY lever in Milvus direct-FAISS). Lower efS at risk of mAP drop.
  efC=400     → recall ↑, zero QPS cost. Always worth testing after baseline established.
  centroid×k  → centroid uses ALL k candidates (n_refs IGNORED — always use n_refs=1).
                k=50: lower QPS cost, fewer centroid candidates. k=150: more candidates.
  n_refs×k    → for first/diverse ONLY: n_refs>1 with first/diverse risky (prior runs scored poorly).
  n_aqe×k     → AQE uses top-(n_aqe-1) images; adds ~2× query time (extra FAISS pass).
  alpha×k     → At k=50, centroid estimate noisier → optimal alpha may shift down (0.80-0.85).
                Always re-tune alpha when changing k.
  efC×M       → M=48 efC=400 may outperform M=32 efC=200 for recall at same QPS.
"""

        scenario_guide = f"""
PHASE-SPECIFIC STRATEGY (current phase: {phase}):

EXPLORATION (iterations 1-{exp_end}): Establish baseline, explore AQE, k, efSearch, and alpha.
  - PRIORITY 1: k=100 α=0.90 n_refs=1 centroid M=32 efC=400 efS=200 n_aqe=1 aqe_w=0.0 — establish baseline.
  - PRIORITY 2: n_aqe=5 aqe_weight=0.2 at baseline config — AQE sharpens Stage-1 query.
  - PRIORITY 3: k=50 n_refs=1 centroid α=0.92 efS=200 n_aqe=5 aqe_w=0.2 — higher QPS + AQE.
  - PRIORITY 4: k=100 efS=100 α=0.90 n_refs=1 centroid M=32 efC=400 — lower efSearch for QPS gain.
  - PRIORITY 5: k=150 n_refs=1 centroid α=0.90 n_aqe=5 aqe_w=0.2 — more candidates + AQE.
  - AVOID: k≥300 (QPS too low), n_aqe=1 with aqe_weight>0 (invalid).
  - Goal: determine optimal k × efSearch × AQE × alpha combination.

EXPLOITATION (iterations {exp_end+1}-{expl_end}): Narrow around best config found.
  - Fix engine/M/efC to best found values.
  - Sweep AQE at best (k, alpha, ref_strategy): try n_aqe ∈ {{3,5,10}} × aqe_weight ∈ {{0.1,0.2,0.3,0.5}}.
  - Sweep alpha ±0.05 around best alpha. Sweep k within {{50, 100, 150}} at best AQE.
  - Also sweep efSearch ±1 step around best efSearch value.

FINE-TUNING (iterations {expl_end+1}-{total_iterations}): ONE parameter change at a time.
  - Change only ONE of: alpha, n_aqe, aqe_weight, k, M, efSearch, ref_strategy.

SCENARIO DECISION MATRIX:

1. "Establish baseline" → k=100 n_refs=1 centroid M=32 efC=400 efS=200 α=0.90 n_aqe=1 aqe_w=0.0
   Run FIRST. AQE adds ~2× search cost — baseline shows pure HNSW performance.

2. "Moderate AQE" → n_aqe=5 aqe_weight=0.2 at best (k, alpha) — safe AQE starting point.

3. "QPS push" → k=50 efS=100 α=0.92 n_refs=1 centroid n_aqe=5 aqe_w=0.2 — both QPS levers.

4. "More candidates" → k=150 n_aqe=5 aqe_weight=0.2 α=0.90 centroid — better AQE reference pool.

5. AVOID:
   - k≥300 (QPS too low at 762K), n_aqe=1 with aqe_weight>0 (invalid)
   - n_refs>1 with first or diverse (poor results in prior runs — scored below τ or very low QPS)
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
            best_h          = max(history, key=lambda r: r.benchmark.score())
            best_engine     = best_h.config["engine"]
            best_k          = best_h.config.get("k_neighbors", 200)
            best_params     = best_h.config.get("params", {})
            best_alpha      = best_h.config.get("alpha", 0.0)
            best_n_refs     = best_h.config.get("n_refs", 1)
            best_ref_strat  = best_h.config.get("ref_strategy", "first")

            tried_alphas_here = {
                round(r.config.get("alpha", 0.0), 2)
                for r in history
                if r.config["engine"] == best_engine
                and r.config.get("k_neighbors", 200) == best_k
                and r.config.get("params", {}) == best_params
                and r.config.get("n_refs", 1) == best_n_refs
                and r.config.get("ref_strategy", "first") == best_ref_strat
            }
            untried_alphas = [a for a in ConfigProvider.ALPHA_VALUES
                              if round(a, 2) not in tried_alphas_here]
            if untried_alphas and phase == "FINE-TUNING":
                untried_alpha_hint = (
                    f"\nUNTRIED alpha values at best config "
                    f"({best_engine} k={best_k} n_refs={best_n_refs} "
                    f"ref_strategy={best_ref_strat} params={best_params}): {untried_alphas}"
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
                and not (rs == "centroid" and nr > 1)
            ]
            if untried_refs:
                untried_refs_hint = (
                    f"\nUNTRIED (n_refs, ref_strategy) combos at best (engine/params/k/alpha): "
                    f"{untried_refs[:8]}"
                    f"\n  → centroid n_refs=1 (vary k) and n_refs=3-5 first/diverse are highest-value probes."
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

        prompt = f"""You are an expert Milvus Optimization Engine for GLDv2 Landmark Retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

{system_context}

DIAGNOSIS OF LAST RUN:
{guidance}

{engine_definitions}

{scenario_guide}

ALL CONFIGS TRIED SO FAR (do NOT repeat any of these exact combinations):
{all_configs_tried if all_configs_tried else "  (none yet)"}
{untried_alpha_hint}{untried_refs_hint}{("DEDUP REJECTION — YOUR LAST SUGGESTION WAS REJECTED:\n" + dedup_hint + "\nYou MUST suggest a genuinely different config.\n\n") if dedup_hint else ""}LAST 5 RESULTS (detailed):
{history_json}

AVAILABLE PARAMETER RANGES:
  FaissHNSWFlat: M={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['M']}
                 efConstruction={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efConstruction']}
                 efSearch={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efSearch']}
  k_neighbors:   {ConfigProvider.K_NEIGHBORS_VALUES}
  alpha:         {ConfigProvider.ALPHA_VALUES}
  n_refs:        {ConfigProvider.N_REFS_VALUES}
  ref_strategy:  {ConfigProvider.REF_STRATEGY_VALUES}
  n_aqe:         {ConfigProvider.N_AQE_VALUES}  (1=disabled)
  aqe_weight:    {ConfigProvider.AQE_WEIGHT_VALUES}  (only when n_aqe>1)

DECISION RULES (apply in order):
1. Read DIAGNOSIS — it tells you the single most important fix.
2. Use CROSS-PARAMETER INTERACTIONS to avoid breaking other metrics.
3. In EXPLORATION phase: probe centroid n_refs=1 at k=100/150, efConstruction=400, lower efSearch.
   CENTROID RULE: ALWAYS n_refs=1 with centroid.
4. In FINE-TUNING phase: change only ONE parameter by ONE step.
5. n_refs and ref_strategy are FREE to vary — they are primary search dimensions.
6. n_aqe and aqe_weight are FREE to vary. When n_aqe=1, aqe_weight MUST be 0.0.
7. Never repeat a config already in HISTORY.
8. ANTI-COLLAPSE: {f"*** MANDATORY *** Your last {consecutive_alpha_only} consecutive iterations changed ONLY alpha (same engine/params/k/n_refs/ref_strategy). You MUST change efConstruction, OR M, OR k_neighbors, OR n_refs, OR ref_strategy in this iteration." if consecutive_alpha_only >= 2 else f"[{consecutive_alpha_only} consecutive alpha-only iter(s). Change efC/M/k/n_refs/ref_strategy if this reaches 2.]"}
9. STAGNATION: {f"WARNING: No improvement for {iters_since_improvement} iterations (best={current_best_score:.4f}). MANDATORY: Try a completely different (n_refs, ref_strategy) combination not yet tested, OR change k, OR change efSearch. Do NOT retry configs near the stuck point." if iters_since_improvement >= 5 else f"[{iters_since_improvement} iter(s) without improvement — OK]"}

OUTPUT JSON ONLY (no markdown, no explanation outside the JSON):
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 32, "efConstruction": 400, "efSearch": 200}},
  "k_neighbors": 100,
  "alpha": 0.90,
  "n_refs": 1,
  "ref_strategy": "centroid",
  "n_aqe": 5,
  "aqe_weight": 0.2,
  "reasoning": "AQE n_aqe=5 sharpens CLIP query. efC=400 for best graph quality. k=100 baseline QPS."
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
                m = re.search(r'\{.*\}', content, re.DOTALL)
                if m:
                    json_str = m.group(0)

            cfg_data = json.loads(json_str.strip())

            def snap(value, allowed):
                return min(sorted(allowed), key=lambda x: abs(x - value))

            provider = ConfigProvider()
            engine   = cfg_data.get("engine", "FaissHNSWFlat")
            params   = cfg_data.get("params", {})

            if engine not in provider.ENGINE_PARAMS:
                return provider.get_random_config(), f"Fallback (invalid engine: {engine})"

            allowed_params = provider.ENGINE_PARAMS[engine]
            clean_params   = {}
            for pk, pvals in allowed_params.items():
                if pk in params:
                    clean_params[pk] = snap(int(params[pk]), pvals)
                elif pvals:
                    clean_params[pk] = pvals[len(pvals) // 2]

            k_neighbors  = snap(int(cfg_data.get("k_neighbors", 200)), provider.K_NEIGHBORS_VALUES)
            alpha        = snap(float(cfg_data.get("alpha", 0.0)), provider.ALPHA_VALUES)
            ref_strat_r  = str(cfg_data.get("ref_strategy", "first"))
            ref_strategy = ref_strat_r if ref_strat_r in provider.REF_STRATEGY_VALUES else "first"
            n_refs_r     = int(cfg_data.get("n_refs", 1))
            n_refs       = 1 if ref_strategy == "centroid" else snap(n_refs_r, provider.N_REFS_VALUES)
            n_aqe_r      = int(cfg_data.get("n_aqe", 1))
            n_aqe        = snap(n_aqe_r, provider.N_AQE_VALUES)
            if n_aqe > 1:
                aqe_weight = snap(float(cfg_data.get("aqe_weight", 0.0)), provider.AQE_WEIGHT_VALUES)
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

            raw_alpha = float(cfg_data.get("alpha", 0.0))
            if abs(raw_alpha - alpha) > 1e-6:
                final_config["_raw_alpha"] = raw_alpha
            raw_k = int(cfg_data.get("k_neighbors", 200))
            if raw_k != k_neighbors:
                final_config["_raw_k"] = raw_k
            raw_naqe = int(cfg_data.get("n_aqe", 1))
            if raw_naqe != n_aqe:
                final_config["_raw_n_aqe"] = raw_naqe

            reasoning = cfg_data.get("reasoning", "(no reasoning field)")
            print(f"\n[LLM] engine={engine} params={clean_params} "
                  f"k={k_neighbors} alpha={alpha:.2f} "
                  f"n_refs={n_refs}/{ref_strategy} n_aqe={n_aqe} aqe_w={aqe_weight:.2f}")
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
    global _MAP_THRESHOLD
    _MAP_THRESHOLD = args.map_threshold

    provider = ConfigProvider(seed=args.seed)
    results: List[IterationResult] = []

    api_key = model = None
    if args.method == "hyperparameter_only":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model   = os.getenv("LLM_MODEL", "minimax/minimax-m2.1")
        llm     = LLMProvider(api_key, model=model, seed=args.seed)

    def _norm_refs(cfg):
        rs = cfg.get("ref_strategy", "first")
        nr = 1 if rs == "centroid" else cfg.get("n_refs", 1)
        return nr, rs

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
                 r.config.get("k_neighbors", 200),
                 f"{r.config.get('alpha', 0.0):.2f}",
                 *_norm_refs(r.config),
                 r.config.get("n_aqe", 1),
                 f"{r.config.get('aqe_weight', 0.0):.1f}")
                for r in results
            }

            consecutive_alpha_only = 0
            if len(results) >= 2:
                ref_cfg    = results[-1].config
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
            for _att in range(4):
                config, reasoning = llm.query_llm(
                    i + 1, results, args.iterations,
                    consecutive_alpha_only=consecutive_alpha_only,
                    iters_since_improvement=iters_since_improvement,
                    dedup_hint=dedup_hint)
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

                nr_raw  = config.get("n_refs", 1)
                rs      = config.get("ref_strategy", "first")
                nr_norm, rs_norm = _norm_refs(config)
                if rs == "centroid" and nr_raw != nr_norm:
                    dedup_hint = (
                        f"You suggested n_refs={nr_raw}, ref_strategy=centroid, "
                        f"k={config.get('k_neighbors')}, alpha={config.get('alpha'):.2f}.\n"
                        f"  centroid ALWAYS normalizes to n_refs=1. The config "
                        f"(n_refs=1, centroid, k={config.get('k_neighbors')}) was already tried.\n"
                        f"  Do NOT suggest centroid with n_refs>1. To vary centroid, change k instead."
                    )
                else:
                    recent_k    = [r.config.get("k_neighbors") for r in results[-6:]]
                    suggested_k = config.get("k_neighbors")
                    if recent_k.count(suggested_k) >= 3:
                        tried_k = sorted({r.config.get("k_neighbors") for r in results})
                        untried_k = [k for k in [50, 100, 150] if k not in tried_k]
                        dedup_hint = (
                            f"You suggested k={suggested_k} again — used {recent_k.count(suggested_k)}/"
                            f"{len(recent_k)} recent iters. STUCK on k={suggested_k}.\n"
                            f"  k values tried: {tried_k}\n"
                            f"  Untried from [50,100,150]: {untried_k or 'all tried — vary alpha or efC'}\n"
                            f"  Try k=50 with best (M, efC, alpha, ref_strategy)."
                        )
                    else:
                        dedup_hint = (
                            f"You suggested engine={config.get('engine')}, "
                            f"params={config.get('params')}, k={config.get('k_neighbors')}, "
                            f"alpha={config.get('alpha'):.2f}, n_refs={nr_norm}, "
                            f"ref_strategy={rs_norm} — already evaluated. "
                            f"Check ALL CONFIGS TRIED and suggest something new."
                        )
                print(f"[Dedup] LLM repeated config (attempt {_att+1}), retrying…")
            else:
                config = ConfigProvider(seed=args.seed + i).get_random_config()
                config["_iter"] = i + 1
                reasoning = "Fallback random (LLM repeated configs 4×)"
                print("[Dedup] Falling back to random config after 4 duplicate attempts.")

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
                    {"name": "k_neighbors",    "type": "enum",
                     "values": ConfigProvider.K_NEIGHBORS_VALUES},
                    {"name": "alpha",          "type": "enum",
                     "values": ConfigProvider.ALPHA_VALUES},
                    {"name": "n_refs",         "type": "enum",
                     "values": ConfigProvider.N_REFS_VALUES},
                    {"name": "ref_strategy",   "type": "enum",
                     "values": ConfigProvider.REF_STRATEGY_VALUES},
                    {"name": "n_aqe",          "type": "enum",
                     "values": ConfigProvider.N_AQE_VALUES},
                    {"name": "aqe_weight",     "type": "enum",
                     "values": ConfigProvider.AQE_WEIGHT_VALUES},
                ])
                run_optimization._vdtuner = VDTunerOptimizer(_enc, seed=args.seed)
            vt = run_optimization._vdtuner
            vt_raw = vt.ask()
            n_refs     = 1 if vt_raw["ref_strategy"] == "centroid" else vt_raw["n_refs"]
            aqe_weight = 0.0 if vt_raw["n_aqe"] == 1 else vt_raw["aqe_weight"]
            config = {
                "engine": "FaissHNSWFlat",
                "params": {"M": vt_raw["M"], "efConstruction": vt_raw["efConstruction"],
                           "efSearch": 200},
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
            config    = provider.get_random_config()
            config["_iter"] = i + 1
            reasoning = "Random"

        try:
            bm = run_benchmark(config, args.dataset_dir, iteration=i + 1)
            results.append(IterationResult(i + 1, config, bm, reasoning, args.method))

            if args.method in ("optuna", "gp_bo") and hasattr(run_optimization, "_optuna_study"):
                run_optimization._optuna_study.tell(trial, bm.score())
            if args.method == "vdtuner":
                run_optimization._vdtuner.tell(config, bm.map_score, bm.qps)

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
    parser.add_argument("--dataset-dir",    required=True)
    parser.add_argument("--method",         default="hyperparameter_only",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo", "vdtuner"])
    parser.add_argument("--iterations",     type=int, default=50)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--map-threshold",  type=float, default=0.15)
    parser.add_argument("--output",         default="milvus_gldv2_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Milvus GLDv2 Optimizer  method={args.method}  seed={args.seed}  τ={args.map_threshold}")
    print("=" * 60)

    results = run_optimization(args)

    best     = max(results, key=lambda r: r.benchmark.score(), default=None)
    feasible = [r for r in results if r.benchmark.map_score >= args.map_threshold]

    summary = {
        "method": args.method, "seed": args.seed, "iterations": args.iterations,
        "map_threshold": args.map_threshold,
        "best_score":  best.benchmark.score()      if best else 0.0,
        "best_map":    best.benchmark.map_score    if best else 0.0,
        "best_qps":    best.benchmark.qps          if best else 0.0,
        "best_config": best.config                 if best else {},
        "n_feasible":  len(feasible),
        "all_results": [
            {"iteration": r.iteration, "config": r.config, "search_method": r.search_method,
             "score": r.benchmark.score(), "map": r.benchmark.map_score,
             "qps": r.benchmark.qps, "latency_ms": r.benchmark.latency_ms}
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.output}")
    if best:
        print(f"Best: mAP={best.benchmark.map_score:.4f}  QPS={best.benchmark.qps:.1f}  "
              f"Score={best.benchmark.score():.1f}")


if __name__ == "__main__":
    main()
