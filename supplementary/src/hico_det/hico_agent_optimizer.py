"""
LLM-Guided VDMS Hyperparameter Optimizer for HICO-DET HOI Retrieval.

Ports optimizer_research_grade.py (SIFT1M) to HICO-DET with a two-stage pipeline:
  Stage 1: MobileCLIP text query → CLIP_Set (VDMS ANN, agent-optimized config)
  Stage 2: DINOv2 weighted-sum rerank (agent-tuned alpha)

Score function: QPS if mAP ≥ τ else 0  (constrained optimisation, τ=--map-threshold)

Agent tunes jointly:
  engine          : FaissHNSWFlat | FaissIVFFlat | FaissFlat
  M, efC, efS     : HNSW params
  nlist, nprobe   : IVF params
  k_neighbors     : Stage-1 candidate pool size; allowed: [50, 100, 150, 200, 300, 500, 750, 1000]
                    (k≥50; small k trades recall for QPS — feasible only if mAP stays ≥ τ)
  alpha           : DINOv2 rerank weight; allowed: 0.05-step grid from 0.0 to 0.90
                    [0.0, 0.05, 0.10, ..., 0.85, 0.90] — 19 values total
                    Optimal alpha is config-dependent — the agent must discover it empirically.

Known baseline (from ablation):
  FaissHNSWFlat M=32 efC=200 efS=500 K=500 alpha=0.30 → mAP=0.2352, QPS≈9.1

VDMS note: DeleteDescriptorSet is a no-op in the container. Each iteration uses a
unique set name (CLIP_Set_{iteration}) to avoid conflicts without restarting VDMS.

Usage:
    python hico_agent_optimizer.py \\
        --port 55575 \\
        --dataset-dir /work/hdd/bdjd/vdms/datasets \\
        --method hyperparameter_only \\
        --iterations 30 \\
        --output hico_agent_results.json
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
from scipy.special import log_softmax

# QDS Adaptive Constraint Strategy (set at startup when --use-qds-objective is passed)
# Constrained optimization threshold τ (SIEVE-style): Score = QPS if mAP ≥ τ else 0.
# Set at startup from --map-threshold CLI arg. Default 0.15 (above Grid baseline 0.121).
_MAP_THRESHOLD: float = 0.15

_QDS_WEIGHTS: Optional[np.ndarray] = None   # shape (n_queries,), weights in [1,2] — kept for tier assignment only
_QDS_STANDARD_THRESHOLD: bool = False       # when True: QDS exploration active but score() uses raw mAP ≥ τ
_QDS_TIERS:   Optional[np.ndarray] = None   # shape (n_queries,), integer tier 1-4 per query
_PER_QUERY_CONSTRAINT: Optional[np.ndarray] = None  # shape (n_queries,), "none" or "object" per query

# Per-class exact indices (built once at startup from clip_db + img_metadata).
# Q3+Q4 queries bypass VDMS and search these ~600-image per-class arrays via numpy dot product.
_PER_CLASS_INDICES: Optional[Dict] = None   # obj_id → (vecs (N,D), img_ids (N,))
_PER_CLASS_OBJ_VOCAB: Optional[Dict] = None  # obj_name → obj_id int

# Per-category feedback (set at startup when --use-category-feedback is passed)
_CATEGORY_FEEDBACK: bool = False
_QDS_FOR_FEEDBACK:  Optional[np.ndarray] = None  # raw QDS scores (not weights) per HOI class


def _compute_qds_weights(text_q: np.ndarray, clip_db: np.ndarray,
                         hoi_keys: List[str], tau: float = 0.01) -> np.ndarray:
    """
    Compute QDS = H_norm - CPR_norm per HOI class, return as weight array w_j = 1 + QDS_norm_j.
    Higher weight = harder class = matters more in the objective.
    """
    script_dir = Path(__file__).parent
    cpr_json   = script_dir / "cpr_results.json"

    # Entropy of CLIP similarity distribution
    print(f"[QDS] Computing per-class entropy (tau={tau})...")
    sim   = text_q @ clip_db.T
    log_p = log_softmax(sim / tau, axis=1)
    H     = -(np.exp(log_p) * log_p).sum(axis=1)   # (n_queries,)

    # CPR
    import json
    cpr_data = json.load(open(cpr_json))
    cpr_map  = {r["hoi"]: r["cpr"] for r in cpr_data}
    cpr_med  = float(np.median(list(cpr_map.values())))
    C        = np.array([cpr_map.get(k, cpr_med) for k in hoi_keys], dtype=np.float32)

    # Normalize both to [0, 1]
    H_n = (H - H.min()) / (H.max() - H.min() + 1e-10)
    C_n = (C - C.min()) / (C.max() - C.min() + 1e-10)

    qds = H_n - C_n                                 # (n_queries,)
    qds_n = (qds - qds.min()) / (qds.max() - qds.min() + 1e-10)
    weights = 1.0 + qds_n                           # w in [1, 2]
    weights = weights.astype(np.float32)

    q25, q50, q75 = np.percentile(qds, [25, 50, 75])
    print(f"[QDS] weights range [{weights.min():.3f}, {weights.max():.3f}]  "
          f"QDS quartiles: Q25={q25:.4f} Q50={q50:.4f} Q75={q75:.4f}")
    return weights


def _compute_qds_scores(text_q: np.ndarray, clip_db: np.ndarray,
                        hoi_keys: List[str], tau: float = 0.01) -> np.ndarray:
    """
    Compute raw QDS = H_norm - CPR_norm per HOI class (not weights — for quartile binning).
    Used by --use-category-feedback to provide per-difficulty-tier mAP breakdown to the LLM.
    """
    script_dir = Path(__file__).parent
    cpr_json   = script_dir / "cpr_results.json"

    sim   = text_q @ clip_db.T
    log_p = log_softmax(sim / tau, axis=1)
    H     = -(np.exp(log_p) * log_p).sum(axis=1)

    import json as _json
    cpr_data = _json.load(open(cpr_json))
    cpr_map  = {r["hoi"]: r["cpr"] for r in cpr_data}
    cpr_med  = float(np.median(list(cpr_map.values())))
    C        = np.array([cpr_map.get(k, cpr_med) for k in hoi_keys], dtype=np.float32)

    H_n = (H - H.min()) / (H.max() - H.min() + 1e-10)
    C_n = (C - C.min()) / (C.max() - C.min() + 1e-10)
    qds = (H_n - C_n).astype(np.float32)
    print(f"[CategoryFeedback] QDS range [{qds.min():.4f}, {qds.max():.4f}]  "
          f"quartiles: Q25={np.percentile(qds,25):.4f} Q75={np.percentile(qds,75):.4f}")
    return qds


def _format_category_feedback(per_query_ap: List[float]) -> str:
    """
    Format per-QDS-quartile mAP breakdown for LLM prompt (--use-category-feedback mode).
    Returns an empty string if feedback is disabled or data is unavailable.
    """
    if not _CATEGORY_FEEDBACK or _QDS_FOR_FEEDBACK is None or len(per_query_ap) == 0:
        return ""
    ap  = np.array(per_query_ap, dtype=np.float32)
    qds = _QDS_FOR_FEEDBACK
    if len(ap) != len(qds):
        return ""

    q25, q50, q75 = float(np.percentile(qds, 25)), float(np.percentile(qds, 50)), float(np.percentile(qds, 75))
    masks = [
        qds <= q25,
        (qds > q25) & (qds <= q50),
        (qds > q50) & (qds <= q75),
        qds > q75,
    ]
    tier_labels = [
        "Q1 easiest  (low QDS: text aligns well with visual cluster, low CPR)",
        "Q2 moderate (modest text-image gap)",
        "Q3 hard     (text-image gap + growing gallery ambiguity)",
        "Q4 hardest  (high H: many distractors; low CPR: text far from visual mean)",
    ]
    tier_aps = [float(ap[m].mean()) if m.sum() > 0 else 0.0 for m in masks]
    tier_ns  = [int(m.sum()) for m in masks]

    lines = [
        "CATEGORY DIFFICULTY BREAKDOWN (QDS quartiles — last run):",
        "  QDS_j = H_j − CPR_j  (gallery entropy − text-visual alignment gap)",
        "  High QDS = hard: many distractors AND text embedding far from visual cluster.",
    ]
    for label, n, ap_val in zip(tier_labels, tier_ns, tier_aps):
        lines.append(f"    {label}:  N={n}  mean AP = {ap_val:.4f}")

    lines += [
        "",
        f"  Overall mAP (unweighted) = {float(ap.mean()):.4f}",
        "",
        "  WHAT EACH TIER NEEDS:",
        "    Q4 (hardest): DINOv2 reranking is CRITICAL — n_refs=10 + centroid/diverse captures",
        "      the visual center of ambiguous categories; alpha=0.65-0.85 lets DINOv2 override",
        "      the CLIP text query. constraint=object reduces 47K→597 images → harder categories",
        "      benefit most. Use k≥150 to maintain recall within the filtered/reranked pool.",
        "    Q3: similar to Q4 but alpha=0.50-0.70 is often sufficient.",
        "    Q1-Q2 (easiest): text query alone is accurate. alpha=0.3-0.5 sufficient; over-weighting",
        "      DINOv2 risks overshooting. Small k (50-100) is fine for speed.",
        "    STRATEGY: configs that help Q4 (large k, high alpha, diverse refs) typically cost QPS.",
        "      The joint objective Score = "
        + ("QPS subject to QDS-weighted_mAP ≥ τ, with adaptive object-constraint for Q3/Q4 queries" if _PER_QUERY_CONSTRAINT is not None else "QPS subject to mAP ≥ τ")
        + " requires balancing all tiers.",
    ]
    return "\n".join(lines)


optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_hico_det import (
    load_hico_data,
    build_hoi_queries,
    build_dinov2_refs,
    setup_clip_index,
    run_hico_text_dinov2_rerank,
    build_per_class_clip_indices,
)


# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

@dataclass
class BenchmarkResult:
    qps:             float
    map_score:       float   # mAP over all 600 HOI queries
    latency_ms:      float   # avg total wall-clock per query (Stage-1 + Stage-2), ms
    t_clip_avg_ms:   float   # avg Stage-1 VDMS FindDescriptor time per query, ms
    t_rerank_avg_ms: float   # avg Stage-2 DINOv2 numpy rerank time per query, ms
    index_build_s:   float   # CLIP_Set index build time, seconds (one-time cost)
    precision_at_10: float
    ndcg_at_10:      float
    recall_at_10:    float   # R@10 (supplementary)
    engine:          str
    params:          Dict
    k_neighbors:     int
    alpha:           float
    timestamp:       float
    per_query_ap:    List[float] = field(default_factory=list)

    def score(self) -> float:
        """Constrained optimization (SIEVE-style): Score = QPS if mAP ≥ τ else 0.
        Maximizing QPS within the feasible region avoids degenerate high-QPS/low-mAP
        solutions without combining accuracy and throughput into an arbitrary product.
        In QDS mode, the constraint is applied to difficulty-weighted mAP so hard
        queries (Q3/Q4) are protected from recall collapse."""
        if _QDS_WEIGHTS is not None and self.per_query_ap and not _QDS_STANDARD_THRESHOLD:
            w = _QDS_WEIGHTS
            n = min(len(w), len(self.per_query_ap))
            weighted_map = float(np.sum(w[:n] * np.array(self.per_query_ap[:n])) / np.sum(w[:n]))
            if weighted_map < _MAP_THRESHOLD:
                return 0.0
            return float(self.qps)
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
# DATA CACHE (avoid reloading 293 MB per iteration)
# ===========================================================================

_HICO_DATA_CACHE: Dict[str, Dict] = {}

# backbone -> (clip_db_file, text_queries_file, dim, label)
_BACKBONE_FILES = {
    "mobileclip_s2": ("hico_clip.npy",          "hico_clip_text_queries.npy",     512, "MobileCLIP-S2"),
    "vitl14":        ("hico_clipvitl14_db.npy",  "hico_clipvitl14_text_queries.npy", 768, "CLIP ViT-L/14"),
}


def _load_hico_data_cached(dataset_dir: str, backbone: str = "mobileclip_s2") -> Dict:
    cache_key = f"{dataset_dir}::{backbone}"
    if cache_key not in _HICO_DATA_CACHE:
        if backbone not in _BACKBONE_FILES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(_BACKBONE_FILES)}")
        clip_file, text_file, dim, label = _BACKBONE_FILES[backbone]

        d = dataset_dir.rstrip("/") + "/hico_det"

        data = load_hico_data(dataset_dir)

        clip_db = np.load(f"{d}/{clip_file}").astype(np.float32)

        text_q_all = np.load(f"{d}/{text_file}").astype(np.float32)
        with open(f"{d}/hico_clip_text_key_order.json") as f:
            text_key_order = json.load(f)
        text_q_map = {k: text_q_all[i] for i, k in enumerate(text_key_order)}

        _, deep_q, rel_sets, junk_sets, hoi_keys = build_hoi_queries(
            data["hoi_index"], data["sift_db"], data["deep_db"]
        )

        missing = [k for k in hoi_keys if k not in text_q_map]
        if missing:
            raise RuntimeError(f"{len(missing)} HOI keys missing from text queries: {missing[:3]}")
        text_q = np.array([text_q_map[k] for k in hoi_keys], dtype=np.float32)

        _HICO_DATA_CACHE[cache_key] = {
            "clip_db":    clip_db,
            "dinov2_db":  data["deep_db"],
            "hoi_index":  data["hoi_index"],   # needed to recompute refs for n_refs > 1
            "text_q":     text_q,
            "dinov2_q":   deep_q,              # pre-built for n_refs=1
            "rel_sets":   rel_sets,
            "junk_sets":  junk_sets,
            "hoi_keys":   hoi_keys,
            "backbone":   backbone,
            "clip_dim":   dim,
            "clip_label": label,
        }
        print(f"[Cache] HICO-DET loaded ({label}): clip_db={clip_db.shape}  "
              f"dinov2_db={data['deep_db'].shape}  text_q={text_q.shape}  "
              f"queries={len(hoi_keys)}")

    return _HICO_DATA_CACHE[cache_key]


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
            "efConstruction": [200],                        # fixed — no effect at query time
            "efSearch":       [32, 64, 100, 150, 200, 300, 500],
        },
        "FaissIVFFlat": {
            "nlist":  [64, 128, 256, 512],                  # 1024 removed: degenerate for 47K vecs
            "nprobe": [4, 8, 16, 32, 64],
        },
    }

    K_NEIGHBORS_VALUES = [50, 100, 150, 200, 300, 500, 750, 1000]
    ALPHA_VALUES       = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                          0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    N_REFS_VALUES        = [1, 3, 5, 10]
    REF_STRATEGY_VALUES  = ["first", "centroid", "diverse"]

    # ---------------------------------------------------------------
    # GRAPH CONSTRAINT STRATEGIES (PMGD metadata pre-filtering)
    #   "none"           — no PMGD constraint, pure ANN (baseline behavior)
    #   "object"         — pre-filter by primary_object_id (~1.2% of DB avg)
    #   "verb"           — pre-filter by primary_verb_id   (~0.9% of DB avg)
    #   "object_and_verb"— both constraints combined        (<0.3% of DB avg)
    # Tradeoff: constraint reduces effective ANN search space (↑ QPS) but
    # may exclude relevant images with secondary objects (↓ mAP). The agent
    # must discover the joint optimum of (k_neighbors, constraint_strategy).
    # ---------------------------------------------------------------
    CONSTRAINT_STRATEGIES = ["none", "object", "verb", "object_and_verb"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def get_random_config(self) -> Dict:
        engine = self.rng.choice(self.ENGINES)
        params = self._sample_params(engine)
        cfg = {
            "engine":               engine,
            "params":               params,
            "n_refs":               int(self.rng.choice(self.N_REFS_VALUES)),
            "ref_strategy":         str(self.rng.choice(self.REF_STRATEGY_VALUES)),
            "constraint_strategy":  str(self.rng.choice(self.CONSTRAINT_STRATEGIES)),
        }
        if _QDS_TIERS is not None:
            for t in range(1, 5):
                cfg[f"alpha_Q{t}"] = float(self.rng.choice(self.ALPHA_VALUES))
        else:
            cfg["alpha"] = float(self.rng.choice(self.ALPHA_VALUES))
        cfg["k_neighbors"] = int(self.rng.choice(self.K_NEIGHBORS_VALUES))
        return cfg

    def _sample_params(self, engine: str) -> Dict:
        params = {}
        if engine in self.ENGINE_PARAMS:
            for pname, pvals in self.ENGINE_PARAMS[engine].items():
                params[pname] = int(self.rng.choice(pvals))
        return params

    def grid_search_configs(self) -> List[Dict]:
        # Blind 1D-sweep grid (30 configs, no pre-loaded domain knowledge).
        #
        # Design rationale: a practitioner without domain knowledge sweeps one
        # parameter at a time, fixing all others at a default midpoint.
        # All stages use n_refs=1 and ref_strategy="first" — the naive defaults.
        # Grid never jointly optimises n_refs × ref_strategy × alpha, which is
        # exactly the interaction the LLM agent is designed to discover.
        #
        #   S1: Engine/M baselines (5)    k=200, α=0.30, n_refs=1
        #       FaissFlat, IVFFlat(nlist=128,nprobe=8),
        #       HNSW M=16/32/64 efS=200
        #   S2: k sweep (5)               HNSW M=32 efS=200, α=0.30, n_refs=1
        #       k ∈ {50, 100, 150, 500, 1000}
        #   S3: alpha sweep (5)           HNSW M=32 efS=200, k=100, n_refs=1
        #       α ∈ {0.10, 0.30, 0.50, 0.70, 0.90}
        #   S4: n_refs sweep (5)          HNSW M=32 efS=200, k=100, α=0.30
        #       n_refs ∈ {1, 3, 5, 10}, ref_strategy="first"  (4 configs)
        #       + n_refs=3, ref_strategy="centroid"            (1 config)
        #   S5: ref_strategy sweep (3)    HNSW M=32 efS=200, k=100, α=0.30, n_refs=3
        #       ref_strategy ∈ {first, centroid, diverse}
        #   S6: efSearch sweep (4)        HNSW M=32 k=100, α=0.30, n_refs=1
        #       efSearch ∈ {32, 64, 100, 150}
        #   S7: IVFFlat nprobe sweep (3)  nlist=128, k=100, α=0.30, n_refs=1
        #       nprobe ∈ {4, 16, 64}
        #
        # Key blind spots vs LLM agent (by design):
        #   - M=48 never tested
        #   - Joint (n_refs=3, ref_strategy=centroid, α=0.55) never tested
        #   - efSearch and n_refs never co-optimised
        #
        # Total: 5+5+5+5+3+4+3 = 30
        configs = []

        # Grid = pure ANN baseline (constraint_strategy="none" throughout).
        # This is intentional: grid never tests PMGD constraints, making it a
        # clean no-graph baseline. LLM agent and random search explore constraints.
        def _gc(**kw):
            kw.setdefault("constraint_strategy", "none")
            return kw

        # ---- S1: Engine/M baselines (5) k=200, alpha=0.30, n_refs=1 ----
        configs.append(_gc(engine="FaissFlat", params={},
                           k_neighbors=200, alpha=0.30, n_refs=1, ref_strategy="first"))
        configs.append(_gc(engine="FaissIVFFlat",
                           params={"nlist": 128, "nprobe": 8},
                           k_neighbors=200, alpha=0.30, n_refs=1, ref_strategy="first"))
        for M in [16, 32, 64]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": M, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=200, alpha=0.30, n_refs=1, ref_strategy="first"))

        # ---- S2: k sweep (5) HNSW M=32 efS=200, alpha=0.30, n_refs=1 ----
        for k in [50, 100, 150, 500, 1000]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=k, alpha=0.30, n_refs=1, ref_strategy="first"))

        # ---- S3: alpha sweep (5) HNSW M=32 efS=200, k=100, n_refs=1 ----
        for alpha in [0.10, 0.30, 0.50, 0.70, 0.90]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=alpha, n_refs=1, ref_strategy="first"))

        # ---- S4: n_refs sweep (5) HNSW M=32 efS=200, k=100, alpha=0.30 ----
        for n in [1, 3, 5, 10]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=0.30, n_refs=n, ref_strategy="first"))
        configs.append(_gc(engine="FaissHNSWFlat",
                           params={"M": 32, "efConstruction": 200, "efSearch": 200},
                           k_neighbors=100, alpha=0.30, n_refs=3, ref_strategy="centroid"))

        # ---- S5: ref_strategy sweep (3) HNSW M=32 efS=200, k=100, alpha=0.30, n_refs=3 ----
        for strategy in ["first", "centroid", "diverse"]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=0.30, n_refs=3, ref_strategy=strategy))

        # ---- S6: efSearch sweep (4) HNSW M=32 k=100, alpha=0.30, n_refs=1 ----
        for efs in [32, 64, 100, 150]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": efs},
                               k_neighbors=100, alpha=0.30, n_refs=1, ref_strategy="first"))

        # ---- S7: IVFFlat nprobe sweep (3) nlist=128, k=100, alpha=0.30, n_refs=1 ----
        for nprobe in [4, 16, 64]:
            configs.append(_gc(engine="FaissIVFFlat",
                               params={"nlist": 128, "nprobe": nprobe},
                               k_neighbors=100, alpha=0.30, n_refs=1, ref_strategy="first"))

        # Total: 5+5+5+5+3+4+3 = 30
        assert len(configs) == 30, f"Grid must have exactly 30 configs, got {len(configs)}"
        return configs


# ===========================================================================
# LLM PROVIDER
# ===========================================================================

class LLMProvider:
    def __init__(self, api_key: str, model: str = "minimax/minimax-m2.1",
                 backbone: str = "mobileclip_s2", seed: int = 42,
                 ablation_no_history: bool = False,
                 ablation_no_phases: bool = False,
                 t_exp_frac: float = 0.40,
                 t_expl_frac: float = 0.75):
        self.api_key             = api_key
        self.model               = model
        self.backbone            = backbone
        self.seed                = seed
        self.ablation_no_history = ablation_no_history  # Ablation A: hide all past results from LLM
        self.ablation_no_phases  = ablation_no_phases   # Ablation B: lock agent in EXPLORATION forever
        self.t_exp_frac          = t_exp_frac
        self.t_expl_frac         = t_expl_frac

    async def query_llm(self, iteration: int, history: List[IterationResult],
                        total_iterations: int,
                        consecutive_alpha_only: int = 0,
                        iters_since_improvement: int = 0) -> Tuple[Dict, str]:

        # Phase boundaries computed dynamically from total_iterations.
        # exp_end / expl_end are used both for the phase label AND the scenario guide
        # text, so the two are always consistent (no float vs int rounding mismatch).
        exp_end  = max(1, int(round(total_iterations * self.t_exp_frac)))
        expl_end = max(exp_end + 1, int(round(total_iterations * self.t_expl_frac)))

        # Ablation B: disable phase transitions — agent stays in EXPLORATION forever.
        # This removes the exploitation signal that tells the agent to commit to a
        # promising zone and refine alpha, without changing what history it sees.
        if self.ablation_no_phases:
            phase = "EXPLORATION"
        else:
            phase = ("EXPLORATION" if iteration <= exp_end else
                     "EXPLOITATION" if iteration <= expl_end else "FINE-TUNING")

        current_best_score = max(r.benchmark.score() for r in history) if history else 0.0

        if history:
            last_run    = history[-1]
            last_score  = last_run.benchmark.score()
            last_map    = last_run.benchmark.map_score
            last_qps    = last_run.benchmark.qps
            last_engine = last_run.config["engine"]
            last_params = last_run.config.get("params", {})
            last_k      = last_run.config.get("k_neighbors", 500)
            if _QDS_TIERS is not None and "alpha_Q1" in last_run.config:
                last_alpha_str = "/".join(
                    f"Q{t}={last_run.config.get(f'alpha_Q{t}', 0.3):.2f}" for t in range(1, 5))
            else:
                last_alpha_str = f"{last_run.config.get('alpha', 0.0):.2f}"

            if current_best_score == 0.0 and last_score == 0.0:
                guidance = (
                    f"INFEASIBLE. mAP={last_map:.4f} < τ={_MAP_THRESHOLD:.3f} → Score=0. "
                    f"No feasible config found yet. "
                    f"To gain mAP: increase k_neighbors (k=100→mAP~0.20, k=150→mAP~0.24). "
                    f"Use constraint_strategy=object + n_refs=10 + alpha=0.65 to push mAP above τ. "
                    f"Do NOT optimize QPS until mAP ≥ τ — all scores are 0 until then.")
            elif last_score >= current_best_score * 0.98:
                if phase == "FINE-TUNING":
                    guidance = (
                        f"EXCELLENT. Near peak. {last_engine} k={last_k} alpha={last_alpha_str}. "
                        f"Score={last_score:.4f}. Fine-tuning: change ONE parameter by ONE step: "
                        f"alpha_Q4 +/-0.05 (or any tier alpha), OR k_neighbors one step, OR efSearch one step.")
                else:
                    guidance = (
                        f"Near best so far ({last_engine} k={last_k} alpha={last_alpha_str} "
                        f"Score={last_score:.4f}). Phase={phase}: do NOT fine-tune yet. "
                        f"KEEP EXPLORING — try a structurally DIFFERENT region: "
                        f"different engine, OR k={'50 or 200' if last_k == 100 else '100 or 150'}, "
                        f"OR IVFFlat if not yet tried, OR M=48/64 vs M=32. "
                        f"The global optimum may be in an unexplored corner of the search space.")
            elif last_map < 0.20 and last_score < current_best_score * 0.80:
                guidance = (
                    f"Score={last_score:.4f} low. mAP={last_map:.4f} QPS={last_qps:.1f}. "
                    f"Remember: OBJECTIVE: maximize QPS subject to mAP ≥ τ (configs below τ score 0 — infeasible) "
                    f"if QPS is high enough to compensate. "
                    f"If QPS is already high (>200), try adjusting alpha. "
                    f"If QPS is also low, reduce k_neighbors (primary QPS lever). Do NOT reduce efSearch — it has negligible QPS effect in batch mode.")
            elif last_map >= 0.20 and last_qps < 100:
                guidance = (
                    f"mAP acceptable ({last_map:.4f}) but QPS critically low ({last_qps:.1f}). "
                    f"PRIMARY FIX: reduce k_neighbors. k=200→QPS~58, k=100→QPS~145, k=50→QPS~295. "
                    f"Do NOT reduce efSearch — empirically it has no measurable QPS effect in batch mode. "
                    f"If you go to k=50, use M=64 + alpha≥0.65 + constraint_strategy=object + n_refs≥5 to keep mAP above τ.")
            elif last_score < current_best_score * 0.90:
                guidance = (
                    f"Score={last_score:.4f} significantly below best={current_best_score:.4f}. "
                    f"Last config: {last_engine} k={last_k} alpha={last_alpha_str} params={last_params}. "
                    f"Return closer to best config and make targeted adjustments.")
            else:
                guidance = (
                    f"Score={last_score:.4f} moderate. mAP={last_map:.4f} QPS={last_qps:.1f}. "
                    f"To gain QPS: reduce k_neighbors one step (halves QPS at each step). "
                    f"To gain mAP: increase k_neighbors or alpha. Do NOT change efSearch for QPS.")
        else:
            if self.backbone == "vitl14":
                if _PER_QUERY_CONSTRAINT is not None:
                    guidance = (
                        "First iteration — QDS-ACS with per-class exact numpy search. "
                        "Hard queries (Q3+Q4, ~300 of 600) bypass VDMS entirely and search a "
                        "pre-built per-class exact index (~600 images per object class, ~0.05ms/query). "
                        "Q1+Q2 (~300 queries) use your configured VDMS engine (unconstrained full-47K search). "
                        "constraint_strategy is ALWAYS forced to 'none' — do not set it. "
                        "RECOMMENDED START: FaissHNSWFlat M=32 efSearch=64-100, k=100-150, "
                        "alpha_Q1=0.75, alpha_Q2=0.75, alpha_Q3=0.65, alpha_Q4=0.75, "
                        "n_refs=5-10, ref_strategy=centroid. "
                        "Expected: Q1/Q2 VDMS QPS~1600, combined system QPS~820, Score~200-260. "
                        "FaissFlat k=100-150 also excellent for Q1/Q2 (exact search, high QPS).")
                else:
                    guidance = (
                        "First iteration — cold start, no prior data. "
                        "OBJECTIVE: maximize QPS subject to mAP ≥ τ (batch: 600 queries/call). "
                        "CRITICAL PHYSICS: k_neighbors is the PRIMARY QPS lever. efSearch has negligible QPS effect. "
                        "Measured QPS: k=50→295, k=100→145, k=200→58. efSearch=32 vs 150 at k=100: no difference. "
                        "SAFE START: FaissHNSWFlat M=64 efSearch=64 k=100 alpha=0.65 constraint_strategy=object n_refs=10. "
                        "This gives QPS~145, mAP~0.20 (safely above τ). Then try k=50 M=64 alpha=0.65 constraint=object n_refs=10 "
                        "for QPS~295 — but mAP is marginal (0.155-0.163) so this is high-risk/high-reward. "
                        "Avoid k>200: QPS drops below 60. Avoid efSearch changes as primary QPS strategy. "
                        "FaissIVFFlat (nlist=128, nprobe=8, k=100) is also valid: QPS~155, mAP~0.18.")
            else:
                guidance = (
                    "First iteration. Start with a known-good baseline to calibrate: "
                    "FaissHNSWFlat M=32 efC=200 efS=500 k_neighbors=500. "
                    "Choose alpha freely — alpha=0.30 was best at K=500 efS=500 in prior ablation, "
                    "but the optimal alpha at other K/efSearch settings is unknown.")

        _, _, clip_dim, clip_label = _BACKBONE_FILES[self.backbone]

        if self.backbone == "vitl14":
            known_results = """\
       alpha=0.00, K=2000, HNSW M=32 efS=500: mAP=0.2398  (text-only, no DINOv2 reranking)
       alpha=0.25, K=2000, HNSW M=32 efS=500: mAP=0.2700  (DINOv2 helps over text-only)
       alpha=0.30, K=2000, HNSW M=32 efS=500: mAP=0.2740
       alpha=0.50, K=2000, HNSW M=32 efS=500: mAP=0.2796  (highest mAP seen at this slow config)
   IMPORTANT: ALL above configs used K=2000 (very high recall, ANN-dominated). Batch QPS≈3-4 here.
   The optimal alpha at smaller K (50-200) is UNKNOWN — you must discover it.
   MEASURED: K=100 efS=64 → QPS~145. K=50 efS=64 M=64 alpha≥0.65 → QPS~295 (but mAP borderline).
   efSearch does NOT affect QPS in batch mode. The target region is k=50-100."""
            dist_note = "~1.55-1.75 for unit-normalized 768-d"
        else:
            known_results = """\
       alpha=0.00, K=500, HNSW M=32 efS=500: mAP=0.2214  (text-only baseline)
       alpha=0.10, K=500, HNSW M=32 efS=500: mAP=0.2262
       alpha=0.20, K=500, HNSW M=32 efS=500: mAP=0.2313
       alpha=0.30, K=500, HNSW M=32 efS=500: mAP=0.2352  (best mAP at this slow config)
       alpha=0.50, K=500, HNSW M=32 efS=500: mAP=0.2280  (DINOv2 overshoots at K=500)
       alpha=0.80, K=500, HNSW M=32 efS=500: mAP=0.1992  (DINOv2 dominates badly)
   IMPORTANT: These configs all used K=500 efS=500 (ANN-dominated). Batch QPS≈15-20 here.
   Alpha=0.30 was best at K=500 efS=500, but optimal alpha at smaller K is UNKNOWN.
   MEASURED: K=100 → QPS~145. K=50 M=64 alpha≥0.65 → QPS~295 (mAP borderline, needs constraint=object).
   efSearch does NOT affect QPS. Reduce k to gain QPS."""
            dist_note = "~1.30-1.45 for unit-normalized 512-d"

        if _PER_QUERY_CONSTRAINT is not None:
            n_constrained = int((_PER_QUERY_CONSTRAINT == "object").sum())
            n_total = len(_PER_QUERY_CONSTRAINT)
            _obj_text = (
                "3. OPTIMIZATION OBJECTIVE: OBJECTIVE: maximize QPS subject to QDS-weighted_mAP ≥ τ (infeasible if below τ, score=0)\n"
                "   weighted_mAP = Σ(w_j × AP_j) / Σ(w_j) where w_j ∈ [1,2] (Q4 hardest queries get 2×).\n"
                "   Improving hard-query (Q3/Q4) AP raises Score MORE than equal gains on easy queries.\n"
                "   QDS-ACS STRUCTURAL ENHANCEMENT (per-class exact numpy search):\n"
                f"   {n_constrained} of {n_total} queries (Q3+Q4 tiers, hardest 50%) BYPASS VDMS.\n"
                "   They search a pre-built per-class exact array (~600 images of the correct object class)\n"
                "   using numpy dot product — ~0.05ms/query, exact result, no approximation error.\n"
                "   This lifts Q3 AP ceiling from ~12% to ~20%, Q4 from ~2% to ~8% — real signal.\n\n"
                f"   {n_total - n_constrained} of {n_total} queries (Q1+Q2, easy 50%) use your VDMS engine config\n"
                "   with constraint_strategy FORCED to 'none' (unconstrained full-47K search).\n\n"
                "   WHAT YOU TUNE (constraint_strategy is NOT tunable — forced to 'none'):\n"
                "   - engine, M/efSearch/nlist/nprobe: affects Q1+Q2 VDMS search ONLY\n"
                "   - k_neighbors: applied to both groups (per-class pool ~600 images; k=100-150 is near-exhaustive)\n"
                "   - alpha_Q1/Q2/Q3/Q4: DINOv2 rerank weight per tier — tune all four independently\n"
                "   - n_refs, ref_strategy: DINOv2 reference quality\n\n"
                "   EMPIRICAL alpha slopes (measured at k=200 per-class):\n"
                "     Q1: +0.274/unit  Q2: +0.176/unit  Q3: -0.052/unit  Q4: +0.040/unit\n"
                "   → ALL tiers benefit from alpha ≥ 0.65. Q3 peaks at 0.65-0.70.\n"
                "   → Recommended start: Q1=0.75, Q2=0.75, Q3=0.65, Q4=0.75.\n\n"
                "   EXPECTED PERFORMANCE:\n"
                "   - Q1/Q2: VDMS FaissHNSWFlat M=32 efS=64, k=100-150 → QPS~1600 for those 300 queries\n"
                "   - Q3/Q4: per-class numpy → ~0.05ms/query (negligible)\n"
                "   - Combined system: socket 56ms + 300×2ms + 300×0.05ms = ~660ms for 600 → QPS~820-900\n"
                "   - mAP improvement: Q3/Q4 exact search → aggregate mAP ~0.26-0.28\n"
                "   - Target Score: 0.27 × 850 ≈ 230 (vs standard LLM baseline ~92)\n"
                f"   - Known results from ablation experiments:\n   {known_results}\n"
                f"   Current best Score in this session: {current_best_score:.4f}"
            )
        else:
            _obj_text = (
                "3. OPTIMIZATION OBJECTIVE: maximize QPS subject to mAP ≥ τ (infeasible if below τ, score=0)\n"
                f"   Score = QPS  if plain mAP ≥ τ={_MAP_THRESHOLD:.3f}  else 0.\n"
                "   - QPS: Queries Per Second — Stage-1 VDMS cost dominates (Stage-2 is ~0.9ms, negligible)\n"
                "   - mAP: mean Average Precision over all 600 HOI queries (unweighted, plain average)\n"
                "   - A config is INFEASIBLE (score=0) if mAP < τ, regardless of how high QPS is.\n"
                "   - Within the feasible region (mAP ≥ τ), maximize QPS — higher QPS = better score.\n"
                f"   - Known results from ablation experiments:\n   {known_results}\n"
                f"   Current best score in this session: {current_best_score:.4f}"
            )

        system_context = f"""
SYSTEM ARCHITECTURE:
1. TARGET SYSTEM: VDMS (Visual Data Management System)
   - C++ middleware for persistent multimodal vector storage.
   - Indexing Backends: FaissHNSWFlat, FaissIVFFlat, FaissFlat.
   - Communication: Client-Server via TCP sockets.
   - Constraint: BATCH_SIZE=50 for AddDescriptor (PMGD journal limit).

2. RETRIEVAL PIPELINE (TWO STAGES — you tune BOTH stages simultaneously):

   Stage 1 — VDMS ANN Search  [{clip_label} — Text-to-Image Semantic Retrieval]:
     WHAT {clip_label} IS: A vision-language model trained with contrastive learning on
       image-text pairs. It maps both text and images into a shared {clip_dim}-d semantic space
       where conceptually related items are nearby.
     WHAT IT IS GOOD AT: Finding images that are CONCEPTUALLY related to the HOI text label.
       e.g., "a person riding a motorcycle" → returns images showing that interaction.
       It operates at the category/concept level — it understands meaning from language.
     WHAT IT IS NOT: A fine-grained visual detector. It retrieves the right semantic neighborhood
       but the ranking within that neighborhood may have noise (visually atypical examples).
     Query: {clip_dim}-d text embedding of HOI label ("a person riding a motorcycle")
     Index: CLIP_Set (you choose engine + engine params)
     Output: top-K candidate images (you choose k_neighbors)
     Cost: ANN search time per query (batch amortizes socket overhead across all 600 queries)

   Stage 2 — DINOv2 Rerank  [DINOv2 ViT-L/14-reg4 — Visual Appearance Refinement]:
     WHAT DINOv2 IS: A self-supervised vision-only model with NO language understanding.
       It encodes visual appearance (textures, shapes, spatial layout) into a 1024-d feature.
     WHAT IT IS GOOD AT: Distinguishing visually similar images at a fine-grained level.
       It re-sorts the K candidates by visual similarity to a reference image for the HOI category.
     WHAT IT IS NOT: A text or semantic model. Its query vector is a VISUAL feature derived from
       a single representative image per HOI category — not from the text label.
     WHY FUSION HELPS: CLIP finds semantically correct images but may rank visually atypical ones
       too highly. DINOv2 corrects that ordering using fine-grained appearance similarity.
     WHY HIGH ALPHA CAN HURT: DINOv2 is anchored to ONE reference image per HOI class.
       If that reference is not representative, or the HOI class has high visual diversity,
       too much DINOv2 weight (high alpha) overrides CLIP's reliable semantic signal with noise.
     Fuse: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm  (WEIGHTED SUM, lower=better)
       where clip_norm   = CLIP L2 distance normalized to [0,1] within K candidates
             dinov2_norm = DINOv2 L2 distance normalized to [0,1] within K candidates
     Sort by final_score ascending. Cost: ~1.4ms/query (negligible).
     alpha=0 → pure CLIP semantic ranking (trust text understanding only).
     alpha=1 → pure DINOv2 visual ranking (not in allowed list — DINOv2 alone is unreliable).

{_obj_text}
"""

        engine_definitions = f"""
PARAMETER DEFINITIONS AND PHYSICS:

============================================================
GROUP A — STAGE-1 ENGINE PARAMETERS (affect VDMS search speed and CLIP recall)
============================================================

[A1. engine — Index Type]
  Choices: FaissHNSWFlat | FaissIVFFlat | FaissFlat
  - FaissHNSWFlat: graph-based. RECOMMENDED. QPS is controlled primarily by k_neighbors
                   (not efSearch in batch mode). Tune M and efSearch for mAP ceiling.
  - FaissFlat:     exact brute-force. Highest mAP at any K. No tunable params.
                   QPS scales with k exactly as HNSW does (PMGD-dominated, same bottleneck).
                   batch QPS~140 at k=100 (no faster than HNSW because efSearch is not the bottleneck).
  - FaissIVFFlat:  cluster-based. QPS is controlled by nprobe, not k primarily.
                   Valid competitor — nlist=128 nprobe=8 k=100 gives QPS~140-190 with mAP~0.17-0.19.

[A2. M — HNSW graph connectivity] (FaissHNSWFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 8, 16, 32, 48, 64
  - Higher M = denser graph = higher CLIP recall = higher mAP ceiling.
  - Higher M = more memory + longer index build (build is one-time, ~7 min anyway).
  - RELATIONSHIP WITH efSearch: Higher M allows LOWER efSearch for same recall.
    e.g., M=48 efS=150 ≈ M=32 efS=200 in recall. Use this to trade memory for speed.
  - Recommended: M=32 (balanced). M=48 if mAP ceiling is priority.

[A3. efConstruction — HNSW build quality] (FaissHNSWFlat only)
  FIXED VALUE: 200. You MUST use 200. Do not use any other value.
  - Does not affect query speed or mAP at query time. Fixed to remove a meaningless
    degree of freedom and ensure all methods explore the same effective space.

[A4. efSearch — HNSW search depth] (FaissHNSWFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 32, 64, 100, 150, 200, 300, 500
  - Controls CLIP recall quality (fraction of true nearest neighbors found).
  - IMPORTANT: In VDMS batch mode (600 queries per call), efSearch has NEGLIGIBLE effect
    on QPS. Empirical measurement at k=100: efS=32→QPS=138, efS=64→QPS=139, efS=150→QPS=157.
    This is because VDMS PMGD journal commit cost scales with k×query_count, NOT with graph
    traversal depth. efSearch is a RECALL lever here, not a QPS lever.
  - USE efSearch to control mAP: lower efSearch → lower CLIP recall → lower mAP ceiling.
    If mAP is too low (below τ), increasing efSearch or k_neighbors is the fix.
  - RELATIONSHIP WITH alpha: lower efSearch → noisier CLIP ranking within top-K
    → DINOv2 rerank (higher alpha) can partially compensate.
    Rule: if you reduce efSearch, consider increasing alpha slightly (+0.05).
  - Empirical at k=100: efSearch=32 → mAP~0.19, efSearch=64 → mAP~0.20, efSearch=200 → mAP~0.24.

[A5. nlist — IVF cluster count] (FaissIVFFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 64, 128, 256, 512
  - nlist=1024 is NOT allowed: degenerate for 47,776 vectors (< 47 vecs/cell).
  - Higher nlist = faster search (fewer items per cell) but lower recall at same nprobe.
  - RELATIONSHIP WITH nprobe: always tune nprobe alongside nlist. Target nprobe ≈ sqrt(nlist).

[A6. nprobe — IVF cells visited] (FaissIVFFlat only)
  ALLOWED VALUES (you MUST choose exactly one): 4, 8, 16, 32, 64
  - QPS ∝ 1/nprobe (linear relationship).
  - Recall increases with nprobe. nprobe=nlist → exact search.
  - RELATIONSHIP WITH nlist: for nlist=128 use nprobe=8-16. For nlist=256 use nprobe=16-32.
  - RELATIONSHIP WITH alpha: same as efSearch — lower nprobe → noisier ranking → may need higher alpha.

============================================================
GROUP B — STAGE-2 RERANKING PARAMETERS (affect DINOv2 fusion)
============================================================

[B1. k_neighbors — Candidate pool size]  ← PRIMARY QPS LEVER IN BATCH MODE
  ALLOWED VALUES (you MUST choose exactly one): 50, 100, 150, 200, 300, 500, 750, 1000
  - Stage-1 retrieves top-K from VDMS. Stage-2 reranks exactly these K images.
  - k IS THE DOMINANT QPS LEVER. VDMS batch cost scales with k × 600 queries (PMGD
    journal commit overhead). Empirical measurements across 150 runs:
      k=50  → QPS~295   (baseline)
      k=100 → QPS~145   (0.49× of k=50)
      k=150 → QPS~85    (0.29× of k=50)
      k=200 → QPS~58    (0.20× of k=50)
      k=300 → QPS~34    (0.11× of k=50)
    Halving k approximately doubles QPS. This dominates all other parameters.
  - LARGER K = higher mAP ceiling (more true positives can surface through reranking).
  - TRADEOFF: Smaller k → higher QPS but lower mAP. mAP at k=50 is ~0.15-0.17 (borderline
    feasible). mAP at k=100 is ~0.19-0.22 (comfortably above τ=0.15). This is the key tension.
  - RELATIONSHIP WITH alpha: At k=50 the CLIP pool is aggressively filtered; DINOv2 rerank
    at high alpha (0.65-0.75) can push mAP above τ=0.15. Always tune alpha when changing k.
    At small k with high M (64), alpha=0.65 reliably achieves mAP~0.16 (just above τ).
  - STRATEGY: Start at k=100 (safe feasibility, QPS~140). Then try k=50 with M=64 alpha=0.65
    to test if mAP ≥ τ holds (QPS~295 if it does). k=50 is high-risk/high-reward.
  - NOTE: efSearch has negligible effect on QPS in batch mode. Do NOT reduce efSearch to
    boost QPS — change k instead. Reduce efSearch only if mAP ceiling needs to be lowered
    (which you almost never want).

[B2. alpha — DINOv2 rerank weight in Weighted Sum]
  ALLOWED VALUES (you MUST choose exactly one from this 0.05-step grid):
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
  - Fusion formula: final_score = (1-alpha)*clip_norm + alpha*dinov2_norm  (lower=better)
    where clip_norm   = CLIP L2 distance normalized to [0,1] within the K candidates,
          dinov2_norm = DINOv2 L2 distance normalized to [0,1] within the K candidates.
  - WHAT ALPHA CONTROLS — the balance between two complementary signals:
      alpha=0.0 → rely ONLY on CLIP's text-semantic ranking (concept-level, language-derived).
      alpha=0.5 → equal weight to CLIP semantics and DINOv2 visual appearance.
      alpha=1.0 → rely ONLY on DINOv2 visual appearance (not in allowed list).
      The optimal balance depends on the specific (K, engine, efSearch) config you are running.
      There is NO universally optimal alpha — you must find it through measurement.
  - WHY IT IS NOT TRIVIALLY MAXIMIZED:
      Increasing alpha helps when DINOv2 corrects CLIP's ordering errors (visually atypical examples).
      Increasing alpha hurts when DINOv2's single-image reference is unrepresentative of the HOI class
      (visual diversity problem) — then DINOv2 adds noise that overrides CLIP's reliable text signal.
      The crossover point between "helpful" and "harmful" depends on K, engine, and the data distribution.
  - Why weighted sum (not RRF): CLIP distances within top-K carry fine-grained signal
    (tight range {dist_note}). Rank-based RRF discards this.
    Ablation confirmed: weighted sum mAP=0.2352 > weighted RRF mAP=0.2334.
  - Reference data (K=500, HNSW efS=500 — SLOW config, not representative of optimal regime):
      alpha=0.0  → mAP=0.2214 (pure CLIP baseline).
      alpha=0.30 → mAP=0.2352 (best at this specific slow config — NOT a universal optimum).
      alpha=0.50 → mAP=0.2280 (DINOv2 overshoots at K=500, hurts mAP).
    These numbers show DINOv2 helps UP TO a point at K=500. At different K the optimum may shift.
  - Values above 0.90 are NOT in the allowed list. alpha=1.0 (pure DINOv2) is excluded.
  - NOTE: prior ablation at K=500 showed alpha>0.50 hurts. At K=100 the optimum may be
    higher (trend 0.30→0.50 kept improving). The range 0.50–0.90 is included to test this.
    Use reasoning: if prior iters show alpha increasing helps, probe higher; if it hurts, stop.
  - RELATIONSHIP WITH K: with large K, CLIP lets in more irrelevant images → DINOv2 correction
    is more valuable → try slightly higher alpha. With small K (100-200), CLIP already filtered
    aggressively → the pool is already precise → DINOv2 adds less marginal value → try lower alpha.
    BUT: this is a hypothesis. Measure it — the actual crossover point is unknown for fast configs.
  - RELATIONSHIP WITH efSearch: lower efSearch → noisier CLIP ordering within top-K
    → DINOv2 may correct more errors → slightly higher alpha could help.
    Lower recall (missed true positives) however cannot be compensated by alpha — only K can help there.

[B3. n_refs — DINOv2 reference image count per HOI class]
  ALLOWED VALUES (you MUST choose exactly one): 1, 3, 5, 10
  - Number of representative images per HOI class used to build the DINOv2 reference vector.
  - n_refs=1 (default): uses a single image as reference. Risk: that image may be
    unrepresentative if the HOI class has high visual diversity (e.g. "person riding
    motorcycle" includes racing, dirt, city — one image covers only one corner).
  - n_refs=3/5/10: averages DINOv2 vectors of n_refs images per class.
    The averaged vector sits closer to the visual CENTROID of the class.
    Benefit: DINOv2 correction becomes more reliable → higher alpha is safer.
  - COST: zero. n_refs computation is pure numpy (<1ms per query). Does NOT affect QPS.
    It is a FREE improvement — only changes mAP through better reranking quality.
  - RELATIONSHIP WITH alpha: better reference (higher n_refs) → DINOv2 is less noisy
    → the optimal alpha is likely HIGHER than with n_refs=1.
    If current best uses alpha=0.45, try n_refs=3 or 5 with alpha=0.55-0.65.
  - RELATIONSHIP WITH K: with better DINOv2 reference, reranking quality improves
    → smaller K can achieve similar mAP → higher QPS → better Score overall.
  - WARNING: n_refs=10 risks over-smoothing if HOI class images are visually diverse
    (average may drift toward a featureless centroid). n_refs=3 or 5 is safer.
  - START HERE: try n_refs=3 early — it is very likely to improve over n_refs=1 at no cost.

[B4. ref_strategy — how n_refs images are selected per HOI class]
  ALLOWED VALUES (you MUST choose exactly one): "first", "centroid", "diverse"
  - COST: zero. Pure numpy (<1ms). Does NOT affect QPS.
  - "first"    (default): take the first n_refs images in dataset order.
               Simple and deterministic. Risk: first images may share a visual
               bias (same scene, same angle) → reference is not representative.
               Use as a baseline only.
  - "centroid": compute the visual centroid of ALL class images, then pick the
               n_refs images CLOSEST to that centroid and average them.
               Result: a reference anchored to the visual core of the class.
               Reduces anchoring on outlier images.
               BEST FOR: HOI classes with moderate diversity and enough images.
               RULE: if n_refs=3 or 5, prefer "centroid" over "first".
  - "diverse":  greedy farthest-point sampling — start from the most-central
               image, then iteratively pick the image FURTHEST from already-
               selected ones. Average the selected vectors.
               Result: a reference spanning the visual diversity of the class.
               BEST FOR: HOI classes with HIGH intra-class visual variation
               (e.g., "person riding motorcycle": racing, city, dirt bikes).
               RULE: if n_refs=5 or 10, try "diverse" — it covers more corners.
  - INTERACTION WITH n_refs: "diverse" is most useful at n_refs≥3; at n_refs=1
    all strategies are identical (one image = same result regardless).
  - INTERACTION WITH alpha: better reference (centroid/diverse) → DINOv2 less noisy
    → the optimal alpha is likely HIGHER than with ref_strategy="first".
    If you switch from "first" to "centroid"/"diverse", consider increasing alpha
    by +0.05 to +0.10 to exploit the improved reference quality.
  - RECOMMENDED COMBINATIONS to try early:
      n_refs=3, ref_strategy="centroid", alpha=+0.05 vs current best alpha
      n_refs=5, ref_strategy="diverse",  alpha=+0.10 vs current best alpha

============================================================
CROSS-PARAMETER INTERACTION SUMMARY (READ CAREFULLY):
============================================================
  k ↓         → QPS ↑↑ (PRIMARY lever: k=50→295, k=100→145, k=200→58). mAP ↓ slightly.
                HALVING k DOUBLES QPS. This is the dominant QPS dial — use it first.
  efSearch ↓  → CLIP recall ↓,  QPS ≈ UNCHANGED (negligible in batch mode — do NOT use for QPS).
                Compensate mAP drop from lower efSearch with: K ↑ or alpha ↑ (slightly).
  K ↑         → mAP ↑,  QPS ↓↓ (MAJOR — see table above). To gain QPS: lower k, NOT efSearch.
  alpha ↑     → mAP ↑ up to some optimum (measured ~0.30 at K=500 efS=500), then mAP ↓.
                The true optimum at lower K is unknown — discover it empirically.
  M ↑         → mAP ↑ (higher recall ceiling). No direct QPS effect. Allows efSearch ↓
                for same mAP (which also has no QPS effect but frees a degree of freedom).
  nprobe ↓    → QPS ↑,  IVF recall ↓   → compensate with: K ↑ or alpha ↑

  THE JOINT OPTIMUM is not obvious — e.g.:
    (HNSW M=64, efS=64, K=50, alpha=0.65) may beat (HNSW M=32, efS=64, K=100, alpha=0.65)
    because halving K doubles QPS while M=64 recovers most of the lost CLIP recall,
    and constraint_strategy=object + n_refs=10 keeps mAP above τ=0.15 at k=50.
  The optimal alpha at (M=64, K=50) may differ from 0.65 — only measurement tells.
  This 4D interaction space (engine × efSearch × K × alpha) is exactly why an agent is needed.
"""

        if _PER_QUERY_CONSTRAINT is not None:
            _scenario_matrix = """\
1. "ACS-Fast" → FaissFlat or HNSW M=32 efS=64-100, k=100-200, alpha Q1/Q2/Q4=0.70-0.80, Q3=0.60-0.70,
   n_refs=5-10 centroid, constraint_strategy='object' (all 600 constrained → tiny ~600-image pool).
   Expected: batch QPS=1000-2000 (constrained ANN is very fast), mAP=0.26-0.32. Score=280-600. START HERE.

2. "ACS-Balanced" → HNSW M=32 efS=64-100, k=150-200, alpha Q1/Q2/Q4=0.75, Q3=0.65, n_refs=10 centroid
   constraint_strategy='object' keeps all queries constrained. Same QPS range, refined alpha.

3. "ACS-DINOv2-heavy" → HNSW M=48 efS=64-100, k=150, alpha Q1=0.80, Q2=0.80, Q3=0.65, Q4=0.80,
   n_refs=10 diverse. Try if centroid refs plateau — diverse sampling spans intra-class variation.

4. "Verify text-only" → alpha=0.0 for all tiers, k=100-200, any engine
   Measures pure CLIP+ACS benefit without DINOv2 contribution."""
        else:
            _scenario_matrix = """\
1. "Fast-feasible" (MAIN GOAL) → k=100 HNSW M=64 efS=64 constraint=object n_refs=10 alpha=0.65
   MEASURED QPS: k=100→145, k=50→295. START AT k=100 (safe mAP~0.19-0.22).
   Then probe k=50 M=64 alpha≥0.65 constraint=object for QPS~295 (high-risk/high-reward, mAP borderline ~0.15-0.17).
   Score target (SIEVE): k=100 feasible → Score~145. k=50 feasible → Score~295.

2. "Balanced" → k=150-200, HNSW M=32-48 efS=64
   MEASURED QPS: k=150→85, k=200→58. mAP~0.22-0.27. Score (SIEVE)~58-85. Only if k=50-100 region is exhausted.

3. "IVFFlat competitor" → nlist=128 nprobe=8 k=100
   QPS~140-190, mAP~0.17-0.19. Score (SIEVE)~140-190 if feasible. Valid alternative to HNSW for exploration.

4. "Maximize mAP only" → k=750-1000, HNSW M=32-48 efS=500
   mAP ceiling ~0.27-0.28, but ANN-dominated: batch QPS~4-10. Score (SIEVE)~4-10. AVOID.

5. "Verify text-only baseline" → alpha=0.0, any K, any engine
   Useful to measure CLIP's raw contribution without DINOv2."""

        scenario_guide = f"""
PHASE-SPECIFIC STRATEGY (current phase: {phase}):

EXPLORATION (iterations 1-{exp_end}): Aggressively probe diverse (engine, K, efSearch, alpha) combinations.
  - PRIORITY: k in {{100, 200}} with efSearch in {{32, 64, 100}} — this is where QPS is highest.
  - Try FaissFlat (exact search, highest CLIP recall at any K) and HNSW M=32/48 with low efSearch.
  - MANDATORY: Try at least ONE IVFFlat config (e.g. nlist=128 nprobe=8 k=100 or nlist=256 nprobe=16 k=200).
    IVFFlat can match or beat HNSW QPS at very low nprobe — do not skip it.
  - Vary alpha FREELY across all allowed values — the optimal alpha is UNKNOWN; discover it.
    Note: alpha=0.45 is a valid choice — it is in the allowed list.
  - Goal: identify which (engine, K, efSearch/nprobe) zone maximizes QPS while keeping mAP ≥ τ, AND find what
    alpha works best in that zone — these two are coupled and must be explored together.

EXPLOITATION (iterations {exp_end+1}-{expl_end}): Narrow around the best zone found.
  - KEEP the same engine as the current best. Do NOT switch engines in this phase.
  - PRIMARY LEVER: sweep k_neighbors around the best value (+/-1 step). k controls QPS.
    Do NOT change efSearch to gain QPS — it has negligible effect. Only adjust efSearch
    if mAP is below τ (lower efSearch → lower recall → may need to compensate with K ↑ or alpha ↑).
  - Sweep alpha around the best alpha found in exploration (+/-0.05 and +/-0.10 steps).
  - Vary n_refs and ref_strategy around the best — these are FREE (zero QPS cost).
    Priority: try n_refs=10+centroid and ref_strategy=diverse if not yet tested.
  - Goal: find the exact (engine, K, alpha, n_refs, ref_strategy) optimum.

FINE-TUNING (iterations {expl_end+1}-{total_iterations}): Micro-adjust ONE parameter at a time.
  - Step size: k_neighbors one value up/down OR alpha +/-0.05.
  - Adjust efSearch only if mAP needs fixing (not for QPS).
  - If alpha optimum is not yet confirmed, try one alpha step above and one below the current best.
  - Goal: squeeze final improvement from the Pareto frontier.

SCENARIO DECISION MATRIX:
{_scenario_matrix}
"""

        # Ablation A: hide all history from the LLM prompt.
        # The agent still runs all iterations (results accumulate in `results` list for
        # deduplication), but the LLM never sees past configs or scores, so it cannot
        # learn. This isolates whether history-conditioning is the mechanism that finds
        # the k=50 / alpha=0.80 / constraint=object operating point.
        if self.ablation_no_history:
            guidance              = "(ABLATION: history hidden — no diagnosis available)"
            all_configs_tried     = "  (ABLATION: history disabled — no prior configs shown)"
            untried_alpha_hint    = ""
            untried_refs_hint     = ""
            history_json          = "[]"
            category_feedback_str = ""
        else:
            # Full history: compact one-line per iteration (all iterations, no truncation)
            # Show snap info (suggested→actual) so LLM knows what was really run
            def _snap_note(cfg):
                notes = []
                if "_raw_alpha" in cfg:
                    notes.append(f"sugg_a={cfg['_raw_alpha']:.2f}")
                for t in range(1, 5):
                    k_raw = f"_raw_alpha_Q{t}"
                    if k_raw in cfg:
                        notes.append(f"sugg_aQ{t}={cfg[k_raw]:.2f}")
                if "_raw_k" in cfg:
                    notes.append(f"sugg_k={cfg['_raw_k']}")
                if "_raw_n_refs" in cfg:
                    notes.append(f"sugg_refs={cfg['_raw_n_refs']}")
                return f" [{', '.join(notes)}]" if notes else ""

            def _k_hist(cfg: Dict) -> str:
                return str(cfg.get('k_neighbors', 500))

            def _alpha_hist(cfg: Dict) -> str:
                if _QDS_TIERS is not None and "alpha_Q1" in cfg:
                    return "/".join(f"{cfg.get(f'alpha_Q{t}', 0.3):.2f}" for t in range(1, 5))
                return f"{cfg.get('alpha', 0.0):.2f}"

            all_configs_tried = "\n".join(
                f"  iter{h.config.get('_iter','?'):02d}: {h.config['engine']} "
                f"p={h.config.get('params',{})} k={_k_hist(h.config)} "
                f"refs={h.config.get('n_refs',1)} "
                f"a={_alpha_hist(h.config)}{_snap_note(h.config)} → Score={h.benchmark.score():.4f} "
                f"mAP={h.benchmark.map_score:.4f} QPS={h.benchmark.qps:.1f}"
                for h in history
            )

            # Show untried alpha values at the current best config's (engine, k, params)
            untried_alpha_hint = ""
            untried_refs_hint  = ""
            if history and phase in ("EXPLOITATION", "FINE-TUNING"):
                best_h = max(history, key=lambda r: r.benchmark.score())
                best_engine = best_h.config["engine"]
                best_params = best_h.config.get("params", {})
                best_alpha  = best_h.config.get("alpha", 0.0)  # scalar; used in standard mode only
                if _QDS_TIERS is not None and "alpha_Q1" in best_h.config:
                    best_alpha_str = "/".join(
                        f"Q{t}={best_h.config.get(f'alpha_Q{t}', 0.3):.2f}" for t in range(1, 5))
                else:
                    best_alpha_str = f"{best_alpha:.2f}"

                best_k = best_h.config.get("k_neighbors", 300)
                if _QDS_TIERS is not None and "alpha_Q1" in best_h.config:
                    # QDS mode: show untried alpha_Q4 (highest-impact tier for hard queries)
                    tried_q4 = {round(r.config.get("alpha_Q4", 0.7), 2)
                                for r in history
                                if r.config["engine"] == best_engine
                                and r.config.get("params", {}) == best_params}
                    untried_q4 = [a for a in ConfigProvider.ALPHA_VALUES
                                  if round(a, 2) not in tried_q4]
                    if untried_q4 and phase == "FINE-TUNING":
                        untried_alpha_hint = (
                            f"\nUNTRIED alpha_Q4 at best engine "
                            f"({best_engine} params={best_params} k={best_k}): {untried_q4}"
                            f"\n  → alpha_Q4 has highest impact: rare HOIs rely most on DINOv2."
                        )
                else:
                    tried_alphas_here = {
                        round(r.config.get("alpha", 0.0), 2)
                        for r in history
                        if r.config["engine"] == best_engine
                        and r.config.get("k_neighbors", 300) == best_k
                        and r.config.get("params", {}) == best_params
                    }
                    untried_alphas = [a for a in ConfigProvider.ALPHA_VALUES
                                      if round(a, 2) not in tried_alphas_here]
                    if untried_alphas and phase == "FINE-TUNING":
                        untried_alpha_hint = (
                            f"\nUNTRIED alpha values at best config "
                            f"({best_engine} k={best_k} params={best_params}): {untried_alphas}"
                            f"\n  → These are valid next probes. Pick one if you want to tune alpha."
                        )

                if _QDS_TIERS is not None:
                    # QDS-ACS mode: show untried k_neighbors (flat k; ACS handles constraint adaptation)
                    tried_k_here = {
                        r.config.get("k_neighbors", 500)
                        for r in history
                        if r.config["engine"] == best_engine
                        and r.config.get("params", {}) == best_params
                    }
                    untried_k = [k for k in ConfigProvider.K_NEIGHBORS_VALUES if k not in tried_k_here]
                    if untried_k and phase in ("EXPLOITATION", "FINE-TUNING"):
                        untried_refs_hint = (
                            f"\nUNTRIED k_neighbors values at best engine "
                            f"({best_engine} params={best_params} alpha={best_alpha_str}): {untried_k}"
                            f"\n  → Larger k helps recall within the ~600-image constrained pool for Q3+Q4 queries."
                        )
                else:
                    # Standard mode: show untried (n_refs, ref_strategy) combos
                    tried_refs_here = {
                        (r.config.get("n_refs", 1), r.config.get("ref_strategy", "first"))
                        for r in history
                        if r.config["engine"] == best_engine
                        and r.config.get("k_neighbors", 300) == best_k
                        and r.config.get("params", {}) == best_params
                        and round(r.config.get("alpha", 0.0), 2) == round(best_alpha, 2)
                    }
                    all_refs_combos = [
                        (n, s)
                        for n in ConfigProvider.N_REFS_VALUES
                        for s in ConfigProvider.REF_STRATEGY_VALUES
                    ]
                    untried_refs = [c for c in all_refs_combos if c not in tried_refs_here]
                    if untried_refs:
                        untried_refs_hint = (
                            f"\nUNTRIED (n_refs, ref_strategy) combos at best config "
                            f"({best_engine} k={best_k} alpha={best_alpha:.2f}): {untried_refs}"
                            f"\n  → FREE improvement — n_refs=10+centroid or n_refs=5+diverse are highest-value probes."
                        )

            # Detailed last 5 for fine-grained diagnosis
            def _k_entry(cfg: Dict):
                return cfg.get("k_neighbors", 500)

            history_json = json.dumps([{
                "eng":   h.config["engine"],
                "p":     h.config.get("params", {}),
                "k":     _k_entry(h.config),
                "refs":  h.config.get("n_refs", 1),
                "alp":   ({f"Q{t}": round(h.config.get(f"alpha_Q{t}", 0.3), 2) for t in range(1, 5)}
                          if _QDS_TIERS is not None and "alpha_Q1" in h.config
                          else round(h.config.get("alpha", 0.0), 2)),
                "cs":    h.config.get("constraint_strategy", "none"),
                "sc":    round(h.benchmark.score(), 4),
                "map":   round(h.benchmark.map_score, 4),
                "qps":   round(h.benchmark.qps, 1),
            } for h in history[-5:]], indent=1)

            # Per-category feedback (ablation mode — only when --use-category-feedback)
            category_feedback_str = ""
            if history and _CATEGORY_FEEDBACK and history[-1].benchmark.per_query_ap:
                category_feedback_str = _format_category_feedback(
                    history[-1].benchmark.per_query_ap)

        prompt = f"""You are an expert VDMS Optimization Engine for HOI Retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

{system_context}

DIAGNOSIS OF LAST RUN:
{guidance}
{chr(10) + category_feedback_str + chr(10) if category_feedback_str else ""}
{engine_definitions}

{scenario_guide}

ALL CONFIGS TRIED SO FAR (do NOT repeat any of these exact combinations):
{all_configs_tried if all_configs_tried else "  (none yet)"}
{untried_alpha_hint}{untried_refs_hint}
LAST 5 RESULTS (detailed):
{history_json}

AVAILABLE PARAMETER RANGES:
  FaissFlat:     params={{}}  (no tunable params — exact brute-force)
  FaissHNSWFlat: M={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['M']}
                 efConstruction={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efConstruction']}
                 efSearch={ConfigProvider.ENGINE_PARAMS['FaissHNSWFlat']['efSearch']}
  FaissIVFFlat:  nlist={ConfigProvider.ENGINE_PARAMS['FaissIVFFlat']['nlist']}
                 nprobe={ConfigProvider.ENGINE_PARAMS['FaissIVFFlat']['nprobe']}
  k_neighbors:          {ConfigProvider.K_NEIGHBORS_VALUES}
  {'alpha_Q1/Q2/Q3/Q4:   each independently from ' + str(ConfigProvider.ALPHA_VALUES) if _QDS_TIERS is not None else 'alpha:                ' + str(ConfigProvider.ALPHA_VALUES)}
  {'  All tiers: do NOT go below 0.50 — empirical data shows AP collapses at low alpha.' if _QDS_TIERS is not None else ''}
  {'  Q3: keep near 0.65-0.70 (negative slope above 0.70). All others: try 0.70-0.85.' if _QDS_TIERS is not None else ''}
  n_refs:               {ConfigProvider.N_REFS_VALUES}  (DINOv2 reference images per HOI class)
  ref_strategy:         {ConfigProvider.REF_STRATEGY_VALUES}  (how those n_refs images are selected — see B4)
  constraint_strategy:  {ConfigProvider.CONSTRAINT_STRATEGIES}

GRAPH CONSTRAINT STRATEGIES (PMGD metadata pre-filtering — NOVEL DIMENSION):
  VDMS stores each image's primary object category and verb category as PMGD graph properties.
  A constraint_strategy pre-filters the ANN search to only images matching the query HOI metadata:
    "none"            — no filter. Full 47,776-image ANN search. (Grid baseline behavior)
    "object"          — filter by primary_object_id. Avg ~597 images/query (1.2% of DB). High QPS gain.
    "verb"            — filter by primary_verb_id.   Avg ~419 images/query (0.9% of DB). Highest QPS.
    "object_and_verb" — both constraints. Avg <200 images/query (<0.4% of DB). Risk of low recall.

  KEY TRADEOFF: constraint reduces search space → higher QPS, but may exclude relevant images that
  have the target HOI as a SECONDARY annotation (their primary object/verb differs) → lower mAP.
  k_neighbors tradeoff depends on pool size:
    - Unconstrained (full 47K): k=100-300 balances recall vs speed.
    - Constrained pool (~600 images with "object"): k=100-200 is already near-exhaustive (17-33%).
      Do NOT use k=500 with constrained pool — it covers 83% of 600 images and adds rerank cost
      with negligible recall gain.
  The agent must jointly optimize (k_neighbors, constraint_strategy) to maximize QPS subject to mAP ≥ τ.

  GUIDANCE: Start with "object" (moderate filter, balanced tradeoff). If Score improves vs "none",
  adjust k to match pool size (k=100-200 for constrained, k=200-300 for unconstrained).
  "verb" is high-risk (many images have "hold" as primary verb — very low selectivity).
  ⚠️ CRITICAL: FaissIVFFlat does NOT support PMGD constraints. Using constraint≠"none" with
  FaissIVFFlat returns 0 candidates (mAP=0, Score=0 — wasted iteration). ONLY use constraints
  with FaissHNSWFlat or FaissFlat.

DECISION RULES (apply in order):
1. Read DIAGNOSIS — it tells you the single most important fix.
2. Use CROSS-PARAMETER INTERACTIONS to avoid breaking other metrics.
3. In EXPLORATION phase: try large parameter jumps across engines/K/alpha/constraint_strategy.
4. In FINE-TUNING phase: change only ONE parameter by ONE step.
5. alpha > 0.90 is outside the allowed list. Any value from 0.0 to 0.90 in 0.05 steps is valid.
6. Never repeat a config already in HISTORY.
7. ANTI-COLLAPSE: {("[ABLATION: history hidden]" if self.ablation_no_history else f"*** MANDATORY *** Your last {consecutive_alpha_only} consecutive iterations changed ONLY alpha (same engine/params/k). You MUST change engine, OR M/efSearch/nlist/nprobe, OR k_neighbors in this iteration. Another alpha-only proposal will be rejected and retried." if consecutive_alpha_only >= 2 else f"[{consecutive_alpha_only} consecutive alpha-only iter(s). Change (engine/params/k) if this reaches 2.]")}
8. STAGNATION: {("[ABLATION: history hidden]" if self.ablation_no_history else "[ABLATION: stagnation detection disabled]" if self.ablation_no_phases else f"⚠ WARNING: No improvement for {iters_since_improvement} consecutive iterations (best stuck at {current_best_score:.4f}). MANDATORY: Escape the current zone — try constraint_strategy='object' if not yet tried, OR try a completely different engine, OR a K value far from the current best k={history[-1].config.get('k_neighbors',100) if history else 100}. Do NOT fine-tune around the stuck config." if iters_since_improvement >= 5 else f"[{iters_since_improvement} iter(s) without improvement — OK]")}

OUTPUT JSON ONLY (no markdown, no explanation outside the JSON):
""" + ("""
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 64, "efConstruction": 200, "efSearch": 64}},
  "k_neighbors": 50,
  "alpha_Q1": 0.80,
  "alpha_Q2": 0.80,
  "alpha_Q3": 0.65,
  "alpha_Q4": 0.85,
  "n_refs": 10,
  "ref_strategy": "centroid",
  "constraint_strategy": "none",
  "reasoning": "k=50 gives ~590 QPS (ACS routes Q3+Q4 to numpy, only 300 queries hit VDMS). M=64 maximizes recall ceiling within Q1+Q2 unconstrained VDMS search. alpha_Q3=0.65 (negative slope above 0.65). alpha_Q4=0.85 (rarest HOIs rely most on DINOv2). n_refs=10 centroid stabilizes mAP above τ. constraint_strategy=none: forced internally by ACS for Q1+Q2 VDMS queries."
}}
""" if _QDS_TIERS is not None else """
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 64, "efConstruction": 200, "efSearch": 64}},
  "k_neighbors": 100,
  "alpha": 0.65,
  "n_refs": 10,
  "ref_strategy": "centroid",
  "constraint_strategy": "object",
  "reasoning": "Safe start: k=100 gives QPS~145 (PMGD-dominated). M=64 maximizes recall ceiling. constraint_strategy=object pre-filters to ~597 images/query, boosting QPS further. alpha=0.65 with n_refs=10 centroid reliably keeps mAP above τ=0.15. Next: try k=50 (QPS~295) after confirming feasibility here."
}}
""") + """
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
                print(f"LLM Error: Empty LLM response (content=None) — attempt {_api_attempt+1}/3, retrying…")
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
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    json_str = match.group(0)

            config_data = json.loads(json_str.strip())

            # Validation + snap-to-canonical layer.
            # If the LLM outputs a value not in the allowed list, snap it to the
            # nearest allowed value so all three methods stay in identical search space.
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
                    # use middle value as default
                    clean_params[k] = allowed_vals[len(allowed_vals) // 2]

            n_refs       = snap(int(config_data.get("n_refs", 1)),
                                provider.N_REFS_VALUES)
            ref_strategy_raw = str(config_data.get("ref_strategy", "first"))
            ref_strategy = (ref_strategy_raw
                            if ref_strategy_raw in provider.REF_STRATEGY_VALUES
                            else "first")

            constraint_strategy_raw = str(config_data.get("constraint_strategy", "none"))
            constraint_strategy = (constraint_strategy_raw
                                   if constraint_strategy_raw in provider.CONSTRAINT_STRATEGIES
                                   else "none")

            final_config = {
                "engine":               engine,
                "params":               clean_params,
                "n_refs":               n_refs,
                "ref_strategy":         ref_strategy,
                "constraint_strategy":  constraint_strategy,
            }

            if _QDS_TIERS is not None:
                # QDS mode: per-tier alpha
                defaults = [0.30, 0.40, 0.55, 0.70]
                for t, default in zip(range(1, 5), defaults):
                    raw_v = float(config_data.get(f"alpha_Q{t}", default))
                    snapped_v = snap(raw_v, provider.ALPHA_VALUES)
                    final_config[f"alpha_Q{t}"] = snapped_v
                    if abs(raw_v - snapped_v) > 1e-6:
                        final_config[f"_raw_alpha_Q{t}"] = raw_v
                alpha_display_str = "/".join(f"Q{t}={final_config[f'alpha_Q{t}']:.2f}" for t in range(1,5))
            else:
                alpha = snap(float(config_data.get("alpha", 0.0)), provider.ALPHA_VALUES)
                final_config["alpha"] = alpha
                raw_alpha_val = float(config_data.get("alpha", 0.0))
                if abs(raw_alpha_val - alpha) > 1e-6:
                    final_config["_raw_alpha"] = raw_alpha_val
                alpha_display_str = f"{alpha:.2f}"

            k_neighbors = snap(int(config_data.get("k_neighbors", 100)),
                               provider.K_NEIGHBORS_VALUES)
            final_config["k_neighbors"] = k_neighbors

            raw_k_val = int(config_data.get("k_neighbors", 100))
            if raw_k_val != k_neighbors:
                final_config["_raw_k"] = raw_k_val
            raw_nrefs_val = int(config_data.get("n_refs", 1))
            if raw_nrefs_val != n_refs:
                final_config["_raw_n_refs"] = raw_nrefs_val

            # ---- Debug: show exactly what the LLM suggested ----
            reasoning_text = config_data.get("reasoning", "(no reasoning field)")
            raw_engine  = config_data.get("engine", "?")
            raw_params  = config_data.get("params", {})
            raw_k       = config_data.get("k_neighbors", "?")
            snapped = []
            if raw_engine != engine:
                snapped.append(f"engine {raw_engine!r}→{engine!r}")
            for pk in clean_params:
                raw_v = raw_params.get(pk)
                if raw_v is not None and int(raw_v) != clean_params[pk]:
                    snapped.append(f"{pk} {raw_v}→{clean_params[pk]}")
            if isinstance(raw_k, (int, float)) and int(raw_k) != k_neighbors:
                snapped.append(f"k_neighbors {raw_k}→{k_neighbors}")
            snap_note = f"  [SNAPPED: {', '.join(snapped)}]" if snapped else ""
            _k_print = str(final_config.get("k_neighbors", 100))
            print(f"\n[LLM] ── Suggested config ──────────────────────────────")
            print(f"[LLM]   engine={engine}  params={clean_params}")
            print(f"[LLM]   k={_k_print}  alpha={alpha_display_str}  n_refs={n_refs}{snap_note}")
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
                  iteration: int = 0, backbone: str = "mobileclip_s2") -> BenchmarkResult:
    """
    Connect to VDMS, build CLIP_Set, run two-stage text+DINOv2 retrieval,
    return BenchmarkResult with mAP and QPS.

    Uses a unique set name per iteration to avoid DeleteDescriptorSet issues.
    """
    from benchmark_hico_det import load_hico_metadata

    engine               = config["engine"]
    engine_params        = config.get("params", {})
    n_refs               = int(config.get("n_refs", 1))
    ref_strategy         = str(config.get("ref_strategy", "first"))
    constraint_strategy  = str(config.get("constraint_strategy", "none"))
    set_name             = f"CLIP_Set_{iteration}"

    if _QDS_TIERS is not None and "alpha_Q1" in config:
        tier_alphas = {i: float(config.get(f"alpha_Q{i}", [0.30, 0.40, 0.55, 0.70][i-1]))
                       for i in range(1, 5)}
        alpha = np.array([tier_alphas[int(_QDS_TIERS[q])] for q in range(len(_QDS_TIERS))],
                         dtype=np.float32)
        alpha_display = (f"Q1={tier_alphas[1]:.2f}/Q2={tier_alphas[2]:.2f}/"
                         f"Q3={tier_alphas[3]:.2f}/Q4={tier_alphas[4]:.2f}")
        alpha_scalar = float(np.mean(alpha))
    else:
        alpha = float(config.get("alpha", 0.0))
        alpha_display = f"{alpha:.2f}"
        alpha_scalar  = float(alpha)

    k_neighbors = int(config.get("k_neighbors", 500))
    k_display   = str(k_neighbors)
    k_scalar    = k_neighbors

    data = _load_hico_data_cached(dataset_dir, backbone=backbone)

    # Load PMGD metadata / per-class indices as needed.
    obj_vocab  = None
    verb_vocab = None
    img_meta   = None
    if _PER_CLASS_INDICES is not None:
        # Per-class mode: Q3/Q4 bypass VDMS (exact numpy search). Q1/Q2 use unconstrained VDMS.
        constraint_strategy = "none"   # force — VDMS constraint not needed, per-class handles it
        obj_vocab = _PER_CLASS_OBJ_VOCAB  # needed by benchmark for per-class obj_id lookup
    elif constraint_strategy != "none" or _PER_QUERY_CONSTRAINT is not None:
        img_meta, obj_vocab, verb_vocab = load_hico_metadata(dataset_dir)

    # Build DINOv2 reference vectors (single n_refs for all queries).
    if n_refs == 1 and ref_strategy == "first":
        dinov2_q = data["dinov2_q"]
    else:
        dinov2_q = build_dinov2_refs(
            data["hoi_index"], data["dinov2_db"], data["hoi_keys"],
            n_refs=n_refs, ref_strategy=ref_strategy)

    print(f"\n[Iter {iteration}] backbone={data['clip_label']}  engine={engine} params={engine_params} "
          f"k={k_display} alpha={alpha_display} n_refs={n_refs} ref_strategy={ref_strategy} "
          f"constraint={constraint_strategy}")

    db = vdms.vdms()
    db.connect("localhost", port)
    res, _ = db.query([{"GetSchema": {}}])
    if not res:
        raise RuntimeError(f"Cannot connect to VDMS on port {port}")

    print(f"[Iter {iteration}] Building {set_name}  (graph={'yes' if constraint_strategy != 'none' else 'no'})...")
    t_build_start = time.time()
    setup_clip_index(db, data["clip_db"], engine, engine_params,
                     set_name=set_name, metadata=img_meta)
    index_build_s = time.time() - t_build_start
    print(f"[Iter {iteration}] Index built in {index_build_s:.1f}s")

    print(f"[Iter {iteration}] Running {len(data['hoi_keys'])} queries...")
    metrics = run_hico_text_dinov2_rerank(
        db,
        data["text_q"],
        dinov2_q,
        data["dinov2_db"],
        data["rel_sets"],
        data["junk_sets"],
        alpha=alpha,
        k_neighbors=k_neighbors,
        k_rrf=60,
        hoi_keys=data["hoi_keys"],
        clip_name=set_name,
        constraint_strategy=constraint_strategy,
        obj_vocab=obj_vocab,
        verb_vocab=verb_vocab,
        per_query_constraint=_PER_QUERY_CONSTRAINT,
        per_class_indices=_PER_CLASS_INDICES,
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
        k_neighbors=     k_scalar,
        alpha=           alpha_scalar,
        timestamp=       time.time(),
        per_query_ap=    metrics.get("per_query_ap", []),
    )


# ===========================================================================
# OPTIMIZATION LOOP
# ===========================================================================

def run_optimization(method: str, iterations: int, port: int, dataset_dir: Path,
                     seed: int = 42, api_key: Optional[str] = None,
                     model: str = "minimax/minimax-m2.1",
                     vdms_restart_cmd: Optional[str] = None,
                     backbone: str = "mobileclip_s2",
                     patience: int = 3,
                     ablation_no_history: bool = False,
                     ablation_no_phases: bool = False,
                     t_exp_frac: float = 0.40,
                     t_expl_frac: float = 0.75) -> List[IterationResult]:

    print(f"Starting {method} optimization. Iterations: {iterations}, Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    config_provider = ConfigProvider(seed=seed)
    llm_provider    = LLMProvider(api_key, model=model, backbone=backbone, seed=seed,
                                  ablation_no_history=ablation_no_history,
                                  ablation_no_phases=ablation_no_phases,
                                  t_exp_frac=t_exp_frac,
                                  t_expl_frac=t_expl_frac) if api_key else None

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

    # Optuna study — created once, ask-tell API used inside the loop
    optuna_study = None
    optuna_trial = None
    if method == "optuna":
        optuna_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
    elif method == "gp_bo":
        # GP-BO (VDTuner-style): GP surrogate with Expected Improvement.
        optuna_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GPSampler(seed=seed),
        )

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
            # Cold-start LLM: no warm-start. The agent must discover the optimal
            # region from scratch. This is required so that convergence speed is
            # a meaningful metric — a warm-start at the near-optimum makes all
            # remaining iterations trivial alpha fine-tuning.

            # Build set of already-run config keys for duplicate detection.
            # Use f-string formatting for alpha to avoid IEEE-754 round() ambiguity.
            # ref_strategy MUST be included — omitting it causes 5-tuple vs 6-tuple
            # mismatch with the lookup key below, making dedup silently do nothing.
            def _k_key(cfg: Dict):
                return cfg.get("k_neighbors", 500)

            def _alpha_key(cfg: Dict) -> str:
                if _QDS_TIERS is not None and "alpha_Q1" in cfg:
                    return "/".join(f"{cfg.get(f'alpha_Q{t}', 0.3):.2f}" for t in range(1, 5))
                return f"{cfg.get('alpha', 0.0):.2f}"

            tried_keys = {
                (r.config["engine"],
                 str(sorted(r.config.get("params", {}).items())),
                 _k_key(r.config),
                 _alpha_key(r.config),
                 r.config.get("n_refs", 1),
                 r.config.get("ref_strategy", "first"),
                 r.config.get("constraint_strategy", "none"))
                for r in results
            }

            # Anti-collapse: count consecutive tail iters that only changed alpha
            # (same engine + params + k).  If ≥3, force structural change.
            consecutive_alpha_only = 0
            if len(results) >= 2:
                ref_cfg = results[-1].config
                ref_struct = (ref_cfg.get("engine"),
                              str(sorted(ref_cfg.get("params", {}).items())),
                              _k_key(ref_cfg),
                              ref_cfg.get("n_refs", 1),
                              ref_cfg.get("ref_strategy", "first"),
                              ref_cfg.get("constraint_strategy", "none"))
                for h in reversed(results):
                    h_struct = (h.config.get("engine"),
                                str(sorted(h.config.get("params", {}).items())),
                                _k_key(h.config),
                                h.config.get("n_refs", 1),
                                h.config.get("ref_strategy", "first"),
                                h.config.get("constraint_strategy", "none"))
                    if h_struct == ref_struct:
                        consecutive_alpha_only += 1
                    else:
                        break

            for _attempt in range(4):
                try:
                    config, reasoning = asyncio.run(
                        llm_provider.query_llm(i + 1, results, iterations,
                                               consecutive_alpha_only=consecutive_alpha_only,
                                               iters_since_improvement=iters_since_improvement))
                except Exception as _llm_err:
                    print(f"[LLM] Fatal error on attempt {_attempt+1}: {_llm_err} — falling back to random.")
                    for _fb in range(8):
                        config = ConfigProvider(seed=seed + i + _attempt + _fb).get_random_config()
                        config["_iter"] = i + 1
                        _fb_key = (config["engine"],
                                   str(sorted(config.get("params", {}).items())),
                                   _k_key(config), _alpha_key(config),
                                   config.get("n_refs", 1),
                                   config.get("ref_strategy", "first"),
                                   config.get("constraint_strategy", "none"))
                        if _fb_key not in tried_keys:
                            break
                    reasoning = f"Fallback random (LLM error: {_llm_err})"
                    break
                config["_iter"] = i + 1
                key = (config["engine"],
                       str(sorted(config.get("params", {}).items())),
                       _k_key(config),
                       _alpha_key(config),
                       config.get("n_refs", 1),
                       config.get("ref_strategy", "first"),
                       config.get("constraint_strategy", "none"))
                if key not in tried_keys:
                    break
                print(f"[Dedup] LLM repeated a tried config (attempt {_attempt+1}), retrying...")
            else:
                # After 4 attempts still duplicate — fall back to random
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
                 r.config.get("k_neighbors", 500),
                 f"{r.config.get('alpha', 0.0):.2f}",
                 r.config.get("n_refs", 1),
                 r.config.get("ref_strategy", "first"),
                 r.config.get("constraint_strategy", "none"))
                for r in results
            }
            for _dedup_attempt in range(4):
                optuna_trial = optuna_study.ask()
                engine = optuna_trial.suggest_categorical(
                    "engine", ["FaissHNSWFlat", "FaissIVFFlat", "FaissFlat"])
                params = {}
                if engine == "FaissHNSWFlat":
                    params["M"]              = optuna_trial.suggest_categorical(
                        "M", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["M"])
                    params["efConstruction"] = 200
                    params["efSearch"]       = optuna_trial.suggest_categorical(
                        "efSearch", ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"])
                elif engine == "FaissIVFFlat":
                    params["nlist"]  = optuna_trial.suggest_categorical(
                        "nlist", ConfigProvider.ENGINE_PARAMS["FaissIVFFlat"]["nlist"])
                    params["nprobe"] = optuna_trial.suggest_categorical(
                        "nprobe", ConfigProvider.ENGINE_PARAMS["FaissIVFFlat"]["nprobe"])
                k_neighbors  = optuna_trial.suggest_categorical(
                    "k_neighbors", ConfigProvider.K_NEIGHBORS_VALUES)
                alpha        = optuna_trial.suggest_categorical(
                    "alpha", ConfigProvider.ALPHA_VALUES)
                n_refs              = optuna_trial.suggest_categorical(
                    "n_refs", ConfigProvider.N_REFS_VALUES)
                ref_strategy        = optuna_trial.suggest_categorical(
                    "ref_strategy", ConfigProvider.REF_STRATEGY_VALUES)
                constraint_strategy = optuna_trial.suggest_categorical(
                    "constraint_strategy", ConfigProvider.CONSTRAINT_STRATEGIES)
                opt_key = (engine, str(sorted(params.items())), int(k_neighbors),
                           f"{float(alpha):.2f}", int(n_refs), ref_strategy, constraint_strategy)
                if opt_key not in opt_tried_keys:
                    break
                dup_score = next(
                    (r.benchmark.score() for r in results if
                     (r.config["engine"],
                      str(sorted(r.config.get("params", {}).items())),
                      r.config.get("k_neighbors", 500),
                      f"{r.config.get('alpha', 0.0):.2f}",
                      r.config.get("n_refs", 1),
                      r.config.get("ref_strategy", "first"),
                      r.config.get("constraint_strategy", "none")) == opt_key), 0.0)
                optuna_study.tell(optuna_trial, dup_score)
                optuna_trial = None
                print(f"[Optuna Dedup] Attempt {_dedup_attempt+1}: duplicate, told score={dup_score:.1f}, retrying...")
            else:
                config = ConfigProvider(seed=seed + i).get_random_config()
                config["_iter"] = i + 1
                reasoning = "Fallback random (Optuna repeated configs 4x)"
                optuna_trial = None
                print("[Optuna Dedup] Falling back to random after 4 duplicate Optuna suggestions.")
            if optuna_trial is not None:
                config = {
                    "engine":               engine,
                    "params":               params,
                    "k_neighbors":          int(k_neighbors),
                    "alpha":                float(alpha),
                    "n_refs":               int(n_refs),
                    "ref_strategy":         ref_strategy,
                    "constraint_strategy":  constraint_strategy,
                    "_iter":                i + 1,
                }
                reasoning = f"Baseline (Optuna TPE trial {optuna_trial.number})"
        elif method == "random":
            config    = configs[i]
            config["_iter"] = i + 1
            reasoning = "Baseline (Random)"
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"  engine:       {config['engine']}")
        print(f"  params:       {config.get('params', {})}")
        print(f"  k_neighbors:  {config.get('k_neighbors', 500)}")
        if _QDS_TIERS is not None and "alpha_Q1" in config:
            print(f"  alpha:        " + "/".join(
                f"Q{t}={config.get(f'alpha_Q{t}', 0.3):.2f}" for t in range(1, 5)))
        else:
            print(f"  alpha:        {config.get('alpha', 0.0):.2f}")

        try:
            benchmark = run_benchmark(port, str(dataset_dir), config, iteration=i + 1, backbone=backbone)

            print(f"\n[Results iter {i+1}]")
            score_label = "QPS [QDS-constrained]" if (_QDS_WEIGHTS is not None and benchmark.per_query_ap) else "QPS [constrained]"
            print(f"  Score ({score_label}): {benchmark.score():.4f}")
            print(f"  mAP (plain):       {benchmark.map_score:.4f}")
            if _QDS_WEIGHTS is not None and benchmark.per_query_ap:
                w = _QDS_WEIGHTS
                n = min(len(w), len(benchmark.per_query_ap))
                weighted_map = float(np.sum(w[:n] * np.array(benchmark.per_query_ap[:n])) / np.sum(w[:n]))
                print(f"  mAP (QDS-weighted): {weighted_map:.4f}  ← threshold check uses this (τ={_MAP_THRESHOLD:.3f})")
            print(f"  QPS:               {benchmark.qps:.2f}")
            print(f"  P@10:              {benchmark.precision_at_10:.4f}")
            print(f"  nDCG@10:           {benchmark.ndcg_at_10:.4f}")

            results.append(IterationResult(i + 1, config, benchmark, reasoning, method))

            # Report score back to Optuna study (ask-tell API)
            if method in ("optuna", "gp_bo") and optuna_trial is not None:
                optuna_study.tell(optuna_trial, benchmark.score())
                optuna_trial = None

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
            sys.exit(1)

    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    if results:
        best = max(results, key=lambda r: r.benchmark.score())
        print(f"Best Score:  {best.benchmark.score():.4f}")
        print(f"Best mAP:    {best.benchmark.map_score:.4f}")
        print(f"Best QPS:    {best.benchmark.qps:.2f}")
        print(f"Best Config: {best.config}")

    return results


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided VDMS Optimizer for HICO-DET HOI Retrieval")
    parser.add_argument("--port", type=int, default=55575)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Root dataset dir (e.g. /work/hdd/bdjd/vdms/datasets)")
    parser.add_argument("--method",
                        choices=["hyperparameter_only", "random", "grid", "optuna", "gp_bo"],
                        required=True)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON path for results")
    parser.add_argument("--model", type=str, default="minimax/minimax-m2.1",
                        help="OpenRouter model ID")
    parser.add_argument("--vdms-restart-cmd", type=str, default=None,
                        help="Shell command to restart VDMS between iterations for clean DB")
    parser.add_argument("--clip-backbone", type=str, default="mobileclip_s2",
                        choices=["mobileclip_s2", "vitl14"],
                        help="CLIP backbone: mobileclip_s2 (512-d) or vitl14 (768-d)")
    parser.add_argument("--use-qds-objective", action="store_true",
                        help="QDS-ACS mode: hard Q3/Q4 queries automatically use object-constraint "
                             "filtering (search within ~600-800 object-class images vs 47K). "
                             "Objective stays: maximize QPS subject to mAP ≥ τ.")
    parser.add_argument("--qds-standard-threshold", action="store_true",
                        help="When set alongside --use-qds-objective: keep QDS-guided exploration "
                             "and per-class search active, but evaluate feasibility with standard "
                             "raw (unweighted) mAP ≥ τ. Makes score directly comparable to other "
                             "methods. Results saved separately from weighted-threshold QDS runs.")
    parser.add_argument("--use-category-feedback", action="store_true",
                        help="Provide per-QDS-quartile mAP breakdown each iteration to the LLM. "
                             "When combined with --use-qds-objective the scoring is QDS-weighted; "
                             "otherwise: maximize QPS subject to plain mAP ≥ τ.")
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop if no improvement for this many consecutive iterations (0=disabled)")
    parser.add_argument("--ablation-no-history", action="store_true",
                        help="Ablation A: hide all past results from LLM prompt — agent cannot learn from history.")
    parser.add_argument("--ablation-no-phases", action="store_true",
                        help="Ablation B: lock agent in EXPLORATION phase forever — removes exploitation/fine-tuning transitions.")
    parser.add_argument("--t-exp-frac", type=float, default=0.40,
                        help="Exploration phase end as fraction of N (default 0.40).")
    parser.add_argument("--t-expl-frac", type=float, default=0.75,
                        help="Exploitation phase end as fraction of N (default 0.75).")
    parser.add_argument("--map-threshold", type=float, default=0.15,
                        help="Constrained optimization threshold τ: Score=QPS if mAP≥τ else 0. "
                             "Default 0.15. Run at multiple values (0.10, 0.15, 0.20) for Pareto analysis.")
    args = parser.parse_args()

    api_key = None
    if args.method == "hyperparameter_only":
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY not set")
            sys.exit(1)

    global _MAP_THRESHOLD, _QDS_STANDARD_THRESHOLD
    _MAP_THRESHOLD = args.map_threshold
    _QDS_STANDARD_THRESHOLD = args.qds_standard_threshold
    threshold_label = "raw mAP (standard)" if _QDS_STANDARD_THRESHOLD else "mAP"
    print(f"[Score] Constrained objective: Score = QPS  if {threshold_label} ≥ {_MAP_THRESHOLD:.3f} else 0")

    if args.use_qds_objective:
        global _QDS_WEIGHTS, _QDS_TIERS, _PER_QUERY_CONSTRAINT, _PER_CLASS_INDICES, _PER_CLASS_OBJ_VOCAB
        print("[QDS-ACS] Loading data to compute QDS tiers...")
        _data = _load_hico_data_cached(str(args.dataset_dir), backbone=args.clip_backbone)
        _QDS_WEIGHTS = _compute_qds_weights(
            _data["text_q"], _data["clip_db"], _data["hoi_keys"]
        )
        # Assign difficulty tiers 1-4 based on QDS weight percentiles.
        _w = _QDS_WEIGHTS
        _q25, _q50, _q75 = float(np.percentile(_w, 25)), float(np.percentile(_w, 50)), float(np.percentile(_w, 75))
        _QDS_TIERS = np.ones(len(_w), dtype=np.int32)
        _QDS_TIERS[(_w > _q25) & (_w <= _q50)] = 2
        _QDS_TIERS[(_w > _q50) & (_w <= _q75)] = 3
        _QDS_TIERS[_w > _q75] = 4
        # Q3+Q4 (hardest 50%) → per-class exact numpy search (bypasses VDMS constraint).
        # Q1+Q2 (easiest 50%) → standard unconstrained VDMS search.
        _PER_QUERY_CONSTRAINT = np.where(_QDS_TIERS >= 3, "object", "none")
        n_constrained = int((_PER_QUERY_CONSTRAINT == "object").sum())
        print(f"[QDS-ACS] Tiers: Q1={(_QDS_TIERS==1).sum()} Q2={(_QDS_TIERS==2).sum()} "
              f"Q3={(_QDS_TIERS==3).sum()} Q4={(_QDS_TIERS==4).sum()} queries")
        print(f"[QDS-ACS] {n_constrained} hard queries (Q3+Q4) will use per-class exact numpy search.")

        # Build per-object-class CLIP indices once (offline, before the optimization loop).
        # Q3+Q4 queries bypass VDMS and search the ~600-image per-class array via dot product.
        print("[QDS-ACS] Building per-class CLIP indices (one-time)...")
        from benchmark_hico_det import load_hico_metadata
        _img_meta_pc, _obj_vocab_pc, _ = load_hico_metadata(str(args.dataset_dir))
        _PER_CLASS_INDICES  = build_per_class_clip_indices(
            _data["clip_db"], _img_meta_pc, _obj_vocab_pc)
        _PER_CLASS_OBJ_VOCAB = _obj_vocab_pc
        print("[QDS-ACS] Per-class indices ready. Q3/Q4 will use exact search; Q1/Q2 use VDMS.")

    if args.use_category_feedback:
        global _CATEGORY_FEEDBACK, _QDS_FOR_FEEDBACK
        _CATEGORY_FEEDBACK = True
        print("[CategoryFeedback] Loading data to compute per-class QDS scores...")
        _data = _load_hico_data_cached(str(args.dataset_dir), backbone=args.clip_backbone)
        _QDS_FOR_FEEDBACK = _compute_qds_scores(
            _data["text_q"], _data["clip_db"], _data["hoi_keys"]
        )
        print(f"[CategoryFeedback] Enabled: LLM will receive per-QDS-quartile mAP each iteration.")

    results = run_optimization(
        method=                args.method,
        iterations=            args.iterations,
        port=                  args.port,
        dataset_dir=           args.dataset_dir,
        seed=                  args.seed,
        api_key=               api_key,
        model=                 args.model,
        vdms_restart_cmd=      args.vdms_restart_cmd,
        backbone=              args.clip_backbone,
        patience=              args.patience,
        ablation_no_history=   args.ablation_no_history,
        ablation_no_phases=    args.ablation_no_phases,
        t_exp_frac=            args.t_exp_frac,
        t_expl_frac=           args.t_expl_frac,
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
