"""
LLM-Guided FAISS Hyperparameter Optimizer for HICO-DET HOI Retrieval.

Port of hico_agent_optimizer.py to FAISS (faiss-cpu) as a second ANN backend.
Demonstrates that the LLM phase-conditioned optimization protocol is system-agnostic:
same parameter space, same SIEVE objective, same 4 optimizers (LLM/Grid/Random/Optuna).

Note: Milvus uses FAISS as its underlying ANN engine. This benchmark represents
the core ANN performance of the FAISS/Milvus family of vector databases using
FAISS directly (faiss-cpu), which avoids server-client overhead and gives a clean
view of ANN index parameter effects.

Key differences from VDMS:
  - ANN backend: faiss-cpu (IndexHNSWFlat, IndexIVFFlat, IndexFlatIP)
  - No server/container overhead — pure in-process index
  - No BATCH_SIZE=50 constraint (no PMGD journal limit)
  - Constraint strategies via numpy subset search (no scalar-filter overhead)
  - mAP and DINOv2 reranking: identical numpy pipeline

Usage:
    python milvus_hico_optimizer.py \\
        --dataset-dir /work/hdd/bdjd/vdms/datasets \\
        --method hyperparameter_only \\
        --iterations 50 \\
        --seed 42 \\
        --output milvus_hico_llm_s42.json

Install: pip install faiss-cpu
"""

import sys
import time
import json
import random
import asyncio
import re
import os
import argparse
import numpy as np
import optuna
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu", file=sys.stderr)
    sys.exit(1)

load_dotenv()

# ── Constrained-optimization threshold (SIEVE) ───────────────────────────────
_MAP_THRESHOLD: float = 0.15   # set from --map-threshold at startup


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    qps:             float
    map_score:       float
    latency_ms:      float
    index_build_s:   float
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (self-contained — no benchmark_hico_det import)
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_CACHE: Dict = {}


def _load_data(dataset_dir: str) -> Dict:
    global _DATA_CACHE
    if _DATA_CACHE:
        return _DATA_CACHE

    d = Path(dataset_dir) / "hico_det"
    print("[Data] Loading HICO-DET embeddings…")
    t0 = time.time()

    clip_db  = np.load(str(d / "hico_clipvitl14_db.npy")).astype(np.float32)
    dino_db  = np.load(str(d / "hico_dinov2.npy")).astype(np.float32)
    text_all = np.load(str(d / "hico_clipvitl14_text_queries.npy")).astype(np.float32)

    with open(d / "hico_clipvitl14_key_order.json") as f:
        key_order = json.load(f)
    text_q_map = {k: text_all[i] for i, k in enumerate(key_order)}

    with open(d / "hico_hoi_index.json") as f:
        hoi_index = json.load(f)

    with open(d / "hico_gt.json") as f:
        gt_list = json.load(f)

    with open(d / "hico_metadata.json") as f:
        metadata = json.load(f)

    # Build per-image sets indexed by image id (int)
    pos_hoi_per_img: Dict[int, set]  = {}
    amb_hoi_per_img: Dict[int, set]  = {}
    for entry in gt_list:
        img_id = int(entry["id"])
        pos_hoi_per_img[img_id] = {tuple(h) for h in entry.get("positive_hoi", [])}
        amb_hoi_per_img[img_id] = {tuple(h) for h in entry.get("ambiguous_hoi", [])}

    # Canonical HOI key list (600 classes)
    hoi_keys = list(hoi_index.keys())

    # rel_sets[i] = set of image IDs positive for hoi_keys[i]
    # junk_sets[i] = set of image IDs ambiguous for hoi_keys[i]
    rel_sets:  List[set] = []
    junk_sets: List[set] = []
    text_q:    List[np.ndarray] = []

    for key in hoi_keys:
        rel_sets.append(set(hoi_index[key]))
        # HOI key format: "obj|verb" e.g. "motorcycle|ride"
        parts = key.split("|", 1)
        hoi_tuple = tuple(parts) if len(parts) == 2 else (key,)
        junk = set()
        for img_id, amb in amb_hoi_per_img.items():
            if hoi_tuple in amb:
                junk.add(img_id)
        junk_sets.append(junk)
        text_q.append(text_q_map.get(key, np.zeros(768, dtype=np.float32)))

    text_q_arr = np.stack(text_q, axis=0)  # (600, 768)

    # Per-image primary object/verb IDs for constraint filtering
    # hico_metadata.json uses "img_id" (not "id")
    primary_obj = {}   # img_id -> obj_id int
    primary_verb = {}  # img_id -> verb_id int
    for entry in metadata:
        img_id = int(entry["img_id"])
        primary_obj[img_id]  = int(entry.get("primary_object_id", -1))
        primary_verb[img_id] = int(entry.get("primary_verb_id",   -1))

    # Build HOI query object/verb IDs for constraint filtering
    hoi_obj_ids  = []  # int per HOI class
    hoi_verb_ids = []
    with open(d / "hico_object_vocab.json") as f:
        obj_vocab = json.load(f)  # {obj_name: obj_id}
    with open(d / "hico_verb_vocab.json") as f:
        verb_vocab = json.load(f)  # {verb_name: verb_id}

    for key in hoi_keys:
        parts = key.split("|", 1)
        obj_name  = parts[0] if len(parts) == 2 else ""
        verb_name = parts[1] if len(parts) == 2 else ""
        hoi_obj_ids.append(obj_vocab.get(obj_name, -1))
        hoi_verb_ids.append(verb_vocab.get(verb_name, -1))

    # Precompute per-object and per-verb image index arrays for constraint filtering
    # obj_id -> sorted array of row indices into clip_db
    obj_img_map:  Dict[int, np.ndarray] = {}
    verb_img_map: Dict[int, np.ndarray] = {}
    for img_id in range(len(clip_db)):
        o = primary_obj.get(img_id, -1)
        v = primary_verb.get(img_id, -1)
        obj_img_map.setdefault(o, []).append(img_id)
        verb_img_map.setdefault(v, []).append(img_id)
    obj_img_map  = {k: np.array(v) for k, v in obj_img_map.items()}
    verb_img_map = {k: np.array(v) for k, v in verb_img_map.items()}

    # Build flat per-image arrays once (used by _build_faiss_index every rebuild)
    n_imgs = len(clip_db)
    img_obj_ids_arr  = np.full(n_imgs, -1, dtype=np.int64)
    img_verb_ids_arr = np.full(n_imgs, -1, dtype=np.int64)
    for oid, idxs in obj_img_map.items():
        img_obj_ids_arr[idxs] = oid
    for vid, idxs in verb_img_map.items():
        img_verb_ids_arr[idxs] = vid

    print(f"[Data] Loaded in {time.time()-t0:.1f}s  "
          f"clip_db={clip_db.shape}  dino_db={dino_db.shape}  text_q={text_q_arr.shape}  "
          f"n_classes={len(hoi_keys)}")

    _DATA_CACHE = {
        "clip_db":           clip_db,
        "dino_db":           dino_db,
        "text_q":            text_q_arr,
        "hoi_keys":          hoi_keys,
        "rel_sets":          rel_sets,
        "junk_sets":         junk_sets,
        "hoi_obj_ids":       hoi_obj_ids,
        "hoi_verb_ids":      hoi_verb_ids,
        "obj_img_map":       obj_img_map,
        "verb_img_map":      verb_img_map,
        "img_obj_ids_arr":   img_obj_ids_arr,
        "img_verb_ids_arr":  img_verb_ids_arr,
    }
    return _DATA_CACHE


# ═══════════════════════════════════════════════════════════════════════════════
# mAP COMPUTATION (Oxford-style)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ap(ranked_ids: List[int], rel_set: set, junk_set: set) -> float:
    n_pos = len(rel_set)
    if n_pos == 0:
        return 0.0
    ap = tp = fp = 0.0
    for img_id in ranked_ids:
        if img_id in junk_set:
            continue
        if img_id in rel_set:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / n_pos


def _compute_map(all_ranked: List[List[int]], rel_sets: List[set],
                 junk_sets: List[set]) -> Tuple[float, List[float]]:
    aps = [_compute_ap(r, rel_sets[i], junk_sets[i]) for i, r in enumerate(all_ranked)]
    return float(np.mean(aps)), aps


def _compute_precision_at_k(ranked_ids: List[int], rel_set: set, k: int = 10) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x in rel_set) / len(top)


def _compute_ndcg_at_k(ranked_ids: List[int], rel_set: set, k: int = 10) -> float:
    dcg = sum((1.0 / np.log2(i + 2)) for i, x in enumerate(ranked_ids[:k]) if x in rel_set)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DINOv2 REFERENCE BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dinov2_refs(dino_db: np.ndarray, rel_sets: List[set],
                       n_refs: int, ref_strategy: str) -> np.ndarray:
    """Build per-class DINOv2 reference vectors. Same logic as VDMS version."""
    refs = []
    for rel in rel_sets:
        ids = sorted(rel)
        if not ids:
            refs.append(np.zeros(dino_db.shape[1], dtype=np.float32))
            continue
        vecs = dino_db[ids]
        if n_refs == 1 or len(ids) == 1:
            ref = vecs[0]
        elif ref_strategy == "first":
            ref = vecs[:n_refs].mean(axis=0)
        elif ref_strategy == "centroid":
            centroid = vecs.mean(axis=0)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            closest = np.argsort(dists)[:n_refs]
            ref = vecs[closest].mean(axis=0)
        else:  # "diverse"
            centroid = vecs.mean(axis=0)
            dists = np.linalg.norm(vecs - centroid, axis=1)
            selected = [int(np.argmin(dists))]
            pool = set(range(len(vecs))) - {selected[0]}
            while len(selected) < min(n_refs, len(ids)) and pool:
                selected_vecs = vecs[selected]
                d2 = np.min(np.linalg.norm(
                    vecs[list(pool)][:, None] - selected_vecs[None], axis=2), axis=1)
                farthest = list(pool)[int(np.argmax(d2))]
                selected.append(farthest)
                pool.discard(farthest)
            ref = vecs[selected].mean(axis=0)
        norm = np.linalg.norm(ref)
        refs.append(ref / norm if norm > 1e-8 else ref)
    return np.stack(refs, axis=0)  # (600, 1024)


# ═══════════════════════════════════════════════════════════════════════════════
# FAISS INDEX + SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

_FAISS_INDEX = None
_FAISS_CLIP_DB: Optional[np.ndarray] = None
_FAISS_CURRENT_ENGINE: str = ""
_FAISS_CURRENT_PARAMS: Dict = {}  # build-time params only (M/efConstruction/nlist)


def _build_faiss_index(clip_db: np.ndarray, engine: str, params: Dict) -> float:
    """Build faiss index. Returns build time (s)."""
    global _FAISS_INDEX, _FAISS_CLIP_DB, _FAISS_CURRENT_ENGINE, _FAISS_CURRENT_PARAMS
    d = clip_db.shape[1]
    t0 = time.time()

    if engine == "FaissFlat":
        idx = faiss.IndexFlatIP(d)
        idx.add(clip_db)

    elif engine == "FaissHNSWFlat":
        M = params.get("M", 32)
        ef_constr = params.get("efConstruction", 200)
        idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = ef_constr
        idx.add(clip_db)

    else:  # FaissIVFFlat
        nlist = params.get("nlist", 128)
        quantizer = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(clip_db)
        idx.add(clip_db)

    build_time = time.time() - t0
    _FAISS_INDEX = idx
    _FAISS_CLIP_DB = clip_db
    _FAISS_CURRENT_ENGINE = engine
    _FAISS_CURRENT_PARAMS = dict(params)
    return build_time


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def _search_faiss(query_vecs: np.ndarray, k: int, engine: str, params: Dict,
                  constraint: str, query_obj_ids: List[int],
                  query_verb_ids: List[int],
                  obj_img_map: Dict[int, np.ndarray],
                  verb_img_map: Dict[int, np.ndarray]) -> Tuple[List[List[int]], List[List[float]], float]:
    """
    Search faiss index for all queries. Returns (ranked_ids, scores, avg_latency_ms).
    For constrained strategies, numpy exact search on the filtered subset
    (faster than full ANN for small pools of ~200-600 images).
    """
    global _FAISS_INDEX, _FAISS_CLIP_DB

    # Set query-time parameters (no rebuild needed)
    if engine == "FaissHNSWFlat":
        _FAISS_INDEX.hnsw.efSearch = params.get("efSearch", 64)
    elif engine == "FaissIVFFlat":
        _FAISS_INDEX.nprobe = params.get("nprobe", 8)

    all_ids: List[List[int]] = []
    all_scores: List[List[float]] = []
    t0 = time.time()

    for qi in range(len(query_vecs)):
        q = query_vecs[qi:qi+1]  # shape (1, d) for faiss

        if constraint == "none":
            D, I = _FAISS_INDEX.search(q, k)
            ids    = [int(x) for x in I[0] if x >= 0]
            scores = [float(x) for x in D[0][:len(ids)]]

        else:
            # Pre-filter to matching images, then numpy exact dot product on subset.
            # Subset sizes: object ~600, verb ~400, object_and_verb <200 images.
            # Numpy is fast for these small pools and gives exact recall within the filter.
            if constraint == "object":
                obj_id = query_obj_ids[qi]
                subset_ids = obj_img_map.get(obj_id, np.empty(0, dtype=np.int64))
            elif constraint == "verb":
                verb_id = query_verb_ids[qi]
                subset_ids = verb_img_map.get(verb_id, np.empty(0, dtype=np.int64))
            else:  # object_and_verb
                obj_id  = query_obj_ids[qi]
                verb_id = query_verb_ids[qi]
                obj_set  = set(obj_img_map.get(obj_id,  np.empty(0, dtype=np.int64)).tolist())
                verb_set = set(verb_img_map.get(verb_id, np.empty(0, dtype=np.int64)).tolist())
                subset_ids = np.array(sorted(obj_set & verb_set), dtype=np.int64)

            if len(subset_ids) == 0:
                # Fallback: unconstrained search
                D, I = _FAISS_INDEX.search(q, k)
                ids    = [int(x) for x in I[0] if x >= 0]
                scores = [float(x) for x in D[0][:len(ids)]]
            else:
                # Numpy exact IP on subset
                subset_vecs = _FAISS_CLIP_DB[subset_ids]  # (pool, d)
                sims = subset_vecs @ q[0]                  # (pool,)
                actual_k = min(k, len(subset_ids))
                top_local = np.argsort(-sims)[:actual_k]
                ids    = subset_ids[top_local].tolist()
                scores = sims[top_local].tolist()

        all_ids.append(ids)
        all_scores.append(scores)

    elapsed_ms = (time.time() - t0) * 1000.0
    avg_ms = elapsed_ms / max(len(query_vecs), 1)
    return all_ids, all_scores, avg_ms


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(config: Dict, dataset_dir: str, iteration: int = 0) -> BenchmarkResult:
    data        = _load_data(dataset_dir)
    clip_db     = data["clip_db"]
    dino_db     = data["dino_db"]
    text_q      = data["text_q"]
    rel_sets    = data["rel_sets"]
    junk_sets   = data["junk_sets"]
    hoi_obj_ids      = data["hoi_obj_ids"]
    hoi_verb_ids     = data["hoi_verb_ids"]
    obj_img_map      = data["obj_img_map"]
    verb_img_map     = data["verb_img_map"]

    engine     = config.get("engine", "FaissHNSWFlat")
    params     = config.get("params", {})
    k          = config.get("k_neighbors", 100)
    alpha      = config.get("alpha", 0.30)
    n_refs     = config.get("n_refs", 1)
    ref_strat  = config.get("ref_strategy", "first")
    constraint = config.get("constraint_strategy", "none")

    # Rebuild faiss index only when engine or index-build params change.
    # efSearch/nprobe are query-time params — they never trigger a rebuild.
    global _FAISS_CURRENT_ENGINE, _FAISS_CURRENT_PARAMS

    def _build_key(eng, p):
        if eng == "FaissHNSWFlat":
            return (eng, p.get("M", 32), p.get("efConstruction", 200))
        elif eng == "FaissIVFFlat":
            return (eng, p.get("nlist", 128))
        return (eng,)

    needs_rebuild = (_build_key(engine, params) !=
                     _build_key(_FAISS_CURRENT_ENGINE, _FAISS_CURRENT_PARAMS))
    if needs_rebuild:
        print(f"[FAISS] Building index: {engine} params={params}…")
        build_s = _build_faiss_index(clip_db, engine, params)
        print(f"[FAISS] Index built in {build_s:.1f}s")
    else:
        build_s = 0.0

    # Build DINOv2 reference vectors
    dino_refs = _build_dinov2_refs(dino_db, rel_sets, n_refs, ref_strat)

    # Stage 1: FAISS ANN search (all 600 queries)
    t_stage1 = time.time()
    all_ids, all_scores, avg_clip_ms = _search_faiss(
        text_q, k, engine, params, constraint,
        hoi_obj_ids, hoi_verb_ids, obj_img_map, verb_img_map)
    stage1_s = time.time() - t_stage1

    # Stage 2: DINOv2 reranking (numpy, same as VDMS version)
    all_ranked: List[List[int]] = []
    t_stage2 = time.time()
    for qi in range(len(text_q)):
        cand_ids = all_ids[qi]
        if not cand_ids:
            all_ranked.append([])
            continue

        cand_ids_arr = np.array(cand_ids)
        # FAISS IP scores: higher = better
        clip_sims = np.array(all_scores[qi], dtype=np.float32)

        if alpha > 0.0:
            dino_cands = dino_db[cand_ids_arr]      # (k, 1024)
            dino_sims  = dino_cands @ dino_refs[qi] # (k,)

            # Normalize to [0,1] (higher=better), then convert to distance form
            clip_norm = _minmax_norm(clip_sims)
            dino_norm = _minmax_norm(dino_sims)

            # Lower final_score = better (distance form, matches VDMS fusion formula)
            final = (1 - alpha) * (1 - clip_norm) + alpha * (1 - dino_norm)
            order = np.argsort(final)
        else:
            order = np.argsort(-clip_sims)  # higher sim = better

        all_ranked.append(cand_ids_arr[order].tolist())

    stage2_s = time.time() - t_stage2
    avg_rerank_ms = stage2_s * 1000.0 / max(len(text_q), 1)

    # Evaluate
    map_score, per_query_ap = _compute_map(all_ranked, rel_sets, junk_sets)
    p10  = np.mean([_compute_precision_at_k(r, rel_sets[i]) for i, r in enumerate(all_ranked)])
    ndcg = np.mean([_compute_ndcg_at_k(r, rel_sets[i]) for i, r in enumerate(all_ranked)])
    r10  = np.mean([
        sum(1 for x in r[:10] if x in rel_sets[i]) / max(len(rel_sets[i]), 1)
        for i, r in enumerate(all_ranked)
    ])

    total_query_time_s = stage1_s + stage2_s
    n_queries = len(text_q)
    qps = n_queries / total_query_time_s if total_query_time_s > 0 else 0.0
    avg_latency_ms = total_query_time_s * 1000.0 / n_queries

    return BenchmarkResult(
        qps=qps, map_score=map_score, latency_ms=avg_latency_ms,
        index_build_s=build_s, precision_at_10=float(p10),
        ndcg_at_10=float(ndcg), recall_at_10=float(r10),
        engine=engine, params=dict(params),
        k_neighbors=k, alpha=alpha, timestamp=time.time(),
        per_query_ap=per_query_ap,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigProvider:
    ENGINES = ["FaissFlat", "FaissHNSWFlat", "FaissIVFFlat"]

    ENGINE_PARAMS = {
        "FaissFlat": {},
        "FaissHNSWFlat": {
            "M":              [8, 16, 32, 48, 64],
            "efConstruction": [200],
            "efSearch":       [32, 64, 100, 150, 200, 300, 500],
        },
        "FaissIVFFlat": {
            "nlist":  [64, 128, 256, 512],
            "nprobe": [4, 8, 16, 32, 64],
        },
    }

    K_NEIGHBORS_VALUES  = [50, 100, 150, 200, 300, 500, 750, 1000]
    ALPHA_VALUES        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                           0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    N_REFS_VALUES       = [1, 3, 5, 10]
    REF_STRATEGY_VALUES = ["first", "centroid", "diverse"]
    CONSTRAINT_STRATEGIES = ["none", "object", "verb", "object_and_verb"]

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def get_random_config(self) -> Dict:
        engine = self.rng.choice(self.ENGINES)
        params = {k: int(self.rng.choice(v))
                  for k, v in self.ENGINE_PARAMS[engine].items()}
        return {
            "engine":              engine,
            "params":              params,
            "k_neighbors":         int(self.rng.choice(self.K_NEIGHBORS_VALUES)),
            "alpha":               float(self.rng.choice(self.ALPHA_VALUES)),
            "n_refs":              int(self.rng.choice(self.N_REFS_VALUES)),
            "ref_strategy":        str(self.rng.choice(self.REF_STRATEGY_VALUES)),
            "constraint_strategy": str(self.rng.choice(self.CONSTRAINT_STRATEGIES)),
        }

    def grid_search_configs(self) -> List[Dict]:
        configs = []

        def _gc(**kw):
            kw.setdefault("constraint_strategy", "none")
            kw.setdefault("n_refs", 1)
            kw.setdefault("ref_strategy", "first")
            return kw

        # S1: Engine/M baselines (5) k=200, alpha=0.30
        configs.append(_gc(engine="FaissFlat", params={}, k_neighbors=200, alpha=0.30))
        configs.append(_gc(engine="FaissIVFFlat", params={"nlist": 128, "nprobe": 8},
                           k_neighbors=200, alpha=0.30))
        for M in [16, 32, 64]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": M, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=200, alpha=0.30))

        # S2: k sweep (5) HNSW M=32 efS=200, alpha=0.30
        for k in [50, 100, 150, 500, 1000]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=k, alpha=0.30))

        # S3: alpha sweep (5) HNSW M=32 efS=200, k=100
        for a in [0.10, 0.30, 0.50, 0.70, 0.90]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=a))

        # S4: n_refs sweep (5)
        for nr in [1, 3, 5, 10]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=0.30, n_refs=nr))
        configs.append(_gc(engine="FaissHNSWFlat",
                           params={"M": 32, "efConstruction": 200, "efSearch": 200},
                           k_neighbors=100, alpha=0.30, n_refs=3, ref_strategy="centroid"))

        # S5: ref_strategy sweep (3)
        for rs in ["first", "centroid", "diverse"]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": 200},
                               k_neighbors=100, alpha=0.30, n_refs=3, ref_strategy=rs))

        # S6: efSearch sweep (4)
        for efs in [32, 64, 100, 150]:
            configs.append(_gc(engine="FaissHNSWFlat",
                               params={"M": 32, "efConstruction": 200, "efSearch": efs},
                               k_neighbors=100, alpha=0.30))

        # S7: IVFFlat nprobe sweep (3)
        for nprobe in [4, 16, 64]:
            configs.append(_gc(engine="FaissIVFFlat",
                               params={"nlist": 128, "nprobe": nprobe},
                               k_neighbors=100, alpha=0.30))

        return configs[:50]  # cap at 50 iterations

    def optuna_suggest(self, trial) -> Dict:
        engine = trial.suggest_categorical("engine", self.ENGINES)
        params = {}
        if engine == "FaissHNSWFlat":
            params = {
                "M":              trial.suggest_categorical("M", self.ENGINE_PARAMS["FaissHNSWFlat"]["M"]),
                "efConstruction": 200,
                "efSearch":       trial.suggest_categorical("efSearch", self.ENGINE_PARAMS["FaissHNSWFlat"]["efSearch"]),
            }
        elif engine == "FaissIVFFlat":
            params = {
                "nlist":  trial.suggest_categorical("nlist",  self.ENGINE_PARAMS["FaissIVFFlat"]["nlist"]),
                "nprobe": trial.suggest_categorical("nprobe", self.ENGINE_PARAMS["FaissIVFFlat"]["nprobe"]),
            }
        return {
            "engine":              engine,
            "params":              params,
            "k_neighbors":         trial.suggest_categorical("k", self.K_NEIGHBORS_VALUES),
            "alpha":               trial.suggest_categorical("alpha", self.ALPHA_VALUES),
            "n_refs":              trial.suggest_categorical("n_refs", self.N_REFS_VALUES),
            "ref_strategy":        trial.suggest_categorical("ref_strategy", self.REF_STRATEGY_VALUES),
            "constraint_strategy": trial.suggest_categorical("constraint", self.CONSTRAINT_STRATEGIES),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LLM AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class LLMAgent:
    def __init__(self, api_key: str, base_url: str, model: str,
                 t_exp_frac: float = 0.30, t_expl_frac: float = 0.70):
        self.api_key    = api_key
        self.base_url   = base_url
        self.model      = model
        self.t_exp_frac = t_exp_frac
        self.t_expl_frac = t_expl_frac

    async def query_llm(self, iteration: int, history: List[IterationResult],
                        total_iterations: int) -> Tuple[Dict, str]:
        import aiohttp

        exp_end  = max(1, int(round(total_iterations * self.t_exp_frac)))
        expl_end = max(exp_end + 1, int(round(total_iterations * self.t_expl_frac)))
        phase = ("EXPLORATION" if iteration <= exp_end else
                 "EXPLOITATION" if iteration <= expl_end else "FINE-TUNING")

        current_best = max((r.benchmark.score() for r in history), default=0.0)

        # Guidance based on last run
        if history:
            last = history[-1]
            lm, lq, ls = last.benchmark.map_score, last.benchmark.qps, last.benchmark.score()
            if current_best == 0.0 and ls == 0.0:
                guidance = (f"INFEASIBLE. mAP={lm:.4f} < τ={_MAP_THRESHOLD:.3f} → Score=0. "
                            "Increase k_neighbors (k=100→mAP~0.20) or alpha (0.50-0.75). "
                            "Use constraint_strategy=object + n_refs=10 + alpha=0.65.")
            elif ls >= current_best * 0.98 and phase == "FINE-TUNING":
                guidance = (f"EXCELLENT. Score={ls:.4f}. Fine-tune: change ONE param by ONE step.")
            elif lm >= 0.20 and lq < 100:
                guidance = (f"mAP OK ({lm:.4f}) but QPS low ({lq:.1f}). "
                            "Reduce k_neighbors AND efSearch — both are QPS levers in FAISS. "
                            "k=100+efSearch=64 → high QPS. k=50+efSearch=32 → very high QPS.")
            else:
                guidance = (f"Score={ls:.4f}. mAP={lm:.4f} QPS={lq:.1f}. "
                            "Explore different (engine, k, efSearch, alpha) combinations.")
        else:
            guidance = ("First iteration. Start: FaissHNSWFlat M=32 efSearch=64 "
                        "k=100 alpha=0.50 n_refs=3 ref_strategy=centroid constraint=object. "
                        "FAISS is much faster than VDMS — expect QPS in hundreds to thousands.")

        all_tried = "\n".join(
            f"  iter{r.config.get('_iter','?'):02d}: {r.config['engine']} "
            f"p={r.config.get('params',{})} k={r.config.get('k_neighbors',500)} "
            f"a={r.config.get('alpha',0.0):.2f} refs={r.config.get('n_refs',1)} "
            f"cs={r.config.get('constraint_strategy','none')} "
            f"→ Score={r.benchmark.score():.4f} mAP={r.benchmark.map_score:.4f} QPS={r.benchmark.qps:.1f}"
            for r in history
        ) or "  (none yet)"

        hist_json = json.dumps([{
            "eng": h.config["engine"], "p": h.config.get("params", {}),
            "k": h.config.get("k_neighbors", 500),
            "a": round(h.config.get("alpha", 0.0), 2),
            "refs": h.config.get("n_refs", 1),
            "rs": h.config.get("ref_strategy", "first"),
            "cs": h.config.get("constraint_strategy", "none"),
            "sc": round(h.benchmark.score(), 4),
            "map": round(h.benchmark.map_score, 4),
            "qps": round(h.benchmark.qps, 1),
        } for h in history[-5:]], indent=1)

        prompt = f"""You are an expert FAISS Optimization Engine for HOI Retrieval.
Iteration: {iteration}/{total_iterations}. Phase: {phase}.

SYSTEM: FAISS (faiss-cpu, in-process) with CLIP ViT-L/14 (768-d, IP metric) + DINOv2 reranking.
Dataset: HICO-DET, 47,776 images, 600 HOI query classes.
Note: FAISS is the ANN engine underlying Milvus. Results directly reflect core ANN performance.

OBJECTIVE: maximize QPS subject to mAP ≥ τ={_MAP_THRESHOLD:.3f} (SIEVE-style).
  Score = QPS if mAP ≥ τ else 0. Only configs above the mAP threshold contribute.

TWO-STAGE PIPELINE:
  Stage 1: FAISS ANN search — retrieve top-k candidates per query using CLIP text embedding.
  Stage 2: DINOv2 reranking — fuse CLIP and DINOv2 similarity scores.
    fusion: final_score = (1-alpha)*(1-clip_norm) + alpha*(1-dino_norm)  [lower=better]

QPS PHYSICS (FAISS — critical differences from VDMS):
  FAISS has NO server-client overhead, NO PMGD journal writes. Expect QPS in hundreds-thousands.

  PRIMARY QPS LEVERS (tune these to control QPS):
    k_neighbors:   Fewer candidates → faster Stage-1 ANN AND faster Stage-2 rerank.
                   k=50→very high QPS, k=100→high, k=300+→slow. Halving k ~doubles QPS.
    efSearch:      HNSW query-time parameter. Real QPS lever in FAISS.
                   Lower efSearch → fewer graph hops → faster search, lower CLIP recall.
                   efSearch=32→fastest (~1800+ QPS), efSearch=200→~673 QPS (measured at k=100).
                   efSearch and k are INDEPENDENT levers — tune both.
    nprobe (IVF):  Like efSearch — lower nprobe → faster search, lower recall.
                   nprobe=8/nlist=256 → ~3000 QPS (measured). nprobe=32 → ~846 QPS.
    constraint:    Pre-filtering to ~600 images (object) via numpy exact search.
                   Strong QPS gain AND better precision for HOI retrieval.

  FREE PARAMETERS (no QPS cost):
    n_refs, ref_strategy: DINOv2 reference building — pure numpy, negligible time.
    alpha: DINOv2 fusion weight — only affects Stage-2 quality, not ANN speed.
    M: Higher M = better recall ceiling, NO query-time QPS effect.

  INDEX REBUILD NOTE: Only M or efConstruction changes require rebuilding the index.
    Build times: FaissHNSWFlat M=32 ~26s, FaissIVFFlat nlist=128 ~3s, FaissFlat ~0.5s.
    Changing efSearch, nprobe, k, alpha, n_refs, constraint → NO rebuild → fast iteration.

MEASURED QPS BASELINES (on this machine, k=100, no DINOv2, no constraint):
  FaissHNSWFlat M=32 efSearch=32:  ~2400 QPS
  FaissHNSWFlat M=32 efSearch=64:  ~1800 QPS
  FaissHNSWFlat M=32 efSearch=200: ~673 QPS
  FaissIVFFlat nlist=128 nprobe=8:  ~1525 QPS
  FaissIVFFlat nlist=256 nprobe=8:  ~3008 QPS
  FaissIVFFlat nlist=256 nprobe=32: ~846 QPS
  DINOv2 reranking (k=100): adds ~0.5ms/query → ~1700 QPS overhead cap

PARAMETER DEFINITIONS:
  [A1] engine: FaissFlat | FaissHNSWFlat | FaissIVFFlat
    FaissFlat:     Exact brute-force. Highest possible recall at any k. No ANN params.
                   QPS ~500-2000 (limited by k). Use as mAP ceiling reference.
    FaissHNSWFlat: Graph-based ANN. Tune M (build-time) + efSearch (query-time, real QPS lever).
    FaissIVFFlat:  Cluster-based ANN. Tune nlist (build-time) + nprobe (query-time, real QPS lever).
                   IVF with high nlist (256-512) + low nprobe → VERY high QPS.

  [A2] M (HNSW only): [8, 16, 32, 48, 64]
    Higher M = denser graph = higher recall ceiling. No query-time QPS effect.
    Start M=32. Try M=16 (faster build, slightly lower recall ceiling).

  [A3] efConstruction: FIXED at 200. Do not change.

  [A4] efSearch (HNSW only): [32, 64, 100, 150, 200, 300, 500]  ← REAL QPS LEVER
    Controls graph traversal depth. efSearch=32→fastest, efSearch=500→best recall.
    JOINT OPTIMIZATION with k: (k=50, efSearch=32) → very high QPS, probe mAP.
    For 47K vectors with M=32: efSearch=32-64 often sufficient for good recall.

  [A5] nlist (IVF only): [64, 128, 256, 512]
    More clusters = finer partitioning. nlist=256 with low nprobe → highest QPS.
    Rule: nlist=sqrt(n)=218 ≈ 256 for n=47K. nprobe=nlist → exact (same as FaissFlat).

  [A6] nprobe (IVF only): [4, 8, 16, 32, 64]  ← REAL QPS LEVER
    QPS ∝ 1/nprobe. nlist=256/nprobe=8 → ~3000 QPS. nlist=256/nprobe=32 → ~846 QPS.

  [B1] k_neighbors: [50, 100, 150, 200, 300, 500, 750, 1000]  ← PRIMARY QPS LEVER
    Candidates retrieved from FAISS. Both Stage-1 (ANN) and Stage-2 (rerank) scale with k.
    Start k=100. Probe k=50 (high QPS, mAP borderline).
    With constraint=object (~600 images), k=150 is near-exhaustive.

  [B2] alpha: [0.0, 0.05, ..., 0.90]  (FREE — no QPS cost)
    DINOv2 weight. alpha=0 → pure CLIP ranking. alpha>0 → DINOv2 corrects CLIP ordering.
    Optimal alpha at small k (50-100) is UNKNOWN — discover empirically.
    Prior ablation (VDMS, k=500): alpha=0.30 best. At k=50-100, optimal may be higher.

  [B3] n_refs: [1, 3, 5, 10]  (FREE — pure numpy)
    DINOv2 reference images per HOI class. n_refs=3-10 → better reference.
    n_refs=3-5 is likely free improvement over n_refs=1.

  [B4] ref_strategy: ["first", "centroid", "diverse"]  (FREE)
    "centroid": picks images closest to class visual center → stable reference.
    "diverse":  greedy farthest-point sampling → covers intra-class variation.
    Prefer centroid at n_refs=3-5, diverse at n_refs=5-10.

  [B5] constraint_strategy: ["none", "object", "verb", "object_and_verb"]
    Pre-filter search to images matching HOI metadata:
      "none"   → full 47K search. Standard ANN behavior.
      "object" → ~600 images/query (1.2% of DB). Numpy exact search on subset.
      "verb"   → ~400 images/query. High risk of missing secondary annotations.
      "object_and_verb" → <200 images. Very small pool, high mAP risk.

CROSS-PARAMETER INTERACTIONS:
  k ↓ + efSearch ↓ → QPS ↑↑ (both levers compound in FAISS)
  k ↓             → mAP ↓ slightly (fewer candidates for reranking)
  efSearch ↓      → CLIP recall ↓ → mAP ↓ (compensate with alpha ↑ or k ↑)
  nlist=256 + nprobe=8 → best IVF QPS (3000+), worth trying
  constraint=obj  → pool 47K→600 → both faster ANN and better precision → QPS ↑↑
  alpha ↑         → mAP ↑ up to optimum, then ↓
  n_refs ↑        → better DINOv2 ref → safer to increase alpha

RECOMMENDED EXPLORATION STRATEGY:
  1. FaissHNSWFlat M=32 efSearch=64 k=100 alpha=0.50 n_refs=3 centroid constraint=object
  2. FaissIVFFlat nlist=256 nprobe=8 k=100 alpha=0.50 n_refs=3 centroid constraint=object
  3. FaissHNSWFlat M=32 efSearch=32 k=50 alpha=0.60 n_refs=5 → max QPS test
  4. FaissFlat k=100 alpha=0.50 n_refs=3 centroid → mAP ceiling reference
  5. Vary efSearch and nprobe systematically to build the QPS-recall tradeoff curve

PHASE ({phase}):
  EXPLORATION (1-{exp_end}): Diverse engine/k/efSearch/nprobe/alpha combos.
    Priority: measure QPS at efSearch={{32,64,200}} and nprobe={{4,8,32}} — key data points.
  EXPLOITATION ({exp_end+1}-{expl_end}): Narrow around best zone. Jointly tune k+efSearch+alpha.
  FINE-TUNING ({expl_end+1}-{total_iterations}): Change ONE param by ONE step.

DIAGNOSIS: {guidance}

ALL CONFIGS TRIED (do NOT repeat exact combinations):
{all_tried}

LAST 5 (detailed):
{hist_json}

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "engine": "FaissHNSWFlat",
  "params": {{"M": 32, "efConstruction": 200, "efSearch": 64}},
  "k_neighbors": 100,
  "alpha": 0.50,
  "n_refs": 3,
  "ref_strategy": "centroid",
  "constraint_strategy": "object",
  "reasoning": "one sentence explaining the choice"
}}"""

        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 400, "temperature": 0.3}

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                    ) as resp:
                        data = await resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Strip markdown code fences if present
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
                cfg = json.loads(content)
                reasoning = cfg.pop("reasoning", "")
                return cfg, reasoning
            except Exception as e:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("LLM query failed after 3 attempts")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG SNAPPING (keep values in allowed lists)
# ═══════════════════════════════════════════════════════════════════════════════

def _snap_config(cfg: Dict) -> Dict:
    out = dict(cfg)
    for f in ["engine", "params", "k_neighbors", "alpha", "n_refs",
              "ref_strategy", "constraint_strategy"]:
        if f not in out:
            defaults = {"engine": "FaissHNSWFlat", "params": {},
                        "k_neighbors": 100, "alpha": 0.30, "n_refs": 1,
                        "ref_strategy": "first", "constraint_strategy": "none"}
            out[f] = defaults[f]

    if out["engine"] not in ConfigProvider.ENGINES:
        out["engine"] = "FaissHNSWFlat"

    def _nearest(val, choices):
        return min(choices, key=lambda x: abs(x - val))

    out["k_neighbors"] = _nearest(out.get("k_neighbors", 100), ConfigProvider.K_NEIGHBORS_VALUES)
    out["alpha"]       = _nearest(out.get("alpha", 0.30),       ConfigProvider.ALPHA_VALUES)
    out["n_refs"]      = _nearest(out.get("n_refs", 1),         ConfigProvider.N_REFS_VALUES)

    if out.get("ref_strategy") not in ConfigProvider.REF_STRATEGY_VALUES:
        out["ref_strategy"] = "first"
    if out.get("constraint_strategy") not in ConfigProvider.CONSTRAINT_STRATEGIES:
        out["constraint_strategy"] = "none"

    engine = out["engine"]
    params = dict(out.get("params", {}))
    if engine == "FaissHNSWFlat":
        allowed = ConfigProvider.ENGINE_PARAMS["FaissHNSWFlat"]
        params["M"]              = _nearest(params.get("M", 32),              allowed["M"])
        params["efConstruction"] = 200
        params["efSearch"]       = _nearest(params.get("efSearch", 200),      allowed["efSearch"])
    elif engine == "FaissIVFFlat":
        allowed = ConfigProvider.ENGINE_PARAMS["FaissIVFFlat"]
        params["nlist"]  = _nearest(params.get("nlist", 128),  allowed["nlist"])
        params["nprobe"] = _nearest(params.get("nprobe", 8),   allowed["nprobe"])
    else:
        params = {}
    out["params"] = params
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZER LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def _config_key(cfg: Dict) -> str:
    p = json.dumps(cfg.get("params", {}), sort_keys=True)
    return (f"{cfg.get('engine')}|{p}|{cfg.get('k_neighbors')}|"
            f"{cfg.get('alpha')}|{cfg.get('n_refs')}|"
            f"{cfg.get('ref_strategy')}|{cfg.get('constraint_strategy')}")


async def run_optimizer(args) -> List[IterationResult]:
    global _MAP_THRESHOLD
    _MAP_THRESHOLD = args.map_threshold

    api_key  = os.getenv("OPENROUTER_API_KEY") or os.getenv("MINIMAX_API_KEY") or ""
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model    = os.getenv("LLM_MODEL", "minimax/minimax-m2.1")

    provider = ConfigProvider(seed=args.seed)
    results: List[IterationResult] = []
    seen: set = set()

    if args.method == "hyperparameter_only":
        agent = LLMAgent(api_key, base_url, model,
                         t_exp_frac=args.t_exp_frac, t_expl_frac=args.t_expl_frac)

    if args.method == "hyperparameter_only":
        for i in range(1, args.iterations + 1):
            cfg, reasoning = await agent.query_llm(i, results, args.iterations)
            cfg = _snap_config(cfg)
            cfg["_iter"] = i

            key = _config_key(cfg)
            retries = 0
            while key in seen and retries < 5:
                cfg2, reasoning2 = await agent.query_llm(i, results, args.iterations)
                cfg2 = _snap_config(cfg2)
                cfg2["_iter"] = i
                k2 = _config_key(cfg2)
                if k2 not in seen:
                    cfg, key, reasoning = cfg2, k2, reasoning2
                    break
                retries += 1
            seen.add(key)

            print(f"\n[Iter {i}/{args.iterations}] LLM → {cfg}  | {reasoning}")
            bm = run_benchmark(cfg, args.dataset_dir, iteration=i)
            print(f"  Score={bm.score():.4f}  mAP={bm.map_score:.4f}  "
                  f"QPS={bm.qps:.1f}  build={bm.index_build_s:.1f}s")
            results.append(IterationResult(
                iteration=i, config=cfg, benchmark=bm,
                llm_reasoning=reasoning, search_method="LLM"))

    elif args.method == "grid":
        configs = provider.grid_search_configs()[:args.iterations]
        for i, cfg in enumerate(configs, 1):
            cfg["_iter"] = i
            print(f"\n[Iter {i}/{args.iterations}] Grid → {cfg}")
            bm = run_benchmark(cfg, args.dataset_dir, iteration=i)
            print(f"  Score={bm.score():.4f}  mAP={bm.map_score:.4f}  QPS={bm.qps:.1f}")
            results.append(IterationResult(
                iteration=i, config=cfg, benchmark=bm,
                llm_reasoning="", search_method="grid"))

    elif args.method == "random":
        for i in range(1, args.iterations + 1):
            cfg = provider.get_random_config()
            cfg["_iter"] = i
            key = _config_key(cfg)
            retries = 0
            while key in seen and retries < 20:
                cfg = provider.get_random_config()
                cfg["_iter"] = i
                key = _config_key(cfg)
                retries += 1
            seen.add(key)
            print(f"\n[Iter {i}/{args.iterations}] Random → {cfg}")
            bm = run_benchmark(cfg, args.dataset_dir, iteration=i)
            print(f"  Score={bm.score():.4f}  mAP={bm.map_score:.4f}  QPS={bm.qps:.1f}")
            results.append(IterationResult(
                iteration=i, config=cfg, benchmark=bm,
                llm_reasoning="", search_method="random"))

    elif args.method == "optuna":
        dataset_dir = args.dataset_dir
        _res_list = results

        def _objective(trial):
            cfg = provider.optuna_suggest(trial)
            cfg["_iter"] = trial.number + 1
            bm = run_benchmark(cfg, dataset_dir, iteration=trial.number + 1)
            _res_list.append(IterationResult(
                iteration=trial.number + 1, config=cfg, benchmark=bm,
                llm_reasoning="", search_method="optuna"))
            print(f"  [Optuna trial {trial.number+1}] Score={bm.score():.4f}  "
                  f"mAP={bm.map_score:.4f}  QPS={bm.qps:.1f}")
            return bm.score()

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(_objective, n_trials=args.iterations)

    return results


def main():
    parser = argparse.ArgumentParser(description="FAISS HICO-DET Optimizer (Milvus backend)")
    parser.add_argument("--dataset-dir",    type=str, required=True)
    parser.add_argument("--method",         type=str, default="hyperparameter_only",
                        choices=["hyperparameter_only", "grid", "random", "optuna"])
    parser.add_argument("--iterations",     type=int, default=50)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--map-threshold",  type=float, default=0.15)
    parser.add_argument("--output",         type=str, default="milvus_hico_results.json")
    parser.add_argument("--t-exp-frac",     type=float, default=0.30)
    parser.add_argument("--t-expl-frac",    type=float, default=0.70)
    args = parser.parse_args()

    print("=" * 70)
    print(f"FAISS HICO-DET Optimizer (Milvus backend)")
    print(f"  Method:     {args.method}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Seed:       {args.seed}")
    print(f"  τ (mAP):    {args.map_threshold}")
    print("=" * 70)

    results = asyncio.run(run_optimizer(args))

    if not results:
        print("No results collected.")
        return

    best = max(results, key=lambda r: r.benchmark.score())
    print(f"\n{'='*70}")
    print(f"BEST: Score={best.benchmark.score():.4f}  mAP={best.benchmark.map_score:.4f}  "
          f"QPS={best.benchmark.qps:.1f}")
    print(f"  Config: {best.config}")

    out = {
        "method":        args.method,
        "seed":          args.seed,
        "iterations":    args.iterations,
        "map_threshold": args.map_threshold,
        "best_score":    best.benchmark.score(),
        "best_map":      best.benchmark.map_score,
        "best_qps":      best.benchmark.qps,
        "best_config":   best.config,
        "all_results":   [
            {
                "iteration":     r.iteration,
                "config":        r.config,
                "score":         r.benchmark.score(),
                "map":           r.benchmark.map_score,
                "qps":           r.benchmark.qps,
                "latency_ms":    r.benchmark.latency_ms,
                "index_build_s": r.benchmark.index_build_s,
                "p10":           r.benchmark.precision_at_10,
                "ndcg10":        r.benchmark.ndcg_at_10,
                "recall10":      r.benchmark.recall_at_10,
                "reasoning":     r.llm_reasoning,
            }
            for r in results
        ],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
