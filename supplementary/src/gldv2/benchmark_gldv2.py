"""
GLDv2 Benchmark — CLIP ViT-L/14 + DINOv2 ViT-L/14-reg4 via VDMS
=================================================================

Two-stage image-to-image retrieval pipeline:
  Stage 1: VDMS ANN search with CLIP ViT-L/14 image embeddings
  Stage 2: DINOv2 ViT-L/14-reg4 weighted-sum reranking

Pseudo-Relevance Feedback (n_refs > 1):
  When n_refs=1: use query image DINOv2 feature directly (standard).
  When n_refs > 1: augment the query DINOv2 vector by averaging it with
  the DINOv2 features of the top-(n_refs-1) Stage-1 CLIP candidates.
  ref_strategy controls HOW those candidates are selected:
    "first"    — top-(n_refs-1) by CLIP distance (default, fast)
    "centroid" — centroid of ALL Stage-1 candidates' DINOv2 features
    "diverse"  — greedy farthest-point subset of Stage-1 DINOv2 features

Score: QPS if mAP ≥ τ else 0  (SIEVE-style constrained, identical structure to HICO-DET)

Required files in <dataset_dir>/gldv2/:
  gldv2_clip.npy           [N, 768]   float32  DB CLIP features (L2-normed)
  gldv2_dinov2.npy         [N, 1024]  float32  DB DINOv2 features (L2-normed)
  gldv2_ids.json           [N]        str      DB image IDs in row order
  gldv2_gt.json            dict       {query_id: {"positive": [ids], "junk": [ids]}}
  gldv2_query_ids.json     [Q]        str      Query IDs in row order
  gldv2_query_clip.npy     [Q, 768]   float32  Query CLIP features (L2-normed)
  gldv2_query_dinov2.npy   [Q, 1024]  float32  Query DINOv2 features (L2-normed)
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import vdms

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_PROPS_SIZE = 500  # batch_properties mode: N vectors per AddDescriptor call.
                        # Uses a single insert_descriptor(blob, path, N) FAISS call.
                        # PMGD ops = 1 QueryNode + 2N (AddNode+AddEdge) vs 3N individual.
                        # Safe at N=500: ~1001 PMGD ops, well within 2MB journal limit.

# Pre-compute IDCG denominators for k=1..200
_IDCG_CACHE = {k: sum(1.0 / math.log2(i + 2) for i in range(k))
               for k in range(1, 201)}


# ── Data cache ────────────────────────────────────────────────────────────────
_GLDV2_DATA_CACHE: Dict = {}


def load_gldv2_data(dataset_dir: str) -> Dict:
    """
    Load all GLDv2 features and ground truth into memory.

    Builds id_to_idx mapping and pre-converts GT string IDs to row indices
    so per-query metric computation is O(1) dict lookup.

    Returns dict with keys:
      clip_db      [N, 768]   float32  — DB CLIP features
      dinov2_db    [N, 1024]  float32  — DB DINOv2 features
      clip_q       [Q, 768]   float32  — Query CLIP features
      dinov2_q     [Q, 1024]  float32  — Query DINOv2 features
      db_ids       list[str]           — DB image IDs (row order)
      query_ids    list[str]           — Query image IDs (row order)
      id_to_idx    dict[str, int]      — DB image ID → row index
      rel_sets     list[set[int]]      — per-query positive row-index sets
      junk_sets    list[set[int]]      — per-query junk row-index sets
    """
    global _GLDV2_DATA_CACHE
    if _GLDV2_DATA_CACHE:
        return _GLDV2_DATA_CACHE

    d = Path(dataset_dir) / "gldv2"

    print("[GLDv2] Loading features and ground truth ...")

    clip_db   = np.load(str(d / "gldv2_clip.npy")).astype(np.float32)
    dinov2_db = np.load(str(d / "gldv2_dinov2.npy")).astype(np.float32)
    clip_q    = np.load(str(d / "gldv2_query_clip.npy")).astype(np.float32)
    dinov2_q  = np.load(str(d / "gldv2_query_dinov2.npy")).astype(np.float32)

    with open(d / "gldv2_ids.json") as f:
        db_ids = json.load(f)
    with open(d / "gldv2_query_ids.json") as f:
        query_ids = json.load(f)
    with open(d / "gldv2_gt.json") as f:
        gt_raw = json.load(f)

    N = len(db_ids)
    Q = len(query_ids)
    assert clip_db.shape[0] == N,   f"clip_db rows {clip_db.shape[0]} != ids {N}"
    assert dinov2_db.shape[0] == N, f"dinov2_db rows {dinov2_db.shape[0]} != ids {N}"
    assert clip_q.shape[0] == Q,   f"clip_q rows {clip_q.shape[0]} != query_ids {Q}"
    assert dinov2_q.shape[0] == Q, f"dinov2_q rows {dinov2_q.shape[0]} != query_ids {Q}"

    # Build string-ID → row-index map
    id_to_idx: Dict[str, int] = {img_id: i for i, img_id in enumerate(db_ids)}

    # Convert GT string IDs → row indices
    rel_sets:  List[Set[int]] = []
    junk_sets: List[Set[int]] = []
    n_orphan_pos  = 0
    n_orphan_junk = 0
    for qid in query_ids:
        entry = gt_raw.get(qid, {"positive": [], "junk": []})
        pos_ids  = entry.get("positive", [])
        junk_ids = entry.get("junk", [])
        rel  = set()
        junk = set()
        for pid in pos_ids:
            idx = id_to_idx.get(pid)
            if idx is not None:
                rel.add(idx)
            else:
                n_orphan_pos += 1
        for jid in junk_ids:
            idx = id_to_idx.get(jid)
            if idx is not None:
                junk.add(idx)
            else:
                n_orphan_junk += 1
        rel_sets.append(rel)
        junk_sets.append(junk)

    pos_counts = [len(r) for r in rel_sets]
    n_queries_with_pos = sum(1 for c in pos_counts if c > 0)
    mean_pos = float(np.mean(pos_counts)) if pos_counts else 0.0
    max_r10  = min(1.0, 10.0 / mean_pos) * 100 if mean_pos > 0 else 0.0

    print(f"[OK] GLDv2 loaded: {N:,} DB images, {Q:,} queries")
    print(f"     DB   clip_db={clip_db.shape}  dinov2_db={dinov2_db.shape}")
    print(f"     Q    clip_q={clip_q.shape}  dinov2_q={dinov2_q.shape}")
    print(f"     GT:  {n_queries_with_pos}/{Q} queries with positives  "
          f"mean_pos={mean_pos:.1f}  max_R@10~{max_r10:.1f}%")
    if n_orphan_pos:
        print(f"     [WARN] {n_orphan_pos} positive IDs not in DB index — mAP underestimated")

    _GLDV2_DATA_CACHE = dict(
        clip_db=clip_db, dinov2_db=dinov2_db,
        clip_q=clip_q, dinov2_q=dinov2_q,
        db_ids=db_ids, query_ids=query_ids,
        id_to_idx=id_to_idx,
        rel_sets=rel_sets, junk_sets=junk_sets,
    )
    return _GLDV2_DATA_CACHE


# ── VDMS index setup ──────────────────────────────────────────────────────────

def setup_clip_index(db, clip_db: np.ndarray,
                     engine: str, engine_params: dict,
                     set_name: str = "CLIP_Set") -> float:
    """
    Create a CLIP DescriptorSet in VDMS and index all DB vectors.
    No PMGD metadata stored (GLDv2 has no per-image annotations).
    Returns elapsed indexing time in seconds.
    """
    clip_cfg: Dict = {
        "name":       set_name,
        "dimensions": int(clip_db.shape[1]),
        "metric":     "L2",
        "engine":     engine,
        "properties": {},
    }
    if engine == "FaissHNSWFlat":
        for k in ["M", "efConstruction", "efSearch"]:
            if k in engine_params:
                clip_cfg["properties"][k] = int(engine_params[k])
    elif engine == "FaissIVFFlat":
        for k in ["nlist", "nprobe"]:
            if k in engine_params:
                clip_cfg["properties"][k] = int(engine_params[k])

    res, _ = db.query([{"AddDescriptorSet": clip_cfg}])
    if not res or res[0].get("AddDescriptorSet", {}).get("status", -1) != 0:
        raise RuntimeError(f"DescriptorSet '{set_name}' creation failed: {res}")
    print(f"[OK] DescriptorSet '{set_name}' created  dim={clip_db.shape[1]}  "
          f"engine={engine}  metric=L2")

    n    = len(clip_db)
    t0   = time.time()
    fail = 0

    print(f"Indexing {n:,} CLIP vectors  set={set_name}  "
          f"(batch_properties={BATCH_PROPS_SIZE})...")
    for start in range(0, n, BATCH_PROPS_SIZE):
        end        = min(start + BATCH_PROPS_SIZE, n)
        batch_size = end - start

        # batch_properties: one AddDescriptor inserts batch_size vectors in a
        # single insert_descriptor(blob, path, batch_size) FAISS call.
        batch_props = [{"id": start + j} for j in range(batch_size)]
        blob        = clip_db[start:end].astype(np.float32).tobytes()

        cmd = {"AddDescriptor": {
            "set":              set_name,
            "batch_properties": batch_props,
        }}
        res, _ = db.query([cmd], [blob])
        if res:
            status = res[0].get("AddDescriptor", {}).get("status", -1)
            if status != 0:
                fail += 1
        if start % 50000 == 0 and start > 0:
            print(f"  progress: {start:,}/{n:,}")

    elapsed    = time.time() - t0
    n_batches  = (n + BATCH_PROPS_SIZE - 1) // BATCH_PROPS_SIZE
    print(f"  batch failures: {fail}/{n_batches}  ({'FAIL' if fail > 0 else 'OK'})")
    print(f"[OK] '{set_name}' indexed  ({elapsed:.1f}s)")
    if fail > 0:
        raise RuntimeError(f"CLIP indexing had {fail}/{n_batches} batch failures")

    return elapsed


# ── VDMS response parser ──────────────────────────────────────────────────────

def _parse_entities(find_res) -> List[Dict]:
    """Extract [{id: int, distance: float}, ...] from a FindDescriptor response."""
    if not find_res:
        return []
    fd = find_res.get("FindDescriptor", {})
    if fd.get("status", -1) != 0:
        return []
    entities = fd.get("entities", [])
    out = []
    for e in entities:
        raw_id = e.get("id", e.get("_label", e.get("VD:id", None)))
        dist   = e.get("_distance", float("inf"))
        if raw_id is not None:
            out.append({"id": int(raw_id), "distance": float(dist)})
    return out


# ── DINOv2 reranker ───────────────────────────────────────────────────────────

def rerank_with_dinov2(clip_results: List[Dict], query_dinov2: np.ndarray,
                       dinov2_db: np.ndarray, alpha: float) -> List[Dict]:
    """
    Stage-2 reranker: normalized weighted sum of CLIP + DINOv2 distances.

    final_score = (1-alpha) * clip_norm + alpha * dinov2_norm   (lower = better)

    alpha=0.0 → CLIP ranking unchanged
    alpha=1.0 → DINOv2 distance only
    """
    if not clip_results or alpha == 0.0:
        return clip_results

    cand_ids  = np.array([r["id"] for r in clip_results])
    clip_dist = np.array([r["distance"] for r in clip_results], dtype=np.float32)

    c_min, c_max = float(clip_dist.min()), float(clip_dist.max())
    clip_norm = (clip_dist - c_min) / (c_max - c_min + 1e-10)

    cand_dinov2  = dinov2_db[cand_ids]
    dinov2_dist  = np.linalg.norm(cand_dinov2 - query_dinov2, axis=1).astype(np.float32)

    d_min, d_max = float(dinov2_dist.min()), float(dinov2_dist.max())
    dinov2_norm  = (dinov2_dist - d_min) / (d_max - d_min + 1e-10)

    combined = (1.0 - alpha) * clip_norm + alpha * dinov2_norm
    order    = np.argsort(combined)
    return [{"id": int(cand_ids[i]), "distance": float(combined[i])}
            for i in order]


# ── Pseudo-Relevance Feedback query vector builder ────────────────────────────

def build_prf_query(base_dinov2: np.ndarray,
                    clip_results: List[Dict],
                    dinov2_db: np.ndarray,
                    n_refs: int,
                    ref_strategy: str) -> np.ndarray:
    """
    Build an augmented DINOv2 query vector using Pseudo-Relevance Feedback.

    n_refs=1 : return base_dinov2 unchanged (no PRF).
    n_refs>1 : average base_dinov2 with (n_refs-1) additional DINOv2 vectors
               selected from Stage-1 CLIP candidates via ref_strategy.

    ref_strategy:
      "first"    — top-(n_refs-1) candidates by CLIP distance rank
      "centroid" — centroid of ALL Stage-1 candidates' DINOv2 features
      "diverse"  — greedy farthest-point subset of Stage-1 DINOv2 features
    """
    if n_refs <= 1 or not clip_results:
        return base_dinov2

    n_extra   = n_refs - 1
    cand_ids  = np.array([r["id"] for r in clip_results])
    cand_vecs = dinov2_db[cand_ids]   # [K, 1024]
    n_cand    = len(cand_vecs)

    if ref_strategy == "centroid":
        # Centroid of ALL Stage-1 DINOv2 features (ignores n_extra count)
        extra = cand_vecs.mean(axis=0, keepdims=True)    # [1, 1024]

    elif ref_strategy == "diverse":
        n_pick = min(n_extra, n_cand)
        if n_pick == 0:
            return base_dinov2
        centroid  = cand_vecs.mean(axis=0)
        dists_c   = np.linalg.norm(cand_vecs - centroid, axis=1)
        seed_idx  = int(np.argmin(dists_c))
        chosen    = [seed_idx]
        min_dists = np.linalg.norm(cand_vecs - cand_vecs[seed_idx], axis=1)
        for _ in range(n_pick - 1):
            next_idx = int(np.argmax(min_dists))
            chosen.append(next_idx)
            d = np.linalg.norm(cand_vecs - cand_vecs[next_idx], axis=1)
            min_dists = np.minimum(min_dists, d)
        extra = cand_vecs[chosen]      # [n_pick, 1024]

    else:  # "first"
        n_pick = min(n_extra, n_cand)
        extra  = cand_vecs[:n_pick]    # [n_pick, 1024]

    # Stack query vector with extra vectors and average
    all_vecs = np.vstack([base_dinov2[np.newaxis, :], extra])   # [1+n_pick, 1024]
    aug      = all_vecs.mean(axis=0).astype(np.float32)
    norm     = np.linalg.norm(aug)
    return aug / (norm + 1e-10)


# ── Metric functions ──────────────────────────────────────────────────────────

def compute_ap(ranked_ids: List[int],
               relevant_set: Set[int],
               junk_set: Set[int]) -> float:
    """Average Precision (junk images are skipped, counted against all relevant)."""
    if not relevant_set:
        return 0.0
    hits, sum_prec, rank_adj = 0, 0.0, 0
    for img_id in ranked_ids:
        if img_id in junk_set:
            continue
        rank_adj += 1
        if img_id in relevant_set:
            hits += 1
            sum_prec += hits / rank_adj
    return sum_prec / len(relevant_set)


def compute_precision_at_k(ranked_ids: List[int],
                            relevant_set: Set[int],
                            junk_set: Set[int],
                            k: int = 10) -> float:
    """Precision@k."""
    if not relevant_set:
        return 0.0
    hits, rank_adj = 0, 0
    for img_id in ranked_ids:
        if img_id in junk_set:
            continue
        rank_adj += 1
        if rank_adj > k:
            break
        if img_id in relevant_set:
            hits += 1
    return hits / k if k > 0 else 0.0


def compute_recall_at_k(ranked_ids: List[int],
                         relevant_set: Set[int],
                         junk_set: Set[int],
                         k: int = 10) -> float:
    """Recall@k."""
    if not relevant_set:
        return 0.0
    hits, rank_adj = 0, 0
    for img_id in ranked_ids:
        if img_id in junk_set:
            continue
        rank_adj += 1
        if rank_adj > k:
            break
        if img_id in relevant_set:
            hits += 1
    return hits / len(relevant_set)


def compute_ndcg_at_k(ranked_ids: List[int],
                       relevant_set: Set[int],
                       junk_set: Set[int],
                       k: int = 10) -> float:
    """nDCG@k."""
    if not relevant_set:
        return 0.0
    dcg, rank_adj = 0.0, 0
    for img_id in ranked_ids:
        if img_id in junk_set:
            continue
        rank_adj += 1
        if rank_adj > k:
            break
        if img_id in relevant_set:
            dcg += 1.0 / math.log2(rank_adj + 1)
    idcg = _IDCG_CACHE.get(min(len(relevant_set), k), 0.0)
    return dcg / idcg if idcg > 0.0 else 0.0


# ── Main query loop ───────────────────────────────────────────────────────────

def run_gldv2_queries(db,
                      clip_q:     np.ndarray,
                      dinov2_q:   np.ndarray,
                      dinov2_db:  np.ndarray,
                      rel_sets:   List[Set[int]],
                      junk_sets:  List[Set[int]],
                      alpha:        float = 0.30,
                      k_neighbors:  int   = 200,
                      n_refs:       int   = 1,
                      ref_strategy: str   = "first",
                      clip_name:    str   = "CLIP_Set",
                      query_ids:    List[str] = None,
                      clip_db:      np.ndarray = None,
                      n_aqe:        int   = 1,
                      aqe_weight:   float = 0.0) -> Dict:
    """
    Two-stage retrieval for all GLDv2 queries — BATCH MODE.

    Stage 1:  All Q FindDescriptor commands issued in one db.query() call, amortising
              TCP socket overhead across all queries (mirrors HICO-DET batch design).
    Stage 1b: CLIP Alpha Query Expansion (AQE) — pure numpy, no extra VDMS calls.
              When n_aqe > 1 and aqe_weight > 0: expand each query vector by blending
              it with the mean CLIP embedding of the top-(n_aqe-1) Stage-1 candidates,
              then recompute candidate distances against the expanded query.
              q_exp = (1-aqe_weight)*q_orig + aqe_weight*mean(clip_db[top_(n_aqe-1)])
    Stage 2:  Weighted-sum DINOv2 reranking (+ optional PRF when n_refs > 1).

    Returns metrics dict:
      mAP, precision_at_10, recall_at_10, ndcg_at_10,
      qps, latency_avg_ms, t_clip_avg_ms, t_rerank_avg_ms, num_queries
    """
    n_queries = len(clip_q)
    aps, precs, recs, ndcgs = [], [], [], []
    total_t_rerank = 0.0

    k_fetch = k_neighbors

    # ── Stage 1: build and fire one batch VDMS call ───────────────────────────
    clip_ops   = []
    clip_blobs = []
    for q_idx in range(n_queries):
        clip_ops.append({"FindDescriptor": {
            "set":         clip_name,
            "k_neighbors": k_fetch,
            "results":     {"list": ["id", "_distance"]},
        }})
        clip_blobs.append(clip_q[q_idx].tobytes())

    search_start = time.time()
    _t0 = time.perf_counter()
    batch_res, _ = db.query(clip_ops, clip_blobs)
    total_t_clip_ms = (time.perf_counter() - _t0) * 1000   # whole batch, ms

    # ── Stage 1b + Stage 2: per-query numpy (AQE + DINOv2) ───────────────────
    for q_idx in range(n_queries):
        cq = clip_q[q_idx]
        dq = dinov2_q[q_idx]

        clip_results = _parse_entities(
            batch_res[q_idx] if (batch_res and q_idx < len(batch_res)) else None
        )

        # AQE: blend original CLIP query with mean of top-(n_aqe-1) candidates
        if clip_db is not None and n_aqe > 1 and aqe_weight > 0.0 and len(clip_results) >= n_aqe:
            cand_ids   = np.array([r["id"] for r in clip_results])
            n_refs_aqe = min(n_aqe - 1, len(cand_ids))
            q_exp      = ((1.0 - aqe_weight) * cq
                          + aqe_weight * clip_db[cand_ids[:n_refs_aqe]].mean(axis=0))
            q_exp     /= (np.linalg.norm(q_exp) + 1e-10)
            new_dists  = np.sum((clip_db[cand_ids] - q_exp) ** 2, axis=1)
            clip_results = sorted(
                [{"id": int(cand_ids[j]), "distance": float(new_dists[j])}
                 for j in range(len(cand_ids))],
                key=lambda r: r["distance"],
            )

        if q_idx == 0:
            qid_label = query_ids[0] if query_ids else "q0"
            print(f"[GLDv2 q0] id={qid_label}  CLIP={len(clip_results)} candidates  "
                  f"alpha={alpha:.2f}  n_aqe={n_aqe}  aqe_w={aqe_weight:.2f}  "
                  f"n_refs={n_refs}  ref_strategy={ref_strategy}  "
                  f"batch_t={total_t_clip_ms:.1f}ms total")

        _t0 = time.perf_counter()
        aug_dq   = build_prf_query(dq, clip_results, dinov2_db, n_refs, ref_strategy)
        reranked = rerank_with_dinov2(clip_results, aug_dq, dinov2_db, alpha)
        total_t_rerank += (time.perf_counter() - _t0) * 1000

        ranked_ids = [r["id"] for r in reranked]
        rel  = rel_sets[q_idx]
        junk = junk_sets[q_idx]

        aps.append(compute_ap(ranked_ids, rel, junk))
        precs.append(compute_precision_at_k(ranked_ids, rel, junk, k=10))
        recs.append(compute_recall_at_k(ranked_ids, rel, junk, k=10))
        ndcgs.append(compute_ndcg_at_k(ranked_ids, rel, junk, k=10))

    total_time = time.time() - search_start
    return {
        "mAP":             float(np.mean(aps)),
        "precision_at_10": float(np.mean(precs)),
        "recall_at_10":    float(np.mean(recs)),
        "ndcg_at_10":      float(np.mean(ndcgs)),
        "qps":             float(n_queries / total_time),
        "latency_avg_ms":  float(total_time / n_queries * 1000),
        "t_clip_avg_ms":   float(total_t_clip_ms / n_queries),   # amortised per query
        "t_rerank_avg_ms": float(total_t_rerank   / n_queries),
        "num_queries":     n_queries,
        "per_query_ap":    [float(a) for a in aps],
    }
