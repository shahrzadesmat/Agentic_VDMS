"""
HICO-DET Benchmark — SIFT + DINOv2 dual DescriptorSet in VDMS
============================================================

HICO-DET (Human-Object Interaction Detection, Chao et al. WACV 2018)
47,776 images annotated with (object, verb) interaction pairs.
600 HOI categories = 80 COCO objects × 117 verbs.

Evaluation protocol:
  - One query per HOI category (first image in the HOI index)
  - DB: all 47,776 images indexed
  - Relevant: all images sharing the same HOI (positive_hoi), excl. query image
  - Junk: the query image itself (trivial self-match)
  - Skip HOIs with fewer than MIN_RELEVANT relevant images after removing query

Metrics (all ceiling = 1.0, LLM-interpretable):
  Precision@10  : hits/10
  nDCG@10       : ranking quality
  Recall@10     : top-10 hits / total_relevant  (useful when many relevant exist)
  mAP           : mean Average Precision (full ranked list)

Dataset advantages for SIFT+DINOv2 fusion:
  - Same object, different verb → SIFT disambiguates pose/geometry
    ("hold+bottle" vs "drink+bottle"; "ride+bicycle" vs "hold+bicycle")
  - DINOv2 strong for object identity; SIFT strong for body/object configuration

Set names: "SIFT_Set" / "DEEP_Set" (short — same as benchmark_semantic.py,
which works correctly with PMGD. Oxford5K used longer names and hit
PropertyTypeMismatch on the deep DescriptorSet.)

Usage:
    python benchmark_hico_det.py <port> <dataset_dir>
    python benchmark_hico_det.py 55570 /path/to/datasets
"""

import sys
import math
import time
import json
import numpy as np
import vdms
from typing import List, Set, Dict, Tuple, Optional

# Pre-compute IDCG denominators for k=1..200
_IDCG_CACHE = {k: sum(1.0 / math.log2(i + 2) for i in range(k)) for k in range(1, 201)}

# ---- DescriptorSet names — SHORT to avoid Oxford PropertyTypeMismatch ----
SIFT_SET_NAME = "SIFT_Set"
DEEP_SET_NAME = "DEEP_Set"

BATCH_SIZE   = 50    # PMGD journal limit (same as benchmark_semantic.py)
MIN_RELEVANT = 5     # skip HOIs with fewer relevant images (no meaningful metric)


# ================================================================
# DATA LOADING
# ================================================================

def load_hico_data(dataset_dir: str) -> Dict:
    """
    Load HICO-DET features and ground truth.

    Geometric modality priority (for sift_db, stored in "SIFT_Set" in VDMS):
      1. hico_pose.npy     [47776,  51]  — pose keypoints (KeypointRCNN, TRULY INDEPENDENT)
      2. hico_sift_spm.npy [47776, 512]  — SPM-VLAD SIFT (spatial pyramid, bbox-crop)
      3. hico_sift.npy     [47776, 512]  — VLAD-SIFT (bbox-crop or full-image)

    hico_pose.npy is the PREFERRED modality: 17 COCO keypoints × (x_norm, y_norm, conf) = 51-d,
    normalized to person bbox, L2-normed. This is ORTHOGONAL to DINOv2 CLS:
      DINOv2 → WHAT objects are present  (semantic identity)
      Pose   → HOW the body is configured (action verb geometry)

    Semantic modality:
      hico_dinov2.npy  [47776, 1024]  — DINOv2 ViT-L/14-reg4 (preferred over MobileCLIP)

    Returns dict with keys:
      sift_db     [47776, D_geom]  float32 — geometric descriptors (pose or VLAD-SIFT)
      deep_db     [47776, D_deep]  float32 — DINOv2 ViT-L/14-reg4 embeddings (unit-normed)
      gt          list[dict]       len=47776 — per-image GT
      hoi_index   dict  "obj|verb" → [image_ids]
    """
    d = dataset_dir.rstrip("/") + "/hico_det"
    import os as _os

    # Auto-prefer SPM-VLAD (spatial pyramid, strong geometric) > bbox-VLAD > full-image VLAD
    # Pose keypoints (51-d) are too low-dimensional and coarse to help fusion over DINOv2.
    if _os.path.exists(f"{d}/hico_sift_spm.npy"):
        sift_db    = np.load(f"{d}/hico_sift_spm.npy").astype(np.float32)
        sift_label = "SPM-VLAD-SIFT (1×1+2×2+1×3=8 cells, bbox-crop, PCA-512)"
    else:
        sift_db    = np.load(f"{d}/hico_sift.npy").astype(np.float32)
        sift_label = "VLAD-SIFT (k=64, bbox-crop, PCA-512)" \
                     if _os.path.exists(f"{d}/hico_bboxes.json") \
                     else "VLAD-SIFT (k=64, full-image, PCA-512)"
    print(f"[GEOM] {sift_label}  shape={sift_db.shape}")

    # Prefer HOI-crop DINOv2 (focused on interaction region) > full-image DINOv2 > MobileCLIP
    if _os.path.exists(f"{d}/hico_dinov2_crop.npy"):
        deep_db    = np.load(f"{d}/hico_dinov2_crop.npy").astype(np.float32)
        deep_label = f"DINOv2 ViT-L/14-reg4 HOI-crop  {deep_db.shape[1]}-d"
    elif _os.path.exists(f"{d}/hico_dinov2.npy"):
        deep_db    = np.load(f"{d}/hico_dinov2.npy").astype(np.float32)
        deep_label = f"DINOv2 ViT-L/14-reg4  {deep_db.shape[1]}-d"
    else:
        deep_db    = np.load(f"{d}/hico_clip.npy").astype(np.float32)
        deep_label = f"MobileCLIP-S2  {deep_db.shape[1]}-d"

    with open(f"{d}/hico_gt.json") as f:
        gt = json.load(f)
    with open(f"{d}/hico_hoi_index.json") as f:
        hoi_index = json.load(f)

    print(f"[OK] HICO-DET loaded: {len(sift_db)} images")
    print(f"     Geom  {sift_db.shape}  ({sift_label[:60]})")
    print(f"     Deep  {deep_db.shape}  ({deep_label})")
    deep_norms = np.linalg.norm(deep_db, axis=1)
    print(f"     Deep norms: {deep_norms.mean():.6f} ± {deep_norms.std():.6f}")
    print(f"     HOI categories: {len(hoi_index)}")

    sizes = [len(v) for v in hoi_index.values()]
    print(f"     Images/HOI: min={min(sizes)}  max={max(sizes)}  mean={np.mean(sizes):.1f}")

    return dict(sift_db=sift_db, deep_db=deep_db, gt=gt, hoi_index=hoi_index)


def load_hico_metadata(dataset_dir: str) -> Tuple[List[Dict], Dict, Dict]:
    """
    Load per-image PMGD metadata produced by extract_hico_metadata.py.

    Returns:
      img_metadata : list[dict]  len=47776, keys: img_id, primary_object_id,
                                  primary_verb_id, n_hoi, all_object_ids, all_verb_ids
      obj_vocab    : dict  object_name → int  (80 COCO objects, sorted)
      verb_vocab   : dict  verb_name   → int  (117 HOI verbs,   sorted)

    Used by setup_hico_index / setup_clip_index to store object_id + verb_id
    as PMGD properties on every AddDescriptor, enabling constrained ANN queries.
    """
    import os as _os
    d = dataset_dir.rstrip("/") + "/hico_det"
    meta_path = f"{d}/hico_metadata.json"
    obj_path  = f"{d}/hico_object_vocab.json"
    verb_path = f"{d}/hico_verb_vocab.json"

    if not _os.path.exists(meta_path):
        raise FileNotFoundError(
            f"PMGD metadata not found: {meta_path}\n"
            "Run: python extract_hico_metadata.py --dataset-dir <dataset_dir>"
        )

    with open(meta_path)  as f: img_metadata = json.load(f)
    with open(obj_path)   as f: obj_vocab    = json.load(f)
    with open(verb_path)  as f: verb_vocab   = json.load(f)

    print(f"[OK] PMGD metadata loaded: {len(img_metadata)} images  "
          f"({len(obj_vocab)} objects, {len(verb_vocab)} verbs)")
    return img_metadata, obj_vocab, verb_vocab


def build_per_class_clip_indices(
    clip_db: np.ndarray,
    img_metadata: List[Dict],
    obj_vocab: Dict,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Pre-build per-object-class arrays for exact numpy CLIP search.

    For each COCO object class (80 classes), collect all DB images that contain
    that object (via all_object_ids) and store their CLIP vectors + original indices.

    Returns:
      dict mapping obj_id (int) → (vectors (N_class, D) float32, image_ids (N_class,) int32)

    At query time, search text_q against vectors using dot product — exact cosine
    similarity (assuming unit-normed CLIP vectors). ~0.05ms per query vs ~2.2ms (47K VDMS).
    """
    from collections import defaultdict
    class_images: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(img_metadata):
        oids = m.get("all_object_ids") or []
        if not oids:
            oid = m.get("primary_object_id", -1)
            if oid >= 0:
                oids = [oid]
        for oid in oids:
            if oid >= 0:
                class_images[int(oid)].append(i)

    indices: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for oid, img_ids in class_images.items():
        arr = np.array(img_ids, dtype=np.int32)
        vecs = clip_db[arr].copy()  # (N_class, D) float32
        indices[oid] = (vecs, arr)

    sizes = [len(v[1]) for v in indices.values()]
    print(f"[OK] Per-class CLIP indices: {len(indices)} classes  "
          f"sizes: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}")
    return indices


def build_hoi_queries(hoi_index: Dict,
                      sift_db: np.ndarray,
                      deep_db: np.ndarray,
                      min_relevant: int = MIN_RELEVANT
                      ) -> Tuple[np.ndarray, np.ndarray, List[Set[int]], List[str]]:
    """
    Build one query per HOI category.

    For each HOI key ("obj|verb"):
      - query image = first image in the HOI index
      - relevant    = all OTHER images with that HOI
      - junk        = {query image idx}  (trivial self-match)
      - skip HOIs with fewer than min_relevant relevant images

    Returns:
      sift_q   [N_queries, D_geom]  float32  (pose: 51-d; VLAD-SIFT: 512-d)
      deep_q   [N_queries, D_deep]  float32  (DINOv2: 1024-d)
      rel_sets list of N_queries sets of relevant DB indices (query excluded)
      hoi_keys list of N_queries HOI key strings ("obj|verb")
    """
    sift_q_list  = []
    deep_q_list  = []
    rel_sets     = []
    hoi_keys     = []
    junk_sets    = []
    skipped      = 0

    for key, img_ids in hoi_index.items():
        img_ids = list(img_ids)
        if len(img_ids) < min_relevant + 1:
            skipped += 1
            continue

        q_idx = img_ids[0]                        # query image
        rel   = set(img_ids) - {q_idx}            # relevant (excl. query)

        sift_q_list.append(sift_db[q_idx])
        deep_q_list.append(deep_db[q_idx])
        rel_sets.append(rel)
        junk_sets.append({q_idx})
        hoi_keys.append(key)

    n = len(hoi_keys)
    print(f"[OK] HOI queries built: {n} queries  (skipped {skipped} HOIs with < {min_relevant+1} images)")
    sizes = [len(r) for r in rel_sets]
    print(f"     Relevant/query: min={min(sizes)}  max={max(sizes)}  mean={np.mean(sizes):.1f}")

    return (np.array(sift_q_list, dtype=np.float32),
            np.array(deep_q_list, dtype=np.float32),
            rel_sets, junk_sets, hoi_keys)


def build_dinov2_refs(hoi_index: Dict, deep_db: np.ndarray,
                      hoi_keys: List[str], n_refs: int = 1,
                      ref_strategy: str = "first") -> np.ndarray:
    """
    Compute DINOv2 reference vectors for each HOI category.

    n_refs    : how many images to aggregate per class (1, 3, 5, 10)
    ref_strategy:
      "first"    — take the first n_refs images in the HOI index order (default).
                   Fast and deterministic; may be unrepresentative if first images
                   share a visual bias (e.g. all from the same scene).
      "centroid" — compute the visual centroid of ALL class images, then pick
                   the n_refs images closest to that centroid and average them.
                   Result: a reference that sits at the visual core of the class,
                   minimising the risk of anchoring on an outlier image.
      "diverse"  — greedy farthest-point sampling: start from the image closest
                   to the centroid, then iteratively add the image whose DINOv2
                   vector is FURTHEST from the nearest already-selected image.
                   Average those n_refs vectors.
                   Result: a reference that spans the visual diversity of the class,
                   making reranking more robust for HOIs with high intra-class
                   appearance variation (e.g. "person riding motorcycle").

    All strategies return one averaged (D,) vector per HOI class.
    Pure numpy — negligible cost (~0ms). Does NOT affect QPS.
    """
    refs = []
    for key in hoi_keys:
        img_ids = list(hoi_index[key])
        n_avail = len(img_ids)
        n       = min(n_refs, n_avail)
        vecs    = deep_db[img_ids]               # (n_avail, D)

        if n == 1 or ref_strategy == "first":
            selected = vecs[:n]

        elif ref_strategy == "centroid":
            centroid  = vecs.mean(axis=0)        # (D,)
            dists     = np.linalg.norm(vecs - centroid, axis=1)
            if n >= n_avail:                      # argpartition requires kth < len
                selected = vecs
            else:
                idx      = np.argpartition(dists, n)[:n]
                selected = vecs[idx]

        elif ref_strategy == "diverse":
            # Greedy farthest-point sampling
            centroid  = vecs.mean(axis=0)
            dists_c   = np.linalg.norm(vecs - centroid, axis=1)
            seed_idx  = int(np.argmin(dists_c))   # start from most-central image
            chosen    = [seed_idx]
            min_dists = np.linalg.norm(vecs - vecs[seed_idx], axis=1)
            for _ in range(n - 1):
                next_idx = int(np.argmax(min_dists))
                chosen.append(next_idx)
                d = np.linalg.norm(vecs - vecs[next_idx], axis=1)
                min_dists = np.minimum(min_dists, d)
            selected = vecs[chosen]

        else:
            raise ValueError(f"Unknown ref_strategy: {ref_strategy!r}. "
                             f"Choose from 'first', 'centroid', 'diverse'.")

        refs.append(selected.mean(axis=0))        # (D,)

    return np.array(refs, dtype=np.float32)


# ================================================================
# WHITENING + AQE  (offline post-processing — applied before VDMS indexing)
# ================================================================

def apply_whitening(db_vecs: np.ndarray, q_vecs: np.ndarray,
                    n_components: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA whitening (Jégou & Chum ECCV 2012).

    Fits PCA on db_vecs, transforms both db and query vectors.
    Reduces dim from D → n_components, decorrelates, equalises variance.
    Output vectors are L2-normalised.

    On Oxford5K DINOv2 (1024-d → 256-d): +0.06 mAP (0.837 → 0.896).
    """
    from sklearn.decomposition import PCA
    n_components = min(n_components, db_vecs.shape[0] - 1, db_vecs.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="full")
    db_w = pca.fit_transform(db_vecs).astype(np.float32)
    q_w  = pca.transform(q_vecs).astype(np.float32)
    db_w /= np.linalg.norm(db_w, axis=1, keepdims=True).clip(min=1e-10)
    q_w  /= np.linalg.norm(q_w,  axis=1, keepdims=True).clip(min=1e-10)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  [Whitening] {db_vecs.shape[1]}-d → {n_components}-d  "
          f"var_explained={var_explained:.3f}")
    return db_w, q_w


def offline_aqe(q_vecs: np.ndarray, db_vecs: np.ndarray,
                k: int = 100, alpha: float = 3.0) -> np.ndarray:
    """
    Alpha-Query Expansion (Chum et al. CVPR 2007, α-weighted variant).

    For each query, takes the top-k most similar DB vectors (cosine sim),
    forms a weighted sum (weights = sim^α), re-normalises.
    Best with whitened features: k=100, α=3 (Oxford5K result).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    sims  = cosine_similarity(q_vecs, db_vecs)         # [Q, N]
    new_q = np.zeros_like(q_vecs)
    for i in range(len(q_vecs)):
        top_idx = np.argsort(-sims[i])[:k]
        weights = sims[i][top_idx] ** alpha
        total   = weights.sum()
        if total > 1e-10:
            weights /= total
        new_q[i] = weights @ db_vecs[top_idx]
        norm = np.linalg.norm(new_q[i])
        if norm > 1e-10:
            new_q[i] /= norm
    return new_q.astype(np.float32)


# ================================================================
# VDMS INDEX
# ================================================================

def setup_hico_index(db, engine: str, engine_params: dict,
                     sift_db: np.ndarray,
                     deep_db: np.ndarray,
                     sift_name: str = SIFT_SET_NAME,
                     deep_name: str = DEEP_SET_NAME,
                     metadata: List[Dict] = None) -> Tuple[float, float]:
    """
    Create both DescriptorSets and index all 47,776 HICO-DET vectors.

    Set names are SHORT by default ("SIFT_Set"/"DEEP_Set") — same as benchmark_semantic.py.
    Custom names (sift_name/deep_name) allow multiple simultaneous index setups
    (e.g., baseline + whitened) on the same VDMS instance without conflicts.

    Dimensions are inferred from the arrays: sift_db.shape[1] and deep_db.shape[1].
    This supports both 512-d MobileCLIP and 1024-d DINOv2 (and whitened variants).

    Returns: (build_time_s, index_time_s)
    """
    sift_cfg = {
        "name": sift_name, "dimensions": int(sift_db.shape[1]), "metric": "L2",
        "engine": engine, "properties": {},
    }
    deep_cfg = {
        "name": deep_name, "dimensions": int(deep_db.shape[1]), "metric": "L2",
        "engine": engine, "properties": {},
    }

    if engine == "FaissHNSWFlat":
        for cfg in [sift_cfg, deep_cfg]:
            if "M"              in engine_params: cfg["properties"]["M"]              = int(engine_params["M"])
            if "efConstruction" in engine_params: cfg["properties"]["efConstruction"] = int(engine_params["efConstruction"])
            if "efSearch"       in engine_params: cfg["properties"]["efSearch"]       = int(engine_params["efSearch"])
    elif engine == "FaissIVFFlat":
        for cfg in [sift_cfg, deep_cfg]:
            if "nlist"  in engine_params: cfg["properties"]["nlist"]  = int(engine_params["nlist"])
            if "nprobe" in engine_params: cfg["properties"]["nprobe"] = int(engine_params["nprobe"])

    build_start = time.time()
    for label, cfg in [("SIFT", sift_cfg), ("DEEP", deep_cfg)]:
        print(f"Creating {label} DescriptorSet ({cfg['name']})...")
        res, _ = db.query([{"AddDescriptorSet": cfg}])
        if not res or "AddDescriptorSet" not in res[0] or res[0]["AddDescriptorSet"]["status"] != 0:
            raise RuntimeError(f"{label} DescriptorSet creation failed: {res}")
    build_time = time.time() - build_start

    n = len(sift_db)
    index_start = time.time()

    # Build per-image properties dict — include object_id + verb_id if metadata provided
    use_graph = (metadata is not None and len(metadata) == n)
    if use_graph:
        print(f"[GRAPH] PMGD metadata provided — storing object_id + verb_id per descriptor")
    else:
        print(f"[GRAPH] No metadata — storing id only (constraint queries disabled)")

    def _make_props(i: int) -> Dict:
        props = {"id": i}
        if use_graph:
            m = metadata[i]
            if m["primary_object_id"] >= 0:
                props["object_id"] = int(m["primary_object_id"])
            if m["primary_verb_id"] >= 0:
                props["verb_id"] = int(m["primary_verb_id"])
        return props

    # Index SIFT first — same order as benchmark_semantic.py
    print(f"Indexing {n:,} SIFT vectors  dim={sift_db.shape[1]}  set={sift_name}  (batch={BATCH_SIZE})...")
    sift_fail = 0
    for start in range(0, n, BATCH_SIZE):
        batch_cmds  = []
        batch_blobs = []
        for i in range(start, min(start + BATCH_SIZE, n)):
            batch_cmds.append({"AddDescriptor": {
                "set": sift_name,
                "label": str(i),
                "properties": _make_props(i),
            }})
            batch_blobs.append(sift_db[i].astype(np.float32).tobytes())
        res, _ = db.query(batch_cmds, batch_blobs)
        if res:
            if start == 0:
                print(f"  [DEBUG] SIFT batch 0: len(res)={len(res)}")
                for ri, r in enumerate(res[:2]):
                    print(f"    res[{ri}] = {r}")
            failed = sum(1 for r in res if r.get("AddDescriptor", {}).get("status", -1) != 0)
            sift_fail += failed
        if start % 5000 == 0 and start > 0:
            print(f"  SIFT progress: {start:,}/{n:,}")
    print(f"  SIFT failures: {sift_fail}/{n}  ({'FAIL' if sift_fail > 0 else 'OK'})")

    # Index deep features (DINOv2/CLIP) second
    print(f"Indexing {n:,} deep vectors  dim={deep_db.shape[1]}  set={deep_name}  (batch={BATCH_SIZE})...")
    deep_fail = 0
    for start in range(0, n, BATCH_SIZE):
        batch_cmds  = []
        batch_blobs = []
        for i in range(start, min(start + BATCH_SIZE, n)):
            batch_cmds.append({"AddDescriptor": {
                "set": deep_name,
                "label": str(i),
                "properties": _make_props(i),
            }})
            batch_blobs.append(deep_db[i].astype(np.float32).tobytes())
        res, _ = db.query(batch_cmds, batch_blobs)
        if res:
            if start == 0:
                print(f"  [DEBUG] DEEP batch 0: len(res)={len(res)}")
                for ri, r in enumerate(res[:2]):
                    print(f"    res[{ri}] = {r}")
            failed = sum(1 for r in res if r.get("AddDescriptor", {}).get("status", -1) != 0)
            deep_fail += failed
        if start % 5000 == 0 and start > 0:
            print(f"  DEEP progress: {start:,}/{n:,}")
    print(f"  DEEP failures: {deep_fail}/{n}  ({'FAIL - PropertyTypeMismatch bug with short names?' if deep_fail > 0 else 'OK'})")

    index_time = time.time() - index_start
    print(f"[OK] HICO-DET index ready  (build: {build_time:.2f}s, index: {index_time:.1f}s)")

    if sift_fail > 0 or deep_fail > 0:
        raise RuntimeError(
            f"Indexing failures: SIFT={sift_fail}/{n}  DEEP={deep_fail}/{n}\n"
            "If DEEP fails with short set names ('SIFT_Set'/'DEEP_Set'), the "
            "PropertyTypeMismatch is NOT name-length-related.  Consider switching engine or "
            "further investigating PMGD root cause."
        )

    return build_time, index_time


# ================================================================
# DISTANCE NORMALIZATION  (shared with benchmark_oxford.py)
# ================================================================

def normalize_distances(distances, metric: str) -> np.ndarray:
    """Normalize to [0,1] where 0 = most similar."""
    distances = np.array(distances, dtype=float)
    if len(distances) == 0:
        return distances
    if metric == "L2":
        p99 = np.percentile(distances, 99)
        if p99 > 0:
            return np.clip(distances / p99, 0.0, 1.0)
        return distances
    elif metric == "IP":
        min_d, max_d = distances.min(), distances.max()
        if max_d > min_d:
            return (max_d - distances) / (max_d - min_d)
        return np.zeros_like(distances)
    return distances


def dbsf_normalize(distances, metric: str = "L2") -> np.ndarray:
    """Distribution-Based Score Fusion normalization (mean ± 3σ window)."""
    distances = np.array(distances, dtype=float)
    if len(distances) == 0:
        return distances
    mu  = np.mean(distances)
    sig = np.std(distances)
    low, high = mu - 3 * sig, mu + 3 * sig
    if high > low:
        normed = np.clip((distances - low) / (high - low), 0.0, 1.0)
    else:
        normed = np.zeros_like(distances)
    if metric == "IP":
        normed = 1.0 - normed
    return normed


# ================================================================
# FUSION  (identical logic to benchmark_oxford.py)
# ================================================================

def _parse_entities(find_res) -> List[Dict]:
    """Extract [{id, distance}, ...] from a FindDescriptor VDMS response."""
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


def combine_results(sift_results: List[Dict], deep_results: List[Dict],
                    lambda_param: float, k: int = 10,
                    fusion_method: str = "weighted_rrf",
                    k_rrf: int = 60) -> List[Dict]:
    """Fuse SIFT and DINOv2 ranked lists — identical to benchmark_oxford.py."""
    if fusion_method == "rrf":
        scores: Dict[int, float] = {}
        for rank, r in enumerate(sift_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k_rrf + rank)
        for rank, r in enumerate(deep_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k_rrf + rank)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"id": rid, "score": s} for rid, s in ranked[:k]]

    if fusion_method == "weighted_rrf":
        w_sift = 1.0 - lambda_param
        w_deep = lambda_param
        scores: Dict[int, float] = {}
        for rank, r in enumerate(sift_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + w_sift / (k_rrf + rank)
        for rank, r in enumerate(deep_results):
            scores[r["id"]] = scores.get(r["id"], 0.0) + w_deep / (k_rrf + rank)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [{"id": rid, "score": s} for rid, s in ranked[:k]]

    if fusion_method == "entropy":
        def _ent(results):
            dists = np.array([r["distance"] for r in results], dtype=float)
            dists = dists[np.isfinite(dists)]
            if len(dists) == 0:
                return 0.5
            dists = dists - dists.min() + 1e-9
            p = dists / dists.sum()
            return -np.sum(p * np.log(p + 1e-12))
        h_sift = _ent(sift_results)
        h_deep = _ent(deep_results)
        total  = h_sift + h_deep
        lam    = (h_sift / total) if total > 0 else 0.5
        return combine_results(sift_results, deep_results, lam, k, "weighted", k_rrf)

    if fusion_method == "entropy_rrf":
        def _ent(results):
            dists = np.array([r["distance"] for r in results], dtype=float)
            dists = dists[np.isfinite(dists)]
            if len(dists) == 0:
                return 0.5
            dists = dists - dists.min() + 1e-9
            p = dists / dists.sum()
            return -np.sum(p * np.log(p + 1e-12))
        h_sift = _ent(sift_results)
        h_deep = _ent(deep_results)
        total  = h_sift + h_deep
        lam    = (h_sift / total) if total > 0 else 0.5
        return combine_results(sift_results, deep_results, lam, k, "weighted_rrf", k_rrf)

    # Distance-based: weighted / dbsf
    all_ids  = set(r["id"] for r in sift_results) | set(r["id"] for r in deep_results)
    sift_map = {r["id"]: r["distance"] for r in sift_results}
    deep_map = {r["id"]: r["distance"] for r in deep_results}

    ids_arr        = np.array(list(all_ids))
    sift_dists_raw = np.array([sift_map.get(i, float("inf")) for i in ids_arr])
    deep_dists_raw = np.array([deep_map.get(i, float("inf")) for i in ids_arr])
    sift_dists     = np.where(np.isinf(sift_dists_raw), np.nan, sift_dists_raw)
    deep_dists     = np.where(np.isinf(deep_dists_raw), np.nan, deep_dists_raw)

    PENALTY = 1.0
    sift_mask = np.isfinite(sift_dists)
    deep_mask = np.isfinite(deep_dists)

    if fusion_method == "dbsf":
        sn = dbsf_normalize(sift_dists[sift_mask], "L2") if sift_mask.any() else np.array([])
        cn = dbsf_normalize(deep_dists[deep_mask], "L2") if deep_mask.any() else np.array([])
    else:  # weighted
        sn = normalize_distances(sift_dists[sift_mask], "L2") if sift_mask.any() else np.array([])
        cn = normalize_distances(deep_dists[deep_mask], "L2") if deep_mask.any() else np.array([])

    sift_norm = np.full(len(ids_arr), PENALTY)
    deep_norm = np.full(len(ids_arr), PENALTY)
    sift_norm[sift_mask] = sn
    deep_norm[deep_mask] = cn

    if lambda_param == 0.0:
        combined = sift_norm
    elif lambda_param == 1.0:
        combined = deep_norm
    else:
        combined = (1.0 - lambda_param) * sift_norm + lambda_param * deep_norm

    order = np.argsort(combined)
    return [{"id": int(ids_arr[i]), "distance": float(combined[i])}
            for i in order[:k]]


# ================================================================
# METRIC COMPUTATION
# ================================================================

def compute_ap(ranked_ids: List[int], relevant_set: Set[int],
               junk_set: Set[int]) -> float:
    """Average Precision (skip junk, count against all relevant in DB)."""
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


def compute_precision_at_k(ranked_ids: List[int], relevant_set: Set[int],
                            junk_set: Set[int], k: int = 10) -> float:
    """Precision@k (ceiling = 1.0)."""
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


def compute_recall_at_k(ranked_ids: List[int], relevant_set: Set[int],
                         junk_set: Set[int], k: int = 10) -> float:
    """Recall@k = hits in top-k / total relevant (ceiling = 1.0)."""
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


def compute_ndcg_at_k(ranked_ids: List[int], relevant_set: Set[int],
                       junk_set: Set[int], k: int = 10) -> float:
    """nDCG@k with junk exclusion (ceiling = 1.0)."""
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
    idcg = _IDCG_CACHE.get(min(k, len(relevant_set)), 0.0)
    return dcg / idcg if idcg > 0 else 0.0


# ================================================================
# QUERY RUNNER
# ================================================================

def run_hico_queries(db, sift_query: np.ndarray, deep_query: np.ndarray,
                     relevant_sets: List[Set[int]], junk_sets: List[Set[int]],
                     lambda_param: float,
                     fusion_method: str = "weighted_rrf",
                     k_neighbors: int = 500,
                     k_rrf: int = 60,
                     sift_name: str = SIFT_SET_NAME,
                     deep_name: str = DEEP_SET_NAME,
                     hoi_keys: List[str] = None,
                     constraint_strategy: str = "none",
                     obj_vocab: Dict = None,
                     verb_vocab: Dict = None) -> Dict:
    """
    Run all HICO-DET HOI queries and compute retrieval metrics.

    Returns dict with Precision@10, nDCG@10, Recall@10, mAP, QPS.
    """
    n_queries = len(sift_query)
    aps       = []
    precs_10  = []
    recs_10   = []
    ndcgs_10  = []

    # Per-component timing accumulators (ms)
    total_t_sift   = 0.0   # VDMS SIFT FindDescriptor
    total_t_deep   = 0.0   # VDMS DINOv2 FindDescriptor
    total_t_fusion = 0.0   # combine_results() fusion logic

    search_start = time.time()

    # Build per-query PMGD constraints from hoi_key ("object|verb")
    use_constraint = (
        constraint_strategy != "none"
        and hoi_keys is not None
        and obj_vocab is not None
        and verb_vocab is not None
    )

    def _build_constraint(q_idx: int) -> Dict:
        """Return constraints dict for FindDescriptor, or {} if none."""
        if not use_constraint:
            return {}
        key = hoi_keys[q_idx]
        obj_name, verb_name = key.split("|", 1)
        c = {}
        if constraint_strategy in ("object", "object_and_verb"):
            oid = obj_vocab.get(obj_name)
            if oid is not None:
                c["object_id"] = ["==", int(oid)]
        if constraint_strategy in ("verb", "object_and_verb"):
            vid = verb_vocab.get(verb_name)
            if vid is not None:
                c["verb_id"] = ["==", int(vid)]
        return c

    for q_idx in range(n_queries):
        sift_q = sift_query[q_idx]
        deep_q = deep_query[q_idx]

        constraint = _build_constraint(q_idx)

        sift_find = {
            "set": sift_name, "k_neighbors": k_neighbors,
            "results": {"list": ["id", "_distance"]}
        }
        deep_find = {
            "set": deep_name, "k_neighbors": k_neighbors,
            "results": {"list": ["id", "_distance"]}
        }
        if constraint:
            sift_find["constraints"] = constraint
            deep_find["constraints"] = constraint

        # Sequential single-blob queries (dual-blob causes VDMS hang)
        _t0 = time.perf_counter()
        sift_res, _ = db.query([{"FindDescriptor": sift_find}], [sift_q.tobytes()])
        total_t_sift += (time.perf_counter() - _t0) * 1000

        _t0 = time.perf_counter()
        deep_res, _ = db.query([{"FindDescriptor": deep_find}], [deep_q.tobytes()])
        total_t_deep += (time.perf_counter() - _t0) * 1000

        sift_results = _parse_entities(sift_res[0] if sift_res else None)
        deep_results = _parse_entities(deep_res[0] if deep_res else None)

        # Debug + sanity on first query
        if q_idx == 0:
            hoi_label = hoi_keys[0] if hoi_keys else "unknown"
            print(f"[DEBUG q0] HOI={hoi_label}  k_neighbors={k_neighbors}")
            print(f"[DEBUG q0] Geom returned {len(sift_results)} results")
            print(f"[DEBUG q0] DINOv2 returned {len(deep_results)} results")
            if sift_results:
                print(f"  Geom top-3: {sift_results[:3]}")
            if deep_results:
                print(f"  DINOv2 top-3: {deep_results[:3]}")
            if not sift_results and not deep_results:
                raise RuntimeError(
                    "VDMS returned 0 entities for query 0 (both Geom and DINOv2).\n"
                    f"sift_res={sift_res}  deep_res={deep_res}"
                )
            if sift_results and not deep_results:
                raise RuntimeError(
                    f"DINOv2 returned 0 results (Geom returned {len(sift_results)}).\n"
                    "DINOv2 DescriptorSet is likely empty — AddDescriptor all failed.\n"
                    f"deep_res={deep_res}"
                )
            if sift_results and deep_results:
                sift_top10 = [r["id"] for r in sift_results[:10]]
                deep_top10 = [r["id"] for r in deep_results[:10]]
                if sift_top10 == deep_top10:
                    raise RuntimeError(
                        "MODALITY SEPARATION FAILURE: DINOv2 top-10 == SIFT top-10 for q0.\n"
                        "Check DINOv2 DescriptorSet indexing for failures."
                    )

        _t0 = time.perf_counter()
        fused = combine_results(sift_results, deep_results,
                                lambda_param, k=k_neighbors,
                                fusion_method=fusion_method,
                                k_rrf=k_rrf)
        total_t_fusion += (time.perf_counter() - _t0) * 1000
        ranked_ids = [r["id"] for r in fused]

        rel  = relevant_sets[q_idx]
        junk = junk_sets[q_idx]

        aps.append(compute_ap(ranked_ids, rel, junk))
        precs_10.append(compute_precision_at_k(ranked_ids, rel, junk, k=10))
        recs_10.append(compute_recall_at_k(ranked_ids, rel, junk, k=10))
        ndcgs_10.append(compute_ndcg_at_k(ranked_ids, rel, junk, k=10))

    total_time = time.time() - search_start
    qps    = n_queries / total_time
    lat_ms = total_time / n_queries * 1000

    return {
        # Primary metrics — ceiling = 1.0 — LLM-interpretable
        "precision_at_10": float(np.mean(precs_10)),
        "ndcg_at_10":      float(np.mean(ndcgs_10)),
        "recall_at_10":    float(np.mean(recs_10)),
        "mAP":             float(np.mean(aps)),
        # Supplementary
        "qps":             float(qps),
        "latency_avg_ms":  float(lat_ms),
        "num_queries":     n_queries,
        "per_query_ap":    [float(a) for a in aps],
        # Per-component timing (avg ms per query)
        "t_sift_avg_ms":       float(total_t_sift   / n_queries),
        "t_deep_avg_ms":       float(total_t_deep   / n_queries),
        "t_fusion_avg_ms":     float(total_t_fusion / n_queries),
        # Graph metadata
        "constraint_strategy": constraint_strategy,
        "graph_enabled":       use_constraint,
    }


# ================================================================
# CLIP TEXT-QUERY INDEX  (separate 512-d DescriptorSet)
# ================================================================

def setup_clip_index(db, clip_db: np.ndarray,
                     engine: str, engine_params: dict,
                     set_name: str = "CLIP_Set",
                     metadata: List[Dict] = None,
                     use_all_verb_ids: bool = False) -> float:
    """
    Create a MobileCLIP-S2 DescriptorSet (512-d L2) and index all images.

    Used for text-query CLIP experiments: text embeddings are sent as query
    vectors against this set, replacing image[0] queries which are noisy.

    Returns elapsed indexing time in seconds.
    """
    clip_cfg = {
        "name": set_name,
        "dimensions": int(clip_db.shape[1]),
        "metric": "L2",
        "engine": engine,
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
        raise RuntimeError(f"CLIP DescriptorSet '{set_name}' creation failed: {res}")
    print(f"[OK] DescriptorSet '{set_name}' created  dim={clip_db.shape[1]}  metric=L2")

    n   = len(clip_db)
    t0  = time.time()
    fail = 0
    use_graph = (metadata is not None and len(metadata) == n)
    if use_graph:
        print(f"[GRAPH] PMGD metadata provided — storing object_id + verb_id per descriptor")

    def _make_clip_props(i: int) -> Dict:
        props = {"id": i}
        if use_graph:
            m = metadata[i]
            if m["primary_object_id"] >= 0:
                props["object_id"] = int(m["primary_object_id"])
            if m["primary_verb_id"] >= 0:
                props["verb_id"] = int(m["primary_verb_id"])
        return props

    BATCH_PROPS_SIZE = 500  # batch_properties mode: single FAISS insert_descriptor call per batch
    n_indexed = n  # total descriptors indexed (> n when use_all_verb_ids)

    if not use_graph:
        # No per-descriptor metadata — use fast batch_properties path (238x speedup vs individual)
        print(f"Indexing {n:,} CLIP vectors  dim={clip_db.shape[1]}  set={set_name}  (batch_properties={BATCH_PROPS_SIZE})...")
        for start in range(0, n, BATCH_PROPS_SIZE):
            end        = min(start + BATCH_PROPS_SIZE, n)
            batch_size = end - start
            batch_props = [{"id": start + j} for j in range(batch_size)]
            blob        = clip_db[start:end].astype(np.float32).tobytes()
            cmd = {"AddDescriptor": {"set": set_name, "batch_properties": batch_props}}
            res, _ = db.query([cmd], [blob])
            if res:
                fail += (1 if res[0].get("AddDescriptor", {}).get("status", -1) != 0 else 0)
            if start % 50000 == 0 and start > 0:
                print(f"  CLIP progress: {start:,}/{n:,}")

    elif use_all_verb_ids:
        # Condition B: multi-row indexing — one descriptor per (image, verb_id) in all_verb_ids.
        # PMGD has no "in" operator, so the only way to match any verb annotation is to index
        # the same CLIP vector once per verb_id. After retrieval, caller deduplicates by image id.
        descriptor_list = []  # (img_idx, props_dict)
        for i in range(n):
            m   = metadata[i]
            oid = int(m["primary_object_id"]) if m["primary_object_id"] >= 0 else None
            verbs = m.get("all_verb_ids") or []
            if not verbs:  # fallback: should never happen with well-formed metadata
                verbs = [m["primary_verb_id"]] if m["primary_verb_id"] >= 0 else []
            for vid in verbs:
                props = {"id": i}
                if oid is not None:
                    props["object_id"] = oid
                props["verb_id"] = int(vid)
                descriptor_list.append((i, props))

        n_indexed = len(descriptor_list)
        print(f"Indexing {n_indexed:,} descriptors  ({n:,} images × avg {n_indexed/n:.2f} verb_ids)  "
              f"dim={clip_db.shape[1]}  set={set_name}  (batch_properties={BATCH_PROPS_SIZE})...")
        for start in range(0, n_indexed, BATCH_PROPS_SIZE):
            end   = min(start + BATCH_PROPS_SIZE, n_indexed)
            batch = descriptor_list[start:end]
            batch_props = [p for _, p in batch]
            # Each row in the blob must match exactly one entry in batch_props.
            blob = np.stack([clip_db[idx] for idx, _ in batch]).astype(np.float32).tobytes()
            cmd = {"AddDescriptor": {"set": set_name, "batch_properties": batch_props}}
            res, _ = db.query([cmd], [blob])
            if res:
                fail += (1 if res[0].get("AddDescriptor", {}).get("status", -1) != 0 else 0)
            if start % 50000 == 0 and start > 0:
                print(f"  CLIP progress: {start:,}/{n_indexed:,}")

    else:
        # Condition A: one descriptor per image, verb_id = primary_verb_id (original behaviour).
        print(f"Indexing {n:,} CLIP vectors  dim={clip_db.shape[1]}  set={set_name}  (batch_properties={BATCH_PROPS_SIZE}, with metadata)...")
        for start in range(0, n, BATCH_PROPS_SIZE):
            end        = min(start + BATCH_PROPS_SIZE, n)
            batch_size = end - start
            batch_props = [_make_clip_props(start + j) for j in range(batch_size)]
            blob        = clip_db[start:end].astype(np.float32).tobytes()
            cmd = {"AddDescriptor": {"set": set_name, "batch_properties": batch_props}}
            res, _ = db.query([cmd], [blob])
            if res:
                fail += (1 if res[0].get("AddDescriptor", {}).get("status", -1) != 0 else 0)
            if start % 50000 == 0 and start > 0:
                print(f"  CLIP progress: {start:,}/{n:,}")

    elapsed = time.time() - t0
    print(f"  CLIP failures: {fail}/{n_indexed}  ({'FAIL' if fail > 0 else 'OK'})")
    print(f"[OK] '{set_name}' indexed  ({elapsed:.1f}s)")
    if fail > 0:
        raise RuntimeError(f"CLIP indexing failures: {fail}/{n_indexed}")
    return elapsed


# ================================================================
# POSE RERANKING  (two-stage retrieval)
# ================================================================

def rerank_with_pose(deep_results: List[Dict], query_pose: np.ndarray,
                     pose_db: np.ndarray, alpha: float) -> List[Dict]:
    """
    Stage-2 reranker: reorder DINOv2 top-K candidates by pose similarity.

    alpha = pose weight (0 = DINOv2 ranking unchanged, 1 = sorted by pose only).
    final_score = (1-alpha) * deep_norm + alpha * pose_norm   (lower = better)

    Why this beats fusion:
      - Fusion mixes weak pose candidates (mAP=0.0025) into strong DINOv2 pool.
      - Reranking only uses pose to sort within the already-good DINOv2 pool.
      - DINOv2 retrieves the right object; pose then discriminates the verb.
    """
    if not deep_results or alpha == 0.0:
        return deep_results

    cand_ids  = np.array([r["id"] for r in deep_results])
    deep_dist = np.array([r["distance"] for r in deep_results], dtype=np.float32)

    # Normalize DINOv2 distances → [0, 1]
    d_min, d_max = deep_dist.min(), deep_dist.max()
    deep_norm = (deep_dist - d_min) / (d_max - d_min + 1e-10)

    # Pose L2 distance between query and each candidate
    cand_poses = pose_db[cand_ids]                               # [K, 51]
    pose_dist  = np.linalg.norm(cand_poses - query_pose, axis=1)

    # Normalize pose distances → [0, 1]
    p_min, p_max = pose_dist.min(), pose_dist.max()
    pose_norm = (pose_dist - p_min) / (p_max - p_min + 1e-10)

    combined = (1.0 - alpha) * deep_norm + alpha * pose_norm
    order    = np.argsort(combined)
    return [{"id": int(cand_ids[i]), "distance": float(combined[i])} for i in order]


def rerank_with_dinov2(clip_results: List[Dict], query_dinov2: np.ndarray,
                       dinov2_db: np.ndarray, alpha: float = 0.30) -> List[Dict]:
    """
    Stage-2 reranker: normalized weighted sum of CLIP distance + DINOv2 distance.

    final_score = (1-alpha) * clip_norm + alpha * dinov2_norm   (lower = better)

    Why weighted sum beats weighted RRF here:
      CLIP text query on a tight unit-normalized 512-d space produces meaningful
      distance values within the top-K pool (distances vary ~1.30-1.45). This
      fine-grained signal is lost when converting to ranks (RRF). Weighted sum
      preserves it. Ablation confirmed: sum mAP=0.2352 > RRF mAP=0.2334.

    alpha=0.0  → CLIP distance only (no DINOv2)
    alpha=0.30 → optimal from ablation
    alpha=1.0  → DINOv2 distance only (collapses mAP)
    """
    if not clip_results or alpha == 0.0:
        return clip_results

    cand_ids  = np.array([r["id"] for r in clip_results])
    clip_dist = np.array([r["distance"] for r in clip_results], dtype=np.float32)

    # Normalize CLIP distances → [0, 1]
    c_min, c_max = clip_dist.min(), clip_dist.max()
    clip_norm = (clip_dist - c_min) / (c_max - c_min + 1e-10)

    # DINOv2 L2 distance between query and each candidate
    cand_dinov2  = dinov2_db[cand_ids]                              # [K, 1024]
    dinov2_dist  = np.linalg.norm(cand_dinov2 - query_dinov2, axis=1)

    # Normalize DINOv2 distances → [0, 1]
    d_min, d_max = dinov2_dist.min(), dinov2_dist.max()
    dinov2_norm  = (dinov2_dist - d_min) / (d_max - d_min + 1e-10)

    combined = (1.0 - alpha) * clip_norm + alpha * dinov2_norm
    order    = np.argsort(combined)
    return [{"id": int(cand_ids[i]), "distance": float(combined[i])} for i in order]


def run_hico_text_dinov2_rerank(db, text_q: np.ndarray, dinov2_q: np.ndarray,
                                dinov2_db: np.ndarray,
                                relevant_sets: List[Set[int]], junk_sets: List[Set[int]],
                                alpha = 0.30,  # float or np.ndarray of per-query alphas
                                k_neighbors: int = 500,
                                k_rrf: int = 60,      # unused, kept for API compat
                                hoi_keys: List[str] = None,
                                clip_name: str = "CLIP_Set",
                                constraint_strategy: str = "none",
                                obj_vocab: Dict = None,
                                verb_vocab: Dict = None,
                                per_query_constraint: Optional[np.ndarray] = None,
                                per_class_indices: Optional[Dict] = None,
                                use_all_verb_ids: bool = False) -> Dict:
    # per_class_indices: dict obj_id → (vecs (N,D), img_ids (N,)) from build_per_class_clip_indices.
    # When provided, queries with per_query_constraint=='object' bypass VDMS and use exact numpy
    # search against the per-class pool (~600 images). Q1/Q2 still go to VDMS unconstrained.
    # per_query_constraint: str array shape (n_queries,) of "none"/"object"/"verb"/"object_and_verb".
    # When provided, overrides constraint_strategy per query (Q3/Q4 → "object", Q1/Q2 → "none").
    """
    Two-stage retrieval: text query (MobileCLIP) → top-K, then weighted sum rerank.

    Stage 1 (VDMS): text embedding → CLIP_Set → top-K candidates.
    Stage 2 (numpy): weighted sum of normalized CLIP + DINOv2 distances with weight alpha on DINOv2.
    """
    n_queries = len(text_q)
    aps, precs_10, recs_10, ndcgs_10 = [], [], [], []
    total_t_clip   = 0.0
    total_t_rerank = 0.0

    search_start = time.time()

    use_constraint = (
        (constraint_strategy != "none" or per_query_constraint is not None)
        and hoi_keys is not None
        and obj_vocab is not None
        and verb_vocab is not None
    )

    def _build_constraint_clip(q_idx: int) -> Dict:
        if not use_constraint:
            return {}
        if per_query_constraint is not None:
            pq = str(per_query_constraint[q_idx])
            # ACS: hard queries (Q3+Q4) are forced to 'object';
            # easy queries (Q1+Q2, pq='none') use the LLM's global constraint_strategy
            cs = pq if pq != "none" else constraint_strategy
        else:
            cs = constraint_strategy
        if cs == "none":
            return {}
        key = hoi_keys[q_idx]
        obj_name, verb_name = key.split("|", 1)
        c = {}
        if cs in ("object", "object_and_verb"):
            oid = obj_vocab.get(obj_name)
            if oid is not None:
                c["object_id"] = ["==", int(oid)]
        if cs in ("verb", "object_and_verb"):
            vid = verb_vocab.get(verb_name)
            if vid is not None:
                c["verb_id"] = ["==", int(vid)]
        return c

    # Classify queries: per-class numpy (Q3/Q4 when per_class_indices provided) vs VDMS batch.
    use_pc = (per_class_indices is not None
              and per_query_constraint is not None
              and hoi_keys is not None
              and obj_vocab is not None)
    pc_mask = np.zeros(n_queries, dtype=bool)
    if use_pc:
        for q_idx in range(n_queries):
            if str(per_query_constraint[q_idx]) == "object":
                pc_mask[q_idx] = True
    vdms_idxs = [i for i in range(n_queries) if not pc_mask[i]]
    pc_idxs   = [i for i in range(n_queries) if pc_mask[i]]

    # VDMS batch: issue all non-per-class queries in one socket call (amortises ~56ms overhead).
    clip_ops   = []
    clip_blobs = []
    q_to_vdms  = {}  # q_idx → position in clip_ops
    # When use_all_verb_ids=True, images with n verbs appear n times in the DB.
    # Queries whose constraint does NOT include verb_id may return duplicate image ids;
    # we request 3× k and deduplicate after retrieval to guarantee ≥ k unique results.
    # Queries with a verb_id constraint are already unique-per-image — no inflation needed.
    q_needs_dedup = set()  # q_idxs that need dedup (inflated k was used)
    for pos, q_idx in enumerate(vdms_idxs):
        constraint = _build_constraint_clip(q_idx)
        k_q = int(k_neighbors[q_idx]) if hasattr(k_neighbors, '__len__') else int(k_neighbors)
        if use_all_verb_ids and "verb_id" not in constraint:
            q_needs_dedup.add(q_idx)
            k_q = min(k_q * 3, 10000)
        clip_find = {
            "set": clip_name, "k_neighbors": k_q,
            "results": {"list": ["id", "_distance"]}
        }
        if constraint:
            clip_find["constraints"] = constraint
        clip_ops.append({"FindDescriptor": clip_find})
        clip_blobs.append(text_q[q_idx].tobytes())
        q_to_vdms[q_idx] = pos

    _t0 = time.perf_counter()
    batch_res = []
    if clip_ops:
        batch_res, _ = db.query(clip_ops, clip_blobs)
    total_t_clip_vdms = (time.perf_counter() - _t0) * 1000

    # Per-class exact numpy search for Q3/Q4 (instant — ~600 images per class).
    _t0 = time.perf_counter()
    pc_clip_results: Dict[int, List[Dict]] = {}
    for q_idx in pc_idxs:
        obj_name = hoi_keys[q_idx].split("|", 1)[0]
        oid = obj_vocab.get(obj_name)
        cands = []
        if oid is not None and oid in per_class_indices:
            vecs, img_ids = per_class_indices[oid]
            q_vec = text_q[q_idx]
            scores = vecs @ q_vec                   # (N_class,) IP similarity
            k_q = int(k_neighbors[q_idx]) if hasattr(k_neighbors, '__len__') else int(k_neighbors)
            k_q = min(k_q, len(img_ids))
            top = np.argpartition(scores, -k_q)[-k_q:]
            top = top[np.argsort(scores[top])[::-1]]
            cands = [{"id": int(img_ids[i]), "distance": float(1.0 - scores[i])} for i in top]
        pc_clip_results[q_idx] = cands
    total_t_clip_pc = (time.perf_counter() - _t0) * 1000

    total_t_clip = total_t_clip_vdms + total_t_clip_pc

    for q_idx in range(n_queries):
        dq = dinov2_q[q_idx]
        if pc_mask[q_idx]:
            clip_results = pc_clip_results[q_idx]
        else:
            vdms_pos = q_to_vdms[q_idx]
            clip_results = _parse_entities(
                batch_res[vdms_pos] if (batch_res and vdms_pos < len(batch_res)) else None
            )
            # Condition B: deduplicate by image id (each image may appear once per verb_id in DB).
            # Results are sorted ascending by distance; keep the first (best) occurrence per image.
            if use_all_verb_ids and q_idx in q_needs_dedup:
                k_orig = int(k_neighbors[q_idx]) if hasattr(k_neighbors, '__len__') else int(k_neighbors)
                seen: Dict[int, Dict] = {}
                for r in clip_results:
                    if r["id"] not in seen:
                        seen[r["id"]] = r
                clip_results = sorted(seen.values(), key=lambda x: x["distance"])[:k_orig]

        a = float(alpha[q_idx]) if hasattr(alpha, '__len__') else float(alpha)

        if q_idx == 0:
            hoi_label = hoi_keys[0] if hoi_keys else "unknown"
            k_q0 = int(k_neighbors[0]) if hasattr(k_neighbors, '__len__') else int(k_neighbors)
            print(f"[TextDINO q0] HOI={hoi_label}  CLIP={len(clip_results)} candidates  "
                  f"k={k_q0}  alpha={a:.2f}  constraint={constraint_strategy}")

        _t0 = time.perf_counter()
        reranked = rerank_with_dinov2(clip_results, dq, dinov2_db, a)
        total_t_rerank += (time.perf_counter() - _t0) * 1000

        ranked_ids = [r["id"] for r in reranked]
        rel  = relevant_sets[q_idx]
        junk = junk_sets[q_idx]

        aps.append(compute_ap(ranked_ids, rel, junk))
        precs_10.append(compute_precision_at_k(ranked_ids, rel, junk, k=10))
        recs_10.append(compute_recall_at_k(ranked_ids, rel, junk, k=10))
        ndcgs_10.append(compute_ndcg_at_k(ranked_ids, rel, junk, k=10))

    total_time = time.time() - search_start
    return {
        "precision_at_10": float(np.mean(precs_10)),
        "ndcg_at_10":      float(np.mean(ndcgs_10)),
        "recall_at_10":    float(np.mean(recs_10)),
        "mAP":             float(np.mean(aps)),
        "qps":             float(n_queries / total_time),
        "latency_avg_ms":  float(total_time / n_queries * 1000),
        "num_queries":     n_queries,
        "t_clip_avg_ms":       float(total_t_clip   / n_queries),
        "t_rerank_avg_ms":     float(total_t_rerank / n_queries),
        "per_query_ap":        [float(a) for a in aps],
        "constraint_strategy": constraint_strategy,
        "graph_enabled":       use_constraint,
    }


def run_hico_rerank(db, deep_q: np.ndarray, pose_q: np.ndarray,
                    pose_db: np.ndarray,
                    relevant_sets: List[Set[int]], junk_sets: List[Set[int]],
                    alpha: float = 0.3,
                    k_neighbors: int = 500,
                    hoi_keys: List[str] = None,
                    deep_name: str = DEEP_SET_NAME) -> Dict:
    """
    Two-stage HICO-DET retrieval with pose reranking.

    Stage 1 (VDMS): DINOv2 whitened → top-K candidates.
    Stage 2 (Python): rerank by pose L2 similarity (no extra VDMS call).

    alpha: pose weight in reranking (0 = DINOv2 only, 1 = pose only).
    """
    n_queries = len(deep_q)
    aps, precs_10, recs_10, ndcgs_10 = [], [], [], []
    total_t_deep   = 0.0
    total_t_rerank = 0.0

    search_start = time.time()

    for q_idx in range(n_queries):
        dq = deep_q[q_idx]
        pq = pose_q[q_idx]

        deep_find = {
            "set": deep_name, "k_neighbors": k_neighbors,
            "results": {"list": ["id", "_distance"]}
        }

        _t0 = time.perf_counter()
        deep_res, _ = db.query([{"FindDescriptor": deep_find}], [dq.tobytes()])
        total_t_deep += (time.perf_counter() - _t0) * 1000

        deep_results = _parse_entities(deep_res[0] if deep_res else None)

        if q_idx == 0:
            hoi_label = hoi_keys[0] if hoi_keys else "unknown"
            print(f"[Rerank q0] HOI={hoi_label}  DINOv2={len(deep_results)} candidates  alpha={alpha}")

        _t0 = time.perf_counter()
        reranked = rerank_with_pose(deep_results, pq, pose_db, alpha)
        total_t_rerank += (time.perf_counter() - _t0) * 1000

        ranked_ids = [r["id"] for r in reranked]
        rel  = relevant_sets[q_idx]
        junk = junk_sets[q_idx]

        aps.append(compute_ap(ranked_ids, rel, junk))
        precs_10.append(compute_precision_at_k(ranked_ids, rel, junk, k=10))
        recs_10.append(compute_recall_at_k(ranked_ids, rel, junk, k=10))
        ndcgs_10.append(compute_ndcg_at_k(ranked_ids, rel, junk, k=10))

    total_time = time.time() - search_start
    return {
        "precision_at_10": float(np.mean(precs_10)),
        "ndcg_at_10":      float(np.mean(ndcgs_10)),
        "recall_at_10":    float(np.mean(recs_10)),
        "mAP":             float(np.mean(aps)),
        "qps":             float(n_queries / total_time),
        "latency_avg_ms":  float(total_time / n_queries * 1000),
        "num_queries":     n_queries,
        "t_deep_avg_ms":   float(total_t_deep   / n_queries),
        "t_rerank_avg_ms": float(total_t_rerank / n_queries),
        "per_query_ap":    [float(a) for a in aps],
    }


# ================================================================
# FULL BENCHMARK (standalone entry point)
# ================================================================

def run_benchmark(port: int, dataset_dir: str,
                  engine: str = "FaissHNSWFlat",
                  engine_params: dict = None,
                  lambda_param: float = 0.5,
                  fusion_method: str = "weighted_rrf",
                  k_neighbors: int = 500):
    """End-to-end HICO-DET benchmark: connect → load → index → query → metrics."""
    if engine_params is None:
        engine_params = {"M": 32, "efConstruction": 200, "efSearch": 200}

    print(f"Connecting to VDMS on port {port}...")
    db = vdms.vdms()
    db.connect("localhost", port)

    data = load_hico_data(dataset_dir)
    sift_q, deep_q, rel_sets, junk_sets, hoi_keys = build_hoi_queries(
        data["hoi_index"], data["sift_db"], data["deep_db"]
    )

    # PCA whitening: reduces DINOv2 1024-d → 256-d, fixes HNSW incomplete traversal
    # (raw 1024-d returns only ~251/500 neighbors; 256-d returns ~492/500)
    from sklearn.decomposition import PCA
    import time as _time
    _t0 = _time.time()
    pca = PCA(n_components=256, whiten=True, svd_solver="full")
    deep_db_w = pca.fit_transform(data["deep_db"]).astype(np.float32)
    deep_db_w /= np.linalg.norm(deep_db_w, axis=1, keepdims=True).clip(min=1e-10)
    deep_q_w  = pca.transform(deep_q).astype(np.float32)
    deep_q_w  /= np.linalg.norm(deep_q_w, axis=1, keepdims=True).clip(min=1e-10)
    print(f"[Whitening] DINOv2 {data['deep_db'].shape[1]}-d → 256-d  ({_time.time()-_t0:.1f}s)")
    data["deep_db"] = deep_db_w
    deep_q = deep_q_w

    build_t, index_t = setup_hico_index(db, engine, engine_params,
                                         data["sift_db"], data["deep_db"])

    print(f"\nRunning {len(sift_q)} queries "
          f"(method={fusion_method}  λ={lambda_param}  k={k_neighbors})...")
    metrics = run_hico_queries(
        db, sift_q, deep_q, rel_sets, junk_sets,
        lambda_param=lambda_param,
        fusion_method=fusion_method,
        k_neighbors=k_neighbors,
        hoi_keys=hoi_keys,
    )

    print(f"\nResults:")
    print(f"  P@10         = {metrics['precision_at_10']:.4f}")
    print(f"  nDCG@10      = {metrics['ndcg_at_10']:.4f}")
    print(f"  Recall@10    = {metrics['recall_at_10']:.4f}")
    print(f"  mAP          = {metrics['mAP']:.4f}")
    print(f"  QPS          = {metrics['qps']:.1f}")
    print(f"  Lat (ms)     = {metrics['latency_avg_ms']:.1f}")
    print(f"  Index time   = {index_t:.1f}s")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark_hico_det.py <port> <dataset_dir> [lambda] [fusion_method]")
        sys.exit(1)

    port          = int(sys.argv[1])
    dataset_dir   = sys.argv[2]
    lambda_param  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    fusion_method = sys.argv[4] if len(sys.argv) > 4 else "weighted_rrf"

    run_benchmark(port, dataset_dir,
                  engine="FaissHNSWFlat",
                  engine_params={"M": 32, "efConstruction": 200, "efSearch": 200},
                  lambda_param=lambda_param,
                  fusion_method=fusion_method,
                  k_neighbors=500)
