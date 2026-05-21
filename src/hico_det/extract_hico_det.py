"""
Extract HICO-DET features + ground truth for SIFT+CLIP fusion benchmark.
========================================================================

HICO-DET (Human-Object Interaction Detection, Chao et al. WACV 2018)
is a compound-GT dataset where each image is annotated with (object, verb)
pairs: e.g. ("motorcycle", "ride"), ("bottle", "hold"), ("bicycle", "ride").

Why HICO-DET for SIFT+CLIP fusion:
  - Compound GT: query "hold+bottle" requires BOTH object ID (CLIP strength)
    AND interaction pose/geometry (SIFT strength)
  - "hold+bottle" vs "drink+bottle": same object, different arm geometry
  - "hold+bicycle" vs "ride+bicycle": same object, different body configuration
  - 47,776 still images — perfect VDMS target (5K–50K range)
  - 600 HOI categories = 80 COCO objects × 117 verbs
  - No approval needed: HuggingFace zhimeng/hico_det

Source:  https://huggingface.co/datasets/zhimeng/hico_det
Paper:   Chao et al., "Learning to Detect Human-Object Interactions", WACV 2018

SIFT representation: VLAD (Vector of Locally Aggregated Descriptors)
  Previous (WRONG): desc.mean(axis=0) → 128-d mean-pooled, unnormalized
    → distances in 400–600 range → retrieval is random → SIFT mAP ≈ 0.002
  Full-image VLAD (--sift-only): correct distances [0,2] but mAP ≈ 0.0007
    → full image is identical across HOI categories (same scene, different pose)
    → "ride+bicycle" vs "hold+bicycle" look the same at full-image SIFT level
  Bbox-crop VLAD (--sift-crop): crop to union(bbox_human, bbox_object) first
    → SIFT sees only the interaction region → pose geometry becomes discriminative
    → Expected: SIFT-only mAP > 0.05; fusion > DINOv2-only

VLAD pipeline (matches extract_oxford_sift.py):
  1. First pass: extract local SIFT keypoints from all DB images (or crops),
     subsample up to VOCAB_KP_PER_IMG per image, collect ~24M total.
  2. Train MiniBatchKMeans vocabulary: k=64 clusters on sampled keypoints.
  3. Second pass: extract full SIFT keypoints per image (or crop), compute VLAD
     (64×128=8192-d, intra-normalized, power-normalized, L2-normalized).
  4. Fit PCA on all DB VLAD vectors: 8192-d → 512-d (whiten=True). L2-normalize.

Bbox format in parquet objects field: [x1, x2, y1, y2] (MATLAB 1-indexed)
  → PIL crop: (x1-1, y1-1, x2, y2)  (0-indexed, exclusive end)
  → Union bbox: min(x1s)-1, min(y1s)-1, max(x2s), max(y2s), clipped to image size
  → Only use bboxes with invis=0 (visible annotations)

Outputs (saved to output_dir):
  images/               JPEG images (0000000.jpg, 0000001.jpg, ...)
  hico_sift.npy         [N, 512]   float32 — VLAD-SIFT per image (L2-normed)
  hico_bboxes.json      {idx: [x1,y1,x2,y2]} — union bbox per image (PIL coords)
                         (only written by --sift-crop; used for cropped SIFT)
  hico_dinov2.npy       [N, 1024]  float32 — DINOv2 ViT-L/14-reg4 CLS, unit-normed
  hico_labels.npy       [N]        int32   — unused (no single class)
  hico_gt.json          list of N dicts:
    {
      "id":              int         (0-indexed, matches npy row),
      "filename":        str         (e.g. "0000042.jpg"),
      "split":           str         ("train" or "test"),
      "positive_hoi":   [(obj,verb),...],   # HOIs present in image
      "negative_hoi":   [(obj,verb),...],   # HOIs confirmed absent
      "ambiguous_hoi":  [(obj,verb),...],   # uncertain
    }
  hico_hoi_index.json   maps (obj, verb) → list of image IDs with that HOI
                         (precomputed for fast GT lookup during benchmark)

Usage:
    python extract_hico_det.py [--output-dir /path/to/datasets/hico_det]
                                [--no-sift]      (skip SIFT, do CLIP only)
                                [--no-clip]      (skip CLIP, do SIFT only)
                                [--sift-only]    (recompute VLAD-SIFT from existing images,
                                                  skip download/GT/DINOv2)
                                [--sift-crop]    (recompute VLAD-SIFT using bbox crops;
                                                  re-reads parquet for bboxes, skips DINOv2)

Estimated runtime: ~3h on 1 GPU (VLAD vocab+encoding ~1.5h CPU, DINOv2 ~1h GPU)
--sift-crop: ~2.5h CPU-only (parquet bbox read ~10min + VLAD on crops ~2h)
"""

import argparse
import ast
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

# ── GPU-aware DINOv2 import ───────────────────────────────────────────────────
try:
    import torch
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    print("[WARN] timm not found — DINOv2 extraction will be skipped")

# ── HuggingFace download ──────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download

# ── parquet reader ────────────────────────────────────────────────────────────
import pyarrow.parquet as pq

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
HF_REPO     = "zhimeng/hico_det"
TRAIN_PARTS = [f"data/train-{i:05d}-of-00013.parquet" for i in range(13)]
TEST_PARTS  = [f"data/test-{i:05d}-of-00004.parquet"  for i in range(4)]

DINOV2_MODEL    = "vit_large_patch14_reg4_dinov2.lvd142m"  # 1024-d, 4 register tokens
DINOV2_BATCH    = 32    # safe for A40 40GB with ViT-L/14

# VLAD parameters — identical to extract_oxford_sift.py
VLAD_K           = 64    # visual vocabulary size (raw_dim = 64×128 = 8192)
VLAD_DIM_OUT     = 512   # PCA output dimension
VOCAB_KP_PER_IMG = 500   # max keypoints per image for vocabulary sampling

# SPM-VLAD parameters (Spatial Pyramid VLAD — Fix 1 for verb disambiguation)
# Grid levels: 1×1 (global) + 2×2 (quadrants) + 1×3 (horizontal thirds)
# → 8 cells × 8192-d = 65536-d raw → PCA → 512-d
# 1×3 horizontal strips capture: top=head/upper-body, mid=torso, bot=legs
# This is the critical spatial signal for HOI verb discrimination:
#   "ride+bicycle": legs straddling → bottom cells differ from "hold+bicycle"
SPM_GRID_LEVELS  = [(1, 1), (2, 2), (1, 3)]   # (rows, cols) per level
SPM_N_CELLS      = sum(r * c for r, c in SPM_GRID_LEVELS)  # 1+4+3 = 8
SPM_RAW_DIM      = SPM_N_CELLS * VLAD_K * 128              # 8 × 8192 = 65536
SPM_PCA_FIT_N    = 15_000   # fit PCA on random subset (full 47K × 65536 = 12GB)


# ─────────────────────────────────────────────────────────────────────────────
# SIFT helpers — VLAD pipeline (matches extract_oxford_sift.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def extract_local_sift(pil_img: Image.Image) -> np.ndarray | None:
    """
    Extract local SIFT keypoints from a PIL image.
    Returns [N, 128] float32 array, or None if no keypoints found.
    No averaging — all keypoints returned for VLAD encoding.
    """
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()  # unlimited keypoints for best VLAD quality
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        return None
    return desc.astype(np.float32)


def compute_vlad(descriptors: np.ndarray | None, centers: np.ndarray) -> np.ndarray:
    """
    VLAD (Vector of Locally Aggregated Descriptors) encoding.

    Standard VLAD: intra-normalization + power normalization + global L2 normalization.
    Power normalization (signed square root) is critical — adds ~8% mAP on Oxford5K
    by reducing the bursty descriptor effect (Jégou et al. 2012).

    Args:
        descriptors: [N, 128] float32 local SIFT descriptors, or None
        centers:     [k, 128] float32 cluster centers (visual vocabulary)

    Returns: [k * 128] float32 VLAD vector (intra-norm, power-norm, L2-norm)
    """
    k, d = centers.shape
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k * d, dtype=np.float32)

    # Efficient nearest-centroid assignment via dot-product trick:
    # ||x - c||^2 = ||x||^2 - 2*x@c.T + ||c||^2
    x_sq    = (descriptors ** 2).sum(axis=1, keepdims=True)   # [N, 1]
    c_sq    = (centers ** 2).sum(axis=1, keepdims=True).T      # [1, k]
    dot     = descriptors @ centers.T                           # [N, k]
    sq_dist = x_sq - 2.0 * dot + c_sq                         # [N, k]
    assignments = sq_dist.argmin(axis=1)                       # [N]

    # Accumulate residuals per cluster
    vlad = np.zeros((k, d), dtype=np.float32)
    for c_idx in range(k):
        mask = assignments == c_idx
        if mask.any():
            vlad[c_idx] = (descriptors[mask] - centers[c_idx]).sum(axis=0)

    # Intra-normalization: L2-normalize each cluster's accumulated residual
    norms = np.linalg.norm(vlad, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vlad = vlad / norms

    # Power normalization (signed square root): sign(x)*sqrt(|x|)
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))

    # Global L2-normalize
    flat = vlad.flatten()
    n    = np.linalg.norm(flat)
    if n > 0:
        flat = flat / n
    return flat.astype(np.float32)


def compute_vlad_no_l2(descriptors: np.ndarray | None, centers: np.ndarray) -> np.ndarray:
    """
    VLAD with intra-norm + power-norm but WITHOUT final global L2 normalization.
    Used as building block for SPM-VLAD: each cell's raw VLAD is concatenated
    before a single global L2 norm is applied to the full SPM vector.
    """
    k, d = centers.shape
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k * d, dtype=np.float32)

    x_sq    = (descriptors ** 2).sum(axis=1, keepdims=True)
    c_sq    = (centers ** 2).sum(axis=1, keepdims=True).T
    dot     = descriptors @ centers.T
    sq_dist = x_sq - 2.0 * dot + c_sq
    assignments = sq_dist.argmin(axis=1)

    vlad = np.zeros((k, d), dtype=np.float32)
    for c_idx in range(k):
        mask = assignments == c_idx
        if mask.any():
            vlad[c_idx] = (descriptors[mask] - centers[c_idx]).sum(axis=0)

    # Intra-normalization
    norms = np.linalg.norm(vlad, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vlad = vlad / norms

    # Power normalization
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))

    return vlad.flatten().astype(np.float32)


def compute_spm_vlad(pil_img: Image.Image, centers: np.ndarray,
                     sift_obj) -> np.ndarray:
    """
    Spatial Pyramid VLAD (Lazebnik et al. CVPR 2006 + VLAD adaptation).

    Splits the image into SPM_GRID_LEVELS cells, computes VLAD per cell
    (intra-norm + power-norm, NO per-cell L2), concatenates, then applies
    ONE global L2 normalization to the full SPM vector.

    Grid: 1×1 (global) + 2×2 (quadrants) + 1×3 (horizontal thirds)
    = 8 cells × 8192-d = 65536-d → PCA(65536→512) applied externally.

    Why 1×3 horizontal strips are key for HOI:
      TOP   strip = head, upper arms (holds, reaches, eats)
      MID   strip = torso, main arm activity (grabs, pushes)
      BOT   strip = legs, feet, object contact point (ride, kick, step)
    """
    img_arr = np.array(pil_img.convert("RGB"))
    H, W    = img_arr.shape[:2]
    gray    = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    cell_vlads = []
    for (nr, nc) in SPM_GRID_LEVELS:
        for r in range(nr):
            for c in range(nc):
                y1 = int(round(r * H / nr))
                y2 = int(round((r + 1) * H / nr))
                x1 = int(round(c * W / nc))
                x2 = int(round((c + 1) * W / nc))
                cell = gray[y1:y2, x1:x2]

                if cell.size == 0 or min(cell.shape) < 8:
                    cell_vlads.append(np.zeros(VLAD_K * 128, dtype=np.float32))
                    continue

                _, descs = sift_obj.detectAndCompute(cell, None)
                if descs is None or len(descs) == 0:
                    cell_vlads.append(np.zeros(VLAD_K * 128, dtype=np.float32))
                else:
                    cell_vlads.append(
                        compute_vlad_no_l2(descs.astype(np.float32), centers)
                    )

    spm = np.concatenate(cell_vlads)  # [SPM_RAW_DIM]
    n = np.linalg.norm(spm)
    if n > 0:
        spm = spm / n
    return spm.astype(np.float32)


def extract_spm_vlad_sift(img_dir: Path, N: int, output_dir: Path,
                          bboxes: dict | None = None) -> np.ndarray:
    """
    SPM-VLAD pipeline: bbox-crop → spatial pyramid → VLAD per cell → PCA → L2.

    Output: hico_sift_spm.npy [N, 512] — same shape as hico_sift.npy but
    each vector encodes 8-cell SPM (65536-d raw → PCA 512-d).

    PCA is fit on SPM_PCA_FIT_N random images (memory-efficient: full 47K×65536
    matrix = ~12GB, randomized subset avoids OOM while retaining accuracy).
    """
    crop_mode = bboxes is not None
    print(f"\nSPM-VLAD parameters: grid_levels={SPM_GRID_LEVELS}")
    print(f"  n_cells={SPM_N_CELLS}  raw_dim={SPM_RAW_DIM}  out_dim={VLAD_DIM_OUT}")
    print(f"  Crop mode: {'bbox-crop (union human+object bbox)' if crop_mode else 'full image'}")

    # ── Vocab pass: sample keypoints from whole crop (not per-cell) ───────────
    # Vocabulary should cover all HOI regions → collect from full crop
    print(f"\nVocab pass: collecting keypoints (max {VOCAB_KP_PER_IMG}/image)...")
    sift_obj    = cv2.SIFT_create()
    vocab_chunks = []
    total_kp     = 0
    for idx in tqdm(range(N), desc="SPM-Vocab"):
        try:
            pil_img = _load_and_crop(img_dir, idx, bboxes)
        except Exception:
            continue
        gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        del pil_img
        _, desc = sift_obj.detectAndCompute(gray, None)
        if desc is not None and len(desc) > 0:
            desc = desc.astype(np.float32)
            if len(desc) > VOCAB_KP_PER_IMG:
                sel  = np.random.choice(len(desc), VOCAB_KP_PER_IMG, replace=False)
                desc = desc[sel]
            vocab_chunks.append(desc)
            total_kp += len(desc)

    vocab_kp = np.vstack(vocab_chunks)
    del vocab_chunks
    print(f"[OK] Collected {total_kp:,} keypoints from {N} images")

    print(f"\nTraining MiniBatchKMeans vocabulary (k={VLAD_K}, n={len(vocab_kp):,})...")
    kmeans = MiniBatchKMeans(
        n_clusters=VLAD_K, batch_size=10_000, n_init=5,
        max_iter=100, random_state=42, verbose=0
    )
    kmeans.fit(vocab_kp)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"[OK] Vocabulary: centers={centers.shape}  inertia={kmeans.inertia_:.2e}")
    del vocab_kp

    # ── SPM-VLAD pass ─────────────────────────────────────────────────────────
    print(f"\nSPM-VLAD pass: computing {N} SPM vectors ({SPM_N_CELLS} cells each)...")
    spm_raw    = np.zeros((N, SPM_RAW_DIM), dtype=np.float32)
    zero_count = 0
    for idx in tqdm(range(N), desc="SPM-VLAD"):
        try:
            pil_img = _load_and_crop(img_dir, idx, bboxes)
        except Exception:
            zero_count += 1
            continue
        spm_raw[idx] = compute_spm_vlad(pil_img, centers, sift_obj)
        del pil_img
        if np.all(spm_raw[idx] == 0):
            zero_count += 1
    print(f"[OK] SPM-VLAD raw shape={spm_raw.shape}  zero-images={zero_count}")

    # ── PCA on random subset (memory-efficient) ────────────────────────────────
    fit_n = min(SPM_PCA_FIT_N, N)
    fit_idx = np.random.choice(N, fit_n, replace=False)
    print(f"\nFitting PCA on {fit_n} random images: {SPM_RAW_DIM}-d → {VLAD_DIM_OUT}-d ...")
    pca = PCA(n_components=VLAD_DIM_OUT, whiten=True,
              svd_solver='randomized', random_state=42)
    pca.fit(spm_raw[fit_idx])
    explained = pca.explained_variance_ratio_.sum()
    print(f"[OK] PCA fitted. Variance explained: {explained:.3f} "
          f"({explained*100:.1f}% of {SPM_RAW_DIM}-d in {VLAD_DIM_OUT}-d)")

    print(f"Transforming all {N} SPM vectors...")
    sift_arr = pca.transform(spm_raw).astype(np.float32)
    del spm_raw

    norms = np.linalg.norm(sift_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sift_arr = sift_arr / norms

    final_norms = np.linalg.norm(sift_arr, axis=1)
    print(f"[OK] SPM norms: mean={final_norms.mean():.6f} ± {final_norms.std():.6f}  "
          f"(should be ~1.000000 ± 0)")

    spm_path = output_dir / "hico_sift_spm.npy"
    np.save(str(spm_path), sift_arr)
    print(f"[OK] Saved: {spm_path}  shape={sift_arr.shape}")
    print(f"     SPM summary: levels={SPM_GRID_LEVELS}  n_cells={SPM_N_CELLS}  "
          f"raw_dim={SPM_RAW_DIM}  pca_dim={VLAD_DIM_OUT}  pca_var={explained:.3f}")

    return sift_arr


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2_model(device):
    """Load DINOv2 ViT-L/14-reg4 via timm. Returns (model, transform, dim, h, w)."""
    model = timm.create_model(DINOV2_MODEL, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    data_cfg   = resolve_data_config(model.pretrained_cfg)
    transform  = create_transform(**data_cfg)
    input_size = data_cfg.get("input_size", (3, 518, 518))
    h, w = input_size[1], input_size[2]
    with torch.no_grad():
        dummy = torch.zeros(1, 3, h, w).to(device)
        dim = model(dummy).shape[-1]
    print(f"[OK] {DINOV2_MODEL} loaded (dim={dim}, input={h}×{w})")
    return model, transform, dim, h, w


@torch.no_grad()
def dinov2_batch(model, images, transform, device) -> np.ndarray:
    """Extract DINOv2 CLS embeddings for a list of PIL images. Returns [N, 1024] float32 unit-normed."""
    tensors = torch.stack([transform(img) for img in images]).to(device)
    feats = model(tensors).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return feats.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loading
# ─────────────────────────────────────────────────────────────────────────────

def parse_hoi_field(raw) -> list:
    """
    Parse positive_captions / negative_captions field.
    In parquet: stored as string repr of list of tuples, e.g.
      "[('motorcycle', 'ride'), ('motorcycle', 'sit_on')]"
    Returns list of [obj, verb] pairs.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [[str(o), str(v)] for o, v in raw]
    try:
        parsed = ast.literal_eval(raw)
        return [[str(o), str(v)] for o, v in parsed]
    except Exception:
        return []


def load_parquet_rows(parquet_path: str, split: str) -> list:
    """
    Load rows from one parquet file.
    Returns list of dicts: {image_bytes, split, positive_hoi, negative_hoi, ambiguous_hoi}
    """
    table = pq.read_table(parquet_path)
    d     = table.to_pydict()
    rows  = []
    n     = len(d["image"])
    for i in range(n):
        img_dict  = d["image"][i]
        img_bytes = img_dict.get("bytes") if isinstance(img_dict, dict) else None
        rows.append({
            "image_bytes":   img_bytes,
            "split":         split,
            "positive_hoi":  parse_hoi_field(d["positive_captions"][i]),
            "negative_hoi":  parse_hoi_field(d["negative_captions"][i]),
            "ambiguous_hoi": parse_hoi_field(d.get("ambiguous_captions", [None]*n)[i]),
        })
    return rows


def load_parquet_bboxes(output_dir: Path) -> dict:
    """
    Read all parquet files and compute union bbox per image from HOI annotations.

    Parquet objects field: list of dicts with keys:
      bbox_human: [x1, x2, y1, y2]  (MATLAB 1-indexed)
      bbox_object:[x1, x2, y1, y2]
      invis: 0 (visible) or 1 (occluded/invisible — skip)

    Returns: dict {image_idx (int) → [x1, y1, x2, y2]} in PIL 0-indexed coords.
    Images where ALL annotations are invisible → not in dict (use full image).

    Also saves hico_bboxes.json to output_dir for reuse.
    """
    bbox_path = output_dir / "hico_bboxes.json"

    print("\nReading parquet files for bbox annotations...")
    bboxes  = {}   # idx → [x1_pil, y1_pil, x2_pil, y2_pil]
    img_idx = 0

    all_parts = (
        [(f, "train") for f in TRAIN_PARTS] +
        [(f, "test")  for f in TEST_PARTS]
    )
    for fname, split in tqdm(all_parts, desc="Parquet-Bbox"):
        path  = hf_hub_download(HF_REPO, fname, repo_type="dataset")
        table = pq.read_table(path)
        d     = table.to_pydict()
        n     = len(d["image"])
        sizes = d.get("size", [[0, 0, 0]] * n)

        for i in range(n):
            # Parse objects field
            raw = d["objects"][i] if "objects" in d else None
            if raw is None:
                img_idx += 1
                continue
            if isinstance(raw, str):
                try:
                    objects = ast.literal_eval(raw)
                except Exception:
                    img_idx += 1
                    continue
            else:
                objects = raw if isinstance(raw, list) else []

            # Image dimensions for clipping
            sz = sizes[i] if isinstance(sizes[i], list) else [0, 0, 0]
            W  = int(sz[0]) if len(sz) > 0 else 0
            H  = int(sz[1]) if len(sz) > 1 else 0

            # Collect all visible bbox corners (MATLAB [x1,x2,y1,y2] → PIL [x1,y1,x2,y2])
            xs1, ys1, xs2, ys2 = [], [], [], []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if int(obj.get("invis", 0)) == 1:
                    continue  # skip invisible/occluded
                for bbox_key in ("bbox_human", "bbox_object"):
                    b = obj.get(bbox_key)
                    if b is None or len(b) < 4:
                        continue
                    # MATLAB format: [x1, x2, y1, y2] (1-indexed)
                    bx1, bx2, by1, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    if bx1 <= 0 and bx2 <= 0:
                        continue  # invalid bbox
                    xs1.append(bx1); xs2.append(bx2)
                    ys1.append(by1); ys2.append(by2)

            if xs1:
                # Union in PIL coords (convert MATLAB 1-indexed to 0-indexed)
                x1_pil = max(0, min(xs1) - 1)
                y1_pil = max(0, min(ys1) - 1)
                x2_pil = min(W, max(xs2)) if W > 0 else max(xs2)
                y2_pil = min(H, max(ys2)) if H > 0 else max(ys2)
                # Sanity: ensure crop has positive area
                if x2_pil > x1_pil + 4 and y2_pil > y1_pil + 4:
                    bboxes[img_idx] = [x1_pil, y1_pil, x2_pil, y2_pil]

            img_idx += 1

    n_with_bbox = len(bboxes)
    n_total     = img_idx
    print(f"[OK] Bboxes: {n_with_bbox}/{n_total} images have valid union bbox "
          f"({100*n_with_bbox/n_total:.1f}%); "
          f"{n_total - n_with_bbox} will use full image")

    # Save for reuse
    with open(bbox_path, "w") as f:
        json.dump({str(k): v for k, v in bboxes.items()}, f)
    print(f"[OK] Saved bboxes → {bbox_path}")

    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# VLAD-SIFT extraction from saved images (used both in full run and --sift-only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_crop(img_dir: Path, idx: int, bboxes: dict | None) -> Image.Image:
    """
    Load image idx from img_dir. If bboxes provided and idx has a valid bbox,
    crop to the union of human+object bounding boxes. Otherwise return full image.
    """
    pil_img = Image.open(img_dir / f"{idx:07d}.jpg").convert("RGB")
    if bboxes is not None and idx in bboxes:
        x1, y1, x2, y2 = bboxes[idx]
        pil_img = pil_img.crop((x1, y1, x2, y2))
    return pil_img


def extract_vlad_sift(img_dir: Path, N: int, output_dir: Path,
                      bboxes: dict | None = None) -> np.ndarray:
    """
    Run full VLAD pipeline over saved JPEG images in img_dir.

    If bboxes is provided (dict: idx → [x1,y1,x2,y2] in PIL coords), each image
    is cropped to its union human+object bbox before SIFT extraction. Images
    without a bbox entry use the full image.

    Steps:
      1. Vocab pass: sample up to VOCAB_KP_PER_IMG keypoints per image (or crop),
         train MiniBatchKMeans(k=VLAD_K).
      2. VLAD pass: compute raw 8192-d VLAD per image (or crop).
      3. PCA compress 8192-d → VLAD_DIM_OUT-d (whiten=True). L2-normalize.

    Returns: sift_arr [N, VLAD_DIM_OUT] float32
    """
    raw_dim   = VLAD_K * 128  # 8192
    crop_mode = bboxes is not None
    print(f"\nVLAD parameters: k={VLAD_K}, raw_dim={raw_dim}, out_dim={VLAD_DIM_OUT}")
    print(f"Crop mode: {'bbox-crop (union human+object bbox)' if crop_mode else 'full image'}")

    # ── Vocab pass ────────────────────────────────────────────────────────────
    print(f"\nVocab pass: collecting keypoints (max {VOCAB_KP_PER_IMG}/image)...")
    vocab_chunks = []
    total_kp     = 0
    for idx in tqdm(range(N), desc="Vocab-Pass"):
        try:
            pil_img = _load_and_crop(img_dir, idx, bboxes)
        except Exception:
            continue
        desc = extract_local_sift(pil_img)
        del pil_img
        if desc is not None:
            if len(desc) > VOCAB_KP_PER_IMG:
                sel  = np.random.choice(len(desc), VOCAB_KP_PER_IMG, replace=False)
                desc = desc[sel]
            vocab_chunks.append(desc)
            total_kp += len(desc)

    vocab_kp = np.vstack(vocab_chunks)
    del vocab_chunks
    print(f"[OK] Collected {total_kp:,} keypoints from {N} images")

    # ── Train vocabulary ──────────────────────────────────────────────────────
    print(f"\nTraining MiniBatchKMeans vocabulary (k={VLAD_K}, n={len(vocab_kp):,})...")
    kmeans = MiniBatchKMeans(
        n_clusters=VLAD_K, batch_size=10_000, n_init=5,
        max_iter=100, random_state=42, verbose=0
    )
    kmeans.fit(vocab_kp)
    centers = kmeans.cluster_centers_.astype(np.float32)
    print(f"[OK] Vocabulary: centers shape={centers.shape}  inertia={kmeans.inertia_:.2e}")
    del vocab_kp

    # ── VLAD pass ─────────────────────────────────────────────────────────────
    print(f"\nVLAD pass: computing {N} VLAD vectors...")
    vlad_raw   = np.zeros((N, raw_dim), dtype=np.float32)
    zero_count = 0
    for idx in tqdm(range(N), desc="VLAD-Pass"):
        try:
            pil_img = _load_and_crop(img_dir, idx, bboxes)
        except Exception:
            zero_count += 1
            continue
        desc = extract_local_sift(pil_img)
        del pil_img
        vlad_raw[idx] = compute_vlad(desc, centers)
        if desc is None:
            zero_count += 1
    print(f"[OK] Raw VLAD shape={vlad_raw.shape}  zero-images={zero_count}")

    # ── PCA compress ──────────────────────────────────────────────────────────
    print(f"\nFitting PCA: {raw_dim}-d → {VLAD_DIM_OUT}-d on {N} vectors...")
    pca = PCA(n_components=VLAD_DIM_OUT, whiten=True, random_state=42)
    pca.fit(vlad_raw)
    explained = pca.explained_variance_ratio_.sum()
    print(f"[OK] PCA fitted. Variance explained: {explained:.3f} "
          f"({explained*100:.1f}% of {raw_dim}-d in {VLAD_DIM_OUT}-d)")

    sift_arr = pca.transform(vlad_raw).astype(np.float32)
    del vlad_raw

    # L2-normalize
    norms = np.linalg.norm(sift_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sift_arr = sift_arr / norms

    # Verify
    final_norms = np.linalg.norm(sift_arr, axis=1)
    print(f"[OK] SIFT norms: mean={final_norms.mean():.6f} ± {final_norms.std():.6f}  "
          f"(should be ~1.000000 ± 0)")

    # Save
    sift_path = output_dir / "hico_sift.npy"
    np.save(str(sift_path), sift_arr)
    print(f"[OK] Saved: {sift_path}  shape={sift_arr.shape}")
    print(f"     VLAD summary: k={VLAD_K}  raw_dim={raw_dim}  "
          f"pca_dim={VLAD_DIM_OUT}  pca_explained={explained:.3f}")

    return sift_arr


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract(output_dir: str, do_sift: bool = True, do_clip: bool = True,
            sift_only: bool = False, sift_crop: bool = False, sift_spm: bool = False,
            dinov2_crop: bool = False):
    out     = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ── --dinov2-crop: re-extract DINOv2 on union(person, object) bbox crops ─────
    if dinov2_crop:
        if not DINOV2_AVAILABLE:
            raise RuntimeError("--dinov2-crop requires timm. Install: pip install timm")
        jpgs = sorted(img_dir.glob("*.jpg"))
        N    = len(jpgs)
        if N == 0:
            raise RuntimeError(f"--dinov2-crop: no images in {img_dir}. Run full extraction first.")

        bbox_path = out / "hico_bboxes.json"
        if bbox_path.exists():
            print(f"[OK] Loading cached bboxes from {bbox_path}")
            with open(bbox_path) as f:
                raw = json.load(f)
            bboxes = {int(k): v for k, v in raw.items()}
        else:
            bboxes = load_parquet_bboxes(out)

        n_with_bbox = sum(1 for i in range(N) if i in bboxes)
        print(f"[--dinov2-crop] {N} images, {n_with_bbox} with HOI bbox "
              f"({100*n_with_bbox/N:.1f}%) — rest use full image")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] DINOv2 device: {device}")
        model, transform, dim, h, w = load_dinov2_model(device)

        crop_path  = out / "hico_dinov2_crop.npy"
        dinov2_arr = np.zeros((N, dim), dtype=np.float32)

        for start in tqdm(range(0, N, DINOV2_BATCH), desc="DINOv2-crop batches"):
            end   = min(start + DINOV2_BATCH, N)
            batch = []
            for i in range(start, end):
                try:
                    pil_img = _load_and_crop(img_dir, i, bboxes)
                except Exception:
                    pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                batch.append(pil_img)
            feats = dinov2_batch(model, batch, transform, device)
            dinov2_arr[start:end] = feats

        np.save(str(crop_path), dinov2_arr)
        norms = np.linalg.norm(dinov2_arr, axis=1)
        print(f"[OK] DINOv2 crop saved: {crop_path}  shape={dinov2_arr.shape}")
        print(f"     norms: mean={norms.mean():.6f} ± {norms.std():.6f}  "
              f"(should be ~1.000000 ± 0)")
        nan_count = np.isnan(dinov2_arr).sum()
        inf_count = np.isinf(dinov2_arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  [WARN] NaN={nan_count}  Inf={inf_count}")
        else:
            print("  [OK] clean: zero NaN, zero Inf")
        return

    # ── --sift-only: recompute VLAD-SIFT from existing images, skip everything else ──
    if sift_only:
        jpgs = sorted(img_dir.glob("*.jpg"))
        N    = len(jpgs)
        if N == 0:
            raise RuntimeError(f"--sift-only: no images found in {img_dir}. "
                               "Run without --sift-only first.")
        print(f"[--sift-only] Found {N} images in {img_dir}. Recomputing VLAD-SIFT...")
        extract_vlad_sift(img_dir, N, out)
        return

    # ── --sift-spm: SPM-VLAD on bbox crops (spatial pyramid, 8 cells) ────────
    if sift_spm:
        jpgs = sorted(img_dir.glob("*.jpg"))
        N    = len(jpgs)
        if N == 0:
            raise RuntimeError(f"--sift-spm: no images found in {img_dir}. "
                               "Run without --sift-spm first to download images.")

        print(f"[--sift-spm] Found {N} images in {img_dir}.")
        print("Step 1: Loading HOI bounding boxes...")

        bbox_path = out / "hico_bboxes.json"
        if bbox_path.exists():
            print(f"[OK] Loading cached bboxes from {bbox_path}")
            with open(bbox_path) as f:
                raw = json.load(f)
            bboxes = {int(k): v for k, v in raw.items()}
        else:
            bboxes = load_parquet_bboxes(out)

        print(f"\nStep 2: Computing SPM-VLAD with bbox crops...")
        print(f"  Grid levels: {SPM_GRID_LEVELS}  ({SPM_N_CELLS} cells, {SPM_RAW_DIM}-d raw)")
        extract_spm_vlad_sift(img_dir, N, out, bboxes=bboxes)
        return

    # ── --sift-crop: re-extract VLAD-SIFT using human+object bbox crops ──────
    if sift_crop:
        jpgs = sorted(img_dir.glob("*.jpg"))
        N    = len(jpgs)
        if N == 0:
            raise RuntimeError(f"--sift-crop: no images found in {img_dir}. "
                               "Run without --sift-crop first to download images.")

        print(f"[--sift-crop] Found {N} images in {img_dir}.")
        print("Step 1: Reading parquet files to extract HOI bounding boxes...")

        # Check if bboxes already saved
        bbox_path = out / "hico_bboxes.json"
        if bbox_path.exists():
            print(f"[OK] Loading cached bboxes from {bbox_path}")
            with open(bbox_path) as f:
                raw = json.load(f)
            bboxes = {int(k): v for k, v in raw.items()}
        else:
            bboxes = load_parquet_bboxes(out)

        print(f"\nStep 2: Recomputing VLAD-SIFT with bbox crops...")
        extract_vlad_sift(img_dir, N, out, bboxes=bboxes)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if (DINOV2_AVAILABLE and do_clip) else None
    print(f"[INFO] Device: {device}")

    # ── 1. Download + collect all parquet rows ────────────────────────────────
    print("\n[1/4] Downloading HICO-DET parquet files from HuggingFace...")
    all_rows = []
    for fname in TRAIN_PARTS:
        path = hf_hub_download(HF_REPO, fname, repo_type="dataset")
        rows = load_parquet_rows(path, "train")
        all_rows.extend(rows)
        print(f"  train {fname.split('/')[-1]}: {len(rows)} rows  (total {len(all_rows)})")

    for fname in TEST_PARTS:
        path = hf_hub_download(HF_REPO, fname, repo_type="dataset")
        rows = load_parquet_rows(path, "test")
        all_rows.extend(rows)
        print(f"  test  {fname.split('/')[-1]}: {len(rows)} rows  (total {len(all_rows)})")

    N = len(all_rows)
    print(f"[OK] {N} total images loaded")

    # ── 2. Save images + build GT ─────────────────────────────────────────────
    # Memory fix: do NOT keep all PIL images in RAM (47K images = OOM at 64GB).
    # CLIP will load images from disk in batches in step 4b.
    print(f"\n[2/4] Saving images to {img_dir} and building GT...")
    gt_list = []

    for idx, row in enumerate(tqdm(all_rows, desc="Saving+GT")):
        fname = f"{idx:07d}.jpg"
        fpath = img_dir / fname

        # Decode image
        if row["image_bytes"]:
            try:
                pil_img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
            except Exception as e:
                print(f"  [WARN] image {idx} decode failed: {e} — using blank")
                pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        else:
            pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Save as JPEG (only if not already saved, for reruns)
        if not fpath.exists():
            pil_img.save(str(fpath), "JPEG", quality=95)

        del pil_img

        gt_list.append({
            "id":            idx,
            "filename":      fname,
            "split":         row["split"],
            "positive_hoi":  row["positive_hoi"],
            "negative_hoi":  row["negative_hoi"],
            "ambiguous_hoi": row["ambiguous_hoi"],
        })

    print(f"[OK] {N} images saved")

    # ── 3. Build HOI index: (obj, verb) → [image_ids] ────────────────────────
    print("\n[3/4] Building HOI index...")
    hoi_index: dict = {}
    for entry in gt_list:
        for obj, verb in entry["positive_hoi"]:
            key = f"{obj}|{verb}"
            if key not in hoi_index:
                hoi_index[key] = []
            hoi_index[key].append(entry["id"])

    n_hoi  = len(hoi_index)
    sizes  = [len(v) for v in hoi_index.values()]
    print(f"[OK] {n_hoi} unique HOI pairs")
    print(f"     images/pair: min={min(sizes)}  max={max(sizes)}  mean={np.mean(sizes):.1f}")

    # Save GT and HOI index
    gt_path  = out / "hico_gt.json"
    idx_path = out / "hico_hoi_index.json"
    with open(gt_path,  "w") as f: json.dump(gt_list,   f)
    with open(idx_path, "w") as f: json.dump(hoi_index, f)
    print(f"[OK] GT saved to {gt_path}")
    print(f"[OK] HOI index saved to {idx_path}")

    # ── 4a. VLAD-SIFT extraction ──────────────────────────────────────────────
    if do_sift:
        print("\n[4a/4] VLAD-SIFT extraction (vocab pass → VLAD pass → PCA)...")
        extract_vlad_sift(img_dir, N, out)
    else:
        print("[SKIP] SIFT extraction skipped (--no-sift)")

    # ── 4b. DINOv2 — load images from disk in batches ────────────────────────
    dinov2_path = out / "hico_dinov2.npy"
    if do_clip and DINOV2_AVAILABLE:
        print(f"\n[4b/4] Extracting DINOv2 for {N} images on {device}...")
        model, transform, dim, h, w = load_dinov2_model(device)
        dinov2_arr = np.zeros((N, dim), dtype=np.float32)

        for start in tqdm(range(0, N, DINOV2_BATCH), desc="DINOv2 batches"):
            end   = min(start + DINOV2_BATCH, N)
            batch = []
            for i in range(start, end):
                fpath = img_dir / f"{i:07d}.jpg"
                try:
                    batch.append(Image.open(fpath).convert("RGB"))
                except Exception:
                    batch.append(Image.new("RGB", (224, 224), color=(128, 128, 128)))
            feats = dinov2_batch(model, batch, transform, device)
            dinov2_arr[start:end] = feats

        np.save(str(dinov2_path), dinov2_arr)
        norms = np.linalg.norm(dinov2_arr, axis=1)
        print(f"[OK] DINOv2 saved: {dinov2_arr.shape}")
        print(f"     norms: mean={norms.mean():.6f} ± {norms.std():.6f}  (should be ~1.000000 ± 0)")

        nan_count = np.isnan(dinov2_arr).sum()
        inf_count = np.isinf(dinov2_arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  [WARN] DINOv2 NaN={nan_count}  Inf={inf_count}")
        else:
            print("  [OK] DINOv2 clean: zero NaN, zero Inf")
    else:
        if not do_clip:
            print("[SKIP] DINOv2 extraction skipped (--no-clip)")
        else:
            print("[SKIP] DINOv2 extraction skipped (timm not available)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"HICO-DET extraction complete")
    print(f"  Total images:     {N}")
    print(f"  Train:            {sum(1 for e in gt_list if e['split']=='train')}")
    print(f"  Test:             {sum(1 for e in gt_list if e['split']=='test')}")
    print(f"  Unique HOI pairs: {n_hoi}")
    print(f"  Images dir:       {img_dir}")
    print(f"  GT file:          {gt_path}")
    print(f"  HOI index:        {idx_path}")
    if do_sift:
        print(f"  SIFT:             {out}/hico_sift.npy  shape=[{N},{VLAD_DIM_OUT}]  (VLAD-PCA-L2)")
    if do_clip and DINOV2_AVAILABLE:
        print(f"  DINOv2:           {dinov2_path}  shape=[{N},1024]")
    print(f"{'='*60}")

    print(f"\nTop-10 most common HOI pairs:")
    for key, ids in sorted(hoi_index.items(), key=lambda x: -len(x[1]))[:10]:
        obj, verb = key.split("|")
        print(f"  {verb:20s} + {obj:20s} : {len(ids):5d} images")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HICO-DET SIFT+CLIP features")
    parser.add_argument(
        "--output-dir",
        default="/path/to/datasets/hico_det",
        help="Output directory for images, features, and GT"
    )
    parser.add_argument("--no-sift",   action="store_true", help="Skip SIFT extraction")
    parser.add_argument("--no-clip",   action="store_true", help="Skip CLIP/DINOv2 extraction")
    parser.add_argument("--sift-only", action="store_true",
                        help="Recompute VLAD-SIFT only from existing images; skip download/GT/DINOv2")
    parser.add_argument("--sift-crop", action="store_true",
                        help="Recompute VLAD-SIFT using union(bbox_human, bbox_object) crops; "
                             "re-reads parquet for bboxes (cached), skips GT/DINOv2. "
                             "Fixes full-image VLAD mAP≈0 by focusing on interaction region.")
    parser.add_argument("--sift-spm", action="store_true",
                        help="Compute SPM-VLAD on bbox crops: spatial pyramid (1×1 + 2×2 + 1×3 = "
                             "8 cells), VLAD per cell, concatenate, PCA(65536→512), L2-norm. "
                             "Fixes plain VLAD's loss of spatial layout — critical for HOI verb "
                             "disambiguation ('ride+bicycle' legs straddling vs 'hold+bicycle'). "
                             "Output: hico_sift_spm.npy (does NOT overwrite hico_sift.npy).")
    parser.add_argument("--dinov2-crop", action="store_true",
                        help="Re-extract DINOv2 using union(bbox_human, bbox_object) crops. "
                             "Focuses semantic features on the interaction region (person+object) "
                             "instead of full image background. Reads hico_bboxes.json (cached). "
                             "Output: hico_dinov2_crop.npy [N, 1024] — preferred over "
                             "hico_dinov2.npy in benchmark_hico_det.py when present. "
                             "Estimated runtime: ~1h on 1 A40 GPU (47,776 images, batch=32).")
    args = parser.parse_args()

    t0 = time.time()
    extract(args.output_dir,
            do_sift=not args.no_sift,
            do_clip=not args.no_clip,
            sift_only=args.sift_only,
            sift_crop=args.sift_crop,
            sift_spm=args.sift_spm,
            dinov2_crop=args.dinov2_crop)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")