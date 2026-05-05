"""
GLDv2 Index Feature Extraction — CLIP ViT-L/14 + DINOv2 ViT-L/14-reg4
=======================================================================

Reads 762K index images from 100 repacked tar archives:
  /work/hdd/bdjd/gldv2/index/images_NNN.repacked.tar  (N=000..099)

Images are NOT extracted to disk (tarfile streaming) to stay within inode limits.

Outputs (saved to --output-dir):
  gldv2_clip.npy        [N, 768]   float32  CLIP ViT-L/14, L2-normed
  gldv2_dinov2.npy      [N, 1024]  float32  DINOv2 ViT-L/14-reg4 CLS, L2-normed
  gldv2_ids.json        list[str]  image IDs in row order (filename stem)
  gldv2_ids.npy         [N]        same IDs as fixed-width bytes (S16 dtype)
  gldv2_query_clip.npy      [Q, 768]   float32  query image CLIP features
  gldv2_query_dinov2.npy    [Q, 1024]  float32  query image DINOv2 features
  gldv2_query_ids.json      [str]      query IDs in row order

Checkpointing: after each tar a small per-tar npy is written:
  clip_tar_NNN.npy    [~7618, 768]   — one per completed tar
  dinov2_tar_NNN.npy  [~7618, 1024]  — one per completed tar
  checkpoint.json     {"n_done": int, "tars_done": [int,...], "ids": [str,...]}
On restart the script skips already-processed tars and picks up where it left off.
Finalize does a single np.vstack of all per-tar files → gldv2_clip.npy.

Usage:
    # Extract index image features (DB):
    python extract_gldv2_features.py \\
        [--index-dir /work/hdd/bdjd/gldv2/index] \\
        [--output-dir /work/hdd/bdjd/vdms/datasets/gldv2] \\
        [--clip-batch 256] [--dinov2-batch 32] \\
        [--no-clip] [--no-dinov2]

    # Extract query image features (requires gldv2_queries.json from compute_gldv2_gt.py):
    python extract_gldv2_features.py --extract-queries \\
        [--gldv2-dir /work/hdd/bdjd/gldv2] \\
        [--output-dir /work/hdd/bdjd/vdms/datasets/gldv2]

Runtime estimate (A40 40GB):
  Index: CLIP 762K @ ~1000 img/s = ~12 min | DINOv2 762K @ ~50 img/s = ~4.2 h
  Queries: ~1129 images — negligible (~30s)
"""

import argparse
import io
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch                  # module-level import — NameError caught early
from PIL import Image

# ── CLIP ViT-L/14 via open_clip ───────────────────────────────────────────────
try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[WARN] open_clip not found — CLIP extraction will be skipped")

# ── DINOv2 ViT-L/14-reg4 via timm ────────────────────────────────────────────
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    DINOV2_AVAILABLE = True
except ImportError:
    DINOV2_AVAILABLE = False
    print("[WARN] timm not found — DINOv2 extraction will be skipped")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DINOV2_MODEL  = "vit_large_patch14_reg4_dinov2.lvd142m"  # 1024-d
CLIP_MODEL    = "ViT-L-14"
CLIP_PRETRAIN = "openai"
N_TARS        = 100   # images_000.repacked.tar .. images_099.repacked.tar


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_model(device):
    print(f"[CLIP] Loading {CLIP_MODEL} pretrained={CLIP_PRETRAIN} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAIN, device=device
    )
    model.eval()
    print("[CLIP] Loaded. Output dim=768")
    return model, preprocess


def load_dinov2_model(device):
    print(f"[DINOv2] Loading {DINOV2_MODEL} ...")
    model = timm.create_model(DINOV2_MODEL, pretrained=True, num_classes=0)
    model = model.to(device).eval()
    cfg       = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    print("[DINOv2] Loaded. Output dim=1024")
    return model, transform


# ─────────────────────────────────────────────────────────────────────────────
# Batch encoders
# ─────────────────────────────────────────────────────────────────────────────

def encode_clip_batch(model, images, preprocess, device):
    """images: list[PIL.Image] → np.ndarray [B, 768] float32 L2-normed"""
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


def encode_dinov2_batch(model, images, transform, device):
    """images: list[PIL.Image] → np.ndarray [B, 1024] float32 L2-normed"""
    tensors = torch.stack([transform(img) for img in images]).to(device)
    with torch.no_grad():
        feats = model(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Tar streaming
# ─────────────────────────────────────────────────────────────────────────────

def iter_tar_images(tar_path: Path):
    """
    Yield (image_id: str, pil_img: PIL.Image) from a tar file.
    image_id is the filename stem (e.g. "0a1b2c3d4e5f0102").
    Does NOT extract files to disk.
    """
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.name.lower().endswith(".jpg"):
                continue
            img_id = Path(member.name).stem
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                pil_img = Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception as e:
                print(f"  [WARN] {member.name}: {e} — using blank")
                pil_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            yield img_id, pil_img


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(out: Path) -> dict:
    ckpt_path = out / "checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            return json.load(f)
    return {"n_done": 0, "tars_done": [], "ids": []}


def save_checkpoint(out: Path, ckpt: dict,
                    tar_idx: int,
                    clip_arr: np.ndarray | None,
                    dinov2_arr: np.ndarray | None,
                    do_clip: bool, do_dinov2: bool):
    """
    Write per-tar npy files and update checkpoint.json.
    Each tar gets its own small file (clip_tar_NNN.npy / dinov2_tar_NNN.npy).
    No growing-file load — O(1) I/O per checkpoint.
    """
    if do_clip and clip_arr is not None:
        np.save(str(out / f"clip_tar_{tar_idx:03d}.npy"), clip_arr)

    if do_dinov2 and dinov2_arr is not None:
        np.save(str(out / f"dinov2_tar_{tar_idx:03d}.npy"), dinov2_arr)

    with open(out / "checkpoint.json", "w") as f:
        json.dump(ckpt, f)


# ─────────────────────────────────────────────────────────────────────────────
# Finalize: concatenate per-tar files → single output
# ─────────────────────────────────────────────────────────────────────────────

def finalize(out: Path, n_done: int, all_ids: list[str],
             do_clip: bool, do_dinov2: bool):

    # ── IDs ───────────────────────────────────────────────────────────────────
    ids_json = out / "gldv2_ids.json"
    with open(ids_json, "w") as f:
        json.dump(all_ids, f)
    print(f"[OK] IDs saved: {ids_json}  ({len(all_ids):,} entries)")

    # gldv2_ids.npy — fixed-width S16 bytes for fast numpy lookup
    ids_npy = out / "gldv2_ids.npy"
    np.save(str(ids_npy), np.array(all_ids, dtype="S16"))
    print(f"[OK] IDs npy:  {ids_npy}  shape=({len(all_ids)},) dtype=S16")

    # ── CLIP concat ───────────────────────────────────────────────────────────
    if do_clip and CLIP_AVAILABLE:
        tar_files = sorted(out.glob("clip_tar_*.npy"))
        if tar_files:
            print(f"[CLIP] Concatenating {len(tar_files)} tar files ...")
            arr = np.vstack([np.load(str(p)) for p in tar_files])
            final = out / "gldv2_clip.npy"
            np.save(str(final), arr)
            norms = np.linalg.norm(arr, axis=1)
            print(f"[OK] CLIP saved: {final}  shape={arr.shape}")
            print(f"     norms: mean={norms.mean():.6f} ± {norms.std():.6f}")
            nan_count = int(np.isnan(arr).sum())
            if nan_count:
                print(f"  [WARN] CLIP NaN count: {nan_count}")
            for p in tar_files:
                p.unlink()

    # ── DINOv2 concat ─────────────────────────────────────────────────────────
    if do_dinov2 and DINOV2_AVAILABLE:
        tar_files = sorted(out.glob("dinov2_tar_*.npy"))
        if tar_files:
            print(f"[DINOv2] Concatenating {len(tar_files)} tar files ...")
            arr = np.vstack([np.load(str(p)) for p in tar_files])
            final = out / "gldv2_dinov2.npy"
            np.save(str(final), arr)
            norms = np.linalg.norm(arr, axis=1)
            print(f"[OK] DINOv2 saved: {final}  shape={arr.shape}")
            print(f"     norms: mean={norms.mean():.6f} ± {norms.std():.6f}")
            nan_count = int(np.isnan(arr).sum())
            if nan_count:
                print(f"  [WARN] DINOv2 NaN count: {nan_count}")
            for p in tar_files:
                p.unlink()

    # ── Clean checkpoint ──────────────────────────────────────────────────────
    ckpt_path = out / "checkpoint.json"
    if ckpt_path.exists():
        ckpt_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Index image extraction (main loop)
# ─────────────────────────────────────────────────────────────────────────────

def extract(index_dir: Path, out: Path, clip_batch: int, dinov2_batch: int,
            do_clip: bool, do_dinov2: bool, max_tars: int = None):

    out.mkdir(parents=True, exist_ok=True)

    needs_gpu = (do_clip and CLIP_AVAILABLE) or (do_dinov2 and DINOV2_AVAILABLE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if needs_gpu else torch.device("cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Index dir: {index_dir}")
    print(f"[INFO] Output dir: {out}")

    clip_model = clip_preprocess = None
    if do_clip and CLIP_AVAILABLE:
        clip_model, clip_preprocess = load_clip_model(device)

    dinov2_model = dinov2_transform = None
    if do_dinov2 and DINOV2_AVAILABLE:
        dinov2_model, dinov2_transform = load_dinov2_model(device)

    ckpt      = load_checkpoint(out)
    all_ids   = ckpt.get("ids", [])
    tars_done = set(ckpt.get("tars_done", []))
    n_done    = ckpt.get("n_done", 0)

    t_start = time.time()

    for tar_idx in range(N_TARS):
        if max_tars is not None and tar_idx >= max_tars:
            print(f"\n[--max-tars {max_tars}] Stopping early for test.")
            break

        tar_name = f"images_{tar_idx:03d}.repacked.tar"
        tar_path = index_dir / tar_name

        if tar_idx in tars_done:
            print(f"[{tar_idx:3d}/99] {tar_name} — already done, skipping")
            continue

        if not tar_path.exists():
            print(f"[{tar_idx:3d}/99] {tar_name} — NOT FOUND, skipping")
            continue

        print(f"\n[{tar_idx:3d}/99] {tar_name}", flush=True)
        tar_ids    = []
        tar_images = []

        for img_id, pil_img in iter_tar_images(tar_path):
            tar_ids.append(img_id)
            tar_images.append(pil_img)

        N_tar = len(tar_images)
        print(f"  {N_tar} images loaded from tar")

        if N_tar == 0:
            print(f"  [WARN] tar is empty — skipping")
            continue

        tar_clip_arr    = None
        tar_dinov2_arr  = None

        if do_clip and clip_model is not None:
            t0 = time.time()
            batches = []
            for i in range(0, N_tar, clip_batch):
                batches.append(
                    encode_clip_batch(clip_model, tar_images[i:i+clip_batch],
                                      clip_preprocess, device)
                )
            tar_clip_arr = np.vstack(batches)
            elapsed = time.time() - t0
            print(f"  CLIP:   {elapsed:.1f}s  ({N_tar/elapsed:.0f} img/s)", flush=True)

        if do_dinov2 and dinov2_model is not None:
            t0 = time.time()
            batches = []
            for i in range(0, N_tar, dinov2_batch):
                batches.append(
                    encode_dinov2_batch(dinov2_model, tar_images[i:i+dinov2_batch],
                                        dinov2_transform, device)
                )
            tar_dinov2_arr = np.vstack(batches)
            elapsed = time.time() - t0
            print(f"  DINOv2: {elapsed:.1f}s  ({N_tar/elapsed:.0f} img/s)", flush=True)

        all_ids.extend(tar_ids)
        n_done    += N_tar
        tars_done.add(tar_idx)

        elapsed_total = time.time() - t_start
        rate      = n_done / elapsed_total if elapsed_total > 0 else 0
        eta_h     = ((762_000 - n_done) / rate / 3600) if rate > 0 else 0
        print(f"  Progress: {n_done:,}/~762,000  ({n_done/7620:.1f}%)  "
              f"rate={rate:.0f} img/s  ETA={eta_h:.1f}h", flush=True)

        ckpt = {"n_done": n_done, "tars_done": list(tars_done), "ids": all_ids}
        save_checkpoint(out, ckpt, tar_idx, tar_clip_arr, tar_dinov2_arr,
                        do_clip, do_dinov2)
        print(f"  [checkpoint saved]", flush=True)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Extraction complete: {n_done:,} images  ({total_time/3600:.2f}h total)")

    finalize(out, n_done, all_ids, do_clip, do_dinov2)
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Query image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_queries(out: Path, clip_batch: int,
                    dinov2_batch: int, do_clip: bool, do_dinov2: bool):
    """
    Collect and encode the ~1129 GLDv2 public query images.

    GLDv2 test images are only accessible as 20 tar archives on S3:
      https://s3.amazonaws.com/google-landmark/test/images_000.tar .. images_019.tar
    Individual image URLs do not exist (HTTP 404).

    This function downloads each tar to /tmp one at a time, streams images
    from it using iter_tar_images, retains only those whose ID appears in
    gldv2_queries.json, deletes the tar immediately, and stops early once
    all query images are found.

    Requires gldv2_queries.json (produced by compute_gldv2_gt.py) in --output-dir.

    Outputs:
      gldv2_query_clip.npy      [Q, 768]   float32
      gldv2_query_dinov2.npy    [Q, 1024]  float32
      gldv2_query_ids.json      [Q]  query IDs in row order
    """
    import tempfile

    out.mkdir(parents=True, exist_ok=True)

    queries_path = out / "gldv2_queries.json"
    if not queries_path.exists():
        raise FileNotFoundError(
            f"{queries_path} not found. Run compute_gldv2_gt.py first."
        )

    with open(queries_path) as f:
        query_list = json.load(f)

    Q = len(query_list)
    print(f"[Queries] {Q} query images to encode")

    needs_gpu = (do_clip and CLIP_AVAILABLE) or (do_dinov2 and DINOV2_AVAILABLE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if needs_gpu else torch.device("cpu")
    print(f"[INFO] Device: {device}")

    clip_model = clip_preprocess = None
    if do_clip and CLIP_AVAILABLE:
        clip_model, clip_preprocess = load_clip_model(device)

    dinov2_model = dinov2_transform = None
    if do_dinov2 and DINOV2_AVAILABLE:
        dinov2_model, dinov2_transform = load_dinov2_model(device)

    query_images = []
    query_ids    = []
    t0 = time.time()

    # ── Collect query images from test tars ───────────────────────────────────
    S3_TEST_BASE = "https://s3.amazonaws.com/google-landmark/test"
    N_TEST_TARS  = 20

    # Map qid → position so we can preserve query order at the end
    needed: dict = {q["id"]: None for q in query_list}   # qid → PIL image or None

    tmp_dir = Path(tempfile.mkdtemp(prefix="gldv2_test_"))
    print(f"[Queries] Streaming from {N_TEST_TARS} test tars  tmp={tmp_dir}")

    try:
        for tar_idx in range(N_TEST_TARS):
            n_found = sum(1 for v in needed.values() if v is not None)
            if n_found == Q:
                print(f"[Queries] All {Q} images found — stopping at tar {tar_idx:03d}")
                break

            tar_name = f"images_{tar_idx:03d}.tar"
            tar_url  = f"{S3_TEST_BASE}/{tar_name}"
            tmp_tar  = tmp_dir / tar_name

            print(f"[Queries] tar {tar_idx+1}/{N_TEST_TARS}: downloading {tar_name} ...",
                  flush=True)
            t_dl = time.time()
            try:
                urllib.request.urlretrieve(tar_url, str(tmp_tar))
            except Exception as e:
                print(f"  [WARN] Download failed: {e} — skipping this tar")
                continue
            mb = tmp_tar.stat().st_size / 1e6
            print(f"  {mb:.0f} MB in {time.time()-t_dl:.1f}s — scanning ...",
                  flush=True)

            n_before = sum(1 for v in needed.values() if v is not None)
            for img_id, pil_img in iter_tar_images(tmp_tar):
                if img_id in needed and needed[img_id] is None:
                    needed[img_id] = pil_img
            tmp_tar.unlink()   # free disk space immediately

            n_new = sum(1 for v in needed.values() if v is not None) - n_before
            total_found = sum(1 for v in needed.values() if v is not None)
            print(f"  +{n_new} new  ({total_found}/{Q} total)", flush=True)
    finally:
        for f in tmp_dir.iterdir():
            f.unlink()
        tmp_dir.rmdir()

    # Assemble in original query_list order; blank placeholder for any not found
    n_missing = sum(1 for v in needed.values() if v is None)
    if n_missing:
        print(f"[WARN] {n_missing}/{Q} query images not found in any test tar "
              f"— using blank placeholder")

    for q in query_list:
        qid = q["id"]
        query_ids.append(qid)
        img = needed.get(qid) or Image.new("RGB", (224, 224), color=(128, 128, 128))
        query_images.append(img)

    print(f"[Queries] {Q - n_missing}/{Q} real images, {n_missing} blank  "
          f"(elapsed {time.time()-t0:.1f}s)")

    # ── Encode ────────────────────────────────────────────────────────────────
    if do_clip and clip_model is not None:
        t0 = time.time()
        batches = []
        for i in range(0, Q, clip_batch):
            batches.append(
                encode_clip_batch(clip_model, query_images[i:i+clip_batch],
                                  clip_preprocess, device)
            )
        arr = np.vstack(batches)
        path = out / "gldv2_query_clip.npy"
        np.save(str(path), arr)
        norms = np.linalg.norm(arr, axis=1)
        print(f"[OK] Query CLIP saved: {path}  shape={arr.shape}  "
              f"norms mean={norms.mean():.6f}  ({time.time()-t0:.1f}s)")

    if do_dinov2 and dinov2_model is not None:
        t0 = time.time()
        batches = []
        for i in range(0, Q, dinov2_batch):
            batches.append(
                encode_dinov2_batch(dinov2_model, query_images[i:i+dinov2_batch],
                                    dinov2_transform, device)
            )
        arr = np.vstack(batches)
        path = out / "gldv2_query_dinov2.npy"
        np.save(str(path), arr)
        norms = np.linalg.norm(arr, axis=1)
        print(f"[OK] Query DINOv2 saved: {path}  shape={arr.shape}  "
              f"norms mean={norms.mean():.6f}  ({time.time()-t0:.1f}s)")

    # Save ordered query ID list
    qids_path = out / "gldv2_query_ids.json"
    with open(qids_path, "w") as f:
        json.dump(query_ids, f)
    print(f"[OK] Query IDs saved: {qids_path}  ({Q} entries)")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GLDv2 features (CLIP ViT-L/14 + DINOv2 ViT-L/14-reg4)"
    )
    parser.add_argument("--index-dir",   default="/work/hdd/bdjd/gldv2/index",
                        help="Directory containing images_NNN.repacked.tar files")
    parser.add_argument("--output-dir",  default="/work/hdd/bdjd/vdms/datasets/gldv2",
                        help="Where to save .npy feature files")
    parser.add_argument("--gldv2-dir",   default="/work/hdd/bdjd/gldv2",
                        help="GLDv2 root dir (used by --extract-queries)")
    parser.add_argument("--clip-batch",  type=int, default=256)
    parser.add_argument("--dinov2-batch", type=int, default=32)
    parser.add_argument("--no-clip",     action="store_true", help="Skip CLIP extraction")
    parser.add_argument("--no-dinov2",   action="store_true", help="Skip DINOv2 extraction")
    parser.add_argument("--max-tars",    type=int, default=None,
                        help="Stop after processing this many tars (for testing)")
    parser.add_argument("--extract-queries", action="store_true",
                        help="Download and encode query images instead of index images")
    args = parser.parse_args()

    do_clip   = not args.no_clip
    do_dinov2 = not args.no_dinov2
    out       = Path(args.output_dir)

    if args.extract_queries:
        extract_queries(
            out          = out,
            clip_batch   = args.clip_batch,
            dinov2_batch = args.dinov2_batch,
            do_clip      = do_clip,
            do_dinov2    = do_dinov2,
        )
    else:
        extract(
            index_dir    = Path(args.index_dir),
            out          = out,
            clip_batch   = args.clip_batch,
            dinov2_batch = args.dinov2_batch,
            do_clip      = do_clip,
            do_dinov2    = do_dinov2,
            max_tars     = args.max_tars,
        )
