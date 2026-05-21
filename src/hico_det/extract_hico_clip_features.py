"""
Extract CLIP image and text embeddings for HICO-DET.

Run extract_hico_det.py first to produce images/ and hico_gt.json,
and extract_hico_text_queries.py to produce hico_hoi_index.json.

Outputs (saved to --dataset-dir):
  hico_clip.npy                     [N, 512]      float32  MobileCLIP-S2 image embeddings
  hico_clipvitl14_db.npy            [N, 768]      float32  CLIP ViT-L/14 image embeddings
  hico_clipvitl14_text_queries.npy  [N_hoi, 768]  float32  CLIP ViT-L/14 text embeddings

hico_clip_text_queries.npy (MobileCLIP-S2 text, 512-d) is produced by
extract_hico_text_queries.py and is not repeated here.

Usage:
  python extract_hico_clip_features.py
  python extract_hico_clip_features.py --dataset-dir /path/to/hico_det
  python extract_hico_clip_features.py --backbone vitl14        # skip MobileCLIP-S2
  python extract_hico_clip_features.py --backbone mobileclip_s2 # skip ViT-L/14
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_BATCH = 128  # safe for A40 40GB with ViT-L/14 at 224×224


def build_prompt(obj: str, verb: str) -> str:
    o = obj.replace("_", " ")
    if verb == "no_interaction":
        return f"a person near a {o}"
    v = verb.replace("_", " ")
    return f"a person {v} a {o}"


def extract_image_embeddings(img_dir: Path, N: int, model, preprocess,
                              device: torch.device) -> np.ndarray:
    arr = np.zeros((N, model.visual.output_dim), dtype=np.float32)
    for start in tqdm(range(0, N, IMAGE_BATCH), desc="  image batches"):
        end = min(start + IMAGE_BATCH, N)
        batch = []
        for i in range(start, end):
            fpath = img_dir / f"{i:07d}.jpg"
            try:
                img = preprocess(Image.open(fpath).convert("RGB"))
            except Exception:
                img = preprocess(Image.new("RGB", (224, 224), color=(128, 128, 128)))
            batch.append(img)
        tensors = torch.stack(batch).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensors)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        arr[start:end] = emb.cpu().numpy().astype(np.float32)
    return arr


def extract_text_embeddings(prompts: list, model, tokenizer,
                             device: torch.device) -> np.ndarray:
    BATCH = 64
    all_embs = []
    for start in tqdm(range(0, len(prompts), BATCH), desc="  text batches"):
        batch = prompts[start:start + BATCH]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(all_embs, axis=0)


def run(dataset_dir: str, backbone: str):
    import open_clip

    d = Path(dataset_dir)
    img_dir = d / "images"
    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found. Run extract_hico_det.py first.", file=sys.stderr)
        sys.exit(1)

    jpgs = sorted(img_dir.glob("*.jpg"))
    N = len(jpgs)
    if N == 0:
        print(f"ERROR: no .jpg files in {img_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] Found {N} images in {img_dir}")

    hoi_index_path = d / "hico_hoi_index.json"
    if not hoi_index_path.exists():
        print(f"ERROR: {hoi_index_path} not found. Run extract_hico_det.py first.", file=sys.stderr)
        sys.exit(1)
    with open(hoi_index_path) as f:
        hoi_index = json.load(f)
    hoi_keys = list(hoi_index.keys())
    prompts  = [build_prompt(*k.split("|")) for k in hoi_keys]
    print(f"[OK] HOI categories: {len(hoi_keys)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Device: {device}")

    do_mobile = backbone in ("mobileclip_s2", "all")
    do_vitl14 = backbone in ("vitl14", "all")

    # ── MobileCLIP-S2 image embeddings ────────────────────────────────────────
    if do_mobile:
        out_path = d / "hico_clip.npy"
        print(f"\n[1/2] MobileCLIP-S2 image embeddings → {out_path}")
        model, preprocess, _ = open_clip.create_model_and_transforms(
            "MobileCLIP-S2", pretrained="datacompdr"
        )
        model = model.eval().to(device)
        arr = extract_image_embeddings(img_dir, N, model, preprocess, device)
        np.save(str(out_path), arr)
        norms = np.linalg.norm(arr, axis=1)
        print(f"[OK] Saved: shape={arr.shape}  norms mean={norms.mean():.6f} ± {norms.std():.2e}")
        del model

    # ── CLIP ViT-L/14 image + text embeddings ─────────────────────────────────
    if do_vitl14:
        print(f"\n[2/2] CLIP ViT-L/14 image + text embeddings")
        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        model = model.eval().to(device)
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        out_img = d / "hico_clipvitl14_db.npy"
        print(f"  image → {out_img}")
        arr = extract_image_embeddings(img_dir, N, model, preprocess, device)
        np.save(str(out_img), arr)
        norms = np.linalg.norm(arr, axis=1)
        print(f"[OK] Saved: shape={arr.shape}  norms mean={norms.mean():.6f} ± {norms.std():.2e}")

        out_txt = d / "hico_clipvitl14_text_queries.npy"
        print(f"  text  → {out_txt}")
        txt = extract_text_embeddings(prompts, model, tokenizer, device)
        np.save(str(out_txt), txt)
        norms = np.linalg.norm(txt, axis=1)
        print(f"[OK] Saved: shape={txt.shape}  norms mean={norms.mean():.6f} ± {norms.std():.2e}")

        del model

    print("\n[DONE] CLIP feature extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features for HICO-DET")
    parser.add_argument(
        "--dataset-dir", default="/path/to/datasets/hico_det",
        help="Directory produced by extract_hico_det.py"
    )
    parser.add_argument(
        "--backbone", choices=["mobileclip_s2", "vitl14", "all"], default="all",
        help="Which backbone(s) to extract (default: all)"
    )
    args = parser.parse_args()
    run(args.dataset_dir, args.backbone)
