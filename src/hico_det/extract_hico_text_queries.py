"""
Generate MobileCLIP-S2 text embeddings for all HICO-DET HOI categories.

For each HOI category "obj|verb", creates a natural-language prompt:
  "a person riding a motorcycle"
  "a person holding an apple"
  "a person next to a motorcycle"  (for no_interaction cases)

Text and image CLIP embeddings share the same 512-d unit-normalized space,
enabling text-to-image retrieval: send the text embedding of the HOI category
as the query vector instead of a single noisy query image.

Why this matters:
  - Current benchmark uses image[0] of the category as query. One image is noisy.
  - The text embedding "a person holding an apple" is the semantic centroid of
    that category — more representative than any single image.
  - Measured signal: text→positive mean sim=0.206, text→random=0.061 (gap=+0.145)
    vs image→AP=0 for apple|hold with 239 images in DB.

Output:
  hico_clip_text_queries.npy     [N_hoi, 512]  float32  unit-normed
  hico_clip_text_key_order.json  [N_hoi]       list of "obj|verb" keys (all 600)

N_hoi = all HOI categories in hoi_index (before min_relevant filtering).
The benchmark aligns by key lookup, not array index, so filtering is irrelevant.

Usage:
  python extract_hico_text_queries.py
  python extract_hico_text_queries.py --dataset-dir /path/to/datasets/hico_det
"""

import argparse
import json
import numpy as np
from pathlib import Path


def build_prompt(obj: str, verb: str) -> str:
    """Convert HOI pair to natural language prompt."""
    o = obj.replace("_", " ")
    if verb == "no_interaction":
        return f"a person near a {o}"
    v = verb.replace("_", " ")
    return f"a person {v} a {o}"


def extract_text_queries(dataset_dir: str):
    d = Path(dataset_dir)

    with open(d / "hico_hoi_index.json") as f:
        hoi_index = json.load(f)

    hoi_keys = list(hoi_index.keys())
    prompts  = [build_prompt(*k.split("|")) for k in hoi_keys]

    print(f"[OK] HOI categories: {len(hoi_keys)}")
    print(f"[OK] Sample prompts:")
    for p in prompts[:6]:
        print(f"     '{p}'")

    import open_clip
    import torch

    print("\n[INFO] Loading MobileCLIP-S2 (datacompdr)...")
    model, _, _ = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr")
    model.eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP-S2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[OK] Model on {device}")

    BATCH = 64
    all_embs = []
    for start in range(0, len(prompts), BATCH):
        batch  = prompts[start:start + BATCH]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embs.append(emb.cpu().numpy().astype(np.float32))

    text_q = np.concatenate(all_embs, axis=0)  # [N_hoi, 512]

    norms = np.linalg.norm(text_q, axis=1)
    print(f"\n[OK] Text embeddings: {text_q.shape}  dtype={text_q.dtype}")
    print(f"     Norms: mean={norms.mean():.6f} ± {norms.std():.2e}  "
          f"min={norms.min():.6f}  max={norms.max():.6f}")

    out_emb  = d / "hico_clip_text_queries.npy"
    out_keys = d / "hico_clip_text_key_order.json"
    np.save(str(out_emb), text_q)
    with open(out_keys, "w") as f:
        json.dump(hoi_keys, f, indent=2)
    print(f"\n[OK] Saved: {out_emb}  shape={text_q.shape}")
    print(f"[OK] Saved: {out_keys}  ({len(hoi_keys)} keys)")

    # Sanity check: text-image alignment
    clip_db_path = d / "hico_clip.npy"
    gt_path      = d / "hico_gt.json"
    if clip_db_path.exists() and gt_path.exists():
        print("\n[Sanity] Text→image CLIP cosine similarity:")
        clip_db = np.load(str(clip_db_path)).astype(np.float32)
        with open(gt_path) as f:
            gt = json.load(f)
        key_map = {k: i for i, k in enumerate(hoi_keys)}
        for check_key in ["motorcycle|ride", "apple|hold", "cup|wash", "bicycle|ride"]:
            if check_key not in key_map:
                continue
            obj, verb = check_key.split("|")
            t_emb   = text_q[key_map[check_key]]
            pos_ids = [e["id"] for e in gt if [obj, verb] in e["positive_hoi"]][:30]
            rand_ids = list(range(400, 500))
            if not pos_ids:
                continue
            pos_sims  = clip_db[pos_ids] @ t_emb
            rand_sims = clip_db[rand_ids] @ t_emb
            print(f"  {check_key:30s}  pos={pos_sims.mean():.3f}  "
                  f"rand={rand_sims.mean():.3f}  signal={pos_sims.mean()-rand_sims.mean():+.3f}")

    print("\n[DONE] Text query extraction complete.")
    print("Next: submit extract_hico_text_queries.sh, then run benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MobileCLIP-S2 text queries for HICO-DET")
    parser.add_argument("--dataset-dir", default="/path/to/datasets/hico_det",
                        help="Path to hico_det dataset directory")
    args = parser.parse_args()
    extract_text_queries(args.dataset_dir)
