#!/usr/bin/env python3
"""
Avatar Matcher: build a CLIP-based embedding index for Discord-style profile pictures
and run fast nearest-neighbor search with FAISS.

Usage:
  # 1) Build the index from a folder of original (square or rectangular) images
  python avatar_matcher.py build-index \
      --db-dir /path/to/pfps \
      --out-dir /path/to/index_out \
      --model ViT-B-32 \
      --pretrained laion2b_s34b_b79k \
      --multi-scales 0.85,0.92,1.0

  # 2) Query with a cropped 75x75 circular avatar
  python avatar_matcher.py search \
      --index-dir /path/to/index_out \
      --image /path/to/query_crop.png \
      --k 10

Notes:
- We normalize embeddings and use FAISS IndexFlatIP (cosine similarity).
- For the DB, we simulate the Discord circular mask at a few radii and average embeddings.
"""
import argparse
import json
import os
from sys import platform
if platform == "darwin":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw

try:
    import faiss  # faiss-cpu
except Exception as e:
    raise RuntimeError("Please install faiss-cpu: pip install faiss-cpu") from e

try:
    import open_clip
    from torchvision import transforms
except Exception as e:
    raise RuntimeError("Please install open_clip_torch and torchvision: pip install open_clip_torch torchvision") from e


def load_model(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess


def to_square(img: Image.Image) -> Image.Image:
    if img.width == img.height:
        return img
    s = min(img.width, img.height)
    return ImageOps.fit(img, (s, s), method=Image.BICUBIC, centering=(0.5, 0.5))


def apply_circular_mask(img: Image.Image, pad_px: int = 0, bg=(0, 0, 0)) -> Image.Image:
    """
    Ensures a circular foreground on a solid background. If pad_px>0 we shrink content
    before masking to emulate different visible radii (simulating Discord crop).
    """
    if img.width != img.height:
        img = to_square(img)
    size = img.size[0]
    if pad_px:
        img = img.crop((pad_px, pad_px, size - pad_px, size - pad_px)).resize((size, size), Image.BICUBIC)

    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size, size), fill=255)

    out = Image.new("RGB", (size, size), bg)
    out.paste(img, mask=mask)
    return out


def preprocess_for_embed(img: Image.Image, preprocess) -> torch.Tensor:
    return preprocess(img).unsqueeze(0)


@torch.no_grad()
def embed_image(model, tensor: torch.Tensor, device: str) -> np.ndarray:
    tensor = tensor.to(device)
    feats = model.encode_image(tensor)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).detach().cpu().numpy().astype("float32")


def build_embeddings_for_image(
    model, preprocess, device: str, img_path: Path, input_size: int, multi_scales: List[float]
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = to_square(img).resize((input_size, input_size), Image.BICUBIC)

    embs = []
    for scale in multi_scales:
        # pad = int((1.0 - scale) * input_size / 2.0)
        pad = 0
        circ = apply_circular_mask(img, pad_px=pad)
        tensor = preprocess_for_embed(circ, preprocess)
        embs.append(embed_image(model, tensor, device))
    emb = np.mean(np.stack(embs, axis=0), axis=0)
    emb /= (np.linalg.norm(emb) + 1e-12)
    return emb.astype("float32")


def build_index(
    in_dir: Path,
    out_dir: Path,
    manifest_path: Path = "manifest.json",
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    multi_scales: List[float] = [0.85, 0.92, 1.0],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_name} ({pretrained}) on {device} ...")
    model, preprocess = load_model(model_name, pretrained, device)

    # Infer model input size from preprocess by sending a dummy image
    # OpenCLIP preprocess will handle resizing; we keep a conventional 224 fallback.
    input_size = 224

    # Collect image paths
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found under: {in_dir}")

    print(f"Found {len(paths)} images. Computing embeddings...")
    vecs = []
    meta = []  # store mapping from index row -> file path
    manifest = json.load(open(in_dir / manifest_path))
    for i, p in enumerate(paths, 1):
        try:
            emb = build_embeddings_for_image(model, preprocess, device, p, input_size, multi_scales)
        except Exception as e:
            print(f"[WARN] Skipping {p} due to error: {e}")
            continue
        vecs.append(emb)
        if manifest is None:
            print(f"[WARN] No manifest found at {in_dir / manifest_path}, using path only")
            meta.append({"path": str(p)})
        else:
            info = manifest.get(str(p), {})
            meta.append({"path": str(p), "user_id": info.get("user_id"), "username": info.get("username")})
        if i % 100 == 0:
            print(f"  Processed {i}/{len(paths)}")

    if not vecs:
        raise RuntimeError("No embeddings were produced. Check your image files.")

    vecs = np.stack(vecs, axis=0).astype("float32")
    # Cosine similarity on normalized vectors -> use inner product
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Save artifacts
    index_path = out_dir / "avatars.index"
    faiss.write_index(index, str(index_path))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"items": meta, "model": model_name, "pretrained": pretrained, "multi_scales": multi_scales}, f, indent=2)
    np.save(out_dir / "embeddings.npy", vecs)

    print(f"Saved index -> {index_path}")
    print(f"Saved metadata -> {out_dir/'meta.json'}")
    print(f"Saved embeddings -> {out_dir/'embeddings.npy'}")


def search_index(
    index_dir: Path,
    query_image: Path,
    k: int = 10,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Tuple[float, str]]:
    # Load FAISS + meta
    index = faiss.read_index(str(index_dir / "avatars.index"))
    with open(index_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta["items"]

    # Load model used to build index (or override with flags, but best to match)
    model, preprocess = load_model(model_name or meta.get("model", "ViT-B-32"),
                                   pretrained or meta.get("pretrained", "laion2b_s34b_b79k"),
                                   device=device)

    # Prepare query
    img = Image.open(query_image).convert("RGB")
    # Query is a circular crop (75x75) w/ ~5px margin; trim a bit then mask again to normalize edge ring
    img = to_square(img)
    # Upscale to model input size (224); keep circle
    img = img.resize((224, 224), Image.BICUBIC)
    img = apply_circular_mask(img, pad_px=img.width // 12)  # trims edge ring slightly
    tensor = preprocess_for_embed(img, preprocess)
    qv = embed_image(model, tensor, device)[None, :]  # (1, d)

    D, I = index.search(qv.astype("float32"), k)
    D = D[0].tolist()
    I = I[0].tolist()

    results = [(float(D[i]), items[I[i]]["path"], items[I[i]]["user_id"], items[I[i]]["username"]) for i in range(len(I))]
    return results


def _parse_yolo_labels(labels_path: Path):
    lines = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for raw in f.readlines():
            s = raw.strip()
            if not s:
                lines.append((raw, None))
                continue
            parts = s.split()
            if len(parts) < 5:
                lines.append((raw, None))
                continue
            try:
                class_id = int(float(parts[0]))
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                conf = float(parts[5]) if len(parts) > 5 else 0.0
                lines.append((raw, (class_id, x, y, w, h, conf)))
            except Exception:
                lines.append((raw, None))
    return lines


def _bbox_norm_to_pixels(xc, yc, w, h, W, H):
    x_center = xc * W
    y_center = yc * H
    bw = w * W
    bh = h * H
    x1 = int(round(x_center - bw / 2.0))
    y1 = int(round(y_center - bh / 2.0))
    x2 = int(round(x_center + bw / 2.0))
    y2 = int(round(y_center + bh / 2.0))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


@torch.no_grad()
def find_avatar_matches(
    index_dir: Path,
    image_path: Path,
    labels_path: Path,
    target_class_id: int = 0,
    k: int = 1,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    # Load FAISS + meta
    index = faiss.read_index(str(index_dir / "avatars.index"))
    with open(index_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    items = meta["items"]

    # Load model
    model, preprocess = load_model(
        model_name or meta.get("model", "ViT-B-32"),
        pretrained or meta.get("pretrained", "laion2b_s34b_b79k"),
        device=device,
    )

    # Load image
    img = Image.open(image_path).convert("RGB")
    W, H = img.width, img.height

    parsed = _parse_yolo_labels(labels_path)
    print(f"Loaded {len(parsed)} detections from {labels_path}")
    updated_lines = []
    for raw_line, det in parsed:
        if det is None:
            updated_lines.append(raw_line.rstrip("\n"))
            continue
        class_id, xc, yc, w, h, conf = det
        if class_id != target_class_id:
            updated_lines.append(raw_line.rstrip("\n"))
            continue

        print(f"Processing detection: {det}")
        x1, y1, x2, y2 = _bbox_norm_to_pixels(xc, yc, w, h, W, H)
        crop = img.crop((x1, y1, x2, y2))
        crop = to_square(crop).resize((224, 224), Image.BICUBIC)
        crop = apply_circular_mask(crop, pad_px=crop.width // 12)
        tensor = preprocess_for_embed(crop, preprocess)
        qv = embed_image(model, tensor, device)[None, :]

        D, I = index.search(qv.astype("float32"), max(1, k))
        top_i = int(I[0][0])
        top_score = float(D[0][0])
        top_path = items[top_i]["path"]

        if conf > 0.0:
            base_tokens = raw_line.strip().split()[:6]
        else:
            base_tokens = raw_line.strip().split()[:5]
        # Derive a stable user identifier from meta, fallback to filename stem
        user_id = items[top_i]["user_id"] or Path(top_path).stem
        username = items[top_i]["username"] or Path(top_path).stem
        # Emit a single JSON object after '#', so values with spaces are preserved
        ann_obj = {"user_id": str(user_id), "username": username, "conf": round(top_score, 6)}
        annotated = " ".join(base_tokens) + "  # " + json.dumps(ann_obj, ensure_ascii=False)
        updated_lines.append(annotated)
        print(f"  .  Found match for {user_id} ({username}) with score {top_score:.6f}")

    with open(labels_path, "w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")

def main():
    parser = argparse.ArgumentParser(description="Discord Avatar Matcher (CLIP + FAISS)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", help="Build FAISS index from a folder of original images")
    p_build.add_argument("--in-dir", type=Path, required=True, help="Folder containing original avatar images")
    p_build.add_argument("--out-dir", type=Path, required=True, help="Output folder for index files")
    p_build.add_argument("--manifest", type=Path, default="manifest.json", help="Manifest file for user info")
    p_build.add_argument("--model", type=str, default="ViT-B-32", help="OpenCLIP model name (e.g., ViT-B-32, ViT-L-14)")
    p_build.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    p_build.add_argument("--multi-scales", type=str, default="0.85,0.92,1.0",
                         help="Comma-separated visible-radius scales to simulate (e.g., 0.85,0.92,1.0)")

    p_search = sub.add_parser("search", help="Search matches for a query crop")
    p_search.add_argument("--index-dir", type=Path, required=True, help="Folder with avatars.index and meta.json")
    p_search.add_argument("--image", type=Path, required=True, help="Query image path (75x75 circular crop)")
    p_search.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to return")
    p_search.add_argument("--model", type=str, default=None, help="(Optional) Override model name used for queries")
    p_search.add_argument("--pretrained", type=str, default=None, help="(Optional) Override pretrained tag")

    p_annot = sub.add_parser("search-labels", help="Annotate YOLO labels by searching avatar crops against index")
    p_annot.add_argument("--index-dir", type=Path, required=True, help="Folder with avatars.index and meta.json")
    p_annot.add_argument("--image", type=Path, required=True, help="Original image containing detections")
    p_annot.add_argument("--labels", type=Path, required=True, help="YOLO .txt labels file to read and update in place")
    p_annot.add_argument("--class-id", type=int, default=0, help="Class id to annotate (default: 0)")
    p_annot.add_argument("--k", type=int, default=1, help="Top-K neighbors to consider (only top-1 used for name)")
    p_annot.add_argument("--model", type=str, default=None, help="(Optional) Override model name used for queries")
    p_annot.add_argument("--pretrained", type=str, default=None, help="(Optional) Override pretrained tag")

    args = parser.parse_args()

    if args.cmd == "build-index":
        multi_scales = [float(x) for x in args.multi_scales.split(",") if x.strip()]
        build_index(
            in_dir=args.in_dir,
            out_dir=args.out_dir,
            manifest_path=args.manifest,
            model_name=args.model,
            pretrained=args.pretrained,
            multi_scales=multi_scales,
        )
    elif args.cmd == "search":
        results = search_index(
            index_dir=args.index_dir,
            query_image=args.image,
            k=args.k,
            model_name=args.model or "",
            pretrained=args.pretrained or "",
        )
        # Print JSON lines for easy parsing
        for score, path, user_id, username in results:
            print(json.dumps({"score": score, "path": path, "user_id": user_id, "username": username}, ensure_ascii=False))
    elif args.cmd == "search-labels":
        find_avatar_matches(
            index_dir=args.index_dir,
            image_path=args.image,
            labels_path=args.labels,
            target_class_id=args.class_id,
            k=args.k,
            model_name=args.model or "",
            pretrained=args.pretrained or "",
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
