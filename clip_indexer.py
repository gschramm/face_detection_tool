#!/usr/bin/env python3
import json
import numpy as np
import faiss
import torch
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def load_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def encode_image(path, model, processor, device):
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARNING] Failed to load {path}: {e}")
        return None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm()
    return emb.squeeze().cpu().numpy()


def collect_images(folder, known_paths):
    return [
        p
        for p in folder.rglob("*")
        if p.suffix in SUPPORTED_EXTENSIONS and str(p) not in known_paths
    ]


def main():
    parser = argparse.ArgumentParser(description="Index images with CLIP and FAISS")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Path to image folder",
        default="/Users/georg/.cache/darktable/mipmaps-26f4ab5d9d6b82f93a1cff57fe5e7e155ae3d6a8.d/3/",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Path to FAISS index file",
        default="/Users/georg/.config/darktable/clip.index",
    )
    parser.add_argument(
        "--paths-json",
        type=Path,
        help="Path to JSON file storing image paths",
        default="/Users/georg/.config/darktable/clip_paths.json",
    )
    parser.add_argument(
        "--model", default="openai/clip-vit-base-patch32", help="CLIP model name"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    args = parser.parse_args()

    model, processor = load_model(args.model, args.device)

    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.paths_json.parent.mkdir(parents=True, exist_ok=True)

    if args.index.exists() and args.paths_json.exists():
        index = faiss.read_index(str(args.index))
        with open(args.paths_json) as f:
            known_paths = set(json.load(f))
    else:
        index = faiss.IndexFlatIP(512)
        known_paths = set()

    new_images = collect_images(args.image_dir, known_paths)
    if not new_images:
        print("No new images found.")
        return

    print(f"Found {len(new_images)} new image(s) to index.")
    new_embeddings = []
    new_paths = []

    for path in tqdm(new_images):
        emb = encode_image(path, model, processor, args.device)
        if emb is not None:
            new_embeddings.append(emb)
            new_paths.append(str(path))

    if not new_embeddings:
        print("No valid embeddings found.")
        return

    emb_array = np.vstack(new_embeddings).astype("float32")
    faiss.normalize_L2(emb_array)
    index.add(emb_array)

    # Save updated index and path list
    faiss.write_index(index, str(args.index))
    all_paths = list(known_paths) + new_paths
    with open(args.paths_json, "w") as f:
        json.dump(all_paths, f, indent=2)

    print(f"Index updated. Total images indexed: {len(all_paths)}")


if __name__ == "__main__":
    main()
