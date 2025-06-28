#!/usr/bin/env python3
import json
import torch
import faiss
import numpy as np
import argparse
from transformers import CLIPModel, CLIPProcessor


def load_model(model_name, device):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def encode_text(text, model, processor, device):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm()
    return emb.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Search CLIP-indexed image collection")
    parser.add_argument("query", type=str, help="Text query (e.g. 'a red bicycle')")
    parser.add_argument(
        "--index",
        default="/Users/georg/.config/darktable/clip.index",
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--paths-json",
        default="/Users/georg/.config/darktable/clip_paths.json",
        help="Path to JSON path list",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to show"
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

    # Load model and processor
    model, processor = load_model(args.model, args.device)

    # Load index and paths
    index = faiss.read_index(args.index)
    with open(args.paths_json) as f:
        paths = json.load(f)

    # Encode query
    query_vec = encode_text(args.query, model, processor, args.device)

    # Search
    D, I = index.search(query_vec, args.top_k)
    print(f'\nTop {args.top_k} matches for: "{args.query}"\n')
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        print(f"{rank:2d}. {score:.3f} — {paths[idx]}")


if __name__ == "__main__":
    main()
