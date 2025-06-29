#!/usr/bin/env python3
import json
import torch
import faiss
import argparse
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

import sqlite3
import pandas as pd
from os.path import expanduser


def get_darktable_index():
    # —2— Paths to your Darktable DBs
    library_db = expanduser("~/.config/darktable/library.db")
    data_db = expanduser("~/.config/darktable/data.db")

    # —3— Connect & attach
    conn = sqlite3.connect(library_db)
    conn.execute(f"ATTACH DATABASE '{data_db}' AS data")

    query = f"""
    SELECT
      i.id        AS id,
      fr.folder   AS folder,
      i.filename  AS filename
    FROM images         AS i
    JOIN film_rolls    AS fr ON i.film_id = fr.id
    GROUP BY i.id
    """

    # —5— Run and grab a neat DataFrame
    df = pd.read_sql_query(query, conn)
    df.set_index("id", inplace=True)

    conn.close()

    return df


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


if __name__ == "__main__":
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

    # get the darktable index
    dt_index = get_darktable_index()

    print(f'\nTop {args.top_k} matches for: "{args.query}"\n')
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        file_id = int(Path(paths[idx]).stem)
        print(f"{rank:2d}. {score:.3f} — {file_id}")

        dt_fileinfo = dt_index.loc[file_id]
        fpath = Path(dt_fileinfo["folder"]) / dt_fileinfo["filename"]
        print(f"   Darktable path: {fpath}")

        print("")
