#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def load_unmatched_faces(master_root: Path, pattern: str = "*.json"):
    """
    Walk master_root for *.json sidecars.
    For each face dict where face['match'] is None, collect:
      - embedding (512-d list)
      - img_path (Path to the JPG)
      - bbox  ([x1,y1,x2,y2])
    Returns a list of tuples: (embedding, img_path, bbox).
    """
    data = []
    for sc in master_root.rglob(pattern):
        records = json.loads(sc.read_text(encoding="utf-8"))
        for face in records:
            if face.get("match") is None:
                emb = face.get("embedding")
                bbox = face.get("bbox")
                if emb and bbox:
                    img_path = sc.parent / face.get("file", sc.stem + ".JPG")
                    data.append((np.array(emb, dtype=np.float32), img_path, bbox))
    return data


def cluster_and_save(faces, out_dir: Path, eps: float, min_samples: int):
    """
    faces: list of (emb, img_path, bbox)
    clusters with DBSCAN(eps, min_samples), then for each cluster >= min_samples:
      - make out_dir/cluster_<label>/
      - for each face in that cluster, load the image, crop bbox, save as PNG
    """
    embeddings = np.stack([f[0] for f in faces], axis=0)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(embeddings)
    labels = db.labels_  # -1 = noise

    # group indices by label
    grouped = defaultdict(list)
    for idx, lbl in enumerate(labels):
        if lbl >= 0:
            grouped[lbl].append(idx)

    # only keep clusters of size >= min_samples
    for lbl, idxs in grouped.items():
        if len(idxs) < min_samples:
            continue
        cluster_dir = out_dir / f"cluster_{lbl}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for i in idxs:
            emb, img_path, bbox = faces[i]
            x1, y1, x2, y2 = map(int, bbox)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            crop = img[y1:y2, x1:x2]
            # unique filename: <origname>_face<i>.png
            out_name = f"{img_path.stem}_f{i}.png"
            try:
                cv2.imwrite(str(cluster_dir / out_name), crop)
            except Exception as e:
                print(f"Error saving {out_name}: {e}")
        print(f"Wrote {len(idxs)} faces to {cluster_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster unmatched faces and save crops per cluster"
    )
    parser.add_argument(
        "--master-path",
        "-m",
        type=Path,
        required=True,
        help="Root folder containing your master images + .json sidecars",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("face_clusters"),
        help="Where to write cluster folders",
    )
    parser.add_argument(
        "--eps",
        "-e",
        type=float,
        default=1.0,
        help="DBSCAN eps parameter (distance threshold)",
    )
    parser.add_argument(
        "--min-samples", "-n", type=int, default=7, help="Minimum cluster size to keep"
    )
    args = parser.parse_args()

    faces = load_unmatched_faces(args.master_path)
    if not faces:
        print("No unmatched faces found.")
        exit(0)

    print(
        f"Loaded {len(faces)} unmatched faces; clustering with eps={args.eps}, min_samples={args.min_samples}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cluster_and_save(faces, args.out_dir, eps=args.eps, min_samples=args.min_samples)
