#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def detect_faces(app: FaceAnalysis, image_path: Path):
    """
    Detect all faces in image_path, return list of dicts with
    bbox, kps, det_score, embedding, and file name.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    faces = app.get(img)
    out = []
    for f in faces:
        out.append(
            {
                "bbox": f.bbox.tolist(),
                "kps": f.kps.tolist(),
                "det_score": float(f.det_score),
                "embedding": f.normed_embedding.tolist(),
                "file": image_path.name,
            }
        )
    return out


def load_gallery_embeddings(gallery_root: Path):
    """
    For each JPG in each subfolder of gallery_root, read
    its .json sidecar, extract the first embedding, and
    collect (emb, label).
    """
    embs, labels = [], []
    for person_dir in gallery_root.iterdir():
        if not person_dir.is_dir():
            continue
        label = person_dir.name
        for img_file in person_dir.glob("*.[jJ][pP][gG]"):
            sc = img_file.with_suffix(".json")
            if not sc.exists():
                continue
            data = json.loads(sc.read_text(encoding="utf-8"))
            if not data:
                continue
            emb = data[0].get("embedding")
            if emb:
                embs.append(np.array(emb, dtype=np.float32))
                labels.append(label)
    if embs:
        return np.stack(embs, axis=0), labels
    else:
        return np.empty((0, 512)), []


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Detect & match faces in master folder")
    p.add_argument("master_path", type=Path, help="Root folder of images to process")
    p.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=1.0,
        help="Distance threshold for match (default 1.0)",
    )
    args = p.parse_args()
    threshold = args.threshold
    master_path = args.master_path

    # read gallery root
    cfg = Path(".gallery_path")
    if not cfg.is_file():
        print(f"Error: .gallery_path not found.")
        exit(1)
    gallery_root = Path(cfg.read_text().strip())
    if not gallery_root.is_dir():
        print(f"Error: gallery root '{gallery_root}' is invalid.")
        exit(1)

    # init detector
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # load gallery embeddings from sidecars
    gallery_embs, gallery_labels = load_gallery_embeddings(gallery_root)

    # process master images
    for img_path in sorted(master_path.rglob("*.[jJ][pP][gG]")):
        print(f"\n{img_path}")
        sidecar = img_path.with_suffix(".json")
        if sidecar.exists():
            faces = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            faces = detect_faces(app, img_path)
            with open(sidecar, "w", encoding="utf-8") as f:
                json.dump(faces, f, indent=2)
            print(f"Wrote {sidecar.name} ({len(faces)} faces)")

        # match each face and annotate
        matches = set()
        for face in faces:
            emb = np.array(face["embedding"], dtype=np.float32)
            match_name = None
            if gallery_embs.size:
                dists = np.linalg.norm(gallery_embs - emb[None, :], axis=1)
                idx = int(np.argmin(dists))
                dist = float(dists[idx])
                if dist < threshold:
                    match_name = gallery_labels[idx]
                    matches.add(match_name)
                    print(f" matched {match_name} (dist={dist:.3f})")
            face["match"] = match_name

        # write updated sidecar (with match field)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(faces, f, indent=2)

        # collect unique matches for exif tagging
        matches = {face["match"] for face in faces if face.get("match")}
        if matches:
            cmd = ["exiftool", "-overwrite_original"]
            for name in sorted(matches):
                cmd.append(f"-XMP-dc:Subject+=person|{name}")
            cmd.append(str(img_path))
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Tagged {img_path.name} with: {', '.join(sorted(matches))}")
        else:
            print("No matches to tag.")
