#!/usr/bin/env python3
import json
from pathlib import Path

import cv2
from insightface.app import FaceAnalysis


def detect_one_face(app: FaceAnalysis, image_path: Path):
    """
    Detect at most one face, return a dict with:
      - name:       parent‐folder name
      - bbox:       [x1, y1, x2, y2]
      - kps:        5 × [x, y]
      - det_score:  confidence
      - embedding:  512‐d list
    or None if no face found.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    f = faces[0]
    return {
        "name": image_path.parent.name,
        "bbox": f.bbox.tolist(),
        "kps": f.kps.tolist(),
        "det_score": float(f.det_score),
        "embedding": f.normed_embedding.tolist(),
    }


if __name__ == "__main__":
    # 1) read gallery root
    cfg = Path(".gallery_path")
    if not cfg.is_file():
        print(f"Error: {cfg} not found.")
        exit(1)
    gallery_root = Path(cfg.read_text().strip())
    if not gallery_root.is_dir():
        print(f"Error: {gallery_root} is not a directory.")
        exit(1)

    # 2) init detector
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    # 3) walk images, write sidecars
    for img_path in gallery_root.rglob("*.[jJp][pPn][gGg]"):
        sidecar = img_path.with_suffix(".json")
        if sidecar.exists():
            continue
        info = detect_one_face(app, img_path)
        data = [] if info is None else [info]
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        status = "1 face" if info else "no faces"
        print(f"Wrote {img_path} ({status})")
