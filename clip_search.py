import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor
import json
from pathlib import Path

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# Encode query
query_text = "a church in front of a mountain"
inputs = processor(text=[query_text], return_tensors="pt")
with torch.no_grad():
    query_emb = model.get_text_features(**inputs)
    query_emb = query_emb / query_emb.norm()
    query_np = query_emb.cpu().numpy().squeeze()

# Load all sidecar embeddings
sidecars = list(
    Path(
        "/Users/georg/.cache/darktable/mipmaps-26f4ab5d9d6b82f93a1cff57fe5e7e155ae3d6a8.d/3/"
    ).rglob("*.clip.json")
)
results = []

for sc in sidecars:
    with open(sc) as f:
        data = json.load(f)
    emb = np.array(data["embedding"])
    score = np.dot(query_np, emb)
    results.append((score, sc.with_suffix("").with_suffix("")))  # remove .clip.json

# Sort and display top matches
for score, img_path in sorted(results, reverse=True)[:10]:
    print(f"{score:.3f} - {img_path}")
