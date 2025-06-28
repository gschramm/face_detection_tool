from pathlib import Path
from PIL import Image
import torch
import json
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Image folder
img_dir = Path(
    "/Users/georg/.cache/darktable/mipmaps-26f4ab5d9d6b82f93a1cff57fe5e7e155ae3d6a8.d/3/"
)
image_paths = list(img_dir.rglob("*.jpg"))

# Load CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()


for img_path in tqdm(image_paths):
    sidecar = img_path.with_suffix(img_path.suffix + ".clip.json")
    if sidecar.exists():
        continue  # Skip already processed images

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
        continue

    # Compute embedding
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
        embedding = embedding / embedding.norm()
        embedding = embedding.squeeze().tolist()

    # Write JSON sidecar
    data = {
        "path": str(img_path.name),
        "embedding": embedding,
        "tags": [],  # optionally auto-fill or leave empty
        "model": model_name,
    }
    with open(sidecar, "w") as f:
        json.dump(data, f)
