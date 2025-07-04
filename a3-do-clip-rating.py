import torch
import clip
from PIL import Image
import json
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

with open('fixed_image_pairs.json', 'r') as f:
    pairs = json.load(f)

results = []
for img1_path, img2_path in pairs:
    img1 = preprocess(Image.open(img1_path)).unsqueeze(0).to(device)
    img2 = preprocess(Image.open(img2_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = model.encode_image(img1).cpu().numpy()
        emb2 = model.encode_image(img2).cpu().numpy()
    cosine_sim = (emb1 @ emb2.T).item() / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    results.append({'img1': img1_path, 'img2': img2_path, 'cosine': cosine_sim})
with open('clip_similarity.json', 'w') as f:
    json.dump(results, f)
