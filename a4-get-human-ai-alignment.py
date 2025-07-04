import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import matplotlib.pyplot as plt

folder = "path/to/image/folder"


with open('human_similarity_ratings.json', 'r') as f:
    human_data = json.load(f)
with open('clip_similarity.json', 'r') as f:
    clip_data = json.load(f)

clip_dict = {(x['img1'], x['img2']): x['cosine'] for x in clip_data}
human_dict = {(x['img1'], x['img2']): x['rating'] for x in human_data}

pairs = [k for k in human_dict if k in clip_dict]
human_ratings = np.array([human_dict[k] for k in pairs])
clip_ratings = np.array([clip_dict[k] for k in pairs])

plt.figure(figsize=(8,6))
plt.scatter(clip_ratings, human_ratings, color='royalblue', alpha=0.7, label='Pairs')
plt.xlabel('CLIP Cosine Similarity')
plt.ylabel('Human Similarity Rating')
plt.title('Human vs. CLIP Similarity')
plt.grid(True)

if len(human_ratings) > 1:
    coeffs = np.polyfit(clip_ratings, human_ratings, 1)
    xs = np.linspace(min(clip_ratings), max(clip_ratings), 100)
    plt.plot(xs, np.polyval(coeffs, xs), color='crimson', lw=2, label='Linear Fit')
    plt.legend()

plt.tight_layout()
plt.savefig('clip_vs_human_similarity.png', dpi=200)

pearson_corr, pearson_p = pearsonr(clip_ratings, human_ratings)
spearman_corr, spearman_p = spearmanr(clip_ratings, human_ratings)

print(f"Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3g})")
print(f"Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3g})")
print(f"Number of rated pairs: {len(pairs)}")

alignment = np.abs(human_ratings - clip_ratings)

def short_pair(pair):
    img1, img2 = pair
    return (os.path.basename(img1), os.path.basename(img2))

top5_agree_idx = np.argsort(alignment)[:10]
print("\nTop 10 highest human-CLIP agreement pairs:")
for idx in top5_agree_idx:
    pair = short_pair(pairs[idx])
    print(f"{pair[0]} vs {pair[1]} | Human: {human_ratings[idx]:.2f}, CLIP: {clip_ratings[idx]:.2f}, Diff: {alignment[idx]:.2f}")

top5_disagree_idx = np.argsort(alignment)[-10:][::-1]
print("\nTop 10 lowest human-CLIP agreement pairs:")
for idx in top5_disagree_idx:
    pair = short_pair(pairs[idx])
    print(f"{pair[0]} vs {pair[1]} | Human: {human_ratings[idx]:.2f}, CLIP: {clip_ratings[idx]:.2f}, Diff: {alignment[idx]:.2f}")

def find_image(root, filename):
    for dirpath, _, files in os.walk(root):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(f"{filename} not found in {root}")

def plot_and_save_pair(img1_path, img2_path, pair_title, fname):
    fig, axs = plt.subplots(1, 2, figsize=(7,4))
    axs[0].imshow(Image.open(img1_path))
    axs[0].set_title(os.path.basename(img1_path))
    axs[0].axis('off')
    axs[1].imshow(Image.open(img2_path))
    axs[1].set_title(os.path.basename(img2_path))
    axs[1].axis('off')
    plt.suptitle(pair_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(fname, dpi=150)
    plt.close(fig)

print("\nSaving top 10 highest agreement pair images...")
for rank, idx in enumerate(top5_agree_idx, 1):
    img1, img2 = pairs[idx]
    img1_path = find_image(folder, os.path.basename(img1))
    img2_path = find_image(folder, os.path.basename(img2))
    title = f"{os.path.basename(img1)} vs {os.path.basename(img2)}\nHuman: {human_ratings[idx]:.2f}, CLIP: {clip_ratings[idx]:.2f}, Diff: {alignment[idx]:.2f}"
    fname = f"high_{rank}.png"
    plot_and_save_pair(img1_path, img2_path, title, fname)

print("\nSaving top 10 lowest agreement pair images...")
for rank, idx in enumerate(top5_disagree_idx, 1):
    img1, img2 = pairs[idx]
    img1_path = find_image(folder, os.path.basename(img1))
    img2_path = find_image(folder, os.path.basename(img2))
    title = f"{os.path.basename(img1)} vs {os.path.basename(img2)}\nHuman: {human_ratings[idx]:.2f}, CLIP: {clip_ratings[idx]:.2f}, Diff: {alignment[idx]:.2f}"
    fname = f"low_{rank}.png"
    plot_and_save_pair(img1_path, img2_path, title, fname)
