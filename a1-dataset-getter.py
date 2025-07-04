import os
import random
import json

def get_image_paths(root_dir):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(dirpath, fname))
    return image_paths

random.seed(42)
img_list = get_image_paths("path/to/image/folder")
pairs = random.sample([(a, b) for a in img_list for b in img_list if a != b], 100) # How many, default: 100 
with open('fixed_image_pairs.json', 'w') as f:
    json.dump(pairs, f)
