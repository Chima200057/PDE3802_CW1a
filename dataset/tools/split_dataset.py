import os, random, shutil
from tqdm import tqdm

SRC_DIR = "dataset_balanced"  
OUT_DIR = "dataset_split"
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

random.seed(42)
os.makedirs(OUT_DIR, exist_ok=True)

splits = {"train": [], "val": [], "test": []}

for cls in sorted(os.listdir(SRC_DIR)):
    path = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(path): 
        continue
    images = [os.path.join(cls, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    splits["train"] += images[:n_train]
    splits["val"] += images[n_train:n_train + n_val]
    splits["test"] += images[n_train + n_val:]

for split in splits:
    with open(os.path.join(OUT_DIR, f"{split}.txt"), "w") as f:
        f.write("\n".join(splits[split]))
    print(f"{split}: {len(splits[split])} images")
