import os
from collections import Counter

data_dir = "dataset/images"  # adjust if needed
counts = Counter()

for cls in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, cls)
    if not os.path.isdir(path):
        continue
    n = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    counts[cls] = n

print("Class counts:")
for k, v in counts.items():
    print(f"{k:<12}: {v}")
