import os
import random
import shutil
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

SRC_DIR = "dataset/images"      # input folder
OUT_DIR = "dataset_balanced"    # output folder
TARGET_COUNT = 1000             # desired number of images per class

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(42)

# Simple augmentation functions
def augment_image(img):
    # Randomly flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.8 + random.random() * 0.4)
    # Random rotation
    img = img.rotate(random.randint(-20, 20))
    return img

for cls in os.listdir(SRC_DIR):
    src_path = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(src_path):
        continue

    dst_path = os.path.join(OUT_DIR, cls)
    os.makedirs(dst_path, exist_ok=True)

    images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    count = len(images)

    if count == 0:
        print(f"⚠️ Skipping empty class: {cls}")
        continue

    print(f"\nProcessing class: {cls} ({count} images)")

    # If too many images — downsample
    if count > TARGET_COUNT:
        selected = random.sample(images, TARGET_COUNT)
        for f in tqdm(selected, desc=f"Downsampling {cls}"):
            shutil.copy(os.path.join(src_path, f), os.path.join(dst_path, f))
    
    # If too few — copy + augment
    else:
        # Copy all existing first
        for f in images:
            shutil.copy(os.path.join(src_path, f), os.path.join(dst_path, f))

        # Generate new augmented images
        i = 0
        while len(os.listdir(dst_path)) < TARGET_COUNT:
            base_img = random.choice(images)
            img = Image.open(os.path.join(src_path, base_img)).convert("RGB")
            img = img.resize((256, 256))
            img = augment_image(img)

            new_name = f"aug_{i}_{base_img}"
            img.save(os.path.join(dst_path, new_name))
            i += 1

    print(f"✅ Final count for {cls}: {len(os.listdir(dst_path))}")
