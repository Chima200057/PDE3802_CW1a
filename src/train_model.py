import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPLIT = os.path.join(BASE_DIR, "dataset_split")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_balanced")
MODEL_PATH = os.path.join(BASE_DIR, "office_item_classifier.h5")

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load data from text file
def load_split_data(txt_file):
    images, labels = [], []
    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        rel_path = line.strip()
        if not rel_path:
            continue

        # Split folder and file name to get the label
        label = rel_path.split("\\")[0]  # folder name = class
        img_path = os.path.join(DATASET_DIR, rel_path)

        if not os.path.exists(img_path):
            print(f"[WARNING] Missing file: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Failed to read: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

# Load all sets
print("[INFO] Loading dataset...")
x_train, y_train = load_split_data(os.path.join(DATASET_SPLIT, "train.txt"))
x_val, y_val = load_split_data(os.path.join(DATASET_SPLIT, "val.txt"))

print(f"[INFO] Training samples: {len(x_train)}, Validation samples: {len(x_val)}")

# Encode labels
le = LabelEncoder()
y_train_enc = to_categorical(le.fit_transform(y_train))
y_val_enc = to_categorical(le.transform(y_val))

# Normalize
x_train = x_train / 255.0
x_val = x_val / 255.0

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator()

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
output = Dense(len(le.classes_), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train
print("[INFO] Training model...")
history = model.fit(
    train_datagen.flow(x_train, y_train_enc, batch_size=BATCH_SIZE),
    validation_data=val_datagen.flow(x_val, y_val_enc, batch_size=BATCH_SIZE),
    epochs=EPOCHS
)

# Save
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
print(f"[INFO] Classes: {le.classes_}")
