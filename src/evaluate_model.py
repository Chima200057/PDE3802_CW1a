import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPLIT = os.path.join(BASE_DIR, "dataset_split")
DATASET_DIR = os.path.join(BASE_DIR, "dataset_balanced")
MODEL_PATH = os.path.join(BASE_DIR, "office_item_classifier.h5")

IMG_SIZE = (224, 224)

# Load data
def load_split_data(txt_file):
    images, labels = [], []
    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        rel_path = line.strip()
        if not rel_path:
            continue
        label = rel_path.split("\\")[0]
        img_path = os.path.join(DATASET_DIR, rel_path)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)

print("[INFO] Loading test set...")
x_test, y_test = load_split_data(os.path.join(DATASET_SPLIT, "test.txt"))
print(f"[INFO] Test samples: {len(x_test)}")

# Encode labels (must match training classes)
le = LabelEncoder()
le.fit(y_test)
y_test_enc = le.transform(y_test)

# Normalize
x_test = x_test / 255.0

# Load model
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)

# Predict
preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)

# Reports
print("\nClassification Report:")
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, digits=4)
print(report)

# Save report
with open(os.path.join(BASE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()

print("[INFO] Evaluation complete. Results saved.")
