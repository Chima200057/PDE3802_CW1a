import cv2
import numpy as np
import sys
import os
from tensorflow.keras.models import load_model

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "office_item_classifier.h5")
IMG_SIZE = (224, 224)

# Your class names in the exact training order:
CLASS_NAMES = [
    "Book", "Bottle", "Chair", "Desk", "Keyboard",
    "Laptop", "Mouse", "Phone", "Stapler", "Trash_Can"
]

# ==========================
# LOAD MODEL
# ==========================
print("[INFO] Loading trained model...")
model = load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# ==========================
# PREDICT FUNCTION
# ==========================
def predict_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img / 255.0, axis=0)

    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = preds[0][class_index]
    return CLASS_NAMES[class_index], confidence

# ==========================
# MAIN LOGIC
# ==========================
if len(sys.argv) > 1:
    # Predict from a single image file
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        sys.exit()

    img = cv2.imread(img_path)
    label, conf = predict_image(img)
    print(f"Prediction: {label} ({conf*100:.2f}%)")

    cv2.putText(img, f"{label} ({conf*100:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    # Real-time webcam prediction
    print("[INFO] Starting webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_image(frame)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Office Item Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam session ended.")
