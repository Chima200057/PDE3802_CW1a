import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "office_item_classifier.h5")
IMG_SIZE = (224, 224)

# Your class names — must match training order
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
# GUI FUNCTIONS
# ==========================
def test_single_image():
    filepath = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not filepath:
        return

    image = cv2.imread(filepath)
    label, conf = predict_image(image)

    # Convert for Tkinter display
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (300, 300))
    im = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im)

    # Display image
    img_label.config(image=imgtk)
    img_label.image = imgtk

    # Display prediction
    result_text.set(f"Prediction: {label} ({conf*100:.1f}%)")


def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    messagebox.showinfo("Webcam Mode", "Press 'q' to quit webcam view.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_image(frame)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Live Office Item Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==========================
# GUI SETUP
# ==========================
root = tk.Tk()
root.title("Office Item Classifier")
root.geometry("500x600")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="Office Item Recognition System",
    font=("Arial", 18, "bold"),
    pady=15
)
title_label.pack()

# Image display area
img_label = tk.Label(root)
img_label.pack(pady=10)

# Prediction result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14))
result_label.pack(pady=5)

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

test_btn = tk.Button(
    btn_frame, text="🖼️ Test Single Image",
    command=test_single_image, font=("Arial", 13), width=20
)
test_btn.grid(row=0, column=0, padx=10)

webcam_btn = tk.Button(
    btn_frame, text="📷 Start Webcam",
    command=start_webcam, font=("Arial", 13), width=20
)
webcam_btn.grid(row=0, column=1, padx=10)

# Exit button
exit_btn = tk.Button(
    root, text="Exit", command=root.quit,
    font=("Arial", 12), bg="red", fg="white", width=10
)
exit_btn.pack(pady=15)

# Run GUI
root.mainloop()
