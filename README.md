# Collaborators
1. Chimaraoke Mbata Benjamin Collins - M00909998 - cm1833live.mdx.ac.uk (chima200057 - GitHub)
2. Kingsley-Ogarashi Samuel Chukwuemeka - M00931065 (buda360 - GitHub)

# Office Item Classifier — PDE3802_CW1a
- Repository: https://github.com/Chima200057/PDE3802_CW1a
- Framework: TensorFlow / Keras, OpenCV, NumPy
- Model file included: office_item_classifier.h5

# Overview
This project implements a single-object office item classifier. It accepts either an image file or a webcam frame and outputs a predicted class label and confidence score. The training pipeline, dataset, evaluation scripts, and a saved model are included.
- Contents
1. src/ — source code: training, inference, utilities.
2. dataset/, dataset_balanced/, dataset_split/ — dataset folders and prepared splits.
3. office_item_classifier.h5 — trained TensorFlow model (weights + architecture).
4. confusion_matrix.png — confusion matrix for held-out test set.
5. classification_report.txt — accuracy, precision, recall, F1 metrics.
6. requirements.txt — Python dependencies.


# Installation
Recommended: use a Python 3.10+ virtual environment.
1.	Clone the repository:
git clone https://github.com/Chima200057/PDE3802_CW1a.git
cd PDE3802_CW1a
2.	(Optional) Create and activate a virtual environment:
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
3.	Install requirements:
pip install -r requirements.txt
Note: requirements.txt typically includes: tensorflow, opencv-python, numpy, matplotlib, scikit-learn, pandas.
If you have GPU and want TensorFlow-GPU, install the appropriate tensorflow package for your CUDA/CuDNN versions.


Quick start — Interface
There is a quick and easy to use GUI script to run inference on an image(be it local or live).

# Example: predict a local single file
python src/gui1_predict_office_item.py
Select "Test Single Image " option
Select a local image from your system
Expected output (console):
Input: examples/book1.jpg
Prediction: book (confidence: 92%)

# Example: predict a live single file
python src/gui1_predict_office_item.py
Select "Start Webcam" option (press q to quit):
The program will show a live video window, detect the most prominent item in the center (single object assumption) and display the predicted class + confidence in the window.


Training (reproduce / fine-tune)
Note: training was performed with TensorFlow 2.x. If you want to retrain or fine-tune:
python src/train_model.py --data_dir dataset_split/train --val_dir dataset_split/val --epochs 25 --batch_size 32 --save_model runs/office_model.h5
Key training options are in src/train.py. Preprocessing resizes images to 224×224 and normalizes to [0,1]. Data augmentation applied during training: random flip, rotation ±15°, brightness jitter.


Evaluation
Evaluate on the held-out test set:
python src/evaluate.py --model office_item_classifier.h5 --test_dir dataset_split/test --report out/classification_report.txt --confusion out/confusion_matrix.png
Expected outputs (included):
•	classification_report.txt — includes accuracy and per-class precision/recall/F1. Reported (example):
o	Test accuracy: 0.94
o	Macro F1: 0.93
•	confusion_matrix.png — visual confusion matrix saved to disk.
The repository already includes classification_report.txt and confusion_matrix.png generated for the saved model.


Troubleshooting
1. Model file too large / memory errors
•	If office_item_classifier.h5 fails to load due to memory constraints, try running on smaller batch sizes or switch to CPU-only: export CUDA_VISIBLE_DEVICES="" (Linux/macOS) or set environment variable in Windows PowerShell.
2. Webcam not opening
•	Ensure camera index is correct (--camera 0 or --camera 1). Close other apps using the camera.
•	On Linux, ensure you have permissions to access /dev/video*.
3. Incorrect labels / low confidence
•	Check image preprocessing: resizing and normalization must match the model training pipeline (224×224, scale to [0,1]).
•	Try running inference with multiple frames averaged to improve confidence.
4. TensorFlow version mismatch
•	If loading the model throws errors, confirm TensorFlow version in requirements.txt. If the model was saved with TF 2.11, use a compatible TF release.
5. Missing dependencies
•	pip install -r requirements.txt should install all required packages. If a package fails, try upgrading pip: python -m pip install --upgrade pip then reinstall.


Results & Error analysis (brief)
Reported metrics (held-out test set):
•	Accuracy: 0.94 (example; see classification_report.txt for exact values)
•	Macro F1: 0.93
Confusion matrix: confusion_matrix.png (included). Observations:
•	The model frequently confuses mouse and keyboard when only partial views are present.
•	bottle vs mug errors occur when the top of the object is occluded.
Error analysis notes & fixes:
•	Add more images with occlusions and varied viewpoints for mouse and keyboard classes.
•	Collect additional images in low-light and cluttered backgrounds.
•	Consider fine-tuning with a small object detector (e.g., YOLO) to crop object region before classification.





Acknowledgements
This project was prepared for PDE 3802 — Artificial Intelligence (in Robotics) 2025–26 coursework 1 By Chimaraoke Mbata & Samuel Ogarashi. The implementation uses public open-source libraries (TensorFlow, OpenCV). 