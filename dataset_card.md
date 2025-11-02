Dataset short description:
A small image classification dataset of common office items collected for the PDE 3802 Coursework 1. The dataset contains 10 classes relevant to office organization tasks (e.g., phone, bottle, keyboard, mouse, stapler, book). Images were gathered from internet sources and from locally photographed items to ensure variability in background, lighting and viewpoint.

Dataset structure:
dataset/
  ├─ book
  ├─ bottle/
  ├─ chair/
  ├─ desk/
  ├─ keyboard/
  ├─ laptop/
  ├─ mouse/
  ├─ phone/
  ├─ stapler/
  ├─trashcan/
  
Number of classes: 10.

Number of images : 10000 total (1000 per class) — see dataset_balanced/ and dataset_split/ for exact train/val/test splits.

Image format: JPEG, color, varied resolutions (preprocessed to 256×256 for model input).

Collection process:
•	Combination of curated downloads and handheld smartphone photos taken by the author.
•	Data augmentation applied during training (random flip, rotation, brightness jitter).

Uses:
•	Training and evaluation of single-object, single-class office-item classifiers.

Limitations & biases:
•	Small dataset size — may not generalize to unseen office environments or unusual viewpoints.
•	Class imbalance may exist in raw dataset; balanced subsets are provided in dataset_balanced/.
•	Mostly daytime indoor images; poor performance is expected on extreme lighting (very dark or backlit scenes).

Sensitive content: None.

Maintenance & contact:
•	Repository owner: Chima200057 (GitHub)
•   Collaborator: buda360 (GitHub)
•	Contact via GitHub issues on the repository.
