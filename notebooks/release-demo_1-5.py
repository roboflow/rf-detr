# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# %% [markdown]
# # RF-DETR 1.5.0 — Custom Augmentations
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/rf-detr/blob/develop/notebooks/release-demo_1-5.ipynb)
#
# This notebook showcases the major new features in RF-DETR 1.5.0.
# The headline addition is **custom training augmentations via Albumentations** —
# a flexible system that lets you control exactly how images are transformed during
# training, with bounding boxes and segmentation masks kept in sync automatically.
# The release also brings **live progress bars** and structured on-screen logs so
# you can track training at a glance.
#
# You will learn how to:
# - Use built-in augmentation presets (`AUG_CONSERVATIVE`, `AUG_AGGRESSIVE`, `AUG_AERIAL`, `AUG_INDUSTRIAL`)
# - Define a fully custom augmentation pipeline
# - Visually inspect augmented samples *before* committing to a full training run
# - Train RF-DETR with your chosen augmentation config

# %% [markdown]
# ## 1. Install RF-DETR 1.5.0
#
# Install directly from the tagged release archive on GitHub.
# Using a pinned tag ensures you get exactly the 1.5.0 feature set shown here.

# %%
!pip install -q https://github.com/roboflow/rf-detr/archive/refs/tags/1.5.0.rc1.zip

# %% [markdown]
# ## 2. Check GPU availability
#
# RF-DETR trains on GPU when one is available and falls back to CPU otherwise.
# The cell below detects your device and prints VRAM size — a useful sanity check
# before choosing batch size later on.

# %%
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 3. Download COCO 2017
#
# We use the official train/val splits — no manual splitting needed.
#
# | Split | Images | Size |
# |---|---|---|
# | `train2017` | ~118 000 | ~18 GB |
# | `val2017` | 5 000 | ~1 GB |
# | annotations | — | ~241 MB |

# %%
!wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip -O train2017.zip
!wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
!wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations.zip

!unzip -q train2017.zip
!unzip -q val2017.zip
!unzip -q annotations.zip

# %% [markdown]
# ### Set up the dataset directory structure
#
# `model.train()` expects:
# ```
# dataset/
#   train/  _annotations.coco.json  + images
#   valid/  _annotations.coco.json  + images
# ```

# %%
!mkdir -p coco_demo
!mv train2017 coco_demo/train
!mv val2017   coco_demo/valid
!cp annotations/instances_train2017.json coco_demo/train/_annotations.coco.json
!cp annotations/instances_val2017.json   coco_demo/valid/_annotations.coco.json

# %%
DATASET_DIR = "coco_demo"
print("Dataset ready.")

# %% [markdown]
# ## 4. Explore the built-in augmentation presets
#
# RF-DETR ships four ready-made presets tuned for common use cases.
# All presets are plain Python dicts — inspect, merge, or extend them freely.
# Each key is an Albumentations transform name; the value is a dict of its
# constructor kwargs (including `p`, the probability of applying it).

# %%
import rfdetr.datasets.aug_config as aug_config
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

for name in ("AUG_CONSERVATIVE", "AUG_AGGRESSIVE", "AUG_AERIAL", "AUG_INDUSTRIAL"):
    preset = getattr(aug_config, name)
    print(f"\n{name}:")
    for transform, params in preset.items():
        print(f"  {transform}: {params}")

# %% [markdown]
# ## 5. Inspect augmented samples before training
#
# Set `save_dataset_grids=True` to write 3×3 JPEG grids of augmented training
# and validation images to `output_dir` **before any weight updates occur**.
# Use this to sanity-check your pipeline in seconds.
#
# Saved files:
# ```
# output/
#   train_batch0_grid.jpg  train_batch1_grid.jpg  train_batch2_grid.jpg
#   val_batch0_grid.jpg    val_batch1_grid.jpg    val_batch2_grid.jpg
# ```

# %%
import os

from rfdetr import RFDETRNano  # smallest & fastest model — ideal for demos

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = RFDETRNano()
model.train(
    dataset_dir=str(DATASET_DIR),
    epochs=1, # one epoch is enough to generate the grids
    batch_size=12,
    aug_config=AUG_AGGRESSIVE,
    save_dataset_grids=True,
    output_dir=OUTPUT_DIR,
    device=device,
    run_test=False,
    progress_bar=True,
)

# %% [markdown]
# ### Display the saved grids inline

# %%
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

grids = sorted(Path(OUTPUT_DIR).glob("*_grid.jpg"))
if grids:
    fig, axes = plt.subplots(len(grids), 1, figsize=(6, 6 * len(grids)))
    if len(grids) == 1:
        axes = [axes]
    for ax, grid_path in zip(axes, grids):
        ax.imshow(mpimg.imread(grid_path))
        ax.set_title(grid_path.name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Choose your augmentation config
#
# | Option | When to use |
# |---|---|
# | `AUG_CONSERVATIVE` | Small datasets < 500 images |
# | `AUG_AGGRESSIVE` | Large datasets 2000+ images |
# | `AUG_AERIAL` | Satellite / overhead imagery |
# | `AUG_INDUSTRIAL` | Manufacturing / inspection data |
# | custom dict | Full control over every transform |
# | `{}` | Disable all augmentations |

# %%
# --- Option A: built-in preset ---
AUG_CONFIG = AUG_AGGRESSIVE

# --- Option B: extend a preset ---
# AUG_CONFIG = {**AUG_AGGRESSIVE, "VerticalFlip": {"p": 0.3}}

# --- Option C: fully custom ---
# AUG_CONFIG = {
#     "HorizontalFlip": {"p": 0.5},
#     "Rotate": {"limit": 15, "p": 0.3},
#     "RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.4},
#     "GaussianBlur": {"blur_limit": 3, "p": 0.2},
# }

# --- Option D: no augmentations ---
# AUG_CONFIG = {}

print("Selected augmentation config:")
for transform, params in AUG_CONFIG.items():
    print(f"  {transform}: {params}")

# %% [markdown]
# ## 7. Train with augmentations
#
# Now run a full training run with the augmentation config chosen above.
# 1.5.0 adds a **live progress bar** per epoch so you can monitor batch-level
# progress without digging through raw log output. Metrics are printed in a
# structured table at the end of each epoch.

# %%
model = RFDETRNano()
model.train(
    dataset_dir=str(DATASET_DIR),
    epochs=2,
    batch_size=24,
    aug_config=AUG_CONFIG,
    output_dir=OUTPUT_DIR,
    device=device,
    progress_bar=True,
    run_test=False,
)

# %% [markdown]
# ## 8. Run inference
#
# Load a sample image and run the trained model. `model.predict()` returns a
# `supervision` `Detections` object, which makes it straightforward to draw
# bounding boxes and labels with the `supervision` annotators.

# %%
import requests
import supervision as sv
from PIL import Image

from rfdetr.util.coco_classes import COCO_CLASSES

image_url = "https://media.roboflow.com/dog.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

detections = model.predict(image, threshold=0.5)

labels = [COCO_CLASSES[class_id] for class_id in detections.class_id]
annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
sv.plot_image(annotated)

# %% [markdown]
# ## Next steps
#
# - [Augmentation docs](https://roboflow.github.io/rf-detr/learn/train/augmentations/)
# - [Advanced training options](https://roboflow.github.io/rf-detr/learn/train/advanced/)
# - [Logger integrations (ClearML, MLflow, W&B)](https://roboflow.github.io/rf-detr/learn/train/loggers/)
# - [Export your model](https://roboflow.github.io/rf-detr/learn/export/)
