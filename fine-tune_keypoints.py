# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# %% [markdown]
# # RF-DETR keypoint training demo on multiple Roboflow Universe datasets
#
# Select one dataset with `DATASET_KEY`, then run the same fine-tuning, plotting, checkpoint loading, and inference cells.
#
# RF-DETR is a real-time detection transformer that extends bounding-box detection to predict structured keypoint skeletons — useful for human pose estimation, sports field calibration, product inspection, and any task where *where* matters as much as *what*. This cookbook walks you through the full cycle from raw dataset to a deployable checkpoint using four ready-made Roboflow Universe datasets: darts (`"dart"`), human body pose (`"human_pose"`), a basketball court (`"basketball_court"`), and a football pitch (`"football_field"`). Change `DATASET_KEY` in the configuration cell and every subsequent cell adapts automatically. By the end you will have a fine-tuned model, training curves, and annotated inference images with optional uncertainty ellipses.

# %% [markdown]
# ## Setup
#
# Run this setup cell from the checked-out repository root before importing `rfdetr`. If an older installed `rfdetr` package was already imported, restart the runtime after this cell.
#
# Installing from source with `pip install -e ".[train,visual]"` gives you the exact code you see in the repository rather than a potentially older PyPI release. The `train` extra pulls in the training loop dependencies (PyTorch Lightning, COCO evaluation tools, and data augmentation libraries), while `visual` adds the visualisation helpers used later in the notebook. The `roboflow` package handles dataset download; `pandas` and `seaborn` are needed for the metrics table and optional plot styling. If you hit an `ImportError` after running this cell, the most likely cause is a stale in-memory import — restart the Python runtime once and re-run from the top.

# %% [bash]
# !pip install -q ".[train,visual]" roboflow pandas seaborn

# %%
"""Shared RF-DETR keypoint fine-tuning demo for several Roboflow COCO keypoint exports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from roboflow import Roboflow

from rfdetr import RFDETRKeypointPreview
from rfdetr.config import KeypointTrainConfig
from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.utilities.reproducibility import seed_all
from rfdetr.visualize.keypoints import _key_points_for_display, _keypoint_prediction_records
from rfdetr.visualize.training import plot_loss_metrics, plot_map_metrics

# %%
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASETS_DIR = PROJECT_ROOT / "datasets"

DATASETS: dict[str, dict[str, Any]] = {
    "dart": {
        "name": "Darts Detection",
        "workspace": "dartdetection-vfgjd",
        "project": "darts_detection-yp7lt",
        "version": 5,
        "source_url": "https://universe.roboflow.com/dartdetection-vfgjd/darts_detection-yp7lt",
        "output_name": "keypoint_darts_detection_demo",
        "keypoint_flip_pairs": [],
    },
    "human_pose": {
        "name": "Dataset Ridimensionato",
        "workspace": "poseestimation-wzidb",
        "project": "dataset-ridimensionato",
        "version": 65,
        "source_url": "https://universe.roboflow.com/poseestimation-wzidb/dataset-ridimensionato/dataset/65",
        "output_name": "keypoint_dataset_ridimensionato_demo",
        "keypoint_flip_pairs": [],
    },
    "basketball_court": {
        "name": "Basketball Court Detection 2",
        "workspace": "roboflow-jvuqo",
        "project": "basketball-court-detection-2",
        "version": 19,
        "source_url": "https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2",
        "output_name": "keypoint_basketball_court_detection_demo",
        "keypoint_flip_pairs": [],
    },
    "football_field": {
        "name": "Football Field Detection",
        "workspace": "roboflow-jvuqo",
        "project": "football-field-detection-f07vi",
        "version": 1,
        "source_url": "https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi",
        "output_name": "keypoint_football_field_detection_demo",
        "keypoint_flip_pairs": [],
    },
}

DATASET_KEY = "dart"
DATASET_INFO = DATASETS[DATASET_KEY]

OUTPUT_DIR = PROJECT_ROOT / "output" / str(DATASET_INFO["output_name"])
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
VALIDATION_METRICS_JSON = OUTPUT_DIR / "validation_metrics.json"
FINAL_CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_final_demo.pth"

SEED = 7
EPOCHS = 50
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 8
LR = 2e-5
LR_ENCODER = 2e-5
SAMPLE_PREVIEW_COUNT = 6
SAMPLE_PREVIEW_COLUMNS = 3
SAMPLE_PREVIEW_FIGURE_SIZE: tuple[float, float] | None = None
INFERENCE_COUNT = 6
INFERENCE_COLUMNS = 3
INFERENCE_THRESHOLD = 0.25
KEYPOINT_THRESHOLD = 0.1
PLOT_LOSS_LOG_SCALE = False
DRAW_UNCERTAINTY_ELLIPSES = True
ELLIPSE_SIGMA = 1.0
MAX_ELLIPSE_RADIUS = 36.0

print(f"dataset_key={DATASET_KEY}")
print(f"dataset={DATASET_INFO['name']}")
print(f"source_url={DATASET_INFO['source_url']}")


# %% [markdown]
# ## Notebook display
#
# This helper registers the `%matplotlib inline` magic so figures render directly beneath each cell when you run the notebook in Jupyter or Google Colab. If you are running this file as a plain Python script the function detects the absence of an IPython kernel and exits silently, so it is always safe to call.


# %%
def _enable_notebook_inline_matplotlib() -> None:
    """Enable inline matplotlib figures when running in IPython."""
    get_ipython_func = globals().get("get_ipython")
    if not callable(get_ipython_func):
        return
    ipython = get_ipython_func()
    if ipython is not None:
        ipython.run_line_magic("matplotlib", "inline")
        ipython.run_line_magic("config", "InlineBackend.close_figures = True")


_enable_notebook_inline_matplotlib()


# %% [markdown]
# ## 1 - Download dataset
#
# Roboflow exports datasets in COCO keypoint format: a JSON file where each annotation contains a flat `keypoints` array of `[x, y, visibility, x, y, visibility, ...]` triplets, one per defined skeleton joint. The download cell fetches the exact dataset version listed in `DATASETS` and places it under `datasets/<DATASET_KEY>/`. You need a Roboflow API key — get one for free at app.roboflow.com/settings/api and set it as the `ROBOFLOW_API_KEY` environment variable (or as a Colab secret with the same name). The download is idempotent: if the target directory already exists Roboflow skips the network transfer, so re-running this cell after a successful download is fast.


# %%
try:
    from google.colab import userdata

    try:
        ROBOFLOW_API_KEY = userdata.get("ROBOFLOW_API_KEY") or ""
    except Exception:
        ROBOFLOW_API_KEY = ""
except ImportError:
    ROBOFLOW_API_KEY = ""

if not ROBOFLOW_API_KEY:
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
if not ROBOFLOW_API_KEY:
    raise RuntimeError(
        "ROBOFLOW_API_KEY not found. "
        "In Colab: add it via Secrets (key icon). "
        "Locally: set the environment variable before running."
    )

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
dataset = (
    rf.workspace(str(DATASET_INFO["workspace"]))
    .project(str(DATASET_INFO["project"]))
    .version(int(DATASET_INFO["version"]))
    .download("coco", location=str(DATASETS_DIR / DATASET_KEY))
)
DATASET_DIR = Path(dataset.location)
print(f"dataset_dir={DATASET_DIR}")


# %% [markdown]
# ## 2 - Infer keypoint schema
#
# `infer_coco_keypoint_schema` reads the training annotation JSON and extracts three things the model needs: the class names, the number of keypoints per class, and the OKS sigmas. OKS stands for Object Keypoint Similarity — it is the keypoint analogue of IoU and ranges from 0 to 1. Each sigma is a per-keypoint scale factor that controls how strictly localisation is penalised during evaluation: a smaller sigma (e.g. 0.025 for a wrist) means the model must predict that joint more precisely to score well, while a larger sigma (e.g. 0.107 for a hip) is more forgiving. `VALIDATE_KEYPOINT_METRICS` will be `False` when your dataset has categories with different keypoint counts, because the standard COCO OKS evaluator assumes a uniform skeleton across all instances; bounding-box mAP is always computed regardless.


# %%
TRAIN_ANNOTATIONS = DATASET_DIR / "train" / "_annotations.coco.json"
schema = infer_coco_keypoint_schema(TRAIN_ANNOTATIONS)
CLASS_NAMES = schema.class_names
NUM_KEYPOINTS_PER_CLASS = schema.num_keypoints_per_class
NUM_CLASSES = len(CLASS_NAMES)
KEYPOINT_OKS_SIGMAS = schema.keypoint_oks_sigmas
ACTIVE_KEYPOINT_COUNTS = [count for count in NUM_KEYPOINTS_PER_CLASS if count > 0]
VALIDATE_KEYPOINT_METRICS = len(set(ACTIVE_KEYPOINT_COUNTS)) <= 1
KEYPOINT_FLIP_PAIRS = list(DATASET_INFO.get("keypoint_flip_pairs", []))

print(f"class_names={CLASS_NAMES}")
print(f"num_keypoints_per_class={NUM_KEYPOINTS_PER_CLASS}")
print("bbox_validation=True")
print(f"keypoint_oks_validation={VALIDATE_KEYPOINT_METRICS}")
if VALIDATE_KEYPOINT_METRICS:
    print(f"keypoint_oks_sigmas={len(KEYPOINT_OKS_SIGMAS)} values")
else:
    print("keypoint_oks_validation=skipped because COCO OKS requires one keypoint count across categories")


# %% [markdown]
# `_save_final_checkpoint` writes a self-contained `.pth` file that bundles model weights with the full training and model config. This is separate from the PTL checkpoint because it can be loaded with a single `from_checkpoint` call on any machine, without reconstructing the original config.


# %%
def _save_final_checkpoint(
    module: RFDETRModelModule,
    trainer: Any,
    train_config: KeypointTrainConfig,
    model_config: Any,
    output_path: Path,
) -> Path:
    """Save a final RF-DETR .pth checkpoint that can be loaded with ``from_checkpoint``."""
    raw_model: Any = getattr(module.model, "_orig_mod", module.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        BestModelCallback._build_checkpoint_payload(
            raw_model.state_dict(),
            train_config.model_dump(),
            trainer,
            model_name="RFDETRKeypointPreview",
            model_config_dict=model_config.model_dump(),
        ),
        output_path,
    )
    return output_path


# %% [markdown]
# `_annotated_keypoint_scene` renders keypoint dots and, when covariance data is available, uncertainty ellipses on top of an image. Drawing ellipses here rather than in the grid helper keeps the annotation logic in one place and makes it easy to toggle `DRAW_UNCERTAINTY_ELLIPSES` without touching the layout code.


# %%
def _annotated_keypoint_scene(image: np.ndarray, key_points: sv.KeyPoints) -> np.ndarray:
    """Return an image with visible vertices and optional uncertainty ellipses."""
    scene = image.copy()
    if DRAW_UNCERTAINTY_ELLIPSES and "covariance" in key_points.data:
        scene = sv.VertexEllipseAnnotator(
            sigma=ELLIPSE_SIGMA,
            color=sv.Color.ROBOFLOW,
            max_axis=MAX_ELLIPSE_RADIUS,
        ).annotate(scene=scene, key_points=key_points)

    return sv.VertexAnnotator(radius=3).annotate(scene=scene, key_points=key_points)


# %% [markdown]
# `_keypoint_grid_figure` arranges multiple annotated images into a fixed-column matplotlib grid. Using a grid rather than individual figures keeps the visual summary compact and makes it easy to compare predictions across images at a glance.


# %%
def _keypoint_grid_figure(items: list[tuple[str, np.ndarray, sv.KeyPoints]], columns: int) -> Figure:
    """Render keypoint annotations in a fixed-column subplot grid."""
    if columns <= 0:
        raise ValueError(f"columns must be positive, got {columns}.")

    rows = max(1, (len(items) + columns - 1) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 5 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis in axes_array:
        axis.axis("off")

    for axis, (title, image, key_points) in zip(axes_array, items, strict=False):
        axis.imshow(_annotated_keypoint_scene(image, key_points))
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    figure.tight_layout()
    return figure


# %% [markdown]
# `_display_keypoint_records` prints per-image keypoint predictions as a pandas table. Fields that are constant across all keypoints in an image (such as the filename or detection score) are printed once as a header line rather than repeated in every row, keeping the output readable for images with many joints.


# %%
def _display_keypoint_records(records: list[dict[str, Any]]) -> None:
    """Display keypoint rows while printing per-image constant fields once."""
    if not records:
        print("keypoint_rows=[]")
        return

    keypoint_columns = {"detection_index", "keypoint_index", "x", "y", "keypoint_confidence"}
    records_frame = pd.DataFrame(records)
    for image_name, image_frame in records_frame.groupby("image", dropna=False, sort=False):
        constant_columns = [
            column
            for column in image_frame.columns
            if column not in keypoint_columns and image_frame[column].nunique(dropna=False) == 1
        ]
        if constant_columns:
            constants = image_frame.iloc[0][constant_columns]
            print(", ".join(f"{column}={constants[column]}" for column in constant_columns))
        elif image_name is not None:
            print(f"image={image_name}")

        display(image_frame.drop(columns=constant_columns).reset_index(drop=True))


# %% [markdown]
# ## 3 - Configure training
#
# `KeypointTrainConfig` centralises every hyperparameter the training loop needs. A few fields are worth understanding before you customise them for your own dataset. `lr` and `lr_encoder` are usually set to the same value for fine-tuning; if you notice the backbone overfitting early you can halve `lr_encoder` to protect the pre-trained features while the detection head continues to adapt. `grad_accum_steps=2` means gradients are accumulated over two mini-batches before a weight update, which effectively doubles the batch size without requiring extra GPU memory — useful on smaller GPUs. `use_ema=False` keeps the demo fast; for a production run you can set this to `True` to maintain an exponential moving average of weights, which often improves final accuracy by a few tenths of a mAP point. `multi_scale=False` and `expanded_scales=False` disable the multi-resolution augmentation that RF-DETR uses during pre-training; turning them off shortens epoch time significantly and is fine for most fine-tuning runs. The `notes` dict is stored in the checkpoint alongside the weights, giving you a lightweight experiment log that travels with the model file.


# %%
seed_all(SEED)
variant = RFDETRKeypointPreview(  # type: ignore[no-untyped-call]
    num_classes=NUM_CLASSES,
    num_keypoints_per_class=NUM_KEYPOINTS_PER_CLASS,
)
variant.model_config.model_name = type(variant).__name__

train_config = KeypointTrainConfig(
    dataset_file="roboflow",
    dataset_dir=str(DATASET_DIR),
    output_dir=str(OUTPUT_DIR),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    grad_accum_steps=GRAD_ACCUM_STEPS,
    num_workers=NUM_WORKERS,
    lr=LR,
    lr_encoder=LR_ENCODER,
    use_ema=False,
    run_test=False,
    compute_train_metrics=True,
    compute_val_loss=True,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    tensorboard=False,
    wandb=False,
    mlflow=False,
    clearml=False,
    class_names=CLASS_NAMES,
    keypoint_flip_pairs=KEYPOINT_FLIP_PAIRS,
    keypoint_oks_sigmas=KEYPOINT_OKS_SIGMAS if VALIDATE_KEYPOINT_METRICS else None,
    notes={
        "demo": f"keypoint-preview PTL fine-tune on Roboflow Universe {DATASET_INFO['project']}",
        "source_url": DATASET_INFO["source_url"],
        "roboflow_workspace": DATASET_INFO["workspace"],
        "roboflow_project": DATASET_INFO["project"],
        "roboflow_version": DATASET_INFO["version"],
        "num_keypoints_per_class": NUM_KEYPOINTS_PER_CLASS,
        "keypoint_flip_pairs": KEYPOINT_FLIP_PAIRS,
        "bbox_validation_enabled": True,
        "keypoint_oks_validation_enabled": VALIDATE_KEYPOINT_METRICS,
        "keypoint_oks_sigmas": KEYPOINT_OKS_SIGMAS if VALIDATE_KEYPOINT_METRICS else None,
    },
    progress_bar="tqdm",
)

datamodule = RFDETRDataModule(variant.model_config, train_config)
model = RFDETRModelModule(variant.model_config, train_config)
trainer = build_trainer(train_config, variant.model_config)


# %% [markdown]
# ## 4 - Preview dataset inputs
#
# Always look at your data before you start a long training run. This cell renders a grid of annotated training images so you can confirm that keypoint skeletons are aligned with objects, that flips and colour jitter look reasonable, and that there are no systematic labelling errors such as swapped left/right joints or keypoints sitting outside their bounding box. Catching label noise here costs a few seconds; catching it after 50 epochs of training costs much more. If the preview shows empty images or missing annotations, the most common cause is a mismatch between the image filenames in the COCO JSON and the files on disk — check that the download completed fully.


# %%
sample_figure = datamodule._show_samples(
    SAMPLE_PREVIEW_COUNT,
    split="train",
    columns=SAMPLE_PREVIEW_COLUMNS,
    figure_size=SAMPLE_PREVIEW_FIGURE_SIZE,
)
display(sample_figure)
plt.close(sample_figure)


# %% [markdown]
# ## 5 - Fine-tune
#
# `trainer.fit` hands control to PyTorch Lightning, which drives the full training loop: forward pass, loss computation, gradient accumulation, weight updates, learning-rate scheduling, and periodic validation. After each epoch the trainer appends a row to `metrics.csv` and, if validation mAP improves, saves a new best checkpoint via `BestModelCallback`. On a single consumer GPU (RTX 3090 or similar) 50 epochs on the dart dataset takes roughly 15–30 minutes depending on `BATCH_SIZE` and `NUM_WORKERS`. If you need to resume an interrupted run, set `train_config.resume` to the path of the last PTL checkpoint before calling this cell again.


# %%
trainer.fit(model, datamodule=datamodule, ckpt_path=train_config.resume or None)
print(f"output_dir={OUTPUT_DIR}")


# %% [markdown]
# ## 6 - Save checkpoint/model
#
# PTL saves its own checkpoints during training (optimizer state, scheduler state, epoch counter), but those files are not directly portable — they require the same class hierarchy to load. The `.pth` file written here is a self-contained RF-DETR checkpoint: it bundles the model weights together with the full training and model configs, including the keypoint schema and class names. This means you can share the file with a colleague and they can run inference with a single `RFDETRKeypointPreview.from_checkpoint(path)` call, with no need to reconstruct the original config. For full reproducibility, keep this checkpoint alongside the dataset version number printed in section 1.


# %%
final_checkpoint = _save_final_checkpoint(model, trainer, train_config, variant.model_config, FINAL_CHECKPOINT_PATH)
print(f"saved_checkpoint_model={final_checkpoint}")


# %% [markdown]
# ## 7 - Validate metrics
#
# The validation pass that runs at the end of every training epoch uses the best weights seen so far and applies augmentation. This cell runs a clean post-training validation: no augmentation, the best checkpoint loaded from `checkpoint_best_total.pth` written by `BestModelCallback`, and results serialised to JSON for downstream comparison. Two metrics are most important to inspect here. `bbox/map` is the standard COCO bounding-box mAP at IoU 0.50:0.95 — it tells you how reliably the detector finds and bounds each object. `keypoint/oks` is the OKS-based mAP — it measures how precisely the model places each joint within the detected bounding boxes. A model can have a high `bbox/map` but a low `keypoint/oks` if it finds objects well but struggles to localise their joints; addressing that usually means more labelled data or tighter OKS sigmas.


# %%
validation_results = trainer.validate(
    model, datamodule=datamodule, ckpt_path=str(OUTPUT_DIR / "checkpoint_best_total.pth")
)
validation_metrics = {key: float(value) for key, value in validation_results[0].items()} if validation_results else {}
if not VALIDATE_KEYPOINT_METRICS:
    validation_metrics["keypoint_oks_skipped_mixed_keypoint_counts"] = 1.0
VALIDATION_METRICS_JSON.write_text(json.dumps(validation_metrics, indent=2, sort_keys=True), encoding="utf-8")
print(f"validation_metrics={validation_metrics}")
print(f"validation_metrics_json={VALIDATION_METRICS_JSON}")


# %% [markdown]
# ## 8 - Plot CSVLogger metrics
#
# Reading loss curves is one of the fastest ways to diagnose training problems. Healthy runs show both training loss and validation loss decreasing together; a large and growing gap between the two is a classic overfitting signal — the model is memorising the training set rather than generalising. For the mAP curves, on small datasets (a few hundred images) you typically see rapid improvement in the first 10–20 epochs followed by a plateau around epoch 30–50; if mAP is still climbing at epoch 50, consider extending `EPOCHS`. Set `PLOT_LOSS_LOG_SCALE = True` if the loss drops by an order of magnitude in the first few epochs and the later, more meaningful portion of the curve gets compressed into a flat line at the bottom of the plot.


# %%
print(f"metrics_csv={METRICS_CSV}")
loss_figure = plot_loss_metrics(str(METRICS_CSV), loss_log_scale=PLOT_LOSS_LOG_SCALE)
display(loss_figure)
plt.close(loss_figure)

# %%
map_figure = plot_map_metrics(str(METRICS_CSV))
display(map_figure)
plt.close(map_figure)


# %% [markdown]
# ## 9 - Load checkpoint/model
#
# `from_checkpoint` is the standard entry point for loading a saved RF-DETR model. It reads both the weights and the stored config from the `.pth` file, so the reconstructed model has the correct number of classes and keypoints without you having to pass them explicitly. You can share this checkpoint file with teammates and they can run inference immediately — the keypoint schema, class names, and OKS sigmas are all embedded in the file alongside the weights. This cell also confirms that the save-load round trip works before you proceed to inference, so any corruption or version mismatch surfaces here rather than silently producing wrong predictions later.


# %%
loaded_model = RFDETRKeypointPreview.from_checkpoint(FINAL_CHECKPOINT_PATH)


# %% [markdown]
# ## 10 - Select inference images
#
# This cell implements a fallback chain — test split first, then validation, then train — so the notebook always finds images to run inference on even when the dataset has no dedicated test split. Using images from the test set gives you an unbiased view of model performance because those images were never seen during training or used to pick the best checkpoint. If you want to run inference on your own images, replace `inference_image_paths` with a list of `Path` objects pointing to your files before running the next cell.


# %%
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


validation_image_paths = _find_images(DATASET_DIR / "test")
if not validation_image_paths:
    validation_image_paths = _find_images(DATASET_DIR / "valid")
if not validation_image_paths:
    validation_image_paths = _find_images(DATASET_DIR / "train")
if not validation_image_paths:
    raise FileNotFoundError(f"No images ({', '.join(sorted(_IMAGE_EXTS))}) found under {DATASET_DIR}")

inference_image_paths = validation_image_paths[:INFERENCE_COUNT]
print(f"inference_images={[str(path) for path in inference_image_paths]}")


# %% [markdown]
# ## 11 - Visualize inference and keypoint table
#
# Two thresholds control what you see in this cell and they operate at different levels. `INFERENCE_THRESHOLD` is a detection confidence threshold: only object detections whose bounding-box score exceeds this value are shown at all. Raise it to suppress false positives; lower it to surface low-confidence detections you might want to investigate. `KEYPOINT_THRESHOLD` operates within each accepted detection: individual keypoints whose confidence falls below this value are treated as not visible and omitted from the overlay. The uncertainty ellipses (drawn when `DRAW_UNCERTAINTY_ELLIPSES = True`) visualise the predicted covariance for each keypoint — a larger ellipse means the model is less certain about that joint's exact location. Below the image grid, the keypoint table gives you the raw coordinates and per-keypoint confidence scores in a format that is easy to copy into a spreadsheet or feed into downstream analysis.


# %%
inference_grid_items = []
keypoint_rows = []
for image_path in inference_image_paths:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        key_points_raw = loaded_model.predict(image, threshold=INFERENCE_THRESHOLD)
    if not isinstance(key_points_raw, sv.KeyPoints):
        raise RuntimeError(
            f"Expected RFDETRKeypointPreview.predict() to return sv.KeyPoints, got {type(key_points_raw)!r}."
        )

    key_points = _key_points_for_display(key_points_raw, keypoint_threshold=KEYPOINT_THRESHOLD)
    rows = _keypoint_prediction_records(key_points, image=image_path, keypoint_threshold=KEYPOINT_THRESHOLD)
    inference_grid_items.append((image_path.name, np.array(image), key_points))
    keypoint_rows.extend(rows)

if inference_grid_items:
    figure = _keypoint_grid_figure(inference_grid_items, columns=INFERENCE_COLUMNS)
    display(figure)
    plt.close(figure)

_display_keypoint_records(keypoint_rows)
