---
description: Train RF-DETR detection and segmentation models on custom datasets. Supports COCO and YOLO formats with one-line Python API and PyTorch Lightning.
---

# Train an RF-DETR Model

!!! tip "Key Takeaways"

    - Train detection, segmentation, or keypoint preview models with a single `model.train(dataset_dir=...)` call
    - Detection and segmentation support COCO JSON and YOLO dataset formats with automatic detection
    - Keypoint preview training supports COCO keypoint JSON and Ultralytics YOLO pose datasets
    - Fine-tune from COCO-pretrained checkpoints (Nano to 2XLarge) for fastest convergence
    - Built on PyTorch Lightning — use the high-level API or access PTL primitives directly for full control
    - EMA weights, early stopping, and best-model checkpointing are included by default

You can train RF-DETR object detection and segmentation models on a custom dataset using the `rfdetr` Python package, or in the cloud using Roboflow.

This guide describes how to train both an object detection and segmentation RF-DETR model.

## Training paths

RF-DETR provides two training paths:

| Path                                        | When to use                                                                                                                                         |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`RFDETR.train()`** (this page)            | Quickstart, fine-tuning with standard options, Colab notebooks. One call sets up and runs everything.                                               |
| **[Custom Training API](customization.md)** | Custom callbacks, alternative loggers, multi-GPU strategies, integration with external frameworks, or any other customisation of the training loop. |

Both paths run the same underlying PyTorch Lightning stack. `RFDETR.train()` constructs `RFDETRModelModule`, `RFDETRDataModule`, and a `Trainer` internally; the Lightning API page shows how to do the same thing explicitly so you can modify each component.

## Quick Start

!!! info "Training requires the `train` extra"

    Training dependencies are not included in the base install. Install them with:

    ```bash
    pip install "rfdetr[train]"
    ```

    For experiment tracking, also add `pip install "rfdetr[train,loggers]"`.

RF-DETR supports training on datasets in both **COCO** and **YOLO** formats. The format is automatically detected based on the structure of your dataset directory.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="<DATASET_PATH>",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="<OUTPUT_PATH>",
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir="<DATASET_PATH>",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="<OUTPUT_PATH>",
    )
    ```

=== "Keypoint Preview"

    ```python
    from rfdetr import RFDETRKeypointPreview

    model = RFDETRKeypointPreview()

    model.train(
        dataset_dir="<KEYPOINT_DATASET_PATH>",
        epochs=50,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-5,
        output_dir="<OUTPUT_PATH>",
    )
    ```

Different GPUs have different VRAM capacities, so adjust batch_size and grad_accum_steps to maintain a total batch size of 16. For example, on a powerful GPU like the A100, use `batch_size=16` and `grad_accum_steps=1`; on smaller GPUs like the T4, use `batch_size=4` and `grad_accum_steps=4`. This gradient accumulation strategy helps train effectively even with limited memory.

Each model class downloads its COCO-pretrained checkpoint automatically when instantiated. To get started quickly with training an object detection model, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

## Keypoint preview custom datasets

The pretrained keypoint preview checkpoint predicts 17 COCO person keypoints. Fine-tuned keypoint preview models can use the keypoint schema from your own COCO or YOLO pose dataset, so the output keypoint count is not limited to 17.

Use COCO keypoint JSON or Ultralytics YOLO pose labels for custom keypoint training. Roboflow COCO exports are supported when split annotations are named `train/_annotations.coco.json`, `valid/_annotations.coco.json`, and optionally `test/_annotations.coco.json`. YOLO pose datasets use the existing RF-DETR YOLO directory layout with `data.yaml`, `train/images`, `train/labels`, `valid/images`, and `valid/labels`.

The keypoint fine-tuning demo infers the class names and keypoint schema from the training annotation file, then passes those values into `RFDETRKeypointPreview` and `model.train()`:

```python
from pathlib import Path

from rfdetr import RFDETRKeypointPreview
from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema

DATASET_DIR = Path("/path/to/coco-keypoint-dataset")
schema = infer_coco_keypoint_schema(DATASET_DIR / "train" / "_annotations.coco.json")

model = RFDETRKeypointPreview(
    num_classes=len(schema.class_names),
    num_keypoints_per_class=schema.num_keypoints_per_class,
)

model.train(
    dataset_file="roboflow",
    dataset_dir=str(DATASET_DIR),
    class_names=schema.class_names,
    keypoint_oks_sigmas=schema.keypoint_oks_sigmas,
    epochs=50,
    batch_size=8,
    grad_accum_steps=2,
    lr=2e-5,
    lr_encoder=2e-5,
    output_dir="output/keypoint_custom",
    use_ema=False,
    run_test=False,
)
```

Set `keypoint_flip_pairs` if horizontal flips should swap left/right keypoints for your schema.

For YOLO pose datasets, use `infer_yolo_keypoint_schema(DATASET_DIR / "data.yaml")` instead. RF-DETR also infers YOLO pose schema automatically during `model.train()` when `data.yaml` declares `kpt_shape`.

## Dataset Format

RF-DETR **automatically detects** whether your dataset is in COCO or YOLO format. Simply pass your dataset directory to the `train()` method and the appropriate data loader will be used.

| Format   | Detection Method                         | Learn More                                          |
| -------- | ---------------------------------------- | --------------------------------------------------- |
| **COCO** | Looks for `train/_annotations.coco.json` | [COCO Format Guide](dataset-formats.md#coco-format) |
| **YOLO** | Looks for `data.yaml` + `train/images/`  | [YOLO Format Guide](dataset-formats.md#yolo-format) |

For keypoint preview training, use COCO keypoint JSON or YOLO pose labels. YOLO pose datasets must declare
`kpt_shape` in `data.yaml`; detection-only YOLO datasets still fail clearly in keypoint mode instead of being treated as
pose labels.

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch and export them in either COCO JSON or YOLO format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

→ **[Learn more about dataset formats](dataset-formats.md)**

## Training Configuration

RF-DETR provides many configuration options to customize your training run. See the complete reference for all available parameters.

→ **[View all training parameters](training-parameters.md)**

## Advanced Topics

- [Resume training](advanced.md#resume-training) from a checkpoint
- [Early stopping](advanced.md#early-stopping) to prevent overfitting
- [Multi-GPU training](advanced.md#multi-gpu-training) with PyTorch Lightning DDP
- [Custom augmentations with Albumentations](augmentations.md) - Dedicated guide
- [Memory optimization](advanced.md#memory-optimization) with gradient checkpointing

→ **[Learn more about advanced training](advanced.md)**

## Custom Training API

RF-DETR's training stack is built on PyTorch Lightning. The `RFDETR.train()` call above constructs and runs PTL primitives internally. Use them directly when you need custom callbacks, non-default loggers, multi-GPU strategies, or full control over the training loop.

→ **[Custom Training API guide](customization.md)**

## Training Loggers

Track your experiments with popular logging platforms:

- [TensorBoard](loggers.md#tensorboard) for local visualization
- [Weights and Biases](loggers.md#weights-and-biases) for cloud-based tracking
- [ClearML](loggers.md#clearml) workaround for SDK auto-binding
- [MLflow](loggers.md#mlflow) for experiment lifecycle management

→ **[Learn more about training loggers](loggers.md)**

## Result Checkpoints

During training, multiple model checkpoints are saved to the output directory:

- `checkpoint.pth` – the most recent checkpoint, saved at the end of the latest epoch.

- `checkpoint_<number>.pth` – periodic checkpoints saved every N epochs (default is every 10).

- `checkpoint_best_ema.pth` – best checkpoint based on validation score, using the EMA (Exponential Moving Average) weights. EMA weights are a smoothed version of the model's parameters across training steps, often yielding better generalization.

- `checkpoint_best_regular.pth` – best checkpoint based on validation score, using the raw (non-EMA) model weights.

- `checkpoint_best_total.pth` – final checkpoint selected for inference and benchmarking. It contains only the model weights (no optimizer state or scheduler) and is chosen as the better of the EMA and non-EMA models based on validation performance.

For detection and segmentation models, the validation score is box mAP (`val/mAP_50_95`). For keypoint preview models,
best-checkpoint selection uses COCO keypoint AP (`val/keypoint_map_50_95`) and checkpoints persist the model keypoint
schema so `RFDETR.from_checkpoint()` can reconstruct the same label/keypoint slots.

??? note "Checkpoint file sizes"

    Checkpoint sizes vary based on what they contain:

    - **Training checkpoints** (e.g. `checkpoint.pth`, `checkpoint_<number>.pth`) include model weights, optimizer state, scheduler state, and training metadata. Use these to resume training.

    - **Evaluation checkpoints** (e.g. `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`) store only the model weights — either EMA or raw — and are used to track the best-performing models. These may come from different epochs depending on which version achieved the highest validation score.

    - **Stripped checkpoint** (e.g. `checkpoint_best_total.pth`) contains only the final model weights and is optimized for inference and deployment.

## Load and Run Fine-Tuned Model

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<CHECKPOINT_PATH>")

    detections = model.predict("<IMAGE_PATH>")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<CHECKPOINT_PATH>")

    detections = model.predict("<IMAGE_PATH>")
    ```

## Next Steps

After training your model, you can:

- [Export your model to ONNX](../export.md) for deployment with various inference frameworks
- [Deploy to Roboflow](../deploy.md) for cloud-based inference and workflow integration
