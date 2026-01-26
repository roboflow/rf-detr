# Train an RF-DETR Model

You can train RF-DETR object detection and segmentation models on a custom dataset using the `rfdetr` Python package, or in the cloud using Roboflow.

This guide describes how to train both an object detection and segmentation RF-DETR model.

## Quick Start

RF-DETR supports training on datasets in both **COCO** and **YOLO** formats. The format is automatically detected based on the structure of your dataset directory.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>
    )
    ```

Different GPUs have different VRAM capacities, so adjust batch_size and grad_accum_steps to maintain a total batch size of 16. For example, on a powerful GPU like the A100, use `batch_size=16` and `grad_accum_steps=1`; on smaller GPUs like the T4, use `batch_size=4` and `grad_accum_steps=4`. This gradient accumulation strategy helps train effectively even with limited memory.

For object detection, the RF-DETR-B checkpoint is used by default. To get started quickly with training an object detection model, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

## Dataset Format

RF-DETR **automatically detects** whether your dataset is in COCO or YOLO format. Simply pass your dataset directory to the `train()` method and the appropriate data loader will be used.

| Format | Detection Method | Learn More |
|--------|------------------|------------|
| **COCO** | Looks for `train/_annotations.coco.json` | [COCO Format Guide](dataset-formats.md#coco-format) |
| **YOLO** | Looks for `data.yaml` + `train/images/` | [YOLO Format Guide](dataset-formats.md#yolo-format) |

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch and export them in either COCO JSON or YOLO format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

→ **[Learn more about dataset formats](dataset-formats.md)**

## Training Configuration

RF-DETR provides many configuration options to customize your training run. See the complete reference for all available parameters.

→ **[View all training parameters](training-parameters.md)**

## Advanced Topics

- [Resume training](advanced.md#resume-training) from a checkpoint
- [Early stopping](advanced.md#early-stopping) to prevent overfitting
- [Multi-GPU training](advanced.md#multi-gpu-training) with PyTorch DDP
- [Logging with TensorBoard](advanced.md#logging-with-tensorboard)
- [Logging with Weights and Biases](advanced.md#logging-with-weights-and-biases)

→ **[Learn more about advanced training](advanced.md)**

## Result Checkpoints

During training, multiple model checkpoints are saved to the output directory:

- `checkpoint.pth` – the most recent checkpoint, saved at the end of the latest epoch.

- `checkpoint_<number>.pth` – periodic checkpoints saved every N epochs (default is every 10).

- `checkpoint_best_ema.pth` – best checkpoint based on validation score, using the EMA (Exponential Moving Average) weights. EMA weights are a smoothed version of the model's parameters across training steps, often yielding better generalization.

- `checkpoint_best_regular.pth` – best checkpoint based on validation score, using the raw (non-EMA) model weights.

- `checkpoint_best_total.pth` – final checkpoint selected for inference and benchmarking. It contains only the model weights (no optimizer state or scheduler) and is chosen as the better of the EMA and non-EMA models based on validation performance.

??? note "Checkpoint file sizes"

    Checkpoint sizes vary based on what they contain:

    - **Training checkpoints** (e.g. `checkpoint.pth`, `checkpoint_<number>.pth`) include model weights, optimizer state, scheduler state, and training metadata. Use these to resume training.

    - **Evaluation checkpoints** (e.g. `checkpoint_best_ema.pth`, `checkpoint_best_regular.pth`) store only the model weights — either EMA or raw — and are used to track the best-performing models. These may come from different epochs depending on which version achieved the highest validation score.

    - **Stripped checkpoint** (e.g. `checkpoint_best_total.pth`) contains only the final model weights and is optimized for inference and deployment.

## Load and Run Fine-Tuned Model

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

## Next Steps

After training your model, you can:

- [Export your model to ONNX](../export.md) for deployment with various inference frameworks
- [Deploy to Roboflow](../deploy.md) for cloud-based inference and workflow integration
