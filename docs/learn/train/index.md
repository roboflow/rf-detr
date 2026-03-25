# Train an RF-DETR Model

You can train RF-DETR object detection, segmentation, and pose estimation models on a custom dataset using the `rfdetr` Python package, or in the cloud using Roboflow.

This guide describes how to train object detection, segmentation, and pose estimation RF-DETR models.

### Dataset structure

RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: `train`, `valid`, and `test`. Each sub-directory should contain its own `_annotations.coco.json` file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

[Roboflow](https://roboflow.com/annotate) allows you to create object detection datasets from scratch or convert existing datasets from formats like YOLO, and then export them in COCO JSON format for training. You can also explore [Roboflow Universe](https://universe.roboflow.com/) to find pre-labeled datasets for a range of use cases.

If you are training a segmentation model, your COCO JSON annotations should have a `segmentation` key with the polygon associated with each annotation.

If you are training a pose estimation model, your COCO JSON annotations should have a `keypoints` key with the keypoint coordinates in the format `[x1, y1, v1, x2, y2, v2, ...]` where `v` is the visibility flag (0=not labeled, 1=labeled but not visible, 2=labeled and visible).

### Category ID handling

RF-DETR automatically handles both 0-indexed and 1-indexed category IDs in your COCO annotations. You don't need to modify your dataset - the model will work correctly regardless of how your `category_id` values are numbered.

| Dataset category_ids | What happens |
|---------------------|--------------|
| `[0, 1, 2]` (0-indexed) | Identity mapping, no change |
| `[1, 2, 3]` (1-indexed) | Automatically mapped to `[0, 1, 2]` internally |
| `[1, 5, 10]` (gaps) | Mapped to contiguous `[0, 1, 2]` |

**How it works:**

- **Training**: Category IDs are mapped to contiguous 0-indexed labels for the model
- **Evaluation**: Predictions are mapped back to original category IDs for correct COCO metrics
- **Prediction**: `model.predict()` returns 0-indexed class labels that directly index into your `class_names` list

```python
detections = model.predict("image.jpg")
class_ids = detections.class_id  # 0-indexed: [0, 1, 2, ...]

# Directly index into class_names
class_names = ["cat", "dog", "bird"]
for class_id in class_ids:
    print(class_names[class_id])  # Works correctly
```

This means you can use datasets exported from Roboflow (which use 1-indexed IDs) or any other source without modification.

## Start Training

You can fine-tune RF-DETR from pre-trained COCO checkpoints.

For object detection, the RF-DETR-B checkpoint is used by default. To get started quickly with training an object detection model, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

For image segmentation, the RF-DETR-Seg (Preview) checkpoint is used by default.
## Training paths

RF-DETR provides two training paths:

| Path                                        | When to use                                                                                                                                         |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`RFDETR.train()`** (this page)            | Quickstart, fine-tuning with standard options, Colab notebooks. One call sets up and runs everything.                                               |
| **[Custom Training API](customization.md)** | Custom callbacks, alternative loggers, multi-GPU strategies, integration with external frameworks, or any other customisation of the training loop. |

Both paths run the same underlying PyTorch Lightning stack. `RFDETR.train()` constructs `RFDETRModelModule`, `RFDETRDataModule`, and a `Trainer` internally; the Lightning API page shows how to do the same thing explicitly so you can modify each component.

## Quick Start

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

    For custom keypoints (e.g., 2 keypoints for start/end points):

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

=== "Pose Estimation"

    RF-DETR Pose is available in multiple sizes. Choose based on your speed/accuracy needs:

    | Model | Resolution | Speed | Import |
    |-------|------------|-------|--------|
    | Nano | 384 | Fastest | `RFDETRPoseNano` |
    | Small | 512 | Fast | `RFDETRPoseSmall` |
    | Medium | 576 | Medium | `RFDETRPoseMedium` |
    | Large | 768 | Slow | `RFDETRPoseLarge` |

    ```python
    from rfdetr import RFDETRPoseNano  # or RFDETRPoseSmall, RFDETRPoseMedium, RFDETRPoseLarge

    model = RFDETRPoseNano(
        num_keypoints=17,  # Number of keypoints to detect
    )

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
    )
    ```

    For custom keypoints (e.g., 2 keypoints for start/end points):

    ```python
    from rfdetr import RFDETRPoseNano

    model = RFDETRPoseNano(
        num_keypoints=2,
        keypoint_names=["start", "end"],
        skeleton=[[0, 1]],  # Connect start to end
    )

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
    )
    ```

Different GPUs have different VRAM capacities, so adjust batch_size and grad_accum_steps to maintain a total batch size of 16. For example, on a powerful GPU like the A100, use `batch_size=16` and `grad_accum_steps=1`; on smaller GPUs like the T4, use `batch_size=4` and `grad_accum_steps=4`. This gradient accumulation strategy helps train effectively even with limited memory.

For object detection, the RF-DETR-B checkpoint is used by default. To get started quickly with training an object detection model, please refer to our fine-tuning Google Colab [notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb).

## Dataset Format

RF-DETR **automatically detects** whether your dataset is in COCO or YOLO format. Simply pass your dataset directory to the `train()` method and the appropriate data loader will be used.

| Format   | Detection Method                         | Learn More                                          |
| -------- | ---------------------------------------- | --------------------------------------------------- |
| **COCO** | Looks for `train/_annotations.coco.json` | [COCO Format Guide](dataset-formats.md#coco-format) |
| **YOLO** | Looks for `data.yaml` + `train/images/`  | [YOLO Format Guide](dataset-formats.md#yolo-format) |

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
- [ClearML](loggers.md#clearml) for MLOps automation
- [MLflow](loggers.md#mlflow) for experiment lifecycle management

→ **[Learn more about training loggers](loggers.md)**

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

    model = RFDETRMedium(pretrain_weights="<CHECKPOINT_PATH>")

    detections = model.predict("<IMAGE_PATH>")
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<CHECKPOINT_PATH>")

    detections = model.predict("<IMAGE_PATH>")
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        resume=<CHECKPOINT_PATH>
    )
    ```

=== "Pose Estimation"

    ```python
    from rfdetr import RFDETRPoseNano  # Use the same size as original training

    model = RFDETRPoseNano(num_keypoints=2)  # Match your keypoint config

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        resume=<CHECKPOINT_PATH>
    )
    ```


### Early stopping

Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This can reduce wasted computation once the model converges. Additional parameters—such as `early_stopping_patience`, `early_stopping_min_delta`, and `early_stopping_use_ema`—let you fine-tune the stopping behavior.

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        early_stopping=True
    )
    ```

## Next Steps

After training your model, you can:

- [Export your model to ONNX](../export.md) for deployment with various inference frameworks
- [Deploy to Roboflow](../deploy.md) for cloud-based inference and workflow integration

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview()

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        early_stopping=True
    )
    ```

=== "Pose Estimation"

    ```python
    from rfdetr import RFDETRPoseNano

    model = RFDETRPoseNano(num_keypoints=2)

    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        early_stopping=True
    )
    ```


### Multi-GPU training

You can fine-tune RF-DETR on multiple GPUs using PyTorch’s Distributed Data Parallel (DDP). Create a `main.py` script that initializes your model and calls `.train()` as usual than run it in terminal.

```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py
```

Replace `8` in the `--nproc_per_node argument` with the number of GPUs you want to use. This approach creates one training process per GPU and splits the workload automatically. Note that your effective batch size is multiplied by the number of GPUs, so you may need to adjust your `batch_size` and `grad_accum_steps` to maintain the same overall batch size.

### Logging with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit that helps you visualize and track training metrics. With TensorBoard set up, you can train your model and keep an eye on the logs to monitor performance, compare experiments, and optimize model training. To enable logging, simply pass `tensorboard=True` when training the model.

<details>
<summary>Using TensorBoard with RF-DETR</summary>

<br>

- TensorBoard logging requires additional packages. Install them with:

    ```bash
    pip install "rfdetr[metrics]"
    ```
  
- To activate logging, pass the extra parameter `tensorboard=True` to `.train()`:

    ```python
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    
    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        tensorboard=True
    )
    ```

- To use TensorBoard locally, navigate to your project directory and run:

    ```bash
    tensorboard --logdir <OUTPUT_DIR>
    ```

    Then open `http://localhost:6006/` in your browser to view your logs.

- To use TensorBoard in Google Colab run:

    ```bash
    %load_ext tensorboard
    %tensorboard --logdir <OUTPUT_DIR>
    ```
      
</details>

### Logging with Weights and Biases

[Weights and Biases (W&B)](https://www.wandb.ai) is a powerful cloud-based platform that helps you visualize and track training metrics. With W&B set up, you can monitor performance, compare experiments, and optimize model training using its rich feature set. To enable logging, simply pass `wandb=True` when training the model.

<details>
<summary>Using Weights and Biases with RF-DETR</summary>

<br>

- Weights and Biases logging requires additional packages. Install them with:

    ```bash
    pip install "rfdetr[metrics]"
    ```

- Before using W&B, make sure you are logged in:

    ```bash
    wandb login
    ```

    You can retrieve your API key at wandb.ai/authorize.

- To activate logging, pass the extra parameter `wandb=True` to `.train()`:

    ```python
    from rfdetr import RFDETRBase
    
    model = RFDETRBase()
    
    model.train(
        dataset_dir=<DATASET_PATH>,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=<OUTPUT_PATH>,
        wandb=True,
        project=<PROJECT_NAME>,
        run=<RUN_NAME>
    )
    ```

    In W&B, projects are collections of related machine learning experiments, and runs are individual sessions where training or evaluation happens. If you don't specify a name for a run, W&B will assign a random one automatically.
  
</details>

### Load and run fine-tuned model

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

    detections = model.predict(<IMAGE_PATH>)
    ```

=== "Pose Estimation"

    ```python
    from rfdetr import RFDETRPoseNano  # Use the same size as training

    model = RFDETRPoseNano(
        pretrain_weights=<CHECKPOINT_PATH>,
        num_keypoints=2,  # Match your training config
    )

    detections = model.predict(<IMAGE_PATH>)
    # Access keypoints
    keypoints = detections.data.get("keypoints")  # [N, K, 3] where K=num_keypoints
    ```

## ONNX export

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

To export your model, first install the `onnxexport` extension:

```
pip install rfdetr[onnxexport]
```

Then, run:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRBase

    model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=<CHECKPOINT_PATH>)

    model.export()
    ```

=== "Pose Estimation"

    ```python
    from rfdetr import RFDETRPoseNano  # Use the same size as training

    model = RFDETRPoseNano(
        pretrain_weights=<CHECKPOINT_PATH>,
        num_keypoints=2,  # Match your training config
    )

    model.export()
    ```

This command saves the ONNX model to the `output` directory.