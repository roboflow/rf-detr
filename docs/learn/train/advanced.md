# Advanced Training

This page covers advanced training topics including resuming training, early stopping, multi-GPU training, and logging with external services.

## Resume Training

You can resume training from a previously saved checkpoint by passing the path to the `checkpoint.pth` file using the `resume` argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model.

The training loop will automatically load:

- Model weights
- Optimizer state
- Learning rate scheduler state
- Training epoch number

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        resume="output/checkpoint.pth",
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        resume="output/checkpoint.pth",
    )
    ```

!!! tip "Resume vs Pretrain Weights"

    - Use `resume="checkpoint.pth"` to continue training with optimizer state
    - Use `pretrain_weights="checkpoint_best_total.pth"` when initializing a model to start fresh training from those weights

---

## Early Stopping

Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This prevents wasted computation once the model has converged.

### Basic Usage

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        early_stopping=True,
    )
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        early_stopping=True,
    )
    ```

### Configuration Options

| Parameter                  | Default | Description                                          |
| -------------------------- | ------- | ---------------------------------------------------- |
| `early_stopping_patience`  | 10      | Number of epochs without improvement before stopping |
| `early_stopping_min_delta` | 0.001   | Minimum mAP change to count as improvement           |
| `early_stopping_use_ema`   | False   | Use EMA model's mAP for comparisons                  |

### Advanced Example

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=200,
    early_stopping=True,
    early_stopping_patience=15,  # Wait 15 epochs before stopping
    early_stopping_min_delta=0.005,  # Require 0.5% mAP improvement
    early_stopping_use_ema=True,  # Track EMA model performance
)
```

### How It Works

1. After each epoch, validation mAP is computed
2. If mAP improves by at least `min_delta`, the patience counter resets
3. If mAP doesn't improve, the patience counter increments
4. When patience counter reaches `patience`, training stops
5. The best checkpoint is already saved as `checkpoint_best_total.pth`

```
Epoch 10: mAP = 0.450 (best: 0.450) - counter: 0
Epoch 11: mAP = 0.455 (best: 0.455) - counter: 0 (improved)
Epoch 12: mAP = 0.454 (best: 0.455) - counter: 1 (no improvement)
Epoch 13: mAP = 0.453 (best: 0.455) - counter: 2
...
Epoch 22: mAP = 0.452 (best: 0.455) - counter: 10 → STOP
```

---

## Multi-GPU Training

You can fine-tune RF-DETR on multiple GPUs using PyTorch's Distributed Data Parallel (DDP). This splits the workload across GPUs for faster training.

### Setup

1. Create a training script (`main.py`):

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=1,
        lr=1e-4,
        output_dir="output",
    )
    ```

2. Run with `torch.distributed.launch`:

    ```bash
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py
    ```

Replace `8` with the number of GPUs you want to use.

### Batch Size with Multiple GPUs

When using multiple GPUs, your effective batch size is multiplied by the number of GPUs:

```
effective_batch_size = batch_size × grad_accum_steps × num_gpus
```

**Example configurations for effective batch size of 16:**

| GPUs | `batch_size` | `grad_accum_steps` | Effective |
| ---- | ------------ | ------------------ | --------- |
| 1    | 4            | 4                  | 16        |
| 2    | 4            | 2                  | 16        |
| 4    | 4            | 1                  | 16        |
| 8    | 2            | 1                  | 16        |

!!! warning "Adjust for GPU count"

    When switching between single and multi-GPU training, remember to adjust `batch_size` and `grad_accum_steps` to maintain the same effective batch size.

### Multi-Node Training

For training across multiple machines, use `torchrun`:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=1234 \
    main.py
```

Run this command on each node, changing `--node_rank` accordingly.

---

## Logging with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful toolkit for visualizing and tracking training metrics.

### Setup

1. Install the required packages:

    ```bash
    pip install "rfdetr[metrics]"
    ```

2. Enable TensorBoard logging in your training:

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        tensorboard=True,
    )
    ```

### Viewing Logs

**Local environment:**

```bash
tensorboard --logdir output
```

Then open `http://localhost:6006/` in your browser.

**Google Colab:**

```ipython
%load_ext tensorboard
%tensorboard --logdir output
```

### Logged Metrics

TensorBoard will track:

- Training loss (total and per-component)
- Validation mAP
- Learning rate schedule
- EMA model metrics (when enabled)

---

## Logging with Weights and Biases

[Weights and Biases (W&B)](https://www.wandb.ai) is a cloud-based platform for experiment tracking and visualization.

### Setup

1. Install the required packages:

    ```bash
    pip install "rfdetr[metrics]"
    ```

2. Log in to W&B:

    ```bash
    wandb login
    ```

    You can retrieve your API key at [wandb.ai/authorize](https://wandb.ai/authorize).

3. Enable W&B logging in your training:

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        wandb=True,
        project="my-detection-project",
        run="experiment-001",
    )
    ```

### W&B Organization

| Parameter | Description                             |
| --------- | --------------------------------------- |
| `project` | Groups related experiments together     |
| `run`     | Identifies individual training sessions |

If you don't specify a run name, W&B assigns a random one automatically.

### Viewing Results

Access your runs at [wandb.ai](https://wandb.ai). W&B provides:

- Real-time metric visualization
- Experiment comparison
- Hyperparameter tracking
- System metrics (GPU usage, memory)
- Training config logging

### Using Both TensorBoard and W&B

You can enable both logging systems simultaneously:

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    tensorboard=True,
    wandb=True,
    project="my-project",
    run="experiment-001",
)
```

---

## Logging with ClearML

[ClearML](https://clear.ml) is an open-source platform that helps you manage, track, and automate your machine learning experiments. With ClearML, you can monitor performance, compare experiments, and optimize model training using its comprehensive suite of tools.

### Setup

1. Install the required packages:

    ```bash
    pip install "rfdetr[metrics]"
    ```

2. Initialize ClearML:

    ```bash
    clearml-init
    ```

    Follow the instructions to connect to your ClearML server (hosted or self-hosted).

3. Enable ClearML logging in your training:

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium()

    model.train(
        dataset_dir="path/to/dataset",
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="output",
        clearml=True,
        project="my-detection-project",
        run="experiment-001",
    )
    ```

### ClearML Organization

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `project` | Groups related experiments together                 |
| `run`     | Identifies individual training sessions (task name) |

### Viewing Results

Access your experiments in the ClearML Web UI. ClearML provides:

- Real-time metric visualization
- Experiment comparison
- Hyperparameter tracking
- Artifact storage
- Model versioning

---

## Custom Augmentations with Albumentations

RF-DETR supports custom data augmentations using the [Albumentations](https://albumentations.ai/) library, providing access to over 70 different image transformations optimized for object detection tasks.

### Why Albumentations?

- **Bounding Box Support:** All geometric transforms automatically update bounding box coordinates
- **Performance:** Highly optimized, faster than torchvision transforms
- **Flexibility:** Mix and match over 70 different augmentations
- **Battle-Tested:** Used in winning solutions of many Kaggle competitions

### Setup

Albumentations is installed automatically with RF-DETR:

```bash
pip install rfdetr
```

### Basic Usage

Augmentations are configured via the `AUG_CONFIG` dictionary in `src/rfdetr/augmentation_config.py`:

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},
}
```

Simply enable the augmentations you want by uncommenting them or adding new ones. The probability `p` controls how often each transform is applied.

### Available Augmentations

**Geometric Transforms** (automatically update bounding boxes):

- `HorizontalFlip` - Flip image horizontally
- `VerticalFlip` - Flip image vertically
- `Rotate` - Rotate image by random angle
- `Affine` - Apply affine transformations (scale, translate, rotate, shear)
- `RandomCrop` - Crop random region
- `ShiftScaleRotate` - Combination of shifting, scaling, and rotating

**Pixel-Level Transforms** (preserve bounding boxes):

- `ColorJitter` - Randomly change brightness, contrast, saturation
- `GaussianBlur` - Apply Gaussian blur
- `GaussNoise` - Add Gaussian noise
- `CLAHE` - Contrast Limited Adaptive Histogram Equalization
- `RandomBrightnessContrast` - Adjust brightness and contrast

### Configuration Examples

**Conservative augmentations (recommended for small datasets):**

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {"brightness_limit": 0.1, "contrast_limit": 0.1, "p": 0.3},
}
```

**Aggressive augmentations (for larger datasets):**

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {"scale": (0.8, 1.2), "translate_percent": (0.1, 0.1), "rotate": (-15, 15), "shear": (-5, 5), "p": 0.5},
    "ColorJitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1, "p": 0.5},
}
```

**Aerial imagery / satellite datasets:**

```python
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},  # Important for overhead views
    "Rotate": {"limit": (90, 90), "p": 0.5},  # 90° rotations common
    "RandomBrightnessContrast": {"brightness_limit": 0.15, "contrast_limit": 0.15, "p": 0.4},
}
```

### How It Works

Augmentations are automatically applied during training:

1. The `AUG_CONFIG` is read when building the dataset
2. Transforms are composed into a pipeline
3. Each training sample is augmented on-the-fly
4. Bounding boxes are automatically transformed for geometric augmentations

No code changes needed in your training script - just modify `augmentation_config.py`.

### Programmatic Configuration

You can also build augmentations programmatically:

```python
from rfdetr.datasets.transforms import build_albumentations_from_config, ComposeAugmentations

# Custom config
custom_config = {
    "HorizontalFlip": {"p": 0.7},
    "Blur": {"blur_limit": 3, "p": 0.2},
}

# Build and compose transforms
transforms = build_albumentations_from_config(custom_config)
augmentation_pipeline = ComposeAugmentations(transforms)

# Apply to image and target
augmented_image, augmented_target = augmentation_pipeline(image, target)
```

### Best Practices

> [!TIP]
> **Start Conservative:** Begin with simple augmentations (horizontal flip, small brightness changes) and gradually add more as needed.

> [!WARNING]
> **Geometric Transforms:** Be careful with aggressive rotations and crops on datasets where object orientation matters (e.g., text detection, oriented objects).

**Recommendations by dataset size:**

| Dataset Size | Recommended Augmentations                                        |
| ------------ | ---------------------------------------------------------------- |
| < 500 images | Horizontal flip, small brightness/contrast adjustments           |
| 500-2000     | Add vertical flip (if applicable), color jitter, blur            |
| 2000+        | Add rotations, affine transforms, aggressive color augmentations |

**Performance Tips:**

- Augmentations run on CPU during data loading
- More augmentations = slower data loading (but better model)
- Use `num_workers` in data loader to parallelize augmentations
- Monitor GPU utilization - if it's not saturated, you can add more augmentations

### Troubleshooting

**Problem:** Training is very slow

- Reduce number of augmentations
- Reduce augmentation complexity (e.g., smaller rotation angles)
- Increase `num_workers` in data loader

**Problem:** Validation mAP is much higher than training mAP

- This is expected with strong augmentations
- Validation uses original images without augmentation
- Training mAP is artificially lower due to augmented data

**Problem:** Some boxes disappear after augmentation

- This is normal for aggressive transforms (e.g., large rotations, crops)
- Albumentations removes boxes that fall outside image boundaries
- Reduce `min_visibility` in `AlbumentationsWrapper` if needed (advanced)

### Reference

For complete list of available transforms and parameters, see the [Albumentations documentation](https://albumentations.ai/docs/api_reference/augmentations/).

---

## Memory Optimization

### Gradient Checkpointing

For large models or high resolutions, enable gradient checkpointing to trade compute for memory:

```python
model.train(
    dataset_dir="path/to/dataset",
    gradient_checkpointing=True,
    batch_size=2,  # May be able to increase with checkpointing
)
```

This re-computes activations during the backward pass instead of storing them, reducing memory usage by ~30-40% at the cost of ~20% slower training.

### Memory-Efficient Configurations

| Memory Level      | Configuration                                                                          |
| ----------------- | -------------------------------------------------------------------------------------- |
| Very Low (8GB)    | `batch_size=1`, `grad_accum_steps=16`, `gradient_checkpointing=True`, `resolution=560` |
| Low (12GB)        | `batch_size=2`, `grad_accum_steps=8`, `gradient_checkpointing=True`                    |
| Medium (16GB)     | `batch_size=4`, `grad_accum_steps=4`                                                   |
| High (24GB)       | `batch_size=8`, `grad_accum_steps=2`                                                   |
| Very High (40GB+) | `batch_size=16`, `grad_accum_steps=1`, `resolution=784`                                |

---

## Training Tips

### Learning Rate Tuning

- **Fine-tuning from COCO weights (default):** Use default learning rates (`lr=1e-4`, `lr_encoder=1.5e-4`)
- **Small dataset (\<1000 images):** Consider lower `lr` (e.g., `5e-5`) to prevent overfitting
- **Large dataset (>10000 images):** May benefit from higher `lr` (e.g., `2e-4`)

### Epoch Count

| Dataset Size      | Recommended Epochs |
| ----------------- | ------------------ |
| < 500 images      | 100-200            |
| 500-2000 images   | 50-100             |
| 2000-10000 images | 30-50              |
| > 10000 images    | 20-30              |

Use early stopping to automatically determine the optimal stopping point.

### Data Augmentation

RF-DETR applies built-in augmentations during training:

- Random resizing
- Random cropping
- Color jittering
- Horizontal flipping

These are automatically configured and don't require manual setup.

---

## Troubleshooting

### Out of Memory (OOM)

If you encounter CUDA out of memory errors:

1. Reduce `batch_size`
2. Enable `gradient_checkpointing=True`
3. Reduce `resolution`
4. Increase `grad_accum_steps` to maintain effective batch size

### Training Too Slow

1. Increase `batch_size` (if memory allows)
2. Use multiple GPUs with DDP
3. Ensure you're using GPU (check `device="cuda"`)
4. Consider using a smaller model (e.g., `RFDETRSmall` instead of `RFDETRLarge`)

### Loss Not Decreasing

1. Check that your dataset is correctly formatted
2. Verify annotations are correct (bounding boxes in correct format)
3. Try reducing the learning rate
4. Check for class imbalance in your dataset
