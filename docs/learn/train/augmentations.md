---
description: Configure RF-DETR data augmentations with Kornia. Built-in presets for aerial, industrial, and small datasets plus supported custom transforms.
---

# Augmentations

RF-DETR uses Kornia for training resize and augmentation. Kornia transforms keep images, bounding boxes, and segmentation
masks aligned for COCO and YOLO training data.

## Quick Start

Pass `aug_config` to your training call. Import one of the built-in presets:

```python
from rfdetr import RFDETRSmall
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model = RFDETRSmall()
model.train(dataset_dir="path/to/dataset", epochs=100, aug_config=AUG_CONSERVATIVE)
```

Or pass a custom dict directly using RF-DETR's supported Kornia transform keys:

```python
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "Rotate": {"limit": 15, "p": 0.3},
        "GaussianBlur": {"p": 0.2},
    },
)
```

To disable optional augmentations: `aug_config={}`. Omitting it uses the default horizontal flip at 50%.

## Built-in Presets

| Preset             | Best for                          |
| ------------------ | --------------------------------- |
| `AUG_CONSERVATIVE` | Small datasets (under 500 images) |
| `AUG_AGGRESSIVE`   | Large datasets (2000+ images)     |
| `AUG_AERIAL`       | Satellite / overhead imagery      |
| `AUG_INDUSTRIAL`   | Manufacturing / inspection data   |

All presets are plain dicts:

```python
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

my_config = {**AUG_AGGRESSIVE, "VerticalFlip": {"p": 0.1}}
model.train(dataset_dir="...", aug_config=my_config)
```

## Supported Keys

The public `aug_config` surface supports:

| Key                        | Kornia operation               |
| -------------------------- | ------------------------------ |
| `HorizontalFlip`           | Horizontal flip                |
| `VerticalFlip`             | Vertical flip                  |
| `Rotate`                   | Random rotation                |
| `Affine`                   | Random affine                  |
| `ColorJitter`              | Color jiggle                   |
| `RandomBrightnessContrast` | Brightness and contrast jitter |
| `GaussianBlur`             | Gaussian blur                  |
| `GaussNoise`               | Gaussian noise                 |

RF-DETR also uses these internal resize/container keys for training and evaluation pipelines: `Resize`,
`SmallestMaxSize`, `LongestMaxSize`, `RandomSizedCrop`, `OneOf`, and `Sequential`.

Unsupported keys raise `ValueError` with the supported key list.

## Nested Transforms

`OneOf` and `Sequential` are supported for RF-DETR's Kornia configs:

```python
aug_config = {
    "HorizontalFlip": {"p": 0.5},
    "OneOf": {
        "transforms": [
            {"Rotate": {"limit": 45, "p": 1.0}},
            {"Affine": {"scale": (0.8, 1.2), "p": 1.0}},
        ],
    },
    "GaussianBlur": {"p": 0.2},
}
```

`OneOf` samples one child uniformly. `Sequential` runs children in order.

## GPU Augmentation

By default, Kornia runs during dataset loading on CPU. Set `augmentation_backend="auto"` or `"gpu"` to run optional
batch augmentations and normalization after transfer to CUDA. Required resize still runs before collation.

## Best Practices

!!! tip "Start Conservative"

    Begin with simple augmentations such as horizontal flip and mild brightness changes, then add stronger transforms as
    validation results justify them.

!!! warning "Geometric Transforms"

    Be careful with aggressive rotations and crops on datasets where object orientation matters.

- **CPU-bound:** More transforms can slow data loading.
- **Use `num_workers`:** Parallelize augmentation across data loader workers.
- **Monitor training mAP vs validation mAP:** With strong augmentations, training mAP can be lower because training
    images are harder than validation images.

## Troubleshooting

**Training is slow** - reduce the number of transforms, increase `num_workers`, or use GPU augmentation when available.

**Boxes disappear after augmentation** - aggressive rotations or crops can push boxes outside the image boundary. Reduce
rotation angles or avoid large crops.

**Model not improving** - augmentations may be too aggressive. Start with `AUG_CONSERVATIVE` and add transforms
gradually.

## Next Steps

- [Monitor training with TensorBoard](loggers.md#tensorboard)
- [Use early stopping](advanced.md#early-stopping) to prevent overfitting
- [Export your trained model](../export.md) for deployment
