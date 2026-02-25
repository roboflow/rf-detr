# Augmentations

RF-DETR supports custom data augmentations via [Albumentations](https://albumentations.ai/), with automatic bounding box and mask handling for geometric transforms.

## Quick Start

Pass `aug_config` to your training call. Import one of the built-in presets:

```python
from rfdetr import RFDETRSmall
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model = RFDETRSmall()
model.train(dataset_dir="path/to/dataset", epochs=100, aug_config=AUG_CONSERVATIVE)
```

Or pass a custom dict directly — keys are Albumentations transform names:

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

To disable augmentations: `aug_config={}`. Omitting it uses the default (horizontal flip at 50%).

## Built-in Presets

| Preset             | Best for                          |
| ------------------ | --------------------------------- |
| `AUG_CONSERVATIVE` | Small datasets (under 500 images) |
| `AUG_AGGRESSIVE`   | Large datasets (2000+ images)     |
| `AUG_AERIAL`       | Satellite / overhead imagery      |
| `AUG_INDUSTRIAL`   | Manufacturing / inspection data   |

All presets are plain dicts — inspect or extend them before passing:

```python
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

my_config = {**AUG_AGGRESSIVE, "VerticalFlip": {"p": 0.1}}
model.train(dataset_dir="...", aug_config=my_config)
```

### Recommendations by Dataset Size

| Dataset Size     | Recommended preset                                              |
| ---------------- | --------------------------------------------------------------- |
| Under 500 images | `AUG_CONSERVATIVE` — flip + mild brightness/contrast            |
| 500–2000 images  | Default or `AUG_CONSERVATIVE` with a few extra transforms added |
| 2000+ images     | `AUG_AGGRESSIVE` — rotations, affine, color jitter              |

## Geometric vs. Pixel-Level Transforms

RF-DETR automatically handles bounding boxes for **geometric transforms** (flips, rotations, crops, affine, perspective). **Pixel-level transforms** (blur, noise, color) preserve coordinates unchanged. You don't need to handle this distinction — it's automatic based on the transform name.

## Best Practices

!!! tip "Start Conservative"

    Begin with simple augmentations (horizontal flip, small brightness changes) and gradually add more as needed.

!!! warning "Geometric Transforms"

    Be careful with aggressive rotations and crops on datasets where object orientation matters (e.g., text detection, oriented objects).

- **CPU-bound:** Augmentations run on CPU during data loading — more transforms means slower loading
- **Use `num_workers`:** Parallelize augmentation across data loader workers
- **Monitor training mAP vs validation mAP:** With strong augmentations it's normal for training mAP to be lower — validation uses original images while training uses augmented (harder) ones

## Troubleshooting

**Training is slow** — reduce the number of transforms or increase `num_workers`.

**Boxes disappear after augmentation** — aggressive rotations or crops can push boxes outside the image boundary. Reduce rotation angles or avoid large crops.

**Model not improving** — augmentations may be too aggressive. Start with `AUG_CONSERVATIVE` and add transforms gradually. Try removing geometric transforms first to isolate the cause.

**Validation mAP is much higher than training mAP** — this is expected with strong augmentations and not a bug. See the monitoring tip above.

## Advanced: Custom Transforms

Any Albumentations transform works by name. If your custom transform is geometric, register it in `rfdetr/datasets/transforms.py` so boxes are updated automatically:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",
}
```

Then use it like any other transform:

```python
model.train(
    dataset_dir="...",
    aug_config={
        "HorizontalFlip": {"p": 0.5},
        "YourCustomTransform": {"param": 1, "p": 0.3},
    },
)
```

## Reference

- [Albumentations docs](https://albumentations.ai/docs/)
- [All available transforms](https://albumentations.ai/docs/api_reference/augmentations/)

## Next Steps

- [Monitor training with TensorBoard](loggers.md#tensorboard)
- [Use early stopping](advanced.md#early-stopping) to prevent overfitting
- [Export your trained model](../export.md) for deployment
