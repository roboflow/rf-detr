---
description: Configure RF-DETR data augmentations. Defaults use torchvision; custom presets use optional Albumentations.
---

# Augmentations

RF-DETR uses torchvision-native default augmentations for training, validation, prediction, and export preprocessing. Omitting `aug_config` uses the default training augmentation stack: resize/crop scale jitter plus horizontal flip at 50%. Passing `aug_config={}` disables the horizontal flip while keeping required resizing and normalization.

RF-DETR also supports advanced custom data augmentations via optional [Albumentations](https://albumentations.ai/), with automatic bounding box and mask handling for geometric transforms. Albumentations 1.4.24+ and 2.x are supported.

## Augmentation Backend Values

`augmentation_backend` (on `model.train()`/`TrainConfig`) selects which pipeline runs the augmentation stack:

| Value              | Behavior                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `"cpu"` (default)  | Auto-picks the best *installed* CPU backend: Albumentations > Kornia (CPU) > torchvision.                                 |
| `"auto"`           | Same priority as `"cpu"`, but prefers GPU-side Kornia first when CUDA is available.                                       |
| `"torchvision"`    | Forces the torchvision-native default pipeline, never auto-upgraded — use to pin behavior regardless of what's installed. |
| `"albumentations"` | Forces the CPU Albumentations pipeline; requires `rfdetr[augment]`.                                                       |
| `"kornia"`         | Forces the GPU Kornia pipeline; requires `rfdetr[augment]`.                                                               |

`"cpu"`/`"auto"` resolution is late (checked at dataset-build time, not when `TrainConfig` is constructed), so a saved config stays portable across environments with different optional packages installed. The legacy strings `"tv"`, `"albu"`, and `"gpu"` are still accepted as aliases for `"torchvision"`, `"albumentations"`, and `"kornia"` respectively.

## Quick Start

Install the optional augmentation extra before using custom `aug_config` dictionaries or built-in Albumentations presets:

```bash
pip install "rfdetr[train,augment]"
```

Then pass `aug_config` to your training call. Import one of the built-in presets:

```python
from rfdetr import RFDETRSmall
from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

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

To disable all optional training augmentation including the torchvision default horizontal flip: `aug_config={}`.

`aug_config` controls only the augmentation stack — the torchvision-native default, or the Albumentations/Kornia stack when a non-empty `aug_config` is passed. To also disable the independent resize → crop → resize branch (Option B) in the training resize pipeline — so no spatial clipping occurs and annotations near image borders stay intact — pass `scale_jitter=False` to `model.train()`.

## Built-in Presets

| Preset             | Best for                          |
| ------------------ | --------------------------------- |
| `AUG_CONSERVATIVE` | Small datasets (under 500 images) |
| `AUG_AGGRESSIVE`   | Large datasets (2000+ images)     |
| `AUG_AERIAL`       | Satellite / overhead imagery      |
| `AUG_INDUSTRIAL`   | Manufacturing / inspection data   |

All presets are Albumentations config dicts and require `rfdetr[augment]`. They are plain dicts, so you can inspect or extend them before passing:

```python
from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE

my_config = {**AUG_AGGRESSIVE, "VerticalFlip": {"p": 0.1}}
model.train(dataset_dir="...", aug_config=my_config)
```

### Recommendations by Dataset Size

| Dataset Size     | Recommended preset                                              |
| ---------------- | --------------------------------------------------------------- |
| Under 500 images | `AUG_CONSERVATIVE` — flip + mild brightness/contrast            |
| 500–2000 images  | Default or `AUG_CONSERVATIVE` with a few extra transforms added |
| 2000+ images     | `AUG_AGGRESSIVE` — rotations, affine, color jitter              |

## Nested Transforms

RF-DETR supports `OneOf`, `SomeOf`, and `Sequential` container transforms from Albumentations. The most common pattern is `OneOf`, which randomly picks one transform from a group:

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

Each child's `p` controls its relative selection weight. The container itself always fires.

If you need the same transform twice, or want explicit ordering, pass a list instead of a dict:

```python
aug_config = [
    {"HorizontalFlip": {"p": 0.5}},
    {"Rotate": {"limit": 45, "p": 0.3}},
    {"Rotate": {"limit": 5, "p": 0.5}},  # second Rotate — only possible with list format
]
```

Bounding boxes are updated automatically when a container holds any geometric transform — no extra configuration needed.

## Geometric vs. Pixel-Level Transforms

RF-DETR automatically handles bounding boxes for **geometric transforms** (flips, rotations, crops, affine, perspective). **Pixel-level transforms** (blur, noise, color) preserve coordinates unchanged. You don't need to handle this distinction — it's automatic based on the transform name.

## Best Practices

!!! tip "Start Conservative"

    Begin with simple augmentations (horizontal flip, small brightness changes) and gradually add more as needed.

!!! warning "Geometric Transforms"

    Be careful with aggressive rotations and crops on datasets where object orientation matters (e.g., text detection, oriented objects).

- **Default path:** Uses torchvision-native transforms and does not require Albumentations.
- **Custom CPU path:** Non-empty `aug_config` dictionaries use Albumentations and require `rfdetr[augment]`.
- **GPU path:** `augmentation_backend="kornia"` uses Kornia and requires `rfdetr[augment]`.
- **CPU-bound custom configs:** More transforms means slower data loading
- **Use `num_workers`:** Parallelize augmentation across data loader workers
- **Monitor training mAP vs validation mAP:** With strong augmentations it's normal for training mAP to be lower — validation uses original images while training uses augmented (harder) ones

## Troubleshooting

**Training is slow** — reduce the number of transforms or increase `num_workers`.

**Boxes disappear after augmentation** — aggressive rotations or crops can push boxes outside the image boundary. Reduce rotation angles or avoid large crops.

**Model not improving** — augmentations may be too aggressive. Start with `AUG_CONSERVATIVE` and add transforms gradually. Try removing geometric transforms first to isolate the cause.

**Validation mAP is much higher than training mAP** — this is expected with strong augmentations and not a bug. See the monitoring tip above.

**Upgrading albumentations to 2.x with existing `RandomSizedCrop` configs?** RF-DETR automatically adapts `height`/`width` kwargs to the `size=(height, width)` format required by albumentations 2.x. No config changes needed.

## Advanced: Custom Transforms

Any Albumentations transform works by name. If your custom transform is geometric, register it in `rfdetr/datasets/transforms.py` so boxes are updated automatically:

```python
GEOMETRIC_TRANSFORMS = {
    ...,
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
