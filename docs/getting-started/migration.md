---
description: Per-version migration guide for RF-DETR. Covers breaking changes and deprecated APIs for each release series.
---

# Migration Guide

Read each section between your current version and your target — every section covers
only the delta between two adjacent releases.

```
1.4.x  →  1.5 →  1.6  →  1.7  →  1.8
```

You can apply all changes in one go; working through sections one release at a time
and verifying between each step is optional but makes failures easier to isolate.
Deprecated APIs emit a `DeprecationWarning` until the version marked for removal.
See the [Changelog](../changelog.md) for the full list of changes in each release.

---

## Upgrade 1.8 → 1.9

### Breaking changes

!!! warning "Breaking: default keypoint schema changed to active-first `[17]`"

    New checkpoints use `class_id=0` for person. Legacy `[0, 17]` checkpoints are still supported —
    RF-DETR auto-detects the schema from the checkpoint at load time.

    If your post-processing code offsets class IDs by 1 (common for background-first models), update it:

    ```python
    # Before (background-first [0, 17]: person was at class_id=1)
    class_name = "person" if detection.class_id == 1 else "other"

    # After (active-first [17]: person is at class_id=0)
    class_name = "person" if detection.class_id == 0 else "other"
    ```

    Use `detection.data["class_name"]` for schema-agnostic name resolution.

    !!! note "Preserving a legacy schema for fine-tuning"

        Pass `num_keypoints_per_class` **at construction time** to keep the old schema:

        ```python
        config = RFDETRKeypointPreviewConfig(num_keypoints_per_class=[0, 17])
        ```

        Assigning the field after construction (`config.num_keypoints_per_class = [0, 17]`)
        does **not** register the override and RF-DETR may still auto-align the schema from the
        checkpoint. Always pass it to the constructor.

---

## Upgrade 1.7 → 1.8

### Breaking changes

!!! warning "Breaking: `rfdetr.datasets.aug_config` renamed to `rfdetr.datasets.aug_configs`"

    The augmentation config module was renamed (singular → plural). If you import from it directly:

    ```python
    # Before
    from rfdetr.datasets.aug_config import AUG_AGGRESSIVE

    # After
    from rfdetr.datasets.aug_configs import AUG_AGGRESSIVE
    ```

    All preset constants (`AUG_AGGRESSIVE`, `AUG_CONSERVATIVE`, etc.) are unchanged.

!!! warning "Breaking: `supervision>=0.29.0` now required"

    Required for `sv.KeyPoints` support. `pip install rfdetr==1.8.0` pulls this automatically.
    If another dependency pins `supervision<0.29.0`, resolve the conflict manually.

!!! warning "Breaking: `pyDeprecate` constraint narrowed to `>=0.9,<0.10`"

    Was `>=0.6,<0.8`. If another package pins an older version, resolve with:

    ```bash
    pip install "rfdetr==1.8.0" "pyDeprecate>=0.9,<0.10"
    ```

---

## Upgrade 1.6 → 1.7

### Breaking changes

!!! warning "Breaking: `peft` removed from the default install"

    LoRA fine-tuning now requires the `lora` extra. If you use LoRA adapters during
    training, update your install command.

    ```bash
    # Before
    pip install rfdetr

    # After
    pip install 'rfdetr[lora]'
    ```

!!! warning "Breaking: `predict()` stores source image in `detections.metadata`"

    **`predict()` stores the source image in `detections.metadata`, not `detections.data`.**

    ```python
    # Before (1.6.4 and earlier)
    source = detections.data["source_image"]

    # After
    source = detections.metadata["source_image"]
    ```

!!! warning "Breaking: `pyDeprecate` constraint changed to `>=0.6,<0.8`"

    Was `>=0.3,<0.6`. If another package pins an older version, resolve with:

    ```bash
    pip install "rfdetr==1.7.0" "pyDeprecate>=0.6,<0.8"
    ```

### Deprecated in v1.7 → Remove in v1.9

!!! note "Deprecated: `build_namespace()` split into two functions"

    **`build_namespace(model_config, train_config)`** — use `build_model_from_config` or
    `build_criterion_from_config` instead.

    ```python
    # Before (deprecated)
    from rfdetr.models import build_namespace

    ns = build_namespace(model_config, train_config)

    # After
    from rfdetr.models import build_model_from_config, build_criterion_from_config

    model = build_model_from_config(model_config)
    criterion = build_criterion_from_config(model_config, train_config)
    ```

!!! note "Deprecated: `load_pretrain_weights()` no longer takes `train_config`"

    **`load_pretrain_weights(nn_model, model_config, train_config)`** — drop the
    `train_config` positional argument.

    ```python
    # Before (deprecated)
    from rfdetr.models import load_pretrain_weights

    load_pretrain_weights(nn_model, model_config, train_config)

    # After
    from rfdetr.models import load_pretrain_weights

    load_pretrain_weights(nn_model, model_config)
    ```

!!! note "Deprecated: config fields moved between `ModelConfig` and `TrainConfig`"

    **Config fields placed in the wrong config object.** Move them as shown:

    | Field               | Was in        | Move to       |
    | ------------------- | ------------- | ------------- |
    | `group_detr`        | `TrainConfig` | `ModelConfig` |
    | `ia_bce_loss`       | `TrainConfig` | `ModelConfig` |
    | `segmentation_head` | `TrainConfig` | `ModelConfig` |
    | `num_select`        | `TrainConfig` | `ModelConfig` |
    | `cls_loss_coef`     | `ModelConfig` | `TrainConfig` |

    ```python
    # Before (deprecated)
    train_config = TrainConfig(group_detr=13, cls_loss_coef=2.0)

    # After
    model_config = ModelConfig(group_detr=13)
    train_config = TrainConfig(cls_loss_coef=2.0)
    ```

### Deprecated in v1.7 → Remove in v2.0

!!! note "Deprecated: `RFDETRBase` replaced by size-specific classes"

    **`RFDETRBase`** defaulted to the small variant and is replaced by size-specific
    classes. Choose the variant that matches your previous model size. If you used
    `RFDETRBase()` without arguments, switch to `RFDETRSmall()`.

    ```python
    # Before (deprecated)
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    # After — pick one
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

    model = RFDETRSmall()
    ```

!!! note "Deprecated: `RFDETRSegPreview` replaced by size-specific segmentation classes"

````
**`RFDETRSegPreview`** defaulted to the small variant and is replaced by size-specific
segmentation classes. If you used `RFDETRSegPreview()` without arguments, switch to
`RFDETRSegSmall()`.

```python
# Before (deprecated)
from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview()

# After — pick one
from rfdetr import RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge

model = RFDETRSegSmall()
```
````

---

## Upgrade 1.5 → 1.6

### Breaking changes

!!! warning "Breaking: `transformers` minimum version raised to `>=5.1.0`"

    **`transformers` minimum version raised to `>=5.1.0,<6.0.0`.**

    Projects pinned to `transformers<5.0.0` must upgrade. If upgrading is not possible,
    pin `rfdetr<1.6.0`.

    ```bash
    pip install 'transformers>=5.1.0,<6.0.0'
    ```

!!! warning "Breaking: PyPI extras renamed"

    **PyPI extras renamed.**

    Update your `pip install` commands and `requirements*.txt` files.

    | Old extra            | New extra         |
    | -------------------- | ----------------- |
    | `rfdetr[metrics]`    | `rfdetr[loggers]` |
    | `rfdetr[onnxexport]` | `rfdetr[onnx]`    |

    ```bash
    # Before
    pip install 'rfdetr[metrics]'
    pip install 'rfdetr[onnxexport]'

    # After
    pip install 'rfdetr[loggers]'
    pip install 'rfdetr[onnx]'
    ```

!!! warning "Breaking: `draw_synthetic_shape()` now returns a tuple"

    **`draw_synthetic_shape()` now returns `(image, polygon)` instead of `image`.**

    Update every call site that unpacks only the image.

    ```python
    # Before
    img = draw_synthetic_shape(...)

    # After
    img, polygon = draw_synthetic_shape(...)
    ```

### Deprecated in v1.6 → Removed in v1.8

!!! note "Deprecated: `simplify` and `force` arguments in `RFDETR.export()`"

    **`RFDETR.export(..., simplify=..., force=...)`** — both arguments are no-ops.
    Remove them from your calls.

    ```python
    # Before (deprecated)
    model.export("model.onnx", simplify=True, force=True)

    # After
    model.export("model.onnx")
    ```

### Deprecated in v1.6 → Remove in v1.9

!!! note "Deprecated: `rfdetr.util.*` and `rfdetr.deploy.*` import paths"

    Backward-compatibility shims are still active but emit `DeprecationWarning` on import.
    Replace with the canonical paths listed in the table below.

    | Deprecated module                 | Canonical replacement              |
    | --------------------------------- | ---------------------------------- |
    | `rfdetr.util.coco_classes`        | `rfdetr.assets.coco_classes`       |
    | `rfdetr.util.misc`                | `rfdetr.utilities`                 |
    | `rfdetr.util.logger`              | `rfdetr.utilities.logger`          |
    | `rfdetr.util.box_ops`             | `rfdetr.utilities.box_ops`         |
    | `rfdetr.util.files`               | `rfdetr.utilities.files`           |
    | `rfdetr.util.package`             | `rfdetr.utilities.package`         |
    | `rfdetr.util.get_param_dicts`     | `rfdetr.training.param_groups`     |
    | `rfdetr.util.drop_scheduler`      | `rfdetr.training.drop_schedule`    |
    | `rfdetr.util.visualize`           | `rfdetr.visualize.data`            |
    | `rfdetr.deploy`                   | `rfdetr.export`                    |
    | `rfdetr.models.segmentation_head` | `rfdetr.models.heads.segmentation` |

    **Examples:**

    ```python
    # Before (deprecated)
    from rfdetr.util.coco_classes import COCO_CLASSES
    from rfdetr.util.misc import get_rank, get_world_size, is_main_process, save_on_master
    from rfdetr.util.logger import get_logger
    from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    from rfdetr.util.get_param_dicts import get_param_dict
    from rfdetr.util.drop_scheduler import drop_scheduler
    from rfdetr.util.visualize import save_gt_predictions_visualization
    from rfdetr.deploy import export_onnx
    from rfdetr.models.segmentation_head import SegmentationHead

    # After
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from rfdetr.utilities.distributed import get_rank, get_world_size, is_main_process, save_on_master
    from rfdetr.utilities.logger import get_logger
    from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
    from rfdetr.training.param_groups import get_param_dict
    from rfdetr.training.drop_schedule import drop_scheduler
    from rfdetr.visualize.data import save_gt_predictions_visualization
    from rfdetr.export.main import export_onnx
    from rfdetr.models.heads.segmentation import SegmentationHead
    ```

---

## Upgrade 1.4 → 1.5

### Breaking changes

!!! warning "Breaking: `ModelConfig` rejects unknown keyword arguments"

    **`ModelConfig` now raises `ValidationError` on unknown keyword arguments.**

    Previously, unrecognised fields were silently ignored. Remove or rename any
    unrecognised keys you pass to `ModelConfig(...)`.

    ```python
    # Before — silently accepted
    config = ModelConfig(unknown_field=True)

    # Now raises ValidationError — remove the unknown key
    config = ModelConfig()
    ```

### Deprecated in v1.5 → Removed in v1.7

!!! note "Deprecated: `OPEN_SOURCE_MODELS` replaced by `ModelWeights` enum"

    **`OPEN_SOURCE_MODELS` constant** — use the `ModelWeights` enum instead. A
    `DeprecationWarning` is emitted on access. See the
    [API reference](../reference/rfdetr.md) for available enum values.

    ```python
    # Before (deprecated)
    from rfdetr import OPEN_SOURCE_MODELS

    # After
    from rfdetr import ModelWeights
    ```
