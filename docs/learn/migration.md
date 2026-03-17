# Deprecated APIs

RF-DETR has reorganised its internal package layout. A set of backward-compatibility
shim modules is provided so that existing code continues to work, but each shim emits
a `DeprecationWarning` on import. Update your imports before the shims are removed in
a future major release.

## Module renames

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

## Migration examples

### `rfdetr.util.coco_classes`

```python
# Before (deprecated)
from rfdetr.util.coco_classes import COCO_CLASSES

# After
from rfdetr.assets.coco_classes import COCO_CLASSES
```

### `rfdetr.util.misc`

```python
# Before (deprecated)
from rfdetr.util.misc import get_rank, get_world_size, is_main_process, save_on_master

# After
from rfdetr.utilities.distributed import get_rank, get_world_size, is_main_process, save_on_master
```

### `rfdetr.util.logger`

```python
# Before (deprecated)
from rfdetr.util.logger import get_logger

# After
from rfdetr.utilities.logger import get_logger
```

### `rfdetr.util.box_ops`

```python
# Before (deprecated)
from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

# After
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
```

### `rfdetr.util.get_param_dicts`

```python
# Before (deprecated)
from rfdetr.util.get_param_dicts import get_param_dict

# After
from rfdetr.training.param_groups import get_param_dict
```

### `rfdetr.util.drop_scheduler`

```python
# Before (deprecated)
from rfdetr.util.drop_scheduler import drop_scheduler

# After
from rfdetr.training.drop_schedule import drop_scheduler
```

### `rfdetr.util.visualize`

```python
# Before (deprecated)
from rfdetr.util.visualize import save_gt_predictions_visualization

# After
from rfdetr.visualize.data import save_gt_predictions_visualization
```

### `rfdetr.deploy`

```python
# Before (deprecated)
from rfdetr.deploy import export_onnx

# After
from rfdetr.export import export_onnx
```

### `rfdetr.models.segmentation_head`

```python
# Before (deprecated)
from rfdetr.models.segmentation_head import SegmentationHead

# After
from rfdetr.models.heads.segmentation import SegmentationHead
```
