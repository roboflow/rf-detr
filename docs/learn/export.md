# Export RF-DETR Model to ONNX

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

## Installation

To export your model, first install the `onnxexport` extension:

```bash
pip install "rfdetr[onnx]"
```

## Basic Export

Export your trained model to ONNX format:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRMedium

    model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

This command saves the ONNX model to the `output` directory by default.

## Export Parameters

The `export()` method accepts several parameters to customize the export process:

| Parameter       | Default    | Description                                                                                                            |
| --------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------- |
| `output_dir`    | `"output"` | Directory where the exported ONNX model will be saved.                                                                 |
| `infer_dir`     | `None`     | Path to an image file to use for tracing. If not provided, a random dummy image is generated.                          |
| `simplify`      | `False`    | Whether to simplify the ONNX model using onnxsim for better compatibility and performance.                             |
| `backbone_only` | `False`    | Export only the backbone feature extractor instead of the full model.                                                  |
| `opset_version` | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                         |
| `verbose`       | `True`     | Whether to print verbose export information.                                                                           |
| `force`         | `False`    | Force re-export even if simplified model already exists.                                                               |
| `shape`         | `None`     | Input shape as tuple `(height, width)`. Must be divisible by 14. If not provided, uses the model's default resolution. |
| `batch_size`    | `1`        | Batch size for the exported model.                                                                                     |
| `tensorrt`      | `False`    | When `True`, convert the ONNX model to a TensorRT `.engine` file. Requires TensorRT (`trtexec`) to be installed.       |

## Advanced Export Examples

### Export with Custom Output Directory

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(output_dir="exports/my_model")
```

### Export with Simplification

Simplifying the ONNX model can improve inference performance and compatibility with various runtimes:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(simplify=True)
```

### Export with Custom Resolution

Export the model with a specific input resolution (must be divisible by 14):

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(shape=(560, 560))
```

### Export Backbone Only

Export only the backbone feature extractor for use in custom pipelines:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(backbone_only=True)
```

## Output Files

After running the export, you will find the following files in your output directory:

- `inference_model.onnx` - The exported ONNX model (or `backbone_model.onnx` if `backbone_only=True`)
- `inference_model.sim.onnx` - The simplified ONNX model (if `simplify=True`)

## Optional: Convert ONNX to TensorRT

If you want lower latency on NVIDIA GPUs, you can convert the exported ONNX model to a TensorRT engine.

> [!IMPORTANT]
> Run TensorRT conversion on the same machine and GPU family where you plan to deploy inference.

### Prerequisites

- Install TensorRT (`trtexec` must be available in your `PATH`)
- Install the `trt` extras: `pip install "rfdetr[trt]"`
- Export an ONNX model first (for example: `output/inference_model.onnx`)

### Export Directly to TensorRT

Pass `tensorrt=True` to `export()` to export ONNX and convert to a TensorRT engine in one step:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

model.export(tensorrt=True)
```

This exports `output/inference_model.onnx` first and then produces `output/inference_model.engine`.

### Manual Python API Conversion

You can also convert an existing ONNX file to a TensorRT engine separately:

```python
from argparse import Namespace

from rfdetr.export.tensorrt import trtexec

args = Namespace(
    verbose=True,
    profile=False,
    dry_run=False,
)

engine_path = trtexec("output/inference_model.onnx", args)
```

`trtexec` returns the path to the generated `.engine` file. If `profile=True`, it also writes an Nsight Systems report (`.nsys-rep`).

## Run Inference with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is the
recommended library for running RF-DETR inference. It supports multiple backends — PyTorch,
ONNX, and TensorRT — with automatic backend selection and a unified API.

### Installation

```bash
# CPU / PyTorch only
pip install inference-models

# With TensorRT support (NVIDIA GPU required)
pip install "inference-models[trt-cu12]"  # CUDA 12.x
```

See the [inference-models installation guide](https://inference-models.roboflow.com/getting-started/installation/)
for all installation options including Jetson and CUDA 11.x.

### Load a Pre-trained RF-DETR Model

```python
import cv2
from inference_models import AutoModel

# Automatically selects the best available backend for your environment
model = AutoModel.from_pretrained("rfdetr-base")

image = cv2.imread("image.jpg")
predictions = model(image)

# Convert to supervision Detections
detections = predictions[0].to_supervision()
print(detections)
```

### Load a Local RF-DETR Checkpoint

```python
import cv2
from inference_models import AutoModel

# Load from a local .pth checkpoint (same file used by rfdetr for training)
model = AutoModel.from_pretrained(
    "/path/to/checkpoint.pth",
    model_type="rfdetr-base",  # specify the architecture variant
)

image = cv2.imread("image.jpg")
predictions = model(image)
```

### Force TensorRT Backend

```python
import cv2
from inference_models import AutoModel, BackendType

# Explicitly request TensorRT — requires TRT to be installed
model = AutoModel.from_pretrained("rfdetr-base", backend=BackendType.TRT)

image = cv2.imread("image.jpg")
predictions = model(image)
```

`AutoModel.from_pretrained` accepts `backend="onnx"`, `backend="torch"`, or
`backend="trt"` to override automatic backend selection.

## Using the Exported ONNX Model

Once exported, you can also use the ONNX model directly with ONNX Runtime:

### ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load the ONNX model
session = ort.InferenceSession("output/inference_model.onnx")

# Prepare input image
image = Image.open("image.jpg").convert("RGB")
image = image.resize((560, 560))  # Resize to model's input resolution
image_array = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Convert to NCHW format
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
outputs = session.run(None, {"input": image_array})
boxes, labels = outputs
```

### TensorRT Engine

Load a `.engine` file and run inference using the `TRTInference` helper:

```python
import torch
import numpy as np
from PIL import Image

from rfdetr.export.benchmark import TRTInference

# Load the TensorRT engine
trt_model = TRTInference("output/inference_model.engine", device="cuda:0")

# Prepare input image (same preprocessing as ONNX Runtime)
resolution = 560  # must match the resolution used during export
image = Image.open("image.jpg").convert("RGB").resize((resolution, resolution))
image_array = np.array(image).astype(np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Convert to NCHW tensor on GPU
input_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0).cuda()

# Run inference — returns {"dets": ..., "labels": ...}
outputs = trt_model({"input": input_tensor})
boxes = outputs["dets"]  # shape: [1, num_queries, 4]  (cx, cy, w, h normalised)
labels = outputs["labels"]  # shape: [1, num_queries, num_classes]
```

> [!NOTE]
> `TRTInference` requires TensorRT and (for async mode) `pycuda` to be installed.
> Run inference on the same GPU family as the one used to build the engine.

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](deploy.md) for cloud-based inference and workflow integration
- Use [`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) for
    multi-backend inference (PyTorch, ONNX, TensorRT) with automatic backend selection
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
