# Export RF-DETR Model to ONNX

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency.

## Installation

To export your model, first install the `onnxexport` extension:

```bash
pip install "rfdetr[onnxexport]"
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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `"output"` | Directory where the exported ONNX model will be saved. |
| `infer_dir` | `None` | Path to an image file to use for tracing. If not provided, a random dummy image is generated. |
| `simplify` | `False` | Whether to simplify the ONNX model using onnxsim for better compatibility and performance. |
| `backbone_only` | `False` | Export only the backbone feature extractor instead of the full model. |
| `opset_version` | `17` | ONNX opset version to use for export. Higher versions support more operations. |
| `verbose` | `True` | Whether to print verbose export information. |
| `force` | `False` | Force re-export even if simplified model already exists. |
| `shape` | `None` | Input shape as tuple `(height, width)`. Must be divisible by 14. If not provided, uses the model's default resolution. |
| `batch_size` | `1` | Batch size for the exported model. |

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

## Using the Exported Model

Once exported, you can use the ONNX model with various inference frameworks:

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

## Next Steps

After exporting your model, you may want to:

- [Deploy to Roboflow](deploy.md) for cloud-based inference and workflow integration
- Use the ONNX model with TensorRT for optimized GPU inference
- Integrate with edge deployment frameworks like ONNX Runtime or OpenVINO
