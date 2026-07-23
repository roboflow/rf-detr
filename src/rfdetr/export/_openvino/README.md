# OpenVINO Export for RF-DETR

This module provides direct PyTorch to OpenVINO IR export for RF-DETR models, without requiring an intermediate ONNX conversion.

## Installation

Install RF-DETR with OpenVINO support:

```bash
pip install "rfdetr[openvino]"
```

Or if you already have RF-DETR installed, just add OpenVINO:

```bash
pip install openvino
```

## Basic Usage

Export your trained model to OpenVINO IR format:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")

# Export to OpenVINO IR
model.export(format="openvino", output_dir="output")
```

This will create two files:
- `output/inference_model.xml` - The OpenVINO IR model
- `output/inference_model.bin` - The model weights

## Export Parameters

The `export()` method with `format="openvino"` accepts these parameters:

- **output_dir** (str, default: `"output"`): Directory where the exported model will be saved
- **format** (str): Set to `"openvino"` for OpenVINO IR export
- **backbone_only** (bool, default: `False`): Export only the backbone feature extractor
- **verbose** (bool, default: `True`): Print export progress information
- **shape** (tuple, optional): Input shape as `(height, width)`. If not provided, uses model's default resolution
- **batch_size** (int, default: `1`): Static batch size for the exported model

## Advanced Examples

### Export with Custom Output Directory

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")
model.export(format="openvino", output_dir="exports/my_model")
```

### Export with Custom Resolution

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")
model.export(format="openvino", shape=(608, 608))
```

### Export Backbone Only

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium(pretrain_weights="<path/to/checkpoint.pth>")
model.export(format="openvino", backbone_only=True)
```

## Using the Exported Model

### Python Inference

```python
import numpy as np
from PIL import Image
from rfdetr.export._openvino.inference import OpenVINOInference

# Load the exported model
model = OpenVINOInference("output/inference_model.xml")

# Prepare input image (NCHW format, ImageNet normalized)
image = Image.open("image.jpg").convert("RGB").resize((576, 576))
image_array = np.array(image).astype(np.float32) / 255.0

# Apply ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_array = (image_array - mean) / std

# Convert to NCHW format
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
outputs = model(image_array)
boxes, labels = outputs
```

### OpenVINO benchmark_app

Test the exported model performance:

```bash
benchmark_app -m output/inference_model.xml -data_shape [1,3,576,576]
```

## Model Outputs

The exported OpenVINO model produces the following outputs:

- **Object Detection Models**: 
  - Output 0: Bounding boxes `[batch, 300, 4]` (x, y, w, h in normalized coordinates)
  - Output 1: Class logits `[batch, 300, num_classes]`

- **Segmentation Models**:
  - Output 0: Bounding boxes `[batch, 300, 4]`
  - Output 1: Class logits `[batch, 300, num_classes]`
  - Output 2: Instance masks (if segmentation head is present)

- **Keypoint Models**:
  - Output 0: Bounding boxes `[batch, 300, 4]`
  - Output 1: Class logits `[batch, 300, num_classes]`
  - Output 2: Keypoints (if keypoint head is present)
