# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## NEVER RUN INSTALLATION COMMANDS
- NEVER run pip install, conda install, apt install, or any package installation commands
- Always provide the command to the user and let them run it
- This applies to ALL installation and system modification commands

## Project Overview

RF-DETR is a real-time, transformer-based object detection and instance segmentation model developed by Roboflow. It achieves state-of-the-art performance on COCO and RF100-VL benchmarks.

## Common Commands

### Running Inference
```python
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRSegPreview

model = RFDETRBase()
model.optimize_for_inference()
detections = model.predict(image, threshold=0.5)
```

### Training
```python
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="path/to/dataset",
    epochs=100,
    batch_size=4,
    output_dir="output"
)
```

### CLI Training
```bash
rfdetr --workspace <workspace> --project_name <project> --api_key <key>
rfdetr --coco_dir path/to/coco/dataset
```

### ONNX Export
```python
model.export(output_dir="output", simplify=True)
```

### Documentation Server
```bash
mkdocs serve  # requires pip install rfdetr[docs]
```

## Architecture Overview

### Core Module Structure

```
rfdetr/
├── detr.py          # High-level API classes (RFDETR, RFDETRBase, RFDETRLarge, etc.)
├── main.py          # Model wrapper, training loop, HOSTED_MODELS download URLs
├── engine.py        # train_one_epoch() and evaluate() functions
├── config.py        # Pydantic configs: ModelConfig, TrainConfig, model size configs
├── models/
│   ├── lwdetr.py    # LWDETR nn.Module - the actual detection model
│   ├── backbone/    # DINOv2 backbone with windowed attention
│   ├── transformer.py
│   └── matcher.py
├── datasets/        # COCO format data loading and evaluation
├── deploy/          # ONNX export and optimization
└── util/            # Utilities, metrics, box ops
```

### Model Hierarchy

1. **User-facing API** (`rfdetr/detr.py`):
   - `RFDETR` base class with `predict()`, `train()`, `export()`, `optimize_for_inference()`
   - Subclasses: `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRBase`, `RFDETRLarge`, `RFDETRSegPreview`

2. **Internal Model** (`rfdetr/main.py`):
   - `Model` class wraps the actual PyTorch model
   - Handles weight loading, training orchestration, ONNX export

3. **Neural Network** (`rfdetr/models/lwdetr.py`):
   - `LWDETR(nn.Module)` - the actual detection transformer
   - Contains backbone, transformer, class_embed, bbox_embed

### Configuration System

- `ModelConfig` (Pydantic): encoder, hidden_dim, resolution, num_classes, etc.
- Size-specific configs inherit from `RFDETRBaseConfig`
- `TrainConfig`: lr, batch_size, epochs, dataset settings, early stopping, logging

### Key Design Patterns

- **Nested Tensors**: `NestedTensor` in `rfdetr/util/misc.py` handles variable-size batch inputs
- **EMA**: Exponential Moving Average models tracked during training for better checkpoint selection
- **Group DETR**: Multiple query groups for faster training convergence
- **Callbacks**: Training supports `on_fit_epoch_end`, `on_train_batch_start`, `on_train_end`

## Model Variants and Resolutions

| Model | Resolution | Patch Size | Windows | Dec Layers |
|-------|------------|------------|---------|------------|
| Nano | 384 | 16 | 2 | 2 |
| Small | 512 | 16 | 2 | 3 |
| Medium | 576 | 16 | 2 | 4 |
| Base | 560 | 14 | 4 | 3 |
| Large | 560 | 14 | 4 | 3 |
| Seg-Preview | 432 | 12 | 2 | 4 |

## Code Conventions

- Google-style docstrings with mandatory type hints for new code
- Apache 2.0 license headers on all files
- Uses `supervision` library for detection result handling
- Pydantic for configuration validation
