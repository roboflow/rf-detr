---
description: Advanced RF-DETR export — full model.export() parameter reference, custom resolution, backbone-only export, and how the export pipeline works internally.
---

# Advanced Export

This page is the reference for every `model.export()` parameter, plus less common export options. New to exporting? Start with [Export Basics](basics.md).

## Export Parameters

The `export()` method accepts several parameters to customize the export process:

| Parameter            | Default    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `output_dir`         | `"output"` | Directory where the exported model will be saved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `format`             | `"onnx"`   | Export format: `"onnx"`, `"tflite"`, `"tensorrt"` (alias: `"trt"`), `"executorch"`, `"openvino"`, `"coreml"`, `"coreai"` or `"litert"`.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `quantization`       | `None`     | TFLite quantization mode: `None`/`"fp32"`, `"fp16"`, or `"int8"`. Only used when `format="tflite"`; `format="litert"` accepts only `None`/`"fp32"` and raises `NotImplementedError` otherwise.                                                                                                                                                                                                                                                                                                                                                                                   |
| `calibration_data`   | `None`     | Optional image directory, `.npy` file path, NumPy array, or `None`. Not consumed when building the generated `.tflite` models.                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `max_images`         | `100`      | Maximum number of images to load from a `calibration_data` directory. Ignored for other calibration data formats.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `infer_dir`          | `None`     | Optional directory of sample images for inference validation during export tracing. If not provided, a random dummy image is generated.                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `backbone_only`      | `False`    | Export only the backbone feature extractor instead of the full model.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `opset_version`      | `17`       | ONNX opset version to use for export. Higher versions support more operations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `verbose`            | `True`     | Whether to print verbose export information.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `shape`              | `None`     | Input shape as tuple `(height, width)`. Each dimension must be divisible by the selected model's block size (`patch_size * num_windows`). If not provided, uses the model's default resolution.                                                                                                                                                                                                                                                                                                                                                                                  |
| `batch_size`         | `1`        | Batch size for the exported model. With `dynamic_batch=True` and `format="tensorrt"`, also the batch the engine's optimization profile is tuned for.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `dynamic_batch`      | `False`    | If `True`, export with a dynamic batch dimension so the model accepts variable batch sizes at runtime. Supported for `format="onnx"` and `format="tensorrt"` (which then needs `max_batch_size`) — TFLite, ExecuTorch, CoreML, Core AI, OpenVINO and LiteRT bake a fixed batch size (TFLite because `onnx2tf` fails on the dynamic-batch graph).                                                                                                                                                                                                                                 |
| `patch_size`         | `None`     | Backbone patch size override. Defaults to the value from `model_config.patch_size`. Must match the instantiated model's patch size when provided.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `backend`            | `None`     | Backend for ExecuTorch: `"xnnpack"` (CPU, fp32), `"coreml"` (Apple, fp16), or `"qnn"` (Qualcomm HTP, fp16). Required when `format="executorch"`.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `soc`                | `None`     | Target SoC chip identifier for the `"qnn"` backend (e.g. `"SM8650"` for Snapdragon 8 Gen 3). Required when `backend="qnn"`.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `fp16`               | `True`     | Build the TensorRT engine with FP16 precision (only used when `format="tensorrt"`). TensorRT 11+ removed the FP16 builder flag, so there the engine is built from an FP16-cast graph instead; engine inputs and outputs stay FP32 either way. On strongly typed TensorRT (11+), this graph cast requires `onnx`/`onnxconverter-common` — install `rfdetr[tensorrt]` for the complete set, or export raises `ImportError`. A lean/partial TensorRT < 11 wheel lacking the FP16 builder flag falls back to an FP32 engine with a warning instead. Pass `False` for an FP32 engine. |
| `max_batch_size`     | `None`     | Largest batch a dynamic TensorRT engine accepts. Required when `format="tensorrt"` and `dynamic_batch=True`: the engine gets one optimization profile spanning batch `1 .. max_batch_size`, tuned for `batch_size`. Ignored for every other format.                                                                                                                                                                                                                                                                                                                              |
| `notes`              | `None`     | Optional user-defined metadata (string, dict, list, or any JSON-serialisable value) to embed in the exported ONNX model under the `"rfdetr_notes"` metadata property.                                                                                                                                                                                                                                                                                                                                                                                                            |
| `coreml_precision`   | `None`     | Compute precision for `format="coreml"`: `None`/`"float32"` (tight CPU parity with eager PyTorch) or `"float16"` (half the size, and the only precision the Apple Neural Engine runs — at a measured accuracy cost, see [Native CoreML](coreml.md#neural-engine-compute-units-and-the-fallback-boundary)). Ignored for every other format.                                                                                                                                                                                                                                       |
| `coreai_precision`   | `None`     | Compute precision for `format="coreai"`: `None`/`"float32"` (matches eager PyTorch on the GPU) or `"float16"` (half the size, and the precision Core AI runs on the Apple Neural Engine — at a measured accuracy cost, see [Core AI](coreai.md#precision-compute-units-and-latency)). Ignored for every other format.                                                                                                                                                                                                                                                            |
| `openvino_precision` | `None`     | IR *storage* weight precision for `format="openvino"`: `None`/`"float16"` (OpenVINO's default FP16 weight compression) or `"float32"` (disables compression). Execution precision still depends on the compiled device — not guaranteed to match eager PyTorch on non-CPU devices. Ignored for every other format. Does not change the output filename.                                                                                                                                                                                                                          |
| `output_name`        | `None`     | Full filename override (without extension). Takes precedence over the model's variant name and suppresses the `_fp32`/`_fp16`/`_{backend}` detail suffix — see [Output Files](basics.md#output-files).                                                                                                                                                                                                                                                                                                                                                                           |

## Format Capabilities

What each format supports, as declared by its exporter. Parameters a format does not use are ignored with a warning.

| Format       | Extra                | `dynamic_batch`           | Embeds `notes` | Experimental |
| ------------ | -------------------- | ------------------------- | -------------- | ------------ |
| `onnx`       | `rfdetr[onnx]`       | yes                       | yes            | no           |
| `tensorrt`   | `rfdetr[tensorrt]`   | yes                       | yes            | no           |
| `tflite`     | `rfdetr[tflite]`     | no                        | yes            | yes          |
| `openvino`   | `rfdetr[openvino]`   | no                        | no             | no           |
| `executorch` | `rfdetr[executorch]` | no                        | no             | yes          |
| `litert`     | `rfdetr[litert]`     | no                        | no             | yes          |
| `coreml`     | `rfdetr[coreml]`     | no                        | no             | yes          |
| `coreai`     | `rfdetr[coreai]`     | no                        | yes            | yes          |

For a format without dynamic batch, export one artifact per batch size.

## Precision by Format

Each format exposes precision through its own parameter; [Export Parameters](#export-parameters) above has the details.

| Format       | Parameter            | Values                                                |
| ------------ | -------------------- | ----------------------------------------------------- |
| `tflite`     | `quantization`       | `None`/`"fp32"`, `"fp16"`, `"int8"` (dynamic-range)   |
| `litert`     | `quantization`       | `None`/`"fp32"` only                                  |
| `tensorrt`   | `fp16`               | `True` (default) or `False`                           |
| `openvino`   | `openvino_precision` | `None`/`"float16"` or `"float32"` (IR weight storage) |
| `coreml`     | `coreml_precision`   | `None`/`"float32"` or `"float16"`                     |
| `coreai`     | `coreai_precision`   | `None`/`"float32"` or `"float16"`                     |
| `executorch` | `backend`            | `"xnnpack"` (fp32), `"coreml"` (fp16), `"qnn"` (fp16) |

Lower precision is not a portable speedup — see [fp16 pays off only where the silicon implements it](index.md#measured-performance-by-hardware).

## Read Embedded Notes

Pass `notes=` to attach your own metadata, such as a dataset version or training run. For ONNX it is stored under the `rfdetr_notes` metadata property; strings are stored as is, any other JSON-serialisable value as JSON:

```python
import json

import onnx

model.export(notes={"dataset": "v3", "run": "2026-10-01"})

onnx_model = onnx.load("output/rfdetr-small.onnx")
notes = {prop.key: prop.value for prop in onnx_model.metadata_props}["rfdetr_notes"]
print(json.loads(notes))
```

## Advanced Export Examples

### Export with Custom Output Directory

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(output_dir="exports/my_model")
```

### Export with Custom Resolution

Export the model with a specific input resolution. For example, `RFDETRSmall` expects dimensions divisible by `32` (`patch_size=16`, `num_windows=2`):

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(shape=(608, 608))
```

### Export Backbone Only

Export only the backbone feature extractor for use in custom pipelines:

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

model.export(backbone_only=True)
```

The backbone export contains the encoder and its feature projector, without the detection decoder or prediction heads. ONNX outputs are feature maps in NCHW layout, ordered by `projector_scale`: `features` for the first level, followed by `features_1`, `features_2`, and so on when more levels are configured. Backbones with a second projector also return its levels as `cross_attn_features`, `cross_attn_features_1`, and so on, after the primary levels. These outputs are feature maps, not decoded boxes, masks, or keypoint coordinates.

## How Export Works

Every format is written by an `Exporter` class built from that format's own configuration, and `model.export()` is a facade over them: it resolves the format to an exporter, narrows this method's union-of-every-format signature down to the settings that format actually reads, prepares one format-independent `ExportGraph`, and hands the graph to the exporter. `RFDETR.export()`'s signature and return value are the supported surface; the classes behind it are internal.

If you want to add a format, or you are reading the export code, see [Exporter Blueprint](blueprint.md) for the contract each format implements and the steps a new one takes.
