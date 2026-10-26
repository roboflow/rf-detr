---
description: Export basics for RF-DETR — install the export extras, run model.export(), find the output files, and run inference with the exported model.
---

# Export Basics

This page covers the everyday export path: install an extra, call `model.export()`, locate the output files, and run inference. For the full parameter reference and less common options, see [Advanced Export](advanced.md). For a comparison of formats and measured performance, see the [Overview](index.md).

For the full guide to loading a live model, checkpoint, or export and predicting images through the shared API, see [Run Inference](../learn/inference.md).

## Installation

Install the export dependencies you need:

=== "ONNX"

    ```bash
    pip install "rfdetr[onnx]"
    ```

=== "OpenVINO"

    ```bash
    pip install "rfdetr[openvino]"
    ```

=== "TFLite"

    ```bash
    pip install "rfdetr[tflite]"
    ```

=== "LiteRT"

    ```bash
    pip install "rfdetr[litert]"
    ```

=== "ExecuTorch"

    ```bash
    pip install "rfdetr[executorch]"
    ```

=== "Native CoreML (macOS)"

    ```bash
    pip install "rfdetr[coreml]"
    ```

=== "Apple Core AI"

    ```bash
    pip install "rfdetr[coreai]"
    ```

## Basic Export

Export your trained model to ONNX format:

=== "Object Detection"

    ```python
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

=== "Image Segmentation"

    ```python
    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium(pretrain_weights="<path/to/checkpoint.pth>")

    model.export()
    ```

This command saves the ONNX model to the `output` directory by default.

## Choose a Format

Pass `format=` to `model.export()` to pick another target; the extra from [Installation](#installation) must be installed first. Which format is fastest depends on the hardware — measured numbers are on the [Overview](index.md#measured-performance-by-hardware).

| Deploying to                | `format=`                      | Guide                                            |
| --------------------------- | ------------------------------ | ------------------------------------------------ |
| Anywhere ONNX Runtime runs  | `"onnx"` (default)             | [ONNX Inference](onnx.md)                        |
| NVIDIA GPU                  | `"tensorrt"` (alias `"trt"`)   | [TensorRT](tensorrt.md)                          |
| Intel CPU, GPU or NPU       | `"openvino"`                   | [OpenVINO](openvino.md)                          |
| Android, embedded, edge CPU | `"tflite"` or `"litert"`       | [TFLite](tflite.md), [LiteRT](litert.md)         |
| On-device PyTorch runtime   | `"executorch"` (alias `"pte"`) | [ExecuTorch](executorch.md)                      |
| Apple platforms (Xcode)     | `"coreml"` or `"coreai"`       | [Native CoreML](coreml.md), [Core AI](coreai.md) |

Format names are case-insensitive. Formats marked experimental in [Advanced Export](advanced.md#format-capabilities) emit a warning when constructed.

## Check the Export

An exported model returns raw tensors — box and logit decoding (sigmoid, background slot, box format) is left to your inference code. The [ONNX guide](onnx.md) spells out the decoding rules and two pitfalls that apply to every format. To confirm an export matches PyTorch, run both on the same image and compare detections; the [export cookbooks](../cookbooks.md) do this per hardware class, with latency and COCO mAP next to each other.

## Output Files

Filenames are built from the model's variant name (e.g. `rfdetr-medium`, falling back to `inference_model` when no variant or `output_name` is set, or `backbone_model` when `backbone_only=True` in that same case) plus a detail suffix whenever a detail materially changes the artifact — even at its default value, since the file needs to say what it actually is:

| Format        | Filename pattern                                                                                                                                                                                                                                                                                                                             | Detail encoded                                                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `onnx`        | `{variant}.onnx` (or `{variant}-backbone.onnx` if `backbone_only=True`); without a variant or `output_name`, `inference_model.onnx` (or `backbone_model.onnx` if `backbone_only=True`)                                                                                                                                                       | none — `-backbone` is structural, not a precision detail                                                                                         |
| `coreml`      | `{variant}_fp32.mlpackage` / `{variant}_fp16.mlpackage` (`{variant}_fp16-ane.mlpackage` if `coreml_neural_engine=True`; `{variant}_fp32-backbone.mlpackage` if `backbone_only=True`); without a variant or `output_name`, `backbone_model_fp32.mlpackage` if `backbone_only=True`                                                            | `coreml_precision`, plus `-ane` and `-backbone` when set                                                                                         |
| `coreai`      | `{variant}_fp32.aimodel` / `{variant}_fp16.aimodel`; without a variant or `output_name`, `inference_model_fp32.aimodel`                                                                                                                                                                                                                      | `coreai_precision`                                                                                                                               |
| `executorch`  | `{variant}_xnnpack.pte` / `{variant}_coreml.pte` / `{variant}_qnn_{soc}.pte` (or `{variant}_xnnpack-backbone.pte` if `backbone_only=True`); without a variant or `output_name`, `backbone_model_xnnpack.pte` if `backbone_only=True`                                                                                                         | `backend` (+ `soc` for `qnn`), plus `-backbone` when named                                                                                       |
| `tensorrt`    | `{variant}_fp16.trt` / `{variant}_fp32.trt` / `{variant}_int8.trt` (or `{variant}-backbone_fp16.trt` / `{variant}-backbone_fp32.trt` if `backbone_only=True`)                                                                                                                                                                                | `fp16` or `quantization="int8"`, plus `_ampere_plus` / `_same_compute_capability` and `_version_compatible` when set, and `-backbone` when named |
| `tflite`      | `{variant}_fp32.tflite` + `{variant}_fp16.tflite` (+ `{variant}_dynamic_range_quant.tflite` for `quantization="int8"`)                                                                                                                                                                                                                       | precision / quantization mode                                                                                                                    |
| `onnx` (int8) | `{variant}_int8.onnx` beside the FP32 `{variant}.onnx` the quantization was derived from                                                                                                                                                                                                                                                     | `_int8`                                                                                                                                          |
| `openvino`    | `{variant}.xml` + `{variant}.bin` (or `{variant}-backbone.xml`/`.bin` if `backbone_only=True`); with `quantization="int8"`, `{variant}_int8.xml` + `{variant}_int8.bin` (or `{variant}_int8-backbone.xml`/`.bin`); without a variant or `output_name`, `inference_model.xml`/`.bin` (or `backbone_model.xml`/`.bin` if `backbone_only=True`) | `quantization="int8"` (`_int8`), plus `-backbone` when named; `openvino_precision` controls IR weight compression, not the filename              |

Non-ONNX exports also write an inference metadata sidecar named `<artifact>.rfdetr.json`, including the artifact's extension (for example, `rfdetr-small_fp16.trt.rfdetr.json`). Copy it alongside the artifact when deploying with `RFDETRInference`; without it, loading fails unless you supply the missing metadata through `metadata=`. ONNX embeds its inference metadata in the `.onnx` file.

Pass `output_name="my-model"` to override the variant name and write `{output_name}.{ext}` verbatim — this suppresses the detail suffix for every format **except** `tflite` and `onnx` with `quantization="int8"`. Both write several files from one call: `tflite` keeps its `_fp32`/`_fp16`/`_dynamic_range_quant` suffix even with a custom name (`{output_name}_fp32.tflite`, etc.), and ONNX int8 writes the FP32 `{output_name}.onnx` beside the quantized `{output_name}_int8.onnx`, so dropping the suffix would make the two collide. OpenVINO int8 writes a single IR, so a custom `output_name` is taken verbatim and the `_int8` suffix is dropped.

With `backbone_only=True`, ONNX, CoreML, ExecuTorch, TensorRT, and OpenVINO retain a `-backbone` marker before the extension even when `output_name` is set, for example `my-model-backbone.onnx`. This distinguishes the backbone artifact from the full detector exported with the same name.

With `trt_metadata=True`, `format="tensorrt"` also writes a JSON description next to the engine, named after it: `{variant}_fp16.trt` gets `{variant}_fp16.json`. This optional engine description is separate from the `{variant}_fp16.trt.rfdetr.json` inference metadata sidecar used by `RFDETRInference`. See [Engine Description File](tensorrt.md#engine-description-file).

## Predict with RFDETRInference

For a native model, use its `inference()` factory to get an independent predictor snapshot. `RFDETRInference` also loads checkpoints and supported full exports, and can explicitly borrow a live model. All paths use the same image inputs and Supervision results as `RFDETR.predict()`.

```python
from rfdetr import RFDETRInference

predictor = RFDETRInference("output/inference_model.onnx")
detections = predictor.predict("image.jpg", threshold=0.5)
```

Detection and segmentation return `supervision.Detections`. Keypoint models return `supervision.KeyPoints`. A path ending in `.pth`, `.pt`, or `.ckpt` loads a native checkpoint. To use an existing model instance, pass it directly:

```python
from rfdetr import RFDETRInference, RFDETRSmall

native_model = RFDETRSmall()
model = RFDETRInference(native_model)
```

This wraps the live model without copying its weights. Each prediction reads the model's current state, including changes from training or legacy inference optimization. Do not pass `device` with a live model. Configure the native model's device when you create it. When you pass a path, the `RFDETRInference` instance loads and owns the native model or exported runtime. For an independent native snapshot, call `native_model.inference()` and use the returned predictor; it keeps its own copy of the current base weights and prediction settings.

The public constructor supports ONNX, TensorRT (`.trt` and `.engine`), TFLite, LiteRT, ExecuTorch, OpenVINO, native CoreML, Core AI, and native `.pth`/`.pt`/`.ckpt` checkpoints. Install the runtime package for the format on the target system. For ONNX, install `rfdetr[onnx]`: the loader reads the metadata embedded in the model with the `onnx` package and runs it with ONNX Runtime, and the extra installs both. LiteRT currently cannot export a full keypoint model. Backbone-only exports contain features rather than predictions and are not supported by this prediction API.

Pass a list to `predict()` to return one result per image. Each call runs the list as one batch. A fixed-shape export must accept that batch size and input shape. If it does not, `predict()` raises before runtime execution. It does not split or pad the list.

`runtime_info` reports the selected runtime and device policy. Some vendor runtimes choose a physical device after model load, so the policy might not identify one chip. An explicit device request must match the runtime and available hardware. For example, TensorRT requires CUDA. Core AI accepts `auto` or `cpu`; its GPU and Neural Engine options express a preference and cannot pin execution to that device.

Pass `runtime_options` to configure the runtime of an exported artifact. Each format reads only its own keys and raises `ValueError` for any other key, so a misspelled key, or one meant for another format, fails when the artifact is loaded instead of being ignored. Native models and checkpoints take no runtime options.

| Format        | Keys                                                                                                                                                                       |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TensorRT      | `cuda_graph`, `engine_host_code_allowed`, `verbose`, each a `bool` that defaults to `False`. See [Predict with RFDETRInference](tensorrt.md#predict-with-rfdetrinference). |
| OpenVINO      | `cache_dir`, `inference_precision`, `config`, with the meaning they have on the [OpenVINO session](openvino.md#advanced-use-the-openvino-session-directly).                |
| Other formats | None.                                                                                                                                                                      |

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/rfdetr-medium.xml", runtime_options={"cache_dir": "output/openvino-cache"})
```

New exports include versioned inference metadata. ONNX stores it in the model. Other formats use an adjacent `<artifact-name>.rfdetr.json` file. For example, `model.trt` uses `model.trt.rfdetr.json`. Keep this file with the artifact. The file records a SHA-256 hash of the artifact so the loader can refuse a sidecar left over from another export; the hash is a staleness check, not proof that the artifact is authentic. For an older export, load its source checkpoint and call `model.write_export_metadata(path, format=...)` with the original export settings; without the source checkpoint, pass a JSON path or mapping through `metadata=`. The metadata must supply missing task, class, output, and preprocessing semantics. The loader rejects missing or conflicting values. User-defined `notes` are separate from inference metadata.

```python
from rfdetr import RFDETRSmall

model = RFDETRSmall(pretrain_weights="checkpoint.pth")  # The source checkpoint of this export.
model.write_export_metadata("output/model.xml", format="openvino")
```

The loader selects the runtime from the metadata format and refuses a file whose extension does not belong to that format. TensorRT engines may carry any extension.

When loading an artifact with a sidecar, the constructor reads and hashes the artifact to check that the metadata still matches it. Reuse the loaded `RFDETRInference` instance across predictions. Intermediate ONNX files created while exporting TensorRT or TFLite are conversion inputs, not final artifacts, and do not receive ONNX metadata. To load one directly, pass the missing semantics through `metadata=`.

The shared API prepares images and decodes results on the native model's device or the runtime adapter's tensor device. TensorRT keeps tensors on CUDA through decoding. The other exported adapters currently use CPU tensors for these stages. Runtime outputs belong to each call, so later calls cannot change earlier results. The format guides retain advanced raw-runtime examples and numerical-reference helpers for low-level access and forward-only benchmarks; `RFDETRInference` is the common prediction path. The benchmarks in the overview measure the per-runtime cookbook pipelines.

### Metadata for older exports

Prefer re-exporting the original model with the current `rfdetr` so the exporter records its inference metadata. If re-exporting is unavailable, pass `metadata=` as a JSON file path or a JSON-serializable mapping. Supply the original model's settings and the artifact's actual input and output identifiers; metadata describes the graph and does not change it. The schema below is defined by `ExportMetadata` in `src/rfdetr/export/_runtime/metadata.py`.

All required fields must be present after merging supplied and stored metadata. Unknown fields and incorrect types are rejected; use JSON numbers and booleans rather than strings containing them.

| Required field | JSON value and meaning                                                                                                                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`       | One of `"onnx"`, `"tensorrt"`, `"openvino"`, `"tflite"`, `"litert"`, `"coreml"`, `"coreai"`, `"executorch"`.                                                                                                           |
| `task`         | `"detect"`, `"segment"`, `"keypoints"`, or `"backbone"`; `RFDETRInference` rejects backbone-only artifacts.                                                                                                            |
| `input_shape`  | Four integers in canonical `[batch, channels, height, width]` order, even for NHWC runtimes. Batch is positive or `-1` for dynamic batch; channels and spatial dimensions are positive.                                |
| `outputs`      | Object mapping semantic names to distinct runtime output names or non-negative positions. Detection needs `pred_boxes` and `pred_logits`; segmentation also needs `pred_masks`, keypoints also needs `pred_keypoints`. |
| `means`        | Array of finite normalization means, one per input channel.                                                                                                                                                            |
| `stds`         | Array of finite, positive normalization standard deviations, one per input channel.                                                                                                                                    |
| `class_names`  | Array of active class names in model order; nonempty for prediction.                                                                                                                                                   |
| `num_classes`  | Non-negative integer describing the model's class-slot count, not necessarily the number of active class names.                                                                                                        |
| `num_select`   | Non-negative integer specifying how many predictions the postprocessor selects.                                                                                                                                        |
| `trace_alpha`  | Finite, non-negative number controlling keypoint uncertainty scoring; required for detection too.                                                                                                                      |
| `patch_size`   | Positive integer: the original backbone's patch size.                                                                                                                                                                  |
| `num_windows`  | Positive integer: the original backbone's window configuration.                                                                                                                                                        |

| Optional field                 | Default and meaning                                                                                                                                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`               | `1`; this reader supports only version 1.                                                                                                                                                                                                                                     |
| `producer_version`             | `null`; producing `rfdetr` version, used in unsupported-schema diagnostics.                                                                                                                                                                                                   |
| `variant`                      | `null`; model variant label.                                                                                                                                                                                                                                                  |
| `backend`                      | `null`; export backend label.                                                                                                                                                                                                                                                 |
| `input_layout`                 | `"NCHW"`; also accepts `"NHWC"`.                                                                                                                                                                                                                                              |
| `input_dtype`                  | `"float32"`; runtime input dtype string.                                                                                                                                                                                                                                      |
| `input_name`                   | `"input"`; runtime input name or integer position.                                                                                                                                                                                                                            |
| `max_batch_size`               | `null`; a positive integer limit, allowed only with dynamic batch (`input_shape[0] == -1`).                                                                                                                                                                                   |
| `pixel_scale`                  | `1.0 / 255.0` (JSON number `0.00392156862745098`); only this native preprocessing scale is accepted.                                                                                                                                                                          |
| `channel_order`                | `"RGB"`; only RGB is accepted.                                                                                                                                                                                                                                                |
| `resize_interpolation`         | `"bilinear"`; only bilinear is accepted.                                                                                                                                                                                                                                      |
| `resize_antialias`             | `false`; only false is accepted.                                                                                                                                                                                                                                              |
| `class_id_to_name`             | `{}`; automatically filled with zero-based IDs when `num_classes == len(class_names)`. Otherwise required for prediction, for example for sparse COCO IDs or background-first keypoint slots. JSON object keys are stringified class IDs, such as `{"0": "cat", "1": "dog"}`. |
| `selection_policy`             | `"all_logits"`; only this policy is accepted.                                                                                                                                                                                                                                 |
| `box_format`                   | `"cxcywh_normalized"`; only normalized center-x, center-y, width, height is accepted.                                                                                                                                                                                         |
| `num_keypoints_per_class`      | `[]`; keypoint counts per class slot. Required for keypoint prediction, with at least one active slot.                                                                                                                                                                        |
| `upsample_masks_to_image_size` | `true`; whether the postprocessor upsamples masks to the source image size.                                                                                                                                                                                                   |

For example, this complete metadata object describes an ONNX detection export of a custom two-class RF-DETR Small model with batch 1, a 512 × 512 input, input named `input`, and outputs named `dets` and `labels`. Save it as `metadata.json`. Replace these illustrative settings with those of your export; in particular, this class list does not describe a pretrained COCO model.

```json
{
  "schema_version": 1,
  "format": "onnx",
  "task": "detect",
  "input_shape": [
    1,
    3,
    512,
    512
  ],
  "outputs": {
    "pred_boxes": "dets",
    "pred_logits": "labels"
  },
  "means": [
    0.485,
    0.456,
    0.406
  ],
  "stds": [
    0.229,
    0.224,
    0.225
  ],
  "class_names": [
    "cat",
    "dog"
  ],
  "num_classes": 2,
  "num_select": 300,
  "trace_alpha": 0.2,
  "patch_size": 16,
  "num_windows": 2
}
```

```python
from rfdetr import RFDETRInference

model = RFDETRInference("output/inference_model.onnx", metadata="metadata.json")
detections = model.predict("image.jpg", threshold=0.5)
```

The JSON file or mapping may instead hold an envelope of the form `{"artifact_sha256": "<artifact digest>", "metadata": { ... }}`, with the complete metadata object under `metadata`. Both keys must be present for an explicit override to be treated as an envelope. For a single-file ONNX artifact, the digest is the SHA-256 hex digest of its bytes. Automatically loaded `<artifact-name>.rfdetr.json` sidecars require this envelope. A digest mismatch is rejected, including when you also pass `metadata=`.

Supplied metadata can fill missing stored fields but cannot replace a stored value with a different value. Embedded ONNX metadata and a sidecar must agree if both exist. Omitting `schema_version` selects version 1; an unsupported version is rejected with an instruction to upgrade `rfdetr`. Do not relabel newer metadata as version 1 to bypass this check. Schema validation is followed by runtime interface checks; a schema-valid object alone does not establish that it matches an artifact.

## Run Inference with `inference-models`

[`inference-models`](https://github.com/roboflow/inference/tree/main/inference_models) is a separate deployment option. It supports PyTorch, ONNX, and TensorRT with automatic backend selection and its own API.

### Installation

```bash
# CPU / PyTorch only
pip install inference-models

# With TensorRT support (NVIDIA GPU required)
pip install "inference-models[trt10]"  # TensorRT 10
```

See the [inference-models installation guide](https://inference-models.roboflow.com/getting-started/installation/) for all installation options including Jetson and CUDA 11.x.

### Load a Pre-trained RF-DETR Model

```python
import cv2
from inference_models import AutoModel

# Automatically selects the best available backend for your environment
model = AutoModel.from_pretrained("rfdetr-small")

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
    model_type="rfdetr-small",  # specify the architecture variant
)

image = cv2.imread("image.jpg")
predictions = model(image)
```

### Force TensorRT Backend

```python
import cv2
from inference_models import AutoModel, BackendType

# Explicitly request TensorRT — requires TRT to be installed
model = AutoModel.from_pretrained("rfdetr-small", backend=BackendType.TRT)

image = cv2.imread("image.jpg")
predictions = model(image)
```

`AutoModel.from_pretrained` accepts `backend="onnx"`, `backend="torch"`, or `backend="trt"` to override automatic backend selection.

## Using the Exported Model

Use `RFDETRInference` for the common prediction path. The format guides also show direct runtime calls for advanced access and forward-only timing.
