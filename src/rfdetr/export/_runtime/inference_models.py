# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Use inference-models 0.39's public pipeline for RF-DETR exports.

The pinned SDK rounds boxes and keypoints, always aligns dense masks to source size, requires class remapping for
keypoints, and fixes keypoint score fusion at alpha=0.20. Recheck these assumptions before changing the SDK version
bound.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image
from supervision import Detections, KeyPoints

if TYPE_CHECKING:
    from rfdetr.export._runtime.metadata import ExportMetadata


def _check_task_shapes(
    input_shape: tuple[int, ...],
    output_shapes: list[tuple[int, ...]],
    metadata: ExportMetadata,
    format_name: str,
) -> int:
    """Validate the RF-DETR shape contract shared by ONNX and TensorRT."""
    boxes_shape, logits_shape = output_shapes[:2]
    if len(input_shape) != 4 or len(boxes_shape) != 3 or len(logits_shape) != 3:
        raise ValueError(f"{format_name} input, boxes, or logits have an unsupported rank.")
    for actual, expected in zip(input_shape, metadata.input_shape):
        if (actual <= 0) != (expected == -1) or (actual > 0 and actual != expected):
            raise ValueError(f"{format_name} input shape disagrees with inference metadata.")
    if any(not output for output in output_shapes):
        raise ValueError(f"{format_name} task output has an unsupported scalar shape.")
    if any(output[0] > 0 and output[0] != input_shape[0] for output in output_shapes):
        raise ValueError(f"{format_name} output batch dimension disagrees with the input.")
    if boxes_shape[-1] != 4 or boxes_shape[1] <= 0 or logits_shape[1] != boxes_shape[1]:
        raise ValueError(f"{format_name} detection outputs have incompatible query shapes.")
    if metadata.num_select != boxes_shape[1]:
        raise ValueError(
            f"inference_models selects every query ({boxes_shape[1]}); metadata requests {metadata.num_select}."
        )
    slots = logits_shape[-1]
    if slots <= 0 or slots < metadata.num_classes:
        raise ValueError(f"{format_name} logit class count disagrees with inference metadata.")
    if metadata.task == "segment":
        mask_shape = output_shapes[2]
        if len(mask_shape) != 4 or mask_shape[1] != boxes_shape[1] or min(mask_shape[2:]) <= 0:
            raise ValueError(f"{format_name} mask shape disagrees with detection queries or spatial size.")
    elif metadata.task == "keypoints":
        keypoint_shape = output_shapes[2]
        if len(metadata.num_keypoints_per_class) != slots:
            raise ValueError(f"Keypoint schema must describe each {format_name} logit slot.")
        expected_slots = slots * max(metadata.num_keypoints_per_class)
        if (
            len(keypoint_shape) != 4
            or keypoint_shape[1] != boxes_shape[1]
            or keypoint_shape[2] != expected_slots
            or keypoint_shape[3] != 8
        ):
            raise ValueError(f"{format_name} keypoint shape disagrees with query and class schema.")
    return slots


def _onnx_signature(path: Path, metadata: ExportMetadata) -> int:
    """Check the graph interface used by inference-models and return its class-slot count."""
    try:
        import onnx
    except ImportError as error:
        raise ImportError("The inference_models ONNX backend needs onnx to inspect the export.") from error

    graph = onnx.load(str(path), load_external_data=False).graph
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.initializer):
        raise ValueError("inference_models bridge does not support ONNX external weight files.")
    if len(graph.input) != 1 or len(graph.output) != (3 if metadata.task != "detect" else 2):
        raise ValueError("inference_models requires one ONNX input and the task's expected outputs.")
    if graph.input[0].name != metadata.input_name:
        raise ValueError("ONNX input name disagrees with inference metadata.")
    expected_outputs = [metadata.outputs["pred_boxes"], metadata.outputs["pred_logits"]]
    if metadata.task == "segment":
        expected_outputs.append(metadata.outputs["pred_masks"])
    elif metadata.task == "keypoints":
        expected_outputs.append(metadata.outputs["pred_keypoints"])
    if [output.name for output in graph.output] != expected_outputs:
        raise ValueError("inference_models requires task outputs in boxes, logits, then task-head order.")
    shapes = [
        tuple(
            dimension.dim_value if dimension.HasField("dim_value") else -1
            for dimension in value.type.tensor_type.shape.dim
        )
        for value in (graph.input[0], *graph.output)
    ]
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for value in (graph.input[0], *graph.output)):
        raise ValueError("inference_models requires float32 ONNX input and outputs.")
    return _check_task_shapes(shapes[0], shapes[1:], metadata, "ONNX")


def _tensorrt_signature(path: Path, metadata: ExportMetadata, device: torch.device) -> tuple[int, int]:
    """Check public TensorRT engine bindings and return class slots and profile opt batch."""
    try:
        trt: Any = importlib.import_module("tensorrt")
    except ImportError as error:
        raise ImportError("The inference_models TensorRT backend needs tensorrt.") from error

    with torch.cuda.device(device):
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise ValueError(f"TensorRT could not deserialize {path}.")
    if engine.num_io_tensors != (4 if metadata.task != "detect" else 3):
        raise ValueError("inference_models requires one TensorRT input and the task's expected outputs.")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    if (
        metadata.input_name not in names
        or metadata.outputs["pred_boxes"] != "dets"
        or metadata.outputs["pred_logits"] != "labels"
    ):
        raise ValueError("inference_models TensorRT requires input, dets, and labels bindings.")
    expected_names = {metadata.input_name, "dets", "labels"}
    if metadata.task == "segment":
        if metadata.outputs["pred_masks"] != "masks":
            raise ValueError("inference_models TensorRT segmentation requires a masks binding.")
        expected_names.add("masks")
        if [name for name in names if name != metadata.input_name] != ["dets", "labels", "masks"]:
            raise ValueError("inference_models TensorRT segmentation requires boxes, logits, then masks.")
    elif metadata.task == "keypoints":
        if metadata.outputs["pred_keypoints"] != "keypoints":
            raise ValueError("inference_models TensorRT keypoints requires a keypoints binding.")
        expected_names.add("keypoints")
    if set(names) != expected_names:
        raise ValueError("TensorRT engine bindings disagree with inference metadata.")
    input_shape = tuple(engine.get_tensor_shape(metadata.input_name))
    if any(engine.get_tensor_dtype(name) != trt.float32 for name in names):
        raise ValueError("inference_models requires float32 TensorRT I/O bindings.")
    output_names = ["dets", "labels"]
    if metadata.task == "segment":
        output_names.append("masks")
    elif metadata.task == "keypoints":
        output_names.append("keypoints")
    output_shapes = [tuple(engine.get_tensor_shape(name)) for name in output_names]
    slots = _check_task_shapes(input_shape, output_shapes, metadata, "TensorRT")
    opt_batch = metadata.input_shape[0]
    if opt_batch == -1:
        if metadata.max_batch_size is None:
            raise ValueError("Dynamic TensorRT inference_models requires max_batch_size metadata.")
        profile_min, profile_opt, profile_max = engine.get_tensor_profile_shape(metadata.input_name, 0)
        if profile_min[0] != 1 or metadata.max_batch_size > profile_max[0]:
            raise ValueError("TensorRT batch profile disagrees with inference metadata.")
        opt_batch = min(profile_opt[0], metadata.max_batch_size)
    return slots, opt_batch


class InferenceModelsPredictor:
    """Run a supported export through inference-models' complete public pipeline."""

    def __init__(self, path: Path, metadata: ExportMetadata, device: str) -> None:
        """Build one local inference-models package around a validated RF-DETR artifact."""
        if metadata.task not in {"detect", "segment", "keypoints"} or metadata.format not in {"onnx", "tensorrt"}:
            raise ValueError("inference_models supports detection, segmentation, or keypoints on ONNX/TRT.")
        if metadata.task == "segment" and not metadata.upsample_masks_to_image_size:
            raise ValueError("inference_models segmentation always returns source-image-size masks.")
        if metadata.task == "keypoints" and metadata.trace_alpha != 0.2:
            raise ValueError("inference_models keypoint score fusion requires trace_alpha=0.2.")
        if metadata.input_layout != "NCHW" or metadata.input_dtype != "float32" or metadata.num_channels != 3:
            raise ValueError("inference_models requires a float32 NCHW RGB detection export.")
        if metadata.selection_policy != "all_logits" or metadata.box_format != "cxcywh_normalized":
            raise ValueError("inference_models requires RF-DETR's raw detection output contract.")
        if (metadata.format == "onnx" and path.suffix != ".onnx") or (
            metadata.format == "tensorrt" and path.suffix not in {".trt", ".engine"}
        ):
            raise ValueError("Artifact extension does not match inference metadata.")
        if not path.is_file():
            raise FileNotFoundError(path)
        slots = _onnx_signature(path, metadata) if metadata.format == "onnx" else None
        requested: torch.device | None = None
        if device != "auto":
            try:
                requested = torch.device(device)
            except RuntimeError as error:
                raise ValueError(f"Unsupported device {device!r}.") from error
            if requested.type not in {"cpu", "cuda"}:
                raise ValueError("inference_models accepts cpu, cuda:N, or auto only.")
        try:
            auto_model_type: Any = importlib.import_module("inference_models").AutoModel
        except ImportError as error:
            raise ImportError(
                "backend='inference_models' requires inference-models>=0.39,<0.40 on Python 3.10-3.13 "
                f"with the selected runtime extra; SDK import failed: {error}"
            ) from error
        if requested is None:
            cuda_ready = torch.cuda.is_available()
            if metadata.format == "onnx":
                try:
                    import onnxruntime as ort
                except ImportError as error:
                    raise ImportError("The inference_models ONNX backend needs onnxruntime.") from error
                cuda_ready = cuda_ready and "CUDAExecutionProvider" in ort.get_available_providers()
            requested = torch.device("cuda:0" if cuda_ready else "cpu")
        if metadata.format == "tensorrt" and requested.type != "cuda":
            raise ValueError("TensorRT requires CUDA; it cannot run on CPU.")
        if requested.type == "cuda" and (
            not torch.cuda.is_available()
            or requested.index is not None
            and requested.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(f"Requested CUDA device {requested} is unavailable.")
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda:0")
        if metadata.format == "onnx":
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise ImportError("The inference_models ONNX backend needs onnxruntime.") from error
            provider = "CUDAExecutionProvider" if requested.type == "cuda" else "CPUExecutionProvider"
            if provider not in ort.get_available_providers():
                raise RuntimeError(f"Requested ONNX provider {provider} is unavailable.")
        else:
            slots, profile_opt_batch = _tensorrt_signature(path, metadata, requested)
        assert slots is not None

        self.metadata = metadata
        self.device = requested
        self._package = tempfile.TemporaryDirectory(prefix="rfdetr-inference-models-")
        package_dir = Path(self._package.name)
        package_backend = "onnx" if metadata.format == "onnx" else "trt"
        self._keypoint_raw_class_ids = [
            index for index, count in enumerate(metadata.num_keypoints_per_class) if count > 0
        ]
        slot_names = [
            f"__rfdetr_slot_{index}__"
            if metadata.task == "keypoints"
            else metadata.class_id_to_name.get(index, f"__unmapped_{index}__")
            for index in range(slots)
        ]
        try:
            (package_dir / "model_config.json").write_text(
                json.dumps(
                    {
                        "model_architecture": "rfdetr",
                        "task_type": {
                            "detect": "object-detection",
                            "segment": "instance-segmentation",
                            "keypoints": "keypoint-detection",
                        }[metadata.task],
                        "backend_type": package_backend,
                    }
                ),
                encoding="utf-8",
            )
            (package_dir / "inference_config.json").write_text(
                json.dumps(
                    {
                        "image_pre_processing": {},
                        "network_input": {
                            "training_input_size": {"height": metadata.shape[0], "width": metadata.shape[1]},
                            "dynamic_spatial_size_supported": False,
                            "color_mode": "rgb",
                            "resize_mode": "stretch",
                            "input_channels": 3,
                            "normalization": [metadata.means, metadata.stds],
                        },
                        "forward_pass": (
                            {"static_batch_size": metadata.input_shape[0]}
                            if metadata.input_shape[0] != -1
                            else {"max_dynamic_batch_size": metadata.max_batch_size}
                        ),
                        # The SDK keypoint decoder requires a remapping object even when every slot is active.
                        "class_names_operations": (
                            [
                                {"type": "class_name_removal", "class_name": slot_names[index]}
                                for index, count in enumerate(metadata.num_keypoints_per_class)
                                if count == 0
                            ]
                            or [{"type": "class_name_removal", "class_name": "__rfdetr_no_slot__"}]
                        )
                        if metadata.task == "keypoints"
                        else None,
                    }
                ),
                encoding="utf-8",
            )
            (package_dir / "class_names.txt").write_text("\n".join(slot_names) + "\n", encoding="utf-8")
            if metadata.task == "keypoints":
                keypoint_descriptions = [
                    {
                        "object_class": slot_names[index],
                        "object_class_id": index,
                        "keypoints": {str(point): f"keypoint_{point}" for point in range(count)},
                        "edges": [],
                    }
                    for index, count in enumerate(metadata.num_keypoints_per_class)
                ]
                (package_dir / "keypoints_metadata.json").write_text(
                    json.dumps(keypoint_descriptions), encoding="utf-8"
                )
            package_weights = package_dir / ("weights.onnx" if package_backend == "onnx" else "engine.plan")
            try:
                os.link(path.resolve(), package_weights)
            except OSError:
                shutil.copy2(path, package_weights)
            if package_backend == "trt":
                batch = metadata.input_shape[0]
                config: dict[str, int]
                if batch == -1:
                    max_batch = metadata.max_batch_size
                    if max_batch is None:
                        raise ValueError("Dynamic TensorRT inference_models requires max_batch_size metadata.")
                    config = {
                        "dynamic_batch_size_min": 1,
                        "dynamic_batch_size_opt": profile_opt_batch,
                        "dynamic_batch_size_max": max_batch,
                    }
                else:
                    config = {"static_batch_size": batch}
                (package_dir / "trt_config.json").write_text(json.dumps(config), encoding="utf-8")
            options: dict[str, Any] = {
                "backend": package_backend,
                "device": str(requested),
                "weights_provider": "local",
                "allow_local_code_packages": False,
                "allow_direct_local_storage_loading": True,
            }
            if package_backend == "onnx":
                options["onnx_execution_providers"] = (
                    [("CUDAExecutionProvider", {"device_id": requested.index or 0})]
                    if requested.type == "cuda"
                    else ["CPUExecutionProvider"]
                )
            self._model: Any = auto_model_type.from_pretrained(str(package_dir), **options)
            probe = np.zeros((metadata.shape[0], metadata.shape[1], 3), dtype=np.uint8)
            prepared, _ = self._model.pre_process(probe, input_color_format="rgb")
            if prepared.device != requested:
                raise RuntimeError(
                    f"inference_models prepared input on {prepared.device} instead of requested {requested}."
                )

        except Exception:
            self._package.cleanup()
            raise

    @property
    def class_names(self) -> list[str]:
        """Return the artifact's public class names rather than padded SDK slots."""
        return list(self.metadata.class_names)

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Report the loaded backend and the SDK's effective pipeline stages."""
        optimization = getattr(self._model, "optimization_runtime_metadata", None)
        return {
            "backend": "inference_models",
            "format": self.metadata.format,
            "device": str(self.device),
            "optimization": dict(optimization) if isinstance(optimization, Mapping) else optimization,
        }

    def predict(
        self,
        images: str
        | Image.Image
        | np.ndarray[Any, Any]
        | torch.Tensor
        | list[str | np.ndarray[Any, Any] | Image.Image | torch.Tensor],
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
        patch_size: int | None = None,
        include_source_image: bool = True,
        **kwargs: Any,
    ) -> Detections | KeyPoints | list[Detections | KeyPoints]:
        """Run one SDK batch and add the native RF-DETR result metadata."""
        if kwargs:
            raise ValueError(f"Unsupported inference_models predict options: {sorted(kwargs)}.")
        from rfdetr.detr import _resolve_patch_size, _validate_shape_dims

        resolved_patch = _resolve_patch_size(patch_size, self.metadata, "predict")
        if shape is not None:
            resolved_shape = _validate_shape_dims(
                shape, resolved_patch * self.metadata.num_windows, resolved_patch, self.metadata.num_windows
            )
            if resolved_shape != self.metadata.shape:
                raise ValueError(f"Export requires shape {self.metadata.shape}, but got {shape}.")
        image_list: list[str | Image.Image | np.ndarray[Any, Any] | torch.Tensor]
        if isinstance(images, (list, tuple)):
            single = False
            image_list = list(images)
        else:
            single = True
            image_list = [images]
        batch_size = len(image_list)
        fixed = self.metadata.input_shape[0]
        if batch_size == 0 or (fixed != -1 and batch_size != fixed):
            raise ValueError(f"Batch size mismatch. Export requires {fixed}, but got {batch_size}.")
        if self.metadata.max_batch_size is not None and batch_size > self.metadata.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds export maximum {self.metadata.max_batch_size}.")

        sdk_images: list[np.ndarray[Any, Any] | torch.Tensor] = []
        source_images: list[np.ndarray[Any, Any]] = []
        sizes: list[tuple[int, int]] = []
        for item in image_list:
            if isinstance(item, str):
                if urlparse(item).scheme in {"http", "https"}:
                    response = requests.get(item, timeout=30)
                    response.raise_for_status()
                    item = Image.open(io.BytesIO(response.content))
                else:
                    item = Image.open(item)
            if isinstance(item, Image.Image):
                array = np.array(item.convert("RGB"), dtype=np.uint8, copy=True)
                sdk_images.append(array)
                if include_source_image:
                    source_images.append(array)
                sizes.append(array.shape[:2])
            elif isinstance(item, np.ndarray):
                if item.ndim != 3 or item.shape[2] != 3:
                    raise ValueError("NumPy images must have HWC RGB shape.")
                if item.dtype == np.uint8:
                    sdk_images.append(item)
                    if include_source_image:
                        source_images.append(item.copy())
                elif np.issubdtype(item.dtype, np.floating):
                    if not np.isfinite(item).all() or (item < 0).any() or (item > 1).any():
                        raise ValueError("Float images must have finite pixel values in [0, 1].")
                    sdk_images.append(torch.from_numpy(np.ascontiguousarray(item, dtype=np.float32)).permute(2, 0, 1))
                    if include_source_image:
                        source_images.append((item * 255).clip(0, 255).astype(np.uint8))
                else:
                    raise ValueError("NumPy images must be uint8 or float in [0, 1].")
                sizes.append(item.shape[:2])
            elif isinstance(item, torch.Tensor):
                if item.ndim != 3 or item.shape[0] != 3:
                    raise ValueError("Tensor images must have CHW RGB shape.")
                if (
                    not item.is_floating_point()
                    or not bool(torch.isfinite(item).all())
                    or bool((item < 0).any())
                    or bool((item > 1).any())
                ):
                    raise ValueError("Tensor images must have finite float pixel values in [0, 1].")
                sdk_images.append(item)
                sizes.append((item.shape[1], item.shape[2]))
                if include_source_image:
                    source_images.append(
                        (item.detach().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    )
            else:
                raise TypeError(f"Unsupported image input type: {type(item).__name__}.")
        infer_options: dict[str, Any] = {"confidence": threshold, "input_color_format": "rgb"}
        if self.metadata.task == "segment":
            infer_options.update(mask_format="dense", max_detections=self.metadata.num_select)
        elif self.metadata.task == "keypoints":
            infer_options["key_points_threshold"] = 0.0
        predictions = self._model.infer(sdk_images, **infer_options)
        if self.metadata.task == "keypoints":
            if not isinstance(predictions, tuple) or len(predictions) != 2:
                raise RuntimeError("inference_models did not return keypoints and companion detections.")
            keypoint_predictions, box_predictions = predictions
            if box_predictions is None or len(keypoint_predictions) != batch_size or len(box_predictions) != batch_size:
                raise RuntimeError("inference_models keypoint batch or companion detections are incomplete.")
            keypoint_results: list[Detections | KeyPoints] = []
            for index, (prediction, box_prediction) in enumerate(zip(keypoint_predictions, box_predictions)):
                key_points = prediction.to_supervision()
                boxes = box_prediction.to_supervision()
                if len(key_points) != len(boxes):
                    raise RuntimeError("inference_models keypoints and companion boxes are misaligned.")
                key_points.xy = key_points.xy.astype(np.float32, copy=False)
                compact_ids = key_points.class_id
                if compact_ids is None or any(
                    int(class_id) < 0 or int(class_id) >= len(self._keypoint_raw_class_ids) for class_id in compact_ids
                ):
                    raise RuntimeError("inference_models returned a keypoint class outside the export schema.")
                key_points.class_id = np.asarray(
                    [self._keypoint_raw_class_ids[int(class_id)] for class_id in compact_ids], dtype=np.int64
                )
                key_points.data["xyxy"] = boxes.xyxy.astype(np.float32)
                key_points.data["class_name"] = np.asarray(
                    [self.metadata.class_id_to_name.get(int(class_id), "") for class_id in key_points.class_id],
                    dtype=object,
                )
                key_points.data["source_shape"] = np.tile(
                    np.asarray(sizes[index], dtype=np.int64), (len(key_points), 1)
                )
                if include_source_image:
                    key_points.data["source_image"] = [source_images[index] for _ in range(len(key_points))]
                keypoint_results.append(key_points)
            return keypoint_results[0] if single else keypoint_results
        if len(predictions) != batch_size:
            raise RuntimeError(f"inference_models returned {len(predictions)} results for {batch_size} images.")
        results: list[Detections | KeyPoints] = []
        for index, prediction in enumerate(predictions):
            # Other SDK task postprocessors may move results; preprocessing probes the selected device for all tasks.
            if self.metadata.task == "detect" and self.device.type == "cuda" and prediction.xyxy.device != self.device:
                raise RuntimeError(
                    f"inference_models returned predictions on {prediction.xyxy.device} "
                    f"instead of requested {self.device}."
                )
            detections = prediction.to_supervision()
            detections.xyxy = detections.xyxy.astype(np.float32, copy=False)
            if detections.class_id is not None:
                detections.class_id = detections.class_id.astype(np.int64, copy=False)
            class_ids = detections.class_id if detections.class_id is not None else np.array([], dtype=int)
            names = [
                self.metadata.class_id_to_name.get(
                    int(class_id),
                    "__background__" if class_id == self.metadata.num_classes else "",
                )
                for class_id in class_ids
            ]
            detections.data["class_name"] = np.asarray(names, dtype=object)
            detections.data["source_shape"] = np.tile(np.asarray(sizes[index], dtype=np.int64), (len(detections), 1))
            if include_source_image:
                detections.metadata["source_image"] = source_images[index]
            results.append(detections)
        return results[0] if single else results
