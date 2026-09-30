# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Use the public inference-models pipeline for supported RF-DETR exports."""

from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image
from supervision import Detections

if TYPE_CHECKING:
    from rfdetr.export._runtime.metadata import ExportMetadata


def _onnx_signature(path: Path, metadata: ExportMetadata) -> int:
    """Check the graph interface used by inference-models and return its class-slot count."""
    try:
        import onnx
    except ImportError as error:
        raise ImportError("The inference_models ONNX backend needs onnx to inspect the export.") from error

    graph = onnx.load(str(path), load_external_data=False).graph
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.initializer):
        raise ValueError("inference_models bridge does not support ONNX external weight files.")
    if len(graph.input) != 1 or len(graph.output) != 2:
        raise ValueError("inference_models requires one ONNX input and exactly two detection outputs.")
    if graph.input[0].name != metadata.input_name:
        raise ValueError("ONNX input name disagrees with inference metadata.")
    if [output.name for output in graph.output] != [
        metadata.outputs["pred_boxes"],
        metadata.outputs["pred_logits"],
    ]:
        raise ValueError("inference_models requires boxes then logits as the first two ONNX outputs.")
    shapes = [
        [
            dimension.dim_value if dimension.HasField("dim_value") else -1
            for dimension in value.type.tensor_type.shape.dim
        ]
        for value in (graph.input[0], *graph.output)
    ]
    input_shape, boxes_shape, logits_shape = shapes
    if len(input_shape) != 4 or len(boxes_shape) != 3 or len(logits_shape) != 3:
        raise ValueError("ONNX input, boxes, or logits have an unsupported rank.")
    if any(value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for value in (graph.input[0], *graph.output)):
        raise ValueError("inference_models requires float32 ONNX input and outputs.")
    for actual, expected in zip(input_shape, metadata.input_shape):
        if (actual <= 0) != (expected == -1) or (actual > 0 and actual != expected):
            raise ValueError("ONNX input shape disagrees with inference metadata.")
    if any(output[0] > 0 and output[0] != input_shape[0] for output in (boxes_shape, logits_shape)):
        raise ValueError("ONNX output batch dimension disagrees with the input.")
    if boxes_shape[-1] != 4 or boxes_shape[1] <= 0 or logits_shape[1] != boxes_shape[1]:
        raise ValueError("ONNX detection outputs have incompatible query shapes.")
    if metadata.num_select != boxes_shape[1]:
        raise ValueError(
            f"inference_models selects every query ({boxes_shape[1]}); metadata requests {metadata.num_select}."
        )
    if logits_shape[-1] <= 0 or logits_shape[-1] < metadata.num_classes:
        raise ValueError("ONNX logit class count disagrees with inference metadata.")
    return logits_shape[-1]


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
    if engine.num_io_tensors != 3:
        raise ValueError("inference_models requires one TensorRT input and two detection outputs.")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    if (
        metadata.input_name not in names
        or metadata.outputs["pred_boxes"] != "dets"
        or metadata.outputs["pred_logits"] != "labels"
    ):
        raise ValueError("inference_models TensorRT requires input, dets, and labels bindings.")
    if set(names) != {metadata.input_name, "dets", "labels"}:
        raise ValueError("TensorRT engine bindings disagree with inference metadata.")
    input_shape = tuple(engine.get_tensor_shape(metadata.input_name))
    boxes_shape = tuple(engine.get_tensor_shape("dets"))
    logits_shape = tuple(engine.get_tensor_shape("labels"))
    if len(input_shape) != 4 or len(boxes_shape) != 3 or len(logits_shape) != 3:
        raise ValueError("TensorRT input, boxes, or logits have an unsupported rank.")
    if any(engine.get_tensor_dtype(name) != trt.float32 for name in names):
        raise ValueError("inference_models requires float32 TensorRT I/O bindings.")
    for actual, expected in zip(input_shape, metadata.input_shape):
        if (actual <= 0) != (expected == -1) or (actual > 0 and actual != expected):
            raise ValueError("TensorRT input shape disagrees with inference metadata.")
    if any(output[0] > 0 and output[0] != input_shape[0] for output in (boxes_shape, logits_shape)):
        raise ValueError("TensorRT output batch dimension disagrees with the input.")
    opt_batch = metadata.input_shape[0]
    if opt_batch == -1:
        if metadata.max_batch_size is None:
            raise ValueError("Dynamic TensorRT inference_models requires max_batch_size metadata.")
        profile_min, profile_opt, profile_max = engine.get_tensor_profile_shape(metadata.input_name, 0)
        if profile_min[0] != 1 or metadata.max_batch_size > profile_max[0]:
            raise ValueError("TensorRT batch profile disagrees with inference metadata.")
        opt_batch = min(profile_opt[0], metadata.max_batch_size)
    if boxes_shape[-1] != 4 or boxes_shape[1] <= 0 or logits_shape[1] != boxes_shape[1]:
        raise ValueError("TensorRT detection outputs have incompatible query shapes.")
    if metadata.num_select != boxes_shape[1]:
        raise ValueError(
            f"inference_models selects every query ({boxes_shape[1]}); metadata requests {metadata.num_select}."
        )
    if logits_shape[-1] <= 0 or logits_shape[-1] < metadata.num_classes:
        raise ValueError("TensorRT logit class count disagrees with inference metadata.")
    return logits_shape[-1], opt_batch


class InferenceModelsPredictor:
    """Run a detection export through inference-models' complete public pipeline."""

    def __init__(self, path: Path, metadata: ExportMetadata, device: str) -> None:
        """Build one local inference-models package around a validated RF-DETR artifact."""
        if metadata.task != "detect" or metadata.format not in {"onnx", "tensorrt"}:
            raise ValueError("inference_models supports full object-detection ONNX and TensorRT exports only.")
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
        auto_device = device
        if device == "auto":
            cuda_ready = torch.cuda.is_available()
            if metadata.format == "onnx":
                try:
                    import onnxruntime as ort
                except ImportError as error:
                    raise ImportError("The inference_models ONNX backend needs onnxruntime.") from error
                cuda_ready = cuda_ready and "CUDAExecutionProvider" in ort.get_available_providers()
            auto_device = "cuda:0" if cuda_ready else "cpu"
        try:
            requested = torch.device(auto_device)
        except RuntimeError as error:
            raise ValueError(f"Unsupported device {device!r}.") from error
        if requested.type not in {"cpu", "cuda"}:
            raise ValueError("inference_models accepts cpu, cuda:N, or auto only.")
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
            slots = _onnx_signature(path, metadata)
        else:
            slots, profile_opt_batch = _tensorrt_signature(path, metadata, requested)

        try:
            auto_model_type: Any = importlib.import_module("inference_models").AutoModel
        except ImportError as error:
            raise ImportError(
                "backend='inference_models' requires inference-models with the selected ONNX or TensorRT extra."
            ) from error

        self.metadata = metadata
        self.device = requested
        self._package = tempfile.TemporaryDirectory(prefix="rfdetr-inference-models-")
        package_dir = Path(self._package.name)
        package_backend = "onnx" if metadata.format == "onnx" else "trt"
        try:
            (package_dir / "model_config.json").write_text(
                json.dumps(
                    {
                        "model_architecture": "rfdetr",
                        "task_type": "object-detection",
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
                    }
                ),
                encoding="utf-8",
            )
            slot_names = [metadata.class_id_to_name.get(index, f"__unmapped_{index}__") for index in range(slots)]
            (package_dir / "class_names.txt").write_text("\n".join(slot_names) + "\n", encoding="utf-8")
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
        return {
            "backend": "inference_models",
            "format": self.metadata.format,
            "device": str(self.device),
            "optimization": getattr(self._model, "optimization_runtime_metadata", None),
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
    ) -> Detections | list[Detections]:
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
                array = np.asarray(item.convert("RGB"), dtype=np.uint8)
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
        predictions = self._model.infer(sdk_images, confidence=threshold, input_color_format="rgb")
        if len(predictions) != batch_size:
            raise RuntimeError(f"inference_models returned {len(predictions)} results for {batch_size} images.")
        results: list[Detections] = []
        for index, prediction in enumerate(predictions):
            if self.device.type == "cuda" and prediction.xyxy.device != self.device:
                raise RuntimeError(
                    f"inference_models returned predictions on {prediction.xyxy.device} "
                    f"instead of requested {self.device}."
                )
            detections = prediction.to_supervision()
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
