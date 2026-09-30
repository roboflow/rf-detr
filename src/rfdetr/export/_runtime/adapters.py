# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Validate exported inference tensors and dispatch to format-owned runtimes."""

from __future__ import annotations

import platform
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rfdetr.export.registry import resolve_runtime_loader

if TYPE_CHECKING:
    from rfdetr.export._runtime.metadata import ExportMetadata

RawOutputs = dict[str, Any] | list[Any] | tuple[Any, ...]


def _validate_batch(batch: torch.Tensor, metadata: ExportMetadata) -> None:
    """Reject a batch that the exported graph cannot accept."""
    if batch.ndim != 4 or batch.shape[1] != metadata.input_shape[1]:
        raise ValueError(f"Expected an NCHW batch with {metadata.input_shape[1]} channels, got {tuple(batch.shape)}.")
    if batch.shape[0] == 0:
        raise ValueError("Export inference requires at least one image.")
    if metadata.max_batch_size is not None and batch.shape[0] > metadata.max_batch_size:
        raise ValueError(f"Batch size {batch.shape[0]} exceeds export maximum {metadata.max_batch_size}.")
    for axis, (actual, size) in enumerate(zip(batch.shape, metadata.input_shape)):
        if size != -1 and actual != size:
            raise ValueError(f"Input axis {axis} must be {size}, got {actual}. Export a model for this batch or size.")


def _input_array(batch: torch.Tensor, metadata: ExportMetadata) -> np.ndarray[Any, Any]:
    """Convert a normalized NCHW batch to the artifact's NumPy input interface."""
    if metadata.input_layout == "NHWC":
        batch = batch.permute(0, 2, 3, 1)
    try:
        dtype = np.dtype(metadata.input_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported export input dtype: {metadata.input_dtype!r}.") from exc
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(f"Export input dtype {dtype} needs quantization parameters, which this adapter cannot infer.")
    return np.ascontiguousarray(batch.detach().cpu().numpy(), dtype=dtype)


def _select_outputs(
    raw: RawOutputs,
    metadata: ExportMetadata,
    device: torch.device,
    *,
    borrowed: bool,
) -> dict[str, torch.Tensor]:
    """Map graph outputs to owned semantic tensors on the runtime device."""
    outputs: dict[str, torch.Tensor] = {}
    for semantic, identifier in metadata.outputs.items():
        try:
            if isinstance(raw, dict):
                value = list(raw.values())[identifier] if isinstance(identifier, int) else raw[identifier]
            elif isinstance(identifier, int):
                value = raw[identifier]
            else:
                raise KeyError(identifier)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Export output {semantic!r} maps to missing runtime output {identifier!r}.") from exc
        if isinstance(value, torch.Tensor):
            output = value.detach().clone() if borrowed else value.detach()
        else:
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.array(value, copy=True) if borrowed else np.asarray(value)
            output = torch.from_numpy(array)
        if not output.is_floating_point():
            raise ValueError(
                f"Export output {semantic!r} has non-floating dtype {output.dtype}; quantization is unknown."
            )
        if output.device != device:
            output = output.to(device)
        outputs[semantic] = output.float()
    if "pred_boxes" not in outputs or "pred_logits" not in outputs:
        raise ValueError("Export metadata must map pred_boxes and pred_logits.")
    boxes, logits = outputs["pred_boxes"], outputs["pred_logits"]
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or logits.ndim != 3 or boxes.shape[:2] != logits.shape[:2]:
        raise ValueError("Export boxes and logits have incompatible shapes.")
    if metadata.task == "segment" and "pred_masks" not in outputs:
        raise ValueError("Segmentation export has no pred_masks mapping.")
    if metadata.task == "keypoints" and "pred_keypoints" not in outputs:
        raise ValueError("Keypoint export has no pred_keypoints mapping.")
    for semantic in ("pred_masks", "pred_keypoints"):
        if semantic in outputs and outputs[semantic].shape[:2] != boxes.shape[:2]:
            raise ValueError(f"Export output {semantic!r} has a different batch or query count.")
    return outputs


class ExportRuntime:
    """A format-owned executor with shared input and output validation."""

    def __init__(
        self,
        backend: str,
        metadata: ExportMetadata,
        session: Any,
        device_policy: str,
        input_name: str | int,
        execute: Callable[[torch.Tensor], RawOutputs],
        *,
        device: torch.device = torch.device("cpu"),
        borrowed_outputs: bool = True,
    ) -> None:
        """Keep the runtime session and its tensor ownership policy."""
        self.metadata = metadata
        self.session = session
        self.input_name = input_name
        self.info = {"backend": backend, "device": device_policy}
        self.device = device
        self._execute = execute
        self._borrowed_outputs = borrowed_outputs
        self._run_lock = threading.Lock()

    def run(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one batch and return owned tensors on the runtime device."""
        _validate_batch(batch, self.metadata)
        with self._run_lock:
            raw = self._execute(batch)
            outputs = _select_outputs(raw, self.metadata, self.device, borrowed=self._borrowed_outputs)
            if self.device.type == "cuda" and self._borrowed_outputs:
                torch.cuda.current_stream(self.device).synchronize()
            return outputs


def _require_apple(format_name: str) -> None:
    """Reject Apple runtimes on other operating systems."""
    if platform.system() != "Darwin":
        raise RuntimeError(f"{format_name} inference requires macOS and its native runtime.")


def load_runtime(path: str | Path, metadata: ExportMetadata, device: str = "auto") -> ExportRuntime:
    """Load an artifact through its format-owned runtime loader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if metadata.task == "backbone":
        raise ValueError("A backbone-only export cannot produce predictions.")
    loader = resolve_runtime_loader(metadata.format.lower())
    return loader(path, metadata, device)
