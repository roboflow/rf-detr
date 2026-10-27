# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Validate exported inference tensors and dispatch to format-owned runtimes."""

from __future__ import annotations

import platform
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import torch

from rfdetr.export.registry import REGISTRY, resolve_runtime_loader

if TYPE_CHECKING:
    from rfdetr.export._runtime.metadata import ExportMetadata

RawOutputs = dict[str, Any] | list[Any] | tuple[Any, ...]
_T = TypeVar("_T")


def _single_input(format_name: str, inputs: Sequence[_T]) -> _T:
    """Return a graph's only input, refusing a graph that declares any other number of inputs.

    Args:
        format_name: Format name as users read it, for the error message.
        inputs: The graph's input descriptions, in the runtime's own form.

    Returns:
        The one input description.

    Raises:
        ValueError: If *inputs* does not hold exactly one entry.

    Examples:
        >>> _single_input("ONNX", ["input"])
        'input'
        >>> _single_input("ONNX", ["input", "mask"])
        Traceback (most recent call last):
        ...
        ValueError: ONNX expected one input, got 2.
    """
    if len(inputs) != 1:
        raise ValueError(f"{format_name} expected one input, got {len(inputs)}.")
    return inputs[0]


def _check_input_interface(
    format_name: str,
    metadata: ExportMetadata,
    name: str | int | None,
    shape: Sequence[int | None],
    dtype: Any,
    *,
    dynamic_batch: bool = False,
) -> None:
    """Refuse a graph input whose name, rank, shape or dtype disagrees with the artifact's inference metadata.

    Args:
        format_name: Format name as users read it, for the error message.
        metadata: The artifact's inference metadata; its ``input_shape`` is NCHW.
        name: The graph input's name, or ``None`` for a format whose loader does not compare input names.
        shape: The graph input shape in the artifact's own layout; ``None`` marks an axis the graph leaves dynamic.
        dtype: The graph input dtype, as a NumPy dtype or dtype name.
        dynamic_batch: Accept any graph batch axis when the metadata batch is ``-1``. Leave it off for a format whose
            graph batch must equal the metadata batch.

    Raises:
        ValueError: If the name, rank, shape or dtype disagrees with *metadata*.

    Examples:
        >>> from types import SimpleNamespace
        >>> metadata = SimpleNamespace(
        ...     input_name="input", input_shape=(-1, 3, 8, 8), input_layout="NCHW", input_dtype="float32"
        ... )
        >>> _check_input_interface("TensorRT", metadata, "input", (4, 3, 8, 8), "float32", dynamic_batch=True)
        >>> _check_input_interface("CoreML", metadata, None, (4, 3, 8, 8), "float32")
        Traceback (most recent call last):
        ...
        ValueError: CoreML input shape (4, 3, 8, 8) disagrees with export metadata (-1, 3, 8, 8).
    """
    if name is not None and name != metadata.input_name:
        raise ValueError(f"{format_name} input is {name!r}; metadata says {metadata.input_name!r}.")
    expected = metadata.input_shape
    if len(shape) != len(expected):
        raise ValueError(f"{format_name} input rank must be {len(expected)}, got {len(shape)}.")
    if metadata.input_layout == "NHWC":
        expected = (expected[0], expected[2], expected[3], expected[1])
    for axis, (got, want) in enumerate(zip(shape, expected)):
        if got is not None and got != want and not (dynamic_batch and axis == 0 and want == -1):
            raise ValueError(f"{format_name} input shape {tuple(shape)} disagrees with export metadata {expected}.")
    try:
        matches = np.dtype(dtype) == np.dtype(metadata.input_dtype)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError(f"{format_name} input dtype {dtype!r} disagrees with export metadata {metadata.input_dtype}.")


def _check_output_interface(format_name: str, metadata: ExportMetadata, outputs: Sequence[str] | int) -> None:
    """Refuse output mappings that name or index an output the graph does not have.

    Args:
        format_name: Format name as users read it, for the error message.
        metadata: The artifact's inference metadata.
        outputs: The graph's output names in graph order, or only their count for a runtime that exposes no names.
            A count admits positional mappings only.

    Raises:
        ValueError: If a mapping names or indexes an absent output, or names an output where only a count is known.

    Examples:
        >>> from types import SimpleNamespace
        >>> metadata = SimpleNamespace(outputs={"pred_boxes": "dets", "pred_logits": 1})
        >>> _check_output_interface("ONNX", metadata, ["dets", "labels"])
        >>> _check_output_interface("OpenVINO", metadata, 2)
        Traceback (most recent call last):
        ...
        ValueError: OpenVINO output mappings must use positions.
    """
    count = outputs if isinstance(outputs, int) else len(outputs)
    for identifier in metadata.outputs.values():
        if isinstance(identifier, int):
            if not 0 <= identifier < count:
                raise ValueError(f"{format_name} output position {identifier} is absent from the graph.")
        elif isinstance(outputs, int):
            raise ValueError(f"{format_name} output mappings must use positions.")
        elif identifier not in outputs:
            raise ValueError(f"{format_name} output name {identifier!r} is absent from the graph.")


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
    torch_dtype = torch.from_numpy(np.empty(0, dtype=dtype)).dtype
    return np.ascontiguousarray(batch.detach().to("cpu", torch_dtype).numpy())


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
    # The detection head emits one logit per class id plus the trailing no-object slot that prediction labels as
    # background, so a narrower or wider tensor would silently shift class ids.
    if logits.shape[-1] != metadata.num_classes + 1:
        raise ValueError(
            f"Export logits have {logits.shape[-1]} class slots, but metadata num_classes={metadata.num_classes}"
            f" needs {metadata.num_classes + 1} (class ids plus the no-object slot)."
        )
    if metadata.task == "segment" and "pred_masks" not in outputs:
        raise ValueError("Segmentation export has no pred_masks mapping.")
    if metadata.task == "keypoints" and "pred_keypoints" not in outputs:
        raise ValueError("Keypoint export has no pred_keypoints mapping.")
    for semantic in ("pred_masks", "pred_keypoints"):
        if semantic in outputs and outputs[semantic].shape[:2] != boxes.shape[:2]:
            raise ValueError(f"Export output {semantic!r} has a different batch or query count.")
    return outputs


class ExportRuntime:
    """A format-owned executor with shared input and output validation.

    Each format's loader wraps its session and execute function in one of these, so every format checks the batch
    against the metadata, maps raw outputs to semantic names, and returns owned float tensors the same way. Calls to
    :meth:`run` are serialized by a lock.

    Args:
        backend: Runtime backend name reported in runtime information.
        metadata: Artifact input contract and semantic output mapping.
        session: Loaded runtime session retained by the executor.
        device_policy: Runtime device policy reported in runtime information.
        execute: Callback that executes one batch and returns raw outputs.
        device: Device on which returned tensors are placed.
        borrowed_outputs: Whether raw outputs use runtime-owned storage that must be copied.
    """

    def __init__(
        self,
        backend: str,
        metadata: ExportMetadata,
        *,
        session: Any,
        device_policy: str,
        execute: Callable[[torch.Tensor], RawOutputs],
        device: torch.device = torch.device("cpu"),
        borrowed_outputs: bool = True,
    ) -> None:
        """Keep the runtime session and its tensor ownership policy.

        Args:
            backend: Runtime name reported in ``info``.
            metadata: The artifact's inference metadata.
            session: The format's loaded session, kept for callers that need it.
            device_policy: Device policy reported in ``info``.
            input_name: Name or position of the graph input.
            execute: Runs one input batch and returns the raw runtime outputs.
            device: Device the returned tensors are placed on.
            borrowed_outputs: Whether *execute* returns buffers the runtime reuses, which :meth:`run` copies.
        """
        self.metadata = metadata
        self.session = session
        self.info = {"backend": backend, "device": device_policy}
        self.device = device
        self._execute = execute
        self._borrowed_outputs = borrowed_outputs
        self._run_lock = threading.Lock()

    def run(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one batch and return owned tensors on the runtime device.

        Args:
            batch: Normalized NCHW input batch.

        Returns:
            Float tensors keyed by semantic name (``pred_boxes``, ``pred_logits``, and ``pred_masks`` or
            ``pred_keypoints`` when the task has them).

        Raises:
            ValueError: If the batch does not match the metadata's input shape, channel count, or batch limit, or the
                runtime outputs are missing, not floating point, or shaped inconsistently with the metadata.
        """
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


def _runtime_options(format_name: str, options: Mapping[str, Any], accepted: Collection[str]) -> dict[str, Any]:
    """Return a copy of a loader's runtime options after refusing every key that loader does not read.

    Each format loader calls this first, before its body imports its runtime package, so a misspelled key, or a key
    meant for another format, fails the same way on every host instead of being ignored. A loader module that imports
    its optional runtime at module scope guards that import with ``except ImportError``, so it stays importable.

    Args:
        format_name: Format name as users read it, for the error message.
        options: The ``runtime_options`` mapping passed to ``RFDETRInference``.
        accepted: Keys the format's loader reads.

    Returns:
        A plain ``dict`` copy of *options*.

    Raises:
        TypeError: If *options* is not a mapping.
        ValueError: If *options* holds a key outside *accepted*.

    Examples:
        >>> _runtime_options("OpenVINO", {"cache_dir": "cache"}, ("cache_dir", "config"))
        {'cache_dir': 'cache'}
        >>> _runtime_options("ONNX", {"providers": []}, ())
        Traceback (most recent call last):
        ...
        ValueError: ONNX runtime_options does not accept 'providers'. Accepted keys: none.
    """
    if not isinstance(options, Mapping):
        raise TypeError(f"runtime_options must be a mapping, got {type(options).__name__}.")
    unknown = sorted(repr(key) for key in options if key not in accepted)
    if unknown:
        allowed = ", ".join(sorted(accepted)) or "none"
        raise ValueError(
            f"{format_name} runtime_options does not accept {', '.join(unknown)}. Accepted keys: {allowed}."
        )
    return dict(options)


def load_runtime(
    path: str | Path,
    metadata: ExportMetadata,
    device: str = "auto",
    options: Mapping[str, Any] | None = None,
) -> ExportRuntime:
    """Load an artifact through its format-owned runtime loader.

    Args:
        path: Exported file or model bundle.
        metadata: The artifact's inference metadata.
        device: Runtime device policy, in the format's own vocabulary.
        options: Format runtime settings; ``None`` passes none. The loader refuses keys it does not read.

    Returns:
        The loaded runtime.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the export is backbone-only, its suffix disagrees with the metadata format, the format is
            unsupported, or the loader refuses *device* or *options*.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if metadata.task == "backbone":
        raise ValueError("A backbone-only export cannot produce predictions.")
    entry = REGISTRY.get(metadata.format.lower())
    if entry is not None and entry.suffixes and path.suffix.lower() not in entry.suffixes:
        raise ValueError(
            f"{entry.label} artifacts require one of {sorted(entry.suffixes)}, got {path.name!r}. "
            "Pass the file the exporter wrote or fix the format in the metadata."
        )
    loader = resolve_runtime_loader(metadata.format.lower())
    return loader(path, metadata, device, {} if options is None else options)
