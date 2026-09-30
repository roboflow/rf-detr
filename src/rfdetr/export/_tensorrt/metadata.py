# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The ``<engine>.json`` sidecar that describes a serialized TensorRT engine.

A ``.trt`` file has no slot for user metadata, so a consumer that does not import the model (a C++ service, Triton,
DeepStream) cannot learn its input size, normalization, outputs or batch profile from it. The sidecar carries them.
:meth:`~rfdetr.export._tensorrt.exporter.TensorRTExporter._write_metadata` writes it, from ``_convert``, when
``trt_metadata=True``.

Schema version 1 does not record class names or the background class slot: how a logit slot maps to a class name
depends on the checkpoint and is decided inside ``RFDETR.predict``, not by anything this package can read.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from rfdetr.export._runtime.preprocess import IMAGENET_MEAN, IMAGENET_STD
from rfdetr.export.base import serialize_notes
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_version

if TYPE_CHECKING:
    from rfdetr.export._tensorrt.exporter import TensorRTConfig

logger = get_logger()

#: Version of the document layout. Bumped when a key is removed, renamed or changes meaning. Adding a key does not bump
#: it: a consumer ignores the keys it does not know.
METADATA_SCHEMA_VERSION: Final[int] = 1

#: dtype of every engine input and output. ``fp16`` builds keep FP32 I/O too: the FP16 cast graph restores FP32 at the
#: boundary and the builder flag only changes the layers in between.
_IO_DTYPE: Final[str] = "float32"


def sidecar_path(engine_path: str | os.PathLike[str]) -> Path:
    """Return the path of the sidecar that belongs to *engine_path*: the same name with a ``.json`` suffix.

    Args:
        engine_path: Path of the ``.trt`` engine.

    Returns:
        The sidecar path, in the engine's directory.

    Examples:
        >>> sidecar_path("output/rfdetr-nano_fp16.trt").as_posix()
        'output/rfdetr-nano_fp16.json'
    """
    return Path(engine_path).with_suffix(".json")


def is_engine_description(path: str | os.PathLike[str]) -> bool:
    """Whether *path* is a description this exporter wrote: a file holding a JSON object with ``schema_version``.

    Used before a warning or an error claims that a ``.json`` beside an engine describes an earlier engine, so a
    directory or an unrelated JSON file with the same name is never mistaken for one.

    Args:
        path: The file to look at.

    Returns:
        ``True`` for a description, ``False`` for anything else, including a file that cannot be read.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     description, other = Path(folder, "a.json"), Path(folder, "b.json")
        ...     _ = description.write_text('{"schema_version": 1}')
        ...     _ = other.write_text('{"name": "not ours"}')
        ...     is_engine_description(description), is_engine_description(other), is_engine_description(folder)
        (True, False, False)
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False
    return isinstance(document, dict) and "schema_version" in document


def gpu_facts() -> dict[str, str] | None:
    """Describe the CUDA device TensorRT builds on, or ``None`` when torch cannot see one.

    Returns:
        ``{"name", "compute_capability"}`` for the current CUDA device. ``None`` for a torch build without CUDA, where
        TensorRT can still build but nothing here can name the GPU.
    """
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {"name": properties.name, "compute_capability": f"{properties.major}.{properties.minor}"}


def build_engine_metadata(
    config: TensorRTConfig,
    graph: ExportGraph,
    *,
    precision: str,
    tensorrt_version: str,
    gpu: dict[str, str] | None,
) -> dict[str, Any]:
    """Assemble the schema version 1 document for one engine.

    Args:
        config: The exporter configuration the engine was built from.
        graph: The prepared graph the engine was built from.
        precision: The precision the engine was actually built with (``"fp16"`` or ``"fp32"``), which can differ from
            the request on a lean TensorRT wheel.
        tensorrt_version: The version TensorRT reported for the build.
        gpu: The building GPU, as :func:`gpu_facts` returns it.

    Returns:
        A JSON-serializable dict.

    Raises:
        ValueError: If ``config.notes`` holds a non-finite float or a circular reference.
        TypeError: If ``config.notes`` holds a value JSON cannot encode.

    Examples:
        >>> import torch
        >>> from rfdetr.export._tensorrt.exporter import TensorRTConfig
        >>> graph = ExportGraph(
        ...     model=torch.nn.Identity(), input_tensors=torch.zeros(1, 3, 8, 8), input_names=("input",),
        ...     output_names=("dets", "labels"), dynamic_axes=None, shape=(8, 8), backbone_only=False,
        ... )
        >>> document = build_engine_metadata(
        ...     TensorRTConfig(), graph, precision="fp16", tensorrt_version="11.3.0.99", gpu=None
        ... )
        >>> document["batch"], document["input"]["height"], [o["name"] for o in document["outputs"]]
        ({'dynamic': False, 'size': 1}, 8, ['dets', 'labels'])
    """
    channels = int(graph.input_tensors.shape[1])
    if config.dynamic_batch:
        batch: dict[str, Any] = {"dynamic": True, "min": 1, "opt": config.opt_batch_size, "max": config.max_batch_size}
    else:
        batch = {"dynamic": False, "size": int(graph.input_tensors.shape[0])}
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "rfdetr_version": get_version(),
        "variant": config.variant_name,
        "backbone_only": graph.backbone_only,
        "input": {
            "name": graph.input_names[0],
            "layout": "NCHW",
            "dtype": _IO_DTYPE,
            "height": int(graph.shape[0]),
            "width": int(graph.shape[1]),
            "channels": channels,
            # Only a three-channel model is RGB; the order of any other channel layout is not something rfdetr knows.
            "channel_order": "RGB" if channels == 3 else None,
            # The convention of ``preprocess_to_nchw``, which ``RFDETR.predict`` follows: pixels divided by ``scale``,
            # then ``(x - mean) / std`` per channel. The runtime cycles the three ImageNet values over extra channels.
            "normalization": {
                "scale": 255.0,
                "mean": [IMAGENET_MEAN[index % 3] for index in range(channels)],
                "std": [IMAGENET_STD[index % 3] for index in range(channels)],
            },
            # ``RFDETR.predict`` stretches the whole image to height x width: no letterbox and no padding, so boxes come
            # out normalized to the image and are scaled by its original width and height.
            "resize": {
                "interpolation": "bilinear",
                "half_pixel_centers": True,
                "antialias": False,
                "aspect_ratio": "stretch",
            },
        },
        "outputs": [{"name": name, "dtype": _IO_DTYPE} for name in graph.output_names],
        "batch": batch,
        "build": {
            "precision": precision,
            "opset": int(config.opset_version),
            "tensorrt_version": tensorrt_version,
            "gpu": gpu,
        },
        "notes": None if config.notes is None else serialize_notes(config.notes),
    }


def write_engine_metadata(engine_path: str | os.PathLike[str], document: dict[str, Any]) -> Path:
    """Write *document* as the sidecar of *engine_path*, replacing any previous one atomically.

    The file is written to a temporary name in the same directory and swapped in with :func:`os.replace`, so a reader
    never sees a half-written sidecar and a failure leaves the previous one untouched. Its permission bits and group
    follow the engine's: ``mkstemp`` would otherwise make it owner-only in the user's own group, unreadable to a service
    that reads the engine as another user or through the engine's group. Where that cannot be done (a filesystem that
    rejects the change, such as exFAT or some SMB shares, or a group the user is not a member of), the file keeps the
    permissions it has: the description is worth more than its permissions.

    Args:
        engine_path: Path of the ``.trt`` engine.
        document: The document to write.

    Returns:
        The sidecar's path.

    Raises:
        ValueError: If *document* holds ``NaN`` or ``Infinity``.
        TypeError: If *document* holds a value JSON cannot encode.
        OSError: If the file cannot be written; the previous sidecar, if any, is kept.
    """
    target = sidecar_path(engine_path)
    # Keys stay in the document's order, which starts with ``schema_version``.
    text = json.dumps(document, indent=2, allow_nan=False)
    # A short fixed prefix: the temporary name is not derived from the engine's, so it fits wherever the sidecar's does.
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".rfdetr-description-", suffix=".tmp")
    try:
        # ``newline="\n"`` keeps the bytes the same on every platform.
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text + "\n")
        try:
            shutil.copymode(engine_path, temporary)
            if hasattr(os, "chown"):  # POSIX only
                os.chown(temporary, -1, os.stat(engine_path).st_gid)
        except OSError as error:
            logger.debug(f"Could not give {target} the engine's permissions: {error}")
        os.replace(temporary, target)
    # ``BaseException``, not ``Exception``: an interrupt in the middle of the write must not leave the temporary file.
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temporary).unlink(missing_ok=True)
        raise
    return target
