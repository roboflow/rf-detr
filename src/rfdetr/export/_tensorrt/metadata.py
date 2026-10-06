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

The engine and its sidecar are two files, replaced one after the other, so a reader can find an engine beside the
description of another build: during an export, or after two exports of the same name ran at once. The sidecar records
the engine file's size and SHA-256 so that a consumer can tell.

Schema version 1 does not record class names or the background class slot: how a logit slot maps to a class name
depends on the checkpoint and is decided inside ``RFDETR.predict``, not by anything this package can read.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from rfdetr.export._runtime.preprocess import IMAGENET_MEAN, IMAGENET_STD
from rfdetr.export.base import serialize_notes
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.files import _mkstemp_default_mode
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

#: Largest file :func:`is_engine_description` reads. A description is a few kilobytes; a bigger file with the same name
#: is not loaded into memory just to learn that it is not one.
_MAX_DESCRIPTION_BYTES: Final[int] = 1 << 20

#: Keys every description :func:`build_engine_metadata` produces carries, whatever its schema version. Another tool's
#: versioned JSON may well have a ``schema_version``; it is the three together that mark a file this exporter wrote.
_OWNERSHIP_KEYS: Final[frozenset[str]] = frozenset({"schema_version", "rfdetr_version", "engine"})


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


def _read_json_object(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Read *path* as a JSON object, answering ``None`` rather than raising whatever sits there.

    The checks that call this run on an export, one of them after a build that can take minutes, so nothing at *path*
    may crash or block them: only a regular file of at most 1 MiB is opened, which also keeps a named pipe from blocking
    the read.

    Args:
        path: The file to read.

    Returns:
        The parsed object, or ``None`` for a file that cannot be read, is not a regular file, is larger than 1 MiB, is
        not UTF-8 JSON, holds JSON nested too deeply to parse, or holds JSON that is not an object.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     document, array = Path(folder, "a.json"), Path(folder, "b.json")
        ...     _ = document.write_text('{"schema_version": 1}')
        ...     _ = array.write_text("[1]")
        ...     _read_json_object(document), _read_json_object(array), _read_json_object(folder)
        ({'schema_version': 1}, None, None)
    """
    try:
        status = os.stat(path)
        if not stat.S_ISREG(status.st_mode) or status.st_size > _MAX_DESCRIPTION_BYTES:
            return None
        with open(path, "rb") as stream:
            # Bounded again: the file may have grown since the ``stat``.
            content = stream.read(_MAX_DESCRIPTION_BYTES + 1)
        if len(content) > _MAX_DESCRIPTION_BYTES:
            return None
        document = json.loads(content.decode("utf-8"))
    # ``UnicodeDecodeError`` is a ``ValueError``; ``RecursionError`` is what ``json`` raises on deeply nested arrays.
    except (OSError, ValueError, RecursionError):
        return None
    return document if isinstance(document, dict) else None


def is_engine_description(path: str | os.PathLike[str]) -> bool:
    """Whether *path* looks like an engine description: a file holding a JSON object with ``schema_version``.

    Used before a warning or an error claims that a ``.json`` beside an engine describes an earlier engine, so a
    directory or an unrelated JSON file with the same name is never mistaken for one.

    It runs after a build that can take minutes, on a default export too, so whatever sits at *path* is answered rather
    than raised: only a regular file of at most 1 MiB is opened, which also keeps a named pipe from blocking the export.

    Args:
        path: The file to look at.

    Returns:
        ``True`` for a description, ``False`` for anything else: a file that cannot be read, is not a regular file, is
        larger than 1 MiB, or holds JSON nested too deeply to parse.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     description, other = Path(folder, "a.json"), Path(folder, "b.json")
        ...     _ = description.write_text('{"schema_version": 1}')
        ...     _ = other.write_text('{"name": "not ours"}')
        ...     is_engine_description(description), is_engine_description(other), is_engine_description(folder)
        (True, False, False)
    """
    document = _read_json_object(path)
    return document is not None and "schema_version" in document


def is_rfdetr_description(path: str | os.PathLike[str]) -> bool:
    """Whether *path* is a description this exporter wrote, and so one a new description may replace.

    Stricter than :func:`is_engine_description`, which only has to keep a warning from naming a file that is not a
    description: here the answer decides whether a file is overwritten, and ``.json`` is a generic extension, so another
    tool's versioned JSON with a ``schema_version`` must not pass. A file passes when it holds a JSON object carrying
    ``schema_version``, ``rfdetr_version`` and ``engine``, the keys every description has.

    Args:
        path: The file to look at.

    Returns:
        ``True`` for a description this exporter wrote, ``False`` for anything else, read as
        :func:`is_engine_description` reads it.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     ours, theirs = Path(folder, "a.json"), Path(folder, "b.json")
        ...     _ = ours.write_text('{"schema_version": 1, "rfdetr_version": "1.0.0", "engine": {}}')
        ...     _ = theirs.write_text('{"schema_version": 1}')
        ...     is_rfdetr_description(ours), is_rfdetr_description(theirs)
        (True, False)
    """
    document = _read_json_object(path)
    return document is not None and _OWNERSHIP_KEYS <= document.keys()


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


def serialized_engine_facts(serialized: Any) -> dict[str, Any]:
    """Identify a serialized engine by its size and the SHA-256 of its bytes: what its ``.trt`` file must hold.

    Args:
        serialized: The engine's bytes, as ``ICudaEngine.serialize()`` returns them or any other buffer.

    Returns:
        ``{"size", "sha256"}``: the size in bytes and the hex digest.

    Examples:
        >>> serialized_engine_facts(b"engine")
        {'size': 6, 'sha256': 'ed9f6f25068608efd412958da4dfc19328ca3511251fa6d5f9c42baf230e32f8'}
    """
    buffer = memoryview(serialized)
    return {"size": buffer.nbytes, "sha256": hashlib.sha256(buffer).hexdigest()}


def build_engine_metadata(
    config: TensorRTConfig,
    graph: ExportGraph,
    *,
    engine: dict[str, Any],
    precision: str,
    tensorrt_version: str,
    gpu: dict[str, str] | None,
) -> dict[str, Any]:
    """Assemble the schema version 1 document for one engine.

    Args:
        config: The exporter configuration the engine was built from.
        graph: The prepared graph the engine was built from.
        engine: The engine's size and digest, from :func:`serialized_engine_facts`.
        precision: The precision the engine was actually built with (``"fp16"``, ``"fp32"`` or ``"int8"``), which can
            differ from the request on a lean TensorRT wheel.
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
        ...     TensorRTConfig(), graph, engine={"size": 6, "sha256": "0" * 64}, precision="fp16",
        ...     tensorrt_version="11.3.0.99", gpu=None,
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
        "engine": engine,
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


def _follow_engine_permissions(descriptor: int, engine_path: str | os.PathLike[str], target: Path) -> None:
    """Give the open, still empty sidecar file the engine's permission bits and group.

    Called before anything is written: the file is created with the mode the umask gives a new file, which can be wider
    than an owner-only engine's, so the description is never on disk readable by more users than the engine. Where the
    change is refused (a filesystem such as exFAT or some SMB shares, or a group the user is not a member of), the file
    keeps the mode it has and a warning names it, since a service that reads the engine as another user or through its
    group may then be unable to read the description. Skipped on Windows, where the permission bits are only a
    read-only flag.

    Args:
        descriptor: The open file descriptor of the temporary sidecar file.
        engine_path: Path of the ``.trt`` engine whose mode and group are copied.
        target: The sidecar's final path, named in the warning.

    Examples:
        >>> import tempfile
        >>> from rfdetr.utilities.files import _mkstemp_default_mode
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     engine = Path(folder, "model.trt")
        ...     _ = engine.write_bytes(b"engine")
        ...     engine.chmod(0o640)
        ...     descriptor, temporary = _mkstemp_default_mode(folder)
        ...     _follow_engine_permissions(descriptor, engine, Path(folder, "model.json"))
        ...     os.close(descriptor)
        ...     os.name == "nt" or stat.S_IMODE(os.stat(temporary).st_mode) == 0o640
        True
    """
    if os.name == "nt":
        return
    try:
        engine = os.stat(engine_path)
        os.fchmod(descriptor, stat.S_IMODE(engine.st_mode))
        os.fchown(descriptor, -1, engine.st_gid)
    except OSError as error:
        mode = stat.S_IMODE(os.fstat(descriptor).st_mode)
        logger.warning(
            f"Could not give {target} the permissions and group of {engine_path} ({error}); it is written with mode "
            f"{mode:#o}, which a service that reads the engine as another user or through its group may not be able "
            "to read."
        )


def write_engine_metadata(engine_path: str | os.PathLike[str], document: dict[str, Any]) -> Path:
    """Write *document* as the sidecar of *engine_path*, replacing any previous one atomically.

    The file is written to a temporary name in the same directory and swapped in with :func:`os.replace`, so a reader
    never sees a half-written sidecar and a failure leaves the previous one untouched. On POSIX its permission bits and
    group follow the engine's, set while the file is still empty, so a service that reads the engine as another user or
    through the engine's group can read the description too, and nobody else can read it for even a moment. Where that
    cannot be done (a filesystem that rejects the change, such as exFAT or some SMB shares, or a group the user is not a
    member of), the file keeps the mode the umask gives a new file and a warning names it: the description is worth
    more than its permissions.

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
    handle, temporary = _mkstemp_default_mode(target.parent, prefix=".rfdetr-description-", suffix=".tmp")
    try:
        # ``newline="\n"`` keeps the bytes the same on every platform.
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            # Before the first write, which is buffered until the file is closed: the content never sits on disk under
            # a mode wider than the engine's.
            _follow_engine_permissions(stream.fileno(), engine_path, target)
            stream.write(text + "\n")
        os.replace(temporary, target)
    # ``BaseException``, not ``Exception``: an interrupt in the middle of the write must not leave the temporary file.
    except BaseException:
        with contextlib.suppress(OSError):
            Path(temporary).unlink(missing_ok=True)
        raise
    return target
