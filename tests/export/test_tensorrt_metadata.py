# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``<engine>.json`` sidecar that ``RFDETR.export(format="tensorrt", trt_metadata=True)`` writes.

The document is built by a pure function from the exporter configuration, the prepared graph and a few facts about the
build, so most of it is tested without TensorRT. The class that drives ``TensorRTExporter._convert`` stubs the
polygraphy chain the same way ``test_tensorrt_export.py`` does. The end-to-end check against a real engine lives in
``TestTensorRTEndToEnd`` there.
"""

from __future__ import annotations

import errno
import hashlib
import itertools
import json
import os
import stat
import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.detr import RFDETR
from rfdetr.export._runtime.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_to_nchw
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import metadata as tensorrt_metadata
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter, _BuiltEngine
from rfdetr.export._tensorrt.metadata import (
    METADATA_SCHEMA_VERSION,
    build_engine_metadata,
    gpu_facts,
    is_engine_description,
    is_rfdetr_description,
    serialized_engine_facts,
    sidecar_path,
    write_engine_metadata,
)
from rfdetr.export.base import serialize_notes
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.package import get_version

#: The exact top-level keys of schema version 1. A test that asserts this set fails when a key is added or dropped
#: without the schema version and the docs following.
_TOP_LEVEL_KEYS = {
    "schema_version",
    "rfdetr_version",
    "variant",
    "backbone_only",
    "engine",
    "input",
    "outputs",
    "batch",
    "build",
    "notes",
}


def _graph(
    *,
    output_names: tuple[str, ...] = ("dets", "labels"),
    channels: int = 3,
    batch: int = 1,
    shape: tuple[int, int] = (8, 12),
    backbone_only: bool = False,
    dynamic: bool = False,
) -> ExportGraph:
    """Build a small ``ExportGraph`` without a real RF-DETR model.

    Args:
        output_names: Names of the graph's outputs.
        channels: Number of input channels.
        batch: Batch size the example input carries.
        shape: ``(height, width)`` of the example input.
        backbone_only: Whether the graph is a backbone-only export.
        dynamic: Whether the batch axis is dynamic.

    Returns:
        The graph.

    Examples:
        >>> graph = _graph(shape=(8, 12), batch=2)
        >>> tuple(graph.input_tensors.shape), graph.output_names
        ((2, 3, 8, 12), ('dets', 'labels'))
    """
    axes = {name: {0: "batch"} for name in ("input", *output_names)} if dynamic else None
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(batch, channels, *shape),
        input_names=("input",),
        output_names=output_names,
        dynamic_axes=axes,
        shape=shape,
        backbone_only=backbone_only,
    )


def _metadata(config: TensorRTConfig | None = None, graph: ExportGraph | None = None, **facts: object) -> dict:
    """Build the sidecar document with fixed build facts, so a test overrides only the one it is about.

    Args:
        config: The exporter configuration, or ``None`` for the defaults.
        graph: The prepared graph, or ``None`` for a small detection graph.
        **facts: Overrides for ``engine``, ``precision``, ``tensorrt_version`` and ``gpu``.

    Returns:
        The document.

    Examples:
        >>> _metadata()["build"]["precision"], _metadata(precision="fp32")["build"]["precision"]
        ('fp16', 'fp32')
    """
    build = {
        "engine": {"size": 6, "sha256": hashlib.sha256(b"engine").hexdigest()},
        "precision": "fp16",
        "tensorrt_version": "11.3.0.99",
        "gpu": {"name": "GPU", "compute_capability": "12.0"},
    }
    build.update(facts)
    return build_engine_metadata(config or TensorRTConfig(), graph or _graph(), **build)


def _raise_os_error(*_args: object, **_kwargs: object) -> None:
    """Stand in for a filesystem call that fails, whatever it is called with.

    Raises:
        OSError: Always, with the message ``disk full``.

    Examples:
        >>> _raise_os_error("anything")
        Traceback (most recent call last):
        ...
        OSError: disk full
    """
    raise OSError("disk full")


#: A description an earlier export of the same name left behind. It carries the keys every description has, so both
#: the stale-description warning (``schema_version``) and the write's ownership check (``rfdetr_version`` and
#: ``engine`` too) recognize it.
_EARLIER_DESCRIPTION = (
    '{"schema_version": 1, "rfdetr_version": "1.0.0", "engine": {"size": 1, "sha256": "0"}, "batch": {"dynamic": true}}'
)

#: A JSON file of the user's own that happens to have an engine's name.
_FOREIGN_JSON = '{"labels": ["cat", "dog"]}'


def _umask() -> int:
    """Read the process umask, which can only be read by setting it, so it is set straight back.

    Returns:
        The umask.

    Examples:
        >>> 0 <= _umask() <= 0o777
        True
    """
    current = os.umask(0)
    os.umask(current)
    return current


def _padded_description(size: int) -> bytes:
    """Build a description of exactly *size* bytes, padded with a long ``notes`` string.

    Args:
        size: The length of the result in bytes.

    Returns:
        UTF-8 JSON holding a ``schema_version``.

    Examples:
        >>> content = _padded_description(64)
        >>> len(content), json.loads(content)["schema_version"]
        (64, 1)
    """
    head, tail = b'{"schema_version": 1, "notes": "', b'"}'
    return head + b"x" * (size - len(head) - len(tail)) + tail


def _raise_keyboard_interrupt(*_args: object) -> None:
    """Stand in for a call interrupted by Ctrl+C, whatever it is called with.

    Raises:
        KeyboardInterrupt: Always.

    Examples:
        A ``KeyboardInterrupt`` is not an ``Exception``, so doctest cannot expect it; the example catches it:

        >>> try:
        ...     _raise_keyboard_interrupt("anything")
        ... except KeyboardInterrupt:
        ...     print("interrupted")
        interrupted
    """
    raise KeyboardInterrupt


def _fail_the_build(*_args: object, **_kwargs: object) -> None:
    """Stand in for an engine build that fails, whatever it is called with.

    Raises:
        RuntimeError: Always, with the message ``build failed``.

    Examples:
        >>> _fail_the_build("network", config="config")
        Traceback (most recent call last):
        ...
        RuntimeError: build failed
    """
    raise RuntimeError("build failed")


class TestBuildEngineMetadata:
    """The document says what the engine expects and what it was built with, and nothing else."""

    @pytest.mark.parametrize(
        ("path", "keys"),
        [
            pytest.param((), _TOP_LEVEL_KEYS, id="top-level"),
            pytest.param(("engine",), {"size", "sha256"}, id="engine"),
            pytest.param(
                ("input",),
                {"name", "layout", "dtype", "height", "width", "channels", "channel_order", "normalization", "resize"},
                id="input",
            ),
            pytest.param(("input", "normalization"), {"scale", "mean", "std"}, id="normalization"),
            pytest.param(
                ("input", "resize"), {"interpolation", "half_pixel_centers", "antialias", "aspect_ratio"}, id="resize"
            ),
            pytest.param(("build",), {"precision", "opset", "tensorrt_version", "gpu"}, id="build"),
        ],
    )
    def test_each_section_has_exactly_the_documented_keys(self, path: tuple[str, ...], keys: set[str]) -> None:
        """Exact sets, not "contains": a key added or dropped without the schema following must fail."""
        section = _metadata(TensorRTConfig(dynamic_batch=True, max_batch_size=4))
        for step in path:
            section = section[step]

        assert set(section) == keys

    def test_the_engine_file_is_recorded_as_given(self) -> None:
        """The size and digest the build read back reach the document unchanged."""
        engine = {"size": 12, "sha256": "0" * 64}

        assert _metadata(engine=engine)["engine"] == engine

    def test_the_schema_version_is_one(self) -> None:
        """Version 1 is what the docs describe; a change to the layout that is not additive must bump it."""
        assert _metadata()["schema_version"] == METADATA_SCHEMA_VERSION == 1

    def test_the_input_describes_the_example_the_graph_was_traced_with(self) -> None:
        """Name, layout, dtype and the spatial size come from the graph, not from the model config."""
        document = _metadata(graph=_graph(shape=(8, 12), channels=3))["input"]

        assert (document["name"], document["layout"], document["dtype"]) == ("input", "NCHW", "float32")
        assert (document["height"], document["width"], document["channels"]) == (8, 12, 3)

    @pytest.mark.parametrize(("channels", "order"), [(3, "RGB"), (1, None), (4, None)])
    def test_the_channel_order_is_known_only_for_three_channels(self, channels: int, order: str | None) -> None:
        """A C++ consumer built on OpenCV reads BGR by default, so the file has to say RGB; it says nothing
        otherwise."""
        assert _metadata(graph=_graph(channels=channels))["input"]["channel_order"] == order

    def test_the_resize_convention_is_the_one_predict_uses(self) -> None:
        """``predict`` stretches the image to height x width, bilinear with half-pixel centres and no antialiasing."""
        resize = _metadata()["input"]["resize"]

        assert resize == {
            "interpolation": "bilinear",
            "half_pixel_centers": True,
            "antialias": False,
            "aspect_ratio": "stretch",
        }

    def test_the_normalization_is_the_one_predict_applies(self) -> None:
        """``RFDETR.predict`` normalizes with ``RFDETR.means`` and ``RFDETR.stds``, a copy of the export runtime's."""
        normalization = _metadata(graph=_graph(channels=3))["input"]["normalization"]

        assert normalization == {"scale": 255.0, "mean": RFDETR.means, "std": RFDETR.stds}

    @pytest.mark.parametrize("channels", [1, 3])
    def test_the_normalization_is_what_the_runtime_preprocess_applies(self, channels: int) -> None:
        """A solid-colour image run through ``preprocess_to_nchw`` equals ``(pixel / scale - mean) / std``.

        This pins the pixel scale and the channel handling to the runtime helper that consumers of the export use. The
        helper handles one and three channels only, so those are what it can cover.
        """
        value = 90
        image = Image.new("RGB" if channels != 1 else "L", (6, 6), (value,) * (3 if channels != 1 else 1))
        normalization = _metadata(graph=_graph(channels=channels))["input"]["normalization"]

        expected = (value / normalization["scale"] - np.array(normalization["mean"])) / np.array(normalization["std"])

        got = preprocess_to_nchw(image, height=6, width=6, channels=channels)[0, :, 0, 0]
        assert len(normalization["mean"]) == len(normalization["std"]) == channels
        np.testing.assert_allclose(got, expected[: got.shape[0]], atol=1e-6)

    def test_the_normalization_cycles_the_imagenet_values_over_extra_channels(self) -> None:
        """A model with more than three channels repeats the three values, as ``RFDETR.__init__`` does."""
        normalization = _metadata(graph=_graph(channels=4))["input"]["normalization"]

        assert normalization["mean"] == [*IMAGENET_MEAN, IMAGENET_MEAN[0]]
        assert normalization["std"] == [*IMAGENET_STD, IMAGENET_STD[0]]

    @pytest.mark.parametrize(
        ("output_names", "backbone_only"),
        [
            pytest.param(("dets", "labels"), False, id="detection"),
            pytest.param(("dets", "labels", "masks"), False, id="segmentation"),
            pytest.param(("dets", "labels", "keypoints"), False, id="keypoints"),
            pytest.param(("features", "features_1"), True, id="backbone"),
        ],
    )
    def test_the_outputs_are_the_graphs_in_order(self, output_names: tuple[str, ...], backbone_only: bool) -> None:
        """Every head lists its own outputs, in the order the engine returns them, all FP32."""
        document = _metadata(graph=_graph(output_names=output_names, backbone_only=backbone_only))

        assert document["outputs"] == [{"name": name, "dtype": "float32"} for name in output_names]
        assert document["backbone_only"] is backbone_only

    def test_a_static_engine_records_the_batch_it_was_built_for(self) -> None:
        """A static engine accepts exactly the traced batch size."""
        document = _metadata(TensorRTConfig(), _graph(batch=3))

        assert document["batch"] == {"dynamic": False, "size": 3}

    def test_a_dynamic_engine_records_its_optimization_profile(self) -> None:
        """A dynamic engine accepts batches from one to its maximum and is tuned for ``opt``, as plain integers.

        The traced batch differs from ``opt`` here, so the profile cannot be read off the graph.
        """
        config = TensorRTConfig(dynamic_batch=True, opt_batch_size=2, max_batch_size=8)

        document = _metadata(config, _graph(batch=4, dynamic=True))

        assert document["batch"] == {"dynamic": True, "min": 1, "opt": 2, "max": 8}

    @pytest.mark.parametrize("precision", ["fp16", "fp32"])
    def test_the_build_records_the_precision_it_was_given(self, precision: str) -> None:
        """The precision is an input (what the build actually did), never re-derived from the request."""
        document = _metadata(TensorRTConfig(fp16=True, opset_version=16), precision=precision)

        assert document["build"] == {
            "precision": precision,
            "opset": 16,
            "tensorrt_version": "11.3.0.99",
            "gpu": {"name": "GPU", "compute_capability": "12.0"},
        }

    def test_a_numpy_opset_is_written_as_a_plain_int(self) -> None:
        """A numpy integer is not JSON serializable and would fail after the engine has been built."""
        document = _metadata(TensorRTConfig(opset_version=np.int64(16)))

        assert json.loads(json.dumps(document))["build"]["opset"] == 16

    def test_the_variant_is_recorded(self) -> None:
        """The model variant the engine was exported from, such as ``rfdetr-nano``."""
        assert _metadata(TensorRTConfig(variant_name="rfdetr-nano"))["variant"] == "rfdetr-nano"

    def test_the_notes_are_stored_the_way_the_onnx_property_stores_them(self) -> None:
        """A string as is, anything else JSON-encoded, like the ONNX ``rfdetr_notes`` property."""
        notes = {"run": 3, "classes": ["box"]}

        assert _metadata(TensorRTConfig(notes=notes))["notes"] == serialize_notes(notes)

    def test_the_rfdetr_version_is_the_installed_one(self) -> None:
        """A consumer can tell which rfdetr release exported the engine."""
        assert _metadata()["rfdetr_version"] == get_version()

    def test_absent_variant_and_notes_are_null(self) -> None:
        """No variant name and no notes are ``null``, not empty strings, so a reader can tell them from ``""``."""
        document = _metadata()

        assert (document["variant"], document["notes"]) == (None, None)


class TestGpuFacts:
    """The building GPU is described by name and compute capability, or not at all without CUDA."""

    def test_the_current_cuda_device_is_described(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Name and ``major.minor`` compute capability of the device TensorRT builds on."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
        properties = {3: types.SimpleNamespace(name="Test GPU", major=8, minor=6)}
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties[index])

        assert gpu_facts() == {"name": "Test GPU", "compute_capability": "8.6"}

    def test_a_torch_without_cuda_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TensorRT can still build, but nothing here can name the GPU, so the field is ``None``."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert gpu_facts() is None


class TestSerializedEngineFacts:
    """The size and SHA-256 of a serialized engine, which a consumer compares with the ``.trt`` it loads."""

    @pytest.mark.parametrize(
        "serialized",
        [
            pytest.param(b"", id="empty"),
            pytest.param(b"engine", id="bytes"),
            pytest.param(memoryview(b"engine!!").cast("I"), id="buffer of 4-byte items"),
        ],
    )
    def test_the_size_and_digest_are_those_of_the_bytes(self, serialized: bytes | memoryview) -> None:
        """The size counts bytes, not buffer items, so it is the size of the file those bytes are saved to."""
        content = bytes(serialized)

        assert serialized_engine_facts(serialized) == {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }


class TestIsEngineDescription:
    """Looking at the ``.json`` beside an engine never crashes or blocks the export, whatever sits there."""

    def test_json_nested_too_deeply_to_parse_is_not_a_description(self, tmp_path: Path) -> None:
        """``json.loads`` raises ``RecursionError`` on deep nesting, which is not an ``OSError`` or a ``ValueError``.

        The check runs right after a build that can take minutes, on a default export too, so a hostile file of the
        engine's name must come out as "not a description" rather than as a crash.
        """
        path = tmp_path / "model.json"
        path.write_text("[" * 200_000)

        assert is_engine_description(path) is False

    @pytest.mark.parametrize(
        "make",
        [
            pytest.param(os.mkdir, id="directory"),
            pytest.param(
                getattr(os, "mkfifo", None),
                id="named pipe",
                marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are POSIX-only"),
            ),
        ],
    )
    def test_a_path_that_is_not_a_regular_file_is_not_read(self, tmp_path: Path, make: object) -> None:
        """Opening a named pipe for reading blocks until a writer appears, so only a regular file is opened.

        A directory with the engine's name is not a description either, and must not be mistaken for one.
        """
        path = tmp_path / "model.json"
        make(path)

        assert is_engine_description(path) is False

    @pytest.mark.parametrize(("size", "expected"), [(1 << 20, True), ((1 << 20) + 1, False)])
    def test_a_file_is_read_only_up_to_one_mebibyte(self, tmp_path: Path, size: int, expected: bool) -> None:
        """A description is a few kilobytes; a larger file of the same name is not loaded just to learn it is not one.

        The pair pins the boundary: a description of exactly 1 MiB is still recognized, one byte more is not read.
        """
        path = tmp_path / "model.json"
        path.write_bytes(_padded_description(size))

        assert is_engine_description(path) is expected


class TestIsRfdetrDescription:
    """Only a file carrying the keys every description this exporter writes has may be replaced by a new one."""

    def test_a_description_this_exporter_wrote_is_recognized(self, tmp_path: Path) -> None:
        """What :func:`write_engine_metadata` writes passes the check, so a re-export replaces its own description."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")

        assert is_rfdetr_description(write_engine_metadata(engine, _metadata())) is True

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param('{"schema_version": 1}', id="schema_version only"),
            pytest.param('{"schema_version": 1, "rfdetr_version": "1.0.0"}', id="no engine"),
            pytest.param('{"schema_version": 1, "engine": {}}', id="no rfdetr_version"),
            pytest.param('[{"schema_version": 1, "rfdetr_version": "1.0.0", "engine": {}}]', id="not an object"),
            pytest.param("[" * 200_000, id="nested too deeply"),
        ],
    )
    def test_a_file_without_all_the_keys_is_not_ours(self, tmp_path: Path, content: str) -> None:
        """A versioned JSON document of another tool carries ``schema_version`` too, so that key alone is not enough.

        Each case misses one of ``schema_version``, ``rfdetr_version`` and ``engine``, or is not a JSON object at all.
        """
        path = tmp_path / "model.json"
        path.write_text(content)

        assert is_rfdetr_description(path) is False

    def test_a_directory_is_not_ours(self, tmp_path: Path) -> None:
        """A folder with the description's name is not something an export wrote."""
        assert is_rfdetr_description(tmp_path) is False


class TestWriteEngineMetadata:
    """The sidecar is written next to the engine, atomically, readable by whoever can read the engine."""

    def test_the_sidecar_sits_beside_the_engine_with_a_json_suffix(self, tmp_path: Path) -> None:
        """``<name>_fp16.trt`` gets ``<name>_fp16.json``."""
        engine = tmp_path / "rfdetr-nano_fp16.trt"

        assert sidecar_path(engine) == tmp_path / "rfdetr-nano_fp16.json"

    def test_the_document_round_trips(self, tmp_path: Path) -> None:
        """What is written is what a reader parses."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        document = _metadata()

        path = write_engine_metadata(engine, document)

        assert path == tmp_path / "model.json"
        assert json.loads(path.read_text()) == document

    def test_the_file_starts_with_the_schema_version(self, tmp_path: Path) -> None:
        """A reader that sniffs the first key finds the version there, so keys are written in the document's order."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")

        path = write_engine_metadata(engine, _metadata())

        assert next(iter(json.loads(path.read_text()))) == "schema_version"

    def test_a_previous_sidecar_is_replaced(self, tmp_path: Path) -> None:
        """A rebuild of the same engine replaces the description of the old one."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        (tmp_path / "model.json").write_text('{"old": true}')

        write_engine_metadata(engine, _metadata())

        assert "old" not in json.loads((tmp_path / "model.json").read_text())

    def test_a_failed_write_keeps_the_previous_sidecar_and_leaves_no_stray_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Torn-write test: the swap fails, the old file is intact, and no temporary file is left behind."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        previous = tmp_path / "model.json"
        previous.write_text('{"previous": true}')
        monkeypatch.setattr(os, "replace", _raise_os_error)

        with pytest.raises(OSError, match="disk full"):
            write_engine_metadata(engine, _metadata())

        assert previous.read_text() == '{"previous": true}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.trt"]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_the_sidecar_takes_the_engines_permission_bits(self, tmp_path: Path) -> None:
        """``mkstemp`` creates files owner-only; a consumer running as another user must still be able to read it."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        engine.chmod(0o640)

        path = write_engine_metadata(engine, _metadata())

        assert stat.S_IMODE(path.stat().st_mode) == 0o640

    @pytest.mark.skipif(os.name == "nt", reason="POSIX ownership")
    def test_the_sidecar_takes_the_engines_group(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A service that reads the engine through its group must be able to read the description the same way."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        calls: list[tuple[int, int]] = []
        monkeypatch.setattr(os, "fchown", lambda descriptor, uid, gid: calls.append((uid, gid)))

        write_engine_metadata(engine, _metadata())

        assert calls == [(-1, engine.stat().st_gid)]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_the_engines_mode_is_set_before_any_content_is_written(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The temporary file is created with the umask's mode, which can be wider than an owner-only engine's.

        Its mode is changed while it is still empty, so the description is never on disk readable by more users than the
        engine, not even for the moment between the write and the change.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        engine.chmod(0o600)
        sizes: list[int] = []
        real_fchmod = os.fchmod
        monkeypatch.setattr(
            os,
            "fchmod",
            lambda descriptor, mode: (sizes.append(os.fstat(descriptor).st_size), real_fchmod(descriptor, mode))[1],
        )

        path = write_engine_metadata(engine, _metadata())

        assert sizes == [0]
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("refused", ["fchmod", "fchown"])
    def test_a_filesystem_that_refuses_the_permission_change_still_gets_the_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, refused: str
    ) -> None:
        """ExFAT, some SMB shares, or a group the user is not in; the description is worth more than its permissions."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        monkeypatch.setattr(os, refused, _raise_os_error)

        path = write_engine_metadata(engine, _metadata())

        assert json.loads(path.read_text())["schema_version"] == 1

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    def test_a_refused_permission_change_leaves_a_readable_mode_and_warns_with_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without the engine's mode the file keeps the umask's mode, as ``open`` gives a new file, not owner-only.

        A service reading the engine as another user may still be unable to read it, so the warning names the mode the
        file was written with, which a debug-level message would hide.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        engine.chmod(0o640)
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_metadata.logger, "warning", lambda message, *args: logged.append(message % args))
        monkeypatch.setattr(os, "fchmod", _raise_os_error)

        path = write_engine_metadata(engine, _metadata())

        mode = 0o666 & ~_umask()
        assert stat.S_IMODE(path.stat().st_mode) == mode
        assert len(logged) == 1
        assert f"{mode:#o}" in logged[0]

    def test_the_temporary_file_is_written_in_the_sidecars_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``os.replace`` is atomic only within one filesystem, so the temporary file sits beside the sidecar."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        sources: list[Path] = []
        real_replace = os.replace
        monkeypatch.setattr(os, "replace", lambda src, dst: (sources.append(Path(src)), real_replace(src, dst))[1])

        write_engine_metadata(engine, _metadata())

        assert [source.parent for source in sources] == [tmp_path]

    def test_the_temporary_name_does_not_grow_with_the_engines_name(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A sidecar name that fits the filesystem must not be turned into a temporary name that does not."""
        names: list[int] = []
        real_replace = os.replace
        monkeypatch.setattr(
            os, "replace", lambda src, dst: (names.append(len(Path(src).name)), real_replace(src, dst))[1]
        )
        short, long = tmp_path / "m.trt", tmp_path / f"{'m' * 100}.trt"
        short.write_bytes(b"engine")
        long.write_bytes(b"engine")

        write_engine_metadata(short, _metadata())
        write_engine_metadata(long, _metadata())

        assert names[0] == names[1]

    def test_two_writers_in_one_directory_each_get_their_own_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Interleaved writers: a second export finishing in the middle of the first must not share its temporary
        file."""
        first, second = tmp_path / "a.trt", tmp_path / "b.trt"
        first.write_bytes(b"engine a")
        second.write_bytes(b"engine b")
        real_replace = os.replace
        interleaved: list[bool] = []

        def replace_after_the_other_writer(source: str, destination: str) -> None:
            if not interleaved:
                interleaved.append(True)
                write_engine_metadata(second, _metadata(precision="fp32"))
            real_replace(source, destination)

        monkeypatch.setattr(os, "replace", replace_after_the_other_writer)

        write_engine_metadata(first, _metadata(precision="fp16"))

        precisions = [json.loads((tmp_path / name).read_text())["build"]["precision"] for name in ("a.json", "b.json")]
        assert precisions == ["fp16", "fp32"]

    @pytest.mark.skipif(os.name == "nt", reason="the permission change this hooks is skipped on Windows")
    def test_two_writers_of_the_same_sidecar_leave_one_complete_document_and_no_temporary_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A second export of the same engine finishing in the middle of the first one's write must not corrupt it.

        The permission change runs after the first writer created its temporary file and before it writes or swaps
        anything, so hooking it runs the second writer to completion in that window. The first writer swaps last, so its
        document is what stays, whole, and neither writer leaves a temporary file behind.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        real_fchmod = os.fchmod
        interleaved: list[bool] = []

        def fchmod_after_the_other_writer(descriptor: int, mode: int) -> None:
            """Run the second writer to completion, the first time only, then make the real permission change."""
            if not interleaved:
                interleaved.append(True)
                write_engine_metadata(engine, _metadata(precision="fp32"))
            real_fchmod(descriptor, mode)

        monkeypatch.setattr(os, "fchmod", fchmod_after_the_other_writer)

        write_engine_metadata(engine, _metadata(precision="fp16"))

        assert json.loads((tmp_path / "model.json").read_text())["build"]["precision"] == "fp16"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.trt"]

    def test_the_file_has_the_same_line_endings_on_every_platform(self, tmp_path: Path) -> None:
        """Unix line endings on every platform, so a consumer that compares or hashes the file sees the same bytes."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")

        path = write_engine_metadata(engine, _metadata())

        assert b"\r\n" not in path.read_bytes()

    def test_an_interrupt_during_the_swap_leaves_the_previous_sidecar_and_no_stray_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A ``KeyboardInterrupt`` is not an ``Exception``, and must still clean up the temporary file."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        previous = tmp_path / "model.json"
        previous.write_text('{"previous": true}')

        monkeypatch.setattr(os, "replace", _raise_keyboard_interrupt)

        with pytest.raises(KeyboardInterrupt):
            write_engine_metadata(engine, _metadata())

        assert previous.read_text() == '{"previous": true}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.trt"]

    def test_a_non_finite_value_is_refused_and_leaves_the_previous_sidecar(self, tmp_path: Path) -> None:
        """``NaN`` is not JSON; a strict reader would reject the file, so nothing is written."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        previous = tmp_path / "model.json"
        previous.write_text('{"previous": true}')

        with pytest.raises(ValueError, match="Out of range float"):
            write_engine_metadata(engine, {"schema_version": 1, "value": float("nan")})

        assert previous.read_text() == '{"previous": true}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.trt"]

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_sidecar_that_is_a_symlink_to_a_file_is_replaced_not_followed(self, tmp_path: Path) -> None:
        """``os.replace`` swaps the link itself, so the file the link pointed at keeps its content.

        A user who links ``<engine>.json`` to a shared description must not find that description overwritten by an
        export: the sidecar becomes a regular file and the link's old target is left alone.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        linked_to = tmp_path / "linked-to.json"
        linked_to.write_text('{"shared": true}')
        sidecar = tmp_path / "model.json"
        sidecar.symlink_to(linked_to)

        write_engine_metadata(engine, _metadata())

        assert not sidecar.is_symlink()
        assert json.loads(sidecar.read_text())["schema_version"] == 1
        assert linked_to.read_text() == '{"shared": true}'

    @pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
    def test_a_sidecar_that_is_a_dangling_symlink_is_replaced_and_its_target_is_not_created(
        self, tmp_path: Path
    ) -> None:
        """A link to nothing is swapped for the regular file; the write does not create what the link named."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        missing = tmp_path / "missing.json"
        sidecar = tmp_path / "model.json"
        sidecar.symlink_to(missing)

        write_engine_metadata(engine, _metadata())

        assert not sidecar.is_symlink()
        assert json.loads(sidecar.read_text())["schema_version"] == 1
        assert not missing.exists()

    def test_a_failed_cleanup_does_not_mask_the_error_that_made_it_necessary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When removing the temporary file fails too, the caller still learns why the swap failed.

        The swap fails with ``disk full``; the cleanup that follows fails with ``busy``. The first error is the one the
        caller can act on, so it, not the cleanup's, is what propagates.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        monkeypatch.setattr(os, "replace", _raise_os_error)
        unlink = MagicMock(side_effect=OSError("busy"))
        monkeypatch.setattr(Path, "unlink", unlink)

        with pytest.raises(OSError, match="disk full"):
            write_engine_metadata(engine, _metadata())

        unlink.assert_called_once()

    @pytest.mark.skipif(os.name == "nt", reason="permissions are copied from the engine on POSIX only")
    def test_a_missing_engine_still_gets_its_description_and_a_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With no engine file to copy permissions from, the description is written anyway and the warning names it.

        This is the degrade path for a filesystem or a race that hides the engine: the description is worth more than
        its permissions.
        """
        engine = tmp_path / "model.trt"
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_metadata.logger, "warning", lambda message, *args: logged.append(message % args))

        path = write_engine_metadata(engine, _metadata())

        assert is_rfdetr_description(path) is True
        assert len(logged) == 1
        assert str(path) in logged[0]

    def test_a_directory_at_the_sidecar_path_is_refused_and_leaves_no_stray_file(self, tmp_path: Path) -> None:
        """A folder with the sidecar's name cannot be replaced by a file, so the write fails with an ``OSError``.

        The error type is ``IsADirectoryError`` on POSIX and ``PermissionError`` on Windows, so only ``OSError`` is
        asserted. The temporary file written before the failed swap must be removed.
        """
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        sidecar = tmp_path / "model.json"
        sidecar.mkdir()

        with pytest.raises(OSError):
            write_engine_metadata(engine, _metadata())

        assert sidecar.is_dir()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["model.json", "model.trt"]


def _patch_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, has_fp16_flag: bool, builds: tuple[bytes, ...] = (b"engine",)
) -> list[str]:
    """Stub the ONNX stage and the polygraphy chain so ``_convert`` runs without TensorRT, and record the engines.

    Each build yields a stand-in engine whose ``serialize()`` returns the next of *builds* (the last one repeats), and
    ``save_file`` really writes those bytes, so the sidecar has an engine to sit next to. The stand-in ``tensorrt``
    module reports a weakly typed version, with or without the FP16 builder flag (a lean wheel lacks it).

    Args:
        monkeypatch: Fixture used to replace the entry points on the module under test.
        tmp_path: Directory the ONNX and engine files live in.
        has_fp16_flag: Whether the stand-in ``tensorrt.BuilderFlag`` carries ``FP16``.
        builds: The serialized bytes of successive builds.

    Returns:
        The list the paths of written engines are appended to.

    Examples:
        >>> with pytest.MonkeyPatch.context() as monkeypatch:
        ...     engines = _patch_build(monkeypatch, Path("."), has_fp16_flag=False)
        ...     hasattr(sys.modules["tensorrt"].BuilderFlag, "FP16")
        False
        >>> engines
        []
    """
    engines: list[str] = []
    fake = types.ModuleType("tensorrt")
    fake.__version__ = "10.16.1.11"
    fake.BuilderFlag = types.SimpleNamespace(**({"FP16": 1} if has_fp16_flag else {}))

    class _Network:
        num_inputs = 1

        @staticmethod
        def get_input(index: int) -> types.SimpleNamespace:
            """Return the one input tensor the fake network declares, whatever the index."""
            return types.SimpleNamespace(name="input", shape=(1, 3, 8, 12))

    serialized = itertools.chain(builds, itertools.repeat(builds[-1]))

    def _save_file(contents: bytes, dest: str, description: str | None = None) -> None:
        Path(dest).write_bytes(contents)
        engines.append(dest)

    onnx_path = str(tmp_path / "m.onnx")
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda self, graph: onnx_path)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _Network(), "parser"))
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
    monkeypatch.setattr(
        tensorrt_export,
        "engine_from_network",
        lambda parsed, config: types.SimpleNamespace(serialize=lambda content=next(serialized): content),
    )
    monkeypatch.setattr(tensorrt_export, "save_file", _save_file)
    return engines


class TestExporterWritesMetadata:
    """``TensorRTExporter._convert`` writes the sidecar only when asked, next to the engine it just built."""

    def test_no_sidecar_is_written_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The default export produces exactly the files it always did."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph()))

        assert engine.is_file()
        assert not engine.with_suffix(".json").exists()

    def test_the_sidecar_is_written_beside_the_engine_when_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """One JSON per engine, with the engine's own stem."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        document = json.loads(engine.with_suffix(".json").read_text())
        assert document["build"]["precision"] == "fp32"
        assert document["build"]["tensorrt_version"] == "10.16.1.11"

    def test_the_description_identifies_the_engine_the_build_wrote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Its size and SHA-256 are those of the ``.trt`` beside it, so a consumer can check that the two belong
        together."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        assert json.loads(engine.with_suffix(".json").read_text())["engine"] == {
            "size": 6,
            "sha256": hashlib.sha256(b"engine").hexdigest(),
        }

    @pytest.mark.parametrize(
        ("owner", "name"),
        [
            pytest.param(tensorrt_export, "save_file", id="after the first engine is saved"),
            pytest.param(TensorRTExporter, "_build", id="after the first build returns"),
        ],
    )
    def test_a_description_written_after_another_export_replaced_the_engine_keeps_its_own_digest(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, owner: object, name: str
    ) -> None:
        """Two exports of one name, interleaved: the second builds and describes its engine in the middle of the first.

        The engine and its description are two files, and the engine is written in place, so the pair left on disk can
        come from different exports. The digest is taken from the bytes the build serialized, not from the
        file:
        read
        from the file, even right after the save, it would vouch for the other export's engine and a consumer could not
        detect the mismatch.
        """
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True, builds=(b"first engine", b"second engine"))
        second = TensorRTExporter(TensorRTConfig(fp16=False, metadata=True, output_name="shared"))
        real = getattr(owner, name)
        interleaved: list[bool] = []

        def then_the_other_export(*args: object, **kwargs: object) -> object:
            """Make the call, then, the first time only, run the second export to completion."""
            result = real(*args, **kwargs)
            if not interleaved:
                interleaved.append(True)
                second._convert(_graph())
            return result

        monkeypatch.setattr(owner, name, then_the_other_export)

        engine = Path(
            TensorRTExporter(TensorRTConfig(fp16=True, metadata=True, output_name="shared"))._convert(_graph())
        )

        recorded = json.loads(engine.with_suffix(".json").read_text())["engine"]["sha256"]
        assert recorded == hashlib.sha256(b"first engine").hexdigest()

    def test_the_default_export_does_not_hash_the_engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Hashing reads every engine byte again; an export that writes no description does not pay for it."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        reads: list[object] = []
        monkeypatch.setattr(tensorrt_export, "serialized_engine_facts", reads.append)

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert reads == []

    def test_build_engine_alone_does_not_hash_the_engine(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``build_engine`` on an ``.onnx`` never writes a description, so ``metadata`` does not make it hash either."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        reads: list[object] = []
        monkeypatch.setattr(tensorrt_export, "serialized_engine_facts", reads.append)

        with pytest.warns(UserWarning, match="has no effect"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True)).build_engine(str(tmp_path / "m.onnx"))

        assert reads == []

    def test_build_engine_alone_says_that_metadata_has_no_effect(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A caller who asked for a description and gets none is told so, the way other ignored settings are, also with
        no earlier description around."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)

        with pytest.warns(UserWarning, match="metadata=True has no effect"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True)).build_engine(str(tmp_path / "m.onnx"))

    def test_an_export_that_writes_the_description_does_not_call_metadata_ineffective(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The warning is for ``build_engine`` alone; the export that writes the description must not give it."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert [str(warning.message) for warning in caught if issubclass(warning.category, UserWarning)] == []

    def test_the_building_gpu_reaches_the_document(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The exporter asks ``gpu_facts`` and records its answer under ``build.gpu``."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        monkeypatch.setattr(tensorrt_export, "gpu_facts", lambda: {"name": "Test GPU", "compute_capability": "9.9"})

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        assert json.loads(engine.with_suffix(".json").read_text())["build"]["gpu"] == {
            "name": "Test GPU",
            "compute_capability": "9.9",
        }

    def test_writing_a_description_reports_nothing_about_an_old_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The new description replaces the old one, so there is nothing stale to warn about."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_EARLIER_DESCRIPTION)
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: logged.append(message % args))

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        assert logged == []
        assert json.loads(engine.with_suffix(".json").read_text())["schema_version"] == 1

    def test_the_precision_is_the_one_actually_built_after_the_lean_wheel_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An FP16 request on a TensorRT wheel without the FP16 flag builds FP32; the sidecar must say so."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=False)

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=True, metadata=True))._convert(_graph()))

        assert json.loads(engine.with_suffix(".json").read_text())["build"]["precision"] == "fp32"

    def test_the_precision_survives_a_custom_output_name(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """``output_name`` drops the ``_fp16``/``_fp32`` suffix from the file name, so the name cannot be the source."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        config = TensorRTConfig(fp16=True, metadata=True, output_name="mine")

        engine = Path(TensorRTExporter(config)._convert(_graph()))

        assert engine.name == "mine.trt"
        assert json.loads(engine.with_suffix(".json").read_text())["build"]["precision"] == "fp16"

    def test_a_backbone_only_export_names_engine_and_description_after_the_onnx_stem(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A custom ``output_name`` does not rename a backbone-only engine: both files follow the ONNX stem.

        The backbone ONNX stem carries the ``-backbone`` marker, so keeping it stops the engine from being
        indistinguishable from a full-detector one. Here the stubbed ONNX file is ``m.onnx``, so the engine is ``m.trt``
        (not ``custom.trt``) and its description ``m.json``, which says ``backbone_only`` is true.
        """
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        config = TensorRTConfig(fp16=False, metadata=True, output_name="custom")

        engine = Path(TensorRTExporter(config)._convert(_graph(output_names=("features",), backbone_only=True)))

        assert engine == tmp_path / "m.trt"
        assert sorted(p.name for p in tmp_path.glob("*.json")) == ["m.json"]
        assert json.loads((tmp_path / "m.json").read_text())["backbone_only"] is True

    def test_a_failed_build_leaves_no_sidecar(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No engine, no description of one."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        monkeypatch.setattr(tensorrt_export, "engine_from_network", _fail_the_build)

        with pytest.raises(RuntimeError, match="build failed"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert list(tmp_path.glob("*.json")) == []

    def test_a_description_that_cannot_be_written_names_the_engine_that_was_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The write happens after a build that can take minutes, so the error must say the engine exists."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        monkeypatch.setattr(os, "replace", _raise_os_error)

        with pytest.raises(OSError, match=r"engine was written to .*\.trt.*disk full"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert len(list(tmp_path.glob("*.trt"))) == 1

    def test_a_failed_write_says_when_an_older_description_is_left_beside_the_engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The failed write keeps the old file, which now sits beside a different engine; the error says so."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_EARLIER_DESCRIPTION)
        monkeypatch.setattr(os, "replace", _raise_os_error)

        with pytest.raises(OSError, match=r"earlier .*m_fp32\.json.*previous engine"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

    def test_a_failed_write_without_an_earlier_description_does_not_claim_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The hint about an earlier description is only given when there is one."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        monkeypatch.setattr(os, "replace", _raise_os_error)

        with pytest.raises(OSError) as refusal:
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert "earlier" not in str(refusal.value)

    @pytest.mark.parametrize(
        ("code", "kind"),
        [
            pytest.param(errno.EACCES, PermissionError, id="permission denied"),
            pytest.param(errno.ENOSPC, OSError, id="disk full"),
        ],
    )
    def test_a_failed_write_keeps_the_kind_and_code_of_the_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int, kind: type[OSError]
    ) -> None:
        """A caller that handles ``PermissionError`` or reads ``errno`` still recognizes why the write failed.

        The error is re-raised with a message saying the engine exists; that must not turn a refused permission into a
        bare ``OSError``. ENOSPC has no subclass of its own, so it stays an ``OSError`` that carries the code.
        """
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        failure = OSError(code, os.strerror(code), str(tmp_path / "m_fp32.json"))
        monkeypatch.setattr(os, "replace", MagicMock(side_effect=failure))

        with pytest.raises(OSError) as refusal:
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert (type(refusal.value), refusal.value.errno) == (kind, code)

    def test_a_failed_write_names_the_file_once(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The re-raised error prints the file name itself, so the message around it must not repeat it."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        failure = PermissionError(errno.EACCES, "Permission denied", str(tmp_path / "m_fp32.json"))
        monkeypatch.setattr(os, "replace", MagicMock(side_effect=failure))

        with pytest.raises(PermissionError) as refusal:
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert str(refusal.value).count("m_fp32.json") == 1

    def test_a_failed_write_is_chained_to_the_error_it_reports(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The filesystem's own error stays reachable as the cause, for a traceback or a caller that inspects it."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        failure = PermissionError(errno.EACCES, "Permission denied", str(tmp_path / "m_fp32.json"))
        monkeypatch.setattr(os, "replace", MagicMock(side_effect=failure))

        with pytest.raises(PermissionError) as refusal:
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert refusal.value.__cause__ is failure

    def test_a_failed_write_without_an_error_code_is_a_plain_os_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An ``OSError`` raised with a message alone has no errno to keep, so the re-raised one has none either."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        monkeypatch.setattr(os, "replace", _raise_os_error)

        with pytest.raises(OSError) as refusal:
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert (type(refusal.value), refusal.value.errno) == (OSError, None)

    def test_a_description_left_by_an_earlier_export_is_reported_when_metadata_is_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The new engine replaced one of the same name, so the old description no longer describes the file."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        stale = tmp_path / "m_fp32.json"
        stale.write_text(_EARLIER_DESCRIPTION)
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: logged.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert len(logged) == 1
        assert "m_fp32.json" in logged[0]
        assert stale.read_text() == _EARLIER_DESCRIPTION, "a file the export did not write is left alone"

    @pytest.mark.parametrize("metadata", [False, True])
    def test_a_rebuild_through_build_engine_reports_an_earlier_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, metadata: bool
    ) -> None:
        """``build_engine`` on an ``.onnx`` replaces the engine too, so it gives the same warning as ``export``.

        It never writes a description, so ``metadata`` does not silence the warning.
        """
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_EARLIER_DESCRIPTION)
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: logged.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False, metadata=metadata)).build_engine(str(tmp_path / "m.onnx"))

        assert sum("m_fp32.json" in warning for warning in logged) == 1

    @pytest.mark.parametrize("beside", [None, "unrelated json", "directory"])
    def test_only_a_description_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, beside: str | None
    ) -> None:
        """No file, a JSON file of the user's own, or a folder with that name is not something an export wrote."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        if beside == "unrelated json":
            (tmp_path / "m_fp32.json").write_text('{"name": "a file of the user\'s own"}')
        elif beside == "directory":
            (tmp_path / "m_fp32.json").mkdir()
        logged: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: logged.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert logged == []

    def test_a_build_asked_for_its_digest_reports_what_it_built(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The build hands back the engine's path, the precision it ended up with and the digest of its bytes.

        These are the facts the description records. They come back from the build itself rather than being left on the
        exporter, so neither a failed build nor a later one can pass off an earlier build's facts as its own. An FP16
        request on a wheel without the FP16 flag shows that the precision is the one built, not the one asked for.
        """
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=False)
        exporter = TensorRTExporter(TensorRTConfig(fp16=True, metadata=True))

        built = exporter._build(str(tmp_path / "m.onnx"), output_name=None, digest=True)

        assert built == _BuiltEngine(
            path=str(tmp_path / "m_fp32.trt"), fp16=False, engine_facts=serialized_engine_facts(b"engine")
        )

    @pytest.mark.parametrize(
        "place",
        [
            pytest.param(lambda path: path.write_text(_FOREIGN_JSON), id="a JSON file of the user's own"),
            pytest.param(lambda path: path.write_text('{"schema_version": 2}'), id="another tool's versioned JSON"),
            pytest.param(Path.mkdir, id="a directory"),
        ],
    )
    def test_a_file_the_export_did_not_write_is_refused_before_the_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, place: object
    ) -> None:
        """``.json`` is a generic extension, so ``<engine>.json`` can be a label map or a manifest of the user's own.

        The description would replace it without a word, so the export refuses, and does so before the build, which can
        take minutes, rather than after it.
        """
        engines = _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        place(tmp_path / "m_fp32.json")

        with pytest.raises(FileExistsError, match=r"m_fp32\.json.*(rename|remove).*output_name"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert engines == []

    def test_a_refused_file_is_left_as_it_was(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The refusal touches nothing: the user's file keeps its content."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        foreign = tmp_path / "m_fp32.json"
        foreign.write_text(_FOREIGN_JSON)

        with pytest.raises(FileExistsError):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph())

        assert foreign.read_text() == _FOREIGN_JSON

    def test_the_file_checked_is_the_one_a_custom_output_name_gives(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With ``output_name`` the description is ``<output_name>.json``, so that is the file the check looks at."""
        engines = _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "mine.json").write_text(_FOREIGN_JSON)

        with pytest.raises(FileExistsError, match=r"mine\.json"):
            TensorRTExporter(TensorRTConfig(fp16=False, metadata=True, output_name="mine"))._convert(_graph())

        assert engines == []

    def test_a_description_an_earlier_export_wrote_is_replaced(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Re-exporting under the same name replaces the earlier description, as it always did."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_EARLIER_DESCRIPTION)

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        assert json.loads(engine.with_suffix(".json").read_text())["batch"] == {"dynamic": False, "size": 1}

    def test_a_file_of_the_users_own_is_not_checked_when_no_description_is_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The default export writes no ``.json``, so a file of that name is no reason to refuse it."""
        engines = _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_FOREIGN_JSON)

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert len(engines) == 1

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_metadata_must_be_a_bool(self, value: object) -> None:
        """A truthy non-``bool`` would write a file by accident, so it is refused before any work on the model."""
        with pytest.raises(ValueError, match="trt_metadata"):
            TensorRTExporter(TensorRTConfig(metadata=value))

    def test_the_description_is_off_unless_the_keyword_is_given(self) -> None:
        """Without ``trt_metadata`` the configuration writes no description."""
        assert TensorRTExporter.build_config().metadata is False
