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

import json
import os
import shutil
import stat
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.detr import RFDETR
from rfdetr.export._runtime.preprocess import IMAGENET_MEAN, IMAGENET_STD, preprocess_to_nchw
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter
from rfdetr.export._tensorrt.metadata import (
    METADATA_SCHEMA_VERSION,
    build_engine_metadata,
    gpu_facts,
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
        **facts: Overrides for ``precision``, ``tensorrt_version`` and ``gpu``.

    Returns:
        The document.

    Examples:
        >>> _metadata()["build"]["precision"], _metadata(precision="fp32")["build"]["precision"]
        ('fp16', 'fp32')
    """
    build = {"precision": "fp16", "tensorrt_version": "11.3.0.99", "gpu": {"name": "GPU", "compute_capability": "12.0"}}
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


#: A description an earlier export of the same name left behind: the checks recognize one by its ``schema_version``.
_EARLIER_DESCRIPTION = '{"schema_version": 1, "batch": {"dynamic": true}}'


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

    @pytest.mark.skipif(not hasattr(os, "chown"), reason="POSIX ownership")
    def test_the_sidecar_takes_the_engines_group(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A service that reads the engine through its group must be able to read the description the same way."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        calls: list[tuple[str, int, int]] = []
        monkeypatch.setattr(os, "chown", lambda path, uid, gid: calls.append((Path(path).parent.name, uid, gid)))

        write_engine_metadata(engine, _metadata())

        assert calls == [(tmp_path.name, -1, engine.stat().st_gid)]

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
    @pytest.mark.parametrize("refused", ["copymode", "chown"])
    def test_a_filesystem_that_refuses_the_permission_change_still_gets_the_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, refused: str
    ) -> None:
        """ExFAT, some SMB shares, or a group the user is not in; the description is worth more than its permissions."""
        engine = tmp_path / "model.trt"
        engine.write_bytes(b"engine")
        monkeypatch.setattr(shutil if refused == "copymode" else os, refused, _raise_os_error)

        path = write_engine_metadata(engine, _metadata())

        assert json.loads(path.read_text())["schema_version"] == 1

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
        real_copymode = shutil.copymode
        interleaved: list[bool] = []

        def copymode_after_the_other_writer(source: object, destination: object, **kwargs: object) -> None:
            if not interleaved:
                interleaved.append(True)
                write_engine_metadata(second, _metadata(precision="fp32"))
            real_copymode(source, destination, **kwargs)

        monkeypatch.setattr(shutil, "copymode", copymode_after_the_other_writer)

        write_engine_metadata(first, _metadata(precision="fp16"))

        precisions = [json.loads((tmp_path / name).read_text())["build"]["precision"] for name in ("a.json", "b.json")]
        assert precisions == ["fp16", "fp32"]

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


def _patch_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, has_fp16_flag: bool) -> list[str]:
    """Stub the ONNX stage and the polygraphy chain so ``_convert`` runs without TensorRT, and record the engines.

    ``save_engine`` really writes the file, so the sidecar has an engine to sit next to. The stand-in ``tensorrt``
    module reports a weakly typed version, with or without the FP16 builder flag (a lean wheel lacks it).

    Args:
        monkeypatch: Fixture used to replace the entry points on the module under test.
        tmp_path: Directory the ONNX and engine files live in.
        has_fp16_flag: Whether the stand-in ``tensorrt.BuilderFlag`` carries ``FP16``.

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

    def _save_engine(engine: object, path: str) -> None:
        Path(path).write_bytes(b"engine")
        engines.append(path)

    onnx_path = str(tmp_path / "m.onnx")
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(tensorrt_export, "_IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr(tensorrt_export, "_IS_POLYGRAPHY_AVAILABLE", True)
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda self, graph: onnx_path)
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("builder", _Network(), "parser"))
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda parsed, config: "engine")
    monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)
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
        warnings: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: warnings.append(message % args))

        engine = Path(TensorRTExporter(TensorRTConfig(fp16=False, metadata=True))._convert(_graph()))

        assert warnings == []
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

    def test_a_description_left_by_an_earlier_export_is_reported_when_metadata_is_off(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The new engine replaced one of the same name, so the old description no longer describes the file."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        stale = tmp_path / "m_fp32.json"
        stale.write_text(_EARLIER_DESCRIPTION)
        warnings: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: warnings.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert len(warnings) == 1
        assert "m_fp32.json" in warnings[0]
        assert stale.read_text() == _EARLIER_DESCRIPTION, "a file the export did not write is left alone"

    def test_a_rebuild_through_build_engine_reports_an_earlier_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_engine`` on an ``.onnx`` replaces the engine too, so it gives the same warning as ``export``."""
        _patch_build(monkeypatch, tmp_path, has_fp16_flag=True)
        (tmp_path / "m_fp32.json").write_text(_EARLIER_DESCRIPTION)
        warnings: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: warnings.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False)).build_engine(str(tmp_path / "m.onnx"))

        assert len(warnings) == 1

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
        warnings: list[str] = []
        monkeypatch.setattr(tensorrt_export.logger, "warning", lambda message, *args: warnings.append(message % args))

        TensorRTExporter(TensorRTConfig(fp16=False))._convert(_graph())

        assert warnings == []

    def test_a_description_without_a_recorded_build_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The precision comes from the build; asking before any build is refused whether or not TensorRT imports."""
        monkeypatch.setitem(sys.modules, "tensorrt", None)
        exporter = TensorRTExporter(TensorRTConfig(metadata=True))

        with pytest.raises(RuntimeError, match="build_engine"):
            exporter._write_metadata(_graph(), "model.trt")

    @pytest.mark.parametrize("value", ["yes", 1, None])
    def test_metadata_must_be_a_bool(self, value: object) -> None:
        """A truthy non-``bool`` would write a file by accident, so it is refused before any work on the model."""
        with pytest.raises(ValueError, match="trt_metadata"):
            TensorRTExporter(TensorRTConfig(metadata=value))

    def test_the_description_is_off_unless_the_keyword_is_given(self) -> None:
        """Without ``trt_metadata`` the configuration writes no description."""
        assert TensorRTExporter.build_config().metadata is False
