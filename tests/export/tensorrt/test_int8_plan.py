# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``export(format="tensorrt", quantization="int8")``: the request, the host checks and the quantization plan.

The request checks and the build wiring run without TensorRT. The plan is pure ONNX work and runs on CPU with ``onnx``,
on small synthetic graphs that mirror RF-DETR's naming, plus exact-count checks on real exported Nano, segmentation and
keypoint graphs. The graph rewrite is in ``test_int8_graph.py``, calibration and the accuracy passes in
``test_int8_calibration.py``, and the GPU end-to-end classes in ``test_end_to_end.py``.
"""

from __future__ import annotations

import contextlib
import logging
import pickle
import re
import sys
import types
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.export._onnx import exporter as onnx_export
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import quantize
from rfdetr.export._tensorrt.exporter import TensorRTConfig, TensorRTExporter
from rfdetr.export.prepare import ExportGraph
from tests._markers import _requires_onnx_and_onnxruntime, _requires_onnxconverter_common
from tests.export.tensorrt.helpers import (
    BACKBONE,
    DECODER,
    HEAD,
    _attention_and_mlp,
    _names,
    _outputs_before_and_after_lift,
    _weight,
)

# Module-level on purpose: parametrized constants and graph builders touch ONNX while the module is collected, so a
# guarded import would raise ``NameError`` instead of skipping.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, compose, helper, numpy_helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402


def _export_graph(*extra_outputs: str) -> ExportGraph:
    """Return a static-batch detector :class:`ExportGraph` with ``dets``/``labels`` plus *extra_outputs*.

    Examples:
        >>> _export_graph("masks").output_names
        ('dets', 'labels', 'masks')
    """
    return ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 3, 8, 8),
        input_names=("input",),
        output_names=("dets", "labels", *extra_outputs),
        dynamic_axes=None,
        shape=(8, 8),
        backbone_only=False,
    )


def _calibration_dir(root: Path) -> Path:
    """Return a calibration directory under *root* that holds one image file, empty because only its suffix is read.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as root:
        ...     [path.name for path in _calibration_dir(Path(root)).iterdir()]
        ['image.jpg']
    """
    images = root / "images"
    images.mkdir(exist_ok=True)
    (images / "image.jpg").touch()
    return images


def _refuse_host(cls: type) -> None:
    """Stand in for ``_require_int8_host`` on a host that cannot build INT8.

    Examples:
        >>> _refuse_host(object)
        Traceback (most recent call last):
        ...
        ImportError: host refused
    """
    raise ImportError("host refused")


def _ints(name: str, values: int | list[int]) -> Any:
    """Return an ``int64`` initializer named *name* holding *values* (a shape, an axis list or an index).

    Examples:
        >>> _ints("axes", [0]).dims
        [1]
    """
    return numpy_helper.from_array(np.array(values, np.int64), name)


def _normed_projection_attention(projection: str, *, prefix: str = BACKBONE, sequence: int = 400) -> Any:
    """Build attention with a ``LayerNormalization`` after one projection (FP16-sized backbone attention by default).

    The projection trace does not walk through a norm, so the ``q``, ``k`` or ``v`` projection named by *projection*
    is hidden from it, as with timm's ``qk_norm``.

    Args:
        projection: ``"q"``, ``"k"`` or ``"v"``.
        prefix: Node-name prefix of the attention block.
        sequence: Tokens per sequence.

    Returns:
        A ``ModelProto`` whose MLP sits in the backbone.

    Examples:
        >>> model = _normed_projection_attention("k")
        >>> next(node.input[0] for node in model.graph.node if node.op_type == "LayerNormalization")
        'k0'
    """
    model = _attention_and_mlp(prefix, sequence=sequence, head_size=16)
    graph = model.graph
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.ones(32, np.float32), "norm_scale"),
            numpy_helper.from_array(np.zeros(32, np.float32), "norm_bias"),
        ]
    )
    reshape = next(i for i, node in enumerate(graph.node) if node.name.endswith(f"{projection}/Reshape"))
    graph.node[reshape].input[0] = "normed"
    norm_inputs = [f"{projection}0", "norm_scale", "norm_bias"]
    graph.node.insert(reshape, helper.make_node("LayerNormalization", norm_inputs, ["normed"], f"{prefix}norm"))
    return model


def _fused_qkv_attention(*, sequence: int, splitter: str) -> Any:
    """Build a backbone ``x -> fused qkv -> attention -> MLP`` graph whose q, k and v come from one projection.

    PE-Core-T (Atto, Femto, Pico) projects q, k and v with one ``qkv`` MatMul, reshapes the result to
    ``(3, batch, heads, tokens, head_size)`` and takes q, k and v apart with *splitter*: one ``Split``, a ``Gather``
    per tensor, or a ``Slice`` + ``Squeeze`` per tensor. Two heads of 16 channels.

    Args:
        sequence: Tokens per sequence.
        splitter: ``"Split"``, ``"Gather"`` or ``"Slice"``.

    Returns:
        A checked ``ModelProto``.

    Examples:
        >>> model = _fused_qkv_attention(sequence=4, splitter="Gather")
        >>> [node.op_type for node in model.graph.node].count("Gather")
        3
    """
    heads, head_size = 2, 16
    width = heads * head_size
    p = BACKBONE
    inits = [_weight("wqkv", width, 3 * width, seed=0), _weight("wo", width, width, seed=1)]
    inits += [_weight("w1", width, 2 * width, seed=2), _weight("w2", 2 * width, width, seed=3)]
    inits += [numpy_helper.from_array(np.zeros(3 * width, np.float32), "bqkv"), _ints("axis0", [0])]
    inits += [_ints("heads", [1, sequence, 3, heads, head_size]), _ints("merged", [1, sequence, width])]
    nodes = [
        helper.make_node("MatMul", ["x", "wqkv"], ["qkv0"], f"{p}qkv/MatMul"),
        helper.make_node("Add", ["bqkv", "qkv0"], ["qkv1"], f"{p}qkv/Add"),
        helper.make_node("Reshape", ["qkv1", "heads"], ["qkv2"], f"{p}qkv/Reshape"),
        helper.make_node("Transpose", ["qkv2"], ["qkv3"], f"{p}qkv/Transpose", perm=[2, 0, 3, 1, 4]),
    ]
    if splitter == "Split":
        inits.append(_ints("sizes", [1, 1, 1]))
        nodes.append(helper.make_node("Split", ["qkv3", "sizes"], ["q0", "k0", "v0"], f"{p}qkv/Split", axis=0))
    for index, name in enumerate("qkv"):
        if splitter == "Gather":
            inits.append(_ints(f"{name}_index", index))
            nodes.append(helper.make_node("Gather", ["qkv3", f"{name}_index"], [name], f"{p}{name}/Gather", axis=0))
            continue
        if splitter == "Slice":
            inits += [_ints(f"{name}_start", [index]), _ints(f"{name}_end", [index + 1])]
            slice_inputs = ["qkv3", f"{name}_start", f"{name}_end", "axis0"]
            nodes.append(helper.make_node("Slice", slice_inputs, [f"{name}0"], f"{p}{name}/Slice"))
        nodes.append(helper.make_node("Squeeze", [f"{name}0", "axis0"], [name], f"{p}{name}/Squeeze"))
    nodes += [
        helper.make_node("Transpose", ["k"], ["kt"], f"{p}k/Transpose", perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["q", "kt"], ["s"], f"{p}scores/MatMul"),
        helper.make_node("Softmax", ["s"], ["probs"], f"{p}Softmax", axis=-1),
        helper.make_node("MatMul", ["probs", "v"], ["c"], f"{p}context/MatMul"),
        helper.make_node("Transpose", ["c"], ["c1"], f"{p}merge/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["c1", "merged"], ["c2"], f"{p}merge/Reshape"),
        helper.make_node("MatMul", ["c2", "wo"], ["a"], f"{p}o/MatMul"),
        helper.make_node("MatMul", ["a", "w1"], ["h"], f"{p}fc1/MatMul"),
        helper.make_node("Relu", ["h"], ["g"], f"{p}act/Relu"),
        helper.make_node("MatMul", ["g", "w2"], ["y"], f"{p}fc2/MatMul"),
    ]
    tokens = [1, sequence, width]
    graph = helper.make_graph(
        nodes,
        "fused_qkv",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, tokens)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, tokens)],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model, full_check=True)
    return model


class TestInt8Request:
    """``quantization="int8"`` is validated from the configuration alone, before any forward pass."""

    @pytest.mark.parametrize("quantization", ["fp16", "fp32", "int4", "INT8"])
    def test_unknown_mode_is_refused(self, quantization: str) -> None:
        """Any quantization other than None or 'int8' is rejected at construction."""
        with pytest.raises(ValueError) as exc_info:
            TensorRTExporter(TensorRTConfig(quantization=quantization))  # type: ignore[arg-type]
        assert "quantization=None or 'int8'" in str(exc_info.value)

    def test_int8_needs_calibration_data(self) -> None:
        """Quantization='int8' without calibration_data is rejected at construction."""
        with pytest.raises(ValueError) as exc_info:
            TensorRTExporter(TensorRTConfig(quantization="int8"))
        assert "needs calibration_data" in str(exc_info.value)

    def test_calibration_data_without_int8_is_refused(self, tmp_path: Path) -> None:
        """calibration_data without quantization='int8' is rejected at construction."""
        with pytest.raises(ValueError) as exc_info:
            TensorRTExporter(TensorRTConfig(calibration_data=str(_calibration_dir(tmp_path))))
        assert "only read with quantization='int8'" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("settings", "reason"),
        [
            pytest.param({"fp16": False}, "fp16=False", id="fp32"),
            pytest.param({"dynamic_batch": True, "max_batch_size": 4}, "static batch", id="dynamic-batch"),
            pytest.param({"backbone_only": True}, "backbone_only", id="backbone-only"),
            pytest.param({"max_images": 0}, "positive integer", id="no-images"),
            pytest.param({"max_images": -1}, "positive integer", id="negative-images"),
            pytest.param({"max_images": True}, "positive integer", id="bool-images"),
            pytest.param({"max_images": 2.5}, "positive integer", id="float-images"),
            pytest.param({"max_images": np.bool_(True)}, "positive integer", id="numpy-bool-images"),
        ],
    )
    def test_unsupported_combination_is_refused(self, tmp_path: Path, settings: dict, reason: str) -> None:
        """INT8 with FP32, dynamic batch, backbone_only or a non-positive-integer max_images is rejected."""
        config = TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), **settings)
        with pytest.raises(ValueError, match=reason):
            TensorRTExporter(config)

    @pytest.mark.parametrize(
        ("data", "message"),
        [
            pytest.param(["a.jpg"], "calibration_data", id="list"),
            pytest.param(torch.zeros(1, 3, 8, 8), "calibration_data", id="tensor"),
            pytest.param(np.zeros((3, 8, 8), np.float32), "must be rank 4", id="rank-3"),
            pytest.param(np.zeros((1, 3, 8, 8), np.uint8), "floating point", id="uint8"),
            pytest.param(np.zeros((0, 3, 8, 8), np.float32), "at least one image", id="empty"),
        ],
    )
    def test_unusable_calibration_data_is_refused(self, data: object, message: str) -> None:
        """Calibration data that is not a path or a non-empty floating rank-4 array is rejected."""
        with pytest.raises(ValueError, match=message):
            TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=data))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("layout", "message"),
        [
            pytest.param("missing", "does not exist", id="missing-path"),
            pytest.param("empty-dir", "No calibration images found", id="empty-directory"),
            pytest.param("text-file", r"must be a \.npy array", id="non-npy-file"),
            pytest.param("empty-string", "calibration_data to be a directory", id="empty-string"),
        ],
    )
    def test_unusable_calibration_path_is_refused(self, tmp_path: Path, layout: str, message: str) -> None:
        """A missing path, an image-less directory, a non-.npy file or an empty string is rejected at construction."""
        data = {
            "missing": tmp_path / "missing",
            "empty-dir": tmp_path,
            "text-file": tmp_path / "notes.txt",
            "empty-string": "",
        }[layout]
        if layout == "text-file":
            data.write_text("not an array")
        with pytest.raises(ValueError, match=message):
            TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=data))

    @pytest.mark.parametrize("max_images", [np.int64(5), 5])
    def test_integer_max_images_is_accepted(self, tmp_path: Path, max_images: int | np.integer) -> None:
        """A Python or NumPy integer max_images is a valid calibration image cap."""
        TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), max_images=max_images)
        )

    def test_export_keywords_reach_the_configuration(self, tmp_path: Path) -> None:
        """The export() keywords quantization, calibration_data and max_images land in the config."""
        config = TensorRTExporter.build_config(
            quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), max_images=7
        )
        assert (config.quantization, config.calibration_data, config.max_images) == (
            "int8",
            str(_calibration_dir(tmp_path)),
            7,
        )

    @pytest.mark.parametrize(
        ("onnx_path", "output_name", "engine"),
        [
            pytest.param("out/model.onnx", None, "out/model_int8.trt", id="plain"),
            pytest.param("model.onnx", None, "model_int8.trt", id="no-directory"),
            pytest.param("out\\model.onnx", None, "out\\model_int8.trt", id="windows-separator"),
            pytest.param("out/model.v1.onnx", None, "out/model.v1_int8.trt", id="dotted-stem"),
            pytest.param("out/model.onnx", "custom", "out/custom.trt", id="custom-name"),
            pytest.param("out/model.onnx", "custom.trt", "out/custom.trt", id="custom-name-with-extension"),
        ],
    )
    def test_engine_is_named_int8(self, tmp_path: Path, onnx_path: str, output_name: str | None, engine: str) -> None:
        """An INT8 engine path carries the _int8 suffix unless the caller names the output."""
        config = TensorRTConfig(
            quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), output_name=output_name
        )
        assert TensorRTExporter(config).build_engine(onnx_path, dry_run=True) == engine

    @pytest.mark.parametrize("fp16", [True, False])
    def test_precision_suffix_follows_the_request(self, tmp_path: Path, fp16: bool) -> None:
        """A non-INT8 engine path ends in _fp16 or _fp32 according to the fp16 flag."""
        config = TensorRTConfig(quantization=None, fp16=fp16)
        expected = "model_fp16.trt" if fp16 else "model_fp32.trt"
        assert TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"), dry_run=True).endswith(expected)


class TestInt8Host:
    """The host must have onnxruntime and TensorRT >= 10; both are checked before the ONNX export."""

    def test_missing_onnxruntime_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An INT8 build without onnxruntime raises ImportError naming it."""
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", False)
        with pytest.raises(ImportError) as exc_info:
            TensorRTExporter._require_int8_host()
        assert "onnxruntime" in str(exc_info.value)

    @pytest.mark.parametrize("version", ["8.6.1", "9.3.0.post12.dev1", "unknown"])
    def test_old_tensorrt_is_refused(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """An INT8 build on TensorRT older than 10, or of unknown version, raises ImportError."""
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__=version))
        with pytest.raises(ImportError) as exc_info:
            TensorRTExporter._require_int8_host()
        assert "TensorRT 10 or newer" in str(exc_info.value)

    @pytest.mark.parametrize("version", ["10.0.1.6", "11.3.0.99", "11.4.0.106"])
    def test_supported_tensorrt_is_accepted(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """TensorRT 10 and newer pass the INT8 host check."""
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", True)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__=version))
        TensorRTExporter._require_int8_host()

    def test_missing_fp16_caster_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An INT8 build without onnxconverter-common raises ImportError naming it."""
        monkeypatch.setattr(tensorrt_export, "_IS_ONNXRUNTIME_AVAILABLE", True)
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="10.16.1.11"))
        with pytest.raises(ImportError) as exc_info:
            TensorRTExporter._require_int8_host()
        assert "onnxconverter-common" in str(exc_info.value)

    @pytest.mark.parametrize("form", ["str", "path"])
    def test_calibration_path_that_vanished_is_refused_before_the_forward_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, form: str
    ) -> None:
        """A calibration path deleted after construction is refused by check_environment."""
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        images = _calibration_dir(tmp_path)
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(images) if form == "str" else images)
        )
        (images / "image.jpg").unlink()
        images.rmdir()
        with pytest.raises(ValueError) as exc_info:
            exporter.check_environment()
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("name", "content", "message"),
        [
            pytest.param("raw.npy", np.zeros((1, 3, 8, 8), np.uint8), "preprocessed", id="uint8-npy"),
            pytest.param("image.npy", np.zeros((3, 8, 8), np.float32), "preprocessed", id="rank-3-npy"),
            pytest.param("empty.npy", np.zeros((0, 3, 8, 8), np.float32), "at least one image", id="no-image-npy"),
            pytest.param("broken.npy", b"", "could not read", id="unreadable-npy"),
            pytest.param("pickle.npy", pickle.dumps([1, 2, 3]), "could not read", id="pickle-npy"),
        ],
    )
    def test_unusable_calibration_file_is_refused_before_the_forward_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, content: np.ndarray | bytes, message: str
    ) -> None:
        """A calibration .npy that is not a preprocessed float rank-4 array is refused by check_environment."""
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            np.save(path, content)
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=path))
        with pytest.raises(ValueError, match=message):
            exporter.check_environment()

    def test_timing_cache_that_is_the_calibration_file_is_refused_before_the_forward_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A timing cache naming the calibration .npy, spelled another way, would be overwritten, so it is refused."""
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        calibration = tmp_path / "calibration.npy"
        np.save(calibration, np.zeros((1, 3, 8, 8), np.float32))
        exporter = TensorRTExporter(
            TensorRTConfig(
                quantization="int8", calibration_data=calibration, timing_cache=str(tmp_path / "." / calibration.name)
            )
        )
        with pytest.raises(ValueError) as exc_info:
            exporter.check_environment()
        assert "same file as calibration_data" in str(exc_info.value)

    def test_timing_cache_preflight_refuses_the_calibration_file(self, tmp_path: Path) -> None:
        """The cache preflight compares the calibration file too, and creates no lock file when it refuses."""
        calibration = tmp_path / "calibration.npy"
        np.save(calibration, np.zeros((1, 3, 8, 8), np.float32))
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=calibration, timing_cache=str(calibration))
        )
        with pytest.raises(ValueError) as exc_info:
            exporter._prepare_timing_cache()
        assert "same file as" in str(exc_info.value)
        assert not (tmp_path / "calibration.npy.lock").exists()

    @pytest.mark.parametrize(
        ("version", "warned"),
        [
            pytest.param("10.8.0.43", True, id="10.x-before-10.16"),
            pytest.param("12.0.0.1", True, id="newer-major"),
            # The placement alone was measured there, not the smoothing, the bias correction or the seg/keypoint models.
            pytest.param("10.16.1.11", True, id="10.16"),
            pytest.param("11.3.0.99", True, id="11.3"),
            pytest.param("11.4.0.106", False, id="11.4"),
            pytest.param("11.5.0.1", False, id="later-11.x"),
        ],
    )
    def test_unmeasured_tensorrt_is_warned_about_before_the_forward_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        version: str,
        warned: bool,
    ) -> None:
        """An INT8 build on an unmeasured TensorRT release still runs, with a warning from check_environment."""
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__=version))
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)))
        )
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            exporter.check_environment()
        assert (f"this build uses TensorRT {version}" in caplog.text) is warned

    @pytest.mark.parametrize(
        ("settings", "named"),
        [
            pytest.param({"version_compatible": True}, "trt_version_compatible=True", id="version-compatible"),
            pytest.param(
                {"hardware_compatibility": "ampere_plus"},
                "trt_hardware_compatibility='ampere_plus'",
                id="hardware-compatible",
            ),
        ],
    )
    def test_portability_flag_on_int8_is_warned_about(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
        settings: dict[str, Any],
        named: str,
    ) -> None:
        """A portable INT8 engine was never measured, so the flag is named in a warning even on a measured TensorRT."""
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="11.4.0.106"))
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), **settings)
        )
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            exporter._warn_if_int8_is_unmeasured()
        assert f"this build uses {named}" in caplog.text

    def test_preprocessed_npy_is_accepted(self, tmp_path: Path) -> None:
        """A float rank-4 .npy file is an acceptable calibration path."""
        path = tmp_path / "images.npy"
        np.save(path, np.zeros((2, 3, 8, 8), np.float32))
        TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=path))._require_calibration_path()

    def test_build_engine_refuses_a_missing_calibration_path_before_calibrating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """build_engine refuses a vanished calibration path before any calibration runs."""
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", lambda *_, **__: pytest.fail("calibration ran"))
        images = _calibration_dir(tmp_path)
        exporter = TensorRTExporter(TensorRTConfig(quantization="int8", calibration_data=str(images)))
        (images / "image.jpg").unlink()
        images.rmdir()
        with pytest.raises(ValueError) as exc_info:
            exporter.build_engine(str(tmp_path / "model.onnx"))
        assert "does not exist" in str(exc_info.value)

    @pytest.mark.parametrize(
        "enter",
        [
            pytest.param(lambda exporter, onnx_path: exporter._convert(_export_graph()), id="convert"),
            pytest.param(lambda exporter, onnx_path: exporter.build_engine(onnx_path), id="build-engine"),
            pytest.param(lambda exporter, onnx_path: exporter.check_environment(), id="check-environment"),
        ],
    )
    def test_unbuildable_host_is_refused_before_any_work(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enter: Any
    ) -> None:
        """A host that cannot build INT8 is refused before the ONNX export or calibration, at each entry point."""
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(_refuse_host))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", lambda *_, **__: pytest.fail("calibration ran"))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)))
        )
        with pytest.raises(ImportError) as exc_info:
            enter(exporter, str(tmp_path / "model.onnx"))
        assert "host refused" in str(exc_info.value)

    def test_check_environment_leaves_the_host_alone_without_int8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without INT8, check_environment does not probe the INT8 host requirements."""
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(_refuse_host))
        TensorRTExporter(TensorRTConfig()).check_environment()

    def test_description_records_the_int8_precision(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The engine description document records precision 'int8' for an INT8 build."""
        written: dict[str, Any] = {}
        monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="11.3.0.99"))
        monkeypatch.setattr(tensorrt_export, "gpu_facts", lambda: None)
        monkeypatch.setattr(
            tensorrt_export, "write_engine_metadata", lambda engine_path, document: written.update(document) or tmp_path
        )
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), metadata=True)
        )
        exporter._write_metadata(
            _export_graph(), str(tmp_path / "m_int8.trt"), fp16=True, engine_facts={"size": 1, "sha256": "0" * 64}
        )
        assert written["build"]["precision"] == "int8"

    @pytest.mark.parametrize("extra", ["masks", "keypoints"])
    def test_segmentation_and_keypoint_models_reach_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture, tmp_path: Path, extra: str
    ) -> None:
        """Segmentation and keypoint graphs pass the output check and reach the ONNX export."""
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        mocker.patch.object(onnx_export.OnnxExporter, "__call__", side_effect=RuntimeError("ONNX export reached"))
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)))
        )
        with pytest.raises(RuntimeError) as exc_info:
            exporter._convert(_export_graph(extra))
        assert "ONNX export reached" in str(exc_info.value)

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param(("masks", "keypoints"), id="both-heads"),
            pytest.param(("boxes",), id="unknown-output"),
            pytest.param(("keypoints", "masks"), id="heads-reversed"),
        ],
    )
    def test_unsupported_outputs_are_refused_before_the_onnx_export(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, extra: tuple[str, ...]
    ) -> None:
        """Output sets other than one mask or keypoint head are refused before the ONNX export."""
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: pytest.fail("ONNX export ran"))
        exporter = TensorRTExporter(
            TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)))
        )
        with pytest.raises(NotImplementedError) as exc_info:
            exporter._convert(_export_graph(*extra))
        assert "detection, segmentation and keypoint models only" in str(exc_info.value)


class TestInt8BuildWiring:
    """How an INT8 request reaches the build: strongly typed parse, builder options, engine and description names."""

    def test_builds_the_quantized_graph_strongly_typed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """The INT8 build parses the quantized graph strongly typed and builds with fp16 off."""
        calls: dict[str, Any] = {}

        @contextlib.contextmanager
        def fake_source(onnx_path: str, **kwargs: Any) -> Iterator[str]:
            calls["source"] = (onnx_path, kwargs)
            yield str(tmp_path / "quantized.onnx")

        def fake_parse(path: str, **kwargs: Any) -> tuple[None, object, None]:
            calls["parse"] = (path, kwargs)
            return (None, types.SimpleNamespace(num_inputs=0), None)

        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", fake_source)
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", fake_parse)
        monkeypatch.setattr(tensorrt_export, "_dynamic_batch_inputs", lambda network: {})
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: ("config", kwargs))
        monkeypatch.setattr(
            tensorrt_export,
            "engine_from_network",
            lambda parsed, config: types.SimpleNamespace(serialize=lambda: ("engine", config)),
        )
        monkeypatch.setattr(
            tensorrt_export,
            "save_file",
            lambda contents, dest, description=None: calls.setdefault("saved", (contents, dest)),
        )

        config = TensorRTConfig(
            quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), max_images=3, verbose=False
        )
        engine_path = TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"))

        assert calls["source"] == (
            str(tmp_path / "model.onnx"),
            {
                "calibration_data": str(_calibration_dir(tmp_path)),
                "max_images": 3,
                "dynamic_batch": False,
                "fp16_graph": tensorrt_export.fp16_source_graph,
            },
        )
        assert calls["parse"] == (str(tmp_path / "quantized.onnx"), {"strongly_typed": True})
        assert calls["saved"] == (("engine", ("config", {"fp16": False})), engine_path)
        assert engine_path == str(tmp_path / "model_int8.trt")

    def test_portability_and_the_timing_cache_reach_the_int8_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Hardware and version compatibility and the timing cache reach the INT8 engine build."""
        calls: dict[str, Any] = {}

        @contextlib.contextmanager
        def fake_source(onnx_path: str, **kwargs: Any) -> Iterator[str]:
            yield str(tmp_path / "quantized.onnx")

        def fake_engine_from_network(parsed: Any, config: Any, **kwargs: Any) -> types.SimpleNamespace:
            calls["engine"] = (config, kwargs)
            return types.SimpleNamespace(serialize=lambda: b"engine")

        cache = tmp_path / "timings.cache"
        cache.write_bytes(b"timings")
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_lean_runtime", lambda self: None)
        monkeypatch.setattr(
            TensorRTExporter, "_warn_if_version_compatibility_is_unverified", staticmethod(lambda: None)
        )
        monkeypatch.setattr(
            TensorRTExporter, "_hardware_compatibility_level", staticmethod(lambda level: f"level:{level}")
        )
        monkeypatch.setattr(tensorrt_export, "int8_source_graph", fake_source)
        monkeypatch.setattr(
            tensorrt_export,
            "network_from_onnx_path",
            lambda path, **kwargs: (None, types.SimpleNamespace(num_inputs=0), None),
        )
        monkeypatch.setattr(tensorrt_export, "_dynamic_batch_inputs", lambda network: {})
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: ("config", kwargs))
        monkeypatch.setattr(tensorrt_export, "engine_from_network", fake_engine_from_network)
        monkeypatch.setattr(tensorrt_export, "save_file", lambda contents, dest, description=None: None)

        config = TensorRTConfig(
            quantization="int8",
            calibration_data=str(_calibration_dir(tmp_path)),
            verbose=False,
            hardware_compatibility="same_compute_capability",
            version_compatible=True,
            timing_cache=cache,
        )
        engine_path = TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"))

        (_, create_kwargs), engine_kwargs = calls["engine"][0], calls["engine"][1]
        assert create_kwargs["hardware_compatibility_level"] == "level:same_compute_capability"
        assert create_kwargs["version_compatible"] is True
        assert Path(create_kwargs["load_timing_cache"]).samefile(cache)
        assert Path(engine_kwargs["save_timing_cache"]).samefile(cache)
        assert engine_path == str(tmp_path / "model_int8_same_compute_capability_version_compatible.trt")

    def test_dry_run_names_an_int8_engine_with_its_portability_details(self, tmp_path: Path) -> None:
        """A dry-run INT8 engine path carries the hardware and version-compatible suffixes."""
        config = TensorRTConfig(
            quantization="int8",
            calibration_data=str(_calibration_dir(tmp_path)),
            hardware_compatibility="ampere_plus",
            version_compatible=True,
        )
        engine_path = TensorRTExporter(config).build_engine(str(tmp_path / "model.onnx"), dry_run=True)
        assert engine_path == str(tmp_path / "model_int8_ampere_plus_version_compatible.trt")

    def test_description_does_not_replace_a_foreign_file_beside_an_int8_engine(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A foreign file at the description path is not overwritten and the engine is not built."""
        (tmp_path / "model_int8.json").write_text("{}")  # a label map of the user's own, not an RF-DETR description
        monkeypatch.setattr(TensorRTExporter, "_require_tensorrt", classmethod(lambda cls: None))
        monkeypatch.setattr(TensorRTExporter, "_require_int8_host", classmethod(lambda cls: None))
        monkeypatch.setattr(onnx_export.OnnxExporter, "__call__", lambda *_: tmp_path / "model.onnx")
        monkeypatch.setattr(TensorRTExporter, "_build", lambda *_, **__: pytest.fail("the engine was built"))
        config = TensorRTConfig(quantization="int8", calibration_data=str(_calibration_dir(tmp_path)), metadata=True)
        with pytest.raises(FileExistsError) as exc_info:
            TensorRTExporter(config)._convert(_export_graph())
        assert "model_int8.json" in str(exc_info.value)


class TestPlanInt8:
    """Which nodes get Q/DQ: the rules in :mod:`rfdetr.export._tensorrt.quantize`, on synthetic graphs."""

    def test_fusable_backbone_attention_is_quantized_whole(self) -> None:
        """A fusable backbone attention block is quantized whole, scores and context included."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        assert _names(plan) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul", "fc1/MatMul", "fc2/MatMul"]
        batched = sorted(
            (name.split("/")[-2], index)
            for name, index in plan.activations
            if "/scores/" in name or "/context/" in name
        )
        assert batched == [("context", 0), ("context", 1), ("scores", 0), ("scores", 1)]

    def test_backbone_attention_left_float_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backbone attention left in FP16 is reported in an info message."""
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=326, head_size=16))
        assert [m for m in messages if "stay FP16" in m] != []

    def test_decoder_attention_left_float_is_not_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Decoder attention left in FP16 is expected and produces no info message."""
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(_attention_and_mlp(DECODER, sequence=326, head_size=16))
        assert [m for m in messages if "stay FP16" in m] == []

    def test_attention_left_float_for_its_projections_is_not_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Attention left FP16 because its output projection is outside the regions is not reported as unfusable."""
        # FP16 because its output projection is outside the quantized regions, not because the kernel cannot fuse it.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("o/MatMul"):
                node.name = f"{HEAD}o/MatMul"
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(model)
        assert [m for m in messages if "stay FP16" in m] == []

    def test_attention_sharing_a_name_stays_float(self) -> None:
        """Attention whose scores and context share a node name is not recognised and stays float."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("context/MatMul"):
                node.name = f"{BACKBONE}scores/MatMul"
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    def test_token_limit_is_quantized_whole(self) -> None:
        """A window of exactly 325 tokens is still quantized whole, scores included."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=325, head_size=32))
        assert sum("/scores/" in name for name, _ in plan.activations) == 2

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Mul", "Sub", "Where"])
    def test_scaled_or_masked_scores_are_still_recognised_as_attention(self, op: str) -> None:
        """Scores scaled, masked or cast before the Softmax are still recognised as attention."""
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = op
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize("op", ["Cast", "Identity"])
    def test_probabilities_through_a_passthrough_are_still_recognised_as_attention(self, op: str) -> None:
        """A Cast or Identity between Softmax and the context multiply keeps the block recognised."""
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16)
        nodes = list(model.graph.node)
        index = next(i for i, node in enumerate(nodes) if node.op_type == "Softmax")
        nodes[index].output[0] = "p32"
        nodes.insert(index + 1, helper.make_node(op, ["p32"], ["p"], f"{DECODER}probs/{op}"))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    def test_unrecognised_attention_in_a_region_is_refused(self) -> None:
        """An attention block in a quantized region with an unknown score op is refused."""
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = "Max"
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "not a recognised attention block" in str(exc_info.value)

    def test_unrecognised_attention_named_outside_the_regions_is_refused(self) -> None:
        """Renaming the Softmax outside the regions does not let an unrecognised block slip through."""
        # Its projections sit in a quantized region, so a rename of the Softmax must not let them slip through.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, scaled_scores=True)
        next(node for node in model.graph.node if node.name.endswith("scale/Div")).op_type = "Max"
        next(node for node in model.graph.node if node.op_type == "Softmax").name = f"{HEAD}Softmax"
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "not a recognised attention block" in str(exc_info.value)

    @pytest.mark.parametrize("op", ["Mul", "Div", "Add", "Sub", "Dropout", "Relu"])
    def test_attention_with_an_op_between_softmax_and_context_is_refused(self, op: str) -> None:
        """An op between the Softmax and the context multiply makes the block unrecognised and refused."""
        # The probabilities reach the context multiply through another op: not a block the planner knows, and a
        # quantized output projection behind an FP16 attention is what TensorRT 11.3 computes wrongly.
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16)
        context = next(node for node in model.graph.node if node.name.endswith("context/MatMul"))
        context.input[0] = "p_after"
        after = helper.make_node(
            op, ["p"] if op in ("Dropout", "Relu") else ["p", "half"], ["p_after"], f"{DECODER}after/{op}"
        )
        nodes = list(model.graph.node)
        nodes.insert(nodes.index(context), after)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "not a recognised attention block" in str(exc_info.value)

    def test_float_attention_with_an_unfollowable_output_is_refused(self) -> None:
        """A float attention whose merged output has a second reader is refused."""
        model = _attention_and_mlp(DECODER, sequence=8, head_size=16, second_reader=True)
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "output projection" in str(exc_info.value)

    def test_fusable_attention_with_a_projection_outside_the_regions_stays_float(self) -> None:
        """A fusable attention with its output projection outside the regions stays float whole."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        for node in model.graph.node:
            if node.name.endswith("o/MatMul"):
                node.name = f"{HEAD}o/MatMul"
        plan = quantize.plan_int8(model)
        assert (_names(plan), [n for n, _ in plan.activations if "/scores/" in n]) == (["fc1/MatMul", "fc2/MatMul"], [])

    @pytest.mark.parametrize("splitter", ["Split", "Gather", "Slice"])
    def test_fused_qkv_behind_float_attention_stays_float(self, splitter: str) -> None:
        """A fused qkv projection in front of FP16 attention (400 tokens) is not quantized."""
        plan = quantize.plan_int8(_fused_qkv_attention(sequence=400, splitter=splitter))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize("splitter", ["Split", "Gather", "Slice"])
    def test_fusable_attention_with_fused_qkv_is_quantized_whole(self, splitter: str) -> None:
        """A fused qkv projection in front of fusable attention (325 tokens) is quantized with it, scores included."""
        plan = quantize.plan_int8(_fused_qkv_attention(sequence=325, splitter=splitter))
        batched = sorted(
            name.split("/")[-2] for name, _ in plan.activations if "/scores/" in name or "/context/" in name
        )
        assert (_names(plan), batched) == (
            ["qkv/MatMul", "o/MatMul", "fc1/MatMul", "fc2/MatMul"],
            ["context", "context", "scores", "scores"],
        )

    @pytest.mark.parametrize("splitter", ["Split", "Gather", "Slice"])
    def test_fusable_attention_with_an_unplanned_fused_qkv_stays_float(self, splitter: str) -> None:
        """Fusable attention whose fused qkv projection cannot be planned stays FP16 whole."""
        model = _fused_qkv_attention(sequence=325, splitter=splitter)
        for node in model.graph.node:
            if node.name.endswith("qkv/MatMul"):
                node.name = f"{HEAD}qkv/MatMul"
        plan = quantize.plan_int8(model)
        assert (_names(plan), [n for n, _ in plan.activations if "/scores/" in n]) == (["fc1/MatMul", "fc2/MatMul"], [])

    @pytest.mark.parametrize(
        ("op", "control", "prepare", "axes"),
        [
            pytest.param("Gather", [0], None, [], id="gather-batch-index"),
            pytest.param("Slice", [8], None, [2], id="slice-token-end"),
            pytest.param("Split", [1, 0], None, [], id="split-batch-sizes"),
            pytest.param("Squeeze", [0], "Unsqueeze", [], id="squeeze-inserted-axis"),
            pytest.param("Unsqueeze", [0], "Squeeze", [], id="unsqueeze-removed-axis"),
            pytest.param("Reshape", [1, 2, 8, 8], None, [], id="reshape-value-shape"),
        ],
    )
    def test_index_input_of_a_splitter_is_not_traced_as_the_projection(
        self, op: str, control: list[int], prepare: str | None, axes: list[int]
    ) -> None:
        """Legal auxiliary inputs may depend on a MatMul without making it the attention projection."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=8)
        samples = np.random.default_rng(7).normal(size=(1, 8, 16)).astype(np.float32)
        (expected,) = ReferenceEvaluator(model).run(None, {"x": samples})
        graph = model.graph
        graph.input.append(helper.make_tensor_value_info("selector", TensorProto.FLOAT, [1]))
        graph.initializer.extend(
            [
                numpy_helper.from_array(np.ones((1, 1), np.float32), "wsel"),
                _ints("offset", [value - 1 for value in control]),
                _ints("zero", [0]),
            ]
        )
        nodes = list(graph.node)
        context = next(i for i, node in enumerate(nodes) if node.name.endswith("context/MatMul"))
        nodes[context].input[1] = "v3"
        data = "v2"
        setup = [
            helper.make_node("MatMul", ["selector", "wsel"], ["sel"], f"{BACKBONE}sel/MatMul"),
            helper.make_node("Cast", ["sel"], ["sel_int"], to=TensorProto.INT64),
            helper.make_node("Add", ["sel_int", "offset"], ["control"]),
        ]
        if prepare:
            setup.append(helper.make_node(prepare, [data, "zero"], ["v_prepared"]))
            data = "v_prepared"
        inputs = [data, "control"]
        if op == "Slice":
            graph.initializer.append(_ints("slice_axes", axes))
            inputs = [data, "zero", "control", "slice_axes"]
        outputs = ["v3", "v_rest"] if op == "Split" else ["v3"]
        setup.append(helper.make_node(op, inputs, outputs))
        del graph.node[:]
        graph.node.extend([*nodes[:context], *setup, *nodes[context:]])

        onnx.checker.check_model(model, full_check=True)
        (actual,) = ReferenceEvaluator(model).run(None, {"x": samples, "selector": np.ones(1, np.float32)})
        np.testing.assert_array_equal(actual, expected)
        assert _names(quantize.plan_int8(model)) == ["sel/MatMul", "fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize(
        "float_first", [pytest.param(False, id="int8-eligible-first"), pytest.param(True, id="fp16-first")]
    )
    def test_shared_projection_keeps_both_attention_blocks_float(self, float_first: bool) -> None:
        """A Q projection shared with FP16 attention cannot be quantized in either node order."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        graph = model.graph
        float_prefix = f"{BACKBONE}float/"
        floating = _attention_and_mlp(float_prefix, sequence=8, head_size=16, mlp_prefix=float_prefix)
        float_graph = compose.add_prefix_graph(floating.graph, "float_", rename_nodes=False, rename_inputs=False)
        heads = next(init for init in float_graph.initializer if init.name == "float_heads")
        heads.CopyFrom(_ints("float_heads", [1, 8, 4, 8]))
        float_graph.node[1].input[0] = "q0"

        shared, *eligible_branch = graph.node
        float_branch = float_graph.node[1:]
        first, second = (float_branch, eligible_branch) if float_first else (eligible_branch, float_branch)
        graph.initializer.extend(float_graph.initializer)
        graph.output.extend(float_graph.output)
        del graph.node[:]
        graph.node.extend([shared, *first, *second])
        onnx.checker.check_model(model, full_check=True)

        plan = quantize.plan_int8(model)
        expected = {f"{prefix}{name}/MatMul" for prefix in (BACKBONE, float_prefix) for name in ("fc1", "fc2")}
        assert set(plan.weighted) == expected
        assert set(plan.activations) == {(name, 0) for name in expected}

    @pytest.mark.parametrize("projection", ["q", "k", "v"])
    def test_projection_the_trace_cannot_follow_into_float_attention_is_refused(self, projection: str) -> None:
        """A planned projection reaching FP16 attention through an op the trace does not follow is refused."""
        model = _normed_projection_attention(projection)
        with pytest.raises(ValueError, match="reaches the FP16 attention") as exc_info:
            quantize.plan_int8(model)
        assert f"{BACKBONE}{projection}/MatMul" in str(exc_info.value)

    def test_hidden_projection_in_front_of_float_decoder_attention_is_refused(self) -> None:
        """The refusal covers decoder attention, which always stays FP16, not only the backbone's."""
        model = _normed_projection_attention("q", prefix=DECODER, sequence=8)
        with pytest.raises(ValueError, match="reaches the FP16 attention"):
            quantize.plan_int8(model)

    def test_hidden_projection_of_a_fusable_block_left_float_is_refused(self) -> None:
        """A 325-token block left FP16 by an unplanned output projection still refuses a hidden planned q."""
        model = _normed_projection_attention("q", sequence=325)
        next(node for node in model.graph.node if node.name.endswith("o/MatMul")).name = f"{HEAD}o/MatMul"
        with pytest.raises(ValueError, match="reaches the FP16 attention"):
            quantize.plan_int8(model)

    @pytest.mark.parametrize("sequence", [325, 400])
    def test_unplanned_projection_the_trace_cannot_follow_keeps_attention_float(self, sequence: int) -> None:
        """A hidden projection outside the quantized regions keeps its attention FP16 and is not refused."""
        model = _normed_projection_attention("q", sequence=sequence)
        graph = model.graph
        next(node for node in graph.node if node.name.endswith("q/MatMul")).name = f"{HEAD}q/MatMul"
        assert _names(quantize.plan_int8(model)) == ["fc1/MatMul", "fc2/MatMul"]

    def test_unplannable_projection_ends_the_walk_into_float_attention(self) -> None:
        """An unnamed q projection, which the plan cannot name, still ends the walk from FP16 attention."""
        model = _attention_and_mlp(BACKBONE, sequence=400, head_size=16)
        graph = model.graph
        graph.initializer.append(_weight("wpre", 32, 32, seed=10))
        query = next(i for i, node in enumerate(graph.node) if node.name.endswith("q/MatMul"))
        graph.node[query].input[0] = "x_pre"
        graph.node[query].name = ""
        graph.node.insert(query, helper.make_node("MatMul", ["x", "wpre"], ["x_pre"], f"{BACKBONE}pre/MatMul"))
        assert _names(quantize.plan_int8(model)) == ["pre/MatMul", "fc1/MatMul", "fc2/MatMul"]

    def test_unnamed_projection_keeps_fusable_attention_float(self) -> None:
        """A fusable block whose q projection has no name, so the plan cannot quantize it, keeps attention FP16."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        next(node for node in model.graph.node if node.name.endswith("q/MatMul")).name = ""
        plan = quantize.plan_int8(model)
        assert (_names(plan), [n for n, _ in plan.activations if "/scores/" in n]) == (["fc1/MatMul", "fc2/MatMul"], [])

    @pytest.mark.parametrize("hidden", [False, True])
    def test_projection_of_fused_attention_dropped_by_a_later_rule_is_refused(self, hidden: bool) -> None:
        """A fused block whose q projection the GELU rule drops afterwards is refused, not left half INT8."""
        if hidden:
            model = _normed_projection_attention("q", sequence=8)
        else:
            model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        graph = model.graph
        graph.initializer.append(_weight("w3", 32, 16, seed=11))
        graph.node.extend(
            [
                helper.make_node("Div", ["q0", "sqrt2"], ["g0"], f"{BACKBONE}side/Div"),
                helper.make_node("Erf", ["g0"], ["g1"], f"{BACKBONE}side/Erf"),
                helper.make_node("Add", ["g1", "one"], ["g2"], f"{BACKBONE}side/Add"),
                helper.make_node("Mul", ["q0", "g2"], ["g3"], f"{BACKBONE}side/Mul"),
                helper.make_node("Mul", ["g3", "half"], ["q_gelu"], f"{BACKBONE}side/Mul_1"),
                helper.make_node("MatMul", ["q_gelu", "w3"], ["side"], f"{HEAD}side/MatMul"),
            ]
        )
        graph.output.append(helper.make_tensor_value_info("side", TensorProto.FLOAT, [1, 8, 16]))
        with pytest.raises(ValueError, match="INT8 attention"):
            quantize.plan_int8(model)

    @pytest.mark.parametrize("op", ["Shape", "Size"])
    def test_shape_of_a_planned_layer_does_not_count_as_feeding_float_attention(self, op: str) -> None:
        """FP16 attention scaled by a size read from a planned layer's output shape does not refuse that layer."""
        model = _attention_and_mlp(BACKBONE, sequence=400, head_size=16)
        graph = model.graph
        graph.initializer.append(_weight("wside", 32, 32, seed=9))
        scores = next(i for i, node in enumerate(graph.node) if node.name.endswith("scores/MatMul"))
        graph.node[scores].input[0] = "q_scaled"
        scale = [
            helper.make_node("MatMul", ["x", "wside"], ["side"], f"{BACKBONE}side/MatMul"),
            helper.make_node(
                op, ["side"], ["side_shape"], f"{BACKBONE}side/{op}", **({"start": -1} if op == "Shape" else {})
            ),
            helper.make_node("Cast", ["side_shape"], ["size"], f"{BACKBONE}side/Cast", to=TensorProto.FLOAT),
            helper.make_node("Div", ["q2", "size"], ["q_scaled"], f"{BACKBONE}q/Div"),
        ]
        for offset, node in enumerate(scale):
            graph.node.insert(scores + offset, node)
        assert _names(quantize.plan_int8(model)) == ["side/MatMul", "fc1/MatMul", "fc2/MatMul"]

    def test_gelu_chain_drops_every_projection_feeding_a_float_consumer(self) -> None:
        """A GELU chain ending in a float consumer leaves nothing to quantize and is refused."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=8, activation="Gelu")
        graph = model.graph
        graph.initializer.append(_weight("w3", 32, 16, seed=7))
        graph.node.extend(
            [
                helper.make_node("Gelu", ["y"], ["y_gelu"], f"{BACKBONE}act2/Gelu"),
                helper.make_node("MatMul", ["y_gelu", "w3"], ["head"], f"{HEAD}out/MatMul"),
            ]
        )
        graph.output.append(helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, 8, 16]))
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "RF-DETR model exports only" in str(exc_info.value)

    def test_decoder_attention_stays_float(self) -> None:
        """Decoder attention is never quantized; only its MLP linears are."""
        plan = quantize.plan_int8(_attention_and_mlp(DECODER, sequence=8, head_size=16))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize(
        ("block", "mlp_quantized"),
        [
            pytest.param(0, False, id="first-block"),
            pytest.param(3, False, id="outlier-block"),
            pytest.param(5, False, id="last-float-block"),
            pytest.param(6, True, id="first-int8-block"),
            pytest.param(11, True, id="last-block"),
        ],
    )
    def test_pe_backbone_keeps_the_mlp_of_its_first_six_blocks_float(self, block: int, mlp_quantized: bool) -> None:
        """A PE-Core-T block below 6 keeps ``fc1`` and ``fc2`` in FP16; its attention placement does not change."""
        prefix = f"/backbone/backbone.0/encoder/blocks.{block}/"
        model = _attention_and_mlp(
            f"{prefix}attn/", sequence=8, head_size=16, mlp_prefix=f"{prefix}mlp/", activation="Erf"
        )
        plan = quantize.plan_int8(model)
        assert _names(plan) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"] + (
            ["fc1/MatMul", "fc2/MatMul"] if mlp_quantized else []
        )
        assert sum("/scores/" in name for name, _ in plan.activations) == 2

    def test_pe_float_mlp_leaves_no_planned_activation_behind(self) -> None:
        """The excluded MLP projections have no Q/DQ input either, so no quantize step lands on the GELU path."""
        prefix = "/backbone/backbone.0/encoder/blocks.3/"
        model = _attention_and_mlp(
            f"{prefix}attn/", sequence=8, head_size=16, mlp_prefix=f"{prefix}mlp/", activation="Erf"
        )
        assert [name for name, _ in quantize.plan_int8(model).activations if "/mlp/" in name] == []

    def test_dinov2_backbone_keeps_its_mlp_in_int8(self) -> None:
        """The first DINOv2 layers are unchanged: the rule keys on the PE block names."""
        prefix = "/backbone/backbone.0/encoder/encoder/encoder/layer.3/"
        model = _attention_and_mlp(f"{prefix}attention/", sequence=8, head_size=16, mlp_prefix=f"{prefix}mlp/")
        assert _names(quantize.plan_int8(model))[-2:] == ["fc1/MatMul", "fc2/MatMul"]

    @pytest.mark.parametrize(
        ("channels", "group", "quantized"),
        [
            pytest.param(3, 1, False, id="rgb"),
            pytest.param(48, 1, False, id="48-channels"),
            pytest.param(32, 32, True, id="depthwise-32"),
        ],
    )
    def test_convolution_is_quantized_only_on_a_multiple_of_32_channels(
        self, channels: int, group: int, quantized: bool
    ) -> None:
        """A convolution is quantized only when its input channels are a multiple of 32."""
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["image", "first"], ["features"], f"{BACKBONE}patch/Conv", group=group),
                helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv"),
            ],
            "convs",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, channels, 8, 8])],
            [helper.make_tensor_value_info("mixed", TensorProto.FLOAT, [1, 16, 8, 8])],
            [
                numpy_helper.from_array(np.ones((32, channels // group, 1, 1), np.float32), "first"),
                numpy_helper.from_array(np.ones((16, 32, 1, 1), np.float32), "mix"),
            ],
        )
        plan = quantize.plan_int8(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
        assert ("patch/Conv" in _names(plan)) == quantized

    def test_heads_are_left_float(self) -> None:
        """A linear outside the backbone and decoder regions is not quantized."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD)
        assert "fc2/MatMul" not in _names(quantize.plan_int8(model))

    @pytest.mark.parametrize("activation", ["Erf", "Gelu"])
    def test_gelu_projection_waits_for_its_consumer(self, activation: str) -> None:
        """A GELU's producing projection stays float while its consumer does."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD, activation=activation)
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_gelu_projection_waits_for_a_consumer_the_plan_cannot_name(self) -> None:
        """An unnamed consumer of a GELU still counts as the float consumer."""
        # An unnamed fc2 cannot be planned, but it is still the FP16 consumer fc1's GELU feeds.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, activation="Erf")
        next(node for node in model.graph.node if node.name.endswith("fc2/MatMul")).name = ""
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_gelu_projection_waits_for_a_consumer_behind_another_op(self) -> None:
        """A GELU consumer hidden behind an Identity still counts as the float consumer."""
        # Only a GELU's direct readers were followed: an Identity before the unplannable fc2 hid it.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, activation="Erf")
        nodes = list(model.graph.node)
        fc2 = next(node for node in nodes if node.name.endswith("fc2/MatMul"))
        fc2.name, fc2.input[0] = "", "g_copy"
        index = nodes.index(fc2)
        nodes.insert(index, helper.make_node("Identity", ["g"], ["g_copy"], f"{BACKBONE}tap/Identity"))
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        assert _names(quantize.plan_int8(model)) == ["q/MatMul", "k/MatMul", "v/MatMul", "o/MatMul"]

    def test_relu_projection_does_not_wait(self) -> None:
        """A projection feeding a ReLU is quantized even when its consumer stays float."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, second_mlp_prefix=HEAD, activation="Relu")
        assert _names(quantize.plan_int8(model))[-1] == "fc1/MatMul"

    def test_graph_without_quantizable_layers_is_refused(self) -> None:
        """A graph with no layer in a quantized region is refused."""
        model = _attention_and_mlp(HEAD, sequence=8, head_size=16, mlp_prefix=HEAD)
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "RF-DETR model exports only" in str(exc_info.value)

    def test_layer_inside_a_subgraph_is_not_planned(self) -> None:
        """A MatMul inside an If branch is not planned."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        inner = helper.make_node("MatMul", ["c2", "wo"], ["inner"], f"{BACKBONE}inner/MatMul")
        branch = helper.make_graph([inner], "branch", [], [helper.make_tensor_value_info("inner", 1, [1, 8, 32])])
        model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
        model.graph.node.append(
            helper.make_node("If", ["flag"], ["picked"], "if", then_branch=branch, else_branch=branch)
        )
        model.graph.output.append(helper.make_tensor_value_info("picked", 1, [1, 8, 32]))
        assert "inner/MatMul" not in _names(quantize.plan_int8(model))

    def test_nodes_sharing_a_name_are_skipped(self) -> None:
        """Nodes that share a name cannot be referred to and are skipped."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=8)
        for node in model.graph.node:
            if node.name.endswith("fc2/MatMul"):
                node.name = f"{BACKBONE}fc1/MatMul"
        with pytest.raises(ValueError) as exc_info:
            quantize.plan_int8(model)
        assert "RF-DETR model exports only" in str(exc_info.value)


class TestAttentionShapes:
    """INT8 attention needs a head size TensorRT's fused kernel accepts and a window of at most 325 tokens."""

    @pytest.mark.parametrize("head_size", [16, 32, 64])
    @pytest.mark.parametrize("sequence", [1, 8, 324, 325])
    def test_supported_shape_is_quantized_whole(self, sequence: int, head_size: int) -> None:
        """Head sizes 16, 32, 64 and windows up to 325 tokens are quantized whole."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=sequence, head_size=head_size))
        assert (sum("/scores/" in n for n, _ in plan.activations), "o/MatMul" in _names(plan)) == (2, True)

    @pytest.mark.parametrize(
        ("sequence", "head_size"),
        [(8, 8), (8, 24), (8, 48), (8, 128), (326, 16), (400, 16), (485, 16), (580, 16)],
    )
    def test_unsupported_head_size_or_long_window_stays_float(self, sequence: int, head_size: int) -> None:
        """Unsupported head sizes and windows over 325 tokens keep the attention float."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=sequence, head_size=head_size))
        assert _names(plan) == ["fc1/MatMul", "fc2/MatMul"]
        assert [name for name, _ in plan.activations] == list(plan.weighted)

    def test_graph_without_shapes_counts_as_not_fusable(self) -> None:
        """A graph whose head layout cannot be inferred is treated as not fusable."""
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        del model.graph.input[:]
        model.graph.input.append(helper.make_empty_tensor_value_info("x"))
        for node in model.graph.node:
            if node.op_type == "Reshape":
                node.input[1] = "unknown_shape"  # a runtime shape: inference cannot see the head layout
        plan = quantize.plan_int8(model)
        assert [n for n, _ in plan.activations if "/scores/" in n] == []

    def test_symbolic_token_count_stays_float(self) -> None:
        """An unknown token count is not proven small, so the attention stays float."""
        # The head size is known and fusable; only the token count is not, and an unknown count is not proven small.
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
        model.graph.input[0].type.tensor_type.shape.dim[1].dim_param = "tokens"
        heads = next(init for init in model.graph.initializer if init.name == "heads")
        heads.CopyFrom(numpy_helper.from_array(np.array([1, -1, 2, 16], np.int64), "heads"))
        plan = quantize.plan_int8(model)
        assert [n for n, _ in plan.activations if "/scores/" in n] == []

    @pytest.mark.parametrize(
        ("queries", "keys", "fusable"),
        [
            pytest.param(8, 325, True, id="long-keys-at-the-limit"),
            pytest.param(325, 8, True, id="long-queries-at-the-limit"),
            pytest.param(8, 326, False, id="keys-over-the-limit"),
            pytest.param(326, 8, False, id="queries-over-the-limit"),
        ],
    )
    def test_each_sequence_length_is_held_to_the_token_limit(self, queries: int, keys: int, fusable: bool) -> None:
        """Query and key lengths are each held to the 325-token limit."""
        head = 16

        shapes = {"q": (queries, head), "k": (head, keys), "v": (keys, head), "c": (queries, head)}
        infos = {
            name: helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2, *dims]) for name, dims in shapes.items()
        }
        nodes = [
            helper.make_node("MatMul", ["q", "k"], ["s"], f"{BACKBONE}scores/MatMul"),
            helper.make_node("Softmax", ["s"], ["p"], f"{BACKBONE}Softmax", axis=-1),
            helper.make_node("MatMul", ["p", "v"], ["c"], f"{BACKBONE}context/MatMul"),
        ]
        graph = helper.make_graph(nodes, "cross", [infos["q"], infos["k"], infos["v"]], [infos["c"]])
        (block,) = quantize._attention_blocks(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
        assert block.int8 is fusable

    @pytest.mark.parametrize(
        ("info", "dims"),
        [
            pytest.param(helper.make_tensor_value_info("t", TensorProto.FLOAT, ["n", 4]), (None, 4), id="symbolic"),
            pytest.param(helper.make_empty_tensor_value_info("t"), None, id="no-shape"),
        ],
    )
    def test_unknown_dimensions_are_reported_as_none(self, info: Any, dims: tuple[int | None, ...] | None) -> None:
        """A symbolic dimension reads as None and a missing shape as None."""
        assert quantize._dims({"t": info}, "t") == dims


class TestNanoInt8Plan:
    """Exact counts on a real RFDETRNano export pin the plan to the architecture it was measured on."""

    @pytest.fixture(scope="class")
    @staticmethod
    def nano_onnx(tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Export an untrained RFDETRNano to ONNX on CPU, once for the class.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = nano_onnx(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano
        from rfdetr.utilities.reproducibility import seed_all

        seed_all(7)
        out_dir = tmp_path_factory.mktemp("nano_int8_plan")
        return Path(
            RFDETRNano(pretrain_weights=None, device="cpu").export(
                output_dir=str(out_dir), format="onnx", verbose=False
            )
        )

    @pytest.fixture(scope="class")
    @staticmethod
    def plan(nano_onnx: Path) -> quantize.Int8Plan:
        """The INT8 plan of the exported Nano graph.

        Examples:
            A pytest fixture cannot be called directly.

            >>> plan = plan(nano_onnx)  # doctest: +SKIP
        """
        return quantize.plan_int8(onnx.load(str(nano_onnx)))

    @pytest.mark.parametrize("region", quantize._QUANTIZED_REGIONS)
    def test_every_quantized_region_names_a_node(self, nano_onnx: Path, region: str) -> None:
        """Each quantized-region prefix matches at least one node name of the Nano export.

        The prefixes come from module attribute paths, so a renamed backbone or decoder module leaves a region empty and
        the INT8 planner quantizes nothing there.
        """
        assert any(node.name.startswith(region) for node in onnx.load(str(nano_onnx)).graph.node)

    def test_backbone_encoder_matmul_count(self, nano_onnx: Path) -> None:
        """The Nano backbone encoder exports 96 ``MatMul`` nodes: 12 blocks of 6 projections and 2 attention products.

        A backbone rewrite that should leave the graph unchanged must keep this count, and the INT8 plan with it.
        """
        nodes = onnx.load(str(nano_onnx)).graph.node

        assert sum(n.op_type == "MatMul" and n.name.startswith(quantize._BACKBONE_ENCODER) for n in nodes) == 96

    def test_attention_blocks_left_fp16_are_counted(self, monkeypatch: pytest.MonkeyPatch, nano_onnx: Path) -> None:
        """The info log reports INT8 attention in 9 of 12 Nano backbone blocks."""
        messages: list[str] = []
        monkeypatch.setattr(quantize.logger, "info", messages.append)
        quantize.plan_int8(onnx.load(str(nano_onnx)))
        assert [m.split(";")[0] for m in messages if "stay FP16" in m] == [
            "INT8 attention in 9 of 12 backbone attention blocks"
        ]

    @_requires_onnxconverter_common
    def test_rewritten_graph_is_valid_onnx(self, nano_onnx: Path, plan: quantize.Int8Plan) -> None:
        """The Q/DQ-rewritten Nano graph passes the ONNX checker and strict shape inference."""
        from rfdetr.export._tensorrt.exporter import fp16_source_graph

        with fp16_source_graph(str(nano_onnx)) as fp16_path:
            model = onnx.load(fp16_path)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 1.0))
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_counts(self, plan: quantize.Int8Plan) -> None:
        """The Nano plan has 74 quantized layers and 110 quantized activation inputs."""
        # Weighted: backbone 9 windowed x (q, k, v, o, fc1, fc2) + 3 global x (fc1, fc2) = 60 (the 3-channel patch
        # embedding is still a Conv here) + decoder 2 x 6 + reference-point MLP = 14. Activations: one per weighted
        # node + 4 per INT8 attention block (9 windowed blocks).
        assert (len(plan.weighted), len(plan.activations)) == (74, 74 + 36)

    def test_patch_embedding_joins_the_plan_as_a_matmul(self, nano_onnx: Path) -> None:
        """After the rewrite the Nano patch embedding joins the plan as a MatMul."""
        model = onnx.load(str(nano_onnx))
        assert quantize._patch_embedding_as_matmul(model) == 1
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)
        plan = quantize.plan_int8(model)
        # One more layer and activation input than ``test_counts``: the matrix multiply over 3 * 16 * 16 values.
        assert (len(plan.weighted), len(plan.activations)) == (75, 75 + 36)
        assert [n for n in plan.weighted if n.endswith("patch_embeddings/projection/MatMul")] != []

    @pytest.fixture(scope="class")
    @staticmethod
    def nano_int8(nano_onnx: Path) -> Any:
        """The INT8 source graph of the exported Nano, built once for the class from two random images.

        Examples:
            A pytest fixture cannot be called directly.

            >>> graph = nano_int8(nano_onnx)  # doctest: +SKIP
        """
        shape = [dim.dim_value for dim in onnx.load(str(nano_onnx)).graph.input[0].type.tensor_type.shape.dim]
        samples = np.random.default_rng(5).standard_normal((2, *shape[1:])).astype(np.float32)
        with quantize.int8_source_graph(
            str(nano_onnx),
            calibration_data=samples,
            max_images=2,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            return onnx.load(int8_path)

    @_requires_onnx_and_onnxruntime
    @_requires_onnxconverter_common
    def test_int8_source_graph_quantizes_the_patch_embedding(self, nano_int8: Any) -> None:
        """The Nano INT8 source graph has no backbone Conv left and one Dequantize per planned input."""
        quantized = nano_int8
        onnx.checker.check_model(quantized, full_check=True)
        ops = Counter(node.op_type for node in quantized.graph.node)
        # One DequantizeLinear per planned weight (74 layers and the patch embedding), plus one per activation.
        assert ops["DequantizeLinear"] == 75 + ops["QuantizeLinear"]
        assert [
            n.name
            for n in quantized.graph.node
            if n.op_type == "Conv" and n.name.startswith(quantize._QUANTIZED_REGIONS[0])
        ] == []

    @_requires_onnx_and_onnxruntime
    @_requires_onnxconverter_common
    def test_int8_source_graph_reads_smoothed_norms(self, nano_int8: Any) -> None:
        """The Nano INT8 source graph has 21 norms reading smoothed scale and bias."""
        norms = [node for node in nano_int8.graph.node if node.op_type == "LayerNormalization"]
        # The 21 norms of ``test_every_backbone_norm_a_quantized_projection_reads_is_smoothed``, scale and bias both.
        assert sum(all(re.search(r"_smoothed(_\d+)?$", name) for name in node.input[1:]) for node in norms) == 21

    @_requires_onnx_and_onnxruntime
    @_requires_onnxconverter_common
    def test_int8_source_graph_reads_corrected_mlp_biases(self, nano_int8: Any) -> None:
        """The Nano INT8 source graph has 24 MLP biases replaced by corrected ones."""
        adds = [node for node in nano_int8.graph.node if node.op_type == "Add"]
        # Both projections of the 12 backbone MLPs (``test_every_backbone_mlp_projection_has_its_bias_corrected``).
        assert sum(any(re.search(r"_int8_corrected(_\d+)?$", name) for name in node.input) for node in adds) == 24

    def test_every_backbone_norm_a_quantized_projection_reads_is_smoothed(self, nano_onnx: Path) -> None:
        """Nano's 12 norm2 and 9 windowed norm1 backbone norms are smoothable."""
        model = onnx.load(str(nano_onnx))
        norms = quantize._smoothable_norms(model, quantize.plan_int8(model))
        # norm2 of the 12 blocks (read by fc1) and norm1 of the 9 windowed blocks (read by INT8 q, k, v); the 3 global
        # blocks' norm1 feeds FP16 projections only.
        assert Counter(name.split("/")[-2] for name in norms) == {"norm2": 12, "norm1": 9}

    def test_every_backbone_mlp_projection_has_its_bias_corrected(self, nano_onnx: Path) -> None:
        """Both projections of Nano's 12 backbone MLPs are selected for bias correction."""
        model = onnx.load(str(nano_onnx))
        quantize._patch_embedding_as_matmul(model)
        biases = quantize._mlp_biases(model, quantize.plan_int8(model))
        # Both projections of the 12 backbone MLPs; no attention projection, decoder layer or patch embedding.
        assert Counter("/".join(name.split("/")[-3:-1]) for name in biases) == {"mlp/fc1": 12, "mlp/fc2": 12}

    def test_global_blocks_keep_attention_in_fp16(self, plan: quantize.Int8Plan) -> None:
        """Attention of Nano's global blocks is not quantized."""
        global_attention = [n for n in plan.weighted if any(f"layer.{i}/attention/" in n for i in (3, 6, 9))]
        assert global_attention == []

    def test_decoder_self_attention_is_left_float(self, plan: quantize.Int8Plan) -> None:
        """Nano's decoder self-attention is not quantized."""
        assert [n for n in plan.weighted if "/self_attn/" in n] == []

    def test_every_mlp_is_quantized_in_pairs(self, plan: quantize.Int8Plan) -> None:
        """Each of Nano's 12 backbone MLPs has both fc1 and fc2 quantized."""
        fc1 = {n.replace("fc1", "fc") for n in plan.weighted if "/mlp/fc1/" in n}
        fc2 = {n.replace("fc2", "fc") for n in plan.weighted if "/mlp/fc2/" in n}
        assert (len(fc1), fc1) == (12, fc2)


#: Per model: the FP16 head prefix, then the plan's quantized layers and activation inputs: 24 (12 backbone MLPs) + 6
#: per decoder layer x 4 (4 cross-attention, 2 feed-forward) + 2 reference-point; keypoint adds 6 per layer. No
#: attention is INT8 (every block has >= 577 tokens, above the 325 measured), so layers equal inputs.
_HEAD_MODELS = {
    "segmentation": ("/segmentation_head/", 50, 50),
    "keypoint": ("/keypoint_embed/", 74, 74),
}


class TestSegmentationAndKeypointInt8Plan:
    """The plan of real segmentation and keypoint exports keeps their heads and extra attention in FP16."""

    @pytest.fixture(scope="class")
    @staticmethod
    def exports(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
        """Export an untrained RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge and RFDETRKeypointPreview.

        Done once for the class, to ONNX on CPU. Small and Medium are the segmentation models whose windows fit INT8
        attention; Nano's and Large's do not.

        Examples:
            A pytest fixture cannot be called directly.

            >>> paths = exports(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRKeypointPreview, RFDETRSegLarge, RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall
        from rfdetr.utilities.reproducibility import seed_all

        # The pretrained keypoint model has the person class behind a padding class, which is what puts ScatterND in
        # its graph; the default untrained schema (one class) has none.
        models = (
            ("segmentation", RFDETRSegNano, {}),
            ("segmentation-small", RFDETRSegSmall, {}),
            ("segmentation-medium", RFDETRSegMedium, {}),
            ("segmentation-large", RFDETRSegLarge, {}),
            ("keypoint", RFDETRKeypointPreview, {"num_classes": 2, "num_keypoints_per_class": [0, 17]}),
        )
        paths = {}
        for kind, model_class, settings in models:
            seed_all(7)
            out_dir = tmp_path_factory.mktemp(f"{kind}_int8_plan")
            paths[kind] = Path(
                model_class(pretrain_weights=None, device="cpu", **settings).export(
                    output_dir=str(out_dir), format="onnx", verbose=False
                )
            )
        return paths

    @pytest.fixture(scope="class")
    @staticmethod
    def plans(exports: dict[str, Path]) -> dict[str, quantize.Int8Plan]:
        """The INT8 plan of each exported graph.

        Examples:
            A pytest fixture cannot be called directly.

            >>> plan = plans(exports)["keypoint"]  # doctest: +SKIP
        """
        return {kind: quantize.plan_int8(onnx.load(str(path))) for kind, path in exports.items()}

    @pytest.fixture(scope="class")
    @staticmethod
    def quantized_graphs(exports: dict[str, Path]) -> dict[str, Any]:
        """The INT8 source graph of the segmentation and keypoint exports, calibrated on two random images, loaded.

        Examples:
            A pytest fixture cannot be called directly.

            >>> graph = quantized_graphs(exports)["keypoint"]  # doctest: +SKIP
        """
        graphs = {}
        for kind in ("segmentation", "keypoint"):
            shape = [dim.dim_value for dim in onnx.load(str(exports[kind])).graph.input[0].type.tensor_type.shape.dim]
            samples = np.random.default_rng(5).standard_normal((2, *shape[1:])).astype(np.float32)
            with quantize.int8_source_graph(
                str(exports[kind]),
                calibration_data=samples,
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ) as int8_path:
                graphs[kind] = onnx.load(int8_path)
        return graphs

    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_counts(self, kind: str, plans: dict[str, quantize.Int8Plan]) -> None:
        """The segmentation and keypoint plans have the pinned layer and activation counts."""
        _, weighted, activations = _HEAD_MODELS[kind]
        assert (len(plans[kind].weighted), len(plans[kind].activations)) == (weighted, activations)

    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_heads_stay_float(self, kind: str, plans: dict[str, quantize.Int8Plan]) -> None:
        """Only layers in the backbone and decoder regions are quantized; the head is not."""
        head, _, _ = _HEAD_MODELS[kind]
        quantized = [name for name in plans[kind].weighted if not name.startswith(quantize._QUANTIZED_REGIONS)]
        assert (quantized, [n for n in plans[kind].weighted if n.startswith(head)]) == ([], [])

    @pytest.mark.parametrize(
        ("kind", "weighted", "activations"),
        [("segmentation-small", 86, 122), ("segmentation-medium", 92, 128)],
    )
    def test_windowed_attention_of_small_and_medium_is_int8(
        self, kind: str, weighted: int, activations: int, plans: dict[str, quantize.Int8Plan]
    ) -> None:
        """Seg Small and Medium quantize windowed attention: 4 extra inputs for each of 9 blocks."""
        # Both counts are pinned: nine windowed blocks add 9 * 4 = 36 activation inputs (the scores' two operands and
        # the context's two) on top of the weights, and a difference alone would pass a regression that moved the
        # projections and the attention by the same amount.
        assert (len(plans[kind].weighted), len(plans[kind].activations)) == (weighted, activations)
        assert activations - weighted == 9 * 4

    def test_attention_of_large_stays_fp16(self, plans: dict[str, quantize.Int8Plan]) -> None:
        """Seg Large's windows exceed 325 tokens, so no attention input is added."""
        # Seg Large's 442-token windows are above the 325 limit, so no attention input is added to the weights: both
        # counts are equal, as for Seg Nano in ``test_counts``.
        plan = plans["segmentation-large"]
        assert (len(plan.weighted), len(plan.activations)) == (56, 56)

    @_requires_onnx_and_onnxruntime
    @_requires_onnxconverter_common
    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_int8_source_graph_builds_from_the_real_export(
        self,
        kind: str,
        exports: dict[str, Path],
        plans: dict[str, quantize.Int8Plan],
        quantized_graphs: dict[str, Any],
    ) -> None:
        """The INT8 source graph of a real export is valid opset 19 with the original outputs."""
        model, quantized = onnx.load(str(exports[kind])), quantized_graphs[kind]
        onnx.checker.check_model(quantized, full_check=True)
        opsets = {entry.domain: entry.version for entry in quantized.opset_import}
        assert (opsets[""], [o.name for o in quantized.graph.output]) == (19, [o.name for o in model.graph.output])
        # One DequantizeLinear per weight of each planned layer, plus one per quantized activation tensor (each of which
        # has one QuantizeLinear).
        ops = Counter(node.op_type for node in quantized.graph.node)
        assert ops["DequantizeLinear"] == len(plans[kind].weighted) + ops["QuantizeLinear"]

    @pytest.mark.parametrize(
        "kind", ["segmentation", "segmentation-small", "segmentation-medium", "segmentation-large", "keypoint"]
    )
    def test_patch_embedding_becomes_a_matmul(self, kind: str, exports: dict[str, Path]) -> None:
        """The 12-pixel patch embedding of each real export becomes a MatMul by default."""
        # 12-pixel patches of an RGB image are 432 values, not a multiple of 32; the default rewrites them all the same.
        model = onnx.load(str(exports[kind]))
        assert quantize._patch_embedding_as_matmul(model) == 1
        convolutions = [node.name for node in model.graph.node if node.op_type == "Conv"]
        assert [name for name in convolutions if "patch_embeddings" in name] == []

    @pytest.mark.parametrize(
        "kind", ["segmentation", "segmentation-small", "segmentation-medium", "segmentation-large", "keypoint"]
    )
    def test_a_multiple_of_32_does_not_rewrite_the_432_value_patch(self, kind: str, exports: dict[str, Path]) -> None:
        """With multiple_of 32 the 432-value patch embedding is left untouched."""
        # What the INT8 source graph asks for on the graph it plans: nothing is rewritten, so nothing is quantized.
        model = onnx.load(str(exports[kind]))
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model, multiple_of=quantize._INT8_CONV_CHANNEL_MULTIPLE) == 0
        assert model.SerializeToString() == before

    @_requires_onnx_and_onnxruntime
    @_requires_onnxconverter_common
    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_patch_embedding_is_a_float_matmul_in_the_quantized_graph(
        self, kind: str, quantized_graphs: dict[str, Any]
    ) -> None:
        """In the INT8 graph the 432-value patch embedding is an unquantized FP16 MatMul."""
        # 432-value patches are not quantized (as INT8 they ran no faster than as FP16 on Seg Small and Keypoint
        # Preview and lowered their AP), but they are still a MatMul, so TensorRT has no 3-channel image to pad.
        graph = quantized_graphs[kind].graph
        producers = {output: node for node in graph.node for output in node.output}
        matmuls = [node for node in graph.node if node.name.endswith("patch_embeddings/projection/MatMul")]
        assert len(matmuls) == 1, "the patch embedding is not a single MatMul"
        matmul = matmuls[0]
        weight = {init.name: init for init in graph.initializer}.get(matmul.input[1])
        assert weight is not None, "the weight is not an initializer: the MatMul is quantized"
        assert [node.name for node in graph.node if node.op_type == "Conv" and "patch_embeddings" in node.name] == []
        assert (producers[matmul.input[0]].op_type, weight.data_type) == ("Reshape", onnx.TensorProto.FLOAT16)

    def test_keypoint_instance_self_attention_is_left_float(self, plans: dict[str, quantize.Int8Plan]) -> None:
        """Keypoint instance self-attention is not quantized."""
        assert [n for n in plans["keypoint"].weighted if "_self_attn/" in n] == []

    def test_keypoint_cross_attention_is_quantized(self, plans: dict[str, quantize.Int8Plan]) -> None:
        """The 16 keypoint cross-attention projections are quantized."""
        # 4 layers x (value, offsets, attention weights, output) projections of the keypoint deformable attention.
        assert len([n for n in plans["keypoint"].weighted if "/kp_cross_attn/" in n]) == 16

    @pytest.mark.parametrize(("kind", "op", "count"), [("segmentation", "Resize", 1), ("keypoint", "ScatterND", 17)])
    def test_graph_holds_the_op_the_opset_lift_must_accept(
        self, kind: str, op: str, count: int, exports: dict[str, Path]
    ) -> None:
        """The real exports contain the Resize and ScatterND ops the opset lift must accept."""
        graph = onnx.load(str(exports[kind]), load_external_data=False).graph
        assert sum(node.op_type == op for node in graph.node) == count

    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_every_mlp_is_quantized_in_pairs(self, kind: str, plans: dict[str, quantize.Int8Plan]) -> None:
        """Each of the 12 backbone MLPs has both fc1 and fc2 quantized."""
        fc1 = {n.replace("fc1", "fc") for n in plans[kind].weighted if "/mlp/fc1/" in n}
        fc2 = {n.replace("fc2", "fc") for n in plans[kind].weighted if "/mlp/fc2/" in n}
        assert (len(fc1), fc1) == (12, fc2)

    @_requires_onnxconverter_common
    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_rewritten_graph_is_valid_onnx(
        self, kind: str, exports: dict[str, Path], plans: dict[str, quantize.Int8Plan]
    ) -> None:
        """The Q/DQ-rewritten real export passes the ONNX checker and strict shape inference."""
        from rfdetr.export._tensorrt.exporter import fp16_source_graph

        with fp16_source_graph(str(exports[kind])) as fp16_path:
            model = onnx.load(fp16_path)
        quantize.insert_qdq(model, plans[kind], dict.fromkeys(plans[kind].activations, 1.0))
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("kind", ["segmentation", "keypoint"])
    def test_lifted_graph_gives_the_same_outputs(self, kind: str, exports: dict[str, Path]) -> None:
        """A real export gives the same outputs before and after the opset lift."""
        model = onnx.load(str(exports[kind]))
        shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
        image = np.random.default_rng(3).standard_normal(shape).astype(np.float32)
        before, after = _outputs_before_and_after_lift(model, {model.graph.input[0].name: image})
        assert len(before) == len(after) == 3
        for names, old, new in zip([o.name for o in model.graph.output], before, after):
            np.testing.assert_allclose(new, old, atol=1e-4, err_msg=names)
