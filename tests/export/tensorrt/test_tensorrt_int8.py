# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``export(format="tensorrt", quantization="int8")``.

The request checks and the build wiring run without TensorRT. The quantization plan and the Q/DQ rewrite are pure ONNX
work and run on CPU with ``onnx``, on small synthetic graphs that mirror RF-DETR's naming, plus exact-count checks on
real exported Nano, segmentation and keypoint graphs. Calibration needs onnxruntime and runs in the ``onnx`` integration
job. The end-to-end classes build real FP16 and INT8 engines on a GPU and compare their outputs.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import os
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
from rfdetr.export._runtime import calibration
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import quantize
from rfdetr.export._tensorrt.exporter import (
    _IS_POLYGRAPHY_AVAILABLE,
    _IS_TENSORRT_AVAILABLE,
    TensorRTConfig,
    TensorRTExporter,
)
from rfdetr.export.imports import _IS_ONNXCONVERTER_COMMON_INSTALLED
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou
from tests._markers import _requires_onnx_and_onnxruntime, _requires_onnxruntime

__doctest_requires__ = {
    ("_outputs_before_and_after_lift",): ["onnxruntime"],
    ("_int8_source",): ["onnxruntime", "onnxconverter_common"],
}

# Module-level on purpose: the parametrized ``CASES`` below build ONNX nodes while the module is collected, so a guarded
# import would raise ``NameError`` instead of skipping. Moving every graph builder behind a local import is a follow-up.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

#: Node-name prefix of a block the plan treats as backbone encoder (INT8 attention allowed).
BACKBONE = "/backbone/backbone.0/encoder/encoder/encoder/layer.0/"
#: Node-name prefix of a block the plan treats as decoder (INT8 linears, FP16 attention).
DECODER = "/transformer/decoder/layers.0/"
#: Node-name prefix outside every quantized region (a detection head).
HEAD = "/class_embed/"

_requires_onnxconverter_common = pytest.mark.skipif(
    not _IS_ONNXCONVERTER_COMMON_INSTALLED, reason="onnxconverter-common not installed (the FP16 graph rewrite)"
)
_requires_tensorrt = pytest.mark.skipif(
    not (_IS_TENSORRT_AVAILABLE and _IS_POLYGRAPHY_AVAILABLE), reason="tensorrt/polygraphy not installed"
)


def _weight(name: str, rows: int, cols: int, dtype: type = np.float32, seed: int = 0) -> Any:
    """Return a seeded random ``rows x cols`` weight initializer.

    Examples:
        >>> _weight("w", 2, 3).dims
        [2, 3]
    """
    values = np.random.default_rng(seed).standard_normal((rows, cols)) * 0.1
    return numpy_helper.from_array(values.astype(dtype), name)


def _attention_and_mlp(
    attention_prefix: str,
    *,
    sequence: int,
    head_size: int,
    mlp_prefix: str = BACKBONE,
    second_mlp_prefix: str | None = None,
    activation: str = "Relu",
    dtype: type = np.float32,
    scaled_scores: bool = False,
    second_reader: bool = False,
) -> Any:
    """Build ``x -> attention block -> MLP`` with RF-DETR-style node names, as a ``ModelProto``.

    The attention block projects ``x`` to q/k/v, splits two heads, runs ``MatMul -> Softmax -> MatMul``, projects back;
    the MLP is ``fc1 -> activation -> fc2``. ``activation="Erf"`` writes GELU in its traced erf form, ``"Gelu"`` as the
    opset-20 op.

    Args:
        attention_prefix: Node-name prefix of the attention block.
        sequence: Tokens per sequence.
        head_size: Channels per head (two heads).
        mlp_prefix: Node-name prefix of ``fc1``.
        second_mlp_prefix: Node-name prefix of ``fc2``; defaults to *mlp_prefix*.
        activation: ``"Relu"``, ``"Erf"`` or ``"Gelu"``.
        dtype: Float type of the graph (``np.float32`` or ``np.float16``).
        scaled_scores: Divide the attention scores by a constant before the ``Softmax`` (eager attention's form).
        second_reader: Give the merged attention output a second reader besides the output projection.

    Returns:
        A checked ``ModelProto``.

    Examples:
        >>> model = _attention_and_mlp(BACKBONE, sequence=4, head_size=16)
        >>> sum(node.op_type == "Softmax" for node in model.graph.node)
        1
    """
    width = 2 * head_size
    elem = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    second_mlp_prefix = second_mlp_prefix or mlp_prefix
    inits = [_weight(f"w{n}", width, width, dtype, seed) for seed, n in enumerate("qkvo")]
    inits += [_weight("w1", width, 2 * width, dtype, 5), _weight("w2", 2 * width, width, dtype, 6)]
    inits += [
        numpy_helper.from_array(np.array(v, dtype), n) for n, v in (("sqrt2", 1.4142), ("one", 1.0), ("half", 0.5))
    ]
    inits += [
        numpy_helper.from_array(np.array([1, sequence, 2, head_size], np.int64), "heads"),
        numpy_helper.from_array(np.array([1, sequence, width], np.int64), "merged"),
    ]
    p = attention_prefix
    nodes = []
    for name in "qkv":
        nodes += [
            helper.make_node("MatMul", ["x", f"w{name}"], [f"{name}0"], f"{p}{name}/MatMul"),
            helper.make_node("Reshape", [f"{name}0", "heads"], [f"{name}1"], f"{p}{name}/Reshape"),
        ]
    nodes += [
        helper.make_node("Transpose", ["q1"], ["q2"], f"{p}q/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Transpose", ["k1"], ["k2"], f"{p}k/Transpose", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["v1"], ["v2"], f"{p}v/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("MatMul", ["q2", "k2"], ["s0" if scaled_scores else "s"], f"{p}scores/MatMul"),
        *([helper.make_node("Div", ["s0", "sqrt2"], ["s"], f"{p}scale/Div")] if scaled_scores else []),
        helper.make_node("Softmax", ["s"], ["p"], f"{p}Softmax", axis=-1),
        helper.make_node("MatMul", ["p", "v2"], ["c"], f"{p}context/MatMul"),
        helper.make_node("Transpose", ["c"], ["c1"], f"{p}merge/Transpose", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["c1", "merged"], ["c2"], f"{p}merge/Reshape"),
        helper.make_node("MatMul", ["c2", "wo"], ["a"], f"{p}o/MatMul"),
        helper.make_node("MatMul", ["a", "w1"], ["h"], f"{mlp_prefix}fc1/MatMul"),
    ]
    if activation == "Erf":
        nodes += [
            helper.make_node("Div", ["h", "sqrt2"], ["h1"], f"{mlp_prefix}act/Div"),
            helper.make_node("Erf", ["h1"], ["h2"], f"{mlp_prefix}act/Erf"),
            helper.make_node("Add", ["h2", "one"], ["h3"], f"{mlp_prefix}act/Add"),
            helper.make_node("Mul", ["h", "h3"], ["h4"], f"{mlp_prefix}act/Mul"),
            helper.make_node("Mul", ["h4", "half"], ["g"], f"{mlp_prefix}act/Mul_1"),
        ]
    else:
        nodes.append(helper.make_node(activation, ["h"], ["g"], f"{mlp_prefix}act/{activation}"))
    nodes.append(helper.make_node("MatMul", ["g", "w2"], ["y"], f"{second_mlp_prefix}fc2/MatMul"))
    outputs = [helper.make_tensor_value_info("y", elem, [1, sequence, width])]
    if second_reader:
        nodes.append(helper.make_node("Identity", ["c2"], ["attention_out"], f"{p}tap/Identity"))
        outputs.append(helper.make_tensor_value_info("attention_out", elem, [1, sequence, width]))
    graph = helper.make_graph(
        nodes, "attention_mlp", [helper.make_tensor_value_info("x", elem, [1, sequence, width])], outputs, inits
    )
    opset = 20 if activation == "Gelu" else 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model, full_check=True)
    return model


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


def _names(plan: quantize.Int8Plan) -> list[str]:
    """Return the last two path segments of each weighted node, e.g. ``"fc1/MatMul"``.

    Examples:
        >>> _names(quantize.Int8Plan(weighted=("/a/b/fc1/MatMul",), activations=()))
        ['fc1/MatMul']
    """
    return ["/".join(name.split("/")[-2:]) for name in plan.weighted]


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


def _node(op_type: str, inputs: list[str], output: str, name: str = "", **attributes: Any) -> Any:
    """Return a one-output ``NodeProto`` called *name*; an empty name stands for an unnamed layer.

    Examples:
        >>> _node("Add", ["a", "b"], "c", "sum").output[0]
        'c'
    """
    return helper.make_node(op_type, inputs, [output], name, **attributes)


def _weights(*names: str, dims: tuple[int, ...] = (4, 4)) -> dict[str, Any]:
    """Return a constant-weight map holding a zero initializer of *dims* for each of *names*.

    Examples:
        >>> sorted(_weights("w", "v", dims=(2, 3)))
        ['v', 'w']
    """
    return {name: numpy_helper.from_array(np.zeros(dims, np.float32), name) for name in names}


def _readers(nodes: list[Any]) -> dict[str, list[Any]]:
    """Map every tensor name to the nodes that read it, as ``plan_int8`` builds it.

    Examples:
        >>> add = _node("Add", ["a", "b"], "c")
        >>> [node.op_type for node in _readers([add])["a"]]
        ['Add']
    """
    readers: dict[str, list[Any]] = {}
    for node in nodes:
        for tensor in node.input:
            readers.setdefault(tensor, []).append(node)
    return readers


def _producers(nodes: list[Any]) -> dict[str, Any]:
    """Map every tensor name to the node that writes it.

    Examples:
        >>> _producers([_node("Add", ["a", "b"], "c", "sum")])["c"].name
        'sum'
    """
    return {output: node for node in nodes for output in node.output}


class TestWeightLookup:
    """``_weight_name`` says which nodes the plan can refer to; ``_holds_weight`` says which ones read a constant."""

    #: Nodes as (node, weight dims). ``w`` is the only constant; ``act`` is an activation.
    CASES = [
        pytest.param(_node("MatMul", ["x", "w"], "y", "fc"), (4, 4), True, True, id="matmul"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc"), (4, 4), True, True, id="gemm"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc", transB=1), (4, 4), True, True, id="gemm-trans-b"),
        pytest.param(_node("Gemm", ["x", "w"], "y", "fc", transA=1), (4, 4), False, True, id="gemm-trans-a"),
        pytest.param(_node("Conv", ["x", "w"], "y", "conv"), (4, 3, 1, 1), True, True, id="conv"),
        pytest.param(_node("Conv", ["x", "w"], "y", "conv"), (4, 4), False, True, id="conv-with-a-matrix"),
        pytest.param(_node("MatMul", ["x", "w"], "y", "fc"), (2, 4, 4), False, True, id="matmul-with-a-batch"),
        pytest.param(_node("MatMul", ["x", "w"], "y", ""), (4, 4), False, True, id="unnamed"),
        pytest.param(_node("MatMul", ["x", "act"], "y", "bmm"), (4, 4), False, False, id="two-activations"),
        pytest.param(_node("Add", ["x", "w"], "y", "add"), (4, 4), False, False, id="not-a-weight-op"),
        pytest.param(_node("MatMul", ["x"], "y", "fc"), (4, 4), False, False, id="one-input"),
    ]

    @pytest.mark.parametrize(("node", "dims", "plannable", "_reads"), CASES)
    def test_weight_name_is_the_weight_only_for_a_plannable_node(
        self, node: Any, dims: tuple[int, ...], plannable: bool, _reads: bool
    ) -> None:
        """_weight_name returns the weight only for a named, untransposed-A weight layer."""
        assert (quantize._weight_name(node, _weights("w", dims=dims)) == "w") is plannable

    @pytest.mark.parametrize(("node", "dims", "_plannable", "reads"), CASES)
    def test_holds_weight_ignores_everything_but_the_constant(
        self, node: Any, dims: tuple[int, ...], _plannable: bool, reads: bool
    ) -> None:
        """_holds_weight is true for any weight op that reads the constant, named or not."""
        assert quantize._holds_weight(node, _weights("w", dims=dims)) is reads

    def test_weight_reached_through_an_identity_is_constant(self) -> None:
        """An Identity or Constant output counts as a constant weight."""
        graph = helper.make_graph(
            [_node("Identity", ["w"], "w_alias"), _node("Constant", [], "c", value=_weights("c")["c"])],
            "g",
            [],
            [],
            [_weights("w")["w"]],
        )
        assert sorted(quantize._constant_weights(graph)) == ["c", "w", "w_alias"]


class TestGraphWalkers:
    """The helpers that find an attention block's neighbours, on graphs of a few nodes."""

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Mul", "Sub", "Where"])
    def test_scores_are_found_behind_scaling_and_masking(self, op: str) -> None:
        """The scores multiply is found behind scale, mask and cast ops."""
        nodes = [
            _node("MatMul", ["q", "k"], "s0", "scores"),
            _node(op, ["s0", "mask"], "s", "scale"),
            _node("Softmax", ["s"], "p", "softmax"),
        ]
        assert quantize._scores_matmul(nodes[2], _producers(nodes), {}).name == "scores"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-producer"),
            pytest.param(
                [_node("MatMul", ["q", "w"], "s", "projection"), _node("Softmax", ["s"], "p", "softmax")],
                id="weight-multiply",
            ),
            pytest.param(
                [
                    _node("Relu", ["s0"], "s", "act"),
                    _node("MatMul", ["q", "k"], "s0", "scores"),
                    _node("Softmax", ["s"], "p", "softmax"),
                ],
                id="blocked-by-another-op",
            ),
        ],
    )
    def test_scores_are_not_found_where_they_are_not_a_multiply_of_activations(self, nodes: list[Any]) -> None:
        """No scores multiply is found for a missing producer, a weight multiply or a blocking op."""
        softmax = next(node for node in nodes if node.op_type == "Softmax")
        assert quantize._scores_matmul(softmax, _producers(nodes), _weights("w")) is None

    def test_scores_are_found_when_a_node_is_reached_twice(self) -> None:
        """A node reached by two paths does not break the scores search."""
        nodes = [
            _node("MatMul", ["q", "k"], "s0", "scores"),
            _node("Add", ["s0", "s0"], "s", "twice"),
            _node("Softmax", ["s"], "p", "softmax"),
        ]
        assert quantize._scores_matmul(nodes[2], _producers(nodes), {}).name == "scores"

    @pytest.mark.parametrize("passthrough", [[], ["Cast"], ["Identity"], ["Cast", "Identity"]])
    def test_context_is_found_behind_casts(self, passthrough: list[str]) -> None:
        """The context multiply is found behind Cast and Identity ops."""
        nodes, tensor = [_node("Softmax", ["s"], "p0", "softmax")], "p0"
        for index, op in enumerate(passthrough):
            nodes.append(_node(op, [tensor], f"p{index + 1}", f"pass{index}"))
            tensor = f"p{index + 1}"
        nodes.append(_node("MatMul", [tensor, "v"], "c", "context"))
        assert quantize._context_matmul(nodes[0], _readers(nodes)).name == "context"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-reader"),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Relu", ["p"], "r", "act"),
                    _node("MatMul", ["r", "v"], "c", "context"),
                ],
                id="blocked-by-another-op",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("MatMul", ["p", "v"], "c", "context"),
                    _node("Identity", ["p"], "tap", "tap"),
                ],
                id="two-readers",
            ),
            pytest.param(
                [_node("Softmax", ["s"], "p", "softmax"), _node("MatMul", ["v", "p"], "c", "context")],
                id="probabilities-as-the-second-operand",
            ),
        ],
    )
    def test_context_is_not_found_unless_probabilities_feed_one_multiply(self, nodes: list[Any]) -> None:
        """No context multiply is found unless the probabilities feed exactly one first operand."""
        assert quantize._context_matmul(nodes[0], _readers(nodes)) is None

    def test_projection_is_traced_through_unnamed_nodes(self) -> None:
        """A projection is traced through unnamed Reshape and Transpose nodes."""
        weights = {"w": numpy_helper.from_array(np.zeros((4, 4), np.float32), "w")}
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["q"], "q_proj"),
            helper.make_node("Reshape", ["q", "shape"], ["q_heads"]),
            helper.make_node("Transpose", ["q_heads"], ["q_t"]),
        ]
        producers = {output: node for node in nodes for output in node.output}
        assert quantize._trace_projection("q_t", producers, weights) == "q_proj"

    @pytest.mark.parametrize(
        "follower",
        ["Abs", "Add", "Cast", "Clip", "Concat", "Div", "Dropout", "Identity", "Mul", "Relu", "Reshape", "Slice", "Sub"]
        + ["Transpose", "Where"],
    )
    def test_softmax_reaches_a_multiply_through_any_op(self, follower: str) -> None:
        """A Softmax reaches an activation multiply through any follower op."""
        nodes = [
            _node("Softmax", ["s"], "p", "softmax"),
            _node(follower, ["p"], "p1", "follow"),
            _node("MatMul", ["p1", "v"], "c", "context"),
        ]
        assert quantize._feeds_activation_matmul(nodes[0], _readers(nodes), {}) is True

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([_node("Softmax", ["s"], "p", "softmax")], id="no-reader"),
            pytest.param(
                [_node("Softmax", ["s"], "p", "softmax"), _node("MatMul", ["p", "w"], "c", "projection")],
                id="weight-multiply",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("MatMul", ["p", "w"], "m", "projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-projection",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Gemm", ["p", "w"], "m", "gemm_projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-gemm",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Conv", ["p", "w"], "m", "conv_projection"),
                    _node("MatMul", ["m", "v"], "c", "later_block"),
                ],
                id="blocked-by-a-convolution",
            ),
            pytest.param(
                [
                    _node("Softmax", ["s"], "p", "softmax"),
                    _node("Identity", ["p"], "a", "a"),
                    _node("Identity", ["p"], "b", "b"),
                    _node("Mul", ["a", "b"], "m", "mul"),
                ],
                id="reader-reached-twice",
            ),
        ],
    )
    def test_softmax_that_never_reaches_a_multiply_of_activations_is_not_attention(self, nodes: list[Any]) -> None:
        """A Softmax that reaches no activation multiply is not attention."""
        assert quantize._feeds_activation_matmul(nodes[0], _readers(nodes), _weights("w")) is False

    @pytest.mark.parametrize("op", ["Add", "Cast", "Div", "Identity", "Mul", "Transpose"])
    def test_projection_is_found_behind_each_passthrough_op(self, op: str) -> None:
        """A projection is found behind each pass-through op."""
        nodes = [_node("MatMul", ["x", "w"], "q0", "q_proj"), _node(op, ["q0", "bias"], "q", "after")]
        assert quantize._trace_projection("q", _producers(nodes), _weights("w", "bias")) == "q_proj"

    @pytest.mark.parametrize("op", ["Reshape", "Squeeze", "Unsqueeze"])
    def test_projection_is_found_through_the_data_input_of_a_shape_op(self, op: str) -> None:
        """A projection is traced through the data input of a shape op, not its shape operand."""
        # The shape operand comes from another projection and must not be followed.
        nodes = [
            _node("MatMul", ["x", "w"], "q0", "q_proj"),
            _node("MatMul", ["x", "w"], "shape", "shape_proj"),
            _node(op, ["q0", "shape"], "q", "after"),
        ]
        assert quantize._trace_projection("q", _producers(nodes), _weights("w")) == "q_proj"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([], id="graph-input"),
            pytest.param([_node("Shape", ["x"], "q", "shape")], id="shape-arithmetic"),
            pytest.param([_node("Relu", ["q0"], "q", "act"), _node("MatMul", ["x", "w"], "q0", "q_proj")], id="act"),
        ],
    )
    def test_projection_is_not_found_behind_other_ops(self, nodes: list[Any]) -> None:
        """No projection is found behind a graph input, shape arithmetic or an activation."""
        assert quantize._trace_projection("q", _producers(nodes), _weights("w")) is None

    @pytest.mark.parametrize("hops", [0, 1, 3])
    def test_output_projection_is_found_within_three_shape_ops(self, hops: int) -> None:
        """The output projection is found within three shape ops of the context."""
        nodes, tensor = [], "ctx"
        for index in range(hops):
            nodes.append(_node("Transpose" if index % 2 else "Reshape", [tensor, "shape"], f"t{index}", f"hop{index}"))
            tensor = f"t{index}"
        nodes.append(_node("MatMul", [tensor, "w"], "y", "out_proj"))
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w", "shape")) == "out_proj"

    @pytest.mark.parametrize(
        "nodes",
        [
            pytest.param([], id="no-reader"),
            pytest.param(
                [_node("MatMul", ["ctx", "w"], "a", "p1"), _node("MatMul", ["ctx", "w"], "b", "p2")], id="two-readers"
            ),
            pytest.param([_node("Relu", ["ctx"], "a", "act"), _node("MatMul", ["a", "w"], "y", "p")], id="other-op"),
            pytest.param([_node("MatMul", ["x", "ctx"], "y", "ctx_as_weight")], id="context-is-the-second-operand"),
        ],
    )
    def test_output_projection_is_not_found_unless_one_weight_reads_the_context(self, nodes: list[Any]) -> None:
        """No output projection is found unless exactly one weight multiply reads the context."""
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w")) is None

    def test_output_projection_is_not_found_beyond_three_shape_ops(self) -> None:
        """The output projection search stops after three shape ops."""
        nodes = [_node("Reshape", [f"t{i}", "shape"], f"t{i + 1}", f"hop{i}") for i in range(4)]
        nodes[0].input[0] = "ctx"
        nodes.append(_node("MatMul", ["t4", "w"], "y", "out_proj"))
        context = _node("MatMul", ["p", "v"], "ctx", "context")
        assert quantize._output_projection(context, _readers(nodes), _weights("w", "shape")) is None


class TestGeluConsumers:
    """``_feeds_gelu_into`` lists what reads a GELU output, so a projection can wait for a quantized consumer."""

    @staticmethod
    def _chain(activation: str, *, after: list[str] | None = None, consumer_name: str = "fc2") -> tuple[list[Any], Any]:
        """Return ``fc1 -> GELU -> [ops in *after*] -> consumer`` as a node list and the ``fc1`` node.

        Examples:
            >>> nodes, fc1 = TestGeluConsumers._chain("Gelu", after=["Identity"])
            >>> [node.op_type for node in nodes]
            ['MatMul', 'Gelu', 'Identity', 'MatMul']
        """
        nodes = [_node("MatMul", ["x", "w1"], "h", "fc1")]
        if activation == "Erf":
            nodes += [
                _node("Div", ["h", "sqrt2"], "h1", "div"),
                _node("Erf", ["h1"], "h2", "erf"),
                _node("Add", ["h2", "one"], "h3", "add"),
                _node("Mul", ["h", "h3"], "h4", "mul"),
                _node("Mul", ["h4", "half"], "g", "mul_half"),
            ]
        else:
            nodes.append(_node(activation, ["h"], "g", "act"))
        tensor = "g"
        for index, op in enumerate(after or []):
            nodes.append(_node(op, [tensor], f"g{index}", f"after{index}"))
            tensor = f"g{index}"
        nodes.append(_node("MatMul", [tensor, "w2"], "y", consumer_name))
        return nodes, nodes[0]

    @pytest.mark.parametrize("activation", ["Erf", "Gelu"])
    def test_both_gelu_forms_lead_to_the_second_projection(self, activation: str) -> None:
        """Erf and Gelu forms both lead from fc1 to the second projection."""
        nodes, fc1 = self._chain(activation)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_relu_is_not_a_gelu(self) -> None:
        """A ReLU activation has no GELU consumers."""
        nodes, fc1 = self._chain("Relu")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == set()

    @pytest.mark.parametrize("after", [["Identity"], ["Cast", "Reshape"]])
    def test_any_op_between_the_gelu_and_its_consumer_counts_as_a_consumer(self, after: list[str]) -> None:
        """The first op after the GELU counts as its consumer."""
        nodes, fc1 = self._chain("Gelu", after=after)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"after0"}

    @pytest.mark.parametrize("consumer_name", ["", "fc2"])
    def test_a_consumer_is_listed_by_name_even_when_it_cannot_be_planned(self, consumer_name: str) -> None:
        """A consumer is listed by its name, empty included."""
        nodes, fc1 = self._chain("Gelu", consumer_name=consumer_name)
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {consumer_name}

    def test_a_consumer_reading_the_gelu_twice_is_listed_once(self) -> None:
        """A consumer reading the GELU output twice is listed once."""
        nodes, fc1 = self._chain("Gelu")
        nodes[-1].input[0] = "g"
        nodes[-1].input.append("g")
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}

    def test_the_branch_that_skips_the_gelu_is_not_a_consumer(self) -> None:
        """A reader of the pre-GELU tensor is not a GELU consumer."""
        nodes, fc1 = self._chain("Gelu")
        nodes.append(_node("MatMul", ["h", "w2"], "skip", "skip_fc"))
        assert quantize._feeds_gelu_into(fc1, _readers(nodes), _weights("w2")) == {"fc2"}


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


def _quantized_with(mutate: Any) -> tuple[Any, quantize.Int8Plan]:
    """Plan on the FP32 attention-and-MLP graph, then quantize its FP16 twin, after *mutate* edited both.

    Examples:
        >>> model, plan = _quantized_with(lambda m: None)
        >>> onnx.checker.check_model(model, full_check=True)
    """
    fp32 = _attention_and_mlp(BACKBONE, sequence=8, head_size=16)
    fp16 = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
    mutate(fp32)
    mutate(fp16)
    plan = quantize.plan_int8(fp32)
    quantize.insert_qdq(fp16, plan, dict.fromkeys(plan.activations, 2.0))
    return fp16, plan


class TestInsertQdq:
    """The rewritten FP16 graph is valid opset-19 ONNX with FP16 scales and per-channel INT8 weights."""

    @pytest.fixture
    def quantized(self) -> tuple[Any, quantize.Int8Plan]:
        """Plan on the FP32 graph, then quantize its FP16 twin, as ``int8_source_graph`` does.

        Examples:
            A pytest fixture cannot be called directly.

            >>> model, plan = quantized()  # doctest: +SKIP
        """
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        return model, plan

    @pytest.fixture
    def ranged(self) -> tuple[Any, quantize.Int8Plan, dict[str, float]]:
        """Like ``quantized``, but each activation tensor has its own range; also returns tensor -> range.

        Examples:
            A pytest fixture cannot be called directly.

            >>> model, plan, by_tensor = ranged()  # doctest: +SKIP
        """
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        tensors = sorted({quantize._input_tensor(model.graph, *key) for key in plan.activations})
        by_tensor = {tensor: 0.5 * (index + 1) for index, tensor in enumerate(tensors)}
        ranges = {key: by_tensor[quantize._input_tensor(model.graph, *key)] for key in plan.activations}
        quantize.insert_qdq(model, plan, ranges)
        return model, plan, by_tensor

    def test_weight_shared_by_two_layers_is_quantized_for_each(self) -> None:
        """A weight read by two layers gets its own quantize pair per layer."""

        def share(model: Any) -> None:
            next(n for n in model.graph.node if n.name.endswith("k/MatMul")).input[1] = "wq"

        model, plan = _quantized_with(share)
        by_name = {n.name: n for n in model.graph.node}
        readers = {by_name[f"{BACKBONE}{layer}/MatMul"].input[1] for layer in "qk"}
        assert len(readers) == 2 and all(r.endswith("_dequantized") for r in readers)
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("tensor", ["c2", "wq", "x"])
    def test_quantized_tensor_that_is_also_a_graph_output_stays_an_output(self, tensor: str) -> None:
        """A quantized tensor that is also a graph output stays an output of valid ONNX."""
        shapes = {"c2": [1, 8, 32], "x": [1, 8, 32], "wq": [32, 32]}

        def expose(model: Any) -> None:
            elem = model.graph.output[0].type.tensor_type.elem_type
            model.graph.output.append(helper.make_tensor_value_info(tensor, elem, shapes[tensor]))

        model, _ = _quantized_with(expose)
        assert tensor in [o.name for o in model.graph.output]
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_tensor_read_inside_a_subgraph_keeps_its_original_binding(self) -> None:
        """A tensor read inside a subgraph keeps its original name there."""

        def capture(model: Any) -> None:
            tap = helper.make_node("Identity", ["c2"], ["inner"], "inner/Identity")
            branch = helper.make_graph([tap], "branch", [], [helper.make_tensor_value_info("inner", 10, [1, 8, 32])])
            model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
            model.graph.node.append(
                helper.make_node("If", ["flag"], ["picked"], "if", then_branch=branch, else_branch=branch)
            )
            model.graph.output.append(helper.make_tensor_value_info("picked", 10, [1, 8, 32]))

        model, _ = _quantized_with(capture)
        if_node = next(n for n in model.graph.node if n.op_type == "If")
        assert if_node.attribute[0].g.node[0].input[0] == "c2"
        onnx.checker.check_model(model, full_check=True)

    def test_every_planned_input_reads_a_dequantize(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        """Every planned activation and weight input is read through a DequantizeLinear."""
        model, plan, _ = ranged
        producer = {output: node for node in model.graph.node for output in node.output}
        by_name = {node.name: node for node in model.graph.node}
        read = [producer[by_name[name].input[index]].op_type for name, index in plan.activations]
        read += [producer[by_name[name].input[1]].op_type for name in plan.weighted]
        assert set(read) == {"DequantizeLinear"}

    def test_activation_scale_is_its_range_over_127(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        """An activation scale is its measured range over 127, stored in FP16."""
        model, plan, by_tensor = ranged
        producer = {output: node for node in model.graph.node for output in node.output}
        by_name = {node.name: node for node in model.graph.node}
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        found = {}
        for name, index in plan.activations:
            dequantize = producer[by_name[name].input[index]]
            quantize_node = producer[dequantize.input[0]]
            found[quantize_node.input[0]] = (float(inits[quantize_node.input[1]]), float(inits[dequantize.input[1]]))
        expected = {t: (float(np.float16(r / 127)),) * 2 for t, r in by_tensor.items()}
        assert found == expected

    def test_activation_zero_point_is_int8_zero(self, ranged: tuple[Any, quantize.Int8Plan, dict]) -> None:
        """Activation zero points are INT8 zero."""
        model, _, _ = ranged
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        points = {
            node.input[2] for node in model.graph.node if node.op_type == "QuantizeLinear" and len(node.input) > 2
        }
        assert {(inits[name].dtype, int(inits[name])) for name in points} == {(np.dtype(np.int8), 0)}

    def test_convolution_weight_is_quantized_per_output_channel(self) -> None:
        """A convolution weight is quantized per output channel on axis 0."""
        graph = helper.make_graph(
            [helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv")],
            "conv",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 32, 4, 4])],
            [helper.make_tensor_value_info("mixed", TensorProto.FLOAT, [1, 16, 4, 4])],
            [numpy_helper.from_array(np.ones((16, 32, 1, 1), np.float32), "mix")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        plan = quantize.plan_int8(model)
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 1.0))
        inits = {init.name: init for init in model.graph.initializer}
        weight_dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits)
        assert ([a.i for a in weight_dq.attribute if a.name == "axis"], list(inits[weight_dq.input[1]].dims)) == (
            [0],
            [16],
        )

    def test_result_is_valid_onnx(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """The rewritten graph passes the ONNX checker and strict shape inference."""
        model, _ = quantized
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_opset_is_lifted(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """The rewritten graph is at opset 19."""
        model, _ = quantized
        assert [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")] == [19]

    def test_one_quantize_per_activation_tensor(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Each distinct activation tensor is quantized once and shared by its readers."""
        model, _ = quantized
        quantized_inputs = [node.input[0] for node in model.graph.node if node.op_type == "QuantizeLinear"]
        # x feeds q, k and v through one pair; then the four batched-multiply inputs, o's input, fc1's and fc2's.
        assert sorted(quantized_inputs) == sorted(["x", "q2", "k2", "p", "v2", "c2", "a", "g"])

    def test_scales_are_fp16(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Every quantize and dequantize scale is FP16."""
        model, _ = quantized
        inits = {init.name: init for init in model.graph.initializer}
        scales = {node.input[1] for node in model.graph.node if node.op_type in ("QuantizeLinear", "DequantizeLinear")}
        assert {inits[name].data_type for name in scales} == {TensorProto.FLOAT16}

    def test_weights_are_int8_per_output_channel(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Weights are stored INT8 and dequantized per output channel on axis 1."""
        model, plan = quantized
        inits = {init.name: init for init in model.graph.initializer}
        weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits]
        assert len(weight_dq) == len(plan.weighted)
        assert {(inits[n.input[0]].data_type, n.attribute[0].i) for n in weight_dq} == {(TensorProto.INT8, 1)}

    def test_weight_zero_point_is_int8_zero(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """Weight zero points are INT8 zero."""
        model, _ = quantized
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        weight_dq = [n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] in inits]
        points = [inits[n.input[2]] for n in weight_dq]
        assert {(point.dtype, int(np.abs(point).max())) for point in points} == {(np.dtype(np.int8), 0)}

    def test_dequantized_weight_matches_the_original(self, quantized: tuple[Any, quantize.Int8Plan]) -> None:
        """A dequantized weight is within half a quantization step of the original."""
        model, _ = quantized
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0].startswith("w1"))
        scales = inits[dq.input[1]].astype(np.float32)  # one per output channel (axis 1)
        restored = inits[dq.input[0]].astype(np.float32) * scales
        original = numpy_helper.to_array(_weight("w1", 32, 64, np.float16, 5)).astype(np.float32)
        # Rounded against the stored FP16 scale, every weight is within half a step of its original value.
        assert float((np.abs(restored - original) / scales).max()) <= 0.5 + 1e-3

    def test_scales_stay_positive_in_fp16(self) -> None:
        """Scales never fall below the smallest normal FP16 value, even for a near-dead channel."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        weight = next(init for init in model.graph.initializer if init.name == "w1")
        values = numpy_helper.to_array(weight).copy()
        values[:, 0] = 2e-6  # a near-dead output channel
        weight.CopyFrom(numpy_helper.from_array(values, "w1"))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 0.0))
        inits = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        scales = [inits[n.input[1]] for n in model.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        # Not merely positive: a subnormal FP16 scale is as unusable as zero, so the floor is the smallest normal value.
        assert min(float(np.min(scale)) for scale in scales) >= np.finfo(np.float16).tiny

    def test_initializer_read_only_inside_a_subgraph_survives(self) -> None:
        """An initializer read only inside a subgraph survives the rewrite."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        model.graph.initializer.append(numpy_helper.from_array(np.ones(32, np.float16), "bias"))
        branch = helper.make_graph(
            [helper.make_node("Add", ["y", "bias"], ["z"], "inner/Add")],
            "branch",
            [],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT16, [1, 8, 32])],
        )
        model.graph.input.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
        model.graph.node.append(helper.make_node("If", ["cond"], ["out"], "if", then_branch=branch, else_branch=branch))
        model.graph.output.append(helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 8, 32]))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize(("transposed", "axis"), [(0, 1), (1, 0)])
    def test_gemm_weight_axis_follows_trans_b(self, transposed: int, axis: int) -> None:
        """A Gemm weight is quantized along the output-channel axis chosen by transB."""
        shape = (4, 8) if transposed else (8, 4)
        graph = helper.make_graph(
            [helper.make_node("Gemm", ["x", "w"], ["y"], f"{DECODER}linear/Gemm", transB=transposed)],
            "gemm",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [2, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [2, 4])],
            [_weight("w", *shape, np.float16)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        plan = quantize.Int8Plan(weighted=(f"{DECODER}linear/Gemm",), activations=((f"{DECODER}linear/Gemm", 0),))
        quantize.insert_qdq(model, plan, {(f"{DECODER}linear/Gemm", 0): 1.0})
        onnx.checker.check_model(model, full_check=True)
        weight_dq = next(n for n in model.graph.node if n.op_type == "DequantizeLinear" and n.input[0] != "x_int8")
        assert weight_dq.attribute[0].i == axis

    def test_generated_names_avoid_existing_tensors(self) -> None:
        """Names the rewrite generates do not collide with existing tensors or initializers."""
        plan = quantize.plan_int8(_attention_and_mlp(BACKBONE, sequence=8, head_size=16))
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=np.float16)
        # Claim the names the rewrite would pick first, as a graph output and an unused initializer.
        model.graph.output.append(helper.make_tensor_value_info("x_int8", TensorProto.FLOAT16, [1, 8, 32]))
        model.graph.node.append(helper.make_node("Identity", ["y"], ["x_int8"], "claim"))
        model.graph.initializer.append(numpy_helper.from_array(np.array(0, np.int8), "int8_zero_point"))
        quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        onnx.checker.check_model(model, full_check=True)
        outputs = [o for n in model.graph.node for o in n.output]
        assert len(outputs) == len(set(outputs))


class TestCalibrationGraph:
    """The probe graph that measures ranges stays valid ONNX whatever names the model already uses."""

    def test_probe_names_avoid_initializer_outputs(self) -> None:
        """Probe output names avoid names already used by graph outputs and initializers."""
        graph = helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"], "relu")],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("x__absmax", TensorProto.FLOAT, []),
            ],
            [numpy_helper.from_array(np.array(1.0, np.float32), "x__absmax")],
        )
        probe = quantize._calibration_graph(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), ["x"]
        )
        onnx.checker.check_model(probe, full_check=True)


#: The reductions whose ``axes`` moved from an attribute to an input in opset 18; spelled out so a shrunken set fails.
REDUCTIONS = [
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
]


def _outputs_before_and_after_lift(model: Any, feeds: dict[str, np.ndarray]) -> tuple[list[Any], list[Any]]:
    """Run *model* on onnxruntime's CPU provider as given and again after ``quantize._lift_opset`` raised it to 19.

    Args:
        model: A ``ModelProto`` at opset 17; left untouched (the lift runs on a copy).
        feeds: Graph inputs by name.

    Returns:
        The outputs at opset 17 and the outputs at opset 19.

    Examples:
        >>> node = helper.make_node("Relu", ["x"], ["y"])
        >>> graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])])
        >>> model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        >>> before, after = _outputs_before_and_after_lift(model, {"x": np.array([-1.0, 2.0], np.float32)})
        >>> before[0].tolist(), after[0].tolist()
        ([0.0, 2.0], [0.0, 2.0])
    """
    import onnxruntime

    original, lifted = type(model)(), type(model)()
    original.CopyFrom(model)
    lifted.CopyFrom(model)
    quantize._lift_opset(lifted)
    outputs = []
    for candidate in (original, lifted):
        candidate.ir_version = 9  # the newest IR onnx writes can be newer than the installed onnxruntime reads
        session = onnxruntime.InferenceSession(candidate.SerializeToString(), providers=["CPUExecutionProvider"])
        outputs.append(session.run(None, feeds))
    return outputs[0], outputs[1]


class TestLiftOpset:
    """FP16 scales need opset 19; lifting a 17 graph keeps every op's meaning or refuses."""

    def test_reduction_inside_a_subgraph_moves_axes_to_an_input(self) -> None:
        """Lifting to opset 19 moves a subgraph reduction's axes attribute to an input."""
        reduce = helper.make_node("ReduceMax", ["x"], ["r"], "inner/ReduceMax", axes=[1], keepdims=0)
        branch = helper.make_graph([reduce], "branch", [], [helper.make_tensor_value_info("r", TensorProto.FLOAT, [2])])
        if_node = helper.make_node("If", ["cond"], ["y"], "if", then_branch=branch, else_branch=branch)
        graph = helper.make_graph(
            [if_node],
            "lift",
            [
                helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        quantize._lift_opset(model)
        onnx.checker.check_model(model, full_check=True)
        inner = model.graph.node[0].attribute[0].g.node[0]
        assert (len(inner.input), [a.name for a in inner.attribute]) == (2, ["keepdims"])

    @staticmethod
    def _reduction_model(op: str, opset: int, *, axes: list[int] | None, taken: str | None = None) -> Any:
        """Return ``x -> op -> y`` at *opset*, with an ``axes`` attribute unless *axes* is ``None``.

        Examples:
            >>> model = TestLiftOpset._reduction_model("ReduceMax", 17, axes=[1])
            >>> (model.opset_import[0].version, model.graph.node[0].attribute[0].name)
            (17, 'axes')
        """
        attributes = {"keepdims": 0} | ({"axes": axes} if axes is not None else {})
        node = helper.make_node(op, ["x"], ["y"], "reduce", **attributes)
        graph = helper.make_graph(
            [node],
            "reduce",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.zeros(1, np.int64), taken)] if taken else [],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])

    @pytest.mark.parametrize("opset", [17, 18])
    @pytest.mark.parametrize("op", REDUCTIONS)
    def test_every_reduction_moves_its_axes_to_an_input(self, op: str, opset: int) -> None:
        """Every reduction op moves its axes attribute to an input when lifted to opset 19."""
        model = self._reduction_model(op, opset, axes=[1])
        quantize._lift_opset(model)
        node = model.graph.node[0]
        assert (model.opset_import[0].version, len(node.input), "axes" in {a.name for a in node.attribute}) == (
            19,
            2,
            False,
        )

    @pytest.mark.parametrize("op", REDUCTIONS)
    def test_reduction_without_axes_is_left_alone(self, op: str) -> None:
        """A reduction without an axes attribute is left untouched by the lift."""
        model = self._reduction_model(op, 17, axes=None)
        quantize._lift_opset(model)
        assert (len(model.graph.node[0].input), len(model.graph.initializer)) == (1, 0)

    def test_axes_input_does_not_reuse_a_taken_name(self) -> None:
        """The new axes initializer takes a fresh name when the obvious one is taken."""
        model = self._reduction_model("ReduceMax", 17, axes=[1], taken="reduce_axes")
        quantize._lift_opset(model)
        new = model.graph.node[0].input[1]
        assert new != "reduce_axes" and sorted(i.name for i in model.graph.initializer) == sorted([new, "reduce_axes"])

    @pytest.mark.parametrize(
        ("inputs", "expected"),
        [pytest.param(["x"], [2], id="unsized"), pytest.param(["x", "sizes"], [], id="sized")],
    )
    def test_split_without_sizes_gets_its_output_count(self, inputs: list[str], expected: list[int]) -> None:
        """A Split without sizes gets num_outputs on lift; one with sizes does not."""
        split = helper.make_node("Split", inputs, ["a", "b"], "split", axis=0)
        sizes = [numpy_helper.from_array(np.array([1, 1], np.int64), "sizes")]
        graph = helper.make_graph(
            [split],
            "split",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1]) for n in "ab"],
            sizes if "sizes" in inputs else [],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        quantize._lift_opset(model)
        onnx.checker.check_model(model, full_check=True)
        assert [a.i for a in model.graph.node[0].attribute if a.name == "num_outputs"] == expected

    def test_graph_already_at_opset_19_is_left_alone(self) -> None:
        """A graph already at opset 19 is not modified."""
        reduce = helper.make_node("ReduceMax", ["x"], ["r"], "reduce", axes=[1], keepdims=0)
        graph = helper.make_graph(
            [reduce],
            "lift",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
            [helper.make_tensor_value_info("r", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
        before = model.SerializeToString()
        quantize._lift_opset(model)
        assert model.SerializeToString() == before

    def test_other_domains_are_left_alone(self) -> None:
        """Only the default domain is lifted; custom domains keep their version."""
        graph = helper.make_graph(
            [helper.make_node("CustomOp", ["x"], ["y"], "custom", domain="com.example")],
            "custom",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.example", 1)]
        )
        quantize._lift_opset(model)
        assert sorted((e.domain, e.version) for e in model.opset_import) == [("", 19), ("com.example", 1)]

    def test_op_whose_meaning_changed_is_refused(self) -> None:
        """An op whose meaning changed between opsets is refused instead of silently lifted."""
        pad = helper.make_node("Pad", ["x", "pads"], ["y"], "pad")
        graph = helper.make_graph(
            [pad],
            "pad",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])],
            [numpy_helper.from_array(np.array([1, 1], np.int64), "pads")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with pytest.raises(ValueError) as exc_info:
            quantize._lift_opset(model)
        assert "'pad' (Pad) changed meaning" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("op", "opset", "refused"),
        [
            ("Resize", 10, True),
            ("Resize", 12, True),
            ("Resize", 13, False),
            ("ScatterND", 13, True),
            ("ScatterND", 15, True),
            ("ScatterND", 16, False),
        ],
    )
    def test_resize_and_scatter_nd_are_lifted_only_from_the_opset_they_were_checked_from(
        self, op: str, opset: int, refused: bool
    ) -> None:
        """``Resize`` (before opset 13) and ``ScatterND`` (before 16) did mean something else than at opset 19.

        Their opset-17 behavior was checked against opset 19, so only a graph at or above the opset from which that
        holds may have them lifted; a hand-made graph below it is refused instead of failing late in the runtime.
        """
        node = helper.make_node(op, ["x", "a", "b"], ["y"], "node")
        graph = helper.make_graph(
            [node],
            "old",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
        if refused:
            with pytest.raises(ValueError, match=f"\\({op}\\) changed meaning"):
                quantize._lift_opset(model)
        else:
            quantize._lift_opset(model)
            assert [imported.version for imported in model.opset_import if imported.domain == ""] == [
                quantize._FP16_QDQ_OPSET
            ]

    def test_unknown_op_is_refused(self) -> None:
        """An op ONNX does not define is refused with a ValueError naming the node, not onnx's own SchemaError.

        ``int8_source_graph`` promises a ValueError for a graph it cannot lift; onnx's schema lookup raises its own
        error type, which is no ValueError.
        """
        graph = helper.make_graph(
            [helper.make_node("NotAnOp", ["x"], ["y"], "mystery")],
            "unknown",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        with pytest.raises(ValueError) as exc_info:
            quantize._lift_opset(model)
        assert "'mystery' (NotAnOp) is not an ONNX op" in str(exc_info.value)

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("coordinate_mode", ["half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"])
    @pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
    def test_resize_gives_the_same_values_at_opset_19(self, mode: str, coordinate_mode: str) -> None:
        """Resize outputs match before and after the lift for every mode and coordinate mode."""
        node = helper.make_node(
            "Resize", ["x", "", "", "sizes"], ["y"], "resize", mode=mode, coordinate_transformation_mode=coordinate_mode
        )
        graph = helper.make_graph(
            [node],
            "resize",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 7, 9])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 20, 31])],
            [numpy_helper.from_array(np.array([1, 2, 20, 31], np.int64), "sizes")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        x = np.random.default_rng(0).standard_normal((1, 2, 7, 9)).astype(np.float32)
        before, after = _outputs_before_and_after_lift(model, {"x": x})
        np.testing.assert_allclose(after[0], before[0], atol=1e-6)

    @_requires_onnx_and_onnxruntime
    @pytest.mark.parametrize("reduction", ["none", "add", "mul"])
    def test_scatter_nd_gives_the_same_values_at_opset_19(self, reduction: str) -> None:
        """ScatterND outputs match before and after the lift for every reduction."""
        node = helper.make_node("ScatterND", ["data", "indices", "updates"], ["y"], "scatter", reduction=reduction)
        graph = helper.make_graph(
            [node],
            "scatter",
            [helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 3])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3])],
            [
                numpy_helper.from_array(np.array([[1], [3], [1]], np.int64), "indices"),
                numpy_helper.from_array(np.arange(1, 10, dtype=np.float32).reshape(3, 3), "updates"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
        data = np.random.default_rng(1).standard_normal((4, 3)).astype(np.float32)
        before, after = _outputs_before_and_after_lift(model, {"data": data})
        np.testing.assert_allclose(after[0], before[0], atol=1e-6)


def _patch_embedding(
    *,
    channels: int = 3,
    patch: int = 16,
    size: int = 32,
    out_channels: int = 8,
    bias: bool = True,
    empty_bias: bool = False,
    prefix: str = BACKBONE,
    batch: int | str = 1,
    height: int | str | None = None,
    width: int | None = None,
    kernel: tuple[int, int] | None = None,
    **attributes: Any,
) -> Any:
    """Build ``image -> Conv -> features`` as a patch embedding: kernel and stride *kernel*, no padding.

    Args:
        channels: Input channels of the image.
        patch: Kernel size and stride, in pixels, when *kernel* is not given.
        size: Image height and width, unless *height* or *width* is given.
        out_channels: Output channels of the kernel.
        bias: Give the ``Conv`` a bias.
        empty_bias: Name the bias input with an empty string, how ONNX spells an omitted optional input.
        prefix: Node-name prefix of the ``Conv``.
        batch: Batch size of the image; a string makes it symbolic.
        height: Overrides the declared image height; a string makes it symbolic.
        width: Overrides the declared image width.
        kernel: ``(rows, columns)`` of a rectangular kernel and stride.
        **attributes: ``Conv`` attributes that replace the patch-embedding ones.

    Returns:
        A checked-shape ``ModelProto`` whose output ``features`` is the ``Conv``'s output.

    Examples:
        >>> model = _patch_embedding(channels=2, patch=4, size=8)
        >>> [node.op_type for node in model.graph.node], model.graph.output[0].name
        (['Conv'], 'features')
    """
    rng = np.random.default_rng(0)
    rows, columns = kernel or (patch, patch)
    high, wide = height or size, width or size
    inits = [
        numpy_helper.from_array(rng.standard_normal((out_channels, channels, rows, columns)).astype(np.float32), "w")
    ]
    inputs = ["image", "w"]
    if bias:
        inits.append(numpy_helper.from_array(rng.standard_normal(out_channels).astype(np.float32), "b"))
        inputs.append("b")
    elif empty_bias:
        inputs.append("")
    conv = {"kernel_shape": [rows, columns], "strides": [rows, columns], **attributes}
    grid = ["out_height" if isinstance(high, str) else high // rows, wide // columns]
    graph = helper.make_graph(
        [helper.make_node("Conv", inputs, ["features"], f"{prefix}patch/Conv", **conv)],
        "patch_embedding",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [batch, channels, high, wide])],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [batch, out_channels, *grid])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _with_detector_outputs(model: Any) -> Any:
    """Make *model*'s ``features`` the source of ``dets`` and ``labels`` and return it, so that INT8 accepts the graph.

    Examples:
        >>> model = _with_detector_outputs(_patch_embedding(channels=2, patch=4, size=8))
        >>> [output.name for output in model.graph.output]
        ['dets', 'labels']
    """
    model.graph.node.extend([helper.make_node("Identity", ["features"], [name], name) for name in ("dets", "labels")])
    del model.graph.output[:]
    model.graph.output.extend(
        helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("dets", "labels")
    )
    return model


class TestPatchEmbeddingAsMatmul:
    """The 3-channel patch-embedding ``Conv`` becomes a ``MatMul`` that INT8 can run, with the same output."""

    @_requires_onnxruntime
    @pytest.mark.parametrize(
        ("batch", "channels", "patch", "height", "width"),
        [
            pytest.param(1, 3, 16, 32, 32, id="rgb-16-pixel-patches"),
            pytest.param(1, 3, 12, 24, 24, id="rgb-12-pixel-patches"),
            pytest.param(2, 3, 16, 32, 48, id="batch-2-non-square"),
            pytest.param(3, 2, 4, 8, 16, id="batch-3-two-channels-non-square"),
            pytest.param(1, 1, 8, 16, 16, id="grayscale"),
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    def test_features_match_the_convolution(
        self, batch: int, channels: int, patch: int, height: int, width: int, bias: bool
    ) -> None:
        """The MatMul rewrite produces the same features as the convolution, with and without bias."""
        import onnxruntime as ort

        model = _patch_embedding(channels=channels, patch=patch, height=height, width=width, batch=batch, bias=bias)
        image = np.random.default_rng(1).standard_normal((batch, channels, height, width)).astype(np.float32)
        expected = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"image": image}
        )[0]
        assert quantize._patch_embedding_as_matmul(model) == 1
        actual = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"image": image}
        )[0]
        # The convolution and the matrix multiply add up a patch's float32 products (768 in the first case) in a
        # different order: macOS differed by 4e-5 on outputs of about 40. A wrong permutation, kernel layout or bias is
        # off by whole units.
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    @pytest.mark.parametrize(
        ("channels", "patch", "multiple_of", "rewritten"),
        [
            pytest.param(3, 12, 1, 1, id="432-values-by-default"),
            pytest.param(3, 12, 32, 0, id="432-values-not-a-multiple-of-32"),
            pytest.param(3, 16, 32, 1, id="768-values-a-multiple-of-32"),
            pytest.param(2, 4, 32, 1, id="32-values-of-two-channels"),
            pytest.param(4, 2, 32, 0, id="16-values-of-four-channels"),
        ],
    )
    def test_multiple_of_rewrites_only_the_patches_it_divides(
        self, channels: int, patch: int, multiple_of: int, rewritten: int
    ) -> None:
        """Only patches whose value count is a multiple of multiple_of are rewritten."""
        model = _patch_embedding(channels=channels, patch=patch, size=2 * patch)
        assert quantize._patch_embedding_as_matmul(model, multiple_of=multiple_of) == rewritten

    @pytest.mark.parametrize("multiple_of", [0, -32])
    def test_multiple_of_below_one_is_refused(self, multiple_of: int) -> None:
        """A multiple_of below one is refused and leaves the model untouched."""
        model = _patch_embedding()
        before = model.SerializeToString()
        with pytest.raises(ValueError) as exc_info:
            quantize._patch_embedding_as_matmul(model, multiple_of=multiple_of)
        assert "multiple_of" in str(exc_info.value)
        assert model.SerializeToString() == before

    def test_convolution_is_replaced_by_the_matmul_the_plan_quantizes(self) -> None:
        """The patch Conv becomes a MatMul that plan_int8 selects for quantization."""
        model = _patch_embedding()
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert [node.op_type for node in model.graph.node].count("Conv") == 0
        assert _names(quantize.plan_int8(model)) == ["patch/MatMul"]

    def test_the_matmul_weight_is_the_flattened_kernel(self) -> None:
        """The MatMul weight is the kernel flattened to (out, C*H*W) and transposed."""
        model = _patch_embedding(channels=3, patch=16, size=32, out_channels=8)
        kernel = numpy_helper.to_array(model.graph.initializer[0])
        quantize._patch_embedding_as_matmul(model)
        matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
        weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        np.testing.assert_array_equal(weights[matmul.input[1]], kernel.reshape(8, 3 * 16 * 16).T)

    def test_the_output_keeps_its_name(self) -> None:
        """The rewritten subgraph still produces the original output tensor name."""
        model = _patch_embedding()
        quantize._patch_embedding_as_matmul(model)
        assert model.graph.node[-1].output[0] == "features"

    @pytest.mark.parametrize("bias", [True, False])
    def test_the_rewritten_graph_is_valid_onnx(self, bias: bool) -> None:
        """The rewritten graph passes the ONNX checker and strict shape inference."""
        model = _patch_embedding(bias=bias)
        quantize._patch_embedding_as_matmul(model)
        onnx.checker.check_model(model, full_check=True)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_an_empty_bias_name_is_no_bias(self) -> None:
        """An empty bias input name means no bias and leaves no empty input behind."""
        model = _patch_embedding(bias=False, empty_bias=True)
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert "" not in [name for node in model.graph.node for name in node.input]
        onnx.checker.check_model(model, full_check=True)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_the_kernel_keeps_its_dtype(self, dtype: type) -> None:
        """The MatMul weight keeps the kernel's FP32 or FP16 dtype."""
        model = _patch_embedding()
        for init in model.graph.initializer:
            init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(dtype), init.name))
        quantize._patch_embedding_as_matmul(model)
        matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
        weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
        assert weights[matmul.input[1]].dtype == dtype

    def test_a_kernel_reached_through_an_identity_is_found(self) -> None:
        """A kernel reached through an Identity alias is still rewritten."""
        model = _patch_embedding()
        conv = model.graph.node[0]
        conv.input[1] = "w_alias"
        model.graph.node.insert(0, helper.make_node("Identity", ["w"], ["w_alias"], "alias"))
        assert quantize._patch_embedding_as_matmul(model) == 1

    def test_every_replaced_kernel_is_dropped(self) -> None:
        """Every kernel initializer whose Conv was replaced is dropped."""
        model = _patch_embedding()
        model.graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(model.graph.initializer[0]), "w2"))
        model.graph.node.append(
            helper.make_node(
                "Conv", ["image", "w2", "b"], ["more"], f"{BACKBONE}more/Conv", kernel_shape=[16, 16], strides=[16, 16]
            )
        )
        assert quantize._patch_embedding_as_matmul(model) == 2
        assert {"w", "w2"}.isdisjoint(init.name for init in model.graph.initializer)

    @pytest.mark.parametrize("reader", ["another-node", "graph-output", "if-branch"])
    def test_a_kernel_something_else_still_reads_is_kept(self, reader: str) -> None:
        """A kernel still read by another node, output or branch stays in the graph."""
        model = _patch_embedding()
        kernel_info = helper.make_tensor_value_info("w_copy", TensorProto.FLOAT, [8, 3, 16, 16])
        if reader == "another-node":
            model.graph.node.append(helper.make_node("Identity", ["w"], ["w_copy"], "copy"))
            model.graph.output.append(kernel_info)
        elif reader == "graph-output":
            model.graph.output.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 16, 16]))
        else:
            # A branch that captures the kernel from the outer scope reads it, though no top-level node does.
            branch = helper.make_graph(
                [helper.make_node("Identity", ["w"], ["w_copy"], "inner")], "branch", [], [kernel_info]
            )
            model.graph.initializer.append(numpy_helper.from_array(np.array(True), "flag"))
            model.graph.node.append(
                helper.make_node("If", ["flag"], ["w_copy"], "if", then_branch=branch, else_branch=branch)
            )
            model.graph.output.append(kernel_info)
        assert quantize._patch_embedding_as_matmul(model) == 1
        assert "w" in [init.name for init in model.graph.initializer]

    def test_an_initializer_that_is_a_graph_input_is_never_dropped(self) -> None:
        """An initializer that is also a graph input is never dropped as unread."""
        # Dropping it would turn a default value into an input the caller must now supply.
        graph = helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"], "relu")],
            "g",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("old", TensorProto.FLOAT, [1]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.zeros(1, np.float32), name) for name in ("old", "unused")],
        )
        quantize._drop_unread_initializers(graph, ["old", "unused"])
        assert [init.name for init in graph.initializer] == ["old"]

    def test_generated_names_never_reuse_a_name_in_the_graph(self) -> None:
        """Generated tensor and node names never collide with names already in the graph."""
        model = _patch_embedding()
        taken = f"{BACKBONE}patch/cut"
        model.graph.node.append(helper.make_node("Identity", ["image"], [taken], f"{BACKBONE}patch/Cut"))
        model.graph.output.append(helper.make_tensor_value_info(taken, TensorProto.FLOAT, [1, 3, 32, 32]))
        quantize._patch_embedding_as_matmul(model)
        produced = [name for node in model.graph.node for name in node.output]
        assert len(produced) == len(set(produced))
        assert len({node.name for node in model.graph.node}) == len(model.graph.node)

    @pytest.mark.parametrize(
        "case",
        [
            pytest.param({"strides": [2, 2]}, id="overlapping-patches"),
            pytest.param({"pads": [1, 1, 1, 1]}, id="padded"),
            pytest.param({"dilations": [2, 2]}, id="dilated"),
            pytest.param({"auto_pad": "SAME_UPPER"}, id="auto-pad"),
            pytest.param({"kernel": (16, 8)}, id="rectangular-kernel"),
            pytest.param({"channels": 32, "patch": 4, "size": 8}, id="channels-TensorRT-does-not-pad"),
            pytest.param({"height": 36}, id="kernel-does-not-tile-the-height"),
            pytest.param({"width": 36}, id="kernel-does-not-tile-the-width"),
            pytest.param({"height": "height"}, id="symbolic-height"),
            pytest.param({"batch": "batch"}, id="symbolic-batch"),
            pytest.param({"size": 0}, id="empty-image"),
            pytest.param({"prefix": HEAD}, id="outside-the-backbone"),
            pytest.param({"prefix": DECODER}, id="in-the-decoder"),
        ],
    )
    def test_other_convolutions_are_left_alone(self, case: dict[str, Any]) -> None:
        """Convolutions that are not a plain backbone patch embedding are left untouched."""
        model = _patch_embedding(**case)
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_a_grouped_convolution_is_left_alone(self) -> None:
        """A grouped convolution is not rewritten even when its patch size would allow it."""
        # Two input channels per group and 4-pixel patches make 32 values per patch: only the grouping rules it out.
        model = _patch_embedding(channels=4, patch=4, size=8, out_channels=4, group=2)
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.ones((4, 2, 4, 4), np.float32), "w"))
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_an_existing_matmul_name_is_never_reused(self) -> None:
        """A convolution is left alone when its MatMul name is already taken."""
        model = _patch_embedding()
        model.graph.node.append(helper.make_node("Identity", ["features"], ["copy"], f"{BACKBONE}patch/MatMul"))
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_a_duplicated_convolution_name_is_left_alone(self) -> None:
        """Convolutions sharing a node name cannot be told apart and are left alone."""
        model = _patch_embedding()
        model.graph.node.append(
            helper.make_node("Conv", ["image", "w"], ["other"], f"{BACKBONE}patch/Conv", kernel_shape=[16, 16])
        )
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before

    def test_two_convolutions_in_one_scope_get_one_matmul(self) -> None:
        """Two convolutions mapping to one MatMul name yield a single rewrite and unique names."""
        # ``patch/Conv`` and ``patch/Conv_1`` would both become ``patch/MatMul``; the second must not take the name.
        model = _patch_embedding()
        model.graph.node.append(
            helper.make_node(
                "Conv",
                ["image", "w", "b"],
                ["other"],
                f"{BACKBONE}patch/Conv_1",
                kernel_shape=[16, 16],
                strides=[16, 16],
            )
        )
        assert quantize._patch_embedding_as_matmul(model) == 1
        names = [node.name for node in model.graph.node]
        assert len(names) == len(set(names))

    def test_a_rejected_convolution_does_not_stop_the_next_one(self) -> None:
        """A convolution that cannot be rewritten does not stop later ones from being rewritten."""
        model = _patch_embedding()
        model.graph.input.append(helper.make_tensor_value_info("narrow", TensorProto.FLOAT, [1, 3, 36, 32]))
        model.graph.node.insert(
            0,
            helper.make_node(
                "Conv", ["narrow", "w"], ["tall"], f"{BACKBONE}tall/Conv", kernel_shape=[16, 16], strides=[16, 16]
            ),
        )
        assert quantize._patch_embedding_as_matmul(model) == 1

    @pytest.mark.parametrize("initializer", [False, True])
    def test_a_kernel_that_can_be_replaced_at_run_time_is_left_alone(self, initializer: bool) -> None:
        """A kernel exposed as a graph input is not constant and its convolution is left alone."""
        # A graph input without an initializer is no constant kernel; one with an initializer can be overridden.
        model = _patch_embedding()
        model.graph.input.append(helper.make_tensor_value_info("w", TensorProto.FLOAT, [8, 3, 16, 16]))
        if not initializer:
            kept = [init for init in model.graph.initializer if init.name != "w"]
            model.graph.ClearField("initializer")
            model.graph.initializer.extend(kept)
        before = model.SerializeToString()
        assert quantize._patch_embedding_as_matmul(model) == 0
        assert model.SerializeToString() == before


@_requires_onnx_and_onnxruntime
@_requires_onnxconverter_common
class TestPatchEmbeddingInInt8Source:
    """``int8_source_graph`` plans, calibrates and quantizes the patch embedding as the matrix multiply it becomes."""

    @pytest.fixture
    def patch_graph(self, tmp_path: Path) -> Path:
        """A 16-pixel patch embedding over a ``(1, 3, 16, 16)`` image, saved to disk with a detector's output names.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = patch_graph(tmp_path)  # doctest: +SKIP
        """
        path = tmp_path / "patch.onnx"
        onnx.save(_with_detector_outputs(_patch_embedding(size=16)), path)
        return path

    def test_patch_embedding_is_quantized_as_a_matrix_multiply(self, patch_graph: Path) -> None:
        """int8_source_graph quantizes the patch embedding as a MatMul with Q/DQ on both inputs."""
        images = np.random.default_rng(0).standard_normal((3, 3, 16, 16)).astype(np.float32)
        with quantize.int8_source_graph(
            str(patch_graph),
            calibration_data=images,
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            graph = onnx.load(int8_path).graph
        producers = {output: node for node in graph.node for output in node.output}
        (matmul,) = [node for node in graph.node if node.name == f"{BACKBONE}patch/MatMul"]
        activation = producers[matmul.input[0]]
        assert [node.op_type for node in graph.node].count("Conv") == 0
        assert activation.op_type == "DequantizeLinear"
        assert producers[activation.input[0]].op_type == "QuantizeLinear"
        assert producers[matmul.input[1]].op_type == "DequantizeLinear"

    def test_the_planned_patch_is_the_one_quantized_when_two_share_a_parent(self, tmp_path: Path) -> None:
        """With two patch embeddings under one parent, the planned one is the one quantized."""
        # The 432-value patch (skipped by the plan) comes first; the 192-value one is planned. Both would be named
        # ``patch/MatMul``, so the FP16 copy must rewrite the planned patch first or the plan quantizes the wrong layer.
        model = _patch_embedding(patch=12, size=24)
        second = _patch_embedding(patch=8, size=24)
        for initializer in second.graph.initializer:
            initializer.name += "_second"
        conv = second.graph.node[0]
        conv.name = f"{BACKBONE}patch/Conv_1"
        conv.input[1:] = [f"{name}_second" for name in conv.input[1:]]
        conv.output[0] = "features_second"
        model.graph.initializer.extend(second.graph.initializer)
        model.graph.node.append(conv)
        model.graph.node.extend(
            helper.make_node("Identity", [source], [name], name)
            for source, name in (("features", "dets"), ("features_second", "labels"))
        )
        del model.graph.output[:]
        model.graph.output.extend(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ("dets", "labels")
        )
        path = tmp_path / "two_patches.onnx"
        onnx.save(model, path)
        with quantize.int8_source_graph(
            str(path),
            calibration_data=np.zeros((3, 3, 24, 24), np.float32),
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            graph = onnx.load(int8_path).graph
        producers = {output: node for node in graph.node for output in node.output}
        weights = {initializer.name: initializer for initializer in graph.initializer}
        (matmul,) = [node for node in graph.node if node.op_type == "MatMul" and "patch/" in node.name]
        (conv,) = [node for node in graph.node if node.op_type == "Conv"]
        quantized_weight = producers[matmul.input[1]]
        assert quantized_weight.op_type == "DequantizeLinear"
        assert tuple(weights[quantized_weight.input[0]].dims) == (192, 8)
        assert tuple(weights[conv.input[1]].dims) == (8, 3, 12, 12)

    def test_the_fp16_copy_is_lifted_before_the_patch_embedding_is_rewritten(self, tmp_path: Path) -> None:
        """An opset-13 source is lifted before the rewrite so the INT8 graph ends at opset 19."""
        # At opset 13 the rewrite's ``Add`` would be refused by the lift that follows it, after calibration has run.
        model = _with_detector_outputs(_patch_embedding(size=16))
        model.opset_import[0].version = 13
        path = tmp_path / "opset13.onnx"
        onnx.save(model, path)
        with quantize.int8_source_graph(
            str(path),
            calibration_data=np.zeros((3, 3, 16, 16), np.float32),
            max_images=3,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            quantized = onnx.load(int8_path)
        assert quantized.opset_import[0].version == 19

    @pytest.mark.parametrize("copy", ["overlapping-patches", "renamed"])
    def test_fp16_copy_without_a_planned_node_is_refused(self, patch_graph: Path, tmp_path: Path, copy: str) -> None:
        """An FP16 copy lacking a planned node is refused with a ValueError, not a KeyError."""
        # Overlapping patches leave nothing to rewrite; a renamed ``Conv`` is rewritten under another ``MatMul`` name.
        # Either way the node the plan quantizes is missing, which would otherwise surface as a ``KeyError``.
        other = (
            _patch_embedding(size=16, strides=[8, 8]) if copy == "overlapping-patches" else _patch_embedding(size=16)
        )
        if copy == "renamed":
            other.graph.node[0].name = f"{BACKBONE}other/Conv"
        onnx.save(_with_detector_outputs(other), tmp_path / "other.onnx")

        @contextlib.contextmanager
        def other_fp16(*_: Any, **__: Any) -> Iterator[str]:
            yield str(tmp_path / "other.onnx")

        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(patch_graph),
                calibration_data=np.zeros((3, 3, 16, 16), np.float32),
                max_images=3,
                dynamic_batch=False,
                fp16_graph=other_fp16,
            ):
                pass
        assert "lacks 1 node" in str(exc_info.value)


class TestInt8SourceGraph:
    """A graph INT8 cannot start from is refused before any calibration, with advice an INT8 request can follow."""

    @pytest.mark.parametrize(
        ("rewrite", "message"),
        [
            pytest.param("qdq", "already explicitly quantized", id="quantized"),
            pytest.param("qdq-subgraph", "already explicitly quantized", id="quantized-subgraph"),
            pytest.param("fp16", "not a float32 export", id="fp16"),
            pytest.param("fp16-weights", "not a float32 export", id="fp16-weights-fp32-io"),
            pytest.param("dynamic", "static batch", id="dynamic-batch"),
            pytest.param("pad", "changed meaning", id="unliftable"),
            pytest.param("two-heads", "detection, segmentation and keypoint models only", id="two-heads"),
            pytest.param("rank-3", "image input", id="not-an-image"),
            pytest.param("symbolic-height", "image input", id="symbolic-height"),
            pytest.param("weight-input", "as graph inputs too", id="overridable-weight"),
            # Refused before calibration: the calibration directory holds no image, which would fail differently.
            pytest.param("nan-weight", "NaN or infinite values, e.g. 'w1'", id="nan-weight"),
            pytest.param("inf-weight", "NaN or infinite values, e.g. 'w1'", id="inf-weight"),
        ],
    )
    def test_unquantizable_source_is_refused(self, tmp_path: Path, rewrite: str, message: str) -> None:
        """A source graph INT8 cannot start from is refused before calibration, naming the reason."""
        dtype = np.float16 if rewrite == "fp16" else np.float32
        model = _attention_and_mlp(BACKBONE, sequence=8, head_size=16, dtype=dtype)
        if rewrite == "qdq":
            plan = quantize.plan_int8(model)
            quantize.insert_qdq(model, plan, dict.fromkeys(plan.activations, 2.0))
        elif rewrite == "qdq-subgraph":
            inner = helper.make_graph(
                [
                    helper.make_node("QuantizeLinear", ["y", "s", "z"], ["yq"], "inner/Q"),
                    helper.make_node("DequantizeLinear", ["yq", "s", "z"], ["w"], "inner/DQ"),
                ],
                "branch",
                [],
                [helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 8, 32])],
                [
                    numpy_helper.from_array(np.array(1.0, np.float32), "s"),
                    numpy_helper.from_array(np.array(0, np.int8), "z"),
                ],
            )
            model.graph.input.append(helper.make_tensor_value_info("cond", TensorProto.BOOL, []))
            model.graph.node.append(
                helper.make_node("If", ["cond"], ["out"], "if", then_branch=inner, else_branch=inner)
            )
            model.graph.output.append(helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 32]))
        elif rewrite == "fp16-weights":
            for init in model.graph.initializer:
                if init.data_type == TensorProto.FLOAT:
                    init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
        elif rewrite == "dynamic":
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        elif rewrite == "weight-input":
            weight = model.graph.initializer[0]
            model.graph.input.append(helper.make_tensor_value_info(weight.name, weight.data_type, weight.dims))
        elif rewrite == "pad":
            model.graph.initializer.append(numpy_helper.from_array(np.zeros(6, np.int64), "pads"))
            model.graph.node.append(helper.make_node("Pad", [model.graph.input[0].name, "pads"], ["labels"], "pad"))
        elif rewrite in ("nan-weight", "inf-weight"):
            w1 = next(init for init in model.graph.initializer if init.name == "w1")
            values = numpy_helper.to_array(w1).copy()
            values[0, 1] = np.nan if rewrite == "nan-weight" else np.inf
            w1.CopyFrom(numpy_helper.from_array(values, "w1"))
        if rewrite in ("pad", "symbolic-height", "nan-weight", "inf-weight"):
            # An image input (never run here), so that only the rewrite under test can be what is refused.
            height = "height" if rewrite == "symbolic-height" else 8
            model.graph.input[0].CopyFrom(helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, height, 32]))
        if rewrite in ("pad", "two-heads", "rank-3", "symbolic-height", "nan-weight", "inf-weight"):
            # A detector's outputs, so that only the rewrite under test can be what is refused.
            model.graph.node.append(helper.make_node("Identity", ["y"], ["dets"], "dets"))
            outputs = ["dets", "labels"] + (["masks", "keypoints"] if rewrite == "two-heads" else [])
            if rewrite != "pad":
                model.graph.node.append(helper.make_node("Identity", ["y"], ["labels"], "labels"))
            if rewrite == "two-heads":
                for head in ("masks", "keypoints"):
                    model.graph.node.append(helper.make_node("Identity", ["y"], [head], head))
            del model.graph.output[:]
            model.graph.output.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in outputs)
        path = tmp_path / "source.onnx"
        onnx.save(model, path)
        # A non-detector gets the type RFDETR.export raises for the same model; a malformed graph gets ValueError.
        with pytest.raises(NotImplementedError if rewrite == "two-heads" else ValueError, match=message):
            with quantize.int8_source_graph(
                str(path),
                calibration_data=str(tmp_path),
                max_images=1,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass

    @pytest.mark.parametrize(
        "outputs",
        [
            pytest.param(("dets", "labels"), id="detection"),
            pytest.param(("dets", "labels", "masks"), id="segmentation"),
            pytest.param(("dets", "labels", "keypoints"), id="keypoint"),
        ],
    )
    def test_detection_segmentation_and_keypoint_outputs_are_accepted(self, outputs: tuple[str, ...]) -> None:
        """Detection, segmentation and keypoint output sets are accepted as INT8 sources."""
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], [name], name) for name in outputs],
            "heads",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 8, 8]) for name in outputs],
        )
        quantize._refuse_unquantizable_source(helper.make_model(graph), "heads.onnx")


@_requires_onnxruntime
@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestCalibration:
    """Ranges come from the FP32 graph under onnxruntime (CPU here); selected by the ``onnx`` integration job."""

    @pytest.fixture
    def image_graph(self, tmp_path: Path) -> Path:
        """A non-square ``(2, 3, 4, 6)``-input graph with two backbone Convs, saved to disk.

        The first reads the 3-channel image and stays FP16; the second reads 32 channels and is quantized. The outputs
        carry a detector's names.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path = image_graph(tmp_path)  # doctest: +SKIP
        """
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["input", "kernel"], ["features"], f"{BACKBONE}patch/Conv"),
                helper.make_node("Conv", ["features", "mix"], ["mixed"], f"{BACKBONE}mix/Conv"),
                helper.make_node("Relu", ["mixed"], ["dets"], "relu"),
                helper.make_node("Identity", ["mixed"], ["labels"], "labels"),
            ],
            "image",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 6])],
            [
                helper.make_tensor_value_info("dets", TensorProto.FLOAT, [2, 2, 4, 6]),
                helper.make_tensor_value_info("labels", TensorProto.FLOAT, [2, 2, 4, 6]),
            ],
            [
                numpy_helper.from_array(np.ones((32, 3, 1, 1), np.float32), "kernel"),
                numpy_helper.from_array(np.ones((2, 32, 1, 1), np.float32), "mix"),
            ],
        )
        path = tmp_path / "image.onnx"
        # Pinned: onnx 1.21 writes IR version 14 by default, which the onnxruntime CI installs (IR 13 at most) refuses.
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)
        return path

    @pytest.mark.parametrize(
        "values",
        [
            # Stopping after the first group would miss the largest magnitude in the padded final group, and keeping
            # only the last group would miss it in the first.
            pytest.param((2.0, 3.0, -5.0), id="largest-in-the-last-group"),
            pytest.param((-5.0, 2.0, 3.0), id="largest-in-the-first-group"),
        ],
    )
    def test_ranges_are_the_absolute_maximum_over_every_image(
        self, image_graph: Path, values: tuple[float, ...]
    ) -> None:
        """The range of a tensor is its absolute maximum over every calibration image."""
        images = [np.full((1, 3, 4, 6), value, np.float32) for value in values]
        ranges = quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input", "features"], images)
        assert ranges == pytest.approx({"input": 5.0, "features": 15.0})

    def test_calibration_data_is_closed_when_calibration_fails(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path
    ) -> None:
        """Calibration batch generators are closed when calibration fails on bad data."""
        generators: list[Iterator[np.ndarray]] = []
        real = calibration.calibration_batches

        def keep(*args: Any, **kwargs: Any) -> Iterator[np.ndarray]:
            """Return the real batch generator and remember it."""
            generators.append(real(*args, **kwargs))
            return generators[-1]

        monkeypatch.setattr(calibration, "calibration_batches", keep)
        images = np.zeros((2, 3, 4, 6), np.float32)
        images[1, 0, 1, 2] = np.nan
        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=images,
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "NaN or infinite" in str(exc_info.value)
        # One pass for the smoothing scales of a detector (this graph has no norm to smooth) and one for the ranges.
        assert [inspect.getgeneratorstate(g) for g in generators] == [inspect.GEN_CLOSED, inspect.GEN_CLOSED]

    @pytest.mark.parametrize(
        ("images", "warned"),
        [(1, True), (calibration.MIN_CALIBRATION_SAMPLES - 1, True), (calibration.MIN_CALIBRATION_SAMPLES, False)],
    )
    def test_too_few_images_are_warned_about(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path, images: int, warned: bool
    ) -> None:
        """Fewer than the minimum calibration samples log a warning; enough samples do not."""
        messages: list[str] = []
        monkeypatch.setattr(calibration.logger, "warning", messages.append)
        samples = [np.zeros((1, 3, 4, 6), np.float32)] * images
        quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], samples)
        assert [m for m in messages if f"only {images} sample" in m] == (messages if warned else [])
        assert bool(messages) is warned

    def test_calibrated_range_becomes_the_scale(self, image_graph: Path) -> None:
        """A calibrated range over 127 becomes the quantize scale in the INT8 graph."""
        images = np.random.default_rng(0).standard_normal((3, 3, 4, 6)).astype(np.float32)
        with quantize.int8_source_graph(
            str(image_graph),
            calibration_data=images,
            max_images=10,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as int8_path:
            model = onnx.load(int8_path)
        initializers = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        (quantize_node,) = [n for n in model.graph.node if n.op_type == "QuantizeLinear"]
        # The all-ones 1x1 kernel sums the three channels, so the quantized tensor's range is that sum's largest value.
        expected = np.float16(np.abs(images.sum(axis=1)).max() / 127)
        assert float(initializers[quantize_node.input[1]]) == float(expected)

    def test_no_image_is_refused(self, image_graph: Path) -> None:
        """Calibrating on no image raises a ValueError."""
        with pytest.raises(ValueError) as exc_info:
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [])
        assert "no image" in str(exc_info.value)

    def test_no_tensor_is_refused_before_the_probe_is_written(self, image_graph: Path) -> None:
        """Asking for no tensor is refused up front, with no probe file left beside the graph.

        An empty output list would otherwise make the probe session return every model output, which then fails far from
        the cause (a real detector's ``dets``/``labels`` are ragged).
        """
        with pytest.raises(ValueError) as exc_info:
            quantize.calibrate_ranges(
                str(image_graph), onnx.load(image_graph), [], [np.zeros((1, 3, 4, 6), np.float32)]
            )
        assert "no tensor to measure" in str(exc_info.value)
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    def test_calibration_runs_on_the_cpu(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        """The probe session is created with the CPU execution provider only."""
        import onnxruntime as ort

        requested: list[object] = []

        def record(*_: object, providers: object = None, **__: object) -> None:
            requested.append(providers)
            raise RuntimeError("recorded")

        monkeypatch.setattr(ort, "InferenceSession", record)
        with pytest.raises(RuntimeError) as exc_info:
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert "recorded" in str(exc_info.value)
        assert requested == [["CPUExecutionProvider"]]

    def test_calibration_runs_in_priority_based_order(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        """The probe session runs its nodes in onnxruntime's priority-based order.

        The probe adds one reduction per measured tensor; under the default order those raised peak host memory far
        above the plain graph's at a large calibration batch.
        """
        import onnxruntime as ort

        requested: list[object] = []

        def record(*_: object, sess_options: object = None, **__: object) -> None:
            requested.append(sess_options)
            raise RuntimeError("recorded")

        monkeypatch.setattr(ort, "InferenceSession", record)
        with pytest.raises(RuntimeError) as exc_info:
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert "recorded" in str(exc_info.value)
        assert [getattr(options, "execution_order", None) for options in requested] == [
            ort.ExecutionOrder.PRIORITY_BASED
        ]

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_calibration_image_is_refused(self, image_graph: Path, value: float) -> None:
        """A NaN or infinite calibration image is refused by the range pass."""
        # Not the first element: onnxruntime's ReduceMax skips a NaN anywhere else and returns a finite range.
        images = np.zeros((2, 3, 4, 6), np.float32)
        images[1, 0, 1, 2] = value
        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=images,
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "NaN or infinite" in str(exc_info.value)

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_calibration_image_is_refused_on_the_first_pass(
        self, mocker: MockerFixture, tmp_path: Path, value: float
    ) -> None:
        """A detector whose norms are smoothed refuses a NaN or infinite image on the smoothing pass, its first one.

        The smoothing pass runs before calibration; refusing only in calibration would cost one full pass over the
        images first. The bad value is not the first element, which onnxruntime's ``ReduceMax`` would not skip.
        """
        path = tmp_path / "mlp.onnx"
        onnx.save(_gelu_mlp(norm=True), path)
        reads = mocker.patch.object(calibration, "calibration_batches", wraps=calibration.calibration_batches)
        images = np.random.default_rng(7).standard_normal((4, 2, 4, 4)).astype(np.float32)
        images[1, 0, 1, 2] = value
        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(path),
                calibration_data=images,
                max_images=10,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "batch 2 holds NaN or infinite" in str(exc_info.value)
        assert reads.call_count == 1

    @pytest.mark.parametrize("opset", [17, 19])
    def test_nan_computed_inside_the_graph_gives_an_infinite_range(self, tmp_path: Path, opset: int) -> None:
        """A NaN the graph computes from a finite image calibrates to an infinite range, which the range check refuses.

        onnxruntime's ``ReduceMax`` skips a NaN anywhere but the first element, so the probe would otherwise report a
        finite range for activations that are not; ``Sqrt`` of a negative value puts the NaN at the second element. The
        probe runs on the exported opset and on the lifted one.
        """
        graph = helper.make_graph(
            [helper.make_node("Sqrt", ["x"], ["root"], "sqrt")],
            "nan",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
            [helper.make_tensor_value_info("root", TensorProto.FLOAT, [1, 3])],
        )
        path = tmp_path / "nan.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10), path)
        image = np.array([[1.0, -1.0, 4.0]], np.float32)
        ranges = quantize.calibrate_ranges(str(path), onnx.load(path), ["root"], [image])
        assert ranges == {"root": np.inf}

    def test_range_too_large_for_an_fp16_scale_is_refused(self, image_graph: Path) -> None:
        """A finite range whose scale overflows FP16 is refused."""
        images = np.full((2, 3, 4, 6), 1e8, np.float32)  # finite, but 1e8 / 127 overflows FP16
        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=images,
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "does not fit an FP16 scale" in str(exc_info.value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_range_that_is_not_finite_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path, value: float
    ) -> None:
        """A NaN or infinite measured range is refused."""
        monkeypatch.setattr(
            quantize, "calibrate_ranges", lambda _path, _model, tensors, _batches: dict.fromkeys(tensors, value)
        )
        images = np.ones((2, 3, 4, 6), np.float32)
        with pytest.raises(ValueError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=images,
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "not finite" in str(exc_info.value)

    def test_probe_is_removed_when_the_session_fails(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        """The probe file is deleted when the inference session fails."""
        import onnxruntime as ort

        def refuse(*_: object, **__: object) -> None:
            raise RuntimeError("session failed")

        monkeypatch.setattr(ort, "InferenceSession", refuse)
        with pytest.raises(RuntimeError) as exc_info:
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert "session failed" in str(exc_info.value)
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    @_requires_onnxconverter_common
    def test_intermediates_are_removed_when_the_rewrite_fails(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path
    ) -> None:
        """Intermediate graphs are deleted when the Q/DQ rewrite fails."""

        def fail(*_: object, **__: object) -> None:
            raise RuntimeError("rewrite failed")

        monkeypatch.setattr(quantize, "insert_qdq", fail)
        calibration = np.zeros((2, 3, 4, 6), np.float32)
        with pytest.raises(RuntimeError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=calibration,
                max_images=10,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "rewrite failed" in str(exc_info.value)
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    def test_max_images_caps_a_calibration_directory(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path, tmp_path: Path
    ) -> None:
        """max_images limits how many images of a calibration directory are read."""
        from PIL import Image

        images = tmp_path / "images"
        images.mkdir()
        for index in range(3):
            Image.new("RGB", (6, 4)).save(images / f"{index}.png")
        seen: list[int] = []

        def count(onnx_path: str, model: Any, tensors: list[str], batches: Any) -> dict[str, float]:
            seen.append(len(list(batches)))
            raise RuntimeError("stop after counting")

        monkeypatch.setattr(quantize, "calibrate_ranges", count)
        with pytest.raises(RuntimeError) as exc_info:
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=str(images),
                max_images=2,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                pass
        assert "stop after counting" in str(exc_info.value)
        assert seen == [2]

    @_requires_onnxconverter_common
    def test_source_graph_is_quantized_and_cleaned_up(self, image_graph: Path) -> None:
        """The quantized source graph is valid, has one quantize node and is deleted on exit."""
        calibration = np.random.default_rng(0).standard_normal((3, 3, 4, 6)).astype(np.float32)
        with quantize.int8_source_graph(
            str(image_graph),
            calibration_data=calibration,
            max_images=10,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as path:
            model = onnx.load(path)
            assert path != str(image_graph)
            onnx.checker.check_model(model, full_check=True)
            assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 1
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    @pytest.mark.parametrize(
        "error", [pytest.param(RuntimeError, id="error"), pytest.param(KeyboardInterrupt, id="interrupt")]
    )
    @_requires_onnxconverter_common
    def test_quantized_graph_is_removed_when_the_build_fails(
        self, image_graph: Path, error: type[BaseException]
    ) -> None:
        """The quantized graph is deleted when the code using it raises, an interrupt included.

        TensorRT parses and builds inside the ``with`` block; a failure or Ctrl+C there must not leave the intermediate
        graph beside the user's export.
        """
        images = np.zeros((2, 3, 4, 6), np.float32)
        with pytest.raises(error):
            with quantize.int8_source_graph(
                str(image_graph),
                calibration_data=images,
                max_images=10,
                dynamic_batch=False,
                fp16_graph=tensorrt_export.fp16_source_graph,
            ):
                raise error()
        assert sorted(os.listdir(image_graph.parent)) == ["image.onnx"]

    @_requires_onnxconverter_common
    def test_quantized_graph_that_cannot_be_removed_is_reported(
        self, monkeypatch: pytest.MonkeyPatch, image_graph: Path
    ) -> None:
        """A quantized graph the cleanup cannot delete (a Windows file lock, say) is logged rather than left silently.

        Deletion is patched to fail only inside the ``with`` block, so calibration and its own warnings run as usual.
        """
        messages: list[str] = []

        def locked(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("locked")

        images = np.zeros((2, 3, 4, 6), np.float32)
        with quantize.int8_source_graph(
            str(image_graph),
            calibration_data=images,
            max_images=10,
            dynamic_batch=False,
            fp16_graph=tensorrt_export.fp16_source_graph,
        ) as path:
            monkeypatch.setattr(Path, "unlink", locked)
            monkeypatch.setattr(quantize.logger, "warning", messages.append)
        assert [path in message for message in messages] == [True]

    def test_probe_that_cannot_be_removed_is_reported(self, monkeypatch: pytest.MonkeyPatch, image_graph: Path) -> None:
        """A calibration probe the cleanup cannot delete is logged rather than left silently.

        The session is made to fail so that calibration ends right after the probe was written.
        """
        import onnxruntime as ort

        messages: list[str] = []

        def refuse(*_: object, **__: object) -> None:
            raise RuntimeError("session failed")

        def locked(self: Path, missing_ok: bool = False) -> None:
            raise PermissionError("locked")

        monkeypatch.setattr(ort, "InferenceSession", refuse)
        monkeypatch.setattr(Path, "unlink", locked)
        monkeypatch.setattr(quantize.logger, "warning", messages.append)
        with pytest.raises(RuntimeError) as exc_info:
            quantize.calibrate_ranges(str(image_graph), onnx.load(image_graph), ["input"], [np.zeros((1, 3, 4, 6))])
        assert "session failed" in str(exc_info.value)
        assert ["image.calibration-" in message for message in messages] == [True]


#: The two layers of the MLP :func:`_gelu_mlp` builds under the backbone prefix, and the ``Add`` of each one's bias.
MLP_FC1, MLP_FC2 = f"{BACKBONE}mlp/fc1/MatMul", f"{BACKBONE}mlp/fc2/MatMul"
MLP_ADD1, MLP_ADD2 = f"{BACKBONE}mlp/fc1/Add", f"{BACKBONE}mlp/fc2/Add"


def _gelu_mlp(
    prefix: str = BACKBONE,
    *,
    activation: str = "Erf",
    outputs: tuple[str, ...] = ("dets", "labels"),
    norm: bool = False,
    batch: int = 1,
) -> Any:
    """Build ``image -> rows -> fc1 + b1 -> activation -> fc2 + b2`` with RF-DETR-style names, as a ``ModelProto``.

    The ``(1, 2, 4, 4)`` image is read as 4 rows of 8 channels; ``fc1`` widens them to 32 and ``fc2`` narrows them back
    to 8. Each of *outputs* (a detector's by default) is a copy of the MLP's output. ``activation="Erf"`` writes GELU in
    its traced erf form, ``"Gelu"`` as the opset-20 op. ``norm=True`` puts a ``LayerNormalization`` (``norm2``, one
    channel far wider than the others) between the rows and ``fc1``, as a backbone block does. *batch* is the static
    batch of the image input.

    Examples:
        >>> [node.op_type for node in _gelu_mlp(activation="Relu").graph.node][:5]
        ['Reshape', 'MatMul', 'Add', 'Relu', 'MatMul']
    """
    rng = np.random.default_rng(3)
    inits = [
        numpy_helper.from_array(np.array([batch, 4, 8], np.int64), "rows_shape"),
        numpy_helper.from_array((rng.standard_normal((8, 32)) * 0.5).astype(np.float32), "w1"),
        numpy_helper.from_array((rng.standard_normal(32) * 0.1).astype(np.float32), "b1"),
        numpy_helper.from_array((rng.standard_normal((32, 8)) * 0.5).astype(np.float32), "w2"),
        numpy_helper.from_array((rng.standard_normal(8) * 0.1).astype(np.float32), "b2"),
        *(
            numpy_helper.from_array(np.array(v, np.float32), n)
            for n, v in (("sqrt2", 1.4142135), ("one", 1.0), ("half", 0.5))
        ),
    ]
    p = f"{prefix}mlp/"
    if norm:
        inits += [
            numpy_helper.from_array(np.array([8.0, *[1.0] * 7], np.float32), "gamma"),
            numpy_helper.from_array(np.full(8, 0.1, np.float32), "beta"),
        ]
    nodes = [
        helper.make_node("Reshape", ["input", "rows_shape"], ["rows0" if norm else "rows"], f"{prefix}rows/Reshape"),
        *(
            [
                helper.make_node(
                    "LayerNormalization", ["rows0", "gamma", "beta"], ["rows"], f"{prefix}norm2/LayerNormalization"
                )
            ]
            if norm
            else []
        ),
        helper.make_node("MatMul", ["rows", "w1"], ["h0"], f"{p}fc1/MatMul"),
        helper.make_node("Add", ["h0", "b1"], ["h"], f"{p}fc1/Add"),
    ]
    if activation == "Erf":
        nodes += [
            helper.make_node("Div", ["h", "sqrt2"], ["h1"], f"{p}act/Div"),
            helper.make_node("Erf", ["h1"], ["h2"], f"{p}act/Erf"),
            helper.make_node("Add", ["h2", "one"], ["h3"], f"{p}act/Add"),
            helper.make_node("Mul", ["h", "h3"], ["h4"], f"{p}act/Mul"),
            helper.make_node("Mul", ["h4", "half"], ["g"], f"{p}act/Mul_1"),
        ]
    else:
        nodes.append(helper.make_node(activation, ["h"], ["g"], f"{p}act/{activation}"))
    nodes += [
        helper.make_node("MatMul", ["g", "w2"], ["y0"], f"{p}fc2/MatMul"),
        helper.make_node("Add", ["y0", "b2"], ["y"], f"{p}fc2/Add"),
        *(helper.make_node("Identity", ["y"], [name], name) for name in outputs),
    ]
    graph = helper.make_graph(
        nodes,
        "gelu_mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 2, 4, 4])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, 4, 8]) for name in outputs],
        inits,
    )
    # Pinned: onnx 1.21 writes IR version 14 by default, which the onnxruntime CI installs (IR 13 at most) refuses.
    opset = 20 if activation == "Gelu" else 17
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


#: The ``LayerNormalization`` :func:`_norm_graph` builds under the backbone prefix, and the two projections reading it.
NORM, NORM_Q, NORM_K = f"{BACKBONE}norm1/LayerNormalization", f"{BACKBONE}q/MatMul", f"{BACKBONE}k/MatMul"


def _norm_graph(prefix: str = BACKBONE, *, axis: int = -1) -> Any:
    """Build ``x -> LayerNormalization -> q, k`` projections with RF-DETR-style names, as a ``ModelProto``.

    ``x`` is 4 rows of 8 channels; the norm's scale makes channel 0 eight times wider than the others, and ``q`` and
    ``k`` are two ``MatMul`` nodes reading the norm's output, each an output of the graph.

    Examples:
        >>> [node.op_type for node in _norm_graph().graph.node]
        ['LayerNormalization', 'MatMul', 'MatMul']
    """
    rng = np.random.default_rng(8)
    inits = [
        numpy_helper.from_array(np.array([8.0, *[1.0] * 7], np.float32), "gamma"),
        numpy_helper.from_array((rng.standard_normal(8) * 0.1).astype(np.float32), "beta"),
        numpy_helper.from_array((rng.standard_normal((8, 8)) * 0.5).astype(np.float32), "wq"),
        numpy_helper.from_array((rng.standard_normal((8, 8)) * 0.5).astype(np.float32), "wk"),
    ]
    nodes = [
        helper.make_node(
            "LayerNormalization", ["x", "gamma", "beta"], ["n"], f"{prefix}norm1/LayerNormalization", axis=axis
        ),
        helper.make_node("MatMul", ["n", "wq"], ["q"], f"{prefix}q/MatMul"),
        helper.make_node("MatMul", ["n", "wk"], ["k"], f"{prefix}k/MatMul"),
    ]
    graph = helper.make_graph(
        nodes,
        "norm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4, 8]) for name in ("q", "k")],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def _two_norm_graph() -> Any:
    """Build ``x -> norm1 -> up -> norm2 -> down`` with RF-DETR-style names, two norms of different widths.

    ``x`` is 4 rows of 8 channels. ``norm1`` normalizes them and ``up`` widens them to 16 channels, which ``norm2``
    normalizes before ``down`` narrows them back to 8, the graph's output. Each norm is read by one projection only, so
    both can be smoothed, and each norm's scale makes its channel 0 wider than the others.

    Examples:
        >>> [node.op_type for node in _two_norm_graph().graph.node]
        ['LayerNormalization', 'MatMul', 'LayerNormalization', 'MatMul']
    """
    rng = np.random.default_rng(10)
    inits = [
        numpy_helper.from_array(np.array([8.0, *[1.0] * 7], np.float32), "g1"),
        numpy_helper.from_array((rng.standard_normal(8) * 0.1).astype(np.float32), "b1"),
        numpy_helper.from_array((rng.standard_normal((8, 16)) * 0.5).astype(np.float32), "w1"),
        numpy_helper.from_array(np.array([4.0, *[1.0] * 15], np.float32), "g2"),
        numpy_helper.from_array((rng.standard_normal(16) * 0.1).astype(np.float32), "b2"),
        numpy_helper.from_array((rng.standard_normal((16, 8)) * 0.5).astype(np.float32), "w2"),
    ]
    nodes = [
        helper.make_node("LayerNormalization", ["x", "g1", "b1"], ["n1"], f"{BACKBONE}norm1/LayerNormalization"),
        helper.make_node("MatMul", ["n1", "w1"], ["h"], f"{BACKBONE}up/MatMul"),
        helper.make_node("LayerNormalization", ["h", "g2", "b2"], ["n2"], f"{BACKBONE}norm2/LayerNormalization"),
        helper.make_node("MatMul", ["n2", "w2"], ["y"], f"{BACKBONE}down/MatMul"),
    ]
    graph = helper.make_graph(
        nodes,
        "two_norms",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    return model


def _layer_norm(rows: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Normalize each row over its last axis as ``LayerNormalization`` does by default, then scale and shift it.

    The variance is the population one and epsilon is ONNX's default ``1e-5``.

    Examples:
        >>> _layer_norm(np.array([[1.0, 3.0]]), np.array([2.0, 1.0]), np.array([0.0, 1.0])).round(3).tolist()
        [[-2.0, 2.0]]
    """
    normalized = (rows - rows.mean(-1, keepdims=True)) / np.sqrt(rows.var(-1, keepdims=True) + 1e-5)
    return normalized * gamma + beta


def _smoothing_scales(activation: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return SmoothQuant's per-channel scales at alpha 0.5, divided by their geometric mean.

    *activation* holds each channel's absolute maximum and *weight* the largest magnitude of the weight rows reading
    it. A channel where either is 0 starts at a raw scale of 1, which the geometric-mean division then applies to as
    well: it ends at 1 over the geometric mean of the raw scales, not at 1.

    Examples:
        >>> _smoothing_scales(np.array([4.0, 0.0]), np.array([1.0, 1.0])).round(4).tolist()
        [1.4142, 0.7071]
    """
    alive = (activation > 0) & (weight > 0)
    raw = np.ones(activation.shape, np.float64)
    raw[alive] = np.sqrt(activation[alive] / weight[alive])
    return raw / np.exp(np.log(raw).mean())


def _read_the_norm_elsewhere(model: Any) -> None:
    """Give the norm's output a reader that is not a projection.

    Examples:
        >>> model = _norm_graph()
        >>> _read_the_norm_elsewhere(model)
        >>> model.graph.node[-1].op_type
        'Identity'
    """
    model.graph.node.append(helper.make_node("Identity", ["n"], ["n_copy"], "tap"))


def _multiply_by_the_norm(model: Any) -> None:
    """Make ``k`` read the norm's output as its second operand, so the norm is not what its rows multiply.

    Examples:
        >>> model = _norm_graph()
        >>> _multiply_by_the_norm(model)
        >>> list(model.graph.node[2].input)
        ['wk', 'n']
    """
    model.graph.node[2].input[:] = ["wk", "n"]


def _root_the_norm_input(model: Any) -> None:
    """Make the norm read the square root of ``x``, so a negative input value becomes a NaN inside the graph.

    Examples:
        >>> model = _norm_graph()
        >>> _root_the_norm_input(model)
        >>> [(node.op_type, list(node.input)) for node in model.graph.node][:2]
        [('Sqrt', ['x']), ('LayerNormalization', ['x_root', 'gamma', 'beta'])]
    """
    model.graph.node[0].input[0] = "x_root"
    model.graph.node.insert(0, helper.make_node("Sqrt", ["x"], ["x_root"], "root"))


def _feed_the_scale(model: Any) -> None:
    """Make the norm's scale a graph input instead of a constant.

    Examples:
        >>> model = _norm_graph()
        >>> _feed_the_scale(model)
        >>> list(model.graph.node[0].input)
        ['x', 'scale_in', 'beta']
    """
    model.graph.node[0].input[1] = "scale_in"
    model.graph.input.append(helper.make_tensor_value_info("scale_in", TensorProto.FLOAT, [8]))


def _expose_the_norm(model: Any) -> None:
    """Make the norm's output a graph output, which smoothing would change.

    Examples:
        >>> model = _norm_graph()
        >>> _expose_the_norm(model)
        >>> model.graph.output[-1].name
        'n'
    """
    model.graph.output.append(helper.make_tensor_value_info("n", TensorProto.FLOAT, [1, 4, 8]))


def _read_the_norm_in_a_subgraph(model: Any) -> None:
    """Give the norm's output a reader inside an ``If`` branch, which captures it from the outer graph.

    Examples:
        >>> model = _norm_graph()
        >>> _read_the_norm_in_a_subgraph(model)
        >>> model.graph.node[-1].op_type
        'If'
    """
    branch = helper.make_graph(
        [helper.make_node("Identity", ["n"], ["n_branch"], "branch_tap")],
        "branch",
        [],
        [helper.make_tensor_value_info("n_branch", TensorProto.FLOAT, [1, 4, 8])],
    )
    model.graph.initializer.append(numpy_helper.from_array(np.array(True), "cond"))
    model.graph.node.append(helper.make_node("If", ["cond"], ["n_if"], "if", then_branch=branch, else_branch=branch))
    model.graph.output.append(helper.make_tensor_value_info("n_if", TensorProto.FLOAT, [1, 4, 8]))


def _drop_the_norm_bias(model: Any) -> None:
    """Make the norm one without a bias input, which ONNX allows.

    Examples:
        >>> model = _norm_graph()
        >>> _drop_the_norm_bias(model)
        >>> list(model.graph.node[0].input)
        ['x', 'gamma']
    """
    del model.graph.node[0].input[2]


def _raise_the_weight_rank(model: Any) -> None:
    """Store ``k``'s weight as ``(8, 8, 1)``, which has no single row per channel to scale.

    Examples:
        >>> model = _norm_graph()
        >>> _raise_the_weight_rank(model)
        >>> next(init.dims for init in model.graph.initializer if init.name == "wk")
        [8, 8, 1]
    """
    weight = next(init for init in model.graph.initializer if init.name == "wk")
    weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight).reshape(8, 8, 1), "wk"))


def _cast_before_the_projection(model: Any) -> None:
    """Put a ``Cast`` between the norm and ``q``, as an FP16 cast that kept the norm in FP32 would.

    Examples:
        >>> model = _norm_graph()
        >>> _cast_before_the_projection(model)
        >>> [node.op_type for node in model.graph.node]
        ['LayerNormalization', 'Cast', 'MatMul', 'MatMul']
    """
    model.graph.node[1].input[0] = "n_cast"
    model.graph.node.insert(1, helper.make_node("Cast", ["n"], ["n_cast"], "cast", to=TensorProto.FLOAT))


def _add_a_constant_to_the_norm(model: Any) -> None:
    """Give the norm a reader that adds a per-channel constant, which smoothing cannot compensate.

    Examples:
        >>> model = _norm_graph()
        >>> _add_a_constant_to_the_norm(model)
        >>> model.graph.node[-1].op_type
        'Add'
    """
    model.graph.initializer.append(numpy_helper.from_array(np.ones(8, np.float32), "shift"))
    model.graph.node.append(helper.make_node("Add", ["n", "shift"], ["n_shifted"], "shift_add"))


def _name_another_node_like_the_norm(model: Any) -> None:
    """Give a second node the norm's name, so a plan cannot tell the two apart.

    Examples:
        >>> model = _norm_graph()
        >>> _name_another_node_like_the_norm(model)
        >>> sum(node.name == NORM for node in model.graph.node)
        2
    """
    model.graph.node.append(helper.make_node("Identity", ["q"], ["q_copy"], NORM))


def _name_another_node_like_a_projection(model: Any) -> None:
    """Give a second node the name of ``q``, one of the norm's readers.

    Examples:
        >>> model = _norm_graph()
        >>> _name_another_node_like_a_projection(model)
        >>> sum(node.name == NORM_Q for node in model.graph.node)
        2
    """
    model.graph.node.append(helper.make_node("Identity", ["k"], ["k_copy"], NORM_Q))


def _read_the_norm_with_a_gemm(model: Any) -> None:
    """Turn ``q`` into a ``Gemm``, whose weight rows a smoothing scale is not known to reach.

    Examples:
        >>> model = _norm_graph()
        >>> _read_the_norm_with_a_gemm(model)
        >>> model.graph.node[1].op_type
        'Gemm'
    """
    model.graph.node[1].op_type = "Gemm"


def _make_the_norm_a_group_norm(model: Any) -> None:
    """Turn the norm into a ``GroupNormalization``, which also reads a scale and a bias but is not a LayerNorm.

    Examples:
        >>> model = _norm_graph()
        >>> _make_the_norm_a_group_norm(model)
        >>> next(node.op_type for node in model.graph.node if node.name == NORM)
        'GroupNormalization'
    """
    next(node for node in model.graph.node if node.name == NORM).op_type = "GroupNormalization"


def _drop_the_projections(model: Any) -> None:
    """Remove both readers of the norm, so nothing undoes a scale of its output.

    Examples:
        >>> model = _norm_graph()
        >>> _drop_the_projections(model)
        >>> [node.op_type for node in model.graph.node]
        ['LayerNormalization']
    """
    del model.graph.node[1:]


class TestSmoothableNorms:
    """SmoothQuant scales a backbone norm only where every reader is a projection whose rows can absorb the scale."""

    def test_a_backbone_norm_read_by_projections_is_smoothed(self) -> None:
        """A backbone norm read only by planned projections is smoothable."""
        model = _norm_graph()
        assert quantize._smoothable_norms(model, quantize.plan_int8(model)) == {NORM: (NORM_Q, NORM_K)}

    @pytest.mark.parametrize(
        "mutate",
        [
            _read_the_norm_elsewhere,
            _multiply_by_the_norm,
            _feed_the_scale,
            _expose_the_norm,
            _read_the_norm_in_a_subgraph,
            _drop_the_norm_bias,
            _raise_the_weight_rank,
            _read_the_norm_with_a_gemm,
            _name_another_node_like_the_norm,
            _name_another_node_like_a_projection,
        ],
    )
    def test_a_norm_a_projection_cannot_absorb_is_left_alone(self, mutate: Any) -> None:
        """A norm with a reader that cannot absorb the scale is left alone."""
        model = _norm_graph()
        plan = quantize.plan_int8(model)
        mutate(model)
        assert quantize._smoothable_norms(model, plan) == {}

    @pytest.mark.parametrize(("prefix", "axis"), [(DECODER, -1), (BACKBONE, 1)])
    def test_a_norm_outside_the_backbone_or_across_another_axis_is_left_alone(self, prefix: str, axis: int) -> None:
        """Decoder norms and norms over another axis are not smoothable."""
        model = _norm_graph(prefix, axis=axis)
        assert quantize._smoothable_norms(model, quantize.plan_int8(model)) == {}

    def test_one_quantized_reader_is_enough(self) -> None:
        """A norm with a single quantized reader is smoothable."""
        plan = quantize.Int8Plan(weighted=(NORM_Q,), activations=())
        assert quantize._smoothable_norms(_norm_graph(), plan) == {NORM: (NORM_Q, NORM_K)}

    def test_a_norm_whose_readers_all_stay_fp16_is_left_alone(self) -> None:
        """A norm whose readers all stay FP16 is not smoothable."""
        model = _norm_graph()
        assert quantize._smoothable_norms(model, quantize.Int8Plan(weighted=(), activations=())) == {}


class TestSmoothNorms:
    """Smoothing divides the norm's channels and multiplies the projections' rows, in new constants of their own."""

    @staticmethod
    def _smoothed(scale: np.ndarray) -> tuple[Any, dict[str, np.ndarray]]:
        """Smooth :func:`_norm_graph` with *scale*; return it and the value each node now reads at each input.

        Examples:
            >>> model, values = TestSmoothNorms._smoothed(np.ones(8))
            >>> sorted(values)
            ['beta', 'gamma', 'wk', 'wq']
        """
        model = _norm_graph()
        original = {node.name: list(node.input) for node in model.graph.node}
        quantize._smooth_norms(model, {NORM: scale})
        constants = quantize._constant_weights(model.graph)
        values = {}
        for node in model.graph.node:
            for before, after in zip(original[node.name], node.input):
                if after in constants:
                    values[before] = numpy_helper.to_array(constants[after]).astype(np.float64)
        return model, values

    @pytest.mark.parametrize(("constant", "operation"), [("gamma", "divide"), ("beta", "divide"), ("wq", "multiply")])
    def test_each_constant_is_rescaled_per_channel(self, constant: str, operation: str) -> None:
        """The norm's scale and bias are divided and the projection rows multiplied by the scale."""
        scale = np.linspace(0.5, 2.0, 8)
        _, values = self._smoothed(scale)
        original = numpy_helper.to_array(next(i for i in _norm_graph().graph.initializer if i.name == constant))
        expected = original / scale if operation == "divide" else original * scale[:, None]
        np.testing.assert_allclose(values[constant], expected, rtol=1e-6)

    def test_a_constant_shared_with_another_node_keeps_its_value_for_it(self) -> None:
        """A constant shared with another node keeps its value for that node."""
        model = _norm_graph()
        model.graph.node.append(helper.make_node("Identity", ["gamma"], ["gamma_copy"], "copy"))
        quantize._smooth_norms(model, {NORM: np.full(8, 2.0)})
        assert list(model.graph.node[-1].input) == ["gamma"]

    @pytest.mark.parametrize("name", ["missing", NORM_Q])
    def test_a_norm_the_graph_lacks_is_refused(self, name: str) -> None:
        """Smoothing a norm the graph lacks, or a node that is not a norm, is refused."""
        with pytest.raises(ValueError) as exc_info:
            quantize._smooth_norms(_norm_graph(), {name: np.ones(8)})
        assert "has no LayerNormalization" in str(exc_info.value)

    @pytest.mark.parametrize(
        "mutate",
        [
            _cast_before_the_projection,
            _add_a_constant_to_the_norm,
            _drop_the_norm_bias,
            _read_the_norm_with_a_gemm,
            _make_the_norm_a_group_norm,
            _name_another_node_like_the_norm,
            _feed_the_scale,
            _drop_the_projections,
        ],
    )
    def test_a_norm_whose_copy_differs_is_refused(self, mutate: Any) -> None:
        """A norm whose FP16 copy differs from the planned one is refused."""
        model = _norm_graph()
        mutate(model)
        with pytest.raises(ValueError) as exc_info:
            quantize._smooth_norms(model, {NORM: np.ones(8)})
        assert "has no LayerNormalization" in str(exc_info.value)


@_requires_onnxruntime
@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestSmoothingScales:
    """The scales are SmoothQuant's, measured on the FP32 graph, and smoothing leaves the graph's outputs unchanged."""

    @pytest.fixture
    def norm(self, tmp_path: Path) -> tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]:
        """The saved norm graph, its plan and four inputs.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path, model, plan, inputs = norm(tmp_path)  # doctest: +SKIP
        """
        model = _norm_graph()
        path = tmp_path / "norm.onnx"
        onnx.save(model, path)
        inputs = list(np.random.default_rng(9).standard_normal((4, 1, 4, 8)).astype(np.float32))
        return path, model, quantize.plan_int8(model), inputs

    @pytest.mark.parametrize(
        "dead",
        [
            pytest.param((), id="all-channels-live"),
            pytest.param(("gamma", "beta"), id="dead-channel"),
            pytest.param(("wq", "wk"), id="dead-weight-row"),
        ],
    )
    def test_scales_balance_the_activation_and_weight_ranges(
        self, norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]], dead: tuple[str, ...]
    ) -> None:
        """Scales balance activation and weight ranges, a dead channel or weight row included.

        A channel with no activation or no weight starts at scale 1, then shares the norm's geometric-mean division.

        Zeroing channel 3 of the norm's scale and bias silences its activation; zeroing row 3 of both projections
        silences its weight. SmoothQuant's ratio is undefined for it, so its raw scale is 1, and it ends at 1 over the
        geometric mean of the raw scales, not at 1: a scale kept at 1 would no longer match the other channels' sizes.
        """
        path, model, plan, inputs = norm
        for name in dead:
            init = next(i for i in model.graph.initializer if i.name == name)
            values = numpy_helper.to_array(init).copy()
            values[3] = 0.0
            init.CopyFrom(numpy_helper.from_array(values, name))
        weights = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
        rows = np.concatenate([x.reshape(4, 8) for x in inputs]).astype(np.float64)
        activation = np.abs(_layer_norm(rows, weights["gamma"], weights["beta"])).max(0)
        weight = np.maximum(np.abs(weights["wq"]).max(1), np.abs(weights["wk"]).max(1))
        scales = quantize.measure_smoothing_scales(str(path), model, plan, inputs)
        np.testing.assert_allclose(scales[NORM], _smoothing_scales(activation, weight), rtol=1e-4)

    def test_two_norms_of_different_widths_each_get_their_own_scales(self, tmp_path: Path) -> None:
        """Each of two chained norms, 8 and 16 channels wide, gets the scales measured on its own channels.

        One probe reads every norm's channel maxima in a single pass and pairs them back with the norms by position, so
        a slip in that pairing would give one norm the other's maxima, of the other's width.
        """
        model = _two_norm_graph()
        path = tmp_path / "norms.onnx"
        onnx.save(model, path)
        inputs = list(np.random.default_rng(11).standard_normal((4, 1, 4, 8)).astype(np.float32))
        weights = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
        rows = np.concatenate([x.reshape(4, 8) for x in inputs]).astype(np.float64)
        narrow = _layer_norm(rows, weights["g1"], weights["b1"])
        wide = _layer_norm(narrow @ weights["w1"], weights["g2"], weights["b2"])
        expected = [
            _smoothing_scales(np.abs(narrow).max(0), np.abs(weights["w1"]).max(1)),
            _smoothing_scales(np.abs(wide).max(0), np.abs(weights["w2"]).max(1)),
        ]
        scales = quantize.measure_smoothing_scales(str(path), model, quantize.plan_int8(model), inputs)
        norms = [f"{BACKBONE}norm1/LayerNormalization", f"{BACKBONE}norm2/LayerNormalization"]
        np.testing.assert_allclose(np.concatenate([scales[n] for n in norms]), np.concatenate(expected), rtol=1e-4)

    def test_a_norm_whose_scaled_weights_would_not_fit_fp16_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]
    ) -> None:
        """A norm whose scaled projection weights exceed the FP16 range is left unsmoothed."""
        path, model, plan, inputs = norm
        weight = next(i for i in model.graph.initializer if i.name == "wk")
        weight.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(weight) * 1e4, "wk"))  # large rows, small scales
        scale = quantize.measure_smoothing_scales(str(path), model, plan, inputs)[NORM]
        values = {i.name: numpy_helper.to_array(i).astype(np.float64) for i in model.graph.initializer}
        norm_terms = max(np.abs(values[name] / scale).max() for name in ("gamma", "beta"))
        weights = max(np.abs(values[name] * scale[:, None]).max() for name in ("wq", "wk"))
        assert norm_terms < weights  # precondition: a limit between the two refuses the weights alone
        monkeypatch.setattr(quantize, "_MAX_FP16", (norm_terms + weights) / 2)
        assert quantize.measure_smoothing_scales(str(path), model, plan, inputs) == {}

    def test_a_norm_with_an_infinite_range_is_refused(
        self, norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]
    ) -> None:
        """A norm output that is infinite on the calibration images is refused, not silently left unsmoothed.

        Leaving the norm out would only defer the refusal to the calibration pass, which finds the same range after one
        more full pass over the images.
        """
        path, model, plan, inputs = norm
        init = next(i for i in model.graph.initializer if i.name == "gamma")
        values = numpy_helper.to_array(init).copy()
        values[0] = np.inf
        init.CopyFrom(numpy_helper.from_array(values, "gamma"))
        with pytest.raises(ValueError) as exc_info:
            quantize.measure_smoothing_scales(str(path), model, plan, inputs)
        assert "norm output" in str(exc_info.value) and "not finite" in str(exc_info.value)

    def test_a_nan_the_norm_computes_after_its_first_row_is_refused(self, tmp_path: Path) -> None:
        """A NaN the graph computes in a later row of the norm output is refused, though ReduceMax would skip it.

        onnxruntime's ``ReduceMax`` skips a NaN anywhere but the first element it reduces, so without the guard the
        probe reports a finite range for the NaN channel. ``Sqrt`` of a negative value in the second row of a finite
        image puts the NaN there.
        """
        model = _norm_graph()
        _root_the_norm_input(model)
        path = tmp_path / "norm.onnx"
        onnx.save(model, path)
        image = np.ones((1, 4, 8), np.float32)
        image[0, 1, 2] = -1.0
        with pytest.raises(ValueError) as exc_info:
            quantize.measure_smoothing_scales(str(path), model, quantize.plan_int8(model), [image])
        assert "norm output" in str(exc_info.value) and "not finite" in str(exc_info.value)

    def test_a_graph_without_a_smoothable_norm_reads_no_image(self, tmp_path: Path) -> None:
        """A graph with no smoothable norm yields no scales and reads no image."""
        model = _gelu_mlp()
        assert quantize.measure_smoothing_scales(str(tmp_path / "mlp.onnx"), model, quantize.plan_int8(model), []) == {}

    def test_a_norm_that_would_not_fit_fp16_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]
    ) -> None:
        """A norm whose smoothed constants exceed the FP16 maximum is left unsmoothed."""
        path, model, plan, inputs = norm
        monkeypatch.setattr(quantize, "_MAX_FP16", 1.0)
        assert quantize.measure_smoothing_scales(str(path), model, plan, inputs) == {}

    def test_a_norm_left_alone_is_logged_with_the_constant_that_would_not_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]],
    ) -> None:
        """A norm left unsmoothed is logged at debug level by name, with the smoothed constant past the FP16 maximum.

        Only the count of smoothed norms is logged at info level, so without this a missing norm could not be traced.
        """
        path, model, plan, inputs = norm
        monkeypatch.setattr(quantize, "_MAX_FP16", 1.0)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        with caplog.at_level(logging.DEBUG, logger="rf-detr"):
            quantize.measure_smoothing_scales(str(path), model, plan, inputs)
        # Repeated objects are one emission under pytest>=9.1; see the calibration tests' range warnings.
        messages = [
            record.getMessage() for record in dict.fromkeys(caplog.records) if "unsmoothed" in record.getMessage()
        ]
        assert [(m.partition(": its smoothed ")[0], m.endswith("past the FP16 maximum 1.")) for m in messages] == [
            (f"Left backbone LayerNorm {NORM!r} unsmoothed for INT8", True)
        ]

    def test_smoothing_leaves_the_outputs_unchanged(
        self, norm: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]
    ) -> None:
        """Smoothing a norm leaves the graph's outputs unchanged."""
        import onnxruntime as ort

        path, model, plan, inputs = norm
        scales = quantize.measure_smoothing_scales(str(path), model, plan, inputs)
        before = ort.InferenceSession(model.SerializeToString()).run(None, {"x": inputs[0]})
        quantize._smooth_norms(model, scales)
        after = ort.InferenceSession(model.SerializeToString()).run(None, {"x": inputs[0]})
        np.testing.assert_allclose(np.stack(after), np.stack(before), rtol=1e-4, atol=1e-5)


def _share_the_bias(model: Any) -> None:
    """Make a second node read ``fc2``'s bias, as the exporter does when two layers have identical biases.

    Examples:
        >>> model = _gelu_mlp()
        >>> _share_the_bias(model)
        >>> [node.op_type for node in model.graph.node][-1]
        'Identity'
    """
    model.graph.node.append(helper.make_node("Identity", ["b2"], ["b2_copy"], "copy"))


def _expose_the_bias(model: Any) -> None:
    """Make ``fc2``'s bias a graph output.

    Examples:
        >>> model = _gelu_mlp()
        >>> _expose_the_bias(model)
        >>> model.graph.output[-1].name
        'b2'
    """
    model.graph.output.append(helper.make_tensor_value_info("b2", TensorProto.FLOAT, [8]))


def _alias_the_bias(model: Any) -> None:
    """Make ``fc2``'s ``Add`` read its bias through an ``Identity``, as the exporter writes a deduplicated bias.

    Examples:
        >>> model = _gelu_mlp()
        >>> _alias_the_bias(model)
        >>> [node.op_type for node in model.graph.node][0]
        'Identity'
    """
    add = next(node for node in model.graph.node if node.name == MLP_ADD2)
    add.input[1] = "b2_alias"
    model.graph.node.insert(0, helper.make_node("Identity", ["b2"], ["b2_alias"], "alias"))


def _tap_the_product(model: Any) -> None:
    """Give ``fc2``'s product a second reader besides its bias ``Add``.

    Examples:
        >>> model = _gelu_mlp()
        >>> _tap_the_product(model)
        >>> list(model.graph.node[-1].input)
        ['y0']
    """
    model.graph.node.append(helper.make_node("Identity", ["y0"], ["tap"], "tap"))


def _broadcast_the_bias(model: Any) -> None:
    """Store ``fc2``'s bias as ``(1, 1, 8)`` rather than one value per output channel.

    Examples:
        >>> model = _gelu_mlp()
        >>> _broadcast_the_bias(model)
        >>> next(init.dims for init in model.graph.initializer if init.name == "b2")
        [1, 1, 8]
    """
    bias = next(init for init in model.graph.initializer if init.name == "b2")
    bias.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(bias).reshape(1, 1, 8), "b2"))


def _add_an_activation(model: Any) -> None:
    """Make ``fc2``'s ``Add`` add an activation instead of a constant.

    Examples:
        >>> model = _gelu_mlp()
        >>> _add_an_activation(model)
        >>> list(next(node for node in model.graph.node if node.name == MLP_ADD2).input)
        ['y0', 'rows']
    """
    next(node for node in model.graph.node if node.name == MLP_ADD2).input[1] = "rows"


def _double_the_product(model: Any) -> None:
    """Make ``fc2``'s ``Add`` add its product to itself.

    Examples:
        >>> model = _gelu_mlp()
        >>> _double_the_product(model)
        >>> list(next(node for node in model.graph.node if node.name == MLP_ADD2).input)
        ['y0', 'y0']
    """
    next(node for node in model.graph.node if node.name == MLP_ADD2).input[1] = "y0"


def _scale_instead_of_bias(model: Any) -> None:
    """Make ``fc2``'s product go through a ``Mul`` by a per-channel constant, as LayerScale does, instead of an ``Add``.

    Examples:
        >>> model = _gelu_mlp()
        >>> _scale_instead_of_bias(model)
        >>> next(node for node in model.graph.node if node.name == MLP_ADD2).op_type
        'Mul'
    """
    next(node for node in model.graph.node if node.name == MLP_ADD2).op_type = "Mul"


def _name_another_node_like_the_add(model: Any) -> None:
    """Give a second node the name of ``fc2``'s bias ``Add``, so a plan cannot tell the two apart.

    Examples:
        >>> model = _gelu_mlp()
        >>> _name_another_node_like_the_add(model)
        >>> sum(node.name == MLP_ADD2 for node in model.graph.node)
        2
    """
    model.graph.node.append(helper.make_node("Identity", ["y"], ["y_copy"], MLP_ADD2))


class TestMlpBiases:
    """The INT8 export corrects the bias of both projections of each backbone GELU MLP, and of nothing else."""

    def test_both_projections_of_a_backbone_gelu_mlp_are_corrected(self) -> None:
        """Both projections of a backbone GELU MLP are selected for bias correction."""
        model = _gelu_mlp()
        assert quantize._mlp_biases(model, quantize.plan_int8(model)) == {MLP_FC1: MLP_ADD1, MLP_FC2: MLP_ADD2}

    @pytest.mark.parametrize(("prefix", "activation"), [(DECODER, "Erf"), (BACKBONE, "Relu"), (BACKBONE, "Gelu")])
    def test_only_a_backbone_gelu_mlp_is_corrected(self, prefix: str, activation: str) -> None:
        """Only a backbone MLP with a GELU activation is selected for bias correction."""
        model = _gelu_mlp(prefix, activation=activation)
        expected = {MLP_FC1: MLP_ADD1, MLP_FC2: MLP_ADD2} if activation == "Gelu" else {}
        assert quantize._mlp_biases(model, quantize.plan_int8(model)) == expected

    @pytest.mark.parametrize("mutate", [_share_the_bias, _expose_the_bias, _alias_the_bias])
    def test_a_bias_read_elsewhere_too_is_still_corrected(self, mutate: Any) -> None:
        """A bias that is also read elsewhere does not stop the correction."""
        model = _gelu_mlp()
        mutate(model)
        assert quantize._mlp_biases(model, quantize.plan_int8(model)) == {MLP_FC1: MLP_ADD1, MLP_FC2: MLP_ADD2}

    @pytest.mark.parametrize(
        "mutate",
        [
            _tap_the_product,
            _broadcast_the_bias,
            _add_an_activation,
            _double_the_product,
            _scale_instead_of_bias,
            _name_another_node_like_the_add,
        ],
    )
    def test_a_layer_without_its_own_bias_add_is_left_alone(self, mutate: Any) -> None:
        """A layer without a bias Add of its own is not corrected; the other projection still is."""
        model = _gelu_mlp()
        mutate(model)
        assert quantize._mlp_biases(model, quantize.plan_int8(model)) == {MLP_FC1: MLP_ADD1}


class TestCorrectBiases:
    """The corrected bias is a new constant that only the corrected ``Add`` reads."""

    @staticmethod
    def _two_adds() -> Any:
        """Two ``Add`` nodes reading one FP16 bias ``b``.

        Examples:
            >>> [node.name for node in TestCorrectBiases._two_adds().graph.node]
            ['first', 'second']
        """
        bias = numpy_helper.from_array(np.array([1.0, 2.0], np.float16), "b")
        nodes = [helper.make_node("Add", [f"y{i}", "b"], [f"z{i}"], name) for i, name in enumerate(("first", "second"))]
        return helper.make_model(helper.make_graph(nodes, "g", [], [], [bias]))

    def test_the_corrected_add_reads_the_bias_minus_the_shift(self) -> None:
        """The corrected Add reads a new bias equal to the old one minus the shift."""
        model = self._two_adds()
        quantize._correct_biases(model, {"first": np.array([0.5, -1.0])})
        constants = quantize._constant_weights(model.graph)
        assert numpy_helper.to_array(constants[model.graph.node[0].input[1]]).tolist() == [0.5, 3.0]

    def test_another_reader_keeps_the_original_bias(self) -> None:
        """An Add that is not corrected keeps reading the original bias."""
        model = self._two_adds()
        quantize._correct_biases(model, {"first": np.array([0.5, -1.0])})
        constants = quantize._constant_weights(model.graph)
        assert numpy_helper.to_array(constants[model.graph.node[1].input[1]]).tolist() == [1.0, 2.0]

    @pytest.mark.parametrize(
        ("corrected", "initializers"),
        [
            pytest.param(("first",), ["b", "b_int8_corrected"], id="still-read-by-another-add"),
            pytest.param(("first", "second"), ["b_int8_corrected", "b_int8_corrected_1"], id="read-by-no-node"),
        ],
    )
    def test_the_original_bias_is_kept_only_while_a_node_reads_it(
        self, corrected: tuple[str, ...], initializers: list[str]
    ) -> None:
        """The replaced bias leaves the graph once no node reads it, and stays while another ``Add`` still does.

        The exporter stores identical biases once, so the shared case is the real one; leaving an unread original in the
        graph would only grow the engine's source by a dead constant per corrected layer.
        """
        model = self._two_adds()
        quantize._correct_biases(model, dict.fromkeys(corrected, np.array([0.5, -1.0])))
        assert sorted(init.name for init in model.graph.initializer) == initializers

    @pytest.mark.parametrize("name", ["missing", "constant-free", "second", "not-an-add"])
    def test_an_add_the_fp16_copy_lacks_is_refused(self, name: str) -> None:
        """Correcting a missing, constant-free, duplicated or non-Add node is refused."""
        model = self._two_adds()
        model.graph.node.append(helper.make_node("Add", ["y0", "y1"], ["z2"], "constant-free"))
        model.graph.node.append(helper.make_node("Sub", ["y0", "b"], ["z4"], "not-an-add"))  # reads a constant
        model.graph.node.append(helper.make_node("Identity", ["z1"], ["z3"], "second"))  # a second node named "second"
        with pytest.raises(ValueError) as exc_info:
            quantize._correct_biases(model, {name: np.zeros(2)})
        assert "has no bias Add" in str(exc_info.value)


def _int8_mean(values: np.ndarray, step: float) -> np.ndarray:
    """Return the per-channel mean of *values* (rows of the last axis) rounded to the INT8 grid of *step*.

    Examples:
        >>> _int8_mean(np.array([[0.2, -0.6], [0.4, 0.0]]), 0.5).tolist()
        [0.25, -0.25]
    """
    rows = values.reshape(-1, values.shape[-1])
    return (np.clip(np.round(rows / step), -128, 127) * step).mean(0)


@_requires_onnxruntime
@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestMlpBiasShifts:
    """The shift subtracted from a bias is the mean error INT8 adds to the layer's output (onnxruntime, CPU)."""

    @pytest.fixture
    def mlp(self, tmp_path: Path) -> tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]:
        """The saved MLP graph, its lifted copy and plan, and four calibration images.

        Examples:
            A pytest fixture cannot be called directly.

            >>> path, model, plan, images = mlp(tmp_path)  # doctest: +SKIP
        """
        model = _gelu_mlp()
        path = tmp_path / "mlp.onnx"
        onnx.save(model, path)
        quantize._lift_opset(model)
        images = list(np.random.default_rng(4).standard_normal((4, 1, 2, 4, 4)).astype(np.float32))
        return path, model, quantize.plan_int8(model), images

    @pytest.mark.parametrize("layer", ["fc1", "fc2"])
    def test_shift_is_the_mean_int8_error_of_the_output(
        self, mlp: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]], layer: str
    ) -> None:
        """A layer's shift is the mean error INT8 adds to its output, per channel."""
        path, model, plan, images = mlp
        weights = {init.name: numpy_helper.to_array(init).astype(np.float64) for init in model.graph.initializer}
        rows = np.concatenate([image.reshape(4, 8) for image in images]).astype(np.float64)
        hidden = rows @ weights["w1"] + weights["b1"]
        inputs = {
            "fc1": rows,
            "fc2": hidden * 0.5 * (1 + torch.special.erf(torch.from_numpy(hidden / np.sqrt(2))).numpy()),
        }
        ranges = {(name, 0): float(np.abs(inputs[name.split("/")[-2]]).max()) for name in (MLP_FC1, MLP_FC2)}
        weight = weights["w1" if layer == "fc1" else "w2"]
        values, scales = quantize._quantized_weight(weight.astype(np.float16), axis=1)
        step = float(quantize._activation_step(ranges[(f"{BACKBONE}mlp/{layer}/MatMul", 0)]))
        expected = (
            _int8_mean(inputs[layer], step) @ (values * scales.astype(np.float64)) - inputs[layer].mean(0) @ weight
        )
        shifts = quantize.measure_mlp_bias_shifts(str(path), model, plan, ranges, images)
        np.testing.assert_allclose(shifts[MLP_ADD1 if layer == "fc1" else MLP_ADD2], expected, rtol=1e-3, atol=1e-6)

    @pytest.mark.parametrize("batch", [1, 2, 3])
    def test_a_static_batch_averages_over_every_image(self, tmp_path: Path, batch: int) -> None:
        """Each real image weighs the same in the means, whatever the static batch the images are grouped into.

        Four images fill two batches of 2 exactly, but at batch 3 the second batch holds one image and two copies of it
        that only pad the batch: counting those copies would weight the last image three times.
        """
        model = _gelu_mlp(batch=batch)
        path = tmp_path / "mlp.onnx"
        onnx.save(model, path)
        quantize._lift_opset(model)
        images = list(np.random.default_rng(4).standard_normal((4, 1, 2, 4, 4)).astype(np.float32))
        weights = {init.name: numpy_helper.to_array(init).astype(np.float64) for init in model.graph.initializer}
        rows = np.concatenate([image.reshape(4, 8) for image in images]).astype(np.float64)
        ranges = {(MLP_FC1, 0): float(np.abs(rows).max()), (MLP_FC2, 0): 1.0}
        values, scales = quantize._quantized_weight(weights["w1"].astype(np.float16), axis=1)
        step = float(quantize._activation_step(ranges[(MLP_FC1, 0)]))
        expected = _int8_mean(rows, step) @ (values * scales.astype(np.float64)) - rows.mean(0) @ weights["w1"]
        shifts = quantize.measure_mlp_bias_shifts(str(path), model, quantize.plan_int8(model), ranges, images)
        np.testing.assert_allclose(shifts[MLP_ADD1], expected, rtol=1e-3, atol=1e-6)

    def test_no_image_is_refused(self, mlp: tuple[Path, Any, quantize.Int8Plan, list[np.ndarray]]) -> None:
        """Measuring shifts with no calibration image raises a ValueError."""
        path, model, plan, _ = mlp
        with pytest.raises(ValueError) as exc_info:
            quantize.measure_mlp_bias_shifts(str(path), model, plan, dict.fromkeys(plan.activations, 1.0), [])
        assert "held no image" in str(exc_info.value)

    def test_graph_without_a_backbone_gelu_mlp_measures_nothing(self, tmp_path: Path) -> None:
        """A graph without a backbone GELU MLP yields no shifts and writes no files."""
        model = _gelu_mlp(activation="Relu")
        plan = quantize.plan_int8(model)
        shifts = quantize.measure_mlp_bias_shifts(str(tmp_path / "relu.onnx"), model, plan, {}, [])
        assert (shifts, os.listdir(tmp_path)) == ({}, [])


def _int8_source(
    root: Path, *, outputs: tuple[str, ...] = ("dets", "labels"), norm: bool = False, seed: int = 7
) -> tuple[Any, Path, np.ndarray]:
    """Build the INT8 source graph of :func:`_gelu_mlp` in *root*; return it, the FP32 graph's path and the images.

    *outputs* and *norm* go to :func:`_gelu_mlp`, whose graph is saved as ``mlp.onnx`` in *root*. Four ``(2, 4, 4)``
    calibration images drawn with *seed* calibrate it at its static batch, and the quantized graph is loaded before
    :func:`quantize.int8_source_graph` removes its file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as root:  # doctest: +IGNORE_OUTPUT
        ...     quantized, path, images = _int8_source(Path(root), outputs=("dets", "labels", "masks"))
        >>> path.name, images.shape, "QuantizeLinear" in {node.op_type for node in quantized.graph.node}
        ('mlp.onnx', (4, 2, 4, 4), True)
    """
    path = root / "mlp.onnx"
    onnx.save(_gelu_mlp(outputs=outputs, norm=norm), path)
    images = np.random.default_rng(seed).standard_normal((4, 2, 4, 4)).astype(np.float32)
    with quantize.int8_source_graph(
        str(path),
        calibration_data=images,
        max_images=10,
        dynamic_batch=False,
        fp16_graph=tensorrt_export.fp16_source_graph,
    ) as int8_path:
        quantized = onnx.load(int8_path)
    return quantized, path, images


def _bias_of(model: Any, add: str) -> np.ndarray:
    """Return the constant that the bias ``Add`` named *add* in *model* adds to its product.

    Examples:
        >>> _bias_of(_gelu_mlp(), MLP_ADD2).shape
        (8,)
    """
    node = next(node for node in model.graph.node if node.name == add)
    constants = quantize._constant_weights(model.graph)
    return next(numpy_helper.to_array(constants[tensor]) for tensor in node.input if tensor in constants)


@_requires_onnxruntime
@_requires_onnxconverter_common
@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestBiasCorrectionInInt8Source:
    """A detector's INT8 graph carries the corrected biases; a segmentation or keypoint model's keeps its own."""

    def test_detector_bias_loses_the_measured_shift(self, tmp_path: Path) -> None:
        """A detector's INT8 graph carries the bias minus the measured shift, in FP16."""
        quantized, path, images = _int8_source(tmp_path, seed=6)
        model = onnx.load(path)
        quantize._lift_opset(model)
        plan = quantize.plan_int8(model)
        tensors = sorted({quantize._input_tensor(model.graph, *key) for key in plan.activations})
        by_tensor = quantize.calibrate_ranges(str(path), model, tensors, list(images[:, None]))
        ranges = {key: by_tensor[quantize._input_tensor(model.graph, *key)] for key in plan.activations}
        shift = quantize.measure_mlp_bias_shifts(str(path), model, plan, ranges, list(images[:, None]))
        original = _bias_of(onnx.load(path), MLP_ADD2).astype(np.float16)
        expected = (original.astype(np.float64) - shift[MLP_ADD2]).astype(np.float16)
        np.testing.assert_array_equal(_bias_of(quantized, MLP_ADD2), expected)

    @pytest.mark.parametrize(
        ("outputs", "smoothed"), [(("dets", "labels"), True), (("dets", "labels", "masks"), False)]
    )
    def test_only_a_detector_norm_reads_a_smoothed_scale(
        self, tmp_path: Path, outputs: tuple[str, ...], smoothed: bool
    ) -> None:
        """Only a detection model's norm is smoothed; segmentation keeps its own scale."""
        quantized, _, _ = _int8_source(tmp_path, outputs=outputs, norm=True)
        norm = next(node for node in quantized.graph.node if node.op_type == "LayerNormalization")
        assert (norm.input[1] != "gamma") == smoothed

    @pytest.mark.parametrize(
        ("outputs", "norm", "verbose"),
        [
            pytest.param(("dets", "labels"), True, [True, False, False], id="detection"),
            pytest.param(("dets", "labels"), False, [True, True, False], id="detection-without-a-norm"),
            pytest.param(("dets", "labels", "masks"), True, [True], id="segmentation"),
            pytest.param(("dets", "labels", "keypoints"), True, [True], id="keypoints"),
        ],
    )
    def test_only_the_first_pass_reports_the_calibration_images(
        self,
        mocker: MockerFixture,
        tmp_path: Path,
        outputs: tuple[str, ...],
        norm: bool,
        verbose: list[bool],
    ) -> None:
        """Only the first pass that reads the calibration images logs their per-sample range warnings.

        A detection export reads the images three times, so each warning about an unnormalized sample would otherwise be
        logged three times; a model whose images are read once keeps its warnings. With no norm to smooth, the smoothing
        pass reads no image, so the calibration pass is the first to read them and keeps the warnings.
        """
        reads = mocker.patch.object(calibration, "calibration_batches", wraps=calibration.calibration_batches)
        _int8_source(tmp_path, outputs=outputs, norm=norm)
        assert [call.kwargs.get("verbose", True) for call in reads.call_args_list] == verbose

    @pytest.mark.parametrize(
        ("outputs", "passes"),
        [(("dets", "labels"), 3), (("dets", "labels", "masks"), 1), (("dets", "labels", "keypoints"), 1)],
    )
    def test_a_detector_reads_the_calibration_images_once_per_step(
        self, mocker: MockerFixture, tmp_path: Path, outputs: tuple[str, ...], passes: int
    ) -> None:
        """A detector reads the calibration images three times, other models once."""
        reads = mocker.patch.object(calibration, "calibration_batches", wraps=calibration.calibration_batches)
        _int8_source(tmp_path, outputs=outputs, norm=True)
        assert reads.call_count == passes

    @pytest.mark.parametrize(("index", "constant"), [(1, "gamma"), (2, "beta")])
    def test_the_fp16_norm_carries_the_scales_the_ranges_were_calibrated_with(
        self, tmp_path: Path, index: int, constant: str
    ) -> None:
        """The FP16 norm's scale and bias are divided by the same smoothing scales used to calibrate."""
        quantized, path, images = _int8_source(tmp_path, norm=True)
        model = onnx.load(path)
        quantize._lift_opset(model)
        name = f"{BACKBONE}norm2/LayerNormalization"
        scales = quantize.measure_smoothing_scales(str(path), model, quantize.plan_int8(model), list(images[:, None]))
        original = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}[constant]
        norm = next(node for node in quantized.graph.node if node.name == name)
        actual = numpy_helper.to_array(quantize._constant_weights(quantized.graph)[norm.input[index]])
        expected = (original.astype(np.float16).astype(np.float64) / scales[name]).astype(np.float16)
        np.testing.assert_array_equal(actual, expected)

    def test_the_fp16_projection_carries_the_scales_the_ranges_were_calibrated_with(self, tmp_path: Path) -> None:
        """The quantized FP16 projection weight carries the smoothing scales used to calibrate."""
        quantized, path, images = _int8_source(tmp_path, norm=True)
        model = onnx.load(path)
        quantize._lift_opset(model)
        name = f"{BACKBONE}norm2/LayerNormalization"
        scales = quantize.measure_smoothing_scales(str(path), model, quantize.plan_int8(model), list(images[:, None]))
        original = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}["w1"]
        fc1 = next(node for node in quantized.graph.node if node.name == f"{BACKBONE}mlp/fc1/MatMul")
        dequantize = next(node for node in quantized.graph.node if fc1.input[1] in node.output)
        smoothed = original.astype(np.float16).astype(np.float64) * scales[name][:, None]
        expected, _ = quantize._quantized_weight(smoothed.astype(np.float16), quantize._weight_axis(fc1))
        actual = numpy_helper.to_array(quantize._constant_weights(quantized.graph)[dequantize.input[0]])
        np.testing.assert_array_equal(actual, expected)

    def test_the_first_projection_is_corrected_on_the_smoothed_graph(self, tmp_path: Path) -> None:
        """The first projection's bias shift is measured on the smoothed graph."""
        quantized, path, images = _int8_source(tmp_path, norm=True)
        corrected = _bias_of(quantized, MLP_ADD1)
        model = onnx.load(path)
        quantize._lift_opset(model)
        plan, batches = quantize.plan_int8(model), list(images[:, None])
        quantize._smooth_norms(model, quantize.measure_smoothing_scales(str(path), model, plan, batches))
        tensors = sorted({quantize._input_tensor(model.graph, *key) for key in plan.activations})
        by_tensor = quantize.calibrate_ranges(str(path), model, tensors, batches)
        ranges = {key: by_tensor[quantize._input_tensor(model.graph, *key)] for key in plan.activations}
        shift = quantize.measure_mlp_bias_shifts(str(path), model, plan, ranges, batches)[MLP_ADD1]
        original = _bias_of(onnx.load(path), MLP_ADD1)
        expected = (original.astype(np.float16).astype(np.float64) - shift).astype(np.float16)
        np.testing.assert_array_equal(corrected, expected)

    def test_the_norm_output_is_calibrated_after_smoothing(self, tmp_path: Path) -> None:
        """The norm output's INT8 scale comes from ranges measured after smoothing."""
        quantized, path, images = _int8_source(tmp_path, norm=True)
        step = next(
            float(numpy_helper.to_array(init)) for init in quantized.graph.initializer if init.name == "rows_int8_scale"
        )
        model = onnx.load(path)
        quantize._lift_opset(model)
        plan = quantize.plan_int8(model)
        batches = list(images[:, None])
        quantize._smooth_norms(model, quantize.measure_smoothing_scales(str(path), model, plan, batches))
        smoothed = quantize.calibrate_ranges(str(path), model, ["rows"], batches)["rows"]
        assert step == float(quantize._activation_step(smoothed))

    @pytest.mark.parametrize("head", ["masks", "keypoints"])
    def test_segmentation_and_keypoint_biases_are_kept(self, tmp_path: Path, head: str) -> None:
        """Segmentation and keypoint models keep their original biases."""
        quantized, path, _ = _int8_source(tmp_path, outputs=("dets", "labels", head), seed=6)
        original = _bias_of(onnx.load(path), MLP_ADD2).astype(np.float16)
        np.testing.assert_array_equal(_bias_of(quantized, MLP_ADD2), original)


class TestNanoInt8Plan:
    """Exact counts on a real RFDETRNano export pin the plan to the architecture it was measured on."""

    @pytest.fixture(scope="class")
    def nano_onnx(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    def plan(self, nano_onnx: Path) -> quantize.Int8Plan:
        """The INT8 plan of the exported Nano graph.

        Examples:
            A pytest fixture cannot be called directly.

            >>> plan = plan(nano_onnx)  # doctest: +SKIP
        """
        return quantize.plan_int8(onnx.load(str(nano_onnx)))

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
    def nano_int8(self, nano_onnx: Path) -> Any:
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
    def exports(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
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
    def plans(self, exports: dict[str, Path]) -> dict[str, quantize.Int8Plan]:
        """The INT8 plan of each exported graph.

        Examples:
            A pytest fixture cannot be called directly.

            >>> plan = plans(exports)["keypoint"]  # doctest: +SKIP
        """
        return {kind: quantize.plan_int8(onnx.load(str(path))) for kind, path in exports.items()}

    @pytest.fixture(scope="class")
    def quantized_graphs(self, exports: dict[str, Path]) -> dict[str, Any]:
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


def _people_photo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Download supervision's PEOPLE_WALKING photo; return a directory of its crops and flips, and the photo itself.

    Examples:
        Needs a network download, so it is documentation rather than a doctest:

        >>> calibration_dir, photo_path = _people_photo(tmp_path_factory)  # doctest: +SKIP
    """
    from PIL import Image, ImageOps
    from supervision.assets import ImageAssets, download_assets

    asset_dir = tmp_path_factory.mktemp("int8_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        photo = Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)
    calibration = tmp_path_factory.mktemp("int8_calibration")
    with Image.open(photo) as image:
        image = image.convert("RGB")
        width, height = image.size
        for index, box in enumerate(
            [(0, 0, width, height), (0, 0, width // 2, height), (width // 2, 0, width, height)]
        ):
            crop = image.crop(box)
            crop.save(calibration / f"{index}.jpg")
            ImageOps.mirror(crop).save(calibration / f"{index}_flipped.jpg")
    return calibration, photo


def _photo_input(photo: Path, resolution: int) -> np.ndarray:
    """Preprocess *photo* to a ``(1, 3, resolution, resolution)`` model input the way ``predict()`` does.

    Examples:
        >>> import tempfile
        >>> from PIL import Image
        >>> with tempfile.TemporaryDirectory() as root:
        ...     Image.new("RGB", (20, 10)).save(Path(root) / "p.jpg")
        ...     _photo_input(Path(root) / "p.jpg", 8).shape
        (1, 3, 8, 8)
    """
    from PIL import Image

    from rfdetr.export._runtime.preprocess import preprocess_to_nchw

    with Image.open(photo) as image:
        return preprocess_to_nchw(image.convert("RGB"), height=resolution, width=resolution)


def _assert_confident_detections_match(
    fp16_engine: Path, int8_engine: Path, example: np.ndarray, batch: int, *, threshold: float = 0.5
) -> None:
    """Run both engines on *batch* copies of *example*; every confident FP16 detection needs an INT8 match.

    A match is an INT8 box with IoU above 0.85 and a score within 0.1 of the FP16 one. An FP16 detection is confident
    above *threshold*, and at least three must be for the check to mean anything. The INT8 engine must also give
    every row of the batch the same boxes for its confident queries.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(np.repeat(example, batch, axis=0)).cuda()}
    results = []
    for path in (fp16_engine, int8_engine):
        outputs = TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs)
        results.append((outputs["dets"].float().cpu(), outputs["labels"].float().sigmoid().max(dim=-1).values.cpu()))
    (fp16_boxes, fp16_scores), (int8_boxes, int8_scores) = results
    assert torch.isfinite(int8_boxes).all() and torch.isfinite(int8_scores).all()
    # Rows are compared on the queries the INT8 engine is confident about: the boxes of near-zero-score queries are
    # unconstrained, and a different INT8 tactic per batch row moves them by more than any useful tolerance.
    confident_int8 = int8_scores[0] > 0.5
    assert torch.allclose(
        int8_boxes[:, confident_int8], int8_boxes[:1, confident_int8].expand(batch, -1, -1), atol=2e-2
    ), "rows of one batch differ"
    for row in range(batch):
        confident = fp16_scores[row] > threshold
        assert int(confident.sum()) >= 3, "the photo must give the FP16 engine several confident people"
        # (confident FP16 boxes, all INT8 boxes); several INT8 queries can cover one person, so a match is the closest
        # score among the INT8 boxes that overlap, not the score of the single best-overlapping box.
        overlaps = box_iou(box_cxcywh_to_xyxy(fp16_boxes[row][confident]), box_cxcywh_to_xyxy(int8_boxes[row]))[0]
        gaps = (int8_scores[row][None, :] - fp16_scores[row][confident][:, None]).abs()
        gaps = gaps.masked_fill(overlaps <= 0.85, float("inf")).min(dim=1).values
        unmatched = [
            (round(float(o), 3), round(float(g), 3)) for o, g in zip(overlaps.max(dim=1).values, gaps) if g >= 0.1
        ]
        assert unmatched == [], (
            f"row {row}: confident FP16 detections without an INT8 match (best IoU, score gap): {unmatched}"
        )


#: Smallest mask IoU between an FP16 and an INT8 mask for a confident detection, and largest keypoint offset (as a
#: fraction of the image side) between them. Measured on RFDETRSegNano and RFDETRKeypointPreview (RTX 5070, TensorRT
#: 11.4): the lowest mask IoU was 0.90 and the largest offset 0.015; the bounds leave room for another GPU.
_MIN_MASK_IOU, _MAX_KEYPOINT_OFFSET = 0.8, 0.05

#: Score above which an FP16 detection of the PEOPLE_WALKING photo counts as confident, per extra output. The keypoint
#: model scores the photo's people low (0.62, 0.50, 0.48, 0.43 on RTX 5070), so 0.5 would let a rebuild flip one.
_CONFIDENCE_THRESHOLD = {"masks": 0.5, "keypoints": 0.4}


def _assert_heads_agree(fp16_engine: Path, int8_engine: Path, example: np.ndarray, head: str) -> None:
    """Run both engines on *example*; every confident FP16 detection needs an INT8 detection with the same *head*.

    A match is the INT8 query whose box has IoU above 0.85 with the FP16 box and whose score is closest within 0.1.
    For ``head="masks"`` the matched masks (logits above zero) need an IoU of at least :data:`_MIN_MASK_IOU`; for
    ``head="keypoints"`` the person keypoints' normalized x and y may differ by at most :data:`_MAX_KEYPOINT_OFFSET`.
    FP16 detections are confident above ``_CONFIDENCE_THRESHOLD[head]``, and at least three must be confident and
    matched for the check to mean anything.

    Examples:
        Needs a GPU, TensorRT and two built engines, so it is documentation rather than a doctest:

        >>> _assert_heads_agree(fp16_path, int8_path, example, head="masks")  # doctest: +SKIP
    """
    from rfdetr.export._tensorrt.inference import TRTInference

    inputs = {"input": torch.from_numpy(example).cuda()}
    fp16, int8 = (
        {
            name: value.float().cpu()
            for name, value in TRTInference(str(path), sync_mode=True, device="cuda:0")(inputs).items()
        }
        for path in (fp16_engine, int8_engine)
    )
    for outputs in (fp16, int8):
        assert torch.isfinite(outputs[head]).all(), f"{head} hold NaN or infinite values"
    fp16_scores = fp16["labels"][0].sigmoid().max(dim=-1).values
    int8_scores = int8["labels"][0].sigmoid().max(dim=-1).values
    confident = torch.nonzero(fp16_scores > _CONFIDENCE_THRESHOLD[head]).flatten()
    overlaps = box_iou(box_cxcywh_to_xyxy(fp16["dets"][0][confident]), box_cxcywh_to_xyxy(int8["dets"][0]))[0]
    gaps = (int8_scores[None, :] - fp16_scores[confident][:, None]).abs().masked_fill(overlaps <= 0.85, float("inf"))
    matched = [
        (int(q), int(m)) for q, m, gap in zip(confident, gaps.argmin(dim=1), gaps.min(dim=1).values) if gap < 0.1
    ]
    assert len(matched) >= 3, f"only {len(matched)} of {len(confident)} confident FP16 detections have an INT8 match"
    for query, other in matched:
        if head == "masks":
            fp16_mask, int8_mask = fp16["masks"][0, query] > 0, int8["masks"][0, other] > 0
            agreement = float((fp16_mask & int8_mask).sum() / (fp16_mask | int8_mask).sum().clamp(min=1))
            assert agreement >= _MIN_MASK_IOU, f"query {query}: mask IoU {agreement:.3f}"
        else:
            offset = float(
                (fp16["keypoints"][0, query, 17:34, :2] - int8["keypoints"][0, other, 17:34, :2]).abs().max()
            )
            assert offset <= _MAX_KEYPOINT_OFFSET, f"query {query}: keypoint offset {offset:.3f}"


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained RFDETRNano agree on a photo's confident detections."""

    @pytest.fixture(scope="class")
    def photo(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray]:
        """Crops and flips of supervision's PEOPLE_WALKING photo for calibration, and the photo as a model input.

        Examples:
            A pytest fixture, and it needs a network download.

            >>> calibration_dir, example = photo(tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, image = _people_photo(tmp_path_factory)
        return calibration, _photo_input(image, int(RFDETRNano().model.resolution))

    @pytest.fixture(scope="class")
    def engines(
        self, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray]:
        """Export FP16 and INT8 engines at batch 1, calibrating on the photo's crops and flips.

        Examples:
            A pytest fixture, and it needs a GPU and TensorRT.

            >>> fp16_engine, int8_engine, example = engines(photo, tmp_path_factory)  # doctest: +SKIP
        """
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp("int8_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        return Path(fp16), Path(int8), example

    def test_engine_is_written_without_leftovers(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """The INT8 export leaves only the ONNX, the engine and its description."""
        _, int8, _ = engines
        assert int8.name == "rfdetr-nano_int8.trt"
        assert sorted(p.name for p in int8.parent.iterdir()) == [
            "rfdetr-nano.onnx",
            "rfdetr-nano_int8.trt",
            "rfdetr-nano_int8.trt.rfdetr.json",
        ]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16_path, int8_path, example = engines
        _assert_confident_detections_match(fp16_path, int8_path, example, batch=1)

    @pytest.mark.parametrize("source", ["npy-path", "array"])
    def test_preprocessed_calibration_gives_a_matching_engine(
        self,
        source: str,
        photo: tuple[Path, np.ndarray],
        engines: tuple[Path, Path, np.ndarray],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Calibrating on a preprocessed .npy path or array gives an engine matching FP16."""
        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.calibration import calibration_batches

        calibration, example = photo
        fp16_path, _, _ = engines
        size = example.shape[-1]
        samples = np.concatenate(list(calibration_batches(calibration, height=size, width=size)))
        data: Any = samples
        if source == "npy-path":
            data = tmp_path_factory.mktemp("int8_samples") / "calibration.npy"
            np.save(data, samples)
        int8 = RFDETRNano().export(
            output_dir=str(tmp_path_factory.mktemp(f"int8_{source}")),
            format="tensorrt",
            quantization="int8",
            calibration_data=data,
            verbose=False,
        )
        assert Path(int8).name == "rfdetr-nano_int8.trt"
        _assert_confident_detections_match(fp16_path, Path(int8), example, batch=1)

    @pytest.mark.parametrize("batch", [2, 4])
    def test_static_batch_engine_matches_fp16_on_every_row(
        self, batch: int, photo: tuple[Path, np.ndarray], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A static-batch INT8 engine matches FP16 on every row of the batch."""
        from rfdetr import RFDETRNano

        calibration, example = photo
        model = RFDETRNano()
        out_dir = tmp_path_factory.mktemp(f"int8_batch_{batch}")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", batch_size=batch, verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            batch_size=batch,
            verbose=False,
        )
        _assert_confident_detections_match(Path(fp16), Path(int8), example, batch=batch)


@_requires_tensorrt
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_tensorrt
class TestSegmentationAndKeypointInt8EndToEnd:
    """Real FP16 and INT8 engines of the pretrained segmentation and keypoint models agree on a photo."""

    @pytest.fixture(scope="class", params=["RFDETRSegNano", "RFDETRKeypointPreview"])
    def engines(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[Path, Path, np.ndarray, str]:
        """Export FP16 and INT8 engines of the model at batch 1, calibrating on the photo's crops and flips.

        Returns the two engine paths, the photo as that model's input, and the name of its extra output.

        Examples:
            A pytest fixture, and it needs a GPU, TensorRT and a network download.

            >>> fp16_engine, int8_engine, example, head = engines(request, tmp_path_factory)  # doctest: +SKIP
        """
        import rfdetr

        calibration, photo = _people_photo(tmp_path_factory)
        model = getattr(rfdetr, request.param)()
        out_dir = tmp_path_factory.mktemp("int8_head_engines")
        fp16 = model.export(output_dir=str(out_dir / "fp16"), format="tensorrt", verbose=False)
        int8 = model.export(
            output_dir=str(out_dir / "int8"),
            format="tensorrt",
            quantization="int8",
            calibration_data=str(calibration),
            verbose=False,
        )
        head = "masks" if request.param == "RFDETRSegNano" else "keypoints"
        return Path(fp16), Path(int8), _photo_input(photo, int(model.model.resolution)), head

    def test_engine_keeps_the_extra_output(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 engine keeps the mask or keypoint output after dets and labels."""
        from rfdetr.export._tensorrt.inference import TRTInference

        _, int8, _, head = engines
        assert TRTInference(str(int8), sync_mode=True, device="cuda:0").get_output_names() == ["dets", "labels", head]

    def test_confident_detections_match_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """Every confident FP16 detection has a matching INT8 detection."""
        fp16, int8, example, head = engines
        _assert_confident_detections_match(fp16, int8, example, batch=1, threshold=_CONFIDENCE_THRESHOLD[head])

    def test_head_matches_fp16(self, engines: tuple[Path, Path, np.ndarray, str]) -> None:
        """The INT8 mask or keypoint head agrees with FP16 on confident detections."""
        fp16, int8, example, head = engines
        _assert_heads_agree(fp16, int8, example, head)
