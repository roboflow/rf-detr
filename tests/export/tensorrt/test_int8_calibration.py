# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for INT8 calibration and the accuracy passes: range measurement, norm smoothing and MLP bias correction.

Calibration needs onnxruntime and runs in the ``onnx`` integration job.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.export._runtime import calibration
from rfdetr.export._tensorrt import exporter as tensorrt_export
from rfdetr.export._tensorrt import quantize
from tests._markers import _requires_onnxconverter_common, _requires_onnxruntime
from tests.export.tensorrt.helpers import BACKBONE, DECODER, _attention_and_mlp

__doctest_requires__ = {
    ("_int8_source",): ["onnxruntime", "onnxconverter_common"],
}

# Module-level on purpose: parametrized constants and graph builders touch ONNX while the module is collected, so a
# guarded import would raise ``NameError`` instead of skipping.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


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
