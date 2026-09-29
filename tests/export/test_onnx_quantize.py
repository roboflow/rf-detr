# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for static INT8 quantization of ONNX exports (:mod:`rfdetr.export._onnx.quantize`)."""

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx not installed; skip ONNX quantization tests")

from onnx import TensorProto, helper  # noqa: E402

from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter  # noqa: E402
from rfdetr.export._onnx.quantize import VALID_QUANTIZATIONS, nodes_to_exclude  # noqa: E402
from rfdetr.export._runtime.calibration import calibration_batches  # noqa: E402


def _attention_graph() -> object:
    """Build a graph shaped like one attention block followed by a head.

    ``projection`` sits deeper than the head sweep reaches and feeds no ``Softmax``, so it stands for the bulk of the
    network: the layers that are supposed to end up in 8-bit.

    Returns:
        An ``onnx.GraphProto`` with a deep ``MatMul``, a score ``MatMul`` feeding a ``Softmax``, a value ``MatMul``,
        and a head ``Gemm`` producing the graph output.

    Examples:
        >>> graph = _attention_graph()
        >>> sorted(node.name for node in graph.node)
        ['head', 'projection', 'scores', 'softmax', 'values']
    """
    nodes = [
        helper.make_node("MatMul", ["x", "w_in"], ["query"], name="projection"),
        helper.make_node("MatMul", ["query", "key"], ["score"], name="scores"),
        helper.make_node("Softmax", ["score"], ["weights"], name="softmax"),
        helper.make_node("MatMul", ["weights", "value"], ["context"], name="values"),
        helper.make_node("Gemm", ["context", "proj"], ["logits"], name="head"),
    ]
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4, 4])]
    return helper.make_graph(nodes, "attention", inputs, outputs)


class TestNodeSelection:
    """Which nodes are held back from quantization."""

    def test_excludes_matmul_feeding_softmax(self) -> None:
        assert "scores" in nodes_to_exclude(_attention_graph())

    def test_excludes_head_producing_graph_output(self) -> None:
        assert "head" in nodes_to_exclude(_attention_graph())

    def test_keeps_deep_projection_quantizable(self) -> None:
        # The point of the exclusion list is to be narrow: a layer that neither feeds a Softmax nor sits near an
        # output must still be quantized, or the mode buys nothing.
        assert "projection" not in nodes_to_exclude(_attention_graph())

    def test_excludes_nodes_within_head_depth_of_an_output(self) -> None:
        # "values" is two hops from the graph output, so the head sweep reaches it.
        assert "values" in nodes_to_exclude(_attention_graph())

    def test_returns_sorted_unique_names(self) -> None:
        excluded = nodes_to_exclude(_attention_graph())
        assert excluded == sorted(set(excluded))


class TestCalibrationBatches:
    """Turning user-supplied calibration data into model inputs."""

    def test_array_yields_one_batch_per_sample(self) -> None:
        batches = list(calibration_batches(np.zeros((3, 3, 8, 8), dtype=np.float32), height=8, width=8))
        assert len(batches) == 3

    def test_array_batches_carry_a_leading_batch_dimension(self) -> None:
        batches = list(calibration_batches(np.zeros((2, 3, 8, 8), dtype=np.float32), height=8, width=8))
        assert batches[0].shape == (1, 3, 8, 8)

    def test_rejects_array_with_wrong_rank(self) -> None:
        with pytest.raises(ValueError, match="rank 4"):
            list(calibration_batches(np.zeros((3, 8, 8), dtype=np.float32), height=8, width=8))

    def test_rejects_array_with_mismatched_resolution(self) -> None:
        with pytest.raises(ValueError, match="expects 16x16"):
            list(calibration_batches(np.zeros((1, 3, 8, 8), dtype=np.float32), height=16, width=16))

    def test_rejects_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            list(calibration_batches(tmp_path / "absent", height=8, width=8))

    def test_rejects_non_npy_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "data.txt"
        text_file.write_text("not an array")
        with pytest.raises(ValueError, match=r"\.npy"):
            list(calibration_batches(text_file, height=8, width=8))

    def test_rejects_directory_without_images(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No calibration images"):
            list(calibration_batches(tmp_path, height=8, width=8))

    def test_reads_npy_file(self, tmp_path: Path) -> None:
        array_path = tmp_path / "calib.npy"
        np.save(array_path, np.zeros((2, 3, 8, 8), dtype=np.float32))
        assert len(list(calibration_batches(array_path, height=8, width=8))) == 2

    def test_preprocesses_images_from_a_directory(self, tmp_path: Path) -> None:
        pil = pytest.importorskip("PIL.Image", reason="Pillow not installed")
        for name in ("a.jpg", "b.jpg"):
            pil.new("RGB", (32, 24)).save(tmp_path / name)
        batches = list(calibration_batches(tmp_path, height=8, width=8))
        assert [batch.shape for batch in batches] == [(1, 3, 8, 8), (1, 3, 8, 8)]

    def test_directory_reading_honours_max_images(self, tmp_path: Path) -> None:
        pil = pytest.importorskip("PIL.Image", reason="Pillow not installed")
        for index in range(4):
            pil.new("RGB", (32, 24)).save(tmp_path / f"{index}.jpg")
        assert len(list(calibration_batches(tmp_path, height=8, width=8, max_images=2))) == 2


class TestConfigValidation:
    """Refusals that happen before any work on the model."""

    def test_int8_is_a_valid_mode(self) -> None:
        assert "int8" in VALID_QUANTIZATIONS

    def test_rejects_unknown_quantization_mode(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="fp16"))

    def test_rejects_int8_without_calibration_data(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires calibration_data"):
            OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="int8"))

    def test_accepts_int8_with_calibration_data(self, tmp_path: Path) -> None:
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization="int8", calibration_data=tmp_path))
        assert exporter.config.quantization == "int8"

    @pytest.mark.parametrize("quantization", [None, "fp32"])
    def test_accepts_float_modes_without_calibration_data(self, quantization: str | None, tmp_path: Path) -> None:
        exporter = OnnxExporter(OnnxConfig(output_dir=tmp_path, quantization=quantization))
        assert exporter.config.quantization == quantization


class TestSettingPlumbing:
    """The keywords ``RFDETR.export`` forwards to this format."""

    def test_quantization_settings_reach_the_config(self) -> None:
        config = OnnxExporter.build_config(quantization="int8", calibration_data="images/", max_images=32)
        assert (config.quantization, config.calibration_data, config.max_images) == ("int8", "images/", 32)

    def test_two_stage_formats_do_not_inherit_quantization(self) -> None:
        # TFLite and TensorRT derive their intermediate ONNX config; their own `quantization` means something else
        # and must not leak into this format's static path.
        from rfdetr.export.base import ExportConfig

        assert OnnxConfig.derive(ExportConfig(), opset_version=17).quantization is None
