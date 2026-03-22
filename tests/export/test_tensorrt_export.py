# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for TensorRT export helpers."""

import subprocess
from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.export import tensorrt as tensorrt_export


def test_run_command_shell_dry_run_handles_missing_cuda_visible_devices(monkeypatch) -> None:
    """Dry-run logging should not crash when CUDA_VISIBLE_DEVICES is unset."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def _fake_run(command, shell, capture_output, text, check):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)
    logged_messages = []
    monkeypatch.setattr(tensorrt_export.logger, "info", logged_messages.append)

    result = tensorrt_export.run_command_shell("trtexec --help", dry_run=True)

    assert result.returncode == 0
    assert any("CUDA_VISIBLE_DEVICES=" in message for message in logged_messages)


def test_trtexec_returns_engine_path(monkeypatch) -> None:
    """trtexec should return the .engine file path derived from the .onnx path."""
    fake_result = subprocess.CompletedProcess("cmd", 0, stdout="", stderr="")

    monkeypatch.setattr(
        tensorrt_export,
        "run_command_shell",
        lambda command, dry_run: fake_result,
    )
    monkeypatch.setattr(tensorrt_export, "parse_trtexec_output", lambda text: {})

    from argparse import Namespace

    args = Namespace(verbose=False, profile=False, dry_run=False)
    result = tensorrt_export.trtexec("output/inference_model.onnx", args)

    assert result == "output/inference_model.engine"


class TestRFDETRTensorRT:
    """Tests for RFDETRTensorRT inference wrapper."""

    def _make_detector(self):
        """Return an RFDETRTensorRT with mocked TRTInference and PostProcess."""
        from rfdetr.export.tensorrt import RFDETRTensorRT

        mock_trt = MagicMock()
        mock_postprocess = MagicMock()

        # Bypass __init__ entirely to avoid requiring TensorRT/pycuda in CI.
        detector = RFDETRTensorRT.__new__(RFDETRTensorRT)
        detector.resolution = 560
        detector.means = [0.485, 0.456, 0.406]
        detector.stds = [0.229, 0.224, 0.225]
        detector.device = torch.device("cpu")
        detector.class_names = None
        detector._trt = mock_trt
        detector._postprocess = mock_postprocess
        return detector, mock_trt, mock_postprocess

    def test_predict_raises_on_wrong_channel_count(self) -> None:
        """predict() must raise ValueError for non-RGB tensors."""
        detector, _, _ = self._make_detector()

        img = torch.zeros(1, 64, 64)  # 1 channel instead of 3
        with pytest.raises(ValueError, match="3 channels"):
            detector.predict(img)

    def test_predict_raises_on_unnormalized_image(self) -> None:
        """predict() must raise ValueError when pixel values exceed 1."""
        detector, _, _ = self._make_detector()

        img = torch.ones(3, 64, 64) * 255.0  # not in [0, 1]
        with pytest.raises(ValueError, match="normalized"):
            detector.predict(img)

    def test_predict_remaps_trt_outputs_to_postprocess_keys(self) -> None:
        """predict() must remap TRT output names to PostProcess expected keys."""

        detector, mock_trt, mock_postprocess = self._make_detector()

        num_queries = 10
        num_classes = 80
        dets = torch.rand(1, num_queries, 4)
        labels = torch.rand(1, num_queries, num_classes)

        mock_trt.return_value = {"dets": dets, "labels": labels}

        fake_scores = torch.tensor([0.9, 0.1])
        fake_labels = torch.tensor([0, 1])
        fake_boxes = torch.rand(2, 4) * 100
        mock_postprocess.return_value = [{"scores": fake_scores, "labels": fake_labels, "boxes": fake_boxes}]

        img = torch.rand(3, 64, 64)
        detector.predict(img, threshold=0.5)

        call_args = mock_postprocess.call_args
        outputs_passed = call_args[0][0]
        assert "pred_logits" in outputs_passed
        assert "pred_boxes" in outputs_passed
        assert torch.equal(outputs_passed["pred_logits"], labels)
        assert torch.equal(outputs_passed["pred_boxes"], dets)

    def test_predict_single_image_returns_single_detections(self) -> None:
        """predict() should return a single Detections object for a single image."""
        import supervision as sv

        detector, mock_trt, mock_postprocess = self._make_detector()

        mock_trt.return_value = {
            "dets": torch.rand(1, 10, 4),
            "labels": torch.rand(1, 10, 80),
        }
        mock_postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.rand(1, 4) * 100,
            }
        ]

        img = torch.rand(3, 64, 64)
        result = detector.predict(img, threshold=0.5)

        assert isinstance(result, sv.Detections)

    def test_predict_batch_returns_list(self) -> None:
        """predict() should return a list of Detections for a batch of images."""
        import supervision as sv

        detector, mock_trt, mock_postprocess = self._make_detector()

        mock_trt.return_value = {
            "dets": torch.rand(2, 10, 4),
            "labels": torch.rand(2, 10, 80),
        }
        mock_postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.rand(1, 4) * 100,
            },
            {
                "scores": torch.tensor([0.8]),
                "labels": torch.tensor([1]),
                "boxes": torch.rand(1, 4) * 100,
            },
        ]

        imgs = [torch.rand(3, 64, 64), torch.rand(3, 64, 64)]
        result = detector.predict(imgs, threshold=0.5)

        assert isinstance(result, list)
        assert len(result) == 2
        for r in result:
            assert isinstance(r, sv.Detections)

    def test_predict_threshold_filters_detections(self) -> None:
        """Detections below the threshold must be filtered out."""

        detector, mock_trt, mock_postprocess = self._make_detector()

        mock_trt.return_value = {
            "dets": torch.rand(1, 10, 4),
            "labels": torch.rand(1, 10, 80),
        }
        scores = torch.tensor([0.9, 0.3, 0.8])
        mock_postprocess.return_value = [
            {
                "scores": scores,
                "labels": torch.tensor([0, 1, 2]),
                "boxes": torch.rand(3, 4) * 100,
            }
        ]

        img = torch.rand(3, 64, 64)
        result = detector.predict(img, threshold=0.5)

        assert len(result) == 2  # only scores 0.9 and 0.8 pass

    def test_missing_trt_raises_import_error(self, monkeypatch) -> None:
        """RFDETRTensorRT() must raise ImportError when TRTInference is unavailable."""
        monkeypatch.setitem(
            __import__("sys").modules,
            "rfdetr.export.benchmark",
            None,
        )
        from rfdetr.export.tensorrt import RFDETRTensorRT

        with pytest.raises(ImportError, match="rfdetr\\[trt\\]") as exc_info:
            RFDETRTensorRT("dummy.engine")

        assert exc_info.value.__cause__ is not None
