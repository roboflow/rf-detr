# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for TensorRT export helpers."""

import subprocess

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
