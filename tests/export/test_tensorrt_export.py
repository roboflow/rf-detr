# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for TensorRT export helpers."""

import argparse
import subprocess

import pytest

from rfdetr.export import _tensorrt as tensorrt_export


def test_run_command_shell_dry_run_handles_missing_cuda_visible_devices(monkeypatch) -> None:
    """Dry-run logging should not crash when CUDA_VISIBLE_DEVICES is unset."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    logged_messages = []
    monkeypatch.setattr(tensorrt_export.logger, "info", logged_messages.append)

    result = tensorrt_export.run_command_shell(["trtexec", "--help"], dry_run=True)

    assert result.returncode == 0
    assert any("CUDA_VISIBLE_DEVICES=" in message for message in logged_messages)


def test_run_command_shell_uses_list_not_string(monkeypatch) -> None:
    """subprocess.run must be called with a list (shell=False) to prevent injection."""
    captured = {}

    def _fake_run(command, shell, capture_output, text, check):
        captured["command"] = command
        captured["shell"] = shell
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)

    tensorrt_export.run_command_shell(["trtexec", "--onnx=/some/model.onnx"], dry_run=False)

    assert isinstance(captured["command"], list), "command must be a list, not a string"
    assert captured["shell"] is False, "shell=False is required to prevent injection"


def test_run_command_shell_dry_run_does_not_invoke_subprocess(monkeypatch) -> None:
    """Dry-run must return early without calling subprocess.run."""
    was_called = []

    def _should_not_run(*args, **kwargs):
        was_called.append(True)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _should_not_run)
    monkeypatch.setattr(tensorrt_export.logger, "info", lambda _: None)

    result = tensorrt_export.run_command_shell(["trtexec", "--help"], dry_run=True)

    assert not was_called, "subprocess.run must not be called during dry_run"
    assert result.returncode == 0


def test_trtexec_returns_engine_path(monkeypatch) -> None:
    """Trtexec() must return the .engine path, not None."""
    captured_argv = []

    def _fake_run(command, **kwargs):
        captured_argv.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)
    monkeypatch.setattr(tensorrt_export, "parse_trtexec_output", lambda _: {})

    args = argparse.Namespace(profile=False, verbose=False, dry_run=False)
    result = tensorrt_export.trtexec("/tmp/model.onnx", args)

    assert result == "/tmp/model.engine"


def test_trtexec_dry_run_returns_engine_path(monkeypatch) -> None:
    """Trtexec() with dry_run=True must still return the engine path."""
    monkeypatch.setattr(tensorrt_export.logger, "info", lambda _: None)
    monkeypatch.setattr(tensorrt_export, "parse_trtexec_output", lambda _: {})

    args = argparse.Namespace(profile=False, verbose=False, dry_run=True)
    result = tensorrt_export.trtexec("/tmp/model.onnx", args)

    assert result == "/tmp/model.engine"


@pytest.mark.parametrize(
    ("onnx_path", "expected_engine"),
    [
        pytest.param("/output/rfdetr.onnx", "/output/rfdetr.engine", id="plain-path"),
        pytest.param("/path with spaces/model.onnx", "/path with spaces/model.engine", id="path-with-spaces"),
    ],
)
def test_trtexec_argv_contains_no_shell_string(monkeypatch, onnx_path: str, expected_engine: str) -> None:
    """Trtexec builds an argv list; no shell string concatenation of user paths."""
    captured = {}

    def _fake_run(command, shell, **kwargs):
        captured["command"] = command
        captured["shell"] = shell
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)
    monkeypatch.setattr(tensorrt_export, "parse_trtexec_output", lambda _: {})

    args = argparse.Namespace(profile=False, verbose=False, dry_run=False)
    result = tensorrt_export.trtexec(onnx_path, args)

    assert result == expected_engine
    assert isinstance(captured["command"], list), "argv must be a list"
    assert captured["shell"] is False, "shell=False required"
    # Verify the ONNX path appears as a standalone argument element
    assert any(onnx_path in arg for arg in captured["command"])
