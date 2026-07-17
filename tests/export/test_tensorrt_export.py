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
        pytest.param("/model;rm -rf /.onnx", "/model;rm -rf /.engine", id="shell-metachar"),
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
    # Verify the ONNX path appears as a standalone argument element (not shell-expanded)
    assert any(onnx_path in arg for arg in captured["command"])


# ---------------------------------------------------------------------------
# parse_trtexec_output (#1)
# ---------------------------------------------------------------------------

_FULL_TRTEXEC_STDOUT = """\
[I] GPU Compute Time: min = 1.23 ms, max = 4.56 ms, mean = 2.34 ms, median = 2.10 ms
[I] Host to Device Transfer Time: min = 0.10 ms, max = 0.20 ms, mean = 0.15 ms
[I] Device to Host Transfer Time: min = 0.05 ms, max = 0.08 ms, mean = 0.06 ms
[I] Latency: min = 1.40 ms, max = 4.80 ms, mean = 2.55 ms
[I] Throughput: 391.22 qps
"""

_PARTIAL_TRTEXEC_STDOUT = """\
[I] GPU Compute Time: min = 1.00 ms, max = 2.00 ms, mean = 1.50 ms, median = 1.45 ms
[I] Throughput: 100.00 qps
"""


@pytest.mark.parametrize(
    ("output_text", "expected"),
    [
        pytest.param(
            _FULL_TRTEXEC_STDOUT,
            {
                "compute_min_ms": 1.23,
                "compute_max_ms": 4.56,
                "compute_mean_ms": 2.34,
                "compute_median_ms": 2.10,
                "h2d_min_ms": 0.10,
                "h2d_max_ms": 0.20,
                "h2d_mean_ms": 0.15,
                "d2h_min_ms": 0.05,
                "d2h_max_ms": 0.08,
                "d2h_mean_ms": 0.06,
                "latency_min_ms": 1.40,
                "latency_max_ms": 4.80,
                "latency_mean_ms": 2.55,
                "throughput_qps": 391.22,
            },
            id="all-5-patterns",
        ),
        pytest.param(
            "",
            {},
            id="empty-stdout",
        ),
        pytest.param(
            _PARTIAL_TRTEXEC_STDOUT,
            {
                "compute_min_ms": 1.00,
                "compute_max_ms": 2.00,
                "compute_mean_ms": 1.50,
                "compute_median_ms": 1.45,
                "throughput_qps": 100.00,
            },
            id="partial-stdout",
        ),
    ],
)
def test_parse_trtexec_output(output_text: str, expected: dict) -> None:
    """parse_trtexec_output extracts timing statistics from trtexec stdout."""
    result = tensorrt_export.parse_trtexec_output(output_text)
    assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# CalledProcessError logging path (#15)
# ---------------------------------------------------------------------------


def test_run_command_shell_called_process_error_is_reraised(monkeypatch) -> None:
    """CalledProcessError from subprocess.run is re-raised after logging."""
    error_messages = []

    def _fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["trtexec"], stderr="engine build failed")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)
    monkeypatch.setattr(tensorrt_export.logger, "error", error_messages.append)

    with pytest.raises(subprocess.CalledProcessError):
        tensorrt_export.run_command_shell(["trtexec", "--onnx=/tmp/model.onnx"], dry_run=False)

    assert error_messages, "logger.error must be called when CalledProcessError is raised"


# ---------------------------------------------------------------------------
# profile=True argv path (#17)
# ---------------------------------------------------------------------------


def test_trtexec_profile_true_wraps_with_nsys(monkeypatch) -> None:
    """Profile=True wraps trtexec with 'nsys profile …' and the output flag is present."""
    captured_argv: list[str] = []

    def _fake_run(command, **kwargs):
        captured_argv.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(tensorrt_export.subprocess, "run", _fake_run)
    monkeypatch.setattr(tensorrt_export, "parse_trtexec_output", lambda _: {})
    monkeypatch.setattr(tensorrt_export.logger, "info", lambda _: None)

    args = argparse.Namespace(profile=True, verbose=False, dry_run=False)
    tensorrt_export.trtexec("/tmp/model.onnx", args)

    assert captured_argv[0] == "nsys", "profile=True must wrap with nsys as argv[0]"
    argv_str = " ".join(captured_argv)
    assert "--output=" in argv_str, "nsys profile must include --output= flag"
