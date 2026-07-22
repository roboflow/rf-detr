# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the in-process TensorRT engine builder (`build_engine`).

The engine is built via the TensorRT Python API through `polygraphy`; these tests monkeypatch the polygraphy entry
points so they run without TensorRT, a GPU, or `polygraphy` installed.
"""

import pytest

from rfdetr.export import _tensorrt as tensorrt_export


@pytest.mark.parametrize(
    ("onnx_path", "expected_engine"),
    [
        pytest.param("/output/rfdetr.onnx", "/output/rfdetr.trt", id="plain-path"),
        pytest.param("/path with spaces/model.onnx", "/path with spaces/model.trt", id="path-with-spaces"),
        pytest.param("/model;rm -rf /.onnx", "/model;rm -rf /.onnx.trt", id="shell-metachar"),
        pytest.param("/data/my.onnx.backup/model.onnx", "/data/my.onnx.backup/model.trt", id="earlier-onnx-in-dir"),
        pytest.param("/output/model_v1.onnx.old.onnx", "/output/model_v1.onnx.old.trt", id="double-onnx-in-filename"),
        pytest.param("/output/model_without_extension", "/output/model_without_extension.trt", id="no-onnx-extension"),
    ],
)
def test_build_engine_dry_run_derives_trt_path(onnx_path: str, expected_engine: str) -> None:
    """Only the final suffix is swapped to ``.trt``; earlier ``.onnx`` segments are never corrupted."""
    result = tensorrt_export.build_engine(onnx_path, dry_run=True)

    assert result == expected_engine


def test_build_engine_dry_run_does_not_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run must return the engine path without invoking the polygraphy build chain."""
    called: list[str] = []
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *a, **k: called.append("built"))

    result = tensorrt_export.build_engine("/tmp/model.onnx", dry_run=True)

    assert result == "/tmp/model.trt"
    assert not called, "dry_run must not invoke the polygraphy build chain"


def test_build_engine_without_polygraphy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing polygraphy/tensorrt install must raise an actionable ImportError."""
    monkeypatch.setattr(tensorrt_export, "engine_from_network", None)

    with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
        tensorrt_export.build_engine("/tmp/model.onnx")


@pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
def test_build_engine_invokes_polygraphy_and_saves_trt(monkeypatch: pytest.MonkeyPatch, fp16: bool) -> None:
    """build_engine wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""
    config_kwargs: dict = {}
    build_args: dict = {}
    saved: dict = {}

    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
    monkeypatch.setattr(
        tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config-sentinel"
    )

    def _engine_from_network(network, config):
        build_args["network"] = network
        build_args["config"] = config
        return "engine-sentinel"

    def _save_engine(engine, path):
        saved["engine"] = engine
        saved["path"] = path

    monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
    monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)

    result = tensorrt_export.build_engine("/tmp/model.onnx", fp16=fp16)

    assert result == "/tmp/model.trt"
    assert config_kwargs == {"fp16": fp16}
    assert build_args == {"network": ("network", "/tmp/model.onnx"), "config": "config-sentinel"}
    assert saved == {"engine": "engine-sentinel", "path": "/tmp/model.trt"}
