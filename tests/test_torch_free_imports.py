# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The package and its NumPy ONNX helpers import and run without ``torch`` and ``torchvision``."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HIDE_TORCH = "import sys; sys.modules.update(torch=None, torchvision=None)\n"
_PREDICT = """
import numpy as np
from types import SimpleNamespace
from PIL import Image
from rfdetr.export._onnx.reference import _run_inference

class Session:
    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 3, 32, 32])]
    def get_outputs(self):
        return [SimpleNamespace(name="dets"), SimpleNamespace(name="labels")]
    def run(self, output_names, feeds):
        assert feeds["input"].shape == (1, 3, 32, 32)
        logits = np.full((1, 10, 82), -10.0, dtype=np.float32)
        logits[0, 3, 5] = 10.0
        return [np.full((1, 10, 4), 0.5, dtype=np.float32), logits]

Image.new("RGB", (64, 48), color=(100, 150, 200)).save("image.jpg")
detections, image = _run_inference(Session(), "image.jpg")
assert detections.class_id.tolist() == [5], detections.class_id
assert detections.xyxy.tolist() == [[16.0, 12.0, 48.0, 36.0]], detections.xyxy
assert image.size == (64, 48)
assert "torch" not in {name for name, module in sys.modules.items() if module is not None}
"""


def _run_without_torch(code: str, cwd: Path) -> "subprocess.CompletedProcess[str]":
    """Run ``code`` in a new interpreter that cannot import ``torch`` or ``torchvision``.

    Examples:
        >>> import tempfile
        >>> _run_without_torch("import torch", Path(tempfile.gettempdir())).returncode
        1
    """
    return subprocess.run(
        [sys.executable, "-c", _HIDE_TORCH + code],
        capture_output=True,
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(_REPO / "src")},
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize(
    "code",
    [
        "import rfdetr",
        "from rfdetr.export._onnx.reference import _create_onnx_session, _run_inference",
        "from rfdetr.export._runtime.decode import decode_detections",
        "from rfdetr.export._runtime.metadata import ExportMetadata",
        pytest.param(_PREDICT, id="predict"),
    ],
)
def test_runs_without_torch(code: str, tmp_path: Path) -> None:
    """The torch-free part of the package imports, and the ONNX reference helper decodes detections."""
    result = _run_without_torch(code, tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["RFDETRNano", "RFDETRInference", "ClassThresholds", "from_checkpoint"])
def test_torch_backed_export_names_the_missing_package(name: str, tmp_path: Path) -> None:
    """A model export asked for without torch raises an ImportError that says what to install."""
    result = _run_without_torch(f"from rfdetr import {name}", tmp_path)
    assert result.returncode != 0
    assert "requires PyTorch" in result.stderr, result.stderr
