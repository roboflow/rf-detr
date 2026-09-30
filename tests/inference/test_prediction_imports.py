# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Package import compatibility without optional capture dependencies."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_package_import_without_capture_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Core model imports must work without video, screenshot, or YouTube packages."""
    other_package = tmp_path / "rfdetr"
    other_package.mkdir()
    (other_package / "__init__.py").write_text('raise AssertionError("Wrong checkout imported")')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules.update(cv2=None, mss=None, yt_dlp=None); "
            "from rfdetr import RFDETRNano, PredictionInput; assert RFDETRNano is not None",
        ],
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
