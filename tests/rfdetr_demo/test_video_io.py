# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for video I/O helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rfdetr_demo.inference.video_io import probe_video_size


def test_probe_video_size_returns_dimensions(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {3: 640.0, 4: 480.0, 5: 30.0}.get(prop, 0.0)

    with patch("rfdetr_demo.inference.video_io.cv2.VideoCapture", return_value=mock_cap):
        width, height, fps = probe_video_size(video_path)

    assert width == 640
    assert height == 480
    assert fps == 30.0
    mock_cap.release.assert_called_once()
