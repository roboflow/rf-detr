# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared config builders and helpers for the RFDETR training-shim tests (``test_detr_shim*.py``).

Import them with ``from tests.training._shim_support import ...``. The ``patch_lit`` fixture these tests use lives in
``tests/training/conftest.py``.
"""

import json
import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from rfdetr.config import RFDETRBaseConfig, TrainConfig


def _make_model_config(**overrides):
    """Build a minimal RFDETRBaseConfig for shim tests.

    Examples:
        >>> config = _make_model_config(num_classes=7)
        >>> config.device, config.num_classes, config.pretrain_weights
        ('cpu', 7, None)
    """
    defaults = dict(pretrain_weights=None, num_classes=3, device="cpu")
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _make_train_config(tmp_path, **overrides):
    """Build a minimal TrainConfig for shim tests.

    Examples:
        >>> from pathlib import Path
        >>> config = _make_train_config(Path("/tmp/example"), epochs=3)
        >>> config.epochs, Path(config.dataset_dir).name, Path(config.output_dir).name
        (3, 'ds', 'out')
    """
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_rfdetr_self(tmp_path, **train_overrides):
    """Return a MagicMock shaped like RFDETR with real config objects.

    No spec is used because RFDETR.model is set in __init__ (instance attr) and spec=RFDETR would block access to it.

    Examples:
        >>> from pathlib import Path
        >>> mock_self = _make_rfdetr_self(Path('/tmp/example'))
        >>> mock_self.model_config.device, Path(mock_self.get_train_config().output_dir).name
        ('cpu', 'out')
    """
    mock = MagicMock()
    mock.model_config = _make_model_config()
    mock.model = MagicMock()  # exposes mock.model.model for sync-back assertions
    mock.get_train_config.return_value = _make_train_config(tmp_path, **train_overrides)
    return mock


def _read_training_config(path: str) -> dict[str, Any] | None:
    """Return the parsed training_config.json at *path*, or None when it does not exist.

    Examples:
        >>> import json, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     written = os.path.join(directory, "training_config.json")
        ...     _ = Path(written).write_text(json.dumps({"num_classes": 2}))
        ...     _read_training_config(written)
        {'num_classes': 2}
        >>> _read_training_config("/nonexistent/training_config.json") is None
        True
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Errors the best-effort dataset class-name lookup must swallow instead of blocking training.
_DATASET_READ_ERRORS = [
    pytest.param(FileNotFoundError("no such dataset"), id="file-not-found"),
    pytest.param(ValueError("bad dataset"), id="value-error"),
    pytest.param(KeyError("missing key"), id="key-error"),
    pytest.param(OSError("io error"), id="os-error"),
]
