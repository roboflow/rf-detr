# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared config builders for the ``build_trainer()`` tests (``test_build_trainer*.py``).

Import them with ``from tests.training._trainer_support import ...``. The ``captured_trainer_kwargs`` and
``mock_xla_precision`` fixtures these tests use live in ``tests/training/conftest.py``.
"""

from rfdetr.config import KeypointTrainConfig, RFDETRBaseConfig, TrainConfig


def _mc(**kwargs):
    """Minimal RFDETRBaseConfig for tests.

    Examples:
        >>> config = _mc(num_classes=7)
        >>> config.device, config.num_classes
        ('cpu', 7)
    """
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
    defaults.update(kwargs)
    return RFDETRBaseConfig(**defaults)


def _tc(tmp_path, config_cls=TrainConfig, **kwargs):
    """Minimal TrainConfig (or ``config_cls``) for tests.

    Loggers are disabled by default to avoid requiring optional deps (tensorboard, wandb, mlflow) in the CPU test
    environment.  Logger-specific tests override these explicitly via kwargs or mocking.

    Examples:
        >>> from pathlib import Path
        >>> config = _tc(Path("/tmp/example"), epochs=3)
        >>> config.epochs, Path(config.dataset_dir).name, Path(config.output_dir).name
        (3, 'ds', 'out')
    """
    defaults = dict(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=2,
        num_workers=0,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    defaults.update(kwargs)
    return config_cls(**defaults)


def _kp_tc(tmp_path, **kwargs):
    """Minimal KeypointTrainConfig for tests that exercise keypoint model paths.

    Examples:
        >>> from pathlib import Path
        >>> config = _kp_tc(Path('/tmp/example'), batch_size=4)
        >>> config.batch_size, Path(config.dataset_dir).name
        (4, 'ds')
    """
    return _tc(tmp_path, config_cls=KeypointTrainConfig, **kwargs)


def _configs(tmp_path, *, keypoint=False, segmentation=False, **tc_kwargs):
    """Return a ``(train_config, model_config)`` pair for a detection, segmentation or keypoint run without EMA.

    Examples:
        >>> from pathlib import Path
        >>> tc, mc = _configs(Path('/tmp/example'), keypoint=True, strategy='ddp')
        >>> type(tc).__name__, tc.strategy, mc.use_grouppose_keypoints
        ('KeypointTrainConfig', 'ddp', True)
    """
    tc_kwargs.setdefault("use_ema", False)
    if keypoint:
        return _kp_tc(tmp_path, **tc_kwargs), _mc(use_grouppose_keypoints=True)
    return _tc(tmp_path, **tc_kwargs), _mc(segmentation_head=segmentation)
