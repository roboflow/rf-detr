# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Package-level pytest fixtures for tests/training/.

Provides cross-test cleanup that prevents class-level state from leaking between individual tests in the training/ test
package, plus shared config factory fixtures used across multiple test modules.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig

# ---------------------------------------------------------------------------
# Shared config factory fixtures (used by test_module, test_datamodule,
# test_args — avoids duplicate fixture definitions across files)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_model_config():
    """Factory fixture — call with **overrides to get a minimal RFDETRBaseConfig."""

    def _make(**overrides):
        defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
        defaults.update(overrides)
        return RFDETRBaseConfig(**defaults)

    return _make


@pytest.fixture
def base_train_config(tmp_path):
    """Factory fixture — call with **overrides to get a minimal TrainConfig.

    tmp_path is injected automatically so test methods do not need to declare it.
    """

    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            lr=1e-4,
            lr_encoder=1.5e-4,
            batch_size=2,
            weight_decay=1e-4,
            lr_scheduler_kwargs={"lr_drop": 8},
            warmup_epochs=1.0,
            drop_path=0.0,
            multi_scale=False,
            expanded_scales=False,
            grad_accum_steps=1,
            num_workers=0,
            tensorboard=False,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return _make


@pytest.fixture
def seg_train_config(tmp_path):
    """Factory fixture — call with **overrides to get a minimal SegmentationTrainConfig.

    tmp_path is injected automatically so test methods do not need to declare it.
    """

    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            batch_size=2,
            multi_scale=False,
            expanded_scales=False,
            grad_accum_steps=1,
            drop_path=0.0,
            num_workers=0,
            tensorboard=False,
        )
        defaults.update(overrides)
        return SegmentationTrainConfig(**defaults)

    return _make


# ---------------------------------------------------------------------------
# Class-level isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_rfdetr_module_trainer_property():
    """Restore RFDETRModelModule.trainer to the LightningModule parent property after each test.

    Several unit tests in test_module_model.py patch the ``trainer`` property directly on the ``RFDETRModelModule``
    class (``type(module).trainer = property(...)``). Without cleanup this mutates the class for the remainder of the
    session and breaks ``Trainer.fit()`` calls in smoke tests (PTL cannot set ``.trainer`` on the module because the
    patched property has no setter).

    This fixture deletes any class-level override from ``RFDETRModelModule.__dict__`` after every test, so the next test
    starts with a clean class that inherits PTL's read/write ``trainer`` descriptor from ``LightningModule``.
    """
    yield
    # Lazy import so the fixture does not force module import at collection time.
    from rfdetr.training.module_model import RFDETRModelModule

    if "trainer" in RFDETRModelModule.__dict__:
        delattr(RFDETRModelModule, "trainer")


@pytest.fixture(autouse=True)
def _restore_rfdetr_datamodule_trainer_property():
    """Restore RFDETRDataModule.trainer to the LightningDataModule parent property after each test.

    Tests that mock the ``trainer`` property on ``RFDETRDataModule`` (e.g. for ``on_after_batch_transfer`` tests) patch
    it at the class level.  Without cleanup this mutates the class for the remainder of the session.

    This fixture deletes any class-level override from ``RFDETRDataModule.__dict__`` after every test, mirroring the
    ``_restore_rfdetr_module_trainer_property`` pattern above.
    """
    yield
    from rfdetr.training.module_data import RFDETRDataModule

    if "trainer" in RFDETRDataModule.__dict__:
        delattr(RFDETRDataModule, "trainer")


# ---------------------------------------------------------------------------
# Training entry-point patches shared by the test_detr_shim*.py files and the
# Trainer capture shared by the test_build_trainer*.py files.
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_lit(mocker: MockerFixture) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Patch the rfdetr.training entry points and return their ``(module_cls, datamodule_cls, build_trainer)`` mocks."""
    return (
        mocker.patch("rfdetr.training.RFDETRModelModule", MagicMock(name="RFDETRModule_cls")),
        mocker.patch("rfdetr.training.RFDETRDataModule", MagicMock(name="RFDETRDataModule_cls")),
        mocker.patch("rfdetr.training.build_trainer", MagicMock(name="build_trainer")),
    )


@pytest.fixture
def captured_trainer_kwargs(mocker: MockerFixture) -> dict[str, Any]:
    """Patch ``rfdetr.training.trainer.Trainer`` and return the kwargs its calls capture.

    Every test that requests it calls ``build_trainer()`` and inspects what it passed to ``Trainer(**kwargs)`` instead
    of constructing a real one; this fixture centralizes that capture in place of a per-test ``_fake_trainer`` closure.

    Examples:
        >>> captured_trainer_kwargs(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    captured: dict[str, Any] = {}

    def _fake_trainer(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    mocker.patch("rfdetr.training.trainer.Trainer", side_effect=_fake_trainer)
    return captured


@pytest.fixture
def mock_xla_precision(mocker: MockerFixture) -> MagicMock:
    """Replace ``XLAPrecision`` so XLA dispatch tests run without ``torch_xla``; returns the mocked class.

    Examples:
        >>> mock_xla_precision(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    return mocker.patch("pytorch_lightning.plugins.XLAPrecision")
