# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for the GPU-memory progress bar mixin (issue #974)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

from rfdetr.training import build_trainer
from rfdetr.training.callbacks.gpu_memory_progress_bar import (
    GpuMemoryRichProgressBar,
    GpuMemoryTQDMProgressBar,
    _is_cuda,
)
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from ..helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_trainer(device: torch.device) -> MagicMock:
    """Create a minimal mock Trainer exposing what ProgressBar.get_metrics reads."""
    trainer = MagicMock()
    trainer.strategy.root_device = device
    trainer.loggers = []
    trainer.progress_bar_metrics = {}
    return trainer


# ---------------------------------------------------------------------------
# TestIsCuda
# ---------------------------------------------------------------------------


class TestIsCuda:
    """Verify _is_cuda covers CPU, uninitialized CUDA, and active cuda:N."""

    def test_returns_false_for_cpu_device(self) -> None:
        assert _is_cuda(torch.device("cpu")) is False

    def test_returns_false_when_cuda_unavailable(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_false_when_cuda_not_initialized(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_true_for_active_cuda_device(self, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda", 1)) is True


# ---------------------------------------------------------------------------
# TestGpuMemoryTQDMProgressBar
# ---------------------------------------------------------------------------


class TestGpuMemoryTQDMProgressBar:
    """get_metrics() on the TQDM variant."""

    def test_no_max_mem_on_cpu(self) -> None:
        """CPU device: no max_mem key, matching pre-#794 behaviour."""
        bar = GpuMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cpu"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    def test_no_max_mem_when_cuda_not_initialized(self, monkeypatch) -> None:
        """A cuda: device with no active CUDA context reports no max_mem."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        bar = GpuMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    @pytest.mark.parametrize(
        ("device", "peak_bytes", "expected"),
        [
            pytest.param(torch.device("cuda", 0), 123 * 1024 * 1024, "123MB", id="cuda:0"),
            pytest.param(torch.device("cuda", 1), 2 * 1024 * 1024 * 1024, "2048MB", id="cuda:1"),
        ],
    )
    def test_max_mem_present_on_active_cuda_device(
        self, monkeypatch, device: torch.device, peak_bytes: int, expected: str
    ) -> None:
        """An active CUDA device reports peak allocated memory in MB."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: peak_bytes)
        bar = GpuMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(device)
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["max_mem"] == expected

    def test_preserves_standard_metrics(self, monkeypatch) -> None:
        """max_mem is additive: metrics from the base progress bar survive."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        bar = GpuMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda"))
        trainer.progress_bar_metrics = {"loss": 0.5}
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["loss"] == 0.5
        assert "max_mem" in metrics


# ---------------------------------------------------------------------------
# TestGpuMemoryRichProgressBar
# ---------------------------------------------------------------------------


class TestGpuMemoryRichProgressBar:
    """get_metrics() on the Rich variant chains through RichProgressBar.get_metrics."""

    def test_no_max_mem_on_cpu(self) -> None:
        """CPU device: no max_mem key."""
        bar = GpuMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cpu"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    def test_max_mem_present_and_tensor_metrics_still_converted(self, monkeypatch) -> None:
        """RichProgressBar's tensor->float conversion still runs (mixin only adds a key)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 456 * 1024 * 1024)
        bar = GpuMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        trainer.progress_bar_metrics = {"loss": torch.tensor(0.5)}
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["max_mem"] == "456MB"
        assert metrics["loss"] == pytest.approx(0.5)
        assert not isinstance(metrics["loss"], torch.Tensor)


# ---------------------------------------------------------------------------
# TestProgressBarEndToEnd
# ---------------------------------------------------------------------------


def _device_matched_fake_postprocess(outputs, orig_sizes):
    """Like ``helpers._fake_postprocess``, but on ``orig_sizes.device``.

    The shared helper hardcodes CPU tensors, which is fine on the "cpu" accelerator (the only one the rest of the suite
    exercises) but crashes ``COCOEvalCallback``'s box-IoU matching with a device mismatch once the model output is a
    real CUDA tensor. Only needed here — no other test in the suite runs ``build_trainer`` on "gpu".
    """
    predictions = _fake_postprocess(outputs, orig_sizes)
    return [{key: value.to(orig_sizes.device) for key, value in pred.items()} for pred in predictions]


def _build_and_fit(mc, tc, monkeypatch, postprocess_fn=_fake_postprocess, **build_trainer_kwargs) -> list[dict]:
    """Run a real ``trainer.fit()`` (mirrors ``TestBuildTrainerSmoke`` in test_trainer_smoke.py) and return every
    metrics dict that reached ``Tqdm.set_postfix`` — the actual rendering sink, not a mocked trainer."""
    captured: list[dict] = []
    original_set_postfix = Tqdm.set_postfix

    def _spy(self, ordered_dict=None, refresh=True, **kwargs):
        if ordered_dict:
            captured.append(dict(ordered_dict))
        return original_set_postfix(self, ordered_dict, refresh, **kwargs)

    monkeypatch.setattr(Tqdm, "set_postfix", _spy)

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=postprocess_fn)),
        ),
        patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(tc, mc, fast_dev_run=2, **build_trainer_kwargs)
        trainer.fit(module, datamodule=datamodule)

    return captured


class TestProgressBarEndToEnd:
    """``get_metrics()`` must fire through the real PTL hook chain, not a hand-built mock ``Trainer``.

    ``TestIsCuda`` / ``TestGpuMemoryTQDMProgressBar`` / ``TestGpuMemoryRichProgressBar`` above call ``get_metrics()``
    directly against a ``MagicMock`` trainer — they verify the mixin's own logic but never exercise the
    ``on_train_batch_end`` -> ``set_postfix`` hook chain that decides what a user actually sees. These run a real
    ``build_trainer() + trainer.fit(fast_dev_run=2)`` (same fixture pattern as ``TestBuildTrainerSmoke`` in
    ``test_trainer_smoke.py``, no real dataset or model weights) and inspect what actually reached the rendered progress
    bar.
    """

    def test_cpu_fit_never_shows_max_mem(self, base_model_config, base_train_config, monkeypatch) -> None:
        """On CPU, max_mem must not reach the rendered postfix (matches pre-#794 behaviour)."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(mc, tc, monkeypatch, accelerator="cpu")

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert all("max_mem" not in metrics for metrics in captured)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_fit_shows_max_mem_in_rendered_postfix(
        self, base_model_config, base_train_config, monkeypatch
    ) -> None:
        """On an active CUDA device, max_mem must reach the actual rendered postfix (issue #974)."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(
            mc, tc, monkeypatch, postprocess_fn=_device_matched_fake_postprocess, accelerator="gpu", devices=1
        )

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert any("max_mem" in metrics for metrics in captured)
