# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for the GPU-memory progress bar mixin (issue #974)."""

from __future__ import annotations

from typing import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.callbacks.gpu_memory_progress_bar import (
    GPUMemoryRichProgressBar,
    GPUMemoryTQDMProgressBar,
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
    """Verify _is_cuda covers non-CUDA devices, non-device inputs, uninitialized CUDA, and active cuda:N."""

    @pytest.mark.parametrize(
        "device_type",
        [
            pytest.param("cpu", id="cpu"),
            pytest.param("mps", id="mps"),
            pytest.param("xla", id="xla"),
        ],
    )
    def test_returns_false_for_non_cuda_device_type(self, device_type: str) -> None:
        """Any device whose type is not exactly "cuda" is rejected (the only device-type logic in _is_cuda)."""
        assert _is_cuda(torch.device(device_type)) is False

    def test_returns_false_for_non_device_input(self) -> None:
        """A None root_device is rejected by the isinstance guard rather than raising AttributeError."""
        assert _is_cuda(None) is False

    def test_returns_false_when_cuda_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_false_when_cuda_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        assert _is_cuda(torch.device("cuda")) is False

    def test_returns_true_for_active_cuda_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        assert _is_cuda(torch.device("cuda", 1)) is True


# ---------------------------------------------------------------------------
# TestGPUMemoryTQDMProgressBar
# ---------------------------------------------------------------------------


class TestGPUMemoryTQDMProgressBar:
    """get_metrics() on the TQDM variant."""

    def test_no_max_mem_on_cpu(self) -> None:
        """CPU device: no max_mem key, matching pre-#794 behaviour."""
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cpu"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    def test_no_max_mem_when_cuda_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cuda: device with no active CUDA context reports no max_mem."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    @pytest.mark.parametrize(
        ("device", "peak_bytes", "expected"),
        [
            pytest.param(torch.device("cuda", 0), 123 * 1024 * 1024, "123 MB", id="cuda:0"),
            pytest.param(torch.device("cuda", 1), 2 * 1024 * 1024 * 1024, "2048 MB", id="cuda:1"),
            # Boundary cases: ``:.0f`` uses round-half-to-even, so an exact .5 MB reading rounds towards the even
            # neighbour (0.5 -> 0, 1.5 -> 2) while anything above the half rounds up as expected.
            pytest.param(torch.device("cuda", 0), 0, "0 MB", id="zero-bytes"),
            pytest.param(torch.device("cuda", 0), 512 * 1024, "0 MB", id="exactly-half-mb-rounds-to-even-zero"),
            pytest.param(torch.device("cuda", 0), 512 * 1024 + 1, "1 MB", id="just-over-half-mb-rounds-up"),
            pytest.param(torch.device("cuda", 0), 1536 * 1024, "2 MB", id="one-and-a-half-mb-rounds-to-even-two"),
        ],
    )
    def test_max_mem_present_on_active_cuda_device(
        self, monkeypatch: pytest.MonkeyPatch, device: torch.device, peak_bytes: int, expected: str
    ) -> None:
        """An active CUDA device reports peak allocated memory as whole MB, space-separated from the unit."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: peak_bytes)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (0, 0))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(device)
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["max_mem"] == expected

    def test_preserves_standard_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_mem is additive: metrics from the base progress bar survive."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (0, 0))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda"))
        trainer.progress_bar_metrics = {"loss": 0.5}
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["loss"] == 0.5
        assert "max_mem" in metrics

    @pytest.mark.parametrize(
        ("device", "free_bytes", "total_bytes", "expected"),
        [
            pytest.param(torch.device("cuda", 0), 512 * 1024 * 1024, 8192 * 1024 * 1024, "512/8192 MB", id="cuda:0"),
            pytest.param(torch.device("cuda", 1), 0, 24576 * 1024 * 1024, "0/24576 MB", id="cuda:1-full"),
            # Boundary cases mirroring test_max_mem_present_on_active_cuda_device: ``:.0f`` uses round-half-to-even,
            # so an exact .5 MB reading rounds towards the even neighbour. Applied to free_bytes; total_bytes is held
            # at a value with no rounding ambiguity of its own so only one boundary is exercised at a time.
            pytest.param(
                torch.device("cuda", 0), 512 * 1024, 8192 * 1024 * 1024, "0/8192 MB", id="free-half-mb-rounds-to-even"
            ),
            pytest.param(
                torch.device("cuda", 0),
                512 * 1024 + 1,
                8192 * 1024 * 1024,
                "1/8192 MB",
                id="free-just-over-half-mb-rounds-up",
            ),
            pytest.param(
                torch.device("cuda", 0),
                1536 * 1024,
                8192 * 1024 * 1024,
                "2/8192 MB",
                id="free-one-and-a-half-mb-rounds-to-even",
            ),
        ],
    )
    def test_free_mem_present_on_active_cuda_device(
        self, monkeypatch: pytest.MonkeyPatch, device: torch.device, free_bytes: int, total_bytes: int, expected: str
    ) -> None:
        """An active CUDA device reports live free/total device memory as whole MB, formatted 'free/total MB'."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (free_bytes, total_bytes))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(device)
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["free_mem"] == expected

    def test_free_mem_queries_mem_get_info_with_the_root_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """mem_get_info must be called with trainer.strategy.root_device, not an implicit default device."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        mem_get_info = MagicMock(return_value=(0, 0))
        monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)
        bar = GPUMemoryTQDMProgressBar()
        device = torch.device("cuda", 1)
        trainer = _make_mock_trainer(device)
        pl_module = MagicMock()

        bar.get_metrics(trainer, pl_module)

        mem_get_info.assert_called_once_with(device)

    def test_free_mem_reflects_a_fresh_read_on_each_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unlike max_mem, free_mem has no reset step: it must re-query mem_get_info on every call rather than caching
        the first reading, since two calls within the same fit can see different sibling-process usage."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        readings = iter([(512 * 1024 * 1024, 8192 * 1024 * 1024), (256 * 1024 * 1024, 8192 * 1024 * 1024)])
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: next(readings))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        pl_module = MagicMock()

        first = bar.get_metrics(trainer, pl_module)["free_mem"]
        second = bar.get_metrics(trainer, pl_module)["free_mem"]

        assert (first, second) == ("512/8192 MB", "256/8192 MB")


# ---------------------------------------------------------------------------
# TestGPUMemoryRichProgressBar
# ---------------------------------------------------------------------------


class TestGPUMemoryRichProgressBar:
    """get_metrics() on the Rich variant chains through RichProgressBar.get_metrics."""

    def test_no_max_mem_on_cpu(self) -> None:
        """CPU device: no max_mem key."""
        bar = GPUMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cpu"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    def test_no_max_mem_when_cuda_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cuda: device with no active CUDA context reports no max_mem (mirrors the TQDM variant)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        bar = GPUMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda"))
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert "max_mem" not in metrics

    def test_max_mem_present_and_tensor_metrics_still_converted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RichProgressBar's tensor->float conversion still runs (mixin only adds a key)."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 456 * 1024 * 1024)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (0, 0))
        bar = GPUMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        trainer.progress_bar_metrics = {"loss": torch.tensor(0.5)}
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["max_mem"] == "456 MB"
        assert metrics["loss"] == pytest.approx(0.5)
        assert not isinstance(metrics["loss"], torch.Tensor)

    def test_free_mem_present_and_tensor_metrics_still_converted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same as above for free_mem: RichProgressBar's tensor->float conversion still runs."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (1024 * 1024 * 1024, 8192 * 1024 * 1024))
        bar = GPUMemoryRichProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        trainer.progress_bar_metrics = {"loss": torch.tensor(0.5)}
        pl_module = MagicMock()

        metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["free_mem"] == "1024/8192 MB"
        assert metrics["loss"] == pytest.approx(0.5)
        assert not isinstance(metrics["loss"], torch.Tensor)


# ---------------------------------------------------------------------------
# TestUserLoggedMaxMem
# ---------------------------------------------------------------------------


class TestUserLoggedMaxMem:
    """A user's own ``self.log("max_mem", ...)`` must not be silently replaced by the callback's reading."""

    def test_user_logged_value_wins_and_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The logged value survives and the collision is reported through PTL's rank-zero warning."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 123 * 1024 * 1024)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (0, 0))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        trainer.progress_bar_metrics = {"max_mem": "budgeted 7 GB"}
        pl_module = MagicMock()

        with pytest.warns(UserWarning, match="already tracks a metric with the name 'max_mem'"):
            metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["max_mem"] == "budgeted 7 GB"


# ---------------------------------------------------------------------------
# TestUserLoggedFreeMem
# ---------------------------------------------------------------------------


class TestUserLoggedFreeMem:
    """A user's own ``self.log("free_mem", ...)`` must not be silently replaced by the callback's reading."""

    def test_user_logged_value_wins_and_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The logged value survives and the collision is reported through PTL's rank-zero warning."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda dev=None: 0)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (123 * 1024 * 1024, 8192 * 1024 * 1024))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        trainer.progress_bar_metrics = {"free_mem": "plenty"}
        pl_module = MagicMock()

        with pytest.warns(UserWarning, match="already tracks a metric with the name 'free_mem'"):
            metrics = bar.get_metrics(trainer, pl_module)

        assert metrics["free_mem"] == "plenty"


# ---------------------------------------------------------------------------
# TestPeakMemoryReset
# ---------------------------------------------------------------------------


class _PeakMemorySpy:
    """Scriptable stand-in for the CUDA allocator's peak-memory counter.

    ``allocate`` raises the running high-water mark the way real allocations would; ``reset_peak_memory_stats`` clears
    it and records the device it was asked to clear.
    """

    def __init__(self, peak_bytes: int = 0) -> None:
        self.peak_bytes = peak_bytes
        self.reset_devices: list[torch.device | None] = []

    def allocate(self, nbytes: int) -> None:
        """Simulate an allocation of ``nbytes``, raising the peak if it is a new high-water mark."""
        self.peak_bytes = max(self.peak_bytes, nbytes)

    def max_memory_allocated(self, device: torch.device | None = None) -> int:
        """Stand in for ``torch.cuda.max_memory_allocated``."""
        return self.peak_bytes

    def reset_peak_memory_stats(self, device: torch.device | None = None) -> None:
        """Stand in for ``torch.cuda.reset_peak_memory_stats``."""
        self.reset_devices.append(device)
        self.peak_bytes = 0


def _install_peak_spy(monkeypatch: pytest.MonkeyPatch, cuda_active: bool = True) -> _PeakMemorySpy:
    """Patch torch.cuda's availability predicates and peak-memory API with a scriptable spy."""
    spy = _PeakMemorySpy()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_active)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: cuda_active)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", spy.max_memory_allocated)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", spy.reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (0, 0))
    return spy


class TestPeakMemoryReset:
    """``on_train_start`` restarts the CUDA peak counter, so ``max_mem`` is per-fit rather than per-process."""

    def test_reset_targets_the_root_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The counter cleared is the one the metric is later read from."""
        spy = _install_peak_spy(monkeypatch)
        device = torch.device("cuda", 1)
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(device)

        bar.on_train_start(trainer, MagicMock())

        assert spy.reset_devices == [device]

    def test_no_reset_without_an_active_cuda_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On CPU the reset is skipped: ``reset_peak_memory_stats`` has no availability guard of its own."""
        spy = _install_peak_spy(monkeypatch, cuda_active=False)
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cpu"))

        bar.on_train_start(trainer, MagicMock())

        assert spy.reset_devices == []

    def test_chains_to_the_base_progress_bar_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The override is additive: the concrete bar's own on_train_start still runs."""
        _install_peak_spy(monkeypatch)
        base_calls: list[tuple] = []
        monkeypatch.setattr(TQDMProgressBar, "on_train_start", lambda self, *args: base_calls.append(args))
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        pl_module = MagicMock()

        bar.on_train_start(trainer, pl_module)

        assert base_calls == [(trainer, pl_module)]

    def test_peak_does_not_accumulate_across_sequential_fits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two fit() runs in one process each report their own peak, not a shared high-water mark."""
        spy = _install_peak_spy(monkeypatch)
        bar = GPUMemoryTQDMProgressBar()
        trainer = _make_mock_trainer(torch.device("cuda", 0))
        pl_module = MagicMock()

        bar.on_train_start(trainer, pl_module)
        spy.allocate(800 * 1024 * 1024)
        first_run = bar.get_metrics(trainer, pl_module)["max_mem"]
        bar.on_train_start(trainer, pl_module)
        spy.allocate(100 * 1024 * 1024)
        second_run = bar.get_metrics(trainer, pl_module)["max_mem"]

        assert (first_run, second_run) == ("800 MB", "100 MB")


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


def _build_and_fit(
    mc: RFDETRBaseConfig,
    tc: TrainConfig,
    monkeypatch: pytest.MonkeyPatch,
    postprocess_fn=_fake_postprocess,
    **build_trainer_kwargs,
) -> list[dict]:
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

    ``TestIsCuda`` / ``TestGPUMemoryTQDMProgressBar`` / ``TestGPUMemoryRichProgressBar`` above call ``get_metrics()``
    directly against a ``MagicMock`` trainer — they verify the mixin's own logic but never exercise the
    ``on_train_batch_end`` -> ``set_postfix`` hook chain that decides what a user actually sees. These run a real
    ``build_trainer() + trainer.fit(fast_dev_run=2)`` (same fixture pattern as ``TestBuildTrainerSmoke`` in
    ``test_trainer_smoke.py``, no real dataset or model weights) and inspect what actually reached the rendered progress
    bar.
    """

    def test_cpu_fit_never_shows_max_mem(
        self,
        base_model_config: Callable[..., RFDETRBaseConfig],
        base_train_config: Callable[..., TrainConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On CPU, max_mem must not reach the rendered postfix (matches pre-#794 behaviour)."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(mc, tc, monkeypatch, accelerator="cpu")

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert all("max_mem" not in metrics for metrics in captured)

    def test_cpu_fit_never_shows_free_mem(
        self,
        base_model_config: Callable[..., RFDETRBaseConfig],
        base_train_config: Callable[..., TrainConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On CPU, free_mem must not reach the rendered postfix either."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(mc, tc, monkeypatch, accelerator="cpu")

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert all("free_mem" not in metrics for metrics in captured)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_fit_shows_max_mem_in_rendered_postfix(
        self,
        base_model_config: Callable[..., RFDETRBaseConfig],
        base_train_config: Callable[..., TrainConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On an active CUDA device, max_mem must reach the actual rendered postfix (issue #974)."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(
            mc, tc, monkeypatch, postprocess_fn=_device_matched_fake_postprocess, accelerator="gpu", devices=1
        )

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert any("max_mem" in metrics for metrics in captured)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_fit_shows_free_mem_in_rendered_postfix(
        self,
        base_model_config: Callable[..., RFDETRBaseConfig],
        base_train_config: Callable[..., TrainConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On an active CUDA device, free_mem must reach the actual rendered postfix (issue #1314)."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, progress_bar="tqdm")

        captured = _build_and_fit(
            mc, tc, monkeypatch, postprocess_fn=_device_matched_fake_postprocess, accelerator="gpu", devices=1
        )

        assert captured, "Tqdm.set_postfix was never called — the real hook chain did not reach get_metrics()"
        assert any("free_mem" in metrics for metrics in captured)
