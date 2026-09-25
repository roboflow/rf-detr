# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Routing tests for the combined CUDA training update."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import strategies as lightning_strategies
from torch import nn

from rfdetr.training import module_model
from rfdetr.training.fused_adamw_ema import LIBDEVICE_FUNCTIONS, UNSUPPORTED_ADAMW_OPTIONS, FusedAdamWEMA
from rfdetr.training.module_model import RFDETRModelModule

from .test_module_model import _base_train_config, _build_module


def _setup_module(tmp_path: Path | None, **train_overrides: Any) -> tuple[RFDETRModelModule, list[dict[str, Any]]]:
    """Build a mocked ``RFDETRModelModule`` with a trainer attached and one optimizer parameter group.

    Examples:
        >>> module, param_dicts = _setup_module(None, use_ema=True)
        >>> module.train_config.use_ema, len(param_dicts)
        (True, 1)
    """
    train_config = _base_train_config(tmp_path, **train_overrides)
    module, _, _, _ = _build_module(train_config=train_config)
    trainer = MagicMock(estimated_stepping_batches=1000)
    module._trainer = trainer
    type(module).trainer = property(lambda self: self._trainer)
    parameter = nn.Parameter(torch.randn(4, 4))
    return module, [{"params": parameter, "lr": train_config.lr}]


_COMPLETE_LIBDEVICE = SimpleNamespace(**{name: object() for name in LIBDEVICE_FUNCTIONS})
_LIBDEVICE_WITHOUT_POW = SimpleNamespace(**{name: object() for name in LIBDEVICE_FUNCTIONS if name != "pow"})


class TestKernelSupportGate:
    """Exercise the real Triton-support predicate; the routing tests below only mock its outcome."""

    @pytest.mark.parametrize(
        ("platform", "hip", "libdevice", "expected"),
        [
            pytest.param("linux", None, _COMPLETE_LIBDEVICE, True, id="linux-cuda-complete-libdevice"),
            pytest.param("win32", None, _COMPLETE_LIBDEVICE, False, id="windows"),
            pytest.param("darwin", None, _COMPLETE_LIBDEVICE, False, id="macos"),
            pytest.param("linux", "6.1.0", _COMPLETE_LIBDEVICE, False, id="rocm-build"),
            pytest.param("linux", None, ModuleNotFoundError("triton"), False, id="triton-missing"),
            pytest.param("linux", None, ImportError("libdevice"), False, id="libdevice-module-missing"),
            pytest.param("linux", None, _LIBDEVICE_WITHOUT_POW, False, id="libdevice-lacks-a-kernel-function"),
        ],
    )
    def test_gate_checks_platform_backend_and_libdevice_functions(
        self, platform: str, hip: str | None, libdevice: object, expected: bool
    ) -> None:
        """Each condition alone keeps the standard optimizer, even when the others would allow the kernel."""
        module_model._has_fused_adamw_ema_kernel.cache_clear()
        try:
            with (
                patch.object(module_model, "sys", SimpleNamespace(platform=platform)),
                patch.object(torch.version, "hip", hip),
                patch.object(
                    module_model.importlib,
                    "import_module",
                    side_effect=libdevice if isinstance(libdevice, Exception) else None,
                    return_value=libdevice,
                ) as import_module,
            ):
                assert module_model._has_fused_adamw_ema_kernel() is expected
            # Whenever the gate imports, it asks for exactly the module the kernels' libdevice functions live in.
            assert {call.args for call in import_module.call_args_list} <= {("triton.language.extra.cuda.libdevice",)}
        finally:
            module_model._has_fused_adamw_ema_kernel.cache_clear()

    def test_required_functions_are_exactly_those_the_kernels_call(self) -> None:
        """The gate's function list follows the kernel source, so a kernel edit cannot outrun the Triton check."""
        spec = importlib.util.find_spec("rfdetr.training._fused_adamw_ema_triton")
        assert spec is not None and spec.origin is not None
        tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        called = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "libdevice"
        }

        assert called, "the kernel source no longer references libdevice; revisit the gate and this test"
        assert called == set(LIBDEVICE_FUNCTIONS)


class TestFusedAdamWEMARouting:
    """Exercise eligibility and construction through RFDETRModelModule."""

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_requires_measured_training_path(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The custom update is limited to compiled, single-GPU AdamW + per-step EMA."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1, devices=1, num_nodes=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_adamw_ema is True

        module.train_config.ema_update_interval = 2
        assert module._use_fused_adamw_ema is False
        module.train_config.ema_update_interval = 1
        module.train_config.devices = 2
        assert module._use_fused_adamw_ema is False
        module.train_config.devices = 1
        module._trainer.world_size = 2
        assert module._use_fused_adamw_ema is False
        module._trainer.world_size = 1
        module.model_config.compile = False
        module._compile_active = False
        assert module._use_fused_adamw_ema is False

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @pytest.mark.parametrize(
        ("owner", "attribute", "value"),
        [
            pytest.param("train_config", "use_ema", False, id="ema-disabled"),
            pytest.param("train_config", "num_nodes", 2, id="multi-node"),
            pytest.param("train_config", "strategy", "ddp", id="explicit-strategy"),
            pytest.param("train_config", "devices", "1", id="devices-as-string"),
            pytest.param("train_config", "devices", "auto", id="devices-auto"),
            pytest.param("model_config", "fused_optimizer", False, id="fused-optimizer-off"),
            pytest.param("model_config", "segmentation_head", True, id="segmentation"),
            pytest.param("model_config", "use_grouppose_keypoints", True, id="keypoints"),
            pytest.param("model_config", "cuda_graphs", True, id="cuda-graphs"),
            pytest.param("_trainer", "precision", "bf16-true", id="bf16-true-precision"),
            pytest.param("_trainer", "precision", "32-true", id="fp32-precision"),
        ],
    )
    def test_each_exclusion_keeps_the_existing_optimizer(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        owner: str,
        attribute: str,
        value: object,
        tmp_path: Path,
    ) -> None:
        """Each documented condition alone excludes the combined update while every other condition holds."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1, devices=1, num_nodes=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"
        assert module._use_fused_adamw_ema is True

        setattr(getattr(module, owner), attribute, value)

        assert module._use_fused_adamw_ema is False

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @pytest.mark.parametrize(
        "strategy_name",
        ["DDPStrategy", "FSDPStrategy", "DeepSpeedStrategy", "XLAStrategy", "SingleDeviceXLAStrategy"],
    )
    def test_distributed_trainer_strategy_keeps_the_existing_optimizer(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        strategy_name: str,
        tmp_path: Path,
    ) -> None:
        """A Lightning strategy that shards, replicates or lowers the model excludes the combined update."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1, devices=1, num_nodes=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"
        module._trainer.strategy = object.__new__(lightning_strategies.SingleDeviceStrategy)
        assert module._use_fused_adamw_ema is True

        module._trainer.strategy = object.__new__(getattr(lightning_strategies, strategy_name))

        assert module._use_fused_adamw_ema is False

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @pytest.mark.parametrize("option", UNSUPPORTED_ADAMW_OPTIONS)
    def test_unsupported_adamw_options_disable_combined_update(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        option: str,
        tmp_path: Path,
    ) -> None:
        """An AdamW option the kernel does not implement keeps the existing optimizer; leaving it off does not."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1, optimizer_kwargs={option: True})
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_adamw_ema is False

        module.train_config.optimizer_kwargs = {option: False}
        assert module._use_fused_adamw_ema is True

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @pytest.mark.parametrize("option", ["amsgrad", "maximize"])
    def test_configure_keeps_fused_adamw_for_options_torch_accepts(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        mock_get_param_dict: MagicMock,
        option: str,
        tmp_path: Path,
    ) -> None:
        """Options that torch's fused AdamW accepts today must not start failing on the eligible route."""
        module, param_dicts = _setup_module(
            tmp_path, use_ema=True, ema_update_interval=1, optimizer_kwargs={option: True}
        )
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert type(optimizer) is torch.optim.AdamW
        assert optimizer.defaults[option] is True

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @pytest.mark.parametrize("combined", [pytest.param(True, id="combined-route"), pytest.param(False, id="standard")])
    def test_invalid_adamw_hyperparameters_keep_torch_value_error(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        mock_get_param_dict: MagicMock,
        combined: bool,
        tmp_path: Path,
    ) -> None:
        """Only unknown arguments become the wrapped TypeError; torch's ValueError for bad values is unchanged."""
        module, param_dicts = _setup_module(
            tmp_path, use_ema=True, ema_update_interval=1, optimizer_kwargs={"betas": (1.5, 0.9)}
        )
        module.model_config.compile = combined
        module._compile_active = combined
        module._trainer.precision = "bf16-mixed"
        mock_get_param_dict.return_value = param_dicts
        assert module._use_fused_adamw_ema is combined

        with pytest.raises(ValueError, match="Invalid beta parameter"):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_configure_selects_combined_optimizer(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        mock_get_param_dict: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Eligible training builds the optimizer combining clipping, AdamW, and EMA."""
        module, param_dicts = _setup_module(tmp_path, use_ema=True, ema_update_interval=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, FusedAdamWEMA)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.max_grad_norm == pytest.approx(module.train_config.clip_max_norm)

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=False)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_falls_back_when_triton_api_is_unavailable(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Older Triton keeps the existing fused AdamW path instead of failing at first step."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_adamw_ema is False

    @patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_reads_runtime_gradient_clip_override(
        self,
        mock_cuda_available: MagicMock,
        mock_bf16_supported: MagicMock,
        mock_kernel_available: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Lightning's active clipping value wins over the nearby TrainConfig value."""
        module, _ = _setup_module(tmp_path, use_ema=True, ema_update_interval=1)
        module.model_config.compile = True
        module._compile_active = True
        module._trainer.precision = "bf16-mixed"
        optimizer = MagicMock()
        optimizer.optimizer = optimizer
        optimizer.set_max_grad_norm = MagicMock()

        module.clip_gradients(optimizer, gradient_clip_val=0.25, gradient_clip_algorithm="norm")

        optimizer.set_max_grad_norm.assert_called_once_with(0.25)
