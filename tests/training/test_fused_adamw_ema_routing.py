# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Routing tests for the combined CUDA training update."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
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


def _eligible_module(
    tmp_path: Path | None, compiled: bool = True, **train_overrides: Any
) -> tuple[RFDETRModelModule, list[dict[str, Any]]]:
    """Build a module on the combined-update path: EMA every step, compiled (unless told otherwise), bf16 mixed.

    Args:
        tmp_path: Output directory for the train config, or ``None``.
        compiled: Whether torch compilation is active.
        **train_overrides: Train config fields overriding the eligible defaults.

    Examples:
        >>> module, param_dicts = _eligible_module(None)
        >>> module.train_config.use_ema, module.train_config.ema_update_interval, module._trainer.precision
        (True, 1, 'bf16-mixed')
    """
    train_overrides = {"use_ema": True, "ema_update_interval": 1, **train_overrides}
    module, param_dicts = _setup_module(tmp_path, **train_overrides)
    module.model_config.compile = compiled
    module._compile_active = compiled
    module._trainer.precision = "bf16-mixed"
    return module, param_dicts


@pytest.fixture
def cuda_bf16_runtime(mocker: MockerFixture) -> None:
    """Report a CUDA device that supports bf16."""
    mocker.patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    mocker.patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)


@pytest.fixture
def cuda_kernel_runtime(cuda_bf16_runtime: None, mocker: MockerFixture) -> None:
    """Report a CUDA device with bf16 support and a Triton that provides the combined kernels."""
    mocker.patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=True)


class TestKernelSupportGate:
    """Exercise the real Triton-support predicate; the routing tests below only mock its outcome."""

    @pytest.fixture(autouse=True)
    def _fresh_gate(self) -> Iterator[None]:
        """Drop the gate's cached verdict before and after each test."""
        module_model._has_fused_adamw_ema_kernel.cache_clear()
        yield
        module_model._has_fused_adamw_ema_kernel.cache_clear()

    @pytest.mark.parametrize(
        ("platform", "hip", "libdevice", "expected"),
        [
            pytest.param("linux", None, _COMPLETE_LIBDEVICE, True, id="linux-cuda-complete-libdevice"),
            pytest.param("win32", None, _COMPLETE_LIBDEVICE, False, id="windows"),
            pytest.param("darwin", None, _COMPLETE_LIBDEVICE, False, id="macos"),
            pytest.param("linux", "6.1.0", _COMPLETE_LIBDEVICE, False, id="rocm-build"),
            pytest.param(
                "linux",
                None,
                ModuleNotFoundError("No module named 'triton'", name="triton"),
                False,
                id="triton-missing",
            ),
            pytest.param(
                "linux",
                None,
                ModuleNotFoundError(
                    "No module named 'triton.language.extra.cuda.libdevice'",
                    name="triton.language.extra.cuda.libdevice",
                ),
                False,
                id="libdevice-module-missing",
            ),
            pytest.param("linux", None, _LIBDEVICE_WITHOUT_POW, False, id="libdevice-lacks-a-kernel-function"),
        ],
    )
    def test_gate_checks_platform_backend_and_libdevice_functions(
        self, mocker: MockerFixture, platform: str, hip: str | None, libdevice: object, expected: bool
    ) -> None:
        """Each condition alone keeps the standard optimizer, even when the others would allow the kernel."""
        mocker.patch.object(module_model, "sys", SimpleNamespace(platform=platform))
        mocker.patch.object(torch.version, "hip", hip)
        import_module = mocker.patch.object(
            module_model.importlib,
            "import_module",
            side_effect=libdevice if isinstance(libdevice, Exception) else None,
            return_value=libdevice,
        )

        assert module_model._has_fused_adamw_ema_kernel() is expected
        # Whenever the gate imports, it asks for exactly the module the kernels' libdevice functions live in.
        assert {call.args for call in import_module.call_args_list} <= {("triton.language.extra.cuda.libdevice",)}

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(ImportError("transitive dependency is broken"), id="broken-transitive-import"),
            pytest.param(
                ModuleNotFoundError("No module named 'unrelated_dependency'", name="unrelated_dependency"),
                id="missing-transitive-dependency",
            ),
        ],
    )
    def test_transitive_import_error_is_not_treated_as_missing_optional_triton(
        self, mocker: MockerFixture, error: ImportError
    ) -> None:
        """A broken dependency imported by libdevice must remain visible to the caller."""
        mocker.patch.object(module_model, "sys", SimpleNamespace(platform="linux"))
        mocker.patch.object(torch.version, "hip", None)
        mocker.patch.object(module_model.importlib, "import_module", side_effect=error)

        with pytest.raises(ImportError) as exc_info:
            module_model._has_fused_adamw_ema_kernel()

        assert str(error) in str(exc_info.value)

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

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    @pytest.mark.parametrize(
        ("owner", "attribute", "value"),
        [
            pytest.param("train_config", "use_ema", False, id="ema-disabled"),
            pytest.param("train_config", "ema_update_interval", 2, id="ema-every-other-step"),
            pytest.param("train_config", "num_nodes", 2, id="multi-node"),
            pytest.param("train_config", "devices", 2, id="two-devices"),
            pytest.param("train_config", "strategy", "ddp", id="explicit-strategy"),
            pytest.param("train_config", "devices", "1", id="devices-as-string"),
            pytest.param("train_config", "devices", "auto", id="devices-auto"),
            pytest.param("model_config", "fused_optimizer", False, id="fused-optimizer-off"),
            pytest.param("model_config", "segmentation_head", True, id="segmentation"),
            pytest.param("model_config", "use_grouppose_keypoints", True, id="keypoints"),
            pytest.param("model_config", "cuda_graphs", True, id="cuda-graphs"),
            pytest.param("_trainer", "precision", "bf16-true", id="bf16-true-precision"),
            pytest.param("_trainer", "precision", "32-true", id="fp32-precision"),
            pytest.param("_trainer", "world_size", 2, id="world-size-two"),
        ],
    )
    def test_each_exclusion_keeps_the_existing_optimizer(
        self, tmp_path: Path, owner: str, attribute: str, value: object
    ) -> None:
        """Each documented condition alone excludes the combined update while every other condition holds."""
        module, _ = _eligible_module(tmp_path, devices=1, num_nodes=1)
        assert module._use_fused_adamw_ema is True

        setattr(getattr(module, owner), attribute, value)

        assert module._use_fused_adamw_ema is False

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    def test_inactive_compilation_keeps_the_existing_optimizer(self, tmp_path: Path) -> None:
        """A model that is neither configured nor running compiled keeps the existing optimizer."""
        module, _ = _eligible_module(tmp_path, compiled=False, devices=1, num_nodes=1)

        assert module._use_fused_adamw_ema is False

    def test_combined_checkpoint_keeps_fp32_ddp_destination_unfused(self, mocker: MockerFixture) -> None:
        """A combined-route checkpoint must not re-enable fused AdamW for an FP32 DDP resume."""
        source_model = nn.Linear(2, 1)
        source_optimizer = FusedAdamWEMA(
            source_model.parameters(),
            named_parameters=dict(source_model.named_parameters()),
            model_buffers=dict(source_model.named_buffers()),
            max_grad_norm=1.0,
            ema_decay=0.99,
            ema_tau=0,
        )
        source_model(torch.ones(1, 2)).sum().backward()
        source_optimizer.step()
        checkpoint = {"optimizer_states": [source_optimizer.state_dict()]}

        module, _ = _setup_module(None, optimizer="adamw")
        module.model_config.fused_optimizer = True
        module._trainer.precision = "32-true"
        module._trainer.strategy = object.__new__(lightning_strategies.DDPStrategy)
        mocker.patch.object(torch.cuda, "is_available", return_value=True)
        mocker.patch.object(torch.cuda, "is_bf16_supported", return_value=True)
        assert module._use_fused_optimizer is False
        module.on_load_checkpoint(checkpoint)

        resumed_model = nn.Linear(2, 1)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), fused=False)
        resumed_optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        assert [group["fused"] for group in resumed_optimizer.param_groups] == [False]

        resumed_model(torch.ones(1, 2)).sum().backward()
        resumed_optimizer.step()

        assert {int(state["step"]) for state in resumed_optimizer.state.values()} == {2}

    @pytest.mark.usefixtures("cuda_bf16_runtime")
    def test_keypoints_do_not_exclude_the_plain_fused_optimizer(self, tmp_path: Path) -> None:
        """``use_grouppose_keypoints`` excludes the combined EMA update but not the plain fused AdamW path.

        ``_use_fused_adamw_ema`` explicitly excludes keypoints; ``_use_fused_optimizer`` (the branch
        :meth:`RFDETRModelModule.configure_gradient_clipping` uses for its non-EMA ``clip_grad_norm_`` call) has no
        such exclusion, so a bf16 keypoint run on CUDA reaches that fused-clip branch — a combination the suite's
        other fused-routing tests never construct.
        """
        module, _ = _setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"
        module.model_config.use_grouppose_keypoints = True

        assert module._use_fused_optimizer is True
        assert module._use_fused_adamw_ema is False

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    @pytest.mark.parametrize(
        "strategy_name", ["DDPStrategy", "FSDPStrategy", "DeepSpeedStrategy", "XLAStrategy", "SingleDeviceXLAStrategy"]
    )
    def test_distributed_trainer_strategy_keeps_the_existing_optimizer(
        self, strategy_name: str, tmp_path: Path
    ) -> None:
        """A Lightning strategy that shards, replicates or lowers the model excludes the combined update."""
        module, _ = _eligible_module(tmp_path, devices=1, num_nodes=1)
        module._trainer.strategy = object.__new__(lightning_strategies.SingleDeviceStrategy)
        assert module._use_fused_adamw_ema is True

        module._trainer.strategy = object.__new__(getattr(lightning_strategies, strategy_name))

        assert module._use_fused_adamw_ema is False

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    @pytest.mark.parametrize("option", UNSUPPORTED_ADAMW_OPTIONS)
    def test_unsupported_adamw_options_disable_combined_update(self, option: str, tmp_path: Path) -> None:
        """An AdamW option the kernel does not implement keeps the existing optimizer; leaving it off does not."""
        module, _ = _eligible_module(tmp_path, optimizer_kwargs={option: True})

        assert module._use_fused_adamw_ema is False

        module.train_config.optimizer_kwargs = {option: False}
        assert module._use_fused_adamw_ema is True

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    @pytest.mark.parametrize("option", ["amsgrad", "maximize"])
    def test_configure_keeps_fused_adamw_for_options_torch_accepts(
        self, mocker: MockerFixture, option: str, tmp_path: Path
    ) -> None:
        """Options that torch's fused AdamW accepts today must not start failing on the eligible route."""
        module, param_dicts = _eligible_module(tmp_path, optimizer_kwargs={option: True})
        mocker.patch("rfdetr.training.module_model.get_param_dict", return_value=param_dicts)

        optimizer = module.configure_optimizers()["optimizer"]

        assert type(optimizer) is torch.optim.AdamW
        assert optimizer.defaults[option] is True

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    @pytest.mark.parametrize("combined", [pytest.param(True, id="combined-route"), pytest.param(False, id="standard")])
    def test_invalid_adamw_hyperparameters_keep_torch_value_error(
        self, mocker: MockerFixture, combined: bool, tmp_path: Path
    ) -> None:
        """Only unknown arguments become the wrapped TypeError; torch's ValueError for bad values is unchanged."""
        module, param_dicts = _eligible_module(tmp_path, compiled=combined, optimizer_kwargs={"betas": (1.5, 0.9)})
        mocker.patch("rfdetr.training.module_model.get_param_dict", return_value=param_dicts)
        assert module._use_fused_adamw_ema is combined

        with pytest.raises(ValueError) as exc_info:
            module.configure_optimizers()

        assert "Invalid beta parameter" in str(exc_info.value)

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    def test_configure_selects_combined_optimizer(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Eligible training builds the optimizer combining clipping, AdamW, and EMA."""
        module, param_dicts = _eligible_module(tmp_path)
        mocker.patch("rfdetr.training.module_model.get_param_dict", return_value=param_dicts)

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, FusedAdamWEMA)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.max_grad_norm == pytest.approx(module.train_config.clip_max_norm)

    @pytest.mark.usefixtures("cuda_bf16_runtime")
    def test_falls_back_when_triton_api_is_unavailable(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Older Triton keeps the existing fused AdamW path instead of failing at first step."""
        mocker.patch("rfdetr.training.module_model._has_fused_adamw_ema_kernel", return_value=False)
        module, _ = _eligible_module(tmp_path)

        assert module._use_fused_adamw_ema is False

    @pytest.mark.usefixtures("cuda_kernel_runtime")
    def test_reads_runtime_gradient_clip_override(self, tmp_path: Path) -> None:
        """Lightning's active clipping value wins over the nearby TrainConfig value."""
        module, _ = _eligible_module(tmp_path)
        optimizer = MagicMock()
        optimizer.optimizer = optimizer
        optimizer.set_max_grad_norm = MagicMock()

        module.configure_gradient_clipping(optimizer, gradient_clip_val=0.25, gradient_clip_algorithm="norm")

        optimizer.set_max_grad_norm.assert_called_once_with(0.25)
