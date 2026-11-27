# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for build_trainer() precision resolution: AMP, ``amp_dtype``, FP8 and float32 matmul precision."""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from pytorch_lightning.accelerators import XLAAccelerator
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from rfdetr.training import build_trainer
from rfdetr.utilities.imports import _IS_TRANSFORMER_ENGINE_INSTALLED
from tests._markers import _requires_torch_xla
from tests.training.trainer.helpers import _mc, _tc

_requires_transformer_engine = pytest.mark.skipif(
    not _IS_TRANSFORMER_ENGINE_INSTALLED, reason="requires the 'cuda' extra (transformer-engine)"
)


@pytest.fixture
def bf16_cuda_host(mocker: MockerFixture) -> None:
    """Mock a CUDA host whose GPU natively supports bfloat16 (Ampere).

    Examples:
        >>> bf16_cuda_host(mocker)  # doctest: +SKIP
        # Needs pytest's mocker fixture; cannot run standalone.
    """
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
    mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))


@pytest.fixture
def resolved_precision(
    mocker: MockerFixture, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
) -> Callable[..., str]:
    """Return a resolver mapping a mocked device capability and ``amp_dtype`` to the Lightning precision string.

    The capability probes are mocked so the expected precision is a property of the config under test rather than of
    whichever machine runs the suite -- an unmocked call resolves to ``"16-mixed"`` on an MPS host and ``"32-true"`` on
    a CPU-only one. A callable is returned because the mocks are installed when the test chooses its hardware.

    Examples:
        >>> resolved_precision(mocker, captured_trainer_kwargs, tmp_path)  # doctest: +SKIP
        # Needs pytest's mocker/tmp_path fixtures; cannot run standalone.
    """

    def _resolve(
        *,
        cuda: bool,
        bf16: bool = False,
        bf16_emulated: bool = False,
        mps: bool = False,
        amp_dtype: str | None = "auto",
        amp: bool = True,
    ) -> str:
        """Resolve the precision for a mocked device and return the string the ``Trainer`` receives.

        Args:
            cuda: Value returned by the mocked ``torch.cuda.is_available``.
            bf16: Whether the mocked CUDA device supports bfloat16 natively.
            bf16_emulated: Whether ``torch.cuda.is_bf16_supported`` also reports bfloat16 through emulation, as PyTorch
                does on pre-Ampere GPUs such as the T4 (``including_emulation=True`` is its default).
            mps: Value returned by the mocked ``torch.backends.mps.is_available``.
            amp_dtype: The ``TrainConfig.amp_dtype`` value under test.
            amp: The deprecated ``ModelConfig.amp`` value under test.

        Returns:
            The ``precision`` string passed to the (mocked) ``Trainer``.
        """

        def _is_bf16_supported(including_emulation: bool = True) -> bool:
            return bf16 or (bf16_emulated and including_emulation)

        capability = (8, 0) if bf16 else (7, 5) if bf16_emulated else (6, 1)
        mocker.patch("torch.cuda.is_available", return_value=cuda)
        mocker.patch("torch.cuda.is_bf16_supported", side_effect=_is_bf16_supported)
        mocker.patch("torch.cuda.get_device_capability", return_value=capability)
        mocker.patch("torch.backends.mps.is_available", return_value=mps)
        mocker.patch("torch.version.hip", None)
        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype=amp_dtype), _mc(amp=amp))
        return captured_trainer_kwargs["precision"]

    return _resolve


class TestBuildTrainerPrecision:
    """build_trainer() must resolve training precision from model_config.amp + device caps."""

    def test_amp_false_gives_32_true(self, tmp_path):
        """Amp=False always produces '32-true' regardless of device."""
        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False))
        assert trainer.precision == "32-true"

    def test_amp_true_cpu_gives_32_true(self, mocker: MockerFixture, tmp_path):
        """Amp=True on CPU (no CUDA, no MPS) must fall back to '32-true'."""
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)

        trainer = build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True))

        assert trainer.precision == "32-true"

    def test_amp_true_explicit_cpu_accelerator_gives_32_true_even_with_mps(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Amp=True with explicit accelerator='cpu' must produce '32-true' even when MPS is present.

        bf16 autocast on macOS CPU (Apple Silicon) is ~13x slower than fp32 — no hardware support for bfloat16 in CPU
        kernels causes software emulation.  When the caller explicitly opts into CPU (e.g. for test isolation), mixed
        precision must not be used.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator="cpu")

        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.parametrize(("accelerator", "expected_precision"), [("xla", "32-true"), ("tpu", "bf16-true")])
    def test_xla_accelerator_uses_xla_precision_plugin_not_precision_string(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        bf16_cuda_host: None,
        tmp_path,
        accelerator,
        expected_precision,
    ):
        """An XLA accelerator uses XLAPrecision instead of a precision string.

        XLAStrategy's precision_plugin setter only accepts the XLAPrecision plugin and raises TypeError for standard
        precision strings like 'bf16-mixed'. ``XLAPrecision.__init__`` itself raises ``ModuleNotFoundError`` unless
        torch_xla is importable (see ``lightning_fabric.accelerators.xla._XLA_AVAILABLE``), so the class is patched
        here to keep this test backend-neutral -- the behaviour under test is build_trainer's accelerator dispatch,
        not PTL's own package-availability guard.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator=accelerator)

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with(expected_precision)
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.parametrize("amp_dtype", ["bf16", "auto"])
    def test_bf16_or_auto_on_tpu_uses_bf16_true(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
        amp_dtype: str,
    ) -> None:
        """Explicit BF16 and the default auto mode select TPU BF16 true precision.

        'auto' matters here as much as the explicit case: it is the amp_dtype every caller gets by just passing
        ``amp=True`` -- the exact recipe issue #1058 itself documents for TPU training -- and without CUDA/MPS on the
        host it used to fall through to the CPU-only "32-true" default, silently training in FP32 on TPU by default.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.backends.mps.is_available", return_value=False)

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype=amp_dtype), _mc(amp=True), accelerator="tpu")

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with("bf16-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    def test_auto_on_available_tpu_uses_bf16_true(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Automatic accelerator selection retains TPU BF16 true precision."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=False)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=True), accelerator="auto")

        mock_xla_precision.assert_called_once_with("bf16-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    def test_explicit_xla_bf16_stays_fp32_without_backend_evidence(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        mocker: MockerFixture,
        tmp_path: Path,
    ) -> None:
        """Explicit XLA never assumes GPU PJRT can execute BF16 true precision."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="bf16"), _mc(amp=True), accelerator="xla")

        mock_xla_precision.assert_called_once_with("32-true")
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.parametrize("amp", [True, False])
    def test_xla_accelerator_preserves_plugins_and_rejects_precision_kwargs(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mock_xla_precision: MagicMock,
        bf16_cuda_host: None,
        tmp_path,
        amp,
    ):
        """XLA appends its precision plugin after the caller's and ignores an incompatible ``precision`` kwarg."""
        caller_plugin = object()

        build_trainer(
            _tc(tmp_path, use_ema=False), _mc(amp=amp), accelerator="xla", plugins=[caller_plugin], precision="16-mixed"
        )

        assert "precision" not in captured_trainer_kwargs
        mock_xla_precision.assert_called_once_with("32-true")
        assert captured_trainer_kwargs["plugins"] == [caller_plugin, mock_xla_precision.return_value]

    def test_non_xla_accelerator_sets_no_plugins_key(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """A non-XLA accelerator does not add a 'plugins' key -- only the XLA path does."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="cpu")

        assert "plugins" not in captured_trainer_kwargs
        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.xla
    @_requires_torch_xla
    def test_tpu_accelerator_refuses_to_launch_off_real_tpu(self, tmp_path) -> None:
        """Resolves plan Sec 1.3 caveat #1: PTL's 'tpu' accelerator needs real TPU chips, not just torch_xla+PJRT.

        ``XLAAccelerator.is_available()`` returns ``False`` under ``PJRT_DEVICE=CPU`` (no TPU silicon), so
        ``Trainer.__init__`` raises ``MisconfigurationException`` naming ``XLAAccelerator`` as unavailable and listing
        ``cpu`` as the only available accelerator (confirmed against the live message from ``ci-tests-xla.yml``'s CPU-
        PJRT run, not just source inspection). This confirms the full ``model.train(accelerator="tpu")`` entry point is
        not launchable under the T1 (CPU-PJRT) CI lane -- only the device-gated unit tests (Tasks 1.1/1.3/1.6/1.7/1.8,
        which move tensors to ``xm.xla_device()`` directly) validate Phase 1 correctness there.
        """
        if XLAAccelerator.is_available():
            pytest.skip(
                "the refusal this pins holds only while ``XLAAccelerator.is_available()`` is False, so the skip is "
                "gated on that same predicate rather than on the configured backend: ``xr.device_type()`` reports "
                "``PJRT_DEVICE`` and would also skip on a host that sets ``PJRT_DEVICE=TPU`` without chips, where "
                "``auto_device_count()`` is 0, the Trainer still raises, and the assertion is still meaningful. "
                "Gating on availability keeps NEURON in the refusal path for the same reason."
            )

        with pytest.raises(MisconfigurationException, match="XLAAccelerator"):
            build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu")

    def test_amp_true_ddp_notebook_probes_bf16_normally(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """ddp_notebook uses standard precision probing (spawn makes CUDA init safe).

        With spawn-based DDP, child processes start fresh — CUDA init in the parent does not propagate.  So
        ``is_bf16_supported()`` is safe to call and pre-Ampere GPUs correctly get ``16-mixed`` instead of the slower
        bf16 emulation path.  Simulates a pre-Ampere GPU as real PyTorch reports it: CUDA available, bf16 supported only
        through emulation.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.is_bf16_supported", side_effect=lambda including_emulation=True: including_emulation)
        mocker.patch("torch.cuda.get_device_capability", return_value=(7, 5))

        build_trainer(_tc(tmp_path, use_ema=False, strategy="ddp_notebook"), _mc(amp=True))

        assert captured_trainer_kwargs["precision"] == "16-mixed"


class TestBuildTrainerAmpDtype:
    """``TrainConfig.amp_dtype`` (a ``train()`` kwarg) lets callers pin the AMP autocast dtype (fp16 vs bf16) — #1132.

    Precision is resolved inside ``build_trainer``; these tests mock the CUDA/MPS capability probes and assert the
    Lightning precision string captured at ``Trainer`` construction time.
    """

    def test_amp_dtype_is_a_train_kwarg_not_dropped(self, tmp_path):
        """amp_dtype is a real TrainConfig field (reachable via train(**kwargs)), not silently dropped."""
        assert _tc(tmp_path, amp_dtype="fp16").amp_dtype == "fp16"

    @pytest.mark.parametrize(
        ("cuda", "bf16", "mps", "amp_dtype", "expected"),
        [
            (True, True, False, "auto", "bf16-mixed"),
            (True, True, False, "fp16", "16-mixed"),
            (True, True, False, "bf16", "bf16-mixed"),
            (True, True, False, "fp8", "transformer-engine"),
            (True, False, False, "auto", "16-mixed"),
            (False, False, True, "fp16", "16-mixed"),
        ],
    )
    def test_resolved_precision(self, resolved_precision, cuda, bf16, mps, amp_dtype, expected):
        """amp_dtype + hardware caps resolve to the correct Lightning precision string."""
        assert resolved_precision(cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype) == expected

    @pytest.mark.parametrize(
        ("cuda", "bf16", "mps", "amp_dtype", "warn_match"),
        [(True, False, False, "bf16", "bf16"), (False, False, True, "bf16", "MPS")],
    )
    def test_resolved_precision_warns(self, resolved_precision, cuda, bf16, mps, amp_dtype, warn_match):
        """amp_dtype falls back to '16-mixed' and emits a UserWarning when hardware cannot satisfy the request."""
        with pytest.warns(UserWarning, match=warn_match):
            precision = resolved_precision(cuda=cuda, bf16=bf16, mps=mps, amp_dtype=amp_dtype)
        assert precision == "16-mixed"

    def test_auto_uses_fp16_on_gpu_with_emulated_bf16(self, resolved_precision) -> None:
        """``amp_dtype="auto"`` on a GPU whose bfloat16 is only emulated (T4, V100) trains in fp16.

        ``torch.cuda.is_bf16_supported()`` counts emulation, so it returns ``True`` on such a GPU and ``"auto"`` used to
        pick ``bf16-mixed``. On a Colab T4 that was about 2x slower per RF-DETR Nano training step than ``16-mixed``.
        """
        precision = resolved_precision(cuda=True, bf16_emulated=True, amp_dtype="auto")
        assert precision == "16-mixed", f"amp_dtype='auto' on an emulated-bf16 GPU resolved to {precision!r}"

    def test_explicit_bf16_on_gpu_with_emulated_bf16_is_kept_with_a_warning(self, resolved_precision) -> None:
        """An explicit ``amp_dtype="bf16"`` is honoured on an emulated-bf16 GPU, with a warning naming faster
        options."""
        with pytest.warns(UserWarning, match="emulation"):
            precision = resolved_precision(cuda=True, bf16_emulated=True, amp_dtype="bf16")
        assert precision == "bf16-mixed", f"explicit bf16 on an emulated-bf16 GPU resolved to {precision!r}"

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            pytest.param("1,", "bf16-mixed", id="a100-at-index-1"),
            pytest.param("0,", "16-mixed", id="t4-at-index-0"),
            pytest.param(2, "16-mixed", id="both-gpus"),
            pytest.param([1], "bf16-mixed", id="a100-by-index-list"),
        ],
    )
    def test_auto_checks_the_gpus_the_run_trains_on(
        self,
        captured_trainer_kwargs: dict[str, Any],
        mocker: MockerFixture,
        tmp_path: Path,
        devices: int | str | list[int],
        expected: str,
    ) -> None:
        """On a host with a T4 at index 0 and an A100 at index 1, ``"auto"`` follows the GPUs the run trains on.

        Selecting a GPU by index (``devices="1,"``, or the ``[1]`` that ``RFDETR.train(device="cuda:1")`` forwards) does
        not change the current device, so checking only the current device would pick fp16 for the A100 run here, and
        bf16 for a T4 run on a host whose index 0 is an A100. A run across both GPUs uses fp16.
        """
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda device=None: capabilities[device or 0])
        mocker.patch("torch.version.hip", None)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(), devices=devices)

        assert captured_trainer_kwargs["precision"] == expected, (
            f"devices={devices!r} resolved to {captured_trainer_kwargs['precision']!r}"
        )

    def test_explicit_bf16_checks_the_gpus_the_run_trains_on(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """An explicit ``amp_dtype="bf16"`` is granted on the strength of the training GPU, not the current device.

        The host here has a pre-Ampere GPU at index 0 and an A100 at index 1 on a CUDA build without bfloat16 emulation,
        so ``torch.cuda.is_bf16_supported()`` — which only ever looks at the current device — says no. A run pinned to
        the A100 with ``devices=[1]`` must still get ``bf16-mixed``, and no emulation warning.
        """
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=False)
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda device=None: capabilities[device or 0])
        mocker.patch("torch.version.hip", None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="bf16"), _mc(), devices=[1])

        assert captured_trainer_kwargs["precision"] == "bf16-mixed"
        assert not [warning for warning in caught if "bf16" in str(warning.message)]

    def test_explicit_amp_dtype_overrides_deprecated_amp_false(self, resolved_precision):
        """An explicit amp_dtype wins over the deprecated amp flag: the stale amp=False is ignored.

        Mocked onto a bf16-capable CUDA device, so the fp16 request is distinguishable both from the '32-true' the old
        amp=False precedence produced and from the 'bf16-mixed' the hardware would otherwise select.
        """
        assert resolved_precision(cuda=True, bf16=True, amp_dtype="fp16", amp=False) == "16-mixed"

    def test_deprecated_amp_false_applies_when_amp_dtype_is_default(self, resolved_precision):
        """Amp=False still disables AMP while amp_dtype is left at its default, and warns.

        The mocked device is bf16-capable CUDA, which would resolve to 'bf16-mixed' on its own — so '32-true' can only
        come from the legacy toggle still being honored.
        """
        with pytest.warns(FutureWarning, match="ModelConfig.amp is deprecated"):
            resolved = resolved_precision(cuda=True, bf16=True, amp_dtype="auto", amp=False)
        assert resolved == "32-true"

    def test_amp_dtype_none_disables_amp(self, resolved_precision):
        """amp_dtype=None is the replacement for the deprecated amp=False and needs no model flag.

        Mocked onto bf16-capable CUDA with amp left at its default True, so '32-true' is attributable to amp_dtype=None
        alone rather than to absent hardware.
        """
        assert resolved_precision(cuda=True, bf16=True, amp_dtype=None, amp=True) == "32-true"

    def test_cpu_accelerator_ignores_amp_dtype(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Explicit accelerator='cpu' yields '32-true' regardless of amp_dtype."""
        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp16"), _mc(amp=True), accelerator="cpu")
        assert captured_trainer_kwargs["precision"] == "32-true"

    @pytest.mark.parametrize("amp", [True, False])
    def test_fp8_rejects_non_cuda_accelerator(self, tmp_path, amp):
        """FP8 must fail clearly on a non-CUDA accelerator, never fall back or be disabled by the deprecated AMP
        flag."""
        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=amp), accelerator="cpu")

    @pytest.mark.parametrize("accelerator", ["cpu", "mps", "xla", "tpu"])
    def test_fp8_rejects_non_cuda_with_cuda_visible(
        self, tmp_path: Path, accelerator: str, mocker: MockerFixture, mock_xla_precision: MagicMock
    ) -> None:
        """Reject FP8 before an explicitly selected XLA backend loads its precision plugin."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), accelerator=accelerator)
        mock_xla_precision.assert_not_called()

    def test_fp8_rejects_auto_resolving_to_xla(self, tmp_path: Path, mocker: MockerFixture) -> None:
        """Lightning selects XLA before CUDA for auto; FP8 must honor that choice."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="FP8 training requires an NVIDIA CUDA GPU"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), accelerator="auto")

    def test_fp8_rejects_deepspeed_strategy(self, tmp_path, mocker: MockerFixture):
        """Lightning cannot combine its Transformer Engine precision plugin with DeepSpeed precision."""
        mocker.patch("torch.cuda.is_available", return_value=True)

        with pytest.raises(ValueError, match="amp_dtype='fp8'.*DeepSpeed"):
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8", strategy="deepspeed_stage_2"), _mc(amp=True))

    @pytest.mark.parametrize(
        ("capabilities", "device_names", "devices", "expected_label"),
        [
            pytest.param({0: (8, 0)}, "NVIDIA A100", None, "cuda:0 (NVIDIA A100)", id="single-pre-ada-gpu"),
            pytest.param({0: (9, 0), 1: (7, 5)}, "NVIDIA T4", 2, "cuda:1 (NVIDIA T4)", id="hopper-plus-t4-uses-both"),
        ],
    )
    def test_fp8_rejects_unsupported_training_device(
        self, tmp_path, mocker: MockerFixture, capabilities, device_names, devices, expected_label
    ):
        """FP8 must fail clearly if any training GPU is CUDA-visible but below Transformer Engine's compute capability.

        ``torch.cuda.is_available()`` alone does not establish FP8/Transformer Engine hardware support: an A100 or T4 is
        CUDA-visible but below the compute capability (8.9, Ada) Transformer Engine requires, and a multi-GPU run must
        validate every device it trains on, not just the first.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda index: capabilities[index])
        mocker.patch("torch.cuda.get_device_name", return_value=device_names)
        device_kwargs = {} if devices is None else {"devices": devices}

        with pytest.raises(ValueError) as exc_info:
            build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), **device_kwargs)

        assert "compute capability >= 8.9" in str(exc_info.value)
        assert expected_label in str(exc_info.value)

    @pytest.mark.parametrize(
        ("capabilities", "devices"),
        [
            pytest.param({0: (8, 9)}, None, id="single-ada-gpu"),
            pytest.param({0: (9, 0), 1: (7, 5)}, [0], id="hopper-pinned-on-a-mixed-host"),
        ],
    )
    def test_fp8_accepts_capable_training_devices(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, mocker: MockerFixture, capabilities, devices
    ):
        """An Ada-or-newer training device resolves to the Transformer Engine precision string, not an error.

        A run pinned to the fp8-capable GPU is not rejected for an older GPU it never touches: with a Hopper at index 0
        and a T4 at index 1, ``devices=[0]`` -- the shape ``RFDETR.train(device="cuda:0")`` forwards -- trains only on
        hardware Transformer Engine supports. The real ``Trainer`` is mocked, so only the capability gate is exercised,
        not Transformer Engine's plugin construction, which needs real hardware.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=len(capabilities))
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda index: capabilities[index])
        device_kwargs = {} if devices is None else {"devices": devices}

        build_trainer(_tc(tmp_path, use_ema=False, amp_dtype="fp8"), _mc(amp=True), **device_kwargs)

        assert captured_trainer_kwargs["precision"] == "transformer-engine"

    @pytest.mark.parametrize("bad_value", ["float8", 42, True])
    def test_invalid_amp_dtype_falls_back_to_auto_with_warning(self, tmp_path, bad_value):
        """An unrecognised or wrong-typed amp_dtype falls back to 'auto' with a warning rather than raising."""
        with pytest.warns(UserWarning, match="amp_dtype"):
            tc = _tc(tmp_path, amp_dtype=bad_value)
        assert tc.amp_dtype == "auto"


def _cuda_supports_fp8() -> bool:
    """Return whether device 0 meets Transformer Engine's FP8 minimum compute capability.

    Examples:
        >>> isinstance(_cuda_supports_fp8(), bool)
        True
    """
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 9)


#: Skip a test node that runs FP8 training, which needs compute capability 8.9 or newer.
_requires_fp8_cuda = pytest.mark.skipif(
    not _cuda_supports_fp8(),
    reason="FP8 requires a Transformer Engine-supported GPU (Ada, Hopper, or newer; compute capability >= 8.9)",
)


class TestBuildTrainerFP8Smoke:
    """Real CUDA + Transformer Engine integration smoke test for FP8 training.

    ``TestBuildTrainerAmpDtype.test_resolved_precision`` (its ``fp8`` case) only mocks ``torch.cuda.is_available`` and
    ``torch.cuda.is_bf16_supported`` and captures the precision string passed to the (also mocked) ``Trainer`` — it
    never imports or instantiates Transformer Engine's PyTorch extension. A missing/mismatched ``cuda`` extra or a
    plugin setup failure would therefore pass every other test in this module and only surface on the first real
    CUDA run. This test builds a real trainer and Lightning module with an actual ``nn.Linear`` layer and executes
    one training step under FP8, skipped everywhere the hardware or the ``transformer-engine`` package is absent.
    """

    @pytest.mark.gpu
    @_requires_transformer_engine
    @_requires_fp8_cuda
    def test_fp8_smoke_builds_and_steps(self, base_model_config, base_train_config, mocker: MockerFixture):
        """A real CUDA + Transformer Engine environment must build the plugin and run one FP8 training step.

        Guards against the gap the mocked ``fp8-cuda`` precision-resolution test cannot close: the actual
        ``transformer_engine.pytorch`` extension being unimportable, unbuilt, or incompatible with the installed
        CUDA/PyTorch stack only fails here, on real hardware, never in the CPU-only unit tests above.
        """
        from rfdetr.training.module_data import RFDETRDataModule
        from rfdetr.training.module_model import RFDETRModelModule
        from tests.training.helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts

        class _LinearModel(torch.nn.Module):
            """Minimal real model with an ``nn.Linear`` layer for Transformer Engine to convert."""

            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3 * 32 * 32, 64)

            def forward(self, samples, targets=None):
                images = samples.tensors if hasattr(samples, "tensors") else samples
                return {"dummy": self.linear(images.flatten(1)).sum()}

            def update_drop_path(self, *args, **kwargs) -> None:
                pass

            def update_dropout(self, *args, **kwargs) -> None:
                pass

            def reinitialize_detection_head(self, *args, **kwargs) -> None:
                pass

        mc = base_model_config(device="cuda", amp=True)
        tc = base_train_config(use_ema=False, run_test=False, amp_dtype="fp8", batch_size=2)

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_LinearModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4))
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        )

        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cuda",
            fast_dev_run=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        assert trainer.precision == "transformer-engine"
        trainer.fit(module, datamodule=datamodule)


class TestFloat32MatmulPrecision:
    """build_trainer must enable TF32 matmul on every entry path (CLI included)."""

    @pytest.fixture(autouse=True)
    def _restore_matmul_precision(self):
        """Snapshot and restore the process-global matmul precision around each test."""
        previous = torch.get_float32_matmul_precision()
        yield
        torch.set_float32_matmul_precision(previous)

    def test_build_trainer_sets_matmul_precision_high(self, tmp_path):
        """build_trainer upgrades the default 'highest' to 'high' (TF32 on Ampere+)."""
        torch.set_float32_matmul_precision("highest")

        build_trainer(_tc(tmp_path), _mc(), accelerator="cpu")
        assert torch.get_float32_matmul_precision() == "high"
