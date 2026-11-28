# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for build_trainer() device and strategy handling: DDP, sharded strategies, XLA and the EMA guards they need."""

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from pytorch_lightning.strategies import DDPStrategy, ModelParallelStrategy

from rfdetr.training import build_trainer
from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from rfdetr.training.trainer import (
    _accelerator_resolves_to_xla,
    _requests_multiple_devices,
    _xla_resolves_to_single_device,
)
from tests._markers import _requires_torch_xla
from tests.training.trainer.helpers import _configs, _kp_tc, _mc, _tc


class TestBuildTrainerEMAShardingGuard:
    """EMA must be disabled and a UserWarning emitted for sharded strategies.

    PTL validates strategy+accelerator compatibility at Trainer construction time, so these tests mock Trainer to
    capture the callback list without triggering platform-specific validation.
    """

    @pytest.mark.parametrize("strategy", ["fsdp", "deepspeed", "deepspeed_stage_2"])
    def test_ema_disabled_with_warning_for_sharded_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, strategy
    ):
        """The EMA callback is absent and a UserWarning says so when a sharded strategy is requested."""
        with pytest.warns(UserWarning, match="EMA disabled"):
            build_trainer(_tc(tmp_path, use_ema=True, strategy=strategy), _mc())

        assert not any(isinstance(cb, RFDETREMACallback) for cb in captured_trainer_kwargs["callbacks"])


class TestBuildTrainerEMAXLAGuard:
    """Multi-device XLA training must disable EMA and its checkpoint/evaluation bookkeeping."""

    @pytest.mark.parametrize(
        ("devices", "num_nodes"),
        [
            pytest.param(4, 1, id="four_devices_one_node"),
            pytest.param(1, 2, id="one_device_per_host_two_hosts"),
        ],
    )
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_device_xla_disables_ema_and_uses_regular_checkpoint_track(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, devices: int, num_nodes: int
    ) -> None:
        """Multi-device XLA, one device per host across nodes included, must omit EMA callbacks and metrics.

        The regular-model checkpoint/evaluation path stays in place.
        """
        with pytest.warns(UserWarning, match="EMA disabled on multi-device XLA"):
            build_trainer(
                _tc(tmp_path, use_ema=True, eval_base_model=False, devices=devices, num_nodes=num_nodes),
                _mc(),
                accelerator="xla",
            )

        callbacks = captured_trainer_kwargs["callbacks"]
        assert not any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is None
        assert best_callback._evaluates_base_model is True

    @pytest.mark.parametrize("accelerator", ["xla", "tpu"])
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_single_device_xla_keeps_ema_enabled(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, accelerator: str
    ) -> None:
        """One-device XLA keeps EMA after the shipped callback completed the one-chip validation runs.

        ``"tpu"`` is the accelerator string ``RFDETR.train(device="xla")`` forwards to ``build_trainer`` (see
        ``test_device_xla_absorbed_as_accelerator_tpu``); the guard must treat it exactly like ``"xla"``.
        """
        build_trainer(_tc(tmp_path, use_ema=True, eval_base_model=False, devices=1), _mc(), accelerator=accelerator)

        assert captured_trainer_kwargs["accelerator"] == accelerator
        assert "precision" not in captured_trainer_kwargs
        callbacks = captured_trainer_kwargs["callbacks"]
        assert any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is not None
        assert best_callback._evaluates_base_model is False

    @pytest.mark.parametrize(
        ("tc_kwargs", "build_kwargs"),
        [
            pytest.param({"devices": 1}, {}, id="single-device-keeps-ema"),
            pytest.param({"devices": 4}, {"include_training_callbacks": False}, id="evaluation-only-builds-no-ema"),
        ],
    )
    @pytest.mark.usefixtures("captured_trainer_kwargs", "mock_xla_precision")
    def test_ema_kept_or_not_built_does_not_warn(self, tmp_path: Path, tc_kwargs, build_kwargs) -> None:
        """The EMA-disabled warning fires neither when EMA is kept nor when evaluation-only trainers never build it.

        The evaluation-only case uses the multi-device configuration that *does* warn when training callbacks are built,
        so the assertion is about ``include_training_callbacks=False`` rather than a device count that would stay silent
        either way.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_trainer(_tc(tmp_path, use_ema=True, **tc_kwargs), _mc(), accelerator="xla", **build_kwargs)

        assert not any("EMA disabled" in str(warning.message) for warning in caught)

    @pytest.mark.parametrize("auto_device_count", [1, 4])
    @pytest.mark.usefixtures("mock_xla_precision")
    def test_auto_devices_on_xla_follows_the_resolved_chip_count(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path: Path, auto_device_count: int
    ) -> None:
        """``devices='auto'`` must resolve through the XLA chip count, not CUDA's auto branch."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.auto_device_count", return_value=auto_device_count)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build_trainer(_tc(tmp_path, use_ema=True, eval_base_model=False, devices="auto"), _mc(), accelerator="xla")

        callbacks = captured_trainer_kwargs["callbacks"]
        has_ema = any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        assert has_ema is (auto_device_count == 1)

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_xla_device_index_list_keeps_ema_disabled_instead_of_raising(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        """A device-index list, as ``RFDETR.train(device='xla:N')`` forwards it, must not raise in the EMA guard.

        A distributed ``strategy`` skips the earlier ``_requests_multiple_devices`` call, so this is the first place the
        list is inspected; the guard has to answer "not provably one device" like it did before.
        """
        with pytest.warns(UserWarning, match="EMA disabled"):
            build_trainer(
                _tc(tmp_path, use_ema=True, eval_base_model=False, strategy="ddp"),
                _mc(),
                accelerator="tpu",
                devices=[0],
            )

        callbacks = captured_trainer_kwargs["callbacks"]
        assert not any(isinstance(callback, RFDETREMACallback) for callback in callbacks)
        best_callback = next(callback for callback in callbacks if isinstance(callback, BestModelCallback))
        assert best_callback._monitor_ema is None
        assert best_callback._evaluates_base_model is True


class TestRequestsMultipleDevices:
    """``_requests_multiple_devices`` gates the distributed branch, so every ``devices`` form must reach an answer."""

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            pytest.param([1], False, id="single-index-list"),
            pytest.param([0, 1], True, id="two-index-list"),
            pytest.param((0, 1), True, id="two-index-tuple"),
            pytest.param([], False, id="empty-list"),
        ],
    )
    def test_an_index_sequence_is_measured_by_its_length(self, devices: Sequence[int], expected: bool) -> None:
        """A sequence of device indices is counted instead of being parsed as a string.

        ``devices=[1]`` is what ``RFDETR.train(device="cuda:1")`` forwards; it used to reach the string branch's
        ``.strip()`` and raise ``AttributeError: 'list' object has no attribute 'strip'`` before training started.
        """
        assert _requests_multiple_devices(devices) is expected


class TestXlaResolvesToSingleDevice:
    """``_xla_resolves_to_single_device`` decides the EMA guard, so each ``int | str`` device form is covered."""

    @pytest.fixture
    def forbidden_auto_device_count(self, mocker: MockerFixture) -> MagicMock:
        """Make any query of the XLA runtime's chip count fail, so a test proves the answer needed no runtime.

        Examples:
            >>> forbidden_auto_device_count(mocker)  # doctest: +SKIP
            # Needs pytest's mocker fixture; cannot run standalone.
        """
        return mocker.patch(
            "pytorch_lightning.accelerators.XLAAccelerator.auto_device_count",
            side_effect=AssertionError("auto_device_count must not be consulted"),
        )

    @pytest.mark.usefixtures("forbidden_auto_device_count")
    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            (1, True),
            (2, False),
            (4, False),
            ("1", True),
            ("4", False),
            ("0,1", False),
            ("", False),
            ("all", False),
            pytest.param([0], False, id="one_index_list"),
            pytest.param([0, 1], False, id="two_index_list"),
            pytest.param((0,), False, id="one_index_tuple"),
            pytest.param([], False, id="empty_list"),
        ],
    )
    def test_explicit_device_forms_do_not_consult_the_accelerator(
        self, devices: int | str | list[int] | tuple[int, ...], expected: bool
    ) -> None:
        """An explicitly named device count resolves without querying the XLA runtime.

        A device-index sequence, as ``train(device='xla:N')`` forwards it, answers ``False``: only the ``int | str``
        count forms are validated on one chip, so a sequence keeps the disabled path instead of raising
        ``AttributeError`` on ``.strip()``.
        """
        assert _xla_resolves_to_single_device(devices) is expected

    @pytest.mark.parametrize("devices", [-1, "auto", "-1", "AUTO"])
    @pytest.mark.parametrize("auto_device_count", [1, 8])
    def test_auto_forms_resolve_through_the_xla_chip_count(
        self, mocker: MockerFixture, devices: int | str, auto_device_count: int
    ) -> None:
        """``auto``/``-1`` must count TPU chips; CUDA's auto branch reports none for XLA."""
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.auto_device_count", return_value=auto_device_count)

        assert _xla_resolves_to_single_device(devices) is (auto_device_count == 1)

    @pytest.mark.usefixtures("forbidden_auto_device_count")
    @pytest.mark.parametrize("devices", [1, "1", -1, "auto"])
    @pytest.mark.parametrize("num_nodes", [2, 4])
    def test_multiple_nodes_are_never_a_single_device(self, devices: int | str, num_nodes: int) -> None:
        """One device per host across several hosts is a multi-replica run, whatever ``devices`` says."""
        assert _xla_resolves_to_single_device(devices, num_nodes) is False


class TestBuildTrainerDDPFields:
    """build_trainer() must thread devices/num_nodes/strategy from TrainConfig to Trainer."""

    @pytest.mark.parametrize(
        ("extra_kwargs", "captured_key", "expected"),
        [
            pytest.param({"devices": 4}, "devices", 4, id="devices-forwarded-from-train-config"),
            pytest.param({"num_nodes": 2}, "num_nodes", 2, id="num-nodes-forwarded-from-train-config"),
            pytest.param({"strategy": "auto"}, "strategy", "auto", id="strategy-forwarded-from-train-config"),
            pytest.param({}, "devices", 1, id="default-devices-is-1-single-gpu-default"),
            pytest.param({}, "num_nodes", 1, id="default-num-nodes-is-1"),
        ],
    )
    def test_ddp_fields_threaded_from_train_config(
        self,
        captured_trainer_kwargs: dict[str, Any],
        tmp_path,
        extra_kwargs: dict[str, Any],
        captured_key: str,
        expected: object,
    ):
        """TrainConfig.devices/num_nodes/strategy are forwarded to Trainer(...) unchanged, defaults included."""
        build_trainer(_tc(tmp_path, use_ema=False, **extra_kwargs), _mc())

        assert captured_trainer_kwargs[captured_key] == expected

    def test_devices_string_accepted(self, tmp_path):
        """TrainConfig.devices accepts a string value such as 'auto'."""
        assert _tc(tmp_path, use_ema=False, devices="auto").devices == "auto"


class TestBuildTrainerDDPStrategy:
    """build_trainer() builds ``DDPStrategy(find_unused_parameters=True)`` for distributed runs and rejects sharded
    ones."""

    @pytest.mark.parametrize(
        ("keypoint", "segmentation", "tc_kwargs"),
        [
            pytest.param(True, False, {"strategy": "ddp"}, id="keypoint-ddp"),
            pytest.param(True, False, {"strategy": "auto", "devices": 2}, id="keypoint-auto-with-multiple-devices"),
            pytest.param(True, False, {"strategy": "ddp", "num_nodes": 2}, id="keypoint-ddp-across-nodes"),
            pytest.param(False, False, {"strategy": "auto", "devices": 2}, id="detection-auto-with-multiple-devices"),
            pytest.param(False, True, {"strategy": "ddp"}, id="segmentation-ddp-sparse-forward"),
            pytest.param(False, False, {"strategy": "ddp"}, id="detection-ddp-regression-1093"),
            pytest.param(False, True, {"strategy": "ddp_spawn"}, id="segmentation-ddp-spawn"),
        ],
    )
    def test_distributed_strategy_enables_find_unused_parameters(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, keypoint, segmentation, tc_kwargs
    ):
        """Every distributed request becomes DDPStrategy(find_unused_parameters=True), and devices/num_nodes reach
        Trainer.

        Detection can leave parameters unused under DDP (two-stage group_detr ModuleLists, conditional aux_loss branches
        -- regression test for https://github.com/roboflow/rf-detr/issues/1093), segmentation's sparse_forward() is
        another source, and ``strategy='auto'`` with more than one device would otherwise let Lightning pick a
        distributed strategy without RF-DETR's unused-parameter guard. ``ddp_spawn`` must not drop the flag the
        interactive-spawn path already sets.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, segmentation=segmentation, **tc_kwargs)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True
        assert captured_trainer_kwargs["devices"] == tc.devices
        assert captured_trainer_kwargs["num_nodes"] == tc.num_nodes

    @pytest.mark.parametrize("keypoint", [True, False])
    @pytest.mark.parametrize("strategy_name", ["ddp_notebook", "ddp_spawn"])
    def test_spawn_strategies_use_interactive_spawn(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path, keypoint, strategy_name
    ):
        """ddp_notebook and ddp_spawn must be replaced with interactive spawn DDPStrategy.

        Fork-based DDP inherits the parent's OpenMP thread pool which is invalid after fork, causing SIGABRT in the
        autograd engine. ddp_spawn is blocked by PTL in notebooks without the override.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, strategy=strategy_name)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._start_method == "spawn"
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    def test_ddp_notebook_raises_clear_error_when_private_launcher_is_missing(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Missing private PTL launcher should raise a targeted compatibility error."""
        mocker.patch("rfdetr.training.trainer._InteractiveSpawnLauncher", None)
        build_trainer(_tc(tmp_path, use_ema=False, strategy="ddp_notebook"), _mc(amp=True))

        strategy = captured_trainer_kwargs["strategy"]
        strategy.cluster_environment = object()
        with pytest.raises(RuntimeError, match="private API"):
            strategy._configure_launcher()

    def test_non_ddp_strategy_with_segmentation_is_unchanged(self, captured_trainer_kwargs: dict[str, Any], tmp_path):
        """Strategies other than 'ddp' must not be wrapped even when segmentation is on."""
        tc, mc = _configs(tmp_path, segmentation=True, strategy="auto")

        build_trainer(tc, mc)

        assert captured_trainer_kwargs["strategy"] == "auto"

    def test_keypoint_auto_devices_multi_gpu_builds_ddp(
        self, captured_trainer_kwargs: dict[str, Any], mocker: MockerFixture, tmp_path
    ):
        """Keypoint mode with devices='auto' resolving to multiple CUDA devices builds DDP (no error)."""
        tc, mc = _configs(tmp_path, keypoint=True, strategy="auto", devices="auto", accelerator="cuda")
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    @pytest.mark.parametrize("strategy", ["fsdp", "deepspeed"])
    def test_keypoint_sharded_strategy_string_raises_clear_error(self, tmp_path, strategy):
        """Keypoint mode rejects sharded strategy strings (FSDP/DeepSpeed) with a clear error."""
        tc, mc = _configs(tmp_path, keypoint=True, strategy=strategy)

        with pytest.raises(NotImplementedError, match="sharded distributed strategies"):
            build_trainer(tc, mc)

    def test_keypoint_model_parallel_strategy_object_raises_clear_error(self, tmp_path):
        """Keypoint mode rejects a ModelParallelStrategy (FSDP2) object, whose repr has no 'fsdp' token."""
        tc, mc = _configs(tmp_path, keypoint=True)

        # Strategy objects reach build_trainer via trainer_kwargs (TrainConfig.strategy is a string field).
        with pytest.raises(NotImplementedError, match="sharded distributed strategies"):
            build_trainer(tc, mc, strategy=ModelParallelStrategy(), devices=2)

    def test_keypoint_ddp_strategy_object_without_find_unused_parameters_raises(self, tmp_path):
        """A supplied DDPStrategy() object lacking find_unused_parameters=True is rejected for keypoint DDP."""
        tc, mc = _configs(tmp_path, keypoint=True)

        with pytest.raises(ValueError, match="find_unused_parameters=True"):
            build_trainer(tc, mc, strategy=DDPStrategy(), devices=2)

    def test_keypoint_ddp_strategy_object_with_find_unused_parameters_ok(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ):
        """A supplied DDPStrategy(find_unused_parameters=True) object passes through unchanged for keypoint DDP."""
        supplied = DDPStrategy(find_unused_parameters=True)
        tc, mc = _configs(tmp_path, keypoint=True)

        build_trainer(tc, mc, strategy=supplied, devices=2)

        assert captured_trainer_kwargs["strategy"] is supplied
        assert captured_trainer_kwargs["strategy"]._ddp_kwargs.get("find_unused_parameters") is True


class TestBuildTrainerDDPStaticGraph:
    """build_trainer() enables DDP ``static_graph`` only for the configurations the throughput measurement covered.

    Those are detection and segmentation with ``grad_accum_steps<=1``; keypoint models and ``grad_accum_steps>1`` were
    not exercised by that measurement, so ``static_graph`` and ``gradient_as_bucket_view`` stay off there.
    """

    @pytest.mark.parametrize(
        ("keypoint", "segmentation", "tc_kwargs", "expected"),
        [
            pytest.param(False, False, {"strategy": "ddp"}, True, id="ddp-detection"),
            pytest.param(False, True, {"strategy": "ddp"}, True, id="ddp-segmentation-same-always-unused-mask-token"),
            pytest.param(False, False, {"strategy": "ddp_spawn"}, True, id="ddp-spawn-via-interactive-spawn"),
            pytest.param(False, False, {"strategy": "auto", "devices": 2}, True, id="auto-with-multiple-devices"),
            pytest.param(False, False, {"strategy": "ddp", "grad_accum_steps": 2}, False, id="grad-accum-disables"),
            pytest.param(True, False, {"strategy": "ddp"}, False, id="keypoint-disables"),
        ],
    )
    def test_static_graph_follows_measured_configurations(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path, keypoint, segmentation, tc_kwargs, expected
    ) -> None:
        """static_graph and gradient_as_bucket_view are on exactly for the measured configurations.

        An empirical probe (real RFDETRSegNano forward, real/empty-target batches) found the same single always-unused
        parameter (backbone mask_token) for segmentation as for detection-only, so segmentation stays in scope.
        ``strategy='auto'`` with more than one device takes the same ``distributed_requested`` path. Gradient
        accumulation is excluded because DDP's no_sync() across several backward calls interacts with static_graph's
        iteration-counted structure-learning in a way the measurement (which used none) did not verify. Keypoint models'
        manual-optimization DDP path was not measured either.
        """
        tc, mc = _configs(tmp_path, keypoint=keypoint, segmentation=segmentation, **tc_kwargs)

        build_trainer(tc, mc)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True
        assert strategy_obj._ddp_kwargs.get("static_graph") is expected
        assert strategy_obj._ddp_kwargs.get("gradient_as_bucket_view") is expected

    def test_ddp_trainer_kwargs_accumulate_grad_batches_disables_static_graph(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path: Path
    ) -> None:
        """A caller passing accumulate_grad_batches= directly (bypassing tc.grad_accum_steps) must also disable
        static_graph.

        Regression test: this exact path (build_trainer(..., accumulate_grad_batches=2), tc.grad_accum_steps left
        at its default of 1) crashed a real 2-rank gloo DDP run with ``RuntimeError: expect_autograd_hooks_
        INTERNAL ASSERT FAILED`` in ``reducer.cpp`` when static_graph was gated on tc.grad_accum_steps alone --
        trainer_kwargs["accumulate_grad_batches"] overrides tc.grad_accum_steps later in this same function
        (see the "accumulate_grad_batches" resolution below) and must be read the same way here.
        """
        tc, mc = _configs(tmp_path, strategy="ddp")
        assert tc.grad_accum_steps <= 1

        build_trainer(tc, mc, accumulate_grad_batches=2)

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("static_graph") is False
        assert strategy_obj._ddp_kwargs.get("gradient_as_bucket_view") is False


class TestAcceleratorResolvesToXLA:
    """``_accelerator_resolves_to_xla`` must agree with Lightning's own ``accelerator="auto"`` resolution.

    Unlike ``TestMultiDeviceXLAStrategy`` below, this does not need real ``torch_xla`` or a chip: it mocks
    ``XLAAccelerator.is_available`` directly and asserts on the pure helper, not on a constructed ``Trainer``.
    """

    @pytest.mark.parametrize(
        ("accelerator", "xla_available", "expected"),
        [
            ("tpu", False, True),
            ("xla", False, True),
            ("TPU", False, True),
            ("auto", True, True),
            ("auto", False, False),
            ("cpu", True, False),
            ("gpu", True, False),
        ],
    )
    def test_matches_lightnings_own_auto_resolution(
        self, accelerator, xla_available, expected, mocker: MockerFixture
    ) -> None:
        """A caller who leaves ``accelerator="auto"`` resolves to XLA only when XLA is available, as Lightning would.

        ``"auto"`` is TrainConfig's own default, and build_trainer decides the strategy/precision arguments before
        Trainer performs that resolution itself. An explicit ``"tpu"``/``"xla"`` (case-insensitive) is always XLA, and
        an explicit ``"cpu"``/``"gpu"`` never is, even when XLA is available.
        """
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=xla_available)

        assert _accelerator_resolves_to_xla(accelerator) is expected


class TestMultiDeviceXLAStrategy:
    """`XLAAccelerator` pairs only with `SingleDeviceXLAStrategy` or `XLAStrategy`.

    The first four tests are marked ``xla`` and guarded with ``_requires_torch_xla`` because they exercise
    ``build_trainer``'s real, unpatched ``XLAPrecision`` construction, which needs ``torch_xla`` present -- no chip is
    touched, so the CPU-PJRT lane runs them. The later tests instead patch ``XLAPrecision`` (and, for the
    ``accelerator="auto"`` case, ``XLAAccelerator.is_available``) the same way ``TestBuildTrainerPrecision`` does, so
    they run on every lane without needing real ``torch_xla``.

    RF-DETR's generic ``strategy="auto"`` distributed branch creates ``DDPStrategy`` before Lightning can resolve an XLA
    accelerator. The guard therefore selects ``"xla"`` only when multiple local XLA devices are requested; one-device-
    per-host XLA is not claimed supported. Single-device XLA is unaffected because Lightning resolves ``"auto"`` to
    `SingleDeviceXLAStrategy`. The guard applies to explicit ``accelerator="tpu"``/``"xla"`` and to
    ``accelerator="auto"`` once it resolves to XLA (``TestAcceleratorResolvesToXLA`` above), for both segmentation and
    plain detection configs; only ``has_keypoints`` is excluded.
    """

    @pytest.mark.xla
    @_requires_torch_xla
    def test_multiple_xla_devices_select_the_xla_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """Without this, asking for more than one chip fails with `found DDPStrategy`."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=4)

        assert captured_trainer_kwargs["strategy"] == "xla"

    @pytest.mark.xla
    @_requires_torch_xla
    def test_single_xla_device_keeps_auto(self, captured_trainer_kwargs: dict[str, Any], tmp_path) -> None:
        """One chip already resolves to SingleDeviceXLAStrategy, so nothing should be overridden."""
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=1)

        assert captured_trainer_kwargs["strategy"] == "auto"

    @pytest.mark.xla
    @_requires_torch_xla
    def test_an_explicit_strategy_is_never_overridden(self, captured_trainer_kwargs: dict[str, Any], tmp_path) -> None:
        """A caller who names a strategy owns that choice, even on multi-device XLA.

        The new guard only rewrites ``"auto"``, so an explicit ``"ddp"`` string still reaches the pre-existing,
        unrelated ``strategy_name == "ddp"`` branch further down in ``build_trainer`` (see
        ``TestBuildTrainerDDPStrategy`` above), which always turns it into a ``DDPStrategy`` object -- on XLA and off it
        alike. Asserting the literal string ``"ddp"`` here would be wrong regardless of this PR.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=4, strategy="ddp")

        strategy_obj = captured_trainer_kwargs["strategy"]
        assert isinstance(strategy_obj, DDPStrategy)
        assert strategy_obj._ddp_kwargs.get("find_unused_parameters") is True

    @pytest.mark.xla
    @_requires_torch_xla
    def test_multi_device_xla_strategy_is_not_selected_for_keypoint_models(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """Keypoint models keep resolving to DDPStrategy on multi-device XLA instead of being promoted to `"xla"`.

        Keypoint training uses manual optimization and the DDP-specific ``find_unused_parameters=True`` handling a few
        lines below this guard (see the keypoint block above) has no validated XLA equivalent, so this combination is
        deliberately excluded from the fix and keeps hitting the pre-existing `DDPStrategy`/`XLAAccelerator` mismatch
        rather than running unverified.
        """
        build_trainer(_kp_tc(tmp_path, use_ema=False), _mc(use_grouppose_keypoints=True), accelerator="tpu", devices=4)

        assert isinstance(captured_trainer_kwargs["strategy"], DDPStrategy)

    def test_accelerator_auto_resolves_to_xla_strategy_when_xla_is_available(
        self, captured_trainer_kwargs: dict[str, Any], mock_xla_precision: MagicMock, mocker: MockerFixture, tmp_path
    ) -> None:
        """``accelerator="auto"`` -- TrainConfig's own default -- must be covered too, not just an explicit string.

        A caller who leaves ``accelerator`` unset passes literal ``"auto"`` here. Without
        ``_accelerator_resolves_to_xla``, RF-DETR's generic distributed branch would create ``DDPStrategy`` before
        Lightning resolves that value to XLA, causing the `XLAAccelerator`/`DDPStrategy` mismatch. Not marked
        ``xla``/``_requires_torch_xla``: it patches ``XLAPrecision`` and ``XLAAccelerator.is_available`` directly, so
        it runs on every CI lane rather than only the CPU-PJRT one.
        """
        mocker.patch("pytorch_lightning.accelerators.XLAAccelerator.is_available", return_value=True)

        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="auto", devices=4)

        assert captured_trainer_kwargs["strategy"] == "xla"
        assert "precision" not in captured_trainer_kwargs
        assert captured_trainer_kwargs["plugins"] == [mock_xla_precision.return_value]

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_node_single_device_keeps_the_existing_ddp_strategy(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """One device per host must not be promoted to unsupported ``XLAStrategy``.

        The generic distributed branch still creates ``DDPStrategy`` for ``devices=1, num_nodes=2``. That topology needs
        real multi-host XLA validation before it can select ``XLAStrategy``; this test only prevents the local strategy
        promotion from claiming it is supported. ``accelerator="tpu"`` is explicit here, so only ``XLAPrecision`` needs
        patching.
        """
        build_trainer(_tc(tmp_path, use_ema=False), _mc(amp=False), accelerator="tpu", devices=1, num_nodes=2)

        assert isinstance(captured_trainer_kwargs["strategy"], DDPStrategy)

    @pytest.mark.usefixtures("mock_xla_precision")
    def test_multi_device_xla_strategy_is_selected_for_segmentation_models(
        self, captured_trainer_kwargs: dict[str, Any], tmp_path
    ) -> None:
        """A segmentation config reaches the same guard as plain detection, since only ``has_keypoints`` is excluded.

        ``segmentation_head.sparse_forward()`` is one of the documented reasons the pre-existing DDP branch a few lines
        below needs ``find_unused_parameters=True`` (unused parameters on some forward steps); this guard does not
        special-case segmentation, so it must still promote it to ``"xla"`` on multi-device auto XLA rather than
        silently falling through to some other strategy.
        """
        build_trainer(
            _tc(tmp_path, use_ema=False), _mc(amp=False, segmentation_head=True), accelerator="tpu", devices=4
        )

        assert captured_trainer_kwargs["strategy"] == "xla"
