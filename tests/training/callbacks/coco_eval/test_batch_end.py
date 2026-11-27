# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for COCOEvalCallback set-up and its per-batch hooks."""

import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import DistributedSampler, SequentialSampler

from rfdetr.evaluation.matching import build_matching_data
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision, _UfcocoBackend, _VernierBackend
from tests._markers import _requires_cuda, _requires_torch_xla, _requires_ufcoco, _requires_vernier
from tests.training.callbacks.helpers import (
    _cpu_module,
    _detection_preds,
    _detection_targets,
    _ema_callback,
    _make_pl_module,
    _make_trainer,
    _minimal_metrics,
)


class TestSetup:
    """Setup() creates map_metric with correct configuration."""

    def test_init_defaults_notebook_flag_to_false_without_ipython(self) -> None:
        """Constructor sets _in_notebook=False when IPython import is unavailable."""
        original_import = __import__

        def _import_with_missing_ipython(name: str, *args, **kwargs):
            if name == "IPython":
                raise ImportError("IPython not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import_with_missing_ipython):
            cb = COCOEvalCallback(in_notebook=None)

        assert cb._in_notebook is False

    def test_detection_iou_type_is_bbox(self) -> None:
        """Detection mode uses iou_type='bbox'."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert isinstance(cb.map_metric, OnePassCocoMeanAveragePrecision)
        assert isinstance(cb.map_metric_train, OnePassCocoMeanAveragePrecision)
        assert "bbox" in cb.map_metric.iou_type
        assert "segm" not in cb.map_metric.iou_type

    def test_detection_max_detection_thresholds(self) -> None:
        """max_dets is forwarded to max_detection_thresholds."""
        cb = COCOEvalCallback(max_dets=300, segmentation=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert 300 in cb.map_metric.max_detection_thresholds

    def test_segmentation_iou_type_includes_segm(self) -> None:
        """Segmentation mode uses iou_type=['bbox','segm']."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert "segm" in cb.map_metric.iou_type

    def test_map_metric_created_on_every_setup_call(self) -> None:
        """Repeated setup() calls replace map_metric (idempotent)."""
        cb = COCOEvalCallback()
        trainer, module = _make_trainer(), _make_pl_module()
        cb.setup(trainer, module, stage="fit")
        first = cb.map_metric
        cb.setup(trainer, module, stage="validate")
        assert cb.map_metric is not first

    @pytest.mark.parametrize("segmentation", [False, True])
    def test_never_selects_pycocotools_backend_name(self, segmentation: bool) -> None:
        """Detection and segmentation modes never resolve torchmetrics' pycocotools backend, which reports ``map=-1``.

        The name is the one torchmetrics resolves its COCO helpers from, not the evaluator RF-DETR ends up running: the
        hotcoco adapter constructs the parent with ``"faster_coco_eval"`` and replaces the resolved modules afterwards,
        so this pins the enum member alone (``test_eval_backend_defaults_to_vernier`` covers which evaluator runs). Mask
        evaluation resolves its RLE utilities from the same backend, so both modes must agree.
        """
        cb = COCOEvalCallback(segmentation=segmentation)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric._coco_backend.backend == "faster_coco_eval"

    def test_eval_backend_reaches_every_metric(self, mocker: MockerFixture) -> None:
        """The configured evaluation backend must reach the validation and train-split metrics alike.

        The backend is chosen once in ``TrainConfig`` but instantiated three times, and the EMA metric is built by a
        separate method that re-lists its keyword arguments by hand, so a missed call site would silently evaluate part
        of a run on the other backend.
        """
        cb = COCOEvalCallback(eval_backend="faster_coco_eval")
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        mocker.patch.object(cb, "_get_ema_callback", return_value=MagicMock())
        cb._prepare_ema_metric(_make_trainer())

        assert cb.map_metric_ema is not None, "EMA metric must exist or this asserts nothing"
        for metric in (cb.map_metric, cb.map_metric_train, cb.map_metric_ema):
            assert not isinstance(metric._coco_backend, _VernierBackend)

    def test_eval_backend_defaults_to_vernier(self) -> None:
        """Omitting the backend must select vernier, which is what makes evaluation fast by default."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert isinstance(cb.map_metric._coco_backend, _VernierBackend)

    @pytest.mark.parametrize("segmentation", [False, True])
    @pytest.mark.parametrize(
        "eval_backend, backend_type",
        [
            pytest.param("ufcoco", _UfcocoBackend, id="ufcoco", marks=_requires_ufcoco),
            pytest.param("vernier", _VernierBackend, id="vernier", marks=_requires_vernier),
        ],
    )
    def test_optional_eval_backend_reaches_every_metric(
        self, segmentation: bool, eval_backend: str, backend_type: type, mocker: MockerFixture
    ) -> None:
        """The ufcoco and vernier backends must reach the validation, train-split and EMA metrics alike.

        Same three call sites as ``test_eval_backend_reaches_every_metric``, asserted positively on the backend type so
        that a call site falling back to the default would fail here rather than evaluate part of a run on vernier.
        """
        cb = COCOEvalCallback(segmentation=segmentation, eval_backend=eval_backend)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        mocker.patch.object(cb, "_get_ema_callback", return_value=MagicMock())
        cb._prepare_ema_metric(_make_trainer())

        assert cb.map_metric_ema is not None, "EMA metric must exist or this asserts nothing"
        for metric in (cb.map_metric, cb.map_metric_train, cb.map_metric_ema):
            assert isinstance(metric._coco_backend, backend_type)
            assert metric._coco_backend.backend == "faster_coco_eval"

    def test_constructor_parameter_order_is_append_only(self) -> None:
        """New constructor parameters must be appended, never inserted among the existing ones.

        None of the parameters are keyword-only and ``eval_ema_only`` documents positional support, so inserting
        a parameter rebinds every argument after it: a caller passing ``keypoint_oks_sigmas`` positionally would
        hand a list of OKS sigmas to whatever took its slot, with no error at the call site.
        """
        assert list(inspect.signature(COCOEvalCallback.__init__).parameters)[1:] == [
            "max_dets",
            "segmentation",
            "eval_interval",
            "log_per_class_metrics",
            "keypoint_oks_sigmas",
            "in_notebook",
            "eval_ema_only",
            "eval_base_model",
            "eval_backend",
            "min_class_support",
            "class_agnostic",
        ]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            pytest.param({}, True, id="default"),
            pytest.param({"log_per_class_metrics": True}, True, id="explicit-true"),
            pytest.param({"log_per_class_metrics": False}, False, id="explicit-false"),
        ],
    )
    def test_log_per_class_metrics_sets_class_metrics_compute(self, kwargs: dict, expected: bool) -> None:
        """log_per_class_metrics (default True) toggles per-class AP computation on the regular and train metrics.

        Regression test for #416: False skips the expensive per-class MeanAveragePrecision compute (not merely its table
        rendering at epoch end), while the default and an explicit True keep it on.
        """
        cb = COCOEvalCallback(**kwargs)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        assert cb.map_metric.class_metrics is expected
        assert cb.map_metric_train.class_metrics is expected

    def test_log_per_class_metrics_false_disables_class_metrics_on_ema_metric(self, mocker: MockerFixture) -> None:
        """The EMA metric mirrors the per-class computation flag."""
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        mocker.patch.object(cb, "_get_ema_callback", return_value=MagicMock())
        cb._prepare_ema_metric(_make_trainer())
        assert cb.map_metric_ema is not None
        assert cb.map_metric_ema.class_metrics is False

    def test_ema_metric_stays_on_cpu_when_module_uses_an_accelerator(self, mocker: MockerFixture) -> None:
        """The EMA metric itself remains on CPU because its accumulated state is CPU-only."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")

        mocker.patch.object(cb, "_get_ema_callback", return_value=MagicMock())
        cb._prepare_ema_metric(_make_trainer())

        assert cb.map_metric_ema is not None
        assert cb.map_metric_ema.device.type == "cpu"

    def test_keypoint_mode_does_not_enable_torchmetrics_keypoint_iou(self) -> None:
        """Keypoint mode must keep torchmetrics on bbox-only iou_type."""
        cb = COCOEvalCallback(segmentation=True)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        cb.setup(_make_trainer(), module, stage="fit")
        assert "bbox" in cb.map_metric.iou_type
        assert "segm" not in cb.map_metric.iou_type
        assert "keypoints" not in cb.map_metric.iou_type


class TestOnFitStart:
    """on_fit_start() populates class names from the datamodule."""

    def test_class_names_loaded_from_datamodule(self) -> None:
        """Class names are taken from trainer.datamodule.class_names."""
        dm = MagicMock()
        dm.class_names = ["cat", "dog"]
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._class_names == ["cat", "dog"]

    @pytest.mark.parametrize(
        "datamodule",
        [
            pytest.param(None, id="no-datamodule"),
            pytest.param(MagicMock(spec=[]), id="datamodule-without-class-names"),
        ],
    )
    def test_missing_class_names_leave_list_empty(self, datamodule) -> None:
        """Without a datamodule, or with one lacking class_names, the callback's class_names stays an empty list."""
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=datamodule), _make_pl_module())
        assert cb._class_names == []

    @pytest.mark.parametrize(
        ("coco", "expected"),
        [
            pytest.param(
                MagicMock(cats={1: {"name": "fish"}, 2: {"name": "shark"}}, label2cat={0: 1, 1: 2}),
                {0: "fish", 1: "shark"},
                id="label2cat-remaps-to-zero-based-labels",
            ),
            pytest.param(
                MagicMock(spec=["cats"], cats={1: {"name": "fish"}, 2: {"name": "shark"}}),
                {1: "fish", 2: "shark"},
                id="raw-cat-ids-without-label2cat",
            ),
        ],
    )
    def test_cat_id_to_name_follows_label2cat(self, coco, expected: dict) -> None:
        """With coco.label2cat (remap_category_ids=True) names map from 0-based labels; standard COCO keeps raw IDs."""
        dataset = MagicMock()
        dataset.coco = coco
        dm = MagicMock()
        dm.class_names = ["fish", "shark"]
        dm._dataset_val = dataset
        dm._dataset_train = None
        cb = COCOEvalCallback()
        cb.on_fit_start(_make_trainer(datamodule=dm), _make_pl_module())
        assert cb._cat_id_to_name == expected


@pytest.mark.parametrize(
    "hook,stage",
    [
        pytest.param("on_validation_batch_end", "fit", id="val"),
        pytest.param("on_test_batch_end", "test", id="test"),
    ],
)
class TestBatchEndCommon:
    """map_metric accumulation shared by on_validation_batch_end and on_test_batch_end."""

    def test_map_metric_update_receives_normalized_inputs(self, hook, stage) -> None:
        """The callback must pass normalized detection inputs to the adapter that owns CPU-state conversion."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        cb.map_metric.update.assert_called_once()
        called_preds, called_targets = cb.map_metric.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}
        assert "orig_size" not in called_targets[0]

    def test_f1_accumulator_grows_across_batches(self, hook, stage) -> None:
        """Calling the batch-end hook twice accumulates more GT in F1 state."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets(label=1)}
        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)
        total_after_1 = sum(v["total_gt"] for v in cb._f1_local.values())

        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 1)
        total_after_2 = sum(v["total_gt"] for v in cb._f1_local.values())

        assert total_after_2 == total_after_1 * 2

    def test_targets_converted_before_update(self, hook, stage) -> None:
        """map_metric.update receives targets with absolute xyxy boxes."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        captured = {}

        def _capture_update(preds, targets):
            captured["targets"] = targets

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.update.side_effect = _capture_update

        outputs = {
            "results": _detection_preds(0),
            "targets": _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1),
        }
        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        # Expected: CxCyWH(0.5,0.5,0.1,0.1) × scale(W=200,H=100) → xyxy(90,45,110,55)
        boxes = captured["targets"][0]["boxes"]
        assert boxes.shape == (1, 4)
        assert boxes[0, 0].item() == pytest.approx(90.0)
        assert boxes[0, 1].item() == pytest.approx(45.0)
        assert boxes[0, 2].item() == pytest.approx(110.0)
        assert boxes[0, 3].item() == pytest.approx(55.0)


def _trainer_with_sampler(loaders_attr: str, dataset_len: int, rank: int, num_replicas: int) -> MagicMock:
    """Return a mock trainer whose evaluation loader carries a real padded ``DistributedSampler``.

    Examples:
        >>> trainer = _trainer_with_sampler("val_dataloaders", dataset_len=21, rank=1, num_replicas=2)
        >>> sampler = trainer.val_dataloaders.sampler
        >>> (sampler.rank, sampler.num_replicas, sampler.total_size, sampler.num_samples)
        (1, 2, 22, 11)
    """
    sampler = DistributedSampler(list(range(dataset_len)), num_replicas=num_replicas, rank=rank, shuffle=False)
    loader = MagicMock(name="loader")
    loader.sampler = sampler
    trainer = _make_trainer()
    setattr(trainer, loaders_attr, loader)
    return trainer


@pytest.mark.parametrize(
    "epoch_start,batch_end,loaders_attr,stage",
    [
        pytest.param("on_validation_epoch_start", "on_validation_batch_end", "val_dataloaders", "fit", id="val"),
        pytest.param("on_test_epoch_start", "on_test_batch_end", "test_dataloaders", "test", id="test"),
    ],
)
class TestDistributedSamplerPaddingFilter:
    """DistributedSampler padding must not be scored: each image counts exactly once across ranks.

    Lightning pads the evaluation split to a multiple of ``world_size`` by repeating the leading images, so with 21
    images on 2 ranks each rank forwards 11 samples and the 22nd is image 0 again. The callback skips the repeat when
    accumulating, using the sampler's own ``rank, num_replicas, total_size`` arithmetic (``shuffle=False``).
    """

    @pytest.mark.parametrize(
        ("rank", "expected_batch_sizes"),
        [
            pytest.param(1, [4, 4, 2], id="padded-rank-drops-repeat"),
            pytest.param(0, [4, 4, 3], id="unpadded-rank-keeps-all"),
        ],
    )
    def test_only_padded_repeat_is_skipped(
        self, epoch_start, batch_end, loaders_attr, stage, rank: int, expected_batch_sizes: list
    ) -> None:
        """Only the DistributedSampler padding repeat is dropped, so a rank without padding scores every sample.

        Rank 1 of 2 over 21 images owns positions 1, 3, ..., 19, 21 and position 21 is the padded repeat; rank 0 owns
        positions 0, 2, ..., 20, all real, so nothing is dropped.
        """
        cb = COCOEvalCallback()
        trainer = _trainer_with_sampler(loaders_attr, dataset_len=21, rank=rank, num_replicas=2)
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        getattr(cb, epoch_start)(trainer, _cpu_module())

        seen: list[int] = []
        cb.map_metric.update.side_effect = lambda preds, targets: seen.append(len(targets))
        for batch_size in (4, 4, 3):  # the 11 samples this rank receives, in batches of 4
            outputs = {"results": _detection_preds(0) * batch_size, "targets": _detection_targets() * batch_size}
            getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0)

        assert seen == expected_batch_sizes

    def test_counter_restarts_each_epoch(self, epoch_start, batch_end, loaders_attr, stage) -> None:
        """A second epoch must skip the padded sample at the same position again, not one sample earlier."""
        cb = COCOEvalCallback()
        trainer = _trainer_with_sampler(loaders_attr, dataset_len=3, rank=1, num_replicas=2)
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        seen: list[int] = []
        cb.map_metric.update.side_effect = lambda preds, targets: seen.append(len(targets))

        for _ in range(2):
            getattr(cb, epoch_start)(trainer, _cpu_module())
            outputs = {"results": _detection_preds(0) * 2, "targets": _detection_targets() * 2}
            getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0)

        assert seen == [1, 1]

    @pytest.mark.parametrize("class_agnostic", [False, True])
    def test_rank_with_only_padding_records_update(
        self, epoch_start, batch_end, loaders_attr, stage, class_agnostic: bool
    ) -> None:
        """With fewer images than ranks, a rank whose whole shard is padding must still update its accumulators.

        Rank 1 of 2 over a single image receives image 0 as a padded repeat and nothing else. Skipping the update
        entirely would leave that rank voting "no updates" in the epoch-end collectives and silence the ranks that do
        hold data, so the batch flows through as empty lists instead. A real class-agnostic metric takes that empty
        update too and keeps voting, since the agnostic vote is a collective of its own.
        """
        cb = COCOEvalCallback(class_agnostic=class_agnostic)
        trainer = _trainer_with_sampler(loaders_attr, dataset_len=1, rank=1, num_replicas=2)
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        getattr(cb, epoch_start)(trainer, _cpu_module())

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0)

        called_preds, called_targets = cb.map_metric.update.call_args.args
        assert (called_preds, called_targets) == ([], [])
        agnostic_votes = cb.map_metric_agnostic is not None and cb.map_metric_agnostic.has_updates
        assert agnostic_votes == class_agnostic

    def test_sequential_sampler_is_left_alone(self, epoch_start, batch_end, loaders_attr, stage) -> None:
        """Single-process evaluation has no padding, so every sample is accumulated."""
        cb = COCOEvalCallback()
        loader = MagicMock(name="loader")
        loader.sampler = SequentialSampler(list(range(3)))
        trainer = _make_trainer()
        setattr(trainer, loaders_attr, loader)
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        getattr(cb, epoch_start)(trainer, _cpu_module())

        outputs = {"results": _detection_preds(0) * 3, "targets": _detection_targets() * 3}
        getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0)

        assert len(cb.map_metric.update.call_args.args[1]) == 3

    def test_each_loader_uses_its_own_padding_rule(self, epoch_start, batch_end, loaders_attr, stage) -> None:
        """A padded second loader must not inherit the unpadded first loader's sample counter or sampler."""
        cb = COCOEvalCallback()
        first_loader = MagicMock(name="first_loader")
        first_loader.sampler = SequentialSampler(list(range(3)))
        second_loader = MagicMock(name="second_loader")
        second_loader.sampler = DistributedSampler(list(range(3)), num_replicas=2, rank=1, shuffle=False)
        trainer = _make_trainer()
        setattr(trainer, loaders_attr, [first_loader, second_loader])
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        getattr(cb, epoch_start)(trainer, _cpu_module())

        outputs = {"results": _detection_preds(0) * 2, "targets": _detection_targets() * 2}
        getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0, dataloader_idx=0)
        getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0, dataloader_idx=1)

        assert [len(call.args[1]) for call in cb.map_metric.update.call_args_list] == [2, 1]


class TestOnTestBatchEnd:
    """Test-loop-specific behaviour of on_test_batch_end."""

    def test_dataloader_idx_param_has_default(self) -> None:
        """on_test_batch_end must accept calls with an explicit dataloader_idx."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        # Must not raise with explicit dataloader_idx=0
        cb.on_test_batch_end(_make_trainer(), _make_pl_module(), outputs, None, 0, dataloader_idx=0)

        cb.map_metric.update.assert_called_once()


class TestValidationBatchEndDeviceRouting:
    """Device split between the CPU-resident mAP metric state and the GPU-original F1/keypoint inputs."""

    def test_xla_syncs_all_live_outputs_before_metric_host_reads(self, mocker: MockerFixture) -> None:
        """XLA evaluation must materialize the full live graph once before per-field CPU conversion."""
        cb = COCOEvalCallback()
        fake_xla = ModuleType("torch_xla")
        fake_xla.sync = MagicMock()  # type: ignore[attr-defined]

        mocker.patch.dict(sys.modules, {"torch_xla": fake_xla})
        cb._sync_xla_metric_inputs(SimpleNamespace(device=torch.device("xla")))

        fake_xla.sync.assert_called_once_with(wait=True)  # type: ignore[attr-defined]

    def test_cpu_metric_inputs_do_not_import_or_sync_xla(self, mocker: MockerFixture) -> None:
        """The materialization barrier must remain an XLA-only behavior change."""
        mocker.patch.dict(sys.modules, {"torch_xla": None})
        cb = COCOEvalCallback()

        cb._sync_xla_metric_inputs(SimpleNamespace(device=torch.device("cpu")))

    @pytest.mark.xla
    @_requires_torch_xla
    def test_sync_xla_metric_inputs_runs_against_real_torch_xla(self) -> None:
        """Real PJRT execution catches a signature mismatch with the installed ``torch_xla.sync`` API.

        The fully-mocked ordering tests below stand in for ``torch_xla.sync``, so they cannot catch such a mismatch.

        This calls the real function against a real XLA device and confirms the graph is left in a readable state
        afterward.
        """
        import torch_xla

        device = torch_xla.device()
        pl_module = SimpleNamespace(device=device)
        tensor = torch.arange(4, dtype=torch.float32, device=device) * 2

        COCOEvalCallback._sync_xla_metric_inputs(pl_module)

        assert tensor.cpu().tolist() == [0.0, 2.0, 4.0, 6.0]

    @pytest.mark.xla
    @pytest.mark.parametrize("hook", ["on_validation_batch_end", "on_test_batch_end"])
    def test_eval_hooks_sync_before_converting_predictions(self, hook: str) -> None:
        """Validation and test place the XLA barrier before both host conversions: predictions and targets."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit" if hook.startswith("on_validation") else "test")
        cb.map_metric = MagicMock(name="map_metric")
        events = []
        original_convert_preds = cb._convert_preds
        original_convert_targets = cb._convert_targets
        cb._sync_xla_metric_inputs = MagicMock(side_effect=lambda module: events.append("sync"))
        cb._convert_preds = MagicMock(
            side_effect=lambda preds: events.append("convert_preds") or original_convert_preds(preds)
        )
        cb._convert_targets = MagicMock(
            side_effect=lambda targets, preds=None, **kwargs: (
                events.append("convert_targets") or original_convert_targets(targets, preds, **kwargs)
            )
        )
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        getattr(cb, hook)(_make_trainer(), _make_pl_module(), outputs, None, 0)

        assert events == ["sync", "convert_preds", "convert_targets"]

    @pytest.mark.gpu
    @_requires_cuda
    def test_build_matching_data_receives_cuda_tensors_while_metric_state_on_cpu(self, mocker: MockerFixture) -> None:
        """build_matching_data keeps receiving the original CUDA tensors while map_metric's state is CPU-resident.

        No CPU-only test can encode this device split.

        A future refactor that passes the CPU-converted ``metric_preds``/``metric_targets`` into ``build_matching_data``
        instead of the GPU originals would silently move ``[N, H, W]`` mask/box IoU onto the accelerator's slower CPU
        sibling, a regression far larger than this PR's sync-removal win, with green CPU-only tests and zero numerical
        change.
        """
        cb = COCOEvalCallback()
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        outputs = {
            "results": [
                {
                    "boxes": torch.zeros(1, 4, device="cuda"),
                    "scores": torch.zeros(1, device="cuda"),
                    "labels": torch.zeros(1, dtype=torch.long, device="cuda"),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]], device="cuda"),
                    "labels": torch.tensor([1], device="cuda"),
                    "orig_size": torch.tensor([100, 200]),
                }
            ],
        }

        matching_spy = mocker.patch(
            "rfdetr.training.callbacks.coco_eval.build_matching_data", wraps=build_matching_data
        )
        cb.on_validation_batch_end(trainer, module, outputs, None, 0)

        matching_spy.assert_called_once()
        matching_preds, matching_targets = matching_spy.call_args[0][:2]
        assert all(value.is_cuda for item in matching_preds + matching_targets for value in item.values())
        assert all(t.device.type == "cpu" for t in cb.map_metric.detection_box)


class TestOnTrainBatchEnd:
    """Train-loop-specific behaviour for optional train mAP logging."""

    def test_delegates_normalized_inputs_to_adapter(self) -> None:
        """Train accumulation hands normalized predictions and targets to the adapter once.

        Scenario: ``compute_train_metrics=True``.
        """
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        cb.map_metric_train.update.assert_called_once()
        called_preds, called_targets = cb.map_metric_train.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}

    @pytest.mark.xla
    def test_syncs_before_converting_predictions(self) -> None:
        """Train-split accumulation places the XLA barrier before both host conversions: predictions and targets."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        module.device = torch.device("xla")
        events = []
        original_convert_preds = cb._convert_preds
        original_convert_targets = cb._convert_targets
        cb._sync_xla_metric_inputs = MagicMock(side_effect=lambda current: events.append("sync"))
        cb._convert_preds = MagicMock(
            side_effect=lambda preds: events.append("convert_preds") or original_convert_preds(preds)
        )
        cb._convert_targets = MagicMock(
            side_effect=lambda targets, preds=None, **kwargs: (
                events.append("convert_targets") or original_convert_targets(targets, preds, **kwargs)
            )
        )
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        assert events == ["sync", "convert_preds", "convert_targets"]

    def test_train_metrics_do_not_use_test_hook(self) -> None:
        """Train mAP must be logged under train/* via the train epoch hook, not through test/* hooks."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        cb.map_metric_train.compute.return_value = _minimal_metrics()
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)

        cb.on_train_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "train/mAP_50_95" in logged_keys
        assert "test/mAP_50_95" not in logged_keys

    def test_train_epoch_end_skips_compute_when_no_train_updates(self) -> None:
        """Train mAP should not call torchmetrics compute() when no train batches updated it."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        cb.map_metric_train.has_updates = False
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)

        cb.on_train_epoch_end(_make_trainer(), module)

        cb.map_metric_train.compute.assert_not_called()
        cb.map_metric_train.reset.assert_called_once()

    def test_validation_start_does_not_clear_train_metric_state(self) -> None:
        """In-fit validation should reset only validation accumulators, leaving train metrics isolated."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="val_map_metric")
        cb.map_metric_train = MagicMock(name="train_map_metric")

        cb.on_validation_epoch_start(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        cb.map_metric_train.reset.assert_not_called()

    @pytest.mark.parametrize(
        ("results", "expect_update"),
        [
            # _detection_preds has no "masks" key, mimicking sparse_forward training mode.
            pytest.param(_detection_preds(1), False, id="no-masks-key"),
            pytest.param(
                [
                    {
                        "boxes": torch.zeros(1, 4),
                        "scores": torch.zeros(1),
                        "labels": torch.zeros(1, dtype=torch.long),
                        "masks": torch.zeros(1, 16, 16, dtype=torch.bool),
                    }
                ],
                True,
                id="with-masks",
            ),
            pytest.param([], True, id="empty-preds"),
        ],
    )
    def test_segm_update_depends_on_masks_in_predictions(self, results: list, expect_update: bool) -> None:
        """Segm train accumulation skips the metric update when predictions lack masks, and updates otherwise.

        Empty predictions fall through the guard (``preds and ...`` short-circuits to False) and still update.
        """
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": results, "targets": _detection_targets()}

        cb.on_train_batch_end(_make_trainer(), module, outputs, None, 0)

        assert cb.map_metric_train.update.call_count == int(expect_update)

    @pytest.mark.parametrize(
        ("current_epoch", "expect_update"),
        [
            pytest.param(0, False, id="epoch-1-not-divisible-by-3"),
            pytest.param(2, True, id="epoch-3-divisible-by-3"),
        ],
    )
    def test_eval_interval_gates_accumulation_by_epoch(self, current_epoch: int, expect_update: bool) -> None:
        """With eval_interval=3 only interval-aligned epochs accumulate train batches.

        Regression guard: every train batch used to run the CPU-syncing accumulation block even on epochs whose
        accumulated state on_train_epoch_end discards unread, wasting a blocking .cpu() sync on each skipped epoch.
        """
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = current_epoch
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric_train = MagicMock(name="map_metric_train")
        module = _make_pl_module()
        module.train_config = SimpleNamespace(compute_train_metrics=True)
        outputs = {"results": _detection_preds(1), "targets": _detection_targets()}

        cb.on_train_batch_end(trainer, module, outputs, None, 0)

        assert cb.map_metric_train.update.called is expect_update


class TestValidationBatchEndEvalPolicy:
    """on_validation_batch_end skips the duplicate EMA forward pass by default (regression for #416).

    validation_step already forwarded through the EMA model directly; eval_base_model=True opts back in to the two-
    forward base+EMA comparison.
    """

    def test_skips_duplicate_ema_forward_by_default(self) -> None:
        """The default policy never re-runs the EMA forward.

        The single forward's predictions, in normalized metric-input form, are routed only to the EMA metric track.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()
        cb.map_metric_ema.update.assert_called_once()
        cb.map_metric.update.assert_not_called()
        assert cb._ema_has_updates is True
        called_preds, called_targets = cb.map_metric_ema.update.call_args.args
        assert called_preds[0].keys() == {"boxes", "scores", "labels"}
        assert called_targets[0].keys() == {"boxes", "labels"}

    def test_falls_back_to_regular_track_when_ema_not_warmed_up(self) -> None:
        """Without an averaged EMA model yet, predictions go to the regular map_metric.

        This mirrors the module's base-model fallback.
        """
        cb = COCOEvalCallback()
        ema_cb = _ema_callback()
        ema_cb._average_model = None
        trainer = _make_trainer(callbacks=[ema_cb])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        cb.map_metric.update.assert_called_once()
        cb.map_metric_ema.update.assert_not_called()
        assert cb._ema_has_updates is False

    def test_eval_base_model_runs_duplicate_ema_forward(self) -> None:
        """eval_base_model=True restores the independent base+EMA forwards.

        Each forward updates its own metric adapter exactly once.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_base_model=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_called_once()
        cb.map_metric.update.assert_called_once()
        cb.map_metric_ema.update.assert_called_once()

    @pytest.mark.xla
    def test_eval_base_model_syncs_each_forward_before_its_host_conversion(self) -> None:
        """The base+EMA policy builds two lazy graphs, so its second XLA sync follows the EMA forward.

        The second sync lands after the EMA forward and its postprocess.

        A misplaced early second sync would satisfy a check after the first host conversion without materializing the
        EMA graph. ``_convert_targets`` (its own host read) is tracked too, so a partial materialization ahead of the
        first barrier cannot slip through.
        """
        events = []
        ema_underlying = MagicMock(
            name="ema_underlying_model",
            side_effect=lambda *args, **kwargs: events.append("ema_forward") or {"ema": True},
        )
        cb = COCOEvalCallback(eval_base_model=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        module.postprocess = MagicMock(
            name="postprocess",
            side_effect=lambda *args, **kwargs: events.append("postprocess") or _detection_preds(0),
        )
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        original_convert_preds = cb._convert_preds
        original_convert_targets = cb._convert_targets
        cb._sync_xla_metric_inputs = MagicMock(side_effect=lambda current: events.append("sync"))
        cb._convert_preds = MagicMock(
            side_effect=lambda preds: events.append("convert_preds") or original_convert_preds(preds)
        )
        cb._convert_targets = MagicMock(
            side_effect=lambda targets, preds=None, **kwargs: (
                events.append("convert_targets") or original_convert_targets(targets, preds, **kwargs)
            )
        )

        cb.on_validation_batch_end(
            trainer,
            module,
            {"results": _detection_preds(0), "targets": _detection_targets()},
            (torch.zeros(1), None),
            0,
        )

        assert events == [
            "sync",
            "convert_preds",
            "convert_targets",
            "ema_forward",
            "postprocess",
            "sync",
            "convert_preds",
        ]

    def test_native_gt_resizes_directly_to_prediction_grid(self) -> None:
        """Native-head evaluation must not round-trip transformed GT masks through ``orig_size``.

        A 4x4 model-input mask with one pixel at ``[1, 1]`` is empty after direct nearest downsampling to 2x2. The
        historical 4x4 -> 5x5 -> 2x2 path instead retains that pixel, changing the metric target for the same model
        output.
        """
        cb = COCOEvalCallback(segmentation=True)
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")

        pred_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        gt_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks[0, 1, 1] = True
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": pred_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": gt_masks,
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, None, 0)

        called_targets = cb.map_metric.update.call_args[0][1]
        assert torch.equal(called_targets[0]["masks"], torch.zeros_like(pred_masks))

    def test_ema_validation_resizes_gt_to_ema_prediction_grid(self) -> None:
        """EMA segmentation metrics must use the EMA predictions' grid, not the base model's grid."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(segmentation=True, eval_base_model=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        base_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        ema_masks = torch.zeros(1, 3, 3, dtype=torch.bool)
        gt_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks[0, 1, 1] = True
        module.postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.zeros(1, 4),
                "masks": ema_masks,
            }
        ]
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": base_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": gt_masks,
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        expected_masks = torch.zeros_like(ema_masks)
        expected_masks[0, 1, 1] = True
        called_targets = cb.map_metric_ema.update.call_args[0][1]
        assert torch.equal(called_targets[0]["masks"], expected_masks)

    def test_ema_only_eval_and_native_resolution_masks_combined(self) -> None:
        """EMA-only evaluation with native-resolution masks routes to map_metric_ema.

        This is the #416 workaround; GT masks are resized to the prediction grid.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(segmentation=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")

        pred_masks = torch.zeros(1, 4, 4, dtype=torch.bool)
        gt_masks = torch.ones(1, 8, 8, dtype=torch.bool)
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": pred_masks,
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([8, 8]),
                    "masks": gt_masks,
                }
            ],
        }
        batch = (torch.zeros(1), None)

        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()
        cb.map_metric.update.assert_not_called()
        called_targets = cb.map_metric_ema.update.call_args[0][1]
        assert called_targets[0]["masks"].shape[-2:] == (4, 4)


class TestValidationBatchEndTargetConversion:
    """Independent base and EMA updates convert ``outputs["targets"]`` once when both conversions must agree."""

    def test_detection_hands_one_converted_target_list_to_both_metrics(self, mocker: MockerFixture) -> None:
        """Detection validation converts the batch's targets once and reuses that list for the EMA update.

        Both call sites pass ``preds=None`` outside segmentation, so the second conversion recomputed a
        byte-identical result: the same box rescale and the same ``orig_size`` host transfer, per validation batch,
        on every epoch that evaluates both models. Reuse is safe because the accumulators store detached CPU copies
        of what they are handed rather than the caller's tensors.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(eval_base_model=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}

        convert_targets = mocker.patch.object(cb, "_convert_targets", wraps=cb._convert_targets)
        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        convert_targets.assert_called_once_with(outputs["targets"], None, crowd_regions={})
        assert cb.map_metric_ema.update.call_args.args[1] is cb.map_metric.update.call_args.args[1]

    def test_segmentation_converts_targets_separately_for_ema_grid(self) -> None:
        """Segmentation validation must keep converting targets twice, once per prediction grid.

        Target masks are resampled onto the paired prediction's mask resolution, and the base and EMA heads can emit
        different resolutions for the same image. Sharing the base conversion here would silently score the EMA masks
        against a grid they were never produced on.
        """
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(segmentation=True, eval_base_model=True)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        module.postprocess.return_value = [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.zeros(1, 4),
                "masks": torch.zeros(1, 3, 3, dtype=torch.bool),
            }
        ]
        outputs = {
            "results": [
                {
                    "scores": torch.tensor([0.9]),
                    "labels": torch.tensor([0]),
                    "boxes": torch.zeros(1, 4),
                    "masks": torch.zeros(1, 2, 2, dtype=torch.bool),
                }
            ],
            "targets": [
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                    "labels": torch.tensor([0]),
                    "orig_size": torch.tensor([5, 5]),
                    "masks": torch.zeros(1, 4, 4, dtype=torch.bool),
                }
            ],
        }

        cb.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        ema_targets = cb.map_metric_ema.update.call_args.args[1]
        assert ema_targets is not cb.map_metric.update.call_args.args[1]
        assert ema_targets[0]["masks"].shape[-2:] == (3, 3)


class TestEvalEmaOnlyBatchToEpochEndToEnd:
    """Full batch -> epoch pipeline for the default EMA-only policy, with real (non-mocked) accumulators (#1285).

    TestValidationBatchEndEvalPolicy and TestOnValidationEpochEnd each replace map_metric / map_metric_ema with
    MagicMock and drive a single hook in isolation. Neither would catch a wiring bug between the two (e.g.
    `_ema_has_updates` set by batch-time routing never reaching epoch-end, or `_prepare_ema_metric` creating a
    `map_metric_ema` that batch-time routing never populates). This test keeps the real MeanAveragePrecision instances
    created by setup() / on_validation_epoch_start() and drives on_validation_batch_end then on_validation_epoch_end in
    sequence on the same callback instance.
    """

    def test_real_batch_then_epoch_end_logs_ema_metrics(self) -> None:
        """A real batch-end then epoch-end logs val/ema_* metrics and mirrors them onto the primary keys."""
        ema_underlying = MagicMock(name="ema_underlying_model", return_value={"ema": True})
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_ema_callback(ema_underlying)])
        trainer.callback_metrics = {}  # the mirror republishes whatever the EMA block wrote here
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb.on_validation_epoch_start(trainer, module)  # real map_metric_ema creation (_prepare_ema_metric)

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        batch = (torch.zeros(1), None)
        cb.on_validation_batch_end(trainer, module, outputs, batch, 0)

        ema_underlying.assert_not_called()  # the default policy routes validation_step's own forward, no duplicate
        assert cb._ema_has_updates is True
        assert cb.map_metric._update_count == 0  # regular track never accumulated this epoch

        cb.on_validation_epoch_end(trainer, module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        # Mirrored onto the primary keys because the base model was never evaluated this epoch.
        assert "val/mAP_50_95" in logged_keys
        assert "val/mAR" in logged_keys
