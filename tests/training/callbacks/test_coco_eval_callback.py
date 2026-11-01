# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for COCOEvalCallback."""

import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import DistributedSampler, SequentialSampler

from rfdetr.evaluation.matching import build_matching_data, merge_matching_data
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.training.coco_map import OnePassCocoMeanAveragePrecision, _UfcocoBackend, _VernierBackend
from rfdetr.utilities.imports import _IS_UFCOCO_INSTALLED, _IS_VERNIER_INSTALLED
from tests._markers import requires_torch_xla

ufcoco_only = pytest.mark.skipif(not _IS_UFCOCO_INSTALLED, reason="ultrafast_pycocotools not installed")
vernier_only = pytest.mark.skipif(not _IS_VERNIER_INSTALLED, reason="vernier not installed")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pl_module() -> MagicMock:
    """Return a minimal mock LightningModule."""
    return MagicMock(name="pl_module")


def _make_trainer(datamodule=None, callbacks: list[object] | None = None) -> MagicMock:
    """Return a minimal mock Trainer with an optional DataModule."""
    trainer = MagicMock(name="trainer")
    trainer.datamodule = datamodule
    trainer.callbacks = callbacks or []
    return trainer


class _TQDMProgressBar:
    """Minimal progress-bar stand-in for callback detection tests."""


def _detection_preds(n: int = 0) -> list[dict]:
    """Return a list with one per-image prediction dict."""
    return [
        {
            "boxes": torch.zeros(n, 4),
            "scores": torch.zeros(n),
            "labels": torch.zeros(n, dtype=torch.long),
        }
    ]


def _detection_targets(cx=0.5, cy=0.5, w=0.1, h=0.1, label=1) -> list[dict]:
    """Return a single-image target dict with one box in normalised CxCyWH."""
    return [
        {
            "boxes": torch.tensor([[cx, cy, w, h]]),
            "labels": torch.tensor([label]),
            "orig_size": torch.tensor([100, 200]),  # H=100, W=200
        }
    ]


def _minimal_metrics(pfx: str = "", max_dets: int = 500) -> dict:
    """Return a minimal torchmetrics-style metrics dict."""
    return {
        f"{pfx}map": torch.tensor(0.4),
        f"{pfx}map_50": torch.tensor(0.6),
        f"{pfx}map_75": torch.tensor(0.3),
        f"{pfx}mar_{max_dets}": torch.tensor(0.5),
    }


def _one_match_f1_state() -> dict:
    """Return per-class F1 accumulator state holding one matched ground-truth box.

    Examples:
        >>> _one_match_f1_state()[0]["total_gt"]
        1
    """
    return {
        0: {
            "scores": np.array([0.9], dtype=np.float32),
            "matches": np.array([1], dtype=np.int64),
            "ignore": np.array([False]),
            "total_gt": 1,
        }
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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
            pytest.param("ufcoco", _UfcocoBackend, id="ufcoco", marks=ufcoco_only),
            pytest.param("vernier", _VernierBackend, id="vernier", marks=vernier_only),
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

    def test_rank_with_only_padding_records_update(self, epoch_start, batch_end, loaders_attr, stage) -> None:
        """With fewer images than ranks, a rank whose whole shard is padding must still update its accumulators.

        Rank 1 of 2 over a single image receives image 0 as a padded repeat and nothing else. Skipping the update
        entirely would leave that rank voting "no updates" in the epoch-end collectives and silence the ranks that do
        hold data, so the batch flows through as empty lists instead.
        """
        cb = COCOEvalCallback()
        trainer = _trainer_with_sampler(loaders_attr, dataset_len=1, rank=1, num_replicas=2)
        cb.setup(trainer, _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        getattr(cb, epoch_start)(trainer, _cpu_module())

        outputs = {"results": _detection_preds(0), "targets": _detection_targets()}
        getattr(cb, batch_end)(trainer, _cpu_module(), outputs, None, 0)

        called_preds, called_targets = cb.map_metric.update.call_args.args
        assert (called_preds, called_targets) == ([], [])

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
    @requires_torch_xla
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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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


class TestMetricsTablePrinting:
    """Metric table terminal/notebook rendering behavior.

    Covers: terminal (console.print path), Rich-missing warning, teardown
    cleanup, RichProgressBar console routing, notebook in-place updates.
    """

    @pytest.mark.parametrize(
        "split,title_pfx",
        [
            pytest.param("val", "Val", id="val"),
            pytest.param("test", "Test", id="test"),
        ],
    )
    def test_terminal_metrics_tables_print_to_console(self, split: str, title_pfx: str, mocker: MockerFixture) -> None:
        """Terminal metric tables print directly through the Rich console each epoch."""
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True
        console = MagicMock(name="console")

        mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console)
        mocker.patch(
            "rfdetr.training.callbacks.coco_eval._render_overall_merged",
            side_effect=["overall-1", "overall-2"],
        )
        render_tables = mocker.patch("rfdetr.training.callbacks.coco_eval._render_summary_tables")
        cb._print_metrics_tables(trainer, split, {"mAP": 0.1}, [])
        cb._print_metrics_tables(trainer, split, {"mAP": 0.2}, [])

        assert render_tables.call_count == 2
        assert render_tables.call_args_list[0].args[0] is console
        assert render_tables.call_args_list[0].args[1].startswith(title_pfx)
        assert "(Epoch" in render_tables.call_args_list[0].args[1]
        assert render_tables.call_args_list[0].args[2] == "overall-1"

    @pytest.mark.parametrize("name", ["sign[/]", "helmet[red]", "price:dollar:"])
    def test_terminal_prints_hostile_class_name_through_real_console(self, mocker: MockerFixture, name: str) -> None:
        """A class name that looks like Rich markup prints verbatim through a real console without aborting.

        ``sign[/]`` used to raise ``MarkupError`` out of ``_print_metrics_tables``, which nothing between the callback
        and Lightning catches, so the existing mocked-console tests could not see the training abort.
        """
        import io

        from rich.console import Console

        mocker.patch("rfdetr.training.callbacks.coco_eval._render_overall_merged", return_value="overall")

        console = Console(file=io.StringIO(), width=200, color_system=None)
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True
        per_class = [{"name": name, "ap": 0.5, "ar": 0.6, "f1": 0.55, "precision": 0.6, "recall": 0.5}]

        mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console", return_value=console)
        cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, per_class)

        assert name in console.file.getvalue()

    def test_missing_rich_warns_once_and_skips_metric_tables(self, mocker: MockerFixture) -> None:
        """Missing Rich emits one warning and skips noisy table rendering."""
        get_console = mocker.patch("rfdetr.training.callbacks.coco_eval._get_rich_console")
        warning = mocker.patch("rfdetr.training.callbacks.coco_eval.logger.warning")
        mocker.patch("rfdetr.training.callbacks.coco_eval._IS_RICH_AVAILABLE", False)
        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer()
        trainer.is_global_zero = True

        cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, [])
        cb._print_metrics_tables(trainer, "val", {"mAP": 0.2}, [])

        warning.assert_called_once_with(
            "Rich is not installed; skipping metric table rendering. Install `rich` to enable tables."
        )
        assert cb._missing_rich_warning_emitted is True
        get_console.assert_not_called()

    def test_teardown_releases_notebook_widget(self) -> None:
        """Teardown clears the notebook output widget reference."""
        cb = COCOEvalCallback(in_notebook=True)
        cb._output_widget = MagicMock(name="output_widget")

        cb.teardown(_make_trainer(), _make_pl_module(), "fit")

        assert cb._output_widget is None

    @pytest.mark.parametrize("stage", ["fit", "validate", "test", "predict"])
    def test_teardown_no_op_when_no_widget(self, stage: str) -> None:
        """Teardown does not raise when no output widget was created."""
        cb = COCOEvalCallback(in_notebook=False)
        assert cb._output_widget is None

        cb.teardown(_make_trainer(), _make_pl_module(), stage)

        assert cb._output_widget is None

    def test_terminal_prints_through_rich_progress_bar_console(self, mocker: MockerFixture) -> None:
        """Metric tables route through RichProgressBar._console when active."""
        mocker.patch("rfdetr.training.callbacks.coco_eval._render_overall_merged", return_value="overall")
        # Create a fake callback whose class name is RichProgressBar so
        # _get_rich_console picks it up without importing PTL.
        rich_progress_bar_fake = type("RichProgressBar", (), {})
        rich_console = MagicMock(name="rich_console")
        fake_pb = rich_progress_bar_fake()
        fake_pb._console = rich_console  # type: ignore[attr-defined]

        cb = COCOEvalCallback(in_notebook=False)
        trainer = _make_trainer(callbacks=[fake_pb])
        trainer.is_global_zero = True

        cb._print_metrics_tables(trainer, "val", {"mAP": 0.5}, [])

        rich_console.print.assert_called_once()

    def test_notebook_metrics_tables_reuse_and_clear_output_widget(self, mocker: MockerFixture) -> None:
        """Notebook metric tables update one output widget instead of appending one table block per epoch."""

        class FakeOutput:
            """Minimal ipywidgets.Output stand-in."""

            def __init__(self) -> None:
                self.clear_output = MagicMock(name="clear_output")
                self.enter_count = 0

            def __enter__(self) -> "FakeOutput":
                self.enter_count += 1
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
                return False

        output_widget = FakeOutput()
        display = MagicMock(name="display")
        widgets_module = ModuleType("ipywidgets")
        widgets_module.Output = MagicMock(return_value=output_widget)
        ipython_module = ModuleType("IPython")
        ipython_module.__path__ = []
        display_module = ModuleType("IPython.display")
        display_module.display = display

        cb = COCOEvalCallback(in_notebook=True)
        trainer = _make_trainer()
        trainer.is_global_zero = True

        mocker.patch.dict(
            sys.modules,
            {
                "ipywidgets": widgets_module,
                "IPython": ipython_module,
                "IPython.display": display_module,
            },
        )
        mocker.patch(
            "rfdetr.training.callbacks.coco_eval._render_overall_merged", side_effect=["overall-1", "overall-2"]
        )
        render_summary_tables = mocker.patch("rfdetr.training.callbacks.coco_eval._render_summary_tables")
        cb._print_metrics_tables(trainer, "val", {"mAP": 0.1}, [])
        cb._print_metrics_tables(trainer, "val", {"mAP": 0.2}, [])

        widgets_module.Output.assert_called_once()
        display.assert_called_once_with(output_widget)
        assert cb._output_widget is output_widget
        assert [call.kwargs for call in output_widget.clear_output.call_args_list] == [
            {"wait": True},
            {"wait": True},
        ]
        assert output_widget.enter_count == 2
        assert render_summary_tables.call_count == 2


@pytest.mark.parametrize(
    "stage,hook,prefix",
    [
        pytest.param("fit", "on_validation_epoch_end", "val/", id="val"),
        pytest.param("test", "on_test_epoch_end", "test/", id="test"),
    ],
)
class TestEpochEndCommon:
    """Metric logging and state reset shared by on_validation_epoch_end and on_test_epoch_end."""

    def test_detection_core_metrics_are_logged(self, stage, hook, prefix) -> None:
        """mAP_50_95, mAP_50, mAP_75, mAR are always logged under the correct prefix."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}mAP_50_95" in logged_keys
        assert f"{prefix}mAP_50" in logged_keys
        assert f"{prefix}mAP_75" in logged_keys
        assert f"{prefix}mAR" in logged_keys

    def test_f1_metrics_logged_when_gt_present(self, stage, hook, prefix) -> None:
        """F1, precision, recall are logged when GT exists."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb._f1_local = _one_match_f1_state()
        module = _make_pl_module()
        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}F1" in logged_keys
        assert f"{prefix}precision" in logged_keys
        assert f"{prefix}recall" in logged_keys

    def test_f1_metrics_zero_when_no_gt(self, stage, hook, prefix) -> None:
        """F1 == 0.0 when no predictions were accumulated (empty epoch)."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == f"{prefix}F1")
        assert f1_call.args[1] == pytest.approx(0.0)

    def test_state_reset_after_epoch(self, stage, hook, prefix) -> None:
        """map_metric.reset() is called and _f1_local is cleared after epoch end."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb._f1_local = _one_match_f1_state()

        getattr(cb, hook)(_make_trainer(), _make_pl_module())

        cb.map_metric.reset.assert_called_once()
        assert cb._f1_local == {}

    def test_segmentation_extra_metrics_logged(self, stage, hook, prefix) -> None:
        """segm_mAP_50_95 and segm_mAP_50 are logged in segmentation mode."""
        cb = COCOEvalCallback(segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        segm_metrics = _minimal_metrics(pfx="bbox_")
        segm_metrics["segm_map"] = torch.tensor(0.35)
        segm_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric.compute.return_value = segm_metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}segm_mAP_50_95" in logged_keys
        assert f"{prefix}segm_mAP_50" in logged_keys


class TestKeypointCocoEvalRouting:
    """Tests for keypoint COCO evaluation routing in keypoint mode."""

    def test_coco_evaluator_accepts_keypoint_predictions(self) -> None:
        """Keypoint mode should forward keypoint predictions to COCO evaluator update()."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        cb.setup(trainer, module, stage="fit")

        evaluator = MagicMock(name="keypoint_coco_eval")
        cb._get_or_create_keypoint_oks_metric = MagicMock(return_value=evaluator)  # type: ignore[method-assign]
        outputs = {
            "results": [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                    "labels": torch.tensor([0], dtype=torch.int64),
                    "keypoints": torch.tensor([[[1.0, 2.0, 0.8]]], dtype=torch.float32),
                }
            ],
            "targets": [{"image_id": torch.tensor([12])}],
        }

        cb._update_keypoint_oks_metric(trainer, outputs, split="val")

        evaluator.update.assert_called_once()
        predictions = evaluator.update.call_args.args[0]
        assert 12 in predictions
        assert "keypoints" in predictions[12]
        assert predictions[12]["keypoints"].shape == (1, 1, 3)

    def test_keypoint_coco_eval_exposes_keypoint_ap_and_ar_metrics(self) -> None:
        """Epoch-end logging should expose keypoint AP and AR metrics from MetricKeypointOKS.compute()."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, module, stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()

        keypoint_metric = MagicMock(name="keypoint_oks_metric")
        keypoint_metric.has_updates = True
        keypoint_metric.compute.return_value = {"map": 0.42, "map_50": 0.72, "map_75": 0.31, "mar": 0.55}
        cb._keypoint_oks_metrics["val"] = keypoint_metric

        cb.on_validation_epoch_end(trainer, module)

        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        assert "val/keypoint_map_50_95" in logged
        assert "val/keypoint_map_50" in logged
        assert "val/keypoint_map_75" in logged
        assert "val/keypoint_mAR" in logged
        assert float(logged["val/keypoint_map_50_95"]) == pytest.approx(0.42)
        assert float(logged["val/keypoint_map_50"]) == pytest.approx(0.72)
        assert float(logged["val/keypoint_map_75"]) == pytest.approx(0.31)
        assert float(logged["val/keypoint_mAR"]) == pytest.approx(0.55)
        keypoint_log_calls = [call for call in module.log.call_args_list if call.args[0] == "val/keypoint_map_50_95"]
        assert keypoint_log_calls[0].kwargs.get("prog_bar") is True
        assert trainer.callback_metrics["val/keypoint_map_50_95"].item() == pytest.approx(0.42)
        assert trainer.callback_metrics["val/keypoint_map_50"].item() == pytest.approx(0.72)
        assert trainer.callback_metrics["val/keypoint_map_75"].item() == pytest.approx(0.31)
        assert trainer.callback_metrics["val/keypoint_mAR"].item() == pytest.approx(0.55)
        keypoint_metric.compute.assert_called_once()

    def test_keypoint_coco_eval_exposes_ema_keypoint_ap_and_ar_metrics(self) -> None:
        """EMA keypoint epoch-end logging should expose val/ema_keypoint_* metrics."""
        cb = COCOEvalCallback(max_dets=500)
        module = _make_pl_module()
        module.model_config = SimpleNamespace(use_grouppose_keypoints=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, module, stage="fit")

        keypoint_metric = MagicMock(name="ema_keypoint_oks_metric")
        keypoint_metric.has_updates = True
        keypoint_metric.compute.return_value = {"map": 0.25, "map_50": 0.5, "map_75": 0.2, "mar": 0.45}
        cb._keypoint_oks_metrics["val_ema"] = keypoint_metric

        cb._compute_and_log_keypoint_map("val_ema", module, trainer, log_split="val", metric_prefix="ema_")

        logged = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        assert "val/ema_keypoint_map_50_95" in logged
        assert "val/ema_keypoint_map_50" in logged
        assert "val/ema_keypoint_map_75" in logged
        assert "val/ema_keypoint_mAR" in logged
        assert float(logged["val/ema_keypoint_map_50_95"]) == pytest.approx(0.25)
        assert float(logged["val/ema_keypoint_map_50"]) == pytest.approx(0.5)
        assert float(logged["val/ema_keypoint_map_75"]) == pytest.approx(0.2)
        assert float(logged["val/ema_keypoint_mAR"]) == pytest.approx(0.45)
        keypoint_log_calls = [
            call for call in module.log.call_args_list if call.args[0] == "val/ema_keypoint_map_50_95"
        ]
        assert keypoint_log_calls[0].kwargs.get("prog_bar") is True
        assert trainer.callback_metrics["val/ema_keypoint_map_50_95"].item() == pytest.approx(0.25)
        assert trainer.callback_metrics["val/ema_keypoint_map_50"].item() == pytest.approx(0.5)
        assert trainer.callback_metrics["val/ema_keypoint_map_75"].item() == pytest.approx(0.2)
        assert trainer.callback_metrics["val/ema_keypoint_mAR"].item() == pytest.approx(0.45)
        keypoint_metric.compute.assert_called_once()

    def test_keypoint_oks_metric_created_with_correct_args(self, mocker: MockerFixture) -> None:
        """_get_or_create_keypoint_oks_metric must construct MetricKeypointOKS with coco_api and sigmas."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_val = dataset
        datamodule._dataset_test = None
        datamodule._dataset_train = None
        trainer = _make_trainer(datamodule=datamodule)
        coco_api = MagicMock(name="coco_api")

        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=coco_api)
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        result = cb._get_or_create_keypoint_oks_metric(trainer, split="val")

        assert result is oks_metric_cls.return_value
        oks_metric_cls.assert_called_once_with(coco_api, keypoint_oks_sigmas=[0.05], max_dets=500)

    @pytest.mark.parametrize(("split", "expected_dataset"), [("train", "train"), ("val_ema", "val")])
    def test_keypoint_oks_metric_uses_dataset_of_split(
        self, mocker: MockerFixture, split: str, expected_dataset: str
    ) -> None:
        """Train keypoint mAP is built from the train dataset and EMA keypoint mAP from the validation dataset."""
        cb = COCOEvalCallback(max_dets=500, keypoint_oks_sigmas=[0.05])
        train_dataset = MagicMock(name="train_dataset")
        val_dataset = MagicMock(name="val_dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = train_dataset
        datamodule._dataset_val = val_dataset
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)
        train_coco_api = MagicMock(name="train_coco_api")
        val_coco_api = MagicMock(name="val_coco_api")

        def _get_coco_api(dataset):
            if dataset is train_dataset:
                return train_coco_api
            if dataset is val_dataset:
                return val_coco_api
            return None

        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", side_effect=_get_coco_api)
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        cb._get_or_create_keypoint_oks_metric(trainer, split=split)

        assert oks_metric_cls.call_args.args[0] is {"train": train_coco_api, "val": val_coco_api}[expected_dataset]

    def test_mixed_keypoint_counts_create_keypoint_oks_metric(self, mocker: MockerFixture) -> None:
        """Mixed keypoint counts should be handled by MetricKeypointOKS instead of being skipped."""
        mocker.patch("rfdetr.training.callbacks.coco_eval.get_coco_api_from_dataset", return_value=MagicMock())
        oks_metric_cls = mocker.patch("rfdetr.training.callbacks.coco_eval.MetricKeypointOKS")
        warning = mocker.patch("rfdetr.training.callbacks.coco_eval.logger.warning")
        cb = COCOEvalCallback(max_dets=500)
        dataset = MagicMock(name="dataset")
        datamodule = MagicMock()
        datamodule._dataset_train = dataset
        datamodule._dataset_val = None
        datamodule._dataset_test = None
        trainer = _make_trainer(datamodule=datamodule)

        result = cb._get_or_create_keypoint_oks_metric(trainer, split="train")

        assert result is oks_metric_cls.return_value
        oks_metric_cls.assert_called_once()
        warning.assert_not_called()


@pytest.mark.parametrize(
    "stage,hook,prefix",
    [
        pytest.param("fit", "on_validation_epoch_end", "val/", id="val"),
        pytest.param("test", "on_test_epoch_end", "test/", id="test"),
    ],
)
class TestPerClassAPLogging:
    """Per-class AP logging behavior for validation and test loops."""

    def test_per_class_ap_logged_when_classes_present(self, stage, hook, prefix) -> None:
        """AP/<name> is logged for each class when class metrics are present."""
        cb = COCOEvalCallback()
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}AP/cat" in logged_keys
        assert f"{prefix}AP/dog" in logged_keys

    def test_per_class_ap_falls_back_to_str_id_when_no_class_names(self, stage, hook, prefix) -> None:
        """AP/<id> is logged when class_names is empty."""
        cb = COCOEvalCallback()
        cb.setup(_make_trainer(), _make_pl_module(), stage=stage)
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5])
        metrics["classes"] = torch.tensor([3])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        getattr(cb, hook)(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert f"{prefix}AP/3" in logged_keys


class TestOnValidationEpochEnd:
    """Validation-specific behaviour of on_validation_epoch_end."""

    @pytest.fixture
    def cb(self) -> COCOEvalCallback:
        """Return a callback whose base metric accumulated nothing while the EMA metric holds data.

        The base model is not evaluated: ``has_updates`` is False because the default policy never routes batches
        there, and every prediction went to ``map_metric_ema`` instead.

        Examples:
            >>> cb  # doctest: +SKIP
            A pytest fixture.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        return cb

    def test_ema_metrics_logged_when_map_metric_ema_populated(self) -> None:
        """val/ema_* metrics are logged when map_metric_ema has accumulated data.

        EMA metrics are now computed from a separate map_metric_ema that is populated during on_validation_batch_end
        (not aliased from base metrics).
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        # Simulate map_metric_ema being populated by on_validation_batch_end.
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAP_75" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()

    def test_still_logs_ema_metrics_when_base_metric_is_empty(self, cb: COCOEvalCallback) -> None:
        """Regression for #1285: EMA metrics are still logged when the base metric accumulated nothing this epoch.

        With the base model not evaluated, on_validation_batch_end routes every prediction to map_metric_ema.

        The empty-state guard in _compute_and_log must not also suppress the EMA metrics — before the fix it returned
        before ever reaching the EMA compute block, so such runs logged no validation output at all.
        """
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        cb.map_metric.compute.assert_not_called()
        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_mAP_50_95" in logged_keys
        assert "val/ema_mAP_50" in logged_keys
        assert "val/ema_mAR" in logged_keys
        cb.map_metric_ema.reset.assert_called_once()
        cb.map_metric.reset.assert_called_once()

    def test_ema_score_mirrored_onto_primary_keys_when_base_metric_empty(self, cb: COCOEvalCallback) -> None:
        """The EMA score must also reach val/mAP_50_95 when the base model was not evaluated this epoch.

        Leaving the primary key unpopulated — what the deprecated eval_ema_only flag did — is safe for an opt-in
        but not for a default: every scheduler, early-stopping hook, checkpoint monitor and dashboard watching
        val/mAP_50_95 would silently see nothing. The mirrored value must be the EMA one, not a stale or zero
        reading, so BestModelCallback's EMA track and external monitors agree on the epoch's score.
        """
        trainer = _make_trainer()
        trainer.callback_metrics = {}  # the mirror republishes whatever the EMA block wrote here
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list}
        assert logged["val/mAP_50_95"] == logged["val/ema_mAP_50_95"]
        assert trainer.callback_metrics["val/mAP_50_95"] == trainer.callback_metrics["val/ema_mAP_50_95"]
        assert logged["val/mAP_75"] == logged["val/ema_mAP_75"]
        assert logged["val/mAR"] == logged["val/ema_mAR"]

    def test_per_class_ap_mirrored_onto_primary_keys(self, cb: COCOEvalCallback) -> None:
        """EMA-only validation publishes per-class AP under both EMA and primary metric namespaces.

        Per-class AP is logged through pl_module.log alone, so the primary alias must be emitted explicitly alongside
        the EMA row when no base model was evaluated.
        """
        cb._class_names = ["cat"]
        cb._cat_id_to_name = {0: "cat"}
        ema_metrics = _minimal_metrics()
        ema_metrics["map_per_class"] = torch.tensor([0.5])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6])
        ema_metrics["classes"] = torch.tensor([0])
        cb.map_metric_ema.compute.return_value = ema_metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_AP/cat" in logged_keys
        assert "val/AP/cat" in logged_keys

        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list}
        assert logged["val/AP/cat"] == logged["val/ema_AP/cat"]

    def test_computes_f1_from_accumulated_matches_when_base_metric_is_empty(self, cb: COCOEvalCallback) -> None:
        """Regression for #1285: val/F1 is computed from accumulated matches even when the base metric is empty.

        on_validation_batch_end merges matching data into f1_local unconditionally, whichever mAP track a batch was
        routed to.

        With the base model not evaluated, map_metric never accumulates, so the empty-state guard in _compute_and_log
        used to discard f1_local via _reset_f1_local before ever computing val/F1 — silently dropping it even though
        real matching data (from the EMA-quality predictions) had been collected.
        """
        # A perfect match (pred == target box/label), mirroring what the unconditional
        # merge_matching_data call in on_validation_batch_end accumulates every batch.
        preds = [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
            }
        ]
        targets = [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([1])}]
        merge_matching_data(cb._f1_local, build_matching_data(preds, targets))
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        f1_call = next(c for c in module.log.call_args_list if c.args[0] == "val/F1")
        assert f1_call.args[1] == pytest.approx(1.0)

    def test_prints_summary_table_when_base_metric_is_empty(self, mocker: MockerFixture) -> None:
        """Regression for #1285: the console summary table is also part of "validation output".

        Before this fix, ``_print_metrics_tables`` was only ever called from the branch that reads the base ``metric`` —
        which never has data when the base model is not evaluated — so no table was ever printed for the run, even after
        the scalar/F1 logging gaps were closed. Per-class AP must be logged under ``ema_``-prefixed keys, mirroring
        ``val/ema_mAP_50_95`` staying separate from ``val/mAP_50_95`` elsewhere, so it never collides with a base-track
        key.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        ema_metrics = _minimal_metrics()
        ema_metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6, 0.3])
        ema_metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
        cb.on_validation_epoch_end(_make_trainer(), module)

        print_metrics_tables.assert_called_once()
        _trainer_arg, title_arg, overall_arg, per_class_arg = print_metrics_tables.call_args.args
        assert title_arg == "val (ema)"
        assert overall_arg["mAP 50:95"] == pytest.approx(0.4)
        assert {row["name"] for row in per_class_arg} == {"cat", "dog"}

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert "val/ema_AP/cat" in logged_keys
        assert "val/ema_AP/dog" in logged_keys
        assert "val/AP/cat" in logged_keys
        assert "val/AP/dog" in logged_keys

    def test_resets_val_ema_keypoints_when_ema_not_yet_warmed_up(self, mocker: MockerFixture) -> None:
        """Regression for #1289: an un-warmed EMA with an empty base metric resets the val_ema keypoint split.

        The split is reset, not computed.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = False  # EMA metric also has no updates this epoch
        module = _make_pl_module()

        compute_keypoint_map = mocker.patch.object(cb, "_compute_and_log_keypoint_map")
        reset_keypoint_split = mocker.patch.object(cb, "_reset_keypoint_split")
        cb.on_validation_epoch_end(_make_trainer(), module)

        compute_keypoint_map.assert_not_called()
        assert call("val_ema") in reset_keypoint_split.call_args_list
        cb.map_metric_ema.compute.assert_not_called()

    def test_summary_table_includes_segm_metrics(self, mocker: MockerFixture) -> None:
        """Regression for #1289: the EMA-only summary table includes segm mAP entries when segmentation is enabled."""
        cb = COCOEvalCallback(max_dets=500, segmentation=True)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb._class_names = ["cat"]
        cb._cat_id_to_name = {0: "cat"}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False  # genuinely empty: the default policy never routes here
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        ema_metrics = _minimal_metrics(pfx="bbox_")
        ema_metrics["map_per_class"] = torch.tensor([0.5])
        ema_metrics["mar_500_per_class"] = torch.tensor([0.6])
        ema_metrics["classes"] = torch.tensor([0])
        ema_metrics["segm_map"] = torch.tensor(0.2)
        ema_metrics["segm_map_50"] = torch.tensor(0.35)
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
        cb.on_validation_epoch_end(_make_trainer(), module)

        print_metrics_tables.assert_called_once()
        _trainer_arg, title_arg, overall_arg, _per_class_arg = print_metrics_tables.call_args.args
        assert title_arg == "val (ema)"
        assert overall_arg["segm mAP 50:95"] == pytest.approx(0.2)
        assert overall_arg["segm mAP 50"] == pytest.approx(0.35)

    def test_eval_interval_skips_non_matching_epochs(self) -> None:
        """Validation metric computation is skipped on non-interval epochs."""
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 0  # epoch 1 (1-based) is not divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        cb.map_metric.compute.assert_not_called()
        cb.map_metric.reset.assert_called_once()
        module.log.assert_not_called()

    def test_eval_interval_runs_on_matching_epochs(self) -> None:
        """Validation metric computation runs on interval-aligned epochs."""
        cb = COCOEvalCallback(eval_interval=3)
        trainer = _make_trainer()
        trainer.current_epoch = 2  # epoch 3 (1-based) is divisible by 3
        trainer.max_epochs = 10
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        cb.map_metric.compute.assert_called_once()
        module.log.assert_called()

    def test_progress_bar_suppresses_duplicate_pycocotools_output(self, capsys, mocker: MockerFixture) -> None:
        """Progress-bar training suppresses duplicate pycocotools stdout but still prints metric tables."""
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_TQDMProgressBar()])
        trainer.callback_metrics = {}
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric_ema = None
        module = _make_pl_module()

        def _compute_with_terminal_summary() -> dict:
            print("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = 0.000")
            return _minimal_metrics()

        cb.map_metric.compute.side_effect = _compute_with_terminal_summary

        print_metrics_tables = mocker.patch.object(cb, "_print_metrics_tables")
        cb._compute_and_log(trainer, module, "val")

        assert "Average Precision" not in capsys.readouterr().out
        print_metrics_tables.assert_called_once()
        cb.map_metric.compute.assert_called_once()
        module.log.assert_called()

    def test_per_class_ap_disabled_logs_no_class_keys(self) -> None:
        """log_per_class_metrics=False suppresses val/AP/<class> logging."""
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb._class_names = ["cat", "dog"]
        cb._cat_id_to_name = {0: "cat", 1: "dog"}
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["map_per_class"] = torch.tensor([0.5, 0.4])
        metrics["classes"] = torch.tensor([0, 1])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/AP/") for k in logged_keys)

    def test_per_class_ap_disabled_tolerates_zero_dim_ar(self) -> None:
        """Regression test for H1: log_per_class_metrics=False must not crash epoch-end validation.

        torchmetrics still emits ``mar_{max_dets}_per_class`` as a 0-dim ``tensor(-1.)`` even when
        ``class_metrics=False`` (only ``map_per_class`` is actually suppressed), while ``classes`` stays 1-d for >=2
        classes. Before the fix, the per-class AR block was gated on key-presence only (not on the flag), so
        ``zip(classes, mar_..._per_class)`` iterated a 0-d tensor and raised ``TypeError: iteration over a 0-d tensor``.
        ``_minimal_metrics()`` alone (used by the sibling ``test_per_class_ap_disabled_logs_no_class_keys``) omits this
        key entirely and would not catch the regression.
        """
        cb = COCOEvalCallback(log_per_class_metrics=False)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        metrics["classes"] = torch.tensor([0, 1])
        metrics["mar_500_per_class"] = torch.tensor(-1.0)  # torchmetrics' real 0-d shape when class_metrics=False
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(_make_trainer(), module)  # must not raise TypeError

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith("val/AP/") for k in logged_keys)

    def test_callback_metrics_updated_for_model_checkpoint(self) -> None:
        """Core metrics land in trainer.callback_metrics each epoch so improvement is detected.

        ModelCheckpoint and BestModelCallback read this dict to detect improvement.

        pl_module.log() from a callback's on_validation_epoch_end goes only to logged_metrics (external loggers), not
        callback_metrics.
        """
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()

        cb.on_validation_epoch_end(trainer, _make_pl_module())

        assert "val/mAP_50_95" in trainer.callback_metrics
        assert "val/mAP_50" in trainer.callback_metrics
        assert "val/mAP_75" in trainer.callback_metrics
        assert "val/mAR" in trainer.callback_metrics
        assert trainer.callback_metrics["val/mAP_50_95"].item() == pytest.approx(0.4)
        assert trainer.callback_metrics["val/mAP_50"].item() == pytest.approx(0.6)

    def test_callback_metrics_updated_with_ema_when_map_metric_ema_populated(self) -> None:
        """EMA metrics are written to callback_metrics when map_metric_ema has data."""
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True

        cb.on_validation_epoch_end(trainer, _make_pl_module())

        assert "val/ema_mAP_50_95" in trainer.callback_metrics
        assert "val/ema_mAP_50" in trainer.callback_metrics
        assert "val/ema_mAR" in trainer.callback_metrics

    def test_ema_segm_metrics_use_ema_values_not_base(self) -> None:
        """EMA segmentation metrics must come from map_metric_ema, not the base map_metric.

        Regression test for #978.
        """
        cb = COCOEvalCallback(max_dets=500, segmentation=True)
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")

        # Base metrics: segm_map=0.35
        base_metrics = _minimal_metrics(pfx="bbox_")
        base_metrics["segm_map"] = torch.tensor(0.35)
        base_metrics["segm_map_50"] = torch.tensor(0.55)
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = base_metrics

        # EMA metrics: segm_map=0.45 (deliberately different)
        ema_metrics = _minimal_metrics(pfx="bbox_")
        ema_metrics["segm_map"] = torch.tensor(0.45)
        ema_metrics["segm_map_50"] = torch.tensor(0.65)
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = ema_metrics
        cb._ema_has_updates = True
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        # EMA segm values must differ from base
        assert trainer.callback_metrics["val/ema_segm_mAP_50_95"].item() == pytest.approx(0.45)
        assert trainer.callback_metrics["val/ema_segm_mAP_50"].item() == pytest.approx(0.65)
        # Base segm values unchanged
        assert trainer.callback_metrics["val/segm_mAP_50_95"].item() == pytest.approx(0.35)
        assert trainer.callback_metrics["val/segm_mAP_50"].item() == pytest.approx(0.55)
        # pl_module.log() must also receive EMA values (covers both changed code paths)
        logged = {c.args[0]: c.args[1] for c in module.log.call_args_list if len(c.args) >= 2}
        assert logged["val/ema_segm_mAP_50_95"].item() == pytest.approx(0.45)
        assert logged["val/ema_segm_mAP_50"].item() == pytest.approx(0.65)

    def test_ghost_class_with_negative_ar_sentinel_is_filtered(self) -> None:
        """A class where both ap=-1 and ar=-1 (negative sentinels, not NaN) must be excluded from the per-class table.

        The old filter checked for NaN only, so ar=-1 (a valid float) escaped the guard.
        """
        cb = COCOEvalCallback()
        cb._cat_id_to_name = {0: "fish"}
        trainer = _make_trainer()
        trainer.callback_metrics = {}
        cb.setup(trainer, _make_pl_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        metrics = _minimal_metrics()
        # class 0 is a real class; class 8 is a ghost with both sentinels = -1
        metrics["map_per_class"] = torch.tensor([0.5, -1.0])
        metrics["classes"] = torch.tensor([0, 8])
        # ar=-1 for ghost (negative sentinel, not NaN)
        metrics["mar_500_per_class"] = torch.tensor([0.6, -1.0])
        cb.map_metric.compute.return_value = metrics
        module = _make_pl_module()

        cb.on_validation_epoch_end(trainer, module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        # real class logged, ghost class suppressed
        assert "val/AP/fish" in logged_keys
        assert "val/AP/8" not in logged_keys


# ---------------------------------------------------------------------------
# Test-epoch-end-only behaviour
# ---------------------------------------------------------------------------


class TestOnTestEpochEnd:
    """Test-loop-specific behaviour of on_test_epoch_end."""

    @pytest.mark.parametrize("forbidden_prefix", ["test/ema_", "val/"])
    def test_does_not_log_forbidden_prefix(self, forbidden_prefix: str) -> None:
        """No test/ema_* aliases (test already runs with EMA weights) and no val/ keys bleeding across loops."""
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="test")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        module = _make_pl_module()

        cb.on_test_epoch_end(_make_trainer(), module)

        logged_keys = {c.args[0] for c in module.log.call_args_list}
        assert not any(k.startswith(forbidden_prefix) for k in logged_keys)


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


class TestConvertPreds:
    """_convert_preds() normalizes prediction dicts for metric consumers."""

    @pytest.mark.parametrize(
        ("boxes", "expected_kept_idxs"),
        [
            pytest.param(
                # Degenerate first -> keep original index 1 (non-zero keep idx).
                [[2.0, 2.0, 2.0, 4.0], [0.0, 0.0, 3.0, 3.0], [5.0, 5.0, 5.0, 7.0]],
                [1],
                id="degenerate-first-keeps-index-1",
            ),
            pytest.param(
                # Degenerate between valid boxes -> keep non-contiguous original indices.
                [[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 2.0, 4.0], [4.0, 4.0, 6.0, 6.0]],
                [0, 2],
                id="degenerate-middle-keeps-noncontiguous",
            ),
        ],
    )
    def test_masks_remain_aligned_with_original_indices_after_degenerate_filtering(
        self,
        boxes: list[list[float]],
        expected_kept_idxs: list[int],
    ) -> None:
        """Filtering degenerate boxes must preserve mask alignment via original indices.

        Regression context: when a degenerate box is not last, keep indices are non-zero/non-contiguous. Downstream
        filtering must keep masks from the same original prediction indices.
        """
        cb = COCOEvalCallback()

        # Distinct one-hot masks so index/mask misalignment is easy to detect.
        masks = torch.zeros(3, 1, 2, 2, dtype=torch.bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, 0, 1] = True
        masks[2, 0, 1, 0] = True

        preds = [
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                "labels": torch.tensor([0, 0, 0], dtype=torch.int64),
                "masks": masks,
            }
        ]

        out = cb._convert_preds(preds)
        out_boxes = out[0]["boxes"]
        out_masks = out[0]["masks"]
        assert out_masks.shape == (3, 2, 2)

        keep = torch.where((out_boxes[:, 2] > out_boxes[:, 0]) & (out_boxes[:, 3] > out_boxes[:, 1]))[0]
        assert keep.tolist() == expected_kept_idxs
        assert torch.equal(out_masks[keep], masks.squeeze(1)[keep])


class TestConvertTargets:
    """_convert_targets() converts normalised CxCyWH to absolute xyxy."""

    def test_box_conversion_known_values(self) -> None:
        """CxCyWH(0.5,0.5,0.4,0.6) × (W=100,H=200) → xyxy(30,40,70,160)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([200, 100]),  # H=200, W=100
            }
        ]
        out = cb._convert_targets(targets)
        boxes = out[0]["boxes"]
        # cx=0.5*100=50, cy=0.5*200=100, w=0.4*100=40, h=0.6*200=120
        # → x1=50-20=30, y1=100-60=40, x2=50+20=70, y2=100+60=160
        assert boxes[0, 0].item() == pytest.approx(30.0)
        assert boxes[0, 1].item() == pytest.approx(40.0)
        assert boxes[0, 2].item() == pytest.approx(70.0)
        assert boxes[0, 3].item() == pytest.approx(160.0)

    def test_labels_passed_through(self) -> None:
        """Labels tensor is preserved unchanged."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert out[0]["labels"][0].item() == 7

    def test_masks_passed_through_as_bool(self) -> None:
        """Masks tensor is cast to bool and included in output."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([8, 8]),
                "masks": torch.ones(1, 8, 8, dtype=torch.uint8),
            }
        ]
        out = cb._convert_targets(targets)
        assert "masks" in out[0]
        assert out[0]["masks"].dtype == torch.bool

    def test_masks_resize_to_original_grid_without_predictions(self) -> None:
        """Default full-resolution evaluation keeps the original-image mask grid."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.uint8),
            }
        ]

        out = cb._convert_targets(targets)

        assert out[0]["masks"].shape == (1, 5, 5)

    def test_iscrowd_passed_through(self) -> None:
        """Iscrowd tensor is included when present."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
                "iscrowd": torch.tensor([1]),
            }
        ]
        out = cb._convert_targets(targets)
        assert "iscrowd" in out[0]
        assert out[0]["iscrowd"][0].item() == 1

    def test_no_masks_no_iscrowd_keys_absent(self) -> None:
        """Output dict contains exactly boxes and labels when extras are absent."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert set(out[0].keys()) == {"boxes", "labels"}

    @pytest.mark.parametrize(
        ("device", "num_targets", "expected_calls"),
        [
            pytest.param("cpu", 3, [(3, 2)], id="multi-target-batch"),
            pytest.param("cpu", 0, [], id="empty-targets"),
            pytest.param(
                "cuda",
                3,
                [(3, 2)],
                id="multi-target-batch-cuda",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
                ],
            ),
        ],
    )
    def test_orig_size_is_read_once_per_batch(
        self, device: str, num_targets: int, expected_calls: list[tuple[int, ...]], mocker: MockerFixture
    ) -> None:
        """A non-empty target batch reads every orig_size in one device-to-host synchronization, not one per target."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4, device=device),
                "labels": torch.tensor([0], device=device),
                "orig_size": torch.tensor([10, 10], device=device),
            }
            for _ in range(num_targets)
        ]
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        mocker.patch.object(torch.Tensor, "tolist", tracked_tolist)
        cb._convert_targets(targets)

        assert calls == expected_calls

    def test_batch_pairs_non_square_orig_sizes_with_their_target_rows(self) -> None:
        """Each target's box uses its own non-square orig_size after batched size conversion.

        This prevents swapped ``orig_size.tolist()`` rows and height/width transposition from assigning a sibling
        image's scale to the wrong target.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.4]]),
                "labels": torch.tensor([3]),
                "orig_size": torch.tensor([100, 300]),  # H=100, W=300
            },
            {
                "boxes": torch.tensor([[0.5, 0.25, 0.4, 0.2]]),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([240, 80]),  # H=240, W=80
            },
        ]

        out = cb._convert_targets(targets)

        assert out[0]["boxes"].shape == (1, 4)
        assert out[1]["boxes"].shape == (1, 4)
        assert out[0]["labels"].item() == 3
        assert out[1]["labels"].item() == 7
        torch.testing.assert_close(out[0]["boxes"], torch.tensor([[45.0, 30.0, 105.0, 70.0]]))
        torch.testing.assert_close(out[1]["boxes"], torch.tensor([[24.0, 36.0, 56.0, 84.0]]))


class TestConvertTargetsWithPreds:
    """_convert_targets(targets, preds) resizes masks to the paired prediction's grid, else to orig_size."""

    @pytest.mark.parametrize(
        ("pred_masks_0", "pred_masks_1", "expected_shape_0", "expected_shape_1"),
        [
            pytest.param(
                torch.zeros(1, 4, 4, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (4, 4),
                (3, 3),
                id="both-images-resize-to-distinct-pred-grids",
            ),
            pytest.param(
                torch.zeros(1, 10, 10, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (10, 10),
                (3, 3),
                id="first-image-pred-grid-matches-orig-size-second-needs-resize",
            ),
        ],
    )
    def test_batch_resizes_each_target_to_its_own_paired_pred_grid(
        self,
        pred_masks_0: torch.Tensor,
        pred_masks_1: torch.Tensor,
        expected_shape_0: tuple[int, int],
        expected_shape_1: tuple[int, int],
    ) -> None:
        """In a 2-image batch with differing orig_size and pred grids, target[i] resizes to preds[i]'s grid.

        It never resizes to the other image's grid.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.ones(1, 10, 10, dtype=torch.bool),
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"masks": pred_masks_0}, {"masks": pred_masks_1}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == expected_shape_0
        assert out[1]["masks"].shape[-2:] == expected_shape_1

    def test_batch_with_one_image_missing_masks_key_leaves_it_absent(self) -> None:
        """In a mixed batch an image without GT 'masks' stays maskless.

        Its sibling still resizes its masks to its own pred grid.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                # no "masks" key -> detection-only target in an otherwise-segmentation batch.
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"boxes": torch.zeros(1, 4)}, {"masks": torch.zeros(1, 3, 3, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert "masks" not in out[0]
        assert out[1]["masks"].shape[-2:] == (3, 3)

    def test_upsize_via_preds_branch_preserves_extent_and_dtype(self) -> None:
        """GT masks at 2x2 nearest-upsize through the preds-aware branch to a 4x4 pred grid.

        The result stays bool with the same extent.
        """
        cb = COCOEvalCallback()
        gt_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        gt_masks[0, 0, 0] = True  # top-left cell
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([2, 2]),  # matches GT already -> orig_size alone would be a no-op
                "masks": gt_masks,
            }
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (4, 4)
        assert out[0]["masks"].dtype == torch.bool
        # nearest-neighbor upsize must preserve the filled top-left quadrant's extent.
        assert out[0]["masks"][0, :2, :2].all()
        assert not out[0]["masks"][0, 2:, :].any()
        assert not out[0]["masks"][0, :, 2:].any()

    def test_preds_shorter_than_targets_raises_assertion_error(self) -> None:
        """A preds list shorter than targets fails loudly instead of misaligning images to the wrong prediction."""
        cb = COCOEvalCallback()
        targets = [
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        with pytest.raises(AssertionError):
            cb._convert_targets(targets, preds)

    def test_empty_targets_and_empty_preds_returns_empty_list(self) -> None:
        """An empty batch (no images) through the preds-aware branch is a no-op, not an error."""
        cb = COCOEvalCallback()

        out = cb._convert_targets([], [])

        assert out == []

    def test_pred_entry_without_masks_key_falls_back_to_orig_size(self) -> None:
        """A prediction dict lacking 'masks' does not crash; GT resizes to orig_size as if preds were None."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.bool),
            }
        ]
        preds = [{"boxes": torch.zeros(1, 4)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (5, 5)

    def test_empty_mask_tensor_through_preds_path_yields_zero_instances_at_pred_grid(self) -> None:
        """K=0 GT masks resize through the preds-aware branch without raising, yielding shape (0, pred_h, pred_w)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.zeros(0, 10, 10, dtype=torch.bool),
            }
        ]
        preds = [{"masks": torch.zeros(3, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape == (0, 4, 4)


@pytest.fixture
def all_reduce(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Pretend a distributed group is initialized and return the mocked ``dist.all_reduce``."""
    mock = MagicMock(name="all_reduce")
    monkeypatch.setattr("rfdetr.training.callbacks.coco_eval.is_dist_avail_and_initialized", lambda: True)
    monkeypatch.setattr("rfdetr.training.callbacks.coco_eval.dist.all_reduce", mock)
    return mock


def _ema_callback(ema_underlying: MagicMock | None = None) -> MagicMock:
    """Return a mock that ``_get_ema_callback`` recognises (has ``get_ema_model_state_dict``).

    With ``ema_underlying`` the mock's averaged model wraps it, so the EMA inner module's ``.model`` is that object.

    Examples:
        >>> underlying = MagicMock(name="ema_underlying_model")
        >>> _ema_callback(underlying)._average_model.module.model is underlying
        True
        >>> callable(_ema_callback().get_ema_model_state_dict)
        True
    """
    cb = MagicMock(name="ema_callback")
    cb.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
    if ema_underlying is not None:
        cb._average_model = SimpleNamespace(module=SimpleNamespace(model=ema_underlying))
    return cb


def _cpu_module() -> MagicMock:
    """Mock LightningModule whose ``device`` is a real string so ``metric.to(device)`` works.

    Examples:
        >>> _cpu_module().device
        'cpu'
    """
    module = MagicMock(name="pl_module")
    module.device = "cpu"
    return module


class TestEmaCollectiveSymmetry:
    """DDP-deadlock fix: the EMA metric's cross-rank sync must be issued symmetrically (#931/#449)."""

    def test_ema_metric_created_on_val_epoch_start_when_ema_active(self) -> None:
        """map_metric_ema is created on validation start whenever the EMA callback is present.

        This makes the EMA ``compute()`` collective rank-invariant — created on every rank regardless of how many (or
        zero) val batches that rank later processes — rather than lazily per-batch.
        """
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        assert cb.map_metric_ema is None  # not created yet at setup

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is not None

    def test_ema_metric_not_created_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None (no EMA collective is ever issued)."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    @pytest.mark.parametrize("has_updates", [False, True])
    def test_should_compute_ema_follows_local_updates_without_distributed(self, has_updates: bool) -> None:
        """Without a distributed group the EMA compute proceeds only when the metric saw updates.

        A rank whose EMA metric saw no updates votes against computing, which avoids empty-state divergence.
        """
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = has_updates

        assert cb._should_compute_ema(_cpu_module()) is has_updates

    @pytest.mark.parametrize(
        ("peer_vote", "expected"),
        [
            pytest.param(0, False, id="peer-lacks-ema"),
            pytest.param(1, True, id="all-ranks-have-ema"),
        ],
    )
    def test_unanimous_gate_follows_reduced_vote(self, all_reduce, peer_vote: int, expected: bool) -> None:
        """With local updates, the gate returns True only if the all_reduce MIN leaves the vote at 1."""
        all_reduce.side_effect = lambda flag, op=None: flag.fill_(peer_vote)  # the reduced vote across ranks
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = True

        assert cb._should_compute_ema(_cpu_module()) is expected
        all_reduce.assert_called_once()


class TestMapCollectiveSymmetry:
    """DDP-deadlock fix: the base/train mAP path's has_updates gate must be a collective vote.

    Unlike ``_should_compute_ema``'s unanimous ``all_reduce(MIN)`` (skip whenever any rank is empty), this path votes
    ``all_reduce(MAX)``: a rank with no local updates must still enter ``merge_distributed_state()`` when a peer has
    data, or the ranks diverge on which collectives they issue next and deadlock validation.
    """

    @pytest.mark.parametrize("has_updates", [False, True])
    def test_any_rank_has_updates_follows_local_updates_without_distributed(self, has_updates: bool) -> None:
        """With no distributed group the vote equals whether this rank has local updates."""
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = has_updates

        assert cb._any_rank_has_updates(metric, _cpu_module()) is has_updates

    @pytest.mark.parametrize(
        ("peer_vote", "expected"),
        [
            pytest.param(1, True, id="peer-has-updates"),
            pytest.param(0, False, id="no-rank-has-updates"),
        ],
    )
    def test_max_vote_follows_reduced_vote(self, all_reduce, peer_vote: int, expected: bool) -> None:
        """A locally-empty rank votes True only when the all_reduce MAX surfaces a peer's update."""
        all_reduce.side_effect = lambda flag, op=None: flag.fill_(peer_vote)  # the reduced vote across ranks
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = False

        assert cb._any_rank_has_updates(metric, _cpu_module()) is expected
        assert all_reduce.call_args.kwargs["op"] is torch.distributed.ReduceOp.MAX

    def test_compute_and_log_enters_merge_when_only_a_peer_has_updates(self, all_reduce) -> None:
        """A locally-empty rank still calls merge_distributed_state(), staying symmetric with its peer.

        Reproduces the deadlock scenario directly at the callback level: rank 0 has no local updates but a peer does.
        Before the fix, this rank took the early-return path and never issued ``merge_distributed_state()``'s
        collectives while its peer did, desynchronising the DDP collective sequence. After the fix, the collective
        vote makes both ranks agree to merge.
        """

        def _peer_has_map_updates_only(flag, op=None):
            # Peer has base mAP data (MAX vote -> 1) but no EMA data (MIN vote stays 0): isolates the has_updates gate
            # from the unrelated EMA collective this path also issues for split="val".
            if op is torch.distributed.ReduceOp.MAX:
                flag.fill_(1)

        all_reduce.side_effect = _peer_has_map_updates_only
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric.compute.return_value = _minimal_metrics()

        cb._compute_and_log(_make_trainer(), _cpu_module(), "val")

        cb.map_metric.merge_distributed_state.assert_called_once()


class TestOnTestEpochStart:
    """on_test_epoch_start resets _ema_has_updates before test to prevent stale val state."""

    def test_map_metric_ema_stays_none_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None after test hook fires."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_test_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    def test_resets_ema_has_updates_to_false(self) -> None:
        """on_test_epoch_start resets _ema_has_updates to False even when stale True from validation."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb._ema_has_updates = True  # simulate stale value from a preceding validation epoch

        cb.on_test_epoch_start(trainer, module)

        assert cb._ema_has_updates is False


class TestPrepareEmaMetricSecondEpoch:
    """_prepare_ema_metric resets (not re-creates) the metric on subsequent epochs."""

    def test_resets_not_recreates_metric(self) -> None:
        """Calling on_validation_epoch_start twice resets the metric rather than replacing it."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)
        assert cb.map_metric_ema is not None

        # Replace with a spy mock so reset() calls are trackable on the second epoch
        spy_metric = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = spy_metric

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is spy_metric  # same object, not replaced
        spy_metric.reset.assert_called_once()


class TestComputeAndLogEmaResetPath:
    """Elif branch in _compute_and_log: gate False + metric not None → reset() fires."""

    def test_resets_ema_metric(self, mocker: MockerFixture) -> None:
        """EMA not computed this epoch but metric exists → reset() clears state for the next epoch."""
        cb = COCOEvalCallback()
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        trainer.callback_metrics = {}

        # EMA metric exists but no batch updated it this epoch → gate returns False → elif fires
        mock_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = mock_ema
        cb._ema_has_updates = False

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric.has_updates = True

        mocker.patch.object(cb, "_build_per_class_rows", return_value=[])
        mocker.patch.object(cb, "_print_metrics_tables")
        mocker.patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value={})
        cb._compute_and_log(trainer, module, "val")

        mock_ema.reset.assert_called_once()

    def test_clears_ema_has_updates_after_gate_true_compute(self) -> None:
        """Regression for #1289 review: after a gate-True EMA compute, `_ema_has_updates` must be cleared to False.

        Before this fix, `_compute_and_log_ema_metrics` reset `map_metric_ema` in both branches but never cleared
        `_ema_has_updates`, so a stale `True` left over from a validation epoch could reach a later
        `_compute_and_log(..., "train", ...)` call (when `compute_train_metrics=True`) and drive a spurious `compute()`
        on the freshly emptied EMA metric.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True

        should_compute_ema, _ema_metrics = cb._compute_and_log_ema_metrics(
            _make_trainer(), _make_pl_module(), "val", "", "mar_500"
        )

        assert should_compute_ema is True
        assert cb._ema_has_updates is False
