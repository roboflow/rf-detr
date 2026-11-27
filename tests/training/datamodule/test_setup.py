# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for RFDETRDataModule init, setup, sample visualization, class names and eval dataloaders.

Setup-time guards for training on the Kornia GPU augmentation path.
"""

import builtins
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader

from rfdetr.config import (
    AugmentationBackend,
    RFDETRKeypointPreviewConfig,
    RFDETRNanoConfig,
    RFDETRSegNanoConfig,
    TrainConfig,
)
from rfdetr.datasets import io_utils
from rfdetr.datasets.coco import CocoDetection
from rfdetr.datasets.kornia_transforms import build_kornia_pipeline
from rfdetr.datasets.yolo import YoloDetection, YoloSplitUnavailableError
from rfdetr.training.module_data import RFDETRDataModule
from tests._markers import _requires_kornia
from tests.datasets.helpers import _write_coco_dataset, _write_yolo_dataset
from tests.training.helpers import _base_model_config, _build_datamodule, _fake_dataset
from tests.training.helpers import _datamodule_train_config as _base_train_config

_EVAL_LOADERS = ["val_dataloader", "test_dataloader", "predict_dataloader"]


_EVAL_LOADER_DATASETS = [
    pytest.param("val_dataloader", "_dataset_val", id="val"),
    pytest.param("test_dataloader", "_dataset_test", id="test"),
    pytest.param("predict_dataloader", "_dataset_val", id="predict"),
]


class _VisualDataset(torch.utils.data.Dataset):
    """Minimal transformed dataset item for DataModule sample visualization."""

    def __len__(self) -> int:
        """Return the fixed fake dataset length."""
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return one normalized image tensor with box and keypoint targets."""
        return (
            torch.full((3, 16, 16), 0.5, dtype=torch.float32),
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
                "keypoints": torch.tensor([[[0.25, 0.25, 2.0], [0.75, 0.75, 0.0]]], dtype=torch.float32),
                "size": torch.tensor([16, 16], dtype=torch.int64),
            },
        )


class _MaskedVisualDataset(torch.utils.data.Dataset):
    """Transformed dataset item carrying the given instance ``masks`` and no keypoints.

    Examples:
        >>> _, target = _MaskedVisualDataset(torch.ones(2, 16, 16, dtype=torch.bool))[0]
        >>> sorted(target)
        ['boxes', 'labels', 'masks', 'size']
    """

    def __init__(self, masks: torch.Tensor) -> None:
        self._masks = masks

    def __len__(self) -> int:
        """Return the fixed fake dataset length."""
        return 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return one normalized image tensor with box and instance-mask targets."""
        return (
            torch.full((3, 16, 16), 0.5, dtype=torch.float32),
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
                "labels": torch.tensor([0], dtype=torch.int64),
                "masks": self._masks,
                "size": torch.tensor([16, 16], dtype=torch.int64),
            },
        )


class _SubstitutedVisualDataset(_VisualDataset):
    """Visual dataset whose sample 0 carries ``image_id`` 7, as a sample replaced by image 7 when unreadable does.

    It has the ``root``, ``ids`` and COCO API of ``CocoDetection``, the API naming image ``i`` ``image<i>.jpg``.

    Examples:
        >>> dataset = _SubstitutedVisualDataset()
        >>> int(dataset[0][1]["image_id"]), dataset.coco.loadImgs(7)[0]["file_name"]
        (7, 'image7.jpg')
    """

    def __init__(self) -> None:
        self.root = "/data/train"
        self.ids = [0]
        self.coco = SimpleNamespace(loadImgs=lambda image_id: [{"file_name": f"image{image_id}.jpg"}])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the sample of :class:`_VisualDataset` with the ``image_id`` of image 7."""
        image, target = super().__getitem__(idx)
        return image, {**target, "image_id": torch.tensor([7])}


@pytest.fixture
def coco_datamodule(tmp_path):
    """Return an RFDETRDataModule configured for a COCO dataset."""
    return _build_datamodule(
        train_config=_base_train_config(tmp_path, dataset_file="coco"),
        tmp_path=tmp_path,
    )


@pytest.fixture
def propagate_rf_detr_logs(monkeypatch):
    """Re-enable propagation on the ``rf-detr`` logger, which ``get_logger()`` disables, so ``caplog`` sees its
    records."""
    monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)


class TestInit:
    """RFDETRDataModule.__init__ stores configs and initialises dataset slots."""

    def test_stores_model_config(self, tmp_path, base_model_config):
        """model_config is accessible as an attribute after construction."""
        mc = base_model_config(num_classes=3)
        dm = _build_datamodule(model_config=mc, tmp_path=tmp_path)
        assert dm.model_config is mc

    def test_stores_train_config(self, tmp_path, base_train_config):
        """train_config is accessible as an attribute after construction."""
        tc = base_train_config(epochs=42)
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm.train_config is tc

    def test_datasets_start_as_none(self, fixture_training_setup):
        """All three dataset slots are None before setup() is called."""
        _, _, dm = fixture_training_setup
        assert dm._dataset_train is None
        assert dm._dataset_val is None
        assert dm._dataset_test is None

    @pytest.mark.parametrize(
        "num_workers, prefetch_factor, expected_prefetch_factor",
        [
            pytest.param(2, None, 2, id="defaults-to-two-when-workers-enabled"),
            pytest.param(2, 5, 5, id="honors-train-config"),
            pytest.param(0, 5, None, id="none-when-workers-disabled"),
        ],
    )
    def test_prefetch_factor(self, tmp_path, base_train_config, num_workers, prefetch_factor, expected_prefetch_factor):
        """_prefetch_factor is derived from num_workers and the configured prefetch_factor."""
        tc = base_train_config(num_workers=num_workers, prefetch_factor=prefetch_factor)
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm._prefetch_factor == expected_prefetch_factor

    def test_pin_memory_override_is_respected(self, tmp_path, base_train_config):
        """pin_memory can be explicitly overridden from TrainConfig."""
        tc = base_train_config(pin_memory=False)
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm._pin_memory is False

    def test_pin_memory_defaults_to_false_when_accelerator_is_cpu(self, mocker, tmp_path, base_train_config):
        """Default pin_memory stays off when training is explicitly CPU-only."""
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        tc = base_train_config(pin_memory=None, accelerator="cpu")
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm._pin_memory is False

    def test_persistent_workers_override_is_respected(self, tmp_path, base_train_config):
        """persistent_workers can be explicitly overridden from TrainConfig."""
        tc = base_train_config(num_workers=2, persistent_workers=False)
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm._persistent_workers is False

    @pytest.mark.parametrize("strategy", ["ddp_notebook", "ddp"])
    def test_strategy_preserves_num_workers(self, tmp_path, base_train_config, strategy):
        """No DDP strategy rewrites num_workers, and prefetch_factor keeps its default of 2 for workers > 0.

        ``ddp_notebook`` spawns children that initialise CUDA fresh, and DataLoader fork workers are CPU-only and never
        touch CUDA, so nested forks are safe.
        """
        tc = base_train_config(num_workers=4, strategy=strategy)
        dm = _build_datamodule(train_config=tc, tmp_path=tmp_path)
        assert dm._num_workers == 4
        assert dm._prefetch_factor == 2


class TestPrivateShowSamples:
    """RFDETRDataModule._show_samples renders transformed input samples."""

    def test_returns_figure_for_keypoint_targets(self, fixture_training_setup, monkeypatch):
        """_show_samples should render transformed boxes and keypoints without raw COCO parsing."""
        _, _, dm = fixture_training_setup
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: _VisualDataset())

        figure = dm._show_samples(1, split="train", columns=1)

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 1
        plt.close(figure)

    def test_accepts_figure_size_and_shortens_long_titles(self, fixture_training_setup, monkeypatch):
        """_show_samples should keep long image names inside subplot titles."""
        _, _, dm = fixture_training_setup
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: _VisualDataset())
        monkeypatch.setattr(dm, "_source_image_path", lambda dataset, target: Path(f"{'very_long_name_' * 8}.jpg"))

        figure = dm._show_samples(1, split="train", columns=1, figure_size=(4.0, 3.0))

        assert list(figure.get_size_inches()) == pytest.approx([4.0, 3.0])
        title = figure.axes[0].get_title()
        assert "..." in title
        assert len(title) <= 48
        plt.close(figure)

    def test_rejects_non_positive_count(self, fixture_training_setup):
        """_show_samples should fail fast for invalid counts."""
        _, _, dm = fixture_training_setup
        with pytest.raises(ValueError) as exc_info:
            dm._show_samples(0)
        assert "count must be positive" in str(exc_info.value)

    def test_rejects_invalid_figure_size(self, fixture_training_setup):
        """_show_samples should fail fast for invalid figure sizes."""
        _, _, dm = fixture_training_setup
        with pytest.raises(ValueError) as exc_info:
            dm._show_samples(1, figure_size=(4.0, 0.0))
        assert "figure_size values must be positive" in str(exc_info.value)

    def test_missing_visual_extra_has_install_hint(self, fixture_training_setup, monkeypatch):
        """_show_samples should explain how to install optional visualization dependencies."""
        _, _, dm = fixture_training_setup
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("matplotlib is intentionally unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: _VisualDataset())
        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError) as exc_info:
            dm._show_samples(1)
        assert "rfdetr[visual]" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("dataset", "expected_annotator_calls"),
        [
            pytest.param(_VisualDataset(), 0, id="no-masks-key"),
            pytest.param(_MaskedVisualDataset(torch.ones((1, 16, 16), dtype=torch.bool)), 1, id="instance-masks"),
            pytest.param(_MaskedVisualDataset(torch.zeros((0, 16, 16), dtype=torch.bool)), 0, id="zero-instance-masks"),
        ],
    )
    def test_mask_annotator_runs_only_for_non_empty_masks(
        self, mocker, fixture_training_setup, monkeypatch, dataset, expected_annotator_calls
    ):
        """_show_samples overlays instance masks only when the targets carry at least one mask."""
        _, _, dm = fixture_training_setup
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure

        mock_mask_ann = mocker.patch("supervision.MaskAnnotator")
        mock_mask_ann.return_value.annotate.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: dataset)

        figure = dm._show_samples(1, split="train", columns=1)

        assert isinstance(figure, Figure)
        assert mock_mask_ann.call_count == expected_annotator_calls
        assert mock_mask_ann.return_value.annotate.call_count == expected_annotator_calls
        plt.close(figure)

    def test_title_names_the_image_the_sample_was_read_from(self, fixture_training_setup, monkeypatch):
        """A substituted sample is titled with the file it was read from, not with the unreadable file requested."""
        _, _, dm = fixture_training_setup
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: _SubstitutedVisualDataset())

        figure = dm._show_samples(1, split="train", columns=1)

        assert figure.axes[0].get_title() == "image7.jpg"
        plt.close(figure)

    def test_title_falls_back_to_split_and_index_without_a_coco_api(self, fixture_training_setup, monkeypatch):
        """A dataset without ``root`` and a COCO API, such as a multi-source one, is titled by split and index."""
        _, _, dm = fixture_training_setup
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        monkeypatch.setattr(dm, "_get_dataset_for_visualization", lambda split: _VisualDataset())

        figure = dm._show_samples(1, split="train", columns=1)

        assert figure.axes[0].get_title() == "train[0]"
        plt.close(figure)

    def test_source_image_path_names_the_substitute_of_an_unreadable_coco_image(self, tmp_path, mocker):
        """The path comes from the id ``CocoDetection`` returns, which names the substitute of an unreadable file."""
        mocker.patch.object(io_utils.time, "sleep")
        annotation_file = _write_coco_dataset(tmp_path, "png")
        (tmp_path / "image1.png").write_bytes(b"not an image")
        dataset = CocoDetection(tmp_path, annotation_file, transforms=None, skip_unreadable=True)
        _, target = dataset[1]

        assert RFDETRDataModule._source_image_path(dataset, target) == tmp_path / "image2.png"

    def test_source_image_path_names_the_substitute_of_an_unreadable_yolo_image(self, tmp_path, mocker):
        """``YoloDetection`` is no ``CocoDetection``, yet its ``root`` and COCO API resolve the substitute's path."""
        mocker.patch.object(io_utils.time, "sleep")
        image_dir, label_dir, data_file = _write_yolo_dataset(tmp_path)
        (image_dir / "image1.png").write_bytes(b"not an image")
        dataset = YoloDetection(str(image_dir), str(label_dir), str(data_file), transforms=None, skip_unreadable=True)
        _, target = dataset[1]

        assert RFDETRDataModule._source_image_path(dataset, target).name == "image2.png"


class TestSetup:
    """Setup(stage) builds the correct dataset(s) for each PTL stage."""

    def _setup_with_mock(self, tmp_path, stage, dataset_file="roboflow", **train_overrides):
        """Helper: construct DataModule and call setup(stage) with build_dataset mocked."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, dataset_file=dataset_file, **train_overrides)
        dm = RFDETRDataModule(mc, tc)
        fake_train = _fake_dataset(100)
        fake_val = _fake_dataset(20)
        fake_test = _fake_dataset(10)
        datasets = {"train": fake_train, "val": fake_val, "test": fake_test}

        def _build(image_set, args, resolution):
            return datasets[image_set]

        with patch("rfdetr.training.module_data.build_dataset", side_effect=_build):
            dm.setup(stage)
        return dm, fake_train, fake_val, fake_test

    def test_fit_builds_train_and_val(self, tmp_path):
        """Setup('fit') populates both _dataset_train and _dataset_val."""
        dm, fake_train, fake_val, _ = self._setup_with_mock(tmp_path, "fit")
        assert dm._dataset_train is fake_train
        assert dm._dataset_val is fake_val
        assert dm._dataset_test is None

    def test_validate_builds_only_val(self, tmp_path):
        """Setup('validate') populates only _dataset_val."""
        dm, _, fake_val, _ = self._setup_with_mock(tmp_path, "validate")
        assert dm._dataset_train is None
        assert dm._dataset_val is fake_val
        assert dm._dataset_test is None

    @pytest.mark.parametrize("enabled", [True, False])
    @pytest.mark.parametrize(
        ("stage", "splits"),
        [("fit", ["train", "val"]), ("validate", ["val"]), ("test", ["test"]), ("predict", ["val"])],
    )
    def test_skip_unreadable_images_reaches_the_builder_of_every_split(self, mocker, tmp_path, stage, splits, enabled):
        """``setup(stage)`` hands every split builder a namespace that carries ``skip_unreadable_images``."""
        train_config = _base_train_config(tmp_path, dataset_file="yolo", skip_unreadable_images=enabled)
        dm = _build_datamodule(train_config=train_config)
        build = mocker.patch(
            "rfdetr.training.module_data.build_dataset",
            side_effect=lambda image_set, args, resolution: _fake_dataset(10),
        )

        dm.setup(stage)

        assert [(call.args[0], call.args[1].skip_unreadable_images) for call in build.call_args_list] == [
            (split, enabled) for split in splits
        ]

    @pytest.mark.parametrize("dataset_file", ["roboflow", "yolo"])
    def test_test_stage_uses_test_split(self, tmp_path, dataset_file):
        """Setup('test') requests the 'test' split for both Roboflow and YOLO datasets."""
        dm, _, _, fake_test = self._setup_with_mock(tmp_path, "test", dataset_file=dataset_file)
        assert dm._dataset_test is fake_test

    @pytest.mark.parametrize("dataset_file", ["roboflow", "yolo"])
    def test_test_stage_falls_back_to_val_without_test_split(self, mocker, tmp_path, dataset_file):
        """Setup('test') falls back to 'val' when the dataset declares no test split.

        A ``dataset_file="roboflow"`` dataset whose detected format is YOLO-style (``build_roboflow`` ->
        ``build_roboflow_from_yolo``, a common Roboflow export format) routes through the same builder as
        ``dataset_file="yolo"`` and can raise the same ``YoloSplitUnavailableError`` -- Roboflow's export UI does not
        require a test split.
        """
        dm = _build_datamodule(train_config=_base_train_config(tmp_path, dataset_file=dataset_file))
        fake_val = _fake_dataset(20)

        def _build(image_set, args, resolution):
            if image_set == "test":
                raise YoloSplitUnavailableError(str(tmp_path / "test" / "images"))
            return fake_val

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=_build)
        dm.setup("test")

        assert dm._dataset_test is fake_val

    def _write_yolo_dataset_without_test_split(self, dataset_dir: Path) -> None:
        """Write an on-disk YOLO dataset with one ``train/`` image and two ``valid/`` images, no ``test/`` split.

        The split sizes differ so that a length assertion on the dataset built for the ``test`` stage distinguishes the
        ``valid`` fallback from an accidental ``train`` one.
        """
        for split, image_count in (("train", 1), ("valid", 2)):
            (dataset_dir / split / "images").mkdir(parents=True)
            (dataset_dir / split / "labels").mkdir(parents=True)
            for idx in range(image_count):
                image_path = dataset_dir / split / "images" / f"sample{idx}.png"
                Image.new("RGB", (8, 6), color=(255, 255, 255)).save(image_path)
                (dataset_dir / split / "labels" / f"sample{idx}.txt").write_text(
                    "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
        (dataset_dir / "data.yaml").write_text("names:\n  - person\n", encoding="utf-8")

    def test_test_stage_roboflow_yolo_format_falls_back_to_val_end_to_end(
        self, tmp_path, caplog, propagate_rf_detr_logs
    ):
        """A real, unmocked Roboflow-YOLO export without a ``test/`` split falls back to ``valid/``.

        Exercises ``detect_roboflow_format`` -> ``build_roboflow_from_yolo`` end to end, not just the
        ``_build_test_dataset`` control flow around a mocked ``build_dataset``.
        """
        dataset_dir = tmp_path / "dataset"
        self._write_yolo_dataset_without_test_split(dataset_dir)
        dm = _build_datamodule(
            model_config=_base_model_config(num_classes=1),
            train_config=_base_train_config(tmp_path, dataset_file="roboflow", dataset_dir=str(dataset_dir)),
        )

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            dm.setup("test")

        assert isinstance(dm._dataset_test, YoloDetection)
        assert len(dm._dataset_test) == 2
        assert any("No resolvable 'test' split" in record.getMessage() for record in caplog.records)

    def test_test_stage_plain_yolo_falls_back_to_val_end_to_end(self, tmp_path):
        """A real, unmocked ``dataset_file="yolo"`` dataset without a ``test/`` split falls back to ``valid/``.

        Unlike the ``roboflow`` route, this one never runs ``detect_roboflow_format``: ``build_dataset`` dispatches
        straight to ``build_roboflow_from_yolo``.
        """
        dataset_dir = tmp_path / "dataset"
        self._write_yolo_dataset_without_test_split(dataset_dir)
        dm = _build_datamodule(
            model_config=_base_model_config(num_classes=1),
            train_config=_base_train_config(tmp_path, dataset_file="yolo", dataset_dir=str(dataset_dir)),
        )

        dm.setup("test")

        assert isinstance(dm._dataset_test, YoloDetection)
        assert len(dm._dataset_test) == 2

    @pytest.mark.parametrize("dataset_file", ["roboflow", "yolo"])
    def test_test_stage_propagates_broken_test_split(self, mocker, tmp_path, dataset_file):
        """Setup('test') propagates builder failures after a test split is resolved."""
        dm = _build_datamodule(train_config=_base_train_config(tmp_path, dataset_file=dataset_file))

        def _build(image_set, args, resolution):
            if image_set == "test":
                raise FileNotFoundError("declared test annotation file is broken")
            return _fake_dataset(20)

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=_build)
        with pytest.raises(FileNotFoundError) as exc_info:
            dm.setup("test")
        assert "declared test annotation file is broken" in str(exc_info.value)

    @pytest.mark.parametrize(
        "dataset_file, dataset_label",
        [pytest.param("roboflow", "Roboflow", id="roboflow"), pytest.param("yolo", "YOLO", id="yolo")],
    )
    def test_test_stage_warns_when_falling_back_to_val(self, mocker, tmp_path, dataset_file, dataset_label):
        """The test-to-val fallback is logged at WARNING rather than applied silently."""
        mock_logger = mocker.patch("rfdetr.training.module_data.logger")
        dm = _build_datamodule(train_config=_base_train_config(tmp_path, dataset_file=dataset_file))

        def _build(image_set, args, resolution):
            if image_set == "test":
                raise YoloSplitUnavailableError(str(tmp_path / "test" / "images"))
            return _fake_dataset(20)

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=_build)
        dm.setup("test")

        mock_logger.warning.assert_called_once_with(
            "No resolvable 'test' split for this %s dataset (%s); evaluating the 'val' split instead.",
            dataset_label,
            str(tmp_path / "test" / "images"),
        )

    def test_test_stage_coco_uses_val_split(self, mocker, coco_datamodule):
        """Setup('test') falls back to 'val' for COCO, whose test2017 split is unlabelled test-dev."""
        requested_splits = []

        def _build(image_set, args, resolution):
            requested_splits.append(image_set)
            return _fake_dataset(10)

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=_build)
        coco_datamodule.setup("test")

        assert "val" in requested_splits
        assert "test" not in requested_splits

    def test_test_stage_does_not_rebuild_after_val_fallback(self, tmp_path):
        """A second setup('test') reuses the val dataset resolved by the first fallback."""
        dm = _build_datamodule(train_config=_base_train_config(tmp_path, dataset_file="yolo"))
        fake_val = _fake_dataset(20)

        def _build(image_set, args, resolution):
            if image_set == "test":
                raise YoloSplitUnavailableError(str(tmp_path / "test" / "images"))
            return fake_val

        with patch("rfdetr.training.module_data.build_dataset", side_effect=_build):
            dm.setup("test")
        with patch("rfdetr.training.module_data.build_dataset") as mock_build:
            dm.setup("test")
            mock_build.assert_not_called()

        assert dm._dataset_test is fake_val

    def test_fit_does_not_rebuild_if_already_set(self, mocker, tmp_path):
        """Setup('fit') skips building if datasets are already populated."""
        mock_build = mocker.patch("rfdetr.training.module_data.build_dataset")
        dm = _build_datamodule(tmp_path=tmp_path)
        existing_train = _fake_dataset(50)
        existing_val = _fake_dataset(10)
        dm._dataset_train = existing_train
        dm._dataset_val = existing_val

        dm.setup("fit")
        mock_build.assert_not_called()

        assert dm._dataset_train is existing_train
        assert dm._dataset_val is existing_val

    def test_predict_stage_builds_val_dataset(self, tmp_path):
        """Setup('predict') populates _dataset_val with the 'val' split."""
        dm, _, fake_val, _ = self._setup_with_mock(tmp_path, "predict")
        assert dm._dataset_val is fake_val
        assert dm._dataset_train is None
        assert dm._dataset_test is None

    def test_predict_stage_does_not_rebuild_existing_val(self, mocker, tmp_path):
        """Setup('predict') skips building when _dataset_val is already set."""
        mock_build = mocker.patch("rfdetr.training.module_data.build_dataset")
        dm = _build_datamodule(tmp_path=tmp_path)
        existing_val = _fake_dataset(20)
        dm._dataset_val = existing_val

        dm.setup("predict")
        mock_build.assert_not_called()

        assert dm._dataset_val is existing_val


class TestKeypointAugmentationWarning:
    """Keypoint mode warns only for keypoint-unsafe GPU augmentation."""

    def _build_dm(self, tmp_path, *, use_grouppose_keypoints: bool, augmentation_backend: str = "cpu"):
        mc = _base_model_config(
            use_grouppose_keypoints=use_grouppose_keypoints,
            num_keypoints_per_class=[17] if use_grouppose_keypoints else [],
        )
        tc = _base_train_config(tmp_path, augmentation_backend=augmentation_backend)
        return RFDETRDataModule(mc, tc)

    @pytest.mark.parametrize("use_grouppose_keypoints", [True, False])
    def test_cpu_augmentation_emits_no_keypoint_warning(self, mocker, tmp_path, use_grouppose_keypoints):
        """Setup('fit') does not warn about keypoint augmentation on the CPU backend, in keypoint or detection mode."""
        dm = self._build_dm(tmp_path, use_grouppose_keypoints=use_grouppose_keypoints, augmentation_backend="cpu")
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.setup("fit")

        assert not [w for w in caught if "Keypoint mode" in str(w.message)]

    @_requires_kornia
    def test_keypoint_mode_gpu_augmentation_builds_pipeline(self, mocker, tmp_path: Path) -> None:
        """Setup('fit') builds a keypoint-enabled Kornia pipeline when the requested GPU backend is ready."""
        dm = self._build_dm(tmp_path, use_grouppose_keypoints=True, augmentation_backend="gpu")
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        build = mocker.patch("rfdetr.datasets.kornia_transforms.build_kornia_pipeline", wraps=build_kornia_pipeline)

        dm.setup("fit")

        assert dm._resolved_augmentation_backend == AugmentationBackend.KORNIA
        build.assert_called_once()
        assert build.call_args.kwargs["with_keypoints"] is True
        assert dm._kornia_pipeline is not None

    @_requires_kornia
    def test_keypoint_mode_auto_can_select_kornia(self, mocker, tmp_path: Path) -> None:
        """Auto backend can resolve to Kornia for keypoints on a CUDA host."""
        dm = self._build_dm(tmp_path, use_grouppose_keypoints=True, augmentation_backend="auto")
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        setup_pipeline = mocker.patch.object(dm, "_setup_kornia_pipeline")

        dm.setup("fit")

        assert dm._resolved_augmentation_backend == AugmentationBackend.KORNIA
        setup_pipeline.assert_called_once()


@pytest.mark.parametrize(("loader_name", "dataset_attribute"), _EVAL_LOADER_DATASETS)
class TestEvalDataloaders:
    """Val/test/predict dataloaders sample sequentially, keep the last partial batch and forward the train config."""

    @staticmethod
    def _build_dm(tmp_path, dataset_attribute, batch_size=2, num_workers=0):
        """Return a DataModule whose ``dataset_attribute`` holds a 50-item fake dataset.

        Examples:
            >>> dm = TestEvalDataloaders._build_dm(Path("dataset-root"), "_dataset_test", batch_size=4)
            >>> dm.train_config.batch_size, len(dm._dataset_test)
            (4, 50)
        """
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, batch_size=batch_size, num_workers=num_workers)
        dm = RFDETRDataModule(mc, tc)
        setattr(dm, dataset_attribute, _fake_dataset(50))
        return dm

    def test_returns_dataloader(self, tmp_path, loader_name, dataset_attribute):
        """The loader factory returns a DataLoader instance."""
        loader = getattr(self._build_dm(tmp_path, dataset_attribute), loader_name)()
        assert isinstance(loader, DataLoader)

    def test_uses_sequential_sampler(self, tmp_path, loader_name, dataset_attribute):
        """The loader samples sequentially, so ordering is deterministic."""
        loader = getattr(self._build_dm(tmp_path, dataset_attribute), loader_name)()
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_drop_last_false(self, tmp_path, loader_name, dataset_attribute):
        """The loader does not drop the last incomplete batch."""
        loader = getattr(self._build_dm(tmp_path, dataset_attribute), loader_name)()
        assert loader.drop_last is False

    def test_batch_size_forwarded(self, tmp_path, loader_name, dataset_attribute):
        """The DataLoader's batch size matches the train config."""
        loader = getattr(self._build_dm(tmp_path, dataset_attribute, batch_size=6), loader_name)()
        assert loader.batch_size == 6

    def test_num_workers_forwarded(self, tmp_path, loader_name, dataset_attribute):
        """The DataLoader's num_workers matches the train config."""
        loader = getattr(self._build_dm(tmp_path, dataset_attribute, num_workers=0), loader_name)()
        assert loader.num_workers == 0


class TestEvalBatchSize:
    """eval_batch_size decouples the val/test/predict DataLoaders from the train micro-batch size."""

    def _setup_dm(
        self,
        tmp_path: Path,
        batch_size: int | str = 2,
        eval_batch_size: int | None = None,
        grad_accum_steps: int = 1,
        dataset_length: int = 50,
    ) -> RFDETRDataModule:
        """Build a data module with every loader dataset injected.

        Examples:
            >>> datamodule = TestEvalBatchSize()._setup_dm(Path("/tmp"), batch_size=4, eval_batch_size=8)
            >>> datamodule.train_config.batch_size, datamodule.train_config.eval_batch_size
            (4, 8)
        """
        mc = _base_model_config()
        tc = _base_train_config(
            tmp_path,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            grad_accum_steps=grad_accum_steps,
            num_workers=0,
        )
        dm = RFDETRDataModule(mc, tc)
        dm._dataset_train = _fake_dataset(dataset_length)
        dm._dataset_val = _fake_dataset(dataset_length)
        dm._dataset_test = _fake_dataset(dataset_length)
        return dm

    @pytest.mark.parametrize("loader_name", _EVAL_LOADERS)
    @pytest.mark.parametrize(
        "batch_size, eval_batch_size, expected_batch_size",
        [
            pytest.param(6, None, 6, id="defaults-to-train-batch-size"),
            pytest.param(2, 16, 16, id="explicit-value-overrides-train-batch-size"),
            pytest.param("auto", 8, 8, id="explicit-value-works-with-unresolved-auto-batch-size"),
        ],
    )
    def test_eval_loader_batch_size(
        self, tmp_path: Path, loader_name: str, batch_size, eval_batch_size, expected_batch_size
    ) -> None:
        """Every eval DataLoader's batch_size follows eval_batch_size when set, else the train batch_size."""
        dm = self._setup_dm(tmp_path, batch_size=batch_size, eval_batch_size=eval_batch_size)
        loader = getattr(dm, loader_name)()
        assert loader.batch_size == expected_batch_size

    def test_train_dataloader_keeps_train_batch_size(self, tmp_path: Path) -> None:
        """The training DataLoader ignores eval_batch_size and keeps the configured train batch size."""
        dm = self._setup_dm(tmp_path, batch_size=2, eval_batch_size=16)
        loader = dm.train_dataloader()
        assert loader.batch_sampler.batch_size == 2

    def test_train_dataloader_grad_accum_alignment_unaffected(self, tmp_path: Path) -> None:
        """Train-side grad-accum padding still aligns to batch_size * grad_accum_steps, not eval_batch_size."""
        dm = self._setup_dm(tmp_path, batch_size=2, eval_batch_size=16, grad_accum_steps=4, dataset_length=50)
        loader = dm.train_dataloader()
        assert len(loader.dataset) % (2 * 4) == 0

    def test_unresolved_auto_batch_size_still_raises_without_explicit_value(self, tmp_path: Path) -> None:
        """Without eval_batch_size, an unresolved batch_size='auto' still fails eval loader construction."""
        dm = self._setup_dm(tmp_path, batch_size="auto", eval_batch_size=None)
        with pytest.raises(RuntimeError) as exc_info:
            dm.val_dataloader()
        assert "was not resolved" in str(exc_info.value)


class TestClassNames:
    """class_names property extracts names from COCO dataset annotations."""

    def test_returns_none_before_setup(self, fixture_training_setup):
        """class_names is None when no dataset has been set up."""
        _, _, dm = fixture_training_setup
        assert dm.class_names is None

    def test_returns_names_from_train_dataset(self, tmp_path):
        """class_names reads from _dataset_train.coco.cats when available."""
        dm = _build_datamodule(tmp_path=tmp_path)
        dm._dataset_train = _fake_dataset(50, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_names_from_val_dataset_when_train_missing(self, tmp_path):
        """class_names falls back to _dataset_val when _dataset_train has no COCO."""
        dm = _build_datamodule(tmp_path=tmp_path)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=True)
        assert dm.class_names == ["cat", "dog"]

    def test_returns_none_when_no_coco_attribute(self, tmp_path):
        """class_names returns None when no dataset has a coco attribute."""
        dm = _build_datamodule(tmp_path=tmp_path)
        dm._dataset_train = _fake_dataset(50, with_coco=False)
        dm._dataset_val = _fake_dataset(20, with_coco=False)
        assert dm.class_names is None

    def test_class_names_sorted_by_category_id(self, tmp_path):
        """class_names are sorted by COCO category ID."""
        dm = _build_datamodule(tmp_path=tmp_path)
        dataset = _fake_dataset(50)
        coco = MagicMock()
        # Deliberately out of order IDs
        coco.cats = {3: {"name": "zebra"}, 1: {"name": "ant"}, 2: {"name": "bee"}}
        dataset.coco = coco
        dm._dataset_train = dataset
        assert dm.class_names == ["ant", "bee", "zebra"]

    def test_class_names_follow_label_slots_when_categories_are_remapped(self, tmp_path):
        """class_names should preserve empty label slots so prediction class IDs map to the right names."""
        dm = _build_datamodule(tmp_path=tmp_path)
        dataset = _fake_dataset(50)
        coco = MagicMock()
        coco.cats = {0: {"name": "person"}}
        dataset.coco = coco
        dataset.label2cat = {1: 0}
        dm._dataset_train = dataset

        assert dm.class_names == ["", "person"]

    def test_dataset_with_class_names_attribute_but_not_webdataset_is_ignored(self, tmp_path):
        """A dataset merely exposing a `class_names` attribute, without being a WebDatasetDetection, is not used.

        Regression test: this property used to duck-type on `getattr(dataset, "class_names", None)`, so any
        dataset happening to carry an attribute of that name would satisfy it without the guarantee a real
        `WebDatasetDetection` gives -- label-indexed names read from its packed shard index. `_FakeDataset` has
        no `class_names` of its own, so setting one directly on the instance stands in for that broader surface.
        """
        dm = _build_datamodule(tmp_path=tmp_path)
        dataset = _fake_dataset(50, with_coco=False)
        dataset.class_names = ["decoy"]
        dm._dataset_train = dataset
        assert dm.class_names is None


class TestClassNamesMatchDataset:
    """Setup("fit") checks explicit ``class_names`` against the class names read from the dataset."""

    @staticmethod
    def _fit_setup(dm: RFDETRDataModule) -> None:
        """Run ``setup("fit")`` on datasets whose COCO categories are ``cat`` and ``dog``.

        The ``label2cat`` mapping makes these datasets the remapped kind a Roboflow COCO export builds, where labels
        are contiguous indices into the category list rather than the raw category ids — the only kind whose names are
        indexed by label, and so the only kind the check compares against ``class_names`` at all.

        Args:
            dm: Data module to set up.

        Examples:
            >>> dm = RFDETRDataModule(_base_model_config(), _base_train_config())
            >>> TestClassNamesMatchDataset._fit_setup(dm)
            >>> dm.class_names
            ['cat', 'dog']
        """
        datasets = {"train": _fake_dataset(10, with_coco=True), "val": _fake_dataset(4, with_coco=True)}
        for dataset in datasets.values():
            dataset.label2cat = {0: 1, 1: 2}
        with patch("rfdetr.training.module_data.build_dataset", side_effect=lambda split, *_: datasets[split]):
            dm.setup("fit")

    def test_roboflow_root_category_in_front_raises(self, tmp_path: Path) -> None:
        """Names read from every entry of a Roboflow export's categories shift each class by one, so fit stops."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=["animals", "cat", "dog"]))

        with pytest.raises(ValueError, match=r"'animals'.*shifted by one.*class_names=\['cat', 'dog'\]"):
            self._fit_setup(dm)

    @pytest.mark.parametrize(
        ("class_names", "listed_extras"),
        [
            pytest.param(["animals", "pets", "cat", "dog"], "'animals', 'pets'", id="two-parents-in-front"),
            pytest.param(["cat", "pets", "dog"], "'pets'", id="parent-between-classes"),
        ],
    )
    def test_extra_names_above_a_class_raise(self, tmp_path: Path, class_names: list[str], listed_extras: str) -> None:
        """Any unannotated parent in the list shifts the classes below it, wherever it sits, so fit stops.

        ``filter_parent_categories`` drops every unannotated grouping category, not only a leading one, so a list read
        from all of them can carry several parents and can carry one between two real classes. Comparing only against
        ``names[1:]`` saw neither shape and let the run start with each affected class mislabelled.
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=class_names))

        with pytest.raises(ValueError) as exc_info:
            self._fit_setup(dm)

        assert f"The extra entries — {listed_extras} —" in str(exc_info.value)

    @pytest.mark.parametrize(
        "class_names",
        [
            pytest.param([], id="empty"),
            pytest.param(["dog"], id="one-shorter"),
            pytest.param(["cat", "dog", "bird"], id="one-longer"),
        ],
    )
    def test_other_length_mismatch_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, propagate_rf_detr_logs: None, class_names: list[str]
    ) -> None:
        """A list of another length can't line up with the labels either, but may be deliberate, so it only warns.

        Covers both directions of a plain length mismatch — empty, shorter, and longer than the dataset's own names —
        none of which carries an extra entry above a real class, so each falls to the generic length-mismatch warning
        rather than the extra-names raise. The warning reaches the ``rf-detr`` logger only: a second ``warnings.warn``
        used to repeat it with a stacklevel that pointed inside Lightning's hook dispatcher instead of the caller's own
        code.
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=class_names))

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            self._fit_setup(dm)

        assert any(
            f"class_names has {len(class_names)} entries but the dataset has 2 classes" in record.getMessage()
            for record in caplog.records
        )

    def test_empty_dataset_categories_short_circuits(self, mocker, tmp_path: Path) -> None:
        """An explicit class_names passes silently when the dataset itself carries no categories.

        ``_check_class_names_match_dataset``'s ``not dataset_class_names`` short-circuit (module_data.py) was
        otherwise untested: a dataset with no COCO categories resolves ``class_names`` to ``[]``, which must skip
        the comparison outright rather than raise or warn on the explicit list.
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=["cat", "dog"]))
        datasets = {"train": _fake_dataset(10, with_coco=True), "val": _fake_dataset(4, with_coco=True)}
        for dataset in datasets.values():
            dataset.coco.cats = {}

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda split, *_: datasets[split])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dm.setup("fit")

        assert dm.class_names == []

    def test_same_length_permutation_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, propagate_rf_detr_logs: None
    ) -> None:
        """The dataset's own names in another order warn, because the equal length otherwise hides the swap.

        ``["dog", "cat"]`` against a ``cat``/``dog`` dataset labels every prediction of one as the other, in checkpoints
        and in ``predict()`` output. It only warns: renaming one class to another's name is legal, and nothing here can
        tell that apart from a list pasted in the wrong order.
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=["dog", "cat"]))

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            self._fit_setup(dm)

        assert any("in a different order" in record.getMessage() for record in caplog.records)

    @pytest.mark.parametrize(
        "class_names",
        [
            None,
            pytest.param(["cat", "dog"], id="same"),
            pytest.param(["Katze", "Hund"], id="renamed"),
        ],
    )
    def test_matching_length_passes(self, tmp_path: Path, class_names: list[str] | None) -> None:
        """Unset, identical or renamed names of the right length set up without a warning."""
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=class_names))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._fit_setup(dm)

        assert dm.class_names == ["cat", "dog"]

    def test_raw_category_id_dataset_is_not_compared(self, mocker, tmp_path: Path) -> None:
        """A raw-id dataset labels objects by category id, so a class_names indexed by those ids must be accepted.

        ``dataset_file="coco"`` builds ``CocoDetection`` without remapping, leaving labels equal to the 1-based COCO
        category ids while the names read back are ordered by id. The ``class_names`` that is correct for such a run
        therefore carries a placeholder for the unused id 0, which the length/shift comparison used to mistake for a
        Roboflow export's extra root category and reject with the off-by-one list as its remedy (#1572).
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=["", "cat", "dog"]))
        datasets = {"train": _fake_dataset(10, with_coco=True), "val": _fake_dataset(4, with_coco=True)}

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda split, *_: datasets[split])

        dm.setup("fit")

        assert dm.class_names == ["cat", "dog"]

    def test_dataset_names_with_empty_slot_warn_without_raising(
        self, mocker, tmp_path: Path, caplog: pytest.LogCaptureFixture, propagate_rf_detr_logs: None
    ) -> None:
        """A keypoint dataset's unnamed background slot is a length difference like any other: it warns, never raises.

        An empty name used to skip the comparison outright, which also disabled it for a dataset carrying a category
        genuinely named ``""``. Whether the labels are remapped now decides what is comparable, so the reserved slot
        only means the two lists differ in length — worth saying, not worth stopping a run over, because keypoint
        ``class_names`` follow the model's own slot convention.
        """
        dm = RFDETRDataModule(_base_model_config(), _base_train_config(tmp_path, class_names=["dog"]))
        datasets = {"train": _fake_dataset(10, with_coco=True), "val": _fake_dataset(4, with_coco=True)}
        for dataset in datasets.values():
            dataset.label2cat = {1: 2}
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda split, *_: datasets[split])

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            dm.setup("fit")

        assert any("dataset has 2 classes ['', 'dog']" in record.getMessage() for record in caplog.records)


class TestSegmentationSupport:
    """DataModule accepts SegmentationTrainConfig without errors."""

    def test_init_with_seg_train_config(self, base_model_config, seg_train_config):
        """RFDETRDataModule can be constructed with a SegmentationTrainConfig."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        dm = RFDETRDataModule(mc, tc)
        assert dm.train_config is tc
        assert dm.model_config.segmentation_head is True

    def test_seg_args_have_mask_loss_coefs(self, base_model_config, seg_train_config):
        """Segmentation-specific loss coefficients are present on train_config."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        dm = RFDETRDataModule(mc, tc)
        assert dm.train_config.mask_ce_loss_coef == pytest.approx(5.0)
        assert dm.train_config.mask_dice_loss_coef == pytest.approx(5.0)


def _build_config_datamodule(tmp_path, model_config_class, **train_overrides):
    """Return an ``RFDETRDataModule`` for *model_config_class* whose train config carries *train_overrides*.

    Examples:
        >>> import pathlib, tempfile
        >>> dm = _build_config_datamodule(pathlib.Path(tempfile.mkdtemp()), RFDETRNanoConfig, multi_scale="off")
        >>> dm.model_config.use_grouppose_keypoints
        False
    """
    model_config = model_config_class(pretrain_weights=None, device="cpu")
    train_config = TrainConfig(
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        batch_size=2,
        **train_overrides,
    )
    return RFDETRDataModule(model_config, train_config)


def _build_keypoint_datamodule(tmp_path, **train_overrides):
    """Return a keypoint ``RFDETRDataModule`` whose train config carries *train_overrides*.

    Examples:
        >>> import pathlib, tempfile
        >>> dm = _build_keypoint_datamodule(pathlib.Path(tempfile.mkdtemp()), multi_scale="off")
        >>> dm.model_config.use_grouppose_keypoints, dm.train_config.multi_scale.value
        (True, 'off')
    """
    return _build_config_datamodule(tmp_path, RFDETRKeypointPreviewConfig, **train_overrides)


def _run_fit_setup(datamodule):
    """Run ``setup("fit")`` with CUDA and every augmentation backend reported available, datasets stubbed out.

    Examples:
        >>> import pathlib, tempfile
        >>> dm = _build_keypoint_datamodule(pathlib.Path(tempfile.mkdtemp()), augmentation_backend="torchvision")
        >>> _run_fit_setup(dm)
        >>> len(dm._dataset_train)
        2
    """
    dataset = torch.utils.data.TensorDataset(torch.zeros(2, 1))
    with (
        patch("rfdetr.training.module_data.build_dataset", return_value=dataset),
        patch("rfdetr.training.module_data._has_cuda_device", return_value=True),
        patch.object(AugmentationBackend, "_is_available", lambda self: True),
        patch.object(RFDETRDataModule, "_setup_kornia_pipeline"),
    ):
        datamodule.setup("fit")


class TestKorniaPaddedBatchGuard:
    """``setup("fit")`` rejects padded Kornia batches for detection and segmentation too."""

    @pytest.mark.parametrize(
        "model_config_class",
        [
            pytest.param(RFDETRNanoConfig, id="detection"),
            pytest.param(RFDETRSegNanoConfig, id="segmentation"),
        ],
    )
    @pytest.mark.parametrize(
        ("square_resize_div_64", "multi_scale"),
        [
            pytest.param(False, "off", id="aspect-ratio-resize"),
            pytest.param(True, "per-sample", id="per-sample-random-resize"),
        ],
    )
    @pytest.mark.parametrize("augmentation_backend", ["kornia", "auto"])
    def test_padded_collation_raises(
        self, tmp_path, model_config_class, square_resize_div_64, multi_scale, augmentation_backend
    ):
        """Boxes are normalized by the batch canvas, so smaller images would train on shrunken targets."""
        datamodule = _build_config_datamodule(
            tmp_path,
            model_config_class,
            augmentation_backend=augmentation_backend,
            square_resize_div_64=square_resize_div_64,
            multi_scale=multi_scale,
        )

        with pytest.raises(ValueError, match="does not support training with padded batches"):
            _run_fit_setup(datamodule)

    @pytest.mark.parametrize("multi_scale", ["off", "per-batch"])
    def test_uniform_square_batches_are_accepted(self, tmp_path, multi_scale):
        """The default square resize never pads, so the canvas is every image's own size."""
        datamodule = _build_config_datamodule(
            tmp_path,
            RFDETRNanoConfig,
            augmentation_backend="kornia",
            square_resize_div_64=True,
            multi_scale=multi_scale,
        )

        _run_fit_setup(datamodule)

        assert datamodule._dataset_train is not None

    def test_cpu_backend_with_padded_collation_is_accepted(self, tmp_path):
        """The CPU path normalizes boxes per image before collation, so padding does not trigger the guard."""
        datamodule = _build_config_datamodule(
            tmp_path,
            RFDETRNanoConfig,
            augmentation_backend="torchvision",
            square_resize_div_64=False,
            multi_scale="off",
        )

        _run_fit_setup(datamodule)

        assert datamodule._dataset_train is not None


class TestKeypointKorniaPaddedBatchGuard:
    """``setup("fit")`` rejects Kornia keypoint training whenever images are padded into a shared batch canvas."""

    @pytest.mark.parametrize(
        ("square_resize_div_64", "multi_scale"),
        [
            pytest.param(False, "off", id="aspect-ratio-resize"),
            pytest.param(True, "per-sample", id="per-sample-random-resize"),
        ],
    )
    def test_padded_collation_raises(self, tmp_path, square_resize_div_64, multi_scale):
        """Padded collation plus Kornia keypoints raises before any dataset is built.

        Kornia keypoints are normalized by the batch canvas; with padding, joints of smaller images would shrink toward
        the origin, so the combination must fail fast with an actionable message.
        """
        datamodule = _build_keypoint_datamodule(
            tmp_path,
            augmentation_backend="kornia",
            square_resize_div_64=square_resize_div_64,
            multi_scale=multi_scale,
        )

        with pytest.raises(ValueError) as exc_info:
            _run_fit_setup(datamodule)

        assert "does not support training with padded batches" in str(exc_info.value)

    @pytest.mark.parametrize("multi_scale", ["off", "per-batch"])
    def test_uniform_square_batches_are_accepted(self, tmp_path, multi_scale):
        """Square resize without per-sample scaling yields unpadded batches, so Kornia keypoint training proceeds.

        ``per-batch`` resizes every sample to the same scale and ``off`` keeps the fixed resolution, so the canvas
        equals every image's own size and normalization stays correct.
        """
        datamodule = _build_keypoint_datamodule(
            tmp_path, augmentation_backend="kornia", square_resize_div_64=True, multi_scale=multi_scale
        )

        _run_fit_setup(datamodule)

        assert datamodule._dataset_train is not None

    def test_cpu_backend_with_padded_collation_is_accepted(self, tmp_path):
        """The CPU path normalizes keypoints per image before collation, so padding does not trigger the guard.

        Pinning ``torchvision`` keeps augmentation on the CPU regardless of which optional backends are installed.
        """
        datamodule = _build_keypoint_datamodule(
            tmp_path, augmentation_backend="torchvision", square_resize_div_64=False, multi_scale="off"
        )

        _run_fit_setup(datamodule)

        assert datamodule._dataset_train is not None
