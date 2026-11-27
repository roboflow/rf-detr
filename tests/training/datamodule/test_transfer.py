# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for RFDETRDataModule batch transfer, augmentation backend resolution and GPU-side augmentation."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.utils.data

from rfdetr.config import AugmentationBackend
from rfdetr.datasets.aug_configs import AUG_CONFIG
from rfdetr.datasets.kornia_transforms import IMAGENET_MEAN, IMAGENET_STD, build_kornia_pipeline, collate_boxes
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import (
    NestedTensor,
    PackedTargets,
    nested_tensor_from_tensor_list,
    pack_targets,
)
from tests._markers import _cuda_marks, _requires_kornia
from tests.training.helpers import _base_model_config, _fake_dataset, _make_batch
from tests.training.helpers import _datamodule_train_config as _base_train_config

_CUDA_DEVICE = pytest.param("cuda", marks=_cuda_marks)


def _setup_fit_with_fake_datasets(datamodule: RFDETRDataModule) -> RFDETRDataModule:
    """Run ``setup("fit")`` with ``build_dataset`` stubbed to a 100-item train and a 20-item val fake dataset.

    Examples:
        >>> from tests.training.helpers import _build_datamodule
        >>> dm = _setup_fit_with_fake_datasets(_build_datamodule())
        >>> len(dm._dataset_train), len(dm._dataset_val)
        (100, 20)
    """
    datasets = {"train": _fake_dataset(100), "val": _fake_dataset(20)}
    with patch("rfdetr.training.module_data.build_dataset", side_effect=lambda image_set, *_: datasets[image_set]):
        datamodule.setup("fit")
    return datamodule


class TestPadTargetsToKorniaGuard:
    """The Kornia GPU pipeline's collate_boxes/unpack_boxes don't know about pad_targets_to's.

    ``valid`` key -- they rebuild their own real/filler mask from the padded box count, then strip the fillers back out,
    undoing the fixed row count the option exists for. `setup('fit')` rejects the combination instead of silently losing
    shape stability.
    """

    def _build_dm(self, tmp_path, *, pad_targets_to, augmentation_backend):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, pad_targets_to=pad_targets_to, augmentation_backend=augmentation_backend)
        return RFDETRDataModule(mc, tc)

    def test_pad_targets_to_with_gpu_augmentation_raises(self, mocker, tmp_path):
        """Setup('fit') should raise ValueError when pad_targets_to is combined with GPU augmentation."""
        dm = self._build_dm(tmp_path, pad_targets_to=12, augmentation_backend="gpu")
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        mocker.patch.object(dm, "_setup_kornia_pipeline")

        with pytest.raises(ValueError) as exc_info:
            dm.setup("fit")

        assert "does not support pad_targets_to" in str(exc_info.value)

    def test_pad_targets_to_with_cpu_augmentation_no_raise(self, mocker, tmp_path):
        """Setup('fit') should not raise when pad_targets_to is combined with CPU augmentation."""
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))
        dm = self._build_dm(tmp_path, pad_targets_to=12, augmentation_backend="cpu")

        dm.setup("fit")

        assert dm.train_config.pad_targets_to == 12


class TestTransferBatchToDevice:
    """Tests for RFDETRDataModule.transfer_batch_to_device().

    Verifies that NestedTensor samples and all target-dict tensors are correctly moved to the target device without
    unwrapping the NestedTensor into plain tensors.
    """

    def test_samples_transferred_to_target_device(self, fixture_training_setup):
        """Both tensors and mask in NestedTensor must land on the target device."""
        _, _, dm = fixture_training_setup
        samples, targets = _make_batch()
        device = torch.device("cpu")

        result_samples, _ = dm.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        assert result_samples.tensors.device == device
        assert result_samples.mask.device == device

    def test_targets_transferred_to_target_device(self, fixture_training_setup):
        """All tensor values in every target dict must be moved to the target device."""
        _, _, dm = fixture_training_setup
        samples, targets = _make_batch()
        device = torch.device("cpu")

        _, result_targets = dm.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        for t in result_targets:
            for v in t.values():
                assert v.device == device

    def test_returns_tuple_of_correct_length(self, fixture_training_setup):
        """Return value must be a (samples, targets) tuple to match batch contract."""
        _, _, dm = fixture_training_setup
        result = dm.transfer_batch_to_device(_make_batch(), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserves_nested_tensor_type(self, fixture_training_setup):
        """Device transfer must not unwrap NestedTensor into plain tensors."""
        _, _, dm = fixture_training_setup
        samples, targets = _make_batch()

        result_samples, _ = dm.transfer_batch_to_device((samples, targets), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result_samples, NestedTensor)

    def test_packed_targets_are_unpacked_to_a_plain_list_on_target_device(self, fixture_training_setup):
        """Packed targets (``TrainConfig.pack_targets=True``) reach downstream code as the plain per-sample dict list.

        ``on_after_batch_transfer`` mutates target dicts by key reassignment, and
        ``PackedTargets.__getitem__``/iteration rebuilds a fresh dict on every access, so a reassignment into an un-
        materialised ``PackedTargets`` would silently vanish on the next access.
        """
        _, _, dm = fixture_training_setup
        samples, plain_targets = _make_batch()
        packed_targets = pack_targets(plain_targets)
        assert isinstance(packed_targets, PackedTargets)
        device = torch.device("cpu")

        _, result_targets = dm.transfer_batch_to_device((samples, packed_targets), device, dataloader_idx=0)

        assert isinstance(result_targets, list)
        assert not isinstance(result_targets, PackedTargets)
        for t in result_targets:
            assert isinstance(t, dict)
            for v in t.values():
                assert v.device == device
        for original, rebuilt in zip(plain_targets, result_targets):
            for key, value in original.items():
                assert torch.equal(rebuilt[key], value)

    def test_packed_targets_are_materialised_without_a_whole_batch_device_copy(
        self, mocker, fixture_training_setup
    ) -> None:
        """Packed transfer must not create a device copy of every field before constructing per-sample tensors."""
        _, _, dm = fixture_training_setup
        samples, plain_targets = _make_batch()
        packed_targets = pack_targets(plain_targets)
        assert isinstance(packed_targets, PackedTargets)
        mocker.patch.object(
            PackedTargets, "to", side_effect=AssertionError("whole packed batch copied to the target device")
        )

        _, result_targets = dm.transfer_batch_to_device(
            (samples, packed_targets), torch.device("cpu"), dataloader_idx=0
        )

        assert isinstance(result_targets, list)


class TestBackendResolution:
    """Backend resolution selects Kornia, CPU, or raises depending on environment.

    All tests run on CPU CI by mocking fork-safe CUDA detection and the ``kornia`` import as needed.
    """

    def _build_dm_with_backend(self, tmp_path, augmentation_backend="cpu"):
        """Construct a DataModule with the given augmentation_backend."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, augmentation_backend=augmentation_backend)
        return RFDETRDataModule(mc, tc)

    def test_auto_no_cuda_falls_back_to_cpu(self, mocker, tmp_path):
        """Auto + no CUDA: _kornia_pipeline stays None, no error."""
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=False)
        dm = _setup_fit_with_fake_datasets(self._build_dm_with_backend(tmp_path, "auto"))
        assert getattr(dm, "_kornia_pipeline", None) is None, (
            "auto backend with no CUDA must not build a Kornia pipeline"
        )

    def test_auto_no_kornia_falls_back_to_cpu(self, mocker, tmp_path):
        """Auto + CUDA available but kornia not installed: fallback to CPU."""
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA)

        dm = _setup_fit_with_fake_datasets(self._build_dm_with_backend(tmp_path, "auto"))

        assert getattr(dm, "_kornia_pipeline", None) is None, (
            "auto backend with kornia missing must fall back to CPU (pipeline=None)"
        )

    def test_gpu_no_cuda_raises_runtime_error(self, mocker, tmp_path):
        """Gpu + no CUDA: must raise RuntimeError."""
        dm = self._build_dm_with_backend(tmp_path, "gpu")
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=False)

        with pytest.raises(RuntimeError) as exc_info:
            _setup_fit_with_fake_datasets(dm)

        assert "CUDA" in str(exc_info.value)

    def test_gpu_no_kornia_raises_import_error(self, mocker, tmp_path):
        """Gpu + CUDA but no kornia: must raise ImportError with install hint."""
        dm = self._build_dm_with_backend(tmp_path, "gpu")
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA)

        with pytest.raises(ImportError) as exc_info:
            _setup_fit_with_fake_datasets(dm)

        assert "rfdetr[augment]" in str(exc_info.value)

    def test_gpu_normalizer_cycles_statistics_over_model_channels(self, mocker, tmp_path):
        """A 4-channel model gets four means and stds on the GPU path, as the CPU ``Normalize`` gives it (#1727)."""
        model_config = _base_model_config(num_channels=4)
        train_config = _base_train_config(tmp_path, augmentation_backend="auto")
        dm = RFDETRDataModule(model_config, train_config)
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: True)
        mocker.patch("rfdetr.datasets.kornia_transforms.build_kornia_pipeline", return_value=MagicMock())
        build_normalize = mocker.patch("rfdetr.datasets.kornia_transforms.build_normalize", return_value=MagicMock())

        _setup_fit_with_fake_datasets(dm)

        build_normalize.assert_called_once_with(mean=(0.485, 0.456, 0.406, 0.485), std=(0.229, 0.224, 0.225, 0.229))

    def test_cpu_backend_builds_no_pipeline(self, tmp_path):
        """Default cpu backend: _kornia_pipeline stays None."""
        dm = _setup_fit_with_fake_datasets(self._build_dm_with_backend(tmp_path, "cpu"))
        assert getattr(dm, "_kornia_pipeline", None) is None, "cpu backend must never build a Kornia pipeline"

    def test_gpu_path_uses_aug_config_fallback(self, mocker, tmp_path):
        """When aug_config=None (default), GPU path passes AUG_CONFIG to build_kornia_pipeline."""
        dm = self._build_dm_with_backend(tmp_path, "auto")
        assert dm.train_config.aug_config is None, "precondition: aug_config must be None for this test"

        captured = {}

        def _fake_build_kornia(aug_cfg, resolution, with_masks=False):
            captured["aug_config"] = aug_cfg
            captured["with_masks"] = with_masks
            return MagicMock()

        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=True)
        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10))
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: True)
        mocker.patch("rfdetr.datasets.kornia_transforms.build_kornia_pipeline", side_effect=_fake_build_kornia)
        mocker.patch("rfdetr.datasets.kornia_transforms.build_normalize", return_value=MagicMock())

        dm.setup("fit")

        assert captured.get("aug_config") is AUG_CONFIG, (
            "GPU path must fall back to AUG_CONFIG when train_config.aug_config is None"
        )
        assert captured.get("with_masks") is True, "GPU path must transport the padding mask for detection batches"

    def test_auto_no_cuda_does_not_strip_cpu_normalize(self, mocker, tmp_path):
        """Auto + no CUDA: the backend resolves to cpu before dataset build so CPU Normalize is retained."""
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=False)
        dm = self._build_dm_with_backend(tmp_path, "auto")
        resolved_backends = {}

        def _spy_build(image_set, args, resolution):
            resolved_backends[image_set] = getattr(args, "augmentation_backend", "cpu")
            return _fake_dataset(10)

        mocker.patch("rfdetr.training.module_data.build_dataset", side_effect=_spy_build)

        dm.setup("fit")

        assert resolved_backends.get("train") == "cpu", (
            "auto + no CUDA must resolve to cpu before dataset build to preserve CPU Normalize"
        )


class TestOnAfterBatchTransfer:
    """on_after_batch_transfer applies Kornia augmentation only during training.

    Uses CPU tensors with a mocked pipeline — no real GPU or Kornia needed.
    """

    def _build_dm(self, tmp_path, segmentation_head=False):
        """Construct a DataModule for on_after_batch_transfer tests."""
        mc = _base_model_config(segmentation_head=segmentation_head)
        tc = _base_train_config(tmp_path)
        return RFDETRDataModule(mc, tc)

    def _attach_mock_trainer(self, dm, training=True):
        """Attach a mock trainer with the given training state to the DataModule."""
        mock_trainer = MagicMock(training=training)
        type(dm).trainer = property(lambda self: mock_trainer)
        return dm

    def _make_kornia_batch(self, batch_size=2, h=16, w=16):
        """Build a batch with xyxy boxes suitable for on_after_batch_transfer.

        Returns (NestedTensor, targets) where boxes are in absolute xyxy format and pixel values are in [0, 1] (pre-
        normalization).
        """
        tensors = torch.rand(batch_size, 3, h, w)  # [0, 1] range
        mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
        samples = NestedTensor(tensors, mask)
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0]], dtype=torch.float32),
                "labels": torch.tensor([1]),
                "area": torch.tensor([64.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor(i),
                "orig_size": torch.tensor([h, w]),
            }
            for i in range(batch_size)
        ]
        return samples, targets

    def _make_kornia_batch_with_masks(self, batch_size=2, h=16, w=16):
        """Build a ``_make_kornia_batch`` batch whose targets also carry a ``[1, H, W]`` bool instance mask each."""
        samples, targets = self._make_kornia_batch(batch_size, h, w)
        for target in targets:
            target["masks"] = torch.ones(1, h, w, dtype=torch.bool)
        return samples, targets

    def test_training_true_applies_augmentation(self, tmp_path):
        """When training=True and _kornia_pipeline is set, image/box outputs match CPU Normalize contract."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        img_aug = samples.tensors.clone()
        # Mock pipeline returns pixel-index boxes, the image, and transported padding mask.
        boxes_padded = torch.tensor([[[1.5, 1.5, 9.5, 9.5]]] * 2)
        padding_masks_aug = torch.zeros(2, 1, 16, 16, dtype=torch.float32)
        mock_pipeline = MagicMock(return_value=(img_aug, boxes_padded, padding_masks_aug))
        dm._kornia_pipeline = mock_pipeline

        # Normalize adds +1 so we can assert the normalization step is applied.
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x + 1.0)

        result_samples, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_called_once()
        dm._kornia_normalize.assert_called_once()
        assert torch.allclose(result_samples.tensors, img_aug + 1.0)
        assert len(result_targets) == 2
        for target in result_targets:
            boxes = target["boxes"]
            assert boxes.shape == (1, 4)
            assert torch.all(boxes >= 0.0)
            assert torch.all(boxes <= 1.0)
            torch.testing.assert_close(
                boxes[0], torch.tensor([0.375, 0.375, 0.5, 0.5], dtype=torch.float32), rtol=1e-4, atol=1e-6
            )

    @_requires_kornia
    @pytest.mark.parametrize("segmentation_head", [False, True])
    @pytest.mark.parametrize("flip", ["HorizontalFlip", "VerticalFlip"])
    def test_detection_box_edges_follow_flipped_pixels(
        self, tmp_path: Path, segmentation_head: bool, flip: str
    ) -> None:
        """Continuous box edges, full-frame boxes, and optional masks follow the same flip."""
        height, width = 16, 32
        dm = self._attach_mock_trainer(self._build_dm(tmp_path, segmentation_head=segmentation_head))
        dm._kornia_pipeline = build_kornia_pipeline({flip: {"p": 1.0}}, height, with_masks=True)
        dm._kornia_normalize = torch.nn.Identity()
        image = torch.zeros(1, 3, height, width)
        image[:, :, 3:9, 4:12] = 1.0
        boxes = torch.tensor([[4.0, 3.0, 12.0, 9.0], [0.0, 0.0, float(width), float(height)]])
        target = {"boxes": boxes, "labels": torch.tensor([1, 2]), "area": torch.tensor([48.0, 512.0])}
        if segmentation_head:
            target["masks"] = torch.cat((image[0, :1].bool(), torch.ones(1, height, width, dtype=torch.bool)))

        samples_out, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), [target]), 0)

        # Expected continuous edges are W-x2/W-x1 or H-y2/H-y1, not the pixel-center W-1-x/H-1-y.
        expected_box = [20.0, 3.0, 28.0, 9.0] if flip == "HorizontalFlip" else [4.0, 7.0, 12.0, 13.0]
        x1, y1, x2, y2 = expected_box
        expected_boxes = torch.tensor(
            [
                [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height],
                [0.5, 0.5, 1.0, 1.0],
            ]
        )
        torch.testing.assert_close(targets_out[0]["boxes"], expected_boxes)
        torch.testing.assert_close(targets_out[0]["labels"], target["labels"])
        torch.testing.assert_close(targets_out[0]["area"], target["area"])
        dimension = -1 if flip == "HorizontalFlip" else -2
        torch.testing.assert_close(samples_out.tensors, image.flip(dimension))
        if segmentation_head:
            torch.testing.assert_close(targets_out[0]["masks"], target["masks"].flip(dimension))

    @_requires_kornia
    def test_detection_identity_preserves_fractional_subpixel_and_padded_boxes(self, tmp_path: Path) -> None:
        """An identity pipeline preserves continuous edges and ignores padding for an empty image."""
        dm = self._attach_mock_trainer(self._build_dm(tmp_path))
        dm._kornia_pipeline = build_kornia_pipeline({}, 16, with_masks=True)
        dm._kornia_normalize = torch.nn.Identity()
        image = torch.zeros(2, 3, 16, 32)
        boxes = torch.tensor([[3.2, 2.1, 8.7, 6.4], [2.0, 2.0, 2.5, 2.5]])
        labels = torch.tensor([1, 2])
        targets = [
            {"boxes": boxes, "labels": labels},
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
        ]

        _, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), targets), 0)

        expected = torch.tensor(
            [[5.95 / 32, 4.25 / 16, 5.5 / 32, 4.3 / 16], [2.25 / 32, 2.25 / 16, 0.5 / 32, 0.5 / 16]]
        )
        torch.testing.assert_close(targets_out[0]["boxes"], expected)
        torch.testing.assert_close(targets_out[0]["labels"], labels)
        assert targets_out[1]["boxes"].shape == (0, 4)
        assert targets_out[1]["labels"].shape == (0,)
        torch.testing.assert_close(targets[0]["boxes"], boxes)

    @_requires_kornia
    @pytest.mark.parametrize("flip", ["HorizontalFlip", "VerticalFlip"])
    def test_detection_flip_preserves_fractional_and_subpixel_edges(self, tmp_path: Path, flip: str) -> None:
        """Flips reflect fractional edges through width/height without changing subpixel extent."""
        dm = self._attach_mock_trainer(self._build_dm(tmp_path))
        dm._kornia_pipeline = build_kornia_pipeline({flip: {"p": 1.0}}, 16, with_masks=True)
        dm._kornia_normalize = torch.nn.Identity()
        image = torch.zeros(1, 3, 16, 32)
        target = {"boxes": torch.tensor([[3.2, 2.1, 8.7, 6.4], [2.0, 2.0, 2.5, 2.5]]), "labels": torch.tensor([1, 2])}

        _, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), [target]), 0)

        expected = torch.tensor(
            [[5.95 / 32, 4.25 / 16, 5.5 / 32, 4.3 / 16], [2.25 / 32, 2.25 / 16, 0.5 / 32, 0.5 / 16]]
        )
        axis = 0 if flip == "HorizontalFlip" else 1
        expected[:, axis] = 1.0 - expected[:, axis]
        torch.testing.assert_close(targets_out[0]["boxes"], expected)
        torch.testing.assert_close(targets_out[0]["labels"], target["labels"])

    @_requires_kornia
    def test_detection_flip_handles_all_empty_targets(self, tmp_path: Path) -> None:
        """An all-empty detection batch keeps zero-length annotation tensors through the bridge."""
        dm = self._attach_mock_trainer(self._build_dm(tmp_path))
        dm._kornia_pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, 16, with_masks=True)
        dm._kornia_normalize = torch.nn.Identity()
        image = torch.zeros(1, 3, 16, 32)
        target = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}

        _, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), [target]), 0)

        assert targets_out[0]["boxes"].shape == (0, 4)
        assert targets_out[0]["labels"].shape == (0,)

    @_requires_kornia
    def test_detection_affine_scales_continuous_box_width(self, tmp_path: Path) -> None:
        """Scaling around the image center doubles fractional box extent without a one-pixel correction."""
        dm = self._attach_mock_trainer(self._build_dm(tmp_path))
        dm._kornia_pipeline = build_kornia_pipeline(
            {"Affine": {"scale": (2.0, 2.0), "rotate": (0.0, 0.0), "p": 1.0}}, 16, with_masks=True
        )
        dm._kornia_normalize = torch.nn.Identity()
        image = torch.zeros(1, 3, 16, 32)
        target = {"boxes": torch.tensor([[13.25, 5.5, 18.75, 10.5]]), "labels": torch.tensor([1])}

        _, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), [target]), 0)

        # Continuous image center is (16, 8): [13.25, 5.5, 18.75, 10.5] -> [10.5, 3, 21.5, 13].
        torch.testing.assert_close(targets_out[0]["boxes"], torch.tensor([[0.5, 0.5, 11 / 32, 10 / 16]]))

    @_requires_kornia
    @pytest.mark.parametrize("device", ["cpu", _CUDA_DEVICE])
    def test_pixel_affine_reaches_segmentation_training_batch(self, tmp_path: Path, device: str) -> None:
        """The real transfer hook moves pixels, boxes, instance masks, and padding masks together."""
        config = {"Affine": {"translate_px": {"x": (4, 4), "y": (0, 0)}, "p": 1.0}}
        dm = RFDETRDataModule(
            _base_model_config(segmentation_head=True),
            _base_train_config(tmp_path, aug_config=config, augmentation_backend="kornia"),
        )
        dm._resolved_augmentation_backend = AugmentationBackend.KORNIA
        dm._setup_kornia_pipeline()
        # The data module keeps translate_px off the GPU (it runs on the CPU before the resize), so build the GPU
        # pixel-affine pipeline directly to exercise the transfer hook's handling of it.
        dm._kornia_pipeline = build_kornia_pipeline(config, dm.model_config.resolution, with_masks=True)
        dm = self._attach_mock_trainer(dm, training=True)
        image = torch.zeros(1, 3, 64, 80, device=device)
        image[:, :, 30:34, 30:34] = 1.0
        padding = torch.zeros(1, 64, 80, device=device, dtype=torch.bool)
        padding[:, :, 70:] = True
        instance_mask = torch.zeros(1, 64, 80, device=device, dtype=torch.bool)
        instance_mask[:, 30:34, 30:34] = True
        target = {
            "boxes": torch.tensor([[30.0, 30.0, 34.0, 34.0]], device=device),
            "labels": torch.tensor([1], device=device),
            "area": torch.tensor([16.0], device=device),
            "iscrowd": torch.tensor([0], device=device),
            "masks": instance_mask,
        }

        samples_out, targets_out = dm.on_after_batch_transfer((NestedTensor(image, padding), [target]), 0)

        assert len(targets_out) == 1
        torch.testing.assert_close(
            targets_out[0]["boxes"],
            torch.tensor([[0.45, 0.5, 0.05, 0.0625]], device=device),
            rtol=0,
            atol=1e-5,
        )
        expected_mask = torch.zeros_like(instance_mask)
        expected_mask[:, 30:34, 34:38] = True
        torch.testing.assert_close(targets_out[0]["masks"], expected_mask, rtol=0, atol=0)
        expected_padding = torch.zeros_like(padding)
        expected_padding[:, :, 4:] = padding[:, :, :-4]
        torch.testing.assert_close(samples_out.mask, expected_padding, rtol=0, atol=0)
        mean = torch.tensor(IMAGENET_MEAN, device=device)
        std = torch.tensor(IMAGENET_STD, device=device)
        # The white square moved onto column 34 (input 1.0) and vacated column 30 (input 0.0); ImageNet-normalized.
        torch.testing.assert_close(samples_out.tensors[0, :, 30, 34], (1.0 - mean) / std, rtol=0, atol=1e-5)
        torch.testing.assert_close(samples_out.tensors[0, :, 30, 30], (0.0 - mean) / std, rtol=0, atol=1e-5)

    @_requires_kornia
    def test_pixel_affine_drops_box_and_mask_shifted_out_of_frame(self, tmp_path: Path) -> None:
        """A translated-out object must not remain as a zero-area training target."""
        config = {"Affine": {"translate_px": {"x": 60}, "p": 1.0}}
        dm = RFDETRDataModule(
            _base_model_config(segmentation_head=True),
            _base_train_config(tmp_path, aug_config=config, augmentation_backend="kornia"),
        )
        dm._resolved_augmentation_backend = AugmentationBackend.KORNIA
        dm._setup_kornia_pipeline()
        # The data module keeps translate_px off the GPU (it runs on the CPU before the resize), so build the GPU
        # pixel-affine pipeline directly to exercise the transfer hook's handling of it.
        dm._kornia_pipeline = build_kornia_pipeline(config, dm.model_config.resolution, with_masks=True)
        dm = self._attach_mock_trainer(dm, training=True)
        image = torch.zeros(1, 3, 64, 80)
        mask = torch.zeros(1, 64, 80, dtype=torch.bool)
        mask[:, 30:34, 30:34] = True
        target = {
            "boxes": torch.tensor([[30.0, 30.0, 34.0, 34.0]]),
            "labels": torch.tensor([1]),
            "area": torch.tensor([16.0]),
            "iscrowd": torch.tensor([0]),
            "masks": mask,
        }

        _, targets_out = dm.on_after_batch_transfer((NestedTensor(image, None), [target]), 0)

        for key in ("boxes", "labels", "area", "iscrowd", "masks"):
            assert targets_out[0][key].shape[0] == 0, key

    def test_training_uses_transformed_padding_mask(self, tmp_path) -> None:
        """The returned NestedTensor mask comes from the same Kornia geometry as the image."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        assert samples.mask is not None
        samples.mask[0, 8:, :] = True
        samples.mask[1, :, 12:] = True
        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        padding_masks_aug = torch.zeros(2, 1, 16, 16, dtype=torch.float32)
        padding_masks_aug[0, :, 5:, :] = 1.0
        padding_masks_aug[1, :, :, 3:] = 1.0
        dm._kornia_pipeline = MagicMock(return_value=(img_aug, boxes_padded, padding_masks_aug))
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        result_samples, _ = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        call_args, call_kwargs = dm._kornia_pipeline.call_args
        assert len(call_args) == 3
        assert not call_kwargs
        torch.testing.assert_close(call_args[2], samples.mask.unsqueeze(1).to(torch.float32), rtol=0, atol=0)
        assert result_samples.mask is not None
        assert result_samples.mask.dtype == torch.bool
        assert torch.equal(result_samples.mask, padding_masks_aug[:, 0].to(torch.bool))

    @_requires_kornia
    def test_perspective_warps_padding_mask_with_real_pipeline(self, tmp_path) -> None:
        """Perspective transports unequal-size batch padding through the real Kornia sequence."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)
        samples = nested_tensor_from_tensor_list([torch.ones(3, 48, 48), torch.ones(3, 64, 64)])
        assert samples.mask is not None
        targets = [
            {
                "boxes": torch.tensor([[4.0, 4.0, 36.0, 36.0]]),
                "labels": torch.tensor([1]),
                "area": torch.tensor([1024.0]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor(index),
                "orig_size": torch.tensor([size, size]),
            }
            for index, size in enumerate((48, 64))
        ]
        config = {"Perspective": {"scale": 0.4, "p": 1.0}}
        dm._kornia_pipeline = build_kornia_pipeline(config, 64, with_masks=True)
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)
        boxes_padded, _ = collate_boxes(targets, samples.tensors.device)
        reference_pipeline = build_kornia_pipeline(config, 64, with_masks=True)

        torch.manual_seed(7)
        _, _, expected_padding = reference_pipeline(
            samples.tensors,
            boxes_padded,
            samples.mask.unsqueeze(1).to(torch.float32),
        )
        torch.manual_seed(7)
        result_samples, _ = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert result_samples.mask is not None
        assert result_samples.mask.dtype == torch.bool
        assert torch.equal(result_samples.mask, expected_padding[:, 0].to(torch.bool))
        assert not torch.equal(result_samples.mask, samples.mask)

    def test_training_false_skips_augmentation(self, tmp_path):
        """When training=False, batch is returned unchanged."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=False)

        samples, targets = self._make_kornia_batch()
        mock_pipeline = MagicMock()
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock()

        result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_not_called()
        # Batch returned as-is
        result_samples, result_targets = result
        assert result_samples is samples
        assert result_targets is targets

    def test_segmentation_model_applies_augmentation_with_masks(self, tmp_path):
        """Phase 2: segmentation_head=True now calls pipeline with image, boxes, and masks."""
        dm = self._build_dm(tmp_path, segmentation_head=True)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch_with_masks()
        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        masks_aug = torch.ones(2, 2, 16, 16, dtype=torch.float32)
        masks_aug[:, 1] = 0.0

        mock_pipeline = MagicMock(return_value=(img_aug, boxes_padded, masks_aug))
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        result_samples, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        mock_pipeline.assert_called_once()
        call_args, call_kwargs = mock_pipeline.call_args
        assert len(call_args) == 3, "segmentation augmentation must call pipeline with image, boxes, and masks"
        assert not call_kwargs, "segmentation augmentation should not pass unexpected keyword arguments"

        masks_arg = call_args[2]
        assert isinstance(masks_arg, torch.Tensor), "third pipeline argument must be a masks tensor"
        assert masks_arg.dtype == torch.float32, "masks passed to pipeline must be float32"
        assert masks_arg.shape == (2, 2, 16, 16), "masks passed to pipeline must include instance and padding channels"
        assert "masks" in result_targets[0], "masks key must be present in output targets for segmentation"
        assert result_samples.mask is not None
        assert not result_samples.mask.any()

    def test_segmentation_masks_stay_in_sync_with_boxes(self, tmp_path):
        """Masks are filtered in sync with boxes: one instance removed → one mask removed."""
        dm = self._build_dm(tmp_path, segmentation_head=True)
        dm = self._attach_mock_trainer(dm, training=True)

        h, w = 16, 16
        tensors = torch.rand(1, 3, h, w)
        mask_nt = torch.zeros(1, h, w, dtype=torch.bool)
        samples = NestedTensor(tensors, mask_nt)
        targets = [
            {
                "boxes": torch.tensor([[2.0, 2.0, 8.0, 8.0], [10.0, 10.0, 14.0, 14.0]]),
                "labels": torch.tensor([1, 2]),
                "area": torch.tensor([36.0, 16.0]),
                "iscrowd": torch.tensor([0, 0]),
                "image_id": torch.tensor(0),
                "orig_size": torch.tensor([h, w]),
                "masks": torch.ones(2, h, w, dtype=torch.bool),
            }
        ]
        # Augmented: box 0 survives, box 1 becomes zero-area
        boxes_aug_out = torch.tensor([[[2.0, 2.0, 8.0, 8.0], [5.0, 5.0, 5.0, 5.0]]])
        masks_aug_out = torch.ones(1, 3, h, w, dtype=torch.float32)
        masks_aug_out[:, 2] = 0.0
        mock_pipeline = MagicMock(return_value=(tensors, boxes_aug_out, masks_aug_out))
        dm._kornia_pipeline = mock_pipeline
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        _, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert result_targets[0]["masks"].shape[0] == 1, (
            f"Expected 1 surviving mask (matching box), got {result_targets[0]['masks'].shape[0]}"
        )

    def test_returns_nested_tensor_in_batch(self, tmp_path):
        """Output batch still has NestedTensor as first element after augmentation."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        padding_masks_aug = torch.zeros(2, 1, 16, 16, dtype=torch.float32)
        dm._kornia_pipeline = MagicMock(return_value=(img_aug, boxes_padded, padding_masks_aug))
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        result_samples, _ = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert isinstance(result_samples, NestedTensor), f"Expected NestedTensor, got {type(result_samples).__name__}"

    def test_detection_gpu_augmentation_leaves_unrelated_keypoints_unchanged(self, tmp_path: Path) -> None:
        """The unchanged detection route does not enable keypoint geometry."""
        dm = self._build_dm(tmp_path)
        dm = self._attach_mock_trainer(dm, training=True)

        samples, targets = self._make_kornia_batch()
        keypoints = torch.tensor([[[3.0, 4.0, 2.0]]], dtype=torch.float32)
        targets[0]["keypoints"] = keypoints.clone()
        targets[1]["keypoints"] = keypoints.clone()
        input_keypoints = [target["keypoints"].clone() for target in targets]

        img_aug = samples.tensors.clone()
        boxes_padded = torch.tensor([[[2.0, 2.0, 10.0, 10.0]]] * 2)
        padding_masks_aug = torch.zeros(2, 1, 16, 16, dtype=torch.float32)
        dm._kornia_pipeline = MagicMock(return_value=(img_aug, boxes_padded, padding_masks_aug))
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)

        _, result_targets = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        for idx, target in enumerate(result_targets):
            torch.testing.assert_close(target["keypoints"], input_keypoints[idx], rtol=1e-4, atol=1e-6)

    @_requires_kornia
    @pytest.mark.parametrize("device_name", ["cpu", _CUDA_DEVICE])
    @pytest.mark.parametrize("height, width", [(16, 32), (32, 16)])
    def test_keypoint_mode_real_pipeline_flips_and_normalizes(
        self, tmp_path: Path, device_name: str, height: int, width: int
    ) -> None:
        """The public batch hook keeps box and joint geometry aligned on the requested device.

        Non-square images make a swapped width/height in the mirror or the normalization visible: x is mirrored and
        divided by the width, y is divided by the height.
        """
        mc = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[2])
        tc = _base_train_config(tmp_path, keypoint_flip_pairs=[0, 1])
        dm = self._attach_mock_trainer(RFDETRDataModule(mc, tc), training=True)
        dm._kornia_pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}},
            16,
            with_masks=True,
            include_keypoints=True,
            with_keypoints=True,
            keypoint_flip_pairs=[0, 1],
        )
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)
        samples, targets = self._make_kornia_batch(batch_size=1, h=height, w=width)
        targets[0]["keypoints"] = torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 1.0]]])
        device = torch.device(device_name)
        samples = NestedTensor(samples.tensors.to(device), samples.mask.to(device))
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        _, result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        # Mirrored x is width - x; the flip swaps the pair, so slot 0 holds the former slot-1 joint.
        expected_keypoints = torch.tensor(
            [[[(width - 6) / width, 7 / height, 1], [(width - 3) / width, 4 / height, 2]]], device=device
        )
        # Box [2, 2, 10, 10] mirrors to [width - 10, 2, width - 2, 10], then becomes normalized cxcywh.
        expected_boxes = torch.tensor([[(width - 6) / width, 6 / height, 8 / width, 8 / height]], device=device)
        torch.testing.assert_close(result[0]["keypoints"], expected_keypoints)
        torch.testing.assert_close(result[0]["boxes"], expected_boxes)

    @_requires_kornia
    def test_keypoint_mode_handles_all_empty_targets(self, tmp_path: Path) -> None:
        """An annotation-free batch still reaches Kornia without its zero-point reshape crash."""
        mc = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[2])
        tc = _base_train_config(tmp_path, keypoint_flip_pairs=[0, 1])
        dm = self._attach_mock_trainer(RFDETRDataModule(mc, tc), training=True)
        dm._kornia_pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}},
            16,
            with_masks=True,
            include_keypoints=True,
            with_keypoints=True,
            keypoint_flip_pairs=[0, 1],
        )
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)
        samples, targets = self._make_kornia_batch(batch_size=1)
        targets[0]["boxes"] = torch.zeros(0, 4)
        targets[0]["labels"] = torch.zeros(0, dtype=torch.long)
        targets[0]["keypoints"] = torch.zeros(0, 2, 3)

        _, result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert result[0]["boxes"].shape == (0, 4)
        assert result[0]["keypoints"].shape == (0, 2, 3)

    @_requires_kornia
    def test_keypoints_and_instance_masks_share_flip(self, tmp_path: Path) -> None:
        """A combined target keeps its instance mask and joints with the flipped box."""
        mc = _base_model_config(segmentation_head=True, use_grouppose_keypoints=True, num_keypoints_per_class=[2])
        tc = _base_train_config(tmp_path, keypoint_flip_pairs=[0, 1])
        dm = self._attach_mock_trainer(RFDETRDataModule(mc, tc), training=True)
        dm._kornia_pipeline = build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}},
            16,
            with_masks=True,
            include_keypoints=True,
            with_keypoints=True,
            keypoint_flip_pairs=[0, 1],
        )
        dm._kornia_normalize = MagicMock(side_effect=lambda x: x)
        samples, targets = self._make_kornia_batch_with_masks(batch_size=1)
        targets[0]["masks"][:, :, :8] = False
        targets[0]["keypoints"] = torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 1.0]]])

        _, result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        assert result[0]["masks"][:, :, :8].all()
        assert not result[0]["masks"][:, :, 8:].any()
        torch.testing.assert_close(result[0]["keypoints"], torch.tensor([[[10 / 16, 7 / 16, 1], [13 / 16, 4 / 16, 2]]]))

    def _build_dm_with_setup_pipeline(self, tmp_path: Path, keypoint_flip_pairs: list[int]) -> RFDETRDataModule:
        """Build a keypoint DataModule whose Kornia pipeline comes from the real ``_setup_kornia_pipeline``.

        Only CUDA detection and dataset construction are patched, so ``setup("fit")`` wires the pipeline exactly as it
        does in training: a certain horizontal flip with the given left/right joint pairs.
        """
        mc = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[2])
        tc = _base_train_config(
            tmp_path,
            augmentation_backend="gpu",
            aug_config={"HorizontalFlip": {"p": 1.0}},
            keypoint_flip_pairs=keypoint_flip_pairs,
        )
        dm = self._attach_mock_trainer(RFDETRDataModule(mc, tc), training=True)
        with (
            patch("rfdetr.training.module_data.build_dataset", side_effect=lambda *a, **k: _fake_dataset(10)),
            patch("rfdetr.training.module_data._has_cuda_device", return_value=True),
        ):
            dm.setup("fit")
        return dm

    def _run_setup_pipeline_batch(self, dm: RFDETRDataModule) -> dict[str, torch.Tensor]:
        """Push a 16x16 batch with two joints (slot 0 at (3, 4), slot 1 at (6, 7)) through the batch hook."""
        samples, targets = self._make_kornia_batch(batch_size=1)
        targets[0]["keypoints"] = torch.tensor([[[3.0, 4.0, 2.0], [6.0, 7.0, 1.0]]])

        _, result = dm.on_after_batch_transfer((samples, targets), dataloader_idx=0)

        return result[0]

    @_requires_kornia
    def test_setup_pipeline_with_flip_pairs_swaps_left_right_keypoints(self, tmp_path: Path) -> None:
        """A flipped sample relabels paired joints: slot 0 receives the mirrored slot-1 joint and vice versa."""
        dm = self._build_dm_with_setup_pipeline(tmp_path, keypoint_flip_pairs=[0, 1])

        target = self._run_setup_pipeline_batch(dm)

        torch.testing.assert_close(target["keypoints"], torch.tensor([[[10 / 16, 7 / 16, 1], [13 / 16, 4 / 16, 2]]]))

    @_requires_kornia
    def test_setup_pipeline_without_flip_pairs_leaves_keypoints_unflipped(self, tmp_path: Path) -> None:
        """Without flip pairs the flip is disabled, so joints keep their coordinates and slots (normalized by W, H)."""
        dm = self._build_dm_with_setup_pipeline(tmp_path, keypoint_flip_pairs=[])

        target = self._run_setup_pipeline_batch(dm)

        torch.testing.assert_close(target["keypoints"], torch.tensor([[[3 / 16, 4 / 16, 2], [6 / 16, 7 / 16, 1]]]))

    @_requires_kornia
    def test_setup_pipeline_without_flip_pairs_leaves_boxes_unflipped(self, tmp_path: Path) -> None:
        """Without flip pairs the box [2, 2, 10, 10] stays put: cxcywh (6, 6, 8, 8) / 16."""
        dm = self._build_dm_with_setup_pipeline(tmp_path, keypoint_flip_pairs=[])

        target = self._run_setup_pipeline_batch(dm)

        torch.testing.assert_close(target["boxes"], torch.tensor([[6 / 16, 6 / 16, 8 / 16, 8 / 16]]))


class TestKorniaSetupDoneSentinel:
    """_kornia_setup_done prevents _setup_kornia_pipeline re-running on repeated setup('fit') calls."""

    def _build_dm(self, tmp_path, augmentation_backend="auto"):
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, augmentation_backend=augmentation_backend)
        return RFDETRDataModule(mc, tc)

    def test_sentinel_starts_false(self, tmp_path):
        """_kornia_setup_done is False immediately after __init__."""
        dm = self._build_dm(tmp_path)
        assert dm._kornia_setup_done is False

    def test_sentinel_set_after_fit(self, mocker, tmp_path):
        """_kornia_setup_done becomes True after the first setup('fit') (no CUDA falls back to CPU)."""
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=False)
        dm = _setup_fit_with_fake_datasets(self._build_dm(tmp_path))
        assert dm._kornia_setup_done is True

    def test_setup_kornia_pipeline_not_called_twice(self, mocker, tmp_path):
        """Calling setup('fit') twice only calls _setup_kornia_pipeline once."""
        mocker.patch("rfdetr.training.module_data._has_cuda_device", return_value=False)
        dm = self._build_dm(tmp_path)
        setup_pipeline = mocker.patch.object(dm, "_setup_kornia_pipeline", wraps=dm._setup_kornia_pipeline)

        _setup_fit_with_fake_datasets(dm)
        _setup_fit_with_fake_datasets(dm)

        assert setup_pipeline.call_count == 1, (
            f"_setup_kornia_pipeline called {setup_pipeline.call_count} times; expected exactly 1"
        )
