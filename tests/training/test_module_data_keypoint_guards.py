# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Setup-time guards for keypoint training on the Kornia GPU augmentation path."""

from unittest.mock import patch

import pytest
import torch
import torch.utils.data

from rfdetr.config import AugmentationBackend, RFDETRKeypointPreviewConfig, TrainConfig
from rfdetr.training.module_data import RFDETRDataModule


def _build_keypoint_datamodule(tmp_path, **train_overrides):
    """Return a keypoint ``RFDETRDataModule`` whose train config carries *train_overrides*.

    Examples:
        >>> import pathlib, tempfile
        >>> dm = _build_keypoint_datamodule(pathlib.Path(tempfile.mkdtemp()), multi_scale="off")
        >>> dm.model_config.use_grouppose_keypoints, dm.train_config.multi_scale.value
        (True, 'off')
    """
    model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None, device="cpu")
    train_config = TrainConfig(
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        batch_size=2,
        **train_overrides,
    )
    return RFDETRDataModule(model_config, train_config)


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

        with pytest.raises(ValueError, match="does not support keypoint training with padded batches"):
            _run_fit_setup(datamodule)

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
