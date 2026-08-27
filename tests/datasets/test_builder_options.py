# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Cross-builder contract for the image-pipeline options every dataset builder reads from ``args``.

``build_coco``, ``build_roboflow_from_coco``, ``build_roboflow_from_yolo`` and ``build_o365_raw`` all derive the same
resize/geometry options from the namespace the DataModule hands them.  These options have no safe constant fallback:
``patch_size``, ``num_windows`` and ``segmentation_head`` are variant-dependent, and ``square_resize_div_64``,
``multi_scale`` and ``expanded_scales`` default to ``True`` on ``TrainConfig``.  Substituting a literal for a missing
attribute silently trains a different pipeline, so every builder must read them directly and fail loudly instead.
"""

import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, RFDETRSegSmallConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.datasets.coco import build_coco, build_roboflow_from_coco
from rfdetr.datasets.o365 import build_o365_raw
from rfdetr.datasets.yolo import build_roboflow_from_yolo

# The options that must come from args, with no literal fallback. Ordered as the builders read them.
REQUIRED_PIPELINE_OPTIONS = (
    "square_resize_div_64",
    "segmentation_head",
    "multi_scale",
    "expanded_scales",
    "do_random_resize_via_padding",
    "patch_size",
    "num_windows",
)


def training_namespace(model_config: ModelConfig, dataset_dir: str = "/fake/dataset") -> types.SimpleNamespace:
    """Build the merged model/train namespace that ``RFDETRDataModule`` hands to a dataset builder.

    Uses the real :func:`~rfdetr._namespace._namespace_from_configs` rather than a hand-rolled stub so the
    test asserts against the values a builder actually receives during training.

    Args:
        model_config: Architecture config supplying ``patch_size``, ``num_windows`` and ``segmentation_head``.
        dataset_dir: Dataset root recorded on the namespace.

    Returns:
        Namespace carrying every field the dataset builders read.

    Examples:
        >>> namespace = training_namespace(RFDETRSmallConfig())
        >>> (namespace.multi_scale, namespace.num_windows, namespace.square_resize_div_64)
        (True, 2, True)
    """
    return _namespace_from_configs(model_config, TrainConfig(dataset_dir=dataset_dir))


def build_roboflow_coco_with_mocks(args: Any, resolution: int = 512) -> dict[str, Any]:
    """Run ``build_roboflow_from_coco`` with the filesystem and transform builders mocked out.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Returns:
        ``{"square_resize_used": bool, "transform_kwargs": dict, "dataset_kwargs": dict}`` describing which
        transform branch ran, the kwargs it received, and the kwargs ``CocoDetection`` received.

    Examples:
        >>> captured = build_roboflow_coco_with_mocks(training_namespace(RFDETRSmallConfig()))
        >>> captured["transform_kwargs"]["multi_scale"]
        True
    """
    with (
        patch("rfdetr.datasets.coco.Path") as mock_path,
        patch("rfdetr.datasets.coco.logger"),  # the builder logs to stdout, which would break the doctest
        patch("rfdetr.datasets.coco.make_coco_transforms_square_div_64") as mock_square,
        patch("rfdetr.datasets.coco.make_coco_transforms") as mock_plain,
        patch("rfdetr.datasets.coco.CocoDetection", return_value=MagicMock()) as mock_dataset,
    ):
        mock_path.return_value.exists.return_value = True
        mock_square.return_value = mock_plain.return_value = MagicMock()
        build_roboflow_from_coco("train", args, resolution=resolution)

    used = mock_square if mock_square.called else mock_plain
    return {
        "square_resize_used": mock_square.called,
        "transform_kwargs": used.call_args.kwargs,
        "dataset_kwargs": mock_dataset.call_args.kwargs,
    }


def call_yolo_builder(args: Any, resolution: int = 512) -> None:
    """Run ``build_roboflow_from_yolo`` with the filesystem and split resolution mocked out.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Examples:
        >>> call_yolo_builder(training_namespace(RFDETRSmallConfig()))
    """
    fake_dirs = (MagicMock(), MagicMock())
    with (
        patch("rfdetr.datasets.yolo.Path") as mock_path,
        patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs", return_value=fake_dirs),
        patch("rfdetr.datasets.yolo.make_coco_transforms_square_div_64", return_value=MagicMock()),
        patch("rfdetr.datasets.yolo.make_coco_transforms", return_value=MagicMock()),
        patch("rfdetr.datasets.yolo.YoloDetection", return_value=MagicMock()),
    ):
        mock_path.return_value.exists.return_value = True
        build_roboflow_from_yolo("train", args, resolution=resolution)


def call_o365_builder(args: Any, resolution: int = 512) -> None:
    """Run ``build_o365_raw`` with the dataset class and transform builders mocked out.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Examples:
        >>> call_o365_builder(training_namespace(RFDETRSmallConfig()))
    """
    with (
        patch("rfdetr.datasets.o365.CocoDetection", return_value=MagicMock()),
        patch("rfdetr.datasets.o365.make_coco_transforms_square_div_64", return_value=MagicMock()),
        patch("rfdetr.datasets.o365.make_coco_transforms", return_value=MagicMock()),
    ):
        build_o365_raw("train", args, resolution=resolution)


class TestPipelineOptionsHaveNoSilentFallback:
    """A namespace missing a pipeline option must raise, not train a silently different pipeline."""

    @pytest.fixture
    def partial_namespace(self, tmp_path) -> types.SimpleNamespace:
        """Namespace carrying only the fields unrelated to the image pipeline."""
        return types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            coco_path=str(tmp_path),
            augmentation_backend="cpu",
        )

    def test_roboflow_coco_builder_raises(self, partial_namespace: types.SimpleNamespace) -> None:
        """build_roboflow_from_coco must not substitute a literal for a missing pipeline option."""
        with pytest.raises(AttributeError, match="square_resize_div_64"):
            build_roboflow_coco_with_mocks(partial_namespace)

    def test_roboflow_yolo_builder_raises(self, partial_namespace: types.SimpleNamespace) -> None:
        """build_roboflow_from_yolo must not substitute a literal for a missing pipeline option."""
        with pytest.raises(AttributeError, match="square_resize_div_64"):
            call_yolo_builder(partial_namespace)

    def test_o365_builder_raises(self, partial_namespace: types.SimpleNamespace) -> None:
        """build_o365_raw must not substitute a literal for a missing pipeline option."""
        with pytest.raises(AttributeError, match="square_resize_div_64"):
            call_o365_builder(partial_namespace)

    def test_coco_builder_raises(self, partial_namespace: types.SimpleNamespace) -> None:
        """build_coco must not substitute a literal for a missing pipeline option."""
        with (
            patch("rfdetr.datasets.coco.Path") as mock_path,
            patch("rfdetr.datasets.coco.CocoDetection", return_value=MagicMock()),
        ):
            mock_path.return_value.exists.return_value = True
            with pytest.raises(AttributeError, match="square_resize_div_64"):
                build_coco("train", partial_namespace, resolution=512)


class TestConfigValuesReachTheTransformPipeline:
    """The values the builders forward must be the configured ones, not the old literal fallbacks."""

    @pytest.mark.parametrize(
        "option,expected",
        [
            pytest.param("multi_scale", True, id="multi_scale_stays_enabled"),
            pytest.param("expanded_scales", True, id="expanded_scales_stays_enabled"),
            pytest.param("num_windows", 2, id="num_windows_from_variant_not_4"),
            pytest.param("patch_size", 16, id="patch_size_from_variant"),
        ],
    )
    def test_small_variant_option_is_forwarded(self, option: str, expected: Any) -> None:
        """RFDETRSmall's real option values must reach the transform builder unchanged."""
        captured = build_roboflow_coco_with_mocks(training_namespace(RFDETRSmallConfig()))
        assert captured["transform_kwargs"][option] == expected

    def test_square_resize_default_selects_the_square_branch(self) -> None:
        """TrainConfig.square_resize_div_64 defaults to True, so the square-resize builder must run."""
        captured = build_roboflow_coco_with_mocks(training_namespace(RFDETRSmallConfig()))
        assert captured["square_resize_used"] is True

    def test_segmentation_variant_enables_masks(self) -> None:
        """A segmentation variant must reach CocoDetection with include_masks=True."""
        captured = build_roboflow_coco_with_mocks(training_namespace(RFDETRSegSmallConfig()))
        assert captured["dataset_kwargs"]["include_masks"] is True

    def test_seg_variant_patch_size_is_not_the_old_literal(self) -> None:
        """Segmentation variants use patch_size=12; the removed fallback would have forced 16."""
        captured = build_roboflow_coco_with_mocks(training_namespace(RFDETRSegSmallConfig()))
        assert captured["transform_kwargs"]["patch_size"] == 12
