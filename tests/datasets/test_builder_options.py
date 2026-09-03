# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for required dataset-builder pipeline options and forwarding."""

import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, RFDETRSegSmallConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.datasets.coco import build_coco, build_roboflow_from_coco
from rfdetr.datasets.o365 import build_o365_raw
from rfdetr.datasets.yolo import build_roboflow_from_yolo

ALL_REQUIRED_PIPELINE_OPTIONS = (
    "square_resize_div_64",
    "segmentation_head",
    "multi_scale",
    "expanded_scales",
    "do_random_resize_via_padding",
    "patch_size",
    "num_windows",
)
O365_REQUIRED_PIPELINE_OPTIONS = tuple(
    option for option in ALL_REQUIRED_PIPELINE_OPTIONS if option != "segmentation_head"
)


def training_namespace(model_config: ModelConfig, dataset_dir: str = "/fake/dataset") -> types.SimpleNamespace:
    """Build the merged model/train namespace passed to dataset builders.

    Args:
        model_config: Architecture config supplying model-dependent options.
        dataset_dir: Dataset root recorded on the namespace.

    Returns:
        Namespace carrying the fields consumed by dataset builders.

    Examples:
        >>> namespace = training_namespace(RFDETRSmallConfig())
        >>> (namespace.multi_scale, namespace.num_windows, namespace.square_resize_div_64)
        (True, 2, True)
    """
    return _namespace_from_configs(model_config, TrainConfig(dataset_dir=dataset_dir))


def call_coco_builder(args: Any, resolution: int = 512) -> dict[str, Any]:
    """Run the standard COCO builder with external dataset operations mocked.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Returns:
        Captured transform and dataset arguments.

    Examples:
        >>> captured = call_coco_builder(training_namespace(RFDETRSmallConfig()))
        >>> captured["transform_kwargs"]["num_windows"]
        2
    """
    with (
        patch("rfdetr.datasets.coco.Path") as mock_path,
        patch("rfdetr.datasets.coco.logger"),
        patch("rfdetr.datasets.coco.make_coco_transforms_square_div_64") as mock_square,
        patch("rfdetr.datasets.coco.make_coco_transforms") as mock_plain,
        patch("rfdetr.datasets.coco.CocoDetection", return_value=MagicMock()) as mock_dataset,
    ):
        mock_path.return_value.exists.return_value = True
        mock_square.return_value = mock_plain.return_value = MagicMock()
        build_coco("train", args, resolution=resolution)

    used = mock_square if mock_square.called else mock_plain
    return {
        "square_resize_used": mock_square.called,
        "transform_kwargs": used.call_args.kwargs,
        "dataset_kwargs": mock_dataset.call_args.kwargs,
    }


def call_roboflow_coco_builder(args: Any, resolution: int = 512) -> dict[str, Any]:
    """Run the Roboflow COCO builder with external dataset operations mocked.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Returns:
        Captured transform and dataset arguments.

    Examples:
        >>> captured = call_roboflow_coco_builder(training_namespace(RFDETRSmallConfig()))
        >>> captured["transform_kwargs"]["multi_scale"]
        True
    """
    with (
        patch("rfdetr.datasets.coco.Path") as mock_path,
        patch("rfdetr.datasets.coco.logger"),
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


def call_yolo_builder(args: Any, resolution: int = 512) -> dict[str, Any]:
    """Run the YOLO builder with filesystem and dataset operations mocked.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Returns:
        Captured transform and dataset arguments.

    Examples:
        >>> captured = call_yolo_builder(training_namespace(RFDETRSmallConfig()))
        >>> captured["transform_kwargs"]["expanded_scales"]
        True
    """
    fake_dirs = (MagicMock(), MagicMock())
    with (
        patch("rfdetr.datasets.yolo.Path") as mock_path,
        patch("rfdetr.datasets.yolo._resolve_yolo_split_dirs", return_value=fake_dirs),
        patch("rfdetr.datasets.yolo.make_coco_transforms_square_div_64") as mock_square,
        patch("rfdetr.datasets.yolo.make_coco_transforms") as mock_plain,
        patch("rfdetr.datasets.yolo.YoloDetection", return_value=MagicMock()) as mock_dataset,
    ):
        mock_path.return_value.exists.return_value = True
        mock_square.return_value = mock_plain.return_value = MagicMock()
        build_roboflow_from_yolo("train", args, resolution=resolution)

    used = mock_square if mock_square.called else mock_plain
    return {
        "square_resize_used": mock_square.called,
        "transform_kwargs": used.call_args.kwargs,
        "dataset_kwargs": mock_dataset.call_args.kwargs,
    }


def call_o365_builder(args: Any, resolution: int = 512) -> dict[str, Any]:
    """Run the Object365 builder with dataset and transform operations mocked.

    Args:
        args: Namespace forwarded to the builder.
        resolution: Base square resolution.

    Returns:
        Captured transform and dataset arguments.

    Examples:
        >>> captured = call_o365_builder(training_namespace(RFDETRSmallConfig()))
        >>> captured["transform_kwargs"]["multi_scale"]
        True
    """
    with (
        patch("rfdetr.datasets.o365.make_coco_transforms_square_div_64") as mock_square,
        patch("rfdetr.datasets.o365.make_coco_transforms") as mock_plain,
        patch("rfdetr.datasets.o365.CocoDetection", return_value=MagicMock()) as mock_dataset,
    ):
        mock_square.return_value = mock_plain.return_value = MagicMock()
        build_o365_raw("train", args, resolution=resolution)

    used = mock_square if mock_square.called else mock_plain
    return {
        "square_resize_used": mock_square.called,
        "transform_kwargs": used.call_args.kwargs,
        "dataset_kwargs": mock_dataset.call_args.kwargs,
    }


BuilderCall = Callable[[Any, int], dict[str, Any]]
BUILDER_CONTRACTS: tuple[tuple[str, BuilderCall, tuple[str, ...]], ...] = (
    ("coco", call_coco_builder, ALL_REQUIRED_PIPELINE_OPTIONS),
    ("roboflow-coco", call_roboflow_coco_builder, ALL_REQUIRED_PIPELINE_OPTIONS),
    ("yolo", call_yolo_builder, ALL_REQUIRED_PIPELINE_OPTIONS),
    ("o365", call_o365_builder, O365_REQUIRED_PIPELINE_OPTIONS),
)
MISSING_OPTION_CASES = tuple(
    pytest.param(builder, option, id=f"{name}-{option}")
    for name, builder, required_options in BUILDER_CONTRACTS
    for option in required_options
)


class TestPipelineOptionsHaveNoSilentFallback:
    """Every builder-consumed pipeline option must be present on its namespace."""

    @pytest.fixture
    def complete_namespace(self, tmp_path: Path) -> types.SimpleNamespace:
        """Provide a complete training namespace rooted in a temporary directory.

        Examples:
            Pytest fixture functions cannot be called directly outside fixture injection.
            >>> TestPipelineOptionsHaveNoSilentFallback().complete_namespace(Path("."))  # doctest: +SKIP
        """
        return training_namespace(RFDETRSmallConfig(), dataset_dir=str(tmp_path))

    @pytest.mark.parametrize("builder,missing_option", MISSING_OPTION_CASES)
    def test_builder_raises_for_each_missing_required_option(
        self,
        complete_namespace: types.SimpleNamespace,
        builder: BuilderCall,
        missing_option: str,
    ) -> None:
        """Removing any consumed option must raise for that exact field."""
        delattr(complete_namespace, missing_option)

        with pytest.raises(AttributeError, match=missing_option):
            builder(complete_namespace, 512)


class TestConfigValuesReachTheTransformPipeline:
    """Builders must forward configured values instead of transform defaults."""

    @pytest.mark.parametrize(
        "builder",
        [
            pytest.param(call_coco_builder, id="coco"),
            pytest.param(call_roboflow_coco_builder, id="roboflow-coco"),
            pytest.param(call_yolo_builder, id="yolo"),
            pytest.param(call_o365_builder, id="o365"),
        ],
    )
    @pytest.mark.parametrize("square_resize_div_64", [True, False])
    def test_builder_selects_configured_resize_branch(
        self,
        builder: BuilderCall,
        square_resize_div_64: bool,
    ) -> None:
        """The configured square-resize flag must select the matching branch."""
        namespace = training_namespace(RFDETRSmallConfig())
        namespace.square_resize_div_64 = square_resize_div_64

        captured = builder(namespace, 512)

        assert captured["square_resize_used"] is square_resize_div_64

    @pytest.mark.parametrize(
        "builder",
        [
            pytest.param(call_coco_builder, id="coco"),
            pytest.param(call_roboflow_coco_builder, id="roboflow-coco"),
            pytest.param(call_yolo_builder, id="yolo"),
            pytest.param(call_o365_builder, id="o365"),
        ],
    )
    @pytest.mark.parametrize(
        "namespace_option,configured_value,transform_option,expected_value",
        [
            ("multi_scale", True, "multi_scale", True),
            ("multi_scale", False, "multi_scale", False),
            ("expanded_scales", True, "expanded_scales", True),
            ("expanded_scales", False, "expanded_scales", False),
            ("do_random_resize_via_padding", True, "skip_random_resize", False),
            ("do_random_resize_via_padding", False, "skip_random_resize", True),
            ("patch_size", 12, "patch_size", 12),
            ("num_windows", 3, "num_windows", 3),
        ],
    )
    @pytest.mark.parametrize("square_resize_div_64", [True, False])
    def test_builder_forwards_each_geometry_option(
        self,
        builder: BuilderCall,
        namespace_option: str,
        configured_value: bool | int,
        transform_option: str,
        expected_value: bool | int,
        square_resize_div_64: bool,
    ) -> None:
        """Each configured geometry value must reach the transform unchanged or intentionally inverted."""
        namespace = training_namespace(RFDETRSmallConfig())
        namespace.square_resize_div_64 = square_resize_div_64
        setattr(namespace, namespace_option, configured_value)

        captured = builder(namespace, 512)

        assert captured["transform_kwargs"][transform_option] == expected_value

    def test_o365_does_not_consume_segmentation_option(self) -> None:
        """Object365's documented detection-only builder must not require or forward mask configuration."""
        namespace = training_namespace(RFDETRSegSmallConfig())
        del namespace.segmentation_head

        captured = call_o365_builder(namespace)

        assert "include_masks" not in captured["dataset_kwargs"]

    @pytest.mark.parametrize(
        "builder",
        [
            pytest.param(call_coco_builder, id="coco"),
            pytest.param(call_roboflow_coco_builder, id="roboflow-coco"),
            pytest.param(call_yolo_builder, id="yolo"),
        ],
    )
    def test_segmentation_builders_forward_masks(self, builder: BuilderCall) -> None:
        """Every segmentation-capable builder must enable masks for a segmentation model."""
        captured = builder(training_namespace(RFDETRSegSmallConfig()), 512)
        assert captured["dataset_kwargs"]["include_masks"] is True
