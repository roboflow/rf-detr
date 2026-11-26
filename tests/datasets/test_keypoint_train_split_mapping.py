# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the train-split keypoint label mapping that validation and test splits reuse."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from rfdetr.datasets.coco import (
    _build_keypoint_cat2label,
    _train_split_keypoint_cat2label,
    build_coco,
    build_roboflow_from_coco,
)
from tests.datasets.test_keypoint_category_rule import _annotation, _as_coco, _write_split

_FOUR_KEYPOINTS = [1.0, 2.0, 2.0] * 4
_SEVENTEEN_KEYPOINTS = [1.0, 2.0, 2.0] * 17
# Neither category carries metadata: both keypoint sets come from annotations only.
_ANNOTATION_ONLY_CATEGORIES = [{"id": 1, "name": "helmet"}, {"id": 2, "name": "person"}]
_TRAIN_ANNOTATIONS = [_annotation(1, 1, _FOUR_KEYPOINTS), _annotation(2, 2, _SEVENTEEN_KEYPOINTS)]


class TestTrainSplitKeypointCat2Label:
    """Validation and test splits reuse the keypoint slots derived from the train split."""

    def test_returns_the_train_mapping(self, tmp_path: Path) -> None:
        """Annotation-only keypoints on both train categories map to slots 0 and 1."""
        train = _write_split(tmp_path / "train.json", _ANNOTATION_ONLY_CATEGORIES, _TRAIN_ANNOTATIONS)

        assert _train_split_keypoint_cat2label(train, [4, 17]) == {1: 0, 2: 1}

    def test_a_split_missing_a_category_would_shift_its_own_slots(self) -> None:
        """Premise: derived alone, a split that lacks the lower category maps the other one onto slot 0."""
        valid_alone = _as_coco(_ANNOTATION_ONLY_CATEGORIES, [_annotation(2, 2, _SEVENTEEN_KEYPOINTS)])

        assert _build_keypoint_cat2label(valid_alone, [4, 17]) == {2: 0, 1: 1}

    @pytest.mark.parametrize("missing", ["file", "malformed"])
    def test_unreadable_train_file_falls_back_to_none(self, tmp_path: Path, missing: str) -> None:
        """No usable train annotation file leaves the caller on its split-local mapping."""
        train = tmp_path / "train.json"
        if missing == "malformed":
            train.write_text("{not json", encoding="utf-8")

        assert _train_split_keypoint_cat2label(train, [17]) is None


@pytest.fixture
def roboflow_root(tmp_path: Path) -> Path:
    """Roboflow root whose train split annotates both keypoint categories and valid only the second."""
    _write_split(tmp_path / "train" / "_annotations.coco.json", _ANNOTATION_ONLY_CATEGORIES, _TRAIN_ANNOTATIONS)
    _write_split(
        tmp_path / "valid" / "_annotations.coco.json",
        _ANNOTATION_ONLY_CATEGORIES,
        [_annotation(1, 2, _SEVENTEEN_KEYPOINTS)],
    )
    return tmp_path


@pytest.fixture
def roboflow_args(roboflow_root: Path) -> types.SimpleNamespace:
    """Roboflow builder args in keypoint mode with schema ``[4, 17]``."""
    return types.SimpleNamespace(
        dataset_dir=str(roboflow_root),
        square_resize_div_64=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        patch_size=16,
        num_windows=4,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[4, 17],
        keypoint_flip_pairs=[],
        aug_config={},
        augmentation_backend="cpu",
    )


class TestBuildersReuseTheTrainKeypointMapping:
    """``build_roboflow_from_coco`` and ``build_coco`` hand every non-train split the train mapping."""

    @pytest.mark.parametrize(
        ("split", "expected"),
        [
            pytest.param("train", None, id="train-derives-its-own"),
            pytest.param("val", {1: 0, 2: 1}, id="val-reuses-train"),
        ],
    )
    def test_roboflow_builder(
        self, mocker, roboflow_args: types.SimpleNamespace, split: str, expected: dict[int, int] | None
    ) -> None:
        """The Roboflow builder passes the train keypoint mapping to a validation split, never to train."""
        mocker.patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target))
        mock_dataset = mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=object())

        build_roboflow_from_coco(split, roboflow_args, resolution=64)

        assert mock_dataset.call_args.kwargs["cat2label"] == expected

    def test_roboflow_builder_falls_back_without_a_train_file(
        self, mocker, roboflow_root: Path, roboflow_args: types.SimpleNamespace
    ) -> None:
        """A missing train annotation file keeps the split-local mapping instead of failing the build."""
        (roboflow_root / "train" / "_annotations.coco.json").unlink()
        mocker.patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target))
        mock_dataset = mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=object())

        build_roboflow_from_coco("val", roboflow_args, resolution=64)

        assert mock_dataset.call_args.kwargs["cat2label"] is None

    @pytest.mark.parametrize(
        ("split", "expected"),
        [
            pytest.param("train", None, id="train-derives-its-own"),
            pytest.param("val", {1: 0, 2: 1}, id="val-reuses-train"),
        ],
    )
    def test_coco_builder(
        self,
        mocker,
        tmp_path: Path,
        roboflow_args: types.SimpleNamespace,
        split: str,
        expected: dict[int, int] | None,
    ) -> None:
        """The plain COCO builder passes the train keypoint mapping to the validation split as well."""
        _write_split(
            tmp_path / "annotations" / "person_keypoints_train2017.json",
            _ANNOTATION_ONLY_CATEGORIES,
            _TRAIN_ANNOTATIONS,
        )
        args = types.SimpleNamespace(**{**vars(roboflow_args), "dataset_dir": None, "coco_path": str(tmp_path)})
        mocker.patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target))
        mock_dataset = mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=object())

        build_coco(split, args, resolution=64)

        assert mock_dataset.call_args.kwargs["cat2label"] == expected
