# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression tests for COCO dataset handling.

Tests cover:
- Sparse COCO category ID remapping in ``ConvertCoco``
- ``_load_classes`` hierarchy detection (GitHub #609)
"""

import json
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema
from rfdetr.datasets.coco import (
    CocoDetection,
    ConvertCoco,
    _build_albumentations_pipeline,
    _build_torchvision_pipeline,
    annotated_category_ids,
    build_coco,
    build_roboflow_from_coco,
    draft_size_for_transforms,
    filter_parent_categories,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
    scale_coco_annotation,
)
from rfdetr.datasets.webdataset.pack import pack_coco_to_shards
from rfdetr.detr import RFDETR
from rfdetr.utilities import PackedTargets, pack_targets
from tests._markers import _requires_albumentations

# Minimal image shared across all tests
_IMAGE = Image.new("RGB", (100, 100))

# Sparse COCO-style category IDs (as in the real COCO dataset: 1-90 with gaps)
# e.g. COCO skips IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91
_SPARSE_CAT_IDS = [1, 2, 3, 7, 8]  # sparse, non-zero-indexed

_ANNOTATIONS = [
    {"bbox": [10, 10, 30, 30], "category_id": 1, "area": 900, "iscrowd": 0},
    {"bbox": [50, 50, 20, 20], "category_id": 7, "area": 400, "iscrowd": 0},
]

_CAT2LABEL = {cat_id: i for i, cat_id in enumerate(sorted(_SPARSE_CAT_IDS))}
# {1: 0, 2: 1, 3: 2, 7: 3, 8: 4}


def _make_target(annotations=_ANNOTATIONS):
    """Build a minimal COCO-style target mapping for converter tests.

    Example:
        >>> _make_target()["image_id"]
        1
    """
    return {"image_id": 1, "annotations": annotations}


class TestConvertCocoWithoutMapping:
    """Without cat2label, sparse IDs pass through unchanged — demonstrating the bug."""

    def test_labels_are_raw_category_ids(self):
        """Labels keep the raw (sparse) COCO category ids when no mapping is given."""
        converter = ConvertCoco(cat2label=None)
        _, target = converter(_IMAGE, _make_target())
        # Raw COCO IDs — NOT safe to use as indices into an 80-class tensor
        assert target["labels"].tolist() == [1, 7]

    def test_raw_ids_would_exceed_num_classes(self):
        """Illustrates why raw IDs cause CUDA out-of-bounds with num_classes=80."""
        converter = ConvertCoco(cat2label=None)
        _, target = converter(_IMAGE, _make_target())
        num_classes = len(_SPARSE_CAT_IDS)  # 5 — same as model would see
        assert any(lbl >= num_classes for lbl in target["labels"].tolist()), (
            "At least one raw category_id should exceed num_classes, "
            "triggering an out-of-bounds index in the matcher/loss."
        )


class TestConvertCocoWithMapping:
    """With cat2label, sparse IDs are remapped to contiguous 0-indexed labels."""

    def test_labels_are_remapped_to_zero_indexed(self):
        """Sparse category ids become contiguous zero-indexed labels."""
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        # category_id 1 → 0, category_id 7 → 3
        assert target["labels"].tolist() == [0, 3]


class TestConvertCocoPlainDetectionDtypeParity:
    """Outside keypoint mode, integer-valued ``area`` annotations must still pack."""

    def test_empty_and_populated_targets_pack_with_integer_area(self) -> None:
        """Integer-area populated targets and float-area empty targets must keep matching dtypes."""
        converter = ConvertCoco(cat2label=None)

        _, empty_target = converter(_IMAGE, {"image_id": 1, "annotations": []})
        _, populated_target = converter(_IMAGE, _make_target())

        assert empty_target["area"].dtype == populated_target["area"].dtype
        assert empty_target["area"].dtype == torch.float32
        assert empty_target["iscrowd"].dtype == populated_target["iscrowd"].dtype
        assert empty_target["iscrowd"].dtype == torch.int64
        packed = pack_targets((empty_target, populated_target))
        assert isinstance(packed, PackedTargets)
        assert [target["area"].dtype for target in packed] == [torch.float32, torch.float32]
        assert [target["iscrowd"].dtype for target in packed] == [torch.int64, torch.int64]

    def test_all_labels_within_num_classes(self):
        """Every remapped label is a valid class index for a model with one class per category."""
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        num_classes = len(_SPARSE_CAT_IDS)
        assert all(lbl < num_classes for lbl in target["labels"].tolist())

    def test_keypoints_retain_instances_with_all_invisible_keypoints(self) -> None:
        """Instances with all-invisible keypoints must be retained for box/class supervision."""
        converter = ConvertCoco(include_keypoints=True, num_keypoints_per_class=[17])
        visible_keypoints = [0.0, 0.0, 0.0] * 17
        visible_keypoints[2] = 2.0
        unlabeled_keypoints = [0.0, 0.0, 0.0] * 17

        _, target = converter(
            _IMAGE,
            _make_target(
                [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10.0, 10.0, 20.0, 20.0],
                        "area": 400.0,
                        "iscrowd": 0,
                        "keypoints": unlabeled_keypoints,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [30.0, 30.0, 20.0, 20.0],
                        "area": 400.0,
                        "iscrowd": 0,
                        "keypoints": visible_keypoints,
                    },
                ]
            ),
        )

        assert target["boxes"].shape == (2, 4)
        assert target["labels"].tolist() == [1, 1]
        assert target["keypoints"].shape == (2, 17, 3)
        assert target["keypoints"][1, 0, 2].item() == 2.0

    def test_roboflow_zero_indexed_is_identity(self):
        """Roboflow datasets already use 0-indexed IDs — mapping must be identity."""
        roboflow_cat2label = {0: 0, 1: 1, 2: 2}
        annotations = [
            {"bbox": [10, 10, 30, 30], "category_id": 0, "area": 900, "iscrowd": 0},
            {"bbox": [50, 50, 20, 20], "category_id": 2, "area": 400, "iscrowd": 0},
        ]
        converter = ConvertCoco(cat2label=roboflow_cat2label)
        _, target = converter(_IMAGE, _make_target(annotations))
        assert target["labels"].tolist() == [0, 2]

    def test_label_tensor_dtype(self):
        """Remapped labels are an int64 tensor."""
        converter = ConvertCoco(cat2label=_CAT2LABEL)
        _, target = converter(_IMAGE, _make_target())
        assert target["labels"].dtype == torch.int64


def _write_coco_json(path: Path, categories: List[Dict]) -> None:
    """Write a minimal valid COCO annotation file.

    Example:
        >>> import tempfile
        >>> output = Path(tempfile.mkdtemp()) / "annotations.json"
        >>> _write_coco_json(output, [{"id": 1, "name": "person"}])
        >>> output.exists()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"images": [], "annotations": [], "categories": categories}
    path.write_text(json.dumps(data))


def _write_roboflow_keypoint_coco(path: Path, *, category_id: int = 0) -> None:
    """Write a minimal Roboflow-style COCO keypoint split.

    Example:
        >>> import tempfile
        >>> output = Path(tempfile.mkdtemp()) / "annotations.json"
        >>> _write_roboflow_keypoint_coco(output)
        >>> output.exists()
        True
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image_path = path.parent / "person.png"
    Image.new("RGB", (64, 48), color=(255, 255, 255)).save(image_path)
    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    keypoints = []
    for idx in range(len(keypoint_names)):
        keypoints.extend([10 + idx, 20 + idx, 2])
    data = {
        "images": [{"id": 1, "file_name": image_path.name, "width": 64, "height": 48}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": category_id,
                "bbox": [8, 18, 24, 24],
                "area": 576,
                "iscrowd": 0,
                "num_keypoints": len(keypoint_names),
                "keypoints": keypoints,
            }
        ],
        "categories": [
            {
                "id": category_id,
                "name": "person",
                "supercategory": "person",
                "keypoints": keypoint_names,
                "skeleton": [],
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _pipeline_args(dataset_dir: object, **overrides: object) -> types.SimpleNamespace:
    """Build a builder namespace carrying the image-pipeline options the dataset builders require.

    The builders read these options directly, with no literal fallback, so a namespace handed to one must spell
    them out. The values here reproduce the pipeline the removed ``getattr`` fallbacks used to produce, which
    keeps tests that only care about label space or backend resolution behaviourally unchanged. Tests asserting
    that the *configured* values reach the pipeline live in ``tests/datasets/test_builder_options.py``.

    Args:
        dataset_dir: Dataset root recorded on the namespace.
        **overrides: Extra fields to add, or pipeline options to replace.

    Returns:
        Namespace accepted by the Roboflow COCO and YOLO builders.

    Examples:
        >>> _pipeline_args("/tmp/ds", augmentation_backend="gpu").multi_scale
        False
    """
    options = {
        "dataset_dir": str(dataset_dir),
        "square_resize_div_64": False,
        "segmentation_head": False,
        "multi_scale": False,
        "expanded_scales": False,
        "patch_size": 16,
        "num_windows": 4,
    }
    options.update(overrides)
    return types.SimpleNamespace(**options)


class TestLoadClassesHierarchy:
    """Regression tests for ``_load_classes`` supercategory filtering (#609).

    When all categories have ``supercategory: "none"`` (flat COCO datasets), ``_load_classes`` previously returned an
    empty list. It should only filter when a Roboflow hierarchical export is detected.
    """

    @pytest.mark.parametrize(
        "categories, expected",
        [
            pytest.param(
                [
                    {"id": 0, "name": "annotations", "supercategory": "none"},
                    {"id": 1, "name": "dog", "supercategory": "annotations"},
                    {"id": 2, "name": "cat", "supercategory": "annotations"},
                ],
                ["dog", "cat"],
                id="roboflow-export-includes-a-parent-node-only-leaf-categories-kept",
            ),
            pytest.param(
                [
                    {"id": 1, "name": "dog", "supercategory": "none"},
                    {"id": 2, "name": "cat", "supercategory": "none"},
                ],
                ["dog", "cat"],
                id="flat-dataset-every-category-has-supercategory-none-issue-609",
            ),
            pytest.param(
                [
                    {"id": 1, "name": "dog", "supercategory": "none"},
                    {"id": 2, "name": "cat", "supercategory": "animal"},
                ],
                ["dog", "cat"],
                # 'animal' is a supercategory but not itself a category name, so has_children is empty and every
                # category passes the "name not in has_children" filter.
                id="mixed-supercategories-where-no-category-is-a-parent-of-another",
            ),
            pytest.param(
                [
                    {"id": 1, "name": "none", "supercategory": "none"},
                    {"id": 2, "name": "dog", "supercategory": "none"},
                    {"id": 3, "name": "cat", "supercategory": "none"},
                ],
                ["none", "dog", "cat"],
                id="category-literally-named-none-must-not-empty-the-list",
            ),
            pytest.param(
                [
                    {"id": 1, "name": "animals", "supercategory": "none"},
                    {"id": 2, "name": "mammal", "supercategory": "animals"},
                    {"id": 3, "name": "dog", "supercategory": "mammal"},
                    {"id": 4, "name": "cat", "supercategory": "mammal"},
                    {"id": 5, "name": "bird", "supercategory": "animals"},
                    {"id": 6, "name": "eagle", "supercategory": "bird"},
                    {"id": 7, "name": "pigeon", "supercategory": "bird"},
                    {"id": 8, "name": "objects", "supercategory": "none"},
                    {"id": 9, "name": "vehicle", "supercategory": "objects"},
                    {"id": 10, "name": "car", "supercategory": "vehicle"},
                    {"id": 11, "name": "truck", "supercategory": "vehicle"},
                    {"id": 12, "name": "appliance", "supercategory": "objects"},
                    {"id": 13, "name": "toaster", "supercategory": "appliance"},
                    {"id": 14, "name": "microwave", "supercategory": "appliance"},
                    {"id": 15, "name": "person", "supercategory": "none"},
                ],
                ["dog", "cat", "eagle", "pigeon", "car", "truck", "toaster", "microwave", "person"],
                # Only leaf classes + standalone top-level categories are forwarded; parent/grouping nodes drop.
                id="mixed-hierarchy-leaf-and-standalone-categories-forwarded-parents-dropped",
            ),
            pytest.param(
                [
                    {"id": 1, "name": "dog", "supercategory": None},
                    {"id": 2, "name": "cat", "supercategory": ""},
                    {"id": 3, "name": "elephant", "supercategory": "null"},
                ],
                ["dog", "cat", "elephant"],
                id="placeholders-none-empty-string-and-literal-null-treated-like-none",
            ),
            pytest.param(
                [
                    {"id": 30, "name": "truck", "supercategory": "vehicle"},
                    {"id": 10, "name": "vehicle", "supercategory": "none"},
                    {"id": 20, "name": "car", "supercategory": "vehicle"},
                    {"id": 40, "name": "person", "supercategory": "none"},
                ],
                ["car", "truck", "person"],
                id="unsorted-category-ids-return-id-sorted-class-order-for-stable-index-mapping",
            ),
        ],
    )
    def test_load_classes_hierarchy_filtering(
        self, tmp_path: Path, categories: list[dict], expected: list[str]
    ) -> None:
        """``_load_classes`` filters supercategory hierarchy across flat, nested, and placeholder-value datasets.

        Regression coverage for #609: when all categories have ``supercategory: "none"`` (flat COCO datasets),
        ``_load_classes`` previously returned an empty list. It should only filter parent/grouping nodes when a Roboflow
        hierarchical export is actually detected.
        """
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == expected


class TestRoboflowCocoKeypointFormat:
    """Roboflow COCO keypoint datasets should align labels with the keypoint schema."""

    def _make_args(self, dataset_dir: Path) -> types.SimpleNamespace:
        """Return minimal args consumed by ``build_roboflow_from_coco`` in keypoint mode."""
        return types.SimpleNamespace(
            dataset_dir=str(dataset_dir),
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[17],
            aug_config={},
            augmentation_backend="cpu",
        )

    def test_keypoint_category_maps_to_active_schema_slot(self, tmp_path: Path) -> None:
        """A one-class Roboflow keypoint dataset maps person to label 0 for the `[17]` preview schema."""
        _write_roboflow_keypoint_coco(tmp_path / "train" / "_annotations.coco.json", category_id=0)

        dataset = build_roboflow_from_coco("train", self._make_args(tmp_path), resolution=64)
        _, target = dataset[0]

        assert target["labels"].tolist() == [0]
        assert target["keypoints"].shape == (1, 17, 3)
        assert dataset.cat2label == {0: 0}
        assert dataset.label2cat == {0: 0}
        assert dataset.coco.label2cat == {0: 0}

    def test_standard_coco_cat_id_maps_to_active_schema_slot(self, tmp_path: Path) -> None:
        """Standard COCO person (cat_id=1) maps to slot 0 under the active-first [17] schema."""
        _write_roboflow_keypoint_coco(tmp_path / "train" / "_annotations.coco.json", category_id=1)

        dataset = build_roboflow_from_coco("train", self._make_args(tmp_path), resolution=64)

        assert dataset.cat2label == {1: 0}

    def test_keypoint_coco_without_keypoint_schema_raises(self, tmp_path: Path) -> None:
        """Keypoint mode should fail clearly if a COCO dataset has no keypoint metadata or annotations."""
        _write_coco_json(
            tmp_path / "train" / "_annotations.coco.json",
            [{"id": 0, "name": "person", "supercategory": "none"}],
        )

        with pytest.raises(ValueError, match="Keypoint COCO dataset"):
            build_roboflow_from_coco("train", self._make_args(tmp_path), resolution=64)


class TestInferCocoKeypointSchema:
    """COCO keypoint schema inference."""

    def test_reads_category_keypoint_metadata(self, tmp_path: Path) -> None:
        """Category keypoint names define the per-class keypoint count."""
        _write_roboflow_keypoint_coco(tmp_path / "train" / "_annotations.coco.json", category_id=0)

        schema = infer_coco_keypoint_schema(tmp_path / "train" / "_annotations.coco.json")

        assert schema.class_names == ["person"]
        assert schema.num_keypoints_per_class == [17]
        assert len(schema.keypoint_oks_sigmas) == 17

    def test_falls_back_to_annotation_keypoint_vectors(self, tmp_path: Path) -> None:
        """Annotation vectors can define keypoint count when category names are absent."""
        annotation_path = tmp_path / "train" / "_annotations.coco.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 0,
                            "bbox": [0, 0, 10, 10],
                            "area": 100,
                            "iscrowd": 0,
                            "keypoints": [1, 2, 2, 3, 4, 2],
                        }
                    ],
                    "categories": [{"id": 0, "name": "person", "supercategory": "none"}],
                }
            ),
            encoding="utf-8",
        )

        schema = infer_coco_keypoint_schema(annotation_path)

        assert schema.num_keypoints_per_class == [2]


# ---------------------------------------------------------------------------
# TestBuildO365RawGpuBackend — validates that build_o365_raw emits a WARNING
# and passes gpu_postprocess when augmentation_backend != 'cpu'.
# ---------------------------------------------------------------------------


class TestBuildO365RawGpuBackend:
    """build_o365_raw warns and wires gpu_postprocess for non-cpu backends."""

    class _FakeArgs:
        """Minimal args stub for build_o365_raw."""

        def __init__(self, augmentation_backend="cpu", square_resize_div_64=False):
            self.augmentation_backend = augmentation_backend
            self.square_resize_div_64 = square_resize_div_64
            self.multi_scale = False
            self.expanded_scales = False
            self.patch_size = 16
            self.num_windows = 4
            self.dataset_dir = "/nonexistent/o365"
            self.coco_path = "/nonexistent/o365"

    def _call_build_o365_raw(self, mocker, augmentation_backend, square_resize_div_64=False):
        """Call build_o365_raw with mocked CocoDetection and transform builders."""
        from rfdetr.datasets.o365 import build_o365_raw

        args = self._FakeArgs(augmentation_backend=augmentation_backend, square_resize_div_64=square_resize_div_64)
        mocker.patch("rfdetr.datasets.o365.CocoDetection", return_value=mocker.MagicMock())
        mock_transform = mocker.patch("rfdetr.datasets.o365.make_coco_transforms")
        mock_sq_transform = mocker.patch("rfdetr.datasets.o365.make_coco_transforms_square_div_64")
        result = build_o365_raw("train", args, resolution=640)
        return result, mock_transform, mock_sq_transform

    @pytest.mark.parametrize("backend", ["cpu", "auto"])
    def test_no_warning_when_backend_resolves_to_cpu(self, mocker, backend):
        """The cpu backend, and auto without CUDA, emit no O365 GPU-augmentation warning."""
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False)
        mock_logger = mocker.patch("rfdetr.datasets.o365.logger")
        self._call_build_o365_raw(mocker, backend)
        o365_warns = [c for c in mock_logger.warning.call_args_list if "O365" in str(c)]
        assert len(o365_warns) == 0, f"{backend} backend without CUDA must not warn about O365 GPU augmentation"

    def test_auto_backend_emits_warning(self, mocker):
        """Auto + CUDA + kornia available: logger.warning about O365 Phase 1 limitation."""
        from rfdetr.config import AugmentationBackend

        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: True)
        mock_logger = mocker.patch("rfdetr.datasets.o365.logger")
        self._call_build_o365_raw(mocker, "auto")
        o365_warns = [c for c in mock_logger.warning.call_args_list if "O365" in str(c)]
        assert len(o365_warns) >= 1, "auto backend must warn about O365 GPU aug limitation"

    @pytest.mark.parametrize(
        "backend,has_cuda,expected",
        [
            pytest.param("cpu", False, False, id="cpu"),
            pytest.param("auto", True, True, id="auto-cuda-kornia"),
            pytest.param("auto", False, False, id="auto-no-cuda"),
        ],
    )
    def test_gpu_postprocess_follows_resolved_backend(self, mocker, backend, has_cuda, expected):
        """gpu_postprocess is True only when the backend resolves to GPU; otherwise CPU Normalize is retained."""
        from rfdetr.config import AugmentationBackend

        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=has_cuda)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: True)
        _, mock_transform, _ = self._call_build_o365_raw(mocker, backend)
        call_kwargs = mock_transform.call_args.kwargs if mock_transform.call_args else {}
        assert call_kwargs.get("gpu_postprocess", False) is expected

    def test_square_resize_uses_square_transform(self, mocker):
        """square_resize_div_64=True delegates to make_coco_transforms_square_div_64."""
        _, mock_transform, mock_sq_transform = self._call_build_o365_raw(mocker, "cpu", square_resize_div_64=True)
        mock_sq_transform.assert_called_once()
        mock_transform.assert_not_called()

    def test_gpu_backend_no_cuda_raises_runtime_error(self, mocker):
        """Gpu backend must fail fast when CUDA is unavailable."""
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False)
        with pytest.raises(RuntimeError, match="CUDA"):
            self._call_build_o365_raw(mocker, "gpu")

    def test_gpu_backend_no_kornia_raises_import_error(self, mocker):
        """Gpu backend must raise with install hint when kornia is missing."""
        from rfdetr.config import AugmentationBackend

        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA)
        with pytest.raises(ImportError, match="rfdetr\\[augment\\]"):
            self._call_build_o365_raw(mocker, "gpu")


class TestBuildRoboflowFromCocoBackendResolution:
    """Roboflow COCO builder should resolve backend for gpu_postprocess consistently."""

    def test_auto_no_cuda_keeps_cpu_normalize(self, mocker):
        """Auto + no CUDA must set gpu_postprocess=False."""
        args = types.SimpleNamespace(
            dataset_dir="/fake/dataset",
            augmentation_backend="auto",
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
            aug_config=None,
        )
        mock_path = mocker.patch("rfdetr.datasets.coco.Path")
        mock_transforms = mocker.patch("rfdetr.datasets.coco.make_coco_transforms")
        mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=mocker.MagicMock())
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False)
        mock_path.return_value.exists.return_value = True
        build_roboflow_from_coco("train", args, resolution=640)
        assert mock_transforms.call_args.kwargs["gpu_postprocess"] is False

    def test_gpu_backend_no_cuda_raises_runtime_error(self, mocker, tmp_path: Path) -> None:
        """Roboflow COCO builder fails fast when 'gpu' is requested without a CUDA device."""
        args = _pipeline_args(tmp_path, augmentation_backend="gpu")
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False)
        with pytest.raises(RuntimeError, match="CUDA"):
            build_roboflow_from_coco("train", args, resolution=640)

    def test_gpu_backend_no_kornia_raises_import_error(self, mocker, tmp_path: Path) -> None:
        """Roboflow COCO builder fails fast with an install hint when 'gpu' is requested but kornia is missing."""
        from rfdetr.config import AugmentationBackend

        args = _pipeline_args(tmp_path, augmentation_backend="gpu")
        mocker.patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True)
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA)
        with pytest.raises(ImportError, match=r"rfdetr\[augment\]"):
            build_roboflow_from_coco("train", args, resolution=640)


class TestBuilderGpuPostprocess:
    """Verify Roboflow COCO builder sets gpu_postprocess for segmentation models."""

    @pytest.mark.parametrize(
        "segmentation_head, augmentation_backend, resolved_backend, expected_gpu_postprocess",
        [
            pytest.param(False, "cpu", "torchvision", False, id="cpu_backend_no_seg"),
            pytest.param(True, "cpu", "torchvision", False, id="cpu_backend_with_seg"),
            pytest.param(False, "gpu", "kornia", True, id="gpu_backend_no_seg"),
            pytest.param(True, "gpu", "kornia", True, id="gpu_backend_with_seg"),
            pytest.param(True, "auto", "kornia", True, id="auto_resolved_gpu_with_seg"),
            pytest.param(True, "auto", "torchvision", False, id="auto_resolved_cpu_with_seg"),
        ],
    )
    def test_gpu_postprocess_flag(
        self,
        mocker,
        tmp_path,
        segmentation_head,
        augmentation_backend,
        resolved_backend,
        expected_gpu_postprocess,
    ):
        """Build Roboflow COCO datasets and assert the GPU postprocess flag passed to transforms."""
        annotations_dir = tmp_path / "train"
        annotations_dir.mkdir()
        (annotations_dir / "_annotations.coco.json").write_text(
            json.dumps({"images": [], "annotations": [], "categories": []}),
            encoding="utf-8",
        )
        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            segmentation_head=segmentation_head,
            augmentation_backend=augmentation_backend,
            square_resize_div_64=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
            aug_config=None,
        )
        mocker.patch("rfdetr.datasets.coco.resolve_backend_for_build", return_value=resolved_backend)
        mock_transforms = mocker.patch("rfdetr.datasets.coco.make_coco_transforms")
        mocker.patch("rfdetr.datasets.coco.CocoDetection")

        build_roboflow_from_coco("train", args, resolution=640)

        call_kwargs = mock_transforms.call_args.kwargs if mock_transforms.call_args else mock_transforms.call_args[1]
        assert call_kwargs["gpu_postprocess"] is expected_gpu_postprocess


def _make_coco_builder_args(tmp_path: Path, *, use_grouppose_keypoints: bool) -> types.SimpleNamespace:
    """Return a namespace with all fields consumed by ``build_coco``."""
    return types.SimpleNamespace(
        dataset_dir=None,
        coco_path=str(tmp_path),
        square_resize_div_64=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        patch_size=16,
        num_windows=4,
        # Empty aug_config disables augmentation — these tests verify annotation routing, not aug.
        aug_config={},
        augmentation_backend="cpu",
        use_grouppose_keypoints=use_grouppose_keypoints,
        num_keypoints_per_class=[17] if use_grouppose_keypoints else [],
        keypoint_flip_pairs=[],
    )


def _make_roboflow_flip_args(tmp_path: Path, *, use_keypoints: bool) -> types.SimpleNamespace:
    """Return a namespace with all fields consumed by ``build_roboflow_from_coco`` in the flip-pairs tests.

    Examples:
        >>> _make_roboflow_flip_args(Path("/tmp/ds"), use_keypoints=True).num_keypoints_per_class
        [0, 4]
        >>> _make_roboflow_flip_args(Path("/tmp/ds"), use_keypoints=False).use_grouppose_keypoints
        False
    """
    return types.SimpleNamespace(
        dataset_dir=str(tmp_path),
        augmentation_backend="cpu",
        square_resize_div_64=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        patch_size=16,
        num_windows=4,
        use_grouppose_keypoints=use_keypoints,
        num_keypoints_per_class=[0, 4] if use_keypoints else [],
        keypoint_flip_pairs=[],
        aug_config={"HorizontalFlip": {"p": 0.5}},
    )


class TestKeypointFlipPairsNoneForwarding:
    """``build_coco``/``build_roboflow_from_coco`` must forward the correct ``keypoint_flip_pairs`` sentinel.

    Regression tests for GitHub #1243, covering all three directions of the ``include_keypoints`` gate on
    ``keypoint_flip_pairs: list[int] | None = ((getattr(args, "keypoint_flip_pairs", []) or []) if
    include_keypoints else None)``:

    1. Detection-only builds (``include_keypoints=False``) must forward ``None``. ``AlbumentationsWrapper.from_config``
       treats ``keypoint_flip_pairs=[]`` as "keypoint pipeline with no flip pairs defined" and silently strips
       ``HorizontalFlip``/``Flip``/``D4`` from any custom ``aug_config`` to avoid corrupting keypoint annotations;
       forwarding that ``[]`` sentinel for datasets that have no keypoints at all silently disables the user's
       requested horizontal-flip augmentation. ``yolo.py`` already gates this correctly on ``include_keypoints``;
       these builders must match.
    2. Keypoint builds (``include_keypoints=True``) with no ``keypoint_flip_pairs`` configured must forward ``[]``
       (not ``None``) -- the mirror-direction bug: forwarding ``None`` would silently re-enable hflip for a dataset
       with unknown left/right correspondence.
    3. Keypoint builds where ``args.keypoint_flip_pairs`` is explicitly ``None`` (e.g. a fresh
       ``KeypointTrainConfig(dataset_dir=...)``, whose default is ``None``) must coerce that to ``[]`` via the
       ``or []`` fallback, not forward ``None`` through untouched.
    """

    @pytest.mark.parametrize("builder_kind", ["roboflow", "coco"])
    @pytest.mark.parametrize(
        ("square_resize_div_64", "transform_factory"),
        [
            pytest.param(False, "make_coco_transforms", id="standard_resize"),
            pytest.param(True, "make_coco_transforms_square_div_64", id="square_resize"),
        ],
    )
    @pytest.mark.parametrize(
        ("use_keypoints", "configured_pairs", "expected_pairs"),
        [
            pytest.param(False, [], None, id="detection-forwards-none"),
            pytest.param(True, [0, 1, 2, 3], [0, 1, 2, 3], id="keypoint-forwards-configured-pairs"),
            pytest.param(True, [], [], id="keypoint-without-pairs-forwards-empty-list"),
            pytest.param(True, None, [], id="keypoint-explicit-none-coerced-to-empty-list"),
        ],
    )
    def test_builder_forwards_flip_pairs_sentinel(
        self,
        mocker,
        tmp_path: Path,
        builder_kind: str,
        square_resize_div_64: bool,
        transform_factory: str,
        use_keypoints: bool,
        configured_pairs: list[int] | None,
        expected_pairs: list[int] | None,
    ) -> None:
        """Detection builds forward ``None``; keypoint builds forward the configured pairs, with ``None`` coerced to
        ``[]``."""
        if builder_kind == "roboflow":
            builder = build_roboflow_from_coco
            args = _make_roboflow_flip_args(tmp_path, use_keypoints=use_keypoints)
        else:
            builder = build_coco
            args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=use_keypoints)
            args.aug_config = {"HorizontalFlip": {"p": 0.5}}
        args.square_resize_div_64 = square_resize_div_64
        args.keypoint_flip_pairs = configured_pairs
        mock_transforms = mocker.patch(f"rfdetr.datasets.coco.{transform_factory}")
        mocker.patch("rfdetr.datasets.coco.CocoDetection")

        builder("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == expected_pairs


def _make_keypoint_annotation(
    *,
    category_id: int = 1,
    bbox: List[float] | None = None,
    area: float = 80.0,
    keypoints: List[float] | None = None,
) -> Dict[str, object]:
    """Build a minimal keypoint annotation used in keypoint conversion tests."""
    return {
        "bbox": bbox if bbox is not None else [10.0, 5.0, 8.0, 10.0],
        "category_id": category_id,
        "area": area,
        "iscrowd": 0,
        "keypoints": keypoints if keypoints is not None else [1.0, 2.0, 2.0] * 17,
    }


class TestConvertCocoKeypoints:
    """ConvertCoco keypoint-mode coverage."""

    def test_empty_and_populated_targets_pack_without_dtype_fallback(self) -> None:
        """Empty and populated keypoint targets must keep matching integer dtypes for lossless packing."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label=None,
            num_keypoints_per_class=[17],
        )

        _, empty_target = converter(_IMAGE, {"image_id": 1, "annotations": []})
        _, populated_target = converter(
            _IMAGE,
            {"image_id": 2, "annotations": [_make_keypoint_annotation()]},
        )

        assert empty_target["iscrowd"].dtype == populated_target["iscrowd"].dtype
        assert empty_target["iscrowd"].dtype == torch.int64
        packed = pack_targets((empty_target, populated_target))
        assert isinstance(packed, PackedTargets)
        assert [target["iscrowd"].dtype for target in packed] == [torch.int64, torch.int64]

    def test_keypoint_target_includes_keypoints(self) -> None:
        """Keypoint-enabled conversion should emit keypoints in ``[N, K, 3]`` format."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label=None,
            num_keypoints_per_class=[17],
        )

        _, target = converter(
            _IMAGE,
            {"image_id": 42, "annotations": [_make_keypoint_annotation()]},
        )

        assert target["keypoints"].shape == (1, 17, 3)
        assert target["keypoints"].dtype == torch.float32
        assert target["labels"].tolist() == [1]

    def test_person_category_stays_raw_coco_id(self) -> None:
        """COCO person category ``1`` remains raw when no category remapping is supplied."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label=None,
            num_keypoints_per_class=[17],
        )
        _, target = converter(
            _IMAGE,
            {"image_id": 7, "annotations": [_make_keypoint_annotation(category_id=1)]},
        )

        assert target["labels"].shape == (1,)
        assert target["labels"].item() == 1

    def test_num_keypoints_zero_annotation_retains_instance_for_box_supervision(self) -> None:
        """All-zero-visibility keypoints must not drop the instance; box/class targets are still valid."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label=None,
            num_keypoints_per_class=[17],
        )
        _, target = converter(
            _IMAGE,
            {"image_id": 3, "annotations": [_make_keypoint_annotation(keypoints=[0.0] * (17 * 3))]},
        )

        assert target["boxes"].shape == (1, 4)
        assert target["labels"].shape == (1,)
        assert target["keypoints"].shape == (1, 17, 3)
        assert torch.count_nonzero(target["keypoints"]) == 0

    def test_empty_image_uses_schema_max_shape(self) -> None:
        """Empty images should emit ``(0, max(num_keypoints_per_class), 3)`` keypoint tensors."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label={1: 0},
            num_keypoints_per_class=[2, 1],
        )
        _, target = converter(_IMAGE, {"image_id": 99, "annotations": []})

        assert target["keypoints"].shape == (0, 2, 3)

    def test_multiclass_keypoints_use_schema_max_shape(self) -> None:
        """Multi-class keypoint targets should be padded to Kmax, not schema sum."""
        converter = ConvertCoco(
            include_masks=False,
            include_keypoints=True,
            cat2label=None,
            num_keypoints_per_class=[2, 1],
        )
        _, target = converter(
            _IMAGE,
            {
                "image_id": 100,
                "annotations": [
                    _make_keypoint_annotation(category_id=0, keypoints=[1.0, 2.0, 2.0, 3.0, 4.0, 2.0]),
                    _make_keypoint_annotation(category_id=1, keypoints=[5.0, 6.0, 2.0]),
                ],
            },
        )

        assert target["labels"].tolist() == [0, 1]
        assert target["keypoints"].shape == (2, 2, 3)
        torch.testing.assert_close(
            target["keypoints"][0],
            torch.tensor([[1.0, 2.0, 2.0], [3.0, 4.0, 2.0]], dtype=torch.float32),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            target["keypoints"][1],
            torch.tensor([[5.0, 6.0, 2.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
            rtol=1e-4,
            atol=1e-6,
        )


class TestBuildCocoKeypointMode:
    """COCO builder mode switch for person keypoints."""

    @pytest.mark.parametrize(
        "use_keypoints,ann_name,remap_category_ids",
        [
            pytest.param(True, "person_keypoints_train2017.json", True, id="keypoint-mode"),
            pytest.param(False, "instances_train2017.json", False, id="default-keeps-raw-coco-ids"),
        ],
    )
    def test_mode_selects_annotation_file_and_id_remapping(
        self, mocker, tmp_path: Path, use_keypoints: bool, ann_name: str, remap_category_ids: bool
    ) -> None:
        """Keypoint mode trains on person keypoints and remaps ids; default mode keeps raw ids for pretrained
        weights."""
        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=use_keypoints)
        mocker.patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target))
        mock_dataset = mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=object())

        build_coco("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        ann_file = Path(mock_dataset.call_args.args[1])
        assert ann_file.parent.name == "annotations"
        assert ann_file.name == ann_name
        assert kwargs["include_keypoints"] is use_keypoints
        assert kwargs["remap_category_ids"] is remap_category_ids


class TestBuildKeypointCat2Label:
    """Unit tests for ``_build_keypoint_cat2label`` schema alignment."""

    def _person_coco(self, cat_id: int = 1) -> types.SimpleNamespace:
        """Return a minimal COCO-like object with a single keypoint-bearing person category."""
        return types.SimpleNamespace(
            cats={cat_id: {"name": "person", "keypoints": ["nose"] * 17}},
            anns={},
        )

    def test_legacy_bgfirst_schema_maps_person_to_slot_1(self) -> None:
        """Legacy [0, 17] schema maps person (cat_id=1) to slot 1, not slot 0."""
        from rfdetr.datasets.coco import _build_keypoint_cat2label

        result = _build_keypoint_cat2label(self._person_coco(cat_id=1), num_keypoints_per_class=[0, 17])

        assert result == {1: 1}

    def test_mixed_detection_and_keypoint_categories(self) -> None:
        """Non-keypoint categories fill free slots after keypoint categories are assigned."""
        from rfdetr.datasets.coco import _build_keypoint_cat2label

        coco = types.SimpleNamespace(
            cats={
                1: {"name": "person", "keypoints": ["nose"] * 17},
                3: {"name": "car"},
            },
            anns={},
        )
        result = _build_keypoint_cat2label(coco, num_keypoints_per_class=[17])

        assert result == {1: 0, 3: 1}


class TestScaleJitter:
    """``scale_jitter`` controls the resize-crop branch independently of ``aug_config``."""

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(make_coco_transforms, id="standard"),
            pytest.param(make_coco_transforms_square_div_64, id="square_div_64"),
        ],
    )
    @pytest.mark.parametrize(
        "scale_jitter,expected",
        [
            pytest.param(True, True, id="enabled_keeps_crop"),
            pytest.param(False, False, id="disabled_drops_crop"),
        ],
    )
    def test_transform_factories_forward_scale_jitter(self, mocker, factory, scale_jitter, expected):
        """Both transform factories pass scale_jitter through to the torchvision-native resize pipeline unchanged."""
        mock_build = mocker.patch("rfdetr.datasets.coco._build_train_resize_transforms")

        factory("train", 640, scale_jitter=scale_jitter)

        assert mock_build.call_args.kwargs["scale_jitter"] is expected

    def test_empty_aug_config_no_longer_affects_crop_branch(self, mocker):
        """aug_config={} disables the augmentation stack only — crop branch stays on by default."""
        mock_build = mocker.patch("rfdetr.datasets.coco._build_train_resize_transforms")

        make_coco_transforms("train", 640, aug_config={})

        assert mock_build.call_args.kwargs["scale_jitter"] is True

    @pytest.mark.parametrize(
        "scale_jitter,expected",
        [
            pytest.param(True, True, id="enabled_keeps_crop"),
            pytest.param(False, False, id="disabled_drops_crop"),
        ],
    )
    def test_non_empty_aug_config_forwards_scale_jitter_to_albumentations(self, mocker, scale_jitter, expected):
        """Non-empty aug_config routes to the Albumentations resize config with scale_jitter forwarded."""
        mock_build = mocker.patch("rfdetr.datasets.coco._build_train_resize_config", return_value=[])
        mocker.patch("rfdetr.datasets.coco.AlbumentationsWrapper.from_config", return_value=mocker.MagicMock())

        make_coco_transforms("train", 640, aug_config={"HorizontalFlip": {"p": 0.5}}, scale_jitter=scale_jitter)

        assert mock_build.call_args.kwargs["scale_jitter"] is expected


def _make_gradient_image(width: int, height: int) -> Image.Image:
    """Build a deterministic RGB gradient image with real pixel content for interpolation comparisons."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    grid = np.broadcast_to(x, (height, width))
    channel_b = np.broadcast_to(y[:, None], (height, width))
    array = np.stack([grid, channel_b, (grid // 2 + channel_b // 2)], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class TestPipelineParity:
    """Torchvision and Albumentations eval pipelines produce statistically close resize+normalize output.

    The backends interpolate differently (torchvision BILINEAR + antialias, Albumentations cv2 INTER_LINEAR without
    antialias), so exact pixel parity is not expected; see the backend-switch UserWarning in
    ``_build_torchvision_pipeline``. Shape, dtype and output distribution are compared instead.
    """

    @_requires_albumentations
    def test_eval_resize_normalize_stats_close(self) -> None:
        """The torchvision and Albumentations eval pipelines agree on shape, dtype, mean and std."""
        image = _make_gradient_image(120, 90)
        target = {
            "boxes": torch.tensor([[10.0, 5.0, 60.0, 45.0]]),
            "labels": torch.tensor([1]),
            "orig_size": torch.tensor([90, 120]),
            "size": torch.tensor([90, 120]),
        }
        pipeline_kwargs = {
            "image_set": "val",
            "resolution": 128,
            "scales": [128],
            "square": False,
            "aug_config": None,
            "gpu_postprocess": False,
            "keypoint_flip_pairs": None,
        }

        tv_pipeline = _build_torchvision_pipeline(**pipeline_kwargs)
        albu_pipeline = _build_albumentations_pipeline(**pipeline_kwargs)

        tv_image, _ = tv_pipeline(image, dict(target))
        albu_image, _ = albu_pipeline(image, dict(target))

        assert tv_image.shape == albu_image.shape
        assert tv_image.dtype == albu_image.dtype
        torch.testing.assert_close(tv_image.mean(), albu_image.mean(), atol=0.05, rtol=0)
        torch.testing.assert_close(tv_image.std(), albu_image.std(), atol=0.05, rtol=0)


def _write_coco_split(
    root: Path, images: dict[str, tuple[int, int]], annotations: List[Dict], categories: List[Dict]
) -> Path:
    """Write blank images and an ``annotations.json`` for a minimal COCO split.

    Args:
        root: Directory receiving the images and ``annotations.json``, created if missing.
        images: File name to ``(width, height)``; image ids count up from 1 in insertion order.
        annotations: COCO annotation dicts.
        categories: COCO category dicts.

    Returns:
        Path of the written ``annotations.json``.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _write_coco_split(root, {"a.jpg": (4, 3)}, [], [_cat(1, "x")]).name
        'annotations.json'
        >>> (root / "a.jpg").exists()
        True
    """
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for image_id, (name, (width, height)) in enumerate(images.items(), start=1):
        Image.new("RGB", (width, height)).save(root / name, quality=95)
        entries.append({"id": image_id, "file_name": name, "width": width, "height": height})
    ann_file = root / "annotations.json"
    ann_file.write_text(
        json.dumps({"images": entries, "annotations": annotations, "categories": categories}), encoding="utf-8"
    )
    return ann_file


def _cat(category_id: object, name: str, supercategory: str = "none") -> Dict:
    """Build a COCO category dict.

    Examples:
        >>> _cat(2, "car", "vehicle")
        {'id': 2, 'name': 'car', 'supercategory': 'vehicle'}
    """
    return {"id": category_id, "name": name, "supercategory": supercategory}


class TestCocoDetectionZeroAnnotations:
    """CocoDetection handles images and datasets that carry no annotations."""

    def test_unannotated_image_yields_empty_boxes_and_labels(self, tmp_path: Path) -> None:
        """An image with no annotations yields boxes (0, 4) float32 and labels (0,) int64 tensors."""
        annotation = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}
        ann_file = _write_coco_split(
            tmp_path, {"img1.jpg": (100, 100), "img2.jpg": (100, 100)}, [annotation], [_cat(1, "cat", "animal")]
        )
        dataset = CocoDetection(tmp_path, ann_file, transforms=None)

        _, target = dataset[dataset.ids.index(2)]

        assert target["boxes"].shape == torch.Size([0, 4])
        assert target["labels"].shape == torch.Size([0])
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64

    def test_all_parent_categories_kept_when_dataset_has_no_annotations(self, tmp_path: Path) -> None:
        """Hierarchy categories with zero total annotations are all kept (the documented fallback)."""
        ann_file = _write_coco_split(tmp_path, {"img1.jpg": (100, 100)}, [], [_cat(1, "a", "b"), _cat(2, "b", "a")])
        dataset = CocoDetection(tmp_path, ann_file, transforms=None, remap_category_ids=True)

        _, target = dataset[0]

        assert dataset.cat2label == {1: 0, 2: 1}
        assert target["boxes"].shape == torch.Size([0, 4])
        assert target["labels"].shape == torch.Size([0])


class TestCocoDetectionDraftDecode:
    """Draft-decoded JPEG samples retain image/annotation geometry alignment."""

    def test_annotation_scaling_keeps_polygon_axes_and_keypoint_visibility(self) -> None:
        """Non-square drafting scales polygon axes per axis and leaves keypoint visibility unchanged."""
        annotation = {
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "area": 1200.0,
            "category_id": 1,
            "segmentation": [[10.0, 20.0, 40.0, 20.0, 40.0, 60.0]],
            "keypoints": [10.0, 20.0, 2.0, 40.0, 60.0, 0.0],
        }

        scaled = scale_coco_annotation(annotation, x_scale=0.5, y_scale=0.25)

        assert annotation["segmentation"] == [[10.0, 20.0, 40.0, 20.0, 40.0, 60.0]]
        assert scaled["bbox"] == [5.0, 5.0, 15.0, 10.0]
        assert scaled["area"] == 150.0
        assert scaled["segmentation"] == [[5.0, 5.0, 20.0, 5.0, 20.0, 15.0]]
        assert scaled["keypoints"] == [5.0, 5.0, 2.0, 20.0, 15.0, 0.0]

    def test_odd_jpeg_scales_each_annotation_axis_to_decoded_image(self, tmp_path: Path) -> None:
        """Draft decoding an odd-sized JPEG scales boxes, keypoints and area by the actual x and y factors."""
        original_width, original_height = 1921, 1081
        annotation = {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100.0, 200.0, 300.0, 400.0],
            "area": 120000.0,
            "iscrowd": 0,
            "keypoints": [100.0, 200.0, 2.0],
        }
        annotation_path = _write_coco_split(
            tmp_path, {"draft.jpg": (original_width, original_height)}, [annotation], [_cat(1, "person", "person")]
        )
        dataset = CocoDetection(
            tmp_path,
            annotation_path,
            transforms=None,
            include_keypoints=True,
            num_keypoints_per_class=[1],
            draft_size=512,
        )

        _, target = dataset[0]

        decoded_height, decoded_width = target["orig_size"].tolist()
        assert (decoded_width, decoded_height) != (original_width, original_height)
        x_scale = decoded_width / original_width
        y_scale = decoded_height / original_height
        expected_boxes = torch.tensor([[100.0 * x_scale, 200.0 * y_scale, 400.0 * x_scale, 600.0 * y_scale]])
        torch.testing.assert_close(target["boxes"], expected_boxes, rtol=1e-5, atol=1e-5)
        expected_keypoints = torch.tensor([[[100.0 * x_scale, 200.0 * y_scale, 2.0]]])
        torch.testing.assert_close(target["keypoints"], expected_keypoints, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(target["area"], torch.tensor([120000.0 * x_scale * y_scale]), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        ("image_set", "include_masks", "expected"),
        [("val", False, None), ("train", True, None), ("train", False, 600)],
    )
    def test_draft_size_with_scale_jitter(self, image_set: str, include_masks: bool, expected: int | None) -> None:
        """Validation and mask datasets disable draft decoding; plain training keeps the 600-pixel crop-branch floor."""
        assert draft_size_for_transforms(image_set, 512, scale_jitter=True, include_masks=include_masks) == expected

    @pytest.fixture
    def mock_coco_dataset(self, mocker) -> object:
        """Patch transforms and ``CocoDetection`` so builders run without disk access; return the dataset mock."""
        mocker.patch("rfdetr.datasets.coco.make_coco_transforms", return_value=mocker.MagicMock())
        return mocker.patch("rfdetr.datasets.coco.CocoDetection", return_value=mocker.MagicMock())

    @pytest.mark.parametrize(("scale_jitter", "expected_draft_size"), [(False, 512), (True, 600)])
    def test_build_coco_forwards_scale_jitter_to_draft_size(
        self, tmp_path: Path, mock_coco_dataset: object, scale_jitter: bool, expected_draft_size: int
    ) -> None:
        """``build_coco`` gives the draft decoder the configured jitter floor."""
        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=False)
        args.scale_jitter = scale_jitter

        build_coco("train", args, resolution=512)

        assert mock_coco_dataset.call_args.kwargs["draft_size"] == expected_draft_size

    @pytest.mark.parametrize(("scale_jitter", "expected_draft_size"), [(False, 512), (True, 600)])
    def test_build_roboflow_forwards_scale_jitter_to_draft_size(
        self, tmp_path: Path, mock_coco_dataset: object, scale_jitter: bool, expected_draft_size: int
    ) -> None:
        """``build_roboflow_from_coco`` gives the draft decoder the configured jitter floor."""
        args = _pipeline_args(
            tmp_path,
            augmentation_backend="cpu",
            use_grouppose_keypoints=False,
            num_keypoints_per_class=[],
            aug_config={},
            scale_jitter=scale_jitter,
        )

        build_roboflow_from_coco("train", args, resolution=512)

        assert mock_coco_dataset.call_args.kwargs["draft_size"] == expected_draft_size


_PHANTOM_ROOT_CATEGORIES = [
    {"id": 0, "name": "eggmasses", "supercategory": "none"},
    {"id": 1, "name": "empty_pot", "supercategory": "eggmasses"},
    {"id": 2, "name": "stake", "supercategory": "eggmasses"},
    {"id": 3, "name": "tree", "supercategory": "eggmasses"},
    {"id": 4, "name": "trunk", "supercategory": "eggmasses"},
]

# Annotated category ids of a Roboflow hierarchy split: the synthetic root (id 0) is either unused or annotated.
_ROOT_ANNOTATION_CASES = [pytest.param([1, 4], id="unannotated-root"), pytest.param([0, 1], id="annotated-root")]


def _write_roboflow_hierarchy_split(directory: Path, annotated_ids: List[int]) -> Path:
    """Write a Roboflow-style COCO split whose category list starts with a synthetic root node.

    Args:
        directory: Split directory, created if missing, receiving the image and the annotation file.
        annotated_ids: Category ids that each receive one annotation on the single image.

    Returns:
        Path of the written ``_annotations.coco.json``.

    Examples:
        >>> import tempfile
        >>> split = Path(tempfile.mkdtemp()) / "train"
        >>> _write_roboflow_hierarchy_split(split, [1]).name
        '_annotations.coco.json'
    """
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100)).save(directory / "img1.jpg")
    annotations = [
        {"id": index, "image_id": 1, "category_id": category_id, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}
        for index, category_id in enumerate(annotated_ids)
    ]
    ann_file = directory / "_annotations.coco.json"
    ann_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
                "annotations": annotations,
                "categories": _PHANTOM_ROOT_CATEGORIES,
            }
        ),
        encoding="utf-8",
    )
    return ann_file


def _open_hierarchy_train(tmp_path: Path, annotated_ids: List[int]) -> CocoDetection:
    """Write a hierarchy split under ``tmp_path / "train"`` and open it with category remapping.

    Examples:
        >>> import tempfile
        >>> _open_hierarchy_train(Path(tempfile.mkdtemp()), [1, 4]).cat2label  # doctest: +ELLIPSIS
        loading annotations...{1: 0, 2: 1, 3: 2, 4: 3}
    """
    ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", annotated_ids)
    return CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)


class TestFilterParentCategories:
    """Unit tests for the grouping-category filter shared by class names, class count and label remapping."""

    @pytest.mark.parametrize(
        ("categories", "annotated", "expected_ids"),
        [
            # A category named as another's supercategory and carrying no annotation is removed.
            pytest.param(_PHANTOM_ROOT_CATEGORIES, {1, 2, 3, 4}, [1, 2, 3, 4], id="unannotated-root-dropped"),
            # A parent that owns annotations stays, so its annotations keep a label slot.
            pytest.param(_PHANTOM_ROOT_CATEGORIES, {0, 1}, [0, 1, 2, 3, 4], id="annotated-parent-kept"),
            # Output order follows category id, matching the contiguous label indices derived from it.
            pytest.param([_cat(2, "cat"), _cat(1, "dog")], set(), [1, 2], id="sorted-by-id"),
            # Exports shipping string ids still match the int-keyed annotated set.
            pytest.param([_cat("1", "vehicle"), _cat("2", "car", "vehicle")], {1, 2}, ["1", "2"], id="string-ids"),
            # An annotated leaf keeps its slot even when an unannotated grouping node reuses its name.
            pytest.param(
                [_cat(1, "tree"), _cat(2, "trunk", "tree"), _cat(3, "tree", "tree")], {2, 3}, [2, 3], id="shared-name"
            ),
            # A category naming itself as its own supercategory is not its own parent.
            pytest.param(
                [_cat(1, "person", "person"), _cat(2, "vehicle"), _cat(3, "car", "vehicle")],
                {3},
                [1, 3],
                id="self-referential-supercategory",
            ),
            # A self-referencing name that another category groups under is still dropped when unannotated.
            pytest.param(
                [_cat(1, "person", "person"), _cat(2, "rider", "person")],
                {2},
                [2],
                id="self-referential-name-is-parent",
            ),
            # Some (not all, not none) categories annotated inside an all-parent cycle: only those survive.
            pytest.param(
                [_cat(1, "a", "c"), _cat(2, "b", "a"), _cat(3, "c", "b")], {2}, [2], id="partial-annotation-in-cycle"
            ),
            # Two disjoint hierarchies filtered together: the annotated root stays, the unannotated sibling drops.
            pytest.param(
                [
                    _cat(0, "rootA"),
                    _cat(1, "a1", "rootA"),
                    _cat(2, "a2", "rootA"),
                    _cat(5, "rootB"),
                    _cat(6, "b1", "rootB"),
                    _cat(7, "b2", "rootB"),
                ],
                {1, 2, 5, 6, 7},
                [1, 2, 5, 6, 7],
                id="two-independent-roots",
            ),
        ],
    )
    def test_keeps_expected_category_ids(self, categories: List[Dict], annotated: set, expected_ids: list) -> None:
        """Unannotated grouping categories are dropped; annotated and leaf categories survive in id order."""
        result = filter_parent_categories(categories, annotated)

        assert [category["id"] for category in result] == expected_ids

    @pytest.mark.parametrize(
        "categories",
        [
            pytest.param([_cat(1, "dog"), _cat(2, "cat")], id="flat-dataset"),
            pytest.param([_cat(1, "a", "b"), _cat(2, "b", "a")], id="every-category-is-a-parent"),
        ],
    )
    def test_returns_full_list_when_nothing_is_a_droppable_parent(self, categories: List[Dict]) -> None:
        """Flat datasets stay untouched, and a pathological all-parent cycle keeps the full list instead of nothing."""
        assert filter_parent_categories(categories, set()) == categories

    def test_category_without_a_name_reports_the_offending_id(self) -> None:
        """A category missing the required ``name`` field names itself in the error instead of raising a bare id."""
        categories = [_cat(1, "vehicle"), {"id": 2, "supercategory": "vehicle"}]

        with pytest.raises(KeyError) as exc_info:
            filter_parent_categories(categories, {1})

        assert "missing the required 'name' field" in str(exc_info.value)

    def test_mixed_and_missing_ids_sort_after_numeric_categories(self) -> None:
        """Numeric IDs sort consistently even when malformed entries are present."""
        categories = [{"name": "missing", "supercategory": "none"}, _cat("2", "two"), _cat(1, "one")]

        result = filter_parent_categories(categories, set())

        assert [category["name"] for category in result] == ["one", "two", "missing"]


class TestAnnotatedCategoryIds:
    """Unit tests for collecting the annotated category ids of a parsed COCO file."""

    @pytest.mark.parametrize(
        ("coco_data", "expected"),
        [
            # Every distinct ``category_id`` referenced by an annotation is reported once.
            pytest.param(
                {"annotations": [{"category_id": 3}, {"category_id": 3}, {"category_id": 5}]}, {3, 5}, id="distinct"
            ),
            # A category-only annotation file reports no annotated categories.
            pytest.param({"categories": _PHANTOM_ROOT_CATEGORIES}, set(), id="missing-annotations-key"),
        ],
    )
    def test_collects_distinct_annotated_ids(self, coco_data: dict, expected: set) -> None:
        """The annotated category ids of a COCO dict are returned as a set, empty without annotations."""
        assert annotated_category_ids(coco_data) == expected


class TestRootCategoryLabelSpace:
    """Roboflow COCO exports prepend a root category; count, remap and names agree whether or not it is annotated."""

    @pytest.mark.parametrize(
        ("annotated_ids", "expected"),
        [
            pytest.param([1, 4], {1: 0, 2: 1, 3: 2, 4: 3}, id="unannotated-root"),
            pytest.param([0, 1], {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, id="annotated-root"),
        ],
    )
    def test_cat2label_gives_an_unannotated_root_no_slot(
        self, tmp_path: Path, annotated_ids: List[int], expected: Dict[int, int]
    ) -> None:
        """An unannotated root consumes no label slot; an annotated root keeps its identity slot."""
        assert _open_hierarchy_train(tmp_path, annotated_ids).cat2label == expected

    @pytest.mark.parametrize(
        ("annotated_ids", "expected"),
        [
            pytest.param([1, 4], {0: 1, 1: 2, 2: 3, 3: 4}, id="unannotated-root"),
            pytest.param([0, 1], {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}, id="annotated-root"),
        ],
    )
    def test_label2cat_is_exposed_for_the_evaluator(
        self, tmp_path: Path, annotated_ids: List[int], expected: Dict[int, int]
    ) -> None:
        """The reverse mapping reaches the COCO API object so predictions convert back to original ids."""
        assert _open_hierarchy_train(tmp_path, annotated_ids).coco.label2cat == expected

    @pytest.mark.parametrize(
        ("annotated_ids", "expected"),
        [pytest.param([1, 4], [0, 3], id="unannotated-root"), pytest.param([0, 1], [0, 1], id="annotated-root")],
    )
    def test_targets_carry_filtered_label_indices(
        self, tmp_path: Path, annotated_ids: List[int], expected: List[int]
    ) -> None:
        """Targets carry the filtered label indices rather than root-shifted ones; an annotated root keeps its slot."""
        _, target = _open_hierarchy_train(tmp_path, annotated_ids)[0]

        assert target["labels"].tolist() == expected

    @pytest.mark.parametrize("annotated_ids", _ROOT_ANNOTATION_CASES)
    def test_detected_num_classes_matches_cat2label(self, tmp_path: Path, annotated_ids: List[int]) -> None:
        """The auto-detected head size equals the number of label slots the remapping actually uses."""
        dataset = _open_hierarchy_train(tmp_path, annotated_ids)

        detected = RFDETR._detect_num_classes_for_training(str(tmp_path))

        assert detected == len(set(dataset.cat2label.values()))

    @pytest.mark.parametrize("annotated_ids", _ROOT_ANNOTATION_CASES)
    def test_class_names_and_class_count_share_one_basis(self, tmp_path: Path, annotated_ids: List[int]) -> None:
        """The detected head size and the loaded class names come from one filtered category list, so they agree."""
        _write_roboflow_hierarchy_split(tmp_path / "train", annotated_ids)

        assert RFDETR._detect_num_classes_for_training(str(tmp_path)) == len(RFDETR._load_classes(str(tmp_path)))

    @pytest.mark.parametrize("annotated_ids", _ROOT_ANNOTATION_CASES)
    def test_load_classes_matches_cat2label_order(self, tmp_path: Path, annotated_ids: List[int]) -> None:
        """Class names line up positionally with the label indices assigned by the remapping."""
        dataset = _open_hierarchy_train(tmp_path, annotated_ids)

        names = RFDETR._load_classes(str(tmp_path))

        assert names == [dataset.coco.cats[dataset.label2cat[label]]["name"] for label in range(len(names))]


class TestDetectNumClassesWebDataset:
    """A packed webdataset shard directory carries the class count directly, without a raw annotation file to read.

    Regression: ``_detect_num_classes_for_training`` once had no webdataset branch, so a shard directory fell through
    to ``_load_classes``, raised FileNotFoundError that the caller swallowed at debug level, and auto-detection never
    worked for this ``dataset_file``.
    """

    def test_remap_policy_matches_the_filtered_category_count(self, tmp_path: Path) -> None:
        """A 'remap' pack detects the same class count the loose-file COCO path would for the same categories."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(tmp_path / "train", ann_file, shard_dir, split="train", category_ids="remap")

        assert RFDETR._detect_num_classes_for_training(str(shard_dir)) == len(set(dataset.cat2label.values()))

    def test_raw_policy_uses_max_category_id_plus_one(self, tmp_path: Path) -> None:
        """A 'raw' pack detects max(category_id) + 1, matching dataset_file='coco''s existing convention."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(tmp_path / "train", ann_file, shard_dir, split="train", category_ids="raw")

        assert RFDETR._detect_num_classes_for_training(str(shard_dir)) == 5

    def test_raw_policy_retains_unannotated_declared_category(self, tmp_path: Path) -> None:
        """Evaluation may contain the highest declared category even when training has no instance of it."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1])
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(tmp_path / "train", ann_file, shard_dir, category_ids="raw")

        assert RFDETR._detect_num_classes_for_training(str(shard_dir)) == 5

    def test_keypoint_mode_does_not_read_the_shard_index(self, tmp_path: Path) -> None:
        """Keypoint training never reaches the webdataset branch: that format rejects keypoints outright."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        shard_dir = tmp_path / "shards"
        pack_coco_to_shards(tmp_path / "train", ann_file, shard_dir, split="train")

        with pytest.raises(FileNotFoundError):
            RFDETR._detect_num_classes_for_training(str(shard_dir), use_grouppose_keypoints=True)


class TestCrossSplitLabelSpace:
    """Val/test splits must reuse the train label space whatever their own annotation coverage is."""

    @pytest.fixture
    def train_val_args(self, tmp_path: Path) -> types.SimpleNamespace:
        """Builder args for a root whose train split annotates ids [0, 1] while val annotates only [1]."""
        _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        _write_roboflow_hierarchy_split(tmp_path / "valid", [1])
        return _pipeline_args(tmp_path)

    def test_val_split_reuses_the_train_label_mapping(self, train_val_args: types.SimpleNamespace) -> None:
        """A grouping category annotated in train only keeps its train label slot in val instead of shifting it."""
        train_dataset = build_roboflow_from_coco("train", train_val_args, resolution=64)
        val_dataset = build_roboflow_from_coco("val", train_val_args, resolution=64)

        assert val_dataset.cat2label == train_dataset.cat2label

    def test_val_targets_use_train_label_indices(self, train_val_args: types.SimpleNamespace) -> None:
        """Val targets carry the label index training assigned, not the one val's own coverage would produce."""
        val_dataset = build_roboflow_from_coco("val", train_val_args, resolution=64)

        _, target = val_dataset[0]

        assert target["labels"].tolist() == [1]

    def test_injected_cat2label_replaces_the_split_local_mapping(self, tmp_path: Path) -> None:
        """An explicitly supplied mapping wins over the one the split would derive from its own annotations."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "valid", [1])
        injected = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

        dataset = CocoDetection(
            tmp_path / "valid", ann_file, transforms=None, remap_category_ids=True, cat2label=injected
        )

        assert dataset.cat2label == injected

    def test_cat2label_without_remapping_raises(self, tmp_path: Path) -> None:
        """Supplying a mapping while remapping is disabled fails loudly instead of silently ignoring it."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "valid", [1])

        with pytest.raises(ValueError) as exc_info:
            CocoDetection(tmp_path / "valid", ann_file, transforms=None, cat2label={1: 0})

        assert "remap_category_ids" in str(exc_info.value)
