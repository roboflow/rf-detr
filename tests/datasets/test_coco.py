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

import pytest
import torch
from PIL import Image

from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema
from rfdetr.datasets.coco import (
    CocoDetection,
    ConvertCoco,
    annotated_category_ids,
    build_coco,
    build_roboflow_from_coco,
    draft_size_for_transforms,
    filter_parent_categories,
    scale_coco_annotation,
)
from rfdetr.detr import RFDETR
from rfdetr.utilities import PackedTargets, pack_targets

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
        "do_random_resize_via_padding": False,
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

    def test_roboflow_hierarchy_filters_parent(self, tmp_path: Path) -> None:
        """Roboflow exports include a parent node — only leaf categories kept."""
        categories = [
            {"id": 0, "name": "annotations", "supercategory": "none"},
            {"id": 1, "name": "dog", "supercategory": "annotations"},
            {"id": 2, "name": "cat", "supercategory": "annotations"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_flat_none_supercategory_keeps_all(self, tmp_path: Path) -> None:
        """Flat datasets where every category has supercategory 'none' (#609)."""
        categories = [
            {"id": 1, "name": "dog", "supercategory": "none"},
            {"id": 2, "name": "cat", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_mixed_supercategories_keeps_all(self, tmp_path: Path) -> None:
        """Mix of 'none' and non-'none' supercategories where no category is a parent of another.

        'animal' appears as a supercategory but is not itself a category name, so ``has_children`` is empty and all
        categories pass the ``name not in has_children`` filter — both 'dog' and 'cat' are returned.
        """
        categories = [
            {"id": 1, "name": "dog", "supercategory": "none"},
            {"id": 2, "name": "cat", "supercategory": "animal"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat"]

    def test_category_named_none_does_not_empty_list(self, tmp_path: Path) -> None:
        """If a category is literally named 'none' and all supercategories are placeholders, the loader must return all
        class names instead of []."""

        categories = [
            {"id": 1, "name": "none", "supercategory": "none"},
            {"id": 2, "name": "dog", "supercategory": "none"},
            {"id": 3, "name": "cat", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["none", "dog", "cat"]

    def test_mixed_hierarchy_leaf_and_standalone_forwarding(self, tmp_path: Path) -> None:
        """Mixed hierarchy: only leaf classes + standalone top-level categories should be forwarded.

        Parent/grouping nodes are dropped.
        """
        categories = [
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
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        expected = [
            "dog",
            "cat",
            "eagle",
            "pigeon",
            "car",
            "truck",
            "toaster",
            "microwave",
            "person",
        ]
        assert result == expected

    def test_placeholder_values_treated_as_no_parent(self, tmp_path: Path) -> None:
        """Placeholders like None, '', and 'null' should be treated the same as 'none'."""
        categories = [
            {"id": 1, "name": "dog", "supercategory": None},
            {"id": 2, "name": "cat", "supercategory": ""},
            {"id": 3, "name": "elephant", "supercategory": "null"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["dog", "cat", "elephant"]

    def test_unsorted_category_ids_return_id_sorted_class_order(self, tmp_path: Path) -> None:
        """Returned class names must follow category-ID order for stable index mapping."""
        categories = [
            {"id": 30, "name": "truck", "supercategory": "vehicle"},
            {"id": 10, "name": "vehicle", "supercategory": "none"},
            {"id": 20, "name": "car", "supercategory": "vehicle"},
            {"id": 40, "name": "person", "supercategory": "none"},
        ]
        _write_coco_json(tmp_path / "train" / "_annotations.coco.json", categories)
        result = RFDETR._load_classes(str(tmp_path))
        assert result == ["car", "truck", "person"]


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
            do_random_resize_via_padding=False,
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
            self.do_random_resize_via_padding = False
            self.patch_size = 16
            self.num_windows = 4
            self.dataset_dir = "/nonexistent/o365"
            self.coco_path = "/nonexistent/o365"

    def _call_build_o365_raw(self, augmentation_backend, square_resize_div_64=False):
        """Call build_o365_raw with mocked CocoDetection and transform builders."""
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.o365 import build_o365_raw

        args = self._FakeArgs(augmentation_backend=augmentation_backend, square_resize_div_64=square_resize_div_64)
        fake_dataset = MagicMock()

        with (
            patch("rfdetr.datasets.o365.CocoDetection", return_value=fake_dataset),
            patch("rfdetr.datasets.o365.make_coco_transforms") as mock_transform,
            patch("rfdetr.datasets.o365.make_coco_transforms_square_div_64") as mock_sq_transform,
        ):
            mock_transform.return_value = MagicMock()
            mock_sq_transform.return_value = MagicMock()
            result = build_o365_raw("train", args, resolution=640)
            return result, mock_transform, mock_sq_transform

    def test_cpu_backend_no_warning(self):
        """Cpu backend does not call logger.warning with O365 content."""
        from unittest.mock import patch

        with patch("rfdetr.datasets.o365.logger") as mock_logger:
            self._call_build_o365_raw("cpu")
        o365_warns = [c for c in mock_logger.warning.call_args_list if "O365" in str(c)]
        assert len(o365_warns) == 0, "cpu backend must not warn about O365 GPU augmentation"

    def test_auto_backend_emits_warning(self):
        """Auto + CUDA + kornia available: logger.warning about O365 Phase 1 limitation."""
        from unittest.mock import patch

        from rfdetr.config import AugmentationBackend

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.object(AugmentationBackend, "_is_available", lambda self: True),
            patch("rfdetr.datasets.o365.logger") as mock_logger,
        ):
            self._call_build_o365_raw("auto")
        o365_warns = [c for c in mock_logger.warning.call_args_list if "O365" in str(c)]
        assert len(o365_warns) >= 1, "auto backend must warn about O365 GPU aug limitation"

    def test_auto_backend_no_cuda_no_warning(self):
        """Auto + no CUDA: resolves to cpu, no O365 warning emitted."""
        from unittest.mock import patch

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False),
            patch("rfdetr.datasets.o365.logger") as mock_logger,
        ):
            self._call_build_o365_raw("auto")
        o365_warns = [c for c in mock_logger.warning.call_args_list if "O365" in str(c)]
        assert len(o365_warns) == 0, "auto + no CUDA must not warn about O365 GPU aug"

    def test_gpu_postprocess_false_for_cpu_backend(self):
        """Cpu backend passes gpu_postprocess=False (or omits it) to make_coco_transforms."""
        _, mock_transform, _ = self._call_build_o365_raw("cpu")
        call_kwargs = mock_transform.call_args.kwargs if mock_transform.call_args else {}
        assert call_kwargs.get("gpu_postprocess", False) is False

    def test_gpu_postprocess_true_for_auto_backend(self):
        """Auto + CUDA + kornia available: gpu_postprocess=True passed to make_coco_transforms."""
        from unittest.mock import patch

        from rfdetr.config import AugmentationBackend

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.object(AugmentationBackend, "_is_available", lambda self: True),
        ):
            _, mock_transform, _ = self._call_build_o365_raw("auto")
        call_kwargs = mock_transform.call_args.kwargs if mock_transform.call_args else {}
        assert call_kwargs.get("gpu_postprocess", False) is True

    def test_gpu_postprocess_false_for_auto_no_cuda(self):
        """Auto + no CUDA: gpu_postprocess=False so CPU Normalize is retained."""
        from unittest.mock import patch

        with patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False):
            _, mock_transform, _ = self._call_build_o365_raw("auto")
        call_kwargs = mock_transform.call_args.kwargs if mock_transform.call_args else {}
        assert call_kwargs.get("gpu_postprocess", False) is False, "auto + no CUDA must not strip CPU Normalize"

    def test_square_resize_uses_square_transform(self):
        """square_resize_div_64=True delegates to make_coco_transforms_square_div_64."""
        _, mock_transform, mock_sq_transform = self._call_build_o365_raw("cpu", square_resize_div_64=True)
        mock_sq_transform.assert_called_once()
        mock_transform.assert_not_called()

    def test_gpu_backend_no_cuda_raises_runtime_error(self):
        """Gpu backend must fail fast when CUDA is unavailable."""
        from unittest.mock import patch

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False),
            pytest.raises(RuntimeError, match="CUDA"),
        ):
            self._call_build_o365_raw("gpu")

    def test_gpu_backend_no_kornia_raises_import_error(self):
        """Gpu backend must raise with install hint when kornia is missing."""
        from unittest.mock import patch

        from rfdetr.config import AugmentationBackend

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA),
            pytest.raises(ImportError, match="rfdetr\\[augment\\]"),
        ):
            self._call_build_o365_raw("gpu")


class TestBuildRoboflowFromCocoBackendResolution:
    """Roboflow COCO builder should resolve backend for gpu_postprocess consistently."""

    def test_auto_no_cuda_keeps_cpu_normalize(self):
        """Auto + no CUDA must set gpu_postprocess=False."""
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.coco import build_roboflow_from_coco

        args = types.SimpleNamespace(
            dataset_dir="/fake/dataset",
            augmentation_backend="auto",
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            aug_config=None,
        )
        with (
            patch("rfdetr.datasets.coco.Path") as mock_path,
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection", return_value=MagicMock()),
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False),
        ):
            mock_path.return_value.exists.return_value = True
            mock_transforms.return_value = MagicMock()
            build_roboflow_from_coco("train", args, resolution=640)
        assert mock_transforms.call_args.kwargs["gpu_postprocess"] is False

    def test_gpu_backend_no_cuda_raises_runtime_error(self, tmp_path: Path) -> None:
        """Roboflow COCO builder fails fast when 'gpu' is requested without a CUDA device."""
        from unittest.mock import patch

        from rfdetr.datasets.coco import build_roboflow_from_coco

        args = _pipeline_args(tmp_path, augmentation_backend="gpu")
        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=False),
            pytest.raises(RuntimeError, match="CUDA"),
        ):
            build_roboflow_from_coco("train", args, resolution=640)

    def test_gpu_backend_no_kornia_raises_import_error(self, tmp_path: Path) -> None:
        """Roboflow COCO builder fails fast with an install hint when 'gpu' is requested but kornia is missing."""
        from unittest.mock import patch

        from rfdetr.config import AugmentationBackend
        from rfdetr.datasets.coco import build_roboflow_from_coco

        args = _pipeline_args(tmp_path, augmentation_backend="gpu")
        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA),
            pytest.raises(ImportError, match=r"rfdetr\[augment\]"),
        ):
            build_roboflow_from_coco("train", args, resolution=640)

    @pytest.mark.parametrize(
        ("square_resize_div_64", "transform_factory"),
        [
            pytest.param(False, "make_coco_transforms", id="standard_resize"),
            pytest.param(True, "make_coco_transforms_square_div_64", id="square_resize"),
        ],
    )
    def test_keypoint_flip_pairs_forwarded_to_transforms(
        self,
        tmp_path: Path,
        square_resize_div_64: bool,
        transform_factory: str,
    ) -> None:
        """Roboflow keypoint datasets must pass flip pairs to CPU augmentation transforms."""
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.coco import build_roboflow_from_coco

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            augmentation_backend="cpu",
            square_resize_div_64=square_resize_div_64,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[0, 4],
            keypoint_flip_pairs=[0, 1, 2, 3],
            aug_config={},
        )

        with (
            patch(f"rfdetr.datasets.coco.{transform_factory}") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == [0, 1, 2, 3]


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
        tmp_path,
        segmentation_head,
        augmentation_backend,
        resolved_backend,
        expected_gpu_postprocess,
    ):
        """Build Roboflow COCO datasets and assert the GPU postprocess flag passed to transforms."""
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.coco import build_roboflow_from_coco

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
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            aug_config=None,
        )

        with (
            patch("rfdetr.datasets.coco.resolve_backend_for_build", return_value=resolved_backend),
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        call_kwargs = mock_transforms.call_args.kwargs if mock_transforms.call_args else mock_transforms.call_args[1]
        assert call_kwargs["gpu_postprocess"] is expected_gpu_postprocess


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

    @pytest.mark.parametrize(
        ("square_resize_div_64", "transform_factory"),
        [
            pytest.param(False, "make_coco_transforms", id="standard_resize"),
            pytest.param(True, "make_coco_transforms_square_div_64", id="square_resize"),
        ],
    )
    def test_build_roboflow_from_coco_forwards_none_for_detection(
        self,
        tmp_path: Path,
        square_resize_div_64: bool,
        transform_factory: str,
    ) -> None:
        """Roboflow detection datasets must forward ``keypoint_flip_pairs=None``, not ``[]``."""
        from unittest.mock import MagicMock, patch

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            augmentation_backend="cpu",
            square_resize_div_64=square_resize_div_64,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=False,
            num_keypoints_per_class=[],
            keypoint_flip_pairs=[],
            aug_config={"HorizontalFlip": {"p": 0.5}},
        )

        with (
            patch(f"rfdetr.datasets.coco.{transform_factory}") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] is None

    @pytest.mark.parametrize(
        ("square_resize_div_64", "transform_factory"),
        [
            pytest.param(False, "make_coco_transforms", id="standard_resize"),
            pytest.param(True, "make_coco_transforms_square_div_64", id="square_resize"),
        ],
    )
    def test_build_coco_forwards_none_for_detection(
        self,
        tmp_path: Path,
        square_resize_div_64: bool,
        transform_factory: str,
    ) -> None:
        """COCO-format detection datasets must forward ``keypoint_flip_pairs=None``, not ``[]``."""
        from unittest.mock import MagicMock, patch

        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=False)
        args.square_resize_div_64 = square_resize_div_64
        args.aug_config = {"HorizontalFlip": {"p": 0.5}}

        with (
            patch(f"rfdetr.datasets.coco.{transform_factory}") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] is None

    def test_build_coco_forwards_flip_pairs_for_keypoint_mode(self, tmp_path: Path) -> None:
        """COCO keypoint-mode builds must forward user-supplied flip pairs to CPU transforms."""
        from unittest.mock import MagicMock, patch

        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=True)
        args.keypoint_flip_pairs = [0, 1, 2, 3]
        args.aug_config = {"HorizontalFlip": {"p": 0.5}}

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == [0, 1, 2, 3]

    def test_build_coco_forwards_empty_list_for_keypoint_mode_without_flip_pairs(self, tmp_path: Path) -> None:
        """COCO keypoint-mode builds with no configured flip pairs must forward ``[]``, not ``None``."""
        from unittest.mock import MagicMock, patch

        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=True)
        args.keypoint_flip_pairs = []

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == []

    def test_build_roboflow_from_coco_forwards_empty_list_for_keypoint_mode_without_flip_pairs(
        self, tmp_path: Path
    ) -> None:
        """Roboflow keypoint-mode builds with no configured flip pairs must forward ``[]``, not ``None``."""
        from unittest.mock import MagicMock, patch

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            augmentation_backend="cpu",
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[0, 4],
            keypoint_flip_pairs=[],
            aug_config={},
        )

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == []

    def test_build_coco_coerces_explicit_none_flip_pairs_to_empty_list_for_keypoint_mode(self, tmp_path: Path) -> None:
        """A ``KeypointTrainConfig`` default of explicit ``keypoint_flip_pairs=None`` must coerce to ``[]``.

        Scenario: a fresh ``KeypointTrainConfig(dataset_dir=...)`` defaults ``keypoint_flip_pairs`` to ``None``
        (not an omitted attribute). ``getattr(args, "keypoint_flip_pairs", []) or []`` must still coerce that
        explicit ``None`` to ``[]`` under ``include_keypoints=True``, rather than forwarding ``None`` through.
        """
        from unittest.mock import MagicMock, patch

        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=True)
        args.keypoint_flip_pairs = None

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == []

    def test_build_roboflow_from_coco_coerces_explicit_none_flip_pairs_to_empty_list_for_keypoint_mode(
        self, tmp_path: Path
    ) -> None:
        """Roboflow keypoint-mode builds must coerce an explicit ``None`` flip-pairs default to ``[]``.

        Scenario: a fresh ``KeypointTrainConfig(dataset_dir=...)`` defaults ``keypoint_flip_pairs`` to ``None``
        (not an omitted attribute). ``getattr(args, "keypoint_flip_pairs", []) or []`` must still coerce that
        explicit ``None`` to ``[]`` under ``include_keypoints=True``, rather than forwarding ``None`` through.
        """
        from unittest.mock import MagicMock, patch

        args = types.SimpleNamespace(
            dataset_dir=str(tmp_path),
            augmentation_backend="cpu",
            square_resize_div_64=False,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            patch_size=16,
            num_windows=4,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=[0, 4],
            keypoint_flip_pairs=None,
            aug_config={},
        )

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        assert mock_transforms.call_args.kwargs["keypoint_flip_pairs"] == []


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


def _make_coco_builder_args(tmp_path: Path, *, use_grouppose_keypoints: bool) -> types.SimpleNamespace:
    """Return a namespace with all fields consumed by ``build_coco``."""
    return types.SimpleNamespace(
        dataset_dir=None,
        coco_path=str(tmp_path),
        square_resize_div_64=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        patch_size=16,
        num_windows=4,
        # Empty aug_config disables augmentation — these tests verify annotation routing, not aug.
        aug_config={},
        augmentation_backend="cpu",
        use_grouppose_keypoints=use_grouppose_keypoints,
        num_keypoints_per_class=[17] if use_grouppose_keypoints else [],
        keypoint_flip_pairs=[],
    )


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

    def test_keypoint_mode_uses_person_keypoints_annotations(self, tmp_path: Path) -> None:
        """Keypoint mode should switch train annotations to ``person_keypoints_train2017.json``."""
        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=True)

        from unittest.mock import patch

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target)),
            patch("rfdetr.datasets.coco.CocoDetection", return_value=object()) as mock_dataset,
        ):
            build_coco("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        ann_file = Path(mock_dataset.call_args.args[1])
        assert ann_file.parent.name == "annotations"
        assert ann_file.name == "person_keypoints_train2017.json"
        assert kwargs["include_keypoints"] is True
        assert kwargs["remap_category_ids"] is True

    def test_default_mode_uses_instances_annotations_with_raw_coco_ids(self, tmp_path: Path) -> None:
        """Default COCO detection mode should keep raw sparse category IDs for pretrained checkpoints."""
        from unittest.mock import patch

        args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=False)
        with (
            patch("rfdetr.datasets.coco.make_coco_transforms", return_value=lambda image, target: (image, target)),
            patch("rfdetr.datasets.coco.CocoDetection", return_value=object()) as mock_dataset,
        ):
            build_coco("train", args, resolution=640)

        _, kwargs = mock_dataset.call_args
        ann_file = Path(mock_dataset.call_args.args[1])
        assert ann_file.parent.name == "annotations"
        assert ann_file.name == "instances_train2017.json"
        assert kwargs["include_keypoints"] is False
        assert kwargs["remap_category_ids"] is False


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
        "scale_jitter,expected",
        [
            pytest.param(True, True, id="enabled_keeps_crop"),
            pytest.param(False, False, id="disabled_drops_crop"),
        ],
    )
    def test_make_coco_transforms_forwards_scale_jitter(self, scale_jitter, expected):
        """make_coco_transforms passes scale_jitter through to the torchvision-native resize pipeline unchanged."""
        from unittest.mock import patch

        from rfdetr.datasets.coco import make_coco_transforms

        with patch("rfdetr.datasets.coco._build_train_resize_transforms") as mock_build:
            make_coco_transforms("train", 640, scale_jitter=scale_jitter)

        assert mock_build.call_args.kwargs["scale_jitter"] is expected

    @pytest.mark.parametrize(
        "scale_jitter,expected",
        [
            pytest.param(True, True, id="enabled_keeps_crop"),
            pytest.param(False, False, id="disabled_drops_crop"),
        ],
    )
    def test_make_coco_transforms_square_forwards_scale_jitter(self, scale_jitter, expected):
        """make_coco_transforms_square_div_64 passes scale_jitter through to the torchvision-native resize pipeline."""
        from unittest.mock import patch

        from rfdetr.datasets.coco import make_coco_transforms_square_div_64

        with patch("rfdetr.datasets.coco._build_train_resize_transforms") as mock_build:
            make_coco_transforms_square_div_64("train", 640, scale_jitter=scale_jitter)

        assert mock_build.call_args.kwargs["scale_jitter"] is expected

    def test_empty_aug_config_no_longer_affects_crop_branch(self):
        """aug_config={} disables the augmentation stack only — crop branch stays on by default."""
        from unittest.mock import patch

        from rfdetr.datasets.coco import make_coco_transforms

        with patch("rfdetr.datasets.coco._build_train_resize_transforms") as mock_build:
            make_coco_transforms("train", 640, aug_config={})

        assert mock_build.call_args.kwargs["scale_jitter"] is True

    @pytest.mark.parametrize(
        "scale_jitter,expected",
        [
            pytest.param(True, True, id="enabled_keeps_crop"),
            pytest.param(False, False, id="disabled_drops_crop"),
        ],
    )
    def test_non_empty_aug_config_forwards_scale_jitter_to_albumentations(self, scale_jitter, expected):
        """Non-empty aug_config routes to the Albumentations resize config with scale_jitter forwarded."""
        from unittest.mock import MagicMock, patch

        from rfdetr.datasets.coco import make_coco_transforms

        with (
            patch("rfdetr.datasets.coco._build_train_resize_config", return_value=[]) as mock_build,
            patch("rfdetr.datasets.coco.AlbumentationsWrapper.from_config", return_value=MagicMock()),
        ):
            make_coco_transforms("train", 640, aug_config={"HorizontalFlip": {"p": 0.5}}, scale_jitter=scale_jitter)

        assert mock_build.call_args.kwargs["scale_jitter"] is expected


def _make_gradient_image(width: int, height: int) -> Image.Image:
    """Build a deterministic RGB gradient image with real pixel content for interpolation comparisons."""
    import numpy as np

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    grid = np.broadcast_to(x, (height, width))
    channel_b = np.broadcast_to(y[:, None], (height, width))
    array = np.stack([grid, channel_b, (grid // 2 + channel_b // 2)], axis=-1).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class TestPipelineParity:
    """Torchvision and Albumentations eval pipelines produce statistically close resize+normalize output.

    Both backends use different interpolation algorithms (torchvision: BILINEAR + antialias;
    Albumentations: cv2 INTER_LINEAR, no antialias) so exact pixel parity is not expected — see the
    backend-switch UserWarning in ``_build_torchvision_pipeline``. This test asserts statistical
    closeness (shape/dtype match, output distribution within tolerance) instead.
    """

    def test_eval_resize_normalize_output_stats_are_close(self) -> None:
        """_build_torchvision_pipeline and _build_albumentations_pipeline agree on shape, dtype, and pixel stats."""
        from rfdetr.datasets.coco import _build_albumentations_pipeline, _build_torchvision_pipeline

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


class TestCocoDetectionZeroAnnotations:
    """CocoDetection correctly handles images with no annotations."""

    def test_zero_annotation_sample_yields_empty_boxes_and_labels(self, tmp_path: Path) -> None:
        """An image with no annotations yields boxes (0, 4) float32 and labels (0,) int64 tensors."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (100, 100)).save(img_dir / "img1.jpg")
        Image.new("RGB", (100, 100)).save(img_dir / "img2.jpg")
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100},
                        {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100},
                    ],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}
                    ],
                    "categories": [{"id": 1, "name": "cat", "supercategory": "animal"}],
                }
            )
        )
        dataset = CocoDetection(img_dir, ann_file, transforms=None)
        zero_ann_idx = dataset.ids.index(2)
        _, target = dataset[zero_ann_idx]
        assert target["boxes"].shape == torch.Size([0, 4])
        assert target["labels"].shape == torch.Size([0])
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64

    def test_all_parent_hierarchy_falls_back_to_full_list_when_dataset_has_zero_annotations(
        self, tmp_path: Path
    ) -> None:
        """A dataset with hierarchy categories but zero total annotations keeps every category (docstring fallback)."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        Image.new("RGB", (100, 100)).save(img_dir / "img1.jpg")
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
                    "annotations": [],
                    "categories": [
                        {"id": 1, "name": "a", "supercategory": "b"},
                        {"id": 2, "name": "b", "supercategory": "a"},
                    ],
                }
            )
        )
        dataset = CocoDetection(img_dir, ann_file, transforms=None, remap_category_ids=True)
        assert dataset.cat2label == {1: 0, 2: 1}
        _, target = dataset[0]
        assert target["boxes"].shape == torch.Size([0, 4])
        assert target["labels"].shape == torch.Size([0])


class TestCocoDetectionDraftDecode:
    """Draft-decoded JPEG samples retain image/annotation geometry alignment."""

    def test_annotation_scaling_keeps_polygon_axes_and_keypoint_visibility(self) -> None:
        """Non-square drafting must scale polygon axes without changing keypoint visibility."""
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
        """Draft decoding an odd JPEG must use its actual x and y scale factors."""
        original_width, original_height = 1921, 1081
        image_path = tmp_path / "draft.jpg"
        Image.new("RGB", (original_width, original_height)).save(image_path, quality=95)
        annotation_path = tmp_path / "annotations.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": image_path.name, "width": original_width, "height": original_height}
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [100.0, 200.0, 300.0, 400.0],
                            "area": 120000.0,
                            "iscrowd": 0,
                            "keypoints": [100.0, 200.0, 2.0],
                        }
                    ],
                    "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
                }
            ),
            encoding="utf-8",
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
        torch.testing.assert_close(
            target["boxes"],
            torch.tensor([[100.0 * x_scale, 200.0 * y_scale, 400.0 * x_scale, 600.0 * y_scale]]),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            target["keypoints"],
            torch.tensor([[[100.0 * x_scale, 200.0 * y_scale, 2.0]]]),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(target["area"], torch.tensor([120000.0 * x_scale * y_scale]), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        ("image_set", "include_masks"),
        [
            pytest.param("val", False, id="validation"),
            pytest.param("train", True, id="mask-training"),
        ],
    )
    def test_draft_size_skips_full_resolution_contracts(self, image_set: str, include_masks: bool) -> None:
        """Validation and mask datasets must disable draft decoding even with scale jitter."""
        assert draft_size_for_transforms(image_set, 512, scale_jitter=True, include_masks=include_masks) is None

    def test_draft_size_uses_scale_jitter_floor(self) -> None:
        """Scale jitter must preserve the crop branch's 600-pixel source floor."""
        assert draft_size_for_transforms("train", 512, scale_jitter=True) == 600

    @pytest.mark.parametrize(
        ("builder_name", "scale_jitter", "expected_draft_size"),
        [
            pytest.param("coco", False, 512, id="coco-jitter-disabled"),
            pytest.param("coco", True, 600, id="coco-jitter-enabled"),
            pytest.param("roboflow", False, 512, id="roboflow-jitter-disabled"),
            pytest.param("roboflow", True, 600, id="roboflow-jitter-enabled"),
        ],
    )
    def test_builders_forward_scale_jitter_to_draft_size(
        self,
        tmp_path: Path,
        builder_name: str,
        scale_jitter: bool,
        expected_draft_size: int,
    ) -> None:
        """Both COCO builders must give the draft decoder the configured jitter floor."""
        from unittest.mock import MagicMock, patch

        if builder_name == "coco":
            args = _make_coco_builder_args(tmp_path, use_grouppose_keypoints=False)
            builder = build_coco
        else:
            args = types.SimpleNamespace(
                dataset_dir=str(tmp_path),
                augmentation_backend="cpu",
                square_resize_div_64=False,
                segmentation_head=False,
                multi_scale=False,
                expanded_scales=False,
                do_random_resize_via_padding=False,
                patch_size=16,
                num_windows=4,
                use_grouppose_keypoints=False,
                num_keypoints_per_class=[],
                aug_config={},
            )
            builder = build_roboflow_from_coco
        args.scale_jitter = scale_jitter

        with (
            patch("rfdetr.datasets.coco.make_coco_transforms", return_value=MagicMock()),
            patch("rfdetr.datasets.coco.CocoDetection", return_value=MagicMock()) as mock_dataset,
        ):
            builder("train", args, resolution=512)

        assert mock_dataset.call_args.kwargs["draft_size"] == expected_draft_size


_PHANTOM_ROOT_CATEGORIES = [
    {"id": 0, "name": "eggmasses", "supercategory": "none"},
    {"id": 1, "name": "empty_pot", "supercategory": "eggmasses"},
    {"id": 2, "name": "stake", "supercategory": "eggmasses"},
    {"id": 3, "name": "tree", "supercategory": "eggmasses"},
    {"id": 4, "name": "trunk", "supercategory": "eggmasses"},
]


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


class TestFilterParentCategories:
    """Unit tests for the grouping-category filter shared by class names, class count and label remapping."""

    def test_unannotated_root_is_dropped(self) -> None:
        """A category named as another category's supercategory and carrying no annotation is removed."""
        result = filter_parent_categories(_PHANTOM_ROOT_CATEGORIES, {1, 2, 3, 4})
        assert [category["name"] for category in result] == ["empty_pot", "stake", "tree", "trunk"]

    def test_annotated_parent_is_kept(self) -> None:
        """A parent category that owns annotations stays, so its annotations keep a label slot."""
        result = filter_parent_categories(_PHANTOM_ROOT_CATEGORIES, {0, 1})
        assert [category["id"] for category in result] == [0, 1, 2, 3, 4]

    def test_flat_dataset_is_unchanged(self) -> None:
        """Datasets whose supercategories are all placeholders are returned untouched."""
        categories = [
            {"id": 1, "name": "dog", "supercategory": "none"},
            {"id": 2, "name": "cat", "supercategory": "none"},
        ]
        assert filter_parent_categories(categories, set()) == categories

    def test_result_is_sorted_by_id(self) -> None:
        """Output order follows category id, matching the contiguous label indices derived from it."""
        categories = [
            {"id": 2, "name": "cat", "supercategory": "none"},
            {"id": 1, "name": "dog", "supercategory": "none"},
        ]
        assert [category["id"] for category in filter_parent_categories(categories, set())] == [1, 2]

    def test_category_without_a_name_reports_the_offending_id(self) -> None:
        """A category missing the required ``name`` field names itself in the error instead of raising a bare id."""
        categories = [
            {"id": 1, "name": "vehicle", "supercategory": "none"},
            {"id": 2, "supercategory": "vehicle"},
        ]
        with pytest.raises(KeyError, match="missing the required 'name' field"):
            filter_parent_categories(categories, {1})

    def test_string_category_id_matches_the_annotated_ids(self) -> None:
        """Exports shipping string ids still match the int-keyed annotated set, so annotated parents survive."""
        categories = [
            {"id": "1", "name": "vehicle", "supercategory": "none"},
            {"id": "2", "name": "car", "supercategory": "vehicle"},
        ]
        assert [category["id"] for category in filter_parent_categories(categories, {1, 2})] == ["1", "2"]

    def test_mixed_and_missing_ids_sort_after_numeric_categories(self) -> None:
        """Numeric IDs sort consistently even when malformed entries are present."""
        categories = [
            {"name": "missing", "supercategory": "none"},
            {"id": "2", "name": "two", "supercategory": "none"},
            {"id": 1, "name": "one", "supercategory": "none"},
        ]
        result = filter_parent_categories(categories, set())
        assert [category["name"] for category in result] == ["one", "two", "missing"]

    def test_categories_sharing_a_name_are_judged_by_their_own_id(self) -> None:
        """An annotated leaf keeps its slot even when an unannotated grouping node reuses its name."""
        categories = [
            {"id": 1, "name": "tree", "supercategory": "none"},
            {"id": 2, "name": "trunk", "supercategory": "tree"},
            {"id": 3, "name": "tree", "supercategory": "tree"},
        ]
        assert [category["id"] for category in filter_parent_categories(categories, {2, 3})] == [2, 3]

    def test_self_referential_supercategory_is_not_a_parent(self) -> None:
        """A category naming itself as its own supercategory is not its own parent, so it keeps its label slot."""
        categories = [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "vehicle", "supercategory": "none"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
        ]
        assert [category["id"] for category in filter_parent_categories(categories, {3})] == [1, 3]

    def test_self_referential_name_stays_a_parent_for_other_categories(self) -> None:
        """A self-referencing name another category genuinely groups under is still dropped when unannotated."""
        categories = [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "rider", "supercategory": "person"},
        ]
        assert [category["id"] for category in filter_parent_categories(categories, {2})] == [2]

    def test_all_categories_parents_falls_back_to_full_list(self) -> None:
        """Pathological input where every category is a parent keeps the full list instead of returning nothing."""
        categories = [
            {"id": 1, "name": "a", "supercategory": "b"},
            {"id": 2, "name": "b", "supercategory": "a"},
        ]
        assert filter_parent_categories(categories, set()) == categories

    def test_all_categories_parents_with_partial_annotation_keeps_only_annotated(self) -> None:
        """When some (not all, not none) categories in an all-parent cycle are annotated, only those survive."""
        categories = [
            {"id": 1, "name": "a", "supercategory": "c"},
            {"id": 2, "name": "b", "supercategory": "a"},
            {"id": 3, "name": "c", "supercategory": "b"},
        ]
        result = filter_parent_categories(categories, {2})
        assert [category["id"] for category in result] == [2]

    def test_two_independent_grouping_roots_in_one_call(self) -> None:
        """Two disjoint hierarchies filtered together: the annotated root stays, the unannotated sibling root drops."""
        categories = [
            {"id": 0, "name": "rootA", "supercategory": "none"},
            {"id": 1, "name": "a1", "supercategory": "rootA"},
            {"id": 2, "name": "a2", "supercategory": "rootA"},
            {"id": 5, "name": "rootB", "supercategory": "none"},
            {"id": 6, "name": "b1", "supercategory": "rootB"},
            {"id": 7, "name": "b2", "supercategory": "rootB"},
        ]
        result = filter_parent_categories(categories, {1, 2, 5, 6, 7})
        assert [category["id"] for category in result] == [1, 2, 5, 6, 7]


class TestAnnotatedCategoryIds:
    """Unit tests for collecting the annotated category ids of a parsed COCO file."""

    def test_ids_collected_from_annotations(self) -> None:
        """Every distinct ``category_id`` referenced by an annotation is reported once."""
        coco_data = {"annotations": [{"category_id": 3}, {"category_id": 3}, {"category_id": 5}]}
        assert annotated_category_ids(coco_data) == {3, 5}

    def test_missing_annotations_key_yields_empty_set(self) -> None:
        """A category-only annotation file reports no annotated categories."""
        assert annotated_category_ids({"categories": _PHANTOM_ROOT_CATEGORIES}) == set()


class TestPhantomRootConsistency:
    """Roboflow COCO exports prepend an unannotated root category; count, remap and names must agree without it."""

    def test_cat2label_skips_unannotated_root(self, tmp_path: Path) -> None:
        """The synthetic root consumes no label slot, so real classes start at index 0."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        assert dataset.cat2label == {1: 0, 2: 1, 3: 2, 4: 3}

    def test_label2cat_is_exposed_for_the_evaluator(self, tmp_path: Path) -> None:
        """The reverse mapping reaches the COCO API object so predictions convert back to original ids."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        assert dataset.coco.label2cat == {0: 1, 1: 2, 2: 3, 3: 4}

    def test_labels_use_filtered_indices(self, tmp_path: Path) -> None:
        """Targets carry the filtered label indices rather than the root-shifted ones."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        _, target = dataset[0]
        assert target["labels"].tolist() == [0, 3]

    def test_detected_num_classes_matches_cat2label(self, tmp_path: Path) -> None:
        """The auto-detected head size equals the number of label slots the remapping actually uses."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        detected = RFDETR._detect_num_classes_for_training(str(tmp_path))
        assert detected == len(set(dataset.cat2label.values()))

    def test_class_names_and_class_count_share_one_basis(self, tmp_path: Path) -> None:
        """The detected head size and the loaded class names come from one filtered category list, so they agree."""
        _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        assert RFDETR._detect_num_classes_for_training(str(tmp_path)) == len(RFDETR._load_classes(str(tmp_path)))

    def test_load_classes_matches_cat2label_order(self, tmp_path: Path) -> None:
        """Class names line up positionally with the label indices assigned by the remapping."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [1, 4])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        names = RFDETR._load_classes(str(tmp_path))
        assert names == [dataset.coco.cats[dataset.label2cat[label]]["name"] for label in range(len(names))]


class TestCrossSplitLabelSpace:
    """Val/test splits must reuse the train label space whatever their own annotation coverage is."""

    def test_val_split_reuses_the_train_label_mapping(self, tmp_path: Path) -> None:
        """A grouping category annotated in train only keeps its train label slot in val instead of shifting it."""
        _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        _write_roboflow_hierarchy_split(tmp_path / "valid", [1])
        args = _pipeline_args(tmp_path)

        train_dataset = build_roboflow_from_coco("train", args, resolution=64)
        val_dataset = build_roboflow_from_coco("val", args, resolution=64)

        assert val_dataset.cat2label == train_dataset.cat2label

    def test_val_targets_use_train_label_indices(self, tmp_path: Path) -> None:
        """Val targets carry the label index training assigned, not the one val's own coverage would produce."""
        _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        _write_roboflow_hierarchy_split(tmp_path / "valid", [1])
        args = _pipeline_args(tmp_path)

        val_dataset = build_roboflow_from_coco("val", args, resolution=64)
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

        with pytest.raises(ValueError, match="remap_category_ids"):
            CocoDetection(tmp_path / "valid", ann_file, transforms=None, cat2label={1: 0})

    def test_annotated_parent_keeps_its_label_slot(self, tmp_path: Path) -> None:
        """A parent category carrying annotations is not dropped, so converting its annotations does not raise."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        _, target = dataset[0]
        assert target["labels"].tolist() == [0, 1]

    def test_cat2label_keeps_annotated_root(self, tmp_path: Path) -> None:
        """When the root itself carries annotations, every category keeps its identity label slot."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        assert dataset.cat2label == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    def test_label2cat_is_exposed_for_the_evaluator_when_root_is_annotated(self, tmp_path: Path) -> None:
        """The reverse mapping still reaches the COCO API object when the root is kept, not just when it is dropped."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        assert dataset.coco.label2cat == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    def test_detected_num_classes_matches_cat2label_when_root_is_annotated(self, tmp_path: Path) -> None:
        """The auto-detected head size still equals the number of label slots when the root keeps its slot."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        detected = RFDETR._detect_num_classes_for_training(str(tmp_path))
        assert detected == len(set(dataset.cat2label.values()))

    def test_load_classes_matches_cat2label_order_when_root_is_annotated(self, tmp_path: Path) -> None:
        """Class names still line up positionally with label indices when the root keeps its slot."""
        ann_file = _write_roboflow_hierarchy_split(tmp_path / "train", [0, 1])
        dataset = CocoDetection(tmp_path / "train", ann_file, transforms=None, remap_category_ids=True)
        names = RFDETR._load_classes(str(tmp_path))
        assert names == [dataset.coco.cats[dataset.label2cat[label]]["name"] for label in range(len(names))]
