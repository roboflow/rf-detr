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
from rfdetr.datasets.coco import ConvertCoco, build_coco, build_roboflow_from_coco
from rfdetr.detr import RFDETR

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
    """Write a minimal valid COCO annotation file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"images": [], "annotations": [], "categories": categories}
    path.write_text(json.dumps(data))


def _write_roboflow_keypoint_coco(path: Path, *, category_id: int = 0) -> None:
    """Write a minimal Roboflow-style COCO keypoint split."""
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
        import sys
        from unittest.mock import MagicMock, patch

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.dict(sys.modules, {"kornia": MagicMock(), "kornia.augmentation": MagicMock()}),
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
        import sys
        from unittest.mock import MagicMock, patch

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch.dict(sys.modules, {"kornia": MagicMock(), "kornia.augmentation": MagicMock()}),
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

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _mock_import(name, *args, **kwargs):
            if name == "kornia" or name.startswith("kornia."):
                raise ImportError("No module named 'kornia'")
            return original_import(name, *args, **kwargs)

        with (
            patch("rfdetr.datasets.kornia_transforms._has_cuda_device", return_value=True),
            patch("builtins.__import__", side_effect=_mock_import),
            pytest.raises(ImportError, match="rfdetr\\[kornia\\]"),
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
            pytest.param(False, "cpu", "cpu", False, id="cpu_backend_no_seg"),
            pytest.param(True, "cpu", "cpu", False, id="cpu_backend_with_seg"),
            pytest.param(False, "gpu", "gpu", True, id="gpu_backend_no_seg"),
            pytest.param(True, "gpu", "gpu", True, id="gpu_backend_with_seg"),
            pytest.param(True, "auto", "gpu", True, id="auto_resolved_gpu_with_seg"),
            pytest.param(True, "auto", "cpu", False, id="auto_resolved_cpu_with_seg"),
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
            patch("rfdetr.datasets.coco._resolve_runtime_augmentation_backend", return_value=resolved_backend),
            patch("rfdetr.datasets.coco.make_coco_transforms") as mock_transforms,
            patch("rfdetr.datasets.coco.CocoDetection") as mock_coco,
        ):
            mock_transforms.return_value = MagicMock()
            mock_coco.return_value = MagicMock()

            build_roboflow_from_coco("train", args, resolution=640)

        call_kwargs = mock_transforms.call_args.kwargs if mock_transforms.call_args else mock_transforms.call_args[1]
        assert call_kwargs["gpu_postprocess"] is expected_gpu_postprocess


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
