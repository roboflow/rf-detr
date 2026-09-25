# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCOEvalCallback scores crowd (``iscrowd=1``) ground truth the way pycocotools does.

``ConvertCoco`` drops crowd annotations from every dataset target so training never matches them. Evaluation must still
see them: under the COCO protocol a detection that lands on a crowd region is ignored, not counted as a false positive.
These tests drive the callback hooks with a real :class:`~rfdetr.datasets.coco.CocoDetection` built from a small
annotation file, so the crowd regions reach the callback exactly as they do from a COCO-format dataset on disk.
"""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch.utils.data import DistributedSampler

from rfdetr.config import CocoEvalBackend
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms_square_div_64
from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from rfdetr.utilities.box_ops import box_xyxy_to_cxcywh

# Power-of-two image sides keep every normalised box coordinate exact in float32, so boxes survive the
# normalise/denormalise round trip bit for bit and the metric comparisons below can be exact.
_WIDTH = 256
_HEIGHT = 128
_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "car", "supercategory": "vehicle"},
]
# Epoch-start, batch-end and epoch-end hooks for each evaluation split.
_HOOKS = {
    "val": ("on_validation_epoch_start", "on_validation_batch_end", "on_validation_epoch_end"),
    "test": ("on_test_epoch_start", "on_test_batch_end", "on_test_epoch_end"),
}
_SPLITS = [pytest.param("val", id="val"), pytest.param("test", id="test")]


def _annotation(
    annotation_id: int,
    image_id: int,
    bbox: list[float],
    category_id: int = 1,
    iscrowd: int = 0,
    segmentation: Any = None,
) -> dict[str, Any]:
    """Return one COCO annotation; without ``segmentation`` its mask is the ``[x, y, w, h]`` box as a polygon.

    Examples:
        >>> _annotation(7, 1, [8, 16, 32, 64], iscrowd=1)["segmentation"]
        [[8, 16, 40, 16, 40, 80, 8, 80]]
    """
    x, y, w, h = bbox
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": w * h,
        "iscrowd": iscrowd,
        "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]] if segmentation is None else segmentation,
    }


def _uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Return a binary ``mask`` as an uncompressed COCO RLE, the format COCO's own crowd annotations use.

    Examples:
        >>> _uncompressed_rle(np.array([[0, 1], [0, 1]], dtype=np.uint8))
        {'size': [2, 2], 'counts': [2, 2]}
    """
    counts: list[int] = []
    current, run = 0, 0
    for value in mask.flatten(order="F").tolist():
        if value != current:
            counts.append(run)
            current, run = value, 0
        run += 1
    counts.append(run)
    return {"size": list(mask.shape), "counts": counts}


def _coco_dataset(
    root: Path,
    annotations: list[dict[str, Any]],
    images: int = 1,
    categories: list[dict[str, Any]] | None = None,
    transforms: Any = None,
    **dataset_kwargs: Any,
) -> CocoDetection:
    """Write ``images`` blank images and a COCO file holding ``annotations``, and load them as a ``CocoDetection``.

    Image ids run from 1. ``transforms=None`` (the default) leaves targets as ``ConvertCoco`` returns them. The load is
    silenced because pycocotools prints its indexing progress.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dataset = _coco_dataset(Path(tmp), [_annotation(1, 2, [8, 8, 16, 16], iscrowd=1)], images=2)
        ...     len(dataset), dataset.coco.anns[1]["iscrowd"]
        (2, 1)
    """
    (root / "images").mkdir(parents=True, exist_ok=True)
    image_entries = []
    for image_id in range(1, images + 1):
        file_name = f"{image_id}.png"
        Image.new("RGB", (_WIDTH, _HEIGHT)).save(root / "images" / file_name)
        image_entries.append({"id": image_id, "file_name": file_name, "width": _WIDTH, "height": _HEIGHT})
    annotation_file = root / "annotations.json"
    annotation_file.write_text(
        json.dumps({"images": image_entries, "annotations": annotations, "categories": categories or _CATEGORIES})
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return CocoDetection(root / "images", annotation_file, transforms=transforms, **dataset_kwargs)


def _eval_targets(dataset: CocoDetection) -> list[dict[str, torch.Tensor]]:
    """Return every sample's target as the evaluation loader hands it to the callback: boxes as normalised CxCyWH.

    With ``transforms=None`` the dataset returns ``ConvertCoco``'s absolute xyxy boxes; the conversion below is the
    evaluation pipeline's ``Normalize`` step.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dataset = _coco_dataset(Path(tmp), [_annotation(1, 1, [64, 32, 128, 64])])
        ...     _eval_targets(dataset)[0]["boxes"].tolist()
        [[0.5, 0.5, 0.5, 0.5]]
    """
    targets = []
    for index in range(len(dataset)):
        _, target = dataset[index]
        height, width = target["orig_size"].tolist()
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        target["boxes"] = box_xyxy_to_cxcywh(target["boxes"]) / scale
        targets.append(target)
    return targets


def _prediction(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    mask_grid: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Return one image's ``PostProcess``-style prediction, boxes in original-image xyxy pixels.

    With ``mask_grid`` each box is also a filled ``[K, 1, H, W]`` mask: the box scaled from the ``_WIDTH`` x
    ``_HEIGHT`` image onto that grid.

    Examples:
        >>> prediction = _prediction([[0, 0, 128, 64]], [0.9], [1], mask_grid=(64, 128))
        >>> tuple(prediction["masks"].shape), int(prediction["masks"].sum())
        ((1, 1, 64, 128), 2048)
    """
    prediction = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    if mask_grid is not None:
        grid_height, grid_width = mask_grid
        masks = torch.zeros(len(boxes), 1, grid_height, grid_width, dtype=torch.bool)
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            rows = slice(round(y0 * grid_height / _HEIGHT), round(y1 * grid_height / _HEIGHT))
            cols = slice(round(x0 * grid_width / _WIDTH), round(x1 * grid_width / _WIDTH))
            masks[index, 0, rows, cols] = True
        prediction["masks"] = masks
    return prediction


def _module(compute_train_metrics: bool = False) -> MagicMock:
    """Return a mock detection LightningModule on CPU.

    Examples:
        >>> _module(compute_train_metrics=True).train_config.compute_train_metrics
        True
    """
    module = MagicMock(name="pl_module")
    module.device = "cpu"
    module.model_config = SimpleNamespace(use_grouppose_keypoints=False)
    module.train_config = SimpleNamespace(compute_train_metrics=compute_train_metrics)
    return module


def _trainer(dataset: Any) -> MagicMock:
    """Return a mock trainer whose datamodule serves ``dataset`` for every split, with metric tables silenced.

    Examples:
        >>> _trainer("split").datamodule._dataset_test
        'split'
    """
    trainer = MagicMock(name="trainer")
    trainer.datamodule = SimpleNamespace(_dataset_train=dataset, _dataset_val=dataset, _dataset_test=dataset)
    trainer.callbacks = []
    trainer.callback_metrics = {}
    trainer.is_global_zero = False  # metric tables are printed by rank zero only
    return trainer


def _metric_targets(
    callback: COCOEvalCallback,
    trainer: MagicMock,
    split: str,
    results: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Run one evaluation batch through ``callback`` and return the ground truth its mAP metric received.

    Examples:
        >>> callback = COCOEvalCallback()
        >>> targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]), "labels": torch.tensor([1]),
        ...             "orig_size": torch.tensor([_HEIGHT, _WIDTH])}]
        >>> _metric_targets(callback, _trainer(None), "val", [_prediction([], [], [])], targets)[0]["boxes"].tolist()
        [[64.0, 32.0, 192.0, 96.0]]
    """
    epoch_start, batch_end, _ = _HOOKS[split]
    module = _module()
    callback.setup(trainer, module, stage="fit" if split == "val" else "test")
    callback.map_metric = MagicMock(name="map_metric")
    getattr(callback, epoch_start)(trainer, module)
    getattr(callback, batch_end)(trainer, module, {"results": results, "targets": targets}, None, 0)
    return callback.map_metric.update.call_args.args[1]


def _evaluate(
    callback: COCOEvalCallback,
    trainer: MagicMock,
    split: str,
    results: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Run a one-batch evaluation epoch with ``callback``'s real metrics and return the ``{split}/*`` values logged.

    Examples:
        >>> callback = COCOEvalCallback()
        >>> targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]), "labels": torch.tensor([1]),
        ...             "orig_size": torch.tensor([_HEIGHT, _WIDTH])}]
        >>> prediction = _prediction([[64, 32, 192, 96]], [0.9], [1])
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     metrics = _evaluate(callback, _trainer(None), "val", [prediction], targets)
        >>> metrics["val/mAP_50_95"]
        1.0
    """
    epoch_start, batch_end, epoch_end = _HOOKS[split]
    module = _module()
    callback.setup(trainer, module, stage="fit" if split == "val" else "test")
    getattr(callback, epoch_start)(trainer, module)
    getattr(callback, batch_end)(trainer, module, {"results": results, "targets": targets}, None, 0)
    getattr(callback, epoch_end)(trainer, module)
    return {key: float(value) for key, value in trainer.callback_metrics.items() if key.startswith(f"{split}/")}


# One person, and a crowd of people to the right of it (COCO xywh boxes).
_PERSON = [16, 16, 32, 32]
_PERSON_CROWD = [128, 16, 96, 96]
_PERSON_AND_CROWD = [_annotation(1, 1, _PERSON), _annotation(2, 1, _PERSON_CROWD, iscrowd=1)]
_NO_PREDICTION = [_prediction([], [], [])]


@pytest.mark.parametrize("split", _SPLITS)
class TestCrowdRegionsReachEvaluation:
    """Validation and test metrics see the crowd regions ``ConvertCoco`` removed from the dataset targets."""

    def test_detection_on_crowd_region_is_ignored_not_a_false_positive(self, tmp_path: Path, split: str) -> None:
        """A confident detection inside a crowd region must not cost AP, as under pycocotools.

        The only real person is found, so COCO AP is 1.0. Counting the higher-scored in-crowd detection as a false
        positive instead halves precision at full recall.
        """
        dataset = _coco_dataset(tmp_path, _PERSON_AND_CROWD)
        results = [_prediction([[16, 16, 48, 48], [144, 32, 176, 64]], [0.8, 0.9], [1, 1])]

        metrics = _evaluate(COCOEvalCallback(), _trainer(dataset), split, results, _eval_targets(dataset))

        assert metrics[f"{split}/mAP_50_95"] == pytest.approx(1.0), "in-crowd detection scored as a false positive"
        assert metrics[f"{split}/mAP_50"] == pytest.approx(1.0)

    def test_crowd_region_is_appended_as_a_crowd_row(self, tmp_path: Path, split: str) -> None:
        """The metric ground truth carries the crowd box from the annotation file, in original pixels, flagged crowd."""
        dataset = _coco_dataset(tmp_path, _PERSON_AND_CROWD)

        targets = _metric_targets(COCOEvalCallback(), _trainer(dataset), split, _NO_PREDICTION, _eval_targets(dataset))

        assert targets[0]["boxes"].tolist() == [[16.0, 16.0, 48.0, 48.0], [128.0, 16.0, 224.0, 112.0]]
        assert targets[0]["labels"].tolist() == [1, 1]
        assert targets[0]["iscrowd"].tolist() == [0, 1], "the crowd region must be flagged, not scored as a person"

    def test_crowd_category_follows_the_dataset_label_space(self, tmp_path: Path, split: str) -> None:
        """Remapped datasets relabel crowd regions like any other annotation; categories without a label are skipped.

        This is the Roboflow COCO layout: a grouping category, and validation labels taken from the training split's
        mapping. A crowd of a category outside that mapping can never match a prediction's label, so it is dropped.
        """
        categories = [
            {"id": 0, "name": "animals", "supercategory": "none"},
            {"id": 1, "name": "cat", "supercategory": "animals"},
            {"id": 2, "name": "dog", "supercategory": "animals"},
            {"id": 3, "name": "bird", "supercategory": "animals"},
        ]
        annotations = [
            _annotation(1, 1, _PERSON, category_id=1),
            _annotation(2, 1, _PERSON_CROWD, category_id=2, iscrowd=1),
            _annotation(3, 1, [0, 64, 32, 32], category_id=3, iscrowd=1),
        ]
        dataset = _coco_dataset(
            tmp_path, annotations, categories=categories, remap_category_ids=True, cat2label={1: 0, 2: 1}
        )

        targets = _metric_targets(COCOEvalCallback(), _trainer(dataset), split, _NO_PREDICTION, _eval_targets(dataset))

        assert targets[0]["labels"].tolist() == [0, 1], "dog crowd takes the dog label; the unmapped bird is skipped"
        assert targets[0]["iscrowd"].tolist() == [0, 1]

    @pytest.mark.parametrize(
        "rle",
        [False, True]
    )
    def test_crowd_mask_is_decoded_onto_the_prediction_grid(self, tmp_path: Path, split: str, rle: bool) -> None:
        """Segmentation ground truth gains the crowd mask, decoded at full size and resized like the other GT masks."""
        crowd_mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        crowd_mask[16:112, 128:224] = 1
        crowd = _annotation(2, 1, _PERSON_CROWD, iscrowd=1, segmentation=_uncompressed_rle(crowd_mask) if rle else None)
        dataset = _coco_dataset(tmp_path, [_annotation(1, 1, _PERSON), crowd], include_masks=True)
        grid = (_HEIGHT // 2, _WIDTH // 2)
        results = [_prediction([[16, 16, 48, 48]], [0.9], [1], mask_grid=grid)]

        callback = COCOEvalCallback(segmentation=True)
        targets = _metric_targets(callback, _trainer(dataset), split, results, _eval_targets(dataset))

        full_size = torch.as_tensor(dataset.coco.annToMask(crowd))[None, None].float()
        expected = F.interpolate(full_size, size=grid, mode="nearest")[0, 0].bool()
        assert targets[0]["masks"].shape == (2, *grid)
        assert torch.equal(targets[0]["masks"][1], expected)
        assert targets[0]["iscrowd"].tolist() == [0, 1]

    def test_padded_duplicate_adds_no_crowd_rows(self, tmp_path: Path, split: str) -> None:
        """DistributedSampler padding is dropped before the crowd lookup, so its crowd regions are not scored twice.

        Rank 1 of 2 over three images receives image 2 and then image 1 again as padding.
        """
        annotations = []
        for image_id in (1, 2, 3):
            annotations.append(_annotation(2 * image_id - 1, image_id, _PERSON))
            annotations.append(_annotation(2 * image_id, image_id, [100 + 8 * image_id, 16, 64, 64], iscrowd=1))
        dataset = _coco_dataset(tmp_path, annotations, images=3)
        trainer = _trainer(dataset)
        loader = MagicMock(name="loader")
        loader.sampler = DistributedSampler(list(range(3)), num_replicas=2, rank=1, shuffle=False)
        setattr(trainer, "val_dataloaders" if split == "val" else "test_dataloaders", loader)
        targets = _eval_targets(dataset)

        scored = _metric_targets(COCOEvalCallback(), trainer, split, _NO_PREDICTION * 2, [targets[1], targets[0]])

        assert len(scored) == 1
        assert scored[0]["iscrowd"].tolist() == [0, 1], "the one real image must carry exactly its own crowd region"
        assert scored[0]["boxes"][1].tolist() == [116.0, 16.0, 180.0, 80.0], "image 2 must get its own crowd region"


class TestCrowdRegionsUnderTheEvalTransform:
    """Crowd rows land where the dataset's own ground truth lands after the real validation transform."""

    def test_crowd_row_matches_its_non_crowd_twin(self, tmp_path: Path) -> None:
        """A compressed-RLE crowd and a polygon non-crowd twin covering the same box give the same box and mask.

        The twin goes through ``ConvertCoco``, the square validation resize and ``Normalize``; the crowd is read back
        from the annotation file. Both must end up in the same frame, box and mask alike, or the crowd would ignore
        detections somewhere other than where it is. Compressed (string ``counts``) RLE is what most exporters write.
        """
        pycocotools_mask = pytest.importorskip("pycocotools.mask")
        crowd_mask = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
        crowd_mask[16:112, 128:224] = 1
        rle = pycocotools_mask.encode(np.asfortranarray(crowd_mask))
        rle["counts"] = rle["counts"].decode()
        annotations = [
            _annotation(1, 1, _PERSON_CROWD),
            _annotation(2, 1, _PERSON_CROWD, iscrowd=1, segmentation=rle),
        ]
        transforms = make_coco_transforms_square_div_64("val", 384)
        dataset = _coco_dataset(tmp_path, annotations, transforms=transforms, include_masks=True)
        image, target = dataset[0]
        grid = (image.shape[-2] // 4, image.shape[-1] // 4)
        prediction = _prediction([], [], [])
        prediction["masks"] = torch.zeros(0, 1, *grid, dtype=torch.bool)

        scored = _metric_targets(COCOEvalCallback(segmentation=True), _trainer(dataset), "val", [prediction], [target])

        boxes, masks = scored[0]["boxes"], scored[0]["masks"]
        assert scored[0]["iscrowd"].tolist() == [0, 1]
        torch.testing.assert_close(boxes[1], boxes[0], atol=1e-3, rtol=0, msg="crowd box left the twin's frame")
        overlap = (masks[0] & masks[1]).sum() / (masks[0] | masks[1]).sum()
        assert masks.shape == (2, *grid)
        assert float(overlap) > 0.99, f"crowd mask IoU with its twin is {float(overlap):.4f}"


class TestCrowdLookupIsANoOpWithoutCocoAnnotations:
    """Datasets or targets that cannot name a crowd region leave the metric ground truth untouched."""

    @pytest.mark.parametrize(
        ("datamodule_dataset", "keep_image_id"),
        [
            pytest.param("none", True, id="no-datamodule-dataset"),
            pytest.param("no-coco-api", True, id="dataset-without-coco-api"),
            pytest.param("coco", False, id="target-without-image-id"),
        ],
    )
    def test_targets_keep_only_their_own_rows(
        self, tmp_path: Path, datamodule_dataset: str, keep_image_id: bool
    ) -> None:
        """No COCO API (custom datasets, webdataset streams) or no ``image_id`` on the target means no crowd rows."""
        dataset = _coco_dataset(tmp_path, _PERSON_AND_CROWD)
        served = {"none": None, "no-coco-api": [dataset[0]], "coco": dataset}[datamodule_dataset]
        targets = _eval_targets(dataset)
        if not keep_image_id:
            del targets[0]["image_id"]

        scored = _metric_targets(COCOEvalCallback(), _trainer(served), "val", _NO_PREDICTION, targets)

        assert scored[0]["labels"].tolist() == [1]
        assert scored[0]["iscrowd"].tolist() == [0]

    def test_train_split_metrics_do_not_place_crowd_regions(self, tmp_path: Path) -> None:
        """Train-split metrics run on augmented images, where annotation-file crowd geometry no longer lines up."""
        dataset = _coco_dataset(tmp_path, _PERSON_AND_CROWD)
        trainer = _trainer(dataset)
        module = _module(compute_train_metrics=True)
        callback = COCOEvalCallback()
        callback.setup(trainer, module, stage="fit")
        callback.map_metric_train = MagicMock(name="map_metric_train")

        outputs = {"results": _NO_PREDICTION, "targets": _eval_targets(dataset)}
        callback.on_train_batch_end(trainer, module, outputs, None, 0)

        assert callback.map_metric_train.update.call_args.args[1][0]["iscrowd"].tolist() == [0]


class TestEmaTrackSeesCrowdRegions:
    """The independent EMA forward (``eval_base_model=True``) is scored against the same crowd regions."""

    def test_segmentation_ema_targets_carry_crowd_rows_on_the_ema_grid(self, tmp_path: Path) -> None:
        """Segmentation converts targets a second time for the EMA grid; that conversion must add the crowd too."""
        dataset = _coco_dataset(tmp_path, _PERSON_AND_CROWD, include_masks=True)
        ema_callback = MagicMock(name="ema_callback")
        ema_callback.get_ema_model_state_dict = MagicMock(name="get_ema_model_state_dict")
        ema_callback._average_model = SimpleNamespace(module=SimpleNamespace(model=MagicMock(return_value={})))
        trainer = _trainer(dataset)
        trainer.callbacks = [ema_callback]
        module = _module()
        module.postprocess.return_value = [_prediction([[16, 16, 48, 48]], [0.9], [1], mask_grid=(32, 64))]
        callback = COCOEvalCallback(segmentation=True, eval_base_model=True)
        callback.setup(trainer, module, stage="fit")
        callback.map_metric = MagicMock(name="map_metric")
        callback.map_metric_ema = MagicMock(name="map_metric_ema")
        outputs = {
            "results": [_prediction([[16, 16, 48, 48]], [0.9], [1], mask_grid=(64, 128))],
            "targets": _eval_targets(dataset),
        }

        callback.on_validation_batch_end(trainer, module, outputs, (torch.zeros(1), None), 0)

        ema_targets = callback.map_metric_ema.update.call_args.args[1]
        assert ema_targets[0]["iscrowd"].tolist() == [0, 1]
        assert ema_targets[0]["masks"].shape == (2, 32, 64)


# Three images: a crowd of people beside two people and a car, a crowd of cars beside one car, and one person with no
# crowd at all. Box coordinates are COCO xywh.
_CAR_CROWD_MASK = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
_CAR_CROWD_MASK[0:128, 0:128] = 1
_PARITY_ANNOTATIONS = [
    _annotation(1, 1, [16, 16, 32, 48]),
    _annotation(2, 1, [64, 16, 32, 48]),
    _annotation(3, 1, [128, 8, 112, 112], iscrowd=1),
    _annotation(4, 1, [16, 80, 64, 32], category_id=2),
    _annotation(5, 2, [0, 0, 128, 128], category_id=2, iscrowd=1, segmentation=_uncompressed_rle(_CAR_CROWD_MASK)),
    _annotation(6, 2, [160, 32, 48, 48], category_id=2),
    _annotation(7, 3, [100, 40, 40, 60]),
]
# Per image: xyxy boxes, scores, labels. The in-crowd person detections lie 100%, 62.5% and 30% inside the crowd, so
# pycocotools ignores each at a different subset of IoU thresholds; a car inside the person crowd stays a false
# positive because a crowd region only ignores its own category. No overlap lands exactly on a 0.05 threshold step,
# where float32 and float64 threshold grids could disagree.
_PARITY_DETECTIONS = [
    (
        [
            [16, 16, 48, 64],
            [68, 20, 100, 68],
            [150, 30, 174, 54],
            [113, 24, 153, 64],
            [100, 20, 140, 60],
            [20, 82, 84, 114],
            [160, 40, 200, 80],
        ],
        [0.95, 0.6, 0.9, 0.85, 0.55, 0.8, 0.75],
        [1, 1, 1, 1, 1, 2, 2],
    ),
    (
        [[162, 30, 210, 82], [8, 8, 40, 40], [60, 60, 120, 120], [101, 90, 141, 126], [200, 90, 240, 120]],
        [0.9, 0.95, 0.5, 0.3, 0.4],
        [2, 2, 2, 2, 1],
    ),
    ([[104, 40, 140, 104], [10, 10, 30, 30]], [0.7, 0.65], [1, 1]),
]


def _pycocotools_stats(
    annotation_file: Path, results: list[dict[str, torch.Tensor]], iou_type: str, max_dets: int
) -> np.ndarray:
    """Score ``results`` (image ids 1, 2, ...) against the raw annotation file with pycocotools, crowd included.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _ = _coco_dataset(Path(tmp), [_annotation(1, 1, _PERSON)])
        ...     prediction = _prediction([[16, 16, 48, 48]], [0.9], [1])
        ...     stats = _pycocotools_stats(Path(tmp) / "annotations.json", [prediction], "bbox", 100)
        >>> round(float(stats[0]), 6)
        1.0
    """
    import pycocotools.mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    detections = []
    for image_id, result in enumerate(results, start=1):
        for index, (box, score, label) in enumerate(
            zip(result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist())
        ):
            detection: dict[str, Any] = {"image_id": image_id, "category_id": label, "score": score}
            if iou_type == "bbox":
                detection["bbox"] = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            else:
                mask = result["masks"][index, 0].numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode()
                detection["segmentation"] = rle
            detections.append(detection)
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation_file))
        evaluator = COCOeval(ground_truth, ground_truth.loadRes(detections), iou_type)
        evaluator.params.maxDets = [1, 10, max_dets]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats


class TestCrowdMetricsMatchPycocotools:
    """On a small COCO file with crowd regions, the callback's metrics equal pycocotools' on the raw file."""

    @pytest.mark.parametrize("backend", [pytest.param(backend, id=backend) for backend in get_args(CocoEvalBackend)])
    @pytest.mark.parametrize("segmentation", [pytest.param(False, id="bbox"), pytest.param(True, id="bbox+segm")])
    def test_callback_metrics_equal_pycocotools(self, tmp_path: Path, backend: str, segmentation: bool) -> None:
        """MAP 50:95 / 50 / 75 and mAR agree with pycocotools, box and mask, on the same predictions."""
        pytest.importorskip("pycocotools")
        pytest.importorskip({"ufcoco": "ultrafast_pycocotools"}.get(backend, backend))
        dataset = _coco_dataset(tmp_path, _PARITY_ANNOTATIONS, images=3, include_masks=segmentation)
        grid = (_HEIGHT, _WIDTH) if segmentation else None
        results = [_prediction(boxes, scores, labels, mask_grid=grid) for boxes, scores, labels in _PARITY_DETECTIONS]
        # pycocotools reads its headline stats[0] at maxDets=100 whatever the configured thresholds, so both use 100.
        callback = COCOEvalCallback(max_dets=100, segmentation=segmentation, eval_backend=backend)

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = _evaluate(callback, _trainer(dataset), "val", results, _eval_targets(dataset))

        expected = {"bbox": _pycocotools_stats(tmp_path / "annotations.json", results, "bbox", 100)}
        observed = {
            "bbox": [metrics["val/mAP_50_95"], metrics["val/mAP_50"], metrics["val/mAP_75"], metrics["val/mAR"]],
        }
        stat_indices = {"bbox": [0, 1, 2, 8]}
        if segmentation:
            expected["segm"] = _pycocotools_stats(tmp_path / "annotations.json", results, "segm", 100)
            observed["segm"] = [metrics["val/segm_mAP_50_95"], metrics["val/segm_mAP_50"]]
            stat_indices["segm"] = [0, 1]
        for iou_type, values in observed.items():
            reference = [float(expected[iou_type][index]) for index in stat_indices[iou_type]]
            assert values == pytest.approx(reference, abs=1e-6), f"{iou_type}: rf-detr {values} vs {reference}"
