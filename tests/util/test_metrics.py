"""
Test suite for precision, recall, and F1 calculation in coco_extended_metrics.

Tests three scenarios:
1. Perfect: All predictions have IoU > 0.95 and match 1-to-1 with GTs (precision = 1.0, recall = 1.0)
2. Degenerate: All predictions have zero IoU with GTs (precision = 0.0, recall = 0.0)
3. Intermediate: Mixed IoU/confidence predictions with hand-calculated expected values

Running this file will create and save visualizations of each scenario in a local
`test_visualizations` directory if you'd like to visually inspect them.
"""

import math
from pathlib import Path

import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rfdetr.engine import coco_extended_metrics
from rfdetr.util.visualize import save_gt_predictions_visualization

VIS_DIR = Path("test_visualizations")
BOX_SIZE = 200
BOX_SPACING = 250
ROW_SPACING = 260


def _create_coco_gt(images: list[dict], annotations: list[dict], categories: list[dict]) -> COCO:
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()
    return coco_gt


def _create_gt_annotation(ann_id: int, image_id: int, category_id: int, bbox: list[float]) -> dict:
    """Create a single GT annotation in COCO format. bbox is [x, y, width, height]."""
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }


def _create_prediction(image_id: int, category_id: int, bbox: list[float], score: float) -> dict:
    """Create a single prediction in COCO results format. bbox is [x, y, width, height]."""
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": score,
    }


def _make_contained_pred_box(gt_box: list[float], target_iou: float) -> list[float]:
    """
    Create a prediction box centered inside gt_box with the specified IoU.

    For a prediction fully contained in GT:
    - Intersection = pred_area = p²
    - Union = gt_area = s²
    - IoU = p²/s², so p = s * sqrt(IoU)
    """
    x, y, gt_size, _ = gt_box

    if target_iou >= 1.0:
        return gt_box.copy()

    pred_size = gt_size * math.sqrt(target_iou)
    offset = (gt_size - pred_size) / 2
    return [x + offset, y + offset, pred_size, pred_size]


def _initialize_coco_eval(coco_gt: COCO, predictions: list[dict]) -> COCOeval:
    """Initialize and run COCOeval, returning the evaluator object."""
    coco_dt = coco_gt.loadRes(predictions) if predictions else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _save_visualization(
    scenario_name: str,
    image_width: int,
    image_height: int,
    gt_boxes: list[list[float]],
    gt_class_ids: list[int],
    pred_boxes: list[list[float]],
    pred_class_ids: list[int],
    pred_confidences: list[float],
    pred_ious: list[float | None],
) -> None:
    """Wrapper to call save_gt_predictions_visualization with VIS_DIR."""
    save_gt_predictions_visualization(
        scenario_name=scenario_name,
        image_width=image_width,
        image_height=image_height,
        gt_boxes=gt_boxes,
        gt_class_ids=gt_class_ids,
        pred_boxes=pred_boxes,
        pred_class_ids=pred_class_ids,
        pred_confidences=pred_confidences,
        pred_ious=pred_ious,
        save_dir=VIS_DIR,
    )


@pytest.fixture
def perfect_scenario_cocoeval():
    """
    Build the "perfect" scenario: all predictions nearly perfectly match GTs.

    Layout (2 rows):
    - Row 0: 5 Class 1 GTs with high-IoU predictions
    - Row 1: 5 Class 2 GTs with high-IoU predictions

    Each prediction has IoU = 0.96 with its GT, which is above COCO's max IoU threshold of 0.95.
    """
    image_id = 1
    n_boxes = 5
    target_iou = 0.96
    image_width = n_boxes * BOX_SPACING
    image_height = 2 * ROW_SPACING

    images = [{"id": image_id, "width": image_width, "height": image_height}]
    categories = [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}]

    annotations = []
    predictions = []
    ann_id = 1
    gt_boxes = []
    gt_class_ids = []
    pred_boxes = []
    pred_class_ids = []
    pred_confidences = []
    pred_ious = []

    for cat_id, row_y in [(1, 0), (2, ROW_SPACING)]:
        for i in range(n_boxes):
            gt_box = [float(i * BOX_SPACING), float(row_y), float(BOX_SIZE), float(BOX_SIZE)]
            pred_box = _make_contained_pred_box(gt_box, target_iou=target_iou)
            annotations.append(_create_gt_annotation(ann_id, image_id, cat_id, gt_box))
            predictions.append(_create_prediction(image_id, cat_id, pred_box, score=1.0))
            ann_id += 1

            gt_boxes.append(gt_box)
            gt_class_ids.append(cat_id)
            pred_boxes.append(pred_box)
            pred_class_ids.append(cat_id)
            pred_confidences.append(1.0)
            pred_ious.append(target_iou)

    _save_visualization(
        "perfect",
        image_width,
        image_height,
        gt_boxes,
        gt_class_ids,
        pred_boxes,
        pred_class_ids,
        pred_confidences,
        pred_ious,
    )

    coco_gt = _create_coco_gt(images, annotations, categories)
    return _initialize_coco_eval(coco_gt, predictions)


@pytest.fixture
def degenerate_scenario_cocoeval():
    """
    Build the degenerate scenario: all predictions have zero IoU with GTs.

    Layout (2 rows, GTs on left, predictions on right with no overlap):
    - Row 0: 5 Class 1 GTs (left) + 5 Class 1 predictions (right, IoU = 0)
    - Row 1: 5 Class 2 GTs (left) + 5 Class 2 predictions (right, IoU = 0)
    """
    image_id = 1
    n_boxes = 5
    gt_pred_gap = 100
    pred_x_offset = n_boxes * BOX_SPACING + gt_pred_gap
    image_width = pred_x_offset + n_boxes * BOX_SPACING
    image_height = 2 * ROW_SPACING

    images = [{"id": image_id, "width": image_width, "height": image_height}]
    categories = [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}]

    annotations = []
    predictions = []
    ann_id = 1
    gt_boxes = []
    gt_class_ids = []
    pred_boxes = []
    pred_class_ids = []
    pred_confidences = []
    pred_ious = []

    # GTs on the left side of the image
    for cat_id, row_y in [(1, 0), (2, ROW_SPACING)]:
        for i in range(n_boxes):
            box = [float(i * BOX_SPACING), float(row_y), float(BOX_SIZE), float(BOX_SIZE)]
            annotations.append(_create_gt_annotation(ann_id, image_id, cat_id, box))
            ann_id += 1

            gt_boxes.append(box)
            gt_class_ids.append(cat_id)

    # Predictions on right side the image with no overlap with GTs
    for cat_id, row_y in [(1, 0), (2, ROW_SPACING)]:
        for i in range(n_boxes):
            box = [float(pred_x_offset + i * BOX_SPACING), float(row_y), float(BOX_SIZE), float(BOX_SIZE)]
            predictions.append(_create_prediction(image_id, cat_id, box, score=1.0))

            pred_boxes.append(box)
            pred_class_ids.append(cat_id)
            pred_confidences.append(1.0)
            pred_ious.append(0.0)

    _save_visualization(
        "degenerate",
        image_width,
        image_height,
        gt_boxes,
        gt_class_ids,
        pred_boxes,
        pred_class_ids,
        pred_confidences,
        pred_ious,
    )

    coco_gt = _create_coco_gt(images, annotations, categories)
    return _initialize_coco_eval(coco_gt, predictions)


@pytest.fixture
def intermediate_scenario_cocoeval():
    """
    Build the intermediate scenario: mixed IoU/confidence predictions.

    Layout (3 rows):
    - Row 0: Class 1 GTs with matching TPs
    - Row 1: Class 2 GTs with matching TPs
    - Row 2: Class 2 FPs (no GTs)

    Class 1: 10 GTs, 10 matching preds, 0 FPs
      - True positive IoU levels:
          [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
      - True positive confidence levels:
          [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

      - No false positives.

    Class 2: 10 GTs, 10 matching preds, 10 FPs
      - True positive IoU levels:
          [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
      - True positive confidence levels:
          [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

      - False positive IoU levels: 0.0 (all false positives are completely non-overlapping with any GT)
      - False positive confidences: [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]


      How to calculate expected metrics
      (Note: These calculations were written out by hand without LLM assistance):
        For macro-F1 calculations, pinning IoU threshold to 0.5, area=all, max_dets=100 and sweeping over confidence:

          - confidence in (0.95, 1.0]:
            - Class 1: 0 TP, 0 FP, 10 FN. Precision = 0.0, Recall = 0.0, F1 = 0.0
            - Class 2: 0 TP, 0 FP, 10 FN. Precision = 0.0, Recall = 0.0, F1 = 0.0
            - Mean-Precision = 0.0, Mean-Recall = 0.0, Mean-F1 = 0.0

          - confidence in (0.9, 0.95]:
            - Class 1: 0 TP, 0 FP, 10 FN. Precision = 0, Recall = 0, F1 = 0.0
            - Class 2: 1 TP, 0 FP, 9 FN. Precision = 1/1, Recall = 1/10, F1 = 0.1818
            - Mean-Precision = 0.5, Mean-Recall = 0.05, Mean-F1 = 0.0909

          ...

          Precision in both class 1 and class 2 will remain 1.0,
          and recall will continue to improve
          until false positives start showing up in class 2 at a confidence of 0.45.

          At a confidence of 0.46, the last entry before false positives show up, it looks like this:

          - confidence in (0.45, 0.5]:
            - Class 1: 5 TP, 0 FP, 5 FN. Precision = 5/5, Recall = 5/10, F1 = 0.6667
            - Class 2: 10 TP, 0 FP, 0 FN. Precision = 10/10, Recall = 10/10, F1 = 1.0
            - Mean-Precision = 1.0, Mean-recall = 0.75, Mean-F1 = 0.8334

          After this, as confidence continues to lower
          class 1's metrics will continue to improve as its recall goes up
          and class 2's metrics will start to degrade as its precision goes down.

          - confidence in (0.40, 0.45]:
            - Class 1: 5 TP, 0 FP, 5 FN. Precision = 5/5, Recall = 5/10, F1 = 0.6667
            - Class 2: 10 TP, 1 FP, 0 FN. Precision = 10/11, Recall = 10/10, F1 = 0.9524
            - Mean-Precision = 0.9545, Mean-Recall = 0.75, Mean-F1 = 0.8096

          - confidence in (0.35, 0.40]:
            - Class 1: 6 TP, 0 FP, 4 FN. Precision = 6/6, Recall = 6/10, F1 = 0.7500
            - Class 2: 10 TP, 2 FP, 0 FN. Precision = 10/12, Recall = 10/10, F1 = 0.9090
            - Mean-Precision = 0.9167, Mean-Recall = 0.8, Mean-F1 = 0.8295

          - confidence in (0.30, 0.35]:
            - Class 1: 6 TP, 0 FP, 4 FN. Precision = 6/6, Recall = 6/10, F1 = 0.7500
            - Class 2: 10 TP, 3 FP, 0 FN. Precision = 10/13, Recall = 10/10, F1 = 0.8696
            - Mean-Precision = 0.8846, Mean-Recall = 0.8, Mean-F1 = 0.8098

          - confidence in (0.25, 0.30]:
            - Class 1: 7 TP, 0 FP, 3 FN. Precision = 7/7, Recall = 7/10, F1 = 0.8235
            - Class 2: 10 TP, 4 FP, 0 FN. Precision = 10/14, Recall = 10/10, F1 = 0.8333
            - Mean-Precision = 0.8571, Mean-Recall = 0.85, Mean-F1 = 0.8284

          - confidence in (0.20, 0.25]:
            - Class 1: 7 TP, 0 FP, 3 FN. Precision = 7/7, Recall = 7/10, F1 = 0.8235
            - Class 2: 10 TP, 5 FP, 0 FN. Precision = 10/15, Recall = 10/10, F1 = 0.8000
            - Mean-Precision = 0.8333, Mean-Recall = 0.85, Mean-F1 = 0.81175

          - confidence in (0.15, 0.20]:
            - Class 1: 8 TP, 0 FP, 2 FN. Precision = 8/8, Recall = 8/10, F1 = 0.8889
            - Class 2: 10 TP, 6 FP, 0 FN. Precision = 10/16, Recall = 10/10, F1 = 0.7692
            - Mean-Precision = 0.8125, Mean-Recall = 0.9, Mean-F1 = 0.8291

          - confidence in (0.10, 0.15]:
            - Class 1: 8 TP, 0 FP, 2 FN. Precision = 8/8, Recall = 8/10, F1 = 0.8889
            - Class 2: 10 TP, 7 FP, 0 FN. Precision = 10/17, Recall = 10/10, F1 = 0.7407
            - Mean-Precision = 0.7941, Mean-Recall = 0.9, Mean-F1 = 0.8148

          - confidence in (0.05, 0.10]:
            - Class 1: 9 TP, 0 FP, 1 FN. Precision = 9/9, Recall = 9/10, F1 = 0.9474
            - Class 2: 10 TP, 8 FP, 0 FN. Precision = 10/18, Recall = 10/10, F1 = 0.7143
            - Mean-precision = 0.7778, Mean-Recall = 0.95, Mean-F1 = 0.8309

          - confidence in (0.00, 0.05]:
            - Class 1: 9 TP, 0 FP, 1 FN. Precision = 9/9, Recall = 9/10, F1 = 0.9474
            - Class 2: 10 TP, 9 FP, 0 FN. Precision = 10/19, Recall = 10/10, F1 = 0.6897
            - Mean-Precision = 0.7632, Mean-Recall = 0.95, Mean-F1 = 0.8186

          - confidence = 0.00:
            - Class 1: 10 TP, 0 FP, 0 FN. Precision = 10/10, Recall = 10/10, F1 = 1.0
            - Class 2: 10 TP, 10 FP, 0 FN. Precision = 10/20, Recall = 10/10, F1 = 0.6667
            - Mean-Precision = 0.75, Mean-Recall = 1.0, Mean-F1 = 0.8334

        Suprisingly, confidence=0 ties with confidence=0.46 for the best macro-F1.
        So an algorithm that sweeps confidence from 0 to 1 will probably just choose
        confidence of 0.0, resulting in a Mean-F1 of 0.8334, a Mean-Precision of 0.75,
        and a Mean-Recall of 1.0.

        I have set these values as the expected metrics for this intermediate scenario.
        The test passes if the precision and recall values are within 0.01 of the
        expected values.

    """
    image_id = 1
    n_boxes = 10

    class1_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class1_confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    class2_ious = [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
    class2_confs = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    class2_fp_confs = [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    image_width = n_boxes * BOX_SPACING
    image_height = 3 * ROW_SPACING

    images = [{"id": image_id, "width": image_width, "height": image_height}]
    categories = [{"id": 1, "name": "class_1"}, {"id": 2, "name": "class_2"}]

    annotations = []
    predictions = []
    ann_id = 1
    gt_boxes = []
    gt_class_ids = []
    pred_boxes = []
    pred_class_ids = []
    pred_confidences = []
    pred_ious = []

    # Row 0: Class 1 GTs with TP predictions
    for i, (iou, conf) in enumerate(zip(class1_ious, class1_confs)):
        gt_box = [float(i * BOX_SPACING), 0.0, float(BOX_SIZE), float(BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        annotations.append(_create_gt_annotation(ann_id, image_id, 1, gt_box))
        predictions.append(_create_prediction(image_id, 1, pred_box, score=conf))
        ann_id += 1

        gt_boxes.append(gt_box)
        gt_class_ids.append(1)
        pred_boxes.append(pred_box)
        pred_class_ids.append(1)
        pred_confidences.append(conf)
        pred_ious.append(iou)

    # Row 1: Class 2 GTs with TP predictions
    for i, (iou, conf) in enumerate(zip(class2_ious, class2_confs)):
        gt_box = [float(i * BOX_SPACING), float(ROW_SPACING), float(BOX_SIZE), float(BOX_SIZE)]
        pred_box = _make_contained_pred_box(gt_box, target_iou=iou)
        annotations.append(_create_gt_annotation(ann_id, image_id, 2, gt_box))
        predictions.append(_create_prediction(image_id, 2, pred_box, score=conf))
        ann_id += 1

        gt_boxes.append(gt_box)
        gt_class_ids.append(2)
        pred_boxes.append(pred_box)
        pred_class_ids.append(2)
        pred_confidences.append(conf)
        pred_ious.append(iou)

    # Row 2: Class 2 FPs
    for i, fp_conf in enumerate(class2_fp_confs):
        fp_box = [float(i * BOX_SPACING), float(2 * ROW_SPACING), float(BOX_SIZE), float(BOX_SIZE)]
        predictions.append(_create_prediction(image_id, 2, fp_box, score=fp_conf))

        pred_boxes.append(fp_box)
        pred_class_ids.append(2)
        pred_confidences.append(fp_conf)
        pred_ious.append(None)

    _save_visualization(
        "intermediate",
        image_width,
        image_height,
        gt_boxes,
        gt_class_ids,
        pred_boxes,
        pred_class_ids,
        pred_confidences,
        pred_ious,
    )

    coco_gt = _create_coco_gt(images, annotations, categories)
    return _initialize_coco_eval(coco_gt, predictions)


def test_perfect_scenario(perfect_scenario_cocoeval):
    """
    Test that perfect predictions (exact GT matches) yield perfect metrics.

    Expected: Precision = 1.0, Recall = 1.0
    """
    results = coco_extended_metrics(perfect_scenario_cocoeval)

    assert results["precision"] == pytest.approx(1.0, abs=0.01)
    assert results["recall"] == pytest.approx(1.0, abs=0.01)


def test_degenerate_scenario(degenerate_scenario_cocoeval):
    """
    Test that completely wrong predictions (zero IoU) yield zero metrics.

    Expected: Precision = 0.0, Recall = 0.0
    """
    results = coco_extended_metrics(degenerate_scenario_cocoeval)

    assert results["precision"] == pytest.approx(0.0, abs=0.01)
    assert results["recall"] == pytest.approx(0.0, abs=0.01)


def test_intermediate_scenario(intermediate_scenario_cocoeval):
    """
    Test intermediate scenario with verifiable expected metrics.

    Class 1: 10 GTs, 10 matching preds, 0 FPs
      - IoU levels: [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
      - Confidence levels: [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

    Class 2: 10 GTs, 10 matching preds, 10 FPs
      - TP IoU levels: [0.975, 0.925, 0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525]
      - TP Confidence levels: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
      - FP confidences: [0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]

    Expected (from hand calculations in docstring):
    - Best macro-F1 = 0.8334 (at confidence 0.0 or 0.46)
    - At confidence 0.0: Mean-Precision = 0.75, Mean-Recall = 1.0
    """
    results = coco_extended_metrics(intermediate_scenario_cocoeval)

    assert results["precision"] == pytest.approx(0.75, abs=0.01)
    assert results["recall"] == pytest.approx(1.0, abs=0.01)

    class_map = results["class_map"]
    per_class_metrics = {entry["class"]: entry for entry in class_map if entry["class"] != "all"}

    assert per_class_metrics["class_1"]["precision"] == pytest.approx(1.0, abs=0.01)
    assert per_class_metrics["class_1"]["recall"] == pytest.approx(1.0, abs=0.01)

    assert per_class_metrics["class_2"]["precision"] == pytest.approx(0.5, abs=0.01)
    assert per_class_metrics["class_2"]["recall"] == pytest.approx(1.0, abs=0.01)
