# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Per-backend tests for the one-pass COCO adapter: hotcoco, ufcoco and vernier against faster-coco-eval."""

import copy
import pickle
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.training.coco_map import (
    OnePassCocoMeanAveragePrecision,
    _hotcoco,
    _silenced_backend_diagnostics,
    _ufcoco,
    _UfcocoBackend,
    _vernier,
    _vernier_thread_budget,
    _VernierBackend,
)
from tests._markers import _requires_hotcoco, _requires_ufcoco, _requires_vernier
from tests.training._coco_map_support import (
    _ALTERNATIVE_BACKENDS,
    _BACKEND_KWARGS,
    _assert_results_close,
    _assert_streams_when_backend_streams,
    _metric_inputs,
    multiclass_detection_state,
)


@pytest.mark.parametrize("backend", _ALTERNATIVE_BACKENDS)
def test_alternative_backend_matches_faster_coco_eval(backend: str) -> None:
    """Hotcoco, ufcoco and vernier must return the same metrics as the faster-coco-eval backend they replace.

    Exact equality is required of all three. ufcoco reproduces pycocotools' float64 summary to the byte, which differs
    from faster-coco-eval's in the last float64 digit on some entries (pycocotools adds ``np.spacing(1)`` to the
    precision denominator); the float32 tensors TorchMetrics reports absorb that here, so the same assertion holds.
    """
    predictions, targets = multiclass_detection_state()
    kwargs: dict[str, Any] = {"class_metrics": True, "sync_on_compute": False}
    expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs, **_BACKEND_KWARGS.get(backend, {}))
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)
    _assert_streams_when_backend_streams(actual_metric)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    assert set(expected["map_per_class"].tolist()) - {-1.0, 0.0, 1.0}, (
        "fixture produced only degenerate per-class values; strengthen it before trusting this parity check"
    )
    _assert_results_close(actual, expected)


@pytest.mark.parametrize("backend", _ALTERNATIVE_BACKENDS)
def test_alternative_backend_matches_faster_coco_eval_for_segmentation(backend: str) -> None:
    """Mask metrics must match across backends, including the per-IoU-type area swap.

    Segmentation is where the backends diverge structurally. hotcoco returns a copy from its ``dataset`` getter, so the
    ``area_bbox``/``area_segm`` swap the multi-IoU-type path performs reaches its evaluator only through
    ``COCO.update_anns`` (the supported hotcoco floor is past the earlier bytes-RLE constructor mismatch). Every backend
    reads the RLE the adapter encodes itself, so an RLE one backend misreads would show up here; a mask AP that silently
    collapses to 0.0 is what either mistake would look like.
    """
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[0, 2:10, 2:10] = True
    mask[1, 11:15, 11:15] = True
    predicted_mask = mask.clone()
    predicted_mask[0, 2:4, 2:10] = False
    predictions = [
        {
            "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0], [11.0, 11.0, 15.0, 15.0]]),
            "masks": predicted_mask,
            "scores": torch.tensor([0.9, 0.6]),
            "labels": torch.tensor([3, 17]),
        }
    ]
    targets = [
        {
            "boxes": torch.tensor([[2.0, 2.0, 10.0, 10.0], [11.0, 11.0, 15.0, 15.0]]),
            "masks": mask,
            "labels": torch.tensor([3, 17]),
        }
    ]
    kwargs: dict[str, Any] = {"iou_type": ("bbox", "segm"), "class_metrics": True, "sync_on_compute": False}
    expected_metric = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual_metric = OnePassCocoMeanAveragePrecision(backend=backend, **kwargs, **_BACKEND_KWARGS.get(backend, {}))
    expected_metric.update(predictions, targets)
    actual_metric.update(predictions, targets)
    _assert_streams_when_backend_streams(actual_metric)

    expected = expected_metric.compute()
    actual = actual_metric.compute()

    assert actual.keys() == expected.keys()
    # A mask AP that silently collapses to 0.0 is the failure this fixture exists to catch, so the fixture itself
    # has to produce a non-zero one first.
    assert expected["segm_map"].item() > 0.0
    _assert_results_close(actual, expected)


@_requires_hotcoco
@pytest.mark.parametrize("backend", ["hotcoco", "hotcoco_streaming"])
def test_hotcoco_evaluation_prints_nothing(capfd: pytest.CaptureFixture[str], backend: str) -> None:
    """Selecting hotcoco must not add backend chatter to a training run's console output.

    hotcoco routes its COCO summary table through ``sys.stdout`` and raises configuration diagnostics as
    ``UserWarning``s, so both are reachable with ordinary Python-level redirection. Without suppression, the table would
    land on the console on every validation epoch of every run. ``test_hotcoco_evaluation_raises_no_warnings`` covers
    the warning channel. ``hotcoco_streaming`` matches in ``update()``, so the window covers it too.
    """
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend=backend, max_detection_thresholds=[1, 10, 500], sync_on_compute=False, num_classes=90
    )
    capfd.readouterr()

    metric.update(predictions, targets)
    metric.compute()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@_requires_hotcoco
@pytest.mark.parametrize("backend", ["hotcoco", "hotcoco_streaming"])
def test_hotcoco_evaluation_raises_no_warnings(backend: str) -> None:
    """Selecting hotcoco must not raise a warning per evaluation for configuration RF-DETR chose deliberately.

    hotcoco reports RF-DETR's configured ``maxDets`` as a warning; its 1.2.1 release no longer reports the float32
    threshold grids. The stdout table is what ``test_hotcoco_evaluation_prints_nothing`` asserts on; the warning channel
    reaches a caller's ``catch_warnings``, a notebook cell, or a ``-W error`` run. ``hotcoco_streaming`` matches in
    ``update()``, so the window covers it too.
    """
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend=backend, max_detection_thresholds=[1, 10, 500], sync_on_compute=False, num_classes=90
    )

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        metric.update(predictions, targets)
        metric.compute()

    assert [str(warning.message) for warning in raised] == []


def test_hotcoco_suppresses_only_the_max_dets_warning() -> None:
    """Suppress hotcoco's intentional detection-limit warning without hiding unrelated warnings."""
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        with _silenced_backend_diagnostics():
            warnings.warn("hotcoco: max_dets differ from expected ([1, 10, 100])", UserWarning)
            warnings.warn("unrelated diagnostic", UserWarning)

    assert [str(warning.message) for warning in raised] == ["unrelated diagnostic"]


@_requires_ufcoco
def test_ufcoco_reports_aggregate_ap_at_the_configured_detection_limit() -> None:
    """``map`` must be a real number, equal to faster-coco-eval's, when ``eval_max_dets`` is not 100.

    pycocotools summarizes ``stats[0]`` at ``maxDets=100`` regardless of the configured thresholds and returns ``-1``
    when 100 is not among them; ufcoco reproduces that, and faster-coco-eval and hotcoco read the largest configured
    threshold instead. RF-DETR evaluates at 500 by default, so without the adapter's evaluator override every
    ``val/mAP`` on this backend would read ``-1`` while the rest of the metrics looked plausible.
    """
    predictions, targets = multiclass_detection_state()

    def aggregate_ap(backend: str) -> float:
        metric = OnePassCocoMeanAveragePrecision(
            backend=backend, max_detection_thresholds=[1, 10, 500], sync_on_compute=False
        )
        metric.update(predictions, targets)
        return float(metric.compute()["map"])

    expected = aggregate_ap("faster_coco_eval")
    assert expected > 0.0, "fixture produced no aggregate AP; strengthen it before trusting this check"
    assert aggregate_ap("ufcoco") == expected


@_requires_ufcoco
def test_ufcoco_evaluation_prints_nothing(capfd: pytest.CaptureFixture[str]) -> None:
    """Selecting ufcoco must not add backend chatter to a training run's console output.

    ufcoco prints pycocotools' evaluation progress lines and twelve-row summary table on ``sys.stdout``, once per IoU
    type per ``compute()``; the adapter's evaluation window has to swallow them the way it does for the other backends.
    """
    predictions, targets = multiclass_detection_state()
    metric = OnePassCocoMeanAveragePrecision(
        backend="ufcoco", max_detection_thresholds=[1, 10, 500], sync_on_compute=False
    )
    metric.update(predictions, targets)
    capfd.readouterr()

    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        metric.compute()

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert [str(warning.message) for warning in raised] == []


@pytest.mark.parametrize(
    ("backend", "extra"),
    [pytest.param("hotcoco", "rfdetr[hotcoco]", id="hotcoco"), pytest.param("ufcoco", "rfdetr[train]", id="ufcoco")],
)
def test_missing_backend_dependency_names_the_extra(backend: str, extra: str, mocker: MockerFixture) -> None:
    """Selecting a backend whose package is not installed must say how to install it.

    The private-contract check that runs right after backend construction resolves the evaluator inside an ``except
    ImportError``, so an import deferred until then would be reported as a torchmetrics incompatibility -- the one
    message that tells a user nothing about the missing extra.
    """
    message = f"backend={backend!r} requires an optional package; install it with: pip install '{extra}'"
    mocker.patch(f"rfdetr.training.coco_map._{backend}", side_effect=ImportError(message))

    with pytest.raises(ImportError) as exc_info:
        OnePassCocoMeanAveragePrecision(backend=backend)

    assert extra in str(exc_info.value)


@pytest.mark.parametrize(
    ("importer", "package"),
    [
        pytest.param(_hotcoco, "hotcoco", id="hotcoco"),
        pytest.param(_ufcoco, "ultrafast_pycocotools", id="ufcoco"),
        pytest.param(_vernier, "vernier", id="vernier"),
    ],
)
def test_missing_backend_package_names_the_extra(
    importer: Callable[[], Any], package: str, mocker: MockerFixture
) -> None:
    """A missing backend package must retain the actionable installation guidance."""
    missing_package = ModuleNotFoundError(f"No module named '{package}'", name=package)

    mocker.patch("builtins.__import__", side_effect=missing_package)

    with pytest.raises(ImportError) as exc_info:
        importer()

    assert "rfdetr[train]" in str(exc_info.value)


def test_ufcoco_propagates_nested_import_error(mocker: MockerFixture) -> None:
    """A failure inside ufcoco must retain the missing nested dependency name."""
    nested_error = ModuleNotFoundError("No module named 'nested_dependency'", name="nested_dependency")

    mocker.patch("builtins.__import__", side_effect=nested_error)

    with pytest.raises(ModuleNotFoundError) as raised:
        _ufcoco()

    assert raised.value is nested_error


@_requires_ufcoco
def test_ufcoco_backend_picks_up_the_optional_package_and_survives_pickling() -> None:
    """The ufcoco backend must resolve its surfaces from ``ultrafast_pycocotools`` and pickle with the metric.

    Lightning's DDP spawn and the callback's checkpoint plumbing pickle metric objects; a backend that stored the
    imported module on itself would break there, which is why the surfaces are resolved on every access.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="ufcoco", sync_on_compute=False)
    backend = metric._coco_backend

    assert isinstance(backend, _UfcocoBackend)
    assert backend.backend == "faster_coco_eval", "torchmetrics must still see the supported enum member"
    assert backend.coco.__module__.startswith("ultrafast_pycocotools")
    assert backend.cocoeval.__mro__[1].__module__.startswith("ultrafast_pycocotools")
    assert backend.mask_utils.area.__module__.startswith("ultrafast_pycocotools")

    restored = pickle.loads(pickle.dumps(metric))

    assert isinstance(restored._coco_backend, _UfcocoBackend)
    assert restored._coco_backend.cocoeval is backend.cocoeval


@_requires_vernier
def test_vernier_backend_survives_pickling() -> None:
    """The vernier backend must pickle with the metric, which Lightning's DDP spawn and checkpoint plumbing require."""
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)

    restored = pickle.loads(pickle.dumps(metric))

    assert isinstance(restored._coco_backend, _VernierBackend)


@pytest.mark.parametrize(
    "local_world_size,expected",
    [
        pytest.param(None, 4, id="unset-uses-the-whole-pool"),
        pytest.param("2", 2, id="divides-evenly-among-local-ranks"),
        pytest.param("8", 1, id="oversized-world-size-clamps-to-one"),
    ],
)
def test_vernier_thread_budget_divides_by_local_world_size(
    monkeypatch: pytest.MonkeyPatch, local_world_size: str | None, expected: int
) -> None:
    """The per-rank thread budget divides the process-wide pool by ``LOCAL_WORLD_SIZE``, never below one.

    Every DDP rank sharing one node sees ``torch.get_num_threads()`` report the same process-wide count; handing vernier
    that whole budget on every rank oversubscribes the node's CPUs by a factor of ``LOCAL_WORLD_SIZE`` unless it is
    divided among the ranks running on that node, floored at one thread.
    """
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)

    assert _vernier_thread_budget() == expected


@_requires_vernier
def test_vernier_rejects_mask_only_evaluation() -> None:
    """Mask-only evaluation must fail at construction, before an epoch of state is accumulated and discarded."""
    with pytest.raises(ValueError) as exc_info:
        OnePassCocoMeanAveragePrecision(backend="vernier", iou_type="segm", sync_on_compute=False)
    assert "requires 'bbox'" in str(exc_info.value)


@_requires_vernier
def test_vernier_rejects_fractional_ground_truth_labels() -> None:
    """A fractional ground-truth class label must be rejected instead of silently truncated by ``.long()``.

    ``coco_inputs`` owns the check now, but a fractional value must still fail loudly rather than become a wrong class
    through an ``int64`` cast. Driven through ``compute()`` so it pins the behaviour wherever the check lives.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "scores": torch.tensor([0.9]), "labels": torch.tensor([3])}],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3.5])}],
    )

    with pytest.raises(ValueError) as exc_info:
        metric.compute()
    assert "must be integral" in str(exc_info.value)


@_requires_vernier
def test_vernier_rejects_fractional_detection_labels_under_segmentation() -> None:
    """A fractional detection class label under ``segm`` must be rejected instead of truncated by ``.long()``.

    ``coco_inputs`` validates both sides and both routes, so this no longer depends on which route an IoU type takes.
    ``segm`` runs first so the failure is attributed to the detection labels.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", iou_type=("segm", "bbox"), sync_on_compute=False)
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, :2, :2] = True
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3.5]),
                "masks": mask,
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([3]), "masks": mask}],
    )

    with pytest.raises(ValueError) as exc_info:
        metric.compute()
    assert "must be integral" in str(exc_info.value)


@_requires_vernier
def test_vernier_accepts_whole_valued_float_labels() -> None:
    """A whole-valued floating-point label must be accepted, matching the equivalent integer label.

    Only a fractional value is a truncation risk; rejecting every floating-point label outright would also reject an
    empty ``float32`` label tensor, which carries no values to truncate.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3.0]),
            }
        ],
        [{"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3.0])}],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


@_requires_vernier
def test_vernier_handles_one_dimensional_empty_boxes() -> None:
    """A 1-D empty box tensor on one image must not crash concatenation against another image's real boxes.

    TorchMetrics' ``_fix_empty_tensors`` reshapes a 1-D empty box tensor to ``(1, 0)`` rather than ``(0, 4)`` to avoid a
    DDP all-reduce hang (``torchmetrics.detection.helpers``). Concatenating that ``(1, 0)`` tensor against a ``(N, 4)``
    tensor from another image in the same batch fails on the mismatched second dimension unless each per-image tensor is
    reshaped to ``(-1, 4)`` first, for both the ground truth and the ``(N, 7)`` detection matrix ``bbox`` uses.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {"boxes": torch.empty(0), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            },
        ],
        [
            {"boxes": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


@_requires_vernier
def test_vernier_handles_one_dimensional_empty_boxes_under_segmentation() -> None:
    """A 1-D empty box tensor on one image must not crash the columnar detection route ``segm`` uses.

    ``_vernier_detections`` takes the columnar route under ``segm`` instead of the ``(N, 7)`` matrix ``bbox`` uses, with
    its own ``torch.cat`` over ``detection_box`` that needs the same ``(-1, 4)`` reshape guard.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", iou_type=("bbox", "segm"), sync_on_compute=False)
    mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    mask[0, :2, :2] = True
    empty_mask = torch.zeros((0, 4, 4), dtype=torch.bool)
    metric.update(
        [
            {
                "boxes": torch.empty(0),
                "scores": torch.empty(0),
                "labels": torch.empty(0, dtype=torch.long),
                "masks": empty_mask,
            },
            {
                "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
                "masks": mask,
            },
        ],
        [
            {"boxes": torch.empty(0), "labels": torch.empty(0, dtype=torch.long), "masks": empty_mask},
            {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([3]), "masks": mask},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


@_requires_vernier
def test_vernier_ground_truth_handles_one_dimensional_empty_boxes() -> None:
    """A one-dimensional empty ground-truth box tensor must not crash concatenation against another image's boxes.

    TorchMetrics' ``_fix_empty_tensors`` (``torchmetrics.detection.helpers``) reshapes an empty ``torch.tensor([])`` box
    tensor to ``(1, 0)`` rather than ``(0, 4)``, to avoid a DDP all-reduce hang. ``_vernier_ground_truth`` concatenates
    every image's ground-truth boxes into one tensor, so a ``(1, 0)`` entry next to a real image's ``(N, 4)`` entry
    fails on the mismatched second dimension unless each per-image tensor is reshaped to ``(-1, 4)`` first.
    """
    metric = OnePassCocoMeanAveragePrecision(backend="vernier", sync_on_compute=False)
    metric.update(
        [
            {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([3]),
            },
        ],
        [
            {"boxes": torch.tensor([]), "labels": torch.empty(0, dtype=torch.long)},
            {"boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]), "labels": torch.tensor([3])},
        ],
    )

    result = metric.compute()

    assert torch.equal(result["classes"].reshape(-1), torch.tensor([3], dtype=torch.int32))


@pytest.mark.parametrize("iou_type", ["bbox", pytest.param(("bbox", "segm"), id="both")])
@pytest.mark.parametrize("max_dets", [100, 500])
@_requires_vernier
def test_vernier_matches_faster_coco_eval_across_updates_and_reuse(iou_type: Any, max_dets: int) -> None:
    """Vernier's native path must match faster-coco-eval exactly on ties, crowds and images missing either side.

    It builds its own arrays instead of the COCO datasets every other backend shares, so it is held to exact equality on
    a fixture carrying everything those datasets encode, at ``eval_max_dets`` both at and above pycocotools' 100.
    """
    predictions, targets = _metric_inputs()
    kwargs: dict[str, Any] = {
        "iou_type": iou_type,
        "class_metrics": True,
        "max_detection_thresholds": [1, 10, max_dets],
    }
    reference = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual = OnePassCocoMeanAveragePrecision(backend="vernier", **kwargs)
    for _ in range(2):  # the second pass runs on a reset metric
        for metric in (reference, actual):
            for index in range(len(predictions)):
                metric.update(copy.deepcopy(predictions[index : index + 1]), copy.deepcopy(targets[index : index + 1]))
            metric.merge_distributed_state()
        expected, observed = reference.compute(), actual.compute()
        assert observed.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)
        reference.reset()
        actual.reset()


#: Side of the canvas `_rasterize` draws on; the fixture keeps every coordinate inside it.
_CANVAS = 128


def _vernier_gt_state(
    seed: int,
    *,
    with_area: bool = True,
    with_crowd: bool = True,
    empty_images: bool = True,
    with_masks: bool = False,
    crowd_value: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ground truth shaped to exercise the columnar builder's edge cases, not just its happy path.

    Args:
        seed: Torch manual seed, so a case is reproducible.
        with_area: Whether targets carry an ``area``, half of it zero so the per-element fallback is exercised.
        with_crowd: Whether targets carry an ``iscrowd`` flag.
        empty_images: Whether every fifth image has no ground truth at all.
        with_masks: Whether both sides carry masks, so the run evaluates ``segm`` as well as ``bbox``.
        crowd_value: Value written for a crowd annotation, to exercise flags wider than ``uint8``.

    Returns:
        Predictions and targets in TorchMetrics detection format, one entry per image.

    Examples:
        >>> predictions, targets = _vernier_gt_state(1)
        >>> len(targets), sorted(targets[1])
        (40, ['area', 'boxes', 'iscrowd', 'labels'])
        >>> int(targets[1]["boxes"].shape[0]), int(targets[1]["iscrowd"].max())
        (6, 1)
    """
    torch.manual_seed(seed)
    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for index in range(40):
        count = 0 if (empty_images and index % 5 == 0) else 6
        origin = torch.rand(count, 2) * 90
        extent = torch.rand(count, 2) * 24 + 8
        boxes = torch.cat([origin, origin + extent], 1) if count else torch.zeros((0, 4))
        labels = torch.randint(0, 15, (count,))
        target: dict[str, Any] = {"boxes": boxes, "labels": labels}
        if with_area and count:
            # Half the areas are 0, which upstream replaces with a computed one per element.
            area = (extent[:, 0] * extent[:, 1]).clone()
            area[::2] = 0.0
            target["area"] = area
        if with_crowd:
            target["iscrowd"] = (torch.arange(count) % 4 == 0).long() * crowd_value
        detections = boxes.repeat(4, 1) + torch.randn(count * 4, 4) * 5 if count else torch.zeros((0, 4))
        if count:
            detections[:, 2:] = torch.maximum(detections[:, 2:], detections[:, :2] + 1)
        prediction: dict[str, Any] = {
            "boxes": detections,
            "scores": torch.rand(count * 4),
            "labels": labels.repeat(4),
        }
        if with_masks:
            # A mask's area differs from its box's `w * h`, which tells the two area rules apart.
            target["masks"] = _rasterize(boxes)
            prediction["masks"] = _rasterize(detections)
        targets.append(target)
        predictions.append(prediction)
    return predictions, targets


def _rasterize(boxes: torch.Tensor) -> torch.Tensor:
    """Return one filled-rectangle mask per box, on the fixture's fixed canvas.

    Args:
        boxes: ``(N, 4)`` boxes in ``xyxy``.

    Returns:
        ``(N, _CANVAS, _CANVAS)`` boolean masks.

    Examples:
        >>> masks = _rasterize(torch.tensor([[1.0, 2.0, 5.0, 4.0]]))
        >>> masks.shape[0], int(masks.sum())
        (1, 8)
    """
    masks = torch.zeros((len(boxes), _CANVAS, _CANVAS), dtype=torch.bool)
    for index, (x0, y0, x1, y1) in enumerate(boxes.round().clamp(0, _CANVAS).int().tolist()):
        masks[index, y0:y1, x0:x1] = True
    return masks


@pytest.mark.parametrize(
    ("with_area", "with_crowd", "empty_images", "with_masks"),
    [
        pytest.param(False, False, False, False, id="boxes_only"),
        pytest.param(True, False, False, False, id="area"),
        pytest.param(False, True, False, False, id="crowd"),
        pytest.param(False, False, True, False, id="empty_images"),
        pytest.param(True, True, True, False, id="all"),
        pytest.param(True, True, True, True, id="masks"),
    ],
)
@_requires_vernier
def test_vernier_columnar_ground_truth_is_the_same_document(
    with_area: bool, with_crowd: bool, empty_images: bool, with_masks: bool
) -> None:
    """The columnar ground truth must be the *document* TorchMetrics' COCO format is, not merely score the same.

    ``coco_inputs_from_columns`` builds the document, but mapping *this* metric's state onto it is still
    RF-DETR's, and that is what this pins -- by ``dataset_hash`` against the route it replaced, ``_get_coco_format``
    through vernier's normalizers and JSON parser.

    Metrics would not catch it: most cases here differ only in ways AP cannot see, such as an image entry with no
    annotations on it or an ``area`` on an annotation that matches nothing. The cases are the rules easiest to get
    wrong -- the per-element ``area`` fallback, ``iscrowd``, an image with no ground truth, and under ``segm`` the
    size each image takes from its own first mask.
    """
    predictions, targets = _vernier_gt_state(
        1, with_area=with_area, with_crowd=with_crowd, empty_images=empty_images, with_masks=with_masks
    )
    iou_type = ("bbox", "segm") if with_masks else ("bbox",)
    metric = OnePassCocoMeanAveragePrecision(
        backend="vernier", box_format="xyxy", iou_type=iou_type, class_metrics=True
    )
    metric.update(predictions, targets)
    classes = sorted({int(label) for target in metric.groundtruth_labels for label in target.tolist()})

    adapters = _vernier().adapters
    reference = metric._coco_backend._get_coco_format(
        labels=metric.groundtruth_labels,
        boxes=metric.groundtruth_box,
        masks=metric.groundtruth_mask if with_masks else None,
        crowds=metric.groundtruth_crowds,
        area=metric.groundtruth_area,
        iou_type=metric.iou_type,
        all_labels=classes,
        average=metric.average,
    )
    detection_sizes = {
        image_id: tuple(masks[0][0]) for image_id, masks in enumerate(metric.detection_mask) if len(masks) > 0
    }
    sized = (
        adapters.with_mask_image_sizes(reference, detection_sizes)
        if with_masks
        else adapters.with_placeholder_image_sizes(reference)
    )
    from_json = _vernier().instance.CocoDataset.from_json(adapters.to_coco_json(sized))
    from_arrays, _ = adapters.coco_inputs_from_columns(
        *metric._vernier_columns(),
        box_format="xywh",
        categories=classes,
        area="auto",
    )

    assert from_arrays.dataset_hash == from_json.dataset_hash
    assert (from_arrays.num_images, from_arrays.num_annotations, from_arrays.num_categories) == (
        from_json.num_images,
        from_json.num_annotations,
        from_json.num_categories,
    )


@_requires_vernier
def test_vernier_ground_truth_builder_keeps_crowd_flags_wider_than_uint8() -> None:
    """The ground-truth builder must keep ``iscrowd`` at its stored width, not through a column that wraps it.

    ``coco_inputs`` builds its columns directly from tensors rather than through JSON, so
    ``test_vernier_columnar_ground_truth_is_the_same_document`` cannot cover a value like 256: vernier's own JSON
    parser rejects anything but 0/1 for ``iscrowd``, so the reference side of that test would raise first. Comparing
    the built dataset's ``dataset_hash`` against one built with no crowd annotation at all catches the same defect
    without JSON in the loop -- a ``uint8`` column would wrap 256 back to 0 and make the two datasets identical.
    """

    def ground_truth(crowd_value: int) -> Any:
        predictions, targets = _vernier_gt_state(1, crowd_value=crowd_value)
        metric = OnePassCocoMeanAveragePrecision(
            backend="vernier", box_format="xyxy", iou_type="bbox", class_metrics=True
        )
        metric.update(predictions, targets)
        classes = sorted({int(label) for target in metric.groundtruth_labels for label in target.tolist()})
        ground_truth, _ = _vernier().adapters.coco_inputs_from_columns(
            *metric._vernier_columns(), box_format="xywh", categories=classes
        )
        return ground_truth

    wide_crowd, no_crowd = ground_truth(256), ground_truth(0)

    assert wide_crowd.dataset_hash != no_crowd.dataset_hash
    assert wide_crowd.num_annotations == no_crowd.num_annotations


@_requires_vernier
def test_vernier_columnar_route_keeps_crowd_flags_wider_than_uint8() -> None:
    """A crowd flag must reach vernier at its stored width, not through a column that wraps it.

    ``iscrowd`` is 0/1 in COCO practice, so a ``uint8`` column looks harmless until 256 wraps to 0 and the annotation is
    matched as a normal one instead of ignored -- silently, and only on that value, since 255 and 257 both survive.
    """
    predictions, targets = _vernier_gt_state(1, crowd_value=256)
    kwargs: dict[str, Any] = {"box_format": "xyxy", "iou_type": "bbox", "class_metrics": True}
    reference = OnePassCocoMeanAveragePrecision(backend="faster_coco_eval", **kwargs)
    actual = OnePassCocoMeanAveragePrecision(backend="vernier", **kwargs)
    for metric in (reference, actual):
        metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
    expected, observed = reference.compute(), actual.compute()

    for key in expected:
        torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("iou_type", ["bbox", pytest.param(("bbox", "segm"), id="both")])
@pytest.mark.parametrize("max_dets", [100, 500])
@_requires_vernier
def test_vernier_parity_modes_differ_only_on_the_aggregate_at_a_non_default_max_dets(
    iou_type: Any, max_dets: int
) -> None:
    """Vernier's ``strict`` and ``corrected`` modes must differ on this path in exactly one metric, and only there.

    ``corrected`` is a bundle that has grown between releases, and ``_vernier_results`` claims exactly one member of
    it reaches RF-DETR: the aggregate ``map``, which pycocotools reads at a literal ``maxDets=100`` and so reports as
    ``-1`` whenever 100 is not among the configured limits. Everything else vernier corrects is unreachable here.

    A failure is therefore a signal about vernier, not a bug in RF-DETR: the numbers RF-DETR publishes would move on
    the version bump. Widen the allowed-to-differ set deliberately -- never by loosening the comparison.
    """
    predictions, targets = _metric_inputs()
    kwargs: dict[str, Any] = {
        "backend": "vernier",
        "iou_type": iou_type,
        "class_metrics": True,
        "max_detection_thresholds": [1, 10, max_dets],
        "sync_on_compute": False,
    }

    def results(parity_mode: str) -> dict[str, torch.Tensor]:
        with patch("rfdetr.training.coco_map._VERNIER_PARITY_MODE", parity_mode):
            metric = OnePassCocoMeanAveragePrecision(**kwargs)
            metric.update(copy.deepcopy(predictions), copy.deepcopy(targets))
            return metric.compute()

    strict, corrected = results("strict"), results("corrected")

    prefixes = ("",) if isinstance(iou_type, str) else tuple(f"{name}_" for name in iou_type)
    # It only bites on a ladder whose largest entry is not the literal 100 pycocotools' summary reads.
    differing_keys = set() if max_dets == 100 else {f"{prefix}map" for prefix in prefixes}
    assert strict.keys() == corrected.keys()
    assert set(corrected[f"{prefixes[0]}map_per_class"].tolist()) - {-1.0, 0.0, 1.0}, (
        "fixture produced only degenerate per-class values; strengthen it before trusting this parity check"
    )
    for key in corrected:
        if key in differing_keys:
            continue
        torch.testing.assert_close(strict[key], corrected[key], rtol=0, atol=0, equal_nan=True)
    for key in differing_keys:
        assert float(strict[key]) == -1.0
        assert float(corrected[key]) > 0.0
