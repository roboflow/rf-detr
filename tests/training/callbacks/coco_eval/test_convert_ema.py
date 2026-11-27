# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for COCOEvalCallback prediction/target conversion and its EMA-metric collectives."""

from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
from tests._markers import _requires_cuda
from tests.training.callbacks.helpers import (
    _cpu_module,
    _ema_callback,
    _make_pl_module,
    _make_trainer,
    _minimal_metrics,
)


class TestConvertPreds:
    """_convert_preds() normalizes prediction dicts for metric consumers."""

    @pytest.mark.parametrize(
        ("boxes", "expected_kept_idxs"),
        [
            pytest.param(
                # Degenerate first -> keep original index 1 (non-zero keep idx).
                [[2.0, 2.0, 2.0, 4.0], [0.0, 0.0, 3.0, 3.0], [5.0, 5.0, 5.0, 7.0]],
                [1],
                id="degenerate-first-keeps-index-1",
            ),
            pytest.param(
                # Degenerate between valid boxes -> keep non-contiguous original indices.
                [[0.0, 0.0, 3.0, 3.0], [2.0, 2.0, 2.0, 4.0], [4.0, 4.0, 6.0, 6.0]],
                [0, 2],
                id="degenerate-middle-keeps-noncontiguous",
            ),
        ],
    )
    def test_masks_remain_aligned_with_original_indices_after_degenerate_filtering(
        self,
        boxes: list[list[float]],
        expected_kept_idxs: list[int],
    ) -> None:
        """Filtering degenerate boxes must preserve mask alignment via original indices.

        Regression context: when a degenerate box is not last, keep indices are non-zero/non-contiguous. Downstream
        filtering must keep masks from the same original prediction indices.
        """
        cb = COCOEvalCallback()

        # Distinct one-hot masks so index/mask misalignment is easy to detect.
        masks = torch.zeros(3, 1, 2, 2, dtype=torch.bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, 0, 1] = True
        masks[2, 0, 1, 0] = True

        preds = [
            {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                "labels": torch.tensor([0, 0, 0], dtype=torch.int64),
                "masks": masks,
            }
        ]

        out = cb._convert_preds(preds)
        out_boxes = out[0]["boxes"]
        out_masks = out[0]["masks"]
        assert out_masks.shape == (3, 2, 2)

        keep = torch.where((out_boxes[:, 2] > out_boxes[:, 0]) & (out_boxes[:, 3] > out_boxes[:, 1]))[0]
        assert keep.tolist() == expected_kept_idxs
        assert torch.equal(out_masks[keep], masks.squeeze(1)[keep])


class TestConvertTargets:
    """_convert_targets() converts normalised CxCyWH to absolute xyxy."""

    def test_box_conversion_known_values(self) -> None:
        """CxCyWH(0.5,0.5,0.4,0.6) × (W=100,H=200) → xyxy(30,40,70,160)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([200, 100]),  # H=200, W=100
            }
        ]
        out = cb._convert_targets(targets)
        boxes = out[0]["boxes"]
        # cx=0.5*100=50, cy=0.5*200=100, w=0.4*100=40, h=0.6*200=120
        # → x1=50-20=30, y1=100-60=40, x2=50+20=70, y2=100+60=160
        assert boxes[0, 0].item() == pytest.approx(30.0)
        assert boxes[0, 1].item() == pytest.approx(40.0)
        assert boxes[0, 2].item() == pytest.approx(70.0)
        assert boxes[0, 3].item() == pytest.approx(160.0)

    def test_labels_passed_through(self) -> None:
        """Labels tensor is preserved unchanged."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert out[0]["labels"][0].item() == 7

    def test_masks_passed_through_as_bool(self) -> None:
        """Masks tensor is cast to bool and included in output."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([8, 8]),
                "masks": torch.ones(1, 8, 8, dtype=torch.uint8),
            }
        ]
        out = cb._convert_targets(targets)
        assert "masks" in out[0]
        assert out[0]["masks"].dtype == torch.bool

    def test_masks_resize_to_original_grid_without_predictions(self) -> None:
        """Default full-resolution evaluation keeps the original-image mask grid."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.uint8),
            }
        ]

        out = cb._convert_targets(targets)

        assert out[0]["masks"].shape == (1, 5, 5)

    def test_iscrowd_passed_through(self) -> None:
        """Iscrowd tensor is included when present."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
                "iscrowd": torch.tensor([1]),
            }
        ]
        out = cb._convert_targets(targets)
        assert "iscrowd" in out[0]
        assert out[0]["iscrowd"][0].item() == 1

    def test_no_masks_no_iscrowd_keys_absent(self) -> None:
        """Output dict contains exactly boxes and labels when extras are absent."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([100, 100]),
            }
        ]
        out = cb._convert_targets(targets)
        assert set(out[0].keys()) == {"boxes", "labels"}

    @pytest.mark.parametrize(
        ("device", "num_targets", "expected_calls"),
        [
            pytest.param("cpu", 3, [(3, 2)], id="multi-target-batch"),
            pytest.param("cpu", 0, [], id="empty-targets"),
            pytest.param(
                "cuda",
                3,
                [(3, 2)],
                id="multi-target-batch-cuda",
                marks=[
                    pytest.mark.gpu,
                    _requires_cuda,
                ],
            ),
        ],
    )
    def test_orig_size_is_read_once_per_batch(
        self, device: str, num_targets: int, expected_calls: list[tuple[int, ...]], mocker: MockerFixture
    ) -> None:
        """A non-empty target batch reads every orig_size in one device-to-host synchronization, not one per target."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4, device=device),
                "labels": torch.tensor([0], device=device),
                "orig_size": torch.tensor([10, 10], device=device),
            }
            for _ in range(num_targets)
        ]
        calls: list[tuple[int, ...]] = []
        original_tolist = torch.Tensor.tolist

        def tracked_tolist(tensor: torch.Tensor) -> object:
            """Record the tensor shape passed to ``tolist`` before delegating to PyTorch."""
            calls.append(tuple(tensor.shape))
            return original_tolist(tensor)

        mocker.patch.object(torch.Tensor, "tolist", tracked_tolist)
        cb._convert_targets(targets)

        assert calls == expected_calls

    def test_batch_pairs_non_square_orig_sizes_with_their_target_rows(self) -> None:
        """Each target's box uses its own non-square orig_size after batched size conversion.

        This prevents swapped ``orig_size.tolist()`` rows and height/width transposition from assigning a sibling
        image's scale to the wrong target.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.4]]),
                "labels": torch.tensor([3]),
                "orig_size": torch.tensor([100, 300]),  # H=100, W=300
            },
            {
                "boxes": torch.tensor([[0.5, 0.25, 0.4, 0.2]]),
                "labels": torch.tensor([7]),
                "orig_size": torch.tensor([240, 80]),  # H=240, W=80
            },
        ]

        out = cb._convert_targets(targets)

        assert out[0]["boxes"].shape == (1, 4)
        assert out[1]["boxes"].shape == (1, 4)
        assert out[0]["labels"].item() == 3
        assert out[1]["labels"].item() == 7
        torch.testing.assert_close(out[0]["boxes"], torch.tensor([[45.0, 30.0, 105.0, 70.0]]))
        torch.testing.assert_close(out[1]["boxes"], torch.tensor([[24.0, 36.0, 56.0, 84.0]]))


class TestConvertTargetsWithPreds:
    """_convert_targets(targets, preds) resizes masks to the paired prediction's grid, else to orig_size."""

    @pytest.mark.parametrize(
        ("pred_masks_0", "pred_masks_1", "expected_shape_0", "expected_shape_1"),
        [
            pytest.param(
                torch.zeros(1, 4, 4, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (4, 4),
                (3, 3),
                id="both-images-resize-to-distinct-pred-grids",
            ),
            pytest.param(
                torch.zeros(1, 10, 10, dtype=torch.bool),
                torch.zeros(1, 3, 3, dtype=torch.bool),
                (10, 10),
                (3, 3),
                id="first-image-pred-grid-matches-orig-size-second-needs-resize",
            ),
        ],
    )
    def test_batch_resizes_each_target_to_its_own_paired_pred_grid(
        self,
        pred_masks_0: torch.Tensor,
        pred_masks_1: torch.Tensor,
        expected_shape_0: tuple[int, int],
        expected_shape_1: tuple[int, int],
    ) -> None:
        """In a 2-image batch with differing orig_size and pred grids, target[i] resizes to preds[i]'s grid.

        It never resizes to the other image's grid.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.ones(1, 10, 10, dtype=torch.bool),
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"masks": pred_masks_0}, {"masks": pred_masks_1}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == expected_shape_0
        assert out[1]["masks"].shape[-2:] == expected_shape_1

    def test_batch_with_one_image_missing_masks_key_leaves_it_absent(self) -> None:
        """In a mixed batch an image without GT 'masks' stays maskless.

        Its sibling still resizes its masks to its own pred grid.
        """
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([10, 10]),
                # no "masks" key -> detection-only target in an otherwise-segmentation batch.
            },
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "orig_size": torch.tensor([6, 6]),
                "masks": torch.ones(1, 6, 6, dtype=torch.bool),
            },
        ]
        preds = [{"boxes": torch.zeros(1, 4)}, {"masks": torch.zeros(1, 3, 3, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert "masks" not in out[0]
        assert out[1]["masks"].shape[-2:] == (3, 3)

    def test_upsize_via_preds_branch_preserves_extent_and_dtype(self) -> None:
        """GT masks at 2x2 nearest-upsize through the preds-aware branch to a 4x4 pred grid.

        The result stays bool with the same extent.
        """
        cb = COCOEvalCallback()
        gt_masks = torch.zeros(1, 2, 2, dtype=torch.bool)
        gt_masks[0, 0, 0] = True  # top-left cell
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([2, 2]),  # matches GT already -> orig_size alone would be a no-op
                "masks": gt_masks,
            }
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (4, 4)
        assert out[0]["masks"].dtype == torch.bool
        # nearest-neighbor upsize must preserve the filled top-left quadrant's extent.
        assert out[0]["masks"][0, :2, :2].all()
        assert not out[0]["masks"][0, 2:, :].any()
        assert not out[0]["masks"][0, :, 2:].any()

    def test_preds_shorter_than_targets_raises_assertion_error(self) -> None:
        """A preds list shorter than targets fails loudly instead of misaligning images to the wrong prediction."""
        cb = COCOEvalCallback()
        targets = [
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
            {"boxes": torch.zeros(1, 4), "labels": torch.tensor([0]), "orig_size": torch.tensor([10, 10])},
        ]
        preds = [{"masks": torch.zeros(1, 4, 4, dtype=torch.bool)}]

        with pytest.raises(AssertionError):
            cb._convert_targets(targets, preds)

    def test_empty_targets_and_empty_preds_returns_empty_list(self) -> None:
        """An empty batch (no images) through the preds-aware branch is a no-op, not an error."""
        cb = COCOEvalCallback()

        out = cb._convert_targets([], [])

        assert out == []

    def test_pred_entry_without_masks_key_falls_back_to_orig_size(self) -> None:
        """A prediction dict lacking 'masks' does not crash; GT resizes to orig_size as if preds were None."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(1, 4),
                "labels": torch.tensor([0]),
                "orig_size": torch.tensor([5, 5]),
                "masks": torch.ones(1, 4, 4, dtype=torch.bool),
            }
        ]
        preds = [{"boxes": torch.zeros(1, 4)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape[-2:] == (5, 5)

    def test_empty_mask_tensor_through_preds_path_yields_zero_instances_at_pred_grid(self) -> None:
        """K=0 GT masks resize through the preds-aware branch without raising, yielding shape (0, pred_h, pred_w)."""
        cb = COCOEvalCallback()
        targets = [
            {
                "boxes": torch.zeros(0, 4),
                "labels": torch.zeros(0, dtype=torch.long),
                "orig_size": torch.tensor([10, 10]),
                "masks": torch.zeros(0, 10, 10, dtype=torch.bool),
            }
        ]
        preds = [{"masks": torch.zeros(3, 4, 4, dtype=torch.bool)}]

        out = cb._convert_targets(targets, preds)

        assert out[0]["masks"].shape == (0, 4, 4)


class TestEmaCollectiveSymmetry:
    """DDP-deadlock fix: the EMA metric's cross-rank sync must be issued symmetrically (#931/#449)."""

    def test_ema_metric_created_on_val_epoch_start_when_ema_active(self) -> None:
        """map_metric_ema is created on validation start whenever the EMA callback is present.

        This makes the EMA ``compute()`` collective rank-invariant — created on every rank regardless of how many (or
        zero) val batches that rank later processes — rather than lazily per-batch.
        """
        cb = COCOEvalCallback(max_dets=500)
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        assert cb.map_metric_ema is None  # not created yet at setup

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is not None

    def test_ema_metric_not_created_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None (no EMA collective is ever issued)."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    @pytest.mark.parametrize("has_updates", [False, True])
    def test_should_compute_ema_follows_local_updates_without_distributed(self, has_updates: bool) -> None:
        """Without a distributed group the EMA compute proceeds only when the metric saw updates.

        A rank whose EMA metric saw no updates votes against computing, which avoids empty-state divergence.
        """
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = has_updates

        assert cb._should_compute_ema(_cpu_module()) is has_updates

    @pytest.mark.parametrize(
        ("peer_vote", "expected"),
        [
            pytest.param(0, False, id="peer-lacks-ema"),
            pytest.param(1, True, id="all-ranks-have-ema"),
        ],
    )
    def test_unanimous_gate_follows_reduced_vote(self, all_reduce, peer_vote: int, expected: bool) -> None:
        """With local updates, the gate returns True only if the all_reduce MIN leaves the vote at 1."""
        all_reduce.side_effect = lambda flag, op=None: flag.fill_(peer_vote)  # the reduced vote across ranks
        cb = COCOEvalCallback()
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb._ema_has_updates = True

        assert cb._should_compute_ema(_cpu_module()) is expected
        all_reduce.assert_called_once()


class TestMapCollectiveSymmetry:
    """DDP-deadlock fix: the base/train mAP path's has_updates gate must be a collective vote.

    Unlike ``_should_compute_ema``'s unanimous ``all_reduce(MIN)`` (skip whenever any rank is empty), this path votes
    ``all_reduce(MAX)``: a rank with no local updates must still enter ``merge_distributed_state()`` when a peer has
    data, or the ranks diverge on which collectives they issue next and deadlock validation.
    """

    @pytest.mark.parametrize("has_updates", [False, True])
    def test_any_rank_has_updates_follows_local_updates_without_distributed(self, has_updates: bool) -> None:
        """With no distributed group the vote equals whether this rank has local updates."""
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = has_updates

        assert cb._any_rank_has_updates(metric, _cpu_module()) is has_updates

    @pytest.mark.parametrize(
        ("peer_vote", "expected"),
        [
            pytest.param(1, True, id="peer-has-updates"),
            pytest.param(0, False, id="no-rank-has-updates"),
        ],
    )
    def test_max_vote_follows_reduced_vote(self, all_reduce, peer_vote: int, expected: bool) -> None:
        """A locally-empty rank votes True only when the all_reduce MAX surfaces a peer's update."""
        all_reduce.side_effect = lambda flag, op=None: flag.fill_(peer_vote)  # the reduced vote across ranks
        cb = COCOEvalCallback()
        metric = MagicMock(name="map_metric")
        metric.has_updates = False

        assert cb._any_rank_has_updates(metric, _cpu_module()) is expected
        assert all_reduce.call_args.kwargs["op"] is torch.distributed.ReduceOp.MAX

    def test_compute_and_log_enters_merge_when_only_a_peer_has_updates(self, all_reduce) -> None:
        """A locally-empty rank still calls merge_distributed_state(), staying symmetric with its peer.

        Reproduces the deadlock scenario directly at the callback level: rank 0 has no local updates but a peer does.
        Before the fix, this rank took the early-return path and never issued ``merge_distributed_state()``'s
        collectives while its peer did, desynchronising the DDP collective sequence. After the fix, the collective
        vote makes both ranks agree to merge.
        """

        def _peer_has_map_updates_only(flag, op=None):
            # Peer has base mAP data (MAX vote -> 1) but no EMA data (MIN vote stays 0): isolates the has_updates gate
            # from the unrelated EMA collective this path also issues for split="val".
            if op is torch.distributed.ReduceOp.MAX:
                flag.fill_(1)

        all_reduce.side_effect = _peer_has_map_updates_only
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _cpu_module(), stage="fit")
        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.has_updates = False
        cb.map_metric.compute.return_value = _minimal_metrics()

        cb._compute_and_log(_make_trainer(), _cpu_module(), "val")

        cb.map_metric.merge_distributed_state.assert_called_once()


class TestOnTestEpochStart:
    """on_test_epoch_start resets _ema_has_updates before test to prevent stale val state."""

    def test_map_metric_ema_stays_none_without_ema_callback(self) -> None:
        """No EMA callback → map_metric_ema stays None after test hook fires."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_test_epoch_start(trainer, module)

        assert cb.map_metric_ema is None

    def test_resets_ema_has_updates_to_false(self) -> None:
        """on_test_epoch_start resets _ema_has_updates to False even when stale True from validation."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        cb._ema_has_updates = True  # simulate stale value from a preceding validation epoch

        cb.on_test_epoch_start(trainer, module)

        assert cb._ema_has_updates is False


class TestPrepareEmaMetricSecondEpoch:
    """_prepare_ema_metric resets (not re-creates) the metric on subsequent epochs."""

    def test_resets_not_recreates_metric(self) -> None:
        """Calling on_validation_epoch_start twice resets the metric rather than replacing it."""
        cb = COCOEvalCallback()
        trainer = _make_trainer(callbacks=[_ema_callback()])
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")

        cb.on_validation_epoch_start(trainer, module)
        assert cb.map_metric_ema is not None

        # Replace with a spy mock so reset() calls are trackable on the second epoch
        spy_metric = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = spy_metric

        cb.on_validation_epoch_start(trainer, module)

        assert cb.map_metric_ema is spy_metric  # same object, not replaced
        spy_metric.reset.assert_called_once()


class TestComputeAndLogEmaResetPath:
    """Elif branch in _compute_and_log: gate False + metric not None → reset() fires."""

    def test_resets_ema_metric(self, mocker: MockerFixture) -> None:
        """EMA not computed this epoch but metric exists → reset() clears state for the next epoch."""
        cb = COCOEvalCallback()
        trainer = _make_trainer()
        module = _cpu_module()
        cb.setup(trainer, module, stage="fit")
        trainer.callback_metrics = {}

        # EMA metric exists but no batch updated it this epoch → gate returns False → elif fires
        mock_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema = mock_ema
        cb._ema_has_updates = False

        cb.map_metric = MagicMock(name="map_metric")
        cb.map_metric.compute.return_value = _minimal_metrics()
        cb.map_metric.has_updates = True

        mocker.patch.object(cb, "_build_per_class_rows", return_value=[])
        mocker.patch.object(cb, "_print_metrics_tables")
        mocker.patch("rfdetr.training.callbacks.coco_eval.distributed_merge_matching_data", return_value={})
        cb._compute_and_log(trainer, module, "val")

        mock_ema.reset.assert_called_once()

    def test_clears_ema_has_updates_after_gate_true_compute(self) -> None:
        """Regression for #1289 review: after a gate-True EMA compute, `_ema_has_updates` must be cleared to False.

        Before this fix, `_compute_and_log_ema_metrics` reset `map_metric_ema` in both branches but never cleared
        `_ema_has_updates`, so a stale `True` left over from a validation epoch could reach a later
        `_compute_and_log(..., "train", ...)` call (when `compute_train_metrics=True`) and drive a spurious `compute()`
        on the freshly emptied EMA metric.
        """
        cb = COCOEvalCallback(max_dets=500)
        cb.setup(_make_trainer(), _make_pl_module(), stage="fit")
        cb.map_metric_ema = MagicMock(name="map_metric_ema")
        cb.map_metric_ema.compute.return_value = _minimal_metrics()
        cb._ema_has_updates = True

        should_compute_ema, _ema_metrics = cb._compute_and_log_ema_metrics(
            _make_trainer(), _make_pl_module(), "val", "", "mar_500"
        )

        assert should_compute_ema is True
        assert cb._ema_has_updates is False
