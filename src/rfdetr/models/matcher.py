# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
"""Modules to compute the matching cost and solve the corresponding LSAP."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment as _linear_sum_assignment  # type: ignore[import-untyped,unused-ignore]
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from rfdetr.models import _assignment
from rfdetr.models.heads.keypoints import compute_keypoint_matching_cost
from rfdetr.models.heads.segmentation import point_sample
from rfdetr.utilities.box_ops import batch_dice_loss, batch_sigmoid_ce_loss, box_cxcywh_to_xyxy, generalized_box_iou
from rfdetr.utilities.logger import get_logger

logger = get_logger()
_SANITIZED_COST_MARGIN = 1.0
_FOCAL_LOSS_GAMMA = 2.0
#: Per-layer padded cost-matrix element budget (``batch * queries * max_targets``) under which
#: ``_match_many`` folds all layers into one stacked cost-construction pass. Calibrated on an
#: NVIDIA L4 (plan-native-linear-assignment.md, M2 results): stacking wins 1.45-5.8x up to ~312K
#: elements but regresses to 0.79-1.03x from ~468K, so the budget sits between those points.
_STACKED_COST_ELEMENT_LIMIT = 350_000
#: Minimum ``len(sizes) * group_detr`` problem count under which ``forward``'s single-layer compact
#: path solves on the host instead of handing the device-resident matrix to the batched solver.
#: Calibrated to the SciPy/GPU crossover measured in the M2 spike (~50 problems on an NVIDIA L4;
#: plan-native-linear-assignment.md, M2 results) — a single layer alone (unlike ``_match_many``,
#: which buckets every layer of a step together) is often below that crossover.
_BATCHED_ASSIGNMENT_MIN_PROBLEMS = 50
_LinearSumAssignment = Callable[[Any], tuple[NDArray[np.int64], NDArray[np.int64]]]
linear_sum_assignment = cast(_LinearSumAssignment, _linear_sum_assignment)


class _TargetSideSafety(NamedTuple):
    """A precomputed compact-path target-side safety result from
    :meth:`HungarianMatcher._precompute_target_side_safety`.

    Tied to the ``targets`` list object (by identity, not equality — the ``targets`` this reflects
    can only be the exact object it was computed from; a different batch that happens to build an
    equal-looking list is not the same targets) and to the ``pred_boxes`` dtype/device and
    ``num_classes`` it was computed against, so :meth:`HungarianMatcher._detection_inputs_are_safe`
    can verify those still match the current call before reusing ``safe``, falling back to a fresh
    computation otherwise. Object identity distinguishes one step's targets list from another's —
    dtype/device/``num_classes`` stay constant for an entire training run, so checking only those
    would let a value computed for one step's targets be silently reused for a different step's —
    but it cannot detect in-place mutation of the same list's contents between precompute and use.
    Callers must not mutate ``targets`` after precomputing against it.

    Attributes:
        safe: Unsynced 0-d bool Tensor — whether the target-side safety sweep passed.
        targets: The exact targets list object this was computed from (identity-checked on reuse).
        pred_boxes_dtype: dtype of the ``pred_boxes`` this was computed against.
        pred_boxes_device: Device of the ``pred_boxes`` this was computed against.
        num_classes: Number of classes this was computed against.
    """

    safe: Tensor
    targets: list[dict[str, Any]]
    pred_boxes_dtype: torch.dtype
    pred_boxes_device: torch.device
    num_classes: int


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network For efficiency reasons,
    the targets don't include the no_object.

    Because of this, in general, there are more predictions than targets. In this case, we do a 1-to-1 matching of the
    best predictions, while the others are un-matched (and thus treated as non-objects).

    Note:
        The focal loss exponent ``gamma`` is fixed at ``_FOCAL_LOSS_GAMMA`` (2.0) and is not
        configurable. Only ``focal_alpha`` can be adjusted at construction time.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        use_pos_only: bool = False,  # reserved for future use; not yet implemented
        use_position_modulated_cost: bool = False,  # reserved for future use; not yet implemented
        mask_point_sample_ratio: int = 16,
        cost_mask_ce: float = 1,
        cost_mask_dice: float = 1,
        num_keypoints_per_class: list[int] | None = None,
        keypoint_l1_loss_coef: float = 0.0,
        keypoint_findable_loss_coef: float = 0.0,
        keypoint_visible_loss_coef: float = 0.0,
        keypoint_nll_loss_coef: float = 0.0,
    ):
        """Creates the matcher.

        Args:
            cost_class: Relative weight of the classification error in the matching cost.
            cost_bbox: Relative weight of the L1 error of the bounding box coordinates.
            cost_giou: Relative weight of the GIoU loss of the bounding box.
            focal_alpha: Alpha parameter for focal loss used in the classification cost.
            use_pos_only: Reserved for future use; currently has no effect.
            use_position_modulated_cost: Reserved for future use; currently has no effect.
            mask_point_sample_ratio: Downsampling ratio for mask point sampling.
            cost_mask_ce: Relative weight of the binary cross-entropy mask cost.
            cost_mask_dice: Relative weight of the Dice mask cost.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.keypoint_l1_loss_coef = keypoint_l1_loss_coef
        self.keypoint_findable_loss_coef = keypoint_findable_loss_coef
        self.keypoint_visible_loss_coef = keypoint_visible_loss_coef
        self.keypoint_nll_loss_coef = keypoint_nll_loss_coef
        self._warned_non_finite_costs = False

    @staticmethod
    def _sanitize_cost_matrix(cost_matrix: Tensor) -> Tensor:
        """Replace non-finite cost entries with a large finite sentinel.

        >>> HungarianMatcher._sanitize_cost_matrix(
        ...     torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
        ... ).tolist()
        [[1.0, 4.0], [4.0, -2.0]]

        Args:
            cost_matrix: Cost matrix to sanitize before Hungarian assignment.

        Returns:
            Cost matrix with all non-finite entries replaced by a finite sentinel that is no smaller than any valid
            entry.
        """
        finite_mask = torch.isfinite(cost_matrix)
        if finite_mask.all():
            return cost_matrix

        dtype_info = torch.finfo(cost_matrix.dtype)
        if finite_mask.any():
            finite_costs = cost_matrix[finite_mask]
            max_cost = finite_costs.max()
            # Add the largest absolute finite cost so the replacement stays
            # strictly larger than every valid entry, even if all costs are negative.
            replacement_cost = max_cost + finite_costs.abs().max() + _SANITIZED_COST_MARGIN
            # Guard against overflow to inf/NaN and clamp to the maximum finite value.
            if not torch.isfinite(replacement_cost):
                replacement_cost = cost_matrix.new_tensor(dtype_info.max)
            else:
                replacement_cost = torch.clamp(replacement_cost, max=dtype_info.max)
        else:
            # If all entries are non-finite, fall back to a large finite sentinel.
            replacement_cost = cost_matrix.new_tensor(dtype_info.max)

        sanitized_cost_matrix = cost_matrix.clone()
        sanitized_cost_matrix[~finite_mask] = replacement_cost
        return sanitized_cost_matrix

    def _focal_classification_cost(self, logits: Tensor) -> Tensor:
        """Focal-loss classification cost for logits already gathered at the target classes they are scored against.

        Every operation is elementwise, so the leading dimensions are free: the compact path passes
        ``[bs, num_queries, max_targets]`` and the full path ``[bs * num_queries, total_targets]``.

        >>> HungarianMatcher(focal_alpha=0.25)._focal_classification_cost(torch.zeros(1, 2)).tolist()
        [[-0.08664339780807495, -0.08664339780807495]]

        Args:
            logits: Classification logits already selected at their target classes.

        Returns:
            Focal classification cost, same shape as ``logits``.
        """
        # neg_cost_class = (1 - alpha) * (tgt_prob ** gamma) * (-(1 - tgt_prob + 1e-8).log())
        # pos_cost_class = alpha * ((1 - tgt_prob) ** gamma) * (-(tgt_prob + 1e-8).log())
        # we refactor these with logsigmoid for numerical stability
        alpha = self.focal_alpha
        gamma = _FOCAL_LOSS_GAMMA
        probabilities = logits.sigmoid()
        negative_cost = (1 - alpha) * (probabilities**gamma) * (-F.logsigmoid(-logits))
        positive_cost = alpha * ((1 - probabilities) ** gamma) * (-F.logsigmoid(logits))
        cost: Tensor = positive_cost - negative_cost
        return cost

    @staticmethod
    def _target_side_precheck(
        pred_boxes_dtype: torch.dtype,
        pred_boxes_device: torch.device,
        num_classes: int,
        targets: list[dict[str, Any]],
    ) -> Tensor:
        """Return whether the *target*-side half of the compact-path safety gate passes.

        Split out of :meth:`_detection_inputs_are_safe` because this half depends only on
        ``targets`` plus ``pred_boxes_dtype``/``pred_boxes_device``/``num_classes`` — all identical
        across every one of the (final layer + aux layers + enc layer) calls
        ``SetCriterion.forward`` makes with the same ``targets`` in one training step — so
        :meth:`_precompute_target_side_safety` can compute it once per step instead of once per call.

        Args:
            pred_boxes_dtype: dtype of the predictions' box tensor.
            pred_boxes_device: Device of the predictions' box tensor.
            num_classes: Number of classes the predictions' logits cover.
            targets: Per-image target dicts containing ``boxes`` and ``labels``.

        Returns:
            An *unsynced* 0-d bool Tensor, false if any target's box dtype or device disagrees with
            the predictions', if any target's label device disagrees with the predictions', if any
            target box coordinate is non-finite or exceeds ``coordinate_limit``, or if any label
            falls outside ``[0, num_classes)``; true otherwise. Deliberately left as a Tensor rather
            than cast to ``bool`` so :meth:`_detection_inputs_are_safe` can fuse it with its
            pred-side check into a single host-device sync.
        """
        # Metadata-only prechecks: attribute comparisons that launch no kernel and force no device
        # sync, and that short-circuit before any reduction below runs. `pad_sequence` allocates
        # from the first sequence, so a target whose boxes disagree in dtype with the rest is
        # silently cast into the padded tensor and matched at whatever precision the batch ordering
        # happens to produce, where the full path's `torch.cat` promotes and then fails loudly.
        # Route those batches to the full path so that loud failure is preserved.
        target_label_dtype = targets[0]["labels"].dtype
        for target in targets:
            if target["boxes"].dtype != pred_boxes_dtype or target["boxes"].device != pred_boxes_device:
                return torch.tensor(False, device=pred_boxes_device)
            if target["labels"].device != pred_boxes_device or target["labels"].dtype != target_label_dtype:
                return torch.tensor(False, device=pred_boxes_device)

        # Concatenates the per-image tensors into one before checking them, instead of looping
        # `isfinite`/`abs`/comparison + `.all()` once per image: each per-image call launches
        # several tiny kernels, so the loop's cost scales with `bs`. The metadata prechecks above
        # guarantee every target's boxes share `pred_boxes_dtype` and every target's labels share
        # dtype/device, so the concatenation cannot fail on metadata mismatches. Concatenation
        # preserves which individual values are non-finite/out-of-range — it does not change what
        # this check returns, only how many kernels it launches.
        checks: list[Tensor] = []
        # Keep enough headroom for cxcywh conversion, pairwise differences, areas, and unions.
        coordinate_limit = torch.finfo(pred_boxes_dtype).max ** 0.5 / 16
        target_boxes = torch.cat([target["boxes"] for target in targets])
        checks.append(torch.isfinite(target_boxes).all() & (target_boxes.abs() <= coordinate_limit).all())
        # `torch.gather` rejects an out-of-range class index outright, where the full path's
        # `flat_pred_logits[:, tgt_ids]` wraps a negative label Python-style onto the last class and
        # raises `IndexError` for a too-large one. Deliberately keep that pre-existing behavior —
        # silent wrap included — by routing such batches to the full path rather than introducing a
        # new hard error here. Appended to the same `checks` list so it rides the single
        # `torch.stack` sync below instead of adding a second one.
        target_labels = torch.cat([target["labels"] for target in targets])
        checks.append((target_labels >= 0).all() & (target_labels < num_classes).all())
        return torch.stack(checks).all()

    @staticmethod
    def _compact_path_applicable(outputs: dict[str, Any], targets: list[dict[str, Any]]) -> bool:
        """Return whether the compact path could apply at all, before the input-safety gate is consulted.

        Sole owner of the compact path's routing precondition, so callers that want to skip work the
        compact path would never use (``SetCriterion.forward`` precomputing the target-side safety
        sweep) test the same rule :meth:`forward` routes on instead of reimplementing it.

        >>> outputs = {"pred_logits": torch.zeros(2, 3, 4)}
        >>> HungarianMatcher._compact_path_applicable(outputs, [{}, {}])
        True
        >>> HungarianMatcher._compact_path_applicable({"pred_logits": torch.zeros(1, 3, 4)}, [{}])
        False

        Args:
            outputs: Model outputs containing at least ``pred_logits``.
            targets: Per-image target dicts.

        Returns:
            ``True`` only for a batch of more than one image with neither mask nor keypoint targets;
            every other batch is matched by the full cartesian path regardless of input safety.
        """
        return (
            outputs["pred_logits"].shape[0] > 1
            and "masks" not in targets[0]
            and not ("pred_keypoints" in outputs and "keypoints" in targets[0])
        )

    @torch.no_grad()
    def _precompute_target_side_safety(
        self, outputs: dict[str, Any], targets: list[dict[str, Any]]
    ) -> _TargetSideSafety | None:
        """Precompute the compact-path safety gate's target-side sweep once, for reuse across the several
        :meth:`forward` calls ``SetCriterion.forward`` makes with the same ``targets`` (once per decoder layer, plus
        aux/enc outputs). Recomputing the target-side sweep on every one of those calls wastes ~47% of the gate's total
        cost on work that never changes within a step (measured on an RTX 4060: ~300-700us/step saved by caching it).

        Examples:
            >>> matcher = HungarianMatcher()
            >>> outputs = {
            ...     "pred_logits": torch.zeros(2, 3, 4),
            ...     "pred_boxes": torch.rand(2, 3, 4),
            ... }
            >>> targets = [
            ...     {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
            ...     {"boxes": torch.rand(1, 4), "labels": torch.tensor([1])},
            ... ]
            >>> result = matcher._precompute_target_side_safety(outputs, targets)
            >>> bool(result.safe)  # .safe is an unsynced Tensor, so compare via bool() to sync it
            True

        Args:
            outputs: Any one of the step's outputs dicts (``pred_boxes``/``pred_logits`` share the
                same dtype/device/``num_classes`` across every layer of one forward pass, by
                construction); typically the final-layer output.
            targets: The step's targets, unchanged across every :meth:`forward` call it feeds.

        Returns:
            ``None`` when :meth:`_compact_path_applicable` rules the compact path out for this batch
            entirely, since :meth:`forward` then never reaches the safety gate and sweeping the
            targets would be pure overhead. Otherwise a :class:`_TargetSideSafety` to pass as
            :meth:`forward`'s ``target_side_safety`` argument. :meth:`forward` verifies the exact
            ``targets`` object (by identity) plus the dtype/device/``num_classes`` it was computed
            against still match the current call before trusting it, falling back to a fresh
            computation otherwise — reusing it never changes what :meth:`forward` returns, only how
            much redundant work it does to get there.
        """
        if not self._compact_path_applicable(outputs, targets):
            return None
        pred_boxes = outputs["pred_boxes"]
        num_classes = outputs["pred_logits"].shape[-1]
        safe = self._target_side_precheck(pred_boxes.dtype, pred_boxes.device, num_classes, targets)
        return _TargetSideSafety(safe, targets, pred_boxes.dtype, pred_boxes.device, num_classes)

    @staticmethod
    def _detection_inputs_are_safe(
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        target_side_safety: _TargetSideSafety | None = None,
    ) -> bool:
        """Return whether the compact path can preserve the full path's finite-cost behavior.

        Args:
            outputs: Model outputs containing ``pred_boxes`` and ``pred_logits``.
            targets: Per-image target dicts containing ``boxes`` and ``labels``.
            target_side_safety: An optional precomputed result from
                :meth:`_precompute_target_side_safety`, reused instead of recomputing the
                target-side sweep from scratch when it was computed from this exact ``targets``
                object (by identity) and its dtype/device/``num_classes`` still match this call's
                ``outputs``.

        Returns:
            ``False`` if any target's box dtype or device disagrees with the predictions', if any
            target's label device disagrees with the predictions', if any box coordinate (predicted or
            target) is non-finite or exceeds ``coordinate_limit``, or if any label falls outside
            ``[0, num_classes)``; ``True`` otherwise. Any ``False`` routes the batch to the full path.
            Both halves stay unsynced Tensors until one final ``bool()`` below, so the whole gate
            costs exactly one host-device sync per call however the target-side half was obtained.
        """
        pred_boxes = outputs["pred_boxes"]
        num_classes = outputs["pred_logits"].shape[-1]
        if (
            target_side_safety is not None
            and target_side_safety.targets is targets
            and target_side_safety.pred_boxes_dtype == pred_boxes.dtype
            and target_side_safety.pred_boxes_device == pred_boxes.device
            and target_side_safety.num_classes == num_classes
        ):
            target_side_safe = target_side_safety.safe
        else:
            target_side_safe = HungarianMatcher._target_side_precheck(
                pred_boxes.dtype, pred_boxes.device, num_classes, targets
            )

        # `pred_logits` is deliberately not swept here. A non-finite logit either lands in a class
        # column the compact matrix consumes — where it survives every coefficient (including
        # `cost_class == 0`, since `0 * inf` is NaN) into the weighted sum, so the post-hoc
        # `torch.isfinite(compact_cost_matrix)` check in `forward` sees it and falls through — or in
        # a column neither path materializes into its extracted diagonal blocks, where it cannot
        # change the assignment. Sweeping all of `[bs, num_queries, num_classes]` up front costs
        # more than building the compact matrix it was guarding.
        #
        # This sweep is per-call, unlike the target-side one above: `pred_boxes` differs by layer,
        # so it cannot be cached across the calls `SetCriterion.forward` makes with the same
        # `targets`.
        coordinate_limit = torch.finfo(pred_boxes.dtype).max ** 0.5 / 16
        pred_side_safe = torch.isfinite(pred_boxes).all() & (pred_boxes.abs() <= coordinate_limit).all()
        # The single sync for the whole gate: both halves are 0-d bool Tensors on `pred_boxes`'
        # device, so stacking them costs one launch and one `bool()` transfer. An early
        # `if not target_side_safe` above would have forced its own separate sync, which is what
        # made a cached multi-call step pay N+1 syncs where recomputing per call paid N.
        return bool(torch.stack([target_side_safe, pred_side_safe]).all())

    def _compute_compact_detection_cost_matrix(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        *,
        clamp_target_labels: bool = False,
    ) -> Tensor:
        """Compute same-image detection costs with targets padded only to the batch maximum.

        Args:
            outputs: Model outputs containing ``pred_boxes`` and ``pred_logits``.
            targets: Per-image target dicts containing ``boxes`` and ``labels``.
            clamp_target_labels: If True, clamp target label ids into ``[0, num_classes)`` before
                the gather so an out-of-range label cannot raise; used only by the batched fast
                path, which defers the safety decision until after the single host transfer (see
                the ``clamp_target_labels`` block below).

        Returns:
            Cost matrix of shape ``[num_queries, sum(sizes)]``, where ``sizes`` is each image's real
            (unpadded) target count. The class, bbox, and GIoU terms are computed on tensors padded to
            ``max(sizes)`` for the gather/cdist/GIoU ops, but the padding columns are dropped per-image
            (``[:, :size]``) before the per-image slices are concatenated along ``dim=-1`` — the
            returned tensor never carries the padded ``max(sizes)`` width or a leading batch dimension.
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        sizes = [len(target["boxes"]) for target in targets]
        padded_target_ids = pad_sequence([target["labels"] for target in targets], batch_first=True)
        padded_target_boxes = pad_sequence([target["boxes"] for target in targets], batch_first=True)
        max_targets = padded_target_ids.shape[1]

        if clamp_target_labels:
            # The batched path checks label safety only after its single host transfer. Clamp solely
            # to let `gather` construct a disposable matrix for unsafe batches; it is never assigned.
            # This is a no-op on every batch the method serves: `pad_sequence` pads with in-range 0,
            # `_target_side_precheck` gates labels to `[0, num_classes)`, and `_match_many` checks
            # equal class counts across layers; weakening any invariant would silently match wrong
            # class columns here instead of raising.
            padded_target_ids = padded_target_ids.clamp(0, outputs["pred_logits"].shape[-1] - 1)

        gather_index = padded_target_ids[:, None, :].expand(batch_size, num_queries, max_targets)
        target_logits = torch.gather(outputs["pred_logits"], 2, gather_index)
        class_cost = self._focal_classification_cost(target_logits)

        bbox_cost = torch.cdist(outputs["pred_boxes"], padded_target_boxes, p=1)
        giou_cost = -torch.vmap(generalized_box_iou)(
            box_cxcywh_to_xyxy(outputs["pred_boxes"]),
            box_cxcywh_to_xyxy(padded_target_boxes),
        )
        padded_cost_matrix = self.cost_bbox * bbox_cost + self.cost_class * class_cost + self.cost_giou * giou_cost
        return torch.cat(
            [padded_cost_matrix[index, :, :size] for index, size in enumerate(sizes)],
            dim=-1,
        )

    def _compute_stacked_compact_cost_matrices(
        self, outputs_list: list[dict[str, Any]], targets: list[dict[str, Any]]
    ) -> list[Tensor]:
        """Compute every layer's compact detection cost matrix in one stacked pass.

        Folds the layer dimension into the batch dimension and reuses
        :meth:`_compute_compact_detection_cost_matrix` on the concatenation, so target padding, the
        class gather, ``cdist``, and the vmapped GIoU each launch once for all layers instead of
        once per layer. The concatenated call sees ``len(outputs_list) * len(targets)`` images
        whose per-image target counts repeat layer-major, so its output columns are each layer's
        compact matrix laid side by side — slicing them apart reproduces the per-layer results
        bitwise; only the kernel-launch count changes.

        Callers must guarantee every layer shares the prediction query count (dtype, device, and
        class count are already enforced by :meth:`_match_many`). Target labels are clamped exactly
        as :meth:`_match_many`'s per-layer calls do.

        Examples:
            >>> matcher = HungarianMatcher()
            >>> outputs = {"pred_logits": torch.zeros(2, 4, 5), "pred_boxes": torch.rand(2, 4, 4)}
            >>> targets = [
            ...     {"boxes": torch.rand(1, 4), "labels": torch.tensor([0])},
            ...     {"boxes": torch.rand(2, 4), "labels": torch.tensor([1, 2])},
            ... ]
            >>> [m.shape for m in matcher._compute_stacked_compact_cost_matrices([outputs, outputs], targets)]
            [torch.Size([4, 3]), torch.Size([4, 3])]

        Args:
            outputs_list: Per-layer detection outputs with identical ``pred_logits``/``pred_boxes``
                shapes.
            targets: Per-image target dicts shared by every layer.

        Returns:
            One ``[num_queries, sum(sizes)]`` cost matrix per layer, in ``outputs_list`` order,
            each a column slice of the single stacked computation.
        """
        stacked_outputs = {
            "pred_logits": torch.cat([outputs["pred_logits"] for outputs in outputs_list]),
            "pred_boxes": torch.cat([outputs["pred_boxes"] for outputs in outputs_list]),
        }
        stacked_matrix = self._compute_compact_detection_cost_matrix(
            stacked_outputs, targets * len(outputs_list), clamp_target_labels=True
        )
        total_targets = sum(len(target["boxes"]) for target in targets)
        return [
            stacked_matrix[:, offset : offset + total_targets]
            for offset in range(0, stacked_matrix.shape[1], total_targets)
        ]

    @torch.no_grad()
    def _match_many(
        self,
        outputs_list: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        group_detr: int = 1,
        target_side_safety: _TargetSideSafety | None = None,
    ) -> list[list[tuple[Tensor, Tensor]]] | None:
        """Solve every compatible detection layer in one batched pass, or decline to preserve fallback behavior.

        This private fast path is used only by ``SetCriterion`` for final, auxiliary, and encoder outputs from one
        training step. It leaves unusual input shapes and unsafe values to :meth:`forward`, whose full-cartesian
        fallback carries the established error and non-finite-cost behavior.

        Every layer's cost matrix is built and left on its own device, then all of them are handed to
        :func:`~rfdetr.models._assignment.assign_many_bucketed`, which groups same-shaped problems across layers so the
        solver sees far more work per call than any single layer could supply.

        Args:
            outputs_list: Detection outputs for the final, auxiliary, and encoder layers to match
                together.
            targets: Per-image target dicts shared by every layer in ``outputs_list``.
            group_detr: Number of query groups to solve independently within each layer.
            target_side_safety: Optional precomputed target-side safety result for these exact
                ``targets`` and the reference layer's box dtype, device, and class count.

        Returns:
            ``list[list[tuple[Tensor, Tensor]]] | None`` containing per-image assignments for each
            layer, or ``None``. ``None`` is the decline contract: the caller MUST execute the
            per-layer fallback. A decline is all-or-nothing and can only be detected after every
            layer's cost matrix has been built, since the safety predicates are reduced together to
            keep the whole batch down to one synchronization — so a single layer with non-finite
            ``pred_boxes`` or a failed safety flag discards the work done for all of them and pays
            for the sequential redo on top.
        """
        if len(outputs_list) < 2 or not all(
            self._compact_path_applicable(outputs, targets) for outputs in outputs_list
        ):
            return None

        reference_boxes = outputs_list[0]["pred_boxes"]
        reference_classes = outputs_list[0]["pred_logits"].shape[-1]
        if any(
            outputs["pred_boxes"].dtype != reference_boxes.dtype
            or outputs["pred_boxes"].device != reference_boxes.device
            or outputs["pred_logits"].device != reference_boxes.device
            or outputs["pred_logits"].shape[-1] != reference_classes
            for outputs in outputs_list[1:]
        ):
            return None
        if any(
            target["boxes"].dtype != reference_boxes.dtype
            or target["boxes"].device != reference_boxes.device
            or target["labels"].device != reference_boxes.device
            or target["labels"].dtype != targets[0]["labels"].dtype
            for target in targets
        ):
            return None

        if (
            target_side_safety is not None
            and target_side_safety.targets is targets
            and target_side_safety.pred_boxes_dtype == reference_boxes.dtype
            and target_side_safety.pred_boxes_device == reference_boxes.device
            and target_side_safety.num_classes == reference_classes
        ):
            target_safe = target_side_safety.safe
        else:
            target_safe = self._target_side_precheck(
                reference_boxes.dtype, reference_boxes.device, reference_classes, targets
            )

        sizes = [len(target["boxes"]) for target in targets]
        total_targets = sum(sizes)
        if total_targets == 0:
            return None

        coordinate_limit = torch.finfo(reference_boxes.dtype).max ** 0.5 / 16
        # Stacked cost construction holds every layer's padded matrix at once, so it is gated to
        # small padded shapes where the L4 calibration measured launch-bound wins; dense
        # compute-bound batches measured 0.79-1.03x stacked and keep the per-layer loop. Layers with
        # differing query counts cannot fold into one batch dimension and also keep the loop.
        query_counts = {outputs["pred_logits"].shape[1] for outputs in outputs_list}
        stacked_cost_matrices = None
        if (
            len(query_counts) == 1
            and len(targets) * next(iter(query_counts)) * max(sizes) <= _STACKED_COST_ELEMENT_LIMIT
        ):
            stacked_cost_matrices = self._compute_stacked_compact_cost_matrices(outputs_list, targets)

        cost_matrices: list[Tensor] = []
        # Every layer's safety predicate stays an unsynced 0-d tensor and is reduced together with
        # every other layer's below, so the whole matcher batch still costs exactly one
        # synchronization however many layers it holds — the property the previous pinned-buffer
        # design carried by appending a safety row to each layer's transferred payload.
        safety_flags: list[Tensor] = [target_safe]
        for layer_index, outputs in enumerate(outputs_list):
            pred_boxes = outputs["pred_boxes"]
            safety_flags.append(torch.isfinite(pred_boxes).all() & (pred_boxes.abs() <= coordinate_limit).all())
            if stacked_cost_matrices is None:
                cost_matrix = self._compute_compact_detection_cost_matrix(
                    outputs, targets, clamp_target_labels=True
                ).float()
            else:
                cost_matrix = stacked_cost_matrices[layer_index].float()
            cost_matrices.append(cost_matrix)
            safety_flags.append(torch.isfinite(cost_matrix).all())

        # The single synchronization for the whole batch.
        if not bool(torch.stack(safety_flags).all()):
            return None
        return _assignment.assign_many_bucketed(cost_matrices, sizes, group_detr)

    @staticmethod
    def _assign_compact_cost_matrix(
        cost_matrix: Tensor,
        sizes: list[int],
        group_detr: int,
    ) -> list[tuple[Tensor, Tensor]]:
        """Solve a compact ``[queries, total_targets]`` matrix by group and image.

        Shared by both the compact and the full-cartesian ``forward`` paths — "compact" in the name
        refers to the padded-to-``max(sizes)`` matrix layout this helper solves, not to which path
        calls it.

        Args:
            cost_matrix: Compact cost matrix of shape ``[num_queries, sum(sizes)]``, as returned by
                ``_compute_compact_detection_cost_matrix``.
            sizes: Each image's real (unpadded) target count, in batch order.
            group_detr: Number of query groups; ``num_queries`` must be evenly divisible by it.

        Returns:
            One ``(row_indices, col_indices)`` index pair per image, in batch order, with each group's
            assignment concatenated onto the same image's pair from every other group.
        """
        target_offsets = [0]
        for size in sizes:
            target_offsets.append(target_offsets[-1] + size)

        num_queries = cost_matrix.shape[0]
        if num_queries % group_detr != 0:
            raise ValueError(f"num_queries ({num_queries}) must be divisible by group_detr ({group_detr})")
        group_num_queries = num_queries // group_detr
        indices = []
        for group_index in range(group_detr):
            group_start = group_index * group_num_queries
            grouped_cost_matrix = cost_matrix[group_start : group_start + group_num_queries]
            group_indices = [
                linear_sum_assignment(grouped_cost_matrix[:, target_offsets[index] : target_offsets[index + 1]])
                for index in range(len(sizes))
            ]
            if group_index == 0:
                indices = group_indices
            else:
                indices = [
                    (
                        np.concatenate([previous[0], current[0] + group_num_queries * group_index]),
                        np.concatenate([previous[1], current[1]]),
                    )
                    for previous, current in zip(indices, group_indices)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Any],
        targets: list[dict[str, Any]],
        group_detr: int = 1,
        target_side_safety: _TargetSideSafety | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        """Performs the matching

        Args:
            outputs: Dict containing at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: List of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates "masks": Tensor of
                 dim [num_target_boxes, H, W] containing the target mask coordinates
            group_detr: Number of groups used for matching.
            target_side_safety: An optional precomputed result from
                :meth:`_precompute_target_side_safety`, forwarded to
                :meth:`_detection_inputs_are_safe` to skip recomputing the target-side half of the
                compact-path safety gate when it still applies.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds len(index_i) == len(index_j). With group_detr == 1 this
            length is min(num_queries, num_target_boxes); with group_detr > 1 the per-group matches are
            concatenated, so the length is that quantity summed over the groups.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        masks_present = "masks" in targets[0]
        keypoints_present = "pred_keypoints" in outputs and "keypoints" in targets[0]
        compact_eligible = self._compact_path_applicable(outputs, targets) and self._detection_inputs_are_safe(
            outputs, targets, target_side_safety
        )
        if compact_eligible:
            sizes = [len(target["boxes"]) for target in targets]
            # A single layer alone (unlike `_match_many`, which buckets every layer of a step
            # together) often sits below the measured SciPy/GPU crossover, so the batched solver is
            # only worth its launch overhead here once this one layer supplies enough problems by
            # itself (see `_BATCHED_ASSIGNMENT_MIN_PROBLEMS`); below that, solve on the host as before.
            solve_on_device = len(sizes) * group_detr >= _BATCHED_ASSIGNMENT_MIN_PROBLEMS
            compact_cost_matrix = self._compute_compact_detection_cost_matrix(outputs, targets).float()
            if not solve_on_device:
                compact_cost_matrix = compact_cost_matrix.cpu()
            if torch.isfinite(compact_cost_matrix).all():
                if solve_on_device:
                    return _assignment.assign_many_bucketed([compact_cost_matrix], sizes, group_detr)[0]
                return self._assign_compact_cost_matrix(compact_cost_matrix, sizes, group_detr)
            # Weighted costs overflowed despite finite, bounded inputs (e.g. an extreme cost
            # coefficient) — fall through to the full path below instead of sanitizing the padded
            # compact matrix in isolation, whose finite-value statistics (and therefore the
            # sentinel `_sanitize_cost_matrix` computes) can differ from the full cartesian
            # matrix's. Falling through keeps this exceptional branch byte-for-byte identical to
            # the pre-existing behaviour; it costs the redundant cross-image compute only on this
            # already-rare path.

        # We flatten to compute the cost matrices in a batch
        flat_pred_logits = outputs["pred_logits"].flatten(0, 1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_keypoints = None

        if keypoints_present:
            tgt_keypoints = torch.cat([v["keypoints"] for v in targets], dim=0)

        # Compute the giou cost between boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou = -giou

        # Compute the classification cost.
        # Gather the target-class columns first: only the tgt_ids columns of the focal terms are
        # consumed, so computing them over all num_classes columns would be wasted work/memory —
        # a real win when num_targets <= num_classes (e.g. large-vocabulary datasets). When
        # num_targets exceeds num_classes (repeated columns, e.g. crowded COCO batches) this
        # gathers marginally more values than the full materialization would; net impact is
        # negligible either way since the Hungarian solve dominates matcher wall-time.
        tgt_logits = flat_pred_logits[:, tgt_ids]  # [batch_size * num_queries, num_targets]
        cost_class = self._focal_classification_cost(tgt_logits)

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        if masks_present:
            tgt_masks = torch.cat([v["masks"] for v in targets])

            if isinstance(outputs["pred_masks"], Tensor):
                out_masks = outputs["pred_masks"].flatten(0, 1)

                num_points = out_masks.shape[-2] * out_masks.shape[-1] // self.mask_point_sample_ratio

                point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
                pred_masks_logits = point_sample(
                    out_masks.unsqueeze(1), point_coords.repeat(out_masks.shape[0], 1, 1), align_corners=False
                ).squeeze(1)
            else:
                spatial_features = outputs["pred_masks"]["spatial_features"]
                query_features = outputs["pred_masks"]["query_features"]
                bias = outputs["pred_masks"]["bias"]

                num_points = spatial_features.shape[-2] * spatial_features.shape[-1] // self.mask_point_sample_ratio
                point_coords = torch.rand(1, num_points, 2, device=spatial_features.device)
                pred_masks_logits = point_sample(
                    spatial_features, point_coords.repeat(spatial_features.shape[0], 1, 1), align_corners=False
                )
                # print(f"pred_masks_logits.shape: {pred_masks_logits.shape}")
                pred_masks_logits = torch.einsum("bcp,bnc->bnp", pred_masks_logits, query_features) + bias
                pred_masks_logits = pred_masks_logits.flatten(0, 1)

            tgt_masks = tgt_masks.to(pred_masks_logits.dtype)
            tgt_masks_flat = point_sample(
                tgt_masks.unsqueeze(1),
                point_coords.repeat(tgt_masks.shape[0], 1, 1),
                align_corners=False,
                mode="nearest",
            ).squeeze(1)

            # Binary cross-entropy with logits cost (mean over pixels), computed pairwise efficiently
            cost_mask_ce = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)

            # Dice loss cost (1 - dice coefficient)
            cost_mask_dice = batch_dice_loss(pred_masks_logits, tgt_masks_flat)

        if keypoints_present and tgt_keypoints is not None:
            target_areas = tgt_bbox[:, 2] * tgt_bbox[:, 3]
            cost_l1, cost_findable, cost_visible, cost_nll = compute_keypoint_matching_cost(
                all_pred_keypoints=outputs["pred_keypoints"],
                target_keypoints=tgt_keypoints,
                target_classes=tgt_ids,
                target_areas=target_areas,
                num_keypoints_per_class=self.num_keypoints_per_class,
            )
            cost_l1 = cost_l1.flatten(0, 1)
            cost_findable = cost_findable.flatten(0, 1)
            cost_visible = cost_visible.flatten(0, 1)
            cost_nll = cost_nll.flatten(0, 1)

        # Final cost matrix
        cost_matrix: Tensor = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        if masks_present:
            cost_matrix = cost_matrix + self.cost_mask_ce * cost_mask_ce + self.cost_mask_dice * cost_mask_dice
        if keypoints_present:
            cost_matrix = (
                cost_matrix
                + self.keypoint_l1_loss_coef * cost_l1
                + self.keypoint_findable_loss_coef * cost_findable
                + self.keypoint_visible_loss_coef * cost_visible
                + self.keypoint_nll_loss_coef * cost_nll
            )
        cost_matrix = cost_matrix.view(bs, num_queries, -1).float()

        # We assume any good match will not cause NaN or Inf, so replace invalid
        # entries with a finite value that is larger than every valid cost.
        all_finite = torch.isfinite(cost_matrix).all().item()
        if not all_finite:
            cost_matrix = cost_matrix.cpu()
            if not self._warned_non_finite_costs:
                logger.warning(
                    "Non-finite values detected in matcher cost matrix; "
                    "replacing with finite sentinel. "
                    "Check for numerical instability."
                )
                self._warned_non_finite_costs = True
            cost_matrix = self._sanitize_cost_matrix(cost_matrix)

        sizes = [len(v["boxes"]) for v in targets]
        target_offsets = [0]
        for size in sizes:
            target_offsets.append(target_offsets[-1] + size)
        diagonal_cost_matrix: Tensor = torch.cat(
            [cost_matrix[i, :, target_offsets[i] : target_offsets[i + 1]] for i in range(bs)], dim=-1
        ).cpu()
        return self._assign_compact_cost_matrix(diagonal_cost_matrix, sizes, group_detr)


def build_matcher(args: Any) -> HungarianMatcher:
    """Build a HungarianMatcher from a training argument namespace.

    Args:
        args: Namespace supplying ``focal_alpha``, ``set_cost_class``, ``set_cost_bbox``,
            ``set_cost_giou``, ``segmentation_head``, and optional keypoint cost
            coefficients (``keypoint_l1_loss_coef``, ``keypoint_findable_loss_coef``,
            ``keypoint_visible_loss_coef``, ``keypoint_nll_loss_coef``). When
            ``segmentation_head`` is truthy, also requires ``mask_ce_loss_coef``,
            ``mask_dice_loss_coef``, and ``mask_point_sample_ratio``.

    Returns:
        Configured HungarianMatcher instance.
    """
    # Detection-only matcher args may omit keypoint costs; zero defaults disable keypoint matching terms.
    common_kwargs = {
        "cost_class": args.set_cost_class,
        "cost_bbox": args.set_cost_bbox,
        "cost_giou": args.set_cost_giou,
        "focal_alpha": args.focal_alpha,
        "num_keypoints_per_class": getattr(args, "num_keypoints_per_class", []),
        "keypoint_l1_loss_coef": getattr(args, "keypoint_l1_loss_coef", 0.0),
        "keypoint_findable_loss_coef": getattr(args, "keypoint_findable_loss_coef", 0.0),
        "keypoint_visible_loss_coef": getattr(args, "keypoint_visible_loss_coef", 0.0),
        "keypoint_nll_loss_coef": getattr(args, "keypoint_nll_loss_coef", 0.0),
    }
    if args.segmentation_head:
        return HungarianMatcher(
            **common_kwargs,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
        )
    else:
        return HungarianMatcher(**common_kwargs)
