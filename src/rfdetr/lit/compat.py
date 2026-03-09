# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Legacy compatibility adapters for the migration period (Phase 0 / T6).

Provides thin wrappers that convert PTL validation_step outputs back to the
stats schema expected by benchmark tests and the existing engine.evaluate() API.

TODO(Chapter 6): delete this entire module once the benchmark tests in
    tests/benchmarks/ have been migrated off engine.evaluate() and the legacy
    stats schema (results_json / coco_eval_*).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import torch

# TODO(Chapter 6): remove once compat.evaluate() callers are migrated.
from rfdetr.engine import evaluate as _engine_evaluate

if TYPE_CHECKING:
    from rfdetr.lit.module import RFDETRModule


def evaluate(
    module: RFDETRModule,
    data_loader: Any,
    base_ds: Any,
    device: torch.device,
) -> Tuple[Dict[str, Any], Any]:
    """Evaluate a RFDETRModule and return the legacy stats schema.

    Thin compatibility wrapper for the migration period.  Extracts the model,
    criterion, postprocessor, and Namespace args from *module* and delegates to
    :func:`rfdetr.engine.evaluate`, returning its output unchanged.

    Use this when you want to swap the legacy
    ``engine.evaluate(model.model, criterion, postprocess, …)`` call with one
    that accepts an :class:`~rfdetr.lit.module.RFDETRModule` directly.  Call
    sites in benchmark tests and downstream code can migrate from
    ``engine.evaluate`` to ``compat.evaluate`` one call-site at a time, with
    no change to the expected return schema.

    Args:
        module: Trained RFDETRModule instance whose ``.model``, ``.criterion``,
            ``.postprocess``, and ``._args`` are forwarded to
            :func:`rfdetr.engine.evaluate`.
        data_loader: DataLoader providing ``(NestedTensor, list[dict])`` batches.
        base_ds: COCO ground-truth object (e.g. from
            :func:`rfdetr.datasets.get_coco_api_from_dataset`).
        device: Target device for evaluation.

    Returns:
        Tuple ``(stats, coco_evaluator)`` identical to the return value of
        :func:`rfdetr.engine.evaluate`:

        - ``stats`` — ``dict`` with loss scalars, ``"results_json"`` (and
          ``"results_json_masks"`` / ``"coco_eval_*"`` keys for segmentation).
        - ``coco_evaluator`` — the
          :class:`~rfdetr.datasets.coco_eval.CocoEvaluator` instance populated
          during evaluation.
    """
    return _engine_evaluate(
        module.model,
        module.criterion,
        module.postprocess,
        data_loader,
        base_ds,
        device,
        args=module._args,
    )
