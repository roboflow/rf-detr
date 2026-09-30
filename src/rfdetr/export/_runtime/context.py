# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build prediction rules from an exported artifact and its runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from rfdetr._prediction import PredictionContext
from rfdetr.export._runtime.adapters import load_runtime
from rfdetr.export._runtime.metadata import read_metadata
from rfdetr.models.postprocess import PostProcess


def load_exported_context(
    path: Path,
    *,
    device: str,
    metadata: dict[str, Any] | str | os.PathLike[str] | None,
) -> PredictionContext:
    """Load one runtime and the artifact's prediction rules.

    Args:
        path: Exported file or model bundle.
        device: Runtime device policy.
        metadata: Missing legacy metadata or a JSON path.

    Returns:
        A context that owns the runtime through its bound execution method.

    Raises:
        ValueError: If the artifact cannot produce task predictions.
    """
    contract = read_metadata(path, Path(metadata) if isinstance(metadata, os.PathLike) else metadata)
    if contract.task == "backbone":
        raise ValueError("Backbone-only exports have no prediction head and cannot produce detections.")
    runtime = load_runtime(path, contract, device=device)
    return PredictionContext(
        config=contract,
        device=runtime.device,
        default_shape=contract.shape,
        means=list(contract.means),
        stds=list(contract.stds),
        class_names=list(contract.class_names),
        class_id_to_name=dict(contract.class_id_to_name),
        num_classes=contract.num_classes,
        num_keypoints_per_class=list(contract.num_keypoints_per_class),
        postprocess=PostProcess(
            num_select=contract.num_select,
            num_keypoints_per_class=contract.num_keypoints_per_class,
            trace_alpha=contract.trace_alpha,
            upsample_masks_to_image_size=contract.upsample_masks_to_image_size,
        ),
        run=runtime.run,
        runtime_info=dict(runtime.info),
        fixed_shape=True,
        batch_size=None if contract.input_shape[0] == -1 else contract.input_shape[0],
        max_batch_size=contract.max_batch_size,
    )
