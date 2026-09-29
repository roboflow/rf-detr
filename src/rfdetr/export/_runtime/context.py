# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Connect exported runtimes to the public RFDETR prediction pipeline."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from rfdetr.export._runtime.adapters import ExportRuntime, load_runtime
from rfdetr.export._runtime.metadata import ExportMetadata, read_metadata
from rfdetr.export.inference import RFDETRInference
from rfdetr.models.postprocess import PostProcess

if TYPE_CHECKING:
    from rfdetr.detr import RFDETR


@dataclass
class ExportedModelContext:
    """Hold runtime state and prediction semantics without native network weights."""

    metadata: ExportMetadata
    runtime: ExportRuntime
    postprocess: PostProcess
    device: torch.device = torch.device("cpu")

    @property
    def class_names(self) -> list[str]:
        """Return the artifact's ordered label names."""
        return list(self.metadata.class_names)


def load_exported_model(
    cls: type[RFDETR],
    path: Path,
    *,
    device: str,
    metadata: dict[str, Any] | str | os.PathLike[str] | None,
) -> RFDETRInference:
    """Construct an inference-only wrapper from a validated artifact."""
    contract = read_metadata(path, Path(metadata) if isinstance(metadata, os.PathLike) else metadata)
    if cls.size is not None and cls.size != contract.variant:
        raise ValueError(f"{cls.__name__} requires variant {cls.size!r}, but artifact declares {contract.variant!r}.")
    runtime = load_runtime(path, contract, device=device)
    context = ExportedModelContext(
        metadata=contract,
        runtime=runtime,
        postprocess=PostProcess(
            num_select=contract.num_select,
            num_keypoints_per_class=contract.num_keypoints_per_class,
            trace_alpha=contract.trace_alpha,
            upsample_masks_to_image_size=contract.upsample_masks_to_image_size,
        ),
    )
    model = cls.__new__(cls)
    model._exported_context = context
    model.means = list(contract.means)
    model.stds = list(contract.stds)
    model.callbacks = defaultdict(list)
    model._initialize_inference_state()
    return RFDETRInference(model)
