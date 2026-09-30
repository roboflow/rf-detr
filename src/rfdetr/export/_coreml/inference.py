# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load and execute native CoreML exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rfdetr.export._runtime.metadata import ExportMetadata


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str) -> Any:
    """Load a CoreML model and validate its tensor interface."""
    from rfdetr.export._runtime.adapters import ExportRuntime, _input_array, _require_apple

    _require_apple("CoreML")
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ImportError("CoreML inference requires coremltools.") from exc
    units = {"auto": ct.ComputeUnit.ALL, "cpu": ct.ComputeUnit.CPU_ONLY}
    if device not in units:
        raise ValueError("CoreML accepts auto or cpu. Its GPU and Neural Engine policies also permit CPU execution.")
    session = ct.models.MLModel(str(path), compute_units=units[device])
    spec = session.get_spec()
    (input_info,) = spec.description.input
    if metadata.input_name not in {0, input_info.name}:
        raise ValueError("CoreML input name disagrees with export metadata.")
    feature_type = input_info.type
    if feature_type.WhichOneof("Type") != "multiArrayType":
        raise ValueError("CoreML input must be a multiArrayType.")
    array_type = feature_type.multiArrayType
    input_shape = tuple(array_type.shape)
    if len(input_shape) != 4:
        raise ValueError(f"CoreML input rank must be 4, got {len(input_shape)}.")
    expected_shape = metadata.input_shape
    if metadata.input_layout == "NHWC":
        expected_shape = (expected_shape[0], expected_shape[2], expected_shape[3], expected_shape[1])
    if input_shape != expected_shape:
        raise ValueError(f"CoreML input shape {input_shape} disagrees with export metadata {expected_shape}.")
    coreml_dtypes = {65552: "float16", 65568: "float32", 65600: "float64"}
    if coreml_dtypes.get(array_type.dataType) != metadata.input_dtype:
        raise ValueError("CoreML input dtype disagrees with export metadata.")
    names = [item.name for item in spec.description.output]
    for name in metadata.outputs.values():
        if isinstance(name, str) and name not in names:
            raise ValueError(f"CoreML output name {name!r} is absent from model.")
        if isinstance(name, int) and not (0 <= name < len(names)):
            raise ValueError(f"CoreML output position {name} is absent from model.")

    def execute(batch: torch.Tensor) -> dict[str, Any]:
        """Return outputs in the model specification's order."""
        prediction = session.predict({input_info.name: _input_array(batch, metadata)})
        return {name: prediction[name] for name in names}

    return ExportRuntime("coreml", metadata, session, device, input_info.name, execute, borrowed_outputs=False)
