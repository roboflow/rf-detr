# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Load a ``.pte`` built by :mod:`rfdetr.export._executorch.exporter` and run it via ExecuTorch's Python runtime.

Backend-agnostic: the same ``Runtime.get().load_program(...).load_method(...)`` call loads a ``.pte`` regardless of
which backend (XNNPACK, CoreML, QNN) it was delegated to at export time -- only the artifact's own content differs.
Nothing decodes detections here, same division of responsibility as :mod:`rfdetr.export._tensorrt.inference`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

from rfdetr.export._runtime.metadata import ExportMetadata


def load_executorch_method(pte_path: str | Path, *, method_name: str = "forward") -> Any:
    """Load one method of a ``.pte`` program through ExecuTorch's Python runtime.

    Calls :func:`rfdetr.export._executorch.exporter._check_executorch_available` with
    ``require_runtime=True`` first, so a torch/executorch ABI mismatch (see that function's docstring) raises a
    friendly, actionable :class:`ImportError` here rather than surfacing later as an opaque native crash (e.g.
    ``RuntimeError: tensor does not have a device``) on the first ``method.execute(...)`` call.

    Args:
        pte_path: Path to a ``.pte`` file written by ``RFDETR.export(format="executorch", ...)``.
        method_name: Name of the method to load; RF-DETR's exporter always writes ``"forward"``.

    Returns:
        A loaded ``executorch.runtime.Method``, callable as ``method.execute([input_tensor])``.

    Raises:
        ImportError: If ``executorch`` is not installed, is older than the minimum supported version, or its
            compiled runtime extension cannot load against the installed ``torch`` (see
            :func:`~rfdetr.export._executorch.exporter._check_executorch_available`).
    """
    from rfdetr.export._executorch.exporter import _check_executorch_available

    _check_executorch_available(require_runtime=True)

    from executorch.runtime import Runtime

    return Runtime.get().load_program(str(pte_path)).load_method(method_name)


def load_export_runtime(path: Path, metadata: ExportMetadata, device: str, options: Mapping[str, Any]) -> Any:
    """Load an ExecuTorch method with its embedded delegate policy; it reads no runtime options."""
    from rfdetr.export._runtime.adapters import ExportRuntime, _input_array, _require_apple, _runtime_options

    _runtime_options("ExecuTorch", options, ())

    delegate = (metadata.backend or "xnnpack").lower()
    if delegate == "xnnpack":
        if device not in {"auto", "cpu"}:
            raise ValueError("ExecuTorch XNNPACK requires cpu or auto.")
        target = "cpu"
    elif delegate == "coreml":
        _require_apple("ExecuTorch CoreML")
        if device != "auto":
            raise ValueError("ExecuTorch CoreML uses an embedded compute policy; request auto.")
        target = "coreml"
    elif delegate == "qnn":
        if device not in {"auto", "qnn"}:
            raise ValueError("ExecuTorch QNN requires auto or qnn.")
        target = "qnn"
    else:
        raise ValueError(f"Unsupported ExecuTorch delegate {delegate!r}.")
    session = load_executorch_method(path)

    def execute(batch: torch.Tensor) -> list[Any]:
        """Clone mutable input storage before each ExecuTorch call."""
        return cast(list[Any], session.execute([torch.from_numpy(_input_array(batch, metadata)).clone()]))

    return ExportRuntime("executorch", metadata, session, target, metadata.input_name, execute)
