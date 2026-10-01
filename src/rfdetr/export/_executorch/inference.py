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

from pathlib import Path
from typing import Any


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
