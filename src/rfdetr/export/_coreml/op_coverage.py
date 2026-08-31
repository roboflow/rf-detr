# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Compare an exported graph to coremltools' Torch op registry (pre-convert checklist)."""

from __future__ import annotations

from collections import Counter

from torch.export import ExportedProgram


def unsupported_coreml_ops(exported_program: ExportedProgram) -> Counter[str]:
    """Return ``{op_kind: count}`` for call_function ops missing from coremltools' registry.

    Walk the decomposed ``torch.export`` graph and ask coremltools whether each op kind is
    registered. Applies :func:`~rfdetr.export._coreml.torch_ops.ensure_coreml_torch_op_patches`
    first so the checklist matches what ``export_coreml`` will convert. Empty result means
    registry-clean (convert may still fail later for other reasons).

    Raises:
        ImportError: If ``coremltools`` is not installed.
    """
    # `_TORCH_OPS_REGISTRY` and `sanitize_op_kind` are coremltools PRIVATE internals — see the pin
    # rationale + fail-loud policy note in torch_ops.py's `ensure_coreml_torch_op_patches`.
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY
    from coremltools.converters.mil.frontend.torch.utils import sanitize_op_kind

    from rfdetr.export._coreml.torch_ops import ensure_coreml_torch_op_patches

    ensure_coreml_torch_op_patches()

    gaps: Counter[str] = Counter()
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        try:
            kind_raw = node.target.name()
        except Exception:
            kind_raw = getattr(node.target, "__name__", str(node.target))
        kind = sanitize_op_kind(str(kind_raw))
        if kind not in _TORCH_OPS_REGISTRY.name_to_func_mapping:
            gaps[kind] += 1
    return gaps
