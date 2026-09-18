# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Local patches to coremltools' Torch op registry for RF-DETR CoreML export.

Prefer package-local registry patches over model call-site rewrites. coremltools
already knows close cousins of the ops ``torch.export`` emits for RF-DETR; we wire
the missing / broken names onto those handlers (or a thin override) so ``ct.convert``
does not fail deep inside coremltools with no RF-DETR line number.

Patches (idempotent):

* ``alias`` → same identity ``noop`` as ``alias_copy`` / ``clone`` / ``contiguous``
  (upstream forgot plain ``aten.alias`` on that alias list).
* ``__and__`` → ``bitwise_and`` (bool ``&``; upstream registers ``and`` / ``bitwise_and``
  but not ``__and__``).
* ``bitwise_not`` → override: same as upstream for int/bool, plus float→bool cast then
  ``logical_not``. FX shows RF-DETR's ``~mask`` ops are bool, but coremltools sometimes
  types the MIL var as float64 before this handler runs.

Process-global, permanent, opt-in only:
    :func:`ensure_coreml_torch_op_patches` mutates coremltools' **process-wide**
    ``_TORCH_OPS_REGISTRY`` in place. There is no automatic teardown in production — once any
    CoreML export runs in a process, these handlers (including the unconditional ``bitwise_not``
    override) stay installed for the rest of that process, including for any other coremltools
    consumer running in the same process afterward (e.g. a subsequent ExecuTorch
    ``backend="coreml"`` conversion). This is intentional: CoreML export is opt-in/experimental and
    the common case is a single-process CLI export where the leak is harmless. Call-site critical
    sections are lock-guarded (below) so concurrent callers cannot observe a partially-patched
    registry; the module-level ``_PATCHED`` flag itself is validated against actual registry
    contents, not trusted blindly, so a failed prior patch attempt always retries cleanly. Tests
    that need isolation use ``_restore_coreml_torch_op_registry`` in
    ``tests/export/test_coreml_op_coverage.py`` (snapshot/restore fixture) — that pattern is not
    applied in production because it would mean every export pays snapshot/restore overhead and
    briefly de-patches the registry, racing any concurrent coremltools user in the same process.
"""

from __future__ import annotations

import threading
from typing import Any

_PATCHED: bool = False
_PATCH_LOCK = threading.Lock()


def _bitwise_not_allowing_float_bool_masks(context: Any, node: Any) -> None:
    """Lower ``bitwise_not`` like coremltools, also accepting float-typed bool masks.

    Args:
        context: coremltools Torch conversion context.
        node: Internal graph node for ``bitwise_not``.

    Raises:
        ValueError: If the input dtype is not int, bool, or float.
    """
    from coremltools.converters.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil.mil import types

    inputs = _get_inputs(context, node)
    x = inputs[0]
    dtype = x.dtype
    if types.is_int(dtype):
        x = mb.add(x=x, y=1)
        x = mb.mul(x=x, y=-1, name=node.name)
    elif types.is_bool(dtype):
        x = mb.logical_not(x=x, name=node.name)
    elif types.is_float(dtype):
        # Bool masks from ``~`` sometimes arrive here typed as float64 in MIL.
        x = mb.cast(x=x, dtype="bool")
        x = mb.logical_not(x=x, name=node.name)
    else:
        raise ValueError(f"Not supported type {dtype} found for 'bitwise_not' op")
    context.add(x)


def ensure_coreml_torch_op_patches() -> None:
    """Register RF-DETR CoreML Torch-op patches on coremltools' registry (idempotent).

    Thread-safe: the check-then-patch critical section is lock-guarded, so two threads racing
    entry cannot leave a partially-patched registry. The ``_PATCHED`` flag is a fast-path hint
    only — it is validated against actual registry contents (not trusted blindly), so an exception
    part-way through a prior attempt always causes a clean retry rather than a silent no-op.

    Raises:
        ImportError: If ``coremltools`` is not installed.
        RuntimeError: If an expected upstream handler is missing (unexpected coremltools
            layout — we refuse to invent unrelated handlers).
    """
    global _PATCHED

    # `_TORCH_OPS_REGISTRY`, `.set_func_by_name`, `.name_to_func_mapping` are coremltools PRIVATE
    # internals (underscore-prefixed) and can rename silently across minors — the
    # `coremltools>=8.0,<10.0` pin (pyproject.toml `[project.optional-dependencies].coreml`,
    # tested against 9.0) is the only guardrail. A `RuntimeError` below (not a silent skip) is the
    # deliberate fail-loud signal that this surface moved.
    # Importing ``ops`` runs @register_torch_op decorators and fills the shared registry.
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

    mapping = _TORCH_OPS_REGISTRY.name_to_func_mapping

    with _PATCH_LOCK:
        if (
            _PATCHED
            and mapping.get("alias") is mapping.get("alias_copy")
            and mapping.get("__and__") is mapping.get("bitwise_and")
            and mapping.get("bitwise_not") is _bitwise_not_allowing_float_bool_masks
        ):
            return

        if "alias" not in mapping or mapping.get("alias") is not mapping.get("alias_copy"):
            if "alias_copy" not in mapping:
                raise RuntimeError(
                    "coremltools Torch registry is missing 'alias_copy' (expected shared noop). "
                    "Cannot patch 'alias' for CoreML export — check the installed coremltools version."
                )
            _TORCH_OPS_REGISTRY.set_func_by_name(mapping["alias_copy"], "alias")

        if "__and__" not in mapping or mapping.get("__and__") is not mapping.get("bitwise_and"):
            if "bitwise_and" not in mapping:
                raise RuntimeError(
                    "coremltools Torch registry is missing 'bitwise_and'. "
                    "Cannot patch '__and__' for CoreML export — check the installed coremltools version."
                )
            _TORCH_OPS_REGISTRY.set_func_by_name(mapping["bitwise_and"], "__and__")

        # Always install our override (upstream raises on float-typed bool masks).
        _TORCH_OPS_REGISTRY.set_func_by_name(_bitwise_not_allowing_float_bool_masks, "bitwise_not")

        _PATCHED = True
