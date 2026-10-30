# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The coremltools graph pass that the Core ML export adds to the default pass pipeline.

coremltools is optional, so this module imports it inside the functions only: the test suite imports every module, also
where coremltools is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rfdetr.export._coreml.interpolation import bilinear_interpolation_matrix
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from coremltools import PassPipeline

logger = get_logger()

#: The one place the pass id is written: registration splits it into namespace and name, the pipeline inserts it by id.
UPSAMPLE_BILINEAR_TO_MATMUL = "rfdetr::upsample_bilinear_to_matmul"

#: The coremltools pass the RF-DETR pass runs after, so it sees the program with complex dialect ops already lowered.
_LOWERING_PASS = "common::lower_complex_dialect_ops"


def rfdetr_pass_pipeline() -> PassPipeline:
    """The default coremltools pass pipeline, with ``rfdetr::upsample_bilinear_to_matmul`` after the dialect lowering.

    On the Neural Engine, ``upsample_bilinear`` by a factor of 3 samples half an output cell off: output cell ``i``
    reads source cell ``i / 3 - 0.5`` instead of ``(i + 0.5) / 3 - 0.5``. The CPU and the GPU are right, and so are
    factors 2, 4, 5 and 6 on the Neural Engine. The segmentation head upsamples its mask features 3x to the stride-4
    mask grid, so under ``CPU_AND_NE`` every mask lands half a mask cell below and right of its object. The pass
    computes the same resize with matmuls (:func:`rewrite_upsample_bilinear`), which are right on every compute unit.

    Returns:
        A ``coremltools.PassPipeline`` for ``coremltools.convert(pass_pipeline=...)``. ``coremltools.convert``
        changes the pipeline it is given, so build one per conversion.

    Raises:
        RuntimeError: If the installed ``coremltools`` has no ``common::lower_complex_dialect_ops`` pass to insert the
            RF-DETR pass after, which happens in a ``coremltools`` release outside the supported range.
    """
    import coremltools as ct

    _register_upsample_bilinear_to_matmul()
    pipeline = ct.PassPipeline.DEFAULT
    try:
        after_lowering = pipeline.passes.index(_LOWERING_PASS) + 1
    except ValueError as exc:
        raise RuntimeError(
            f"coremltools {ct.__version__} has no {_LOWERING_PASS!r} pass to run "
            f"{UPSAMPLE_BILINEAR_TO_MATMUL!r} after. The Core ML export supports coremltools >=8.0,<10.0."
        ) from exc
    pipeline.insert_pass(index=after_lowering, pass_name=UPSAMPLE_BILINEAR_TO_MATMUL)
    return pipeline


def _register_upsample_bilinear_to_matmul() -> None:
    """Register :func:`rewrite_upsample_bilinear` as the graph pass ``rfdetr::upsample_bilinear_to_matmul``.

    The registration overrides an existing one, so it is safe to repeat: two threads running their first export at once,
    or a test that restores ``sys.modules`` and imports this module again, register the same pass twice.
    """
    from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
    from coremltools.converters.mil.mil.passes.pass_registry import register_pass

    namespace, name = UPSAMPLE_BILINEAR_TO_MATMUL.split("::")

    class UpsampleBilinearToMatmul(AbstractGraphPass):  # type: ignore[misc]
        def apply(self, prog: Any) -> None:
            rewrite_upsample_bilinear(prog)

    register_pass(namespace=namespace, name=name, override=True)(UpsampleBilinearToMatmul)


def rewrite_upsample_bilinear(prog: Any) -> None:
    """Replace each ``upsample_bilinear`` of a MIL program with matmuls by constant matrices, ``R_h @ x @ R_w^T``.

    The Neural Engine compiles a matmul by a constant only with the constant on the right, so the height pass swaps
    the last two axes, multiplies by ``R_h^T`` and swaps them back. Written as ``R_h @ y``, it runs on the CPU.

    Args:
        prog: The ``coremltools`` MIL ``Program`` the pass pipeline is converting. It is changed in place.
    """
    for function in prog.functions.values():
        _rewrite_block(function)


def _rewrite_block(block: Any) -> None:
    """Rewrite the ``upsample_bilinear`` ops of ``block`` and of the blocks nested in its ops.

    Args:
        block: A MIL function body or a nested block, such as a branch of a ``cond``.
    """
    with block:
        for op in list(block.operations):
            for nested in op.blocks:
                _rewrite_block(nested)
            if op.op_type == "upsample_bilinear":
                _try_rewrite(op, block)


def _try_rewrite(op: Any, block: Any) -> None:
    """Replace one ``upsample_bilinear`` with the matmuls, or keep it and log a warning when it cannot.

    Args:
        op: The ``upsample_bilinear`` op.
        block: The block that holds ``op``.
    """
    from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias
    from coremltools.converters.mil.mil import types
    from coremltools.converters.mil.mil.types.symbolic import is_symbolic

    x, output = op.x, op.outputs[0]
    align_corners = bool(op.align_corners.val)
    half_pixel_centers = op.inputs.get("half_pixel_centers")
    # iOS 16 can set half_pixel_centers apart from align_corners; only the two modes PyTorch emits are rewritten.
    if half_pixel_centers is not None and bool(half_pixel_centers.val) == align_corners:
        logger.warning(
            f"Keeping {op.name} as upsample_bilinear, which the Neural Engine can misplace: the matmul rewrite needs "
            f"half_pixel_centers == not align_corners (got half_pixel_centers={bool(half_pixel_centers.val)}, "
            f"align_corners={align_corners})."
        )
        return
    (height_in, width_in), (height_out, width_out) = x.shape[-2:], output.shape[-2:]
    if any(is_symbolic(size) for size in (height_in, width_in, height_out, width_out)):
        logger.warning(
            f"Keeping {op.name} as upsample_bilinear, which the Neural Engine can misplace: the matmul rewrite needs "
            f"fixed sizes (got {x.shape[-2:]} to {output.shape[-2:]})."
        )
        return

    dtype = types.nptype_from_builtin(x.dtype)
    height_matrix_t = bilinear_interpolation_matrix(height_in, height_out, align_corners).T.astype(dtype)
    width_matrix_t = bilinear_interpolation_matrix(width_in, width_out, align_corners).T.astype(dtype)
    swap_last_two = [*range(x.rank - 2), x.rank - 1, x.rank - 2]
    resized_width = mb.matmul(x=x, y=width_matrix_t, before_op=op)
    transposed = mb.transpose(x=resized_width, perm=swap_last_two, before_op=op)
    resized_transposed = mb.matmul(x=transposed, y=height_matrix_t, before_op=op)
    resized = mb.transpose(x=resized_transposed, perm=swap_last_two, before_op=op, name=op.name)
    block.replace_uses_of_var_after_op(anchor_op=op, old_var=output, new_var=resized)
    op.remove_from_block()
