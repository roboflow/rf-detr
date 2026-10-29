# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The coremltools graph pass that the Core ML export adds to the default pass pipeline.

Importing this module registers ``rfdetr::upsample_bilinear_to_matmul``, so it needs coremltools; the exporter
imports it inside the conversion only.
"""

from __future__ import annotations

from typing import Any

import coremltools as ct
from coremltools.converters.mil.mil import Block, Operation, Program, types
from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY, register_pass
from coremltools.converters.mil.mil.types.symbolic import is_symbolic

from rfdetr.export._coreml.interpolation import bilinear_interpolation_matrix
from rfdetr.utilities.logger import get_logger

logger = get_logger()

UPSAMPLE_BILINEAR_TO_MATMUL = "rfdetr::upsample_bilinear_to_matmul"


class upsample_bilinear_to_matmul(AbstractGraphPass):  # type: ignore[misc]  # noqa: N801 — the pass is named after the class
    """Replace each ``upsample_bilinear`` with matmuls by constant interpolation matrices, ``R_h @ x @ R_w^T``.

    On the Neural Engine, ``upsample_bilinear`` by a factor of 3 samples half an output cell off: output cell ``i``
    reads source cell ``i / 3 - 0.5`` instead of ``(i + 0.5) / 3 - 0.5``. The CPU and the GPU are right, and so are
    factors 2, 4, 5 and 6 on the Neural Engine. The segmentation head upsamples its patch-12 features 3x to the
    stride-4 mask grid, so under ``CPU_AND_NE`` every mask lands half a mask cell below and right of its object. The
    matmuls compute the same interpolation on every compute unit.

    The Neural Engine compiles a matmul by a constant only with the constant on the right, so the height pass swaps
    the last two axes, multiplies by ``R_h^T`` and swaps them back. Written as ``R_h @ y``, it runs on the CPU.
    """

    def apply(self, prog: Program) -> None:
        """Rewrite every ``upsample_bilinear`` in ``prog`` in place.

        Args:
            prog: The MIL program the pass pipeline is converting.
        """
        for function in prog.functions.values():
            self._rewrite_block(function)

    @block_context_manager  # type: ignore[untyped-decorator]  # coremltools ships no type hints
    def _rewrite_block(self, block: Block) -> None:
        """Rewrite the ``upsample_bilinear`` ops of ``block`` and of the blocks nested in its ops.

        Args:
            block: A function body or a nested block, such as a branch of a ``cond``.
        """
        for op in list(block.operations):
            for nested in op.blocks:
                self._rewrite_block(nested)
            if op.op_type == "upsample_bilinear":
                self._try_rewrite(op, block)

    @staticmethod
    def _try_rewrite(op: Operation, block: Block) -> None:
        """Replace one ``upsample_bilinear`` with the matmuls, or keep it and log a warning when it cannot.

        Args:
            op: The ``upsample_bilinear`` op.
            block: The block that holds ``op``.
        """
        x, output = op.x, op.outputs[0]
        align_corners = bool(op.align_corners.val)
        half_pixel_centers = op.inputs.get("half_pixel_centers")
        # iOS 16 can set half_pixel_centers apart from align_corners; only the two modes PyTorch emits are rewritten.
        if half_pixel_centers is not None and bool(half_pixel_centers.val) == align_corners:
            logger.warning(
                f"Keeping {op.name} as upsample_bilinear, which the Neural Engine can misplace: the matmul rewrite "
                f"needs half_pixel_centers == not align_corners (got "
                f"half_pixel_centers={bool(half_pixel_centers.val)}, align_corners={align_corners})."
            )
            return
        (height_in, width_in), (height_out, width_out) = x.shape[-2:], output.shape[-2:]
        if any(is_symbolic(size) for size in (height_in, width_in, height_out, width_out)):
            logger.warning(
                f"Keeping {op.name} as upsample_bilinear, which the Neural Engine can misplace: the matmul rewrite "
                f"needs fixed sizes (got {x.shape[-2:]} to {output.shape[-2:]})."
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


# A test that restores sys.modules imports this module again, and the registry refuses a second registration.
if UPSAMPLE_BILINEAR_TO_MATMUL not in PASS_REGISTRY:
    register_pass(namespace="rfdetr")(upsample_bilinear_to_matmul)


def rfdetr_pass_pipeline() -> Any:
    """The default coremltools pass pipeline, with :class:`upsample_bilinear_to_matmul` after the dialect lowering.

    Returns:
        A ``coremltools.PassPipeline`` for ``coremltools.convert(pass_pipeline=...)``.
    """
    pipeline = ct.PassPipeline.DEFAULT
    after_lowering = pipeline.passes.index("common::lower_complex_dialect_ops") + 1
    pipeline.insert_pass(index=after_lowering, pass_name=UPSAMPLE_BILINEAR_TO_MATMUL)
    return pipeline
