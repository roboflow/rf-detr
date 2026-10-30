# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The branches of ``rewrite_upsample_bilinear``, run on MIL programs without the Core ML runtime.

``tests/export/test_coreml_upsample.py`` converts and runs real models, so only the Core ML integration job on macOS
reaches it. Here the pass rewrites a MIL program built in memory, which needs coremltools and nothing else.
"""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.passes import rewrite_upsample_bilinear

pytestmark = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="needs coremltools")


def _upsample_program(input_shape: tuple[Any, ...], *, opset: str = "iOS15", **upsample_kwargs: Any) -> Any:
    """Build a MIL program that applies one 3x ``upsample_bilinear`` to its input.

    Args:
        input_shape: The input shape; a symbolic size from ``coremltools`` makes a dimension dynamic.
        opset: The name of the ``coremltools.target`` member whose opset the program uses.
        **upsample_kwargs: Arguments of ``upsample_bilinear`` besides the input and the scale factors.

    Returns:
        The ``coremltools`` MIL ``Program``.

    Examples:
        Needs coremltools, so the example is not run:

        >>> _upsample_program((1, 4, 26, 20), align_corners=False)  # doctest: +SKIP
    """
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias

    @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=getattr(ct.target, opset))
    def program(x: Any) -> Any:
        return mb.upsample_bilinear(x=x, scale_factor_height=3.0, scale_factor_width=3.0, **upsample_kwargs)

    return program


def _op_types(block: Any) -> list[str]:
    """List the op types of ``block`` and of the blocks nested in its ops, constants left out.

    Args:
        block: A MIL function body or a nested block.

    Returns:
        The op types in program order, each op followed by the ops of its nested blocks.

    Examples:
        Needs coremltools, so the example is not run:

        >>> _op_types(_upsample_program((1, 4, 26, 20), align_corners=False).functions["main"])  # doctest: +SKIP
    """
    types: list[str] = []
    for op in block.operations:
        if op.op_type != "const":
            types.append(op.op_type)
        for nested in op.blocks:
            types.extend(_op_types(nested))
    return types


class TestRewriteUpsampleBilinear:
    """``rewrite_upsample_bilinear`` swaps a fixed-size ``upsample_bilinear`` for matmuls and keeps the rest."""

    def test_replaces_the_resize_of_a_non_square_input_with_two_matmuls(self) -> None:
        """A 26x20 input becomes ``matmul``, ``transpose``, ``matmul``, ``transpose`` and no ``upsample_bilinear``.

        The width pass multiplies the input by ``R_w^T``, the height pass swaps the last two axes around a second matmul
        by ``R_h^T``. The runtime tests check the values; this checks the structure on any host with coremltools.
        """
        program = _upsample_program((1, 4, 26, 20), align_corners=False)

        rewrite_upsample_bilinear(program)

        assert _op_types(program.functions["main"]) == ["matmul", "transpose", "matmul", "transpose"]

    @pytest.mark.parametrize(
        ("align_corners", "half_pixel_centers"),
        [pytest.param(False, True, id="pytorch-default"), pytest.param(True, False, id="align-corners")],
    )
    def test_rewrites_the_two_modes_pytorch_emits(self, align_corners: bool, half_pixel_centers: bool) -> None:
        """``half_pixel_centers`` set to ``not align_corners`` is a mode PyTorch emits, so the resize is rewritten."""
        program = _upsample_program(
            (1, 4, 26, 20), opset="iOS16", align_corners=align_corners, half_pixel_centers=half_pixel_centers
        )

        rewrite_upsample_bilinear(program)

        assert "upsample_bilinear" not in _op_types(program.functions["main"])

    @pytest.mark.parametrize("mode", [False, True])
    def test_keeps_and_warns_on_a_half_pixel_mode_pytorch_never_emits(self, mode: bool) -> None:
        """``half_pixel_centers`` equal to ``align_corners`` stays an ``upsample_bilinear`` and logs a warning.

        iOS 16 can set the two apart from each other; the interpolation matrices only match the two modes PyTorch emits.
        """
        program = _upsample_program((1, 4, 26, 20), opset="iOS16", align_corners=mode, half_pixel_centers=mode)

        with mock.patch("rfdetr.export._coreml.passes.logger") as logger:
            rewrite_upsample_bilinear(program)

        assert _op_types(program.functions["main"]) == ["upsample_bilinear"]
        logger.warning.assert_called_once()
        assert "half_pixel_centers" in logger.warning.call_args.args[0]

    def test_keeps_and_warns_on_a_symbolic_size(self) -> None:
        """A resize with a dynamic input height stays an ``upsample_bilinear`` and logs a warning.

        The interpolation matrices are constants of fixed size, so a size that is only known at run time has none.
        """
        from coremltools.converters.mil.mil import get_new_symbol

        program = _upsample_program((1, 4, get_new_symbol(), 20), align_corners=False)

        with mock.patch("rfdetr.export._coreml.passes.logger") as logger:
            rewrite_upsample_bilinear(program)

        assert _op_types(program.functions["main"]) == ["upsample_bilinear"]
        logger.warning.assert_called_once()
        assert "fixed sizes" in logger.warning.call_args.args[0]

    def test_rewrites_a_resize_inside_a_nested_block(self) -> None:
        """Both branches of a ``cond`` lose their ``upsample_bilinear``, which sits in a block nested in the op.

        Without the recursion into ``op.blocks``, only resizes at the top level of a function would be rewritten.
        """
        import coremltools as ct
        from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 26, 20))], opset_version=ct.target.iOS15)
        def program(x: Any) -> Any:
            def resize() -> Any:
                return mb.upsample_bilinear(x=x, scale_factor_height=3.0, scale_factor_width=3.0, align_corners=False)

            return mb.cond(pred=True, _true_fn=resize, _false_fn=resize)

        rewrite_upsample_bilinear(program)

        types = _op_types(program.functions["main"])
        assert "upsample_bilinear" not in types
        assert types.count("matmul") == 4
