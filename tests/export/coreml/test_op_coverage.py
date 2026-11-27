# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""CoreML op-coverage checklist.

``ct.convert`` can fail deep inside coremltools without pointing at an RF-DETR line. Export and decompose the graph,
walk ``call_function`` nodes, and check each op kind against coremltools' Torch registry (via
``unsupported_coreml_ops``, after the package-local patches in ``torch_ops.py``) before converting.

Prefer registry patches over model call-site rewrites. After patches, every released size (detection + segmentation)
plus the keypoint preview must have no registry gaps; ``_KNOWN_NANO_UNSUPPORTED_KINDS`` stays empty unless a gap is
accepted deliberately.

The branches of ``rewrite_upsample_bilinear``, run on MIL programs without the Core ML runtime.

``tests/export/coreml/test_upsample.py`` converts and runs real models, so only the Core ML integration job on macOS
reaches it. Here the pass rewrites a MIL program built in memory, which needs coremltools and nothing else.

The matrices the Core ML export multiplies by in place of ``upsample_bilinear``.

No coremltools needed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from pytest_mock import MockerFixture
from torch import nn
from torch.export import ExportedProgram

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE, passes, torch_ops
from rfdetr.export._coreml.interpolation import bilinear_interpolation_matrix
from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._coreml.passes import rewrite_upsample_bilinear
from rfdetr.export._coreml.torch_ops import ensure_coreml_torch_op_patches

_requires_coremltools = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="coremltools not installed")


# Known Nano registry gaps *after* ``ensure_coreml_torch_op_patches``. Keep empty;
# fix new kinds (or add them here deliberately) before merge.
_KNOWN_NANO_UNSUPPORTED_KINDS: frozenset[str] = frozenset()


# Registry keys that tests in this module may pop or override. Snapshotted/restored by
# ``_restore_coreml_torch_op_registry`` so a failing assert cannot leak mutations.
_REGISTRY_KEYS_MUTATED_IN_TESTS: frozenset[str] = frozenset({"alias", "__and__", "bitwise_not"})


class _RangeMaskAnd(nn.Module):
    """Boolean ``&`` of two comparisons, which exports as ``__and__``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return whether every value along the last axis lies strictly inside the open range."""
        return ((x > 0.01) & (x < 0.99)).all(dim=-1, keepdim=True)


def _export_decomposed(model: nn.Module, example: torch.Tensor) -> ExportedProgram:
    """Match ``export_coreml`` graph prep: ``torch.export`` then ``run_decompositions``.

    Examples:
        Requires a full RF-DETR model export (heavy) — not runnable standalone.
        See ``_nano_decomposed_ep`` fixture for real usage.

        >>> callable(_export_decomposed)
        True
    """
    with torch.no_grad():
        return torch.export.export(model, (example,), strict=False).run_decompositions({})


def _op_kinds(ep: ExportedProgram) -> list[str]:
    """List the op kind of every ``call_function`` node in the graph of ``ep``.

    Examples:
        >>> class _Add(nn.Module):
        ...     def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...         return x + 1
        >>> any("add" in kind for kind in _op_kinds(_export_decomposed(_Add().eval(), torch.zeros(1))))
        True
    """
    kinds = []
    for node in ep.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        try:
            kinds.append(node.target.name())
        except Exception:
            kinds.append(getattr(node.target, "__name__", str(node.target)))
    return kinds


def _assert_registry_clean(gaps: Counter[str], label: str) -> None:
    """Assert ``gaps`` is empty and matches ``_KNOWN_NANO_UNSUPPORTED_KINDS``, so the allowlist cannot go stale.

    Args:
        gaps: The unsupported op kinds ``unsupported_coreml_ops`` reported.
        label: The model name used in the failure messages.

    Examples:
        >>> _assert_registry_clean(Counter(), "Nano")
    """
    unexpected = set(gaps) - _KNOWN_NANO_UNSUPPORTED_KINDS
    assert not unexpected, f"new CoreML registry gaps for {label}: {dict(gaps)}"
    missing = _KNOWN_NANO_UNSUPPORTED_KINDS - set(gaps)
    assert not missing, f"allowlist stale (gaps cleared?): {sorted(missing)}; actual={dict(gaps)}"
    assert gaps == Counter(), f"expected registry-clean {label}, got {dict(gaps)}"


@pytest.fixture(scope="module")
def _nano_decomposed_ep() -> ExportedProgram:
    """Build + decompose ``RFDETRNano`` once per test module.

    Both Nano-specific tests below (alias-emission + registry-clean) only *read* the graph via
    ``unsupported_coreml_ops`` (cheap); the export itself (full DINOv2-backbone ``torch.export``) is seconds + GBs, so
    share one decomposed program instead of exporting Nano twice.
    """
    from rfdetr import RFDETRNano

    model = RFDETRNano(pretrain_weights=None).model.model
    model.eval()
    model.export()
    resolution = int(getattr(model, "resolution", 384))
    example = torch.randn(1, 3, resolution, resolution)
    return _export_decomposed(model, example)


def _reset_patches_for_test() -> None:
    """Clear package patch flag so the next ``ensure_*`` call re-applies handlers.

    Examples:
        >>> _reset_patches_for_test()
        >>> torch_ops._PATCHED
        False
    """
    torch_ops._PATCHED = False


@pytest.fixture
def _restore_coreml_torch_op_registry() -> Generator[None, None, None]:
    """Snapshot coremltools Torch-op registry entries and our patch flag; restore on teardown.

    Tests may pop or override registry keys to exercise ``ensure_coreml_torch_op_patches`` and the scanner. Restore in
    fixture teardown (not only on the success path) so a failing assert cannot leak mutated global registry state into
    later tests.
    """
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

    mapping = _TORCH_OPS_REGISTRY.name_to_func_mapping
    saved_handlers: dict[str, Any] = {key: mapping.get(key) for key in _REGISTRY_KEYS_MUTATED_IN_TESTS}
    saved_patched = torch_ops._PATCHED
    yield
    for key, handler in saved_handlers.items():
        if handler is None:
            mapping.pop(key, None)
        else:
            mapping[key] = handler
    torch_ops._PATCHED = saved_patched


@pytest.fixture
def registry(_restore_coreml_torch_op_registry: None) -> dict[str, Any]:
    """The coremltools Torch-op registry's name-to-handler mapping, restored after the test.

    Examples:
        Skipped: a pytest fixture that needs coremltools, so it cannot run standalone.

        >>> "alias_copy" in registry  # doctest: +SKIP
        True
    """
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

    return _TORCH_OPS_REGISTRY.name_to_func_mapping


@_requires_coremltools
@pytest.mark.usefixtures("_restore_coreml_torch_op_registry")
class TestEnsureCoremlTorchOpPatches:
    """Package-local coremltools registry patches."""

    def test_alias_maps_to_alias_copy_noop(self, registry: dict[str, Any]) -> None:
        """``alias`` must share coremltools' ``alias_copy`` identity handler, and patching twice changes nothing."""
        registry.pop("alias", None)
        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert "alias" in registry
        assert registry["alias"] is registry["alias_copy"]
        ensure_coreml_torch_op_patches()  # idempotent
        assert registry["alias"] is registry["alias_copy"]

    def test_dunder_and_maps_to_bitwise_and(self, registry: dict[str, Any]) -> None:
        """``__and__`` (bool ``&``) must share coremltools' ``bitwise_and`` handler."""
        registry.pop("__and__", None)
        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert "__and__" in registry
        assert registry["__and__"] is registry["bitwise_and"]

    def test_bitwise_not_override_accepts_float(self, registry: dict[str, Any]) -> None:
        """Our ``bitwise_not`` override must be installed (float-typed bool masks)."""
        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert registry["bitwise_not"] is torch_ops._bitwise_not_allowing_float_bool_masks

    def test_nano_still_emits_alias_but_is_registry_clean(self, _nano_decomposed_ep: ExportedProgram) -> None:
        """Nano's export graph must still contain ``aten.alias``; the registry patch must clear it."""
        alias_kinds = [kind for kind in _op_kinds(_nano_decomposed_ep) if "alias" in kind]
        assert alias_kinds, "expected Nano export graph to still emit aten.alias"
        gaps = unsupported_coreml_ops(_nano_decomposed_ep)
        assert "alias" not in gaps
        assert gaps == Counter()


@_requires_coremltools
@pytest.mark.usefixtures("_restore_coreml_torch_op_registry")
class TestUnsupportedCoremlOps:
    """Checklist against the real coremltools registry."""

    def test_bool_and_is_registry_clean_after_patches(self) -> None:
        """Bool ``&`` exports as ``__and__``; package patch must make it registry-clean."""
        ep = _export_decomposed(_RangeMaskAnd().eval(), torch.rand(1, 4, 4))
        kinds = _op_kinds(ep)
        assert any("and" in kind for kind in kinds), f"expected an and-like op in graph, got {kinds}"
        gaps = unsupported_coreml_ops(ep)
        assert "__and__" not in gaps
        assert gaps == Counter()

    def test_unpatched_dunder_and_is_detected_as_unsupported(
        self, registry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scanner must still flag ``__and__`` when the package patch is not applied."""
        ep = _export_decomposed(_RangeMaskAnd().eval(), torch.rand(1, 4, 4))
        registry.pop("__and__", None)
        # Stub the patcher itself: `_PATCHED = True` would not hold, since the identity check in
        # ensure_coreml_torch_op_patches re-patches the missing `__and__`. Patch `torch_ops` (imported lazily on every
        # call), not `op_coverage`.
        monkeypatch.setattr(torch_ops, "ensure_coreml_torch_op_patches", lambda: None)
        gaps = unsupported_coreml_ops(ep)
        assert "__and__" in gaps

    def test_nano_registry_clean_after_patches(self, _nano_decomposed_ep: ExportedProgram) -> None:
        """Nano must have no registry gaps after package-local Torch-op patches."""
        _assert_registry_clean(unsupported_coreml_ops(_nano_decomposed_ep), "Nano")

    @pytest.mark.parametrize(
        "model_cls_name", ["RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRSegNano", "RFDETRKeypointPreview"]
    )
    def test_registry_clean_after_patches(self, model_cls_name: str) -> None:
        """Each other detection size, one segmentation size, and the keypoint preview must be registry-clean.

        Previously proven only for Nano-detection (see ``test_nano_registry_clean_after_patches`` above, which reuses
        the shared Nano fixture) — a flagged op kind on an untested size/variant could otherwise false-fail a valid
        export with no evidence it is actually unclean. The ``RFDETRKeypointPreview`` case checks coremltools' stock
        registry against the keypoint graph, not the package-local ``torch_ops`` patches: that graph never emits
        ``alias``, ``__and__``, or ``bitwise_not``, so its clean result holds regardless of those patches.
        """
        import rfdetr

        model_cls = getattr(rfdetr, model_cls_name)
        detector = model_cls(pretrain_weights=None)
        model = detector.model.model
        model.eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = torch.randn(1, 3, resolution, resolution)
        _assert_registry_clean(unsupported_coreml_ops(_export_decomposed(model, example)), model_cls_name)


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


@_requires_coremltools
class TestRewriteUpsampleBilinear:
    """``rewrite_upsample_bilinear`` swaps a fixed-size ``upsample_bilinear`` for matmuls and keeps the rest."""

    def test_non_square_resize_becomes_two_matmuls(self) -> None:
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
    def test_keeps_and_warns_on_unsupported_half_pixel_mode(self, mode: bool, mocker: MockerFixture) -> None:
        """``half_pixel_centers`` equal to ``align_corners`` stays an ``upsample_bilinear`` and logs a warning.

        iOS 16 can set the two apart from each other; the interpolation matrices only match the two modes PyTorch emits.
        """
        logger = mocker.patch.object(passes, "logger")
        program = _upsample_program((1, 4, 26, 20), opset="iOS16", align_corners=mode, half_pixel_centers=mode)

        rewrite_upsample_bilinear(program)

        assert _op_types(program.functions["main"]) == ["upsample_bilinear"]
        logger.warning.assert_called_once()
        assert "half_pixel_centers" in logger.warning.call_args.args[0]

    def test_keeps_and_warns_on_a_symbolic_size(self, mocker: MockerFixture) -> None:
        """A resize with a dynamic input height stays an ``upsample_bilinear`` and logs a warning.

        The interpolation matrices are constants of fixed size, so a size that is only known at run time has none.
        """
        from coremltools.converters.mil.mil import get_new_symbol

        logger = mocker.patch.object(passes, "logger")
        program = _upsample_program((1, 4, get_new_symbol(), 20), align_corners=False)

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


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    ("size_in", "size_out"),
    [
        pytest.param((26, 20), (78, 60), id="3x"),
        pytest.param((36, 36), (72, 72), id="2x"),
        pytest.param((5, 7), (1, 1), id="to-one-cell"),
    ],
)
def test_resizes_like_torch(size_in: tuple[int, int], size_out: tuple[int, int], align_corners: bool) -> None:
    """``R_h @ x @ R_w.T`` equals ``F.interpolate(mode="bilinear")``, up by 3 or 2 and down to one cell."""
    x = torch.randn(2, 3, *size_in, generator=torch.Generator().manual_seed(0))
    height = torch.from_numpy(bilinear_interpolation_matrix(size_in[0], size_out[0], align_corners))
    width = torch.from_numpy(bilinear_interpolation_matrix(size_in[1], size_out[1], align_corners))

    expected = F.interpolate(x, size=size_out, mode="bilinear", align_corners=align_corners)
    torch.testing.assert_close(height @ x @ width.T, expected)
