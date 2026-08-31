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
must have no registry gaps; ``_KNOWN_NANO_UNSUPPORTED_KINDS`` stays empty unless a gap is accepted deliberately.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator
from typing import Any

import pytest
import torch
from torch import nn
from torch.export import ExportedProgram

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._coreml.torch_ops import ensure_coreml_torch_op_patches

coreml_only = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="coremltools not installed")

# Known Nano registry gaps *after* ``ensure_coreml_torch_op_patches``. Keep empty;
# fix new kinds (or add them here deliberately) before merge.
_KNOWN_NANO_UNSUPPORTED_KINDS: frozenset[str] = frozenset()

# Registry keys that tests in this module may pop or override. Snapshotted/restored by
# ``_restore_coreml_torch_op_registry`` so a failing assert cannot leak mutations.
_REGISTRY_KEYS_MUTATED_IN_TESTS: frozenset[str] = frozenset({"alias", "__and__", "bitwise_not"})


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
        >>> from rfdetr.export._coreml import torch_ops
        >>> torch_ops._PATCHED
        False
    """
    from rfdetr.export._coreml import torch_ops

    torch_ops._PATCHED = False


@pytest.fixture
def _restore_coreml_torch_op_registry() -> Generator[None, None, None]:
    """Snapshot coremltools Torch-op registry entries and our patch flag; restore on teardown.

    Tests may pop or override registry keys to exercise ``ensure_coreml_torch_op_patches`` and the scanner. Restore in
    fixture teardown (not only on the success path) so a failing assert cannot leak mutated global registry state into
    later tests.
    """
    from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

    from rfdetr.export._coreml import torch_ops

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


@coreml_only
class TestEnsureCoremlTorchOpPatches:
    """Package-local coremltools registry patches."""

    @pytest.fixture(autouse=True)
    def _autouse_restore_registry(self, _restore_coreml_torch_op_registry: None) -> None:
        """Restore registry mutations after every test (see ``_restore_coreml_torch_op_registry``)."""

    def test_alias_maps_to_alias_copy_noop(self) -> None:
        """``alias`` must share coremltools' ``alias_copy`` identity handler."""
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

        mapping = _TORCH_OPS_REGISTRY.name_to_func_mapping
        mapping.pop("alias", None)
        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert "alias" in mapping
        assert mapping["alias"] is mapping["alias_copy"]
        ensure_coreml_torch_op_patches()  # idempotent
        assert mapping["alias"] is mapping["alias_copy"]

    def test_dunder_and_maps_to_bitwise_and(self) -> None:
        """``__and__`` (bool ``&``) must share coremltools' ``bitwise_and`` handler."""
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

        mapping = _TORCH_OPS_REGISTRY.name_to_func_mapping
        mapping.pop("__and__", None)
        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert "__and__" in mapping
        assert mapping["__and__"] is mapping["bitwise_and"]

    def test_bitwise_not_override_accepts_float(self) -> None:
        """Our ``bitwise_not`` override must be installed (float-typed bool masks)."""
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

        from rfdetr.export._coreml import torch_ops

        _reset_patches_for_test()
        ensure_coreml_torch_op_patches()
        assert (
            _TORCH_OPS_REGISTRY.name_to_func_mapping["bitwise_not"] is torch_ops._bitwise_not_allowing_float_bool_masks
        )

    def test_nano_still_emits_alias_but_is_registry_clean(self, _nano_decomposed_ep: ExportedProgram) -> None:
        """Nano's export graph must still contain ``aten.alias``; the registry patch must clear it."""
        ep = _nano_decomposed_ep
        alias_kinds = []
        for node in ep.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            try:
                kind = node.target.name()
            except Exception:
                kind = getattr(node.target, "__name__", str(node.target))
            if "alias" in kind:
                alias_kinds.append(kind)
        assert alias_kinds, "expected Nano export graph to still emit aten.alias"
        gaps = unsupported_coreml_ops(ep)
        assert "alias" not in gaps
        assert gaps == Counter()


@coreml_only
class TestUnsupportedCoremlOps:
    """Checklist against the real coremltools registry."""

    @pytest.fixture(autouse=True)
    def _autouse_restore_registry(self, _restore_coreml_torch_op_registry: None) -> None:
        """Restore registry mutations after every test (see ``_restore_coreml_torch_op_registry``)."""

    def test_bool_and_is_registry_clean_after_patches(self) -> None:
        """Bool ``&`` exports as ``__and__``; package patch must make it registry-clean."""

        class _RangeMaskAnd(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return ((x > 0.01) & (x < 0.99)).all(dim=-1, keepdim=True)

        ep = _export_decomposed(_RangeMaskAnd().eval(), torch.rand(1, 4, 4))
        kinds = []
        for node in ep.graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            try:
                kinds.append(node.target.name())
            except Exception:
                kinds.append(getattr(node.target, "__name__", str(node.target)))
        assert any("__and__" in k or "bitwise_and" in k or "and" in k for k in kinds), (
            f"expected an and-like op in graph, got {kinds}"
        )
        gaps = unsupported_coreml_ops(ep)
        assert "__and__" not in gaps
        assert gaps == Counter()

    def test_unpatched_dunder_and_is_detected_as_unsupported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scanner must still flag ``__and__`` when the package patch is not applied."""
        from coremltools.converters.mil.frontend.torch.ops import _TORCH_OPS_REGISTRY

        from rfdetr.export._coreml import torch_ops

        class _RangeMaskAnd(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return ((x > 0.01) & (x < 0.99)).all(dim=-1, keepdim=True)

        ep = _export_decomposed(_RangeMaskAnd().eval(), torch.rand(1, 4, 4))
        mapping = _TORCH_OPS_REGISTRY.name_to_func_mapping
        mapping.pop("__and__", None)
        # Bypass the real patcher entirely — faking `_PATCHED = True` no longer works: the
        # identity-validated check in ensure_coreml_torch_op_patches would see mapping["__and__"]
        # missing and correctly re-patch it regardless of the flag (that's the fix under test
        # elsewhere; here we need the *unpatched* state to actually hold). Patch must target
        # torch_ops (the source module unsupported_coreml_ops lazily imports from on every call),
        # not op_coverage — op_coverage never reads its own namespace for this name.
        monkeypatch.setattr(torch_ops, "ensure_coreml_torch_op_patches", lambda: None)
        gaps = unsupported_coreml_ops(ep)
        assert "__and__" in gaps

    def test_nano_registry_clean_after_patches(self, _nano_decomposed_ep: ExportedProgram) -> None:
        """Nano must have no registry gaps after package-local Torch-op patches."""
        gaps = unsupported_coreml_ops(_nano_decomposed_ep)
        unexpected = set(gaps) - _KNOWN_NANO_UNSUPPORTED_KINDS
        assert not unexpected, f"new CoreML registry gaps: {dict(gaps)}"
        missing = _KNOWN_NANO_UNSUPPORTED_KINDS - set(gaps)
        assert not missing, f"allowlist stale (gaps cleared?): {sorted(missing)}; actual={dict(gaps)}"
        assert gaps == Counter(), f"expected registry-clean Nano, got {dict(gaps)}"

    @pytest.mark.parametrize(
        "model_cls_name",
        [
            pytest.param("RFDETRSmall", id="small"),
            pytest.param("RFDETRMedium", id="medium"),
            pytest.param("RFDETRLarge", id="large"),
            pytest.param("RFDETRSegNano", id="seg-nano"),
        ],
    )
    def test_registry_clean_after_patches(self, model_cls_name: str) -> None:
        """Each other released size (detection + one segmentation) must have no registry gaps after patches.

        Previously proven only for Nano-detection (see ``test_nano_registry_clean_after_patches`` above, which reuses
        the shared Nano fixture) — a flagged op kind on an untested size/variant could otherwise false-fail a valid
        export with no evidence it is actually unclean.
        """
        import rfdetr

        model_cls = getattr(rfdetr, model_cls_name)
        detector = model_cls(pretrain_weights=None)
        model = detector.model.model
        model.eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = torch.randn(1, 3, resolution, resolution)
        gaps = unsupported_coreml_ops(_export_decomposed(model, example))
        unexpected = set(gaps) - _KNOWN_NANO_UNSUPPORTED_KINDS
        assert not unexpected, f"new CoreML registry gaps for {model_cls_name}: {dict(gaps)}"
        missing = _KNOWN_NANO_UNSUPPORTED_KINDS - set(gaps)
        assert not missing, f"allowlist stale (gaps cleared?): {sorted(missing)}; actual={dict(gaps)}"
        assert gaps == Counter(), f"expected registry-clean {model_cls_name}, got {dict(gaps)}"
