# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for package metadata helpers and structural import paths."""

import subprocess
import sys
from unittest.mock import patch

import pytest

from rfdetr.utilities.package import get_sha


def test_get_sha_marks_dirty_worktree_when_diff_command_returns_exit_code_1() -> None:
    """A diff exit code of 1 should report uncommitted changes, not unknown."""

    def _fake_check_output(command: list[str], cwd: str | None = None) -> bytes:
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return b"abc123\n"
        if command[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return b"feature/test\n"
        raise AssertionError(f"Unexpected command: {command!r}")

    class _RunResult:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    with (
        patch("rfdetr.utilities.package.subprocess.check_output", side_effect=_fake_check_output),
        patch("rfdetr.utilities.package.subprocess.run", return_value=_RunResult(returncode=1)),
    ):
        sha = get_sha()

    assert sha == "sha: abc123, status: has uncommitted changes, branch: feature/test"


def test_peft_not_imported_eagerly_on_backbone_import_characterization() -> None:
    """Importing backbone.backbone must NOT pull peft into sys.modules (peft is optional).

    This characterization test captures the invariant introduced in PR 1 (chore/packaging-peft-lora):
    after the lazy-import refactor, importing backbone at module-load time must not trigger a
    top-level ``from peft import PeftModel``.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import rfdetr.models.backbone.backbone; "
                "assert 'peft' not in sys.modules, "
                "'peft was eagerly imported by backbone.backbone'"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Subprocess for backbone import failed:\n"
        f"return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


class TestImportPaths:
    """Structural tests for the detr.py → inference.py + variants.py split (PR 6).

    After the split:
    - ``rfdetr.inference`` exports ``ModelContext`` and ``_build_model_context``
    - ``rfdetr.variants`` exports all 14 concrete model classes
    - ``rfdetr.detr`` re-exports both for backward compatibility
    - ``rfdetr`` (top-level) continues to export public names unchanged
    """

    def test_model_context_importable_from_inference(self) -> None:
        """ModelContext must be importable from the new rfdetr.inference module."""
        from rfdetr.inference import ModelContext

        assert ModelContext is not None

    def test_build_model_context_importable_from_inference(self) -> None:
        """_build_model_context must be importable from rfdetr.inference."""
        from rfdetr.inference import _build_model_context

        assert callable(_build_model_context)

    def test_rfdetr_large_importable_from_variants(self) -> None:
        """RFDETRLarge must be importable from the new rfdetr.variants module."""
        from rfdetr.variants import RFDETRLarge

        assert RFDETRLarge is not None

    def test_variants_import_first_does_not_trigger_circular_import(self) -> None:
        """Importing variants before detr must still preserve shared class identity."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from rfdetr.variants import RFDETRLarge; "
                    "from rfdetr.detr import RFDETRLarge as FromDetr; "
                    "assert RFDETRLarge is FromDetr"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "Subprocess for variants-first import failed:\n"
            f"return code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    @pytest.mark.parametrize(
        "class_name",
        [
            pytest.param("RFDETRBase", id="base"),
            pytest.param("RFDETRNano", id="nano"),
            pytest.param("RFDETRSmall", id="small"),
            pytest.param("RFDETRMedium", id="medium"),
            pytest.param("RFDETRLargeDeprecated", id="large-deprecated"),
            pytest.param("RFDETRLarge", id="large"),
            pytest.param("RFDETRSeg", id="seg-base"),
            pytest.param("RFDETRSegPreview", id="seg-preview"),
            pytest.param("RFDETRSegNano", id="seg-nano"),
            pytest.param("RFDETRSegSmall", id="seg-small"),
            pytest.param("RFDETRSegMedium", id="seg-medium"),
            pytest.param("RFDETRSegLarge", id="seg-large"),
            pytest.param("RFDETRSegXLarge", id="seg-xlarge"),
            pytest.param("RFDETRSeg2XLarge", id="seg-2xlarge"),
        ],
    )
    def test_all_variant_classes_importable_from_variants(self, class_name: str) -> None:
        """Every concrete variant class must be importable from rfdetr.variants."""
        import rfdetr.variants as variants_mod

        cls = getattr(variants_mod, class_name, None)
        assert cls is not None, f"{class_name} not found in rfdetr.variants"

    def test_model_context_backward_compat_from_detr(self) -> None:
        """ModelContext must remain importable from rfdetr.detr (backward compat)."""
        from rfdetr.detr import ModelContext

        assert ModelContext is not None

    def test_rfdetr_large_backward_compat_from_detr(self) -> None:
        """RFDETRLarge must remain importable from rfdetr.detr (backward compat)."""
        from rfdetr.detr import RFDETRLarge

        assert RFDETRLarge is not None

    def test_rfdetr_large_importable_from_top_level(self) -> None:
        """RFDETRLarge must remain importable from rfdetr (top-level package)."""
        from rfdetr import RFDETRLarge

        assert RFDETRLarge is not None

    def test_model_context_importable_from_top_level(self) -> None:
        """ModelContext must remain importable from rfdetr (top-level package)."""
        from rfdetr import ModelContext

        assert ModelContext is not None

    def test_identity_across_import_paths(self) -> None:
        """The same class object must be returned regardless of import path.

        This ensures re-exports are true re-exports (not copies) so that
        isinstance() checks work across all import paths.
        """
        import rfdetr
        from rfdetr.detr import ModelContext as FromDetr
        from rfdetr.detr import RFDETRLarge as LargeFromDetr
        from rfdetr.inference import ModelContext as FromInference
        from rfdetr.variants import RFDETRLarge as LargeFromVariants

        assert FromDetr is FromInference, (
            "rfdetr.detr.ModelContext and rfdetr.inference.ModelContext must be the same object"
        )
        assert LargeFromDetr is LargeFromVariants, (
            "rfdetr.detr.RFDETRLarge and rfdetr.variants.RFDETRLarge must be the same object"
        )
        assert rfdetr.RFDETRLarge is LargeFromVariants, (
            "top-level rfdetr.RFDETRLarge and rfdetr.variants.RFDETRLarge must be the same object"
        )
        assert rfdetr.ModelContext is FromInference, (
            "top-level rfdetr.ModelContext and rfdetr.inference.ModelContext must be the same object"
        )

    def test_build_model_context_backward_compat_from_detr(self) -> None:
        """_build_model_context must remain importable from rfdetr.detr (backward compat)."""
        from rfdetr.detr import _build_model_context

        assert callable(_build_model_context)

    def test_getattr_raises_for_unknown_names(self) -> None:
        """Accessing an unknown name via rfdetr.detr must raise AttributeError."""
        import rfdetr.detr as detr_mod

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = detr_mod._this_name_does_not_exist_12345

    def test_dir_includes_lazy_exports(self) -> None:
        """dir(rfdetr.detr) must include all lazy re-export names."""
        import rfdetr.detr as detr_mod

        names = dir(detr_mod)
        assert "ModelContext" in names, "ModelContext missing from dir(rfdetr.detr)"
        assert "RFDETRLarge" in names, "RFDETRLarge missing from dir(rfdetr.detr)"
        assert "RFDETRBase" in names, "RFDETRBase missing from dir(rfdetr.detr)"

    def test_detr_first_import_order_preserves_identity(self) -> None:
        """Importing detr before variants must preserve object identity for variant classes."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from rfdetr.detr import RFDETRLarge; "
                    "from rfdetr.variants import RFDETRLarge as FromVariants; "
                    "assert RFDETRLarge is FromVariants"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            "Subprocess for detr-first import failed:\n"
            f"return code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
