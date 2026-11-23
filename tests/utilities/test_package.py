# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for package metadata helpers and structural import paths."""

import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from rfdetr.models import criterion
from rfdetr.utilities import box_ops
from rfdetr.utilities.package import get_sha, is_installed


def _assert_subprocess_succeeds(command: list[str], description: str) -> None:
    """Run ``command`` and fail with its return code and output unless it exits 0.

    Args:
        command: Program and arguments to run.
        description: What the subprocess checks, named in the failure message.

    Examples:
        >>> _assert_subprocess_succeeds([sys.executable, "-c", "pass"], "a no-op interpreter")
    """
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"Subprocess for {description} failed:\n"
        f"return code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


class TestIsInstalled:
    """``is_installed`` answers from import metadata and never raises on an unusual ``sys.modules`` entry."""

    def test_stub_module_without_spec_is_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``sys.modules`` stub is not an install, and asking about one must not raise.

        ``find_spec`` raises ``ValueError`` — not ``ImportError`` — for an entry whose ``__spec__`` is ``None``, which
        is what a bare ``types.ModuleType`` left in ``sys.modules`` by a test or a library looks like. Unguarded, that
        aborts the import of whichever module probes for the optional package.
        """
        monkeypatch.setitem(sys.modules, "tensorrt", types.ModuleType("tensorrt"))

        assert is_installed("tensorrt") is False

    def test_bare_directory_is_not_installed(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A namespace package — a directory carrying the name with no module in it — is not an install.

        An export folder named after the format it writes is the realistic case: it sits on ``sys.path`` and answers
        ``find_spec`` with an ``origin``-less spec, which must not be read as the optional package being present.
        """
        (tmp_path / "tensorrt").mkdir()
        monkeypatch.delitem(sys.modules, "tensorrt", raising=False)
        monkeypatch.setattr(sys, "path", [str(tmp_path)])

        assert is_installed("tensorrt") is False


def test_get_sha_marks_dirty_worktree_when_diff_command_returns_exit_code_1(mocker) -> None:
    """A diff exit code of 1 should report uncommitted changes, not unknown."""

    def _fake_check_output(command: list[str], cwd: str | None = None) -> bytes:
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return b"abc123\n"
        if command[:4] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
            return b"feature/test\n"
        raise AssertionError(f"Unexpected command: {command!r}")

    mocker.patch("rfdetr.utilities.package.subprocess.check_output", side_effect=_fake_check_output)
    mocker.patch("rfdetr.utilities.package.subprocess.run", return_value=mocker.MagicMock(returncode=1))

    sha = get_sha()

    assert sha == "sha: abc123, status: has uncommitted changes, branch: feature/test"


def test_backbone_import_does_not_import_peft() -> None:
    """Importing backbone.backbone must NOT pull peft into sys.modules (peft is optional).

    This captures the invariant introduced in PR 1 (chore/packaging-peft-lora): after the lazy-import refactor,
    importing backbone at module-load time must not trigger a top-level ``from peft import PeftModel``.
    """
    code = (
        "import sys; "
        "import rfdetr.models.backbone.backbone; "
        "assert 'peft' not in sys.modules, "
        "'peft was eagerly imported by backbone.backbone'"
    )

    _assert_subprocess_succeeds([sys.executable, "-c", code], "backbone import")


#: Skip a test node that checks the Python 3.14 path, where TorchScript is no longer used.
_requires_python_314 = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="TorchScript remains supported before Python 3.14"
)


class TestImportPaths:
    """Structural tests for the detr.py → inference.py + variants.py split (PR 6).

    After the split:
    - ``rfdetr.inference`` exports ``ModelContext`` and ``_build_model_context``
    - ``rfdetr.variants`` exports all 15 concrete model classes
    - ``rfdetr.detr`` re-exports both for backward compatibility
    - ``rfdetr`` (top-level) continues to export public names unchanged
    """

    @pytest.mark.parametrize(
        ("module_name", "attribute"),
        [
            pytest.param("rfdetr.inference", "ModelContext", id="model-context-from-inference"),
            pytest.param("rfdetr.inference", "_build_model_context", id="build-model-context-from-inference"),
            pytest.param("rfdetr.variants", "RFDETRLarge", id="large-from-variants"),
            pytest.param("rfdetr.detr", "ModelContext", id="model-context-from-detr"),
            pytest.param("rfdetr.detr", "RFDETRLarge", id="large-from-detr"),
            pytest.param("rfdetr.detr", "_build_model_context", id="build-model-context-from-detr"),
            pytest.param("rfdetr", "RFDETRLarge", id="large-from-top-level"),
            pytest.param("rfdetr", "ModelContext", id="model-context-from-top-level"),
        ],
    )
    def test_name_importable_from_module(self, module_name: str, attribute: str) -> None:
        """Each split-out name is reachable from its new module, from rfdetr.detr (backward compat) and rfdetr."""
        obj = getattr(importlib.import_module(module_name), attribute, None)

        assert callable(obj), f"{attribute} not importable from {module_name}"

    @pytest.mark.parametrize(
        ("first_module", "second_module"),
        [
            pytest.param("rfdetr.variants", "rfdetr.detr", id="variants-first"),
            pytest.param("rfdetr.detr", "rfdetr.variants", id="detr-first"),
        ],
    )
    def test_import_order_preserves_class_identity(self, first_module: str, second_module: str) -> None:
        """Importing variants before detr, or detr before variants, shares one class object and no circular import."""
        code = (
            f"from {first_module} import RFDETRLarge; "
            f"from {second_module} import RFDETRLarge as Other; "
            "assert RFDETRLarge is Other"
        )

        _assert_subprocess_succeeds([sys.executable, "-c", code], f"{first_module}-first import")

    @pytest.mark.parametrize(
        "class_name",
        [
            "RFDETRBase",
            "RFDETRKeypointPreview",
            "RFDETRNano",
            "RFDETRSmall",
            "RFDETRMedium",
            "RFDETRLargeDeprecated",
            "RFDETRLarge",
            "RFDETRSeg",
            "RFDETRSegPreview",
            "RFDETRSegNano",
            "RFDETRSegSmall",
            "RFDETRSegMedium",
            "RFDETRSegLarge",
            "RFDETRSegXLarge",
            "RFDETRSeg2XLarge",
        ],
    )
    def test_all_variant_classes_importable_from_variants(self, class_name: str) -> None:
        """Every concrete variant class must be importable from rfdetr.variants."""
        import rfdetr.variants as variants_mod

        cls = getattr(variants_mod, class_name, None)
        assert cls is not None, f"{class_name} not found in rfdetr.variants"

    def test_top_level_import_sets_numpy_complex_alias(self) -> None:
        """Verify rfdetr shim sets np.complex_ in a fresh interpreter with complex_ absent.

        Runs in a subprocess so the shim code path is actually executed — importing rfdetr inside pytest is a no-op
        because rfdetr is already cached in sys.modules.
        """
        code = (
            "import numpy\n"
            "if hasattr(numpy, 'complex_'):\n"
            "    del numpy.complex_\n"
            "import rfdetr\n"
            "assert hasattr(numpy, 'complex_'), 'shim did not set numpy.complex_'\n"
            "assert numpy.complex_ is numpy.complex128, 'complex_ must alias complex128'\n"
        )

        _assert_subprocess_succeeds([sys.executable, "-c", code], "NumPy complex_ shim")

    @_requires_python_314
    def test_top_level_import_avoids_unsupported_torchscript(self) -> None:
        """Python 3.14+ imports must not invoke unsupported TorchScript compilation.

        TorchScript remains the compatibility contract on supported earlier Python versions, where its warning does not
        represent the reported regression.
        """
        code = "import warnings\nwarnings.filterwarnings('error', message=r'.*torch\\.jit\\.script.*')\nimport rfdetr\n"

        _assert_subprocess_succeeds([sys.executable, "-c", code], "top-level import")

    @pytest.mark.parametrize(
        ("wrap_before_torch", "wrap_before_rfdetr"),
        [
            pytest.param("warnings.warn = _wrapped\n", "warnings.warn = _orig\n", id="wrapped-at-torch-import"),
            pytest.param("", "warnings.warn = _wrapped\n", id="wrapped-at-rfdetr-import"),
        ],
    )
    def test_top_level_import_survives_wrapped_warnings_warn(
        self, wrap_before_torch: str, wrap_before_rfdetr: str
    ) -> None:
        """Importing rfdetr must not fail when the host has replaced ``warnings.warn`` around the torch import.

        Databricks notebooks wrap ``warnings.warn``. TorchScript looks builtins up by object identity, so scripting a
        loss that reaches a torch warning failed during ``import rfdetr``: with ``RuntimeError`` when torch saw the
        wrapper. When the wrapper appeared after torch was imported, the error depends on the wrapper: ``OSError``
        without retrievable source (``python -c`` before Python 3.13), ``TypeError`` for a ``functools.wraps`` wrapper,
        or a TorchScript frontend error where the source parses.
        """
        code = (
            "import warnings\n"
            "_orig = warnings.warn\n"
            "def _wrapped(*args, **kwargs):\n"
            "    return _orig(*args, **kwargs)\n"
            f"{wrap_before_torch}"
            "import torch\n"
            f"{wrap_before_rfdetr}"
            "import rfdetr\n"
            "from rfdetr.utilities import box_ops\n"
            "assert box_ops.batch_sigmoid_ce_loss_jit is box_ops.batch_sigmoid_ce_loss\n"
        )

        _assert_subprocess_succeeds([sys.executable, "-c", code], "import under wrapped warnings.warn")

    @pytest.mark.parametrize(
        "definition",
        [
            pytest.param(
                "def _wrapped(*args, **kwargs):\n    return _orig(*args, **kwargs)\n",
                id="file-defined-wrapper",
            ),
            pytest.param(
                "import functools\n"
                "@functools.wraps(_orig)\n"
                "def _wrapped(*args, **kwargs):\n"
                "    return _orig(*args, **kwargs)\n",
                id="functools-wraps-wrapper",
            ),
            pytest.param(
                "_scope = {'_orig': _orig}\n"
                "exec('def _wrapped(*args, **kwargs):\\n    return _orig(*args, **kwargs)\\n', _scope)\n"
                "_wrapped = _scope['_wrapped']\n",
                id="source-less-wrapper",
            ),
            pytest.param("_wrapped = lambda *args, **kwargs: _orig(*args, **kwargs)\n", id="lambda-wrapper"),
        ],
    )
    def test_top_level_import_survives_wrapper_shapes(self, tmp_path: Path, definition: str) -> None:
        """Importing rfdetr falls back to eager losses for every shape of ``warnings.warn`` wrapper.

        The wrapper replaces ``warnings.warn`` after torch is imported and the host runs as a script file, so the
        outcome does not depend on whether ``python -c`` source is retrievable on the interpreter. Each shape makes
        TorchScript fail with a different exception class (``OSError`` without source, ``TypeError`` for
        ``functools.wraps``, a frontend error for a file-defined variadic wrapper), and none may fail the import. The
        sigmoid cross-entropy aliases reach ``warnings.warn`` when scripted, so they must end up as the eager functions.
        """
        host = tmp_path / "host.py"
        host.write_text(
            "import warnings\n"
            "_orig = warnings.warn\n"
            f"{definition}"
            "import torch\n"
            "warnings.warn = _wrapped\n"
            "import rfdetr\n"
            "from rfdetr.models import criterion\n"
            "from rfdetr.utilities import box_ops\n"
            "assert criterion.sigmoid_ce_loss_jit is criterion.sigmoid_ce_loss\n"
            "assert box_ops.batch_sigmoid_ce_loss_jit is box_ops.batch_sigmoid_ce_loss\n",
            encoding="utf-8",
        )

        _assert_subprocess_succeeds([sys.executable, str(host)], "import under a wrapped warnings.warn")

    @pytest.mark.parametrize(
        ("alias", "eager"),
        [
            pytest.param(criterion.dice_loss_jit, criterion.dice_loss, id="dice"),
            pytest.param(criterion.sigmoid_ce_loss_jit, criterion.sigmoid_ce_loss, id="sigmoid-ce"),
            pytest.param(box_ops.batch_dice_loss_jit, box_ops.batch_dice_loss, id="batch-dice"),
            pytest.param(box_ops.batch_sigmoid_ce_loss_jit, box_ops.batch_sigmoid_ce_loss, id="batch-sigmoid-ce"),
        ],
    )
    def test_jit_loss_aliases_select_versioned_implementation(self, alias: object, eager: object) -> None:
        """Loss aliases select the documented implementation for this Python version.

        Python 3.14+ must not compile TorchScript during import; earlier supported versions preserve their historical
        scripted aliases.
        """
        if sys.version_info >= (3, 14):
            assert alias is eager
        else:
            assert isinstance(alias, torch.jit.ScriptFunction)
            assert alias is not eager

    def test_identity_across_import_paths(self) -> None:
        """The same class object must be returned regardless of import path.

        This ensures re-exports are true re-exports (not copies) so that isinstance() checks work across all import
        paths.
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

    def test_getattr_raises_for_unknown_names(self) -> None:
        """Accessing an unknown name via rfdetr.detr must raise AttributeError."""
        import rfdetr.detr as detr_mod

        with pytest.raises(AttributeError) as exc_info:
            _ = detr_mod._this_name_does_not_exist_12345

        assert "has no attribute" in str(exc_info.value)

    def test_dir_includes_lazy_exports(self) -> None:
        """dir(rfdetr.detr) must include all lazy re-export names."""
        import rfdetr.detr as detr_mod

        names = dir(detr_mod)
        assert "ModelContext" in names, "ModelContext missing from dir(rfdetr.detr)"
        assert "RFDETRLarge" in names, "RFDETRLarge missing from dir(rfdetr.detr)"
        assert "RFDETRBase" in names, "RFDETRBase missing from dir(rfdetr.detr)"
