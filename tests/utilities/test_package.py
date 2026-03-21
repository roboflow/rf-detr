# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for package metadata helpers."""

import subprocess
import sys
from unittest.mock import patch

import pytest

import rfdetr
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


def test_getattr_hook_resolves_removed_util_module_while_shim_exists() -> None:
    """Removed-name aliases should still resolve while shim modules exist."""
    util_module = rfdetr.__getattr__("util")
    assert util_module.__name__ == "rfdetr.util"


def test_getattr_hook_resolves_removed_deploy_module_while_shim_exists() -> None:
    """Removed-name aliases should still resolve while shim modules exist."""
    deploy_module = rfdetr.__getattr__("deploy")
    assert deploy_module.__name__ == "rfdetr.deploy"


def test_getattr_hook_raises_importerror_when_removed_shim_is_missing() -> None:
    """Missing removed shim should raise ImportError with migration hint."""
    missing_name = "rfdetr.missing_removed_shim"
    missing_exc = ModuleNotFoundError(f"No module named '{missing_name}'", name=missing_name)
    with (
        patch.dict(rfdetr._REMOVED_IN_V17, {"missing_removed_shim": "migration hint"}),
        patch("rfdetr.importlib.import_module", side_effect=missing_exc),
        pytest.raises(ImportError, match="migration hint"),
    ):
        rfdetr.__getattr__("missing_removed_shim")


def test_getattr_hook_does_not_mask_nested_module_not_found_error() -> None:
    """Nested ModuleNotFoundError from inside shim import should propagate."""
    with (
        patch.dict(rfdetr._REMOVED_IN_V17, {"missing_dep_shim": "migration hint"}),
        patch(
            "rfdetr.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'torchvision_ops'", name="torchvision_ops"),
        ),
        pytest.raises(ModuleNotFoundError, match="torchvision_ops"),
    ):
        rfdetr.__getattr__("missing_dep_shim")
