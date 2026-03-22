# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for package metadata helpers."""

import subprocess
import sys
from unittest.mock import patch

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
