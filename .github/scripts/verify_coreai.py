# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify the Core AI runtime is importable before running its parity tests.

Purpose:
    Guard against a silently-skipped parity test: ``TestCoreAIEndToEnd`` skips
    without coreai-torch or the macOS 27 Core AI runtime, so a broken wheel or
    an older runner image would turn the job green instead of red.
Usage:
    Run ``python .github/scripts/verify_coreai.py`` after installing the
    ``[coreai]`` extra, on a runner exposing the Core AI runtime (``xcode-27``).
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: coreai``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import the Core AI runtime and assert coreai-torch is detected as available.

    Returns:
        Process exit code; 0 once the import check passes.

    Examples:
        >>> callable(main)
        True
    """
    import coreai.runtime as rt

    from rfdetr.export._coreai import _IS_COREAI_TORCH_AVAILABLE

    assert _IS_COREAI_TORCH_AVAILABLE, "coreai-torch installed but not detected as available"
    print("Core AI compute units:", rt.ComputeUnitKind.available_kinds())
    return 0


if __name__ == "__main__":
    sys.exit(main())
