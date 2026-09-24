# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify ExecuTorch and coremltools are both importable before the delegate tests.

Purpose:
    Guard against a silently-skipped test: the ExecuTorch-CoreML delegate class
    is gated on executorch, coremltools, and macOS, so a missing wheel would
    skip it (false green) instead of failing the job red.
Usage:
    Run ``python .github/scripts/verify_executorch_coreml.py`` after installing
    the combined ``[executorch,coreml]`` extras.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    the ``executorch`` + ``coreml`` combination row, ``verify: executorch_coreml``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import the ExecuTorch runtime and coremltools, failing loudly if either is missing.

    Returns:
        Process exit code; 0 once both import checks pass.

    Examples:
        >>> callable(main)
        True
    """
    from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
    from rfdetr.export._executorch.exporter import _check_executorch_available

    _check_executorch_available(require_runtime=True)
    assert _IS_EXECUTORCH_AVAILABLE, "executorch installed but not detected as available"
    import coremltools  # noqa: F401

    print("executorch runtime + coremltools import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
