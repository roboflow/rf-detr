# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify coremltools is importable before running the CoreML parity tests.

Purpose:
    Guard against a silently-skipped parity test: ``TestCoreMLEndToEnd`` is
    gated by a skipif on coremltools availability, so a broken or missing
    wheel would skip it (false green) instead of failing the job red.
Usage:
    Run ``python .github/scripts/verify_coreml.py`` after installing the
    ``[coreml]`` extra.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: coreml``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import coremltools and assert it is detected as available.

    Returns:
        Process exit code; 0 once the import check passes.

    Examples:
        >>> callable(main)
        True
    """
    from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE

    assert _IS_COREMLTOOLS_AVAILABLE, "coremltools installed but not detected as available"
    print("coremltools import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
