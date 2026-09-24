# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify the ExecuTorch runtime is importable before running its parity tests.

Purpose:
    Guard against a silently-skipped parity test: the class exercising ExecuTorch
    export is gated behind a skipif on runtime availability, so a broken or missing
    wheel would skip it (false green) instead of failing the job red.
Usage:
    Run ``python .github/scripts/verify_executorch.py`` after installing the
    ``[executorch]`` extra.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: executorch``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import the ExecuTorch runtime and assert it is usable, failing loudly if not.

    ``_check_executorch_available(require_runtime=True)`` raises a chained,
    actionable ABI-compatibility error (rather than a bare "undefined symbol")
    if the torch/executorch pin ever drifts stale.

    Returns:
        Process exit code; 0 once the import check passes.

    Examples:
        >>> callable(main)
        True
    """
    from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
    from rfdetr.export._executorch.exporter import _check_executorch_available

    _check_executorch_available(require_runtime=True)
    assert _IS_EXECUTORCH_AVAILABLE, "executorch installed but not detected as available"
    print("executorch runtime import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
