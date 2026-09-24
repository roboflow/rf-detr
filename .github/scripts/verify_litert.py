# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify litert-torch is importable before running the LiteRT parity tests.

Purpose:
    Guard against a silently-skipped parity test: ``TestLiteRTEndToEnd`` is
    gated by ``pytest.importorskip("litert_torch")``, so a broken or missing
    wheel would skip it (false green) instead of failing the job red.
Usage:
    Run ``python .github/scripts/verify_litert.py`` after installing the
    ``[litert]`` extra.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: litert``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import litert-torch and assert it is usable, failing loudly if not.

    Calls the checker itself rather than asserting a package-level flag: the
    flag folds any ``ImportError`` from litert-torch's own import tree into a
    bare ``False``, while the checker re-raises it with the chained cause, so
    the log shows which transitive package broke instead of only "not
    detected".

    Returns:
        Process exit code; 0 once the import check passes.

    Examples:
        >>> callable(main)
        True
    """
    from rfdetr.export._litert.exporter import _check_litert_available

    _check_litert_available()
    print("litert-torch import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
