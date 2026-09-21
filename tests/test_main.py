# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exercise the ``python -m rfdetr`` entrypoint shim in ``rfdetr.__main__``."""

import runpy
from unittest.mock import patch

#: Import path patched to isolate the CLI entrypoint during module execution.
RFDETR_CLI_MAIN = "rfdetr.cli.main"


class TestMainEntrypoint:
    """The ``__main__`` guard dispatches to ``rfdetr.cli.main`` only when run as a script."""

    def test_dispatches_to_cli_main_when_run_as_script(self) -> None:
        """Running the module with ``__name__ == "__main__"`` invokes ``rfdetr.cli.main``.

        This reproduces what ``python -m rfdetr`` does: runpy executes ``rfdetr/__main__.py`` with
        ``run_name="__main__"``, the same value the interpreter sets for a ``-m`` invocation. ``rfdetr.cli.main`` is
        mocked so the (heavyweight, optional-dependency-gated) training CLI it delegates to is never actually
        constructed.
        """
        with patch(RFDETR_CLI_MAIN) as mock_main:
            runpy.run_module("rfdetr", run_name="__main__")

        mock_main.assert_called_once_with()

    def test_skips_cli_main_when_imported_as_a_module(self) -> None:
        """Executing the module under a non-``__main__`` name must not invoke ``rfdetr.cli.main``.

        This is the state ``rfdetr/__main__.py`` runs in for an ordinary import (``__name__`` set to its dotted module
        path rather than ``"__main__"``), which must leave the guard's body unrun.
        """
        with patch(RFDETR_CLI_MAIN) as mock_main:
            runpy.run_module("rfdetr", run_name="rfdetr.__main__")

        mock_main.assert_not_called()
