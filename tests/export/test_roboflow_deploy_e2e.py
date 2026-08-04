# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Live end-to-end validation of ``RFDETR.deploy_to_roboflow`` (GitHub issue #1116).

``deploy_to_roboflow`` delegates the actual upload to ``roboflow.Version.deploy``, whose ``_upload_zip`` swallows the
upload ``requests.put`` failure in a bare ``try/except`` that only prints — it never raises. That means a failed server-
side upload looks identical to a successful one from ``deploy_to_roboflow``'s return value alone (see issue #1116:
notebook reports success, Roboflow UI shows "Model Upload Failed"). Every other ``deploy_to_roboflow`` test in this repo
(``tests/export/test_export_for_roboflow.py``, ``tests/inference/test_predict.py``,
``tests/training/test_detr_shim.py``) mocks ``roboflow.Roboflow`` entirely, so none of them can catch a real server-side
rejection.

This module instead deploys against a real Roboflow workspace/project and independently polls
``roboflow.adapters.rfapi.get_version`` for ``response["version"]["train"]["model"]`` — the exact field the ``roboflow``
SDK itself uses to decide whether a version has a trained model (see ``roboflow.core.version.Version.__init__``) —
rather than trusting ``deploy_to_roboflow``'s silent-success return.

Opt-in only (``-m e2e_roboflow``, see ``.github/workflows/ci-integrations.yml``): requires ``ROBOFLOW_API_KEY``,
``ROBOFLOW_TEST_WORKSPACE``, ``ROBOFLOW_TEST_PROJECT``, and ``ROBOFLOW_TEST_VERSION`` pointing at a dedicated throwaway
Roboflow project reserved for this test; skips cleanly when any are unset (local runs, fork PRs).
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from roboflow.adapters.rfapi import RoboflowError, get_version

from rfdetr import RFDETRNano

_API_KEY = os.getenv("ROBOFLOW_API_KEY")
_WORKSPACE = os.getenv("ROBOFLOW_TEST_WORKSPACE")
_PROJECT = os.getenv("ROBOFLOW_TEST_PROJECT")
_VERSION = os.getenv("ROBOFLOW_TEST_VERSION")

_MISSING_ENV_REASON = (
    "requires ROBOFLOW_API_KEY, ROBOFLOW_TEST_WORKSPACE, ROBOFLOW_TEST_PROJECT, and "
    "ROBOFLOW_TEST_VERSION pointing at a dedicated Roboflow test project (live deploy "
    "validation for issue #1116)"
)

# Roboflow's server-side model processing after upload is asynchronous; give it a generous
# window before treating "no trained model yet" as a genuine failure rather than in-progress.
_POLL_INTERVAL_SECONDS = 15
_POLL_TIMEOUT_SECONDS = 300


def _poll_until_trained(workspace: str, project: str, version: str, api_key: str) -> dict[str, Any]:
    """Poll the Roboflow version endpoint until a trained model appears or the timeout elapses.

    Args:
        workspace: Roboflow workspace slug.
        project: Roboflow project slug.
        version: Roboflow dataset version identifier.
        api_key: Roboflow API key.

    Returns:
        The last raw JSON response received from ``get_version``.

    Examples:
        >>> _poll_until_trained  # doctest: +SKIP
        <function _poll_until_trained at ...>
    """
    deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
    response: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            response = get_version(api_key, workspace, project, version, nocache=True)
        except RoboflowError:
            response = {}
        else:
            if response.get("version", {}).get("train", {}).get("model"):
                return response
        time.sleep(_POLL_INTERVAL_SECONDS)
    return response


@pytest.mark.skipif(not all([_API_KEY, _WORKSPACE, _PROJECT, _VERSION]), reason=_MISSING_ENV_REASON)
@pytest.mark.e2e_roboflow
class TestDeployToRoboflowEndToEnd:
    """``deploy_to_roboflow`` must land a genuinely trained model server-side (``-m e2e_roboflow``)."""

    def test_deploy_lands_trained_model(self) -> None:
        """After deploy_to_roboflow(), the live Roboflow version must show a trained model.

        Asserts on the polled server-side status, not on deploy_to_roboflow()'s return value — the whole point of this
        test is that the return value alone cannot be trusted (issue #1116).
        """
        model = RFDETRNano(pretrain_weights=None)

        model.deploy_to_roboflow(
            workspace=_WORKSPACE,
            project_id=_PROJECT,
            version=_VERSION,
            api_key=_API_KEY,
        )

        status = _poll_until_trained(_WORKSPACE, _PROJECT, _VERSION, _API_KEY)

        assert status.get("version", {}).get("train", {}).get("model"), (
            "deploy_to_roboflow() returned without error, but the Roboflow version has no "
            f"trained model after {_POLL_TIMEOUT_SECONDS}s — server-side upload silently failed "
            f"(issue #1116). Raw status response: {status!r}"
        )
