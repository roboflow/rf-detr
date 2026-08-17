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

This module instead generates a **fresh dataset version per run** (``Project.generate_version`` — the server assigns the
next number itself), deploys with ``version=None`` to exercise the auto-latest resolution in ``deploy_to_roboflow``, and
independently polls ``roboflow.adapters.rfapi.get_version`` for ``response["version"]["train"]["model"]`` — the exact
field the ``roboflow`` SDK itself uses to decide whether a version has a trained model (see
``roboflow.core.version.Version.__init__``) — rather than trusting ``deploy_to_roboflow``'s silent-success return. A
fresh version can never carry a trained model before the deploy, so a re-run can never go green on a previous run's
leftover model state.

Opt-in only (``-m e2e_roboflow``, see ``.github/workflows/ci-integrations.yml``): requires ``ROBOFLOW_API_KEY``,
``ROBOFLOW_TEST_WORKSPACE``, and ``ROBOFLOW_TEST_PROJECT`` pointing at a dedicated throwaway Roboflow project (with at
least one annotated image, so version generation succeeds) reserved for this test; skips cleanly when any are unset
(local runs, fork PRs). Maintenance note: every run appends one dataset version to the throwaway project.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from roboflow import Roboflow
from roboflow.adapters.rfapi import RoboflowError, get_version

from rfdetr import RFDETRNano

_API_KEY = os.getenv("ROBOFLOW_API_KEY")
_WORKSPACE = os.getenv("ROBOFLOW_TEST_WORKSPACE")
_PROJECT = os.getenv("ROBOFLOW_TEST_PROJECT")

_MISSING_ENV_REASON = (
    "requires ROBOFLOW_API_KEY, ROBOFLOW_TEST_WORKSPACE, and ROBOFLOW_TEST_PROJECT pointing at a "
    "dedicated Roboflow test project (live deploy validation for issue #1116)"
)

# Server-side version generation is asynchronous; the new version must be visible in the project's
# version list before deploy(version=None) runs, or auto-latest would resolve to the previous
# version — which already carries a trained model from the last CI run (silent false-green).
_GENERATE_POLL_INTERVAL_SECONDS = 5
_GENERATE_TIMEOUT_SECONDS = 120

# Roboflow's server-side model processing after upload is asynchronous; give it a generous
# window before treating "no trained model yet" as a genuine failure rather than in-progress.
_POLL_INTERVAL_SECONDS = 15
_POLL_TIMEOUT_SECONDS = 300


def _generate_fresh_version(project: Any) -> int:
    """Generate a new dataset version (no preprocessing/augmentation) and wait until it is listed.

    Args:
        project: A ``roboflow.core.project.Project`` for the dedicated test project.

    Returns:
        The freshly generated version number, guaranteed visible in ``get_version_information()``.

    Raises:
        pytest.fail.Exception: If the generated version does not appear within the timeout — deploying
            at that point would silently target the previous version, defeating the test.

    Examples:
        >>> _generate_fresh_version  # doctest: +SKIP
        (needs a live Roboflow project with at least one annotated image)
    """
    new_version = project.generate_version(settings={"preprocessing": {}, "augmentation": {}})
    deadline = time.monotonic() + _GENERATE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        listed = {os.path.basename(info["id"]) for info in project.get_version_information()}
        if str(new_version) in listed:
            return int(new_version)
        time.sleep(_GENERATE_POLL_INTERVAL_SECONDS)
    pytest.fail(
        f"generated version {new_version} did not appear in the project's version list within "
        f"{_GENERATE_TIMEOUT_SECONDS}s — aborting instead of deploying to a stale version"
    )


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


@pytest.mark.skipif(not all([_API_KEY, _WORKSPACE, _PROJECT]), reason=_MISSING_ENV_REASON)
@pytest.mark.e2e_roboflow
class TestDeployToRoboflowEndToEnd:
    """``deploy_to_roboflow`` must land a genuinely trained model server-side (``-m e2e_roboflow``)."""

    def test_deploy_lands_trained_model_on_fresh_version(self) -> None:
        """After deploy_to_roboflow(version=None), the freshly generated version must show a trained model.

        Generates a new dataset version first (so no previous run's model can satisfy the check), deploys with version
        omitted (exercising auto-latest resolution), then asserts on the polled server-side status of that specific
        version — the return value alone cannot be trusted (issue #1116).
        """
        rf_project = Roboflow(api_key=_API_KEY).workspace(_WORKSPACE).project(_PROJECT)
        fresh_version = _generate_fresh_version(rf_project)

        model = RFDETRNano(pretrain_weights=None)
        model.deploy_to_roboflow(
            workspace=_WORKSPACE,
            project_id=_PROJECT,
            api_key=_API_KEY,
        )

        status = _poll_until_trained(_WORKSPACE, _PROJECT, str(fresh_version), _API_KEY)

        assert status.get("version", {}).get("train", {}).get("model"), (
            f"deploy_to_roboflow() returned without error, but Roboflow version {fresh_version} has no "
            f"trained model after {_POLL_TIMEOUT_SECONDS}s — server-side upload silently failed "
            f"(issue #1116). Raw status response: {status!r}"
        )
