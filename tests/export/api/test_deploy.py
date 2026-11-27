# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for ``RFDETR.export_for_roboflow``.

``export_for_roboflow`` is the extracted, network-free core of ``deploy_to_roboflow``: it writes ``weights.pt`` (a dict
with ``"model"`` and ``"args"`` keys) and ``class_names.txt`` into a target directory.  A lightweight stub stands in for
``self.model`` so the file-writing contract is exercised without building a real model or downloading weights.

Live end-to-end validation of ``RFDETR.deploy_to_roboflow`` (GitHub issue #1116).

``deploy_to_roboflow`` delegates the actual upload to ``roboflow.Version.deploy``, whose ``_upload_zip`` swallows the
upload ``requests.put`` failure in a bare ``try/except`` that only prints — it never raises. That means a failed server-
side upload looks identical to a successful one from ``deploy_to_roboflow``'s return value alone (see issue #1116:
notebook reports success, Roboflow UI shows "Model Upload Failed"). Every other ``deploy_to_roboflow`` test in this repo
(``tests/export/api/test_for_roboflow.py``, ``tests/inference/prediction/test_core.py``,
``tests/training/shim/test_detr_shim.py``) mocks ``roboflow.Roboflow`` entirely, so none of them can catch a real
server-side rejection.

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

Public export, load, and predict round trips on real runtime artifacts.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from roboflow import Roboflow
from roboflow.adapters.rfapi import RoboflowError, get_version
from supervision import KeyPoints

from rfdetr import RFDETRNano
from rfdetr.detr import RFDETR
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export._tflite import _IS_ONNX2TF_AVAILABLE
from rfdetr.export.imports import (
    _IS_AI_EDGE_LITERT_INSTALLED,
    _IS_LITERT_TORCH_INSTALLED,
    _IS_TENSORFLOW_INSTALLED,
    _IS_TFLITE_RUNTIME_INSTALLED,
)
from tests._markers import _requires_coreai_runtime, _requires_onnx_and_onnxruntime, _requires_openvino
from tests.export.conftest import _prediction_image, _prediction_model_for_task, assert_prediction_roundtrip


def _make_stub_model(class_names: list[str]) -> RFDETR:
    """Build an RFDETR instance whose model/state are stubbed for export_for_roboflow.

    ``RFDETR.__init__`` is bypassed; only the attributes ``export_for_roboflow`` reads are populated.

    Examples:
        >>> m = _make_stub_model(["cat", "dog"])
        >>> m.model.args.resolution
        560
        >>> m.model.class_names
        ['cat', 'dog']
    """
    instance = RFDETR.__new__(RFDETR)
    args = SimpleNamespace(resolution=560)
    inner_module = SimpleNamespace(state_dict=lambda: {"weight": torch.zeros(2, 2)})
    instance.model = SimpleNamespace(model=inner_module, args=args, class_names=class_names)
    return instance


def _load_bundle(directory: Path) -> dict:
    """Load the ``weights.pt`` bundle written by ``export_for_roboflow`` from ``directory``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     torch.save({"model": {}, "args": None}, Path(directory) / "weights.pt")
        ...     sorted(_load_bundle(Path(directory)))
        ['args', 'model']
    """
    return torch.load(directory / "weights.pt", map_location="cpu", weights_only=False)


class TestExportForRoboflow:
    """export_for_roboflow writes a deploy-ready bundle into a directory."""

    def test_writes_weights_pt_with_model_and_args(self, tmp_path: Path) -> None:
        """weights.pt is a dict with 'model' and 'args', and args carries resolution."""
        model = _make_stub_model(["cat", "dog"])

        model.export_for_roboflow(str(tmp_path))

        bundle = _load_bundle(tmp_path)
        assert set(bundle) == {"model", "args"}
        assert "weight" in bundle["model"]
        assert bundle["args"].resolution == 560

    @pytest.mark.parametrize(
        ("class_names", "expected_text"),
        [
            (["cat", "dog"], "cat\ndog"),
            ([], ""),
        ],
    )
    def test_writes_class_names_txt(self, tmp_path: Path, class_names: list[str], expected_text: str) -> None:
        """class_names.txt lists one class name per line, with no trailing newline, and is empty for no classes."""
        model = _make_stub_model(class_names)

        model.export_for_roboflow(str(tmp_path))

        assert (tmp_path / "class_names.txt").read_text(encoding="utf-8") == expected_text

    def test_embeds_class_names_in_args(self, tmp_path: Path) -> None:
        """class_names are embedded in the saved args namespace when absent."""
        model = _make_stub_model(["cat", "dog"])

        model.export_for_roboflow(str(tmp_path))

        assert _load_bundle(tmp_path)["args"].class_names == ["cat", "dog"]

    def test_does_not_overwrite_existing_args_class_names(self, tmp_path: Path) -> None:
        """args.class_names already set on the model is preserved in the saved bundle."""
        model = _make_stub_model(["cat", "dog"])
        model.model.args.class_names = ["pre_existing"]

        model.export_for_roboflow(str(tmp_path))

        assert _load_bundle(tmp_path)["args"].class_names == ["pre_existing"]

    def test_sanitizes_training_paths_without_mutating_runtime_args(self, tmp_path: Path) -> None:
        """Exported args hide local training paths while runtime args retain them."""
        model = _make_stub_model(["cat", "dog"])
        model.model.args.dataset_dir = "/private/training-dataset"
        model.model.args.output_dir = "/private/training-output"
        model.model.args.resume = "/private/checkpoints/last.ckpt"

        model.export_for_roboflow(str(tmp_path))

        bundle = _load_bundle(tmp_path)
        assert bundle["args"].dataset_dir is None
        assert bundle["args"].output_dir == "output"
        assert bundle["args"].resume == ""
        assert model.model.args.dataset_dir == "/private/training-dataset"
        assert model.model.args.output_dir == "/private/training-output"
        assert model.model.args.resume == "/private/checkpoints/last.ckpt"

    def test_creates_output_dir_when_missing(self, tmp_path: Path) -> None:
        """output_dir is created if it does not already exist."""
        model = _make_stub_model(["cat", "dog"])
        target = tmp_path / "nested" / "bundle"

        model.export_for_roboflow(str(target))

        assert (target / "weights.pt").exists()
        assert (target / "class_names.txt").exists()


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


#: Skip a test node that deploys to Roboflow when its credentials are not set in the environment.
_requires_roboflow_credentials = pytest.mark.skipif(
    not all([_API_KEY, _WORKSPACE, _PROJECT]), reason=_MISSING_ENV_REASON
)


@_requires_roboflow_credentials
@pytest.mark.integration
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


#: Skip an onnx2tf round trip unless some TFLite interpreter can run the converted model.
_requires_tflite_runtime = pytest.mark.skipif(
    not _IS_ONNX2TF_AVAILABLE
    or not (_IS_AI_EDGE_LITERT_INSTALLED or _IS_TFLITE_RUNTIME_INSTALLED or _IS_TENSORFLOW_INSTALLED),
    reason="TFLite round trips need onnx2tf and one of ai_edge_litert, tflite_runtime, or tensorflow",
)


#: Skip a direct LiteRT round trip unless both its converter and its interpreter are installed.
_requires_litert_runtime = pytest.mark.skipif(
    not (_IS_LITERT_TORCH_INSTALLED and _IS_AI_EDGE_LITERT_INSTALLED),
    reason="LiteRT round trips need litert_torch and ai_edge_litert",
)


#: Skip an ExecuTorch round trip when the ExecuTorch package cannot be imported.
_requires_executorch = pytest.mark.skipif(not _IS_EXECUTORCH_AVAILABLE, reason="executorch not installed")


#: Skip a CoreML round trip off macOS, where the CoreML runtime cannot execute a model.
_requires_coreml_runtime = pytest.mark.skipif(
    not _IS_COREMLTOOLS_AVAILABLE or sys.platform != "darwin",
    reason="CoreML round trips need coremltools and the macOS CoreML runtime",
)


class TestPredictionFixture:
    """Keep native score variation larger than the export comparison tolerances."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_score_spread(self, task: str) -> None:
        """Every task spans more than three times the largest backend score tolerance."""
        model = _prediction_model_for_task(task)
        size = 64 if task == "detect" else 96
        original = model.predict(_prediction_image(size), threshold=0.0, include_source_image=False)
        confidence = original.detection_confidence if isinstance(original, KeyPoints) else original.confidence
        assert confidence is not None
        assert np.ptp(confidence) > 3 * 3e-2


@pytest.mark.integration
class TestExportRoundTrip:
    """Compare public predictions before and after a real export."""

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_onnx
    @_requires_onnx_and_onnxruntime
    def test_onnx(self, task: str, tmp_path: Path) -> None:
        """ONNX preserves prediction coordinates, scores, labels, and task data."""
        assert_prediction_roundtrip("onnx", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_openvino
    @_requires_openvino
    def test_openvino(self, task: str, tmp_path: Path) -> None:
        """OpenVINO preserves all three public prediction tasks."""
        assert_prediction_roundtrip("openvino", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_tflite
    @pytest.mark.timeout(300)
    @_requires_tflite_runtime
    def test_tflite(self, task: str, tmp_path: Path) -> None:
        """ONNX-converted TFLite preserves all three public prediction tasks."""
        try:
            assert_prediction_roundtrip("tflite", task, tmp_path)
        except pytest.fail.Exception as exc:
            # Expected failure applies only to the experimental converter exceeding its budget.
            if not re.fullmatch(r"Timeout \(>[0-9.]+s\) from pytest-timeout\.", str(exc)):
                raise
            pytest.xfail("Experimental onnx2tf conversion exceeded the per-test time budget.")

    @pytest.mark.parametrize("task", ["detect", "segment"])
    @pytest.mark.e2e_litert
    @_requires_litert_runtime
    def test_litert(self, task: str, tmp_path: Path) -> None:
        """Direct LiteRT export preserves detection and segmentation."""
        assert_prediction_roundtrip("litert", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_executorch
    @_requires_executorch
    def test_executorch(self, task: str, tmp_path: Path) -> None:
        """Portable XNNPACK exports preserve all three prediction tasks."""
        assert_prediction_roundtrip("executorch", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreml
    @_requires_coreml_runtime
    def test_coreml(self, task: str, tmp_path: Path) -> None:
        """CoreML float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreml", task, tmp_path)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    @pytest.mark.e2e_coreai
    @_requires_coreai_runtime
    def test_coreai(self, task: str, tmp_path: Path) -> None:
        """Core AI float32 CPU execution preserves all three prediction tasks."""
        assert_prediction_roundtrip("coreai", task, tmp_path)

    @pytest.mark.e2e_litert
    @_requires_litert_runtime
    def test_litert_keypoints_refused(self, tmp_path: Path) -> None:
        """LiteRT reports its known keypoint lowering limit before conversion."""
        model = _prediction_model_for_task("keypoints")
        with pytest.raises(NotImplementedError, match="keypoint"):
            model.export(format="litert", output_dir=str(tmp_path), verbose=False)
